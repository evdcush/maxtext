#!/usr/bin/python3

"""
 Copyright 2023 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

'''This script is used to measure the performance of different sharding schemes on TPU.'''

import jax
from jax.sharding import PartitionSpec
from jax.experimental.pjit import pjit
from jax.sharding import Mesh
from jax.experimental import mesh_utils
from jax.experimental.compilation_cache import compilation_cache as cc
from jax.experimental.pjit import with_sharding_constraint
from jax.experimental.topologies import get_topology_desc
from jax.experimental.serialize_executable import serialize, deserialize_and_load

import argparse
import datetime
import numpy as np
import os
import pickle

cc.initialize_cache(os.path.expanduser("~/jax_cache_2"))

parser = argparse.ArgumentParser(
  description="Experiment different sharding techniques with a simple NN.\
  Ensure 1) The product of dcn dimensions == number of slices \
  2) product of ici dimension = number of devices per slice"
  )
parser.add_argument(
    "--profiler_path", "-p",
    required=False,
    default="",
    help="Path to the profiler where the script will write to.",
    type=str
)
parser.add_argument(
    "--embedding_dimension", "-d",
    required=False,
    default=2048,
    type=int
)
parser.add_argument(
    "--batch_size", "-b",
    required=False,
    default=131072,
    type=int
)
parser.add_argument(
    "--num_layers", "-n",
    required=False,
    default=4,
    type=int
)
parser.add_argument(
    "--dcn_data_parallelism", "-dd",
    help="N-way Data Parallelism across slices",
    required=False,
    default=1,
    type=int
)
parser.add_argument(
    "--dcn_fsdp_parallelism", "-df",
    help="Fsdp parallelism across slices that is expected to be 1 in most cases",
    required=False,
    default=1,
    type=int
)
parser.add_argument(
    "--dcn_tensor_parallelism", "-dt",
    help="Tensor parallelism across slices that is expected to be 1 in most cases",
    required=False,
    default=1,
    type=int
)
parser.add_argument(
    "--ici_data_parallelism", "-id",
    help="Data parallelism within each slice that is expected to be 1 in most cases",
    required=False,
    default=1,
    type=int
)
parser.add_argument(
    "--ici_fsdp_parallelism", "-if",
    help="Number of shards for Fsdp Parallelism within each slice.",
    required=False,
    default=4,
    type=int
)
parser.add_argument(
    "--ici_tensor_parallelism", "-it",
    help="Number of shards for Tensor Parallelism within each slice.",
    required=False,
    default=1,
    type=int
)
args = parser.parse_args()

def activate_profiler(profiler_path):
  if profiler_path:
    jax.profiler.start_trace(profiler_path)

def deactivate_profiler(profiler_path):
  if profiler_path:
    jax.profiler.stop_trace()

def simple_timeit(f, tries = 5, verbose = True):
  '''Simple utility to time a function for multiple runs'''
  outcomes = []
  f() #warm it up!
  for _ in range(tries):
    s = datetime.datetime.now()
    f()
    e = datetime.datetime.now()
    outcomes.append((e-s).total_seconds())
  average_time = sum(outcomes)/len(outcomes)
  if verbose:
    print(f"average time: {average_time}, timings (seconds) {outcomes}")
  return average_time

dcn_parallelism = [args.dcn_data_parallelism, args.dcn_fsdp_parallelism, args.dcn_tensor_parallelism]
ici_parallelism = [args.ici_data_parallelism, args.ici_fsdp_parallelism, args.ici_tensor_parallelism]

topo='v4-8'
if topo=='v4-8':
  topology_devices = get_topology_desc(
      platform='tpu',
      topology_name=f'v4:2x2x1',
      chip_config_name='megacore',
      chips_per_host_bounds=(2, 2, 1),
      num_slices=1,
  ).devices
elif topo=='v4-16':
  topology_devices = get_topology_desc(
  platform='tpu',
  topology_name=f'v4:2x2x2',
  chip_config_name='megacore',
  chips_per_host_bounds=(2, 2, 2),
  num_slices=1,
).devices
devices = topology_devices #  topology_devices or jax.devices()
num_devices = len(devices)
print(f"Devices: {devices} (num_devices: {num_devices})")
assert len(devices) > 1, "You must have at least two devices"

# Assert that we have correct inputs of sharding that fit the number of chips
assert np.product(dcn_parallelism) * np.product(ici_parallelism) == num_devices, f"Number of devices {num_devices} \
      does not match the product of the parallelism {np.product(dcn_parallelism) * np.product(ici_parallelism)}"

multi_slice_env = hasattr(devices[0], 'slice_index')
# Create device mesh

if multi_slice_env:
  assert args.dcn_data_parallelism == 1 + max(x.slice_index for x in jax.devices()), \
   f"Number of slices given {args.dcn_data_parallelism} \
        does not match the number fetched from jax devices {jax.devices()[0]}"
  devices_array = mesh_utils.create_hybrid_device_mesh(ici_parallelism, dcn_parallelism, devices=devices)
else:
  devices_array = mesh_utils.create_device_mesh(ici_parallelism, devices=devices)

print(f"Decided on mesh shape: {devices_array}")

mesh = Mesh(devices_array, ["data", "fsdp", "tensor"])

data_sharding = PartitionSpec(("data", "fsdp"),  "tensor")
# We assume parameters are stored in a decreasing order of dimension size
parameter_sharding = PartitionSpec("tensor", "fsdp")

BATCH = len(devices) * args.batch_size
D_EMB = args.embedding_dimension
D_FF =  4 * D_EMB
NUM_LAYERS = args.num_layers

parameters = 2 * D_FF * D_EMB * NUM_LAYERS
parameter_bytes = 2 * parameters
activation_bytes = 2 * (  BATCH  * ( D_FF+D_EMB) ) * NUM_LAYERS
memory_bytes = parameter_bytes + activation_bytes

print(f"total {memory_bytes/10**9} GB, parameters {parameter_bytes/10**9} GB, activations {activation_bytes/10**9} GB")

def gen_layer(random_key):
  keys = jax.random.split(random_key, num = 4)
  return {
    "EMB2FF" : 1e-4 * jax.random.normal( keys[0], (D_FF, D_EMB), dtype=jax.numpy.bfloat16),
    "FF2EMB" : 1e-4 * jax.random.normal( keys[1], (D_FF, D_EMB), dtype=jax.numpy.bfloat16),
  }

def gen_layers(random_key):
  layers = []
  for _ in range(NUM_LAYERS):
    random_key, sub_key = jax.random.split(random_key)
    layers.append(gen_layer(sub_key))
  return tuple(layers)

def gen_data(random_key):
  return jax.random.uniform(random_key, (BATCH, D_EMB), dtype=jax.numpy.bfloat16 )


def multiply_layer(in_act, in_layer):
  with jax.named_scope("M1"):
    M1 = jax.nn.sigmoid(in_act @ in_layer["EMB2FF"].T)
    M1 = with_sharding_constraint(M1, data_sharding)
  with jax.named_scope("M2"):
    M2 = jax.nn.sigmoid(M1 @ in_layer["FF2EMB"])
    M2 = with_sharding_constraint(M2, data_sharding)

  return M2

def multiply_layers(in_act, in_layers):
  x = in_act

  for i, layer in enumerate(in_layers):
    with jax.named_scope(f"layer_{i}"):
      x = with_sharding_constraint(multiply_layer(x, layer), data_sharding)

  return x, in_layers

def multiply_layers_with_loss(in_act, in_layers):
  x, _ =  multiply_layers(in_act, in_layers)
  return jax.numpy.sum(x)

multiply_layers_and_grad = jax.value_and_grad(multiply_layers_with_loss, argnums=[1])

def training_step(in_act, in_layers):
  _, grad_layers = multiply_layers_and_grad(in_act, in_layers)
  out_layers = jax.tree_map(lambda param, grad: param - 1e-4 * grad, in_layers, grad_layers[0])
  return out_layers

def xaot_save(func, mesh, pickle_filename, *example_args, in_shardings=None, out_shardings=None, verbose=True):
  # TODO(mattdavidow) : support static_argnums and donate_argnums
  def xprint(string, verbose):
    if verbose:
      print(string)

  with mesh:
    xprint("Jitting func...", verbose)
    pjit_func = pjit(
      func,
      in_shardings=in_shardings,
      out_shardings=out_shardings
    )
    xprint("Jitted func!!!", verbose)
    xprint("Lowering jitted func...", verbose)
    lowered = pjit_func.lower(*example_args)
    xprint("Lowered jitted func!!!", verbose)

  compiled = lowered.compile()
  serialized, in_tree, out_tree = serialize(compiled)

  print(f"{type(serialized)=}")
  # save the serialized via pickle
  xprint("Saving the serialized compiled train step...", verbose)
  with open(pickle_filename, "wb") as f:
      pickle.dump(serialized, f)
  xprint("Saved the serialized compiled!!!", verbose)
  return pjit_func, serialized, in_tree, out_tree, compiled

save_xaot = True
use_mesh = Mesh(mesh.devices, mesh.axis_names)
key = jax.random.PRNGKey(0)
fake_key = jax.core.ShapedArray(key.shape, key.dtype)
print(f"{key=}")
if save_xaot:
  print("saving gen_data...")
  pjit_gen_data, _, _, _, _ = xaot_save(
    gen_data,
    use_mesh,
    'data_sharding.pkl',
    fake_key,
    in_shardings=None,
    out_shardings=data_sharding,
    verbose=False
  )
  print("gen_data saved!")

  print("saving gen_layers...")
  pjit_gen_layers, _, _, _, _ = xaot_save(
    gen_layers,
    use_mesh,
    'layers_sharding.pkl',
    fake_key,
    in_shardings=None,
    out_shardings=parameter_sharding,
    verbose=False
  )
  print("gen_layers saved!")

  # Sadness we need to create example training_step args using the previous jits
  with use_mesh:
    presharded_X = jax.block_until_ready(pjit_gen_data(key))
    presharded_layers = jax.block_until_ready(pjit_gen_layers(key))

  pjit_func, _, _, _, _ = xaot_save(
    training_step,
    use_mesh,
    'layers_sharding.pkl',
    presharded_X,
    presharded_layers,
    in_shardings=(data_sharding, parameter_sharding),
    out_shardings=parameter_sharding
  )


with use_mesh:
  presharded_X = jax.block_until_ready(pjit_gen_data(key))
  presharded_layers = jax.block_until_ready(pjit_gen_layers(key))
  activate_profiler(args.profiler_path)
  TFLOPs_per_device = parameters * 6 * BATCH  / 10**12 / len(devices)
  time = simple_timeit(lambda : jax.block_until_ready(pjit_func(presharded_X, presharded_layers)))
  print(f"time is {time} seconds, TFLOP is {TFLOPs_per_device}, TFLOP/s is {TFLOPs_per_device/time}", flush = True)
  deactivate_profiler(args.profiler_path)