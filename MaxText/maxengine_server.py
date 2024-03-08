"""Runs a server with maxtext."""

import jax
import os
import sys
import pyconfig

import  maxengine_config
from jetstream.core import server_lib
from jax.experimental.compilation_cache import compilation_cache as cc

# _PORT = flags.DEFINE_integer('port', 9000, 'port to listen on')
# _THREADS = flags.DEFINE_integer(
#     'threads', 64, 'number of worker threads in thread pool'
# )
# _CONFIG = flags.DEFINE_string(
#     'config',
#     'MaxtextInterleavedServer',
#     'available servers',
# )


def main(config):
  # No devices for local cpu test. A None for prefill and a None for generate.
  devices = server_lib.get_devices()
  server_config = maxengine_config.get_server_config('MaxtextInterleavedServer', config)
  # We separate credential from run so that we can unit test it with
  # local credentials.
  # TODO: Add grpc credentials for OSS.
  jetstream_server = server_lib.run(
      threads=128,
      port=9000,
      config=server_config,
      devices=devices,
  )
  jetstream_server.wait_for_termination()


if __name__ == '__main__':
  jax.config.update('jax_default_prng_impl', 'unsafe_rbg')
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  pyconfig.initialize(sys.argv)
  cfg = pyconfig.config
  cc.set_cache_dir(os.path.expanduser(cfg.jax_cache_dir))
  main(cfg)
