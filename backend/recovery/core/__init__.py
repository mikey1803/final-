"""The shared simulation kernel: protocols, RNG streams, episode log, episode loop.

Depends only on ``numpy``/``pandas``/the standard library. Must never import
concrete policies, environments, or classifiers (see recovery/policies/,
recovery/envs/, recovery/classify/) -- only their protocol shapes.
"""
