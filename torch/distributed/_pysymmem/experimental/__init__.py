"""Opt-in collective experiments; importing registers the experimental backends.

Triton and CUDA symmetric memory are required. These implementations are not
production backends and are currently validated on a single eight-H100 domain.
"""

from .all_reduce import FP32AccumBackend
from .all_reduce_candidate2 import FP32AccumCandidate2Backend
from .uneven import UnevenBackend


__all__ = ["FP32AccumBackend", "FP32AccumCandidate2Backend", "UnevenBackend"]
