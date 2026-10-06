"""Shared validation for the public reconstruction API."""
from numbers import Integral, Real
import math


def validate_solver_options(timeout, n_threads, seed):
    if isinstance(timeout, bool) or not isinstance(timeout, Real) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a finite positive number of seconds.")
    if isinstance(n_threads, bool) or not isinstance(n_threads, Integral) or (n_threads != -1 and n_threads < 1):
        raise ValueError("n_threads must be -1 (all available threads) or a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed <= 2**31 - 1:
        raise ValueError("seed must be an integer between 0 and 2**31 - 1.")
