import functools
import typing
import numpy as np

# define the cache decorator
cache = functools.lru_cache(maxsize=None)


def is_power_of_two(x: float):
    exp = float(np.log2(x))
    return exp.is_integer()
