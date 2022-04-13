import functools
import typing

# define the cache decorator
cache = functools.lru_cache(maxsize=None)

