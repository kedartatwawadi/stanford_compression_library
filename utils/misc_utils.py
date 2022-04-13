import functools

# define the cache decorator
cache = functools.lru_cache(maxsize=None)
