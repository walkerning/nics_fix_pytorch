# -*- coding: utf-8 -*-

import copy
import inspect
from functools import wraps

def try_parse_int(something):
    try:
        return int(something)
    except ValueError:
        return something

def cache(format_str):
    def _cache(func):
        _cache_dct = {}
        argspec = inspect.getargspec(func)
        default_kwargs = dict(zip(argspec.args[len(argspec.args) - len(argspec.defaults):],
                                  argspec.defaults))
        @wraps(func)
        def _func(*args, **kwargs):
            args_dct = copy.copy(default_kwargs)
            args_dct.update(dict(zip(argspec.args, args)))
            args_dct.update(kwargs)
            cache_str = format_str.format(**args_dct)
            if cache_str not in _cache_dct:
                _cache_dct[cache_str] = func(*args, **kwargs)
            return _cache_dct[cache_str]
        return _func
    return _cache
