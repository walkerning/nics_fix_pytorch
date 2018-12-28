# -*- coding: utf-8 -*-

import six
import copy
import inspect
from functools import wraps

import numpy as np
import torch

def try_parse_int(something):
    try:
        return int(something)
    except ValueError:
        return something

def cache(format_str):
    def _cache(func):
        _cache_dct = {}
        sig = inspect.signature(func)
        default_kwargs = {n: v.default for n, v in six.iteritems(sig.parameters) if v.default != inspect._empty}
        @wraps(func)
        def _func(*args, **kwargs):
            args_dct = copy.copy(default_kwargs)
            args_dct.update(dict(zip(sig.keys(), args)))
            args_dct.update(kwargs)
            cache_str = format_str.format(**args_dct)
            if cache_str not in _cache_dct:
                _cache_dct[cache_str] = func(*args, **kwargs)
            return _cache_dct[cache_str]
        return _func
    return _cache

def _generate_default_fix_cfg(names, scale=0, bitwidth=8, method=0):
    return {n: {
        "method": torch.autograd.Variable(torch.IntTensor(np.array([method])), requires_grad=False),
        "scale": torch.autograd.Variable(torch.IntTensor(np.array([scale])), requires_grad=False),
        "bitwidth": torch.autograd.Variable(torch.IntTensor(np.array([bitwidth])), requires_grad=False)
    } for n in names}
