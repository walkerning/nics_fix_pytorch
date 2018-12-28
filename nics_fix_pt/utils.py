# -*- coding: utf-8 -*-

from six import iteritems
from collections import OrderedDict
import copy
import inspect
from functools import wraps

import numpy as np
import torch

def try_parse_variable(something):
    if isinstance(something, (dict, OrderedDict)): # recursive parse into dict values
        return {k: try_parse_variable(v) for k, v in iteritems(something)}
    try:
        return torch.autograd.Variable(torch.IntTensor(np.array([something])), requires_grad=False)
    except ValueError:
        return something

def try_parse_int(something):
    if isinstance(something, (dict, OrderedDict)): # recursive parse into dict values
        return {k: try_parse_int(v) for k, v in iteritems(something)}
    try:
        return int(something)
    except ValueError:
        return something

def cache(format_str):
    def _cache(func):
        _cache_dct = {}
        sig = inspect.signature(func)
        default_kwargs = {n: v.default for n, v in iteritems(sig.parameters) if v.default != inspect._empty}
        @wraps(func)
        def _func(*args, **kwargs):
            args_dct = copy.copy(default_kwargs)
            args_dct.update(dict(zip(sig.parameters.keys(), args)))
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
