# -*- coding: utf-8 -*-

from __future__ import print_function

import sys
from collections import OrderedDict

import torch
from torch.nn import Module
from . import nn_fix, utils
from .fix_modules import FixMeta, Activation_fix

__all__ = []

container_registry = []

class FixTopModule(Module):
    """
    A module with some simple fix configuration manage utilities.
    """

# @utils.cache("{grad}")
def get_fix_configs(self, grad=False):
    cfg_dct = OrderedDict()
    for name, module in self._modules.iteritems():
        if isinstance(module.__class__, FixMeta) or isinstance(module, Activation_fix):
            cfg_dct[name] = getattr(module, "nf_fix_params" if not grad else "nf_fix_params_grad")
        elif isinstance(module, tuple(container_registry)):
            cfg_dct[name] = module.get_fix_configs(grad=grad)
    return cfg_dct
    
def get_fix_weights(self, copy=False):
    weights_dct = OrderedDict()
    for name, module in self._modules.iteritems():
        if isinstance(module.__class__, FixMeta) or isinstance(module, Activation_fix):
            dct = {}
            for n, param in module._parameters.iteritems():
                fix_cfg = module.nf_fix_params.get(n, {})
                fix_grad_cfg = module.nf_fix_params_grad.get(n, {})
                dct[n] = quantitize(param, fix_cfg, fix_grad_cfg, kwarg_cfg={}, name=n)
                if copy:
                    dct[n] = dct[n].clone()
            weights_dct[name] = dct
        elif isinstance(module, tuple(container_registry)):
            weights_dct[name] = module.get_fix_weights(copy=copy)
    return weights_dct

def print_fix_configs(self, data_fix_cfg=None, grad_fix_cfg=None, mds=None, prefix_spaces=0):
    if data_fix_cfg is None:
        data_fix_cfg = self.get_fix_configs(grad=False)
    if grad_fix_cfg is None:
        grad_fix_cfg = self.get_fix_configs(grad=True)
    if mds is None:
        mds = self._modules

    def _print(string, **kwargs):
        print("\n".join([" " * prefix_spaces + line for line in string.split("\n")]), **kwargs)

    for key in data_fix_cfg:
        _print("{}: {}".format(key, mds[key].__class__.__name__ if isinstance(mds[key], tuple(container_registry)) else mds[key]))
        d_cfg = data_fix_cfg[key]
        g_cfg = grad_fix_cfg[key]
        if isinstance(d_cfg, OrderedDict):
            self.print_fix_configs(d_cfg, g_cfg, mds[key]._modules, prefix_spaces=4 + prefix_spaces)
        else:
            # a dict of configs
            keys = set(d_cfg.keys()).union(g_cfg.keys())
            for param_name in keys:
                d_bw = utils.try_parse_int(d_cfg.get(param_name, {}).get("bitwidth", "f"))
                g_bw = utils.try_parse_int(g_cfg.get(param_name, {}).get("bitwidth", "f"))
                d_sc = utils.try_parse_int(d_cfg.get(param_name, {}).get("scale", "f"))
                g_sc = utils.try_parse_int(g_cfg.get(param_name, {}).get("scale", "f"))
                d_mt = utils.try_parse_int(d_cfg.get(param_name, {}).get("method", 0))
                g_mt = utils.try_parse_int(g_cfg.get(param_name, {}).get("method", 0))
                _print(("    {param_name:10}: d: bitwidth: {d_bw:3}; scale: {d_sc:3}; method: {d_mt:3}\n" + 
                        " " * 16+"g: bitwidth: {g_bw:3}; scale: {g_sc:3}; method: {g_mt:3}").format(param_name=param_name,
                                                                                                    d_bw=d_bw, g_bw=g_bw,
                                                                                                    d_sc=d_sc, g_sc=g_sc,
                                                                                                    d_mt=d_mt, g_mt=g_mt))

def set_fix_method(self, method, grad=False):
    for module in self._modules.itervalues():
        if isinstance(module, tuple(container_registry)):
            module.set_fix_method(method, grad=grad)
        elif isinstance(module.__class__, FixMeta) or isinstance(module, Activation_fix):
            fix_params = getattr(module, "nf_fix_params" if not grad else "nf_fix_params_grad")
            if isinstance(module.__class__, FixMeta):
                names = ["weight", "bias"]
            else:
                names = ["activation"]
            for n in names:
                if n in fix_params:
                    if "method" in fix_params[n]:
                        ori_method = fix_params[n]["method"]
                        if isinstance(ori_method, torch.autograd.Variable):
                            ori_method.data.numpy()[0] = method
                        elif torch.is_tensor(ori_method):
                            ori_method.numpy()[0] = method
                        else:
                            print("WARINING: setting a config field that is not a Tensor/variable might be of no use...")

Module.get_fix_configs = get_fix_configs
Module.print_fix_configs = print_fix_configs
Module.set_fix_method = set_fix_method

def register_fix_container(cls):
    for method_name in ["get_fix_configs", "get_fix_weights", "print_fix_configs", "set_fix_method"]:
        setattr(cls, method_name, getattr(sys.modules[__name__], method_name))
        container_registry.append(cls)

register_fix_container(torch.nn.Sequential)
register_fix_container(torch.nn.ModuleList)
register_fix_container(torch.nn.ParameterList)
register_fix_container(FixTopModule)

nn_fix.FixTopModule = FixTopModule
