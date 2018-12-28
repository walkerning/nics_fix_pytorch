# -*- coding: utf-8 -*-

from __future__ import print_function

import six
from collections import OrderedDict

import torch
from torch.nn import Module
from .quant import quantitize
from . import nn_fix, utils

def fix_forward(self, input, **kwargs):
    if not isinstance(input, dict):
        input = {"input": input}
    for n, param in six.iteritems(self._parameters):
        fix_cfg = self.nf_fix_params.get(n, {})
        fix_grad_cfg = self.nf_fix_params_grad.get(n, {})
        set_n, self.step = quantitize(param, fix_cfg, fix_grad_cfg, kwarg_cfg=input, name=n)
        object.__setattr__(self, n, set_n)
    return super(self.__class__, self).forward(input["input"], **kwargs)

class FixMeta(type):
    def __new__(mcls, name, bases, attrs):
        # Construct class name
        name = bases[0].__name__ + "_fix"
        attrs["forward"] = fix_forward
        cls = super(FixMeta, mcls).__new__(mcls, name, bases, attrs)
        setattr(nn_fix, name, cls)
        return cls

def register_fix_module(cls):
    @six.add_metaclass(FixMeta)
    class __a_not_use_name(cls):
        def __init__(self, *args, **kwargs):
            # Pop and parse fix configuration from kwargs
            assert "nf_fix_params" in kwargs and isinstance(kwargs["nf_fix_params"], dict), "Must specifiy `nf_fix_params` keyword arguments, and `nf_fix_params_grad` is optional."
            self.nf_fix_params = kwargs.pop("nf_fix_params")
            self.nf_fix_params_grad = kwargs.pop("nf_fix_params_grad", {})
            self.step = 0
            cls.__init__(self, *args, **kwargs)

class Activation_fix(Module):
    def __init__(self, **kwargs):
        super(Activation_fix, self).__init__()
        assert "nf_fix_params" in kwargs and isinstance(kwargs["nf_fix_params"], dict),\
            "Must specifiy `nf_fix_params` keyword arguments, and `nf_fix_params_grad` is optional."
        self.nf_fix_params = kwargs.pop("nf_fix_params")
        self.nf_fix_params_grad = kwargs.pop("nf_fix_params_grad", {})
        self.step = 0
    
    def forward(self, input):
        if not isinstance(input, dict):
            input = {"input": input}
        name = "activation"
        fix_cfg = self.nf_fix_params.get(name, {})
        fix_grad_cfg = self.nf_fix_params_grad.get(name, {})
        self.activation, self.step = quantitize(input["input"], fix_cfg, fix_grad_cfg, kwarg_cfg=input, name=name)
        return self.activation

class FixTopModule(Module):
    """
    A module with some simple fix configuration manage utilities.
    """

    def load_fix_configs(self, cfgs, grad=False):
        assert isinstance(cfgs, (OrderedDict, dict))
        for name, module in six.iteritems(self._modules):
            if isinstance(module, FixTopModule):
                module.load_fix_config(cfgs[name], grad=grad)
            elif isinstance(module.__class__, FixMeta) or isinstance(module, Activation_fix):
                setattr(module, "nf_fix_params" if not grad else "nf_fix_params_grad", utils.try_parse_variable(cfgs[name]))

    def get_fix_configs(self, grad=False, data_only=False):
        cfg_dct = OrderedDict()
        for name, module in six.iteritems(self._modules):
            if isinstance(module, FixTopModule):
                cfg_dct[name] = module.get_fix_configs(method, grad=grad)
            elif isinstance(module.__class__, FixMeta) or isinstance(module, Activation_fix):
                cfg_dct[name] = getattr(module, "nf_fix_params" if not grad else "nf_fix_params_grad")
                if data_only:
                    cfg_dct[name] = utils.try_parse_int(cfg_dct[name])
        return cfg_dct
        
    def print_fix_configs(self, data_fix_cfg=None, grad_fix_cfg=None, prefix_spaces=0):
        if data_fix_cfg is None:
            data_fix_cfg = self.get_fix_configs(grad=False)
        if grad_fix_cfg is None:
            grad_fix_cfg = self.get_fix_configs(grad=True)
        def _print(string, **kwargs):
            print("\n".join([" " * prefix_spaces + line for line in string.split("\n")]), **kwargs)
        for key in data_fix_cfg:
            _print(key)
            d_cfg = data_fix_cfg[key]
            g_cfg = grad_fix_cfg[key]
            if isinstance(d_cfg, OrderedDict):
                self.print_fix_configs(d_cfg, g_cfg, prefix_spaces=2)
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
                    _print(("  {param_name:10}: d: bitwidth: {d_bw:3}; scale: {d_sc:3}; method: {d_mt:3}\n" + 
                            " " * 14+"g: bitwidth: {g_bw:3}; scale: {g_sc:3}; method: {g_mt:3}").format(param_name=param_name,
                                                                                                        d_bw=d_bw, g_bw=g_bw,
                                                                                                        d_sc=d_sc, g_sc=g_sc,
                                                                                                        d_mt=d_mt, g_mt=g_mt))

    def set_fix_method(self, method, grad=False):
        for module in six.itervalues(self._modules):
            if isinstance(module, FixTopModule):
                module.set_fix_method(method, grad=grad)
            elif isinstance(module.__class__, FixMeta) or isinstance(module, Activation_fix):
                fix_params = getattr(module, "nf_fix_params" if not grad else "nf_fix_params_grad")
                for n in fix_params:
                    if "method" in fix_params[n]:
                        ori_method = fix_params[n]["method"]
                        if isinstance(ori_method, torch.autograd.Variable):
                            ori_method.data.numpy()[0] = method
                        elif torch.is_tensor(ori_method):
                            ori_method.numpy()[0] = method
                        else:
                            fix_params[n]["method"] = method
                
nn_fix.Activation_fix = Activation_fix
nn_fix.FixTopModule = FixTopModule
