# -*- coding: utf-8 -*-

from __future__ import print_function

import six

import torch
from torch.nn import Module
from .quant import quantitize
from . import nn_fix

def fix_forward(self, input, **kwargs):
    if not isinstance(input, dict):
        input = {"input": input}
    for n, param in self._parameters.iteritems():
        fix_cfg = self.nf_fix_params.get(n, {})
        fix_grad_cfg = self.nf_fix_params_grad.get(n, {})
        object.__setattr__(self, n, quantitize(param, fix_cfg, fix_grad_cfg, kwarg_cfg=input, name=n))
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

            cls.__init__(self, *args, **kwargs)

class Activation_fix(Module):
    def __init__(self, **kwargs):
        super(Activation_fix, self).__init__()
        assert "nf_fix_params" in kwargs and isinstance(kwargs["nf_fix_params"], dict),\
            "Must specifiy `nf_fix_params` keyword arguments, and `nf_fix_params_grad` is optional."
        self.nf_fix_params = kwargs.pop("nf_fix_params")
        self.nf_fix_params_grad = kwargs.pop("nf_fix_params_grad", {})
    
    def forward(self, input):
        if not isinstance(input, dict):
            input = {"input": input}
        name = "activation"
        fix_cfg = self.nf_fix_params.get(name, {})
        fix_grad_cfg = self.nf_fix_params_grad.get(name, {})
        self.activation = quantitize(input["input"], fix_cfg, fix_grad_cfg, kwarg_cfg=input, name=name)
        return self.activation
                
nn_fix.Activation_fix = Activation_fix
