# -*- coding: utf-8 -*-

from __future__ import print_function

import copy

import numpy as np
import torch

__all__ = ["quantitize", "FIX_NONE", "FIX_AUTO", "FIX_FIXED"]

def _do_quantitize(data, scale, bit_width):
    scale_f = scale.float()
    # print(scale_f)
    step = torch.pow(torch.autograd.Variable(torch.FloatTensor([2.]), requires_grad=False), (scale_f - (bit_width - 1).float()))
    minimum = -torch.pow(torch.autograd.Variable(torch.FloatTensor([2.]), requires_grad=False), scale_f)
    if data.is_cuda:
        step = step.cuda()
        minimum = minimum.cuda()
    # maximum = -minimum - step
    maximum = -minimum
    # TODO: Even if the quantitize cfg is "auto", some overflow may occur, and maybe cause some problems.
    #       such as maybe weights won't be able to be trained to change scale if the learning rate is not big enough.
    # Two possible solutions:
    # * Do not minus step at maximum when training on software, 
    #   this may cause some small discrepancy between software simulation and actual hardware deployment.
    # * Modify the `new_scale` calculation.
    return torch.min(torch.max(StraightThroughRound.apply(data / step) * step, minimum), maximum), step

# quantitze methods
FIX_NONE = 0
FIX_AUTO = 1
FIX_FIXED = 2

def quantitize_cfg(data, scale, bitwidth, method):
    if not isinstance(method, torch.autograd.Variable) and not torch.is_tensor(method) and method == FIX_NONE:
        return data

    if torch.is_tensor(method):
        method_v = int(method.numpy()[0])
    elif isinstance(method, torch.autograd.Variable):
        method_v = int(method.data.numpy()[0])
    else:
        assert isinstance(method, (int, np.int))
        method_v = int(method)

    if method_v == FIX_NONE:
        return data
    elif method_v == FIX_AUTO:
        EPS = 1e-5
        new_scale = torch.ceil(torch.log(torch.max(torch.max(torch.abs(data)), torch.FloatTensor(1).fill_(EPS))) / np.log(2.))
        scale.data.numpy()[0] = new_scale
        return _do_quantitize(data, scale, bitwidth)
    elif method_v == FIX_FIXED:
        return _do_quantitize(data, scale, bitwidth)
    assert False, "Quantitize method not legal: {}".format(method_v)

# https://discuss.pytorch.org/t/how-to-override-the-gradients-for-parameters/3417/6
class StraightThroughRound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()
    @staticmethod
    def backward(ctx, g):
        return g

class QuantitizeGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x
    @staticmethod
    def backward(ctx, g):
        return quantitize_cfg(g)

def quantitize(param, fix_cfg={}, fix_grad_cfg={}, kwarg_cfg={}, name=""):
    data_cfg = copy.copy(fix_cfg)
    data_cfg.update(kwarg_cfg.get(name + "_fix", {}))
    grad_cfg = copy.copy(fix_grad_cfg)
    grad_cfg.update(kwarg_cfg.get(name + "_grad_fix", {}))
    method = data_cfg.get("method", FIX_NONE)
    step = 0
    if isinstance(method, torch.autograd.Variable) or torch.is_tensor(method) or method != FIX_NONE:
        param, step = quantitize_cfg(param, data_cfg["scale"],
                               data_cfg["bitwidth"], data_cfg["method"])
    # TODO: quantitize gradient
    # method = grad_cfg.get("method", FIX_NONE) 
    # if isinstance(method, torch.autograd.Variable) or torch.is_tensor(method) or method != FIX_NONE:
    #     param, step = quantitize_cfg(param, grad_cfg["scale"],
    #                            grad_cfg["bitwidth"], grad_cfg["method"])
    return param, step
