# -*- coding: utf-8 -*-

from __future__ import print_function

import copy

import numpy as np
import torch

from nics_fix_pt.utils import get_int
from nics_fix_pt.consts import QuantizeMethod, RangeMethod

__all__ = ["quantize"]


def _do_quantize(data, scale, bit_width, symmetric=True, stochastic=False, group=False):
    '''
    when sym is not true, the input scale will be 2-value tensor [min,max]
    by defalutm the scale is a single fp-value, denoting range [-max, max]
    '''
    # The grouping are only applied for conv Parts
    # print(data.shape)
    if len(data.shape)<4:
        group = False

        
    bit_width = bit_width.to(data.device)
    tensor_2 = torch.autograd.Variable(torch.FloatTensor([2.0]),
                                       requires_grad=False).to(data.device)
    if symmetric:
        # assert len(scale) == 1
        dynamic_range=2*scale
        maxs = scale
        mins = scale*(-1)
    else:
        '''
        actually in real hardware implmention, the asymmetric quantization
        will be implemented through 1-fp scale and 1-int zero-point to 
        constraint the range, here for simplicity of software simulation,
        we simply define its range
        '''
        # assert len(scale) == 2
        dynamic_range = scale[1] - scale[0]
        maxs = scale[1]
        mins = scale[0]

    # TODO: add different grouping methods here
    if group == "batch":
        maxs = maxs.reshape(-1,1,1,1).expand_as(data)
        mins = mins.reshape(-1,1,1,1).expand_as(data)
        data_to_devise = dynamic_range.reshape(-1,1,1,1)
    elif group == "channel":
        maxs = maxs.reshape(1,-1,1,1).expand_as(data)
        mins = mins.reshape(1,-1,1,1).expand_as(data)
        data_to_devise = dynamic_range.reshape(1,-1,1,1)
    else:
        data_to_devise = dynamic_range

    
    step = data_to_devise/torch.pow(2, bit_width)

    if stochastic:
        output = StraightThroughStochasticRound.apply(data / step)*step
    else:
        output = StraightThroughRound.apply(data / step)*step

    # Since torch.clamp dose not support clamp with multiple value, so here is an alternate
    if group is not False:
        output = torch.min(torch.max(mins,output), maxs)
    else:
        output = torch.clamp(output,float(mins),float(maxs))





    # Symmetric Rounding
    # minimum = -float(2. ** (scale.cpu().data.numpy()))
    # maximum = -minimum
    # maximum = -minimum - step
    # TODO: Even if the quantize cfg is "auto", some overflow may occur,
    #       and maybe cause some problems.
    #       such as maybe weights won't be able to be trained to change scale
    #       if the learning rate is not big enough.
    # Two possible solutions:
    # * Do not minus step at maximum when training on software, this may cause some
    #   small discrepancy between software simulation and actual hardware deployment.
    # * Modify the `new_scale` calculation.
    return (
        output,
        step,
    )


def quantize_cfg(data, scale, bitwidth, method, range_method=RangeMethod.RANGE_MAX, stochastic=False, float_scale=True, zero_point=True, group=False):
    '''
    stochastic - stochastic rounding
    range_method - how to decide dynamic range
    float_scale - whether the scale is chosen to be 2^K
    zero_point  - symm/asymm quantize
    '''
    if (
        not isinstance(method, torch.autograd.Variable)
        and not torch.is_tensor(method)
        and method == QuantizeMethod.FIX_NONE
    ):
        return data, None

    if group == "batch" and len(data.shape)==4:
        # Only applied for conv units 
        max_data = data.view(data.shape[0],-1).max(dim=1)[0]
        min_data = data.view(data.shape[0],-1).min(dim=1)[0]
    if group == "channel" and len(data.shape)==4:
        max_data = data.view(data.shape[1],-1).max(dim=1)[0]
        min_data = data.view(data.shape[1],-1).min(dim=1)[0]
    else:
        max_data = data.max()
        min_data = data.min()

    # Avoid extreme value
    EPS = torch.cuda.FloatTensor(max_data.shape).fill_(1e-5)
    max_data = torch.max(max_data, EPS)
    # min_data = torch.max(min_data, EPS) # FIXME: this could be useless since min_data could be pos/neg

    method_v = get_int(method)
    # EPS = torch.cuda.FloatTensor([1]).fill_(1e-5)

    if method_v == QuantizeMethod.FIX_NONE:
        return data, None
    elif method_v == QuantizeMethod.FIX_AUTO:
        range_method_v = get_int(range_method)
        if range_method_v == RangeMethod.RANGE_MAX:
            if float_scale:
                # Only support float scale with zero-point for now
                if zero_point:
                    new_scale = torch.stack([min_data, max_data])
                    scale.data = new_scale
                    return _do_quantize(data, scale, bitwidth, stochastic=stochastic,symmetric=False, group=group)
                else:
                    new_scale = max_data
                    scale.data = new_scale
                    return _do_quantize(data, scale, bitwidth, stochastic=stochastic,symmetric=True, group=group)
            else:
                new_scale = torch.pow(2,torch.ceil(
                    torch.log(max_data)
                    )/torch.cuda.FloatTensor([1]).fill_(np.log(2.)))

                scale.data = new_scale
                return _do_quantize(data, scale, bitwidth, stochastic=stochastic,group=group)

        elif range_method_v == RangeMethod.RANGE_MAX_TENPERCENT:
            # FIXME: Too slow
            scale = torch.pow(2,torch.ceil(
                torch.log(
                    torch.max(
                        # torch.kthvalue(torch.abs(data.view(-1)), 9 * (data.nelement() // 10))[0],
                        torch.topk(torch.abs(data.view(-1)), data.nelement() // 10)[0][-1],
                        # torch.tensor(EPS).float().to(data.device))
                        torch.cuda.FloatTensor(1).fill_(EPS))
                ) / torch.cuda.FloatTensor([1]).fill_(np.log(2.0))
            ))
            return _do_quantize(data, scale, bitwidth, stochastic=stochastic)

        elif range_method_v == RangeMethod.RANGE_3SIGMA:
            new_scale = torch.ceil(torch.log(new_boundary))
            new_boundary = torch.max(3*torch.std(data)+torch.abs(torch.mean(data)), torch.tensor(EPS).float().to(data.device),)
            new_scale = torch.pow(2,torch.ceil(torch.log(new_boundary) / np.log(2.0)))
            scale = new_scale
            return _do_quantize(data, scale, bitwidth, stochastic=stochastic, symmetric=not zero_point)

        elif range_method_v == RangeMethod.RANGE_SWEEP:
            # Iterat through other scale to find the proper scale to minimize error 
            # Noted that the scale is [(MAX - SWEEP),MAX]
            SWEEP = 3
            temp_scale = torch.ceil(torch.log(torch.max(
                torch.max(abs(data)),
                torch.tensor(EPS).float().to(data.device))) / np.log(2.0))
            for i in range(SWEEP):
                errors[i] = torch.abs(_do_quantize(data, temp_scale-i, bitwidth)[0] - data).sum()
            new_scale = torch.pow(2,temp_scale - errors.argmin())
            scale.data = new_scale
            return _do_quantize(data, scale, bitwidth, stochastic=stochastic)

        else:
            raise NotImplementedError()

    elif method_v == QuantizeMethod.FIX_FIXED:

        # TODO: Check whether float_scale automatically adjust through inference
        # If float_scale, do as FIX_AUTO does
        if float_scale:
            # Only support float scale with zero-point for now
            if zero_point:
                new_scale = [data.min(), data.max()]
                scale.data = torch.FloatTensor(new_scale)
                return _do_quantize(data, scale, bitwidth, stochastic=stochastic,symmetric=False)
            else:
                EPS = 1e-5
                new_scale = torch.max(torch.max(torch.abs(data)),torch.cuda.FloatTensor([1]).fill_(EPS))
                scale.data = new_scale
                return _do_quantize(data, scale, bitwidth, stochastic=stochastic,symmetric=True)
        else:
            return _do_quantize(data, scale, bitwidth,stochastic=stochastic, symmetric=not zero_point)

    raise Exception("Quantitize method not legal: {}".format(method_v))


# https://discuss.pytorch.org/t/how-to-override-the-gradients-for-parameters/3417/6
class StraightThroughRound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, g):
        return g

class StraightThroughStochasticRound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # The Binary tensor denoting whether ceil or not, closer to ceil means for probabily choose ceil
        # return x.floor() + (torch.rand(x.shape).to(x.device) > x.ceil() - x)*torch.ones(x.shape).to(x.device)
        # out =  x.floor() + (torch.cuda.FloatTensor(x.shape).uniform_() > x.ceil() - x)*torch.cuda.FloatTensor(x.shape).fill_(1.)
        # out =  x.floor() + ((x.ceil() - x) < torch.cuda.FloatTensor([1]).fill_(np.random.uniform()))*torch.cuda.FloatTensor(x.shape).fill_(1.)
        noise = torch.cuda.FloatTensor(x.shape).uniform_(-0.5,0.5)
        x.add_(noise)
        return x

    @staticmethod
    def backward(ctx, g):
        return g



class QuantitizeGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, bitwidth, method, range_method=RangeMethod.RANGE_MAX, stochastic=False, float_scale=False, zero_point=False, group=False):
        # FIXME: save the tensor/variables for backward,
        #        maybe should use `ctx.save_for_backward` for standard practice
        # but `save_for_backward` requires scale/bitwidth/method all being of type `Variable`...
        ctx.saved = (scale, bitwidth, method, range_method, stochastic, float_scale, zero_point, group)
        return x

    @staticmethod
    def backward(ctx, g):
        return quantize_cfg(g, *ctx.saved)[0], None, None, None, None, None, None, None, None


def quantize(param, fix_cfg={}, fix_grad_cfg={}, kwarg_cfg={}, name=""):
    # fix_cfg/fix_grad_cfg is the configuration saved;
    # kwarg_cfg is the overriding configuration supplied for each `forward` call
    data_cfg = copy.copy(fix_cfg)
    data_cfg.update(kwarg_cfg.get(name + "_fix", {}))
    grad_cfg = copy.copy(fix_grad_cfg)
    grad_cfg.update(kwarg_cfg.get(name + "_grad_fix", {}))
    method = data_cfg.get("method", QuantizeMethod.FIX_NONE)

    step = 0
    # quantize data
    out_param = param
    # TODO: Insert grouping method here

    if (
        isinstance(method, torch.autograd.Variable)
        or torch.is_tensor(method)
        or method != QuantizeMethod.FIX_NONE
    ):
        out_param, stepp = quantize_cfg(
            out_param,
            data_cfg["scale"],
            data_cfg["bitwidth"],
            data_cfg["method"],
            data_cfg.get("range_method", RangeMethod.RANGE_MAX),
            data_cfg.get("stochastic", False),
            data_cfg.get("float_scale", False),
            data_cfg.get("zero_point", False),
            data_cfg.get("group", False),
        )

    # quantize gradient
    method = grad_cfg.get("method", QuantizeMethod.FIX_NONE)
    if (
        isinstance(method, torch.autograd.Variable)
        or torch.is_tensor(method)
        or method != QuantizeMethod.FIX_NONE
    ):
        out_param = QuantitizeGradient().apply(
            out_param,
            grad_cfg["scale"],
            grad_cfg["bitwidth"],
            grad_cfg["method"],
            grad_cfg.get("range_method", RangeMethod.RANGE_MAX),
            grad_cfg.get("stochastic", False),
            grad_cfg.get("float_scale", False),
            grad_cfg.get("zero_point", False),
            data_cfg.get("group", False),
        )


    out_param.data_cfg = data_cfg
    out_param.grad_cfg = grad_cfg
    if param is not out_param:
        # avoid memory leaking: old `buffer` tensors could remain referenced unexpectedly
        if hasattr(param, "nfp_actual_data"):
            del param.nfp_actual_data
            del param.data_cfg
            del param.grad_cfg
        out_param.nfp_actual_data = param  # avoid loop ref
    # NOTE: the returned step is data fix stepsize, not gradient fix step size;
    return out_param, step
