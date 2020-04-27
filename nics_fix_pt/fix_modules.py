# -*- coding: utf-8 -*-

from __future__ import print_function

import warnings
from collections import OrderedDict

import six

import torch
from torch.nn import Module
import torch.nn.functional as F
from . import nn_fix, utils, quant


# ---- helpers ----
def _get_kwargs(self, true_kwargs):
    default_kwargs = utils.get_kwargs(self.__class__)
    if not default_kwargs:
        return true_kwargs
    # NOTE: here we do not deep copy the default values,
    # so non-atom type default value such as dict/list/tensor will be shared
    kwargs = {k: v for k, v in six.iteritems(default_kwargs)}
    kwargs.update(true_kwargs)
    return kwargs

def _get_fix_cfg(self, name, grad=False):
    if not grad:
        cfg = self.nf_fix_params.get(name, {})
        if "scale" in cfg:
            cfg["scale"] = self._buffers["{}_fp_scale".format(name)]
    else:
        cfg = self.nf_fix_params_grad.get(name, {})
        if "scale" in cfg:
            cfg["scale"] = self._buffers["{}_grad_fp_scale".format(name)]
    return cfg

def _register_fix_buffers(self, patch_register=True):
    # register scale tensors as buffers, for correct use in multi-gpu data parallel model
    avail_keys = list(self._parameters.keys()) + list(self._buffers.keys())
    for name, cfg in six.iteritems(self.nf_fix_params):
        if patch_register and name not in avail_keys:
            warnings.warn(("{} not available in {}, this specific fixed config "
                           "will not have effects").format(name, self))
        if "scale" in cfg:
            self.register_buffer("{}_fp_scale".format(name), cfg["scale"])

    avail_keys = list(self._parameters.keys())
    for name, cfg in six.iteritems(self.nf_fix_params_grad):
        if patch_register and name not in avail_keys:
            warnings.warn(("{} not available in {}, this specific grads fixed config "
                           "will not have effects").format(name, self))
        if "scale" in cfg:
            self.register_buffer("{}_grad_fp_scale".format(name), cfg["scale"])

# --------

def get_fix_forward(cur_cls):
    #pylint: disable=protected-access
    def fix_forward(self, inputs, **kwargs):
        if not isinstance(inputs, dict):
            inputs = {"inputs": inputs}
        for n, param in six.iteritems(self._parameters):
            if not isinstance(param, (torch.Tensor, torch.autograd.Variable)):
                continue
            fix_cfg = _get_fix_cfg(self, n)
            fix_grad_cfg = _get_fix_cfg(self, n, grad=True)
            set_n, _ = quant.quantize(param, fix_cfg, fix_grad_cfg, kwarg_cfg=inputs, name=n)
            object.__setattr__(self, n, set_n)
        for n, param in six.iteritems(self._buffers):
            if not isinstance(param, (torch.Tensor, torch.autograd.Variable)):
                continue
            fix_cfg = _get_fix_cfg(self, n)
            fix_grad_cfg = _get_fix_cfg(self, n, grad=True)
            set_n, _ = quant.quantize(param, fix_cfg, fix_grad_cfg, kwarg_cfg=inputs, name=n)
            object.__setattr__(self, n, set_n)
        res = super(cur_cls, self).forward(inputs["inputs"], **kwargs)
        for n, param in six.iteritems(self._buffers):
            # set buffer back, as there will be no gradient, just in-place modification
            # FIXME: For fixed-point batch norm,
            # the running mean/var accumulattion is on quantized mean/var,
            # which means it might fail to update the running mean/var
            # if the updating momentum is too small
            updated_buffer = getattr(self, n)
            if updated_buffer is not self._buffers[n]:
                self._buffers[n].copy_(updated_buffer)
        return res
    return fix_forward


class FixMeta(type):
    def __new__(mcs, name, bases, attrs):
        # Construct class name
        if not attrs.get("__register_name__", None):
            attrs["__register_name__"] = bases[0].__name__ + "_fix"
        name = attrs["__register_name__"]
        cls = super(FixMeta, mcs).__new__(mcs, name, bases, attrs)
        # if already subclass
        if not isinstance(bases[0], FixMeta):
            cls.forward = get_fix_forward(cur_cls=cls)
        setattr(nn_fix, name, cls)
        return cls


def register_fix_module(cls, register_name=None):
    @six.add_metaclass(FixMeta)
    class __a_not_use_name(cls):
        __register_name__ = register_name

        def __init__(self, *args, **kwargs):
            kwargs = _get_kwargs(self, kwargs)
            # Pop and parse fix configuration from kwargs
            assert "nf_fix_params" in kwargs and isinstance(
                kwargs["nf_fix_params"], dict
            ), (
                "Must specifiy `nf_fix_params` keyword arguments, "
                "and `nf_fix_params_grad` is optional."
            )
            self.nf_fix_params = kwargs.pop("nf_fix_params")
            self.nf_fix_params_grad = kwargs.pop("nf_fix_params_grad", {}) or {}
            cls.__init__(self, *args, **kwargs)
            _register_fix_buffers(self, patch_register=True)
            # avail_keys = list(self._parameters.keys()) + list(self._buffers.keys())
            # self.nf_fix_params = {k: self.nf_fix_params[k]
            #                       for k in avail_keys if k in self.nf_fix_params}
            # self.nf_fix_params_grad = {k: self.nf_fix_params_grad[k]
            #                            for k in avail_keys if k in self.nf_fix_params_grad}


class Activation_fix(Module):
    def __init__(self, **kwargs):
        super(Activation_fix, self).__init__()
        kwargs = _get_kwargs(self, kwargs)
        assert "nf_fix_params" in kwargs and isinstance(
            kwargs["nf_fix_params"], dict
        ), "Must specifiy `nf_fix_params` keyword arguments, and `nf_fix_params_grad` is optional."
        self.nf_fix_params = kwargs.pop("nf_fix_params")
        self.nf_fix_params_grad = kwargs.pop("nf_fix_params_grad", {}) or {}
        self.activation = None

        # register scale as buffers
        _register_fix_buffers(self, patch_register=False)

    def forward(self, inputs):
        if not isinstance(inputs, dict):
            inputs = {"inputs": inputs}
        name = "activation"
        fix_cfg = self.nf_fix_params.get(name, {})
        fix_grad_cfg = self.nf_fix_params_grad.get(name, {})
        self.activation, _ = quant.quantize(
            inputs["inputs"], fix_cfg, fix_grad_cfg, kwarg_cfg=inputs, name=name
        )
        return self.activation

class ConvBN_fix(Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                padding=0, dilation=1, groups=1,
                eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, **kwargs):
        super(ConvBN_fix, self).__init__()
        kwargs = _get_kwargs(self, kwargs)
        assert "nf_fix_params" in kwargs and isinstance(
            kwargs["nf_fix_params"], dict
        ), "Must specifiy `nf_fix_params` keyword arguments, and `nf_fix_params_grad` is optional."
        self.nf_fix_params = kwargs.pop("nf_fix_params")
        self.nf_fix_params_grad = kwargs.pop("nf_fix_params_grad", {}) or {}
        if self.nf_fix_params_grad:
            warnings.warn(
                "Gradient fixed-point cfgs will NOT take effect! Because, "
                "Gradient quantization is usually used to simulate training on hardware. "
                "However, merged ConvBN is designed for mitigating the discrepancy between "
                "training and behaviour on deploy-only hardware; "
                "and enable relatively more accurate running mean/var accumulation "
                "during software training. Use these two together might not make sense."
            )

        # init the two floating-point sub-modules
        self.conv = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels, eps, momentum, affine, track_running_stats)

        # conv and bn attributes
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_size = self.conv.kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        # the quantized combined weights and bias
        self.weight = self.conv.weight
        self.bias = self.bn.bias

        # register scale as buffers
        _register_fix_buffers(self, patch_register=False)

    def forward(self, inputs):
        if self.training:
            out = self.conv(inputs)
            # dummy output, just to accumulate running mean and running var (floating-point)
            _ = self.bn(out)

            # calculate batch var/mean
            mean = torch.mean(out, dim=[0, 2, 3])
            var = torch.var(out, dim=[0, 2, 3])
        else:
            mean = self.bn.running_mean
            var = self.bn.running_var

        inputs = {"inputs": inputs}
        # parameters/buffers to be combined
        bn_scale = self.bn.weight
        bn_bias = self.bn.bias
        bn_eps = self.bn.eps
        conv_weight = self.conv.weight
        conv_bias = self.conv.bias or 0. # could be None

        # combine new weights/bias
        comb_weight = conv_weight * (bn_scale / torch.sqrt(var + bn_eps)).view(-1, 1, 1, 1)
        comb_bias = bn_bias + (conv_bias - mean) * bn_scale / torch.sqrt(var + bn_eps)

        # quantize the combined weights/bias (as what would be done in hardware deploy scenario)
        comb_weight, _ = quant.quantize(comb_weight, self.nf_fix_params.get("weight", {}), {},
                                          kwarg_cfg=inputs, name="weight")
        comb_bias, _ = quant.quantize(comb_bias, self.nf_fix_params.get("bias", {}), {},
                                        kwarg_cfg=inputs, name="bias")

        # run the fixed-point combined convbn
        convbn_out = F.conv2d(inputs["inputs"], comb_weight, comb_bias,
                              self.stride, self.padding, self.dilation, self.groups)
        object.__setattr__(self, "weight", comb_weight)
        object.__setattr__(self, "bias", comb_bias)
        return convbn_out


class FixTopModule(Module):
    """
    A module with some simple fix configuration manage utilities.
    """

    def __init__(self, *args, **kwargs):
        super(FixTopModule, self).__init__(*args, **kwargs)

        # To be portable between python2/3, use staticmethod for these utility methods,
        # and patch instance method here.
        # As Python2 do not support binding instance method to a class that is not a FixTopModule
        self.fix_state_dict = FixTopModule.fix_state_dict.__get__(self)
        self.load_fix_configs = FixTopModule.load_fix_configs.__get__(self)
        self.get_fix_configs = FixTopModule.get_fix_configs.__get__(self)
        self.print_fix_configs = FixTopModule.print_fix_configs.__get__(self)
        self.set_fix_method = FixTopModule.set_fix_method.__get__(self)

    @staticmethod
    def fix_state_dict(self, destination=None, prefix="", keep_vars=False):
        r"""FIXME: maybe do another quantization to make sure all vars are quantized?

        Returns a dictionary containing a whole fixed-point state of the module.

        Both parameters and persistent buffers (e.g. running averages) are
        included. Keys are corresponding parameter and buffer names.

        Returns:
            dict:
                a dictionary containing a whole state of the module

        Example::

            >>> module.state_dict().keys()
            ['bias', 'weight']

        """
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        destination._metadata[prefix[:-1]] = local_metadata = dict(
            version=self._version
        )
        for name, param in self._parameters.items():
            if param is not None:
                if isinstance(self.__class__, FixMeta):  # A fixed-point module
                    # Get the last used version of the parameters
                    thevar = getattr(self, name)
                else:
                    thevar = param
                destination[prefix + name] = thevar if keep_vars else thevar.data
        for name, buf in self._buffers.items():
            if buf is not None:
                if isinstance(self.__class__, FixMeta):  # A fixed-point module
                    # Get the last saved version of the buffers,
                    # which can be of float precision
                    # (as buffers will be turned into fixed-point precision on the next forward)
                    thevar = getattr(self, name)
                else:
                    thevar = buf
                destination[prefix + name] = thevar if keep_vars else thevar.data
        for name, module in self._modules.items():
            if module is not None:
                FixTopModule.fix_state_dict(
                    module, destination, prefix + name + ".", keep_vars=keep_vars
                )
        for hook in self._state_dict_hooks.values():
            hook_result = hook(self, destination, prefix, local_metadata)
            if hook_result is not None:
                destination = hook_result
        return destination

    @staticmethod
    def load_fix_configs(self, cfgs, grad=False):
        assert isinstance(cfgs, (OrderedDict, dict))
        for name, module in six.iteritems(self._modules):
            if isinstance(module.__class__, FixMeta) or isinstance(
                module, Activation_fix
            ):
                if name not in cfgs:
                    print(
                        (
                            "WARNING: Fix configuration for {} not found in the configuration! "
                            "Make sure you know why this happened or "
                            "there might be some subtle error!"
                        ).format(name)
                    )
                else:
                    setattr(
                        module,
                        "nf_fix_params" if not grad else "nf_fix_params_grad",
                        utils.try_parse_variable(cfgs[name]),
                    )
            elif isinstance(module, FixTopModule):
                module.load_fix_configs(cfgs[name], grad=grad)
            else:
                FixTopModule.load_fix_configs(module, cfgs[name], grad=grad)

    @staticmethod
    def get_fix_configs(self, grad=False, data_only=False):
        """
        get_fix_configs:

        Parameters:
            grad: BOOLEAN(default False), whether or not to get the gradient configs
                instead of data configs.
            data_only: BOOLEAN(default False), whether or not to get the numbers instead
                of `torch.Tensor` (which can be modified in place).
        """
        cfg_dct = OrderedDict()
        for name, module in six.iteritems(self._modules):
            if isinstance(module.__class__, FixMeta) or isinstance(
                module, Activation_fix
            ):
                cfg_dct[name] = getattr(
                    module, "nf_fix_params" if not grad else "nf_fix_params_grad"
                )
                if data_only:
                    cfg_dct[name] = utils.try_parse_int(cfg_dct[name])
            elif isinstance(module, FixTopModule):
                cfg_dct[name] = module.get_fix_configs(grad=grad, data_only=data_only)
            else:
                cfg_dct[name] = FixTopModule.get_fix_configs(
                    module, grad=grad, data_only=data_only
                )
        return cfg_dct

    @staticmethod
    def print_fix_configs(self, data_fix_cfg=None, grad_fix_cfg=None, prefix_spaces=0):
        if data_fix_cfg is None:
            data_fix_cfg = self.get_fix_configs(grad=False)
        if grad_fix_cfg is None:
            grad_fix_cfg = self.get_fix_configs(grad=True)

        def _print(string, **kwargs):
            print(
                "\n".join([" " * prefix_spaces + line for line in string.split("\n")])
                + "\n",
                **kwargs
            )

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
                    d_bw = utils.try_parse_int(
                        d_cfg.get(param_name, {}).get("bitwidth", "f")
                    )
                    g_bw = utils.try_parse_int(
                        g_cfg.get(param_name, {}).get("bitwidth", "f")
                    )
                    d_sc = utils.try_parse_int(
                        d_cfg.get(param_name, {}).get("scale", "f")
                    )
                    g_sc = utils.try_parse_int(
                        g_cfg.get(param_name, {}).get("scale", "f")
                    )
                    d_mt = utils.try_parse_int(
                        d_cfg.get(param_name, {}).get("method", 0)
                    )
                    g_mt = utils.try_parse_int(
                        g_cfg.get(param_name, {}).get("method", 0)
                    )
                    _print(
                        (
                            "  {param_name:10}: d: bitwidth: {d_bw:3}; "
                            "scale: {d_sc:3}; method: {d_mt:3}\n"
                            + " " * 14
                            + "g: bitwidth: {g_bw:3}; scale: {g_sc:3}; method: {g_mt:3}"
                        ).format(
                            param_name=param_name,
                            d_bw=d_bw,
                            g_bw=g_bw,
                            d_sc=d_sc,
                            g_sc=g_sc,
                            d_mt=d_mt,
                            g_mt=g_mt,
                        )
                    )

    @staticmethod
    def set_fix_method(self, method=None, method_by_type=None, method_by_name=None, grad=False):
        for module_name, module in six.iteritems(self._modules):
            if isinstance(module.__class__, FixMeta) or isinstance(
                module, Activation_fix
            ):
                fix_params = getattr(
                    module, "nf_fix_params" if not grad else "nf_fix_params_grad"
                )
                if method_by_name is not None and module_name in method_by_name:
                    for param_n, param_method in six.iteritems(method_by_name[module_name]):
                        assert param_n in fix_params, \
                            "{} is not a quantized parameter of module {}".format(
                                param_n, module_name)
                        _set_method(fix_params[param_n], param_method)
                else:
                    if method_by_type is not None:
                        param_method_cfg = method_by_type.get(
                            type(module).__name__,
                            method_by_type.get(type(module), None))
                    else:
                        param_method_cfg = None
                    if param_method_cfg is not None:
                        # specifiedd by method_by_type
                        for param_n, param_method in six.iteritems(param_method_cfg):
                            assert param_n in fix_params, \
                                "{} is not a quantized parameter of module {}".format(
                                    param_n, module_name)
                            _set_method(fix_params[param_n], param_method)
                    elif method is not None:
                        for param_n in fix_params:
                            _set_method(fix_params[param_n], method)
            elif isinstance(module, FixTopModule):
                module.set_fix_method(method, grad=grad)
            else:
                FixTopModule.set_fix_method(module, method, grad=grad)

# helpers
def _set_method(param_cfg, new_method):
    if new_method is None:
        return
    if "method" in param_cfg:
        ori_method = param_cfg["method"]
        if isinstance(ori_method, torch.autograd.Variable):
            ori_method.data.numpy()[0] = new_method
        elif torch.is_tensor(ori_method):
            ori_method.numpy()[0] = new_method
        else:
            param_cfg["method"] = new_method


nn_fix.Activation_fix = Activation_fix
nn_fix.FixTopModule = FixTopModule
nn_fix.ConvBN_fix = ConvBN_fix
