import os
with open(os.path.join(os.path.dirname(__file__), "VERSION")) as f:
    __version__ = f.read().strip()

from nics_fix_pt.quant import *
import nics_fix_pt.nn_fix_inner
from nics_fix_pt import nn_fix, fix_modules

class nn_auto_register(object):
    """
    An auto register helper that automatically register all not-registered modules by proxing to modules in torch.nn.

    NOTE: We do not guarantee all auto-registered fixed nn modules will well behave, as they are not tested. Although, I thought it will work in normal cases.
    Use with care!!

    Usage: from nics_fix_pt import NAR as nnf
    then e.g. `nnf.Bilinear_fix` and `nnf.Bilinear` can all be used as a fixed-point module.
    """
    def __getattr__(self, name):
        import torch
        attr = getattr(nn_fix, name, None)
        if attr is None:
            if name.endswith("_fix"):
                ori_name = name[:-4]
            else:
                ori_name = name
            ori_cls = getattr(torch.nn, ori_name)
            fix_modules.register_fix_module(ori_cls, register_name=ori_name + "_fix")
            return getattr(nn_fix, ori_name + "_fix", None)
        return attr

NAR = nn_auto_register()
