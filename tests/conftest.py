import pytest
import torch
from torch.nn import Module
from torch.nn import functional as F
from torch.nn.parameter import Parameter

class TestModule(Module):
    def __init__(self, input_num):
        super(TestModule, self).__init__()
        self.param = Parameter(torch.Tensor(1, input_num))
        self.reset_parameters()

    def reset_parameters(self):
        # fake data
        with torch.no_grad():
            self.param.fill_(0)
            self.param[0, 0] = 0.25111
            self.param[0, 1] = 0.5

    def forward(self, input):
        return F.linear(input, self.param, None)

@pytest.fixture
def module_cfg(request):
    import nics_fix_pt as nfp
    import nics_fix_pt.nn_fix as nnf
    nfp.fix_modules.register_fix_module(TestModule)
    cfg = nfp.utils._generate_default_fix_cfg(["param"], scale=-1, bitwidth=2, method=nfp.FIX_AUTO)
    update_cfg = getattr(request, "param", {})
    input_num = update_cfg.pop("input_num")
    cfg["param"].update(update_cfg)
    module = nnf.TestModule_fix(input_num=input_num, nf_fix_params=cfg)
    return module, cfg
