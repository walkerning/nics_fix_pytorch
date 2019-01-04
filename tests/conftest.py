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
    # default data/grad fix cfg for the parameter `param` of TestModule
    data_cfg = nfp.utils._generate_default_fix_cfg(["param"], scale=-1, bitwidth=2, method=nfp.FIX_AUTO)
    grad_cfg = nfp.utils._generate_default_fix_cfg(["param"], scale=-1, bitwidth=2, method=nfp.FIX_NONE)
    # the specified overriding cfgs: input_num, data fix cfg, grad fix cfg
    update_cfg = getattr(request, "param", {})
    input_num = update_cfg.pop("input_num", 3)
    data_update_cfg = update_cfg.get("data_cfg", {})
    grad_update_cfg = update_cfg.get("grad_cfg", {})
    data_cfg["param"].update(data_update_cfg)
    grad_cfg["param"].update(grad_update_cfg)
    module = nnf.TestModule_fix(input_num=input_num, nf_fix_params=data_cfg, nf_fix_params_grad=grad_cfg)
    return module, data_cfg, grad_cfg
