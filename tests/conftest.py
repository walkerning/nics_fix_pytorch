import pytest
import numpy as np
import torch
from torch.nn import Module
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import nics_fix_pt.nn_fix as nnf

def _generate_default_fix_cfg(names, scale=0, bitwidth=8, method=0):
    return {
        n: {
            "method": torch.autograd.Variable(
                torch.IntTensor(np.array([method])), requires_grad=False
            ),
            "scale": torch.autograd.Variable(
                torch.IntTensor(np.array([scale])), requires_grad=False
            ),
            "bitwidth": torch.autograd.Variable(
                torch.IntTensor(np.array([bitwidth])), requires_grad=False
            ),
        }
        for n in names
    }

class TestNetwork(nnf.FixTopModule):
    def __init__(self):
        super(TestNetwork, self).__init__()
        self.fix_params = {}
        for conv_name in ["conv1", "conv2"]:
            self.fix_params[conv_name] = _generate_default_fix_cfg(
                ["weight", "bias"], method=1, bitwidth=8)
        for bn_name in ["bn1", "bn2"]:
            self.fix_params[bn_name] = _generate_default_fix_cfg(
                ["weight", "bias", "running_mean", "running_var"], method=1, bitwidth=8)
        self.conv1 = nnf.Conv2d_fix(3, 64, (3, 3), padding=1,
                                    nf_fix_params=self.fix_params["conv1"])
        self.bn1 = nnf.BatchNorm2d_fix(64, nf_fix_params=self.fix_params["bn1"])
        self.conv2 = nnf.Conv2d_fix(64, 128, (3, 3), padding=1,
                                    nf_fix_params=self.fix_params["conv2"])
        self.bn2 = nnf.BatchNorm2d_fix(128, nf_fix_params=self.fix_params["bn2"])

@pytest.fixture
def test_network():
    return TestNetwork()

# ----
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
        # print("input: ", input, "param: ", self.param)
        return F.linear(input, self.param, None)


@pytest.fixture
def module_cfg(request):
    import nics_fix_pt as nfp
    import nics_fix_pt.nn_fix as nnf

    nfp.fix_modules.register_fix_module(TestModule)
    # default data/grad fix cfg for the parameter `param` of TestModule
    data_cfg = nfp.utils._generate_default_fix_cfg(
        ["param"], scale=-1, bitwidth=2, method=nfp.FIX_AUTO
    )
    grad_cfg = nfp.utils._generate_default_fix_cfg(
        ["param"], scale=-1, bitwidth=2, method=nfp.FIX_NONE
    )
    # the specified overriding cfgs: input_num, data fix cfg, grad fix cfg
    update_cfg = getattr(request, "param", {})
    input_num = update_cfg.pop("input_num", 3)
    data_update_cfg = update_cfg.get("data_cfg", {})
    grad_update_cfg = update_cfg.get("grad_cfg", {})
    data_cfg["param"].update(data_update_cfg)
    grad_cfg["param"].update(grad_update_cfg)
    module = nnf.TestModule_fix(
        input_num=input_num, nf_fix_params=data_cfg, nf_fix_params_grad=grad_cfg
    )
    return module, data_cfg, grad_cfg
