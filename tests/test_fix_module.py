import pytest

import numpy as np
import torch
from torch import nn
import torch.optim as optim

import nics_fix_pt as nfp

# When module_cfg's nf_fix_paramparam is set , it means scale=-1, bitwidth=2, method=FIX_AUTO, see the default config in conftest module_cfg fixture.
@pytest.mark.parametrize(
    "module_cfg, case",
    [
        (
            {"input_num": 3},
            {
                "inputs": [1, 1, 0],
                "data": [0.2513, -0.52, 0],
                "out_scale": 1,
                "result": 0,
                "output": [0.5, -0.5, 0],  # quantized parameters, step 0.5
            },
        ),
        (
            {"input_num": 3},
            {
                "inputs": [1, 1, 0],
                "data": [0.2513, -0.5, 0],
                "out_scale": 0.5,
                "result": -0.25,
                "output": [0.25, -0.5, 0],  # quantized parameters, step 0.25
            },
        ),
    ],
    indirect=["module_cfg"],
)
def test_fix_forward_auto(module_cfg, case):
    module, cfg, _ = module_cfg
    if "data" in case:
        module.param[0, :] = torch.tensor(case["data"])
    with torch.no_grad():
        res = module.forward(torch.tensor(case["inputs"]).float())
        assert np.isclose(res, case["result"])  # calc output
        assert np.isclose(module.param, case["output"]).all()  # quantized parameter
        assert cfg["param"]["scale"] == case["out_scale"]  # scale

@pytest.mark.parametrize(
    "module_cfg, case",
    [
        (
            {"input_num": 3},
            {
                "inputs": [[1, 1, 0], [1, 2, 0]],
                "data": [0.2513, -0.52, 0],
                "out_scale": 1,
                "result": [[0], [-0.5]],
                "output": [0.5, -0.5, 0],  # quantized parameters, step 0.5
            },
        ),
        (
            {"input_num": 3},
            {
                "inputs": [[1, 1, 0], [1, 1, 0]],
                "data": [0.2513, -0.52, 0],
                "out_scale": 1,
                "result": [[0], [0]],
                "output": [0.5, -0.5, 0],  # quantized parameters, step 0.5
            },
        ),
        (
            {"input_num": 3},
            {
                "inputs": [[1, 1, 0], [1, 1, 0]],
                "data": [0.2513, -0.5, 0],
                "out_scale": 0.5,
                "result": [[-0.25], [-0.25]],
                "output": [0.25, -0.5, 0],  # quantized parameters, step 0.25
            },
        ),
    ],
    indirect=["module_cfg"],
)
def test_fix_forward_parallel_gpu(module_cfg, case):
    module, cfg, _ = module_cfg
    if "data" in case:
        module.param.data[0, :] = torch.tensor(case["data"])
    model = nn.DataParallel(module.cuda(), [0, 1])
    with torch.no_grad():
        res = model(torch.tensor(case["inputs"]).float().cuda())
        assert cfg["param"]["scale"] == case["out_scale"]  # scale
        assert np.isclose(res.cpu(), case["result"]).all()  # calc output
        # assert np.isclose(module.param.cpu(), case["output"]).all()  # quantized parameter
        # this will not change,
        # but the gradient will still be accumulated in module_parameters[name].grad

@pytest.mark.parametrize(
    "module_cfg, case",
    [
        (
            {"input_num": 3, "grad_cfg": {"method": nfp.FIX_AUTO}},
            {
                "inputs": [0.52, -0.27, 0],
                "data": [0, 0, 0],
                "grad_scale": 1,
                "output": [0.5, -0.5, 0],
            },
        ),
        (
            {"input_num": 3, "grad_cfg": {"method": nfp.FIX_AUTO}},
            {
                "inputs": [0.5, -0.27, 0],
                "data": [0, 0, 0],
                "grad_scale": 0.5,
                "output": [0.5, -0.25, 0],  # quantized gradients
            },
        ),
    ],
    indirect=["module_cfg"],
)
def test_fix_backward_auto(module_cfg, case):
    module, _, cfg = module_cfg
    if "data" in case:
        module.param.data[0, :] = torch.tensor(case["data"])
    res = module.forward(torch.tensor(case["inputs"]).float())
    res.backward()
    assert np.isclose(
        module._parameters["param"].grad, case["output"]
    ).all()  # quantized gradient
    assert cfg["param"]["scale"] == case["grad_scale"]  # scale

@pytest.mark.parametrize(
    "module_cfg, case",
    [
        (
            {"input_num": 3, "data_cfg": {"method": nfp.FIX_NONE},
             "grad_cfg": {"method": nfp.FIX_AUTO}},
            {
                "inputs": [[0.52, -0.27, 0], [0.52, -0.27, 0]],
                "data": [0, 0, 0],
                "grad_scale": 1,
                "output": [0.5, -0.5, 0],
            },
        ),
        (
            {"input_num": 3, "grad_cfg": {"method": nfp.FIX_AUTO}},
            {
                "inputs": [[0.5, -0.27, 0], [0.5, -0.27, 0]],
                "data": [0, 0, 0],
                "grad_scale": 0.5,
                "output": [0.5, -0.25, 0],  # quantized gradients
            },
        ),
    ],
    indirect=["module_cfg"],
)
def test_fix_backward_parallel_gpu(module_cfg, case):
    module, _, cfg = module_cfg
    if "data" in case:
        module.param.data[0, :] = torch.tensor(case["data"])
    model = nn.DataParallel(module.cuda(), [0, 1])
    res = torch.sum(model(torch.tensor(case["inputs"]).float().cuda()))
    res.backward()
    assert np.isclose(
        module._parameters["param"].grad.cpu(), 2 * np.array(case["output"])
    ).all()  # quantized gradient, 2 batch, grad x 2
    assert cfg["param"]["scale"] == case["grad_scale"]  # scale

@pytest.mark.parametrize(
    "module_cfg, case",
    [
        (
            {"input_num": 3, "grad_cfg": {"method": nfp.FIX_AUTO}},
            {
                "inputs": [0.52, -0.27, 0],
                "data": [0, 0, 0],
                "grad_scale": 1,
                "output": [0.5, -0.5, 0],
            },
        ),
        (
            {"input_num": 3, "grad_cfg": {"method": nfp.FIX_AUTO}},
            {
                "inputs": [0.5, -0.27, 0],
                "data": [0, 0, 0],
                "grad_scale": 0.5,
                "output": [0.5, -0.25, 0],  # quantized gradients
            },
        ),
    ],
    indirect=["module_cfg"],
)
def test_fix_update_auto(module_cfg, case):
    module, _, cfg = module_cfg
    if "data" in case:
        module.param.data[0, :] = torch.tensor(case["data"])
    optimizer = optim.SGD(module.parameters(), lr=1.0, momentum=0)
    res = module.forward(torch.tensor(case["inputs"]).float())
    res.backward()
    optimizer.step()
    assert np.isclose(
        -module._parameters["param"].detach(), case["output"]
    ).all()  # updated parameter should be - lr * gradient
    assert cfg["param"]["scale"] == case["grad_scale"]  # scale

def test_ConvBN_fix():
    from nics_fix_pt.nn_fix import ConvBN_fix
    # float forward and combine forward get the same results
    module = ConvBN_fix(3, 32, nf_fix_params={}).cuda()
    module.train()
    data = torch.tensor(np.random.rand(128, 3, 32, 32).astype(np.float32)).cuda()
    comb_out = module(data)
    float_out = module.bn(module.conv(data))
    assert (float_out - comb_out < 1e-3).all()

    module.eval()
    module = ConvBN_fix(3, 32, nf_fix_params={}).cuda()
    data = torch.tensor(np.random.rand(128, 3, 32, 32).astype(np.float32)).cuda()
    comb_out = module(data)
    float_out = module.bn(module.conv(data))
    assert (float_out - comb_out < 1e-3).all()
    
