import pytest
import numpy as np
import torch

import nics_fix_pt.quant as nfpq

@pytest.mark.parametrize("case",
                         [{
                             "data": [0.2513, -0.5, 0],
                             "scale": -1,
                             "bitwidth": 2,
                             "method": nfpq.FIX_FIXED,
                             "output": ([0.25, -0.5, 0], 0.25)
                         }, {
                             "data": [0.2513, -0.5, 0],
                             "scale": -1,
                             "bitwidth": 16,
                             "method": nfpq.FIX_FIXED,
                             "output": ([np.round(0.2513/(2**(-16))) * 2**(-16), -0.5, 0], 2**(-16))
                         }, {
                             "data": [0.2513, -0.5, 0],
                             "scale": -1,
                             "bitwidth": 2,
                             "method": nfpq.FIX_AUTO,
                             "output": ([0.25, -0.5, 0], 0.25)
                         }, {
                             "data": [0.2513, -0.52, 0],
                             "scale": -1,
                             "bitwidth": 2,
                             "method": nfpq.FIX_AUTO,
                             "out_scale": 0,
                             "output": ([0.5, -0.5, 0], 0.5)
                         }])
def test_quantitize_cfg(case):
    scale_tensor = torch.tensor([case["scale"]])
    out = nfpq.quantitize_cfg(torch.tensor(case["data"]), scale_tensor,
                              torch.tensor(case["bitwidth"]), case["method"])
    assert np.isclose(out[0], case["output"][0]).all()
    assert np.isclose(out[1], case["output"][1])
    if "out_scale" in case:
        assert bool(scale_tensor == case["out_scale"])

def test_straight_through_gradient():
    inputs = torch.autograd.Variable(torch.tensor([1.1, 0.9]), requires_grad=True)
    outputs = nfpq.StraightThroughRound().apply(inputs)
    outputs.sum().backward()
    assert np.isclose(inputs._grad, [1, 1]).all()

    # when Round is applied without straight through, there is no gradient
    inputs.grad.detach_()
    inputs.grad.zero_()
    output_nost = inputs.round()
    assert np.isclose(inputs._grad, [0, 0]).all()

def test_quantitize_gradient():
    quant_grad = nfpq.QuantitizeGradient()
    scale = torch.Tensor([0])
    inputs = torch.autograd.Variable(torch.tensor([1.1, 0.9]), requires_grad=True)
    quanted = quant_grad.apply(inputs, scale, torch.tensor(2), nfpq.FIX_AUTO)
    output = (quanted * torch.autograd.Variable(torch.tensor([0.5, 0.26]), requires_grad=True)).sum()
    output.backward()
    assert np.isclose(inputs._grad, [0.5, 0.25]).all()
    assert scale.item() == -1
