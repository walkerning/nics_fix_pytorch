import pytest
import numpy as np
import torch

import nics_fix_pt as nfp
import nics_fix_pt.quant as nfpq


@pytest.mark.parametrize(
    "case",
    [
        {
            "data": [0.2513, -0.5, 0],
            "scale": 0.5,
            "bitwidth": 2,
            "method": nfp.FIX_FIXED,
            "output": ([0.25, -0.5, 0], 0.25),
        },
        {
            "data": [0.2513, -0.5, 0],
            "scale": 0.5,
            "bitwidth": 16,
            "method": nfp.FIX_FIXED,
            "output": (
                [np.round(0.2513 / (2 ** (-16))) * 2 ** (-16), -0.5, 0],
                2 ** (-16),
            ),
        },
        {
            "data": [0.2513, -0.5, 0],
            "scale": 0.5,
            "bitwidth": 2,
            "method": nfp.FIX_AUTO,
            "output": ([0.25, -0.5, 0], 0.25),
        },
        {
            "data": [0.2513, -0.52, 0],
            "scale": 0.5,
            "bitwidth": 2,
            "method": nfp.FIX_AUTO,
            "out_scale": 1,
            "output": ([0.5, -0.5, 0], 0.5),
        },
        {
            "data": [0.2513, -0.52, 0],
            "scale": 0.5,
            "bitwidth": 4,
            "method": nfp.FIX_AUTO,
            "range_method": nfp.RANGE_3SIGMA,
            "output": ([0.25, -0.5, 0], 0.25),
        },
        # test the float scale
        {
            "data": [0.2513, -0.52, 0.0],
            "scale": 0.52,
            "bitwidth": 4,
            "method": nfp.FIX_AUTO,
            "output": ([0.26, -0.52, 0.0], 0.065),
            "float_scale": True,
        },
        {
            "data": [0.2513, -0.52, 0.0],
            "scale": 0.5,
            "bitwidth": 4,
            "method": nfp.FIX_AUTO,
            "output": ([(0.2513 + 0.52) * 5 / 16, -0.52, 0.0], (0.2513 + 0.52) / 16),
            "float_scale": True,
            "zero_point": True,
        },
        {
            "data": [[[[0.2513]], [[-0.52]]], [[[0.3321]], [[-0.4532]]]],
            "scale": [
                0.2513,
                0.3321,
            ],  # max_data = data.view(data.shape[0],-1).max(dim=1)[0]
            "bitwidth": 4,
            "method": nfp.FIX_AUTO,
            "output": (
                [[[[4 * 0.52 / 8]], [[-0.52]]], [[[6 * 0.4532 / 8]], [[-0.4532]]]],
                [0.52 / 8, 0.4532 / 8],
            ),
            "float_scale": True,
            "group": "batch",
        },
    ],
)
def test_quantize_cfg(case):
    scale_tensor = torch.tensor([case["scale"]])
    out = nfpq.quantize_cfg(
        torch.tensor(case["data"]),
        scale_tensor,
        torch.tensor(case["bitwidth"]),
        case["method"],
        case["range_method"] if "range_method" in case.keys() else nfp.RANGE_MAX,
        stochastic=case["stochastic"] if "stochastic" in case.keys() else False,
        float_scale=case["float_scale"] if "float_scale" in case.keys() else False,
        zero_point=case["zero_point"] if "zero_point" in case.keys() else False,
        group=case["group"] if "group" in case.keys() else False,
    )
    assert np.isclose(out[0], case["output"][0]).all()
    assert np.isclose(out[1].view(-1), case["output"][1]).all()
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

    # Stochastic rounding
    inputs = torch.autograd.Variable(torch.Tensor(100).fill_(0.5), requires_grad=True)
    outputs = nfpq.StraightThroughStochasticRound().apply(inputs)
    assert outputs.max() > 0.9 and outputs.min() < 0.1


def test_quantize_gradient():
    quant_grad = nfpq.QuantitizeGradient()
    scale = torch.Tensor([0])
    inputs = torch.autograd.Variable(torch.tensor([1.1, 0.9]), requires_grad=True)
    quanted = quant_grad.apply(inputs, scale, torch.tensor(2), nfp.FIX_AUTO)
    output = (
        quanted * torch.autograd.Variable(torch.tensor([0.5, 0.26]), requires_grad=True)
    ).sum()
    output.backward()
    assert np.isclose(inputs._grad, [0.5, 0.25]).all()
    assert scale.item() == 0.5
