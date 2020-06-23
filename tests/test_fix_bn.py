import pytest

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import nics_fix_pt as nfp
from nics_fix_pt import nn_fix as nnf


@pytest.mark.parametrize(
    "case", [{"input_num": 3, "momentum": 0.5, "inputs": [[1, 1, 0], [2, 1, 2]]}]
)
def test_fix_bn_test_auto(case):
    # TEST: the first update is the same (not quantized)
    bn_fix = nnf.BatchNorm1d_fix(
        case["input_num"],
        nf_fix_params={
            "running_mean": {
                "method": nfp.FIX_AUTO,
                "bitwidth": torch.tensor([2]),
                "scale": torch.tensor([0]),
            },
            "running_var": {
                "method": nfp.FIX_AUTO,
                "bitwidth": torch.tensor([2]),
                "scale": torch.tensor([0]),
            },
        },
        affine=False,
        momentum=case["momentum"],
    )
    bn = nn.BatchNorm1d(case["input_num"], affine=False, momentum=case["momentum"])
    bn_fix.train()
    bn.train()
    inputs = torch.autograd.Variable(
        torch.tensor(case["inputs"]).float(), requires_grad=True
    )
    out_fix = bn_fix(inputs)
    out = bn(inputs)
    assert (bn.running_mean == bn_fix.running_mean).all()  # not quantized here
    assert (bn.running_var == bn_fix.running_var).all()
    assert (out == out_fix).all()

    # TEST: Quantitized on the next forward
    bn_fix.train(False)
    bn.train(False)
    out_fix = bn_fix(inputs)
    # Let's explicit quantize the mean/var of the normal BN model for comparison
    object.__setattr__(
        bn,
        "running_mean",
        nfp.quant.quantize_cfg(
            bn.running_mean,
            **{
                "method": nfp.FIX_AUTO,
                "bitwidth": torch.tensor([2]),
                "scale": torch.tensor([0]),
            }
        )[0],
    )
    object.__setattr__(
        bn,
        "running_var",
        nfp.quant.quantize_cfg(
            bn.running_var,
            **{
                "method": nfp.FIX_AUTO,
                "bitwidth": torch.tensor([2]),
                "scale": torch.tensor([0]),
            }
        )[0],
    )
    assert (bn.running_mean == bn_fix.running_mean).all()
    assert (bn.running_var == bn_fix.running_var).all()

    out = bn(inputs)
    assert (out == out_fix).all()

    # TEST: the running mean/var update is on the quantized running mean
    bn_fix.train()
    bn.train()
    out_fix = bn_fix(inputs)
    out = bn(inputs)
    assert (
        bn.running_mean == bn_fix.running_mean
    ).all()  # quantized on the next forward
    assert (bn.running_var == bn_fix.running_var).all()

    # runnig_mean_should = np.mean(inputs.detach().numpy(), axis=0) * case["momentum"]
    # runnig_var_should = np.var(inputs.detach().numpy(), axis=0) * case["momentum"] + np.ones(case["input_num"]) * (1 - case["momentum"])
