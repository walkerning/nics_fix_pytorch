import pytest
import numpy as np
import torch

import nics_fix_pt as nfp

@pytest.mark.parametrize("case",
                         [{
                             "data": [0.2513, -0.5, 0],
                             "scale": -1,
                             "bitwidth": 2,
                             "method": nfp.FIX_FIXED,
                             "output": ([0.25, -0.5, 0], 0.25)
                         }, {
                             "data": [0.2513, -0.5, 0],
                             "scale": -1,
                             "bitwidth": 16,
                             "method": nfp.FIX_FIXED,
                             "output": ([np.round(0.2513/(2**(-16))) * 2**(-16), -0.5, 0], 2**(-16))
                         }, {
                             "data": [0.2513, -0.5, 0],
                             "scale": -1,
                             "bitwidth": 2,
                             "method": nfp.FIX_AUTO,
                             "output": ([0.25, -0.5, 0], 0.25)
                         }, {
                             "data": [0.2513, -0.52, 0],
                             "scale": -1,
                             "bitwidth": 2,
                             "method": nfp.FIX_AUTO,
                             "out_scale": 0,
                             "output": ([0.5, -0.5, 0], 0.5)
                         }])
def test_quantitize_cfg(case):
    scale_tensor = torch.tensor([case["scale"]])
    out = nfp.quant.quantitize_cfg(torch.tensor(case["data"]), scale_tensor,
                                   torch.tensor(case["bitwidth"]), case["method"])
    assert np.isclose(out[0], case["output"][0]).all()
    assert np.isclose(out[1], case["output"][1])
    if "out_scale" in case:
        assert bool(scale_tensor == case["out_scale"])
