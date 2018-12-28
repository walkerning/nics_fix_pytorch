import pytest
import numpy as np
import torch

# When module_cfg's nf_fix_paramparam is set , it means scale=-1, bitwidth=2, method=FIX_AUTO, see the default config in conftest module_cfg fixture.
@pytest.mark.parametrize("module_cfg, case",
                         [({"input_num": 3}, {
                             "inputs": [1,1,0],
                             "data": [0.2513, -0.52, 0],
                             "out_scale": 0,
                             "result": 0,
                             "output": ([0.5, -0.5, 0], 0.5) # quantitized parameters, step
                         }), ({"input_num": 3}, {
                             "inputs": [1,1,0],
                             "data": [0.2513, -0.5, 0],
                             "out_scale": -1,
                             "result": -0.25,
                             "output": ([0.25, -0.5, 0], 0.25) # quantitized parameters, step
                         })],
                         indirect=["module_cfg"])
def test_fix_forward_auto(module_cfg, case):
    module, cfg = module_cfg
    if "data" in case:
        module.param[0, :] = torch.tensor(case["data"])
    with torch.no_grad():
        res = module.forward(torch.tensor(case["inputs"]).float())
        assert np.isclose(res, case["result"]) # calc output
        assert np.isclose(module.param, case["output"][0]).all() # quantitized parameter
        assert bool(module.step == case["output"][1]) # step
        assert cfg["param"]["scale"] == case["out_scale"] # scale
