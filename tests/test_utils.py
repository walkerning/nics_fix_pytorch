import pytest


def test_auto_register():
    import torch
    from nics_fix_pt import nn_fix as nnf
    from nics_fix_pt import NAR as nnf_auto

    torch.nn.Linear_another2 = torch.nn.Linear

    class Linear_another(torch.nn.Linear):  # a stub test case
        pass

    torch.nn.Linear_another = Linear_another
    with pytest.raises(AttributeError) as _:
        fix_module = nnf.Linear_another_fix(
            3, 1, nf_fix_params={}
        )  # not already registered
    fix_module = nnf_auto.Linear_another(
        3, 1, nf_fix_params={}
    )  # will automatically register this fix module; `NAR.linear_another_fix` also works.
    fix_module = nnf_auto.Linear_another2_fix(
        3, 1, nf_fix_params={}
    )  # will automatically register this fix module; `NAR.linear_another2` also works.
    fix_module = nnf.Linear_another_fix(3, 1, nf_fix_params={})
    fix_module = nnf.Linear_another2_fix(3, 1, nf_fix_params={})


def test_save_state_dict(tmp_path):
    import os
    import numpy as np
    import torch
    from torch import nn

    import nics_fix_pt as nfp
    from nics_fix_pt import nn_fix as nnf
    from nics_fix_pt.utils import _generate_default_fix_cfg

    class _View(nn.Module):
        def __init__(self):
            super(_View, self).__init__()

        def forward(self, inputs):
            return inputs.view(inputs.shape[0], -1)
    data = torch.tensor(np.random.rand(8, 3, 4, 4).astype(np.float32)).cuda()
    ckpt = os.path.join(tmp_path, "tmp.pt")

    model = nn.Sequential(*[
        nnf.Conv2d_fix(3, 10, kernel_size=3, padding=1,
                       nf_fix_params=_generate_default_fix_cfg(
                           ["weight", "bias"], scale=np.random.randint(low=-10, high=10),
                           method=nfp.FIX_FIXED)),
        nnf.Conv2d_fix(10, 20, kernel_size=3, padding=1,
                       nf_fix_params=_generate_default_fix_cfg(
                           ["weight", "bias"], scale=np.random.randint(low=-10, high=10),
                           method=nfp.FIX_FIXED)),
        nnf.Activation_fix(
            nf_fix_params=_generate_default_fix_cfg(
                ["activation"], scale=np.random.randint(low=-10, high=10),
                method=nfp.FIX_FIXED)),
        nn.AdaptiveAvgPool2d(1),
        _View(),
        nnf.Linear_fix(20, 10, nf_fix_params=_generate_default_fix_cfg(
            ["weight", "bias"], scale=np.random.randint(low=-10, high=10),
            method=nfp.FIX_FIXED))
    ])
    model.cuda()
    pre_results = model(data)
    torch.save(model.state_dict(), ckpt)
    model2 = nn.Sequential(*[
        nnf.Conv2d_fix(3, 10, kernel_size=3, padding=1,
                       nf_fix_params=_generate_default_fix_cfg(
                           ["weight", "bias"], scale=np.random.randint(low=-10, high=10),
                           method=nfp.FIX_FIXED)),
        nnf.Conv2d_fix(10, 20, kernel_size=3, padding=1,
                       nf_fix_params=_generate_default_fix_cfg(
                           ["weight", "bias"], scale=np.random.randint(low=-10, high=10),
                           method=nfp.FIX_FIXED)),
        nnf.Activation_fix(
            nf_fix_params=_generate_default_fix_cfg(
                ["activation"], scale=np.random.randint(low=-10, high=10),
                method=nfp.FIX_FIXED)),
        nn.AdaptiveAvgPool2d(1),
        _View(),
        nnf.Linear_fix(20, 10, nf_fix_params=_generate_default_fix_cfg(
            ["weight", "bias"], scale=np.random.randint(low=-10, high=10),
            method=nfp.FIX_FIXED))
    ])
    model2.cuda()
    model2.load_state_dict(torch.load(ckpt))
    post_results = model2(data)
    assert (post_results - pre_results < 1e-3).all()


def test_fix_state_dict(module_cfg):
    import torch
    from nics_fix_pt.nn_fix import FixTopModule
    import nics_fix_pt.quant as nfpq

    module, cfg, _ = module_cfg
    dct = FixTopModule.fix_state_dict(module)
    assert (dct["param"] == module._parameters["param"]).all()  # not already fixed

    # forward the module once
    res = module.forward(torch.tensor([0, 0, 0]).float())
    dct = FixTopModule.fix_state_dict(module)
    dct_vars = FixTopModule.fix_state_dict(module, keep_vars=True)
    quantized, _ = nfpq.quantitize_cfg(module._parameters["param"], **cfg["param"])
    dct_vars = FixTopModule.fix_state_dict(module, keep_vars=True)
    assert (dct["param"] == quantized).all()  # already fixed
    assert (dct_vars["param"] == quantized).all()  # already fixed
    assert (
        dct_vars["param"].nfp_actual_data == module._parameters["param"]
    ).all()  # underhood float-point data


def test_set_fix_method(test_network):
    test_network.set_fix_method(method=0, method_by_type={
        "BatchNorm2d_fix": {"running_mean": 2, "running_var": 2, "weight": 1, "bias": 0}
    }, method_by_name={"conv1": {"weight": 2}})
    assert int(test_network.bn1.nf_fix_params["weight"]["method"]) == 1
    assert int(test_network.bn1.nf_fix_params["bias"]["method"]) == 0
    assert int(test_network.bn1.nf_fix_params["running_mean"]["method"]) == 2
    assert int(test_network.conv1.nf_fix_params["weight"]["method"]) == 2
    assert int(test_network.conv1.nf_fix_params["bias"]["method"]) == 1
    assert int(test_network.conv2.nf_fix_params["weight"]["method"]) == 0
    assert int(test_network.conv2.nf_fix_params["bias"]["method"]) == 0
