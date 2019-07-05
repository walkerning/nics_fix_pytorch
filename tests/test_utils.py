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
