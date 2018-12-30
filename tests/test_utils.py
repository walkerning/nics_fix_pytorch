import pytest

def test_auto_register():
    import torch
    from nics_fix_pt import nn_fix as nnf
    from nics_fix_pt import NAR as nnf_auto
    torch.nn.Linear_another2 = torch.nn.Linear
    class Linear_another(torch.nn.Linear): # a stub test case
        pass
    torch.nn.Linear_another = Linear_another
    with pytest.raises(AttributeError) as _:
        fix_module = nnf.Linear_another_fix(3, 1, nf_fix_params={}) # not already registered
    fix_module = nnf_auto.Linear_another(3, 1, nf_fix_params={}) # will automatically register this fix module; `NAR.linear_another_fix` also works.
    fix_module = nnf_auto.Linear_another2_fix(3, 1, nf_fix_params={}) # will automatically register this fix module; `NAR.linear_another2` also works.
    fix_module = nnf.Linear_another_fix(3, 1, nf_fix_params={})
    fix_module = nnf.Linear_another2_fix(3, 1, nf_fix_params={})

