# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import functional as F

import nics_fix_pt as nfp
import nics_fix_pt.nn_fix as nnf
import numpy as np

def _generate_default_fix_cfg(names, scale=0, bitwidth=8, method=0,
                              range_method=nfp.RangeMethod.RANGE_MAX):
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
            "range_method": range_method
        }
        for n in names
    }

class FixNet(nnf.FixTopModule):
    def __init__(self, fix_bn=True, fix_grad=True, bitwidth_data=8, bitwidth_grad=16,
                 range_method=nfp.RangeMethod.RANGE_MAX,
                 grad_range_method=nfp.RangeMethod.RANGE_MAX):
        super(FixNet, self).__init__()

        print("fix bn: {}; fix grad: {}; range method: {}; grad range method: {}".format(
            fix_bn, fix_grad, range_method, grad_range_method
        ))

        # fix configurations (data/grad) for parameters/buffers
        self.fix_param_cfgs = {}
        self.fix_grad_cfgs = {}
        layers = [("conv1_1", 128, 3), ("bn1_1",), ("conv1_2", 128, 3), ("bn1_2",),
                  ("conv1_3", 128, 3), ("bn1_3",), ("conv2_1", 256, 3), ("bn2_1",),
                  ("conv2_2", 256, 3), ("bn2_2",), ("conv2_3", 256, 3), ("bn2_3",),
                  ("conv3_1", 512, 3), ("bn3_1",), ("nin3_2", 256, 1), ("bn3_2",),
                  ("nin3_3", 128, 1), ("bn3_3",), ("fc4", 10)]
        for layer_cfg in layers:
            name = layer_cfg[0]
            if "bn" in name and not fix_bn:
                continue
            # data fix config
            self.fix_param_cfgs[name] = _generate_default_fix_cfg(
                ["weight", "bias", "running_mean", "running_var"] \
                if "bn" in name else ["weight", "bias"],
                method=1, bitwidth=bitwidth_data, range_method=range_method
            )
            if fix_grad:
                # grad fix config
                self.fix_grad_cfgs[name] = _generate_default_fix_cfg(
                    ["weight", "bias"], method=1, bitwidth=bitwidth_grad,
                    range_method=grad_range_method
                )

        # fix configurations for activations
        # data fix config
        self.fix_act_cfgs = [
            _generate_default_fix_cfg(["activation"], method=1, bitwidth=bitwidth_data,
                                      range_method=range_method)
            for _ in range(20)
        ]
        if fix_grad:
            # grad fix config
            self.fix_act_grad_cfgs = [
                _generate_default_fix_cfg(["activation"], method=1, bitwidth=bitwidth_grad,
                                          range_method=grad_range_method)
                for _ in range(20)
            ]

        # construct layers
        cin = 3
        for layer_cfg in layers:
            name = layer_cfg[0]
            if "conv" in name or "nin" in name:
                # convolution layers
                cout, kernel_size = layer_cfg[1:]
                layer = nnf.Conv2d_fix(
                    cin, cout,
                    nf_fix_params=self.fix_param_cfgs[name],
                    nf_fix_params_grad=self.fix_grad_cfgs[name] if fix_grad else None,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2 if name != "conv3_1" else 0)
                cin = cout
            elif "bn" in name:
                # bn layers
                if fix_bn:
                    layer = nnf.BatchNorm2d_fix(
                        cin,
                        nf_fix_params=self.fix_param_cfgs[name],
                        nf_fix_params_grad=self.fix_grad_cfgs[name] if fix_grad else None)
                else:
                    layer = nn.BatchNorm2d(cin)
            elif "fc" in name:
                # fully-connected layers
                cout = layer_cfg[1]
                layer = nnf.Linear_fix(
                    cin, cout,
                    nf_fix_params=self.fix_param_cfgs[name],
                    nf_fix_params_grad=self.fix_grad_cfgs[name] if fix_grad else None)
                cin = cout
            # call setattr
            setattr(self, name, layer)

        for i in range(20):
            setattr(self, "fix" + str(i), nnf.Activation_fix(
                nf_fix_params=self.fix_act_cfgs[i],
                nf_fix_params_grad=self.fix_act_grad_cfgs[i] if fix_grad else None))

        self.pool1 = nn.MaxPool2d((2, 2))
        self.pool2 = nn.MaxPool2d((2, 2))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.fix0(x)
        x = self.fix2(F.relu(self.bn1_1(self.fix1(self.conv1_1(x)))))
        x = self.fix4(F.relu(self.bn1_2(self.fix3(self.conv1_2(x)))))
        x = self.pool1(self.fix6(F.relu(self.bn1_3(self.fix5(self.conv1_3(x))))))
        x = self.fix8(F.relu(self.bn2_1(self.fix7(self.conv2_1(x)))))
        x = self.fix10(F.relu(self.bn2_2(self.fix9(self.conv2_2(x)))))
        x = self.pool2(self.fix12(F.relu(self.bn2_3(self.fix11(self.conv2_3(x))))))
        x = self.fix14(F.relu(self.bn3_1(self.fix13(self.conv3_1(x)))))
        x = self.fix16(F.relu(self.bn3_2(self.fix15(self.nin3_2(x)))))
        x = self.fix18(F.relu(self.bn3_3(self.fix17(self.nin3_3(x)))))
        # x = self.fix2(F.relu(self.bn1_1(self.conv1_1(x))))
        # x = self.fix4(F.relu(self.bn1_2(self.conv1_2(x))))
        # x = self.pool1(self.fix6(F.relu(self.bn1_3(self.conv1_3(x)))))
        # x = self.fix8(F.relu(self.bn2_1(self.conv2_1(x))))
        # x = self.fix10(F.relu(self.bn2_2(self.conv2_2(x))))
        # x = self.pool2(self.fix12(F.relu(self.bn2_3(self.conv2_3(x)))))
        # x = self.fix14(F.relu(self.bn3_1(self.conv3_1(x))))
        # x = self.fix16(F.relu(self.bn3_2(self.nin3_2(x))))
        # x = self.fix18(F.relu(self.bn3_3(self.nin3_3(x))))
        x = self.avg_pool(x)
        x = x.view(-1, 128)
        x = self.fix19(self.fc4(x))

        return x
