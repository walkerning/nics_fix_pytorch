
from __future__ import print_function

import math

import nics_fix_pt as nfp
import nics_fix_pt.nn_fix as nnf

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


__all__ = [
    'VGG_ugly', 'vgg11_ugly', 'VGG_elegant', 'vgg11_elegant',
]

BITWIDTH = 8
NEW_BITWIDTH = 6

def _generate_default_fix_cfg(names, scale=0, bitwidth=8, method=0):
    return {n: {
        "method": torch.autograd.Variable(torch.IntTensor(np.array([method])), requires_grad=False),
        "scale": torch.autograd.Variable(torch.IntTensor(np.array([scale])), requires_grad=False),
        "bitwidth": torch.autograd.Variable(torch.IntTensor(np.array([bitwidth])), requires_grad=False)
    } for n in names}


#################################### ugly version ####################################

class VGG_ugly(nnf.FixTopModule):
    def __init__(self):
        super(VGG_ugly, self).__init__()
        # initialize some fix configurations
        self.conv1_fix_params = _generate_default_fix_cfg(["weight", "bias"], method=1, bitwidth=6)#BITWIDTH)
        self.conv2_fix_params = _generate_default_fix_cfg(["weight", "bias"], method=1, bitwidth=6)#BITWIDTH)
        self.conv3_1_fix_params = _generate_default_fix_cfg(["weight", "bias"], method=1, bitwidth=6)#BITWIDTH)
        self.conv3_2_fix_params = _generate_default_fix_cfg(["weight", "bias"], method=1, bitwidth=6)#BITWIDTH)
        self.conv4_1_fix_params = _generate_default_fix_cfg(["weight", "bias"], method=1, bitwidth=6)#BITWIDTH)
        self.conv4_2_fix_params = _generate_default_fix_cfg(["weight", "bias"], method=1, bitwidth=6)#BITWIDTH)
        self.conv5_1_fix_params = _generate_default_fix_cfg(["weight", "bias"], method=1, bitwidth=6)#BITWIDTH)
        self.conv5_2_fix_params = _generate_default_fix_cfg(["weight", "bias"], method=1, bitwidth=6)#BITWIDTH)
        self.fc1_fix_params = _generate_default_fix_cfg(["weight", "bias"], method=1, bitwidth=BITWIDTH)
        self.fc2_fix_params = _generate_default_fix_cfg(["weight", "bias"], method=1, bitwidth=BITWIDTH)
        self.fc3_fix_params = _generate_default_fix_cfg(["weight", "bias"], method=1, bitwidth=BITWIDTH)
        self.fix_params = [_generate_default_fix_cfg(["activation"], method=1, bitwidth=BITWIDTH) for _ in range(12)]
        # initialize modules
        kwargs = {'kernel_size': 3, 'padding': 1}
        self.conv1 = nnf.Conv2d_fix(3, 64,  nf_fix_params=self.conv1_fix_params, **kwargs)
        self.conv2 = nnf.Conv2d_fix(64, 128, nf_fix_params=self.conv2_fix_params, **kwargs)
        self.conv3_1 = nnf.Conv2d_fix(128, 256, nf_fix_params=self.conv3_1_fix_params, **kwargs)
        self.conv3_2 = nnf.Conv2d_fix(256, 256, nf_fix_params=self.conv3_2_fix_params, **kwargs)
        self.conv4_1 = nnf.Conv2d_fix(256, 512, nf_fix_params=self.conv4_1_fix_params, **kwargs)
        self.conv4_2 = nnf.Conv2d_fix(512, 512, nf_fix_params=self.conv4_2_fix_params, **kwargs)
        self.conv5_1 = nnf.Conv2d_fix(512, 512, nf_fix_params=self.conv5_1_fix_params, **kwargs)
        self.conv5_2 = nnf.Conv2d_fix(512, 512, nf_fix_params=self.conv5_2_fix_params, **kwargs)
        self.fc1 = nnf.Linear_fix(512, 512, nf_fix_params=self.fc1_fix_params)
        self.fc2 = nnf.Linear_fix(512, 512, nf_fix_params=self.fc2_fix_params)
        self.fc3 = nnf.Linear_fix(512, 10, nf_fix_params=self.fc3_fix_params)
        self.fix0 = nnf.Activation_fix(nf_fix_params=self.fix_params[0])
        self.fix1 = nnf.Activation_fix(nf_fix_params=self.fix_params[1])
        self.fix2 = nnf.Activation_fix(nf_fix_params=self.fix_params[2])
        self.fix3 = nnf.Activation_fix(nf_fix_params=self.fix_params[3])
        self.fix4 = nnf.Activation_fix(nf_fix_params=self.fix_params[4])
        self.fix5 = nnf.Activation_fix(nf_fix_params=self.fix_params[5])
        self.fix6 = nnf.Activation_fix(nf_fix_params=self.fix_params[6])
        self.fix7 = nnf.Activation_fix(nf_fix_params=self.fix_params[7])
        self.fix8 = nnf.Activation_fix(nf_fix_params=self.fix_params[8])
        self.fix9 = nnf.Activation_fix(nf_fix_params=self.fix_params[9])
        self.fix10 = nnf.Activation_fix(nf_fix_params=self.fix_params[10])
        self.fix11 = nnf.Activation_fix(nf_fix_params=self.fix_params[11])

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nnf.Conv2d_fix):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        kwargs = {'kernel_size': 2, 'stride': 2}
        x = self.fix0(x)
        x = F.relu(self.fix1(self.conv1(x)))
        x = F.relu(self.fix2(self.conv2(F.max_pool2d(x, **kwargs))))
        x = F.relu(self.fix3(self.conv3_1(F.max_pool2d(x, **kwargs))))
        x = F.relu(self.fix4(self.conv3_2(x)))
        x = F.relu(self.fix5(self.conv4_1(F.max_pool2d(x, **kwargs))))
        x = F.relu(self.fix6(self.conv4_2(x)))
        x = F.relu(self.fix7(self.conv5_1(F.max_pool2d(x, **kwargs))))
        x = F.relu(self.fix8(self.conv5_2(x)))
        x = F.relu(self.fix9(self.fc1(F.dropout(F.max_pool2d(x, **kwargs).view(-1, 512)))))
        x = F.relu(self.fix10(self.fc2(F.dropout(x))))
        x = self.fix11(self.fc3(x))
        # return F.log_softmax(x, dim=-1)
        return x

def vgg11_ugly():
    return VGG_ugly()


#################################### elegant version ####################################

class VGG_elegant(nnf.FixTopModule):
    def __init__(self, features, classifier=None):
        super(VGG_elegant, self).__init__()

        self.features = features
        self.classifier = classifier
        if self.classifier is None:
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nnf.Linear_fix(512, 512, nf_fix_params=_generate_default_fix_cfg(["weight", "bias"], method=1, bitwidth=BITWIDTH)),
                nnf.Activation_fix(nf_fix_params=_generate_default_fix_cfg(["activation"], method=1, bitwidth=BITWIDTH)),
                nn.ReLU(True),
                nn.Dropout(),
                nnf.Linear_fix(512, 512, nf_fix_params=_generate_default_fix_cfg(["weight", "bias"], method=1, bitwidth=BITWIDTH)),
                nnf.Activation_fix(nf_fix_params=_generate_default_fix_cfg(["activation"], method=1, bitwidth=BITWIDTH)),
                nn.ReLU(True),
                nnf.Linear_fix(512, 10, nf_fix_params=_generate_default_fix_cfg(["weight", "bias"], method=1, bitwidth=BITWIDTH)),
                nnf.Activation_fix(nf_fix_params=_generate_default_fix_cfg(["activation"], method=1, bitwidth=BITWIDTH)),
            )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nnf.Conv2d_fix):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
    def forward(self, x):
        activation = nnf.Activation_fix(nf_fix_params=_generate_default_fix_cfg(["activation"], method=1, bitwidth=BITWIDTH))
        x = activation(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v[0] == 'maxpooling':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v[0] == 'dropout':
            layers += [nn.Dropout()]
        elif 'fc' in v[0]:
            fc =  nnf.Linear_fix(v[1], v[2], 
                nf_fix_params=_generate_default_fix_cfg(["weight", "bias"], method=v[3], bitwidth=v[4]))
            activation = nnf.Activation_fix(
                nf_fix_params=_generate_default_fix_cfg(["activation"], method=v[5], bitwidth=v[6]))
            layers += [fc, activation, nn.ReLU(inplace=True)]
        elif 'conv' in v[0]:
            conv2d = nnf.Conv2d_fix(in_channels, v[1], kernel_size=3, padding=1, 
                nf_fix_params=_generate_default_fix_cfg(["weight", "bias"], method=v[2], bitwidth=v[3]))
            activation = nnf.Activation_fix(
                nf_fix_params=_generate_default_fix_cfg(["activation"], method=v[4], bitwidth=v[5]))

            if batch_norm:
                layers += [conv2d, activation, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, activation, nn.ReLU(inplace=True)]
            in_channels = v[1]

    print(layers)
    return nn.Sequential(*layers)

features_cfg = [
    ['conv1', 64, 1, NEW_BITWIDTH, 1, BITWIDTH],
    ['maxpooling'],
    ['conv2', 128, 1, NEW_BITWIDTH, 1, BITWIDTH],
    ['maxpooling'],
    ['conv3_1', 256, 1, NEW_BITWIDTH, 1, BITWIDTH],
    ['conv3_2', 256, 1, NEW_BITWIDTH, 1, BITWIDTH],
    ['maxpooling'],
    ['conv4_1', 512, 1, NEW_BITWIDTH, 1, BITWIDTH],
    ['conv4_2', 512, 1, NEW_BITWIDTH, 1, BITWIDTH],
    ['maxpooling'],
    ['conv5_1', 512, 1, NEW_BITWIDTH, 1, BITWIDTH],
    ['conv5_2', 512, 1, NEW_BITWIDTH, 1, BITWIDTH],
    ['maxpooling']
]

classifier_cfg = [
    ['dropout'],
    ['fc1', 512, 512, 1, BITWIDTH, 1, BITWIDTH],
    ['dropout'],
    ['fc2', 512, 512, 1, BITWIDTH, 1, BITWIDTH],
    ['fc3', 512, 10, 1, BITWIDTH, 1, BITWIDTH]
]

def vgg11_elegant():
    return VGG_elegant(make_layers(features_cfg))

