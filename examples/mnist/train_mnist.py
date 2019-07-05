# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import nics_fix_pt as nfp
import nics_fix_pt.nn_fix as nnf

# Training settings
parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
parser.add_argument(
    "--batch-size",
    type=int,
    default=64,
    metavar="N",
    help="input batch size for training (default: 64)",
)
parser.add_argument(
    "--test-batch-size",
    type=int,
    default=1000,
    metavar="N",
    help="input batch size for testing (default: 1000)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=1,
    metavar="N",
    help="number of epochs to train (default: 1)",
)
parser.add_argument(
    "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.5,
    metavar="M",
    help="SGD momentum (default: 0.5)",
)
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="disables CUDA training"
)
parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=10,
    metavar="N",
    help="how many batches to wait before logging training status",
)
parser.add_argument(
    "--float",
    action="store_true",
    default=False,
    help="use float point training/testing",
)
parser.add_argument("--save", default=None, help="save fixed-point paramters to file")
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {"num_workers": 1, "pin_memory": True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=args.batch_size,
    shuffle=True,
    **kwargs
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../data",
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=args.test_batch_size,
    shuffle=True,
    **kwargs
)


def _generate_default_fix_cfg(names, scale=0, bitwidth=8, method=0):
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
        }
        for n in names
    }


BITWIDTH = 4


class Net(nnf.FixTopModule):
    def __init__(self):
        super(Net, self).__init__()
        # initialize some fix configurations
        self.fc1_fix_params = _generate_default_fix_cfg(
            ["weight", "bias"], method=1, bitwidth=BITWIDTH
        )
        self.bn_fc1_params = _generate_default_fix_cfg(
            ["weight", "bias", "running_mean", "running_var"],
            method=1,
            bitwidth=BITWIDTH,
        )
        self.fc2_fix_params = _generate_default_fix_cfg(
            ["weight", "bias"], method=1, bitwidth=BITWIDTH
        )
        self.fix_params = [
            _generate_default_fix_cfg(["activation"], method=1, bitwidth=BITWIDTH)
            for _ in range(4)
        ]
        # initialize modules
        self.fc1 = nnf.Linear_fix(784, 100, nf_fix_params=self.fc1_fix_params)
        # self.bn_fc1 = nnf.BatchNorm1d_fix(100, nf_fix_params=self.bn_fc1_params)
        self.fc2 = nnf.Linear_fix(100, 10, nf_fix_params=self.fc2_fix_params)
        self.fix0 = nnf.Activation_fix(nf_fix_params=self.fix_params[0])
        # self.fix0_bn = nnf.Activation_fix(nf_fix_params=self.fix_params[1])
        self.fix1 = nnf.Activation_fix(nf_fix_params=self.fix_params[2])
        self.fix2 = nnf.Activation_fix(nf_fix_params=self.fix_params[3])

    def forward(self, x):
        x = self.fix0(x.view(-1, 784))
        x = F.relu(self.fix1(self.fc1(x)))
        # x = F.relu(self.fix0_bn(self.bn_fc1(self.fix1(self.fc1(x)))))
        self.logits = x = self.fix2(self.fc2(x))
        return F.log_softmax(x, dim=-1)


model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


def train(epoch, fix_method=nfp.FIX_AUTO):
    model.set_fix_method(fix_method)
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.data.item(),
                ),
                end="",
            )
    print("")


def test(fix_method=nfp.FIX_FIXED):
    model.set_fix_method(fix_method)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, size_average=False
            ).data.item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[
                1
            ]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


for epoch in range(1, args.epochs + 1):
    train(epoch, nfp.FIX_NONE if args.float else nfp.FIX_AUTO)
    test(nfp.FIX_NONE if args.float else nfp.FIX_FIXED)

model.print_fix_configs()
fix_cfg = {
    "data": model.get_fix_configs(data_only=True),
    "grad": model.get_fix_configs(grad=True, data_only=True),
}
with open("mnist_fix_config.yaml", "w") as wf:
    yaml.dump(fix_cfg, wf, default_flow_style=False)

if args.save:
    state = {"model": model.fix_state_dict(), "epoch": args.epochs}
    torch.save(state, args.save)
    print("Saving fixed state dict to", args.save)

# Let's try float test
print("test float: ", end="")
test(nfp.FIX_NONE)  # after 1 epoch: ~ 92%

if not args.float:
    # Let's load the fix config again, and test it using FIX_FIXED
    print("load from the yaml config and test fixed again: ", end="")
    with open("mnist_fix_config.yaml", "r") as rf:
        fix_cfg = yaml.load(rf)
        model.load_fix_configs(fix_cfg["data"])
        model.load_fix_configs(fix_cfg["grad"], grad=True)
        test(nfp.FIX_FIXED)  # after 1 epoch: ~ 89%
