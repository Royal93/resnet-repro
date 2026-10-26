"""
Plain and residual networks for CIFAR-10, following He et al. (2016), Section 4.2.

Architecture (depth = 6n + 2):
    conv3x3(16) -> [2n layers @ 16 filters, 32x32] -> [2n @ 32, 16x16] -> [2n @ 64, 8x8]
    -> global average pool -> 10-way fully connected
Shortcuts connect every pair of 3x3 layers (3n shortcuts in total).

Shortcut options (paper, Sec. 3.3 / Table 3):
    A: identity, zero-padding the extra channels when dimensions increase (no parameters).
       This is what the paper uses on CIFAR-10, so plain and residual nets have
       exactly the same number of parameters.
    B: 1x1 convolution projection (+BN) only when dimensions increase.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ShortcutA(nn.Module):
    """Option A: subsample spatially with stride 2 and zero-pad the new channels."""

    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.extra = out_planes - in_planes

    def forward(self, x):
        x = x[:, :, ::2, ::2]
        pad = self.extra // 2
        return F.pad(x, (0, 0, 0, 0, pad, self.extra - pad))


class BasicBlock(nn.Module):
    """Two 3x3 conv layers. With residual=True the block outputs relu(F(x) + x)."""

    def __init__(self, in_planes, planes, stride=1, residual=True, shortcut="A"):
        super().__init__()
        self.residual = residual
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Identity()
        if residual and (stride != 1 or in_planes != planes):
            if shortcut == "A":
                self.shortcut = ShortcutA(in_planes, planes)
            elif shortcut == "B":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                    nn.BatchNorm2d(planes),
                )
            else:
                raise ValueError(f"unknown shortcut option {shortcut}")

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.residual:
            out = out + self.shortcut(x)  # F(x) + x
        return F.relu(out)                # ReLU after the addition (paper, Fig. 2)


class CifarNet(nn.Module):
    def __init__(self, depth=20, residual=True, shortcut="A", num_classes=10):
        super().__init__()
        assert (depth - 2) % 6 == 0, "depth must be 6n+2 (20, 32, 44, 56, 110, ...)"
        n = (depth - 2) // 6
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        layers, in_planes = [], 16
        for planes, stride in [(16, 1), (32, 2), (64, 2)]:
            for i in range(n):
                layers.append(BasicBlock(in_planes, planes, stride if i == 0 else 1,
                                         residual=residual, shortcut=shortcut))
                in_planes = planes
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(64, num_classes)

        # He initialisation (paper ref. [13])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


def build_model(arch, depth, shortcut="A"):
    if arch not in ("plain", "resnet"):
        raise ValueError("arch must be 'plain' or 'resnet'")
    return CifarNet(depth=depth, residual=(arch == "resnet"), shortcut=shortcut)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    # Sanity check: parameter counts should match Table 6 (0.27M, 0.46M, 0.66M, 0.85M, 1.7M).
    for d in (20, 32, 44, 56, 110):
        r = count_params(build_model("resnet", d))
        p = count_params(build_model("plain", d))
        print(f"depth {d:4d}: resnet {r/1e6:.2f}M  plain {p/1e6:.2f}M")
