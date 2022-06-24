import torch
import torch.nn as nn
import torch.nn.functional as F

from .blurpool import BlurPool


class ConvBNReLU_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# Squeeze-and-Excitation Networks
# https://arxiv.org/abs/1709.01507
class SE_block(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)

    def forward(self, x):
        bs = x.size(0)
        out = self.avg_pool(x).view(bs, -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = torch.sigmoid(out)
        out = out.view(bs, -1, 1, 1)
        return x * out


def noOp(x):
    return x


class ConvBlurPool(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, **kwargs)
        self.bp = BlurPool(out_channels, stride=stride) if (stride != 1) else noOp

    def forward(self, x):
        x = self.conv(x)
        x = self.bp(x)
        return x


def conv1d(ni: int, no: int, ks: int = 1, stride: int = 1, padding: int = 0, bias: bool = False):
    "Create and initialize a `nn.Conv1d` layer with spectral normalization."
    conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
    nn.init.kaiming_normal_(conv.weight)
    if bias:
        conv.bias.data.zero_()
    return nn.utils.spectral_norm(conv)


class SimpleSelfAttention(nn.Module):
    def __init__(self, n_in: int, ks=1, sym=False):
        super().__init__()
        self.conv = conv1d(n_in, n_in, ks, padding=ks // 2, bias=False)
        self.gamma = nn.Parameter(torch.tensor([0.]))
        self.sym = sym
        self.n_in = n_in

    def forward(self, x):
        if self.sym:
            # symmetry hack by https://github.com/mgrankin
            c = self.conv.weight.view(self.n_in, self.n_in)
            c = (c + c.t()) / 2
            self.conv.weight = c.view(self.n_in, self.n_in, 1)

        size = x.size()
        x = x.view(*size[:2], -1)  # (C,N)

        # changed the order of mutiplication to avoid O(N^2) complexity
        # (x*xT)*(W*x) instead of (x*(xT*(W*x)))
        convx = self.conv(x)  # (C,C) * (C,N) = (C,N)   => O(NC^2)
        xxT = torch.bmm(x, x.permute(0, 2, 1).contiguous())  # (C,N) * (N,C) = (C,C)   => O(NC^2)

        o = torch.bmm(xxT, convx)  # (C,C) * (C,N) = (C,N)   => O(NC^2)
        o = self.gamma * o + x
        return o.view(*size).contiguous()
