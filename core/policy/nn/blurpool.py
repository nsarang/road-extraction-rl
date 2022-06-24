import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class BlurPool(nn.Module):
    def __init__(self, channels, pad_type="reflect", kernel_size=4, stride=2, pad_off=0):
        super(BlurPool, self).__init__()
        self.kernel_size = kernel_size
        self.pad_off = pad_off
        self.pad_sizes = [
            int(1.0 * (kernel_size - 1) / 2),
            int(np.ceil(1.0 * (kernel_size - 1) / 2)),
            int(1.0 * (kernel_size - 1) / 2),
            int(np.ceil(1.0 * (kernel_size - 1) / 2)),
        ]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.0)
        self.channels = channels

        kernel_1d = np.array([math.comb(self.kernel_size - 1, x) for x in range(self.kernel_size)])
        kernel_2d = torch.Tensor(kernel_1d[:, None] * kernel_1d)
        kernel_2d = kernel_2d / torch.sum(kernel_2d)
        self.register_buffer("kernel_2d", kernel_2d[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, x):
        if self.kernel_size == 1:
            if self.pad_off == 0:
                return x[:, :, :: self.stride, :: self.stride]
            else:
                return self.pad(x)[:, :, :: self.stride, :: self.stride]
        else:
            return F.conv2d(self.pad(x), self.kernel_2d, stride=self.stride, groups=x.shape[1])


def get_pad_layer(pad_type):
    if pad_type in ["refl", "reflect"]:
        PadLayer = nn.ReflectionPad2d
    elif pad_type in ["repl", "replicate"]:
        PadLayer = nn.ReplicationPad2d
    elif pad_type == "zero":
        PadLayer = nn.ZeroPad2d
    else:
        print("Pad type [%s] not recognized" % pad_type)
    return PadLayer