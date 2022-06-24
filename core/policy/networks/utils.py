import torch
import torch.nn as nn
from inspect import signature
import numpy as np

from tianshou.data.batch import Batch
from numbers import Number
from typing import Dict, Union, Optional, List, Tuple


def to_torch(
    x: Union[Dict, List, Tuple, np.number, np.bool_, Number, np.ndarray, torch.Tensor],
    dtype: Optional[torch.dtype] = None,
    device: Union[str, int, torch.device] = "cpu",
) -> Union[Dict, List, tuple, torch.Tensor]:
    """Return an object without np.ndarray."""
    if isinstance(x, np.ndarray) and issubclass(x.dtype.type, (np.bool_, np.number)):  # most often case
        x = torch.from_numpy(x).to(device)  # type: ignore
        if dtype is not None:
            x = x.type(dtype)
        return x
    elif isinstance(x, torch.Tensor):  # second often case
        if dtype is not None:
            x = x.type(dtype)
        return x.to(device)  # type: ignore
    elif isinstance(x, (np.number, np.bool_, Number)):
        return to_torch(np.asanyarray(x), dtype, device)
    elif isinstance(x, (dict, Batch)):
        if isinstance(x, Batch):
            x = dict(x.items())
        return {k: to_torch(v, dtype, device) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        return [to_torch(e, dtype, device) for e in x]
    else:  # fallback
        raise TypeError(f"object {x} cannot be converted to torch.")


def to_torch_as(
    x: Union[dict, list, tuple, np.ndarray, torch.Tensor],
    y: torch.Tensor,
) -> Union[dict, list, tuple, torch.Tensor]:
    """Return an object without np.ndarray.
    Same as ``to_torch(x, dtype=y.dtype, device=y.device)``.
    """
    assert isinstance(y, torch.Tensor)
    return to_torch(x, dtype=y.dtype, device=y.device)


def model_has_state(model):
    return isinstance(model, nn.Module) and len(signature(model.forward).parameters) == 2


def stateInputInjection(model):
    def forward(self, s, state=None, info={}):
        return self.forward_ex(s), state

    model.forward_ex = model.forward
    model.forward = forward.__get__(model)
    return model


class StatelessWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, s, state=None, info={}):
        return self.module(s), state


class LambdaModule(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


class TSDP(nn.DataParallel):
    def forward(self, *args, **kwargs):
        args = to_torch(args, dtype=torch.float32)
        # sol for DP scatter bug
        if "state" in kwargs and (kwargs["state"] is None):
            kwargs.pop("state")
        if "info" in kwargs:
            kwargs.pop("info")
        return super().forward(*args, **kwargs)


def orthogonal_initialization(model: nn.Module):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight)
        if isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
