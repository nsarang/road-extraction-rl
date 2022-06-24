from typing import Any, Dict, Tuple, Union, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from tianshou.utils.net.common import MLP
from tianshou.data.utils.converter import to_torch, to_torch_as
from core.policy.networks.utils import LambdaModule


class MultitaskActor(nn.Module):
    def __init__(
        self,
        preprocess_net: nn.Module,
        action_shape: Sequence[int],
        mlp_input_dim: int,
        hidden_sizes: Sequence[int] = (),
        softmax_output: bool = True,
        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.output_dim = np.prod(action_shape)
        self.preprocess = preprocess_net
        self.roadm = nn.Sequential(
            nn.Upsample(scale_factor=2),
            ConvModule(40, 40, 3, stride=1, padding=1, norm_cfg=dict(type="BN"), act_cfg=dict(type="ReLU")),
            nn.Upsample(scale_factor=2),
            ConvModule(40, 20, 3, stride=1, padding=1, norm_cfg=dict(type="BN"), act_cfg=dict(type="ReLU")),
            ConvModule(20, 2, 3, stride=1, padding=1, act_cfg=None),
        )
        self.visitedm = nn.Sequential(
            nn.Upsample(scale_factor=2),
            ConvModule(40, 40, 3, stride=1, padding=1, norm_cfg=dict(type="BN"), act_cfg=dict(type="ReLU")),
            nn.Upsample(scale_factor=2),
            ConvModule(40, 20, 3, stride=1, padding=1, norm_cfg=dict(type="BN"), act_cfg=dict(type="ReLU")),
            ConvModule(20, 2, 3, stride=1, padding=1, act_cfg=None),
        )
        self.actions = nn.Sequential(
            ConvModule(40, 64, 3, stride=2, padding=1, norm_cfg=dict(type="BN"), act_cfg=dict(type="ReLU")),
            ConvModule(64, 64, 3, stride=2, padding=1, norm_cfg=dict(type="BN"), act_cfg=dict(type="ReLU")),
            ConvModule(64, 128, 3, stride=2, padding=1, norm_cfg=dict(type="BN"), act_cfg=dict(type="ReLU")),
            ConvModule(128, mlp_input_dim, 3, stride=2, padding=1, norm_cfg=dict(type="BN"), act_cfg=dict(type="ReLU")),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            MLP(mlp_input_dim, self.output_dim, hidden_sizes, device=self.device),
        )
        self.softmax_output = softmax_output

    def forward(
        self,
        s: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        s = to_torch(s, device=self.device, dtype=torch.float32)
        features = self.preprocess(s)[0]
        road_mask = self.roadm(features)
        visited_mask = self.visitedm(features)
        logits = self.actions(features)
        if self.softmax_output:
            logits = F.softmax(logits, dim=-1)
        return (logits, road_mask, visited_mask), state
