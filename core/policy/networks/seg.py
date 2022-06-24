from typing import Any, Dict, List, Tuple, Union, Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pydoc import locate
from copy import deepcopy
from timm.models.efficientnet_builder import efficientnet_init_weights
from core.policy.nn.common import ConvBNReLU_block, SE_block
from configs.arch import DDRNET_CFG
from .ddrnet import SegmentationHead


class SECNN(nn.Module):
    def __init__(self, obs_channels, feature_vector_size):
        super().__init__()

        self.mods = nn.Sequential(
            ConvBNReLU_block(obs_channels, 128, 3, stride=4, padding=1, dilation=1, bias=False),
            ConvBNReLU_block(128, 128, 3, stride=2, padding=1, dilation=1, bias=False),
            ConvBNReLU_block(128, 128, 3, stride=2, padding=1, dilation=1, bias=False),
            ConvBNReLU_block(128, 128, 3, stride=2, padding=1, dilation=1, bias=False),
            ConvBNReLU_block(128, 128, 3, stride=2, padding=1, dilation=1, bias=False),
            SE_block(128),
            ConvBNReLU_block(128, 256, 3, stride=2, padding=1, dilation=1, bias=False),
            ConvBNReLU_block(256, 256, 3, stride=1, padding=1, dilation=1, bias=False),
            SE_block(256),
            ConvBNReLU_block(256, 256, 3, stride=1, padding=1, dilation=1, bias=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.Linear(256, feature_vector_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mapping: s -> flatten -> logits."""
        logits = self.mods(x)
        return logits


class RoadSegExtendedHead(nn.Module):
    def __init__(self, model, feature_vector_size):
        super().__init__()
        self.model = model
        self.convs = nn.Sequential(
            ConvBNReLU_block(2, 16, 3, stride=1, padding=1, dilation=1, bias=False),
            ConvBNReLU_block(16, 16, 3, stride=1, padding=1, dilation=1, bias=True),
            ConvBNReLU_block(16, 32, 3, stride=2, padding=1, dilation=1, bias=False),
            SE_block(32),
            ConvBNReLU_block(32, 32, 3, stride=2, padding=1, dilation=1, bias=False),
            ConvBNReLU_block(32, 32, 3, stride=2, padding=1, dilation=1, bias=False),
            SE_block(32),
            ConvBNReLU_block(32, 64, 3, stride=2, padding=1, dilation=1, bias=False),
            ConvBNReLU_block(64, 128, 3, stride=2, padding=1, dilation=1, bias=False),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, feature_vector_size, bias=False)

    def forward(self, s, state=None, info={}):
        rgb = s["obs"]
        movement = s["movement"]
        bs = rgb.shape[0]

        output, aux = self.model(rgb)
        heatmap = torch.softmax(output, dim=1)
        road_heatmap = heatmap[:, 1, ...]
        concatenated = torch.stack((road_heatmap, movement), dim=1)

        spatial_features = self.convs(concatenated)
        feature_vector = self.head(spatial_features.view(bs, -1))
        return feature_vector, state


class JointSegHead(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.roadH = SegmentationHead(**kwargs)
        self.juncH = SegmentationHead(**kwargs)

    def forward(self, x):
        road = self.roadH(x)
        junc = self.juncH(x)
        return road, junc


class MultiMaskDDRNet(nn.Module):
    def __init__(self, name, use_aux=False, **kwargs):
        super().__init__()
        cfg = dict(deepcopy(DDRNET_CFG[name]), **kwargs, num_classes=0, augment=use_aux)
        self.use_aux = use_aux
        self.backbone = locate("core.policy.networks.ddrnet." + name)(**cfg)
        self.head = JointSegHead(
            inplanes=cfg["planes"] * 4, interplanes=cfg["head_planes"], outplanes=2, scale_factor=8
        )
        self.aux_head = (
            JointSegHead(inplanes=cfg["planes"] * 2, interplanes=cfg["head_planes"], outplanes=2, scale_factor=8)
            if use_aux
            else None
        )

        # efficientnet_init_weights(self)

    def forward(self, x):
        f, f_inter = self.backbone.forward_features(x)
        out = self.head(f)
        if self.use_aux:
            out_aux = self.aux_head(f_inter)
            return [out, out_aux]
        return out