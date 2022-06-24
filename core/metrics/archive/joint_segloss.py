"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np

# from core.nn.DualTaskLoss import DualTaskLoss
# from core.config import cfg
from .utils import batched_bincount


class JointEdgeSegLoss(nn.Module):
    def __init__(
        self,
        num_classes,
        ignore_index=255,
        upper_bound=1.0,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.weighted_seg_loss = WeightedCrossEntropyLoss(num_classes=self.num_classes)
        self.seg_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dual_task = DualTaskLoss(self.num_classes)

    def bce2d(self, output, target):
        output = output.view(-1)
        target = target.view(-1)

        pos_ratio = (target == 1).float().mean()
        class_weight = torch.stack([pos_ratio, 1 - pos_ratio])
        position_weight = class_weight[target.long()]

        loss = position_weight * F.binary_cross_entropy(output, target.float(), reduction="none")
        loss = loss.mean()

        return loss

    def edge_attention(self, input, target, edge):
        n, c, h, w = input.size()
        filler = torch.ones_like(target) * 255
        return self.seg_loss(input, torch.where(edge.max(1)[0] > 0.8, target, filler))

    def forward(self, inputs, targets, mode):
        segin, edgein = inputs
        segmask, edgemask = targets

        losses = {}

        seg_loss_fn = self.weighted_seg_loss if mode == "train" else self.seg_loss
        losses["loss_seg"] = seg_loss_fn(segin, segmask)
        losses["loss_edge"] = self.bce2d(edgein, edgemask)
        losses["loss_att"] = self.edge_attention(segin, segmask, edgein)
        losses["loss_dual"] = self.dual_task(segin, segmask)

        return losses


# Mask Weighted Loss
class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes, smooth=0.6):
        super().__init__()
        logging.info("Using Per Segmap based weighted loss")
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, outputs, targets):
        counts = batched_bincount(targets, self.num_classes).float()
        weights = 1 - counts / counts.sum(dim=1, keepdim=True)
        weights = weights * (1 - self.smooth) + weights.mean(dim=1, keepdim=True) * self.smooth
        loss = sum(
            F.cross_entropy(output.unsqueeze(0), target.unsqueeze(0), weight=weight)
            for output, target, weight in zip(outputs, targets, weights)
        ) / len(outputs)
        return loss
