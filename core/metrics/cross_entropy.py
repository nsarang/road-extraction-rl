import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import batch_bincount, one_hot


class CrossEntropyLoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super(CrossEntropyLoss, self).__init__()
        self.reduction = reduction
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprob = self.log_softmax(input)
        loss = -torch.sum(logprob * target, dim=1)
        if self.reduction == "mean":
            loss = loss.mean()
        if self.reduction == "sum":
            loss = loss.sum()
        return loss


class InstanceWeightedCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes, smooth=0.6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, outputs, targets):
        counts = batch_bincount(targets, self.num_classes).float()
        weights = 1 - counts / counts.sum(dim=1, keepdim=True)
        weights = weights * (1 - self.smooth) + weights.mean(dim=1, keepdim=True) * self.smooth
        loss = sum(
            F.cross_entropy(output.unsqueeze(0), target.unsqueeze(0), weight=weight)
            for output, target, weight in zip(outputs, targets, weights)
        ) / len(outputs)
        return loss



class BatchWeightedCrossEntropyLoss(nn.Module):
    def __init__(self, smooth=0, reduction: str = "mean"):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if input.ndim != target.ndim:
            assert input.ndim == target.ndim + 1
            target = one_hot(target, num_classes=input.shape[1])
            # counts = torch.bincount(target, minlength=input.shape[1]).float()
            # weights = 1 - counts / counts.sum(dim=1)
        
        weights = 1 - target.transpose(0, 1).reshape(target.shape[1], -1).float().mean(-1)
        weights = (1 - self.smooth) * weights + self.smooth * weights.mean()
        
        logprob = self.log_softmax(input)
        loss = -torch.sum(logprob * target, dim=1)
        if self.reduction == "mean":
            loss = loss.mean()
        if self.reduction == "sum":
            loss = loss.sum()
        return loss