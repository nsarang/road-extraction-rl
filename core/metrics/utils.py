import torch
import torch.nn as nn
import torch.nn.functional as F


def metric_wrapper_picklable(output, target, metric_fn, name, **kwargs):
    prediction = torch.argmax(output, dim=1)
    results = metric_fn(prediction, target, **kwargs)
    named_results = {f"{name}_{k}": v for k, v in results.items()}
    return named_results


def metric_wrapper(metric_fn, name, **kwargs):
    def wrapper(output, target):
        prediction = torch.argmax(output, dim=1)
        results = metric_fn(prediction, target, **kwargs)
        named_results = {f"{name}_{k}": v for k, v in results.items()}
        return named_results

    return wrapper


def confusion_matrix(prediction, target, num_classes):
    prediction = prediction.flatten()
    target = target.flatten()

    target_mask = (target >= 0) & (target < num_classes)
    target = target[target_mask]
    prediction = prediction[target_mask]

    indices = num_classes * target + prediction
    cm = torch.bincount(indices, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return cm


def batch_bincount(target, num_classes):
    batch_size = target.shape[0]
    shifter = torch.arange(batch_size).unsqueeze(1).to(target.device) * num_classes
    target = target.view(batch_size, -1)
    result = torch.bincount((target + shifter).view(-1), minlength=batch_size * num_classes)
    result = result.view(batch_size, num_classes)
    return result


def batch_bincount_v0(target, num_classes):  # Too much memory consumption
    bs = target.shape[0]
    result = torch.zeros(bs, num_classes).to(target.device)
    index = target.view(bs, -1).long()
    src = torch.ones_like(index).type_as(result)
    result.scatter_add_(dim=1, index=index, src=src)
    return result


def one_hot(tensor, num_classes=-1):
    tensor = F.one_hot(tensor, num_classes=num_classes).float()
    dims = torch.arange(tensor.ndim)
    tensor = tensor.permute(0, dims[-1], *dims[1:-1])
    return tensor
