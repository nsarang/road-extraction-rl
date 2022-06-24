import torch


def orthogonal_regularization_loss(model):
    orth_losses = []
    for name, p in model.named_parameters():
        if ("bias" in name) or ("bn" in name):
            continue
        p = p.view(p.shape[0], -1)
        if p.shape[0] > p.shape[1]:
            p = p.t()
        loss = (torch.mm(p, p.t()) - torch.eye(p.shape[0], device=p.device)).abs().mean()
        orth_losses.append(loss)
    return sum(orth_losses) / len(orth_losses)
