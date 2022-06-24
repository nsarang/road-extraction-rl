import numpy as np
import torch
from torch._six import inf
from collections import deque


def grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        norms = [p.grad.detach().abs().max().to(device) for p in parameters]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type
        )
    return total_norm


class AutoClip:
    def __init__(self, maxlen: int = 1000, clip_percentile: float = 75) -> None:
        self.clip_percentile = clip_percentile
        self.grad_history = deque(maxlen=maxlen)

    def __call__(self, model: torch.nn.Module) -> None:
        m_grad_norm = grad_norm(model.parameters()).item()
        self.grad_history.append(m_grad_norm)
        clip_value = np.percentile(self.grad_history, self.clip_percentile)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
