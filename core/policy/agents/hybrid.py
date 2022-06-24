from typing import Any, Dict, Tuple, Union, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from gym.spaces import Dict as DictSpace
from tianshou.utils.net.common import MLP
from core.policy.networks.utils import to_torch, to_torch_as, orthogonal_initialization

SIGMA_MIN = -20
SIGMA_MAX = 2


class HybridActor(nn.Module):
    def __init__(
        self,
        preprocess_net: nn.Module,
        action_space: DictSpace,
        head: Optional[nn.Module] = None,
        hidden_sizes: Sequence[int] = (),
        softmax_output: bool = True,
        device: Union[str, int, torch.device] = "cpu",
        conditioned_sigma: bool = False,
        preprocess_net_output_dim: Optional[int] = None,
        unbounded: bool = False,
        no_grad_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.preprocess = preprocess_net
        self.device = device
        self.output_dim_d = action_space["retrace"].n
        self.output_dim_c = int(np.prod(action_space["direction"].shape))

        input_dim_d = getattr(preprocess_net, "output_dim", preprocess_net_output_dim)
        input_dim_c = input_dim_d + self.output_dim_d

        self.head = head or MLP(input_dim_d, self.output_dim_d, hidden_sizes, device=self.device)
        self.softmax_output = softmax_output

        self.mu = head or MLP(input_dim_c, self.output_dim_c, hidden_sizes, device=self.device)
        self._c_sigma = conditioned_sigma
        if conditioned_sigma:
            self.sigma = MLP(input_dim_c, self.output_dim_c, hidden_sizes, device=self.device)
        else:
            self.sigma_param = nn.Parameter(torch.zeros(self.output_dim_c, 1))

        self._max = np.max(np.abs([action_space["direction"].low[0], action_space["direction"].high[0]]))
        self._unbounded = unbounded
        self.no_grad_backbone = no_grad_backbone

        orthogonal_initialization(self)

    def forward(
        self,
        s: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Any]:
        """Mapping: s -> logits -> (mu, sigma)."""
        s = to_torch(s, device=self.device, dtype=torch.float32)
        if self.no_grad_backbone:
            with torch.no_grad():
                features = self.preprocess(s["obs"])
        else:
            features = self.preprocess(s["obs"])

        logits_d = self.head(features)
        features_d = torch.cat((features, logits_d), dim=1)
        if self.softmax_output:
            logits_d = F.softmax(logits_d, dim=-1)

        mu = self.mu(features_d)
        if not self._unbounded:
            mu = self._max * torch.tanh(mu)
        if self._c_sigma:
            sigma = torch.clamp(self.sigma(features_d), min=SIGMA_MIN, max=SIGMA_MAX).exp()
        else:
            shape = [1] * len(mu.shape)
            shape[1] = -1
            sigma = (self.sigma_param.view(shape) + torch.zeros_like(mu)).exp()
        return (logits_d, (mu, sigma)), state


class HybridCritic(nn.Module):
    def __init__(
        self,
        preprocess_net: nn.Module,
        action_space: DictSpace,
        head: Optional[nn.Module] = None,
        hidden_sizes: Sequence[int] = (),
        device: Union[str, int, torch.device] = "cpu",
        preprocess_net_output_dim: Optional[int] = None,
        discrete_action_index: bool = True,
    ) -> None:
        super().__init__()
        self.device = device
        self.preprocess = preprocess_net
        self.a_d_index = discrete_action_index
        self.output_dim = 1

        input_dim_d = 1 if discrete_action_index else action_space["retrace"].n
        input_dim_c = int(np.prod(action_space["direction"].shape))
        input_dim_act = input_dim_d + input_dim_c
        branch_dim = getattr(preprocess_net, "output_dim", preprocess_net_output_dim)
        input_dim_fusion = 2 * branch_dim

        self.feat_norm = nn.LayerNorm(branch_dim)
        self.act_norm = nn.LayerNorm(branch_dim)
        self.act_expansion = nn.Linear(input_dim_act, branch_dim)
        self.last = head or MLP(input_dim_fusion, self.output_dim, hidden_sizes, device=self.device)

        orthogonal_initialization(self)

    def forward(
        self,
        s: Union[np.ndarray, torch.Tensor],
        a: Optional[Union[np.ndarray, torch.Tensor]] = None,
        info: Dict[str, Any] = {},
    ) -> torch.Tensor:
        """Mapping: (s, a) -> logits -> Q(s, a)."""
        s = to_torch(s, device=self.device, dtype=torch.float32)
        features = self.preprocess(s["obs"])
        features = self.feat_norm(features)
        bs = features.shape[0]
        a = to_torch_as(a, features)
        a_d, a_c = a["retrace"], a["direction"]

        if self.a_d_index:
            a_d, a_c = a_d.reshape(bs, -1), a_c.reshape(bs, -1)
            a_c = torch.where(torch.isclose(a_d, torch.ones_like(a_d)), torch.zeros_like(a_c), a_c)
        else:
            a_c = torch.where(torch.argmax(a_d, dim=1,keepdim=True) == 1, torch.zeros_like(a_c), a_c)

        act = torch.cat([a_d, a_c], dim=1)
        act = self.act_expansion(act)
        act = self.act_norm(act)

        logits = torch.cat([features, act], dim=1)
        logits = self.last(logits)
        return logits, (features, act)
