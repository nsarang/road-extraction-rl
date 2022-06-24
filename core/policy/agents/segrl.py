from typing import Any, Dict, Tuple, Union, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from tianshou.utils.net.common import MLP
from tianshou.data.utils.converter import to_torch, to_torch_as
from .transformer import TransformerModel




# class SegRL(nn.Module):
#     def __init__(
#         self,
#         preprocess_net: nn.Module,
#         action_shape: Sequence[int],
#         preprocess_net_output_dim: int,
#         actions_seq_len: int,
#         actions_output_dim: int,
#         output_shape: Sequence[int] = (),
#         hidden_sizes: Sequence[int] = (),
#         softmax_output: bool = True,
#         device: Union[str, int, torch.device] = "cpu",
#         **kwargs,
#     ) -> None:
#         super().__init__()
#         self.device = device
#         nact = np.prod(action_shape)
#         self.output_shape = output_shape or (nact, )
#         self.softmax_output = softmax_output

#         self.preprocess_net = preprocess_net
#         self.action_encoder = TransformerModel(
#             slen=actions_seq_len, ntoken=nact + 1, nout=actions_output_dim, **kwargs
#         )
#         self.head = MLP(
#             preprocess_net_output_dim + actions_output_dim,
#             np.prod(self.output_shape),
#             hidden_sizes,
#             device=self.device,
#             norm_layer=nn.LayerNorm,
#         )

#     def forward(
#         self,
#         s: Union[np.ndarray, torch.Tensor],
#         state: Any = None,
#         info: Dict[str, Any] = {},
#     ) -> Tuple[torch.Tensor, Any]:
#         r"""Mapping: s -> Q(s, \*)."""
#         s = to_torch(s, device=self.device, dtype=torch.float32)
#         obs, actions = s["obs"], s["actions"].long()
#         rbg, movement = obs[:, :3], obs[:, 3:]

#         with torch.no_grad():
#         r_mask = self.convnet(img)
#         action_feat = self.action_encoder(actions)
#         feat = torch.cat([img_feat, action_feat], dim=1)
#         logits = self.head(feat)
#         if self.softmax_output:
#             logits = F.softmax(logits, dim=-1)
#         logits = logits.reshape(logits.size(0), *self.output_shape)
#         return logits, state
