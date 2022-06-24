from typing import Any, Dict, Tuple, Union, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from tianshou.utils.net.common import MLP
from tianshou.data.utils.converter import to_torch, to_torch_as
from core.policy.networks.utils import LambdaModule


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, slen, ntoken, ndim=32, nout=16, nhead=8, nhid=128, nlayers=6, dropout=0.1):
        super().__init__()
        self.model_type = "Transformer"
        self.ntoken = ntoken + 1  # class token
        slen = slen + 1
        self.encoder = nn.Embedding(self.ntoken, ndim)
        self.pos_encoder = PositionalEncoding(ndim, dropout, max_len=slen)
        encoder_layers = nn.TransformerEncoderLayer(ndim, nhead, nhid, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.decoder = nn.Linear(ndim, nout)
        self.ndim = ndim
        self.register_buffer("class_token", torch.tensor(self.ntoken - 1))
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        for mod in self.decoder.modules():
            if isinstance(mod, nn.Linear):
                mod.bias.data.zero_()
                mod.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        cind = self.class_token.expand(src.size(0), 1)
        src = torch.cat((src, cind), dim=1)
        src = self.encoder(src) * math.sqrt(self.ndim)
        src = self.pos_encoder(src)
        attmap = self.transformer_encoder(src)
        output = attmap[:, -1, :]
        output = self.decoder(output)
        return output


class TransEfficient(nn.Module):
    def __init__(
        self,
        preprocess_net: nn.Module,
        action_shape: Sequence[int],
        preprocess_net_output_dim: int,
        actions_seq_len: int,
        actions_output_dim: int,
        output_shape: Sequence[int] = (),
        hidden_sizes: Sequence[int] = (),
        softmax_output: bool = True,
        normalize_fusion=True,
        device: Union[str, int, torch.device] = "cpu",
        **kwargs,
    ) -> None:
        super().__init__()
        self.device = device
        nact = np.prod(action_shape)
        self.output_shape = output_shape or (nact, )
        self.softmax_output = softmax_output

        self.convnet = preprocess_net
        self.action_encoder = TransformerModel(
            slen=actions_seq_len, ntoken=nact + 1, nout=actions_output_dim, **kwargs
        )
        self.head = MLP(
            preprocess_net_output_dim + actions_output_dim,
            np.prod(self.output_shape),
            hidden_sizes,
            device=self.device,
            # norm_layer=nn.LayerNorm,
        )

        self.normalize_fusion = normalize_fusion
        self.norm_im = nn.LayerNorm(preprocess_net_output_dim)
        self.norm_act = nn.LayerNorm(actions_output_dim)

    def forward(
        self,
        s: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        s = to_torch(s, device=self.device, dtype=torch.float32)
        img, actions = s["obs"], s["actions"].long()
        img_feat = self.convnet(img)
        action_feat = self.action_encoder(actions)
        if self.normalize_fusion:
            img_feat = self.norm_im(img_feat)
            action_feat = self.norm_act(action_feat)
        feat = torch.cat([img_feat, action_feat], dim=1)
        logits = self.head(feat)
        if self.softmax_output:
            logits = F.softmax(logits, dim=-1)
        logits = logits.reshape(logits.size(0), *self.output_shape)
        return logits, state
