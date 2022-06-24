from typing import Any, Dict, Union, Optional

import torch
import numpy as np

from tianshou.policy import DQNPolicy
from tianshou.data import Batch, to_torch_as, to_numpy


class Horizont(DQNPolicy):
    def __init__(
        self,
        target_horizon: float,
        horizons: np.ndarray,
        weights: np.ndarray = None,
        **kwargs: Any,
    ) -> None:
        kwargs["discount_factor"] = np.array(horizons)
        super().__init__(**kwargs)
        
        assert target_horizon in horizons
        self._target_horizon = horizons.index(target_horizon)
        weights = np.ones(len(horizons)) if weights is None else np.array(weights)
        weights = weights / weights.sum() # normalize
        self.horizont_weights = torch.tensor(weights)

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        model: str = "model",
        input: str = "obs",
        **kwargs: Any,
    ) -> Batch:
        batch = super().forward(batch, state, model, input, **kwargs)
        batch.act = batch.act[:, self._target_horizon]
        batch.target_horizon = np.tile(self._target_horizon, batch.act.shape[0])
        return batch

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        if self._target and self._iter % self._freq == 0:
            self.sync_weight()
        self.optim.zero_grad()
        weight = batch.pop("weight", 1.0)
        q = self(batch).logits
        q = q[np.arange(len(q)), batch.act]
        r = to_torch_as(batch.returns, q)
        td = r - q
        se = td.pow(2) * weight.unsqueeze(-1)
        se = se * to_torch_as(self.horizont_weights, se)
        loss = se.mean()
        batch.weight = td[:, self._target_horizon]  # prio-buffer
        loss.backward()
        self.optim.step()
        self._iter += 1
        log_horizons = {f"loss/horiz:{h}": l.item() for h, l in zip(self._gamma, se.mean(dim=0))}
        return {"loss": loss.item(), **log_horizons}
