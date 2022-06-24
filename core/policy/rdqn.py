from typing import Any, Dict, Tuple, Union, Optional

import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt

from tianshou.policy.modelfree.dqn import DQNPolicy
from tianshou.utils import BasicLogger
from tianshou.data import Batch, to_torch_as, to_numpy


class RDQN(DQNPolicy):
    def __init__(
        self,
        mask_criterion: torch.nn.Module,
        loss_weights: list,
        log_interval: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.mask_criterion = mask_criterion
        self.loss_weights = loss_weights
        self.log_interval = log_interval
        self.configure_plotter()

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        model: str = "model",
        input: str = "obs",
        **kwargs: Any,
    ) -> Batch:
        """Compute action over the given batch data.

        If you need to mask the action, please add a "mask" into batch.obs, for
        example, if we have an environment that has "0/1/2" three actions:
        ::

            batch == Batch(
                obs=Batch(
                    obs="original obs, with batch_size=1 for demonstration",
                    mask=np.array([[False, True, False]]),
                    # action 1 is available
                    # action 0 and 2 are unavailable
                ),
                ...
            )

        :param float eps: in [0, 1], for epsilon-greedy exploration method.

        :return: A :class:`~tianshou.data.Batch` which has 3 keys:

            * ``act`` the action.
            * ``logits`` the network's raw output.
            * ``state`` the hidden state.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        model = getattr(self, model)
        obs = batch[input]
        obs_ = obs.obs if hasattr(obs, "obs") else obs
        (logits, road_mask, visited_mask), h = model(obs_, state=state, info=batch.info)
        q = self.compute_q_value(logits, getattr(obs, "mask", None))
        if not hasattr(self, "max_action_num"):
            self.max_action_num = q.shape[1]
        act = to_numpy(q.max(dim=1)[1])
        return Batch(logits=logits, act=act, state=h, road_mask=road_mask, visited_mask=visited_mask)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        if self._target and self._iter % self._freq == 0:
            self.sync_weight()
        self.optim.zero_grad()
        pred = self(batch)
        # Q loss
        weight = batch.pop("weight", 1.0)
        q = pred.logits
        q = q[np.arange(len(q)), batch.act]
        r = to_torch_as(batch.returns.flatten(), q)
        td = r - q
        loss_q = (td.pow(2) * weight).mean()
        batch.weight = td  # prio-buffer
        # road loss
        pred_road = pred.road_mask
        gt_road = torch.round(torch.from_numpy(batch.obs.mask_road).to(pred_road.device)).long()
        loss_r = self.mask_criterion(pred_road, gt_road).mean()
        # visited loss
        pred_visited = pred.visited_mask
        gt_visited = (torch.from_numpy(batch.obs.mask_visited).to(pred_visited.device) > 0).long()
        loss_v = self.mask_criterion(pred_visited, gt_visited).mean()
        # combine loss
        loss = sum(c * l for c, l in zip([loss_q, loss_r, loss_v], self.loss_weights))
        loss.backward()
        self.optim.step()
        self._iter += 1
        # log plots
        self.log_prediction(batch, (pred.road_mask, pred.visited_mask), (gt_road, gt_visited), kwargs["logger"])

        return {"loss": loss.item(), "loss/q": loss_q.item(), "loss/road": loss_r.item(), "loss/visited": loss_v.item()}

    def log_prediction(
        self, batch: Batch, pred: Tuple[torch.Tensor], masks: Tuple[torch.Tensor], logger: BasicLogger, idx: int = 0
    ):
        if self._iter % self.log_interval:
            return

        tag = f"Block {int(self._iter / self.log_interval)} prediction"

        x = batch.obs.obs[idx, :3]
        out_r, out_v = pred[0][idx], pred[1][idx]
        y_r, y_v = masks[0][idx], masks[1][idx]

        fig_r = self.plot_prediction(x, y_r, out_r)
        logger.writer.add_figure(tag + "/road", fig_r, global_step=self._iter)
        plt.close(fig_r)

        fig_v = self.plot_prediction(x, y_v, out_v)
        logger.writer.add_figure(tag + "/visited", fig_v, global_step=self._iter)
        plt.close(fig_v)

    def plot_prediction(self, x: np.ndarray, y: torch.Tensor, out: torch.Tensor) -> plt.Figure:
        x = np.transpose(x, (1, 2, 0))
        x = np.clip((x * (0.229, 0.224, 0.225)) + (0.485, 0.456, 0.406), 0, 1)
        y = y.detach().cpu().numpy()
        y_pred = F.softmax(out.detach(), dim=0)[1].cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(9, 4))
        fig.suptitle("Prediction Plot")

        axes[0].imshow(x)
        axes[0].set_title("Input")

        axes[1].imshow(y, cmap="gray")
        axes[1].set_title("Groundtruth")

        axes[2].imshow(y_pred, cmap="gray")
        axes[2].set_title("Prediction")

        fig.subplots_adjust(left=None, bottom=None, right=None, top=0.9, wspace=0.05, hspace=0.15)
        # fig.tight_layout(rect=[0, 0, 1, 0.9])
        return fig

    def configure_plotter(self):
        DPI = 150
        mpl.rc("savefig", dpi=DPI)
        mpl.rcParams["figure.dpi"] = DPI
        mpl.rcParams["figure.figsize"] = 6.4, 4.8  # Default.
        mpl.rcParams["font.sans-serif"] = "Roboto"
        mpl.rcParams["font.family"] = "sans-serif"

        # Set title text color to dark gray (https://material.io/color) not black.
        TITLE_COLOR = "#212121"
        mpl.rcParams["text.color"] = TITLE_COLOR

        rc = {
            "axes.spines.left": False,
            "axes.spines.right": False,
            "axes.spines.bottom": False,
            "axes.spines.top": False,
            "xtick.bottom": False,
            "xtick.labelbottom": False,
            "ytick.labelleft": False,
            "ytick.left": False,
        }
        mpl.rcParams.update(rc)
