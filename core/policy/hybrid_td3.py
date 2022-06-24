import torch
import numpy as np
from copy import deepcopy
from typing import Any, Dict, Optional, Union, Tuple

from torch.distributions import Independent, Normal, Categorical
from tianshou.policy import DDPGPolicy
from tianshou.data import Batch, ReplayBuffer
from tianshou.exploration import BaseNoise, GaussianNoise
from core.optim.autoclip import grad_norm
from scipy.special import softmax
from core.metrics.orth import orthogonal_regularization_loss


class HybridTD3Policy(DDPGPolicy):
    """Implementation of TD3, arXiv:1802.09477.

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer actor_optim: the optimizer for actor network.
    :param torch.nn.Module critic1: the first critic network. (s, a -> Q(s, a))
    :param torch.optim.Optimizer critic1_optim: the optimizer for the first
        critic network.
    :param torch.nn.Module critic2: the second critic network. (s, a -> Q(s, a))
    :param torch.optim.Optimizer critic2_optim: the optimizer for the second
        critic network.
    :param float tau: param for soft update of the target network. Default to 0.005.
    :param float gamma: discount factor, in [0, 1]. Default to 0.99.
    :param float exploration_noise: the exploration noise, add to the action.
        Default to ``GaussianNoise(sigma=0.1)``
    :param float policy_noise: the noise used in updating policy network.
        Default to 0.2.
    :param int update_actor_freq: the update frequency of actor network.
        Default to 2.
    :param float noise_clip: the clipping range used in updating policy network.
        Default to 0.5.
    :param bool reward_normalization: normalize the reward to Normal(0, 1).
        Default to False.
    :param bool action_scaling: whether to map actions from range [-1, 1] to range
        [action_spaces.low, action_spaces.high]. Default to True.
    :param str action_bound_method: method to bound action to range [-1, 1], can be
        either "clip" (for simply clipping the action) or empty string for no bounding.
        Default to "clip".
    :param Optional[gym.Space] action_space: env's action space, mandatory if you want
        to use option "action_scaling" or "action_bound_method". Default to None.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        actor: torch.nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1: torch.nn.Module,
        critic1_optim: torch.optim.Optimizer,
        critic2: torch.nn.Module,
        critic2_optim: torch.optim.Optimizer,
        tau: float = 0.005,
        gamma: float = 0.99,
        noise_discrete: float = 0,
        noise_continuous: float = 1e-8,
        policy_noise: float = 0.2,
        update_actor_freq: int = 2,
        noise_clip: float = 0.5,
        reward_normalization: bool = False,
        estimation_step: int = 1,
        orth_weight: float = 1e-5,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            actor,
            actor_optim,
            None,
            None,
            tau,
            gamma,
            None,
            reward_normalization,
            estimation_step,
            **kwargs,
        )
        self.critic1, self.critic1_old = critic1, deepcopy(critic1)
        self.critic1_old.eval()
        self.critic1_optim = critic1_optim
        self.critic2, self.critic2_old = critic2, deepcopy(critic2)
        self.critic2_old.eval()
        self.critic2_optim = critic2_optim
        self._policy_noise = policy_noise
        self._freq = update_actor_freq
        self._noise_d = noise_discrete
        self._noise_c = GaussianNoise(sigma=noise_continuous)
        self._noise_clip = noise_clip
        self._cnt = 0
        self._last = 0
        self._last_orth = 0
        self._orth_weight = orth_weight

    def train(self, mode: bool = True) -> "HybridTD3Policy":
        self.training = mode
        self.actor.train(mode)
        self.critic1.train(mode)
        self.critic2.train(mode)
        return self

    def sync_weight(self) -> None:
        for o, n in zip(self.actor_old.parameters(), self.actor.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic1_old.parameters(), self.critic1.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic2_old.parameters(), self.critic2.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)

    def forward(  # type: ignore
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        input: str = "obs",
        **kwargs: Any,
    ) -> Batch:
        obs = batch[input]
        (logits_d, logits_c), h = self.actor(obs, state=state, info=batch.info)
        act_d = logits_d
        act_c = logits_c[0]
        act = dict(retrace=act_d, direction=act_c)

        with torch.no_grad():
            target_q = torch.min(
                self.critic1_old(obs, act)[0],
                self.critic2_old(obs, act)[0],
            )

        return Batch(
            act=act,
            logits_disc=logits_d,
            logits_cont=torch.stack(logits_c, dim=2),
            value=target_q,
            state=h,
        )

    def _target_q(self, buffer: ReplayBuffer, indice: np.ndarray) -> torch.Tensor:
        batch = buffer[indice]  # batch.obs: s_{t+n}
        a_ = self(batch, model="actor_old", input="obs_next").act
        noise = torch.empty_like(a_["direction"]).normal_(std=self._policy_noise)
        if self._noise_clip > 0.0:
            noise = noise.clamp(-self._noise_clip, self._noise_clip)
        a_["direction"] += noise
        target_q = torch.min(self.critic1_old(batch.obs_next, a_)[0], self.critic2_old(batch.obs_next, a_)[0])
        return target_q

    @staticmethod
    def _mse_optimizer(
        batch: Batch,
        critic: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        orth_weight: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A simple wrapper script for updating critic network."""
        weight = getattr(batch, "weight", 1.0)
        current_q = critic(batch.obs, batch.act)[0].flatten()
        target_q = batch.returns.flatten()
        td = current_q - target_q
        # critic_loss = F.mse_loss(current_q1, target_q)
        critic_loss = (td.pow(2) * weight).mean()
        orth_loss = orth_weight * orthogonal_regularization_loss(critic)
        loss = critic_loss + orth_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        norm = grad_norm(critic.parameters())
        return td, (critic_loss, orth_loss), norm

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        # critic 1&2
        td1, (critic1_loss, orth1_loss), norm1 = self._mse_optimizer(
            batch, self.critic1, self.critic1_optim, self._orth_weight
        )
        td2, (critic2_loss, orth2_loss), norm2 = self._mse_optimizer(
            batch, self.critic2, self.critic2_optim, self._orth_weight
        )
        batch.weight = (td1 + td2) / 2.0  # prio-buffer

        # actor
        if self._cnt % self._freq == 0:
            actor_loss = -self.critic1(batch.obs, self(batch, eps=0.0).act)[0].mean()
            actor_orth_loss = self._orth_weight * orthogonal_regularization_loss(self.actor)
            loss = actor_loss + actor_orth_loss
            self.actor_optim.zero_grad()
            loss.backward()
            self._last = actor_loss.item()
            self._last_orth = actor_orth_loss.item()
            self.actor_optim.step()
            self.sync_weight()
        self._cnt += 1

        return {
            "loss/actor": self._last,
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item(),
            "grad_norm/actor": grad_norm(self.actor_optim.param_groups[0]["params"]).item(),
            "grad_norm/critic1_actor": grad_norm(self.critic1.parameters()).item(),
            "grad_norm/critic2_actor": grad_norm(self.critic2.parameters()).item(),
            "grad_norm/critic1_td": norm1.item(),
            "grad_norm/critic2_td": norm2.item(),
            "loss/orth_critic1": orth1_loss.item(),
            "loss/orth_critic2": orth2_loss.item(),
            "loss/orth_actor": self._last_orth,
        }

    def exploration_noise(self, act: Union[np.ndarray, Batch], batch: Batch) -> Union[np.ndarray, Batch]:
        if self._noise_d is not None:
            bsz, num = act["retrace"].shape
            rand_mask = np.random.rand(bsz) < self._noise_d  # eps
            rand_dist = softmax(np.random.rand(bsz, num), axis=1)
            act["retrace"][rand_mask] = rand_dist[rand_mask]
        if self._noise_c is not None:
            act["direction"] = np.clip(act["direction"] + self._noise_c(act["direction"].shape), -1, 1)
        return act
