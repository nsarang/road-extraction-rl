import torch
import numpy as np
from copy import deepcopy
from torch.distributions import Independent, Normal, Categorical
from typing import Any, Dict, Tuple, Union, Optional

from tianshou.policy import DDPGPolicy
from tianshou.exploration import BaseNoise
from tianshou.data import Batch, ReplayBuffer, to_torch_as
from core.optim.autoclip import grad_norm


class HybridSACPolicy(DDPGPolicy):
    """Implementation of Soft Actor-Critic. arXiv:1812.05905.

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
    :param (float, torch.Tensor, torch.optim.Optimizer) or float alpha: entropy
        regularization coefficient. Default to 0.2.
        If a tuple (target_entropy, log_alpha, alpha_optim) is provided, then
        alpha is automatatically tuned.
    :param bool reward_normalization: normalize the reward to Normal(0, 1).
        Default to False.
    :param BaseNoise exploration_noise: add a noise to action for exploration.
        Default to None. This is useful when solving hard-exploration problem.
    :param bool deterministic_eval: whether to use deterministic action (mean
        of Gaussian policy) instead of stochastic action sampled by the policy.
        Default to True.
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
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
        reward_normalization: bool = False,
        estimation_step: int = 1,
        exploration_noise: Optional[BaseNoise] = None,
        deterministic_eval: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(None, None, None, None, tau, gamma, None, reward_normalization, estimation_step, **kwargs)
        self.actor, self.actor_optim = actor, actor_optim
        self.critic1, self.critic1_old = critic1, deepcopy(critic1)
        self.critic1_old.eval()
        self.critic1_optim = critic1_optim
        self.critic2, self.critic2_old = critic2, deepcopy(critic2)
        self.critic2_old.eval()
        self.critic2_optim = critic2_optim
        self._noise = exploration_noise

        self._is_auto_alpha = False
        if len(alpha) == 3:
            self._is_auto_alpha = True
            (
                (self._target_entropy_d, self._target_entropy_c),
                (self._log_alpha_d, self._log_alpha_c),
                self._alpha_optim,
            ) = alpha

            assert self._log_alpha_d.shape == torch.Size([1]) and self._log_alpha_d.requires_grad
            assert self._log_alpha_c.shape == torch.Size([1]) and self._log_alpha_c.requires_grad

            self._alpha_d = self._log_alpha_d.detach().exp()
            self._alpha_c = self._log_alpha_c.detach().exp()
        else:
            self._alpha_d = alpha[0]
            self._alpha_c = alpha[1]

        self._deterministic_eval = deterministic_eval
        self.__eps = np.finfo(np.float32).eps.item()

    def train(self, mode: bool = True) -> "HybridSACPolicy":
        self.training = mode
        self.actor.train(mode)
        self.critic1.train(mode)
        self.critic2.train(mode)
        return self

    def sync_weight(self) -> None:
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
        (logits_disc, logits_cont), h = self.actor(obs, state=state, info=batch.info)

        dist_retrace = Categorical(logits=logits_disc)
        dist_direction = Independent(Normal(*logits_cont), 1)
        if self._deterministic_eval and not self.training:
            act_retrace = torch.argmax(logits_disc, dim=1)
            act_direction = logits_cont[0]
        else:
            act_retrace = dist_retrace.sample()
            act_direction = dist_direction.rsample()

        log_prob_cont = dist_direction.log_prob(act_direction).unsqueeze(-1)
        # apply correction for Tanh squashing when computing logprob from Gaussian
        # You can check out the original SAC paper (arXiv 1801.01290): Eq 21.
        # in appendix C to get some understanding of this equation.
        if self.action_scaling and self.action_space is not None:
            action_scale = to_torch_as((self.action_space.high - self.action_space.low) / 2.0, act_direction)
        else:
            action_scale = 1.0  # type: ignore
        squashed_action = torch.tanh(act_direction)
        log_prob_cont = log_prob_cont - torch.log(action_scale * (1 - squashed_action.pow(2)) + self.__eps).sum(
            -1, keepdim=True
        )

        entropy_cont = -log_prob_cont.reshape(-1, 1)
        entropy_disc = dist_retrace.entropy().reshape(-1, 1)
        entropy = self._alpha_c * entropy_cont + self._alpha_d * entropy_disc
        act = dict(retrace=act_retrace, direction=squashed_action)

        with torch.no_grad():
            target_q = torch.min(
                self.critic1_old(obs, act)[0],
                self.critic2_old(obs, act)[0],
            )

        return Batch(
            act=act,
            logits_disc=logits_disc,
            logits_cont=torch.stack(logits_cont, dim=2),
            value=target_q,
            entropy=entropy,
            dist_retrace=dist_retrace,
            dist_direction=dist_direction,
            entropy_d=entropy_disc,
            entropy_c=entropy_cont,
            state=h,
        )

    def _target_q(self, buffer: ReplayBuffer, indice: np.ndarray) -> torch.Tensor:
        batch = buffer[indice]  # batch.obs: s_{t+n}
        obs_next_result = self(batch, input="obs_next")
        a_ = obs_next_result.act
        target_q = (
            torch.min(
                self.critic1_old(batch.obs_next, a_)[0],
                self.critic2_old(batch.obs_next, a_)[0],
            )
            + obs_next_result.entropy
        )
        return target_q

    @staticmethod
    def _mse_optimizer(
        batch: Batch,
        critic: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A simple wrapper script for updating critic network."""
        weight = getattr(batch, "weight", 1.0)
        current_q = critic(batch.obs, batch.act)[0].flatten()
        target_q = batch.returns.flatten()
        td = current_q - target_q
        # critic_loss = F.mse_loss(current_q1, target_q)
        critic_loss = (td.pow(2) * weight).mean()
        optimizer.zero_grad()
        critic_loss.backward()
        optimizer.step()
        norm = grad_norm(critic.parameters())
        return td, critic_loss, norm

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        # critic 1&2
        td1, critic1_loss, norm1 = self._mse_optimizer(batch, self.critic1, self.critic1_optim)
        td2, critic2_loss, norm2 = self._mse_optimizer(batch, self.critic2, self.critic2_optim)
        batch.weight = (td1 + td2) / 2.0  # prio-buffer

        # actor
        obs_result = self(batch)
        a = obs_result.act
        current_q1a, (feats1, act1) = self.critic1(batch.obs, a)
        current_q2a, (feats2, act2) = self.critic2(batch.obs, a)
        current_q1a = current_q1a.flatten()
        current_q2a = current_q2a.flatten()

        if "logger" in kwargs:
            logger, step = kwargs["logger"], kwargs["step"]
            logger.writer.add_histogram("act/critic1", act1, step)
            logger.writer.add_histogram("act/critic2", act2, step)
            logger.writer.add_histogram("feats/critic1", feats1, step)
            logger.writer.add_histogram("feats/critic2", feats2, step)

        actor_loss = -(torch.min(current_q1a, current_q2a) + obs_result.entropy).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self._is_auto_alpha:
            entropy_diff_d = self._target_entropy_d - obs_result.entropy_d.detach()
            alpha_loss_d = -(self._log_alpha_d * entropy_diff_d).mean()
            entropy_diff_c = self._target_entropy_c - obs_result.entropy_c.detach()
            alpha_loss_c = -(self._log_alpha_c * entropy_diff_c).mean()
            alpha_loss = alpha_loss_d + alpha_loss_c

            self._alpha_optim.zero_grad()
            alpha_loss.backward()
            self._alpha_optim.step()
            self._alpha_d = self._log_alpha_d.detach().exp()
            self._alpha_c = self._log_alpha_c.detach().exp()

        self.sync_weight()

        result = {
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item(),
            "entropy/total": obs_result.entropy.mean().item(),
            "entropy/discrete": obs_result.entropy_d.mean().item(),
            "entropy/continuous": obs_result.entropy_c.mean().item(),
            "grad_norm/actor": grad_norm(self.actor.parameters()).item(),
            "grad_norm/critic1_actor": grad_norm(self.critic1.parameters()).item(),
            "grad_norm/critic2_actor": grad_norm(self.critic2.parameters()).item(),
            "grad_norm/critic1_td": norm1.item(),
            "grad_norm/critic2_td": norm2.item(),
        }
        if self._is_auto_alpha:
            result["loss/alpha_d"] = alpha_loss_d.item()
            result["loss/alpha_c"] = alpha_loss_c.item()
            result["alpha/discrete"] = self._alpha_d.item()
            result["alpha/continuous"] = self._alpha_c.item()

        return result

    def exploration_noise(self, act: Union[np.ndarray, Batch], batch: Batch) -> Union[np.ndarray, Batch]:
        if self._noise is None:
            return act
        act["direction"] += self._noise["direction"](act["direction"].shape)
        return act
