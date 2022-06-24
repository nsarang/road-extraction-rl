import os
from pydoc import locate

from core.data import load_data
from core.env.factory import create_env

import torch
import torch.nn as nn
import torchvision.models as models
from torch.distributions import Independent, Normal
from torch.utils.tensorboard import SummaryWriter

import tianshou as ts
from tianshou.env import DummyVectorEnv
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.utils.net.discrete import Actor, Critic as DCritic
from tianshou.data import Collector, PrioritizedReplayBuffer, ReplayBuffer
from core.policy.networks import TSDP
from core.env.actions import DiscreteSpace, ContinuousSpace
from .utils import load_env, create_experiment_dir, save_hparams


def a2c_trainer(config):
    # CORE
    create_experiment_dir(config)
    save_hparams(config)

    # ENV
    train_env_udf, test_env_udf = load_env(config)

    # CREATE ENVS
    cfg = config.collector
    train_envs = DummyVectorEnv([train_env_udf for _ in range(cfg.num_train_envs)])
    test_envs = DummyVectorEnv([test_env_udf for _ in range(cfg.num_test_envs)])
    train_envs.seed(cfg.train_seed)
    test_envs.seed(cfg.test_seed)

    # NETWORKS
    cfg = config.policy
    observation_shape = train_envs.observation_space[0].shape
    action_space = train_envs.action_space[0]

    backbone = locate(cfg.backbone.name)(**cfg.backbone.args)

    # POLICY
    def dist_normal(*logits):
        return Independent(Normal(*logits), 1)

    device = None if cfg.device == "dp" else cfg.device
    if isinstance(train_envs.workers[0].env.action_scheme, ContinuousSpace):
        actor = ActorProb(
            preprocess_net=backbone,
            action_shape=action_space.shape,
            max_action=action_space.high[0],
            device=device,
            preprocess_net_output_dim=cfg.backbone.args.feature_vector_size,
        )
        critic = Critic(
            preprocess_net=backbone,
            device=device,
            preprocess_net_output_dim=cfg.backbone.args.feature_vector_size,
        )
        dist = dist_normal
    else:
        actor = Actor(
            preprocess_net=backbone,
            action_shape=action_space.n,
            device=device,
            preprocess_net_output_dim=cfg.backbone.args.feature_vector_size,
        )
        critic = DCritic(
            preprocess_net=backbone,
            device=device,
            preprocess_net_output_dim=cfg.backbone.args.feature_vector_size,
        )
        dist = torch.distributions.Categorical

    if cfg.device == "dp":
        actor = TSDP(actor).cuda()
        critic = TSDP(critic).cuda()
    else:
        actor.to(device)
        critic.to(device)

    optim = torch.optim.AdamW(
        list(actor.parameters()) + list(critic.parameters()), **cfg.optimizer
    )
    policy = ts.policy.A2CPolicy(actor, critic, optim, dist, **cfg.ppo)

    # COLLECTOR
    cfg = config.collector
    if cfg.buffer_type.upper() == "PERB":
        buffer = PrioritizedReplayBuffer(cfg.buffer_size, cfg.alpha, cfg.beta)
    else:
        buffer = ReplayBuffer(cfg.buffer_size)

    train_collector = Collector(policy, train_envs, buffer)
    test_collector = Collector(
        policy,
        test_envs,
        # action_noise=ts.exploration.GaussianNoise(sigma=cfg.test_noise),
    )

    # TRAINER
    cfg = config.trainer
    checkpoint_path = os.path.join(cfg.experiment_dir, "checkpoints")
    log_path = os.path.join(cfg.experiment_dir, "logs/train")
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_path)

    def save_fn(policy):
        torch.save(policy, os.path.join(checkpoint_path, "latest_policy.pth"))

    def stop_fn(mean_rewards):
        return mean_rewards >= cfg.reward_threshold

    # trainer
    result = ts.trainer.onpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        stop_fn=stop_fn,
        save_fn=save_fn,
        writer=writer,
        **cfg.args,
    )
    assert stop_fn(result["best_reward"])
