import os
from pydoc import locate


import torch
from torch.distributions import Independent, Normal
from torch.utils.tensorboard import SummaryWriter

import tianshou as ts
from tianshou.env import DummyVectorEnv
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.utils.net.discrete import Actor, Critic as DiscreteCritic
from tianshou.data import Collector, PrioritizedVectorReplayBuffer, VectorReplayBuffer
from tianshou.utils import BasicLogger
from core.policy.networks import TSDP
from core.env.actions import DiscreteSpace, ContinuousSpace
from .utils import load_env_v2, create_experiment_dir, save_hparams, ModelCheckpoint

from core.exps import EXPS


@EXPS.register_module("PPO")
def ppo_trainer(config):
    # VERSION
    assert str(ts.__version__).endswith("+"), "import 'frameworks' before running"
    assert config.version == "2.1.0", "Config file incompatible with the training script"
    # EXP
    create_experiment_dir(config.experiment)
    save_hparams(config)

    # ENV
    train_env_udf, test_env_udf = load_env_v2(config)
    
    print("Configs loaded")

    # CREATE ENVS
    cfg = config.collector
    train_envs = DummyVectorEnv([train_env_udf for _ in range(cfg.num_train_envs)])
    test_envs = DummyVectorEnv([test_env_udf for _ in range(cfg.num_test_envs)])
    train_envs.seed(cfg.train_seed)
    test_envs.seed(cfg.test_seed)

    print("Created envs")

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
            softmax_output=True,
        )
        critic = DiscreteCritic(
            preprocess_net=backbone,
            device=device,
            preprocess_net_output_dim=cfg.backbone.args.feature_vector_size,
        )
        dist = torch.distributions.Categorical

    if cfg.device == "dp":
        if cfg.device_ids:
            ids = cfg.device_ids
            if isinstance(ids, str):
                ids = [int(x) for x in ids.split(",")]
            print("Using GPU IDs: ", ids)
        else:
            ids = None
        src_device = f"cuda:{ids[0]}" if ids else "cuda"
        actor = TSDP(actor, device_ids=ids).to(src_device)
        critic = TSDP(critic, device_ids=ids).to(src_device)
    else:
        actor.to(device)
        critic.to(device)

    # # orthogonal initialization
    # for m in list(actor.modules()) + list(critic.modules()):
    #     if isinstance(m, torch.nn.Linear):
    #         torch.nn.init.orthogonal_(m.weight)
    #         torch.nn.init.zeros_(m.bias)
    optim = torch.optim.AdamW(list(actor.parameters()) + list(critic.parameters()), **cfg.optimizer)
    policy = ts.policy.PPOPolicy(actor, critic, optim, dist, **cfg.ppo)
    if cfg.checkpoint:
        policy.load_state_dict(torch.load(cfg.checkpoint, map_location=(device or src_device)))

    print("Policy loaded")
    
    # COLLECTOR
    cfg = config.collector
    if cfg.buffer_type.upper() == "PERB":
        buffer = PrioritizedVectorReplayBuffer(
            total_size=cfg.buffer_size,
            buffer_num=config.collector.num_train_envs,
            alpha=cfg.alpha,
            beta=cfg.beta,
        )
    else:
        buffer = VectorReplayBuffer(cfg.buffer_size, config.collector.num_train_envs)

    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)

    print("Collector loaded")

    # TRAINER
    cfg = config.trainer
    checkpoint_path = os.path.join(cfg.experiment_dir, "checkpoints")
    log_path = os.path.join(cfg.experiment_dir, "logs/train")
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_path)
    logger = BasicLogger(writer, update_interval=10)

    ckpt_args = dict(dirpath=checkpoint_path)
    ckpt_args.update(**cfg.checkpointer)
    model_checkpoint = ModelCheckpoint(**ckpt_args)

    def save_fn(policy, epoch, metrics):
        model_checkpoint.save_checkpoint(policy, epoch, metrics)

    def stop_fn(epoch, metrics):
        return metrics["rew_avg"] >= cfg.reward_threshold

    # trainer
    result = ts.trainer.onpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        stop_fn=stop_fn,
        save_fn=save_fn,
        logger=logger,
        **cfg.args,
    )
    print(result)
