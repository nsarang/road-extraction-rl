import os
from pydoc import locate
from core.exps import EXPS

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import tianshou as ts
from tianshou.env import DummyVectorEnv
from tianshou.data import Collector, PrioritizedVectorReplayBuffer, VectorReplayBuffer
from tianshou.utils.net.discrete import Actor
from tianshou.utils import BasicLogger

from core.policy.networks import TSDP
from core.exps.utils import load_env_v2, create_experiment_dir, save_hparams
from core.exps.utils import ModelCheckpoint

@EXPS.register_module("DQN_V1")
def dqn_trainer(config):
    # VERSION
    assert str(ts.__version__).endswith("+"), "import 'frameworks' before running"
    assert config.version == "2.1.0", "Config file incompatible with the training script"

    # EXP
    create_experiment_dir(config.experiment)
    save_hparams(config)

    # ENV
    train_env_udf, test_env_udf = load_env_v2(config)

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
    device = None if cfg.device == "dp" else cfg.device
    model = Actor(
        preprocess_net=backbone,
        action_shape=action_space.n,
        device=device,
        preprocess_net_output_dim=cfg.backbone.args.feature_vector_size,
        softmax_output=False,
    )

    if cfg.device == "dp":
        if cfg.device_ids:
            ids = cfg.device_ids
            if isinstance(ids, str):
                ids = [int(x) for x in ids.split(",")]
            print("Using GPU IDs: ", ids)
        else:
            ids = None
        src_device = ids[0] if ids else None
        model = TSDP(model, device_ids=ids).cuda(src_device)
    else:
        model.to(device)

    # POLICY
    optim = torch.optim.AdamW(list(model.parameters()), **cfg.optimizer)
    policy = ts.policy.DQNPolicy(model, optim, **cfg.dqn)
    if cfg.checkpoint:
        policy.load_state_dict(
            torch.load(cfg.checkpoint, map_location=(device or "cuda"))
        )

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

    # REWARD METRIC
    if cfg.reward_metric:
        reward_space = train_envs.reward_scheme[0].reward_space
        max_magnitude = max(abs(reward_space.low[0]), abs(reward_space.high[0]))
        exponent = cfg.reward_metric.exponent

        def reward_metric(r):
            sign = 1 if r > 0 else -1
            r = abs(r)
            if cfg.reward_metric.normalize:
                r /= max_magnitude
            r = r ** exponent
            return r
    else:
        reward_metric = None        

    # TRAINER
    cfg = config.trainer
    checkpoint_path = os.path.join(cfg.experiment_dir, "checkpoints")
    log_path = os.path.join(cfg.experiment_dir, "logs/train")
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_path)
    logger = BasicLogger(writer)

    ckpt_args = dict(dirpath=checkpoint_path)
    ckpt_args.update(**cfg.checkpointer)
    model_checkpoint = ModelCheckpoint(**ckpt_args)

    def save_fn(policy, epoch, metrics):
        model_checkpoint.save_checkpoint(policy, epoch, metrics)

    def stop_fn(epoch, metrics):
        return metrics["rew_avg"] >= cfg.reward_threshold

    def train_fn(epoch, env_step):
        # eps annnealing
        if epoch <= cfg.train.sched[0]:
            policy.set_eps(cfg.train.eps)
        elif epoch <= cfg.train.sched[1]:
            eps = cfg.train.eps * (
                1
                - 0.9
                * (
                    (epoch - cfg.train.sched[0])
                    / (cfg.train.sched[1] - cfg.train.sched[0])
                )
            )
            policy.set_eps(eps)
        else:
            policy.set_eps(0.1 * cfg.train.eps)

    def test_fn(epoch, env_step):
        policy.set_eps(cfg.test.eps)

    # trainer
    result = ts.trainer.offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_fn=save_fn,
        logger=logger,
        **cfg.args
    )
    print(result)
