from PIL.Image import NONE
import os
from pydoc import locate

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import tianshou as ts
from tianshou.env import DummyVectorEnv
from tianshou.data import Collector, PrioritizedVectorReplayBuffer, VectorReplayBuffer
from tianshou.utils.net.discrete import Actor
from tianshou.utils import BasicLogger

from core.policy.networks import TSDP
from core.policy.rdqn import RDQN
from core.policy.agents.multi_task import MultitaskActor
from core.exps.utils import load_env_v2, create_experiment_dir, save_hparams
from core.exps.utils import ModelCheckpoint

from core.metrics.cldice import soft_dice_cldice, soft_cldice
from core.metrics.cross_entropy import InstanceWeightedCrossEntropyLoss

from core.exps import EXPS


class CombinedSegLoss(nn.Module):
    def __init__(self, num_classes, alpha, wce_smooth=0.6):
        super().__init__()
        self.wce = InstanceWeightedCrossEntropyLoss(num_classes, wce_smooth)
        self.cldice = soft_cldice()
        self.alpha = alpha

    def forward(self, outputs, targets):
        return (1 - self.alpha) * self.wce(outputs, targets) + self.alpha * self.cldice(outputs, targets)


@EXPS.register_module("RDQN")
def dqn_trainer(config):
    # VERSION
    assert str(ts.__version__).endswith("+"), "import 'frameworks' before running"
    assert config.version == "2.2.0", "Config file incompatible with the training script"

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

    backbone = locate(cfg.arch.backbone.name)(**cfg.arch.backbone.args)
    device = None if cfg.device == "dp" else cfg.device
    model = MultitaskActor(
        preprocess_net=backbone,
        action_shape=action_space.n,
        device=device,
        mlp_input_dim=cfg.arch.mlp_input_dim,
        hidden_sizes=cfg.arch.hidden_sizes,
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
    # criterion = CombinedSegLoss(**cfg.segmentation_criterion)
    criterion = InstanceWeightedCrossEntropyLoss(num_classes=2, smooth=0.7)
    policy = RDQN(model=model, optim=optim, mask_criterion=criterion, **cfg.dqn)
    if cfg.checkpoint:
        policy.load_state_dict(torch.load(cfg.checkpoint, map_location=(device or "cuda")))

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
            eps = cfg.train.eps * (1 - 0.9 * ((epoch - cfg.train.sched[0]) / (cfg.train.sched[1] - cfg.train.sched[0])))
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
