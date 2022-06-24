import os
import atexit
import logging
from pydoc import locate
from core.exps import EXPS
from omegaconf import DictConfig
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import tianshou as ts
from tianshou.env import DummyVectorEnv
from tianshou.data import Collector, PrioritizedVectorReplayBuffer, VectorReplayBuffer
from tianshou.utils import BasicLogger
from tianshou.exploration import GaussianNoise
from pytorch_lightning.callbacks.finetuning import BaseFinetuning

# from core.policy.ssdqn import SS_MixIn
# from core.policy.horizont import Horizont
from core.policy.hybrid_td3 import HybridTD3Policy
from core.policy.networks import TSDP
from core.policy.agents.hybrid import HybridActor, HybridCritic
from core.exps.utils import load_env_v2, create_experiment_dir, save_hparams
from core.exps.utils import ModelCheckpoint


@EXPS.register_module("HYBRID_TD3", force=True)
class HybridTD3Engine:
    def __init__(self, config):
        self.state = {}
        self.status = {}

        assert str(ts.__version__).endswith("+"), "import 'frameworks' before running"
        assert config.version == "2.3.0", "Config file incompatible with the training script"
        self.register(config=config)

    @property
    def stages(self):
        return [
            "init_experiment",
            "load_env",
            "create_env",
            "create_policy",
            "create_collector",
            "run_trainer",
        ]

    def train(self):
        self.run("run_trainer")

    def run(self, stage, reload=0):
        assert stage in self.stages
        rank = self.stages.index(stage)
        if (rank > 0) and ((reload > 1) or (self.stages[rank - 1] not in self.status)):
            self.run(self.stages[rank - 1], reload=reload - 1)
        elif (rank == 0) and reload:  # we're at the base level
            self.reset()
        getattr(self, stage)(**self.state)
        self.status[stage] = True

    def reset(self):
        self.state = {k: v for k, v in self.state.items() if k == "config"}

    def register(self, **kwargs):
        self.state.update(**kwargs)

    def init_experiment(self, config, **kwargs):
        print("Initializing experiment...")
        create_experiment_dir(config.experiment)
        save_hparams(config)

    def load_env(self, config, **kwargs):
        print("Loading envs...")
        train_env_udf, test_env_udf = load_env_v2(config)
        self.register(train_env_udf=train_env_udf, test_env_udf=test_env_udf)

    def create_env(self, config, train_env_udf, test_env_udf, **kwargs):
        print("Creating envs...")

        cfg = config.collector
        train_envs = DummyVectorEnv([train_env_udf for _ in range(cfg.num_train_envs)])
        test_envs = DummyVectorEnv([test_env_udf for _ in range(cfg.num_test_envs)])
        train_envs.seed(cfg.train_seed)
        test_envs.seed(cfg.test_seed)

        self.register(train_envs=train_envs, test_envs=test_envs)

    def create_policy(self, config, train_envs, **kwargs):
        print("Loading policy...")

        # NETWORKS
        cfg = config.policy
        observation_shape = train_envs.observation_space[0].shape
        action_space = train_envs.action_space[0]

        backbone_cfg = cfg.arch.backbone
        backbone = locate(backbone_cfg.name)(**backbone_cfg.args)
        finetune_callback = None

        # if backbone_cfg.finetune:
        #     finetune = backbone_cfg.finetune
        #     assert (finetune == "freeze") or isinstance(finetune, int)
        #     BaseFinetuning.freeze(backbone, train_bn=False)
        #     if isinstance(finetune, int):

        #         def add_params_callback(epoch, optimizer):
        #             if epoch == finetune:
        #                 BaseFinetuning.unfreeze_and_add_param_group(
        #                     modules=backbone, optimizer=optimizer, train_bn=True, lr=backbone_cfg.lr
        #                 )

        #         finetune_callback = add_params_callback

        device = None if cfg.device == "dp" else cfg.device
        # agent_kwargs = {k: v for k, v in cfg.arch.items() if k != "backbone"}

        for p in backbone.parameters():
            p.requires_grad = False

        actor = HybridActor(
            preprocess_net=backbone,
            action_space=action_space,
            device=device,
            preprocess_net_output_dim=cfg.arch.backbone.args.feature_vector_size,
            softmax_output=True,
            conditioned_sigma=False,
            unbounded=False,
            no_grad_backbone=True,
        )
        actor_optim = locate(cfg.opt.actor.name)(filter(lambda p: p.requires_grad, actor.parameters()), **cfg.opt.actor.args)

        for p in backbone.parameters():
            p.requires_grad = True

        # backbone_c1 = deepcopy(backbone)
        backbone_c1 = backbone
        critic1 = HybridCritic(
            preprocess_net=backbone_c1,
            action_space=action_space,
            device=device,
            preprocess_net_output_dim=cfg.arch.backbone.args.feature_vector_size,
            discrete_action_index=False,
        )
        critic1_optim = locate(cfg.opt.critic1.name)(critic1.parameters(), **cfg.opt.critic1.args)

        # backbone_c2 = deepcopy(backbone)
        backbone_c2 = backbone
        critic2 = HybridCritic(
            preprocess_net=backbone_c2,
            action_space=action_space,
            device=device,
            preprocess_net_output_dim=cfg.arch.backbone.args.feature_vector_size,
            discrete_action_index=False,
        )
        critic2_optim = locate(cfg.opt.critic2.name)(critic2.parameters(), **cfg.opt.critic2.args)

        if cfg.device == "dp":
            if cfg.device_ids:
                ids = cfg.device_ids
                if isinstance(ids, str):
                    ids = [int(x) for x in ids.split(",")]
                print("Using GPU IDs: ", ids)
            else:
                ids = None
            src_device = ids[0] if ids else None
            actor = TSDP(actor, device_ids=ids).cuda(src_device)
            critic1 = TSDP(critic1, device_ids=ids).cuda(src_device)
            critic2 = TSDP(critic2, device_ids=ids).cuda(src_device)
        else:
            actor.to(device)
            critic1.to(device)
            critic2.to(device)

        policy = HybridTD3Policy(
            actor,
            actor_optim,
            critic1,
            critic1_optim,
            critic2,
            critic2_optim,
            **cfg.td3,
        )

        if cfg.checkpoint.filepath:
            ckpt = torch.load(cfg.checkpoint.filepath, map_location=(device or "cuda"))
            state_dict = ckpt["model_state_dict"]
            try:
                policy.load_state_dict(state_dict, strict=cfg.checkpoint.strict)
            except:
                if device in ["cpu", "cuda"]:
                    state_dict = OrderedDict((k.replace("module.", "", 1), v) for k, v in state_dict.items())
                    policy.load_state_dict(state_dict, strict=cfg.checkpoint.strict)
            print("Checkpoint loaded!")

        self.register(policy=policy, optim=None, finetune_callback=finetune_callback)

    def create_collector(self, config, train_envs, test_envs, policy, **kwargs):
        print("Creating collectors...")
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
        test_collector = Collector(policy, test_envs, exploration_noise=False)

        self.register(train_collector=train_collector, test_collector=test_collector)

    def run_trainer(self, config, train_collector, test_collector, policy, optim, finetune_callback, **kwargs):
        print("Starting trainer...")

        # ERROR HANDLER
        def dump():
            print(":: TERMINATION SIGNAL RECIEVED ::")
            train_collector.env.save(render=False)
            test_collector.env.save(render=False)

        atexit.register(dump)

        # TRAINER
        cfg = config.trainer
        checkpoint_path = os.path.join(cfg.experiment_dir, "checkpoints")
        log_path = os.path.join(cfg.experiment_dir, "logs/train")
        os.makedirs(checkpoint_path, exist_ok=True)
        os.makedirs(log_path, exist_ok=True)
        writer = SummaryWriter(log_path)
        logger = BasicLogger(writer, update_interval=50)

        # checkpointer
        ckpt_args = dict(dirpath=checkpoint_path)
        ckpt_args.update(**cfg.checkpointer)
        model_checkpoint = ModelCheckpoint(**ckpt_args)

        # scheduler
        if cfg.lr_scheduler:
            scheduler = locate(cfg.lr_scheduler.name)(optim, **cfg.lr_scheduler.args)
        else:
            scheduler = None

        def save_fn(policy, epoch, metrics):
            model_checkpoint.save_checkpoint(policy, epoch=epoch, metrics=metrics, optimizer=optim)

        def stop_fn(epoch, metrics):
            return metrics["rew_avg"] >= cfg.reward_threshold

        def train_fn(epoch, env_step):
            policy.eval()
            
            epoch = epoch - 1  # since train_fn is called at the beginning

            # eps annnealing
            # if epoch < cfg.train.sched[0]:
            #     policy.set_eps(cfg.train.eps)
            # elif epoch <= cfg.train.sched[1]:
            #     eps = cfg.train.eps * (
            #         1 - 0.95 * ((epoch - cfg.train.sched[0]) / (cfg.train.sched[1] - cfg.train.sched[0]))
            #     )
            #     policy.set_eps(eps)
            # else:
            #     policy.set_eps(0.05)

            # lr scheduler
            if epoch and scheduler:
                scheduler.step()
            # finetune
            if finetune_callback:
                finetune_callback(epoch, optim)

        def test_fn(epoch, env_step):
            pass
            # policy.set_eps(cfg.test.eps)

        result = ts.trainer.offpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            train_fn=train_fn,
            test_fn=test_fn,
            stop_fn=stop_fn,
            save_fn=save_fn,
            logger=logger,
            **cfg.args,
        )
        print(result)
