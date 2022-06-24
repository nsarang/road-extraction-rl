import os
import atexit
import logging
import logging.config
import yaml
from pydoc import locate
from core.exps import EXPS
from omegaconf import DictConfig
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import tianshou as ts
from tianshou.env import DummyVectorEnv
from tianshou.data import Collector, PrioritizedVectorReplayBuffer, VectorReplayBuffer
from tianshou.utils import TensorboardLogger
from pytorch_lightning.callbacks.finetuning import BaseFinetuning

from core.policy.ssdqn import SS_MixIn
from core.policy.horizont import Horizont
from core.policy.networks import TSDP
from core.policy.agents.transformer import TransEfficient
from core.exps.utils import load_env_v2, create_experiment_dir, save_hparams
from core.exps.utils import ModelCheckpoint


@EXPS.register_module("DQN_SEQ", force=True)
class DQNSeqEngine:
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

    def run(self, stage, reload=False):
        assert stage in self.stages
        rank = self.stages.index(stage)
        if (rank > 0) and (reload or (self.stages[rank - 1] not in self.status)):
            self.run(self.stages[rank - 1], reload=reload)
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
        cfg = config.experiment
        create_experiment_dir(cfg)
        print("Experiment dir:", cfg.dir)
        save_hparams(config)
        if cfg.logger:
            logpath = os.path.join(cfg.dir, "logs")
            os.makedirs(logpath, exist_ok=True)
            with open(cfg.logger) as f:
                content = f.read()
                content = content.format(logdir=logpath)
                config = yaml.safe_load(content)
                logging.config.dictConfig(config)

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

        if backbone_cfg.finetune:
            finetune = backbone_cfg.finetune
            assert (finetune == "freeze") or isinstance(finetune, int)
            BaseFinetuning.freeze(backbone, train_bn=False)
            if isinstance(finetune, int):

                def add_params_callback(epoch, optimizer):
                    if epoch == finetune:
                        BaseFinetuning.unfreeze_and_add_param_group(
                            modules=backbone, optimizer=optimizer, train_bn=True, lr=backbone_cfg.lr
                        )

                finetune_callback = add_params_callback

        device = None if cfg.device == "dp" else cfg.device
        agent_kwargs = {k: v for k, v in cfg.arch.items() if k != "backbone"}
        model = TransEfficient(
            preprocess_net=backbone,
            action_shape=action_space.n,
            output_shape=(action_space.n, len(cfg.dqn.horizons)) if cfg.horizont else (),
            device=device,
            preprocess_net_output_dim=cfg.arch.backbone.args.feature_vector_size,
            softmax_output=False,
            **agent_kwargs,
        )

        if cfg.device == "dp":
            if cfg.device_ids:
                ids = cfg.device_ids
                if isinstance(ids, str):
                    ids = [int(x) for x in ids.split(",")]
            else:
                ids = list(range(torch.cuda.device_count()))
            print("Using GPU IDs: ", ids)
            src_device = ids[0]
            model = TSDP(model, device_ids=ids).cuda(src_device)
        else:
            model.to(device)

        # POLICY
        if cfg.horizont:
            policy_cls = Horizont
        else:
            policy_cls = ts.policy.DQNPolicy
        if cfg.ss_reg:
            name = f"SS_{policy_cls.__class__.__name__}"
            policy_cls = type(name, (SS_MixIn, policy_cls), {})

        optim = locate(cfg.optimizer.name)(filter(lambda p: p.requires_grad, model.parameters()), **cfg.optimizer.args)
        policy = policy_cls(model=model, optim=optim, **cfg.dqn)
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

            if cfg.checkpoint.load_optimizer:
                try:
                    optim.load_state_dict(ckpt["optimizer_state_dict"], strict=cfg.checkpoint.strict)
                except:
                    logging.warn("failed to load optimzer's state dict")

        self.register(policy=policy, optim=optim, finetune_callback=finetune_callback)

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
        test_collector = Collector(policy, test_envs, exploration_noise=True)

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
        logger = TensorboardLogger(writer, train_interval=10, update_interval=10)

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
            epoch = epoch - 1  # since train_fn is called at the beginning

            # eps annnealing
            if epoch < cfg.train.sched[0]:
                policy.set_eps(cfg.train.eps)
            elif epoch <= cfg.train.sched[1]:
                eps = cfg.train.eps * (
                    1 - 0.99 * ((epoch - cfg.train.sched[0]) / (cfg.train.sched[1] - cfg.train.sched[0]))
                )
                policy.set_eps(eps)
            else:
                policy.set_eps(0.01)

            # lr scheduler
            if epoch and scheduler:
                scheduler.step()
            # finetune
            if finetune_callback:
                finetune_callback(epoch, optim)

        def test_fn(epoch, env_step):
            policy.set_eps(cfg.test.eps)

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
