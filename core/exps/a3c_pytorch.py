import os
import time

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from .utils import load_env, create_experiment_dir, save_hparams

from a3c_pytorch.environment import atari_env
from a3c_pytorch.model import A3Clstm
from a3c_pytorch.train import train
from a3c_pytorch.test import test
from a3c_pytorch.shared_optim import SharedRMSprop, SharedAdam


def trainer(config):
    # CORE
    create_experiment_dir(config)
    save_hparams(config)
    os.environ["OMP_NUM_THREADS"] = "1"

    # ENV
    train_env_udf, test_env_udf = load_env(config)

    # MAIN
    args = config.trainer

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_model_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    if args.gpu_ids == -1:
        args.gpu_ids = [-1]
    else:
        torch.cuda.manual_seed(args.seed)
        mp.set_start_method("spawn")

    env = train_env_udf()
    print(env.action_space)
    shared_model = A3Clstm(env.observation_space.shape[0], env.action_space)
    if args.load:
        fp = os.path.join(args.load_model_dir, f"{args.env}.dat")
        saved_state = torch.load(fp, map_location=lambda storage, loc: storage)
        shared_model.load_state_dict(saved_state)
    shared_model.share_memory()

    if args.shared_optimizer:
        if args.optimizer == "RMSprop":
            optimizer = SharedRMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == "Adam":
            optimizer = SharedAdam(
                shared_model.parameters(), lr=args.lr, amsgrad=args.amsgrad
            )
        optimizer.share_memory()
    else:
        optimizer = None

    processes = []

    p = mp.Process(target=test, args=(args, shared_model, test_env_udf), daemon=True)
    p.start()
    processes.append(p)
    time.sleep(0.1)
    for rank in range(0, args.workers):
        p = mp.Process(
            target=train,
            args=(rank, args, shared_model, optimizer, train_env_udf),
            daemon=True,
        )
        p.start()
        processes.append(p)
        time.sleep(0.1)
    for p in processes:
        time.sleep(0.1)
        p.join()
