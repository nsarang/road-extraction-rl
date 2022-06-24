import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import pytorch_lightning as pl

from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger

from core.policy.networks import LitRoadS


def rss_trainer(config=None):
    experiment = "w-smooth"

    # seed
    pl.seed_everything(7)

    lr_logger = LearningRateMonitor(logging_interval="step")  # log the learning rate
    logger = TensorBoardLogger("lightning_logs", name=experiment)  # log to tensorboard
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filepath=os.path.join(
            logger.log_dir,
            "checkpoints",
            "LitRoadS-epoch={epoch:02d}--val_loss={val_loss:.4f}",
        ),
        save_top_k=7,
        mode="min",
    )

    model = LitRoadS(
        learning_rate=5e-4,
        val_ratio=0.1,
        batch_size=2,
        crop_size=(512, 512),
        log_interval=10,
        log_val_interval=5,
        num_workers=0,
    )
    num_gpus = torch.cuda.device_count()
    accelerator = "ddp_spawn" if num_gpus > 0 else None

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, lr_logger],
        logger=logger,
        # fast_dev_run=30,
        gpus=num_gpus,
        accelerator=accelerator,
        sync_batchnorm=False,
        num_sanity_val_steps=4,
        max_epochs=50,
        flush_logs_every_n_steps=10,
        progress_bar_refresh_rate=1,
        # weights_summary="full",
    )

    trainer.fit(model)