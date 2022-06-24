from typing import Any, Callable, Dict, Iterable, List, Tuple, Union

from core.exps import EXPS

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
import pytorch_lightning as pl
import matplotlib as mpl
from matplotlib import pyplot as plt

try:
    from encoding.models.sseg import DeepLabV3
except:
    import logging

    logging.warn("Failed to load torch-encoding")

from core.data import rss
from core.optim import Ranger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from torchmetrics import (
    Metric,
    Accuracy,
    FBeta,
    AveragePrecision,
    Precision,
    Recall,
)
from kornia.losses import DiceLoss, FocalLoss
from core.metrics.cross_entropy import InstanceWeightedCrossEntropyLoss

from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger


class LitRoadS(pl.LightningModule):
    def __init__(
        self,
        backbone="resnest50",
        data_dir="data/segmentation",
        val_ratio=0.1,
        crop_size=(256, 256),
        num_classes=2,
        learning_rate=1e-3,
        loss=None,
        logging_metrics: nn.ModuleList = None,
        batch_size=4,
        num_workers=4,
        use_aux=True,
        log_interval: Union[int, float] = 10,
        log_val_interval: Union[int, float] = None,
        seed=None,
    ):
        super().__init__()
        if seed is not None:
            pl.seed_everything(seed)

        if logging_metrics is None:
            logging_metrics = nn.ModuleList(
                [
                    Accuracy(),
                    FBeta(num_classes=num_classes, average="macro"),
                    Precision(num_classes=num_classes, average="macro"),
                    Recall(num_classes=num_classes, average="macro"),
                ]
            )
        if loss is None:
            # loss = soft_cldice_loss
            loss = InstanceWeightedCrossEntropyLoss(num_classes=num_classes)
            # loss = IoULoss(do_bg=False)
            # loss = SoftDiceLoss(batch_dice=True, do_bg=False)
            # loss = DC_and_topk_loss()
            # loss = FocalLoss(alpha=1, gamma=3, reduction="mean")
            # loss = DiceLoss()
            # loss = SegmentationLosses(
            #     nclass=num_classes,
            #     aux=use_aux,
            #     aux_weight=0.2,
            #     se_loss=False,
            #     se_weight=0.2,
            #     # weight=torch.tensor([1.0, 15.0]),
            # )
        if log_val_interval is None:
            log_val_interval = log_interval

        self.loss = loss
        self.logging_metrics = nn.ModuleList([l for l in logging_metrics])

        self.save_hyperparameters()
        self._load_model()
        self._load_datasets()
        self._configure_plotter()

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        """
        Train on batch.
        """
        x, y = batch
        loss = self.step(x, y, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.step(x, y, batch_idx)  # log loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def step(
        self,
        x: Dict[str, torch.Tensor],
        y: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        **kwargs,
    ):
        """
        run at each step for training or validation
        """
        # extract data and run model
        out = self(x)
        y = y.long()
        loss = self.loss(out[0], y) + 0.1 * self.loss(out[1], y)

        self.log_metrics(x, y, out)
        if self.log_interval > 0:
            self.log_prediction(x, y, out, batch_idx)
        return loss

    def configure_optimizers(self):
        opt = Ranger(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.95, 0.999),
            use_gc=False,
            weight_decay=1e-3,
        )
        sched = {
            "scheduler": ReduceLROnPlateau(opt, mode="min", factor=0.8, patience=2, threshold=1e-6),
            "monitor": "val_loss",
            "interval": "epoch",
        }
        return [opt], [sched]

    def plot_prediction(
        self,
        x: Dict[str, torch.Tensor],
        y: torch.Tensor,
        out: Dict[str, torch.Tensor],
        idx: int = 0,
        add_loss_to_title: Union[Metric, torch.Tensor, bool] = False,
    ) -> plt.Figure:
        """
        Plot prediction of prediction vs actuals

        Args:
            x: network input
            out: network output
            idx: index of prediction to plot
            add_loss_to_title: if to add loss to title or loss function to calculate. Can be either metrics,
                bool indicating if to use loss metric or tensor which contains losses for all samples.
                Calcualted losses are determined without weights. Default to False.
            show_future_observed: if to show actuals for future. Defaults to True.
            ax: matplotlib axes to plot on

        Returns:
            matplotlib figure
        """
        # TODO: fix for ddp_spawn
        self._configure_plotter()

        x = x[idx].detach().cpu().numpy()
        x = np.transpose(x, (1, 2, 0))
        x = np.clip((x * (0.229, 0.224, 0.225)) + (0.485, 0.456, 0.406), 0, 1)

        y = y[idx].detach().cpu().numpy()

        pred1 = F.softmax(out[0], dim=1)
        out1 = pred1[idx].detach().cpu().numpy()
        y_pred = out1[1, ...]  # HEATMAP

        pred2 = F.softmax(out[1], dim=1)
        out2 = pred2[idx].detach().cpu().numpy()
        aux = out2[1, ...]

        fig, axes = plt.subplots(2, 2, figsize=(6, 6))

        fig.suptitle("Prediction Plot")

        axes[0, 0].imshow(x)
        axes[0, 0].set_title("Input")

        axes[0, 1].imshow(y, cmap="gray")
        axes[0, 1].set_title("Groundtruth")

        axes[1, 1].imshow(y_pred, cmap="gray")
        axes[1, 1].set_title("Prediction")
        axes[1, 0].imshow(aux, cmap="gray")
        axes[1, 0].set_title("Aux")

        # TODO: add loss to tile
        fig.subplots_adjust(left=None, bottom=None, right=None, top=0.9, wspace=0.05, hspace=0.15)
        # fig.tight_layout(rect=[0, 0, 1, 0.9])
        return fig

    def log_prediction(
        self,
        x: Dict[str, torch.Tensor],
        y: torch.Tensor,
        out: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """
        Log metrics every training/validation step.

        Args:
            x (Dict[str, torch.Tensor]): x as passed to the network by the dataloader
            y (torch.Tensor): y as passed to the loss function by the dataloader
            out (Dict[str, torch.Tensor]): output of the network
            batch_idx (int): current batch index
        """
        # log single prediction figure
        if (batch_idx % self.log_interval == 0 or self.log_interval < 1.0) and self.log_interval > 0:
            if self.log_interval < 1.0:  # log multiple steps
                log_indices = torch.arange(
                    0,
                    self.hparams.batch_size,
                    max(1, round(self.log_interval * self.hparams.batch_size)),
                )
            else:
                log_indices = [0]

            for idx in log_indices:
                fig = self.plot_prediction(x, y, out, idx=idx, add_loss_to_title=True)
                tag = f"{['Val', 'Train'][self.training]} prediction"

                # if self.training:
                #     tag += f" of item {idx} in global batch {self.global_step}"
                # else:
                tag += f" of item {idx} in batch {batch_idx}"

                if isinstance(fig, (list, tuple)):
                    for idx, f in enumerate(fig):
                        self.logger.experiment.add_figure(
                            f"Target {idx} {tag}",
                            f,
                            global_step=self.global_step,
                        )
                        plt.close(f)
                else:
                    self.logger.experiment.add_figure(
                        tag,
                        fig,
                        global_step=self.global_step,
                    )
                    plt.close(fig)

    def log_metrics(
        self,
        x: Dict[str, torch.Tensor],
        y: torch.Tensor,
        out: Dict[str, torch.Tensor],
    ) -> None:
        """
        Log metrics every training/validation step.

        Args:
            x (Dict[str, torch.Tensor]): x as passed to the network by the dataloader
            y (torch.Tensor): y as passed to the loss function by the dataloader
            out (Dict[str, torch.Tensor]): output of the network
        """
        if self.hparams.use_aux:
            out = out[0]

        # logging losses - for each target
        for metric in self.logging_metrics:
            name = metric.__class__.__name__.lower()
            self.log(
                name=f"{name}/{['val', 'train'][self.training]}",
                value=metric(out, y),
                on_step=self.training,
                on_epoch=True,
            )

    @property
    def model(self):
        return self._model

    @property
    def log_interval(self) -> float:
        """
        Log interval depending if training or validating
        """
        if self.training:
            return self.hparams.log_interval
        else:
            return self.hparams.log_val_interval

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=True,
        )

    def _load_model(self):
        self.model = DeepLabV3(
            nclass=self.hparams.num_classes,
            backbone=self.hparams.backbone,
            aux=self.hparams.use_aux,
            se_loss=False,
        )

    def _load_datasets(self):
        image_dir = os.path.join(self.hparams.data_dir, "images")
        mask_dir = os.path.join(self.hparams.data_dir, "masks")
        filepaths = rss.RoadSegDataset._get_images_filepaths(image_dir, mask_dir)

        cities = list(set(img.name.split("_")[0] for img, mask in filepaths))
        np.random.shuffle(cities)
        cutoff = int(self.hparams.val_ratio * len(cities))
        train_cities = cities[cutoff:]
        val_cities = cities[:cutoff]
        print("train_cities", train_cities)
        print("val_cities", val_cities)

        train_filepaths = [(a, b) for a, b in filepaths if a.name.split("_")[0] in train_cities]
        val_filepaths = [(a, b) for a, b in filepaths if a.name.split("_")[0] in val_cities]

        train_transforms, test_transforms = rss.transforms_v1(self.hparams.crop_size)

        self.train_dataset = rss.RoadSegDataset(train_filepaths, transform=train_transforms)
        self.val_dataset = rss.RoadSegDataset(val_filepaths, transform=test_transforms)

    def _configure_plotter(self):
        DPI = 200
        mpl.rc("savefig", dpi=DPI)
        mpl.rcParams["figure.dpi"] = DPI
        mpl.rcParams["figure.figsize"] = 6.4, 4.8  # Default.
        mpl.rcParams["font.sans-serif"] = "Roboto"
        mpl.rcParams["font.family"] = "sans-serif"

        # Set title text color to dark gray (https://material.io/color) not black.
        TITLE_COLOR = "#212121"
        mpl.rcParams["text.color"] = TITLE_COLOR

        rc = {
            "axes.spines.left": False,
            "axes.spines.right": False,
            "axes.spines.bottom": False,
            "axes.spines.top": False,
            "xtick.bottom": False,
            "xtick.labelbottom": False,
            "ytick.labelleft": False,
            "ytick.left": False,
        }
        mpl.rcParams.update(rc)


@EXPS.register_module("Road_Res")
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