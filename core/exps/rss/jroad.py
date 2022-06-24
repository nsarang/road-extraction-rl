from typing import Any, Callable, Dict, Iterable, List, Tuple, Union
from pytorch_lightning import plugins

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

import pytorch_lightning as pl
import torchmetrics as tm
from pytorch_lightning.plugins import DDPPlugin

from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger

from configs.defaults import TRAIN_SET, TEST_SET
from core.data import rss
from core.policy.networks.seg import MultiMaskDDRNet
from core.optim import Ranger
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import DataLoader
from kornia.losses import DiceLoss, FocalLoss
from core.metrics.cross_entropy import BatchWeightedCrossEntropyLoss
from core.metrics.cldice import soft_skel
from core.optim.autoclip import AutoClip
from core.exps import EXPS


class JRoadMultiS(pl.LightningModule):
    def __init__(
        self,
        data_dir="data/segmentation",
        backbone="DDRNet39",
        val_ratio=0.1,
        crop_size=(256, 256),
        num_classes=2,
        learning_rate=1e-3,
        loss_fn=None,
        logging_metrics: nn.ModuleList = None,
        batch_size=4,
        num_workers=4,
        junc_weight=5,
        skel_weight=5,
        aux_weight=0.1,
        autoclip=True,
        focus_loss_w_min=None,
        eval_crop=None,
        epochs=None,
        log_interval: Union[int, float] = 10,
        log_val_interval: Union[int, float] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.train_metrics = tm.MetricCollection(
            [
                tm.Accuracy(num_classes=2, dist_sync_on_step=True, average="macro", mdmc_average="global"),
                tm.Precision(num_classes=2, dist_sync_on_step=True, average="macro", mdmc_average="global"),
                tm.Recall(num_classes=2, dist_sync_on_step=True, average="macro", mdmc_average="global"),
                tm.F1(num_classes=2, dist_sync_on_step=True, average="macro", mdmc_average="global"),
                tm.IoU(num_classes=2, dist_sync_on_step=True),
            ]
        )
        self.val_metrics = self.train_metrics.clone()

        if loss_fn is None:
            loss_fn = BatchWeightedCrossEntropyLoss(smooth=0.1)
        self.loss_fn = loss_fn

        if self.hparams.autoclip:
            self.autoclip = AutoClip()

        if log_val_interval is None:
            log_val_interval = log_interval

        self._load_model()
        self._load_datasets()
        self._configure_plotter()

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.step(x, y, batch_idx)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.step(x, y, batch_idx)  # log loss

    def criterion(self, out, y):
        if self.hparams.focus_loss_w_min:
            height, width = list(y[0].shape[-2:])
            d_min = self.hparams.focus_loss_w_min
            center = [height / 2, width / 2]
            sigma = max(center) / 2
            X, Y = np.mgrid[:height, :width] - np.array(center)[:, None, None]
            d = np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))
            dist = np.interp(d, (np.min(d), np.max(d)), (d_min, 1))
            weight = torch.from_numpy(dist).to(y[0].device)
        else:
            weight = torch.ones_like(y[0])

        skl_out, skl_y = soft_skel(out[0], 3), soft_skel(y[0], 3)
        loss_skel = self.loss_fn(skl_out, skl_y * weight)
        loss_road = self.loss_fn(out[0], y[0] * weight)
        loss_junc = self.loss_fn(out[1], y[1] * weight)
        loss = loss_road + (self.hparams.skel_weight * loss_skel) + (self.hparams.junc_weight * loss_junc)
        logs = {
            "loss/total": loss.item(),
            "loss/skel": loss_skel.item(),
            "loss/road": loss_road.item(),
            "loss/junc": loss_junc.item(),
        }
        return loss, logs

    def step(self, x: Dict[str, torch.Tensor], y: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, **kwargs):
        # extract data and run model
        out = self(x)

        # loss calculation
        loss_main, logs_main = self.criterion(out[0], y)
        loss_aux, logs_aux = self.criterion(out[1], y)
        logs_aux = {f"{k}_aux": v for k, v in logs_aux.items()}
        loss = loss_main + self.hparams.aux_weight * loss_aux
        loss_logs = dict(**logs_main, **logs_aux)

        # evaluation metrics + logging
        out = [[x.detach() for x in o] for o in out]  # detach for safety reasons!
        self.log_metrics(x, y, out, loss_logs)
        if self.log_interval > 0:
            self.log_prediction(x, y, out, batch_idx, loss_logs)
        return loss

    def configure_optimizers(self):
        opt = Ranger(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.95, 0.999),
            use_gc=True,
            weight_decay=5e-4,
        )
        # sched = {
        #     "scheduler": ReduceLROnPlateau(opt, mode="min", factor=0.85, patience=3, threshold=1e-6, verbose=True),
        #     "monitor": "loss/total_val",
        #     "interval": "epoch",
        # }
        sched = CosineAnnealingLR(opt, T_max=self.hparams.epochs)        
        return [opt], [sched]

    def on_after_backward(self):
        if self.hparams.autoclip:
            self.autoclip(self.model)

    def plot_prediction(
        self,
        x: Dict[str, torch.Tensor],
        y: torch.Tensor,
        out: Dict[str, torch.Tensor],
        idx: int = 0,
        add_loss_to_title: Union[tm.Metric, torch.Tensor, bool] = False,
        logs: Dict = None,
    ) -> plt.Figure:
        # TODO: fix for ddp_spawn
        # self._configure_plotter()

        x = x[idx].cpu().numpy()
        x = np.transpose(x, (1, 2, 0))
        x = np.clip((x * (0.229, 0.224, 0.225)) + (0.485, 0.456, 0.406), 0, 1)

        y_road, y_junc = [t[idx][1].cpu().numpy() for t in y]
        (out_road, out_junc), (out_road_aux, out_junc_aux) = out

        def _cls_heatmap(out):
            out = out[idx]
            pred1 = F.softmax(out, dim=0)
            out1 = pred1.cpu().numpy()
            y_pred = out1[1, ...]  # HEATMAP
            return y_pred

        y_pred_road = _cls_heatmap(out_road)
        y_aux_road = _cls_heatmap(out_road_aux)
        y_pred_junc = _cls_heatmap(out_junc)

        fig, axes = plt.subplots(2, 3, figsize=(8, 6))
        fig.suptitle("Prediction Plot")

        axes[0, 0].imshow(x)
        axes[0, 0].set_title("Input")

        axes[0, 1].imshow(y_road, cmap="gray")
        axes[0, 1].set_title("Road GT")

        axes[0, 2].imshow(y_junc, cmap="gray")
        axes[0, 2].set_title("Junction GT")

        axes[1, 0].imshow(y_aux_road, cmap="gray")
        axes[1, 0].set_title("Road Aux")

        axes[1, 1].imshow(y_pred_road, cmap="gray")
        axes[1, 1].set_title("Road Pred")

        axes[1, 2].imshow(y_pred_junc, cmap="gray")
        axes[1, 2].set_title("Junction Pred")

        # TODO: add loss to tile
        fig.subplots_adjust(left=None, bottom=None, right=None, top=0.9, wspace=0.05, hspace=0.1)
        # fig.tight_layout(rect=[0, 0, 1, 0.9])
        return fig

    def log_prediction(
        self,
        x: Dict[str, torch.Tensor],
        y: torch.Tensor,
        out: Dict[str, torch.Tensor],
        batch_idx: int,
        logs: Dict = None,
    ) -> None:
        # log single prediction figure
        if (batch_idx % self.log_interval == 0 or self.log_interval < 1.0) and self.log_interval > 0:
            if self.log_interval < 1.0:  # log multiple steps
                log_indices = torch.arange(
                    0, self.hparams.batch_size, max(1, round(self.log_interval * self.hparams.batch_size))
                )
            else:
                log_indices = [0]

            for idx in log_indices:
                fig = self.plot_prediction(x, y, out, idx=idx, add_loss_to_title=True, logs=logs)
                tag = f"{['Val', 'Train'][self.training]} prediction"

                # if self.training:
                #     tag += f" of item {idx} in global batch {self.global_step}"
                # else:
                tag += f" of item {idx} in batch {batch_idx}"

                if isinstance(fig, (list, tuple)):
                    for idx, f in enumerate(fig):
                        self.logger.experiment.add_figure(f"Target {idx} {tag}", f, global_step=self.global_step)
                        plt.close(f)
                else:
                    self.logger.experiment.add_figure(tag, fig, global_step=self.global_step)
                    plt.close(fig)

    def log_metrics(
        self, x: Dict[str, torch.Tensor], y: torch.Tensor, out: Dict[str, torch.Tensor], additional_logs: Dict
    ) -> None:
        out = out[0]  # use the main prediction
        # out = [torch.argmax(t, dim=1) for t in out]  # lightning stupid fu*&#ing shit
        y = [(t[:, 1] >= 0.5) for t in y]  # convert to label encoding
        logs = additional_logs.copy()

        if self.hparams.eval_crop:
            cw, ch = self.hparams.eval_crop
            rh, rw = list(out[0].shape[-2:])
            sx, sy = (rw - cw) // 2, (rh - ch) // 2
            y = [t[..., sy:-sy, sx:-sx] for t in y]
            out = [o[..., sy:-sy, sx:-sx] for o in out]

        logging_metrics = self.train_metrics if self.training else self.val_metrics
        # logging losses - for each target
        for idx, tg in enumerate(["road", "junc"]):
            for name, metric in logging_metrics.items():
                key = f"{name}/{tg}"
                value = metric(out[idx], y[idx])  # second class
                logs[key] = value

                self.log(
                    name=f"{key}_{['val', 'train'][self.training]}",
                    value=metric,
                    on_step=self.training,
                    on_epoch=True,
                    prog_bar=True,
                )

        for k, v in additional_logs.items():
            self.log(
                name=f"{k}_{['val', 'train'][self.training]}",
                value=v,
                on_step=self.training,
                on_epoch=True,
                prog_bar=True,
            )

        return logs

    @property
    def model(self):
        return self._model

    @property
    def log_interval(self) -> float:
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
            self.val_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, drop_last=True
        )

    def _load_model(self):
        self._model = MultiMaskDDRNet(name=self.hparams.backbone, use_aux=True)

    def _load_datasets(self):
        image_dir = os.path.join(self.hparams.data_dir, "images")
        mask_dir = os.path.join(self.hparams.data_dir, "masks")
        train_transforms, test_transforms = rss.transforms_v1(self.hparams.crop_size)

        train_cities = TRAIN_SET
        val_cities = TEST_SET

        filepaths = rss.RSS_MultiMask._get_images_filepaths(image_dir, mask_dir)
        train_filepaths = [(a, b) for a, b in filepaths if a.name.split("_")[0] in train_cities]
        val_filepaths = [(a, b) for a, b in filepaths if a.name.split("_")[0] in val_cities]

        self.train_dataset = rss.RSS_MultiMask(train_filepaths, transform=train_transforms)
        self.val_dataset = rss.RSS_MultiMask(val_filepaths, transform=test_transforms)

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


@EXPS.register_module("J_Road")
def jroad_trainer(config=None):
    # seed
    pl.seed_everything(config.experiment.seed)

    lr_logger = LearningRateMonitor(logging_interval="step")  # log the learning rate
    logger = TensorBoardLogger(save_dir=config.experiment.dir, name=config.experiment.name)  # log to tensorboard
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(logger.log_dir, "checkpoints"),
        filename="JRoad--{epoch:02d}--{loss/total_val:.4f}",
        monitor="loss/total_val",
        save_top_k=5,
        mode="min",
    )

    num_gpus = torch.cuda.device_count()
    accelerator = "ddp" if num_gpus else None
    plugin = None

    if num_gpus:
        config.model.batch_size = int(config.model.batch_size / num_gpus)
    model = JRoadMultiS(**config.model)


    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, lr_logger],
        logger=logger,
        gpus=num_gpus,
        accelerator=accelerator,
        plugins=plugin,
        **config.trainer
    )

    trainer.fit(model)
