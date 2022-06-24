from typing import Union, Dict, Any

import os
import re
import torch
import torch.nn as nn
from pathlib import Path
from sortedcontainers import SortedList


class ModelCheckpoint:
    CHECKPOINT_NAME_LAST = "last"
    FILE_EXTENSION = ".ckpt"
    CHECKPOINT_JOIN_CHAR = "-"
    VERSION = 2

    def __init__(
        self,
        monitor: str,
        dirpath: Union[str, Path] = ".",
        filename: str = "{epoch}",
        save_last: bool = False,
        save_top_k: int = 0,
        mode: str = "min",
        period: int = 1,
        prefix: str = "",
        verbose: bool = False,
    ):
        self.monitor = monitor
        self.dirpath = dirpath
        self.filename = filename
        self.save_last = save_last
        self.save_top_k = save_top_k
        self.mode = mode
        self.period = period
        self.prefix = prefix
        self.verbose = verbose

        mode_sign = 1 if mode == "min" else -1
        self.best_checkpoints = SortedList([], key=lambda x: (mode_sign * x[0], x[1]))
        os.makedirs(self.dirpath, exist_ok=True)

    def save_checkpoint(
        self, model: nn.Module, epoch: int, metrics: Dict[str, float], optimizer: torch.optim.Optimizer = None
    ):
        # SAVE LAST
        if self.save_last:
            name = f"{self.CHECKPOINT_NAME_LAST}{self.FILE_EXTENSION}"
            filepath = os.path.join(self.dirpath, name)
            self._save(filepath, model, epoch, metrics, optimizer)

        # CHECK PERIOD
        if len(self.best_checkpoints) > 0 and (epoch - max(t[1] for t in self.best_checkpoints) < self.period):
            return

        # UPDATE TOP K
        basename = self.format_checkpoint_name(epoch, metrics)
        filepath = os.path.join(self.dirpath, f"{basename}{self.FILE_EXTENSION}")
        monitor = metrics[self.monitor]
        curr_ckpt = (monitor, epoch, filepath)
        self.best_checkpoints.add(curr_ckpt)

        if len(self.best_checkpoints) > self.save_top_k:
            del_ckpt = self.best_checkpoints.pop()
            if curr_ckpt == del_ckpt:
                return

            os.remove(del_ckpt[-1])

        self._save(filepath, model, epoch, metrics, optimizer)

    def format_checkpoint_name(
        self,
        epoch: int,
        metrics: Dict[str, Any],
        auto_insert_metric_name: bool = True,
    ) -> str:
        filename = self.filename

        # check and parse user passed keys in the string
        groups = re.findall(r"(\{.*?)[:\}]", filename)
        if len(groups) >= 0:
            metrics.update({"epoch": epoch})
            for group in groups:
                name = group[1:]

                if auto_insert_metric_name:
                    filename = filename.replace(group, name + "={" + name)

                if name not in metrics:
                    metrics[name] = 0

            # fieldnames = [fname for _, fname, _, _ in Formatter().parse(filename) if fname]
            # metrics = {k:metrics[k] for k in metrics if k in fieldnames}
            filename = filename.format(**metrics)

        if self.prefix:
            filename = self.CHECKPOINT_JOIN_CHAR.join([self.prefix, filename])

        return filename

    def _save(
        self,
        filepath: str,
        model: nn.Module,
        epoch: int = None,
        metrics: Dict[str, float] = None,
        optimizer: torch.optim.Optimizer = None,
    ):
        state = {"model_state_dict": model.state_dict(), "meta": {"checkpointer_version": self.VERSION}}

        if epoch is not None:
            state["epoch"] = epoch
        if metrics is not None:
            state["metrics"] = metrics
        if optimizer is not None:
            state["optimizer_state_dict"] = optimizer.state_dict()

        torch.save(state, filepath)
