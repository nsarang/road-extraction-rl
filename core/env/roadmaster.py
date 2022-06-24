from __future__ import annotations
from typing import Dict, Any, Tuple, Union, Optional, List

import uuid
import torch
import os
import gym
import numpy as np
import json
from PIL import Image
from pathlib import Path
import random
from collections import defaultdict, Sequence
from datetime import datetime
from time import time

from gym.spaces import Box
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import DictConfig

from .drawers import LazyDrawer, LazyIntensityDrawer
from .renderers import GraphRenderer, Renderer, VideoRenderer
from .rewards import RewardScheme
from .actions import ActionScheme, MaskBasedScheme
from .stoppers import Stopper
from core.utils.profiler import profile
from core.utils.misc import DummyObject, JSONSerializer, TStackU, DefaultIndexedOrderedDict
from core.utils.clock import Clock


class RoadMasterEnv(gym.Env):
    def __init__(
        self,
        aerial_data: Union[np.array, List[np.array]],
        action_scheme: ActionScheme,
        reward_scheme: RewardScheme,
        observation_shape: tuple,
        road_properties: DictConfig,
        stopper: Optional[Stopper] = None,
        starting_location: Optional[tuple] = None,
        renderer: 'Optional[Union[Renderer, Literal["video", "graph", "plotly"]]]' = "video",
        episode_window_size: Optional[Tuple] = None,
        seed: Optional[int] = None,
        observation_mode: 'Literal["rgb", "rgb+movement"]' = "rgb+movement",
        actions_obs_len: int = 10,
        name: str = "",
        use_logger: bool = False,
        experiment_dir: Optional[Path] = None,
        fixed_seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if isinstance(aerial_data, list) is False:
            aerial_data = [aerial_data]
        self.aerial_rgb = aerial_data[0]
        self.aerial_data = aerial_data

        # SPECS
        H, W, _ = self.aerial_rgb.shape
        self.map_size = np.array([W, H])
        self.episode_window_size = episode_window_size
        self.observation_shape = observation_shape
        self.road_properties = road_properties

        self.action_space = action_scheme.action_space
        num_channels = 3 if observation_mode == "rgb" else 4
        self.observation_space = Box(low=-3, high=3, shape=(num_channels, *observation_shape), dtype=np.float32)

        # COMPONENTS
        self.action_scheme = action_scheme
        self.reward_scheme = reward_scheme
        self.stopper = stopper or Stopper()

        # RENDERING
        self.renderer = renderer or DummyObject()
        self.gtGraphDrawer = LazyDrawer(self.size)
        GraphRenderer.draw_graph(
            self.gtGraphDrawer,
            self.action_scheme.gtGraph,
            draw_edges=True,
            draw_nodes=False,
            edge_width=road_properties.lane_width,
            base_width=road_properties.base_width,
            apply_lane_width=road_properties.apply_lane_width,
        )
        self.gtVisitedDrawer = LazyIntensityDrawer(self.size)
        self.movement_drawer = LazyIntensityDrawer(self.size)
        self.retrace_drawer = LazyDrawer(self.size, mode="F")

        # MISC
        self.clock = Clock()
        self.starting_location = tuple(starting_location) if starting_location else None
        self._seed = seed
        self.fixed_seed = fixed_seed
        self.observation_mode = observation_mode
        self.actions_obs_len = actions_obs_len

        self.name = name
        self.episode_id: str = None
        self.bounds = None
        self.history = defaultdict(lambda: defaultdict(self._syncedLog))
        self.hparams = dict()
        self.logger = None
        self.use_logger = use_logger
        self.exp_dir = experiment_dir
        self.history_dir = os.path.join(self.exp_dir, "history")
        os.makedirs(self.history_dir, exist_ok=True)

        self._setClocks()

    @property
    def components(self) -> "Dict[str, Any]":
        return {
            "action_scheme": self.action_scheme,
            "reward_scheme": self.reward_scheme,
            "stopper": self.stopper,
            "movement_drawer": self.movement_drawer,
            "retrace_drawer": self.retrace_drawer,
            "gtVisitedDrawer": self.gtVisitedDrawer,
        }

    # @profile
    def step(self, action: Any, **kwargs) -> "Tuple[np.array, float, bool, dict]":
        self.clock.increment()
        self.log(**kwargs)

        self.action_scheme.perform(self, action)
        self.reward_scheme.reward(self)
        self.draw()
        done = self.stopper.stop(self)
        observation = self.observe()

        reward = self.last("reward_total")
        info = {}

        return observation, reward, done, info

    def undo(self):
        assert self.steps
        self.action_scheme.undo(self)
        self.reward_scheme.undo()
        self.movement_drawer.undo()
        self.retrace_drawer.undo()
        self.gtVisitedDrawer.undo()

        for key in self.logs.keys():
            if key in ["UndoW_BP"]:
                continue
            for cat in self.history:
                self.history[cat][key].undo()
        self.clock.decrement()

    # @profile
    def reset(self) -> "np.array":
        self.episode_id = str(uuid.uuid4())
        self.logs.clear()
        self.clock.reset()

        if self.fixed_seed and self._seed:
            self.seed(self._seed)

        location = self.starting_location or self.action_scheme.start_location()
        gLocation = location

        self.reward_scheme.reset()
        self.renderer.reset()
        self.movement_drawer.reset()
        self.retrace_drawer.reset()
        self.gtVisitedDrawer.reset()

        self.history.clear()
        self.hparams.clear()

        self.log({"location-agent": location, "location-graph": gLocation})
        self.bounds = self.coord_bounds()
        self.log({"bounds": self.bounds})

        if self.episode_window_size is None:
            self.action_scheme.reset()
        else:
            self.action_scheme.reset(self.bounds)

        if self.use_logger:
            if self.logger is not None:
                self.logger.experiment.close()
            logdir = os.path.join(self.exp_dir, "logs/episodes")
            self.logger = TensorBoardLogger(logdir, self.episode_id)

        observation = self.observe()
        return observation

    def seed(self, seed):
        self._seed = seed
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        return seed

    def observe(self, mode=None):
        mode = mode or self.observation_mode
        modes = mode.split("+")

        o_width, o_height = self.observation_shape
        ((x1_r, y1_r, x2_r, y2_r), (x1_o, y1_o, x2_o, y2_o)) = self.get_observation_window_coords()

        r = {}
        if "rgb" in modes:
            obs = np.zeros((o_height, o_width, 3), dtype=np.uint8)
            obs[y1_o:y2_o, x1_o:x2_o] = self.aerial_rgb[y1_r:y2_r, x1_r:x2_r]
            r.update(obs=obs)
        if "gt-obs" in modes:
            assert "rgb" not in modes
            obs = np.zeros((o_height, o_width, 3), dtype=np.uint8)
            obs[y1_o:y2_o, x1_o:x2_o] = np.array(self.gtGraphDrawer.window((x1_r, y1_r, x2_r, y2_r)).convert("RGB"))
            r.update(obs=obs)
        if "movement" in modes:
            movement = np.zeros((o_height, o_width, 2))
            movement_raw = self.movement_drawer.window((x1_r, y1_r, x2_r, y2_r), mode="intensity-gray")
            retrace_map = self.retrace_drawer.window((x1_r, y1_r, x2_r, y2_r), mode="numpy")
            movement[y1_o:y2_o, x1_o:x2_o, 0] = movement_raw
            movement[y1_o:y2_o, x1_o:x2_o, 1] = retrace_map
            if len(self.aerial_data) > 1:
                data = []
                for image in self.aerial_data[1:]:
                    channels = image.shape[2] if image.ndim == 3 else 1
                    patch = np.zeros((o_height, o_width, channels), dtype=np.uint8)
                    patch[y1_o:y2_o, x1_o:x2_o, 0] = image[y1_r:y2_r, x1_r:x2_r]
                    data.append(patch)
                movement = np.concatenate((movement, *data), axis=-1)
            r.update(movement=movement)
        if "road-mask" in modes:
            # gt road mask
            mask_road = np.zeros((o_height, o_width), dtype=np.uint8)
            road = self.gtGraphDrawer.window((x1_r, y1_r, x2_r, y2_r), mode="numpy")
            mask_road[y1_o:y2_o, x1_o:x2_o][np.any(road, axis=-1)] = 255
            # gt visited mask
            mask_visited = np.zeros((o_height, o_width), dtype=np.uint8)
            mask_visited[y1_o:y2_o, x1_o:x2_o] = self.gtVisitedDrawer.window(
                (x1_r, y1_r, x2_r, y2_r), mode="intensity-gray"
            )
            r.update(mask_road=mask_road, mask_visited=mask_visited)
        if "actions" in modes:
            actions = self.last("action", size=self.actions_obs_len, shrink=False)
            r.update(actions=actions)

        return r

    def draw(self):
        if self.last("action_type") == "Normal" and self.last("divergence-status") == False:
            graph_path = self.last("traversed-path")
            for u, v in zip(graph_path, graph_path[1:]):
                width = (
                    self.road_properties.base_width
                    + self.road_properties.lane_width * self.action_scheme.graph[u][v]["lanes"]
                )
                self.gtVisitedDrawer.line(u, v, width)

    def get_observation_window_coords(self, location=None):
        x1_frame, y1_frame, x2_frame, y2_frame = self.bounds
        o_width, o_height = self.observation_shape

        cx, cy = location or self.location
        x1, y1 = int(cx - o_width // 2), int(cy - o_height // 2)
        x2, y2 = (x1 + o_width), (y1 + o_height)

        x1_r, y1_r = max(x1, x1_frame), max(y1, y1_frame)
        x2_r, y2_r = min(x2, x2_frame - 1), min(y2, y2_frame - 1)

        x1_o, y1_o = (x1_r - x1), (y1_r - y1)
        x2_o = x1_o + (x2_r - x1_r)
        y2_o = y1_o + (y2_r - y1_r)
        return (x1_r, y1_r, x2_r, y2_r), (x1_o, y1_o, x2_o, y2_o)

    def log(self, *args, **kwargs) -> None:
        for v in args:
            if isinstance(v, dict):
                self.log(**v)
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
            if isinstance(v, np.ndarray) and v.size == 1:
                v = v.item()
            for logcat in ["logs", "all"]:
                self.history[logcat][k].append(v)

        if self.use_logger:
            kwargs = {k: v for k, v in kwargs.items() if not isinstance(v, list)}
            for k in list(kwargs.keys()):
                v = kwargs[k]
                if "_" in k:
                    k, k_old = k.replace("_", "/", 1), k
                    kwargs[k] = kwargs.pop(k_old)
                if isinstance(v, tuple) and len(v) == 2:
                    kwargs[f"{k}_x"] = v[0]
                    kwargs[f"{k}_y"] = v[1]
                    del kwargs[k]
            self.logger.log_metrics(kwargs, self.steps)

    def render(self, mode=None, **kwargs) -> None:
        return self.renderer.render(self, mode=mode, **kwargs)

    def save(self, mode=None, render=True, history=True, **kwargs) -> None:
        if history:
            with open(os.path.join(self.history_dir, self.savename + ".json"), "w", encoding="utf-8") as f:
                data = self.history
                f.write(json.dumps(data, indent=4, cls=JSONSerializer))
        if render:
            self.renderer.save(self, mode=mode, **kwargs)

    @property
    def savename(self):
        uid = self.episode_id.split("-")[0]
        timestamp = datetime.now().strftime("%y%m%d_%H%M")
        name = f"{self.name}_{uid}_{timestamp}"
        return name

    def close(self) -> None:
        self.renderer.close()

    def __repr__(self):
        return "RoadMaster"

    @property
    def steps(self):
        return self.clock.step

    @property
    def location(self):
        return self.last("location-agent")

    @property
    def gLocation(self):
        return self.last("location-graph")

    @property
    def logs(self):
        return self.history["logs"]

    def unlog(self, keys, times=1, cats=["logs", "all"]):
        if isinstance(keys, str):
            keys = [keys]

        for key in keys:
            for cat in cats:
                for _ in range(times):
                    self.history[cat][key].pop()

    def last(self, name, size=1, fillNa=None, iloc=True, shrink=True):
        log = self.logs[name]
        if iloc == True:
            items = log.values()[-size:]
            if fillNa is not None:
                items = [fillNa] * max(size - len(log), 0) + items
        else:
            t = self.steps
            items = []
            for ind in range(t - size + 1, t + 1):
                if ind in log:
                    items.append(log[ind])
                elif fillNa is not None:
                    items.append(fillNa)
        if shrink and len(items) == 1:
            items = items[0]
        return items

    @property
    def rgb(self):
        return self.aerial_rgb

    @property
    def rgb_bounded(self):
        x1, y1, x2, y2 = self.bounds
        return self.aerial_rgb[y1:y2, x1:x2]

    @property
    def size(self):
        if self.episode_window_size is None:
            return self.map_size
        return self.episode_window_size

    def coord_bounds(self):
        if self.episode_window_size is None:
            width, height = self.map_size
            x1_frame, y1_frame, x2_frame, y2_frame = 0, 0, width, height
        else:
            global_width, global_height = self.map_size
            if isinstance(self.episode_window_size[0], Sequence):
                width, height = random.choice(self.episode_window_size)
            else:
                width, height = self.episode_window_size

            cx_loc, cy_loc = self.logs["location-agent"][0]
            x1_frame, y1_frame = int(cx_loc - width // 2), int(cy_loc - height // 2)
            x2_frame, y2_frame = (x1_frame + width), (y1_frame + height)

            x_shift, y_shift = 0, 0
            if x1_frame < 0:
                x_shift = -x1_frame
            if y1_frame < 0:
                y_shift = -y1_frame
            if x2_frame > global_width:
                x_shift = global_width - x2_frame
            if y2_frame > global_height:
                y_shift = global_height - y2_frame

            x1_frame += x_shift
            x2_frame += x_shift
            y1_frame += y_shift
            y2_frame += y_shift

        return x1_frame, y1_frame, x2_frame, y2_frame

    def _setClocks(self):
        def _cHelper(c):
            if hasattr(c, "clock"):
                c.clock = self.clock
            if hasattr(c, "components"):
                for s in c.components.values():
                    _cHelper(s)

        _cHelper(self)

    def _syncedLog(self):
        s = TStackU()
        s.clock = self.clock
        return s
