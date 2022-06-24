import os
import numpy as np
from functools import partial
from pydoc import locate
from omegaconf import OmegaConf
from collections import MutableSequence, MutableMapping
from itertools import cycle

from core.data import load_data

# from core.env.factory import create_env, create_env_lazy
from core.env.actions import MaskBasedScheme, GraphBasedScheme
from core.env.roadmaster import RoadMasterEnv
from core.env.renderers import GraphRenderer, AggregateRenderer


def load_env_v2(config):
    def from_config(data, cfg, name):
        (images, graph) = data
        size = images[0].shape[1::-1]  # (W, H)
        # ACTION SCHEME
        action_scheme_cls = locate("core.env.actions." + cfg.action_scheme["name"])
        actions_args = dict(**cfg.action_scheme["args"])
        actions_args["road_graph"] = graph
        if issubclass(action_scheme_cls, MaskBasedScheme):
            rps = cfg.base.road_properties
            actions_args["size"] = size
            actions_args["base_width"] = actions_args.get("base_width", rps.base_width)
            actions_args["lane_width"] = actions_args.get("lane_width", rps.lane_width)
        elif issubclass(action_scheme_cls, GraphBasedScheme):
            rps = cfg.base.road_properties
            line_width = rps.base_width + rps.lane_width
            actions_args["line_width"] = actions_args.get("line_width", line_width)

        action_scheme = action_scheme_cls(**actions_args)
        # REWARD SCHEME
        reward_scheme = locate("core.env.rewards." + cfg.reward_scheme["name"])(**cfg.reward_scheme["args"])
        # STOPPER
        stopper = None
        if cfg.stopper:
            stopper = locate("core.env.stoppers." + cfg.stopper["name"])(**cfg.stopper["args"])
        # RENDERER
        renderer = None

        def load_renderer(name, kwargs):
            args = dict()
            if name in ["GraphRenderer", "VideoRenderer", "VideoRenderer9000", "VideoRendererContinuousSpace"]:
                rps = cfg.base.road_properties
                args.update(base_width=rps.base_width, edge_width=rps.lane_width)
            if name in ["VideoRenderer", "VideoRenderer9000", "VideoRendererContinuousSpace"]:
                map_size = size
                args.update(road_network=action_scheme.gtGraph, size=map_size)
                baseRenderer = GraphRenderer(**args)
                return locate("core.env.renderers." + name)(
                    baseRenderer=baseRenderer,
                    save_dir=os.path.join(cfg.base.experiment_dir, "visualizations"),
                    **kwargs,
                )
            elif name == "TopViewRenderer":
                return locate("core.env.renderers.TopViewRenderer")(
                    graph=graph,
                    size=size,
                    save_dir=os.path.join(cfg.base.experiment_dir, "visualizations"),
                    **kwargs,
                )
            else:
                assert NotImplementedError

        if cfg.renderer is not None:
            renderer = AggregateRenderer([load_renderer(name, rcfg) for name, rcfg in cfg.renderer.items()])

        # ROADMASTER
        env_config = dict(
            aerial_data=images,
            action_scheme=action_scheme,
            reward_scheme=reward_scheme,
            stopper=stopper,
            renderer=renderer,
            name=name,
            **cfg.base,
        )
        env = RoadMasterEnv(**env_config)
        # WRAPPERS
        for w in cfg.wrappers:
            wrapper_cls = locate("core.env.wrappers." + w)
            kwargs = cfg.wrappers[w] if isinstance(cfg.wrappers[w], MutableMapping) else {}
            env = wrapper_cls(env, **kwargs)
        return env

    def alternate_cities(config, name):
        # DATA
        cfg = config.data
        cities = cfg.cities
        for city in cycle(cities):
            images, graph = load_data(city, cfg.dir, imagery_dir=cfg.imagery_dir, osm_dir=cfg.osm_dir)
            images[0] = images[0].convert("RGB")
            images = [np.array(x) for x in images]
            yield from_config(data=(images, graph), cfg=config, name=f"{name}_{city}")

    # CREATE ENV
    cfg = config.env
    train_generator = alternate_cities(config=cfg.train, name="train")
    test_generator = alternate_cities(config=cfg.test, name="test")
    train_env_udf = lambda: next(train_generator)
    test_env_udf = lambda: next(test_generator)
    return train_env_udf, test_env_udf


def load_env(config):
    # DATA
    cfg = config.data
    rgb, graph = load_data(cfg.city, cfg.data_dir)
    rgb = np.array(rgb.convert("RGB"))

    # ENV
    cfg = config.env
    train_env_udf = partial(
        create_env_lazy,
        data=(rgb, graph),
        actions_config=cfg.action_scheme,
        rewards_config=cfg.reward_scheme,
        stopper_config=cfg.stopper,
        env_config=cfg.base.train,
        wrapper_config=cfg.wrappers.train,
    )
    test_env_udf = partial(
        create_env_lazy,
        data=(rgb, graph),
        actions_config=cfg.action_scheme,
        rewards_config=cfg.reward_scheme,
        stopper_config=cfg.stopper,
        env_config=cfg.base.test,
        wrapper_config=cfg.wrappers.test,
    )
    return train_env_udf, test_env_udf


# GROUNDTRUTH AS INPUT EXP
def load_env_gt(config):
    # DATA
    cfg = config.data
    rgb, graph = load_data(cfg.city, cfg.data_dir)
    rgb = np.array(rgb.convert("RGB"))
    mask = MaskBasedScheme.generate_gt_mask(rgb.shape[1::-1], graph, config.env.action_scheme.args.brush_width)
    mask = np.repeat(mask[..., None], repeats=3, axis=-1)
    rgb = mask

    # ENV
    cfg = config.env
    train_env_udf = partial(
        create_env_lazy,
        data=(rgb, graph),
        actions_config=cfg.action_scheme,
        rewards_config=cfg.reward_scheme,
        stopper_config=cfg.stopper,
        env_config=cfg.base.train,
        wrapper_config=cfg.wrappers.train,
    )
    test_env_udf = partial(
        create_env_lazy,
        data=(rgb, graph),
        actions_config=cfg.action_scheme,
        rewards_config=cfg.reward_scheme,
        stopper_config=cfg.stopper,
        env_config=cfg.base.test,
        wrapper_config=cfg.wrappers.test,
    )
    return train_env_udf, test_env_udf


def save_hparams(config):
    save_dir = os.path.join(config.experiment.dir, "hparams")
    filepath = os.path.join(save_dir, "config.yaml")
    os.makedirs(save_dir, exist_ok=True)
    OmegaConf.save(config=config, f=filepath)


def create_experiment_dir(config):
    root = os.path.join(config.root_dir, config.name)
    v = next_version(root)
    exp_dir = os.path.join(root, f"version_{v}")
    os.makedirs(exp_dir)
    config.dir = exp_dir


def next_version(root_dir):
    if not os.path.isdir(root_dir):
        return 0

    existing_versions = []
    for d in os.listdir(root_dir):
        d = os.path.join(root_dir, d)
        bn = os.path.basename(d)
        if os.path.isdir(d) and bn.startswith("version_"):
            dir_ver = bn.split("_")[1].replace("/", "")
            existing_versions.append(int(dir_ver))
    if len(existing_versions) == 0:
        return 0

    return max(existing_versions) + 1
