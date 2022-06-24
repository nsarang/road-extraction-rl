from core.env.actions import MaskBasedScheme
import numpy as np
from pydoc import locate

from numpy.lib.arraysetops import isin

from core.data import load_data
from core.env.roadmaster import RoadMasterEnv
from core.env.wrappers import EpisodicEnvW, PreprocessObservationW


def create_env(config, max_steps=None, max_episodes=None, preprocess_obs=None):
    env = RoadMasterEnv(**config)
    env = EpisodicEnvW(env, max_episodes, max_steps)
    if preprocess_obs:
        env = PreprocessObservationW(env)
    return env


def create_env_lazy(
    data, env_config, actions_config, rewards_config, wrapper_config, stopper_config=None
):
    # DATA
    if isinstance(data, tuple):
        rgb, graph = data
    else:
        rgb, graph = load_data(data["city"], data["data_dir"])
        rgb = np.array(rgb.convert("RGB"))

    # ACTION SCHEME
    action_scheme_cls = locate("core.env.actions." + actions_config["name"])
    actions_args = dict(**actions_config["args"])
    actions_args["road_graph"] = graph
    if issubclass(action_scheme_cls, MaskBasedScheme):
        actions_args["size"] = rgb.shape[1::-1]  # (W, H)
    action_scheme = action_scheme_cls(**actions_args)

    # REWARD SCHEME
    reward_scheme = locate("core.env.rewards." + rewards_config["name"])(
        **rewards_config["args"]
    )

    # STOPPER
    if stopper_config:
        stopper = locate("core.env.stoppers." + stopper_config["name"])(
            **stopper_config["args"]
        )
    else:
        stopper = None

    env_config = dict(
        rgb=rgb,
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        stopper=stopper,
        **env_config
    )
    return create_env(env_config, **wrapper_config)
