from lzma import MODE_NORMAL
from typing import Optional, Any

import logging
import numpy as np
import cv2
import albumentations as A
import random
from functools import cached_property
import tempfile
import networkx as nx
from torch.autograd.grad_mode import _DecoratorContextManager

ACTIVE_STATUS = True


class Wrapper:
    """Wraps the environment to allow a modular transformation.

    This class is the base class for all wrappers. The subclass could override
    some methods to change the behavior of the original environment without touching the
    original code.

    .. note::

        Don't forget to call ``super().__init__(env)`` if the subclass overrides :meth:`__init__`.

    """

    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.env, name)

    def findEnv(self, name):
        if type(self).__name__ == name:
            return self
        elif isinstance(self.env, Wrapper):
            return self.env.findEnv(name)
        elif str(self.env) == name:
            return self.env

    @property
    def originalEnv(self):
        if isinstance(self.env, Wrapper):
            return self.env.originalEnv
        return self.env

    def __str__(self):
        return "<{}{}>".format(type(self).__name__, self.env)

    def __repr__(self):
        return str(self)


class UndoW(Wrapper):
    def __init__(
        self, env, num_undo=10, hard_reset_rate=0.01, max_divergences=1, min_reward_percent=0.7
    ):
        assert not isinstance(env, Wrapper), "UndoW must be the first wrapper"
        super().__init__(env)
        self.num_undo = num_undo
        self.hard_reset_rate = hard_reset_rate
        self.max_divergences = max_divergences
        self.min_avg_reward_percent = min_reward_percent
        self.done = False

    def step(self, *args, **kwargs):
        obs, reward, done, info = self.env.step(*args, **kwargs)
        if ACTIVE_STATUS:
            self.done = done
        return obs, reward, done, info

    def reset(self, *args, **kwargs):
        if (
            (self.done == 1)
            and (random.random() > self.hard_reset_rate)
            and self._isEpisodeFeasible()
        ):
            num_undo = min(self.num_undo, self.env.steps)
            for _ in range(num_undo):
                self.env.undo()
            self.env.log({"UndoW_BP": self.env.steps})
            return self.env.observe()
        return self.env.reset(*args, **kwargs)

    def _isEpisodeFeasible(self) -> bool:
        if self.env.steps == 0:
            return True
        divStatus = sum(self.env.logs["divergence-status"][:]) <= self.max_divergences
        rewStatus = (
            self.env.reward_scheme.rewardPercentile(np.mean(self.env.logs["reward_total"][:]))
            >= self.min_avg_reward_percent
        )
        return divStatus and rewStatus


class EpisodicEnvW(Wrapper):
    def __init__(self, env, max_episodes=None, max_steps=None):
        # assert not isinstance(env, Wrapper), "EpisodicEnvW must be the first wrapper"
        super().__init__(env)
        self._max_episodes = max_episodes or 1
        self._max_steps = max_steps or np.inf
        self._total_episodes = None
        self._steps = None
        self._done = True

    def step(self, *args, **kwargs):
        obs, reward, done, info = self.env.step(*args, **kwargs)

        if ACTIVE_STATUS:
            self._steps += 1
            if done:
                self._done = True
            elif self._steps >= self._max_steps:
                done = True  # artifical done signal
                self._steps = 0
        return obs, reward, done, info

    def reset(self, **kwargs):
        if self._done is False:
            self._total_episodes += 1
            if self._total_episodes >= self._max_episodes:
                self._done = True
        if self._done:
            obs = self.env.reset(**kwargs)
            self._total_episodes = 0
            self._steps = 0
            self._done = False
        else:
            obs = self.env.observe()
        return obs


class AutoRenderW(Wrapper):
    def __init__(
        self,
        env,
        mode: 'Optional[Literal["step", "episode"]]' = None,
        period: float = 1,
        start: int = 0,
    ):
        super().__init__(env)
        self._mode = mode
        self._period = period
        self._start = start
        self._counter = 0

    def step(self, *args, **kwargs):
        observation, reward, done, info = self.env.step(*args, **kwargs)

        if ACTIVE_STATUS:
            if self._mode == "step" or (self._mode == "episode" and done):
                if (self._counter % self._period == 0) and (self._counter >= self._start - 1):
                    self.save()
                self._counter += 1

        return observation, reward, done, info


class AutoSaveW(Wrapper):
    def __init__(
        self,
        env,
        mode: 'Optional[Literal["step", "episode"]]' = None,
        period: float = 1,
        start: int = 0,
    ):
        super().__init__(env)
        self._mode = mode
        self._period = period
        self._start = start
        self._counter = 0

    def step(self, *args, **kwargs):
        observation, reward, done, info = self.env.step(*args, **kwargs)

        if ACTIVE_STATUS:
            if self._mode == "step" or (self._mode == "episode" and done):
                if (self._counter % self._period == 0) and (self._counter >= self._start):
                    self.save()
                self._counter += 1

        return observation, reward, done, info


class ObservationAugmentationW(Wrapper):
    def __init__(
        self,
        env,
        mode: 'Literal["train", "test", "mask"]',
        output_shape,
        mask_shape=(64, 64),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        super().__init__(env)
        self._mode = mode
        self._output_shape = output_shape
        self._mask_shape = mask_shape
        self._obs_mean = mean
        self._obs_std = std

    def reset(self, *args, **kwargs):
        observation = self.env.reset(*args, **kwargs)
        return self.augment(observation)

    def step(self, *args, **kwargs):
        observation, reward, done, info = self.env.step(*args, **kwargs)
        return self.augment(observation), reward, done, info

    def augment(self, inputs):
        inputs["obs"] = self.transform(image=inputs["obs"])["image"]

        for name in ["movement", "mask_road", "mask_visited"]:
            if name not in inputs:
                continue
            if name == "movement":
                size = self._output_shape
                interpolation = cv2.INTER_AREA
            else:
                size = self._mask_shape
                interpolation = cv2.INTER_NEAREST
            inputs[name] = A.Resize(*size).apply(inputs[name], interpolation) / 255

        return inputs

    def revert_preprocess(self, obs):
        c = obs.shape[-1]
        mean = self._obs_mean[:c]
        std = self._obs_std[:c]
        obs = np.transpose(obs, (2, 0, 1))
        obs = (obs * std) + mean
        obs = obs * 255
        return obs

    @property
    def transform(self):
        if self._mode == "train":
            return self.train_transforms
        elif self._mode == "test":
            return self.test_transforms
        elif self._mode == "mask":
            return self.mask_transforms
        raise RuntimeError

    @cached_property
    def train_transforms(self):
        return A.Compose(
            [
                A.Resize(*self._output_shape, interpolation=cv2.INTER_AREA),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                A.GaussianBlur(sigma_limit=0.7, p=0.5),
                A.GaussNoise(mean=20, p=0.5),
                A.Normalize(
                    mean=self._obs_mean,
                    std=self._obs_std,
                    max_pixel_value=255,
                ),
            ]
        )

    @cached_property
    def test_transforms(self):
        return A.Compose(
            [
                A.Resize(*self._output_shape),
                A.Normalize(
                    mean=self._obs_mean,
                    std=self._obs_std,
                    max_pixel_value=255,
                ),
            ]
        )

    @cached_property
    def mask_transforms(self):
        return A.Compose([A.Resize(*self._output_shape), A.ToFloat(max_value=255)])


class InputRepresentationW(Wrapper):
    def __init__(
        self,
        env,
        mode: 'Optional[Literal["normal", "combined", "stacked"]]' = None,
    ):
        super().__init__(env)
        self.mode = mode

    def reset(self, *args, **kwargs):
        observation = self.env.reset(*args, **kwargs)
        return self.process(observation)

    def step(self, *args, **kwargs):
        observation, reward, done, info = self.env.step(*args, **kwargs)
        return self.process(observation), reward, done, info

    def process(self, inputs):
        obs, movement = inputs.pop("obs"), inputs.pop("movement")
        if movement.ndim == 2:
            movement = movement[np.newaxis]
        obs = obs.transpose(2, 0, 1)
        movement = movement.transpose(2, 0, 1)

        if "actions" in inputs:
            actions = np.array(inputs["actions"]) + 2  # [-1, naction - 2] to [1, nactions]
            actions_obs_len = self.env.originalEnv.actions_obs_len
            actions = np.pad(actions, (actions_obs_len - len(actions), 0))
            inputs["actions"] = actions

        if self.mode == "combined":  # NOTE: deprecated since retrace map was added
            mask = np.any(movement)
            if np.any(mask):
                obs[mask, :] = movement[mask]
            return dict(obs=obs, **inputs)
        elif self.mode == "stacked":
            obs = np.concatenate((obs, movement), axis=0)
            return dict(obs=obs, **inputs)

        return dict(obs=obs, movement=movement, **inputs)


class wrapper_no_mod(_DecoratorContextManager):
    def __init__(self):
        self.prev = ACTIVE_STATUS

    def __enter__(self):
        global ACTIVE_STATUS
        self.prev = ACTIVE_STATUS
        ACTIVE_STATUS = False

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        global ACTIVE_STATUS
        ACTIVE_STATUS = self.prev