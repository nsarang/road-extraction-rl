import gym
import numpy as np
from typing import Any, List, Callable, MutableMapping, Optional

from tianshou.env.worker import EnvWorker
from tianshou.data import Batch

class DummyEnvWorker(EnvWorker):
    """Dummy worker used in sequential vector environments."""

    def __init__(self, env_fn: Callable[[], gym.Env]) -> None:
        self.env = env_fn()
        super().__init__(env_fn)

    def __getattr__(self, key: str) -> Any:
        return getattr(self.env, key)

    def reset(self) -> Any:
        return self.env.reset()

    @staticmethod
    def wait(  # type: ignore
        workers: List["DummyEnvWorker"], wait_num: int, timeout: Optional[float] = None
    ) -> List["DummyEnvWorker"]:
        # Sequential EnvWorker objects are always ready
        return workers

    def send_action(self, action: Any) -> None:
        if isinstance(action, list):
            self.result = self.env.step(*action)
        elif isinstance(action, (MutableMapping, Batch)):
            self.result = self.env.step(**action)
        else:
            self.result = self.env.step(action)

    def seed(self, seed: Optional[int] = None) -> List[int]:
        super().seed(seed)
        return self.env.seed(seed)

    def render(self, **kwargs: Any) -> Any:
        return self.env.render(**kwargs)

    def close_env(self) -> None:
        self.env.close()
