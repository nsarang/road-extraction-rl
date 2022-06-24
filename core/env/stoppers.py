from __future__ import annotations
from networkx.algorithms.triads import all_triads
import numpy as np


class Stopper:
    """A component for determining if the environment satisfies a defined
    stopping criteria.
    """

    def stop(self, env: "RoadMasterEnv") -> bool:
        """Computes if the environment satisfies the defined stopping criteria.

        Parameters
        ----------
        env : `RoadMasterEnv`
            The road environment.

        Returns
        -------
        bool
            If the environment should stop or continue.
        """
        return False

    def reset(self) -> None:
        """Resets the stopper."""
        pass


class MaxPenaltyTotal(Stopper):
    def __init__(self, max_negative) -> None:
        super().__init__()
        self.max_negative = max_negative

    def stop(self, env: "RoadMasterEnv") -> bool:
        rewards = env.logs["reward_total"]
        total = sum(rewards)
        done = total < self.max_negative
        env.log({"stop": done})
        return done


class MaxPenaltyLatest(Stopper):
    def __init__(self, max_negative, window) -> None:
        super().__init__()
        self.max_negative = max_negative
        self.window = window

    def stop(self, env: "RoadMasterEnv") -> bool:
        rewards = env.logs["reward_total"]
        window = rewards[-self.window :]
        total = sum(window)
        done = total < self.max_negative
        env.log({"stop": done})
        return done


class WindowAverage(Stopper):
    def __init__(
        self, window_length, gameover_multiplier, min_average=None, min_percent=None, oscillation_window=5
    ) -> None:
        super().__init__()
        assert (min_average or min_percent) and not (min_average and min_percent)
        self.min_average = min_average
        self.min_percent = min_percent
        self.window = window_length
        self.gameover_multiplier = gameover_multiplier
        self.oscillation_window = oscillation_window

    def stop(self, env: "RoadMasterEnv") -> bool:
        done = self._window_average_check(env) or self._oscillation_check(env) or self._ending_check(env)
        if done == 1:
            reward = self.gameover_multiplier * self._get_min_reward(env)
            self._log_new_reward(env, reward)
        env.log({"stop_status": done})
        return done

    def _window_average_check(self, env):
        rewards = env.logs["reward_total"]
        avg = sum(rewards[-self.window :]) / self.window  # avg will be smaller if len(r) < window
        if self.min_average:
            done = avg < self.min_average
        else:
            percent = env.reward_scheme.rewardPercentile(avg)
            done = percent < self.min_percent
        return int(done)

    def _oscillation_check(self, env):
        act = env.last("action", size=self.oscillation_window, fillNa=np.nan)
        all_equal = lambda x, v=None: x.count(v or x[0]) == len(x)
        # done = (act[0] != act[1]) and all_equal(act[0::2]) and all_equal(act[1::2]) and (-1 in act[0:2])
        done = any([(all_equal(act[b::2], -1) and (-1 not in act[1 - b :: 2])) for b in [0, 1]])
        locs = env.last("location-agent", size=2*self.oscillation_window)
        done_bbox = len(locs) == 2*self.oscillation_window and len(np.unique(locs, axis=0)) <= 3
        return int(done or done_bbox)

    def _ending_check(self, env):
        status = (
            env.last("reward_flag")
            in [env.reward_scheme.flags.RETRACE_NO_PATH, env.reward_scheme.flags.RETRACE_NO_EASY_PATH]
        ) and (len(env.logs["step_correspondence"]) == 0)
        return 2 if status else 0

    def _log_new_reward(self, env, reward):
        env.logs["reward_total"].pop()
        env.log({"reward_total": reward})

    def _get_min_reward(self, env):
        r_space = env.reward_scheme.reward_space
        return r_space.low[0]
