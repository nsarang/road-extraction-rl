from __future__ import annotations
from abc import abstractmethod
from typing import Any
from functools import cached_property

import numpy as np
from gym.spaces import Space, Box
from collections import defaultdict
from queue import PriorityQueue
import networkx as nx
from networkx.exception import NetworkXNoPath
from math import sqrt
from enum import Enum
from core.utils.misc import UndoMixin

from shapely.geometry import Polygon
from core.utils.geometry import (
    project2line,
    dist2line,
    dist,
    angle_between,
    path_length,
)


class RewardScheme:
    """A component to compute the reward at each step of an episode."""

    @abstractmethod
    def reward(self, env: "RoadMasterEnv") -> float:
        """Computes the reward for the current step of an episode.

        Parameters
        ----------
        env : `RoadMasterEnv`

        Returns
        -------
        float
            The computed reward.
        """
        raise NotImplementedError()

    def reset(self) -> None:
        """Resets the reward scheme."""
        pass

    @property
    def reward_space(self) -> Space:
        raise NotImplementedError

    def undo(self):
        pass


class AreaLengthReward(RewardScheme):
    def __init__(self, area_weight, length_weight, max_abs_reward, revisit_ratio) -> None:
        super().__init__()
        self.area_weight = area_weight
        self.length_weight = length_weight
        self.max_abs_reward = max_abs_reward
        self.revisit_ratio = revisit_ratio

    def reward(self, env: "RoadMasterEnv"):
        prev_location, location = env.logs["location-agent"][-2:]
        graph_traversed_path = env.logs["traversed-path"][-1]
        out_of_bound = env.logs["out-of-bound"][-1]
        path_revisited = env.logs["path-revisited"][-1]

        movement = (prev_location, location)
        length_reward, info1 = self._get_length_diff_reward(graph_traversed_path, movement)
        area_reward, info2 = self._get_area_reward(graph_traversed_path, movement)

        reward = (self.length_weight * length_reward) + (self.area_weight * -sqrt(-area_reward))

        if path_revisited:
            reward *= self.revisit_ratio
        if out_of_bound:
            reward = -self.max_abs_reward

        reward = max(min(reward, self.max_abs_reward), -self.max_abs_reward)
        info = dict(reward_total=reward, **info1, **info2)
        env.log(info)
        return reward

    def _get_length_diff_reward(self, path: list, movement: tuple):
        movement_length = dist(*movement)
        traversed_length = path_length(path)
        reward = -abs(movement_length - traversed_length)
        info = {
            "reward_length-diff": reward,
            "reward_graph-path-length": traversed_length,
        }
        return reward, info

    def _get_area_reward(self, path: list, movement: tuple):
        reward = 0
        for line_segment in zip(path[:-1], path[1:]):
            a, b = line_segment
            u = project2line(a, movement)
            v = project2line(b, movement)
            poly = Polygon([a, u, v, b])
            reward += -poly.area
        info = {
            "reward_area": reward,
        }
        return reward, info

    def _get_length_ratio_reward(self, path: list, movement: tuple):
        movement_length = dist(*movement)
        if movement_length == 0:
            return 0

        p_length = path_length(path)
        ratio = movement_length / (p_length + np.finfo(float).eps)
        if ratio < 1:
            ratio = 1 / ratio
        reward = -ratio + 1

        info = {"reward_length-ratio": reward, "reward_graph-path-length": p_length}
        return reward, info


class AngleReward(RewardScheme):
    def __init__(self, max_abs_reward, revisit_ratio, divergence_penalty=True) -> None:
        super().__init__()
        self.max_abs_reward = max_abs_reward
        self.revisit_ratio = revisit_ratio
        self.divergence_penalty = divergence_penalty

    def reward(self, env: "RoadMasterEnv"):
        prev_location, location = env.logs["location-agent"][-2:]
        prev_graph_location, graph_location = env.logs["location-graph"][-2:]
        graph_traversed_path = env.logs["traversed-path"][-1]
        out_of_bound = env.logs["out-of-bound"][-1]
        path_revisited = env.logs["path-revisited"][-1]
        divergence_status = False

        movement_length = dist(prev_location, location)
        graph_path_length = path_length(graph_traversed_path)

        movement_vector = np.array(location) - np.array(prev_location)
        desired_vector = np.array(graph_location) - np.array(prev_location)
        angle = angle_between(movement_vector, desired_vector)
        reward = -angle

        if path_revisited:
            reward *= self.revisit_ratio
        if out_of_bound:
            reward = -self.max_abs_reward
        if movement_length < graph_path_length:
            divergence_status = True
            if self.divergence_penalty:
                reward = -self.max_abs_reward

        reward = max(min(reward, self.max_abs_reward), -self.max_abs_reward)
        info = {
            "reward_total": reward,
            "reward_angle_rad": angle,
            "reward_movement-length": movement_length,
            "reward_graph-path-length": graph_path_length,
            "reward_divergence-status": divergence_status,
        }
        env.log(info)
        return reward


class DistReward(RewardScheme):
    def __init__(
        self,
        neutral_distance,
        revisit_distance,
        max_distance,
        nearest_unvisited_len_thresh,
        min_viable_path_len,
        exponent=1,
        normalize=True,
        lane_width=3,
        base_road_width=2,
        apply_road_width=False,
        divergence_criteria=None,
    ) -> None:
        super().__init__()
        self.neutral_distance = neutral_distance
        self.revisit_distance = revisit_distance
        self.max_distance = max_distance
        self.lane_width = lane_width
        self.base_road_width = base_road_width
        self.nearest_unvisited_len_thresh = nearest_unvisited_len_thresh
        self.min_viable_path_len = min_viable_path_len
        self.apply_road_width = apply_road_width
        self.divergence_criteria = divergence_criteria
        self.exponent = exponent
        self.normalize = normalize

    @cached_property
    def reward_space(self) -> Space:
        e = self.exponent
        high = self.neutral_distance ** e
        low = self.neutral_distance ** e - self.max_distance ** e
        if self.normalize:
            max_magnitude = max(abs(low), abs(high))
            high /= max_magnitude
            low /= max_magnitude
        return Box(low=low, high=high, shape=(1,), dtype=np.float32)

    def reward(self, env: "RoadMasterEnv"):
        action = env.logs["action"][-1]
        prev_location, location = env.logs["location-agent"][-2:]
        prev_graph_location, graph_location = env.logs["location-graph"][-2:]
        graph_traversed_path = env.logs["traversed-path"][-1]
        out_of_bound = env.logs["out-of-bound"][-1]
        path_revisited = env.logs["path-revisited"][-1]
        divergence_status = False

        graph = env.action_scheme.graph
        movement_length = dist(prev_location, location)
        dist2road = dist(location, graph_location)

        if self.apply_road_width:
            road_width = self.base_road_width + (
                self.lane_width * list(graph.edges(graph_location, data=True))[0][2]["lanes"]
            )
            dist2road -= road_width / 2
            dist2road = max(dist2road, 0)

        pseudo_d = min(dist2road, self.max_distance)

        if action == -1:
            result = self.dijkstra_viable_path(
                graph,
                source=prev_graph_location,
                max_length_search=self.nearest_unvisited_len_thresh + self.min_viable_path_len,
                min_unvisited_length=self.min_viable_path_len,
            )
            if result is None:  # no viable path
                pass
            else:
                nearest, _, distance_old, _ = result
                try:
                    distance_new = path_length(nx.shortest_path(graph, source=graph_location, target=nearest))
                except NetworkXNoPath:  # fallback
                    distance_new = dist(graph_location, nearest)
                    distance_old = dist(prev_graph_location, nearest)
                if distance_new < distance_old:  # agent is moving towards the target
                    pass
                else:
                    pseudo_d = self.max_distance

            graph_path_length = None
        else:
            graph_path_length = path_length(graph_traversed_path)
            if path_revisited:
                pseudo_d = min(pseudo_d + self.revisit_distance, self.max_distance)
            if out_of_bound:
                pseudo_d = self.max_distance

        pseudo_reward = self.neutral_distance ** self.exponent - pseudo_d ** self.exponent
        if self.normalize:
            pseudo_reward /= max(
                self.max_distance ** self.exponent - self.neutral_distance ** self.exponent,
                self.neutral_distance ** self.exponent,
            )
        reward = pseudo_reward

        # sign = 1 if pseudo_reward > 0 else -1
        # reward = sign * (abs(pseudo_reward) / (self.max_distance - self.neutral_distance)) ** self.exponent

        # divergence = graph_path_length / (movement_length + 1e-8)
        # if self.divergence_criteria and (divergence >= self.divergence_criteria):
        #     reward = -self.max_abs_reward

        info = {
            "reward_total": reward,
            "reward_dist": dist2road,
            # "reward_angle_rad": angle,
            "reward_movement-length": movement_length,
            "reward_graph-path-length": graph_path_length,
            # "reward_divergence": divergence,
        }
        env.log(info)
        return reward

    @staticmethod
    def dijkstra_viable_path(graph, source, max_length_search, min_unvisited_length):
        visited = set()
        cost = defaultdict(lambda: float("inf"))
        unvisited_length = defaultdict(lambda: 0)
        breakpoint = {}
        queue = PriorityQueue()
        cost[source] = 0
        queue.put((0, source))

        while queue.empty() is False:
            _, u = queue.get()
            if cost[u] > max_length_search:  # don't expand if we're too far in
                return
            visited.add(u)
            for v in graph.neighbors(u):
                if v in visited:
                    continue
                length_uv = dist(u, v)
                new_cost = cost[u] + length_uv
                if new_cost < cost[v]:
                    queue.put((new_cost, v))
                    cost[v] = new_cost

                    if graph[u][v]["visited"] == False:  # found an unvisited road
                        breakpoint[v] = u if unvisited_length[u] == 0 else breakpoint[u]
                        unvisited_length[v] = unvisited_length[u] + length_uv
                        if unvisited_length[v] >= min_unvisited_length:
                            target = breakpoint[v]
                            # assert np.allclose(cost[target], cost[v] - unvisited_length[v]) # NOTE: this can be wrong where the path has visited segments
                            return target, v, cost[target], unvisited_length[v]


class DistRewardV2(UndoMixin, RewardScheme):
    def __init__(
        self,
        movement_length,
        traversed_weight,
        revisit_penalty,
        divergence_penalty,
        max_distance,
        max_visited_len_search,  # maximum visited length allowed in dijkstra search
        max_accepted_visited,  # the maximum visited distance allowed for still enforcing exploration
        min_viable_path_len,  # the minimum criteria to identify as a viable path
        exponent=1,
        retrace_diveregence_cooldown=4,  # disables retrace penalty for this many steps
        out_of_bound_coupon_renew=5,  # coupon here means the penalty is disabled when touching the boundary for the first time
        normalize=False,
        lane_width=2,  # NOTE: it's less than the visuliaztion
        base_road_width=2,
        apply_road_width=False,
    ) -> None:
        super().__init__()

        self.movement_length = movement_length
        self.traversed_weight = traversed_weight
        self.revisit_penalty = revisit_penalty
        self.divergence_penalty = divergence_penalty
        self.max_distance = max_distance
        self.lane_width = lane_width
        self.base_road_width = base_road_width
        self.max_visited_len_search = max_visited_len_search
        self.max_accepted_visited = max_accepted_visited
        self.min_viable_path_len = min_viable_path_len
        self.apply_road_width = apply_road_width
        self.exponent = exponent
        self.normalize = normalize
        self.retrace_diveregence_cooldown = retrace_diveregence_cooldown
        self.out_of_bound_coupon_renew = out_of_bound_coupon_renew
        self.out_of_bound_coupons = 1
        self.retrace_cooldown_status = 0

    @cached_property
    def reward_space(self) -> Space:
        e = self.exponent
        high = (self.movement_length * self.traversed_weight) ** e
        low = -(self.max_distance ** e)
        if self.normalize:
            max_magnitude = max(abs(low), abs(high))
            high /= max_magnitude
            low /= max_magnitude
        return Box(low=low, high=high, shape=(1,), dtype=np.float32)

    @cached_property
    def flags(self):
        class RewardFlags(int, Enum):
            RETRACE_CD = 1
            RETRACE_NO_PATH = 2
            RETRACE_TOWARDS_TG = 3
            RETRACE_FORCE_EXPLORE = 4
            RETRACE_NO_EASY_PATH = 5
            NORMAL = 6

        return RewardFlags

    def rewardPercentile(self, reward):
        low, high = self.reward_space.low[0], self.reward_space.high[0]
        percent = (reward - low) / (high - low)
        return percent

    def reward(self, env: "RoadMasterEnv"):
        prev_action_t, action_t = env.last("action_type", 2, fillNa=np.nan)
        location = env.last("location-agent")
        prev_graph_location, graph_location = env.last("location-graph", 2)
        graph_traversed_len = env.last("graph-traversed-len", fillNa=0)
        reward_flag = None

        # OOB coupons
        self._set("out_of_bound_coupons", min(1, self.out_of_bound_coupons + 1))

        # distance to road
        graph = env.action_scheme.graph
        dist2road = dist(location, graph_location)

        # road width consideration
        if self.apply_road_width:
            road_width = self.base_road_width + (
                self.lane_width * list(graph.edges(graph_location, data=True))[0][2]["lanes"]
            )
            dist2road -= road_width / 2
            dist2road = max(dist2road, 0)

        pseudo_d = min(dist2road, self.max_distance)

        # action evaluation
        if action_t == "Retrace":
            originalTime = env.last("step_archived")
            if originalTime and env.logs["divergence-status"][originalTime]:  # don't give penalty on diveregence
                self._set("retrace_cooldown_status", self.retrace_diveregence_cooldown)

            if self.retrace_cooldown_status > 0:  # don't process if it's on cooldown
                self._set("retrace_cooldown_status", self.retrace_cooldown_status - 1)
                reward_flag = self.flags.RETRACE_CD
            else:
                result = DistReward.dijkstra_viable_path(
                    graph,
                    source=prev_graph_location,
                    max_length_search=self.max_visited_len_search + self.min_viable_path_len,
                    min_unvisited_length=self.min_viable_path_len,
                )
                if result is None:  # no viable path to explore
                    reward_flag = self.flags.RETRACE_NO_PATH
                    pass
                else:
                    nearest, _, distance_old, _ = result
                    try:
                        distance_new = path_length(nx.shortest_path(graph, source=graph_location, target=nearest))
                    except NetworkXNoPath:  # fallback
                        distance_new = dist(graph_location, nearest)
                        distance_old = dist(prev_graph_location, nearest)

                    if distance_new < distance_old:  # agent is moving towards the target
                        pseudo_d = pseudo_d // 2  # TODO: evaluate performance
                        reward_flag = self.flags.RETRACE_TOWARDS_TG
                    elif (
                        distance_old <= self.max_accepted_visited
                    ):  # there are some visited edges between the agent and the viable path, but since it's neligible, exploration must be enforced
                        pseudo_d = self.max_distance
                        reward_flag = self.flags.RETRACE_FORCE_EXPLORE
                    else:  # distance_old > self.max_accepted_visited
                        reward_flag = self.flags.RETRACE_NO_EASY_PATH
                        pass

        else:  # normal action
            out_of_bound = env.last("out-of-bound")
            path_revisited = env.last("path-revisited")
            divergence_status = env.last("divergence-status")
            soft_divergence = env.last("soft-divergence")

            if path_revisited:
                if prev_action_t == "Retrace":
                    pseudo_d = (self.max_distance + self.revisit_penalty) / 2  # punish broken retrace chain
                else:
                    pseudo_d += self.revisit_penalty
            if out_of_bound:
                if self.out_of_bound_coupons > 0:
                    self._set("out_of_bound_coupons", -self.out_of_bound_coupon_renew)
                else:
                    pseudo_d = self.max_distance
            if divergence_status and (not soft_divergence):
                pseudo_d += self.divergence_penalty

            pseudo_d = min(pseudo_d, self.max_distance)
            reward_flag = self.flags.NORMAL

        # traveresed reward
        graph_traversed_len = min(graph_traversed_len, self.movement_length)
        traversed_reward = graph_traversed_len * self.traversed_weight
        # final reward
        # TODO: bug when negative to the power of floating value
        pseudo_reward = (traversed_reward ** self.exponent) - (pseudo_d ** self.exponent)

        if self.normalize:
            pseudo_reward /= max(
                (self.movement_length * self.traversed_weight) ** self.exponent,
                self.max_distance ** self.exponent,
            )
        reward = pseudo_reward

        info = {
            "reward_total": reward,
            "reward_dist": dist2road,
            "reward_traversed": traversed_reward,
            "reward_flag": reward_flag,
        }
        env.log(info)
        return reward

    def _set(self, name: str, value: Any) -> None:
        if hasattr(self, name):
            curValue = getattr(self, name)
            self.add_revert(setattr, self, name, curValue, __copy__=False)
        else:
            self.add_revert(delattr, self, name, __copy__=False)
        setattr(self, name, value)


class MaskReward(RewardScheme):
    def __init__(self, revisit_weight, error_weight) -> None:
        super().__init__()
        self.revisit_weight = revisit_weight
        self.error_weight = error_weight

    def reward(self, env: "RoadMasterEnv"):
        newly_visited = env.logs["pixels_newly-visited"][-1]
        revisited = env.logs["pixels_revisited"][-1]
        error = env.logs["pixels_error"][-1]
        out_of_bound = env.logs["out-of-bound"][-1]

        reward = newly_visited - (self.revisit_weight * revisited) - (self.error_weight * error)
        env.log({"reward_total": reward})
        return reward
