from __future__ import annotations
from typing import Dict, Any, Tuple, Union, Optional, List

from abc import abstractmethod, ABC
from functools import cached_property

import random
import numpy as np
import networkx as nx
from networkx.exception import NetworkXNoPath
from gym.spaces import Space, Box, Discrete, Dict as DictSpace

from copy import deepcopy
from numbers import Number

from core.data.utils import graph_crop
from core.env.drawers import Drawer
from core.utils.search import SearchTree
from core.utils.geometry import is_in_bound, polar2cartesian, project2line, dist2line, dist, path_length
from core.utils.misc import UndoMixin


class ActionScheme(ABC):
    """A component for determining the action to take at each step of an
    episode.
    """

    @property
    @abstractmethod
    def action_space(self) -> Space:
        """The action space of the `RoadMasterEnv`. (`Space`, read-only)"""
        pass

    @abstractmethod
    def perform(self, env: "RoadMasterEnv", action: Any) -> None:
        """Performs an action on the environment.

        Parameters
        ----------
        env : `RoadMasterEnv`
            The environment to perform the `action` on.
        action : Any
            The action to perform on `env`.
        """
        pass

    @abstractmethod
    def start_location(self) -> tuple:
        """Picks a starting location."""
        pass

    @property
    def gtGraph(self):
        """Groundtruth road network graph"""
        return self._gt_graph

    @cached_property
    def log_keys(self):
        return []

    def reset(self) -> None:
        """Resets the action scheme."""
        pass

    def undo(self, env: "RoadMasterEnv"):
        pass


class GraphBasedScheme(UndoMixin, ActionScheme):
    def __init__(
        self,
        road_graph,
        divergence_tolerance=3,
        line_width=5,
        num_rtree_search=50,
        flex_assignment=0.6,
        min_lane_dist=15,
        min_start_loc_space=30,
    ) -> None:
        super().__init__()

        self._gt_graph = road_graph
        self.line_width = line_width
        self.divergence_tolerance = divergence_tolerance
        self.num_rtree_search = num_rtree_search
        self.flex_assignment = flex_assignment
        self.min_lane_dist = min_lane_dist
        self.min_start_loc_space = min_start_loc_space
        self.graph = None
        self.search_tree = SearchTree(road_graph.nodes)

    def perform(self, env: "RoadMasterEnv", direction: tuple):
        dx, dy = direction
        next_location = (env.location[0] + dx, env.location[1] + dy)

        # boundary check
        next_location, out_of_bound = self._enforce_boundary(next_location, env.bounds)

        # find the corresponding street on the graph
        (
            edge,
            traversed_path,
            next_gLocation,
            divergence_flag,
            soft_divergence,
            revisited_flag,
        ) = self._graph_position_assignment(env.location, next_location, env.gLocation)
        graph_traversed_len = path_length(traversed_path) if len(traversed_path) >= 2 else 0

        # drawings
        env.movement_drawer.rounded_line(env.location, next_location, width=self.line_width, color=(255, 255, 255))

        # logging
        logs = {
            "location-agent": next_location,
            "location-graph": next_gLocation,
            "out-of-bound": out_of_bound,
            "path-revisited": revisited_flag,
            "divergence-status": divergence_flag,
            "soft-divergence": soft_divergence,
            "graph-traversed-len": graph_traversed_len,
            "graph-path-num-edges": len(traversed_path) - 1,
            "traversed-path": traversed_path,
        }
        env.log(logs)

        # new node added
        if next_gLocation not in edge:

            def _r_node(edge, projection):
                (u, v), mid = edge, projection
                attrs = self.graph[u][mid]
                self.graph.add_edge(u, v, **attrs)
                self.graph.remove_node(mid)
                self.search_tree.undo(count=1)

            self.add_revert(_r_node, edge, next_gLocation)
        # visited attrs modified
        if divergence_flag == False:
            self.add_revert(self._update_visited_flag, traversed_path, -1)

    def undo(self, env):
        super().undo()

    def reset(self, bounds=None) -> None:
        if bounds is None:
            self.graph = deepcopy(self._gt_graph)
        else:
            self.graph = graph_crop(self._gt_graph, bounds)
        self.search_tree = SearchTree(self.graph.nodes)
        nx.set_edge_attributes(self.graph, name="visited", values=0)
        self._history.clear()

    def start_location(self):
        degrees = self.gtGraph.degree()

        def _filter(node, deg_thresh=2):
            degs = np.array([degrees[node], *[degrees(x) for x in self.gtGraph.neighbors(node)]])
            dists = np.array([dist(*e) for e in self.gtGraph.edges(node)])
            return np.all(degs >= deg_thresh) and np.all(dists >= self.min_start_loc_space)

        candids = [x for x in self.gtGraph.nodes if _filter(x)]
        return random.choice(candids)

    def exportGraph(self, env):
        g = nx.Graph()
        for (u, v) in zip(env.logs["location-agent"][:-1], env.logs["location-agent"][1:]):
            g.add_edge(u, v)
        return g

    def _enforce_boundary(self, location, bounds):
        if is_in_bound(location, bounds, exclusive_end=True) == False:
            x, y = location
            x1_b, y1_b, x2_b, y2_b = bounds
            x_next = min(max(x1_b, x), x2_b - 1)
            y_next = min(max(y1_b, y), y2_b - 1)
            next_location = (x_next, y_next)
            return next_location, True
        return location, False

    def _graph_position_assignment(self, location, next_location, gLocation, topK=5):
        candid_edges: list[tuple] = self._get_closest_road_segments(next_location)[:topK]
        paths: list[list] = [self._get_traversed_path(gLocation, next_location, e[0]) for e in candid_edges]
        divergence_flags: list[bool] = [self._divergence_check(location, next_location, p) for p in paths]

        selected_idx = 0
        soft_divergence = False

        if (not np.all(divergence_flags)) and (divergence_flags[0] == True):
            viable_idx = np.argmin(divergence_flags)
            dist_div, dist_r, = (
                candid_edges[0][1],
                candid_edges[viable_idx][1],
            )
            proj_div = paths[0][-1] if len(paths[0]) else project2line(next_location, candid_edges[0][0])
            proj_r = paths[viable_idx][-1]
            dist_roads = dist(proj_div, proj_r)
            if dist_roads <= self.min_lane_dist:  # it's too tight!
                soft_divergence = True  # used in reward scheme
            elif dist_r / (dist_r + dist_div) <= self.flex_assignment:  # flexible assignment / not a divergence
                selected_idx = viable_idx

        edge = candid_edges[selected_idx][0]
        path = paths[selected_idx]
        projection = path[-1] if len(path) else project2line(next_location, edge)
        divergence_flag = divergence_flags[selected_idx]

        # split edge
        if projection not in edge:
            self._split_partily_visited_edge(*edge, projection)
        # revisitation check
        revisited_flag = self._revisitation_check(path, edge, projection)
        # update visited counter
        if divergence_flag == False:
            self._update_visited_flag(path, +1)

        return edge, path, projection, divergence_flag, soft_divergence, revisited_flag

    def _revisitation_check(self, path, edge, projection):
        if len(path) == 0:
            return any(self.graph[u][projection]["visited"] for u in edge if u != projection)
        if len(path) == 1:
            return True
        return sum(dist(p, q) * (1 if self.graph[p][q]["visited"] else -1) for p, q in zip(path[:-1], path[1:])) > 0

    def _update_visited_flag(self, path, value):
        for p, q in zip(path[:-1], path[1:]):
            self.graph[p][q]["visited"] += value

    def _divergence_check(self, location, next_location, path):
        if len(path) == 0:  # component jump
            return True
        elif len(path) == 1:  # orthogonal movement
            return False
        else:
            movement_length = dist(location, next_location)
            graph_traversed_len = path_length(path)
            return graph_traversed_len > movement_length * self.divergence_tolerance

    def _get_traversed_path(self, prev_graph_location, next_location, edge):
        u, v = edge
        projection = project2line(next_location, edge)
        try:
            if prev_graph_location == projection:  # orthogonal movement
                traversed_path = [projection]
            else:
                traversed_path = nx.shortest_path(self.graph, source=prev_graph_location, target=u)
                # uv orientation correction
                if (
                    len(traversed_path) >= 2 and traversed_path[-2] == v
                ):  # v is closer to previous location + u overlaps with the projection
                    u, v = v, u
                    traversed_path = traversed_path[:-1]
                # add the endpoint
                if projection != u:
                    traversed_path += [projection]
        except NetworkXNoPath:
            traversed_path = []

        return traversed_path

    def _get_closest_road_segments(self, point):
        indices = self.search_tree.nearest(
            point, num_results=self.num_rtree_search
        )  # get the neighbor nodes in the graph
        indices = list(indices)
        neighbors = np.array(self.graph.nodes)[indices]

        close_edges = [list(self.graph.edges(tuple(node))) for node in neighbors]  # get list of edges per node
        close_edges = sum(close_edges, [])  # concatenate lists
        close_edges = [tuple(sorted(x)) for x in close_edges]  # sort the endpoints of each edge
        close_edges = set(close_edges)  # remove duplicates

        edges_sorted = sorted(
            [(e, dist2line(point, e)) for e in close_edges],
            key=lambda x: x[1],
        )
        return edges_sorted

    def _split_partily_visited_edge(self, u, v, mid):
        attrs = self.graph[u][v]
        self.graph.remove_edge(u, v)
        self.graph.add_edge(u, mid, **attrs)
        self.graph.add_edge(mid, v, **attrs)
        # add the node the search tree
        self.search_tree.insert(len(self.graph.nodes) - 1, mid)


class MaskBasedScheme(ActionScheme):
    def __init__(self, size, road_graph, lane_width=3, base_width=2) -> None:
        super().__init__()
        self.size = size
        self.lane_width = lane_width
        self.base_width = base_width
        self._gt_graph = road_graph
        self._gt_mask = type(self).generate_gt_mask(size, road_graph, lane_width, base_width)

    @property
    def mask(self):
        return self._gt_mask

    @staticmethod
    def generate_gt_mask(size, graph, lane_width, base_width):
        drawer = Drawer(size, mode="F")

        for u, v, attrs in graph.edges(data=True):
            width = base_width + lane_width * attrs["lanes"]
            drawer.line(u, v, width, color=255)

        mask = np.array(drawer.canvas)
        return mask

    def perform(self, env: "RoadMasterEnv", direction: tuple):
        out_of_bound = False

        dx, dy = direction
        location = env.location
        next_location = (int(location[0] + dx), int(location[1] + dy))

        # out of bound check
        if is_in_bound(next_location, env.bounds, exclusive_right=True) == False:
            out_of_bound = True
            x, y = next_location
            x1_b, y1_b, x2_b, y2_b = env.bounds
            x_next = min(max(x1_b, x), x2_b - 1)
            y_next = min(max(y1_b, y), y2_b - 1)
            next_location = (x_next, y_next)

        # get maps
        obs_bbox, _ = env.get_observation_window_coords()
        (x1, y1, x2, y2) = obs_bbox
        obs_size = ((x2 - x1), (y2 - y1))

        movement = (
            Drawer(obs_size, "F")
            .line(
                u=(location[0] - x1, location[1] - y1),
                v=(next_location[0] - x1, next_location[1] - y1),
                width=self.lane_width,
                color=255,
            )
            .window((0, 0, x2 - x1, y2 - y1), mode="numpy")
        ).astype(bool)
        movement_history = env.movement_drawer.window(obs_bbox, mode="numpy").astype(bool)
        groundtruth = self.mask[y1:y2, x1:x2].astype(bool)

        # comparison
        curr_visited = groundtruth & movement
        prev_visited = groundtruth & movement_history
        revisited = curr_visited & prev_visited
        newly_visited = curr_visited & (~revisited)
        error = movement & (~curr_visited)

        scale = movement.sum() + 1e-8
        revisited_density = revisited.sum() / scale
        newly_visited_density = newly_visited.sum() / scale
        error_density = error.sum() / scale

        # drawings
        env.movement_drawer.rounded_line(env.location, next_location, width=self.lane_width, color=(255, 255, 255))
        # env.renderer.rounded_line(env.location, next_location, width=self.lane_width, color=(255, 255, 255))

        # logging
        logs = {
            "location-agent": next_location,
            "out-of-bound": out_of_bound,
            "pixels_revisited": revisited_density,
            "pixels_newly-visited": newly_visited_density,
            "pixels_error": error_density,
        }
        env.log(logs)

    def start_location(self) -> tuple[int, int]:
        filled = np.argwhere(self.mask)
        idx = np.random.choice(filled.shape[0])
        loc = tuple(filled[idx][::-1])  # (cx, cy)
        return loc

    def reset(self) -> None:
        pass


class ContinuousSpace(ActionScheme):
    def __init__(self, step_size: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.step_size = step_size

    @property
    def action_space(self):
        return Box(low=-1, high=1, shape=(1,), dtype=np.float32)

    def translate_action(self, action):
        theta = action * np.pi
        dx, dy = polar2cartesian(self.step_size, theta)
        return (dx, -dy)

    def perform(self, env: "RoadMasterEnv", action: Any):
        direction = self.translate_action(action)
        env.log({"action": action, "action_type": "Normal", "direction": direction})
        return super().perform(env, direction)


class DiscreteSpace(ActionScheme):
    def __init__(self, step_size: float, num_directions, **kwargs) -> None:
        super().__init__(**kwargs)
        self.step_size = step_size
        self.directions = np.array(
            [(np.cos(d), np.sin(d)) for d in np.linspace(0, 2 * np.pi, num=num_directions, endpoint=False)]
        )
        # self.directions = [
        #     unit_vector((i, j)) for i in range(-1, 2) for j in range(-1, 2) if i or j
        # ]

    @property
    def action_space(self):
        return Discrete(len(self.directions))

    def translate_action(self, action):
        dx, dy = self.step_size * self.directions[action]
        return (dx, -dy)

    def perform(self, env: "RoadMasterEnv", action: Any):
        direction = self.translate_action(action)
        env.log({"action": action, "action_type": "Normal", "direction": direction})
        return super().perform(env, direction)


class DiscreteRetraceSpace(DiscreteSpace):
    @property
    def action_space(self):
        return Discrete(1 + len(self.directions))

    def retrace(self, env):
        stepTracker = env.logs["step_correspondence"]
        if len(stepTracker):  # pop the current location
            lastLoc = stepTracker.pop(iloc=True)
            env.log({"step_archived": lastLoc})

        currLoc = stepTracker.iloc[-1] if len(stepTracker) else 0
        next_location = env.logs["location-agent"][currLoc]
        next_gLocation = env.logs["location-graph"][currLoc]

        env.retrace_drawer.rounded_line(env.location, next_location, width=self.line_width, color=255)
        env.log(
            {
                "action": -1,
                "action_type": "Retrace",
                "direction": None,
                "location-agent": next_location,
                "location-graph": next_gLocation,
            }
        )

    def perform(self, env: "RoadMasterEnv", action: Any):
        if action == 0:
            return self.retrace(env)
        else:
            env.log({"step_correspondence": env.steps})
            return super().perform(env, action - 1)

    def undo(self, env):
        if env.last("action_type") != "Retrace":
            return super().undo(env)


class ContinuousRetraceSpace(ContinuousSpace):
    @cached_property
    def action_space(self):
        return DictSpace({"retrace": Discrete(2), "direction": Box(-1, 1, shape=(1,))})

    def perform(self, env: "RoadMasterEnv", action: Any):
        retrace = action["retrace"]
        if isinstance(retrace, np.ndarray):
            retrace = np.argmax(retrace)
        if retrace == 1:
            return DiscreteRetraceSpace.retrace(self, env)
        else:
            env.log({"step_correspondence": env.steps})
            return super().perform(env, action["direction"])

    def undo(self, env):
        if env.last("action_type") != "Retrace":
            return super().undo(env)


# MIXINS
GraphContinuous = type("GraphContinuous", (ContinuousSpace, GraphBasedScheme), {})
GraphDiscrete = type("GraphDiscrete", (DiscreteSpace, GraphBasedScheme), {})
GraphDiscreteRetrace = type("GraphDiscreteRetrace", (DiscreteRetraceSpace, GraphBasedScheme), {})
GraphContinuousRetrace = type("GraphContinuousRetrace", (ContinuousRetraceSpace, GraphBasedScheme), {})
MaskContinuous = type("MaskContinuous", (ContinuousSpace, MaskBasedScheme), {})
MaskDiscrete = type("MaskDiscrete", (DiscreteSpace, MaskBasedScheme), {})
