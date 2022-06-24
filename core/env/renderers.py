from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Union, Tuple, List

import os
import numpy as np
import scipy
import ffmpeg
from PIL import Image, ImageOps
from skimage.transform import rescale
from math import ceil
from imageio import mimsave
from copy import deepcopy
from collections import deque
from gym.spaces import Discrete

from IPython.display import display, clear_output
import matplotlib.pylab as plt
import matplotlib as mpl
import plotly.graph_objects as go
from cairosvg import svg2png
from io import BytesIO

from core.env.drawers import Drawer, LazyDrawer, LazyIntensityDrawer
from core.env.actions import ContinuousSpace, DiscreteRetraceSpace
from core.utils.math import symceil
from core.utils.file import check_path


class Renderer(ABC):
    """A component for rendering a view of the environment at each step of
    an episode."""

    @abstractmethod
    def render(self, env, **kwargs):
        """Renders a view of the environment at the current step of an episode."""
        raise NotImplementedError()

    def save(self, env, **kwargs) -> None:
        """Saves the rendered view of the environment."""
        pass

    def reset(self) -> None:
        """Resets the renderer."""
        pass

    def close(self) -> None:
        """Closes the renderer."""
        pass


class AggregateRenderer(Renderer):
    """A renderer that aggregates compatible renderers so they can all be used
    to render a view of the environment.

    Parameters
    ----------
    renderers : List[Renderer]
        A list of renderers to aggregate.

    Attributes
    ----------
    renderers : List[Renderer]
        A list of renderers to aggregate.
    """

    def __init__(self, renderers: List[Renderer]) -> None:
        super().__init__()
        self.renderers = renderers

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.location_drawer, name)

    def render(self, env, **kwargs) -> None:
        for r in self.renderers:
            r.render(env, **kwargs)

    def save(self, env, **kwargs) -> None:
        for r in self.renderers:
            r.save(env, **kwargs)

    def reset(self) -> None:
        for r in self.renderers:
            r.reset()

    def close(self) -> None:
        for r in self.renderers:
            r.close()


class GraphRenderer(Renderer):
    def __init__(self, road_network, size, edge_width=3, base_width=2, apply_lane_width=True):
        self.size = size
        self.edge_width = edge_width
        self.base_width = base_width
        self.apply_lane_width = apply_lane_width

        self.graph_drawer = LazyDrawer(size)
        type(self).draw_graph(
            self.graph_drawer,
            road_network,
            draw_edges=True,
            draw_nodes=True,
            edge_width=edge_width,
            base_width=base_width,
            apply_lane_width=apply_lane_width,
        )
        self.location_drawer = LazyDrawer(self.size)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.location_drawer, name)

    @staticmethod
    def draw_graph(
        drawer: LazyDrawer,
        road_network,
        draw_edges,
        draw_nodes,
        edge_width,
        apply_lane_width,
        base_width,
        edge_color="#f48c06",
        node_color="#6930c3",
    ):
        if draw_edges:
            for u, v, attrs in road_network.edges(data=True):
                width = base_width + edge_width
                if apply_lane_width:
                    width += edge_width * (attrs["lanes"] - 1)
                drawer.line(u, v, width, edge_color)

        if draw_nodes:
            for v in road_network.nodes:
                drawer.circle(v, 3, node_color)

    def render(self, env: "RoadMasterEnv", alpha=0.8, **kwargs):
        bbox_r, (
            x1_o,
            y1_o,
            x2_o,
            y2_o,
        ) = env.get_observation_window_coords()

        d = env.observe(mode="rgb+movement")
        obs, movement = d["obs"], d["movement"]
        graph = np.asarray(self.graph_drawer.window(bbox_r))
        locations = np.asarray(self.location_drawer.window(bbox_r))

        graph_mask = graph[..., -1] > 0
        location_mask = locations[..., -1] > 0
        obs[y1_o:y2_o, x1_o:x2_o][graph_mask] = (
            obs[y1_o:y2_o, x1_o:x2_o][graph_mask] * (1 - alpha) + graph[graph_mask, :3] * alpha
        )
        obs[y1_o:y2_o, x1_o:x2_o][location_mask] = locations[location_mask, :3]

        movement = movement[..., 0]
        mask = movement > 0
        if np.any(mask):
            obs[mask, :] = movement[mask, None]

        return obs

    def reset(self):
        self.location_drawer.reset()

    def close(self):
        self.road_network = None
        self.graph_drawer = None
        self.location_drawer = None


class PlotlyGraphRenderer(GraphRenderer):
    def __init__(self, road_network, size, display):
        super().__init__(road_network, size)
        self.display = display
        self._show_chart = None
        self._fig = None

    def render(self, *args, **kwargs):
        obs = super().render(*args, **kwargs)

        if self._show_chart:
            display(self.fig)
            self._show_chart = False

        self.fig.data[0].update({"z": obs})

    def reset(self):
        super().reset()
        self._fig = self._create_figure()
        if self.display:
            clear_output(wait=True)
            self._show_chart = True

    def close(self):
        super().close()
        self._fig = None

    def _create_figure(self):
        self.fig = go.FigureWidget(go.Image())
        self.fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
        self.fig.update_layout(
            autosize=False,
            width=400,
            height=400,
            margin=dict(
                l=10,
                r=10,
                b=10,
                t=10,
            ),
        )


class VideoRenderer(Renderer):
    def __init__(
        self,
        baseRenderer,
        mode: str,
        max_stack_size=200,
        fps=4,
        save_dir="logs/gifs/",
    ):
        super().__init__()
        self.baseRenderer = baseRenderer
        self.stack = deque(maxlen=max_stack_size)
        self.mode = mode
        self.fps = fps
        self.save_dir = save_dir
        check_path(save_dir)

    def __getattr__(self, name):
        if not hasattr(self.baseRenderer, name):
            raise AttributeError(f"Attribute '{name}' not found")
        return getattr(self.baseRenderer, name)

    def render(self, *args, **kwargs):
        obs = self.baseRenderer.render(*args, **kwargs)
        self.stack.append(obs.astype(np.uint8))

    def reset(self):
        self.baseRenderer.reset()
        self.stack.clear()

    def close(self):
        self.baseRenderer.close()
        self.stack.clear()

    def save(self, env: "RoadMasterEnv", mode=None, **kwargs):
        mode = mode or self.mode
        prefix = env.name + "_"

        if len(self.stack) == 0:
            return
        if self.mode == "gif":
            filepath = os.path.join(self.save_dir, env.savename + ".gif")
            type(self).save_gif_pillow(filepath, self.stack, self.fps)

        elif self.mode == "stack":
            return np.concatenate(self.stack)
        else:
            raise ValueError

    @staticmethod
    def save_gif_pillow(filepath, images, fps):
        first, *others = [Image.fromarray(img) for img in images]
        first.save(
            fp=filepath,
            format="GIF",
            append_images=others,
            save_all=True,
            duration=int(1000 / fps),
            loop=0,
        )

    @staticmethod
    def save_gif_imageio(filepath, images, fps):
        mimsave(filepath, images, fps=fps)


class VideoRenderer9000(VideoRenderer):
    def __init__(
        self,
        frame_size,
        gamma,
        heatmap_window=10,
        reward_window=30,
        rewards_mid_point=None,
        skip_rate=0,
        min_steps_activate=2,
        max_frames=np.inf,
        **kwargs,
    ):
        super().__init__(mode=None, **kwargs)
        self.frame_size = frame_size
        self.gamma = gamma
        self.heatmap_window = heatmap_window
        self.reward_window = reward_window
        self.rewards_mid_point = rewards_mid_point
        self.skip_rate = skip_rate
        assert min_steps_activate >= 2
        self.min_steps_activate = min_steps_activate
        self.max_frames = max_frames
        self.plot_ranges = np.array(
            [
                [[0.025, 0.6], [0.025, 0.6]],  # obs
                [[0.6, 0.975], [0.025, 0.4]],  # heatmap
                [[0.6, 0.975], [0.4, 0.775]],  # logits
                [[0.025, 0.65], [0.65, 0.975]],  # rewards
            ]
        )

    def render(self, env, **kwargs):
        raise NotImplementedError("VideoRenderer9000 doesn't implement 'render' method")

    def save(self, env: "RoadMasterEnv", mode=None, **kwargs):
        mode = mode or self.mode
        filepath = os.path.join(self.save_dir, env.savename + ".mp4")

        start_step = env.last("UndoW_BP", fillNa=0)
        end_step = env.steps
        total_frames = end_step - start_step + 1
        if total_frames < self.min_steps_activate:
            return

        # DATA
        action_space = env.action_scheme.action_space
        reward_space = env.reward_scheme.reward_space
        assert isinstance(action_space, Discrete)
        reward_range = (reward_space.low[0], reward_space.high[0])

        logits_range = (1 / (1 - self.gamma)) * np.array(reward_range)
        logits_range[0] = logits_range[0] * 1 / 2  # TODO: no hacks
        logits_range = symceil(logits_range, decimals=0).astype(float)

        if env.reward_scheme.exponent <= 1:
            rewards_plot_domain = logits_range
            # rewards_plot_domain = logits_range + [-1, 1]
        else:
            # rewards_plot_domain = symceil(logits_range / 2, decimals=1)
            rewards_plot_domain = [reward_range[0] - 0.1, logits_range[1] + 0.1]

        logits = np.array(env.logs["logits"][:])
        if logits.ndim == 3:
            logits = logits[..., env.last("target_horizon")]
        logits = np.concatenate((logits, logits[-1:]))
        rewards = np.array(env.logs["reward_total"][:])
        rewards = np.concatenate(([0], rewards))

        if isinstance(env.action_scheme, DiscreteRetraceSpace):
            logits = logits[:, 1:]

        # ENCODER
        process = (
            ffmpeg.input(
                "pipe:",
                framerate="{}".format(self.fps),
                format="rawvideo",
                pix_fmt="rgb24",
                s="{}x{}".format(self.frame_size[0], self.frame_size[1]),
            )
            .output(
                filepath,
                pix_fmt="yuv420p",
                vcodec="libx264",
                crf=23,
                movflags="+faststart",
            )
            .overwrite_output()
            .run_async(pipe_stdin=True, quiet=True)
        )

        num_actual_frames = min(ceil(total_frames * (1 - self.skip_rate)), self.max_frames)
        selected_indices = np.round(np.arange(0, num_actual_frames) * (total_frames / num_actual_frames)).astype(int)

        replays = self.replay_episode(env)
        for idx in range(total_frames):
            obs = next(replays)
            if idx not in selected_indices:
                continue

            im = self._create_viz(
                step=idx + start_step,
                obs=obs,
                logits=logits,
                rewards=rewards,
                logits_range=logits_range,
                rewards_plot_domain=rewards_plot_domain,
            )
            process.stdin.write(im.tobytes())

        process.stdin.close()
        process.wait()

    def replay_episode(self, env):
        locations = env.history["all"]["location-agent"][:]
        graph_locations = env.history["all"]["location-graph"][:]
        actions_t = env.history["all"]["action_type"][:]

        # movement_drawer = LazyIntensityDrawer(env.size)
        graph_drawer = self.graph_drawer
        location_drawer = self.location_drawer
        location_drawer.reset()
        line_width = self.base_width + self.edge_width

        def render(location, alpha=0.8):
            bbox_r, bbox_o = env.get_observation_window_coords(location)
            x1_r, y1_r, x2_r, y2_r = bbox_r
            x1_o, y1_o, x2_o, y2_o = bbox_o

            o_width, o_height = env.observation_shape
            obs = np.zeros((o_height, o_width, 3), dtype=np.uint8)
            obs[y1_o:y2_o, x1_o:x2_o] = env.rgb[y1_r:y2_r, x1_r:x2_r]
            graph = np.asarray(graph_drawer.window(bbox_r))
            graph_mask = graph[..., -1] > 0
            obs[y1_o:y2_o, x1_o:x2_o][graph_mask] = (
                obs[y1_o:y2_o, x1_o:x2_o][graph_mask] * (1 - alpha) + graph[graph_mask, :3] * alpha
            )

            # movement = np.zeros((o_height, o_width))
            # movement_raw = movement_drawer.window((x1_r, y1_r, x2_r, y2_r), mode="intensity-gray")
            # movement[y1_o:y2_o, x1_o:x2_o] = movement_raw
            # obs[movement > 0, :] = 255

            # graph locations
            locations = np.asarray(location_drawer.window(bbox_r))
            mask = locations[..., -1] > 0
            obs[y1_o:y2_o, x1_o:x2_o][mask] = locations[mask, :3]
            return obs

        # real start point
        real_start_point = len(locations) - len(env.logs["location-agent"]) + 1  # soft reset diff
        real_start_point += env.last("UndoW_BP", fillNa=0)

        # draw history
        location_drawer.circle(graph_locations[0], radius=2.5, color=(255, 0, 0))
        for idx in range(1, real_start_point):
            # movement_drawer.rounded_line(locations[idx - 1], locations[idx], width=line_width, color=(255, 255, 255))
            location_drawer.circle(graph_locations[idx], radius=2.5, color=(255, 0, 0))
            color = (255, 255, 255) if actions_t[idx - 1] == "Normal" else (0, 255, 0)
            location_drawer.rounded_line(locations[idx - 1], locations[idx], width=line_width + 1, color=color)

        # first obs
        yield render(locations[real_start_point - 1])

        for idx in range(real_start_point, len(locations)):
            # movement_drawer.rounded_line(locations[idx - 1], locations[idx], width=line_width, color=(255, 255, 255))
            location_drawer.circle(graph_locations[idx], radius=2.5, color=(255, 0, 0))
            color = (255, 255, 255) if actions_t[idx - 1] == "Normal" else (0, 255, 0)
            location_drawer.rounded_line(locations[idx - 1], locations[idx], width=line_width + 1, color=color)
            yield render(locations[idx])

    def _create_viz(self, step, obs, logits, rewards, logits_range, rewards_plot_domain):
        obs_image = type(self).frame(Image.fromarray(obs))
        heatmap_img = Image.open(
            BytesIO(
                self._plot_direction_heatmap(
                    predictions=logits[max(0, step - self.heatmap_window + 1) : step + 1],
                    window=self.heatmap_window,
                ).to_image(format="png", engine="kaleido")
            )
        )
        mid_point = self.rewards_mid_point or round(((logits_range[1] + logits_range[0]) / 2), ndigits=1)
        logits_img = Image.open(
            BytesIO(
                self._plot_q_values(
                    logits=logits[step],
                    logits_range=logits_range,
                    rmid=mid_point,
                    circle_width=5,
                ).to_image(format="png", engine="kaleido")
            )
        )
        rew_img = Image.open(
            BytesIO(
                self._plot_reward_history(
                    rewards,
                    gamma=self.gamma,
                    window=self.reward_window,
                    logits_range=rewards_plot_domain,
                    step_num=step,
                ).to_image(format="png", engine="kaleido")
            )
        )
        width, height = self.frame_size
        canvas = Image.new("RGBA", (width, height), "WHITE")

        for image, coords in zip([obs_image, heatmap_img, logits_img, rew_img], self.plot_ranges):
            lx, ly = ((coords[:, 1] - coords[:, 0]) * [width, height]).astype(int)
            ox, oy = (coords[:, 0] * [width, height]).astype(int)
            im = image.resize((lx, ly), resample=Image.BILINEAR)
            canvas.paste(im, (ox, oy), im)

        viz = np.asarray(canvas.convert("RGB"))
        return viz

    def _plot_direction_heatmap(self, predictions, window, scale=1):
        num_dirs = predictions.shape[1]
        b_width = 360 / num_dirs
        theta = np.linspace(0, 360, num=num_dirs, endpoint=False)

        fig = go.FigureWidget()

        assert len(predictions) <= window
        values = np.zeros((window, num_dirs))
        values[-len(predictions) :] = predictions
        # values = values[::-1] # from center to outer

        for idx in range(window):
            logits = values[idx]
            l_normal = (logits - logits.min()) / (logits.max() - logits.min() + 1e-8)
            colors = [type(self).color2str(c) for c in mpl.cm.get_cmap("plasma")(l_normal)]

            bar = go.Barpolar(
                r=np.full(num_dirs, 1),
                theta=theta,
                width=np.full(num_dirs, b_width),
                marker=dict(
                    color=colors,
                    line=dict(color=colors, width=2),
                ),
                opacity=0.95,
                subplot="polar",
            )
            fig.add_trace(bar)

        fig.update_layout(
            title=dict(
                text="<b>Direction Heatmap</b><br>" f"<i>(window: {window})</i>",
                font=dict(family="Comic Sans MS", color="black", size=52 * scale),
            ),
            template=None,
            showlegend=False,
            polar=dict(
                radialaxis=dict(range=[0, window], showticklabels=False, ticks=""),
                angularaxis=dict(ticks=""),  # , showticklabels=False)
            ),
            font=dict(size=35 * scale),
            margin=dict(t=290 * scale),
            width=1200 * scale,
            height=1200 * scale,
        )
        return fig

    def _plot_q_values(
        self,
        logits,
        logits_range,
        rmid,
        zero_circle_points=100,
        circle_width=2,
        scale=1,
    ):
        rmin, rmax = logits_range
        num_dirs = len(logits)
        b_width = 360 / num_dirs
        theta = np.linspace(0, 360, num=num_dirs, endpoint=False)

        logits = np.clip(logits, rmin, rmax)
        l_normal = (logits - rmin) / (rmax - rmin + 1e-8)

        diff_lower = np.clip(logits, rmin, rmid) - rmin
        low_bar = go.Barpolar(
            r=diff_lower,
            theta=theta,
            width=np.full(num_dirs, b_width),
            opacity=0.01,
            subplot="polar",
            marker=dict(
                color="rgb(255, 255, 255)",
            ),
        )

        m_colors = [type(self).color2str(c) for c in mpl.cm.get_cmap("plasma")(l_normal)]

        main_bar = go.Barpolar(
            r=np.abs(logits - rmid),
            theta=theta,
            width=np.full(num_dirs, b_width),
            opacity=1,
            subplot="polar",
            marker=dict(color=m_colors, line=dict(color=m_colors)),
        )

        zero_circle = go.Scatterpolar(
            r=np.full(zero_circle_points, rmid - rmin),
            theta=np.linspace(0, 370, num=zero_circle_points, endpoint=False),
            mode="lines",
            line=dict(color="deepskyblue", width=circle_width),
        )

        fig = go.FigureWidget([low_bar, main_bar, zero_circle])

        fig.update_layout(
            title=dict(
                text=f"<b>Policy's Prediction (Logits)<b>",
                font=dict(family="Comic Sans MS", color="black", size=52 * scale),
                y=0.9,
                x=0.5,
                xanchor="center",
                yanchor="top",
            ),
            #         template=None,
            showlegend=False,
            polar=dict(
                radialaxis=dict(
                    range=[0, rmax - rmin],
                    tickmode="array",
                    tickvals=[0, rmid - rmin, rmax - rmin],
                    ticktext=[rmin, rmid, rmax]
                    #             ticks="",
                    #             showticklabels=False,
                ),
                angularaxis=dict(ticks=""),  # , showticklabels=False)
            ),
            font=dict(size=36 * scale),
            margin=dict(t=(280 * scale)),
            width=1200 * scale,
            height=1200 * scale,
        )
        return fig

    def _plot_reward_history(self, rewards, gamma, window, logits_range, step_num, scale=1):
        d_cumsum = type(self).discount_cumsum(rewards, gamma)
        r_avg_all = np.cumsum(rewards) / (np.arange(0, len(rewards)) + 1e-8)  # first reward is artificial zero

        r = rewards[max(0, step_num - window + 1) : step_num + 1]
        r_avg = r_avg_all[max(0, step_num - window + 1) : step_num + 1]
        dsr = d_cumsum[max(0, step_num - window + 1) : step_num + 1]
        o_shift = max(window - len(r), 0)

        fig = go.FigureWidget()
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(r)) + o_shift,
                y=r_avg,
                mode="lines",
                line=dict(color="green", width=10 * scale),
                name="Running Average",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(dsr)) + o_shift,
                y=dsr,
                mode="lines",
                line=dict(color="orange", width=10 * scale),
                name="Discounted Sum of Futures",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(r)) + o_shift,
                y=r,
                mode="lines",
                line=dict(color="firebrick", width=10 * scale),
                name="Actual",
            )
        )

        rmin, rmax = logits_range

        all_ticks = np.arange(step_num - window + 1, step_num + 1)
        selected_idx = np.argwhere(all_ticks % 5 == 0).flatten()
        selected_vals = all_ticks[selected_idx]
        greater_zero = selected_vals >= 0

        selected_vals = selected_vals[greater_zero]
        selected_idx = selected_idx[greater_zero]

        fig.update_layout(
            xaxis=dict(
                title="<b>Timestep</b>",
                range=[0, window - 1],
                tickvals=selected_idx,
                ticktext=selected_vals,
            ),
            yaxis=dict(
                title="<b>Rewards</b>",
                range=[rmin, rmax],
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor="Black",
            ),
            title=dict(
                text=f"<b>Reward History</b><br><i>(gamma: {gamma:.2f})</i>",
                font=dict(color="black", size=33 * scale),
                y=0.9,
                x=0.5,
                xanchor="center",
                yanchor="top",
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            font=dict(family="Comic Sans MS", size=28 * scale),
            margin=dict(t=220 * scale),
            width=1200 * scale,
            height=600 * scale,
        )
        return fig

    @staticmethod
    def color2str(c):
        return f"rgba({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)}, {c[3]})"

    @staticmethod
    def discount_cumsum(x, discount):
        """
        magic from rllab for computing discounted cumulative sums of vectors.
        https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/algos/pytorch/ppo/core.py#L29

        input:
            vector x,
            [x0,
            x1,
            x2]
        output:
            [x0 + discount * x1 + discount^2 * x2,
            x1 + discount * x2,
            x2]
        """
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    @staticmethod
    def frame(im, thickness=5):
        # Get input image width and height, and calculate output width and height
        iw, ih = im.size
        ow, oh = iw + 2 * thickness, ih + 2 * thickness

        # Draw outer black rounded rect into memory as PNG
        outer = f'<svg width="{ow}" height="{oh}" style="background-color:none"><rect rx="20" ry="20" width="{ow}" height="{oh}" fill="black"/></svg>'
        png = svg2png(bytestring=outer)
        outer = Image.open(BytesIO(png))

        # Draw inner white rounded rect, offset by thickness into memory as PNG
        inner = f'<svg width="{ow}" height="{oh}"><rect x="{thickness}" y="{thickness}" rx="20" ry="20" width="{iw}" height="{ih}" fill="white"/></svg>'
        png = svg2png(bytestring=inner)
        inner = Image.open(BytesIO(png)).convert("L")

        # Expand original canvas with black to match output size
        expanded = ImageOps.expand(im, border=thickness, fill=(0, 0, 0)).convert("RGB")

        # Paste expanded image onto outer black border using inner white rectangle as mask
        outer.paste(expanded, None, inner)
        return outer


class TopViewRenderer(Renderer):
    def __init__(
        self,
        graph,
        scale,
        fps,
        size,
        save_dir,
        crf=23,
        skip_rate=0,
        min_steps_activate=0,
        max_frames=np.inf,
        base_width=3,
        edge_width=6,
        node_radius=3,
        movement_color=(255, 255, 0),
        retrace_color=(0, 255, 0),
        prefix: str = "topview",
        **kwargs,
    ):
        self.rgb = None
        self.scale = scale
        self.fps = fps
        self.skip_rate = skip_rate
        self.crf = crf
        self.min_steps_activate = min_steps_activate
        self.max_frames = max_frames
        self.save_dir = save_dir
        self.base_width = base_width
        self.edge_width = edge_width
        self.node_radius = node_radius
        self.movement_color = movement_color
        self.retrace_color = retrace_color
        self.prefix = prefix

        self.size = size
        self.graphDrawer = Drawer(self.size)
        self.movementDrawer = LazyDrawer(self.size)

        check_path(self.save_dir)

        for u, v, attrs in graph.edges(data=True):
            width = base_width
            width += edge_width * attrs["lanes"]
            width = max(round(width * scale), 1)
            self.graphDrawer.line(self._tr(u), self._tr(v), width, "#f48c06")
        for v in graph.nodes:
            radius = max(round(node_radius * scale), 1)
            self.graphDrawer.circle(self._tr(v), radius, "#6930c3")

        super().__init__(**kwargs)

    def render(self, env, **kwargs):
        raise NotImplementedError("TopViewRenderer doesn't implement 'render' method")

    def save(self, env: "RoadMasterEnv", **kwargs):
        filepath = os.path.join(self.save_dir, f"{self.prefix}_{env.savename}.mp4")

        total_frames = 1 + env.steps - env.last("UndoW_BP", fillNa=0)
        if total_frames < self.min_steps_activate:
            return

        if (env.episode_window_size is not None) or (self.rgb is None):
            self.rgb = self._downscale(env.rgb_bounded)

        # ENCODER
        process = (
            ffmpeg.input(
                "pipe:",
                framerate="{}".format(self.fps),
                format="rawvideo",
                pix_fmt="rgb24",
                s="{}x{}".format(*self.rgb.shape[1::-1]),
            )
            .output(
                filepath,
                pix_fmt="yuv420p",
                vcodec="libx264",
                crf=self.crf,
                movflags="+faststart",
            )
            .overwrite_output()
            .run_async(pipe_stdin=True, quiet=True)
        )
        # skip mask
        num_actual_frames = min(ceil(total_frames * (1 - self.skip_rate)), self.max_frames)
        actual_indices = np.round(np.arange(0, num_actual_frames) * (total_frames / num_actual_frames)).astype(int)

        replays = self.replay_episode(env)
        for idx in range(total_frames):
            frame = next(replays)
            if idx not in actual_indices:
                continue
            process.stdin.write(frame.tobytes())

        process.stdin.close()
        process.wait()

    def replay_episode(self, env):
        locations = env.logs["location-agent"][:]
        graph_locations = env.logs["location-graph"][:]
        actions_t = env.logs["action_type"][:]

        graph_drawer = self.graphDrawer
        location_drawer = self.movementDrawer
        location_drawer.reset()
        line_width = (self.edge_width * self.scale) * 4  # same as 4 lanes
        line_width = max(round(line_width), 1)
        alpha = 0.8

        x1_b, y1_b, x2_b, y2_b = np.round(np.array(env.bounds) * self.scale).astype(int)

        canvas = np.copy(self.rgb)
        graph = np.asarray(graph_drawer.canvas)[y1_b:y2_b, x1_b:x2_b]
        graph_mask = graph[..., -1] > 0
        canvas[graph_mask] = canvas[graph_mask] * (1 - alpha) + graph[graph_mask, :3] * alpha

        def render():
            # graph locations
            locations = np.asarray(location_drawer.canvas)[y1_b:y2_b, x1_b:x2_b]
            mask = locations[..., -1] > 0
            canvas[mask] = locations[mask, :3]
            return canvas

        # real start point
        real_start_point = len(locations) - len(env.logs["location-agent"]) + 1  # soft reset diff
        real_start_point += env.last("UndoW_BP", fillNa=0)

        # draw history
        for idx in range(1, real_start_point):
            color = self.retrace_color if actions_t[idx - 1] == "Retrace" else self.movement_color
            location_drawer.rounded_line(
                self._tr(locations[idx - 1]), self._tr(locations[idx]), width=line_width, color=color
            )

        # first obs
        yield render()

        for idx in range(real_start_point, len(locations)):
            color = self.retrace_color if actions_t[idx - 1] == "Retrace" else self.movement_color
            location_drawer.rounded_line(
                self._tr(locations[idx - 1]), self._tr(locations[idx]), width=line_width, color=color
            )
            yield render()

    def _tr(self, pts):
        return tuple(np.clip(np.array(pts) * self.scale, [0, 0], self.size))

    def _downscale(self, rgb):
        return rescale(rgb, self.scale, preserve_range=True, anti_aliasing=True, multichannel=True).astype(np.uint8)

    def __getattr__(self, name):
        # temporary fix for unwanted calls from the action scheme
        def noOp(*args, **kwargs):
            pass

        return noOp


class VideoRenderer(Renderer):
    def __init__(
        self,
        baseRenderer,
        mode: str,
        max_stack_size=200,
        fps=4,
        save_dir="logs/gifs/",
    ):
        super().__init__()
        self.baseRenderer = baseRenderer
        self.stack = deque(maxlen=max_stack_size)
        self.mode = mode
        self.fps = fps
        self.save_dir = save_dir
        check_path(save_dir)

    def __getattr__(self, name):
        if not hasattr(self.baseRenderer, name):
            raise AttributeError(f"Attribute '{name}' not found")
        return getattr(self.baseRenderer, name)

    def render(self, *args, **kwargs):
        obs = self.baseRenderer.render(*args, **kwargs)
        self.stack.append(obs.astype(np.uint8))

    def reset(self):
        self.baseRenderer.reset()
        self.stack.clear()

    def close(self):
        self.baseRenderer.close()
        self.stack.clear()

    def save(self, env: "RoadMasterEnv", mode=None, **kwargs):
        mode = mode or self.mode
        prefix = env.name + "_"

        if len(self.stack) == 0:
            return
        if self.mode == "gif":
            filepath = os.path.join(self.save_dir, env.savename + ".gif")
            type(self).save_gif_pillow(filepath, self.stack, self.fps)

        elif self.mode == "stack":
            return np.concatenate(self.stack)
        else:
            raise ValueError

    @staticmethod
    def save_gif_pillow(filepath, images, fps):
        first, *others = [Image.fromarray(img) for img in images]
        first.save(
            fp=filepath,
            format="GIF",
            append_images=others,
            save_all=True,
            duration=int(1000 / fps),
            loop=0,
        )

    @staticmethod
    def save_gif_imageio(filepath, images, fps):
        mimsave(filepath, images, fps=fps)


class VideoRendererContinuousSpace(VideoRenderer):
    def __init__(
        self,
        frame_size,
        gamma,
        num_directions=32,
        heatmap_window=10,
        reward_window=30,
        rewards_mid_point=None,
        skip_rate=0,
        min_steps_activate=2,
        max_frames=np.inf,
        **kwargs,
    ):
        super().__init__(mode=None, **kwargs)
        self.frame_size = frame_size
        self.gamma = gamma
        self.num_directions = num_directions
        self.heatmap_window = heatmap_window
        self.reward_window = reward_window
        self.rewards_mid_point = rewards_mid_point
        self.skip_rate = skip_rate
        assert min_steps_activate >= 2
        self.min_steps_activate = min_steps_activate
        self.max_frames = max_frames
        self.plot_ranges = np.array(
            [
                [[0.025, 0.6], [0.025, 0.6]],  # obs
                [[0.6, 0.975], [0.025, 0.4]],  # direction heatmap
                [[0.6, 0.975], [0.4, 0.775]],  # logits
                [[0.025, 0.65], [0.65, 0.975]],  # rewards
            ]
        )

    def render(self, env, **kwargs):
        raise NotImplementedError("VideoRenderer9000 doesn't implement 'render' method")

    def save(self, env: "RoadMasterEnv", mode=None, **kwargs):
        mode = mode or self.mode
        filepath = os.path.join(self.save_dir, env.savename + ".mp4")

        start_step = env.last("UndoW_BP", fillNa=0)
        end_step = env.steps
        total_frames = end_step - start_step + 1
        if total_frames < self.min_steps_activate:
            return

        # DATA
        assert isinstance(env.action_scheme, ContinuousSpace)
        reward_space = env.reward_scheme.reward_space
        reward_range = (reward_space.low[0], reward_space.high[0])

        value_range = (1 / (1 - self.gamma)) * np.array(reward_range)
        value_range[0] = value_range[0] * 1 / 2  # TODO: no hacks
        value_range = symceil(value_range, decimals=0).astype(float)

        if env.reward_scheme.exponent <= 1:
            rewards_plot_domain = value_range
            # rewards_plot_domain = logits_range + [-1, 1]
        else:
            # rewards_plot_domain = symceil(logits_range / 2, decimals=1)
            rewards_plot_domain = [reward_range[0] - 0.1, value_range[1] + 0.1]

        logits = np.concatenate(env.logs["logits_cont"][:])
        mu, sigma = logits[:, 0, None], logits[:, 1, None]

        angles = np.linspace(-1, 1, num=self.num_directions, endpoint=False)
        angles = np.concatenate((angles[self.num_directions // 2 :], angles[: self.num_directions // 2]))
        angles = np.clip(angles, -0.98, 0.98)
        points = np.arctanh(angles).reshape(1, -1)

        probs = np.exp(-(((points - mu) / sigma) ** 2) / 2) / (sigma * (2 * np.pi) ** 0.5)
        probs = np.concatenate((probs, probs[-1:]))

        # soft normalization
        probs = np.clip(
            probs, np.percentile(probs, 1, axis=1, keepdims=True), np.percentile(probs, 99, axis=1, keepdims=True)
        )
        probs = probs / (probs.max(axis=1, keepdims=True) + 1e-8)

        # probs = probs / (probs.sum(axis=1, keepdims=True) + 1e-8)

        # amin, amax = probs.min(axis=1, keepdims=True), probs.max(axis=1, keepdims=True)
        # probs = (probs - amin) / (amax - amin)

        rewards = np.array(env.logs["reward_total"][:])
        rewards = np.concatenate(([0], rewards))

        entropy = np.array(env.logs["entropy"][:])
        entropy = np.concatenate(([0], entropy))

        target_q = np.array(env.logs["value"][:])
        target_q = np.concatenate(([0], target_q))

        # ENCODER
        process = (
            ffmpeg.input(
                "pipe:",
                framerate="{}".format(self.fps),
                format="rawvideo",
                pix_fmt="rgb24",
                s="{}x{}".format(self.frame_size[0], self.frame_size[1]),
            )
            .output(
                filepath,
                pix_fmt="yuv420p",
                vcodec="libx264",
                crf=23,
                movflags="+faststart",
            )
            .overwrite_output()
            .run_async(pipe_stdin=True, quiet=True)
        )

        num_actual_frames = min(ceil(total_frames * (1 - self.skip_rate)), self.max_frames)
        selected_indices = np.round(np.arange(0, num_actual_frames) * (total_frames / num_actual_frames)).astype(int)

        replays = self.replay_episode(env)
        for idx in range(total_frames):
            obs = next(replays)
            if idx not in selected_indices:
                continue

            im = self._create_viz(
                step=idx + start_step,
                obs=obs,
                logits=probs,
                entropy=entropy,
                target_q=target_q,
                rewards=rewards,
                logits_range=value_range,
                rewards_plot_domain=rewards_plot_domain,
            )
            process.stdin.write(im.tobytes())

        process.stdin.close()
        process.wait()

    def replay_episode(self, env):
        locations = env.history["all"]["location-agent"][:]
        graph_locations = env.history["all"]["location-graph"][:]
        actions_t = env.history["all"]["action_type"][:]

        # movement_drawer = LazyIntensityDrawer(env.size)
        graph_drawer = self.graph_drawer
        location_drawer = self.location_drawer
        location_drawer.reset()
        line_width = self.base_width + self.edge_width

        def render(location, alpha=0.8):
            bbox_r, bbox_o = env.get_observation_window_coords(location)
            x1_r, y1_r, x2_r, y2_r = bbox_r
            x1_o, y1_o, x2_o, y2_o = bbox_o

            o_width, o_height = env.observation_shape
            obs = np.zeros((o_height, o_width, 3), dtype=np.uint8)
            obs[y1_o:y2_o, x1_o:x2_o] = env.rgb[y1_r:y2_r, x1_r:x2_r]
            graph = np.asarray(graph_drawer.window(bbox_r))
            graph_mask = graph[..., -1] > 0
            obs[y1_o:y2_o, x1_o:x2_o][graph_mask] = (
                obs[y1_o:y2_o, x1_o:x2_o][graph_mask] * (1 - alpha) + graph[graph_mask, :3] * alpha
            )

            # movement = np.zeros((o_height, o_width))
            # movement_raw = movement_drawer.window((x1_r, y1_r, x2_r, y2_r), mode="intensity-gray")
            # movement[y1_o:y2_o, x1_o:x2_o] = movement_raw
            # obs[movement > 0, :] = 255

            # graph locations
            locations = np.asarray(location_drawer.window(bbox_r))
            mask = locations[..., -1] > 0
            obs[y1_o:y2_o, x1_o:x2_o][mask] = locations[mask, :3]
            return obs

        # real start point
        real_start_point = len(locations) - len(env.logs["location-agent"]) + 1  # soft reset diff
        real_start_point += env.last("UndoW_BP", fillNa=0)

        # draw history
        location_drawer.circle(graph_locations[0], radius=2.5, color=(255, 0, 0))
        for idx in range(1, real_start_point):
            # movement_drawer.rounded_line(locations[idx - 1], locations[idx], width=line_width, color=(255, 255, 255))
            location_drawer.circle(graph_locations[idx], radius=2.5, color=(255, 0, 0))
            color = (255, 255, 255) if actions_t[idx - 1] == "Normal" else (0, 255, 0)
            location_drawer.rounded_line(locations[idx - 1], locations[idx], width=line_width + 1, color=color)

        # first obs
        yield render(locations[real_start_point - 1])

        for idx in range(real_start_point, len(locations)):
            # movement_drawer.rounded_line(locations[idx - 1], locations[idx], width=line_width, color=(255, 255, 255))
            location_drawer.circle(graph_locations[idx], radius=2.5, color=(255, 0, 0))
            color = (255, 255, 255) if actions_t[idx - 1] == "Normal" else (0, 255, 0)
            location_drawer.rounded_line(locations[idx - 1], locations[idx], width=line_width + 1, color=color)
            yield render(locations[idx])

    def _create_viz(self, step, obs, logits, entropy, target_q, rewards, logits_range, rewards_plot_domain):
        obs_image = type(self).frame(Image.fromarray(obs))
        heatmap_img = Image.open(
            BytesIO(
                self._plot_direction_heatmap(
                    predictions=logits[max(0, step - self.heatmap_window + 1) : step + 1],
                    window=self.heatmap_window,
                ).to_image(format="png", engine="kaleido")
            )
        )
        logits_img = Image.open(BytesIO(self._plot_probs(logits=logits[step]).to_image(format="png", engine="kaleido")))
        rew_img = Image.open(
            BytesIO(
                self._plot_reward_history(
                    rewards,
                    target_q,
                    entropy,
                    gamma=self.gamma,
                    window=self.reward_window,
                    logits_range=rewards_plot_domain,
                    step_num=step,
                ).to_image(format="png", engine="kaleido")
            )
        )
        width, height = self.frame_size
        canvas = Image.new("RGBA", (width, height), "WHITE")

        for image, coords in zip([obs_image, heatmap_img, logits_img, rew_img], self.plot_ranges):
            lx, ly = ((coords[:, 1] - coords[:, 0]) * [width, height]).astype(int)
            ox, oy = (coords[:, 0] * [width, height]).astype(int)
            im = image.resize((lx, ly), resample=Image.BILINEAR)
            canvas.paste(im, (ox, oy), im)

        viz = np.asarray(canvas.convert("RGB"))
        return viz

    def _plot_direction_heatmap(self, predictions, window, scale=1):
        num_dirs = predictions.shape[1]
        b_width = 360 / num_dirs
        theta = np.linspace(0, 360, num=num_dirs, endpoint=False)

        fig = go.FigureWidget()

        assert len(predictions) <= window
        values = np.zeros((window, num_dirs))
        values[-len(predictions) :] = predictions
        # values = values[::-1] # from center to outer

        for idx in range(window):
            logits = values[idx]
            # l_normal = (logits - logits.min()) / (logits.max() - logits.min() + 1e-8)
            l_normal = logits
            colors = [type(self).color2str(c) for c in mpl.cm.get_cmap("plasma")(l_normal)]

            bar = go.Barpolar(
                r=np.full(num_dirs, 1),
                theta=theta,
                width=np.full(num_dirs, b_width),
                marker=dict(
                    color=colors,
                    line=dict(color=colors, width=2),
                ),
                opacity=0.95,
                subplot="polar",
            )
            fig.add_trace(bar)

        fig.update_layout(
            title=dict(
                text="<b>Direction Heatmap</b><br>" f"<i>(window: {window})</i>",
                font=dict(family="Comic Sans MS", color="black", size=52 * scale),
            ),
            template=None,
            showlegend=False,
            polar=dict(
                radialaxis=dict(range=[0, window], showticklabels=False, ticks=""),
                angularaxis=dict(ticks=""),  # , showticklabels=False)
            ),
            font=dict(size=35 * scale),
            margin=dict(t=290 * scale),
            width=1200 * scale,
            height=1200 * scale,
        )
        return fig

    def _plot_probs(self, logits, scale=1):
        rmin, rmax = logits.min(), logits.max()
        num_dirs = len(logits)
        b_width = 360 / num_dirs
        theta = np.linspace(0, 360, num=num_dirs, endpoint=False)
        # l_normal = (logits - rmin) / (rmax - rmin + 1e-8)
        l_normal = logits

        m_colors = [type(self).color2str(c) for c in mpl.cm.get_cmap("plasma")(l_normal)]
        main_bar = go.Barpolar(
            r=logits,
            theta=theta,
            width=np.full(num_dirs, b_width),
            opacity=1,
            subplot="polar",
            marker=dict(color=m_colors, line=dict(color=m_colors)),
        )

        fig = go.FigureWidget([main_bar])

        fig.update_layout(
            title=dict(
                text=f"<b>Policy's Prediction (Logits)<b>",
                font=dict(family="Comic Sans MS", color="black", size=52 * scale),
                y=0.9,
                x=0.5,
                xanchor="center",
                yanchor="top",
            ),
            #         template=None,
            showlegend=False,
            polar=dict(
                radialaxis=dict(
                    # range=[0, rmax - rmin],
                    tickmode="array",
                    tickvals=[0, 1],
                    ticktext=[0, 1]
                    #             ticks="",
                    #             showticklabels=False,
                ),
                angularaxis=dict(ticks=""),  # , showticklabels=False)
            ),
            font=dict(size=36 * scale),
            margin=dict(t=(280 * scale)),
            width=1200 * scale,
            height=1200 * scale,
        )
        return fig

    def _plot_reward_history(self, rewards, target_q, entropy, gamma, window, logits_range, step_num, scale=1):
        d_cumsum = type(self).discount_cumsum(rewards, gamma)
        r_avg_all = np.cumsum(rewards) / (np.arange(0, len(rewards)) + 1e-8)  # first reward is artificial zero

        st, tt = max(0, step_num - window + 1), step_num + 1

        r = rewards[st:tt]
        r_avg = r_avg_all[st:tt]
        dsr = d_cumsum[st:tt]
        e = entropy[st:tt]
        q = target_q[st:tt]
        ret_ent = dsr + e

        o_shift = max(window - len(r), 0)

        fig = go.FigureWidget()
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(r)) + o_shift,
                y=r_avg,
                mode="lines",
                line=dict(color="green", width=10 * scale),
                name="Running Avg",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(dsr)) + o_shift,
                y=dsr,
                mode="lines",
                line=dict(color="orange", width=10 * scale),
                name="Returns",
            )
        )
        if len(e) > 1:
            print(e.shape, e)
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(ret_ent)) + o_shift,
                    y=ret_ent,
                    mode="lines",
                    line=dict(color="purple", width=10 * scale),
                    name="Returns+Entropy",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(q)) + o_shift,
                y=q,
                mode="lines",
                line=dict(color="black", width=10 * scale),
                name="Target Q",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(r)) + o_shift,
                y=r,
                mode="lines",
                line=dict(color="firebrick", width=10 * scale),
                name="Reward",
            )
        )

        rmin, rmax = logits_range

        all_ticks = np.arange(step_num - window + 1, step_num + 1)
        selected_idx = np.argwhere(all_ticks % 5 == 0).flatten()
        selected_vals = all_ticks[selected_idx]
        greater_zero = selected_vals >= 0

        selected_vals = selected_vals[greater_zero]
        selected_idx = selected_idx[greater_zero]

        fig.update_layout(
            xaxis=dict(
                title="<b>Timestep</b>",
                range=[0, window - 1],
                tickvals=selected_idx,
                ticktext=selected_vals,
            ),
            yaxis=dict(
                title="<b>Rewards</b>",
                range=[rmin, rmax],
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor="Black",
            ),
            title=dict(
                text=f"<b>Reward History</b><br><i>(gamma: {gamma:.2f})</i>",
                font=dict(color="black", size=33 * scale),
                y=0.9,
                x=0.5,
                xanchor="center",
                yanchor="top",
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            font=dict(family="Comic Sans MS", size=28 * scale),
            margin=dict(t=220 * scale),
            width=1200 * scale,
            height=600 * scale,
        )
        return fig

    @staticmethod
    def color2str(c):
        return f"rgba({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)}, {c[3]})"

    @staticmethod
    def discount_cumsum(x, discount):
        """
        magic from rllab for computing discounted cumulative sums of vectors.
        https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/algos/pytorch/ppo/core.py#L29

        input:
            vector x,
            [x0,
            x1,
            x2]
        output:
            [x0 + discount * x1 + discount^2 * x2,
            x1 + discount * x2,
            x2]
        """
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    @staticmethod
    def frame(im, thickness=5):
        # Get input image width and height, and calculate output width and height
        iw, ih = im.size
        ow, oh = iw + 2 * thickness, ih + 2 * thickness

        # Draw outer black rounded rect into memory as PNG
        outer = f'<svg width="{ow}" height="{oh}" style="background-color:none"><rect rx="20" ry="20" width="{ow}" height="{oh}" fill="black"/></svg>'
        png = svg2png(bytestring=outer)
        outer = Image.open(BytesIO(png))

        # Draw inner white rounded rect, offset by thickness into memory as PNG
        inner = f'<svg width="{ow}" height="{oh}"><rect x="{thickness}" y="{thickness}" rx="20" ry="20" width="{iw}" height="{ih}" fill="white"/></svg>'
        png = svg2png(bytestring=inner)
        inner = Image.open(BytesIO(png)).convert("L")

        # Expand original canvas with black to match output size
        expanded = ImageOps.expand(im, border=thickness, fill=(0, 0, 0)).convert("RGB")

        # Paste expanded image onto outer black border using inner white rectangle as mask
        outer.paste(expanded, None, inner)
        return outer
