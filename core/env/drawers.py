import numpy as np
from PIL import Image, ImageDraw
from core.utils.search import SearchTree
from core.utils.clock import TimedIdentifiable
from core.utils.misc import StackU


class Drawer:
    def __init__(self, size, mode="RGBA"):
        self.size = size
        self.mode = mode
        self.layer = Image.new(self.mode, tuple(self.size))
        self.draw = ImageDraw.Draw(self.layer)

    @property
    def canvas(self):
        return self.layer

    def window(self, bbox, mode="pillow"):
        image = self.canvas.crop(bbox)
        if mode == "pillow":
            return image
        elif mode == "numpy":
            return np.array(image)
        else:
            raise NotImplementedError

    def line(self, u, v, width, color):
        self.draw.line((u, v), width=width, fill=color)
        return self

    def rounded_line(self, u, v, width, color):
        self.line(u, v, width, color)
        self.circle(u, radius=width / 2 - 1, color=color)
        self.circle(v, radius=width / 2 - 1, color=color)
        return self

    def circle(self, center, radius, color):
        self.draw.ellipse(
            (
                center[0] - radius,
                center[1] - radius,
                center[0] + radius,
                center[1] + radius,
            ),
            fill=color,
            outline=None,
        )
        return self

    def reset(self):
        if self.mode in ["RGB", "RGBA"]:
            color = (0, 0, 0, 0)
        else:
            color = 0
        self.canvas.paste(color, (0, 0, *self.size))


class LazyDrawer:
    def __init__(self, size=None, mode="RGBA"):
        self.size = size
        self.mode = mode
        self.rtree = SearchTree()
        self.ops = StackU()

    @property
    def components(self) -> "Dict[str, Any]":
        return {
            "search_tree": self.rtree,
            "ops": self.ops,
        }

    @property
    def canvas(self):
        return self.window((0, 0))

    def window(self, bbox, mode="pillow"):
        if len(bbox) == 2:
            # upper left corner given; get size from image
            bbox += self.size
        image = self._draw(bbox)
        if mode == "pillow":
            return image
        elif mode == "numpy":
            return np.array(image)
        else:
            raise NotImplementedError

    def line(self, u, v, width, color):
        self.ops.append(("line", [u, v, width, color]))
        self.rtree.insert(id=len(self.ops) - 1, coordinates=self._bbox_from_points([u, v]))
        return self

    def circle(self, center, radius, color):
        self.ops.append(("circle", [center, radius, color]))
        self.rtree.insert(
            id=len(self.ops) - 1,
            coordinates=(
                center[0] - radius,
                center[1] - radius,
                center[0] + radius,
                center[1] + radius,
            ),
        )
        return self

    def rounded_line(self, u, v, width, color):
        self.line(u, v, width, color)
        self.circle(u, radius=width / 2 - 1, color=color)
        self.circle(v, radius=width / 2 - 1, color=color)
        return self

    def _draw(self, bbox):
        left, top, right, bottom = bbox
        width, height = (right - left), (bottom - top)

        canvas = Image.new(self.mode, (width, height))
        draw = ImageDraw.Draw(canvas)

        ops_indices = self.rtree.intersection(bbox)
        ops_indices = sorted(ops_indices)
        for idx in ops_indices:
            (fname, args) = self.ops[idx]
            if fname == "line":
                u, v, width, color = args
                draw.line(
                    (u[0] - left, u[1] - top, v[0] - left, v[1] - top),
                    width=width,
                    fill=color,
                )
            elif fname == "circle":
                center, radius, color = args
                draw.ellipse(
                    (
                        center[0] - radius - left,
                        center[1] - radius - top,
                        center[0] + radius - left,
                        center[1] + radius - top,
                    ),
                    fill=color,
                    outline=None,
                )
        return canvas

    def _bbox_from_points(self, points):
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        return (x_min, y_min, x_max, y_max)

    def undo(self, *args, **kwargs):
        for c in self.components.values():
            c.undo(*args, **kwargs)

    def reset(self):
        for c in self.components.values():
            c.reset()


class IntensityDrawingW(TimedIdentifiable):
    def __init__(self, *args, target_range=(0.2, 1), **kwargs):
        kwargs["mode"] = "F"
        super().__init__(*args, **kwargs)
        self._target_range = target_range

    @property
    def step(self):
        return self.clock.step

    def window(self, bbox, mode="intensity"):
        image = super().window(bbox, mode="numpy")
        if mode == "numpy":
            return image
        elif mode == "binary":
            image[image > 0] = 255
            return image.astype(np.uint8)
        elif mode in ["intensity", "intensity-gray", "intensity-rgb"]:
            footprint = image > 0
            if np.any(footprint):
                vmin = np.min(image[footprint])
                vmax = np.max(image[footprint])
                image[footprint] = np.interp(image[footprint], (vmin, vmax), self._target_range)
            if mode == "intensity-gray":
                image = (image * 255).astype(int)
            if mode == "intensity-rgb":
                image = np.repeat(np.array(image)[..., np.newaxis], repeats=3, axis=-1)
                image = (image * 255).astype(int)
            return image
        else:
            raise NotImplemented

    def line(self, u, v, width, color=None):
        color = self.step
        return super().line(u, v, width, color)

    def rounded_line(self, u, v, width, color=None):
        color = self.step
        return super().rounded_line(u, v, width, color)

    def circle(self, center, radius, color=None):
        color = self.step
        return super().circle(center, radius, color)


# MIXINS
IntensityDrawer = type("IntensityDrawer", (IntensityDrawingW, Drawer), {})
LazyIntensityDrawer = type("LazyIntensityDrawer", (IntensityDrawingW, LazyDrawer), {})
