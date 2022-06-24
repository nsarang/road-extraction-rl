import numpy as np
from math import sqrt, cos, sin
from shapely.geometry import Point, LineString


def is_in_bound(point, bbox, exclusive_end=True):
    px, py = point
    x1, y1, x2, y2 = bbox
    if exclusive_end:
        x2 -= 1
        y2 -= 1
    return (px >= x1) and (py >= y1) and (px <= x2) and (py <= y2)


def dist(u, v):
    return sqrt(sum([(x - y) ** 2 for x, y in zip(u, v)]))


def path_length(path):
    return sum(dist(*edge) for edge in zip(path[:-1], path[1:]))


def polar2cartesian(r, theta):
    x = r * cos(theta)
    y = r * sin(theta)
    return (x, y)


def dist2line(point, line_endpoints):
    projection = Point(project2line(point, line_endpoints))
    point = Point(point)
    return point.distance(projection)


def project2line(point, line_endpoints):
    point = Point(point)
    line = LineString(line_endpoints)
    dist = line.project(point)
    projection = line.interpolate(dist)
    xy = list(projection.coords)[0]
    return xy


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / (np.linalg.norm(vector) + 1e-8)


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'"""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
