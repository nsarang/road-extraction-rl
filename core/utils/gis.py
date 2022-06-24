import numpy as np
from pyproj import Transformer


def LatLonToMeters(p):
    lat, lon = p
    x, y = Transformer.from_crs("EPSG:4326", "EPSG:3857").transform(lat, lon)
    return np.array([x, y])


def MetersToLatLon(p):
    x, y = p
    lat, lon = Transformer.from_crs("EPSG:3857", "EPSG:4326").transform(x, y)
    return np.array([lat, lon])


def GetMetersPerPixel(zoom):
    return (2 * np.pi * 6378137) / (2 ** zoom) / 256


def LatLonToPixel(p, origin, zoom):
    p = LatLonToMeters(p) - LatLonToMeters(origin)
    p /= GetMetersPerPixel(zoom)  # get pixel coordinates
    p[1] *= -1  # invert y axis to correspond to sat image orientation
    return p


def PixelToLatLon(p, origin, zoom):
    p[1] *= -1
    p *= GetMetersPerPixel(zoom)
    p = MetersToLatLon(p + LatLonToMeters(origin))
    return p
