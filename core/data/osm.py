import os
import numpy as np
import networkx as nx
import osmnx as ox
import json
from numbers import Number
from core.utils.gis import LatLonToMeters, GetMetersPerPixel, LatLonToPixel


roadtracer_filters = (
    '["highway"]["area"!~"yes"]["highway"!~"cycleway|footway|path|pedestrian|emergency_access_point'
    '|steps|elevator|construction|bridleway|sidewalk|proposed|bus_stop|crossing|escape|give_way"]'
    '["service"!~"parking|parking_aisle"]'
)


roadtracer_filters_v2 = (
    '["highway"]["area"!~"yes"]["tunnel"!~"yes"]'
    '["highway"!~"pedestrian|footway|bridleway|steps|path|sidewalk|cycleway|proposed|construction|bus_stop|crossing|elevator|emergency_access_point|escape|give_way"]'
    '["amenity"!~"parking"]'
    '["service"!~"parking_aisle|driveway|drive-through|slipway"]'
)


def fetch_raw_osm_from_bbox(bbox, filters):
    graph = ox.graph_from_bbox(*bbox, custom_filter=filters)
    return graph


def convert_gdf_to_graph(gdf, bounds):
    gdf.name = gdf.name.fillna("UNK").astype(str)

    (S, W), (N, E) = bounds
    top = (N, W)
    origin = LatLonToMeters(top)
    zoom = 18
    meters_pixel = 1 / GetMetersPerPixel(zoom)
    gdf = gdf.to_crs("EPSG:3857")

    gdf.geometry = gdf.geometry.translate(xoff=-origin[0], yoff=-origin[1])
    gdf.geometry = gdf.geometry.scale(xfact=meters_pixel, yfact=-meters_pixel, origin=(0, 0))

    # preprocess lanes field
    def num_lanes(x):
        if isinstance(x, str) and (";" in x):
            x = x.split(";")
        if isinstance(x, list):
            lanes = [num_lanes(i) for i in x]
            return int(np.ceil(np.mean(lanes)))
        try:
            return int(x)
        except:
            return 1

    gdf.lanes = gdf.lanes.apply(num_lanes)

    def linestring_to_edges(s):
        coords = list(s.geometry.coords)
        edges = [(u, v) for u, v in zip(coords[:-1], coords[1:])]
        attrs = s.to_dict()
        del attrs["geometry"]
        return edges, attrs

    edges = gdf.apply(linestring_to_edges, axis=1)
    g = nx.Graph()
    width, height = LatLonToPixel((S, E), top, zoom)

    for data in edges:
        edge_coords, attrs = data
        g.add_edges_from(edge_coords, **attrs)

    return g


def load_osm(
    city, osm_dir, city_bounds, padding=0, overpass_filters=roadtracer_filters_v2, pickle_protocol=4, force_reload=False
):
    graph_fp = os.path.join(osm_dir, f"{city}_pad{padding:.2f}.pkl")

    if os.path.isfile(graph_fp) and force_reload == False:
        raw_graph = nx.read_gpickle(graph_fp)
    else:
        (S, W), (N, E) = city_bounds
        raw_graph = fetch_raw_osm_from_bbox((N + padding, S - padding, E + padding, W - padding), overpass_filters)
        os.makedirs(osm_dir, exist_ok=True)
        nx.write_gpickle(raw_graph, graph_fp, protocol=pickle_protocol)

    gdf = ox.graph_to_gdfs(raw_graph, nodes=False)
    return gdf, city_bounds
