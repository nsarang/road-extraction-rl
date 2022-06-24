from PIL import Image
import networkx as nx
import os
import dill
import numpy as np
import json
from functools import lru_cache
from typing import MutableSequence

from .osm import load_osm, convert_gdf_to_graph
from .utils import graph_crop
from core.utils.geometry import dist
from configs.defaults import CACHE_DIR


@lru_cache
def load_data(
    city,
    data_dir,
    imagery_dir=None,
    osm_dir=None,
    edge_max_len=150,
    connected_network=False,
    bbox_padding=0.05,
    force_reload=False,
    **kwargs
):
    Image.MAX_IMAGE_PIXELS = None  # bypass DecompressionBombError

    # load city bounds
    with open(os.path.join(data_dir, "city_bounds.txt")) as f:
        dataset_bounds = json.load(f)

    # load image 
    if imagery_dir is None:
        imagery_dir = os.path.join(data_dir, "imagery")
    if not isinstance(imagery_dir, (list, MutableSequence)):
        imagery_dir = [imagery_dir]
    images = [Image.open(os.path.join(im_dir, city + ".png")) for im_dir in imagery_dir]
    rgb = images[0]

    # load/construct graph
    graphPath = os.path.join(CACHE_DIR, city + "_.pkl")
    if (not force_reload) and os.path.isfile(graphPath):
        with open(graphPath, "rb") as handle:
            graph = dill.load(handle)
    else:
        if osm_dir is None:
            osm_dir = os.path.join(data_dir, "osm")
        gdf, city_bounds = load_osm(
            city, osm_dir, dataset_bounds[city], padding=bbox_padding, force_reload=force_reload, **kwargs
        )
        graph = convert_gdf_to_graph(gdf, city_bounds)
        # save
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(graphPath, "wb") as handle:
                dill.dump(graph, handle, protocol=4)
        except:
            pass

    # remove graph padded area
    graph = graph_crop(graph, (0, 0, *rgb.size))

    # split long edges for faster search
    def split(u, v, max_len):
        length = dist(u, v)
        ratio = np.ceil(length / max_len).astype(int)
        if ratio == 1:
            return
        points = np.stack([np.linspace(c1, c2, num=ratio + 1) for c1, c2 in zip(u, v)], axis=1)
        attrs = graph[u][v]
        graph.remove_edge(u, v)
        for p1, p2 in zip(points[:-1], points[1:]):
            graph.add_edge(tuple(p1), tuple(p2), **attrs)

    if edge_max_len:
        edges = list(graph.edges)
        for e in edges:
            split(*e, max_len=edge_max_len)

    # filter small components
    if connected_network:
        largest_cc = max(nx.connected_components(graph), key=len)
        graph = graph.subgraph(largest_cc).copy()

    return images, graph
