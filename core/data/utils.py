from PIL import Image
import networkx as nx
from copy import deepcopy
from shapely.geometry import Polygon, LineString


def graph_crop(graph: nx.Graph, coords: tuple):
    n_graph = nx.Graph()
    x1, y1, x2, y2 = coords
    x2, y2 = (x2 - 1), (y2 - 1) # exclusive end
    bbox = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

    for (u, v, attrs) in graph.edges(data=True):
        l = LineString([u, v]).intersection(bbox)
        if len(l.coords) == 0:
            continue
        u, v = l.coords
        attrs = deepcopy(attrs)
        n_graph.add_edge(u, v, **attrs)

    return n_graph
