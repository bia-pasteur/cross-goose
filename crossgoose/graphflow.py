

import json
import logging
from typing import Dict

import networkx as nx
import numpy as np
from scipy.spatial import KDTree
from skimage.morphology import skeletonize
from tqdm import tqdm
import edt
from crossgoose.graph_utils import get_networkx_graph_from_array


def get_nth_predecessor(graph: nx.DiGraph, vert, n: int) -> int:
    for _ in range(n):
        vert = next(iter(graph.succ[vert]))
    return vert


def store_pos_as_attribute(graph: nx.Graph, distance_map: np.ndarray | None = None) -> nx.Graph:
    """stores node position as a list in 'pos' attribute, 
    if distance_map is supplied, the thickness/radius is stored as 'rad',
    nodes are expected to be tuples (i,j), then relabeled as just ints

    Args:
        graph (nx.Graph): graph
        distance_map (np.ndarray | None, optional): distance map. Defaults to None.

    Returns:
        nx.Graph: _description_
    """
    for n in graph.nodes():
        i, j = n[0].item(), n[1].item()
        graph.nodes[n]['pos'] = [i, j]
        if distance_map is not None:
            graph.nodes[n]['rad'] = distance_map[i, j].item()
    return nx.relabel_nodes(graph, {n: i for i, n in enumerate(graph.nodes())})


def one_hot_labels_to_graphs(labels_one_hot: np.ndarray, smoothing: int = 16):
    n_instances = len(labels_one_hot)
    graphs = {}
    for k in tqdm(range(n_instances)):
        mask = labels_one_hot[k]
        skel = skeletonize(mask)
        dist = edt.edt(mask)
        graph = get_networkx_graph_from_array(skel)
        # graph = convert_graph_to_native_int(graph)
        graph = store_pos_as_attribute(graph, distance_map=dist)
        # store_predecessor_and_distance(graph)
        graph = to_digraph_with_distance(graph)
        graph = relax_attribute(graph, 'pos', niter=smoothing)
        compute_tangents(graph)

        graphs[k+1] = graph
    return graphs


def to_digraph_with_distance(graph: nx.Graph) -> nx.DiGraph:
    center = nx.center(graph)
    if len(center) > 1:
        logging.warning("found more than one center !")
    center = center[0]

    # create digraph with no edges
    digraph = graph.to_directed()
    digraph.remove_edges_from(list(digraph.edges()))

    digraph.nodes[center]['dist'] = 0
    # center loops on itself
    digraph.add_edge(center, center)

    stack = [center]
    while len(stack) > 0:
        vert = stack.pop(-1)
        d = digraph.nodes[vert]['dist']
        # get neighbors from source graph
        neighbors = graph.adj[vert]
        for n in neighbors:
            if digraph.nodes[n].get('dist', np.inf) > (d+1):

                digraph.add_edge(n, vert)
                if digraph.has_edge(vert, n):
                    digraph.remove_edge(vert, n)

                digraph.nodes[n]['dist'] = d + 1
                stack.append(n)
    return digraph


def relax_attribute(graph: nx.DiGraph, attr: str, niter: int = 1):
    if niter == 0:
        return graph
    new_pos = {}
    for n in graph.nodes():
        neighbors = set(graph.succ[n])
        neighbors.update(graph.pred[n]) 
        neighbors.add(n)
        attr_vals = np.array(
            [graph.nodes[nn][attr] for nn in neighbors])
        avg_pos = np.mean(attr_vals, axis=0)
        new_pos[n] = avg_pos.tolist()
    nx.set_node_attributes(graph, new_pos, name=attr)
    if niter == 1:
        return graph
    else:
        return relax_attribute(graph, attr, niter-1)


def read_graphs_from_yaml(file: str):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    graphs = {}
    for k, v in data.items():
        graphs[int(k)] = nx.adjacency_graph(v)
    return graphs


def store_graphs_to_yaml(graphs: Dict[int, nx.Graph], file: str):
    # graph_repr = {k:dict(g.adjacency()) for k,g in graphs.items()}
    graph_repr = {k: nx.adjacency_data(g) for k, g in graphs.items()}
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(graph_repr, f)


def normalize_vec(vec: np.ndarray, axis: int | None = None) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=axis, keepdims=True)
    if axis is None:
        if norm > 0.0:
            vec = vec / norm
    else:
        vec = np.where(norm > 0.0, vec/norm, vec)
    return vec


def compute_tangents(graph: nx.DiGraph):
    for n in graph.nodes():
        suc = next(iter(graph.succ[n]))
        if suc == n:
            # this is the center
            vec = np.zeros(2)
        else:
            vec = np.array(graph.nodes[suc]['pos']) - \
                np.array(graph.nodes[n]['pos'])

        graph.nodes[n]['tan'] = normalize_vec(vec).tolist()


class AnalyticalFlow:
    def __init__(
        self,
        graph_dict: Dict[str, nx.DiGraph],
        n_interpol: int
    ):
        self.graph_dict = graph_dict
        self.n_interpol = n_interpol

        self.kdtrees = {
            k: KDTree(np.array([g.nodes[n]['pos'] for n in g.nodes()])) for k, g in self.graph_dict.items()
        }

        self.positions = {k: np.array([
            g.nodes[n]['pos'] for n in g.nodes()
        ]) for k, g in self.graph_dict.items()}
        self.radiuses = {k: np.array([
            g.nodes[n]['rad'] for n in g.nodes()
        ]) for k, g in self.graph_dict.items()}
        self.tangents = {k: np.array([
            g.nodes[n]['tan'] for n in g.nodes()
        ]) for k, g in self.graph_dict.items()}

    @classmethod
    def from_onehot(
        self,
        labels_one_hot: np.ndarray,
        n_neighbors: int
    ):
        graphs = one_hot_labels_to_graphs(
            labels_one_hot=labels_one_hot
        )
        return AnalyticalFlow(
            graph_dict=graphs,
            n_interpol=n_neighbors
        )

    def get_flow(self, label: int, pos: np.ndarray):
        # inverse distance weighting https://stackoverflow.com/questions/3104781/inverse-distance-weighted-idw-interpolation-with-python

        distance, nearest_vertex = self.kdtrees[label].query(
            pos, k=self.n_interpol)

        # distances and nearest_vertex might be arrays if self.n_interpol > 1
        target = self.positions[label][nearest_vertex]
        tangent = self.tangents[label][nearest_vertex]
        radius = self.radiuses[label][nearest_vertex]

        antinormal = normalize_vec(target - pos, axis=-1)
        alpha = np.clip(distance / radius, 0.0, 1.0)
        if self.n_interpol > 1:
            alpha = alpha[:, None]
        vec = alpha * antinormal + (1-alpha) * tangent

        if self.n_interpol > 1:
            # agglomerate results if interpolating
            weights = 1 / np.clip(distance, 1e-16, np.inf)
            weights = weights / np.sum(weights)
            vec = np.sum(vec * weights[:, None], axis=0)

        vec = normalize_vec(vec)
        return vec
