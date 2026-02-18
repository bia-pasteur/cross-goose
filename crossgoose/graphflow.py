

import json
from typing import Dict

import networkx as nx
import numpy as np
from scipy.spatial import KDTree
from skimage.morphology import skeletonize
from tqdm import tqdm

from crossgoose.graph_utils import get_networkx_graph_from_array


def get_nth_predecessor(graph: nx.Graph, vert, n: int) -> int:
    for _ in range(n):
        vert = graph.nodes[vert]['pred']
    return vert


def subsample_graph(graph: nx.Graph, factor: int):
    to_remove = [
        n for n in graph.nodes()
        if (graph.degree[n] == 2) and (n % factor != 0)
    ]
    for n in to_remove:
        adj0, adj1 = graph.adj[n]
        graph.remove_node(n)
        graph.add_edge(adj0, adj1)
    return graph


def one_hot_labels_to_graphs(labels_one_hot: np.ndarray):
    n_instances = len(labels_one_hot)
    graphs = {}
    for k in tqdm(range(n_instances)):
        mask = labels_one_hot[k]
        skel = skeletonize(mask)
        graph = get_networkx_graph_from_array(skel)
        # graph = convert_graph_to_native_int(graph)
        graph = store_pos_in_key(graph)
        store_predecessor_and_distance(graph)

        graphs[k+1] = graph
    return graphs


class AnalyticalFlow:
    def __init__(
        self,
        graph_dict: Dict[str, nx.Graph],
        degree: int,
        n_neighbors: int
    ):

        self.graph_dict = graph_dict

        self.kdtrees = {
            k: KDTree(np.array([g.nodes[n]['pos'] for n in g.nodes()])) for k, g in self.graph_dict.items()
        }

        self.targets = {
            k: np.array([
                g.nodes[
                    get_nth_predecessor(g, n, degree)
                ]['pos']
                for n in g.nodes()])
            for k, g in self.graph_dict.items()}

        self.n_neighbors = n_neighbors

    @classmethod
    def from_onehot(
        self,
        labels_one_hot: np.ndarray,
        degree: int,
        n_neighbors: int
    ):
        graphs = one_hot_labels_to_graphs(
            labels_one_hot=labels_one_hot
        )
        return AnalyticalFlow(
            graph_dict=graphs,
            degree=degree,
            n_neighbors=n_neighbors
        )

    def get_flow(self, label: int, pos: np.ndarray):
        # inverse distance weighting https://stackoverflow.com/questions/3104781/inverse-distance-weighted-idw-interpolation-with-python

        if self.n_neighbors > 1:
            distances, nearest_vertices = self.kdtrees[label].query(
                pos, k=self.n_neighbors)
            inv_distances = 1 / np.clip(distances, 1e-16, np.inf)
            inv_distances = inv_distances / np.sum(inv_distances)
            targets = self.targets[label][nearest_vertices]
            target = np.sum(targets * inv_distances[:, None], axis=0)
        elif self.n_neighbors == 1:
            _, nearest_vertex = self.kdtrees[label].query(pos)
            target = self.targets[label][nearest_vertex]
        else:
            raise ValueError(self.n_neighbors)

        vec = (target - pos)
        norm = np.linalg.norm(vec)
        if norm > 0.0:
            vec = vec / norm
        return vec


def store_pos_in_key(graph: nx.Graph):
    for n in graph.nodes():
        graph.nodes[n]['pos'] = [n[0].item(), n[1].item()]
    return nx.relabel_nodes(graph, {n: i for i, n in enumerate(graph.nodes())})


def store_predecessor_and_distance(graph: nx.Graph):
    center = nx.center(graph)
    # assert len(center) == 1
    start = center[0]

    graph.nodes[start]['dist'] = 0
    graph.nodes[start]['pred'] = start

    # visited = set()
    stack = [start]
    while len(stack) > 0:
        vert = stack.pop(-1)
        d = graph.nodes[vert]['dist']
        neighbors = graph.adj[vert]
        for n in neighbors:
            if graph.nodes[n].get('dist', np.inf) > (d+1):
                graph.nodes[n]['pred'] = vert
                graph.nodes[n]['dist'] = d + 1
                stack.append(n)


def relax_positions(graph: nx.Graph, niter: int = 1):
    if niter == 0:
        return graph
    new_pos = {}
    for n in graph.nodes():
        positions = np.array(
            [graph.nodes[n]['pos']]
            + [
                graph.nodes[nn]['pos'] for nn in graph.adj[n]
            ])
        avg_pos = np.mean(positions, axis=0)
        new_pos[n] = avg_pos.tolist()
    nx.set_node_attributes(graph, new_pos, name='pos')
    if niter == 1:
        return graph
    else:
        return relax_positions(graph, niter-1)


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
