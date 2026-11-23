"""Dispatch graph algorithms through a shared step interface."""

from topics.network_optimization.models import Graph, Method, Step
from topics.network_optimization.shortest_paths import dijkstra, floyd_warshall
from topics.network_optimization.spanning import minimum_spanning
from topics.network_optimization.traversal import search


def run_algorithm(graph: Graph, method: Method, start: int = 0, target: int | None = None) -> tuple[Step, ...]:
    if method.requires_undirected and graph.directed:
        raise ValueError("Dieses Verfahren benötigt einen ungerichteten Graphen.")
    if not 0 <= start < graph.size or (target is not None and not 0 <= target < graph.size):
        raise ValueError("Start und Ziel müssen vorhandene Knoten sein.")
    if method in (Method.DFS, Method.BFS):
        return search(graph, start, target, depth_first=method is Method.DFS)
    if method.requires_undirected:
        return minimum_spanning(graph, method, start)
    if method is Method.DIJKSTRA:
        return dijkstra(graph, start, target)
    if method is Method.FLOYD:
        return floyd_warshall(graph, start, target)
    raise ValueError("Unbekanntes Verfahren.")
