"""Dijkstra's algorithm and Floyd-Warshall with exact distances."""

from fractions import Fraction

from topics.network_optimization.models import (
    DijkstraState, FloydState, Graph, MatrixChange, Step, Weight, path_edges, predecessor_path,
)


def dijkstra(graph: Graph, start: int, target: int | None) -> tuple[Step, ...]:
    if any(graph.weight(edge) < 0 for edge in graph.edges()):
        raise ValueError("Dijkstra benötigt nichtnegative Kantengewichte.")
    distances: list[Weight] = [None] * graph.size
    distances[start] = Fraction(0)
    parents: list[int | None] = [None] * graph.size
    fixed: list[int] = []
    visited = set()
    steps = []

    def record(title: str, text: str, focus=(), edge=None, *, final=False):
        state = DijkstraState(tuple(distances), tuple(parents), tuple(fixed), start, target, final)
        selected = frozenset(graph.edge(p, v) for v, p in enumerate(parents) if p is not None)
        path = predecessor_path(parents, start, target) if final and target is not None else None
        steps.append(Step(title, text, state, selected=selected, focus_nodes=focus,
                          focus_edge=edge, path=path_edges(graph, path)))

    record("Initialisierung", f"Distanz zu {graph.names[start]} = 0, alle anderen Distanzen = ∞.", (start,))
    while True:
        candidates = [(d, u) for u, d in enumerate(distances) if d is not None and u not in visited]
        if not candidates:
            break
        distance, u = min(candidates)
        visited.add(u)
        fixed.append(u)
        record("Knoten fixieren", f"{graph.names[u]} hat die kleinste vorläufige Distanz: {distance}.", (u,))
        for v in graph.neighbors(u):
            if v in visited:
                continue
            edge = graph.edge(u, v)
            weight = graph.weight(edge)
            alternative = distance + weight
            old = distances[v]
            expression = f"d({graph.names[u]}) + w = {distance} + {weight} = {alternative}"
            if old is None or alternative < old:
                distances[v] = alternative
                parents[v] = u
                record("Distanz verbessern", f"{expression} < {old if old is not None else '∞'} "
                       f"für {graph.names[v]}. Vorgänger wird {graph.names[u]}.", (u, v), edge)
            else:
                record("Distanz beibehalten", f"{expression} ≥ {old} für {graph.names[v]}.", (u, v), edge)
    message = f"Alle {len(fixed)} erreichbaren Knoten sind fixiert."
    if target is not None:
        distance = distances[target]
        message += f" Distanz zum Ziel: {distance}." if distance is not None else " Das Ziel ist nicht erreichbar."
    record("Ergebnis", message, final=True)
    return tuple(steps)


def floyd_warshall(graph: Graph, start: int, target: int | None) -> tuple[Step, ...]:
    n = graph.size
    distances = graph.weights
    hops = tuple(tuple(j if weight is not None else None for j, weight in enumerate(row)) for row in distances)
    steps = [Step("Initialisierung", "Direkte Gewichte, 0 auf der Diagonale und ∞ für fehlende Kanten.",
                  FloydState(distances))]
    for k in range(n):
        following = [list(row) for row in distances]
        next_hops = [list(row) for row in hops]
        changes = []
        for i, row in enumerate(distances):
            if row[k] is None:
                continue
            for j, second in enumerate(distances[k]):
                if second is None:
                    continue
                candidate = row[k] + second
                if row[j] is None or candidate < row[j]:
                    following[i][j] = candidate
                    next_hops[i][j] = hops[i][k]
                    changes.append(MatrixChange(i, j, row[j], candidate))
        distances = tuple(tuple(row) for row in following)
        hops = tuple(tuple(row) for row in next_hops)
        name = graph.names[k]
        steps.append(Step(
            f"Zwischenknoten {name}",
            f"Vergleiche jede bisherige Distanz mit dem Weg über {name}. {len(changes)} Einträge verbessert.",
            FloydState(distances, via=k, changes=tuple(changes)), focus_nodes=(k,),
        ))

    cycles = tuple(i for i in range(n) if distances[i][i] < 0)
    unbounded = frozenset(
        (i, j) for k in cycles for i in range(n) for j in range(n)
        if distances[i][k] is not None and distances[k][j] is not None
    )
    query = (start, target) if target is not None else None
    path = _next_hop_path(hops, *query) if query is not None and query not in unbounded else None
    message = (
        "Negative Zyklen erkannt. Für die markierten Paare (−∞) existiert keine endliche kürzeste Distanz."
        if cycles else "Die Matrix enthält alle kürzesten Distanzen."
    )
    if query is not None and query not in unbounded and path is None:
        message += " Das gewählte Ziel ist vom Start aus nicht erreichbar."
    steps.append(Step("Ergebnis", message, FloydState(distances, unbounded=unbounded, path=path, query=query),
                      focus_nodes=cycles, path=path_edges(graph, path)))
    return tuple(steps)


def _next_hop_path(hops, start: int, target: int) -> tuple[int, ...] | None:
    path = [start]
    while path[-1] != target:
        hop = hops[path[-1]][target]
        if hop is None or len(path) == len(hops):
            return None
        path.append(hop)
    return tuple(path)
