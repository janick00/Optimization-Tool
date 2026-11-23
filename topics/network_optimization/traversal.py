"""Depth-first and breadth-first search in matrix order."""

from collections import deque

from topics.network_optimization.models import Graph, Step, TraversalState, path_edges, predecessor_path


def search(graph: Graph, start: int, target: int | None, *, depth_first: bool) -> tuple[Step, ...]:
    discovery = [start]
    finishing: list[int] = []
    discovered = {start}
    parents: list[int | None] = [None] * graph.size
    frontier = deque([start])
    steps = []
    label = "Stack · Spitze rechts" if depth_first else "Queue · nächster Knoten links"

    def record(title: str, explanation: str, focus: int | None = None, edge=None, path=None) -> None:
        selected = frozenset(graph.edge(parent, i) for i, parent in enumerate(parents) if parent is not None)
        discovery_order = tuple(discovery)
        finishing_order = tuple(finishing)
        state = TraversalState(
            order=discovery_order if depth_first else finishing_order,
            discovered=tuple(sorted(discovered)), frontier=tuple(frontier), frontier_label=label,
            parents=tuple(parents), discovery_order=discovery_order, finishing_order=finishing_order,
            depth_first=depth_first, path=path,
        )
        steps.append(Step(title, explanation, state, selected=selected,
                          focus_nodes=() if focus is None else (focus,), focus_edge=edge,
                          path=path_edges(graph, path)))

    record("Initialisierung", f"{graph.names[start]} wird in die {'Stack' if depth_first else 'Queue'}-Struktur gelegt.", start)
    if depth_first:
        # Keep the active path on the stack and advance one neighbor at a time.
        while frontier:
            u = frontier[-1]
            v = next((v for v in graph.neighbors(u) if v not in discovered), None)
            if v is None:
                frontier.pop()
                finishing.append(u)
                record("Zurückgehen", f"{graph.names[u]} ist vollständig bearbeitet und wird vom Stack entfernt.", u)
            else:
                discovered.add(v)
                discovery.append(v)
                parents[v] = u
                frontier.append(v)
                record("In die Tiefe gehen", f"{graph.names[v]} ist der erste unbesuchte Nachbar von {graph.names[u]}.",
                       v, graph.edge(u, v))
    else:
        while frontier:
            u = frontier.popleft()
            added = []
            for v in graph.neighbors(u):
                if v not in discovered:
                    discovered.add(v)
                    discovery.append(v)
                    parents[v] = u
                    frontier.append(v)
                    added.append(graph.names[v])
            finishing.append(u)
            detail = "Neu in der Queue: " + ", ".join(added) if added else "Keine neuen Nachbarn."
            record("Nächsten Knoten besuchen",
                   f"{graph.names[u]} wird vorne aus der Queue genommen. Alle Nachbarn wurden geprüft. {detail}", u)

    path = predecessor_path(parents, start, target) if target is not None else None
    explanation = f"{len(discovery)} von {graph.size} Knoten sind vom Start erreichbar."
    if target is not None:
        if path is None:
            explanation += f" {graph.names[target]} ist nicht erreichbar."
        elif not depth_first:
            explanation += f" Der Weg zum Ziel hat minimal {len(path) - 1} Kanten."
        else:
            explanation += f" Der DFS-Baum liefert einen Weg mit {len(path) - 1} Kanten."
    record("Ergebnis", explanation, path=path)
    return tuple(steps)
