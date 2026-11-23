"""Minimum spanning forests by Kruskal, reverse-delete and Prim."""

from fractions import Fraction

from topics.network_optimization.models import Edge, Graph, Method, SpanningState, Step


class _Components:
    def __init__(self, size: int) -> None:
        self.parents = list(range(size))
        self.count = size

    def root(self, u: int) -> int:
        while self.parents[u] != u:
            self.parents[u] = self.parents[self.parents[u]]
            u = self.parents[u]
        return u

    def join(self, u: int, v: int) -> bool:
        a, b = self.root(u), self.root(v)
        if a == b:
            return False
        self.parents[b] = a
        self.count -= 1
        return True


def minimum_spanning(graph: Graph, method: Method, start: int) -> tuple[Step, ...]:
    edges = graph.edges()
    selected: set[Edge] = set(edges) if method is Method.REVERSE_DELETE else set()
    rejected: set[Edge] = set()
    steps = []
    label = "Verbleibende Kanten" if method is Method.REVERSE_DELETE else "Gewählte Kanten"

    def record(title: str, text: str, edge: Edge | None = None, focus: tuple[int, ...] = ()) -> None:
        components = _Components(graph.size)
        for pair in selected:
            components.join(*pair)
        state = SpanningState(sum((graph.weight(pair) for pair in selected), Fraction(0)),
                              components.count, label)
        steps.append(Step(title, text, state, frozenset(selected), frozenset(rejected), focus, edge))

    if method is Method.KRUSKAL:
        record("Initialisierung", "Prüfe die Kanten aufsteigend nach Gewicht. Bei Gleichstand gilt die Matrixreihenfolge.")
        components = _Components(graph.size)
        for edge in sorted(edges, key=lambda e: (graph.weight(e), e)):
            if components.join(*edge):
                selected.add(edge)
                record("Kante aufnehmen", f"{graph.edge_name(edge)} verbindet zwei getrennte Komponenten.", edge, edge)
            else:
                rejected.add(edge)
                record("Kante verwerfen", f"{graph.edge_name(edge)} würde einen Zyklus bilden.", edge, edge)
    elif method is Method.REVERSE_DELETE:
        record("Initialisierung", "Beginne mit allen Kanten und prüfe die teuersten zuerst.")
        for edge in sorted(edges, key=lambda e: (-graph.weight(e), e)):
            selected.remove(edge)
            components = _Components(graph.size)
            for pair in selected:
                components.join(*pair)
            if components.root(edge[0]) == components.root(edge[1]):
                rejected.add(edge)
                record("Kante entfernen", f"Die Endknoten von {graph.edge_name(edge)} bleiben verbunden.", edge, edge)
            else:
                selected.add(edge)
                record("Kante behalten", f"Das Entfernen von {graph.edge_name(edge)} würde eine Komponente trennen.", edge, edge)
    else:
        seen = {start}
        record("Initialisierung", f"Die erste Komponente startet bei {graph.names[start]}.", focus=(start,))
        while len(seen) < graph.size:
            candidates = [edge for edge in edges if (edge[0] in seen) != (edge[1] in seen)]
            if not candidates:
                u = next(i for i in range(graph.size) if i not in seen)
                seen.add(u)
                record("Neue Komponente", f"Keine Kante erreicht weitere Knoten. Starte neu bei {graph.names[u]}.",
                       focus=(u,))
                continue
            edge = min(candidates, key=lambda e: (graph.weight(e), e))
            selected.add(edge)
            seen.update(edge)
            rejected.update(pair for pair in edges if set(pair) <= seen and pair not in selected)
            record("Günstigste Verbindung", f"{graph.edge_name(edge)} ist die günstigste Kante zu einem neuen Knoten.",
                   edge, edge)

    components = graph.size - len(selected)
    kind = "Minimaler Spannbaum" if components == 1 else f"Minimaler Spannwald mit {components} Komponenten"
    record("Ergebnis", f"{kind}. {len(selected)} Kanten verbleiben.")
    return tuple(steps)
