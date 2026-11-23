"""Render a saved step without advancing the algorithm."""

from html import escape

from topics.network_optimization.models import (
    DijkstraState, FloydState, Graph, SpanningState, Step, TraversalState, predecessor_path,
)


def _nodes(graph: Graph, values, separator: str = ", ") -> str:
    return separator.join(escape(graph.names[node]) for node in values) or "—"


def _edges(graph: Graph, values) -> str:
    return ", ".join(f"{escape(graph.edge_name(edge))} ({graph.weight(edge)})" for edge in sorted(values)) or "—"


def _number(value) -> str:
    return "∞" if value is None else str(value)


def format_step(graph: Graph, step: Step, index: int, count: int) -> str:
    detail = ""
    if step.focus_edge is not None:
        detail = f"<p><b>Betrachtete Kante:</b> {_edges(graph, (step.focus_edge,))}</p>"
    elif step.focus_nodes:
        detail = f"<p><b>Betrachtete Knoten:</b> {_nodes(graph, step.focus_nodes)}</p>"
    state = step.state
    if isinstance(state, TraversalState):
        finish_label = "Finishing Order" if state.depth_first else "Processing / Finishing Order"
        body = (
            f"<p><b>Discovery Order:</b> {_nodes(graph, state.discovery_order, ' → ')}<br>"
            f"<b>{finish_label}:</b> {_nodes(graph, state.finishing_order, ' → ')}<br>"
            f"<b>Entdeckt:</b> {_nodes(graph, state.discovered)}<br>"
            f"<b>{escape(state.frontier_label)}:</b> {_nodes(graph, state.frontier)}<br>"
            f"<b>Suchbaumkanten:</b> {_edges(graph, step.selected)}</p>"
        )
        if state.path is not None:
            body += f"<p><b>Weg zum Ziel:</b> {_nodes(graph, state.path, ' → ')}</p>"
    elif isinstance(state, SpanningState):
        body = (
            f"<p><b>Gewichtssumme:</b> {state.cost} · <b>Komponenten:</b> {state.components}</p>"
            f"<p><b>{escape(state.remaining_label)}:</b> {_edges(graph, step.selected)}<br>"
            f"<b>Verworfene Kanten:</b> {_edges(graph, step.rejected)}</p>"
        )
    elif isinstance(state, DijkstraState):
        body = _format_dijkstra(graph, state)
    else:
        body = _format_floyd(graph, state)
    return (
        "<html><head><style>body{color:#e2e8f0;font-size:10pt}"
        "h3,b,th{color:#9bc6ef} th,td{padding:4px 7px;border-bottom:1px solid #334155}"
        ".changed{color:#fde68a;background:#334155}</style></head><body>"
        f"<h3>Schritt {index + 1}/{count} · {escape(step.title)}</h3>"
        f"<p>{escape(step.explanation)}</p>{detail}{body}</body></html>"
    )


def _format_dijkstra(graph: Graph, state: DijkstraState) -> str:
    rows = []
    for v, distance in enumerate(state.distances):
        parent = state.parents[v]
        row = (
            f"<td>{escape(graph.names[v])}</td><td>{_number(distance)}</td>"
            f"<td>{'—' if parent is None else escape(graph.names[parent])}</td>"
            f"<td>{'Fixiert' if v in state.fixed else 'Unerreichbar' if state.final else 'Offen'}</td>"
        )
        if state.final:
            path = predecessor_path(state.parents, state.start, v)
            row += f"<td>{_nodes(graph, path or (), ' → ')}</td>"
        rows.append(f"<tr>{row}</tr>")
    path_header = "<th>Kürzester Weg</th>" if state.final else ""
    return (
        f"<p><b>Fixierreihenfolge:</b> {_nodes(graph, state.fixed, ' → ')}</p>"
        f"<table><tr><th>Knoten</th><th>Distanz</th><th>Vorgänger</th><th>Status</th>{path_header}</tr>"
        + "".join(rows) + "</table>"
    )


def _format_floyd(graph: Graph, state: FloydState) -> str:
    changed = {(change.source, change.target) for change in state.changes}
    headers = "<th>Von / Nach</th>" + "".join(f"<th>{escape(name)}</th>" for name in graph.names)
    rows = []
    for i, row in enumerate(state.distances):
        cells = [f"<th>{escape(graph.names[i])}</th>"]
        for j, value in enumerate(row):
            pair = (i, j)
            number = "−∞" if pair in state.unbounded else _number(value)
            style = ' class="changed"' if pair in changed or pair in state.unbounded else ""
            cells.append(f"<td{style}>{number}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    body = f"<table><tr>{headers}</tr>{''.join(rows)}</table>"
    if state.changes:
        changes = ", ".join(
            f"{escape(graph.names[c.source])} → {escape(graph.names[c.target])}: {_number(c.before)} → {c.after}"
            for c in state.changes
        )
        body += f"<p><b>Geänderte Einträge:</b> {changes}</p>"
    if state.query is not None:
        start, target = state.query
        value = "−∞" if state.query in state.unbounded else _number(state.distances[start][target])
        body += f"<p><b>Gewähltes Paar:</b> {_nodes(graph, state.query, ' → ')} · Distanz {value}</p>"
        if state.path is not None:
            body += f"<p><b>Kürzester Weg:</b> {_nodes(graph, state.path, ' → ')}</p>"
    return body
