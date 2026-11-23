"""Graph and polytope views and mathematical explanations for the connector lesson."""

from html import escape

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PySide6.QtGui import QBrush, QColor, QPen

from topics.integer_programming.formatting import _document, point_text
from topics.network_optimization.visualization import EDGE, FOCUS, PATH, REJECTED, SELECTED, NetworkView


GROUP_COLORS = ("#2563eb", "#c2410c", "#047857", "#9333ea", "#be123c", "#0e7490")
FRACTIONAL = "#fb923c"
STEPS = (
    "Original graph", "Binary edge variables", "Cut formulation", "Solve the weak LP",
    "Inspect integrality", "Add partition inequalities", "Solve the stronger LP",
    "Compare both versions", "Optimal binary connector and spanning tree",
)


class ConnectorGraphView(NetworkView):
    def show_formulation(self, graph, *, point=None, row=None, selected=None) -> None:
        self.set_step(None)
        edges = graph.edges()
        crossing = set(row.indices) if row else set()
        for name, node in zip(graph.names, self._nodes):
            node.setToolTip(name)
        if row:
            for group, nodes in enumerate(row.groups):
                color = (QColor(GROUP_COLORS[group]) if len(row.groups) <= len(GROUP_COLORS)
                         else QColor.fromHsv((group * 137) % 360, 210, 205))
                for i in nodes:
                    self._nodes[i].setBrush(QBrush(color.darker(145)))
                    self._nodes[i].setPen(QPen(color.lighter(145), 3))
                    self._nodes[i].setToolTip(f"{graph.names[i]} · Group {group + 1}")
        for i, edge in enumerate(edges):
            item = self._edges[edge]
            value = point[i] if point is not None else None
            label = str(graph.weight(edge))
            if value is not None:
                label += f" | x={value}"
            item.set_label(label)
            u, v = edge
            item.position(self._positions[u], self._positions[v])
            if i in crossing:
                item.style(FOCUS, 3.3, False)
            elif selected is not None:
                in_tree = edge in selected
                item.style(PATH if in_tree else REJECTED, 3.5 if in_tree else 1.3, not in_tree)
            elif value is not None:
                color = FRACTIONAL if 0 < value < 1 else SELECTED if value == 1 else REJECTED
                item.style(color, 3 if value else 1.3, not value)
            else:
                item.style(EDGE, 1.5, False)
        self._update_scene_extent()


def row_text(models, row) -> str:
    lhs = " + ".join(f"x{i + 1}" for i in row.indices) or "0"
    groups = " | ".join(
        "{" + ", ".join(escape(models.graph.names[i]) for i in group) + "}" for group in row.groups
    )
    return f"{groups} ⟹ {lhs} ≥ {row.rhs}"


def _lp_text(result, complete) -> str:
    if result.status == "infeasible":
        return "Infeasible. The input graph is disconnected."
    if result.status == "bound_only":
        return f"Certified lower bound {result.bound}. Full LP solve skipped at this size or pivot limit."
    scope = "Complete model" if complete else "Limited model with selected valid inequalities"
    return f"{scope}. z<sub>LP</sub> = {result.bound}, x = {point_text(result.point)}."


def _display_rows(models, rows) -> str:
    return "<ul>" + "".join(f"<li>{row_text(models, row)}</li>" for row in rows) + "</ul>" if rows else ""


def format_spanning(models, index, weak=None, strong=None, row=None, witness=None, optimum=None,
                    *, diagram=False) -> str:
    graph = models.graph
    body = f"<h3>{STEPS[index]}</h3>"
    if index == 0:
        return _document(body + (
            f"<p>{graph.size} nodes and {len(models.edges)} edges. A connector reaches every node. "
            "A spanning tree is a connected connector without a cycle and has |V| − 1 edges.</p>"
        ))
    if index == 1:
        terms = " + ".join(f"{graph.weight(e)}·x{i + 1}" for i, e in enumerate(models.edges)) or "0"
        edges = "<br/>".join(f"x{i + 1} ↔ {escape(graph.edge_name(e))}, c = {graph.weight(e)}"
                            for i, e in enumerate(models.edges))
        return _document(body + f"<p>min z = {terms}<br/>xₑ ∈ {{0, 1}} for every edge.</p>"
                         f"<p>{edges}</p><p>xₑ = 1 selects that edge.</p>")
    if index == 2:
        active = f"<p><b>{row_text(models, row)}</b></p>" if row else ""
        return _document(body + (
            "<p>For every nonempty proper S ⊂ V, Σ<sub>e∈δ(S)</sub>xₑ ≥ 1. "
            "In the graph the highlighted edges join the two sides. At least one of those edges "
            "must be selected to keep the graph connected. With 0 ≤ xₑ ≤ 1 this gives the weak LP.</p>"
            f"{active}<p><b>{len(models.cuts)} cut rows</b> · {'Complete' if models.cuts_complete else 'Selected cuts'}</p>"
            f"{_display_rows(models, models.cuts)}"
        ))
    if index == 3:
        return _document(body + (
            "<p>Relax xₑ ∈ {0, 1} to 0 ≤ xₑ ≤ 1, keeping the same cut constraints and objective.</p>"
            f"<p>{_lp_text(weak, models.cuts_complete)}</p>"
        ))
    if index == 4:
        if witness is None:
            note = ("The computed LP optimum is binary." if weak.point is not None else
                    "There is no computed LP point to inspect.")
        else:
            status = "optimal" if witness == weak.point else "feasible vertex"
            fractional = ", ".join(f"x{i + 1} = {x}" for i, x in enumerate(witness) if 0 < x < 1)
            note = f"The {status} point {point_text(witness)} uses fractional edges {fractional}."
        return _document(body + f"<p>{note}</p><p>Fractional edges can satisfy all cut sums in the LP.</p>"
                         + _diagram_note(diagram))
    if index == 5:
        detail = ""
        if row:
            detail = f"<p><b>{row_text(models, row)}</b></p>"
            point = witness if witness is not None else weak.point
            if point is not None:
                activity = row.activity(point)
                relation = "&lt;" if activity < row.rhs else "≥"
                effect = "This row excludes the weak LP point shown." if activity < row.rhs else (
                    "The weak LP point shown already satisfies this row."
                )
                detail += f"<p>LHS at {point_text(point)} = {activity} {relation} {row.rhs}. {effect}</p>"
        return _document(body + (
            "<p>For a partition ℘ with k groups, require Σ<sub>e∈δ(℘)</sub>xₑ ≥ k − 1. "
            "A connector must link the k contracted groups. The marked edges cross those groups.</p>"
            f"{detail}<p>{len(models.additions)} additional rows. The box bounds remain 0 ≤ xₑ ≤ 1.</p>"
            f"{_display_rows(models, models.additions)}"
        ))
    if index == 6:
        return _document(body + (
            f"<p>{_lp_text(strong, models.partitions_complete)}</p>"
            "<p>The LP uses the cut rows and the added partition rows with the same edge bounds.</p>"
            + _diagram_note(diagram)
        ))
    if index == 7:
        return _document(body + _comparison(models, weak, strong) + _diagram_note(diagram))
    if optimum is None:
        return _document(body + "<p>The graph is disconnected. There is no spanning tree or binary connector.</p>"
                         + _comparison(models, weak, strong))
    point = tuple(int(e in optimum.connector) for e in models.edges)
    tree = ", ".join(escape(graph.edge_name(e)) for e in sorted(optimum.tree)) or "∅"
    weights_positive = all(graph.weight(e) > 0 for e in models.edges)
    explanation = (
        "With strictly positive weights, removing any cycle edge from a connector improves its cost. "
        "The optimal connector is therefore a minimum spanning tree."
        if weights_positive else
        "With zero or negative edges a connector can contain cycles at optimum. Kruskal still computes "
        "the minimum-cost tree under the separate acyclicity requirement."
    )
    return _document(body + (
        f"<p>Optimal binary connector x = {point_text(point)} and z<sub>ILP</sub> = {optimum.connector_cost}.</p>"
        f"<p><b>Kruskal tree</b> ({tree}), total weight {optimum.tree_cost}. "
        "The green edges are the tree; the other input edges remain visible.</p>"
        f"<p>{optimum.verification}. {explanation}</p><p>{_chain(weak, strong, optimum.connector_cost)}</p>"
    ) + _comparison(models, weak, strong))


def _diagram_note(enabled) -> str:
    return "<p>Toggle the 3D polytopes and drag to rotate the view.</p>" if enabled else ""


def _chain(weak, strong, integer=None) -> str:
    valid = weak.bound is not None and strong.bound is not None
    if not valid:
        return ""
    name = "LP values" if weak.status == strong.status == "optimal" else "Certified LP bounds"
    return (f"{name} {weak.bound} ≤ {strong.bound}" + (f" ≤ {integer} = z<sub>ILP</sub>." if integer is not None
            else " ≤ z<sub>ILP</sub>."))


def _comparison(models, weak, strong) -> str:
    vertices = "Integral for the full connector polytope" if models.partitions_complete else (
        "Full-system integrality guarantee; the selected subset can allow fractional vertices"
    )
    return (
        "<table><tr><th width='50%'>V1</th><th width='50%'>V2</th></tr>"
        f"<tr><td>{len(models.cuts)} cut rows<br/>{_lp_text(weak, models.cuts_complete)}</td>"
        f"<td>{len(models.partition_rows)} partition rows<br/>{_lp_text(strong, models.partitions_complete)}</td></tr>"
        f"<tr><td>Fractional vertices possible</td><td>{vertices}</td></tr></table>"
        f"<p>{_chain(weak, strong)}</p><p>Adding valid partition rows shrinks the LP region while retaining "
        "every binary connector. Both full formulations impose the same binary connectivity.</p>"
    )


def _sort_face(face):
    points = np.asarray(face, dtype=float)
    centered = points - points.mean(axis=0)
    u = centered[0]
    v = max((np.cross(u, point) for point in centered[1:]), key=np.linalg.norm)
    if not np.any(v):
        return points
    u = u / np.linalg.norm(u)
    v = np.cross(v / np.linalg.norm(v), u)
    return points[np.argsort(np.arctan2(centered @ v, centered @ u))]


def plot_polytopes(figure, polytopes) -> None:
    figure.clear()
    figure.subplots_adjust(left=.03, right=.90, bottom=.12, top=.91, wspace=.23)
    for i, (title, polytope) in enumerate(polytopes, 1):
        axis = figure.add_subplot(1, len(polytopes), i, projection="3d")
        axis.set_facecolor("#111827")
        axis.set_title(title, color="#e2e8f0", fontsize=9, pad=6)
        for k, dimension in enumerate((axis.xaxis, axis.yaxis, axis.zaxis), 1):
            dimension.set_label_text(f"x{k}", color="#cbd5e1")
            dimension.labelpad = -4
            dimension.set_pane_color((.13, .19, .28, .26))
            dimension.set_ticks((0, .5, 1))
        axis.tick_params(colors="#cbd5e1", labelsize=6, pad=-1)
        axis.set(xlim=(-.04, 1.04), ylim=(-.04, 1.04), zlim=(-.04, 1.04))
        axis.set_box_aspect((1, 1, 1))
        axis.view_init(22, -55)
        if polytope.faces:
            color = "#60a5fa" if i == 1 else "#2dd4bf"
            axis.add_collection3d(Poly3DCollection([_sort_face(face) for face in polytope.faces],
                                                   facecolor=color, edgecolor=color, alpha=.22))
        for fractional, color in ((False, "#4ade80"), (True, FRACTIONAL)):
            points = [p for p in polytope.vertices if any(x.denominator != 1 for x in p) == fractional]
            if points:
                axis.scatter(*np.asarray(points, dtype=float).T, c=color, s=22, depthshade=False)
    figure.text(.5, .04, "Green · binary vertices       Orange · fractional vertices",
                color="#cbd5e1", fontsize=7, ha="center")
