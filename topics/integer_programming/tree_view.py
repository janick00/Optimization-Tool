"""Zoomable, selectable tree backed by an immutable playback frame."""

from collections.abc import Iterable
from typing import Protocol

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QGraphicsRectItem, QGraphicsScene, QGraphicsSimpleTextItem, QGraphicsView

from topics.integer_programming.formatting import point_text
from topics.integer_programming.models import Node, NodeStatus


COLORS = {
    NodeStatus.OPEN: "#334155", NodeStatus.LP_SOLVED: "#1e40af", NodeStatus.FRACTIONAL: "#854d0e",
    NodeStatus.INTEGER: "#155e75", NodeStatus.PRUNED: "#583b84",
    NodeStatus.INFEASIBLE: "#991b1b", NodeStatus.OPTIMAL: "#166534",
    NodeStatus.UNRESOLVED: "#7c2d12",
}
BOX_WIDTH, BOX_HEIGHT = 202, 96
HORIZONTAL, VERTICAL = 230, 152


class TreeNode(Protocol):
    id: int
    parent: int | None
    status: NodeStatus


def tree_positions(nodes: Iterable[TreeNode], *, layer_gap: int = VERTICAL) -> dict[int, tuple[float, float]]:
    """Assign each subtree a disjoint horizontal interval of leaf slots."""

    ordered = sorted(nodes, key=lambda node: node.id)
    if not ordered:
        return {}
    children: dict[int, list[int]] = {node.id: [] for node in ordered}
    for node in ordered:
        if node.parent is not None:
            children[node.parent].append(node.id)
    widths = {}
    for node in reversed(ordered):
        widths[node.id] = sum(widths[child] for child in children[node.id]) or 1
    positions = {}
    stack = [(ordered[0].id, 0, 0)]
    while stack:
        node_id, left, depth = stack.pop()
        positions[node_id] = ((left + (widths[node_id] - 1) / 2) * HORIZONTAL, depth * layer_gap)
        for child in children[node_id]:
            stack.append((child, left, depth + 1))
            left += widths[child]
    return positions


def _default_node_lines(node: Node) -> tuple[str, ...]:
    return (f"{'Root' if node.parent is None else 'Node'} {node.id}",
            f"x = {point_text(node.solution)}", f"J = {node.value if node.value is not None else '—'}",
            node.status.value)


def _default_edge_label(node: Node) -> str:
    return node.branches[-1].label(node.problem.variable_names)


class _NodeItem(QGraphicsRectItem):
    def __init__(self, node: TreeNode, x: float, y: float, select, lines, height: int) -> None:
        super().__init__(QRectF(-BOX_WIDTH / 2, 0, BOX_WIDTH, height))
        self.node = node
        self._select = select
        self.setPos(x, y)
        self.setData(0, node.id)
        self.setBrush(QBrush(QColor(COLORS[node.status])))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setZValue(1)
        font = QFont("Segoe UI", 10)
        metrics = QFontMetrics(font)
        content = "\n".join(
            metrics.elidedText(line, Qt.TextElideMode.ElideRight, BOX_WIDTH - 24) for line in lines
        )
        self.setToolTip("\n".join(lines))
        label = QGraphicsSimpleTextItem(content, self)
        label.setBrush(QBrush(QColor("#f8fafc")))
        label.setFont(font)
        label.setPos(-BOX_WIDTH / 2 + 12, 7)
        label.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._select(self.node.id)
            event.accept()
        else:
            super().mousePressEvent(event)


class BranchTreeView(QGraphicsView):
    node_selected = Signal(int)

    def __init__(self, *, node_lines=_default_node_lines, edge_label=_default_edge_label,
                 box_height: int = BOX_HEIGHT) -> None:
        super().__init__()
        self._node_lines = node_lines
        self._edge_label = edge_label
        self._box_height = box_height
        self.setScene(QGraphicsScene(self))
        self.setBackgroundBrush(QBrush(QColor("#0f172a")))
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._node_items: dict[int, _NodeItem] = {}
        self._fit_when_shown = False

    def show_nodes(self, nodes: dict[int, TreeNode], selected: int) -> None:
        scene = self.scene()
        scene.clear()
        self._node_items.clear()
        height = self._box_height
        positions = tree_positions(nodes.values(), layer_gap=VERTICAL - BOX_HEIGHT + height)
        line = QPen(QColor("#94a3b8"), 1.3)
        for node in nodes.values():
            x, y = positions[node.id]
            if node.parent is not None:
                px, py = positions[node.parent]
                middle = (py + height + y) / 2
                scene.addLine(px, py + height, px, middle, line)
                scene.addLine(px, middle, x, middle, line)
                scene.addLine(x, middle, x, y, line)
                condition = self._edge_label(node)
                label = scene.addSimpleText("")
                label.setText(QFontMetrics(label.font()).elidedText(
                    condition, Qt.TextElideMode.ElideRight, HORIZONTAL - 20,
                ))
                label.setToolTip(condition)
                label.setBrush(QBrush(QColor("#cbd5e1")))
                label.setPos(x + 5, y - label.boundingRect().height())
            item = _NodeItem(node, x, y, self.node_selected.emit, self._node_lines(node), height)
            scene.addItem(item)
            self._node_items[node.id] = item
        scene.setSceneRect(scene.itemsBoundingRect().adjusted(-18, -16, 18, 16))
        self.highlight(selected)

    def highlight(self, selected: int) -> None:
        for node, item in self._node_items.items():
            item.setPen(QPen(QColor("#f8fafc" if node == selected else "#64748b"),
                             3 if node == selected else 1))

    def center_on_node(self, node: int) -> None:
        self.centerOn(self._node_items[node])

    def zoom(self, factor: float) -> None:
        target = self.transform().m11() * factor
        if .002 <= target <= 6:
            self.scale(factor, factor)

    def fit_tree(self) -> None:
        if self._node_items:
            if not self.isVisible():
                self._fit_when_shown = True
                return
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            if self.transform().m11() > 1:
                self.resetTransform()
                self.centerOn(self.sceneRect().center())

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._fit_when_shown:
            self._fit_when_shown = False
            self.fit_tree()
