"""A draggable graph canvas for the editor and step playback."""

from math import cos, hypot, pi, sin

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (
    QGraphicsEllipseItem, QGraphicsItem, QGraphicsRectItem, QGraphicsScene,
    QGraphicsSimpleTextItem, QGraphicsView,
)

from topics.network_optimization.models import DijkstraState, Graph, Step, TraversalState


RADIUS = 30
EDGE = "#8393ad"
SELECTED = "#57b7f4"
FOCUS = "#fbbf24"
PATH = "#4ade80"
REJECTED = "#516077"


def _unit(vector: QPointF) -> QPointF:
    length = hypot(vector.x(), vector.y())
    return vector / length if length else QPointF()


def _normal(vector: QPointF) -> QPointF:
    return QPointF(-vector.y(), vector.x())


class _Vertex(QGraphicsEllipseItem):
    def __init__(self, index, name, point, moved):
        super().__init__(-RADIUS, -RADIUS, 2 * RADIUS, 2 * RADIUS)
        self._index = index
        self._moved = moved
        self.setPos(point)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setToolTip(name)
        self.setZValue(3)
        font = QFont("Segoe UI", 10)
        font.setBold(True)
        caption = QFontMetrics(font).elidedText(name, Qt.TextElideMode.ElideRight, 2 * RADIUS - 8)
        label = QGraphicsSimpleTextItem(caption, self)
        label.setFont(font)
        label.setBrush(QBrush(QColor("#f8fafc")))
        label.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        label.setPos(-label.boundingRect().width() / 2, -label.boundingRect().height() / 2)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self._moved(self._index, value)
        return super().itemChange(change, value)


class _Connection:
    def __init__(self, scene, label: str, directed: bool, reciprocal: bool):
        self._directed = directed
        self._reciprocal = reciprocal
        self._line = scene.addPath(QPainterPath())
        self._line.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._arrow = scene.addPolygon(QPolygonF()) if directed else None
        self._weight_box = QGraphicsRectItem()
        self._weight_box.setBrush(QBrush(QColor("#0f172a")))
        self._weight_box.setPen(QPen(Qt.PenStyle.NoPen))
        self._weight_box.setZValue(2)
        self._weight_box.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        scene.addItem(self._weight_box)
        font = QFont("Segoe UI", 9)
        self._label = QGraphicsSimpleTextItem("", self._weight_box)
        self._label.setFont(font)
        self._label.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.set_label(label)

    def set_label(self, label: str) -> None:
        short = QFontMetrics(self._label.font()).elidedText(label, Qt.TextElideMode.ElideRight, 74)
        self._label.setText(short)
        self._label.setToolTip(label)
        bounds = self._label.boundingRect().adjusted(-3, -1, 3, 1)
        self._weight_box.setRect(bounds)

    def position(self, a: QPointF, b: QPointF) -> None:
        direction = _unit(b - a)
        separated = hypot((b-a).x(), (b-a).y()) > 2 * RADIUS + 3
        self._line.setVisible(separated)
        self._weight_box.setVisible(separated)
        if self._arrow is not None:
            self._arrow.setVisible(separated)
        if not separated:
            return
        middle = (a + b) / 2
        normal = _normal(direction)
        control = middle + normal * (56 if self._reciprocal else 0)
        start = a + _unit(control - a) * RADIUS
        end = b + _unit(control - b) * RADIUS
        path = QPainterPath(start)
        path.quadTo(control, end)
        self._line.setPath(path)
        middle = path.pointAtPercent(.5) + normal * (4 if self._reciprocal else 8)
        self._weight_box.setPos(middle - self._weight_box.rect().center())

        if self._arrow is not None:
            tangent = _unit(end - control)
            base = end - tangent * 10
            cross = _normal(tangent) * 4
            self._arrow.setPolygon(QPolygonF([end, base + cross, base - cross]))

    def style(self, color: str, width: float, dashed: bool) -> None:
        pen = QPen(QColor(color), width)
        if dashed:
            pen.setStyle(Qt.PenStyle.DashLine)
        self._line.setPen(pen)
        if self._arrow is not None:
            self._arrow.setPen(QPen(Qt.PenStyle.NoPen))
            self._arrow.setBrush(QBrush(QColor(color)))
        self._label.setBrush(QBrush(QColor("#7c8da7" if dashed else "#e2e8f0")))


class NetworkView(QGraphicsView):
    """Graph edits rebuild geometry. Playback updates only the styles."""

    def __init__(self) -> None:
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setBackgroundBrush(QBrush(QColor("#0f172a")))
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setMinimumSize(205, 145)
        self._positions: list[QPointF] = []
        self._nodes: list[_Vertex] = []
        self._edges: dict[tuple[int, int], _Connection] = {}
        self._pending_fit = False

    def set_graph(self, graph: Graph | None, *, reset_positions: bool = False) -> None:
        scene = self.scene()
        scene.clear()
        self._nodes.clear()
        self._edges.clear()
        if graph is None:
            return
        resize = len(self._positions) != graph.size or reset_positions
        if resize:
            radius = max(110, graph.size * 18)
            self._positions = [
                QPointF(radius * sin(2 * pi * i / graph.size), -radius * cos(2 * pi * i / graph.size))
                for i in range(graph.size)
            ] if graph.size > 1 else [QPointF()]
        for i, name in enumerate(graph.names):
            item = _Vertex(i, name, self._positions[i], self._moved)
            self._nodes.append(item)
            scene.addItem(item)
        edges = set(graph.edges())
        for edge in sorted(edges):
            u, v = edge
            item = _Connection(scene, str(graph.weight(edge)), graph.directed, (v, u) in edges)
            self._edges[edge] = item
            item.position(self._positions[u], self._positions[v])
        self.set_step(None)
        self._update_scene_extent()
        if resize:
            self.fit_graph()

    def _moved(self, index: int, point: QPointF) -> None:
        self._positions[index] = point
        for (u, v), item in self._edges.items():
            if index in (u, v):
                item.position(self._positions[u], self._positions[v])
        self._update_scene_extent()

    def _update_scene_extent(self) -> None:
        self.scene().setSceneRect(self.scene().itemsBoundingRect().adjusted(-34, -34, 34, 34))

    def set_step(self, step: Step | None) -> None:
        marked = ()
        if step is not None:
            if isinstance(step.state, TraversalState):
                marked = step.state.order
            elif isinstance(step.state, DijkstraState):
                marked = step.state.fixed
        path = set(step.path) if step else set()
        for i, node in enumerate(self._nodes):
            focus = step is not None and i in step.focus_nodes
            color = FOCUS if focus else SELECTED if i in marked else EDGE
            node.setBrush(QBrush(QColor("#145078" if i in marked else "#1e293b")))
            node.setPen(QPen(QColor(color), 3 if focus else 2))
        for edge, item in self._edges.items():
            rejected = step is not None and edge in step.rejected
            if edge in path:
                color, width = PATH, 4
            elif step is not None and edge == step.focus_edge and not rejected:
                color, width = FOCUS, 3.5
            elif rejected:
                color, width = REJECTED, 1.4
            elif step is not None and edge in step.selected:
                color, width = SELECTED, 3
            else:
                color, width = EDGE, 1.5
            item.style(color, width, rejected)

    def zoom(self, factor: float) -> None:
        size = self.transform().m11() * factor
        if .08 <= size <= 7:
            self.scale(factor, factor)

    def fit_graph(self) -> None:
        if not self._nodes:
            return
        if not self.isVisible():
            self._pending_fit = True
            return
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        if self.transform().m11() > 1.5:
            self.resetTransform()
            self.centerOn(self.sceneRect().center())

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._pending_fit:
            self._pending_fit = False
            self.fit_graph()
