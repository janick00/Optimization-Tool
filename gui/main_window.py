"""Main application window and topic navigation."""

from __future__ import annotations

from PySide6.QtCore import QSize
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from gui.topic_registry import TOPIC_REGISTRY, TopicDefinition


class MainWindow(QMainWindow):
    """Application shell with persistent navigation and a stacked content area."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._buttons_by_topic_id: dict[str, QPushButton] = {}
        self._topic_indices: dict[str, int] = {}
        self._current_topic_id: str | None = None

        self.setWindowTitle("Optimization Solver")
        self.setMinimumSize(QSize(960, 600))
        self.resize(1180, 720)
        self._build_ui()
        self._select_initial_topic()

    @property
    def current_topic_id(self) -> str | None:
        """Return the identifier of the currently selected topic."""

        return self._current_topic_id

    @property
    def topic_button_ids(self) -> tuple[str, ...]:
        """Return topic identifiers in their navigation order."""

        return tuple(self._buttons_by_topic_id)

    def _build_ui(self) -> None:
        central_widget = QWidget()
        central_layout = QHBoxLayout(central_widget)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)

        navigation_panel = self._create_navigation_panel()
        content_panel = self._create_content_panel()

        central_layout.addWidget(navigation_panel)
        central_layout.addWidget(content_panel, stretch=1)
        self.setCentralWidget(central_widget)

    def _create_navigation_panel(self) -> QFrame:
        navigation_panel = QFrame()
        navigation_panel.setObjectName("navigationPanel")
        navigation_panel.setFixedWidth(250)

        layout = QVBoxLayout(navigation_panel)
        layout.setContentsMargins(18, 28, 18, 24)
        layout.setSpacing(8)

        title_label = QLabel("Optimization Solver")
        title_label.setObjectName("applicationTitle")
        title_label.setWordWrap(True)

        subtitle_label = QLabel("Optimization topics")
        subtitle_label.setObjectName("applicationSubtitle")

        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)
        layout.addSpacing(22)

        self._navigation_button_group = QButtonGroup(self)
        self._navigation_button_group.setExclusive(True)

        for topic in TOPIC_REGISTRY:
            button = self._create_navigation_button(topic)
            self._navigation_button_group.addButton(button)
            self._buttons_by_topic_id[topic.identifier] = button
            layout.addWidget(button)

        layout.addStretch()
        return navigation_panel

    def _create_navigation_button(self, topic: TopicDefinition) -> QPushButton:
        button = QPushButton(topic.label)
        button.setObjectName("navigationButton")
        button.setCheckable(True)
        button.setAutoDefault(False)
        button.setMinimumHeight(42)
        button.clicked.connect(lambda _checked=False, topic_id=topic.identifier: self.show_topic(topic_id))
        return button

    def _create_content_panel(self) -> QFrame:
        content_panel = QFrame()
        content_panel.setObjectName("contentPanel")

        layout = QVBoxLayout(content_panel)
        layout.setContentsMargins(0, 0, 0, 0)

        self._page_stack = QStackedWidget()
        for index, topic in enumerate(TOPIC_REGISTRY):
            self._topic_indices[topic.identifier] = index
            self._page_stack.addWidget(topic.page_factory())

        layout.addWidget(self._page_stack)
        return content_panel

    def _select_initial_topic(self) -> None:
        if TOPIC_REGISTRY:
            self.show_topic(TOPIC_REGISTRY[0].identifier)

    def show_topic(self, topic_id: str) -> None:
        """Show a registered topic and highlight its navigation button."""

        if topic_id not in self._topic_indices:
            raise KeyError(f"Unknown topic identifier: {topic_id}")

        self._page_stack.setCurrentIndex(self._topic_indices[topic_id])
        self._buttons_by_topic_id[topic_id].setChecked(True)
        self._current_topic_id = topic_id

