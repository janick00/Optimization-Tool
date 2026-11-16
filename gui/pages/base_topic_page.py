"""Base widget shared by the initial topic pages."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class EmptyTopicPage(QWidget):
    """Display the stable phase-one shell for a topic without calculations."""

    def __init__(self, topic_title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.topic_title = topic_title
        self._build_layout()

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(48, 42, 48, 42)
        layout.setSpacing(14)

        title_label = QLabel(self.topic_title)
        title_label.setObjectName("topicTitle")

        description_label = QLabel(
            "This topic page is ready for its solver, inputs, and visualizations."
        )
        description_label.setObjectName("topicDescription")
        description_label.setWordWrap(True)

        placeholder_label = QLabel(
            "Phase 1: no mathematical calculations are implemented yet."
        )
        placeholder_label.setObjectName("placeholderMessage")
        placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder_label.setMinimumHeight(100)
        placeholder_label.setWordWrap(True)

        layout.addWidget(title_label)
        layout.addWidget(description_label)
        layout.addSpacing(18)
        layout.addWidget(placeholder_label)
        layout.addStretch()

