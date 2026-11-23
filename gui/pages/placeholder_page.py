"""Shared page for planned topics."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class PlaceholderPage(QWidget):
    """Show the title and status of a planned topic."""

    def __init__(self, topic_title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(48, 42, 48, 42)
        layout.setSpacing(14)

        title_label = QLabel(topic_title)
        title_label.setObjectName("topicTitle")

        placeholder_label = QLabel("This topic is planned.")
        placeholder_label.setObjectName("placeholderMessage")
        placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder_label.setMinimumHeight(100)
        placeholder_label.setWordWrap(True)

        layout.addWidget(title_label)
        layout.addSpacing(18)
        layout.addWidget(placeholder_label)
        layout.addStretch()

