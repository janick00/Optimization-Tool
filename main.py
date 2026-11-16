"""Application entry point for the Optimization Solver."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from PySide6.QtWidgets import QApplication

from gui.main_window import MainWindow
from gui.styles import DARK_STYLESHEET


def create_application(arguments: Sequence[str] | None = None) -> QApplication:
    """Create or reuse the Qt application and apply the global appearance."""

    application = QApplication.instance()
    if application is None:
        application = QApplication(list(arguments) if arguments is not None else sys.argv)

    application.setApplicationName("Optimization Solver")
    application.setOrganizationName("Optimization Solver")
    application.setStyle("Fusion")
    application.setStyleSheet(DARK_STYLESHEET)
    return application


def main() -> int:
    """Start the desktop application."""

    application = create_application()
    window = MainWindow()
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())

