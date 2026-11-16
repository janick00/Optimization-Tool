"""Global dark theme for the desktop application."""


DARK_STYLESHEET = """
QWidget {
    color: #e6edf3;
    background-color: #111827;
    font-size: 14px;
}

QMainWindow {
    background-color: #111827;
}

QFrame#navigationPanel {
    background-color: #172033;
    border-right: 1px solid #2b3952;
}

QLabel#applicationTitle {
    color: #f8fafc;
    font-size: 20px;
    font-weight: 700;
}

QLabel#applicationSubtitle {
    color: #91a4be;
    font-size: 12px;
}

QPushButton#navigationButton {
    color: #b8c4d6;
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 10px 12px;
    text-align: left;
}

QPushButton#navigationButton:hover {
    color: #f8fafc;
    background-color: #22304a;
}

QPushButton#navigationButton:checked {
    color: #ffffff;
    background-color: #2563eb;
    border-color: #3b82f6;
}

QGroupBox {
    color: #dce7f5;
    background-color: #172033;
    border: 1px solid #2b3952;
    border-radius: 8px;
    margin-top: 10px;
    padding: 18px 12px 12px 12px;
    font-weight: 600;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 5px;
}

QLineEdit, QPlainTextEdit, QTextBrowser, QComboBox {
    color: #e6edf3;
    background-color: #0f172a;
    border: 1px solid #334155;
    border-radius: 5px;
    padding: 6px 8px;
    selection-background-color: #2563eb;
}

QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus {
    border-color: #38bdf8;
}

QLineEdit:disabled {
    color: #71839d;
    background-color: #1b2639;
}

QComboBox QAbstractItemView {
    color: #e6edf3;
    background-color: #172033;
    selection-background-color: #2563eb;
}

QPushButton#primaryButton {
    color: #ffffff;
    background-color: #2563eb;
    border: 1px solid #3b82f6;
    border-radius: 5px;
    padding: 8px 18px;
    font-weight: 600;
}

QPushButton#primaryButton:hover {
    background-color: #1d4ed8;
}

QLabel#fieldHelp {
    color: #8193ad;
    font-size: 12px;
    font-weight: normal;
}

QLabel#sectionLabel {
    color: #9fb0c5;
    font-weight: 600;
}

QLabel#conversionStatus {
    color: #7dd3fc;
    font-weight: 600;
}

QLabel#conversionStatus[state="error"] {
    color: #fca5a5;
}

QLabel#conversionStatus[state="success"] {
    color: #86efac;
}

QTextBrowser#resultEditor, QTextBrowser#stepsEditor {
    color: #dbeafe;
    background-color: #0f172a;
    border: 1px solid #334155;
    border-radius: 5px;
    padding: 10px;
}

QTableWidget {
    color: #e6edf3;
    background-color: #0f172a;
    alternate-background-color: #141f33;
    border: 1px solid #334155;
    gridline-color: #2b3952;
    selection-background-color: #1d4ed8;
}

QHeaderView::section {
    color: #cbd5e1;
    background-color: #1e293b;
    border: none;
    border-right: 1px solid #334155;
    border-bottom: 1px solid #334155;
    padding: 5px;
    font-weight: 600;
}

QTextBrowser#optimizationDetails {
    color: #cbd5e1;
    background-color: #0f172a;
    border: 1px solid #334155;
    border-radius: 5px;
    padding: 8px;
}

QLabel#optimizationStatus {
    color: #bbf7d0;
    background-color: #193b2a;
    border: 1px solid #245c3b;
    border-radius: 4px;
    padding: 5px 8px;
    font-size: 12px;
    font-weight: 600;
}

QLabel#optimizationStatus[state="warning"] {
    color: #fde68a;
    background-color: #493b1a;
    border-color: #766022;
}

QLabel#optimizationStatus[state="error"] {
    color: #fecaca;
    background-color: #4a1f2a;
    border-color: #7f3045;
}

QFrame#contentPanel {
    background-color: #111827;
}

QLabel#topicTitle {
    color: #f8fafc;
    font-size: 28px;
    font-weight: 700;
}

QLabel#topicDescription {
    color: #9fb0c5;
    font-size: 15px;
}

QLabel#placeholderMessage {
    color: #71839d;
    background-color: #172033;
    border: 1px solid #2b3952;
    border-radius: 8px;
    padding: 18px;
}
"""

