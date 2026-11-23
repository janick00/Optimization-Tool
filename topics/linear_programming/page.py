"""Structured problem-formulation editor for linear programs."""

from __future__ import annotations

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from topics.linear_programming.form_conversion import (
    convert_problem,
    form_defaults,
)
from topics.linear_programming.formatting import (
    DisplayNotation,
    format_conversion_steps_html,
    format_linear_program_html,
)
from topics.linear_programming.models import ConversionResult, ObjectiveSense, ProblemForm
from topics.linear_programming.problem_input import build_linear_program


class ProblemFormulationPage(QWidget):
    """Edit an LP and convert it between the four course representations."""

    _RELATION_OPTIONS = (("<=", "<="), ("=", "="), (">=", ">="))
    _DOMAIN_OPTIONS = (("x >= 0", ">=0"), ("x free", "free"), ("x <= 0", "<=0"))

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._general_relation_values: list[str] = []
        self._general_domain_values: list[str] = []
        self._objective_edits: list[QLineEdit] = []
        self._constraint_coefficient_edits: list[list[QLineEdit]] = []
        self._right_hand_side_edits: list[QLineEdit] = []
        self._relation_combos: list[QComboBox] = []
        self._domain_combos: list[QComboBox] = []
        self._build_ui()
        self._load_initial_example()
        self._rebuild_dynamic_fields()
        self._update_form_fields()
        self._convert()

    def _build_ui(self) -> None:
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(34, 28, 34, 28)
        outer_layout.setSpacing(12)

        title_label = QLabel("Problem Formulation")
        title_label.setObjectName("topicTitle")
        subtitle_label = QLabel(
            "Enter a linear program and convert it between the forms from Chapter 3.1."
        )
        subtitle_label.setObjectName("topicDescription")
        subtitle_label.setWordWrap(True)
        outer_layout.addWidget(title_label)
        outer_layout.addWidget(subtitle_label)

        form_selection_layout = QGridLayout()
        form_selection_layout.setHorizontalSpacing(12)
        form_selection_layout.addWidget(QLabel("Input form"), 0, 0)
        self._source_form_combo = self._create_form_combo()
        form_selection_layout.addWidget(self._source_form_combo, 0, 1)
        form_selection_layout.addWidget(QLabel("Convert to"), 0, 2)
        self._target_form_combo = self._create_form_combo()
        form_selection_layout.addWidget(self._target_form_combo, 0, 3)
        form_selection_layout.addWidget(QLabel("Display notation"), 1, 0)
        self._notation_combo = QComboBox()
        for notation in DisplayNotation:
            self._notation_combo.addItem(notation.label, notation.value)
        form_selection_layout.addWidget(self._notation_combo, 1, 1, 1, 3)
        form_selection_layout.setColumnStretch(1, 1)
        form_selection_layout.setColumnStretch(3, 1)
        outer_layout.addLayout(form_selection_layout)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(14)

        input_group = self._create_input_group()
        input_scroll = QScrollArea()
        input_scroll.setObjectName("inputScrollArea")
        input_scroll.setFrameShape(QFrame.Shape.NoFrame)
        input_scroll.setWidgetResizable(True)
        input_scroll.setWidget(input_group)
        input_scroll.setMinimumWidth(500)
        content_layout.addWidget(input_scroll, stretch=1)
        content_layout.addWidget(self._create_output_group(), stretch=1)
        outer_layout.addLayout(content_layout, stretch=1)

        self._source_form_combo.currentIndexChanged.connect(self._handle_form_change)
        self._target_form_combo.currentIndexChanged.connect(self._convert)
        self._notation_combo.currentIndexChanged.connect(self._convert)
        self._objective_sense_combo.currentIndexChanged.connect(self._handle_form_change)
        self._variable_count_combo.currentIndexChanged.connect(self._handle_count_change)
        self._constraint_count_combo.currentIndexChanged.connect(self._handle_count_change)

    @staticmethod
    def _create_form_combo() -> QComboBox:
        combo = QComboBox()
        for form in ProblemForm:
            combo.addItem(form.label, form.value)
        return combo

    def _create_input_group(self) -> QGroupBox:
        group = QGroupBox("Problem input")
        layout = QVBoxLayout(group)
        layout.setSpacing(12)
        self._editor_fields = QWidget()
        fields = QVBoxLayout(self._editor_fields)
        fields.setContentsMargins(0, 0, 0, 0)
        fields.setSpacing(12)
        layout.addWidget(self._editor_fields)

        settings_layout = QGridLayout()
        settings_layout.setHorizontalSpacing(10)
        settings_layout.setVerticalSpacing(8)

        settings_layout.addWidget(QLabel("Objective"), 0, 0)
        self._objective_sense_combo = QComboBox()
        for sense in ObjectiveSense:
            self._objective_sense_combo.addItem(sense.label, sense.value)
        settings_layout.addWidget(self._objective_sense_combo, 0, 1)

        settings_layout.addWidget(QLabel("Number of variables"), 1, 0)
        self._variable_count_combo = self._create_count_combo()
        settings_layout.addWidget(self._variable_count_combo, 1, 1)

        settings_layout.addWidget(QLabel("Number of constraints"), 2, 0)
        self._constraint_count_combo = self._create_count_combo()
        settings_layout.addWidget(self._constraint_count_combo, 2, 1)
        settings_layout.setColumnStretch(1, 1)
        fields.addLayout(settings_layout)

        objective_label = QLabel("Objective coefficients (enter numbers before the variables)")
        objective_label.setObjectName("sectionLabel")
        objective_label.setWordWrap(True)
        fields.addWidget(objective_label)
        self._objective_row_layout = QHBoxLayout()
        self._objective_row_layout.setSpacing(4)
        fields.addLayout(self._objective_row_layout)

        constraints_label = QLabel(
            "Constraints (enter coefficients; choose the relation and enter b)"
        )
        constraints_label.setObjectName("sectionLabel")
        constraints_label.setWordWrap(True)
        fields.addWidget(constraints_label)
        self._constraints_layout = QVBoxLayout()
        self._constraints_layout.setSpacing(6)
        fields.addLayout(self._constraints_layout)

        domains_label = QLabel("Variable domains")
        domains_label.setObjectName("sectionLabel")
        fields.addWidget(domains_label)
        self._domains_layout = QHBoxLayout()
        self._domains_layout.setSpacing(10)
        fields.addLayout(self._domains_layout)
        self._add_input_sections(layout)

        self._variable_hint = QLabel(
            "The variable names x1, x2, ... and constraint labels are generated "
            "from the selected counts."
        )
        self._variable_hint.setObjectName("fieldHelp")
        self._variable_hint.setWordWrap(True)
        layout.addWidget(self._variable_hint)

        self._input_button_layout = button_layout = QHBoxLayout()
        self._convert_button = QPushButton("Convert")
        self._convert_button.setObjectName("primaryButton")
        self._convert_button.clicked.connect(self._convert)
        button_layout.addWidget(self._convert_button)
        button_layout.addStretch()
        layout.addLayout(button_layout)
        layout.addStretch()
        return group

    def _add_input_sections(self, layout: QVBoxLayout) -> None:
        """Optional sections after the shared LP input fields."""

    @staticmethod
    def _create_count_combo() -> QComboBox:
        combo = QComboBox()
        for count in range(1, 21):
            combo.addItem(str(count), count)
        combo.setMinimumWidth(90)
        return combo

    def _create_output_group(self) -> QGroupBox:
        group = QGroupBox("Converted problem")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        self._status_label = QLabel()
        self._status_label.setObjectName("conversionStatus")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        self._output_edit = QTextBrowser()
        self._output_edit.setObjectName("resultEditor")
        self._output_edit.setReadOnly(True)
        self._output_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._set_document_font(self._output_edit)
        layout.addWidget(self._output_edit, stretch=3)

        steps_label = QLabel("Transformation steps")
        steps_label.setObjectName("sectionLabel")
        layout.addWidget(steps_label)

        self._steps_edit = QTextBrowser()
        self._steps_edit.setObjectName("stepsEditor")
        self._steps_edit.setReadOnly(True)
        self._steps_edit.setMinimumHeight(100)
        self._set_document_font(self._steps_edit)
        layout.addWidget(self._steps_edit, stretch=2)
        return group

    @staticmethod
    def _set_document_font(editor: QTextBrowser) -> None:
        font = QFont("DejaVu Serif")
        font.setStyleHint(QFont.StyleHint.Serif)
        font.setPointSize(10)
        editor.setFont(font)

    def _load_initial_example(self) -> None:
        self._load_default_inputs()
        self._source_form_combo.setCurrentIndex(self._form_index(ProblemForm.GENERAL))
        self._target_form_combo.setCurrentIndex(self._form_index(ProblemForm.STANDARD))

    def _load_default_inputs(self) -> None:
        self._objective_sense_combo.setCurrentIndex(0)
        variable_signals_blocked = self._variable_count_combo.blockSignals(True)
        constraint_signals_blocked = self._constraint_count_combo.blockSignals(True)
        self._variable_count_combo.setCurrentIndex(self._count_index(2))
        self._constraint_count_combo.setCurrentIndex(self._count_index(2))
        self._variable_count_combo.blockSignals(variable_signals_blocked)
        self._constraint_count_combo.blockSignals(constraint_signals_blocked)
        self._general_relation_values = ["<=", "<="]
        self._general_domain_values = [">=0", ">=0"]

    @staticmethod
    def _count_index(count: int) -> int:
        return count - 1

    @staticmethod
    def _form_index(form: ProblemForm) -> int:
        return list(ProblemForm).index(form)

    def _handle_count_change(self) -> None:
        self._store_general_selections()
        self._rebuild_dynamic_fields()
        self._update_form_fields()
        self._convert()

    def _handle_form_change(self) -> None:
        self._update_form_fields()
        self._convert()

    def _rebuild_dynamic_fields(self) -> None:
        old_objective = [edit.text() for edit in self._objective_edits]
        old_coefficients = [
            [edit.text() for edit in row] for row in self._constraint_coefficient_edits
        ]
        old_right_hand_sides = [edit.text() for edit in self._right_hand_side_edits]
        old_relations = [str(combo.currentData()) for combo in self._relation_combos]
        old_domains = [str(combo.currentData()) for combo in self._domain_combos]

        self._clear_layout(self._objective_row_layout)
        self._clear_layout(self._constraints_layout)
        self._clear_layout(self._domains_layout)
        self._objective_edits = []
        self._constraint_coefficient_edits = []
        self._right_hand_side_edits = []
        self._relation_combos = []
        self._domain_combos = []

        variable_count = int(self._variable_count_combo.currentData())
        constraint_count = int(self._constraint_count_combo.currentData())
        self._build_objective_fields(variable_count, old_objective)
        self._build_constraint_fields(
            variable_count, constraint_count, old_coefficients, old_right_hand_sides, old_relations
        )
        self._build_domain_fields(variable_count, old_domains)

    def _build_objective_fields(self, variable_count: int, old_objective: list[str]) -> None:
        for variable_index in range(variable_count):
            if variable_index:
                self._objective_row_layout.addWidget(QLabel("+"))
            coefficient_edit = self._create_number_edit(
                old_objective[variable_index]
                if variable_index < len(old_objective)
                else ("3" if variable_index == 0 else "2" if variable_index == 1 else "0")
            )
            self._objective_edits.append(coefficient_edit)
            self._objective_row_layout.addWidget(coefficient_edit)
            self._objective_row_layout.addWidget(QLabel(f"x{variable_index + 1}"))
        self._objective_row_layout.addStretch()

    def _build_constraint_fields(
        self,
        variable_count: int,
        constraint_count: int,
        old_coefficients: list[list[str]],
        old_right_hand_sides: list[str],
        old_relations: list[str],
    ) -> None:
        for constraint_index in range(constraint_count):
            row_layout = QHBoxLayout()
            row_layout.setSpacing(4)
            row_layout.addWidget(QLabel(f"C{constraint_index + 1}:"))
            coefficient_edits: list[QLineEdit] = []
            old_row = (
                old_coefficients[constraint_index]
                if constraint_index < len(old_coefficients)
                else []
            )
            for variable_index in range(variable_count):
                if variable_index:
                    row_layout.addWidget(QLabel("+"))
                if constraint_index == 0:
                    default_coefficient = "2" if variable_index == 1 else "1"
                else:
                    default_coefficient = "2" if variable_index == 0 else "1"
                coefficient_edit = self._create_number_edit(
                    old_row[variable_index]
                    if variable_index < len(old_row)
                    else default_coefficient
                )
                coefficient_edits.append(coefficient_edit)
                row_layout.addWidget(coefficient_edit)
                row_layout.addWidget(QLabel(f"x{variable_index + 1}"))

            relation_combo = self._create_relation_combo(
                old_relations[constraint_index]
                if constraint_index < len(old_relations)
                else ("<=" if constraint_index < 2 else "=")
            )
            relation_combo.currentIndexChanged.connect(self._convert)
            self._relation_combos.append(relation_combo)
            row_layout.addWidget(relation_combo)

            row_layout.addWidget(QLabel(f"b{constraint_index + 1}"))
            rhs_edit = self._create_number_edit(
                old_right_hand_sides[constraint_index]
                if constraint_index < len(old_right_hand_sides)
                else ("8" if constraint_index < 2 else "0")
            )
            rhs_edit.setToolTip("Right-hand side b")
            self._right_hand_side_edits.append(rhs_edit)
            row_layout.addWidget(rhs_edit)
            row_layout.addStretch()
            self._constraint_coefficient_edits.append(coefficient_edits)
            self._constraints_layout.addLayout(row_layout)

    def _build_domain_fields(self, variable_count: int, old_domains: list[str]) -> None:
        for variable_index in range(variable_count):
            domain_combo = self._create_domain_combo(
                old_domains[variable_index]
                if variable_index < len(old_domains)
                else (self._general_domain_values[variable_index]
                      if variable_index < len(self._general_domain_values)
                      else ">=0")
            )
            domain_combo.currentIndexChanged.connect(self._convert)
            self._domain_combos.append(domain_combo)
            domain_layout = QHBoxLayout()
            domain_layout.setSpacing(4)
            domain_layout.addWidget(QLabel(f"x{variable_index + 1}"))
            domain_layout.addWidget(domain_combo)
            self._domains_layout.addLayout(domain_layout)
        self._domains_layout.addStretch()

    @staticmethod
    def _create_number_edit(initial_value: str) -> QLineEdit:
        edit = QLineEdit(str(initial_value))
        edit.setFixedWidth(62)
        edit.setAlignment(Qt.AlignmentFlag.AlignRight)
        edit.setPlaceholderText("0")
        return edit

    def _create_relation_combo(self, current_value: str) -> QComboBox:
        combo = QComboBox()
        for label, value in self._RELATION_OPTIONS:
            combo.addItem(label, value)
        self._set_combo_value(combo, current_value)
        combo.setFixedWidth(62)
        combo.setToolTip("Constraint relation")
        return combo

    def _create_domain_combo(self, current_value: str) -> QComboBox:
        combo = QComboBox()
        for label, value in self._DOMAIN_OPTIONS:
            combo.addItem(label, value)
        self._set_combo_value(combo, current_value)
        combo.setFixedWidth(88)
        return combo

    @staticmethod
    def _set_combo_value(combo: QComboBox, value: str | None) -> None:
        if value is None:
            return
        index = combo.findData(value)
        if index >= 0:
            with QSignalBlocker(combo):
                combo.setCurrentIndex(index)

    @staticmethod
    def _clear_layout(layout: QHBoxLayout | QVBoxLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.hide()
                widget.deleteLater()
            elif child_layout is not None:
                ProblemFormulationPage._clear_layout(child_layout)  # type: ignore[arg-type]

    def _store_general_selections(self) -> None:
        if not self._relation_combos or not self._domain_combos:
            return
        if not self._relation_combos[0].isEnabled():
            return
        self._general_relation_values = [
            str(combo.currentData()) for combo in self._relation_combos
        ]
        self._general_domain_values = [
            str(combo.currentData()) for combo in self._domain_combos
        ]

    def _update_form_fields(self) -> None:
        source_form = ProblemForm(self._source_form_combo.currentData())
        sense = ObjectiveSense(self._objective_sense_combo.currentData())
        defaults = form_defaults(source_form, sense)
        relation = defaults[0].value if defaults is not None else None
        domain = defaults[1].value if defaults is not None else None
        self._update_form_combos(
            self._relation_combos, self._general_relation_values, relation
        )
        self._update_form_combos(
            self._domain_combos, self._general_domain_values, domain
        )

    def _update_form_combos(
        self, combos: list[QComboBox], general_values: list[str], fixed_value: str | None
    ) -> None:
        for index, combo in enumerate(combos):
            combo.setEnabled(fixed_value is None)
            value = fixed_value
            if value is None and index < len(general_values):
                value = general_values[index]
            self._set_combo_value(combo, value)

    def _build_input_text(self) -> tuple[str, str, str, str, str]:
        objective_text = ", ".join(edit.text().strip() for edit in self._objective_edits)
        matrix_text = "\n".join(
            ", ".join(edit.text().strip() for edit in row)
            for row in self._constraint_coefficient_edits
        )
        right_hand_side_text = ", ".join(edit.text().strip() for edit in self._right_hand_side_edits)
        relation_text = ", ".join(str(combo.currentData()) for combo in self._relation_combos)
        domain_text = ", ".join(str(combo.currentData()) for combo in self._domain_combos)
        return objective_text, matrix_text, right_hand_side_text, relation_text, domain_text

    def _convert(self) -> None:
        self._store_general_selections()
        try:
            source_form = ProblemForm(self._source_form_combo.currentData())
            target_form = ProblemForm(self._target_form_combo.currentData())
            notation = DisplayNotation(self._notation_combo.currentData())
            sense = ObjectiveSense(self._objective_sense_combo.currentData())
            objective_text, matrix_text, rhs_text, relation_text, domain_text = self._build_input_text()
            problem = build_linear_program(
                source_form=source_form,
                objective_sense=sense,
                objective_text=objective_text,
                matrix_text=matrix_text,
                right_hand_side_text=rhs_text,
                relation_text=relation_text,
                domain_text=domain_text,
            )
            result = convert_problem(problem, target_form, source_form)
        except ValueError as error:
            self._status_label.setText(f"Input error: {error}")
            self._status_label.setProperty("state", "error")
            self._output_edit.clear()
            self._steps_edit.clear()
            self._refresh_style(self._status_label)
            return

        self._show_conversion(result, notation)

    def _show_conversion(self, result: ConversionResult, notation: DisplayNotation) -> None:
        self._status_label.setText(f"Converted successfully to {result.target_form.label}.")
        self._status_label.setProperty("state", "success")
        self._output_edit.setHtml(
            format_linear_program_html(result.problem, result.target_form, notation)
        )
        self._steps_edit.setHtml(format_conversion_steps_html(result))
        self._refresh_style(self._status_label)

    @staticmethod
    def _refresh_style(widget: QWidget) -> None:
        widget.style().unpolish(widget)
        widget.style().polish(widget)
