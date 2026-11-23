"""Central registration of all topics shown in the navigation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtWidgets import QWidget

from topics.integer_programming.page import IntegerProgrammingPage
from topics.linear_programming.page import ProblemFormulationPage
from topics.network_optimization.page import NetworkOptimizationPage
from topics.simplex_method.page import SimplexMethodPage
from topics.unconstrained_optimization.aitken_page import AitkenPage
from topics.unconstrained_optimization.page import UnconstrainedOptimizationPage


PageFactory = Callable[[], QWidget]


@dataclass(frozen=True)
class TopicDefinition:
    """Configuration required to expose one topic in the application."""

    identifier: str
    label: str
    page_factory: PageFactory


# Add a new topic by adding one TopicDefinition here. The main window does not
# need to be changed because it builds both navigation and pages from this list.
TOPIC_REGISTRY: tuple[TopicDefinition, ...] = (
    TopicDefinition(
        identifier="linear_programming",
        label="Problem Formulation",
        page_factory=ProblemFormulationPage,
    ),
    TopicDefinition(
        identifier="unconstrained_optimization",
        label="Unconstrained Opt.",
        page_factory=UnconstrainedOptimizationPage,
    ),
    TopicDefinition(
        identifier="simplex_method",
        label="Simplex Method",
        page_factory=SimplexMethodPage,
    ),
    TopicDefinition(
        identifier="integer_programming",
        label="Integer Linear Programming",
        page_factory=IntegerProgrammingPage,
    ),
    TopicDefinition(
        identifier="aitken",
        label="Aitken Acceleration",
        page_factory=AitkenPage,
    ),
    TopicDefinition(
        identifier="network_optimization",
        label="Graphen & Netzwerke",
        page_factory=NetworkOptimizationPage,
    ),
)

