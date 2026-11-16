"""Central registration of all topics shown in the navigation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtWidgets import QWidget

from topics.gradient_descent.page import GradientDescentPage
from topics.linear_programming.page import ProblemFormulationPage
from topics.newton_method.page import NewtonMethodPage
from topics.simplex_method.page import SimplexMethodPage
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
        identifier="gradient_descent",
        label="Gradient Descent",
        page_factory=GradientDescentPage,
    ),
    TopicDefinition(
        identifier="newton_method",
        label="Newton's Method",
        page_factory=NewtonMethodPage,
    ),
)

