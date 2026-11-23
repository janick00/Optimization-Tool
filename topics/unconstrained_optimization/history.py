"""A shared read-only view of accepted iterations and attempted steps."""

from dataclasses import dataclass
from functools import cached_property

from topics.unconstrained_optimization.methods import LineSearchTrial, ParabolaSearchResult
from topics.unconstrained_optimization.models import IterationStep, OptimizationResult, SecondOrderDetails


@dataclass(frozen=True)
class IterationFrame:
    current: IterationStep
    following: IterationStep | None = None
    trials: tuple[LineSearchTrial, ...] = ()
    parabola: ParabolaSearchResult | None = None
    second_order: SecondOrderDetails | None = None
    failure: str = ""

    @property
    def number(self) -> int:
        return self.current.iteration + int(self.following is not None or bool(self.failure))

    @property
    def beta(self) -> float | None:
        if self.second_order and self.second_order.direction is not None:
            return self.second_order.beta
        return self.following.step_size if self.following else None

    @property
    def direction(self) -> tuple[float, ...] | None:
        if self.second_order:
            return self.second_order.direction
        if self.trials and self.current.gradient is not None:
            return tuple(-g for g in self.current.gradient)
        return None

    @cached_property
    def trial_points(self) -> tuple[LineSearchTrial, ...]:
        return (self.parabola.start, *self.trials) if self.parabola else self.trials

    @cached_property
    def chosen_trial(self) -> int | None:
        if self.parabola:
            return next((i for i, trial in enumerate(self.trial_points) if trial is self.parabola.selected), None)
        return next((i for i, trial in enumerate(self.trials) if trial.accepted), None)

    def trial_label(self, index: int) -> str:
        if self.parabola:
            return ("0 · f₀", "β · f₁", "2β · f₂", "β*")[index]
        return str(index + 1)

    def trial_decision(self, index: int) -> str:
        trial = self.trial_points[index]
        if index == self.chosen_trial:
            return "Selected" if self.parabola else "Accepted"
        if trial.value is None:
            return "Invalid"
        if self.parabola and index <= 2:
            return ("Reference", "Not selected", "Support")[index]
        return "Rejected"


def iteration_history(result: OptimizationResult) -> tuple[IterationFrame, ...]:
    """Expose each transition and a failed attempt, if present, without adding iterates."""

    frames = [
        IterationFrame(current, following, following.trials, following.parabola, following.second_order)
        for current, following in zip(result.steps, result.steps[1:])
    ]
    if result.failed_trials or result.failed_parabola or result.failed_second_order:
        parabola = result.failed_parabola
        frames.append(IterationFrame(
            result.steps[-1], trials=parabola.trials if parabola else result.failed_trials,
            parabola=parabola, second_order=result.failed_second_order, failure=result.message,
        ))
    return tuple(frames) or (IterationFrame(result.steps[0]),)
