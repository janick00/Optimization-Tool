"""Run the numerical optimization steps and apply their stopping rules."""

from __future__ import annotations

from dataclasses import replace
from math import hypot, isfinite
from typing import Sequence

import numpy as np

from topics.unconstrained_optimization.methods import (
    LineSearchTrial,
    broyden_step,
    newton_step,
    parabola_fitting,
    successive_halving,
)
from topics.unconstrained_optimization.methods.newton_methods import (
    broyden_update_details,
    classify_stationary_hessian,
    initial_inverse_hessian,
    matrix_values,
)
from topics.unconstrained_optimization.models import (
    IterationStep,
    OptimizationMethod,
    OptimizationResult,
    OptimizationSettings,
    SecondOrderDetails,
)
from topics.unconstrained_optimization.objective import ParsedFunction, validate_point


def optimize(
    function: ParsedFunction,
    method: OptimizationMethod | str,
    start: Sequence[float],
    settings: OptimizationSettings | None = None,
) -> OptimizationResult:
    """Run one selected unconstrained optimization method."""

    method = OptimizationMethod(method)
    settings = settings or OptimizationSettings.for_method(method)
    point = validate_point(start, function.dimension)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        current_value = function.evaluate(point)
        gradient = function.gradient_at(point)
    gradient_norm = hypot(*gradient)
    initial_step = IterationStep(
        iteration=0,
        point=tuple(point),
        value=current_value,
        gradient=tuple(gradient),
        gradient_norm=gradient_norm,
        step_size=0.0,
        note="Initial point",
    )
    if method is OptimizationMethod.GRADIENT_HALVING:
        result = _run_successive_halving(function, initial_step, point, gradient, settings)
    elif method in (OptimizationMethod.NEWTON, OptimizationMethod.BROYDEN):
        result = _run_second_order(function, method, initial_step, point, gradient, settings)
    else:
        result = _run_parabola(function, initial_step, point, gradient, settings)
    return replace(result, settings=settings)


def _run_parabola(
    function: ParsedFunction, initial_step: IterationStep, point: np.ndarray,
    gradient: np.ndarray, settings: OptimizationSettings,
) -> OptimizationResult:
    method = OptimizationMethod.GRADIENT_PARABOLA
    steps = [initial_step]
    current_value = initial_step.value
    if initial_step.gradient_norm < settings.tolerance:
        return OptimizationResult(function, method, tuple(steps), True, "Initial point is stationary.")

    converged = False
    message = "Maximum number of iterations reached."
    for iteration in range(1, settings.max_iterations + 1):
        try:
            parabola = parabola_fitting(
                function.evaluate, point, gradient, settings.initial_step, base_value=current_value
            )
            selected = parabola.selected
            if selected is None or selected.value is None:
                return OptimizationResult(
                    function, method, tuple(steps), False,
                    f"Stopped at iteration {iteration}. {parabola.failure_reason}",
                    failed_parabola=parabola,
                )
            next_point = np.asarray(selected.point, dtype=float)
            next_value, step_size = selected.value, selected.step_size
            note = f"Selected β={step_size:.6g}."
        except (ValueError, TypeError, ArithmeticError, np.linalg.LinAlgError) as error:
            message = f"Stopped at iteration {iteration}: {error}"
            break

        try:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                next_gradient = function.gradient_at(next_point)
        except (ValueError, TypeError, ArithmeticError, np.linalg.LinAlgError) as error:
            message = f"Stopped at iteration {iteration}: {error}"
            steps.append(IterationStep(
                iteration, tuple(next_point), next_value, None, None, step_size,
                note, trials=parabola.trials, parabola=parabola,
            ))
            break

        gradient_norm = hypot(*next_gradient)
        steps.append(IterationStep(
            iteration=iteration,
            point=tuple(next_point),
            value=next_value,
            gradient=tuple(next_gradient),
            gradient_norm=gradient_norm,
            step_size=step_size,
            note=note,
            trials=parabola.trials,
            parabola=parabola,
        ))

        if gradient_norm < settings.tolerance:
            converged, message = True, "Converged: gradient norm is below the tolerance."
            break

        point = next_point
        current_value = next_value
        gradient = next_gradient

    return OptimizationResult(function, method, tuple(steps), converged, message)


def _run_successive_halving(
    function: ParsedFunction,
    start: IterationStep,
    point: np.ndarray,
    gradient: np.ndarray,
    settings: OptimizationSettings,
) -> OptimizationResult:
    steps = [start]
    first_convergence = 0 if hypot(*gradient) < settings.tolerance else None
    failed_trials: tuple[LineSearchTrial, ...] = ()
    completed = False
    beta = settings.initial_step

    for iteration in range(1, settings.max_iterations + 1):
        if not np.any(gradient):
            message = "Stationary point reached"
            break

        search = successive_halving(
            function.evaluate, point, gradient, beta,
            halving_factor=settings.halving_factor,
            max_halvings=settings.max_halvings,
            base_value=steps[-1].value,
        )
        selected = search.selected
        if selected is None or selected.value is None:
            failed_trials = search.trials
            message = f"No improving step was found at iteration {iteration}. {search.failure_reason}"
            break

        point = np.asarray(selected.point, dtype=float)
        beta = selected.step_size
        note = f"Step accepted at β={beta:.6g}."
        try:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                gradient = function.gradient_at(point)
        except (ValueError, TypeError, ArithmeticError) as error:
            steps.append(IterationStep(
                iteration, selected.point, selected.value, None, None, beta, note, search.trials
            ))
            message = f"Stopped after iteration {iteration}: {error}"
            break

        gradient_norm = hypot(*gradient)
        steps.append(IterationStep(
            iteration, selected.point, selected.value, tuple(gradient), gradient_norm,
            beta, note, search.trials,
        ))
        if first_convergence is None and gradient_norm < settings.tolerance:
            first_convergence = iteration
    else:
        completed = True
        message = "Done" if np.any(gradient) else "Done. Stationary point reached"

    return OptimizationResult(
        function, OptimizationMethod.GRADIENT_HALVING, tuple(steps),
        first_convergence is not None, message, failed_trials=failed_trials,
        first_convergence_iteration=first_convergence,
        completed_requested_iterations=completed,
    )


def _run_second_order(
    function: ParsedFunction,
    method: OptimizationMethod,
    start: IterationStep,
    point: np.ndarray,
    gradient: np.ndarray,
    settings: OptimizationSettings,
) -> OptimizationResult:
    steps = [start]
    start_norm = hypot(*gradient)
    if start_norm < settings.tolerance:
        return _stationary_result(function, method, steps)
    if not isfinite(start_norm):
        return OptimizationResult(function, method, tuple(steps), False, "Initial gradient norm is not finite.")

    inverse_hessian = None
    initial_hessian = None
    if method is OptimizationMethod.BROYDEN:
        try:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                hessian = function.hessian_at(point)
                initial_hessian = matrix_values(hessian)
                inverse_hessian = initial_inverse_hessian(hessian)
        except (ValueError, TypeError, ArithmeticError, np.linalg.LinAlgError) as error:
            return OptimizationResult(
                function, method, tuple(steps), False, f"Stopped before the first iteration: {error}",
                initial_hessian=initial_hessian, failed_second_order=SecondOrderDetails(None),
            )

    message = "Maximum number of iterations reached."
    failed = None
    for iteration in range(1, settings.max_iterations + 1):
        matrix = direction = next_point = value = None
        try:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                if inverse_hessian is None:
                    matrix = function.hessian_at(point)
                    direction = newton_step(gradient, matrix)
                    note = "Newton step using the exact Hessian."
                else:
                    matrix = inverse_hessian
                    direction = broyden_step(gradient, matrix)
                    note = "Broyden step using the current inverse-Hessian approximation."

                next_point = point + direction
                if not np.all(np.isfinite(next_point)):
                    raise ArithmeticError("The new point is not finite.")
                if np.array_equal(next_point, point):
                    raise ArithmeticError("The point did not change at floating-point precision.")

                value = function.evaluate(next_point)
                next_gradient = function.gradient_at(next_point)
                displacement = next_point - point
                gradient_change = next_gradient - gradient if inverse_hessian is not None else None
            gradient_norm = hypot(*next_gradient)
            step_size = hypot(*displacement)
            if not isfinite(gradient_norm) or not isfinite(step_size):
                raise ArithmeticError("The gradient norm or step size is not finite.")
        except (ValueError, TypeError, ArithmeticError, np.linalg.LinAlgError) as error:
            message = f"Stopped at iteration {iteration}: {error}"
            failed = SecondOrderDetails(
                matrix_values(matrix) if matrix is not None else None,
                direction=tuple(direction) if direction is not None else None,
                candidate=tuple(next_point) if next_point is not None else None,
                candidate_value=value,
            )
            break

        if method is OptimizationMethod.NEWTON and value > steps[-1].value:
            note += " The full Newton step increased the function value."
        details = SecondOrderDetails(
            matrix_values(matrix),
            tuple(displacement),
            tuple(gradient_change) if gradient_change is not None else None,
            direction=tuple(direction),
            candidate=tuple(next_point),
            candidate_value=value,
        )
        reached = gradient_norm < settings.tolerance
        update_failed = False
        if gradient_change is not None:
            if reached:
                details = replace(details, update_note="Stopped at the gradient tolerance.")
            elif iteration == settings.max_iterations:
                details = replace(details, update_note="Stopped at the iteration limit.")
            else:
                try:
                    inverse_hessian, update = broyden_update_details(matrix, displacement, gradient_change)
                    details = replace(details, update=update)
                except ArithmeticError as error:
                    message = f"Stopped after iteration {iteration}: {error}"
                    details = replace(details, update_note=str(error))
                    update_failed = True
        steps.append(IterationStep(
            iteration, tuple(next_point), value, tuple(next_gradient), gradient_norm,
            step_size, note, second_order=details,
        ))
        if reached:
            return replace(_stationary_result(function, method, steps), initial_hessian=initial_hessian)
        if update_failed:
            break
        point, gradient = next_point, next_gradient

    return OptimizationResult(function, method, tuple(steps), False, message,
                              failed_second_order=failed, initial_hessian=initial_hessian)


def _stationary_result(
    function: ParsedFunction,
    method: OptimizationMethod,
    steps: list[IterationStep],
) -> OptimizationResult:
    hessian = None
    try:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            hessian = function.hessian_at(steps[-1].point)
            classification = f"Hessian test: {classify_stationary_hessian(hessian)}."
    except (ValueError, TypeError, ArithmeticError, np.linalg.LinAlgError) as error:
        classification = f"Hessian classification unavailable: {error}"
    message = f"Gradient norm is below the tolerance. {classification}"
    return OptimizationResult(function, method, tuple(steps), True, message,
                              stationary_hessian=matrix_values(hessian) if hessian is not None else None)
