"""Newton steps and Broyden inverse-Hessian updates from the formula sheet."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import ArrayLike

MAX_HESSIAN_CONDITION = 1e12
DEFAULT_BROYDEN_DENOMINATOR_TOLERANCE = 1e-14


@dataclass(frozen=True)
class BroydenUpdate:
    """Operands of the numerically scaled inverse Broyden update."""

    row: tuple[float, ...]
    scaled_gradient_change: tuple[float, ...]
    discrepancy: tuple[float, ...]
    denominator: float
    correction: tuple[tuple[float, ...], ...]
    matrix: tuple[tuple[float, ...], ...]
    displacement_scale: float
    gradient_scale: float
    row_scale: float


def matrix_values(matrix: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(map(float, row)) for row in matrix)


def newton_step(gradient: ArrayLike, hessian: ArrayLike) -> np.ndarray:
    """Return the displacement d satisfying H d = -gradient."""

    return _solve_hessian(hessian, -np.asarray(gradient, dtype=float))


def initial_inverse_hessian(hessian: ArrayLike) -> np.ndarray:
    """Compute the initial inverse Hessian for Broyden by solving H B = I."""

    matrix = np.asarray(hessian, dtype=float)
    return _solve_hessian(matrix, np.eye(matrix.shape[0]))


def _solve_hessian(hessian: ArrayLike, right_hand_side: np.ndarray) -> np.ndarray:
    matrix = np.asarray(hessian, dtype=float)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("The Hessian contains non-finite values.")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        condition = np.linalg.cond(matrix)
        if not isfinite(condition):
            raise np.linalg.LinAlgError("The Hessian is singular or its condition number is not finite.")
        if condition > MAX_HESSIAN_CONDITION:
            raise np.linalg.LinAlgError(f"The Hessian is ill-conditioned (condition number {condition:.3g}).")
        solution = np.linalg.solve(matrix, right_hand_side)
    if not np.all(np.isfinite(solution)):
        raise ArithmeticError("The Hessian solve produced non-finite values.")
    return solution


def classify_stationary_hessian(hessian: ArrayLike) -> str:
    """Apply the numerical second-order test at a stationary point."""

    classification = hessian_spectrum(hessian)[1]
    return {
        "positive definite": "strict local minimum (positive definite)",
        "negative definite": "strict local maximum (negative definite)",
        "indefinite": "saddle point (indefinite)",
        "singular or nearly singular": "inconclusive (zero or near-zero Hessian eigenvalues)",
    }[classification]


def hessian_spectrum(hessian: ArrayLike) -> tuple[tuple[float, ...], str]:
    """Classify the local quadratic model using a relative eigenvalue threshold."""

    eigenvalues = np.linalg.eigvalsh(np.asarray(hessian, dtype=float))
    if not np.all(np.isfinite(eigenvalues)):
        raise ArithmeticError("The Hessian eigenvalues are not finite.")
    scale = float(np.max(np.abs(eigenvalues)))
    tolerance = np.finfo(float).eps * len(eigenvalues) * scale
    positive = eigenvalues > tolerance
    negative = eigenvalues < -tolerance
    classification = "singular or nearly singular"
    if np.all(positive):
        classification = "positive definite"
    elif np.all(negative):
        classification = "negative definite"
    elif np.any(positive) and np.any(negative):
        classification = "indefinite"
    return tuple(map(float, eigenvalues)), classification


def broyden_step(gradient: ArrayLike, inverse_hessian: ArrayLike) -> np.ndarray:
    """Return the displacement using the current inverse Hessian B."""

    with np.errstate(over="ignore", invalid="ignore"):
        displacement = -(np.asarray(inverse_hessian, dtype=float) @ gradient)
    if not np.all(np.isfinite(displacement)):
        raise ArithmeticError("The Broyden displacement is not finite.")
    return displacement


def broyden_update(
    inverse_hessian: ArrayLike,
    displacement: ArrayLike,
    gradient_change: ArrayLike,
    *,
    denominator_tolerance: float = DEFAULT_BROYDEN_DENOMINATOR_TOLERANCE,
) -> np.ndarray:
    """Apply the inverse Broyden update with a relative denominator tolerance."""

    return _inverse_update(inverse_hessian, displacement, gradient_change, denominator_tolerance)[0]


def broyden_update_details(
    inverse_hessian: ArrayLike, displacement: ArrayLike, gradient_change: ArrayLike,
) -> tuple[np.ndarray, BroydenUpdate]:
    """Apply the same update and retain the operands used by the solver."""

    updated, correction, row, scaled_g, discrepancy, denominator, d_scale, g_scale, row_scale = (
        _inverse_update(inverse_hessian, displacement, gradient_change, DEFAULT_BROYDEN_DENOMINATOR_TOLERANCE)
    )
    return updated, BroydenUpdate(
        tuple(row), tuple(scaled_g), tuple(discrepancy), denominator,
        matrix_values(correction), matrix_values(updated), d_scale, g_scale, row_scale,
    )


def _inverse_update(inverse_hessian, displacement, gradient_change, denominator_tolerance):
    if not isfinite(denominator_tolerance) or denominator_tolerance < 0:
        raise ValueError("The denominator tolerance must be finite and non-negative.")
    matrix = np.asarray(inverse_hessian, dtype=float)
    d = np.asarray(displacement, dtype=float)
    g = np.asarray(gradient_change, dtype=float)
    if not all(np.all(np.isfinite(value)) for value in (matrix, d, g)):
        raise ArithmeticError("Broyden update received non-finite values.")

    error = "Broyden update has a zero or numerically small denominator."
    d_scale = np.max(np.abs(d))
    g_scale = np.max(np.abs(g))
    if d_scale == 0 or g_scale == 0:
        raise ZeroDivisionError(error)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        # Normalize the row and gradient difference. Their scales cancel in the update.
        row = (d / d_scale) @ matrix
        row_scale = np.max(np.abs(row))
        if not isfinite(row_scale):
            raise ArithmeticError("Broyden update produced a non-finite denominator.")
        if row_scale == 0:
            raise ZeroDivisionError(error)
        row /= row_scale
        scaled_g = g / g_scale
        denominator = float(row @ scaled_g)
        relative_scale = np.linalg.norm(row) * np.linalg.norm(scaled_g)
        if abs(denominator) <= denominator_tolerance * relative_scale:
            raise ZeroDivisionError(error)
        discrepancy = matrix @ scaled_g - d / g_scale
        correction = np.outer(discrepancy, row) / denominator
        updated = matrix - correction
    if not np.all(np.isfinite(updated)):
        raise ArithmeticError("The updated inverse-Hessian approximation is not finite.")
    return updated, correction, row, scaled_g, discrepancy, denominator, d_scale, g_scale, row_scale
