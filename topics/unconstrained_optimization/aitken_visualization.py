"""Plots of ordinary iterates, Aitken estimates and their residuals."""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from topics.unconstrained_optimization.methods.aitken import FixedPointRun


def plot_aitken_values(figure: Figure, result: FixedPointRun) -> None:
    _plot_comparison(
        figure, result.points, tuple(row.estimate for row in result.estimates),
        tuple(row.index for row in result.estimates), "Iteration value", "Ordinary xⁱ", "Aitken yⁱ",
    )


def plot_aitken_residuals(figure: Figure, result: FixedPointRun) -> None:
    _plot_comparison(
        figure, result.residuals, tuple(row.residual for row in result.estimates),
        tuple(row.index for row in result.estimates), "Absolute residual",
        "Ordinary r(xⁱ)", "Aitken r(yⁱ)", prefer_log=True,
    )


def _plot_comparison(
    figure: Figure,
    ordinary: tuple[float | None, ...],
    estimates: tuple[float | None, ...],
    indices: tuple[int, ...],
    y_label: str,
    ordinary_label: str,
    estimate_label: str,
    *,
    prefer_log: bool = False,
) -> None:
    figure.clear()
    figure.patch.set_facecolor("#111827")
    axis = figure.add_subplot(111)
    axis.set_facecolor("#0f172a")
    visible = [value for series in (ordinary, estimates) for value in series if value is not None]
    largest = max(map(abs, visible), default=0.0)
    scale = largest if largest >= 1e150 or 0 < largest <= 1e-150 else 1.0
    if scale != 1:
        y_label += f" (× {scale:.6g})"
    axis.plot(range(len(ordinary)), np.asarray(ordinary, dtype=float) / scale,
              marker="o", markersize=4, color="#60a5fa", label=ordinary_label)
    axis.plot(indices, np.asarray(estimates, dtype=float) / scale,
              marker="s", markersize=4, color="#f59e0b", label=estimate_label)
    if prefer_log and visible and all(value / scale >= 1e-300 for value in visible):
        axis.set_yscale("log")
    axis.set_xlabel("Iteration i", color="#cbd5e1")
    axis.set_ylabel(y_label, color="#cbd5e1")
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.tick_params(colors="#cbd5e1")
    for spine in axis.spines.values():
        spine.set_color("#475569")
    axis.grid(color="#475569", alpha=0.3, linewidth=0.6)
    axis.legend(facecolor="#172033", edgecolor="#334155", labelcolor="#f8fafc", fontsize=8)
    figure.tight_layout(pad=1.1)
