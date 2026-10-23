"""Shared chart styling: a validated categorical palette, recessive axes, headline titles."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import config  # noqa: E402

# Categorical slots in fixed order (light mode), plus sequential/diverging steps.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE, ORANGE, AQUA, YELLOW = SERIES[:4]
RED = "#e34948"
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def style(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def figure(width: float = 9, height: float = 4.6, ncols: int = 1):
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for ax in (axes if ncols > 1 else [axes]):
        style(ax)
    return fig, axes


def headline(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(0.015, 0.965, title, fontsize=13, color=INK, fontweight="bold", va="top")
    fig.text(0.015, 0.895, subtitle, fontsize=9.5, color=INK_2, va="top")


def save(fig: plt.Figure, name: str, top: float = 0.82) -> str:
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(config.FIG_DIR / name, facecolor=SURFACE)
    plt.close(fig)
    return f"figures/{name}"


def pct(v: float, digits: int = 1) -> str:
    return f"{100 * v:.{digits}f}%"
