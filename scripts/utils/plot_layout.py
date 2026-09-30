"""Shared canvas and panel sizing for the README training plots.

``plot_training_run.py`` and ``plot_eval_curves.py`` render wide, short figures
whose panels must line up on GitHub, so the canvas size and the per-panel
footprint live here instead of being duplicated in each script.
"""

from __future__ import annotations

from matplotlib.axes import Axes

FIGURE_WIDTH = 12.0
FIGURE_HEIGHT = 3.2
FIGURE_DPI = 150
PANEL_WIDTH = 2.85
PANEL_HEIGHT = 2.15
PANEL_GAP = 0.9
PANEL_BOTTOM = 0.65


def place_panel_row(*, axes: list[Axes]) -> None:
    """Center a row of equal-sized panels on the shared canvas."""
    group_width = len(axes) * PANEL_WIDTH + (len(axes) - 1) * PANEL_GAP
    group_left = (FIGURE_WIDTH - group_width) / 2
    for index, axis in enumerate(axes):
        # A shared canvas and fixed panel size keep the panels equally sized on GitHub.
        panel_left = group_left + index * (PANEL_WIDTH + PANEL_GAP)
        axis.set_position(
            (
                panel_left / FIGURE_WIDTH,
                PANEL_BOTTOM / FIGURE_HEIGHT,
                PANEL_WIDTH / FIGURE_WIDTH,
                PANEL_HEIGHT / FIGURE_HEIGHT,
            )
        )
