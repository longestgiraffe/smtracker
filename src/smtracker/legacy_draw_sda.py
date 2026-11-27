"""The legacy window's SDA tab, drawn: matplotlib only, no Qt (thin-Qt rule 1, `tests/test_ui_thin_qt.py`).

`ui.legacy.tab_sda` lays out widgets and calls these inside `PlotPanel.draw(...)`. All in 2.0's reversed hot colour map
(`flip(hot(1024))`). Numbers: `smtracker.legacy_view_sda`; the model cell's helpers: `smtracker.legacy_draw_cell`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import legacy_view_sda as sd
from .legacy_draw_cell import cell_axes, clip_to_cell, draw_outline, hot_reversed_cmap
from .legacy_style import matlab_rc

BAR_BLUE = "#0072bd"
BAR_ORANGE = "#d95319"


def paint_heat_map(fig, grid: dict, clim: list, title: str, *, three_d: bool = False) -> None:
    """One heat map as `plotHeatmap.m` draws it: the reversed hot colour map from 0 to ``clim``'s top, the cell's
    outline over it, the colour bar, "Long axis" / "Short axis". ``three_d``: the value as height."""
    cmap = hot_reversed_cmap()
    top = float(clim[1]) if clim[1] > 0 else 1.0
    if three_d:
        ax = fig.add_subplot(111, projection="3d")
        if grid["kind"] == "location":
            step = 16
            xx, yy = np.meshgrid(grid["x"][::step], grid["y"][::step])
            zz = np.clip(grid["image"][::step, ::step], 0, None)
        else:
            xx, yy = np.meshgrid((grid["x_edges"][:-1] + grid["x_edges"][1:]) / 2,
                                 (grid["y_edges"][:-1] + grid["y_edges"][1:]) / 2)
            zz = np.nan_to_num(grid["mean"], nan=0.0)
        surf = ax.plot_surface(xx, yy, zz, cmap=cmap, vmin=0.0, vmax=top, linewidth=0, antialiased=False)
        contour = sd.model_cell_contour()
        ax.plot(contour[:, 0], contour[:, 1], np.zeros(len(contour)), "-", color="#000000", linewidth=1.5)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.6, 0.6)
        ax.set_zlim(0, top)
        ax.set_xlabel("Long axis")
        ax.set_ylabel("Short axis")
        ax.set_title(title, fontsize=9)
        fig.colorbar(surf, ax=ax, fraction=0.04, pad=0.08)
        return
    ax = fig.add_subplot(111)
    if grid["kind"] == "location":
        artist = ax.imshow(grid["image"], origin="lower", cmap=cmap, vmin=0.0, vmax=top, aspect="auto",
                           extent=[grid["x"][0], grid["x"][-1], grid["y"][0], grid["y"][-1]],
                           interpolation="nearest")
    else:
        artist = ax.pcolormesh(grid["x_edges"], grid["y_edges"], np.ma.masked_invalid(grid["mean"]), cmap=cmap,
                               vmin=0.0, vmax=top, shading="flat", linewidth=0)
    clip_to_cell(artist, ax)
    draw_outline(ax, linewidth=2.0)
    cell_axes(ax)
    ax.set_title(title, fontsize=9)
    bar = fig.colorbar(artist, ax=ax, fraction=0.04, pad=0.02)
    if grid["kind"] == "dynamics":
        bar.set_ticks([0, top / 2, top])


def paint_x_distribution(fig, source: int, dist: dict, title: str) -> None:
    ax = fig.add_subplot(111)
    if source == 0:
        d = dist["x"]
        centres = (d["edges"][:-1] + d["edges"][1:]) / 2
        ax.bar(centres, d["probability"], width=np.diff(d["edges"]), color=BAR_BLUE, edgecolor="#262626",
               linewidth=0.4, label="Empirical probability")
        ax.set_xlim(-0.1, 1.1)
        ax.set_xlabel("distribution long axis")
        ax.set_ylabel("probability")
    else:
        n = len(dist["x_median"])
        index = np.arange(1, n + 1)
        ax.errorbar(index, dist["x_median"], yerr=[np.zeros(n), np.nan_to_num(dist["x_std"])], fmt=".",
                    color=BAR_BLUE, linewidth=0.6, capsize=0)
        ax.bar(index, np.nan_to_num(dist["x_median"]), color=BAR_ORANGE, edgecolor="#262626", linewidth=0.3,
               width=0.8, label="Average Value")
        ax.set_xlim(0, n + 1)
        ax.set_xticks([0, (n + 1) / 2, n + 1], ["0", "0.5", "1"])
        ax.set_xlabel("Distribution along normalized X axis")
        ax.set_ylabel(sd.SOURCE_UNITS[sd.SOURCES[source]])
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()


def paint_y_distribution(fig, source: int, dist: dict, title: str) -> None:
    """The y distribution as 2.0 rotated it (`view(ax, -90, -90)`): the cell's short axis runs down the page."""
    ax = fig.add_subplot(111)
    if source == 0:
        d = dist["y"]
        centres = (d["edges"][:-1] + d["edges"][1:]) / 2
        ax.barh(centres, d["probability"], height=np.diff(d["edges"]), color=BAR_BLUE, edgecolor="#262626",
                linewidth=0.4, label="Empirical probability")
        ax.set_ylabel("distribution short axis")
        ax.set_xlabel("probability")
    else:
        n = len(dist["y_median"])
        index = np.arange(1, n + 1)
        ax.errorbar(dist["y_median"], index, xerr=[np.zeros(n), np.nan_to_num(dist["y_std"])], fmt=".",
                    color=BAR_BLUE, linewidth=0.6, capsize=0)
        ax.barh(index, np.nan_to_num(dist["y_median"]), color=BAR_ORANGE, edgecolor="#262626", linewidth=0.3,
                height=0.8, label="Average Value")
        ax.set_ylim(0, n + 1)
        ax.set_yticks([0, (n + 1) / 2, n + 1], ["0", "0.5", "1"])
        ax.set_ylabel("Distribution along normalized Y axis")
        ax.set_xlabel(sd.SOURCE_UNITS[sd.SOURCES[source]])
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()


def save_individual_maps(project, condition: str, folder, title: str) -> list:
    """2.0's "Individual cells": a location heat map for every cell of the condition, written into ``folder`` as PNGs.
    Returns the paths."""
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    written = []
    for cell in sd.individual_cells(project, condition):
        grid = sd.location_map(cell["x"], cell["y"], False)
        limit = sd.grow_clim([0.0, 0.0], grid["vmax"])          # `reset_caxis` before every cell
        with matplotlib.rc_context(matlab_rc()):
            fig = Figure(figsize=(6, 3), dpi=150, facecolor="white")
            FigureCanvasAgg(fig)
            paint_heat_map(fig, grid, limit, f"{title} - {condition}")
            safe = "".join(ch if ch.isalnum() else "_" for ch in f"{cell['movie']}_{cell['cell']}")
            path = Path(folder) / f"legacy_sda_{condition}_{safe}.png"
            fig.savefig(path, dpi=150, facecolor="white")
        written.append(path)
    return written
