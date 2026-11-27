"""The legacy window's Clustering tab, drawn: matplotlib only, no Qt (thin-Qt rule 1, `tests/test_ui_thin_qt.py`).

`ui.legacy.tab_clustering` lays out widgets and calls these inside `PlotPanel.draw(...)`. Numbers: `legacy_view_clustering`.
"""
from __future__ import annotations

import math

import numpy as np

from . import legacy_view_clustering as lc
from .legacy_style import MATLAB_LINES

#: The heat maps' grid: 2.0 put clusters 1 to 3 in the left column and 4 to 6 in the right one.
HEAT_ROWS, HEAT_COLS = 3, 2


def draw_msd(fig, run: dict) -> None:
    """One MSD line per cluster, its width the cluster's share."""
    ax = fig.add_subplot(111)
    for i, cluster in enumerate(run["clusters"]):
        curve = cluster["curve"]
        with np.errstate(invalid="ignore", divide="ignore"):
            sem = curve[:, 2] / np.sqrt(curve[:, 3])
        ax.errorbar(curve[:, 0], curve[:, 1], yerr=np.where(np.isfinite(sem), sem, 0.0), linestyle="-",
                    linewidth=math.ceil(cluster["fraction"] * 10), marker="+", markersize=8,
                    color=MATLAB_LINES[i % len(MATLAB_LINES)])
    ax.set_xlim(0, run["clusters"][0]["curve"][-1, 0] + run["tau_s"])
    ax.set_title(f"MSD in {run['condition']}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"MSD [$\mu$m$^2$/s]")
    ax.grid(True)
    fig.tight_layout()


def draw_heat(fig, run: dict, overlay: bool) -> None:
    """One heat map per cluster on the model cell, in jet on one colour scale, the tracks over them if ``overlay``."""
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    clusters = run["clusters"][: HEAT_ROWS * HEAT_COLS]
    top = max(c["heat"]["maximum"] for c in clusters)            # one colour scale: every map to the largest
    top = top if top > 0 else 1.0
    x0, x1 = lc.HEAT_X
    y0, y1 = lc.HEAT_Y
    rectangle = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]])
    contour = lc.MODEL_CELL_CONTOUR
    area = 0.5 * float(np.sum(contour[:, 0] * np.roll(contour[:, 1], -1) - np.roll(contour[:, 0], -1) * contour[:, 1]))
    hole = contour[::-1] if area > 0 else contour          # the hole winds against the rectangle (counter-clockwise)
    # white everywhere but inside the model cell (2.0 filled the outside, `plotHeatmap.m:49`)
    mask = Path(np.vstack([rectangle, hole, hole[:1]]),
                [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY]
                + [Path.MOVETO] + [Path.LINETO] * (len(contour) - 1) + [Path.CLOSEPOLY])
    for i, cluster in enumerate(clusters):
        col, row = divmod(i, HEAT_ROWS)
        ax = fig.add_subplot(HEAT_ROWS, HEAT_COLS, row * HEAT_COLS + col + 1)
        heat = cluster["heat"]
        image = ax.imshow(heat["surface"].T, origin="lower", extent=heat["extent"], cmap="jet", vmin=0,
                          vmax=top, aspect="auto", interpolation="nearest")
        ax.add_patch(PathPatch(mask, facecolor="white", edgecolor="white", linewidth=2, zorder=2))
        ax.plot(contour[:, 0], contour[:, 1], "k-", linewidth=1.6, zorder=3)
        ink = MATLAB_LINES[i % len(MATLAB_LINES)]
        if overlay and len(heat["overlay"]):
            ax.plot(heat["overlay"][:, 0], heat["overlay"][:, 1], color=ink, linewidth=0.4, zorder=4)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_box_aspect(1 / 3)
        ax.tick_params(colors="white", labelcolor="white")             # 2.0 hid the ticks (`XColor 'w'`)
        for spine in ax.spines.values():
            spine.set_edgecolor("white")
        ax.set_xlabel(f"Long axis / N:{heat['n_inside']} - {heat['percent']:.6g} %", fontsize=8)
        ax.set_ylabel("Short axis", fontsize=8)
        ax.set_title(f"{run['condition']}\n{lc.cluster_title(cluster)}", color=ink, fontsize=8)
        colourbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
        colourbar.ax.tick_params(labelsize=7)
    fig.subplots_adjust(left=0.04, right=0.96, top=0.93, bottom=0.07, hspace=0.75, wspace=0.18)
