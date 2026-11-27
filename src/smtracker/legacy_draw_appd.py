"""The legacy window's APPD tab and its Speed maps window, drawn: matplotlib only, no Qt (thin-Qt rule 1,
`tests/test_ui_thin_qt.py`).

`ui.legacy.tab_appd` lays out widgets and calls these inside `PlotPanel.draw(...)`. Numbers: `legacy_view_appd`.
"""
from __future__ import annotations

import numpy as np

from . import legacy_view_appd as la
from .legacy_draw_sqd import BUBBLE_AREA, BUBBLE_STEP, BUBBLE_Y, D_LABEL, bubble_legend_handles
from .legacy_style import APPD_POP_COLOURS

#: 2.0's words for the populations (`AppDPlot.m`, `crossedErrorbar`).
POP_WORDS = {1: ("1 Population model",), 2: ("Static Population", "Diffusive Population"),
             3: ("Static Population", "Diffusive Population", "Fast Diffusive Population")}
DIFFUSION_WORDS = ("Static", "Mobile", "Fast Mobile")
#: Component lines as 2.0 drew them: dotted, dashed and (for the three-population case's first) solid thin.
POP_LINES = {2: (":", "--"), 3: ("-", ":", "--")}
MARKERS = ("o", "s", "d", "^", "+", "*", ".", "x", "v", ">", "<", "p", "h")
PDF_LABEL = r"pdf [per $\mu$m$^{2}$s$^{-1}$]"


def x_axis_max(run: dict) -> int:
    """The largest apparent D across the conditions, rounded up: the histograms' common x range."""
    return int(np.ceil(max(float(np.max(v)) for v in run["values"].values())))


def _limit(text: str, default: float) -> float:
    try:
        return float(text)
    except ValueError:
        return default


def draw_hist(fig, run: dict, condition: str, choice: str, x_max: float, *, colour: str, bin_width: float,
              title: str, max_x_text: str, max_y_text: str) -> tuple:
    """One condition's apparent-D histogram with its fitted populations. ``max_x_text`` / ``max_y_text``: the axes
    boxes as typed ("" = the default). Returns ``(x_lim, y_default)`` so the boxes can be filled where they were blank."""
    values = run["values"][condition]
    ax = fig.add_subplot(111)
    edges = np.arange(0.0, max(float(values.max()), bin_width) + bin_width, bin_width)
    ax.hist(values, bins=edges, density=True, color=colour, alpha=0.25, edgecolor=colour, linewidth=0.5,
            label=f"D with R² >= {run['r2_lim']:g}")
    grid = np.arange(0.0, x_max + 0.001, 0.001)
    grid[0] = grid[1] / 10.0 if len(grid) > 1 else 1e-6
    curves = la.curves_for(run, condition, choice, grid)
    n = curves["n_pop"]
    if n == 1:
        ax.plot(grid, curves["sum"], "-", color=colour, linewidth=2.0, label=POP_WORDS[1][0])
    else:
        for k, part in enumerate(curves["parts"]):
            ax.plot(grid, part, linestyle=POP_LINES[n][k], color=APPD_POP_COLOURS[k], linewidth=1.5,
                    label=POP_WORDS[n][k])
        ax.plot(grid, curves["sum"], "-", color=colour, linewidth=2.2, label="Sum")
    all_max = max(float(np.max(la.curves_for(run, c, choice, grid)["sum"])) for c in run["conditions"])
    x_lim = _limit(max_x_text, float(x_max))
    ax.set_xlim(0, x_lim)
    y_default = max(all_max, ax.get_ylim()[1])
    ax.set_ylim(0, _limit(max_y_text, y_default))
    ax.set_title(title)
    ax.set_xlabel(r"D [$\mu$m$^{2}$s$^{-1}$] " + f"(N = {len(values)})")
    ax.set_ylabel(PDF_LABEL)
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    return x_lim, y_default


def draw_bubbles(fig, run: dict, choice: str, colours: dict, title: str) -> bool:
    """The bubble plot. Returns True where fraction sizes <= 0 were left out (the Console says so)."""
    data = la.bubble_data(run, choice)
    ax = fig.add_subplot(111)
    conditions = run["conditions"]
    share = np.where(data["a"] <= 0, 1e-5, data["a"])
    x = BUBBLE_STEP * (data["condition_index"] + 1 - 0.2)        # offset(1) = -0.2 (`bubbleplotAPPD`)
    ink = [colours.get(conditions[i], "#0072bd") for i in data["condition_index"]]
    ax.scatter(x, data["y"], s=BUBBLE_AREA * share, c=ink, edgecolors="#000000", linewidths=0.8, zorder=3)
    ax.errorbar(x, data["y"], yerr=np.where(np.isfinite(data["y_sd"]), data["y_sd"], 0.0), fmt="k.", linewidth=0.8,
                zorder=2)
    for xi, yi, si in zip(x, data["y"], share):
        ax.text(xi + 0.2 * BUBBLE_STEP, yi, f"{si:.3g}%", va="center", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylim(*BUBBLE_Y)
    ax.set_xlim(0.4 * BUBBLE_STEP, (len(conditions) + 0.8) * BUBBLE_STEP)
    ax.set_xticks([BUBBLE_STEP * (i + 1) - 0.2 * BUBBLE_STEP for i in range(len(conditions))],
                  [f"{i + 1}-{c}" for i, c in enumerate(conditions)])
    ax.set_ylabel(D_LABEL)
    ax.set_xlabel("Conditions")
    ax.yaxis.grid(True, which="major")
    ax.set_title(title)
    ax.legend(handles=bubble_legend_handles(), loc="lower right")
    fig.tight_layout()
    return bool(data["any_nonpositive"])


def draw_crossed(fig, run: dict, choice: str, colours: dict, title: str) -> None:
    data = la.bubble_data(run, choice)
    ax = fig.add_subplot(111)
    conditions = run["conditions"]
    words = DIFFUSION_WORDS
    for i, condition in enumerate(conditions):
        colour = colours.get(condition, "#0072bd")
        marker = MARKERS[i % len(MARKERS)]
        mask = data["condition_index"] == i
        for k, (a, a_sd, y, y_sd) in enumerate(zip(data["a"][mask], data["a_sd"][mask], data["y"][mask],
                                                   data["y_sd"][mask])):
            ax.errorbar([a], [y], xerr=[a_sd if np.isfinite(a_sd) else 0.0], yerr=[y_sd if np.isfinite(y_sd) else 0.0],
                        marker=marker, color=colour, markerfacecolor=colour, markersize=5, capsize=2, linewidth=0.8,
                        label=f"{condition} - {words[k]}")
    ax.set_xlabel("Fraction [%]")
    ax.set_ylabel(D_LABEL)
    ax.yaxis.grid(True)
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=7)
    fig.subplots_adjust(bottom=0.32, left=0.16, right=0.96, top=0.9)


# -- the Speed maps window ------------------------------------------------------------------------------ #
def speed_title(result: dict, what: str) -> str:
    return f"{result['condition']}\n{what} between [{result['lower']:g} and {result['upper']:g}]"


def draw_speed_tracks(fig, r: dict, colour: str) -> None:
    ax = fig.add_subplot(111)
    ax.plot(r["stacked"][:, 0], r["stacked"][:, 1], linewidth=0.5, color=colour, alpha=0.6)
    ax.plot(r["contour"][:, 0], r["contour"][:, 1], "k-", linewidth=2)
    ax.plot(r["contour"][[-1, 0], 0], r["contour"][[-1, 0], 1], "k-", linewidth=2)
    starts = np.array([xy[0] for xy in r["tracks"].values()])
    ax.scatter(starts[:, 0], starts[:, 1], s=3, color=colour)
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.65, 0.65)
    ax.set_yticks([-0.5, 0, 0.5])
    ax.set_box_aspect(1 / 3)
    ax.set_xlabel(r"[$\mu$m]")
    ax.set_ylabel(r"[$\mu$m]")
    ax.set_title(speed_title(r, "Tracks with an apparent diffusion according to their MSD curve"), fontsize=7)
    ax.legend([f"D: [{r['lower']:g}-{r['upper']:g}]"], loc="upper right", fontsize=7)
    fig.tight_layout()


def draw_speed_heat(fig, r: dict) -> None:
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    ax = fig.add_subplot(111)
    image = ax.imshow(r["heat"].T, origin="lower", extent=(-0.1, 1.1, -0.6, 0.6), aspect="auto", cmap="jet")
    contour = np.vstack([r["contour"], r["contour"][:1]])
    outer = np.array([[-0.1, -0.6], [1.1, -0.6], [1.1, 0.6], [-0.1, 0.6], [-0.1, -0.6]])
    hole = contour[::-1]                                  # opposite winding: the cell is left uncovered
    verts = np.vstack([outer, hole])
    codes = ([Path.MOVETO] + [Path.LINETO] * (len(outer) - 1) + [Path.MOVETO] + [Path.LINETO] * (len(hole) - 1))
    ax.add_patch(PathPatch(Path(verts, codes), facecolor="white", edgecolor="white", linewidth=2, zorder=2))
    ax.plot(contour[:, 0], contour[:, 1], "k-", linewidth=2, zorder=3)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.6, 0.6)
    ax.set_xlabel(rf"[$\mu$m] / N:{r['n_label']}")
    ax.set_ylabel(r"[$\mu$m]")
    ax.set_title(speed_title(r, "Heatmap of Tracks with an apparent diffusion according to their MSD curve"),
                 fontsize=7)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
