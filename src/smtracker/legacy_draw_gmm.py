"""The legacy window's GMM tab and its GMM & GoF window, drawn: matplotlib only, no Qt (thin-Qt rule 1,
`tests/test_ui_thin_qt.py`).

`ui.legacy.tab_gmm` lays out widgets and calls these inside `PlotPanel.draw(...)`. Numbers: `legacy_view_gmm`.
"""
from __future__ import annotations

import numpy as np

from . import legacy_view_gmm as lg
from .legacy_draw_sqd import BUBBLE_AREA, BUBBLE_STEP, BUBBLE_Y, D_LABEL, bubble_legend_handles

#: `params.colorOrder(1:3)` (Okabe-Ito, `params.mat`): the single, double and triple fits' lines.
FIT_INKS = {1: "#0072b2", 2: "#e69f00", 3: "#009e73"}
#: `plotGMMFit`'s line styles: the total solid, then the populations dotted and dashed (triple: solid, dotted, dashed).
COMPONENT_STYLES = {2: ("-", ":", "--"), 3: ("-", ":", "--")}
MARKERS = ("o", "s", "D", "^", "+", "*", ".", "x", "v", ">", "<", "p", "h")
POPULATION_LEGEND = {2: ("Double fit", "Static Population", "Mobile Population"),
                     3: ("Triple fit", "Static Population", "Slow-mobile Population", "Mobile Population")}


def draw_qq(fig, run: dict) -> None:
    """The quantile-quantile plot of the reference against the comparison."""
    ref, comp = run["reference"], run["comparison"]
    q = lg.qq_points(run["samples"][ref], run["samples"][comp])
    ax = fig.add_subplot(111)
    ax.plot(q["x"], q["y"], "+", color="#0000ff", markersize=4)
    x1, y1, slope = q["line"]
    span = np.array([q["x"].min(), q["x"].max()])
    ax.plot(span, y1 + slope * (span - x1), "--", color="#ff0000", linewidth=1.0)
    ax.set_title("GMM - Quantile-Quantile Plot")
    ax.set_xlabel(f"{ref} - Quantiles Observed data", fontsize=9)
    ax.set_ylabel(f"{comp} - Quantiles Predicted data", fontsize=9)
    ax.grid(True)
    fig.tight_layout()


def draw_pp(fig, run: dict, condition: str, n_pop: int, colour: str) -> None:
    """One condition's probability-probability plot."""
    p = lg.pp_points(run, condition, n_pop)
    ax = fig.add_subplot(111)
    ax.plot(p["observed"], p["predicted"], "o", markersize=4, markerfacecolor="none", markeredgecolor=colour,
            label="Data")
    ax.plot(p["observed"], p["line"], "--", color="#e6e6e6", linewidth=0.8,
            label=f"Slope = {p['slope']:.5g}\nIntercept={p['intercept']:.3g}\nR$^2$ = {p['r2']:.3g}")
    ax.plot([-0.1, 1.1], [-0.1, 1.1], "k--", linewidth=1.0, label="f(x) = x")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_title(f"GMM - {condition} - {lg.FIT_NAMES[n_pop]}")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    fig.subplots_adjust(left=0.13, right=0.62, top=0.9, bottom=0.17)


def draw_hist(fig, condition: str, values, run, colour: str, title: str, *, bin_width: float, xmin: float,
              xmax: float, ymax: float, grid: int) -> None:
    """One condition's histogram of steps with the fitted Gaussians of ``run`` (None before a run at these settings)."""
    from matplotlib.colors import to_rgba

    ax = fig.add_subplot(111)
    bw = bin_width
    h = lg.histogram(values, bw)
    bars = ax.bar(h["edges"][:-1], h["density"], width=bw, align="edge", facecolor=to_rgba(colour, 0.1),
                  edgecolor=to_rgba("#262626", 0.5), linewidth=0.3, label="Probability")
    handles = [bars]
    if run is not None:
        x = np.arange(xmin, xmax + bw / 20, bw / 10)
        for k in sorted(run["fits"]):
            fit = run["fits"][k]
            if condition not in fit["per"]:
                continue
            curves = lg.gmm_pdf(run, condition, k, x)
            ink = FIT_INKS[k]
            if k == 1:
                handles += ax.plot(x, curves["total"], color=ink, linewidth=2, label="Single fit")
                continue
            names = POPULATION_LEGEND[k]
            styles = COMPONENT_STYLES[k]
            handles += ax.plot(x, curves["total"], color=ink, linewidth=2, linestyle=styles[0], label=names[0])
            for j, comp in enumerate(curves["components"]):
                handles += ax.plot(x, comp, color=ink, linewidth=1 if (k == 3 and j == 0) else 2,
                                   linestyle=("-" if (k == 3 and j == 0) else styles[1 + (j % 2)]),
                                   label=names[1 + j])
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, ymax)
    grid = max(int(grid), 2)
    ax.set_xticks(np.linspace(xmin, xmax, grid))
    ax.grid(True)
    ax.set_xlabel(rf"Displacement $\Delta$x, $\Delta$y [$\mu$m]. (N = {h['n']})")
    ax.set_ylabel(r"Probability density [$\mu$m$^{-1}$]")
    ax.set_title(title)
    ax.legend(handles=handles, loc="upper right", fontsize=7)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.9, bottom=0.17)


def draw_bubbles(fig, run: dict, colours: dict, title: str) -> None:
    rows = lg.bubble_rows(run)
    n_models = len(rows)
    offsets = {k: (i - (n_models - 1) / 2) * 0.2 for i, k in enumerate(sorted(rows))}   # `-.2 0 +.2` of a column
    ax = fig.add_subplot(111)
    group = next(iter(run["fits"].values()))["conditions"]
    for k in sorted(rows):
        for column, (condition, D, D_sd, share, _share_sd) in enumerate(rows[k], start=1):
            x = BUBBLE_STEP * (column + offsets[k])
            for j in range(len(D)):
                ax.errorbar(x, D[j], yerr=D_sd[j] if np.isfinite(D_sd[j]) else None, color="#000000",
                            linewidth=0.8, capsize=2, zorder=2)
                ax.scatter([x], [D[j]], s=BUBBLE_AREA * share[j], color=colours.get(condition, "#000000"),
                           edgecolors="#000000", linewidths=0.8, zorder=3)
                ax.text(x + 0.12 * BUBBLE_STEP, D[j], f"{share[j]:.3g}%", va="center", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylim(*BUBBLE_Y)
    ax.set_xlim(0.4 * BUBBLE_STEP, (len(group) + 0.8) * BUBBLE_STEP)
    ax.set_xticks([i * BUBBLE_STEP for i in range(1, len(group) + 1)],
                  [f"{run['conditions'].index(c) + 1}-{c}" for c in group], fontsize=8)
    ax.set_ylabel(D_LABEL)
    ax.set_xlabel("Conditions")
    ax.yaxis.grid(True, which="major")
    ax.yaxis.grid(True, which="minor", alpha=0.15)
    ax.set_title(title)
    ax.legend(handles=bubble_legend_handles(), loc="lower right")
    fig.tight_layout()


def draw_crossed(fig, run: dict, colours: dict, title: str) -> None:
    rows = lg.bubble_rows(run)
    k = max(rows)                                  # 2.0 drew the last model's pairs only (`crossedErrorbar`)
    words = ("Static", "Mobile", "Fast Mobile")
    ax = fig.add_subplot(111)
    for i, (condition, D, D_sd, share, share_sd) in enumerate(rows[k]):
        ink = colours.get(condition, "#000000")
        for j in range(len(D)):
            ax.errorbar(share[j], D[j], xerr=share_sd[j] if np.isfinite(share_sd[j]) else 0.0,
                        yerr=D_sd[j] if np.isfinite(D_sd[j]) else 0.0, marker=MARKERS[i % len(MARKERS)],
                        color=ink, markerfacecolor=ink, markersize=5, capsize=2, linewidth=0.8,
                        label=f"{condition} - {words[j] if j < 3 else j + 1}")
    ax.set_xlabel("Fraction [%]")
    ax.set_ylabel(D_LABEL)
    ax.grid(True, axis="y")
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=7, ncol=2)
    fig.subplots_adjust(left=0.15, right=0.96, top=0.9, bottom=0.34)
