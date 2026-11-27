"""The legacy window's SQD tab and its Jump Distance window, drawn: matplotlib only, no Qt (thin-Qt rule 1,
`tests/test_ui_thin_qt.py`).

`ui.legacy.tab_sqd` and `ui.legacy.jd_window` lay out widgets and call these inside `PlotPanel.draw(...)`. The bubble
plot's constants live here because the GMM and APPD tabs draw the same plot. Numbers: `legacy_view`.
"""
from __future__ import annotations

import numpy as np

from . import legacy_view as lv
from .legacy_style import POP_LINESTYLES

#: 2.0's bubble plot (`bubbleplotLayout`, `SMTracker.m:6361-6420`): D on a log axis 1e-3 to 10, conditions 100 apart,
#: bubble area 20 x the share in per cent, the share written beside it.
BUBBLE_Y = (1e-3, 10.0)
BUBBLE_AREA = 20.0
BUBBLE_STEP = 100.0
#: 2.0's D axis label (`params.units`), in mathtext so every glyph is in the font.
D_LABEL = r"Diffusion [$\mu$m$^{2}$ s$^{-1}$]"
#: 2.0's names for a fit by its population count ("Triple fit" in the manual's figure).
FIT_NAMES = {1: "Single fit", 2: "Double fit", 3: "Triple fit"}


def draw_cdf(fig, run: dict, lag: int, pops: list, conditions: list, colours: dict, axes: tuple, title: str) -> None:
    """The cumulative distribution P(r^2, t) on a log axis with the fits, and their residuals under it."""
    x0, x1, y0, y1 = axes
    top = fig.add_axes([0.08, 0.36, 0.9, 0.56])
    bottom = fig.add_axes([0.08, 0.12, 0.9, 0.18], sharex=top)
    for condition in conditions:
        colour = colours.get(condition, "#000000")
        any_fit = run["fits"][lag][run["n_pop"]][condition]
        x, y = np.asarray(any_fit["cdf_x"]), np.asarray(any_fit["cdf_y"])
        keep = x > 0
        top.plot(x[keep], y[keep], color=colour, linewidth=1.2)
        for k in pops:
            fit_y = lv.sqd_curve(run, condition, k, lag, x[keep])
            top.plot(x[keep], fit_y, color="#000000", linestyle=POP_LINESTYLES[k], linewidth=1.0)
            bottom.plot(x[keep], y[keep] - fit_y, color=colour, linestyle=POP_LINESTYLES[k], linewidth=0.8)
    top.set_xscale("log")
    top.set_xlim(x0, x1)
    top.set_ylim(y0, y1)
    top.set_ylabel("P(r²,t)")
    top.grid(True, which="major")
    top.set_title(title)
    top.tick_params(labelbottom=False)
    bottom.set_ylabel("residuals")
    bottom.set_xlabel("r² (in µm²)")
    bottom.grid(True, which="major")


def bubble_legend_handles() -> list:
    """The legend 2.0's bubble plots share: a hollow circle for the fraction size, a dash for the diffusion."""
    from matplotlib.lines import Line2D

    return [Line2D([], [], marker="o", linestyle="", markerfacecolor="none", markeredgecolor="#000000",
                   label="Fraction Size"),
            Line2D([], [], marker="_", linestyle="-", color="#000000", label="Diffusion")]


def draw_bubbles(fig, run: dict, lag: int, pops: list, conditions: list, colours: dict, title: str) -> None:
    """The bubble plot: D on a log axis, bubble area the share, the share written beside it."""
    ax = fig.add_subplot(111)
    n_models = len(pops)
    offsets = {k: (i - (n_models - 1) / 2) * 0.2 * BUBBLE_STEP for i, k in enumerate(pops)}  # `:5144-5183`
    for ci, condition in enumerate(conditions, start=1):
        colour = colours.get(condition, "#000000")
        for k in pops:
            fit = run["fits"][lag][k][condition]
            x = ci * BUBBLE_STEP + offsets[k]
            half = (np.asarray(fit["D_hi"]) - np.asarray(fit["D_lo"])) / 2.0
            for j in range(k):
                share = 100.0 * float(fit["fraction"][j])
                ax.errorbar(x, fit["D"][j], yerr=half[j] if np.isfinite(half[j]) else None, color="#000000",
                            linewidth=0.8, capsize=2, zorder=2)
                ax.scatter([x], [fit["D"][j]], s=BUBBLE_AREA * share, color=colour, edgecolors="#000000",
                           linewidths=0.8, zorder=3)
                ax.text(x + 0.18 * BUBBLE_STEP, fit["D"][j], f"{share:.3g}%", va="center", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylim(*BUBBLE_Y)
    ax.set_xlim(0.4 * BUBBLE_STEP, (len(conditions) + 0.8) * BUBBLE_STEP)
    ax.set_xticks([i * BUBBLE_STEP for i in range(1, len(conditions) + 1)],
                  [f"{run['conditions'].index(c) + 1}-{c}" for c in conditions])
    ax.set_ylabel(D_LABEL)
    ax.set_xlabel("Conditions")
    ax.yaxis.grid(True, which="major")
    ax.yaxis.grid(True, which="minor", alpha=0.15)
    ax.set_title(title)
    ax.legend(handles=bubble_legend_handles(), loc="lower right")
    fig.tight_layout()


def draw_crossed(fig, run: dict, lag: int, pops: list, conditions: list, colours: dict, title: str) -> None:
    """The crossed error bars: share against D, one marker per population count."""
    ax = fig.add_subplot(111)
    markers = {1: "o", 2: "s", 3: "^"}
    for condition in conditions:
        colour = colours.get(condition, "#000000")
        for k in pops:
            fit = run["fits"][lag][k][condition]
            share = 100.0 * np.asarray(fit["fraction"], float)
            share_half = 100.0 * (np.asarray(fit["fraction_hi"]) - np.asarray(fit["fraction_lo"])) / 2.0
            d_half = (np.asarray(fit["D_hi"]) - np.asarray(fit["D_lo"])) / 2.0
            ax.errorbar(share, fit["D"], xerr=np.where(np.isfinite(share_half), share_half, 0),
                        yerr=np.where(np.isfinite(d_half), d_half, 0), linestyle="", marker=markers[k],
                        color=colour, markersize=5, capsize=2, linewidth=0.8)
    ax.set_xlabel("Fraction [%]")
    ax.set_ylabel(D_LABEL)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_title(title)
    fig.tight_layout()


# -- the Jump Distance window -------------------------------------------------------------------------- #
def draw_jd_histogram(fig, data: dict, colour: str, index: int, name: str) -> None:
    """One condition's single-step distances as a histogram with the fit's populations."""
    ax = fig.add_subplot(111)
    edges = data["edges"]
    centres = (edges[:-1] + edges[1:]) / 2.0
    width = edges[1] - edges[0]
    ax.bar(centres, data["probability"], width=width, color=colour, alpha=0.35, edgecolor="#262626",
           linewidth=0.3, label=f"{index}-{name} data")
    for part in data["components"]:
        ax.plot(centres, part, "--", color="#0072bd", linewidth=0.8)
    ax.plot(centres, data["model"], "-", color="#0072bd", linewidth=1.4,
            label=f"fit {data['n_pop']} pop{'s' * (data['n_pop'] > 1)}")
    ax.set_title(f"Single step distances\nfor protein {index}-{name}")
    ax.set_xlabel(f"Single step distances. (N= {data['n']})")
    ax.set_ylabel("Probability")
    ax.legend(loc="upper right")
    fig.tight_layout()


def draw_jd_pp(fig, data: dict) -> None:
    """The P-P plot of the fit with its straight line."""
    ax = fig.add_subplot(111)
    ax.plot(data["observed"], data["predicted"], "-", color="#0072bd", linewidth=1.2, label="Data")
    line = data["slope"] * data["observed"] + data["intercept"]
    ax.plot(data["observed"], line, "--", color="#d00000", linewidth=0.9,
            label=f"Slope = {data['slope']:.5g}\nIntercept={data['intercept']:.3g}\n"
                  f"R$^2$ = {data['r2']:.3g}")
    ax.set_title(FIT_NAMES.get(data["n_pop"], f"{data['n_pop']} populations"))
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.legend(loc="lower right")
    fig.tight_layout()
