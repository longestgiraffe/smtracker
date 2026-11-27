"""The legacy window's SLA tab, drawn: matplotlib only, no Qt (thin-Qt rule 1, `tests/test_ui_thin_qt.py`).

`ui.legacy.tab_sla` lays out widgets and calls these inside `PlotPanel.draw(...)`; each takes a figure and the run
`legacy_view_sla` made. Numbers: `smtracker.legacy_view_sla`; the model cell's helpers: `smtracker.legacy_draw_cell`.
"""
from __future__ import annotations

import numpy as np

from . import legacy_view_sla as ls
from .legacy_draw_cell import cell_axes, clip_to_cell, draw_outline, jet_cmap, lines_with_gaps
from .legacy_style import SLA_COLOURS

SET_KEYS = {"free": "free_xy", "mixed": "mixed_xy", "dwelling": "segments_xy"}


def draw_map(fig, run: dict, shown: tuple, title: str) -> None:
    """The Confined Map: the shown classes of tracks on the standard cell, the outline, the radius written under it."""
    ax = fig.add_subplot(111)
    for key in ("free", "mixed", "dwelling"):
        if key in shown:
            lines_with_gaps(ax, run[SET_KEYS[key]], color=SLA_COLOURS[key], linewidth=1.0 if key == "dwelling"
                            else 0.6, solid_capstyle="round")
    draw_outline(ax)
    cell_axes(ax, outer=True)
    ax.text(0.0, -0.55, f"Confinement radius: {run['radius_um'] * 1000:.4g} nm", fontsize=9)
    ax.set_title(f"{title} / {run['condition']}")
    fig.tight_layout()


def draw_transitions(fig, run: dict) -> None:
    t = run["transitions"]
    ax = fig.add_subplot(111)
    ax.bar(t["length"], t["free_to_confined"], color="#0072bd", label="free -> confined")
    ax.bar(t["length"], t["confined_to_free"], bottom=t["free_to_confined"], color="#d95319",
           label="confined -> free")
    ax.set_title("# transitions/tracklength")
    ax.set_xlabel("Track length [frames]")
    ax.set_ylabel(f"Avg # of transitions. (Total: {t['total']})")
    ax.legend(loc="upper right")
    fig.tight_layout()


def draw_dwell_events(fig, run: dict) -> None:
    e = run["dwell_events"]
    ax = fig.add_subplot(111)
    ax.bar(e["length"], e["mean"], color="#0072bd")
    ax.errorbar(e["length"], e["mean"], yerr=np.where(np.isfinite(e["sem"]), e["sem"], 0), fmt=".",
                color="#d95319")
    ax.set_title("# dwell events")
    ax.set_xlabel("Track length [frames]")
    ax.set_ylabel(f"Avg # of dwell events. #total: {e['total']}")
    fig.tight_layout()


def draw_msd(fig, run: dict) -> None:
    m = run["msd"]
    ax = fig.add_subplot(111)
    ax.errorbar(m["t"], m["msd"], yerr=m["err"], fmt="o", color="#0000ff", markerfacecolor="none",
                label="MSD ± sem")
    ax.plot(m["fine_t"], m["anomalous"], "-", color="#ff0000", linewidth=1.0,
            label=f"alpha (D$_{{alpha}}$) = {m['alpha']:.4g} ({m['D_alpha']:.4g} $\\mu$m$^2$/s$^{{\\alpha}}$)")
    ax.plot(m["linear_t"], m["linear"], "-", color="#00a000", linewidth=1.0,
            label=f"D= {m['D']:.4g} $\\mu$m$^2$/s")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"MSD [$\mu$m$^2$]")
    ax.set_title(m["title"])
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()


def draw_dwell_fit(fig, run: dict) -> None:
    fit = run["dwell_fit"]
    ax = fig.add_subplot(111)
    if fit is None:
        fig.clear()
        fig.text(0.5, 0.5, run["dwell_fit_error"] or "No dwell events.", ha="center", va="center", wrap=True,
                 color="#a00000")
        return
    ax.bar(fit["x"], fit["y"], width=0.8 * fit["tau_s"], color="#0072bd", label="data")
    lines = []
    if fit["single"]:
        s = fit["single"]
        ax.plot(fit["fine"], s["curve"], "-", color="#00ff00", linewidth=1.0, label="1-comp")
        for band in (s["upper"], s["lower"]):
            if band is not None:
                ax.plot(fit["fine"], band, ":", color="#00ff00", linewidth=0.8)
        lines.append(rf"$\tau$(1-comp) = {ls._num2str(s['Tau_1'], 2)} s ± {ls._num2str(s['sd'], 2)}")
    if fit["double"]:
        d = fit["double"]
        ax.plot(fit["fine"], d["curve"], "-", color="#ff0000", linewidth=1.0, label="2-comp")
        lines.append(rf"$\tau_1$(2-comp) = {ls._num2str(d['Tau_1'], 2)} s ± {ls._num2str(d['sd'][1], 2)} "
                     f"({ls._num2str(d['a'] * 100, 2)} ± {ls._num2str(100 * d['sd'][0], 2)} %)")
        lines.append(rf"$\tau_2$(2-comp) = {ls._num2str(d['Tau_2'], 2)} s ± {ls._num2str(d['sd'][2], 2)} "
                     f"({ls._num2str((1 - d['a']) * 100, 2)} ± {ls._num2str(100 * d['sd'][0], 2)} %)")
    lines.append(f"Confinement radius: {run['radius_um'] * 1000:.4g} nm")
    ax.text(0.4, 0.5, "\n".join(lines), transform=ax.transAxes, fontsize=8, va="center")
    ax.set_xlim(-0.01, float(fit["x"].max()))
    ax.set_ylim(0, 1)
    ax.set_xlabel(f"time [s]. (N of tracks with {run['min_steps']} steps = {run['n_tracks_with_min_steps']})")
    ax.set_ylabel("1 - ecdf")
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: ("data", "1-comp", "2-comp").index(labels[i]))
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper right")
    fig.tight_layout()


# -- the heat map window ------------------------------------------------------------------------------ #
def draw_heat_map(fig, data: dict, run: dict, shown: tuple, overlay: bool, title: str) -> None:
    """2.0's confinement heat map (jet, 0 to the largest density), the outline, the tracks over it if ``overlay``."""
    ax = fig.add_subplot(111)
    im = ax.imshow(data["image"], origin="lower", cmap=jet_cmap(), vmin=0.0, vmax=data["vmax"], aspect="auto",
                   extent=[data["x"][0], data["x"][-1], data["y"][0], data["y"][-1]], interpolation="nearest")
    clip_to_cell(im, ax)
    draw_outline(ax, linewidth=2.0)
    if overlay:
        for key in shown:
            lines_with_gaps(ax, run[SET_KEYS[key]], color="#e6e6e6", linewidth=0.4)
    cell_axes(ax)
    ax.set_title(f"{title} - {run['condition']}\n{data['title'].strip()}", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)


def draw_distribution(fig, d: dict, key: str, label: str, title: str) -> None:
    """One of the two normalised distributions beside the heat map (``key``: "x_distribution" or "y_distribution")."""
    ax = fig.add_subplot(111)
    centres = (d["edges"][:-1] + d["edges"][1:]) / 2
    ax.bar(centres, d["probability"], width=np.diff(d["edges"]), color="#0072bd", edgecolor="#262626",
           linewidth=0.4, label="Empirical probability")
    if key == "x_distribution":
        ax.set_xlim(-0.1, 1.1)
    ax.set_xlabel(label)
    ax.set_ylabel("probability")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()


def draw_movie_tracks(fig, movie: dict) -> None:
    """One movie of the "All Movies" gallery: outlines grey, every track blue, mixed green, dwelling red."""
    ax = fig.add_subplot(111)
    for polygon in movie["outlines"]:
        ax.plot(polygon[:, 0], polygon[:, 1], "-", color="#808080", linewidth=0.8)
    lines_with_gaps(ax, movie["all"], color=SLA_COLOURS["free"], linewidth=0.5)
    lines_with_gaps(ax, movie["mixed"], color=SLA_COLOURS["mixed"], linewidth=0.6)
    lines_with_gaps(ax, movie["segments"], color=SLA_COLOURS["dwelling"], linewidth=1.0, marker=".",
                    markersize=2)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel(r"[$\mu$m]")
    ax.set_ylabel(r"[$\mu$m]")
    ax.set_title(movie["movie"])
    fig.tight_layout()
