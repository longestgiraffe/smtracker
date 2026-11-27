"""The legacy window's MSD tab and its Brownian-vs-Anomalous window, drawn: matplotlib only, no Qt (thin-Qt rule 1,
`tests/test_ui_thin_qt.py`).

`ui.legacy.tab_msd` lays out widgets and calls these inside `PlotPanel.draw(...)`. Numbers: `legacy_view_msd`.
"""
from __future__ import annotations

import numpy as np

from . import legacy_view_msd as lm

#: 2.0's mean MSD plot labels, in mathtext so every glyph is in the font.
MSD_LABEL = r"MSD [$\mu$m$^{2}$]"
LAG_LABEL = r"time lag [$\tau$]"
#: `plotMSDTracks.m:48`: the y range of the per-track panel.
TRACKS_YLIM = (-0.05, 1.2)
#: MATLAB's parula, from its anchor colours (matplotlib has none).
PARULA = ((0.2422, 0.1504, 0.6603), (0.2810, 0.3228, 0.9579), (0.1786, 0.5289, 0.9682), (0.0689, 0.6948, 0.8394),
          (0.2161, 0.7843, 0.5923), (0.6720, 0.7793, 0.2227), (0.9970, 0.7659, 0.2199), (0.9769, 0.9839, 0.0805))
GUIDE_STYLES = ((0.1, ":", "#0000ff"), (1.0, "-.", "#0000ff"), (lm.FP_DIFFUSION, "--", "#ff0000"))


def parula():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("parula", PARULA, N=256)


def draw_msd(fig, run: dict, shown: list, colours: dict, limits, title: str):
    """The mean squared displacement plot. ``limits``: ``(xmin, xmax, ymin, ymax)`` typed by the user, or None.
    Returns ``(xlim, ylim)`` as the axes autoscaled them when ``limits`` is None (the controls' fields show them),
    else None."""
    ax = fig.add_subplot(111)
    styles = {"xy": "-", "x": "--", "y": ":"}
    for condition in shown:
        colour = colours.get(condition, "#0072bd")
        for d in run["directions"]:
            e = run["ensemble"][condition][d]
            curve = e["curve"]
            err = curve[:, 2] / np.sqrt(curve[:, 3])
            ax.errorbar(curve[:, 0], curve[:, 1], yerr=np.where(np.isfinite(err), err, 0.0), color=colour,
                        linewidth=1.5, linestyle=styles[d], capsize=3, marker=None,
                        label=f"{condition} -{d.upper()}")
    x_max = ax.get_xlim()[1] if limits is None else limits[1]
    x = np.arange(0.0, x_max + 0.001, 0.001)
    for condition in shown:
        for d in run["directions"]:
            fit = run["ensemble"][condition][d]["fit"]
            ax.plot(x, fit["slope"] * x + fit["intercept"], color="#000000", linewidth=0.8, label="_nolegend_")
    autoscaled = None
    if limits is not None:
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
    else:
        autoscaled = (ax.get_xlim(), ax.get_ylim())
    ax.set_title(title)
    ax.grid(True)
    ax.set_xlabel(LAG_LABEL)
    ax.set_ylabel(MSD_LABEL)
    ax.legend(loc="upper left")
    return autoscaled


def draw_curves(fig, tracks: dict, ticked, title: str) -> None:
    """Every ticked track's MSD curve with the D = 0.1, 1 and free-diffusion guide lines."""
    ax = fig.add_subplot(111)
    for i in np.flatnonzero(np.asarray(ticked, dtype=bool)):
        t, y = tracks["curves"][i]
        ax.plot(t, y, color=(0, 0, 0, 0.1))
    x = np.linspace(0.0, max(ax.get_xlim()[1], 0.2), 50)
    for d, style, colour in GUIDE_STYLES:
        ax.plot(x, 4 * d * x, linestyle=style, color=colour, linewidth=0.9)
    ax.set_ylim(*TRACKS_YLIM)
    ax.set_xlabel(LAG_LABEL)
    ax.set_ylabel(MSD_LABEL)
    ax.grid(True)
    ax.set_title(title, fontsize=8, pad=3)
    ax.legend(["MSD curves", "D=0.1", "D=1", f"D={lm.FP_DIFFUSION:g}"], loc="upper left", fontsize=7)


def draw_loc_error(fig, tracks: dict, ticked) -> str:
    """The histogram of the localisation errors of the ticked tracks; returns the title of the box around it
    ("Localization error estimation", with the mean in nm where there is one)."""
    ax = fig.add_subplot(111)
    try:
        info = lm.localisation_error(tracks, ticked)
    except lm.LegacyViewError as exc:
        ax.text(0.5, 0.5, str(exc), ha="center", va="center", transform=ax.transAxes, color="#555555")
        ax.set_axis_off()
        return "Localization error estimation"
    ax.bar(info["edges"][:-1], info["counts"], width=lm.LOC_ERR_BIN_UM, align="edge", color="#0072bd",
           edgecolor="#262626", linewidth=0.5)
    ax.set_xlabel(r"Localization error [$\mu$m]")
    ax.set_ylabel("events [ counts]")
    ax.grid(True)
    return f"Localization error estimation: {info['mean_nm']:.2f} nm"


def draw_anomalous(fig, s: dict) -> None:
    """2.0's auxiliary window of anomalous analysis results (manual Fig. 8)."""
    from matplotlib.gridspec import GridSpec

    gs = GridSpec(5, 2, figure=fig, hspace=0.9, wspace=0.35, left=0.09, right=0.96, top=0.96, bottom=0.05)
    blue = "#0072bd"
    counts, edges = s["alpha_hist"]
    ax = fig.add_subplot(gs[0, 0])
    ax.barh(edges[:-1], counts, height=0.1, align="edge", color=blue, edgecolor="#262626", linewidth=0.4)
    ax.set_ylim(0, 2)
    ax.set_title("Histogram for the alpha exponent")
    ax.set_ylabel("alpha")
    ax.set_xlabel("Probabilty")
    counts, edges = s["r2_hist"]
    ax = fig.add_subplot(gs[0, 1])
    ax.barh(edges[:-1], counts, height=0.1, align="edge", color=blue, edgecolor="#262626", linewidth=0.4)
    ax.set_title("Histogram for the R-Squared values")
    ax.set_ylabel("R-Squared")
    ax.set_xlabel("Probabilty")
    ax = fig.add_subplot(gs[1, 0])
    labels = list(s["share"])
    ax.bar(labels, [s["share"][k] for k in labels], color=blue)
    ax.set_title(f"Number of steps :{s['min_length']}")
    ax.set_ylabel("Probability")
    ax.set_xlabel("Type of motion")
    ax = fig.add_subplot(gs[1, 1])
    ax.scatter(s["D"], s["alpha"], s=8, facecolors="none", edgecolors=blue, linewidths=0.6)
    ax.set_xlabel(r"D$_{\alpha}$")
    ax.set_ylabel("alpha")
    ax = fig.add_subplot(gs[2, :])
    heat = s["heat"]
    x0, x1 = heat["x_edges"]
    y0, y1 = heat["y_edges"]
    image = ax.imshow(heat["grid"], origin="lower", extent=(x0, x1, y0, y1), aspect="auto", cmap=parula())
    ax.set_ylim(0, 2)
    ax.set_xlabel(r"D$_{\alpha}$")
    ax.set_ylabel("alpha")
    fig.colorbar(image, ax=ax)
    ax = fig.add_subplot(gs[3, :])
    order = np.argsort(s["density"])
    sc = ax.scatter(s["D"][order], s["alpha"][order], c=s["density"][order], s=6, cmap=parula())
    ax.set_xlabel(r"D$_{\alpha}$")
    ax.set_ylabel("alpha")
    fig.colorbar(sc, ax=ax)
    ax = fig.add_subplot(gs[4, :])
    if s["brownian_D_hist"] is None:
        ax.text(0.5, 0.5, "no track was taken as Brownian", ha="center", va="center", transform=ax.transAxes,
                color="#555555")
    else:
        counts, edges = s["brownian_D_hist"]
        ax.bar(edges[:-1], counts, width=0.05, align="edge", color=blue, edgecolor="#262626", linewidth=0.4)
    ax.set_title("Diffusion Histogram for the brownian tracks")
    ax.set_ylabel("Probabilty")
    ax.set_xlabel("Diffusion")
