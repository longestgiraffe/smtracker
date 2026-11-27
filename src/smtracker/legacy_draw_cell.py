"""Drawing on 2.0's standard cell, shared by the legacy window's SLA and SDA tabs (headless: no Qt, thin-Qt rule 1).

2.0 drew every track and map in a model cell of 3 x 1 um (axes x in [-0.1, 1.1] over the cell's length, y in
[-0.6, 0.6] over its width, `pbaspect [3 1 1]`, the outline in black, `Help_fcn/plotHeatmap.m`, `@SMT/confTracks.m`).
The numbers come from `smtracker.legacy_view_sla` / `legacy_view_sda`; this module only paints; `ui.legacy` calls it.
"""
from __future__ import annotations

import numpy as np

from .legacy_view_sda import model_cell_contour

CELL_LENGTH_UM = 3.0


def jet_cmap():
    """MATLAB's ``jet(1024)``."""
    import matplotlib

    return matplotlib.colormaps["jet"].resampled(1024)


def hot_reversed_cmap():
    """2.0's SDA colour map, ``flip(hot(1024))`` (white at the bottom, black at the top)."""
    from matplotlib.colors import ListedColormap

    from .colour_ramps import V2_HOT_N, matlab_hot

    return ListedColormap(np.asarray(matlab_hot(V2_HOT_N), dtype=float)[::-1], name="hot_flipped")


def cell_axes(ax, *, outer: bool = False) -> None:
    """The axes 2.0 gave a map: x [-0.1, 1.1], y [-0.6, 0.6], boxed 3 : 1. ``outer`` is the SLA map's (ticks in um,
    `confTracks.m:107-114`); without it the map's own (no ticks, `plotHeatmap.m:130-135`)."""
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.6, 0.6)
    ax.set_box_aspect(1.0 / 3.0)
    if outer:
        ax.set_xticks([0, 1 / 3, 2 / 3, 1], ["0", "1", "2", "3"])
        ax.set_yticks([-0.5, 0, 0.5])
        ax.set_xlabel(r"[$\mu$m]")
        ax.set_ylabel(r"[$\mu$m]")
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_xlabel("Long axis")
        ax.set_ylabel("Short axis")


def draw_outline(ax, *, linewidth: float = 1.0) -> np.ndarray:
    """The model cell's outline in black; returns the polygon."""
    contour = model_cell_contour()
    ax.plot(contour[:, 0], contour[:, 1], "-", color="#000000", linewidth=linewidth)
    return contour


def clip_to_cell(artist, ax) -> None:
    """Show an image only inside the model cell (2.0 filled the outside white)."""
    from matplotlib.patches import Polygon

    patch = Polygon(model_cell_contour(), closed=True, transform=ax.transData, facecolor="none", edgecolor="none")
    ax.add_patch(patch)
    artist.set_clip_path(patch)


def lines_with_gaps(ax, xy: np.ndarray, **style):
    """One polyline for every track, the tracks apart by the NaN rows `prepareToplotAllTracks` leaves."""
    if len(xy):
        return ax.plot(xy[:, 0], xy[:, 1], **style)
    return []
