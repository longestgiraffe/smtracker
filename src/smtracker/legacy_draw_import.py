"""The legacy window's Import tab and Track Explorer, drawn: matplotlib only, no Qt (thin-Qt rule 1, `tests/test_ui_thin_qt.py`).

`ui.legacy.tab_import` and `ui.legacy.explorer` lay out widgets and call these inside `PlotPanel.draw(...)`. Every
function takes a figure and the plain payload `legacy_view` made (lists, dicts, numbers), paints it as the Qt classes
painted it before they moved here, and returns nothing the panel needs unless said. Numbers: `smtracker.legacy_view`.
"""
from __future__ import annotations

import numpy as np

from .legacy_style import boxplot_style
from .legacy_view import LegacyViewError

#: 2.0's colours on the overlay (`SMTracker.m`, manual Fig. 4): outlines white on an image, tracks blue.
OUTLINE_ON_IMAGE = "#ffffff"
OUTLINE_ON_WHITE = "#000000"
TRACK_INK = "#0000ff"
PICKED_INK = "#ff0000"
#: How far around the track the explorer's movie panel crops (px).
MOVIE_PAD_PX = 8


def _box(box) -> tuple:
    """``((x0, y0), (x1, y1))`` from the images payload's cell box, padded a little."""
    xs, ys = (box[0], box[2]), (box[1], box[3])
    pad = 0.15 * max(xs[1] - xs[0], ys[1] - ys[0], 1.0)
    return (xs[0] - pad, ys[0] - pad), (xs[1] + pad, ys[1] + pad)


def draw_lengths(fig, lengths: dict, title: str) -> None:
    """The cell-length boxplot, one box per condition."""
    ax = fig.add_subplot(111)
    names = list(lengths)
    data = [lengths[n] for n in names]
    ax.boxplot(data, vert=False, widths=0.5, **boxplot_style())
    ax.set_yticks(range(1, len(names) + 1), [f"{i}-{n}" for i, n in enumerate(names, start=1)])
    ax.set_xlabel("cell length [µm]")
    ax.set_title(title)
    fig.tight_layout()


def draw_overlay(fig, payload: dict, cell_index: int, track_index: int, show_cells: bool, show_tracks: bool) -> None:
    """The cells and tracks on the image (2.0's overlay); ``cell_index`` / ``track_index`` -1 = all."""
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.9])
    image = payload.get("image")
    if image and image.get("pixels"):
        pixels = np.frombuffer(image["pixels"], dtype=np.uint8).reshape(image["height"], image["width"])
        ax.imshow(pixels, cmap="gray", origin="upper", interpolation="nearest")
        outline_ink = OUTLINE_ON_IMAGE
    else:
        outline_ink = OUTLINE_ON_WHITE
        ax.set_facecolor("white")
        ax.invert_yaxis()
    if show_cells:
        for outline in payload["outlines"]:
            xs, ys = zip(*outline["points"]) if outline["points"] else ((), ())
            ax.plot(list(xs) + list(xs[:1]), list(ys) + list(ys[:1]), color=outline_ink, linewidth=0.8)
    if show_tracks:
        for i, track in enumerate(payload["tracks"]):
            xs, ys = zip(*track["points"])
            ax.plot(xs, ys, color=PICKED_INK if i == track_index else TRACK_INK,
                    linewidth=1.4 if i == track_index else 0.7)
    if 0 <= cell_index < len(payload["outlines"]):
        (x0, y0), (x1, y1) = _box(payload["outlines"][cell_index]["box"])
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
    elif not (image and image.get("pixels")):
        ax.autoscale()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_title(f"{payload['condition']} /{payload['movie']}")


def draw_spots_per_frame(fig, payload: dict) -> None:
    ax = fig.add_subplot(111)
    frames = [f for t in payload["tracks"] for f in t["frames"]]
    if frames:
        lo, hi = min(frames), max(frames)
        counts = np.bincount(np.asarray(frames) - lo, minlength=hi - lo + 1)
        ax.plot(np.arange(lo, hi + 1), counts, color="#0072bd")
    ax.set_xlabel("frame")
    ax.set_ylabel("# spots")
    ax.set_title(f"{payload['movie']}")
    fig.tight_layout()


def draw_track_lengths(fig, lengths, condition: str) -> None:
    """The histogram of the condition's track lengths (``lengths``: one frame count per track)."""
    lengths = np.asarray(lengths)
    ax = fig.add_subplot(111)
    if len(lengths):
        ax.hist(lengths, bins=np.arange(lengths.min(), lengths.max() + 2) - 0.5,
                color="#0072bd", edgecolor="#262626", linewidth=0.3)
    ax.set_xlabel("length of tracks [frames]")
    ax.set_ylabel("# tracks")
    ax.set_title(condition)
    fig.tight_layout()


def draw_all_tracks(fig, table, condition: str) -> None:
    """Every track of the condition on the standard cell (``table``: the projected track table)."""
    from .plots import draw_all_tracks_on_cell

    ax = fig.add_subplot(111)
    draw_all_tracks_on_cell(ax, table, colour=TRACK_INK)
    ax.set_title("", loc="left")                 # 2.0 named only the condition here
    ax.set_title(condition)
    fig.tight_layout()


# -- the Track Explorer's panels ------------------------------------------------------------------------- #
def draw_projection(fig, table, data: dict) -> None:
    from .plots import draw_cell_explorer

    ax = fig.add_subplot(111)
    if data["cell"] is None:
        raise LegacyViewError("this track lies in no cell outline")
    draw_cell_explorer(ax, table, cell=data["cell"], uid=data["uid"])
    ax.set_title("", loc="left")                     # 2.0 drew the cell alone, no caption line
    ax.set_xlabel("[µm]")
    ax.set_ylabel("[µm]")
    fig.tight_layout()


def draw_movement(fig, data: dict, radius) -> None:
    from matplotlib import colormaps
    from matplotlib.patches import Circle

    ax = fig.add_subplot(111)
    rel = np.asarray(data["relative_um"])
    ax.scatter(rel[:, 0], rel[:, 1], s=14, color="#0072bd", zorder=3)
    cmap = colormaps["autumn"]
    for i in range(len(rel) - 1):
        ax.plot(rel[i:i + 2, 0], rel[i:i + 2, 1], color=cmap(i / max(len(rel) - 1, 1)), linewidth=1.5)
    reach = float(np.abs(rel).max()) if len(rel) else 0.1
    if radius:
        ax.add_patch(Circle((0, 0), radius, fill=False, linestyle="--", linewidth=1.0, color="#d00000"))
        reach = max(reach, radius)
    reach = round(reach * 1.15, 2) or 0.1
    ax.set_xlim(-reach, reach)
    ax.set_ylim(-reach, reach)
    ax.set_aspect("equal")
    ax.set_xlabel("[µm]")
    ax.set_ylabel("[µm]")
    fig.tight_layout()


def draw_distance(fig, data: dict, radius) -> None:
    ax = fig.add_subplot(111)
    t = data["time_s"]
    ax.plot(t, data["origin_distance_um"], "-k", label="distance to origin")
    ax.plot(t, data["step_um"], "-g", label="step")
    if radius:
        ax.plot(t, np.full(len(t), radius), "--r")
    ax.grid(True, which="both")
    ax.minorticks_on()
    ax.set_ylabel("[µm]")
    ax.set_xlabel("time [s]")
    fig.tight_layout()


def draw_intensity(fig, data: dict) -> None:
    ax = fig.add_subplot(111)
    t, values = data["time_s"], data["intensity"]
    if not np.isfinite(values).any():
        raise LegacyViewError("this tracker recorded no intensity for the track")
    ax.plot(t, values, "-or", linewidth=0.25, markersize=4)
    ax.set_ylabel("[norm. aU]")
    ax.set_xlabel("time [s]")
    fig.tight_layout()


def draw_msd(fig, data: dict) -> None:
    ax = fig.add_subplot(111)
    lags = data["msd_lags_s"]
    for curve in data["msd_curves"]:
        ax.plot(lags[:len(curve)], curve, color="#9a9a9a", linewidth=0.6)
    if len(data["msd_mean"]):
        ax.plot(lags, data["msd_mean"], color="#000000", linewidth=1.5)
    ax.set_xlabel("time lag [s]")
    ax.set_ylabel("MSD [µm²]")
    fig.tight_layout()


def draw_movie_frame(fig, frames: dict, index: int, track: dict, frame_no: int) -> None:
    """One recorded frame with the track drawn up to it (``track``: ``legacy_view.track_in_pixels``)."""
    xs, ys = np.asarray(track["x"]), np.asarray(track["y"])
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.88])
    pixels = np.frombuffer(frames["frames"][int(index)], dtype=np.uint8).reshape(frames["height"], frames["width"])
    ax.imshow(pixels, cmap="gray", origin="upper", interpolation="nearest")
    shown = np.asarray(track["frame"]) <= frame_no
    ax.plot(xs[shown], ys[shown], color="#ff0000", linewidth=1.0)
    x0, x1 = xs.min() - MOVIE_PAD_PX, xs.max() + MOVIE_PAD_PX
    y0, y1 = ys.min() - MOVIE_PAD_PX, ys.max() + MOVIE_PAD_PX
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"frame {frame_no}")
