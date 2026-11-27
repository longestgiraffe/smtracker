"""The mesh editor's VIEW: drawing and hit testing, with no Qt anywhere.

matplotlib only, exactly as `smtracker.plots` and `smtracker.playback` are, so the
window that hosts this adds no untested logic, a click that picks the wrong cell fails
HERE, in a headless test, rather than by looking subtly wrong on screen.

Hit testing is geometry and belongs outside the widget. `viewer.nearest_track`
already established that in this project: pure numpy, click in, a hit or `None` out. The
same rule holds here and for the same reason, a viewer that always selects *something*
teaches the user to distrust it, so everything below returns `None` past its tolerance
rather than the least-bad answer.

Tolerances are in screen pixels, not image pixels. A grab radius fixed in image
coordinates would shrink as the user zoomed in (the one moment they are trying to be
precise) and swell to cover the whole cell when zoomed out. The caller passes the
current scale and the tolerance is converted, so a handle is the same size under the
mouse at every zoom.

Drawing is blitted, as `playback.TrackPlayer` is: the phase image is a static
background drawn once, and only the cell artists are redrawn on a drag. That is what
makes dragging a pole feel attached to the mouse rather than lagging behind it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: `width_axis` LIVES IN `editing`, because the one-sided width drag is a document
#: operation and the axis is the geometry it needs. Re-exported here, where it was
#: first written and where callers already reach for it.
from .editing import Cell, Document, width_axis

__all__ = [
    "Hit",
    "HANDLE_GRAB_PX",
    "BODY_GRAB_PX",
    "DISPLAY_MODES",
    "VIEWS",
    "stretch",
    "adjust",
    "compose",
    "hit_test",
    "nearest_cell",
    "cell_artists",
    "draw_document",
    "preview_capsule",
]

#: The three channel views, in the order their keys 1/2/3 select them.
VIEWS = ("phase", "sum", "overlay")

#: Tints available for the fluorescence channel in the overlay, as RGB weights.
#:
#: RED IS THE DEFAULT because it was asked for. MAGENTA/GREEN IS THE ACCESSIBLE PAIR and
#: is offered beside it: red-on-grey is perfectly legible to most people and red is the
#: one hue a deuteranope loses against a grey background, so the choice costs nothing to
#: provide and the default costs nothing to change.
OVERLAY_COLOURS = {
    "red": (1.0, 0.0, 0.0),
    "magenta": (1.0, 0.0, 1.0),
    "green": (0.0, 1.0, 0.0),
    "cyan": (0.0, 1.0, 1.0),
    "yellow": (1.0, 1.0, 0.0),
}

#: How near a handle the click must be, in SCREEN pixels. Roughly a fingertip on a
#: trackpad; below about 6 the handles become hard to grab at any zoom.
HANDLE_GRAB_PX = 9.0

#: How near the CENTRELINE a click selects the cell body, in screen pixels. Larger than
#: a handle because the body is a target you aim at rather than one you pick out.
BODY_GRAB_PX = 14.0

#: Percentile windows for the phase display, carried over from the browser editor where
#: they were chosen against these images.
#:
#: No single stretch serves both jobs: a wide window keeps the cell/background edge crisp
#: but saturates the interior, hiding the septum; a window ending at the background level
#: spreads the cell's own dark range over the whole display, which is where septal
#: contrast lives. "imagej" is what ImageJ shows on opening a 16-bit image.
DISPLAY_MODES = {
    "imagej": (0.1, 99.9),
    "outline": (2.0, 98.0),
    "septa": (0.2, 50.0),
}


@dataclass(frozen=True)
class Hit:
    """What is under the cursor.

    ``kind`` is ``"pole"``, ``"bend"``, ``"width"`` or ``"body"``. ``point`` is the spine
    control point index for a pole or a bend, the HANDLE index (0 or 1) for a width, and
    ``None`` otherwise.

    WHICH width handle matters, because each one moves its own edge and leaves the other
    where it is. Reporting only "a width handle" made the two indistinguishable, so the
    drag could not tell which edge the pointer had hold of.
    """

    cell: int
    kind: str
    point: int | None = None
    distance_px: float = 0.0


def stretch(image, mode: str = "septa"):
    """One frame to a 0-1 float array for display, by percentile window.

    Returns the array and the (low, high) intensities used, because a display window
    that is not reported cannot be reproduced, and on these images the window decides
    whether a septum is visible at all.
    """
    if mode not in DISPLAY_MODES:
        raise ValueError(f"mode must be one of {sorted(DISPLAY_MODES)}, got {mode!r}")
    image = np.asarray(image, dtype=float)
    if image.ndim == 3:
        image = image[0]
    low, high = np.percentile(image, DISPLAY_MODES[mode])
    scaled = np.clip((image - low) / max(high - low, 1e-9), 0.0, 1.0)
    return scaled, (float(low), float(high))


def adjust(scaled, brightness: float = 100.0, contrast: float = 100.0):
    """Brightness and contrast on an already-stretched 0-1 image, in per cent.

    Css semantics on purpose, because the browser editor's controls are a CSS
    `filter: brightness(b%) contrast(c%)` and the two editors must not disagree about
    what the same slider position means:

        brightness(b):  out = x * b
        contrast(c):    out = (x - 0.5) * c + 0.5

    applied in that order, which is the order the filter string writes them.
    """
    b, c = float(brightness) / 100.0, float(contrast) / 100.0
    out = np.asarray(scaled, dtype=float) * b
    if c != 1.0:
        out = (out - 0.5) * c + 0.5
    return np.clip(out, 0.0, 1.0)


def compose(phase=None, sum_image=None, *, view: str = "phase", opacity: float = 0.5,
            mode: str = "septa", brightness: float = 100.0, contrast: float = 100.0,
            colour: str = "red"):
    """The image to draw: one channel, or a channel merge. Returns a 0-1 array or None.

    Greyscale `(H, W)` for a single channel; **RGB `(H, W, 3)` for the overlay**, where
    the phase image is the grey ground and the fluorescence is TINTED and added on top.

    Added, not averaged. A weighted average darkens the phase image everywhere the
    fluorescence is dim, so raising the slider fades the thing the outline is being
    checked against. Adding leaves the phase at full strength and makes the signal glow
    on it, which is what a channel merge means everywhere else and what the eye expects.
    The slider then scales the tint alone, from invisible to saturated.

    A view whose channel is missing falls back to the one that exists, because a field
    with no `sum/` is ordinary and a black canvas reads as a broken viewer.
    """
    if view not in VIEWS:
        raise ValueError(f"view must be one of {VIEWS}, got {view!r}")
    if colour not in OVERLAY_COLOURS:
        raise ValueError(f"colour must be one of {sorted(OVERLAY_COLOURS)}, "
                         f"got {colour!r}")
    first = None if phase is None else stretch(phase, mode)[0]
    second = None if sum_image is None else stretch(sum_image, mode)[0]

    if view == "phase" or second is None:
        chosen = first if first is not None else second
    elif view == "sum" or first is None:
        chosen = second
    elif first.shape != second.shape:
        # Different crops of the same field cannot be merged pixel for pixel, and
        # stretching one onto the other would INVENT an alignment. The phase is shown.
        chosen = first
    else:
        ground = adjust(first, brightness, contrast)
        weight = min(max(float(opacity), 0.0), 1.0)
        tint = np.asarray(OVERLAY_COLOURS[colour], dtype=float)
        merged = ground[..., None] + weight * second[..., None] * tint
        return np.clip(merged, 0.0, 1.0)

    if chosen is None:
        return None
    return adjust(chosen, brightness, contrast)


def preview_capsule(start, end, width: float):
    """The outline of the cell a second click would create, or ``None`` if too short.

    Drawn while the pointer moves in draw mode, so a user sees the cell BEFORE
    committing it: the browser editor has this and its absence was the first thing
    reported about the Qt one. It calls the same `capsule_from_spine` the commit does,
    so the preview cannot promise a shape the commit will not make.
    """
    from .geometry import GeometryError, capsule_from_spine

    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    length = float(np.hypot(*(end - start)))
    if length < float(width):
        return None, length
    try:
        _mesh, outline = capsule_from_spine([start, end], float(width))
    except GeometryError:
        return None, length
    return outline, length


def width_handles(cell: Cell) -> np.ndarray:
    """BOTH ends of the middle rib: `(2, 2)`, one handle on each side of the cell.

    Two, not one. A single handle can only be pulled from one side, so half the time
    the grab is on the far edge from the pointer and the cell appears to have no handle
    there at all: reported exactly that way.

    On the RIB rather than at a fixed offset from the spine, so on a curved cell they
    stay on the cell's own edge instead of floating off it at the bend.
    """
    if len(cell.ribs) < 2:
        middle = (cell.spine[0] + cell.spine[-1]) / 2.0
        return np.array([middle, middle], dtype=float)
    row = cell.ribs[len(cell.ribs) // 2]
    return np.array([[row[0], row[1]], [row[2], row[3]]], dtype=float)


def _width_handle(cell: Cell) -> np.ndarray:
    """The nearer of the two handles, kept for callers that want a single point."""
    return width_handles(cell)[0]


def _distance_to_polyline(point: np.ndarray, line: np.ndarray) -> float:
    """Shortest distance from a point to a polyline, measured to the SEGMENTS.

    To the segments, not to the vertices: a spine has five control points over ~30 px,
    so vertex distance would make the middle of every span unclickable.
    """
    if len(line) == 1:
        return float(np.hypot(*(point - line[0])))
    a, b = line[:-1], line[1:]
    ab = b - a
    length2 = (ab ** 2).sum(axis=1)
    length2 = np.where(length2 > 1e-12, length2, 1e-12)
    t = np.clip(((point - a) * ab).sum(axis=1) / length2, 0.0, 1.0)
    nearest = a + t[:, None] * ab
    return float(np.hypot(*(point - nearest).T).min())


def hit_test(document: Document, point, *, scale: float = 1.0,
             handle_px: float = HANDLE_GRAB_PX,
             body_px: float = BODY_GRAB_PX,
             selected: int | None = None) -> Hit | None:
    """What the click at ``point`` (IMAGE coordinates) grabs, or ``None``.

    ``scale`` is screen pixels per image pixel, so the tolerances stay constant on
    screen at every zoom: see the module docstring.

    Handles beat bodies, and the selected cell beats its neighbours. Cells touch,
    and their grab regions overlap at exactly the place a user works, the septum. Two
    ties broken deliberately rather than by list order: a handle is a small target the
    user aimed at, so it wins over any body; and among equals the cell already selected
    wins, because otherwise dragging a pole at a junction silently jumps to the
    neighbour and edits the wrong cell.
    """
    point = np.asarray(point, dtype=float)
    scale = float(scale) if scale > 0 else 1.0
    handle_tol = handle_px / scale
    body_tol = body_px / scale

    handles: list[Hit] = []
    bodies: list[Hit] = []
    for index, cell in enumerate(document.cells):
        for k, node in enumerate(cell.spine):
            distance = float(np.hypot(*(point - node)))
            if distance > handle_tol:
                continue
            is_pole = k in (0, len(cell.spine) - 1)
            handles.append(Hit(cell=index, kind="pole" if is_pole else "bend",
                               point=k, distance_px=distance * scale))
        along = _distance_to_polyline(point, cell.spine)
        # BOTH handles are tested and the nearer one wins, so the edge is grabbable from
        # whichever side the pointer is on.
        grips = width_handles(cell)
        reach = [float(np.hypot(*(point - g))) for g in grips]
        nearer = 0 if reach[0] <= reach[1] else 1
        distance = reach[nearer]
        # NEARER THE EDGE THAN THE MIDDLE, or it is not a grab at the edge.
        #
        # The width handle sits ON the membrane, which is the physical affordance: you
        # pull the edge to make the cell wider. But these cells are 6.75 to 12 px wide,
        # so the handle is 3 to 6 px from the centreline while the grab radius is 9
        # SCREEN px: zoomed out far enough, the handle's circle covers the whole cell
        # and the body becomes unclickable. Found by the test, not by looking.
        #
        # Fixed with a rule rather than a fudge factor: a click grabs the edge when it
        # is closer to the edge than to the centreline. At working zoom the two never
        # compete; zoomed out, the body stays reachable.
        if distance <= handle_tol and distance < along:
            handles.append(Hit(cell=index, kind="width", point=nearer,
                               distance_px=distance * scale))

        if along <= max(body_tol, cell.width / 2.0):
            bodies.append(Hit(cell=index, kind="body", distance_px=along * scale))

    def rank(hit: Hit):
        # The selected cell first, then whichever is nearest.
        return (0 if hit.cell == selected else 1, hit.distance_px)

    if handles:
        return min(handles, key=rank)
    if bodies:
        return min(bodies, key=rank)
    return None


def nearest_cell(document: Document, point, *, scale: float = 1.0,
                 body_px: float = BODY_GRAB_PX) -> int | None:
    """Just the cell index under ``point``, or ``None``. For a plain click-to-select."""
    hit = hit_test(document, point, scale=scale, body_px=body_px)
    return None if hit is None else hit.cell


def cell_artists(ax, cell: Cell, *, selected: bool = False, show_handles: bool = True,
                 colour: str | None = None):
    """Every artist for ONE cell, returned so the caller can blit them.

    Returned, not just drawn. The window redraws only these on a drag and leaves the
    phase image alone; handing them back is what makes that possible without the view
    knowing anything about Qt or about the window's cache.
    """
    import matplotlib.lines as mlines

    artists = []
    edge = colour or ("#ffd400" if selected else "#00e5a0")
    outline = cell.outline
    if len(outline) >= 3:
        closed = np.vstack([outline, outline[:1]])
        artists.append(ax.add_line(mlines.Line2D(
            closed[:, 0], closed[:, 1], color=edge,
            lw=2.0 if selected else 1.3, zorder=4)))
    if selected:
        # The SPINE is drawn only for the selected cell. Drawn for every cell it turns
        # a field of eighty into a thicket, and the spine is only actionable on the one
        # being edited anyway.
        artists.append(ax.add_line(mlines.Line2D(
            cell.spine[:, 0], cell.spine[:, 1], color=edge, lw=1.0, ls="--",
            alpha=0.8, zorder=5)))
        if show_handles:
            poles = cell.spine[[0, -1]]
            artists.append(ax.add_line(mlines.Line2D(
                poles[:, 0], poles[:, 1], ls="none", marker="o", ms=7.5,
                mfc=edge, mec="black", mew=1.0, zorder=6)))
            if len(cell.spine) > 2:
                bends = cell.spine[1:-1]
                artists.append(ax.add_line(mlines.Line2D(
                    bends[:, 0], bends[:, 1], ls="none", marker="s", ms=6.0,
                    mfc="white", mec=edge, mew=1.6, zorder=6)))
            # BOTH handles are DRAWN, not just hit-tested. Making the second one
            # grabbable without drawing it left the cell looking as though it had one
            # edge handle, which is what was reported, the hit test and the drawing are
            # the same affordance and have to be changed together.
            grips = width_handles(cell)
            artists.append(ax.add_line(mlines.Line2D(
                grips[:, 0], grips[:, 1], ls="none", marker="D", ms=6.0,
                mfc="#ff6b35", mec="black", mew=0.8, zorder=6)))
    return artists


def draw_document(ax, document: Document, image=None, *, selected: int | None = None,
                  mode: str = "septa", show_handles: bool = True,
                  already_scaled: bool = False):
    """The whole field: the image, then every cell. Returns the cell artists.

    ``already_scaled`` says the caller has run :func:`compose` and the array is a 0-1
    display image, which is how the window drives it, so brightness, contrast and the
    channel blend all live in one place instead of being reapplied here.

    The image is drawn with ``origin="upper"`` and an extent in image pixels, so a mesh
    coordinate and a pixel coordinate are the same number. Every mesh in this project is
    in camera pixels and `cellmesh` assigns tracks in them, so a display that rescaled
    would put the outlines somewhere the analysis does not agree with.
    """
    if image is not None:
        scaled = (np.asarray(image, dtype=float) if already_scaled
                  else stretch(image, mode)[0])
        height, width = scaled.shape[:2]
        # An RGB array takes NO cmap and NO vmin/vmax, matplotlib reads three channels
        # as colour directly, and passing a colormap alongside them is an error.
        colour_kw = ({} if scaled.ndim == 3
                     else {"cmap": "gray", "vmin": 0.0, "vmax": 1.0})
        ax.imshow(scaled, origin="upper", zorder=0,
                  extent=[0.0, float(width), float(height), 0.0],
                  interpolation="nearest", **colour_kw)
        ax.set_xlim(0.0, float(width))
        ax.set_ylim(float(height), 0.0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    artists: list = []
    for index, cell in enumerate(document.cells):
        artists.extend(cell_artists(ax, cell, selected=index == selected,
                                    show_handles=show_handles))
    return artists
