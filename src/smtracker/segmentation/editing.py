"""The mesh editor's MODEL: what a cell is while it is being edited, and every edit.

No Qt, no matplotlib, no drawing. Everything here is a plain function or a dataclass
over numpy, so the operations are covered headless and the window that hosts them adds
no untested logic: the same split this project already uses for `playback.TrackPlayer`
against `widgets.py`, and for `statemaps` against the GUI.

A cell being edited is a spine and a width, not a mesh. That is the whole design,
and it is what the browser editor could not do. Ribs are an OUTPUT: they follow from a
centreline and a width through `geometry.capsule_from_spine`, which is the one
implementation of that shape in this project. Editing the ribs directly is what forced
the old editor to rebuild every touched cell as a STRAIGHT capsule: its own comment
says the rungs are kept "so a correctly curved mesh is not replaced with a straight
one", and then `splitCellAt` calls `cellFromPoles`, which does exactly that. With a
spine, curvature survives an edit because there is nothing to reconstruct.

A loaded cell keeps its own ribs until it is edited. Deriving a spine from a mesh
and immediately rebuilding would resample and re-cap every cell on open, so a project
opened and saved with no edit would come back changed, which `test_cellmesh_write`
exists to guarantee it does not. So `ribs` holds what was read, `spine` is the handle,
and only an actual edit regenerates.

**What the first edit costs, measured, because it is not zero.** Regenerating a cell
that came from Oufti keeps the spine EXACTLY and keeps the body to **0.03 px**, but it
replaces Oufti's pole caps with hemispheres, worth up to **2.32 px (246 nm)** at the
very tip on a 10.5 px-wide cell, and the deviation is confined to the caps (nothing in
the middle 60 % of the cell exceeds 0.03 px). Every rib count also becomes
`DEFAULT_N_RUNGS` rather than the file's 28–37. That is a real change to which
localisations fall inside a cell near its poles, so `Cell.regenerated` records which
cells it happened to and the saved marker counts them. The alternative (editing ribs
in place) is what forced the browser editor to straighten every cell it touched, which
is a far larger error in the same place.

Every edit goes through `Document`, which owns the undo stack. An operation that
mutates a cell without recording an undo step is a bug the user finds at the worst
possible moment, so the operations are methods rather than free functions on a list.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .geometry import (GeometryError, capsule_from_spine,
                       resample_spine)

__all__ = [
    "EditError",
    "Cell",
    "Document",
    "DEFAULT_WIDTH_PX",
    "SPINE_CONTROL_POINTS",
    "MIN_CELL_LENGTH_FACTOR",
    "MERGE_GAP_PX",
    "STRAIGHT_TOLERANCE_PX",
    "chord_deviation",
    "cell_from_ribs",
]


class EditError(ValueError):
    """An edit that cannot be made, with a reason meant for the person making it."""


#: Width for a freshly drawn cell, in pixels. NOT a measurement: a starting point the
#: user drags. Real widths here run 6.75 to 12.0 px (measured over 132 Oufti cells), so
#: this sits mid-range rather than claiming to be typical of anything.
DEFAULT_WIDTH_PX = 9.0

#: Control points a spine recovered FROM RIBS is reduced to.
#:
#: Not the rib count. A mesh has 20-30 ribs and they carry sub-pixel noise; keeping one
#: control point per rib would make every wobble a handle and hand the user a spine that
#: is harder to straighten than to redraw. Five is enough for the curvature these cells
#: actually have (measured, the most curved real cell bows 0.067 of its length) and
#: few enough to grab.
SPINE_CONTROL_POINTS = 5

#: How far a loaded mesh's midline may bow from its own chord and still be STRAIGHT,
#: in pixels.
#:
#: MEASURED over 183 real Oufti cells: **177 of them bow exactly 0.000 px**: they were
#: made by a capsule generator and are straight to floating point, and the six that
#: bend at all bow 0.25 to 1.62 px. There is no population between, so the threshold is
#: not a judgement call; a quarter of a pixel sits in an empty gap and is far below the
#: localisation precision anyway.
#:
#: WITHOUT THIS EVERY CELL CAME BACK CURVED. `cell_from_ribs` reduced every midline to
#: `SPINE_CONTROL_POINTS`, so a cell annotated long before the curved tool existed
#: opened with five control points, three bend handles on it and the word "curved"
#: beside it in the list. Reported exactly that way.
STRAIGHT_TOLERANCE_PX = 0.25

#: A cell must be at least this many times its own width, end to end.
#:
#: Below one, the two end caps overlap and there is no body at all: the "capsule" is two
#: hemispheres intersecting, whose outline is not a cell shape. Refused rather than
#: drawn.
MIN_CELL_LENGTH_FACTOR = 1.0

#: Two cells further apart than this at their nearest poles are not neighbours.
#:
#: From the browser editor, where it was 12 px, kept so a merge behaves as it did. At
#: 106 nm/px that is 1.3 um, about half a cell length.
MERGE_GAP_PX = 12.0


@dataclass
class Cell:
    """One cell: the spine and width that DEFINE it, and the ribs currently drawn.

    `ribs` is `(N, 4)` `[x_left, y_left, x_right, y_right]` and `outline` is `(M, 2)`,
    both exactly as `smtracker.cellmesh` reads and writes them. They are regenerated
    from `spine` and `width` by :meth:`rebuilt`, and left alone until something edits
    the cell: see the module docstring for why that matters.
    """

    spine: np.ndarray                 # (K, 2) control points, K >= 2
    width: float
    ribs: np.ndarray                  # (N, 4)
    outline: np.ndarray               # (M, 2)
    cell_id: str | None = None
    #: True once this cell's geometry came from `capsule_from_spine` rather than from
    #: the file. Reported, not decorative: it is the difference between "we redrew this"
    #: and "this is what Oufti wrote".
    regenerated: bool = False

    @property
    def poles(self) -> tuple[np.ndarray, np.ndarray]:
        """The two ends of the spine, which are what a user drags."""
        return self.spine[0], self.spine[-1]

    @property
    def length_px(self) -> float:
        return float(np.hypot(*np.diff(self.spine, axis=0).T).sum())

    @property
    def is_curved(self) -> bool:
        return len(self.spine) > 2

    def rebuilt(self, *, spine=None, width=None, n_rungs: int | None = None) -> Cell:
        """A copy with new geometry built from the spine.

        The turn-radius refusal comes through, so an edit that would make a
        self-intersecting outline fails HERE, at the edit, rather than writing a bowtie
        into `cell_meshes/`.

        One exception type leaves this module. `GeometryError` is a `RuntimeError`
        and `EditError` a `ValueError`, so a caller that caught only one would show a
        traceback for an ordinary user action: bending a cell too far is a thing people
        will do all day. The geometry's own message is kept verbatim, because it is the
        better one: it names the two fixes that were measured to work.
        """
        spine = self.spine if spine is None else np.asarray(spine, dtype=float)
        width = self.width if width is None else float(width)
        span = float(np.hypot(*(spine[-1] - spine[0])))
        if span < width * MIN_CELL_LENGTH_FACTOR:
            raise EditError(
                f"a cell {width:.1f} px wide needs to be at least {width:.1f} px long "
                f"pole to pole, and this one is {span:.1f}. Shorter than that, the two end caps "
                f"overlap.")
        try:
            ribs, outline = capsule_from_spine(spine, width, n_rungs=n_rungs)
        except GeometryError as exc:
            raise EditError(str(exc)) from exc
        return replace(self, spine=spine, width=width, ribs=ribs, outline=outline,
                       regenerated=True)


def chord_deviation(midline) -> float:
    """How far a centreline bows from its own pole-to-pole chord, in pixels.

    The scale-free `spine_curvature` sagitta is the one to threshold biology on; this is
    the ABSOLUTE version, because "is this straight enough to hand the user two handles
    instead of five" is a question about pixels and noise, not about shape.
    """
    midline = np.asarray(midline, dtype=float)
    if len(midline) < 3:
        return 0.0
    chord = midline[-1] - midline[0]
    length = float(np.hypot(*chord))
    if length < 1e-9:
        return 0.0
    unit = chord / length
    rel = midline - midline[0]
    return float(np.abs(rel[:, 0] * unit[1] - rel[:, 1] * unit[0]).max())


def width_axis(cell: Cell):
    """`(centre, unit normal)` of the middle rib: the axis width is measured along.

    Width is measured across the cell, not as a radius. The first version took the
    distance from a spine point to the pointer, which meant dragging ALONG the cell
    changed its width too; and on a straight cell, whose spine has only two points,
    `spine[len // 2]` is the far POLE, so the width was measured from the end of the
    cell instead of its middle. Together those made the handle respond at ninety degrees
    to the direction it was pulled.
    """
    if len(cell.ribs) < 2:
        centre = (cell.spine[0] + cell.spine[-1]) / 2.0
        axis = cell.spine[-1] - cell.spine[0]
        length = float(np.hypot(*axis)) or 1.0
        along = axis / length
        return centre, np.array([-along[1], along[0]])
    row = cell.ribs[len(cell.ribs) // 2]
    left, right = np.array([row[0], row[1]]), np.array([row[2], row[3]])
    centre = (left + right) / 2.0
    across = right - left
    span = float(np.hypot(*across))
    if span < 1e-9:
        axis = cell.spine[-1] - cell.spine[0]
        length = float(np.hypot(*axis)) or 1.0
        along = axis / length
        return centre, np.array([-along[1], along[0]])
    return centre, across / span


def _width_of(ribs: np.ndarray) -> float:
    """The cell's body width: the WIDEST rib.

    Not the mean. The ribs taper to zero at both poles by construction, so a mean over
    all of them reports a value no part of the cell has, and it would shrink a cell a
    little every time it was opened and saved.
    """
    widths = np.hypot(ribs[:, 2] - ribs[:, 0], ribs[:, 3] - ribs[:, 1])
    finite = widths[np.isfinite(widths)]
    return float(finite.max()) if finite.size else 0.0


def cell_from_ribs(ribs, outline=None, cell_id: str | None = None) -> Cell:
    """Recover an editable cell from a mesh read off disk.

    The spine is the rib midline, reduced to :data:`SPINE_CONTROL_POINTS` evenly spaced
    by arc length. **The ribs are NOT rebuilt**, see the module docstring: a project
    opened and saved with no edit must come back byte-identical, and it does.
    """
    ribs = np.asarray(ribs, dtype=float)
    if ribs.ndim != 2 or ribs.shape[1] != 4 or len(ribs) < 2:
        raise EditError(f"need an (N>=2, 4) rib array, got {ribs.shape}")
    if not np.isfinite(ribs).all():
        raise EditError("this outline holds non-finite values, so it has no centreline. "
                        "The cell cannot be edited, only deleted.")
    midline = np.column_stack([(ribs[:, 0] + ribs[:, 2]) / 2.0,
                               (ribs[:, 1] + ribs[:, 3]) / 2.0])
    # A STRAIGHT CELL COMES BACK STRAIGHT, with two handles and not five. See
    # `STRAIGHT_TOLERANCE_PX`: giving every loaded cell the full set made every mesh
    # ever drawn read as curved.
    if chord_deviation(midline) <= STRAIGHT_TOLERANCE_PX:
        spine = np.vstack([midline[0], midline[-1]])
    else:
        n = min(SPINE_CONTROL_POINTS, len(midline))
        spine = resample_spine(midline, n) if n >= 2 else midline
    if outline is None:
        outline = np.vstack([ribs[:, 0:2], ribs[::-1, 2:4]])
    return Cell(spine=spine, width=_width_of(ribs), ribs=ribs,
                outline=np.asarray(outline, dtype=float), cell_id=cell_id)


@dataclass
class Document:
    """The cells of ONE field, plus undo. Every edit is a method here.

    `modified` is the flag the window shows and the one that decides whether closing
    needs a warning, so it is set by the operations rather than by the caller
    remembering to.
    """

    cells: list[Cell] = field(default_factory=list)
    modified: bool = False
    _undo: list[tuple[list[Cell], bool]] = field(default_factory=list, repr=False)
    _redo: list[tuple[list[Cell], bool]] = field(default_factory=list, repr=False)
    _gesture: bool = field(default=False, repr=False)

    #: Undo depth. Deep enough to cover a run of edits, bounded because each step holds
    #: every cell in the field.
    UNDO_LIMIT = 50

    # ── history ──────────────────────────────────────────────────────────────
    def _checkpoint(self) -> None:
        if self._gesture:
            # ONE UNDO STEP PER GESTURE, NOT PER MOUSE EVENT. A drag calls the edit
            # methods on every pointer move (sixty times a second) and without this
            # each one pushed its own undo entry, so Ctrl+Z walked back through a drag
            # a pixel at a time and you could not get behind it in any usable number of
            # presses. The browser editor calls `saveUndo()` once, at `mousedown`.
            return
        self._undo.append((list(self.cells), self.modified))
        del self._undo[:-self.UNDO_LIMIT]
        self._redo.clear()

    def begin_gesture(self) -> None:
        """Start a drag: take ONE checkpoint, then suppress the rest until it ends."""
        self._checkpoint()
        self._gesture = True

    def end_gesture(self) -> None:
        self._gesture = False

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append((list(self.cells), self.modified))
        self.cells, self.modified = self._undo.pop()
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append((list(self.cells), self.modified))
        self.cells, self.modified = self._redo.pop()
        return True

    # ── editing ──────────────────────────────────────────────────────────────
    def _check(self, index: int) -> Cell:
        if not 0 <= index < len(self.cells):
            raise EditError(f"no cell {index}: this field has {len(self.cells)}")
        return self.cells[index]

    def add_cell(self, spine, width: float = DEFAULT_WIDTH_PX) -> int:
        """Draw a new cell. Returns its index."""
        spine = np.asarray(spine, dtype=float)
        if spine.ndim != 2 or spine.shape[1] != 2 or len(spine) < 2:
            raise EditError(f"a spine is (K>=2, 2) points, got {spine.shape}")
        blank = Cell(spine=spine, width=float(width),
                     ribs=np.zeros((0, 4)), outline=np.zeros((0, 2)))
        cell = blank.rebuilt()
        self._checkpoint()
        self.cells.append(cell)
        self.modified = True
        return len(self.cells) - 1

    def delete_cell(self, index: int) -> None:
        self._check(index)
        self._checkpoint()
        del self.cells[index]
        self.modified = True

    def move_cell(self, index: int, dx: float, dy: float) -> None:
        """Translate a whole cell. This is what dragging its BODY does.

        Nothing is regenerated. A translation is rigid, so the spine, the ribs and
        the outline all move by the same vector and the SHAPE is untouched, which
        matters for a cell that came from Oufti: moving it must not silently replace its
        pole caps with hemispheres, the way any edit that goes through `rebuilt` does.
        `regenerated` therefore stays as it was.
        """
        cell = self._check(index)
        shift = np.array([float(dx), float(dy)])
        moved = replace(cell,
                        spine=cell.spine + shift,
                        ribs=cell.ribs + np.tile(shift, 2),
                        outline=cell.outline + shift)
        self._replace(index, moved)

    def move_pole(self, index: int, which: int, point) -> None:
        """Drag one end of the cell. ``which`` is 0 for the first pole, -1 for the last.

        The pole is a SPINE point, so on a curved cell the body between the poles keeps
        its bend, which is the edit the browser editor could not make without
        straightening the whole cell.
        """
        cell = self._check(index)
        if which not in (0, -1):
            raise EditError(
                f"a pole is spine point 0 or -1, not {which}. To drag a bend, use "
                "move_spine_point.")
        spine = cell.spine.copy()
        spine[which] = np.asarray(point, dtype=float)
        self._replace(index, cell.rebuilt(spine=spine))

    def move_spine_point(self, index: int, k: int, point) -> None:
        """Drag any control point, including the interior ones that make the curve."""
        cell = self._check(index)
        if not 0 <= k < len(cell.spine):
            raise EditError(f"no control point {k}: this spine has {len(cell.spine)}")
        spine = cell.spine.copy()
        spine[k] = np.asarray(point, dtype=float)
        self._replace(index, cell.rebuilt(spine=spine))

    def insert_spine_point(self, index: int, k: int, point) -> None:
        """Add a control point AFTER `k`, so a straight cell can be made to bend."""
        cell = self._check(index)
        if not 0 <= k < len(cell.spine) - 1:
            raise EditError(
                f"a point is inserted between two existing ones; {k} is not an "
                f"interior span of a spine with {len(cell.spine)} points")
        spine = np.insert(cell.spine, k + 1, np.asarray(point, dtype=float), axis=0)
        self._replace(index, cell.rebuilt(spine=spine))

    def delete_spine_point(self, index: int, k: int) -> None:
        """Remove an interior control point. The poles are not removable: a cell with
        one end is not a cell, and the user meant to delete it or move it."""
        cell = self._check(index)
        if k in (0, len(cell.spine) - 1):
            raise EditError("that is a pole, not a bend. Drag it, or delete the cell.")
        if len(cell.spine) <= 2:
            raise EditError("this cell is already straight: nothing to remove.")
        spine = np.delete(cell.spine, k, axis=0)
        self._replace(index, cell.rebuilt(spine=spine))

    def set_width(self, index: int, width: float) -> None:
        cell = self._check(index)
        if not width > 0:
            raise EditError(f"width must be positive, got {width!r}")
        self._replace(index, cell.rebuilt(width=float(width)))

    def set_width_from_edge(self, index: int, side: int, width: float) -> None:
        """Set the width by moving ONE edge (the one being dragged) not both.

        Asked for: "each handle should move the width in one direction (the one im
        dragging)". `set_width` is symmetric about the centreline, so pulling the right
        edge also moved the left one and the cell grew in a direction nobody dragged.
        Here the OPPOSITE edge stays exactly where it is, which means the centreline
        shifts by half the width change: a cell is defined by its spine and its width,
        so moving one edge is necessarily both.

        ``side`` is the handle index from `editor_view.width_handles`: 0 is the end of
        the middle rib the normal points AWAY from, 1 the end it points toward.

        ONE `_replace`, so a drag is one undo step and the width and the shift can never
        be applied apart: half of this edit alone is a cell in the wrong place.
        """
        cell = self._check(index)
        if not width > 0:
            raise EditError(f"width must be positive, got {width!r}")
        if side not in (0, 1):
            raise EditError(f"side must be 0 or 1 (a width handle), got {side!r}")
        _centre, normal = width_axis(cell)
        sign = -1.0 if side == 0 else 1.0
        shift = sign * normal * (float(width) - float(cell.width)) / 2.0
        self._replace(index, cell.rebuilt(spine=cell.spine + shift,
                                          width=float(width)))

    def split_cell(self, index: int, point, *, min_ribs: int = 3) -> tuple[int, int]:
        """Cut a cell at the septum nearest ``point``. Returns the two new indices.

        The curve survives the cut, which is the thing the browser editor said it
        was doing and did not. Its `splitCellAt` carries the comment "the rungs are kept
        ... so a correctly curved mesh is not replaced with a straight one" and then
        calls `cellFromPoles`, which builds two STRAIGHT capsules. Splitting a spine
        needs no such reconstruction: each half keeps the control points on its own side
        of the cut, plus the cut point as its new pole.
        """
        cell = self._check(index)
        dense = resample_spine(cell.spine, max(len(cell.spine) * 16, 64))
        point = np.asarray(point, dtype=float)
        cut = int(np.argmin(np.hypot(*(dense - point).T)))

        # A piece shorter than its own width has no body, so refuse in the terms the
        # user is thinking in: too near the end.
        span = np.hypot(*np.diff(dense, axis=0).T)
        along = np.concatenate([[0.0], np.cumsum(span)])
        if along[cut] < cell.width or along[-1] - along[cut] < cell.width:
            raise EditError(
                "that is too close to a pole: each half must be at least as long as "
                f"the cell is wide ({cell.width:.1f} px). Click nearer the middle.")

        head = _reduce(dense[:cut + 1])
        tail = _reduce(dense[cut:])
        self._checkpoint()
        try:
            a = cell.rebuilt(spine=head)
            b = cell.rebuilt(spine=tail)
        except (EditError, GeometryError) as exc:
            self._undo.pop()
            raise EditError(f"that split cannot be drawn: {exc}") from None
        self.cells[index:index + 1] = [a, b]
        self.modified = True
        return index, index + 1

    def merge_cells(self, i: int, j: int, *, gap_px: float = MERGE_GAP_PX) -> int:
        """Join two cells into one. Returns the surviving index.

        Joined at whichever pair of poles is closest, reversing either spine as needed
        so the result runs continuously from one free pole to the other.
        """
        a, b = self._check(i), self._check(j)
        if i == j:
            raise EditError("a cell cannot be merged with itself")
        options = [
            (float(np.hypot(*(a.spine[-1] - b.spine[0]))), a.spine, b.spine),
            (float(np.hypot(*(a.spine[-1] - b.spine[-1]))), a.spine, b.spine[::-1]),
            (float(np.hypot(*(a.spine[0] - b.spine[0]))), a.spine[::-1], b.spine),
            (float(np.hypot(*(a.spine[0] - b.spine[-1]))), a.spine[::-1], b.spine[::-1]),
        ]
        gap, head, tail = min(options, key=lambda o: o[0])
        if gap > gap_px:
            raise EditError(
                f"those two cells do not touch: their nearest poles are {gap:.1f} px "
                f"apart and the limit is {gap_px:.0f}. Merge cancelled.")
        spine = _reduce(np.vstack([head, tail]))
        width = (a.width + b.width) / 2.0
        self._checkpoint()
        try:
            merged = a.rebuilt(spine=spine, width=width)
        except (EditError, GeometryError) as exc:
            self._undo.pop()
            raise EditError(f"those two cannot be joined into one shape: {exc}") from None
        keep, drop = min(i, j), max(i, j)
        del self.cells[drop]
        self.cells[keep] = merged
        self.modified = True
        return keep

    def _replace(self, index: int, cell: Cell) -> None:
        # The caller has already built `cell`, so a refused edit raised before getting
        # here and no undo step was taken for an edit that did not happen.
        self._checkpoint()
        self.cells[index] = cell
        self.modified = True


def _reduce(polyline: np.ndarray, n: int = SPINE_CONTROL_POINTS) -> np.ndarray:
    """A dense polyline back to a handful of evenly spaced control points."""
    polyline = np.asarray(polyline, dtype=float)
    if len(polyline) <= 2:
        return polyline
    return resample_spine(polyline, min(n, len(polyline)))
