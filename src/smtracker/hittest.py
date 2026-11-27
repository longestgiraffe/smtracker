"""Canvas hit-tests, map a click position to the data identity it lands on.

The one rule here, learned the hard way: a hit-test must be derived from the
plotted geometry, never from an assumed axis orientation. The old drill routed
every view through ``condition_at_x`` ("conditions live at integer x") which is
true for the column views (bubble/ridgeline/kinetics/fraction) but WRONG for the
heatmap (conditions on Y), saSPT (x is D) and the forest (x is Δ). A click there
resolved to a different condition than the one under the cursor: the UI
confidently acting on something other than what is shown, the same failure
family as an export re-deriving from a live picker instead of the plotted data.

So each ``draw_*`` builds the matching hit-test from the exact labels/positions it
just drew and attaches it to the figure (``figure._smtracker_hit_test``). The Qt
layer only calls ``.pick(x, y)`` and drills on the returned :class:`Selection`.
Every class here is pure and unit-tested headless, per the thin-Qt rule, a wrong
drill fails a test, not subtly on screen.

Coordinate note: these read matplotlib DATA coordinates (``event.xdata`` /
``event.ydata``), which are orientation-independent. An imshow with the default
``origin="upper"`` visually flips rows, but row ``i`` still sits at data ``y = i``
so reading ``round(ydata)`` recovers the right row regardless of the flip. That
is precisely why deriving from data coordinates, not from where a label visually
sits, is the robust choice.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Selection:
    """What a canvas click resolved to.

    ``condition`` is always present: it is what a drill needs. The finer fields
    carry the extra identity a view can offer (the heatmap knows the ``method``
    column; the forest knows the ``method`` and its ``target`` vs the
    ``reference``), so a drill can be as specific as the click was. ``kind`` names
    the granularity for callers/tests.
    """

    condition: str
    method: str | None = None
    target: str | None = None
    reference: str | None = None
    pop: int | None = None            # population index, for per-population panels
    kind: str = "condition"


def _nearest_index(value, n, tol=0.5):
    """Index of the integer position nearest ``value`` within ``tol``, else None.

    The shared primitive for the integer-position layouts (imshow cells, column
    positions): a mark at integer ``k`` owns the band ``[k - tol, k + tol]``.
    """
    if value is None:
        return None
    idx = int(round(value))
    if 0 <= idx < n and abs(value - idx) <= tol:
        return idx
    return None


@dataclass
class ColumnHitTest:
    """Conditions laid at integer x (bubble / ridgeline / kinetics / fraction).

    Built from the SAME condition order the view drew, so it cannot drift from the
    plot. y is ignored: the whole column is the condition.
    """

    conditions: list

    def pick(self, x, y=None, axes=None) -> Selection | None:
        idx = _nearest_index(x, len(self.conditions))
        if idx is None:
            return None
        return Selection(condition=self.conditions[idx], kind="condition")


@dataclass
class BubbleTableHitTest:
    """ONE hit-test for the bubble plot AND its side table (Item 4).

    A click resolves to the same ``(condition, pop)`` identity whether it lands on
    a bubble or on a table row, so the row↔bubble highlight reuses a single
    mapping, never two coordinate systems that could disagree. Built from the exact
    positions the view drew:
    - ``column_pops[i]`` is column ``i``'s ``[(pop, D), …]`` (D on the log y-axis),
      so a bubble click picks the nearest population by ``log10 D``.
    - ``table_rows`` is ``[(y_center, condition, pop), …]`` in the TABLE axes' data
      coords, picked by nearest ``y``.
    The two axes are told apart by ``id(axes)`` (matplotlib's ``event.inaxes``).
    """

    conditions: list
    column_pops: list                     # per column: list[(int pop, float D)]
    bubble_axes_id: int
    table_axes_id: "int | None" = None
    table_rows: tuple = ()                # list[(float y, str condition, int pop)]
    row_tol: float = 0.5

    def pick(self, x, y, axes=None) -> "Selection | None":
        aid = id(axes) if axes is not None else None
        if self.table_axes_id is not None and aid == self.table_axes_id:
            if y is None:
                return None
            best, best_d = None, self.row_tol
            for y_c, cond, pop in self.table_rows:
                d = abs(y - y_c)
                if d <= best_d:
                    best, best_d = (cond, pop), d
            if best is None:
                return None
            return Selection(condition=best[0], pop=best[1], kind="bubble")
        # otherwise the bubble axes: condition by column, population by nearest log D
        idx = _nearest_index(x, len(self.conditions))
        if idx is None:
            return None
        cond = self.conditions[idx]
        pops = self.column_pops[idx] if idx < len(self.column_pops) else []
        pop = None
        if y is not None and y > 0 and pops:
            import math
            ly = math.log10(y)
            pop = min(pops, key=lambda pd: abs(math.log10(pd[1]) - ly))[0]
        return Selection(condition=cond, pop=pop, kind="bubble")


@dataclass
class GridHitTest:
    """An imshow grid: rows on Y, columns on X (the fraction heatmap).

    ``row_labels`` ARE the y-axis identities and ``col_labels`` the x-axis ones,
    taken from the arrays the heatmap actually plotted, so the orientation is
    read off the data, never assumed. Cell ``(row, col)`` sits at data
    ``(x=col, y=row)``.
    """

    row_labels: list          # conditions (Y)
    col_labels: list          # methods (X)

    def pick(self, x, y, axes=None) -> Selection | None:
        row = _nearest_index(y, len(self.row_labels))
        col = _nearest_index(x, len(self.col_labels))
        if row is None or col is None:
            return None
        return Selection(condition=self.row_labels[row],
                         method=self.col_labels[col], kind="cell")


@dataclass
class RowHitTest:
    """Marks that each own a y row, with x a value axis ignored for identity.

    The forest: every drawn point is recorded as ``(y_position, Selection)`` as it
    is placed, so the pick reads the exact layout, single-column (rows = methods)
    or multi-target (banded) alike, with no re-derivation of the position formula.
    """

    rows: list                # list[(float y, Selection)]
    tol: float = 0.5

    def pick(self, x, y, axes=None) -> Selection | None:
        if y is None:
            return None
        best, best_d = None, self.tol
        for y_pos, sel in self.rows:
            d = abs(y - y_pos)
            if d <= best_d:
                best, best_d = sel, d
        return best


@dataclass
class CurveHitTest:
    """Overlaid curves, one per condition (saSPT posterior spectra).

    x is D, not a condition, so identity is "which curve is the click nearest?".
    Each curve is recorded as the ``(xs, ys)`` actually plotted; the pick
    interpolates each curve's y at the click x and takes the nearest, within a
    tolerance scaled to the y-range so a click far from every curve resolves to
    nothing rather than snapping to the least-far one.
    """

    curves: list              # list[(condition, xs, ys)]
    tol_frac: float = 0.15

    def pick(self, x, y, axes=None) -> Selection | None:
        if x is None or y is None:
            return None
        import numpy as np

        # tolerance scales to the PLOT's y-extent (all curves together), so a click
        # far from every curve misses. Fall back to an absolute band when the plot
        # is flat (e.g. a single near-constant posterior): else zero span would
        # make the tolerance infinite and snap any click to the lone curve.
        stacked = [np.asarray(ys, dtype=float) for _, _, ys in self.curves
                   if np.asarray(ys).size]
        if stacked:
            allys = np.concatenate(stacked)
            y_span = float(np.nanmax(allys) - np.nanmin(allys))
        else:
            y_span = 0.0
        tol = self.tol_frac * y_span if y_span > 0 else self.tol_frac

        best, best_d = None, tol
        for condition, xs, ys in self.curves:
            xs = np.asarray(xs, dtype=float)
            ys = np.asarray(ys, dtype=float)
            if xs.size < 1 or not (xs.min() <= x <= xs.max()):
                continue
            y_hat = float(np.interp(x, xs, ys))
            d = abs(y - y_hat)
            if d <= best_d:
                best, best_d = Selection(condition=condition, kind="condition"), d
        return best


@dataclass
class AxesHitTest:
    """A grid of subplots, each subplot one identity (the spatial map panels).

    A click's identity is which subplot it landed in, not an (x, y) within one,
    so the pick keys on the axes object matplotlib reports (``event.inaxes``),
    recorded per panel as it was drawn. ``panels`` maps ``id(axes) → Selection``.
    Uses ``id()`` (not the axes as a dict key) so it never depends on an axes
    being hashable/comparable in some future matplotlib.
    """

    panels: dict              # id(axes) -> Selection

    def pick(self, x=None, y=None, axes=None) -> Selection | None:
        if axes is None:
            return None
        return self.panels.get(id(axes))


def attach(figure, hit_test) -> None:
    """Record a view's hit-test on the figure it drew, for the Qt layer to read.

    Kept as a named function (not a bare attribute set) so the coupling between a
    ``draw_*`` and the canvas click handler is greppable and intentional.
    """
    figure._smtracker_hit_test = hit_test          # type: ignore[attr-defined]


def hit_test_of(figure):
    """The hit-test a ``draw_*`` attached, or None if the view provides none."""
    return getattr(figure, "_smtracker_hit_test", None)
