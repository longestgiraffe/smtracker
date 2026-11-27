"""shadcn's **Table** (`ui/table.tsx`, not the Data Table example), ported
from `prototype/shadcn/toolbar.py`'s `_TableBody`/`_TableRow` almost verbatim.

Read off the registry rather than remembered::

    <div>              relative w-full overflow-x-auto
      <Table>          w-full caption-bottom text-sm
        <TableHeader>  [&_tr]:border-b
          <TableHead>  h-10 px-2 text-left align-middle
                       font-medium whitespace-nowrap text-foreground
        <TableBody>    [&_tr:last-child]:border-0
          <TableRow>   border-b transition-colors hover:bg-muted/50
            <TableCell>  p-2 align-middle whitespace-nowrap
        <TableCaption> mt-4 text-sm text-muted-foreground

**No bordered container** -- `rounded-md border` comes from the Data Table
EXAMPLE in the docs, not from the Table component itself; the component's own
wrapper is `relative w-full overflow-x-auto` and draws nothing. A Data Table is
Table + TanStack + sorting + pagination, none of which is here: these are
static rows. Borrowing the example's frame would dress a Table up as something
it is not. What closes the table now is the header's grey band and its own
bottom rule -- enough structure at this size, and the registry's own answer.

Built from labels rather than a `QTableWidget`, for the reason the rest of
this UI stack gives: a Qt item view brings its own palette, its own selection
colours and its own header style, and every one would have to be overridden
back to the registry's. Labels start where the registry starts.
"""
from __future__ import annotations

from qtpy.QtCore import QRect, QSize, Qt, Signal
from qtpy.QtGui import QFontMetricsF
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLayout,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..theme import typography as ty
from . import smoothscroll
from .controls import Button
from .menus import Panel, retriggered
from .primitives import SCROLLBAR_W, Badge, Checkbox, HoverFill, Surface, label
from .primitives import discard

__all__ = ["FacetedFilter", "TableBody"]

#: MUST equal `workspace.toolbar.ROOT_PX` -- restated rather than imported,
#: the same call `toolbar.py`'s own copy of this constant makes: `widgets` sits
#: BELOW `workspace` in this UI stack's dependency direction, and importing a
#: pixel helper back up from it would be the cycle that comment warns about.
ROOT_PX = 16


def rem(x: float) -> int:
    return round(x * ROOT_PX)


def text_px(rem_: float) -> int:
    return int(rem_ * ROOT_PX)


def _spread_widths(want: list, room: float) -> list:
    """`w-full`: every column at its natural width plus an EQUAL share of the slack, and
    exactly the natural widths when there is none. The one formula for it."""
    spare = room - sum(want)
    if spare > 0 and want:
        want = [w + spare / len(want) for w in want]
    # Round the column EDGES, not each width: rounding widths one by one lets the total
    # drift a pixel or two from the room, so the last column's right edge twitched by
    # that much from one frame of the sidebar's slide to the next.
    edges, total = [0], 0.0
    for w in want:
        total += w
        edges.append(round(total))
    return [b - a for a, b in zip(edges, edges[1:])]


class _ColumnsLayout(QLayout):
    """A table row's columns, placed in the SAME layout pass that sizes the row.

    **WHY NOT A `QHBoxLayout` WITH CAPPED CELLS.** That was the previous shape: each cell
    bounded (min natural, max spread) and the caps reset in `TableBody.resizeEvent`. A
    `resizeEvent` runs AFTER the row has been laid out, so the row was always laid out
    with the PREVIOUS width's caps and the new ones only took effect on a second, queued
    pass. At rest that is invisible; while the sidebar slides the width changes on
    every frame, so every frame showed stale columns: MEASURED 18-28 px of empty row at
    the right of the QC overview table, a different amount each frame, the right-aligned
    numbers jumping with it. Here `setGeometry` computes the widths from the width it is
    given, so a row is never laid out with any width but its own.

    `room_of` is where the width comes from: the TABLE's width less the cell inset and
    the scroll gutter (`TableBody._room`), so the header (outside the scroll viewport)
    and the rows (inside it) compute identical columns. Without one, the row's own
    contents width is used.

    The minimum is the natural widths, a constant: a minimum derived from a current
    width ratchets every parent layout up (the window-pinning defect `CLAUDE.md` records).
    """

    def __init__(self, parent: QWidget, natural: list, room_of=None):
        super().__init__(parent)
        self._items: list = []
        self._natural = [float(w) for w in natural]
        self._room_of = room_of

    # -- QLayout's container protocol --
    def addItem(self, item) -> None:
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int):
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    # -- sizes --
    def _extent(self) -> tuple:
        m = self.contentsMargins()
        height = max((it.sizeHint().height() for it in self._items), default=0)
        return round(sum(self._natural)) + m.left() + m.right(), height + m.top() + m.bottom()

    def sizeHint(self) -> QSize:
        return QSize(*self._extent())

    def minimumSize(self) -> QSize:
        return QSize(*self._extent())

    def expandingDirections(self):
        return Qt.Horizontal

    def setGeometry(self, rect: QRect) -> None:
        super().setGeometry(rect)
        m = self.contentsMargins()
        inner = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        room = self._room_of() if self._room_of is not None else inner.width()
        x = inner.left()
        for item, w in zip(self._items, _spread_widths(self._natural, room)):
            # The row's height, or the cell's own maximum centred in it (a badge),
            # which is what a `QHBoxLayout` did with the same cells.
            h = min(inner.height(), item.maximumSize().height())
            item.setGeometry(QRect(x, inner.top() + (inner.height() - h) // 2, w, h))
            x += w


class _TableRow(Surface):
    """`TableRow`: `border-b transition-colors hover:bg-muted/50`.

    A row rather than a bare strip of labels, because the registry gives it a
    HOVER and a bottom rule, and both are per-row -- the hover is the whole
    reason a data table reads as something you can scan: it tells you which
    line your eye is on across several columns.
    """

    #: `TableHead` is `h-10`; `TableCell` is `p-2` around a `text-sm` line.
    HEAD_H = rem(2.5)
    CELL_H = rem(2.25)

    #: A row was clicked -- only ever emitted when `clickable=True` (see
    #: `TableBody`'s own docstring on why this is opt-in rather than a
    #: standing behaviour every existing caller would inherit unasked).
    clicked = Signal()

    def __init__(self, t, cells, natural, *, head: bool = False,
                 last: bool = False, clickable: bool = False,
                 bottom_radius: float = 0, room_of=None, head_fill: str | None = None,
                 parent=None):
        super().__init__(t, radius=0, shadowed=False, parent=parent)
        self.t = t
        self.setObjectName(f"trow{id(self)}")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setFixedHeight(self.HEAD_H if head else self.CELL_H)
        # `[&_tr:last-child]:border-0` on the body: the container's own rule
        # (a separate widget in `TableBody`'s layout, see there) closes the
        # table, so a rule under the last row would double it.
        self._last = last
        #: Set externally by `set_highlighted` -- the chart-hover sync, NOT
        #: the real pointer.
        self._highlighted = False
        #: A head row is never clickable, whatever the caller asked for --
        #: it is not a row you can act on.
        self._clickable = clickable and not head
        if self._clickable:
            self.setCursor(Qt.PointingHandCursor)
        # `bottom_radius` is ONLY ever non-zero on the LAST row of a table
        # that a caller has said sits flush against its own card's bottom
        # edge -- see `TableBody.__init__`'s `bottom_radius` docstring. A
        # `Surface` never clips its children to the rounded rect it paints
        # (see that class), so a square-cornered row touching the card's
        # rounded corner squares IT off; the margin that used to sit below
        # the table as a buffer was removed once it became dead space with
        # no caption to justify it (reported directly, twice: once for the
        # dead space, once for the jagged corners the margin's removal then
        # exposed). Rounding the row's OWN bottom corners is what a browser
        # gets for free from `overflow-hidden` -- no buffer, no square edge.
        # ONLY the bottom corners: the row's top edge butts against the
        # rule above it, which shares none of this curvature.
        radius_css = (f"border-bottom-left-radius:{bottom_radius:g}px;"
                      f"border-bottom-right-radius:{bottom_radius:g}px;"
                      if bottom_radius else "")
        css = (f"QWidget#{self.objectName()} {{ background:%s; border:none;"
              f"{radius_css} }}")
        if head:
            # `TableHeader` has no hover: it is not a row you can act on. The
            # grey is `--sidebar`, the value asked for, so the band reads the
            # same grey the app already uses for "this is a header strip".
            self.setStyleSheet(css % (head_fill or t.sidebar))
            self._fill = None
        else:
            from .primitives import mix
            self._fill = HoverFill(self, lambda c: css % c, t.background,
                                   mix(t.muted, t.background, 0.5))
        lay = _ColumnsLayout(self, natural, room_of)
        lay.setContentsMargins(rem(0.75), 0, rem(0.75), 0)
        self.cells = []
        for i, cell in enumerate(cells):
            # A cell is EITHER a plain value (rendered as the registry's own
            # text cell, as always) OR an already-built QWidget, used as-is
            # -- a swatch, a sparkline, a `Badge`. Added this session so a
            # richer row (QC's own Overview table) can still be a real
            # `TableBody` instead of a hand-rolled parallel of one; every
            # EXISTING caller only ever passed plain values, so this is
            # purely additive.
            if isinstance(cell, QWidget):
                w = cell
            else:
                w = label(str(cell),
                         ty.ui("sm", "medium" if head else "normal",
                               px=text_px(0.875), tabular=not head),
                         t.foreground)
                # `text-left` for the KEY column (index 0); every other
                # column here is a measurement, and a column of numbers is
                # only readable right-aligned -- also the only way
                # `tabular-nums` pays for itself. A deliberate deviation
                # from the registry's own default, kept from the
                # prototype's own reasoning.
                w.setAlignment(Qt.AlignVCenter
                               | (Qt.AlignLeft if i == 0 else Qt.AlignRight))
            # Its natural width as a floor, set once: the columns themselves are placed by
            # `_ColumnsLayout`, never by per-resize setters (see there).
            w.setMinimumWidth(round(natural[i]))
            lay.addWidget(w)
            self.cells.append(w)

    def mousePressEvent(self, event) -> None:
        if self._clickable:
            self.clicked.emit()
        super().mousePressEvent(event)

    def set_highlighted(self, on: bool) -> None:
        """Externally driven -- `BubbleCard`'s chart-hover sync -- and
        INDEPENDENT of the real pointer, which is over the CHART while this
        fires.

        Routed through `HoverFill.override` (see its own docstring), not a
        one-off `setStyleSheet`: a plain paint here left `HoverFill._now`
        stale, and a stale `_now` made every LATER attempt to clear the
        highlight a silent no-op -- `resync()` and `ColorFade.to()` both
        compare against it to decide whether a repaint is even needed, and
        both agreed nothing had, however many times switching to a DIFFERENT
        hovered bubble called this again. Reported directly: hovering a new
        population left the previous row still lit. `override` keeps
        `_now` truthful through the whole on/off cycle, which is what lets
        `override(None)` actually reach the paint call.
        """
        if on == self._highlighted:
            return
        self._highlighted = on
        if self._fill is not None:          # the header row is never a match
            self._fill.override(self.t.accent if on else None)


def _rule(t) -> QWidget:
    """`border-b` between rows, as a 1 px band."""
    w = QWidget()
    w.setFixedHeight(1)
    w.setAttribute(Qt.WA_StyledBackground, True)
    w.setStyleSheet(f"background:{t.border};")
    return w


def _section_row(t, text: str) -> QWidget:
    """A section's band inside a table: its name in the muted label face on `muted/50`."""
    from .primitives import mix

    w = QWidget()
    w.setFixedHeight(rem(1.75))
    w.setAttribute(Qt.WA_StyledBackground, True)
    w.setStyleSheet(f"background:{mix(t.muted, t.background, 0.5)};")
    lay = QHBoxLayout(w)
    lay.setContentsMargins(rem(0.75), 0, rem(0.75), 0)
    lay.addWidget(label(text, ty.ui("xs", "semibold"), t.muted_foreground))
    lay.addStretch(1)
    return w


def cell_text(cell) -> str:
    """A table cell as plain text: a string as it is, a widget by its label(s)."""
    if isinstance(cell, str):
        return cell
    if cell is None:
        return ""
    if hasattr(cell, "text") and callable(cell.text):
        try:
            return str(cell.text())
        except TypeError:
            pass
    from qtpy.QtWidgets import QLabel

    return " ".join(lab.text() for lab in cell.findChildren(QLabel) if lab.text())


class TableBody(QWidget):
    """One static table: a header, its rows (optionally in a scroll
    viewport), an optional filter toolbar, and a caption. See the module
    docstring for the registry this reproduces.

    COPY (asked 15 Oct 2026: "a copy table button where u copy the values to directly paste in excel"):
    `copy_to_clipboard` puts the table on the clipboard Excel-ready (`table_copy.excel_text`, the system's decimal
    mark), and a right-click anywhere on the table offers it.
    """

    #: `p-2` inside a cell, and the container's inset from the slab.
    CELL_PAD = rem(0.5)
    EDGE_PAD = rem(0.75)

    #: A row was clicked -- carries its index into `rows`. Only ever
    #: emitted when `clickable=True`; every existing caller leaves this at
    #: its default and sees no behaviour change at all.
    row_clicked = Signal(int)

    def __init__(self, t, columns, rows, caption: str, parent=None,
                 *, scrollable: bool = False, toolbar: QWidget | None = None,
                 row_keys: list | None = None, clickable: bool = False,
                 bottom_radius: float = 0, head_fill: str | None = None,
                 sections: list | None = None):
        super().__init__(parent)
        self.t = t
        self.setStyleSheet("background:transparent;")
        self.columns = tuple(columns)
        self.scrollable = scrollable
        self.clickable = clickable
        #: One entry per row in `rows`, or `None` where a row has no
        #: identity to be highlighted by (an error row -- see
        #: `bubble_chart._table_rows`). Read by `highlight_key`, which is
        #: what `BubbleCard`'s chart-hover sync calls; nothing here knows
        #: what a key MEANS, only how to find the row carrying one.
        self.row_keys = list(row_keys) if row_keys is not None else [None] * len(rows)
        #: The one row currently lit by `highlight_key`, or `None`.
        self._highlighted_row: "_TableRow | None" = None
        #: `bubble_chart._TableToolbar` -- OWNED by the caller, never built
        #: or destroyed here. Adding it to `lay` below reparents it into
        #: THIS table; a later table built with the same instance reparents
        #: it again and Qt detaches it from this one first, which is what
        #: lets one toolbar (and its live selection) survive every rebuild.
        self.toolbar = toolbar
        #: What COPY puts on the clipboard: ``callable() -> (columns, rows)`` where the shown cells are rounded
        #: for reading (the SQD table: `bubble_view.copy_rows`), else None and the cells as shown.
        self.copy_source = None
        self._columns_in, self._rows_in = list(columns), [list(r) for r in rows]
        # Whether there is a caption to give room to -- read ONCE, up front,
        # because both the outer layout's bottom margin below AND the
        # caption holder's own visibility further down have to agree on the
        # same answer. `_caption_visible` is the ancestor-independent record
        # `natural_height` reads back (see there for why NOT `isVisible()`).
        self._caption_visible = bool(caption.strip())
        # `bottom_radius` rounds the LAST row's own bottom corners instead
        # of reserving a margin below it -- see `_TableRow`'s own docstring
        # for why a margin was the wrong tool (it is either a visible gap or
        # not enough to clear the corner; the row can only be one of those).
        # It ONLY applies when this table truly sits flush against a card's
        # bottom edge with nothing after it: a caption, or a toolbar that
        # turned out to have something to show, both put something ELSE
        # last, so both are checked here rather than trusted to the caller.
        _flush = (not scrollable and not self._caption_visible
                  and (toolbar is None or toolbar.height_used() == 0))
        self._bottom_radius = bottom_radius if _flush else 0
        lay = QVBoxLayout(self)
        # NO horizontal inset -- asked for directly: the table's own left and
        # right edges sit ON the card's, the same way the divider above it
        # already does, rather than carrying its own margin the way a table
        # embedded in a plain panel (no card edge to align to) would want.
        # `bubble_chart.BubbleCard`, the only caller of `scrollable=True`
        # here, is that card.
        #
        # THE BOTTOM MARGIN IS THE CAPTION'S, NOT THE TABLE'S -- it exists to
        # keep caption TEXT off a card's rounded bottom edge and is 0 right
        # alongside `_caption_visible` when there is none (untouched by
        # `bottom_radius`, which is a SEPARATE answer to a separate case --
        # a caller with no caption AND no margin still has a square-cornered
        # ROW flush against the edge; a margin can only be a visible gap or
        # too thin to matter, so the row's own corner is rounded instead.
        # See `_TableRow`'s docstring).
        lay.setContentsMargins(0, 0, 0, self.EDGE_PAD if self._caption_visible else 0)
        lay.setSpacing(0)

        self._want = self._column_widths(columns, rows)
        # ``head_fill``: the header's band, `--sidebar` grey unless a caller says otherwise -- the
        # Data Table example (a table in its own bordered frame) has none: the page colour
        self.head = _TableRow(t, columns, self._want, head=True, room_of=self._room,
                              head_fill=head_fill)
        lay.addWidget(self.head)
        self.rules = [_rule(t)]
        lay.addWidget(self.rules[0])

        # THE ROWS SCROLL; THEY ARE NEVER SQUEEZED. Where the table's height
        # is decided by something else (a figure card, fixed so its grid
        # stays aligned), a plain layout answers "not enough room" by taking
        # it back from the rows, and they collapse into each other until the
        # text overlaps. A row's height is a property of the row -- so the
        # ROWS get a viewport, while the header and caption stay outside it:
        # a header that scrolls away leaves numbers with no names.
        body = QWidget()
        body.setStyleSheet("background:transparent;")
        bl = QVBoxLayout(body)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(0)
        self.rows = []
        # SECTION ROWS (the HMM Test page's "Pinned · 3" / "Others · 4", 22 Oct 2026): a quiet
        # labelled band before the row it names, ``[(row index, text)]``; no hover, no click
        self.sections = []
        heads = dict(sections or ())
        for i, cells in enumerate(rows):
            if i in heads:
                band = _section_row(t, heads[i])
                self.sections.append(band)
                bl.addWidget(band)
                rule = _rule(t)
                self.rules.append(rule)
                bl.addWidget(rule)
            last = i == len(rows) - 1
            r = _TableRow(t, cells, self._want, last=last, clickable=clickable,
                         room_of=self._room,
                         bottom_radius=self._bottom_radius if last else 0)
            r.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            if clickable:
                r.clicked.connect(lambda idx=i: self.row_clicked.emit(idx))
            self.rows.append(r)
            bl.addWidget(r)
            if not last:
                rule = _rule(t)
                self.rules.append(rule)
                bl.addWidget(rule)
        bl.addStretch(1)
        self.body = body

        if scrollable:
            scroll = QScrollArea()
            scroll.setWidget(body)
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QScrollArea.NoFrame)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            scroll.setStyleSheet("QScrollArea { background:transparent; border:none; }")
            self.scroll = scroll
            self.scroller = smoothscroll.attach(scroll, background=t.background)
            lay.addWidget(scroll, 1)
        else:
            self.scroll = None
            lay.addWidget(body)

        # The filter toolbar, BELOW the rows and ABOVE the caption -- asked
        # for directly, moved from sitting above the table entirely (see
        # `bubble_chart._Drawer`'s own note). A plain `DataTableToolbar`
        # sits above its table because it usually stands alone; this one
        # narrows a table that is already in view, so it reads better as
        # part of what you have just finished reading than as a preamble.
        # Full width, no extra inset here -- the toolbar carries its own
        # (`_TableToolbar`'s `CARD_PAD`), matching where it sat before.
        if toolbar is not None:
            toolbar.setVisible(toolbar.height_used() > 0)
            lay.addWidget(toolbar)

        # `TableCaption`: `mt-4 text-sm text-muted-foreground`, OUTSIDE the
        # container -- it says what a row MEANS, which is about the table
        # rather than part of it.
        #
        # THE HOLDER IS HIDDEN, NOT JUST EMPTY, when `caption` is blank.
        # `label("", ...)` still has a real line-height and `holder` still
        # carries its own top margin, so a caller with nothing to say (the
        # common case once a caller only wants row content) got a strip of
        # unexplained dead air at the bottom of the table instead of a
        # clean edge -- reported directly, against a real project, after
        # `""` had already been the deliberate choice once. `natural_height`
        # below has to know this too, for the exact reason `self.toolbar
        # .height_used()` already guards against a few lines up: a HIDDEN
        # widget keeps reporting its last real `height()`, not 0.
        self.caption = label(caption, ty.ui("sm", px=text_px(0.875)),
                             t.muted_foreground)
        self.caption.setWordWrap(True)
        self._caption_holder = QWidget()
        self._caption_holder.setStyleSheet("background:transparent;")
        hl = QHBoxLayout(self._caption_holder)
        hl.setContentsMargins(self.EDGE_PAD, rem(0.5), self.EDGE_PAD, 0)
        hl.setSpacing(0)
        hl.addWidget(self.caption)
        # `_caption_visible` was set at the top of `__init__`, before `lay`
        # even existed -- reused here rather than recomputed so the margin
        # and the holder's visibility can never disagree about the same
        # blank-vs-real caption. NOT `isVisible()` read back later: that is
        # gated by every ancestor's visibility too (the same trap
        # `_FilterButton.has_filter` documents for the toolbar), so a
        # `TableBody` not yet shown would read False for a caption that IS
        # meant to show.
        # ADDED FIRST, then made visible: `setVisible(True)` on a widget with no
        # parent yet opens it as its own WINDOW, one per table -- seven of them
        # flashing open as the workspace was built (`discard`'s note).
        lay.addWidget(self._caption_holder)
        self._caption_holder.setVisible(self._caption_visible)

    def natural_height(self) -> int:
        """What this table wants: header, rule, every row, the filter
        toolbar (0 when there is none, or it has nothing worth filtering --
        see `_TableToolbar.height_used`), and the caption.

        Measured from the PARTS rather than taken from `sizeHint`, which
        stops meaning anything once the rows are inside a viewport -- and,
        for the toolbar specifically, because a HIDDEN widget keeps
        reporting its last real `height()` rather than 0, which is exactly
        why `height_used()` exists instead of trusting it directly.
        """
        rows = (sum(r.height() for r in self.rows) + sum(b.height() for b in self.sections)
                + max(0, len(self.rules) - 1))
        rows = max(rows, self.body.sizeHint().height())
        extra = self.toolbar.height_used() if self.toolbar is not None else 0
        # The caption's own height AND the outer layout's bottom margin --
        # the margin exists FOR the caption (see `__init__`) and is 0 right
        # alongside it when there is none, so both terms are gated together.
        caption = (rem(0.5) + self.caption.height() + self.EDGE_PAD
                   if self._caption_visible else 0)
        return (self.head.height() + 1
                + rows
                + extra
                + caption)

    def highlight_key(self, key) -> bool:
        """Scroll the row carrying `key` (see `row_keys`) into view and
        highlight it, replacing whatever this table last highlighted.
        Returns whether a match was found in THIS table -- `False` means
        the row is not here (filtered out, most likely), and the caller
        (`BubbleCard._on_chart_hover`) is the one that decides what that
        means; this method never guesses on its behalf.
        """
        self.clear_highlight()
        for row, row_key in zip(self.rows, self.row_keys):
            if row_key == key:
                row.set_highlighted(True)
                self._highlighted_row = row
                if self.scroll is not None:
                    self.scroll.ensureWidgetVisible(row, 0, rem(0.5))
                return True
        return False

    def clear_highlight(self) -> None:
        if self._highlighted_row is not None:
            self._highlighted_row.set_highlighted(False)
            self._highlighted_row = None

    def _column_widths(self, columns, rows) -> list:
        """Column widths from the CONTENT, so nothing is elided or ragged --
        `whitespace-nowrap` on both `TableHead` and `TableCell` means the
        registry never wraps a cell.

        A cell that is already a `QWidget` (see `_TableRow`) is measured by
        its own `sizeHint()` rather than by text metrics — asking
        `QFontMetricsF` to measure a `Badge` or a sparkline's `str()` would
        return the width of its Python repr, not of anything visible.
        """
        metrics = QFontMetricsF(ty.ui("sm", "medium", px=text_px(0.875)))
        pad = rem(0.75)

        def w_of(value) -> float:
            if isinstance(value, QWidget):
                return float(value.sizeHint().width())
            return metrics.horizontalAdvance(str(value))

        want = [max(w_of(c), *(w_of(r[i]) for r in rows)) + pad
                if rows else w_of(c) + pad
                for i, c in enumerate(columns)]
        return want

    def _room(self) -> float:
        """The width the columns share: this table's width less `_TableRow`'s own `px-3`
        cell inset on both sides, and less `SCROLLBAR_W` when `scrollable` -- ALWAYS, not
        only once a bar happens to be showing. The header sits OUTSIDE the scroll
        viewport, at the table's full width, so without a constant reservation its
        columns line up with the body's only while a bar is visible and drift the moment
        the row count crosses the viewport's height. A fixed gutter -- present or not --
        keeps the header and the rows agreeing always.

        Read by every row's `_ColumnsLayout` while it is laid out, so the columns always
        follow the width being laid out (`w-full`, spread by `_spread_widths`)."""
        reserve = SCROLLBAR_W if self.scrollable else 0
        return self.width() - 2 * rem(0.75) - reserve

    # -- COPY: the table for Excel ----------------------------------------------------- #
    def copy_rows(self) -> tuple[list, list]:
        """``(columns, rows)`` that COPY writes: the copy source's, else the cells as shown."""
        if self.copy_source is not None:
            return self.copy_source()
        return ([cell_text(c) for c in self._columns_in],
                [[cell_text(c) for c in row] for row in self._rows_in])

    def copy_to_clipboard(self) -> int:
        """The table on the clipboard, Excel-ready; returns how many rows. Said in the window's toasts."""
        from qtpy.QtCore import QLocale
        from qtpy.QtWidgets import QApplication

        from ...table_copy import excel_text

        columns, rows = self.copy_rows()
        QApplication.clipboard().setText(excel_text(columns, rows, decimal=QLocale.system().decimalPoint()))
        toasts = getattr(self.window(), "toasts", None)
        if toasts is not None:
            toasts.post("Table copied", description=f"{len(rows)} row{'s' if len(rows) != 1 else ''}: paste "
                                                    f"into Excel")
        return len(rows)

    def contextMenuEvent(self, e) -> None:  # noqa: N802 - Qt's name
        """Right-click anywhere on the table: Copy table."""
        from .menus import Panel, menu_row

        menu = Panel(self.t, self.window())
        menu_row(self.t, menu, "Copy table", glyph="copy",
                 on_click=lambda: (menu.close_chain(), self.copy_to_clipboard()))
        menu.popup(e.globalPos())
        self._copy_menu = menu


# --------------------------------------------------------------------------- #
# The Data Table example's faceted filter -- see `FacetedFilter`'s own
# docstring for what this is and, more importantly, is NOT: the registry has
# no packaged filter component, only this recipe.
# --------------------------------------------------------------------------- #


class _FilterOption(Surface):
    """One checkable row inside a `FacetedFilter` popover: shadcn's
    `CommandItem` holding a `Checkbox`, plus the facet count
    (`getFacetedUniqueValues()`) right-aligned. No search box lives above
    these -- see `FacetedFilter`'s own docstring.

    The checkbox is `primitives.Checkbox` -- the same shared, pixel-verified
    widget `onboarding.groups` uses -- rather than a second, rougher one built
    just for this popover. It handles its OWN click (a release inside its
    16 px box toggles and emits `toggled(bool)`); the row forwards that as its
    own no-arg `toggled` and separately toggles the box for a release
    ANYWHERE ELSE in the row, since the whole `CommandItem` is the click
    target, not just the box.
    """

    toggled = Signal()

    ROW_H = rem(2)

    def __init__(self, t, value: str, count: int, *, checked: bool, parent=None):
        super().__init__(t, radius=t.radius_sm, shadowed=False, parent=parent)
        self.t, self.value = t, value
        self.setObjectName(f"filtopt{id(self)}")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(self.ROW_H)
        self._fill = HoverFill(
            self, lambda c: (f"QWidget#{self.objectName()} {{ background:{c};"
                             f" border:none; border-radius:{t.radius_sm:g}px; }}"),
            t.popover, t.accent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(rem(0.5), 0, rem(0.5), 0)
        lay.setSpacing(rem(0.5))
        self._box = Checkbox(t, checked=checked)
        self._box.toggled.connect(lambda _on: self.toggled.emit())
        lay.addWidget(self._box)
        lay.addWidget(label(value, ty.ui("sm", px=text_px(0.875)), t.foreground))
        lay.addStretch(1)
        lay.addWidget(label(str(count), ty.ui("xs", px=text_px(0.75), tabular=True),
                            t.muted_foreground))

    def is_checked(self) -> bool:
        return self._box.checked

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self.rect().contains(e.position().toPoint()):
            self._box.set_checked(not self._box.checked)
            self.toggled.emit()


class _ClearRow(Surface):
    """`"Clear filters"`: the `CommandItem` shown at the bottom of a
    `FacetedFilter` popover once anything in it is checked."""

    clicked = Signal()

    def __init__(self, t, parent=None):
        super().__init__(t, radius=t.radius_sm, shadowed=False, parent=parent)
        self.t = t
        self.setObjectName(f"clearrow{id(self)}")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(rem(2))
        self._fill = HoverFill(
            self, lambda c: (f"QWidget#{self.objectName()} {{ background:{c};"
                             f" border:none; border-radius:{t.radius_sm:g}px; }}"),
            t.popover, t.accent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(rem(0.5), 0, rem(0.5), 0)
        lay.addWidget(label("Clear filters", ty.ui("sm", px=text_px(0.875)),
                            t.foreground), alignment=Qt.AlignCenter)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self.rect().contains(e.position().toPoint()):
            self.clicked.emit()


class FacetedFilter(Button):
    """shadcn's `DataTableFacetedFilter`, ported onto this project's OWN
    trigger `Button` rather than the reference's dashed-outline one -- asked
    for directly, to match the `Filter` button already built for
    `onboarding.groups.GroupsPanel`'s header:
    `Button(t, "Filter", size="small", glyph="filter")`. Two filter triggers
    that look like a second, rougher control is worse than one that looks
    like the button already in the app; this IS that button, just opening a
    different popover and carrying its own title per column.

    A `Panel` popover lists every distinct value the column holds, each with
    a checkbox and how many rows carry it -- plus "Clear filters" once
    anything is checked. `set_expanded` (Button's own `aria-expanded` fill
    hold) runs the same way it does for every other popover trigger in this
    app -- set on open, released from the popover's `closed` signal.

    **UNCHECKED = NO FILTER**, matching the reference exactly:
    `column.getFilterValue()` starts `undefined` (every row shown); checking
    a value NARROWS to rows carrying it. This is not "all boxes start
    ticked" -- an empty `selected()` means "not filtering on this column at
    all", the same meaning `None`/`""` carries everywhere else in this app.

    **NO SEARCH BOX.** The reference builds this on top of `Command`, whose
    own `CommandInput` exists for a filter with dozens of distinct values --
    every column this project puts a `FacetedFilter` on (a condition, a
    fit-mode series) has at most a handful, where a search box is furniture
    rather than a feature.

    **Counts are STATIC**, taken once from `set_options` rather than
    recomputed against whatever the OTHER filter currently hides (TanStack's
    `getFacetedUniqueValues()` does the latter, tracking every other active
    filter). At the row counts this table ever holds that distinction is not
    worth the coupling it would add between two otherwise-independent
    filters.
    """

    changed = Signal()

    def __init__(self, t, title: str, parent=None):
        super().__init__(t, title, variant="outline", size="small",
                         glyph="filter", parent=parent)
        self._options: list = []            # [(value, count), ...]
        self._checked: set = set()
        self._badges: list = []             # the separator + Badge widgets
        self.clicked.connect(self._on_clicked)
        # Hidden until `set_options` has something worth filtering -- a
        # button open onto an empty popover, or a single-value column where
        # checking the one box can only ever reproduce the unfiltered table,
        # is a control with nothing to do.
        self.hide()

    def set_options(self, options) -> None:
        """(Re)point at a fresh Run's distinct values + counts.

        Clears whatever was checked: a filter selection made against the
        LAST Run's conditions is not assumed to still mean anything against
        a new one (the same rule `BubbleCard.set_data` already applies to
        the table's own remembered height).
        """
        self._options = list(options)
        self._checked = set()
        self._refresh_badges()
        self.setVisible(len(self._options) > 1)

    def selected(self) -> set:
        return set(self._checked)

    @property
    def has_filter(self) -> bool:
        """Whether this trigger has more than one distinct value to offer --
        i.e. whether `set_options` chose to show it.

        NOT `isVisible()`. `_TableToolbar` (the container this sits in) is
        itself hidden/shown from `height_used()`, which reads THIS property
        for exactly that reason: `isVisible()` is gated by every ANCESTOR's
        visibility too, so the moment the container is hidden this widget's
        own `isVisible()` would read False regardless of whether `set_options`
        had asked for it to be shown -- and `height_used()` would then never
        see a reason to unhide the container again. Caught by `TableBody`
        gating the toolbar at CONSTRUCTION, before any Run had ever called
        `set_options`: the container hid itself once, on the placeholder
        table, and nothing after that could tell `height_used()` otherwise.
        """
        return len(self._options) > 1

    def clear(self) -> None:
        """Uncheck everything -- the SELECTION only; `_options` (and
        therefore the button's own visibility) is untouched, unlike
        `set_options([])` which would also drop the options themselves."""
        if self._checked:
            self._checked = set()
            self._refresh_badges()

    def _on_clicked(self) -> None:
        # Same guard as `groups.GroupsPanel`'s own `Filter` button: `clicked`
        # fires on the release that DISMISSES an outside click too, so
        # without this a click meant to close the popover reopens it at once.
        if retriggered(self):
            return
        self._open()

    def _open(self) -> None:
        # `aria-expanded`: held while the popover is open, released from its
        # own `closed` signal so it clears however the popover goes away --
        # a pick, "Clear filters", or a click outside. Same idiom as
        # `groups.GroupsPanel._open_filter`.
        self.set_expanded(True)
        panel = Panel(self.t, self.window(), root=ROOT_PX)
        panel.closed.connect(lambda: self.set_expanded(False))
        holder = QWidget()
        holder.setStyleSheet("background:transparent;")
        vlay = QVBoxLayout(holder)
        vlay.setContentsMargins(rem(0.25), rem(0.25), rem(0.25), rem(0.25))
        vlay.setSpacing(rem(0.125))
        for value, count in self._options:
            row = _FilterOption(self.t, value, count, checked=value in self._checked)
            row.toggled.connect(lambda _v=value, _r=row: self._toggle(_v, _r))
            vlay.addWidget(row)
        if self._checked:
            vlay.addWidget(_rule(self.t))
            clear = _ClearRow(self.t)
            clear.clicked.connect(lambda: self._clear(panel))
            vlay.addWidget(clear)
        panel.add_widget(holder)
        panel.set_minimum_visible_width(max(self.width(), rem(10)))
        panel.popup_under(self, align="start")

    def _toggle(self, value: str, row: "_FilterOption") -> None:
        if row.is_checked():
            self._checked.add(value)
        else:
            self._checked.discard(value)
        self._refresh_badges()
        self.changed.emit()

    def _clear(self, panel: Panel) -> None:
        self._checked = set()
        self._refresh_badges()
        self.changed.emit()
        panel.close_chain()

    def _refresh_badges(self) -> None:
        """One `"k/M"` badge, ALWAYS shown once there is more than one
        option -- not the list of checked names this used to grow into as
        more were picked.

        `k` reads `len(self._checked)` when narrowed, and the OPTION COUNT
        when it is not -- "unchecked = no filter" (see the class docstring)
        means every row is showing, so `"2/2"` is the true count there, not
        a placeholder for "nothing selected". Showing it AT REST too (rather
        than only once something is checked, as the name-list version did)
        is what keeps a fresh trigger from reading as ambiguous between "no
        filter yet" and "filtered to nothing" -- asked for directly, after
        the name list itself was asked for and dropped: it grows without
        bound and answers a question ("which ones, exactly") nobody these
        columns need at their size.
        """
        lay = self.layout()
        for w in self._badges:
            lay.removeWidget(w)
            discard(w)
        self._badges = []
        total = len(self._options)
        if total > 1:
            sep = QWidget()
            sep.setFixedWidth(1)
            sep.setFixedHeight(rem(1.25))
            sep.setStyleSheet(f"background:{self.t.border};")
            lay.addWidget(sep)
            self._badges.append(sep)
            shown = len(self._checked) if self._checked else total
            badge = Badge(self.t, f"{shown}/{total}", self.t.muted, self.t.foreground)
            lay.addWidget(badge)
            self._badges.append(badge)
        # `w-fit`: `Button` is Fixed/Fixed and does not re-measure on its
        # own -- the same call its own `set_text` makes for a relabel.
        self.adjustSize()
        self.updateGeometry()
