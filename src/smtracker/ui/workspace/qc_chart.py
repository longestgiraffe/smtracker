"""QC section: the Qt shell only (Overview table, the fair-comparison and contributes-nothing cards, the
Detail cards and the outlier Inspect dialog).

It computes and decides nothing. Every number, rule and sentence comes from the headless
`smtracker.qc_view` (which in turn reads `smtracker.qc` and, for the brightness check,
`smtracker.spot_intensity`), as plain lists, dicts, strings and numbers: the Thin Qt rule
(`tests/test_ui_thin_qt.py`). The charts here map those lists to pixels and draw them; the
tooltips and captions are the view's own sentences. Nothing here edits, wraps, or depends on
the legacy browser layer (`browser.py`, `browser_controller.py`, `widgets.py`, `plots.py`).

Two pages, per the design settled after a long back-and-forth on the shape (mockups, a
hand-drawn wireframe, and a full walk through the biology so the decisions are understood
rather than just agreed to):

* a **General** page: every loaded condition, one light row each; then **Is the comparison
  fair?** (`QCFairTable`) and **Contributes nothing** (`QCNothingCard`), which replaced the
  Comparator on 13 Oct 2026;
* a **Detail** page: the original `_QCTab`'s five panels, reimplemented native (QPainter, no
  matplotlib), for exactly one condition; the outlier card keeps its Inspect dialog (the
  Localizations-per-frame card's five extra views were retired with the Comparator).
"""

from __future__ import annotations

import math
from typing import ClassVar

from qtpy.QtCore import (
    QPointF,
    QRectF,
    QSize,
    Qt,
    Signal,
)
from qtpy.QtGui import (
    QBitmap,
    QColor,
    QFontMetricsF,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
)
from qtpy.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ... import chart_text
from ...population_style import SHAPES
from ...qc_view import (
    CELL_LENGTH_RULE,
    cell_length_hover,
    sorted_axis_title,
    column_groups,
    column_hover,
    in_cell_hover,
    length_hover,
    movie_runs,
    neighbour_band_text,
    neighbour_bar_text,
    outlier_description,
    outlier_readout,
    overview_row,
    short_labels,
    unit_hover,
)
from ..onboarding.panels import section_font
from ..theme import hcl, icons
from ..theme import typography as ty
from ..widgets import tooltip
from ..widgets.controls import Button, Segmented
from ..widgets.dialogs import CloseButton, OverlayDialog
from ..widgets.primitives import (
    SCROLLBAR_W,
    Badge,
    Card,
    FitLabel,
    Icon,
    ShrinkLabel,
    band,
    col,
    icon,
    label,
    mix,
    row,
)
from ..widgets.table import TableBody
from .bubble_chart import BubbleChart
from .dynamics_chart import _KIN_ON_COLOR, _WARN_COLOR, _ChartTooltip
from .fractions_chart import _MiniChart
from ..widgets.settings import SettingNav as _SettingNav
from .toolbar import rem
from ..widgets.primitives import discard

__all__ = [
    "GroupCountsTable",
    "InCellShareChart",
    "NeighbourShareChart",
    "QCConditionDetail",
    "QCFairTable",
    "QCNothingCard",
    "QCOutlierInspectDialog",
    "QCOverviewTable",
    "TrackLengthChart",
    "UnitDotsChart",
]


# --------------------------------------------------------------------------- #
# Widgets
# --------------------------------------------------------------------------- #



def draw_marker(p, shape: str, centre: QPointF, r: float) -> None:
    """One hollow marker of a population (`population_style.SHAPES`) with the painter's pen: a circle, a square or a
    triangle of radius ``r``."""
    if shape == "square":
        p.drawRect(QRectF(centre.x() - r, centre.y() - r, 2 * r, 2 * r))
    elif shape == "triangle":
        p.drawPolygon(QPolygonF([QPointF(centre.x(), centre.y() - 1.15 * r),
                                 QPointF(centre.x() - 1.1 * r, centre.y() + 0.85 * r),
                                 QPointF(centre.x() + 1.1 * r, centre.y() + 0.85 * r)]))
    else:
        p.drawEllipse(centre, r, r)


class _Sparkline(QWidget):
    """The Overview table's Length column — the track-length histogram's
    own counts, scaled to their own max, no axis. The shape this project
    actually agreed on (`notes/qc-dashboard-overview-drilldown.html`'s
    own ``.spark``), not the plain "median 6.0" text this row shipped
    with the first time — a simplification made silently during
    implementation and not something anyone signed off on.
    """

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self._counts = None
        self._colour = t.foreground
        self.setFixedSize(64, 22)

    def set_data(self, bars, colour: str) -> None:
        """``bars`` is `qc_view.spark_bars`' own list (already bucketed), or None for no histogram."""
        self._counts = bars
        self._colour = colour
        self.update()

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        if self._counts:
            counts = self._counts
            hi = max(counts) or 1.0
            n = len(counts)
            w = self.width() / n
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(self._colour))
            for i, c in enumerate(counts):
                h = max(1.5, (c / hi) * self.height())
                p.drawRect(QRectF(i * w + 0.5, self.height() - h,
                                  max(1.0, w - 1), h))
        p.end()


def _right_aligned_cell(widget: QWidget) -> QWidget:
    """A widget cell (a `Badge`, a `_Sparkline`) that stays at its own
    compact size instead of stretching to fill the column — `_TableRow`
    calls `setFixedWidth` on the whole cell to spread `TableBody`'s
    leftover width evenly across every column, so a widget handed to it
    directly is forced wide.

    The stretch goes BEFORE the widget, not after: `_TableRow`'s own text
    cells right-align every column but the first (see that module), and a
    widget cell left-aligned inside the same column reads as sitting under
    the WRONG header — measured directly against a real project: the
    Flagged badge landed hard against the previous column's text instead
    of under "Flagged". Matching the text cells' own alignment is what
    makes a value actually sit under its header.
    """
    host = QWidget()
    lay = QHBoxLayout(host)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.addStretch(1)
    lay.addWidget(widget, 0, Qt.AlignVCenter)
    return host


class _ConditionCell(QWidget):
    """One `TableBody` cell: a condition-coloured dot beside its name —
    the one piece of content the registry's plain-text cell can't already
    express on its own. Everything else in this table is a real
    `TableBody` cell (plain text, a `_Sparkline`, or a `Badge`), because
    that component IS this project's shadcn Table port, already used
    everywhere else — building a parallel one for this table specifically
    was a mistake, corrected here rather than re-justified.
    """

    def __init__(self, t, condition: str, colour: str, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(rem(0.5))
        dot = QWidget()
        dot.setFixedSize(9, 9)
        dot.setStyleSheet(f"background:{colour}; border-radius:2px;")
        lay.addWidget(dot, 0, Qt.AlignVCenter)
        lay.addWidget(label(condition, ty.ui("sm", "medium"), t.foreground),
                     0, Qt.AlignVCenter)
        lay.addStretch(1)


class QCOverviewTable(Card):
    """The General page's per-condition summary — a real `table.TableBody`
    (this project's shadcn Table port), wrapped in a bordered `Card` the
    way `TableBody`'s own docstring says a caller should (the border is
    the Data Table EXAMPLE's, not the plain Table's — see that module),
    not a hand-rolled parallel of one.

    Columns: Condition (a coloured dot + name), Tracks, Locs/cells,
    Length (a `_Sparkline`), In-cell %, Flagged and Status (`Badge`
    widgets) — deliberately compact (the smallest set of facts that
    answers "should I look closer"), never the dense five-panel detail
    itself. A row click opens that condition's Detail page via
    `row_clicked`, using `TableBody`'s own `clickable=True` (added this
    session — its rows were display-only before). Tracks/localizations is
    shown here and NOT repeated in the Comparator below, for the same
    reason.
    """

    row_clicked = Signal(str)

    _COLUMNS = ("Condition", "Tracks", "Locs / cells", "Length",
               "In-cell", "Flagged", "Status", "")

    def __init__(self, t, parent=None):
        super().__init__(t, parent=parent)
        self.t = t
        lay = QVBoxLayout(self)

        # Header: icon + bold title + a live count badge, then a subtitle
        # underneath — `onboarding.panels.ReplicateDays`'s own header shape,
        # reused directly rather than approximated from a screenshot: same
        # `icon(...)`/`row(...)`/`col(...)` helpers, `section_font()`
        # IMPORTED rather than replicated (an earlier version of this
        # comment said three lines wasn't worth an import — it was, the
        # moment `QCComparator`'s own header needed the identical font and
        # a second copy would have been the thing to drift), same muted
        # count badge.
        self._count_badge = Badge(t, "0", t.muted, t.foreground)
        self.copy_button = _copy_button(t, lambda: getattr(self, "_table", None))
        head = col(
            row(icon("flask-conical", icons.MD, t.muted_foreground),
                label("Conditions", section_font(), t.foreground),
                self._count_badge,
                None, self.copy_button, spacing=8),
            label("Loaded conditions · click a row for its full detail",
                 ty.ui("xs"), t.muted_foreground),
            spacing=3, margins=(20, 16, 20, 12))
        lay.addWidget(head)

        # NO margin on either edge. `head` is TRANSPARENT (`col()`'s own
        # background), so it has none of the square-corner-on-a-round-corner
        # collision a solid fill would; its own 16px top padding
        # (`margins=(20, 16, 20, 12)` above) is already the inset. The
        # bottom needs no margin either, now that `TableBody` itself rounds
        # its own last row to match this card (`bottom_radius=self.radius`
        # in `set_payload`) instead of a margin trying to buffer a square
        # corner off a round one -- a margin can only be a visible gap or
        # too thin to matter, never neither, which is what made the bottom
        # edge bounce between "there's a gap" and "the corners are jagged"
        # twice before landing here. Found by running this against a real
        # project each time, not by reading the layout code.
        lay.setContentsMargins(0, 0, 0, 0)
        self._host = QVBoxLayout()
        lay.addLayout(self._host)
        self._table: TableBody | None = None
        self._condition_order: list[str] = []

    def set_payload(self, per_condition: dict, colours: dict | None = None) -> None:
        """``per_condition`` is `compute_qc_overview`'s own dict, keyed by
        condition name. ``colours`` is condition -> hex string, the SAME
        per-condition palette every other chart in this workspace already
        draws with (`window._condition_colours`) — not a new one invented
        for this table."""
        colours = colours or {}
        self._condition_order = list(per_condition.keys())
        rows = []
        for condition, entry in per_condition.items():
            colour = colours.get(condition, self.t.muted_foreground)
            cond_cell = _ConditionCell(self.t, condition, colour)
            words = overview_row(entry)
            if words["failed"]:
                rows.append([cond_cell, words["tracks"], words["locs_cells"], words["tracks"],
                            words["in_cell"], words["n_flagged"], words["status"], "›"])
                continue
            spark = _Sparkline(self.t)
            spark.set_data(words["bars"], colour)
            fill = _WARN_COLOR if words["flagged"] else self.t.muted
            ink = "#FFFFFF" if words["flagged"] else self.t.muted_foreground
            status = Badge(self.t, words["status"], fill, ink)
            if words.get("yield_text"):
                # which replicate days are low in track yield (`qc_view.yield_text`), on
                # demand: the badge has room for the word, not the days
                tooltip.attach(self.t, status, words["yield_text"])
            rows.append([
                cond_cell,
                words["tracks"],
                words["locs_cells"],
                _right_aligned_cell(spark),
                words["in_cell"],
                _right_aligned_cell(Badge(self.t, words["n_flagged"], fill, ink)),
                _right_aligned_cell(status),
                "›",
            ])
        if self._table is not None:
            discard(self._table)
        self._count_badge.set_text(str(len(self._condition_order)))
        self._table = TableBody(self.t, list(self._COLUMNS), rows, "",
                                clickable=True, bottom_radius=self.radius)
        self._table.row_clicked.connect(self._on_row_clicked)
        self._host.addWidget(self._table)

    def _on_row_clicked(self, index: int) -> None:
        self.row_clicked.emit(self._condition_order[index])


def _card_head(t, glyph: str, title: str, subtitle: str, badge: Badge | None = None, action=None):
    """The General page's card header: icon, bold title (and a count badge), a muted subtitle -- the
    Overview table's own shape; ``action`` (a button) at the right."""
    parts = [icon(glyph, icons.MD, t.muted_foreground), label(title, section_font(), t.foreground)]
    if badge is not None:
        parts.append(badge)
    parts.append(None)
    if action is not None:
        parts.append(action)
    caption = _Caption(t)
    caption.setText(subtitle)
    return col(row(*parts, spacing=8), caption, spacing=3, margins=(20, 16, 20, 12)), caption


def _copy_button(t, table_of):
    """The Copy-table button a General-page card carries (asked 15 Oct 2026): ``table_of()`` is the card's
    current `TableBody`, or None before a run."""
    from .toolbar import ToolButton, rem

    button = ToolButton(t, "copy-table", "copy", size=rem(1.75), rest=t.card)
    button.clicked.connect(lambda _k: table_of() and table_of().copy_to_clipboard())
    button._tip = tooltip.attach(t, button, "Copy the table (paste into Excel; a spread gets its own column)", side="bottom", top_level=True)
    return button


class QCFairTable(Card):
    """IS THE COMPARISON FAIR? (13 Oct 2026, asked for with "some info is not needed, while i feel we have
    urgent info we can provide"). The conditions side by side on what biases a fraction -- tracker, days,
    frame interval, track length, yield, in-cell share, tracks just outside an outline, crowding -- each
    heading's tooltip stating its rule, and a value that breaks it drawn in the warning colour with the
    reason on hover. Every number and rule is `qc_view.fair_comparison`'s; this only draws.

    The same table draws "Do the replicates differ?" (`qc_view.replicates_table`, 24 Oct 2026): a payload of the
    same shape under its own ``glyph``, ``title`` and ``subtitle``."""

    def __init__(self, t, parent=None, *, glyph: str = "arrow-left-right", title: str = "Is the comparison fair?",
                 subtitle: str = "What biases a fraction, condition by condition"):
        super().__init__(t, parent=parent)
        self.t = t
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._badge = Badge(t, "0", t.muted, t.foreground)
        self.copy_button = _copy_button(t, lambda: self._table)
        head, self._caption = _card_head(t, glyph, title, subtitle, self._badge, action=self.copy_button)
        lay.addWidget(head)
        self._host = QVBoxLayout()
        lay.addLayout(self._host)
        self._table: TableBody | None = None

    def set_payload(self, fair: dict | None, colours: dict | None = None) -> None:
        """``fair`` None (a result that has no such part) hides the card."""
        self.setVisible(fair is not None)
        if fair is None:
            return
        colours = colours or {}
        rows = []
        for entry in fair["rows"]:
            cells = [_ConditionCell(self.t, entry["condition"],
                                    colours.get(entry["condition"], self.t.muted_foreground))]
            for (heading, rule), text, marked in zip(fair["columns"], entry["cells"], entry["marked"]):
                if marked:
                    cell = label(text, ty.ui("sm", "semibold"), _WARN_COLOR)
                    cell.setAttribute(Qt.WA_TransparentForMouseEvents, False)
                    tooltip.attach(self.t, cell, f"{heading}: {rule}")
                    # a widget cell sits where the column's text would: right-aligned
                    cells.append(_right_aligned_cell(cell))
                else:
                    cells.append(text)
            rows.append(cells)
        if self._table is not None:
            discard(self._table)
        headings = ["Condition"] + [h for h, _r in fair["columns"]]
        self._table = TableBody(self.t, headings, rows, "")
        # EACH HEADING CARRIES ITS RULE, on hover: the caption says so, and a separate
        # note would be the same words twice
        for cell, (_heading, rule) in zip(self._table.head.cells[1:], fair["columns"]):
            cell.setAttribute(Qt.WA_TransparentForMouseEvents, False)
            tooltip.attach(self.t, cell, rule)
        self._host.addWidget(self._table)
        self._caption.setText(fair["caption"])
        n = int(fair.get("n_marked", 0))
        self._badge.set_text(str(n))
        self._badge.set_colours(_WARN_COLOR if n else self.t.muted,
                                "#FFFFFF" if n else self.t.foreground)


class QCNothingCard(Card):
    """WHAT CONTRIBUTES NOTHING: everything in the folders that is not in the numbers, and why, per
    condition (`qc_view.contributes_nothing`) -- movies with no tracks, no outlines or every track outside,
    low-yield days, the track cap and exclusions -- gathered from the hovers and badges they were spread
    over. One line when there is nothing to say."""

    #: The header's Exclusions button (16 Oct 2026): the window owns the project, so it opens the dialog.
    exclusions_requested = Signal()

    def __init__(self, t, parent=None):
        super().__init__(t, parent=parent)
        self.t = t
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 12)
        self._badge = Badge(t, "0", t.muted, t.foreground)
        self.exclusions_button = Button(t, "Exclusions…", variant="outline", size="small", glyph="eye-off")
        self.exclusions_button.clicked.connect(self.exclusions_requested.emit)
        head, self._caption = _card_head(t, "eye-off", "Contributes nothing",
                                         "In the folders, not in the numbers", self._badge,
                                         action=self.exclusions_button)
        lay.addWidget(head)
        self._body = QVBoxLayout()
        self._body.setContentsMargins(20, 0, 20, 0)
        self._body.setSpacing(6)
        lay.addLayout(self._body)

    def set_payload(self, nothing: dict, colours: dict | None = None) -> None:
        colours = colours or {}
        while self._body.count():
            item = self._body.takeAt(0)
            if item.widget() is not None:
                discard(item.widget())
        for condition, lines in nothing["rows"]:
            block = QWidget()
            block_lay = QVBoxLayout(block)
            block_lay.setContentsMargins(0, 0, 0, 0)
            block_lay.setSpacing(2)
            block_lay.addWidget(_ConditionCell(self.t, condition,
                                               colours.get(condition, self.t.muted_foreground)))
            for line in lines:
                text = FitLabel()
                text.setFont(ty.ui("sm"))
                text.setStyleSheet(f"color:{self.t.foreground}; background:transparent;")
                text.setContentsMargins(18, 0, 0, 0)
                text.setText(f"\u2022 {line}")
                block_lay.addWidget(text)
            self._body.addWidget(block)
        self._caption.setText(nothing["caption"])
        n = int(nothing.get("n", 0))
        self._badge.set_text(str(n))
        self._badge.set_colours(_WARN_COLOR if n else self.t.muted,
                                "#FFFFFF" if n else self.t.foreground)


# --------------------------------------------------------------------------- #
# The five Detail-page panels
# --------------------------------------------------------------------------- #

#: Data ink and the "look at this" ink, shared with the dynamics charts so
#: the workspace reads as one chart language (blue = data, orange = flagged).
_DATA = _KIN_ON_COLOR
_FLAG = _WARN_COLOR

#: Small type for rotated category labels, where a full tick font would not fit.
_CATEGORY_PX = 9


def _dashed(t) -> QPen:
    pen = QPen(QColor(t.foreground))
    pen.setStyle(Qt.DashLine)
    return pen


def series_colours(n: int, *, dark: bool) -> list:
    """``n`` distinct line colours, for ANY ``n``.

    A palette for series that are not conditions (a movie is one of dozens),
    kept apart from the conditions' Okabe-Ito set: that one is deliberately
    finite and refuses to wrap, which is right for a handful of conditions and
    wrong for thirty movies. Hues step by the golden angle, so any prefix of
    the sequence is spread around the wheel and neighbours in the list are far
    apart in hue; lightness cycles through three levels so hues that do land
    close are still told apart. Each colour is fitted to the page with the
    app's own `hcl.fit_to_ground`.

    Candidates in the orange band the flag colour lives in are SKIPPED, not
    moved (moving them lands on hues already used), and so is any candidate
    that fits to a colour already handed out — which is what makes ``n``
    colours ``n`` different ones, and keeps the sequence stable: the first
    ``k`` colours are the same whatever ``n`` is.
    """
    ground = hcl.GROUNDS["dark" if dark else "light"]
    lightness = (50.0, 62.0, 74.0)
    out: list = []
    seen: set = set()
    k = 0
    while len(out) < n:
        hue = (k * 137.508 + 200.0) % 360.0
        k += 1
        if 20.0 <= hue <= 60.0:
            continue
        colour = hcl.fit_to_ground(hcl.to_hex(lightness[len(out) % 3], 60.0, hue), ground)
        if colour in seen:
            continue
        seen.add(colour)
        out.append(colour)
    return out


class _QCChart(_MiniChart):
    """The shared plot base for the QC panels.

    Room is spent only where the window has it: a chart shows its x-axis (and
    rotated category labels) once it is tall enough to carry them, and gives
    that margin back to the plot below that. It also thins its ticks and drops
    the rotated y label on a short plot, so the smallest window degrades
    instead of overlapping.
    """

    #: Under the plot when the chart is too short for an x-axis / labels.
    _BOTTOM_MARGIN = 14
    #: Under the plot when it is tall enough (`_ROOMY_H`) to carry them.
    _FULL_BOTTOM = 14
    _ROOMY_H = 10 ** 9
    _RIGHT = 8

    def __init__(self, t, parent=None):
        super().__init__(t, parent)
        self.setMinimumHeight(64)
        self.setMouseTracking(True)
        self._roomy = False
        #: What the last paint drew, for hit-testing: subclasses fill it.
        self._geom: dict = {}
        #: shadcn's chart tooltip (`dynamics_chart._ChartTooltip`), not Qt's native one --
        #: the same card the Spatial Map's bins and the Apparent-D charts show.
        self._tip = _ChartTooltip(t, self, compact=True)
        self._tip_text: str | None = None
        #: The data ink: the condition's colour from the Conditions tool where the
        #: chart shows one condition (`set_data_colour`), else the default blue.
        self.data_colour = _DATA

    def _hover_text(self, pos: QPointF) -> str | None:
        """The tooltip for a point over the plot, or None. Every bar and line
        can be named here, including the ones whose axis label was thinned to
        keep the labels from overlapping."""
        return None

    def mouseMoveEvent(self, e) -> None:
        text = self._hover_text(e.position()) if self._payload is not None else None
        if text:
            self._show_tip(text, e.position())
        else:
            self._hide_tip()
        super().mouseMoveEvent(e)

    def leaveEvent(self, e) -> None:
        self._hide_tip()
        super().leaveEvent(e)

    def hideEvent(self, e) -> None:
        self._hide_tip()
        super().hideEvent(e)

    def set_data_colour(self, colour: str | None) -> None:
        self.data_colour = colour or _DATA
        self.update()

    def _show_tip(self, text: str, pos: QPointF) -> None:
        """Above the pointer, clamped to the window, flipped below when there is no room
        (`dynamics_chart._HoverChart._place`'s rule). Re-filled only when the text changes."""
        try:
            if text != self._tip_text or not self._tip.isVisible():
                self._tip_text = text
                self._tip.set_content(heading=text, rows=(), colour=self.t.muted_foreground)
            window = self.window()
            if self._tip.parentWidget() is not window:
                self._tip.setParent(window)
            w, h = self._tip.width(), self._tip.height()
            anchor = self.mapTo(window, pos.toPoint())
            gap = 10
            x = max(4, min(anchor.x() - w // 2, max(4, window.width() - w - 4)))
            y = anchor.y() - gap - h
            if y < 4:
                y = anchor.y() + gap + 8
            self._tip.move(x, y)
            self._tip.raise_()
            self._tip.show()
        except RuntimeError:
            # the window owns the reparented tooltip and may tear it down first on close
            return

    def _hide_tip(self) -> None:
        self._tip_text = None
        try:
            self._tip.hide()
        except RuntimeError:
            pass

    def _plot_box(self) -> QRectF:
        self._roomy = self.height() >= self._ROOMY_H
        bottom = self._FULL_BOTTOM if self._roomy else self._BOTTOM_MARGIN
        r = self.rect()
        return QRectF(r.left() + self._PLOT_LEFT, r.top() + self._TOP_MARGIN,
                      r.width() - self._PLOT_LEFT - self._RIGHT,
                      max(1.0, r.height() - self._TOP_MARGIN - bottom))

    # -- y axes -------------------------------------------------------------- #

    def _draw_grid(self, p, box: QRectF, ticks: list, y_label: str) -> None:
        """Gridlines, the two axis rules, and the tick labels. ``ticks`` is
        ``[(y_pixel, text), ...]``; the caller decides the scale."""
        dpr = max(1.0, self.devicePixelRatioF())
        pen = QPen(QColor(mix(self.t.border, self.t.background, 0.5)))
        pen.setWidthF(1.0 / dpr)
        p.setPen(pen)
        for y, _text in ticks:
            p.drawLine(QPointF(box.left(), y), QPointF(box.right(), y))
        unit = 1.0 / dpr
        p.fillRect(QRectF(round(box.left() * dpr) / dpr, box.top(), unit, box.height()),
                  QColor(self.t.border))
        p.fillRect(QRectF(box.left(), round(box.bottom() * dpr) / dpr, box.width(), unit),
                  QColor(self.t.border))
        p.setFont(self._tick_font)
        p.setPen(QColor(self.t.muted_foreground))
        fm = QFontMetricsF(self._tick_font)
        for y, text in ticks:
            p.drawText(QRectF(box.left() - self._TICK_RECT_W, y - fm.height() / 2,
                              self._TICK_RECT_W - 4, fm.height()),
                       Qt.AlignRight | Qt.AlignVCenter, text)
        if y_label and box.height() >= 90:
            BubbleChart._y_axis_label(p, box, y_label, self._tick_font,
                                      self.t.muted_foreground)

    def _grid_exact(self, p, box: QRectF, lo: float, hi: float, y_label: str) -> tuple:
        """A linear y axis from exactly ``lo`` to ``hi`` -- a typed range (`axes_view`), which `_grid` would round
        out to a nice tick -- its round ticks inside it."""
        n = 5 if box.height() >= 130 else 3 if box.height() >= 80 else 2
        _a, _b, ticks = self._nice_ticks(lo, hi, n)
        self._draw_grid(p, box, [(self._map_y(box, v, lo, hi), f"{v:g}") for v in ticks
                                 if lo - 1e-12 <= v <= hi + 1e-12], y_label)
        return lo, hi

    def _grid(self, p, box: QRectF, lo: float, hi: float, y_label: str):
        """A linear y axis, thinned to what fits. Returns ``(axis_lo, axis_hi)``."""
        n = 5 if box.height() >= 130 else 3 if box.height() >= 80 else 2
        axis_lo, axis_hi, ticks = self._nice_ticks(lo, hi, n)
        self._draw_grid(p, box, [(self._map_y(box, v, axis_lo, axis_hi), f"{v:g}")
                                 for v in ticks], y_label)
        return axis_lo, axis_hi

    @staticmethod
    def _map_log(box: QRectF, v: float, lo: float, hi: float) -> float:
        f = (math.log10(max(v, lo)) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
        return box.bottom() - f * box.height()

    def _grid_log(self, p, box: QRectF, lo: float, hi: float, y_label: str) -> None:
        """A log10 y axis with a tick at every decade inside ``[lo, hi]``."""
        decades = [10.0 ** e for e in range(math.floor(math.log10(lo)),
                                            math.ceil(math.log10(hi)) + 1)
                   if lo * 0.999 <= 10.0 ** e <= hi * 1.001]
        if box.height() < 80 and len(decades) > 3:
            decades = decades[::2]
        self._draw_grid(p, box, [(self._map_log(box, v, lo, hi), f"{v:g}")
                                 for v in decades], y_label)

    # -- x axes -------------------------------------------------------------- #

    def _x_numeric(self, p, box: QRectF, lo: float, hi: float, title: str) -> None:
        """Numeric x ticks and a title under the plot, when there is room."""
        if not self._roomy or hi <= lo:
            return
        _, _, ticks = self._nice_ticks(lo, hi, max(2, int(box.width() // 80)))
        p.setFont(self._tick_font)
        p.setPen(QColor(self.t.muted_foreground))
        fm = QFontMetricsF(self._tick_font)
        for v in ticks:
            if v < lo - 1e-9 or v > hi + 1e-9:
                continue
            x = box.left() + (v - lo) / (hi - lo) * box.width()
            # centred on its tick, but kept inside the widget: a typed range (`axes_view`) often ends ON a tick at
            # the plot's edge, where a centred "600" lost its last digit
            text = f"{v:g}"
            w = fm.horizontalAdvance(text) + 2
            left = min(max(x - w / 2, 0.0), self.width() - w)
            p.drawText(QRectF(left, box.bottom() + 3, w, 14), Qt.AlignHCenter | Qt.AlignTop, text)
        p.drawText(QRectF(box.left(), box.bottom() + 19, box.width(), 14),
                   Qt.AlignHCenter | Qt.AlignTop, title)

    def _x_categories(self, p, box: QRectF, names) -> None:
        """One rotated label per category (thinned so they never overlap),
        with the part every label shares removed."""
        n = len(names)
        if not self._roomy or n == 0:
            return
        names = short_labels(names)
        font = ty.ui("xs", px=_CATEGORY_PX, tabular=True)
        fm = QFontMetricsF(font)
        pitch = box.width() / n
        step = max(1, math.ceil((fm.height() + 2) / pitch))
        p.setFont(font)
        p.setPen(QColor(self.t.muted_foreground))
        for i in range(0, n, step):
            w = fm.horizontalAdvance(names[i])
            p.save()
            p.translate(box.left() + (i + 0.5) * pitch, box.bottom() + 4)
            p.rotate(-90)
            p.drawText(QRectF(-w - 2, -fm.height() / 2, w + 2, fm.height()),
                       Qt.AlignRight | Qt.AlignVCenter, names[i])
            p.restore()

    # -- legend -------------------------------------------------------------- #

    @staticmethod
    def _legend_metrics(entries: list, horizontal: bool) -> tuple:
        """``(font, metrics, row height, swatch width, widths, gap, total width, total height)`` of a key: the one
        measure `_legend` draws by and `legend_size` reports."""
        font = ty.ui("xs", px=10)
        fm = QFontMetricsF(font)
        h = fm.height() + 2
        sw = 12.0
        widths = [sw + 5 + fm.horizontalAdvance(text) for _c, text, _k in entries]
        gap = 12.0
        total_w = (sum(widths) + gap * (len(entries) - 1)) if horizontal else max(widths)
        total_h = h if horizontal else h * len(entries)
        return font, fm, h, sw, widths, gap, total_w, total_h

    def legend_size(self, entries: list, *, horizontal: bool = False) -> tuple:
        """``(width, height)`` in px a key of ``entries`` covers, its backdrop included (4 px beside, 2 above and
        below the rows)."""
        *_rest, total_w, total_h = self._legend_metrics(entries, horizontal)
        return total_w + 8.0, total_h + 4.0

    def _legend(self, p, entries: list, at: QPointF, *, horizontal: bool,
                right_aligned: bool = False, backdrop: bool = True) -> None:
        """``entries``: ``[(colour, text, kind), ...]`` with kind in
        ``box`` / ``line`` / ``dot`` / ``edge`` / ``shape:<k>`` (population k's line style and marker). Drawn from ``at`` (its top-left,
        or its top-right when ``right_aligned``)."""
        font, fm, h, sw, widths, gap, total_w, total_h = self._legend_metrics(entries, horizontal)
        x0 = at.x() - total_w if right_aligned else at.x()
        if backdrop:
            bg = QColor(self.t.card)
            bg.setAlpha(215)
            p.setPen(Qt.NoPen)
            p.setBrush(bg)
            p.drawRoundedRect(QRectF(x0 - 4, at.y() - 2, total_w + 8, total_h + 4), 3, 3)
        p.setFont(font)
        x, y = x0, at.y()
        for (colour, text, kind), w in zip(entries, widths):
            cy = y + h / 2
            if kind == "line":
                p.setPen(QPen(QColor(colour), 2.0))
                p.drawLine(QPointF(x, cy), QPointF(x + sw, cy))
            elif kind.startswith("shape:"):
                # a population's line style AND marker (`population_style.SHAPES`), as its chart draws it
                shape = SHAPES[int(kind.split(":")[1])]
                pen = QPen(QColor(colour), 2.0)
                if shape["dash"]:
                    pen.setDashPattern([float(d) for d in shape["dash"]])
                p.setPen(pen)
                p.drawLine(QPointF(x, cy), QPointF(x + sw, cy))
                p.setPen(QPen(QColor(colour), 1.3))
                p.setBrush(QColor(self.t.card))
                draw_marker(p, shape["marker"], QPointF(x + sw / 2, cy), 2.8)
            elif kind == "dash":
                dash = QPen(QColor(colour), 1.6)
                dash.setStyle(Qt.DashLine)
                p.setPen(dash)
                p.drawLine(QPointF(x, cy), QPointF(x + sw, cy))
            elif kind == "dot":
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(colour))
                p.drawEllipse(QPointF(x + sw / 2, cy), 3.2, 3.2)
            elif kind == "edge":
                p.setPen(QPen(QColor(colour), 1.6))
                p.setBrush(Qt.NoBrush)
                p.drawRect(QRectF(x + 1, cy - 4, sw - 2, 8))
            else:
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(colour))
                p.drawRect(QRectF(x + 1, cy - 4, sw - 2, 8))
            p.setPen(QColor(self.t.muted_foreground))
            p.drawText(QRectF(x + sw + 5, y, w, h), Qt.AlignLeft | Qt.AlignVCenter, text)
            if horizontal:
                x += w + gap
            else:
                y += h


class TrackLengthChart(_QCChart):
    """Panel 1 — track-length histogram on a LOG y axis, the ≤``min_steps`` cut
    marked, as the original dashboard draws it: track lengths pile up at the
    minimum and thin out over orders of magnitude, so a linear axis hides the
    whole tail.

    ``payload`` is `qc_view.length_payload`'s plain dict (counts, edges, which bins the
    filter keeps) — nothing here recomputes the histogram.
    """

    _FULL_BOTTOM = 40
    _ROOMY_H = 160

    def _paint(self, p, box: QRectF, pl) -> None:
        counts, edges = pl["counts"], pl["edges"]
        if not counts or edges[-1] <= edges[0]:
            return
        top = max(max(counts), 1.0)
        lo, hi = 0.5, 10.0 ** max(1, math.ceil(math.log10(top * 1.0001)))
        self._grid_log(p, box, lo, hi, chart_text.QC_TRACKS_LABEL)
        span = edges[-1] - edges[0]

        def px(step: float) -> float:
            return box.left() + (step - edges[0]) / span * box.width()

        removed = QColor(self.t.muted_foreground)
        removed.setAlpha(110)
        p.setPen(Qt.NoPen)
        for i, c in enumerate(counts):
            if c <= 0:
                continue
            kept = pl["kept"][i]
            x0, x1 = px(edges[i]), px(edges[i + 1])
            y = self._map_log(box, float(c), lo, hi)
            p.fillRect(QRectF(x0 + 0.5, y, max(1.0, x1 - x0 - 1.0), box.bottom() - y),
                      QColor(self.data_colour) if kept else removed)
        pen = QPen(QColor(_FLAG), 1.4)
        pen.setStyle(Qt.DashLine)
        p.setPen(pen)
        cuts = [pl["min_steps"] + 0.5]
        if pl.get("max_steps") is not None:            # a user's cap: its own dashed line
            cuts.append(pl["max_steps"] + 0.5)
        for step in cuts:
            cut = px(step)
            p.drawLine(QPointF(cut, box.top()), QPointF(cut, box.bottom()))
        self._geom = {"box": box}
        self._x_numeric(p, box, edges[0], edges[-1], chart_text.QC_STEPS_LABEL)
        if box.height() >= 90 and box.width() >= 260:
            self._legend(p, [(mix(self.t.muted_foreground, self.t.card, 110 / 255),
                              "removed by the filter", "box"),
                             (self.data_colour, "kept", "box")],
                         QPointF(box.right() - 6, box.top() + 4), horizontal=False,
                         right_aligned=True)

    def _hover_text(self, pos: QPointF) -> str | None:
        g = self._geom
        if not g or not g["box"].contains(pos):
            return None
        edges = self._payload["edges"]
        f = (pos.x() - g["box"].left()) / g["box"].width()
        return length_hover(self._payload, edges[0] + f * (edges[-1] - edges[0]))


class CellLengthRows(_QCChart):
    """One row per condition on ONE length axis: the middle half as a bar in the condition's colour, the median as a
    tick, 5-95 % as whiskers; the name on the left, the median and the cell count on the right (in the warning colour
    when marked). Every number is `qc_view.cell_length_comparison`'s; this only draws."""

    ROW_H = 24
    _TOP = 6
    _AXIS = 36
    _NAME_W = 220
    _VALUE_W = 132

    def __init__(self, t, parent=None):
        super().__init__(t, parent)
        self._colours: dict = {}
        self.setMinimumHeight(self._TOP + self.ROW_H + self._AXIS)

    def set_rows(self, payload: dict, colours: dict | None = None) -> None:
        self._colours = dict(colours or {})
        n = max(1, len(payload.get("rows") or []))
        self.setFixedHeight(self._TOP + n * self.ROW_H + self._AXIS)
        self.set_payload(payload)

    def _plot_box(self) -> QRectF:
        self._roomy = True
        name_w = min(self._NAME_W, 0.3 * self.width())
        r = self.rect()
        return QRectF(r.left() + name_w + 12, r.top() + self._TOP,
                      max(1.0, r.width() - name_w - 12 - self._VALUE_W - 8),
                      max(1.0, r.height() - self._TOP - self._AXIS))

    def _paint(self, p, box: QRectF, pl) -> None:
        rows = pl["rows"]
        lo, hi = pl["axis"]
        if not rows or hi <= lo:
            return
        _, _, ticks = self._nice_ticks(lo, hi, max(2, int(box.width() // 80)))

        def px(v: float) -> float:
            return box.left() + (min(max(v, lo), hi) - lo) / (hi - lo) * box.width()

        dpr = max(1.0, self.devicePixelRatioF())
        grid = QPen(QColor(mix(self.t.border, self.t.background, 0.5)))
        grid.setWidthF(1.0 / dpr)
        p.setPen(grid)
        for v in ticks:
            if lo - 1e-9 <= v <= hi + 1e-9:
                p.drawLine(QPointF(px(v), box.top()), QPointF(px(v), box.bottom()))
        p.fillRect(QRectF(box.left(), round(box.bottom() * dpr) / dpr, box.width(), 1.0 / dpr),
                   QColor(self.t.border))
        self._x_numeric(p, box, lo, hi, chart_text.QC_CELL_LENGTH_LABEL)

        name_font = ty.ui("sm")
        value_font = ty.ui("sm", tabular=True)
        fm = QFontMetricsF(name_font)
        name_w = box.left() - 12
        self._geom = {"box": box, "rows": []}
        for i, row_ in enumerate(rows):
            top = box.top() + i * self.ROW_H
            cy = top + self.ROW_H / 2.0
            self._geom["rows"].append((top, top + self.ROW_H, row_))
            colour = QColor(self._colours.get(row_["condition"], self.data_colour))
            p.setPen(Qt.NoPen)
            p.setBrush(colour)
            p.drawEllipse(QPointF(8.0, cy), 4.0, 4.0)
            p.setFont(name_font)
            p.setPen(QColor(self.t.foreground))
            p.drawText(QRectF(18.0, top, name_w - 18.0, self.ROW_H), Qt.AlignLeft | Qt.AlignVCenter,
                       fm.elidedText(row_["condition"], Qt.ElideMiddle, name_w - 18.0))
            value_box = QRectF(box.right() + 8, top, self._VALUE_W, self.ROW_H)
            if row_.get("error"):
                p.setFont(ty.ui("xs"))
                p.setPen(QColor(self.t.muted_foreground))
                p.drawText(QRectF(box.left() + 4, top, box.width() - 4, self.ROW_H),
                           Qt.AlignLeft | Qt.AlignVCenter,
                           QFontMetricsF(ty.ui("xs")).elidedText(row_["error"], Qt.ElideRight,
                                                                 box.width() - 8))
                continue
            whisker = QPen(QColor(self.t.muted_foreground), 1.2)
            p.setPen(whisker)
            p.drawLine(QPointF(px(row_["p5"]), cy), QPointF(px(row_["p95"]), cy))
            for v in (row_["p5"], row_["p95"]):
                p.drawLine(QPointF(px(v), cy - 4), QPointF(px(v), cy + 4))
            fill = QColor(colour)
            fill.setAlpha(90)
            p.setPen(QPen(colour, 1.2))
            p.setBrush(fill)
            p.drawRoundedRect(QRectF(px(row_["q25"]), cy - 6, max(2.0, px(row_["q75"]) - px(row_["q25"])), 12),
                              2, 2)
            p.setPen(QPen(QColor(self.t.foreground), 2.2))
            p.drawLine(QPointF(px(row_["median"]), cy - 7), QPointF(px(row_["median"]), cy + 7))
            marked = bool(row_.get("marked"))
            p.setFont(ty.ui("sm", "semibold", tabular=True) if marked else value_font)
            p.setPen(QColor(_WARN_COLOR if marked else self.t.muted_foreground))
            p.drawText(value_box, Qt.AlignLeft | Qt.AlignVCenter,
                       f"{row_['median']:.2f} \u00b5m \u00b7 {row_['n']} cells")

    def _hover_text(self, pos: QPointF) -> str | None:
        for top, bottom, row_ in (self._geom or {}).get("rows", []):
            if top <= pos.y() < bottom:
                return cell_length_hover(row_)
        return None


class QCCellLengthCard(Card):
    """CELL LENGTH (asked 15 Oct 2026: "would be cool to have cell length distribution in qc"): every condition's
    outlined cells on one length axis, the General page's view of whether the cells being compared are alike --
    `qc_view.cell_length_comparison`; the rule is the heading's tooltip, a marked row is drawn in the warning
    colour. Rows, not overlaid histograms: a row per condition stays readable at twenty conditions."""

    def __init__(self, t, parent=None):
        super().__init__(t, parent=parent)
        self.t = t
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 14)
        self._badge = Badge(t, "0", t.muted, t.foreground)
        head, self._caption = _card_head(t, "ruler", "Cell length",
                                         "Are the cells being compared alike?", self._badge)
        self._tip = tooltip.attach(t, head, CELL_LENGTH_RULE)
        lay.addWidget(head)
        self.rows = CellLengthRows(t)
        body = QHBoxLayout()
        body.setContentsMargins(20, 0, 20, 0)
        body.addWidget(self.rows)
        lay.addLayout(body)

    def set_payload(self, cells: dict | None, colours: dict | None = None) -> None:
        if not cells or not cells.get("rows"):
            self.rows.set_message("No outlined cells to measure.")
            self._caption.setText("")
            self._badge.set_text("0")
            return
        self.rows.set_rows(cells, colours)
        self._caption.setText(cells["caption"])
        n = int(cells.get("n_marked", 0))
        self._badge.set_text(str(n))
        self._badge.set_colours(_WARN_COLOR if n else self.t.muted,
                                "#FFFFFF" if n else self.t.foreground)


class GroupCountsTable(QWidget):
    """Panel 3 — N per group (day/movie), table-shaped, not a chart — the
    sample size made visible. ``payload`` is `qc_view.group_table`'s own
    dict (``columns`` and ``rows`` as text); the condition column is already dropped
    (this page is one condition).

    It runs edge to edge inside its card. The rows scroll, so the last visible
    row is not the table's last row and `TableBody`'s own bottom rounding does
    not apply; instead this widget is clipped to the card's rounded bottom
    corners (`set_bottom_radius`), so nothing pokes through them.
    """

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._table: TableBody | None = None
        self._message = label("Not yet computed.", ty.ui("sm"), t.muted_foreground)
        self._lay.addWidget(self._message)
        # The registry's Table wrapper is `overflow-x-auto`: when the card is
        # narrower than the columns need, the table scrolls sideways instead of
        # letting its columns overlap. Wide enough, there is no scrollbar.
        self._area = QScrollArea()
        self._area.setFrameShape(QScrollArea.NoFrame)
        self._area.setWidgetResizable(True)
        self._area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._area.setStyleSheet("QScrollArea { background:transparent; border:none; }")
        self._area.hide()
        self._lay.addWidget(self._area, 1)
        self.setMinimumHeight(rem(4))
        self._bottom_radius = 0.0

    def sizeHint(self) -> QSize:
        """Asks for its minimum, not for room for every row: the table
        scrolls inside its card, and a large preferred height would stop the
        page's row weights from making this row as short as it should be."""
        return self.minimumSizeHint()

    def set_message(self, text: str) -> None:
        self._message.setText(text)
        self._message.show()
        self._area.hide()

    def set_bottom_radius(self, radius: float) -> None:
        """Round this widget's bottom corners to the card it sits flush in."""
        self._bottom_radius = float(radius)
        self._apply_mask()

    def _apply_mask(self) -> None:
        if self._bottom_radius <= 0 or self.width() < 2 or self.height() < 2:
            self.clearMask()
            return
        w, h, r = self.width(), self.height(), self._bottom_radius
        bits = QBitmap(w, h)
        bits.fill(Qt.color0)
        p = QPainter(bits)
        p.setRenderHint(QPainter.Antialiasing, False)
        p.setPen(Qt.NoPen)
        p.setBrush(Qt.color1)
        p.drawRoundedRect(QRectF(0.0, 0.0, float(w), float(h)), r, r)
        p.drawRect(QRectF(0.0, 0.0, float(w), h / 2.0))      # the top corners stay square
        p.end()
        self.setMask(bits)

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._apply_mask()

    def set_payload(self, table: dict) -> None:
        self._message.hide()
        if self._table is not None:
            self._area.takeWidget()
            self._table.deleteLater()
        columns, rows = table["columns"], table["rows"]
        self._table = TableBody(self.t, columns, rows, "", scrollable=True)
        # What the columns need: their widths, the row's own inset, and the
        # scrollbar gutter the table reserves.
        self._table.setMinimumWidth(int(sum(self._table._want)) + 2 * rem(0.75)
                                    + SCROLLBAR_W)
        self._area.setWidget(self._table)
        self._area.show()


class UnitDotsChart(_QCChart):
    """Panel 4 \u2014 one DOT per unit (cell/movie/day) with the median line and
    the median \u00b1 k\u00b7scale band, flagged units larger and orange.
    ``payload`` is ``{"units": qc_view.unit_payload's dict, "ordered": bool,
    "columns": bool}``; the block carries its own flagging results (the usual range,
    the median, which units are flagged) and nothing here re-derives the threshold.

    Dots, not bars: the unit's value is a POSITION against the band, and a dot
    stays visible where 100+ one-pixel bars did not. The axis still starts at
    zero, the original rule (a truncated axis would exaggerate the spread this
    panel exists to judge).

    ``ordered=False`` sorts the units from lowest to highest, so the dots form a
    curve that reads at any count in a small space. ``ordered=True`` keeps the
    order the units were loaded in, with each movie named under its cells.
    ``columns=True`` is the categorical scatter plot: one equal-width column per
    movie (for movies and days, one per unit), its dots spread side by side at
    fixed offsets so they do not sit on top of each other (deterministic: the
    same data always draws the same picture).
    """

    _PLOT_LEFT = 82
    _TICK_RECT_W = 58
    _FULL_BOTTOM = 62
    _ROOMY_H = 200

    def _paint(self, p, box: QRectF, pl) -> None:
        u = pl["units"]
        ordered = bool(pl.get("ordered"))
        columns = bool(pl.get("columns"))
        target, values, names, flagged = u["target"], u["values"], u["names"], u["flagged"]
        finite = [v for v in values if v is not None]
        if target is None or not finite:
            return
        low, high = u["low"], u["high"]
        hi_v = max(finite)
        if high is not None:
            hi_v = max(hi_v, high)
        lo_v = min(0.0, min(finite))
        axis_lo, axis_hi = self._grid(p, box, lo_v, hi_v, u["axis_label"])

        if low is not None and high is not None:
            y_lo = self._map_y(box, low, axis_lo, axis_hi)
            y_hi = self._map_y(box, high, axis_lo, axis_hi)
            band = QColor(self.t.muted_foreground)
            band.setAlpha(34)
            p.fillRect(QRectF(box.left(), y_hi, box.width(), y_lo - y_hi), band)

        unit = u["unit"]
        n = u["n"]
        order = u["order"] if ordered else u["order_sorted"]
        pitch = box.width() / n
        ys = [self._map_y(box, v, axis_lo, axis_hi) if v is not None else None for v in values]
        groups = None
        if columns:
            groups = column_groups(names, unit)
            colw = box.width() / len(groups)
            radius = min(4.0, max(1.0, colw * 0.12))
            xs = self._swarm_x(box, groups, values, ys, radius, colw)
        else:
            # Every unit is drawn in the width there is: the dots shrink with the pitch
            # instead of the chart scrolling (a slice of the data loses the shape this
            # panel is for), and overlapping dots are a little translucent so a crowd
            # reads as density. Only space limits this, never a count.
            radius = min(4.5, max(0.8, pitch * 0.42))
            xs = [0.0] * n
            for rank, i in enumerate(order):
                xs[i] = box.left() + (rank + 0.5) * pitch
        flag_radius = max(3.0, radius + 1.8)
        self._geom = {"box": box, "n": n, "order": order, "xs": xs, "ys": ys,
                      "radius": radius, "groups": groups}

        movies = u["movies"] if (ordered and unit == "cell") else None

        p.setPen(Qt.NoPen)
        ring = QColor(self.t.foreground)
        ring.setAlpha(150)
        dot = QColor(self.data_colour)
        if radius < 1.6:
            dot.setAlpha(200)
        for pass_flagged in (False, True):            # flagged dots are drawn last, on top
            for i in range(n):
                if flagged[i] != pass_flagged or values[i] is None:
                    continue
                c = QPointF(xs[i], ys[i])
                if pass_flagged:
                    p.setBrush(QColor(_FLAG))
                    p.setPen(QPen(ring, 1.2))
                    p.drawEllipse(c, flag_radius, flag_radius)
                    p.setPen(Qt.NoPen)
                else:
                    p.setBrush(dot)
                    p.drawEllipse(c, radius, radius)

        if u["has_spread"]:
            my = self._map_y(box, u["median"], axis_lo, axis_hi)
            line = QColor(self.t.foreground)
            line.setAlpha(200)
            p.setPen(QPen(line, 1.0))
            p.drawLine(QPointF(box.left(), my), QPointF(box.right(), my))

        if not self._roomy:
            return
        if columns:
            self._column_labels(p, box, [g[0] for g in groups], box.width() / len(groups))
        elif not ordered:
            p.setFont(self._tick_font)
            p.setPen(QColor(self.t.muted_foreground))
            p.drawText(QRectF(box.left(), box.bottom() + 6, box.width(), 14),
                       Qt.AlignHCenter | Qt.AlignTop, sorted_axis_title(unit))
        elif movies is None:
            self._x_categories(p, box, names)
        else:
            self._movie_labels(p, box, [movies[i] for i in order], pitch)

    @staticmethod
    def _swarm_x(box: QRectF, groups: list, values, ys, radius: float, colw: float):
        """Horizontal position for every unit (None for a unit without a value): each
        column's dots are placed, lowest value first, at the slot nearest the column's
        centre where they do not touch a dot already placed. Fixed slot order, so
        nothing is random."""
        xs: list = [None] * len(values)
        step = 2.0 * radius
        half = max(0.0, colw / 2.0 - radius - 0.5)
        slots = [0.0]
        k = 1
        while k * step <= half:
            slots += [k * step, -k * step]
            k += 1
        for gi, (_name, idx) in enumerate(groups):
            cx = box.left() + (gi + 0.5) * colw
            placed: list = []                       # (y, offset), y falling as values rise
            for i in sorted((j for j in idx if values[j] is not None), key=lambda j: values[j]):
                y = ys[i]
                chosen = None
                for slot in slots:
                    clear = True
                    for py, px in reversed(placed):
                        if abs(y - py) >= step:
                            break
                        if (slot - px) ** 2 + (y - py) ** 2 < step * step:
                            clear = False
                            break
                    if clear:
                        chosen = slot
                        break
                if chosen is None:                  # no free slot: spread by rank
                    chosen = slots[len(placed) % len(slots)]
                placed.append((y, chosen))
                xs[i] = cx + chosen
        return xs

    def _column_labels(self, p, box: QRectF, movies: list, colw: float) -> None:
        """One rotated label per column, centred under it, thinned so two never
        overlap."""
        short = short_labels(movies)
        font = ty.ui("xs", px=_CATEGORY_PX, tabular=True)
        fm = QFontMetricsF(font)
        p.setFont(font)
        p.setPen(QColor(self.t.muted_foreground))
        last_x = -1e9
        for gi, text in enumerate(short):
            x = box.left() + (gi + 0.5) * colw
            if x - last_x < fm.height() + 2:
                continue
            last_x = x
            w = fm.horizontalAdvance(text)
            p.save()
            p.translate(x, box.bottom() + 4)
            p.rotate(-90)
            p.drawText(QRectF(-w - 2, -fm.height() / 2, w + 2, fm.height()),
                       Qt.AlignRight | Qt.AlignVCenter, text)
            p.restore()

    def _movie_labels(self, p, box: QRectF, movies: list, pitch: float) -> None:
        """One rotated label per movie, centred under its run of cells, thinned
        so two never overlap."""
        runs = movie_runs(movies)
        short = short_labels([r[0] for r in runs])
        font = ty.ui("xs", px=_CATEGORY_PX, tabular=True)
        fm = QFontMetricsF(font)
        p.setFont(font)
        p.setPen(QColor(self.t.muted_foreground))
        last_x = -1e9
        for (_name, a, b), text in zip(runs, short):
            x = box.left() + (a + b) / 2.0 * pitch
            if x - last_x < fm.height() + 2:
                continue
            last_x = x
            w = fm.horizontalAdvance(text)
            p.save()
            p.translate(x, box.bottom() + 4)
            p.rotate(-90)
            p.drawText(QRectF(-w - 2, -fm.height() / 2, w + 2, fm.height()),
                       Qt.AlignRight | Qt.AlignVCenter, text)
            p.restore()

    def _hover_text(self, pos: QPointF) -> str | None:
        g = self._geom
        if not g or not g["box"].contains(pos) or not g["n"]:
            return None
        u = self._payload["units"]
        if g.get("groups") is not None:
            # Categorical layout: the dot under (or nearest) the pointer, else the
            # column's own summary.
            j, best = 0, float("inf")
            for k, (x, y) in enumerate(zip(g["xs"], g["ys"])):
                if x is None or y is None:
                    continue
                d2 = (x - pos.x()) ** 2 + (y - pos.y()) ** 2
                if d2 < best:
                    j, best = k, d2
            if best <= (g["radius"] + 4.0) ** 2:
                i = j
            else:
                colw = g["box"].width() / len(g["groups"])
                name, idx = g["groups"][min(len(g["groups"]) - 1, int((pos.x() - g["box"].left()) / colw))]
                return column_hover(name, len(idx))
        else:
            k = min(g["n"] - 1, int((pos.x() - g["box"].left()) / g["box"].width() * g["n"]))
            i = g["order"][k]
        return unit_hover(u, i)


class InCellShareChart(_QCChart):
    """Panel 5 — tracks found per movie, stacked inside/outside a cell
    outline, a movie unlike its condition marked with an orange edge, and the
    share inside as a line on a second axis (0-100 %).

    ``payload`` is `qc_view.in_cell_payload`'s plain dict. Where the spine function
    raised `QCError` (no per-movie cell-filter counts recorded — see its
    own docstring), the caller calls `set_message` with the caught
    message instead of `set_payload` — SURFACED, not swallowed into a
    blank panel, correcting the legacy `QCDashboardWidget`'s known drift
    from its own documented intent.
    """

    _FULL_BOTTOM = 62
    _ROOMY_H = 200
    _RIGHT = 60

    def _paint(self, p, box: QRectF, pl) -> None:
        names = pl["names"]
        if not names:
            return
        found, inside, share, flagged = pl["found"], pl["inside"], pl["share"], pl["flagged"]
        n = len(names)
        self._geom = {"box": box, "n": n}
        _, axis_hi = self._grid(p, box, 0.0, float(max(max(found), 1.0)), chart_text.QC_TRACKS_LABEL)
        pitch = box.width() / n
        outside_ink = QColor(self.t.muted_foreground)
        outside_ink.setAlpha(110)
        p.setPen(Qt.NoPen)
        for i in range(n):
            x = box.left() + i * pitch + 1
            w = max(1.0, pitch - 2)
            y_in = self._map_y(box, float(inside[i]), 0.0, axis_hi)
            y_all = self._map_y(box, float(found[i]), 0.0, axis_hi)
            p.fillRect(QRectF(x, y_in, w, box.bottom() - y_in), QColor(self.data_colour))
            p.fillRect(QRectF(x, y_all, w, y_in - y_all), outside_ink)
            if flagged[i]:
                p.setPen(QPen(QColor(_FLAG), 1.6))
                p.setBrush(Qt.NoBrush)
                p.drawRect(QRectF(x, y_all, w, max(2.0, box.bottom() - y_all)))
                p.setPen(Qt.NoPen)

        # share inside, on its own 0-105 % axis at the right
        def sy(v: float) -> float:
            return box.bottom() - v / 105.0 * box.height()

        ink = QColor(self.t.foreground)
        ink.setAlpha(200)
        path = QPainterPath()
        started = False
        for i in range(n):
            if share[i] is None:
                continue
            cx = box.left() + (i + 0.5) * pitch
            if started:
                path.lineTo(cx, sy(share[i]))
            else:
                path.moveTo(cx, sy(share[i]))
                started = True
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(ink, 1.0))
        p.drawPath(path)
        p.setPen(Qt.NoPen)
        p.setBrush(ink)
        for i in range(n):
            if share[i] is not None:
                p.drawEllipse(QPointF(box.left() + (i + 0.5) * pitch, sy(share[i])),
                              1.8, 1.8)
        p.setFont(self._tick_font)
        p.setPen(QColor(self.t.muted_foreground))
        fm = QFontMetricsF(self._tick_font)
        if box.height() >= 80:
            for v in (0, 50, 100):
                p.drawText(QRectF(box.right() + 4, sy(v) - fm.height() / 2, 40, fm.height()),
                           Qt.AlignLeft | Qt.AlignVCenter, f"{v}")
        p.drawText(QRectF(box.right() + 4, box.top() - 16, 56, 14),
                   Qt.AlignLeft | Qt.AlignVCenter, chart_text.QC_INSIDE_LABEL)

        entries = [(self.data_colour, "inside a cell", "box"),
                   (mix(self.t.muted_foreground, self.t.card, 110 / 255),
                    "outside every outline", "box")]
        if any(flagged):
            entries.append((_FLAG, "unlike its condition", "edge"))
        self._legend(p, entries, QPointF(box.left(), 0.0), horizontal=True,
                     backdrop=False)
        self._x_categories(p, box, names)

    def _hover_text(self, pos: QPointF) -> str | None:
        g = self._geom
        if not g or not g["box"].contains(pos) or not g["n"]:
            return None
        i = min(g["n"] - 1, int((pos.x() - g["box"].left()) / g["box"].width() * g["n"]))
        return in_cell_hover(self._payload, i)


class NeighbourShareChart(_QCChart):
    """The Localizations-per-frame card's chart: one bar per movie, the share
    of its localizations that have another localization within the search
    radius (`qc_view.compute_qc_neighbour_share`), the usual range as a grey band, the
    median as a line, and movies unlike the rest in orange.

    The flag is RELATIVE (median ± k·MAD across movies), so the bar heights
    carry the size and the card's caption prints the median: if every movie is
    crowded, nothing turns orange, and this project has no established cut-off
    for an unacceptable share that could be drawn instead.
    """

    _FULL_BOTTOM = 62
    _ROOMY_H = 200

    def _paint(self, p, box: QRectF, pl) -> None:
        names = pl["names"]
        if not names:
            return
        share = pl["share"]
        low, high, median = pl["low"], pl["high"], pl["median"]
        hi_v = max(share)
        if high is not None:
            hi_v = max(hi_v, high)
        axis_lo, axis_hi = self._grid(p, box, 0.0, max(hi_v, 1.0), chart_text.QC_NEIGHBOUR_LABEL)
        if low is not None and high is not None:
            y_hi = self._map_y(box, float(high), axis_lo, axis_hi)
            y_lo = self._map_y(box, max(float(low), 0.0), axis_lo, axis_hi)
            band = QColor(self.data_colour)
            band.setAlpha(34)
            p.fillRect(QRectF(box.left(), y_hi, box.width(), y_lo - y_hi), band)
        n = len(names)
        pitch = box.width() / n
        flagged = pl["flagged"]
        p.setPen(Qt.NoPen)
        for i in range(n):
            y = self._map_y(box, float(share[i]), axis_lo, axis_hi)
            p.fillRect(QRectF(box.left() + i * pitch + 1, y, max(1.0, pitch - 2),
                              box.bottom() - y), QColor(_FLAG if flagged[i] else self.data_colour))
        if median is not None:
            my = self._map_y(box, float(median), axis_lo, axis_hi)
            line = QColor(self.t.foreground)
            line.setAlpha(150)
            p.setPen(QPen(line, 1.0))
            p.drawLine(QPointF(box.left(), my), QPointF(box.right(), my))
            if box.height() >= 90:
                p.setFont(self._tick_font)
                p.setPen(QColor(self.t.muted_foreground))
                p.drawText(QRectF(box.right() - 130, box.top() - 15, 128, 13),
                           Qt.AlignRight | Qt.AlignVCenter, f"median {median:.1f}%")
        self._geom = {"box": box, "n": n, "axis": (axis_lo, axis_hi)}
        self._x_categories(p, box, names)

    def _hover_text(self, pos: QPointF) -> str | None:
        g = self._geom
        if not g or not g["box"].contains(pos) or not g["n"]:
            return None
        i = min(g["n"] - 1, int((pos.x() - g["box"].left()) / g["box"].width() * g["n"]))
        pl = self._payload
        low, high = pl["low"], pl["high"]
        if low is not None and high is not None:
            axis_lo, axis_hi = g["axis"]
            y_top = self._map_y(g["box"], high, axis_lo, axis_hi)
            y_bar = self._map_y(g["box"], pl["share"][i], axis_lo, axis_hi)
            # The band, where no bar is under the pointer: the bar's own tooltip
            # keeps priority over it and over everything outside it.
            if y_top <= pos.y() < y_bar:
                return neighbour_band_text(pl)
        return neighbour_bar_text(pl, i)


class _Caption(FitLabel):
    """A wrapped caption whose height follows its width: a `primitives.FitLabel`, so it
    never makes the layouts above it height-for-width (which cost the QC detail page
    ~26 ms of layout per frame while the sidebar slid)."""

    def __init__(self, t, parent=None, size: str = "xs"):
        super().__init__(parent=parent)
        # `xs`, like every other card subtitle in the app ("Loaded conditions
        # · ...", "What the files recorded · ..."), so the chart leads.
        self.setFont(ty.ui(size))
        self.setStyleSheet(f"color:{t.muted_foreground}; background:transparent;")
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)


class _PanelCard(Card):
    """One panel of the Detail grid: a title, a caption that says what the
    panel is and prints its own N and threshold, and the chart under them."""

    inspect_requested = Signal()
    #: `enable_export`'s button: the section opens `export_dialog.ExportDialog`.
    export_requested = Signal()

    def __init__(self, t, title: str, body: QWidget, parent=None, *, flush: bool = False):
        """``flush=True`` lets the body run edge to edge and down to the card's
        bottom (a table), with the title and caption keeping their own inset,
        the way the Conditions table sits in its card."""
        super().__init__(t, parent=parent)
        self._inspect: Button | None = None
        self._export = None
        lay = QVBoxLayout(self)
        if flush:
            lay.setContentsMargins(0, 14, 0, 0)
        else:
            lay.setContentsMargins(20, 14, 20, 14)
        lay.setSpacing(4)
        # THE TITLE FITS ITS CARD (30 Oct 2026, "when something gets cut -> scale down text size"): smaller in steps
        # to 11 px, then elided in the middle with the whole title on hover (`ShrinkLabel`, `chart_text.fit_title`),
        # clear of the Axes / Export buttons floating over its right end (`_place_inspect`). A plain label asked for
        # its whole text as the card's minimum width: a tile's long title widened its column or ran under them.
        self._title = ShrinkLabel(title, section_font(), t.foreground, t=t)
        self._caption = _Caption(t)
        self._caption.hide()
        if flush:
            for w in (self._title, self._caption):
                w.setContentsMargins(20, 0, 20, 0)
            body.set_bottom_radius(self.radius)
        lay.addWidget(self._title)
        lay.addWidget(self._caption)
        lay.addSpacing(4)
        lay.addWidget(body, 1)

    def set_title(self, text: str) -> None:
        self._title.setText(text)

    def set_caption(self, text: str) -> None:
        self._caption.setText(text)
        self._caption.setVisible(bool(text))

    def enable_inspect(self, text: str = "Inspect", glyph: str = "scan-search") -> None:
        """A button at the card's top right that asks for the detailed views, in the
        same style as the Comparator's "Configure" (small, with its glyph and its word).
        It floats over the header instead of joining its layout, so the cards in a row
        keep the same baselines."""
        btn = Button(self.t, text, size="small", glyph=glyph)
        # Smaller than the shared "small" button, to the registry's `xs` (h-6, text-xs,
        # size-3 glyph): beside a card title the full small size reads as heavy. Done
        # here, to THIS button, so the shared widget and the Comparator's Configure keep
        # their size.
        btn.setFixedHeight(24)
        btn.title.setFont(ty.ui("xs", "medium"))
        for glyph in btn.findChildren(Icon):
            glyph.setFixedSize(12, 12)
        btn.layout().setContentsMargins(8, 0, 8, 0)
        btn.layout().setSpacing(4)
        btn.setParent(self)
        btn.clicked.connect(self.inspect_requested.emit)
        btn.adjustSize()
        btn.show()
        btn.raise_()
        self._inspect = btn
        self._place_inspect()

    def set_inspect_enabled(self, on: bool) -> None:
        if self._inspect is not None:
            self._inspect.setEnabled(on)

    def enable_export(self, tip_text: str = "Export figure (PDF, SVG, PNG, TIFF)") -> None:
        """An Export button at the card's top right (left of Inspect, if any), floating
        over the header like Inspect so the cards in a row keep their baselines."""
        from ..widgets import tooltip as _tip
        from .toolbar import ToolButton

        if self._export is not None:
            return
        btn = ToolButton(self.t, "export", "download", size=rem(1.75), rest=self.t.card)
        btn.setParent(self)
        btn.clicked.connect(lambda _k: self.export_requested.emit())
        self._export_tip = _tip.attach(self.t, btn, tip_text, side="bottom", top_level=True)
        btn.show()
        btn.raise_()
        self._export = btn
        self._place_inspect()

    def _place_inspect(self) -> None:
        right = self.width() - 14
        if self._inspect is not None:
            self._inspect.move(right - self._inspect.width(), 13)
            right -= self._inspect.width() + 6
        if self._export is not None:
            self._export.move(right - self._export.width(), 10)
            right -= self._export.width() + 6
        # the title stops short of the buttons over its right end, 8 px clear of them
        if self._inspect is not None or self._export is not None:
            end = self._title.mapTo(self, self._title.contentsRect().topRight()).x()
            self._title.set_reserve(end - right + 8)

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._place_inspect()


#: name, the small value line in the list, then the one-sentence purpose and the
#: three tips (shows / reads as / catch), each written for someone who has not
#: met the panel before.
_INSPECT_TEXT = {
    "share": (
        "Neighbor share",
        "PER MOVIE",
        ("Shows the percentage of localisations per movie that have another "
         "localisation within {radius} in the same frame. The higher it is, "
         "the more likely the tracker links the wrong molecules. Most movies fall in the "
         "blue band; movies outside it are orange. Only localisations in tracks "
         "are counted, so the true share is at least this high."),
    ),
    "distribution": (
        "Frame distribution",
        "ALL MOVIES",
        ("Shows the distribution of localisations per frame across all movies. The "
         "orange line shows the same distribution for the movie with the most "
         "localisations per frame on average."),
    ),
    "outliers": (
        "Outlier frames",
        "BUSIEST PER MOVIE",
        ("Shows each movie's typical frame (dot) and its busiest frame (top of the "
         "bar), as localisations per frame. Orange movies are the ones flagged in "
         "the neighbour-share chart."),
    ),
    "drift": (
        "Density over time",
        "MEDIAN + BAND",
        ("Shows localisations per frame across movies over time, as the position in "
         "each movie (0\u2013100%). The line is the median of all movies, the darker "
         "band the middle 50% of movies and the lighter band the lowest to the "
         "highest movie."),
    ),
    "heat": (
        "Frame crowdedness",
        "HEAT MAP",
        ("Shows the average localisations per frame for every movie (rows) over "
         "time (columns, 0\u2013100% of each movie); darker means more. Orange rows "
         "are the movies flagged in the neighbour-share chart."),
    ),
    "lines": (
        "Per-frame lines",
        "ORIGINAL",
        "One line per movie, frame by frame. Orange dots: above the crowding limit.",
    ),
}


class _QCViewsDialog(OverlayDialog):
    """The detailed views of one Detail card, in the layout of the Map
    Configuration dialog: every view listed on the left with what it is, a
    large picture on the right under a short description, and an optional row
    of controls under the picture.

    It changes nothing and runs nothing, so it has one exit (Close). Only the
    selected view is drawn; each is built the first time it is opened. Rebuilt
    fresh on every open, seeded from the card's condition. The panel keeps one
    height and the card one width for every view, so the dialog does not move
    as you switch.

    A subclass supplies ``TEXT`` (key -> (name, list label, description)),
    ``SUBTITLE``, `_heading_text`, `_build`, and may override `_describe` (fill
    the description in) and `_after_show` (react to the chosen view), adding its
    controls to ``self._controls``.
    """

    #: The panel keeps one height for every view: a short description leaves
    #: the room to the chart instead of shrinking the dialog.
    PANEL_H = 414
    CARD_REM = 59.25
    CHART_H = 290
    TEXT: ClassVar[dict] = {}
    SUBTITLE = ""

    def __init__(self, t, *, content=None, parent=None):
        super().__init__(t, content, parent=parent)
        self.t = t
        # One width for every view (the widest description used to set it, so the
        # first view was 947 px wide and the rest 720), held to the window's width.
        want = rem(self.CARD_REM)
        room = parent.width() - rem(2) if parent is not None else want
        self._card.setMinimumWidth(min(want, room))
        self._entry: dict | None = None
        self._keys: list = []
        self._pages: dict = {}

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(8)
        self._heading = label("", ty.ui("base", "semibold"), t.foreground)
        head.addWidget(self._heading)
        head.addStretch(1)
        close = CloseButton(t)
        close.clicked.connect(self._dismiss)
        head.addWidget(close)
        self.card_lay.addLayout(head)

        self._subtitle = _Caption(t, size="sm")
        self._subtitle.setText(self.SUBTITLE)
        self.card_lay.addWidget(self._subtitle)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(rem(1.25))
        self.nav = _SettingNav(t, [])
        self.nav.changed.connect(self._show)
        body.addWidget(self.nav, 0, Qt.AlignTop)

        panel = QWidget()
        panel.setStyleSheet("background:transparent;")
        panel.setMinimumHeight(self.PANEL_H)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(rem(0.5))
        self._title = label("", ty.ui("sm", "semibold", px=rem(0.9375)), t.foreground)
        self._desc = _Caption(t, size="sm")
        # Words never decide the dialog's width; the card's minimum does.
        self._desc.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Minimum)
        self._stack = QStackedWidget()
        self._stack.setMinimumHeight(self.CHART_H)
        lay.addWidget(self._title)
        lay.addWidget(self._desc)
        lay.addWidget(self._stack, 1)
        #: Where a subclass puts the controls that belong under the picture.
        self._controls = QVBoxLayout()
        self._controls.setContentsMargins(0, 0, 0, 0)
        self._controls.setSpacing(rem(0.5))
        lay.addLayout(self._controls)
        body.addWidget(panel, 1)
        self.card_lay.addLayout(body)

        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.setSpacing(8)
        footer.addStretch(1)
        self._close_btn = Button(t, "Close", variant="default")
        self._close_btn.clicked.connect(self._dismiss)
        footer.addWidget(self._close_btn)
        self.card_lay.addLayout(footer)

    # -- what the caller sets --------------------------------------------------- #

    def _heading_text(self, condition: str) -> str:
        raise NotImplementedError

    def set_context(self, condition: str, entry: dict) -> None:
        """The condition's QC entry."""
        self._entry = entry
        self._heading.setText(self._heading_text(condition))
        self._keys = list(self.TEXT)
        self.nav.set_items([self.TEXT[k][0] for k in self._keys])
        for i, key in enumerate(self._keys):
            self.nav.set_value(i, self.TEXT[key][1])
        self._pages = {}
        self._show(0)

    # -- views ------------------------------------------------------------------- #

    def _build(self, key: str) -> QWidget:
        raise NotImplementedError

    def _describe(self, key: str, desc: str) -> str:
        return desc

    def _after_show(self, key: str) -> None:
        return None

    def set_data_colour(self, colour: str | None) -> None:
        """The condition's colour (Conditions tool) for every chart this dialog
        shows, built already or later."""
        self._data_colour = colour
        for page in getattr(self, "_pages", {}).values():
            for chart in ([page] if isinstance(page, _QCChart) else []) + \
                    page.findChildren(_QCChart):
                chart.set_data_colour(colour)

    def _show(self, index: int) -> None:
        key = self._keys[index]
        name, _sub, desc = self.TEXT[key]
        if key not in self._pages:
            page = self._build(key)
            # Pages are built on first show, so the condition's colour is applied
            # HERE, to every chart the page holds -- not once when the dialog opens.
            for chart in ([page] if isinstance(page, _QCChart) else []) + \
                    page.findChildren(_QCChart):
                chart.set_data_colour(getattr(self, "_data_colour", None))
            page.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self._stack.addWidget(page)
            self._pages[key] = page
        self._stack.setCurrentWidget(self._pages[key])
        self._after_show(key)
        self._title.setText(name)
        self._desc.setText(self._describe(key, desc))

    def current_key(self) -> str:
        return self._keys[self.nav.current()]


#: The outlier dialog's two views. Descriptions are templates: the unit and the
#: quantity are filled in from the chosen unit and what the flagging ran on.
_OUTLIER_TEXT = {
    "ordered": (
        "In order",
        "LOADING ORDER",
        ("Shows the {quantity} of every {unit}, in the order they were recorded. The "
         "grey band is where most {unit}s fall; orange {unit}s lie outside it."),
    ),
    "sorted": (
        "Sorted",
        "LOWEST TO HIGHEST",
        ("Shows the {quantity} of every {unit}, sorted from lowest to highest. The "
         "grey band is where most {unit}s fall; orange {unit}s lie outside it."),
    ),
    "bymovie": (
        "By movie",
        "ONE COLUMN PER MOVIE",
        ("Shows the {quantity} of every {unit}, {grouping}. The grey band is where most "
         "{unit}s fall; orange {unit}s lie outside it."),
    ),
}

_OUTLIER_UNITS = ("cell", "movie", "day")


class QCOutlierInspectDialog(_QCViewsDialog):
    """The per-unit outlier panel at full size, with the original's "outlier
    unit" choice (cell, movie or day) under the picture. The unit changes what a
    dot is and what the flagging runs over, nothing else on the page."""

    TEXT = _OUTLIER_TEXT
    SUBTITLE = "Three ways to look at the same outliers. Nothing here changes the analysis."

    def __init__(self, t, *, content=None, parent=None):
        super().__init__(t, content=content, parent=parent)
        self._unit = "cell"
        self._unit_row = QWidget()
        self._unit_row.setStyleSheet("background:transparent;")
        ur = QHBoxLayout(self._unit_row)
        ur.setContentsMargins(0, 0, 0, 0)
        ur.setSpacing(rem(0.75))
        ur.addWidget(label("Outlier unit", ty.ui("sm"), t.muted_foreground))
        self._seg = Segmented(t, [u.capitalize() for u in _OUTLIER_UNITS], current=0)
        self._seg.changed.connect(self._on_unit)
        ur.addWidget(self._seg)
        ur.addStretch(1)
        self._readout = label("", ty.ui("sm"), t.foreground)
        ur.addWidget(self._readout)
        self._controls.addWidget(self._unit_row)

    def _heading_text(self, condition: str) -> str:
        return f"Outliers \u00b7 {condition}" if condition else "Outliers"

    def set_context(self, condition: str, entry: dict) -> None:
        self._unit = "cell"
        self._seg.select(0)
        super().set_context(condition, entry)
        self._fit_readout()
        self._refresh_readout()

    def _units(self) -> dict:
        return (self._entry or {}).get("units") or {}

    def _current(self):
        """The chosen unit's plain block, or the ``{"error": ...}`` that replaced it."""
        got = self._units().get(self._unit)
        return got if got is not None else {"error": f"no {self._unit} summary"}

    def _apply(self, key: str, chart: UnitDotsChart) -> None:
        got = self._current()
        if "error" in got:
            chart.set_message(got["error"])
        else:
            chart.set_payload({"units": got, "ordered": key == "ordered",
                               "columns": key == "bymovie"})

    def _build(self, key: str) -> QWidget:
        chart = UnitDotsChart(self.t)
        self._apply(key, chart)
        return chart

    def _describe(self, key: str, desc: str) -> str:
        got = self._current()
        quantity = got["quantity"] if "error" not in got else "value"
        return outlier_description(desc, self._unit, quantity)

    def _on_unit(self, index: int) -> None:
        self._unit = _OUTLIER_UNITS[index]
        for key, page in self._pages.items():
            self._apply(key, page)
        self._desc.setText(self._describe(self.current_key(), self.TEXT[self.current_key()][2]))
        self._refresh_readout()

    def _refresh_readout(self) -> None:
        got = self._current()
        if "error" in got:
            self._readout.setText("not available")
            return
        self._readout.setText(outlier_readout(got["n_flagged"], got["n"], self._unit))

    def _fit_readout(self) -> None:
        """Fixed width for the readout's widest text, so it never moves the
        control beside it."""
        fm = QFontMetricsF(self._readout.font())
        widest = 0.0
        for unit in _OUTLIER_UNITS:
            got = self._units().get(unit)
            n = got["n"] if got is not None and "error" not in got else 0
            for digit in "0123456789":
                text = outlier_readout(n, n, unit)
                text = "".join(digit if c.isdigit() else c for c in text)
                widest = max(widest, fm.horizontalAdvance(text))
        self._readout.setFixedWidth(max(math.ceil(widest), fm.horizontalAdvance("not available")) + 8)


def qc_panel_payload(entry: dict, condition: str, key: str, colour=None):
    """``(kind, payload)`` for one Detail card of any condition's entry (the page's Export
    and "export every figure" alike), or None when that card has nothing to draw."""
    if key == "groups":
        block = entry.get("groups")
        if not isinstance(block, dict) or not block.get("rows"):
            return None
        from ...qc_view import groups_table

        return "table", groups_table(block, condition)
    blocks = {"length": entry.get("length"), "frame": entry.get("neighbour"),
              "outliers": (entry.get("units") or {}).get("cell"),
              "incell": None if entry.get("in_cell_share_error") else entry.get("in_cell")}
    block = blocks.get(key)
    kinds = QCConditionDetail.EXPORT_KINDS
    if key not in kinds or not isinstance(block, dict) or "error" in block:
        return None
    return kinds[key], {"block": block, "condition": condition, "colour": colour}


class QCConditionDetail(QWidget):
    """The Detail page: the five panels for exactly one condition, laid out
    as a two-column grid of panel cards with the in-cell panel across the
    full width. No heading of its own: the breadcrumb already names the
    section and the condition.

    ``set_payload`` takes
    one entry of `compute_qc_overview`'s ``per_condition`` dict — the same
    dict the Overview row was built from, so opening a Detail page never
    triggers a second compute. Which condition is shown is picked from the
    breadcrumb's dropdown (`WorkspaceWindow._on_dropdown_picked`).

    ONE PAGE, NOT SCROLLED: the grid's rows share whatever height the window
    gives them, and the charts are small enough at their minimum that the
    whole page fits an ordinary window. The N-per-group table, which grows
    with the number of movies, scrolls inside its own card instead.
    """

    #: Row weights: the middle row (a table and a bar per unit) needs the
    #: least height, so the charts above and below it get the room.
    _ROW_STRETCH = (5, 4, 4)
    #: card -> the figure its Export draws (`figure_export.KINDS`). The N-per-group card is
    #: a table, not a chart: its numbers are already in every panel's Source Data.
    EXPORT_KINDS = {"length": "qc_length", "frame": "qc_neighbour",
                    "outliers": "qc_outliers", "incell": "qc_incell", "groups": "table"}
    #: A card's Export, with its key in `EXPORT_KINDS`.
    export_requested = Signal(str)
    #: The length card's "Max length": set a cap on track length (the window owns the
    #: project, so it opens the dialog).
    cap_requested = Signal()

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, rem(0.5))
        lay.setSpacing(rem(0.75))

        self.length_chart = TrackLengthChart(t)
        self.frame_chart = NeighbourShareChart(t)
        self.group_table = GroupCountsTable(t)
        self.outlier_chart = UnitDotsChart(t)
        self.in_cell_chart = InCellShareChart(t)
        self._cards = {
            "length": _PanelCard(t, "Track-length distribution", self.length_chart),
            "frame": _PanelCard(t, "Localizations per frame", self.frame_chart),
            "groups": _PanelCard(t, "N per group", self.group_table, flush=True),
            "outliers": _PanelCard(t, "Per-cell outliers", self.outlier_chart),
            "incell": _PanelCard(t, "Tracks inside a cell outline, per movie",
                                 self.in_cell_chart),
        }
        self._entry: dict | None = None
        self._condition = ""
        self._cards["outliers"].enable_inspect("Inspect")
        self._cards["outliers"].inspect_requested.connect(self._open_outlier_inspect)
        # A CAP ON TRACK LENGTH (asked for after a 91-step track showed here): it changes
        # the table every analysis reads, so the window applies it.
        self._cards["length"].enable_inspect("Max length", glyph="filter")
        self._cards["length"].inspect_requested.connect(self.cap_requested.emit)
        for key in self.EXPORT_KINDS:
            self._cards[key].enable_export(
                "Export the table (PDF, SVG, PNG, TIFF), or add it to the Composer"
                if self.EXPORT_KINDS[key] == "table" else "Export figure (PDF, SVG, PNG, TIFF)")
            self._cards[key].export_requested.connect(
                lambda k=key: self.export_requested.emit(k))
        grid = QGridLayout()
        grid.setSpacing(rem(1.0))
        # Equal columns whatever the content: `Ignored` makes each half-width
        # card's width come from the grid's stretch alone, not from the widest
        # child's size hint (the table's columns would otherwise widen the
        # left column at the right one's expense).
        for key in ("length", "frame", "groups", "outliers"):
            self._cards[key].setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        grid.addWidget(self._cards["length"], 0, 0)
        grid.addWidget(self._cards["frame"], 0, 1)
        grid.addWidget(self._cards["groups"], 1, 0)
        grid.addWidget(self._cards["outliers"], 1, 1)
        grid.addWidget(self._cards["incell"], 2, 0, 1, 2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        for row_index, weight in enumerate(self._ROW_STRETCH):
            grid.setRowStretch(row_index, weight)
        lay.addLayout(grid, 1)

    # -- captions, from the spine's own numbers ------------------------------ #

    def _open_dialog(self, cls) -> None:
        window = self.window()
        dialog = cls(self.t, content=window.centralWidget(), parent=window)
        dialog.set_data_colour(getattr(self, "_colour", None))
        dialog.set_context(self._condition, self._entry)
        dialog.closed.connect(dialog.deleteLater)
        dialog.open()

    def export_payload(self, key: str):
        """``(kind, payload)`` for card ``key``'s Export -- the block that card draws, the
        condition and its colour -- or None when the card shows nothing to draw."""
        return qc_panel_payload(self._entry or {}, self._condition, key,
                                getattr(self, "_colour", None))

    def set_colour(self, colour: str | None) -> None:
        """Re-colour this condition's charts -- a change in the Conditions tool."""
        self._colour = colour
        for chart in (self.length_chart, self.frame_chart, self.outlier_chart,
                      self.in_cell_chart):
            chart.set_data_colour(colour)

    def _open_outlier_inspect(self) -> None:
        if not self._entry or "units" not in self._entry:
            return
        self._open_dialog(QCOutlierInspectDialog)

    def set_payload(self, condition: str, entry: dict,
                    colour: str | None = None) -> None:
        """``colour`` is the condition's colour from the Conditions tool: every
        chart here draws its data in it (flags stay orange)."""
        self.set_colour(colour)
        if "error" in entry:
            # Forget the previous condition's entry, or the Inspect icon would
            # open that condition's frames under this one's name.
            self._entry, self._condition = {}, condition
            for chart in (self.length_chart, self.frame_chart, self.group_table,
                         self.outlier_chart, self.in_cell_chart):
                chart.set_message(entry["error"])
            for card in self._cards.values():
                card.set_caption("")
            self._cards["outliers"].set_inspect_enabled(False)
            return
        self.length_chart.set_payload(entry["length"])
        self._cards["length"].set_caption(entry["length"]["caption"])
        self._entry, self._condition = entry, condition
        neighbour = entry.get("neighbour")
        if neighbour is not None:
            self.frame_chart.set_payload(neighbour)
            self._cards["frame"].set_caption(neighbour["caption"])
        else:
            self.frame_chart.set_message(entry.get("neighbour_share_error") or "Not measured.")
            self._cards["frame"].set_caption("")
        self.group_table.set_payload(entry["groups"])
        self._cards["groups"].set_caption("")
        units = entry["units"]["cell"]
        self.outlier_chart.set_payload({"units": units, "ordered": False})
        self._cards["outliers"].set_title(units["title"])
        self._cards["outliers"].set_caption(units["caption"])
        self._cards["outliers"].set_inspect_enabled(True)
        if entry.get("in_cell_share_error"):
            self.in_cell_chart.set_message(entry["in_cell_share_error"])
            self._cards["incell"].set_caption("")
        else:
            self.in_cell_chart.set_payload(entry["in_cell"])
            self._cards["incell"].set_caption(entry["in_cell"]["caption"])
