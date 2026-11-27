"""SuperPlots, painted native -- what remains of Group 1 of
`docs/method_browser_tab_groups.md` (the legacy `MethodBrowser`'s
"Fractions (SQD + cross-method)", "SuperPlot -- fraction", "Spot-On" and
"SuperPlot -- distribution" tabs) after "Fractions (SQD)" and "Spot-On"
were dropped, asked for directly -- the two survivors are both per-unit
SuperPlot scatters (per-cell HMM fraction, per-track appD), not
population-level fit summaries, and stay together as a matched pair on
the Apparent-D Distribution condition page (`appd_chart.ApparentDDetail`)
-- see the design doc for why the group existed in the first place.

**Two SEPARATE charts, each its own `Card`**, arranged side by side by
`FractionsChecksGrid` -- not one combined canvas: asked for directly, so
each chart reads as its own thing (title, border, shadow) the way every
other chart in this workspace does, rather than two panels sharing one
undivided widget. Each faithfully matches its own legacy picture
(`browser_controller.superplot_fraction`, `plots.draw_superplot_distribution`)
rather than an invented comparison.

**Nothing is computed here (Thin Qt).** The fits, the grouping by replicate
day, the units, the jitter and the table rows all come from
`smtracker.fractions_view.compute_fractions_checks` as plain lists, floats
and strings; these classes only map them to pixels. That module calls the
SAME analysis functions the legacy controllers call, so the numbers can
never disagree with the legacy tabs; only the picture differs.

**Per CONDITION, not per comparison** -- the compute takes one condition
because it is reached by drilling into ONE condition's Apparent-D page, not
from a Compute dialog covering every loaded condition. NOT CHEAP, so it runs
off the GUI thread on its own Run press inside the tab
(`window._run_fractions_checks`), and `ApparentDDetail` caches the result
per condition so switching conditions and back never re-fits anything.

**A condition whose HMM will not converge draws the SuperPlot's own
fallback (per-cell appD, its OWN units) rather than a fraction** --
`superplot_fraction`'s own behaviour, faithfully ported.
"""
from __future__ import annotations

import math

from qtpy.QtCore import (
    QEasingCurve,
    QParallelAnimationGroup,
    QPointF,
    QRect,
    QRectF,
    Qt,
    QVariantAnimation,
)
from qtpy.QtGui import QColor, QFontMetricsF, QPainter, QPen
from qtpy.QtWidgets import (
    QGraphicsOpacityEffect,
    QGridLayout,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)

from ... import figstyle
from ..theme import typography as ty
from ..widgets import tooltip as tip
from ..widgets.primitives import Card, label, mix, slide_ms
from ..widgets.table import TableBody
from .bubble_chart import (
    CARD_GAP,
    CARD_PAD,
    HANDLE_H,
    TABLE_MIN,
    TICK_PX,
    BubbleChart,
    TableHandle,
    _Drawer,
)
from .toolbar import ToolButton, rem

__all__ = ["FractionsChecksGrid"]

#: The fallback palette, used only when no condition colour was handed over (a
#: bare chart). With one, every replicate is a SHADE of the condition's colour
#: (`figstyle.day_shade`, the app's one rule), exactly as the Apparent-D overview,
#: the SQD replicate page and every export draw the days -- a separate day palette
#: put blue/red/green on a page whose colours already meant conditions.
_DAY_COLORS = figstyle.DAY_COLORS
_SINGLE_SESSION_COLOR = figstyle.SINGLE_SESSION_COLOR

_BAND_COLOR = "#a0aec0"

#: The chart's own minimum while its card's table drawer is CLOSED (the
#: ordinary case) -- comfortable room for axes/labels at a normal card
#: size. While the drawer is OPEN, `_ChartCard.set_table_open` lowers
#: this to `_CHART_SQUEEZE_MIN_H` instead -- see that constant's own
#: note for why: asked for directly, after both "grow the card" and
#: "scroll the page" were tried and rejected ("I want the dimension of
#: cards for charts/graphs to be fixed... the chart is squeezed").
_CHART_MIN_H = 220

#: The chart's minimum HEIGHT WHILE ITS TABLE IS OPEN -- small enough
#: that the table can claim its own share of a card whose TOTAL size
#: never changes, per the explicit ruling above: the card's footprint is
#: fixed, so opening a table can only ever come out of the chart's own
#: room, never out of the card growing or the page scrolling. `_paint`
#: guards every plot box with `max(1.0, ...)` already (`_plot_box`), so
#: this floor exists for legibility, not correctness -- small enough to
#: read as "squeezed", not so small the chart stops being a chart.
_CHART_SQUEEZE_MIN_H = 64


def _day_color(index, n_days: int = 1, base: str | None = None) -> str:
    """The colour of replicate column `index` (0-based) of `n_days`; `None` is the
    single-session group. With the condition's colour `base`, a shade of it
    (`figstyle.day_shade`) and the single session the colour itself."""
    if base:
        return base if index is None else figstyle.day_shade(base, index, max(1, n_days))
    if index is None:
        return _SINGLE_SESSION_COLOR
    return _DAY_COLORS[index % len(_DAY_COLORS)]


def _n_days(groups) -> int:
    return sum(1 for g in groups if g.get("colour_index") is not None)


class _MiniChart(QWidget):
    """Shared paint plumbing for the four charts below -- linear-grid
    axis drawing and the `set_payload`/`set_message` contract every
    other chart in this workspace already follows. Each subclass
    overrides `_paint(p, box, payload)` only.
    """

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("background:transparent;")
        self.setMinimumHeight(_CHART_MIN_H)
        self._tick_font = ty.ui("xs", px=TICK_PX, tabular=True)
        self._message: str | None = "Not yet computed."
        self._payload: dict | None = None
        #: The open condition's colour (the Conditions tool's), or None.
        self._colour: str | None = None

    def set_colour(self, colour: str | None) -> None:
        """The condition's colour: its replicates are drawn as its shades."""
        self._colour = colour
        self.update()

    def set_message(self, text: str) -> None:
        self._message = text
        self._payload = None
        self.update()

    def set_payload(self, payload: dict) -> None:
        self._payload = payload
        self._message = None
        self.update()

    #: Fixed, not `PAD_L`-derived: `BubbleChart._y_axis_label` translates
    #: to a FIXED absolute x (`rem(0.6)` from the WIDGET's own left edge,
    #: not from `box.left()`), so shrinking `box.left()` for a smaller
    #: chart brings the tick-number rect right on top of it -- measured,
    #: not guessed: that collision is exactly what a first pass reported
    #: as the label and the numbers overlapping. This value leaves the
    #: label (occupying roughly the first ~20px) and the widened tick
    #: rect below (`_TICK_RECT_W`, starting at `box.left() - 42`) a clear
    #: gap between them.
    _PLOT_LEFT = 66
    #: Wide enough for "0.005"-shaped tick text -- MEASURED with
    #: `QFontMetricsF.horizontalAdvance` at this exact tick font,
    #: "0.005"/"0.002" need 34.3px. A first pass used 36 (text width 32),
    #: which right-aligned text overflowing its own rect clips from the
    #: side away from the alignment -- `AlignRight` clips the LEFT, i.e.
    #: the leading "0", which is what read as ")005" on screen. Ticks on
    #: the log-scale distribution chart run wider than any other panel's
    #: (0.002 to 2), so it alone needs this width; the other three panels'
    #: "0"/"20"/"100" have room to spare.
    _TICK_RECT_W = 42

    #: Room for the TOP-MOST tick's own LABEL, not just a bare 4px gap --
    #: the top gridline sits exactly at `box.top()` (`_map_y` maps `hi`
    #: there) and its label is drawn VERTICALLY CENTRED on that line
    #: (`y - fm.height() / 2`), so half the label's own height needs to
    #: fit ABOVE `box.top()` before it collides with this widget's own
    #: top edge -- MEASURED at `self._tick_font`: height 14.5px, so ~7.3px
    #: is the bare minimum and the old 4px was already short of it.
    #: Reported directly, against a screenshot, once `_nice_ticks` made
    #: the top tick a clean round number sitting flush at the very top
    #: ("10000 and 12 ... interfere with the top border") -- the
    #: crowding was always there, just harder to notice against an
    #: un-round padding value like the old "10.4453". 16px clears the
    #: 7.3px minimum with visible air above the label besides.
    _TOP_MARGIN = 16

    #: 4px gap + room for the TILTED "Replicate N" labels below (see
    #: `_tilted_label` below) -- MEASURED with `QFontMetricsF` at
    #: `self._tick_font` (the same 12px muted font the y-axis numbers
    #: use, and `BubbleChart`'s own x-axis condition labels): "Replicate
    #: 10" (the longest realistic case) is 71.0px wide x 14.5px tall,
    #: which at a 45-degree tilt occupies 60.4px of VERTICAL space
    #: (`w*sin(45) + h*cos(45)`), not the 14.5px its own height alone
    #: would suggest -- the old 20px horizontal-label margin would clip a
    #: tilted label's descent by more than 3x.
    _BOTTOM_MARGIN = 4 + 64

    #: Degrees a category tick label is tilted -- the standard
    #: `rotation=45, ha='right'` convention (matplotlib's own terms): the
    #: label's RIGHT edge anchors at the tick, tilted so it reads from
    #: lower-left up to the tick, which is what keeps a longer run of
    #: category labels ("Replicate 1".."Replicate 10") from colliding the
    #: way a purely horizontal run would once there are more than a
    #: couple of them. Asked for directly.
    _LABEL_ANGLE_DEG = 45

    def _plot_box(self) -> QRectF:
        r = self.rect()
        return QRectF(r.left() + self._PLOT_LEFT, r.top() + self._TOP_MARGIN,
                     r.width() - self._PLOT_LEFT - 8,
                     max(1.0, r.height() - self._TOP_MARGIN - self._BOTTOM_MARGIN))

    @staticmethod
    def _nice_ticks(lo: float, hi: float, n: int = 5) -> tuple[float, float, list]:
        """A 'nice' (round-number) axis -- extends `[lo, hi]` OUTWARD to
        the nearest boundary of a step from `{1, 2, 5} x 10^k`, rather
        than a raw `n`-way even split of the (already padded) data
        range. The classic axis-labelling algorithm (Heckbert 1990,
        "Nice Numbers for Graph Labels"), matching what every
        general-purpose plotting library does by default.

        Reported directly, against a screenshot: dividing an arbitrary
        padded range into quarters gives ticks like `2.61131`,
        `7.83394` -- correct, but no reader expects to align a value
        against a tick carrying five significant figures. Returns
        `(axis_lo, axis_hi, ticks)`: the axis bounds themselves move
        out to the nearest nice tick (never in past the data), so a
        caller that passed `lo=0.0` for a bar chart's zero baseline
        still gets exactly `axis_lo=0.0` back (`floor(0/step)*step`).
        """
        if hi <= lo:
            hi = lo + 1.0
        raw_step = (hi - lo) / max(1, n - 1)
        magnitude = 10.0 ** math.floor(math.log10(raw_step))
        residual = raw_step / magnitude
        if residual < 1.5:
            step = magnitude
        elif residual < 3.0:
            step = 2 * magnitude
        elif residual < 7.0:
            step = 5 * magnitude
        else:
            step = 10 * magnitude
        axis_lo = math.floor(lo / step) * step
        axis_hi = math.ceil(hi / step) * step
        count = round((axis_hi - axis_lo) / step) + 1
        ticks = [axis_lo + i * step for i in range(count)]
        return axis_lo, axis_hi, ticks

    def _linear_grid(self, p: QPainter, box: QRectF, lo: float, hi: float,
                     ticks: list, y_label: str = "") -> None:
        dpr = max(1.0, self.devicePixelRatioF())
        pen = QPen(QColor(mix(self.t.border, self.t.background, 0.5)))
        pen.setWidthF(1.0 / dpr)
        p.setPen(pen)
        for v in ticks:
            y = self._map_y(box, v, lo, hi)
            p.drawLine(QPointF(box.left(), y), QPointF(box.right(), y))
        unit = 1.0 / dpr
        p.fillRect(QRectF(round(box.left() * dpr) / dpr, box.top(), unit,
                          box.height()), QColor(self.t.border))
        p.fillRect(QRectF(box.left(), round(box.bottom() * dpr) / dpr,
                          box.width(), unit), QColor(self.t.border))
        p.setFont(self._tick_font)
        p.setPen(QColor(self.t.muted_foreground))
        fm = QFontMetricsF(self._tick_font)
        for v in ticks:
            y = self._map_y(box, v, lo, hi)
            p.drawText(QRectF(box.left() - self._TICK_RECT_W, y - fm.height() / 2,
                              self._TICK_RECT_W - 4, fm.height()),
                       Qt.AlignRight | Qt.AlignVCenter, f"{v:g}")
        if y_label:
            BubbleChart._y_axis_label(p, box, y_label, self._tick_font,
                                      self.t.muted_foreground)

    @staticmethod
    def _map_y(box: QRectF, y: float, lo: float, hi: float) -> float:
        f = (y - lo) / (hi - lo) if hi > lo else 0.0
        return box.bottom() - f * box.height()

    def _tilted_label(self, p: QPainter, fm: QFontMetricsF, x: float, y: float,
                      text: str) -> None:
        """One category tick label, tilted `_LABEL_ANGLE_DEG` degrees --
        shared by both charts' own per-day x-axis (see `_BOTTOM_MARGIN`'s
        own comment for why the margin below the plot is sized for this).
        `(x, y)` is the ANCHOR the label's right edge ends at (the tick
        itself), matching `rotation=45, ha='right'` rather than a bare
        centered rotation -- Qt rotates its coordinate system CLOCKWISE
        for a positive angle, so `-_LABEL_ANGLE_DEG` here is what tilts
        the label counter-clockwise on screen (lower-left rising to the
        tick), the conventional direction for a tilted x-axis label.
        """
        text_w = fm.horizontalAdvance(text)
        p.save()
        p.translate(x, y)
        p.rotate(-self._LABEL_ANGLE_DEG)
        p.drawText(QRectF(-text_w - 2, -fm.height() / 2, text_w + 2, fm.height()),
                  Qt.AlignRight | Qt.AlignVCenter, text)
        p.restore()

    def _x_labels(self, p: QPainter, box: QRectF, xr: tuple, groups: list,
                  by_day: bool) -> None:
        """The per-replicate x labels (tilted), or one "cells" label."""
        if by_day:
            fm = QFontMetricsF(self._tick_font)
            for col, group in enumerate(groups, start=1):
                x = BubbleChart._map_x(box, col, xr)
                self._tilted_label(p, fm, x, box.bottom() + 4, group["label"])
        else:
            p.drawText(QRectF(box.left(), box.bottom() + 4, box.width(), 20),
                      Qt.AlignHCenter | Qt.AlignTop, "cells")

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        if self._payload is None:
            p.setFont(ty.ui("sm"))
            p.setPen(QColor(self.t.muted_foreground))
            p.drawText(self.rect(), Qt.AlignCenter | Qt.TextWordWrap,
                       self._message or "No data.")
        else:
            self._paint(p, self._plot_box(), self._payload)
        p.end()

    def _paint(self, p: QPainter, box: QRectF, pl: dict) -> None:
        raise NotImplementedError


class SuperPlotFractionChart(_MiniChart):
    """SuperPlot -- fraction: per-cell points by day (day-mean diamonds) +
    the SQD cell-weighted CI as a reference band -- `superplot_fraction`'s
    own picture. Falls back to per-cell appD (its own units, NOT forced
    onto a 0-100% axis) when the HMM will not converge, matching the
    legacy tab's own quantity swap."""

    def _paint(self, p: QPainter, box: QRectF, pl: dict) -> None:
        d = pl["fraction"]
        lo, hi, band, overall = d["lo"], d["hi"], d["band"], d["overall"]
        by_day, groups = d["by_day"], d["groups"]
        ticks = [lo + f * (hi - lo) for f in (0, 0.25, 0.5, 0.75, 1.0)]
        self._linear_grid(p, box, lo, hi, ticks, d["y_label"])

        n_cols = max(1, len(groups)) if by_day else 1
        xr = (0.4, n_cols + 0.6)
        if band is not None:
            y_lo = self._map_y(box, band[0], lo, hi)
            y_hi = self._map_y(box, band[1], lo, hi)
            p.fillRect(QRectF(box.left(), min(y_lo, y_hi), box.width(),
                              abs(y_lo - y_hi)), QColor(_BAND_COLOR).lighter(140))
            yo = self._map_y(box, overall, lo, hi)
            pen = QPen(QColor(self.t.foreground))
            pen.setWidthF(1.2)
            p.setPen(pen)
            p.drawLine(QPointF(box.left(), yo), QPointF(box.right(), yo))

        for col, group in enumerate(groups, start=1):
            colour = QColor(_day_color(group["colour_index"], _n_days(groups), self._colour))
            for v, jit in zip(group["values"], group["jitter"]):
                x = BubbleChart._map_x(box, (col + jit) if by_day else (1 + jit), xr)
                y = self._map_y(box, v, lo, hi)
                p.setPen(Qt.NoPen)
                c = QColor(colour)
                c.setAlpha(140)
                p.setBrush(c)
                p.drawEllipse(QPointF(x, y), 3.0, 3.0)
            if by_day and group["mean"] is not None:
                x = BubbleChart._map_x(box, col, xr)
                y = self._map_y(box, group["mean"], lo, hi)
                p.setPen(QPen(QColor(self.t.foreground), 1.0))
                p.setBrush(colour)
                diamond = [QPointF(x, y - 6), QPointF(x + 6, y),
                          QPointF(x, y + 6), QPointF(x - 6, y)]
                p.drawPolygon(diamond)

        #: `self._tick_font`/`muted_foreground` -- MATCHING `BubbleChart`'s
        #: own "condition labels (x axis)" convention (its `_tick_font`,
        #: same muted colour as its y-axis numbers), not the larger `"sm"`/
        #: full-foreground pairing this used before: at `"sm"` these
        #: labels rendered the SAME size as the card's own title, reading
        #: as competing with it rather than as axis furniture. Reported
        #: directly, against a screenshot.
        p.setFont(self._tick_font)
        p.setPen(QColor(self.t.muted_foreground))
        self._x_labels(p, box, xr, groups, by_day)


class SuperPlotDistributionChart(_MiniChart):
    """SuperPlot -- distribution: per-track apparent D, scattered by day,
    log scale -- `plots.draw_superplot_distribution`'s own picture (a
    pure strip, no density estimate, matching the legacy exactly)."""

    def _paint(self, p: QPainter, box: QRectF, pl: dict) -> None:
        d = pl["distribution"]
        if d["empty"]:
            p.setFont(ty.ui("xs"))
            p.setPen(QColor(self.t.muted_foreground))
            p.drawText(box, Qt.AlignCenter, "No tracks.")
            return
        lo, hi = d["log_lo"], d["log_hi"]
        by_day, groups = d["by_day"], d["groups"]
        yt = BubbleChart._log_ticks(10.0 ** lo, 10.0 ** hi)

        dpr = max(1.0, self.devicePixelRatioF())
        pen = QPen(QColor(mix(self.t.border, self.t.background, 0.5)))
        pen.setWidthF(1.0 / dpr)
        p.setPen(pen)
        for v in yt:
            y = box.bottom() - (math.log10(v) - lo) / (hi - lo) * box.height()
            p.drawLine(QPointF(box.left(), y), QPointF(box.right(), y))
        unit = 1.0 / dpr
        p.fillRect(QRectF(round(box.left() * dpr) / dpr, box.top(), unit,
                          box.height()), QColor(self.t.border))
        p.fillRect(QRectF(box.left(), round(box.bottom() * dpr) / dpr,
                          box.width(), unit), QColor(self.t.border))
        p.setFont(self._tick_font)
        p.setPen(QColor(self.t.muted_foreground))
        fm = QFontMetricsF(self._tick_font)
        for v in yt:
            y = box.bottom() - (math.log10(v) - lo) / (hi - lo) * box.height()
            p.drawText(QRectF(box.left() - self._TICK_RECT_W, y - fm.height() / 2,
                              self._TICK_RECT_W - 4, fm.height()),
                       Qt.AlignRight | Qt.AlignVCenter, f"{v:g}")
        BubbleChart._y_axis_label(p, box, "Apparent D (µm²/s, log)",
                                  self._tick_font, self.t.muted_foreground)
        n_cols = max(1, len(groups)) if by_day else 1
        xr = (0.4, n_cols + 0.6)
        for col, group in enumerate(groups, start=1):
            colour = QColor(_day_color(group["colour_index"], _n_days(groups), self._colour))
            for v, jit in zip(group["values"], group["jitter"]):
                x = BubbleChart._map_x(box, (col + jit) if by_day else (1 + jit), xr)
                y = box.bottom() - (math.log10(v) - lo) / (hi - lo) * box.height()
                p.setPen(Qt.NoPen)
                c = QColor(colour)
                c.setAlpha(110)
                p.setBrush(c)
                p.drawEllipse(QPointF(x, y), 2.4, 2.4)

        # `self._tick_font`/`muted_foreground` -- see the matching x-axis
        # block's own comment in `SuperPlotFractionChart._paint`.
        p.setFont(self._tick_font)
        p.setPen(QColor(self.t.muted_foreground))
        self._x_labels(p, box, xr, groups, by_day)


class _ChartCard(Card):
    """One chart, its own data table, and a show/hide toggle for it --
    `bubble_chart.BubbleCard`'s own drawer mechanism, REUSED rather than
    reimplemented (`_Drawer`/`TableHandle`, the exact same classes SQD's
    own "SQD populations" card uses), asked for directly after a first
    version built a smaller, always-visible table: two cards a click
    apart on one page is not a reason for a second visual language for
    "here is the data behind this chart" -- one drawer, wherever a chart
    has one.

    A header (title left, a table-toggle `ToolButton` right), the chart,
    then -- on request -- a drag-resizable `TableHandle` + `TableBody`
    drawer holding the exact rows the chart drew. CLOSED by default,
    matching `BubbleCard`'s own reasoning verbatim: most of the time the
    chart is the message.

    `table_key` names this card's own entry in the payload
    (`{"columns", "rows", "caption"}`, built by the view module);
    everything below `set_message`/`set_payload` is `BubbleCard`'s own
    `_apply`/`_drag`/`_slide`/`_inside`/`set_table_open` shape, ported
    rather than generalised into a shared base -- this file's own
    `_MiniChart` already draws differently from `BubbleChart`, so the
    drawer is the one piece worth sharing, not the chart underneath it.
    """

    #: Chart-dominant, matching `BubbleCard.DEFAULT_SHARE` -- the chart is
    #: what is being read; the table is consulted, not stared at.
    DEFAULT_SHARE = 0.4

    def __init__(self, t, title: str, chart: _MiniChart, table_key: str, parent=None):
        super().__init__(t, parent=parent)
        self.t = t
        self.chart = chart
        self._table_key = table_key
        self._table_h: int | None = None
        self.open = False

        # NO horizontal padding on the card's own layout -- `BubbleCard`'s
        # own reason: the header, the chart and the table carry their own,
        # so the divider under the header (`Card`'s cell rule) and the
        # drawer's own rule can run edge to edge.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, CARD_PAD, 0, CARD_PAD)
        outer.setSpacing(0)

        head = QWidget()
        head.setStyleSheet("background:transparent;")
        hl = QHBoxLayout(head)
        hl.setContentsMargins(CARD_PAD, 0, CARD_PAD, 0)
        hl.setSpacing(rem(0.5))
        hl.addWidget(label(title, ty.ui("sm", "medium"), t.foreground))
        hl.addStretch(1)
        self.toggle = ToolButton(t, "table", "table", size=rem(1.75), rest=t.card)
        self.toggle.clicked.connect(lambda _k: self.set_table_open(not self.open))
        self._tip = tip.attach(t, self.toggle, "Show data table",
                               side="bottom", top_level=True)
        hl.addWidget(self.toggle)
        outer.addWidget(head)
        outer.addSpacing(CARD_GAP)

        chart.setContentsMargins(CARD_PAD, 0, CARD_PAD, 0)
        outer.addWidget(chart, 1)

        self.handle = TableHandle(t)
        self.handle.dragged.connect(self._drag)
        self.handle.reset.connect(lambda: self.set_table_open(True, share=None))
        self.table = TableBody(t, [], [], "Not yet computed.", scrollable=True)
        self.drawer = _Drawer(t, self.handle, self.table, CARD_GAP)
        outer.addWidget(self.drawer)

        self._anim = QVariantAnimation(self)
        self._anim.setEasingCurve(QEasingCurve.Linear)
        self._anim.valueChanged.connect(
            lambda v: self.drawer.setFixedHeight(round(v)))
        # See `_on_slide_finished`'s own note -- fires at the end of
        # EVERY slide (open or close), and only acts on a close.
        self._anim.finished.connect(self._on_slide_finished)

    # -- message / data -------------------------------------------------- #
    def set_message(self, text: str) -> None:
        self.chart.set_message(text)
        self._rebuild_table([], [], text)

    def set_payload(self, payload: dict) -> None:
        self.chart.set_payload(payload)
        table = payload[self._table_key]
        self._rebuild_table(table["columns"], table["rows"], table["caption"])
        # A fresh Run's table gets a fresh default share rather than
        # keeping a pixel height sized for the LAST Run's row count --
        # `BubbleCard.set_data`'s own rule.
        self._table_h = None
        if self.open:
            self._apply()
            self.drawer.setFixedHeight(self.drawer.content_h())

    def _rebuild_table(self, columns: list, rows: list, caption: str) -> None:
        new_table = TableBody(self.t, columns, rows, caption, scrollable=True)
        self.drawer.replace_table(new_table)
        self.table = new_table

    # -- the table toggle, ported from `BubbleCard` verbatim ------------- #
    def set_table_open(self, on: bool, share: float | None = -1.0, *,
                       animate: bool = True) -> None:
        """Show or hide this chart's own data. `share` of `None` resets
        to the default split; the sentinel keeps whatever was dragged."""
        self.open = on
        self.toggle.set_active(on)
        self._tip.text = "Hide data table" if on else "Show data table"
        # OPENING lowers the chart's minimum NOW, before `_apply()` needs
        # the room it frees up -- Qt otherwise refuses to shrink the
        # chart below `_CHART_MIN_H` regardless of what `_apply()`
        # computes for the table. CLOSING must NOT restore it here: the
        # drawer is still at its OPEN height at this point (the slide
        # animation that shrinks it hasn't run yet), so snapping the
        # chart's minimum back up immediately asks the layout to fit
        # `_CHART_MIN_H` (220) *and* the still-open table into the same
        # fixed room in one frame -- reported directly as "the charts
        # flicker when I collapse a table". `_on_slide_finished` restores
        # it once the drawer has actually reached 0, so the chart grows
        # smoothly IN STEP with the drawer's own animated shrink instead
        # of jumping ahead of it.
        if on:
            self.chart.setMinimumHeight(_CHART_SQUEEZE_MIN_H)
            if share is None or self._table_h is None:
                self._table_h = None
            self._apply()
        self._slide(self.drawer.content_h() if on else 0, animate)

    def _on_slide_finished(self) -> None:
        """Restore the chart's normal minimum once a CLOSE has actually
        finished -- see `set_table_open`'s own note on why this cannot
        happen any earlier. A no-op after an OPEN (`self.open` is True by
        then, and `set_table_open` already lowered it up front)."""
        if not self.open:
            self.chart.setMinimumHeight(_CHART_MIN_H)

    def _slide(self, target: int, animate: bool) -> None:
        now = self.drawer.height()
        self._anim.stop()
        if not animate or target == now or not self.isVisible():
            self.drawer.setFixedHeight(target)
            self._on_slide_finished()
            return
        self._anim.setDuration(slide_ms(target - now))
        blocked = self._anim.blockSignals(True)
        self._anim.setStartValue(float(now))
        self._anim.setEndValue(float(target))
        self._anim.blockSignals(blocked)
        self._anim.start()

    def _inside(self) -> int:
        used = CARD_PAD * 2 + CARD_GAP * 2 + HANDLE_H
        used += self.layout().itemAt(0).widget().height()
        return max(0, self.contentsRect().height() - used)

    def _apply(self) -> None:
        # THE CARD'S OWN SIZE NEVER CHANGES -- ruled directly, after both
        # "let the table grow past `room` so the card/grid row grows to
        # fit it" and "wrap the page in a scroll area" were tried and
        # rejected: the card's footprint is fixed, so `want` is bounded
        # by `room` alone (never by `self.table.natural_height()` pushing
        # past it). Opening a table can only ever come out of the
        # CHART's own share -- see `_CHART_SQUEEZE_MIN_H`, which
        # `set_table_open` lowers the chart's own Qt-level minimum to so
        # the layout is actually free to give the table that share
        # rather than refusing to shrink the chart below `_CHART_MIN_H`
        # and clipping the overflow somewhere else.
        #
        # `_CHART_SQUEEZE_MIN_H` is STILL a hard Qt-level floor on the
        # chart, though -- capping `want` at `room` alone (not also
        # reserving that floor) let a DRAG push the table past
        # `room - _CHART_SQUEEZE_MIN_H`, and since Qt refuses to shrink
        # the chart below its own stated minimum, the card was forced
        # to grow to fit both anyway -- reported directly as "when I
        # resized the table for Kinetics, the height of the Kinetics
        # card also changes". Reserving the floor here is what makes the
        # drag handle actually STOP at a limit instead of just growing
        # the card past it.
        room = self._inside()
        want = (self._table_h if self._table_h is not None
               else round(room * self.DEFAULT_SHARE))
        want = min(want, self.table.natural_height())
        want = max(0, min(want, max(0, room - _CHART_SQUEEZE_MIN_H)))
        want = max(want, min(TABLE_MIN, max(0, room - _CHART_SQUEEZE_MIN_H)))
        self._table_h = want
        self.table.setFixedHeight(want)
        self.drawer.place()

    def _drag(self, dy: int) -> None:
        if not self.open:
            return
        self._table_h = (self._table_h or 0) - dy
        self._apply()
        self.drawer.setFixedHeight(self.drawer.content_h())

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.open and self._anim.state() != QVariantAnimation.Running:
            self._apply()
            # `_apply()` only resizes `self.table`; the DRAWER is its own
            # separate clipping container and does not follow along on its
            # own (`_drag`'s own resync, below, is the one place this was
            # already done). Without this, a card resized while its table
            # is open (any layout pass while open, not just a user drag --
            # e.g. the settling resizes a tab does when it first appears)
            # leaves the drawer's own height stale and SHORTER than the
            # table it now contains, clipping the table down to a sliver --
            # exactly the "handle bar + one cut-off row" screenshot this
            # fixes, on a real window whose layout settles in more than
            # one resize pass.
            self.drawer.setFixedHeight(self.drawer.content_h())


def _settle_curve() -> QEasingCurve:
    """The SAME bezier `window._fade_in` uses -- one visual language for
    "something just settled in"/"something just went away" across the
    whole app, not a second animation curve invented for this feature.

    ALWAYS forward, both open and close, every card -- a reversed-in-
    time variant for closing was tried (to mirror "hide is the reverse
    of open") and reverted: reversing this specific curve does not just
    delay its motion, it turns its ease-OUT shape (fast first ~20-30%,
    settling for the rest -- the "spring that recoils" feel) into
    ease-IN (barely moving, then a rushed snap at the very end) --
    reported directly as "lazy", not merely slower. Keeping every card
    on the SAME forward curve, both directions, is also what keeps a
    closing card and the neighbour claiming its space moving at matched
    relative rates throughout, with no extra bookkeeping needed to hold
    them in step. (If a reversed pacing is ever wanted again:
    `QAbstractAnimation.Backward` looks right in isolation but is
    silently forced back to `Forward` by `QAnimationGroup.start()` --
    verified directly; a custom `QEasingCurve.setCustomType` sidesteps
    that but SEGFAULTS once actually run, a Python callback stored
    inside a C++-copied Qt value type -- neither is a dead end worth
    retrying without a different mechanism.)"""
    curve = QEasingCurve(QEasingCurve.BezierSpline)
    curve.addCubicBezierSegment(QPointF(0.25, 0.1), QPointF(0.25, 1.0), QPointF(1.0, 1.0))
    return curve


def _geo_anim(start: QRect, end: QRect, ms: int) -> QVariantAnimation:
    anim = QVariantAnimation()
    anim.setStartValue(start)
    anim.setEndValue(end)
    anim.setDuration(ms)
    anim.setEasingCurve(_settle_curve())
    return anim


def _opacity_anim(start: float, end: float, ms: int, curve: QEasingCurve) -> QVariantAnimation:
    """`curve` is direction-specific -- see the two call sites below, not
    `_geo_anim`'s shared `_settle_curve()`: a card sliding smoothly into
    place looks right eased the same way in both directions, but the SAME
    curve on opacity spends a long, visually busy stretch of the
    animation only half-faded -- overlapping a NEIGHBOUR that is
    simultaneously moving into that same space, reported directly,
    against a screenshot, as the two chart's ink visibly blending
    together. `QEasingCurve.OutExpo` (closing) drops a card to near-
    transparent almost immediately, rather than lingering half-opaque for
    most of the animation and only becoming transparent right at the
    very end; `InExpo` (opening) keeps an incoming card close to fully
    transparent at first and only ramps up to opaque near the very end --
    asked for directly (and CORRECTED once, the two curves the other way
    round from a mis-worded first ask): what needs to happen fast is
    becoming TRANSPARENT, on both ends of a reflow, so the two charts
    sharing a moment of overlap read as one fading out and one fading in
    rather than a blended double-exposure of both."""
    anim = QVariantAnimation()
    anim.setStartValue(start)
    anim.setEndValue(end)
    anim.setDuration(ms)
    anim.setEasingCurve(curve)
    return anim


class _ReflowGrid(QWidget):
    """A `QGridLayout` of `_ChartCard`s whose per-card VISIBILITY can be
    toggled at runtime -- with an animated collapse and reflow, asked for
    directly (the Apparent-D detail page's own Visibility list): hiding a
    chart must not rebuild or re-render the page, only that card
    collapses while its siblings animate into the space it leaves.

    `placement_fn(visible: frozenset[str]) -> {key: (row, col, rowspan,
    colspan)}` is supplied per grid (`_fractions_placement`/
    `dynamics_chart._dynamics_placement`) -- the two arrangements this
    feature currently serves are genuinely different (2-across vs
    1-full-width-row-plus-2-across), so this stays a per-grid pure
    function rather than a shared lookup table. Row indices it returns
    are computed from the CURRENT visible set, not fixed per key -- that
    is what makes a full-width row's siblings slide UP to claim its space
    once it is hidden, rather than leaving a blank row.

    Geometry is animated via `QVariantAnimation` (this project's own house
    style for every custom animated value -- `_ChartCard._anim` above,
    `window._fade_in`), interpolating a `QRect` directly. VERIFIED
    directly (a throwaway spike script, not guessed) that a
    `QVariantAnimation` driving a `QGridLayout`-managed child's
    `setGeometry` frame by frame holds cleanly against the layout --
    `QLayout` only reasserts a child's geometry on `invalidate()`/a
    `LayoutRequest` event, not on every `setGeometry()` call, and even a
    child whose OWN `resizeEvent` fires `setFixedHeight` on something
    every frame (`_ChartCard`'s own drawer resync, when its table is
    open) does not fight the animation.
    """

    #: Started as an exact match to `window._fade_in`'s own duration
    #: (200), slowed to 320 from there -- a reflow moves TWO things at
    #: once (the collapsing card and its siblings claiming the space),
    #: so it read better a little slower than a plain one-widget fade.
    #: Nudged down to 250, then to 300, then back to 320, once the
    #: neighbour-pacing fix above landed -- asked for directly each time.
    ANIM_MS = 320

    #: How much of a CLOSE the departing card's chart gets to fade over,
    #: as a fraction of `ANIM_MS`. It buys the "blank canvas first" half
    #: of the close WITHOUT slowing the box down -- slowing the box was
    #: tried instead and starves the neighbour, which can only expand
    #: into room this card has actually vacated.
    #:
    #: It is pulled in two directions and the value is a JUDGED
    #: COMPROMISE between them, settled by watching all four:
    #:
    #:   | value | fade   | frames | chart illegible | box width then |
    #:   |-------|--------|--------|-----------------|----------------|
    #:   | 0.30  | 96 ms  |  ~5.8  |      32 ms      |      91%       |
    #:   | 0.45  | 144 ms |  ~8.6  |      48 ms      |      81%       |
    #:   | 0.65  | 208 ms |  ~12.5 |      80 ms      |      59%       |
    #:   | 0.80  | 256 ms |  ~15.4 |      96 ms      |      49%       |
    #:
    #: UPWARD, because it answers to the OPEN: open paints its chart in
    #: over ~11 frames, and at 0.30 close erased it in ~4 -- a 3x gap,
    #: reported as the chart "exiting too quickly compared to how it
    #: appears". DOWNWARD, because of the defect this replaced: the
    #: longer the fade, the further the box has shrunk while the chart
    #: is still legible, until it is once again a readable chart
    #: visibly shrinking away, and at the top of the range its
    #: still-tinted background washes over the neighbour beneath it.
    #:
    #: 0.45 was chosen, explicitly as "not fully satisfying, but good
    #: enough": ~8.6 frames is short of the open's ~11, so the exit is
    #: still the quicker of the two, but the chart is gone by 48 ms
    #: with the box at 81%, which keeps the close firmly "blank canvas,
    #: then zoom". Do not read it as an optimum -- it is where the
    #: tension was settled, and nothing else in the reflow moves if it
    #: is changed.
    _BLANK_FRACTION = 0.45

    def __init__(self, t, cards: dict, placement_fn, parent=None):
        super().__init__(parent)
        self.t = t
        self._cards = dict(cards)
        self._placement_fn = placement_fn
        self._visible: frozenset = frozenset(cards)
        self._group: QParallelAnimationGroup | None = None
        self.setStyleSheet("background:transparent;")

        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(CARD_GAP)
        self.grid.setVerticalSpacing(CARD_GAP)
        for key, (row, col, rspan, cspan) in placement_fn(self._visible).items():
            self.grid.addWidget(self._cards[key], row, col, rspan, cspan)

        self._empty_hint = label(
            "All charts are hidden. Use Visibility to show one.",
            ty.ui("sm"), t.muted_foreground)
        self._empty_hint.setParent(self)
        self._empty_hint.setAlignment(Qt.AlignCenter)
        self._empty_hint.setWordWrap(True)
        self._empty_hint.hide()

    def visibility_states(self) -> dict:
        return {k: (k in self._visible) for k in self._cards}

    def _start_state(self, card, key: str, old_geo: dict,
                     fallback_new_geo: QRect) -> tuple[QRect, float]:
        """Where THIS card's animation should start from. A card already
        carrying an opacity effect is mid-transition (a toggle just
        interrupted an earlier one) -- continue from its LIVE current
        geometry/opacity, never a synthetic start, or a rapid re-toggle
        would visibly jump. A genuinely at-rest card starts from its own
        remembered old geometry (still visible before this toggle) or,
        for a fresh reveal, a zero-size rect centred on where it is
        headed."""
        eff = card.graphicsEffect()
        if isinstance(eff, QGraphicsOpacityEffect):
            return card.geometry(), eff.opacity()
        if key in old_geo:
            return old_geo[key], 1.0
        c = fallback_new_geo.center()
        return QRect(c.x(), c.y(), 0, 0), 0.0

    def set_card_visible(self, key: str, on: bool) -> None:
        if (key in self._visible) == on:
            return
        # A prior toggle's reflow may still be running -- `stop()` leaves
        # every animated property at its CURRENT mid-flight value, never
        # snapping to start/end, so `_start_state` below (reading live
        # geometry/opacity off a card that still carries its effect) is
        # what makes resuming from here continuous rather than a jump --
        # for `key`, the card THIS call is about to touch.
        if self._group is not None:
            self._group.stop()
            self._group = None
            # Every OTHER card, though, has nothing left to resume it --
            # `.stop()` does not emit `finished`, so the interrupted
            # transaction's own `_cleanup` never ran for it, and it
            # would otherwise sit forever at whatever tiny, faint,
            # mid-collapse size and opacity it happened to be
            # interrupted at (reported directly, against a screenshot,
            # as a lingering "ghost" after hiding several cards in quick
            # succession). Settle it to its already-committed state
            # right now.
            for k, card in self._cards.items():
                if k != key and isinstance(card.graphicsEffect(), QGraphicsOpacityEffect):
                    card.setGraphicsEffect(None)
                    card.setVisible(k in self._visible)

        new_visible = (self._visible | {key}) if on else (self._visible - {key})
        old_geo = {k: c.geometry() for k, c in self._cards.items() if c.isVisible()}

        for k in new_visible - self._visible:
            self._cards[k].show()

        while self.grid.count():
            self.grid.takeAt(0)
        placement = self._placement_fn(new_visible)
        for k, (row, col, rspan, cspan) in placement.items():
            self.grid.addWidget(self._cards[k], row, col, rspan, cspan)
        # Forces Qt to compute the REAL target geometries with its own
        # row/column/stretch math right now, synchronously -- no
        # `processEvents()` between this and the `setGeometry` calls
        # below, or a stray frame could show cards at their freshly
        # activated (not yet start-snapped) positions before the
        # animation's own first frame.
        self.grid.activate()

        group = QParallelAnimationGroup(self)
        touched: list[tuple] = []

        for k, (row, col, rspan, cspan) in placement.items():
            card = self._cards[k]
            new_geo = card.geometry()
            start_geo, start_op = self._start_state(card, k, old_geo, new_geo)
            if start_geo == new_geo and start_op == 1.0 and k in old_geo:
                continue   # unaffected by this toggle -- nothing to animate
            card.setGeometry(start_geo)
            eff = card.graphicsEffect()
            if not isinstance(eff, QGraphicsOpacityEffect):
                eff = QGraphicsOpacityEffect(card)
                card.setGraphicsEffect(eff)
            eff.setOpacity(start_op)
            # Geometry ALWAYS plays `_settle_curve()` forward, both
            # directions, every card -- reverting an earlier attempt at
            # a reversed pacing for CLOSING (`_geo_anim_reversed`, still
            # below for the record). That fixed one problem (a neighbor
            # visibly expanding into space a still-large closing card
            # was occupying) by introducing another, reported directly:
            # `_settle_curve()` is an ease-OUT shape (fast first
            # ~20-30%, settling for the rest) -- the "spring that
            # recoils" feel open already has, on every card. Reversing
            # it in time for close does not just delay that motion, it
            # turns ease-out into ease-IN (barely moving, then a rushed
            # snap at the very end) -- "lazy", not slow-close-open. The
            # ORIGINAL overlap bug is still avoided here: with BOTH the
            # closing card and its neighbours back on the SAME forward
            # curve, they move at matched relative rates throughout,
            # which is exactly what already kept them in step on open.
            ga = _geo_anim(start_geo, new_geo, self.ANIM_MS)
            ga.valueChanged.connect(card.setGeometry)
            # Ghost trails otherwise -- a card wearing a
            # `QGraphicsOpacityEffect` (every touched card, below) is
            # composited through Qt's own offscreen-buffer path, whose
            # dirty-region tracking does not reliably cover the AREA a
            # `setGeometry` call just vacated the way a plain widget's
            # does; reported directly, against a screenshot, as
            # concentric leftover outlines at every past frame's
            # position. Forcing THIS container to repaint on every tick
            # (cheap -- Qt coalesces same-turn `update()` calls from all
            # the animations in this one group into a single repaint)
            # is what actually erases them.
            ga.valueChanged.connect(lambda _v: self.update())
            oa = _opacity_anim(start_op, 1.0, self.ANIM_MS,
                               QEasingCurve(QEasingCurve.InExpo))
            oa.valueChanged.connect(eff.setOpacity)
            group.addAnimation(ga)
            group.addAnimation(oa)
            touched.append((card, True))

        hidden_key = key if not on else None
        if hidden_key is not None:
            card = self._cards[hidden_key]
            start_geo, start_op = self._start_state(
                card, hidden_key, old_geo, old_geo.get(hidden_key, card.geometry()))
            collapsed = QRect(start_geo.center().x(), start_geo.center().y(), 0, 0)
            card.setGeometry(start_geo)
            eff = card.graphicsEffect()
            if not isinstance(eff, QGraphicsOpacityEffect):
                eff = QGraphicsOpacityEffect(card)
                card.setGraphicsEffect(eff)
            eff.setOpacity(start_op)
            # EVERY card in a reflow -- this one and every neighbour,
            # both directions -- is back on one shared `_settle_curve()`.
            # A card being hidden was briefly given that curve MIRRORED,
            # so its blank box came down late instead of springing; that
            # reads correctly on its own but starves the NEIGHBOUR, which
            # can only expand into room this card has actually vacated
            # (measured: a springing neighbour wants 233 px at 96 ms
            # against 26 px freed). Sharing one curve is what keeps the
            # two exactly in step, so the neighbour recoils as briskly
            # expanding as it does squeezing and neither ever overlaps
            # the other. What makes THIS card still read as "blank first,
            # then zoom" is `_BLANK_FRACTION` below, not a slower box.
            ga = _geo_anim(start_geo, collapsed, self.ANIM_MS)
            ga.valueChanged.connect(card.setGeometry)
            ga.valueChanged.connect(lambda _v: self.update())  # see note above
            # The fade runs over a FRACTION of the reflow, not all of
            # it -- this is what carries "the chart is gone before the
            # card is". Over the full duration the chart was still ~50%
            # opaque at 32 ms and ~25% at 64 ms, so the eye caught a
            # legible chart shrinking, which open has no counterpart to.
            # Compressed, it is ~10% by 32 ms and ~1% by 64 ms, i.e.
            # blank while the box has given up under a third of its
            # width, leaving the rest of the reflow to a blank box
            # zooming out -- open's "zoom while blank, chart only in the
            # last few frames", the other way round.
            oa = _opacity_anim(start_op, 0.0,
                               max(1, round(self.ANIM_MS * self._BLANK_FRACTION)),
                               QEasingCurve(QEasingCurve.OutExpo))
            oa.valueChanged.connect(eff.setOpacity)
            group.addAnimation(ga)
            group.addAnimation(oa)
            touched.append((card, False))

        def _cleanup():
            for c, stayed_visible in touched:
                c.setGraphicsEffect(None)
                if not stayed_visible:
                    c.hide()
            self.grid.activate()
            self._group = None
            if not new_visible:
                # See the note below on why this half is deferred here --
                # the OTHER half (hiding the hint the instant a card is
                # revealed) already happened synchronously above.
                self._empty_hint.setVisible(True)
            # One final repaint -- a faint residual outline of the LAST
            # collapsed card's near-zero final frame otherwise survives
            # even once fully settled: the per-tick `self.update()` the
            # geometry animations trigger races the tick that actually
            # reaches size zero, and nothing after that point asks for a
            # repaint again.
            self.update()

        group.finished.connect(_cleanup)
        self._visible = new_visible
        # Showing the hint is deferred to `_cleanup` (once the LAST
        # card's own collapse has actually finished) -- reported
        # directly, against a screenshot: showing it immediately, here,
        # sat "All charts hidden" as static text directly on top of a
        # chart still visibly fading away underneath it for the whole
        # animation. Hiding it is NOT deferred: the moment a card is
        # revealed the hint is already wrong, so it disappears at once
        # rather than lingering over the incoming card's own fade-in.
        if new_visible:
            self._empty_hint.setVisible(False)
        self._empty_hint.setGeometry(self.rect())
        # Kept alive on `self` -- `window._fade_in`'s own precedent: Qt's
        # C++ parenting (the group owns its child animations) keeps THOSE
        # alive, but a collected Python wrapper for the GROUP itself,
        # mid-flight, is a real, separate risk.
        self._group = group
        group.start()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._group is not None:
            # A real window resize mid-reflow would otherwise fight the
            # animation (whose start/end rects were computed for the OLD
            # size) -- settle immediately to the already-decided target
            # VISIBILITY (`self._visible`, updated before the animation
            # even started) rather than try to reconcile in-flight
            # progress with a new size.
            self._group.stop()
            self._group = None
            for k, card in self._cards.items():
                card.setGraphicsEffect(None)
                card.setVisible(k in self._visible)
            self.grid.activate()
            # `set_card_visible`'s own deferred-hint logic never ran (no
            # `_cleanup` here) -- settle it to match `self._visible` too.
            self._empty_hint.setVisible(not self._visible)
        self._empty_hint.setGeometry(self.rect())


#: Internal keys for `_ReflowGrid` -- NOT the display titles (those live
#: only in `toolbar.FIGURES`/`EyeRow`), so a future title wording change
#: never touches grid internals. See `_ReflowGrid`'s own docstring.
def _fractions_placement(visible: frozenset) -> dict:
    present = [k for k in ("fraction", "distribution") if k in visible]
    if len(present) == 2:
        return {present[0]: (0, 0, 1, 1), present[1]: (0, 1, 1, 1)}
    if len(present) == 1:
        return {present[0]: (0, 0, 1, 2)}   # the sole survivor claims the row
    return {}


class FractionsChecksGrid(QWidget):
    """The SuperPlots tab's WHOLE content: two `_ChartCard`s side by
    side, one payload from `compute_fractions_checks` fanned out to
    both. Laid out through `_ReflowGrid` so the Apparent-D detail page's
    own Visibility list can hide/show either chart with an animated
    collapse + reflow, see that class's own docstring."""

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.setStyleSheet("background:transparent;")
        self.superplot_fraction_chart = SuperPlotFractionChart(t)
        self.superplot_distribution_chart = SuperPlotDistributionChart(t)
        self._fraction_card = _ChartCard(
            t, "Fraction (per-cell)", self.superplot_fraction_chart,
            "fraction_table")
        self._distribution_card = _ChartCard(
            t, "Distribution (per-track appD)", self.superplot_distribution_chart,
            "distribution_table")
        self._cards = (self._fraction_card, self._distribution_card)

        self._reflow = _ReflowGrid(
            t, {"fraction": self._fraction_card,
               "distribution": self._distribution_card},
            _fractions_placement, parent=self)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._reflow)

    def set_card_visible(self, key: str, on: bool) -> None:
        self._reflow.set_card_visible(key, on)

    def visibility_states(self) -> dict:
        return self._reflow.visibility_states()

    def set_message(self, text: str) -> None:
        for card in self._cards:
            card.set_message(text)

    def set_payload(self, payload: dict) -> None:
        for card in self._cards:
            card.set_payload(payload)

    def set_colour(self, colour: str | None) -> None:
        """The open condition's colour, for both SuperPlots' replicate shades."""
        self.superplot_fraction_chart.set_colour(colour)
        self.superplot_distribution_chart.set_colour(colour)
