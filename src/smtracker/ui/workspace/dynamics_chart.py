"""Dynamics, painted native -- Group 2 of `docs/method_browser_tab_groups.md`
("State-count & dynamics evidence"): the legacy `MethodBrowser`'s
"Kinetics", "State count" and "Clustering" tabs, as a SECOND tab on the
Apparent-D condition drill-in page (`appd_chart.ApparentDDetail`), beside
the existing "SuperPlots" tab (`fractions_chart.py`).

**Why these three sit in ONE tab, not three**: the design doc's own
reasoning, and it is load-bearing rather than decorative. Kinetics is
what tells you whether State count's own mode count can be trusted at
all -- a condition that barely switches is fine for saSPT; one that
switches at this project's own measured rate (6-8/s) is not (CLAUDE.md:
"saSPT resolves two because of switching, not because there are two").
**Nothing in `smtracker.saspt_adapter` computes that warning** -- it is
prose only. `dynamics_view` composes it, from the Kinetics rate in the
SAME payload, because that cross-reference is the one thing a
per-tab-in-isolation port could never do.

**Reuses `fractions_chart.py`'s own `_MiniChart`/`_ChartCard` directly,
not duplicated** -- `_ChartCard` already gives every card its show/hide
data-table drawer (`bubble_chart._Drawer`/`TableHandle`, SQD's own
mechanism), and `_MiniChart` already gives every chart its axis-drawing
plumbing and `set_message`/`set_payload` contract. The SAME cross-module
reuse this whole UI stack already leans on (`_DimOverlay`, `_LineTabs`,
`BubbleChart`'s own `@staticmethod`s).

**Nothing is computed here (Thin Qt).** The three analyses, the composed
switching caveat, the axis bounds and the table rows come from
`smtracker.dynamics_view.compute_dynamics_evidence` as plain lists, floats
and strings, and each tooltip's words come from that module's
`kinetics_hover_rows`, `saspt_hover` and `clustering_hover`; these classes
only map them to pixels. Each of the three analyses fails independently
there, so one card's analysis being unavailable never blanks the other two.
"""
from __future__ import annotations

import math
from itertools import pairwise
from typing import NamedTuple

from qtpy.QtCore import QPoint, QPointF, QRectF, Qt
from qtpy.QtGui import QColor, QFontMetricsF, QPainter, QPen
from qtpy.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from ...dynamics_view import clustering_hover, kinetics_hover_rows, saspt_hover
from ..theme import typography as ty
from ..widgets.primitives import draw_shadow, label, mix, shadow_room
from .bubble_chart import BubbleChart
from .fractions_chart import _BAND_COLOR, _ChartCard, _MiniChart, _ReflowGrid
from .toolbar import rem, text_px

__all__ = ["DynamicsEvidenceGrid"]

_KIN_ON_COLOR = "#2b6cb0"
_KIN_OFF_COLOR = "#c05621"
_WARN_COLOR = "#c05621"
_SASPT_LINE_COLOR = "#2b6cb0"
_SASPT_MODE_COLOR = "#c05621"
_CLUSTER_OBSERVED_COLOR = "#2b6cb0"

#: Room reserved at the TOP of the plot box for an in-chart caveat --
#: MEASURED against `ty.ui("xs")` word-wrapping at this tab's card width
#: (narrower than SuperPlots' two-column cards, three here): enough for
#: ~3 wrapped lines. Shared by `KineticsChart`'s degenerate-chain flag and
#: `SaSPTChart`'s composed switching-blind caveat -- see the module
#: docstring for why both are drawn IN the chart, not only in the table
#: caption (the drawer starts closed).
_CAVEAT_STRIP_H = 44


def _draw_top_caveat(p: QPainter, box: QRectF, text: str, font, colour: str) -> QRectF:
    """Word-wrapped caveat text in a reserved strip at the top of `box`;
    returns the remaining box with that strip removed."""
    p.setFont(font)
    p.setPen(QColor(colour))
    p.drawText(QRectF(box.left(), box.top(), box.width(), _CAVEAT_STRIP_H),
              Qt.AlignLeft | Qt.AlignTop | Qt.TextWordWrap, text)
    return QRectF(box.left(), box.top() + _CAVEAT_STRIP_H, box.width(),
                 max(1.0, box.height() - _CAVEAT_STRIP_H))


class _ChartTooltip(QWidget):
    """shadcn's `ChartTooltipContent`, in its `indicator="line"` style: a
    heading, then any number of `name -> value` rows, with ONE vertical
    bar running the full height of the content on the left edge.

    **A SIBLING of `bubble_chart._BubbleTooltip`, not a refactor of it.**
    That class is this same shadcn component pinned to exactly ONE shape
    -- a Condition heading and its two fixed rows, `ROW_NAMES = ("D",
    "Fraction")` -- because one bubble always answers the same two
    questions. These three charts do not: a bar carries a rate and its
    CI, a spectrum point carries a D and an occupancy, a `k` column
    carries observed/chance/excess AT ONCE. Generalising the bubble's
    class to a variable row count would put this file's needs inside a
    chart that has no use for them, on the hover path of the busiest
    chart in the app -- so the row-building is what differs here, and
    everything that decides how the thing LOOKS (padding, radius, the
    popover fill, the bar geometry, the self-drawn shadow) is ported from
    it deliberately unchanged, so the two read as one component.

    **NOT `widgets.tooltip.Tooltip`**, for the reason that class's own
    docstring gives: it is ONE LINE of plain text sized from
    `horizontalAdvance`, shared app-wide for button hover-help, and
    multi-row content does not belong in it.
    """

    #: `w-1 rounded-[2px]` -- shadcn's own `indicator="line"` bar, and the
    #: hollow/dashed variant is deliberately the WIDER of the two: matched
    #: exactly, a thin stroke's ink still reads lighter than a filled
    #: block of the same footprint. Both ported from `_BubbleTooltip`.
    BAR_W = rem(0.25)
    BAR_OUTLINE_W = rem(0.375)
    BAR_LEFT = rem(0.5)
    BAR_GAP = rem(0.5)
    #: ``indicator="square"``: one SQUARE swatch, sharp corners, beside the heading.
    #: Asked for on the Spatial Map, where the colour IS a bin's: a 4 px rounded
    #: strip was hard to read as a colour, and a square echoes the grid cell hovered.
    SWATCH = rem(0.875)

    def __init__(self, t, parent=None, *, compact: bool = False,
                 indicator: str = "line"):
        super().__init__(parent)
        # "none": a sentence-only card with no mark at all -- the Metadata receipt's
        # explanations, which describe a setting, not a coloured datum.
        if indicator not in ("line", "square", "none"):
            raise ValueError(
                f"indicator must be 'line', 'square' or 'none', got {indicator!r}")
        self.t = t
        self._indicator = indicator
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._room = shadow_room(t, t.shadow_md)
        self._bar_colour = QColor(Qt.transparent)
        self._bar_line_style = Qt.SolidLine
        lay = QVBoxLayout(self)
        pad_x, pad_y = rem(0.625), rem(0.5)
        #: See `paintEvent` -- the bar hugs the content the way shadcn's
        #: own `self-stretch` row does, so it is tied to the SAME margin
        #: the heading and last row are laid out with.
        self._pad_y = pad_y
        mark_w = (self.SWATCH if indicator == "square"
                  else max(self.BAR_W, self.BAR_OUTLINE_W))
        left = (self._room + pad_x if indicator == "none"
                else self._room + self.BAR_LEFT + mark_w + self.BAR_GAP)
        lay.setContentsMargins(left, self._room + pad_y,
                               self._room + pad_x, self._room + pad_y)
        lay.setSpacing(rem(0.3))
        # `font-medium` (500), not `semibold` -- shadcn's own label weight.
        # `compact`: the heading at the rows' own 12 px, for a tooltip that is ONLY a heading
        # (the QC charts' sentences), where the 14 px label reads as too large.
        size = ("xs", text_px(0.75)) if compact else ("sm", text_px(0.875))
        self._heading = label("", ty.ui(size[0], "medium", px=size[1]), t.foreground)
        lay.addWidget(self._heading)
        self._lay = lay
        #: Row widgets are POOLED, never rebuilt: a hover moving along a
        #: 60-point spectrum re-populates this tooltip on every mouse
        #: move, and tearing down/re-adding child widgets at that rate is
        #: what makes a hover feel sticky. Rows past the current content
        #: are hidden, and a hidden widget is already excluded from the
        #: layout's own size, so `adjustSize` shrinks correctly.
        self._rows: list[tuple[QWidget, object, object]] = []
        self.hide()

    def _row(self, i: int) -> tuple:
        while len(self._rows) <= i:
            holder = QWidget(self)
            holder.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            holder.setStyleSheet("background:transparent;")
            row = QHBoxLayout(holder)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(rem(0.5))
            name = label("", ty.ui("xs", px=text_px(0.75)), self.t.muted_foreground)
            value = label("", ty.ui("xs", px=text_px(0.75), tabular=True), self.t.foreground)
            row.addWidget(name)
            row.addStretch(1)
            row.addWidget(value)
            self._lay.addWidget(holder)
            self._rows.append((holder, name, value))
        return self._rows[i]

    def set_content(self, *, heading: str, rows, colour, dashed: bool = False) -> None:
        self._heading.setText(heading)
        self._bar_colour = QColor(colour)
        self._bar_line_style = Qt.DashLine if dashed else Qt.SolidLine
        rows = list(rows)
        for i, (name, value) in enumerate(rows):
            holder, name_label, value_label = self._row(i)
            name_label.setText(name)
            value_label.setText(value)
            holder.setVisible(True)
        for holder, _, _ in self._rows[len(rows):]:
            holder.setVisible(False)
        # EVERY LAYOUT RUN NOW, rows first, before sizing: a row's layout recomputes on a
        # POSTED event, so `adjustSize` read the LAST content's width -- a longer tooltip after
        # a shorter one was cut off ("3 days, low to hig", measured on the HMM strip, 21 Oct
        # 2026), and while hidden it never caught up
        for holder, _, _ in self._rows[:len(rows)]:
            holder.layout().invalidate()
            holder.layout().activate()
            holder.updateGeometry()
        self._lay.invalidate()
        self._lay.activate()
        self.adjustSize()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        box = QRectF(self.rect().adjusted(
            self._room, self._room, -self._room, -self._room))
        radius = self.t.radius_md
        # Drawn DIRECTLY, not through `Surface`/`ShadowHost` -- a widget
        # reparented onto the window at show time has no `ShadowHost`
        # ancestor walking `findChildren` for it. Same call `menus.Panel`
        # and `_BubbleTooltip` both make.
        draw_shadow(p, box, radius, self.t, self.t.shadow_md)
        p.setPen(QPen(QColor(self.t.border), 1.0))
        p.setBrush(QColor(self.t.popover))
        p.drawRoundedRect(box.adjusted(0.5, 0.5, -0.5, -0.5), radius, radius)
        if self._indicator == "none":
            p.end()
            return
        if self._indicator == "square":
            # Sharp corners, centred on the HEADING line it describes; the 1 px border
            # sits OUTSIDE the fill, so a pale or grey (masked) bin still shows its edge
            # against the popover while every pixel of the square is the bin's colour.
            head = self._heading.geometry()
            s = self.SWATCH
            # Whole pixels: a square's edge is crisp only on the pixel grid.
            swatch = QRectF(round(box.left() + self.BAR_LEFT),
                            round(head.center().y() + 0.5 - s / 2.0), s, s)
            p.setRenderHint(QPainter.Antialiasing, False)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(self.t.border))
            p.drawRect(swatch.adjusted(-1, -1, 1, 1))
            p.setBrush(self._bar_colour)
            p.drawRect(swatch)
            p.end()
            return
        bar_w = self.BAR_W if self._bar_line_style == Qt.SolidLine else self.BAR_OUTLINE_W
        bar = QRectF(box.left() + self.BAR_LEFT, box.top() + self._pad_y,
                     bar_w, box.height() - 2 * self._pad_y)
        bar_radius = bar_w / 2
        if self._bar_line_style == Qt.SolidLine:
            p.setPen(Qt.NoPen)
            p.setBrush(self._bar_colour)
            p.drawRoundedRect(bar, bar_radius, bar_radius)
        else:
            pen = QPen(self._bar_colour)
            pen.setWidthF(1.4)
            pen.setStyle(self._bar_line_style)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            inset = pen.widthF() / 2
            p.drawRoundedRect(bar.adjusted(inset, inset, -inset, -inset),
                              bar_radius, bar_radius)
        p.end()


class _Hover(NamedTuple):
    """One hovered thing, as the tooltip needs it. `key` is what the
    chart's own `_paint` compares against to draw its cursor, so it must
    identify the item rather than its position -- a geometry that moves
    (a resize, a reflow) must not read as a different hover."""

    key: tuple
    heading: str
    rows: tuple
    colour: str
    dashed: bool
    #: Widget coordinates the tooltip points at -- its bottom edge is
    #: placed just above this, flipping below when there is no room.
    anchor: QPointF


class _HoverChart(_MiniChart):
    """A `_MiniChart` that answers "what is this?" on hover.

    **Hit regions are RECORDED BY `_paint`, never recomputed.** Every
    subclass here derives its axis bounds inside its own `_paint` (from
    the payload, via `_nice_ticks`, after `_draw_top_caveat` has already
    taken its slice off the top), so a second copy of that arithmetic
    for hit-testing would be a second source of truth for where a bar
    IS -- and this file's own history is a string of geometry fixes
    measured against screenshots, every one of which would have had to
    be made twice. `_paint` stores what it drew; `_hit_at` reads it
    back. A hit region cannot drift from the thing it belongs to,
    because it IS the thing that was drawn.
    """

    def __init__(self, t, parent=None):
        super().__init__(t, parent)
        self.setMouseTracking(True)
        self._hover: _Hover | None = None
        self._tip = _ChartTooltip(t, self)

    # -- subclass hook ------------------------------------------------------ #
    def _hit_at(self, pos: QPointF) -> _Hover | None:
        """The item under `pos`, from whatever `_paint` last recorded."""
        return None

    def hover_key(self) -> tuple | None:
        """What `_paint` compares against to draw its cursor."""
        return self._hover.key if self._hover else None

    # -- plumbing ----------------------------------------------------------- #
    def mouseMoveEvent(self, e):
        self._set_hover(self._hit_at(e.position()) if self._payload else None)
        super().mouseMoveEvent(e)

    def leaveEvent(self, e):
        self._set_hover(None)
        super().leaveEvent(e)

    # NO `hideEvent` OVERRIDE, and that is a MEASURED decision rather than
    # an omission. A card hidden by the Visibility tool would ideally take
    # its tooltip with it (the tooltip is parented onto the WINDOW, so it
    # does not follow this widget out of view on its own), and hooking
    # `hideEvent` to do that is the obvious way. It SEGFAULTS: `hideEvent`
    # is also reached while Qt tears the widget tree down, so it runs
    # Python against half-destroyed C++ objects. Counted over 8 runs of an
    # otherwise identical script: this class with every other override but
    # no `hideEvent` crashed 0/8, with `hideEvent` alone 5/8, and HEAD
    # before any of this 0/8. A `try/except RuntimeError` does NOT help --
    # the fault has `<no Python frame>` at all, so there is nothing to
    # catch. Same family as the `QThread`/`QCloseEvent` teardown crashes
    # CLAUDE.md already records for this binding.
    #
    # `leaveEvent` covers the real case anyway: Qt delivers Leave when the
    # widget under the pointer is hidden, and reaching the Visibility
    # toggle means moving the pointer off the chart first regardless.

    def set_payload(self, payload: dict) -> None:
        self._set_hover(None)
        super().set_payload(payload)

    def set_message(self, text: str) -> None:
        self._set_hover(None)
        super().set_message(text)

    def _set_hover(self, hover: _Hover | None) -> None:
        same = (hover.key if hover else None) == self.hover_key()
        self._hover = hover
        try:
            if hover is None:
                self._tip.hide()
                self.unsetCursor()
            else:
                self._tip.set_content(heading=hover.heading, rows=hover.rows,
                                      colour=hover.colour, dashed=hover.dashed)
                window = self.window()
                if self._tip.parentWidget() is not window:
                    self._tip.setParent(window)
                self._place(hover.anchor)
                self._tip.raise_()
                self._tip.show()
        except RuntimeError:
            # The tooltip is REPARENTED ONTO THE WINDOW (it has to be, to
            # escape this chart's own bounds), so the window owns it and
            # tears it down FIRST -- and Qt then hides this widget on the
            # way out, which lands right here, on a `self._tip` whose C++
            # object is already gone. A destroyed child raises
            # `RuntimeError` through its Python wrapper rather than
            # returning something falsy, which is the same shape
            # `primitives.ShadowHost.event` documents and catches for its
            # own cached children. VERIFIED, not defensive: without this,
            # tearing down a window with a tooltip still up segfaults
            # (exit 139), and closing the window first exits 0.
            return
        if not same:
            # Only the CURSOR needs a repaint, and only when the hovered
            # item actually changed -- a repaint per mouse move across the
            # same bar would redraw the whole chart for nothing.
            self.update()

    def _place(self, anchor: QPointF) -> None:
        """Centred above `anchor`, clamped to the window, flipped below
        when there is no room -- the same "clamp, don't escape" rule
        `Tooltip.show_for` and `BubbleChart._position_tip` both follow."""
        window = self.window()
        w, h = self._tip.width(), self._tip.height()
        pos = self.mapTo(window, QPoint(round(anchor.x()), round(anchor.y())))
        gap = rem(0.375)
        x = max(4, min(pos.x() - w // 2, max(4, window.width() - w - 4)))
        y = pos.y() - gap - h
        if y < 4:
            y = pos.y() + gap
        self._tip.move(x, y)

    # -- shared cursor painting --------------------------------------------- #
    def _cursor_band(self, p: QPainter, band: QRectF) -> None:
        """shadcn's own bar-chart cursor: a muted block behind the hovered
        category, drawn BEFORE the bar so the bar keeps its own colour."""
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(mix(self.t.muted, self.t.background, 0.45)))
        p.drawRect(band)

    def _cursor_line(self, p: QPainter, x: float, box: QRectF) -> None:
        """shadcn's line/area-chart cursor: a single vertical rule at the
        hovered x, dashed and muted so it reads as a guide rather than as
        another series."""
        pen = QPen(QColor(self.t.muted_foreground))
        pen.setWidthF(1.0)
        pen.setDashPattern([3, 3])
        p.setPen(pen)
        p.drawLine(QPointF(x, box.top()), QPointF(x, box.bottom()))


class KineticsChart(_HoverChart):
    """Kinetics (HMM): two bars, on/off switching rate (1/s), with
    asymmetric bootstrap-percentile whiskers -- `browser_controller.
    kinetics`'s own picture. LINEAR y-axis, not log: a rarely switching
    condition's off-rate CI can span three orders of magnitude,
    and a visibly huge linear whisker IS the
    honest signal a log axis would compress away -- matches the
    legacy's own unset `ylim` too."""

    #: Widened from `_MiniChart`'s own 42px/66px -- MEASURED with
    #: `QFontMetricsF` at the tick font: an autoscaled rate axis's own
    #: quartile ticks aren't bounded to a fixed range like the SuperPlots
    #: charts' 0-100%/0.002-2, so a value like "10.695" needs 42.1px
    #: alone, already past the old rect's usable 38px (`_TICK_RECT_W - 4`)
    #: -- reported directly, against a screenshot: `AlignRight` clips the
    #: LEFT, which is what read as "L0.695" on screen (the exact same
    #: mechanism `fractions_chart.py`'s own `_TICK_RECT_W` comment
    #: already names). `_PLOT_LEFT` widened by the SAME 10px so the
    #: y-axis label keeps its original ~24px gap from the tick rect
    #: rather than the two colliding in turn.
    _TICK_RECT_W = 52
    _PLOT_LEFT = 76

    def __init__(self, t, parent=None):
        super().__init__(t, parent)
        #: `(band, name, value, ci, colour)` per bar, recorded by `_paint`
        #: -- see `_HoverChart`'s own docstring on why hit regions are
        #: stored rather than recomputed.
        self._bands: list[tuple] = []

    def _hit_at(self, pos: QPointF):
        for i, (band, name, val, ci, colour) in enumerate(self._bands):
            if band.contains(pos):
                # The rate and its CI (or a stated absence): `kinetics_hover_rows`.
                return _Hover(key=("bar", i), heading=name,
                              rows=kinetics_hover_rows(val, ci),
                              colour=colour, dashed=False,
                              anchor=QPointF(band.center().x(), band.top()))
        return None

    def _paint(self, p: QPainter, box: QRectF, pl: dict) -> None:
        self._bands = []
        d = pl["kinetics"]
        if not d["ok"]:
            p.setFont(ty.ui("xs"))
            p.setPen(QColor(self.t.muted_foreground))
            p.drawText(box, Qt.AlignCenter | Qt.TextWordWrap,
                      d["error"] or "HMM did not converge for this condition.")
            return

        if d["caveat"]:
            box = _draw_top_caveat(p, box, d["caveat"],
                                   ty.ui("xs", "medium"), _WARN_COLOR)

        lo, hi, ticks = self._nice_ticks(0.0, d["axis_top"])
        self._linear_grid(p, box, lo, hi, ticks, "Rate (1/s)")

        xr = (-0.6, 1.6)
        colours = {"on": _KIN_ON_COLOR, "off": _KIN_OFF_COLOR}
        bars = [(b["name"], b["value"], b["ci"], colours[b["colour_key"]])
                for b in d["bars"]]
        for i, (name, val, ci, colour) in enumerate(bars):
            x0 = BubbleChart._map_x(box, i - 0.3, xr)
            x1 = BubbleChart._map_x(box, i + 0.3, xr)
            y0 = self._map_y(box, 0.0, lo, hi)
            y1 = self._map_y(box, val, lo, hi)
            # The hit region is the whole CATEGORY column, not the bar's
            # own painted rect: a low bar is a few px tall, and a hover
            # that only answers while the pointer is inside those few px
            # is one most people never find. Half a bar-width of slack
            # either side keeps the two columns from ever meeting.
            band = QRectF(x0 - (x1 - x0) * 0.25, box.top(),
                          (x1 - x0) * 1.5, box.height())
            self._bands.append((band, name, val, ci, colour))
            if self.hover_key() == ("bar", i):
                self._cursor_band(p, band)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(colour))
            p.drawRoundedRect(QRectF(x0, min(y0, y1), x1 - x0, abs(y0 - y1)), 2, 2)
            if ci is not None:
                xm = BubbleChart._map_x(box, i, xr)
                ylo, yhi = self._map_y(box, ci[0], lo, hi), self._map_y(box, ci[1], lo, hi)
                pen = QPen(QColor(self.t.foreground))
                pen.setWidthF(1.2)
                p.setPen(pen)
                p.drawLine(QPointF(xm, ylo), QPointF(xm, yhi))
                cap = (x1 - x0) * 0.3
                p.drawLine(QPointF(xm - cap, ylo), QPointF(xm + cap, ylo))
                p.drawLine(QPointF(xm - cap, yhi), QPointF(xm + cap, yhi))
            p.setFont(ty.ui("sm"))
            p.setPen(QColor(self.t.foreground))
            p.drawText(QRectF(x0 - 10, box.bottom() + 4, (x1 - x0) + 20, 20),
                      Qt.AlignHCenter | Qt.AlignTop, name)


class SaSPTChart(_HoverChart):
    """State count (saSPT): posterior-occupancy spectrum, `diffusion_grid`
    (log-x) vs `occupations` (linear-y) -- `browser_controller.
    state_count`'s own picture. Hand-rolled log-x/linear-y axis (mirrors
    `SuperPlotDistributionChart`'s own manual log-axis block in
    `fractions_chart.py` -- `_MiniChart._linear_grid` is y-only and
    assumes a linear x). The COMPOSED switching-blind caveat
    (`payload["saspt_caveat"]`) is drawn IN the chart -- see the module
    docstring for why: the drawer starts closed, and this warning is the
    entire reason these tabs sit together.
    """

    def __init__(self, t, parent=None):
        super().__init__(t, parent)
        #: `(x, y, D, occupancy %)` per spectrum point, plus the plot box
        #: they were mapped into -- recorded by `_paint`, read by
        #: `_hit_at`. See `_HoverChart`'s own docstring.
        self._points: list[tuple] = []
        self._box = QRectF()

    def _hit_at(self, pos: QPointF):
        if not self._points or not self._box.contains(pos):
            return None
        # NEAREST IN X, not a hit area per point: the grid is ~60 points
        # over a log axis, so at the dense end they are a couple of px
        # apart and per-point regions would leave gaps the pointer falls
        # through. Reading off the x the pointer is at is also what a
        # spectrum is FOR -- shadcn's own line-chart tooltip does the
        # same, which is why the cursor is a vertical rule.
        i = min(range(len(self._points)),
                key=lambda j: abs(self._points[j][0] - pos.x()))
        x, y, d, occ = self._points[i]
        heading, rows = saspt_hover(d, occ)
        return _Hover(
            key=("point", i), heading=heading, rows=rows,
            colour=_SASPT_LINE_COLOR, dashed=False, anchor=QPointF(x, y))

    def _paint(self, p: QPainter, box: QRectF, pl: dict) -> None:
        self._points = []
        self._box = QRectF()
        d = pl["saspt"]
        if not d["ok"]:
            p.setFont(ty.ui("xs"))
            p.setPen(QColor(self.t.muted_foreground))
            p.drawText(box, Qt.AlignCenter | Qt.TextWordWrap,
                      d["error"] or "saSPT unavailable for this condition.")
            return

        if d["caveat"]:
            box = _draw_top_caveat(p, box, d["caveat"], ty.ui("xs"), self.t.muted_foreground)

        grid, occ = d["grid"], d["occupancy"]
        if len(grid) < 2:
            p.setFont(ty.ui("xs"))
            p.setPen(QColor(self.t.muted_foreground))
            p.drawText(box, Qt.AlignCenter, "No spectrum.")
            return

        lo_x, hi_x = math.log10(min(grid)), math.log10(max(grid))
        hi_y = d["y_top"]
        lo_y = 0.0

        dpr = max(1.0, self.devicePixelRatioF())
        pen = QPen(QColor(mix(self.t.border, self.t.background, 0.5)))
        pen.setWidthF(1.0 / dpr)
        p.setPen(pen)
        y_ticks = [lo_y + f * (hi_y - lo_y) for f in (0, 0.25, 0.5, 0.75, 1.0)]
        for v in y_ticks:
            y = self._map_y(box, v, lo_y, hi_y)
            p.drawLine(QPointF(box.left(), y), QPointF(box.right(), y))
        unit = 1.0 / dpr
        p.fillRect(QRectF(round(box.left() * dpr) / dpr, box.top(), unit,
                          box.height()), QColor(self.t.border))
        p.fillRect(QRectF(box.left(), round(box.bottom() * dpr) / dpr,
                          box.width(), unit), QColor(self.t.border))

        p.setFont(self._tick_font)
        p.setPen(QColor(self.t.muted_foreground))
        fm = QFontMetricsF(self._tick_font)
        for v in y_ticks:
            y = self._map_y(box, v, lo_y, hi_y)
            p.drawText(QRectF(box.left() - self._TICK_RECT_W, y - fm.height() / 2,
                              self._TICK_RECT_W - 4, fm.height()),
                       Qt.AlignRight | Qt.AlignVCenter, f"{v:.2f}")
        BubbleChart._y_axis_label(p, box, "Occupancy", self._tick_font,
                                  self.t.muted_foreground)

        xt = BubbleChart._log_ticks(10.0 ** lo_x, 10.0 ** hi_x)
        for v in xt:
            x = box.left() + (math.log10(v) - lo_x) / (hi_x - lo_x) * box.width()
            p.drawText(QRectF(x - 20, box.bottom() + 4, 40, 20),
                      Qt.AlignHCenter | Qt.AlignTop, f"{v:g}")

        def map_x(d: float) -> float:
            return box.left() + (math.log10(d) - lo_x) / (hi_x - lo_x) * box.width()

        pts = [QPointF(map_x(g), self._map_y(box, o, lo_y, hi_y))
              for g, o in zip(grid, occ)]
        self._box = QRectF(box)
        self._points = [(pt.x(), pt.y(), g, o)
                        for pt, g, o in zip(pts, grid, d["occupancy_pct"])]
        # Before the spectrum, so the guide sits UNDER the line it is
        # pointing at rather than across it.
        key = self.hover_key()
        if key is not None and key[0] == "point":
            self._cursor_line(p, self._points[key[1]][0], box)
        pen = QPen(QColor(_SASPT_LINE_COLOR))
        pen.setWidthF(1.6)
        p.setPen(pen)
        for a, b in pairwise(pts):
            p.drawLine(a, b)
        if key is not None and key[0] == "point":
            hx, hy, _, _ = self._points[key[1]]
            # A ring, not a filled dot -- `_SASPT_MODE_COLOR` dots below
            # mark the RESOLVED MODES, which is a claim about the data;
            # the hover marker must not be mistakable for one.
            p.setBrush(QColor(self.t.popover))
            p.setPen(QPen(QColor(_SASPT_LINE_COLOR), 1.6))
            p.drawEllipse(QPointF(hx, hy), 3.0, 3.0)

        p.setPen(Qt.NoPen)
        p.setBrush(QColor(_SASPT_MODE_COLOR))
        for mode in d["modes"]:
            x = map_x(mode["D"])
            y = self._map_y(box, mode["occupancy"], lo_y, hi_y)
            p.drawEllipse(QPointF(x, y), 3.5, 3.5)


class ClusteringChart(_HoverChart):
    """Clustering: k-means Calinski-Harabasz score at k=2,3 -- observed
    (solid) vs the free-Brownian null's chance value (dashed) -- with a
    shaded `± EXCEEDS_NULL_SD * chance_sd` band, which the legacy chart
    never drew (CLAUDE.md's "HONEST ERROR BARS" rule): the band's own
    top edge is then literally the decision boundary
    `KSelection.exceeds_null` tests against, not just a stated verdict.
    """

    #: Same widening as `KineticsChart`'s own, same reason: Calinski-
    #: Harabasz scores autoscale to whatever this condition's real numbers
    #: are (measured against a screenshot: "372.975" clipped to "'2.975"
    #: at `_MiniChart`'s default 42px/66px).
    _TICK_RECT_W = 52
    _PLOT_LEFT = 76

    def __init__(self, t, parent=None):
        super().__init__(t, parent)
        #: `(x, k, observed, chance, chance_sd, excess, y_observed)` per k
        #: column, plus the plot box -- recorded by `_paint`, read by
        #: `_hit_at`. See `_HoverChart`'s own docstring.
        self._columns: list[tuple] = []
        self._box = QRectF()

    def _hit_at(self, pos: QPointF):
        if not self._columns or not self._box.contains(pos):
            return None
        i = min(range(len(self._columns)),
                key=lambda j: abs(self._columns[j][0] - pos.x()))
        x, k, observed, chance, chance_sd, excess, y_obs = self._columns[i]
        # BOTH series at this k, which is the comparison the chart exists
        # to make -- shadcn's own tooltip lists every series at the
        # hovered x rather than the one nearest the pointer. The heading
        # carries k, matching the axis label directly beneath it.
        heading, rows = clustering_hover(k, observed, chance, chance_sd, excess)
        return _Hover(
            key=("k", i), heading=heading, rows=rows,
            colour=_CLUSTER_OBSERVED_COLOR, dashed=False,
            anchor=QPointF(x, y_obs))

    def _paint(self, p: QPainter, box: QRectF, pl: dict) -> None:
        self._columns = []
        self._box = QRectF()
        d = pl["clustering"]
        if not d["ok"]:
            p.setFont(ty.ui("xs"))
            p.setPen(QColor(self.t.muted_foreground))
            p.drawText(box, Qt.AlignCenter | Qt.TextWordWrap,
                      d["error"] or "Clustering unavailable.")
            return

        rows = d["rows"]
        band_lo, band_hi = d["band_lo"], d["band_hi"]
        lo, hi, ticks = self._nice_ticks(d["lo"], d["hi"])
        self._linear_grid(p, box, lo, hi, ticks, "Calinski-Harabasz")

        ks = [r["k"] for r in rows]
        xr = (min(ks) - 0.6, max(ks) + 0.6)

        self._box = QRectF(box)
        self._columns = [
            (BubbleChart._map_x(box, r["k"], xr), r["k"], r["observed"], r["chance"],
             r["chance_sd"], r["excess"], self._map_y(box, r["observed"], lo, hi))
            for r in rows]
        # Before the band and both lines, so the guide sits behind every
        # mark it is pointing at.
        key = self.hover_key()
        if key is not None and key[0] == "k":
            self._cursor_line(p, self._columns[key[1]][0], box)

        band_poly = ([QPointF(BubbleChart._map_x(box, k, xr), self._map_y(box, v, lo, hi))
                      for k, v in zip(ks, band_hi)]
                     + [QPointF(BubbleChart._map_x(box, k, xr), self._map_y(box, v, lo, hi))
                        for k, v in zip(ks[::-1], band_lo[::-1])])
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(_BAND_COLOR).lighter(140))
        p.drawPolygon(band_poly)

        def line(values, colour, dashed):
            pts = [QPointF(BubbleChart._map_x(box, k, xr), self._map_y(box, v, lo, hi))
                  for k, v in zip(ks, values)]
            pen = QPen(QColor(colour))
            pen.setWidthF(1.6)
            if dashed:
                pen.setDashPattern([4, 3])
            p.setPen(pen)
            for a, b in pairwise(pts):
                p.drawLine(a, b)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(colour))
            for pt in pts:
                p.drawEllipse(pt, 3.0, 3.0)

        line([r["chance"] for r in rows], _BAND_COLOR, True)
        line([r["observed"] for r in rows], _CLUSTER_OBSERVED_COLOR, False)

        p.setFont(self._tick_font)
        p.setPen(QColor(self.t.muted_foreground))
        for k in ks:
            x = BubbleChart._map_x(box, k, xr)
            p.drawText(QRectF(x - 20, box.bottom() + 4, 40, 20),
                      Qt.AlignHCenter | Qt.AlignTop, f"k={int(k)}")


#: Internal keys for `_ReflowGrid` -- NOT the display titles (those live
#: only in `toolbar.FIGURES`/`EyeRow`). Row indices are computed from the
#: CURRENT visible set: hiding "saspt" lets kinetics/clustering slide up
#: into row 0 rather than leaving a blank row above them; hiding one of
#: kinetics/clustering lets the other claim the full row. See
#: `fractions_chart._ReflowGrid`'s own docstring.
def _dynamics_placement(visible: frozenset) -> dict:
    out: dict = {}
    row = 0
    if "saspt" in visible:
        out["saspt"] = (row, 0, 1, 2)
        row += 1
    bottom = [k for k in ("kinetics", "clustering") if k in visible]
    if len(bottom) == 2:
        out["kinetics"] = (row, 0, 1, 1)
        out["clustering"] = (row, 1, 1, 1)
    elif len(bottom) == 1:
        out[bottom[0]] = (row, 0, 1, 2)
    return out


class DynamicsEvidenceGrid(QWidget):
    """The Dynamics tab's WHOLE content: three `_ChartCard`s, two rows --
    State count (saSPT) full-width on its own row, Kinetics and
    Clustering side by side beneath it -- one payload from
    `compute_dynamics_evidence` fanned out to all three. Each card's own
    chart/table-rows function independently checks its own `*_ok` flag,
    so this class stays branch-free -- mirrors `fractions_chart.
    FractionsChecksGrid`'s existing shape. Asked for directly: State
    count's own spectrum reads better with the room a full row gives it
    than sharing a third of the width three-across did.

    Laid out through `_ReflowGrid` (`fractions_chart.py`, reused
    cross-module like `_ChartCard`/`_MiniChart` already are) so the
    Apparent-D detail page's own Visibility list can hide/show any of the
    three with an animated collapse + reflow."""

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.setStyleSheet("background:transparent;")
        self.kinetics_chart = KineticsChart(t)
        self.saspt_chart = SaSPTChart(t)
        self.clustering_chart = ClusteringChart(t)
        self._kinetics_card = _ChartCard(
            t, "Kinetics (HMM)", self.kinetics_chart, "kinetics_table")
        self._saspt_card = _ChartCard(
            t, "State count (saSPT)", self.saspt_chart, "saspt_table")
        self._clustering_card = _ChartCard(
            t, "Clustering", self.clustering_chart, "clustering_table")
        self._cards = (self._kinetics_card, self._saspt_card, self._clustering_card)

        self._reflow = _ReflowGrid(
            t, {"kinetics": self._kinetics_card,
               "saspt": self._saspt_card,
               "clustering": self._clustering_card},
            _dynamics_placement, parent=self)
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
