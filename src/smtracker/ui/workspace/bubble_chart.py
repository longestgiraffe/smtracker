"""The SQD bubble chart, painted native -- `prototype/shadcn/charts.py`'s own
convention (`QPainter`, theme tokens, 12 px muted tick labels, `stroke-border/50`
grid, the y axis KEPT because these are measurements) ported here rather than
matplotlib, which is what `window.py` used the first time and was corrected.

**Colour is by CONDITION, not by population -- a deliberate departure from the
matplotlib figure this chart replaces.** `compare_plots.draw_bubble` colours by
STATE (`figstyle.STATE_COLORS`), the right choice for a MANUSCRIPT figure where
every condition sits in its own panel or column and the reader is comparing
STATES across them. This is a GUI comparison view instead, where several
conditions sit on one x axis, and this app already has a colour identity for
"which condition is this" -- the workspace's own Conditions tool
(`toolbar.ConditionsBody`), the same swatches its popover shows and the palette
picker (Okabe-Ito by default) drives. Colouring bubbles by state here would
give every condition's column the SAME two colours and tell two conditions
apart only by their x position and label; colouring by condition instead makes
"which dots belong together" legible at a glance, and keeps this chart
speaking the one colour language the rest of the app already uses for
conditions. `WorkspaceWindow` resolves the mapping (`ConditionsBody.colour_for`)
on the GUI thread and hands it to `set_data`/`set_colours` -- never resolved
inside `compute_bubble_view` itself, which runs off the GUI thread and must not
touch a live widget.

**Area carries the fraction.** `plots.draw_bubble_panel`'s own rule: the drawn
RADIUS is the fraction's SQUARE ROOT, never the fraction itself, so two states
are never made to look twice as far apart as they really are.

Two shapes render here, matching `comparison_controller.sqd_view`'s own two
draw paths -- `compute_bubble_view` below builds both from the SAME
`assemble_view` / `_overlay_table` calls `sqd_view` itself makes, so the
NUMBERS can never disagree with the browser's matplotlib figure; only the
picture drawn from them differs:
  - the classic, single-arrangement case: one column of populations per
    condition (`comparison.sqd_bubbles(...)["columns"]`).
  - the overlay case (the dialog's "Overlay: Both" default): several
    `(fit_mode, n_pop)` SERIES drawn side by side within each condition's
    column, told apart by a REDUNDANT shade + outline pair
    (`plots.BUBBLE_SERIES_STYLES`) -- this project has already measured that
    colour alone does not survive greyscale reproduction.

Deliberately NOT ported: the Berglund D-axis correction toggle (the Compute
dialog exposes no control for it, so `spec.d_correction` is always False and
drawing the toggle's machinery would be dead code), and a CLICK-to-inspect a
single replicate (`compare_plots.draw_bubble`'s own feature, no controlling
UI yet in this window).

**HOVER, not click, is where a single bubble's detail lives** -- asked for
directly, and scoped narrowly on purpose after weighing a richer "several
hover modes" idea (single point / whole column / same population across
conditions) against it. A passive gesture doing three different things at
once (a tooltip, the chart dimming, a table row jumping) was judged more
confusing than useful, and CLICK is already reserved for a condition's own
Detailed View, so it cannot carry a second meaning here. What ships is ONE
thing on hover: `BubbleChart.hovered` reports the `BubblePoint` under the
cursor (`_hit_test`, sharing `_point_geometry` with painting so a hit can
never disagree with what is drawn) and drives `_BubbleTooltip` -- shadcn's
`ChartTooltipContent` "line" indicator style, a Condition label over two
rows (D, Fraction) with ONE vertical bar running their full height,
condition-coloured and series-dashed (see `_show_tip`). A bespoke class,
not `widgets.tooltip.Tooltip`: see its own docstring for why retrofitting
multi-row, bar-carrying content into code shared by every button's
hover-help was the wrong place to put this. `BubbleCard`
listens to the same signal to scroll the matching table row into view and
highlight it -- ONLY when the table is open and that row is not currently
filtered out; it never reaches across an active filter or forces the
drawer open to show one. Richer comparisons (a whole column, one
population across every condition) are left for later, and land as
something the user asks for explicitly rather than as more hover output.
Hovering a bubble ALSO turns the cursor into a pointing hand
(`_set_hovered`) -- asked for once the click above existed, since nothing
on the chart itself said a bubble was clickable until this did.

`BubbleCard` wraps the chart in a real `widgets.primitives.Card`, the shape
every figure in `prototype/shadcn/figures.py`'s `FigureCard` uses: a header
(title left, a table toggle right), then the chart, then -- ported in full,
not deferred a second time -- the SAME table-belongs-to-the-figure drawer
`FigureCard` carries: a draggable handle and a scrollable
table of every drawn population, opened by the header's toggle, sliding out
from the chart's own bottom edge rather than growing the card (`figures.py`'s
own module docstring gives the reasons: a taller card would ragged two
side-by-side figures, and a chart degrades badly below its floor while a
table just gets a scrollbar, so the CHART is the side that gives up room).
`TableHandle`/`_Drawer` are ported near-verbatim from `figures.py`;
`widgets.table.TableBody` is the registry-faithful `<Table>` itself, ported
from `prototype/shadcn/toolbar.py`'s `_TableBody` into `widgets/` because
nothing in this UI stack had a generic table primitive yet -- the prototype's
copy lived in `toolbar.py` only because that was where its one caller sat.

**One difference from the prototype's table, forced by the data being live
rather than static mock rows:** `FigureCard` builds its `TableBody` ONCE, from
a fixed `TABLES` registry entry, and never touches it again. This chart's
numbers change on every Run, so `BubbleCard.set_data` REBUILDS the table from
the SAME `BubblePlotData` the chart itself paints -- one source of truth for
both, via the module-level `table_rows` -- and swaps it into the drawer
(`_Drawer.replace_table`), which the prototype's `_Drawer` never needed.

**Typed ranges (30 Oct 2026):** the header's ruler button (`BubbleCard.axes_button`) opens the shared Axes popover
(`ui.widgets.axes_popover`, rules in `smtracker.axes_view`): the bubbles' D (y, log), the Fraction x D plot's
fraction (x) and D (y). One range per view, kept by the card through a new Run until Auto (`typed_axes`,
`set_typed_axes`, `axes_state`); the chart draws `apply_range` of its automatic ends and records `drawn_y`.
"""
from __future__ import annotations

from ...axes_view import apply_range, axis_rows, check
from ...axes_view import empty as axes_empty
from ..widgets.axes_popover import AxesPopover

import math

from qtpy.QtCore import (
    QEasingCurve,
    QPoint,
    QPointF,
    QRectF,
    Qt,
    QTimer,
    QVariantAnimation,
    Signal,
)
from qtpy.QtGui import QColor, QCursor, QFontMetricsF, QPainter, QPen
from qtpy.QtWidgets import (
    QHBoxLayout,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ... import chart_text, figstyle
from ...bubble_view import (
    crossed_payload,
    BubblePlotData,
    BubblePoint,
    condition_facets,
    copy_rows,
    d_text,
    fraction_text,
    point_key,
    replicate_footer,
    replicate_table_rows,
    shows_pairs,
    PAIR_COLUMN_NOTE,
    sample_lines,
    sample_marks,
    series_facets,
    shared_d_caption,
    table_rows,
)
from ..theme import hcl
from ..theme import typography as ty
from ..widgets import tooltip as tip
from ..widgets.primitives import Card, draw_shadow, icon, label, mix, shadow_room, slide_ms
from ..widgets.table import FacetedFilter, TableBody
from .toolbar import ToolButton, rem, text_px
from ..widgets.primitives import discard

__all__ = ["BubbleCard", "BubbleChart", "ReplicateChart", "ReplicateDetail", "wrap_label",
           "column_label_width", "label_line_count", "draw_column_labels"]

#: `text-xs` on `ChartContainer` -- see `prototype/shadcn/charts.py`'s own note
#: on where this comes from.
TICK_PX = text_px(0.75)
PAD_L, PAD_R, PAD_T, PAD_B = rem(3.75), rem(1.25), rem(0.75), rem(2.0)
#: SIGNIFICANCE BRACKETS over the plot (14 Oct 2026, `significance.chart_view`), in pixels: the
#: gap between the plot and the lowest tier, a leg's length, the p's lift off its line, and how
#: far a leg stands in from its column's centre (two brackets meeting at one column keep their
#: own legs). The export draws the same shape in points (`figure_export._bracket_geometry`).
BRACKET_GAP, BRACKET_LEG, BRACKET_LIFT, BRACKET_INSET = rem(0.4), rem(0.35), rem(0.15), rem(0.3)

#: `ChartLegendContent`'s own layout, read off shadcn's `chart.tsx` (not
#: guessed): `"flex items-center justify-center gap-4"` for the row, centred
#: BELOW the chart (`pt-3` when `verticalAlign` defaults to bottom), each item
#: `"flex items-center gap-1.5"`. Reused verbatim rather than approximated.
LEGEND_ITEM_GAP = rem(1.0)         # gap-4
LEGEND_SWATCH_GAP = rem(0.375)     # gap-1.5
LEGEND_PT = rem(0.75)              # pt-3, the gap above the legend row
#: `"h-2 w-2 shrink-0 rounded-[2px]"` -- an 8 px ROUNDED SQUARE, not a dot.
#: Bumped slightly over the bare 8 px so this chart's own dashed/dotted
#: outline (the series encoding below) still reads as dashed at this size;
#: shadcn's own swatch is a flat fill with no outline to distinguish.
LEGEND_SWATCH = rem(0.65)
LEGEND_RADIUS = 2.0                # rounded-[2px] -- a literal pixel value,
                                    # not a rem step, exactly as the class reads
#: Reserved at the bottom of the plot ONLY when a legend is actually drawn
#: (the overlay case) -- see `_plot_rect`. Generous on purpose: the exact
#: figure depends on live font metrics (`_legend` computes the real `cy`),
#: this only has to be enough room for them to fit inside PLUS `PAD_B`'s own
#: share (the condition-label row).
LEGEND_ROW_H = rem(2.0)

#: The bubble radius range. AREA carries the fraction (see the module
#: docstring), so the radius interpolates by its square root, never linearly.
MIN_R = rem(0.45)
MAX_R = rem(2.2)

#: A condition name under its column WRAPS (`chart_text.wrap_text`, the rule the figure
#: exporter uses too). Reported on a smaller screen: names such as
#: ``20260923_B.subtilis_HNC_mNG.d2.1`` were drawn in a fixed 8-rem box on one line, cut
#: at BOTH ends, so three conditions differing only in their ending read the same.
LABEL_MAX_LINES = chart_text.LABEL_MAX_LINES


def wrap_label(text, fm: QFontMetricsF, width: float,
               max_lines: int = LABEL_MAX_LINES) -> list:
    """``text`` as at most ``max_lines`` lines no wider than ``width``."""
    return chart_text.wrap_text(
        text, fm.horizontalAdvance, width, max_lines,
        elide=lambda s, w: fm.elidedText(s, Qt.ElideLeft, int(w)))


def column_label_width(box_width: float, n_columns: int, span: float | None = None) -> float:
    """What one column's name may span: its share of the axis (the x range runs
    ``n + 0.2`` column steps, see `_scales`, or ``span`` steps where groups add gaps), less a gap so
    neighbours never touch."""
    spacing = box_width / (span if span else max(1, n_columns) + 0.2)
    return max(rem(4), spacing - rem(0.75))


def label_line_count(names, fm: QFontMetricsF, box_width: float, span: float | None = None) -> int:
    """How many lines the tallest wrapped name needs: the plot leaves room for them."""
    width = column_label_width(box_width, len(names), span)
    return max([len(wrap_label(n, fm, width)) for n in names] or [1])


def draw_column_labels(p: QPainter, fm: QFontMetricsF, box: QRectF, xr: tuple,
                       names, xs=None) -> None:
    """Each name centred under its column, wrapped (see `wrap_label`); ``xs`` the columns' positions when they
    are grouped (`BubblePlotData.layout`), else column k stands at k."""
    width = column_label_width(box.width(), len(names), (xr[1] - xr[0]) if xs is not None else None)
    for i, name in enumerate(names):
        x = BubbleChart._map_x(box, i if xs is None else xs[i], xr)
        for k, line in enumerate(wrap_label(name, fm, width)):
            p.drawText(QRectF(x - width / 2, box.bottom() + rem(0.3) + k * fm.height(),
                              width, fm.height()), Qt.AlignHCenter | Qt.AlignTop, line)


#: The row a group's header takes under the column labels: a rule over the group's columns and the outer factor's
#: value under it (folder designs phase 4).
GROUP_HEADER_GAP = rem(0.35)


def group_header_room(groups, fm: QFontMetricsF) -> float:
    """The height the group headers take under the labels: none without groups."""
    return GROUP_HEADER_GAP * 2 + fm.height() if groups else 0.0


def draw_group_headers(p: QPainter, fm: QFontMetricsF, box: QRectF, xr: tuple, groups, top: float,
                       ink: QColor, rule: QColor) -> None:
    """Each group's outer value centred under its columns, a thin rule over them, from ``top`` down."""
    for group in groups:
        x0 = BubbleChart._map_x(box, group["x0"] - 0.4, xr)
        x1 = BubbleChart._map_x(box, group["x1"] + 0.4, xr)
        line = top + GROUP_HEADER_GAP
        pen = QPen(rule)
        pen.setWidthF(1.0)
        p.setPen(pen)
        p.drawLine(QPointF(x0 + rem(0.25), line), QPointF(x1 - rem(0.25), line))
        p.setPen(ink)
        text = fm.elidedText(str(group["label"]), Qt.ElideRight, int(max(rem(3), x1 - x0)))
        p.drawText(QRectF(x0, line + rem(0.15), x1 - x0, fm.height()), Qt.AlignHCenter | Qt.AlignTop, text)

#: THE SAMPLE UNDER THE COLUMNS (30 Oct 2026: "3 replicates · 1,271 tracks", `bubble_view.sample_marks`): the gap
#: over its lines. It stands under each column's label, or with fits side by side once under each group's header.
SAMPLE_GAP = rem(0.15)


def sample_widths(box: QRectF, xr: tuple, marks) -> list:
    """Each sample's room in pixels: its span on the axis, less the gap column labels keep."""
    return [max(rem(3), BubbleChart._map_x(box, m["x1"], xr) - BubbleChart._map_x(box, m["x0"], xr) - rem(0.5))
            for m in marks]


def sample_line_count(marks, fm: QFontMetricsF, box: QRectF, xr: tuple) -> int:
    """How many lines the tallest sample takes (`bubble_view.sample_lines`): 0 without one."""
    return max([len(sample_lines(m["text"], fm.horizontalAdvance, w))
                for m, w in zip(marks, sample_widths(box, xr, marks))] or [0])


def sample_room(marks, fm: QFontMetricsF, box: QRectF, xr: tuple) -> float:
    """The height the sample lines take under the axis: none without them."""
    lines = sample_line_count(marks, fm, box, xr)
    return SAMPLE_GAP + lines * fm.height() if lines else 0.0


def draw_samples(p: QPainter, fm: QFontMetricsF, box: QRectF, xr: tuple, marks, top: float, ink: QColor) -> float:
    """Each sample centred under its column (or group), from ``top`` down, in ``ink``; returns where they end."""
    if not marks:
        return top
    p.setPen(ink)
    y = top + SAMPLE_GAP
    for mark, width in zip(marks, sample_widths(box, xr, marks)):
        x = (BubbleChart._map_x(box, mark["x0"], xr) + BubbleChart._map_x(box, mark["x1"], xr)) / 2.0
        for k, line in enumerate(sample_lines(mark["text"], fm.horizontalAdvance, width)):
            p.drawText(QRectF(x - width / 2.0 - rem(0.25), y + k * fm.height(), width + rem(0.5), fm.height()),
                       Qt.AlignHCenter | Qt.AlignTop, line)
    return top + sample_room(marks, fm, box, xr)


#: How each SERIES is told apart in an overlay -- the exact encoding
#: `plots.BUBBLE_SERIES_STYLES` uses, re-expressed for `QPen`/`QColor` instead
#: of matplotlib's linestyle strings. Deliberately REDUNDANT (a shade AND an
#: outline), for the reason recorded there: colour alone does not survive
#: greyscale reproduction.
_SERIES_STYLES = (
    {"fill": 1.00, "line": Qt.SolidLine, "width": 1.1},
    {"fill": 0.00, "line": Qt.DashLine, "width": 1.6},
    {"fill": 0.55, "line": Qt.DotLine, "width": 1.6},
    {"fill": 0.25, "line": Qt.DashDotLine, "width": 1.6},
)


def _series_style(index: int) -> dict:
    return _SERIES_STYLES[index % len(_SERIES_STYLES)]


def _lighten(colour: QColor, amount: float) -> QColor:
    """Blend towards white -- the same formula as `compare_plots._lighten`,
    re-expressed for `QColor`. ``amount`` 1.0 keeps the colour, 0.0 gives white."""
    def ch(c):
        return round(255 - amount * (255 - c))
    return QColor(ch(colour.red()), ch(colour.green()), ch(colour.blue()))


def readable_ink(fill: QColor) -> QColor:
    """White or near-black, whichever reads on ``fill`` (`chart_text.readable_ink`)."""
    return QColor(chart_text.readable_ink(fill.name()))


#: How the fitted D is marked: a short tick either side of the bubble, outside it, starting AT its edge (1 Nov 2026,
#: asked: "the lines left and right to bubble, they are too far away from bubble, move them at the bubble edge"; it
#: stood 3 px off, and the export 0.3 of its text size).
D_TICK_GAP = 0.0
D_TICK_LEN = rem(0.45)
#: Between a bubble's edge and the number printed under it.
LABEL_GAP = 2.0


def paint_bubble(p: QPainter, t, centre: QPointF, r: float, colour: QColor, style: dict,
                 *, hovered: bool = False) -> None:
    """One bubble, as both bubble charts draw it.

    Asked for "more modern and clean", and "check where the numbers are, it's a bit weird":
    the D was a dark bar through the bubble's middle, so it struck through the fraction
    printed there ("45%" read as crossed out), every bubble had a near-black outline, and
    a small bubble's number spilled over its edge. Now:

    * the fill is the condition's colour, ringed in the page's own background so the grid
      and a neighbour separate cleanly (an OUTLINED series keeps its dashed edge in the
      colour -- the redundant encoding that survives greyscale, `_SERIES_STYLES`);
    * the D is marked by two short ticks just OUTSIDE the bubble, still exact, never
      over the number;
    * the number stands OUTSIDE every bubble, in the page's ink -- as the exported figure
      prints it (asked: "we need the same in both"). It was inside, or beside when it did
      not fit: a 5 % bubble cannot hold its number at any readable size, so one column read
      as two kinds of number. It is drawn by `paint_numbers`, after every bubble, under its
      bubble, over it only where under hits another bubble, never beside (1 Nov 2026: a
      bubble drawn later covered the number under its neighbour; numbers moved aside read as
      noise);
    * a hovered bubble gets a soft halo.
    """
    x, y = centre.x(), centre.y()
    solid = style["fill"] >= 1.0
    if hovered:
        halo = QColor(colour)
        halo.setAlphaF(0.22)
        p.setPen(Qt.NoPen)
        p.setBrush(halo)
        p.drawEllipse(centre, r + rem(0.3), r + rem(0.3))
    fill = QColor(colour) if solid else _lighten(colour, style["fill"])
    pen = QPen(QColor(t.background) if solid else colour)
    pen.setWidthF(1.5 if solid else style["width"])
    pen.setStyle(Qt.SolidLine if solid else style["line"])
    p.setPen(pen)
    p.setBrush(fill)
    p.drawEllipse(centre, r, r)

    tick = QPen(QColor(colour).darker(135))
    tick.setWidthF(1.6)
    tick.setCapStyle(Qt.RoundCap)
    p.setPen(tick)
    p.drawLine(QPointF(x - r - D_TICK_GAP - D_TICK_LEN, y), QPointF(x - r - D_TICK_GAP, y))
    p.drawLine(QPointF(x + r + D_TICK_GAP, y), QPointF(x + r + D_TICK_GAP + D_TICK_LEN, y))


def paint_numbers(p: QPainter, t, spots, font, box: QRectF) -> list:
    """Every bubble's number, drawn AFTER every bubble so none is covered: ``spots`` are
    ``(x, y, r, text)``, each number at the side of its own bubble `chart_text.place_numbers`
    picks (under it; over it only where under hits another bubble), backed in the page's
    colour where neither side was clear. Returns the placements, which the significance
    brackets stand clear of."""
    fm = QFontMetricsF(font)
    sizes = [(fm.horizontalAdvance(text) + 2.0, fm.height()) if text else (0.0, 0.0)
             for _x, _y, _r, text in spots]
    places = chart_text.place_numbers([(x, y, r) for x, y, r, _text in spots], sizes,
                                      gap=LABEL_GAP, reach=D_TICK_GAP + D_TICK_LEN,
                                      bounds=(box.left(), box.top(), box.right(), box.bottom()))
    p.setFont(font)
    for (_x, _y, _r, text), place in zip(spots, places):
        if place["rect"] is None:
            continue
        left, top, right, bottom = place["rect"]
        rect = QRectF(left, top, right - left, bottom - top)
        if not place["clear"]:
            back = QColor(t.background)
            back.setAlphaF(0.85)
            p.setPen(Qt.NoPen)
            p.setBrush(back)
            p.drawRoundedRect(rect.adjusted(-2.0, 0.0, 2.0, 0.0), 3.0, 3.0)
        p.setPen(QColor(t.foreground))
        p.drawText(rect, Qt.AlignCenter, text)
    return places


def bubble_room(font) -> tuple:
    """``(top, bottom)`` px the D axis keeps beyond the outermost Ds: the largest bubble and a
    number, either way (`chart_text.bubble_axis_range`, the exported figure's rule too). Over
    the top bubble too since 1 Nov 2026: a number goes over its bubble where one of its column
    stands under it, and without the room the top one had nowhere to go but over its
    neighbour."""
    label_h = QFontMetricsF(font).height()
    room = MAX_R + LABEL_GAP + label_h + 4.0
    return room, room


class _BubbleTooltip(QWidget):
    """The bubble chart's own hover tooltip -- shadcn's `ChartTooltipContent`
    in its "line" indicator style, populated for ONE hovered `BubblePoint`:
    a Condition LABEL, then D and Fraction as two plain rows, with ONE
    vertical bar running the full height of the content on the left edge --
    condition-coloured, series-dashed -- instead of a swatch per row. Asked
    for directly, after the per-row square swatch this replaced.

    **NOT `widgets.tooltip.Tooltip`.** That class is ONE LINE of plain text,
    sized from `QFontMetrics.horizontalAdvance` alone, and shared across the
    whole app for exactly that -- a button's hover-help. Retrofitting
    multi-row, bar-carrying content into it would risk that far more
    common case for the sake of this one chart. A small, bespoke
    class scoped to this one chart is the same call this whole file already
    makes for its axes, legend and bubbles -- hand-painted rather than
    composed from a shared, general widget.

    **The bar is painted directly in `paintEvent`, not a child widget.** A
    dashed vertical line is one `QPainter.drawLine` call with the right pen;
    a widget would need its own `paintEvent` for the same thing, which is
    what the per-row swatch this replaced actually was. Its colour is the
    condition's OWN colour, undiminished -- not `_paint_bubbles`'s
    `_lighten`d fill, which exists to make an overlaid series read as
    filled-vs-outline on a filled CIRCLE and has nothing to say about a
    4 px bar. Series is carried by the pen's dash pattern alone
    (`_series_style`'s own `"line"`), matching the legend beneath the chart.

    **The shadow is drawn DIRECTLY (`draw_shadow`), not through
    `Surface`/`ShadowHost`.** A `Surface` only paints ITS OWN shadow if some
    ANCESTOR is a `ShadowHost` walking `findChildren` for it -- and
    `menus.Panel`, the one other floating popover in this app, already found
    that unreliable for a widget reparented onto the window at show time
    (see its own module docstring) and self-shadows instead. Ported
    verbatim from `Panel.paintEvent`.

    **NOT ANIMATED** -- no fade, no `GRACE_MS` leave-deferral, unlike
    `Tooltip`. A bubble's hit area is real screen area, not a 1 px widget
    edge, so a hover rarely jitters at its boundary the way that class was
    built to survive; revisit if that turns out false.
    """

    #: The two metric rows, in order. The Series NAME ("Independent") is
    #: never spelled out as its own text -- see the module docstring's own
    #: note on why hover stays to ONE thing: the bar's dash pattern already
    #: carries it, matching the legend beneath the chart, and a third line
    #: here would be the richer-hover-modes idea this project chose not to
    #: build.
    ROW_NAMES = ("D", "Fraction")

    #: `w-1 rounded-[2px]` -- shadcn's own `indicator="line"` bar, for the
    #: FILLED (solid-series) shape. 4 px, not `LEGEND_SWATCH`'s 10: a
    #: full-height BAR reads at a different scale than a per-row dot.
    BAR_W = rem(0.25)
    #: The HOLLOW (dashed-series) outline, slightly WIDER than the filled
    #: bar -- asked for directly: matched exactly, a thin stroke's ink
    #: still reads as lighter than a filled block of the same footprint (see
    #: `paintEvent`'s own note on the stroke width), and a touch more room
    #: closes that gap better than a thicker stroke alone did.
    BAR_OUTLINE_W = rem(0.375)
    #: The bar's own inset from the card's left edge, and its gap to the
    #: text -- sized from the WIDER of the two shapes so the text column
    #: lands in the same place whichever kind of bar is showing.
    BAR_LEFT = rem(0.5)
    BAR_GAP = rem(0.5)

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._room = shadow_room(t, t.shadow_md)
        #: Set by `set_point`, read by `paintEvent` -- see the class
        #: docstring on why the bar is painted directly rather than as a
        #: child widget.
        self._bar_colour = QColor(Qt.transparent)
        self._bar_line_style = Qt.SolidLine
        lay = QVBoxLayout(self)
        pad_x, pad_y = rem(0.625), rem(0.5)
        #: The bar's own vertical extent -- see `paintEvent`. Tied to `pad_y`
        #: (the SAME margin the content above/below is laid out with) rather
        #: than a separate constant: asked for directly, after the first cut
        #: put the bar's ends a couple of px PAST where the label and last
        #: row actually start/end, reading as longer than shadcn's own
        #: (which hugs its content tightly, `self-stretch` on the row).
        self._pad_y = pad_y
        left = self._room + self.BAR_LEFT + max(self.BAR_W, self.BAR_OUTLINE_W) + self.BAR_GAP
        lay.setContentsMargins(left, self._room + pad_y,
                               self._room + pad_x, self._room + pad_y)
        lay.setSpacing(rem(0.3))
        # `font-medium` (500), not `semibold` (600) -- shadcn's own
        # `ChartTooltipContent` label weight; asked for directly after the
        # first cut read heavier than the reference.
        self._label = label("", ty.ui("sm", "medium", px=text_px(0.875)), t.foreground)
        lay.addWidget(self._label)
        self._values: list = []
        for name in self.ROW_NAMES:
            row = QHBoxLayout()
            row.setSpacing(rem(0.5))
            row.addWidget(label(name, ty.ui("xs", px=text_px(0.75)), t.muted_foreground))
            row.addStretch(1)
            val = label("", ty.ui("xs", px=text_px(0.75), tabular=True), t.foreground)
            row.addWidget(val)
            lay.addLayout(row)
            self._values.append(val)
        self.hide()

    def set_point(self, *, condition: str, bar_colour: QColor, line_style,
                  d_text: str, fraction_text: str) -> None:
        self._label.setText(condition)
        self._bar_colour = bar_colour
        self._bar_line_style = line_style
        self._values[0].setText(d_text)
        self._values[1].setText(fraction_text)
        self.adjustSize()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        box = QRectF(self.rect().adjusted(
            self._room, self._room, -self._room, -self._room))
        radius = self.t.radius_md
        draw_shadow(p, box, radius, self.t, self.t.shadow_md)
        p.setPen(QPen(QColor(self.t.border), 1.0))
        p.setBrush(QColor(self.t.popover))
        p.drawRoundedRect(box.adjusted(0.5, 0.5, -0.5, -0.5), radius, radius)
        # The bar itself: a FILLED rounded rect for the solid series (a
        # plain bar, matching shadcn's own), a HOLLOW one, dash-STROKED
        # around its own rounded outline, for every other series -- asked
        # for directly, after the first cut drew a straight dashed LINE
        # instead, which (at 4 px with round caps) reads as a stack of short
        # blobs rather than the open, dashed CAPSULE shadcn's own reference
        # for a non-solid indicator shows. The two widths (`BAR_W`/
        # `BAR_OUTLINE_W`) are deliberately close but not equal any more --
        # a plain HOLLOW shape at the SAME footprint as a FILLED one still
        # reads visually lighter/narrower (less ink), which a slightly wider
        # outline corrects better than thickening its stroke alone did.
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
            # THINNER than `_series_style`'s own 1.6 -- that width is
            # measured against a ~40 px BUBBLE; at the bar's small footprint
            # the same stroke leaves almost no hollow interior and reads as
            # a single chunky line rather than an open, dashed outline.
            pen.setWidthF(1.4)
            pen.setStyle(self._bar_line_style)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            inset = pen.widthF() / 2
            p.drawRoundedRect(bar.adjusted(inset, inset, -inset, -inset),
                              bar_radius, bar_radius)
        p.end()


class BubbleChart(QWidget):
    """Paints one `BubblePlotData`, or a centred status message before the
    first Run / after a failed one. Never accumulates: each `set_data` /
    `set_message` call replaces the whole picture, matching the rule
    `window._show_message`'s matplotlib ancestor followed.

    Also owns the ONE hover behaviour this chart has -- see the module
    docstring's own note on why it is one thing (a tooltip) and not three.
    """

    #: The `BubblePoint` under the cursor, or `None` -- emitted on every
    #: CHANGE (never on every mouse-move frame). `BubbleCard` listens to
    #: drive the table's scroll-and-highlight; nothing here knows the table
    #: exists, matching the split `widgets.py`'s own docstring draws between
    #: a plain-numpy hit test and the Qt widget that reacts to it.
    hovered = Signal(object)

    #: A bubble was CLICKED -- the condition it belongs to (`BubblePoint.
    #: column` resolved through `BubblePlotData.conditions`, the same
    #: resolution `_show_tip` already does for the hover tooltip). Emits the
    #: CONDITION, not the population: the detail page this drives shows
    #: every population of that condition at once (matching the legacy
    #: inspector), so which one of possibly several bubbles was clicked is
    #: not itself meaningful -- any bubble in a condition's column leads to
    #: the same page.
    population_clicked = Signal(str)

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("background:transparent;")
        self.setMouseTracking(True)
        # `tabular=True` on BOTH: `_tick_font` draws the y-axis D VALUES
        # and (mixed in) the x-axis condition names -- `tnum` only touches
        # digit glyphs, so the names are unaffected -- and `_label_font`
        # draws the percentage baked onto every bubble, which shifts width
        # digit-to-digit without it (measured already for `_BubbleTooltip`'s
        # own D/Fraction rows, `tabular=True` there since Phase 1; the chart
        # itself never carried it).
        self._tick_font = ty.ui("xs", px=TICK_PX, tabular=True)
        self._label_font = ty.ui("xs", px=TICK_PX, weight="semibold", tabular=True)
        self._message: str | None = "Press Compute on the right, then Run."
        self._data: BubblePlotData | None = None
        #: Typed ranges (the card's Axes button, `axes_view`): ``{"y": (from, to) or None}``; None ends automatic.
        self.axes: dict = {}
        #: The y range as last drawn (µm²/s), for the tests and the export's "as shown".
        self.drawn_y: tuple = (None, None)
        #: The significance brackets painted over the plot (`set_brackets`): ``[{"a", "b",
        #: "text"}]``, cleared by every new `set_data` so none outlives the fit it was for.
        self._brackets: list = []
        #: condition name -> hex colour, resolved by `WorkspaceWindow` from
        #: the workspace's own Conditions tool -- see the module docstring.
        self._colours: dict = {}
        #: The bubble currently under the cursor, or `None`.
        self._hovered: BubblePoint | None = None
        # `_BubbleTooltip`, not `tooltip.Tooltip` -- see that class's own
        # docstring for why. Constructed with THIS chart as its parent, like
        # `tip.attach` does; `_show_tip` reparents it onto the window the
        # first time it is actually shown, for the same reason `Tooltip.
        # show_for` defers that: at construction the chart may not be in its
        # final window yet.
        self._tip = _BubbleTooltip(t, self)

    def set_message(self, text: str) -> None:
        self._message = text
        self._data = None
        self._set_hovered(None)
        self.update()

    def set_data(self, data: BubblePlotData, colours: dict) -> None:
        self._data = data
        self._brackets = []
        self._colours = dict(colours)
        self._message = None
        # A fresh Run's points are new objects; whatever this was hovering
        # from the LAST one is gone, and holding onto it would drive the
        # table to highlight a key that may no longer mean what it did.
        self._set_hovered(None)
        # RE-CHECKED against the cursor's CURRENT position shortly after,
        # not just left cleared -- a first Run's page-swap (`self.
        # _placeholder` -> `self._card`) delivers this widget a fresh
        # Show/Enter that re-triggers `mouseMoveEvent` on its own, but a
        # RE-run leaves the chart the same already-visible, already-
        # enabled widget throughout (`BubbleCard.set_running` now covers
        # it with `_DimOverlay` instead of touching THIS widget at all --
        # see that class's own docstring for why), so nothing else ever
        # asks "where is the mouse" again on its own; hiding that overlay
        # does ordinarily deliver a fresh Enter here, but this recheck is
        # the belt to that overlay's suspenders rather than trusting it
        # alone. Deferred one event-loop tick (`QTimer.singleShot(0,
        # ...)`), which costs nothing observable and is one fewer thing
        # to re-litigate if a future caller's timing turns out less clean
        # than today's.
        QTimer.singleShot(0, self, self._recheck_hover)   # dropped by Qt if the chart is gone
        self.update()

    def set_brackets(self, marks: list) -> None:
        """The significance brackets to paint over the plot (`significance.chart_view`'s
        ``marks``: column indices low first, the p as printed). Empty paints none; the plot
        gives up the room they take at its top, as the exported figure does."""
        self._brackets = [dict(m) for m in marks]
        self.update()

    def brackets(self) -> list:
        return [dict(m) for m in self._brackets]

    def _bracket_tiers(self, width: float) -> list:
        """``[(mark, tier)]`` with each mark's legs (``x0``, ``x1``) in column units:
        `significance.bracket_tiers`, the export's rule, with the text measured in this font.
        Tiers are counted per POPULATION and side: each group stacks at its own bubbles."""
        from ... import significance as sig

        if not self._brackets or self._data is None:
            return []
        data = self._data
        lo, hi = data.x_range()
        col_px = max(1.0, width / max(1.0, hi - lo - 0.2))   # one column step, in pixels
        fm = QFontMetricsF(self._label_font)
        inset = min(0.1, BRACKET_INSET / col_px)
        groups: dict = {}
        for m in self._brackets:
            # IN POSITIONS, not column numbers: grouped columns stand apart (`BubblePlotData.layout`); the
            # numbers stay as ``ia`` / ``ib`` for finding the bubbles a bracket spans
            xa, xb = data.column_x(m["a"]), data.column_x(m["b"])
            mark = dict(m, ia=m["a"], ib=m["b"], a=xa, b=xb, x0=xa + inset, x1=xb - inset)
            mark.setdefault("population", 1)
            mark.setdefault("side", "above")
            groups.setdefault((mark["population"], mark["side"]), []).append(mark)
        placed = []
        for marks in groups.values():
            tiers = sig.bracket_tiers(
                marks, lambda m: (fm.horizontalAdvance(m["text"]) / 2.0 + rem(0.25)) / col_px)
            placed += list(zip(marks, tiers))
        return placed

    def _tier_px(self) -> float:
        return BRACKET_LEG + BRACKET_LIFT + QFontMetricsF(self._label_font).height() + rem(0.2)

    def _bracket_bands(self, width: float) -> tuple:
        """``(above, below)`` in pixels: the room the D axis makes over the TOPMOST population
        for its brackets, and under the slowest for its. A middle population's brackets stand
        between the populations and take no axis room."""
        placed = self._bracket_tiers(width)
        if not placed:
            return 0.0, 0.0
        top = max((pt.population for pt in self._data.points), default=1)

        def band(tiers):
            return BRACKET_GAP + (max(tiers) + 1) * self._tier_px() if tiers else 0.0

        return (band([t for m, t in placed if m["side"] == "above" and m["population"] == top]),
                band([t for m, t in placed if m["side"] == "below"]))

    def _paint_brackets(self, p: QPainter, box: QRectF, xr: tuple, data, numbers=None) -> None:
        """Each bracket AT ITS POPULATION's bubbles in the columns it spans: over their
        highest edge, legs down; or, for the slowest, under the lowest number, legs up and the
        text beneath (`significance.bracket_side`). ``numbers`` are where the bubbles' numbers
        stand (`paint_numbers`, in `_point_geometry`'s order): a bracket clears its bubbles'
        numbers on its side, wherever they stand."""
        placed = self._bracket_tiers(box.width())
        if not placed:
            return
        fm = QFontMetricsF(self._label_font)
        bubbles: dict = {}
        for k, (pt, x, y, r) in enumerate(self._point_geometry(data)):
            place = numbers[k] if numbers else {"rect": (x, y + r + LABEL_GAP, x, y + r + LABEL_GAP + fm.height())}
            over, under = chart_text.number_reach((x, y, r), place)
            bubbles.setdefault(pt.column, []).append((pt, y, over, under))
        pen = QPen(QColor(self.t.foreground))
        pen.setWidthF(1.0)
        p.setPen(pen)
        p.setFont(self._label_font)
        for mark, tier in placed:
            span = [b for c in range(mark["ia"], mark["ib"] + 1) for b in bubbles.get(c, [])]
            if mark["population"] == "below":
                # the share below the D threshold: every bubble below it (the slowest's
                # when none is), as the exported figure picks (`_bracket_spots`)
                chosen = ([b for b in span if b[0].d < mark["below_d"]]
                          or [b for b in span if b[0].population == 1])
            else:
                chosen = [b for b in span if b[0].population == mark["population"]]
            spots = [(y, over, under) for _pt, y, over, under in chosen]
            if not spots:
                continue
            x0 = self._map_x(box, mark["x0"], xr)
            x1 = self._map_x(box, mark["x1"], xr)
            mid = (x0 + x1) / 2.0
            width = fm.horizontalAdvance(mark["text"]) + rem(0.5)
            if mark["side"] == "below":
                anchor = max(y + under for y, _over, under in spots)
                line = anchor + BRACKET_GAP + BRACKET_LEG + tier * self._tier_px()
                leg = line - BRACKET_LEG
                text = QRectF(mid - width / 2.0, line + BRACKET_LIFT, width, fm.height())
                align = Qt.AlignHCenter | Qt.AlignTop
            else:
                anchor = min(y - over for y, over, _under in spots)
                line = anchor - BRACKET_GAP - BRACKET_LEG - tier * self._tier_px()
                leg = line + BRACKET_LEG
                text = QRectF(mid - width / 2.0, line - BRACKET_LIFT - fm.height(), width,
                              fm.height())
                align = Qt.AlignHCenter | Qt.AlignBottom
            p.drawLine(QPointF(x0, leg), QPointF(x0, line))
            p.drawLine(QPointF(x0, line), QPointF(x1, line))
            p.drawLine(QPointF(x1, line), QPointF(x1, leg))
            p.drawText(text, align, mark["text"])

    def _recheck_hover(self) -> None:
        """Only a chart that can BE hovered may show its tooltip. Without the guard, a Run
        landing while this chart's page is hidden (the Apparent Diffusion Bubble Plot is
        computed behind Distribution) showed the tooltip on the window whenever the pointer
        happened to rest where a bubble is drawn -- reported as a tooltip "out of nowhere"."""
        pos = self.mapFromGlobal(QCursor.pos())
        if not self.isVisible() or not self.rect().contains(pos):
            self._set_hovered(None)
            return
        self._set_hovered(self._hit_test(pos))

    def set_colours(self, colours: dict) -> None:
        """Re-colour the ALREADY-DRAWN data -- a palette change in the
        Conditions tool, which never re-fits anything."""
        self._colours = dict(colours)
        self.update()

    def conditions(self) -> tuple:
        return tuple(self._data.conditions) if self._data is not None else ()

    def _colour_for(self, condition: str) -> QColor:
        return QColor(self._colours.get(condition, self.t.muted_foreground))

    # -- hover: one tooltip, plus a signal for BubbleCard's table sync ------ #
    def leaveEvent(self, e):
        self._set_hovered(None)
        super().leaveEvent(e)

    def mouseMoveEvent(self, e):
        self._set_hovered(self._hit_test(e.position()))
        super().mouseMoveEvent(e)

    # -- click: drill into the clicked condition's per-replicate detail --- #
    def mouseReleaseEvent(self, e):
        """A click, the SAME convention every plain button in this app uses
        (`CloseButton`, `_Row`'s own `triggered`): tested at RELEASE, inside
        this widget's own bounds -- never reused from `_hit_test`'s own
        hover state, so a press-and-drag that ends outside the chart (or
        outside any bubble) does not fire one.
        """
        if e.button() == Qt.LeftButton:
            pt = self._hit_test(e.position())
            if pt is not None and self.rect().contains(e.position().toPoint()):
                self.population_clicked.emit(self._data.conditions[pt.column])
        super().mouseReleaseEvent(e)

    def _hit_test(self, pos: QPointF) -> BubblePoint | None:
        """The TOPMOST bubble containing `pos` -- last drawn wins on an
        overlap, matching paint order (later points sit visually on top)."""
        if self._data is None:
            return None
        hit = None
        for pt, x, y, r in self._point_geometry(self._data):
            dx, dy = pos.x() - x, pos.y() - y
            if dx * dx + dy * dy <= r * r:
                hit = pt
        return hit

    def _set_hovered(self, pt: BubblePoint | None) -> None:
        if pt is self._hovered:
            return
        self._hovered = pt
        # A pointing hand, ONLY over a bubble -- the same affordance every
        # other clickable control in this app already carries (`_CrumbLink`,
        # `ToolButton`), so hovering one now tells you it leads somewhere
        # before you click it. `unsetCursor` rather than `setCursor(Qt.
        # ArrowCursor)`: this widget has no cursor override of its own
        # otherwise, and unsetting restores whatever an ancestor or the
        # application asks for instead of pinning the arrow here forever.
        if pt is None:
            self.unsetCursor()
        else:
            self.setCursor(Qt.PointingHandCursor)
        if pt is None:
            self._tip.hide()
        else:
            for p, x, y, r in self._point_geometry(self._data):
                if p is pt:
                    self._show_tip(pt, x, y, r)
                    break
        self.update()                       # the hovered bubble's halo (`paint_bubble`)
        self.hovered.emit(pt)

    def _show_tip(self, pt: BubblePoint, x: float, y: float, r: float) -> None:
        """Populate and position `_BubbleTooltip` for the bubble at widget
        coordinates `(x, y)`, radius `r`. The tooltip's bar is the
        condition's own colour (undiminished -- see `_BubbleTooltip`'s own
        docstring on why it does not reuse the bubble's `_lighten`d fill)
        and the series' dash pattern, the SAME `_series_style` the bubble
        itself is drawn with. The tooltip answers "what is this single
        point"; the table (kept in sync via `hovered`) answers "how does it
        sit next to the others" -- see the module docstring for why both
        stay.
        """
        data = self._data
        condition = data.conditions[pt.column]
        style = _series_style(pt.style_index)
        base = self._colour_for(condition)
        pct = fraction_text(pt)
        self._tip.set_point(
            condition=condition, bar_colour=base, line_style=style["line"],
            d_text=d_text(pt, " µm²/s"), fraction_text=f"{pct}, n={pt.n_replicates}")
        window = self.window()
        if self._tip.parentWidget() is not window:
            self._tip.setParent(window)
        self._position_tip(window, x, y, r)
        self._tip.raise_()
        self._tip.show()

    def _position_tip(self, window, x: float, y: float, r: float) -> None:
        """Centred ABOVE the bubble, clamped to `window` -- flipped BELOW
        when there is not enough room above (a bubble near the plot's own
        top edge), the same "clamp, don't escape" rule `Tooltip.show_for`
        follows for an in-window tooltip."""
        w, h = self._tip.width(), self._tip.height()
        top_left = self.mapTo(window, QPoint(round(x - r), round(y - r)))
        cx = top_left.x() + round(r)
        bubble_top = top_left.y()
        bubble_bottom = top_left.y() + round(2 * r)
        gap = rem(0.375)
        tip_x = max(4, min(cx - w // 2, window.width() - w - 4))
        tip_y = bubble_top - gap - h
        if tip_y < 4:
            tip_y = bubble_bottom + gap
        self._tip.move(tip_x, tip_y)

    # -- geometry ----------------------------------------------------------- #
    def _plot_rect(self) -> QRectF:
        # The legend row (series only, see `_legend`) sits BELOW the x-axis
        # labels, shadcn's own placement -- so it needs its own slice of
        # `PAD_B`, reserved only when there is a legend to draw at all.
        has_legend = bool(self._data and self._data.series_labels)
        width = max(1.0, self.width() - PAD_L - PAD_R)
        pad_b = PAD_B + (LEGEND_ROW_H if has_legend else 0) + self._label_room(width)
        return QRectF(PAD_L, PAD_T, width, max(1.0, self.height() - PAD_T - pad_b))

    def _label_room(self, box_width: float) -> float:
        """The extra height wrapped condition names need below the axis, the group headers under them, and the sample
        under each column or group (`bubble_view.sample_marks`)."""
        if not self._data:
            return 0.0
        data = self._data
        fm = QFontMetricsF(self._tick_font)
        lo, hi = data.x_range()
        span = (hi - lo) if data.layout else None
        groups = data.layout.get("groups") if data.layout else []
        return ((label_line_count(data.column_labels(), fm, box_width, span) - 1) * fm.height()
                + group_header_room(groups, fm)
                + sample_room(sample_marks(data), fm, QRectF(PAD_L, 0.0, box_width, 1.0), (lo, hi)))

    def _scales(self, data: BubblePlotData) -> tuple:
        """`(box, xr, yr_log)` -- the plot box and axis ranges `_paint_bubbles`
        and `_point_geometry` both need, split out so hit-testing reads the
        SAME scales painting used and can never disagree with what is drawn."""
        box = self._plot_rect()
        xr = data.x_range()
        # a D of 0 (the immobile population of exact-zero jumps) is drawn at a "0" tick under the rest
        # (`chart_text.zero_d_stand_in`); every other D where it is
        self.zero_d = chart_text.zero_d_stand_in(pt.d for pt in data.points)
        ds = [chart_text.log_drawn(pt.d, self.zero_d) for pt in data.points] or [0.01, 1.0]
        # Room for the largest bubble above and for a bubble AND its number below, in
        # pixels -- the rule the exported figure uses in points -- and for the significance
        # brackets over the topmost population and under the slowest (`_bracket_bands`).
        top, bottom = bubble_room(self._label_font)
        above, below = self._bracket_bands(box.width())
        lo, hi = chart_text.bubble_axis_range(min(ds), max(ds), box.height(),
                                              top + above, bottom + below)
        # a typed range (the card's Axes, 30 Oct 2026): its ends replace the automatic ones
        lo, hi = apply_range(lo, hi, (self.axes or {}).get("y"), log=True)
        self.drawn_y = (lo, hi)
        yr_log = (math.log10(lo), math.log10(hi))
        return box, xr, yr_log

    def _point_geometry(self, data: BubblePlotData):
        """Yield `(point, x, y, r)` for every drawn bubble, in WIDGET
        coordinates -- the one source of truth `_paint_bubbles` (drawing)
        and `_hit_test`/`_set_hovered` (hover) both read."""
        box, xr, yr_log = self._scales(data)
        for pt in data.points:
            x = self._map_x(box, data.column_x(pt.column) + pt.x_offset, xr)
            y = self._map_y_log(box, chart_text.log_drawn(pt.d, self.zero_d), yr_log)
            r = MIN_R + (MAX_R - MIN_R) * math.sqrt(max(pt.fraction, 0.0))
            yield pt, x, y, r

    @staticmethod
    def _map_x(box: QRectF, x: float, xr: tuple) -> float:
        f = (x - xr[0]) / (xr[1] - xr[0]) if xr[1] > xr[0] else 0.0
        return box.left() + f * box.width()

    @staticmethod
    def _map_y_log(box: QRectF, y: float, yr_log: tuple) -> float:
        ly = math.log10(max(y, 1e-12))
        f = (ly - yr_log[0]) / (yr_log[1] - yr_log[0]) if yr_log[1] > yr_log[0] else 0.0
        return box.bottom() - f * box.height()

    @staticmethod
    def _log_ticks(lo: float, hi: float, max_ticks: int | None = None) -> list:
        """Decade + {1, 2, 5} ticks covering [lo, hi], thinned to what the axis holds (`chart_text.log_ticks`, the
        exported figure's rule too). `@staticmethod` so `ReplicateChart` (a different
        widget, same axis convention) can call it without an instance."""
        return chart_text.log_ticks(lo, hi, max_ticks)

    # -- painting ------------------------------------------------------------ #
    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        if self._data is None:
            self._paint_message(p)
        else:
            self._paint_bubbles(p, self._data)
        p.end()

    def _paint_message(self, p: QPainter) -> None:
        p.setFont(ty.ui("sm"))
        p.setPen(QColor(self.t.muted_foreground))
        p.drawText(self.rect(), Qt.AlignCenter | Qt.TextWordWrap,
                   self._message or "No data.")

    def _paint_bubbles(self, p: QPainter, data: BubblePlotData) -> None:
        box, xr, yr_log = self._scales(data)
        lo, hi = 10.0 ** yr_log[0], 10.0 ** yr_log[1]
        # the ticks with their labels: a "0" first where a D of 0 is drawn (`chart_text.zero_d_ticks`), as many as the
        # axis's height holds (`chart_text.log_tick_limit`)
        limit = chart_text.log_tick_limit(box.height(), QFontMetricsF(self._tick_font).height())
        labelled = chart_text.zero_d_ticks(self._log_ticks(lo, hi, limit), self.zero_d)
        yt = [v for v, _label in labelled]

        # -- grid (horizontal only, `stroke-border/50`) --
        pen = QPen(QColor(mix(self.t.border, self.t.background, 0.5)))
        pen.setWidthF(1.0 / max(1.0, self.devicePixelRatioF()))
        p.setPen(pen)
        for v in yt:
            y = self._map_y_log(box, v, yr_log)
            p.drawLine(QPointF(box.left(), y), QPointF(box.right(), y))

        # -- axes, filled on a snapped device pixel -- see `Chart._axes` --
        dpr = max(1.0, self.devicePixelRatioF())
        unit = 1.0 / dpr
        xline = round(box.left() * dpr) / dpr
        p.fillRect(QRectF(xline, box.top(), unit, box.height()), QColor(self.t.border))
        yline = round(box.bottom() * dpr) / dpr
        p.fillRect(QRectF(box.left(), yline, box.width(), unit), QColor(self.t.border))

        # -- y ticks + label --
        p.setFont(self._tick_font)
        p.setPen(QColor(self.t.muted_foreground))
        fm = QFontMetricsF(self._tick_font)
        for v, label in labelled:
            y = self._map_y_log(box, v, yr_log)
            p.drawText(QRectF(0, y - fm.height() / 2, box.left() - rem(0.375),
                              fm.height()), Qt.AlignRight | Qt.AlignVCenter, label)
        self._y_axis_label(p, box, chart_text.D_AXIS_LABEL, self._tick_font,
                          self.t.muted_foreground)

        # -- condition labels (x axis), wrapped under their column --
        p.setFont(self._tick_font)
        p.setPen(QColor(self.t.muted_foreground))
        xs = [data.column_x(i) for i in range(len(data.conditions))] if data.layout else None
        labels = data.column_labels()
        draw_column_labels(p, fm, box, xr, labels, xs)
        groups = data.layout.get("groups") if data.layout else []
        lines = label_line_count(labels, fm, box.width(), (xr[1] - xr[0]) if data.layout else None)
        below = box.bottom() + rem(0.3) + lines * fm.height()
        # THE SAMPLE (`bubble_view.sample_marks`): under each column's label, or once under a group of fits side by side
        marks = sample_marks(data)
        muted = QColor(self.t.muted_foreground)
        below = draw_samples(p, fm, box, xr, [m for m in marks if m["under"] == "label"], below, muted)
        if groups:
            draw_group_headers(p, fm, box, xr, groups, below, QColor(self.t.foreground), QColor(self.t.border))
            below += group_header_room(groups, fm)
        p.setFont(self._tick_font)
        draw_samples(p, fm, box, xr, [m for m in marks if m["under"] == "group"], below, muted)

        # -- error columns: a condition too thin to fit, or that failed --
        for i, reason in data.errors.items():
            x = self._map_x(box, data.column_x(i), xr)
            p.setPen(QColor("#c05621"))
            p.drawText(QRectF(x - rem(4.5), box.top(), rem(9), box.height()),
                       Qt.AlignHCenter | Qt.AlignTop | Qt.TextWordWrap, str(reason))

        # -- the bubbles themselves (inside the plot when a range was typed: one outside it is not drawn over the
        # axes) --
        typed = (self.axes or {}).get("y") is not None
        if typed:
            p.save()
            p.setClipRect(QRectF(0.0, box.top(), float(self.width()), box.height()))
        geometry = list(self._point_geometry(data))
        for pt, x, y, r in geometry:
            # colour = CONDITION (see the module docstring), never the
            # population: every bubble in one column shares one hue.
            base = self._colour_for(data.conditions[pt.column])
            # The ticks either side mark the fitted D itself -- a bubble has extent
            # and a D does not, so without them the eye reads its edge as a range.
            paint_bubble(p, self.t, QPointF(x, y), r, base, _series_style(pt.style_index),
                         hovered=pt is self._hovered)
        # the numbers after every bubble, clear of the brackets' room over the topmost population
        above, _below = self._bracket_bands(box.width())
        numbers = paint_numbers(p, self.t, [(x, y, r, fraction_text(pt, compact=True)) for pt, x, y, r in geometry],
                                self._label_font, box.adjusted(0.0, above, 0.0, 0.0))
        self.drawn_numbers = numbers
        if typed:
            p.restore()

        self._paint_brackets(p, box, xr, data, numbers)
        self._legend(p, box, data)

    @staticmethod
    def _y_axis_label(p: QPainter, box: QRectF, text: str, tick_font,
                      colour: str) -> None:
        """Rotated -90°, matching `prototype/shadcn/charts.Chart._axis_label`
        formula exactly. The vertical offset MUST be `-fm.height() / 2`, not
        `-fm.height()`: after the rotation, the text rect's local y axis maps
        onto SCREEN x, so a `-fm.height()` origin (rather than one centred on
        it) drew the whole label band to the LEFT of the translate point --
        off the widget's left edge, which is what read as "broken" on screen.

        `@staticmethod`, like its three neighbours above, so `ReplicateChart`
        draws the SAME D-axis title (`chart_text.D_AXIS_LABEL`) rather than a page that forgot
        one entirely -- asked for directly, once its own chart shipped
        without it.
        """
        p.setFont(tick_font)
        p.setPen(QColor(colour))
        fm = QFontMetricsF(tick_font)
        # A TITLE LONGER THAN THE PLOT IS TALL goes onto two lines (`chart_text.axis_title_lines`, 3 Nov 2026,
        # reported cut at both ends on a Size S jump tile and QC's per-cell outliers); the pair moves right by half a
        # line so its outer line stays inside the widget
        lines = chart_text.axis_title_lines(text, fm.horizontalAdvance, box.height())
        h = fm.height()
        p.save()
        p.translate(rem(0.6) if len(lines) == 1 else max(rem(0.6), h + 1.0), box.center().y())
        p.rotate(-90)
        for i, line in enumerate(lines):
            p.drawText(QRectF(-box.height() / 2, -len(lines) * h / 2 + i * h, box.height(), h),
                       Qt.AlignCenter, line)
        p.restore()

    def _legend(self, p: QPainter, box: QRectF, data: BubblePlotData) -> None:
        """`ChartLegendContent`, read off shadcn's own `chart.tsx`
        (`"flex items-center justify-center gap-4"`, each item
        `"flex items-center gap-1.5"`, swatch `"h-2 w-2 shrink-0
        rounded-[2px]"`) -- CENTRED, BELOW the plot, rounded SQUARES rather
        than the dots this drew before. See `LEGEND_ITEM_GAP` et al.

        ONLY for telling overlaid SERIES apart (fill vs outline); nothing is
        drawn for the classic, single-arrangement case -- colour already
        means condition, and every condition already names itself on the x
        axis directly beneath its own column, so a colour legend would repeat
        that rather than add to it. Swatches here are NEUTRAL
        (`muted_foreground`/`foreground`), never a condition's colour or a
        state's -- a series is orthogonal to both, and colouring its swatch
        would claim otherwise.
        """
        if not data.series_labels:
            return
        p.setFont(self._tick_font)
        fm = QFontMetricsF(self._tick_font)
        neutral = self.t.muted_foreground
        s = LEGEND_SWATCH

        widths = [fm.horizontalAdvance(text) for text in data.series_labels]
        item_w = [s + LEGEND_SWATCH_GAP + w for w in widths]
        total = sum(item_w) + LEGEND_ITEM_GAP * (len(item_w) - 1)
        x = box.left() + (box.width() - total) / 2.0
        # Below the condition-label row (`box.bottom() + rem(0.3) .. + tick
        # height`), `LEGEND_PT` ("pt-3") further down, then centred on its
        # own row -- computed from the REAL font metrics, not a guessed rem.
        tick_h = fm.height()
        lo, hi = data.x_range()
        lines = label_line_count(data.column_labels(), fm, box.width(), (hi - lo) if data.layout else None)
        groups = data.layout.get("groups") if data.layout else []
        cy = (box.bottom() + rem(0.3) + tick_h * lines + group_header_room(groups, fm)
              + sample_room(sample_marks(data), fm, box, (lo, hi)) + LEGEND_PT + tick_h / 2.0)

        for si, text in enumerate(data.series_labels):
            style = _series_style(si)
            fill = _lighten(QColor(neutral), style["fill"])
            pen = QPen(QColor(self.t.foreground) if style["fill"] >= 1.0
                       else QColor(neutral))
            pen.setWidthF(style["width"])
            pen.setStyle(style["line"])
            p.setPen(pen)
            p.setBrush(fill)
            p.drawRoundedRect(QRectF(x, cy - s / 2, s, s),
                              LEGEND_RADIUS, LEGEND_RADIUS)
            p.setPen(QColor(neutral))
            p.drawText(QRectF(x + s + LEGEND_SWATCH_GAP, cy - fm.height() / 2,
                              widths[si] + 2, fm.height()),
                       Qt.AlignLeft | Qt.AlignVCenter, text)
            x += item_w[si] + LEGEND_ITEM_GAP


#: `CardTitle`'s size/weight. **NOT** `prototype/shadcn/figures.py`'s own 18 px
#: / 475 -- that pair compensates for the PROTOTYPE's sidebar sitting on an
#: 18 px root (later found to be a screenshot measurement, not the registry's
#: own 16 -- see the git history for that fix). This workspace already uses
#: the CORRECTED 16 px root (`toolbar.ROOT_PX`), so re-applying the
#: prototype's compensation on top of an already-correct root double-corrects
#: and renders the title visibly oversized -- which is exactly what this was.
#: `onboarding.stages.StageCard`, built after that fix at this same root, uses
#: 16 px at the registry's own `font-semibold` (600) with no compensation, and
#: that is what this reuses.
CARD_TITLE_PX = 16
CARD_TITLE_WGHT = 600
CARD_TITLE_WEIGHT = "semibold"

#: `p-6` on the Card, trimmed the same way `figures.PAD` is: the chart draws
#: its own margins, so the registry's full padding would inset it twice.
CARD_PAD = rem(0.875)
#: The one gap inside the card: under the header, above the chart -- AND,
#: reused, the drawer's own inset above its handle (`_Drawer`'s `top`).
CARD_GAP = rem(0.5)
#: What the chart cannot go below inside the card -- see `figures.CHART_MIN`.
CARD_CHART_MIN = rem(9)
#: And what an OPEN table cannot go below (its header row + one row of data)
#: -- see `figures.TABLE_MIN`'s own note on why a drag must not empty it.
TABLE_MIN = rem(4.75)
#: The grip's own band -- see `figures.HANDLE_H`/`GRIP_W`/`GRIP_H`.
HANDLE_H = rem(0.75)
GRIP_W, GRIP_H = rem(1.4), rem(0.35)


class TableHandle(QWidget):
    """`ResizableHandle withHandle`: a rule with a grip you can see, ported
    near-verbatim from `figures.py`. Visible, not hover-revealed -- an
    invisible splitter that only appears once the pointer is already on it is
    a feature nobody finds."""

    dragged = Signal(int)
    reset = Signal()

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.setFixedHeight(HANDLE_H)
        self.setCursor(Qt.SplitVCursor)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self._from: int | None = None

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        dpr = max(1.0, self.devicePixelRatioF())
        unit = 1.0 / dpr
        # THE PILL'S BOTTOM EDGE sits at this handle's own bottom -- flush
        # with the table (`_Drawer.place()` puts the table at
        # `top + HANDLE_H`, which is this widget's own bottom edge) -- and
        # the LINE passes through the pill's CENTRE, same as it always did.
        # Asked for directly: the pill sitting entirely above the line (the
        # previous fix, to dodge clipping the pill against this widget's own
        # bounds) read as "the pill floats above the bar" instead of "the
        # pill lies on it". Moving the whole pair down by half the pill's own
        # height keeps the pill inside this widget exactly (its bottom is
        # this widget's bottom, not past it) while putting the line back
        # through its middle.
        y = round((self.height() - GRIP_H / 2) * dpr) / dpr
        p.fillRect(QRectF(0, y, self.width(), unit), QColor(self.t.border))
        box = QRectF(self.width() / 2 - GRIP_W / 2, y - GRIP_H / 2, GRIP_W, GRIP_H)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(self.t.border))
        r = GRIP_H / 2.0
        p.drawRoundedRect(box, r, r)
        p.end()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._from = e.globalPosition().toPoint().y()

    def mouseMoveEvent(self, e):
        if self._from is None:
            return
        now = e.globalPosition().toPoint().y()
        self.dragged.emit(now - self._from)
        self._from = now

    def mouseReleaseEvent(self, e):
        self._from = None

    def mouseDoubleClickEvent(self, e):
        """Back to the default split -- see `BubbleCard.set_table_open`."""
        if e.button() == Qt.LeftButton:
            self.reset.emit()


class _Drawer(QWidget):
    """The handle and the table, shown by how much of them is not clipped --
    ported near-verbatim from `figures.py`, plus `replace_table` (see the
    module docstring: this chart's table is rebuilt every Run, the
    prototype's never was).

    Qt clips a child to its parent's rect, so a container whose height is
    animated reveals its contents from the TOP down without touching them.

    **The filter toolbar lives INSIDE `table`, not as a third child here.**
    It did sit here once, as its own row between the handle and the table --
    moved on request, below the table's own rows instead (see `TableBody`'s
    `toolbar` parameter), so `_Drawer` goes back to exactly the two children
    `figures.py`'s own version has.
    """

    def __init__(self, t, handle: TableHandle, table: TableBody, top: int,
                parent=None):
        super().__init__(parent)
        self.setStyleSheet("background:transparent;")
        self.top = top
        self.handle = handle
        self.table = table
        # `setParent` alone leaves a widget INVISIBLE -- Qt's own documented
        # behaviour, not a hidden default this prototype's own `_Drawer`
        # happened to dodge. `place()`/the clip do the rest: an explicit
        # `show()` here does not defeat the drawer's own reveal, because a
        # shown widget positioned OUTSIDE its parent's visible rect (height 0
        # at construction) still paints nothing until that rect grows.
        handle.setParent(self)
        handle.show()
        table.setParent(self)
        table.show()
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.setFixedHeight(0)

    def content_h(self) -> int:
        """The drawer fully open: the gap, the handle, and the table -- whose
        own `natural_height`/fixed height already accounts for the filter
        toolbar sitting inside it, see `TableBody.natural_height`."""
        return self.top + HANDLE_H + self.table.height()

    def place(self) -> None:
        """Children at their FULL size, whatever the drawer's height is."""
        w = self.width()
        self.handle.setGeometry(0, self.top, w, HANDLE_H)
        self.table.setGeometry(0, self.top + HANDLE_H, w, self.table.height())

    def resizeEvent(self, e):
        """Lay the contents out -- unless the drawer is shut (see
        `figures._Drawer`'s own note: a closed drawer is clipped to nothing,
        and re-laying out an invisible table on every resize frame is most of
        what made a page resize stall)."""
        super().resizeEvent(e)
        if self.height() > 0:
            self.place()

    def replace_table(self, table: TableBody) -> None:
        """Swap in a freshly-built table for new Run data. The OLD one holds
        the PREVIOUS Run's rows and must be discarded, not kept beside the
        new one -- there is only ever one table for one figure.

        WIDTHED HERE, BEFORE ANYTHING MEASURES IT. A brand-new `TableBody`
        has never been through Qt's (deferred) layout pass, so right after
        construction its caption is still wrapped to whatever width it was
        born with, not the width it will actually get from `place()` --
        measured at 629 px against a true 998, wrapping one extra line and
        making `natural_height()` read 259 where the settled answer is 242.
        `BubbleCard._apply()` calls `natural_height()` in the SAME call stack
        as this, before any event loop tick could fix it up, and then FREEZES
        that wrong number into the table's fixed height -- which is what
        read as "a scrollbar for content that already fits": the table was
        sized against a caption wrap that was never going to be real.
        `self.width()` is the one width this drawer's table will ever be
        placed at (`place()` uses the exact same value), so handing it over
        NOW makes every measurement taken after this line -- including the
        very next line the caller runs -- see the true, final width.
        """
        old = self.table
        table.setParent(self)
        table.resize(self.width(), table.height())
        table.show()
        self.table = table
        discard(old)


class _ResetLink(QWidget):
    """`isFiltered && <Button variant="ghost">Reset<X/></Button>`: clears
    every `FacetedFilter` in a `_TableToolbar` at once, shown only while at
    least one of them has something checked."""

    clicked = Signal()

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.setCursor(Qt.PointingHandCursor)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(rem(0.5), 0, rem(0.5), 0)
        lay.setSpacing(rem(0.25))
        lay.addWidget(label("Reset", ty.ui("sm", px=text_px(0.875)), t.foreground))
        lay.addWidget(icon("x", rem(0.875), t.muted_foreground, stroke=2.0))
        self.setFixedHeight(rem(2))

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self.rect().contains(e.position().toPoint()):
            self.clicked.emit()


class _TableToolbar(QWidget):
    """shadcn's `DataTableToolbar`, minus the free-text title search (the
    table has no free-text column to search): the Condition and Series
    `FacetedFilter`s, plus a `_ResetLink` once either has a selection.

    **Owned by `BubbleCard` directly, never rebuilt.** The table's own ROWS
    are rebuilt on every Run and every filter change
    (`BubbleCard._rebuild_table`), but this toolbar is not -- if it were,
    checking a box would destroy the very control the user just clicked,
    which is what a naive "just rebuild everything" version of this did
    first. `set_options` re-points the existing `FacetedFilter`s at a new
    Run's distinct values instead.
    """

    changed = Signal()

    HEIGHT = rem(2.5)

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.setStyleSheet("background:transparent;")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(CARD_PAD, rem(0.25), CARD_PAD, rem(0.25))
        lay.setSpacing(rem(0.5))
        self.condition = FacetedFilter(t, "Condition")
        self.condition.changed.connect(self._on_changed)
        lay.addWidget(self.condition)
        self.series = FacetedFilter(t, "Series")
        self.series.changed.connect(self._on_changed)
        lay.addWidget(self.series)
        lay.addStretch(1)
        self.reset = _ResetLink(t)
        self.reset.clicked.connect(self._clear_all)
        self.reset.hide()
        lay.addWidget(self.reset)

    def set_options(self, condition_options: list, series_options: list) -> None:
        """(Re)point at a fresh Run's distinct values -- see
        `FacetedFilter.set_options`: this also clears whatever was checked."""
        self.condition.set_options(condition_options)
        self.series.set_options(series_options)
        self._sync_reset()

    def selections(self) -> tuple:
        """`(condition_selected, series_selected)`, each a possibly-empty
        set -- see `table_rows`'s own `condition_filter`/`series_filter`."""
        return self.condition.selected(), self.series.selected()

    def height_used(self) -> int:
        """0 when NEITHER filter has more than one distinct value to offer
        -- see `FacetedFilter.set_options` -- so a single-condition,
        non-overlay Run reserves no row for a toolbar with nothing in it.

        Reads `has_filter`, not `isVisible()` -- see that property's own
        docstring for why: this container's OWN visibility is set FROM this
        method (in `TableBody`), so checking ancestor-gated visibility here
        would be circular the first time the container is hidden."""
        if self.condition.has_filter or self.series.has_filter:
            return self.HEIGHT
        return 0

    def _on_changed(self) -> None:
        self._sync_reset()
        self.changed.emit()

    def _sync_reset(self) -> None:
        active = bool(self.condition.selected()) or bool(self.series.selected())
        self.reset.setVisible(active)

    def _clear_all(self) -> None:
        self.condition.clear()
        self.series.clear()
        self._sync_reset()
        self.changed.emit()


class _DimOverlay(QWidget):
    """A translucent scrim, covering whatever it is raised over -- the
    "still current, hold on" look for `BubbleCard.set_running`, painted
    rather than a `QGraphicsOpacityEffect`.

    **NOT a graphics effect, and that is the fix for a real bug, not a
    style choice.** The first version dimmed via `QGraphicsOpacityEffect`
    plus `setEnabled(False)`/`True` on `BubbleCard` itself. Reported
    directly: after a re-run finished, hovering a bubble showed its
    tooltip (so `BubbleChart`'s OWN hover tracking was correct) but the
    cursor stayed a plain arrow -- and switching to another application
    and back (Alt+Tab) made the correct hand cursor appear immediately,
    with no mouse movement at all in between. That is the signature of a
    PLATFORM cursor cache, not a Qt-property bug: macOS invalidates a
    window's cursor rects on window activation, which is exactly what
    Alt+Tab forces and what a `QGraphicsEffect` attach/detach or an
    `setEnabled` toggle on the widget UNDER the cursor apparently does
    not reliably do on this backend. A `QTimer.singleShot(0, ...)` retry
    (tried first) did not fix it either -- the widget's own `cursor()`
    property was already correct even then, which is what let that
    attempt pass a headless, property-level check and still not fix what
    was on screen.

    The robust fix is to never touch `BubbleChart`'s own enabled state or
    graphics pipeline at all: this scrim sits ON TOP of the whole card as
    a plain sibling widget, blocking input by ordinary Z-order (nothing
    marks it `WA_TransparentForMouseEvents`) while the chart underneath
    keeps its own cursor/enabled state completely untouched -- there is
    nothing left to desync. Shown/raised or hidden, never destroyed, so
    `BubbleCard.set_running` is a plain, cheap `show()`/`hide()`.
    """

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(Qt.NoPen)
        fill = QColor(self.t.background)
        fill.setAlphaF(0.55)
        p.setBrush(fill)
        p.drawRoundedRect(QRectF(self.rect()), self.t.radius_xl, self.t.radius_xl)
        p.end()


class BubbleCard(Card):
    """One figure, in a card -- `prototype/shadcn/figures.FigureCard`'s shape,
    IN FULL: a header (title left, a table toggle right), the chart, then --
    on request -- the SAME data the chart drew, as a table. See the module
    docstring for what is and is not ported.

    The table is CLOSED by default. Most of the time the chart is the
    message; the numbers are consulted to check something the chart has
    already raised -- `figures.FigureCard`'s own reasoning, unchanged here.
    """

    #: What the table gets when first opened, as a share of the card's
    #: inside. Chart-dominant, because that is the thing being read.
    DEFAULT_SHARE = 0.38

    #: The header's Export button: the window opens `export_dialog.ExportDialog` with
    #: this Run's data (it alone holds the run's settings and provenance).
    export_requested = Signal()
    #: The data table's own Export: the rows the table shows, as a table figure.
    table_export_requested = Signal()
    #: The view moved between the bubbles and v2's crossed plot (`VIEWS`).
    view_changed = Signal(str)
    #: The card's two views of one Run (17 Oct 2026): the bubbles, and v2's crossed plot (fraction against D, bars
    #: both ways; `bubble_view.crossed_payload`).
    VIEWS = ("bubbles", "crossed")

    def __init__(self, t, parent=None, *, title: str = "SQD populations", crossed: bool = False):
        super().__init__(t, parent=parent)
        self.t = t
        self._table_h: int | None = None
        #: See `_DimOverlay`'s own docstring for why this exists instead
        #: of a `QGraphicsOpacityEffect` on `self`. Built once, hidden;
        #: `set_running` shows/raises or hides it, and `resizeEvent` keeps
        #: it covering the card while it is up.
        self._dim = _DimOverlay(t, self)
        self._dim.hide()

        # NO horizontal padding on the card's own layout: the header, the
        # chart and the table carry their own, so the DIVIDER can run edge to
        # edge -- see `figures.FigureCard`'s own comment on why.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, CARD_PAD, 0, CARD_PAD)
        outer.setSpacing(0)

        # -- header: title left, meta + toggle right --
        head = QWidget()
        head.setStyleSheet("background:transparent;")
        hl = QHBoxLayout(head)
        hl.setContentsMargins(CARD_PAD, 0, CARD_PAD, 0)
        hl.setSpacing(rem(0.5))
        hl.addWidget(label(title,
                           ty.ui("base", CARD_TITLE_WEIGHT, px=CARD_TITLE_PX,
                                 wght=CARD_TITLE_WGHT),
                           t.foreground))
        self.view_choice = None
        if crossed:
            from ..widgets.controls import Segmented

            self.view_choice = Segmented(t, ["Bubbles", "Fraction \u00d7 D"], current=0, size="sm")
            self.view_choice.changed.connect(self._on_view)
            hl.addSpacing(rem(0.5))
            hl.addWidget(self.view_choice)
        hl.addStretch(1)
        # THE AXES (30 Oct 2026, asked: "the axes button is cool, we can use it in other plots like sqd for y axis
        # etc?"): the D axis's range typed, per view (the bubbles' y, the Fraction x D plot's x and y)
        self.axes_button = ToolButton(t, "axes", "ruler", size=rem(1.75), rest=t.card)
        self.axes_button.clicked.connect(lambda _k: self.open_axes())
        self._axes_tip = tip.attach(t, self.axes_button, "Axes: type where an axis starts and ends (Auto: fitted)",
                                    side="bottom", top_level=True)
        hl.addWidget(self.axes_button)
        #: Typed ranges per view, kept through a new Run until Auto (`axes_view`).
        self._typed: dict = {"bubbles": {}, "crossed": {}}
        self._axes_kind = "appd_bubbles" if title.lower().startswith("apparent") else "sqd_bubbles"
        self._popover = None
        self.export_button = ToolButton(t, "export", "download", size=rem(1.75), rest=t.card)
        self.export_button.clicked.connect(lambda _k: self.export_requested.emit())
        self._export_tip = tip.attach(t, self.export_button,
                                      "Export figure (PDF, SVG, PNG, TIFF)",
                                      side="bottom", top_level=True)
        hl.addWidget(self.export_button)
        self.table_export_button = ToolButton(t, "export-table", "file-text", size=rem(1.75),
                                              rest=t.card)
        self.table_export_button.clicked.connect(
            lambda _k: self.table_export_requested.emit())
        self._table_tip = tip.attach(t, self.table_export_button, "Export the table (PDF, SVG, PNG, TIFF), or add it to the Composer",
                                     side="bottom", top_level=True)
        hl.addWidget(self.table_export_button)
        # COPY THE TABLE, for Excel (asked 15 Oct 2026), at full precision (`bubble_view.copy_rows`)
        self.copy_button = ToolButton(t, "copy-table", "copy", size=rem(1.75), rest=t.card)
        self.copy_button.clicked.connect(lambda _k: self.table.copy_to_clipboard())
        self._copy_tip = tip.attach(t, self.copy_button, "Copy the table (paste into Excel; a spread gets its own column)",
                                    side="bottom", top_level=True)
        hl.addWidget(self.copy_button)
        self.toggle = ToolButton(t, "table", "table", size=rem(1.75), rest=t.card)
        self.toggle.clicked.connect(lambda _k: self.set_table_open(not self.open))
        self._tip = tip.attach(t, self.toggle, "Show data table",
                               side="bottom", top_level=True)
        hl.addWidget(self.toggle)
        outer.addWidget(head)
        outer.addSpacing(CARD_GAP)

        self.chart = BubbleChart(t)
        self.chart.setContentsMargins(CARD_PAD, 0, CARD_PAD, 0)
        self.chart.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.chart.setMinimumHeight(CARD_CHART_MIN)
        self.chart.hovered.connect(self._on_chart_hover)
        self.crossed = None
        if crossed:
            from .crossed_chart import CrossedChart

            from .condition_key import ConditionKey

            self.crossed = CrossedChart(t)
            self.crossed.setContentsMargins(CARD_PAD, 0, CARD_PAD, 0)
            self.crossed.setMinimumHeight(CARD_CHART_MIN)
            #: An eye per condition over the crossed plot (17 Oct 2026); the export leaves a hidden one out.
            self.crossed_key = ConditionKey(t)
            self.crossed_key.hidden_changed.connect(self.crossed.set_hidden)
            crossed_page = QWidget()
            crossed_page.setStyleSheet("background:transparent;")
            crossed_lay = QVBoxLayout(crossed_page)
            crossed_lay.setContentsMargins(CARD_PAD, 0, CARD_PAD, 0)
            crossed_lay.setSpacing(rem(0.25))
            crossed_lay.addWidget(self.crossed_key)
            crossed_lay.addWidget(self.crossed, 1)
            self.crossed.setContentsMargins(0, 0, 0, 0)
            self._views = QStackedWidget()
            self._views.setStyleSheet("background:transparent;")
            self._views.addWidget(self.chart)
            self._views.addWidget(crossed_page)
            outer.addWidget(self._views, 1)
        else:
            outer.addWidget(self.chart, 1)

        self.handle = TableHandle(t)
        self.handle.dragged.connect(self._drag)
        self.handle.reset.connect(lambda: self.set_table_open(True, share=None))
        # Owned here, never rebuilt -- see `_TableToolbar`'s own docstring on
        # why it is not part of what `_rebuild_table` swaps out. PLACED into
        # each `TableBody` below (its `toolbar=` parameter) rather than kept
        # as a sibling of the table -- `TableBody` reparents it into its own
        # layout on construction, which is what lets the SAME toolbar survive
        # a table swap: Qt detaches a widget from its old parent the moment
        # it is added to a new layout, so nothing here has to do that by hand.
        self.toolbar_filters = _TableToolbar(t)
        self.toolbar_filters.changed.connect(self._on_filter_changed)
        # A placeholder table before the first Run -- `table_rows` is never
        # called with no data because there IS no `BubblePlotData` yet.
        self.table = TableBody(t, ["Condition"], [],
                               "Run to populate this table.", scrollable=True,
                               toolbar=self.toolbar_filters)
        self.drawer = _Drawer(t, self.handle, self.table, CARD_GAP)
        outer.addWidget(self.drawer)

        self.open = False
        self._anim = QVariantAnimation(self)
        # LINEAR, as the sidebar's collapse is -- see `figures.FigureCard`.
        self._anim.setEasingCurve(QEasingCurve.Linear)
        self._anim.valueChanged.connect(
            lambda v: self.drawer.setFixedHeight(round(v)))

    # -- message / data ------------------------------------------------------ #
    def set_message(self, text: str) -> None:
        self.chart.set_message(text)

    def _on_view(self, index: int) -> None:
        self._views.setCurrentIndex(index)
        self.view_changed.emit(self.VIEWS[index])

    def current_view(self) -> str:
        """``"bubbles"`` or ``"crossed"`` (what Export draws)."""
        return self.VIEWS[self.view_choice.current()] if self.view_choice is not None else "bubbles"

    # -- typed ranges (30 Oct 2026) ------------------------------------------------------------------------------ #
    def axes_rows(self, view: str | None = None) -> list:
        """The Axes popover's rows for ``view`` (the one shown): the bubbles' D (y, log); the crossed plot's
        fraction (x) and D (y, log)."""
        view = view or self.current_view()
        return axis_rows("sqd_crossed" if view == "crossed" else self._axes_kind)

    def typed_axes(self, view: str | None = None) -> dict:
        """The ranges typed for ``view`` (``{key: (from, to) or None}``; empty: automatic)."""
        return dict(self._typed.get(view or self.current_view()) or {})

    def set_typed_axes(self, view: str, axes) -> None:
        """Ranges for ``view`` (`axes_view.check`, refused by name), drawn now; all None: automatic again."""
        axes = check(self.axes_rows(view), axes or {})
        self._typed[view] = {} if axes_empty(axes) else axes
        if view == "crossed" and self.crossed is not None:
            self.crossed.axes = dict(self._typed[view])
            self.crossed.update()
        elif view == "bubbles":
            self.chart.axes = dict(self._typed[view])
            self.chart.update()

    def axes_state(self) -> dict:
        """Both views' typed ranges (a session keeps them)."""
        return {view: dict(axes) for view, axes in self._typed.items() if axes}

    def set_axes_state(self, state: dict) -> None:
        for view, axes in (state or {}).items():
            if view in self._typed:
                self.set_typed_axes(view, axes)

    def open_axes(self) -> None:
        """The Axes popover for the view shown, under the header's button."""
        if self._popover is None:
            self._popover = AxesPopover(self.t, self)
            self._popover.applied.connect(lambda axes: self.set_typed_axes(self._popover_view, axes))
        self._popover_view = self.current_view()
        self._popover.show_for(self.axes_rows(self._popover_view), self.typed_axes(self._popover_view),
                               self.axes_button)

    def set_data(self, data: BubblePlotData, colours: dict) -> None:
        self.chart.set_data(data, colours)
        if self.crossed is not None:
            crossed = crossed_payload(data)
            self.crossed.set_payload(crossed, colours)
            self.crossed_key.set_entries([(c, n, colours.get(c, "#2b6cb0"))
                                          for c, n in crossed.get("legend_conditions") or []])
        # Re-pointed at THIS Run's distinct values before the table is
        # built, so the very first `_rebuild_table` below already reads the
        # fresh (empty, per `FacetedFilter.set_options`) selection rather
        # than one carried over from a condition set that may not even be
        # in this Run.
        self.toolbar_filters.set_options(condition_facets(data),
                                         series_facets(data))
        self._rebuild_table(data)
        # A fresh Run's table gets a fresh default share rather than keeping
        # a pixel height sized for the LAST Run's row count.
        self._table_h = None
        if self.open:
            self._apply()
            self.drawer.setFixedHeight(self.drawer.content_h())

    def _rebuild_table(self, data: BubblePlotData) -> None:
        """Build the table for the CURRENT filter selection and swap it in.
        Called for a fresh Run's data (`set_data`) and again whenever a
        filter changes (`_on_filter_changed`) -- the toolbar itself is never
        rebuilt, only this."""
        condition_sel, series_sel = self.toolbar_filters.selections()
        columns, rows, keys, caption = table_rows(
            data, condition_filter=condition_sel, series_filter=series_sel)
        new_table = TableBody(self.t, columns, rows, caption, scrollable=True,
                              toolbar=self.toolbar_filters, row_keys=keys)
        new_table.copy_source = lambda: copy_rows(data, condition_filter=condition_sel,
                                                  series_filter=series_sel)
        self.drawer.replace_table(new_table)
        self.table = new_table

    def _on_chart_hover(self, pt: BubblePoint | None) -> None:
        """Scroll to and highlight the hovered bubble's own row -- ONLY when
        the table is open and that row is not currently filtered out. See
        the module docstring for why this never reaches across an active
        filter or forces the drawer open to show one: the tooltip
        (`BubbleChart`'s own job) already answers the hover on its own,
        with the table's sync as a bonus when it happens to be available.
        """
        if not self.open:
            return
        if pt is None:
            self.table.clear_highlight()
            return
        data = self.chart._data
        if data is None:                    # a stale signal from a torn-down Run
            return
        self.table.highlight_key(point_key(data, pt))

    def _on_filter_changed(self) -> None:
        data = self.chart._data
        if data is None:                    # a filter fired before any Run
            return
        self._rebuild_table(data)
        # THE CARD'S OWN FOOTPRINT DOES NOT CHANGE HERE -- a filter narrows
        # or widens which ROWS sit inside the table, not the table's own
        # height. `_table_h` was fixed once already, the moment the drawer
        # opened (`_apply`, from a drag or the default share); reapplying
        # that SAME number to the freshly rebuilt table is what keeps the
        # panel still while you click a checkbox, rather than visibly
        # resizing to hug whatever the new row count happens to need.
        #
        # SUPERSEDED, and left here because the trap it names is real and
        # worth not re-finding: an earlier version reset `_table_h = None`
        # and called `_apply()` here, on the reasoning that without it
        # `_apply`'s `min(want, natural_height())` clamp could only ever
        # SHRINK the table as filtering narrowed the rows, with nothing to
        # grow it back once a filter relaxed. Asked directly, that
        # "shrink-to-fit" behaviour was itself the problem, not the bug in
        # it: a table that resizes on every filter click is worse than one
        # that stays put and shows blank space below a short filtered set.
        # `_table_h` is therefore never reset from here at all -- only a
        # fresh Run (`set_data`) gets a fresh default share.
        if self.open:
            self.table.setFixedHeight(self._table_h)
            self.drawer.setFixedHeight(self.drawer.content_h())
            # EXPLICITLY, not left to the `setFixedHeight` above. The whole
            # point of this method is that `content_h()` comes out the SAME
            # as before (the table's own height is unchanged), so that call
            # resizes nothing and fires no `resizeEvent` -- `_Drawer.place()`
            # then never runs. `replace_table` (inside `_rebuild_table`,
            # above) only RESIZES the freshly swapped-in table; `setParent`
            # leaves a reparented widget's POSITION at (0, 0) in its new
            # parent, which sits ON TOP of the handle rather than below it.
            # Without this line the handle and its pill render invisible --
            # not removed, just painted over by the table that now shares
            # its top-left corner. `place()` puts both back at their real
            # geometry regardless of whether a resize happened to do it.
            self.drawer.place()

    def set_colours(self, colours: dict) -> None:
        self.chart.set_colours(colours)
        if self.crossed is not None:
            self.crossed.set_colours(colours)
            pl = self.crossed.payload() or {}
            self.crossed_key.set_entries([(c, n, colours.get(c, "#2b6cb0"))
                                          for c, n in pl.get("legend_conditions") or []])

    def crossed_hidden(self) -> set:
        """The conditions the crossed plot's eyes hide."""
        return self.crossed_key.hidden() if self.crossed is not None else set()

    def conditions(self) -> tuple:
        return self.chart.conditions()

    def set_running(self, on: bool) -> None:
        """A RE-run in flight, with a result already on screen -- dim this
        card in place rather than blanking it or bouncing back to
        `SectionPlaceholder`. Keeps the "what did I have a second ago"
        anchor a full collapse-and-rebuild would throw away, and is cheap
        to repeat on every iteration of a tune-and-rerun workflow, since
        `SectionPlaceholder` is a ONE-WAY door -- see its own module
        docstring.

        A raised `_DimOverlay`, not `setEnabled` + a `QGraphicsOpacityEffect`
        on `self` -- see that class's own docstring for the real bug that
        combination caused (a stuck native arrow cursor after a re-run,
        fixed only by Alt+Tabbing away and back). Blocking input by
        covering the card is the SAME "stale numbers should not be
        filterable or hoverable" intent `widgets.dialogs.OverlayDialog`
        already applies to the whole workspace while IT is open, achieved
        a different way. No animation either direction -- that overlay's
        own content gets none for the same disable either, and this is
        the far more frequent action, not the one moment worth ceremony
        for.
        """
        if on:
            self._dim.setGeometry(self.rect())
            self._dim.show()
            self._dim.raise_()
        else:
            self._dim.hide()

    # -- the table toggle, ported verbatim from `figures.FigureCard` -------- #
    def set_table_open(self, on: bool, share: float | None = -1.0, *,
                       animate: bool = True) -> None:
        """Show or hide this figure's own data.

        ``share`` of ``None`` resets to the default split; the sentinel keeps
        whatever the user had dragged to.
        """
        self.open = on
        self.toggle.set_active(on)
        self._tip.text = "Hide data table" if on else "Show data table"
        if on:
            if share is None or self._table_h is None:
                self._table_h = None
            # The table is sized ONCE, here, and then only uncovered.
            self._apply()
        self._slide(self.drawer.content_h() if on else 0, animate)

    def _slide(self, target: int, animate: bool) -> None:
        now = self.drawer.height()
        self._anim.stop()
        if not animate or target == now or not self.isVisible():
            self.drawer.setFixedHeight(target)
            return
        self._anim.setDuration(slide_ms(target - now))
        # Signals blocked while the range is set -- see `figures.FigureCard`'s
        # own note: `stop()` leaves `currentTime` at the end of the previous
        # run, so `setEndValue` alone would emit a snap-to-finished frame.
        blocked = self._anim.blockSignals(True)
        self._anim.setStartValue(float(now))
        self._anim.setEndValue(float(target))
        self._anim.blockSignals(blocked)
        self._anim.start()

    def _inside(self) -> int:
        """The height the chart and table share -- the card, less its own
        padding, the header, and the handle. The filter toolbar is no longer
        a term here: it is drawn INSIDE the table now, so its footprint (0
        when it has nothing worth filtering -- see `_TableToolbar.
        height_used`) is already part of `self.table.natural_height()`,
        which `_apply` reads separately. This is the BUDGET, and it does not
        change when the table opens; that is the whole rule."""
        used = CARD_PAD * 2 + CARD_GAP * 2 + HANDLE_H
        used += self.layout().itemAt(0).widget().height()
        return max(0, self.contentsRect().height() - used)

    def _apply(self) -> None:
        room = self._inside()
        want = (self._table_h if self._table_h is not None
                else round(room * self.DEFAULT_SHARE))
        # CLAMPED BY THE CHART, not by the table -- a stored height is a cap
        # on what the table may take, never a demand the chart has to meet.
        want = min(want, self.table.natural_height())
        want = min(want, max(0, room - CARD_CHART_MIN))
        want = max(want, min(TABLE_MIN, room - CARD_CHART_MIN))
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
        # Not while the drawer is mid-slide: the animation owns its height.
        if self.open and self._anim.state() != QVariantAnimation.Running:
            self._apply()
            self.drawer.setFixedHeight(self.drawer.content_h())
        # Kept covering the card through a resize while a re-run is up --
        # a resize this rare mid-compute is not worth its own animation,
        # just correctness.
        self._dim.setGeometry(self.rect())


# == the population detail page (Populations -> a condition) =============== #
#
# The in-app replacement for the legacy `browser.BubbleInspector` -- a
# floating `Qt.Tool` window titled "Replicate fit", raised by
# `_show_bubble_inspector` on a bubble click. Here the SAME payload
# (`comparison_controller.replicate_fit_payload`, precomputed for every
# condition during `_run_pressed` -- see `compute_bubble_view`'s own
# docstring) drives a PAGE instead: `WorkspaceWindow` swaps a
# `QStackedWidget` from the populations chart onto a `ReplicateDetail` and
# extends the breadcrumb with the condition's own name, so leaving the page
# is a crumb click rather than closing a window.

class ReplicateChart(QWidget):
    """One condition's per-replicate simultaneous fit, painted with the same
    axis convention as `BubbleChart` -- x categorical, y = D on a log scale,
    bubble AREA carrying the fraction -- but with the column now a REPLICATE
    rather than a condition.

    Colour is the CONDITION's, from the Conditions tool (`set_colour`), one
    SHADE PER REPLICATE by the app's one day-shade rule (`figstyle.day_shade`: the
    first replicate darkest, the last lightest, the condition colour between) -- the
    same shades Apparent Diffusion and every exported figure use. Replicates are one
    condition's repeats, so they take one hue at stepped lightness -- a new hue would
    read as another condition. Populations are told apart by D on the y axis, as in
    `BubbleChart`; every number sits under its bubble in the page's ink
    (`paint_bubble`). Without a colour (a bare chart) it falls back to one
    `hcl.series` hue per population.
    """

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.setStyleSheet("background:transparent;")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(CARD_CHART_MIN)
        # `tabular=True` -- see `BubbleChart.__init__`'s own note; the same
        # two fonts here draw the same two kinds of digit (y-axis D values,
        # the per-bubble percentage).
        self._tick_font = ty.ui("xs", px=TICK_PX, tabular=True)
        self._label_font = ty.ui("xs", px=TICK_PX, weight="semibold", tabular=True)
        self._payload: dict | None = None
        #: The condition's own colour (the Conditions tool's), or None.
        self._colour: str | None = None

    def set_payload(self, payload: dict | None) -> None:
        self._payload = payload
        self.update()

    def set_colour(self, colour: str | None) -> None:
        """The condition's colour, as the Conditions tool sets it -- every
        population is drawn in it, exactly as `BubbleChart` draws this
        condition's column (populations are told apart by D, on the y axis)."""
        self._colour = colour
        self.update()

    def _plot_rect(self) -> QRectF:
        # No legend row here (see `BubbleChart._plot_rect`) -- population
        # colour is named beside its own fraction label on every bubble
        # (below), so a separate swatch row would repeat that.
        width = max(1.0, self.width() - PAD_L - PAD_R)
        fm = QFontMetricsF(self._tick_font)
        names = [f"Replicate {r[0]}" for r in self._rows(self._payload)] if (
            self._payload and self._payload.get("replicates")) else []
        room = (label_line_count(names, fm, width) - 1) * fm.height()
        return QRectF(PAD_L, PAD_T, width, max(1.0, self.height() - PAD_T - PAD_B - room))

    @staticmethod
    def _rows(payload: dict) -> list[tuple[str, list[float], list[float]]]:
        """`(label, D_values, fractions)` per replicate, one population per
        entry -- resolved once so `paintEvent` stays pure drawing. `D_values`
        is that replicate's own fitted D under `scope == "replicate"`, the
        one shared D under every other scope (`condition_replicate_fit`'s
        two payload shapes, read directly rather than re-derived)."""
        rows = []
        for rep in payload["replicates"]:
            d_values = rep.get("D") or payload.get("shared_D") or []
            rows.append((str(rep["replicate"]), list(d_values), list(rep["fractions"])))
        return rows

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        payload = self._payload
        if not payload or not payload.get("replicates"):
            p.setFont(ty.ui("sm"))
            p.setPen(QColor(self.t.muted_foreground))
            p.drawText(self.rect(), Qt.AlignCenter | Qt.TextWordWrap,
                       "No replicate data.")
            p.end()
            return

        rows = self._rows(payload)
        # a D of 0, the immobile population of exact-zero jumps, at a "0" tick under the rest (`chart_text`)
        zero_d = chart_text.zero_d_stand_in(d for _, ds, _ in rows for d in ds)
        all_d = [chart_text.log_drawn(d, zero_d) for _, ds, _ in rows for d in ds
                 if d > 0 or (d == 0 and zero_d is not None)]
        if not all_d or max(all_d) <= 0:
            p.end()
            return
        box = self._plot_rect()
        lo, hi = chart_text.bubble_axis_range(min(all_d), max(all_d), box.height(),
                                              *bubble_room(self._label_font))
        yr_log = (math.log10(lo), math.log10(hi))
        limit = chart_text.log_tick_limit(box.height(), QFontMetricsF(self._tick_font).height())
        labelled = chart_text.zero_d_ticks(BubbleChart._log_ticks(lo, hi, limit), zero_d)
        yt = [v for v, _label in labelled]

        # -- grid, matching `BubbleChart._paint_bubbles` exactly -- #
        dpr = max(1.0, self.devicePixelRatioF())
        pen = QPen(QColor(mix(self.t.border, self.t.background, 0.5)))
        pen.setWidthF(1.0 / dpr)
        p.setPen(pen)
        for v in yt:
            y = BubbleChart._map_y_log(box, v, yr_log)
            p.drawLine(QPointF(box.left(), y), QPointF(box.right(), y))

        unit = 1.0 / dpr
        xline = round(box.left() * dpr) / dpr
        p.fillRect(QRectF(xline, box.top(), unit, box.height()), QColor(self.t.border))
        yline = round(box.bottom() * dpr) / dpr
        p.fillRect(QRectF(box.left(), yline, box.width(), unit), QColor(self.t.border))

        p.setFont(self._tick_font)
        p.setPen(QColor(self.t.muted_foreground))
        fm = QFontMetricsF(self._tick_font)
        for v, label in labelled:
            y = BubbleChart._map_y_log(box, v, yr_log)
            p.drawText(QRectF(0, y - fm.height() / 2, box.left() - rem(0.375),
                              fm.height()), Qt.AlignRight | Qt.AlignVCenter, label)
        BubbleChart._y_axis_label(p, box, chart_text.D_AXIS_LABEL, self._tick_font,
                                 self.t.muted_foreground)

        n = max(1, len(rows))
        xr = (-0.6, n - 0.4)
        p.setFont(self._tick_font)
        p.setPen(QColor(self.t.muted_foreground))
        # Capitalised ("Replicate 1"), matching `BubbleChart`'s own
        # condition labels -- asked for directly, once this read
        # "replicate 1" beside "Glucose"/"Malate" on the parent chart.
        draw_column_labels(p, fm, box, xr, [f"Replicate {rep[0]}" for rep in rows])

        n_pop = max((len(ds) for _, ds, _ in rows), default=1)
        colours = [QColor(c) for c in hcl.series(max(1, n_pop))]
        # One shade of the condition colour PER REPLICATE, by the app's one rule
        # (`figstyle.day_shade`), so the screen and the export agree.
        shades = ([figstyle.day_shade(self._colour, i, len(rows)) for i in range(len(rows))]
                  if self._colour else None)
        spots = []
        for i, (_label, ds, fractions) in enumerate(rows):
            x = BubbleChart._map_x(box, i, xr)
            for k, (d, f) in enumerate(zip(ds, fractions)):
                if d < 0 or (d == 0 and zero_d is None):
                    continue
                y = BubbleChart._map_y_log(box, chart_text.log_drawn(d, zero_d), yr_log)
                r = MIN_R + (MAX_R - MIN_R) * math.sqrt(max(f, 0.0))
                # `BubbleChart`'s own solid style, there being no overlay/series
                # concept on this page. The fill is the REPLICATE's shade of the
                # condition colour, or a population colour when none was handed over.
                fill = QColor(shades[i]) if shades else colours[k % len(colours)]
                paint_bubble(p, self.t, QPointF(x, y), r, fill, _SERIES_STYLES[0])
                spots.append((x, y, r, f"{100 * f:.0f}%"))
        paint_numbers(p, self.t, spots, self._label_font, box)
        p.end()


class ReplicateDetail(Card):
    """The Populations -> condition detail page: `ReplicateChart` above a
    caption line (the legacy inspector's shared-D header) above a
    `TableBody` -- `BubbleCard`'s own card shape (`Card`, header-then-
    content, `TableBody`'s edge-to-edge inset), minus the drawer/toggle:
    unlike `BubbleCard`, the table here is not opened on request, it is the
    whole other half of what this page exists to show, so it is always
    visible. No back button either -- the breadcrumb `WorkspaceWindow`
    extends on drill-in IS the back-navigation, the one control this page
    does not need to duplicate.

    Built as a `Card`, not a bare `QWidget`, so a click that drills in
    lands on the SAME rounded/bordered/shadowed surface the chart it came
    from sits in -- asked for directly, after the first version rendered
    this page flat against the workspace background while `BubbleCard`
    beside it carried the registry's own card chrome.
    """

    #: The header's Export button (`export_dialog.ExportDialog`, kind "sqd_replicates").
    export_requested = Signal()
    #: The replicate table's own Export, as a table figure.
    table_export_requested = Signal()

    def __init__(self, t, parent=None):
        super().__init__(t, parent=parent)
        self.t = t
        # NO horizontal padding on the outer layout, matching `BubbleCard`
        # exactly: the header and chart carry their own (`CARD_PAD`), and
        # the table's edge-to-edge inset (`TableBody`'s own convention, see
        # its docstring) needs the card's bare edge to sit on.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, CARD_PAD, 0, CARD_PAD)
        outer.setSpacing(CARD_GAP)

        head = QWidget()
        head.setStyleSheet("background:transparent;")
        head_lay = QHBoxLayout(head)
        head_lay.setContentsMargins(CARD_PAD, 0, CARD_PAD, 0)
        self._title = label("Replicate fit",
                            ty.ui("base", CARD_TITLE_WEIGHT, px=CARD_TITLE_PX,
                                  wght=CARD_TITLE_WGHT), t.foreground)
        head_lay.addWidget(self._title)
        head_lay.addStretch(1)
        # The same Export as the SQD card: every figure in the workspace has one.
        self.export_button = ToolButton(t, "export", "download", size=rem(1.75), rest=t.card)
        self.export_button.clicked.connect(lambda _k: self.export_requested.emit())
        self._export_tip = tip.attach(t, self.export_button,
                                      "Export figure (PDF, SVG, PNG, TIFF)",
                                      side="bottom", top_level=True)
        head_lay.addWidget(self.export_button)
        self.table_export_button = ToolButton(t, "export-table", "file-text", size=rem(1.75),
                                              rest=t.card)
        self.table_export_button.clicked.connect(
            lambda _k: self.table_export_requested.emit())
        self._table_tip = tip.attach(t, self.table_export_button, "Export the table (PDF, SVG, PNG, TIFF), or add it to the Composer",
                                     side="bottom", top_level=True)
        head_lay.addWidget(self.table_export_button)
        self.copy_button = ToolButton(t, "copy-table", "copy", size=rem(1.75), rest=t.card)
        self.copy_button.clicked.connect(lambda _k: self.table.copy_to_clipboard())
        self._copy_tip = tip.attach(t, self.copy_button, "Copy the table (paste into Excel; a spread gets its own column)",
                                    side="bottom", top_level=True)
        head_lay.addWidget(self.copy_button)
        outer.addWidget(head)

        self.chart = ReplicateChart(t)
        self.chart.setContentsMargins(CARD_PAD, 0, CARD_PAD, 0)
        outer.addWidget(self.chart, 1)

        self._caption = label("", ty.ui("sm"), t.muted_foreground)
        self._caption.setContentsMargins(CARD_PAD, 0, CARD_PAD, 0)
        outer.addWidget(self._caption)

        self.table = TableBody(t, ["Replicate", "Cells", "Tracks"], [],
                               "No replicate data.", scrollable=True)
        outer.addWidget(self.table)

    def set_payload(self, condition: str, payload: dict | None,
                    colour: str | None = None, pairs: dict | None = None) -> None:
        """``payload`` is `bubble_view.compute_bubble_view`'s replicate payload for
        ``condition``: the fit, ``{"error": ...}`` naming why the spine refused one,
        or None when none was made (a Run before this condition was fitted).
        ``colour`` is the condition's colour from the Conditions tool; ``pairs`` its replicates' pair tags from the
        Groups board (``{replicate number: tag}``), drawn as the table's Pair column."""
        self._title.setText(f"Replicate fit: {condition}")
        self.chart.set_colour(colour)
        #: What an export draws: this condition, its fit, its colour.
        self.condition, self.colour = condition, colour
        if payload is not None and "error" in payload:
            # A refused fit says WHY on the page the click opened, rather than
            # reading as the same blank as "not computed yet".
            self.chart.set_payload(None)
            self._caption.setText(f"No replicate fit for this condition: {payload['error']}")
            new_table = TableBody(self.t, ["Replicate", "Cells", "Tracks"], [],
                                  "No replicate data.", scrollable=True)
        elif payload is None:
            self.chart.set_payload(None)
            self._caption.setText("No replicate fit is available for this condition.")
            new_table = TableBody(self.t, ["Replicate", "Cells", "Tracks"], [],
                                  "No replicate data.", scrollable=True)
        else:
            self.chart.set_payload(payload)
            self._caption.setText(shared_d_caption(payload))
            columns, rows = replicate_table_rows(payload, pairs)
            footer = f"{replicate_footer(payload, len(rows))}." + (
                f" {PAIR_COLUMN_NOTE}." if shows_pairs(payload, pairs) else "")
            new_table = TableBody(self.t, columns, rows, footer, scrollable=True)
        # `QLayout.replaceWidget` only detaches the OLD widget from the
        # LAYOUT -- it stays a child of `self`, sitting at whatever geometry
        # it last had, and renders right there unless it is ALSO detached
        # from the widget tree. `_Drawer.replace_table` (above) already
        # names this exact trap; the fix is the same two lines.
        old = self.table
        self.layout().replaceWidget(old, new_table)
        new_table.show()
        self.table = new_table
        discard(old)
