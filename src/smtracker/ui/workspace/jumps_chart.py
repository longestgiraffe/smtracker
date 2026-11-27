"""The SQD section's Jump distances page: the jump histogram against the 1-, 2- and
3-population fits, and how well each fits (asked 13 Oct 2026: "jump distance histograms,
where u can compare how different pop fits compare" and "goodness of fit ... better than
v2").

Each condition is a card, ONE PER ROW: the histogram with the shown fits' expected counts
(v2's own line styles: dotted one, solid two, dashed three populations) and, one fit alone, its
components, BESIDE it the PEARSON residuals of the shown fits with a +-2 band over their ECDF
residuals, under them the fit-quality table (BIC, the largest ECDF gap, the
held-out likelihood) and one verdict line (`jumps_view.verdict_line`). What every card used
to repeat is said once, in the page's "How to read" hover card (`jumps_view.READING_NOTES`;
13 Oct 2026: "too much text and the 2x2 grid is not well supported"). No card is given a
minimum height: an explicit one REPLACES the layout's own in Qt, and 40 rem on content that
needed 919 px drew the three charts on top of each other. The
numbers and words are `smtracker.jumps_view`'s (headless, on the pool); nothing here imports
numpy, pandas or scipy (`tests/test_ui_thin_qt.py`). The axis titles are `chart_text`'s, the
exported figure's too. Each card's Export button asks the window through
`export_requested(key)`; the figure is `figure_export`'s "jumps_condition", drawn from
`export_entries(key)` -- the card's result and the fits its Show pills have on.

WHAT A CARD DRAWS (26 Oct 2026, asked: "if 1 or 2 or 3 or 4 pop selected -> u only see this fit. right next to it
"all" where all curves shown with different colors" and "the histogram color can be same color as the protein in
the bubble"): the bars in the condition's colour; one fit is drawn alone with its components, its Pearson and its
ECDF residuals; two or more each in its own colour and v2's line style (`jumps_view.FIT_COLOURS`), their Pearson
and ECDF residuals as lines in the same colours. The Signed jumps page reuses `JumpHistogram` WITHOUT a ``"show"``
and keeps its old drawing (every fit, grey bars), as its export does.

SHOW IS TOGGLES (30 Oct 2026, asked: "instead of 1 2 3 all, just make it clickable and the one u click appears, when
u click again it disappears ... then u click 3 again and 3 disappears and 1 and 2 still visible", and "make
residuals and ecdf according how many shown of course"): one pill per count fitted (`jumps_view.show_choices`, no
All), each on or off (`controls.Toggles`). A click asks `jumps_view.toggle_show` what is on now -- the last fit on
stays on -- and every chart of the card draws `jumps_view.drawn_fits` of that set (`_JumpAxes.drawn`). A card
starts with one fit on, BIC's, else two (`default_show`); a set picked by hand survives the full pass replacing the
quick one.

ALL CONDITIONS (asked the same day: "a way to print all currently loaded jd plots histogram into one screen and
export together"): the page's view switch puts every condition's histogram in a grid (`JumpsOverview`), each drawn as
its card shows it, on one x range; "Show for every condition" (the same toggles) sets every card at once -- its
pills lit when the cards agree, none lit when they differ, so a click there starts from that one fit -- and Export
all asks the window through `export_all_requested` for `figure_export`'s "jumps_overview" (`export_all_entries`).

LINES, COMPONENTS AND THE OVERVIEW'S LAYOUT (30 Oct 2026, asked: "i think we can make the lines a bit bigger and
better visible", "its a shame that the component lines are not visible anymore when i do 1 and 2 pop fit together.
maybe we can make this optional", and in the all conditions window "show in one plot (like it is now). or show one
plot for each fit/population ... user can choose that conditions or populations go from up to down or left to
right"): every line is thicker (`_W_ONE` and the widths beside it; the old ones are said there). Each card's Show
row has a Components box (on to start; `jumps_view.components_drawn` says whose are drawn: one fit's in shades of the
condition's colour, as before; several fits' thin and dashed in each fit's colour). The All conditions head has the
same box for every card, and "One plot per condition | One plot per fit" with, per fit, "Conditions down |
Conditions across": `jumps_view.overview_tiles` places the tiles, every tile on one x range and, per fit, each
condition's tiles on one y range (`jumps_view.histogram_top`). Export all draws the same (`overview_settings`).

THE KEY AND THE TILES' SIZE (30 Oct 2026, asked: "do u think we can make these cards more interactable? like dragging
the legend when it blocks the view, or increasing size of these cards in a way if its too small etc?"). Every
histogram's key starts in the corner where it covers least (`jumps_view.legend_corner`), can be DRAGGED (an open
hand over it, a closed one while it moves; it stays inside the plot) and a double-click on it puts it back
(`JumpHistogram.legend_moved`). The section keeps each chart's place while the window lives (`legend_places`: each
card's, each tile's; saved with a session), a card's export puts its key there (the entry's ``"legend_at"``). The
All conditions view has a Size switch, S / M / L (`TILE_SIZES`), and a double-click on a tile (not on its key)
spans it across the grid and back (`JumpsOverview.toggle_wide`). The Signed jumps page's histogram, this class too,
gets the corner and the drag on screen; its export keeps its key above the plot.

THE Y AXIS, THE RANGES, ONE Y SCALE (30 Oct 2026, asked: "lets make all optional with textbook (check first) as
standard. rescale whenever u change setting. make y and x axis changeable. make a button scale y axis for all", and of
the key's "its components": "1 and 2 and 3 etc populations as it is. its components -> what was used in old
smtracker?"). Each card has a Y axis switch -- Probability (per bin, SMTracker 2's and Spot-On's, the standard),
Counts, Density (`jumps_view.Y_MODES`; the bars and the curves scaled by one divisor, `jumps_view.in_units`) -- and an
Axes button on its head (`_AxesPopover`: x and y from / to, Auto). The All conditions head sets every card's unit, has
the Axes for every tile, and "Same y scale" (also over the cards: `JumpsSection.toggle_same_y`) puts every chart of
one unit on the largest automatic range among them (`jumps_view.shared_tops`). Automatic axes are fitted to what is
drawn at every paint, so any change of fits, components or size rescales them; a typed range stays until Auto, and a
new unit drops a typed y range. The key names the fits only, as SMTracker 2's did; the components are explained in
How to read. The Pearson strip stays in counts and the ECDF strip is untouched; a typed x range moves them with the
histogram.

THE RESIDUAL PLOTS IN ALL CONDITIONS (12 Nov 2026, reported: "i noticed that u can only see pearson and reisduals with
export function for sqd jump histogram. not in the software itself", then "Yes , why not add Pearson and ecdf"). The
All conditions head has a box for each residual plot, Pearson and ECDF, off to start (two plots under every tile make
a grid tall, as Export all's standard says). On, every tile -- one plot per condition or per fit -- draws it under its
histogram as the card does (`_OverviewTile`: `PearsonStrip`, `EcdfResiduals`), the tile's fits only, on the
histogram's x range: every tile's range, as the histograms', reaches the longest ECDF point when that plot is on
(`jumps_view.residual_end`). The histogram carries the x axis; the plots under it are short and carry none. "Same y
scale" puts every residual plot of a kind on one range (`jumps_view.residual_spans`); otherwise each its own, as on
the cards. Export all starts with its "Pearson residuals" and "ECDF residuals" boxes as these are (the window's
target, `residuals_shown`), and a session keeps them (`view_state`).
"""
from __future__ import annotations

from qtpy.QtCore import QEvent, QObject, QPointF, QRectF, Qt, Signal
from qtpy.QtGui import QColor, QPen, QPolygonF
from qtpy.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ...condition_order import apply as order_apply
from ...jumps_view import (
    CV_LABELS,
    CV_UNITS,
    DEFAULTS,
    FIT_STYLE_WORDS,
    MAX_POP_CHOICES,
    READING_NOTES,
    OVERVIEW_LAYOUT_LABELS,
    OVERVIEW_LAYOUTS,
    OVERVIEW_ORDER_LABELS,
    OVERVIEW_ORDERS,
    as_shown,
    Y_MODE_LABELS,
    Y_MODES,
    DEFAULT_Y_MODE,
    as_y_mode,
    bin_width,
    check_axes,
    in_units,
    shared_tops,
    y_label,
    card_caption,
    clamp_place,
    components_drawn,
    corner_rect,
    config_problem,
    default_show,
    drawn_fits,
    ecdf_span,
    pearson_span,
    residual_end,
    residual_spans,
    RESIDUAL_PLOTS,
    fit_colour,
    histogram_top,
    in_colour,
    jumps_config,
    legend_corner,
    legend_rect_at,
    overview_tiles,
    place_wide,
    plot_fractions,
    setting_help,
    show_choices,
    show_words,
    toggle_show,
    verdict_line,
)
from ...chart_text import (
    ECDF_RESIDUAL_LABEL,
    JUMP_DISTANCE_LABEL,
    JUMPS_COUNT_LABEL,
    PEARSON_LABEL,
)
from ..theme import typography as ty
from ...axes_view import axis_rows
from ..widgets.axes_popover import AxesPopover as _AxesPopover
from ..widgets.cardgrid import CardGrid
from ..widgets.controls import Button, Segmented, Toggles
from ..widgets.dialogs import CloseButton, OverlayDialog
from ..widgets.primitives import SCROLLBAR_W, Checkbox, discard, label
from ..widgets.settings import InfoLink
from ..widgets.wraprow import WrapRow
from .condition_list import ConditionList as _ConditionList
from .qc_chart import _DATA, _FLAG, _Caption, _PanelCard, _QCChart
from .section_placeholder import SectionPlaceholder
from .toolbar import rem

__all__ = ["JumpCard", "JumpHistogram", "JumpsConfigDialog", "JumpsSection", "JumpsOverview",
           "EcdfResiduals", "PearsonStrip"]

#: v2's own line styles for 1, 2, 3 populations (`plotCDF`: ':', '-', '--'), a fourth dash-dot.
_STYLES = {1: Qt.DotLine, 2: Qt.SolidLine, 3: Qt.DashLine, 4: Qt.DashDotLine}
#: The bars' alpha in the condition's colour, and how much darker its components are drawn over them.
_BAR_ALPHA = 105
_COMPONENT_DARKER = 150
#: LINE WIDTHS, px (30 Oct 2026: "make the lines a bit bigger and better visible"). One fit alone 2.8 (was 2.0), two
#: or more 2.4 each (was 1.6); one fit's components 1.8 (was 1.3), several fits' components 1.5, dashed (new); the
#: Pearson strip's lines 1.9 (was 1.3); the ECDF residual 2.4 alone (was 1.8), 1.9 each of several (was 1.3).
_W_ONE, _W_MANY = 2.8, 2.4
_W_COMPONENT, _W_COMPONENT_MANY = 1.8, 1.5
_W_RESIDUAL = 1.9
_W_ECDF_ONE = 2.4
#: Several fits' components: dashes and gaps in pen widths, so a component never reads as a fit (v2's styles).
_COMPONENT_DASHES = [3.0, 2.2]


def _alpha(colour: str, a: int) -> QColor:
    c = QColor(colour)
    c.setAlpha(a)
    return c


def _dark(t) -> bool:
    return getattr(t, "color_scheme", "light") == "dark"


def _fit_pen(t, k: int, every: bool, *, one: float = _W_ONE, many: float = _W_MANY) -> QPen:
    """One fit alone in the ink; two or more (``every``, `jumps_view.in_colour`) each in its own colour
    (`jumps_view.fit_colour`, the theme's ground) and v2's style. Round joins: a thick line drawn bin to bin
    shows no notch at a bend."""
    if not every:
        pen = QPen(QColor(t.foreground), one)
    else:
        pen = QPen(QColor(fit_colour(k, _dark(t))), many)
        pen.setStyle(_STYLES.get(k, Qt.SolidLine))
    pen.setJoinStyle(Qt.RoundJoin)
    return pen


def _component_pen(t, k: int, every: bool, shade: QColor) -> QPen:
    """A component: one fit's in a darker shade of the condition's colour (as before); several fits' thin and
    dashed in that fit's colour, so the sum stays the line to read."""
    if not every:
        pen = QPen(_alpha(shade.name(), 235), _W_COMPONENT)
    else:
        pen = QPen(_alpha(fit_colour(k, _dark(t)), 225), _W_COMPONENT_MANY)
        pen.setDashPattern(_COMPONENT_DASHES)
        pen.setCapStyle(Qt.FlatCap)
    pen.setJoinStyle(Qt.RoundJoin)
    return pen


def _show_label(key) -> str:
    """A Show pill's words: ``"2 populations"``."""
    return show_words(key)


def _any_fits(keys) -> dict:
    """The counts ``keys`` as `jumps_view`'s ``fits`` (only their keys are read): what a switch offers."""
    return {int(k): None for k in keys}


class _JumpAxes(_QCChart):
    _FULL_BOTTOM = 40
    _ROOMY_H = 150
    _PLOT_LEFT = 66

    def drawn(self) -> list:
        """The fits this chart draws: `jumps_view.drawn_fits` of its payload's Show (every fit without one, the
        Signed jumps page's histogram), ``[]`` before a payload. The one place each chart's paint asks."""
        pl = self._payload
        if not pl or "fits" not in pl:
            return []
        if "show" not in pl:
            return sorted(int(k) for k in pl["fits"])
        return drawn_fits(pl["fits"], pl["show"])


class JumpHistogram(_JumpAxes):
    """Observed jumps per bin and the fits. ``payload``: ``{"edges", "observed", "fits", "colour"}``, and on the
    Jump distances page ``"show"`` (the counts on, a tuple: `_paint_shown`) and ``"components"`` (the card's box,
    on when absent) with an optional ``"x_end"`` / ``"y_end"`` (the All conditions view's shared ranges) and
    ``"legend"``. Without ``"show"`` (the Signed jumps page) every fit is drawn, ``"selected"`` strong, over grey bars
    (`_paint_every`).

    THE KEY (30 Oct 2026): in the corner where it covers least, unless it was put somewhere (``"legend_at"`` in the
    payload, which wins, else the place this widget was dragged to): a rectangle as fractions of the plot box, y up
    (`jumps_view.legend_rect_at`). Over the key the cursor is an open hand; dragging moves it inside the plot and
    ends in `legend_moved(place)`; a double-click on it puts it back in its corner (`legend_moved(None)`). A
    double-click anywhere else is left to the parent (an All conditions tile spans the grid on it)."""

    #: The key was dragged (its place, fractions of the plot box, y up) or put back (None).
    legend_moved = Signal(object)

    def __init__(self, t, parent=None):
        super().__init__(t, parent)
        #: Where this widget's key was put (the Signed jumps page; a payload's ``"legend_at"`` wins).
        self._legend_place = None
        #: Where the key was last drawn, and the plot box it was drawn in (px): what the pointer is tested against.
        self._legend_rect = QRectF()
        self._legend_box = QRectF()
        #: A drag in progress: ``(press point, key rectangle at the press)``, and where it is now.
        self._drag = None
        self._drag_place = None

    def legend_place(self):
        """Where the key is put: the payload's ``"legend_at"`` when it has one, else this widget's own; None for its
        corner."""
        pl = self._payload or {}
        return pl["legend_at"] if "legend_at" in pl else self._legend_place

    def legend_rect(self) -> QRectF:
        """Where the key was last drawn (px), empty without a key."""
        return QRectF(self._legend_rect)

    def _draw_key(self, p, box: QRectF, entries: list, series: list, edges, observed, x_lo, x_hi, lo, hi) -> None:
        """The key, at its place, else in the corner where it covers least of the bars and ``series``."""
        if not entries:
            self._legend_rect = QRectF()
            return
        w, h = self.legend_size(entries)
        size = (min(1.0, w / max(1.0, box.width())), min(1.0, h / max(1.0, box.height())))
        place = self._drag_place if self._drag is not None and self._drag_place is not None else self.legend_place()
        if place is None:
            curves, bars = plot_fractions(edges, observed, series, x_lo, x_hi, lo, hi)
            rect = corner_rect(legend_corner(curves, bars, size), size)
        else:
            rect = legend_rect_at(place, size)
        left = box.left() + rect[0] * box.width()
        top = box.bottom() - rect[3] * box.height()
        self._legend_rect = QRectF(left, top, w, h)
        self._legend_box = QRectF(box)
        self._legend(p, entries, QPointF(left + 4.0, top + 2.0), horizontal=False)

    def _place_of(self, rect: QRectF) -> tuple:
        """A key rectangle (px) as a place: fractions of the plot box, y up, kept inside it."""
        b = self._legend_box
        bw, bh = max(1.0, b.width()), max(1.0, b.height())
        return clamp_place(((rect.left() - b.left()) / bw, (b.bottom() - rect.bottom()) / bh,
                            (rect.right() - b.left()) / bw, (b.bottom() - rect.top()) / bh))

    def mousePressEvent(self, e) -> None:
        pos = e.position()
        if e.button() == Qt.LeftButton and self._legend_rect.contains(pos):
            self._drag, self._drag_place = (QPointF(pos), QRectF(self._legend_rect)), None
            self.setCursor(Qt.ClosedHandCursor)
            e.accept()
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e) -> None:
        pos = e.position()
        if self._drag is not None:
            start, rect = self._drag
            self._drag_place = self._place_of(rect.translated(pos - start))
            self.update()
            e.accept()
            return
        if self._legend_rect.contains(pos):
            self.setCursor(Qt.OpenHandCursor)
        else:
            self.unsetCursor()
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e) -> None:
        if self._drag is not None:
            place = self._drag_place
            self._drag, self._drag_place = None, None
            self.setCursor(Qt.OpenHandCursor)
            if place is not None:
                self._legend_place = place
                self.legend_moved.emit(place)
            self.update()
            e.accept()
            return
        super().mouseReleaseEvent(e)

    def mouseDoubleClickEvent(self, e) -> None:
        if self._legend_rect.contains(e.position()):
            self._legend_place = None
            self.legend_moved.emit(None)
            self.update()
            e.accept()
            return
        e.ignore()                          # the parent's: an All conditions tile spans the grid

    def legend_entries(self) -> list:
        """The key ``[(colour, words, "line")]``: the fits drawn and nothing else -- "1 population", "2 populations"
        (with their line style when several are drawn), as SMTracker 2's key (30 Oct 2026: "1 and 2 and 3 etc
        populations as it is"); the component curves are explained in How to read, never named here."""
        pl = self._payload
        if not pl or "show" not in pl:
            return []
        drawn = self.drawn()
        if in_colour(drawn):
            return [(fit_colour(k, _dark(self.t)), f"{show_words(k)} ({FIT_STYLE_WORDS.get(k, 'line')})", "line")
                    for k in drawn]
        return [(self.t.foreground, show_words(drawn[0]), "line")]

    def y_range(self) -> tuple:
        """``(from, to)`` of the y axis as last drawn, in the payload's unit."""
        return tuple(self._y_drawn) if getattr(self, "_y_drawn", None) else (None, None)

    def x_range(self) -> tuple:
        """``(from, to)`` of the x axis as last drawn, nm."""
        return tuple(self._x_drawn) if getattr(self, "_x_drawn", None) else (None, None)

    def components(self) -> list:
        """The fits whose components this histogram draws (`jumps_view.components_drawn`), ``[]`` without a Show."""
        pl = self._payload
        if not pl or "show" not in pl:
            return []
        return components_drawn(pl["fits"], self.drawn(), pl.get("components", True))

    def _paint(self, p, box: QRectF, pl) -> None:
        if "show" in pl:
            self._paint_shown(p, box, pl)
        else:
            self._paint_every(p, box, pl)

    def _paint_shown(self, p, box: QRectF, pl) -> None:
        """The bars in the condition's colour; one fit with its components, or the fits on, each in its own colour --
        in the payload's y unit (``"y_mode"``, the standard when absent: `jumps_view.in_units`), on its typed ranges
        (``"axes"``) or else fitted to what is drawn (``"y_end"``: a shared top, "Same y scale" or one condition's
        plots)."""
        edges = pl["edges"]
        mode = as_y_mode(pl.get("y_mode"))
        observed, fits = in_units(mode, pl["observed"], pl["fits"], edges)
        drawn = self.drawn()
        every = in_colour(drawn)
        parts = self.components()
        axes = pl.get("axes") or {}
        title = pl.get("y_label") or y_label(mode, [bin_width(edges)])
        top = max(histogram_top(observed, fits, drawn), float(pl.get("y_end") or 0.0))
        y_typed = axes.get("y") or (None, None)
        if y_typed == (None, None):
            lo, hi = self._grid(p, box, 0.0, top * 1.05, title)
        else:
            lo = 0.0 if y_typed[0] is None else float(y_typed[0])
            hi = top * 1.05 if y_typed[1] is None else float(y_typed[1])
            lo, hi = self._grid_exact(p, box, lo, max(hi, lo + 1e-12), title)
        x_typed = axes.get("x") or (None, None)
        x0 = edges[0] if x_typed[0] is None else float(x_typed[0])
        x1 = max(edges[-1], float(pl.get("x_end") or edges[-1])) if x_typed[1] is None else float(x_typed[1])
        self._y_drawn, self._x_drawn = (lo, hi), (x0, x1)

        def px(x):
            return box.left() + (x - x0) / (x1 - x0) * box.width()

        def py(v):
            return self._map_y(box, v, lo, hi)

        colour = QColor(pl.get("colour") or _DATA)
        shade = colour.darker(_COMPONENT_DARKER)
        p.save()
        p.setClipRect(box.adjusted(-1, -1, 1, 1))
        p.setPen(Qt.NoPen)
        p.setBrush(_alpha(colour.name(), _BAR_ALPHA))
        for a, b, c in zip(edges, edges[1:], observed):
            p.drawRect(QRectF(px(a) + 0.5, py(c), max(1.0, px(b) - px(a) - 1.0), py(0) - py(c)))
        centres = [(a + b) / 2 for a, b in zip(edges, edges[1:])]
        p.setBrush(Qt.NoBrush)
        for k in parts:
            p.setPen(_component_pen(self.t, k, every, shade))
            for comp in fits[k]["components"]:
                self._line(p, list(zip(centres, comp)), px, py)
        for k in drawn:
            p.setPen(_fit_pen(self.t, k, every))
            self._line(p, list(zip(centres, fits[k]["expected"])), px, py)
        p.restore()
        if pl.get("legend", True) and drawn:
            series = ([fits[k]["expected"] for k in drawn]
                      + [comp for k in parts for comp in fits[k]["components"]])
            self._draw_key(p, box, self.legend_entries(), series, edges, observed, x0, x1, lo, hi)
        else:
            self._legend_rect = QRectF()
        self._x_numeric(p, box, x0, x1, pl.get("x_label", JUMP_DISTANCE_LABEL))

    def _paint_every(self, p, box: QRectF, pl) -> None:
        edges, observed, fits = pl["edges"], pl["observed"], pl["fits"]
        top = max(max(observed), *(max(f["expected"]) for f in fits.values())) or 1.0
        lo, hi = self._grid(p, box, 0.0, top * 1.05, pl.get("y_label", JUMPS_COUNT_LABEL))
        x0, x1 = edges[0], edges[-1]

        def px(x):
            return box.left() + (x - x0) / (x1 - x0) * box.width()

        def py(v):
            return self._map_y(box, v, lo, hi)

        p.save()
        p.setClipRect(box.adjusted(-1, -1, 1, 1))
        p.setPen(Qt.NoPen)
        p.setBrush(_alpha(self.t.muted_foreground, 60))
        for a, b, c in zip(edges, edges[1:], observed):
            p.drawRect(QRectF(px(a) + 0.5, py(c), max(1.0, px(b) - px(a) - 1.0),
                              py(0) - py(c)))
        centres = [(a + b) / 2 for a, b in zip(edges, edges[1:])]
        colour = QColor(pl.get("colour") or _DATA)
        selected = pl["selected"]
        if selected in fits:
            for comp in fits[selected]["components"]:
                pen = QPen(_alpha(colour.name(), 170), 1.1)
                p.setPen(pen)
                p.setBrush(Qt.NoBrush)
                self._line(p, [(x, v) for x, v in zip(centres, comp)], px, py)
        for k, fit in sorted(fits.items()):
            pen = QPen(QColor(self.t.foreground) if k == selected
                       else QColor(self.t.muted_foreground), 2.0 if k == selected else 1.2)
            pen.setStyle(_STYLES.get(k, Qt.SolidLine))
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            self._line(p, list(zip(centres, fit["expected"])), px, py)
        p.restore()
        style = {1: "dotted", 2: "solid", 3: "dashed"}
        noun = pl.get("fit_noun", "population")
        entries = [(self.t.foreground if k == selected else self.t.muted_foreground,
                    f"{k} {noun}{'' if k == 1 else 's'} ({style.get(k, 'line')})", "line")
                   for k in sorted(fits)]
        entries.append((colour.name(), f"components of {selected}", "line"))
        series = ([f["expected"] for _k, f in sorted(fits.items())]
                  + (list(fits[selected]["components"]) if selected in fits else []))
        self._draw_key(p, box, entries, series, edges, observed, x0, x1, lo, hi)
        self._x_numeric(p, box, x0, x1, pl.get("x_label", JUMP_DISTANCE_LABEL))

    @staticmethod
    def _line(p, pts, px, py) -> None:
        """One polyline (not a segment per bin: a dash pattern then runs on along the curve, and a thick line has no
        notch at each bend)."""
        if len(pts) > 1:
            p.drawPolyline(QPolygonF([QPointF(px(x), py(y)) for x, y in pts]))


class PearsonStrip(_JumpAxes):
    """The shown fit's Pearson residuals per bin, (observed - expected) / sqrt(expected), as bars beyond +-2 in the
    flag colour, over the +-2 band counting noise alone would mostly stay inside; two or more fits on, each one's as
    a line in its own colour, and nothing for a fit that is off. ``payload``: ``{"edges", "fits", "show"}``, with an
    optional ``"span"`` (a range shared with other plots: "Same y scale") and ``"x_end"`` (the All conditions view's
    shared x range)."""

    def _paint(self, p, box: QRectF, pl) -> None:
        edges = pl["edges"]
        fits = pl.get("fits") or {}
        drawn = self.drawn()
        every = in_colour(drawn)
        series = [fits[k]["pearson"] for k in drawn]
        span = max(pearson_span(fits, drawn), float(pl.get("span") or 0.0))
        self._span_drawn = span
        lo, hi = self._grid(p, box, -span, span, PEARSON_LABEL)
        x_typed = (pl.get("axes") or {}).get("x") or (None, None)
        x0 = edges[0] if x_typed[0] is None else float(x_typed[0])
        x1 = max(edges[-1], float(pl.get("x_end") or edges[-1])) if x_typed[1] is None else float(x_typed[1])
        self._x_drawn = (x0, x1)

        def px(x):
            return box.left() + (x - x0) / (x1 - x0) * box.width()

        def py(v):
            return self._map_y(box, v, lo, hi)

        p.save()
        p.setClipRect(box)
        p.fillRect(QRectF(box.left(), py(2.0), box.width(), py(-2.0) - py(2.0)),
                   _alpha(self.t.muted_foreground, 28))
        if every:
            p.setPen(QPen(QColor(self.t.border), 1.0))
            p.drawLine(QPointF(box.left(), py(0.0)), QPointF(box.right(), py(0.0)))
            centres = [(a + b) / 2 for a, b in zip(edges, edges[1:])]
            for k in drawn:
                p.setPen(_fit_pen(self.t, k, True, many=_W_RESIDUAL))
                JumpHistogram._line(p, list(zip(centres, fits[k]["pearson"])), px, py)
            p.restore()
            self._x_numeric(p, box, x0, x1, JUMP_DISTANCE_LABEL)
            return
        p.setPen(Qt.NoPen)
        for a, b, v in zip(edges, edges[1:], series[0] if series else []):
            p.setBrush(QColor(_FLAG) if abs(v) > 2 else _alpha(self.t.foreground, 150))
            top, bottom = sorted((py(v), py(0.0)))
            p.drawRect(QRectF(px(a) + 0.5, top, max(1.0, px(b) - px(a) - 1.0),
                              max(0.8, bottom - top)))
        p.restore()
        self._x_numeric(p, box, x0, x1, JUMP_DISTANCE_LABEL)


class EcdfResiduals(_JumpAxes):
    """The shown fits' model CDF minus the ECDF, against jump distance: exact, unbinned; one fit in the ink, two or
    more each in its own colour, nothing for a fit that is off. An optional ``"span"`` and ``"x_end"`` as
    `PearsonStrip`'s."""

    def _paint(self, p, box: QRectF, pl) -> None:
        drawn = self.drawn()
        every = in_colour(drawn)
        fits = {k: pl["fits"][k] for k in drawn}
        span = max(ecdf_span(fits, drawn), float(pl.get("span") or 0.0))
        self._span_drawn = span
        lo, hi = self._grid(p, box, -span, span, ECDF_RESIDUAL_LABEL)
        x_typed = (pl.get("axes") or {}).get("x") or (None, None)
        x0 = 0.0 if x_typed[0] is None else float(x_typed[0])
        x1 = (max(residual_end(fits, drawn), float(pl.get("x_end") or 0.0)) if x_typed[1] is None
              else float(x_typed[1]))
        x1 = x1 if x1 > x0 else x0 + 1.0
        self._x_drawn = (x0, x1)

        def px(x):
            return box.left() + (x - x0) / (x1 - x0) * box.width()

        def py(v):
            return self._map_y(box, v, lo, hi)

        p.save()
        p.setClipRect(box.adjusted(-1, -1, 1, 1))
        p.setPen(QPen(QColor(self.t.border), 1.0))
        p.drawLine(QPointF(box.left(), py(0.0)), QPointF(box.right(), py(0.0)))
        for k, fit in sorted(fits.items()):
            p.setPen(_fit_pen(self.t, k, every, one=_W_ECDF_ONE, many=_W_RESIDUAL))
            JumpHistogram._line(p, fit["ecdf_residual"], px, py)
        p.restore()
        self._x_numeric(p, box, x0, x1, JUMP_DISTANCE_LABEL)


class _Table(QWidget):
    """The fit-quality table as a plain grid of labels (headers muted, numbers tabular)."""

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.setStyleSheet("background:transparent;")
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(rem(1.0))
        self._grid.setVerticalSpacing(rem(0.25))
        self._cells: list = []

    def set_table(self, columns: list, rows: list) -> None:
        for w in self._cells:
            discard(w)
        self._cells = []
        for c, name in enumerate(columns):
            w = label(name, ty.ui("xs", "medium"), self.t.muted_foreground)
            self._grid.addWidget(w, 0, c)
            self._cells.append(w)
        for r, row in enumerate(rows, start=1):
            for c, text in enumerate(row):
                w = label(text, ty.ui("xs", tabular=True), self.t.foreground)
                self._grid.addWidget(w, r, c)
                self._cells.append(w)
        # the columns keep their natural widths; the room left over goes after the last
        self._grid.setColumnStretch(len(columns), 1)


class _JumpBody(QWidget):
    """A card's content: a message, or the charts, the table and the notes."""

    #: The fits shown changed (by hand, by `set_shown`, or a result arriving): the All conditions view redraws.
    shown_changed = Signal()
    #: The typed ranges changed (the Axes popover, Auto, or a new unit dropping y): ``{"x", "y"}``.
    axes_changed = Signal(object)
    #: The y unit changed (`jumps_view.Y_MODES`).
    y_mode_changed = Signal(str)

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self._result = None
        self._colour = _DATA
        self.setStyleSheet("background:transparent;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(rem(1.25), 0, rem(1.25), rem(1.0))
        lay.setSpacing(rem(0.5))
        self.message = _Caption(t, size="sm")
        self.message.setText("computing…")
        lay.addWidget(self.message)
        row = QHBoxLayout()
        row.addWidget(label("Show", ty.ui("sm"), t.muted_foreground))
        #: What the pills offer (`jumps_view.show_choices`), rebuilt when a result's fits differ. Before a result:
        #: 1, 2 and 3 with two on (`default_show` without a BIC).
        self._keys: list = [1, 2, 3]
        self.choice = Toggles(t, [_show_label(k) for k in self._keys], on=[self._keys.index(2)], size="sm")
        self.choice.pressed.connect(self._pressed)
        #: The fits shown were picked by hand: the full pass, replacing the quick one, keeps them.
        self._by_hand = False
        row.addWidget(self.choice)
        row.addSpacing(rem(1.0))
        #: Components (30 Oct 2026): on to start -- one fit's, as before, and now several fits' too.
        self._components = True
        self.components_box = Checkbox(t, checked=True)
        self.components_box.setToolTip("Show each fit's populations. One fit uses shades of the condition's "
                                       "colour; several fits use thin dashed lines in each fit's colour.")
        self.components_box.toggled.connect(self._components_toggled)
        row.addWidget(self.components_box)
        row.addWidget(label("Components", ty.ui("sm"), t.muted_foreground))
        row.addStretch(1)
        self._row = row
        # THE Y AXIS (30 Oct 2026): its unit, Probability to start (`jumps_view.DEFAULT_Y_MODE`); a row of its own,
        # so the card's narrowest width stays the Show row's
        units = QHBoxLayout()
        units.addWidget(label("Y axis", ty.ui("sm"), t.muted_foreground))
        self._y_mode = DEFAULT_Y_MODE
        self.y_choice = Segmented(t, list(Y_MODE_LABELS), current=Y_MODES.index(DEFAULT_Y_MODE), size="sm")
        self.y_choice.setToolTip("Probability: the bars sum to 1, as in SMTracker 2 and Spot-On. Counts: jumps per "
                                 "bin. Density: per nm, total area 1. The fit curves use the same unit.")
        self.y_choice.changed.connect(lambda i: self.set_y_mode(Y_MODES[i]))
        units.addWidget(self.y_choice)
        units.addStretch(1)
        #: Typed ranges (the Axes popover): ``{"x": (from, to) or None, "y": ...}``; None ends are automatic.
        self._axes = {"x": None, "y": None}
        #: "Same y scale": the top every chart of this unit shares, else None (`JumpsSection.toggle_same_y`).
        self._shared_top = None
        self._choice_row = QWidget()
        rows = QVBoxLayout(self._choice_row)
        rows.setContentsMargins(0, 0, 0, 0)
        rows.setSpacing(rem(0.4))
        rows.addLayout(row)
        rows.addLayout(units)
        lay.addWidget(self._choice_row)
        # The histogram beside its two residual strips, each a FIXED height: the card's
        # height is its content's, never a number set on the card.
        self._charts = QWidget()
        self._charts.setStyleSheet("background:transparent;")
        charts = QHBoxLayout(self._charts)
        charts.setContentsMargins(0, 0, 0, 0)
        charts.setSpacing(rem(1.0))
        self.histogram = JumpHistogram(t)
        self.histogram.setFixedHeight(300)
        #: Where the key was put (None: its corner); the section keeps it across runs (`JumpsSection.legend_places`).
        self._legend_at = None
        self.histogram.legend_moved.connect(self.set_legend_at)
        charts.addWidget(self.histogram, 3)
        strips = QVBoxLayout()
        strips.setSpacing(rem(0.25))
        self.pearson = PearsonStrip(t)
        self.pearson.setFixedHeight(134)
        self.ecdf = EcdfResiduals(t)
        self.ecdf.setFixedHeight(162)            # >= `_ROOMY_H`: it carries the x axis
        strips.addWidget(self.pearson)
        strips.addWidget(self.ecdf)
        charts.addLayout(strips, 2)
        lay.addWidget(self._charts)
        self.table = _Table(t)
        lay.addWidget(self.table)
        self.verdict = _Caption(t, size="xs")
        lay.addWidget(self.verdict)
        lay.addStretch(1)
        self._lay = lay
        self._show(False)

    def _show(self, on: bool) -> None:
        for w in (self._choice_row, self._charts, self.table, self.verdict):
            w.setVisible(on)
        self.message.setVisible(not on)

    def set_error(self, text: str) -> None:
        self._result = None
        self.message.setText(text)
        self._show(False)

    def _pressed(self, index: int) -> None:
        """A pill clicked: that fit on or off (`jumps_view.toggle_show`; the last one on stays on)."""
        if not 0 <= index < len(self._keys):
            return
        self._by_hand = True
        self._light(toggle_show(self.shown(), self._keys[index], _any_fits(self._keys)))
        self._redraw()

    def legend_at(self):
        """Where this card's key was put (fractions of the plot box, y up), or None for its corner."""
        return self._legend_at

    def y_mode(self) -> str:
        """The y axis's unit (`jumps_view.Y_MODES`)."""
        return self._y_mode

    def set_y_mode(self, mode: str) -> None:
        """The y axis in ``mode``; a typed y range is dropped (its numbers meant the old unit) and the axis fitted
        again."""
        mode = as_y_mode(mode)
        if mode == self._y_mode:
            return
        self._y_mode = mode
        if self.y_choice.current() != Y_MODES.index(mode):
            self.y_choice.blockSignals(True)
            self.y_choice.select(Y_MODES.index(mode))
            self.y_choice.blockSignals(False)
        self.y_mode_changed.emit(mode)
        if self._axes.get("y") is not None:
            self._axes = {**self._axes, "y": None}
            self.axes_changed.emit(dict(self._axes))
        self._redraw()

    def axes(self) -> dict:
        """The typed ranges, ``{"x", "y"}`` (None: automatic)."""
        return dict(self._axes)

    def set_axes(self, axes) -> None:
        """Typed ranges (`jumps_view.check_axes`; None or both None: automatic)."""
        self._axes = check_axes(axes or {})
        self.axes_changed.emit(dict(self._axes))
        self._redraw()

    def set_shared_top(self, top) -> None:
        """"Same y scale": the top this card shares (None: its own); a redraw that tells nobody (it is the section
        that asked)."""
        if top != self._shared_top:
            self._shared_top = top
            self._redraw(emit=False)

    def open_axes(self, anchor=None) -> None:
        """The Axes popover for this card, under ``anchor``."""
        if getattr(self, "_popover", None) is None:
            self._popover = _AxesPopover(self.t, self)
            self._popover.applied.connect(self.set_axes)
        result = self._result
        unit = y_label(self._y_mode, [bin_width(result["edges_nm"])] if result else [])
        self._popover.show_for(axis_rows("jumps", unit=unit), self._axes, anchor,
                               note="Empty: automatic. A new y unit sets y back to automatic.")

    def set_legend_at(self, place) -> None:
        self._legend_at = None if place is None else tuple(float(v) for v in place)
        self._redraw()

    def _components_toggled(self, on: bool) -> None:
        self._components = bool(on)
        self._redraw()

    def components(self) -> bool:
        """Whether the Components box is ticked."""
        return self._components

    def set_components(self, on: bool) -> None:
        """Tick or untick Components, as asked by the All conditions view (silent on the box)."""
        on = bool(on)
        if on != self._components:
            self._components = on
            self.components_box.set_checked(on, animate=False)
            self._redraw()

    def _set_choices(self, keys: list) -> None:
        """Offer ``keys`` (the fits made): the pills are rebuilt only when they change, keeping on the counts still
        offered."""
        if keys == self._keys:
            return
        current = self.shown()
        old = self.choice
        self.choice = Toggles(self.t, [_show_label(k) for k in keys], size="sm",
                              on=[keys.index(k) for k in current if k in keys])
        self.choice.pressed.connect(self._pressed)
        self._row.replaceWidget(old, self.choice)
        discard(old)
        self._keys = list(keys)

    def _light(self, counts) -> None:
        """Pills ``counts`` on, the others off (silent: `Toggles.set_on`)."""
        self.choice.set_on([self._keys.index(k) for k in counts if k in self._keys])

    def set_shown(self, show) -> None:
        """Show the counts ``show`` names (a set of counts; an old single count or All is read by
        `jumps_view.as_shown`), as picked by hand: the All conditions view's "every condition". A card shows those of
        them it has; one that has none keeps its own."""
        counts = as_shown(show, _any_fits(self._keys))
        if not counts:
            return
        self._by_hand = True
        self._light(counts)
        self._redraw()

    def set_result(self, result: dict, colour: str) -> None:
        self._result, self._colour = result, colour
        self._set_choices([k for k, _w in show_choices(result["fits"])])
        if not self._by_hand or not self.shown():
            self._light(default_show(result))
        self.table.set_table(result["table"]["columns"], result["table"]["rows"])
        self.verdict.setText(verdict_line(result))
        self._show(True)
        self._redraw()

    def set_colour(self, colour: str) -> None:
        self._colour = colour
        self._redraw()

    def shown(self) -> tuple:
        """The counts Show has on, ascending (a tuple; at least one once a result is in)."""
        return tuple(self._keys[i] for i in self.choice.on() if 0 <= i < len(self._keys))

    def histogram_payload(self, **more) -> dict | None:
        """What this card's histogram draws (the All conditions view draws the same), or None without a result."""
        r = self._result
        if r is None:
            return None
        fits = {int(k): v for k, v in r["fits"].items()}
        return {"edges": r["edges_nm"], "observed": r["observed"], "fits": fits, "show": self.shown(),
                "components": self._components, "colour": self._colour, "legend_at": self._legend_at,
                "y_mode": self._y_mode, "axes": dict(self._axes), "y_end": self._shared_top, **more}

    def _redraw(self, emit: bool = True) -> None:
        """The histogram and BOTH residual strips on the same fits (the Show set) and the same typed x range."""
        payload = self.histogram_payload()
        if payload is None:
            return
        base = {"edges": payload["edges"], "fits": payload["fits"], "show": payload["show"],
                "colour": self._colour, "axes": {"x": self._axes.get("x")}}
        self.histogram.set_payload(payload)
        self.pearson.set_payload(base)
        self.ecdf.set_payload(base)
        if emit:
            self.shown_changed.emit()


class JumpCard(_PanelCard):
    def __init__(self, t, title: str, parent=None):
        self.jump_body = _JumpBody(t)
        super().__init__(t, title, self.jump_body, parent=parent, flush=False)
        self.shown_changed = self.jump_body.shown_changed
        # the Axes button on the card's head (30 Oct 2026: "make y and x axis changeable")
        self.enable_inspect("Axes", glyph="ruler")
        self.inspect_requested.connect(lambda: self.jump_body.open_axes(self._inspect))

    def set_result(self, result: dict, colour: str) -> None:
        self.set_caption(card_caption(result))
        self.jump_body.set_result(result, colour)

    def set_colour(self, colour: str) -> None:
        self.jump_body.set_colour(colour)

    def set_error(self, text: str) -> None:
        self.set_caption("")
        self.jump_body.set_error(text)

    def result(self) -> dict | None:
        """The result this card draws, or None (computing, or an error)."""
        return self.jump_body._result


class JumpsConfigDialog(OverlayDialog):
    """The Run's settings: which conditions, and what is held out to test each fit."""

    run = Signal(dict)

    def __init__(self, t, conditions: list, *, seed: dict | None = None, content=None,
                 parent=None):
        super().__init__(t, content, parent=parent)
        self.t = t
        seed = {**DEFAULTS, **(seed or {})}
        head = QHBoxLayout()
        head.addWidget(label("Configure jump distances", ty.ui("base", "semibold"),
                             t.foreground))
        head.addStretch(1)
        close = CloseButton(t)
        close.clicked.connect(self._dismiss)
        head.addWidget(close)
        self.card_lay.addLayout(head)
        self.card_lay.addWidget(label(
            "Fits 1, 2 and 3 populations (optionally 4) to every condition. Nothing runs until "
            "you press Run.", ty.ui("sm"), t.muted_foreground))
        columns = QHBoxLayout()
        columns.setSpacing(rem(2.0))
        left = QVBoxLayout()
        left.addWidget(label("Conditions", ty.ui("sm", "medium"), t.muted_foreground))
        self._conditions = _ConditionList(t, conditions, seed.get("conditions"))
        self._conditions.toggled.connect(self._refresh)
        self._conditions.setMinimumWidth(rem(13))
        left.addWidget(self._conditions)
        left.addStretch(1)
        columns.addLayout(left)
        right = QVBoxLayout()
        right.addWidget(label("Fit up to", ty.ui("sm", "medium"), t.muted_foreground))
        self._max_pop = Segmented(t, [f"{n} populations" for n in MAX_POP_CHOICES],
                                  current=MAX_POP_CHOICES.index(int(seed.get("max_pop", MAX_POP_CHOICES[0]))))
        top = QHBoxLayout()
        top.addWidget(self._max_pop)
        top.addStretch(1)
        right.addLayout(top)
        line, more = setting_help("max_pop")
        self.max_pop_helper = InfoLink(t, line, more)
        right.addWidget(self.max_pop_helper)
        right.addSpacing(rem(0.5))
        right.addWidget(label("Held out", ty.ui("sm", "medium"), t.muted_foreground))
        self._unit = Segmented(t, list(CV_LABELS), current=CV_UNITS.index(seed["cv_unit"]))
        holder = QHBoxLayout()
        holder.addWidget(self._unit)
        holder.addStretch(1)
        right.addLayout(holder)
        line, more = setting_help("cv")
        self.helper = InfoLink(t, line, more)
        right.addWidget(self.helper)
        right.addStretch(1)
        columns.addLayout(right, 1)
        self.card_lay.addLayout(columns)
        footer = QHBoxLayout()
        self._hint = label("", ty.ui("xs"), t.muted_foreground)
        footer.addWidget(self._hint)
        footer.addStretch(1)
        cancel = Button(t, "Cancel", variant="outline")
        cancel.clicked.connect(self._dismiss)
        self._run_btn = Button(t, "Run", variant="default")
        self._run_btn.clicked.connect(self._on_run)
        footer.addWidget(cancel)
        footer.addWidget(self._run_btn)
        self.card_lay.addLayout(footer)
        self._refresh()

    def _refresh(self) -> None:
        problem = config_problem(self._conditions.included())
        self._hint.setText(problem)
        self._run_btn.setEnabled(not problem)

    def config(self) -> dict:
        return jumps_config(self._conditions.included(), CV_UNITS[self._unit.current()],
                            MAX_POP_CHOICES[self._max_pop.current()])

    def _on_run(self) -> None:
        if config_problem(self._conditions.included()):
            return
        self._acted = True
        self.run.emit(self.config())
        self._dismiss()


class _OverviewTile(QWidget):
    """An All conditions plot's content (12 Nov 2026): its histogram and, under it, the residual plots the head's
    boxes have on (`JumpsOverview.residuals`), each the tile's fits on the histogram's x range. Short, they carry no x
    axis of their own: the histogram's is theirs."""

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background:transparent;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(rem(0.25))
        self.histogram = JumpHistogram(t)
        self.pearson = PearsonStrip(t)
        self.ecdf = EcdfResiduals(t)
        for chart in (self.histogram, self.pearson, self.ecdf):
            lay.addWidget(chart)
        self.pearson.setVisible(False)
        self.ecdf.setVisible(False)

    def strips(self) -> dict:
        """The residual plots by name (`jumps_view.RESIDUAL_PLOTS`)."""
        return {"pearson": self.pearson, "ecdf": self.ecdf}


class JumpsOverview(QWidget):
    """Every condition's histogram on one page, each as its card draws it, on one x range (asked 26 Oct 2026: "a
    way to print all currently loaded jd plots histogram into one screen and export together"). "Show for every
    condition" -- toggles like a card's -- asks the section to set every card to a set of counts
    (`show_all_requested`), its Components box every card's box (`components_requested`); Export all asks for one
    figure.

    TWO LAYOUTS (30 Oct 2026): "One plot per condition" (as before: a tile per condition in a flowing grid of up to
    three columns) or "One plot per fit" (a tile per condition and fit shown, that fit alone, on a fixed grid,
    "Conditions down" -- a row per condition -- or "Conditions across"); `jumps_view.overview_tiles` decides where,
    and `arrangement_changed` asks the section to lay them out again.

    THE RESIDUAL PLOTS (12 Nov 2026): a box each, Pearson and ECDF, off to start; on, every tile draws it under its
    histogram (`_OverviewTile`), and `residuals_changed` asks the section to draw them."""

    show_all_requested = Signal(object)
    components_requested = Signal(bool)
    residuals_changed = Signal()
    arrangement_changed = Signal()
    export_requested = Signal()
    #: A tile's key was dragged or put back: ``(tile key, place)`` -- a condition, or ``(condition, count)``.
    tile_legend_moved = Signal(object, object)
    #: The head's y unit for every card; "Same y scale" pressed; the head's Axes (every tile); a tile's Axes button.
    y_mode_requested = Signal(str)
    same_y_pressed = Signal()
    all_axes_requested = Signal()
    tile_axes_requested = Signal(object)

    #: A tile's histogram height, and the narrowest a column may be (Size S, as before).
    TILE_H = 230
    TILE_MIN_W = 340
    #: One plot per fit: a smaller tile, so three or four fits sit side by side on a laptop.
    FIT_TILE_H = 200
    FIT_TILE_MIN_W = 180
    #: The Size switch (30 Oct 2026, "increasing size of these cards in a way if its too small"): per size, one plot
    #: per condition's column width and histogram height, and one plot per fit's histogram height (its columns are
    #: the fits or the conditions, so its width is the grid's). On a 1536 px laptop (about 1 250 px beside the
    #: sidebar): S three tiles a row, M two, L one (two from about 1 380 px).
    SIZES = ("S", "M", "L")
    TILE_SIZES = {"S": (340, 230), "M": (480, 300), "L": (680, 400)}
    FIT_TILE_HEIGHTS = {"S": 200, "M": 260, "L": 330}
    #: A tile spanning the grid (a double-click): this tall.
    WIDE_H = 400
    #: A residual plot's height under a tile, per size (12 Nov 2026): under `_JumpAxes._ROOMY_H`, so it carries no x
    #: axis (the histogram over it does, on the same range), and at least 120 so its plot is the 90 px a y axis title
    #: needs (`_draw_grid`; at 92 the first render had two untitled plots, Pearson and ECDF not told apart); a wide
    #: tile's at L.
    STRIP_HEIGHTS = {"S": 120, "M": 130, "L": 142}
    #: The Columns switch (one plot per condition): Auto, or exactly 1-6 across (`figure_export.GRID_MAX_COLUMNS`).
    COLUMNS = ((None, "Auto"), (1, "1"), (2, "2"), (3, "3"), (4, "4"), (5, "5"), (6, "6"))

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.setStyleSheet("background:transparent;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(rem(0.75))
        # THE HEADER ROWS WRAP (`WrapRow`, 30 Oct 2026): as one `QHBoxLayout` each, on a 1536 px laptop the page was
        # narrower than the row and cut at its right edge ("All c", Export all's "E"). A label stays with its control.
        row = WrapRow(spacing=rem(0.75), line_spacing=rem(0.5))
        head = QHBoxLayout()
        head.addWidget(label("Show for every condition", ty.ui("sm"), t.muted_foreground))
        self._keys: list = [1, 2, 3]
        self.choice = Toggles(t, [_show_label(k) for k in self._keys], on=[self._keys.index(2)], size="sm")
        self.choice.pressed.connect(self._pressed)
        head.addWidget(self.choice)
        row.addLayout(head)
        box = QHBoxLayout()
        self.components_box = Checkbox(t, checked=True)
        self.components_box.setToolTip("Every condition's components on or off.")
        self.components_box.toggled.connect(self.components_requested.emit)
        box.addWidget(self.components_box)
        box.addWidget(label("Components", ty.ui("sm"), t.muted_foreground))
        row.addLayout(box)
        # THE RESIDUAL PLOTS (12 Nov 2026, "why not add Pearson and ecdf"): off to start, as Export all's are; the
        # label stays with its boxes when the row wraps
        residuals = QHBoxLayout()
        residuals.addWidget(label("Residuals", ty.ui("sm"), t.muted_foreground))
        self.pearson_box = Checkbox(t, checked=False)
        self.pearson_box.setToolTip("Show each fit's Pearson residuals, (observed - expected) / sqrt(expected), "
                                    "under every plot. Counting noise alone mostly stays within ±2. Export all "
                                    "follows this setting.")
        self.ecdf_box = Checkbox(t, checked=False)
        self.ecdf_box.setToolTip("Show each fit's model CDF minus the empirical CDF of the jumps, unbinned, "
                                 "under every plot. Export all follows this setting.")
        for name, check in (("Pearson", self.pearson_box), ("ECDF", self.ecdf_box)):
            check.toggled.connect(lambda _on: self._residuals_toggled())
            residuals.addWidget(check)
            residuals.addWidget(label(name, ty.ui("sm"), t.muted_foreground))
        row.addLayout(residuals)
        row.addStretch(1)
        self.export_button = Button(t, "Export all…", size="sm", glyph="download")
        self.export_button.setToolTip("Export one figure with every condition's histogram, laid out as shown here.")
        self.export_button.clicked.connect(self.export_requested.emit)
        row.addWidget(self.export_button)
        self._head = head
        lay.addLayout(row)
        arrange = WrapRow(spacing=rem(0.5), line_spacing=rem(0.5))
        plots = QHBoxLayout()
        plots.addWidget(label("Plots", ty.ui("sm"), t.muted_foreground))
        self.layout_choice = Segmented(t, list(OVERVIEW_LAYOUT_LABELS), current=0, size="sm")
        self.layout_choice.changed.connect(self._arranged)
        plots.addWidget(self.layout_choice)
        arrange.addLayout(plots)
        self.order_choice = Segmented(t, list(OVERVIEW_ORDER_LABELS), current=0, size="sm")
        self.order_choice.changed.connect(self._arranged)
        arrange.addWidget(self.order_choice)
        # A GRID OF ONE'S CHOICE (7 Nov 2026, asked: "would be cool to draw and export as grids for example 4x2 /
        # 2x4"): one plot per condition in exactly that many columns, here and in Export all (its Grid row starts on
        # it); Auto flows them as before. Wider than the page, the grid scrolls sideways.
        columns = QHBoxLayout()
        self._columns_label = label("Columns", ty.ui("sm"), t.muted_foreground)
        columns.addWidget(self._columns_label)
        self.columns_choice = Segmented(t, [w for _v, w in self.COLUMNS], current=0, size="sm")
        self.columns_choice.setToolTip("The number of plots across, here and in Export all. Auto fits as many as "
                                       "the page holds.")
        self.columns_choice.changed.connect(self._columns_changed)
        columns.addWidget(self.columns_choice)
        arrange.addLayout(columns)
        arrange.addStretch(1)
        size = QHBoxLayout()                                   # the label stays with its control when the row wraps
        size.addWidget(label("Size", ty.ui("sm"), t.muted_foreground))
        self.size_choice = Segmented(t, list(self.SIZES), current=0, size="sm")
        self.size_choice.setToolTip("The size of each plot. Double-click a plot to span the page; double-click "
                                    "again to restore it.")
        self.size_choice.changed.connect(self._resized)
        size.addWidget(self.size_choice)
        arrange.addLayout(size)
        lay.addLayout(arrange)
        # THE Y AXIS FOR EVERY CARD, ONE Y SCALE, THE AXES OF EVERY TILE (30 Oct 2026), a row of their own
        # wrapping like the rows above (`WrapRow`), so it is never cut at a laptop's right edge
        units = WrapRow(spacing=rem(0.75), line_spacing=rem(0.5))
        y_axis = QHBoxLayout()
        y_axis.addWidget(label("Y axis", ty.ui("sm"), t.muted_foreground))
        self.y_choice = Segmented(t, list(Y_MODE_LABELS), current=Y_MODES.index(DEFAULT_Y_MODE), size="sm")
        self.y_choice.setToolTip("The y unit of every condition: Probability (per bin), Counts or Density (per nm).")
        self.y_choice.pressed.connect(lambda i: self.y_mode_requested.emit(Y_MODES[i]))
        y_axis.addWidget(self.y_choice)
        units.addLayout(y_axis)
        self.same_y_button = Button(t, "Same y scale", size="sm")
        self.same_y_button.setToolTip(_SAME_Y_TIP)
        self.same_y_button.clicked.connect(self.same_y_pressed.emit)
        units.addWidget(self.same_y_button)
        self.axes_button = Button(t, "Axes…", size="sm", glyph="ruler")
        self.axes_button.setToolTip("Set the x and y ranges of every plot. A range set on a single plot takes "
                                    "precedence.")
        self.axes_button.clicked.connect(self.all_axes_requested.emit)
        units.addWidget(self.axes_button)
        units.addStretch(1)
        lay.addLayout(units)
        self.order_choice.setVisible(False)                   # only under one plot per fit
        holder = QWidget()
        holder.setStyleSheet("background:transparent;")
        self._grid = CardGrid(holder, min_width=self.TILE_MIN_W, max_columns=3, spacing=rem(1.0))
        # in a sideways scroll too: a Columns count wider than the page scrolls instead of being cut
        self._flow_scroll = _SidewaysScroll(holder)
        lay.addWidget(self._flow_scroll)
        self._flow_holder = holder
        fit_holder = QWidget()
        fit_holder.setStyleSheet("background:transparent;")
        self._fit_grid = QGridLayout(fit_holder)
        self._fit_grid.setContentsMargins(0, 0, 0, 0)
        self._fit_grid.setSpacing(rem(0.75))
        # SIDEWAYS WHEN WIDER THAN THE PAGE (7 Nov 2026, reported: "when i have 8 conditions ... one plot per fit and
        # conditions across, i dont even see all conditions and there is no way to scroll sidewards"): the grid's
        # columns keep their minimum width and the grid alone scrolls sideways; the page around it stays the page's
        self._fit_scroll = _SidewaysScroll(fit_holder)
        lay.addWidget(self._fit_scroll)
        self._fit_scroll.setVisible(False)
        self._fit_holder = fit_holder
        self._tiles: dict = {}
        #: One plot per fit: ``(condition, count) -> tile``, and where they stand (rebuilt only when that changes).
        self._fit_tiles: dict = {}
        self._fit_places: tuple = ()
        #: The tiles spanning the grid (a condition, or ``(condition, count)``), kept while the window lives.
        self._wide: set = set()
        self._fit_columns = 1

    def _pressed(self, index: int) -> None:
        """A pill clicked: the set marked with that count toggled (`jumps_view.toggle_show`), for every card."""
        if 0 <= index < len(self._keys):
            self.show_all_requested.emit(toggle_show(self.marked(), self._keys[index], _any_fits(self._keys)))

    def _arranged(self, _index: int = 0) -> None:
        by_fit = self.layout_mode() == "fit"
        self.order_choice.setVisible(by_fit)
        self._columns_label.setVisible(not by_fit)           # one plot per fit's grid is the conditions x the fits
        self.columns_choice.setVisible(not by_fit)
        self.arrangement_changed.emit()

    def _columns_changed(self, _index: int = 0) -> None:
        self._grid.set_columns(self.columns())

    def columns(self) -> int | None:
        """The plots across under one plot per condition: a number, or None (Auto)."""
        return self.COLUMNS[max(0, self.columns_choice.current())][0]

    def set_columns(self, columns: int | None) -> None:
        """Choose the Columns (as a click would); a count not offered reads as Auto."""
        values = [v for v, _w in self.COLUMNS]
        self.columns_choice.select(values.index(columns) if columns in values else 0)
        self._columns_changed()

    def layout_mode(self) -> str:
        """``"condition"`` (one plot per condition) or ``"fit"`` (one plot per fit)."""
        return OVERVIEW_LAYOUTS[max(0, self.layout_choice.current())]

    def order(self) -> str:
        """``"down"`` (a row per condition) or ``"across"`` (a column per condition), under one plot per fit."""
        return OVERVIEW_ORDERS[max(0, self.order_choice.current())]

    def set_arrangement(self, layout: str, order: str = "down") -> None:
        """Choose the layout and the order (as a click on the switches would)."""
        self.layout_choice.select(OVERVIEW_LAYOUTS.index(layout))
        self.order_choice.select(OVERVIEW_ORDERS.index(order))
        self._arranged()

    def set_choices(self, keys: list) -> None:
        """Offer ``keys`` (every count some condition was fitted at)."""
        if keys == self._keys:
            return
        current = self.marked()
        old = self.choice
        self.choice = Toggles(self.t, [_show_label(k) for k in keys], size="sm",
                              on=[keys.index(k) for k in current if k in keys])
        self.choice.pressed.connect(self._pressed)
        self._head.replaceWidget(old, self.choice)
        discard(old)
        self._keys = list(keys)

    def marked(self) -> tuple:
        """The counts lit, ascending: the set every card shows, or ``()`` while the cards differ."""
        return tuple(self._keys[i] for i in self.choice.on() if 0 <= i < len(self._keys))

    def mark(self, show) -> None:
        """Light the counts ``show`` names (the set every card agrees on), or none (``()``: the cards differ)."""
        counts = as_shown(show, _any_fits(self._keys))
        self.choice.set_on([self._keys.index(k) for k in counts])

    def tile_size(self) -> str:
        """``"S"``, ``"M"`` or ``"L"``."""
        return self.SIZES[max(0, self.size_choice.current())]

    def set_tile_size(self, size: str) -> None:
        self.size_choice.select(self.SIZES.index(size))

    def _height(self, key, fit: bool) -> int:
        if key in self._wide:
            return self.WIDE_H
        size = self.tile_size()
        return self.FIT_TILE_HEIGHTS[size] if fit else self.TILE_SIZES[size][1]

    def _strip_height(self, key) -> int:
        return self.STRIP_HEIGHTS["L" if key in self._wide else self.tile_size()]

    def _size_tile(self, tile, key, fit: bool) -> None:
        """A tile's histogram and residual plots at the size picked (a wide one taller)."""
        tile.histogram.setFixedHeight(self._height(key, fit))
        for strip in tile.body.strips().values():
            strip.setFixedHeight(self._strip_height(key))

    def _resized(self, _index: int = 0) -> None:
        """Every tile at the size picked: one plot per condition's columns and heights, one plot per fit's heights."""
        self._grid.set_column_width(self.TILE_SIZES[self.tile_size()][0])
        for key, tile in self._tiles.items():
            self._size_tile(tile, key, False)
        for key, tile in self._fit_tiles.items():
            self._size_tile(tile, key, True)

    def residuals(self) -> tuple:
        """The residual plots ticked, in the order they stand (`jumps_view.RESIDUAL_PLOTS`): ``("pearson", "ecdf")``,
        one of them, or ``()``."""
        on = {"pearson": self.pearson_box.checked, "ecdf": self.ecdf_box.checked}
        return tuple(name for name in RESIDUAL_PLOTS if on[name])

    def set_residuals(self, names) -> None:
        """Tick the residual plots ``names`` (as clicks would), the others off."""
        names = set(names or ())
        for name, check in (("pearson", self.pearson_box), ("ecdf", self.ecdf_box)):
            check.set_checked(name in names, animate=False)
        self._residuals_toggled()

    def _residuals_toggled(self) -> None:
        for tile in [*self._tiles.values(), *self._fit_tiles.values()]:
            self._show_strips(tile)
        self.residuals_changed.emit()

    def _show_strips(self, tile) -> None:
        on = self.residuals()
        for name, strip in tile.body.strips().items():
            strip.setVisible(name in on)

    def wide(self) -> set:
        """The tiles spanning the grid."""
        return set(self._wide)

    def toggle_wide(self, key) -> None:
        """Tile ``key`` across the whole grid, or back among the others (a double-click on it)."""
        if key in self._wide:
            self._wide.discard(key)
        else:
            self._wide.add(key)
        tile = self._tiles.get(key) if not isinstance(key, tuple) else None
        if tile is not None:
            self._grid.set_wide(tile, key in self._wide)
            self._size_tile(tile, key, False)
        elif key in self._fit_tiles:
            self._size_tile(self._fit_tiles[key], key, True)
            self._place_fit_tiles()

    def _watch(self, tile, key) -> None:
        """A tile's key reports its place, a double-click on the tile (not on its key) spans it, and its head's Axes
        button asks for its ranges."""
        tile.histogram.legend_moved.connect(lambda place, k=key: self.tile_legend_moved.emit(k, place))
        tile._double_clicks = _DoubleClicks(tile, lambda k=key: self.toggle_wide(k))
        tile.enable_inspect("Axes", glyph="ruler")
        tile.inspect_requested.connect(lambda k=key: self.tile_axes_requested.emit(k))

    def mark_y_mode(self, mode) -> None:
        """Light the unit every card has (None: they differ, nothing lit)."""
        index = -1 if mode is None else Y_MODES.index(mode)
        if self.y_choice.current() != index:
            self.y_choice.blockSignals(True)
            self.y_choice.select(index)
            self.y_choice.blockSignals(False)

    def mark_same_y(self, on: bool) -> None:
        self.same_y_button.set_text("Own y scales" if on else "Same y scale")

    def tile_widget(self, key):
        """The tile ``key`` (a condition, or ``(condition, count)``), or None."""
        return self._fit_tiles.get(key) if isinstance(key, tuple) else self._tiles.get(key)

    def mark_components(self, on: bool) -> None:
        """Tick the Components box when every card has it ticked (silent)."""
        self.components_box.set_checked(bool(on), animate=False)

    def set_tiles(self, order: list, labels: dict) -> None:
        """One tile per condition, in the cards' order, each empty until its card has a result."""
        for tile in self._tiles.values():
            discard(tile)
        self._tiles = {}
        for key in order:
            tile = self._new_tile(labels.get(key, key), key, False)
            self._watch(tile, key)
            self._grid.addWidget(tile)
            self._grid.set_wide(tile, key in self._wide)
            tile.setVisible(False)
            self._tiles[key] = tile
        self._clear_fit_tiles()

    def _new_tile(self, title: str, key, fit: bool):
        """A tile: its histogram and the residual plots under it (`_OverviewTile`), sized, the plots as ticked."""
        body = _OverviewTile(self.t)
        tile = _PanelCard(self.t, title, body)
        tile.body, tile.histogram, tile.pearson, tile.ecdf = body, body.histogram, body.pearson, body.ecdf
        self._size_tile(tile, key, fit)
        self._show_strips(tile)
        return tile

    @staticmethod
    def _draw_strips(tile, strips: dict | None) -> None:
        """The residual plots' payload (``{"edges", "fits", "show", "axes", "x_end", "span": {kind: range}}``), or
        nothing to draw."""
        if not strips:
            return
        spans = strips.get("span") or {}
        base = {k: v for k, v in strips.items() if k != "span"}
        for name, strip in tile.body.strips().items():
            strip.set_payload({**base, "span": spans.get(name)})

    def set_tile(self, key: str, payload: dict | None, caption: str = "", strips: dict | None = None) -> None:
        tile = self._tiles.get(key)
        if tile is None:
            return
        if payload is None:
            tile.setVisible(False)
            return
        tile.histogram.set_payload(payload)
        self._draw_strips(tile, strips)
        tile.set_caption(caption)
        tile.setVisible(True)

    def show_layout(self, layout: str) -> None:
        """The flowing tiles (one plot per condition) or the fixed grid (one plot per fit)."""
        self._flow_scroll.setVisible(layout == "condition")
        self._fit_scroll.setVisible(layout == "fit")

    def _clear_fit_tiles(self) -> None:
        for tile in self._fit_tiles.values():
            self._fit_grid.removeWidget(tile)
            discard(tile)
        self._fit_tiles, self._fit_places = {}, ()

    def set_fit_tiles(self, tiles: list) -> None:
        """One plot per fit: ``tiles`` ``[{"key", "count", "row", "column", "title", "payload", "caption"}]``
        (`jumps_view.overview_tiles`' places). Rebuilt only when the places change; otherwise each tile is drawn
        again."""
        places = tuple((t["key"], t["count"], t["row"], t["column"], t["title"]) for t in tiles)
        if places != self._fit_places:
            self._clear_fit_tiles()
            for spot in tiles:
                key = (spot["key"], spot["count"])
                tile = self._new_tile(spot["title"], key, True)
                tile.histogram.setMinimumWidth(self.FIT_TILE_MIN_W)
                tile.grid_place = (spot["row"], spot["column"])
                self._watch(tile, key)
                self._fit_tiles[key] = tile
            self._fit_columns = 1 + max((t["column"] for t in tiles), default=0)
            self._fit_places = places
            self._place_fit_tiles()
        for spot in tiles:
            tile = self._fit_tiles[(spot["key"], spot["count"])]
            tile.histogram.set_payload(spot["payload"])
            self._draw_strips(tile, spot.get("strips"))
            tile.set_caption(spot.get("caption", ""))

    def _place_fit_tiles(self) -> None:
        """Every tile of one plot per fit in its cell; a wide one in a row of its own across the grid
        (`jumps_view.place_wide`)."""
        for tile in self._fit_tiles.values():
            self._fit_grid.removeWidget(tile)
        spots = [(key, tile.grid_place[0], tile.grid_place[1]) for key, tile in self._fit_tiles.items()]
        for key, row, column, rows, columns in place_wide(spots, self._wide, self._fit_columns):
            self._fit_grid.addWidget(self._fit_tiles[key], row, column, rows, columns)

    def tiles(self) -> dict:
        return dict(self._tiles)

    def fit_tiles(self) -> dict:
        """One plot per fit's tiles, ``(condition, count) -> tile``."""
        return dict(self._fit_tiles)


class _SidewaysScroll(QScrollArea):
    """A grid that scrolls SIDEWAYS ONLY, when it is wider than the page: its height is always the grid's whole height
    (plus the bar's, kept whether the bar shows or not, so nothing jumps), so the page scrolls it up and down with
    everything else. The grid's height follows its tiles (their size, a wide one, a caption), read on every layout
    request."""

    def __init__(self, content: QWidget, parent=None):
        super().__init__(parent)
        self.setWidget(content)
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.NoFrame)
        self.setStyleSheet("QScrollArea { background:transparent; border:none; }")
        self.viewport().setStyleSheet("background:transparent;")
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        content.installEventFilter(self)
        self._follow()

    def _follow(self) -> None:
        content = self.widget()
        if content is not None:
            self.setFixedHeight(content.sizeHint().height() + SCROLLBAR_W)

    def eventFilter(self, obj, event):
        if obj is self.widget() and event.type() == QEvent.LayoutRequest:
            self._follow()
        return super().eventFilter(obj, event)

    def minimumSizeHint(self):
        hint = super().minimumSizeHint()
        hint.setWidth(0)                    # never the page's minimum: the grid scrolls instead
        return hint


#: "Same y scale"'s tooltip, said wherever the button is.
_SAME_Y_TIP = ("Put every plot of one y unit on the largest automatic range among them. Press again to return "
               "to each plot's own. Counts of different numbers of jumps do not compare; use Probability or "
               "Density for that.")


class _DoubleClicks(QObject):
    """A double-click reaching a tile (anywhere but its key, which takes its own) runs ``action``."""

    def __init__(self, target, action):
        super().__init__(target)
        self._action = action
        target.installEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.MouseButtonDblClick:
            self._action()
            return True
        return False


class JumpsSection(QWidget):
    """The Jump distances page: a placeholder until the first result, then one card per
    included condition, filling in as each lands -- or, under the view switch's All
    conditions, every condition's histogram on one page (`JumpsOverview`)."""

    run_requested = Signal()
    cancel_requested = Signal()
    #: A card's Export button, with the condition's key.
    export_requested = Signal(str)
    #: The All conditions view's Export all.
    export_all_requested = Signal()
    #: The view switch's choices.
    VIEWS = ("Each condition", "All conditions")

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.setStyleSheet("background:transparent;")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.placeholder = SectionPlaceholder(
            t, title="Jump distances",
            description="Each condition's jumps against the 1-, 2- and 3-population fits, "
                        "with how well each fits, on the data and on cells or replicates held out of the fit.",
            icon_name="chart")
        self.placeholder.run_requested.connect(self.run_requested.emit)
        self.placeholder.cancel_requested.connect(self.cancel_requested.emit)
        self._page = QWidget()
        self._page.setStyleSheet("background:transparent;")
        page = QVBoxLayout(self._page)
        page.setContentsMargins(0, 0, 0, rem(0.5))
        page.setSpacing(rem(1.0))
        #: Said once for every card (`jumps_view.READING_NOTES`).
        self.reading = InfoLink(t, "How to read these charts and the table", READING_NOTES)
        top = WrapRow(spacing=rem(0.75), line_spacing=rem(0.5))      # wraps, never runs past the page (`WrapRow`)
        top.addWidget(self.reading)
        top.addStretch(1)
        self.view = Segmented(t, list(self.VIEWS), current=0, size="sm")
        self.view.changed.connect(self._view_changed)
        top.addWidget(self.view)
        page.addLayout(top)
        self.overview = JumpsOverview(t)
        self.overview.show_all_requested.connect(self.show_everywhere)
        self.overview.components_requested.connect(self.components_everywhere)
        self.overview.residuals_changed.connect(self._refresh_overview)
        self.overview.arrangement_changed.connect(self._refresh_overview)
        self.overview.tile_legend_moved.connect(self._tile_legend_moved)
        self.overview.y_mode_requested.connect(self.y_mode_everywhere)
        self.overview.same_y_pressed.connect(self.toggle_same_y)
        self.overview.all_axes_requested.connect(lambda: self.open_axes(("all",), self.overview.axes_button))
        self.overview.tile_axes_requested.connect(self._tile_axes_requested)
        self.overview.export_requested.connect(self.export_all_requested.emit)
        page.addWidget(self.overview)
        self.overview.setVisible(False)
        grid_host = QWidget()
        grid_host.setStyleSheet("background:transparent;")
        # ONE COLUMN: a card's histogram and residual strips sit side by side and need the
        # row's width (the 2x2 grid squeezed them).
        self._grid = QGridLayout(grid_host)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(rem(1.0))
        # over the cards (row 0; the cards from row 1): one y scale for every card (30 Oct 2026)
        cards_bar = QWidget()
        cards_bar.setStyleSheet("background:transparent;")
        bar = QHBoxLayout(cards_bar)
        bar.setContentsMargins(0, 0, 0, 0)
        self.cards_same_y_button = Button(t, "Same y scale", size="sm")
        self.cards_same_y_button.setToolTip(_SAME_Y_TIP)
        self.cards_same_y_button.clicked.connect(self.toggle_same_y)
        bar.addWidget(self.cards_same_y_button)
        bar.addStretch(1)
        self._grid.addWidget(cards_bar, 0, 0)
        page.addWidget(grid_host)
        self._cards_host = grid_host
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { background:transparent; }")
        host = QWidget()
        host.setStyleSheet("background:transparent;")
        host_lay = QVBoxLayout(host)
        host_lay.setContentsMargins(0, 0, 0, 0)
        host_lay.addWidget(self._page)
        host_lay.addStretch(1)
        scroll.setWidget(host)
        self._pages = QStackedWidget()
        self._pages.addWidget(self.placeholder)
        self._pages.addWidget(scroll)
        outer.addWidget(self._pages)
        self._cards: dict[str, JumpCard] = {}
        self._labels: dict[str, str] = {}
        self._colours: dict[str, str] = {}
        self._ever_shown = False
        self.last_config: dict = {}
        #: Where each chart's key was put, kept while the window lives (and in a saved session): ``("card", key)``,
        #: ``("tile", key)`` and ``("fit", key, count)`` -> a place (fractions of the plot box, y up).
        self._legend_places: dict = {}
        #: Typed ranges, the same charts' keys and ``("all",)`` (the All conditions head's, for every tile).
        self._axes: dict = {}
        #: "Same y scale" (every chart of one unit on the largest automatic range among them).
        self._same_y = False
        self._popover = None

    def begin_run(self, conditions: list, labels: dict | None = None,
                  colours: dict | None = None, config: dict | None = None) -> None:
        for card in self._cards.values():
            discard(card)
        self._cards = {}
        self._labels = dict(labels or {})
        self._colours = dict(colours or {})
        for i, key in enumerate(conditions):
            card = JumpCard(self.t, f"Jump distances · {self._labels.get(key, key)}")
            card.enable_export()
            card.export_requested.connect(lambda k=key: self.export_requested.emit(k))
            card.shown_changed.connect(self._refresh_overview)
            card.jump_body.set_legend_at(self._legend_places.get(("card", key)))
            card.jump_body.histogram.legend_moved.connect(
                lambda place, k=key: self._keep_place(("card", k), place))
            if self._axes.get(("card", key)):
                card.jump_body.set_axes(self._axes[("card", key)])
            card.jump_body.axes_changed.connect(lambda axes, k=key: self._keep_axes(("card", k), axes))
            card.jump_body.y_mode_changed.connect(lambda _mode, k=key: self._unit_changed(k))
            self._cards[key] = card
            self._grid.addWidget(card, i + 1, 0)
        self.overview.set_tiles(list(conditions), self._labels)
        self.last_config = dict(config or {})
        self.placeholder.set_running(True)
        self._pages.setCurrentIndex(1 if self._ever_shown else 0)

    def set_order(self, order) -> None:
        """The conditions in ``order`` (`condition_order`, the Conditions tool's arrows, 3 Nov 2026): the cards and the
        All conditions tiles move; nothing is refitted or recomputed."""
        keys = order_apply(list(self._cards), order)
        if keys == list(self._cards):
            return
        self._cards = {k: self._cards[k] for k in keys}
        for card in self._cards.values():
            self._grid.removeWidget(card)
        for i, card in enumerate(self._cards.values()):
            self._grid.addWidget(card, i + 1, 0)
        self.overview.set_tiles(keys, self._labels)
        self._refresh_overview()

    def set_running(self, on: bool) -> None:
        self.placeholder.set_running(on)

    def set_colours(self, colours: dict) -> None:
        self._colours.update(colours)
        for key, card in self._cards.items():
            if key in colours:
                card.set_colour(colours[key])
        self._refresh_overview()

    # -- the All conditions view -------------------------------------------------------- #
    def _view_changed(self, index: int) -> None:
        overview = index == 1
        self._cards_host.setVisible(not overview)
        self.overview.setVisible(overview)
        if overview:
            self._refresh_overview()

    def set_view(self, index: int) -> None:
        self.view.select(index)
        self._view_changed(index)

    def show_everywhere(self, show) -> None:
        """Every card to the counts ``show`` names, as picked by hand (`_JumpBody.set_shown`: a card shows those of
        them it has, one with none keeps its own)."""
        for card in self._cards.values():
            card.jump_body.set_shown(show)
        self._refresh_overview()

    def components_everywhere(self, on: bool) -> None:
        """Every card's Components box to ``on``."""
        for card in self._cards.values():
            card.jump_body.set_components(on)
        self._refresh_overview()

    def _keep_place(self, chart: tuple, place) -> None:
        if place is None:
            self._legend_places.pop(chart, None)
        else:
            self._legend_places[chart] = tuple(float(v) for v in place)

    def _tile_legend_moved(self, key, place) -> None:
        self._keep_place(("fit", *key) if isinstance(key, tuple) else ("tile", key), place)
        self._refresh_overview()

    def legend_places(self) -> dict:
        """Where every chart's key was put (plain tuples, for a session)."""
        return dict(self._legend_places)

    def set_legend_places(self, places: dict) -> None:
        """Put every chart's key back where ``places`` says (a session opening)."""
        self._legend_places = {tuple(k): tuple(float(v) for v in p) for k, p in (places or {}).items() if p}
        for key, card in self._cards.items():
            card.jump_body.set_legend_at(self._legend_places.get(("card", key)))
        self._refresh_overview()

    def residuals_shown(self) -> dict:
        """The All conditions view's residual plots, ``{"pearson": bool, "ecdf": bool}``: where Export all's
        "Pearson residuals" and "ECDF residuals" boxes start (the window's target)."""
        on = self.overview.residuals()
        return {name: name in on for name in RESIDUAL_PLOTS}

    def overview_settings(self) -> dict:
        """How the All conditions view is laid out, for Export all (`figure_export`'s "jumps_overview"):
        ``{"layout", "order", "same_y", "axes"}`` -- "axes" the head's typed ranges for every tile."""
        return {"layout": self.overview.layout_mode(), "order": self.overview.order(), "same_y": self._same_y,
                "axes": self._axes.get(("all",)) or {"x": None, "y": None}, "columns": self.overview.columns()}

    # -- the y axis, typed ranges, one y scale ------------------------------------------------------------------------ #
    def _keep_axes(self, chart: tuple, axes) -> None:
        axes = check_axes(axes or {})
        if axes["x"] is None and axes["y"] is None:
            self._axes.pop(chart, None)
        else:
            self._axes[chart] = axes

    def _unit_changed(self, key) -> None:
        """A card's y unit changed: its tiles' typed y ranges, and the head's, meant the old unit -- dropped."""
        for chart in list(self._axes):
            if chart == ("all",) or (len(chart) >= 2 and chart[0] in ("tile", "fit") and chart[1] == key):
                self._keep_axes(chart, {**self._axes[chart], "y": None})
        self._refresh_overview()

    def y_mode_everywhere(self, mode: str) -> None:
        """Every card's y axis in ``mode`` (the All conditions head)."""
        for card in self._cards.values():
            card.jump_body.set_y_mode(mode)
        self._refresh_overview()

    def same_y(self) -> bool:
        return self._same_y

    def toggle_same_y(self) -> None:
        """"Same y scale" pressed: one y range for every chart of one unit, or each its own again."""
        self._same_y = not self._same_y
        for button in (self.cards_same_y_button, self.overview.same_y_button):
            button.set_text("Own y scales" if self._same_y else "Same y scale")
        self._refresh_overview()

    def _refresh_tops(self) -> None:
        """"Same y scale" on the cards: each card's top the largest automatic one among the cards of its unit
        (`jumps_view.shared_tops`); off, its own."""
        tops = {}
        for key, card in self._cards.items():
            payload = card.jump_body.histogram_payload()
            if payload is not None:
                mode = as_y_mode(payload["y_mode"])
                bars, fits = in_units(mode, payload["observed"], payload["fits"], payload["edges"])
                tops[key] = (mode, histogram_top(bars, fits, drawn_fits(payload["fits"], payload["show"])))
        shared = shared_tops(tops) if self._same_y else {}
        for key, card in self._cards.items():
            card.jump_body.set_shared_top(shared.get(key))

    def axes_of(self, chart: tuple) -> dict:
        """A chart's typed ranges: a tile's own, else the head's (every tile); a card's own."""
        own = self._axes.get(chart)
        if own is not None or chart[0] == "card":
            return own or {"x": None, "y": None}
        return self._axes.get(("all",)) or {"x": None, "y": None}

    def _tile_axes_requested(self, key) -> None:
        chart = ("fit", *key) if isinstance(key, tuple) else ("tile", key)
        tile = self.overview.tile_widget(key)
        self.open_axes(chart, None if tile is None else tile._inspect)

    def open_axes(self, chart: tuple, anchor=None) -> None:
        """The Axes popover for a tile (``("tile", key)``, ``("fit", key, count)``) or every tile (``("all",)``)."""
        if self._popover is None:
            self._popover = _AxesPopover(self.t, self)
            self._popover.applied.connect(lambda axes: self.set_axes(self._popover_chart, axes))
        self._popover_chart = chart
        key = chart[1] if len(chart) > 1 else next(iter(self._cards), None)
        card = self._cards.get(key)
        result = None if card is None else card.result()
        unit = y_label(card.jump_body.y_mode() if card else DEFAULT_Y_MODE,
                       [bin_width(result["edges_nm"])] if result else [])
        self._popover.show_for(axis_rows("jumps", unit=unit), self._axes.get(chart) or {}, anchor,
                               note="Empty: automatic. A new y unit sets y back to automatic.")

    def set_axes(self, chart: tuple, axes) -> None:
        """Typed ranges for ``chart`` (a card's go to its card)."""
        if chart[0] == "card" and chart[1] in self._cards:
            self._cards[chart[1]].jump_body.set_axes(axes)
            return
        self._keep_axes(chart, axes)
        self._refresh_overview()

    def view_state(self) -> dict:
        """What a session keeps of this page's view: where the keys are, the typed ranges, each card's unit,
        "Same y scale" and the All conditions view's residual plots."""
        return {"legend_places": dict(self._legend_places), "axes": dict(self._axes), "same_y": self._same_y,
                "y_modes": {key: card.jump_body.y_mode() for key, card in self._cards.items()},
                "residuals": list(self.overview.residuals())}

    def set_view_state(self, state: dict) -> None:
        """A session's view state back (`view_state`; anything missing keeps its standard)."""
        state = state or {}
        for key, mode in (state.get("y_modes") or {}).items():
            if key in self._cards:
                self._cards[key].jump_body.set_y_mode(mode)
        for chart, axes in (state.get("axes") or {}).items():
            self.set_axes(tuple(chart), axes)
        if bool(state.get("same_y")) != self._same_y:
            self.toggle_same_y()
        if state.get("legend_places"):
            self.set_legend_places(state["legend_places"])
        if "residuals" in state:
            self.overview.set_residuals([n for n in state["residuals"] or () if n in RESIDUAL_PLOTS])
        self._refresh_overview()

    def _refresh_overview(self) -> None:
        """Every tile as its card draws it, on the widest card's x range; the switch offers every count made. One plot
        per fit: a tile per condition and fit it shows, each condition's tiles on one y range ("Same y scale": every
        tile of one unit on one). Each tile on its own typed ranges, else the head's."""
        self._refresh_tops()
        payloads = {key: card.jump_body.histogram_payload() for key, card in self._cards.items()}
        made = [p for p in payloads.values() if p is not None]
        x_end = max((p["edges"][-1] for p in made), default=None)
        # THE RESIDUAL PLOTS (12 Nov 2026): with the ECDF's on, every tile's x range reaches its longest point (they
        # run a little past the last bin), as Export all's; "Same y scale" puts every plot of a kind on one range
        strips_on = self.overview.residuals()
        if "ecdf" in strips_on and made:
            x_end = max([x_end] + [residual_end(p["fits"], drawn_fits(p["fits"], p["show"])) for p in made])
        spans = (residual_spans((p["fits"], drawn_fits(p["fits"], p["show"])) for p in made)
                 if strips_on and self._same_y else {})

        def strips(pl, show, axes):
            if not strips_on:
                return None
            return {"edges": pl["edges"], "fits": pl["fits"], "show": tuple(show), "colour": pl["colour"],
                    "axes": {"x": (axes or {}).get("x")}, "x_end": x_end, "span": spans}
        counts = sorted({int(k) for p in made for k in p["fits"]})
        if counts:
            self.overview.set_choices(counts)
        shown = {tuple(p["show"]) for p in made}
        if shown:
            # lit as every card shows it; nothing lit while they differ
            self.overview.mark(next(iter(shown)) if len(shown) == 1 else ())
            self.overview.mark_components(all(p["components"] for p in made))
            modes = {p["y_mode"] for p in made}
            self.overview.mark_y_mode(next(iter(modes)) if len(modes) == 1 else None)
        layout = self.overview.layout_mode()
        self.overview.show_layout(layout)
        if layout == "condition":
            for key, payload in payloads.items():
                result = self._cards[key].result()
                axes = self.axes_of(("tile", key))
                self.overview.set_tile(key, None if payload is None else
                                       {**payload, "x_end": x_end, "legend_at": self._legend_places.get(("tile", key)),
                                        "axes": axes},
                                       caption="" if result is None else f"{result['n_steps']:,} jumps · "
                                       f"{result['n_tracks']:,} tracks · {show_words(payload['show'])}",
                                       strips=None if payload is None else strips(payload, payload["show"], axes))
            return
        ready = [(key, p["show"], p["fits"]) for key, p in payloads.items() if p is not None]
        arrangement = overview_tiles(ready, "fit", self.overview.order())
        units = {}
        for key, _show, _fits in ready:
            pl = payloads[key]
            mode = as_y_mode(pl["y_mode"])
            bars, scaled = in_units(mode, pl["observed"], pl["fits"], pl["edges"])
            units[key] = (mode, histogram_top(bars, scaled, drawn_fits(pl["fits"], pl["show"])))
        tops = shared_tops(units) if self._same_y else {key: top for key, (_m, top) in units.items()}
        tiles = []
        for spot in arrangement["tiles"]:
            key, (k,) = spot["key"], spot["show"]
            result = self._cards[key].result()
            axes = self.axes_of(("fit", key, k))
            tiles.append({"key": key, "count": k, "row": spot["row"], "column": spot["column"],
                          "title": f"{self._labels.get(key, key)} · {show_words(k)}",
                          "payload": {**payloads[key], "show": (k,), "x_end": x_end, "y_end": tops[key],
                                      "legend_at": self._legend_places.get(("fit", key, k)),
                                      "axes": axes},
                          "strips": strips(payloads[key], (k,), axes),
                          "caption": f"{result['n_steps']:,} jumps"})
        self.overview.set_fit_tiles(tiles)

    def set_payload(self, condition: str, result: dict) -> None:
        card = self._cards.get(condition)
        if card is None:
            return
        if "error" in result:
            card.set_error(result["error"])
        else:
            card.set_result(result, self._colours.get(condition, _DATA))
        self._ever_shown = True
        self._pages.setCurrentIndex(1)

    def set_error(self, condition: str, message: str) -> None:
        card = self._cards.get(condition)
        if card is not None:
            card.set_error(message)
        if self._cards:
            self._ever_shown = True
            self._pages.setCurrentIndex(1)

    def has_result(self) -> bool:
        return self._ever_shown

    def cards(self) -> dict:
        return dict(self._cards)

    def export_all_entries(self) -> list:
        """What Export all draws (`figure_export`'s "jumps_overview"): every card with a result, in order."""
        return [e for key in self._cards for e in self.export_entries(key)]

    def export_entries(self, key: str) -> list:
        """What the card ``key``'s Export draws: ``[{"key", "name", "colour", "result",
        "selected", "components", "legend_at", "y_mode", "y_end", "axes"}]`` -- ``selected`` the
        counts the card's Show has on (a tuple), which the figure draws unless its own dialog's Fit
        row sets others, ``components`` its Components box, ``legend_at`` where its key was put
        (None: its corner; the export then keeps its own key above the plot), ``y_mode`` its y
        unit, ``y_end`` the top it shares under "Same y scale" (None: its own), ``axes`` its
        typed ranges -- or ``[]`` without a result."""
        card = self._cards.get(key)
        result = None if card is None else card.result()
        if result is None:
            return []
        return [{"key": key, "name": self._labels.get(key, key),
                 "colour": self._colours.get(key, _DATA), "result": result,
                 "selected": card.jump_body.shown(), "components": card.jump_body.components(),
                 "legend_at": card.jump_body.legend_at(), "y_mode": card.jump_body.y_mode(),
                 "y_end": card.jump_body._shared_top, "axes": card.jump_body.axes()}]
