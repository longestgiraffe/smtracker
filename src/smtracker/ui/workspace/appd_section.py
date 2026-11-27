"""The Apparent Diffusion section, rebuilt (16 Oct 2026): the Qt shell. Every number comes from `appd_view`
(`appd_condition`, one job per condition, and the page helpers `histogram_payload`, `cells_payload`,
`switching_payload`, `length_payload`, `day_bubbles`, `gallery_lines`); nothing is computed here.

Four pages behind a strip, as SQD's (the sidebar's sub-views follow it):

* **Populations** -- the SQD bubble card itself (`BubbleCard`, its table, copy and export), fed the gamma mixture's
  bubbles; a bubble opens that condition's replicate days (`_DayPage`), from which the SQD/HMM cross-checks page
  (`appd_chart.ApparentDDetail`) is still one button away.
* **Distributions** -- per condition the histogram of per-track apparent D on a log axis with each population's
  expected counts, the cut where they cross, what an immobile molecule reads and the detection ceiling, BIC over
  1-3, and the **Track gallery**: drag over the histogram and the tracks in that range are listed, longest first,
  each a click from Images & tracks.
* **Per cell** -- every cell's slow share as a dot, by day, with the day means: is the slow population everywhere?
* **Checks** -- per condition the **Switching check** (the histogram SQD's populations predict without switching,
  beside the observed one) and the **Length check** (slow share and apparent D by track length).

Typed ranges (30 Oct 2026, `smtracker.axes_view`, the shared popover): the populations card's D (its own header
button), each Distributions card's apparent D (x, log) and tracks (y) through its Axes button; the section keeps them
by condition through a new Run until Auto (`dist_axes`, `set_dist_axes`, `axes_state`).
"""
from __future__ import annotations

from ...axes_view import apply_range, axis_rows, check
from ...axes_view import empty as axes_empty
from ..widgets.axes_popover import AxesPopover

import math

from qtpy.QtCore import QPointF, QRectF, Qt, Signal
from qtpy.QtGui import QColor, QFontMetricsF, QPainter, QPen
from qtpy.QtWidgets import QHBoxLayout, QScrollArea, QStackedWidget, QVBoxLayout, QWidget

from ... import figstyle
from ...appd_fits import (
    BEST_MAX_POP,
    DEFAULTS as SECTION_DEFAULTS,
    HELP as SECTION_HELP,
    MODEL_LABELS,
    PRECISION_LABELS,
    PRECISION_MODES,
    SECTION_MIN_STEPS,
    SECTION_MODELS,
    SECTION_UNCERTAINTY,
    SECTION_UNCERTAINTY_LABELS,
    AppDRunError,
    bubble_plot,
    day_bubbles,
    models_of,
    run_config,
)
from ...appd_view import (
    CELL_SHARE_LABELS,
    CELL_SHARE_SOURCES,
    cell_length_payload,
    bic_text,
    cells_payload,
    fit_text,
    gallery_lines,
    length_payload,
    switching_payload,
)
from ...workspace_view import (
    GROUP_PREFIX,
    ROW_SHARED_D,
    SHARED_D_NAME,
    SHARED_D_ORDER,
    SHARED_D_PILLS,
    every_models,
    factor_order,
    group_label,
    is_set_wide,
    set_wide_models,
    set_wide_note,
    toggle_every_model,
    toggle_model,
)
from ...bubble_view import BubblePlotData, BubblePoint, column_colours
# the Distributions page at the Jump distances page's level (3 Nov 2026): its rules are `appd_view`'s and `jumps_view`'s
from ...appd_view import (default_show, distribution_payload, pooled_note, quality_table, quality_verdict,
                          show_choices)
from ...jumps_view import fit_colour, show_words, toggle_show
from ..theme import typography as ty
from ..widgets.cardgrid import CardGrid
from ..widgets.controls import Button, NumberField, Segmented, Toggles
from ..widgets.primitives import Checkbox, discard, fit_label, label, mix
from ..widgets.wraprow import WrapRow
from .bubble_chart import BubbleCard, BubbleChart
from .jumps_chart import _STYLES, _Table, _dark
from .condition_key import ConditionKey
from .qc_chart import _DATA, _FLAG, _PanelCard
from .run_dialog import FitRow, FitsPage, RunDialog
from .section_placeholder import SectionPlaceholder
from .toolbar import rem, text_px

__all__ = ["AppDSection", "AppDRunDialog", "PAGES", "PAGE_LABELS"]

#: The strip's pages, in order; the sidebar's sub-views are ``appd.<page>``.
PAGES = ("populations", "distributions", "cells", "checks")
PAGE_LABELS = ("Populations", "Distributions", "Per cell", "Checks")
#: The minimum track length the dialog offers, as points (= steps + 1), what a user counts.
_LENGTH_LABELS = tuple(f"{n + 1} points" for n in SECTION_MIN_STEPS)
#: A Distributions card's Show pills, said (3 Nov 2026).
_SHOW_TIP = ("Click a count to turn its fit on or off; the last one stays on. One fit is drawn in the "
             "condition's colour with its populations; two or more use their own colours and line styles.")
#: The Distributions page's views: a card per condition, or every condition's histogram on one page.
DIST_VIEWS = ("Each condition", "All conditions")


def _bubble_data(payload: dict) -> BubblePlotData:
    """The chart's own data classes, CONSTRUCTED from a plain bubble payload (nothing calculated)."""
    return BubblePlotData(
        conditions=payload["conditions"], points=[BubblePoint(**pt) for pt in payload["points"]],
        errors=payload["errors"], series_labels=[], scope=payload["scope"],
        fraction_weighting=payload["fraction_weighting"], uncertainty=payload.get("uncertainty", ""),
        unit_values=payload.get("unit_values", {}), warnings=list(payload.get("warnings") or ()),
        samples=dict(payload.get("samples") or {}))


def _scroll(widget: QWidget) -> QScrollArea:
    host = QWidget()
    host.setStyleSheet("background:transparent;")
    lay = QVBoxLayout(host)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.addWidget(widget)
    lay.addStretch(1)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QScrollArea.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    scroll.setStyleSheet("QScrollArea { background:transparent; }")
    scroll.setWidget(host)
    return scroll


# =====================================================================================================================
# The charts: painting only
# =====================================================================================================================

class _LogHistogram(QWidget):
    """Counts on a log apparent-D axis: the observed bars, then either a fit (each population dashed, their sum
    solid, the cut, what an immobile molecule reads, the ceiling) or the Switching check (the no-switching
    prediction and the band between the states). A drag picks a range of D (`range_selected`); a click picks the
    bar under it."""

    range_selected = Signal(float, float)

    def __init__(self, t, parent=None, *, height: int = 210):
        super().__init__(parent)
        self.t = t
        self.setMinimumHeight(height)
        self.setMouseTracking(True)
        self._payload: dict | None = None
        self._colour = _DATA
        self._mode = "fit"
        self._selection: tuple | None = None
        self._drag_from: float | None = None
        self._font = ty.ui("xs", tabular=True, px=text_px(0.75))
        #: Typed ranges (the card's Axes, `axes_view`): ``{"x": (from, to) or None, "y": ...}``; the ranges as last
        #: drawn.
        self.axes: dict = {}
        self.drawn: dict = {}

    def set_payload(self, payload: dict | None, colour: str, mode: str = "fit") -> None:
        self._payload, self._colour, self._mode = payload, colour, mode
        self._selection = None
        self.update()

    def set_colour(self, colour: str) -> None:
        self._colour = colour
        self.update()

    def selection(self) -> tuple | None:
        return self._selection

    # -- geometry --------------------------------------------------------------------------------------------- #
    def _box(self) -> QRectF:
        # under the plot TWO rows of text, the decades and the axis title (found by the overnight tester, 4 Nov 2026:
        # with 32 px left the title was drawn over the Pearson strip under the histogram)
        below = 2.0 * QFontMetricsF(self._font).height() + 8.0
        return QRectF(44.0, 14.0, max(10.0, self.width() - 56.0), max(10.0, self.height() - 14.0 - below))

    def _xr(self) -> tuple:
        """The D axis's ends: the bins', or a typed range's (30 Oct 2026)."""
        edges = self._payload["edges"]
        return apply_range(edges[0], edges[-1], (self.axes or {}).get("x"), log=True)

    def _yr(self) -> tuple:
        """The count axis's ends: 0 and the payload's top, or a typed range's."""
        return apply_range(0.0, self._payload["y_max"], (self.axes or {}).get("y"))

    def _x(self, d: float) -> float:
        box, (a, b) = self._box(), self._xr()
        lo, hi = math.log10(a), math.log10(b)
        return box.left() + (math.log10(max(d, 1e-12)) - lo) / (hi - lo) * box.width()

    def _d(self, x: float) -> float:
        box, (a, b) = self._box(), self._xr()
        lo, hi = math.log10(a), math.log10(b)
        frac = min(1.0, max(0.0, (x - box.left()) / box.width()))
        return 10 ** (lo + frac * (hi - lo))

    def _y(self, v: float) -> float:
        box, (lo, hi) = self._box(), self._yr()
        return box.bottom() - min(1.0, max(0.0, (v - lo) / (hi - lo))) * box.height()

    # -- painting ---------------------------------------------------------------------------------------------- #
    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        payload = self._payload
        if not payload:
            p.end()
            return
        box, edges = self._box(), payload["edges"]
        x_lo, x_hi = self._xr()
        self.drawn = {"x": (x_lo, x_hi), "y": self._yr()}
        p.setFont(self._font)
        fm = QFontMetricsF(self._font)
        muted = QColor(self.t.muted_foreground)
        p.save()
        p.setClipRect(box.adjusted(0, -1, 0, 1))           # a typed range: nothing drawn beyond the plot
        # a band (the Switching check's "between the states") and a picked range sit under everything
        band = payload.get("band")
        if band:
            shade = QColor(self.t.accent)
            p.fillRect(QRectF(self._x(band[0]), box.top(), self._x(band[1]) - self._x(band[0]), box.height()), shade)
        if self._selection:
            a, b = self._selection
            pick = QColor(self._colour)
            pick.setAlphaF(0.14)
            p.fillRect(QRectF(self._x(a), box.top(), self._x(b) - self._x(a), box.height()), pick)
        # the observed bars
        bar = QColor(mix(self._colour, self.t.card, 0.45))
        p.setPen(Qt.NoPen)
        p.setBrush(bar)
        for (a, b), count in zip(zip(edges, edges[1:]), payload["observed"]):
            if count:
                top = self._y(count)
                p.drawRect(QRectF(self._x(a) + 0.5, top, max(0.5, self._x(b) - self._x(a) - 1.0), box.bottom() - top))
        # the curves, through the bin centres
        centres = payload["centres"]

        def curve(values, colour, width, dashes=None, style=None):
            pen = QPen(QColor(colour))
            pen.setWidthF(width)
            if style is not None:
                pen.setStyle(style)
            if dashes:
                pen.setDashPattern(dashes)
            p.setPen(pen)
            pts = [None if v is None else QPointF(self._x(c), self._y(v)) for c, v in zip(centres, values)]
            for u, v in zip(pts, pts[1:]):
                if u is not None and v is not None:         # a gap where a population carries nothing
                    p.drawLine(u, v)

        ink = figstyle.day_shade(self._colour, 0, 3)
        key = []
        if self._mode == "fit":
            # THE FITS SHOW DRAWS (`appd_view.distribution_payload`, 3 Nov 2026): one alone in the condition's ink with
            # its populations in its shades; two or more each in its own colour and v2's line style (`jumps_view`'s),
            # their populations thin and dashed in that colour when Components is on
            parts = set(payload.get("parts") or ())
            for k in payload.get("show") or ():
                fit = payload["fits"][k]
                comps = fit["components_drawn"]
                if payload.get("every"):
                    colour = fit_colour(k, _dark(self.t))
                    if k in parts:
                        for values in comps:
                            curve(values, colour, 1.2, [3.0, 2.2])
                    curve(fit["expected"], colour, 2.2, style=_STYLES.get(k, Qt.SolidLine))
                    key.append((colour, _STYLES.get(k, Qt.SolidLine), show_words(k)))
                    continue
                curve(fit["expected"], ink, 2.2)
                if k in parts:                               # one population is its own sum
                    for c, values in enumerate(comps):
                        curve(values, figstyle.day_shade(self._colour, c, len(comps)), 1.4,
                              [4, 3] if c % 2 == 0 else [1.5, 2.5])
        elif payload.get("predicted"):
            curve(payload["predicted"], self.t.foreground, 2.0, [5, 3])
        p.restore()
        if key:
            # which colour is which fit, in the plot's top right corner over a card-coloured ground
            row_h = fm.height() + 2.0
            wide = max(fm.horizontalAdvance(text) for _c, _s, text in key) + 30.0
            panel = QRectF(box.right() - wide - 6.0, box.top() + 4.0, wide + 4.0, row_h * len(key) + 4.0)
            ground = QColor(self.t.card)
            ground.setAlphaF(0.85)
            p.fillRect(panel, ground)
            for i, (colour, style, text) in enumerate(key):
                y = panel.top() + 2.0 + row_h * (i + 0.5)
                pen = QPen(QColor(colour))
                pen.setWidthF(2.2)
                pen.setStyle(style)
                p.setPen(pen)
                p.drawLine(QPointF(panel.left() + 4.0, y), QPointF(panel.left() + 22.0, y))
                p.setPen(QColor(self.t.foreground))
                p.drawText(QRectF(panel.left() + 26.0, y - row_h / 2, wide, row_h), Qt.AlignLeft | Qt.AlignVCenter,
                           text)
        # the marks
        def mark(d, text, dashes, colour):
            if d is None or not x_lo <= d <= x_hi:
                return
            pen = QPen(QColor(colour))
            pen.setWidthF(1.2)
            pen.setDashPattern(dashes)
            p.setPen(pen)
            x = self._x(d)
            p.drawLine(QPointF(x, box.top()), QPointF(x, box.bottom()))
            p.setPen(QColor(colour))
            p.drawText(QRectF(x + 3, box.top() - 2, 140, fm.height()), Qt.AlignLeft | Qt.AlignTop, text)

        for cut in payload.get("cuts") or []:
            mark(cut, f"cut {cut:.3g}", [4, 3], self.t.foreground)
        mark(payload.get("immobile"), "immobile", [1.5, 2.5], muted.name())
        mark(payload.get("ceiling"), "ceiling", [1.5, 2.5], muted.name())
        # the axes: decades along the bottom, counts up the side
        axis = QPen(QColor(self.t.border))
        axis.setWidthF(1.0)
        p.setPen(axis)
        p.drawLine(QPointF(box.left(), box.bottom()), QPointF(box.right(), box.bottom()))
        p.setPen(muted)
        for k in range(math.floor(math.log10(x_lo)), math.ceil(math.log10(x_hi)) + 1):
            d = 10.0 ** k
            if x_lo <= d <= x_hi:
                x = self._x(d)
                p.drawLine(QPointF(x, box.bottom()), QPointF(x, box.bottom() + 3))
                p.drawText(QRectF(x - 30, box.bottom() + 4, 60, fm.height()), Qt.AlignHCenter | Qt.AlignTop, f"{d:g}")
        p.drawText(QRectF(box.left(), box.bottom() + 4 + fm.height(), box.width(), fm.height()),
                   Qt.AlignRight | Qt.AlignTop, "apparent D, µm²/s")
        y_lo, y_hi = self._yr()
        top = payload["y_max"] / 1.08
        ticks = (0.0, top / 2.0, top) if (self.axes or {}).get("y") is None else (y_lo, (y_lo + y_hi) / 2.0, y_hi)
        for v in ticks:
            y = self._y(v)
            p.drawText(QRectF(0, y - fm.height() / 2, box.left() - 6, fm.height()), Qt.AlignRight | Qt.AlignVCenter,
                       f"{v:,.0f}")
        p.save()
        p.translate(10, box.center().y())
        p.rotate(-90)
        p.drawText(QRectF(-40, -fm.height() / 2, 80, fm.height()), Qt.AlignCenter, "tracks")
        p.restore()
        p.end()

    # -- picking a range ---------------------------------------------------------------------------------------- #
    def mousePressEvent(self, e) -> None:
        if self._payload and e.button() == Qt.LeftButton:
            self._drag_from = e.position().x()

    def mouseMoveEvent(self, e) -> None:
        if self._payload and self._drag_from is not None:
            a, b = sorted((self._drag_from, e.position().x()))
            self._selection = (self._d(a), self._d(b))
            self.update()

    def mouseReleaseEvent(self, e) -> None:
        if not self._payload or self._drag_from is None:
            return
        a, b = sorted((self._drag_from, e.position().x()))
        self._drag_from = None
        if b - a < 3:                                   # a click: the bar under it
            d = self._d(e.position().x())
            edges = self._payload["edges"]
            for lo, hi in zip(edges, edges[1:]):
                if lo <= d <= hi:
                    a_d, b_d = lo, hi
                    break
            else:
                return
        else:
            a_d, b_d = self._d(a), self._d(b)
        self._selection = (a_d, b_d)
        self.update()
        self.range_selected.emit(a_d, b_d)


def _nice(value: float) -> float:
    """The largest round number (1, 2 or 5 times a power of ten) at or under ``value``: a residual plot's tick."""
    if value <= 0:
        return 1.0
    power = 10 ** math.floor(math.log10(value))
    return max(m * power for m in (1, 2, 5) if m * power <= value)


class _LogStrip(QWidget):
    """A residual plot under a Distributions histogram, on the same log apparent-D axis (3 Nov 2026): the Pearson
    residual per bin (``kind`` "pearson": one fit's as bars, beyond ±2 in the flag colour, over the ±2 band; several
    each a line in its colour and style) or the model's CDF minus the ECDF (``kind`` "ecdf": one fit in the ink,
    several in their colours), of the fits the histogram draws (`appd_view.distribution_payload`). Painting only."""

    def __init__(self, t, kind: str, height: int, parent=None):
        super().__init__(parent)
        self.t = t
        self.kind = kind
        self.setFixedHeight(height)
        self._payload: dict | None = None
        self._x_range = (1.0, 10.0)
        self._font = ty.ui("xs", tabular=True, px=text_px(0.75))

    def set_payload(self, payload: dict | None, x_range=None) -> None:
        """What the histogram draws, and its x range as drawn (so the strip lines up with it)."""
        self._payload = payload
        if x_range is not None:
            self._x_range = tuple(x_range)
        self.update()

    def _box(self) -> QRectF:
        # the histogram's left and right margins, so the bins line up
        return QRectF(44.0, 4.0, max(10.0, self.width() - 56.0), max(10.0, self.height() - 8.0))

    def _series(self) -> list:
        pl = self._payload or {}
        out = []
        for k in pl.get("show") or ():
            fit = pl["fits"][k]
            if self.kind == "pearson":
                out.append((k, list(zip(pl["centres"], fit["pearson"]))))
            else:
                out.append((k, list(fit["ecdf_residual"])))
        return out

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        pl = self._payload
        if not pl:
            p.end()
            return
        p.setFont(self._font)
        fm = QFontMetricsF(self._font)
        muted = QColor(self.t.muted_foreground)
        box = self._box()
        series = self._series()
        values = [abs(v) for _k, pts in series for _x, v in pts if v is not None]
        # the ECDF residual's ticks in percentage points: "-0.02" ran into the name up the side
        name, scale = ("Pearson", 1.0) if self.kind == "pearson" else ("ECDF, pp", 100.0)
        if not values:
            p.setPen(muted)
            p.drawText(box, Qt.AlignCenter, "ECDF residuals: run the section again to compute them"
                       if self.kind == "ecdf" else "no bin expects a track")
            p.end()
            return
        span = max(3.0 if self.kind == "pearson" else 0.005, max(values) * 1.1)
        a, b = self._x_range
        lo, hi = math.log10(max(a, 1e-12)), math.log10(max(b, a * 1.0001))

        def px(d):
            return box.left() + (math.log10(max(d, 1e-12)) - lo) / (hi - lo) * box.width()

        def py(v):
            return box.center().y() - max(-1.0, min(1.0, v / span)) * box.height() / 2.0

        p.save()
        p.setClipRect(box.adjusted(0, -1, 0, 1))
        if self.kind == "pearson":
            band = QColor(self.t.muted_foreground)
            band.setAlpha(28)
            p.fillRect(QRectF(box.left(), py(2.0), box.width(), py(-2.0) - py(2.0)), band)
        p.setPen(QPen(QColor(self.t.border), 1.0))
        p.drawLine(QPointF(box.left(), py(0.0)), QPointF(box.right(), py(0.0)))
        every = bool(pl.get("every"))
        if self.kind == "pearson" and not every and series:
            edges = pl["edges"]
            p.setPen(Qt.NoPen)
            for (e0, e1), v in zip(zip(edges, edges[1:]), pl["fits"][series[0][0]]["pearson"]):
                if v is None:
                    continue
                flag = QColor(_FLAG) if abs(v) > 2 else QColor(self.t.foreground)
                if abs(v) <= 2:
                    flag.setAlpha(150)
                p.setBrush(flag)
                top, bottom = sorted((py(v), py(0.0)))
                p.drawRect(QRectF(px(e0) + 0.5, top, max(1.0, px(e1) - px(e0) - 1.0), max(0.8, bottom - top)))
        else:
            for k, pts in series:
                pen = QPen(QColor(fit_colour(k, _dark(self.t)) if every else self.t.foreground))
                pen.setWidthF(1.9 if every else 2.2)
                if every:
                    pen.setStyle(_STYLES.get(k, Qt.SolidLine))
                p.setPen(pen)
                dots = [None if v is None else QPointF(px(x), py(v)) for x, v in pts]
                for u, v in zip(dots, dots[1:]):
                    if u is not None and v is not None:
                        p.drawLine(u, v)
        p.restore()
        # the ticks: ±2 (the Pearson band) or a round number inside the range, and 0; the name up the side
        tick = 2.0 if self.kind == "pearson" else _nice(span * scale * 0.8) / scale
        p.setPen(muted)
        for v in (tick, 0.0, -tick):
            y = py(v)
            p.drawText(QRectF(14, y - fm.height() / 2, box.left() - 20, fm.height()), Qt.AlignRight | Qt.AlignVCenter,
                       f"{v * scale:g}")
        p.save()
        p.translate(10, box.center().y())
        p.rotate(-90)
        p.drawText(QRectF(-box.height() / 2, -fm.height() / 2, box.height(), fm.height()), Qt.AlignCenter, name)
        p.restore()
        p.end()


class _LengthChart(QWidget):
    """The Length check: per track-length bin its slow share as a bar, with its tracks and median apparent D."""

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.setMinimumHeight(170)
        self._payload: dict | None = None
        self._colour = _DATA
        self._font = ty.ui("xs", tabular=True, px=text_px(0.75))

    def set_payload(self, payload: dict | None, colour: str) -> None:
        self._payload, self._colour = payload, colour
        self.update()

    def set_colour(self, colour: str) -> None:
        self._colour = colour
        self.update()

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        bins = (self._payload or {}).get("bins") or []
        if not bins:
            p.end()
            return
        p.setFont(self._font)
        fm = QFontMetricsF(self._font)
        box = QRectF(52.0, 10.0, max(10.0, self.width() - 60.0), max(10.0, self.height() - 10.0 - 3 * fm.height() - 8))
        muted = QColor(self.t.muted_foreground)
        p.setPen(muted)
        for v in (0, 50, 100):
            y = box.bottom() - v / 100.0 * box.height()
            p.drawText(QRectF(0, y - fm.height() / 2, box.left() - 6, fm.height()), Qt.AlignRight | Qt.AlignVCenter,
                       f"{v} %")
            grid = QPen(QColor(self.t.border))
            grid.setWidthF(0.8)
            p.setPen(grid)
            p.drawLine(QPointF(box.left(), y), QPointF(box.right(), y))
            p.setPen(muted)
        width = box.width() / len(bins)
        p.setPen(Qt.NoPen)
        for i, b in enumerate(bins):
            if b["slow"] is None:
                continue                        # one population: no slow share (`appd_view.ONE_POPULATION`)
            fill = QColor(self._colour)
            if b["few"]:
                fill.setAlphaF(0.3)             # too few tracks to read a share from: drawn, faded
            p.setBrush(fill)
            h = b["slow"] / 100.0 * box.height()
            p.drawRect(QRectF(box.left() + i * width + width * 0.2, box.bottom() - h, width * 0.6, h))
        p.setPen(muted)
        if not self._payload.get("available", True):
            p.drawText(box, Qt.AlignCenter, f"Not available: {self._payload['why']}")
        for i, b in enumerate(bins):
            x = box.left() + i * width
            tracks = f"{b['n']:,} track{'' if b['n'] == 1 else 's'}"
            for k, text in enumerate((b["label"], tracks, f"median {b['median']:.3g}")):
                p.drawText(QRectF(x, box.bottom() + 4 + k * fm.height(), width, fm.height()), Qt.AlignHCenter, text)
        p.end()


class _CellsChart(QWidget):
    """The Per cell SuperPlot: a column per condition, each cell a dot (area by its tracks, shade by its day),
    each day's mean an open circle, the condition's mean +- SD a line. Hovering a dot names it (`hovered`)."""

    hovered = Signal(str)

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.setMinimumHeight(300)
        self.setMouseTracking(True)
        self._payload: dict | None = None
        self._labels: dict = {}
        self._colours: dict = {}
        self._dots: list = []
        self._font = ty.ui("xs", tabular=True, px=text_px(0.75))

    def set_payload(self, payload: dict | None, labels: dict, colours: dict) -> None:
        self._payload, self._labels, self._colours = payload, dict(labels), dict(colours)
        self.update()

    def set_colours(self, colours: dict) -> None:
        self._colours.update(colours)
        self.update()

    def _grouping(self) -> dict:
        layout = (self._payload or {}).get("layout") or {}
        columns = (self._payload or {}).get("columns") or []
        return layout if layout.get("groups") and len(layout.get("x") or ()) == len(columns) else {}

    def _box(self, fm) -> QRectF:
        header = (fm.height() + 10.0) if self._grouping() else 0.0       # the group headers' row (folder designs)
        return QRectF(48.0, 10.0, max(10.0, self.width() - 56.0),
                      max(10.0, self.height() - 16.0 - 2 * fm.height() - header))

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        columns = (self._payload or {}).get("columns") or []
        self._dots = []
        if not columns:
            p.end()
            return
        p.setFont(self._font)
        fm = QFontMetricsF(self._font)
        box = self._box(fm)
        muted = QColor(self.t.muted_foreground)
        for v in (0, 25, 50, 75, 100):
            y = box.bottom() - v / 100.0 * box.height()
            grid = QPen(QColor(self.t.border))
            grid.setWidthF(0.8)
            p.setPen(grid)
            p.drawLine(QPointF(box.left(), y), QPointF(box.right(), y))
            p.setPen(muted)
            p.drawText(QRectF(0, y - fm.height() / 2, box.left() - 6, fm.height()), Qt.AlignRight | Qt.AlignVCenter,
                       f"{v} %")
        # GROUPED BY THE FACTORS (`appd_view.grouped_cells`): columns at the layout's positions, a gap between
        # groups, named by their inner values, the group's name under them
        grouping = self._grouping()
        if grouping:
            xs = [float(x) for x in grouping["x"]]
            width = box.width() / (xs[-1] - xs[0] + 1.0)
            centres = [box.left() + (x - xs[0] + 0.5) * width for x in xs]
            names = [str(v) for v in grouping["labels"]]
            line = box.bottom() + 6 + 2 * fm.height()
            for group in grouping["groups"]:
                x0 = box.left() + (group["x0"] - xs[0] + 0.08) * width
                x1 = box.left() + (group["x1"] - xs[0] + 0.92) * width
                rule = QPen(QColor(self.t.border))
                rule.setWidthF(1.0)
                p.setPen(rule)
                p.drawLine(QPointF(x0, line), QPointF(x1, line))
                p.setPen(QColor(self.t.foreground))
                p.drawText(QRectF(x0, line + 2, x1 - x0, fm.height()), Qt.AlignHCenter, str(group["label"]))
        else:
            width = box.width() / len(columns)
            centres = [box.left() + (i + 0.5) * width for i in range(len(columns))]
            names = [str(self._labels.get(c["condition"], c["condition"])) for c in columns]

        def y_of(v):
            return box.bottom() - v / 100.0 * box.height()

        for i, col in enumerate(columns):
            cx = centres[i]
            base = self._colours.get(col["condition"], _DATA)
            n_days = max(1, col["n_days"])
            if col.get("why"):
                # one population: no slow share to draw (`appd_view.ONE_POPULATION`), said in the column
                p.setPen(muted)
                p.drawText(QRectF(cx - width / 2, box.top(), width, box.height()), Qt.AlignCenter | Qt.TextWordWrap,
                           f"Not available: {col['why']}")
            for pt in col["points"]:
                r = 2.0 + 1.6 * math.sqrt(pt["n"])
                fill = QColor(figstyle.day_shade(base, pt["day_index"], n_days))
                fill.setAlphaF(0.55)
                p.setPen(Qt.NoPen)
                p.setBrush(fill)
                centre = QPointF(cx + pt["x"] * width, y_of(pt["y"]))
                p.drawEllipse(centre, r, r)
                self._dots.append((centre, r, pt["label"]))
            for m in col["day_means"]:
                pen = QPen(QColor(figstyle.day_shade(base, m["day_index"], n_days)))
                pen.setWidthF(2.0)
                p.setPen(pen)
                p.setBrush(QColor(self.t.card))
                p.drawEllipse(QPointF(cx + width * 0.34, y_of(m["y"])), 5.0, 5.0)
            if col["mean"] is not None:
                pen = QPen(QColor(self.t.foreground))
                pen.setWidthF(2.0)
                p.setPen(pen)
                y = y_of(col["mean"])
                p.drawLine(QPointF(cx - width * 0.3, y), QPointF(cx + width * 0.3, y))
                if col["sd"] is not None:
                    pen.setWidthF(1.2)
                    p.setPen(pen)
                    p.drawLine(QPointF(cx + width * 0.42, y_of(col["mean"] - col["sd"])),
                               QPointF(cx + width * 0.42, y_of(col["mean"] + col["sd"])))
            p.setPen(muted)
            p.drawText(QRectF(cx - width / 2, box.bottom() + 4, width, fm.height()), Qt.AlignHCenter, names[i])
            p.drawText(QRectF(cx - width / 2, box.bottom() + 4 + fm.height(), width, fm.height()), Qt.AlignHCenter,
                       f"{col.get('n_cells', len(col['points']))} cells")
        p.end()

    def mouseMoveEvent(self, e) -> None:
        pos = e.position()
        for centre, r, text in reversed(self._dots):
            if (pos.x() - centre.x()) ** 2 + (pos.y() - centre.y()) ** 2 <= (r + 2) ** 2:
                self.hovered.emit(text)
                return
        self.hovered.emit("")


# =====================================================================================================================
# The cards
# =====================================================================================================================

def _without(lengths: dict, hidden: set) -> dict:
    """The cell-length payload without the conditions an eye hides (their dots, lines and table rows)."""
    if not hidden or not lengths.get("available"):
        return lengths
    return {**lengths, "points": {c: v for c, v in lengths["points"].items() if c not in hidden},
            "binned": {c: v for c, v in lengths["binned"].items() if c not in hidden},
            "rows": [r for r in lengths["rows"] if r[0] not in hidden]}


class _CellLengthChart(QWidget):
    """Slow share against cell length: each cell a dot in its condition's colour (area by its tracks), each
    condition's mean per length class a line over them (`appd_view.cell_length_payload`)."""

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.setMinimumHeight(260)
        self._payload: dict | None = None
        self._colours: dict = {}
        self._labels: dict = {}
        self._font = ty.ui("xs", tabular=True, px=text_px(0.75))

    def set_payload(self, payload: dict | None, labels: dict, colours: dict) -> None:
        self._payload, self._labels, self._colours = payload, dict(labels), dict(colours)
        self.update()

    def set_colours(self, colours: dict) -> None:
        self._colours.update(colours)
        self.update()

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        pl = self._payload
        if not pl or not pl.get("available"):
            p.end()
            return
        p.setFont(self._font)
        fm = QFontMetricsF(self._font)
        box = QRectF(48.0, 22.0, max(10.0, self.width() - 60.0), max(10.0, self.height() - 26.0 - 2.4 * fm.height()))
        lo, hi = pl["x_range"]
        pad = 0.04 * ((hi - lo) or 1.0)
        lo, hi = lo - pad, hi + pad
        muted = QColor(self.t.muted_foreground)

        def px(v):
            return box.left() + (v - lo) / (hi - lo) * box.width()

        def py(v):
            return box.bottom() - v / 100.0 * box.height()

        for v in (0, 25, 50, 75, 100):
            grid = QPen(QColor(self.t.border))
            grid.setWidthF(0.8)
            p.setPen(grid)
            p.drawLine(QPointF(box.left(), py(v)), QPointF(box.right(), py(v)))
            p.setPen(muted)
            p.drawText(QRectF(0, py(v) - fm.height() / 2, box.left() - 6, fm.height()), Qt.AlignRight | Qt.AlignVCenter,
                       f"{v} %")
        step = 0.5 if hi - lo > 1.5 else 0.25
        tick = math.ceil(lo / step) * step
        while tick <= hi:
            p.drawText(QRectF(px(tick) - 30, box.bottom() + 4, 60, fm.height()), Qt.AlignHCenter, f"{tick:g}")
            tick += step
        p.drawText(QRectF(box.left(), box.bottom() + 4 + fm.height(), box.width(), fm.height()), Qt.AlignRight,
                   "cell length, µm")
        for c, pts in pl["points"].items():
            colour = QColor(self._colours.get(c, _DATA))
            fill = QColor(colour)
            fill.setAlphaF(0.45)
            p.setPen(Qt.NoPen)
            p.setBrush(fill)
            for pt in pts:
                r = 1.6 + 1.1 * math.sqrt(pt["n"])
                p.drawEllipse(QPointF(px(pt["x"]), py(pt["y"])), r, r)
        for c, line in pl["binned"].items():
            pen = QPen(QColor(self._colours.get(c, _DATA)))
            pen.setWidthF(2.4)
            p.setPen(pen)
            pts = [QPointF(px(xc), py(yc)) for xc, yc in line]
            for a, b in zip(pts, pts[1:]):
                p.drawLine(a, b)
            p.setBrush(QColor(self.t.card))
            for q in pts:
                p.drawEllipse(q, 3.4, 3.4)
        p.end()


class _TrackRow(QWidget):
    """One track of the gallery: what it is, and a button to see it in Images & tracks."""

    show_requested = Signal(str)

    def __init__(self, t, line: dict, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background:transparent;")
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 2, 0, 2)
        row.setSpacing(8)
        text = QVBoxLayout()
        text.setSpacing(0)
        text.addWidget(label(line["title"], ty.ui("xs", "medium", tabular=True), t.foreground))
        text.addWidget(label(line["detail"], ty.ui("xs", tabular=True), t.muted_foreground))
        row.addLayout(text, 1)
        self.button = Button(t, "Show", size="small", variant="outline")
        self.button.clicked.connect(lambda: self.show_requested.emit(line["uid"]))
        row.addWidget(self.button, 0, Qt.AlignVCenter)


class _DistributionBody(QWidget):
    """One condition's histogram with the fits its Show draws, the residual plots under it, the fit-quality table and
    its verdict, the notes, and the Track gallery under them.

    SINCE 3 NOV 2026 AT THE JUMP DISTANCES PAGE'S LEVEL: Show is a set of fits (`jumps_view.toggle_show`: each count
    on or off, the last one on staying on), Components draws each fit's populations, Residuals the Pearson and ECDF
    residual plots; the table compares every count by ΔBIC, the largest ECDF gap and the held-out replicates (or
    cells). Every number is `appd_view`'s (`distribution_payload`, `quality_table`, `quality_verdict`)."""

    track_requested = Signal(str)
    #: Show, Components or Residuals changed: the All conditions view draws this condition again.
    shown_changed = Signal()

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.setStyleSheet("background:transparent;")
        self._result: dict | None = None
        self._colour = _DATA
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(rem(0.4))
        # THE CONTROLS, one row that wraps on a narrow card (`WrapRow`): Show, Components, Residuals
        self._controls = QWidget()
        self._controls.setStyleSheet("background:transparent;")
        row = WrapRow(spacing=rem(0.75), line_spacing=rem(0.4))
        self._controls.setLayout(row)
        self._show_row = QHBoxLayout()
        self._show_row.setSpacing(rem(0.4))
        self._show_row.addWidget(label("Show", ty.ui("xs", "medium"), t.muted_foreground))
        #: What the pills offer (`appd_view.show_choices`), rebuilt when a result's fits differ.
        self._keys: list = [1, 2, 3]
        self.choice = Toggles(t, [str(k) for k in self._keys], on=[1], size="sm")
        self.choice.setToolTip(_SHOW_TIP)
        self.choice.pressed.connect(self._pressed)
        self._show_row.addWidget(self.choice)
        row.addLayout(self._show_row)
        #: The fits shown were picked by hand: a new result keeps them where it can.
        self._by_hand = False
        self._components = True
        self.components_box = Checkbox(t, checked=True)
        self.components_box.setToolTip("Show each fit's populations. One fit uses shades of the condition's "
                                       "colour; several fits use thin dashed lines in each fit's colour.")
        self.components_box.toggled.connect(self._components_toggled)
        boxes = QHBoxLayout()
        boxes.setSpacing(rem(0.4))
        boxes.addWidget(self.components_box)
        boxes.addWidget(label("Components", ty.ui("xs"), t.muted_foreground))
        boxes.addSpacing(rem(0.5))
        self._residuals = True
        self.residuals_box = Checkbox(t, checked=True)
        self.residuals_box.setToolTip("Show the Pearson residual per bin and the ECDF residual under the "
                                      "histogram, for the fits shown.")
        self.residuals_box.toggled.connect(self._residuals_toggled)
        boxes.addWidget(self.residuals_box)
        boxes.addWidget(label("Residuals", ty.ui("xs"), t.muted_foreground))
        row.addLayout(boxes)
        lay.addWidget(self._controls)
        self.fit_line = label("", ty.ui("xs", tabular=True), t.foreground)
        self.fit_line.setWordWrap(True)
        lay.addWidget(self.fit_line)
        self.chart = _LogHistogram(t)
        self.chart.range_selected.connect(self._on_range)
        lay.addWidget(self.chart)
        self.pearson = _LogStrip(t, "pearson", 74)
        self.ecdf = _LogStrip(t, "ecdf", 84)
        lay.addWidget(self.pearson)
        lay.addWidget(self.ecdf)
        self.table = _Table(t)
        lay.addWidget(self.table)
        #: The verdict under the table (what BIC and the held-out units prefer), or why there is no result.
        self.bic = label("", ty.ui("xs", tabular=True), t.muted_foreground)
        self.bic.setWordWrap(True)
        lay.addWidget(self.bic)
        self.notes = label("", ty.ui("xs"), t.muted_foreground)
        self.notes.setWordWrap(True)
        lay.addWidget(self.notes)
        self.gallery_head = label("Drag across the histogram (or click a bar) to list the tracks in that range.",
                                  ty.ui("xs"), t.muted_foreground)
        self.gallery_head.setWordWrap(True)
        lay.addWidget(self.gallery_head)
        self._rows_host = QWidget()
        self._rows_host.setStyleSheet("background:transparent;")
        self._rows = QVBoxLayout(self._rows_host)
        self._rows.setContentsMargins(0, 0, 0, 0)
        self._rows.setSpacing(0)
        lay.addWidget(self._rows_host)
        lay.addStretch(1)                       # a card made taller by its row neighbour keeps this at the top
        self._row_widgets: list = []

    def set_result(self, result: dict, colour: str) -> None:
        self._result, self._colour = result, colour
        self._set_choices([k for k, _w in show_choices(result)])
        if not self._by_hand or not self.shown():
            self._light(default_show(result))
        table = quality_table(result)
        self.table.set_table(table["columns"], table["rows"])
        self.table.setToolTip("\n\n".join(table["notes"]))
        self.bic.setText(quality_verdict(result) or bic_text(result))
        notes = list(result.get("notes") or [])
        if result.get("n_too_short"):
            notes.append(f"{result['n_too_short']} track(s) under {result['settings']['min_steps'] + 1} points left out")
        notes.append(pooled_note(result))
        self.notes.setText(" · ".join(notes))
        self.notes.setVisible(bool(notes))
        for w in (self._controls, self.table):
            w.setVisible(True)
        self._fill([])
        self.redraw()

    def set_error(self, text: str) -> None:
        self._result = None
        self.chart.set_payload(None, self._colour)
        for strip in (self.pearson, self.ecdf):
            strip.set_payload(None)
            strip.hide()
        self.table.set_table([], [])
        for w in (self._controls, self.table):
            w.setVisible(False)
        self.fit_line.setText("")
        self.bic.setText(text)
        self.notes.hide()
        self._fill([])

    def set_colour(self, colour: str) -> None:
        self._colour = colour
        self.chart.set_colour(colour)
        self.shown_changed.emit()

    # -- Show, Components, Residuals (the Jump distances card's rules: `jumps_view.toggle_show`) -------------- #
    def _set_choices(self, keys: list) -> None:
        """Offer ``keys`` (the fits that converged): the pills are rebuilt only when they change, keeping on the
        counts still offered."""
        if keys == self._keys or not keys:
            return
        current = self.shown()
        old = self.choice
        self.choice = Toggles(self.t, [str(k) for k in keys], size="sm",
                              on=[keys.index(k) for k in current if k in keys])
        self.choice.setToolTip(_SHOW_TIP)
        self.choice.pressed.connect(self._pressed)
        self._show_row.replaceWidget(old, self.choice)
        discard(old)
        self._keys = list(keys)

    def _light(self, counts) -> None:
        """Pills ``counts`` on, the others off (silent: `Toggles.set_on`)."""
        self.choice.set_on([self._keys.index(k) for k in counts if k in self._keys])

    def shown(self) -> tuple:
        """The counts Show has on, ascending (at least one once a result is in)."""
        return tuple(self._keys[i] for i in self.choice.on() if 0 <= i < len(self._keys))

    def set_shown(self, show) -> None:
        """Show the counts ``show`` names, as picked by hand; a card that has none of them keeps its own."""
        counts = [k for k in show if k in self._keys]
        if not counts:
            return
        self._by_hand = True
        self._light(counts)
        self.redraw()

    def _pressed(self, index: int) -> None:
        """A pill clicked: that fit on or off (the last one on stays on)."""
        if not 0 <= index < len(self._keys):
            return
        self._by_hand = True
        self._light(toggle_show(self.shown(), self._keys[index], {k: None for k in self._keys}))
        self.redraw()

    def components(self) -> bool:
        return self._components

    def residuals(self) -> bool:
        return self._residuals

    def _components_toggled(self, on: bool) -> None:
        self._components = bool(on)
        self.redraw()

    def _residuals_toggled(self, on: bool) -> None:
        self._residuals = bool(on)
        self.redraw()

    def histogram_payload(self) -> dict | None:
        """What this card's histogram draws (`appd_view.distribution_payload`; the All conditions view draws the
        same), or None without a result."""
        if self._result is None:
            return None
        return distribution_payload(self._result, self.shown() or None, components=self._components)

    def redraw(self) -> None:
        """The histogram and its residual plots on the fits shown, the strips on the histogram's x range."""
        payload = self.histogram_payload()
        if payload is None:
            return
        shown = payload["show"]
        self.fit_line.setText(f"Pooled fit · {fit_text(self._result, shown[0])}" if len(shown) == 1
                              else f"Pooled fits of every track: {show_words(shown)}")
        self.chart.set_payload(payload, self._colour)
        x_range = self.chart._xr()
        for strip in (self.pearson, self.ecdf):
            strip.set_payload(payload, x_range)
            strip.setVisible(self._residuals)
        self.shown_changed.emit()

    def _draw(self) -> None:
        self.redraw()

    def _on_range(self, lo: float, hi: float) -> None:
        if self._result is None:
            return
        found = gallery_lines(self._result, lo, hi)
        self.gallery_head.setText(
            f"{found['n']:,} track{'' if found['n'] == 1 else 's'} between {lo:.3g} and {hi:.3g} µm²/s"
            + (f"; the {len(found['lines'])} longest:" if found["n"] > len(found["lines"]) else ":"
               if found["n"] else "."))
        self._fill(found["lines"])

    def _fill(self, lines: list) -> None:
        for w in self._row_widgets:
            discard(w)
        self._row_widgets = []
        for line in lines:
            row = _TrackRow(self.t, line)
            row.show_requested.connect(self.track_requested.emit)
            self._rows.addWidget(row)
            self._row_widgets.append(row)


class _ChecksBody(QWidget):
    """One condition's Switching check and Length check."""

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.setStyleSheet("background:transparent;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(rem(0.4))
        lay.addWidget(label("Switching: observed, and SQD's populations if no molecule switched within its track",
                            ty.ui("xs", "medium"), t.foreground))
        self.switching = _LogHistogram(t, height=180)
        lay.addWidget(self.switching)
        self.switching_text = label("", ty.ui("xs"), t.muted_foreground)
        self.switching_text.setWordWrap(True)
        lay.addWidget(self.switching_text)
        lay.addSpacing(rem(0.4))
        lay.addWidget(label("Track length: the slow share by how many steps a track has", ty.ui("xs", "medium"),
                            t.foreground))
        self.length = _LengthChart(t)
        lay.addWidget(self.length)
        self.length_text = label("", ty.ui("xs"), t.muted_foreground)
        self.length_text.setWordWrap(True)
        lay.addWidget(self.length_text)
        lay.addStretch(1)

    def set_result(self, result: dict, colour: str) -> None:
        sw = switching_payload(result)
        if sw["available"]:
            self.switching.set_payload(sw, colour, mode="switching")
            self.switching.show()
            self.switching_text.setText(sw["text"])
        else:
            self.switching.set_payload(None, colour)
            self.switching.hide()
            self.switching_text.setText(f"Not available: {sw['why']}.")
        ln = length_payload(result)
        self.length.set_payload(ln, colour)
        self.length_text.setText(ln["text"])

    def set_error(self, text: str) -> None:
        self.switching.set_payload(None, _DATA)
        self.length.set_payload(None, _DATA)
        self.switching_text.setText(text)
        self.length_text.setText("")

    def set_colour(self, colour: str) -> None:
        self.switching.set_colour(colour)
        self.length.set_colour(colour)


#: What a replicate page says of how its fractions were fitted, per Force same D across (the column's design).
_DAY_NOTES = {
    "condition": ("Each of the condition's {word} has its own fractions at the condition's one D, shared over its "
                  "{word}. A difference between them comes from which tracks fell in each; the fit is the same. "
                  "Replicates are not paired across conditions."),
    "replicate": (f"Each of the condition's {{word}} is fitted alone ({SHARED_D_PILLS['replicate']}), with its own D "
                  "and fractions, so D differs between them too. Replicates are not paired across conditions."),
    "global": (f"Each of the condition's {{word}} has its own fractions at the D shared by every condition "
               f"({SHARED_D_PILLS['global']}). Replicates are not paired across conditions."),
    "group": ("Each of the condition's {word} has its own fractions at the D shared by its group's conditions "
              "(Conditions). Replicates are not paired across conditions."),
}


class _DayPage(QWidget):
    """One condition's replicate days: their bubbles (its own fractions per day at the condition's shared D) and
    the table, with the way back and the SQD/HMM cross-checks one button away."""

    back_requested = Signal()
    crosschecks_requested = Signal(str)

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.setStyleSheet("background:transparent;")
        self._condition: str | None = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(rem(0.75))
        head = QHBoxLayout()
        back = Button(t, "← All conditions", size="small", variant="outline")
        back.clicked.connect(self.back_requested.emit)
        head.addWidget(back)
        self.title = label("", ty.ui("base", "semibold"), t.foreground)
        head.addWidget(self.title)
        head.addStretch(1)
        self.cross = Button(t, "Cross-checks (SQD, HMM, saSPT)…", size="small", variant="outline")
        self.cross.clicked.connect(lambda: self._condition and self.crosschecks_requested.emit(self._condition))
        head.addWidget(self.cross)
        lay.addLayout(head)
        self.chart = BubbleChart(t)
        self.chart.setMinimumHeight(rem(18))
        lay.addWidget(self.chart)
        self.table = _Table(t)
        lay.addWidget(self.table)
        self.note = label("", ty.ui("xs"), t.muted_foreground)
        self.note.setWordWrap(True)
        lay.addWidget(self.note)
        lay.addStretch(1)

    def show_condition(self, condition: str, name: str, design: dict | None, colour: str) -> None:
        """``condition``'s column ``name`` (its name, or "glucose (2-pop)" with fits side by side): its replicates,
        or chunks, from the column's own design (`appd_fits.day_bubbles`)."""
        self._condition = condition
        payload = day_bubbles(design)
        word = "chunks" if (design or {}).get("units", {}) and design["units"]["kind"] == "chunk" else "replicates"
        self.title.setText(f"{name} · {word}")
        if payload is None:
            self.chart.set_message("No replicates to compare for this column: one replicate, or an Uncertainty that "
                                   "pools them. Run with Uncertainty Replicates (or Chunks) to see them.")
            self.table.set_table([], [])
            self.note.setText("")
            return
        self.chart.set_data(_bubble_data(payload), {label_: colour for label_ in payload["conditions"]})
        self.table.set_table(payload["columns"], payload["rows"])
        self.note.setText(_DAY_NOTES.get(design["scope"], _DAY_NOTES["condition"]).format(word=word))

    def condition(self) -> str | None:
        return self._condition


class _ModelRow(FitRow):
    """One condition's line on the Run dialog's Fits page (`run_dialog.FitRow`): its name, its models (several may be
    lit) and its own Force same D across (Replicates / Don't force anywhere)."""

    pressed = Signal(str, int)

    def __init__(self, t, key: str, name: str, parent=None):
        models = Toggles(t, [MODEL_LABELS[m] for m in SECTION_MODELS], size="sm")
        super().__init__(t, key, name, models, parent=parent)
        self.models = models
        self.models.pressed.connect(lambda i: self.pressed.emit(self.key, i))


def _mirror(tray, index: int) -> None:
    """``tray`` lit at ``index`` without saying so (the Fits page's copy of the General page's pills)."""
    if tray.current() != index:
        tray.blockSignals(True)
        tray.select(index)
        tray.blockSignals(False)


class AppDRunDialog(RunDialog):
    """The Run's settings (`appd_fits.HELP`): SQD's own, in SQD's words, on SQD's two pages (`run_dialog.RunDialog`,
    4 Nov 2026) -- General: Force same D across, Uncertainty, the shortest track and Best's range; Fits: every
    condition's models on one line each (several lit = fits side by side) and its own Force same D across, under the
    Every condition line. Every rule is headless (`appd_fits.run_config`, `workspace_view`'s toggles); a refusal is
    said above Run and nothing runs. Transactional like every Run dialog here: Cancel and the corner X discard the
    draft."""

    def __init__(self, t, *, conditions=(), labels=None, factors=None, seed: dict | None = None, content=None,
                 parent=None, factor_names=()):
        super().__init__(t, "Configure Apparent Diffusion",
                         "Every loaded condition is fitted. Nothing runs until you press Run.",
                         content=content, parent=parent)
        self._conditions = [str(c) for c in conditions]
        self._labels = dict(labels or {})
        self._factors = dict(factors or {})
        try:
            seed = run_config(seed or {}, conditions=self._conditions, factors=self._factors)
        except AppDRunError:
            seed = run_config({}, conditions=self._conditions, factors=self._factors)
        # FORCE SAME D ACROSS: SQD's pills in SQD's order, a Conditions pill per folder-design factor in the design's
        # order (`workspace_view.factor_order`)
        names = factor_order(self._factors, factor_names)
        self._shared_values = []
        for value in SHARED_D_ORDER:
            if value == "group":
                self._shared_values += [GROUP_PREFIX + f for f in names]
            else:
                self._shared_values.append(value)
        pill = {v: (group_label(v[len(GROUP_PREFIX):], len(names)) if v.startswith(GROUP_PREFIX)
                    else SHARED_D_PILLS[v]) for v in self._shared_values}
        self._pills = pill
        start = seed["shared_d"] if seed["shared_d"] in self._shared_values else "condition"
        self._shared = Segmented(t, [pill[v] for v in self._shared_values],
                                 current=self._shared_values.index(start), size="sm")
        self._shared.changed.connect(self._on_shared)
        self._uncertainty = Segmented(t, list(SECTION_UNCERTAINTY_LABELS),
                                      current=SECTION_UNCERTAINTY.index(seed["uncertainty"]), size="sm")
        self._uncertainty.changed.connect(lambda _i: self._sync())
        self._chunks = NumberField(t, 2, 20, int(seed["chunks"]), width=rem(7))
        self._chunks_label = label("chunks", ty.ui("xs"), t.muted_foreground)
        self._length = Segmented(t, list(_LENGTH_LABELS), current=SECTION_MIN_STEPS.index(int(seed["min_steps"])),
                                 size="sm")
        self._best = Segmented(t, [f"{k} populations" for k in BEST_MAX_POP],
                               current=BEST_MAX_POP.index(int(seed["best_max_pop"])), size="sm")
        # LOCALISATION ERROR (6 Nov 2026): SQD's words, Not modelled | Per spot
        self._precision = Segmented(t, list(PRECISION_LABELS), current=PRECISION_MODES.index(seed["precision"]),
                                    size="sm")
        # THE GENERAL PAGE
        for name, control, key, extra in ((SHARED_D_NAME, self._shared, "shared_d", ()),
                                          ("Uncertainty", self._uncertainty, "uncertainty",
                                           (self._chunks, self._chunks_label)),
                                          ("Shortest track", self._length, "min_steps", ()),
                                          ("Best (BIC) up to", self._best, "best_max_pop", ()),
                                          ("Localisation error", self._precision, "precision", ())):
            line, more = SECTION_HELP[key]
            self.add_setting(name, control, line, more, extra)
        # THE FITS PAGE: the Every condition line, then a line per condition
        every = Toggles(t, [MODEL_LABELS[m] for m in SECTION_MODELS], size="sm")
        line, more = SECTION_HELP["models"]
        fits = FitsPage(t, help_line=line, help_more=more, choice_title="Models", every_choice=every)
        self._every = every
        self._every.pressed.connect(self._on_every)
        self._every_label = fits.every.title
        fits.every.shared_changed.connect(self._on_every_shared)
        self._set_note = fits.note
        # FORCE SAME D ACROSS ON TOP OF THE FITS PAGE TOO, as SQD's (5 Nov 2026): the General page's pills, one value
        # in two places -- a click here selects the General page's pill, whose handler does the rest; `_sync` lights it
        self._fits_shared = Segmented(t, [pill[v] for v in self._shared_values], current=self._shared.current(),
                                      size="sm")
        self._fits_shared.changed.connect(self._shared.select)
        fits.add_set_wide(self._fits_shared)
        self._row_widgets: dict = {}
        self._models = {c: models_of(seed, c) for c in self._conditions}
        self._own = {c: str(((seed.get("rows") or {}).get(c) or {}).get("shared_d") or "") for c in self._conditions}
        for c in self._conditions:
            row = _ModelRow(t, c, str(self._labels.get(c, c)))
            row.pressed.connect(self._on_row_model)
            row.shared_changed.connect(self._on_row_shared)
            self._row_widgets[c] = row
        fits.set_rows(self._row_widgets.values())
        self.set_fits(fits)
        self._sync()

    # -- the rules, all headless -------------------------------------------------------------------------------- #
    def _shared_value(self) -> str:
        return self._shared_values[self._shared.current()]

    def _on_shared(self, _index: int) -> None:
        if is_set_wide(self._shared_value()):
            # one D across conditions: one set of models, the rows' common ones (`workspace_view.set_wide_models`)
            common = set_wide_models([self._models[c] for c in self._conditions], "best")
            self._models = {c: list(common) for c in self._conditions}
        self._sync()

    def _on_every(self, index: int) -> None:
        model = SECTION_MODELS[index]
        rows = toggle_every_model([self._models[c] for c in self._conditions], model, SECTION_MODELS)
        self._models = dict(zip(self._conditions, rows))
        if is_set_wide(self._shared_value()) and self._conditions:
            first = self._models[self._conditions[0]]
            self._models = {c: list(first) for c in self._conditions}
        self._sync()

    def _on_row_model(self, key: str, index: int) -> None:
        self._models[key] = toggle_model(self._models[key], SECTION_MODELS[index], SECTION_MODELS)
        self._sync()

    def _on_row_shared(self, key: str, index: int) -> None:
        self._own[key] = ROW_SHARED_D[index]
        self._sync()

    def _on_every_shared(self, _key: str, index: int) -> None:
        """The Every condition line's Force same D across: every row's own, as clicking each row would make it (SQD's
        `ComputePanel._set_every_shared_d`)."""
        self._own = {c: ROW_SHARED_D[index] for c in self._conditions}
        self._sync()

    def _row_shared(self, c: str, shared: str) -> str:
        """A row's own Force same D across: its pick, else the General page's (Replicates under a set-wide one)."""
        return self._own.get(c) or (shared if shared in ROW_SHARED_D else ROW_SHARED_D[0])

    def _sync(self) -> None:
        """Every control back on the draft: the rows' models and own Force same D, what a set-wide value disables,
        the chunks field, and the draft's refusal (if any) above Run."""
        shared = self._shared_value()
        _mirror(self._fits_shared, self._shared.current())
        wide = is_set_wide(shared)
        lit = every_models([self._models[c] for c in self._conditions])
        self._every.set_on([SECTION_MODELS.index(m) for m in lit])
        for c, row in self._row_widgets.items():
            row.models.set_on([SECTION_MODELS.index(m) for m in self._models[c]])
            row.models.setEnabled(not wide)
            # under a set-wide value the row SHOWS it (SQD's Fits page), never a lit Replicates of its own
            row.shared.set_wide(self._pills[shared] if wide else None)
            if not wide:
                row.shared.set_value(self._row_shared(c, shared))
            row.shared.setEnabled(not wide)
        # the Every condition line's: the set-wide value, else the rows' common one (none while they differ)
        every = self.fits.every.shared
        every.set_wide(self._pills[shared] if wide else None)
        if not wide:
            found = {self._row_shared(c, shared) for c in self._conditions}
            every.set_value(found.pop() if len(found) == 1 else None)
        every.setEnabled(not wide)
        self.fits.set_note(set_wide_note(shared))
        chunk = SECTION_UNCERTAINTY[self._uncertainty.current()] == "chunk"
        self._chunks.setVisible(chunk)
        self._chunks_label.setVisible(chunk)
        try:
            self.config()
            self.hint.setText("")
        except AppDRunError as exc:
            self.hint.setText(str(exc))

    def config(self) -> dict:
        """The draft as `appd_fits.run_config` checks it. RAISES `AppDRunError` on a refusal."""
        shared = self._shared_value()
        wide = is_set_wide(shared)
        first = self._models[self._conditions[0]] if self._conditions else ["best"]
        every = every_models([self._models[c] for c in self._conditions]) or first
        rows = {}
        for c in self._conditions:
            row = {}
            if self._models[c] != list(every):
                row["models"] = list(self._models[c])
            own = self._own.get(c)
            if not wide and own and own != shared:
                row["shared_d"] = own
            if row:
                rows[c] = row
        return run_config({"models": list(every), "shared_d": shared,
                           "uncertainty": SECTION_UNCERTAINTY[self._uncertainty.current()],
                           "min_steps": SECTION_MIN_STEPS[self._length.current()],
                           "best_max_pop": BEST_MAX_POP[self._best.current()],
                           "precision": PRECISION_MODES[self._precision.current()],
                           "chunks": int(self._chunks.value()), "rows": rows},
                          conditions=self._conditions, factors=self._factors)

    def _on_run(self) -> None:
        try:
            config = self.config()
        except AppDRunError as exc:
            self.hint.setText(str(exc))
            return
        self._acted = True
        self.run.emit(config)
        self._dismiss()


# =====================================================================================================================
# The section
# =====================================================================================================================

class AppDSection(QWidget):
    """The Apparent Diffusion sidebar section: a placeholder until the first result, then four pages behind a
    strip, filling in as conditions land (one job per condition, `appd_view.appd_condition`)."""

    run_requested = Signal()
    cancel_requested = Signal()
    #: The strip moved to page ``index`` (`PAGES`): the window moves the sidebar's sub-view.
    page_changed = Signal(int)
    #: "Map this fit" (`map_fit_source.MAP_THIS_FIT`): the window opens the Spatial Map's settings for this page's fit.
    map_requested = Signal()
    #: The Populations card's figure Export / table Export.
    export_requested = Signal()
    table_export_requested = Signal()
    #: A Distributions card's Export / a Checks card's Export, with the condition; the Per cell card's Export
    #: (asked 16 Oct 2026: "build appd export"; `figure_export`'s "appd_histogram", "appd_checks", "appd_cells").
    distribution_export_requested = Signal(str)
    checks_export_requested = Signal(str)
    cells_export_requested = Signal()
    #: The Distributions page's "Export all conditions…" (3 Nov 2026; `figure_export`'s "appd_overview").
    overview_export_requested = Signal()
    #: The cell-length card's Export (17 Oct 2026).
    cell_length_export_requested = Signal()
    #: A gallery track's Show: (condition, project uid).
    track_requested = Signal(str, str)
    #: The day page's cross-checks button, with the condition.
    crosschecks_requested = Signal(str)

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.setStyleSheet("background:transparent;")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(rem(0.75))
        self.placeholder = SectionPlaceholder(
            t, title="Apparent Diffusion",
            description="One apparent D per track, from all its steps: the populations they form, how they "
                        "spread over cells and replicates, and how switching and track length affect them.",
            icon_name="activity")
        self.placeholder.run_requested.connect(self.run_requested.emit)
        self.placeholder.cancel_requested.connect(self.cancel_requested.emit)
        self.strip = Segmented(t, list(PAGE_LABELS), current=0, size="sm")
        self.strip.changed.connect(self.page_changed.emit)
        strip_row = QHBoxLayout()
        strip_row.setContentsMargins(0, 0, 0, 0)
        strip_row.addWidget(self.strip)
        strip_row.addStretch(1)
        from ...map_fit_source import MAP_THIS_FIT, map_this_fit_tip

        self.map_button = Button(t, MAP_THIS_FIT, size="small", variant="outline", glyph="grid")
        self.map_button.setToolTip(map_this_fit_tip("appd"))
        self.map_button.clicked.connect(self.map_requested.emit)
        strip_row.addWidget(self.map_button)
        self._strip_host = QWidget()
        self._strip_host.setStyleSheet("background:transparent;")
        self._strip_host.setLayout(strip_row)
        outer.addWidget(self._strip_host)
        # ON SHOW FROM THE START (22 Nov 2026: "right now sqd is the only one with different tabs" -- hidden until the
        # first Run, the pages looked like none): before a Run every page is the placeholder

        # Populations: the SQD bubble card itself, and the days behind a bubble
        self.populations = BubbleCard(t, title="Apparent-D populations")
        self.populations.export_requested.connect(self.export_requested.emit)
        self.populations.table_export_requested.connect(self.table_export_requested.emit)
        self.populations.chart.population_clicked.connect(self.show_days)
        self.caption = label("", ty.ui("xs"), t.muted_foreground)
        self.caption.setWordWrap(True)
        #: a significance bracket standing on a population "Check the fit" flags (`significance.bracket_warnings`,
        #: ruled 4 Nov 2026), in the flag colour above the caption, hidden when there is none (SQD's `fit_label`)
        self.bracket_warning = fit_label("", ty.ui("sm", "medium"), _FLAG)
        pop_page = QWidget()
        pop_page.setStyleSheet("background:transparent;")
        pop_lay = QVBoxLayout(pop_page)
        pop_lay.setContentsMargins(0, 0, 0, 0)
        pop_lay.setSpacing(rem(0.5))
        pop_lay.addWidget(self.populations, 1)
        pop_lay.addWidget(self.bracket_warning)
        self.bracket_warning.hide()
        pop_lay.addWidget(self.caption)
        self.days = _DayPage(t)
        self.days.back_requested.connect(lambda: self._pop_stack.setCurrentIndex(0))
        self.days.crosschecks_requested.connect(self.crosschecks_requested.emit)
        self._pop_stack = QStackedWidget()
        self._pop_stack.addWidget(pop_page)
        self._pop_stack.addWidget(self.days)
        self._cross_host = QWidget()
        self._cross_host.setStyleSheet("background:transparent;")
        self._cross_lay = QVBoxLayout(self._cross_host)
        self._cross_lay.setContentsMargins(0, 0, 0, 0)
        self._pop_stack.addWidget(self._cross_host)

        # Distributions and Checks: a card per condition, as many columns as fit
        # DISTRIBUTIONS: a card per condition, or ALL CONDITIONS on one page (3 Nov 2026, the Jump distances page's
        # switch), each tile drawn as its card shows it; the Export over them writes one figure of every condition
        dist_page = QWidget()
        dist_page.setStyleSheet("background:transparent;")
        dist_lay = QVBoxLayout(dist_page)
        dist_lay.setContentsMargins(0, 0, 0, 0)
        dist_lay.setSpacing(rem(0.75))
        dist_head = QHBoxLayout()
        self.dist_view = Segmented(t, list(DIST_VIEWS), current=0, size="sm")
        self.dist_view.changed.connect(self._dist_view_changed)
        dist_head.addWidget(self.dist_view)
        dist_head.addStretch(1)
        self.overview_export = Button(t, "Export all conditions…", size="small", variant="outline", glyph="download")
        self.overview_export.setToolTip("Export every condition's histogram in one figure, drawn as on its card.")
        self.overview_export.clicked.connect(self.overview_export_requested.emit)
        dist_head.addWidget(self.overview_export)
        dist_lay.addLayout(dist_head)
        self._dist_stack = QStackedWidget()
        cards_host = QWidget()
        cards_host.setStyleSheet("background:transparent;")
        # 40 rem a card: the fit-quality table's seven columns need it (found by the overnight tester, 4 Nov 2026: at
        # 1366 px two cards of 30 rem cut it to "Populatio", "D (µm²/s"); one card a row on a laptop, two on a wide screen
        self._dist_grid = CardGrid(cards_host, min_width=rem(40), max_columns=2, spacing=rem(1.0))
        overview_host = QWidget()
        overview_host.setStyleSheet("background:transparent;")
        self._overview_grid = CardGrid(overview_host, min_width=rem(20), max_columns=3, spacing=rem(1.0))
        self._dist_stack.addWidget(cards_host)
        self._dist_stack.addWidget(overview_host)
        dist_lay.addWidget(self._dist_stack)
        #: The All conditions view's tiles, by condition: (the tile, its histogram).
        self._tiles: dict[str, tuple] = {}
        checks_page = QWidget()
        checks_page.setStyleSheet("background:transparent;")
        self._checks_grid = CardGrid(checks_page, min_width=rem(30), max_columns=2, spacing=rem(1.0))
        # Per cell: one chart over every condition
        self.cells_chart = _CellsChart(t)
        self.cells_hover = label("Hover a dot to name its cell.", ty.ui("xs", tabular=True), t.muted_foreground)
        self.cells_chart.hovered.connect(lambda text: self.cells_hover.setText(text or "Hover a dot to name its cell."))
        cells_body = QWidget()
        cells_body.setStyleSheet("background:transparent;")
        cells_lay = QVBoxLayout(cells_body)
        cells_lay.setContentsMargins(0, 0, 0, 0)
        cells_lay.addWidget(self.cells_chart)
        cells_lay.addWidget(self.cells_hover)
        self.cells_card = _PanelCard(t, "Per cell · each cell's slow share", cells_body, flush=False)
        self.cells_card.enable_export()
        self.cells_card.export_requested.connect(self.cells_export_requested.emit)
        # FRACTION AGAINST CELL LENGTH (16 Oct 2026): each cell's slow share against its length, and the conditions
        # compared at matched cell length
        self.length_chart = _CellLengthChart(t)
        self.length_source = Segmented(t, list(CELL_SHARE_LABELS), current=0, size="sm")
        self.length_source.changed.connect(lambda _i: self._draw_cell_length())
        self.length_table = _Table(t)
        self.length_text = label("", ty.ui("xs", tabular=True), t.foreground)
        length_body = QWidget()
        length_body.setStyleSheet("background:transparent;")
        length_lay = QVBoxLayout(length_body)
        length_lay.setContentsMargins(0, 0, 0, 0)
        length_lay.setSpacing(rem(0.5))
        source_row = QHBoxLayout()
        source_row.addWidget(label("Slow share from", ty.ui("xs", "medium"), t.muted_foreground))
        source_row.addWidget(self.length_source)
        source_row.addStretch(1)
        length_lay.addLayout(source_row)
        #: An eye per condition (asked 17 Oct 2026): a display filter -- the length-matched numbers stay over every
        #: condition's cells, a hidden one leaves the chart, the table and the export.
        self.length_key = ConditionKey(t)
        self.length_key.hidden_changed.connect(lambda _h: self._draw_cell_length())
        length_lay.addWidget(self.length_key)
        length_lay.addWidget(self.length_chart)
        length_lay.addWidget(self.length_table)
        length_lay.addWidget(self.length_text)
        self.cell_length_card = _PanelCard(t, "Slow share against cell length", length_body, flush=False)
        self.cell_length_card.enable_export()
        self.cell_length_card.export_requested.connect(self.cell_length_export_requested.emit)
        cells_page = QWidget()
        cells_page.setStyleSheet("background:transparent;")
        cells_page_lay = QVBoxLayout(cells_page)
        cells_page_lay.setContentsMargins(0, 0, 0, 0)
        cells_page_lay.setSpacing(rem(1.0))
        cells_page_lay.addWidget(self.cells_card)
        cells_page_lay.addWidget(self.cell_length_card)

        self._content = QStackedWidget()
        self._content.addWidget(self._pop_stack)
        self._content.addWidget(_scroll(dist_page))
        self._content.addWidget(_scroll(cells_page))
        self._content.addWidget(_scroll(checks_page))
        self._pages = QStackedWidget()
        self._pages.addWidget(self.placeholder)
        self._pages.addWidget(self._content)
        outer.addWidget(self._pages, 1)

        self._results: dict[str, dict] = {}
        self._conditions: list[str] = []
        self._labels: dict[str, str] = {}
        self._colours: dict[str, str] = {}
        #: condition -> (its card, the card's body): `_PanelCard` keeps no handle on its body.
        self._dist_cards: dict[str, tuple] = {}
        self._check_cards: dict[str, tuple] = {}
        self._bubbles: BubblePlotData | None = None
        #: What the Per cell card paints (`appd_view.cells_payload`), and so what its Export draws.
        self._cells: dict | None = None
        self._ever_shown = False
        self.last_config: dict = dict(SECTION_DEFAULTS)
        #: Each Distributions histogram's typed ranges, by condition, kept through a new Run until Auto.
        self._dist_axes: dict = {}
        self._axes_popover = None

    # -- the run ------------------------------------------------------------------------------------------------ #
    def begin_run(self, conditions: list, labels: dict, colours: dict, config: dict) -> None:
        for card, _body in (*self._dist_cards.values(), *self._check_cards.values()):
            discard(card)
        for tile, _chart in self._tiles.values():
            discard(tile)
        self._dist_cards, self._check_cards, self._results, self._tiles = {}, {}, {}, {}
        self._joint, self._payloads = None, {}
        self._conditions, self._labels, self._colours = list(conditions), dict(labels), dict(colours)
        try:
            # one form for every config, a session's from before 3 Nov 2026 included (`appd_fits.run_config`)
            self.last_config = run_config(config or {}, conditions=self._conditions)
        except AppDRunError:
            self.last_config = dict(config or {})
        for key in conditions:
            name = self._labels.get(key, key)
            body = _DistributionBody(self.t)
            body.track_requested.connect(lambda uid, k=key: self.track_requested.emit(k, uid))
            body.shown_changed.connect(lambda k=key: self._draw_tile(k))
            # ALL CONDITIONS: this condition's tile, drawn as its card shows it
            tile = QWidget()
            tile.setStyleSheet("background:transparent;")
            tile_lay = QVBoxLayout(tile)
            tile_lay.setContentsMargins(0, 0, 0, 0)
            tile_lay.setSpacing(rem(0.25))
            tile_lay.addWidget(label(name, ty.ui("sm", "medium"), self.t.foreground))
            tile_chart = _LogHistogram(self.t, height=190)
            tile_lay.addWidget(tile_chart)
            self._tiles[key] = (tile, tile_chart)
            self._overview_grid.addWidget(tile)
            card = _PanelCard(self.t, f"{name} · apparent D", body, flush=False)
            card.set_caption("computing…")
            card.enable_export()
            card.export_requested.connect(lambda k=key: self.distribution_export_requested.emit(k))
            # THE AXES (30 Oct 2026): the histogram's D (x, log) and track count (y), typed; kept through a new Run
            card.enable_inspect("Axes", glyph="ruler")
            card.inspect_requested.connect(lambda k=key: self.open_dist_axes(k))
            body.chart.axes = dict(self._dist_axes.get(key) or {})
            self._dist_cards[key] = (card, body)
            self._dist_grid.addWidget(card)
            checks_body = _ChecksBody(self.t)
            check = _PanelCard(self.t, f"{name} · checks", checks_body, flush=False)
            check.set_caption("computing…")
            check.enable_export()
            check.export_requested.connect(lambda k=key: self.checks_export_requested.emit(k))
            self._check_cards[key] = (check, checks_body)
            self._checks_grid.addWidget(check)
        self.placeholder.set_running(True)
        self._pop_stack.setCurrentIndex(0)

    def set_order(self, order) -> None:
        """The conditions in ``order`` (`condition_order`, the Conditions tool's arrows, 3 Nov 2026): the bubbles, the
        Per cell columns, the Distributions and Checks cards move; nothing is refitted or recomputed."""
        from ...bubble_view import reordered
        from ...condition_order import apply, sort_by

        self._conditions = apply(self._conditions, order)
        self._dist_cards = {k: self._dist_cards[k] for k in apply(list(self._dist_cards), order)}
        self._check_cards = {k: self._check_cards[k] for k in apply(list(self._check_cards), order)}
        self._dist_grid.reorder([card for card, _body in self._dist_cards.values()])
        self._checks_grid.reorder([card for card, _body in self._check_cards.values()])
        self._tiles = {k: self._tiles[k] for k in apply(list(self._tiles), order)}
        self._overview_grid.reorder([tile for tile, _chart in self._tiles.values()])
        if getattr(self, "_bubbles", None) is not None:
            self._bubbles = self._regroup(reordered(self._bubbles, order))
            self.populations.set_data(self._bubbles, self._bubble_colours())
        if getattr(self, "_cells", None):
            cells = dict(self._cells, columns=sort_by(self._cells.get("columns") or [], order,
                                                       lambda c: c["condition"]))
            self._cells = self._regroup_cells(cells)
            self.cells_chart.set_payload(self._cells, self._labels, self._colours)

    def set_running(self, on: bool) -> None:
        self.placeholder.set_running(on)

    # -- typed ranges (30 Oct 2026) ------------------------------------------------------------------------------ #
    def dist_axes(self, key: str) -> dict:
        """The ranges typed for condition ``key``'s Distributions histogram (``{"x", "y"}``; empty: automatic)."""
        return dict(self._dist_axes.get(key) or {})

    def set_dist_axes(self, key: str, axes) -> None:
        """Ranges for ``key``'s histogram (`axes_view.check`, refused by name), drawn now; all None: automatic."""
        axes = check(axis_rows("appd_histogram"), axes or {})
        if axes_empty(axes):
            self._dist_axes.pop(key, None)
        else:
            self._dist_axes[key] = axes
        if key in self._dist_cards:
            body = self._dist_cards[key][1]
            body.chart.axes = dict(self._dist_axes.get(key) or {})
            body.chart.update()
            body.redraw()                      # the residual plots follow the histogram's x range

    # -- the All conditions view (3 Nov 2026) --------------------------------------------------------------------- #
    def _dist_view_changed(self, index: int) -> None:
        self._dist_stack.setCurrentIndex(index)
        if index == 1:
            for key in self._tiles:
                self._draw_tile(key)

    def _draw_tile(self, key: str) -> None:
        """Condition ``key``'s All conditions tile, as its card shows it (`_DistributionBody.histogram_payload`)."""
        if key not in self._tiles or key not in self._dist_cards:
            return
        _tile, chart = self._tiles[key]
        body = self._dist_cards[key][1]
        chart.set_payload(body.histogram_payload(), self._colours.get(key, _DATA))

    def overview_entries(self) -> list:
        """What "Export all conditions…" draws: every condition with a result, as its card shows it (the
        Distributions cards' `export_entries`), in the page's order."""
        return [e for key in self._conditions for e in self.export_entries(key, "distributions")]

    def open_dist_axes(self, key: str) -> None:
        """The Axes popover for ``key``'s histogram, under its card's button."""
        if self._axes_popover is None:
            self._axes_popover = AxesPopover(self.t, self)
            self._axes_popover.applied.connect(lambda axes: self.set_dist_axes(self._axes_key, axes))
        self._axes_key = key
        card = self._dist_cards.get(key, (None, None))[0]
        self._axes_popover.show_for(axis_rows("appd_histogram"), self.dist_axes(key),
                                    None if card is None else card._inspect)

    def axes_state(self) -> dict:
        """Every typed range of the section (a session keeps them): the bubbles' and each histogram's."""
        return {"bubbles": self.populations.axes_state(), "distributions": {k: dict(v) for k, v in self._dist_axes.items()}}

    def set_axes_state(self, state: dict) -> None:
        self.populations.set_axes_state((state or {}).get("bubbles") or {})
        for key, axes in ((state or {}).get("distributions") or {}).items():
            self.set_dist_axes(key, axes)

    def set_result(self, condition: str, result: dict) -> None:
        self._results[condition] = result
        colour = self._colours.get(condition, _DATA)
        if condition in self._dist_cards:
            card, body = self._dist_cards[condition]
            card.set_caption(f"{result['n_tracks']:,} tracks · {result['n_pop']} population"
                             f"{'' if result['n_pop'] == 1 else 's'}"
                             + (" (best by BIC)" if result["chosen_by"] == "best" else ""))
            body.set_result(result, colour)
        if condition in self._check_cards:
            card, body = self._check_cards[condition]
            card.set_caption("")
            body.set_result(result, colour)
        self._reveal()

    def set_error(self, condition: str, message: str) -> None:
        self._results[condition] = {"error": message}
        for cards in (self._dist_cards, self._check_cards):
            if condition in cards:
                card, body = cards[condition]
                card.set_caption("")
                body.set_error(message)
        self._reveal()

    def set_grouping(self, factors: dict, nesting) -> None:
        """The folder design's factors and the nesting chosen (`factor_layout`): the bubbles are grouped by them,
        now and at every later Run."""
        self._grouping = (dict(factors or {}), tuple(nesting or ()))
        if getattr(self, "_bubbles", None) is not None:
            self._bubbles = self._regroup(self._bubbles)
            self.populations.set_data(self._bubbles, self._bubble_colours())
        if getattr(self, "_cells", None):
            self._cells = self._regroup_cells(self._cells)
            self.cells_chart.set_payload(self._cells, self._labels, self._colours)

    def _regroup_cells(self, cells):
        from ...appd_view import grouped_cells

        factors, nesting = getattr(self, "_grouping", ({}, ()))
        return grouped_cells(cells, factors, nesting) if cells is not None else cells

    def _regroup(self, data):
        from ...bubble_view import regrouped

        factors, nesting = getattr(self, "_grouping", ({}, ()))
        return regrouped(data, factors, nesting) if factors and len(nesting) >= 2 else data

    def finish(self, joint: dict | None = None) -> None:
        """Every condition landed (and, under a set-wide Force same D across, the joint fit, ``joint``): the bubbles
        and the Per cell chart, which compare conditions."""
        self._joint = joint
        try:
            plot, caption, self._payloads = bubble_plot(self._results, self._conditions, self.last_config, joint)
        except (KeyError, ValueError) as exc:
            plot, caption, self._payloads = None, f"The populations could not be drawn: {exc}", {}
        self._bubbles = self._regroup(plot) if plot is not None else None
        if self._bubbles is not None:
            self.populations.set_data(self._bubbles, self._bubble_colours())
        self.caption.setText(caption)
        cells = self._regroup_cells(cells_payload(self._results, self._conditions))
        self._cells = cells
        self.cells_chart.set_payload(cells, self._labels, self._colours)
        self.cells_card.set_caption(cells["caption"])
        self._draw_cell_length()
        self.set_running(False)

    def _draw_cell_length(self) -> None:
        """The cell-length card, from the source picked (apparent D or SQD)."""
        source = CELL_SHARE_SOURCES[self.length_source.current()]
        lengths = cell_length_payload(self._results, self._conditions, source=source)
        self._cell_lengths = lengths
        drawn = list((lengths.get("points") or {}).keys())
        self.length_key.blockSignals(True)
        self.length_key.set_entries([(c, self._labels.get(c, c), self._colours.get(c, _DATA)) for c in drawn])
        self.length_key.blockSignals(False)
        shown = _without(lengths, self.length_key.hidden())
        self.length_chart.set_payload(shown, self._labels, self._colours)
        if lengths["available"]:
            self.length_table.set_table(lengths["columns"],
                                        [[self._labels.get(r[0], r[0]), *r[1:]] for r in shown["rows"]])
            self.length_text.setText(lengths["text"])
            self.cell_length_card.set_caption(lengths["caption"])
        else:
            self.length_table.set_table([], [])
            self.length_text.setText("")
            self.cell_length_card.set_caption(f"Not available: {lengths['why']}.")

    def _reveal(self) -> None:
        self._ever_shown = True
        self._strip_host.show()
        self._pages.setCurrentIndex(1)

    # -- navigation -------------------------------------------------------------------------------------------- #
    def show_page(self, index: int) -> None:
        if self.strip.current() != index:
            self.strip.blockSignals(True)
            self.strip.select(index)
            self.strip.blockSignals(False)
        self._content.setCurrentIndex(index)

    def show_days(self, column: str) -> None:
        """A bubble's column (a condition, or "glucose (2-pop)" with fits side by side): its replicates, from that
        column's own design."""
        condition, name = str(column), self._labels.get(column, column)
        for fit_name, fit in zip(getattr(self._bubbles, "conditions", None) or [],
                                 getattr(self._bubbles, "fits", None) or []):
            if str(fit_name) == str(column):
                condition = str(fit["condition"])
                name = f"{self._labels.get(condition, condition)} ({fit.get('label') or fit['model']})"
        result = self._results.get(condition)
        if not result or "error" in result:
            return
        payloads = getattr(self, "_payloads", None) or {}
        design = payloads.get(str(column)) or payloads.get(condition)
        self.days.show_condition(condition, name, design, self._colours.get(condition, _DATA))
        self.show_page(0)
        self._pop_stack.setCurrentIndex(1)

    def show_crosschecks(self, widget: QWidget) -> None:
        """The SQD/HMM cross-checks page (`appd_chart.ApparentDDetail`), held here so it stays reachable."""
        if widget.parent() is not self._cross_host:
            self._cross_lay.addWidget(widget)
        self.show_page(0)
        self._pop_stack.setCurrentIndex(2)

    def show_overview(self) -> None:
        self._pop_stack.setCurrentIndex(0)

    # -- what other parts of the window read --------------------------------------------------------------------- #
    def has_result(self) -> bool:
        return self._ever_shown

    def results(self) -> dict:
        return dict(self._results)

    def joint(self) -> dict | None:
        """The set-wide joint fit the bubbles were drawn from (`appd_fits.joint_designs`), None under every other
        Force same D across: a session keeps it beside the results."""
        return getattr(self, "_joint", None)

    def bubble_data(self) -> BubblePlotData | None:
        return self._bubbles

    def set_bracket_warning(self, text: str) -> None:
        """The line naming a bracket that stands on a flagged population (the window's `_sync_fit_warning`)."""
        self.bracket_warning.setText(text)
        self.bracket_warning.setVisible(bool(text))

    def _bubble_colours(self) -> dict:
        """The bubbles' colours by column: each condition's own, or under fits side by side a shade of it per model
        (`bubble_view.column_colours`, SQD's rule; found by the overnight tester 4 Nov 2026: every side-by-side
        column was drawn grey, its name "Glucose (1-pop)" in no condition's colour)."""
        return column_colours(self._bubbles, self._colours)

    def conditions(self) -> list:
        return list(self._conditions)

    def export_entries(self, key: str, page: str) -> list:
        """What a card's Export draws: ``[{"key", "name", "colour", "result", "selected"}]`` -- the condition's
        result without its per-track and per-cell lists (no figure reads them), and for a Distributions card
        ("distributions") the fit it shows; ``[]`` while the condition has no result."""
        result = self._results.get(key)
        if not result or "error" in result:
            return []
        selected, more = None, {}
        if page == "distributions" and key in self._dist_cards:
            # the card's Show (a tuple of counts since 3 Nov 2026), its Components and Residuals boxes
            body = self._dist_cards[key][1]
            selected = body.shown() or None
            more = {"components": body.components(), "residuals": body.residuals()}
        return [{"key": key, "name": self._labels.get(key, key), "colour": self._colours.get(key, _DATA),
                 "result": {k: v for k, v in result.items() if k not in ("tracks", "cells")},
                 "selected": selected, **more}]

    def cell_length_export(self) -> dict | None:
        """The cell-length card's Export: the payload it paints, with the names and colours."""
        lengths = getattr(self, "_cell_lengths", None)
        if not lengths or not lengths.get("available"):
            return None
        return {**_without(lengths, self.length_key.hidden()), "labels": dict(self._labels),
                "colours": dict(self._colours)}

    def cells_export(self) -> dict | None:
        """The Per cell card's Export: the payload it paints, with the names and colours it paints them in."""
        if not self._cells or not self._cells.get("columns"):
            return None
        return {"cells": self._cells, "labels": dict(self._labels), "colours": dict(self._colours)}

    def set_colours(self, colours: dict) -> None:
        self._colours.update(colours)
        for cards in (self._dist_cards, self._check_cards):
            for key, (_card, body) in cards.items():
                if key in colours:
                    body.set_colour(colours[key])
        if self._bubbles is not None:
            self.populations.chart.set_colours(self._bubble_colours())
        self.cells_chart.set_colours(colours)
        self.length_chart.set_colours(colours)
