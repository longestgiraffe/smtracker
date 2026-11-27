"""The MSD sidebar section: v2's MSD tab, kept, with what it got wrong fixed and named -- rebuilt on 16 Oct 2026 in
the Apparent Diffusion section's pattern (asked: "msd is still in a weird shape with lots of text etc.").

FOUR PAGES behind a strip, the sidebar's sub-views following it (`PAGES`): CURVES -- every condition's ensemble curve
on shared axes and one table of the numbers (tracks, D, its spread, the intercept); PER CONDITION -- each condition's
curve over a pale sample of its tracks, its numbers in the caption and nothing under it; CHECKS -- D against the fit
window (by pairs and each track once) and D by track length; v2 -- what v2 printed for the same data, as a table, with
its sigma histogram. What every card used to repeat is said once per page (`msd_view.PAGE_NOTES`). A Run dialog of fixed
choices (`MSDRunDialog`, `msd_view.SECTION_HELP`); every loaded condition is fitted, one pool job each.

The numbers, words and rules come from `smtracker.msd_view` (headless, picklable, run on the process pool); this
module reads only the plain lists and strings it returns, and imports no numpy, pandas or scipy
(`tests/test_ui_thin_qt.py`). The Run itself lives in `window.py`, this workspace's convention for every Run.

Typed ranges (30 Oct 2026, `smtracker.axes_view`, the shared popover): the shared Curves chart and each condition's
card have an Axes button, lag (x) and MSD (y); the section keeps them (``""`` the shared chart's, else a condition's)
through a new Run until Auto (`typed_axes`, `set_typed_axes`, `axes_state`). A typed lag axis may start above 0: the
pointer's lag is `_CurveAxes._lag_at`.
"""
from __future__ import annotations

from ...condition_order import apply as order_apply
from ...axes_view import apply_range, axis_rows, check
from ...axes_view import empty as axes_empty
from ..widgets.axes_popover import AxesPopover

from qtpy.QtCore import QPointF, QRectF, Qt, Signal
from qtpy.QtGui import QColor, QPainterPath, QPen
from qtpy.QtWidgets import QHBoxLayout, QScrollArea, QStackedWidget, QVBoxLayout, QWidget

from ... import chart_text, figstyle
from ...msd_view import (
    days_hover,
    has_per_cell,
    per_cell_csv,
    PAGE_NOTES,
    SECTION_CURVE,
    SECTION_DEFAULTS,
    SECTION_DIRECTION_LABELS,
    SECTION_DIRECTIONS,
    SECTION_HELP,
    SECTION_LAGS,
    SECTION_UNCERTAINTY,
    SECTION_UNCERTAINTY_LABELS,
    SECTION_WEIGHTING,
    SECTION_WEIGHTING_LABELS,
    curve_hover,
    curves_table,
    histogram_hover,
    length_check,
    overlay_hover,
    overlay_range,
    section_config,
    short_caption,
    v2_rows,
    window_payload,
)
from ..theme import typography as ty
from ..widgets.cardgrid import CardGrid
from ..widgets.controls import Button, Segmented
from ..widgets.dialogs import CloseButton, OverlayDialog
from ..widgets.primitives import Checkbox, discard, label
from ..widgets.settings import InfoLink
from .condition_key import ConditionKey
from .jumps_chart import _Table
from .qc_chart import _DATA, _Caption, _PanelCard, _QCChart
from .section_placeholder import SectionPlaceholder
from .toolbar import rem

__all__ = ["MSDCard", "MSDChart", "MSDOverlayChart", "MSDRunDialog", "MSDSection", "PAGES", "SigmaHistogram"]


# --------------------------------------------------------------------------- #
# The charts
# --------------------------------------------------------------------------- #


def _alpha(colour: str, a: int) -> QColor:
    c = QColor(colour)
    c.setAlpha(a)
    return c


def _polyline(p, pts, px, py) -> None:
    if len(pts) < 2:
        return
    path = QPainterPath()
    for i, (t, v) in enumerate(pts):
        (path.moveTo if i == 0 else path.lineTo)(px(t), py(v))
    p.drawPath(path)


class _CurveAxes(_QCChart):
    """Shared axes: lag time from zero, MSD from the data's floor."""

    _FULL_BOTTOM = 40
    _ROOMY_H = 160
    _PLOT_LEFT = 66

    def _axes(self, p, box: QRectF, t_max: float, y_lo: float, y_hi: float):
        """The lag axis from zero and the MSD axis from the data's floor -- or, typed (the card's Axes, 30 Oct 2026),
        exactly where they were put (`axes_view.apply_range`). Returns ``(the lag axis's end, px, py)``; its start is
        `_t0`."""
        typed = getattr(self, "axes", None) or {}
        t0, t_axis = apply_range(0.0, max(t_max * 1.04, 1e-9), typed.get("x"))
        if typed.get("y") is None:
            axis_lo, axis_hi = self._grid(p, box, y_lo, y_hi + (y_hi - y_lo) * 0.06,
                                          chart_text.MSD_AXIS_LABEL)
        else:
            lo, hi = apply_range(y_lo, y_hi + (y_hi - y_lo) * 0.06, typed["y"])
            axis_lo, axis_hi = self._grid_exact(p, box, lo, hi, chart_text.MSD_AXIS_LABEL)
        self._t0 = t0
        self.drawn = {"x": (t0, t_axis), "y": (axis_lo, axis_hi)}

        def px(t):
            return box.left() + (float(t) - t0) / (t_axis - t0) * box.width()

        def py(v):
            return self._map_y(box, float(v), axis_lo, axis_hi)

        return t_axis, px, py

    def _lag_at(self, x: float, g: dict) -> float:
        """The lag under a pointer's ``x`` (the axis may start above 0 when typed)."""
        t0 = getattr(self, "_t0", 0.0)
        return t0 + (x - g["box"].left()) / g["box"].width() * (g["t_axis"] - t0)

    def _draw_curve(self, p, chart: dict, colour: QColor, px, py, *, radius: float = 3.2,
                    whiskers: bool = True) -> None:
        """The ensemble points: filled where fitted, hollow in the tail, with their bars (`msd_view._bars`: the SD
        over replicate days under Days, else the SEM over tracks)."""
        n_fit = chart["n_fit"]
        if whiskers:
            p.setPen(QPen(_alpha(colour.name(), 150), 1.0))
            bars = chart.get("bars")
            if bars is None:
                bars = [(t, m - s, m + s) for t, m, s, _n in chart["curve"] if s > 0]
            for t, lo, hi in bars:
                p.drawLine(QPointF(px(t), py(lo)), QPointF(px(t), py(hi)))
                p.drawLine(QPointF(px(t) - 2.5, py(lo)), QPointF(px(t) + 2.5, py(lo)))
                p.drawLine(QPointF(px(t) - 2.5, py(hi)), QPointF(px(t) + 2.5, py(hi)))
        for i, (t, m, _s, _n) in enumerate(chart["curve"]):
            p.setPen(QPen(colour, 1.4))
            p.setBrush(colour if i < n_fit else QColor(self.t.card))
            p.drawEllipse(QPointF(px(t), py(m)), radius, radius)

    def _draw_days(self, p, chart: dict, colour: QColor, px, py) -> list:
        """Each replicate day's own curve and fit in a shade of the condition's colour; returns the key's
        entries."""
        days = chart.get("days") or []
        entries = []
        for i, d in enumerate(days):
            shade = QColor(figstyle.day_shade(colour.name(), i, len(days)))
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(shade, 1.3))
            _polyline(p, d["curve"], px, py)
            if d.get("line"):
                pen = QPen(shade, 1.0)
                pen.setStyle(Qt.DashLine)
                p.setPen(pen)
                _polyline(p, d["line"], px, py)
            p.setPen(Qt.NoPen)
            p.setBrush(shade)
            for t, m in d["curve"]:
                p.drawEllipse(QPointF(px(t), py(m)), 2.3, 2.3)
            entries.append((shade.name(), d["name"], "line"))
        return entries

    def _draw_fit(self, p, chart: dict, colour: QColor, px, py) -> None:
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(colour, 1.7))
        _polyline(p, chart["fit"], px, py)
        if chart["extension"]:
            pen = QPen(_alpha(colour.name(), 120), 1.2)
            pen.setStyle(Qt.DashLine)
            p.setPen(pen)
            _polyline(p, chart["extension"], px, py)


class MSDChart(_CurveAxes):
    """One condition: pale per-track curves, the ensemble, its early-lag fit -- or, after a click, each replicate
    day's own curve (17 Oct 2026); a second click goes back.

    ``payload``: ``{"chart": compute_msd_view's chart, "colour": hex}``.
    """

    def __init__(self, t, parent=None):
        super().__init__(t, parent)
        #: The replicate days drawn instead of the per-track sample (a click toggles it).
        self.show_days = False
        #: Typed ranges (the card's Axes, `axes_view`), and the ranges as last drawn.
        self.axes: dict = {}
        self.drawn: dict = {}

    def mousePressEvent(self, e) -> None:
        g = self._geom
        if e.button() == Qt.LeftButton and g and g["box"].contains(e.position()) and g["chart"].get("days"):
            self.show_days = not self.show_days
            self._hide_tip()
            self.update()
        super().mousePressEvent(e)

    def _paint(self, p, box: QRectF, pl) -> None:
        chart = pl["chart"]
        colour = QColor(pl.get("colour") or _DATA)
        y_lo, y_hi = chart["y_range"]
        t_axis, px, py = self._axes(p, box, chart["t_max"], y_lo, y_hi)
        days = self.show_days and bool(chart.get("days"))
        self._geom = {"box": box, "t_axis": t_axis, "chart": chart, "days": days}
        self.setCursor(Qt.PointingHandCursor if chart.get("days") else Qt.ArrowCursor)
        p.save()
        p.setClipRect(box.adjusted(-1, -1, 1, 1))
        if days:
            day_entries = self._draw_days(p, chart, colour, px, py)
            self._draw_fit(p, chart, QColor(self.t.foreground), px, py)
            self._draw_curve(p, chart, colour, px, py)
            p.restore()
            self._legend(p, [(colour.name(), "all replicates pooled", "dot")] + day_entries,
                         QPointF(box.left() + 8, box.top() + 4), horizontal=True)
            self._x_numeric(p, box, self._t0, t_axis, chart_text.LAG_AXIS_LABEL)
            return
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(_alpha(colour.name(), 26), 1.0))
        for pts in chart["tracks"]:
            _polyline(p, pts, px, py)
        if chart["legacy_fit"]:
            pen = QPen(QColor(self.t.muted_foreground), 1.6)
            pen.setStyle(Qt.DotLine)
            p.setPen(pen)
            _polyline(p, chart["legacy_fit"], px, py)
        self._draw_fit(p, chart, QColor(self.t.foreground), px, py)
        self._draw_curve(p, chart, colour, px, py)
        p.restore()
        entries = [(colour.name(), "fitted lags", "dot"),
                   (self.t.foreground, "fit", "line")]
        if chart["legacy_fit"]:
            entries.append((self.t.muted_foreground, "v2's fit (dotted)", "line"))
        self._legend(p, entries, QPointF(box.left() + 8, box.top() + 4), horizontal=True)
        self._x_numeric(p, box, self._t0, t_axis, chart_text.LAG_AXIS_LABEL)

    def _hover_text(self, pos: QPointF) -> str | None:
        g = self._geom
        if not g or not g["box"].contains(pos):
            return None
        t = self._lag_at(pos.x(), g)
        return days_hover(g["chart"], t) if g.get("days") else curve_hover(g["chart"], t)


class MSDOverlayChart(_CurveAxes):
    """Every condition's ensemble curve and fit on shared axes; a click on one shows that condition's replicate
    days (17 Oct 2026), a second click all conditions again. Under Uncertainty = Days each point carries the SD over
    days (`msd_view._bars`); the SEM over tracks is left off here, as before.

    ``payload``: ``{"entries": [(name, chart, colour), ...]}``.
    """

    #: The condition whose days are drawn changed (its name, or None for every condition).
    focus_changed = Signal(object)

    def __init__(self, t, parent=None):
        super().__init__(t, parent)
        #: The condition whose replicate days are drawn, or None for every condition.
        self.focus: str | None = None
        #: Typed ranges (the card's Axes, `axes_view`), and the ranges as last drawn.
        self.axes: dict = {}
        self.drawn: dict = {}

    def set_focus(self, name: str | None) -> None:
        """Draw ``name``'s replicate days (None: every condition)."""
        if name != self.focus:
            self.focus = name
            self._hide_tip()
            self.update()
            self.focus_changed.emit(name)

    def set_payload(self, payload) -> None:
        names = {e[0] for e in (payload or {}).get("entries", []) if e[1]}
        if self.focus not in names:
            self.set_focus(None)
        super().set_payload(payload)

    def mousePressEvent(self, e) -> None:
        g = self._geom
        if e.button() == Qt.LeftButton and g and g["box"].contains(e.position()):
            self.set_focus(None if self.focus is not None else self._nearest(e.position()))
        super().mousePressEvent(e)

    def _nearest(self, pos: QPointF) -> str | None:
        """The condition whose curve passes nearest the click, at the lag under it."""
        g = self._geom
        t = self._lag_at(pos.x(), g)
        best, best_d = None, None
        for name, chart in g["entries"]:
            curve = chart.get("curve") or []
            if not curve:
                continue
            i = min(range(len(curve)), key=lambda k: abs(curve[k][0] - t))
            d = abs(g["py"](curve[i][1]) - pos.y())
            if best_d is None or d < best_d:
                best, best_d = name, d
        return best

    def _paint(self, p, box: QRectF, pl) -> None:
        entries = [e for e in pl["entries"] if e[1]]
        shown = [e for e in entries if e[0] == self.focus] or entries
        shared = overlay_range([chart for _n, chart, _c in shown])
        if shared is None:
            return
        t_max, y_lo, y_hi = shared
        t_axis, px, py = self._axes(p, box, t_max, y_lo, y_hi)
        self._geom = {"box": box, "t_axis": t_axis, "py": py,
                      "entries": [(name, chart) for name, chart, _c in entries],
                      "focus": [(name, chart) for name, chart, _c in shown] if self.focus else None}
        self.setCursor(Qt.PointingHandCursor)
        p.save()
        p.setClipRect(box.adjusted(-1, -1, 1, 1))
        if self.focus is not None:
            name, chart, colour = shown[0]
            c = QColor(colour or _DATA)
            day_entries = self._draw_days(p, chart, c, px, py)
            self._draw_fit(p, chart, c, px, py)
            self._draw_curve(p, chart, c, px, py, radius=2.8)
            p.restore()
            key = [(c.name(), f"{name}, all replicates pooled", "dot")] + (
                day_entries or [(self.t.muted_foreground, "one session: no replicates", "line")])
            self._legend(p, key, QPointF(box.left() + 8, box.top() + 4), horizontal=True)
            self._x_numeric(p, box, self._t0, t_axis, chart_text.LAG_AXIS_LABEL)
            return
        for _name, chart, colour in entries:
            c = QColor(colour or _DATA)
            self._draw_fit(p, chart, c, px, py)
            day_bars = (chart.get("bar_words") or "").startswith(("± SD over", "95 % interval"))
            self._draw_curve(p, chart, c, px, py, radius=2.8, whiskers=day_bars)
        p.restore()
        if not pl.get("keyless"):
            self._legend(p, [(colour or _DATA, name, "dot") for name, _c, colour in entries],
                         QPointF(box.left() + 8, box.top() + 4), horizontal=True)
        self._x_numeric(p, box, self._t0, t_axis, chart_text.LAG_AXIS_LABEL)

    def _hover_text(self, pos: QPointF) -> str | None:
        g = self._geom
        if not g or not g["box"].contains(pos):
            return None
        t = self._lag_at(pos.x(), g)
        if g.get("focus"):
            return days_hover(g["focus"][0][1], t) or "one session: no replicates\nclick: back"
        text = overlay_hover(g["entries"], t)
        return None if text is None else text + "\nclick a curve: its replicates"


class SigmaHistogram(_QCChart):
    """v2's localisation-error histogram, 20 nm bins, the negative intercepts kept left.

    ``payload``: ``{"histogram": the result's histogram, "colour": hex}``.
    """

    _FULL_BOTTOM = 40
    _ROOMY_H = 140
    _PLOT_LEFT = 66
    #: Room for the last tick's label, centred on the plot's right edge.
    _RIGHT = 22

    def _paint(self, p, box: QRectF, pl) -> None:
        hist = pl["histogram"]
        bins = hist["bins"]
        if not bins:
            return
        lo, hi = bins[0][0], bins[-1][1]
        axis_lo, axis_hi = self._grid(p, box, 0.0, max(1.0, hist["y_max"] * 1.08), "tracks")

        def px(v):
            return box.left() + (v - lo) / (hi - lo) * box.width()

        colour = QColor(pl.get("colour") or _DATA)
        # Grey, not the warning ink: a condition coloured orange would otherwise read the
        # same on both sides of zero.
        folded = _alpha(self.t.muted_foreground, 150)
        p.setPen(Qt.NoPen)
        for a, b, count, negative in bins:
            if count <= 0:
                continue
            top = self._map_y(box, count, axis_lo, axis_hi)
            p.setBrush(folded if negative else colour)
            p.drawRect(QRectF(px(a) + 0.5, top, max(1.0, px(b) - px(a) - 1.0),
                              box.bottom() - top))
        if lo < 0 < hi:
            pen = QPen(QColor(self.t.foreground), 1.0)
            pen.setStyle(Qt.DashLine)
            p.setPen(pen)
            p.drawLine(QPointF(px(0.0), box.top()), QPointF(px(0.0), box.bottom()))
        self._geom = {"box": box, "lo": lo, "hi": hi, "histogram": hist}
        self._legend(p, [(folded.name(), "negative intercept (v2 folds it right)", "box"),
                         (colour.name(), "positive", "box")],
                     QPointF(box.right() - 4, box.top() + 4), horizontal=True,
                     right_aligned=True)
        self._x_numeric(p, box, lo, hi, "per-track σ = √|b/4|, sign of b kept (nm)")

    def _hover_text(self, pos: QPointF) -> str | None:
        g = self._geom
        if not g or not g["box"].contains(pos):
            return None
        value = g["lo"] + (pos.x() - g["box"].left()) / g["box"].width() * (g["hi"] - g["lo"])
        return histogram_hover(g["histogram"], value)


# --------------------------------------------------------------------------- #
# The checks' charts
# --------------------------------------------------------------------------- #


class _WindowChart(_QCChart):
    """D against the number of lags fitted, by pairs (solid, the condition's colour) and each track once (dashed),
    the Run's window marked. ``payload``: ``{"window": msd_view.window_payload(...), "colour": hex}``."""

    _FULL_BOTTOM = 40
    _ROOMY_H = 140
    _PLOT_LEFT = 66

    def _paint(self, p, box: QRectF, pl) -> None:
        w = pl["window"]
        values = [v for v in w["pairs"] + w["tracks"] if v is not None]
        lo = min(values) * 0.9
        axis_lo, axis_hi = self._grid(p, box, lo, w["y_max"], "D (µm²/s)")
        x0, x1 = w["lags"][0] - 0.5, w["lags"][-1] + 0.5

        def px(n):
            return box.left() + (n - x0) / (x1 - x0) * box.width()

        def py(v):
            return self._map_y(box, v, axis_lo, axis_hi)

        colour = QColor(pl.get("colour") or _DATA)
        mark = QPen(QColor(self.t.muted_foreground), 1.0)
        mark.setStyle(Qt.DashLine)
        p.setPen(mark)
        p.drawLine(QPointF(px(w["chosen"]), box.top()), QPointF(px(w["chosen"]), box.bottom()))
        for key, ink, style in (("tracks", QColor(self.t.foreground), Qt.DashLine), ("pairs", colour, Qt.SolidLine)):
            pts = [(n, v) for n, v in zip(w["lags"], w[key]) if v is not None]
            pen = QPen(ink, 1.6)
            pen.setStyle(style)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            _polyline(p, pts, px, py)
            p.setPen(Qt.NoPen)
            p.setBrush(ink)
            for n, v in pts:
                p.drawEllipse(QPointF(px(n), py(v)), 2.6, 2.6)
        self._geom = {"box": box, "px": (x0, x1), "window": w}
        self._legend(p, [(colour.name(), "by pairs", "line"), (self.t.foreground, "each track once", "line")],
                     QPointF(box.right() - 4, box.top() + 4), horizontal=True, right_aligned=True)
        self._x_numeric(p, box, x0, x1, "lags fitted")

    def _hover_text(self, pos: QPointF) -> str | None:
        g = self._geom
        if not g or not g["box"].contains(pos):
            return None
        x0, x1 = g["px"]
        n = x0 + (pos.x() - g["box"].left()) / g["box"].width() * (x1 - x0)
        w = g["window"]
        i = min(range(len(w["lags"])), key=lambda k: abs(w["lags"][k] - n))

        def fmt(v):
            return "–" if v is None else f"{v:.3g} µm²/s"

        return (f"first {w['lags'][i]} lags fitted\nby pairs {fmt(w['pairs'][i])}\n"
                f"each track once {fmt(w['tracks'][i])}")


class _LengthChart(_QCChart):
    """D from the early lags of each track-length class: a bar each (faded under `LENGTH_MIN_TRACKS` tracks), its
    class and tracks under it. ``payload``: ``{"length": msd_view.length_check(...), "colour": hex}``."""

    _FULL_BOTTOM = 40
    _ROOMY_H = 140
    _PLOT_LEFT = 66

    def _paint(self, p, box: QRectF, pl) -> None:
        bins = pl["length"]["bins"]
        if not bins:
            return
        axis_lo, axis_hi = self._grid(p, box, 0.0, pl["length"]["y_max"], "D (µm²/s)")
        colour = pl.get("colour") or _DATA
        pitch = box.width() / len(bins)
        p.setPen(Qt.NoPen)
        for i, b in enumerate(bins):
            if b["D"] is None:
                continue
            fill = QColor(colour)
            if b["few"]:
                fill.setAlphaF(0.3)             # too few tracks to read a D from: drawn, faded
            p.setBrush(fill)
            top = self._map_y(box, b["D"], axis_lo, axis_hi)
            p.drawRect(QRectF(box.left() + i * pitch + pitch * 0.22, top, pitch * 0.56, box.bottom() - top))
        if self._roomy:
            p.setFont(self._tick_font)
            p.setPen(QColor(self.t.muted_foreground))
            for i, b in enumerate(bins):
                x = box.left() + i * pitch
                p.drawText(QRectF(x, box.bottom() + 3, pitch, 14), Qt.AlignHCenter | Qt.AlignTop, b["label"])
                p.drawText(QRectF(x, box.bottom() + 19, pitch, 14), Qt.AlignHCenter | Qt.AlignTop,
                           f"{b['n']:,} track{'' if b['n'] == 1 else 's'}")
        self._geom = {"box": box, "bins": bins}

    def _hover_text(self, pos: QPointF) -> str | None:
        g = self._geom
        if not g or not g["box"].contains(pos):
            return None
        i = int((pos.x() - g["box"].left()) / (g["box"].width() / len(g["bins"])))
        b = g["bins"][max(0, min(i, len(g["bins"]) - 1))]
        d = "not fitted" if b["D"] is None else f"D {b['D']:.3g} µm²/s"
        return f"{b['label']} steps\n{b['n']:,} tracks · {d}" + (" (too few to read)" if b["few"] else "")


# --------------------------------------------------------------------------- #
# The cards
# --------------------------------------------------------------------------- #


class _MSDBody(QWidget):
    """A Per condition card's body: a message, or the chart -- its numbers are the card's caption."""

    CHART_H = 250

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
        self.chart = MSDChart(t)
        self.chart.setMinimumHeight(self.CHART_H)
        self.chart.hide()
        lay.addWidget(self.chart)
        lay.addStretch(1)

    def set_error(self, text: str) -> None:
        self._result = None
        self.chart.hide()
        self.message.setText(text)
        self.message.show()

    def set_colour(self, colour: str) -> None:
        self._colour = colour
        if self._result is not None:
            self.chart.set_payload({"chart": self._result["chart"], "colour": colour})

    def set_result(self, result: dict, colour: str) -> None:
        self.message.hide()
        self._result, self._colour = result, colour
        self.chart.set_payload({"chart": result["chart"], "colour": colour})
        self.chart.show()


class MSDCard(_PanelCard):
    """One condition's curve."""

    def __init__(self, t, title: str, parent=None):
        self.msd_body = _MSDBody(t)
        super().__init__(t, title, self.msd_body, parent=parent, flush=False)

    def set_result(self, result: dict, colour: str) -> None:
        self.set_caption(short_caption(result))
        self.msd_body.set_result(result, colour)

    def set_colour(self, colour: str) -> None:
        self.msd_body.set_colour(colour)

    def set_error(self, text: str) -> None:
        self.set_caption("")
        self.msd_body.set_error(text)


class _OverlayCard(_PanelCard):
    """Every condition's curve on shared axes, under a key with an eye per condition (17 Oct 2026): the eye hides
    or shows it, a click on its name opens its replicate days (as a click on its curve does), again to go back."""

    def __init__(self, t, parent=None):
        self.chart = MSDOverlayChart(t)
        self.chart.setMinimumHeight(280)
        self.key = ConditionKey(t, pickable=True)
        host = QWidget()
        host.setStyleSheet("background:transparent;")
        lay = QVBoxLayout(host)
        lay.setContentsMargins(rem(1.25), 0, rem(1.25), rem(1.0))
        lay.setSpacing(rem(0.25))
        lay.addWidget(self.key)
        lay.addWidget(self.chart)
        super().__init__(t, "Ensemble MSD, every condition", host, parent=parent)
        self._entries: dict = {}
        #: The conditions' order (the Run's, then the Conditions tool's arrows): the curves and the key follow it, not
        #: the order the conditions finish in (3 Nov 2026, reported: the key read "Malate, Glucose").
        self._order: list = []
        self.key.hidden_changed.connect(lambda _h: self._redraw())
        self.key.picked.connect(self._picked)
        self.chart.focus_changed.connect(self._focus_changed)

    def _picked(self, key: str) -> None:
        entry = self._entries.get(key)
        if entry is None or entry[1] is None:
            return
        self.chart.set_focus(None if self.chart.focus == entry[0] else entry[0])

    def _focus_changed(self, name) -> None:
        self.key.set_picked(next((k for k, e in self._entries.items() if e[0] == name), None))

    def hidden(self) -> set:
        """The conditions hidden by their eye (left out of the chart and its export)."""
        return self.key.hidden()

    def set_entry(self, key: str, name: str, chart: dict | None, colour: str) -> None:
        self._entries[key] = (name, chart, colour)
        self._entries = {k: self._entries[k] for k in order_apply(list(self._entries), self._order)}
        self._redraw()

    def set_colours(self, colours: dict) -> None:
        for key, (name, chart, _c) in list(self._entries.items()):
            if key in colours:
                self._entries[key] = (name, chart, colours[key])
        self._redraw()

    def set_order(self, order) -> None:
        """The curves and the key in ``order`` (`condition_order`); nothing recomputed."""
        self._order = [str(k) for k in order or ()]
        self._entries = {k: self._entries[k] for k in order_apply(list(self._entries), order)}
        self._redraw()

    def clear(self, caption: str, order=()) -> None:
        """Empty, for a Run over the conditions in ``order``: each curve takes its place in it as it lands."""
        self._order = [str(k) for k in order or ()]
        self._entries = {}
        self.key.set_entries([])
        self.set_caption(caption)
        self.chart.set_message("computing…")

    def _redraw(self) -> None:
        drawn = [(k, e) for k, e in self._entries.items() if e[1]]
        self.key.set_entries([(k, e[0], e[2] or _DATA) for k, e in drawn])
        hidden = self.key.hidden()
        entries = [e for k, e in drawn if k not in hidden]
        if entries:
            self.chart.set_payload({"entries": entries, "keyless": True})
        elif drawn:
            self.chart.set_message("Every condition is hidden: click an eye in the key to show it again.")
        else:
            self.chart.set_message("No condition could be drawn.")


class _Padded(QWidget):
    """A card body with the cards' own side and bottom padding."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background:transparent;")
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(rem(1.25), 0, rem(1.25), rem(1.0))
        self.lay.setSpacing(rem(0.4))


class _ChecksBody(_Padded):
    """One condition's checks: D against the fit window, and D by track length."""

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.lay.addWidget(label("Fit window: D from the first n lags", ty.ui("xs", "medium"), t.foreground))
        self.window = _WindowChart(t)
        self.window.setMinimumHeight(190)
        self.lay.addWidget(self.window)
        self.window_text = _Caption(t, size="xs")
        self.lay.addWidget(self.window_text)
        self.lay.addSpacing(rem(0.4))
        self.lay.addWidget(label("Track length: D from the early lags of each length", ty.ui("xs", "medium"),
                                 t.foreground))
        self.length = _LengthChart(t)
        self.length.setMinimumHeight(190)
        self.lay.addWidget(self.length)
        self.length_text = _Caption(t, size="xs")
        self.lay.addWidget(self.length_text)
        self.lay.addStretch(1)
        self._result = None

    def set_result(self, result: dict, colour: str) -> None:
        self._result = result
        w, ln = window_payload(result), length_check(result)
        if w is None:
            self.window.set_message("No fit window could be fitted.")
            self.window_text.setText("")
        else:
            self.window.set_payload({"window": w, "colour": colour})
            self.window_text.setText(w["text"])
        if ln is None or not ln["bins"]:
            self.length.set_message("No track-length class could be fitted.")
            self.length_text.setText("")
        else:
            self.length.set_payload({"length": ln, "colour": colour})
            self.length_text.setText(ln["text"])

    def set_error(self, text: str) -> None:
        self._result = None
        self.window.set_message(text)
        self.length.set_message("")
        self.window_text.setText("")
        self.length_text.setText("")

    def set_colour(self, colour: str) -> None:
        if self._result is not None:
            self.set_result(self._result, colour)


class _V2Body(_Padded):
    """One condition beside v2: a table of what v2 printed and this page's numbers, and v2's sigma histogram."""

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.table = _Table(t)
        self.lay.addWidget(self.table)
        self.histogram = SigmaHistogram(t)
        self.histogram.setMinimumHeight(170)
        self.histogram.hide()
        self.lay.addWidget(self.histogram)
        self.lay.addStretch(1)
        self._result = None

    def set_result(self, result: dict, colour: str) -> None:
        self._result = result
        self.table.set_table(["", ""], [list(row) for row in v2_rows(result)])
        if result.get("histogram"):
            self.histogram.set_payload({"histogram": result["histogram"], "colour": colour})
            self.histogram.show()
        else:
            self.histogram.hide()

    def set_error(self, text: str) -> None:
        self._result = None
        self.table.set_table([""], [[text]])
        self.histogram.hide()

    def set_colour(self, colour: str) -> None:
        if self._result is not None and self._result.get("histogram"):
            self.histogram.set_payload({"histogram": self._result["histogram"], "colour": colour})


# --------------------------------------------------------------------------- #
# The Run dialog
# --------------------------------------------------------------------------- #


class MSDRunDialog(OverlayDialog):
    """The Run's five settings (`msd_view.SECTION_HELP`), each a fixed choice. Transactional like every Run dialog
    here: nothing runs until Run; Cancel and the corner X discard the draft. Every loaded condition is fitted."""

    run = Signal(dict)

    def __init__(self, t, *, seed: dict | None = None, content=None, parent=None):
        super().__init__(t, content, parent=parent)
        self.t = t
        seed = {**SECTION_DEFAULTS, **{k: v for k, v in (seed or {}).items() if k in SECTION_DEFAULTS}}
        head = QHBoxLayout()
        head.addWidget(label("Configure MSD", ty.ui("base", "semibold"), t.foreground))
        head.addStretch(1)
        close = CloseButton(t)
        close.clicked.connect(self._dismiss)
        head.addWidget(close)
        self.card_lay.addLayout(head)
        self.card_lay.addWidget(label("Every loaded condition is fitted. Nothing runs until you press Run.",
                                      ty.ui("sm"), t.muted_foreground))
        self._lags = Segmented(t, [str(n) for n in SECTION_LAGS], current=SECTION_LAGS.index(int(seed["n_lags"])))
        self._curve = Segmented(t, [f"to lag {n}" for n in SECTION_CURVE],
                                current=SECTION_CURVE.index(int(seed["max_lag"])))
        self._weighting = Segmented(t, list(SECTION_WEIGHTING_LABELS),
                                    current=SECTION_WEIGHTING.index(seed["weighting"]))
        self._direction = Segmented(t, list(SECTION_DIRECTION_LABELS),
                                    current=SECTION_DIRECTIONS.index(seed["direction"]))
        self._uncertainty = Segmented(t, list(SECTION_UNCERTAINTY_LABELS),
                                      current=SECTION_UNCERTAINTY.index(seed["uncertainty"]))
        #: The helper under each setting, by key (read by the tests).
        self.helpers: dict = {}
        for name, control, key in (("Lags fitted", self._lags, "n_lags"), ("Curve drawn", self._curve, "max_lag"),
                                   ("Weighting", self._weighting, "weighting"),
                                   ("Direction", self._direction, "direction"),
                                   ("Uncertainty", self._uncertainty, "uncertainty")):
            row = QHBoxLayout()
            row.setSpacing(rem(1.0))
            name_label = label(name, ty.ui("sm", "medium"), t.muted_foreground)
            name_label.setMinimumWidth(rem(8))
            row.addWidget(name_label)
            row.addWidget(control)
            row.addStretch(1)
            self.card_lay.addLayout(row)
            line, more = SECTION_HELP[key]
            self.helpers[key] = InfoLink(t, line, more)
            self.card_lay.addWidget(self.helpers[key])
        footer = QHBoxLayout()
        footer.addStretch(1)
        cancel = Button(t, "Cancel", variant="outline")
        cancel.clicked.connect(self._dismiss)
        self._run_btn = Button(t, "Run", variant="default")
        self._run_btn.clicked.connect(self._on_run)
        footer.addWidget(cancel)
        footer.addWidget(self._run_btn)
        self.card_lay.addLayout(footer)

    def config(self) -> dict:
        return section_config(SECTION_LAGS[self._lags.current()], SECTION_CURVE[self._curve.current()],
                              SECTION_WEIGHTING[self._weighting.current()],
                              SECTION_DIRECTIONS[self._direction.current()],
                              SECTION_UNCERTAINTY[self._uncertainty.current()])

    def _on_run(self) -> None:
        self._acted = True
        self.run.emit(self.config())
        self._dismiss()


# --------------------------------------------------------------------------- #
# The section
# --------------------------------------------------------------------------- #

#: The strip's pages, in order; the sidebar's sub-views are ``msd.<page>``. Two since 23 Nov 2026 (asked: "all tabs in
#: msd really needed?", then "Two tabs: Curves + Checks"): the single tracks behind each curve are a switch on Curves
#: (they were the "Per condition" page) and what SMTracker 2 printed closes Checks (it was the "v2" page).
PAGES = ("curves", "checks")
PAGE_LABELS = ("Curves", "Checks")


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


class MSDSection(QWidget):
    """The MSD sidebar section: a placeholder until the first result, then two pages behind a strip, filling in as
    conditions land (one job per condition, `msd_view.msd_condition`): Curves (every condition on shared axes, the
    numbers, and behind a switch each condition's curve over its single tracks) and Checks (the fit window and track
    length per condition, then what SMTracker 2 printed beside this page's D)."""

    run_requested = Signal()
    cancel_requested = Signal()
    #: The strip moved to page ``index`` (`PAGES`): the window moves the sidebar's sub-view.
    page_changed = Signal(int)
    #: An Export button: "" for the shared chart, else the condition's key.
    export_requested = Signal(str)
    #: The Curves table's Export.
    table_export_requested = Signal()
    #: A Checks card's / a v2 card's Export, with the condition (asked 16 Oct 2026: "Export for MSD pages").
    checks_export_requested = Signal(str)
    v2_export_requested = Signal(str)
    #: "Save MSD per cell (CSV)" under the numbers (17 Oct 2026, v2's "Save MSD in csv...").
    cells_csv_requested = Signal()

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.setStyleSheet("background:transparent;")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(rem(0.75))
        self.placeholder = SectionPlaceholder(
            t, title="MSD", description="Mean squared displacement against lag time, fitted over the early lags: a "
                                        "cross-check on the populations.",
            icon_name="chart-spline")
        self.placeholder.run_requested.connect(self.run_requested.emit)
        self.placeholder.cancel_requested.connect(self.cancel_requested.emit)
        self.strip = Segmented(t, list(PAGE_LABELS), current=0, size="sm")
        self.strip.changed.connect(self.page_changed.emit)
        strip_row = QHBoxLayout()
        strip_row.setContentsMargins(0, 0, 0, 0)
        strip_row.addWidget(self.strip)
        strip_row.addStretch(1)
        self._strip_host = QWidget()
        self._strip_host.setStyleSheet("background:transparent;")
        self._strip_host.setLayout(strip_row)
        outer.addWidget(self._strip_host)
        # ON SHOW FROM THE START (22 Nov 2026: "right now sqd is the only one with different tabs" -- hidden until the
        # first Run, the pages looked like none): before a Run every page is the placeholder

        def page(note_key):
            host = QWidget()
            host.setStyleSheet("background:transparent;")
            lay = QVBoxLayout(host)
            lay.setContentsMargins(0, 0, 0, rem(0.5))
            lay.setSpacing(rem(1.0))
            line, more = PAGE_NOTES[note_key]
            link = InfoLink(t, line, more)
            lay.addWidget(link)
            return host, lay, link

        def grid():
            holder = QWidget()
            holder.setStyleSheet("background:transparent;")
            return holder, CardGrid(holder, min_width=rem(30), max_columns=2, spacing=rem(1.0))

        # Curves: every condition on shared axes, and the numbers in one table
        curves, curves_lay, self.reading = page("curves")
        self.overlay = _OverlayCard(t)
        self.overlay.enable_export()
        self.overlay.export_requested.connect(lambda: self.export_requested.emit(""))
        # THE AXES (30 Oct 2026): lag (x) and MSD (y) typed, the shared chart's and each condition's
        self.overlay.enable_inspect("Axes", glyph="ruler")
        self.overlay.inspect_requested.connect(lambda: self.open_axes(""))
        #: Typed ranges: "" the shared chart's, else a condition's; kept through a new Run until Auto.
        self._typed: dict = {}
        self._axes_popover = None
        curves_lay.addWidget(self.overlay)
        table_host = _Padded()
        self.table = _Table(t)
        table_host.lay.addWidget(self.table)
        self.cells_csv = Button(t, "Save MSD per cell (CSV)\u2026", size="small", variant="outline")
        self.cells_csv.setToolTip("Every cell's MSD at the Run's lags and weighting, one row per cell and lag "
                                  "(SMTracker 2's \u201cSave MSD in csv\u201d, for all cells at once).")
        self.cells_csv.clicked.connect(self.cells_csv_requested.emit)
        self.cells_csv.setEnabled(False)
        csv_row = QHBoxLayout()
        csv_row.setContentsMargins(0, rem(0.25), 0, 0)
        csv_row.addWidget(self.cells_csv)
        csv_row.addStretch(1)
        table_host.lay.addLayout(csv_row)
        self.table_card = _PanelCard(t, "The numbers", table_host, flush=False)
        self.table_card.enable_export("Export table (PDF, SVG, PNG, TIFF)")
        self.table_card.export_requested.connect(self.table_export_requested.emit)
        curves_lay.addWidget(self.table_card)
        # SINGLE TRACKS, behind a switch under the numbers: each condition's curve over a sample of its tracks, a
        # card per condition, as many columns as fit (the "Per condition" page until 23 Nov 2026)
        tracks_row = QHBoxLayout()
        tracks_row.setContentsMargins(0, 0, 0, 0)
        tracks_row.setSpacing(rem(0.4))
        self.tracks_box = Checkbox(t, checked=False)
        self.tracks_box.toggled.connect(self.set_single_tracks)
        tracks_row.addWidget(self.tracks_box)
        tracks_row.addWidget(label("Single tracks", ty.ui("xs"), t.foreground))
        tracks_row.addSpacing(rem(0.5))
        _line, more = PAGE_NOTES["conditions"]
        tracks_row.addWidget(InfoLink(t, "Each condition's curve over a sample of its tracks", more))
        tracks_row.addStretch(1)
        curves_lay.addLayout(tracks_row)
        self._tracks_holder, self._grid = grid()
        self._tracks_holder.hide()
        curves_lay.addWidget(self._tracks_holder)
        # Checks: a card per condition, then what SMTracker 2 printed (the "v2" page until 23 Nov 2026)
        checks_page, checks_lay, _link = page("checks")
        holder, self._checks_grid = grid()
        checks_lay.addWidget(holder)
        line, more = PAGE_NOTES["v2"]
        checks_lay.addWidget(label(line, ty.ui("sm", "medium"), t.foreground))
        checks_lay.addWidget(InfoLink(t, "How v2's MSD differs", more))
        holder, self._v2_grid = grid()
        checks_lay.addWidget(holder)

        self._content = QStackedWidget()
        for widget in (curves, checks_page):
            self._content.addWidget(_scroll(widget))
        self._pages = QStackedWidget()
        self._pages.addWidget(self.placeholder)
        self._pages.addWidget(self._content)
        outer.addWidget(self._pages, 1)
        self._cards: dict[str, MSDCard] = {}
        self._check_cards: dict[str, tuple] = {}
        self._v2_cards: dict[str, tuple] = {}
        self._results: dict[str, dict] = {}
        self._conditions: list[str] = []
        self._labels: dict[str, str] = {}
        self._colours: dict[str, str] = {}
        self._table: dict | None = None
        self._ever_shown = False
        #: The last committed configuration, to seed the next dialog.
        self.last_config: dict = {}

    # -- typed ranges (30 Oct 2026) ------------------------------------------------------------------------------ #
    def typed_axes(self, key: str = "") -> dict:
        """The ranges typed for the shared chart (``""``) or condition ``key``'s (empty: automatic)."""
        return dict(self._typed.get(key) or {})

    def set_typed_axes(self, key: str, axes) -> None:
        """Ranges for ``key``'s chart (`axes_view.check`, refused by name), drawn now; all None: automatic."""
        axes = check(axis_rows("msd"), axes or {})
        if axes_empty(axes):
            self._typed.pop(key, None)
        else:
            self._typed[key] = axes
        chart = self.overlay.chart if not key else (self._cards[key].msd_body.chart if key in self._cards else None)
        if chart is not None:
            chart.axes = dict(self._typed.get(key) or {})
            chart.update()

    def open_axes(self, key: str = "") -> None:
        """The Axes popover for the shared chart (``""``) or a condition's, under its card's button."""
        if self._axes_popover is None:
            self._axes_popover = AxesPopover(self.t, self)
            self._axes_popover.applied.connect(lambda axes: self.set_typed_axes(self._axes_key, axes))
        self._axes_key = key
        card = self.overlay if not key else self._cards.get(key)
        self._axes_popover.show_for(axis_rows("msd"), self.typed_axes(key), None if card is None else card._inspect)

    def axes_state(self) -> dict:
        return {k: dict(v) for k, v in self._typed.items()}

    def set_axes_state(self, state: dict) -> None:
        for key, axes in (state or {}).items():
            self.set_typed_axes(key, axes)

    # -- the run ---------------------------------------------------------------------------------------------- #
    def begin_run(self, conditions: list, labels: dict | None = None,
                  colours: dict | None = None, config: dict | None = None) -> None:
        """A new Run over ``conditions``: drops every previous card."""
        for card in self._cards.values():
            discard(card)
        for card, _body in (*self._check_cards.values(), *self._v2_cards.values()):
            discard(card)
        self._cards, self._check_cards, self._v2_cards, self._results = {}, {}, {}, {}
        self._conditions = list(conditions)
        self._labels = dict(labels or {})
        self._colours = dict(colours or {})
        for key in conditions:
            name = self._labels.get(key, key)
            card = MSDCard(self.t, f"MSD · {name}")
            card.enable_export()
            card.export_requested.connect(lambda k=key: self.export_requested.emit(k))
            card.enable_inspect("Axes", glyph="ruler")
            card.inspect_requested.connect(lambda k=key: self.open_axes(k))
            card.msd_body.chart.axes = dict(self._typed.get(key) or {})
            self._cards[key] = card
            self._grid.addWidget(card)
            body = _ChecksBody(self.t)
            check = _PanelCard(self.t, f"{name} · checks", body, flush=False)
            check.enable_export()
            check.export_requested.connect(lambda k=key: self.checks_export_requested.emit(k))
            self._check_cards[key] = (check, body)
            self._checks_grid.addWidget(check)
            v2_body = _V2Body(self.t)
            v2 = _PanelCard(self.t, f"{name} · beside v2", v2_body, flush=False)
            v2.enable_export()
            v2.export_requested.connect(lambda k=key: self.v2_export_requested.emit(k))
            self._v2_cards[key] = (v2, v2_body)
            self._v2_grid.addWidget(v2)
        self.last_config = dict(config or {})
        self.overlay.clear("", self._conditions)
        self.table.set_table([], [])
        self._table = None
        self.placeholder.set_running(True)
        self._pages.setCurrentIndex(1 if self._ever_shown else 0)

    def set_order(self, order) -> None:
        """The conditions in ``order`` (`condition_order`, the Conditions tool's arrows, 3 Nov 2026): the cards, the
        checks and v2 cards, the shared chart's curves and the table move; nothing is refitted or recomputed."""
        self._conditions = order_apply(self._conditions, order)
        self._cards = {k: self._cards[k] for k in order_apply(list(self._cards), order)}
        self._check_cards = {k: self._check_cards[k] for k in order_apply(list(self._check_cards), order)}
        self._v2_cards = {k: self._v2_cards[k] for k in order_apply(list(self._v2_cards), order)}
        for grid, cards in ((self._grid, list(self._cards.values())),
                            (self._checks_grid, [c for c, _b in self._check_cards.values()]),
                            (self._v2_grid, [c for c, _b in self._v2_cards.values()])):
            grid.reorder(cards)
        self.overlay.set_order(order)
        if self._results:
            self._refresh_table()

    def set_running(self, on: bool) -> None:
        self.placeholder.set_running(on)

    def set_single_tracks(self, on: bool) -> None:
        """The single tracks behind each condition's curve, under the numbers on Curves, shown or not."""
        on = bool(on)
        if self.tracks_box.checked != on:
            self.tracks_box.set_checked(on, animate=False)
        self._tracks_holder.setVisible(on)

    def single_tracks(self) -> bool:
        return not self._tracks_holder.isHidden()

    def set_colours(self, colours: dict) -> None:
        """Re-colour every drawn card and the shared chart -- no refit."""
        self._colours.update(colours)
        for key, card in self._cards.items():
            if key in colours:
                card.set_colour(colours[key])
        for cards in (self._check_cards, self._v2_cards):
            for key, (_card, body) in cards.items():
                if key in colours:
                    body.set_colour(colours[key])
        self.overlay.set_colours(colours)

    def set_payload(self, condition: str, result: dict) -> None:
        card = self._cards.get(condition)
        if card is None:
            return
        self._results[condition] = result
        name = self._labels.get(condition, condition)
        colour = self._colours.get(condition, _DATA)
        if "error" in result:
            self.set_error(condition, result["error"])
            return
        card.set_result(result, colour)
        self._check_cards[condition][1].set_result(result, colour)
        self._v2_cards[condition][1].set_result(result, colour)
        self.overlay.set_entry(condition, name, result["chart"], colour)
        self._refresh_table()
        self._reveal()

    def set_error(self, condition: str, message: str) -> None:
        """A failure reported IN ITS OWN CARDS once there is a page to report it on."""
        self._results[condition] = {"error": message}
        card = self._cards.get(condition)
        if card is not None:
            card.set_error(message)
            self._check_cards[condition][1].set_error(message)
            self._v2_cards[condition][1].set_error(message)
            self.overlay.set_entry(condition, self._labels.get(condition, condition), None,
                                   self._colours.get(condition, _DATA))
        if self._cards:
            self._refresh_table()
            self._reveal()

    def finish(self) -> None:
        """Every condition landed: the table, which compares them, is final."""
        self._refresh_table()
        self.set_running(False)

    def _refresh_table(self) -> None:
        table = curves_table(self._results, self._conditions, self._labels)
        self._table = table if table["rows"] else None
        self.table.set_table(table["columns"], table["rows"])
        self.cells_csv.setEnabled(has_per_cell(self._results))
        self.overlay.set_caption(table["caption"])

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

    # -- what other parts of the window read --------------------------------------------------------------------- #
    def has_result(self) -> bool:
        return self._ever_shown

    def cards(self) -> dict:
        return dict(self._cards)

    def cells_csv_text(self) -> str | None:
        """Every cell's MSD as CSV (`msd_view.per_cell_csv`), or None before a cell has one."""
        if not has_per_cell(self._results):
            return None
        return per_cell_csv(self._results, self._conditions)

    def table_export(self) -> dict | None:
        """The Curves table's Export: the rows it shows, as the "table" figure's payload."""
        if not self._table:
            return None
        # its own file name, as the SQD table's (3 Nov 2026, reported: Export every figure wrote it as "table.*")
        return {"title": "MSD", "columns": list(self._table["columns"]), "rows": [list(r) for r in self._table["rows"]],
                "caption": self._table["caption"], "file_name": "msd_table"}

    def export_entries(self, key: str = "") -> list:
        """What an export draws: ``[{"key", "name", "colour", "result"}]`` for the
        condition ``key``, or for every condition with a result when ``key`` is ""."""
        out = []
        hidden = set() if key else self.overlay.hidden()
        for k, card in self._cards.items():
            result = card.msd_body._result
            if result is None or (key and k != key) or k in hidden:
                continue
            out.append({"key": k, "name": self._labels.get(k, k),
                        "colour": self._colours.get(k, _DATA), "result": result})
        return out
