"""The SQD populations as v2's crossed plot (agreed 17 Oct 2026; v2 `SMTracker.m:5019-5073`, axes "crossedsqd").

Each population of each condition one point: its fraction across, its D up (log), a bar each way where the Run has
a spread for it -- painted from `bubble_view.crossed_payload`, which decides every number and every bar. Colour is
the condition, the marker the population (v2 told populations apart only by legend text).

``axes`` (set by `BubbleCard.set_typed_axes`, rules in `smtracker.axes_view`): typed x / y ranges, drawn exactly, the
points clipped to them; ``drawn`` records the ranges painted."""
from __future__ import annotations

from ... import chart_text
from ...axes_view import apply_range

import math

from qtpy.QtCore import QPointF, QRectF, Qt
from qtpy.QtGui import QColor, QFontMetricsF, QPainter, QPainterPath, QPen
from qtpy.QtWidgets import QWidget

from ..theme import typography as ty
from .toolbar import text_px

_DATA = "#2b6cb0"


def marker_path(shape: str, centre: QPointF, r: float) -> QPainterPath:
    """The population's marker (`bubble_view.CROSSED_MARKERS`): circle, square, diamond or triangle."""
    path = QPainterPath()
    x, y = centre.x(), centre.y()
    if shape == "square":
        path.addRect(QRectF(x - r * 0.88, y - r * 0.88, 1.76 * r, 1.76 * r))
    elif shape == "diamond":
        path.moveTo(x, y - 1.2 * r)
        path.lineTo(x + 1.2 * r, y)
        path.lineTo(x, y + 1.2 * r)
        path.lineTo(x - 1.2 * r, y)
        path.closeSubpath()
    elif shape == "triangle":
        path.moveTo(x, y - 1.15 * r)
        path.lineTo(x + 1.1 * r, y + 0.85 * r)
        path.lineTo(x - 1.1 * r, y + 0.85 * r)
        path.closeSubpath()
    else:
        path.addEllipse(centre, r, r)
    return path


class CrossedChart(QWidget):
    """Fraction (%) against D (µm²/s, log), a point per population per condition with its bars."""

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.setMinimumHeight(260)
        self._payload: dict | None = None
        self._colours: dict = {}
        self._hidden: set = set()
        self._font = ty.ui("xs", tabular=True, px=text_px(0.75))
        #: Typed ranges (the card's Axes, `axes_view`): ``{"x": ..., "y": ...}``; the ranges as last drawn.
        self.axes: dict = {}
        self.drawn: dict = {}

    def set_payload(self, payload: dict | None, colours: dict) -> None:
        self._payload, self._colours = payload, dict(colours)
        self.update()

    def set_hidden(self, hidden: set) -> None:
        """The conditions an eye hides (17 Oct 2026): not drawn; the axes stay the Run's."""
        self._hidden = set(hidden)
        self.update()

    def set_colours(self, colours: dict) -> None:
        self._colours.update(colours)
        self.update()

    def payload(self) -> dict | None:
        return self._payload

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        pl = self._payload
        p.setFont(self._font)
        fm = QFontMetricsF(self._font)
        muted = QColor(self.t.muted_foreground)
        area = QRectF(self.contentsRect())
        if not pl or not pl.get("available"):
            p.setPen(muted)
            p.drawText(area, Qt.AlignCenter, (pl or {}).get("why", "Run to draw the populations."))
            p.end()
            return
        legend_h = fm.height() + 6
        box = QRectF(area.left() + 64.0, area.top() + legend_h + 8, max(10.0, area.width() - 80.0),
                     max(10.0, area.height() - legend_h - 16 - 2.4 * fm.height()))
        typed = self.axes or {}
        x_lo, x_hi = apply_range(*pl["x_range"], typed.get("x"))
        d_lo, d_hi = apply_range(*pl["y_range"], typed.get("y"), log=True)
        self.drawn = {"x": (x_lo, x_hi), "y": (d_lo, d_hi)}
        y_lo, y_hi = math.log10(d_lo), math.log10(d_hi)
        y_ticks = ([d for d in pl["y_ticks"] if d_lo <= d <= d_hi] if typed.get("y") is None
                   else chart_text.log_ticks(d_lo, d_hi))
        # a D of 0 at its "0" tick (`chart_text.zero_d_ticks`, the payload's labels)
        tick_label = dict(zip(pl["y_ticks"], pl.get("y_tick_labels") or [f"{d:g}" for d in pl["y_ticks"]]))
        zero_d = pl.get("zero_d")
        x_ticks = [v for v in pl["x_ticks"] if x_lo - 1e-9 <= v <= x_hi + 1e-9] or [x_lo, x_hi]

        def px(v):
            return box.left() + (v - x_lo) / (x_hi - x_lo) * box.width()

        def py(v):
            return box.bottom() - (math.log10(max(v, 1e-9)) - y_lo) / (y_hi - y_lo) * box.height()

        grid = QPen(QColor(self.t.border))
        grid.setWidthF(0.8)
        for d in y_ticks:
            p.setPen(grid)
            p.drawLine(QPointF(box.left(), py(d)), QPointF(box.right(), py(d)))
            p.setPen(muted)
            p.drawText(QRectF(area.left(), py(d) - fm.height() / 2, box.left() - area.left() - 6, fm.height()),
                       Qt.AlignRight | Qt.AlignVCenter, tick_label.get(d, f"{d:g}"))
        for v in x_ticks:
            p.setPen(muted)
            p.drawText(QRectF(px(v) - 30, box.bottom() + 4, 60, fm.height()), Qt.AlignHCenter, f"{v:g}")
        p.setPen(muted)
        p.drawText(QRectF(box.left(), box.bottom() + 4 + fm.height(), box.width(), fm.height()), Qt.AlignRight,
                   pl["x_label"])
        p.save()
        p.translate(area.left() + 12, box.center().y())
        p.rotate(-90)
        p.drawText(QRectF(-box.height() / 2, -fm.height() / 2, box.height(), fm.height()), Qt.AlignCenter,
                   pl["y_label"])
        p.restore()
        # the key: the conditions are the eye key's above the chart (`BubbleCard`); populations by marker here
        x = box.left()
        for shape, name in pl["legend_populations"]:
            p.setPen(QPen(muted, 1.0))
            p.setBrush(Qt.NoBrush)
            p.drawPath(marker_path(shape, QPointF(x + 5, area.top() + 2 + 0.5 * fm.height()), 4.0))
            p.setPen(muted)
            p.drawText(QRectF(x + 12, area.top() + 2, fm.horizontalAdvance(name) + 4, fm.height()),
                       Qt.AlignLeft | Qt.AlignVCenter, name)
            x += 26 + fm.horizontalAdvance(name)
        p.save()
        p.setClipRect(box.adjusted(-6, -6, 6, 6))           # a typed range: what lies outside it is not drawn
        for pt in pl["points"]:
            if pt["condition"] in self._hidden:
                continue
            colour = QColor(self._colours.get(pt["condition"], _DATA))
            centre = QPointF(px(pt["x"]), py(chart_text.log_drawn(pt["y"], zero_d)))
            pen = QPen(colour, 1.4)
            p.setPen(pen)
            cap = 4.0
            if pt["x_bar"]:
                a, b = px(pt["x_bar"][0]), px(pt["x_bar"][1])
                p.drawLine(QPointF(a, centre.y()), QPointF(b, centre.y()))
                for e in (a, b):
                    p.drawLine(QPointF(e, centre.y() - cap), QPointF(e, centre.y() + cap))
            if pt["y_bar"]:
                a, b = py(pt["y_bar"][0]), py(pt["y_bar"][1])
                p.drawLine(QPointF(centre.x(), a), QPointF(centre.x(), b))
                for e in (a, b):
                    p.drawLine(QPointF(centre.x() - cap, e), QPointF(centre.x() + cap, e))
            p.setPen(QPen(QColor(self.t.card), 1.2))
            p.setBrush(colour)
            p.drawPath(marker_path(pt["marker"], centre, 5.0))
        p.restore()
        p.end()
