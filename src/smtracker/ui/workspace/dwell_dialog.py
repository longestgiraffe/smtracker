"""SMMTrack's dwell times, as v2's Import drew them (`dwellTime.m`; added 17 Oct 2026): per condition the histogram
of trapping times on a log axis and their cumulative distribution, and a table of the counts. THIN QT: every number,
range and word is `smmtrack_view.dwell_payload`'s; this paints it."""
from __future__ import annotations

import math

from qtpy.QtCore import QPointF, QRectF, Qt
from qtpy.QtGui import QColor, QFontMetricsF, QPainter, QPen
from qtpy.QtWidgets import QHBoxLayout, QWidget

from ..theme import typography as ty
from ..widgets.controls import Button
from ..widgets.dialogs import CloseButton, OverlayDialog
from ..widgets.primitives import label
from .condition_key import ConditionKey
from .jumps_chart import _Table
from .toolbar import text_px

__all__ = ["DwellChart", "DwellDialog"]

_DATA = "#2b6cb0"


class DwellChart(QWidget):
    """Two panels over the same time axis: the probability density (log) above, the cumulative share below."""

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.setMinimumSize(520, 420)
        self._payload: dict | None = None
        self._colours: dict = {}
        self._hidden: set = set()
        self._font = ty.ui("xs", tabular=True, px=text_px(0.75))

    def set_payload(self, payload: dict | None, colours: dict) -> None:
        self._payload, self._colours = payload, dict(colours)
        self.update()

    def set_hidden(self, hidden: set) -> None:
        """The conditions an eye hides: not drawn; the axes stay."""
        self._hidden = set(hidden)
        self.update()

    def _shown(self):
        return [c for c in self._payload["conditions"] if c["n"] and c["key"] not in self._hidden]

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setFont(self._font)
        fm = QFontMetricsF(self._font)
        muted = QColor(self.t.muted_foreground)
        area = QRectF(self.contentsRect())
        pl = self._payload
        if not pl or not pl.get("available"):
            p.setPen(muted)
            p.drawText(area, Qt.AlignCenter, "No dwell events to draw.")
            p.end()
            return
        left, right = area.left() + 64.0, area.right() - 12.0
        gap = 2.6 * fm.height()
        height = (area.height() - gap - 2.4 * fm.height() - 8) / 2.0
        top = QRectF(left, area.top() + 0.6 * fm.height() + 4, right - left, height - 0.6 * fm.height())
        bottom = QRectF(left, top.bottom() + gap, right - left, height)
        x_lo, x_hi = pl["x_range"]

        def px(v):
            return left + (min(max(v, x_lo), x_hi) - x_lo) / (x_hi - x_lo) * (right - left)

        grid = QPen(QColor(self.t.border))
        grid.setWidthF(0.8)

        def x_axis(box, title):
            for v in pl["x_ticks"]:
                p.setPen(grid)
                p.drawLine(QPointF(px(v), box.top()), QPointF(px(v), box.bottom()))
                p.setPen(muted)
                p.drawText(QRectF(px(v) - 30, box.bottom() + 4, 60, fm.height()), Qt.AlignHCenter, f"{v:g}")
            p.drawText(QRectF(box.left(), box.bottom() + 4 + fm.height(), box.width(), fm.height()), Qt.AlignRight,
                       title)

        def y_title(box, text):
            p.save()
            p.translate(area.left() + 12, box.center().y())
            p.rotate(-90)
            p.setPen(muted)
            p.drawText(QRectF(-box.height() / 2, -fm.height() / 2, box.height(), fm.height()), Qt.AlignCenter, text)
            p.restore()

        # the density, log, one step a frame-wide bin
        rng = pl.get("density_range")
        if rng:
            y_lo, y_hi = math.log10(rng[0]), math.log10(rng[1])

            def dy(v):
                return top.bottom() - (math.log10(v) - y_lo) / (y_hi - y_lo) * top.height()

            for d in pl["density_ticks"]:
                p.setPen(grid)
                p.drawLine(QPointF(top.left(), dy(d)), QPointF(top.right(), dy(d)))
                p.setPen(muted)
                p.drawText(QRectF(area.left(), dy(d) - fm.height() / 2, top.left() - area.left() - 6, fm.height()),
                           Qt.AlignRight | Qt.AlignVCenter, f"{d:.0e}".replace("e-0", "e-").replace("e+0", "e"))
            edges = pl["edges_ms"]
            for c in self._shown():
                pen = QPen(QColor(self._colours.get(c["key"], _DATA)), 1.6)
                p.setPen(pen)
                for lo, hi, d in zip(edges[:-1], edges[1:], c["density"]):
                    if d > 0:
                        p.drawLine(QPointF(px(lo), dy(d)), QPointF(px(hi), dy(d)))
        x_axis(top, "Trapping time (ms)")
        y_title(top, "Probability density (1/ms)")
        # the cumulative share, a step at every event
        def cy(v):
            return bottom.bottom() - v * bottom.height()

        for v in (0.0, 0.5, 1.0):
            p.setPen(grid)
            p.drawLine(QPointF(bottom.left(), cy(v)), QPointF(bottom.right(), cy(v)))
            p.setPen(muted)
            p.drawText(QRectF(area.left(), cy(v) - fm.height() / 2, bottom.left() - area.left() - 6, fm.height()),
                       Qt.AlignRight | Qt.AlignVCenter, f"{v:g}")
        for c in self._shown():
            p.setPen(QPen(QColor(self._colours.get(c["key"], _DATA)), 1.6))
            prev_x, prev_y = px(x_lo), 0.0
            for e, share in c["cdf"]:
                x = px(e)
                p.drawLine(QPointF(prev_x, cy(prev_y)), QPointF(x, cy(prev_y)))
                p.drawLine(QPointF(x, cy(prev_y)), QPointF(x, cy(share)))
                prev_x, prev_y = x, share
            p.drawLine(QPointF(prev_x, cy(prev_y)), QPointF(px(x_hi), cy(prev_y)))
        x_axis(bottom, "Trapping time (ms)")
        y_title(bottom, "Cumulative share")
        p.end()


class DwellDialog(OverlayDialog):
    """The imported dwell times: a key, the two panels, the table, what was not matched, and the note."""

    def __init__(self, t, payload: dict, colours: dict, *, unmatched: str = "", content=None, parent=None):
        super().__init__(t, content, parent=parent)
        self.t = t
        self.payload = payload
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.addWidget(label("SMMTrack dwell times", ty.ui("base", "semibold"), t.foreground))
        head.addStretch(1)
        close = CloseButton(t)
        close.clicked.connect(self._dismiss)
        head.addWidget(close)
        self.card_lay.addLayout(head)
        #: An eye per condition, as on every chart conditions share
        self.key = ConditionKey(t)
        self.key.set_entries([(c["key"], c["label"], colours.get(c["key"], _DATA)) for c in payload["conditions"]])
        self.card_lay.addWidget(self.key)
        self.chart = DwellChart(t)
        self.chart.set_payload(payload, colours)
        self.key.hidden_changed.connect(self.chart.set_hidden)
        self.card_lay.addWidget(self.chart, 1)
        self.table = _Table(t)
        self.table.set_table(payload["columns"], payload["rows"])
        self.card_lay.addWidget(self.table)
        # EXPLICIT LINES, NOT WORD WRAP (a wrapped note reports less height than it takes)
        words = ["Dwell times as SMMTrack counted them. The events carry no cell or replicate,",
                 "so no test is offered: a test on pooled events treats every event as independent."]
        if unmatched:
            words.append(unmatched)
        self.card_lay.addWidget(label("\n".join(words), ty.ui("xs"), t.muted_foreground))
        footer = QHBoxLayout()
        footer.addStretch(1)
        done = Button(t, "Close", variant="outline")
        done.clicked.connect(self._dismiss)
        footer.addWidget(done)
        self.card_lay.addLayout(footer)
