"""Distances to marked foci (v2's Active Tools > Distances measurements, 17 Oct 2026): per condition the share of
localisations at each distance from the nearest focus, solid, beside chance (uniform points in the same outlines),
dashed; the table of cells, foci and the excess. THIN QT: every number and word is `distances_view.distance_payload`'s.
"""
from __future__ import annotations

from qtpy.QtCore import QPointF, QRectF, Qt
from qtpy.QtGui import QColor, QPen
from qtpy.QtWidgets import QHBoxLayout

from ..theme import typography as ty
from ..widgets.controls import Button
from ..widgets.dialogs import CloseButton, OverlayDialog
from ..widgets.primitives import label
from .condition_key import ConditionKey
from .jumps_chart import _JumpAxes, _Table

__all__ = ["DistanceChart", "DistancesDialog"]

_DATA = "#2b6cb0"


class DistanceChart(_JumpAxes):
    """Probability per 50 nm bin against the distance to the nearest focus: each shown condition's localisations
    (solid) and its chance (dashed)."""

    _ROOMY_H = 140

    def __init__(self, t, parent=None):
        super().__init__(t, parent)
        self.setMinimumSize(560, 300)
        self._colours: dict = {}
        self._hidden: set = set()

    def set_look(self, colours: dict, hidden: set) -> None:
        self._colours, self._hidden = dict(colours), set(hidden)
        self.update()

    def _paint(self, p, box: QRectF, pl) -> None:
        edges = pl["edges"]
        shown = [c for c in pl["conditions"] if c.get("n_cells") and c["key"] not in self._hidden]
        if not shown:
            return
        top = max(max(c["hist"]) for c in shown)
        top = max(top, max((max(c["chance_hist"]) for c in shown if c.get("chance_hist")), default=0.0))
        lo, hi = self._grid(p, box, 0.0, top * 1.08, "Share of localisations")
        x0, x1 = edges[0], edges[-1]

        def px(x):
            return box.left() + (x - x0) / (x1 - x0) * box.width()

        def py(v):
            return self._map_y(box, v, lo, hi)

        def steps(values, pen):
            p.setPen(pen)
            for a, b, v in zip(edges, edges[1:], values):
                p.drawLine(QPointF(px(a), py(v)), QPointF(px(b), py(v)))
            for (a, b), (v, w) in zip(zip(edges[1:], edges[2:]), zip(values, values[1:])):
                p.drawLine(QPointF(px(a), py(v)), QPointF(px(a), py(w)))

        p.save()
        p.setClipRect(box.adjusted(-1, -2, 1, 1))
        for c in shown:
            colour = QColor(self._colours.get(c["key"], _DATA))
            if c.get("chance_hist"):
                dash = QPen(colour, 1.2)
                dash.setStyle(Qt.DashLine)
                steps(c["chance_hist"], dash)
            steps(c["hist"], QPen(colour, 2.0))
        p.restore()
        self._legend(p, [(self.t.foreground, "localisations (solid)", "line"),
                         (self.t.muted_foreground, "chance, uniform in the outline (dashed)", "line")],
                     QPointF(box.right() - 4, box.top() + 4), horizontal=False, right_aligned=True)
        self._x_numeric(p, box, x0, x1, "Distance to the nearest focus (µm)")


class DistancesDialog(OverlayDialog):
    """The eye key, the chart, the table, what was skipped, and the note."""

    def __init__(self, t, payload: dict, colours: dict, *, content=None, parent=None):
        super().__init__(t, content, parent=parent)
        self.t = t
        self.payload = payload
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.addWidget(label("Distances to foci", ty.ui("base", "semibold"), t.foreground))
        head.addStretch(1)
        close = CloseButton(t)
        close.clicked.connect(self._dismiss)
        head.addWidget(close)
        self.card_lay.addLayout(head)
        measured = [c for c in payload["conditions"] if c.get("n_cells")]
        self.key = ConditionKey(t)
        self.key.set_entries([(c["key"], c["label"], colours.get(c["key"], _DATA)) for c in measured])
        self.card_lay.addWidget(self.key)
        self.chart = DistanceChart(t)
        self.chart.set_look(colours, set())
        self.chart.set_payload(payload)
        self.key.hidden_changed.connect(lambda hidden: self.chart.set_look(colours, hidden))
        self.card_lay.addWidget(self.chart, 1)
        self.table = _Table(t)
        self.table.set_table(payload["columns"], payload["rows"])
        self.card_lay.addWidget(self.table)
        # EXPLICIT LINES, NOT WORD WRAP (a wrapped note reports less height than it takes)
        words = ["Distance from each localisation of a marked cell's tracks to its nearest focus; dashed: uniform",
                 "points in the same outline, measured to the same foci. Excess is a cell's median minus its chance",
                 "median (negative means closer to the foci than chance), averaged over cells; across replicates, the",
                 "SD of the replicates' means."]
        skipped = [s for c in payload["conditions"] for s in c.get("skipped", [])]
        if skipped:
            words.append(f"Not measured: {'; '.join(skipped[:3])}" + (" ..." if len(skipped) > 3 else ""))
        self.card_lay.addWidget(label("\n".join(words), ty.ui("xs"), t.muted_foreground))
        footer = QHBoxLayout()
        footer.addStretch(1)
        done = Button(t, "Close", variant="outline")
        done.clicked.connect(self._dismiss)
        footer.addWidget(done)
        self.card_lay.addLayout(footer)
