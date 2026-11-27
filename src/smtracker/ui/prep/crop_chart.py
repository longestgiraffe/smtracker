"""The crop chart: one movie's frame mean, spots per frame, and the kept range on its own
scale, painted in the workspace's chart language, with the crop edges draggable.

THE SAME THREE PANELS `plots.draw_crop_curves` has always drawn, because the decision is
made by reading them together -- the frame mean says how far the field has bleached, the
density whether what is left is single molecules, and the kept range alone shows whether
it is still falling. Painted with QPainter like every workspace chart (`qc_chart`), not
matplotlib: the classic window redrew a three-axes matplotlib figure on every edit and
every movie switch, which is much of why "changing some values takes long".

The data arrives as `prep_view.crop_chart`'s plain payload, already thinned to about 1200
points per curve with every spike kept. While an edge is dragged only that edge moves
and the panels repaint; the payload is rebuilt once, on release.
"""
from __future__ import annotations

from qtpy.QtCore import QPointF, QRectF, Qt, Signal
from qtpy.QtGui import QColor, QPainterPath, QPen
from qtpy.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from ..theme import hcl
from ..theme import typography as ty
from ..workspace.qc_chart import _QCChart

__all__ = ["CropChart", "chart_colours"]

#: The hues the classic matplotlib figure used, so the new chart reads the same:
#: whole acquisition grey-blue, kept range orange, floor red, proposal green.
_HUES = {"whole": "#7f9bb5", "kept": "#d9791f", "floor": "#a1362a", "proposal": "#2f8f5b"}


def chart_colours(t) -> dict:
    """`_HUES` fitted to this theme's page, so the dark theme keeps its contrast."""
    ground = hcl.GROUNDS["dark" if getattr(t, "_dark", False) else "light"]
    return {k: hcl.fit_to_ground(v, ground) for k, v in _HUES.items()}


class _Panel(_QCChart):
    """One of the three panels. ``kind`` is "mean", "density" or "zoom"."""

    _ROOMY_H = 0           # always room for the x axis it is asked to draw
    #: Room for the last x tick label ("4000") to the right of the plot.
    _RIGHT = 22

    def __init__(self, t, kind: str, owner: "CropChart", parent=None):
        super().__init__(t, parent)
        self.kind = kind
        self._owner = owner
        self._colours = chart_colours(t)
        self.setMinimumHeight(110)
        if kind == "mean":
            self._TOP_MARGIN = 30          # the title sits above the plot
        self._FULL_BOTTOM = 34 if kind != "mean" else 8
        self._BOTTOM_MARGIN = self._FULL_BOTTOM

    # -- geometry shared with the owner, for dragging -------------------------- #
    def x_range(self) -> tuple:
        pl = self._payload
        if pl is None:
            return (1.0, 2.0)
        if self.kind == "zoom":
            lo, hi = pl["zoom"]["range"]
            return float(lo), float(max(hi, lo + 1))
        return 1.0, float(max(pl["n_frames"], 2))

    def frame_at(self, x: float) -> float:
        box = self._plot_box()
        lo, hi = self.x_range()
        return lo + (x - box.left()) / max(1.0, box.width()) * (hi - lo)

    def x_of(self, frame: float) -> float:
        box = self._plot_box()
        lo, hi = self.x_range()
        return box.left() + (float(frame) - lo) / (hi - lo) * box.width()

    # -- painting ---------------------------------------------------------------- #
    def _paint(self, p, box: QRectF, pl) -> None:
        series = pl[self.kind]
        frames, values = series["frames"], series["values"]
        if not frames:
            return
        lo_v, hi_v = min(values), max(values)
        pad = 0.06 * (hi_v - lo_v) if hi_v > lo_v else max(1.0, abs(hi_v) * 0.05)
        floor_v = 0.0 if self.kind == "density" else lo_v - pad
        y_label = {"mean": "frame mean", "density": "spots / frame",
                   "zoom": pl["zoom"]["label"]}[self.kind]
        axis_lo, axis_hi = self._grid(p, box, floor_v, hi_v + pad, y_label)
        x_lo, x_hi = self.x_range()

        def pt(f, v):
            return QPointF(box.left() + (f - x_lo) / (x_hi - x_lo) * box.width(),
                           self._map_y(box, v, axis_lo, axis_hi))

        c = self._colours
        edges = self._owner.edges()
        whole = edges is None
        # THE WHOLE CURVE, then the kept part over it in its own colour. The zoom panel
        # is all "kept" unless it is the floor PREVIEW, which is drawn in the whole
        # colour so a preview cannot be mistaken for a crop somebody set.
        if self.kind == "zoom":
            colour = c["whole"] if pl["zoom"]["preview"] else c["kept"]
            self._polyline(p, [pt(f, v) for f, v in zip(frames, values)], colour, 1.1)
        else:
            self._polyline(p, [pt(f, v) for f, v in zip(frames, values)], c["whole"], 0.9)
            if not whole:
                a, b = edges
                kept = [pt(f, v) for f, v in zip(frames, values) if a <= f <= b]
                self._polyline(p, kept, c["kept"], 1.5)
            self._overlays(p, box, pl, edges)
        self._geom = {"box": box}
        if self.kind != "mean":
            self._x_numeric(p, box, x_lo, x_hi, "frame")
        self._labels(p, box, pl)

    def _polyline(self, p, points, colour: str, width: float) -> None:
        if len(points) < 2:
            return
        path = QPainterPath(points[0])
        for q in points[1:]:
            path.lineTo(q)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(colour), width))
        p.drawPath(path)

    def _overlays(self, p, box: QRectF, pl, edges) -> None:
        """The floor (dashed, labelled), the proposal (dotted) and the crop edges
        (solid, with a grip at the top -- these are what a drag moves)."""
        c = self._colours

        def vline(frame, colour, style, width):
            x = self.x_of(frame)
            if box.left() - 1 <= x <= box.right() + 1:
                pen = QPen(QColor(colour), width)
                pen.setStyle(style)
                p.setPen(pen)
                p.drawLine(QPointF(x, box.top()), QPointF(x, box.bottom()))
            return x

        if pl["floor"] is not None:
            vline(pl["floor"], c["floor"], Qt.DashLine, 1.1)
        if pl["proposal"] is not None:
            for frame in pl["proposal"]:
                vline(frame, c["proposal"], Qt.DotLine, 1.2)
        if edges is not None:
            for frame in edges:
                x = vline(frame, c["kept"], Qt.SolidLine, 1.6)
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(c["kept"]))
                p.drawRoundedRect(QRectF(x - 3.5, box.top() - 1, 7, 10), 2, 2)

    def _labels(self, p, box: QRectF, pl) -> None:
        muted = QColor(self.t.muted_foreground)
        if self.kind == "mean":
            font = ty.ui("sm", "semibold")
            p.setFont(font)
            p.setPen(QColor(self.t.foreground))
            title = self._owner.title()
            p.drawText(QRectF(box.left(), 4, box.width(), 20),
                       Qt.AlignLeft | Qt.AlignVCenter, title)
            entries = [(self._colours["whole"], "whole acquisition", "line")]
            if self._owner.edges() is not None:
                entries.append((self._colours["kept"], "kept", "line"))
            if pl["floor"] is not None:
                entries.append((self._colours["floor"], pl["floor_label"], "line"))
            if box.width() >= 320:
                self._legend(p, entries, QPointF(box.right() - 4, box.top() + 4),
                             horizontal=False, right_aligned=True)
        elif self.kind == "zoom" and pl["verdict"]:
            p.setFont(ty.ui("xs"))
            p.setPen(muted)
            p.drawText(QRectF(box.left() + 4, box.top() - 15, box.width() - 8, 14),
                       Qt.AlignLeft | Qt.AlignVCenter, pl["verdict"])

    # -- dragging: the owner decides; the panel only reports --------------------- #
    def mousePressEvent(self, e) -> None:
        if self.kind != "zoom" and e.button() == Qt.LeftButton:
            self._owner._press(self, e.position().x())
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e) -> None:
        if self._owner._dragging is not None:
            self._owner._move(self, e.position().x())
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e) -> None:
        if e.button() == Qt.LeftButton:
            self._owner._release()
        super().mouseReleaseEvent(e)


class CropChart(QWidget):
    """The three panels, one payload, and the drag that moves the crop's edges.

    ``crop_moving(start, end)`` fires while an edge is dragged (for the number fields);
    ``crop_committed(start, end)`` once, on release (for the session). An edge is grabbed
    within max(6 px, 2 % of the plot) of it; it is clamped inside the movie and can never
    cross the other edge.
    """

    crop_moving = Signal(int, int)
    crop_committed = Signal(int, int)

    #: How close a press must land to an edge to grab it, as a fraction of the plot.
    GRAB_FRACTION = 0.02
    GRAB_MIN_PX = 6.0

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.setStyleSheet("background:transparent;")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self.panels = {kind: _Panel(t, kind, self) for kind in ("mean", "density", "zoom")}
        for kind, stretch in (("mean", 5), ("density", 4), ("zoom", 5)):
            lay.addWidget(self.panels[kind], stretch)
        self._payload: dict | None = None
        #: The edges as drawn, which during a drag run ahead of the payload.
        self._edges: tuple | None = None
        #: "start" or "end" while dragging, else None.
        self._dragging: str | None = None

    # -- data ------------------------------------------------------------------- #
    def set_payload(self, payload: dict) -> None:
        self._payload = payload
        self._edges = None if payload["crop"] is None else tuple(payload["crop"])
        for panel in self.panels.values():
            panel.set_payload(payload)

    def set_message(self, text: str) -> None:
        self._payload, self._edges = None, None
        for kind, panel in self.panels.items():
            panel.set_message(text if kind == "mean" else "")

    def set_edges(self, start: int, end: int) -> None:
        """Move the drawn edges (from the number fields) without rebuilding the payload."""
        if self._payload is None:
            return
        self._edges = (int(start), int(end))
        for panel in self.panels.values():
            panel.update()

    def edges(self) -> tuple | None:
        return self._edges

    def title(self) -> str:
        pl = self._payload or {}
        if self._dragging is not None and self._edges is not None:
            return f"{pl.get('movie', '')}    keeping {self._edges[0]}–{self._edges[1]}"
        return pl.get("title", "")

    # -- dragging -------------------------------------------------------------- #
    def _press(self, panel: _Panel, x: float) -> None:
        if self._payload is None:
            return
        n = int(self._payload["n_frames"])
        edges = self._edges or (1, n)
        grab = max(self.GRAB_MIN_PX, panel._plot_box().width() * self.GRAB_FRACTION)
        distances = {"start": abs(panel.x_of(edges[0]) - x),
                     "end": abs(panel.x_of(edges[1]) - x)}
        which = min(distances, key=distances.get)
        if distances[which] <= grab:
            self._edges = edges
            self._dragging = which

    def _move(self, panel: _Panel, x: float) -> None:
        if self._payload is None or self._edges is None:
            return
        n = int(self._payload["n_frames"])
        frame = int(round(panel.frame_at(x)))
        start, end = self._edges
        if self._dragging == "start":
            start = max(1, min(frame, end - 1))
        else:
            end = min(n, max(frame, start + 1))
        if (start, end) != self._edges:
            self._edges = (start, end)
            for p in self.panels.values():
                p.update()
            self.crop_moving.emit(start, end)

    def _release(self) -> None:
        if self._dragging is None:
            return
        self._dragging = None
        if self._edges is not None:
            self.crop_committed.emit(*self._edges)
