"""The generator window's pictures (25 Nov 2026): each page's live drawing, painted from a plain payload.

THIN QT: every number drawn is `smtracker.generator_preview`'s (positions in µm or as fractions of the axis, the
frames of the animation, the grey values of a frame); these classes scale them to the widget and paint. Nothing here
imports an analysis module.

  BubblePicture   the declared populations as the SQD chart draws them (`generator_preview.bubbles`)
  CellPicture     one cell with its places and molecules moving in them, played on a timer that runs only while
                  the picture is shown (`generator_preview.cell_animation`)
  DiagramPicture  the populations and the rates between them (`generator_preview.state_diagram`)
  CurvePicture    the response after an addition, the movies on its clock, the shares before and after
                  (`generator_preview.kinetic_curve`)
  FieldPicture    the camera field and its cells at the pixel's scale (`generator_preview.field_layout`)
  FramePicture    one rendered movie frame (`generator_preview.frame_preview`)
"""
from __future__ import annotations

import math

from qtpy.QtCore import QPointF, QRectF, QSize, Qt, QTimer
from qtpy.QtGui import QBrush, QColor, QImage, QPainter, QPainterPath, QPen, QPolygonF
from qtpy.QtWidgets import QSizePolicy, QWidget

from ..theme import typography as ty
from ..widgets.settings import CUT_INK

__all__ = ["BubblePicture", "CellPicture", "DiagramPicture", "CurvePicture", "FieldPicture", "FramePicture"]

#: The places' words in the cell picture's legend (`generator._REGION_WORDS` says them in the form).
_PLACE_WORDS = {"cell": "whole cell", "nucleoid": "nucleoid", "membrane": "membrane", "poles": "poles",
                "midcell": "mid-cell", "foci": "foci"}


def _colour(hex_: str, alpha: int = 255) -> QColor:
    c = QColor(hex_)
    c.setAlpha(alpha)
    return c


class _Picture(QWidget):
    """A picture: a payload, or a message while there is none (computing, or why there is nothing to draw)."""

    def __init__(self, t, *, min_height: int = 240, parent=None):
        super().__init__(parent)
        self.t = t
        self.payload: dict | None = None
        self.message = "Drawing…"
        self.setMinimumHeight(min_height)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def sizeHint(self) -> QSize:                           # noqa: N802 - Qt's name
        return QSize(384, 320)

    def set_payload(self, payload: dict | None) -> None:
        self.payload = payload
        self.update()

    def set_message(self, text: str) -> None:
        self.payload = None
        self.message = text
        self.update()

    def paintEvent(self, _event):                           # noqa: N802 - Qt's name
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        rect = QRectF(self.rect())
        if self.payload is None:
            self._words(p, rect, self.message, self.t.muted_foreground)
            return
        error = self.payload.get("error")
        if error and not self._draws_with_error():
            self._words(p, rect, error, CUT_INK)
            return
        self.draw(p, rect)

    def _draws_with_error(self) -> bool:
        return False

    def _words(self, p: QPainter, rect: QRectF, text: str, colour: str, *, size: str = "sm",
               flags=Qt.AlignCenter | Qt.TextWordWrap) -> None:
        p.setPen(QColor(colour))
        p.setFont(ty.ui(size))
        p.drawText(rect.adjusted(16, 8, -16, -8), int(flags), text)

    def draw(self, p: QPainter, rect: QRectF) -> None:      # pragma: no cover - each picture's own
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Populations: the bubbles
# --------------------------------------------------------------------------- #


class BubblePicture(_Picture):
    """Columns of conditions, D up a log axis, each population a bubble whose area is its share."""

    def draw(self, p: QPainter, rect: QRectF) -> None:
        t, data = self.t, self.payload
        left, bottom, top, right = 54.0, 40.0, 14.0, 12.0
        plot = QRectF(rect.left() + left, rect.top() + top, rect.width() - left - right,
                      rect.height() - top - bottom)
        if plot.width() < 40 or plot.height() < 40:
            return

        def y_px(y):
            return plot.bottom() - y * plot.height()

        p.setFont(ty.ui("xs"))
        for words, y in data["ticks"]:
            yy = y_px(y)
            p.setPen(QPen(QColor(t.border), 1))
            p.drawLine(QPointF(plot.left(), yy), QPointF(plot.right(), yy))
            p.setPen(QColor(t.muted_foreground))
            p.drawText(QRectF(rect.left(), yy - 9, left - 8, 18), int(Qt.AlignRight | Qt.AlignVCenter), words)
        p.save()
        p.translate(rect.left() + 12, plot.center().y())
        p.rotate(-90)
        p.drawText(QRectF(-80, -10, 160, 20), int(Qt.AlignCenter), "D (µm²/s)")
        p.restore()
        columns = data["columns"]
        n = max(len(columns), 1)
        col_w = plot.width() / n
        r_max = max(6.0, min(col_w * 0.36, plot.height() * 0.13, 34.0))
        for ci, name in enumerate(columns):
            cx = plot.left() + (ci + 0.5) * col_w
            p.setPen(QColor(t.foreground))
            p.setFont(ty.ui("xs", "medium"))
            text = p.fontMetrics().elidedText(name, Qt.ElideRight, int(col_w - 6))
            p.drawText(QRectF(cx - col_w / 2, plot.bottom() + 8, col_w, 18), int(Qt.AlignCenter), text)
        for pt in sorted(data["points"], key=lambda q: -q["r"]):
            cx = plot.left() + (pt["column"] + 0.5) * col_w
            cy = y_px(pt["y"])
            r = max(3.0, pt["r"] * r_max)
            p.setPen(QPen(_colour(pt["colour"]), 1.4))
            p.setBrush(QBrush(_colour(pt["colour"], 150)))
            p.drawEllipse(QPointF(cx, cy), r, r)
            p.setPen(QColor(t.foreground))
            p.setFont(ty.ui("xs", tabular=True))
            p.drawText(QRectF(cx + r + 4, cy - 9, col_w / 2, 18), int(Qt.AlignLeft | Qt.AlignVCenter),
                       f"{100 * pt['share']:.0f} %")


# --------------------------------------------------------------------------- #
# Where in the cell: the animated cell
# --------------------------------------------------------------------------- #


class CellPicture(_Picture):
    """One cell with the molecules moving, each place tinted in the colour of the population living there; under
    it the six places, each on a small cell of its own. Played forward and back on a timer (Brownian motion and
    switching between two states look the same either way, so the loop has no jump); the timer runs only while the
    picture is shown."""

    TRAIL = 3

    def __init__(self, t, *, fps: int = 20, parent=None):
        super().__init__(t, min_height=260, parent=parent)
        self.frame = 0
        self._step = 1
        self.timer = QTimer(self)
        self.timer.setInterval(int(round(1000 / max(fps, 1))))
        self.timer.timeout.connect(self._advance)

    def set_payload(self, payload: dict | None) -> None:
        self.frame, self._step = 0, 1
        super().set_payload(payload)
        self._sync_timer()

    def playing(self) -> bool:
        return self.timer.isActive()

    def _frames(self) -> list:
        return (self.payload or {}).get("frames") or []

    def _sync_timer(self) -> None:
        if self.isVisible() and len(self._frames()) > 1:
            if not self.timer.isActive():
                self.timer.start()
        else:
            self.timer.stop()

    def showEvent(self, event):                             # noqa: N802 - Qt's name
        super().showEvent(event)
        self._sync_timer()

    def hideEvent(self, event):                             # noqa: N802 - Qt's name
        self.timer.stop()
        super().hideEvent(event)

    def _advance(self) -> None:
        n = len(self._frames())
        if n < 2:
            self.timer.stop()
            return
        nxt = self.frame + self._step
        if not 0 <= nxt < n:
            self._step = -self._step
            nxt = self.frame + self._step
        self.frame = nxt
        self.update()

    def _draws_with_error(self) -> bool:
        return bool((self.payload or {}).get("outline"))

    def _paths(self, data: dict, cx: float, cy: float, scale: float) -> tuple:
        """The cell's outline and each place as painter paths, the cell centred on (cx, cy) at ``scale`` px per
        µm."""
        width = data["width"]

        def pt(x, y):
            return QPointF(cx + x * scale, cy - y * scale)

        def polygon(points):
            path = QPainterPath()
            if points:
                path.addPolygon(QPolygonF([pt(x, y) for x, y in points]))
                path.closeSubpath()
            return path

        outline = polygon(data["outline"])
        places = {"cell": QPainterPath(outline)}
        lobes = QPainterPath()
        for centre, half_l, half_w in data["lobes"]:
            lobes.addEllipse(QRectF(pt(centre - half_l, half_w), pt(centre + half_l, -half_w)))
        places["nucleoid"] = lobes
        if data.get("inner"):
            places["membrane"] = outline.subtracted(polygon(data["inner"]))
        far = data["length"] / 2.0 + width
        caps = QPainterPath()
        caps.addRect(QRectF(pt(data["pole_x"], width), pt(far, -width)))
        caps.addRect(QRectF(pt(-far, width), pt(-data["pole_x"], -width)))
        places["poles"] = outline.intersected(caps)
        band = QPainterPath()
        band.addRect(QRectF(pt(-data["midcell_half"], width), pt(data["midcell_half"], -width)))
        places["midcell"] = outline.intersected(band)
        foci = QPainterPath()
        rho = max(2.5, data["focus_radius"] * scale)
        for x, y in data["foci"]:
            foci.addEllipse(pt(x, y), rho, rho)
        places["foci"] = foci
        return outline, places, pt

    def draw(self, p: QPainter, rect: QRectF) -> None:
        t, data = self.t, self.payload
        names = data.get("names") or []
        colours = data.get("colours") or ["#000000"]
        tint = {}
        for colour, where in zip(colours, data.get("wheres") or []):
            tint.setdefault(where, colour)
        # the places, each on a small cell of its own, coloured where a population lives
        strip_h = 2 * 58.0
        legend_h = 20.0 * max(1, len(names)) + 6
        area = QRectF(rect.left() + 10, rect.top() + 6, rect.width() - 20,
                      rect.height() - strip_h - legend_h - 18)
        length, width = data["length"], data["width"]
        scale = min(area.width() / (length * 1.04), area.height() / (width * 1.8), 160.0)
        outline, places, pt = self._paths(data, area.center().x(), area.center().y(), scale)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(t.muted))
        p.drawPath(outline)
        p.setBrush(QColor(t.border))
        p.drawPath(places["nucleoid"])
        for where, path in places.items():
            if where != "cell" and where in tint:
                p.setPen(QPen(_colour(tint[where], 160), 1) if where == "foci" else Qt.NoPen)
                p.setBrush(_colour(tint[where], 70))
                p.drawPath(path)
        if "foci" not in tint:
            p.setPen(QPen(QColor(t.ring), 1, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawPath(places["foci"])
        p.setPen(QPen(QColor(t.foreground), 1.6))
        p.setBrush(Qt.NoBrush)
        p.drawPath(outline)
        # the molecules and a short trail
        frames = self._frames()
        if frames:
            k = min(self.frame, len(frames) - 1)
            back = -self._step
            span = [j for j in (k + back * i for i in range(self.TRAIL, -1, -1)) if 0 <= j < len(frames)]
            for m, (x, y, s_) in enumerate(frames[k]):
                colour = colours[s_ % len(colours)]
                trail = [frames[j][m] for j in span]
                p.setPen(QPen(_colour(colour, 70), 1.0))
                for a, b in zip(trail, trail[1:]):
                    p.drawLine(pt(a[0], a[1]), pt(b[0], b[1]))
                p.setPen(QPen(QColor(t.card), 1))
                p.setBrush(_colour(colour))
                p.drawEllipse(pt(x, y), 3.6, 3.6)
        # the strip of places: 2 rows of 3
        top = area.bottom() + 8
        col_w = rect.width() / 3.0
        mini = min((col_w - 24) / (length * 1.04), 26.0 / width)
        p.setFont(ty.ui("xs"))
        for i, where in enumerate(("cell", "nucleoid", "membrane", "poles", "midcell", "foci")):
            cx = rect.left() + (i % 3 + 0.5) * col_w
            cy = top + (i // 3) * 58.0 + 16
            o, pl, _pt = self._paths(data, cx, cy, mini)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(t.muted))
            p.drawPath(o)
            colour = tint.get(where)
            if where in pl:
                p.setPen(QPen(_colour(colour), 1) if (where == "foci" and colour) else Qt.NoPen)
                p.setBrush(_colour(colour, 150) if colour else _colour(t.ring, 120))
                p.drawPath(pl[where])
            p.setPen(QPen(QColor(t.ring), 1))
            p.setBrush(Qt.NoBrush)
            p.drawPath(o)
            who = [n for n, w_ in zip(names, data.get("wheres") or []) if w_ == where]
            words = _PLACE_WORDS[where] + (f": {', '.join(who)}" if who else "")
            p.setPen(QColor(t.foreground if who else t.muted_foreground))
            p.drawText(QRectF(cx - col_w / 2 + 2, cy + 16, col_w - 4, 16), int(Qt.AlignCenter),
                       p.fontMetrics().elidedText(words, Qt.ElideRight, int(col_w - 6)))
        # the legend: each population, its colour and its place
        y0 = rect.bottom() - legend_h + 4
        for i, name in enumerate(names):
            colour = colours[i % len(colours)]
            where = (data.get("wheres") or [""] * (i + 1))[i]
            p.setPen(Qt.NoPen)
            p.setBrush(_colour(colour))
            p.drawEllipse(QPointF(rect.left() + 16, y0 + 20 * i + 8), 5, 5)
            p.setPen(QColor(t.foreground))
            p.drawText(QRectF(rect.left() + 28, y0 + 20 * i, rect.width() - 40, 16),
                       int(Qt.AlignLeft | Qt.AlignVCenter), f"{name}: {_PLACE_WORDS.get(where, where)}")
        if data.get("error"):
            self._words(p, QRectF(rect.left(), rect.top(), rect.width(), 40), data["error"], CUT_INK, size="xs")


# --------------------------------------------------------------------------- #
# Switching: the state diagram
# --------------------------------------------------------------------------- #


class DiagramPicture(_Picture):
    """The populations as circles (area = share) and each rate as an arrow, bent so that the two directions
    between a pair are both seen."""

    def draw(self, p: QPainter, rect: QRectF) -> None:
        t, data = self.t, self.payload
        nodes = data["nodes"]
        if not nodes:
            return
        note_h = 52.0 if data.get("note") else 0.0
        # room over the top circle for its words where three or more stand in a ring
        area = rect.adjusted(20, 16 + (56 if len(nodes) > 2 else 0), -20, -40 - note_h)
        side = min(area.width(), area.height() * 1.4)
        r_max = max(16.0, min(side * 0.17, 72.0))
        r_min = 14.0

        def centre(n):
            return QPointF(area.left() + n["x"] * area.width(), area.top() + n["y"] * area.height())

        def radius(n):
            return r_min + (r_max - r_min) * math.sqrt(max(n["share"], 0.0))

        pairs = {(e["from"], e["to"]) for e in data["edges"]}
        for e in data["edges"]:
            a, b = nodes[e["from"]], nodes[e["to"]]
            pa, pb = centre(a), centre(b)
            dx, dy = pb.x() - pa.x(), pb.y() - pa.y()
            dist = math.hypot(dx, dy) or 1.0
            ux, uy = dx / dist, dy / dist
            nx, ny = -uy, ux
            bend = (0.22 if len(nodes) == 2 else 0.16) * dist if (e["to"], e["from"]) in pairs else 0.0
            start = QPointF(pa.x() + ux * (radius(a) + 4) + nx * bend * 0.25,
                            pa.y() + uy * (radius(a) + 4) + ny * bend * 0.25)
            end = QPointF(pb.x() - ux * (radius(b) + 6) + nx * bend * 0.25,
                          pb.y() - uy * (radius(b) + 6) + ny * bend * 0.25)
            ctrl = QPointF((start.x() + end.x()) / 2 + nx * bend, (start.y() + end.y()) / 2 + ny * bend)
            path = QPainterPath(start)
            path.quadTo(ctrl, end)
            p.setPen(QPen(QColor(t.muted_foreground), 1.6))
            p.setBrush(Qt.NoBrush)
            p.drawPath(path)
            tx, ty_ = end.x() - ctrl.x(), end.y() - ctrl.y()
            tl = math.hypot(tx, ty_) or 1.0
            tx, ty_ = tx / tl, ty_ / tl
            head = QPolygonF([end, QPointF(end.x() - 9 * tx + 4.5 * ty_, end.y() - 9 * ty_ - 4.5 * tx),
                              QPointF(end.x() - 9 * tx - 4.5 * ty_, end.y() - 9 * ty_ + 4.5 * tx)])
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(t.muted_foreground))
            p.drawPolygon(head)
            # with three or more, each direction's rate nearer its own start, so a pair's two never meet
            mid = path.pointAtPercent(0.5 if len(nodes) == 2 else 0.36)
            words = f"{e['rate']:.3g} per s"
            p.setFont(ty.ui("xs", tabular=True))
            w = p.fontMetrics().horizontalAdvance(words) + 10
            lift = 12.0 if bend else 0.0
            box = QRectF(mid.x() - w / 2 + nx * lift, mid.y() - 9 + ny * lift, w, 18)
            p.setBrush(QColor(t.card))
            p.drawRoundedRect(box, 4, 4)
            p.setPen(QColor(t.foreground))
            p.drawText(box, int(Qt.AlignCenter), words)
        for n in nodes:
            c, r = centre(n), radius(n)
            p.setPen(QPen(_colour(n["colour"]), 2))
            p.setBrush(_colour(n["colour"], 60))
            p.drawEllipse(c, r, r)
            lines = [f"D {n['D']:g} µm²/s, {100 * n['share']:.0f} %"]
            if n.get("dwell_ms"):
                lines.append(f"stays about {n['dwell_ms']:.0f} ms")
            # the words under a circle, or over it where it sits in the top half (the arrows run below)
            above = len(nodes) > 2 and n["y"] < 0.4
            top = c.y() - r - 6 - 18 - 16 * len(lines) if above else c.y() + r + 4
            p.setPen(QColor(t.foreground))
            p.setFont(ty.ui("sm", "semibold"))
            p.drawText(QRectF(c.x() - 90, top, 180, 18), int(Qt.AlignCenter), n["name"])
            p.setFont(ty.ui("xs", tabular=True))
            p.setPen(QColor(t.muted_foreground))
            for i, line in enumerate(lines):
                p.drawText(QRectF(c.x() - 100, top + 18 + 16 * i, 200, 16), int(Qt.AlignCenter), line)
        if data.get("note"):
            self._words(p, QRectF(rect.left(), rect.bottom() - note_h, rect.width(), note_h), data["note"],
                        t.muted_foreground, size="xs")


# --------------------------------------------------------------------------- #
# Kinetics: the response and the movies
# --------------------------------------------------------------------------- #


class CurvePicture(_Picture):
    """The response h(t) over the time since the addition, each movie a band on the clock, the delay and the
    time h reaches one half marked; under it each population's share before and after."""

    def draw(self, p: QPainter, rect: QRectF) -> None:
        t, data = self.t, self.payload
        if not data.get("on"):
            self._words(p, rect, "Kinetics is off for this condition: turn it on to draw its response.",
                        t.muted_foreground)
            return
        ts, hs = data["t_min"], data["h"]
        if len(ts) < 2:
            return
        left, right = 44.0, 14.0
        plot = QRectF(rect.left() + left, rect.top() + 14, rect.width() - left - right, rect.height() * 0.5)
        lo, hi = ts[0], ts[-1]

        def x_px(x):
            return plot.left() + (x - lo) / (hi - lo) * plot.width()

        def y_px(y):
            return plot.bottom() - y * plot.height()

        for a, b in data["movies"]:
            p.setPen(Qt.NoPen)
            p.setBrush(_colour(t.ring, 70))
            p.drawRect(QRectF(QPointF(x_px(a), plot.top()), QPointF(max(x_px(b), x_px(a) + 2), plot.bottom())))
        p.setPen(QPen(QColor(t.border), 1))
        p.drawLine(QPointF(plot.left(), plot.bottom()), QPointF(plot.right(), plot.bottom()))
        p.drawLine(QPointF(plot.left(), plot.top()), QPointF(plot.left(), plot.bottom()))
        p.setFont(ty.ui("xs"))
        p.setPen(QColor(t.muted_foreground))
        for y, words in ((0.0, "0"), (0.5, "0.5"), (1.0, "1")):
            p.drawText(QRectF(rect.left(), y_px(y) - 8, left - 6, 16), int(Qt.AlignRight | Qt.AlignVCenter), words)
        for x in data.get("x_ticks") or []:
            p.drawText(QRectF(x_px(x) - 20, plot.bottom() + 2, 40, 16), int(Qt.AlignCenter), f"{x:g}")
        p.drawText(QRectF(plot.left(), plot.bottom() + 16, plot.width(), 16), int(Qt.AlignCenter),
                   "minutes since the addition (grey: each movie)")
        zero = x_px(0.0)
        p.setPen(QPen(QColor(t.foreground), 1, Qt.DashLine))
        p.drawLine(QPointF(zero, plot.top()), QPointF(zero, plot.bottom()))
        path = QPainterPath(QPointF(x_px(ts[0]), y_px(hs[0])))
        for x, y in zip(ts[1:], hs[1:]):
            path.lineTo(QPointF(x_px(x), y_px(y)))
        p.setPen(QPen(QColor(t.foreground), 2))
        p.setBrush(Qt.NoBrush)
        p.drawPath(path)
        half = QPointF(x_px(data["half_min"]), y_px(0.5))
        p.setBrush(QColor(t.foreground))
        p.drawEllipse(half, 3.5, 3.5)
        p.setFont(ty.ui("xs"))
        p.drawText(QRectF(half.x() + 6, half.y() - 18, 160, 16), int(Qt.AlignLeft | Qt.AlignVCenter),
                   f"half at {data['half_min']:.2g} min")
        p.drawText(QRectF(zero + 4, plot.top(), 160, 16), int(Qt.AlignLeft | Qt.AlignVCenter),
                   f"addition, delay {60 * data['delay_min']:g} s")
        # before and after
        bars = QRectF(plot.left(), plot.bottom() + 48, plot.width(), rect.bottom() - plot.bottom() - 60)
        names, colours = data["names"], data["colours"]
        n = max(len(names), 1)
        group_w = bars.width() / n
        p.setFont(ty.ui("xs"))
        for i, name in enumerate(names):
            gx = bars.left() + i * group_w
            for j, (key, words) in enumerate((("before", "before"), ("after", "after"))):
                share = data[key][i] if i < len(data[key]) else 0.0
                bw = min(28.0, group_w * 0.28)
                x = gx + group_w / 2 - bw - 3 + j * (bw + 6)
                h = share * (bars.height() - 34)
                p.setPen(Qt.NoPen)
                p.setBrush(_colour(colours[i], 230 if j else 110))
                p.drawRect(QRectF(x, bars.bottom() - 34 - h, bw, h))
                p.setPen(QColor(t.muted_foreground))
                p.drawText(QRectF(x - 10, bars.bottom() - 34 - h - 16, bw + 20, 14), int(Qt.AlignCenter),
                           f"{100 * share:.0f} %")
                p.drawText(QRectF(x - 10, bars.bottom() - 32, bw + 20, 14), int(Qt.AlignCenter), words)
            p.setPen(QColor(t.foreground))
            d0, d1 = data["D_before"][i], data["D_after"][i]
            d_words = f"D {d0:g}" if d0 == d1 else f"D {d0:g} to {d1:g}"
            p.drawText(QRectF(gx, bars.bottom() - 16, group_w, 16), int(Qt.AlignCenter),
                       p.fontMetrics().elidedText(f"{name}, {d_words}", Qt.ElideRight, int(group_w - 4)))


# --------------------------------------------------------------------------- #
# Cells and field
# --------------------------------------------------------------------------- #


class FieldPicture(_Picture):
    """The field at its pixel's scale (a line per pixel where they are 5 screen pixels apart or more, else every
    ten), the cells as a build places them, a 5 µm scale bar."""

    def _draws_with_error(self) -> bool:
        return True

    def draw(self, p: QPainter, rect: QRectF) -> None:
        t, data = self.t, self.payload
        area = rect.adjusted(16, 16, -16, -46)
        W, H = data["width_um"], data["height_um"]
        if not (W > 0 and H > 0):
            return
        scale = min(area.width() / W, area.height() / H)
        box = QRectF(area.center().x() - W * scale / 2, area.center().y() - H * scale / 2, W * scale, H * scale)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(t.subtle))
        p.drawRect(box)
        px = data["pixel_um"] * scale
        every = 1 if px >= 5 else 10
        p.setPen(QPen(QColor(t.border), 1))
        for i in range(0, data["width_px"] + 1, every):
            x = box.left() + i * px
            p.drawLine(QPointF(x, box.top()), QPointF(x, box.bottom()))
        for j in range(0, data["height_px"] + 1, every):
            y = box.top() + j * px
            p.drawLine(QPointF(box.left(), y), QPointF(box.right(), y))
        for cell in data["cells"]:
            poly = QPolygonF([QPointF(box.left() + x * scale, box.top() + y * scale) for x, y in cell])
            p.setPen(QPen(QColor(t.foreground), 1.2))
            p.setBrush(_colour(t.ring, 90))
            p.drawPolygon(poly)
        p.setPen(QPen(QColor(t.foreground), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(box)
        bar = 5.0 * scale
        y = box.bottom() + 14
        p.setPen(QPen(QColor(t.foreground), 3))
        p.drawLine(QPointF(box.left(), y), QPointF(box.left() + bar, y))
        p.setFont(ty.ui("xs"))
        p.setPen(QColor(t.foreground))
        p.drawText(QRectF(box.left(), y + 4, bar + 80, 16), int(Qt.AlignLeft | Qt.AlignVCenter), "5 µm")
        p.setPen(QColor(t.muted_foreground))
        p.drawText(QRectF(box.left(), y + 4, box.width(), 16), int(Qt.AlignRight | Qt.AlignVCenter),
                   f"{data['width_px']} x {data['height_px']} px, {W:.1f} x {H:.1f} µm")
        if data.get("error"):
            self._words(p, box, data["error"], CUT_INK)


# --------------------------------------------------------------------------- #
# Camera and movie: one rendered frame
# --------------------------------------------------------------------------- #


class FramePicture(_Picture):
    """A rendered frame, each camera pixel a square (no smoothing), grey from the 1st to the 99.8th percentile."""

    def __init__(self, t, parent=None):
        super().__init__(t, min_height=260, parent=parent)
        self._bytes = b""
        self._image: QImage | None = None

    def set_payload(self, payload: dict | None) -> None:
        self._image = None
        if payload and payload.get("grey"):
            self._bytes = bytes(payload["grey"])
            self._image = QImage(self._bytes, int(payload["width"]), int(payload["height"]), int(payload["width"]),
                                 QImage.Format_Grayscale8)
        super().set_payload(payload)

    def draw(self, p: QPainter, rect: QRectF) -> None:
        if self._image is None:
            return
        area = rect.adjusted(10, 10, -10, -10)
        w, h = self._image.width(), self._image.height()
        scale = max(1, int(min(area.width() / w, area.height() / h)))
        target = QRectF(area.center().x() - w * scale / 2, area.center().y() - h * scale / 2, w * scale, h * scale)
        p.setRenderHint(QPainter.SmoothPixmapTransform, False)
        p.drawImage(target, self._image)
