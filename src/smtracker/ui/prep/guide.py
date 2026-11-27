"""The preparation window's setting guides: a column of settings beside a live picture.

The same shape as the analysis's Run dialog (`ui.workspace.spatial_map_chart`): the
column keeps every value on screen (`widgets.settings.SettingNav`), the picture is a
MECHANISM diagram in the guides' own inks, and the longer words sit behind an info line
as a tooltip (`widgets.settings.InfoLink`). Asked for: "helpers and animation like in
analysis as well, especially for tracking and mb some image prep".

THIN QT. Every payload is `prep_guide.picture`'s -- plain lists, every count already
computed. The painter only maps numbers to pixels and compares a moving value (the line,
the disc, the gap) against what the payload drew, so the picture at rest is exactly the
payload's and the animation is only the way between two of them.

ANIMATED THE WAY THE ANALYSIS ANIMATES: one `QVariantAnimation` slot per moving value,
held as an attribute (a collected wrapper mid-flight is a real risk), a value sitting
still reproduces the resting picture, and an intro plays once per setting shown -- the
walk walks, the frames tick past, the threshold line comes down to where it belongs.
Clicking the picture plays it again.
"""
from __future__ import annotations

import math
import warnings

from qtpy.QtCore import QEasingCurve, QPointF, QRectF, Qt, QVariantAnimation, Signal
from qtpy.QtGui import QColor, QFontMetricsF, QPainter, QPainterPath, QPen
from qtpy.QtWidgets import QHBoxLayout, QSizePolicy, QVBoxLayout, QWidget

from .. import theme
from ..theme import typography as ty
from ..widgets.primitives import Card, Checkbox, label
from ..widgets.settings import (
    ACCENT_INK,
    CUT_INK,
    LONG_INK,
    SHORT_INK,
    WARN_INK,
    InfoLink,
    SettingNav,
)

__all__ = ["GuideArt", "GuidePanel"]

#: Each track of a picture in turn, so neighbouring tracks never share an ink.
_TRACK_INKS = (SHORT_INK, LONG_INK, ACCENT_INK)


def _ease(curve=QEasingCurve.OutCubic) -> QEasingCurve:
    return QEasingCurve(curve)


class GuideArt(QWidget):
    """The picture for whichever setting is open, painted from a `prep_guide` payload."""

    HEIGHT = 210
    PAD = 14

    def __init__(self, t, height: int | None = None, parent=None):
        super().__init__(parent)
        self.t = t
        self.HEIGHT = int(height or self.HEIGHT)
        self.setStyleSheet("background:transparent;")
        self.setMinimumHeight(self.HEIGHT)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Click to play it again")
        self._payload: dict = {}
        #: The moving value of the open picture (the threshold's k, a radius, the gap,
        #: the tolerance, the chosen start) and the intro's progress, 0 to 1 (None at
        #: rest). Both are what `paintEvent` reads.
        self._value = 0.0
        self._intro: float | None = None
        self._value_anim: QVariantAnimation | None = None
        self._intro_anim: QVariantAnimation | None = None
        self._played: set = set()

    # -- what the panel tells the picture ---------------------------------------- #
    @staticmethod
    def _moving(payload: dict) -> float:
        key = payload.get("key")
        return float({"threshold": payload.get("k", 0.0),
                      "radius_px": payload.get("radius", 0.0),
                      "linking_max_distance_px": payload.get("radius", 0.0),
                      "max_frame_gap": payload.get("gap", 0),
                      "flat_tolerance": payload.get("tolerance", 0.0),
                      "start_rule": payload.get("chosen") or 0,
                      "end": payload.get("end") or 0}.get(key, 0.0))

    def show_payload(self, payload: dict) -> None:
        """Move to ``payload``: the same setting animates its value across, a different
        setting starts at rest and plays its intro the first time it is shown."""
        same = payload.get("key") == self._payload.get("key")
        old, new = self._value, self._moving(payload)
        self._payload = payload
        if same and abs(old - new) > 1e-9 and self.isVisible():
            self._animate_value(old, new)
        else:
            self._value = new
            self.update()
        if not same and payload.get("key") not in self._played and self.isVisible():
            self.play_intro()

    def _animate_value(self, old: float, new: float) -> None:
        if self._value_anim is not None:
            self._value_anim.stop()
        anim = QVariantAnimation(self)
        anim.setStartValue(float(old))
        anim.setEndValue(float(new))
        anim.setDuration(380)
        anim.setEasingCurve(_ease())
        anim.valueChanged.connect(self._set_value)
        self._value_anim = anim
        anim.start()

    def _set_value(self, value) -> None:
        self._value = float(value)
        self.update()

    def play_intro(self) -> None:
        key = self._payload.get("key")
        if key is None:
            return
        self._played.add(key)
        if self._intro_anim is not None:
            self._intro_anim.stop()
        anim = QVariantAnimation(self)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setDuration({"linking_max_distance_px": 3200, "max_frame_gap": 1600}.get(key, 900))
        anim.setEasingCurve(QEasingCurve(QEasingCurve.Linear
                                         if key in ("linking_max_distance_px", "max_frame_gap")
                                         else QEasingCurve.OutCubic))
        anim.valueChanged.connect(self._set_intro)
        anim.finished.connect(self._intro_done)
        self._intro_anim = anim
        anim.start()

    def _set_intro(self, value) -> None:
        self._intro = float(value)
        self.update()

    def _intro_done(self) -> None:
        self._intro = None
        self.update()

    def stop(self) -> None:
        for anim in (self._value_anim, self._intro_anim):
            if anim is not None:
                anim.stop()
        self._intro = None

    def mouseReleaseEvent(self, e) -> None:
        if e.button() == Qt.LeftButton:
            self.play_intro()
        super().mouseReleaseEvent(e)

    # -- painting ------------------------------------------------------------------ #
    def _inner(self) -> QRectF:
        return QRectF(self.PAD, self.PAD, self.width() - 2 * self.PAD,
                      self.HEIGHT - 2 * self.PAD)

    def _pen(self, colour, width=1.25, style=Qt.SolidLine) -> QPen:
        pen = QPen(QColor(colour))
        pen.setWidthF(width)
        pen.setStyle(style)
        pen.setCapStyle(Qt.RoundCap)
        return pen

    def _label_room(self, *texts, bold=False) -> float:
        """The width the right-hand labels need, so none is cut at the widget's edge: 92 px held
        "your movies' noise" only in part (Detect & link read "your movies' nois", 26 Nov 2026)."""
        font = ty.ui("xs", "semibold" if bold else "normal", tabular=True)
        return max(QFontMetricsF(font).horizontalAdvance(s) for s in texts) + 8 + 4

    def _text(self, p: QPainter, x: float, y: float, text: str, *, colour=None,
              align=Qt.AlignLeft, bold=False) -> None:
        font = ty.ui("xs", "semibold" if bold else "normal", tabular=True)
        p.setFont(font)
        p.setPen(QColor(colour or self.t.muted_foreground))
        width = QFontMetricsF(font).horizontalAdvance(text)
        if align == Qt.AlignRight:
            x -= width
        elif align == Qt.AlignHCenter:
            x -= width / 2.0
        p.drawText(QPointF(x, y), text)

    def paintEvent(self, _e) -> None:
        key = self._payload.get("key")
        if key is None:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        getattr(self, "_draw_" + key)(p, self._payload)
        p.end()

    # threshold -------------------------------------------------------------------- #
    def _draw_threshold(self, p: QPainter, d: dict) -> None:
        t = self.t
        box = self._inner()
        strip = 38 if d["movies"] else 0
        k = self._value
        room = max(92.0, self._label_room(f"{k:.1f} x noise", bold=True),
                   self._label_room("noise level", *(["your movies' noise"] if d["movies"] else [])))
        plot = QRectF(box.left(), box.top(), box.width() - room, box.height() - strip)
        lo, hi = d["range"]

        def X(v):
            return plot.left() + v * plot.width()

        def Y(v):
            return plot.bottom() - (v - lo) / (hi - lo) * plot.height()

        if self._intro is not None:          # the line comes down from the top
            k = hi - (hi - k) * self._intro
        path = QPainterPath()
        for i, (x, y) in enumerate(zip(d["x"], d["trace"])):
            (path.lineTo if i else path.moveTo)(QPointF(X(x), Y(y)))
        p.setPen(self._pen(t.muted_foreground, 1.0))
        p.setBrush(Qt.NoBrush)
        p.drawPath(path)
        p.setPen(self._pen(t.border, 1.0, Qt.DashLine))
        p.drawLine(QPointF(plot.left(), Y(0)), QPointF(plot.right(), Y(0)))
        for peak in d["noise_peaks"]:
            if peak["y"] >= k:
                c = QPointF(X(peak["x"]), Y(peak["y"]))
                p.setPen(self._pen(CUT_INK, 1.6))
                p.drawLine(c + QPointF(-4, -4), c + QPointF(4, 4))
                p.drawLine(c + QPointF(-4, 4), c + QPointF(4, -4))
        for spot in d["spots"]:
            c = QPointF(X(spot["x"]), Y(spot["peak"]))
            found = spot["peak"] >= k
            p.setPen(self._pen(SHORT_INK if found else t.muted_foreground, 1.6))
            p.setBrush(QColor(SHORT_INK) if found else QColor(t.card))
            p.drawEllipse(c, 5.0, 5.0)
        p.setPen(self._pen(CUT_INK, 1.75))
        p.drawLine(QPointF(plot.left(), Y(k)), QPointF(plot.right(), Y(k)))
        self._text(p, plot.right() + 8, Y(k) + 4, f"{k:.1f} x noise", colour=CUT_INK,
                   bold=True)
        self._text(p, plot.right() + 8, Y(0) + 4, "noise level")
        found = sum(s["peak"] >= k for s in d["spots"])
        self._text(p, plot.left(), plot.top() + 10,
                   f"{found} of {len(d['spots'])} spots found  (filled)", colour=t.foreground)
        if d["movies"]:
            self._draw_noise_strip(p, QRectF(box.left(), box.bottom() - strip + 10,
                                             box.width() - room, strip - 10), d["movies"])

    def _draw_noise_strip(self, p: QPainter, area: QRectF, movies: list) -> None:
        """Each movie's measured noise against the median, on a log axis: the band is
        the 1.5x a movie may sit from it before it reads as another session."""
        t = self.t
        lo, hi = math.log(0.4), math.log(2.5)

        def X(r):
            return area.left() + (math.log(max(min(r, 2.5), 0.4)) - lo) / (hi - lo) * area.width()

        y = area.center().y()
        band = QRectF(QPointF(X(1 / 1.5), area.top()), QPointF(X(1.5), area.bottom()))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(t.muted))
        p.drawRoundedRect(band, 3, 3)
        p.setPen(self._pen(t.border, 1.0))
        p.drawLine(QPointF(X(1.0), area.top()), QPointF(X(1.0), area.bottom()))
        for m in movies:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(WARN_INK if m["odd"] else t.foreground))
            p.drawEllipse(QPointF(X(m["ratio"]), y), 3.2, 3.2)
        self._text(p, area.right() + 8, y + 4, "your movies' noise")

    # spot radius ------------------------------------------------------------------- #
    def _draw_radius_px(self, p: QPainter, d: dict) -> None:
        t = self.t
        box = self._inner()
        n = 2 * d["half"] + 1
        cell = min(box.height() / n, (box.width() * 0.55) / n)
        side = cell * n
        origin = QPointF(box.left() + (box.width() * 0.55 - side) / 2, box.top() + (box.height() - side) / 2)
        sigma = d["psf_px"]
        centre = QPointF(d["half"] + 0.5 + 0.18, d["half"] + 0.5 - 0.12)   # off the grid
        ink, paper = QColor(t.foreground), QColor(t.card)
        for i in range(n):
            for j in range(n):
                dx, dy = i + 0.5 - centre.x(), j + 0.5 - centre.y()
                w = math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma))
                colour = QColor(round(paper.red() + (ink.red() - paper.red()) * w * 0.85),
                                round(paper.green() + (ink.green() - paper.green()) * w * 0.85),
                                round(paper.blue() + (ink.blue() - paper.blue()) * w * 0.85))
                p.setPen(self._pen(t.border, 0.6))
                p.setBrush(colour)
                p.drawRect(QRectF(origin.x() + i * cell, origin.y() + j * cell, cell, cell))
        c = QPointF(origin.x() + centre.x() * cell, origin.y() + centre.y() * cell)
        radius = self._value * (self._intro if self._intro is not None else 1.0)
        p.setBrush(Qt.NoBrush)
        p.setPen(self._pen(t.muted_foreground, 1.0, Qt.DashLine))
        p.drawEllipse(c, sigma * cell, sigma * cell)
        p.setPen(self._pen(CUT_INK, 2.0))
        p.drawEllipse(c, radius * cell, radius * cell)
        x0 = origin.x() + side + 24
        self._text(p, x0, box.top() + 16, f"radius {self._value:.2f} px", colour=CUT_INK,
                   bold=True)
        self._text(p, x0, box.top() + 34, f"spot sigma {sigma:.2f} px (dashed)")
        self._text(p, x0, box.top() + 52, "one square = one camera pixel")
        if not d["derived"]:
            self._text(p, x0, box.top() + 76, "invented spot: the optics are not set",
                       colour=WARN_INK)

    # linking radius ---------------------------------------------------------------- #
    def _draw_linking_max_distance_px(self, p: QPainter, d: dict) -> None:
        t = self.t
        box = self._inner()
        radius = self._value
        x0, x1, y0, y1 = d["bounds"]
        room = max(radius, d["radius"])
        span_x, span_y = (x1 - x0) + 2 * room, (y1 - y0) + 2 * room
        area = QRectF(box.left(), box.top(), box.width() * 0.6, box.height())
        scale = min(area.width() / span_x, area.height() / span_y)
        mid = QPointF((x0 + x1) / 2, (y0 + y1) / 2)

        def P(xy):
            return QPointF(area.center().x() + (xy[0] - mid.x()) * scale,
                           area.center().y() + (xy[1] - mid.y()) * scale)

        pts = d["points"]
        segments = d["segments"]

        def ink(track):
            # A TRACK TOO SHORT TO USE IS GREY, as its line in the legend is.
            if track >= len(segments) or segments[track] < 5:
                return t.muted_foreground
            return _TRACK_INKS[track % len(_TRACK_INKS)]

        upto = len(pts) - 1 if self._intro is None else self._intro * (len(pts) - 1)
        last = int(math.floor(upto))
        track = 0
        for i in range(min(last, len(pts) - 1)):
            if d["steps"][i] > radius:
                track += 1
                p.setPen(self._pen(CUT_INK, 1.4, Qt.DashLine))
            else:
                p.setPen(self._pen(ink(track), 1.8))
            p.drawLine(P(pts[i]), P(pts[i + 1]))
        track = 0
        for i in range(last + 1):
            if i and d["steps"][i - 1] > radius:
                track += 1
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(ink(track)))
            p.drawEllipse(P(pts[i]), 2.6, 2.6)
        here = P(pts[min(last, len(pts) - 1)])
        disc = QColor(t.foreground)
        disc.setAlphaF(0.06)
        p.setBrush(disc)
        p.setPen(self._pen(t.muted_foreground, 1.0, Qt.DashLine))
        p.drawEllipse(here, radius * scale, radius * scale)
        x0 = box.left() + box.width() * 0.66
        self._text(p, x0, box.top() + 16, f"linking radius {radius:.1f} px",
                   colour=t.foreground, bold=True)
        self._text(p, x0, box.top() + 34, "dashed red: a jump longer than the radius,")
        self._text(p, x0, box.top() + 50, "so a new track starts")
        y = box.top() + 76
        for i, n in enumerate(d["segments"]):
            short = n < 5
            self._text(p, x0, y, f"track {i + 1}: {n} localisations"
                       + ("  (too short, dropped)" if short else ""),
                       colour=t.muted_foreground if short else _TRACK_INKS[i % len(_TRACK_INKS)])
            y += 16
            if y > box.bottom():
                break

    # frame gap --------------------------------------------------------------------- #
    def _draw_max_frame_gap(self, p: QPainter, d: dict) -> None:
        t = self.t
        box = self._inner()
        present = d["present"]
        n = len(present)
        step = box.width() / n
        y = box.top() + 56
        shown = n if self._intro is None else int(self._intro * n)
        gap = self._value
        # Bridges over every absence, fading in as the gap reaches its length.
        run_start = None
        for f in range(n + 1):
            here = f < n and present[f]
            if f < n and not here and run_start is None:
                run_start = f
            elif (here or f == n) and run_start is not None:
                length = f - run_start
                if 0 < run_start and f < n and run_start < shown:
                    a = QPointF(box.left() + (run_start - 0.5) * step, y - 7)
                    b = QPointF(box.left() + (f + 0.5) * step, y - 7)
                    reach = max(0.0, min(1.0, gap - length + 1))
                    colour = QColor(SHORT_INK if reach > 0.5 else CUT_INK)
                    colour.setAlphaF(0.25 + 0.75 * (reach if reach > 0.5 else 1 - reach))
                    pen = QPen(colour)
                    pen.setWidthF(1.6)
                    pen.setStyle(Qt.SolidLine if reach > 0.5 else Qt.DashLine)
                    p.setPen(pen)
                    p.setBrush(Qt.NoBrush)
                    path = QPainterPath(a)
                    path.quadTo(QPointF((a.x() + b.x()) / 2, y - 24 - 4 * length), b)
                    p.drawPath(path)
                    self._text(p, (a.x() + b.x()) / 2, y - 28 - 4 * length,
                               f"{length}", align=Qt.AlignHCenter,
                               colour=SHORT_INK if reach > 0.5 else CUT_INK)
                run_start = None
        for f in range(min(shown, n)):
            c = QPointF(box.left() + (f + 0.5) * step, y)
            if present[f]:
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(t.foreground))
                p.drawEllipse(c, 3.4, 3.4)
            else:
                p.setPen(self._pen(t.border, 1.2))
                p.setBrush(QColor(t.card))
                p.drawEllipse(c, 3.4, 3.4)
        if self._intro is None:
            for i, ((a, b), locs) in enumerate(zip(d["tracks"], d["locs"])):
                short = locs < 5
                ink = t.muted_foreground if short else _TRACK_INKS[i % len(_TRACK_INKS)]
                x1 = box.left() + (a + 0.2) * step
                x2 = box.left() + (b + 0.8) * step
                p.setPen(self._pen(ink, 4.0))
                p.drawLine(QPointF(x1, y + 18), QPointF(x2, y + 18))
                self._text(p, (x1 + x2) / 2, y + 36, f"{locs}" + (" (dropped)" if short else ""),
                           align=Qt.AlignHCenter, colour=ink)
        self._text(p, box.left(), box.bottom() - 4,
                   f"gap {int(round(gap))}: an absence of up to {int(round(gap))} frame(s) is "
                   f"bridged (solid); a longer one ends the track (dashed red)")

    # flat tolerance ---------------------------------------------------------------- #
    def _curve_axes(self, d: dict, plot: QRectF, top_value: float, bottom_value: float):
        f0, f1 = d["frames"][0], d["frames"][-1]

        def X(f):
            return plot.left() + (f - f0) / max(1.0, f1 - f0) * plot.width()

        def Y(v):
            v = min(max(v, bottom_value), top_value)
            return plot.bottom() - (v - bottom_value) / (top_value - bottom_value) * plot.height()

        return X, Y

    def _draw_curve(self, p: QPainter, d: dict, X, Y) -> None:
        path = QPainterPath()
        for i, (f, v) in enumerate(zip(d["frames"], d["means"])):
            (path.lineTo if i else path.moveTo)(QPointF(X(f), Y(v)))
        p.setPen(self._pen(self.t.muted_foreground, 1.1))
        p.setBrush(Qt.NoBrush)
        p.drawPath(path)

    def _draw_flat_tolerance(self, p: QPainter, d: dict) -> None:
        t = self.t
        box = self._inner()
        plot = QRectF(box.left(), box.top() + 14, box.width() - 150, box.height() - 30)
        level = d["level"]
        dark = d["dark"] if d["dark"] is not None else min(d["means"])
        tol = self._value * (self._intro if self._intro is not None else 1.0)
        band = tol * level
        top = level + max(4 * d["band"], 0.9 * (level - dark), 1.0)
        bottom = dark - 0.08 * (top - dark)
        X, Y = self._curve_axes(d, plot, top, bottom)
        shade = QColor(SHORT_INK)
        shade.setAlphaF(0.16)
        start = d["start"] if d["start"] else d["frames"][-1]
        p.setPen(Qt.NoPen)
        p.setBrush(shade)
        p.drawRect(QRectF(QPointF(X(start), Y(level + band / 2)),
                          QPointF(plot.right(), Y(level - band / 2))))
        self._draw_curve(p, d, X, Y)
        if d["dark"] is not None:
            p.setPen(self._pen(t.foreground, 1.0, Qt.DashLine))
            p.drawLine(QPointF(plot.left(), Y(dark)), QPointF(plot.right(), Y(dark)))
            self._text(p, plot.right() + 8, Y(dark) + 4, f"camera dark {dark:.0f}")
        self._text(p, plot.right() + 8, Y(level) + 4,
                   f"band {100 * tol:.1f} % of {level:.0f}", colour=SHORT_INK, bold=True)
        if d["start"]:
            p.setPen(self._pen(CUT_INK, 1.75))
            p.drawLine(QPointF(X(start), plot.top()), QPointF(X(start), plot.bottom()))
            self._text(p, X(start) + 4, plot.top() + 2, f"flat from {d['start']}",
                       colour=CUT_INK, bold=True)
        if d["share"] is not None:
            self._text(p, plot.right() + 8, Y(level) + 22,
                       f"= {100 * d['share']:.0f} % of the signal",
                       colour=WARN_INK if d["share"] > 0.25 else t.muted_foreground)
        self._text(p, plot.left(), box.bottom(), "frame mean, clipped to the tail",
                   colour=t.muted_foreground)

    # start rule -------------------------------------------------------------------- #
    def _draw_start_rule(self, p: QPainter, d: dict) -> None:
        t = self.t
        box = self._inner()
        plot = QRectF(box.left(), box.top() + 6, box.width() - 170, box.height() - 22)
        means = d["means"]
        tail = sorted(means[len(means) // 2:])
        level, low = tail[len(tail) // 2], min(means)
        # CLIPPED TO THE TAIL: the laser-on peak is many times the settled level, and
        # the floors all sit where the curve flattens, which a full scale would squash.
        top = min(max(means), level + 1.5 * (level - low) + 1.0)
        X, Y = self._curve_axes(d, plot, top, low - 0.05 * (top - low))
        chosen = self._value
        if chosen:
            keep = QColor(SHORT_INK)
            keep.setAlphaF(0.10)
            p.setPen(Qt.NoPen)
            p.setBrush(keep)
            p.drawRect(QRectF(QPointF(X(chosen), plot.top()),
                              QPointF(plot.right(), plot.bottom())))
        self._draw_curve(p, d, X, Y)
        names = {"strictest": "strictest", "density": "density", "amplitude": "amplitude",
                 "field_mean": "field mean"}
        # THE FLOORS ARE LISTED BESIDE THE PLOT, not written on it: three of them often
        # sit within a hundred frames of each other, where their names would overprint.
        y = box.top() + 14
        for key, frame in d["floors"].items():
            mine = key == d["rule"]
            if frame:
                p.setPen(self._pen(t.muted_foreground if mine else t.border, 1.0,
                                   Qt.DashLine))
                p.drawLine(QPointF(X(frame), plot.top()), QPointF(X(frame), plot.bottom()))
            self._text(p, plot.right() + 14, y,
                       f"{names[key]}  {frame if frame else 'none'}",
                       colour=CUT_INK if mine else t.muted_foreground, bold=mine)
            y += 17
        if chosen:
            p.setPen(self._pen(CUT_INK, 2.0))
            p.drawLine(QPointF(X(chosen), plot.top()), QPointF(X(chosen), plot.bottom()))
        self._text(p, plot.left(), box.bottom(), "kept: shaded, from the rule's start",
                   colour=t.muted_foreground)


    def _draw_end(self, p: QPainter, d: dict) -> None:
        """Bars: the background's share of the in-cell rate, block by block (red where it is
        highest). Above them, the in-cell data still kept if every movie ended there. Without
        a measurement: the movie's spots per frame. The end marker slides when it changes."""
        t = self.t
        box = self._inner()
        plot = QRectF(box.left(), box.top() + 18, box.width() - 170, box.height() - 40)
        end = self._value
        if d["mode"] == "purity":
            rows = d["rows"]
            f0, f1 = rows[0]["first"], rows[-1]["last"]

            def X(f):
                return plot.left() + (f - f0) / max(1.0, f1 - f0) * plot.width()

            top = max(0.01, d["worst"] * 1.15)
            reveal = 1.0 if self._intro is None else self._intro
            for r in rows:
                share = (r["share"] or 0.0) * reveal
                h = share / top * plot.height()
                colour = QColor(CUT_INK if (r["share"] or 0) >= 0.5 * d["worst"] else SHORT_INK)
                colour.setAlphaF(0.55)
                p.setPen(Qt.NoPen)
                p.setBrush(colour)
                p.drawRect(QRectF(QPointF(X(r["first"]) + 2, plot.bottom() - h),
                                  QPointF(X(r["last"]) - 2, plot.bottom())))
                self._text(p, (X(r["first"]) + X(r["last"])) / 2, plot.bottom() - h - 4,
                           f"{100 * (r['share'] or 0):.1f} %", align=Qt.AlignHCenter,
                           colour=t.foreground)
            # what is kept, cumulatively, if every movie ended at this frame
            path = QPainterPath(QPointF(X(f0), plot.bottom()))
            acc = 0.0
            for r in rows:
                acc += r["of_all"]
                path.lineTo(QPointF(X(r["last"]), plot.bottom() - acc * plot.height()))
            p.setPen(self._pen(t.muted_foreground, 1.2, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawPath(path)
            self._text(p, plot.right() + 10, plot.top() + 10, "bars: background as a share")
            self._text(p, plot.right() + 10, plot.top() + 26, "of the in-cell rate")
            self._text(p, plot.right() + 10, plot.top() + 48, "dashed: in-cell data kept")
            self._text(p, plot.right() + 10, plot.top() + 64, "if every movie ended there")
            x_of = X
        else:
            frames, spots = d["frames"], d["spots"]

            def x_of(f):
                return plot.left() + (f - frames[0]) / max(1.0, frames[-1] - frames[0]) * plot.width()

            top = max(spots) * 1.05 or 1.0
            path = QPainterPath()
            for i, (f, v) in enumerate(zip(frames, spots)):
                pt = QPointF(x_of(f), plot.bottom() - v / top * plot.height())
                (path.lineTo if i else path.moveTo)(pt)
            p.setPen(self._pen(t.muted_foreground, 1.2))
            p.setBrush(Qt.NoBrush)
            p.drawPath(path)
            self._text(p, plot.right() + 10, plot.top() + 10, "spots per frame")
        if end:
            x = x_of(end)
            if plot.left() <= x <= plot.right():
                p.setPen(self._pen(CUT_INK, 2.0))
                p.drawLine(QPointF(x, plot.top() - 14), QPointF(x, plot.bottom()))
                self._text(p, x + 4, plot.top() - 4, f"ends at {int(end)}", colour=CUT_INK,
                           bold=True)
        self._text(p, plot.left(), box.bottom(), "frame of the acquisition", colour=t.muted_foreground)


class GuidePanel(Card):
    """A titled, foldable guide: the settings column, and the open setting's picture."""

    #: The setting picked in the column.
    key_changed = Signal(str)

    def __init__(self, t, keys, names: dict, *, remember_as: str, default_open: bool,
                 art_height: int | None = None, parent=None):
        super().__init__(t, shadowed=False, parent=parent)
        self.t = t
        self.keys = tuple(keys)
        self._remember_as = remember_as
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 12)
        lay.setSpacing(10)
        head = QHBoxLayout()
        head.setSpacing(8)
        head.addWidget(label("How these settings work", ty.ui("sm", "semibold"),
                             t.foreground))
        head.addStretch(1)
        remembered = theme.remembered(remember_as)
        self.toggle = Checkbox(t, checked=default_open if remembered is None else bool(remembered))
        self.toggle.toggled.connect(self.set_open)
        head.addWidget(self.toggle)
        head.addWidget(label("Show", ty.ui("sm"), t.muted_foreground))
        lay.addLayout(head)

        self.body = QWidget()
        self.body.setStyleSheet("background:transparent;")
        lay.addWidget(self.body)
        row = QHBoxLayout(self.body)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(20)
        self.nav = SettingNav(t, [names[k] for k in self.keys])
        self.nav.changed.connect(lambda i: self.key_changed.emit(self.keys[i]))
        row.addWidget(self.nav, 0, Qt.AlignTop)
        panel = QVBoxLayout()
        panel.setSpacing(8)
        self.title = label("", ty.ui("sm", "semibold"), t.foreground)
        self.desc = label("", ty.ui("sm"), t.muted_foreground)
        self.desc.setWordWrap(True)
        self.art = GuideArt(t, art_height)
        self.readout = label("", ty.ui("xs", tabular=True), t.muted_foreground)
        self.readout.setWordWrap(True)
        self.info = InfoLink(t)
        for w in (self.title, self.desc, self.art, self.readout, self.info):
            panel.addWidget(w)
        #: Buttons that belong to ONE setting (Measure, Check wider…), shown only while
        #: that setting is open: a helper sits with the explanation it acts on.
        self._actions: dict = {}
        self._action_row = QHBoxLayout()
        self._action_row.setSpacing(8)
        self._action_row.addStretch(1)
        panel.addLayout(self._action_row)
        self.result = label("", ty.ui("xs", tabular=True), t.foreground)
        self.result.setWordWrap(True)
        self.result.setTextInteractionFlags(Qt.TextSelectableByMouse)
        panel.addWidget(self.result)
        row.addLayout(panel, 1)
        self.body.setVisible(self.toggle.checked)
        self.nav.changed.connect(lambda _i: self._sync_actions())

    @property
    def is_open(self) -> bool:
        return self.toggle.checked

    @property
    def key(self) -> str:
        return self.keys[self.nav.current()]

    def set_open(self, on: bool) -> None:
        self.toggle.set_checked(bool(on), animate=False)
        self.body.setVisible(bool(on))
        if not on:
            self.art.stop()
        try:
            theme.remember(self._remember_as, bool(on))
        except OSError as exc:
            # SAID, NOT SWALLOWED -- and not fatal: a fold that cannot be remembered
            # must not stop the guide from folding (the colour scheme's own rule).
            warnings.warn(f"could not remember the guide's fold: {exc}", stacklevel=2)
        self.key_changed.emit(self.key)

    def show_key(self, key: str) -> None:
        """Open ``key`` from outside (a grid cell was clicked)."""
        if key in self.keys and key != self.key:
            self.nav.select(self.keys.index(key))
            self._sync_actions()
            self.key_changed.emit(key)

    def add_action(self, key: str, widget) -> None:
        """A button for setting ``key`` alone, placed under its explanation."""
        self._action_row.insertWidget(self._action_row.count() - 1, widget)
        self._actions.setdefault(key, []).append(widget)
        widget.setVisible(key == self.key)

    def _sync_actions(self) -> None:
        for key, widgets in self._actions.items():
            for w in widgets:
                w.setVisible(key == self.key)
        self.result.setText("")

    def set_result(self, text: str) -> None:
        """What an action found, under the picture it belongs to."""
        self.result.setText(text)

    def set_values(self, values: dict) -> None:
        for i, key in enumerate(self.keys):
            self.nav.set_value(i, values.get(key, ""))

    def set_text(self, text: dict) -> None:
        self.title.setText(text["title"])
        self.desc.setText(text["desc"])
        self.info.setText(text["info"])
        self.info.set_tip(text["tip"])

    def set_payload(self, payload: dict) -> None:
        self.readout.setText(payload.get("readout", ""))
        self.art.show_payload(payload)
