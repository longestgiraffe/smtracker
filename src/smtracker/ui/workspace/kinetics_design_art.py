"""`KineticsDesignArt`: the live drawing beside the Run dialog's Design choice.

What is drawn is decided in `smtracker.kinetics_design_art_view.scene` (headless); this paints it. The pattern is the
Spatial Map dialog's `_SettingArt`: a MECHANISM drawing in ink and greys with one accent for the addition, driven by Qt
animations, ONE animation in flight at a time (`_anim`, kept as an attribute so it is not collected mid-flight), and
STILL when reduced motion is set (`set_reduced_motion`): the finished picture, no timer. There is no application-wide
reduced-motion setting in this app yet, so the switch is the widget's own.
"""
from __future__ import annotations

from qtpy.QtCore import QEasingCurve, QPointF, QRectF, Qt, QVariantAnimation
from qtpy.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen
from qtpy.QtWidgets import QSizePolicy, QWidget

from ...kinetics_design_art_view import HEIGHT, LOOP_S, STILL, scene

__all__ = ["KineticsDesignArt"]

#: The one accent, the addition (Okabe-Ito vermillion; not a condition colour).
_ACCENT = "#D55E00"


class KineticsDesignArt(QWidget):
    """The animation of one design; ``set_design`` swaps the drawing."""

    WIDTH = 300

    def __init__(self, t, design: str = "culture", parent=None, *, reduced_motion: bool = False):
        super().__init__(parent)
        self.t = t
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("background:transparent;")
        self.setFixedSize(self.WIDTH, int(self.WIDTH * HEIGHT) + 6)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._design = design
        self._reduced = bool(reduced_motion)
        self._p = STILL
        #: The one animation this widget can have in flight.
        self._anim: QVariantAnimation | None = None
        self._sync()

    # -- state ------------------------------------------------------------------ #
    def design(self) -> str:
        return self._design

    def progress(self) -> float:
        return self._p

    def items(self) -> list:
        """What is drawn now (`kinetics_design_art_view.scene`)."""
        return scene(self._design, self._p)

    def reduced_motion(self) -> bool:
        return self._reduced

    def set_design(self, design: str) -> None:
        if design == self._design:
            return
        scene(design, 0.0)                         # refuses an unknown design by name
        self._design = design
        self._p = STILL if self._reduced else 0.0
        self._sync()
        self.update()

    def set_reduced_motion(self, on: bool) -> None:
        self._reduced = bool(on)
        self._sync()
        self.update()

    def is_animating(self) -> bool:
        return self._anim is not None and self._anim.state() == QVariantAnimation.Running

    # -- the loop --------------------------------------------------------------- #
    def _sync(self) -> None:
        """Start the loop where it should run (visible, motion allowed), stop it where it should not."""
        self._stop()
        if self._reduced:
            self._p = STILL
            return
        if not self.isVisible():
            return
        anim = QVariantAnimation(self)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setDuration(int(LOOP_S * 1000))
        anim.setLoopCount(-1)
        anim.setEasingCurve(QEasingCurve.Linear)
        anim.valueChanged.connect(self._tick)
        self._anim = anim
        anim.start()

    def _stop(self) -> None:
        if self._anim is not None:
            anim, self._anim = self._anim, None
            anim.stop()
            anim.deleteLater()

    def _tick(self, value) -> None:
        self._p = float(value)
        self.update()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._reduced and self._anim is None:
            self._sync()

    def hideEvent(self, event) -> None:
        self._stop()
        super().hideEvent(event)

    # -- painting --------------------------------------------------------------- #
    def _colour(self, style: str, alpha: float) -> QColor:
        t = self.t
        base = {"ink": t.foreground, "grey": t.muted_foreground, "faint": t.border, "accent": _ACCENT,
                "hatch": t.muted_foreground}[style]
        c = QColor(base)
        c.setAlphaF(max(0.0, min(1.0, alpha)))
        return c

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        w = float(self.width() - 2)
        ox, oy = 1.0, 3.0

        def X(v):
            return ox + v * w

        def Y(v):
            return oy + v * w

        for it in self.items():
            kind, style, alpha = it["kind"], it["style"], it.get("alpha", 1.0)
            colour = self._colour(style, alpha)
            if kind == "text":
                font = QFont(self.font())
                font.setPixelSize(max(7, int(round(it["size"] * w))))
                p.setFont(font)
                p.setPen(colour)
                width = 400.0
                x = X(it["x"])
                align = {"left": Qt.AlignLeft, "center": Qt.AlignHCenter, "right": Qt.AlignRight}[it["align"]]
                left = x if it["align"] == "left" else x - width / 2 if it["align"] == "center" else x - width
                p.drawText(QRectF(left, Y(it["y"]), width, it["size"] * w * 1.6), int(align | Qt.AlignTop),
                           it["text"])
                continue
            pen = QPen(colour, it.get("width", 1.0))
            fill = it.get("fill", False)
            if style == "hatch":
                brush = QBrush(colour, Qt.BDiagPattern)
            else:
                brush = QBrush(colour) if fill else Qt.NoBrush
            p.setPen(pen if (kind == "line" or not fill or style == "hatch") else Qt.NoPen)
            p.setBrush(brush)
            if kind == "rect":
                p.drawRect(QRectF(X(it["x"]), Y(it["y"]), it["w"] * w, it["h"] * w))
            elif kind == "ellipse":
                p.drawEllipse(QPointF(X(it["x"]), Y(it["y"])), it["rx"] * w, it["ry"] * w)
            elif kind == "line":
                p.drawLine(QPointF(X(it["x1"]), Y(it["y1"])), QPointF(X(it["x2"]), Y(it["y2"])))
            elif kind == "poly":
                path = QPainterPath()
                pts = it["points"]
                path.moveTo(X(pts[0][0]), Y(pts[0][1]))
                for x, y in pts[1:]:
                    path.lineTo(X(x), Y(y))
                path.closeSubpath()
                p.drawPath(path)
        p.end()
