"""A row of colours you pick from by pointing at them.

Replaces a vertical list of ``name + chip + "swap with X"`` rows. That list had
three problems at once: it was tall, it made you read seven colour NAMES to
choose a colour, and the swap note repeated on every row while applying to at
most two of them.

**A colour does not need a label.** "Vermillion" is a worse description of
``#D55E00`` than the swatch itself, and reading it is slower than seeing it. So
the row is swatches only, and the one line of text underneath names whichever
you are pointing at -- which is also where the swap warning belongs, because it
is a property of the hovered colour rather than of the list.

**Hover GROWS the swatch**, 22 px to 30 px over 120 ms. Not decoration: with
labels gone, size is what tells you which swatch the pointer has, and an
instant jump reads as a flicker where an eased one reads as a response. The
box each swatch occupies is fixed at 34 px, so growing one never nudges its
neighbours -- the row must not reflow under a moving pointer.
"""

from __future__ import annotations

from qtpy.QtCore import Property, QEasingCurve, QPropertyAnimation, QRectF, Qt, Signal
from qtpy.QtGui import QColor, QPainter
from qtpy.QtWidgets import QGraphicsOpacityEffect, QHBoxLayout, QVBoxLayout, QWidget

from ..theme import icons
from ..theme import typography as ty
from .primitives import label

__all__ = ["PaletteRow", "Swatch", "SwatchRow"]


class Swatch(QWidget):
    """One colour. Grows when pointed at, ticked when it is the current one."""

    picked = Signal(int)
    entered = Signal(int)

    BOX = 34
    REST = 22.0
    GROWN = 30.0

    def __init__(self, t, colour: str, index: int, current: bool = False,
                 parent=None):
        super().__init__(parent)
        self.t = t
        self.colour = colour
        self.index = index
        self.current = current
        self._size = self.REST
        self.setFixedSize(self.BOX, self.BOX)
        self.setCursor(Qt.PointingHandCursor)
        self.setAttribute(Qt.WA_Hover, True)
        self._anim = QPropertyAnimation(self, b"swatchSize", self)
        self._anim.setDuration(120)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)

    # -- the animated property ------------------------------------------------
    def _get_size(self) -> float:
        return self._size

    def _set_size(self, value: float) -> None:
        self._size = value
        self.update()

    #: Animated by QPropertyAnimation, so it has to be a real Qt property --
    #: a plain attribute is invisible to the animation system.
    swatchSize = Property(float, _get_size, _set_size)

    def _animate_to(self, target: float) -> None:
        self._anim.stop()
        self._anim.setStartValue(self._size)
        self._anim.setEndValue(target)
        self._anim.start()

    def enterEvent(self, e):
        self._animate_to(self.GROWN)
        self.entered.emit(self.index)
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._animate_to(self.REST)
        super().leaveEvent(e)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self.rect().contains(e.pos()):
            self.picked.emit(self.index)

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        # SNAP TO WHOLE DEVICE PIXELS.
        #
        # The shape is antialiased either way -- that was measured, 11 distinct
        # tones across the corner -- but at a 1.25 ratio a 22 px square centred
        # in a 34 px box starts at logical 6.0, which is device 7.5. Every
        # straight edge then lands on a device HALF-pixel and gets smeared
        # across two rows at 50 % each. Antialiasing a straight vertical edge
        # is pure loss: there is nothing to smooth, only sharpness to give up,
        # and at 22 px it reads as a rough border rather than as softness.
        #
        # Measured: unsnapped leaves one partial pixel on each straight edge,
        # snapped leaves none.
        ratio = self.devicePixelRatioF() or 1.0
        side = round(self._size * ratio) / ratio
        offset = round((self.BOX - side) / 2 * ratio) / ratio
        box = QRectF(offset, offset, side, side)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(self.colour))
        # Radius scales with the swatch so the corner stays the same SHAPE as it
        # grows; a fixed radius reads as the square getting boxier.
        p.drawRoundedRect(box, side * 0.27, side * 0.27)
        if self.current:
            # The tick is drawn ON the colour, in whichever of black or white
            # the colour can actually carry.
            ink = "#ffffff" if _is_dark(self.colour) else "#09090B"
            inset = side * 0.26
            icons.paint(p, "check", box.adjusted(inset, inset, -inset, -inset),
                        ink, stroke=3.4)
        p.end()


def _is_dark(hexcode: str) -> bool:
    """Relative luminance, the WCAG way, rounded to a yes/no.

    Not ``hcl.contrast`` -- this is a two-way choice on a solid fill, and
    pulling the whole colour-science module in for one threshold would make
    every importer of this file pay for CIELUV.
    """
    r, g, b = (int(hexcode[i:i + 2], 16) / 255 for i in (1, 3, 5))
    channels = [(c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4)
                for c in (r, g, b)]
    lum = 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]
    return lum < 0.45


class SwatchRow(QWidget):
    """The swatches on one line, and a caption that follows the pointer."""

    picked = Signal(int)

    def __init__(self, t, colours, names, *, current: int = 0, holders=None,
                 owner: str = "", parent=None):
        super().__init__(parent)
        self.t = t
        self._names = list(names)
        self._holders = list(holders or [None] * len(colours))
        self._owner = owner
        self._current = current

        strip = QWidget()
        strip.setStyleSheet("background:transparent;")
        sl = QHBoxLayout(strip)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(2)
        for i, colour in enumerate(colours):
            chip = Swatch(t, colour, i, current=i == current)
            chip.picked.connect(self.picked.emit)
            chip.entered.connect(self._describe)
            sl.addWidget(chip)

        self._caption = label("", ty.ui("xs"), t.muted_foreground)
        # The caption is fixed-width and the row never reflows, so naming a
        # longer colour cannot widen the popover under the pointer.
        self._caption.setFixedWidth(len(colours) * (Swatch.BOX + 2))

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(6)
        lay.addWidget(strip)
        lay.addWidget(self._caption)
        self._describe(current)

    def _describe(self, index: int) -> None:
        """Name the pointed-at colour, and warn if taking it would swap."""
        name = self._names[index].capitalize()
        holder = self._holders[index]
        if holder and holder != self._owner:
            self._caption.setText(f"{name}: swap with {holder}")
        elif index == self._current:
            self._caption.setText(f"{name} (current)")
        else:
            self._caption.setText(name)


class PaletteRow(QWidget):
    """One palette in the picker: its name, its colours, and a tick if current.

    Showing the COLOURS is the whole point. "Okabe-Ito" and "Tol muted" are
    names of sets, and a menu that offers two names asks you to already know
    which set is which -- where five swatches simply show you.
    """

    triggered = Signal()

    def __init__(self, t, label_text: str, colours, current: bool = False,
                 disabled: bool = False, note: str = "", parent=None):
        """``disabled``: an UNSUPPORTED palette (fewer colours than conditions) --
        shown so the choice is visible, muted, with ``note`` saying why, and inert."""
        super().__init__(parent)
        self.t = t
        self.disabled = disabled
        self.setObjectName("prow")
        # No WA_StyledBackground and no stylesheet on the row itself: the
        # hover fill is painted (see paintEvent). It went through a stylesheet
        # first, which needed that attribute to draw at all on a QWidget
        # subclass -- and then rendered its radius in device pixels anyway.
        self._paint(False)
        self.setAttribute(Qt.WA_Hover, not disabled)
        if not disabled:
            self.setCursor(Qt.PointingHandCursor)
        # Matches DropdownMenuItem: ``py-1.5`` + ``text-sm`` = 32.
        self.setFixedHeight(32)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 0, 10, 0)
        lay.setSpacing(10)
        tick = icons_check(t) if current else None
        if tick is not None:
            lay.addWidget(tick)
        else:
            spacer = QWidget()
            spacer.setFixedWidth(13)
            spacer.setStyleSheet("background:transparent;")
            lay.addWidget(spacer)
        lay.addWidget(label(label_text, ty.ui("sm"),
                            t.muted_foreground if disabled else t.foreground))
        if note:
            lay.addWidget(label(note, ty.ui("xs"), t.muted_foreground))
        lay.addStretch(1)
        chips = QWidget()
        chips.setStyleSheet("background:transparent;")
        cl = QHBoxLayout(chips)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(3)
        for hexcode in colours:
            dot = QWidget()
            dot.setFixedSize(12, 12)
            dot.setStyleSheet(f"background:{hexcode}; border-radius:3px;")
            cl.addWidget(dot)
        if disabled:
            fade = QGraphicsOpacityEffect(chips)
            fade.setOpacity(0.4)
            chips.setGraphicsEffect(fade)
        lay.addWidget(chips)

    def _paint(self, hover: bool) -> None:
        """Only a repaint request now -- see :meth:`paintEvent`."""
        self._hover = hover
        self.update()

    def paintEvent(self, e):
        """Painted, not styled: a stylesheet ``border-radius`` lands in DEVICE
        pixels here, rendering 20 %% small at ratio 1.25. See
        ``menus._Row.paintEvent`` for the calibration."""
        if not getattr(self, "_hover", False):
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(self.t.accent))
        r = self.t.radius_sm
        p.drawRoundedRect(QRectF(self.rect()), r, r)
        p.end()

    def enterEvent(self, e):
        if not self.disabled:
            self._paint(True)
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._paint(False)
        super().leaveEvent(e)

    def mouseReleaseEvent(self, e):
        if (not self.disabled and e.button() == Qt.LeftButton
                and self.rect().contains(e.pos())):
            self.triggered.emit()


def icons_check(t) -> QWidget:
    """The tick used by :class:`PaletteRow`, kept out of the class so the
    import of ``primitives.icon`` stays at module level."""
    from .primitives import icon
    return icon("check", icons.SM, t.foreground)
