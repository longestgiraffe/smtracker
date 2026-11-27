"""shadcn's HoverCard: a paragraph shown while the pointer rests on its trigger.

Read off the registry (``shadcn view @shadcn/hover-card``)::

    HoverCardContent   w-64 rounded-md border bg-popover p-4 text-popover-foreground shadow-md
    sideOffset         4
    align              center

**Why not `tooltip.Tooltip`.** That component is ONE LINE of plain text, sized from
`horizontalAdvance` and inverted (`bg-foreground`) because it labels a control. An
explanation of a mechanism is a paragraph -- set on one line it would run the width of
the screen -- and shadcn's own answer for "more than a label, on hover" is the hover
card: the popover's colours, a border, a shadow, and wrapped text. Same placement rules
as the tooltip: painted as a CHILD of the window (no popup compositing), clamped to it,
flipped above the trigger when there is no room below.

The one departure is the width: `w-80` (20 rem) rather than `w-64`, because the texts here
run to a hundred words and 16 rem made them a tall, narrow column.
"""

from __future__ import annotations

from qtpy.QtCore import QEvent, QObject, QPoint, QRectF, Qt
from qtpy.QtGui import QColor, QPainter, QPen
from qtpy.QtWidgets import QVBoxLayout, QWidget

from ..theme import typography as ty
from .primitives import draw_shadow, label, shadow_room
from .table import rem


class HoverCard(QWidget):
    """The card itself. Shown by :class:`_HoverWatch`; never takes the mouse."""

    WIDTH = rem(20)
    SIDE_OFFSET = 4

    def __init__(self, t, parent: QWidget | None = None):
        super().__init__(parent)
        self.t = t
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._room = shadow_room(t, t.shadow_md)
        lay = QVBoxLayout(self)
        pad = rem(1)
        lay.setContentsMargins(self._room + pad, self._room + pad,
                               self._room + pad, self._room + pad)
        # `text-popover-foreground`, which is `--foreground` in this theme (no separate token)
        self._text = label("", ty.ui("sm"), t.foreground)
        self._text.setWordWrap(True)
        self._text.setFixedWidth(self.WIDTH - 2 * pad)
        lay.addWidget(self._text)
        self.hide()

    def text(self) -> str:
        return self._text.text()

    def set_text(self, text: str) -> None:
        self._text.setText(text)
        self.adjustSize()

    def show_for(self, target: QWidget) -> None:
        """Below `target`, centred on it, clamped to the window; above it when the window
        has no room below."""
        window = target.window()
        if self.parentWidget() is not window:
            self.setParent(window)
        self.adjustSize()
        w, h = self.width(), self.height()
        top_left = target.mapTo(window, QPoint(0, 0))
        x = top_left.x() + target.width() // 2 - w // 2
        x = max(0, min(x, max(0, window.width() - w)))
        y = top_left.y() + target.height() + self.SIDE_OFFSET - self._room
        if y + h > window.height():
            y = top_left.y() - self.SIDE_OFFSET - h + self._room
        self.move(x, max(0, y))
        self.raise_()
        self.show()

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        box = QRectF(self.rect().adjusted(self._room, self._room, -self._room, -self._room))
        radius = self.t.radius_md
        # Drawn directly, like the chart tooltips: a widget reparented onto the window at
        # show time has no `ShadowHost` ancestor to draw one for it.
        draw_shadow(p, box, radius, self.t, self.t.shadow_md)
        p.setPen(QPen(QColor(self.t.border), 1.0))
        p.setBrush(QColor(self.t.popover))
        p.drawRoundedRect(box.adjusted(0.5, 0.5, -0.5, -0.5), radius, radius)
        p.end()


class _HoverWatch(QObject):
    """Shows the card while the pointer is over the trigger, hides it on leave, and
    stays silent while the text is empty (an empty card is not a card)."""

    def __init__(self, target: QWidget, card: HoverCard, anchor: QWidget | None = None):
        super().__init__(target)
        self.target, self.card = target, card
        #: What the card is PLACED against; the hover is still watched on `target`. A
        #: trigger that stretches across its row (a label beside a stretch) would centre
        #: the card on empty space, so the caller can name the visible text instead.
        self.anchor = anchor or target
        target.installEventFilter(self)

    def eventFilter(self, obj, event):
        kind = event.type()
        try:
            if kind == QEvent.Enter and self.card.text():
                self.card.show_for(self.anchor)
            elif kind in (QEvent.Leave, QEvent.Hide, QEvent.MouseButtonPress):
                self.card.hide()
        except RuntimeError:
            # The card is reparented onto the WINDOW, which may tear it down first on
            # close -- the same teardown order `dynamics_chart._HoverChart` catches.
            pass
        return False


def attach(t, target: QWidget, text: str = "", *, anchor: QWidget | None = None) -> HoverCard:
    """Give `target` a hover card holding `text`; returns the card (`set_text` to change
    it later, an empty text to silence it). `anchor` is what it is placed against, when
    that is not the whole of `target`."""
    card = HoverCard(t, target)
    card.set_text(text)
    card._watch = _HoverWatch(target, card, anchor)
    return card
