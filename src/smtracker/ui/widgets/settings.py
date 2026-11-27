"""The two-level settings column and the info line, shared by every settings guide.

Moved here from `ui.workspace.spatial_map_chart`, where the Spatial Map's Run dialog
built them, so the preparation window's guides use the SAME widgets rather than a
second copy: a column of settings that keeps every value on screen, and an info line
whose words are a tooltip. `spatial_map_chart` and `qc_chart` import them from here.
"""
from __future__ import annotations

from qtpy.QtCore import QRectF, Qt, Signal
from qtpy.QtGui import QColor, QFontMetricsF, QPainter, QPen
from qtpy.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from ..theme import typography as ty
from . import hovercard
from .primitives import discard, icon, label
from .table import rem, text_px

__all__ = ["SettingRow", "SettingNav", "InfoLink", "SHORT_INK", "LONG_INK", "ACCENT_INK",
           "CUT_INK", "WARN_INK"]

#: THE GUIDES' OWN PALETTE, for mechanism diagrams -- never a map's colormap, so a
#: diagram cannot be mistaken for a result. Short / long (or slow / fast), a third, what
#: a cut removes, and a warning.
SHORT_INK = "#3f5d82"
LONG_INK = "#a8553f"
ACCENT_INK = "#8c6f4e"
CUT_INK = "#c0362c"
WARN_INK = "#c05621"


class SettingRow(QWidget):
    """One row of the Run dialog's nav: a dot, the setting's NAME, and the
    value it currently holds.

    **The value is on screen at all times, which is the whole point of the
    two-level nav.** A tabbed dialog hides five settings behind the one you
    are looking at, so checking what you set means clicking around; here the
    column answers that by being read.

    Selection is carried by the DOT and by the name's ink -- never by making
    the value dark too. The first draft did that, and on the selected row
    name and value went the same colour and read as one run of text.
    """

    clicked = Signal(int)

    DOT = 9
    GUTTER = 19
    #: Read by BOTH the layout and `paintEvent`'s dot placement -- they have
    #: to agree, and two literals eventually would not.
    PAD_Y = 7

    def __init__(self, t, index: int, name: str, parent=None):
        super().__init__(parent)
        self.t = t
        self.index = index
        self._active = False
        self.setCursor(Qt.PointingHandCursor)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("background:transparent;")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(self.GUTTER, self.PAD_Y, rem(0.5), self.PAD_Y)
        lay.setSpacing(2)
        self._name = label(name, ty.ui("sm", "medium", px=text_px(0.875)),
                           t.muted_foreground)
        self._name.setWordWrap(True)
        #: CAPS, and written in the VALUE rather than applied as a style --
        #: Qt has no `text-transform`, but the rule behind that choice holds
        #: anyway: `100 nm` must not become `100 NM`, which is not a unit,
        #: and a micro sign upper-cases to M, i.e. micrometres to
        #: millimetres. `SettingRow.set_value`'s callers pass the exact
        #: string to draw.
        #: 12px against the name's 14, not 10: caps at 10px read as a
        #: footnote beside a 14px name, and the value is the thing you scan
        #: this column FOR.
        self._value = label("", ty.ui("xs", "semibold", px=text_px(0.75),
                                      tabular=True), t.muted_foreground)
        lay.addWidget(self._name)
        lay.addWidget(self._value)

    def set_value(self, text: str) -> None:
        self._value.setText(text)

    def set_active(self, on: bool) -> None:
        if on == self._active:
            return
        self._active = on
        self._name.setFont(ty.ui("sm", "semibold" if on else "medium",
                                 px=text_px(0.875)))
        self._name.setStyleSheet(
            f"color:{self.t.foreground if on else self.t.muted_foreground};"
            " background:transparent;")
        self.update()

    def mouseReleaseEvent(self, e) -> None:
        if e.button() == Qt.LeftButton:
            self.clicked.emit(self.index)

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        # Centred on the name's FIRST line -- measured from the font rather
        # than guessed from the point size, and taken from the first line so
        # a name that wraps ("Minimum jumps per bin") does not drag the dot
        # down to the middle of two.
        y = self.PAD_Y + QFontMetricsF(self._name.font()).height() / 2.0
        box = QRectF(2.0, y - self.DOT / 2.0, self.DOT, self.DOT)
        if self._active:
            p.setBrush(QColor(self.t.foreground))
            p.setPen(Qt.NoPen)
        else:
            p.setBrush(QColor(self.t.card))
            pen = QPen(QColor(self.t.border))
            pen.setWidthF(1.0)
            p.setPen(pen)
        p.drawEllipse(box)
        p.end()


class SettingNav(QWidget):
    """The Run dialog's left column: every setting, and what it is set to."""

    changed = Signal(int)

    def __init__(self, t, names, parent=None):
        super().__init__(parent)
        self.t = t
        self.setStyleSheet("background:transparent;")
        self.setFixedWidth(rem(12.25))
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(0)
        self.rows: list[SettingRow] = []
        self.set_items(names)

    def set_items(self, names) -> None:
        """Rebuild the row set for a new list of names, keeping row 0 active.

        Needed because the Computational method control changes which OTHER
        rows even apply (`_KEYS_BY_METHOD`) -- unlike every other control
        here, which only changes what ONE already-existing row is set to.
        `takeAt`, never `setParent(None)` alone plus a stale Python list:
        `_show`'s own rebuild of `_control_lay` is the precedent this
        follows.
        """
        while self._lay.count():
            item = self._lay.takeAt(0)
            w = item.widget()
            if w is not None:
                discard(w)
        self.rows = []
        for i, name in enumerate(names):
            row = SettingRow(self.t, i, name)
            row.clicked.connect(self._pick)
            self._lay.addWidget(row)
            self.rows.append(row)
        self._lay.addStretch(1)
        self._current = 0
        if self.rows:
            self.rows[0].set_active(True)

    def _pick(self, index: int) -> None:
        if index == self._current:
            return
        self.rows[self._current].set_active(False)
        self._current = index
        self.rows[index].set_active(True)
        self.changed.emit(index)

    def current(self) -> int:
        return self._current

    def select(self, index: int) -> None:
        """Make row ``index`` the open one WITHOUT emitting `changed`: for a caller that
        already knows which setting it is showing (a grid cell was clicked)."""
        if index == self._current or not 0 <= index < len(self.rows):
            return
        self.rows[self._current].set_active(False)
        self._current = index
        self.rows[index].set_active(True)

    def set_value(self, index: int, text: str) -> None:
        self.rows[index].set_value(text)


class InfoLink(QWidget):
    """An info line whose words are a hover card (shadcn's HoverCard), never printed: the app's own lucide
    `info` icon beside a short link text. An ICON and not the U+24D8 character, because
    Inter has no such glyph and a glyph missing from Inter sends Qt searching every
    installed font on the GUI thread (`ui/theme/typography.py` measured seven seconds
    for one character)."""

    def __init__(self, t, text: str = "", tip: str = "", parent=None):
        super().__init__(parent)
        self.setStyleSheet("background:transparent;")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        lay.addWidget(icon("info", 12, t.muted_foreground, stroke=1.75), 0, Qt.AlignVCenter)
        self._text = label(text, ty.ui("xs"), t.muted_foreground)
        lay.addWidget(self._text)
        lay.addStretch(1)
        self.setCursor(Qt.WhatsThisCursor)
        #: shadcn's HoverCard, not a tooltip: these words are a paragraph, and the
        #: one-line `Tooltip` would stretch it across the screen (`widgets.hovercard`).
        self._card = hovercard.attach(t, self, anchor=self._text)
        self.set_tip(tip)

    def text(self) -> str:
        return self._text.text()

    def setText(self, text: str) -> None:  # noqa: N802 - the QLabel spelling, on purpose
        self._text.setText(text)

    def set_tip(self, tip: str) -> None:
        self._card.set_text(tip)

    def tip(self) -> str:
        return self._card.text()
