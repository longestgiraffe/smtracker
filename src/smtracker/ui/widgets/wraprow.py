"""A row of controls that WRAPS onto further lines when the width runs out: `WrapRow`.

Reported 30 Oct 2026 from a 1536 px laptop: Jump distances' All conditions view drew its header past the window's
edge -- the "Each condition | All conditions" switch read "All c", Export all only its icon and "E". Each header was
a `QHBoxLayout`, so its minimum width was the sum of its controls; inside a page that scrolls only vertically, a
page narrower than that sum is CUT at its right edge (the GUI sweep's CLIPPED finding), never squeezed or wrapped.
Here the minimum width is the WIDEST single item, and the items flow onto as many lines as the width needs.

**WHERE THE LINES ARE DECIDED: `setGeometry`**, the pass that is given the width, as `cardgrid.CardGrid` decides its
columns -- never a `resizeEvent`, which lands one pass late.

**NOT HEIGHT-FOR-WIDTH**, for the same reason as `CardGrid` (a height-for-width layout makes every layout above it
re-solve on each width change, 29 ms a frame while the rail slides): the height reported is the one for the line
breaks of the LAST geometry; when a new width changes them the layout invalidates itself and the parent asks again,
which settles in one more pass since the breaks depend on the width alone.

USE: `addWidget` / `addLayout` (a group that must stay together -- a label and its control) in reading order, and
`addStretch()` once: every item after it stands at the RIGHT end of its line (the old row's stretch), on the same line
as the items before it where it fits there, else on a line of its own, still at the right.
"""
from __future__ import annotations

from qtpy.QtCore import QRect, QSize, Qt
from qtpy.QtWidgets import QLayout

__all__ = ["WrapRow", "wrap_lines"]


def wrap_lines(widths, available: int, spacing: int, right_from: int | None = None) -> list:
    """The items' line breaks for ``widths`` in ``available``: a list of lines, each a list of item indexes.

    Greedy, in order; an item never starts a line it shares only with nothing (a first item wider than the line still
    gets its own line). Items from ``right_from`` on (the right-aligned group) go on the last line if they all fit
    there, else start a line of their own (and wrap among themselves the same way)."""
    lines: list = []
    line: list = []
    used = 0
    n = len(widths)
    split = n if right_from is None else max(0, min(int(right_from), n))

    def place(i: int) -> None:
        nonlocal line, used
        w = int(widths[i])
        need = w if not line else used + spacing + w
        if line and need > available:
            lines.append(line)
            line, used = [i], w
        else:
            line.append(i)
            used = need

    for i in range(split):
        place(i)
    right = list(range(split, n))
    if right:
        right_w = sum(int(widths[i]) for i in right) + spacing * (len(right) - 1)
        if line and used + spacing + right_w > available:
            lines.append(line)
            line, used = [], 0
        for i in right:
            place(i)
    if line:
        lines.append(line)
    return lines


class WrapRow(QLayout):
    """Items left to right at their preferred widths, wrapping onto further lines; see the module docstring."""

    def __init__(self, parent=None, *, spacing: int = 8, line_spacing: int | None = None):
        super().__init__(parent)
        self._items: list = []
        self._right_from: int | None = None
        self._lines = 1
        #: The inner width of the last geometry (`invalidate` clears the layout's own record of it).
        self._width: int | None = None
        self._line_spacing = int(spacing if line_spacing is None else line_spacing)
        self.setSpacing(int(spacing))
        self.setContentsMargins(0, 0, 0, 0)

    # -- QLayout's container protocol --
    def addItem(self, item) -> None:
        self._items.append(item)

    def addLayout(self, layout) -> None:
        """A group that stays on one line (a label and its control)."""
        self.addChildLayout(layout)
        self.addItem(layout)

    def addStretch(self, _stretch: int = 1) -> None:
        """Every item added after this stands at the right end of its line."""
        self._right_from = len(self._items)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int):
        if not 0 <= index < len(self._items):
            return None
        if self._right_from is not None and index < self._right_from:
            self._right_from -= 1
        return self._items.pop(index)

    def expandingDirections(self):
        return Qt.Orientation(0)

    def lines(self) -> int:
        """The number of lines of the last geometry."""
        return self._lines

    # -- sizes --
    def _shown(self) -> tuple:
        """The items not hidden, and where the right-aligned group starts among them."""
        shown, right_from = [], None
        for i, item in enumerate(self._items):
            if self._right_from is not None and i >= self._right_from and right_from is None:
                right_from = len(shown)
            if not item.isEmpty():
                shown.append(item)
        if right_from is not None:
            right_from = min(right_from, len(shown))
        return shown, right_from

    @staticmethod
    def _width_of(item) -> int:
        return max(item.sizeHint().width(), item.minimumSize().width())

    @staticmethod
    def _height_of(item) -> int:
        return max(item.sizeHint().height(), item.minimumSize().height())

    def _breaks(self, width: int) -> tuple:
        shown, right_from = self._shown()
        return shown, right_from, wrap_lines([self._width_of(it) for it in shown], width, self.spacing(), right_from)

    def _height_for_lines(self, shown, lines) -> int:
        m = self.contentsMargins()
        body = sum(max(self._height_of(shown[i]) for i in line) for line in lines)
        return body + self._line_spacing * max(0, len(lines) - 1) + m.top() + m.bottom()

    def minimumSize(self) -> QSize:
        shown, _right = self._shown()
        m = self.contentsMargins()
        widest = max((max(it.minimumSize().width(), self._width_of(it)) for it in shown), default=0)
        height = 0
        if shown:
            # the breaks of the LAST geometry (one line before the first); never the layout's `geometry()`, which
            # `invalidate` empties -- the height then read one line while the items stood on two
            _s, _r, lines = self._breaks(self._width if self._width is not None else 1 << 20)
            height = self._height_for_lines(shown, lines)
        return QSize(widest + m.left() + m.right(), height)

    def sizeHint(self) -> QSize:
        shown, right_from = self._shown()
        m = self.contentsMargins()
        one_line = sum(self._width_of(it) for it in shown) + self.spacing() * max(0, len(shown) - 1)
        return QSize(one_line + m.left() + m.right(), self.minimumSize().height())

    def setGeometry(self, rect: QRect) -> None:
        super().setGeometry(rect)
        m = self.contentsMargins()
        inner = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        shown, right_from, lines = self._breaks(inner.width())
        changed = len(lines) != self._lines
        self._width, self._lines = inner.width(), len(lines)
        if changed:
            self.invalidate()           # the height for the new breaks: the parent asks again
        spacing = self.spacing()
        y = inner.top()
        for line in lines:
            height = max(self._height_of(shown[i]) for i in line)
            left = [i for i in line if right_from is None or i < right_from]
            right = [i for i in line if right_from is not None and i >= right_from]
            x = inner.left()
            for i in left:
                w = min(self._width_of(shown[i]), inner.width())
                h = self._height_of(shown[i])
                shown[i].setGeometry(QRect(x, y + (height - h) // 2, w, h))
                x += w + spacing
            widths = [min(self._width_of(shown[i]), inner.width()) for i in right]
            x = inner.right() + 1 - (sum(widths) + spacing * max(0, len(right) - 1))
            for i, w in zip(right, widths):
                h = self._height_of(shown[i])
                shown[i].setGeometry(QRect(max(inner.left(), x), y + (height - h) // 2, w, h))
                x += w + spacing
            y += height + self._line_spacing


def align_names(labels) -> None:
    """Every label in ``labels`` (the names leading a dialog's rows) as wide as the widest of them -- its whole text, or
    its minimum where that is wider -- so the controls after the names start in one column whatever a name's length
    (SQD's Fits page widens every name to the longest; the apparent-D and HMM Run dialogs did not, and a long name pushed
    its row's pills out of line: found by the overnight tester, 4 Nov 2026)."""
    labels = [w for w in labels if w is not None]
    if not labels:
        return
    width = max(max(w.sizeHint().width(), w.minimumWidth()) for w in labels)
    for w in labels:
        w.setFixedWidth(width)
