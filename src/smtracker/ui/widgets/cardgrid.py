"""Cards in as many equal columns as the width holds: `CardGrid`.

Asked for on 13 Oct 2026 ("the 2x2 grid is not well supported"): the HMM cards sat in a
fixed two-column grid, and at the width the workspace leaves beside the rail each card's
diagram and rates table had no room. Here the COLUMN COUNT follows the width: as many
columns of at least ``min_width`` as fit, at most ``max_columns``.

**WHERE THE COLUMNS ARE DECIDED: `setGeometry`**, the pass that is given the width, never a
`resizeEvent` (which lands one pass late -- on every frame of the rail's slide; the record
is `docs/claude/qt_and_packaging.md`, "A FIXED WIDTH DERIVED FROM THE CURRENT WIDTH IS A
RATCHET").

**NOT HEIGHT-FOR-WIDTH.** A layout that answers `heightForWidth` makes every layout above it
re-solve on each width change (29 ms a frame measured on the QC page while the rail slides).
The height reported is the one for the column count of the LAST geometry; when a new width
changes that count the layout invalidates itself and the parent asks again. The count
depends on the width alone, never on the height, so this settles in one more pass.

**THE MINIMUM WIDTH IS A CONSTANT** (``min_width``): a minimum taken from a current width
ratchets every parent layout up.

Rows are as tall as their tallest card; cards in a row share its height. No card is given a
minimum height by anyone: its layout's own is the only one (an explicit `setMinimumHeight`
REPLACES it, and a card set shorter than its content draws its children over each other).

A CARD'S OWN MINIMUM WIDTH COUNTS (3 Nov 2026, reported: the Clustering and Signed jumps cards' tables were cut at
1366 px, the cards needing 555 and 596 px in columns of 491): the column count is decided on the larger of
``min_width`` and the widest card's own minimum, so a card is never laid narrower than its content while one column
fewer would hold it. Only the COUNT reads it; the layout's minimum stays ``min_width`` (no ratchet: a card's minimum
does not follow the width it is given).

TWO OPTIONS, NOTHING CHANGES WITHOUT THEM (30 Oct 2026, the Jump distances page's All conditions
view: "increasing size of these cards in a way if its too small"): `set_column_width` asks for
wider columns (fewer of them) while the layout's MINIMUM stays ``min_width`` -- a minimum taken
from a size the user picked would ratchet the window too; and `set_wide` lets a card take a row
of its own across every column.
"""
from __future__ import annotations

from qtpy.QtCore import QRect, QSize, Qt
from qtpy.QtWidgets import QLayout

__all__ = ["CardGrid", "columns_for"]


def columns_for(width: int, *, min_width: int, max_columns: int, spacing: int) -> int:
    """How many columns of at least ``min_width`` (``spacing`` between them) fit in
    ``width``: at least one, at most ``max_columns``."""
    fit = (int(width) + spacing) // max(1, int(min_width) + spacing)
    return max(1, min(int(max_columns), fit))


class CardGrid(QLayout):
    def __init__(self, parent=None, *, min_width: int, max_columns: int = 2,
                 spacing: int = 16):
        super().__init__(parent)
        self._items: list = []
        self._min_width = int(min_width)
        self._max_columns = int(max_columns)
        self._columns = 1
        #: The column width asked for (`set_column_width`); None: ``min_width``.
        self._column_width = None
        #: Cards that span every column (`set_wide`).
        self._wide: set = set()
        #: Exactly this many columns (`set_columns`), or None: as many as fit, at most ``max_columns``.
        self._fixed = None
        self.setSpacing(int(spacing))
        self.setContentsMargins(0, 0, 0, 0)

    # -- QLayout's container protocol --
    def addItem(self, item) -> None:
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int):
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def columns(self) -> int:
        """The column count of the last geometry."""
        return self._columns

    def set_column_width(self, width: int | None) -> None:
        """Columns of at least ``width`` (None: ``min_width``) as fit; the layout's minimum stays ``min_width``."""
        self._column_width = None if width is None else int(width)
        self.invalidate()

    def set_columns(self, columns: int | None) -> None:
        """EXACTLY ``columns`` columns, each at least its minimum width (the layout is then as wide as they need: put
        it where it may scroll sideways), or None: as many as fit, at most ``max_columns``, as before (7 Nov 2026,
        asked: "draw and export as grids for example 4x2 / 2x4")."""
        self._fixed = None if columns is None else max(1, int(columns))
        self.invalidate()

    def fixed_columns(self) -> int | None:
        return self._fixed

    def _floor(self) -> int:
        """The narrowest a column may be: the width asked for, and no card's minimum cut."""
        return max([self._column_width or self._min_width]
                   + [it.minimumSize().width() for it in self._visible() if not self._spans(it)])

    def set_wide(self, widget, on: bool) -> None:
        """``widget`` (a card in this grid) on a row of its own across every column, or back among the others."""
        if on:
            self._wide.add(id(widget))
        else:
            self._wide.discard(id(widget))
        self.invalidate()

    def is_wide(self, widget) -> bool:
        return id(widget) in self._wide

    def reorder(self, widgets) -> None:
        """The cards in the order of ``widgets`` (the conditions' order, `condition_order`, 3 Nov 2026); a card not
        named keeps its place after them. Nothing is rebuilt: the same widgets, laid out again."""
        rank = {id(w): i for i, w in enumerate(widgets)}
        far = len(rank)
        self._items.sort(key=lambda item: rank.get(id(item.widget()), far))
        self.invalidate()

    def _spans(self, item) -> bool:
        widget = item.widget()
        return widget is not None and id(widget) in self._wide

    # -- sizes --
    def _visible(self) -> list:
        return [it for it in self._items if not it.isEmpty()]

    @staticmethod
    def _height_of(item) -> int:
        return max(item.sizeHint().height(), item.minimumSize().height())

    def _rows(self, columns: int) -> list:
        """The visible cards in rows of ``columns``, a wide card (`set_wide`) closing its row and taking one of its
        own; without wide cards, plain chunks of ``columns`` as always."""
        rows, row = [], []
        for item in self._visible():
            if self._spans(item):
                if row:
                    rows.append(row)
                    row = []
                rows.append([item])
                continue
            row.append(item)
            if len(row) == columns:
                rows.append(row)
                row = []
        if row:
            rows.append(row)
        return rows

    def _total_height(self, columns: int) -> int:
        rows = self._rows(columns)
        m = self.contentsMargins()
        body = sum(max(self._height_of(it) for it in row) for row in rows)
        return body + self.spacing() * max(0, len(rows) - 1) + m.top() + m.bottom()

    def minimumSize(self) -> QSize:
        m = self.contentsMargins()
        if self._fixed is not None:
            n = self._fixed
            return QSize(n * self._floor() + self.spacing() * (n - 1) + m.left() + m.right(),
                         self._total_height(n))
        return QSize(self._min_width + m.left() + m.right(), self._total_height(self._columns))

    def sizeHint(self) -> QSize:
        return self.minimumSize()

    def setGeometry(self, rect: QRect) -> None:
        super().setGeometry(rect)
        m = self.contentsMargins()
        inner = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        spacing = self.spacing()
        columns = (self._fixed if self._fixed is not None else
                   columns_for(inner.width(), min_width=self._floor(), max_columns=self._max_columns, spacing=spacing))
        if columns != self._columns:
            self._columns = columns
            self.invalidate()           # the height for the new count: the parent asks again
        width = (inner.width() - spacing * (columns - 1)) / columns
        y = inner.top()
        for row in self._rows(columns):
            height = max(self._height_of(it) for it in row)
            if len(row) == 1 and self._spans(row[0]):
                row[0].setGeometry(QRect(inner.left(), y, inner.width(), height))
                y += height + spacing
                continue
            for k, item in enumerate(row):
                x = inner.left() + round(k * (width + spacing))
                item.setGeometry(QRect(x, y, round(width), height))
            y += height + spacing
