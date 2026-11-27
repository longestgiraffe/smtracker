"""The HMM section's Table page's table (22 Nov 2026): one state in every condition, a row a condition, the columns
`hmm_view.condition_table` gives -- D, dwell, the three shares, the rates in and out -- right-aligned, tabular, the
decimal points of a column under one another, a click picking a column. The UI designer's table from the HMM redesign (mockups
24-27), kept when the redesign was retired (23 Nov 2026: "retire hmm redesign. keep old"). Thin Qt: the numbers and
words are `hmm_view`'s."""
from __future__ import annotations

from qtpy.QtCore import QPoint, QRectF, Qt, Signal
from qtpy.QtGui import QFontMetricsF
from qtpy.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from ..theme import typography as ty
from ..widgets.primitives import discard, label

__all__ = ["ConditionTable", "TableCell"]


def _clear(w: QWidget) -> QWidget:
    w.setStyleSheet("background:transparent;")
    return w


class TableCell(QLabel):
    """One value of the Table view: right-aligned, tabular, a click picks its column. The picked
    column is shadcn's `bg-muted` band (header included); text as `TableCell` sets it."""

    clicked = Signal(str)

    #: The values' font, also what `_ConditionTable` measures the decimal points with.
    @staticmethod
    def value_font():
        return ty.ui("sm", "normal", px=14, tabular=True)

    def __init__(self, t, text: str, key: str | None, *, head: bool = False, picked: bool = False,
                 muted: bool = False, align: float = 0.0):
        super().__init__(text)
        self.key = key
        self.setFont(ty.ui("sm", "medium", px=14) if head else self.value_font())
        self.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        ink = t.muted_foreground if muted else t.foreground
        fill = t.muted if picked else "transparent"
        # `p-2`: the number keeps its padding from the band's edge; ``align`` more on the right
        # puts its decimal point under the column's others
        right = 8 + round(align)
        self.pad_right = right
        self.setStyleSheet(f"color:{ink}; background:{fill}; padding:0 {right}px 0 8px;")
        if key is not None and not head:
            self.setCursor(Qt.PointingHandCursor)

    def mouseReleaseEvent(self, e) -> None:
        if e.button() == Qt.LeftButton and self.key is not None and self.rect().contains(e.pos()):
            self.clicked.emit(self.key)
        super().mouseReleaseEvent(e)


class ConditionTable(QWidget):
    """The Table view (the UI designer, mockups 24-27): ONE state, a row per condition the filter shows
    (pinned first under All, each with a "Pinned" badge), the columns `hmm_view.condition_table`
    gives -- D, dwell, the three shares, the rates in and out. A click on a value picks its column
    page-wide (`cell_clicked`); the picked column is shadcn's muted band.

    shadcn's DATA TABLE: the app's Table (`widgets.table.TableBody`, the SQD section's) in a
    `rounded-md border` frame, the header on the page colour with medium foreground text, a rule
    under every row, the row hover, numbers right-aligned and tabular, decimal points aligned (the UI designer, 21 Oct 2026: the hand-painted table -- a
    bordered box, small grey headers, a bar under every number, a heavy rule after the pinned
    rows, a divider before the rates -- "doesn't follow the shadcn's style at all"). The bars
    went with it (the UI designer chose to drop them)."""

    cell_clicked = Signal(str)

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.table, self.picked, self.pinned = None, None, 0
        self.body = None
        self.cells: dict = {}
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(0)
        self.setStyleSheet("background:transparent;")
        from qtpy.QtWidgets import QFrame

        self._frame = QFrame()
        self._frame.setObjectName("datatable")
        self._frame.setStyleSheet(
            f"QFrame#datatable {{ background:{t.background}; border:1px solid {t.border};"
            f" border-radius:{t.radius_md:g}px; }}")
        fl = QVBoxLayout(self._frame)
        fl.setContentsMargins(1, 1, 1, 1)                 # inside the 1 px border
        fl.setSpacing(0)
        self._lay.addWidget(self._frame)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set(self, table, picked, pinned_count: int, groups=()) -> None:
        """``groups`` ``[(first row, outer value)]``: the folder design's groups as section rows (`factor_layout`),
        used when nothing is pinned (pinning has its own two sections)."""
        from ..widgets.table import TableBody

        self.table, self.picked, self.pinned = table, picked, int(pinned_count)
        if self.body is not None:
            discard(self.body)
            self.body = None
        self.cells = {}
        if not table:
            self.setFixedHeight(0)
            return
        t = self.t
        cols = table["columns"]
        heads = [label("Condition", ty.ui("sm", "medium", px=14), t.foreground)]
        heads += [TableCell(t, h, k, head=True, picked=k == picked) for k, h, _g in cols]
        # DECIMAL POINTS ALIGNED in every column (the UI designer, 21 Oct 2026: "so that the users could
        # easily compare things"), as the cards' table aligns D: each column's widest fraction
        # (the point and its digits), and every value padded on the right by what its own
        # fraction lacks -- so 0.82 sits under 5.4 at the point, and a whole number or a dash
        # ends where the points are
        fm = QFontMetricsF(TableCell.value_font())

        def frac(text: str) -> float:
            return fm.horizontalAdvance(text[text.index("."):]) if "." in text else 0.0

        widest = {k: max([frac(r["values"][k][0]) for r in table["rows"] if r["values"]] or [0.0])
                  for k, _h, _g in cols}
        rows = []
        for i, row in enumerate(table["rows"]):
            cells = [self._name_cell(row)]
            for k, _h, _g in cols:
                if row["values"]:
                    text = row["values"][k][0]
                    cell = TableCell(t, text, k, picked=k == picked,
                                      align=widest[k] - frac(text))
                    cell.clicked.connect(self.cell_clicked.emit)
                else:
                    cell = TableCell(t, "–", None, picked=k == picked, muted=True,
                                      align=widest[k])
                self.cells[(i, k)] = cell
                cells.append(cell)
            rows.append(cells)
        # shadcn's DATA TABLE (the UI designer, 21 Oct 2026, with its docs page): the table in its own
        # `rounded-md border` frame, the header on the page colour (no grey band), the last
        # row's corners rounded inside the frame as `overflow-hidden` would clip them
        radius = float(t.radius_md)
        # the cards' groups as section rows (mockup 28.3): the pinned first, then the others
        n_rows = len(table["rows"])
        sections = []
        if 0 < self.pinned:
            sections.append((0, f"Pinned · {self.pinned}"))
            if self.pinned < n_rows:
                sections.append((self.pinned, f"Others · {n_rows - self.pinned}"))
        elif groups:
            sections = [(int(first), str(name)) for first, name in groups]
        self.body = TableBody(t, heads, rows, "", head_fill=t.background,
                              bottom_radius=radius - 1, sections=sections)
        self._frame.layout().addWidget(self.body)
        # the body, the frame's margins AND its 1 px border (+2 alone cut the last row's foot by 2 px, the GUI sweep
        # of 23 Nov 2026): never under what the layout itself needs
        self.setFixedHeight(max(self.body.natural_height() + 4, self._lay.totalMinimumSize().height()))

    def _name_cell(self, row) -> QWidget:
        """The condition: its colour, its name, and -- a fit without this state -- why its row
        is empty. (Pinned is the "Pinned" section row above it, not a badge.)"""
        t = self.t
        w = _clear(QWidget())
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        lay.addWidget(label("●", ty.ui("sm"), row["colour"]))
        lay.addWidget(label(row["name"], ty.ui("sm", "medium", px=14), t.foreground))
        if row["missing"]:
            lay.addWidget(label(row["missing"], ty.ui("xs"), t.muted_foreground))
        lay.addStretch(1)
        return w

    def cell_rect(self, i: int, key: str) -> QRectF:
        """Where row ``i``'s ``key`` value sits, in this widget's coordinates."""
        cell = self.cells[(i, key)]
        return QRectF(cell.geometry().translated(cell.parentWidget().mapTo(self, QPoint(0, 0))))
