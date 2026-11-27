"""The Batch page both stage windows share: parent folders, a table of every condition
found under them, settings, Run / Stop and a log.

Asked for (13 Oct 2026): "a batch mode where u can run multiple parent folders in batches
overnight. Both for image prep / tracking and segmentation". The preparation window
(`ui.prep.batch_page`) and the outlines window (`ui.outlines.batch_page`) each add their own
settings to `settings` and fill the table from `smtracker.batch_model`.

THIN QT: rows arrive as plain dicts (``{key, cells}``) and lines as strings; this page
decides nothing and computes nothing. Rows are rebuilt only when the set of conditions
changes, and otherwise relabelled in place, so a running batch does not flicker. A removed
widget goes through `primitives.discard`, and every widget is in a layout before it is
shown (the stray-window rule, `tests/test_no_stray_windows.py`).
"""
from __future__ import annotations

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..theme import typography as ty
from .controls import Button
from .primitives import Card, MiddleElided, discard, label
from .wraprow import WrapRow

__all__ = ["BatchPage"]


class BatchPage(QWidget):
    """Parents, settings, the conditions table, Run / Stop and the log."""

    add_requested = Signal()
    #: the parent folder's path, as listed
    remove_requested = Signal(str)
    run_requested = Signal()
    stop_requested = Signal()
    #: a row's Ignore / Include: its key, and whether it is now ignored
    ignore_toggled = Signal(str, bool)

    def __init__(self, t, *, title: str, intro: str, columns, run_text: str,
                 stop_text: str = "Stop after the movie in hand", ignorable: bool = False,
                 parent=None):
        super().__init__(parent)
        self.t = t
        #: Each row carries Ignore / Include (asked 14 Oct 2026 for the outlines batch: "a
        #: way in the list to skip certain conditions ... just an ignore button"); a row's
        #: ``ignored`` says which, and the window keeps the list.
        self._ignorable = bool(ignorable)
        self._ignore_buttons: list = []
        self.setStyleSheet("background:transparent;")
        self._columns = tuple(columns)
        self._row_keys: list = []
        self._row_widgets: list = []
        self._parent_widgets: list = []
        self._busy = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)
        lay.addWidget(label(title, ty.ui("lg", "semibold"), t.foreground))
        self.intro = label(intro, ty.ui("sm"), t.muted_foreground)
        self.intro.setWordWrap(True)
        lay.addWidget(self.intro)

        parents = Card(t, shadowed=False)
        pl = QVBoxLayout(parents)
        pl.setContentsMargins(14, 12, 14, 12)
        pl.setSpacing(6)
        head = QHBoxLayout()
        head.addWidget(label("Parent folders", ty.ui("sm", "semibold"), t.foreground))
        head.addStretch(1)
        self.add_button = Button(t, "Add parent folder…", size="sm", glyph="folder-open")
        self.add_button.clicked.connect(self.add_requested.emit)
        head.addWidget(self.add_button)
        pl.addLayout(head)
        self._parents_lay = QVBoxLayout()
        self._parents_lay.setSpacing(2)
        pl.addLayout(self._parents_lay)
        self.parents_empty = label("None yet. A parent may hold several condition folders, "
                                   "or be one condition itself.", ty.ui("xs"),
                                   t.muted_foreground)
        self.parents_empty.setWordWrap(True)
        pl.addWidget(self.parents_empty)
        lay.addWidget(parents)

        #: The window's or the subclass's own controls (tracker, Replace, ...). A `WrapRow`
        #: (3 Nov 2026, tester B5): as a `QHBoxLayout` its minimum was the sum of its
        #: controls, which widened the page past a 1366 px screen and cut Add parent
        #: folder, Remove, Movies at once and the Status column at the right edge. Add a
        #: control and its words as one group (`group`), so a pair never splits.
        self.settings = WrapRow(spacing=10, line_spacing=6)
        lay.addLayout(self.settings)
        self.settings_note = label("", ty.ui("xs"), t.muted_foreground)
        self.settings_note.setWordWrap(True)
        lay.addWidget(self.settings_note)

        table = Card(t, shadowed=False)
        self._grid = QGridLayout(table)
        self._grid.setContentsMargins(14, 12, 14, 12)
        self._grid.setHorizontalSpacing(12)
        self._grid.setVerticalSpacing(6)
        self._grid.setColumnStretch(len(self._columns) - 1, 1)
        for col, text in enumerate(self._columns):
            self._grid.addWidget(label(text, ty.ui("xs", "medium"), t.muted_foreground),
                                 0, col)
        lay.addWidget(table)
        self.layout_lines = label("", ty.ui("xs"), t.muted_foreground)
        self.layout_lines.setWordWrap(True)
        self.layout_lines.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.layout_lines)
        self.layout_lines.setVisible(False)

        foot = QHBoxLayout()
        foot.setSpacing(8)
        self.summary = label("", ty.ui("sm"), t.foreground)
        self.summary.setWordWrap(True)
        foot.addWidget(self.summary, 1)
        self.stop_button = Button(t, stop_text, size="sm")
        self.stop_button.clicked.connect(self.stop_requested.emit)
        foot.addWidget(self.stop_button)
        self.stop_button.setVisible(False)
        self.run_button = Button(t, run_text, variant="default", glyph="play")
        self.run_button.clicked.connect(self.run_requested.emit)
        foot.addWidget(self.run_button)
        lay.addLayout(foot)
        self.note = label("", ty.ui("xs", "medium"), t.foreground)
        self.note.setWordWrap(True)
        lay.addWidget(self.note)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(ty.mono("xs"))
        self.log.setMinimumHeight(200)
        self.log.setStyleSheet(
            f"QPlainTextEdit {{ background:{t.card}; color:{t.foreground};"
            f" border:1px solid {t.border}; border-radius:{t.radius_md:g}px;"
            f" padding:6px; }}")
        lay.addWidget(self.log, 1)

    @staticmethod
    def group(*widgets, spacing: int = 6) -> QHBoxLayout:
        """Widgets that stay together on one line of the settings row (a box and its words)."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(spacing)
        for widget in widgets:
            row.addWidget(widget)
        return row

    # -- what the window tells the page --------------------------------------------- #
    def set_parents(self, paths) -> None:
        """The parent folders, one row each with Remove (off while a batch runs)."""
        for widget in self._parent_widgets:
            discard(widget)
        self._parent_widgets = []
        for path in paths:
            row = QWidget()
            row.setStyleSheet("background:transparent;")
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.setSpacing(8)
            # shortened in the middle to the room it has (the whole path on hover): a long
            # path held at its full width widened the page past the screen
            text = MiddleElided(str(path), ty.ui("sm"), self.t.foreground)
            rl.addWidget(text, 1)
            remove = Button(self.t, "Remove", size="sm", variant="ghost", glyph="x")
            remove.clicked.connect(lambda p=str(path): self.remove_requested.emit(p))
            remove.setEnabled(not self._busy)
            rl.addWidget(remove)
            self._parents_lay.addWidget(row)
            self._parent_widgets.append(row)
        self.parents_empty.setVisible(not self._parent_widgets)

    def set_rows(self, rows) -> None:
        """``rows``: ``[{key, cells: [text per column]}]``. Rebuilt when the conditions
        change, relabelled in place otherwise."""
        keys = [r["key"] for r in rows]
        if keys != self._row_keys:
            for widgets in self._row_widgets:
                for widget in widgets:
                    discard(widget)
            for button in self._ignore_buttons:
                discard(button)
            self._row_widgets, self._ignore_buttons = [], []
            for r, row in enumerate(rows, start=1):
                widgets = []
                for col, _title in enumerate(self._columns):
                    cell = label("", ty.ui("sm" if col < 2 else "xs"),
                                 self.t.foreground if col < 2 else self.t.muted_foreground)
                    cell.setWordWrap(col == len(self._columns) - 1)
                    self._grid.addWidget(cell, r, col, alignment=Qt.AlignTop)
                    widgets.append(cell)
                self._row_widgets.append(widgets)
                if self._ignorable:
                    button = Button(self.t, "Ignore", size="sm", variant="ghost")
                    button.clicked.connect(lambda _k=None, key=row["key"]: self._ignore(key))
                    button.setEnabled(not self._busy)
                    self._grid.addWidget(button, r, len(self._columns), alignment=Qt.AlignTop)
                    self._ignore_buttons.append(button)
            self._row_keys = keys
        self._ignored = {row["key"] for row in rows if row.get("ignored")}
        for index, (widgets, row) in enumerate(zip(self._row_widgets, rows)):
            ignored = bool(row.get("ignored"))
            for col, (cell, text) in enumerate(zip(widgets, row["cells"])):
                cell.setText(str(text))
                ink = (self.t.muted_foreground if ignored or col >= 2 else self.t.foreground)
                cell.setStyleSheet(f"color:{ink}; background:transparent;")
            if self._ignorable and index < len(self._ignore_buttons):
                self._ignore_buttons[index].set_text("Include" if ignored else "Ignore")
                self._ignore_buttons[index].setToolTip(
                    "Segment this condition again in the batch" if ignored else
                    "Leave this condition out of the batch (remembered)")

    def _ignore(self, key: str) -> None:
        self.ignore_toggled.emit(key, key not in getattr(self, "_ignored", set()))

    def ignored_keys(self) -> set:
        return set(getattr(self, "_ignored", set()))

    def cell(self, key: str, column: int) -> str:
        """What one cell says (for tests and for reading back)."""
        index = self._row_keys.index(key)
        return self._row_widgets[index][column].text()

    def set_layout_lines(self, lines) -> None:
        self.layout_lines.setText("\n".join(lines))
        self.layout_lines.setVisible(bool(lines))

    def set_summary(self, text: str) -> None:
        self.summary.setText(text)

    def set_note(self, text: str) -> None:
        self.note.setText(text)

    def set_busy(self, busy: bool, *, can_run: bool = True) -> None:
        """While a batch runs: Stop shown, Run, Add and Remove off."""
        self._busy = bool(busy)
        self.stop_button.setVisible(self._busy)
        self.run_button.setEnabled(not self._busy and can_run)
        self.add_button.setEnabled(not self._busy)
        for button in self._ignore_buttons:
            button.setEnabled(not self._busy)
        for row in self._parent_widgets:
            for button in row.findChildren(Button):
                button.setEnabled(not self._busy)

    def append(self, *lines: str) -> None:
        for line in lines:
            self.log.appendPlainText(line)
