"""The PALETTE MAKER (6 Nov 2026, asked: "we should also add a color palette maker"). THIN QT: every check, every
generated colour and the store are `smtracker.palettes`'; this edits a list of colours and shows the checks.

Opened from both palette menus (the Confirm screen's and the workspace Conditions tool's, "Make a palette..."). Start
from any palette -- a built-in one is copied, one of the user's is edited -- then add a colour (the system colour
picker), add a DISTINCT one (the colour farthest from every colour already in, `palettes.distinct`), change, move or
remove any, order them for the checks (`palettes.best_order`), and read the checks as they change: the bar every
built-in palette clears (its first three and four colours apart for normal and red-green vision, every colour 3 : 1
on the page) and the closest pair of all (`palettes.check_lines`). A palette below the bar can still be saved; the
checks say so. Save keeps it with the user's settings and in every palette list (`saved(key)`)."""
from __future__ import annotations

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QFrame, QHBoxLayout, QLineEdit, QVBoxLayout, QWidget

from ..theme import typography as ty
from .controls import Button
from .dialogs import CloseButton, OverlayDialog
from .primitives import discard, fit_label, label

__all__ = ["PaletteMaker"]

FLAG_INK = "#b42318"
#: The colour list's height before it scrolls (about twelve colours).
ROWS_MAX_PX = 360
#: EXPLICIT LINES, NOT WORD WRAP (`dwell_dialog`).
NOTE_LINES = ("Colours in the order conditions take them. The first three and four are what the checks measure:",
              "put the colours that must never be confused there.")


class _Swatch(QFrame):
    """A colour, clickable to change it."""

    clicked = Signal()

    def __init__(self, t, colour: str, parent=None):
        super().__init__(parent)
        self.setFixedSize(22, 22)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(f"QFrame {{ background:{colour}; border:1px solid {t.border}; border-radius:4px; }}")
        self.setToolTip("Change this colour")

    def mouseReleaseEvent(self, e):          # noqa: N802 - Qt's name
        if e.button() == Qt.LeftButton:
            self.clicked.emit()


class PaletteMaker(OverlayDialog):
    """Make a palette, or edit one of the user's. ``saved(key)`` after Save, ``deleted(key)`` after Delete."""

    saved = Signal(str)
    deleted = Signal(str)

    def __init__(self, t, *, start: str | None = None, content=None, parent=None):
        super().__init__(t, content, parent=parent)
        from ...palettes import ensure_registered
        from ..theme import hcl

        ensure_registered()
        self.t = t
        self._card.setMinimumWidth(560)
        self.colours: list = []
        self.editing: str | None = None
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        self.title = label("Make a palette", ty.ui("base", "semibold"), t.foreground)
        head.addWidget(self.title)
        head.addStretch(1)
        close = CloseButton(t)
        close.clicked.connect(self._dismiss)
        head.addWidget(close)
        self.card_lay.addLayout(head)
        top = QHBoxLayout()
        top.setSpacing(8)
        top.addWidget(label("Start from", ty.ui("sm"), t.muted_foreground))
        # the workspace's drop-down, a chevron at its end (a plain combo box read as a text field)
        from ..workspace.subpage_rows import PageChoice

        self.start = PageChoice(t)
        for key, palette in hcl.PALETTES.items():
            self.start.addItem(palette.label, key)
        self.start.activated.connect(lambda i: self.load(self.start.itemData(i)))
        top.addWidget(self.start, 1)
        top.addWidget(label("Name", ty.ui("sm"), t.muted_foreground))
        self.name = QLineEdit("")
        self.name.setFont(ty.ui("sm"))
        self.name.setPlaceholderText("A name for the palette")
        self.name.setStyleSheet(f"QLineEdit {{ background:{t.card}; color:{t.foreground};"
                                f" border:1px solid {t.input}; border-radius:{t.radius_md:g}px; padding:4px 8px; }}")
        self.name.textChanged.connect(lambda _t: self._sync_buttons())
        top.addWidget(self.name, 2)
        self.card_lay.addLayout(top)
        self.card_lay.addWidget(label("\n".join(NOTE_LINES), ty.ui("xs"), t.muted_foreground))
        self.rows_box = QWidget()
        self.rows_box.setStyleSheet("background:transparent;")
        self.rows_lay = QVBoxLayout(self.rows_box)
        self.rows_lay.setContentsMargins(0, 0, 0, 0)
        self.rows_lay.setSpacing(4)
        # A LONG PALETTE SCROLLS (twenty colours made the dialog the window's height): the list's own height up to
        # `ROWS_MAX_PX`
        from qtpy.QtWidgets import QScrollArea

        self.rows_scroll = QScrollArea()
        self.rows_scroll.setWidget(self.rows_box)
        self.rows_scroll.setWidgetResizable(True)
        self.rows_scroll.setFrameShape(QFrame.NoFrame)
        self.rows_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.rows_scroll.setStyleSheet("QScrollArea { background:transparent; border:none; }")
        self.rows_scroll.viewport().setStyleSheet("background:transparent;")
        self.card_lay.addWidget(self.rows_scroll)
        self.row_widgets: list = []
        actions = QHBoxLayout()
        actions.setSpacing(8)
        self.add = Button(t, "Add colour…", variant="outline", size="sm", glyph="plus")
        self.add.clicked.connect(self._ask_add)
        actions.addWidget(self.add)
        self.add_distinct = Button(t, "Add a distinct colour", variant="outline", size="sm", glyph="sparkles")
        self.add_distinct.setToolTip("The colour farthest from every colour already in the palette")
        self.add_distinct.clicked.connect(self.add_distinct_colour)
        actions.addWidget(self.add_distinct)
        self.reorder = Button(t, "Order for the checks", variant="outline", size="sm", glyph="shuffle")
        self.reorder.setToolTip("The order whose first three and four colours are farthest apart")
        self.reorder.clicked.connect(self.order_for_checks)
        actions.addWidget(self.reorder)
        actions.addStretch(1)
        self.card_lay.addLayout(actions)
        self.checks_box = QWidget()
        self.checks_box.setStyleSheet("background:transparent;")
        self.checks_lay = QVBoxLayout(self.checks_box)
        self.checks_lay.setContentsMargins(0, 4, 0, 0)
        self.checks_lay.setSpacing(2)
        self.card_lay.addWidget(self.checks_box)
        self.check_labels: list = []
        self.status = fit_label("", ty.ui("sm"), FLAG_INK)
        self.card_lay.addWidget(self.status)
        footer = QHBoxLayout()
        self.delete = Button(t, "Delete", variant="ghost", size="sm", glyph="trash")
        self.delete.clicked.connect(self.delete_palette)
        footer.addWidget(self.delete)
        footer.addStretch(1)
        cancel = Button(t, "Cancel", variant="outline")
        cancel.clicked.connect(self._dismiss)
        footer.addWidget(cancel)
        self.save = Button(t, "Save", variant="default", glyph="save")
        self.save.clicked.connect(self.save_palette)
        footer.addWidget(self.save)
        self.card_lay.addLayout(footer)
        self.load(start if start in hcl.PALETTES else next(iter(hcl.PALETTES)))

    # -- the list ------------------------------------------------------------------------------------------------- #
    def load(self, key: str) -> None:
        """Start from palette ``key``: its colours as declared; one of the user's is edited (Save replaces it)."""
        from ...palettes import is_user
        from ..theme import hcl

        palette = hcl.PALETTES[key]
        self.colours = [palette.colors[k].lower() for k in palette.order]
        self.editing = key if is_user(key) else None
        self.title.setText(f"Edit “{palette.label}”" if self.editing else "Make a palette")
        self.name.setText(palette.label if self.editing else f"{palette.label}, my version")
        index = self.start.findData(key)
        if index >= 0:
            self.start.setCurrentIndex(index)
        self._rebuild()

    def _rebuild(self) -> None:
        for w in self.row_widgets:
            discard(w)
        self.row_widgets = []
        for i, colour in enumerate(self.colours):
            row = QWidget()
            row.setStyleSheet("background:transparent;")
            lay = QHBoxLayout(row)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(6)
            lay.addWidget(label(f"{i + 1}", ty.ui("xs", tabular=True), self.t.muted_foreground))
            swatch = _Swatch(self.t, colour)
            swatch.clicked.connect(lambda k=i: self._ask_change(k))
            lay.addWidget(swatch)
            lay.addWidget(label(colour, ty.ui("sm", tabular=True), self.t.foreground), 1)
            for glyph, step, tip in (("chevron-up", -1, "Earlier"), ("chevron-down", 1, "Later")):
                button = Button(self.t, "", variant="ghost", size="sm", glyph=glyph)
                button.setToolTip(tip)
                button.setEnabled(0 <= i + step < len(self.colours))
                button.clicked.connect(lambda k=i, s=step: self.move(k, s))
                lay.addWidget(button)
            remove = Button(self.t, "", variant="ghost", size="sm", glyph="x")
            remove.setToolTip("Remove this colour")
            remove.clicked.connect(lambda k=i: self.remove(k))
            lay.addWidget(remove)
            self.rows_lay.addWidget(row)
            self.row_widgets.append(row)
        self.rows_box.adjustSize()
        self.rows_scroll.setFixedHeight(min(self.rows_box.sizeHint().height() + 2, ROWS_MAX_PX))
        self._sync_checks()

    def _sync_checks(self) -> None:
        from ...palettes import check_lines

        for w in self.check_labels:
            discard(w)
        self.check_labels = []
        for line in check_lines(self.colours):
            flagged = " below " in f" {line} " or "hard to tell apart" in line or line.endswith(": no")
            w = fit_label(line, ty.ui("xs"), FLAG_INK if flagged else self.t.muted_foreground)
            self.checks_lay.addWidget(w)
            self.check_labels.append(w)
        self.status.setText("")
        self._sync_buttons()

    def _sync_buttons(self) -> None:
        self.save.setEnabled(len(self.colours) >= 2 and bool(self.name.text().strip()))
        self.delete.setVisible(self.editing is not None)

    # -- editing (callable without the colour picker) --------------------------------------------------------------- #
    def add_colour(self, colour: str) -> None:
        colour = QColor(colour).name().lower()
        if colour in self.colours:
            self.status.setText(f"{colour} is in the palette already")
            return
        self.colours.append(colour)
        self._rebuild()

    def set_colour(self, index: int, colour: str) -> None:
        colour = QColor(colour).name().lower()
        if colour in self.colours[:index] + self.colours[index + 1:]:
            self.status.setText(f"{colour} is in the palette already")
            return
        self.colours[index] = colour
        self._rebuild()

    def move(self, index: int, step: int) -> None:
        j = index + int(step)
        if 0 <= index < len(self.colours) and 0 <= j < len(self.colours):
            self.colours[index], self.colours[j] = self.colours[j], self.colours[index]
            self._rebuild()

    def remove(self, index: int) -> None:
        if 0 <= index < len(self.colours):
            del self.colours[index]
            self._rebuild()

    def add_distinct_colour(self) -> None:
        from ...palettes import distinct

        made = distinct(1, taken=self.colours)
        if made:
            self.colours.append(made[0])
            self._rebuild()

    def order_for_checks(self) -> None:
        from ...palettes import best_order

        self.colours = best_order(self.colours)
        self._rebuild()

    def save_palette(self) -> str | None:
        from ...palettes import save_user_palette

        try:
            key = save_user_palette(self.name.text(), self.colours, key=self.editing)
        except ValueError as exc:
            self.status.setText(str(exc))
            return None
        self.saved.emit(key)
        self._dismiss()
        return key

    def delete_palette(self) -> None:
        from ...palettes import delete_user_palette

        if self.editing is None:
            return
        key = self.editing
        delete_user_palette(key)
        self.deleted.emit(key)
        self._dismiss()

    # -- the system colour picker ------------------------------------------------------------------------------------ #
    def _pick(self, initial: str, title: str) -> str | None:
        from qtpy.QtWidgets import QColorDialog

        chosen = QColorDialog.getColor(QColor(initial), self.window(), title)
        return chosen.name().lower() if chosen.isValid() else None

    def _ask_add(self) -> None:
        colour = self._pick(self.colours[-1] if self.colours else "#3f90da", "A colour to add")
        if colour:
            self.add_colour(colour)

    def _ask_change(self, index: int) -> None:
        colour = self._pick(self.colours[index], f"Colour {index + 1}")
        if colour:
            self.set_colour(index, colour)
