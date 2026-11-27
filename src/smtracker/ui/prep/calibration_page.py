"""Calibration & save: one row per condition, its pixel size, frame interval, label,
emission and NA, and the button that writes its `crops.json`.

THIN QT. Cells show `prep_view.calibration_cells`; an edit is reported as typed and
filed by the window through `prep_model.set_calibration`, which refuses an empty
required cell and converts the interval from ms. The status beside each row is
`prep_view.save_state`. Nothing is written until a Save is pressed: `crops.json` is the
single record of what will be tracked.
"""
from __future__ import annotations

from qtpy.QtCore import QEvent, QLocale, Qt, Signal
from qtpy.QtGui import QDoubleValidator
from qtpy.QtWidgets import (
    QCompleter,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from ...prep_view import CALIBRATION_COLUMNS, fluorophore_names
from ..theme import typography as ty
from ..widgets.controls import Button, Segmented
from ..widgets.primitives import Card, Checkbox, Progress, label

__all__ = ["CalibrationPage", "Field", "PROJECTIONS"]

#: What Save writes into `sum/`, ``(key, label)``: the macro's SD projection is the
#: default because it is what every existing `sum/` file contains; ``None`` writes none.
PROJECTIONS = (("sd", "SD projection"), ("sum", "Sum projection"), (None, "No projection"))


class Field(QFrame):
    """One calibration cell: shadcn's Input, committing when focus leaves it."""

    committed = Signal(str)
    #: The field was clicked or tabbed into -- a guide opens the setting it holds.
    focused = Signal()

    def __init__(self, t, *, numeric: bool, unit: str, width: int, parent=None):
        super().__init__(parent)
        self.setObjectName("calfield")
        self.setStyleSheet(
            f"QFrame#calfield {{ background:{t.card}; border:1px solid {t.input};"
            f" border-radius:{t.radius_md:g}px; }}")
        self.setFixedSize(width, t.control_height)
        self.edit = QLineEdit("")
        self.edit.setFrame(False)
        self.edit.setFont(ty.ui("sm", tabular=True))
        self.edit.setStyleSheet(
            f"QLineEdit {{ background:transparent; color:{t.foreground};"
            f" border:none; padding:0px; }}")
        if numeric:
            validator = QDoubleValidator(0.0, 1e7, 6, self.edit)
            validator.setNotation(QDoubleValidator.StandardNotation)
            validator.setLocale(QLocale.c())
            self.edit.setValidator(validator)
        #: the text last committed or shown, so an edit still pending can be told (`pending`)
        self._last = ""
        self.edit.editingFinished.connect(self.commit)
        self.edit.installEventFilter(self)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 0, 10, 0)
        lay.setSpacing(6)
        lay.addWidget(self.edit, 1)
        if unit:
            lay.addWidget(label(unit, ty.ui("sm"), t.muted_foreground))

    def set_text(self, text: str) -> None:
        if not self.edit.hasFocus():
            self.edit.setText(text)
            self._last = text

    def commit(self, text: str | None = None) -> None:
        """Send what the field holds (or ``text``, a suggestion picked from the list)."""
        if text is not None:
            self.edit.setText(text)
        self._last = self.edit.text()
        self.committed.emit(self._last)

    def pending(self) -> bool:
        """Typed or picked but not yet sent: the field is still being edited."""
        return self.edit.text() != self._last

    def eventFilter(self, obj, event):  # noqa: N802 - Qt's name
        if obj is self.edit and event.type() == QEvent.FocusIn:
            self.focused.emit()
        return False


class CalibrationPage(QWidget):
    """The grid. Signals carry requests; the window answers them."""

    #: condition, field key, the text as typed
    edited = Signal(str, str, str)
    save_requested = Signal(str)
    save_all_requested = Signal()
    copy_requested = Signal(str)
    pixel_helper_requested = Signal()
    export_requested = Signal()
    #: The warning's button: install the packages this install lacks
    #: (`optional_deps`). A LIST, so the window never re-derives what was offered.
    install_requested = Signal(list)
    export_stop_requested = Signal()
    export_replace_changed = Signal(bool)

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.setStyleSheet("background:transparent;")
        self._rows: dict = {}
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)
        lay.addWidget(label("Calibration & save", ty.ui("lg", "semibold"), t.foreground))
        intro = label(
            "Pixel size and frame interval have no default; every D scales with them. "
            "The frame interval is filled from the acquisition where none was recorded. "
            "The pixel size is not read from a .nd2, which can report the wrong value.",
            ty.ui("sm"), t.muted_foreground)
        intro.setWordWrap(True)
        lay.addWidget(intro)

        self._card = Card(t, shadowed=False)
        self._grid = QGridLayout(self._card)
        self._grid.setContentsMargins(14, 12, 14, 12)
        self._grid.setHorizontalSpacing(10)
        self._grid.setVerticalSpacing(8)
        head_font = ty.ui("xs", "medium")
        self._grid.addWidget(label("Condition", head_font, t.muted_foreground), 0, 0)
        for col, (_key, heading, _unit) in enumerate(CALIBRATION_COLUMNS, start=1):
            self._grid.addWidget(label(heading, head_font, t.muted_foreground), 0, col)
        self._grid.addWidget(label("Status", head_font, t.muted_foreground), 0,
                             len(CALIBRATION_COLUMNS) + 1)
        lay.addWidget(self._card)

        foot = QHBoxLayout()
        foot.setSpacing(10)
        self._copy = Button(t, "Copy first row to all", size="sm")
        self._copy.clicked.connect(
            lambda: self.commit_pending() or (self._rows and self.copy_requested.emit(next(iter(self._rows)))))
        foot.addWidget(self._copy)
        self._pixel = Button(t, "Pixel size from the camera…", size="sm")
        self._pixel.setToolTip("Camera pixel size divided by objective magnification times "
                               "tube lens factor. The result is shown before you apply it.")
        self._pixel.clicked.connect(self.pixel_helper_requested.emit)
        foot.addWidget(self._pixel)
        foot.addStretch(1)
        foot.addWidget(label("Save writes", ty.ui("sm", "medium"), t.muted_foreground))
        self.projection = Segmented(t, [lab for _k, lab in PROJECTIONS], current=0,
                                    size="sm")
        foot.addWidget(self.projection)
        self._save_all = Button(t, "Save all", variant="default")
        self._save_all.clicked.connect(lambda: self.commit_pending() or self.save_all_requested.emit())
        foot.addWidget(self._save_all)
        lay.addLayout(foot)
        lay.addWidget(self._missing_card(t))
        self.note = label("", ty.ui("xs"), t.muted_foreground)
        self.note.setWordWrap(True)
        lay.addWidget(self.note)
        lay.addWidget(self._export_card(t))
        lay.addStretch(1)

    # -- an optional reader this install lacks ----------------------------------- #
    def _missing_card(self, t):
        """The warning, and the offer to run the install.

        RULED (18 Oct 2026): a git clone is WARNED and OFFERED the install; the packaged
        build must simply work. It sits ABOVE the note and below Save, because it is the
        reason a Save would write no projection -- a warning under the outcome is read
        after the outcome.

        Hidden until there IS something missing: a permanently visible row about packages
        is furniture, and this project's own rule is that a control which cannot act is
        worse than an absent one.
        """
        card = Card(t, shadowed=False)
        cl = QVBoxLayout(card)
        cl.setContentsMargins(14, 12, 14, 12)
        cl.setSpacing(8)
        self.missing_words = label("", ty.ui("xs"), t.foreground)
        self.missing_words.setWordWrap(True)
        cl.addWidget(self.missing_words)
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addStretch(1)
        self.missing_log = label("", ty.ui("xs", tabular=True), t.muted_foreground)
        self.missing_log.setWordWrap(True)
        self._install = Button(t, "Install", variant="default")
        self._install.clicked.connect(
            lambda: self.install_requested.emit(list(self._missing)))
        row.addWidget(self._install)
        cl.addLayout(row)
        cl.addWidget(self.missing_log)
        self._missing: list = []
        self._missing_box = card
        card.setVisible(False)
        return card

    def set_missing(self, packages: list, words: str, *, can_install: bool = True,
                    busy: bool = False) -> None:
        """What is missing (`optional_deps.missing_for_movies`) and the words for it.

        ``can_install`` False hides the button rather than disabling it: in a frozen
        build there is nothing to press and the words say to rebuild instead.
        """
        self._missing = list(packages)
        self.missing_words.setText(words)
        self._missing_box.setVisible(bool(packages))
        self._install.setVisible(bool(packages) and can_install)
        self._install.setEnabled(not busy)
        self._install.set_text("Installing…" if busy else "Install")

    def set_install_log(self, text: str) -> None:
        self.missing_log.setText(text)

    # -- cropped stacks for u-track ---------------------------------------------- #
    def _export_card(self, t):
        card = Card(t, shadowed=False)
        cl = QVBoxLayout(card)
        cl.setContentsMargins(14, 12, 14, 12)
        cl.setSpacing(8)
        cl.addWidget(label("Cropped stacks for u-track", ty.ui("sm", "semibold"),
                           t.foreground))
        what = label(
            "u-track cannot read a .nd2 or .stk. Export cuts each movie's saved range into "
            "tif/<movie>.tif and writes the projection to sum/. TrackMate does not need "
            "this. Existing stacks are replaced only if you tick Replace.",
            ty.ui("xs"), t.muted_foreground)
        what.setWordWrap(True)
        cl.addWidget(what)
        self.export_lines = label("", ty.ui("xs", tabular=True), t.foreground)
        self.export_lines.setWordWrap(True)
        cl.addWidget(self.export_lines)
        row = QHBoxLayout()
        row.setSpacing(8)
        self.export_replace = Checkbox(t, checked=False)
        self.export_replace.toggled.connect(self.export_replace_changed.emit)
        row.addWidget(self.export_replace)
        row.addWidget(label("Replace stacks already in tif/", ty.ui("sm"), t.foreground))
        row.addStretch(1)
        self.export_bar = Progress(t, 0.0, width=160)
        row.addWidget(self.export_bar)
        self.export_bar.setVisible(False)
        self._export_stop = Button(t, "Stop after the movies in hand", size="sm")
        self._export_stop.clicked.connect(self.export_stop_requested.emit)
        row.addWidget(self._export_stop)
        self._export_stop.setVisible(False)
        self._export = Button(t, "Export stacks", variant="default")
        self._export.clicked.connect(lambda: self.commit_pending() or self.export_requested.emit())
        row.addWidget(self._export)
        cl.addLayout(row)
        self.export_note = label("", ty.ui("xs"), t.muted_foreground)
        self.export_note.setWordWrap(True)
        cl.addWidget(self.export_note)
        return card

    def set_export(self, rows: list, *, busy: bool, progress=None) -> None:
        """``rows``: `prep_view.export_rows`; ``progress`` ``(done, total)`` while busy."""
        self.export_lines.setText("\n".join(f"{r['name']}: {r['status']}" for r in rows))
        self._export.setEnabled(not busy and any(r["ready"] for r in rows))
        self._export_stop.setVisible(busy)
        self.export_bar.setVisible(busy)
        if busy and progress:
            done, total = progress
            self.export_bar.set_value(done / total if total else 0.0)

    def set_export_note(self, text: str) -> None:
        self.export_note.setText(text)

    def projection_kind(self):
        return PROJECTIONS[self.projection.current()][0]

    def set_rows(self, rows: list) -> None:
        """``rows``: ``[{"name", "cells": {key: text}, "status", "ready"}]``. Built once
        per condition and updated in place afterwards."""
        for row in rows:
            name = row["name"]
            widgets = self._rows.get(name)
            if widgets is None:
                widgets = self._add_row(name)
            for key, field in widgets["fields"].items():
                field.set_text(row["cells"].get(key, ""))
            widgets["status"].setText(row["status"])
            widgets["save"].setEnabled(bool(row["ready"]))

    def _add_row(self, name: str) -> dict:
        t = self.t
        r = len(self._rows) + 1
        self._grid.addWidget(label(name, ty.ui("sm", "medium"), t.foreground), r, 0)
        fields = {}
        for col, (key, _heading, unit) in enumerate(CALIBRATION_COLUMNS, start=1):
            numeric = key != "label"
            field = Field(t, numeric=numeric, unit=unit,
                           width=150 if key == "label" else 112)
            if key == "label":
                completer = QCompleter(list(fluorophore_names()), field.edit)
                completer.setCaseSensitivity(Qt.CaseInsensitive)
                field.edit.setCompleter(completer)
                # PICKING A SUGGESTION IS A CHOICE (23 Oct 2026: "mNG emission not coming when i
                # click on mneongreen ... and save"): it commits at once, the emission follows
                completer.activated.connect(lambda text, f=field: f.commit(text))
            field.committed.connect(lambda text, k=key, n=name: self.edited.emit(n, k, text))
            self._grid.addWidget(field, r, col)
            fields[key] = field
        status = label("", ty.ui("xs"), t.muted_foreground)
        self._grid.addWidget(status, r, len(CALIBRATION_COLUMNS) + 1)
        save = Button(t, "Save", size="sm")
        save.clicked.connect(lambda n=name: self.commit_pending() or self.save_requested.emit(n))
        self._grid.addWidget(save, r, len(CALIBRATION_COLUMNS) + 2)
        self._rows[name] = {"fields": fields, "status": status, "save": save}
        return self._rows[name]

    def commit_pending(self) -> None:
        """Send every field still being edited, BEFORE a button acts on the calibration.
        The buttons take no keyboard focus, so pressing Save right after typing or picking
        a fluorophore never ended the edit, and the row was saved without it."""
        for widgets in list(self._rows.values()):
            for field in widgets["fields"].values():
                if field.pending():
                    field.commit()

    def set_note(self, text: str) -> None:
        self.note.setText(text)
