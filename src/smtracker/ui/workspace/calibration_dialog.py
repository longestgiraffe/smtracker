"""The workspace header menu's "Per-spot calibration..." (6 Nov 2026; asked: "i wonder where to do the calibraion ?",
"ja build it", then "people should choose (internal calibration this folder), own folder. measure on current data set
can be tricky?" and "yes to all"). THIN QT: the choices, the library and every word are `calibrations` /
`calibration_view`'s; this shows them and asks. The window runs the measurement off the GUI thread and keeps the
choice (`WorkspaceWindow.open_calibration`).

Two pages. CHOOSE: which calibration every Per spot Run of this project uses -- Built-in (the tracker's own, from the
stationary synthetic set), This project's file (where a `precision_calibration.json` is found), or one from the
library -- each with its curve; a library entry can be removed. MAKE ONE: from a SEPARATE stationary measurement only
(a folder of tracked movies: fixed cells or mNG on glass, never the data being analysed), Measure, see the curve and
the checks, Add to library under a name (and use it for this project), or also save it into a project folder."""
from __future__ import annotations

from datetime import date
from pathlib import Path

from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtWidgets import QHBoxLayout, QStackedWidget, QVBoxLayout, QWidget

from ..theme import typography as ty
from ..widgets.controls import Button
from ..widgets.dialogs import CloseButton, OverlayDialog
from ..widgets.primitives import Checkbox, MiddleElided, discard, fit_label, label
from .exclusions_dialog import _Field
from .jumps_chart import _Table
from .toolbar import rem

__all__ = ["CalibrationDialog"]

#: EXPLICIT LINES, NOT WORD WRAP (a wrapped note reports less height than it takes; `dwell_dialog`).
CHOOSE_LINES = ("Every Per spot Run of this project (SQD and its pages, HMM, Apparent Diffusion, the maps, Kinetics)",
                "uses the calibration ticked here, and its caption names it.")
MAKE_LINES = ("From a separate stationary measurement, not the data being analysed: fixed cells expressing the",
              "fusion, or mNG stuck to glass, imaged and tracked with u-track at the real data's settings. Every",
              "track is read; no cell outlines are needed.")
FLAG_INK = "#b42318"


class _Choice(QWidget):
    """One row of the choice: a tick that acts as a radio, the title, the curve and where it comes from."""

    picked = Signal()
    remove = Signal()

    def __init__(self, t, row: dict, parent=None):
        super().__init__(parent)
        self.row = row
        self.setStyleSheet("background:transparent;")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(rem(0.5))
        self.tick = Checkbox(t, checked=bool(row.get("selected")))
        self.tick.setEnabled(bool(row.get("available")))
        self.tick.toggled.connect(lambda on: self.picked.emit() if on else self.tick.set_checked(True, animate=False))
        lay.addWidget(self.tick, 0, Qt.AlignTop)
        words = QVBoxLayout()
        words.setSpacing(0)
        words.addWidget(label(str(row["title"]), ty.ui("sm", "medium"), t.foreground))
        words.addWidget(fit_label(str(row.get("detail", "")), ty.ui("xs"),
                                  t.muted_foreground if row.get("available") else FLAG_INK))
        lay.addLayout(words, 1)
        if row.get("kind") == "library" and row.get("available"):
            self.remove_button = Button(t, "Remove", variant="ghost", size="sm")
            self.remove_button.clicked.connect(self.remove.emit)
            lay.addWidget(self.remove_button, 0, Qt.AlignTop)


class CalibrationDialog(OverlayDialog):
    """Choose the project's per-spot calibration, or make one from a stationary measurement."""

    #: ``{"kind": "builtin" | "project" | "library", "id"?}``: the project's calibration from now on
    choice_made = Signal(object)
    #: a library entry's id, to be removed
    remove_requested = Signal(str)
    #: the folder of tracked stationary movies is to be chosen (the window's folder picker, `set_folder`)
    folder_wanted = Signal()
    #: measure the folder chosen
    measure_requested = Signal(str)
    #: ``(name, use for this project)``: put the measurement in the library
    add_requested = Signal(str, bool)
    #: also save the measurement into a project folder: ``replace`` an existing one
    save_requested = Signal(bool)

    def __init__(self, t, *, content=None, parent=None):
        super().__init__(t, content, parent=parent)
        self.t = t
        self.payload: dict | None = None
        self.folder: str | None = None
        self.choices: list = []
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.addWidget(label("Per-spot calibration", ty.ui("base", "semibold"), t.foreground))
        head.addStretch(1)
        close = CloseButton(t)
        close.clicked.connect(self._dismiss)
        head.addWidget(close)
        self.card_lay.addLayout(head)
        self.pages = QStackedWidget()
        self.pages.setStyleSheet("background:transparent;")
        self.pages.addWidget(self._choose_page(t))
        self.pages.addWidget(self._make_page(t))
        self.card_lay.addWidget(self.pages)
        footer = QHBoxLayout()
        footer.addStretch(1)
        done = Button(t, "Close", variant="outline")
        done.clicked.connect(self._dismiss)
        footer.addWidget(done)
        self.card_lay.addLayout(footer)
        self._sync_buttons()

    # -- the pages ---------------------------------------------------------------------------------------------- #
    def _choose_page(self, t) -> QWidget:
        page = QWidget()
        page.setStyleSheet("background:transparent;")
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(rem(0.5))
        lay.addWidget(label("\n".join(CHOOSE_LINES), ty.ui("xs"), t.muted_foreground))
        self.rows_box = QWidget()
        self.rows_box.setStyleSheet("background:transparent;")
        self.rows_lay = QVBoxLayout(self.rows_box)
        self.rows_lay.setContentsMargins(0, 0, 0, 0)
        self.rows_lay.setSpacing(rem(0.5))
        lay.addWidget(self.rows_box)
        self.choose_status = fit_label("", ty.ui("sm"), t.muted_foreground)
        lay.addWidget(self.choose_status)
        self.make = Button(t, "Make one from a stationary measurement…", variant="outline", size="sm",
                           glyph="crosshair")
        self.make.clicked.connect(self.show_make)
        lay.addWidget(self.make, 0, Qt.AlignLeft)
        return page

    def _make_page(self, t) -> QWidget:
        page = QWidget()
        page.setStyleSheet("background:transparent;")
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(rem(0.5))
        lay.addWidget(label("\n".join(MAKE_LINES), ty.ui("xs"), t.muted_foreground))
        folder = QHBoxLayout()
        folder.setSpacing(rem(0.5))
        self.choose = Button(t, "Choose folder…", variant="outline", size="sm", glyph="folder-open")
        self.choose.clicked.connect(self.folder_wanted.emit)
        folder.addWidget(self.choose, 0, Qt.AlignVCenter)
        self.folder_label = MiddleElided("No folder chosen", ty.ui("sm"), t.muted_foreground)
        folder.addWidget(self.folder_label, 1)
        lay.addLayout(folder)
        measure_row = QHBoxLayout()
        self.measure = Button(t, "Measure", variant="default", size="sm")
        self.measure.clicked.connect(lambda: self.folder and self.measure_requested.emit(self.folder))
        measure_row.addWidget(self.measure)
        # wrapped to the room they have (a path or a refusal as one line would set the dialog's minimum width)
        self.status = fit_label("", ty.ui("sm"), t.muted_foreground)
        measure_row.addWidget(self.status, 1)
        lay.addLayout(measure_row)
        self.result = label("", ty.ui("sm"), t.foreground)
        self.result.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.result)
        self.error = fit_label("", ty.ui("sm"), FLAG_INK)
        lay.addWidget(self.error)
        self.table = _Table(t)
        lay.addWidget(self.table)
        # INTO THE LIBRARY, under a name; used for this project when ticked
        self.keep_box = QWidget()
        self.keep_box.setStyleSheet("background:transparent;")
        keep = QHBoxLayout(self.keep_box)
        keep.setContentsMargins(0, 0, 0, 0)
        keep.setSpacing(rem(0.5))
        self.name = _Field(t, "A name: microscope, tracker, sample")
        keep.addWidget(self.name, 1)
        self.use = Checkbox(t, checked=True)
        keep.addWidget(self.use, 0, Qt.AlignVCenter)
        keep.addWidget(label("Use it for this project", ty.ui("sm"), t.foreground), 0, Qt.AlignVCenter)
        self.add = Button(t, "Add to library", variant="default", size="sm", glyph="save")
        self.add.clicked.connect(lambda: self.add_requested.emit(self.name.text(), bool(self.use.checked)))
        keep.addWidget(self.add, 0, Qt.AlignVCenter)
        lay.addWidget(self.keep_box)
        # ALSO INTO A PROJECT FOLDER (only when asked), an existing file replaced only on Replace
        self.replace_box = QWidget()
        self.replace_box.setStyleSheet("background:transparent;")
        rl = QHBoxLayout(self.replace_box)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(rem(0.5))
        self.replace_note = fit_label("", ty.ui("xs"), t.foreground)
        rl.addWidget(self.replace_note, 1)
        self.keep_file = Button(t, "Keep it", variant="ghost", size="sm")
        self.keep_file.clicked.connect(lambda: self.replace_box.hide())
        rl.addWidget(self.keep_file, 0, Qt.AlignVCenter)
        self.replace = Button(t, "Replace", variant="default", size="sm")
        self.replace.clicked.connect(lambda: self.save_requested.emit(True))
        rl.addWidget(self.replace, 0, Qt.AlignVCenter)
        lay.addWidget(self.replace_box)
        self.replace_box.hide()
        bottom = QHBoxLayout()
        self.back = Button(t, "Back to the choice", variant="ghost", size="sm", glyph="chevron-left")
        self.back.clicked.connect(self.show_choose)
        bottom.addWidget(self.back)
        bottom.addStretch(1)
        self.save = Button(t, "Also save into a project folder…", variant="outline", size="sm")
        self.save.clicked.connect(lambda: self.save_requested.emit(False))
        bottom.addWidget(self.save)
        lay.addLayout(bottom)
        for w in (self.result, self.error, self.table, self.keep_box):
            w.hide()
        return page

    def show_choose(self) -> None:
        self.pages.setCurrentIndex(0)

    def show_make(self) -> None:
        self.pages.setCurrentIndex(1)

    # -- the choice --------------------------------------------------------------------------------------------- #
    def set_options(self, rows) -> None:
        """The rows of the choice (`calibrations.options`), the one in force ticked."""
        for row in self.choices:
            discard(row)
        self.choices = []
        for row in rows:
            choice = _Choice(self.t, row)
            # after the tick's own click is handled: the window redraws these rows, that tick among them
            choice.picked.connect(lambda r=row: QTimer.singleShot(0, lambda: self.choice_made.emit(
                {"kind": r["kind"], **({"id": r["id"]} if r.get("id") else {})})))
            choice.remove.connect(lambda r=row: self.remove_requested.emit(str(r["id"])))
            self.rows_lay.addWidget(choice)
            self.choices.append(choice)

    def set_choose_status(self, text: str) -> None:
        self.choose_status.setText(text)

    # -- making one ---------------------------------------------------------------------------------------------- #
    def set_folder(self, folder: str | None) -> None:
        if folder:
            self.folder = str(folder)
            self.folder_label.set_full_text(self.folder)
            if not self.name.text():
                self.name.set_text(f"{Path(self.folder).name}, {date.today().isoformat()}")
        self._sync_buttons()

    def _sync_buttons(self) -> None:
        busy = self.status.text().startswith("Measuring")
        self.measure.setEnabled(bool(self.folder) and not busy)
        ready = bool(self.payload and self.payload.get("ok")) and not busy
        self.add.setEnabled(ready)
        self.save.setEnabled(ready)

    def set_busy(self, text: str) -> None:
        self.status.setText(text)
        self._sync_buttons()

    def set_result(self, payload: dict | None, error: str | None = None) -> None:
        """A measurement (`calibration_view.measure`'s payload), or the error that stopped it."""
        from ...calibration_view import decile_table, summary_lines

        self.payload = payload if payload and payload.get("ok") else None
        self.status.setText("")
        self.replace_box.hide()
        if self.payload is None:
            self.error.setText(error or "\n".join(summary_lines(payload or {"ok": False,
                                                                             "error": "nothing was measured"})))
            self.error.show()
            for w in (self.result, self.table, self.keep_box):
                w.hide()
        else:
            self.result.setText("\n".join(summary_lines(payload)))
            self.table.set_table(*decile_table(payload))
            self.error.hide()
            for w in (self.result, self.table, self.keep_box):
                w.show()
        self._sync_buttons()

    def ask_replace(self, folder: str, source: str) -> None:
        """``folder`` holds a calibration already (``source``): Replace or Keep it."""
        self.replace_note.setText(f"{folder} already has a calibration ({source}). Replace it?")
        self.replace_box.show()

    def saved(self, path: str) -> None:
        self.replace_box.hide()
        self.status.setText(f"Saved: {path}")
