"""Segment: each condition's model and the images it reads, then the run, watched.

THIN QT. The rules are `segmentation.outline_model` (the folder follows the model, an
unresolved folder stays unset, a run never overwrites outlines unasked) and so are the
words (`readiness`, `progress_words`, `field_line`); this page shows rows and carries
requests, and the window runs the job.

NEVER LOOKS STUCK: the first field also loads the model, which takes seconds to a
minute, so the progress line says that before the first field lands, then counts fields,
times them and estimates what is left; every field is a line in the log as it finishes.
"""
from __future__ import annotations

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..theme import typography as ty
from ..widgets.controls import Button
from ..widgets.primitives import Card, Checkbox, Progress, label
from ..widgets.settings import InfoLink

__all__ = ["SegmentPage", "CHOOSE_FOLDER", "BROWSE_FOLDER"]

CHOOSE_FOLDER = "(set the folder)"
BROWSE_FOLDER = "choose another folder…"


def _combo_css(t) -> str:
    return (f"QComboBox {{ background:{t.card}; color:{t.foreground};"
            f" border:1px solid {t.input}; border-radius:{t.radius_md:g}px;"
            f" padding:3px 8px; min-height:{t.control_height_sm - 8}px; }}"
            f"QComboBox QAbstractItemView {{ background:{t.card}; color:{t.foreground};"
            f" selection-background-color:{t.accent}; selection-color:{t.foreground}; }}"
            f"QComboBox::drop-down {{ border:none; width:22px; }}")


class SegmentPage(QWidget):
    """The settings grid, the run's progress and its log. Signals carry requests."""

    model_changed = Signal(str, str)        # condition, model
    folder_changed = Signal(str, str)       # condition, folder name, or BROWSE_FOLDER
    copy_requested = Signal()
    replace_changed = Signal(bool)
    segment_requested = Signal(str)
    segment_all_requested = Signal()
    stop_requested = Signal()
    pack_requested = Signal()               # the segmentation pack's offer
    open_editor_requested = Signal(str)
    #: A row's Ignore / Include (asked 15 Oct 2026: "should be in the normal window as well when i
    #: load multiple conditions"): the condition, and whether it is now ignored.
    ignore_toggled = Signal(str, bool)

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.setStyleSheet("background:transparent;")
        self._rows: dict = {}
        self._rebuilding = False
        self.last_done: str | None = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)
        lay.addWidget(label("Segment", ty.ui("lg", "semibold"), t.foreground))
        intro = label(
            "Draw outlines automatically for every field of a condition. They are saved to "
            "cell_meshes/<field>.mat, which the cell filter and the spatial maps read. "
            "You can correct them afterwards in Edit outlines.", ty.ui("sm"), t.muted_foreground)
        intro.setWordWrap(True)
        lay.addWidget(intro)
        lay.addWidget(InfoLink(
            t, "Which model?",
            "bact_phase_omni reads the phase-contrast image; bact_fluor_omni reads the "
            "summed fluorescence (sum/). For a membrane- or DNA-bound label, fluorescence "
            "gives outlines about half the true cell area, so use phase. A model you "
            "trained reads the images it was trained on."))

        card = Card(t, shadowed=False)
        self._grid = QGridLayout(card)
        self._grid.setContentsMargins(14, 12, 14, 12)
        self._grid.setHorizontalSpacing(12)
        self._grid.setVerticalSpacing(8)
        self._grid.setColumnStretch(4, 1)
        for col, text in enumerate(("Condition", "Model", "Reads", "Fields", "Status")):
            self._grid.addWidget(label(text, ty.ui("xs", "medium"), t.muted_foreground),
                                 0, col)
        lay.addWidget(card)

        foot = QHBoxLayout()
        foot.setSpacing(8)
        copy = Button(t, "Copy the first row to all", size="sm")
        copy.setToolTip("Give every condition the first row's model and image folder. A condition "
                        "without a folder of that name keeps its own.")
        copy.clicked.connect(self.copy_requested.emit)
        foot.addWidget(copy)
        self.replace = Checkbox(t, checked=False)
        self.replace.toggled.connect(self.replace_changed.emit)
        foot.addWidget(self.replace)
        foot.addWidget(label("Replace existing outlines", ty.ui("sm"), t.foreground))
        self.gpu = Checkbox(t, checked=False)
        self.gpu.setEnabled(False)
        foot.addWidget(self.gpu)
        self.gpu_label = label("Use GPU: checking…", ty.ui("sm"), t.muted_foreground)
        foot.addWidget(self.gpu_label)
        foot.addStretch(1)
        self._stop = Button(t, "Stop after this field", size="sm")
        self._stop.clicked.connect(self.stop_requested.emit)
        foot.addWidget(self._stop)
        self._stop.setVisible(False)
        self._all = Button(t, "Segment all ready", variant="default")
        self._all.clicked.connect(self.segment_all_requested.emit)
        foot.addWidget(self._all)
        lay.addLayout(foot)
        self.runtime_note = label("", ty.ui("xs", "medium"), "#c05621")
        self.runtime_note.setWordWrap(True)
        lay.addWidget(self.runtime_note)
        self.runtime_note.setVisible(False)
        self.pack_button = Button(t, "Get segmentation models…", size="sm")
        self.pack_button.clicked.connect(self.pack_requested.emit)
        lay.addWidget(self.pack_button, 0, Qt.AlignLeft)
        self.pack_button.setVisible(False)

        run = Card(t, shadowed=False)
        rl = QVBoxLayout(run)
        rl.setContentsMargins(14, 12, 14, 12)
        rl.setSpacing(8)
        head = QHBoxLayout()
        head.addWidget(label("Run", ty.ui("sm", "semibold"), t.foreground))
        head.addStretch(1)
        self._open_editor = Button(t, "Open in the editor", size="sm", glyph="pencil")
        self._open_editor.clicked.connect(
            lambda: self.last_done and self.open_editor_requested.emit(self.last_done))
        head.addWidget(self._open_editor)
        self._open_editor.setVisible(False)
        rl.addLayout(head)
        self.bar = Progress(t, 0.0)
        rl.addWidget(self.bar)
        self.progress = label("nothing running", ty.ui("xs", tabular=True), t.muted_foreground)
        self.progress.setWordWrap(True)
        rl.addWidget(self.progress)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(ty.mono("xs"))
        self.log.setMinimumHeight(180)
        self.log.setStyleSheet(
            f"QPlainTextEdit {{ background:{t.card}; color:{t.foreground};"
            f" border:1px solid {t.border}; border-radius:{t.radius_md:g}px; padding:6px; }}")
        rl.addWidget(self.log, 1)
        lay.addWidget(run, 1)

    # -- what the window tells the page ------------------------------------------- #
    def set_rows(self, rows: list, *, busy: bool) -> None:
        """``rows``: ``[{name, models: [(value, label)], model, folders: [names],
        folder (name or None), fields, status, ready, running}]``."""
        self._rebuilding = True
        try:
            for row in rows:
                widgets = self._rows.get(row["name"]) or self._add_row(row["name"])
                self._fill(widgets["model"], [lab for _v, lab in row["models"]],
                           next((lab for v, lab in row["models"] if v == row["model"]),
                                row["model"]))
                widgets["models"] = row["models"]
                folders = list(row["folders"]) + [BROWSE_FOLDER]
                current = row["folder"] or CHOOSE_FOLDER
                if row["folder"] is None:
                    folders = [CHOOSE_FOLDER] + folders
                elif row["folder"] not in folders:
                    folders = [row["folder"]] + folders
                self._fill(widgets["folder"], folders, current)
                widgets["fields"].setText(str(row["fields"]))
                widgets["status"].setText(row["status"])
                widgets["run"].setEnabled(bool(row["ready"]) and not busy)
                widgets["run"].set_text(row.get("action", "Segment"))
                ignored = bool(row.get("ignored"))
                widgets["ignored"] = ignored
                widgets["ignore"].set_text("Include" if ignored else "Ignore")
                widgets["ignore"].setToolTip(
                    "Segment this condition again" if ignored else
                    "Leave this condition out of Segment all and the batch")
                widgets["ignore"].setEnabled(not busy)
                widgets["name"].setStyleSheet(
                    f"color:{self.t.muted_foreground if ignored else self.t.foreground};"
                    " background:transparent;")
        finally:
            self._rebuilding = False
        self._all.setEnabled(any(r["ready"] for r in rows) and not busy)
        self._stop.setVisible(busy)

    @staticmethod
    def _fill(box: QComboBox, items, current) -> None:
        if [box.itemText(i) for i in range(box.count())] != list(items):
            box.clear()
            box.addItems(list(items))
        box.setCurrentText(current)

    def _add_row(self, name: str) -> dict:
        t = self.t
        r = len(self._rows) + 1
        name_label = label(name, ty.ui("sm", "medium"), t.foreground)
        self._grid.addWidget(name_label, r, 0)
        model = QComboBox()
        model.setFont(ty.ui("sm"))
        model.setStyleSheet(_combo_css(t))
        model.setMinimumWidth(220)
        folder = QComboBox()
        folder.setFont(ty.ui("sm"))
        folder.setStyleSheet(_combo_css(t))
        folder.setMinimumWidth(150)
        fields = label("", ty.ui("sm", tabular=True), t.foreground)
        status = label("", ty.ui("xs"), t.muted_foreground)
        status.setWordWrap(True)
        # room for a status line ("ignored -- press Include to segment it") beside the model box,
        # which otherwise takes the row's width and wraps the status to a word a line
        status.setMinimumWidth(190)
        run = Button(t, "Segment", size="sm")
        run.clicked.connect(lambda n=name: self.segment_requested.emit(n))
        ignore = Button(t, "Ignore", size="sm", variant="ghost")
        ignore.clicked.connect(
            lambda n=name: self.ignore_toggled.emit(n, not self._rows[n].get("ignored")))
        for col, w in enumerate((model, folder, fields, status, run, ignore), start=1):
            self._grid.addWidget(w, r, col)
        widgets = {"model": model, "folder": folder, "fields": fields, "status": status,
                   "run": run, "ignore": ignore, "name": name_label, "ignored": False,
                   "models": []}

        # A REDRAW IS NOT A CHOICE: refilling a box fires its signal, so the page only
        # reports a change the user made.
        def model_picked(text, n=name):
            if self._rebuilding:
                return
            value = next((v for v, lab in self._rows[n]["models"] if lab == text), text)
            self.model_changed.emit(n, value)

        def folder_picked(text, n=name):
            if self._rebuilding or text == CHOOSE_FOLDER:
                return
            self.folder_changed.emit(n, text)

        model.currentTextChanged.connect(model_picked)
        folder.currentTextChanged.connect(folder_picked)
        self._rows[name] = widgets
        return widgets

    def set_gpu(self, available: bool, why: str, *, use: bool = True) -> None:
        """``use`` is the user's standard (`options`): off leaves a found GPU unticked."""
        self.gpu.setEnabled(bool(available))
        self.gpu.set_checked(bool(available and use), animate=False)
        self.gpu_label.setText(("Use GPU" if use else "Use GPU (off by default, see Options)")
                               if available else f"Use GPU: unavailable ({why})")
        self.gpu_label.setToolTip(why)

    def set_runtime(self, available: bool, why: str, offer: bool = False) -> None:
        """``offer``: the packaged program can download the models (`smtracker.segpack`); the button opens that offer."""
        self.runtime_note.setText("" if available else
                                  f"{why} The classical segmenter (sum image) and Edit outlines "
                                  f"still work.")
        self.runtime_note.setVisible(not available)
        self.pack_button.setVisible(bool(offer and not available))

    def set_progress(self, fraction: float, text: str) -> None:
        self.bar.set_value(fraction)
        self.progress.setText(text)

    def finished(self, name: str) -> None:
        self.last_done = name
        self._open_editor.set_text(f"Open {name} in the editor")
        self._open_editor.setVisible(True)

    def append(self, *lines: str) -> None:
        for line in lines:
            self.log.appendPlainText(line)
