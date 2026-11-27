"""The outlines window: automatic segmentation, the mesh editor and training, in the
workspace's style.

Asked for: "finish the segmentation part and mesh editor with guys gui. Always try to
implement our philosophy of making the software never look stuck and with helpers,
animations sometimes and loading information. For the mesh editor its important to keep
the shortcuts like d for draw cell etc."

- **Segment** (`segment_page`): each condition's model and images, then the run. The
  rules are `segmentation.outline_model`; the run is `segment_condition` on a thread,
  reported field by field with a rate and an estimate.
- **Edit outlines** (`editor_page`): the classic editor's own handlers
  (`editor_gui._core_class`), so every shortcut works as before, in the UI designer's widgets.
- **Train a model**: the trainer window, which is not ported.
- **Batch** (asked 13 Oct 2026, "multiple parent folders in batches overnight"): several
  parent folders, every condition under them with its fields, how many lack an outline and
  its layout, then one run segmenting ONLY the missing fields of each (Replace as on
  Segment), one condition at a time through the same `segment_condition`
  (`batch_model.run_outlines`, on a thread). No folder needs to be open; no Next button.
  While it runs it holds the window as a run does and keeps Windows awake.

The window paints first; opening a folder lists it on a thread. A run holds the model,
so conditions are segmented ONE AT A TIME with a queue for the rest, and Stop takes
effect after the field in hand -- a field's outlines are written whole or not at all.
`--segment-classic` / `--editor-classic` open the old windows.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from qtpy.QtCore import QObject, Qt, QThreadPool, QTimer, Signal
from qtpy.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ... import segpack
from ...segmentation.editor_project import scan_fields
from ...segmentation.outline_model import (
    copy_settings,
    existing_outlines,
    field_line,
    models,
    next_step,
    open_outlines,
    can_run,
    progress_words,
    usable_model,
    readiness,
    run_action,
    run_fields,
    segment_condition,
    set_folder,
    set_model,
)
from ...batch_model import (
    IGNORED_WORDS,
    add_parent,
    batch_progress,
    event_line,
    finished_summary,
    folder_key,
    awake_note,
    keep_awake,
    layout_notes,
    outline_batch_cells,
    outline_summary,
    plan_lines,
    result_status,
    running_status,
)
from .. import theme
from ..prep.conditions import ConditionList
from ..prep.jobs import _Call
from ..theme import typography as ty
from ..widgets.help import show_help_menu
from ..widgets.controls import Button
from ..widgets.options_dialog import open_options
from ..widgets.primitives import FitScroll, ShadowHost, activity_label, label, scrollbar_css
from ..widgets.progress import RunProgress, notify_after_s, show_finished
from ... import options as opts
from ...run_progress import finished_payload, progress_payload
from ..workspace.sidebar import Breadcrumb, Sidebar, SidebarTrigger
from .batch_page import OutlineBatchPage
from .classical_page import ClassicalPage
from .editor_page import EditorPage
from .segment_page import BROWSE_FOLDER, SegmentPage

__all__ = ["OutlinesWindow", "open_outlines_window", "OUTLINE_SECTIONS"]

#: ui.json key: the condition folders left out of the outlines batch (`set_batch_ignored`).
BATCH_IGNORED_KEY = "outline_batch_ignored"

OUTLINE_SECTIONS = (
    ("segment", "Segment", "scan", (("segment.run", "Run"),)),
    # the classical segmenter's sliders and live preview (asked 15 Oct 2026: "i mean with the settings")
    ("classical", "Classical settings", "sliders-horizontal", (("classical.tune", "Tune"),)),
    ("edit", "Edit outlines", "pencil", (("edit.fields", "Fields"),)),
    ("train", "Train a model", "flask-conical", (("train.open", "Open"),)),
    # several parent folders, overnight (asked 13 Oct 2026): no folder needed, no Next
    ("batch", "Batch", "layers", (("batch.run", "Run"),)),
)
_TITLES = {key: title for key, title, _g, _s in OUTLINE_SECTIONS}

#: Windows opened here, held: a parentless window nobody references is destroyed at once
#: and takes its threads with it (the segmentation door's exit 139).
_OPEN_WINDOWS: set = set()


class _Signals(QObject):
    #: generation, ``({condition: OutlineCondition}, the editor's fields)`` or None, error
    opened = Signal(int, object, str)
    #: generation, folders read, folders in all, the folder in hand: the folder being opened
    scanning = Signal(int, int, int, str)
    #: generation, the editor's fields read again after a run (None on failure), error
    rescanned = Signal(int, object, str)
    #: available, why
    gpu = Signal(bool, str)
    #: condition, done, total, the field's plain row
    field = Signal(str, int, int, object)
    #: condition, `segment_condition`'s result or None, error
    segmented = Signal(str, object, str)
    #: condition, what the run is doing before its first field (a model download, the load)
    status = Signal(str, str)
    #: generation, ``{"rows": batch_model.plan_outlines rows}`` or None, error
    batch_planned = Signal(int, object, str)
    #: kind, condition key, payload: `batch_model.run_outlines`' events
    batch_event = Signal(str, str, object)
    #: the batch's rows or None, error
    batch_done = Signal(object, str)


class _TrainPage(QWidget):
    def __init__(self, t, on_open, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background:transparent;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)
        lay.addWidget(label("Train a model", ty.ui("lg", "semibold"), t.foreground))
        text = label(
            "Train a model on your corrected outlines. Some fields are held out to compare the "
            "new model with its starting model. You can train here or export a bundle for a "
            "GPU machine; a trained model appears in Segment's model list. The trainer opens "
            "in its own window.",
            ty.ui("sm"), t.muted_foreground)
        text.setWordWrap(True)
        text.setMaximumWidth(680)
        lay.addWidget(text)
        button = Button(t, "Open the trainer", glyph="arrow-right")
        button.clicked.connect(on_open)
        lay.addWidget(button, alignment=Qt.AlignLeft)
        lay.addStretch(1)


class _StartPage(QWidget):
    def __init__(self, t, on_open, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background:transparent;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 40, 0, 0)
        lay.setSpacing(12)
        lay.addWidget(label("Cell outlines", ty.ui("2xl", "semibold"), t.foreground))
        self.text = label(
            "Open a condition that holds phase/ (and sum/), or a folder of conditions. Segment "
            "draws outlines automatically; Edit outlines lets you correct them by hand.",
            ty.ui("sm"), t.muted_foreground)
        self.text.setWordWrap(True)
        self.text.setMaximumWidth(620)
        lay.addWidget(self.text)
        button = Button(t, "Open folder…", variant="default", glyph="folder-open")
        button.clicked.connect(on_open)
        lay.addWidget(button, alignment=Qt.AlignLeft)
        lay.addStretch(1)


class OutlinesWindow(QMainWindow):
    """The shell, the conditions, and the wiring between the pages and the jobs."""

    def __init__(self, t, *, section: str = "segment", threads: QThreadPool | None = None,
                 parent=None):
        super().__init__(parent)
        self.t = t
        from ...buildstamp import build_stamp

        self.setWindowTitle(f"SMTracker · cell outlines  ·  {build_stamp()}")
        self.setStyleSheet(f"QMainWindow {{ background:{t.background}; }}" + scrollbar_css(t))
        self._threads = threads or QThreadPool.globalInstance()
        self.signals = _Signals()
        self.signals.opened.connect(self._opened)
        self.signals.scanning.connect(self._scanning)
        self.signals.rescanned.connect(self._rescanned)
        self.signals.gpu.connect(self._gpu)
        self.signals.field.connect(self._field)
        self.signals.segmented.connect(self._segmented)
        self.signals.status.connect(self._status)
        self.signals.batch_planned.connect(self._batch_planned)
        self.signals.batch_event.connect(self._batch_event)
        self.signals.batch_done.connect(self._batch_done)
        self.states: dict = {}
        self.folder: Path | None = None
        self.condition: str | None = None
        self._generation = 0
        #: The folder being opened, as the progress card shows it (None when none is).
        self._opening: dict | None = None
        self._models: list = []
        self._folders: dict = {}
        self._replace = False
        self._running: str | None = None
        self._rows: list = []
        self._queue: list = []
        #: Whether Cellpose-SAM's download was agreed to this session (`_download_agreed`).
        self._download_ok = False
        self._stop: threading.Event | None = None
        self.return_to_menu = False
        self._trainers: list = []
        #: The run as the progress card and the finished pop-up see it: when it started,
        #: its plan ({condition: fields}), the conditions done, the field in hand, the
        #: summary lines, and how long it must run before its end gets a pop-up.
        self._run_started: float | None = None
        self._run_plan: dict = {}
        self._run_done: list = []
        self._field_done: tuple = (0, 0)
        self._run_lines: list = []
        self._run_failed = False
        self._notify_after = notify_after_s()
        #: The batch over several parent folders (`batch_model`): the parents, the rows last
        #: planned, which listing is current, Replace, and -- while one runs -- its stop flag
        #: (None when none runs), plan, running and finished conditions, fields done, each
        #: row's status, the finished rows, and its summary line once it ended.
        self._batch_parents: list = []
        self._batch_rows: list = []
        #: Condition folders (their batch keys) left out of the batch, REMEMBERED per user:
        #: "i dont wanna segment a condition again when i load the batch" (14 Oct 2026).
        self._batch_ignored: set = set(theme.remembered(BATCH_IGNORED_KEY) or [])
        self._batch_generation = 0
        self._batch_replace = False
        self._batch_stop: threading.Event | None = None
        self._batch_started: float | None = None
        self._batch_plan: list = []
        self._batch_running: list = []
        self._batch_finished: list = []
        self._batch_counts: dict = {}
        self._batch_status: dict = {}
        self._batch_results: dict = {}
        self._batch_summary = ""

        central = QWidget()
        outer = QHBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.sidebar = Sidebar(t, project_title="Cell outlines", project_subtitle="no folder",
                               sections=OUTLINE_SECTIONS, group_label="Cell outlines")
        self.sidebar.selected.connect(self._on_section)
        self.sidebar.opened.connect(self.choose_folder)
        self.conditions = ConditionList(t)
        self.conditions.picked.connect(self.show_condition)
        self.sidebar.add_group("Conditions", self.conditions)
        outer.addWidget(self.sidebar)

        column = ShadowHost(t)
        column.setStyleSheet(f"background:{t.background};")
        col = QVBoxLayout(column)
        col.setContentsMargins(26, 16, 26, 16)
        col.setSpacing(12)
        self._column_layout = col
        #: The header row as a widget, so the editor's focus mode can hide it.
        self.header = QWidget()
        self.header.setStyleSheet("background:transparent;")
        head = QHBoxLayout(self.header)
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(8)
        trigger = SidebarTrigger(t)
        trigger.clicked.connect(lambda: self.sidebar.set_collapsed(not self.sidebar.collapsed))
        head.addWidget(trigger)
        self.breadcrumb = Breadcrumb(t)
        head.addWidget(self.breadcrumb)
        # the activity line takes the header's free room and wraps there (never a minimum
        # width that pushes the window past the screen: `primitives.activity_label`)
        self.activity = activity_label(ty.ui("xs"), t.muted_foreground)
        head.addWidget(self.activity, 1)
        open_button = Button(t, "Open folder…", size="sm", glyph="folder-open")
        open_button.clicked.connect(self.choose_folder)
        head.addWidget(open_button)
        help_button = Button(t, "Help", size="sm", glyph="info")
        help_button.setToolTip("The manual: on the website, or the copy in SMTracker.")
        help_button.clicked.connect(lambda: show_help_menu("outlines", self, help_button))
        head.addWidget(help_button)
        if segpack.offered_here():
            self.pack_button = Button(t, "Segmentation pack", size="sm", glyph="download")
            self.pack_button.setToolTip("PyTorch and Cellpose for Cellpose-SAM and training: download them, see "
                                        "where they are, or remove them.")
            self.pack_button.clicked.connect(self.open_pack)
            head.addWidget(self.pack_button)
        options_button = Button(t, "Options", size="sm", glyph="sliders-horizontal")
        options_button.setToolTip("Defaults for new sessions: the model, the GPU, and when a "
                                  "finished-run pop-up appears.")
        options_button.clicked.connect(self.open_options)
        head.addWidget(options_button)
        back = Button(t, "Back to menu", size="sm")
        back.setToolTip("Return to the stage menu. Closing the window quits the tool.")
        back.clicked.connect(self.back_to_menu)
        head.addWidget(back)
        col.addWidget(self.header)
        self.run_progress = RunProgress(t)
        col.addWidget(self.run_progress)
        self._progress_timer = QTimer(self)
        self._progress_timer.setInterval(1000)
        self._progress_timer.timeout.connect(self._refresh_progress)

        self.pages = QStackedWidget()
        self.start_page = _StartPage(t, self.choose_folder)
        self.segment_page = SegmentPage(t)
        self.editor = EditorPage(t)
        self.editor.saved.connect(self._outlines_saved)
        self.editor.edits_changed.connect(self._edits_changed)
        self.train_page = _TrainPage(t, self.open_trainer)
        segment_scroll = QScrollArea()
        segment_scroll.setWidget(self.segment_page)
        segment_scroll.setWidgetResizable(True)
        segment_scroll.setFrameShape(QScrollArea.NoFrame)
        segment_scroll.setStyleSheet("QScrollArea { background:transparent; border:none; }")
        self.batch_page = OutlineBatchPage(t)
        self.classical_page = ClassicalPage(t)
        from ...segmentation.classical_view import SETTINGS_KEY as _CLASSICAL_KEY

        #: The classical segmenter's settings, remembered per user; every classical run uses them.
        self._classical = theme.remembered(_CLASSICAL_KEY) or None
        self.classical_page.set_settings(self._classical)
        self.classical_page.settings_saved.connect(self._classical_saved)
        batch_scroll = QScrollArea()
        batch_scroll.setWidget(self.batch_page)
        batch_scroll.setWidgetResizable(True)
        batch_scroll.setFrameShape(QScrollArea.NoFrame)
        batch_scroll.setStyleSheet("QScrollArea { background:transparent; border:none; }")
        # THE EDITOR SCROLLS WHERE THE WINDOW IS SHORTER THAN IT (30 Oct 2026): its controls, canvas and keys asked for
        # 726 px and held the whole window at 802, past a 1536 x 816 laptop's maximised window (785). `FitScroll`:
        # no scroll range where the screen holds it; the canvas takes its wheel for zooming either way.
        editor_scroll = FitScroll()
        # held by a plain widget: a scroll area asks its widget's LAYOUT whether its height follows its width, and
        # the holder's layout asks the editor, which says no (`EditorPage.hasHeightForWidth`)
        editor_holder = QWidget()
        editor_holder.setStyleSheet("background:transparent;")
        holder_lay = QVBoxLayout(editor_holder)
        holder_lay.setContentsMargins(0, 0, 0, 0)
        holder_lay.addWidget(self.editor)
        editor_scroll.setWidget(editor_holder)
        editor_scroll.setWidgetResizable(True)
        editor_scroll.setFrameShape(QScrollArea.NoFrame)
        editor_scroll.setStyleSheet("QScrollArea { background:transparent; border:none; }")
        self.editor_scroll = editor_scroll
        # the classical settings too: its preview and settings card asked for 572 px, the window 648 -- past a 1280 x
        # 720 laptop's 641 (`scripts/gui_sweep.py`, 30 Oct 2026). Its wrapped intro keeps it height-for-width, so it
        # is laid out at its preferred height for the width (606 px): it fills a laptop's view and scrolls below that.
        classical_scroll = FitScroll()
        classical_scroll.setWidget(self.classical_page)
        classical_scroll.setWidgetResizable(True)
        classical_scroll.setFrameShape(QScrollArea.NoFrame)
        classical_scroll.setStyleSheet("QScrollArea { background:transparent; border:none; }")
        self.classical_scroll = classical_scroll
        for page in (self.start_page, segment_scroll, editor_scroll, self.train_page,
                     batch_scroll, classical_scroll):
            self.pages.addWidget(page)
        self._section_page = {"segment": segment_scroll, "edit": editor_scroll,
                              "train": self.train_page, "batch": batch_scroll,
                              "classical": classical_scroll}
        col.addWidget(self.pages, 1)
        # THE NEXT BUTTON (asked 13 Oct 2026), as in the preparation window: lit when
        # `outline_model.next_step` says the page's work is done.
        foot = QHBoxLayout()
        foot.setSpacing(10)
        self.next_note = label("", ty.ui("xs"), t.muted_foreground)
        self.next_note.setWordWrap(True)
        foot.addWidget(self.next_note, 1)
        self.next_button = Button(t, "Next", variant="default", glyph="arrow-right")
        self.next_button.clicked.connect(self._next_clicked)
        foot.addWidget(self.next_button)
        self.next_bar = QWidget()
        self.next_bar.setLayout(foot)
        self.next_bar.setVisible(False)
        col.addWidget(self.next_bar)
        outer.addWidget(column, 1)
        self.setCentralWidget(central)
        self.resize(1360, 900)

        sp = self.segment_page
        sp.model_changed.connect(self._model_changed)
        sp.folder_changed.connect(self._folder_changed)
        sp.copy_requested.connect(self._copy)
        sp.replace_changed.connect(self._replace_changed)
        sp.segment_requested.connect(self.segment)
        sp.segment_all_requested.connect(self.segment_all)
        sp.stop_requested.connect(self.stop)
        sp.open_editor_requested.connect(self.edit_condition)
        bp = self.batch_page
        bp.add_requested.connect(self.choose_batch_parent)
        bp.remove_requested.connect(self.remove_batch_parent)
        bp.run_requested.connect(self.run_batch)
        bp.stop_requested.connect(self.stop_batch)
        bp.ignore_toggled.connect(self.set_batch_ignored)
        sp.ignore_toggled.connect(self.set_condition_ignored)
        bp.replace_changed.connect(self._batch_replace_changed)

        from ...segmentation.outline_model import runtime_words

        why = runtime_words()
        sp.set_runtime(not why, why, offer=segpack.offered_here())
        sp.pack_requested.connect(self.open_pack)
        self._show_batch_rows()
        first = {key: subs[0][0] for key, _t, _g, subs in OUTLINE_SECTIONS}
        self.sidebar.select(first.get(section, "segment.run"))
        self.pages.setCurrentWidget(self.start_page)

    def _can_run(self, model) -> bool:
        """Whether ``model`` can segment in this build: the classical segmenter always, Cellpose-SAM
        where Cellpose 4 is installed, the Omnipose models only where Omnipose is
        (`outline_model.can_run`)."""
        return can_run(model)

    # -- screen size ----------------------------------------------------------------- #
    def showEvent(self, e) -> None:  # noqa: N802 - Qt's name
        """On a screen narrower than the window's minimum, start with the sidebar as its
        icon rail: Qt otherwise squeezes every page below its minimum and clips it."""
        super().showEvent(e)
        if getattr(self, "_fitted", False):
            return
        self._fitted = True
        screen = self.screen()
        if screen is not None and (self.minimumSizeHint().width()
                                   > screen.availableGeometry().width()):
            self.sidebar.set_collapsed(True, animate=False)

    def set_focus_mode(self, on: bool) -> None:
        """The editor's focus mode, the window's half: full screen, no sidebar, no
        header, slim margins; leaving restores how the window was."""
        if on:
            self._before_focus = (self.isMaximized(), self.isFullScreen())
            self.sidebar.hide()
            self.header.hide()
            self._column_layout.setContentsMargins(8, 8, 8, 8)
            self.showFullScreen()
            return
        self.sidebar.show()
        self.header.show()
        self._column_layout.setContentsMargins(26, 16, 26, 16)
        maximized, full = getattr(self, "_before_focus", (False, False))
        if full:
            return
        if maximized:
            self.showMaximized()
        else:
            self.showNormal()

    # -- folder ---------------------------------------------------------------------- #
    def choose_folder(self) -> None:
        if self._refuse("opening another folder"):
            return
        folder = QFileDialog.getExistingDirectory(
            self, "Open a condition, or a folder of conditions", str(self.folder or Path.home()))
        if folder:
            self.open_folder(folder)

    def _refuse(self, what: str) -> bool:
        if self._batch_stop is not None:
            self.activity.setText(f"A batch is running. Stop it on Batch before {what}.")
            return True
        if self._running is not None:
            self.activity.setText(f"Segmenting {self._running}. Stop it before {what}.")
            return True
        unsaved = self.editor.unsaved()
        if unsaved:
            self.activity.setText(f"{len(unsaved)} edited field(s) are not saved. Use Save all "
                                  f"edited before {what}.")
            return True
        return False

    def open_folder(self, folder) -> None:
        """List the folder AND read the editor's fields on a thread, the progress card
        showing how far it got: on the window's thread a large tree froze the window (20 s on
        a user's analysis folder; asked 15 Oct 2026, "why not just load and show a bar")."""
        if self._refuse("opening another folder"):
            return
        self._generation += 1
        generation = self._generation
        self.folder = Path(folder)
        self.states = {}
        self.sidebar.set_project(self.folder.name, "opening…")
        self.activity.setText(f"Listing {self.folder.name}…")
        self._opening = {"name": self.folder.name, "started": time.monotonic(), "done": 0,
                         "total": 0, "current": ""}
        self._refresh_progress()
        folder = self.folder
        model = self._option("segment_model")
        bright = self.editor.bright_folder

        def report(done, total, name):
            self._emit(self.signals.scanning, generation, done, total, name)

        def job():
            states = open_outlines(folder, model=model)
            fields = scan_fields(folder, bright=bright, on_progress=report)
            # the editor's first field reads its outlines through these: imported here, not
            # on the window's thread when that field opens
            from ...segmentation import editing  # noqa: F401
            from ... import cellmesh  # noqa: F401
            return states, fields

        self._threads.start(_Call(
            job, lambda r, e: self._emit(self.signals.opened, generation, r, e)))

    def _scanning(self, generation: int, done: int, total: int, name: str) -> None:
        if generation != self._generation or self._opening is None:
            return
        self._opening.update(done=done, total=total, current=name)
        self._refresh_progress()

    def _rescan_editor(self) -> None:
        """The editor's fields read again (new outlines after a run), on a thread."""
        if self.folder is None:
            return
        generation, folder, bright = self._generation, self.folder, self.editor.bright_folder
        self._threads.start(_Call(
            lambda: scan_fields(folder, bright=bright),
            lambda r, e: self._emit(self.signals.rescanned, generation, r, e)))

    def _rescanned(self, generation: int, fields, error: str) -> None:
        if generation != self._generation or self.folder is None or fields is None:
            return
        if not self.editor.unsaved():                  # an edit made meanwhile is never dropped
            self.editor.set_fields(self.folder, fields)

    def _emit(self, signal, *args) -> None:
        try:
            signal.emit(*args)
        except RuntimeError:                           # the window is gone
            pass

    def _opened(self, generation: int, result, error: str) -> None:
        if generation != self._generation:
            return
        self._opening = None
        self._refresh_progress()
        states, fields = result if result else ({}, [])
        self.activity.setText("")
        if error or not states:
            self.sidebar.set_project(self.folder.name if self.folder else "Cell outlines",
                                     "nothing to segment")
            self.start_page.text.setText(error.split(": ", 1)[-1] if error else
                                         "Nothing to segment in this folder.")
            self.pages.setCurrentWidget(self.start_page)
            return
        from ...segmentation.channels import folders_in

        self.states = states
        self._models = models()
        self._folders = {n: [p.name for p in folders_in(c.folder)] for n, c in states.items()}
        n_fields = sum(len(c.fields) for c in states.values())
        self.sidebar.set_project(self.folder.name, f"{len(states)} condition(s) · "
                                                    f"{n_fields} fields")
        self.editor.set_fields(self.folder, fields)
        self.condition = next(iter(states))
        self.editor.open_condition(self.condition)
        self._refresh()
        self._on_section(self.sidebar.current)
        self._threads.start(_Call(self._probe_gpu,
                                  lambda r, e: self._emit(self.signals.gpu,
                                                          *(r or (False, e or "unknown")))))

    @staticmethod
    def _probe_gpu():
        # `gpu_available` imports torch -- two seconds -- so it runs here, after the window
        # is up and the folder listed, never while either is being built.
        from ...segmentation.segment_gui import gpu_available

        return gpu_available()

    def _gpu(self, available: bool, why: str) -> None:
        self._gpu_found = (bool(available), why)
        self.segment_page.set_gpu(available, why, use=self._option("segment_use_gpu"))

    # -- the options screen ----------------------------------------------------------- #
    def _model_choices(self) -> tuple:
        return tuple((name, label_) for name, _roles, label_ in models())

    def _option(self, key: str, record=None):
        """One standard from the options screen (`smtracker.options`)."""
        record = theme.remembered(opts.OPTIONS_KEY) if record is None else record
        choices = self._model_choices() if key == "segment_model" else None
        value = opts.read(key, record, tool="outlines", choices=choices)
        # A STANDARD SAVED WHERE OMNIPOSE WAS INSTALLED must not hand this build a model it cannot run
        return usable_model(value) if key == "segment_model" else value

    def open_options(self):
        """The options screen (`ui.widgets.options_dialog`); a save applies at once."""
        return open_options(self, self.t, "outlines",
                            choices={"segment_model": self._model_choices()},
                            on_saved=self._options_saved)

    def _options_saved(self, old_record: dict, new_record: dict) -> None:
        """Apply new standards: a condition still on the OLD standard model follows the
        new one (not while it runs), and so does the GPU box; a choice made here stays."""
        self._notify_after = notify_after_s()
        moved = []
        old_model = self._option("segment_model", old_record)
        new_model = self._option("segment_model", new_record)
        if new_model != old_model:
            for name, cond in self.states.items():
                if cond.model == old_model and name != self._running:
                    note = set_model(cond, new_model)
                    if note:
                        self.segment_page.append(note)
                    moved.append(name)
        old_gpu = self._option("segment_use_gpu", old_record)
        new_gpu = self._option("segment_use_gpu", new_record)
        found = getattr(self, "_gpu_found", None)
        if found and found[0] and new_gpu != old_gpu \
                and bool(self.segment_page.gpu.checked) == old_gpu:
            self.segment_page.set_gpu(*found, use=new_gpu)
        self._refresh()
        if self.sidebar.current.startswith("batch"):
            self._plan_batch()                         # the batch runs the new standard model
        self.activity.setText("Options saved" + (
            f": {len(moved)} condition(s) switched to {new_model}" if moved else ""))

    # -- showing ---------------------------------------------------------------------- #
    def _rows_for_page(self) -> list:
        rows = []
        for name, cond in self.states.items():
            ready, status = readiness(cond, replace=self._replace)
            if name == self._running:
                status, ready = "segmenting…", False
            elif name in self._queue:
                status, ready = "queued", False
            elif not self._can_run(cond.model):
                ready = False
            # ONE LIST WITH THE BATCH'S, by folder: ignored here is ignored there (15 Oct 2026)
            ignored = folder_key(cond.folder) in self._batch_ignored
            if ignored and name != self._running:
                status, ready = IGNORED_WORDS, False
            role = cond.roles[0]
            resolved = cond.resolution.resolved.get(role) if cond.resolution else None
            rows.append({"name": name,
                         "models": [(v, lab) for v, _r, lab in self._models],
                         "model": cond.model, "folders": self._folders.get(name, []),
                         "folder": (cond.resolution.label(role)
                                    if resolved is not None else None),
                         "fields": len(cond.fields), "status": status, "ready": ready,
                         "ignored": ignored,
                         "action": run_action(cond, replace=self._replace)})
        return rows

    def _refresh(self) -> None:
        self.segment_page.set_rows(self._rows_for_page(), busy=self._running is not None)
        self.conditions.set_rows([
            {"name": n, "done": c.outlines[0], "total": max(len(c.fields), 1),
             "line": (f"{c.outlines[0]}/{len(c.fields)} outlined"
                      + (f" · {c.outlines[1]} corrected" if c.outlines[1] else "")
                      + (" · segmenting" if n == self._running else ""))}
            for n, c in self.states.items()])
        self.conditions.set_active(self.condition)
        self._refresh_next()
        self._refresh_progress()

    def _refresh_next(self) -> None:
        section = self.sidebar.current.split(".")[0]
        step = (next_step(section, self.states, running=self._running,
                          unsaved=bool(self.editor.unsaved()))
                if self.states else None)
        if step is None:
            self.next_bar.setVisible(False)
            return
        self._next_target = step["target"]
        self.next_button.set_text(step["label"])
        self.next_button.setEnabled(step["ready"])
        self.next_note.setText("" if step["ready"] else step["why"])
        self.next_bar.setVisible(True)

    def _edits_changed(self, _dirty: bool) -> None:
        if hasattr(self, "next_bar"):
            self._refresh_next()

    def _next_clicked(self) -> None:
        target = getattr(self, "_next_target", None)
        if target:
            self.sidebar.select_section(target)
        else:
            self.back_to_menu()

    def _refresh_progress(self) -> None:
        payload = None
        if self._batch_stop is not None:
            payload = batch_progress(
                title="Batch: segmenting", plan=self._batch_plan, done=self._batch_counts,
                finished=self._batch_finished, running=self._batch_running,
                started=self._batch_started, unit="field", count="todo")
        elif self._running is not None:
            done_fields, total_fields = self._field_done
            done = sum(self._run_plan.get(k, 0) for k in self._run_done) + done_fields
            payload = progress_payload(
                title="Segmenting", unit="field", done=done,
                total=sum(self._run_plan.values()), group="condition",
                groups_done=len(self._run_done), groups_total=len(self._run_plan),
                current=self._running, current_done=done_fields,
                current_total=total_fields or self._run_plan.get(self._running, 0),
                started=self._run_started)
        elif self._opening is not None:
            o = self._opening
            payload = progress_payload(
                title=f"Opening {o['name']}", unit="folder", done=o["done"], total=o["total"],
                current=o["current"], started=o["started"])
        self.run_progress.set_payload(payload)
        if payload and not self._progress_timer.isActive():
            self._progress_timer.start()
        elif not payload and self._progress_timer.isActive():
            self._progress_timer.stop()

    def _outlines_saved(self) -> None:
        """The editor wrote outlines: recount every condition's, so the list and the Segment
        rows say so. Reported: outlines drawn by hand and saved with Save all stayed "0/n
        outlined", because the count was taken only on opening. (A run recounts its own
        condition in `_segmented`; before 13 Oct 2026 it did not, so the Segment rows kept
        the count from before the run.)"""
        for cond in self.states.values():
            cond.outlines = existing_outlines(cond.folder)
        self._refresh()

    def _on_section(self, view: str) -> None:
        section = view.split(".")[0]
        if section == "batch":
            # NEEDS NO OPEN FOLDER, and is planned afresh on arrival: outlines drawn or
            # segmented meanwhile change what is missing.
            self.pages.setCurrentWidget(self._section_page[section])
            self._plan_batch()
        elif not self.states:
            self.pages.setCurrentWidget(self.start_page)
        else:
            self.pages.setCurrentWidget(self._section_page[section])
            if section == "edit" and self.condition:
                self.editor.open_condition(self.condition)
            if section == "classical":
                self._show_classical()
        trail = [_TITLES[section]] + ([self.condition] if self.condition
                                      and section != "batch" else [])
        self.breadcrumb.set_trail(trail)
        self._refresh_next()

    def show_condition(self, name: str) -> None:
        if name not in self.states:
            return
        self.condition = name
        self.conditions.set_active(name)
        if self.sidebar.current.startswith("edit"):
            self.editor.open_condition(name)
        self._on_section(self.sidebar.current)

    def edit_condition(self, name: str) -> None:
        """Open the editor on ``name``'s first field (the run's "Open in the editor")."""
        self.condition = name
        self.sidebar.select("edit.fields")
        self.editor.open_condition(name)
        self._on_section("edit.fields")

    # -- settings --------------------------------------------------------------------- #
    def _model_changed(self, name: str, model: str) -> None:
        note = set_model(self.states[name], model)
        if note:
            self.segment_page.append(note)
        self._refresh()

    def _folder_changed(self, name: str, folder: str) -> None:
        cond = self.states[name]
        if folder == BROWSE_FOLDER:
            picked = QFileDialog.getExistingDirectory(self, f"The images {name} reads",
                                                      str(cond.folder))
            if not picked:
                self._refresh()                        # restore the cell
                return
            target = Path(picked)
        else:
            target = cond.folder / folder
        set_folder(cond, target)
        self._refresh()

    def _copy(self) -> None:
        if not self.states:
            return
        source = next(iter(self.states))
        changed = copy_settings(self.states, source)
        self.segment_page.append(f"Copied the model and folder of {source} to "
                                 f"{len(changed)} condition(s)" if changed else
                                 "Every condition already matches the first row.")
        self._refresh()

    def _replace_changed(self, on: bool) -> None:
        self._replace = bool(on)
        self._refresh()

    # -- running ---------------------------------------------------------------------- #
    def segment(self, name: str) -> None:
        """Segment ``name`` now, or queue it behind the run in flight."""
        if name not in self.states:
            return
        if self._batch_stop is not None:
            # one model on one GPU at a time: a run beside a batch would compete with it
            self.segment_page.append(f"Not segmented: {name}. A batch is running; segment it "
                                     f"after the batch is done, or stop the batch on Batch.")
            return
        if self._running is not None:
            if name != self._running and name not in self._queue:
                self._queue.append(name)
                self._run_plan.setdefault(
                    name, len(run_fields(self.states[name], replace=self._replace)))
            self._refresh()
            return
        self._start(name)

    def segment_all(self) -> None:
        if self._batch_stop is not None:
            self.segment_page.append("A batch is running. Segment after it is done, or stop "
                                     "it on Batch.")
            return
        ready = [r["name"] for r in self._rows_for_page() if r["ready"]]
        if not ready:
            self.segment_page.append("No condition is ready to segment. Each row says why.")
            return
        self.segment_page.append(f"Segmenting {len(ready)} condition(s), one at a time: "
                                 + ", ".join(ready))
        for name in ready:
            self.segment(name)

    def _start(self, name: str) -> bool:
        cond = self.states[name]
        ready, why = readiness(cond, replace=self._replace)
        if not self._can_run(cond.model):
            ready, why = False, ("this build cannot run the models. Pick the classical "
                                 "segmenter, or see the note above.")
        if not ready:
            self.segment_page.append(f"Not segmented: {name}: {why}")
            self._next()
            return False
        use_gpu = bool(self.segment_page.gpu.checked and self.segment_page.gpu.isEnabled())
        if not self._download_agreed(name, cond.model):
            self._next()
            return False
        # Replace AS IT WAS WHEN PRESSED: the run must not change what it does if the box
        # is ticked while it is going.
        replace = self._replace
        targets = run_fields(cond, replace=replace)
        kept = len(cond.fields) - len(targets)
        reads = ", ".join(
            f"{role} <- {cond.resolution.label(role)}"
            + ("" if role in cond.resolution.stems_of else "/")
            for role in cond.resolution.resolved)
        self.segment_page.append(
            "", f"{name}: {len(targets)} field(s), model {Path(cond.model).name}, reads "
                f"{reads}, {'GPU' if use_gpu else 'CPU'} -> {cond.folder.name}/cell_meshes/",
            *([f"    continuing: {kept} field(s) already outlined are kept"] if kept else []),
            "    the first field also loads the model, so it is the slowest")
        self._running = name
        self._rows = []
        self._stop = threading.Event()
        if self._run_started is None:
            self._run_started = time.monotonic()
        self._run_plan[name] = len(targets)
        self._field_done = (0, len(targets))
        stop = self._stop
        self.segment_page.set_progress(0.0, progress_words(name, 0, len(targets), []))
        cond.classical = self._classical                 # the classical segmenter's settings
        self._threads.start(_Call(
            lambda: segment_condition(
                cond, use_gpu=use_gpu, replace=replace, stop=stop.is_set,
                on_field=lambda d, n, row: self._emit(self.signals.field, name, d, n, row),
                on_status=lambda text: self._emit(self.signals.status, name, text)),
            lambda r, e: self._emit(self.signals.segmented, name, r, e)))
        self.activity.setText(f"segmenting {name}…")
        self._refresh()
        return True

    def _download_agreed(self, name: str, model) -> bool:
        """Say BEFORE a run that needs Cellpose-SAM's weights (about 1.2 GB, not shipped) that they download now,
        from where and to where, and go on only on a yes. Asked once per session; where they are already here
        nothing is asked."""
        from ...segmentation import backends
        from ...segmentation.segment import CELLPOSE

        if str(model) not in CELLPOSE or backends.cpsam_downloaded() or self._download_ok:
            return True
        folder = backends.cellpose_model_dir()
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle("Cellpose-SAM must be downloaded")
        box.setText(f"Cellpose-SAM's model file is not on this computer yet. It is about "
                    f"{backends.CPSAM_APPROX_BYTES / 2**30:.1f} GB and downloads once, now, from "
                    f"Hugging Face to {folder}. Segmenting starts when it is done.")
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.button(QMessageBox.Yes).setText("Download and segment")
        box.button(QMessageBox.No).setText("Not now")
        agreed = box.exec() == QMessageBox.Yes
        if not agreed:
            self.segment_page.append(f"Not segmented: {name}: Cellpose-SAM's model file was not downloaded. "
                                     f"Pick the classical segmenter, or run it again and agree to the download.")
        self._download_ok = agreed
        return agreed

    def _status(self, name: str, text: str) -> None:
        """What the run does before its first field: shown on the progress line and in the log."""
        if name != self._running:
            return
        self.segment_page.set_progress(0.0, f"{name}: {text}")
        self.segment_page.append(text)
        self.activity.setText(text)

    def _field(self, name: str, done: int, total: int, row) -> None:
        if name != self._running:
            return
        self._rows.append(row)
        self._field_done = (done, total)
        self._refresh_progress()
        self.segment_page.append(field_line(done, total, row))
        self.segment_page.set_progress(done / total if total else 1.0,
                                       progress_words(name, done, total, self._rows))

    def _segmented(self, name: str, result, error: str) -> None:
        if name != self._running:
            return
        self._running, self._stop = None, None
        self.activity.setText("")
        self._run_done.append(name)
        self._field_done = (0, 0)
        if name in self.states:
            # what the run wrote (a failed or stopped run may still have written some):
            # the rows, the editor's count and the Next button read it
            self.states[name].outlines = existing_outlines(self.states[name].folder)
        if error or result is None:
            self.segment_page.append(f"Segmentation failed for {name}: {error}")
            self._download_ok = False                  # a failed download is asked about again, not retried unasked
            self._run_lines.append(f"{name}: failed ({error})")
            self._run_failed = True
        else:
            self.segment_page.append(result["summary"])
            self._run_lines.append(f"{name}: {result['cells']} cell(s) in "
                                   f"{len(result['rows'])} field(s)")
            self.segment_page.finished(name)
            self.segment_page.set_progress(1.0, f"{name}: done, {result['cells']} cell(s) in "
                                                f"{len(result['rows'])} field(s)")
            if not self.editor.unsaved():
                self._rescan_editor()                  # the new outlines, in the editor
        self._next()

    def _next(self) -> None:
        while self._queue:
            if self._start(self._queue.pop(0)):
                return
        if self._run_lines:
            # the last queued condition is done: a pop-up when the run was long
            started, lines, failed = self._run_started, self._run_lines, self._run_failed
            show_finished(self, self.t, finished_payload(
                title=("Segmentation finished" if not failed
                       else "Segmentation finished with failures"),
                lines=lines, started=started, notify_after_s=self._notify_after,
                failed=failed))
        self._run_started, self._run_plan, self._run_done = None, {}, []
        self._run_lines, self._run_failed = [], False
        self._refresh()

    def stop(self) -> None:
        if self._stop is None:
            return
        self._stop.set()
        dropped, self._queue = len(self._queue), []
        self.segment_page.append(f"Stopping {self._running} after the field in hand"
                                 + (f"; {dropped} queued condition(s) dropped" if dropped else ""))
        self._refresh()

    # -- the batch over several parent folders (`batch_model`) ------------------------ #
    def choose_batch_parent(self) -> None:
        start = self._batch_parents[-1] if self._batch_parents else (self.folder or Path.home())
        folder = QFileDialog.getExistingDirectory(
            self, "A parent folder for the batch (it may hold several conditions)", str(start))
        if folder:
            self.add_batch_parent(folder)

    def add_batch_parent(self, folder) -> None:
        if self._batch_stop is not None:
            return
        self._batch_parents, note = add_parent(self._batch_parents, folder)
        self.batch_page.set_note(note)
        self.batch_page.set_parents(self._batch_parents)
        self._plan_batch()

    def remove_batch_parent(self, folder: str) -> None:
        if self._batch_stop is not None:
            return
        self._batch_parents = [p for p in self._batch_parents
                               if folder_key(p) != folder_key(folder)]
        self.batch_page.set_note("")
        self.batch_page.set_parents(self._batch_parents)
        self._plan_batch()

    def _batch_replace_changed(self, on: bool) -> None:
        self._batch_replace = bool(on)
        self._plan_batch()

    def _plan_batch(self) -> None:
        """List the parents and plan, on a thread; a newer request supersedes an older.
        Every condition is planned with the Options' standard model."""
        if self._batch_stop is not None:
            return
        self._batch_generation += 1
        self._batch_status, self._batch_summary = {}, ""
        if not self._batch_parents:
            self._batch_rows = []
            self._show_batch_rows()
            return
        generation, parents = self._batch_generation, list(self._batch_parents)
        model, replace = self._option("segment_model"), self._batch_replace
        ignored = set(self._batch_ignored)
        self.batch_page.set_summary("listing the parent folders…")

        def work():
            from ...batch_model import plan_outlines, scan_outlines

            labels = {name: words for name, _roles, words in models()}
            return {"rows": plan_outlines(scan_outlines(parents, model=model),
                                          replace=replace, model_labels=labels,
                                          ignored=ignored)}

        self._threads.start(_Call(work, lambda r, e: self._emit(
            self.signals.batch_planned, generation, r, e)))

    def _batch_planned(self, generation: int, got, error: str) -> None:
        if generation != self._batch_generation or self._batch_stop is not None:
            return
        if error or got is None:
            self.batch_page.set_summary(f"the parent folders could not be listed: {error}")
            return
        self._batch_rows = list(got["rows"])
        self._show_batch_rows()

    def _show_batch_rows(self) -> None:
        page = self.batch_page
        rows = self._batch_rows
        page.set_rows([{"key": r["key"], "ignored": r["key"] in self._batch_ignored,
                        "cells": outline_batch_cells(r, self._batch_status.get(r["key"]))}
                       for r in rows])
        page.set_layout_lines(layout_notes(rows))
        page.settings_note.setText(
            f"Model for every condition: {Path(str(self._option('segment_model'))).name} "
            f"(Options); "
            + ("the GPU is used where the Segment page ticks it."
               if self._can_run(self._option("segment_model"))
               else "this build cannot run the models. Pick the classical segmenter in "
                    "Options, or see the Segment page."))
        if self._batch_stop is None:
            page.set_summary(self._batch_summary or outline_summary(rows))
        page.set_busy(self._batch_stop is not None,
                      can_run=(bool(self._batch_parents)
                               and self._can_run(self._option("segment_model"))
                               and any(r["ready"] for r in rows)))

    def run_batch(self) -> None:
        """Segment the missing fields of every ready condition under the parents, one
        condition at a time (`batch_model.run_outlines`). The folders are listed again as
        the run starts, so it runs what is on disk NOW."""
        page = self.batch_page
        if self._batch_stop is not None:
            return
        if not self._batch_parents:
            page.set_note("Add a parent folder first.")
            return
        if self._running is not None:
            page.set_note(f"Segmenting {self._running}. The batch can start once it is done.")
            return
        if not self._can_run(self._option("segment_model")):
            page.set_note("This build cannot run the models. Pick the classical segmenter in "
                          "Options, or see the Segment page.")
            return
        use_gpu = bool(self.segment_page.gpu.checked and self.segment_page.gpu.isEnabled())
        if not self._download_agreed("the batch", self._option("segment_model")):
            page.set_note("The batch was not started: Cellpose-SAM's model file was not downloaded.")
            return
        parents, model, replace = (list(self._batch_parents), self._option("segment_model"),
                                   self._batch_replace)
        ignored = set(self._batch_ignored)
        self._batch_stop = threading.Event()
        stop = self._batch_stop.is_set
        classical = self._classical
        self._batch_plan, self._batch_running, self._batch_finished = [], [], []
        self._batch_counts, self._batch_status, self._batch_results = {}, {}, {}
        self._batch_started = time.monotonic()
        # KEPT AWAKE FROM THIS THREAD, which outlives the batch; cleared when it ends.
        keep_awake(True)
        page.set_note(" ".join(filter(None, ("Running.", awake_note()))))
        page.set_busy(True)

        def event(kind, key, payload):
            self._emit(self.signals.batch_event, kind, key, payload)

        def work():
            from ...batch_model import run_outlines, scan_outlines

            labels = {name: words for name, _roles, words in models()}
            return run_outlines(scan_outlines(parents, model=model), use_gpu=use_gpu,
                                replace=replace, stop=stop, on_event=event,
                                model_labels=labels, ignored=ignored, classical=classical)

        self._threads.start(_Call(work, lambda r, e: self._emit(
            self.signals.batch_done, r, e)))
        self.activity.setText("A batch is running. See Batch.")
        self._refresh_progress()

    def _show_classical(self) -> None:
        """The Classical settings page: the open conditions, each field's sum image."""
        conditions = {}
        for name, cond in self.states.items():
            resolution = cond.resolution

            def path_of(field, resolution=resolution, folder=Path(cond.folder)):
                # the condition's own sum folder when its model reads another channel
                found = resolution.image_path("sum", field) if resolution is not None else None
                if found is None:
                    for suffix in (".tif", ".tiff"):
                        candidate = folder / "sum" / f"{field}{suffix}"
                        if candidate.is_file():
                            return candidate
                return found
            # THE FIELDS WITH A SUM IMAGE, whatever the condition's model reads
            sum_dir = Path(cond.folder) / "sum"
            stems = sorted(q.stem for q in sum_dir.glob("*.tif*")) if sum_dir.is_dir() else []
            fields = [f for f in cond.fields if path_of(f) is not None]
            fields += [f for f in stems if f not in fields]
            conditions[name] = (fields, path_of)
        self.classical_page.set_conditions(conditions, current=self.condition)

    def _classical_saved(self, saved: dict) -> None:
        from ...segmentation.classical_view import SETTINGS_KEY, params_from, words

        self._classical = dict(saved)
        try:
            theme.remember(SETTINGS_KEY, self._classical)
        except OSError:
            pass
        self.activity.setText("Classical settings saved: " + words(params_from(self._classical)))

    def set_condition_ignored(self, name: str, ignored: bool) -> None:
        """The Segment page's Ignore / Include: the same remembered list as the batch's, keyed
        by the condition's folder, so a condition left out here is left out there too."""
        cond = self.states.get(name)
        if cond is None or name == self._running:
            return
        key = folder_key(cond.folder)
        (self._batch_ignored.add if ignored else self._batch_ignored.discard)(key)
        try:
            theme.remember(BATCH_IGNORED_KEY, sorted(self._batch_ignored))
        except OSError:
            pass
        self._refresh()

    def set_batch_ignored(self, key: str, ignored: bool) -> None:
        """Leave a condition out of the batch, or take it back in; remembered per user."""
        if self._batch_stop is not None:
            return
        (self._batch_ignored.add if ignored else self._batch_ignored.discard)(key)
        try:
            theme.remember(BATCH_IGNORED_KEY, sorted(self._batch_ignored))
        except OSError:
            pass
        self._plan_batch()

    def stop_batch(self) -> None:
        """Stop after the field in hand; conditions not started are reported so."""
        if self._batch_stop is None:
            return
        self._batch_stop.set()
        self.batch_page.set_note("Stopping: the field in hand finishes, then nothing new "
                                 "starts.")

    def _batch_event(self, kind: str, key: str, payload) -> None:
        if self._batch_stop is None:
            return
        page = self.batch_page
        if kind == "planned":
            self._batch_plan = [dict(r) for r in payload]
            self._batch_rows = [dict(r) for r in payload]
            page.append("", *plan_lines(self._batch_rows, what="segmentation"))
        else:
            labels = {r["key"]: r["label"] for r in self._batch_plan}
            totals = {r["key"]: int(r["todo"]) for r in self._batch_plan}
            if kind == "started":
                self._batch_running.append(key)
                self._batch_counts[key] = 0
                self._batch_status[key] = running_status(0, totals.get(key, 0), unit="field")
            elif kind == "field":
                self._batch_counts[key] = int(payload["index"])
                self._batch_status[key] = running_status(
                    int(payload["index"]), int(payload["total"]), unit="field")
            elif kind == "finished":
                if key in self._batch_running:
                    self._batch_running.remove(key)
                self._batch_finished.append(key)
                self._batch_results[key] = dict(payload)
                self._batch_status[key] = result_status(payload)
            line = event_line(kind, labels.get(key, key), payload)
            if line:
                page.append(line)
        self._show_batch_rows()
        self._refresh_progress()

    def _batch_done(self, rows, error: str) -> None:
        if self._batch_stop is None:
            return
        self._batch_stop = None
        keep_awake(False)
        if error or rows is None:
            rows = list(self._batch_results.values())
            self.batch_page.append(f"The batch failed: {error}")
        summary = finished_summary(rows, what="Segmentation")
        if error:
            summary["failed"] = True
        self.batch_page.append("", summary["title"], *summary["lines"])
        self.batch_page.set_note(summary["title"])
        self.activity.setText(f"{summary['title']}. See Batch.")
        self._batch_summary = f"{summary['title']}: {summary['lines'][0]}"
        started, self._batch_started = self._batch_started, None
        self._show_batch_rows()
        # A CONDITION OPEN HERE MAY HAVE BEEN SEGMENTED: its counts, and the editor's
        # fields, must see the new outlines.
        self._outlines_saved()
        if self.folder is not None and not self.editor.unsaved():
            self._rescan_editor()
        show_finished(self, self.t, finished_payload(
            title=summary["title"], lines=summary["lines"], started=started,
            notify_after_s=self._notify_after, failed=summary["failed"]))

    # -- elsewhere ------------------------------------------------------------------- #
    def open_pack(self):
        """The segmentation pack's window (`pack_dialog`): the offer, or where it is installed."""
        from .pack_dialog import PackDialog

        dialog = PackDialog(self.t, content=self.centralWidget(), parent=self)
        dialog.installed.connect(lambda _folder: self._pack_changed())
        dialog.removed.connect(self._pack_changed)
        dialog.closed.connect(dialog.deleteLater)
        self._pack_dialog = dialog
        dialog.open()
        return dialog

    def _pack_changed(self) -> None:
        """A pack was installed (its folder is on sys.path now) or removed: the models, the GPU box and the note follow."""
        from ...segmentation import backends
        from ...segmentation.outline_model import runtime_words

        backends.clear_model_cache()
        why = runtime_words()
        self.segment_page.set_runtime(not why, why, offer=segpack.offered_here())
        self._models = models()
        if self.states:
            self._refresh()
            self._threads.start(_Call(self._probe_gpu, lambda r, e: self._emit(
                self.signals.gpu, *(r or (False, e or "unknown")))))
        self.activity.setText("Segmentation pack changed: " + ("Cellpose-SAM is offered." if not why
                                                               else "no model can run until it is downloaded."))

    def open_trainer(self) -> None:
        from ...segmentation import backends

        if segpack.offered_here() and not backends.cellpose_importable():
            self.activity.setText("Training needs PyTorch and Cellpose. Download them from Segmentation pack, "
                                  "or go on to export a training set.")
            self.open_pack()
            return
        from ...segmentation.trainer_gui import launch as launch_trainer

        self._trainers.append(launch_trainer(str(self.folder) if self.folder else None,
                                             show=True, block=False))

    def back_to_menu(self) -> None:
        if self._refuse("leaving"):
            return
        self.return_to_menu = True
        self.close()

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._stop is not None:
            self._stop.set()                           # after the field in hand
        if self._batch_stop is not None:
            self._batch_stop.set()                     # the same, and the machine may sleep
            keep_awake(False)
        self._queue = []
        self._progress_timer.stop()
        self.editor._close_event(event)                # asks about unsaved edits
        if event.isAccepted():
            self.editor.shutdown()
            _OPEN_WINDOWS.discard(self)
        super().closeEvent(event) if event.isAccepted() else None


def open_outlines_window(root=None, *, section: str = "segment", color_scheme=None,
                         show: bool = True, threads=None):
    """Build the window, show it, and open ``root`` AFTER it is on screen."""
    app = QApplication.instance() or QApplication([])
    from ...appicon import apply as _apply_icon

    _apply_icon()
    t = theme.current(color_scheme)
    window = OutlinesWindow(t, section=section, threads=threads)
    _OPEN_WINDOWS.add(window)
    # 1360 x 900 where the screen holds it, maximised where it does not, the minimum held to the screen
    # (`ui.widgets.screenfit`, 30 Oct 2026: it opened 88 px past a 1536 x 816 laptop's taskbar)
    from ..widgets.screenfit import show_fitted

    if show:
        show_fitted(window)
    if root is not None:
        QTimer.singleShot(0, window, lambda: window.open_folder(root))
    return app, window
