"""The segmentation STAGE's window: open a folder, pick a model, run, then correct.

THIN QT, the house rule: no numpy, no pandas, no scipy at module level and no geometry
anywhere. Every number on screen came from `segment.py` or `channels.py`, both pure and
covered headless, and the run happens on a worker thread so the console fills as it goes.

The shape is the experimenter's, recorded in `docs/segmentation_integration.md`:

    [Open folder]  [Train model]
    ... what was found: channels, and whether meshes already exist
    [Correct by hand]            -> the mesh editor, no segmentation
    -- automated ------------------------------------------------
    channels   detected, names editable
    model      the two pretrained, plus any pointed at
    [Run]      CPU always; GPU only where it is actually available
    console    errors AND the QC
    [Open mesh editor]  [Finish]

Two layouts, both real. A folder holding `sum/` and `phase/` is one condition; a
folder holding `glucose/` and `malate/` is a comparison. Both are opened; the second
lists its conditions and segments the one selected, because a model and a channel choice
are per-condition decisions and running four at once would hide which one went wrong.

The channel is never guessed. A role whose folder is not recognised waits to be
pointed at. Measured, and it is why: a DNA-bound protein segmented on its own
fluorescence returns outlines at 48 % of the true cell area, plausible-looking, and a
cell filter built on them drops exactly the peripheral tracks the spatial claims rest on.

GPU is offered only if it is there. The shipped installer carries CPU torch by
decision (`docs/segmentation_integration.md`), so on most machines the box is disabled
and says so. An enabled control that cannot work is worse than an absent one.
"""
from __future__ import annotations

__all__ = ["SegmentWindow", "launch", "describe_folder"]

#: Shown in a condition's folder picker when nothing resolved. NOT the first folder:
#: `resolve_channels` refuses to choose between candidates on purpose, and a picker that
#: quietly answered for it would hand the run an override that is never second-guessed.
CHOOSE_FOLDER = "(set the folder)"

#: Last entry in a condition's folder picker. The list offers the sub-folders that are
#: there, which is the ordinary case; this is for images that live somewhere else or
#: under a name nothing recognises. An override is never second-guessed, so what is
#: chosen here is what the run reads.
BROWSE_FOLDER = "choose another folder…"

#: Last entry in a condition's model picker. A model that is not one of the two
#: pretrained ones cannot appear in a list until somebody points at it.
ADD_MODEL = "add a model…"

#: What the window shows about a folder before anything is run. Pure, so the summary a
#: user reads is the same one a test asserts.
def describe_folder(root, roles=("sum",), overrides=None):
    """`(resolution, conditions, lines)` for a folder, without opening anything.

    `conditions` is non-empty when the folder holds condition sub-folders rather than
    channels: the comparison layout. `lines` is what goes in the console.
    """
    from pathlib import Path

    from .channels import folders_in, resolve_channels

    root = Path(root)
    lines = [f"opened {root}"]
    resolution = resolve_channels(root, roles, overrides=overrides)

    # A CONDITION FOLDER IS ONE WHOSE CHILDREN HAVE CHANNELS. Checked by looking, not by
    # depth: `grouping.discover_conditions` makes the same distinction for the analysis
    # and the folders above a movie carry no meaning in this project.
    conditions = []
    if not resolution.resolved:
        for child in folders_in(root):
            if resolve_channels(child, roles).resolved:
                conditions.append(child)
        if conditions:
            lines.append(f"this holds {len(conditions)} condition(s): "
                         + ", ".join(c.name for c in conditions))
            lines.append("set each one's model and images in the table, then segment "
                         "the selected one or all of them")
            return resolution, conditions, lines

        # NOT A CHANNEL PROBLEM, A LAYOUT ONE. A session straight off the microscope is
        # several sibling folders of LOOSE files, `series001.nd2` beside `series001.tif`,
        # and this stage needs a `phase/` folder per condition. Reporting "no folder is
        # named like 'phase'; set the folder in the table, one of: condition_3 ..." offers the
        # folders as if they were candidate channels, which is the wrong question. The
        # organise step is what files them, and it is the one step where a mistake is
        # silent, so it is named rather than guessed at here.
        try:
            from ..organise import needs_organising

            loose = needs_organising(root)
        except Exception:                                 # noqa: BLE001 - advisory only
            loose = []
        if loose:
            lines.append(
                f"{len(loose)} folder(s) hold loose files, not channel folders: "
                + ", ".join(loose))
            lines.append("These folders are not organised yet: the acquisitions and their phase "
                         "images sit side by side. Run the organise step first (the "
                         "tracking window's folder panel). It files .nd2 into raw/ "
                         "and .tif into phase/, and then they appear as conditions.")
            return resolution, conditions, lines

    lines.append("channels:")
    lines.extend(resolution.describe().splitlines())
    if resolution.resolved:
        from .channels import common_fields, missing_fields

        fields = common_fields(resolution)
        lines.append(f"{len(fields)} field(s) present in every channel")
        for role, absent in missing_fields(resolution).items():
            lines.append(f"  {role} is missing {len(absent)}: "
                         f"{', '.join(absent[:5])}{' ...' if len(absent) > 5 else ''}")
    from ..cellmesh import mesh_folder

    existing = mesh_folder(root)
    if existing.is_dir():
        n = len(list(existing.glob("*.mat")))
        lines.append(f"cell_meshes/ already holds {n} file(s): running will overwrite "
                     f"them; 'Correct by hand' opens them instead")
    else:
        lines.append("no cell_meshes/ yet")
    return resolution, conditions, lines


#: What every segmentation run needs. Checked by NAME, never by importing: the point is to answer before
#: anything heavy loads. The ENGINES are checked apart (`backends.available_backends`).
RUNTIME_MODULES = ("numpy", "scipy", "skimage", "tifffile")


def runtime_available() -> tuple[bool, str]:
    """`(available, why not)` for the segmentation MODELS (Cellpose-SAM, Omnipose).

    The LITE build carries no torch (ruled 26 Nov 2026), so there this window opens and its Run
    button cannot work for a model; the NORMAL build carries Cellpose 4. An enabled control that
    cannot work is worse than an absent one, the same rule the GPU box follows, so Run is
    disabled and says what is missing, in plain words and never with an offer to install.

    By `find_spec`, so asking costs nothing: importing the stack is the ~2 s this window
    was just taught not to do.
    """
    import importlib.util

    from . import backends

    missing = []
    for name in RUNTIME_MODULES:
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append(name)
    if missing:
        return False, ("this build is missing " + ", ".join(missing))
    if len(backends.available_backends()) > 1:
        return True, "a segmentation model is present"
    from .outline_model import runtime_words

    return False, runtime_words()


def gpu_available() -> tuple[bool, str]:
    """`(available, why not)`. Never assumed: the box is disabled on the answer.

    Never call this while building a widget. `import torch` is about 2 s from source
    and can be far worse frozen or on a cold disk, and `cuda.is_available()` probes the
    driver. Running it in `__init__` meant the window did not appear until it finished,
    which from outside is indistinguishable from a hang, reported as "loading for 10
    years and nothing is opening". The window asks for it on a thread AFTER it is up.
    """
    try:
        import torch
    except ImportError:
        return False, "torch is not installed in this application"
    try:
        if torch.cuda.is_available():
            return True, torch.cuda.get_device_name(0)
    except Exception as exc:                              # noqa: BLE001 - reported
        return False, f"{type(exc).__name__}: {exc}"
    return False, gpu_reason(getattr(getattr(torch, "version", None), "cuda", None),
                             getattr(torch, "__version__", ""))


def gpu_reason(cuda_build, version: str = "") -> str:
    """Why no GPU is offered when torch sees none, from what torch was BUILT with (``torch.version.cuda``; None for a
    CPU-only build). Found 3 Nov 2026 (tester B18): on a machine with a GTX 1650 and torch 2.13.0+cpu the line said
    "no CUDA device is visible", blaming the machine; the cause was the CPU-only torch, as the trainer window said."""
    if not cuda_build:
        named = f" ({version})" if version else ""
        return (f"the torch here is CPU-only{named}, so it cannot use a GPU even if the machine has one; "
                "segmenting runs on the CPU")
    return (f"torch is built for CUDA {cuda_build}, but it sees no CUDA device (no NVIDIA GPU, or its driver is "
            "too old for this CUDA); segmenting runs on the CPU")


#: Windows this module has opened, and the reason it must hold them.
#:
#: **A TOP-LEVEL PySide6 WINDOW WITH NO PYTHON REFERENCE IS DESTROYED AT ONCE.** The
#: moment `launch`'s caller discards the return value its refcount hits zero, CPython
#: deletes the wrapper, and the C++ widget goes with it, taking any QThread it owns.
#: MEASURED on the real route (`smtracker` -> Segmentation): "QThread: Destroyed while
#: thread '' is still running", then a **segfault, exit 139**, reported from a terminal.
#: Qt does NOT own a parentless widget on Python's behalf, and being visible does not
#: save it.
#:
#: THE DOOR OWNS WHAT IT OPENS, rather than every caller having to remember: the launch
#: sites in `app.py` all discarded, and so would the next one. Dropped again in
#: `closeEvent`, so a session that opens several windows does not accumulate them.
_OPEN_WINDOWS: list = []


def _keep(window):
    """Hold `window` until it closes. See `_OPEN_WINDOWS`."""
    if window not in _OPEN_WINDOWS:
        _OPEN_WINDOWS.append(window)
    return window


def _release(window) -> None:
    """Drop `window` on close, by IDENTITY: a QWidget's `==` can be overridden."""
    for i, held in enumerate(_OPEN_WINDOWS):
        if held is window:
            del _OPEN_WINDOWS[i]
            return


def launch(root=None, *, show: bool = True):
    """Open the segmentation window. Returns it.

    A `QApplication` is made if none exists: constructing a widget before one does
    ABORTS the process, with no exception to catch. `smtracker` with no arguments once
    died at exit 127 having printed nothing, for exactly that.
    """
    from qtpy.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    from ..appicon import apply as _apply_icon

    _apply_icon()
    window = _keep(_build_window_class()(root))
    if show:
        from ..ui.widgets.screenfit import show_fitted   # never past the screen (30 Oct 2026)

        show_fitted(window)
    return window


def _build_window_class():
    """Qt imported inside, so this module imports on a machine with no GUI stack."""
    from qtpy.QtCore import QObject, Qt, Signal, QThread
    from qtpy.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QGroupBox,
                                QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                                QListWidget, QMainWindow, QMessageBox, QPlainTextEdit,
                                QPushButton, QTableWidget, QVBoxLayout, QWidget)

    class _Worker(QObject):
        """Runs the segmentation off the GUI thread and reports each field.

        ON A THREAD because a thirty-field run is half a minute and a frozen window
        during it reads as a crash. Nothing here touches a widget: it emits, and the
        window does the drawing: the same rule as everywhere else in this project.
        """

        line = Signal(str)
        done = Signal(object)
        failed = Signal(str)

        def __init__(self, resolution, out_dir, cfg, fields, conditions=None,
                     roles=None, configs=None, overrides=None):
            super().__init__()
            self._resolution, self._out = resolution, out_dir
            self._cfg, self._fields = cfg, fields
            #: When set, every condition is run in turn and `resolution`/`out_dir`
            #: are ignored. `configs` gives each condition its OWN model, and its
            #: channels follow that model, because the two pretrained models read
            #: different IMAGES rather than being different weights over one.
            self._conditions, self._roles = conditions, roles
            self._configs = configs
            self._overrides = overrides

        def run(self):
            try:
                from .segment import segment_folder

                def progress(done, total, result):
                    detail = (f"{result.cells} cells from {result.masks_found} masks"
                              if result.ok else f"failed: {result.error}")
                    if result.rejected:
                        detail += "  (rejected: " + ", ".join(
                            f"{n} {why}" for why, n in result.rejected.items()) + ")"
                    self.line.emit(f"[{done}/{total}] {result.field}: {detail}"
                                   f"   {result.seconds:.1f}s")
                    # A SHIFT IS SILENT EVERYWHERE ELSE. Every downstream number still
                    # computes when outlines sit off the cells; the filter just keeps the
                    # wrong localisations. So it is said on its own line, and only when
                    # it is bad news, a per-field "aligned" for thirty fields is noise
                    # that teaches the reader to skip the console.
                    if getattr(result, "alignment", "") and result.alignment_shift_px:
                        if not result.alignment.startswith("aligned"):
                            self.line.emit(f"     ⚠ {result.field}: {result.alignment}")

                if self._conditions:
                    from .segment import segment_conditions

                    def each(done, total, name):
                        self.line.emit(f"=== [{done}/{total}] {name} done")

                    def per_field(name, fdone, ftotal, result):
                        self.line.emit(f"  {name}  ")
                        progress(fdone, ftotal, result)

                    report = segment_conditions(
                        self._conditions, self._roles, cfg=self._cfg,
                        configs=self._configs, overrides=self._overrides,
                        progress=each, field_progress=per_field)
                else:
                    report = segment_folder(self._resolution, self._out, self._cfg,
                                            fields=self._fields, progress=progress)
                self.done.emit(report)
            except Exception as exc:                      # noqa: BLE001 - reported
                import traceback

                self.failed.emit(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")

    class SegmentWindow(QMainWindow):
        """Segment a folder into cell outlines, then correct them."""

        def __init__(self, root=None, parent=None):
            super().__init__(parent)
            from pathlib import Path

            from .segment import builtin_models
            PRETRAINED = builtin_models()

            from ..buildstamp import build_stamp

            self.setWindowTitle(f"SMTracker · Cell segmentation  ·  {build_stamp()}")
            self.resize(920, 760)
            self._root = Path(root) if root else None
            self._resolution = None
            self._conditions = []
            self._overrides = {}
            self._thread = None
            self._worker = None
            self._report = None
            self._runtime = None
            self._gpu_thread = None
            self._gpu_probe = None

            central = QWidget()
            layout = QVBoxLayout(central)

            top = QHBoxLayout()
            self._open = QPushButton("Open folder…")
            self._open.clicked.connect(self._pick_folder)
            # NAMED FOR BOTH JOBS. It read "Train on my corrections", which names
            # only the fine-tune -- so somebody looking for "train a new model"
            # could not find it, and asked. The window offers both; the door has
            # to say so.
            self._train = QPushButton("Train a model…  (fine-tune, or a new one)")
            self._train.setToolTip(
                "Open the model trainer on this folder. You can fine-tune an existing "
                "model or train a new one from scratch.\n\n"
                "Your corrected outlines are the training labels; there is no separate "
                "labelling step. Correct some cells first.")
            self._train.clicked.connect(self._train_clicked)
            top.addWidget(self._open)
            top.addStretch(1)
            layout.addLayout(top)

            self._conditions_list = QListWidget()
            # SIZED TO WHAT IS IN IT, not capped at three rows' worth. A fixed 90 px
            # put a scrollbar on a list of three conditions -- the controls that decide
            # the run were the ones you had to scroll, while the console below them took
            # every spare pixel. `_fit_list` grows it to the contents up to a sensible
            # ceiling, past which scrolling is genuinely the right answer.
            self._conditions_list.setMinimumHeight(64)
            self._conditions_list.currentTextChanged.connect(self._pick_condition)
            self._conditions_label = QLabel("conditions in this folder")
            layout.addWidget(self._conditions_label)
            layout.addWidget(self._conditions_list)
            self._conditions_label.hide()
            self._conditions_list.hide()

            # A MODEL PER CONDITION. Asked for: "we need a picker which model for each
            # of the condition to use, with an option to copy for all like we had
            # before". The batch took ONE model for every condition and said so in
            # capitals; that was never a property of the batch, only of this control.
            # `describe_folder` has always called the model a per-condition decision,
            # and phase against fluorescence is exactly that decision.
            #
            # Same shape as the tracking window's grids, which are the experimenter's
            # own design: one row, one column per condition, and the copy button under
            # the FIRST column, copying the column it sits under.
            self._model_table = QTableWidget(1, 1)
            # EXACTLY ITS ROWS. Two rows plus a header do not fit in 72 px, so the
            # model and the folder -- the two things the run is made of -- arrived behind
            # a scrollbar.
            self._model_table.setSizeAdjustPolicy(
                QTableWidget.AdjustToContents)
            self._model_table.verticalHeader().setVisible(False)
            self._model_table.horizontalHeader().setStretchLastSection(True)
            self._model_table.hide()
            layout.addWidget(self._model_table)
            #: {target name: model} and {target name: folder name}. The table SHOWS
            #: what these hold, so there is one place a run's inputs come from.
            self._condition_models = {}
            self._condition_folders = {}
            #: True while `_refresh_model_table` is replacing the cell widgets.
            #: `setCellWidget` DESTROYS the widget it replaces, and a QComboBox being
            #: torn down emits `currentTextChanged` -- so a redraw called `_folder_picked`
            #: with the old value and marked folders as hand-chosen that nobody chose.
            #: Visible only as a log line arriving one step late, which is how it was
            #: found. Blocking the TABLE's signals does not help: the combos are their
            #: own widgets.
            self._rebuilding = False
            #: Conditions whose folder was CHOSEN rather than derived. It survives a
            #: redraw of the table and NOT a model change: the model decides which image
            #: is read, and a correction that outlived it would run a fluorescence model
            #: over phase contrast, which is the failure the per-condition model exists
            #: to prevent.
            self._folder_chosen = set()

            # SEGMENT THEM ALL. Asked for: "it would be cool if multiple folders would
            # be detected and analysed at the same time. a parent folder name could
            # differentiate between them". The per-condition route stays, because
            # `describe_folder` is right that a model and a channel mapping are
            # per-condition decisions: this button is for the ordinary case where a
            # parent holds the same experiment under several carbon sources.
            self._run_all = QPushButton("Segment all conditions")
            self._run_all.setToolTip(
                "Run every condition in this folder, each with the model and the images "
                "set for it in the table. They do not have to match.")
            self._run_all.clicked.connect(self._start_all)
            self._run_all.hide()
            layout.addWidget(self._run_all)

            self._by_hand = QPushButton("Correct by hand  (open the mesh editor)")
            self._by_hand.clicked.connect(self._open_editor)
            layout.addWidget(self._by_hand)
            # AFTER CORRECTING, because that is the order of the work: segment, correct,
            # then train on what you corrected. It sat in the TOP row beside "Open
            # folder", i.e. before there was anything to train on -- which is how
            # somebody looking for it read past it.
            layout.addWidget(self._train)

            # THE BOX IS GONE. Ruled: "why do we need the automated segmentation
            # below? everything should be in the condition table." It held a model and a
            # channel mapping for ALL conditions, beneath a table that already says both
            # per condition -- two controls for one pair of numbers, and the lower one
            # silently outranked the grid above it.
            #
            # The WIDGETS are kept, hidden, as the model behind the table: `_pick_model`
            # writes the loaded model into `self._model`, and the table offers whatever
            # it holds. Same move as the tracking window's condition combo -- a control
            # is hidden, not a value.
            box = QWidget()
            inner = QVBoxLayout(box)
            inner.setContentsMargins(0, 0, 0, 0)

            channel_row = QHBoxLayout()
            self._model = QComboBox()
            self._model.addItems(list(PRETRAINED))
            self._model.currentTextChanged.connect(self._model_changed)
            self._model.hide()
            self._roles = QLineEdit()
            self._roles.hide()

            run_row = QHBoxLayout()
            self._run = QPushButton("Run segmentation")
            self._run.clicked.connect(self._start)
            run_row.addWidget(self._run)
            # LOADING A MODEL IS STILL A BUTTON, because a model that is not one of the
            # two pretrained ones cannot appear in a dropdown until somebody points at
            # it. It then joins every condition's picker.
            self._load_model = QPushButton("Load model…")
            self._load_model.clicked.connect(self._pick_model)
            run_row.addWidget(self._load_model)
            # DISABLED UNTIL THE ANSWER ARRIVES, and the answer is fetched on a thread
            # after the window is up. Asking here would import torch before the first
            # paint; see `gpu_available`.
            self._gpu = QCheckBox("use GPU: checking…")
            self._gpu.setEnabled(False)
            self._gpu.setToolTip("looking for a CUDA device")
            run_row.addWidget(self._gpu)
            run_row.addStretch(1)
            inner.addLayout(run_row)
            layout.addWidget(box)

            self._console = QPlainTextEdit()
            self._console.setReadOnly(True)
            self._console.setStyleSheet("font-family: Consolas, monospace;")
            # THE CONSOLE GIVES WAY. Ruled: "console space can be reduced". It still
            # takes the leftover room, but it no longer insists on 160 px before the
            # controls above it get theirs.
            self._console.setMinimumHeight(90)
            layout.addWidget(self._console, 1)

            bottom = QHBoxLayout()
            self._to_editor = QPushButton("Open mesh editor")
            self._to_editor.clicked.connect(self._open_editor)
            self._finish = QPushButton("Finish")
            self._finish.clicked.connect(self.close)
            bottom.addStretch(1)
            bottom.addWidget(self._to_editor)
            bottom.addWidget(self._finish)
            layout.addLayout(bottom)

            self.setCentralWidget(central)
            self._start_gpu_check()
            self._model_changed(self._model.currentText())
            if self._root is not None:
                self._scan(self._root)
            else:
                self._say("open a folder to begin")

        # ── plumbing ─────────────────────────────────────────────────────────
        def _start_gpu_check(self) -> None:
            """Answer the GPU question off the GUI thread, then set the box."""
            from qtpy.QtCore import QThread

            class _Probe(QObject):
                answered = Signal(bool, str)

                def run(self):
                    self.answered.emit(*gpu_available())

            self._gpu_thread = QThread(self)
            self._gpu_probe = _Probe()
            self._gpu_probe.moveToThread(self._gpu_thread)
            self._gpu_thread.started.connect(self._gpu_probe.run)
            self._gpu_probe.answered.connect(self._gpu_answered)
            # QUITTING THE APP WITH THE WINDOW OPEN JOINS IT TOO, not only closing the window: the window stays
            # held (`_OPEN_WINDOWS`) until the interpreter exits, and if the torch import is still running then,
            # the window and its RUNNING thread are destroyed together -- abort, 0xC0000409, nothing printed.
            # Seen 19 Oct 2026 in `test_launch_survives_its_caller_discarding_it` on a loaded machine (1 in 3),
            # where the import outlasts the test's 400 ms.
            app = QApplication.instance()
            if app is not None:
                app.aboutToQuit.connect(self._stop_gpu_thread)
            self._gpu_thread.start()

        def _gpu_answered(self, available: bool, why: str) -> None:
            self._gpu.setEnabled(available)
            self._gpu.setText("use GPU" if available
                              else f"use GPU: unavailable ({why})")
            self._gpu.setToolTip(f"CUDA: {why}" if available else why)
            self._stop_gpu_thread()

        def _runtime_ok(self) -> bool:
            """Run is only offered where it can actually run. Cached, since a rescan
            asks again and the answer cannot change inside one session."""
            if self._runtime is None:
                self._runtime = runtime_available()
            return self._runtime[0]

        def _stop_gpu_thread(self) -> None:
            """Join the probe. **A QThread destroyed while running calls `abort()`**,
            no exception, no traceback, exit 127, the same silent death this project
            already records for a dialog built with no QApplication. Closing the window
            within the ~2 s import therefore has to WAIT rather than let Qt tear the
            thread down underneath itself.
            """
            if self._gpu_thread is not None:
                self._gpu_thread.quit()
                self._gpu_thread.wait(10000)
                self._gpu_thread = None
            self._gpu_probe = None

        def _say(self, *lines) -> None:
            for line in lines:
                self._console.appendPlainText(str(line))

        def _model_changed(self, name: str) -> None:
            """The model declares its channels; the box follows unless edited.

            "Auto" is only honest if the model carries what it needs. A model trained on
            [phase, sum] is not interchangeable with one trained on sum alone, and
            handing either the wrong input produces masks rather than an error.
            """
            from .segment import builtin_models
            PRETRAINED = builtin_models()

            roles = PRETRAINED.get(name)
            if roles:
                self._roles.setText(", ".join(roles))
            self._rescan()

        def _roles_wanted(self) -> tuple:
            return tuple(r.strip() for r in self._roles.text().split(",") if r.strip())

        def _pick_folder(self) -> None:
            from pathlib import Path

            chosen = QFileDialog.getExistingDirectory(self, "Folder to segment")
            if chosen:
                self._overrides.clear()
                self._scan(Path(chosen))

        def _pick_condition(self, name: str) -> None:
            if not name:
                return
            for path in self._conditions:
                if path.name == name:
                    self._overrides.clear()
                    self._scan(path, keep_conditions=True)
                    return

        def _scan(self, root, *, keep_conditions: bool = False) -> None:
            self._root = root
            resolution, conditions, lines = describe_folder(
                root, self._roles_wanted(), overrides=self._overrides)
            self._resolution = resolution
            self._say("", *lines)
            if conditions and not keep_conditions:
                self._conditions = conditions
                self._conditions_list.clear()
                self._conditions_list.addItems([c.name for c in conditions])
                self._run_all.setText(
                    f"Segment ALL {len(conditions)} conditions")
                self._run_all.setVisible(len(conditions) > 1)
                self._conditions_label.show()
                self._conditions_list.show()
                self._fit_list()
                # THE GRID FOLLOWS THE CONDITIONS. Stale entries are dropped rather than
                # kept: a model remembered for a condition that is no longer loaded would
                # be handed to `segment_conditions`, which refuses a config naming a
                # condition not in the batch — correctly, and this is where it comes from.
                keep = {c.name for c in conditions}
                self._condition_models = {k: v for k, v in
                                          self._condition_models.items() if k in keep}
                self._condition_folders = {k: v for k, v in
                                           self._condition_folders.items() if k in keep}
            # OUTSIDE the branch, because a single condition gets the same one-column
            # table now that the box below it is gone. Inside, it was refreshed only for
            # a parent of conditions, so opening one folder left no controls at all.
            self._refresh_model_table()
            # RUN STAYS OFF WHILE A CHANNEL IS UNRESOLVED, which the first draft of the
            # table dropped: it enabled Run for any target at all, so the press reached
            # `_start`, which refused and printed why. A button that is on and then
            # declines is worse than one that is off with the reason above it.
            self._run.setEnabled(self._runtime_ok() and self._can_run_selected())
            if not self._runtime_ok():
                self._run.setToolTip(self._runtime[1])
                self._say("cannot run here: " + self._runtime[1],
                          "  'Correct by hand' still opens the editor, which needs "
                          "none of it.")
            if resolution.resolved and not resolution.ok:
                self._say("cannot run until every channel is resolved: set the folder "
                          "in the table above")

        def _rescan(self) -> None:
            if self._root is not None:
                self._scan(self._root, keep_conditions=True)

        def _point_channel(self) -> None:
            """Resolve a role by pointing at its folder. An override is never
            second-guessed: the person looking at the images knows what is in them."""
            from pathlib import Path

            if self._resolution is None:
                return
            unresolved = list(self._resolution.unresolved)
            if not unresolved:
                self._say("every channel is already resolved")
                return
            role = unresolved[0]
            chosen = QFileDialog.getExistingDirectory(
                self, f"Folder holding the {role!r} images",
                str(self._root) if self._root else "")
            if chosen:
                self._overrides[role] = Path(chosen)
                self._say(f"{role} -> {Path(chosen).name} (you chose this)")
                self._rescan()

        def _pick_model(self) -> None:
            from pathlib import Path

            chosen, _ = QFileDialog.getOpenFileName(self, "Trained model file")
            if not chosen:
                return
            # RECOGNISED FROM THE FILE, never guessed: Cellpose 4 or Omnipose, or refused by name
            from . import backends
            from .segment import SegmentationError

            try:
                kind = backends.checkpoint_kind(chosen)
                why = backends.backend_missing_words(kind)
            except SegmentationError as exc:
                self._say(f"Not loaded: {exc}")
                return
            if why:
                self._say(f"Not loaded: {Path(chosen).name} is an {backends.BACKEND_NAMES[kind]} model. {why}")
                return
            self._model.addItem(chosen)
            self._model.setCurrentText(chosen)
            self._say(f"model: {chosen} ({backends.BACKEND_NAMES[kind]} model)",
                      "Note: a model file does not record the channels it was "
                      "trained on, so check the channel box yourself.")

        # ── the run ──────────────────────────────────────────────────────────
        def _start(self) -> None:
            from .segment import SegmentConfig
            from .channels import common_fields

            # THE TABLE, not the boxes that used to sit below it. Which condition
            # this is comes from the list selection, or is the folder itself when one
            # was opened directly -- `_targets` makes those the same case.
            from .channels import resolve_channels

            name = self._selected_target()
            if name is None:
                return
            here = dict(self._targets())[name]
            model = self._condition_models.get(name, self._model.currentText())
            roles = self._roles_for(model)
            overrides = self._overrides_per_condition().get(name)
            resolution = resolve_channels(here, roles, overrides=overrides)
            if not resolution.ok:
                self._say(f"{name}: cannot run until every channel is resolved")
                return
            self._resolution = resolution
            fields = common_fields(resolution)
            if not fields:
                self._say("no fields present in every channel; nothing to do")
                return
            from ..cellmesh import mesh_folder

            out = mesh_folder(here)
            cfg = SegmentConfig(model=model, channels=roles,
                                use_gpu=self._gpu.isChecked())
            self._say("", f"running {cfg.model} on {name}: {len(fields)} field(s) "
                          f"from {'+'.join(p.name for p in resolution.resolved.values())}"
                          f"/ -> {out}",
                      "the first field also loads the model, so it is slower")
            self._run.setEnabled(False)
            self._open.setEnabled(False)

            self._thread = QThread(self)
            self._worker = _Worker(resolution, out, cfg, fields)
            self._worker.moveToThread(self._thread)
            self._thread.started.connect(self._worker.run)
            self._worker.line.connect(self._say)
            self._worker.done.connect(self._finished)
            self._worker.failed.connect(self._crashed)
            self._thread.start()

        def _fit_list(self) -> None:
            """Grow the conditions list to its contents, up to eight rows.

            Eight because past that a list IS a scrolling control and pretending
            otherwise pushes everything else off the window. Below it, a scrollbar over
            three conditions is the window asking you to hunt for a control it had room
            to show.
            """
            count = self._conditions_list.count()
            if not count:
                self._conditions_list.setMaximumHeight(0)
                return
            row = self._conditions_list.sizeHintForRow(0)
            if row <= 0:
                row = 22
            frame = 2 * self._conditions_list.frameWidth() + 4
            self._conditions_list.setMaximumHeight(row * min(count, 8) + frame)
            self._conditions_list.setMinimumHeight(row * min(count, 8) + frame)

        def _fit_table(self) -> None:
            """Show every row of the settings table without a scrollbar."""
            table = self._model_table
            height = table.horizontalHeader().height() + 2 * table.frameWidth() + 4
            for row in range(table.rowCount()):
                height += table.rowHeight(row)
            table.setMinimumHeight(height)
            table.setMaximumHeight(height)

        def _targets(self):
            """``[(name, path)]`` the table has a column for.

            The conditions when a parent was opened, otherwise the folder itself -- so a
            single condition gets the same one-column table rather than a different set
            of controls. The box that used to serve that case is gone.
            """
            if self._conditions:
                return [(c.name, c) for c in self._conditions]
            if self._root is not None:
                return [(self._root.name, self._root)]
            return []

        def _auto_folder(self, target, model: str):
            """The folder a model would read under `target`, or "" if it cannot tell."""
            from .channels import resolve_channels
            from .segment import builtin_models
            PRETRAINED = builtin_models()

            roles = PRETRAINED.get(model) or self._roles_wanted() or ("phase",)
            got = resolve_channels(target, roles)
            for role in roles:
                found = got.resolved.get(role)
                if found is not None:
                    return found.name
            return ""

        def _refresh_model_table(self) -> None:
            """Redraw the per-condition rows: which model, and which images it reads.

            RULED: "everything should be in the condition table. as well as point to
            folder that automatically is set to ... like we have before but changeable."
            So the folder is FILLED from what resolves and stays a dropdown of what is
            actually there -- never typed, because a folder that does not exist is a run
            that fails at the first field.
            """
            from .channels import folders_in
            from .segment import builtin_models
            PRETRAINED = builtin_models()

            targets = self._targets()
            if not targets:
                self._model_table.hide()
                return
            # EVERY MODEL THIS WINDOW KNOWS, which is what `self._model` holds: the
            # pretrained pair plus anything pointed at. Offering only the CURRENT one
            # meant a model just loaded for one condition was absent from the list, so
            # the cell fell back to the first entry and the choice was lost the instant
            # it was made -- and it could never be given to a second condition.
            choices = list(PRETRAINED)
            for index in range(self._model.count()):
                text = self._model.itemText(index)
                if text and text not in choices:
                    choices.append(text)
            for text in self._condition_models.values():
                if text and text not in choices:
                    choices.append(text)
            self._rebuilding = True
            try:
                self._fill_model_table(targets, choices)
            finally:
                self._rebuilding = False
            self._model_table.show()
            self._fit_table()

        def _fill_model_table(self, targets, choices) -> None:
            """The redraw itself. Split out so `_rebuilding` is released on any path."""
            from .channels import folders_in

            self._model_table.blockSignals(True)
            self._model_table.setRowCount(2)
            self._model_table.setColumnCount(len(targets) + 1)
            self._model_table.setHorizontalHeaderLabels(
                [""] + [name for name, _path in targets])
            self._model_table.setVerticalHeaderLabels(["model", "reads"])
            self._model_table.verticalHeader().setVisible(True)

            first = targets[0][0]
            copy_model = QPushButton("copy →")
            copy_model.setToolTip(f"Use {first}'s model everywhere.")
            copy_model.clicked.connect(self._copy_model_to_all)
            self._model_table.setCellWidget(0, 0, copy_model)
            copy_folder = QPushButton("copy →")
            copy_folder.setToolTip(
                f"Use the folder name of {first} everywhere. A condition without a folder "
                f"of that name keeps its own.")
            copy_folder.clicked.connect(self._copy_folder_to_all)
            self._model_table.setCellWidget(1, 0, copy_folder)

            for column, (name, path) in enumerate(targets, start=1):
                box = QComboBox()
                box.addItems(choices + [ADD_MODEL])
                chosen = self._condition_models.get(name, self._model.currentText())
                found = box.findText(str(chosen))
                if found >= 0:
                    box.setCurrentIndex(found)
                box.currentTextChanged.connect(
                    lambda text, who=name: self._model_picked(who, text))
                self._model_table.setCellWidget(0, column, box)
                # THE GRID IS THE TRUTH once it is on screen, so what it SHOWS is what is
                # recorded. Leaving a displayed value out of the model made "absent means
                # follow the box above" a second source of truth for the same number,
                # which is the shape this project has had to unpick three times.
                self._condition_models[name] = box.currentText()

                folders = [f.name for f in folders_in(path)]
                want = self._condition_folders.get(name) or self._auto_folder(
                    path, box.currentText())
                picker = QComboBox()
                # NOTHING RESOLVED MEANS A QUESTION, NOT THE FIRST FOLDER. `resolve_
                # channels` refuses to guess between two matches -- "two matches is a
                # question, not a coin toss" -- and defaulting this picker to whatever
                # sorts first would answer it on the user's behalf, then hand the run an
                # override that is never second-guessed. So an unresolved role shows
                # UNSET, Run stays off, and the console says where to set it.
                if not want:
                    picker.addItem(CHOOSE_FOLDER)
                picker.addItems(folders)
                # A FOLDER THAT IS ALREADY CHOSEN AND IS NOT A CHILD of this condition
                # still has to be shown, or the cell would display something the run is
                # not reading.
                if want and want not in folders:
                    picker.addItem(want)
                picker.addItem(BROWSE_FOLDER)
                at = picker.findText(str(want)) if want else 0
                if at >= 0:
                    picker.setCurrentIndex(at)
                picker.setToolTip(
                    "The images this model is run on. Filled from what resolves, and "
                    "changeable: an override is never second-guessed, including one "
                    "naming a folder whose name says something else.")
                picker.currentTextChanged.connect(
                    lambda text, who=name: self._folder_picked(who, text))
                self._model_table.setCellWidget(1, column, picker)
                if picker.currentText() not in (CHOOSE_FOLDER, BROWSE_FOLDER):
                    self._condition_folders[name] = picker.currentText()

            self._model_table.resizeColumnToContents(0)
            self._model_table.horizontalHeader().setSectionResizeMode(
                QHeaderView.Stretch)
            self._model_table.horizontalHeader().setSectionResizeMode(
                0, QHeaderView.ResizeToContents)
            self._model_table.blockSignals(False)

        def _target_resolution(self, name, path):
            """How `name`'s channels resolve under the model and folder set for it."""
            from .channels import resolve_channels

            model = self._condition_models.get(name, self._model.currentText())
            return resolve_channels(
                path, self._roles_for(model),
                overrides=self._overrides_per_condition().get(name))

        def _can_run_selected(self) -> bool:
            """Whether 'Run segmentation' has a target whose channels resolve.

            QUIET: this runs on every scan, so it never writes to the console. With one
            target the answer is that target; with several, whether ANY resolves, since
            which one is chosen in the list and the press names it.
            """
            targets = self._targets()
            if not targets:
                return False
            try:
                return any(self._target_resolution(name, path).ok
                           for name, path in targets)
            except Exception:                             # noqa: BLE001 - advisory
                return False

        def _selected_target(self):
            """The condition 'Run segmentation' acts on, by name.

            The list selection when a parent was opened; the folder itself otherwise. A
            run must name its target -- the tracking window learned that the hard way,
            where "this condition" meant whichever folder a panel happened to show.
            """
            targets = dict(self._targets())
            if not targets:
                return None
            if len(targets) == 1:
                return next(iter(targets))
            item = self._conditions_list.currentItem()
            chosen = item.text() if item is not None else None
            if chosen in targets:
                return chosen
            self._say("pick a condition in the list above, or use "
                      "'Segment ALL conditions'")
            return None

        def _folder_path(self, condition):
            """Where `condition` is read from, as a path, or ``None`` if unset.

            The cell holds a NAME for a sub-folder of the condition and a full PATH for
            one chosen elsewhere, because a session's conditions each have their own
            `phase/` and copying a name across them is the ordinary case -- copying one
            absolute path would point every condition at one folder.
            """
            from pathlib import Path

            folder = self._condition_folders.get(condition)
            if not folder:
                return None
            here = Path(folder)
            if not here.is_absolute():
                here = dict(self._targets()).get(condition, Path(".")) / folder
            return here if here.is_dir() else None

        def _folder_picked(self, condition: str, text: str) -> None:
            if self._rebuilding:
                return
            """Remember which images one condition is read from."""
            if not text or text == CHOOSE_FOLDER:
                return
            if text == BROWSE_FOLDER:
                self._browse_folder(condition)
                return
            if self._condition_folders.get(condition) == text:
                return
            self._condition_folders[condition] = text
            self._folder_chosen.add(condition)
            self._say(f"{condition}: reading {text}/ (until the model changes, which "
                      f"re-derives it)")

        def _browse_folder(self, condition: str) -> None:
            """Point one condition at a folder the list does not offer.

            ASKED FOR: "we need a folder picker if their folder is differently named."
            The list holds the sub-folders that ARE there, which covers the ordinary
            case; this covers images kept elsewhere, or under a name nothing recognises.
            `resolve_channels` never second-guesses an override, so what is chosen here
            is what the run reads.
            """
            from pathlib import Path

            start = dict(self._targets()).get(condition)
            chosen = QFileDialog.getExistingDirectory(
                self, f"Folder holding {condition}'s images",
                str(start) if start else "")
            if not chosen:
                # CANCEL PUTS THE CELL BACK. Leaving "choose another folder…" selected
                # would show a folder name nothing reads.
                self._refresh_model_table()
                return
            picked = Path(chosen)
            # RELATIVE WHERE IT CAN BE, so "copy to all" stays a name every condition
            # can have one of. Absolute only where the folder is genuinely elsewhere.
            target = dict(self._targets()).get(condition)
            try:
                if target is not None and picked.parent == Path(target):
                    self._condition_folders[condition] = picked.name
                else:
                    self._condition_folders[condition] = str(picked)
            except (OSError, ValueError):
                self._condition_folders[condition] = str(picked)
            self._folder_chosen.add(condition)
            self._refresh_model_table()
            self._say(f"{condition}: reading {picked} (you chose this, so it is used "
                      f"as-is)")

        def _add_model(self, condition: str) -> None:
            """Point at a trained model file and use it for this condition.

            ASKED FOR: "and a way to add another model here." It joins every condition's
            picker, so the next one can be given the same model without browsing again.
            """
            chosen, _ = QFileDialog.getOpenFileName(self, "Trained model file")
            if not chosen:
                self._refresh_model_table()
                return
            if self._model.findText(chosen) < 0:
                self._model.addItem(chosen)
            self._condition_models[condition] = chosen
            # A LOADED MODEL DECLARES NOTHING, so its folder cannot be derived and the
            # one already set stands. Said out loud, because every other model change
            # moves the folder and this one silently does not.
            self._refresh_model_table()
            self._say(f"{condition}: {chosen}",
                      "  a model file carries no record of the channels it was trained "
                      "on, so check the folder in the row below yourself")

        def _copy_folder_to_all(self) -> None:
            """The FIRST column's folder NAME onto every other target.

            BY NAME, not by path: the point of a session is that every condition has its
            own `phase/`. A target with no folder of that name keeps what it had, because
            an override naming nothing is a run that fails at the first field.
            """
            from .channels import folders_in

            targets = self._targets()
            if len(targets) < 2:
                return
            source = targets[0][0]
            want = self._condition_folders.get(source)
            if not want:
                return
            missing = []
            for name, path in targets:
                if want in {f.name for f in folders_in(path)}:
                    self._condition_folders[name] = want
                    self._folder_chosen.add(name)
                else:
                    missing.append(name)
            self._refresh_model_table()
            self._say(f"every condition now reads {want}/")
            if missing:
                self._say(f"  except {', '.join(missing)}, which have no {want}/: "
                          f"left as they were")

        def _model_picked(self, condition: str, text: str) -> None:
            if self._rebuilding:
                return
            """Remember one condition's model, and say what it reads."""
            from .segment import builtin_models
            PRETRAINED = builtin_models()

            if not text:
                return
            if text == ADD_MODEL:
                self._add_model(condition)
                return
            if self._condition_models.get(condition) == text:
                return
            self._condition_models[condition] = text
            # THE FOLDER ALWAYS FOLLOWS THE MODEL. Ruled: "picking the model should
            # change the folder automatically, then after that i can manually change
            # folder if its wrong" -- so a hand-set folder lasts until the NEXT model
            # change and no longer. The two pretrained models read different IMAGES, so
            # a model change that left the folder alone would run a fluorescence model
            # over phase contrast; the earlier pin protected a correction at the cost of
            # exactly that.
            # WAS IT ACTUALLY CHOSEN? The first version compared the folder before and
            # after and announced any change as "set by hand", which fires on the very
            # first model switch, where the old folder was merely the derived default.
            # A note that cries wolf about the user's own input is worse than none.
            had = (self._condition_folders.get(condition)
                   if condition in self._folder_chosen else None)
            self._folder_chosen.discard(condition)
            self._condition_folders.pop(condition, None)
            self._refresh_model_table()
            reads = self._condition_folders.get(condition) or "+".join(
                PRETRAINED.get(text) or ())
            self._say(f"{condition}: {text}" + (f", reading {reads}/" if reads else ""))
            if had and had != self._condition_folders.get(condition):
                # SAID, because it replaces something the user typed. Silently keeping a
                # hand-set folder across a model change is the bug above; silently
                # dropping one without a word would be the same fault mirrored.
                self._say(f"  ({had}/ was set by hand and has been re-derived: set it "
                          f"again if it was right)")

        def _copy_model_to_all(self) -> None:
            """The FIRST condition's model onto every other.

            The column the button sits under, which is the only reading its position
            supports -- the tracking window's copy buttons were both fixed to this after
            one of them copied whichever condition happened to be selected elsewhere.
            """
            names = [c.name for c in (self._conditions or [])]
            if len(names) < 2:
                return
            source = names[0]
            model = self._condition_models.get(source, self._model.currentText())
            for name in names:
                self._condition_models[name] = model
            self._refresh_model_table()
            self._say(f"every condition now uses {source}'s model: {model}")

        def _roles_for(self, model: str) -> tuple:
            """The role names a model reads. One for each pretrained model."""
            from .segment import builtin_models
            PRETRAINED = builtin_models()

            return tuple(PRETRAINED.get(model) or self._roles_wanted() or ("phase",))

        def _configs_per_condition(self):
            """``{condition: SegmentConfig}`` from the table; ``None`` below two targets,
            where `cfg` is the whole answer and there is nothing to vary."""
            from .segment import SegmentConfig

            targets = self._targets()
            if len(targets) < 2:
                return None
            out = {}
            for name, _path in targets:
                model = self._condition_models.get(name, self._model.currentText())
                out[name] = SegmentConfig(model=model,
                                          channels=self._roles_for(model),
                                          use_gpu=self._gpu.isChecked())
            return out

        def _overrides_per_condition(self):
            """``{condition: {role: path}}`` -- the folder each one is READ FROM.

            Sent as an OVERRIDE rather than left to resolve, because the table has
            already shown the user which folder it is and they may have changed it.
            `resolve_channels` never second-guesses an override, so what the table says
            is what the run reads.
            """
            targets = self._targets()
            out = {}
            for name, path in targets:
                here = self._folder_path(name)
                if here is None:
                    continue
                model = self._condition_models.get(name, self._model.currentText())
                out[name] = {role: here for role in self._roles_for(model)}
            return out

        def _start_all(self) -> None:
            """Every condition in this folder, each with the model chosen for it.

            Shares `_start`'s thread plumbing rather than duplicating it, so the console,
            the disabled buttons and the teardown behave identically whichever button was
            pressed. What differs is only what the worker is given.
            """
            from .segment import SegmentConfig

            if not self._conditions:
                return
            if not self._runtime_ok():
                return
            cfg = SegmentConfig(model=self._model.currentText(),
                                channels=self._roles_wanted(),
                                use_gpu=self._gpu.isChecked())
            configs = self._configs_per_condition()
            overrides = self._overrides_per_condition()
            names = ", ".join(c.name for c in self._conditions)
            self._say("", f"segmenting {len(self._conditions)} condition(s): {names}")
            if configs:
                # SAID PER CONDITION, because the whole point of the grid is that they
                # can differ, and a single summary line would hide exactly the case it
                # was added for.
                for name, one in configs.items():
                    reads = overrides.get(name) or {}
                    where = "+".join(sorted({p.name for p in reads.values()})) \
                        or "+".join(one.channels)
                    self._say(f"  {name}: {one.model}, reading {where}/")
                distinct = {one.model for one in configs.values()}
                if len(distinct) > 1:
                    self._say("  these conditions are not segmented alike: the outlines "
                              "come from different models and different images")
            else:
                self._say(f"one model and one channel mapping "
                          f"({'+'.join(cfg.channels)}) for all of them")
            self._say("the first field also loads the model, so it is slower")
            self._run.setEnabled(False)
            self._run_all.setEnabled(False)
            self._open.setEnabled(False)

            self._thread = QThread(self)
            self._worker = _Worker(None, None, cfg, None,
                                   conditions=list(self._conditions),
                                   roles=self._roles_wanted(),
                                   configs=configs, overrides=overrides)
            self._worker.moveToThread(self._thread)
            self._thread.started.connect(self._worker.run)
            self._worker.line.connect(self._say)
            self._worker.done.connect(self._finished)
            self._worker.failed.connect(self._crashed)
            self._thread.start()

        def _finished(self, report) -> None:
            self._report = report
            self._say("", *report.summary().splitlines())
            self._stop_thread()

        def _crashed(self, text: str) -> None:
            self._say("", "the run stopped:", *text.splitlines()[:12])
            self._stop_thread()

        def _stop_thread(self) -> None:
            if self._thread is not None:
                self._thread.quit()
                self._thread.wait(5000)
            self._thread = self._worker = None
            # BOTH buttons come back, and on the SAME condition. Re-enabling only `_run`
            # would leave the batch button dead after one run, which reads as "it broke";
            # re-enabling the batch button unconditionally would offer a run the runtime
            # cannot perform, which is the thing `runtime_available` exists to prevent.
            self._run.setEnabled(self._runtime_ok())
            self._run_all.setEnabled(self._runtime_ok())
            self._open.setEnabled(True)

        # ── training ─────────────────────────────────────────────────────────
        def _train_clicked(self) -> None:
            """Open the trainer on this folder.

            A SEPARATE WINDOW, not more of this one. Training consumes CORRECTIONS
            across conditions while this window configures a RUN over movies within one,
            and the first version did it inline -- one press that built a set and printed
            a summary, with no way to choose the base model, the held-out share or which
            conditions to include, and nowhere to keep the result. Those are the controls
            the experimenter asked for.
            """
            from .trainer_gui import launch as launch_trainer

            self._say("", "opening the model trainer on this folder")
            # HELD BY THE TRAINER ITSELF (`_OPEN_WINDOWS`), so it is not collected the
            # moment this returns -- which would take its worker thread with it.
            self._trainer = launch_trainer(self._root, show=True, block=False)

        def _open_editor(self) -> None:
            if self._root is None:
                self._say("open a folder first")
                return
            from .editor_gui import launch as launch_editor

            self._editor = launch_editor(root=str(self._root))

        def closeEvent(self, event):                  # noqa: N802
            # RELEASED ONLY IF THE CLOSE ACTUALLY HAPPENS. `segment_gui` asks
            # "a run is still going, stop it?" and IGNORES the event when the
            # answer is no: dropping the reference there would leave a window
            # on screen that nothing holds, which is the exact segfault this
            # list exists to prevent, reintroduced by the fix for it.
            self._close_event(event)
            if event.isAccepted():
                _release(self)

        def _close_event(self, event):                      # noqa: N802 - Qt virtual
            if self._thread is not None:
                answer = QMessageBox.question(
                    self, "Segmentation is running",
                    "A run is still going. Stop it and close?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if answer != QMessageBox.Yes:
                    event.ignore()
                    return
                self._stop_thread()
            self._stop_gpu_thread()
            event.accept()

    return SegmentWindow


def __getattr__(name):
    """`SegmentWindow` built on first use, so importing needs no Qt."""
    if name == "SegmentWindow":
        return _build_window_class()
    raise AttributeError(name)
