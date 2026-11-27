"""The ANALYSIS stage's front door: a window first, then the folder, then the load.

Why this exists at all. Every other stage opens its window and picks a folder from
inside it. Analysis could not: `ComparisonBrowser` takes a LOADED project and builds its
whole top bar from `project.conditions`, so there is nothing to put on screen until a
folder has been read. The old route therefore ran a folder dialog, then a day-map dialog,
then a full `load_project`: .mat files, MCOS decoding, cell meshes, canonicalisation,
with **nothing visible the entire time**. That is the worst case of the complaint this
window answers ("its loading for 10 years and nothing is opening"), because here the wait
is real work rather than a stray import.

So the answer is not to open the browser empty; it is to put a small window in front of
it that says what is happening.

    [Choose project folder…]
    what was found: the conditions, counted, before anything is loaded
    [Open]      -> resolve the replicate days, load, hand over to the browser
    log         every step, as it happens

THIN QT, the house rule: no numpy, no pandas, no scipy at module level, and nothing is
computed here. Discovery comes from `project.discover_conditions`, the day mapping from
`app.resolve_day_maps` (shared with the command line, so the two cannot drift), and the
load from `project.load_project`.

The load runs on a thread, because it is the slow step and a frozen window during it
reads as a crash, which is the whole point of the window. The thread is JOINED on close:
a QThread destroyed while running calls `abort()`, with no exception and no traceback.
"""
from __future__ import annotations

__all__ = ["OpenProjectWindow", "launch_open", "describe_project_folder"]


def describe_project_folder(root):
    """`(conditions, lines)` for a folder, WITHOUT loading anything.

    Cheap on purpose: it globs for movies, it does not read them. The point is to tell
    the user they have pointed at the right place before they commit to a load that
    takes minutes.
    """
    from pathlib import Path

    root = Path(root)
    lines = [f"opened {root}"]
    try:
        from .project import discover_conditions

        found = discover_conditions(root)
    except Exception as exc:                              # noqa: BLE001 - reported
        # THE HINT GOES ON THE FAILURE TOO. `discover_conditions` raises on a folder
        # with nothing in it, so the commonest mistake, pointing at one condition, or
        # at the drive, produced an exception message and no clue what was wanted.
        lines.append(f"{type(exc).__name__}: {exc}")
        lines.append("this should be the folder containing your condition "
                     "sub-folders")
        return [], lines

    conditions = sorted(found) if found else []
    if not conditions:
        lines.append("no conditions found here; this should be the folder containing "
                     "your condition sub-folders")
        return [], lines
    lines.append(f"{len(conditions)} condition(s): " + ", ".join(conditions))

    # WHICH TRACKER, ASKED RATHER THAN REFUSED. A folder tracked twice holds two
    # datasets with one name, and `grouping` is right to refuse it, a library with
    # nobody in front of it cannot choose. A window can ask, and failing instead was
    # making the user read a library's refusal as an error about their folder.
    from .grouping import sources_present

    both = sorted(name for name in conditions
                  if len(sources_present(found[name])) > 1)
    if both:
        counts = sources_present(found[both[0]])
        lines.append(
            f"{len(both)} condition(s) hold both trackers ("
            + ", ".join(f"{k}: {v} file(s)" for k, v in sorted(counts.items()))
            + "): choose which to read")
        lines.append("they differ by about 1 pp in the fractions and 10-18 % in D, "
                     "so the choice matters")
    lines.append("press Open to read them; a first load is minutes, a warm one is not")
    return conditions, lines


#: Windows this module has opened. A top-level PySide6 window with no Python reference is
#: destroyed the moment its caller discards it, taking any thread it owns, measured as a
#: segfault on the segmentation door. The door owns what it opens.
_OPEN_WINDOWS: list = []


def launch_open(root=None, *, show: bool = True):
    """Open the analysis front door. Returns the window."""
    from qtpy.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    from .appicon import apply as _apply_icon

    _apply_icon()
    window = _build_window_class()(root)
    if window not in _OPEN_WINDOWS:
        _OPEN_WINDOWS.append(window)
    if show:
        window.show()
    return window


def _build_window_class():
    """Qt imported inside, so this module imports on a machine with no GUI stack."""
    from qtpy.QtCore import QObject, QThread, Signal
    from qtpy.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QLabel,
                                QMainWindow, QPlainTextEdit, QPushButton,
                                QVBoxLayout, QWidget)

    class _Loader(QObject):
        """Reads the project off the GUI thread. Emits; never touches a widget."""

        line = Signal(str)
        done = Signal(object)
        failed = Signal(str)

        def __init__(self, root, condition, day_maps, localization_nm, source=None):
            super().__init__()
            self._root, self._condition = root, condition
            self._day_maps, self._nm = day_maps, localization_nm
            self._source = source

        def run(self):
            try:
                from . import options as opts
                from .project import APP_FRAME_INTERVAL, APP_MIXED_TAU, load_project
                from .ui import theme

                self.line.emit("reading tracks, meshes and acquisition metadata…")
                kwargs = {}
                if self._source is not None:
                    # ONLY WHEN ASKED. Passing source=None would override the loader's
                    # own resolution, which is right for a folder holding one tracker.
                    kwargs["source"] = self._source
                project = load_project(
                    self._root,
                    conditions=[self._condition] if self._condition else None,
                    localization_precision_nm=self._nm,
                    day_maps=self._day_maps,
                    mixed_tau=APP_MIXED_TAU, frame_interval_from=APP_FRAME_INTERVAL,
                    max_tau_spread=opts.max_tau_spread(theme.remembered(opts.OPTIONS_KEY)),
                    outline_tolerance_nm=opts.outline_tolerance_nm(
                        theme.remembered(opts.OPTIONS_KEY)),
                    skip_unsegmented=True,
                    **kwargs,
                )
                # THE GROUP BOARD'S HOLD-OUTS, remembered in day_map.json: left out here too, one rule at every
                # door (ruled 4 Nov 2026)
                from .project import with_replicate_pairs, without_held_out

                project = with_replicate_pairs(without_held_out(project, self._day_maps), self._day_maps)
                from .onboarding_view import unsegmented_notice

                notice = unsegmented_notice(project.canonical.attrs.get("dropped_unsegmented"),
                                            list(project.conditions))
                if notice is not None:
                    self.line.emit(f"{notice['title']}: {notice['body']}")
            except Exception as exc:                      # noqa: BLE001 - reported
                self.failed.emit(f"{type(exc).__name__}: {exc}")
                return
            self.done.emit(project)

    class OpenProjectWindow(QMainWindow):
        """Choose a project, watch it load, hand it to the browser."""

        def __init__(self, root=None, parent=None):
            super().__init__(parent)
            self.setWindowTitle("SMTracker · analysis")
            self._root = str(root) if root else None
            self._conditions: list[str] = []
            #: Conditions holding output from both trackers. While non-empty the source
            #: must be chosen before anything is read.
            self._tracked_twice: list[str] = []
            self._thread = None
            self._loader = None
            self._browser = None
            #: Set once the browser is up, so a test can see the hand-over happened
            #: without a live event loop.
            self.project = None

            central = QWidget()
            layout = QVBoxLayout(central)

            self._header = QLabel()
            self._header.setWordWrap(True)
            layout.addWidget(self._header)

            row = QHBoxLayout()
            self._choose = QPushButton("Choose project folder…")
            self._choose.clicked.connect(self._choose_folder)
            row.addWidget(self._choose)
            self._open = QPushButton("Open")
            self._open.setDefault(True)
            self._open.clicked.connect(self._start_load)
            row.addWidget(self._open)
            row.addStretch(1)
            layout.addLayout(row)

            # SHOWN ONLY WHEN THE QUESTION EXISTS, and with NO DEFAULT when it does.
            # Picking one silently would move reported numbers with nothing on screen
            # to say why, which is exactly what `grouping` refuses to do.
            self._source_row = QWidget()
            source_layout = QHBoxLayout(self._source_row)
            source_layout.setContentsMargins(0, 0, 0, 0)
            source_layout.addWidget(QLabel("Tracked twice. Read:"))
            self._source = QComboBox()
            self._source.addItem("choose…", None)
            self._source.addItem("u-track", "utrack")
            self._source.addItem("TrackMate", "trackmate")
            self._source.addItem("Track table", "table")
            self._source.currentIndexChanged.connect(lambda _i: self._apply_state())
            source_layout.addWidget(self._source)
            source_layout.addStretch(1)
            self._source_row.setVisible(False)
            layout.addWidget(self._source_row)

            self._console = QPlainTextEdit()
            self._console.setReadOnly(True)
            layout.addWidget(self._console, 1)

            self.setCentralWidget(central)
            self.resize(620, 340)
            if self._root:
                self._rescan()
            else:
                self._say("choose the folder containing your condition sub-folders")
            self._apply_state()

        # ── plumbing ─────────────────────────────────────────────────────────
        def _say(self, *lines) -> None:
            for line in lines:
                self._console.appendPlainText(str(line))

        def _apply_state(self) -> None:
            """Open is offered only when there is something to open AND the tracker
            question, if there is one, has been answered."""
            busy = self._thread is not None
            self._header.setText(
                f"<b>{self._root}</b>" if self._root
                else "<b>no folder chosen</b>: press “Choose project folder…”")
            self._choose.setEnabled(not busy)
            self._source_row.setVisible(bool(self._tracked_twice))
            answered = not self._tracked_twice or self._source.currentData() is not None
            self._open.setEnabled(bool(self._conditions) and answered and not busy)

        def _choose_folder(self) -> None:
            picked = QFileDialog.getExistingDirectory(
                self, "Project folder (the one holding your conditions)",
                self._root or "")
            if not picked:
                return
            self._root = picked
            self._rescan()

        def _rescan(self) -> None:
            self._conditions, lines = describe_project_folder(self._root)
            self._tracked_twice = []
            if self._conditions:
                from .grouping import sources_present
                from .project import discover_conditions

                try:
                    roots = discover_conditions(self._root)
                except Exception:                         # noqa: BLE001 - reported
                    roots = {}
                self._tracked_twice = sorted(
                    name for name, path in roots.items()
                    if len(sources_present(path)) > 1)
            self._say(*lines)
            self._apply_state()

        # ── the load ─────────────────────────────────────────────────────────
        def _start_load(self) -> None:
            if self._root is None or self._thread is not None:
                return
            # THE DAY MAP IS CONFIRMED ON THIS THREAD, because it is a DIALOG. Only the
            # reading goes to the worker. Shared with the command line
            # (`app.resolve_day_maps`) so a run from either cannot resolve days
            # differently: the trap this project has already hit twice, where a second
            # load path re-derived what the first had settled.
            from .app import resolve_day_maps

            try:
                day_maps = resolve_day_maps(self._root, report=self._say)
            except Exception as exc:                      # noqa: BLE001 - reported
                self._say(f"could not resolve replicates: "
                          f"{type(exc).__name__}: {exc}")
                return

            self._say(f"loading {self._root}…")
            self._thread = QThread(self)
            self._loader = _Loader(self._root, None, day_maps, None,
                                   source=self._source.currentData())
            self._loader.moveToThread(self._thread)
            self._thread.started.connect(self._loader.run)
            self._loader.line.connect(self._say)
            self._loader.done.connect(self._loaded)
            self._loader.failed.connect(self._load_failed)
            self._thread.start()
            self._apply_state()

        def _stop_thread(self) -> None:
            """Join the loader. A QThread destroyed while running calls `abort()`,
            no exception, no traceback, and the process is simply gone."""
            if self._thread is not None:
                self._thread.quit()
                self._thread.wait(30000)
                self._thread = None
            self._loader = None

        def _load_failed(self, message: str) -> None:
            """Report the failure, and add a hint ONLY where it applies.

            It used to append "check that this is the folder holding the condition
            folders" to everything, so a folder that was perfectly correct, and had
            simply been tracked twice, was told it was the wrong folder. A hint that
            fires on every failure teaches the reader to ignore hints.
            """
            self._stop_thread()
            lines = ["FAILED: " + message]
            if "no tracker output" in message or "does not exist" in message:
                lines.append("check that this is the folder that holds the condition "
                             "folders, not a single condition")
            self._say(*lines)
            self._apply_state()

        def _loaded(self, project) -> None:
            self._stop_thread()
            self.project = project
            try:
                self._say(project.summary())
            except Exception:                             # noqa: BLE001 - cosmetic
                pass
            self._say("opening the comparison browser…")
            from .browser import launch_browser

            _app, self._browser = launch_browser(project, root=self._root)
            self.hide()

        def closeEvent(self, event):                      # noqa: N802 - Qt virtual
            self._stop_thread()
            for i, held in enumerate(_OPEN_WINDOWS):
                if held is self:
                    del _OPEN_WINDOWS[i]
                    break
            event.accept()

    return OpenProjectWindow


def __getattr__(name):
    """`OpenProjectWindow` built on first use, so importing needs no Qt."""
    if name == "OpenProjectWindow":
        return _build_window_class()
    raise AttributeError(name)
