"""The TRACKING stage: run TrackMate over a whole condition, then judge the batch.

Reached from the launcher (`smtracker.launcher`), not from the analysis catalogue, and
the distinction is structural rather than cosmetic. Every catalogue entry CONSUMES the
canonical table; tracking PRODUCES it. The catalogue's defining property is that it
computes each entry on EVERY loaded condition side by side, which for a tracker means a
quarter of an hour of Fiji subprocesses each time a panel opens.

Thin QT, same rule as `widgets` AND `statemaps`. Nothing here computes: the window reads
numbers out of its own boxes and hands them to `trackmate_run.settings_from_preset`,
`run_batch`, `batch_qc` and `describe_batch`, all of which are pure, Qt-free and tested
headless. A wrong threshold fails in `test_trackmate_run.py`, not by looking subtly odd
on screen.

Three things this window refuses to let slide, each because the alternative reads as
something it is not:

* **The settings stamp is live and says when it is NOT measured.** `UTRACK_COMPARABLE`
  holds two per-condition thresholds that were matched to u-track's yield and confirmed
  end to end; anything else is `custom linking, not measured against u-track`, and the
  window prints that sentence rather than a green tick. A number a user typed is not a
  number anybody measured.
* **A failed movie is a row, not an absence.** `run_batch` keeps failures with their
  error and `describe_batch` names them FIRST, because a table listing only the successes
  reads as "these are the movies".
* **The cost is shown before the run, not during it.** About 30 s per movie measured on
  this machine, so a 30-movie condition is roughly a quarter of an hour. A progress bar
  that starts at zero with no estimate teaches people to kill the job.

Nothing here reaches the network: Fiji is found or pointed at, never downloaded (the old
download address went 404 when Fiji renamed its archives; see `smtracker.fiji`).
"""
from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal

from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QLineEdit,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QTableWidget,
    QTableWidgetItem,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
# The window offers exactly the modes `apply_layout` accepts. A second
# hard-coded list would drift, and it would drift into a combo box offering
# something the function then refuses. `organise` is Qt-free.
#: HOW FILES ARE FILED, and it is "move" by RULING rather than by default.
#: `organise.apply_layout` refuses to assume a mode, deliberately -- moving is not undone
#: by pressing anything and copying doubles a folder. That refusal was built for a
#: STANDALONE Organise button, which no longer exists: saving the crop ranges is what
#: files a folder now, and the experimenter ruled the box out ("cropping saving moves it
#: in the right folder"). So the choice is made here, once, in the open, instead of being
#: asked on a press nobody associates with moving files. `apply_layout` keeps its refusal
#: for every other caller.
ORGANISE_DEFAULT_MODE = "move"

#: What each settings row actually holds, shown on the row header. Only a row
#: whose number is a SUMMARY needs one, and the threshold is the case: it is a
#: MEDIAN, while each movie is detected against its own noise.
ROW_NOTES = {
    "threshold": (
        "The median of the per-movie thresholds. Each movie is detected against its "
        "own LoG noise. Type a number to use one threshold for every movie of this "
        "condition; measure again to go back to per-movie."),
    "radius_px": (
        "Derived from this condition's emission wavelength, NA and pixel size. "
        "Type to override."),
    "linking_max_distance_px": (
        "How far a molecule may move between frames. This is not the spot radius."),
    "max_frame_gap": (
        "Frames a molecule may vanish for and still be one track. The gap is a time: "
        "3 frames is 61 ms at 20.47 ms and 155 ms at 51.75 ms."),
}

__all__ = ["TrackingWindow", "launch_tracking"]

#: Measured on this machine over 16 movies: 27-34 s each, dominated by JVM startup rather
#: than by the tracking. Used only to print an estimate BEFORE the run.
SECONDS_PER_MOVIE = 30.0

#: This project's layout, `tif/<movie>/<movie>/*.tif`. Editable in the window
#: because it is one lab's convention rather than a property of TrackMate.
#: EMPTY MEANS RESOLVE. `trackmate_run.discover_movies` tries the flat prepared
#: stack `tif/<movie>.tif` first and u-track's exploded `tif/<movie>/<movie>/`
#: folder second, so the box does not need to name either. It used to default to
#: "tif/*/*", which pointed every run at u-track's own working copy.
DEFAULT_MOVIE_PATTERN = ""

#: The two ways in. PREPARED is canonical for this project's existing data because
#: u-track cannot read a .nd2 and those stacks already exist. RAW is the alternative for
#: NEW data: it saves writing a second copy to disk, and it needs a per-movie frame range,
#: which the preparation stage already establishes and records in `crops.json`.
#:
#: VALIDATED EQUIVALENT on one lab movie against real Fiji: the raw file over its
#: crop range and the prepared stack give 1312 spots, 253 tracks and 922 localisations
#: each, ELEMENT-WISE IDENTICAL. So the raw path inherits `movieprep`'s bit-identical
#: standing rather than asking to be trusted on its own.
#: Returned by a stage window when the user finished that STAGE rather than the tool, so
#: `app.main` can show the launcher again instead of exiting. A sentinel and not an
#: integer: an exit code says how a process ended, and this says what to do next.
RETURN_TO_MENU = "return-to-menu"

SOURCE_PREPARED = "Prepared stacks (tif/<movie>.tif)"
SOURCE_RAW = "Raw acquisitions + recorded crops"
SOURCES = (SOURCE_PREPARED, SOURCE_RAW)

#: Asked whether the prepared source is needed at all: "we track from raw in
#: trackmate anyways". It is needed, for two reasons that have nothing to do with
#: TrackMate, and the answer is kept here rather than in a commit message:
#:
#: 1. **u-track CANNOT READ A `.nd2`.** Every number in this project's reference data came
#:    from u-track on prepared stacks. Re-running u-track, on new data, or to reproduce
#:    an existing result: requires them. TrackMate reading raw does not change that.
#: 2. **THE RAW PATH REFUSES A MOVIE WITH NO RECORDED CROP RANGE**, deliberately: a raw
#:    acquisition still holds the dense early frames the crop exists to remove. Ranges
#:    come from `crops.json`, which the PREPARATION stage writes. The existing 119 movies
#:    were cropped with the lab's Fiji macro years before that file existed, so for them
#:    the prepared stacks are the ONLY working route.
#:
#: What was wrong was presenting it as a question. `organise.inspect_folder` reads the
#: folder and resolves it, so the control is behind a disclosure and the state line says
#: which source is in use.
SOURCE_NOTE = (
    "Chosen from what is in the folder. Prepared stacks are required for u-track, "
    "which cannot read a .nd2. Raw acquisitions need a recorded crop range and "
    "save writing a second copy of the data."
)



#: The tube lens between the objective and the camera, which NO file records.
#:
#: NIS derives its pixel-size field from the objective magnification alone, so on this
#: setup it reports 160 nm where the truth is 106: 16 µm camera pixel / 100× = 160, and
#: / (100× × 1.5) = 106.7 against the 106.0 u-track records. It is the ONE factor that
#: turns two read values into a pixel size, which is why the result is offered as a
#: derivation the user can change rather than as a reading.
TUBE_LENS_DEFAULT = 1.5

#: THE PIXEL SIZE THIS PROJECT ACTUALLY USES, and what the derivation is FOR.
#:
#: The Fiji preparation macro wrote `pixel_width=0.106000` with `unit=[um]`, u-track read
#: it from the baked-in calibration, and `movieData.mat` records **106.0 nm**. Every
#: deposited number, every figure and the published tables rest on it.
#:
#: The camera-and-objective derivation gives 106.667: 0.63 % away, which is 1.26 % on
#: every D since it enters squared. Neither is MEASURED: the derivation assumes the tube
#: lens is exactly 1.5x, and 106.0 is what someone typed once. Only a stage micrometer
#: would settle it and it is not worth the trip.
#:
#: So the DERIVATION'S job is to confirm, not to replace. Ruled by the experimenter,
#: "106 if it was used before, forget the 0.7", and it is the right call for a reason
#: beyond preference: being 1.3 % off your own published numbers is a worse problem than
#: 0.6 % of optics, and "why doesn't this match the published numbers" costs more than it buys.
PROJECT_PIXEL_SIZE_NM = 106.0

#: How far the derivation may sit from the project value and still be treated as agreeing.
#: Wide enough for the 0.63 % tube-lens gap, narrow enough that a genuinely different
#: setup (a 60x objective, a different camera) falls outside and fills its own number.
PIXEL_SIZE_AGREEMENT = 0.02


class _CalibrationSpinBox(QDoubleSpinBox):
    """A spin box that selects its text when clicked, so typing replaces it.

    ASKED FOR: "NA is not set, which makes it hard to type a number cause clicking on it
    to write something doesnt work well. clicking on not set should select whole text so
    typing replaces it."

    The special-value text ("not set") is the reason it bites hardest: clicking lands a
    caret inside a WORD, so the first digit typed produces "n1ot set" and the box rejects
    it. Selecting on focus makes the first keystroke replace the whole field, which is
    what every other numeric field in the world does.

    A subclass rather than an assigned `focusInEvent`, because Qt dispatches the C++
    virtual and only reaches Python when the CLASS overrides it, patching the instance
    looks like it works and silently does nothing.

    The selection is deferred by one event-loop turn: Qt places the caret AFTER
    `focusInEvent` returns, so selecting inside it is undone immediately.
    """

    def focusInEvent(self, event):
        super().focusInEvent(event)
        QTimer.singleShot(0, self.selectAll)

    def mousePressEvent(self, event):
        # A CLICK IS NOT ALWAYS A FOCUS CHANGE. Clicking a box that already has focus
        # emits no focusInEvent, so the text would stay unselected exactly when the user
        # clicks it a second time to retype.
        had_focus = self.hasFocus()
        super().mousePressEvent(event)
        if not had_focus:
            QTimer.singleShot(0, self.selectAll)


class _ConditionRow(QWidget):
    """One condition in the overview, and the control that selects it.

    ASKED FOR: "see the picker? why not remove the picker and just click on the condition
    below to see their movies". The dropdown and these rows named the same set twice, and
    only one of them showed what state each condition was in -- so the picker was the copy
    with less information on it.

    The combo box is KEPT, hidden, as the model underneath: `_advance_to_next_condition`
    and both restore paths drive selection through `setCurrentText`, so routing every
    change through it means the walk and a click cannot end up disagreeing about which
    condition is on screen.
    """

    clicked = Signal(str)

    def __init__(self, name: str):
        super().__init__()
        self._name = name
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(f"show {name} in the bleaching-curve browser below")

    def mousePressEvent(self, event):                     # noqa: N802 - Qt's name
        self.clicked.emit(self._name)
        super().mousePressEvent(event)


class _NoiseSignals(QObject):
    progress = Signal(int, int, object)
    finished = Signal(object)
    failed = Signal(str)


class _NoiseCancelled(Exception):
    """Raised inside `_NoiseJob`'s progress callback to stop between movies."""


class _NoiseJob(QRunnable):
    """`batch_threshold` off the GUI thread. It CALLS; it does not compute.

    REPORTED: "changing some values sometimes takes long". The LoG noise is read inside
    every movie's crop -- MEASURED 1.6 s a movie on one user's MetaMorph .stk, so about 16 s
    a condition and two and a half minutes over her nine -- and it ran ON the GUI thread
    after every condition the walk read and after every preset or folder change,
    pumping `processEvents` between movies. `_ExportJob` already records why that is not
    a fix: the window repaints BETWEEN movies and is frozen DURING each one.

    Nothing here touches a widget. ``credit`` is who the answer belongs to, captured
    when the request was made; ``cancelled`` is set from the GUI thread and stops the
    pass after the movie in hand.
    """

    def __init__(self, movies, *, radius_px, crops, credit):
        super().__init__()
        self._movies = list(movies)
        self._radius_px = float(radius_px)
        self._crops = crops
        self.credit = credit
        self.cancelled = False
        self.signals = _NoiseSignals()

    def run(self):                                     # noqa: D102 - QRunnable protocol
        from .trackmate_run import batch_threshold

        def report(index, total, row):
            if self.cancelled:
                raise _NoiseCancelled()
            self.signals.progress.emit(index, total, row)

        try:
            got = batch_threshold(self._movies, radius_px=self._radius_px,
                                  crops=self._crops, on_progress=report)
        except _NoiseCancelled:
            return
        except Exception as exc:                       # noqa: BLE001 - named on screen
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.signals.finished.emit(got)


class _BatchSignals(QObject):
    progress = Signal(int, int, str)
    finished = Signal(object)
    failed = Signal(str)


class _BatchJob(QRunnable):
    """Runs `run_batch` off the GUI thread. It CALLS; it does not compute."""

    def __init__(self, movies, output_dir, settings, install, *, crops=None,
                 thresholds=None):
        super().__init__()
        self._crops = crops
        #: {movie stem: threshold}, or None for one value across the batch.
        self._thresholds = thresholds
        self._movies = movies
        self._output_dir = output_dir
        self._settings = settings
        self._install = install
        self.signals = _BatchSignals()

    def run(self):                                     # noqa: D102 - QRunnable protocol
        from .trackmate_run import batch_qc, run_batch

        try:
            def on_progress(index, total, row):
                note = (f"{row['movie']}: {row['n_spots']} spots, "
                        f"{row['n_tracks']} tracks, {row['seconds']:.0f} s"
                        if row["ok"] else f"{row['movie']}: FAILED: {row['error']}")
                self.signals.progress.emit(index, total, note)

            table = run_batch(self._movies, self._output_dir,
                              settings=self._settings,
                              thresholds=self._thresholds,
                              install=self._install, on_progress=on_progress,
                              crops=self._crops)
            self.signals.finished.emit((table, batch_qc(table)))
        except Exception as exc:                       # noqa: BLE001 - reported, never eaten
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}")


#: How many movies the radius check tracks, at each of two radii. Two is enough for the
#: statistic -- the twelve-group table was built on six -- and it is what keeps the check
#: at about two minutes rather than a quarter of an hour.
RADIUS_CHECK_MOVIES = 2

#: How much wider the second run goes. NOT "wide enough to be uncensored": measured on
#: this project's own sweep, p99/radius is 0.97 at 5 px and still 0.91 at 12, so no
#: affordable pilot is uncensored. 1.6x is wide enough to REVEAL what the narrow radius
#: hid -- 8 px against 5 px shows 17.5 per cent on a lab project -- which is the question.
RADIUS_CHECK_MULTIPLE = 1.6


class _RadiusCheckJob(QRunnable):
    """Track the SAME movies at two radii and report what the narrow one could not see.

    TWO RUNS, NOT ONE AGAINST THE EXISTING TRACKS. The obvious saving is to compare a new
    wide run against the u-track output already in the folder -- and it would confound the
    radius with the DETECTOR, which is the one thing this statistic cannot see and would
    report as truncation. Everything except the linking distance is held identical.

    It CALLS; it does not compute. `run_batch` and `widening_yield` are both pure and
    covered headless.
    """

    def __init__(self, movies, output_dir, settings, install, *, crops,
                 thresholds, narrow_px, wide_px, tau_s, pixel_size_nm):
        super().__init__()
        self._movies = movies
        self._output_dir = output_dir
        self._settings = settings
        self._install = install
        self._crops = crops
        self._thresholds = thresholds
        self._narrow_px = float(narrow_px)
        self._wide_px = float(wide_px)
        self._tau_s = float(tau_s)
        self._pixel_size_nm = float(pixel_size_nm)
        self.signals = _BatchSignals()

    def _one(self, radius_px, label, done, total):
        from dataclasses import replace

        from .trackmate_run import read_batch, run_batch

        settings = replace(self._settings,
                           linking_max_distance_px=radius_px,
                           gap_closing_max_distance_px=radius_px)
        out = self._output_dir / f"link{radius_px:g}"

        def on_progress(index, _total, row):
            note = (f"{label} {radius_px:g} px: {row['movie']}: "
                    + (f"{row['n_tracks']} tracks" if row["ok"]
                       else f"FAILED: {row['error']}"))
            self.signals.progress.emit(done + index, total, note)

        table = run_batch(self._movies, out, settings=settings,
                          thresholds=self._thresholds, install=self._install,
                          on_progress=on_progress, crops=self._crops)
        return read_batch(table)

    def run(self):                                     # noqa: D102 - QRunnable protocol
        from .trackmate_run import widening_yield

        try:
            total = 2 * len(self._movies)
            narrow = self._one(self._narrow_px, "narrow", 0, total)
            wide = self._one(self._wide_px, "wide", len(self._movies), total)
            got = widening_yield(narrow, wide,
                                 narrow_radius_px=self._narrow_px,
                                 wide_radius_px=self._wide_px,
                                 tau_s=self._tau_s,
                                 pixel_size_nm=self._pixel_size_nm)
            self.signals.finished.emit(got)
        except Exception as exc:                       # noqa: BLE001 - reported, never eaten
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}")


class _ExportSignals(QObject):
    progress = Signal(int, int, str)
    finished = Signal(object)
    failed = Signal(str)


class _SaveCropsJob(QRunnable):
    """Writes crops.json, the bleaching curves and the projections, off the GUI thread.

    REPORTED: "clicking on save crop ranges freezes the tool. its doing the work, but we
    said no freeze and always an indicator or something."

    It was doing the work on the GUI thread. `CropPanel.write_projections` READS every
    movie's cropped frames, and it tried to stay responsive by moving a bar and calling
    `QApplication.processEvents()` between movies, which `_ExportJob` below already
    records as not a fix: the window repaints BETWEEN movies while each movie's read is
    a solid block of frozen GUI, so a folder of large acquisitions still reads as a
    hang, and Windows paints it "(Not Responding)".

    Nothing here touches a widget. The panel's write methods take an `on_progress`
    callback now and touch no widget when given one, because touching a QWidget from a
    worker thread is undefined behaviour in Qt: it does not raise, it corrupts.
    """

    def __init__(self, panel, **kwargs):
        super().__init__()
        self._panel = panel
        self._kwargs = kwargs
        self.signals = _ExportSignals()

    def run(self):                                     # noqa: D102 - QRunnable protocol
        try:
            path = self._panel.save(
                on_progress=lambda done, total, note:
                    self.signals.progress.emit(done, total, note),
                **self._kwargs)
        except Exception as exc:                       # noqa: BLE001 - named on screen
            self.signals.failed.emit(f"could not save: {type(exc).__name__}: {exc}")
            return
        self.signals.finished.emit(path)


class _ExportJob(QRunnable):
    """Writes the cropped stacks off the GUI thread.

    The export froze the window, and `processEvents` in the loop is not a fix: it keeps
    the window repainting while the thread is still inside a multi-gigabyte write, so
    the progress moves in lurches and a click can re-enter the handler. The tracking
    batch already runs on a `QRunnable` for the same reason; this is the same shape.

    Cancelling stops between movies, not inside one. A half-written stack is worse than
    a missing one (the folder would hold a movie with frames nobody chose) so the flag
    is checked at the top of each movie and the ones already written stay.
    """

    def __init__(self, raw_dir, out_root, crops, *, pixel_size_nm, frame_interval_s,
                 projection):
        super().__init__()
        self._raw_dir = raw_dir
        self._out_root = out_root
        self._crops = dict(crops)
        self._pixel_size_nm = pixel_size_nm
        self._frame_interval_s = frame_interval_s
        self._projection = projection
        self._cancelled = False
        self.signals = _ExportSignals()

    def cancel(self):
        self._cancelled = True

    def run(self):                                     # noqa: D102 - QRunnable protocol
        from .movieprep import prepare_batch

        try:
            def on_progress(index, total, row):
                self.signals.progress.emit(
                    index, total,
                    f"{row.get('movie', '?')}"
                    + ("" if row.get("ok", True) else f" FAILED: {row.get('error')}"))
                if self._cancelled:
                    raise _ExportCancelled()

            table = prepare_batch(
                self._raw_dir, self._out_root, self._crops,
                pixel_size_nm=self._pixel_size_nm,
                frame_interval_s=self._frame_interval_s,
                projection=self._projection, on_progress=on_progress)
            self.signals.finished.emit(table)
        except _ExportCancelled:
            self.signals.failed.emit("cancelled: movies already written are kept")
        except Exception as exc:                       # noqa: BLE001 - reported, never eaten
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}")


class _ExportCancelled(RuntimeError):
    """Raised inside the progress callback to unwind a cancelled export."""


class TrackingWindow(QWidget):
    """Fiji → movies → settings → batch → QC, in that order down the window."""

    def __init__(self, root, parent=None):
        super().__init__(parent)
        # THE RUNNING BUILD IS IN THE TITLE, where a screenshot always
        # catches it. Three things in this repository are called smtracker
        # and they looked identical on screen; a day was spent proving a fix
        # worked in source while an older bundle was open. See `buildstamp`.
        from .buildstamp import build_stamp
        self.setWindowTitle("SMTracker · tracking  ·  " + build_stamp())
        # NO FOLDER YET IS A REAL STATE, not an error. The window opens first and its
        # own "Choose folder…" button picks, asked for, and the same rule the
        # segmentation and preparation doors already follow. `_root is None` was already
        # TESTED FOR at `_acquisition_folder`, and could never happen, because this line
        # used to be a bare `Path(root)` that raised on None: a branch written for a
        # state the constructor made unreachable.
        self._root = Path(root) if root is not None else None
        #: The folder that was OPENED. Picking a condition repoints `_root` at it, so
        #: organising, which is a decision about the whole session, needs this instead.
        self._parent_dir = Path(root) if root is not None else None
        #: `{condition name: folder}` when a parent of conditions was opened.
        self._conditions: dict = {}
        # ASSIGNED HERE, NOT REACHED FOR WITH getattr. A `getattr(self, x, default)`
        # read of state that __init__ never set is what made clicking a condition raise
        # `_calibration_from` -- the default masked the missing attribute until the one
        # path that needed a real value found nothing.
        self._outline_notes: dict = {}
        self._said_outlines: list | None = None
        self._skipped_outlines: list = []
        self._detect_note: str = ""
        #: `{condition: {stem: (start, end)}}` for the WHOLE loaded dataset.
        #:
        #: THE SESSION IS THE UNIT, not the folder on screen. Reported: switching
        #: conditions sent the first one back to 0/10, and switching back showed counts
        #: from two folders mixed together. The crop panel holds one folder and starts
        #: clean, so the window has to be the thing that remembers, and the dropdown then
        #: only chooses which condition you are LOOKING at.
        self._session_crops: dict = {}
        #: `{condition: (n_movies, n_curves, n_without_a_floor)}` once read through.
        self._session_read: dict = {}
        #: `{condition: {stem: MovieCurves}}` and the movies that would not read.
        #:
        #: KEPT SO A SWITCH IS A VIEW CHANGE. Re-reading cost 7.7 s per movie to
        #: recompute an answer already in hand, and discarded the crops with it.
        self._session_curves: dict = {}
        #: condition -> the flagged movie set already announced, so the log says it once.
        self._said_crowding: dict = {}
        self._session_errors: dict = {}
        #: Conditions still to be exported, and where to put the picker back.
        self._export_queue: list = []
        self._export_return_to = None
        self._exporting_condition = None
        #: Conditions still to be written by "Save crop ranges", and where to return.
        #: Conditions whose spot radius was typed rather than derived from the optics.
        self._radius_typed: dict = {}
        self._save_queue: list = []
        self._save_return_to = None
        self._saving_condition = None
        #: True only while a condition is being PUT BACK on screen from memory.
        #:
        #: `adopt` settles the panel, which emits `reading_changed(False)`, which the
        #: walk reads as "a read finished, move on" -- so showing a condition again
        #: immediately switched away from it. The signal is honest; what it means here is
        #: not "a pass completed", so the walk is told to sit still.
        self._adopting = False
        #: Conditions still to be walked. Emptied as each is read.
        self._to_read: list = []
        #: `{condition name: threshold}` for every condition measured this session.
        #:
        #: PER CONDITION, because the threshold is `k x sigma_LoG` of THAT condition's
        #: own movies. Carrying one across a folder change would report one session's
        #: noise over another's data, which `_scan_movies` already refuses to do. Keeping
        #: them here is what lets the overview say where a setting exists and where it
        #: does not, instead of only ever showing the condition on screen.
        self._thresholds: dict = {}
        #: {condition: {movie stem: threshold}}. Each movie detected against ITS OWN
        #: noise, which is what a batch median cannot do once the movies differ:
        #: measured on a lab project, day 1 sits at sigma 38.5 against 18.3 and 19.9, so a
        #: median set by the twenty quiet movies detects the nine noisy ones at about
        #: 1.6 sigma where the rest get 3.2.
        self._movie_thresholds: dict = {}
        #: Conditions whose threshold was TYPED. One number then applies to every movie
        #: of that condition, which is a legitimate answer and must not be silently
        #: replaced by the per-movie table on the next redraw.
        #:
        #: NOT `_threshold_typed`, which is already a SLOT on this window
        #: (`self._threshold.valueChanged.connect(self._threshold_typed)`). Shadowing it
        #: with a set made every construction raise "Expected signal or callable, got
        #: set" -- 161 tests, one name. Third collision in this window after the
        #: duplicate organise button and `_movie_count`.
        self._typed_thresholds: set = set()
        self._install = None
        self._movies: list[Path] = []
        self._pool = QThreadPool.globalInstance()
        #: The threshold measurement in flight (`_NoiseJob`), the requests waiting
        #: behind it as (owner, credit), and the conditions a "measure all" press is
        #: collecting for its summary line (None when no such press is pending).
        self._noise_job = None
        self._noise_queue: list = []
        self._noise_batch = None
        self._noise_batch_here = None
        # Written by `_on_finished`; initialised here so reading it before a run is
        # None rather than AttributeError.
        self._last_batch = None
        # Written by the crop-save job; initialised so reading it before a save is None
        # rather than AttributeError, the same reason `_last_batch` is here.
        self._save_job = None
        self._pattern = None
        self._source = SOURCE_PREPARED
        # Set by the Done button so `launch_tracking` can tell "finished this stage"
        # from "closed the tool". Everything else that closes the window leaves it False.
        self.return_to_menu = False
        self._crops: dict[str, tuple[int, int]] = {}
        # A Fiji the user picked, which BEATS the search order. Read by
        # `_find_fiji`; None means 'whatever the search finds'.
        self._fiji_choice = None

        shell = QVBoxLayout(self)
        self._header = QLabel(self._root_label())
        shell.addWidget(self._header)

        # TWO SCREENS, NOT ONE TALL ONE. Everything below used to sit in a single
        # window and the experimenter's verdict was "user gets overloaded", which is
        # right: preparing a folder and tuning a detector are different jobs, done at
        # different moments, and the second is meaningless until the first has produced
        # movies. So the window is a two-step: what to track, then how.
        #
        # ONE WINDOW WITH TWO PAGES rather than two windows, because every control here
        # already shares state (the folder, the crops, the movie list) and two windows
        # would need that state synchronised between them, two places to be wrong.
        self._pages = QStackedWidget()
        shell.addWidget(self._pages, 1)

        def _page():
            """A page that SCROLLS. Qt reported "Unable to set geometry 779x772 ...
            Resulting geometry: 779x1154" on the experimenter's display: the window was
            taller than the screen, so the bottom of the page (Organise, and the
            buttons that lead onward) was simply not reachable. A window that cannot
            be resized small enough to fit is a window with hidden controls."""
            area = QScrollArea()
            area.setWidgetResizable(True)
            area.setFrameShape(QScrollArea.NoFrame)
            inner = QWidget()
            box = QVBoxLayout(inner)
            box.setContentsMargins(0, 0, 0, 0)
            area.setWidget(inner)
            self._pages.addWidget(area)
            return box

        layout = _page()

        # --- Fiji ------------------------------------------------------------
        box = QGroupBox("1 · Fiji")
        fiji_layout = QVBoxLayout(box)
        row = QHBoxLayout()
        self._fiji_label = QLabel("looking…")
        self._fiji_label.setWordWrap(True)
        row.addWidget(self._fiji_label, 1)
        fiji_layout.addLayout(row)

        # WHICH FIJI, AND IT HAD BETTER BE ASKABLE. `find_fiji` already reports every
        # install it saw: this machine has three, two of them TrackMate 7.11.1 and one
        # being updated to 8, and the window printed "1 other install(s) also found"
        # while offering no way to use it. A tracker version is a SETTING: two Fijis
        # produce results that nothing downstream can tell apart, so choosing between
        # them cannot be left to whichever the search happened to reach first.
        pick_row = QHBoxLayout()
        pick_row.addWidget(QLabel("use"))
        self._fiji_box = QComboBox()
        self._fiji_box.setToolTip(
            "Every Fiji found, in search order. The SMTRACKER_FIJI setting takes priority.")
        self._fiji_box.currentIndexChanged.connect(self._fiji_chosen)
        pick_row.addWidget(self._fiji_box, 1)
        browse = QPushButton("Other folder…")
        browse.setToolTip("Point at a Fiji folder the search did not reach.")
        browse.clicked.connect(self._browse_fiji)
        pick_row.addWidget(browse)
        fiji_layout.addLayout(pick_row)
        layout.addWidget(box)

        # --- movies ----------------------------------------------------------
        # THE PICKER ASKS FOR A PROJECT FOLDER; TRACKING NEEDS A CONDITION FOLDER.
        # The launcher's folder question is the analysis stage's question, and the two
        # are one level apart, so a user who lands here on the wrong folder must be able
        # to correct it WITHOUT restarting. The pattern is editable for the same reason:
        # `tif/<movie>/<movie>/*.tif` is this lab's layout, not a law.
        box = QGroupBox("2 · Movies")
        outer = QVBoxLayout(box)
        # WHAT IS IN THIS FOLDER, READ RATHER THAN ASSUMED. The window used to open on
        # "Prepared stacks" whatever it was pointed at, so a folder of raw acquisitions
        # reported zero movies, which reads as "your data is unreadable" when the truth
        # is "you are on the wrong tab". `organise.inspect_folder` says what is there and
        # the source starts accordingly; it is a DEFAULT, never a lock, because
        # re-cropping an already-prepared project is a real thing to want.
        self._folder_state = QLabel("not scanned yet")
        self._folder_state.setWordWrap(True)
        self._folder_state.setStyleSheet("font-weight: 600;")
        outer.addWidget(self._folder_state)
        # THE SOURCE IS RESOLVED, NOT ASKED. `inspect_folder` reads what is actually in
        # the folder and picks; the control is kept because the choice is REAL - see
        # `SOURCE_NOTE` - but it is behind a disclosure, because presenting a resolved
        # question as an open one is how "do we need this?" gets asked about a control
        # nobody has to touch.
        self._source_toggle = QCheckBox("change source / layout")
        self._source_toggle.setToolTip(SOURCE_NOTE)
        self._source_toggle.toggled.connect(self._show_source_controls)
        outer.addWidget(self._source_toggle)

        self._source_widget = QWidget()
        source_row = QHBoxLayout(self._source_widget)
        source_row.setContentsMargins(0, 0, 0, 0)
        source_row.addWidget(QLabel("source"))
        self._source_box = QComboBox()
        self._source_box.addItems(SOURCES)
        self._source_box.setToolTip(SOURCE_NOTE)
        self._source_box.currentTextChanged.connect(self._source_changed)
        source_row.addWidget(self._source_box, 1)
        outer.addWidget(self._source_widget)
        # A PARENT OF CONDITIONS. A session off the microscope is several sibling
        # folders, and this window is the door the launcher opens for preparation, so it
        # is where they have to be detected. Hidden when one condition was opened, so
        # that case looks exactly as it did before.
        cond_row = QHBoxLayout()
        self._conditions_label = QLabel("condition")
        self._conditions_box = QComboBox()
        self._conditions_box.setToolTip(
            "This folder holds several conditions. Each is tracked on its own, because "
            "the detection threshold is measured from that condition's noise.")
        self._conditions_box.currentTextChanged.connect(self._pick_condition)
        # NO SECOND ORGANISE BUTTON. This window already has one, with a mode combo
        # and a guard against filing files the crop panel currently has open (WinError
        # 32 on Windows). Adding another collided with its attribute name and would have
        # put two organise controls on one page. Both are gone now: saving the crop
        # ranges files every condition it saves, which is what the button was for.
        cond_row.addWidget(self._conditions_label)
        cond_row.addWidget(self._conditions_box, 1)
        outer.addLayout(cond_row)
        self._conditions_label.hide()
        self._conditions_box.hide()

        # ONE BAR PER CONDITION. Asked for after a fair question: with a parent open and
        # the proposer ticked, does it read all of them or only the active one? It reads
        # the ACTIVE one, and there was no way to see that. The overview makes the answer
        # visible instead of leaving it to be inferred from a panel that shows one folder.
        self._progress_box = QWidget()
        self._progress_rows = QVBoxLayout(self._progress_box)
        self._progress_rows.setContentsMargins(0, 0, 0, 0)
        self._progress_rows.setSpacing(2)
        self._progress_bars = {}
        outer.addWidget(self._progress_box)
        self._progress_box.hide()

        # ONE COLUMN PER CONDITION, one row per setting. Asked for in exactly this shape:
        # after the bleaching curve every condition still needs an interval, a
        # fluorophore, a pixel size and an NA, and setting them one window at a time hides
        # the thing that matters, which is whether they AGREE.
        #
        # Most sessions are imaged alike, so the ordinary case is to fill one column and
        # copy it. The button does that; the grid is what makes a real difference visible
        # instead of discovered later.
        self._calibrations = {}
        # WHICH CONDITION THE OPTICS BOXES CURRENTLY DESCRIBE, and it is initialised HERE
        # rather than on the first path that happens to set it. `_apply_condition_
        # calibration` reads it to decide whether the boxes moving is worth a log line;
        # clicking a condition before anything else had set it raised AttributeError and
        # took the whole window down. `_scan_raw` read the same attribute through
        # `getattr(..., None)`, which is what hid the missing initialisation -- a
        # defensive read at one site conceals absent state from every other site.
        self._calibration_from = None
        self._calib_table = QTableWidget(0, 0)
        self._calib_table.setAlternatingRowColors(True)
        self._calib_table.setMaximumHeight(170)
        self._calib_table.itemChanged.connect(self._calibration_edited)
        self._calib_copy = QPushButton("Copy the first condition to all")
        self._calib_copy.setToolTip(
            "Fill every other condition with the first condition's settings. Empty "
            "cells here do not erase values already entered elsewhere. Nothing is "
            "written to disk until you save the crop ranges.")
        self._calib_copy.clicked.connect(self._copy_calibration_to_all)
        self._calib_box = QWidget()
        calib_layout = QVBoxLayout(self._calib_box)
        calib_layout.setContentsMargins(0, 0, 0, 0)
        calib_layout.addWidget(QLabel("acquisition settings per condition"))
        calib_layout.addWidget(self._calib_table)
        # UNDER THE FIRST CONDITION'S COLUMN, because that is the column it copies. A
        # button stretched under the whole table reads as "copy the table", which is not
        # a thing it can do.
        calib_buttons = QHBoxLayout()
        calib_buttons.addSpacing(self._calib_table.verticalHeader().width())
        calib_buttons.addWidget(self._calib_copy)
        calib_buttons.addStretch(1)
        calib_layout.addLayout(calib_buttons)
        self._calib_box.hide()

        # THE SAME SHAPE FOR THE TRACKING SETTINGS. Conditions across, settings down.
        # The threshold is the one row that is MEASURED rather than chosen, so it gets
        # its own button: "not measured" on two rows out of three was the complaint that
        # started this, and visiting each condition to press a button is the work the
        # overview exists to remove.
        self._tracking = {}
        self._track_table = QTableWidget(0, 0)
        self._track_table.setAlternatingRowColors(True)
        self._track_table.setMaximumHeight(150)
        self._track_table.itemChanged.connect(self._tracking_edited)
        self._track_copy = QPushButton("Use these settings for all conditions")
        self._track_copy.setToolTip(
            "Copy the selected condition's radii and frame gap to every other. The "
            "threshold is left alone because it is measured from each condition's noise.")
        self._track_copy.clicked.connect(self._copy_tracking_to_all)
        self._track_measure = QPushButton("Compute thresholds for all conditions")
        self._track_measure.setToolTip(
            "Measure the LoG detection threshold for every condition from its own "
            "movies' noise. Each needs its crop ranges first, because the noise is "
            "measured inside the crop.")
        self._track_measure.clicked.connect(self._measure_all_thresholds)
        self._track_box = QWidget()
        track_layout = QVBoxLayout(self._track_box)
        track_layout.setContentsMargins(0, 0, 0, 0)
        track_layout.addWidget(QLabel("tracking settings per condition"))
        track_layout.addWidget(self._track_table)
        track_buttons = QHBoxLayout()
        track_buttons.addWidget(self._track_copy)
        track_buttons.addWidget(self._track_measure)
        track_layout.addLayout(track_buttons)
        self._track_box.hide()

        row = QHBoxLayout()
        self._movies_label = QLabel("not scanned yet")
        self._movies_label.setWordWrap(True)
        row.addWidget(self._movies_label, 1)
        choose = QPushButton("Choose folder…")
        choose.clicked.connect(self._choose_folder)
        row.addWidget(choose)
        # ASKED FOR: "there should be a tick box next open folder, automated crop range
        # detection photobleaching. so it does both, reading the files and already
        # deciding where to crop".
        #
        # ON BY DEFAULT, ruled by the experimenter ("propose crop while reading always
        # ticked in the beginning as default"). This reverses the panel's own long-
        # standing refusal to propose, so what that refusal protected has to be carried
        # by the WORDS instead: the proposal is a FLOOR, and `CropFloor` measured that
        # the crop actually chosen sits a median 2124 frames (50.5 per cent of the
        # acquisition) LATER. So a default proposal is systematically EARLIER than this
        # lab's own practice, and accepting it unread means tracking a denser regime
        # than usual. Every value stays editable, the rule is named on open, and the
        # panel prints that margin beside the floor.
        self._auto_crop = QCheckBox("propose crop while reading")
        self._auto_crop.setChecked(True)
        self._auto_crop.setToolTip(
            "Propose a crop range for each movie as its bleaching curve is read, using "
            "the rule shown in the crop panel below (default 'strictest'). Every "
            "proposed value stays editable. A proposal is the earliest defensible "
            "start, so read the curve before accepting one.")
        self._auto_crop.toggled.connect(self._auto_crop_toggled)
        row.addWidget(self._auto_crop)
        rescan = QPushButton("Rescan")
        rescan.clicked.connect(self._scan_movies)
        row.addWidget(rescan)
        # THE FOLDER PICKER BEFORE THE CONDITION PICKER, which is the order the work
        # goes in: choose a folder, then choose which of its conditions to look at. The
        # row is built here because it needs widgets defined above it, so it is INSERTED
        # at the right place rather than appended where it happens to be created.
        outer.insertLayout(outer.indexOf(self._source_widget) + 1, row)

        self._pattern_widget = QWidget()
        pattern_row = QHBoxLayout(self._pattern_widget)
        pattern_row.setContentsMargins(0, 0, 0, 0)
        pattern_row.addWidget(QLabel("layout"))
        self._pattern_edit = QLineEdit(DEFAULT_MOVIE_PATTERN)
        self._pattern_edit.setPlaceholderText(
            "auto: tif/<movie>.tif, then u-track's tif/<movie>/<movie>/")
        self._pattern_edit.setToolTip(
            "Leave empty to find the stacks automatically. A glob here is relative to "
            "the folder above and may match a multi-page TIFF or a folder of "
            "single-frame TIFFs.")
        self._pattern_edit.editingFinished.connect(self._scan_movies)
        pattern_row.addWidget(self._pattern_edit, 1)
        outer.addWidget(self._pattern_widget)
        self._show_source_controls(False)

        # ORGANISE BEFORE ANYTHING ELSE, because a microscope leaves a FLAT folder -
        # series001.nd2 beside series001.tif, ten of each - and every stage here expects
        # the lab's own raw/ + phase/ layout. Doing that rename by hand is the one step
        # where a mistake is silent: a `.tif` filed as an acquisition becomes a
        # twentieth movie, which this project has already measured.
        #
        # THE PLAN IS SHOWN AND NOTHING HAPPENS UNTIL A BUTTON IS PRESSED, and the mode
        # is chosen rather than assumed. `smtracker.organise` holds all of it; nothing
        # is decided in this class.
        # HIDDEN BY DEFAULT, and the reason is that saving already does it. Reported:
        # THE CROP LIVES HERE NOW. Tracking from raw needs exactly one thing from the
        # old preparation stage - the frame range - and nothing downstream consumes the
        # cropped stacks; they existed to feed u-track, which cannot read a .nd2. So the
        # decision is made where it is used, and `crops.json` stays the single record of
        # it so the two halves cannot disagree.
        from .croppanel import CropPanel

        self._crop_panel = CropPanel()
        self._crop_panel.changed.connect(self._crops_changed)
        # NOTHING THAT WRITES IS PRESSABLE MID-PASS. The curves arrive one movie at a
        # time, so saving or exporting while the pass runs produced output for the
        # movies read SO FAR, a partial result nobody asked for and nothing labelled
        # as partial. The same reason Organise refuses during the pass, and here the
        # button is simply not offered.
        self._crop_panel.reading_changed.connect(self._reading_changed)
        # THE BARS FOLLOW THE WORK. `changed` fires as each crop lands and
        # `reading_changed` at both ends of the pass, so the active row fills live while
        # the other rows keep the state read off disk.
        self._crop_panel.changed.connect(self._refresh_progress)
        # A READ SURVIVES A SWITCH. The panel forwards every movie it reads, tagged with
        # the condition it belongs to, so looking at another condition no longer abandons
        # the pass: reported as "if i pick a different condition while a condition is
        # being read, it stops reading".
        self._crop_panel.curve_arrived.connect(self._curve_arrived)
        self._crop_panel.end_frame_applied.connect(self._end_frame_everywhere)
        self._crop_panel.start_frame_applied.connect(self._start_frame_everywhere)
        self._crop_panel.pass_finished.connect(self._pass_finished)
        self._crop_box = QGroupBox("Crop range (raw acquisitions only)")
        crop_layout = QVBoxLayout(self._crop_box)
        crop_layout.addWidget(self._crop_panel)
        # CALIBRATION TRAVELS WITH THE RANGES, and neither has a default. A run
        # reproduced at the wrong pixel size or frame interval is wrong in a way nothing
        # downstream can detect: the numbers simply come out plausible.
        # --- optics ----------------------------------------------------------
        # NA AND EMISSION WAVELENGTH SET THE PSF, and the PSF sets two things this
        # window has been taking on faith: the detection radius (3 px because that is
        # what was used, not because anything derived it) and the LOCALISATION FLOOR,
        # which on this data decides whether the slow state's D may be quoted at all,
        # at sigma 30 nm the reference D sits 4.7x above the floor, at 76 nm below it.
        #
        # u-track RECORDS both and this project reads them. A raw movie has no such
        # file, so on the raw path they have to be entered, which is exactly why they
        # need a place here rather than a constant in the code.
        #
        # THE LABEL PRESET FILLS THE WAVELENGTH AND NEVER LOCKS IT. It also names what
        # changes about the EXPERIMENT, which a number cannot carry: a photoactivated
        # label arrives sparsely over the whole acquisition instead of bleaching down to
        # single molecules, so the crop rules above do not describe it.
        from .optics import preset_names as label_preset_names

        self._label = QComboBox()
        self._label.addItems(("(not set)",) + label_preset_names())
        self._label.currentTextChanged.connect(self._label_changed)
        # placed on page one, beside the pixel size, see `calib` above. All four
        # are CALIBRATION, and splitting them across two screens is what made the
        # pixel-size field read as arbitrary: the experimenter asked "where is the
        # fluorophore helper?" while it sat one page away from the field it helps.

        self._emission_nm = _CalibrationSpinBox()
        self._emission_nm.setRange(0.0, 1200.0)
        self._emission_nm.setDecimals(0)
        self._emission_nm.setSuffix(" nm")
        self._emission_nm.setSpecialValueText("not set")
        self._emission_nm.setToolTip(
            "Emission maximum. A preset fills it; edit it to match your filter set.")
        self._numerical_aperture = _CalibrationSpinBox()
        self._numerical_aperture.setRange(0.0, 2.0)
        self._numerical_aperture.setDecimals(2)
        self._numerical_aperture.setSingleStep(0.05)
        self._numerical_aperture.setSpecialValueText("not set")
        self._numerical_aperture.setToolTip(
            "Numerical aperture of the objective, as printed on it or recorded by "
            "u-track.")
        optics_row = QHBoxLayout()
        optics_row.addWidget(QLabel("emission"))
        optics_row.addWidget(self._emission_nm)
        optics_row.addWidget(QLabel("NA"))
        optics_row.addWidget(self._numerical_aperture)
        optics_row.addStretch(1)

        self._optics_note = QLabel()
        self._optics_note.setWordWrap(True)
        self._optics_note.setStyleSheet("color: #555;")

        calib = QHBoxLayout()
        calib.addWidget(QLabel("pixel size"))
        self._pixel_nm = _CalibrationSpinBox()
        self._pixel_nm.setRange(0.0, 10000.0)
        self._pixel_nm.setDecimals(1)
        self._pixel_nm.setSuffix(" nm")
        self._pixel_nm.setSpecialValueText("not set")
        self._pixel_nm.setToolTip(
            "Camera pixel size / (objective magnification x tube lens). It cannot "
            "be derived from NA. A .nd2 file may report a pixel size that leaves "
            "out the tube lens, so check it against your optics.")
        calib.addWidget(self._pixel_nm)
        pixel_help = QPushButton("?")
        pixel_help.setMaximumWidth(28)
        pixel_help.setToolTip("Work it out from the camera and the magnification.")
        pixel_help.clicked.connect(self._pixel_size_helper)
        calib.addWidget(pixel_help)
        calib.addWidget(QLabel("frame interval"))
        self._interval_ms = _CalibrationSpinBox()
        self._interval_ms.setRange(0.0, 100000.0)
        self._interval_ms.setDecimals(3)
        self._interval_ms.setSuffix(" ms")
        self._interval_ms.setSpecialValueText("not set")
        self._interval_ms.setToolTip(
            "Every D scales with 1/tau, so enter the measured frame interval.")
        calib.addWidget(self._interval_ms)
        calib.addStretch(1)
        # NOT SHOWN HERE ANY MORE. These are per-CONDITION settings and the grid below
        # the curves holds them per condition; two places to set one number is how the
        # two disagree. The widgets stay because everything downstream reads them: they
        # are now the working copy for whichever condition is selected, filled from the
        # grid, and the grid is what a person edits.
        self._calib_row_holder = QWidget()
        self._calib_row_holder.setLayout(calib)
        self._calib_row_holder.hide()
        crop_layout.addWidget(self._calib_row_holder)

        # THE OPTICS BELONG WITH THE CALIBRATION, not one page away. NA and the label's
        # emission wavelength set the PSF WIDTH, which is what u-track derives from them
        # (`psfSigma_`); the pixel size is a separate, stored quantity that no optical
        # formula recovers. Having them on one line is what makes that distinction
        # legible instead of surprising.
        optics_row = QHBoxLayout()
        optics_row.addWidget(QLabel("label"))
        optics_row.addWidget(self._label, 1)
        optics_row.addWidget(QLabel("emission"))
        optics_row.addWidget(self._emission_nm)
        optics_row.addWidget(QLabel("NA"))
        optics_row.addWidget(self._numerical_aperture)
        optics_row.addStretch(1)
        self._optics_row_holder = QWidget()
        self._optics_row_holder.setLayout(optics_row)
        self._optics_row_holder.hide()
        crop_layout.addWidget(self._optics_row_holder)
        crop_layout.addWidget(self._optics_note)

        save_row = QHBoxLayout()
        # ONE PRESS FOR THE TWO THINGS NOBODY WANTS SEPARATELY. Splitting them was an
        # over-correction: the curves were a SILENT side effect of saving the ranges,
        # which was the actual fault, and the fix for a hidden action is to NAME it, not
        # to make it a second press. The ranges and the record of how they were chosen
        # belong together. Exporting stacks stays separate: it writes gigabytes.
        # THE PROJECTION IS THE MACRO'S CHOICE, exposed. `Z Project...` offers Sum
        # Slices and Standard Deviation; the macro hard-codes Standard Deviation, which
        # is why every existing sum/ file IS one and the folder's NAME is the only thing
        # that says sum.
        from .movieprep import PROJECTIONS

        self._projection = QComboBox()
        self._projection.addItems([{"sd": "standard deviation",
                                    "sum": "sum"}[k] for k in PROJECTIONS])
        self._projection.setToolTip(
            "How the sum/ image is computed. Standard deviation highlights the pixels "
            "that vary, which is where molecules moved; a true sum is dominated by "
            "static background.")
        save_row.addWidget(QLabel("sum image"))
        save_row.addWidget(self._projection)

        self._save_crops = QPushButton("Save crop ranges, photobleaching curves + sum")
        self._save_crops.setToolTip(
            "Writes bleaching/crops.json (the ranges plus the calibration), "
            "bleaching/<movie>_full.csv, _cropped.csv, <movie>.png, and "
            "sum/<movie>.tif. No cropped stacks.")
        self._save_crops.clicked.connect(self._save_crops_clicked)
        save_row.addWidget(self._save_crops)
        # EXPORTING STACKS LIVES HERE NOW, not in the launcher. It used to be a stage of
        # its own, which was the shape of the code rather than of the work: tracking
        # here reads the raw file directly and needs only the crop range. But u-track
        # CANNOT read a .nd2, so the export is genuinely needed and dropping it from the
        # menu without rehoming it would have left a built, tested window reachable from
        # nothing, the defect this project keeps recording.
        # THREE OUTPUTS, THREE BUTTONS. They were one press and a silent side effect:
        # 'Save crop ranges' also wrote the curves, and the stacks were behind a button
        # named for the stacks alone. Each of these is wanted on its own: the ranges
        # for tracking from raw, the curves as the lab-book record, the stacks only to
        # feed u-track, so each is asked for on its own and says what it writes.
        self._export_stacks = QPushButton("Export cropped stacks (u-track)…")
        self._export_stacks.setToolTip(
            "Writes tif/<movie>.tif (one multi-page stack per movie) and sum/<movie>.tif. "
            "Only needed for u-track, which cannot read a .nd2; tracking here reads "
            "the raw file directly.")
        self._export_stacks.clicked.connect(self._export_stacks_clicked)
        save_row.addWidget(self._export_stacks)
        # THE COST IS SHOWN WHILE IT IS PAID, and it can be stopped. Writing 30 movies
        # is minutes; a window that simply freezes teaches people to kill it.
        self._export_progress = QProgressBar()
        self._export_progress.hide()
        save_row.addWidget(self._export_progress, 1)
        self._export_cancel = QPushButton("Cancel")
        self._export_cancel.setToolTip(
            "Stops after the movie being written, so no half-written stack is left.")
        self._export_cancel.clicked.connect(self._cancel_export)
        self._export_cancel.hide()
        save_row.addWidget(self._export_cancel)
        # CELL OUTLINES, OFFERED HERE BECAUSE THIS IS WHEN THEY BECOME POSSIBLE.
        # `prepare_movie` writes sum/<movie>.tif and `phase/` comes off the microscope,
        # so the moment the writers above have run, segmentation has both channels it
        # needs. Doing it BEFORE tracking is what lets the detection threshold be chosen
        # against the false-positive rate outside cells (`smtracker.detection` refuses
        # without meshes), makes linking ambiguity an IN-CELL number, and turns the
        # cell-filter loss into a pre-flight check rather than a post-mortem.
        #
        # AN OFFER, NEVER A GATE. Tracking stays reachable with no outlines at all --
        # a project may take them from Oufti, or not want them yet.
        self._outlines_button = QPushButton("Cell outlines…")
        self._outlines_button.setToolTip(
            "Opens the segmentation stage on this condition. Outlines are not required; "
            "with them the detection threshold can be checked against false positives "
            "outside cells and crowding is counted inside cells.")
        self._outlines_button.clicked.connect(self._open_outlines)
        save_row.addWidget(self._outlines_button)
        self._crop_status = QLabel()
        self._crop_status.setWordWrap(True)
        save_row.addWidget(self._crop_status, 1)
        # THE WRITERS COME AFTER THE SETTINGS, not inside the crop box above them.
        # Reported: "maybe its not saved cause i first save crop range etc and then
        # change the settings?" -- exactly right. The fluorophore, emission, NA, pixel
        # size and frame interval are written INTO crops.json at save time, and the
        # Save button sat ABOVE the grid holding them, so the natural order of the page
        # was: crop, save, then notice the settings and change them into memory only.
        #
        # It stays after the CURVES, which was the earlier ruling and still holds: the
        # crop is decided by looking, and two grids of numbers in front of that picture
        # put the wrong thing first. So the order is curves, settings, write.
        self._save_box = QWidget()
        self._save_box.setLayout(save_row)
        outer.addWidget(self._crop_box)
        self._crop_box.setVisible(self._source == SOURCE_RAW)

        # FILING THE MOVIES COMES LAST, and the reason is mechanical rather than
        # aesthetic. Reaching this page starts a background pass that OPENS every
        # acquisition to read its bleaching curve, and on Windows an open file cannot be
        # moved, the window already refuses to organise while that runs. So an Organise
        # button at the TOP is a button that is unusable at exactly the moment the eye
        # lands on it. Below the crop panel it sits where the work actually ends: the
        # curves have been read, the ranges are set, and filing is the step before
        # moving on.
        # THE SETTINGS COME AFTER THE CURVES. Ruled by the experimenter, and it is the
        # order of the work: the crop is decided by looking, and only then is there
        # anything to say about how the result should be tracked. They sat above the
        # curves, which put two grids of numbers in front of the picture they depend on.
        outer.addWidget(self._calib_box)
        # LAST on this page, so nothing that goes into crops.json is still unset when
        # the press that writes it is reached.
        outer.addWidget(self._save_box)
        self._save_box.setVisible(self._source == SOURCE_RAW)

        layout.addWidget(box)

        # --- on to the settings ----------------------------------------------
        forward = QHBoxLayout()
        self._movie_count = QLabel()
        forward.addWidget(self._movie_count, 1)
        # PREPARATION IS SOMETIMES THE WHOLE JOB. Someone exporting stacks for u-track
        # never wants the detector page, and a window whose only exit is 'onward' makes
        # them close it with the title bar and wonder whether that discarded anything.
        self._finish_button = QPushButton("Finish")
        self._finish_button.setToolTip(
            "Close the window. Everything already written stays.")
        self._finish_button.clicked.connect(self.close)
        forward.addWidget(self._finish_button)

        self._next_button = QPushButton("Detection and tracking  →")
        self._next_button.setDefault(True)
        self._next_button.clicked.connect(self._go_to_tracking)
        forward.addWidget(self._next_button)
        layout.addLayout(forward)
        layout.addStretch(1)

        # ===================== page 2: how to track ==========================
        layout = _page()

        back_row = QHBoxLayout()
        back = QPushButton("←  Movies")
        back.clicked.connect(lambda: self._pages.setCurrentIndex(0))
        back_row.addWidget(back)
        self._track_header = QLabel()
        self._track_header.setWordWrap(True)
        back_row.addWidget(self._track_header, 1)
        layout.addLayout(back_row)

        # THE ASK SITS AT THE TRANSITION, because this is where the consequence lands.
        # A button on the preparation page is available; it is not a question, and the
        # reason to answer it is on THIS page -- the detection threshold below can be
        # measured against the false-positive rate outside cells only when outlines
        # exist, and the linking ambiguity beside it is a whole-field number without
        # them (36.6 per cent against 3.3 on one lab condition, where one movie carries a
        # pile of out-of-cell signal).
        #
        # AN INLINE ROW, NEVER A MODAL. A QMessageBox blocks the event loop until
        # somebody clicks, which headless is forever -- it burned this suite's full
        # 900 s timeout once. And it never blocks the transition: outlines are optional
        # for tracking and may come from Oufti.
        self._outlines_prompt = QWidget()
        prompt_row = QHBoxLayout(self._outlines_prompt)
        prompt_row.setContentsMargins(0, 0, 0, 0)
        self._outlines_prompt_label = QLabel()
        self._outlines_prompt_label.setWordWrap(True)
        prompt_row.addWidget(self._outlines_prompt_label, 1)
        self._outlines_prompt_button = QPushButton("Segment cells now…")
        # THE SAME HANDLER AS THE PAGE-1 BUTTON. Two controls over one action is how
        # this window acquired a duplicate organise button that silently rebound the
        # first one's attribute; two buttons calling one method is not that.
        self._outlines_prompt_button.clicked.connect(self._open_outlines)
        prompt_row.addWidget(self._outlines_prompt_button)
        self._outlines_prompt_skip = QPushButton("Not now")
        self._outlines_prompt_skip.setToolTip(
            "Hides this until the outline state changes. Tracking does not need outlines.")
        self._outlines_prompt_skip.clicked.connect(self._skip_outlines_prompt)
        prompt_row.addWidget(self._outlines_prompt_skip)
        self._outlines_prompt.hide()
        layout.addWidget(self._outlines_prompt)

        # HOW BUSY THE FIELD IS, ON THE PAGE WHERE OUTLINES ARE BEING OFFERED. It sits
        # here rather than beside the curves because this is the last screen before
        # tracking, and because a movie unlike its neighbours is exactly what someone is
        # about to spend half an hour tracking without noticing.
        #
        # A WARNING, NEVER A GATE. Nothing on the run path reads it -- an AST test in
        # `test_launcher` already asserts that nothing on that path consults the outline
        # state, and this is the same kind of advisory. It names movies; the analyst
        # decides.
        self._crowding_label = QLabel()
        self._crowding_label.setWordWrap(True)
        self._crowding_label.setToolTip(
            "Detections per frame over the range that will be tracked, after merging "
            "peaks closer than one PSF.\n\n"
            "The crowded share is how many detections have another within the linking "
            "radius, so how often a link would have a second candidate. It is an upper "
            "bound, and a whole-field number because no outlines exist yet.")
        self._crowding_label.hide()
        layout.addWidget(self._crowding_label)

        # --- settings --------------------------------------------------------
        from .trackmate_run import (ADAPTIVE_BASE, FRAME_GAP_NOTE, LINKING_RADIUS_NOTE,
                                    LINKING_RADIUS_PX, UTRACK_EQUIVALENT,
                                    preset_choices)

        self._detect_box = box = QGroupBox("3 · Detection and linking")
        form = self._detect_form = QFormLayout(box)
        # THE BOX NO LONGER OFFERS CONDITION NAMES. "Glucose" and "Malate" were this
        # lab's conditions wearing a settings label; the choice that generalises is
        # whether the numbers come from the data or from you. See `smtracker.presets`.
        self._preset = QComboBox()
        self._preset.addItems(preset_choices())
        # RE-PICKING THE SAME ENTRY MUST STILL APPLY IT. `currentTextChanged` emits only
        # on a CHANGE, so after saving, when the box already sits on the new name,
        # selecting it again did nothing at all, and the obvious use (fiddle with the
        # numbers, then go back to the preset) silently failed. `activated` fires on any
        # user selection, so it covers the re-pick.
        #
        # THE FLAG IS SET BY THE SIGNAL, NOT BY THE WORK. Qt emits `currentTextChanged`
        # first and `activated` second, so one user click would otherwise apply twice.
        # An earlier version set the flag inside `_apply_preset`, which is also called
        # directly at startup, leaving it latched, so the FIRST genuine re-pick was
        # swallowed. A flag that outlives its event is not a flag.
        self._preset_just_changed = False
        self._preset.currentTextChanged.connect(self._preset_text_changed)
        self._preset.activated.connect(self._preset_activated)
        self._save_preset = QPushButton("Save…")
        self._save_preset.setToolTip(
            "Save these settings under a name and a comment, so the next replicate of the "
            "same experiment can start from them.")
        self._save_preset.clicked.connect(self._save_preset_clicked)
        preset_row = QHBoxLayout()
        preset_row.addWidget(self._preset, 1)
        preset_row.addWidget(self._save_preset)
        form.addRow("Preset", preset_row)
        # EVERY ROW BELOW THIS ONE IS DETAIL, and it is hidden once the per-condition
        # grid is on screen. See `_show_detail_rows`.
        self._detect_detail_from = form.rowCount()

        # SPOT RADIUS IS NOT THE LINKING RADIUS, and the window used to let those two
        # be confused. This one is how big a MOLECULE LOOKS - the LoG kernel matched to
        # the PSF - so it follows from emission wavelength, NA and pixel size.
        self._radius = QDoubleSpinBox()
        self._radius.setRange(0.1, 20.0)
        self._radius.setSingleStep(0.5)
        self._radius.setSuffix(" px")
        # A SANE STARTING VALUE, not the spinbox's own minimum. Left unset it opened at
        # 0.1 px, which is not a molecule at any magnification, and the settings stamp
        # dutifully reported it as a valid custom setting.
        self._radius.setValue(ADAPTIVE_BASE["radius_px"])
        self._radius.setToolTip(
            "How big a molecule looks, matched to the PSF. 'From optics' derives it "
            "from emission wavelength, NA and pixel size. It is not the linking "
            "radius below.")
        self._radius_from_optics = QPushButton("From optics")
        self._radius_from_optics.setToolTip(
            "Set the spot radius from the emission wavelength, NA and pixel size on "
            "the previous page.")
        self._radius_from_optics.clicked.connect(self._radius_from_optics_clicked)
        radius_row = QHBoxLayout()
        radius_row.addWidget(self._radius, 1)
        radius_row.addWidget(self._radius_from_optics)
        form.addRow("Spot radius", radius_row)

        self._threshold = QDoubleSpinBox()
        self._threshold.setRange(0.0, 100000.0)
        self._threshold.setDecimals(1)
        self._threshold.setSingleStep(5.0)
        # ZERO READS AS "NOT MEASURED", not as a number. Reported: "opening the tracking
        # window still doesnt automatically calculate the values, but keeps normal
        # settings". The adaptive compute cannot run before there are movies, on a RAW
        # folder with no crop ranges there are none, which is correct, and it said so
        # only in the log, while the box showed a confident 0.0. The same
        # `setSpecialValueText` convention the pixel size, frame interval and laser
        # power already use.
        self._threshold.setSpecialValueText("not measured")
        # COMPUTE, DON'T TYPE. A TrackMate LoG threshold is an absolute number and does
        # not transfer between datasets; u-track's criterion is statistical and does.
        # This measures the loaded movies' own LoG noise and applies the constant that
        # reproduces u-track's yield, so the number comes from THIS data. It fills the
        # box and never locks it - every setting stays editable.
        self._compute_threshold = QPushButton("Compute from movies")
        self._compute_threshold.setToolTip(
            "Set the threshold from the loaded movies' own noise. Reads a few frames "
            "per movie.")
        self._compute_threshold.clicked.connect(self._compute_threshold_clicked)
        threshold_row = QHBoxLayout()
        threshold_row.addWidget(self._threshold, 1)
        threshold_row.addWidget(self._compute_threshold)
        form.addRow("LoG detection threshold", threshold_row)
        # WHY IT IS NOT MEASURED, where the empty box is. Reported as "it doesn't auto
        # calculate the settings": adaptive was selected and nothing happened, because
        # the threshold measures the noise INSIDE each crop and this folder had no crop
        # ranges yet, so there were no usable movies to measure. That is correct and it
        # was invisible: the movies line said "no usable crop set" one page away, and
        # this page showed an empty box with no link between the two.
        self._threshold_note = QLabel("")
        self._threshold_note.setWordWrap(True)
        self._threshold_note.setStyleSheet("color: #888; font-size: 11px;")
        form.addRow("", self._threshold_note)

        # THE THIRD NUMBER, AND IT WAS NOT ON SCREEN AT ALL. Every run used the default
        # 5 px and nothing said so; the user asking "in utrack 5?" was asking about a
        # setting the window did not show. It is DERIVABLE: from the fastest
        # population's D, which is what the helper button reports.
        self._link = QDoubleSpinBox()
        self._link.setRange(0.5, 100.0)
        self._link.setSingleStep(0.5)
        self._link.setSuffix(" px")
        # the validated default, not u-track's cap, the two agree at 5 px, and that is
        # a measured coincidence rather than an inheritance. See LINKING_RADIUS_NOTE.
        self._link.setValue(LINKING_RADIUS_PX)
        self._link.setToolTip(
            "How far a molecule may move between frames: sqrt(4·D·τ) for the fastest "
            "population, times an allowance.\n\n" + LINKING_RADIUS_NOTE)
        # TWO ROUTES, AND THE MEASURED ONE IS THE BETTER ANSWER TO "D IS NOT KNOWN YET".
        # D is FITTED FROM step lengths, so a pilot run's steps are one level LESS
        # derived than D: no diffusion model in between, and no number carried over
        # from another microscope. "From D" stays for a folder with nothing tracked yet.
        # THE "From D..." BUTTON IS GONE. It asked for a diffusion coefficient in order
        # to choose the setting that DECIDES the diffusion coefficient, the circularity
        # the experimenter objected to, and the radius is no longer chosen that way at
        # all: 5 px is validated against a deposited reference on six proteins.
        # `trackmate_run.linking_distance_px` stays for scripts and for the docs.
        self._radius_check = QPushButton("Check wider…")
        self._radius_check.setToolTip(
            "Tracks a couple of movies at this radius and at a wider one, then reports "
            "what share of steps the current radius could not see. About two minutes.")
        self._radius_check.clicked.connect(self._radius_check_clicked)
        self._link_measure = QPushButton("Measure")
        self._link_measure.setToolTip(
            "Read the radius off a tracked result already in this folder: the "
            "step-length distribution and how often a second molecule sits within one "
            "radius. No D needed.")
        self._link_measure.clicked.connect(self._measure_link_clicked)
        link_row = QHBoxLayout()
        link_row.addWidget(self._link, 1)
        link_row.addWidget(self._link_measure)
        link_row.addWidget(self._radius_check)
        form.addRow("Linking radius", link_row)
        # WHAT THIS RADIUS CAN STILL MEASURE, at THIS frame interval. The radius buys
        # sqrt(D·tau), so the same 5 px that measures a fast state at 20 ms tops out
        # near a third of it at 50 ms, and the number does not stop existing there, it
        # stops meaning what it says.
        self._link_reach = QLabel()
        self._link_reach.setWordWrap(True)
        self._link_reach.setStyleSheet("color: #666; font-size: 11px;")
        form.addRow("", self._link_reach)
        self._link.valueChanged.connect(self._refresh_reach)
        self._interval_ms.valueChanged.connect(self._refresh_reach)
        self._pixel_nm.valueChanged.connect(self._refresh_reach)
        # `self._gap` does not exist yet (it is built further down) so its connection
        # is made there instead. A signal wired before its widget is constructed is an
        # AttributeError at window construction, i.e. the window simply never opens.


        self._gap = QSpinBox()
        self._gap.setRange(0, 10)
        self._gap.setSuffix(" frames")
        self._gap.setValue(ADAPTIVE_BASE["max_frame_gap"])
        # NOT DERIVABLE, AND THE WINDOW SAYS SO rather than offering a button that would
        # imply otherwise. What it has is a measurement, printed beside the box.
        self._gap.setToolTip(FRAME_GAP_NOTE)
        self._gap_note = QLabel(
            "A detector setting; it cannot be derived from the optics.")
        # THE GAP IS A TIME, NOT A FRAME COUNT, and the box cannot say so on its own.
        # Carried from a 20 ms project to a 50 ms one, "3 frames" silently becomes 2.5x
        # the unobserved motion. Filled by `_refresh_reach` once tau is known.
        self._gap_time = QLabel()
        self._gap_time.setWordWrap(True)
        self._gap_time.setStyleSheet("color: #666; font-size: 11px;")
        self._gap.valueChanged.connect(self._refresh_reach)
        self._gap_note.setWordWrap(True)
        self._gap_note.setStyleSheet("color: #666; font-size: 11px;")
        gap_col = QVBoxLayout()
        gap_col.addWidget(self._gap_time)
        gap_col.addWidget(self._gap)
        gap_col.addWidget(self._gap_note)
        form.addRow("Max frame gap", gap_col)

        for widget in (self._radius, self._threshold, self._gap, self._link,
                       self._emission_nm, self._numerical_aperture):
            widget.valueChanged.connect(self._refresh_stamp)
        # A HAND EDIT DISCARDS THE PROVENANCE, because it is no longer true. A computed
        # threshold nudged by 5 is a typed threshold, and a preset saved afterwards must
        # not claim it was measured.
        self._threshold_provenance = "typed"
        self._threshold.valueChanged.connect(self._threshold_typed)

        self._stamp = QLabel()
        self._stamp.setWordWrap(True)
        self._stamp.setStyleSheet("color: #555;")
        form.addRow("", self._stamp)
        layout.addWidget(box)
        # THE TRACKING GRID BELONGS ON THIS PAGE, beside the settings it holds per
        # condition. It was on page 1 with the crop, where the numbers it shows have
        # nothing to do with anything else on screen.
        layout.addWidget(self._track_box)

        # --- run -------------------------------------------------------------
        self._run_box = box = QGroupBox("4 · Run")
        run_layout = QVBoxLayout(box)
        row = QHBoxLayout()
        # WHICH CONDITION, said out loud. "Track this condition" meant whichever
        # folder the panel happened to be showing, and this page does not show it -- the
        # overview rows are on the page before. A button naming a condition it does not
        # name is how a run lands on the wrong folder.
        self._run_condition_box = QComboBox()
        self._run_condition_box.setToolTip(
            "Which condition the single-condition run tracks. Each condition uses its "
            "own threshold and spot radius.")
        self._run_condition_box.currentTextChanged.connect(
            lambda _t: self._refresh_run_buttons())
        self._run_button = QPushButton("Track this condition")
        self._run_button.clicked.connect(self._run)
        row.addWidget(self._run_condition_box)
        row.addWidget(self._run_button)
        # EVERY CONDITION, ONE AFTER ANOTHER. Asked for: "a button track all conditions,
        # that tracks first condition first, then 2nd etc and stops when all conditions
        # are tracked." A queue, not a loop -- see `_run_all_clicked`.
        self._run_all_button = QPushButton("Track all conditions")
        self._run_all_button.setToolTip(
            "Tracks every condition in turn, in the order shown, each with its own "
            "settings, and stops when the last one is done.")
        self._run_all_button.clicked.connect(self._run_all_clicked)
        row.addWidget(self._run_all_button)
        self._estimate = QLabel()
        row.addWidget(self._estimate, 1)
        run_layout.addLayout(row)
        self._run_condition_box.hide()
        self._run_all_button.hide()
        #: What `_refresh_run_buttons` restores the single-condition button to. The
        #: readiness check lives in `_refresh_estimate`; the queue must not override it.
        self._run_button_ready = False

        self._overwrite = QCheckBox(
            "overwrite an existing trackmate/ folder (off: the run refuses, so two "
            "settings are never mixed in one folder)")
        run_layout.addWidget(self._overwrite)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        run_layout.addWidget(self._progress)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMinimumHeight(160)
        run_layout.addWidget(self._log)

        # A SECOND EXIT, AND IT NEEDS ITS OWN NAME. Page 1 already has `_finish_button`
        # for the case where preparation was the whole job; reusing that attribute here
        # left TWO buttons under ONE name and orphaned the first, the collision this
        # project keeps recording, this time inside a single window.
        # HIDDEN UNTIL THERE IS SOMETHING TO FINISH: a "Done" pressable before anything
        # ran invites closing mid-way and reads as if the stage were optional.
        self._done_button = QPushButton("Done: tracking finished  ✓")
        self._done_button.setToolTip(
            "Close the tracking window. The XMLs stay in trackmate/ for the analysis "
            "stage.")
        self._done_button.clicked.connect(self._done_clicked)
        self._done_button.setVisible(False)
        run_layout.addWidget(self._done_button)
        layout.addWidget(box, 1)
        self._pages.setCurrentIndex(0)

        self.resize(760, 720)
        # SMALLER THAN THE CONTENT ON PURPOSE: the pages scroll, so the window must be
        # allowed to be shorter than they are.
        self.setMinimumSize(640, 420)
        self._find_fiji()
        # BEFORE the first scan, so the scan happens on the right source rather than
        # reporting zero movies and then being corrected.
        #
        # THE CONDITION CHECK RUNS HERE TOO, not only behind "Choose folder…". This is
        # the door the launcher opens for preparation, so a root handed in at startup is
        # the common case, and detecting a parent only on the button would leave the
        # feature absent on the path most people are on.
        if self._parent_dir is not None:
            self._detect_conditions(self._parent_dir)
        self._apply_root_state()
        self._refresh_reach()
        self._apply_folder_state()
        self._scan_movies()
        # SAID AT STARTUP, because the box is ticked before its signal is connected, so
        # nothing announces a default that is now doing something on the user's behalf.
        # An unannounced automatic crop is precisely what this panel refused for years.
        if self._auto_crop.isChecked():
            self._auto_crop_toggled(True)
        self._reload_presets(select=self._preset.currentText())
        self._apply_preset(self._preset.currentText())

    # -- Fiji -----------------------------------------------------------------
    def _find_fiji(self):
        """Locate Fiji, honouring a folder the user picked over the search order."""
        from .fiji import describe_absence, find_fiji

        chosen = getattr(self, "_fiji_choice", None)
        self._install = find_fiji(extra_paths=(chosen,) if chosen else ())
        if self._install is None:
            self._fiji_label.setText(describe_absence())
        else:
            self._fiji_label.setText(self._install.stamp())
        self._fill_fiji_choices()
        self._refresh_stamp()

    def _fill_fiji_choices(self):
        """List every install `find_fiji` saw, with the one in use selected.

        The combo is populated from the RESULT rather than from a second search, so
        what it offers and what is running cannot disagree.
        """
        self._fiji_box.blockSignals(True)
        self._fiji_box.clear()
        if self._install is None:
            self._fiji_box.addItem("none found")
            self._fiji_box.setEnabled(False)
        else:
            roots = [self._install.root, *self._install.other_candidates]
            for root in roots:
                self._fiji_box.addItem(str(root))
            self._fiji_box.setCurrentIndex(0)
            self._fiji_box.setEnabled(len(roots) > 1)
        self._fiji_box.blockSignals(False)

    def _fiji_chosen(self, index):
        if index < 0 or self._install is None:
            return
        self._use_fiji(Path(self._fiji_box.itemText(index)))

    def _browse_fiji(self):
        picked = QFileDialog.getExistingDirectory(
            self, "Fiji folder", str(getattr(self, "_fiji_choice", "") or ""))
        if picked:
            from .fiji import resolve_root

            # A download's outer folder resolves to the Fiji inside it, so the choice is
            # compared with the install it names, not with the folder picked.
            hit = resolve_root(picked)
            self._use_fiji(hit[0] if hit is not None else Path(picked))

    def _use_fiji(self, root: Path):
        """Switch installs, and SAY when the chosen one cannot track.

        A folder that is not a Fiji, or a Fiji without TrackMate, is reported here
        rather than at the first movie: the run is a quarter of an hour and finding out
        then is finding out too late.
        """
        self._fiji_choice = root
        self._find_fiji()
        # THE CHOICE MUST BE SEEN TO TAKE EFFECT, and `find_fiji` is not the place to
        # check that: given a folder holding no Fiji it falls through to its own search
        # and returns a DIFFERENT install, correct for finding Fiji, wrong as an answer
        # to "use this one". Without this the window would report the old install's path
        # under the new choice, which reads as success. Found by the test, not by eye.
        from .fiji import launcher_hint

        took_effect = self._install is not None and self._install.root == root
        if not took_effect:
            self._fiji_choice = None
            self._find_fiji()
            self._fiji_label.setText(
                f"No Fiji executable in {root}, so the choice was not applied: "
                f"still using {self._install.root if self._install else 'nothing'}. "
                f"Choose {launcher_hint()}.")
            self._log.appendPlainText(f"Fiji: {root} rejected, no executable in it")
            return
        if not self._install.has_trackmate:
            self._fiji_label.setText(
                f"{self._install.stamp()}\nTrackMate was not found in this install, so "
                f"it cannot track. Pick another, or update this one through Fiji's own "
                f"updater.")
        self._log.appendPlainText(f"Fiji: {root}")

    def _root_label(self) -> str:
        if self._root is None:
            return "<b>no folder chosen</b>: press “Choose folder…” below"
        return f"<b>{self._root}</b>"

    def _apply_root_state(self) -> None:
        """Everything that reads the folder is off until there IS one.

        A control that cannot work is worse than an absent one, the rule the GPU box
        and the segmentation Run button already follow. Section 1 (Fiji) stays live
        because finding Fiji has nothing to do with which movies are being tracked.
        """
        ready = self._root is not None
        self._header.setText(self._root_label())
        for box in (self._crop_box, self._detect_box, self._run_box):
            box.setEnabled(ready)

    # -- movies ---------------------------------------------------------------
    def _choose_folder(self):
        """Re-point at the condition folder without restarting the application."""
        picked = QFileDialog.getExistingDirectory(
            self, "Folder holding this condition's movies",
            str(self._root) if self._root is not None else "")
        if not picked:
            return
        self._open_root(Path(picked))

    def _open_root(self, folder):
        """THE ONE DOOR into a folder, whichever route asked for it.

        A parent of conditions is detected here rather than in each caller, so opening
        with `--root` and pressing "Choose folder…" cannot disagree. The prep window
        learned the same lesson the hard way an hour ago: a capability reachable on one
        path only is a capability silently ignored.
        """
        self._root = Path(folder)
        self._parent_dir = Path(folder)
        self._detect_conditions(self._parent_dir)
        self._apply_root_state()
        self._crop_root = None            # a new folder means a new crop panel
        self._apply_folder_state()
        if self._source == SOURCE_RAW:
            self._load_crop_panel()
        self._scan_movies()

    def _detect_conditions(self, parent) -> None:
        """Fill the condition box when ``parent`` holds several, and offer organising.

        Everything here is advisory: a folder must still open when the layout is
        unfamiliar, so a failure leaves the controls hidden rather than blocking.
        """
        self._conditions = {}
        try:
            from .organise import discover_project_folders, needs_organising

            found = discover_project_folders(parent)
            loose = needs_organising(parent)
        except Exception as exc:                          # noqa: BLE001 - REPORTED
            # NAMED, NOT SWALLOWED. This bare except used to set `found = []` and say
            # nothing, so a folder that failed to resolve looked exactly like a folder
            # with nothing in it -- and the window then inspected the PARENT, found no
            # acquisitions there, and settled on the prepared-stack source. Three
            # symptoms, one silent line: "nothing trackable", the wrong source, and no
            # condition. Rule 3 of this project is FAIL LOUD, and this was the exception.
            found, loose = [], []
            self._log.appendPlainText(
                f"could not resolve {parent} into conditions: "
                f"{type(exc).__name__}: {exc}")
        # WHAT WAS RESOLVED, ALWAYS, because "no subfolder was a condition" and "the scan
        # failed" and "this folder IS the condition" are three different states that
        # produced one identical screen.
        self._log.appendPlainText(
            f"{parent}: " + (", ".join(st.folder.name for st in found)
                             if found else "no condition folder found")
            + f" ({len(found)})")
        # KEPT FOR THE VISIBLE LABEL. The log is not on this page, and the three states
        # this distinguishes -- resolved, nothing found, scan failed -- all produced the
        # same "Nothing trackable" line, which is what made this untraceable from a
        # screenshot: one sentence for three different faults.
        subfolders = []
        try:
            subfolders = sorted(q.name for q in Path(parent).iterdir() if q.is_dir())
        except OSError:
            pass
        if found:
            self._detect_note = ""
        elif subfolders:
            shown = ", ".join(subfolders[:4]) + (" ..." if len(subfolders) > 4 else "")
            self._detect_note = (
                f"{len(subfolders)} subfolder(s) here ({shown}) and none holds a "
                f"readable acquisition or a prepared stack. Open one directly, or press "
                f"Rescan.")
        else:
            self._detect_note = "no subfolders here to look in."
        names = [state.folder.name for state in found]
        # ONE SUBFOLDER IS A CONDITION TOO, and this gate used to read `> 1`. Discovery
        # already resolved it correctly -- `discover_project_folders` returns ['Alpha'] --
        # and the window then threw the answer away: `_conditions` stayed empty and
        # `_root` stayed on the PARENT, so `acquisition_folder` looked for the movies
        # beside the condition instead of inside it and found none. A project holding a
        # single condition opened on the wrong folder.
        #
        # The PICKER stays hidden below two conditions, which was always right: "this
        # condition" is unambiguous there and a one-entry dropdown cannot help. What was
        # wrong is that hiding the CONTROL also dropped the STATE behind it.
        if found:
            self._conditions = {state.folder.name: state.folder for state in found}
            self._conditions_box.blockSignals(True)
            self._conditions_box.clear()
            self._conditions_box.addItems(names)
            self._conditions_box.blockSignals(False)
            # THE PICKER STAYS HIDDEN and is the model, not a control -- the rows below
            # are what you click. See `_ConditionRow`.
            self._conditions_label.hide()
            self._conditions_box.hide()
            # The FIRST one is loaded, so the window is never showing a parent it cannot
            # scan while a picker sits above it saying nothing happened.
            self._root = self._conditions[names[0]]
            # EVERY CONDITION GETS READ, not only the one on screen. Ruled by the
            # experimenter: "just go through all conditions, switching the condition just
            # changes which one i can look at in the bleaching curve browser".
            self._to_read = [x for x in names if x not in self._session_read]
            for holder in list(self._progress_bars):
                self._progress_bars.pop(holder, None)
            while self._progress_rows.count():
                item = self._progress_rows.takeAt(0)
                if item.widget() is not None:
                    item.widget().deleteLater()
        else:
            self._conditions_label.hide()
            self._conditions_box.hide()
            self._calibrations = {}
            self._calib_box.hide()
            self._tracking = {}
            self._track_box.hide()
        if len(self._conditions) > 1:
            try:
                from .movieprep import read_calibrations

                self._calibrations = read_calibrations(parent)
            except Exception:                             # noqa: BLE001 - advisory
                self._calibrations = {n: {} for n in self._conditions}
            self._refresh_calibration_table()
            # FROM THE BOXES, not from `current_settings()`. That returns None until a
            # threshold exists, which is precisely the state a fresh session opens in, so
            # seeding from it left every radius blank on exactly the folders this grid is
            # for. The boxes always hold something.
            seed = {"radius_px": self._radius.value() or None,
                    "linking_max_distance_px": self._link.value() or None,
                    "max_frame_gap": int(self._gap.value())}
            seeded = {n: {k: v for k, v in seed.items() if v is not None}
                      for n in self._conditions}
            # KEEP WHAT IS ALREADY MEASURED. Re-detecting conditions must not throw away
            # a threshold somebody waited for.
            for name, values in seeded.items():
                values.update({k: v for k, v in self._tracking.get(name, {}).items()
                               if v is not None})
                if name in self._thresholds:
                    values["threshold"] = self._thresholds[name]
            self._tracking = seeded
            self._refresh_tracking_table()
        self._refresh_progress()

    def _show_detail_rows(self, visible: bool) -> None:
        """Show or hide the numeric half of the detection box.

        RULED: "we don't need the big detection and linking part on top anymore, we just
        need the preset settings for each condition as top, adaptive standard, with a way
        to save and load."

        Hidden only when the per-condition GRID is on screen, because that grid is then
        the one place the numbers live and two editable copies of one number is how they
        come to disagree. With a single condition there is no grid, so these rows are the
        only way to set a threshold or a radius and they stay.

        The widgets are kept either way -- they back `settings_from_preset` and seed the
        grid -- so this hides a control, not a value. Same move as the condition combo.
        """
        form = getattr(self, "_detect_form", None)
        if form is None:
            return
        first = getattr(self, "_detect_detail_from", 1)
        for row in range(first, form.rowCount()):
            try:
                form.setRowVisible(row, visible)
            except (AttributeError, TypeError):          # Qt < 6.4
                for role in (QFormLayout.LabelRole, QFormLayout.FieldRole):
                    item = form.itemAt(row, role)
                    if item is not None and item.widget() is not None:
                        item.widget().setVisible(visible)
        self._detect_box.setTitle(
            "3 · Detection preset"
            if not visible else "3 · Detection and linking")

    def _refresh_tracking_table(self) -> None:
        """Redraw the tracking grid from `self._tracking`.

        The SPOT RADIUS is recomputed here from each condition's own optics rather than
        read back from whatever seeded the grid. Reported: "its always 3 but in the top
        part of tracking, it shows 2.76" -- the grid was seeded from the preset's flat
        3.0 while the optics box derived 2.80 from the emission, NA and pixel size in
        the calibration grid beside it. A typed value still wins, so the column can be
        overridden; it is only the DEFAULT that now follows the optics.
        """
        from .trackmate_run import TRACKING_FIELDS, spot_radius_per_condition

        if len(self._conditions) < 2:
            self._track_box.hide()
            self._show_detail_rows(True)
            self._refresh_run_conditions()
            return
        self._show_detail_rows(False)
        self._refresh_run_conditions()
        derived = spot_radius_per_condition(self._calibrations)
        for name, radius in derived.items():
            row = self._tracking.setdefault(name, {})
            if not self._radius_typed.get(name):
                row["radius_px"] = round(radius, 2)
        # NAMED, ONCE. A condition with no fluorophore keeps the preset's 3.0, and on
        # the real folder that is what "the spot radius is always 3" turned out to be:
        # one condition had an emission wavelength saved and derived 2.76, the other two
        # had none and showed a fallback with nothing marking it as one. A default
        # printed in a column beside real measurements reads as a measurement.
        undecided = [n for n in self._conditions
                     if n not in derived and not self._radius_typed.get(n)]
        if undecided and getattr(self, "_said_no_optics", None) != sorted(undecided):
            self._said_no_optics = sorted(undecided)
            self._log.appendPlainText(
                "spot radius not derived for " + ", ".join(sorted(undecided))
                + ": each needs a fluorophore (or an emission wavelength) and an NA in "
                  "the settings grid. Until then the column shows the preset's value, "
                  "which suits mNeonGreen and is off by 0.7 px for a red dye.")
        # ZERO IS A VALUE HERE, not an absence: a frame gap of 0 is u-track's own
        # setting. Only None means nothing has been decided.
        self._fill_grid(
            self._track_table, TRACKING_FIELDS, self._tracking,
            on_copy_row=lambda key: self._copy_row_to_all(
                self._tracking, key, self._refresh_tracking_table,
                dict((k, lab) for k, lab, _u in TRACKING_FIELDS)[key]),
            render=lambda _key, value: "" if value is None else f"{float(value):g}")
        self._track_box.show()

    def _tracking_edited(self, item) -> None:
        from .trackmate_run import TRACKING_FIELDS

        names = list(self._conditions)
        if item.row() >= len(TRACKING_FIELDS) or not 1 <= item.column() <= len(names):
            return
        key, label, _unit = TRACKING_FIELDS[item.row()]
        name = names[item.column() - 1]
        text = item.text().strip()
        if not text:
            self._tracking.setdefault(name, {})[key] = None
            return
        try:
            value = float(text)
        except ValueError:
            self._log.appendPlainText(
                f"{name}: {label!r} must be a number, got {text!r}; not changed")
            self._refresh_tracking_table()
            return
        self._tracking.setdefault(name, {})[key] = (
            int(value) if key == "max_frame_gap" else value)
        if key == "radius_px":
            # A TYPED RADIUS STICKS. The derived value fills the column, so without this
            # the next redraw would silently put the optics' answer back over the one
            # somebody deliberately entered.
            self._radius_typed[name] = True
        if key == "threshold":
            self._thresholds[name] = value
            # TYPED MEANS ONE NUMBER FOR THE CONDITION. Leaving the per-movie table in
            # place would show the typed value in the grid while the run used the
            # measured ones -- the grid saying one thing and the tracks another, which
            # is the defect this window has now had three times.
            self._typed_thresholds.add(name)
            self._movie_thresholds.pop(name, None)
            self._refresh_progress()

    def _copy_tracking_to_all(self) -> None:
        """Radii and gap to every condition. NOT the threshold, NOT the spot radius.

        THE FIRST CONDITION, the column the button sits under. It read the hidden
        condition combo, so the same press did different things depending on which row
        was last clicked, and it did nothing at all when that condition was the empty
        one -- reported as "use these settings for all conditions button not working
        here". The calibration grid's copy button had already been fixed this way; this
        one was missed.

        The SPOT RADIUS is excluded for the same reason as the threshold: it is derived
        from each condition's own emission wavelength, NA and pixel size, so copying it
        would put one condition's optics over another's and quietly undo the per-column
        calculation.
        """
        from .trackmate_run import TRACKING_FIELDS

        names = list(self._conditions)
        if not names:
            return
        source = names[0]
        keep_own = {"threshold", "radius_px"}
        template = {k: v for k, v in self._tracking.get(source, {}).items()
                    if v is not None and k not in keep_own}
        if not template:
            self._log.appendPlainText(
                f"{source} has no linking radius or frame gap set yet, so there is "
                f"nothing to copy. The threshold and the spot radius are never copied: "
                f"both are measured or derived per condition.")
            return
        for name in names:
            self._tracking.setdefault(name, {}).update(template)
        self._refresh_tracking_table()
        shown = ", ".join(
            f"{label} {template[key]:g}" for key, label, _u in TRACKING_FIELDS
            if key in template)
        self._log.appendPlainText(
            f"every condition uses {source}'s {shown}. The threshold and the spot "
            f"radius are not copied: the threshold is measured from each condition's own noise and "
            f"the spot radius is derived from its own optics.")

    def _measure_all_thresholds(self) -> None:
        """Measure the LoG threshold for every condition, from its own movies.

        Conditions whose ranges are incomplete are NAMED rather than skipped: a silent
        omission here leaves a row reading "not measured" with nothing to say whether it
        failed or was never attempted.
        """
        if not self._conditions:
            return
        self._stash_condition()
        here = self._root.name if self._root is not None else None
        blocked = []
        # QUEUED, NOT LOOPED TO COMPLETION: each measurement runs off the GUI thread
        # (`_NoiseJob`), one at a time, and the summary line is written when the last
        # one lands (`_noise_batch_done`).
        self._noise_batch = []
        self._noise_batch_here = here
        for name in list(self._conditions):
            movies, crops = self._measurable_movies(name)
            if not movies:
                blocked.append(name)
                continue
            self._compute_threshold_clicked(owner=name)
        if blocked:
            self._log.appendPlainText(
                "not measured, because every movie needs a crop range first: "
                + ", ".join(blocked))
        if self._noise_job is None and not self._noise_queue:
            self._noise_batch_done()

    def _noise_batch_done(self) -> None:
        """The summary of a "measure all" press, once its last measurement landed."""
        done, here = self._noise_batch or [], self._noise_batch_here
        self._noise_batch, self._noise_batch_here = None, None
        if here:
            self._apply_condition_calibration(here)
        self._refresh_tracking_table()
        self._refresh_progress()
        if done:
            self._log.appendPlainText(
                "measured: " + ", ".join(f"{n} {self._thresholds[n]:.0f}" for n in done
                                         if n in self._thresholds))

    def _apply_condition_calibration(self, name) -> None:
        """Put one condition's grid values into the boxes, ranges or no ranges.

        `_scan_raw` reads the calibration out of `crops.json`, and returns EARLY when a
        condition has no saved ranges -- so a condition that has never been cropped never
        reached that code and kept whatever the last one left in the boxes. That is the
        case the copy button exists for, so it has to work before anything is saved.
        """
        from .movieprep import CALIBRATION_FIELDS

        values = self._calibrations.get(name)
        if not values:
            return
        moved = []
        boxes = {"pixel_size_nm": (self._pixel_nm, 1.0),
                 "frame_interval_s": (self._interval_ms, 1000.0),
                 "emission_nm": (self._emission_nm, 1.0),
                 "numerical_aperture": (self._numerical_aperture, 1.0)}
        for key, _label, _unit in CALIBRATION_FIELDS:
            value = values.get(key)
            if value is None:
                continue
            if key == "label":
                index = self._label.findText(str(value))
                if index >= 0:
                    self._label.setCurrentIndex(index)
                continue
            box, scale = boxes[key]
            new_value = float(value) * scale
            digits = box.decimals() if hasattr(box, "decimals") else 6
            old_value = box.value()
            if old_value and round(old_value, digits) != round(new_value, digits):
                moved.append(f"{_label} {old_value:g} -> {new_value:g}")
            box.setValue(new_value)
            box.setToolTip(f"from the per-condition settings grid ({name})")
        if moved and self._calibration_from not in (None, name):
            # THE BOXES JUST CHANGED AND NOBODY PRESSED ANYTHING. This is the one change
            # on screen a person did not make and that moves every D, so it is said here,
            # where it happens, rather than left to the set-level line that fires once at
            # load. Applying the grid used to suppress it by settling `_calibration_from`
            # before `_scan_raw` could compare.
            self._log.appendPlainText(
                f"calibration follows the condition: {name} uses "
                + ", ".join(moved)
                + ". These conditions were not imaged alike, so comparing their D "
                  "values also compares acquisitions.")
        self._calibration_from = name

    def _fill_grid(self, table, fields, values, *, on_copy_row, render) -> None:
        """Draw one settings grid: a copy button, then one column per condition.

        COLUMN 0 IS THE ROW'S OWN COPY BUTTON. Asked for directly, and it is the finer
        tool the whole-column button is not: most sessions share an objective and a frame
        rate while differing in the fluorophore, so copying ONE row is the common
        correction and copying all of them would undo the difference that matters.
        """
        names = list(self._conditions)
        table.blockSignals(True)
        table.setRowCount(len(fields))
        table.setColumnCount(len(names) + 1)
        table.setHorizontalHeaderLabels([""] + names)
        table.setVerticalHeaderLabels(
            [f"{label} ({unit})" if unit else label for _key, label, unit in fields])
        # WHAT THE ROW MEANS, on the row itself. Only a number that is a SUMMARY needs
        # one, and the threshold is exactly that: the cell holds the MEDIAN of the
        # per-movie values, and a reader who does not know reads it as the value every
        # movie got.
        for r, (key, _label, _unit) in enumerate(fields):
            note = ROW_NOTES.get(key)
            header = table.verticalHeaderItem(r)
            if note and header is not None:
                header.setToolTip(note)
        for r, (key, label, _unit) in enumerate(fields):
            button = QPushButton("→ all")
            button.setToolTip(
                f"Copy the first condition's {label} to every other condition. Only this "
                f"row.")
            button.clicked.connect(lambda _checked=False, k=key: on_copy_row(k))
            table.setCellWidget(r, 0, button)
            for c, name in enumerate(names, start=1):
                if key == "label":
                    # A PICKER, because the fluorophore is a choice from a list and it
                    # FILLS the emission below it. Typing a name would leave the
                    # wavelength to be looked up by hand, which is the step the preset
                    # exists to remove.
                    from .optics import preset_names

                    combo = QComboBox()
                    combo.addItems(["(not set)"] + list(preset_names()))
                    current = values.get(name, {}).get(key)
                    if current:
                        found = combo.findText(str(current))
                        if found >= 0:
                            combo.setCurrentIndex(found)
                    combo.currentTextChanged.connect(
                        lambda text, who=name: self._label_picked(who, text))
                    table.setCellWidget(r, c, combo)
                    continue
                table.setItem(r, c, QTableWidgetItem(
                    render(key, values.get(name, {}).get(key))))
        table.resizeColumnToContents(0)
        table.blockSignals(False)

    def _label_picked(self, condition, text) -> None:
        """A fluorophore choice fills that condition's emission wavelength.

        FILLS, never locks: the preset is a convenience and the number stays editable,
        because a filter set or a variant can move it. Clearing the picker clears the
        label and LEAVES the wavelength, since a number somebody may have typed is not
        the preset's to remove.
        """
        from .optics import preset

        values = self._calibrations.setdefault(condition, {})
        if not text or text == "(not set)":
            values["label"] = None
            self._write_calibration_if_saved(condition)
            return
        values["label"] = text
        try:
            wavelength = float(preset(text).emission_nm)
        except Exception:                                 # noqa: BLE001 - advisory
            self._write_calibration_if_saved(condition)
            return
        # A PRESET REPLACING A RECORDED NUMBER SAYS SO. The catalogue value is a
        # nominal peak; a wavelength already in the calibration was read from the
        # acquisition, so overwriting it silently swaps a measurement for a lookup.
        # Measured on this project's own data: 509 nm from the .nd2 against mNeonGreen's
        # 517, which moves the derived spot radius 2.76 -> 2.80 px. Small, and the point
        # is that nothing said it had happened.
        recorded = values.get("emission_nm")
        if recorded is not None and abs(float(recorded) - wavelength) > 0.5:
            self._log.appendPlainText(
                f"{condition}: emission {float(recorded):g} nm -> {wavelength:g} nm, the "
                f"{text} preset's value replacing one already recorded. The recorded "
                f"number came from the acquisition; type it back if it was right.")
        values["emission_nm"] = wavelength
        self._write_calibration_if_saved(condition)
        self._refresh_calibration_table()
        if condition == (self._root.name if self._root is not None else None):
            self._apply_condition_calibration(condition)

    def _copy_row_to_all(self, values, key, table_refresh, label) -> None:
        """One row of one grid, from the FIRST condition to every other.

        The first rather than the selected one, because the button sits in the row and a
        row has no selection: making it follow the condition picker would mean the same
        press did different things depending on something on the other side of the
        window.
        """
        names = list(self._conditions)
        if not names:
            return
        source = names[0]
        value = values.get(source, {}).get(key)
        if value is None:
            self._log.appendPlainText(
                f"{source} has no {label} set, so there is nothing to copy along that "
                f"row")
            return
        for name in names:
            values.setdefault(name, {})[key] = value
        table_refresh()
        self._log.appendPlainText(
            f"every condition uses {source}'s {label}")

    def _refresh_calibration_table(self) -> None:
        """Redraw the grid from `self._calibrations`. Reads no movie and no file."""
        from .movieprep import CALIBRATION_FIELDS

        if len(self._conditions) < 2:
            self._calib_box.hide()
            return

        def render(key, value):
            if value is None:
                return ""
            if key == "frame_interval_s":
                return f"{float(value) * 1000:g}"          # shown in ms, stored in s
            if isinstance(value, str):
                return value
            return f"{float(value):g}"

        self._fill_grid(
            self._calib_table, CALIBRATION_FIELDS, self._calibrations,
            on_copy_row=lambda key: self._copy_row_to_all(
                self._calibrations, key, self._refresh_calibration_table,
                dict((k, lab) for k, lab, _u in CALIBRATION_FIELDS)[key]),
            render=render)
        self._calib_box.show()

    def _calibration_edited(self, item) -> None:
        """Take one edited cell back into the model, or refuse it and say so."""
        from .movieprep import CALIBRATION_FIELDS

        names = list(self._conditions)
        # COLUMN 0 IS THE COPY BUTTON, so a condition sits at column index + 1. Reading
        # the raw column here would edit the wrong condition, and the last one would be
        # unreachable.
        if item.row() >= len(CALIBRATION_FIELDS) or not 1 <= item.column() <= len(names):
            return
        key, label, _unit = CALIBRATION_FIELDS[item.row()]
        name = names[item.column() - 1]
        text = item.text().strip()
        if not text:
            self._calibrations.setdefault(name, {})[key] = None
            self._write_calibration_if_saved(name)
            return
        if key == "label":
            self._calibrations.setdefault(name, {})[key] = text
            self._write_calibration_if_saved(name)
            return
        try:
            value = float(text)
        except ValueError:
            # NOT SILENTLY DROPPED. A cell that keeps whatever was typed while the model
            # holds something else is the worst of both, so the cell is put back and the
            # log says why.
            self._log.appendPlainText(
                f"{name}: {label!r} must be a number, got {text!r}; not changed")
            self._refresh_calibration_table()
            return
        self._calibrations.setdefault(name, {})[key] = (
            value / 1000.0 if key == "frame_interval_s" else value)
        self._write_calibration_if_saved(name)

    def _write_calibration_if_saved(self, condition) -> None:
        """Put a settings change into that condition's crops.json, if it has one.

        THE ORDER OF THE WORK NO LONGER DECIDES WHAT IS ON DISK. The calibration is
        written INTO crops.json at save time, so choosing a fluorophore AFTER pressing
        Save left the file saying `null` while the grid said mNeonGreen. Moving the
        writers below the settings makes the natural order right; this makes the wrong
        order harmless, which is the stronger half.

        Only the calibration is rewritten -- never the ranges. A condition with no saved
        crop set is left alone: there is nothing to keep in step, and creating one here
        would write a crop nobody has pressed Save for.
        """
        from .movieprep import (CROPS_CALIBRATION_KEYS, MoviePrepError, find_crops,
                                update_crops_calibration)

        folder = self._conditions.get(condition) if self._conditions else self._root
        if folder is None:
            return
        try:
            found = find_crops(folder / "raw" if (folder / "raw").is_dir() else folder,
                               folder)
        except (MoviePrepError, OSError):
            return
        if not found or not Path(found).is_file():
            return
        values = self._calibrations.get(condition) or {}
        try:
            changed = update_crops_calibration(
                found, **{k: values.get(k) for k in CROPS_CALIBRATION_KEYS
                          if k in values})
        except MoviePrepError as exc:
            self._log.appendPlainText(f"{condition}: {exc}")
            return
        if changed:
            self._log.appendPlainText(
                f"{condition}: updated {Path(found).name}: "
                + ", ".join(f"{k} {old!r} -> {new!r}"
                            for k, (old, new) in sorted(changed.items()))
                + ". The crop ranges were not touched.")

    def _copy_calibration_to_all(self) -> None:
        """Copy the FIRST condition to every other.

        It used to copy whichever condition the picker showed, which made the same press
        do different things depending on a control on the other side of the window, and
        it did nothing at all when that condition happened to be the empty one. The
        button sits under the first column and copies the column it sits under, which is
        the only reading its position supports.
        """
        from .movieprep import MoviePrepError, replicate_calibration

        names = list(self._conditions)
        if not names:
            return
        source = names[0]
        try:
            self._calibrations = replicate_calibration(self._calibrations, source)
        except MoviePrepError as exc:
            self._log.appendPlainText(str(exc))
            return
        self._refresh_calibration_table()
        # AND ONTO DISK, for every condition that already has a crop set. Copying into
        # memory only would put the whole grid out of step with the files in one press.
        for name in self._conditions:
            self._write_calibration_if_saved(name)
        # RE-READ THE FOLDER, so the boxes for the condition on screen pick the new
        # values up. Without this the grid and the run disagree until the next switch.
        self._crop_root = None
        self._apply_condition_calibration(self._root.name if self._root else source)
        self._scan_movies()
        self._log.appendPlainText(
            f"copied {source}'s settings to every condition. Nothing is written until "
            f"you save the crop ranges. A field {source} does not have is left "
            f"unchanged.")
        self._refresh_progress()

    def _open_outlines(self) -> None:
        """Open the segmentation stage on the condition being worked on.

        KEYED TO THE CONDITION, and that is not a detail: the save queue adopts each
        condition in turn WITHOUT moving `self._root`, which is how the retired organise
        button filed only the first condition and how "Track this condition" once read
        the panel instead of a condition. Here `_root` IS the right source, because this
        is a press on the page in front of the user, and it is read once at the press.

        `segment_gui.launch` holds its own window in `_OPEN_WINDOWS`, so nothing is held
        here -- a parentless widget nobody references is destroyed the moment the call
        returns, which was the exit-139 segfault.
        """
        root = self._root
        if root is None:
            self._log.appendPlainText(
                "no condition is loaded, so there is nothing to segment yet.")
            return
        try:
            from .segmentation.segment_gui import launch as launch_segment

            launch_segment(root=str(root))
        except Exception as exc:                          # noqa: BLE001 - reported
            # NAMED, NOT SWALLOWED. The segmentation stack is excluded from the frozen
            # build, so this genuinely can fail in the shipped installer -- and the
            # editor half needs none of that stack, which is what the message must say
            # or a user reads "segmentation is unavailable" for "the model is".
            self._log.appendPlainText(
                f"could not open the segmentation stage: {type(exc).__name__}: {exc}. "
                "Correcting outlines by hand needs no extra runtime; automatic "
                "segmentation needs torch and cellpose, which the installer omits.")

    def _skip_outlines_prompt(self) -> None:
        """Answer the question, and remember the ANSWER rather than a flag.

        Keyed to the set of conditions that were missing outlines when it was pressed, so
        segmenting one of them later brings the prompt back for the rest -- a flag would
        silence it for the session and hide a state that had genuinely changed.
        """
        self._skipped_outlines = sorted(self._outline_notes and
                                        [n for n, note in self._outline_notes.items()
                                         if "no outlines" in note or "/" in note])
        self._outlines_prompt.hide()

    def _refresh_outline_state(self, groups=None) -> None:
        """Which conditions have outlines, and the offer where they do not.

        SEPARATE FROM THE BARS, because the bars are hidden below two conditions and this
        must not be: a single-condition project is exactly the case where somebody is
        preparing a folder for the first time. Reads directories, never movies --
        `outline_coverage` parses nothing and creates nothing. ``groups`` is the listing
        `_refresh_progress` already made, so the overview lists the folders once.
        """
        from .movieprep import discover_across_conditions
        from .prep_model import outline_states
        from .prep_view import outline_note, outlines_log, outlines_prompt

        folders = dict(self._conditions or {})
        if not folders and self._root is not None:
            folders = {self._root.name: self._root}
        if groups is None:
            try:
                groups = discover_across_conditions(self._parent_dir)
            except Exception:                             # noqa: BLE001 - advisory
                groups = {}
        states = outline_states(folders, groups)
        self._outline_notes = {name: outline_note(state) for name, state in states.items()}
        missing = sorted(name for name, state in states.items() if not state.complete)

        # SAID ONCE, AND IT NAMES WHAT IT BUYS. An offer that only says "you could
        # segment now" reads as a chore and gets skipped; the lines are the measured
        # reasons, and the figures in them are why two of the three are not
        # hypothetical.
        if missing and self._said_outlines != missing:
            self._said_outlines = missing
            self._log.appendPlainText(outlines_log(missing))

        # THE PROMPT ON THE TRACKING PAGE. Shown only where outlines are actually
        # missing, and never after "Not now" for that same set -- a question that
        # returns after it has been answered is a question nobody reads.
        prompt = getattr(self, "_outlines_prompt", None)
        if prompt is not None:
            show = bool(missing) and missing != self._skipped_outlines
            if show:
                self._outlines_prompt_label.setText(outlines_prompt(missing))
            prompt.setVisible(show)

    def _refresh_crowding(self) -> None:
        """How busy each condition's field is, and which movies are unlike their own.

        Reads the curves already held -- nothing is re-read and no movie is opened. A
        condition that has not been read yet simply has no line, which is honest: the
        number is a property of the detections, and there are none until a pass has run.

        THE RANGE IS THE CROP THE ANALYST SET, falling back to that movie's own floor.
        Crowding over a whole acquisition measures the dense phase the crop exists to
        remove, so it would name every movie in every project.
        """
        label = getattr(self, "_crowding_label", None)
        if label is None:
            return
        from .movieprep import MoviePrepError, crowding_summary

        # THE RADIUS THE RUN WILL ACTUALLY USE, not a constant. Crowding is a question
        # about a radius, so reading one number while the box holds another would answer
        # a question nobody asked.
        radius = self._linking_radius_for_crowding()

        lines, flagged_now = [], {}
        for name in sorted(self._conditions or {}):
            curves = dict(self._session_curves.get(name, {}))
            if name == getattr(self._root, "name", None):
                curves.update(self._crop_panel.curves())
            if not curves:
                continue
            ranges = dict(self._session_crops.get(name, {}))
            if name == getattr(self._root, "name", None):
                ranges.update(self._crop_panel.crops())
            try:
                got = crowding_summary(curves, radius_px=radius, ranges=ranges)
            except MoviePrepError as exc:
                # NAMED, NOT SWALLOWED -- but only once, and never as a failure of the
                # window. Curves cached before crowding was recorded are the ordinary
                # case here, and the fix is a re-read, which the message says.
                if self._said_crowding.get(name) != "stale":
                    self._said_crowding[name] = "stale"
                    self._log.appendPlainText(f"{name}: {exc}")
                continue
            if got.empty:
                continue
            bad = got[got["flagged"]]
            flagged_now[name] = sorted(bad["movie"])
            head = (f"{name}: {got.attrs['median_loc_per_frame']:.1f} detections/frame "
                    f"over the tracked range, {100 * got.attrs['median_crowded_share']:.0f} % "
                    f"with another within {radius:g} px")
            if len(bad):
                head += f"  ·  {len(bad)} movie(s) unlike the rest: " + ", ".join(
                    f"{r.movie} ({r.reason})" for r in bad.itertuples())
            lines.append(head)

        if not lines:
            label.hide()
            return
        label.setText("\n".join(lines))
        label.show()

        # SAID IN THE LOG ONCE PER SET, for the reason the outlines offer is: a line that
        # returns on every redraw is a line nobody reads.
        for name, movies in flagged_now.items():
            if movies and self._said_crowding.get(name) != movies:
                self._said_crowding[name] = movies
                self._log.appendPlainText(
                    f"{name}: {', '.join(movies)} differ from the other movies in "
                    f"detection density. That is not a problem by itself and nothing is blocked; "
                    f"open them before tracking if you have not.")

    def _linking_radius_for_crowding(self) -> float:
        """The linking radius on screen, snapped to the ladder crowding is recorded at.

        SNAPPED RATHER THAN INTERPOLATED, and it says which it used. `crowding_column`
        refuses a radius off the ladder precisely so a number between two of them cannot
        be invented; picking the nearest recorded one is the honest reading of a box that
        holds 4.9.
        """
        from .movieprep import CROWDING_RADII_PX

        wanted = None
        box = getattr(self, "_link", None)
        if box is not None:
            try:
                wanted = float(box.value())
            except Exception:                              # noqa: BLE001 - advisory
                wanted = None
        if wanted is None:
            wanted = 5.0
        return min(CROWDING_RADII_PX, key=lambda r: abs(r - wanted))

    def _refresh_progress(self) -> None:
        """Redraw the per-condition bars. Reads directories, never movies.

        The active condition is taken from the PANEL rather than from disk: curves and
        crops only reach disk when they are saved, so the folder being worked on would
        otherwise sit at zero for the whole time somebody watches it fill. The counts and
        every word come from `prep_model.resolve_counts` and `prep_view`, which the
        workspace-style window shares.
        """
        from .movieprep import (calibration_mismatch, condition_progress,
                                discover_across_conditions)
        from .prep_model import resolve_counts
        from .prep_view import (calibration_note, condition_note, mismatch_log,
                                threshold_setting)

        # ONE LISTING, SHARED -- by the outline state and the bars alike. A second
        # discovery is a second opinion about what a condition contains.
        try:
            groups = discover_across_conditions(self._parent_dir)
        except Exception:                                 # noqa: BLE001 - advisory
            groups = None
        # COMPUTED FIRST, because the bars below are hidden for a single condition and
        # the offer must not be hidden with them -- the same defect one level down from
        # the `> 1` gate in `_detect_conditions`.
        self._refresh_outline_state(groups)
        if len(self._conditions) < 2 or groups is None:
            self._progress_box.hide()
            return
        try:
            rows = condition_progress(self._parent_dir, groups=groups)
            differs = calibration_mismatch(rows)
        except Exception:                                 # noqa: BLE001 - advisory
            self._progress_box.hide()
            return

        active = self._root.name if self._root is not None else None
        for row in rows:
            counts = resolve_counts(
                row,
                live=self._crop_panel.progress() if row.name == active else None,
                remembered=self._session_read.get(row.name),
                session_crops=self._session_crops.get(row.name, {}))
            widgets = self._progress_bars.get(row.name)
            if widgets is None:
                line = QHBoxLayout()
                name = QLabel(row.name)
                name.setMinimumWidth(120)
                bar = QProgressBar()
                bar.setTextVisible(False)
                bar.setMaximumHeight(12)
                note = QLabel("")
                note.setMinimumWidth(330)
                line.addWidget(name)
                line.addWidget(bar, 1)
                line.addWidget(note)
                holder = _ConditionRow(row.name)
                holder.setLayout(line)
                holder.clicked.connect(self._pick_condition_row)
                self._progress_rows.addWidget(holder)
                widgets = self._progress_bars[row.name] = (name, bar, note)
            name_label, bar, note = widgets
            bar.setRange(0, max(counts.n_movies, 1))
            bar.setValue(counts.done)
            setting = threshold_setting(
                self._thresholds.get(row.name),
                per_movie=self._movie_thresholds.get(row.name),
                typed=row.name in self._typed_thresholds,
                n_crops=counts.n_crops, n_movies=counts.n_movies)
            note.setText(condition_note(
                counts, setting=setting, calibration=calibration_note(row, differs),
                outlines=self._outline_notes.get(row.name, ""),
                active=row.name == active))
            name_label.setStyleSheet(
                "font-weight: bold;" if row.name == active else "color: #888;")
        # SAID ONCE, not per row, because it is a fact about the SET. Preparation and
        # tracking are per movie and are unaffected; what it decides is whether these
        # conditions' D values may be compared afterwards, and that is settled here
        # rather than discovered in the analysis months later.
        if differs and getattr(self, "_said_mismatch", None) != sorted(differs):
            self._said_mismatch = sorted(differs)
            self._log.appendPlainText(mismatch_log(differs))
        self._progress_box.show()

    def _advance_to_next_condition(self) -> None:
        """Move to the next condition that has not been read yet, if there is one.

        This is what makes the whole loaded dataset progress rather than whichever folder
        happens to be selected. It stops as soon as every condition has been through,
        so the dropdown is yours again and switching only chooses what to look at.

        A condition the user selected by hand is read like any other and removed from the
        queue, so a manual switch mid-walk does not cause it to be read twice.
        """
        if not self._conditions:
            return
        here = self._root.name if self._root is not None else None
        self._to_read = [x for x in self._to_read
                         if x != here and x not in self._session_read]
        if not self._to_read:
            # ONCE, not on every pass. `reading_changed(False)` fires whenever a folder
            # settles, including from the disk cache, so an unguarded line here would
            # repeat for the rest of the session.
            if not getattr(self, "_said_all_read", False):
                self._said_all_read = True
                self._log.appendPlainText(
                    f"all {len(self._conditions)} conditions read; switching only "
                    f"chooses which one you are looking at")
            return
        self._said_all_read = False
        nxt = self._to_read[0]
        # READ IT WHERE IT SITS. This used to point the panel at `nxt`, which made the
        # walk and the view one action: click a condition that was already read and the
        # walk took the view straight back off you, which is exactly what a click is
        # for. Verified on D:/data/project, where the one condition with a saved
        # crops.json loaded instantly and bounced the view to the next unread folder.
        #
        # The results reach `_curve_arrived` under `nxt`'s own token, so the session
        # fills whatever is on screen.
        from .rawmovie import discover_raw_movies

        folder = self._conditions[nxt]
        try:
            acq = folder / "raw" if (folder / "raw").is_dir() else folder
            movies = discover_raw_movies(acq)[0]
        except Exception as exc:                          # noqa: BLE001 - named below
            self._log.appendPlainText(f"{nxt} could not be listed: {exc}")
            self._session_read[nxt] = (0, 0, 0)
            self._refresh_progress()
            return
        self._to_read = [x for x in self._to_read if x != nxt]
        # THE SAVED RANGES COME WITH IT. A cached read replays the CURVES, and the crop
        # ranges live in crops.json rather than in those CSVs -- so a condition read this
        # way arrived with "0 ranges, threshold: needs ranges first" while its own file
        # held ten. The proposer only fills what is missing, so seeding here cannot
        # overwrite a range somebody chose.
        try:
            from .movieprep import find_crops, load_crops

            saved = find_crops(acq, folder)
            payload = load_crops(saved) if saved else None
        except Exception:                                 # noqa: BLE001 - advisory
            payload = None
        if payload:
            have = self._session_crops.setdefault(nxt, {})
            for stem, span in payload.get("crops", {}).items():
                have.setdefault(stem, tuple(span))
        self._log.appendPlainText(
            f"reading {nxt} in the background ({len(movies)} movie(s)); the view stays "
            f"where you put it")
        # The results folder is the CONDITION folder, not `raw/` -- the same split
        # `set_folder` makes, because organising moves reading and writing apart.
        self._crop_panel.read_in_background(movies, nxt, cache_dir=folder / "bleaching")

    def _end_frame_everywhere(self, frame: int) -> None:
        """'Cut all tails' for the conditions NOT on screen: the panel has done its own."""
        self._common_cut_everywhere(end=frame)

    def _start_frame_everywhere(self, frame: int) -> None:
        """'Cut all starts' for the conditions NOT on screen: the panel has done its own."""
        self._common_cut_everywhere(start=frame)

    def _common_cut_everywhere(self, *, start=None, end=None) -> None:
        """One common start or end frame for every condition of the session.

        Asked for as "set frame X where all tails are cut from all movies", and then the
        same for the start, so every condition, including one never opened here whose
        ranges are only on disk: those are loaded first. A movie read but never given a
        range gets one, as on the panel. Nothing is written until Save, which is the
        single record of a crop.
        """
        from .prep_model import common_cut, saved_crops
        from .prep_view import common_cut_log

        here = self._root.name if self._root is not None else None
        reports = {}
        for name, folder in (self._conditions or {}).items():
            if name == here:
                continue
            crops = dict(self._session_crops.get(name) or {}) or saved_crops(folder)
            n_frames = {stem: int(got.n_frames)
                        for stem, got in self._session_curves.get(name, {}).items()
                        if getattr(got, "n_frames", None)}
            if not crops and not n_frames:
                continue
            report = common_cut(crops, n_frames, start=start, end=end)
            self._session_crops[name] = dict(report.crops)
            reports[name] = report
        line = common_cut_log(reports)
        if line:
            self._log.appendPlainText(line)
        self._refresh_progress()

    def _curve_arrived(self, token, stem, curves, error) -> None:
        """File one movie's curves under the condition it came from, always.

        The panel shows one folder; the SESSION holds them all. Recording here is what
        makes a background read outlive the switch that used to kill it, and it also
        proposes the crop for a condition nobody is watching, which is the other half of
        what the tick box promises.
        """
        if not token or token not in self._conditions:
            return
        if error:
            self._session_errors.setdefault(token, {})[stem] = error
        else:
            self._session_curves.setdefault(token, {})[stem] = curves
            if self._crop_panel.auto_propose:
                crops = self._session_crops.setdefault(token, {})
                if stem not in crops:
                    try:
                        from .movieprep import propose_crop

                        got = propose_crop(curves, start_rule=self._crop_panel.rule,
                                           **self._crop_panel.cut_rule())
                    except Exception:                     # noqa: BLE001 - named below
                        self._session_errors.setdefault(token, {})[stem] = "no proposal"
                    else:
                        crops[stem] = (got["start"], got["end"])
        # THE TOTAL COMES FROM THE FOLDER, once. Deriving it from what has arrived so
        # far makes the bar full at every step, which is the one thing a progress bar
        # must never do.
        total = self._count_movies(token)
        self._session_read[token] = (total,
                                     len(self._session_curves.get(token, {})),
                                     len(self._session_errors.get(token, {})))
        self._refresh_progress()

    def _count_movies(self, condition) -> int:
        """How many acquisitions a condition holds, listed once and remembered.

        NOT `_movie_count`, which is already a QLabel on this window. Rebinding it turned
        every later read into `'QLabel' object is not callable`: the same collision as
        the second organise button, and the reason this window's attributes are checked.
        """
        cache = self.__dict__.setdefault("_movie_counts", {})
        if condition not in cache:
            from .rawmovie import discover_raw_movies

            folder = self._conditions.get(condition)
            try:
                acq = folder / "raw" if (folder / "raw").is_dir() else folder
                cache[condition] = len(discover_raw_movies(acq)[0])
            except Exception:                             # noqa: BLE001 - advisory
                cache[condition] = 0
        return cache[condition]

    def _pass_finished(self, token) -> None:
        """A background pass ended, displayed or not: finish that condition off.

        `_reading_changed` does this for the condition ON SCREEN -- it stashes it and
        measures its threshold. A pass the user switched away from never reaches that
        handler, because the panel is showing somebody else by then, so a condition read
        in the background arrived complete and permanently "not measured". Measured here
        instead, against the token that actually finished.
        """
        if not token or token not in self._conditions:
            return
        if token not in self._session_read:
            self._session_read[token] = (0, 0, 0)
        here = self._root.name if self._root is not None else None
        if token != here and token not in self._thresholds:
            movies, _crops = self._measurable_movies(token)
            if movies:
                self._compute_threshold_clicked(owner=token)
                self._refresh_tracking_table()
        self._refresh_progress()
        self._refresh_crowding()

    def _stash_condition(self) -> None:
        """Keep the condition on screen before the panel is pointed elsewhere."""
        if self._root is None or not self._conditions:
            return
        name = self._root.name
        if name not in self._conditions:
            return
        crops = dict(self._crop_panel.crops())
        if crops:
            self._session_crops[name] = crops
        curves = self._crop_panel.curves()
        if curves:
            self._session_curves[name] = curves
        errors = self._crop_panel.errors()
        if errors:
            self._session_errors[name] = errors
        live = self._crop_panel.progress()
        if live[0]:
            self._session_read[name] = (live[0], live[1],
                                        live[3] if len(live) > 3 else 0)

    def _pick_condition_row(self, name: str) -> None:
        """A click on an overview row selects that condition.

        Routed THROUGH the hidden combo rather than calling `_pick_condition` directly,
        so the box, the rows and the read walk share one notion of what is selected.
        Calling the handler straight would leave the model behind, and
        `_advance_to_next_condition` reads the model.
        """
        if name in self._conditions:
            self._conditions_box.setCurrentText(name)

    def _pick_condition(self, name: str) -> None:
        if not name or name not in self._conditions:
            return
        self._stash_condition()
        self._root = self._conditions[name]
        self._apply_condition_calibration(name)
        self._apply_root_state()
        self._crop_root = None
        self._apply_folder_state()
        if self._source == SOURCE_RAW:
            # ALREADY READ MEANS ALREADY READ. The picker chooses what to look at; only a
            # condition this session has never opened costs a read.
            if name in self._session_curves:
                folder, _why = self._acquisition_root()
                self._crop_root = folder
                self._adopting = True
                try:
                    self._crop_panel.adopt(
                        folder, output_root=self._root,
                        curves=self._session_curves.get(name),
                        crops=self._session_crops.get(name),
                        errors=self._session_errors.get(name))
                finally:
                    self._adopting = False
            else:
                self._load_crop_panel()
                self._crop_panel.restore_crops(self._session_crops.get(name))
        self._scan_movies()
        self._refresh_progress()


    def _show_source_controls(self, shown):
        """Reveal the source and layout controls. Hidden by default (resolved, not
        asked) and the state line names the source either way."""
        self._source_widget.setVisible(bool(shown))
        self._pattern_widget.setVisible(
            bool(shown) and self._source == SOURCE_PREPARED)

    def _source_changed(self, text):
        self._source = text
        self._pattern_edit.setEnabled(text == SOURCE_PREPARED)
        self._show_source_controls(self._source_toggle.isChecked())
        self._crop_box.setVisible(text == SOURCE_RAW)
        self._save_box.setVisible(text == SOURCE_RAW)
        if text == SOURCE_RAW:
            self._load_crop_panel()
        self._scan_movies()

    def _fill_pixel_size_from_folder(self):
        """Fill the pixel size from what this folder already records, and say from where.

        The number was never typed into u-track. It was typed once into the Fiji
        preparation macro (`pixel_width=0.106000` with `unit=[um]`) which bakes the
        calibration into the multi-page TIFF; u-track read it there and `movieData.mat`
        records 106.0 nm. On a prepared project it is therefore on disk twice, and
        asking for it again asks the user to re-derive something long settled.

        Never over a value already in the box. If someone has typed a pixel size,
        replacing it when the folder changes would silently discard a deliberate choice,
        the same rule the threshold and the optics follow.
        """
        if self._pixel_nm.value():
            return
        from .optics import pixel_size_from_project

        value, why = pixel_size_from_project(self._root)
        if value is None:
            self._pixel_nm.setToolTip(
                f"{why}\n\ncamera pixel / (objective x tube lens). It cannot be derived from "
                f"NA, which gives the PSF width.")
            return
        self._pixel_nm.setValue(value)
        self._pixel_nm.setToolTip(f"read from {why}")
        self._log.appendPlainText(f"pixel size {value:.1f} nm read from {why}")

    def _fill_optics_from_folder(self):
        """Fill NA and emission wavelength from what u-track already recorded.

        The best fix for "NA is not set" is not having to type it. `movieData.mat` holds
        both (this project's own reads give NA 1.49 and emission 517 nm) and it is the
        same provenance the pixel size already comes from, so trusting one and not the
        other would be arbitrary. A RAW-only folder has no such file and the boxes stay
        empty, which is why they are also easy to type into now.

        Never over a value already on screen, the rule the pixel size and the threshold
        both follow: replacing a typed number when the folder changes discards a
        deliberate choice.
        """
        if self._numerical_aperture.value() and self._emission_nm.value():
            return
        try:
            from .utrack_meta import read_project_metadata

            recorded = read_project_metadata(self._root)
        except Exception:                              # noqa: BLE001 - optional source
            # NO movieData.mat AT ALL is the NORMAL case going forward: that file is
            # u-track's, so anything tracked with TrackMate will never have one.
            self._fill_optics_from_acquisition()
            return
        aperture = next((m.num_aperture for m in recorded if m.num_aperture), None)
        emission = next((m.emission_wavelength_nm for m in recorded
                         if m.emission_wavelength_nm), None)
        _ = aperture, emission                          # filled below
        filled = []
        if aperture and not self._numerical_aperture.value():
            self._numerical_aperture.setValue(float(aperture))
            self._numerical_aperture.setToolTip("read from movieData.mat")
            filled.append(f"NA {float(aperture):.2f}")
        if emission and not self._emission_nm.value():
            self._emission_nm.setValue(float(emission))
            self._emission_nm.setToolTip("read from movieData.mat")
            filled.append(f"emission {float(emission):.0f} nm")
        if filled:
            self._log.appendPlainText(
                f"{' and '.join(filled)} read from movieData.mat, the file the "
                f"pixel size comes from")
        if not self._numerical_aperture.value():
            self._fill_optics_from_acquisition()
            # The spot radius follows from these, so derive it now rather than leaving
            # a value nobody chose sitting in the box.
            self._radius_from_optics_clicked(quiet=True)

    def _fill_optics_from_acquisition(self):
        """NA from the acquisition, and the camera pixel behind the pixel size.

        The only source for a trackmate project. `movieData.mat` is u-track's file,
        so nothing tracked here will ever have one, and NA sets the PSF, which sets the
        localisation floor, which decides whether a slow-state D may be quoted at all.
        Leaving it blank meant typing it from memory on every new dataset.

        NA is filled; the pixel size is not. `objectiveNumericalAperture` is recorded by
        the instrument, the same standing as the frame interval. The pixel size is
        DERIVED by NIS from a magnification that drops the tube lens (160 nm where the
        truth is 106) so what is offered instead is the CAMERA pixel, recovered from
        that very error (claim x magnification = 16 um), leaving the tube lens as the one
        term to supply. It goes in the log, not into the box: the pixel size is the
        number this project refuses to guess.

        The emission is not touched. The file records the FILTER (540 nm, YFPHQ) and the
        PSF wants the DYE (mNeonGreen, 517 nm). Filling it would silently replace a dye
        maximum with a filter centre, which is a different quantity that looks right.
        """
        folder, _why = self._acquisition_root()
        if folder is None:
            return
        try:
            from .rawmovie import discover_raw_movies, read_raw_movie

            found, _companions = discover_raw_movies(folder)
            if not found:
                return
            meta = read_raw_movie(found[0]).meta
        except Exception:                              # noqa: BLE001 - optional source
            return
        if meta.numerical_aperture and not self._numerical_aperture.value():
            self._numerical_aperture.setValue(float(meta.numerical_aperture))
            self._numerical_aperture.setToolTip(
                f"read from {found[0].name} ({meta.objective_name or 'objective'})")
            self._log.appendPlainText(
                f"NA {meta.numerical_aperture:.2f} read from {found[0].name} "
                f"({meta.objective_name}): recorded by the instrument, like the frame "
                f"interval")
            self._radius_from_optics_clicked(quiet=True)
        if meta.camera_pixel_um and not self._pixel_nm.value():
            self._camera_pixel_um = float(meta.camera_pixel_um)
            self._objective_magnification = float(meta.objective_magnification or 0)
            implied = (meta.camera_pixel_um * 1000.0
                       / (meta.objective_magnification * TUBE_LENS_DEFAULT)
                       if meta.objective_magnification else None)
            note = (f"camera pixel {meta.camera_pixel_um:g} µm at "
                    f"{meta.objective_magnification:g}× read from {found[0].name}. The "
                    f"file's own {meta.reported_pixel_size_nm_UNTRUSTED:g} nm leaves out the "
                    f"tube lens and is not used")
            if implied:
                # THE DERIVED VALUE GOES IN THE BOX, and it did not before.
                #
                # Asked, reasonably: "utrack automatically detects 106 nm pixel size
                # from the movie data. why cant we?" We already did every hard part,
                # the camera pixel pitch and the objective magnification are READ from
                # the acquisition, and dividing by the tube lens gives 106.7 against the
                # 106.0 u-track records, 0.7 % apart. It was then printed in the log and
                # thrown away, leaving an empty box: this file's most-recorded defect
                # shape, a capability wired at one end.
                #
                # IT IS A DERIVATION, NOT A READING, and the difference is the tube lens
                #, the one factor no file states. So it fills only an EMPTY box, the
                # provenance says "derived" rather than "read", and `?` changes the
                # factor. What it replaces is not a safer state: an empty box is filled
                # by hand from memory, which is how 160 nm gets typed.
                # THE PROJECT'S OWN VALUE WHERE THE DERIVATION CONFIRMS IT. Filling
                # 106.7 on a fresh folder would put every new D 1.26 % away from every
                # old one, for 0.63 % of optics that is not measured either way.
                agrees = (abs(implied - PROJECT_PIXEL_SIZE_NM)
                          <= PIXEL_SIZE_AGREEMENT * PROJECT_PIXEL_SIZE_NM)
                self._pixel_nm.setValue(PROJECT_PIXEL_SIZE_NM if agrees
                                        else round(implied, 1))
                self._pixel_nm.setToolTip(
                    (f"{PROJECT_PIXEL_SIZE_NM:g} nm: the optics below agree with it to "
                     f"{abs(implied - PROJECT_PIXEL_SIZE_NM) / PROJECT_PIXEL_SIZE_NM * 100:.2f} %.\n\n"
                     if agrees else "") +
                    f"Derived: {meta.camera_pixel_um:g} µm camera pixel / "
                    f"{meta.objective_magnification:g}× / {TUBE_LENS_DEFAULT:g}× tube "
                    f"lens.\n\nThe file's own field says "
                    f"{meta.reported_pixel_size_nm_UNTRUSTED:g} nm, which leaves out the "
                    f"tube lens, so it is not used.\n\nPress ? to change the tube lens, "
                    f"and check the result against a cell width.")
                if agrees:
                    note += (f"; with a {TUBE_LENS_DEFAULT:g}× tube lens that gives "
                             f"{implied:.1f} nm, which agrees with "
                             f"{PROJECT_PIXEL_SIZE_NM:g} nm to "
                             f"{abs(implied - PROJECT_PIXEL_SIZE_NM) / PROJECT_PIXEL_SIZE_NM * 100:.2f} %"
                             f", so the box holds {PROJECT_PIXEL_SIZE_NM:g}")
                else:
                    note += (f"; with a {TUBE_LENS_DEFAULT:g}× tube lens that gives "
                             f"{implied:.1f} nm, which differs from "
                             f"{PROJECT_PIXEL_SIZE_NM:g} nm, so the box holds the "
                             f"derivation. Press ? to change the tube lens")
            self._log.appendPlainText(note)

    def _fill_interval_from_acquisition(self):
        """Fill the frame interval from the acquisition's OWN per-frame timing.

        **The pixel size is not taken from the file and the interval is, and the
        difference is provenance rather than format.** Nis derives the pixel size from a
        magnification that ignores the ~1.5x tube lens, giving 160 nm where the truth is
        106, and it is wrong on 24 of 29 groups and right on 5, which is worse than
        being wrong everywhere. The frame interval is MEASURED by the instrument, one
        timestamp per frame, and it is the value this project corrected by hand twice:
        20.4717 ms where the nominal says 20, 51.7477 where it says 50.

        Never over a value already in the box, the same rule the pixel size follows.
        """
        if self._interval_ms.value():
            return
        folder, _why = self._acquisition_root()
        if folder is None:
            return
        from .rawmovie import discover_raw_movies, read_raw_movie

        try:
            found, _companions = discover_raw_movies(folder)
            if not found:
                return
            meta = read_raw_movie(found[0]).meta
        except Exception:                              # noqa: BLE001 - optional source
            return
        if meta.frame_interval_ms is None:
            return
        self._interval_ms.setValue(float(meta.frame_interval_ms))
        self._interval_ms.setToolTip(
            f"measured from {found[0].name} ({meta.frame_interval_source})")
        self._log.appendPlainText(
            f"frame interval {meta.frame_interval_ms:.4f} ms measured from "
            f"{found[0].name} ({meta.frame_interval_source}). The exposure setting "
            f"({meta.exposure_ms:g} ms) is the shutter time and does not give the interval.")

    def _pixel_size_helper(self):
        """Compute the pixel size from the camera and the magnification.

        Why a helper and not a derivation. This is the one calibration number that
        cannot be read off the optics: NA and wavelength give the PSF WIDTH, not the
        sampling. u-track looks as though it derives it and does not: `pixelSize_` is
        a field the user supplies there too, while `psfSigma_` is the derived one.

        The tube lens is the term that gets dropped, and dropping it is exactly the
        error this project's .nd2 files make: 16 um / 100x = 160 nm, against
        16 um / 150x = 106.7 nm with the ~1.5x tube lens. The live result is shown
        BEFORE anything is applied, so the mistake is visible rather than inherited.
        """
        from qtpy.QtWidgets import QDialog, QDialogButtonBox

        from .optics import OpticsError, pixel_size_nm

        dialog = QDialog(self)
        dialog.setWindowTitle("Pixel size from the camera")
        form = QFormLayout(dialog)

        camera = QDoubleSpinBox()
        camera.setRange(0.0, 100.0)
        camera.setDecimals(2)
        camera.setSuffix(" µm")
        camera.setSpecialValueText("not set")
        camera.setToolTip("The sensor's physical pixel size.")
        # PRE-FILLED FROM THE ACQUISITION where it was readable, so the user supplies the
        # ONE term the file drops (the tube lens) rather than all three.
        if getattr(self, "_camera_pixel_um", None):
            camera.setValue(self._camera_pixel_um)
        objective = QDoubleSpinBox()
        objective.setRange(0.0, 250.0)
        objective.setDecimals(1)
        objective.setSuffix(" ×")
        objective.setSpecialValueText("not set")
        tube = QDoubleSpinBox()
        tube.setRange(0.0, 10.0)
        tube.setDecimals(2)
        tube.setValue(1.0)
        tube.setSuffix(" ×")
        if getattr(self, "_objective_magnification", None):
            objective.setValue(self._objective_magnification)
        tube.setToolTip(
            "The magnifier between objective and camera. Use 1.0 if there is none.")
        result = QLabel("–")
        result.setStyleSheet("font-weight: 600;")

        def recompute():
            try:
                value = pixel_size_nm(camera_pixel_um=camera.value(),
                                      magnification=objective.value(),
                                      tube_lens=tube.value())
            except OpticsError:
                result.setText("–")
                return
            result.setText(f"{value:.1f} nm / pixel")

        for widget in (camera, objective, tube):
            widget.valueChanged.connect(recompute)
        form.addRow("Camera pixel", camera)
        form.addRow("Objective", objective)
        form.addRow("Tube lens", tube)
        form.addRow("Pixel size", result)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)

        accepted = dialog.exec_() if hasattr(dialog, "exec_") else dialog.exec()
        if not accepted:
            return
        try:
            value = pixel_size_nm(camera_pixel_um=camera.value(),
                                  magnification=objective.value(),
                                  tube_lens=tube.value())
        except OpticsError as exc:
            self._crop_status.setText(str(exc))
            return
        self._pixel_nm.setValue(value)
        self._log.appendPlainText(
            f"pixel size {value:.1f} nm from {camera.value():g} µm / "
            f"({objective.value():g}x x {tube.value():g}x)")

    def _reading_changed(self, reading: bool):
        """Grey the writers while the background pass holds the movies."""
        for button in (self._save_crops, self._export_stacks):
            button.setEnabled(not reading)
        if not reading and not self._adopting:
            # THE CROPS ARRIVE DURING THE READ, so the threshold becomes measurable only
            # when it ends. Without this the settings for a condition stayed empty until
            # somebody visited it and pressed something, which is the overview's whole
            # complaint: read through, and still "not measured".
            self._stash_condition()
            self._autocompute_threshold()
            self._advance_to_next_condition()
        self._refresh_progress()
        if reading:
            self._crop_status.setText(
                "reading the movies: saving and exporting wait for that to finish, "
                "so nothing is written for only part of the set")
        elif "reading the movies" in self._crop_status.text():
            self._crop_status.setText("")

    # -- what is in the folder ------------------------------------------------
    def _apply_folder_state(self):
        """Read the folder and start on the source that matches it.

        The routing the experimenter asked for, and it is one rule rather than two:
        raw acquisitions and nothing prepared means the crop question has not been
        answered, so open on RAW and ask it. Prepared stacks present means it HAS been
        answered, so open on PREPARED and offer tracking straight away, with the raw
        source one dropdown away for anyone who wants to change the crop.

        RE-CROPPING is offered, not forced. Asking the crop question again by default on
        a prepared project invites two answers to one question, and `crops.json` is the
        single record precisely so the two halves cannot disagree.
        """
        from .organise import OrganiseError, inspect_folder

        if self._root is None:
            self._folder_state.setText("no folder chosen")
            return None
        try:
            state = inspect_folder(self._root)
        except OrganiseError as exc:
            self._folder_state.setText(f"{type(exc).__name__}: {exc}")
            return None
        self._fill_pixel_size_from_folder()
        self._fill_optics_from_folder()
        self._fill_interval_from_acquisition()
        wanted = SOURCE_RAW if state.recommends_raw_source else SOURCE_PREPARED
        if self._source_box.currentText() != wanted:
            # Setting the combo fires `_source_changed`, which rescans - so this is the
            # whole of the routing rather than the first half of it.
            self._source_box.setCurrentText(wanted)
        # THE RESOLVED SOURCE IS NAMED HERE, because its dropdown is hidden by default.
        # A control put away must leave its ANSWER on screen, or hiding it turns a
        # question nobody had to answer into a fact nobody can see.
        # THE CAPTION ALREADY SAYS WHICH SOURCE AND WHY, so this no longer appends a
        # second line, with the movies line below also naming it, the window said the
        # same thing three times over.
        # THE CAPTION PLUS WHY, when there is a why. `caption()` describes the FOLDER
        # that was opened; when that folder is empty and subfolders WERE examined, what
        # happened one level down is the whole answer, and it was invisible.
        note = getattr(self, "_detect_note", "")
        caption = state.caption()
        if note:
            caption = caption + chr(10) + note
        self._folder_state.setText(caption)
        return state

    # -- project layout -------------------------------------------------------
    def _acquisition_root(self):
        """Where the acquisitions ARE, which is not always the folder that was picked.

        Organising moves them into `raw/`, and nothing in this window or in `movieprep`
        appends that. Without this, the press that files the movies is followed by a
        scan reporting zero, which reads as "my data disappeared" rather than "they are
        one level down". `organise.acquisition_folder` returns the REASON as well, and
        the reason is shown: a folder that silently redirects itself is
        indistinguishable from one that found nothing, and the difference is "ready"
        against "wrong folder".
        """
        from .organise import OrganiseError, acquisition_folder

        if self._root is None:
            return None, "no folder chosen"
        try:
            return acquisition_folder(self._root)
        except OrganiseError as exc:
            return None, str(exc)

    def _auto_crop_toggled(self, on: bool) -> None:
        """Arm or disarm the crop panel's proposer, and say which rule is in force.

        Said out loud because a range that appeared without a press, and without a rule
        named beside it, is exactly the silent default this pipeline refuses.
        """
        self._crop_panel.set_auto_propose(on)
        if on:
            rule = getattr(self._crop_panel, "rule", None) or "strictest"
            self._log.appendPlainText(
                f"crop ranges will be proposed as each movie is read, rule '{rule}'. "
                f"Every value stays editable; untick to stop proposing.")
            # SAID EVERY TIME IT ARMS, because it is now on by default and nobody
            # pressed anything to see it. A floor read as a recommendation is the one
            # way this convenience costs data.
            self._log.appendPlainText(
                "  a proposal is the earliest defensible start, not a recommendation. "
                "Read the curve before accepting it.")
        else:
            self._log.appendPlainText(
                "no crop will be proposed; set each range yourself")

    def _load_crop_panel(self):
        """Point the crop panel at the current folder, once per folder.

        Reloading on every rescan would restart the background curve pass, which is the
        wait this panel exists to remove.
        """
        folder, _why = self._acquisition_root()
        if folder is None or getattr(self, "_crop_root", None) == folder:
            return
        self._crop_root = folder
        try:
            # READ from where the movies are, WRITE to the condition folder. After
            # organising those differ, and writing beside the movies buried crops.json
            # and the bleaching curves inside raw/, where nobody looks, and oddly
            # beside nothing, while tif/, sum/ and phase/ sit one level up.
            self._crop_panel.set_folder(folder, output_root=self._root)
            # THE BOX OUTLIVES THE FOLDER. `set_folder` rebuilds the panel's state, and
            # a standing choice that quietly lapsed on the next folder would be worse
            # than no box at all.
            self._crop_panel.set_auto_propose(self._auto_crop.isChecked())
        except Exception as exc:                       # noqa: BLE001 - named on screen
            self._crop_status.setText(f"{type(exc).__name__}: {exc}")
            return
        self._crops_changed()

    def _scan_raw(self):
        """Raw acquisitions plus the crop ranges the preparation stage recorded.

        The range is required, and not because of a rule: a raw acquisition still holds
        the dense early frames the crop exists to remove, so tracking one without a range
        would track a regime the experiment does not occupy. A movie with no recorded
        crop is NAMED rather than tracked whole.
        """
        from .movieprep import CROPS_FILENAME, MoviePrepError, load_crops
        from .rawmovie import discover_raw_movies

        # Same scan the preparation stage uses, so the two stages cannot disagree about
        # what a movie is: a `.tif` beside a `.nd2` of the same name is the phase
        # snapshot, not a twentieth acquisition.
        acq_root, why = self._acquisition_root()
        found, _companions = (discover_raw_movies(acq_root)
                              if acq_root and acq_root.is_dir() else ([], []))
        # BOTH PLACES, because the results moved to the condition folder while older
        # sets sit beside the movies. Looking in only one is how a folder with a
        # perfectly good crops.json was reported as having none, the panel had already
        # loaded it and this scan had not.
        try:
            if acq_root is None:
                raise MoviePrepError(why)
            # NOT `found`: that name already holds the discovered MOVIES four lines
            # up, and reusing it made the scan iterate a Path.
            crops_path = self._crop_panel.crops_file()
            if crops_path is None:
                raise MoviePrepError(
                    f"no {CROPS_FILENAME} in {self._root} or {acq_root}")
            payload = load_crops(crops_path)
        except MoviePrepError as exc:
            self._movies, self._crops = [], {}
            self._movies_label.setText(
                f"{why}\n"
                f"{len(found)} raw acquisition(s), but no usable crop set: {exc}\n\n"
                "Set a range per movie in the crop panel below and press "
                "'Save crop ranges'.")
            return
        self._crops = payload["crops"]
        # THE CALIBRATION TRAVELS WITH THE RANGES, which is the whole reason it is
        # written into the same file, so a folder that already has one needs nothing
        # typed.
        #
        # PER CONDITION, and "empty boxes only" was right for ONE folder and wrong the
        # moment this window learned to walk several. A condition records its own pixel
        # size, frame interval, emission and NA, and the boxes kept whichever condition
        # filled them first: measured, opening a parent and switching kept 106 nm and
        # 20.47 ms on a condition whose own record says 65 nm and 51.75 ms. That is D
        # wrong by 2.66x from the pixel size and 2.53x from tau, in the settings a run
        # is about to use, with nothing on screen to say so.
        #
        # So a RECORDED value now wins for its own condition, and the change is named.
        # A typed value still stands where the condition records nothing, which is the
        # case the old rule was protecting.
        here = self._root.name if self._root is not None else None
        switched = self._calibration_from != here
        moved = []

        def adopt(box, value, label, scale=1.0):
            if value is None:
                return
            new = float(value) * scale
            old = box.value()
            # AT THE BOX'S OWN PRECISION. Comparing the raw floats reported a change the
            # control cannot even display: a spin box with three decimals stores 20.4717
            # as 20.472, so two conditions recording the SAME interval looked different
            # and every switch warned. A warning that fires when nothing changed is worse
            # than none, because the real one then reads as noise.
            digits = box.decimals() if hasattr(box, "decimals") else 6
            if old and round(old, digits) != round(new, digits) and switched:
                moved.append(f"{label} {old:g} -> {new:g}")
            if not old or switched:
                box.setValue(new)
                box.setToolTip(f"read from {crops_path.name}")

        # THE GRID WINS OVER THE FILE, for the condition on screen. It is seeded FROM
        # the file, so they agree until somebody edits a cell or presses copy -- and at
        # that moment the grid is the newer answer and the file is what it is about to
        # replace. Reading the file here instead would make the copy button decorative:
        # it would fill the table and the run would still use the old numbers.
        recorded = dict(payload)
        recorded.update({k: v for k, v in self._calibrations.get(here, {}).items()
                         if v is not None})

        adopt(self._pixel_nm, recorded.get("pixel_size_nm"), "pixel nm")
        adopt(self._interval_ms, recorded.get("frame_interval_s"), "interval ms", 1000.0)
        # THE RECORDED INTERVAL IS CHECKED AGAINST THE ACQUISITION ITSELF, because a
        # wrong one is invisible: every D scales as 1/tau while the fractions do not
        # move, so nothing downstream looks wrong. `.nd2` carries per-frame timing after
        # all, so this costs one metadata read of one movie.
        if found:
            self._check_interval(payload.get("frame_interval_s"), found[0])
        adopt(self._emission_nm, recorded.get("emission_nm"), "emission nm")
        adopt(self._numerical_aperture, recorded.get("numerical_aperture"), "NA")
        self._calibration_from = here
        if moved:
            # LOUD, because this is the one change on screen that a person did not make
            # and that moves every D. A silent re-calibration is the same defect as an
            # inherited one, pointing the other way.
            self._log.appendPlainText(
                f"calibration follows the condition: {here} records "
                + ", ".join(moved)
                + ". These conditions were not imaged alike, so comparing their D "
                  "values also compares acquisitions.")
        if recorded.get("label") and self._label.currentText() == "(not set)":
            index = self._label.findText(recorded["label"])
            if index >= 0:
                self._label.setCurrentIndex(index)
        from .rawmovie import movie_name

        self._movies = [p for p in found if movie_name(p) in self._crops]
        missing = [movie_name(p) for p in found if movie_name(p) not in self._crops]
        note = (f"{why}\n"
                f"Tracking {len(self._movies)} raw acquisition(s) over their "
                f"recorded crop"
                f" ({payload['pixel_size_nm']:g} nm/px, "
                f"{payload['frame_interval_s'] * 1000:.4f} ms)")
        if missing:
            note += ("\n" + f"No crop for {len(missing)}: "
                     + ", ".join(missing[:6]) + (" ..." if len(missing) > 6 else "")
                     + ": these movies are left out of tracking.")
        self._movies_label.setText(note)

    def _scan_movies(self):
        from .trackmate_run import TrackMateRunError, discover_movies

        if self._root is None:
            # no folder, so no movies, and the label says which of those it is. An
            # empty list under a blank label reads as "this folder has none", which is
            # a different problem from not having chosen one yet.
            self._movies = []
            self._movies_label.setText(
                "no folder chosen: press “Choose folder…” above")
            self._refresh_stamp()
            return
        if self._source == SOURCE_RAW:
            self._scan_raw()
            self._refresh_stamp()
            # A NEW MOVIE SET IS A NEW MEASUREMENT. The threshold is a property of these
            # movies' noise, so carrying one across a folder change would report the old
            # session's number over the new session's data.
            self._autocompute_threshold()
            return

        self._crops = {}
        self._pattern = self._pattern_edit.text().strip() or None
        try:
            self._movies = list(discover_movies(self._root, pattern=self._pattern))
        except TrackMateRunError as exc:
            self._movies = []
            self._movies_label.setText(
                f"{exc}\n\nUse 'Choose folder…' if you opened the project root instead of "
                "one condition, or edit the layout below.")
        else:
            # WHICH LAYOUT WAS RESOLVED IS SAID OUT LOUD. The flat stack is the prepared
            # movie; `tif/<movie>/<movie>/` is u-track's own exploded copy and only
            # exists after u-track has run, so "30 movies" alone would not distinguish
            # a fresh project from one already tracked.
            where = self._pattern or (
                "tif/<movie>.tif" if self._movies[0].is_file()
                else "u-track's tif/<movie>/<movie>/")
            self._movies_label.setText(
                f"Tracking {len(self._movies)} prepared stack(s): {where}, "
                f"{self._movies[0].name} … {self._movies[-1].name}")
        self._refresh_stamp()
        self._autocompute_threshold()

    # -- settings -------------------------------------------------------------
    def _go_to_tracking(self):
        """Show the tracking page, and have the numbers ready when it appears.

        ASKED FOR: "in window 2, when it opens -> automatically calculate the values like
        in adaptive mode". The compute is already scheduled when the window loads a
        folder, but a user who sets a crop, or picks a folder, and only then walks to
        this page would arrive before that had anything to work with. Asking here as well
        costs nothing: `_autocompute_threshold` is cached on (movies, radius), so
        arriving a second time re-measures nothing.
        """
        # RE-READ ON THE WAY THROUGH, so the prompt describes the folder as it is now
        # rather than as it was when the window opened -- somebody may have segmented in
        # between, which is exactly what the prompt asks them to do.
        self._refresh_outline_state()
        self._pages.setCurrentIndex(1)
        self._autocompute_threshold()
        self._explain_threshold()

    def _refresh_reach(self, *_args):
        """Fill the two lines that depend on the frame interval.

        Nothing is invented when tau is not set. Both quantities are functions of the
        frame interval, and this project has twice paid for a calibration that stood in
        where a real value was missing, so with no interval the lines say what is
        missing rather than printing a number that describes a default.
        """
        from .trackmate_run import (apparent_D_ceiling, gap_window,
                                    max_measurable_D)

        tau_s = float(self._interval_ms.value()) / 1000.0
        pixel_nm = float(self._pixel_nm.value())
        if not tau_s or not pixel_nm:
            missing = " and ".join(
                [w for w, ok in (("frame interval", tau_s), ("pixel size", pixel_nm))
                 if not ok])
            self._link_reach.setText(
                f"set the {missing} to see the fastest D this radius can measure")
            self._gap_time.setText("set the frame interval to see the gap in ms")
            return

        radius = float(self._link.value())
        reach = max_measurable_D(radius_px=radius, tau_s=tau_s,
                                 pixel_size_nm=pixel_nm)
        ceiling = apparent_D_ceiling(radius_px=radius, tau_s=tau_s,
                                     pixel_size_nm=pixel_nm)
        self._link_reach.setText(
            f"at {self._interval_ms.value():.4g} ms this radius measures D up to "
            f"<b>{reach:.2f}</b> µm²/s within 10 %; a faster population reads low and "
            f"never above {ceiling:.2f}, however fast it really is")

        gap = gap_window(gap_frames=int(self._gap.value()), tau_s=tau_s)
        # the time, and not a distance, and the first version of this line got it
        # wrong in a way this project has a name for. It quoted how far a molecule
        # at "this setup's fastest measurable D" travels during the gap, and printed
        # **483 nm for both 20.47 ms and 51.75 ms**. That is the tell: the max
        # measurable D itself scales as 1/tau, so D x t cancels and the distance is a
        # function of the GAP COUNT alone. It looked like a property of the
        # acquisition and was a property of the formula.
        #
        # `gap_window` refuses to default `d_um2_s` for exactly this reason, and the
        # window follows it: the unobserved TIME is the quantity that actually
        # varies between acquisitions, so that is what is shown.
        self._gap_time.setText(
            f"= <b>{gap['unobserved_ms']:.0f} ms</b> unobserved. The frame count "
            f"does not carry between acquisitions, the time does: 3 frames is 61 ms "
            f"at 20.5 ms and 155 ms at 51.7 ms.")

    def _threshold_typed(self, _value):
        self._threshold_provenance = "typed"

    def _measurable_movies(self, owner=None):
        """The movies a threshold can be measured on, which is not the same set that may
        be RUN.

        `_scan_raw` requires a SAVED `crops.json`, and it is right to: `crops.json` is the
        single record of what was tracked, so a run against ranges that exist only in this
        window would be unreproducible. Measuring is not running. The noise inside a crop
        can be measured from the ranges on screen, and refusing to until they are saved
        left every condition reading "threshold: not measured" while its ranges sat
        proposed and complete.

        Returns ``(movies, crops)`` or ``([], {})`` when the ranges are incomplete: a
        threshold measured over some of the movies is a different quantity from one
        measured over the set, and nothing on screen would say which it was.
        """
        if owner is not None and owner in getattr(self, "_conditions", {}):
            # A DEFERRED MEASUREMENT MEASURES ITS OWN CONDITION. Reading the live panel
            # here would measure whichever folder the walk had moved to and file it under
            # the one that finished.
            from .rawmovie import discover_raw_movies

            folder = self._conditions[owner]
            acq = folder / "raw" if (folder / "raw").is_dir() else folder
            found, _companions = discover_raw_movies(acq)
            crops = self._session_crops.get(owner, {})
            if found and len(crops) >= len(found):
                return list(found), dict(crops)
            return [], {}
        if self._movies:
            return list(self._movies), dict(self._crops)
        from .rawmovie import discover_raw_movies

        acq_root, _why = self._acquisition_root()
        if acq_root is None or not acq_root.is_dir():
            return [], {}
        found, _companions = discover_raw_movies(acq_root)
        crops = self._crop_panel.crops()
        if not found or len(crops) < len(found):
            return [], {}
        return list(found), dict(crops)

    def _autocompute_threshold(self):
        """Measure the threshold now, if adaptive is selected and there is data.

        ASKED FOR: "when window 2 is opened or adaptive mode is chosen -> calculate the
        adaptive values automatically". A mode called adaptive that then waits to be told
        to adapt is a mode in name only.

        Three things it will not do, each because the alternative reads wrong:
        - it does not run without movies (an empty measurement reported as a threshold);
        - it does not RE-run for a movie set and radius it has already measured, so
          flipping back to adaptive is instant rather than another wait;
        - it does not run before the window is on screen, or the first thing a user sees
          is a frozen grey rectangle. `QTimer.singleShot(0, ...)` lets it paint first.
        """
        from .presets import ADAPTIVE

        # `_scan_movies` can run while the settings page is still being built, so the
        # box may not exist yet. Nothing is lost: the constructor applies the preset at
        # the end, which computes then.
        if not hasattr(self, "_preset") or not hasattr(self, "_log"):
            return
        movies, crops = self._measurable_movies()
        if self._preset.currentText() != ADAPTIVE or not movies:
            self._explain_threshold()
            return
        # THE CROP RANGES ARE PART OF THE KEY, and leaving them out is what made the
        # settings page arrive with nothing computed.
        #
        # `batch_threshold` measures the noise INSIDE each crop (`crops=self._crops`),
        # so the answer depends on them, and the ranges are typically set AFTER the
        # folder is first scanned. With the ranges out of the key, the scan that
        # follows "Save crop ranges" saw the same movies at the same radius, decided it
        # had already measured, and skipped. A cache keyed on half the inputs serves the
        # wrong answer for the other half; this project records the same shape in
        # `require_all_inside` and in the canonical cache.
        # THE SET ACTUALLY MEASURED, which is not always `self._movies`: a condition
        # whose ranges are proposed but not yet saved is measurable and has no scanned
        # movie list. Keying on the scan would give every such condition the same empty
        # key and serve the first one's answer to all of them.
        key = (tuple(sorted(str(m) for m in movies)),
               round(float(self._radius.value()), 3),
               tuple(sorted((k, tuple(v)) for k, v in (crops or {}).items())))
        if getattr(self, "_autocomputed_for", None) == key:
            return
        self._autocomputed_for = key
        # THE CONDITION IS CAPTURED NOW, not read when the timer fires. The measurement
        # is deferred so the window can paint first, and the walk moves to the next
        # condition as soon as this one finishes reading -- so by the time it ran,
        # `self._root` had already moved and the result was filed under the condition
        # that happened to be current. A threshold is `k x sigma_LoG` of ITS OWN movies;
        # recording one against another condition is the same defect as lending a crop
        # range across folders, and this window's own comments warn about it.
        owner = self._root.name if self._root is not None else None
        # `self` IS THE TIMER'S CONTEXT, so a window deleted before the timer fires
        # cancels it. A bare lambda outlives the window and then writes to a log that no
        # longer exists ("Internal C++ object already deleted") -- 21 teardown errors in
        # the suite, every one from this line.
        QTimer.singleShot(0, self, lambda: self._compute_threshold_clicked(owner=owner))

    def _set_quietly(self, pairs):
        """Fill boxes without each one re-triggering the stamp mid-update."""
        for widget, value in pairs:
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)

    def _preset_text_changed(self, name):
        self._preset_just_changed = True
        self._apply_preset(name)

    def _preset_activated(self, _index):
        """A user selection. Applies unless `currentTextChanged` just did."""
        if self._preset_just_changed:
            self._preset_just_changed = False
            return
        self._apply_preset(self._preset.currentText())

    def _apply_preset(self, name):
        """Apply whichever KIND of choice the box is on.

        Two of the entries are not settings at all: they say where the settings come
        from. ADAPTIVE fills what can be derived and then measures the threshold from
        the movies; MANUAL leaves every box alone. Everything after them is a saved set.
        """
        from .presets import ADAPTIVE, MANUAL, PresetError, load_preset
        from .trackmate_run import ADAPTIVE_BASE

        if name == MANUAL:
            self._log.appendPlainText(
                "manual: every box is yours. The stamp below says whether the result "
                "stays comparable with u-track.")
            self._refresh_stamp()
            return
        if name == ADAPTIVE:
            self._set_quietly(((self._gap, ADAPTIVE_BASE["max_frame_gap"]),
                               (self._link,
                                ADAPTIVE_BASE["linking_max_distance_px"])))
            # THE RADIUS IS DERIVED IF IT CAN BE, and if it cannot the window says why
            # rather than quietly leaving the previous value looking derived.
            self._radius_from_optics_clicked(quiet=True)
            self._log.appendPlainText(
                "adaptive: radius from the optics, gap and linking from the measured "
                "u-track-comparable set. The threshold does not transfer between "
                "datasets, so it is measured on these movies.")
            self._refresh_stamp()
            # ADAPTIVE MEANS ADAPTIVE, so it computes rather than asking the user to
            # press a second button whose necessity the mode's own name denies.
            self._autocompute_threshold()
            return
        try:
            saved = load_preset(name)
        except PresetError as exc:
            self._log.appendPlainText(f"could not load preset {name!r}: {exc}")
            self._refresh_stamp()
            return
        boxes = {"radius_px": self._radius, "threshold": self._threshold,
                 "max_frame_gap": self._gap, "linking_max_distance_px": self._link}
        unknown = [k for k in saved.settings if k not in boxes]
        self._set_quietly([(boxes[k], v) for k, v in saved.settings.items()
                           if k in boxes])
        # The preset's own provenance is inherited, so re-saving it does not launder a
        # measured number into a typed one or the other way about.
        self._threshold_provenance = saved.provenance or f"from preset '{saved.name}'"
        note = f"preset '{saved.name}' applied"
        if saved.comment:
            note += f": {saved.comment}"
        if saved.provenance:
            note += f"  [{saved.provenance}]"
        self._log.appendPlainText(note)
        if unknown:
            # NAMED, NOT SWALLOWED: a preset written by a newer version may carry a
            # setting this window has no box for, and silently dropping it would mean
            # the run does not match the preset the user believes they selected.
            self._log.appendPlainText(
                f"  Warning: {', '.join(unknown)} is in the preset but has no box in "
                f"this window, so the run will not use it")
        self._refresh_stamp()

    def _save_preset_clicked(self):
        """Name and comment the settings on screen, and write them out."""
        from .presets import PresetError, save_preset

        settings = self.current_settings()
        if settings is None:
            self._log.appendPlainText(
                "the boxes do not make a valid setting, so there is nothing to save")
            return
        name, comment = self._ask_preset_name()
        if not name:
            return
        values = {"radius_px": settings.radius_px, "threshold": settings.threshold,
                  "max_frame_gap": settings.max_frame_gap,
                  "linking_max_distance_px": settings.linking_max_distance_px}
        provenance = getattr(self, "_threshold_provenance", "typed")
        try:
            path = save_preset(name, values, comment=comment, provenance=provenance)
        except PresetError as exc:
            # OVERWRITING IS THE USER'S CALL, ASKED IN WORDS. Two settings under one
            # name is exactly the ambiguity nothing downstream could untangle.
            if "already exists" not in str(exc):
                self._log.appendPlainText(f"not saved: {exc}")
                return
            if not self._confirm(
                    f"A preset named '{name}' already exists.\n\nReplace it? The old "
                    f"values are discarded; runs already made with them keep their "
                    f"own stamp."):
                self._log.appendPlainText(f"not saved: '{name}' left as it was")
                return
            path = save_preset(name, values, comment=comment, provenance=provenance,
                               overwrite=True)
        self._log.appendPlainText(f"saved preset '{name}' to {path}")
        self._reload_presets(select=name)

    def _reload_presets(self, *, select=None):
        from .presets import broken_presets
        from .trackmate_run import preset_choices

        # A PRESET THAT CANNOT BE READ IS NAMED, NOT SKIPPED SILENTLY. A user who saved
        # one and then does not see it in the list must be told why, or they conclude
        # the save button does nothing.
        for bad in broken_presets():
            self._log.appendPlainText(f"preset file unreadable, so not offered: {bad}")
        self._preset.blockSignals(True)
        self._preset.clear()
        self._preset.addItems(preset_choices())
        if select is not None:
            index = self._preset.findText(select)
            if index >= 0:
                self._preset.setCurrentIndex(index)
        self._preset.blockSignals(False)

    def _ask_preset_name(self):
        """Ask for a name and a comment. Returns ``(name, comment)``, name empty if
        cancelled.

        The comment is asked for, not OPTIONAL-LOOKING. A threshold with no note is a
        number somebody typed once; six months on nobody can say whether it was measured
        or guessed.
        """
        from qtpy.QtWidgets import (QDialog, QDialogButtonBox, QFormLayout, QLabel,
                                    QLineEdit)

        dialog = QDialog(self)
        dialog.setWindowTitle("Save these settings as a preset")
        form = QFormLayout(dialog)
        name = QLineEdit()
        name.setPlaceholderText("e.g. strain A minimal medium, March session")
        form.addRow("Name", name)
        comment = QLineEdit()
        comment.setPlaceholderText("what it is for, and anything you had to judge")
        form.addRow("Comment", comment)
        where = QLabel(self._preset_location_note())
        where.setWordWrap(True)
        where.setStyleSheet("color: #666; font-size: 11px;")
        form.addRow("", where)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() != QDialog.Accepted:
            return "", ""
        return name.text().strip(), comment.text().strip()

    def _preset_location_note(self):
        """Where presets go, and why not the installation folder. Pure string work."""
        from .presets import preset_dir

        return (f"Saved in {preset_dir()}, your own settings folder. It is kept when "
                f"SMTracker is reinstalled.")

    def _confirm(self, question):
        from qtpy.QtWidgets import QMessageBox

        return QMessageBox.question(
            self, "SMTracker", question,
            QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes

    def _radius_from_optics_clicked(self, quiet=False):
        """Set the spot radius from the optics, or say exactly what is missing."""
        from .optics import OpticsError, detection_radius_px

        emission = self._emission_nm.value()
        aperture = self._numerical_aperture.value()
        pixel = self._pixel_nm.value()
        missing = [label for label, value in (("emission wavelength", emission),
                                              ("NA", aperture),
                                              ("pixel size", pixel)) if not value]
        if missing:
            self._log.appendPlainText(
                f"cannot derive the spot radius: {' and '.join(missing)} not set on "
                f"the Movies page. The box keeps {self._radius.value():.2f} px, which "
                f"was not derived from this microscope.")
            return
        try:
            radius = detection_radius_px(emission_nm=emission,
                                         numerical_aperture=aperture,
                                         pixel_size_nm=pixel)
        except OpticsError as exc:
            self._log.appendPlainText(f"cannot derive the spot radius: {exc}")
            return
        before = self._radius.value()
        self._set_quietly(((self._radius, radius),))
        if not quiet or abs(radius - before) > 0.005:
            self._log.appendPlainText(
                f"spot radius {radius:.2f} px, from emission {emission:.0f} nm, "
                f"NA {aperture:.2f}, pixel {pixel:.1f} nm (was {before:.2f})")
        self._refresh_stamp()

    def _measure_link_clicked(self):
        """Read the linking radius off a tracked result in this folder.

        The answer to "D is not known yet". It is not needed: D is fitted from the
        step lengths, so a pilot run's steps are one level less derived than D. This
        measures the distribution and the ambiguity ceiling and prints both, the floor
        from how far molecules actually moved, the ceiling from how close the next one
        is. Nothing here is computed in Qt; both come from `trackmate_run`.
        """
        from .trackmate_run import (TrackMateRunError, linking_ambiguity,
                                    step_length_summary)

        table, source, radius_used = self._pilot_table()
        if table is None:
            self._log.appendPlainText(source)          # `source` carries the reason
            return
        try:
            steps = step_length_summary(table, radius_used_px=radius_used)
            crowd = linking_ambiguity(table, radius_px=float(self._link.value()))
        except TrackMateRunError as exc:
            self._log.appendPlainText(f"could not measure: {exc}")
            return
        self._log.appendPlainText(
            f"linking radius from {source}: {steps['n_steps']} consecutive-frame steps: "
            f"median {steps['median']:.2f}, p95 {steps['p95']:.2f}, "
            f"p99 {steps['p99']:.2f}, max {steps['max']:.2f} px")
        if radius_used:
            self._log.appendPlainText(
                f"  the pilot ran at {radius_used:g} px, so no step above that can "
                f"appear at any percentile; {steps['censored_fraction'] * 100:.1f} % of "
                f"steps sit in its top tenth. Re-measure from a deliberately generous "
                f"run if that fraction is large.")
        self._log.appendPlainText(
            f"  at the current {crowd['radius_px']:g} px, "
            f"{crowd['fraction_with_neighbour'] * 100:.1f} % of detections have another "
            f"within one radius: the link is a guess for those. This is the ceiling; "
            f"the step distribution above is the floor.")
        self._log.appendPlainText(
            f"  p99 = {steps['p99']:.2f} px would be a defensible setting; the box "
            f"holds {self._link.value():.2f}.")

    def _radius_check_clicked(self):
        """Track a couple of movies at two radii and say what the narrow one hid.

        WHY A SECOND RUN IS UNAVOIDABLE. A pilot can only ever show steps its own radius
        allowed, so reading a percentile off one measures the setting: on this project's
        sweep the p99 rule returns 4.87 px from a 5 px pilot and 10.98 px from a 12 px one,
        the same movies both times. Counting what APPEARS when the radius opens is the
        only statistic that sees past that, and it needs two runs.
        """
        from .movieprep import find_crops, load_crops

        # THE CONDITION THE RUN BUTTON WOULD USE, resolved the same way, so the check
        # and the run cannot disagree about what they are talking about.
        name = (self._run_condition_box.currentText()
                if self._run_condition_box.isVisible() else None)
        inputs, why = self._run_inputs(name)
        if inputs is None:
            self._log.appendPlainText(f"cannot check the radius: {why}")
            return
        movies, crops, settings, _out = inputs
        if self._install is None:
            self._log.appendPlainText("Fiji is not set up yet")
            return

        folder = (self._conditions.get(name) if (self._conditions and name)
                  else self._root)
        acq = folder / "raw" if (folder / "raw").is_dir() else folder
        # THE ACQUISITION HAS NO DEFAULT, and this is where that rule bites hardest: the
        # verdict is `implied_D` against what the radius can measure, and both are
        # functions of tau. Guessing 20 ms on a 51.7477 ms acquisition would judge this
        # project by another one's rule, which is the defect the stored bands had.
        found = find_crops(acq, folder)
        payload = load_crops(found) if found else {}
        tau_s = payload.get("frame_interval_s")
        pixel_nm = payload.get("pixel_size_nm")
        if not tau_s or not pixel_nm:
            self._log.appendPlainText(
                "cannot check the radius: crops.json holds no frame interval and pixel "
                "size, and neither has a default: a radius means a different D at a "
                "different frame interval. Fill them in the settings grid and save.")
            return

        narrow = float(self._link.value())
        wide = round(narrow * RADIUS_CHECK_MULTIPLE, 1)
        sample = list(movies)[:RADIUS_CHECK_MOVIES]
        # THE COST IS SHOWN BEFORE IT IS PAID, as the batch already does. Two runs over
        # the same movies is the whole design, so the number is 2 x n.
        seconds = 2 * len(sample) * SECONDS_PER_MOVIE
        self._log.appendPlainText(
            f"--- radius check: {len(sample)} movie(s) at {narrow:g} px AND at "
            f"{wide:g} px, everything else identical. About {seconds / 60:.0f} min.")
        self._radius_check.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setRange(0, 2 * len(sample))
        self._progress.setValue(0)
        out = folder / "trackmate_radius_check"
        job = _RadiusCheckJob(
            sample, out, settings, self._install, crops=crops,
            thresholds=self._thresholds_for(name), narrow_px=narrow, wide_px=wide,
            tau_s=float(tau_s), pixel_size_nm=float(pixel_nm))
        job.signals.progress.connect(self._on_progress)
        job.signals.finished.connect(self._on_radius_check)
        job.signals.failed.connect(self._on_radius_check_failed)
        self._pool.start(job)

    def _on_radius_check(self, got):
        """Report the yield, the D it implies, and what that D is judged against."""
        self._radius_check.setEnabled(True)
        self._progress.setVisible(False)
        pct = 100 * got["beyond_narrow"]
        self._log.appendPlainText(
            f"--- {pct:.1f} % of steps lie beyond {got['narrow_radius_px']:g} px "
            f"({got['n_steps_narrow']} steps at {got['narrow_radius_px']:g} px, "
            f"{got['n_steps_wide']} at {got['wide_radius_px']:g})")
        lo, hi = got["implied_D_interval"]
        self._log.appendPlainText(
            f"    that implies D of at least {got['implied_D_floor']:.2f} "
            f"[{lo:.2f}, {hi:.2f}] um2/s, against {got['max_measurable_D']:.2f} this "
            f"radius measures to within 10 % and a {got['apparent_D_ceiling']:.2f} "
            f"ceiling where a fit returns the setting: {got['band'].upper()}: "
            f"{got['verdict']}")
        # THE ONE THING IT MUST NOT BE READ AS. Widening buys those steps and pays in
        # wrong links: 5 px was validated at +1.4 pp against the deposited references
        # while 12 px inflated the fractions by 12.8 pp in 11 of 12 groups.
        self._log.appendPlainText(
            "    this does not call for a wider radius. Report it beside a "
            "fast D as a measure of how far that D is from being identified.")

    def _on_radius_check_failed(self, message):
        self._radius_check.setEnabled(True)
        self._progress.setVisible(False)
        self._log.appendPlainText(f"radius check failed: {message}")

    def _pilot_table(self):
        """A tracked result already in this folder, if there is one.

        Returns ``(table, description, radius_used_px)`` or ``(None, reason, None)``.
        u-track output is preferred because this project's folders have it and its
        settings are recorded; a `trackmate/` folder is read otherwise.
        """
        from pathlib import Path

        root = Path(self._root)
        mats = sorted(root.glob("tif/*/TrackingPackage/tracks/*.mat"))
        if mats:
            import pandas as pd

            from .ingest import read_utrack_mat

            rows, failed = [], 0
            for path in mats[:8]:
                try:
                    one = read_utrack_mat(path)
                except Exception:                     # noqa: BLE001 - counted, not shown
                    failed += 1
                    continue
                one = one.copy()
                one["movie"] = path.parent.parent.parent.name
                rows.append(one)
            if rows:
                # THE RADIUS THAT PRODUCED THESE STEPS, so the truncation can be
                # reported. `read_project_metadata` returns a LIST, one entry per movie,
                # a `getattr` on the list quietly returned None and the truncation line
                # simply never printed, which is the silent-wrong-answer shape rather
                # than a missing feature. The MAXIMUM is the right reduction: it is the
                # widest link any movie here could have made.
                radius = None
                try:
                    from .utrack_meta import read_project_metadata

                    recorded = [m.max_search_radius_px
                                for m in read_project_metadata(root)
                                if m.max_search_radius_px]
                    radius = max(recorded) if recorded else None
                except Exception as exc:               # noqa: BLE001 - named, not hidden
                    self._log.appendPlainText(
                        f"  (no recorded search radius: {type(exc).__name__}: {exc}: "
                        f"the truncation of the step distribution cannot be reported)")
                note = f"u-track output, {len(rows)} movie(s)"
                if failed:
                    note += f" ({failed} unreadable)"
                return pd.concat(rows, ignore_index=True), note, radius
        xmls = sorted(root.glob("trackmate/*.xml"))
        if xmls:
            from .trackmate import read_trackmate_xml

            import pandas as pd

            rows = []
            for path in xmls[:8]:
                try:
                    one = read_trackmate_xml(path)
                except Exception:                      # noqa: BLE001
                    continue
                one = one.copy()
                from .rawmovie import movie_name

                one["movie"] = movie_name(path)
                rows.append(one)
            if rows:
                return (pd.concat(rows, ignore_index=True),
                        f"TrackMate output, {len(rows)} movie(s)",
                        float(self._link.value()))
        return (None,
                f"no tracked result in {root} to measure from. Track a movie or two at "
                f"a generous linking radius first: the step distribution is "
                f"truncated by the radius that produced it.", None)

    def _label_changed(self, name):
        """Fill the emission wavelength from the chosen label, and say what else it
        changes. Fills, never locks: a preset that decided the localisation floor on
        the user's behalf would be choosing the fluorophore for them."""
        from .optics import OpticsError, preset

        if name == "(not set)":
            self._refresh_stamp()
            return
        try:
            chosen = preset(name)
        except OpticsError:
            self._refresh_stamp()
            return
        self._emission_nm.blockSignals(True)
        self._emission_nm.setValue(chosen.emission_nm)
        self._emission_nm.blockSignals(False)
        self._refresh_stamp()

    def optics_note(self):
        """What the two optical numbers imply, or what is missing. Pure string work;
        every number comes from `smtracker.optics`.

        The floor is the point, not the radius. It is what decides whether a slow-state
        D is a measurement or a bound, and this project's own candidates span 30 to
        99 nm (a factor of 3.2) with the crossover for the reference D at 66 nm
        sitting inside that range.
        """
        from .optics import OpticsError, detection_radius_px, psf_sigma_nm, preset

        emission = self._emission_nm.value()
        aperture = self._numerical_aperture.value()
        if not emission or not aperture:
            missing = [name for name, value in (("emission wavelength", emission),
                                                ("NA", aperture)) if not value]
            return (f"PSF width not derivable: {' and '.join(missing)} not set. "
                    f"Where u-track recorded psfSigma it is used instead; a raw movie "
                    f"has no such record.")
        try:
            sigma = psf_sigma_nm(emission_nm=emission, numerical_aperture=aperture)
            pixel = self._pixel_nm.value()
            radius = (detection_radius_px(emission_nm=emission,
                                          numerical_aperture=aperture,
                                          pixel_size_nm=pixel) if pixel else None)
        except OpticsError as exc:
            return str(exc)
        note = f"PSF sigma {sigma:.1f} nm"
        if radius is not None:
            note += (f"; a radius derived from it would be {radius:.2f} px "
                     f"(the box above is {self._radius.value():.2f})")
        else:
            note += "; set a pixel size to see the radius this implies"
        chosen = self._label.currentText()
        if chosen != "(not set)":
            try:
                item = preset(chosen)
            except OpticsError:
                item = None
            if item is not None and item.photoactivated:
                note += ("  ⚠ Photoactivated: molecules appear sparsely throughout the "
                         "acquisition and do not bleach down to single molecules, "
                         "so a crop chosen from a bleaching curve does not describe "
                         "this experiment.")
        return note

    def current_settings(self):
        """The boxes as a `TrackMateSettings`, or ``None`` if they do not make one."""
        from .trackmate_run import TrackMateRunError, TrackMateSettings

        try:
            return TrackMateSettings(
                radius_px=self._radius.value(),
                threshold=self._threshold.value(),
                max_frame_gap=self._gap.value(),
                # BOTH linking distances move together. TrackMate takes them separately
                # and u-track has one search radius, so letting them drift apart here
                # would create a policy u-track has no equivalent of, silently.
                linking_max_distance_px=self._link.value(),
                gap_closing_max_distance_px=self._link.value())
        except TrackMateRunError:
            return None

    def _check_interval(self, recorded_s, movie_path):
        """Compare the recorded frame interval with the acquisition's own timing.

        Reported, never corrected silently. The recorded value may be the deliberate one,
        and a window that overwrote a calibration would be doing the thing this project
        most guards against. It is checked ONCE, on one movie, and only when a raw
        acquisition is beside the record.
        """
        from .rawmovie import check_recorded_interval

        key = (str(movie_path), recorded_s)
        if getattr(self, "_interval_checked", None) == key:
            return
        self._interval_checked = key
        verdict = check_recorded_interval(recorded_s, movie_path)
        if verdict["agrees"] is False:
            self._log.appendPlainText(verdict["note"])
            self._log.appendPlainText(
                f"  the acquisition's value is {verdict['measured_s'] * 1000:.4f} ms: "
                f"set it above if the recorded one is wrong. Nothing is changed for you.")
        elif verdict["note"]:
            self._log.appendPlainText(verdict["note"])

    def _crops_changed(self):
        """A range was set or proposed, so the movie list can change."""
        self._crop_status.setText(
            f"{len(self._crop_panel.crops())} range(s) set, "
            f"{len(self._crop_panel.missing())} still missing")

    def _calibration_for(self, name) -> dict:
        """One condition's calibration, with the boxes filling whatever the grid lacks.

        RESOLVED AT THE POINT OF USE, not seeded once. `_detect_conditions` runs before
        the boxes have been filled from the active condition's crops.json, so seeding
        there copied zeros; and the boxes keep changing as conditions are switched. The
        grid always wins where it has an answer, which keeps a per-condition edit
        authoritative.
        """
        values = dict(self._calibrations.get(name, {}))
        fallback = {"pixel_size_nm": self._pixel_nm.value() or None,
                    "frame_interval_s": (self._interval_ms.value() / 1000.0
                                         if self._interval_ms.value() else None),
                    "emission_nm": self._emission_nm.value() or None,
                    "numerical_aperture": self._numerical_aperture.value() or None}
        for key, value in fallback.items():
            if value is not None and values.get(key) is None:
                values[key] = value
        return values

    def _conditions_ready_to_save(self):
        """``(ready, skipped)`` for the whole session.

        A condition is ready when every one of its movies has a range and its calibration
        carries the two numbers that cannot be defaulted. Everything else is NAMED with
        the reason: a save that quietly covered three conditions of four would leave a
        folder looking complete.
        """
        ready, skipped = [], {}
        for name in self._conditions:
            # THE PANEL IS THE TRUTH FOR THE CONDITION ON SCREEN, but only when it
            # holds something: a panel that has not loaded yet must not erase a set the
            # session already knows about.
            crops = self._session_crops.get(name, {})
            if self._root is not None and name == self._root.name:
                crops = self._crop_panel.crops() or crops
            movies = self._session_read.get(name, (0, 0, 0))[0]
            values = self._calibration_for(name)
            if not crops:
                skipped[name] = ("no crop ranges yet: it has not been read"
                                 if not movies else f"0 of {movies} movies have a range")
            elif movies and len(crops) < movies:
                skipped[name] = f"{len(crops)} of {movies} movies have a range"
            elif not values.get("pixel_size_nm") or not values.get("frame_interval_s"):
                skipped[name] = "no pixel size or frame interval set"
            else:
                ready.append(name)
        return ready, skipped

    def _start_next_save(self):
        """Save one condition from the queue, then let the finish handler take the next.

        A QUEUE RATHER THAN A LOOP, because each save is a background job: looping would
        either block the window or start every condition at once, and the second is worse
        because they write through the same panel.
        """
        if not self._save_queue:
            return False
        name = self._save_queue.pop(0)
        # SHOWN PER CONDITION, because the finish handler hides it. Reported: "it shows
        # the progress bar only for the first condition for save crop". `_save_crops_done`
        # tears the bar down when a job ends, and the next job started with it hidden --
        # so a four-condition save looked like it had stopped after the first.
        total = getattr(self, "_save_total", 0) or 1
        index = total - len(self._save_queue)
        self._export_progress.setRange(0, 0)
        self._export_progress.setFormat(f"saving {name} ({index} of {total})…")
        self._export_progress.show()
        values = self._calibration_for(name)
        # ALWAYS ADOPT, and compare against the PANEL rather than `_root`. The queue moves
        # the panel and leaves `_root` where it was, so the second condition looked like
        # the current one, was not adopted, and its job wrote to the FIRST condition's
        # folder while the log said otherwise: one crops.json overwritten by another
        # condition's ranges, which is the worst outcome this queue could have.
        folder = self._conditions[name]
        acq = folder / "raw" if (folder / "raw").is_dir() else folder
        self._adopting = True
        try:
            self._crop_panel.adopt(
                acq, output_root=folder,
                curves=self._session_curves.get(name),
                crops=self._session_crops.get(name),
                errors=self._session_errors.get(name))
        finally:
            self._adopting = False
        self._saving_condition = name
        self._crop_status.setText(f"saving {name}…")
        job = _SaveCropsJob(
            self._crop_panel,
            pixel_size_nm=float(values["pixel_size_nm"]),
            frame_interval_s=float(values["frame_interval_s"]),
            write_curves=True,
            label=values.get("label"),
            emission_nm=values.get("emission_nm"),
            numerical_aperture=values.get("numerical_aperture"),
            projection=self.projection_kind())
        job.signals.progress.connect(self._save_crops_progress)
        job.signals.finished.connect(self._save_crops_finished)
        job.signals.failed.connect(self._save_crops_failed)
        self._save_job = job
        self._pool.start(job)
        return True

    def _save_crops_clicked(self):
        """Write crops.json: ranges plus calibration, no stacks.

        The calibration travels WITH the ranges deliberately: ranges alone would let a
        run be reproduced at the wrong pixel size or frame interval, and those are the
        two numbers this project keeps having to correct.
        """
        # THE WHOLE SESSION, not the folder on screen. Reported: saving covered only
        # the active condition, so three conditions of four were left with no crops.json
        # and nothing said so. The queue below walks them; a condition that cannot be
        # saved is NAMED with the reason rather than skipped.
        if len(self._conditions) > 1:
            ready, skipped = self._conditions_ready_to_save()
            if skipped:
                self._log.appendPlainText(
                    "not saved: "
                    + "; ".join(f"{k} ({v})" for k, v in skipped.items()))
            if not ready:
                self._crop_status.setText(
                    "no condition is ready to save; see the log for what each needs")
                return
            # STASH FIRST, so the condition on screen contributes what the panel holds
            # rather than an older copy: adopting it back out of the session record is
            # what the queue is about to do to every condition including this one.
            self._stash_condition()
            self._save_queue = list(ready)
            self._save_total = len(ready)
            self._save_return_to = self._root.name if self._root is not None else None
            self._save_crops.setEnabled(False)
            self._export_stacks.setEnabled(False)
            self._export_progress.setRange(0, 0)
            self._export_progress.setFormat("saving crop ranges…")
            self._export_progress.show()
            self._start_next_save()
            return

        if not self._pixel_nm.value() or not self._interval_ms.value():
            self._crop_status.setText(
                "set the pixel size and frame interval first; they are saved in "
                "crops.json with the ranges and have no default")
            return

        missing = self._crop_panel.missing()
        if missing:
            self._crop_status.setText(
                f"{len(missing)} movie(s) have no range yet: "
                + ", ".join(missing[:5]) + (" …" if len(missing) > 5 else ""))
            return
        # OFF THE GUI THREAD, WITH AN INDICATOR. The projections read every movie's
        # cropped frames, which on a real folder is tens of seconds; doing that here
        # froze the window.
        self._save_crops.setEnabled(False)
        self._export_stacks.setEnabled(False)
        self._export_progress.setRange(0, 0)          # busy until the first callback
        self._export_progress.setFormat("saving crop ranges…")
        self._export_progress.show()
        self._crop_status.setText("writing crops.json, curves and projections…")

        job = _SaveCropsJob(
            self._crop_panel,
            pixel_size_nm=float(self._pixel_nm.value()),
            frame_interval_s=float(self._interval_ms.value()) / 1000.0,
            write_curves=True,
            # The optics go in too: they set the localisation floor, which is at
            # least as consequential as the pixel size already recorded.
            label=(None if self._label.currentText() == "(not set)"
                   else self._label.currentText()),
            emission_nm=self._emission_nm.value() or None,
            numerical_aperture=self._numerical_aperture.value() or None,
            projection=self.projection_kind())
        job.signals.progress.connect(self._save_crops_progress)
        job.signals.finished.connect(self._save_crops_finished)
        job.signals.failed.connect(self._save_crops_failed)
        self._save_job = job
        self._pool.start(job)

    def _save_crops_progress(self, done, total, note):
        self._export_progress.setRange(0, max(total, 1))
        self._export_progress.setValue(done)
        who = getattr(self, "_saving_condition", None)
        queued = getattr(self, "_save_total", 0)
        where = ""
        if who and queued > 1:
            where = f"{who} ({queued - len(self._save_queue)} of {queued}): "
        self._export_progress.setFormat(f"{where}{note}: %v / %m")
        self._export_progress.show()
        self._log.appendPlainText(f"  {done}/{total} {note}")

    def _save_crops_finished(self, path):
        self._save_crops_done()
        done = getattr(self, "_saving_condition", None)
        self._crop_status.setText(f"wrote {path.name}"
                                  + (f" for {done}" if done else ""))
        if done:
            self._log.appendPlainText(f"saved {done}: {path}")
        # AFTER the write, on the GUI thread: both of these move files and rebuild the
        # movie list, which is not a worker's business.
        self._organise_if_loose(done)
        self._scan_movies()
        if getattr(self, "_save_queue", None):
            self._start_next_save()
            return
        if getattr(self, "_save_return_to", None):
            # BACK WHERE IT STARTED. The queue moved the panel through every condition,
            # and leaving it on the last one would look like the picker had changed by
            # itself.
            back, self._save_return_to = self._save_return_to, None
            self._conditions_box.setCurrentText(back)
        self._saving_condition = None

    def _save_crops_failed(self, message):
        self._save_crops_done()
        self._crop_status.setText(message)
        self._log.appendPlainText(f"save: {message}")

    def _save_crops_done(self):
        self._export_progress.hide()
        self._save_job = None
        self._save_crops.setEnabled(True)
        self._export_stacks.setEnabled(True)

    def _organise_if_loose(self, condition=None):
        """File loose acquisitions into raw/ and phase/ once the curves are written.

        ASKED FOR: the move "should be happening when i click to generate the bleaching
        curves if the files are in parent folder and not already in raw or phase".

        After the save, not before, and the order is the point: the curves are already in
        memory and are written to the CONDITION folder, which does not move, so saving
        first means a failed organise leaves the lab-book record intact rather than
        costing both. The reverse order would abort before writing anything.

        Safe to move here because this button is disabled while the crop panel is reading
        on Windows a file another process holds open cannot be moved (WinError 32), and
        that guard is why the press can only land when the files are closed.

        The mode comes from the box beside the plan, so "move" versus "copy" stays the
        user's answer rather than this method's.
        """
        from .organise import (OrganiseError, acquisition_folder, apply_layout,
                               inspect_folder, plan_layout)

        # THE CONDITION THAT WAS JUST SAVED, not the one on screen. The save queue adopts
        # the panel through every condition and leaves `_root` where it started, so this
        # filed the FIRST condition's folder once per save and never touched the others:
        # four conditions saved, one organised, and nothing said which. Same shape as the
        # bug in `_start_next_save`'s own comment, one call further down.
        if condition is not None and condition in self._conditions:
            try:
                folder, _why = acquisition_folder(self._conditions[condition])
            except OrganiseError as exc:
                self._log.appendPlainText(f"{condition}: {exc}")
                return
        else:
            folder, _why = self._acquisition_root()
        if folder is None:
            return
        try:
            if not inspect_folder(folder).needs_organising:
                return
            plan = plan_layout(folder)
            if not plan.moves or plan.collisions:
                if plan.collisions:
                    self._log.appendPlainText(
                        f"not filing the acquisitions: {len(plan.collisions)} name "
                        f"collision(s): see the Organise box")
                return
            report = apply_layout(plan, mode=ORGANISE_DEFAULT_MODE)
        except OrganiseError as exc:
            self._log.appendPlainText(f"could not file the acquisitions: {exc}")
            return
        # SAID EVERY TIME, because no control shows it any more. The files moved; that
        # is not something to discover later from an empty folder.
        self._log.appendPlainText(
            f"filed {condition or 'the'} acquisitions into raw/ and phase/ "
            f"({report['mode']}, originals not kept): {report['note']}")
        # ONLY WHEN IT IS THE FOLDER ON SCREEN. Rescanning here while the save queue is
        # walking would point the panel at a condition nobody asked for, which is the
        # view-stealing this window just had removed.
        if condition is None or (self._root is not None
                                 and condition == self._root.name):
            self._crop_root = None
            self._apply_folder_state()

    def projection_kind(self) -> str:
        """"standard deviation" / "sum" on screen, "sd" / "sum" in the code."""
        return "sum" if self._projection.currentText() == "sum" else "sd"

    def _pick_export_conditions(self):
        """Ask whether to export every condition, and let them choose if not.

        ASKED FOR in exactly this shape: "click on export -> asks you write for all
        conditions? yes -> writes for all, no -> condition picker".

        The question is asked rather than assumed because this writes GIGABYTES, unlike
        saving the ranges, which writes kilobytes and needs no permission. Returns a list
        of condition names, or None when the user backed out.
        """
        from qtpy.QtWidgets import (QAbstractItemView, QDialog, QDialogButtonBox,
                                    QListWidget, QVBoxLayout)

        ready, skipped = self._conditions_ready_to_export()
        if not ready:
            self._crop_status.setText(
                "no condition is ready to export; see the log for what each needs")
            if skipped:
                self._log.appendPlainText(
                    "not exported: "
                    + "; ".join(f"{k} ({v})" for k, v in skipped.items()))
            return None

        box = QMessageBox(self)
        box.setWindowTitle("Export cropped stacks")
        box.setText(f"Write cropped stacks for all {len(ready)} conditions?")
        box.setInformativeText(
            "This writes one multi-page TIFF per movie and is the slow, large step: "
            "about a gigabyte per condition.\n\n"
            + ("Not ready: " + "; ".join(f"{k} ({v})" for k, v in skipped.items())
               if skipped else ""))
        all_button = box.addButton("All conditions", QMessageBox.AcceptRole)
        pick_button = box.addButton("Choose…", QMessageBox.ActionRole)
        cancel = box.addButton(QMessageBox.Cancel)
        box.setDefaultButton(all_button)
        box.exec()
        clicked = box.clickedButton()
        if clicked is cancel or clicked is None:
            return None
        if clicked is all_button:
            return list(ready)

        dialog = QDialog(self)
        dialog.setWindowTitle("Which conditions?")
        layout = QVBoxLayout(dialog)
        listing = QListWidget()
        listing.addItems(ready)
        listing.setSelectionMode(QAbstractItemView.MultiSelection)
        # THE ONE ON SCREEN IS PRESELECTED, because "no, not all of them" most often
        # means "just this one" and that is the condition being looked at.
        here = self._root.name if self._root is not None else None
        for row in range(listing.count()):
            if listing.item(row).text() == here:
                listing.item(row).setSelected(True)
        layout.addWidget(listing)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if not dialog.exec():
            return None
        chosen = [item.text() for item in listing.selectedItems()]
        return chosen or None

    def _conditions_ready_to_export(self):
        """``(ready, skipped)``: everything saving needs, plus a free `tif/`.

        The overwrite rule is per condition and it matters more here than anywhere: two
        crops in one `tif/` cannot be told apart afterwards, and this is the step that
        writes them.
        """
        ready, skipped = self._conditions_ready_to_save()
        if self._overwrite.isChecked():
            return ready, skipped
        keep = []
        for name in ready:
            existing = self._conditions[name] / "tif"
            if existing.is_dir() and any(existing.iterdir()):
                skipped[name] = "tif/ already holds stacks; tick overwrite to replace"
            else:
                keep.append(name)
        return keep, skipped

    def _export_stacks_clicked(self):
        """Write the cropped stacks HERE, with the settings already on this page.

        It used to open a second window, and that window asked again for the folder, the
        ranges, the pixel size and the frame interval, every one of which had just been
        decided here. Two UIs for one job is how the two come to disagree, and
        `crops.json` exists precisely so the range is answered once. The export is a
        batch job, not a stage: it needs a progress bar and a log, both of which this
        window already has.
        """
        from .movieprep import MoviePrepError

        # THE QUESTION FIRST, when there is more than one condition. This writes
        # gigabytes, unlike saving the ranges, so it asks rather than assuming either way.
        # THE MARKER IS "AM I INSIDE A QUEUED EXPORT", not "is the queue non-empty".
        # `_start_next_export` pops the LAST condition before calling this, so an
        # emptiness test saw no queue, asked again, and started another one: infinite
        # recursion that exported the first conditions and never reached the last.
        if (len(self._conditions) > 1
                and getattr(self, "_exporting_condition", None) is None):
            chosen = self._pick_export_conditions()
            if not chosen:
                return
            self._stash_condition()
            self._export_queue = list(chosen)
            self._export_return_to = self._root.name if self._root is not None else None
            self._log.appendPlainText(
                f"export: {len(chosen)} condition(s): " + ", ".join(chosen))
            self._start_next_export()
            return

        folder, _why = self._acquisition_root()
        if folder is None:
            self._crop_status.setText("no folder to export from")
            return
        if not self._pixel_nm.value() or not self._interval_ms.value():
            self._crop_status.setText(
                "set the pixel size and frame interval first; they are saved in "
                "the stack's calibration and have no default")
            return
        crops = self._crop_panel.crops()
        missing = self._crop_panel.missing()
        if missing or not crops:
            self._crop_status.setText(
                f"{len(missing)} movie(s) have no range yet: "
                + ", ".join(missing[:5]) + (" …" if len(missing) > 5 else ""))
            return
        # AN EXISTING tif/ IS NOT OVERWRITTEN SILENTLY, the same rule the tracking run
        # follows: two crops in one folder cannot be told apart afterwards.
        existing = self._root / "tif"
        if existing.is_dir() and any(existing.iterdir()) and not self._overwrite.isChecked():
            self._crop_status.setText(
                f"{existing} already holds stacks. Tick overwrite to replace them: two "
                f"crops in one folder cannot be told apart afterwards.")
            return

        self._export_stacks.setEnabled(False)
        self._save_crops.setEnabled(False)
        self._export_progress.setRange(0, len(crops))
        self._export_progress.setValue(0)
        self._export_progress.setFormat("exporting %v of %m")
        self._export_progress.show()
        self._export_cancel.show()
        self._crop_status.setText(f"writing {len(crops)} stack(s) to {self._root}…")
        self._log.appendPlainText(f"export: {len(crops)} movie(s) -> {self._root}")

        job = _ExportJob(folder, self._root, crops,
                         pixel_size_nm=float(self._pixel_nm.value()),
                         frame_interval_s=float(self._interval_ms.value()) / 1000.0,
                         projection=self.projection_kind())
        job.signals.progress.connect(self._export_progress_update)
        job.signals.finished.connect(self._export_finished)
        job.signals.failed.connect(self._export_failed)
        self._export_job = job
        self._pool.start(job)

    def _export_progress_update(self, index, total, note):
        self._export_progress.setRange(0, total)
        self._export_progress.setValue(index)
        self._log.appendPlainText(f"  {index}/{total} {note}")

    def _start_next_export(self):
        """Export one condition from the queue, as the save queue does.

        Same shape and the same reason: each export is a background job, so a loop would
        either block the window or start every condition at once through one panel.
        """
        if not self._export_queue:
            return False
        name = self._export_queue.pop(0)
        folder = self._conditions[name]
        acq = folder / "raw" if (folder / "raw").is_dir() else folder
        self._adopting = True
        try:
            self._crop_panel.adopt(
                acq, output_root=folder,
                curves=self._session_curves.get(name),
                crops=self._session_crops.get(name),
                errors=self._session_errors.get(name))
        finally:
            self._adopting = False
        self._root = folder
        self._apply_condition_calibration(name)
        self._exporting_condition = name
        self._export_stacks_clicked()
        return True

    def _export_finished(self, table):
        self._export_done()
        ok = int(table["ok"].sum()) if "ok" in table else len(table)
        where = getattr(self, "_exporting_condition", None)
        self._crop_status.setText(
            f"{ok} of {len(table)} stack(s) written to {self._root / 'tif'}"
            + (f" for {where}" if where else "")
            + ("" if ok == len(table) else ": failures are in the log"))
        self._scan_movies()
        if self._export_queue:
            self._start_next_export()
            return
        if getattr(self, "_export_return_to", None):
            back, self._export_return_to = self._export_return_to, None
            self._conditions_box.setCurrentText(back)
        self._exporting_condition = None

    def _export_failed(self, message):
        self._export_done()
        where = getattr(self, "_exporting_condition", None)
        self._crop_status.setText(message)
        self._log.appendPlainText(
            f"export{f' ({where})' if where else ''}: {message}")
        self._scan_movies()
        # A FAILED CONDITION DOES NOT END THE RUN. The others are independent folders,
        # and stopping would leave a queue half done with nothing saying where it got to.
        if self._export_queue:
            self._start_next_export()

    def _export_done(self):
        self._export_progress.hide()
        self._export_cancel.hide()
        self._export_job = None
        self._export_stacks.setEnabled(True)
        self._save_crops.setEnabled(True)

    def _cancel_export(self):
        job = getattr(self, "_export_job", None)
        if job is not None:
            job.cancel()
            self._crop_status.setText("cancelling after the current movie…")

    def _explain_threshold(self) -> None:
        """Say why the threshold box is empty, in the place it is empty.

        Every branch names the next action rather than the failure, because each of
        these is an ordinary stage of the work and not an error.
        """
        from .presets import ADAPTIVE

        if not hasattr(self, "_threshold_note"):
            return
        if self._threshold.value() > 0:
            self._threshold_note.setText("")
            return
        if self._preset.currentText() != ADAPTIVE:
            self._threshold_note.setText(
                "This preset carries its own threshold. Choose 'Adaptive' to measure "
                "one from these movies' noise, or type a value.")
        elif not self._movies and self._source == SOURCE_RAW:
            self._threshold_note.setText(
                "Nothing to measure yet: the threshold is k × the LoG noise inside each "
                "crop, so it needs the crop ranges first. Set a range for every movie on "
                "the previous page, then this fills itself.")
        elif not self._movies:
            self._threshold_note.setText(
                "No movies loaded, so there is no noise to measure. Choose a folder, or "
                "switch the source if these are prepared stacks.")
        else:
            self._threshold_note.setText(
                "Press 'Compute from movies' to measure it from this condition's own "
                "noise.")

    def _compute_threshold_clicked(self, *, owner=None):
        """Fill the threshold from the movies' own noise, and say what it came from.

        The spread is reported, not just the number. `run_batch` takes one settings
        object for the whole batch, so this is a median - and a median hides exactly the
        case that matters. One lab project spans a 2.3x noise range because it holds three
        sessions, one twice as noisy as the others, and the shipped preset gives all 29
        movies the number matched on the noisy one. When the batch looks mixed, this
        says so rather than quietly averaging over it.

        OFF THE GUI THREAD (`_NoiseJob`), ONE AT A TIME. A request arriving while one
        runs is QUEUED, so the walk's next condition is measured after this one instead
        of being dropped or run at the same time against the same files.
        """
        # TWO DIFFERENT QUESTIONS. `owner` says WHERE THE MOVIES COME FROM and is only
        # set when the walk scheduled this, because a deferred pass must measure the
        # condition that finished rather than the one now on screen. `credit` says WHO
        # THE ANSWER BELONGS TO and is captured here either way. Deriving the source from
        # the credit sent a pressed button to the session record instead of the panel in
        # front of the person pressing it.
        credit = owner or (self._root.name if self._root is not None else None)
        if self._noise_job is not None:
            if (owner, credit) not in self._noise_queue:
                self._noise_queue.append((owner, credit))
            return
        self._start_noise(owner, credit)

    def _start_noise(self, owner, credit) -> None:
        movies, crops = self._measurable_movies(owner)
        if not movies:
            self._log.appendPlainText(
                "nothing to measure the noise on yet: every movie needs a crop range "
                "first, because the threshold is measured inside the crop")
            self._autocomputed_for = None
            self._noise_next()
            return
        self._compute_threshold.setEnabled(False)
        self._compute_threshold.setText("Measuring…")
        self._log.appendPlainText(
            f"measuring LoG noise on {len(movies)} movie(s) at radius "
            f"{self._radius.value():.1f} px ...")
        # RUNNING OR STUCK IS A QUESTION THE WINDOW MUST ANSWER. Ten movies is tens of
        # seconds, and silence and a hang look identical. `batch_threshold` has taken an
        # `on_progress` callback since it was written; the job forwards it.
        self._progress.setVisible(True)
        self._progress.setRange(0, len(movies))
        self._progress.setValue(0)
        self._progress.setFormat("%v / %m movies measured")
        job = _NoiseJob(movies, radius_px=float(self._radius.value()),
                        crops=crops or None, credit=credit)
        job.signals.progress.connect(self._noise_progress)
        job.signals.finished.connect(self._noise_measured)
        job.signals.failed.connect(self._noise_failed)
        self._noise_job = job
        self._pool.start(job)

    def _noise_progress(self, index, total, row) -> None:
        self._progress.setValue(index)
        if row["ok"]:
            self._log.appendPlainText(
                f"  {index}/{total} {row['movie']}: LoG noise "
                f"{row['sigma_log']:.1f} → {row['threshold']:.1f}")
        else:
            self._log.appendPlainText(
                f"  {index}/{total} {row['movie']}: {row['error']}")

    def _noise_idle(self) -> None:
        """Put the button and the bar back, whatever the outcome."""
        self._compute_threshold.setEnabled(True)
        self._compute_threshold.setText("Compute from movies")
        self._progress.setVisible(False)
        self._progress.setFormat("%p%")

    def _noise_next(self) -> None:
        """Start the next queued measurement, or close a "measure all" press."""
        if self._noise_queue:
            owner, credit = self._noise_queue.pop(0)
            self._start_noise(owner, credit)
        elif self._noise_job is None and self._noise_batch is not None:
            self._noise_batch_done()

    def _noise_failed(self, message: str) -> None:
        self._noise_job = None
        self._noise_idle()
        self._log.appendPlainText(f"could not compute a threshold: {message}")
        # THE CLAIM IS RELEASED ON FAILURE. `_autocompute_threshold` marks the key
        # before the work starts, so without this a measurement that raised was
        # never retried: the window would sit with an empty threshold and a line
        # in the log, and every later scan would decide it had already measured.
        self._autocomputed_for = None
        self._noise_next()

    def _noise_measured(self, got) -> None:
        job, self._noise_job = self._noise_job, None
        credit = None if job is None else job.credit
        self._noise_idle()
        here = self._root.name if self._root is not None else None
        # THE BOX SHOWS THE CONDITION ON SCREEN. A measurement credited to another
        # condition (the walk moves on while one runs) is filed under that condition
        # and must not overwrite what is displayed for this one.
        if credit is None or credit == here:
            self._threshold.setValue(got["threshold"])
        # EACH MOVIE'S OWN, kept beside the median. `batch_threshold` already computes
        # them and then reduces to a median; nothing was measuring twice, the values
        # were simply thrown away.
        if credit:
            self._movie_thresholds[credit] = {
                row["movie"]: float(row["threshold"])
                for row in got.get("per_movie", ()) if row.get("ok")}
            self._typed_thresholds.discard(credit)
        if credit is not None:
            self._thresholds[credit] = float(got["threshold"])
            self._tracking.setdefault(credit, {})["threshold"] = float(got["threshold"])
            self._refresh_tracking_table()
            self._refresh_progress()
            if self._noise_batch is not None:
                self._noise_batch.append(credit)
        if credit is not None and here is not None and credit != here:
            self._log.appendPlainText(
                f"  (that threshold belongs to {credit}; the window has moved on to "
                f"{here})")
        # WHERE THE NUMBER CAME FROM TRAVELS WITH IT, into any preset saved afterwards.
        # A threshold with no provenance cannot be told apart from one somebody typed.
        if credit is None or credit == here:
            self._threshold_provenance = (
                f"computed from {got['n_movies']} movie(s): {got['k']:g} × median LoG "
                f"noise {got['sigma_median']:.1f} at radius {got['radius_px']:g} px")
        self._log.appendPlainText(
            f"threshold {got['threshold']:.1f} = {got['k']} x the median LoG noise "
            f"({got['sigma_median']:.1f}), over {got['n_movies']} movie(s)")
        if got["n_failed"]:
            failed = [r for r in got["per_movie"] if not r["ok"]]
            self._log.appendPlainText(
                f"  {got['n_failed']} movie(s) could not be measured: "
                + "; ".join(f"{r['movie']} ({r['error']})" for r in failed[:3]))
        if got["mixed_sessions"]:
            from .chart_text import fold_words

            self._log.appendPlainText(
                f"  Warning: the noise spans {got['sigma_min']:.1f} to "
                f"{got['sigma_max']:.1f} ({fold_words(got['sigma_ratio'], 2)}) across these "
                f"movies, which is more than one session's spread. One threshold for "
                f"the batch is a compromise here: consider tracking each session "
                f"separately, each with its own computed value.")
            # WHICH MOVIE, not just how wide the span is. A min/max ratio can be set by
            # one very quiet movie on its own and says nothing about which to look at,
            # so the movies are NAMED. Flagged means both unlike its neighbours and far
            # enough from them to matter -- the statistical test alone fires on 2 per
            # cent of ordinary variation, measured over eight real conditions.
            try:
                from .trackmate_run import noise_outliers

                odd = noise_outliers(got["per_movie"])
            except Exception:                             # noqa: BLE001 - advisory
                odd = None
            if odd is not None and odd.attrs.get("n_flagged"):
                for _index, row in odd[odd["noise_flagged"]].iterrows():
                    self._log.appendPlainText(
                        f"    {row['movie']}: noise {row['sigma_log']:.1f} "
                        f"({fold_words(row['sigma_ratio_to_median'], 2, of=True)} the median): look at "
                        f"this one; nothing is excluded")
        self._refresh_stamp()
        self._noise_next()

    def closeEvent(self, event):                          # noqa: N802 - Qt's name
        """Stop a threshold measurement after the movie in hand; drop the queue."""
        job = getattr(self, "_noise_job", None)
        if job is not None:
            job.cancelled = True
        self._noise_queue = []
        super().closeEvent(event)

    def _refresh_stamp(self):
        settings = self.current_settings()
        self._stamp.setText(settings.stamp() if settings is not None
                            else "these numbers do not make a valid setting")
        self._optics_note.setText(self.optics_note())

        minutes = len(self._movies) * SECONDS_PER_MOVIE / 60.0
        self._estimate.setText(
            f"{len(self._movies)} movies × about {SECONDS_PER_MOVIE:.0f} s "
            f"≈ {minutes:.0f} min" if self._movies else "nothing to track")
        # REMEMBERED, not just applied. `_refresh_run_buttons` re-enables the run
        # controls when the queue empties, and it must not hand back a button that was
        # off for a reason -- no movies, no Fiji, no threshold.
        self._run_button_ready = (
            bool(self._movies) and settings is not None and self._install is not None)
        self._refresh_run_buttons()
        # WHY RUN IS OFF, where the user is looking. A disabled button with no reason
        # sends people to the log, and the commonest case here, adaptive selected, no
        # movies yet, so nothing to measure the noise on: is not an error at all.
        if not self._movies:
            why = "no movies loaded yet"
        elif self._install is None:
            why = "Fiji not found"
        elif settings is None:
            why = ("the threshold is not measured: press 'Compute from movies', or "
                   "type one")
        else:
            why = ""
        self._run_button.setToolTip(why or "run the batch")

        # THE COUNT TRAVELS TO BOTH PAGES. Page two decides how to track; without the
        # count in front of it, "Track this condition" is greyed out with the reason on
        # a screen the user has left.
        count = (f"{len(self._movies)} movie(s) ready"
                 if self._movies else "no movie is ready to track yet")
        self._movie_count.setText(count)
        where = self._root.name if self._root is not None else "no folder chosen"
        self._track_header.setText(f"<b>{where}</b>: {count}")

    # -- run ------------------------------------------------------------------
    def _run_inputs(self, name):
        """Everything one condition needs to be tracked, or ``(None, reason)``.

        RESOLVED FROM THE CONDITION, never from what is on screen. `_run` used to read
        `self._root`, `self._movies`, `self._crops` and the settings boxes, all of which
        follow the panel -- so tracking a second condition meant moving the view onto it,
        which is the coupling this window has had removed twice already (the read walk
        and the save queue).

        THE RANGES COME FROM `crops.json`, NOT FROM THE WINDOW. That file is the single
        record of what was tracked, so a run against ranges existing only here would be
        unreproducible: the record would say one thing and the tracks another. A
        condition whose ranges are not saved is REFUSED by name, with what to press.
        """
        from .movieprep import MoviePrepError, find_crops, load_crops
        from .rawmovie import discover_raw_movies
        from .trackmate_run import TrackMateRunError, TrackMateSettings

        folder = (self._conditions.get(name) if (self._conditions and name)
                  else self._root)
        if folder is None:
            return None, "no folder chosen"
        acq = folder / "raw" if (folder / "raw").is_dir() else folder
        try:
            movies, _companions = discover_raw_movies(acq)
        except Exception as exc:                          # noqa: BLE001 - named below
            return None, f"cannot list {acq}: {type(exc).__name__}: {exc}"
        if not movies:
            return None, "no acquisitions"

        try:
            found = find_crops(acq, folder)
            payload = load_crops(found) if found else None
        except (MoviePrepError, OSError) as exc:
            return None, f"crops.json unreadable ({exc})"
        if not payload:
            return None, ("no saved crop ranges; press 'Save crop ranges' first "
                          "(crops.json records what was tracked)")
        crops = {k: tuple(v) for k, v in payload.get("crops", {}).items()}
        from .rawmovie import movie_name

        absent = [movie_name(m) for m in movies if movie_name(m) not in crops]
        if absent:
            return None, (f"{len(absent)} movie(s) have no saved range: "
                          + ", ".join(absent[:4]) + (" ..." if len(absent) > 4 else ""))

        values = dict(self._tracking.get(name) or {}) if name else {}

        def pick(key, box, cast=float):
            got = values.get(key)
            return cast(box.value() if got is None else got)

        try:
            link = pick("linking_max_distance_px", self._link)
            settings = TrackMateSettings(
                radius_px=pick("radius_px", self._radius),
                threshold=pick("threshold", self._threshold),
                max_frame_gap=pick("max_frame_gap", self._gap, int),
                # BOTH linking distances move together, as in `current_settings`:
                # TrackMate takes them separately and u-track has one search radius, so
                # letting them drift apart would create a policy u-track has no
                # equivalent of, silently.
                linking_max_distance_px=link,
                gap_closing_max_distance_px=link)
        except (TrackMateRunError, TypeError, ValueError) as exc:
            return None, f"settings do not make a run ({exc})"
        if not settings.threshold:
            return None, ("no detection threshold; measure it, or type one into the "
                          "settings grid")
        return (list(movies), crops, settings, folder / "trackmate"), ""

    def _thresholds_for(self, name):
        """``{movie: threshold}`` for one condition, or ``None`` for one shared value.

        None where the threshold was TYPED, or where nothing has been measured: both
        mean "one number for every movie", which is `run_batch`'s behaviour with no
        override and is what makes this an override rather than a second way to
        configure a run.
        """
        if name in self._typed_thresholds:
            return None
        table = self._movie_thresholds.get(name)
        return dict(table) if table else None

    def _conditions_ready_to_track(self):
        """``(ready, {name: reason})``. A condition that cannot run is NAMED.

        A silent omission here reads as "tracked and found nothing", which is the one
        thing a tracking run must never be confused with.
        """
        ready, skipped = [], {}
        for name in (self._conditions or {}):
            inputs, why = self._run_inputs(name)
            if inputs is None:
                skipped[name] = why
            else:
                ready.append(name)
        return ready, skipped

    def _run_all_clicked(self):
        """Track every condition in turn, stopping when all of them are done.

        A QUEUE RATHER THAN A LOOP, for the reason the save queue is one: each run is a
        background job of roughly 30 s per movie, so looping would either freeze the
        window or start every condition at once against a single Fiji install.
        """
        if not self._conditions:
            return
        ready, skipped = self._conditions_ready_to_track()
        if skipped:
            self._log.appendPlainText(
                "not tracked: " + "; ".join(f"{k} ({v})" for k, v in skipped.items()))
        if not ready:
            self._log.appendPlainText(
                "no condition is ready to track; see the reasons above")
            return
        movies = 0
        for name in ready:
            inputs, _why = self._run_inputs(name)
            movies += len(inputs[0])
        self._run_queue = list(ready)
        self._run_total = len(ready)
        # THE COST BEFORE THE RUN, as the single-condition estimate already does. Thirty
        # conditions is hours, and a window that simply goes busy teaches people to kill
        # it.
        self._log.appendPlainText(
            f"=== tracking {len(ready)} condition(s), {movies} movie(s) in all, "
            f"about {movies * 30 / 60:.0f} min: " + ", ".join(ready))
        self._start_next_run()

    def _start_next_run(self) -> bool:
        """Track one condition from the queue; the finish handler takes the next."""
        while getattr(self, "_run_queue", None):
            name = self._run_queue.pop(0)
            index = getattr(self, "_run_total", 1) - len(self._run_queue)
            self._log.appendPlainText("")
            self._log.appendPlainText(
                f"--- {name} ({index} of {getattr(self, '_run_total', 1)})")
            if self._run_condition(name):
                return True
            # NOT A DEAD END. A condition that cannot start must not strand the rest of
            # the queue, so the loop moves on rather than the run simply stopping.
        self._running_condition = None
        self._refresh_run_buttons()
        return False

    def _run_condition(self, name) -> bool:
        """Start the batch for one condition. ``False`` if it could not start."""
        inputs, why = self._run_inputs(name)
        if inputs is None:
            self._log.appendPlainText(f"{name or 'this condition'}: {why}")
            return False
        movies, crops, settings, output_dir = inputs
        if (output_dir.exists() and any(output_dir.glob("*.xml"))
                and not self._overwrite.isChecked()):
            self._log.appendPlainText(
                f"{output_dir} already holds XMLs from an earlier run. Tick overwrite, "
                "or move them: two settings in one folder cannot be told apart later.")
            return False
        if self._install is None:
            self._log.appendPlainText("Fiji is not set up yet")
            return False

        self._running_condition = name
        self._run_button.setEnabled(False)
        self._run_all_button.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setRange(0, len(movies))
        self._progress.setValue(0)
        self._log.appendPlainText(f"--- {settings.stamp()}")
        self._log.appendPlainText(f"--- {len(movies)} movies -> {output_dir}")
        # WRITTEN BEFORE THE RUN, not after. These are the run's INPUTS, and a run that
        # dies half way is exactly when knowing what it was given matters most.
        per_movie = self._thresholds_for(name)
        if per_movie:
            values = sorted(per_movie.values())
            self._log.appendPlainText(
                f"--- each movie at its own threshold: {values[0]:.0f} to "
                f"{values[-1]:.0f} over {len(values)} movie(s)")
        self._write_run_record(output_dir, settings, crops, per_movie=per_movie)
        job = _BatchJob(movies, output_dir, settings, self._install, crops=crops,
                        thresholds=per_movie)
        job.signals.progress.connect(self._on_progress)
        job.signals.finished.connect(self._on_finished)
        job.signals.failed.connect(self._on_failed)
        self._pool.start(job)
        return True

    def _refresh_run_buttons(self) -> None:
        """Re-enable the run controls when nothing is in flight."""
        busy = bool(getattr(self, "_running_condition", None)
                    or getattr(self, "_run_queue", None))
        self._run_button.setEnabled(not busy and self._run_button_ready)
        self._run_all_button.setEnabled(not busy and bool(self._conditions))

    def _refresh_run_conditions(self) -> None:
        """Keep the run picker in step with the loaded conditions.

        Hidden entirely for a single condition, where "this condition" is unambiguous
        and a one-entry dropdown is a control that cannot be wrong and cannot help.

        The SELECTION FOLLOWS THE FOLDER ON SCREEN when the picker is rebuilt, so
        pressing Track after clicking a condition row tracks the one just clicked -- but
        it stays a separate control, because this page does not show those rows and a
        run must name its own target.
        """
        names = list(self._conditions or {})
        several = len(names) > 1
        self._run_condition_box.setVisible(several)
        self._run_all_button.setVisible(several)
        if not several:
            self._run_condition_box.blockSignals(True)
            self._run_condition_box.clear()
            self._run_condition_box.blockSignals(False)
            self._refresh_run_buttons()
            return
        current = self._run_condition_box.currentText()
        if [self._run_condition_box.itemText(i)
                for i in range(self._run_condition_box.count())] != names:
            self._run_condition_box.blockSignals(True)
            self._run_condition_box.clear()
            self._run_condition_box.addItems(names)
            self._run_condition_box.blockSignals(False)
            here = self._root.name if self._root is not None else None
            pick = current if current in names else (here if here in names else names[0])
            self._run_condition_box.setCurrentText(pick)
        self._run_all_button.setText(f"Track all {len(names)} conditions")
        self._refresh_run_buttons()

    def _picked_run_condition(self):
        """Which condition the single-condition button would track.

        The picker when there is one, the folder on screen otherwise. `None` means a
        project with no conditions, where `_run_inputs` falls back to `_root`.
        """
        if self._conditions and self._run_condition_box.count():
            return self._run_condition_box.currentText()
        return None

    def _run(self):
        """Track the condition in the picker, with that condition's own settings."""
        self._run_queue = []
        self._run_total = 1
        self._run_condition(self._picked_run_condition())

    def _done_clicked(self):
        """Finish tracking and go back to the stage menu, not out of the tool.

        Tracking is a middle step: what follows is segmentation or analysis, and closing
        the window used to end the process, so the next step meant starting `smtracker`
        again and re-picking the folder.

        Closing the window another way still exits, and the difference is deliberate: the
        title-bar X means "I am done with the tool", the button means "I am done with
        this stage". A window that reopened a menu however it was closed could not be
        left.
        """
        self.return_to_menu = True
        self.close()

    def _write_run_record(self, output_dir, settings, crops, outcome=None,
                          per_movie=None):
        """Record what this run was given, beside what it produces.

        The XMLs carry TrackMate's settings but not the crop range, and not the pixel
        size (the script forces 1 px = 1 unit so a calibrated TIFF cannot reinterpret
        every radius). Without this, a run cannot be repeated: measured on a real one.
        """
        from .trackmate_run import record_run

        try:
            path = record_run(
                output_dir, settings=settings, movies=self._movies, crops=crops,
                pixel_size_nm=self._pixel_nm.value() or None,
                frame_interval_s=(self._interval_ms.value() / 1000.0
                                  if self._interval_ms.value() else None),
                source=self._source,
                threshold_provenance=("adaptive_per_movie" if per_movie else
                                      getattr(self, "_threshold_provenance", None)),
                per_movie_thresholds=per_movie,
                outcome=outcome)
        except Exception as exc:                       # noqa: BLE001 - named, not hidden
            self._log.appendPlainText(
                f"could not write the run record: {type(exc).__name__}: {exc}")
            return
        if outcome is None:
            self._log.appendPlainText(f"--- run record: {path}")

    def _on_progress(self, index, total, note):
        self._progress.setValue(index)
        self._log.appendPlainText(f"[{index}/{total}] {note}")

    def _on_finished(self, payload):
        from .trackmate_run import describe_batch, summarise_run

        table, qc = payload
        self._log.appendPlainText("")
        self._log.appendPlainText(describe_batch(qc))
        self._log.appendPlainText("")

        # THE RESULT, NOT JUST THE FAILURES. `describe_batch` reports what went wrong;
        # this reports what came out, and above all the USABLE track count, since a run
        # can report thousands of tracks while leaving a quarter of them long enough for
        # the analysis to use.
        self._log.appendPlainText(self._summarise())
        self._log.appendPlainText("")
        self._log.appendPlainText(
            "Tracks are XMLs in trackmate/. Open the analysis stage on this folder to "
            "read them.")
        self._progress.setVisible(False)
        # THE CONDITION THAT RAN, not the one on screen. The queue never moves the view,
        # so `self._root` is wherever the user left it -- writing the record against that
        # would file one condition's outcome under another's name, the defect this window
        # has already had in its save queue and its filing.
        done = getattr(self, "_running_condition", None)
        folder = (self._conditions.get(done) if (self._conditions and done)
                  else self._root)
        inputs, _why = self._run_inputs(done)
        self._write_run_record(
            folder / "trackmate",
            inputs[2] if inputs else self.current_settings(),
            inputs[1] if inputs else None,
            per_movie=self._thresholds_for(done),
            outcome={"movies": int(len(table)),
                     "failed": int((~table["ok"]).sum()) if "ok" in table else 0,
                     "flagged": int(qc["flagged"].sum()) if "flagged" in qc else 0})
        self._last_batch = (table, qc)                  # kept for a test to read
        self._running_condition = None
        if self._start_next_run():
            return
        self._refresh_run_buttons()
        self._done_button.setVisible(True)
        self._done_button.setDefault(True)
        self._done_button.setFocus()

    def _summarise(self):
        """Read the XMLs back and report what the run produced.

        Read back rather than trusted. The batch table counts what TrackMate SAID it
        found; the usable-track count and the step distribution can only come from the
        files themselves, and those are the numbers that decide whether the run is worth
        analysing.
        """
        from .trackmate_run import summarise_run

        table = self._last_batch[0] if self._last_batch else None
        picked, _why = self._run_inputs(getattr(self, "_running_condition", None))
        settings = picked[2] if picked else self.current_settings()
        if settings is None:
            return "the settings differ from the run, so there is no summary"
        try:
            from .trackmate import read_trackmate_xml

            import pandas as pd

            done = getattr(self, "_running_condition", None)
            folder = (self._conditions.get(done) if (self._conditions and done)
                      else self._root)
            parts = []
            for xml in sorted((folder / "trackmate").glob("*.xml")):
                one = read_trackmate_xml(xml)
                from .rawmovie import movie_name

                parts.append(one.assign(movie=movie_name(xml)))
                QApplication.processEvents()
            raw = pd.concat(parts, ignore_index=True) if parts else None
            return summarise_run(raw, settings, table=table)
        except Exception as exc:                       # noqa: BLE001 - named on screen
            return (f"tracked, but the summary could not be built: "
                    f"{type(exc).__name__}: {exc}")

    def _on_failed(self, message):
        # NAMED, AND THE QUEUE GOES ON. One condition failing must not silently end a
        # run of eight: the others are independent, and stopping would leave them
        # untracked with nothing saying they were skipped rather than empty.
        who = getattr(self, "_running_condition", None)
        self._log.appendPlainText(
            f"BATCH FAILED{f' for {who}' if who else ''}: {message}")
        self._progress.setVisible(False)
        self._running_condition = None
        if self._start_next_run():
            return
        self._refresh_run_buttons()


def launch_tracking(root, *, show=True, block=True):
    """Open the tracking window on ``root``. Returns 0, like the other entry points."""
    # `block` DECIDES WHETHER TO RUN THE LOOP, AND IT IS THE CALLER'S TO KNOW.
    #
    # Opening this window from another one, the tracking window's "Export cropped
    # stacks": must NOT call `exec`: Qt answers "The event loop is already running"
    # and the window never works. The first fix for that guessed, running the loop only
    # when this call had CREATED the QApplication. That guess was wrong, because the
    # LAUNCHER creates it before either window opens: from `smtracker` the application
    # already existed, so this returned the window instead of running, and `main`
    # printed its repr at the shell. Creating the application and running its loop are
    # simply different things, and no public Qt call reports whether a loop is running.
    #
    # So the caller says. It always knows: the CLI wants a blocking window, a button
    # inside a live window does not.
    app = QApplication.instance() or QApplication([])
    from .appicon import apply as _apply_icon
    _apply_icon()
    window = TrackingWindow(root)
    if not show:
        return window
    # THROUGH THE DOORS' ONE RULE (tester B25, 3 Nov 2026: it was `window.show()`), so it never opens past a laptop's
    # screen. Its colours stay light in a dark session: they are written into its labels, not read from the theme.
    from .ui.widgets.screenfit import show_fitted

    show_fitted(window)
    if not block:
        # AND THE WINDOW IS RETURNED so the caller can hold it. Returning 0 left the
        # caller storing an integer while Python collected the window out from under
        # it, a window that opens and vanishes.
        return window
    app.exec_() if hasattr(app, "exec_") else app.exec()
    # BACK TO THE MENU IS NOT AN EXIT CODE, so it is a sentinel rather than a number:
    # `main` must be able to tell it from the 0 that means "the user closed the tool",
    # and a second integer would be indistinguishable from a status.
    return RETURN_TO_MENU if window.return_to_menu else 0
