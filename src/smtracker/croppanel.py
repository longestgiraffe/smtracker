"""Choosing the crop range, as a widget the tracking window owns.

Why this is not its own stage any more. Tracking from the raw acquisition needs exactly
one thing from preparation (the frame range) and nothing else in the pipeline consumes
the cropped stacks: the analysis needs tracks plus cell outlines, and the viewer can use
the phase image. Writing ~5 GB of TIFFs was only ever required to feed u-track, which
cannot read a `.nd2`. So the crop belongs where it is used, and preparing stacks is an
export for the u-track route rather than a stage to pass through.

`crops.json` stays the single record of the range. Setting it in one place is what stops
the two halves disagreeing, which is invisible downstream: two datasets, one name.

THIN QT, as `widgets` and `tracking_gui` are. Every number comes from `movieprep`
(`movie_curves`, `propose_crop`) and every line from `plots.draw_crop_curves`, so a wrong
crop fails in `test_movieprep.py` rather than by looking plausible on screen.

The curves are precomputed. Measured on a 4500-frame `.nd2`: 7.7 s per movie, which was
paid on every click. A background pass does all of them once while the first is being
read, so selecting a movie is instant afterwards.
"""
from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

__all__ = ["CropPanel"]




def _movie_name(path) -> str:
    """The movie a raw file IS (`rawmovie.movie_name`): a MetaMorph `series001_YFPLaser`
    is `series001` where the folder's tifs, phase and outlines already say so."""
    from .rawmovie import movie_name

    return movie_name(path)

class _Signals(QObject):
    one = Signal(str, object, str, int, str)   # movie, curves|None, error, gen, token
    done = Signal(int, str)                    # generation, token


class _CurveJob(QRunnable):
    """Compute every movie's curves once, off the UI thread."""

    def __init__(self, movies, generation: int = 0, token: str = ""):
        super().__init__()
        self._movies = list(movies)
        #: WHICH FOLDER THIS PASS BELONGS TO. `set_folder` starts a new pass and used to
        #: leave the old one running, with no way to tell its results apart: a movie from
        #: the condition you just left arrived into the condition you just opened, so the
        #: counts exceeded the folder's own movie count (13 read of 10) and, worse, an
        #: auto-proposed crop could be recorded against another condition's stem and
        #: saved that way. The receiver drops anything not from the current generation.
        self._generation = int(generation)
        #: WHOSE pass this is. A switch used to abandon a read: the generation moved, the
        #: results were dropped, and minutes of work went with them. Naming the owner
        #: lets the results be FILED where they belong while somebody looks elsewhere.
        self._token = str(token)
        self.signals = _Signals()

    def run(self):
        from .movieprep import movie_curves
        from .rawmovie import movie_name, read_raw_movie

        for path in self._movies:
            name = movie_name(path)
            try:
                curves = movie_curves(read_raw_movie(path), name=name)
            except Exception as exc:            # noqa: BLE001 - kept and named
                if not self._emit(name, None, f"{type(exc).__name__}: {exc}"):
                    return
            else:
                if not self._emit(name, curves, ""):
                    return
        try:
            self.signals.done.emit(self._generation, self._token)
        except RuntimeError:
            pass

    def _emit(self, stem, curves, error) -> bool:
        """Report one movie. ``False`` means the receiver is gone: stop reading.

        Closing the window is not an error, but it deletes the signal source under a
        worker that is still running, and the resulting `RuntimeError` is raised on a
        background thread where no caller can catch it: it just prints a traceback
        under a window that is no longer there. So this is the one place a swallowed
        exception is right, and it is narrow: only `RuntimeError` from the emit itself,
        and it STOPS the pass rather than continuing to read movies nobody is waiting
        for.
        """
        try:
            self.signals.one.emit(stem, curves, error, self._generation, self._token)
        except RuntimeError:
            return False
        return True


from .movieprep import FLAT_TOLERANCE


class CropPanel(QWidget):
    """Pick a frame range per movie, and save them as ``crops.json``."""

    changed = Signal()
    #: True while the background pass still holds the acquisitions open. The window
    #: greys its save and export buttons on it: pressing them mid-pass wrote only the
    #: movies read SO FAR, which looks like a partial save nobody asked for.
    reading_changed = Signal(bool)
    #: ``(token, movie, MovieCurves|None, error)`` for every movie read, whether or not
    #: this panel is still showing the folder it came from.
    curve_arrived = Signal(str, str, object, str)
    #: "Cut all tails" was pressed: every movie of THIS folder now ends at this frame, and
    #: the window applies the same frame to the other conditions of the session.
    end_frame_applied = Signal(int)
    #: "Cut all starts" was pressed: the same, for the first frame. Each end is kept.
    start_frame_applied = Signal(int)
    #: ``token`` of a background pass that has just finished, displayed or not.
    pass_finished = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._movies: list[Path] = []
        self._curves: dict[str, object] = {}
        self._errors: dict[str, str] = {}
        self._crops: dict[str, tuple[int, int]] = {}
        self._proposal = None
        self._root: Path | None = None
        #: The rule in force for movies as they arrive, or None. Set by a press, which
        #: covers the movies in front of you, or by the tick box, which is a standing
        #: choice and therefore survives a folder change.
        self._auto = None
        self._auto_armed = False
        #: Which condition the current pass belongs to, carried on its results.
        self._token = ""
        #: Bumped by every `set_folder`; results tagged with an older one are dropped.
        self._generation = 0

        layout = QHBoxLayout(self)

        left = QVBoxLayout()
        self._list = QListWidget()
        self._list.currentTextChanged.connect(self._show)
        left.addWidget(self._list, 1)
        self._progress = QProgressBar()
        self._progress.setTextVisible(True)
        self._progress.hide()
        left.addWidget(self._progress)
        layout.addLayout(left)

        right = QVBoxLayout()
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self._figure = Figure(figsize=(5.5, 3.6), tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._figure)
        # DRAG THE EDGES ON THE CURVE. Typing a frame number means reading one off the
        # plot by eye and then transcribing it; grabbing the line is the same decision
        # without the transcription. The spin boxes stay, and stay authoritative: this
        # writes into them, so there is one place the crop lives either way.
        self._canvas.mpl_connect("button_press_event", self._drag_start)
        self._canvas.mpl_connect("motion_notify_event", self._drag_move)
        self._canvas.mpl_connect("button_release_event", self._drag_end)
        self._dragging = None
        right.addWidget(self._canvas, 1)

        box = QGroupBox("Crop")
        form = QFormLayout(box)
        self._start = QSpinBox()
        self._start.setRange(1, 10_000_000)
        self._end = QSpinBox()
        self._end.setRange(1, 10_000_000)
        for widget in (self._start, self._end):
            widget.valueChanged.connect(self._crop_edited)
        row = QHBoxLayout()
        row.addWidget(self._start)
        row.addWidget(QLabel("to"))
        row.addWidget(self._end)
        form.addRow("Frames", row)

        self._rule = QComboBox()
        from .movieprep import CROP_START_RULES

        self._rule.addItems(list(CROP_START_RULES))
        self._rule.setToolTip(
            "strictest takes the latest of the three floors, so every criterion is met. "
            "The others use one criterion on its own.")
        # "keep tail  1 spots/frame" said nothing about what the number DID. It is a
        # cut: frames after the density last reached this value are dropped, because
        # past that point the field is emptier than the background it sits on and those
        # frames add links without adding molecules.
        self._tail = QDoubleSpinBox()
        self._tail.setRange(0.0, 1000.0)
        self._tail.setDecimals(1)
        self._tail.setValue(0.0)
        self._tail.setSpecialValueText("keep whole tail")
        self._tail.setPrefix("cut tail below ")
        self._tail.setSuffix(" spots/frame")
        self._tail.setToolTip(
            "Drop the end of the movie once spot density falls below this. "
            "At 'keep whole tail' nothing is cut.")
        # THE FLATNESS TOLERANCE, because "a few per cent" is a judgement about these
        # images and not a constant of nature. 2 % is where the experimenter put it, ruled
        # on SPOT DENSITY at the resulting floor rather than on how much of the movie each
        # value keeps -- "the biggest problem is too much photo density". What that costs
        # is recorded on `movieprep.FLAT_TOLERANCE`, and is why this is a BOX and not a
        # constant: at 2 % the floor lands ON measured practice instead of below it.
        #
        # CHANGING IT COSTS NOTHING. The floor is recomputed from the curves already in
        # memory -- no movie is re-read -- so this is a slider over an answer, not a
        # re-measurement.
        self._flatness = QDoubleSpinBox()
        # DOWN TO 0.1 %, because a camera with a large offset needs it: on one user's
        # MetaMorph movies 0.5 % is where the floor reaches the ~500 frames asked for
        # (see `movieprep.OFFSET_WARN_SHARE`), and 0.5 was the bottom of the box.
        self._flatness.setRange(0.1, 25.0)
        self._flatness.setDecimals(1)
        self._flatness.setSingleStep(0.5)
        self._flatness.setValue(100.0 * FLAT_TOLERANCE)
        self._flatness.setPrefix("flat within ")
        self._flatness.setSuffix(" %")
        self._flatness.setToolTip(
            "How far the frame mean may wander inside a stretch still called flat, as a "
            "percentage of that stretch's own level. The floor is where the longest such "
            "stretch begins. Default 2 %; a looser value keeps more of the movie but "
            "starts at a higher spot density.")
        self._flatness.valueChanged.connect(self._flatness_changed)
        # ONE END FRAME FOR EVERY MOVIE, asked for as the more reliable tail cut: "set
        # frame X where all tails are cut from all movies". The density cut above lands
        # wherever each movie's noise puts it; this is one number, chosen on the curves.
        self._end_frame = QSpinBox()
        self._end_frame.setRange(0, 10_000_000)
        self._end_frame.setValue(0)
        self._end_frame.setSpecialValueText("no common end")
        self._end_frame.setPrefix("cut every movie at frame ")
        self._end_frame.setToolTip(
            "One last frame for every movie. Proposals use it instead of the density "
            "cut, and 'Cut all tails' applies it to every movie that already has a "
            "range, keeping each start. A movie shorter than this keeps its own last "
            "frame; one that starts after it is left alone and named.")
        # AND ONE START FRAME, asked for beside it: "we have a tail cut option for all
        # movies, but why no start cut option we can apply for all movies as well?" A lab
        # that crops every movie at one frame (one user's macro cut all of its at 1000)
        # gets that in one press instead of one edit per movie.
        self._start_frame = QSpinBox()
        self._start_frame.setRange(0, 10_000_000)
        self._start_frame.setValue(0)
        self._start_frame.setSpecialValueText("no common start")
        self._start_frame.setPrefix("start every movie at frame ")
        self._start_frame.setToolTip(
            "One first frame for every movie. Proposals use it instead of the start "
            "rule, and 'Cut all starts' applies it to every movie read so far, "
            "keeping each end. A movie that ends before this frame is left alone "
            "and named.")
        cut_starts = QPushButton("Cut all starts")
        cut_starts.setToolTip("Set the start of every movie to the frame on the left, in "
                              "every condition, keeping each movie's end.")
        cut_starts.clicked.connect(self._cut_all_starts)
        cut_all = QPushButton("Cut all tails")
        cut_all.setToolTip("Set the end of every movie to the frame on the left, in "
                           "every condition, keeping each movie's start.")
        cut_all.clicked.connect(self._cut_all_tails)
        propose = QPushButton("Propose")
        propose.clicked.connect(self._propose_one)
        propose_all = QPushButton("Propose for all")
        propose_all.clicked.connect(self._propose_all)
        self._reread = QPushButton("Re-read movies")
        self._reread.setToolTip(
            "Recompute the curves from the raw files. Only needed if the acquisitions "
            "have changed; the saved curves hold the same numbers.")
        self._reread.clicked.connect(self.reread)
        auto = QHBoxLayout()
        auto.addWidget(self._rule)
        auto.addWidget(self._tail)
        # BESIDE THE TAIL CUT, because the two are the same kind of decision:
        # where the usable part of the movie starts and where it ends.
        auto.addWidget(self._flatness)
        auto.addWidget(propose)
        auto.addWidget(propose_all)
        form.addRow("Auto", auto)
        common_start = QHBoxLayout()
        common_start.addWidget(self._start_frame)
        common_start.addWidget(cut_starts)
        common_start.addStretch(1)
        form.addRow("Start", common_start)
        common_end = QHBoxLayout()
        common_end.addWidget(self._end_frame)
        common_end.addWidget(cut_all)
        common_end.addStretch(1)
        form.addRow("Tail", common_end)

        # ITS OWN ROW, next to where the curves came from. It was tucked into the
        # "Auto" row beside Propose and Propose-for-all, five controls on one line, and
        # re-reading is not an automation choice, it is what you press when the movies
        # changed under a saved set. The experimenter could not find it.
        curves_row = QHBoxLayout()
        self._source_note = QLabel()
        self._source_note.setWordWrap(True)
        self._source_note.setStyleSheet("color: #555;")
        curves_row.addWidget(self._source_note, 1)
        curves_row.addWidget(self._reread)
        form.addRow("Curves", curves_row)

        self._note = QLabel()
        self._note.setWordWrap(True)
        self._note.setStyleSheet("color: #555;")
        form.addRow("", self._note)
        right.addWidget(box)
        layout.addLayout(right, 1)

    # -- data ----------------------------------------------------------------
    def set_folder(self, root, *, output_root=None, use_cache: bool = True,
                   token: str = ""):
        """Load a folder of raw acquisitions and start computing their curves.

        ``output_root`` is where RESULTS go, and it is separate from where acquisitions
        are READ because organising moves the two apart. After filing, the movies live
        in `<condition>/raw/`, so writing beside them buried `crops.json` and the
        bleaching curves inside `raw/`, where nobody looks and where they sit oddly next
        to `tif/`, `sum/` and `phase/` at the level above. Results go to the CONDITION
        folder; only the reading follows the movies.
        """
        from .movieprep import MoviePrepError, load_crops
        from .rawmovie import discover_raw_movies

        self._root = Path(root)
        self._output_root = Path(output_root) if output_root else Path(root)
        self._reading = False
        # A NEW FOLDER IS NOT COVERED BY THE LAST PRESS, but it IS covered by a standing
        # choice. Pressing "Propose for all" is about the movies in front of you;
        # ticking the box is a decision about how you work, so it survives the folder
        # change and arms the read that is about to start.
        self._auto = ((self._rule.currentText(), self.cut_rule())
                      if self._auto_armed else None)
        # A NEW PASS FOR DISPLAY, so anything still arriving from the last one is not
        # for this panel. It is still forwarded to the owner, which is what keeps a read
        # alive across a switch.
        self._generation = getattr(self, "_generation", 0) + 1
        # THE FOLDER'S OWN NAME, never its parent's. This read `parent.name`, so every
        # pass on `<parent>/<condition>` filed under the PARENT -- one token for all of
        # them -- and the session recorded curves against "20240101" instead of
        # "condition1". Invisible until reading was separated from displaying, because
        # until then the token was only used to drop stale passes and any constant did
        # that. It is the same rule the project layout already follows: a folder is
        # identified by its own name, since the folders above it carry no meaning.
        # THE OUTPUT ROOT, matching `adopt`. Reading follows the movies into `raw/`
        # while results go to the condition folder, so `root` alone tokens a call that
        # passes `<condition>/raw` as "raw" -- one token for every condition again, by a
        # different route. The two entry points now name a folder the same way.
        self._token = token or Path(output_root or root).name
        self._curves, self._errors, self._crops = {}, {}, {}
        self._movies, _companions = discover_raw_movies(self._root)
        self._list.clear()
        self._list.addItems([_movie_name(p) for p in self._movies])

        try:
            found = self.crops_file()
            payload = load_crops(found) if found else None
        except MoviePrepError:
            payload = None
        if payload is None:
            pass
        else:
            self._crops = {k: tuple(v) for k, v in payload.get("crops", {}).items()}

        if not self._movies:
            self._note.setText(f"no raw acquisitions in {self._root}")
            self._reading = False
            self.reading_changed.emit(False)
            return

        # ALREADY DONE IS ALREADY DONE. Reading the curves costs about 7.7 s per movie,
        # and on a folder that has been through this once the answer is already on disk
        #, the same numbers, written by the pass that produced them. Recomputing them
        # is a wait that buys nothing, so the CSVs are loaded instead and the raw files
        # are not opened at all.
        #
        # RE-READING STAYS ONE PRESS AWAY, because the cache is only right while the
        # movies are: replace an acquisition and the file on disk describes the old one.
        # `reread()` is that press, and it is the only way back to the raw files.
        why_not = self._load_cached_curves() if use_cache else "re-read requested"
        if not why_not:
            self._reading = False
            self.reading_changed.emit(False)
            self._note.setText(
                f"curves loaded from {self._output_root.name}/bleaching: the movies "
                f"were not re-read. Press 'Re-read movies' to recompute them.")
            self._source_note.setText(
                f"loaded from {self._output_root.name}/bleaching: the movies were not "
                f"re-read")
            if self._list.count():
                self._list.setCurrentRow(0)
            return
        # NOT SILENT. Falling back to reading is correct, but a wait nobody was told
        # about reads as a feature that does not work.
        self._source_note.setText(f"read from the raw movies: {why_not}")
        # READING IS A STATE ANYONE ELSE CAN ASK ABOUT, because while this pass runs the
        # acquisitions are OPEN. On Windows an open file cannot be moved (WinError 32),
        # so the tracking window's Organise button has to know not to file them yet -
        # otherwise the press fails part way through and leaves the folder half filed.
        self._reading = True
        self.reading_changed.emit(True)
        self._progress.setRange(0, len(self._movies))
        self._progress.setValue(0)
        self._progress.setFormat("reading %v of %m")
        self._note.setText(f"reading the movies: {why_not}.")
        self._progress.show()
        # KEEP THE JOB ALIVE. `QThreadPool.start` takes ownership of the runnable on
        # the C++ side, but the `_Signals` QObject it carries is a PYTHON object owned
        # by a local variable, let that go and the worker emits into a deleted signal
        # source, which surfaces as `RuntimeError: Signal source has been deleted` from
        # a background thread, where nothing can catch it.
        job = _CurveJob(self._movies, generation=self._generation,
                        token=self._token)
        job.signals.one.connect(self._got_curves)
        job.signals.done.connect(self._all_done)
        self._job = job
        self._inflight = getattr(self, "_inflight", 0) + 1
        QThreadPool.globalInstance().start(job)

    #: A generation no `set_folder` can ever produce, so a pass started with it is
    #: permanently "not the folder on screen" and every display path declines it while
    #: `curve_arrived` still files its results by token.
    BACKGROUND = -1

    def read_in_background(self, movies, token: str, cache_dir=None) -> None:
        """Read a condition NOBODY is looking at, filing its results under `token`.

        THIS IS WHAT SEPARATES READING FROM DISPLAYING. The walk through a session used
        to advance by pointing the panel at the next condition, so "read the rest" and
        "show me this one" were the same action -- click a condition that was already
        read and the walk instantly took the view away again, which is the opposite of
        what a click is for.

        Started at `BACKGROUND`, so `_got_curves` and `_all_done` both take their
        stale-generation branch: nothing is drawn, nothing on screen moves, and the
        results reach the window through `curve_arrived` / `pass_finished` exactly as an
        abandoned pass does. The files are open while it runs, so it counts as reading
        for anyone asking (the Organise button, which cannot move an open file on
        Windows).

        ``cache_dir`` is that condition's `bleaching/` folder. Given one whose CSVs are
        complete, no movie is opened at all and the curves are emitted straight from
        disk -- the same saving `set_folder` makes, and worth 7.7 s per movie.
        """
        movies = list(movies)
        if not movies:
            self.pass_finished.emit(str(token))
            return
        # THE CACHE FIRST, exactly as `set_folder` does. Skipping it made the walk
        # re-read a condition whose curves were already on disk -- 7.7 s per movie to
        # recompute a file it could have read -- and the panel would have loaded the
        # same condition instantly the moment anyone clicked it.
        if cache_dir is not None and Path(cache_dir).is_dir():
            loaded, missing = self.curves_from_cache(
                movies, Path(cache_dir), flat_tolerance=self.flat_tolerance())
            if not missing:
                # IT COUNTS AS A PASS, and that is not bookkeeping. `reading_changed`
                # going False is what advances the walk to the next condition, and the
                # cache path emitted only `pass_finished` -- so a condition loaded from
                # disk ended the walk where it stood. Measured on three conditions whose
                # curves were all cached: the third was never read and sat reading
                # "threshold: not measured" until it was clicked.
                self._inflight = getattr(self, "_inflight", 0) + 1
                for stem, curves in loaded.items():
                    self.curve_arrived.emit(str(token), stem, curves, "")
                # DEFERRED, so the walk does not recurse through every remaining
                # condition on one stack: `_all_done` emits the edge, the window starts
                # the next read, and that read may be cached too.
                QTimer.singleShot(
                    0, lambda t=str(token): self._all_done(self.BACKGROUND, t))
                return
        self._background = getattr(self, "_background", 0) + 1
        job = _CurveJob(movies, generation=self.BACKGROUND, token=str(token))
        job.signals.one.connect(self._got_curves)
        job.signals.done.connect(self._all_done)
        # HELD, like the foreground job: `QThreadPool.start` owns the runnable but not
        # the Python `_Signals` object, and letting that go emits into a deleted source
        # from a worker thread where nothing can catch it.
        self._background_jobs = getattr(self, "_background_jobs", [])
        self._background_jobs.append(job)
        # COUNTED BEFORE ANNOUNCED. `reading_changed(True)` runs the window's handler
        # synchronously, and that handler asks `is_reading` -- so incrementing after the
        # emit would have it answer False about the very pass being announced, and
        # re-enable the Organise button over files this job is opening.
        self._inflight = getattr(self, "_inflight", 0) + 1
        self.reading_changed.emit(True)
        QThreadPool.globalInstance().start(job)

    def set_auto_propose(self, enabled: bool) -> None:
        """Propose a crop for every movie AS IT IS READ, rather than on a later press.

        Asked for: a tick box beside the folder button so opening a folder both reads
        the movies and decides where to crop.

        THIS DOES NOT MAKE A CROP A DEFAULT, which is the rule the panel is built on. A
        proposal still comes from a NAMED rule that is on screen beside it, every value
        stays editable, and the box is off until somebody turns it on. What changes is
        only WHEN the proposal happens: arming it before the read spares a second press
        on a folder whose curves take minutes to compute.
        """
        self._auto_armed = bool(enabled)
        if not enabled:
            self._auto = None
            return
        self._auto = (self._rule.currentText(), self.cut_rule())
        # Movies already read are covered too, or ticking the box after a read would do
        # nothing visible and read as a broken control.
        for stem in list(self._curves):
            if stem not in self._crops:
                try:
                    self._apply_proposal(stem, show=False)
                except Exception:                   # noqa: BLE001 - named on the panel
                    self._errors.setdefault(stem, "no proposal")
        self._show(self._current())
        self.changed.emit()

    def adopt(self, root, *, output_root=None, curves, crops, errors=None) -> None:
        """Show a condition this window has ALREADY read, without reading it again.

        The condition picker is a view control: "the condition picker just should be a
        picker to see which conditions i can see at the photobleaching curve picker
        below, not read and adjust movies again". `set_folder` cannot serve that, because
        it starts a folder clean and launches the background pass, so every switch paid
        the full 7.7 s per movie a second time and discarded the answer it already had.

        The owner keeps the session; this puts one condition of it back on screen. No job
        is started and nothing is recomputed.
        """
        from .rawmovie import discover_raw_movies

        # A NEW GENERATION ANYWAY. A pass may still be in flight from the folder being
        # left, and its results must not land in the one now displayed.
        self._generation = getattr(self, "_generation", 0) + 1
        self._token = Path(output_root or root).name
        self._root = Path(root)
        self._output_root = Path(output_root) if output_root else Path(root)
        self._movies, _companions = discover_raw_movies(self._root)
        known = {_movie_name(p) for p in self._movies}
        self._curves = {k: v for k, v in (curves or {}).items() if k in known}
        self._crops = {k: tuple(v) for k, v in (crops or {}).items() if k in known}
        self._errors = {k: v for k, v in (errors or {}).items() if k in known}
        self._list.clear()
        self._list.addItems([_movie_name(p) for p in self._movies])
        self._reading = False
        self.reading_changed.emit(False)
        self._progress.hide()
        self._note.setText(
            f"{len(self._curves)} of {len(self._movies)} movie(s) already read this "
            f"session: shown from memory, not re-read")
        if self._list.count():
            self._list.setCurrentRow(0)
        self.changed.emit()

    def curves(self) -> dict:
        """The curves computed for the folder on screen, for the owner to keep."""
        return dict(self._curves)

    def errors(self) -> dict:
        """Movies that could not be read, by stem, with the reason."""
        return dict(self._errors)

    def restore_crops(self, crops) -> None:
        """Put a remembered crop set back, for the condition now on screen.

        The window owns the SESSION; this panel owns one folder of it. Switching away
        and back used to lose every range set here, because `set_folder` starts clean and
        the ranges are not on disk until they are saved. They are the expensive part of
        this window, so the owner keeps them and hands them back.

        Only for movies this folder actually holds: a stem from another condition would
        be recorded against a movie that is not there. `series001` exists in every
        condition, so this is not a hypothetical.
        """
        known = {_movie_name(p) for p in self._movies}
        for stem, value in (crops or {}).items():
            if stem in known:
                self._crops[stem] = tuple(value)
        if self._list.currentItem():
            self._show(self._list.currentItem().text())
        self.changed.emit()

    def progress(self) -> tuple:
        """``(n_movies, n_curves, n_crops, n_unreadable)`` for the folder on screen.

        The overview reads the other conditions off disk, where curves and crops only
        appear once they are SAVED. The folder being worked on has neither yet, so its
        row would sit at zero while the user watches it fill. This is that row's truth.
        """
        return (len(self._movies), len(self._curves), len(self._crops),
                len(self._errors))

    @property
    def rule(self) -> str:
        """The start rule currently selected, so a caller can name it in a log."""
        return self._rule.currentText()

    def end_rule(self) -> dict:
        """How a proposal ends a movie, as `propose_crop` keywords: one common frame when
        it is set, otherwise the density cut (None keeps the whole tail)."""
        frame = int(self._end_frame.value())
        if frame:
            return {"end_frame": frame}
        return {"end_min_spots_per_frame": self._tail.value() or None}

    def cut_rule(self) -> dict:
        """Everything a proposal takes from the common cuts, as `propose_crop` keywords:
        the end (`end_rule`) and, when one is set, the common start frame."""
        rule = self.end_rule()
        start = int(self._start_frame.value())
        if start:
            rule["start_frame"] = start
        return rule

    def _n_frames_read(self) -> dict:
        """``{movie: length}`` for every movie whose curves are in hand."""
        return {stem: int(curves.n_frames) for stem, curves in self._curves.items()
                if getattr(curves, "n_frames", None)}

    def _cut_all_starts(self) -> None:
        """Every movie now starts at the common frame; each end is kept."""
        frame = int(self._start_frame.value())
        if not frame:
            self._note.setText("set 'start every movie at frame' first")
            return
        self._apply_common_cut(start=frame)
        self.start_frame_applied.emit(frame)

    def _cut_all_tails(self) -> None:
        """Every movie now ends at the common frame; each start is kept."""
        frame = int(self._end_frame.value())
        if not frame:
            self._note.setText("set 'cut every movie at frame' first")
            return
        self._apply_common_cut(end=frame)
        self.end_frame_applied.emit(frame)

    def _apply_common_cut(self, **edge) -> None:
        """`prep_model.common_cut` on this folder, worded by `prep_view`. ONE rule for the
        panel and the window, so the two cannot cut differently."""
        from .prep_model import common_cut
        from .prep_view import common_cut_note

        report = common_cut(self._crops, self._n_frames_read(), **edge)
        self._crops = dict(report.crops)
        self._note.setText(common_cut_note(report))
        if self._list.currentItem():
            self._show(self._list.currentItem().text())
        self.changed.emit()

    @property
    def auto_propose(self) -> bool:
        """Whether a crop is proposed for each movie as it is read."""
        return bool(getattr(self, "_auto_armed", False))

    def crops(self):
        return dict(self._crops)

    def missing(self):
        """Movies with no range yet: the run cannot start until this is empty."""
        return [name for name in map(_movie_name, self._movies) if name not in self._crops]

    @staticmethod
    def curves_from_cache(movies, bleach_dir, *,
                          flat_tolerance: float = FLAT_TOLERANCE):
        """Rebuild curves from `bleaching/*.csv`. Returns (loaded, missing).

        `prep_model.curves_from_cache` is the one reader of the cache; this keeps the
        panel's name for it, which the window and the tests call.
        """
        from .prep_model import curves_from_cache

        return curves_from_cache(movies, bleach_dir, flat_tolerance=flat_tolerance)

    def flat_tolerance(self) -> float:
        """The flatness tolerance on screen, as a FRACTION. Read, never assumed."""
        from .movieprep import FLAT_TOLERANCE

        box = getattr(self, "_flatness", None)
        return FLAT_TOLERANCE if box is None else float(box.value()) / 100.0

    def _flatness_changed(self, percent: float) -> None:
        """Recompute every floor from the curves already held. Reads no movie.

        The expensive half of `crop_floor` is detecting spots, and that is already done
        and cached, so a new tolerance costs a scan of two arrays per movie. A movie whose
        floor cannot be rebuilt keeps the one it has rather than losing it -- an absent
        floor reads as "this movie has no defensible start", which is a different claim.
        """
        from .prep_model import refloor

        tolerance = float(percent) / 100.0
        for stem, curves in list(self._curves.items()):
            self._curves[stem] = refloor(curves, tolerance)
        # NAMED DIRECTLY. The first version looked the redraw up by trying four
        # plausible method names and fell through all of them, so the new floors were
        # computed and never drawn -- a handler that silently does half its job.
        self._redraw()
        self.changed.emit()

    def _load_cached_curves(self) -> str:
        """Load the curves from disk. Returns "" on success, else WHY it could not.

        The reason is the point. The first version returned a bool, so a folder whose
        curves were saved before the density file existed simply started reading again
        with nothing on screen to say why: indistinguishable from the cache not
        working. Ten `_full.csv`, ten `_cropped.csv`, zero `_density.csv` is a complete
        explanation and the window should give it.
        """
        return self._load_cached_curves_impl()

    def _load_cached_curves_impl(self) -> str:
        """Rebuild every movie's curves from `bleaching/*.csv`. True if ALL of them.

        ALL OR NONE, deliberately. A half-loaded set would leave some movies instant and
        others blank, and the panel would have to run the background pass anyway for the
        rest, so the only thing a partial load buys is a confusing screen.
        """
        bleach_dir = self._output_root / "bleaching"
        if not bleach_dir.is_dir():
            return f"no {self._output_root.name}/bleaching folder"
        loaded, missing = self.curves_from_cache(
            self._movies, bleach_dir, flat_tolerance=self.flat_tolerance())
        if missing:
            kinds = sorted({k for _stem, absent in missing for k in absent})
            return (f"{len(missing)} of {len(self._movies)} movie(s) have no saved "
                    f"{', '.join(kinds)} curve: reading them once will complete the "
                    f"set, and the next open will be instant")
        if not loaded:
            return "nothing saved yet"
        self._curves = loaded
        return ""

    def reread(self):
        """Read the movies again, discarding the cached curves.

        The cache is right only while the movies are: replace an acquisition and the
        CSVs on disk describe the one before it. Nothing here can detect that (a
        mtime is not a measurement) so it is a press rather than a guess.
        """
        if self._root is None:
            return
        # The CSVs are left alone: they are still a true record of the last pass, and
        # the next save overwrites them. Only the in-memory copy is dropped.
        self._curves = {}
        self.set_folder(self._root, output_root=self._output_root, use_cache=False)

    def crops_file(self):
        """An existing `crops.json` for this folder, or None.

        Both places are searched, output root first. New sets are written beside the
        other results; sets written before that change sit beside the acquisitions, and
        silently ignoring those would re-ask a question already answered.
        """
        from .movieprep import find_crops

        return find_crops(self._root, getattr(self, "_output_root", None))


    def save(self, *, pixel_size_nm, frame_interval_s, write_curves: bool = False,
             label=None, emission_nm=None, numerical_aperture=None,
             projection: str | None = None, on_progress=None):
        """Write `crops.json`, and the bleaching curves beside it.

        The curves are already in memory, so writing them costs a file each. Until now
        they were written only by `prepare_movie`, i.e. only if someone exported cropped
        stacks, and tracking from raw, which is the DEFAULT route for a raw folder,
        exports nothing. So the macro's record simply did not exist for anyone taking
        that path, which is not a trade anybody chose.

        The writing itself is `prep_model.save_condition`, shared with the workspace-style
        window. It touches no widget, because this runs on a worker thread
        (`tracking_gui._SaveCropsJob`).
        """
        from .prep_model import save_condition

        if self._root is None:
            raise RuntimeError("no folder loaded")
        # WRITTEN WHERE THE CURVES ARE, not at the condition root. They are one record.
        return save_condition(
            self._output_root, self._crops, pixel_size_nm=pixel_size_nm,
            frame_interval_s=frame_interval_s,
            curves=self._curves if write_curves else None, movies=self._movies,
            label=label, emission_nm=emission_nm, numerical_aperture=numerical_aperture,
            projection=projection, on_progress=on_progress)

    def write_projections(self, kind: str, *, on_progress=None):
        """The `sum/` image per movie, from the frames inside each crop
        (`prep_model.write_projections`). With no callback the note lands on the panel;
        with one, NO WIDGET IS TOUCHED -- it may be running on a worker thread."""
        from .prep_model import write_projections

        note = write_projections(self._movies, self._crops, self._output_root, kind,
                                 on_progress=on_progress)
        if on_progress is None:
            self._progress.hide()
            self._note.setText(note)
        return note

    def write_curves(self, frame_interval_s, *, on_progress=None):
        """The full, cropped and density curves as CSV plus the picture, per movie
        (`prep_model.write_curves`). Widgets only when no callback is given."""
        from .prep_model import write_curves

        note = write_curves(self._curves, self._crops, self._output_root,
                            frame_interval_s, on_progress=on_progress)
        if on_progress is None:
            self._note.setText(note)
        return note

    # -- slots ---------------------------------------------------------------
    def _at_current_tolerance(self, curves):
        """``curves`` with its floor at the tolerance ON SCREEN (`prep_model.refloor`).

        A fresh read computes its floor at `FLAT_TOLERANCE`, whatever the box says, so a
        movie read after the box was moved arrived with a 2 % floor beside movies already
        held at the box's value, and a proposal for it used that 2 %.
        """
        from .prep_model import refloor

        if getattr(curves, "floor", None) is None:
            return curves
        return refloor(curves, self.flat_tolerance())

    def _got_curves(self, stem, curves, error, generation=None, token=""):
        curves = self._at_current_tolerance(curves)
        # FORWARDED FIRST, WHATEVER HAPPENS TO IT HERE. A pass belonging to a folder that
        # is no longer displayed must not be thrown away: reading a movie takes seconds,
        # so switching condition always leaves one in flight, and discarding it meant a
        # switch silently abandoned minutes of work. The owner records it; this panel
        # only DISPLAYS, and displays what is its own.
        self.curve_arrived.emit(token, stem, curves, error or "")
        if generation is not None and generation != self._generation:
            return
        if error:
            self._errors[stem] = error
        else:
            self._curves[stem] = curves
        self._progress.setValue(len(self._curves) + len(self._errors))
        # A MOVIE THAT ARRIVES AFTER "Propose for all" GETS THE SAME RULE. Without this
        # the button means "propose for the ones read so far", which is not what it
        # says and not what anybody reads it as.
        auto = getattr(self, "_auto", None)
        if auto is not None and stem in self._curves and stem not in self._crops:
            rule, ends = auto
            try:
                from .movieprep import propose_crop

                got = propose_crop(self._curves[stem], start_rule=rule, **ends)
            except Exception:                       # noqa: BLE001 - named on the panel
                self._errors.setdefault(stem, f"no {rule} proposal")
            else:
                self._crops[stem] = (got["start"], got["end"])
                self.changed.emit()
        if self._list.currentItem() and self._list.currentItem().text() == stem:
            self._show(stem)

    @property
    def is_reading(self) -> bool:
        """True while ANY pass still has acquisitions open, displayed or not.

        The background passes count. They hold the same files, and this property's one
        consumer is the Organise button, which fails part way through on Windows if it
        tries to move a file another process has open (WinError 32).
        """
        # BOTH, and deliberately the conservative reading. `_reading` is the DISPLAYED
        # pass; `_inflight` counts every pass, including one the user switched away from
        # and the off-screen reads the walk starts. Either alone unlocks the button over
        # files somebody still holds.
        return bool(getattr(self, "_reading", False)
                    or getattr(self, "_inflight", 0) > 0)

    def _all_done(self, generation=None, token=""):
        """One pass ended. Retire it, and report the SET rather than this one.

        `is_reading` answers "are the acquisitions open", and its consumer is the
        Organise button, which on Windows cannot move a file another process holds. That
        is a question about EVERY pass in flight, including one the user switched away
        from -- which keeps running by design, so that a read is not lost to a click.
        Counting the displayed pass alone let the button unlock over an orphan still
        holding ten movies open.
        """
        self._inflight = max(0, getattr(self, "_inflight", 0) - 1)
        self.pass_finished.emit(token or "")
        if generation == self.BACKGROUND:
            self._background = max(0, getattr(self, "_background", 0) - 1)
            self._background_jobs = [j for j in getattr(self, "_background_jobs", [])
                                     if j.signals is not self.sender()]
        elif generation is None or generation == self._generation:
            self._reading = False
            self._progress.hide()
            if not self._list.currentItem() and self._list.count():
                self._list.setCurrentRow(0)
            if self._errors:
                self._note.setText(
                    f"{len(self._errors)} movie(s) could not be read: "
                    + "; ".join(f"{k} ({v})" for k, v in list(self._errors.items())[:2]))
        # THE EDGE, not this pass's own end. `reading_changed(False)` unlocks the
        # writers and lets the walk start the next condition, so it belongs to the last
        # pass standing.
        if not self.is_reading:
            self.reading_changed.emit(False)

    def _show(self, stem):
        if not stem:
            return
        if stem in self._errors:
            self._note.setText(f"{stem}: {self._errors[stem]}")
            return
        curves = self._curves.get(stem)
        if curves is None:
            self._note.setText(f"{stem}: reading …")
            return
        for widget in (self._start, self._end):
            widget.blockSignals(True)
        self._start.setMaximum(curves.n_frames)
        self._end.setMaximum(curves.n_frames)
        low, high = self._crops.get(stem, (1, curves.n_frames))
        self._start.setValue(low)
        self._end.setValue(high)
        for widget in (self._start, self._end):
            widget.blockSignals(False)
        self._proposal = None
        self._redraw()

    def _redraw(self):
        stem = self._current()
        curves = self._curves.get(stem) if stem else None
        if curves is None:
            return
        from .plots import draw_crop_curves

        draw_crop_curves(self._figure, curves,
                         crop=(self._start.value(), self._end.value()),
                         proposal=self._proposal)
        self._canvas.draw_idle()
        # ONE LINE, NOT THE PARAGRAPH. `floor.describe()` is four sentences of
        # justification, and the experimenter's verdict on it was "users won't read all
        # of this": they press Propose and change it if unhappy. The full reasoning
        # stays in `CropFloor.describe()` for anyone who wants it, on the tooltip.
        floor = curves.floor
        if floor is None:
            self._note.setText("")
        else:
            line = (f"floors: density {floor.by_density} · amplitude {floor.by_amplitude} "
                    f"· field mean {floor.by_field_mean}  →  no earlier than "
                    f"{floor.earliest_defensible} of {floor.n_frames}")
            # THE ONE CAVEAT THAT CHANGES WHAT THE NUMBER MEANS goes on the line itself,
            # not the tooltip: a floor computed on camera offset looks like any other.
            warning = getattr(floor, "offset_warning", "")
            self._note.setText(line + (f"\n{warning}" if warning else ""))
            self._note.setToolTip(floor.describe())

    # -- dragging the crop edges ---------------------------------------------
    #: How close to an edge a click has to land, as a fraction of the x range. A pixel
    #: tolerance would mean something different on every window size.
    _GRAB_FRACTION = 0.02

    def _drag_start(self, event):
        """Grab whichever edge the click is nearest, if it is near enough."""
        if event.inaxes is None or event.xdata is None or self._current() is None:
            return
        lo, hi = self._start.value(), self._end.value()
        span = max(self._end.maximum() - 1, 1)
        tolerance = span * self._GRAB_FRACTION
        distances = [abs(event.xdata - lo), abs(event.xdata - hi)]
        nearest = 0 if distances[0] <= distances[1] else 1
        if distances[nearest] > tolerance:
            return
        self._dragging = nearest

    def _drag_move(self, event):
        """Follow the pointer. The BOX is updated, not a private copy of the range.

        Signals are blocked so a full redraw does not run on every mouse event, three
        panels of a few thousand points each would make the drag stutter. The lines are
        moved directly instead, and the real update happens once on release.
        """
        if self._dragging is None or event.xdata is None:
            return
        frame = int(round(event.xdata))
        if self._dragging == 0:
            frame = max(1, min(frame, self._end.value() - 1))
            box = self._start
        else:
            frame = max(self._start.value() + 1, min(frame, self._end.maximum()))
            box = self._end
        box.blockSignals(True)
        box.setValue(frame)
        box.blockSignals(False)
        for index, line in getattr(self._figure, "crop_edges", ()):
            if index == self._dragging:
                line.set_xdata([frame, frame])
        self._canvas.draw_idle()

    def _drag_end(self, event):
        """Commit once. `_crop_edited` is what stores the range and redraws, so the
        released drag goes through exactly the path a typed number does."""
        if self._dragging is None:
            return
        self._dragging = None
        self._crop_edited()

    def _current(self):
        item = self._list.currentItem()
        return item.text() if item else None

    def _crop_edited(self):
        stem = self._current()
        if not stem or stem not in self._curves:
            return
        if self._end.value() <= self._start.value():
            self._note.setText("the end must come after the start")
            return
        self._crops[stem] = (self._start.value(), self._end.value())
        self._redraw()
        self.changed.emit()

    def _propose_one(self):
        stem = self._current()
        if not stem or stem not in self._curves:
            return
        self._apply_proposal(stem, show=True)
        self.changed.emit()

    def _propose_all(self):
        """The same rule across every movie, applied and then editable per movie.

        Applied rather than only shown, because the point of the button is to stop the
        analyst typing ten ranges by hand, but the rule is named on screen and every
        value stays editable, so it is automation the analyst chose rather than a
        default nobody saw.

        "all" includes movies not yet read. The curves arrive one at a time from a
        background pass, and this used to iterate only the ones already in hand, so
        pressing the button early proposed for a SUBSET, and every later movie still
        showed 1 to the last frame. Nothing said so: the count in the message was the
        only clue, and a range that silently stayed at the whole movie reads as "the
        proposal did not stick" rather than "that movie had not been read yet". The
        rule is now remembered and applied to each movie as it arrives.
        """
        self._auto = (self._rule.currentText(), self.cut_rule())
        applied, failed = 0, []
        for stem in list(self._curves):
            try:
                self._apply_proposal(stem, show=False)
                applied += 1
            except Exception as exc:                # noqa: BLE001 - named below
                failed.append(f"{stem} ({exc})")
        self._show(self._current())
        pending = len(self._movies) - len(self._curves) - len(self._errors)
        message = (f"proposed {self._rule.currentText()} start for {applied} movie(s); "
                   f"every value stays editable")
        if pending > 0:
            message += (f": {pending} still being read, and the same rule will be "
                        f"applied to each as it arrives")
        if failed:
            message += f": no proposal for {len(failed)}: " + "; ".join(failed[:2])
        self._note.setText(message)
        self.changed.emit()

    def _apply_proposal(self, stem, *, show):
        from .movieprep import propose_crop

        got = propose_crop(self._curves[stem], start_rule=self._rule.currentText(),
                           **self.cut_rule())
        self._crops[stem] = (got["start"], got["end"])
        if show:
            self._proposal = got
            self._show(stem)
            self._crops[stem] = (got["start"], got["end"])
            self._proposal = got
            self._redraw()
            self._note.setText(
                f"{got['start']}–{got['end']} by {got['start_rule']}; "
                f"other rules would give "
                + ", ".join(f"{k} {v}" for k, v in got["alternatives"].items()
                            if v is not None)
                + f". End: {got['end_reason']}.")
