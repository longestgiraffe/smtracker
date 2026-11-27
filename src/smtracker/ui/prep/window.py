"""The preparation window in the workspace's style: open a folder, read every movie in
the background, set ranges on the crop chart, fill the calibration, save.

Asked for: "in general the gui for image prep is clunky, long load times, and changing
some values sometimes takes long. can we use the qt analysis gui from hui for this as
well?" -- then "yes guys style. go". Movies & crop, Calibration & save, Detect & link
and Track are all here; the classic window (`tracking_gui`) stays reachable for what is
not ported yet (exporting cropped stacks, the wider-radius check).

WHAT MAKES IT FAST, each a measured fix rather than a restyle:
- the window paints first; listing the folder runs on a thread (0.6-1.6 s measured on
  One user's and the protein A folders, which the classic window spent on the GUI thread);
- movies are read in a spawned PROCESS pool, two or three at once, so reading never
  stutters the window; saved curves load from `bleaching/*.csv` without a frame read;
- the crop chart is painted (payload 8 ms, paint 16 ms on a 4000-frame movie) instead
  of redrawing a three-axes matplotlib figure on every edit;
- an edit updates the session and redraws ONE movie's chart and ONE condition's rows.

BATCH (asked 13 Oct 2026, "multiple parent folders in batches overnight") is a fifth
section, reachable with no folder open and with no Next button: several parent folders,
every condition under them with its readiness and layout, then one run through the same
`track_condition` / `track_condition_utrack` as Track (`batch_model.run_tracking`, on a
thread). While it runs it holds the window as a run does (no other run, no folder change,
no tracker change), keeps Windows awake, and writes its CSV report into the first parent.

THIN QT: every rule is `prep_model` / `batch_model`, every word `prep_view` / `batch_model`.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QApplication,
    QFileDialog,
    QMessageBox,
    QHBoxLayout,
    QMainWindow,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ...chart_text import fold_words
from ...prep_model import (
    MAX_AT_ONCE,
    RADIUS_CHECK_MULTIPLE,
    PrepError,
    default_at_once,
    apply_common_cut,
    apply_flat_tolerance,
    auto_flat_tolerance,
    copy_calibration,
    copy_tracking,
    measurable,
    record_noise,
    rethreshold,
    seed_tracking,
    set_tracking,
    summarise_noise,
    file_curves,
    fill_from_acquisition,
    next_step,
    propose,
    propose_all,
    refloor,
    export_inputs,
    run_inputs,
    stack_files,
    track_files,
    set_calibration,
    set_crop,
    short_crop_note,
    set_utrack,
    utrack_inputs,
)
from ...prep_view import (
    START_RULES,
    UTRACK_STARTING,
    calibration_cells,
    common_cut_note,
    condition_status,
    crop_chart,
    crowding_line,
    noise_lines,
    outline_note,
    reach_lines,
    tracking_cells,
    tracking_status,
    fiji_lines,
    finished_banner,
    progress_line,
    run_entry,
    run_header,
    track_rows,
    export_line,
    export_rows,
    pilot_lines,
    radius_check_lines,
    field_names,
    floor_note,
    interval_note,
    movie_rows,
    propose_note,
    save_state,
    sidebar_row,
    utrack_cells,
    utrack_condition_rows,
    utrack_differences,
    utrack_engine_lines,
    utrack_progress_line,
    utrack_run_header,
    utrack_track_rows,
)
from ... import options as opts
from ... import prep_guide
from ...batch_model import (
    add_parent,
    batch_progress,
    event_line,
    finished_summary,
    folder_key,
    awake_note,
    keep_awake,
    layout_notes,
    plan_lines,
    plan_summary,
    report_path,
    result_status,
    running_status,
    settings_line,
    snapshot,
    track_batch_cells,
)
from ...prep_view import at_once_tip, layout_note
from ...run_progress import finished_payload, progress_payload
from .. import theme
from ..theme import typography as ty
from ..widgets.help import show_help_menu
from ..widgets.controls import Button
from ..widgets.primitives import ShadowHost, activity_label, label, scrollbar_css
from ..widgets.options_dialog import open_options
from ..widgets.progress import RunProgress, notify_after_s, show_finished
from ..workspace.sidebar import Breadcrumb, Sidebar, SidebarTrigger
from .batch_page import TrackBatchPage
from .calibration_page import CalibrationPage
from .conditions import ConditionList
from .detect_page import DetectPage
from .jobs import PrepJobs
from .movies_page import MoviesPage
from .track_page import TrackPage

__all__ = ["PrepWindow", "open_prep", "PREP_SECTIONS"]

#: The preparation steps, `SECTIONS`-shaped for the workspace sidebar.
PREP_SECTIONS = (
    ("movies", "Movies & crop", "scan", (("movies.crop", "Crop"),)),
    ("calibration", "Calibration & save", "ruler", (("calibration.save", "Save"),)),
    ("detect", "Detect & link", "crosshair", (("detect.settings", "Settings"),)),
    ("track", "Track", "play", (("track.run", "Run"),)),
    # NOT A STEP OF ONE FOLDER'S WORK: several parent folders, overnight (asked 13 Oct
    # 2026). Reachable with no folder open, and it has no Next button.
    ("batch", "Batch", "layers", (("batch.run", "Run"),)),
)
_TITLES = {key: title for key, title, _g, _s in PREP_SECTIONS}

#: Windows opened by `open_prep`, held: a parentless window nobody references is
#: destroyed at once and takes its threads with it (the segmentation door's exit 139).
_OPEN_WINDOWS: set = set()


#: what Track all says when no row can start; cleared once one can (`_refresh_track`)
NOT_READY_NOTE = "no condition is ready to track; each row says why"

class _StartPage(QWidget):
    """Before a folder: one sentence and one button. Also where an opening error lands."""

    def __init__(self, t, on_open, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background:transparent;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 40, 0, 0)
        lay.setSpacing(12)
        lay.addWidget(label("Image preparation", ty.ui("2xl", "semibold"), t.foreground))
        self.text = label(
            "Open a folder holding raw acquisitions (.nd2 or .stk), or a parent folder "
            "holding several conditions. Every movie is read in the background; saved "
            "curves load instantly.", ty.ui("sm"), t.muted_foreground)
        self.text.setWordWrap(True)
        self.text.setMaximumWidth(620)
        lay.addWidget(self.text)
        button = Button(t, "Open folder…", variant="default", glyph="folder-open")
        button.clicked.connect(on_open)
        lay.addWidget(button, alignment=Qt.AlignLeft)
        lay.addStretch(1)


class PrepWindow(QMainWindow):
    """The shell, the session, and the wiring between them."""

    def __init__(self, t, *, jobs: PrepJobs | None = None, parent=None):
        super().__init__(parent)
        self.t = t
        from ...buildstamp import build_stamp

        self.setWindowTitle(f"SMTracker · preparation  ·  {build_stamp()}")
        self.setStyleSheet(f"QMainWindow {{ background:{t.background}; }}"
                           + scrollbar_css(t))
        self.jobs = jobs or PrepJobs(self)
        s = self.jobs.signals
        s.opened.connect(self._opened)
        s.cached.connect(self._cached)
        s.curve.connect(self._curve)
        s.probed.connect(self._probed)
        s.saved.connect(self._saved)
        s.save_progress.connect(self._save_progress)
        s.noise_row.connect(self._noise_row)
        s.fiji.connect(self._fiji_found)
        s.track_progress.connect(self._track_progress)
        s.tracked.connect(self._tracked)
        s.export_row.connect(self._export_row)
        s.install_line.connect(self._install_line)
        s.installed.connect(self._install_done)
        s.pilot.connect(self._pilot_done)
        s.purity.connect(self._purity_done)
        s.radius_progress.connect(self._radius_progress)
        s.radius_checked.connect(self._radius_checked)
        s.utrack_found.connect(self._utrack_found)
        s.utrack_health.connect(self._utrack_health_done)
        s.batch_planned.connect(self._batch_planned)
        s.batch_event.connect(self._batch_event)
        s.batch_done.connect(self._batch_done)

        #: The open session: ``{condition: ConditionState}``, empty before a folder.
        self.states: dict = {}
        self.folder: Path | None = None
        self.condition: str | None = None
        self._generation = 0
        #: The standards from the options screen (`smtracker.options`): what the crop rule,
        #: the tolerance, the TrackMate grid and the u-track card start from.
        self._options = opts.values("prep", theme.remembered(opts.OPTIONS_KEY))
        self._rule = self._options["crop_start_rule"]
        self._tolerance = float(self._options["crop_flat_tolerance_pct"]) / 100.0
        #: Conditions whose "Propose for all" was pressed: movies read later get it too.
        self._auto_propose: set = set()
        self._saving: list = []
        #: The calibration page's notes, one per subject (see `_show_notes`).
        self._notes: dict = {}
        #: Noise rows arriving per condition: ``{condition: (expected, {movie: row},
        #: radius)}`` until the last one lands.
        self._measuring: dict = {}
        #: The Fiji in use (`fiji.FijiInstall`) or None, what to say when there is none,
        #: and a folder picked by hand that is still being checked.
        self._install = None
        self._fiji_absence = ""
        self._fiji_wanted: Path | None = None
        #: The condition being tracked, its ``(done, total)``, the stop flag the run asks
        #: before each movie, the conditions waiting, and whether an earlier run may be
        #: replaced.
        self._tracking: str | None = None
        self._track_done: tuple | None = None
        self._stop: threading.Event | None = None
        self._track_queue: list = []
        self._overwrite = False
        #: The conditions finished in the run in flight (`prep_view.run_entry`), shown in
        #: the banner when the last queued one is done; and every condition tracked this
        #: session, so its row says what the run produced.
        self._run_entries: list = []
        self._finished: dict = {}
        #: The condition the tracking guide shows: the one whose grid cell was clicked
        #: into, else the condition picked in the sidebar.
        self._guide_condition: str | None = None
        #: Stack exports in flight: ``{condition: (expected, [rows])}``.
        self._exporting: dict = {}
        self._export_replace = False
        #: The packages an offered install is fetching (`optional_deps`), empty when idle.
        self._installing: list = []
        self._install_lines: list = []
        #: The condition whose wider-radius check is running (it holds Fiji, like a run).
        self._checking: str | None = None
        self._check_stop: threading.Event | None = None
        #: Set by "Back to menu": `app.main` reopens the stage menu when this closes.
        self.return_to_menu = False
        #: `prep_model.tail_purity` per condition, measured on request.
        self._purity: dict = {}
        #: Which tracker the Detect & link and Track pages are for ("trackmate" |
        #: "utrack"), remembered per user; the tracker of the run in flight.
        self._tracker = (theme.remembered("prep_tracker")
                         if theme.remembered("prep_tracker") in ("trackmate", "utrack")
                         else "trackmate")
        self._run_tracker = self._tracker
        #: u-track: the settings (one set for every condition, remembered), the MATLAB and
        #: u-track found, the health check's answer, and paths picked by hand.
        self._utrack_settings = _remembered_utrack_settings()
        self._matlab = None
        self._utrack = None
        self._utrack_health = None
        self._utrack_error = ""
        self._utrack_absence = ""
        self._utrack_searching = False
        self._matlab_wanted: Path | None = None
        #: MATLAB is being started for the u-track check, and what to do when it answers:
        #: "track" (the queued conditions) or "batch". MATLAB starts only then -- never on
        #: opening the window (14 Oct 2026).
        self._utrack_checking = False
        self._after_check: str | None = None
        self._utrack_wanted: Path | None = None
        #: Long jobs as the progress card and the finished pop-up see them
        #: (`run_progress`): when each kind started (monotonic seconds), what each run
        #: holds ({condition: movies}), the conditions done, the summary lines gathered,
        #: and how long a job must run before its end gets a pop-up.
        self._started: dict = {}
        self._run_plan: dict = {}
        self._run_done: list = []
        self._export_plan: dict = {}
        self._export_lines: list = []
        self._measure_plan: dict = {}
        self._measure_lines: list = []
        self._save_lines: list = []
        self._save_counts: tuple | None = None
        self._check_done: tuple | None = None
        self._notify_after = notify_after_s()
        #: The batch over several parent folders (`batch_model`): the parents, the rows
        #: last planned, which listing is current, the choices on its page, and -- while
        #: one runs -- its stop flag (None when none runs), its plan, the conditions
        #: running and finished with their movies done, each row's status, the report.
        self._batch_parents: list = []
        self._batch_rows: list = []
        self._batch_generation = 0
        self._batch_replace = False
        from ...batch_model import MEASURE_MISSING_THRESHOLD

        self._batch_measure = MEASURE_MISSING_THRESHOLD
        #: "Movies at once", one number for the Track and Batch pages (`prep_model.default_at_once`), remembered
        saved = theme.remembered("prep_at_once")
        self._at_once = (int(saved) if isinstance(saved, int) and 1 <= saved <= MAX_AT_ONCE
                         else default_at_once())
        self._batch_stop: threading.Event | None = None
        self._batch_tracker = self._tracker
        self._batch_plan: list = []
        self._batch_running: list = []
        self._batch_finished: list = []
        self._batch_counts: dict = {}
        self._batch_status: dict = {}
        self._batch_results: dict = {}
        self._batch_report: Path | None = None
        self._batch_report_error = ""
        #: A finished batch's own summary line, kept until anything is planned again.
        self._batch_summary = ""

        central = QWidget()
        outer = QHBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.sidebar = Sidebar(t, project_title="Preparation", project_subtitle="no folder",
                               sections=PREP_SECTIONS, group_label="Preparation")
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
        head = QHBoxLayout()
        head.setSpacing(8)
        self.trigger = SidebarTrigger(t)
        self.trigger.clicked.connect(
            lambda: self.sidebar.set_collapsed(not self.sidebar.collapsed))
        head.addWidget(self.trigger)
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
        help_button.clicked.connect(lambda: show_help_menu("prep", self, help_button))
        head.addWidget(help_button)
        options_button = Button(t, "Options", size="sm", glyph="sliders-horizontal")
        options_button.setToolTip("The standards a new session starts from: crop rule, "
                                  "TrackMate and u-track settings, when to pop up.")
        options_button.clicked.connect(self.open_options)
        head.addWidget(options_button)
        back = Button(t, "Back to menu", size="sm")
        back.setToolTip("Done with preparation and tracking: back to the stage menu, for "
                        "segmentation or analysis. Closing the window leaves the tool.")
        back.clicked.connect(self.back_to_menu)
        head.addWidget(back)
        col.addLayout(head)
        self.run_progress = RunProgress(t)
        col.addWidget(self.run_progress)
        #: The card's clock: time so far and the estimate move once a second while a
        #: job runs, and the timer stops when nothing does.
        self._progress_timer = QTimer(self)
        self._progress_timer.setInterval(1000)
        self._progress_timer.timeout.connect(self._refresh_progress)

        self.pages = QStackedWidget()
        self.start_page = _StartPage(t, self.choose_folder)
        self.movies_page = MoviesPage(t)
        self._show_crop_standard()
        self.calibration_page = CalibrationPage(t)
        self.detect_page = DetectPage(t)
        self.track_page = TrackPage(t)
        self.batch_page = TrackBatchPage(t)
        def scrolled(page):
            area = QScrollArea()
            area.setWidget(page)
            area.setWidgetResizable(True)
            area.setFrameShape(QScrollArea.NoFrame)
            area.setStyleSheet("QScrollArea { background:transparent; border:none; }")
            return area

        calibration_scroll = scrolled(self.calibration_page)
        detect_scroll = scrolled(self.detect_page)
        track_scroll = scrolled(self.track_page)
        # SCROLLED TOO: Qt never shrinks a widget below its minimum, so a short screen
        # (or the crop guide open) used to push the chart's panels over the controls
        # under it. Now the page scrolls and every panel keeps its size.
        movies_scroll = scrolled(self.movies_page)
        batch_scroll = scrolled(self.batch_page)
        for page in (self.start_page, movies_scroll, calibration_scroll, detect_scroll,
                     track_scroll, batch_scroll):
            self.pages.addWidget(page)
        self._section_page = {"movies": movies_scroll, "calibration": calibration_scroll,
                              "detect": detect_scroll, "track": track_scroll,
                              "batch": batch_scroll}
        col.addWidget(self.pages, 1)
        # THE NEXT BUTTON (asked 13 Oct 2026: "a next button ... when the current screen is
        # ready and done"): one footer under every page, lit when `prep_model.next_step`
        # says the page's work is done, and saying what is left when it is not.
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
        self.resize(1320, 860)

        mp = self.movies_page
        mp.movie_selected.connect(self.show_movie)
        mp.rule_changed.connect(self._rule_changed)
        mp.tolerance_changed.connect(self._tolerance_changed)
        mp.propose_all.connect(self._propose_all)
        mp.propose_every.connect(self._propose_every)
        mp.propose_one.connect(self._propose_one)
        mp.common_start.connect(lambda frame: self._common_cut(start=frame))
        mp.common_end.connect(lambda frame: self._common_cut(end=frame))
        mp.common_length.connect(lambda n: self._common_cut(length=n))
        mp.crop_edited.connect(self._crop_edited)
        mp.purity_requested.connect(self.measure_purity)
        mp.end_all.changed.connect(lambda _v: self._end_changed())
        #: the value "ends at" was last set to by `_follow_longest` (its default until a length is known): a
        #: different value is the user's own
        self._end_auto = mp.end_all.value()
        cp = self.calibration_page
        cp.edited.connect(self._calibration_edited)
        cp.copy_requested.connect(self._copy_calibration)
        cp.save_requested.connect(self.save_condition)
        cp.save_all_requested.connect(self._save_all)
        cp.pixel_helper_requested.connect(self.open_pixel_helper)
        cp.export_requested.connect(self.export_stacks)
        cp.install_requested.connect(self.install_packages)
        cp.export_stop_requested.connect(self.stop_exports)
        cp.export_replace_changed.connect(self._export_replace_changed)
        dp = self.detect_page
        dp.edited.connect(self._tracking_edited)
        dp.measure_requested.connect(self.measure_noise)
        dp.measure_all_requested.connect(self._measure_all)
        dp.copy_requested.connect(self._copy_tracking)
        dp.classic_requested.connect(self.open_classic)
        dp.focus.connect(self._guide_focus)
        dp.pilot_requested.connect(self.measure_pilot)
        dp.radius_check_requested.connect(self.check_wider_radius)
        dp.tracker_changed.connect(self.set_tracker)
        dp.utrack.edited.connect(self._utrack_edited)
        dp.utrack.lab_requested.connect(self._utrack_lab)
        dp.guide.key_changed.connect(lambda _k: self._refresh_detect_guide())
        mp.guide.key_changed.connect(lambda _k: self._refresh_crop_guide())
        tp = self.track_page
        tp.track_requested.connect(self.track)
        tp.track_all_requested.connect(self._track_all)
        tp.stop_requested.connect(self.stop_tracking)
        tp.overwrite_changed.connect(self._overwrite_changed)
        tp.choose_fiji_requested.connect(self.choose_fiji)
        tp.choose_matlab_requested.connect(self.choose_matlab)
        tp.choose_utrack_requested.connect(self.choose_utrack)
        tp.tracker_changed.connect(self.set_tracker)
        tp.done_requested.connect(self.back_to_menu)
        bp = self.batch_page
        bp.add_requested.connect(self.choose_batch_parent)
        bp.remove_requested.connect(self.remove_batch_parent)
        bp.run_requested.connect(self.run_batch)
        bp.stop_requested.connect(self.stop_batch)
        bp.tracker_changed.connect(self.set_tracker)
        bp.replace_changed.connect(self._batch_replace_changed)
        bp.measure_changed.connect(self._batch_measure_changed)
        bp.sessions_changed.connect(self._at_once_changed)
        tp.at_once_changed.connect(self._at_once_changed)
        self._show_at_once()
        bp.set_measure_k(float(self._options["trackmate_threshold_k_tm"]))
        dp.set_tracker(self._tracker)
        tp.set_tracker(self._tracker)
        bp.set_tracker(self._tracker)
        self._show_batch_rows()

        #: The sidebar rows redraw at most every 150 ms while movies stream in.
        self._rows_timer = QTimer(self)
        self._rows_timer.setSingleShot(True)
        self._rows_timer.setInterval(150)
        self._rows_timer.timeout.connect(self._refresh_rows)
        self._update_crumb()

    # -- opening ------------------------------------------------------------------ #
    def choose_folder(self) -> None:
        if self._refuse_while_tracking():
            return
        folder = QFileDialog.getExistingDirectory(
            self, "Open a condition, or a folder of conditions",
            str(self.folder or Path.home()))
        if folder:
            self.open_folder(folder)

    def open_folder(self, folder) -> None:
        """List the folder on a thread; nothing on the GUI thread reads a movie."""
        if self._refuse_while_tracking():
            return
        self._generation += 1
        self.jobs.cancel_pending()
        self.folder = Path(folder)
        self.states = {}
        self.condition = None
        self._auto_propose = set()
        self._notes = {}
        self._measuring = {}
        self._measure_plan, self._measure_lines = {}, []
        for kind in ("read", "measure", "save"):
            self._started.pop(kind, None)
        # Another folder's conditions: what this session tracked belongs to the last one.
        self._run_entries = []
        self._finished = {}
        self.track_page.hide_finished()
        self.sidebar.set_project(self.folder.name, "opening…")
        self.activity.setText(f"listing {self.folder.name}…")
        self.jobs.open_folder(self._generation, self.folder)

    def _opened(self, generation: int, states, error: str) -> None:
        if generation != self._generation:
            return
        if error or not states:
            self.activity.setText("")
            self.sidebar.set_project(self.folder.name if self.folder else "Preparation",
                                     "nothing to prepare")
            self.start_page.text.setText(
                f"{self.folder}: {error or 'no conditions found'}")
            self.pages.setCurrentWidget(self.start_page)
            return
        self.states = dict(states)
        n_movies = sum(len(s.movies) for s in self.states.values())
        from ...prep_model import loose_acquisitions

        loose = ([] if list(self.states) == [self.folder.name]
                 else loose_acquisitions(self.folder))
        if loose:
            self._notes["loose"] = (
                f"{len(loose)} acquisition(s) sit loose in {self.folder.name} beside its "
                f"conditions and were not opened: {', '.join(loose[:4])}")
        # EACH CONDITION'S LAYOUT (`organise.folder_layout`, listed on the opening thread):
        # the word is on its row; a folder not organised is said in full here.
        for name, state in self.states.items():
            note = layout_note(state)
            if note:
                self._notes[f"layout:{name}"] = note
        self._show_notes()
        self.sidebar.set_project(self.folder.name,
                                 f"{len(self.states)} condition(s) · {n_movies} movies")
        self._refresh_rows()
        for name, state in self.states.items():
            bleach = self._bleach_dir(state)
            self.jobs.load_cache(generation, name, state.movies, bleach, self._tol(name))
            # EVERY CONDITION IS PROBED, not only one missing a value: a RECORDED
            # interval is checked against the acquisition too (one metadata read).
            if state.movies:
                self.jobs.probe(generation, name, state.movies[0],
                                state.calibration.get("frame_interval_s"))
        self.show_condition(next(iter(self.states)))
        self.sidebar.select("movies.crop")
        self._activity()

    @staticmethod
    def _bleach_dir(state) -> Path:
        """`bleaching/` beside the results -- or the lab macro's `Bleaching/` (one user's
        folders), so curves the classic window saved there are found."""
        for name in ("bleaching", "Bleaching"):
            if (state.folder / name).is_dir():
                return state.folder / name
        return state.folder / "bleaching"

    def _cached(self, generation: int, condition: str, result, error: str) -> None:
        if generation != self._generation or condition not in self.states:
            return
        state = self.states[condition]
        loaded = (result or ({}, []))[0]
        for stem, curves in loaded.items():
            # after a common cut, a movie arriving from the cache gets the rule's start and the cut
            file_curves(state, stem, refloor(curves, self._tol(condition)),
                        propose_rule=self._rule if (condition in self._auto_propose or state.common) else None,
                        cut=self.movies_page_cut())
        # EVERYTHING THE CACHE DID NOT HOLD IS READ, in the process pool.
        missing = state.unread_named()
        if missing:
            self._started.setdefault("read", time.monotonic())
            self.jobs.read(generation, condition, missing, self._tol(condition))
        else:
            self._auto_flat(condition)
        self._condition_changed(condition)

    def _curve(self, generation: int, condition: str, stem: str, curves, error: str) -> None:
        if generation != self._generation or condition not in self.states:
            return
        state = self.states[condition]
        if curves is not None:
            curves = refloor(curves, self._tol(condition))
        file_curves(state, stem, curves, error,
                    propose_rule=self._rule if (condition in self._auto_propose or state.common) else None,
                    cut=self.movies_page_cut())
        if not state.unread:
            self._auto_flat(condition)
        self._condition_changed(condition, movie=stem)

    def _tol(self, condition) -> float:
        """The flat tolerance a condition's floors are read at: its own, else the box's."""
        state = self.states.get(condition)
        own = getattr(state, "flat_tolerance", None)
        return float(own) if own is not None else self._tolerance

    def _auto_flat(self, condition: str) -> None:
        """Once a condition's movies are read, its own tolerance (`auto_flat_tolerance`),
        unless somebody typed one for it. The box shows it when the condition is on screen."""
        state = self.states.get(condition)
        if state is None or state.flat_typed or state.flat_tolerance is not None or not state.curves:
            return
        tol, words = auto_flat_tolerance(list(state.curves.values()))
        if tol is None:
            state.flat_note = words
        else:
            apply_flat_tolerance(state, tol, note=words)
        if condition == self.condition:
            self._show_crop_standard()

    def _probed(self, generation: int, condition: str, measured, error: str) -> None:
        if generation != self._generation or condition not in self.states or error:
            return
        result = measured or {}
        filled = fill_from_acquisition(self.states[condition], result.get("measured", {}))
        if filled:
            self._notes[f"filled:{condition}"] = (
                f"{condition}: {field_names(filled)} read from the acquisition")
        if result.get("camera"):
            self.states[condition].camera = dict(result["camera"])
        warning = interval_note(condition, result.get("check"))
        if warning:
            self._notes[f"interval:{condition}"] = warning
        self._show_notes()
        self._refresh_calibration()

    def _show_notes(self) -> None:
        """The calibration page's notes are KEPT, one per subject, so a later note cannot
        overwrite an earlier one nobody has read yet (a stray-files note was lost to an
        interval note that way). Warnings first."""
        ordered = sorted(self._notes.items(),
                         key=lambda kv: (not kv[0].startswith(("interval:", "loose",
                                                               "layout:")),
                                         kv[0]))
        self.calibration_page.set_note("\n".join(text for _k, text in ordered))

    # -- showing -------------------------------------------------------------------- #
    def show_condition(self, name: str) -> None:
        if name not in self.states:
            return
        self.condition = name
        self._guide_condition = None
        self.conditions.set_active(name)
        state = self.states[name]
        page = self.movies_page
        page.set_condition(name, condition_status(state))
        page.set_rows(movie_rows(state))
        self._follow_longest(state, force=True)
        current = page.current()
        if current not in state.names:
            current = state.names[0] if state.names else None
        self._show_crop_standard()
        self.show_movie(current)
        self._update_crumb()

    def show_movie(self, stem: str | None) -> None:
        page = self.movies_page
        page.select(stem)
        state = self.states.get(self.condition)
        if state is None or stem is None:
            page.set_chart(None, message="No movies in this condition.")
            page.set_floor("")
            self._refresh_crop_guide()
            return
        curves = state.curves.get(stem)
        if curves is None:
            reason = state.errors.get(stem)
            page.set_chart(None, message=reason or f"Reading {stem}…")
            page.set_floor("")
        else:
            page.set_chart(crop_chart(curves, crop=state.crops.get(stem),
                                      proposal=state.proposals.get(stem)))
            page.set_floor(floor_note(curves))
        self._update_crumb()
        self._refresh_crop_guide()

    def _condition_changed(self, condition: str, movie: str | None = None) -> None:
        """One condition's state moved: its rows, and its chart if it is on screen."""
        self._rows_timer.start()
        if condition != self.condition:
            return
        state = self.states[condition]
        self.movies_page.set_condition(condition, condition_status(state))
        self.movies_page.set_rows(movie_rows(state))
        self._follow_longest(state)
        current = self.movies_page.current()
        if current is None and state.names:
            self.show_movie(state.names[0])
        elif movie is None or movie == current:
            self.show_movie(current)
        self._activity()

    def _follow_longest(self, state, *, force: bool = False) -> None:
        """"ends at" stands at the condition's LONGEST movie, and follows it as the movies are read (tester B16,
        3 Nov 2026: set at the open, before any length was known, it stayed at its 4000 default and "Cut all tails"
        said "now end at frame 4000; 3 end earlier" on 900-frame movies) -- until the user types one: a value that
        is no longer the one set here is the user's, and is left alone. ``force`` (a condition shown) sets it."""
        page = self.movies_page
        lengths = list(state.n_frames().values())
        if not lengths:
            return
        auto = getattr(self, "_end_auto", None)
        if not force and auto is not None and page.end_all.value() != auto:
            return                                        # typed: the user's
        longest = int(max(lengths))
        self._end_auto = longest
        if page.end_all.value() == longest:
            return
        if force:
            page.end_all.setValue(longest)
        else:
            page.end_all.blockSignals(True)               # a length arriving is not the user pointing at "end"
            try:
                page.end_all.setValue(longest)
            finally:
                page.end_all.blockSignals(False)

    def _refresh_rows(self) -> None:
        self.conditions.set_rows([sidebar_row(s) for s in self.states.values()])
        self.conditions.set_active(self.condition)
        self._refresh_calibration()
        self._refresh_next()

    def _refresh_next(self) -> None:
        """The footer for the page on screen (only its own check runs)."""
        if not self.states or self.pages.currentWidget() is self.start_page:
            self.next_bar.setVisible(False)
            return
        step = next_step(self.sidebar.current.split(".")[0], self.states,
                         tracker=self._tracker, utrack_settings=self._utrack_settings,
                         busy=self._busy(), finished=self._finished)
        if step is None:                               # a page without one (Batch)
            self.next_bar.setVisible(False)
            return
        self._next_target = step["target"]
        self.next_button.set_text(step["label"])
        self.next_button.setEnabled(step["ready"])
        self.next_note.setText("" if step["ready"] else step["why"])
        self.next_bar.setVisible(True)

    def _next_clicked(self) -> None:
        target = getattr(self, "_next_target", None)
        if target:
            self.sidebar.select_section(target)
        else:
            self.back_to_menu()

    def _refresh_calibration(self) -> None:
        rows = []
        for name, state in self.states.items():
            status, ready = save_state(state)
            rows.append({"name": name, "cells": calibration_cells(state),
                         "status": status, "ready": ready})
        self.calibration_page.set_rows(rows)
        self._refresh_missing()
        self._refresh_export()
        self._refresh_detect()

    def _refresh_missing(self) -> None:
        """Warn, BEFORE a Save, about a reader this install lacks (`optional_deps`).

        Asked of the acquisitions actually loaded, so a MetaMorph folder is never told to
        install the Nikon reader. Before this, the first sign was a Save that wrote no
        projection and left an empty `sum/` -- which `channels.resolve_channels` then read
        as a resolved `sum` channel with no fields, so the classical segmenter silently
        found nothing. One missing package, two symptoms, neither naming it.
        """
        from ...optional_deps import can_install, describe_missing, missing_for_movies

        movies = [m for state in self.states.values() for m in state.movies]
        missing = missing_for_movies(movies) if movies else []
        offer, _why = can_install()
        self.calibration_page.set_missing(missing, describe_missing(missing),
                                          can_install=offer,
                                          busy=bool(self._installing))

    def install_packages(self, packages: list) -> None:
        """Run the offered install on a THREAD, streaming pip's output to the card.

        It reaches the network and takes a minute, so it cannot run on the GUI thread --
        and a silent minute is the shape this project keeps having to fix. When it
        finishes the warning is re-derived rather than cleared: the question is whether
        THIS interpreter can import it now, which `optional_deps.install` checks itself.
        """
        from ...optional_deps import install

        if self._installing or not packages:
            return
        self._installing = list(packages)
        self._install_lines = []
        self.calibration_page.set_install_log("starting…")
        self._refresh_missing()

        def on_line(line: str) -> None:
            self._install_lines.append(line)
            self.jobs.signals.install_line.emit(self._generation, line)

        self.jobs.install(self._generation, list(packages), install, on_line)

    def _install_line(self, generation: int, line: str) -> None:
        if generation != self._generation:
            return
        self.calibration_page.set_install_log(line[:200])

    def _install_done(self, generation: int, ok: bool, words: str) -> None:
        if generation != self._generation:
            return
        self._installing = []
        self.calibration_page.set_install_log(("" if ok else "Failed: ") + words)
        self._refresh_missing()

    def _refresh_export(self) -> None:
        running = next(iter(self._exporting), None)
        done = total = 0
        for expected, rows in self._exporting.values():
            done, total = done + len(rows), total + expected
        self.calibration_page.set_export(
            export_rows(self.states, overwrite=self._export_replace, running=running,
                        progress=(len(self._exporting[running][1]),
                                  self._exporting[running][0]) if running else None,
                        queued=tuple(list(self._exporting)[1:])),
            busy=bool(self._exporting), progress=(done, total) if total else None)
        self._refresh_progress()

    def _refresh_detect(self) -> None:
        """The tracking grid and the lines under it, for every condition. The spot radius
        is re-derived here, so a calibration filled later (the probe, a typed emission)
        reaches it -- unless the radius was typed."""
        rows, lines = [], []
        for name, state in self.states.items():
            seed_tracking(state, self._tracking_base())
            rows.append({"name": name, "cells": tracking_cells(state),
                         "status": tracking_status(state),
                         "measuring": name in self._measuring})
            lines += reach_lines(state)
            crowd = crowding_line(state)
            if crowd:
                lines.append(crowd)
            if state.outlines is not None:
                lines.append(f"{name}:{outline_note(state.outlines)}")
            lines += noise_lines(state)
        self.detect_page.set_rows(rows)
        self.detect_page.set_lines(lines)
        self._refresh_detect_guide()
        self._refresh_utrack_card()
        self._refresh_track()

    def _refresh_utrack_card(self) -> None:
        """The u-track settings and the per-condition table -- only while u-track is the
        tracker, since the table asks each condition whether it could run."""
        if self._tracker != "utrack":
            return
        card = self.detect_page.utrack
        card.set_values(utrack_cells(self._utrack_settings),
                        utrack_differences(self._utrack_settings))
        card.set_rows(utrack_condition_rows(self.states, self._utrack_settings))

    # -- the guides ---------------------------------------------------------------------- #
    def _guide_focus(self, condition: str, key: str) -> None:
        self._guide_condition = condition
        self.detect_page.guide.show_key(key)
        self._refresh_detect_guide()

    def _refresh_detect_guide(self) -> None:
        """The tracking guide, for the condition clicked into or picked. Its values are
        always written; its picture is built only while it is open."""
        guide = self.detect_page.guide
        name = (self._guide_condition if self._guide_condition in self.states
                else self.condition)
        state = self.states.get(name)
        guide.set_values({k: prep_guide.value_text(k, state) if state else ""
                          for k in guide.keys})
        if not guide.is_open:
            return
        guide.set_text(prep_guide.guide_text(guide.key))
        guide.set_payload(prep_guide.picture(guide.key, state))
        if name:
            guide.title.setText(f"{prep_guide.NAMES[guide.key]}: {name}")

    def _refresh_crop_guide(self) -> None:
        """The crop guide, for the movie on the chart: its own curve and floors once it
        is read, an invented movie before."""
        guide = self.movies_page.guide
        end = self.movies_page.end_all.value()
        guide.set_values({k: prep_guide.value_text(k, None, rule=self._rule,
                                                   tolerance=self._tol(self.condition), end=end)
                          for k in guide.keys})
        if not guide.is_open:
            return
        state = self.states.get(self.condition)
        stem = self.movies_page.current()
        curves = state.curves.get(stem) if (state is not None and stem) else None
        guide.set_text(prep_guide.guide_text(guide.key))
        guide.set_payload(prep_guide.picture(guide.key, curves=curves, rule=self._rule,
                                             tolerance=self._tol(self.condition), end=end,
                                             purity=self._purity.get(self.condition)))

    def _track_rows(self, **state) -> list:
        """The Track page's rows for the tracker in use."""
        if self._tracker == "utrack":
            return utrack_track_rows(self.states, self._utrack_settings, at_once=self._at_once, **state)
        return track_rows(self.states, at_once=self._at_once, **state)

    def _refresh_track(self) -> None:
        rows = self._track_rows(running=self._tracking, progress=self._track_done,
                                queued=self._track_queue, overwrite=self._overwrite,
                                finished=self._finished)
        self.track_page.set_rows(rows, busy=self._tracking is not None)
        # "no condition is ready" is true only until one is: pressed while the thresholds were still being
        # measured, it stayed under rows that had since turned ready (seen 26 Nov 2026)
        if self.track_page.note.text() == NOT_READY_NOTE and any(r["ready"] for r in rows):
            self.track_page.set_note("")
        self._refresh_progress()
        self._refresh_next()

    def _activity(self) -> None:
        reading = self.jobs.reading
        self.activity.setText(f"reading {reading} movie(s)…" if reading else "")
        if not reading and "read" in self._started:
            counts = [s.counts() for s in self.states.values()]
            bad = sum(c.n_bad for c in counts)
            self._job_finished(
                "read", "All movies read",
                [f"{sum(c.n_curves for c in counts)} movie(s) in {len(counts)} condition(s) "
                 f"read; their curves are on Movies & crop.",
                 f"{bad} could not be read; each row says why." if bad else ""],
                failed=bool(bad))
        self._refresh_progress()

    # -- the progress card and the finished pop-up ----------------------------------- #
    def _refresh_progress(self) -> None:
        payload = self._progress_payload()
        self.run_progress.set_payload(payload)
        if payload and not self._progress_timer.isActive():
            self._progress_timer.start()
        elif not payload and self._progress_timer.isActive():
            self._progress_timer.stop()

    def _progress_payload(self) -> dict | None:
        """The job the card shows, the longest first: a batch, tracking, a radius check,
        stack export, threshold measuring, saving, reading movies."""
        if self._batch_stop is not None:
            engine = "u-track" if self._batch_tracker == "utrack" else "TrackMate"
            return batch_progress(
                title=f"Batch: tracking with {engine}", plan=self._batch_plan,
                done=self._batch_counts, finished=self._batch_finished,
                running=self._batch_running, started=self._started.get("batch"))
        if self._tracking is not None:
            index, n = self._track_done or (0, self._run_plan.get(self._tracking, 0))
            done = sum(self._run_plan.get(k, 0) for k in self._run_done) + index
            engine = "u-track" if self._run_tracker == "utrack" else "TrackMate"
            return progress_payload(
                title=f"Tracking with {engine}", unit="movie", done=done,
                total=sum(self._run_plan.values()), group="condition",
                groups_done=len(self._run_done), groups_total=len(self._run_plan),
                current=self._tracking, current_done=index, current_total=n,
                started=self._started.get("track"))
        if self._checking is not None:
            index, n = self._check_done or (0, 0)
            return progress_payload(
                title=f"Checking a wider radius on {self._checking}", unit="run",
                done=index, total=n, started=self._started.get("check"))
        if self._exporting:
            current = next(iter(self._exporting))
            expected, rows = self._exporting[current]
            finished = [k for k in self._export_plan if k not in self._exporting]
            done = (sum(self._export_plan[k] for k in finished)
                    + sum(len(r) for _e, r in self._exporting.values()))
            return progress_payload(
                title="Writing cropped stacks", unit="stack", done=done,
                total=sum(self._export_plan.values()), group="condition",
                groups_done=len(finished), groups_total=len(self._export_plan),
                current=current, current_done=len(rows), current_total=expected,
                started=self._started.get("export"))
        if self._measuring:
            current = next(iter(self._measuring))
            expected, rows, _radius, _k = self._measuring[current]
            finished = [k for k in self._measure_plan if k not in self._measuring]
            done = (sum(self._measure_plan[k] for k in finished)
                    + sum(len(r) for _e, r, *_x in self._measuring.values()))
            return progress_payload(
                title="Measuring the detection threshold", unit="movie", done=done,
                total=sum(self._measure_plan.values()), group="condition",
                groups_done=len(finished), groups_total=len(self._measure_plan),
                current=current, current_done=len(rows), current_total=expected,
                started=self._started.get("measure"))
        if self._saving:
            condition, done, total, what = self._save_counts or (self._saving[0], 0, 0, "")
            return progress_payload(
                title=f"Saving {len(self._saving)} condition(s)", unit="file", done=done,
                total=total, current=f"{condition}{f' ({what})' if what else ''}",
                started=self._started.get("save"))
        if self.jobs.reading:
            counts = {n: s.counts() for n, s in self.states.items()}
            total = sum(c.n_movies for c in counts.values())
            done = sum(min(c.n_curves + c.n_bad, c.n_movies) for c in counts.values())
            complete = sum(1 for c in counts.values() if c.n_curves + c.n_bad >= c.n_movies)
            return progress_payload(
                title="Reading movies", unit="movie", done=done, total=total,
                group="condition", groups_done=complete, groups_total=len(counts),
                started=self._started.get("read"))
        return None

    def _job_finished(self, kind: str, title: str, lines, *, failed: bool = False) -> None:
        """A long job ended: a pop-up when it ran at least `_notify_after` seconds."""
        started = self._started.pop(kind, None)
        show_finished(self, self.t, finished_payload(
            title=title, lines=lines, started=started, notify_after_s=self._notify_after,
            failed=failed))
        self._refresh_progress()

    def _on_section(self, view: str) -> None:
        section = view.split(".")[0]
        if section == "batch":
            # NEEDS NO OPEN FOLDER, and is planned afresh on arrival: a save on another
            # page, or a new standard, changes what would run.
            self.pages.setCurrentWidget(self._section_page[section])
            self._plan_batch()
        elif not self.states:
            self.pages.setCurrentWidget(self.start_page)
        else:
            self.pages.setCurrentWidget(self._section_page[section])
        self._update_crumb()
        self._refresh_next()

    def _update_crumb(self) -> None:
        section = self.sidebar.current.split(".")[0]
        trail = [_TITLES[section]]
        if section == "batch":
            self.breadcrumb.set_trail(trail)
            return
        if self.condition:
            trail.append(self.condition)
            stem = self.movies_page.current()
            if section == "movies" and stem:
                trail.append(stem)
        self.breadcrumb.set_trail(trail)

    # -- editing ----------------------------------------------------------------------- #
    def movies_page_cut(self) -> dict:
        """The proposal keywords: the rule's start, the whole tail (common cuts are
        applied with their own buttons)."""
        return {}

    def _end_changed(self) -> None:
        self.movies_page.guide.show_key("end")
        self._refresh_crop_guide()

    def measure_purity(self) -> None:
        """The tail's purity for the condition on screen, from its tracks and outlines."""
        state = self.states.get(self.condition)
        if state is None:
            return
        crops = dict(state.saved) or dict(state.crops)
        self.movies_page.guide.set_result(f"reading the tracks and outlines of {state.name}…")
        self.jobs.tail_purity(self._generation, state.name, state.folder, crops)

    def _purity_done(self, generation: int, condition: str, got, error: str) -> None:
        if generation != self._generation:
            return
        if error or got is None:
            self.movies_page.guide.set_result(f"{condition}: {error.split(': ', 1)[-1]}")
            return
        self._purity[condition] = got
        skipped = f" ({got['skipped_note']})" if got.get("skipped_note") else ""
        self.movies_page.guide.set_result(
            f"{condition}: measured on {len(got['movies'])} movie(s){skipped}")
        self._refresh_crop_guide()

    def _rule_changed(self, rule: str) -> None:
        self._rule = rule
        self.movies_page.guide.show_key("start_rule")
        self._refresh_crop_guide()

    def _standard_tolerance(self, tol: float) -> None:
        """The Options' flat tolerance: the STANDARD for every condition without its own (none read yet, or none
        the automatic rule could set), whose held floors follow it; a condition's own, automatic or typed, stays.
        Found 23 Oct 2026: after each condition got its own, saving the Options sent the new standard down the
        TYPED path, onto the condition on screen only, and the standard itself never moved."""
        self._tolerance = float(tol)
        for state in self.states.values():
            if getattr(state, "flat_tolerance", None) is None:
                for stem, curves in list(state.curves.items()):
                    state.curves[stem] = refloor(curves, self._tolerance)
        if self.condition:
            self.show_movie(self.movies_page.current())

    def _tolerance_changed(self, percent: float) -> None:
        """The condition on screen's floors at the typed tolerance -- two arrays a movie, no
        read. Each condition keeps its own (23 Oct 2026: "inbetween conditions can be
        sometimes very different"); the page sends this once typing pauses, not per key."""
        self.movies_page.guide.show_key("flat_tolerance")
        state = self.states.get(self.condition)
        if state is None:
            self._tolerance = float(percent) / 100.0
            return
        apply_flat_tolerance(state, float(percent) / 100.0, typed=True, note="typed for this condition")
        self._show_crop_standard()
        self._condition_changed(self.condition)

    def _propose_all(self) -> None:
        """'Propose for this condition': every movie of the condition on screen."""
        state = self.states.get(self.condition)
        if state is None:
            return
        self._auto_propose.add(self.condition)
        done, failed = propose_all(state, self._rule)
        short = short_crop_note({stem: state.crops[stem] for stem in done if stem in state.crops})
        self.movies_page.set_note(propose_note(done, failed, self._rule,
                                               pending=len(state.unread)) + (f". {short}" if short else ""))
        self._condition_changed(self.condition)

    def _propose_every(self) -> None:
        """'Propose for all conditions': each condition at its OWN tolerance."""
        done_all, failed_all, pending, ranges = [], [], 0, {}
        for name, state in self.states.items():
            self._auto_propose.add(name)
            done, failed = propose_all(state, self._rule)
            done_all += done
            failed_all += [(f"{name}/{stem}", why) for stem, why in failed]
            pending += len(state.unread)
            ranges.update({f"{name}/{stem}": state.crops[stem] for stem in done if stem in state.crops})
        short = short_crop_note(ranges)
        self.movies_page.set_note(propose_note(done_all, failed_all, self._rule, pending=pending)
                                  + f" (in {len(self.states)} condition(s), each at its own flat tolerance)"
                                  + (f". {short}" if short else ""))
        for name in self.states:
            self._condition_changed(name)

    def _propose_one(self, stem: str) -> None:
        state = self.states.get(self.condition)
        if state is None:
            return
        try:
            propose(state, stem, self._rule)
        except PrepError as exc:
            self.movies_page.set_note(str(exc))
            return
        self.movies_page.set_note(short_crop_note({stem: state.crops[stem]}) if stem in state.crops else "")
        self._condition_changed(self.condition, movie=stem)

    def _common_cut(self, **edge) -> None:
        """A common start or end for EVERY condition of the session, as asked for the
        classic window; the note names what the condition on screen did."""
        notes = {}
        for name, state in self.states.items():
            # a read movie with no range yet starts where the rule puts it, never at 1
            notes[name] = common_cut_note(apply_common_cut(state, start_rule=self._rule,
                                                           cut=self.movies_page_cut(), **edge))
        if self.condition in notes:
            others = len(notes) - 1
            self.movies_page.set_note(notes[self.condition] + (
                f" The same in {others} other condition(s)." if others else ""))
        for name in self.states:
            self._condition_changed(name)

    def _crop_edited(self, stem: str, start: int, end: int) -> None:
        state = self.states.get(self.condition)
        if state is None:
            return
        try:
            set_crop(state, stem, start, end)
        except PrepError as exc:
            self.movies_page.set_note(str(exc))
            self.show_movie(stem)
            return
        # A VALID EDIT CLEARS THE OLD REFUSAL (tester B17, 3 Nov 2026: "-5-99999 is not a range" stayed after a valid
        # range), and a short range is named, never refused
        self.movies_page.set_note(short_crop_note({stem: state.crops[stem]}))
        state.proposals.pop(stem, None)
        self._condition_changed(self.condition, movie=stem)

    def _calibration_edited(self, condition: str, key: str, text: str) -> None:
        state = self.states.get(condition)
        if state is None:
            return
        try:
            note = set_calibration(state, key, text)
        except PrepError as exc:
            note = str(exc)
        # always: a valid value clears the last refusal (tester B17), an unusual one is said (`plausibility_note`)
        self.calibration_page.set_note(note)
        self._refresh_rows()

    def _copy_calibration(self, source: str) -> None:
        try:
            changed = copy_calibration(self.states, source)
        except PrepError as exc:
            self.calibration_page.set_note(str(exc))
            return
        self.calibration_page.set_note(
            f"{source}'s calibration copied to {len(changed)} condition(s)"
            if changed else "every condition already matches")
        self._refresh_rows()

    # -- detection and linking -------------------------------------------------------- #
    def _tracking_edited(self, condition: str, key: str, text: str) -> None:
        state = self.states.get(condition)
        if state is None:
            return
        try:
            note = set_tracking(state, key, text, self._tracking_base())
        except PrepError as exc:
            note = str(exc)
        self.detect_page.set_note(note)
        self._refresh_detect()

    def _copy_tracking(self, source: str) -> None:
        changed = copy_tracking(self.states, source)
        self.detect_page.set_note(
            f"{source}'s linking radius and frame gap copied to {len(changed)} "
            f"condition(s)" if changed else "every condition already matches")
        self._refresh_detect()

    def measure_noise(self, name: str) -> None:
        """Measure one condition's LoG noise inside its crops, one movie per worker."""
        state = self.states.get(name)
        if state is None or name in self._measuring:
            return
        items = measurable(state)
        if not items:
            self.detect_page.set_note(
                f"{name}: every movie needs a range first; the threshold is measured "
                f"inside the crop")
            return
        radius = state.tracking.get("radius_px")
        if not radius:
            self.detect_page.set_note(
                f"{name}: no spot radius: set the emission, NA and pixel size on "
                f"Calibration & save, or type a radius")
            return
        k = float(self._options["trackmate_threshold_k_tm"])
        self._measuring[name] = (len(items), {}, float(radius), k)
        self._measure_plan[name] = len(items)
        self._started.setdefault("measure", time.monotonic())
        self.jobs.measure_noise(self._generation, name, items, float(radius), k)
        self.detect_page.set_note(f"measuring {name}: 0/{len(items)} movies")
        self._refresh_detect()

    def _measure_all(self) -> None:
        for name in self.states:
            self.measure_noise(name)

    def _noise_row(self, generation: int, condition: str, stem: str, row, error: str) -> None:
        if generation != self._generation or condition not in self._measuring:
            return
        expected, rows, radius, k = self._measuring[condition]
        rows[stem] = row if row is not None else {
            "movie": stem, "sigma_log": None, "threshold": None, "ok": False,
            "error": error}
        self.detect_page.set_note(f"measuring {condition}: {len(rows)}/{expected} movies")
        if len(rows) < expected:
            self._refresh_progress()
            return
        del self._measuring[condition]
        state = self.states[condition]
        ordered = [rows[s] for s in state.names if s in rows]
        try:
            record_noise(state, summarise_noise(ordered, radius_px=radius, k=k))
            note = f"{condition}: threshold measured"
        except PrepError as exc:
            note = f"{condition}: {exc}"
        self.detect_page.set_note(note)
        self._measure_lines.append(note)
        if not self._measuring:
            lines, self._measure_lines, self._measure_plan = self._measure_lines, [], {}
            self._job_finished("measure", "Detection thresholds measured", lines,
                               failed=not all(x.endswith("threshold measured") for x in lines))
        self._refresh_detect()

    # -- saving -------------------------------------------------------------------------- #
    def save_condition(self, name: str) -> None:
        state = self.states.get(name)
        if state is None:
            return
        self._saving.append(name)
        self._started.setdefault("save", time.monotonic())
        self.jobs.save(self._generation, state,
                       projection=self.calibration_page.projection_kind())
        self.calibration_page.set_note(f"saving {name}…")

    def _save_all(self) -> None:
        ready = [name for name, state in self.states.items() if save_state(state)[1]]
        if not ready:
            self.calibration_page.set_note("nothing is ready to save yet")
            return
        for name in ready:
            self.save_condition(name)

    def _save_progress(self, generation, condition, done, total, what) -> None:
        if generation == self._generation:
            self.calibration_page.set_note(f"saving {condition}: {what} {done}/{total}")
            self._save_counts = (condition, int(done), int(total), str(what))
            self._refresh_progress()

    def _saved(self, generation: int, condition: str, path, error: str) -> None:
        if condition in self._saving:
            self._saving.remove(condition)
        if generation != self._generation:
            return
        line = f"{condition}: {error}" if error else f"{condition}: saved to {path}"
        self.calibration_page.set_note(line)
        self._save_lines.append(line)
        if not self._saving:
            lines, self._save_lines, self._save_counts = self._save_lines, [], None
            self._job_finished("save", "Saved", lines,
                               failed=any("saved to" not in x for x in lines))
        self._condition_changed(condition)
        self._refresh_rows()

    # -- Fiji ------------------------------------------------------------------------ #
    def find_fiji(self) -> None:
        """Look for Fiji off the window's thread: a folder chosen earlier is asked first
        (remembered per user), then `fiji.find_fiji`'s usual places."""
        # THE REMEMBERED FOLDER STAYS BEHIND A NEW PICK: a picked folder holding no Fiji
        # must leave the working install in use, not fall through to the default search.
        paths = [Path(p) for p in (self._fiji_wanted, theme.remembered("fiji_folder")) if p]
        self.jobs.find_fiji(tuple(dict.fromkeys(paths)))

    def _fiji_found(self, install, absence: str, error: str) -> None:
        wanted, self._fiji_wanted = self._fiji_wanted, None
        if error:
            self.track_page.set_note(f"looking for Fiji failed: {error}")
        self._install, self._fiji_absence = install, absence
        lines = fiji_lines(install, absence=absence)
        if wanted is not None:
            from ...fiji import launcher_hint, resolve_root

            roots = [] if install is None else [install.root, *install.other_candidates]
            hit = resolve_root(wanted)
            if hit is not None and hit[0] in roots:
                # The FIJI found there is remembered, not the folder picked: a download's
                # outer folder resolves to the Fiji inside it.
                theme.remember("fiji_folder", str(hit[0]))
            else:
                # THE CHOICE MUST BE SEEN TO TAKE EFFECT: `find_fiji` given a folder with
                # no Fiji in it falls through to its own search and returns ANOTHER
                # install, which would read as success under the new choice.
                lines.insert(0, f"No Fiji in {wanted}, so it was not used. Choose "
                                f"{launcher_hint()}.")
        self.track_page.set_fiji(lines)
        self._refresh_track()

    def choose_fiji(self) -> None:
        picked = QFileDialog.getExistingDirectory(self, "The Fiji folder",
                                                  str(theme.remembered("fiji_folder") or ""))
        if picked:
            self._fiji_wanted = Path(picked)
            self.find_fiji()

    # -- tracking ---------------------------------------------------------------------- #
    def _busy(self) -> str:
        """What is running that a folder change or leaving would cut short, or ``""``."""
        if self._batch_stop is not None:
            return "a batch is running"
        if self._tracking is not None:
            return f"tracking {self._tracking}"
        if self._checking is not None:
            return f"checking a wider radius on {self._checking}"
        if self._utrack_checking:
            return "MATLAB is starting"
        if self._exporting:
            return f"exporting stacks for {', '.join(self._exporting)}"
        return ""

    def _refuse_while_tracking(self) -> bool:
        busy = self._busy()
        if not busy:
            return False
        self.activity.setText(f"{busy}; stop it before opening another folder")
        return True

    def back_to_menu(self) -> None:
        busy = self._busy()
        if busy:
            self.activity.setText(f"{busy}; stop it before leaving")
            return
        self.return_to_menu = True
        self.close()

    def _overwrite_changed(self, on: bool) -> None:
        self._overwrite = bool(on)
        self._refresh_track()

    def track(self, name: str) -> None:
        """Track one condition now, or queue it behind the run in flight."""
        if name not in self.states:
            return
        if self._batch_stop is not None:
            # ONE FIJI, AND MATLAB SESSIONS ALREADY CHOSEN: a run beside a batch would
            # compete with it for both.
            self.track_page.set_note(f"a batch is running; track {name} once it is done, "
                                     f"or stop it on Batch")
            return
        if self._tracking is not None or self._utrack_checking:
            # BEHIND A RUN, OR BEHIND MATLAB STARTING for the u-track check: in order, so a
            # second press during the start does not jump ahead of the first.
            if name != self._tracking and name not in self._track_queue:
                self._track_queue.append(name)
                self._run_plan.setdefault(name, len(self.states[name].movies))
                self.track_page.set_note(f"{name} queued behind {self._tracking}"
                                         if self._tracking else
                                         f"{name} queued; MATLAB is starting")
            self._refresh_track()
            return
        if not self._start_track(name):
            self._refresh_track()

    def _track_all(self) -> None:
        names = [row["name"] for row in self._track_rows(overwrite=self._overwrite)
                 if row["ready"]]
        if not names:
            self.track_page.set_note(NOT_READY_NOTE)
            return
        movies = sum(len(self.states[n].movies) for n in names)
        self.track_page.append(f"Track all: {len(names)} condition(s), {movies} "
                               f"movie(s): " + ", ".join(names))
        for name in names:
            self.track(name)

    def _start_track(self, name: str) -> bool:
        """Start one condition's run. ``False``, with the reason logged, if it cannot."""
        if self._tracker == "utrack":
            return self._start_utrack(name)
        self._run_tracker = "trackmate"
        install = self._install
        if install is None or not install.has_trackmate:
            self.track_page.set_note("Fiji with TrackMate is not set up yet: see the Fiji "
                                     "card above")
            self._track_queue.clear()
            return False
        if self._checking is not None:
            self.track_page.set_note(f"a wider-radius check is running on {self._checking}; "
                                     f"track once it is done")
            self._track_queue.clear()
            return False
        state = self.states[name]
        try:
            inputs = run_inputs(state)
        except PrepError as exc:
            self.track_page.append(f"not tracked: {exc}")
            return False
        if state.earlier_run and not self._overwrite:
            self.track_page.append(
                f"not tracked: {name}: trackmate/ already holds {state.earlier_run} track "
                f"file(s) from an earlier run. Tick Replace, or move them; two runs "
                f"in one folder cannot be told apart afterwards")
            return False
        self.track_page.append("", *run_header(inputs))
        self.track_page.hide_finished()
        self._started.setdefault("track", time.monotonic())
        self._run_plan.setdefault(name, len(inputs.movies))
        self._tracking = name
        self._track_done = (0, len(inputs.movies))
        self._stop = threading.Event()
        self.jobs.track(inputs, install, self._stop.is_set, at_once=self._at_once)
        self.track_page.set_note(f"tracking {name}…")
        self._refresh_track()
        return True

    def _next_track(self) -> None:
        while self._track_queue:
            if self._start_track(self._track_queue.pop(0)):
                return
        self._refresh_track()
        if self._tracking is None and self._run_entries:
            self._run_finished()
        elif self._tracking is None:
            # nothing in the queue could start: no run happened, nothing to announce
            self._run_plan, self._run_done = {}, []
            self._started.pop("track", None)
            self._refresh_progress()

    def _run_finished(self) -> None:
        """The last queued condition is done: say so where it cannot be missed -- the
        banner on the Track page, the header line seen from every page, and the taskbar,
        which flashes when the window is behind another program."""
        banner = finished_banner(self._run_entries, tracker=self._run_tracker)
        self._run_entries = []
        self.track_page.show_finished(banner)
        self.track_page.set_note("")
        self.activity.setText(f"{banner['title']}; see Track")
        QApplication.alert(self)
        self._run_plan, self._run_done = {}, []
        self._job_finished("track", banner["title"], banner["lines"],
                           failed=banner["title"] != "Tracking finished")

    def stop_tracking(self) -> None:
        """Stop after the movie in hand, and drop the queue."""
        if self._stop is None:
            return
        self._stop.set()
        dropped, self._track_queue = len(self._track_queue), []
        self.track_page.set_note(
            f"stopping {self._tracking} after the movie in hand"
            + (f"; {dropped} queued condition(s) dropped" if dropped else ""))
        self._refresh_track()

    def _track_progress(self, name: str, index: int, total: int, row) -> None:
        if name != self._tracking:
            return
        self._track_done = (index, total)
        line = (utrack_progress_line if self._run_tracker == "utrack" else progress_line)
        self.track_page.append(line(index, total, row))
        self._refresh_track()

    def _tracked(self, name: str, result, error: str) -> None:
        if name != self._tracking:
            return
        self._tracking, self._track_done, self._stop = None, None, None
        entry = run_entry(name, result, error)
        self._run_entries.append(entry)
        self._run_done.append(name)
        self._finished[name] = entry
        if error or result is None:
            self.track_page.append(f"Tracking failed for {name}: {error}")
        else:
            self.track_page.append("", result["verdict"], "", result["summary"],
                                   f"run record: {result['record']}")
            if name in self.states and self._run_tracker == "trackmate":
                self.states[name].earlier_run = int(result["track_files"])
        self.track_page.set_note(f"{name} done")
        self._next_track()

    # -- u-track ------------------------------------------------------------------- #
    def set_tracker(self, key: str) -> None:
        """Make ``key`` ("trackmate" | "utrack") the tracker of both pages. Refused while
        a run or a check is going: its rows and banner belong to the tracker it started
        with."""
        if key == self._tracker:
            return
        busy = self._busy()
        if (self._tracking is not None or self._checking is not None
                or self._batch_stop is not None or self._utrack_checking):
            self.detect_page.set_tracker(self._tracker)
            self.track_page.set_tracker(self._tracker)
            self.batch_page.set_tracker(self._tracker)
            self.activity.setText(f"{busy}; the tracker can change once it is done")
            return
        self._tracker = key
        theme.remember("prep_tracker", key)
        self.detect_page.set_tracker(key)
        self.track_page.set_tracker(key)
        self.batch_page.set_tracker(key)
        self._show_at_once()
        self.detect_page.set_note("")
        if key == "utrack" and self._matlab is None and self._utrack is None:
            self.find_utrack()
        self._refresh_detect()
        if self.sidebar.current.startswith("batch"):
            self._plan_batch()

    def find_utrack_if_wanted(self) -> None:
        if self._tracker == "utrack":
            self.find_utrack()

    def find_utrack(self) -> None:
        """Look for MATLAB and u-track off the window's thread: paths picked earlier first
        (remembered per user), then `utrack_run`'s usual places."""
        if self._utrack_searching:
            return
        self._utrack_searching = True
        matlab = [Path(p) for p in (self._matlab_wanted, theme.remembered("matlab_path")) if p]
        utrack = [Path(p) for p in (self._utrack_wanted, theme.remembered("utrack_folder"))
                  if p]
        self.track_page.set_utrack(["looking for MATLAB and u-track…"])
        self.jobs.find_utrack(tuple(dict.fromkeys(matlab)), tuple(dict.fromkeys(utrack)))

    def _utrack_found(self, matlab, utrack, absence: str, error: str) -> None:
        from ...utrack_run import resolve_utrack

        self._utrack_searching = False
        wanted_m, self._matlab_wanted = self._matlab_wanted, None
        wanted_u, self._utrack_wanted = self._utrack_wanted, None
        self._matlab, self._utrack, self._utrack_absence = matlab, utrack, absence
        self._utrack_health, self._utrack_error = None, error
        lines = utrack_engine_lines(matlab, utrack, error=error, absence=absence)
        # A PICK THAT HOLDS NOTHING IS SAID AND NOT REMEMBERED (Fiji's rule): the search
        # falls through to its own places and would otherwise read as the pick working.
        if wanted_m is not None:
            if matlab is not None and Path(matlab.executable) == Path(wanted_m):
                theme.remember("matlab_path", str(matlab.executable))
            else:
                lines.insert(0, f"No MATLAB at {wanted_m}, so it was not used. Choose "
                                f"matlab.exe in a release's bin folder.")
        if wanted_u is not None:
            hit = resolve_utrack(wanted_u)
            if utrack is not None and hit is not None and Path(hit) == Path(utrack.software):
                theme.remember("utrack_folder", str(utrack.software))
            else:
                lines.insert(0, f"No u-track in {wanted_u}, so it was not used. Choose the "
                                f"folder holding software/@UTrackPackage.")
        self.track_page.set_utrack(lines)
        # MATLAB IS NOT STARTED HERE: finding it is a path search; starting it is the
        # health check, run when tracking begins (`_check_utrack_then`).
        self._refresh_track()

    def _check_utrack_then(self, after: str) -> None:
        """Start MATLAB for the u-track check, say so, and resume ``after`` ("track" |
        "batch") when it answers (`_utrack_health_done`)."""
        self._after_check = after
        if self._utrack_checking:
            return
        self._utrack_checking = True
        note = UTRACK_STARTING
        (self.batch_page if after == "batch" else self.track_page).set_note(note)
        self.activity.setText(note)
        self.track_page.set_utrack(utrack_engine_lines(
            self._matlab, self._utrack, absence=self._utrack_absence, checking=True))
        self.jobs.check_utrack(self._matlab, self._utrack)

    def _needs_utrack_check(self) -> bool:
        return (self._matlab is not None and self._utrack is not None
                and self._utrack_health is None and not self._utrack_error)

    def _utrack_health_done(self, health, error: str) -> None:
        self._utrack_health, self._utrack_error = health, error
        self._utrack_checking = False
        self.activity.setText("")
        self.track_page.set_utrack(utrack_engine_lines(
            self._matlab, self._utrack, health=health, error=error,
            absence=self._utrack_absence))
        self._refresh_track()
        after, self._after_check = self._after_check, None
        if after == "track":
            if self._utrack_not_ready():
                self.track_page.set_note(self._utrack_not_ready())
                self._track_queue.clear()
            else:
                self.track_page.set_note("")
                self._next_track()
        elif after == "batch":
            self.batch_page.set_note("")
            self.run_batch()

    def choose_matlab(self) -> None:
        start = str(theme.remembered("matlab_path") or "")
        picked, _filter = QFileDialog.getOpenFileName(
            self, "matlab.exe, in a MATLAB release's bin folder", start,
            "MATLAB (matlab.exe matlab);;All files (*)")
        if picked:
            self._matlab_wanted = Path(picked)
            self.find_utrack()

    def choose_utrack(self) -> None:
        picked = QFileDialog.getExistingDirectory(
            self, "The u-track folder", str(theme.remembered("utrack_folder") or ""))
        if picked:
            self._utrack_wanted = Path(picked)
            self.find_utrack()

    def _utrack_not_ready(self) -> str:
        """Why u-track cannot run now, or ``""``."""
        if self._matlab is None or self._utrack is None:
            return "MATLAB and u-track are not set up yet: see the card above"
        if self._utrack_error:
            return f"MATLAB did not answer: {self._utrack_error}"
        if self._utrack_health is None:
            return "MATLAB is still being checked; wait a moment"
        missing = [k for k in ("statistics", "optimization", "image")
                   if not self._utrack_health.get(k)]
        if missing:
            return f"MATLAB lacks a toolbox u-track needs ({', '.join(missing)})"
        return ""

    def _start_utrack(self, name: str) -> bool:
        if self._needs_utrack_check():
            # The condition goes back to the front of the queue and runs when MATLAB has
            # answered; True stops `_next_track` from moving on meanwhile.
            self._track_queue.insert(0, name)
            self._check_utrack_then("track")
            return True
        reason = self._utrack_not_ready()
        if reason:
            self.track_page.set_note(reason)
            self._track_queue.clear()
            return False
        state = self.states[name]
        try:
            inputs = utrack_inputs(state, self._utrack_settings)
        except PrepError as exc:
            self.track_page.append(f"not tracked: {exc}")
            return False
        if inputs.earlier and not self._overwrite:
            self.track_page.append(
                f"not tracked: {name}: {inputs.earlier} movie(s) already hold a u-track run "
                f"in tif/; tick Replace to run again")
            return False
        self._run_tracker = "utrack"
        self.track_page.append("", *utrack_run_header(inputs))
        self.track_page.hide_finished()
        self._started.setdefault("track", time.monotonic())
        self._run_plan.setdefault(name, len(inputs.movies))
        self._tracking = name
        self._track_done = (0, len(inputs.movies))
        self._stop = threading.Event()
        self.jobs.track_utrack(inputs, self._matlab, self._utrack, self._overwrite,
                               self._stop.is_set,
                               self._options.get("utrack_work_folder") or None,
                               at_once=self._at_once)
        self.track_page.set_note(f"tracking {name} with u-track… (MATLAB runs minimised; "
                                 f"each movie's line appears as it finishes)")
        self._refresh_track()
        return True

    def _utrack_edited(self, key: str, text: str) -> None:
        try:
            self._utrack_settings = set_utrack(self._utrack_settings, key, text)
            note = ""
        except PrepError as exc:
            note = f"not changed: {exc}"
        self.detect_page.set_note(note)
        theme.remember("prep_utrack_settings", self._utrack_settings.as_record())
        self._refresh_utrack_card()
        self._refresh_track()

    def _utrack_lab(self) -> None:
        """Back to the standard: the user's (`options`) where one is set, else the lab's."""
        from ...utrack_run import LAB_STANDARD

        standard = _utrack_standard()
        self._utrack_settings = standard
        theme.remember("prep_utrack_settings", standard.as_record())
        self.detect_page.set_note(
            "every u-track setting is back to the standard settings" if standard == LAB_STANDARD
            else "every u-track setting is back to your standard (Options); the line under "
                 "the card names where it differs from the standard settings")
        self._refresh_utrack_card()
        self._refresh_track()

    # -- the options screen ----------------------------------------------------------- #
    def _tracking_base(self) -> dict:
        """The TrackMate standard an untyped linking radius and frame gap follow."""
        return {"linking_max_distance_px": float(self._options["trackmate_linking_px"]),
                "max_frame_gap": int(self._options["trackmate_frame_gap"]),
                "spot_precision": bool(self._options["trackmate_spot_precision"])}

    def _show_crop_standard(self) -> None:
        """The Movies page's rule and tolerance boxes, set to the session's values."""
        keys = [k for k, _w in START_RULES]
        if self._rule in keys:
            self.movies_page.rule.blockSignals(True)
            self.movies_page.rule.select(keys.index(self._rule))
            self.movies_page.rule.blockSignals(False)
        state = self.states.get(self.condition)
        self.movies_page.set_tolerance(self._tol(self.condition) * 100.0,
                                       note=getattr(state, "flat_note", "") if state else "")

    def open_options(self):
        """The options screen (`ui.widgets.options_dialog`); a save applies at once."""
        return open_options(self, self.t, "prep", on_saved=self._options_saved)

    def _options_saved(self, old_record: dict, new_record: dict) -> None:
        """Apply new standards. A value still at the OLD standard follows the new one; a
        value changed on a page this session stays (`options.follows`)."""
        old = opts.values("prep", old_record)
        new = opts.values("prep", new_record)
        self._options = new
        self._notify_after = notify_after_s()
        moved = []
        rule = opts.follows(self._rule, old["crop_start_rule"], new["crop_start_rule"])
        if rule != self._rule:
            self._rule = rule
            moved.append("the start rule")
        pct = opts.follows(round(self._tolerance * 100.0, 6),
                           round(float(old["crop_flat_tolerance_pct"]), 6),
                           float(new["crop_flat_tolerance_pct"]))
        if abs(pct - self._tolerance * 100.0) > 1e-9:
            self._standard_tolerance(pct / 100.0)
            moved.append("the flat tolerance")
        self._show_crop_standard()
        if new["trackmate_threshold_k_tm"] != old["trackmate_threshold_k_tm"]:
            redone = [name for name, state in self.states.items()
                      if rethreshold(state, float(new["trackmate_threshold_k_tm"]))]
            if redone:
                moved.append(f"the measured thresholds of {len(redone)} condition(s)")
        try:
            old_utrack = opts.utrack_standard(old_record)
        except opts.OptionError:
            old_utrack = None
        standard = _utrack_standard()
        if self._utrack_settings == old_utrack and standard != self._utrack_settings:
            self._utrack_settings = standard
            theme.remember("prep_utrack_settings", standard.as_record())
            moved.append("the u-track settings")
        if self.states:
            self._refresh_detect()
            self._refresh_utrack_card()
            self._refresh_track()
            self._refresh_crop_guide()
        self.batch_page.set_measure_k(float(new["trackmate_threshold_k_tm"]))
        if self.sidebar.current.startswith("batch"):
            self._plan_batch()
        self.activity.setText("Options saved" + (f": {', '.join(moved)} followed"
                                                 if moved else ""))

    # -- stacks for u-track ------------------------------------------------------------ #
    def _export_replace_changed(self, on: bool) -> None:
        self._export_replace = bool(on)
        self._refresh_export()

    def export_stacks(self) -> None:
        """Every ready condition's stacks, one movie per worker."""
        ready = [r["name"] for r in export_rows(self.states, overwrite=self._export_replace)
                 if r["ready"]]
        if not ready:
            self.calibration_page.set_export_note("no condition is ready to export; each "
                                                  "line says why")
            return
        projection = self.calibration_page.projection_kind()
        self._started.setdefault("export", time.monotonic())
        for name in ready:
            inputs = export_inputs(self.states[name])
            self._exporting[name] = (len(inputs["movies"]), [])
            self._export_plan[name] = len(inputs["movies"])
            self.jobs.export(self._generation, name, inputs, projection)
        movies = sum(n for n, _r in self._exporting.values())
        self.calibration_page.set_export_note(
            f"writing {movies} stack(s) for {len(ready)} condition(s)…")
        self._refresh_export()

    def stop_exports(self) -> None:
        # THE NOTE FIRST: a movie that had not started reports at once when cancelled, so a
        # condition can complete INSIDE the call below, and its line must not be
        # overwritten by "stopping" afterwards.
        self.calibration_page.set_export_note("stopping: the movies in hand finish whole, the "
                                              "rest are not written")
        self.jobs.stop_exports()

    def _export_row(self, generation: int, condition: str, row) -> None:
        if generation != self._generation or condition not in self._exporting:
            return
        expected, rows = self._exporting[condition]
        rows.append(row)
        done = sum(len(r) for _e, r in self._exporting.values())
        total = sum(e for e, _r in self._exporting.values())
        self.calibration_page.set_export_note(
            f"{condition} {export_line(len(rows), expected, row)}   ({done} of {total})")
        if len(rows) >= expected:
            del self._exporting[condition]
            state = self.states.get(condition)
            if state is not None:
                state.stacks = len(stack_files(state.folder))
            ok = sum(1 for r in rows if r.get("ok"))
            failed = [f"{r['movie']} ({r['error']})" for r in rows if not r.get("ok")]
            line = (f"{condition}: {ok} of {expected} stack(s) written to tif/"
                    + (f"; not written: {'; '.join(failed[:4])}" if failed else ""))
            self.calibration_page.set_export_note(line)
            self._export_lines.append(line)
            if not self._exporting:
                lines, self._export_lines, self._export_plan = self._export_lines, [], {}
                self._job_finished("export", "Cropped stacks written", lines,
                                   failed=any("not written" in x for x in lines))
        self._refresh_export()

    def open_pixel_helper(self) -> None:
        from .pixel_dialog import PixelSizeDialog

        name = self.condition or next(iter(self.states), None)
        if name is None:
            return
        dialog = PixelSizeDialog(self.t, condition=name, camera=self.states[name].camera,
                                 content=self.centralWidget(), parent=self)
        dialog.applied.connect(lambda value, every, n=name: self._pixel_applied(n, value, every))
        dialog.closed.connect(dialog.deleteLater)
        dialog.open()

    def _pixel_applied(self, name: str, value: float, every: bool) -> None:
        names = list(self.states) if every else [name]
        for n in names:
            set_calibration(self.states[n], "pixel_size_nm", f"{value:.1f}")
        self.calibration_page.set_note(
            f"pixel size {value:.1f} nm set for {', '.join(names)} ; press Save to record it")
        self._refresh_rows()

    # -- the linking radius's helpers ---------------------------------------------------- #
    def _guide_state(self):
        name = (self._guide_condition if self._guide_condition in self.states
                else self.condition)
        return name, self.states.get(name)

    def measure_pilot(self) -> None:
        name, state = self._guide_state()
        if state is None:
            return
        radius = state.tracking.get("linking_max_distance_px") or 5.0
        self.detect_page.guide.set_result(f"reading the tracks in {name}…")
        self.jobs.pilot(self._generation, name, state.folder, float(radius))

    def _pilot_done(self, generation: int, condition: str, got, error: str) -> None:
        if generation != self._generation:
            return
        state = self.states.get(condition)
        text = (f"{condition}: {error.split(': ', 1)[-1]}" if error or got is None else
                "\n".join(pilot_lines(got, (state.tracking.get("linking_max_distance_px")
                                             or 5.0) if state else 5.0)))
        self.detect_page.guide.set_result(text)

    def check_wider_radius(self) -> None:
        name, state = self._guide_state()
        if state is None:
            return
        guide = self.detect_page.guide
        if (self._tracking is not None or self._checking is not None
                or self._batch_stop is not None):
            guide.set_result(f"{self._busy()}; one Fiji run at a time")
            return
        install = self._install
        if install is None or not install.has_trackmate:
            guide.set_result("Fiji with TrackMate is not set up yet: see the Track page")
            return
        try:
            inputs = run_inputs(state)
        except PrepError as exc:
            guide.set_result(f"cannot check: {exc}")
            return
        self._checking = name
        self._check_stop = threading.Event()
        self._started["check"] = time.monotonic()
        self._check_done = None
        guide.set_result(f"checking {name}: two movies at "
                         f"{inputs.settings.linking_max_distance_px:g} px and at "
                         f"{fold_words(RADIUS_CHECK_MULTIPLE, of=True)} that, "
                         f"about 2 min…")
        self.jobs.radius_check(inputs, install, self._check_stop.is_set)
        self._refresh_track()

    def _radius_progress(self, name: str, index: int, total: int, row) -> None:
        if name == self._checking:
            self._check_done = (index, total)
            self._refresh_progress()
            self.detect_page.guide.set_result(
                f"checking {name}: run {index} of {total}, {row['movie']} at "
                f"{row['radius']:g} px: "
                + (f"{row['n_tracks']} tracks" if row.get("ok") else f"FAILED {row['error']}"))

    def _radius_checked(self, name: str, got, error: str) -> None:
        if name != self._checking:
            return
        self._checking, self._check_stop = None, None
        lines = ([f"radius check failed for {name}: {error}"] if error or got is None
                 else radius_check_lines(got))
        self.detect_page.guide.set_result("\n".join(lines))
        self.track_page.append("", *lines)
        self._check_done = None
        self._job_finished("check", f"Wider-radius check done: {name}", lines,
                           failed=bool(error or got is None))
        self._refresh_track()

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

    def _batch_measure_changed(self, on: bool) -> None:
        self._batch_measure = bool(on)
        self._plan_batch()

    def _at_once_changed(self, n: int) -> None:
        """Movies at once, from either page: kept for both, remembered, and the estimates follow."""
        n = max(1, min(int(n), MAX_AT_ONCE))
        if n == self._at_once:
            return
        self._at_once = n
        try:
            theme.remember("prep_at_once", n)
        except OSError:
            pass
        self._show_at_once()
        self._refresh_track()
        self._plan_batch()

    def _show_at_once(self) -> None:
        tip = at_once_tip(self._tracker, default_at_once())
        self.track_page.set_at_once(self._at_once, tip)
        self.batch_page.set_at_once(self._at_once, tip)

    def _batch_live(self) -> dict:
        """The window's own conditions, copied (`batch_model.snapshot`): a folder open here
        is planned and run with its Detect & link values, and an unsaved edit is refused
        by name, exactly as on the Track page."""
        return {state.folder: snapshot(state) for state in self.states.values()}

    def _batch_settings(self) -> dict:
        """What decides whether a condition is ready (`batch_model.plan_tracking`)."""
        return {"tracker": self._tracker, "utrack_settings": self._utrack_settings,
                "replace": self._batch_replace, "measure_thresholds": self._batch_measure,
                "base": self._tracking_base(),
                "k": float(self._options["trackmate_threshold_k_tm"])}

    def _plan_batch(self) -> None:
        """List the parents and plan, on a thread; a newer request supersedes an older."""
        if self._batch_stop is not None:
            return
        self._batch_generation += 1
        self._batch_status, self._batch_summary = {}, ""
        if not self._batch_parents:
            self._batch_rows = []
            self._show_batch_rows()
            return
        self.batch_page.set_summary("listing the parent folders…")
        self.jobs.batch_plan(self._batch_generation, self._batch_parents, self._batch_live(),
                             dict(self._batch_settings(), at_once=self._at_once))

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
        page.set_rows([{"key": r["key"], "cells": track_batch_cells(r, self._batch_status.get(
            r["key"]))} for r in rows])
        page.set_layout_lines(layout_notes(rows))
        tracker = self._batch_tracker if self._batch_stop is not None else self._tracker
        page.settings_note.setText(settings_line(
            tracker, self._options.get("utrack_work_folder") or None))
        if self._batch_stop is None:
            # after a run its own summary stands until anything is planned again
            page.set_summary(self._batch_summary or plan_summary(
                rows, tracker=tracker, sessions=self._at_once))
        page.set_busy(self._batch_stop is not None,
                      can_run=bool(self._batch_parents) and any(r["ready"] for r in rows))

    def _batch_engine_not_ready(self) -> str:
        """Why the chosen tracker cannot run a batch now, or ``""`` -- the Track page's
        refusals."""
        if self._tracker == "utrack":
            return self._utrack_not_ready()
        install = self._install
        if install is None or not install.has_trackmate:
            return "Fiji with TrackMate is not set up yet: see the Fiji card on Track"
        return ""

    def run_batch(self) -> None:
        """Run every ready condition under the parents (`batch_model.run_tracking`). The
        folders are listed again as the run starts, so it runs what is on disk NOW."""
        page = self.batch_page
        if self._batch_stop is not None:
            return
        if not self._batch_parents:
            page.set_note("add a parent folder first")
            return
        if self._tracking is not None or self._checking is not None:
            page.set_note(f"{self._busy()}; the batch can start once it is done")
            return
        if self._tracker == "utrack" and self._needs_utrack_check():
            self._check_utrack_then("batch")
            return
        if self._utrack_checking:
            page.set_note("MATLAB is starting; the batch starts when it answers")
            return
        reason = self._batch_engine_not_ready()
        if reason:
            page.set_note(reason)
            return
        self._batch_stop = threading.Event()
        self._batch_tracker = self._tracker
        self._batch_report = report_path(self._batch_parents[0])
        self._batch_plan, self._batch_running, self._batch_finished = [], [], []
        self._batch_counts, self._batch_status, self._batch_results = {}, {}, {}
        self._batch_report_error = ""
        self._started["batch"] = time.monotonic()
        # KEPT AWAKE FROM THIS THREAD, which outlives the batch; cleared when it ends.
        keep_awake(True)
        settings = dict(self._batch_settings(), install=self._install, matlab=self._matlab,
                        utrack=self._utrack, sessions=self._at_once,
                        work_folder=self._options.get("utrack_work_folder") or None,
                        report=self._batch_report)
        page.set_note(f"running; {awake_note()}" if awake_note() else "running")
        page.set_busy(True)
        self.jobs.batch_run(self._batch_parents, self._batch_live(), settings,
                            self._batch_stop.is_set)
        self.activity.setText("a batch is running; see Batch")
        self._refresh_progress()

    def stop_batch(self) -> None:
        """Stop after the movie in hand: running conditions finish that movie, the rest
        are not started (each reported so)."""
        if self._batch_stop is None:
            return
        self._batch_stop.set()
        self.batch_page.set_note("stopping: the movie in hand finishes, then nothing new "
                                 "starts")

    def _batch_event(self, kind: str, key: str, payload) -> None:
        if self._batch_stop is None:
            return
        page = self.batch_page
        if kind == "planned":
            self._batch_plan = [dict(r) for r in payload]
            self._batch_rows = [dict(r) for r in payload]
            page.append("", *plan_lines(self._batch_rows,
                                        what="u-track" if self._batch_tracker == "utrack"
                                        else "TrackMate"))
            self._show_batch_rows()
            self._refresh_progress()
            return
        labels = {r["key"]: r["label"] for r in self._batch_plan}
        totals = {r["key"]: int(r["movies"]) for r in self._batch_plan}
        if kind == "started":
            self._batch_running.append(key)
            self._batch_counts[key] = 0
            self._batch_status[key] = running_status(0, totals.get(key, 0))
        elif kind == "measuring":
            self._batch_status[key] = running_status(0, totals.get(key, 0), measuring=True)
        elif kind == "movie":
            self._batch_counts[key] = int(payload["index"])
            self._batch_status[key] = running_status(int(payload["index"]),
                                                     int(payload["total"]))
        elif kind == "finished":
            if key in self._batch_running:
                self._batch_running.remove(key)
            self._batch_finished.append(key)
            self._batch_results[key] = dict(payload)
            self._batch_status[key] = result_status(payload)
        elif kind == "report_error":
            self._batch_report_error = str(payload.get("error", ""))
        line = event_line(kind, labels.get(key, key), payload, tracker=self._batch_tracker)
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
            # THE BATCH ITSELF FAILED (a parent vanished mid-listing, say): what finished
            # is still reported, and the failure is named first.
            rows = list(self._batch_results.values())
            self.batch_page.append(f"The batch failed: {error}")
            self._batch_report_error = self._batch_report_error or error
        summary = finished_summary(rows, what="Tracking", report=self._batch_report,
                                   report_error=self._batch_report_error)
        self.batch_page.append("", summary["title"], *summary["lines"])
        self.batch_page.set_note(f"{summary['title']}; report: {self._batch_report}")
        self.activity.setText(f"{summary['title']}; see Batch")
        # A CONDITION OPEN HERE MAY HAVE BEEN TRACKED: its rows must see the new run.
        for state in self.states.values():
            state.earlier_run = len(track_files(state.folder))
        self._batch_summary = f"{summary['title']}: {summary['lines'][0]}"
        self._show_batch_rows()
        if self.states:
            self._refresh_track()
        self._job_finished("batch", summary["title"], summary["lines"],
                           failed=summary["failed"])

    # -- the classic window, for the steps not ported yet ----------------------------- #
    def open_classic(self) -> None:
        from ...tracking_gui import launch_tracking

        window = launch_tracking(str(self.folder) if self.folder else None, show=True,
                                 block=False)
        _OPEN_WINDOWS.add(window)

    def closeEvent(self, event) -> None:
        # A RUN IN FLIGHT STOPS AFTER THE MOVIE IN HAND -- Fiji is a subprocess, and
        # killing it would leave a half-written XML -- and writes its run record.
        if self._stop is not None:
            self._stop.set()
        if self._check_stop is not None:
            self._check_stop.set()
        if self._batch_stop is not None:
            # the batch stops the same way, its report still written; the machine may
            # sleep again once the window is gone
            self._batch_stop.set()
            keep_awake(False)
        self._track_queue = []
        self._progress_timer.stop()
        self.jobs.stop_exports()
        self.jobs.shutdown()
        super().closeEvent(event)
        if event.isAccepted():
            _OPEN_WINDOWS.discard(self)


def _utrack_standard():
    """The u-track standard: the user's (`options.utrack_standard`), or the lab's when
    none is set or the stored one is refused."""
    from ...utrack_run import LAB_STANDARD

    try:
        return opts.utrack_standard(theme.remembered(opts.OPTIONS_KEY))
    except opts.OptionError:
        return LAB_STANDARD


def _remembered_utrack_settings():
    """The u-track settings last used (`theme.remembered`), or the standard (the user's,
    else the lab's). A record from another version -- a field renamed, a value u-track
    would refuse -- falls back to the standard rather than half-applying."""
    from dataclasses import fields

    from ...utrack_run import UTrackError, UTrackSettings

    record = theme.remembered("prep_utrack_settings")
    if not isinstance(record, dict):
        return _utrack_standard()
    names = {f.name for f in fields(UTrackSettings)}
    values = {}
    for key, value in record.items():
        if key in names:
            values[key] = tuple(value) if isinstance(value, list) else value
    try:
        return UTrackSettings(**values)
    except (UTrackError, TypeError, ValueError):
        return _utrack_standard()


def open_prep(root=None, *, color_scheme: str | None = None, show: bool = True,
              jobs: PrepJobs | None = None):
    """Build the window, show it, and open ``root`` if one was given -- AFTER the window
    is on screen, so nothing slow happens before there is something to look at."""
    app = QApplication.instance() or QApplication([])
    from ...appicon import apply as _apply_icon
    _apply_icon()
    t = theme.current(color_scheme)
    window = PrepWindow(t, jobs=jobs)
    _OPEN_WINDOWS.add(window)
    # 1320 x 860 where the screen holds it, maximised where it does not, the minimum held to the screen
    # (`ui.widgets.screenfit`, 30 Oct 2026: it opened 44 px past a 1536 x 816 laptop's taskbar)
    from ..widgets.screenfit import show_fitted

    if show:
        show_fitted(window)
    # THE READERS START WHILE THE USER IS STILL LOOKING AT THE WINDOW, off its thread.
    QTimer.singleShot(0, window, window.jobs.warm)
    QTimer.singleShot(0, window, window.find_fiji)
    QTimer.singleShot(0, window, window.find_utrack_if_wanted)
    if root is not None:
        QTimer.singleShot(0, window, lambda: window.open_folder(root))
    return app, window


def selftest_prep(root, *, timeout_s: float = 900.0, jobs: PrepJobs | None = None) -> int:
    """PACKAGING CHECK: the real window, hidden, reads every movie under ``root``.

    The preparation window reads movies in a SPAWNED process pool, and a frozen app
    starts each worker by re-launching its own executable -- which works only if the
    entry point calls `multiprocessing.freeze_support()` and every reader made it into
    the bundle. `--selftest` builds a napari viewer and touches neither, so this opens the
    production window on a folder WITHOUT saved curves and waits until each movie has
    arrived from the pool or failed. Prints one line per condition; 0 only when every
    movie was read and at least one came through the pool."""
    import time

    app, window = open_prep(root, show=False, jobs=jobs)
    pooled, opened = {"n": 0}, {"done": False}
    window.jobs.signals.opened.connect(
        lambda *_args: opened.__setitem__("done", True))
    window.jobs.signals.curve.connect(
        lambda generation, condition, stem, curves, error: pooled.__setitem__(
            "n", pooled["n"] + (curves is not None)))
    started = time.monotonic()

    def finished() -> bool:
        if not window.states:
            # Still listing, or the folder held nothing to prepare (the start page says).
            return opened["done"]
        return all(len(s.curves) + len(s.errors) >= len(s.movies)
                   for s in window.states.values())

    timer = QTimer()
    timer.setInterval(250)

    def poll() -> None:
        if finished() or time.monotonic() - started > timeout_s:
            timer.stop()
            app.quit()

    timer.timeout.connect(poll)
    timer.start()
    app.exec()
    seconds = time.monotonic() - started
    states = dict(window.states)
    window.close()

    if not states:
        print(f"selftest-prep: {root}: no condition opened: "
              f"{window.start_page.text.text() or 'nothing to prepare'}")
        return 1
    ok = True
    for name, state in states.items():
        read, failed = len(state.curves), dict(state.errors)
        print(f"selftest-prep: {name}: {read}/{len(state.movies)} movie(s) read"
              + (f", {len(failed)} failed" if failed else ""))
        for stem, error in failed.items():
            print(f"    {stem}: {error}")
        ok = ok and not failed and read == len(state.movies)
    print(f"selftest-prep: {pooled['n']} read through the worker pool in {seconds:.0f} s")
    if not pooled["n"]:
        print("selftest-prep: the pool was not exercised (saved curves were loaded "
              "instead); point it at a folder without bleaching/ curves")
    return 0 if ok and pooled["n"] else 1
