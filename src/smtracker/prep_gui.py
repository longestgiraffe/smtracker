"""The PREPARATION stage's window: raw acquisitions to the stacks everything else reads.

The first stage of the launcher, and the one that was shown DISABLED because the pipeline
behind it did not exist. It does now (`smtracker.movieprep`, validated bit-identical to
the lab's Fiji macro on a real 1251-frame stack), so the button opens something.

THIN QT, the same rule as `tracking_gui` and `widgets`. Every number on screen comes from
`movieprep` (`bleaching_curve`, `density_curve`, `crop_floor`, `prepare_batch`) all pure
and covered headless. An AST test keeps numpy/pandas/scipy out of this file.

What this window refuses, and why each one matters

* **It will not propose a crop.** Measured on twelve crops actually made: field mean, spot
  density and spot amplitude ALL place the single-molecule floor a median 2124 frames,
  50.5 per cent of each acquisition: before the range the experimenter chose. The images
  do not determine that margin, so a suggested number would be the tool's judgement
  wearing the analyst's name. `CropFloor` is drawn as a FLOOR line and labelled as one.
* **The crop is per movie, and every movie must get one.** `prepare_batch` refuses a
  movie with no range rather than reusing a neighbour's, because these acquisitions
  differ in length (4000 and 4500 frames inside one protein) and a silently-inherited
  range produces a stack nobody looked at.
* **Pixel size and frame interval have no defaults.** The `.nd2` reports 160 nm (the bare
  objective magnification with no tube lens) and taking it would rescale every D by
  2.28x. The frame interval is 20.472 ms measured from the camera against the 20.0 in
  `movieData.mat`; the window shows both and makes the user pick.
* **A failed movie is a row with its error**, named first, never an absence.
"""
from __future__ import annotations

from pathlib import Path

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtCore import QObject, QRunnable, QThreadPool, Signal
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QListWidget,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

__all__ = ["PrepWindow", "launch_prep"]

#: The camera's own timestamps say 20.472 ms; `movieData.mat` carries the nominal 20.0.
#: Both are offered because the choice belongs to the analyst, and mixing them across
#: datasets shifts every D by about 2.4 per cent.
MEASURED_INTERVAL_MS = 20.472
NOMINAL_INTERVAL_MS = 20.0

#: From `movieData.mat` for this microscope. Shown, never assumed.
PIXEL_SIZE_NM = 106.0


class _PrepSignals(QObject):
    progress = Signal(int, int, str)
    finished = Signal(object)
    failed = Signal(str)


class _PrepJob(QRunnable):
    """Runs `prepare_batch` off the GUI thread. It CALLS; it does not compute."""

    def __init__(self, raw_dir, out_root, crops, pixel_size_nm, frame_interval_s,
                 groups=None):
        super().__init__()
        self._raw_dir = raw_dir
        self._out_root = out_root
        self._crops = crops
        #: `[(condition, raw_dir, out_root, crops)]` when several conditions are being
        #: prepared at once. The looping and the table concatenation live in `movieprep`,
        #: not here: this module may not import pandas, and an AST test enforces it.
        self._groups = groups
        self._pixel_size_nm = pixel_size_nm
        self._frame_interval_s = frame_interval_s
        self.signals = _PrepSignals()

    def run(self):                                     # noqa: D102 - QRunnable protocol
        from .movieprep import prepare_batch

        try:
            def on_progress(index, total, row):
                note = (f"{row['movie']}: {row['n_frames_kept']} frames "
                        f"({row['crop_start']}-{row['crop_end']})" if row["ok"]
                        else f"{row['movie']}: FAILED: {row['error']}")
                self.signals.progress.emit(index, total, note)

            if self._groups:
                from .movieprep import prepare_conditions

                table = prepare_conditions(
                    self._groups, pixel_size_nm=self._pixel_size_nm,
                    frame_interval_s=self._frame_interval_s,
                    on_progress=on_progress)
            else:
                table = prepare_batch(self._raw_dir, self._out_root, self._crops,
                                      pixel_size_nm=self._pixel_size_nm,
                                      frame_interval_s=self._frame_interval_s,
                                      on_progress=on_progress)
            self.signals.finished.emit(table)
        except Exception as exc:                       # noqa: BLE001 - reported, never eaten
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}")


class PrepWindow(QWidget):
    """raw/ → per-movie curves → per-movie crop → prepared stacks."""

    def __init__(self, root=None, parent=None):
        super().__init__(parent)
        # THE RUNNING BUILD IS IN THE TITLE, where a screenshot always
        # catches it. Three things in this repository are called smtracker
        # and they looked identical on screen; a day was spent proving a fix
        # worked in source while an older bundle was open. See `buildstamp`.
        from .buildstamp import build_stamp
        self.setWindowTitle("SMTracker · movie preparation  ·  " + build_stamp())
        self._raw_dir = Path(root) if root else None
        self._movies: list[Path] = []
        self._crops: dict[str, tuple[int, int]] = {}
        #: `{condition name: [acquisition paths]}` when a PARENT folder was opened, so a
        #: whole experiment is one session instead of one window per condition.
        self._conditions: dict = {}
        #: The folder that was OPENED, which is not `_raw_dir`: picking a condition
        #: points `_raw_dir` at that condition's acquisitions. Organising is a decision
        #: about the whole session, so it needs the parent, and using `_raw_dir` filed
        #: one condition while reporting the session.
        self._parent_dir = None
        #: Crops kept PER CONDITION. `series001` exists in every condition, so one flat
        #: stem-keyed mapping would hand one condition's movie another's crop range, and
        #: nothing downstream would show it had happened.
        self._crops_by_condition: dict = {}
        self._curves = {}
        self._floor = None
        self._pool = QThreadPool.globalInstance()
        self._last_batch = None

        layout = QVBoxLayout(self)

        box = QGroupBox("1 · Raw acquisitions")
        row = QHBoxLayout(box)
        self._folder_label = QLabel("no folder chosen")
        self._folder_label.setWordWrap(True)
        row.addWidget(self._folder_label, 1)
        choose = QPushButton("Choose raw/ folder…")
        choose.clicked.connect(self._choose_folder)
        row.addWidget(choose)
        layout.addWidget(box)

        # CONDITIONS, when the chosen folder holds several. Hidden otherwise, so opening
        # one condition looks exactly as it did before this existed.
        self._conditions_row = QHBoxLayout()
        self._conditions_label = QLabel("condition")
        self._conditions_box = QComboBox()
        self._conditions_box.currentTextChanged.connect(self._pick_condition)
        self._conditions_row.addWidget(self._conditions_label)
        self._conditions_row.addWidget(self._conditions_box, 1)
        # ORGANISE, offered only where it is needed. A session straight off the
        # microscope is loose files, and segmentation cannot see a condition until the
        # images are in `phase/`. Hidden when everything is already filed, because a
        # button that would do nothing still invites a press.
        self._organise_button = QPushButton("Organise all conditions…")
        self._organise_button.setToolTip(
            "File each condition's acquisitions into raw/ and its images into phase/. "
            "Shows what would move first and asks whether to move or copy.")
        self._organise_button.clicked.connect(self._organise_all)
        self._organise_button.hide()
        self._conditions_row.addWidget(self._organise_button)
        layout.addLayout(self._conditions_row)
        self._conditions_label.hide()
        self._conditions_box.hide()

        middle = QHBoxLayout()
        box = QGroupBox("2 · Movies")
        col = QVBoxLayout(box)
        self._list = QListWidget()
        self._list.currentTextChanged.connect(self._show_movie)
        col.addWidget(self._list)
        middle.addWidget(box, 1)

        box = QGroupBox("3 · Bleaching and density: set the crop yourself")
        col = QVBoxLayout(box)
        self._figure = Figure(figsize=(6, 4), layout="constrained")
        self._canvas = FigureCanvasQTAgg(self._figure)
        col.addWidget(self._canvas, 1)
        self._floor_label = QLabel("choose a movie")
        self._floor_label.setWordWrap(True)
        self._floor_label.setStyleSheet("color:#555;")
        col.addWidget(self._floor_label)

        form = QFormLayout()
        self._start = QSpinBox()
        self._start.setRange(1, 100000)
        self._end = QSpinBox()
        self._end.setRange(1, 100000)
        form.addRow("crop start (1-based, inclusive)", self._start)
        form.addRow("crop end", self._end)
        apply_one = QPushButton("Set crop for this movie")
        apply_one.clicked.connect(self._set_crop)
        form.addRow("", apply_one)
        # Tracking straight from raw needs the ranges but not the stacks, so they can be
        # recorded without preparing anything.
        save_only = QPushButton("Save crops only (for raw tracking)")
        save_only.clicked.connect(self._save_crops_only)
        form.addRow("", save_only)
        col.addLayout(form)
        middle.addWidget(box, 2)
        layout.addLayout(middle)

        box = QGroupBox("4 · Calibration: neither value has a default")
        form = QFormLayout(box)
        self._pixel = QDoubleSpinBox()
        self._pixel.setRange(1.0, 1000.0)
        self._pixel.setDecimals(2)
        self._pixel.setValue(PIXEL_SIZE_NM)
        self._pixel.setSuffix(" nm")
        self._pixel.setToolTip(
            "From movieData.mat. The .nd2 reports 160 nm, which is the objective magnification alone "
            "without the tube lens; using it would rescale every D by 2.28 times.")
        form.addRow("pixel size", self._pixel)

        self._interval = QDoubleSpinBox()
        self._interval.setRange(0.1, 1000.0)
        self._interval.setDecimals(4)
        self._interval.setValue(MEASURED_INTERVAL_MS)
        self._interval.setSuffix(" ms")
        self._interval.setToolTip(
            f"{MEASURED_INTERVAL_MS} ms is measured from the camera timestamps; "
            f"movieData.mat carries the nominal {NOMINAL_INTERVAL_MS} ms. They differ by "
            "2.4 %, which shifts every D by that much and leaves fractions untouched")
        form.addRow("frame interval", self._interval)
        layout.addWidget(box)

        box = QGroupBox("5 · Prepare")
        col = QVBoxLayout(box)
        row = QHBoxLayout()
        self._run_button = QPushButton("Prepare all movies")
        self._run_button.clicked.connect(self._run)
        row.addWidget(self._run_button)
        self._run_all_button = QPushButton("Prepare all conditions")
        self._run_all_button.setToolTip(
            "Prepare every condition in this folder. Each keeps its own crop ranges, "
            "and every movie in every condition must have one first.")
        self._run_all_button.clicked.connect(self._run_all)
        self._run_all_button.hide()
        row.addWidget(self._run_all_button)
        self._status = QLabel("")
        row.addWidget(self._status, 1)
        col.addLayout(row)
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        col.addWidget(self._progress)
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMinimumHeight(120)
        col.addWidget(self._log)
        layout.addWidget(box, 1)

        self.resize(1080, 820)
        if self._raw_dir is not None:
            self._open(self._raw_dir)
        self._refresh_status()

    # -- folder ---------------------------------------------------------------
    def _choose_folder(self):
        picked = QFileDialog.getExistingDirectory(
            self, "Folder holding the raw acquisitions",
            str(self._raw_dir or Path.home()))
        if not picked:
            return
        self._open(Path(picked))

    def _open(self, folder) -> None:
        """THE ONE DOOR. Every route into this window comes through here.

        Opening with a `root` used to call `_scan` directly while only the folder button
        went through the condition check, so the launcher and `--root` never detected a
        parent of conditions and the feature was silently absent on the path most people
        use. That is this project's own recorded shape: a capability reachable on one
        path only is a capability silently ignored, and it is why `app._open_segmentation`
        exists. A test asserts both callers land here.
        """
        self._raw_dir = Path(folder)
        self._parent_dir = Path(folder)
        # A PARENT OF CONDITIONS FIRST. Falls through to the ordinary single-folder scan
        # when there is only one, so opening one condition is unchanged.
        if self._load_conditions(self._raw_dir):
            return
        self._conditions = {}
        self._conditions_label.hide()
        self._conditions_box.hide()
        self._run_all_button.hide()
        # THE ORGANISE BUTTON TOO. Hiding only the batch button left this one on screen
        # after falling back to a single folder, still pointing at a job already done.
        self._organise_button.setVisible(self._folder_needs_organising(self._raw_dir))
        self._scan()

    def _folder_needs_organising(self, folder) -> bool:
        """Whether anything under ``folder`` still has its files loose. Advisory: a
        failure here must not stop a folder opening."""
        try:
            from .organise import needs_organising

            return bool(needs_organising(folder))
        except Exception:                                 # noqa: BLE001 - advisory
            return False

    def _scan(self):
        from .movieprep import MoviePrepError
        from .rawmovie import _READERS, discover_raw_movies

        readable = set(_READERS)

        self._companions = []
        try:
            # ONE SCAN FOR THE WHOLE PIPELINE. Listing every readable suffix counted the
            # phase snapshot beside each acquisition as a movie of its own.
            self._movies, self._companions = discover_raw_movies(
                self._raw_dir, report=self._log.appendPlainText)
        except OSError as exc:
            self._movies = []
            self._folder_label.setText(f"{self._raw_dir}: {exc}")
            self._refresh_status()
            return
        self._crops.clear()
        self._curves.clear()
        self._list.clear()
        for path in self._movies:
            self._list.addItem(path.stem)
        if self._movies:
            self._folder_label.setText(
                f"{self._raw_dir}  ·  {len(self._movies)} acquisition(s)")
        else:
            self._folder_label.setText(
                f"{self._raw_dir}: no readable acquisitions "
                f"({', '.join(sorted(readable))})")
        self._refresh_status()

    # -- per movie ------------------------------------------------------------
    def _show_movie(self, stem):
        if not stem:
            return
        path = next((p for p in self._movies if p.stem == stem), None)
        if path is None:
            return
        from .movieprep import MoviePrepError, bleaching_curve, crop_floor, density_curve
        from .rawmovie import read_raw_movie

        self._status.setText(f"reading {stem} …")
        QApplication.processEvents()
        try:
            movie = read_raw_movie(path)
            curve = bleaching_curve(movie)
            density = density_curve(movie)
            try:
                floor = crop_floor(movie)
            except MoviePrepError as exc:
                floor = None
                self._floor_label.setText(f"no floor available: {exc}")
        except Exception as exc:                       # noqa: BLE001 - named, not eaten
            self._floor_label.setText(f"{type(exc).__name__}: {exc}")
            self._status.setText("")
            return

        self._curves[stem] = (curve, density, floor)
        self._floor = floor
        n = int(len(curve))
        self._start.setMaximum(n)
        self._end.setMaximum(n)
        if stem in self._crops:
            lo, hi = self._crops[stem]
        else:
            lo, hi = (floor.earliest_defensible if floor else 1), n
        self._start.setValue(lo)
        self._end.setValue(hi)
        if floor is not None:
            self._floor_label.setText(floor.describe())
        self._draw(stem)
        self._status.setText("")

    def _draw(self, stem):
        curve, density, floor = self._curves[stem]
        self._figure.clear()
        top, bottom = self._figure.subplots(2, 1, sharex=True)
        top.plot(curve["frame"], curve["mean_counts"], lw=0.8)
        top.set_ylabel("frame mean")
        top.set_title(f"{stem}: you set the crop", loc="left", fontsize=9)
        bottom.plot(density["frame"], density["n_spots"], lw=0.8, color="C1")
        bottom.set_ylabel("spots / frame")
        bottom.set_xlabel("frame")
        for axis in (top, bottom):
            if floor is not None:
                axis.axvline(floor.earliest_defensible, color="C3", ls="--", lw=1.0)
            axis.axvspan(self._start.value(), self._end.value(), color="C2", alpha=0.12)
        if floor is not None:
            top.legend(["frame mean", "earliest defensible start (a lower limit, not a proposal)"],
                       fontsize=7, loc="upper right")
        self._canvas.draw_idle()

    def _set_crop(self):
        stem = self._list.currentItem().text() if self._list.currentItem() else None
        if not stem:
            return
        lo, hi = int(self._start.value()), int(self._end.value())
        if hi <= lo:
            self._log.appendPlainText(f"{stem}: end must be after start")
            return
        self._crops[stem] = (lo, hi)
        self._log.appendPlainText(f"{stem}: crop {lo}-{hi} ({hi - lo + 1} frames)")
        # PRE-FILLING THE BOX AT THE FLOOR MAKES IT A PROPOSAL IN EVERYTHING BUT NAME,
        # and accepting it is the one mistake with a measured cost: on twelve real crops
        # the range chosen sits a median 2124 frames LATER than any floor the images
        # support, so clicking through would widen every acquisition. Not blocked - the
        # analyst may have a reason - but never silent.
        floor = self._curves.get(stem, (None, None, None))[2]
        if floor is not None and lo == floor.earliest_defensible:
            self._log.appendPlainText(
                f"    Note: {lo} is the earliest defensible start, not a recommendation. A later "
                "start is usual; check the curves above.")
        if stem in self._curves:
            self._draw(stem)
        self._refresh_status()

    def _save_crops_only(self):
        from .movieprep import CROPS_FILENAME, MoviePrepError, save_crops

        if not self._crops:
            self._log.appendPlainText("no crops set yet")
            return
        path = self._raw_dir / CROPS_FILENAME
        try:
            save_crops(path, self._crops, pixel_size_nm=float(self._pixel.value()),
                       frame_interval_s=float(self._interval.value()) / 1000.0)
        except MoviePrepError as exc:
            self._log.appendPlainText(f"could not record crops: {exc}")
            return
        self._log.appendPlainText(
            f"{len(self._crops)} crop(s) recorded in {path}. The tracking stage's raw "
            "path reads this; no stacks were written.")

    def _refresh_status(self):
        total = len(self._movies)
        done = len(self._crops)
        missing = total - done
        if total == 0:
            # "0/0 ... ready" reads as ready. Nothing loaded is not ready.
            self._status.setText("no acquisitions loaded")
        else:
            self._status.setText(
                f"{done}/{total} movies have a crop"
                + (f": {missing} still to set" if missing else ": ready"))
        self._run_button.setEnabled(total > 0 and missing == 0)

    # -- run ------------------------------------------------------------------
    def _load_conditions(self, parent) -> bool:
        """Open a PARENT of conditions, or report that it is a single condition.

        Returns True when several were found. The single case falls through to the
        ordinary scan untouched, so nothing about opening one folder changed.
        """
        from .movieprep import discover_across_conditions

        try:
            groups = discover_across_conditions(parent, report=self._log.appendPlainText)
        except Exception as exc:                      # noqa: BLE001 - reported
            self._log.appendPlainText(f"{type(exc).__name__}: {exc}")
            return False
        # ONE SUBFOLDER IS A CONDITION TOO, and this gate used to read `< 2`. Discovery
        # was never wrong -- `discover_across_conditions` returns ['sub1'] for a lone
        # condition, loose files or filed into `raw/` alike -- and returning False here
        # threw that away: the caller fell back to scanning `self._raw_dir`, which is the
        # PARENT, so the movies inside the condition were never found and the window
        # showed nothing.
        #
        # THE SAME DEFECT WAS FIXED IN `tracking_gui` FIRST AND DID NOT REACH HERE, which
        # is this project's own recorded shape: every entry point does its own discovery,
        # so one fix does not reach the next. Both are now pinned by tests.
        if not groups:
            return False
        alone = len(groups) == 1
        # WHAT WAS LEFT OUT, and why. A folder holding no readable acquisition is not a
        # condition, but dropping it silently reads as "the tool lost one": the real
        # `D:/data/project` has four subfolders and one of them is empty.
        from .organise import discover_project_folders

        try:
            seen = {st.folder.name for st in discover_project_folders(parent)}
        except Exception:                                 # noqa: BLE001 - best effort
            seen = set(groups)
        skipped = sorted(seen - set(groups))
        subfolders = sorted(q.name for q in Path(parent).iterdir() if q.is_dir())
        empty = [q for q in subfolders if q not in seen]
        if (skipped or empty) and not alone:
            self._log.appendPlainText(
                "not offered: "
                + ", ".join([f"{q} (no readable acquisition)" for q in skipped]
                            + [f"{q} (empty)" for q in empty]))
        self._conditions = groups
        self._crops_by_condition = {name: {} for name in groups}
        self._conditions_box.blockSignals(True)
        self._conditions_box.clear()
        self._conditions_box.addItems(list(groups))
        self._conditions_box.blockSignals(False)
        # HIDDEN BELOW TWO, because "this condition" is unambiguous there and a
        # one-entry dropdown cannot help. What must NOT happen is the state going with
        # the control, which is exactly how this returned False and lost the folder.
        self._conditions_label.setVisible(not alone)
        self._conditions_box.setVisible(not alone)
        self._run_all_button.setText(f"Prepare all {len(groups)} conditions")
        self._run_all_button.setVisible(not alone)
        self._log.appendPlainText(
            (f"{len(groups)} conditions: " if not alone else "one condition: ")
            + ", ".join(f"{k} ({len(v)})" for k, v in groups.items()))
        # Offered only where something is still loose.
        try:
            from .organise import needs_organising

            loose = needs_organising(parent)
        except Exception:                                 # noqa: BLE001 - advisory
            loose = []
        self._organise_button.setVisible(bool(loose))
        self._parent_dir = Path(parent)
        if loose:
            self._log.appendPlainText(
                f"{len(loose)} of these hold loose files: " + ", ".join(loose)
                + ". Segmentation needs phase/, so 'Organise all conditions' files them.")
        self._pick_condition(next(iter(groups)))
        return True

    def _organise_all(self) -> None:
        """File every loose condition, after showing what would move and asking how.

        NOTHING HAPPENS WITHOUT AN EXPLICIT CHOICE. `mode` has no default anywhere in
        `organise` because moving is not undone by pressing anything and copying doubles
        the folder, so the dialog offers both and Cancel, and Cancel is the default
        button. The plan is shown first: this is the one step where a mistake is silent.
        """
        from .organise import (OrganiseError, apply_layouts, describe_layouts,
                               plan_layouts)

        parent = self._parent_dir or self._raw_dir
        if parent is None:
            return
        try:
            plans = plan_layouts(parent)
        except OrganiseError as exc:
            self._log.appendPlainText(str(exc))
            return
        if not any(plan.moves for plan in plans.values()):
            self._log.appendPlainText("every condition is already organised")
            return

        box = QMessageBox(self)
        box.setWindowTitle("Organise these folders?")
        box.setText("Acquisitions go to raw/, images to phase/, one folder at a time.")
        box.setInformativeText(describe_layouts(plans))
        move = box.addButton("Move", QMessageBox.AcceptRole)
        copy = box.addButton("Copy", QMessageBox.AcceptRole)
        cancel = box.addButton(QMessageBox.Cancel)
        box.setDefaultButton(cancel)
        box.exec()
        clicked = box.clickedButton()
        if clicked is cancel or clicked is None:
            self._log.appendPlainText("organising cancelled; nothing moved")
            return

        mode = "move" if clicked is move else "copy"
        self._log.appendPlainText(f"--- organising ({mode})")
        results = apply_layouts(plans, mode=mode)
        for name, result in results.items():
            self._log.appendPlainText(
                f"  {name}: {result['moved']} file(s)" if result["ok"]
                else f"  {name}: FAILED, {result['error']}")
        # RE-OPEN THE PARENT, not `_raw_dir`: that points at whichever condition is
        # showing, so re-opening it would drop the other conditions from the window.
        self._open(parent)

    def _pick_condition(self, name: str) -> None:
        """Switch to a condition, remembering the crops set for the one being left.

        The crops are the expensive part of this window: they are a per-movie physical
        judgement a person makes by eye. Losing them on a switch would make the condition
        selector worse than opening two windows.
        """
        if not name or name not in self._conditions:
            return
        if self._raw_dir is not None and self._crops:
            previous = self._conditions_box.property("loaded")
            if previous and previous in self._crops_by_condition:
                self._crops_by_condition[previous] = dict(self._crops)
        movies = self._conditions[name]
        self._raw_dir = movies[0].parent
        self._conditions_box.setProperty("loaded", name)
        self._scan()
        self._crops = dict(self._crops_by_condition.get(name, {}))
        self._refresh_status()

    def _run_all(self):
        """Every condition, each with its own crops.

        REFUSES a condition whose movies are not all cropped, by NAME. `prepare_batch`
        already refuses a movie with no range rather than lending it a neighbour's, and
        this is the same rule one level up: a batch that silently skipped a condition
        would produce a folder that looks complete.
        """
        if not self._conditions:
            return
        current = self._conditions_box.currentText()
        if current in self._crops_by_condition:
            self._crops_by_condition[current] = dict(self._crops)

        groups, missing = [], []
        for name, movies in self._conditions.items():
            crops = self._crops_by_condition.get(name, {})
            if len(crops) != len(movies):
                missing.append(f"{name} ({len(crops)}/{len(movies)})")
                continue
            raw_dir = movies[0].parent
            groups.append((name, raw_dir, raw_dir.parent, dict(crops)))
        if missing:
            self._log.appendPlainText(
                "every movie needs a crop range first; not yet set for: "
                + ", ".join(missing))
            return

        total = sum(len(c) for _n, _r, _o, c in groups)
        self._run_button.setEnabled(False)
        self._run_all_button.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setRange(0, total)
        self._progress.setValue(0)
        self._log.appendPlainText(
            f"--- preparing {total} movie(s) across {len(groups)} condition(s)")
        job = _PrepJob(None, None, None,
                       float(self._pixel.value()),
                       float(self._interval.value()) / 1000.0,
                       groups=groups)
        job.signals.progress.connect(self._on_progress)
        job.signals.finished.connect(self._on_finished)
        job.signals.failed.connect(self._on_failed)
        self._pool.start(job)

    def _run(self):
        if not self._movies or len(self._crops) != len(self._movies):
            return
        out_root = self._raw_dir.parent
        self._run_button.setEnabled(False)
        self._run_all_button.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setRange(0, len(self._movies))
        self._progress.setValue(0)
        self._log.appendPlainText(
            f"--- preparing {len(self._movies)} movies into {out_root}")
        job = _PrepJob(self._raw_dir, out_root, dict(self._crops),
                       float(self._pixel.value()), float(self._interval.value()) / 1000.0)
        job.signals.progress.connect(self._on_progress)
        job.signals.finished.connect(self._on_finished)
        job.signals.failed.connect(self._on_failed)
        self._pool.start(job)

    def _on_progress(self, index, total, note):
        self._progress.setValue(index)
        self._log.appendPlainText(f"[{index}/{total}] {note}")

    def _on_finished(self, table):
        self._last_batch = table
        failed = [r for _, r in table.iterrows() if not r["ok"]]
        if failed:
            self._log.appendPlainText(
                f"{len(failed)} movie(s) FAILED: "
                + ", ".join(str(r["movie"]) for r in failed[:6]))
            for r in failed[:6]:
                self._log.appendPlainText(f"    {r['movie']}: {r['error']}")
        else:
            self._log.appendPlainText("no movie failed")
        # THE CROP IS THE EXPENSIVE DECISION, so it is RECORDED rather than asked for
        # twice. The tracking stage's raw path reads this file, which is what stops the
        # two stages from disagreeing about a range, a disagreement invisible
        # downstream, since both produce a dataset under the same name.
        from .movieprep import CROPS_FILENAME, MoviePrepError, save_crops

        crops_path = self._raw_dir / CROPS_FILENAME
        try:
            save_crops(crops_path, self._crops,
                       pixel_size_nm=float(self._pixel.value()),
                       frame_interval_s=float(self._interval.value()) / 1000.0)
            self._log.appendPlainText(f"crops recorded in {crops_path}")
        except MoviePrepError as exc:
            self._log.appendPlainText(f"could not record crops: {exc}")

        self._log.appendPlainText(
            "Prepared. tif/ holds the cropped stacks, sum/ the projections, and "
            "bleaching/ the curve values as CSV.")
        self._progress.setVisible(False)
        self._restore_buttons()

    def _on_failed(self, message):
        self._log.appendPlainText(f"BATCH FAILED: {message}")
        self._progress.setVisible(False)
        self._restore_buttons()

    def _restore_buttons(self) -> None:
        """BOTH come back, on both paths. Restoring only `_run_button` would leave the
        batch button dead after one run, which reads as the tool having broken."""
        self._run_button.setEnabled(True)
        self._run_all_button.setEnabled(True)


def launch_prep(root=None, *, show=True, block=True):
    """Open the preparation window. Returns 0, like the other entry points."""
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
    window = PrepWindow(root)
    if not show:
        return window
    window.show()
    if not block:
        # AND THE WINDOW IS RETURNED so the caller can hold it. Returning 0 left the
        # caller storing an integer while Python collected the window out from under
        # it, a window that opens and vanishes.
        return window
    app.exec_() if hasattr(app, "exec_") else app.exec()
    return 0
