"""Everything slow the preparation window does, off the GUI thread.

WHY A PROCESS POOL FOR THE MOVIES. Reading one movie's curves is 2-8 s of numpy over
every frame. On a worker THREAD of the window's own process that work holds the GIL in
long stretches and the window stutters for as long as it runs -- the same measurement the
analysis workspace records for its fits (`ui.workspace.window`, "OFF THE GUI PROCESS").
A spawned process has its own GIL, so the window stays smooth however many movies are
in flight, and two or three run at once.

Opening a folder and saving are directory listings and file writes, so they run on a
`QThreadPool` thread: the GIL is released while they wait on the disk.

EVERY RESULT CARRIES ITS GENERATION. Opening another folder starts a new generation;
anything still arriving from the last one is dropped by the window rather than filed
under the wrong condition (the "13 of 10 read" defect the classic window recorded).

The callables a pool runs are defined OUTSIDE `smtracker.ui` (`prep_model`), as
`tests/test_ui_thin_qt.py` requires.
"""
from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

from qtpy.QtCore import QObject, QRunnable, QThreadPool, Signal

__all__ = ["PrepSignals", "PrepJobs", "default_workers"]


def default_workers() -> int:
    """Two or three readers: each holds one movie in memory (a 4500-frame acquisition is
    about 150-200 MB), and the disk, not the CPU, is usually the limit past three."""
    return max(1, min(3, (os.cpu_count() or 2) - 1))


class PrepSignals(QObject):
    #: generation, ``{condition: ConditionState}`` or None, error text
    opened = Signal(int, object, str)
    #: generation, condition, movie, curves or None, error text
    curve = Signal(int, str, str, object, str)
    #: generation, condition, path or None, error text
    saved = Signal(int, str, object, str)
    #: generation, condition, done, total, what
    save_progress = Signal(int, str, int, int, str)
    #: generation, condition, ``(loaded, missing)`` from the curve cache, error text
    cached = Signal(int, str, object, str)
    #: generation, condition, what the acquisition measured (``{field: value}``), error
    probed = Signal(int, str, object, str)
    #: generation, condition, movie, its noise row (`movie_threshold_row`), error
    noise_row = Signal(int, str, str, object, str)
    #: the `FijiInstall` found or None, what to say when none was, error
    fiji = Signal(object, str, str)
    #: condition, movie index, movies in all, the movie's plain row
    track_progress = Signal(str, int, int, object)
    #: condition, `prep_model.track_condition`'s plain result or None, error
    tracked = Signal(str, object, str)
    #: generation, condition, one movie's export row (`prep_model.export_movie`)
    export_row = Signal(int, str, object)
    #: generation, condition, `prep_model.pilot_steps`'s result or None, error
    pilot = Signal(int, str, object, str)
    #: generation, condition, `prep_model.tail_purity`'s result or None, error
    purity = Signal(int, str, object, str)
    #: condition, run index, runs in all, the movie's plain row
    radius_progress = Signal(str, int, int, object)
    #: condition, `prep_model.radius_check`'s result or None, error
    radius_checked = Signal(str, object, str)
    #: the `MatlabInstall` and `UTrackInstall` found (or None), what to say when one is
    #: missing, error
    utrack_found = Signal(object, object, str, str)
    #: `utrack_run.check_utrack`'s answer or None, error
    utrack_health = Signal(object, str)
    #: generation, ``{"rows": batch_model.plan_tracking rows}`` or None, error
    batch_planned = Signal(int, object, str)
    #: kind, condition key, payload: `batch_model.run_tracking`'s events
    batch_event = Signal(str, str, object)
    #: the batch's report rows or None, error
    batch_done = Signal(object, str)
    #: pip's output, one line at a time, so a minute of installing is not silent.
    install_line = Signal(int, str)
    #: `(generation, ok, last words)` when the offered install finishes.
    installed = Signal(int, bool, str)


class _Call(QRunnable):
    """Run ``fn`` on a pool thread and hand the result (or the error) to ``done``."""

    def __init__(self, fn, done):
        super().__init__()
        self._fn, self._done = fn, done

    def run(self):                                      # noqa: D102 - QRunnable protocol
        try:
            result = self._fn()
        except Exception as exc:                        # noqa: BLE001 - reported by name
            self._done(None, f"{type(exc).__name__}: {exc}")
            return
        self._done(result, "")


class PrepJobs(QObject):
    """The window's one door to slow work. ``executor`` is injectable so a test runs the
    reads in-process; by default a spawned process pool is made on the FIRST read, after
    the window has painted, since each worker pays about a second of imports."""

    def __init__(self, parent=None, *, executor=None, workers: int | None = None,
                 thread_pool: QThreadPool | None = None):
        super().__init__(parent)
        self.signals = PrepSignals()
        self._executor = executor
        self._owns_executor = executor is None
        self._workers = workers or default_workers()
        self._threads = thread_pool or QThreadPool.globalInstance()
        #: Futures not yet finished, so closing can cancel what has not started.
        self._pending: set = set()
        #: The subset that are movie reads, for the "reading N movies" line.
        self._reads: set = set()
        #: The stack exports in flight, so Stop cancels those and nothing else.
        self._exports: set = set()

    # -- the pool ------------------------------------------------------------- #
    def _pool(self):
        if self._executor is None:
            self._executor = ProcessPoolExecutor(
                max_workers=self._workers, mp_context=multiprocessing.get_context("spawn"))
        return self._executor

    def warm(self) -> None:
        """Start the worker processes from a BACKGROUND thread, before anything is read.

        `ProcessPoolExecutor` spawns a worker on `submit` when none is idle, and it does
        so on the CALLING thread: three spawns on the window's thread measured as a
        0.7-1 s freeze the first time a folder opened. Submitting one trivial task per
        worker from a pool thread pays that cost where nobody sees it.
        """
        pool = self._pool()
        n = self._workers
        self._threads.start(_Call(lambda: [pool.submit(os.getpid) for _ in range(n)],
                                  lambda _r, _e: None))

    @property
    def reading(self) -> int:
        """Movies submitted and not yet back. A non-zero count means acquisitions may be
        open in another process, which on Windows forbids moving them (WinError 32)."""
        return len(self._reads)

    # -- requests ------------------------------------------------------------- #
    def open_folder(self, generation: int, folder) -> None:
        from ...prep_model import open_session

        self._threads.start(_Call(
            lambda: open_session(folder),
            lambda result, error: self.signals.opened.emit(generation, result, error)))

    def _submit(self, fn, args: tuple, emit, *, movie_read: bool = False,
                bucket: set | None = None, on_cancel=None) -> None:
        """Run ``fn(*args)`` in the PROCESS pool and hand ``(result, error)`` to ``emit``.

        Everything that parses an acquisition goes here, the metadata probe included:
        on a thread of this process, `tifffile` walking a 4000-page `.stk` held the GIL
        long enough to stall the window for 1.5 s, measured on one user's d2 folder."""
        future = self._pool().submit(fn, *args)
        self._pending.add(future)
        if movie_read:
            self._reads.add(future)
        if bucket is not None:
            bucket.add(future)

        def done(f):
            self._pending.discard(f)
            self._reads.discard(f)
            if bucket is not None:
                bucket.discard(f)
            if f.cancelled():
                # A CANCELLED JOB STILL REPORTS when asked to: a count waiting on it would
                # otherwise never complete, and "not done" must not read as "still going".
                if on_cancel is not None:
                    try:
                        on_cancel()
                    except RuntimeError:                # the window is gone
                        pass
                return
            try:
                result, error = f.result(), ""
            except Exception as exc:                    # noqa: BLE001 - reported by name
                result, error = None, f"{type(exc).__name__}: {exc}"
            try:
                emit(result, error)
            except RuntimeError:                        # the window is gone
                pass

        future.add_done_callback(done)

    def read(self, generation: int, condition: str, movies, flat_tolerance=None) -> None:
        """Read every ``(path, movie name)``; each result arrives on `signals.curve`. The
        names come from the session, which worked them out once."""
        from ...prep_model import read_movie_curves

        for path, stem in movies:
            self._submit(read_movie_curves, (path, flat_tolerance),
                         lambda r, e, stem=stem: self.signals.curve.emit(
                             generation, condition, stem, r, e),
                         movie_read=True)

    def save(self, generation: int, state, *, projection=None) -> None:
        """On a THREAD, not the pool: saving writes files the session already holds
        (and the projection, which reads frames, reports progress as it goes)."""
        from ...prep_model import save

        def progress(done, total, what):
            self.signals.save_progress.emit(generation, state.name, done, total, what)

        self._threads.start(_Call(
            lambda: save(state, projection=projection, on_progress=progress),
            lambda path, error: self.signals.saved.emit(generation, state.name, path,
                                                        error)))

    def install(self, generation: int, packages: list, runner, on_line) -> None:
        """Run the offered install on a THREAD, not the pool.

        A pool worker is a SPAWNED PROCESS here, and pip's output would have to be
        pickled back line by line; a thread streams it directly and spends its minute
        waiting on a subprocess, where the GIL is free anyway. ``runner`` is passed in
        (`optional_deps.install`) rather than imported, so a test drives this without
        pip and without the network.
        """
        def work():
            return runner(packages, on_line=on_line)

        self._threads.start(_Call(
            work,
            lambda result, error: self.signals.installed.emit(
                generation, bool(result and result[0]),
                error or (result[1] if result else "no result"))))

    def load_cache(self, generation: int, condition: str, movies, bleach_dir,
                   flat_tolerance=None) -> None:
        """The curves already saved in `bleaching/*.csv` (`prep_model.curves_from_cache`):
        CSVs and a floor per movie, no frame read, so a reopened folder is instant."""
        from ...prep_model import curves_from_cache

        from functools import partial

        self._submit(partial(curves_from_cache, flat_tolerance=flat_tolerance),
                     (list(movies), bleach_dir),
                     lambda r, e: self.signals.cached.emit(generation, condition, r, e))

    def probe(self, generation: int, condition: str, movie, recorded_interval_s=None) -> None:
        """What one acquisition MEASURED (its frame interval and NA, never its pixel
        size), and whether a recorded interval agrees (`prep_model.probe_acquisition`)."""
        from ...prep_model import probe_acquisition

        self._submit(probe_acquisition, (movie, recorded_interval_s),
                     lambda r, e: self.signals.probed.emit(generation, condition, r, e))

    def measure_noise(self, generation: int, condition: str, items, radius_px: float,
                      k: float | None = None) -> None:
        """The LoG noise of every ``(path, movie, (first, last))``, ONE MOVIE PER WORKER,
        inside its crop. The window summarises the rows with `prep_model.summarise_noise`
        once all are in. The classic window measured a condition's movies one after the
        other on its own thread (about 1.6 s a movie on a MetaMorph `.stk`)."""
        from functools import partial

        from ...trackmate_run import UTRACK_MATCHED_K, movie_threshold_row

        k = UTRACK_MATCHED_K if k is None else float(k)
        for path, stem, crop in items:
            self._submit(partial(movie_threshold_row, name=stem, radius_px=float(radius_px),
                                 k=k, frame_range=tuple(crop)),
                         (path,),
                         lambda r, e, stem=stem: self.signals.noise_row.emit(
                             generation, condition, stem, r, e))

    # -- Fiji and the TrackMate run: subprocesses, so THREADS -------- #
    def _emit(self, signal, *args) -> None:
        """Emit from a pool thread, unless the window (and these signals) is gone: a run
        outlives a closed window by the movie in hand."""
        try:
            signal.emit(*args)
        except RuntimeError:
            pass

    def find_fiji(self, extra_paths=()) -> None:
        """`fiji.find_fiji` stats the usual install folders and globs the plugins for
        TrackMate, so it runs here rather than before the window paints."""
        from ...fiji import describe_absence, find_fiji

        def work():
            install = find_fiji(extra_paths=tuple(extra_paths))
            return install, ("" if install is not None else describe_absence())

        self._threads.start(_Call(
            work, lambda r, e: self._emit(self.signals.fiji, *(r or (None, "")), e)))

    def track(self, inputs, install, stop, at_once: int = 1) -> None:
        """Track one condition (`prep_model.track_condition`) on a THREAD: each movie is a
        Fiji subprocess, so the GIL is free while it waits. ``stop()`` is asked before
        every movie; the movie in hand always finishes."""
        from ...prep_model import track_condition

        name = inputs.condition

        def progress(index, total, row):
            self._emit(self.signals.track_progress, name, index, total, row)

        self._threads.start(_Call(
            lambda: track_condition(inputs, install=install, on_progress=progress,
                                    stop=stop, at_once=at_once),
            lambda r, e: self._emit(self.signals.tracked, name, r, e)))

    def find_utrack(self, matlab_paths=(), utrack_paths=()) -> None:
        """`utrack_run.find_matlab` lists every drive's MATLAB folders and `find_utrack`
        reads MATLAB's saved path, so both run here rather than before the window
        paints. Chosen paths are asked first."""
        from ...utrack_run import describe_absence, find_matlab, find_utrack

        def work():
            matlab = find_matlab(extra_paths=tuple(matlab_paths))
            utrack = find_utrack(extra_paths=tuple(utrack_paths), matlab=matlab)
            absence = ("" if matlab is not None and utrack is not None
                       else describe_absence(matlab=matlab, utrack=utrack))
            return matlab, utrack, absence

        self._threads.start(_Call(
            work, lambda r, e: self._emit(self.signals.utrack_found,
                                          *(r or (None, None, "")), e)))

    def check_utrack(self, matlab, utrack) -> None:
        """Start MATLAB headless and ask u-track and the toolboxes to answer (~10 s)."""
        from ...utrack_run import check_utrack

        self._threads.start(_Call(
            lambda: check_utrack(matlab, utrack),
            lambda r, e: self._emit(self.signals.utrack_health, r, e)))

    def track_utrack(self, inputs, matlab, utrack, replace: bool, stop,
                     work_folder=None, at_once: int = 1) -> None:
        """Track one condition with u-track (`prep_model.track_condition_utrack`) on a
        THREAD: the work is one MATLAB subprocess, so the GIL is free while it waits.
        Reports on `track_progress` and `tracked`, as a TrackMate run does."""
        from ...prep_model import track_condition_utrack

        name = inputs.condition

        def progress(index, total, row):
            self._emit(self.signals.track_progress, name, index, total, row)

        self._threads.start(_Call(
            lambda: track_condition_utrack(inputs, matlab=matlab, utrack=utrack,
                                           replace=replace, on_progress=progress,
                                           stop=stop, work_folder=work_folder,
                                           at_once=at_once),
            lambda r, e: self._emit(self.signals.tracked, name, r, e)))

    # -- stacks for u-track, and the linking radius's helpers ------------------------ #
    def export(self, generation: int, condition: str, inputs: dict, projection) -> None:
        """Cut every ``(path, movie, (first, last))`` into ``tif/``, ONE MOVIE PER WORKER
        (`prep_model.export_movie`); each row arrives on `signals.export_row`, and a movie
        Stop kept from starting arrives as a row saying so."""
        from functools import partial

        from ...prep_model import export_movie

        for path, stem, (first, last) in inputs["movies"]:
            def stopped(stem=stem, first=first, last=last):
                self.signals.export_row.emit(
                    generation, condition,
                    {"movie": stem, "ok": False, "error": "not written: stopped",
                     "n_frames_kept": None, "crop_start": first, "crop_end": last})

            self._submit(partial(export_movie, name=stem, start=first, end=last,
                                 pixel_size_nm=inputs["pixel_size_nm"],
                                 frame_interval_s=inputs["frame_interval_s"],
                                 projection=projection or "sd"),
                         (path, inputs["out_root"]),
                         lambda r, e, stem=stem, first=first, last=last:
                             self.signals.export_row.emit(
                                 generation, condition, r if r is not None else
                                 {"movie": stem, "ok": False, "error": e,
                                  "n_frames_kept": None, "crop_start": first,
                                  "crop_end": last}),
                         bucket=self._exports, on_cancel=stopped)

    def stop_exports(self) -> None:
        """Cancel every export not yet started; the movies in hand finish whole -- a
        half-written stack is worse than a missing one."""
        for future in list(self._exports):
            future.cancel()

    def pilot(self, generation: int, condition: str, folder, radius_px: float) -> None:
        """`prep_model.pilot_steps` in the pool: reading up to eight track files."""
        from ...prep_model import pilot_steps

        self._submit(pilot_steps, (folder, float(radius_px)),
                     lambda r, e: self.signals.pilot.emit(generation, condition, r, e))

    def tail_purity(self, generation: int, condition: str, folder, crops: dict) -> None:
        """`prep_model.tail_purity` in the pool: parsing every track file of a condition."""
        from ...prep_model import tail_purity

        self._submit(tail_purity, (folder, dict(crops)),
                     lambda r, e: self.signals.purity.emit(generation, condition, r, e))

    def radius_check(self, inputs, install, stop) -> None:
        """`prep_model.radius_check` on a THREAD: four Fiji runs."""
        from ...prep_model import radius_check

        name = inputs.condition

        def progress(index, total, row):
            self._emit(self.signals.radius_progress, name, index, total, row)

        self._threads.start(_Call(
            lambda: radius_check(inputs, install=install, on_progress=progress, stop=stop),
            lambda r, e: self._emit(self.signals.radius_checked, name, r, e)))

    # -- the batch over several parent folders: listings, then Fiji or MATLAB --------- #
    def batch_plan(self, generation: int, parents, live, settings: dict) -> None:
        """List every parent (`batch_model.scan_tracking`) and say which conditions would
        run (`plan_tracking`), on a THREAD: listings and crops.json reads, never the GUI's."""
        from ...batch_model import plan_tracking, scan_tracking

        parents, live, settings = list(parents), dict(live or {}), dict(settings)

        def work():
            return {"rows": plan_tracking(scan_tracking(parents, live=live), **settings)}

        self._threads.start(_Call(
            work, lambda r, e: self._emit(self.signals.batch_planned, generation, r, e)))

    def batch_run(self, parents, live, settings: dict, stop) -> None:
        """List the parents again (what is on disk NOW) and run the batch
        (`batch_model.run_tracking`) on a THREAD; its events arrive on `batch_event`, its
        rows on `batch_done`. Fiji and MATLAB are subprocesses, so the GIL is free while
        they work."""
        from ...batch_model import run_tracking, scan_tracking

        parents, live, settings = list(parents), dict(live or {}), dict(settings)

        def event(kind, key, payload):
            self._emit(self.signals.batch_event, kind, key, payload)

        def work():
            return run_tracking(scan_tracking(parents, live=live), stop=stop,
                                on_event=event, **settings)

        self._threads.start(_Call(
            work, lambda r, e: self._emit(self.signals.batch_done, r, e)))

    def cancel_pending(self) -> None:
        """Drop every read that has not started; a running one finishes and is ignored."""
        for future in list(self._pending):
            if future.cancel():
                self._pending.discard(future)
                self._reads.discard(future)

    def shutdown(self) -> None:
        self.cancel_pending()
        if self._owns_executor and self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
