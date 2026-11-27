"""What a folder holds — cheaply first, then properly on a thread.

Two speeds, because the onboarding flow needs both and they are minutes apart.

`discover()` GLOBS. It answers "is this the right folder, and what is in it" in
milliseconds: the conditions, the movie folders under each. It reads no data, so
it cannot report a track count, and it is what the Confirm screen draws the
instant a folder is chosen.

`Loader` READS. It runs `project.load_project` on a worker thread and reports
each condition as the loader finishes it, so the numbers arrive on the screen
that is already open. Nothing here reimplements any of that: the per-condition
progress comes from the loader's own hook (`load_project(on_condition=…)`),
because a UI that looped the conditions itself would be running a DIFFERENT load
— separate cache keys, and no cross-condition acquisition-parameter check, which
is the thing that catches two conditions imaged at different pixel sizes.

THE THREAD IS JOINED ON CLOSE. A QThread destroyed while running calls `abort()`
— no exception, no traceback, no message. `open_gui` carries the same rule for
the same reason.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from qtpy.QtCore import QObject, QThread, Signal

from ...grouping import GroupingError
from ...onboarding_view import condition_cards, day_table

__all__ = ["Discovery", "Loader", "Scout", "day_rows", "discover",
           "metric_values"]

@dataclass
class Discovery:
    """What a glob can say about a folder. No data has been read."""

    root: Path
    #: condition -> movie folder names, in the order the tracker wrote them.
    conditions: dict = field(default_factory=dict)
    #: Why nothing was found, in the user's words rather than an exception's.
    problem: str = ""
    #: condition -> {"utrack": n, "trackmate": n}: which trackers left files there
    #: (`project.tracker_options`), so a condition tracked twice can be ASKED about.
    trackers: dict = field(default_factory=dict)
    #: What the folder-design page asks (`design_view.design_question`), or None when it need not ask.
    design: dict | None = None

    @property
    def ok(self) -> bool:
        return bool(self.conditions)

    @property
    def series(self) -> int:
        return sum(len(v) for v in self.conditions.values())

    def summary(self) -> str:
        """`3 conditions · 28 series`, the line a recent row carries."""
        c, s = len(self.conditions), self.series
        return f"{c} condition{'s'[:c != 1]} · {s} series"


def _design_question(root):
    """`design_view.design_question`, or None where the folders cannot be walked."""
    try:
        from ...design_view import design_question

        return design_question(root)
    except OSError:
        return None


def discover(root) -> Discovery:
    """Look, do not read. Never raises — a bad folder is an answer, not an error.

    `discover_conditions` refuses a folder with nothing in it, and the commonest
    mistake by far is pointing one level too deep (at a single condition) or too
    shallow (at the drive). Both come back as `problem`, phrased as the thing to
    do next, because an exception message here reaches a user who did nothing
    wrong.
    """
    root = Path(root)
    question = _design_question(root)
    try:
        from ...project import discover_conditions

        found = discover_conditions(root)
    # broad on purpose: "a bad folder is an answer, not an error" (see the docstring)
    except Exception as exc:                           # noqa: BLE001 - reported
        return Discovery(root, problem=(
            f"{type(exc).__name__}: {exc}\nThis should be the folder containing "
            f"your condition sub-folders."), design=question)
    if not found:
        return Discovery(root, problem=(
            "No conditions found here. This should be the folder containing "
            "your condition sub-folders."), design=question)
    return Discovery(root, {name: _movies(path) for name, path in sorted(found.items())},
                     trackers=_trackers(found), design=question)


def _trackers(found: dict) -> dict:
    """`grouping.sources_present` per condition: a listing, never a read."""
    from ...grouping import sources_present

    return {name: sources_present(path) for name, path in sorted(found.items())}


def _movies(condition_root: Path) -> list:
    """The movies of one condition, NAMED AS THE LOADER WILL NAME THEM.

    Asked of `grouping.discover_utrack_project` / `discover_trackmate_project`
    rather than derived here, and both halves of that matter.

    THE DEPTH CANNOT BE ASSUMED. An earlier version of this listed the
    condition's own subfolders and called them series, which on a real project
    (`<condition>/tif/series001/TrackingPackage/…`) listed `cell_meshes`, `raw`,
    `phase` and `.smtracker_warm` as if they were movies. `project.py` records
    this exact trap -- only the movie TAIL is fixed, so only the tail may be
    matched -- and the discovery functions are where that glob already lives.

    THE NAMES MUST MATCH THE TABLE. These rows are filled from the canonical
    `movie` column later, so a name invented here would join against nothing and
    every row would read zero while the totals underneath said thousands. Taking
    the loader's own names makes that mismatch impossible rather than unlikely.
    """
    from ...grouping import (discover_table_project, discover_trackmate_project,
                             discover_utrack_project)

    found: list[str] = []
    for discover_one in (discover_utrack_project, discover_trackmate_project,
                         discover_table_project):
        try:
            found += [movie for movie, _path in discover_one(condition_root)]
        except GroupingError:                          # this folder holds the other kind of output
            continue
    return sorted(set(found))


def metric_values(project, condition: str) -> list:
    """The five acquisition cards for one condition, as `(value, caption)`: THAT condition's own recorded values
    (`onboarding_view.condition_cards`), never the project's merged copy.

    Every one comes off `AcquisitionParams`, which is READ from the tracker's own
    files. A parameter the files did not record is `—` with a caption saying so,
    never a plausible default: this screen is what a user checks before trusting
    a number, so a default here would be the worst place in the application to
    put one.
    """
    return condition_cards(project, condition)


def day_rows(project, condition: str) -> list:
    """One dict per replicate day, for the Confirm screen's table.

    Keys: ``day``, ``series`` (the range, as the folders are numbered), ``date``,
    ``frames``, ``cells``, ``kept``, ``found``, ``note``.

    TWO ARE ALWAYS ``None``, and that is the data rather than an omission.
    ``date`` -- the canonical table carries no acquisition date, and a file's
    modification time is not one. ``found`` -- the table is POST-filter, so a
    track the outline or length filter dropped has no row here at all; "kept"
    is the honest half, and the other stays a dash until the assignment funnel
    is wired to this screen.

    ``frames`` is the HIGHEST FRAME INDEX SEEN, not the acquisition length: a
    movie whose last seconds hold no localizations reads short. It is labelled
    as what the files recorded for that reason.
    """
    return day_table(project.condition(condition))


class Scout(QObject):
    """`discover`, on a worker, reporting each condition as it is counted.

    THE GLOB IS NOT FREE. `discover_conditions` searches recursively and then
    each condition's movies are listed the same way -- 1.8 s on a 30-series
    project over a spinning disk, all of it on the GUI thread if it is called
    there, and all of it BEFORE the screen that would have said what was
    happening. So it runs here, the screen opens empty, and the rows arrive.

    Signals::

        started(root)              a folder is being looked at
        found(condition, count)    one condition, and how many movies it holds
        done(discovery)            the whole answer, ok or not
    """

    started = Signal(object)
    found = Signal(str, int)
    done = Signal(object)

    def __init__(self, root, parent=None):
        super().__init__(parent)
        self.root = Path(root)
        self._thread: QThread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = QThread()
        self.moveToThread(self._thread)
        self._thread.started.connect(self._run)
        self._thread.start()

    def stop(self) -> None:
        thread, self._thread = self._thread, None
        if thread is None:
            return
        thread.quit()
        thread.wait(10_000)

    def _run(self) -> None:
        self.started.emit(self.root)
        # THE FOLDER-DESIGN QUESTION, here on the worker: it walks the folders (`design_view.design_question`).
        # Asked whatever discovery says, since a design is what fixes a folder discovery gets wrong.
        question = _design_question(self.root)
        try:
            from ...project import discover_conditions

            found = discover_conditions(self.root)
        # broad on purpose: runs on a QThread; an unexpected error must reach the user as a message (a thread cannot raise into the GUI)
        except Exception as exc:                       # noqa: BLE001 - reported
            self.done.emit(Discovery(self.root, problem=(
                f"{type(exc).__name__}: {exc}\nThis should be the folder "
                f"containing your condition sub-folders."), design=question))
            return
        if not found:
            self.done.emit(Discovery(self.root, problem=(
                "No conditions found here. This should be the folder "
                "containing your condition sub-folders."), design=question))
            return
        conditions = {}
        for name, path in sorted(found.items()):
            movies = _movies(path)
            conditions[name] = movies
            # EMITTED AS EACH IS COUNTED, so a project with eight conditions
            # fills its card row by row instead of appearing all at once after
            # the last one.
            self.found.emit(name, len(movies))
        self.done.emit(Discovery(self.root, conditions, trackers=_trackers(found), design=question))


class Loader(QObject):
    """`load_project` on a worker thread, reporting each condition as it lands.

    Signals, in the order a caller sees them::

        progress(name, done, total)   a condition has been read
        loaded(project)               everything is in
        failed(message)               nothing is, and here is why

    `failed` carries a MESSAGE rather than an exception, because the window that
    shows it has no business formatting a traceback -- and because the useful
    part of a `ProjectError` here is already a sentence written for a user.
    """

    #: `(what, done, total)`. `what` is a sentence fragment the screen prints
    #: verbatim -- a movie name while reading, a condition name otherwise --
    #: because a progress line that has one prefix bolted onto whatever it is
    #: given comes out reading "Reading building the screen".
    progress = Signal(str, int, int)
    #: ``(loaded project, {condition: model.Project})``. The ADAPTER RUNS HERE,
    #: on the worker, not on the GUI thread that receives this: it reads a
    #: `movieData.mat` per movie for the acquisition timestamps -- 0.8 s on a
    #: 30-series project -- and disk I/O on the GUI thread is a stopped spinner.
    loaded = Signal(object, object)
    failed = Signal(str)
    #: ``{condition: [(movie, field, path)]}``: movies tracked without a pixel size or frame interval
    #: (`utrack_meta.MissingAcquisitionError.by_condition`), so the page can ASK to leave them out
    missing = Signal(object)

    def __init__(self, root, *, series_per_day: int = 10, day_maps=None,
                 sources=None, leave_out=None, parent=None):
        super().__init__(parent)
        self.root = Path(root)
        #: ``{condition: {movie: reason}}`` left out on the user's answer (`load_project(leave_out=...)`)
        self.leave_out = leave_out
        self.series_per_day = series_per_day
        #: ``{condition: "utrack" | "trackmate"}`` chosen on `TrackerChoicePage`, or None
        #: for the loader's own detection (one tracker everywhere).
        self.sources = sources
        # RESOLVED ON THE GUI THREAD, BEFORE `start()` -- see `window._scouted`.
        # Without it this loader re-derives days from `series_per_day` alone,
        # which is the literal per-10 rule rather than the naming authority /
        # any remembered `day_map.json`, and raises `GroupingError` outright on
        # a condition whose series blocks are non-contiguous (e.g. 001-009 /
        # 031-040 / 041-050) instead of asking or falling back gracefully.
        self.day_maps = day_maps
        self._thread: QThread | None = None

    def start(self) -> None:
        """Move to a thread and begin. Safe to call once."""
        if self._thread is not None:
            return
        self._thread = QThread()
        self.moveToThread(self._thread)
        self._thread.started.connect(self._run)
        self._thread.start()

    def stop(self) -> None:
        """JOIN the thread. Destroying a running QThread calls abort()."""
        thread, self._thread = self._thread, None
        if thread is None:
            return
        thread.quit()
        # A load that is mid-`.mat` cannot be interrupted, so this waits for the
        # step in flight rather than pretending it can cancel one.
        thread.wait(30_000)

    def _run(self) -> None:
        from ...utrack_meta import MissingAcquisitionError

        try:
            from ... import options as opts
            from ...project import APP_FRAME_INTERVAL, APP_MIXED_TAU, load_project
            from .. import theme

            project = load_project(
                self.root,
                mixed_tau=APP_MIXED_TAU, frame_interval_from=APP_FRAME_INTERVAL,
                max_tau_spread=opts.max_tau_spread(theme.remembered(opts.OPTIONS_KEY)),
                outline_tolerance_nm=opts.outline_tolerance_nm(
                    theme.remembered(opts.OPTIONS_KEY)),
                series_per_day=self.series_per_day,
                day_maps=self.day_maps,
                source=self.sources or "auto",
                leave_out=self.leave_out,
                # A MOVIE WITHOUT OUTLINES IS LEFT OUT AND SAID (asked 22 Oct 2026: "just load it and give a
                # window telling us"), not a refusal of the whole folder: `window._show_confirm` names it
                skip_unsegmented=True,
                # PER MOVIE, not per condition. The per-condition hook fires
                # once on a single-condition project and then says nothing for
                # the whole read -- six seconds of a line that does not move,
                # which is indistinguishable from a frozen window. This one
                # ticks thirty times on the same project.
                on_movie=lambda cond, movie, done, total: self.progress.emit(
                    f"{cond} · {movie}", done, total),
                # The steps after the files, which have no count -- `total` of 0
                # says "name this, do not measure it".
                on_stage=lambda text: self.progress.emit(text, 0, 0),
            )
        except MissingAcquisitionError as exc:         # a question, not a failure: leave the movies out?
            self.missing.emit(dict(getattr(exc, "by_condition", None) or {"": exc.missing}))
            return
        # broad on purpose: runs on a QThread; an unexpected error must reach the user as a message (a thread cannot raise into the GUI)
        except Exception as exc:                       # noqa: BLE001 - shown to a user
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        try:
            from .adapt import projects_from

            self.progress.emit("the imaging dates", 0, 0)
            projects = projects_from(project, Discovery(self.root))
        # broad on purpose: runs on a QThread; an unexpected error must reach the user as a message (a thread cannot raise into the GUI)
        except Exception as exc:                       # noqa: BLE001 - shown
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.loaded.emit(project, projects)
