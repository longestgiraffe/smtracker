"""Overnight batches over several PARENT folders: tracking and segmentation, headless.

Asked for (13 Oct 2026): "a batch mode where u can run multiple parent folders in batches
overnight. Both for image prep / tracking and segmentation". The preparation window's and
the outlines window's Batch pages (`ui.prep.batch_page`, `ui.outlines.batch_page`) show what
this module decides and start what it runs; every rule and every word is here.

RULED: A CONDITION RUNS ONLY WHERE EVERY MOVIE HAS A SAVED CROP RANGE (crops.json) AND THE
CALIBRATION ITS RUN NEEDS. Anything missing is NAMED in its row and in the report, and the
condition is SKIPPED -- never proposed, never guessed. An overnight batch nobody watches is
exactly where a guessed range would go unseen.

THE SAME REFUSALS AND THE SAME RUNS AS ONE CONDITION ON THE TRACK PAGE, not a second rule set:
readiness is `prep_model.run_inputs` (TrackMate) or `prep_model.utrack_inputs` (u-track), and
for outlines `outline_model.readiness`; a run is `prep_model.track_condition`,
`prep_model.track_condition_utrack` or `outline_model.segment_condition`, each looked up on
its MODULE at call time (so one fake replaces it for the window and the batch alike).
- An earlier run is not replaced unless asked ("Replace earlier runs"), as on the Track page.
- TrackMate's detection threshold is MEASURED per condition and lives only in a session, so a
  folder opened fresh holds none and is refused "no detection threshold", as a run would be.
  "Measure a missing threshold" measures it inside the SAVED crops at the options' k, exactly
  as Detect & link's Measure does (`movie_threshold_row` -> `summarise_noise` ->
  `record_noise`), before that condition's run. TICKED BY DEFAULT in the window
  (`MEASURE_MISSING_THRESHOLD`, ruled 13 Oct 2026: it is the measurement Measure would make,
  and without it a batch over folders never opened runs nothing); the run record says
  `adaptive_per_movie`, so it can be checked afterwards. The headless `plan_tracking` keeps
  ``measure_thresholds=False`` as its default: a script asks for a measurement by name.
- A condition open in the preparation window is read from the WINDOW's session (`snapshot`),
  so its Detect & link values and an unsaved edit count exactly as on the Track page.

DISCOVERY IS THE WINDOWS' OWN: `prep_model.open_session` for tracking, `outline_model.
open_outlines` for segmentation, once per parent. No deeper search (the folders above a movie
carry no meaning). Each condition carries its `organise.folder_layout`.

MOVIES AT ONCE, CONDITIONS IN TURN (23 Oct 2026; was: up to ``sessions`` u-track conditions side by side,
TrackMate one movie at a time). Conditions run one after another, and within each ``sessions`` movies run side
by side -- MATLAB sessions sharing the condition's movies, or one Fiji per movie -- the Track page's "Movies
at once" (`prep_model.MAX_AT_ONCE`). The same number of processes, for both trackers, and no tail where the
last large condition runs alone. STOP takes effect after the movies in hand, as on the Track page; a
condition never started is reported "not run: stopped".

THE REPORT IS REWRITTEN AFTER EVERY CONDITION (`write_report`), so a batch that dies at 3 a.m.
still says what it did. One row per condition, the skipped ones included.

KEEPING WINDOWS AWAKE: `keep_awake` asks `SetThreadExecutionState(ES_CONTINUOUS |
ES_SYSTEM_REQUIRED)` for the calling thread (the window's own, which outlives the batch) and
clears it at the end. A no-op off Windows and under pytest.

INVARIANTS
- Imports no Qt (`tests/test_ui_thin_qt.py`, rule 5) and never `smtracker.ui`.
- Writes nothing itself but the CSV report; everything else is written by the runs it calls.
- A condition's key is its folder's normalised absolute path: one name under two parents is two
  rows, and one folder reached through two parents is one row (the first parent keeps it).
- Plain dicts and strings out, so the pages only paint.
"""
from __future__ import annotations

import csv
import os
import sys
import threading
from dataclasses import dataclass
from dataclasses import replace as copy_with
from datetime import datetime
from pathlib import Path

from .run_progress import duration_words, progress_payload

__all__ = ["BatchCondition", "folder_key", "add_parent", "snapshot", "scan_tracking",
           "tracking_readiness", "plan_tracking", "plan_summary", "layout_notes",
           "measure_threshold", "run_tracking", "STATUS_SKIPPED", "STATUS_WAITING",
           "STATUS_TRACKED", "STATUS_PARTIAL", "STATUS_STOPPED", "STATUS_FAILED",
           "STATUS_NOT_RUN", "REPORT_COLUMNS", "REPORT_PREFIX", "report_path",
           "write_report", "plan_lines", "event_line", "running_status", "result_status",
           "batch_progress",
           "finished_summary", "work_folder_line", "settings_line", "track_batch_cells",
           "scan_outlines", "outline_readiness", "plan_outlines", "outline_batch_cells",
           "outline_summary", "run_outlines", "OUTLINE_SEGMENTED",
           "OUTLINE_PARTIAL", "keep_awake", "awake_note", "ES_CONTINUOUS", "ES_SYSTEM_REQUIRED",
           "MAX_SESSIONS", "DEFAULT_SESSIONS"]

#: Movies tracked at once (`prep_model.MAX_AT_ONCE`, `default_at_once`): MATLAB sessions or Fijis. On one
#: spinning disk three u-track sessions ran no faster than one (`utrack_run.run_batch`): a work folder on an SSD.
from .prep_model import MAX_AT_ONCE as MAX_SESSIONS  # noqa: E402
DEFAULT_SESSIONS = 2

#: What a condition's row says it is at, in the table and in the report.
STATUS_SKIPPED = "skipped"
STATUS_WAITING = "waiting"
STATUS_TRACKED = "tracked"
STATUS_PARTIAL = "tracked, with failures"
STATUS_STOPPED = "stopped part way"
STATUS_FAILED = "failed"
STATUS_NOT_RUN = "not run: stopped"
OUTLINE_SEGMENTED = "segmented"
OUTLINE_PARTIAL = "segmented, with failures"


def folder_key(path) -> str:
    """A folder's identity: its absolute path, case-folded where the OS ignores case. No
    `resolve()`, which can stall on a network or USB drive."""
    return os.path.normcase(os.path.abspath(str(path)))


@dataclass
class BatchCondition:
    """One condition of a batch: where it is, what it holds, and its layout. ``state`` is
    a `prep_model.ConditionState` (tracking) or an `outline_model.OutlineCondition`
    (segmentation); None for a parent in which nothing could be found, ``error`` saying why."""

    parent: Path
    name: str
    folder: Path | None
    state: object = None
    layout: object = None
    error: str = ""

    @property
    def key(self) -> str:
        return folder_key(self.folder if self.folder is not None else self.parent)

    @property
    def label(self) -> str:
        """``parent / condition``, or the condition alone when the parent IS it."""
        if self.folder is not None and folder_key(self.folder) == folder_key(self.parent):
            return self.name
        return f"{self.parent.name} / {self.name}"


def add_parent(parents, folder) -> tuple:
    """``(parents with folder added, note)``; a folder already listed is refused by name,
    and so is one that is not a folder."""
    parents = [Path(p) for p in parents]
    folder = Path(folder)
    if not folder.is_dir():
        return parents, f"{folder} is not a folder"
    if any(folder_key(p) == folder_key(folder) for p in parents):
        return parents, f"{folder.name} is listed already"
    return parents + [folder], ""


def _row_base(cond: BatchCondition) -> dict:
    layout = cond.layout
    return {"key": cond.key, "label": cond.label, "parent": cond.parent.name,
            "parent_path": str(cond.parent), "name": cond.name,
            "folder": "" if cond.folder is None else str(cond.folder),
            "layout": getattr(layout, "kind", "") or "",
            "layout_words": getattr(layout, "words", "") or ""}


def _reason(exc) -> str:
    """A refusal without its leading ``condition: ``, since the row names the condition."""
    return str(exc).split(": ", 1)[-1]


# ================================================================ tracking
def snapshot(state):
    """A copy of a window's `ConditionState` that a batch may read and change on its own
    thread: the ranges, the calibration (saved and on screen, so an unsaved edit is still
    refused by name), the Detect & link values and the noise. The curves stay behind -- no
    run reads them."""
    import copy

    return copy_with(state, crops=dict(state.crops), saved=dict(state.saved),
                   saved_calibration=dict(state.saved_calibration),
                   calibration=dict(state.calibration), curves={},
                   errors=dict(state.errors), proposals={}, tracking=dict(state.tracking),
                   typed=set(state.typed), noise=copy.deepcopy(state.noise),
                   camera=dict(state.camera), movies=list(state.movies),
                   movie_names=list(state.movie_names))


def _projects_below(parent: Path, opener, errors) -> list:
    """The sub-folders of ``parent`` that ``opener`` would open as parents themselves (a parent of projects: a folder
    holding two projects, each holding the conditions), by name. Only one level down: deeper is
    a choice the user makes by adding the folder."""
    found = []
    try:
        children = sorted(c for c in parent.iterdir() if c.is_dir() and not c.name.startswith("."))
    except OSError:
        return []
    for child in children:
        try:
            if opener(child):
                found.append(child.name)
        except errors:
            continue
    return found


def parent_of_projects(parent: Path, names: list, *, what: str) -> str:
    """The row's words for a parent holding no conditions of its own but projects one level down (tester B20,
    3 Nov 2026: it said "nothing here holds acquisitions ..." of a folder whose two projects held them all)."""
    listed = ", ".join(f'"{n}"' for n in names[:4]) + (" ..." if len(names) > 4 else "")   # names hold commas
    return (f"nothing to {what} directly in {parent.name}: it holds {len(names)} project folder"
            f"{'s' if len(names) != 1 else ''} one level down ({listed}); add each as a parent folder")


def scan_tracking(parents, *, live=None) -> list:
    """Every condition under every parent, as `prep_model.open_session` finds them (the
    preparation window's own discovery), each with its layout. ``live`` maps a folder to a
    `snapshot` of the window's session, which wins over the folder read fresh. A parent in
    which nothing can be prepared is ONE row naming why. Listings and crops.json only."""
    from .organise import folder_layout
    from .prep_model import PrepError, open_session

    live = {folder_key(k): v for k, v in (live or {}).items()}
    out, seen = [], set()
    for parent in parents:
        parent = Path(parent)
        try:
            states = open_session(parent)
        except (PrepError, OSError) as exc:
            below = _projects_below(parent, open_session, (PrepError, OSError))
            out.append(BatchCondition(parent=parent, name=parent.name, folder=None,
                                      error=(parent_of_projects(parent, below, what="track") if below
                                             else f"nothing to track: {exc}")))
            continue
        for name, state in states.items():
            key = folder_key(state.folder)
            if key in seen:
                continue
            seen.add(key)
            chosen = live.get(key, state)
            layout = getattr(chosen, "layout", None) or folder_layout(state.folder)
            out.append(BatchCondition(parent=parent, name=name, folder=Path(state.folder),
                                      state=chosen, layout=layout))
    return out


def tracking_readiness(cond: BatchCondition, *, tracker: str = "trackmate",
                       utrack_settings=None, replace: bool = False,
                       measure_thresholds: bool = False, base: dict | None = None,
                       k: float | None = None, at_once: int = 1) -> dict:
    """One condition's row: ``ready``, the ``status`` words, ``movies``, ``minutes``,
    ``earlier`` (files of an earlier run in the way) and ``measure`` (the threshold is to
    be measured first). The refusals are the Track page's (`run_inputs`, `utrack_inputs`).

    TrackMate's linking radius, frame gap and spot radius are seeded as Detect & link seeds
    them (`seed_tracking`: the options' standard, the radius from the saved optics); a
    typed value is kept."""
    from . import prep_model as pm

    row = dict(_row_base(cond), movies=0, ready=False, status="", minutes=None,
               earlier=0, measure=False)
    if cond.state is None:
        row["status"] = f"{STATUS_SKIPPED}: {cond.error}"
        return row
    state = cond.state
    row["movies"] = len(state.movies)
    if tracker == "utrack":
        try:
            inputs = pm.utrack_inputs(state, utrack_settings)
        except pm.PrepError as exc:
            row["status"] = f"{STATUS_SKIPPED}: {_reason(exc)}"
            return row
        minutes = pm.minutes_at_once(inputs.minutes, len(state.movies), at_once)
        row.update(minutes=minutes, earlier=int(inputs.earlier))
        if inputs.earlier and not replace:
            row["status"] = (f"{STATUS_SKIPPED}: {inputs.earlier} movie(s) already hold a "
                             f"u-track run in tif/; tick Replace earlier runs")
            return row
        row["status"] = (f"ready, about {duration_words(minutes * 60)}"
                         + (f"; replaces the u-track run of {inputs.earlier} movie(s)"
                            if inputs.earlier else ""))
        row["ready"] = True
        return row

    pm.seed_tracking(state, base)
    try:
        inputs = pm.run_inputs(state)
    except pm.PrepError as exc:
        if not (measure_thresholds and "threshold" in pm.tracking_gaps(state)):
            row["status"] = f"{STATUS_SKIPPED}: {_reason(exc)}"
            return row
        # EVERY OTHER REFUSAL STILL APPLIES: asked of a copy holding a stand-in threshold,
        # which never reaches a run (the real one is measured first).
        trial = copy_with(state, tracking={**state.tracking, "threshold": 1.0},
                        typed=set(state.typed) | {"threshold"})
        try:
            inputs = pm.run_inputs(trial)
        except pm.PrepError as exc2:
            row["status"] = f"{STATUS_SKIPPED}: {_reason(exc2)}"
            return row
        row["measure"] = True
    minutes = pm.minutes_at_once(inputs.minutes, len(state.movies), at_once)
    row.update(minutes=minutes, earlier=int(state.earlier_run))
    if state.earlier_run and not replace:
        row["status"] = (f"{STATUS_SKIPPED}: trackmate/ already holds {state.earlier_run} "
                         f"track file(s) from an earlier run; tick Replace earlier runs")
        return row
    from .trackmate_run import UTRACK_MATCHED_K

    words = "ready"
    if row["measure"]:
        words += (f" once its threshold is measured inside the saved crops "
                  f"(k = {float(UTRACK_MATCHED_K if k is None else k):g})")
    words += f", about {duration_words(minutes * 60)}"
    if state.earlier_run:
        words += f"; replaces an earlier run ({state.earlier_run} files)"
    row.update(status=words, ready=True)
    return row


def plan_tracking(conds, **settings) -> list:
    """`tracking_readiness` for every condition, in order."""
    return [tracking_readiness(c, **settings) for c in conds]


def plan_summary(rows, *, tracker: str = "trackmate", sessions: int = 1) -> str:
    """The line under the table: how many conditions will run, how many movies, about how
    long (the Track page's per-condition estimates, summed), and how many are skipped."""
    if not rows:
        return "No parent folder yet: add the folders holding your conditions."
    ready = [r for r in rows if r.get("ready")]
    skipped = len(rows) - len(ready)
    n_movies = sum(int(r.get("movies", 0) or 0) for r in ready)
    text = f"{len(ready)} of {len(rows)} condition(s) will run, {n_movies} movie(s)"
    minutes = [r.get("minutes") for r in ready]
    if ready and all(m is not None for m in minutes):
        text += f", about {duration_words(sum(minutes) * 60)} of work in all"
        if sessions > 1:
            text += f" ({sessions} movies at once)"
    if skipped:
        text += f"; {skipped} skipped: each row says why"
    return text + "."


def layout_notes(rows) -> list:
    """The layout of every condition NOT organised, in full -- what filing would move where
    -- one line each, for under the table."""
    return [f"{r['label']}: {r['layout_words']}" for r in rows
            if r.get("layout") and r["layout"] != "organised" and r.get("layout_words")]


def work_folder_line(folder) -> str:
    """Where u-track works (the options' "u-track work folder")."""
    if folder:
        return (f"u-track works in {folder} and moves each movie's results into its "
                f"condition when it is done (Options: u-track work folder).")
    return ("u-track works beside the data. If the data is on a hard disk, a work folder on an "
            "SSD is much faster (Options: u-track work folder).")


def settings_line(tracker: str, work_folder=None) -> str:
    """The line under the settings row: how the tracker runs."""
    if tracker == "utrack":
        return ("Conditions run one after another; within each, the movies chosen at once are shared by as "
                "many MATLAB sessions side by side. " + work_folder_line(work_folder))
    return ("Conditions run one after another; the movies chosen at once run side by side, each in its own "
            "Fiji. Linking radius and frame gap follow Options unless typed on Detect & link.")


def track_batch_cells(row: dict, status: str | None = None) -> list:
    """A tracking row's cells, `ui.prep.batch_page.TRACK_COLUMNS`' order. ``status``
    replaces the plan's words while the condition runs and once it ran."""
    return [row["parent"], row["name"], row["layout"] or "--", str(row["movies"]),
            status or row["status"]]


#: The Batch page's "Measure a missing threshold" as a window opens: ticked (ruled 13 Oct 2026).
MEASURE_MISSING_THRESHOLD = True


def measure_threshold(state, *, k: float | None = None) -> dict:
    """Measure ``state``'s TrackMate threshold INSIDE ITS SAVED CROPS, one movie after the
    other, and file it as Detect & link's Measure does: `trackmate_run.movie_threshold_row`
    per movie, `prep_model.summarise_noise`, `prep_model.record_noise` (a typed threshold
    stays). Returns the summary; raises `PrepError` when no movie measured."""
    from . import prep_model as pm
    from . import trackmate_run as tmr

    k = tmr.UTRACK_MATCHED_K if k is None else float(k)
    radius = state.tracking.get("radius_px")
    if not radius:
        raise pm.PrepError(f"{state.name}: no spot radius to measure the threshold with")
    rows = [tmr.movie_threshold_row(path, name=stem, radius_px=float(radius), k=k,
                                    frame_range=tuple(state.saved[stem]))
            for path, stem in zip(state.movies, state.names)]
    summary = pm.summarise_noise(rows, radius_px=float(radius), k=k)
    pm.record_noise(state, summary)
    return summary


REPORT_COLUMNS = ("parent", "condition", "folder", "layout", "tracker", "status", "movies",
                  "movies_tracked", "movies_failed", "movies_stopped", "tracks",
                  "usable_tracks", "error")
REPORT_PREFIX = "smtracker_batch_"


def report_path(parent, when: datetime | None = None) -> Path:
    """``<first parent>/smtracker_batch_<YYYYmmdd-HHMMSS>.csv``."""
    when = when or datetime.now()
    return Path(parent) / f"{REPORT_PREFIX}{when:%Y%m%d-%H%M%S}.csv"


def write_report(path, rows) -> Path:
    """Write the report (one row per condition, `REPORT_COLUMNS`), replacing the last
    version whole: written beside it and moved over it, so a reader never sees half a
    file. Raises `OSError` when it cannot be written (a spreadsheet holding it open)."""
    path = Path(path)
    temporary = path.with_name(path.name + ".part")
    with open(temporary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: ("" if row.get(c) is None else row.get(c))
                             for c in REPORT_COLUMNS})
    os.replace(temporary, path)
    return path


def _report_row(plan_row: dict, tracker: str) -> dict:
    status = plan_row["status"]
    skipped = not plan_row.get("ready")
    return {"key": plan_row["key"], "label": plan_row["label"],
            "parent": plan_row["parent_path"], "condition": plan_row["name"],
            "folder": plan_row["folder"], "layout": plan_row["layout"], "tracker": tracker,
            "status": STATUS_SKIPPED if skipped else STATUS_WAITING,
            "movies": plan_row["movies"], "movies_tracked": None, "movies_failed": None,
            "movies_stopped": None, "tracks": None, "usable_tracks": None,
            "error": status.split(": ", 1)[-1] if skipped else ""}


def _outcome(row: dict, result, error: str) -> dict:
    """Fill a report row from a run's result (`track_condition`'s shape) or its error."""
    if error or result is None:
        return dict(row, status=STATUS_FAILED, error=error or "no result")
    out = dict(result.get("outcome") or {})
    movies = int(out.get("movies", row.get("movies") or 0) or 0)
    failed, stopped = int(out.get("failed", 0) or 0), int(out.get("stopped", 0) or 0)
    tracked = max(movies - failed - stopped, 0)
    row = dict(row, movies=movies, movies_tracked=tracked, movies_failed=failed,
               movies_stopped=stopped, tracks=out.get("tracks"),
               usable_tracks=out.get("usable_tracks"), error="")
    verdict = str(result.get("verdict") or "").strip().splitlines()
    if stopped:
        return dict(row, status=STATUS_STOPPED,
                    error=f"{stopped} movie(s) not run: the batch was stopped")
    if not tracked:
        return dict(row, status=STATUS_FAILED,
                    error=(verdict[0] if verdict else "no movie was tracked"))
    if failed:
        return dict(row, status=STATUS_PARTIAL,
                    error=next((v for v in verdict if v.startswith("Failed")),
                               f"{failed} movie(s) failed"))
    return dict(row, status=STATUS_TRACKED)


def _tracking_status(row: dict) -> str:
    status = row["status"]
    if status in (STATUS_TRACKED, STATUS_PARTIAL, STATUS_STOPPED):
        parts = [f"{row.get('movies_tracked') or 0} of {row.get('movies') or 0} movie(s)"]
        if row.get("usable_tracks") is not None:
            parts.append(f"{int(row['usable_tracks']):,} usable tracks")
        elif row.get("tracks") is not None:
            parts.append(f"{int(row['tracks']):,} tracks")
        text = f"{status}: " + ", ".join(parts)
        return text + (f": {row['error']}" if row.get("error") else "")
    if row.get("error"):
        return f"{status}: {row['error']}"
    return status


def plan_lines(rows, *, what: str) -> list:
    """The log's opening lines for a run: what will run, and every skip by name."""
    ready = [r for r in rows if r.get("ready")]
    lines = [f"Batch, {what}: {len(ready)} of {len(rows)} condition(s) to run"
             + (": " + ", ".join(r["label"] for r in ready) if ready else "")]
    lines += [f"    skipped {r['label']}: {r['status'].split(': ', 1)[-1]}"
              for r in rows if not r.get("ready")]
    return lines


def event_line(kind: str, label: str, payload, *, tracker: str = "trackmate") -> str:
    """One log line for one batch event (`run_tracking` / `run_outlines`), or ``""``."""
    payload = payload or {}
    if kind == "started":
        if "fields" in payload:
            return f"{label}: started, {payload['fields']} field(s) to segment"
        return f"{label}: started, {payload.get('movies', 0)} movie(s)"
    if kind == "measuring":
        return f"{label}: measuring the detection threshold inside the saved crops"
    if kind == "movie":
        from .prep_view import progress_line, utrack_progress_line

        line = utrack_progress_line if tracker == "utrack" else progress_line
        return f"{label}: " + line(payload["index"], payload["total"], payload["row"])
    if kind == "field":
        from .segmentation.outline_model import field_line

        return f"{label}: " + field_line(payload["index"], payload["total"], payload["row"])
    if kind == "finished":
        return f"{label}: {result_status(payload)}"
    if kind == "report_error":
        return (f"the report could not be written to {payload.get('path')}: "
                f"{payload.get('error')}; the batch goes on")
    return ""


def running_status(done: int, total: int, *, unit: str = "movie", measuring: bool = False) -> str:
    """A running condition's status cell."""
    if measuring:
        return "measuring the threshold inside the saved crops…"
    return f"running: {unit} {done} of {total}"


def run_tracking(conds, *, tracker: str = "trackmate", install=None, matlab=None,
                 utrack=None, utrack_settings=None, replace: bool = False,
                 measure_thresholds: bool = False, base: dict | None = None,
                 k: float | None = None, sessions: int = 1, work_folder=None, stop=None,
                 on_event=None, report=None) -> list:
    """Run every READY condition of ``conds`` and return one report row per condition.

    ``on_event(kind, key, payload)`` reports, from whichever thread is running:
    ``planned`` (key "", the plan rows), ``started`` (``{movies}``), ``measuring``,
    ``movie`` (``{index, total, row}``, the run's own plain row), ``finished`` (the report
    row) and ``report_error`` (``{error}``, once per failed write; the batch goes on).
    ``stop()`` is asked before each condition and handed to each run, which asks it before
    each movie. ``report`` is the CSV path, rewritten after every condition."""
    from . import prep_model as pm

    stop = stop or (lambda: False)
    emit = on_event or (lambda *_a: None)
    at_once = max(1, min(int(sessions or 1), MAX_SESSIONS))
    settings = dict(tracker=tracker, utrack_settings=utrack_settings, replace=replace,
                    measure_thresholds=measure_thresholds, base=base, k=k, at_once=at_once)
    plan = plan_tracking(conds, **settings)
    emit("planned", "", [dict(r) for r in plan])
    by_key = {c.key: c for c in conds}
    rows = {p["key"]: _report_row(p, tracker) for p in plan}
    lock = threading.Lock()
    failed_writes: list = []

    def write() -> None:
        if report is None:
            return
        with lock:
            try:
                write_report(report, list(rows.values()))
            except OSError as exc:
                text = f"{type(exc).__name__}: {exc}"
                if text not in failed_writes:
                    failed_writes.append(text)
                    emit("report_error", "", {"error": text, "path": str(report)})

    write()

    def one(p: dict) -> None:
        key, cond = p["key"], by_key[p["key"]]
        if stop():
            with lock:
                rows[key] = dict(rows[key], status=STATUS_NOT_RUN,
                                 error="the batch was stopped before this condition")
            emit("finished", key, dict(rows[key]))
            write()
            return
        emit("started", key, {"movies": p["movies"]})

        def progress(index, total, row):
            emit("movie", key, {"index": int(index), "total": int(total), "row": dict(row)})

        result, error = None, ""
        try:
            if tracker == "utrack":
                inputs = pm.utrack_inputs(cond.state, utrack_settings)
                result = pm.track_condition_utrack(
                    inputs, matlab=matlab, utrack=utrack, replace=replace,
                    on_progress=progress, stop=stop, work_folder=work_folder or None,
                    at_once=at_once)
            else:
                if p.get("measure"):
                    emit("measuring", key, {})
                    measure_threshold(cond.state, k=k)
                inputs = pm.run_inputs(cond.state)
                result = pm.track_condition(inputs, install=install, on_progress=progress,
                                            stop=stop, at_once=at_once)
        except Exception as exc:                        # noqa: BLE001 - named in its row
            error = f"{type(exc).__name__}: {exc}"
        with lock:
            rows[key] = _outcome(rows[key], result, error)
        emit("finished", key, dict(rows[key]))
        write()

    for p in plan:
        if p["ready"]:
            one(p)
    write()
    return [dict(r) for r in rows.values()]


def batch_progress(*, title: str, plan, done: dict, finished, running, started,
                   unit: str = "movie", count: str = "movies") -> dict:
    """The progress card for a batch: ``unit`` over the whole batch (every READY
    condition of ``plan``, across all folders, each counted by its ``count`` -- "movies",
    or "todo" for the fields a segmentation run will do), ``condition`` as the group, and
    the condition(s) running now with their own counts. ``done`` is ``{key: units done}``
    of the running ones."""
    totals = {r["key"]: int(r.get(count, 0) or 0) for r in plan if r.get("ready")}
    labels = {r["key"]: r["label"] for r in plan}
    finished = [k for k in finished if k in totals]
    running = [k for k in running if k in totals]
    done_all = (sum(totals[k] for k in finished)
                + sum(min(int(done.get(k, 0)), totals[k]) for k in running))
    return progress_payload(
        title=title, unit=unit, done=done_all, total=sum(totals.values()),
        group="condition", groups_done=len(finished), groups_total=len(totals),
        current=" + ".join(labels[k] for k in running),
        current_done=sum(min(int(done.get(k, 0)), totals[k]) for k in running),
        current_total=sum(totals[k] for k in running), started=started)


def finished_summary(rows, *, what: str = "Tracking", report=None,
                     report_error: str = "") -> dict:
    """``{title, lines, failed}`` for the end of a batch: what ran, then every condition
    that failed, stopped or was skipped BY NAME (never folded into a total), then the
    rest, then where the report is."""
    done_words = (STATUS_TRACKED, STATUS_PARTIAL, OUTLINE_SEGMENTED, OUTLINE_PARTIAL)
    stopped = [r for r in rows if r["status"] in (STATUS_STOPPED, STATUS_NOT_RUN)]
    failed = [r for r in rows if r["status"] in (STATUS_FAILED, STATUS_PARTIAL,
                                                   OUTLINE_PARTIAL)]
    skipped = [r for r in rows if r["status"] == STATUS_SKIPPED]
    ran = [r for r in rows if r["status"] in done_words + (STATUS_STOPPED,)]
    if stopped:
        title = f"{what} batch stopped"
    elif failed:
        title = f"{what} batch finished, with failures"
    else:
        title = f"{what} batch finished"
    head = f"{len(ran)} of {len(rows) - len(skipped)} condition(s) run"
    usable = [r.get("usable_tracks") for r in ran]
    if ran and all(u is not None for u in usable):
        head += f", {sum(int(u) for u in usable):,} usable tracks"
    cells = [r.get("cells") for r in ran]
    if ran and all(c is not None for c in cells):
        head += f", {sum(int(c) for c in cells):,} cells"
    if skipped:
        head += f"; {len(skipped)} skipped"
    lines = [head + "."]
    named = failed + stopped + skipped
    lines += [f"{r['label']}: {result_status(r)}" for r in named]
    lines += [f"{r['label']}: {result_status(r)}" for r in rows if r not in named]
    if report is not None:
        lines.append(f"Report: {report}" + (f" (could not be written: {report_error})"
                                             if report_error else ""))
    return {"title": title, "lines": lines, "failed": bool(failed or report_error)}


# ================================================================ segmentation
def scan_outlines(parents, *, model: str | None = None) -> list:
    """Every condition under every parent, as `outline_model.open_outlines` finds them (the
    outlines window's own discovery), each set to ``model`` (the options' standard) and
    carrying its layout. A parent with nothing to segment is ONE row naming why."""
    from .organise import folder_layout
    from .segmentation.outline_model import OutlineError, open_outlines

    out, seen = [], set()
    for parent in parents:
        parent = Path(parent)
        try:
            found = open_outlines(parent, model=model)
        except (OutlineError, OSError) as exc:
            below = _projects_below(parent, lambda child: open_outlines(child, model=model),
                                    (OutlineError, OSError))
            out.append(BatchCondition(parent=parent, name=parent.name, folder=None,
                                      error=(parent_of_projects(parent, below, what="segment") if below
                                             else str(exc))))
            continue
        for name, cond in found.items():
            key = folder_key(cond.folder)
            if key in seen:
                continue
            seen.add(key)
            out.append(BatchCondition(parent=parent, name=name, folder=Path(cond.folder),
                                      state=cond, layout=folder_layout(cond.folder)))
    return out


#: Why an ignored condition is not run (14 Oct 2026: "add a way in the list to skip certain
#: conditions, for example i dont wanna segment a condition again when i load the batch").
IGNORED_WORDS = "ignored; press Include to segment it"


def outline_readiness(cond: BatchCondition, *, replace: bool = False,
                      model_labels: dict | None = None, ignored: bool = False) -> dict:
    """One condition's row: its fields, how many have no outline yet
    (`outline_model.missing_fields`), the model, and ``ready``/``status`` from
    `outline_model.readiness` -- the Segment page's own words."""
    from .segmentation.outline_model import missing_fields, readiness, run_fields

    row = dict(_row_base(cond), fields=0, missing=0, todo=0, model="", ready=False,
               status="", minutes=None)
    if cond.state is None:
        row["status"] = f"{STATUS_SKIPPED}: {cond.error}"
        return row
    state = cond.state
    model = str(state.model)
    row.update(fields=len(state.fields), missing=len(missing_fields(state)),
               todo=len(run_fields(state, replace=replace)),
               model=(model_labels or {}).get(model, Path(model).name))
    ready, words = readiness(state, replace=replace)
    row.update(ready=bool(ready), status=words if ready else f"{STATUS_SKIPPED}: {words}")
    if ignored:
        row.update(ready=False, ignored=True, status=f"{STATUS_SKIPPED}: {IGNORED_WORDS}")
    return row


def plan_outlines(conds, *, replace: bool = False, model_labels: dict | None = None,
                  ignored=()) -> list:
    """Every condition's row; one whose key is in ``ignored`` is not run and says so."""
    ignored = set(ignored or ())
    return [outline_readiness(c, replace=replace, model_labels=model_labels,
                              ignored=c.key in ignored) for c in conds]


def outline_batch_cells(row: dict, status: str | None = None) -> list:
    """An outlines row's cells, `ui.outlines.batch_page.OUTLINE_COLUMNS`' order."""
    return [row["parent"], row["name"], row["layout"] or "--", str(row["fields"]),
            str(row["missing"]), row["model"] or "--", status or row["status"]]


def outline_summary(rows) -> str:
    """The line under the outlines table: conditions, fields to segment, skipped."""
    if not rows:
        return "No parent folder yet: add the folders holding your conditions."
    ready = [r for r in rows if r.get("ready")]
    skipped = len(rows) - len(ready)
    todo = sum(int(r.get("todo", 0)) for r in ready)
    kept = sum(int(r.get("fields", 0)) - int(r.get("todo", 0)) for r in ready)
    text = f"{len(ready)} of {len(rows)} condition(s) will run, {todo} field(s) to segment"
    if kept:
        text += f"; {kept} field(s) already outlined are kept"
    if skipped:
        text += f"; {skipped} skipped: each row says why"
    return text + "."


def run_outlines(conds, *, use_gpu: bool, replace: bool = False, stop=None,
                 on_event=None, model_labels: dict | None = None, ignored=(),
                 classical: dict | None = None) -> list:
    """Segment every READY condition, ONE AT A TIME (each run loads the model), through
    `outline_model.segment_condition` -- only the fields with no outline unless
    ``replace``. Events as `run_tracking`'s (``field`` for a field, ``{index, total,
    row}``). ``classical`` is the classical segmenter's settings (`classical_view`), used when
    that is the model. Returns one row per condition."""
    from .segmentation import outline_model as om

    stop = stop or (lambda: False)
    emit = on_event or (lambda *_a: None)
    plan = plan_outlines(conds, replace=replace, model_labels=model_labels, ignored=ignored)
    emit("planned", "", [dict(r) for r in plan])
    by_key = {c.key: c for c in conds}
    rows = {}
    for p in plan:
        rows[p["key"]] = {"key": p["key"], "label": p["label"], "parent": p["parent_path"],
                          "condition": p["name"], "folder": p["folder"],
                          "layout": p["layout"], "model": p["model"],
                          "status": STATUS_WAITING if p["ready"] else STATUS_SKIPPED,
                          "fields": p["todo"] if p["ready"] else p["fields"],
                          "cells": None, "failed_fields": None,
                          "error": "" if p["ready"] else p["status"].split(": ", 1)[-1]}
    for p in plan:
        if not p["ready"]:
            continue
        key = p["key"]
        if stop():
            rows[key] = dict(rows[key], status=STATUS_NOT_RUN,
                             error="the batch was stopped before this condition")
            emit("finished", key, dict(rows[key]))
            continue
        emit("started", key, {"fields": p["todo"]})
        result, error = None, ""
        try:
            by_key[key].state.classical = classical
            result = om.segment_condition(
                by_key[key].state, use_gpu=bool(use_gpu), replace=replace, stop=stop,
                on_field=lambda d, n, row, key=key: emit(
                    "field", key, {"index": int(d), "total": int(n), "row": dict(row)}))
        except Exception as exc:                        # noqa: BLE001 - named in its row
            error = f"{type(exc).__name__}: {exc}"
        row = dict(rows[key])
        if error or result is None:
            row.update(status=STATUS_FAILED, error=error or "no result")
        else:
            bad = list(result.get("failed") or [])
            row.update(cells=int(result.get("cells", 0)), failed_fields=len(bad),
                       error="")
            if result.get("stopped"):
                row.update(status=STATUS_STOPPED,
                           error=f"{result['stopped']} field(s) not segmented: stopped")
            elif bad:
                row.update(status=OUTLINE_PARTIAL,
                           error=f"{len(bad)} field(s) failed: " + ", ".join(bad[:4]))
            else:
                row.update(status=OUTLINE_SEGMENTED)
        rows[key] = row
        emit("finished", key, dict(row))
    return [dict(r) for r in rows.values()]


def _outline_status(row: dict) -> str:
    if row["status"] in (OUTLINE_SEGMENTED, OUTLINE_PARTIAL, STATUS_STOPPED) \
            and row.get("cells") is not None:
        text = f"{row['status']}: {row['cells']:,} cell(s)"
        return text + (f": {row['error']}" if row.get("error") else "")
    return f"{row['status']}: {row['error']}" if row.get("error") else row["status"]


def result_status(row: dict) -> str:
    """A finished condition's status cell (and its line in the pop-up), tracking or
    segmentation: what it produced, or why it did not run."""
    if "cells" in row:
        return _outline_status(row)
    return _tracking_status(row)


# ================================================================ keeping Windows awake
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def _set_execution_state(flags: int) -> bool:
    """The one Windows call. A no-op (False) off Windows and under pytest, where a test
    must not change how the machine running it sleeps."""
    if sys.platform != "win32" or os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    import ctypes

    return bool(ctypes.windll.kernel32.SetThreadExecutionState(int(flags)))


def keep_awake(on: bool) -> bool:
    """Keep Windows from sleeping while a batch runs (``on``), or let it sleep again. Ask
    from the thread that will clear it -- the state belongs to the calling thread. The
    display may still turn off; only the system is kept running."""
    return _set_execution_state(ES_CONTINUOUS | ES_SYSTEM_REQUIRED if on else ES_CONTINUOUS)


def awake_note() -> str:
    """What a running batch's note may say about sleep: true on THIS machine, never more.

    Only Windows is kept awake (`keep_awake`); on macOS and Linux nothing is, so the note
    said "Windows is kept awake" to a Mac user whose machine was free to sleep (the UI designer,
    26 Nov 2026). Follows the platform, not `keep_awake`'s return, which is False under
    pytest on Windows too. Empty elsewhere: say nothing rather than promise it.
    """
    return "Windows is kept awake until the batch ends." if sys.platform == "win32" else ""
