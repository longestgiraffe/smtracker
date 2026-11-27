"""The Kinetics section's "Populations over time" readout: its numbers and sentences, with no Qt anywhere in it.

`ui/workspace/kinetics_chart.py` draws it; this is what it draws from (the Thin Qt rule, `tests/test_ui_thin_qt.py`).
The model is `smtracker.induction` (one SQD mixture whose D's and shares move from before to after the induction,
the change and its shape chosen on whole units held out, intervals from whole units resampled); this module only runs
it for one condition and turns the result into plain lists and strings.

WHY A SECOND READOUT BESIDE THE SLOW FRACTION. The section's first readout (`kinetics_view`) bins one HMM's slow-state
posterior: a D that changes reads there as a share that changes. This one fits the D's AND the shares over time and
lets held-out movies say which moved (asked 21 Oct 2026: "plot the Ds over time to measure how fast it drops"). On
SynK's recovery grid (`docs/claude/states_and_kinetics.md`): the change right in 71 of 72 groups, the half-time
within about +0.2 to +0.9 s; the SHAPE mostly not identifiable at 12 movies, and onsets spread between CELLS not
measurable -- so the card gives the half-time as the range over the shapes it cannot tell apart, and the onset test is
offered between MOVIES only, against its shuffles.

WHAT THE CARD REFUSES TO HIDE: the shapes it cannot tell apart, a response not complete where the data end (its
"after" values are then the shape's extrapolation, and the values the data reach are given), few units, and the
onset test's p beside the shuffles' own spread (an SD is a sign, not a size).

`compute_induction` is picklable and runs on the process pool, one condition at a time; a failure comes back as
``{"error": "<Type>: <message>"}``, never as an absence.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .kinetics_designs import DESIGN_EXPLANATIONS, DESIGN_WORDS, DESIGNS, design_lines, design_of

__all__ = [
    "HALF_TIME_KINDS",
    "half_time_record",
    "half_time_panels",
    "reference_half_time",
    "half_time_words",
    "half_time_mark",
    "DESIGNS",
    "DESIGN_WORDS",
    "DESIGN_EXPLANATIONS",
    "design_of",
    "condition_movies",
    "camera_starts",
    "sample_starts_since_addition",
    "CLOCKS",
    "CLOCK_WORDS",
    "CLOCK_NOTES",
    "condition_days",
    "parse_addition",
    "addition_problem",
    "movie_starts_since_addition",
    "movie_extents",
    "READOUTS",
    "READOUT_WORDS",
    "POPULATION_READOUTS",
    "MOBILITY_CHANGES",
    "MOBILITY_NOTE",
    "linking_limit_lines",
    "movie_mobility",
    "ALTERNATIVES_LABEL",
    "d_panels",
    "shape_words",
    "TIME_UNITS",
    "TIME_UNIT_WORDS",
    "convert_range",
    "N_POPS",
    "SPREAD_SHUFFLES",
    "INDUCTION_MODEL_HEADINGS",
    "compute_induction",
    "permutation_chunk",
    "permutation_lines",
    "attach_permutation",
    "run_permutation",
    "REFERENCE_COLUMNS",
    "REFERENCE_LABEL",
    "reference_curves",
    "find_reference",
    "attach_reference",
    "has_reference",
    "induction_model_cells",
    "induction_caption",
    "population_names",
    "population_hover",
]

#: The unit the time axis is drawn in (display only; every computed time stays seconds): seconds per unit, and the
#: words. Seconds is the default (7 Nov 2026: a minutes option, in the Axes popover and carried into the export).
TIME_UNITS = {"s": 1.0, "min": 60.0}
TIME_UNIT_WORDS = {"s": "seconds", "min": "minutes"}


def convert_range(pair, old: str, new: str):
    """A typed time range ``(from, to)`` (either end may be None) moved from unit ``old`` to ``new``."""
    if pair is None or old == new:
        return pair
    k = TIME_UNITS[old] / TIME_UNITS[new]
    return tuple(None if v is None else float(v) * k for v in pair)


#: The section's readouts: the D's and shares over time (this module), or the HMM slow fraction (`kinetics_view`).
READOUTS = ("mobility", "populations", "slow_fraction")
READOUT_WORDS = {"mobility": "Mean mobility over time", "populations": "Populations over time",
                 "slow_fraction": "Slow fraction (HMM)"}
#: The readouts the induction model answers (this module): the mean mobility is the same model with ONE population
#: and only a change in D (8 Nov 2026), so every card, export and test of the populations readout serves it.
POPULATION_READOUTS = ("mobility", "populations")
#: The mean mobility's model: one population, "none" or "D" (a share cannot change with one population).
MOBILITY_CHANGES = ("none", "D")
#: Populations the model may hold; the manuscript reports two (CLAUDE.md, the open question).
N_POPS = (2, 3)
#: Shuffles for the onset test between movies (19 reach p = 0.05).
SPREAD_SHUFFLES = 19
#: Where the model's curves and bands are drawn.
GRID_POINTS = 160
#: The model table's headings (`induction_model_cells`): every model scored on whole units held out.
INDUCTION_MODEL_HEADINGS = ("model", "against the best", "behind by", "units won", "")

#: What the mean mobility card says about itself, once.
MOBILITY_NOTE = ("One D, the mean of every jump: a change in D and a change in the bound share both move it; "
                 "Populations over time tells which.")

_CHANGE_WORDS = {"none": "no change", "fractions": "the shares changed", "D": "the D's changed",
                 "both": "the D's and the shares changed"}
#: A change the held-out units could not rule out, in the flag that names it.
_OPEN_CHANGE_WORDS = {"fractions": "a change in the shares alone", "D": "a change in the D's alone",
                      "both": "a change in both the D's and the shares"}


def population_names(n_pop: int) -> list:
    """Population k's name on the card, slowest first (numbered by their D before the induction)."""
    if n_pop == 1:
        return ["all jumps"]
    if n_pop == 2:
        return ["slow", "fast"]
    if n_pop == 3:
        return ["slow", "middle", "fast"]
    return [f"population {k + 1}" for k in range(n_pop)]


# --------------------------------------------------------------------------- #
# The second clock: movies recorded at known times after the addition
# --------------------------------------------------------------------------- #

#: The two clocks (`induction.induction_steps`): the trigger on one frame of every movie, or each movie's own time
#: since the addition (asked 23 Oct 2026: "Movies after addition" -- added, then movies recorded minutes to hours on).
CLOCKS = ("trigger", "addition", "sample")
CLOCK_WORDS = {"trigger": "Trigger frame in each movie", "addition": "Time after the addition",
               "sample": "Time after each sample's own addition"}
#: What each clock means, under the conditions in the Run dialog.
CLOCK_NOTES = {
    "trigger": ("Counted in the tracked movie, where frame 1 is its first frame. Every movie in a condition is taken "
                "to have the trigger at this frame."),
    "addition": ("The clock time you added it, once per replicate (10:30, or YYYY-MM-DD 10:30 for another "
                 "date). Each movie's time since then is read from its camera file plus the frames before its "
                 "tracked range. Movies recorded before the addition are the baseline."),
    "sample": ("Each movie is its own sample unless you give movies of one sample the same name. Type the clock "
               "time each sample was added (10:30, or YYYY-MM-DD 10:30), or paste a column from Excel."),
}
#: What a typed addition time may look like: a clock time on the day's first recording date, or a full date and time.
_TIME_FORMATS = ("%H:%M", "%H:%M:%S")
_DATETIME_FORMATS = ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%d.%m.%Y %H:%M", "%d.%m.%Y %H:%M:%S")


def condition_days(project, condition: str) -> list:
    """``[(day, n movies, first movie)]`` of a condition, in day order: the dialog's rows for the addition times
    (each replicate day is its own experiment, Rule 0, so each has its own addition)."""
    table = project.condition(condition)
    if "day" not in table.columns:
        return []
    out = []
    for day, part in table.groupby("day", sort=True):
        movies = sorted(map(str, part["movie"].unique()))
        source = getattr(project, "movie_source", {}) or {}
        first = str(source.get(movies[0], (None, movies[0]))[1]) if movies else ""
        out.append((int(day), len(movies), first))
    return out


def parse_addition(text: str, on_date=None):
    """The datetime a typed addition time means, or None: ``HH:MM[:SS]`` on ``on_date`` (the day's first recording
    date), or a full ``YYYY-MM-DD HH:MM[:SS]`` / ``DD.MM.YYYY HH:MM[:SS]``. Never a guess: anything else is None."""
    import datetime as _dt

    text = (text or "").strip()
    if not text:
        return None
    for fmt in _DATETIME_FORMATS:
        try:
            return _dt.datetime.strptime(text, fmt)
        except ValueError:
            pass
    for fmt in _TIME_FORMATS:
        try:
            clock = _dt.datetime.strptime(text, fmt).time()
        except ValueError:
            continue
        return None if on_date is None else _dt.datetime.combine(on_date, clock)
    return None


def addition_problem(rows: dict) -> str:
    """Why Run must wait under the addition clock, or "": ``rows`` ``{condition: (included, {day: text})}``; every
    day of an included condition needs a time that reads (`parse_addition`, a date stood in for the check)."""
    import datetime as _dt

    included = [key for key, (on, _days) in rows.items() if on]
    if not included:
        return "Include at least one condition."
    for key in included:
        if not rows[key][1]:
            return f"No replicates are known for {key}: its movies carry no replicate."
        for day, text in sorted(rows[key][1].items()):
            if not str(text).strip():
                return f"Type the time of the addition for {key}, replicate {day}."
            if parse_addition(text, _dt.date(2000, 1, 1)) is None:
                return f"{key}, replicate {day}: write the time as 10:30 or YYYY-MM-DD 10:30."
    return ""


def _camera_file(root, movie: str):
    """A movie's raw acquisition: ``raw/<movie>.nd2|.stk``, else beside the condition, else u-track's own mapping
    (`project.camera_files`, which knows day folders). None where there is none."""
    from pathlib import Path

    from .project import camera_files

    root = Path(root)
    for place in (root / "raw", root):
        for suffix in (".nd2", ".stk", ".ND2", ".STK"):
            candidate = place / f"{movie}{suffix}"
            if candidate.is_file():
                return candidate
    return camera_files(root).get(movie)


def condition_movies(project, condition: str) -> list:
    """``[(movie as the analysis names it, the name its file wears, replicate)]`` in movie order: the rows of the
    per-sample table of the design "Added to each sample"."""
    table = project.condition(condition)
    source = getattr(project, "movie_source", {}) or {}
    per_movie = table.groupby("movie", sort=True)["day"].first()
    return [(str(m), str(source.get(m, (None, m))[1]), int(d)) for m, d in per_movie.items()]


def _simulated_starts(root) -> dict:
    """``{movie: recording start}`` from a simulated set's `generator.SIMULATED_STARTS_FILE`, or {} without one. A
    file that does not read raises: a clock is never guessed."""
    import datetime as _dt
    import json
    from pathlib import Path

    from .generator import SIMULATED_STARTS_FILE
    from .induction import InductionError

    path = Path(root) / SIMULATED_STARTS_FILE
    if not path.is_file():
        return {}
    try:
        return {str(k): _dt.datetime.fromisoformat(str(v))
                for k, v in json.loads(path.read_text(encoding="utf-8"))["movies"].items()}
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise InductionError(f"{path.name}: the simulated recording starts cannot be read ({exc})") from exc


def camera_starts(project, condition: str) -> dict:
    """``{movie as the analysis names it: (recording start, first tracked frame)}``: each movie's recording start from
    its camera file (`rawmovie.acquisition_time`) and the 1-based frame its tracked range starts on
    (`project.movie_first_frames`). RAISES naming every movie without a camera file, a readable start or a recorded
    range: a clock is never guessed."""
    from .induction import InductionError
    from .project import movie_first_frames
    from .rawmovie import acquisition_time

    table = project.condition(condition)
    source = getattr(project, "movie_source", {}) or {}
    root = project.roots[condition]
    firsts = movie_first_frames(project, condition)
    per_movie = table.groupby("movie", sort=True)["day"].first()
    started, problems = {}, []
    simulated = _simulated_starts(root)
    for movie in per_movie.index:
        original = str(source.get(movie, (None, movie))[1])
        raw = _camera_file(root, original)
        if raw is None and original in simulated:
            # A SIMULATED SET (`generator`, 25 Nov 2026): its tracks have no camera file, and the generator writes
            # each movie's simulated recording start beside them; a real experiment never has this file
            when = simulated[original]
        elif raw is None:
            problems.append(f"{original}: no camera file (.nd2 / .stk) beside it")
            continue
        else:
            when = acquisition_time(raw)
        if when is None:
            problems.append(f"{original}: {raw.name} records no start time")
            continue
        if original not in firsts:
            problems.append(f"{original}: no record of its tracked range")
            continue
        started[movie] = (when, firsts[original][0])
    if problems:
        raise InductionError("the clock since the addition cannot be set: " + "; ".join(problems[:6])
                             + (f"; and {len(problems) - 6} more" if len(problems) > 6 else ""))
    return started


def sample_starts_since_addition(project, condition: str, additions_by_sample: dict, groups: dict | None = None):
    """The design "Added to each sample": ``(starts, samples)``. ``starts`` ``{movie: seconds from ITS SAMPLE's
    addition to its first tracked frame}``; ``additions_by_sample`` ``{sample: typed time}``; ``groups`` ``{movie:
    sample}``, a movie it does not name being its own sample (the default). ``samples`` is the shuffle's record,
    ``{"movie_sample": {movie: sample}, "addition_s": {sample: the addition on one absolute clock, s}}``
    (`induction.shuffle_clock`): the additions of different samples are only comparable on a clock they share.

    A clock time is on the date its sample's first movie was recorded. RAISES naming every sample without a time and
    every movie without a camera file, a start or a recorded range."""
    tau = float(project.params.tau_s)
    return _sample_starts(camera_starts(project, condition), tau, additions_by_sample, groups)


def _sample_starts(started: dict, tau: float, additions_by_sample: dict, groups: dict | None = None):
    """`sample_starts_since_addition` on the movies' camera starts already read (``started`` as `camera_starts`
    gives it): the one definition, shared with `movie_extents`' Auto bin width."""
    import datetime as _dt

    from .induction import InductionError

    groups = {str(k): str(v) for k, v in (groups or {}).items()}
    movie_sample = {str(m): groups.get(str(m), str(m)) for m in started}
    members: dict = {}
    for m, name in movie_sample.items():
        members.setdefault(name, []).append(m)
    out, addition_s = {}, {}
    epoch = _dt.datetime(1970, 1, 1)
    for name, movies in members.items():
        text = additions_by_sample.get(name)
        if text is None or not str(text).strip():
            raise InductionError(f"sample {name}: no time of the addition (type it as 10:30)")
        first_date = min(started[m][0] for m in movies).date()
        at = parse_addition(str(text), first_date)
        if at is None:
            raise InductionError(f"sample {name}: the addition time {text!r} does not read (use 10:30 or "
                                 f"YYYY-MM-DD 10:30)")
        addition_s[name] = (at - epoch) / _dt.timedelta(seconds=1)
        for movie in movies:
            when, first = started[movie]
            out[movie] = (when - at) / _dt.timedelta(seconds=1) + (int(first) - 1) * tau
    return out, {"movie_sample": movie_sample, "addition_s": addition_s}


def movie_starts_since_addition(project, condition: str, additions: dict) -> dict:
    """``{movie as the analysis names it: seconds from the addition to its first TRACKED frame}``.

    ``additions`` ``{day: typed time}`` (`parse_addition`; a clock time is on the date the day's first movie was
    recorded). Each movie: its recording start from its camera file (`camera_starts`) plus the frames before its
    tracked range at the analysis's frame interval. A movie recorded before the addition comes out negative: it is
    the baseline, kept. RAISES naming every movie without a camera file, a readable start or a recorded range, and
    every day without a time: a clock is never guessed."""
    table = project.condition(condition)
    tau = float(project.params.tau_s)
    per_movie = table.groupby("movie", sort=True)["day"].first()
    return _addition_starts(camera_starts(project, condition), {str(m): int(d) for m, d in per_movie.items()}, tau,
                            additions)


def _addition_starts(started: dict, day_of: dict, tau: float, additions: dict) -> dict:
    """`movie_starts_since_addition` on the movies' camera starts already read (``started`` as `camera_starts` gives
    it, ``day_of`` ``{movie: replicate}``): the one definition, shared with `movie_extents`' Auto bin width."""
    import datetime as _dt

    from .induction import InductionError

    per_movie = pd.Series({m: day_of[m] for m in started})
    out = {}
    for day, movies in per_movie.groupby(per_movie).groups.items():
        movies = list(movies)
        first_date = min(started[m][0] for m in movies).date()
        at = parse_addition(str(additions.get(int(day), additions.get(str(day), ""))), first_date)
        if at is None:
            raise InductionError(f"replicate {int(day)}: no time of the addition (type it as 10:30)")
        for movie in movies:
            when, first = started[movie]
            out[str(movie)] = (when - at) / _dt.timedelta(seconds=1) + (int(first) - 1) * tau
    return out


def movie_extents(project, condition: str, *, camera: bool = True) -> dict:
    """What the Run dialog's Auto bin width needs of a condition, read once: ``{"tau_s", "movies": {movie:
    (replicate, first frame, last frame)}, "started": {movie: (recording start, first tracked frame)} or None,
    "problem": "" or why the camera starts could not be read}`` (`camera_starts`). A failure to read the starts is
    kept as ``problem`` (the trigger clock needs none: ``camera=False`` does not read them), never raised and never
    replaced by a guess."""
    from .induction import InductionError

    table = project.condition(condition)
    per = table.groupby("movie", sort=True).agg(day=("day", "first"), lo=("frame", "min"), hi=("frame", "max"))
    movies = {str(m): (int(r.day), int(r.lo), int(r.hi)) for m, r in per.iterrows()}
    try:
        started, problem = (camera_starts(project, condition) if camera else None), ""
    except InductionError as exc:
        started, problem = None, str(exc)
    return {"tau_s": float(project.params.tau_s), "movies": movies, "started": started, "problem": problem}


def movie_mobility(project, condition: str) -> dict:
    """What the Run dialog's raw-data preview needs of a condition, read once and WITHOUT a fit: ``{"tau_s", "movies":
    {movie: {"replicate", "n", "mean_sq_um2", "mean_frame"}}}``, each movie's number of lag-1 squared jumps, their mean
    (um2) and the mean first frame of the pairs. The mean squared jump over 4 tau is a model-free apparent D."""
    from .observables import squared_displacements

    sd = squared_displacements(project.condition(condition), lag_frames=1)
    tau = float(project.params.tau_s)
    per = sd.groupby("movie", sort=True).agg(n=("sq_disp_um2", "size"), mean_sq=("sq_disp_um2", "mean"),
                                             mean_frame=("frame", "mean"), day=("day", "first"))
    return {"tau_s": tau, "lag_time_s": float(sd.attrs.get("lag_time_s", tau)),
            "movies": {str(m): {"replicate": int(r.day), "n": int(r.n), "mean_sq_um2": float(r.mean_sq),
                                "mean_frame": float(r.mean_frame)} for m, r in per.iterrows()}}


def _model_label(change: str, shape: str) -> str:
    if change == "none":
        return "no change"
    return f"{_CHANGE_WORDS[change].replace('the ', '').replace(' changed', '')}, {shape}"


def _clock_steps(project, condition: str, *, trigger_frame, movie_start_s, precision=None, samples=None):
    """The condition's squared displacements on the induction clock (`induction.induction_steps`), one definition for
    the model's job and the permutation test's. ``precision`` ``(c_nm, p, source)``: each step's own noise
    (`induction.with_step_noise`, Localisation error Per spot), so the shuffles are refitted as the model was.
    ``samples``: the design "Added to each sample"'s record (`sample_starts_since_addition`), kept in the table's
    ``attrs`` where `induction.shuffle_clock` reads it, so the shuffles deal the samples' addition times."""
    from . import induction as ind
    from .observables import squared_displacements

    canonical = project.condition(condition)
    tau = float(project.params.tau_s)
    sd = squared_displacements(canonical, lag_frames=1)
    steps = ind.induction_steps(sd, trigger_frame=trigger_frame, movie_start_s=movie_start_s, tau_s=tau)
    if samples:
        steps.attrs["samples"] = samples
    if precision is not None:
        from .precision import Calibration

        steps = ind.with_step_noise(steps, Calibration(c_nm=float(precision[0]), p=float(precision[1]),
                                                       source=str(precision[2])))
    return steps


def clock_steps_for(project, condition: str, *, trigger_frame=None, movie_start_s=None, additions=None, samples=None,
                    precision=None):
    """``(steps, movie_start_s, sample_record)``: ONE condition's steps on its own clock, however the Run set it
    (`compute_induction` and the mock comparison both come here, so the clock is resolved once). ``trigger_frame``;
    ``additions`` ``{replicate: typed time}`` (`movie_starts_since_addition`); ``samples`` ``{sample: {"movies",
    "time"}}`` (`sample_starts_since_addition`); or ``movie_start_s`` given. ``precision``: see `_clock_steps`."""
    sample_record = None
    if samples is not None:
        groups = {m: name for name, e in samples.items() for m in e["movies"]}
        movie_start_s, sample_record = sample_starts_since_addition(
            project, condition, {name: e["time"] for name, e in samples.items()}, groups)
    elif additions is not None:
        movie_start_s = movie_starts_since_addition(project, condition, additions)
    steps = _clock_steps(project, condition, trigger_frame=trigger_frame, movie_start_s=movie_start_s,
                         precision=precision, samples=sample_record)
    return steps, movie_start_s, sample_record


def linking_limit_lines(project, condition: str, names, d_before, d_end) -> list:
    """The card's caveat when a population's D is near what the condition's linking radius can measure (8 Nov 2026).

    The fitted D of a fast population is apparent: a molecule that moves farther than the search radius in one frame
    is not linked, so the D reads low. `trackmate_run.max_measurable_D` is the fastest D that still reads within 10 %
    at the condition's own radius (`AcquisitionParams.search_radius_px_for`; a u-track radius is a cap on an adaptive
    one, so a fast population reads lower still). When a population's D before or after the induction is above HALF
    that limit, ONE line says so. A condition with no recorded radius, or a project with no pixel size or frame
    interval, says nothing: no radius is ever borrowed or assumed. Returns ``[]`` or ``[the line]``."""
    from .canonical import CanonicalError

    params = getattr(project, "params", None)
    try:
        radius = params.search_radius_px_for(condition)
    except (AttributeError, CanonicalError):
        return []
    pixel, tau = getattr(params, "pixel_size_nm", None), getattr(params, "tau_s", None)
    if not radius or not pixel or not tau:
        return []
    from .trackmate_run import max_measurable_D

    limit = max_measurable_D(radius_px=float(radius), tau_s=float(tau), pixel_size_nm=float(pixel))
    near = [k for k in range(len(names)) if max(float(d_before[k]), float(d_end[k])) > 0.5 * limit]
    if not near:
        return []
    which = "mean" if len(names) == 1 else names[near[-1]]
    return [f"The {which} D is near what a {float(radius):g} px link can measure at {float(tau) * 1000:g} ms "
            f"({limit:.2f} µm²/s). It reads low, so a change from it reads smaller than it is, and the part nearest "
            f"the limit shows least: a drop reads late and a rise early."]


def _failure(exc: BaseException) -> dict:
    """A job's failure as the card reads it: the message, and the whole traceback kept (the card's hover; it was
    cut to the message alone until 7 Nov 2026, which hid WHERE a fit refused)."""
    import traceback

    return {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def compute_induction(project, condition: str, *, trigger_frame: int | None = None,
                      movie_start_s: dict | None = None, additions: dict | None = None, unit: str = "movie",
                      bin_seconds: float | None = None, n_pop: int = 2, spread: bool = False,
                      n_boot: int | None = None, seed: int = 0, changes=None, shapes=None,
                      permutation: bool = False, precision=None, design: str | None = None,
                      samples: dict | None = None) -> dict:
    """One condition's populations over time: the chosen model, its bands, the free per-bin fits, the sentences.

    The clock, one of: ``trigger_frame`` (the induction on that frame of every tracked movie), ``additions``
    (``{day: typed time}``: each movie's own time since its day's addition, `movie_starts_since_addition`), or
    ``movie_start_s`` (each movie's first frame in seconds since the induction, given). ``bin_seconds`` draws the
    free per-bin fits beside the model (None: none drawn); they are what the data look like, never the estimate.
    ``spread`` adds the onset test between movies against `SPREAD_SHUFFLES` shuffles (about 20 fits).
    ``changes`` / ``shapes`` narrow the models compared (tests). ``unit`` is what is held out and resampled
    (`induction.UNIT_COLUMNS`: movie, cell or replicate).

    ``permutation`` only PREPARES the time-label permutation test (``result["permutation"]``: the model to refit and
    where it starts, or why no test can be made); its shuffles are the jobs of `run_permutation`, which a window
    spreads over the pool. The answer is on the card once `attach_permutation` has put it there.

    ``precision``: the Run's per-spot calibration ``(c_nm, p, source)`` (Localisation error Per spot, 6 Nov 2026):
    every fit with each step's own noise (`induction.with_step_noise`), the shuffles' too; None for a condition whose
    tracks carry none.

    ``design`` (`kinetics_designs.DESIGNS`; None reads as `design_of` does for an old call): "Added to each sample"
    takes ``samples`` ``{sample: {"movies": [...], "time": text}}`` instead of ``additions`` and sets each movie's clock
    from ITS sample's addition (`sample_starts_since_addition`). Every design's card carries its design lines
    (`kinetics_designs.design_lines`) in ``result["text"]["design"]``.
    """
    from . import induction as ind
    from .kinetics_mock_view import response_record

    sample_record = None
    try:
        design = design_of({"design": design, "clock": "trigger" if trigger_frame is not None else "addition"})
        steps, movie_start_s, sample_record = clock_steps_for(
            project, condition, trigger_frame=trigger_frame, movie_start_s=movie_start_s, additions=additions,
            samples=samples, precision=precision)
        compare = ind.compare_models(steps, n_pop=int(n_pop), unit=unit, seed=seed,
                                     **({"changes": tuple(changes)} if changes else {}),
                                     **({"shapes": tuple(shapes)} if shapes else {}))
        fit = compare["fit"]
        clock = steps[ind.CLOCK].to_numpy(float)
        clock = clock[np.isfinite(clock)]
        t0, t1 = float(clock.min()), float(clock.max())
        grid = np.linspace(t0, t1, GRID_POINTS)
        boot = ind.bootstrap(steps, fit, unit=unit, n_boot=n_boot, seed=seed, grid=grid)
        bins = (ind.binned_fits(steps, n_pop=int(n_pop), bin_s=float(bin_seconds), seed=seed)
                if bin_seconds else None)
        # each replicate's own per-bin fit, drawn as open circles (7 Nov 2026; why not an interval:
        # `induction.replicate_binned_fits`)
        rep_bins = (ind.replicate_binned_fits(steps, n_pop=int(n_pop), bin_s=float(bin_seconds), seed=seed)
                    if bin_seconds else None)
        onset = (ind.onset_spread_null(steps, fit, level="movie", n_shuffles=SPREAD_SHUFFLES, seed=seed)
                 if spread and fit.change != "none" else None)
        plan = _permutation_plan(steps, compare, int(n_pop), design) if permutation else None
        limit_lines = linking_limit_lines(project, condition, population_names(fit.n_pop), fit.D_before, fit.D_end)
        design_text = design_lines(steps, design=design, tau_s=float(steps.attrs["tau_s"]),
                                   trigger_frame=trigger_frame, change=fit.change, d_before=list(fit.D_before),
                                   d_end=list(fit.D_end))
    except Exception as exc:                            # noqa: BLE001 - the card names it
        return _failure(exc)
    settings = {"trigger_frame": None if trigger_frame is None else int(trigger_frame),
                "clock": steps.attrs.get("clock", ""), "additions": dict(additions) if additions else None,
                "movie_start_s": ({str(k): float(v) for k, v in movie_start_s.items()} if movie_start_s else None),
                "unit": unit, "bin_seconds": bin_seconds, "n_pop": int(n_pop), "spread": bool(spread),
                "permutation": bool(permutation), "design": design,
                "changes": list(changes) if changes else None,
                "samples": ({k: {"movies": list(e["movies"]), "time": str(e["time"])} for k, e in samples.items()}
                            if samples else None), "sample_record": sample_record,
                **({"precision": [float(precision[0]), float(precision[1]), str(precision[2])]}
                   if precision is not None else {})}
    return {"readout": "populations", "mobility": int(n_pop) == 1, "settings": settings,
            "n_steps": int(fit.n_steps), "n_units": int(compare["n_units"]),
            "chart": _chart(fit, grid, boot, bins, t0, t1, rep_bins, _alternatives(compare, fit, grid)),
            "text": _with_flags(dict(_text(compare, fit, boot, onset, spread, unit), design=design_text), limit_lines),
            "held_out_change": fit.change, "permutation": plan,
            # the half-time as numbers (8 Nov 2026): the card's line, the figure's mark and the source data read it
            "half_time": half_time_record(compare, fit, boot),
            # baseline movies exist (steps before the addition): the card says so when no mock is chosen
            "has_baseline": bool(np.any(clock < 0.0)),
            # the fitted response, for the mock comparison's Auto after window (`kinetics_mock_view`)
            "response": response_record(fit, grid)}


# --------------------------------------------------------------------------- #
# The time-label permutation test, beside the held-out verdict (7 Nov 2026)
# --------------------------------------------------------------------------- #

#: What the model that gained says, in the card's words.
_PERMUTATION_WORDS = {"fractions": "shares changed", "D": "D changed", "both": "D and shares changed"}


def _permutation_plan(steps, compare: dict, n_pop: int, design: str = "culture") -> dict:
    """What the shuffles need: EVERY changing model the Run compared (`compare_models`' table, "none" left out), each
    clock scored by the best of them (the exact test, `induction` section comment), the constant mixture where each
    refit starts and its log-likelihood; or, where fewer clocks exist than a test needs, the reason it is not run
    (shown on the card, not hidden)."""
    from . import induction as ind

    table = compare["table"]
    models = [[str(c), str(sh)] for c, sh in zip(table["change"], table["shape"]) if c != "none"]
    try:
        arrangements = ind.permutation_arrangements(steps)
    except ind.InductionError as exc:                   # review, 8 Nov 2026: the test's problem, never the card's
        return {"problem": f"Not run: {exc}"}
    if arrangements < ind.PERM_MIN_ARRANGEMENTS:
        what = "samples" if design == "sample" else "movies"
        return {"problem": (f"Not run: only {arrangements:.0f} different clocks can be dealt among the {what} of "
                            f"these replicates, fewer than {ind.PERM_MIN_ARRANGEMENTS}, so no permutation test could "
                            f"reach p = 0.05.")}
    constant = compare["constant"]
    return {"problem": None, "models": models, "n_pop": int(n_pop),
            "tau_lag_s": float(constant.tau_lag_s), "D": [float(v) for v in constant.D_before],
            "fractions": [float(v) for v in constant.fractions_before],
            "constant_ll": float(constant.log_likelihood), "arrangements": float(arrangements), "summary": None}


def permutation_chunk(project, condition: str, *, clock: dict, plan: dict, seeds: list) -> list:
    """The gains of ``seeds`` for ``plan``'s model on ``condition``'s steps (`induction.permutation_gains`): ONE POOL
    JOB of the test, through the section cache (`sectioncache.cached_job`; the key holds the seeds, so a chunk made
    before is read). ``clock`` is the Run's ``trigger_frame`` or ``movie_start_s`` (the card's ``settings``)."""
    from . import induction as ind

    steps = _clock_steps(project, condition, trigger_frame=clock.get("trigger_frame"),
                         movie_start_s=clock.get("movie_start_s"), precision=clock.get("precision"),
                         samples=clock.get("sample_record"))
    return ind.permutation_gains(steps, n_pop=plan["n_pop"], models=plan["models"],
                                 tau_lag_s=plan["tau_lag_s"], D=plan["D"], fractions=plan["fractions"],
                                 constant_ll=plan["constant_ll"], seeds=list(seeds))


def permutation_lines(result: dict) -> list:
    """The card's lines for the permutation test: ``"Shares changed (permutation p = 0.005)."`` beside the held-out
    verdict, which it never replaces, then what was compared and how the two answers differ in what they ask."""
    plan = result.get("permutation")
    if not plan:
        return []
    if plan.get("problem"):
        return [plan["problem"]]
    s = plan.get("summary")
    if s is None:
        return []
    word = _PERMUTATION_WORDS[s["change"]]
    best = f"{word}, {s['shape']}" if s.get("shape") else word
    n = s["n_shuffles"]
    p = s["p"]
    p_text = f"{p:.3f}" if p < 0.1 else f"{p:.2f}"
    floor = " (the smallest that many shuffles can give)" if p <= s["smallest_p"] + 1e-12 else ""
    m = int(s.get("n_models", 1))
    # EXACT (7 Nov 2026): every clock is scored by its own best changing model, so p answers "does the clock matter
    # at all"; the kind of change is the held-out comparison's answer, the model named here only gained most
    head = (f"A change over time (permutation p = {p_text}{floor}); the model that gains most: {best}."
            if s["shown"] else f"No change shown by the permutation test (p = {p_text}; the model that gains most: "
                               f"{best}).")
    null = np.asarray(s["null_gains"], float)
    dealt = ("the samples' addition times among the samples" if (result.get("settings") or {}).get("design") == "sample"
             else "the movies' clocks")
    lines = [head,
             (f"The best of the {m} changing models gains {s['observed_gain']:.0f} log-likelihood units over the "
              f"constant mixture; on {n} shuffles of {dealt} within each replicate, the best of them gains "
              f"{null.min():.0f} to {null.max():.0f} (median {np.median(null):.0f}). Each shuffle is scored by its "
              f"own best model.")
             if m > 1 else
             (f"The changing model gains {s['observed_gain']:.0f} log-likelihood units over the constant mixture; on "
              f"{n} shuffles of {dealt} within each replicate it gains {null.min():.0f} to "
              f"{null.max():.0f} (median {np.median(null):.0f}).")]
    held = result.get("held_out_change", "none")
    if s["shown"] and held == "none":
        lines.append("The held-out verdict above stands. It asks whether a changing model predicts new movies better, "
                     "which scatter between movies can hide; the permutation test asks whether the order of the "
                     "clock matters at all.")
    elif not s["shown"] and held != "none":
        lines.append("The held-out verdict above stands, but the permutation test does not confirm it.")
    return lines


def attach_permutation(result: dict, summary: dict) -> dict:
    """``result`` with the finished test (`induction.permutation_summary`) in place and its lines on the card."""
    plan = dict(result["permutation"])
    plan["summary"] = summary
    out = dict(result)
    out["permutation"] = plan
    out["text"] = dict(result["text"], permutation=permutation_lines(out))
    return out


def run_permutation(project, condition: str, result: dict, *, n_shuffles: int | None = None, seed: int = 0,
                    submit=None, cache=None, progress=None, cancelled=None) -> dict:
    """``result`` (`compute_induction` with ``permutation=True``) with the time-label permutation test run and
    attached. ``submit(function, *args, **kwargs) -> Future`` runs each chunk of shuffles (a pool, through the section
    cache when ``cache`` is given); None runs them here. ``progress(done, total)`` counts chunks. A result that is an
    error, or has no test to run (``result["permutation"]`` empty or with a problem), is returned as it is."""
    from . import induction as ind
    from .sectioncache import cached_job

    plan = result.get("permutation")
    if "error" in result or not plan or plan.get("problem"):
        if plan and plan.get("problem") and "error" not in result:
            return dict(result, text=dict(result["text"], permutation=[plan["problem"]]))
        return result
    shuffles = ind.PERMUTATIONS if n_shuffles is None else int(n_shuffles)
    # the shuffles are built as the observed steps were: the same clock and, Per spot, the same step noise
    clock = {k: result["settings"].get(k) for k in ("trigger_frame", "movie_start_s", "precision", "sample_record")}
    seeds = ind.permutation_seeds(shuffles, seed)
    submit = submit or ind._run_now

    def chunk(part):
        return submit(cached_job, cache, "kinetics", permutation_chunk, project, condition,
                      {"clock": clock, "plan": {k: plan[k] for k in ("models", "n_pop", "tau_lag_s", "D",
                                                                       "fractions", "constant_ll")},
                       "seeds": part})

    gains = ind.run_permutation_chunks(chunk, seeds, progress=progress, cancelled=cancelled)
    summary = ind.permutation_summary(seeds, gains, models=plan["models"], replicate_count=0)
    return attach_permutation(result, summary)


# --------------------------------------------------------------------------- #
# A reference curve drawn beside the model (7 Nov 2026)
# --------------------------------------------------------------------------- #
#
# ANY curves a person has, as a CSV: ``t_s, population, D_um2_s, share``. The app knows nothing of where they came
# from -- it draws them dashed, called "Reference" -- so a synthetic set's declared truth can be laid over its
# analysis without the app ever loading that truth as such (the deposit firewall,
# `docs/claude/manuscript_deposits.md`). `scripts/synthetic_builds/synk_reference_curves.py` makes one from SynK's
# truth table.
#
# THE FORMAT, nothing guessed: the four columns by name (any order; an optional ``condition`` column limits a row to
# that condition, empty = every condition); ``t_s`` seconds since the induction, on the card's clock; ``population``
# the card's name ("slow", "middle", "fast") or its number (1 = slowest); ``D_um2_s`` in um2/s; ``share`` a FRACTION
# between 0 and 1 (0.35, never 35 %). An empty D or share is skipped; a D of 0 is kept in the file and not drawn on
# the log axis. A file that does not read is refused by name, never drawn partly.

REFERENCE_COLUMNS = ("t_s", "population", "D_um2_s", "share")
REFERENCE_LABEL = "Reference"


def find_reference(condition_roots) -> list:
    """The ``reference*.csv`` files sitting in the project root, the folder BESIDE the condition folders, as paths
    sorted by name (empty where the conditions do not share one parent folder, or none is there). The Run dialog takes
    the first on its own and says so (`kinetics_view.reference_note`)."""
    from pathlib import Path

    parents = {Path(r).parent for r in condition_roots}
    if len(parents) != 1:
        return []
    parent = parents.pop()
    try:
        return sorted((f for f in parent.iterdir() if f.is_file() and f.suffix.lower() == ".csv"
                       and f.name.lower().startswith("reference")), key=lambda f: f.name.lower())
    except OSError:
        return []


def reference_curves(path, condition: str | None = None) -> dict:
    """``{"path", "curves": {population label: {"t_s": [...], "D": [...], "share": [...]}}}`` read from ``path``
    (the module comment above is the format), limited to ``condition``'s rows where the file has a ``condition``
    column. RAISES ValueError naming the file and what is wrong."""
    import pandas as pd

    try:
        table = pd.read_csv(path)
    except Exception as exc:                          # noqa: BLE001 - named
        raise ValueError(f"reference curves {path}: cannot be read ({type(exc).__name__}: {exc})") from exc
    missing = [c for c in REFERENCE_COLUMNS if c not in table.columns]
    if missing:
        raise ValueError(f"reference curves {path}: no column {', '.join(missing)} (needs "
                         f"{', '.join(REFERENCE_COLUMNS)})")
    if "condition" in table.columns and condition is not None:
        mine = table["condition"].isna() | (table["condition"].astype(str).str.strip() == "") \
            | (table["condition"].astype(str) == str(condition))
        table = table[mine]
    if table.empty:
        raise ValueError(f"reference curves {path}: no rows" + (f" for {condition}" if condition else ""))
    for column in ("t_s", "D_um2_s", "share"):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    if table["t_s"].isna().any():
        raise ValueError(f"reference curves {path}: t_s must be a number on every row")
    if (table["D_um2_s"].dropna() < 0).any():
        raise ValueError(f"reference curves {path}: D_um2_s cannot be negative")
    share = table["share"].dropna()
    if ((share < 0) | (share > 1)).any():
        raise ValueError(f"reference curves {path}: share is a fraction between 0 and 1 (0.35, not 35); "
                         f"found {share.min():g} to {share.max():g}")
    curves = {}
    for label, part in table.groupby(table["population"].astype(str).str.strip(), sort=False):
        part = part.sort_values("t_s", kind="stable")
        curves[label] = {"t_s": [float(v) for v in part["t_s"]],
                         "D": [None if pd.isna(v) else float(v) for v in part["D_um2_s"]],
                         "share": [None if pd.isna(v) else float(v) for v in part["share"]]}
    return {"path": str(path), "curves": curves}


def attach_reference(result: dict, reference: dict) -> dict:
    """``result`` with ``reference``'s curves in the card's series (``series["reference"]``: ``(t, value)``, D in
    um2/s and share in %, a missing value skipped), each matched to a population by its name or number. A label the
    card has no population for is refused by name; a card population the file does not cover has none drawn."""
    if "error" in result:
        return result
    chart = result["chart"]
    names = [s["name"] for s in chart["D"]["series"]]
    where = {}
    for label in reference["curves"]:
        key = label.lower()
        if key in names:
            where[label] = names.index(key)
        elif key.isdigit() and 1 <= int(key) <= len(names):
            where[label] = int(key) - 1
        else:
            raise ValueError(f"reference curves {reference.get('path', '')}: population {label!r} is not one of the "
                             f"card's ({', '.join(names)}) or its number 1-{len(names)}")
    if len(set(where.values())) != len(where):
        raise ValueError(f"reference curves {reference.get('path', '')}: two labels name one population")
    new = {"D": [dict(s, reference=[]) for s in chart["D"]["series"]],
           "fractions": [dict(s, reference=[]) for s in chart["fractions"]["series"]]}
    for label, k in where.items():
        c = reference["curves"][label]
        new["D"][k]["reference"] = [(t, v) for t, v in zip(c["t_s"], c["D"]) if v is not None and v > 0]
        new["fractions"][k]["reference"] = [(t, 100.0 * v) for t, v in zip(c["t_s"], c["share"]) if v is not None]
    values = [v for s in new["D"] for _t, v in s["reference"]]
    d_range = chart["D"]["y_range"]
    if values:                                       # the axis holds the reference too (whole decades, as `_chart`)
        d_range = (min(d_range[0], 10.0 ** math.floor(math.log10(min(values)))),
                   max(d_range[1], 10.0 ** math.ceil(math.log10(max(values)))))
    out = dict(result)
    out["chart"] = dict(chart, D=dict(chart["D"], series=new["D"], y_range=d_range),
                        fractions=dict(chart["fractions"], series=new["fractions"]))
    out["reference"] = {"path": reference.get("path", ""), "label": REFERENCE_LABEL,
                        "populations": sorted(where.values())}
    return out


def has_reference(result: dict) -> bool:
    return bool(result.get("reference"))


def _band(values: np.ndarray) -> tuple:
    """The 95 % percentile band of whole refits at each grid point (n_boot x len(grid))."""
    return np.percentile(values, 2.5, axis=0), np.percentile(values, 97.5, axis=0)


def _replicate_points(rep_bins, column: str, scale: float) -> list:
    """``[(replicate, t, value)]`` from the per-replicate bin fits (`induction.replicate_binned_fits`), a bin with no fit
    left out; the replicate as its integer number where it is one."""
    out = []
    if rep_bins is None or not len(rep_bins) or column not in rep_bins.columns:
        return out
    for _i, row in rep_bins.iterrows():
        v = row.get(column, math.nan)
        if np.isfinite(v):
            rep = row["replicate"]
            out.append((int(rep) if float(rep).is_integer() else str(rep), float(row["t_s"]), float(v) * scale))
    return out


#: The legend entry of the faint curves behind the chosen one (card and export).
#: Shortened 8 Nov 2026 (was "shapes the data cannot tell apart"; the legend sentence still says what they are).
ALTERNATIVES_LABEL = "equally good shapes"


def _alternatives(compare: dict, fit, grid) -> dict:
    """``{shape: fit}`` of every shape of the chosen change the held-out units could not tell from the chosen one,
    refitted on all steps (the fits `induction.compare_models` already made), drawn faintly behind the chosen curve."""
    return {s: compare["fits"][(fit.change, s)] for s in compare["shape_undetermined"]
            if (fit.change, s) in compare["fits"]}


def shape_words(shape: str, shape_open) -> str:
    """The verdict's words on the shape: nothing open, the shape alone ("exponential response"); otherwise how it was
    chosen (the declared default order, `induction.SHAPE_PREFERENCE`) and what fits as well."""
    from .induction import SHAPE_PREFERENCE

    if not shape_open:
        return f"{shape} response"
    others = " and ".join(sorted(shape_open, key=SHAPE_PREFERENCE.index))
    default = ("the first-order default" if shape == SHAPE_PREFERENCE[0]
               else f"the first of the default order {', '.join(SHAPE_PREFERENCE)} that fits")
    return f"drawn as {shape}, {default}; {others} fit as well"


def _chart(fit, grid, boot, bins, t0: float, t1: float, rep_bins=None, alternatives=None) -> dict:
    """Plain lists for the two charts: D (um2/s, log axis) and share (%), each population a series. A series'
    ``replicates`` are each replicate's own per-bin fit (``(replicate, t, value)``), its ``bins`` the pooled fits."""
    names = population_names(fit.n_pop)
    D_model, f_model = fit.D_at(grid), fit.fractions_at(grid) * 100.0
    curves = boot.get("curves") or {}
    d_series, f_series = [], []
    for k, name in enumerate(names):
        d_lo, d_hi = _band(curves["D"][:, :, k]) if len(curves.get("D", ())) else (D_model[:, k], D_model[:, k])
        f_lo, f_hi = (_band(curves["fractions"][:, :, k] * 100.0) if len(curves.get("fractions", ()))
                      else (f_model[:, k], f_model[:, k]))
        d_bins, f_bins = [], []
        if bins is not None and len(bins):
            for _i, row in bins.iterrows():
                t = float(row["t_s"])                       # the bin's middle
                d, f = row.get(f"D_{k + 1}", math.nan), row.get(f"fraction_{k + 1}", math.nan)
                if np.isfinite(d):
                    d_bins.append((t, float(d)))
                if np.isfinite(f):
                    f_bins.append((t, float(f) * 100.0))
        alt_D = {s: f.D_at(grid) for s, f in (alternatives or {}).items()}
        alt_f = {s: f.fractions_at(grid) * 100.0 for s, f in (alternatives or {}).items()}
        d_series.append({"name": name, "curve": [(float(t), float(v)) for t, v in zip(grid, D_model[:, k])],
                         "alternatives": [{"shape": s, "curve": [(float(t), float(v)) for t, v in zip(grid, a[:, k])]}
                                          for s, a in alt_D.items()],
                         "band": [(float(t), float(a), float(b)) for t, a, b in zip(grid, d_lo, d_hi)],
                         "bins": d_bins, "replicates": _replicate_points(rep_bins, f"D_{k + 1}", 1.0)})
        f_series.append({"name": name, "curve": [(float(t), float(v)) for t, v in zip(grid, f_model[:, k])],
                         "alternatives": [{"shape": s, "curve": [(float(t), float(v)) for t, v in zip(grid, a[:, k])]}
                                          for s, a in alt_f.items()],
                         "band": [(float(t), float(a), float(b)) for t, a, b in zip(grid, f_lo, f_hi)],
                         "bins": f_bins, "replicates": _replicate_points(rep_bins, f"fraction_{k + 1}", 100.0)})
    d_values = [v for s in d_series for _t, lo, hi in s["band"] for v in (lo, hi) if v > 0]
    d_values += [v for s in d_series for _t, v in s["bins"] if v > 0]
    d_values += [v for s in d_series for _r, _t, v in s["replicates"] if v > 0]
    d_values += [v for s in d_series for alt in s["alternatives"] for _t, v in alt["curve"] if v > 0]
    d_lo = 10.0 ** math.floor(math.log10(max(min(d_values), 1e-4))) if d_values else 1e-3
    d_hi = 10.0 ** math.ceil(math.log10(max(d_values))) if d_values else 10.0
    return {"t_range": (t0, t1), "induction_s": 0.0,
            # one population has no share to draw: its share is 100 % throughout (the mean mobility readout)
            "share_panel": fit.n_pop > 1,
            "D": {"series": d_series, "y_range": (d_lo, max(d_hi, d_lo * 10.0)), "log": True},
            "fractions": {"series": f_series, "y_range": (0.0, 100.0), "log": False}}


def _with_flags(text: dict, extra: list) -> dict:
    """``text`` with ``extra`` lines appended to its flags (the caveats drawn in the flag colour)."""
    return dict(text, flags=list(text["flags"]) + list(extra))


def d_panels(chart: dict) -> list:
    """One D panel PER POPULATION (8 Nov 2026): ``[{"name", "series": [that population's series], "y_range", "log"}]``
    from the card's ``chart["D"]``, read at draw time so a reference attached afterwards is held too.

    The slow (about 0.02) and fast (about 0.5 um2/s) populations on one log axis from 0.01 to 1 made a 1.6-fold change
    of the fast D a short step. Each panel is LINEAR and its y range is fitted to its own curve, band, per-bin fits,
    replicates, reference and faint alternative shapes, with 8 % of room on each side (never below 0)."""
    panels = []
    for s in chart["D"]["series"]:
        values = [v for _t, v in s["curve"]] + [v for _t, lo, hi in s["band"] for v in (lo, hi)]
        values += [v for _t, v in s["bins"]] + [v for _r, _t, v in s.get("replicates") or ()]
        values += [v for _t, v in s.get("reference") or ()]
        values += [v for alt in s.get("alternatives") or () for _t, v in alt["curve"]]
        values = [float(v) for v in values if v is not None and math.isfinite(v)]
        lo, hi = (min(values), max(values)) if values else (0.0, 1.0)
        room = 0.08 * (hi - lo) if hi > lo else 0.1 * max(abs(hi), 1e-3)
        panels.append({"name": s["name"], "series": [s], "log": False,
                       "y_range": (max(lo - room, 0.0), hi + room)})
    return panels


#: A D on the fit's floor (`induction.d_at_floor`), as the HMM card says it (`hmm_view.NOT_RESOLVED`).
NOT_RESOLVED = "not resolved"


def _fmt_D(value: float) -> str:
    from .induction import d_at_floor

    return NOT_RESOLVED if d_at_floor(value) else f"{value:.3g}"


def _d_words(values, band=None) -> str:
    """``values`` (before, or before and where the data end) and their 95 % band as the card writes them, "0.574 ->
    0.304 um2/s [0.282, 0.338]". A value on the fit's floor reads "not resolved" (10 Nov 2026), and a band lying wholly
    on it is left out, since it is the floor twice; a band reaching it keeps its ends as numbers."""
    from .induction import d_at_floor

    shown = [_fmt_D(v) for v in values]
    text = " → ".join(shown)
    if any(s != NOT_RESOLVED for s in shown):
        text += " µm²/s"
    if band is not None and not all(d_at_floor(b) for b in band):
        text += f" [{band[0]:.3g}, {band[1]:.3g}]"
    return text


# --------------------------------------------------------------------------- #
# The half-time as numbers (8 Nov 2026)
# --------------------------------------------------------------------------- #
# Asked: "can our software calculate halftime or some kinetic value out of this? and if not -> implement. and print
# here in figure 6?" The fit always had it (the card's sentence, `_text`); it is now kept as numbers too, so the card
# and the figure mark it (a dotted line, "t1/2 = 3.3-3.4 min") and the source data hold it. One rule with the sentence:
# a range over the shapes the data cannot tell apart where they cannot, else the one shape's value with its 95 %
# interval over resampled units where the bootstrap gave one.

#: What a half-time record's two ends are: the range over the equally good shapes, or the 95 % interval
HALF_TIME_KINDS = ("range", "interval", "value")


def half_time_record(compare: dict, fit, boot: dict) -> dict | None:
    """``{"value_s", "low_s", "high_s", "kind"}`` (`HALF_TIME_KINDS`) for a fit that found a change; None without one.
    The same choice as the card's sentence (`_text`)."""
    if fit.change == "none":
        return None
    value = None if fit.half_time_s is None else float(fit.half_time_s)
    span = compare.get("half_time_range_s")
    if compare["shape_undetermined"] and span:
        return {"value_s": value, "low_s": float(span[0]), "high_s": float(span[1]), "kind": "range"}
    if value is None:
        return None
    ci = (boot.get("interval") or {}).get("half_time_s")
    if ci:
        return {"value_s": value, "low_s": float(ci[0]), "high_s": float(ci[1]), "kind": "interval"}
    return {"value_s": value, "low_s": None, "high_s": None, "kind": "value"}


def half_time_panels(change: str | None) -> tuple:
    """The panels that carry the half-time mark: the D panels where the D's changed, the share panel where the shares
    did ("D" / "share"); none without a change."""
    return {"D": ("D",), "fractions": ("share",), "both": ("D", "share")}.get(change or "none", ())


def reference_half_time(chart: dict, which: str) -> float | None:
    """When a reference curve (a known curve the person chose, `attach_reference`) crossed its midpoint, in s:
    ``which`` "D" the population whose reference D changed most (relatively), "share" the slowest's share. Its
    baseline is every point before the induction, its end every point in the last fifth of the record; the crossing is
    interpolated between the points around it. None without a reference, a baseline or a crossing."""
    if which == "D":
        candidates = [s.get("reference") or () for s in chart["D"]["series"]]
    else:
        candidates = [(chart["fractions"]["series"][0].get("reference") or ())]
    best, best_change = None, 0.0
    for ref in candidates:
        pts = sorted((float(t), float(v)) for t, v in ref)
        if len(pts) < 3:
            continue
        last = pts[-1][0]
        before = [v for t, v in pts if t < 0]
        end = [v for t, v in pts if t >= 0.8 * last]
        if not before or not end or last <= 0:
            continue
        b, e = sum(before) / len(before), sum(end) / len(end)
        change = abs(e - b) / max(abs(b), 1e-12)
        if change <= best_change:
            continue
        mid = (b + e) / 2
        after = [(t, v) for t, v in pts if t > 0]
        for (t0, v0), (t1, v1) in zip(after, after[1:]):
            if (v1 - mid) * (e - b) >= 0 > (v0 - mid) * (e - b):
                best, best_change = t0 + (mid - v0) / (v1 - v0) * (t1 - t0), change
                break
    return best


def half_time_words(record: dict | None, *, time_unit: str = "s", reference_s: float | None = None) -> list:
    """The mark's words without the symbol: ``["= 3.3-3.4 min", "vs 2.3 min (reference)"]`` (the second only with a
    reference); [] without a half-time. The card writes "t₁/₂" before the first, the figure its math."""
    if not record:
        return []
    k = TIME_UNITS[time_unit]
    digits = 1 if time_unit == "min" else 0

    def fmt(v):
        return f"{v / k:.{digits}f}"

    if record["kind"] == "range" and record["low_s"] is not None:
        lo, hi = fmt(record["low_s"]), fmt(record["high_s"])
        first = f"= {lo} {time_unit}" if lo == hi else f"= {lo}–{hi} {time_unit}"
    elif record["kind"] == "interval":
        first = f"= {fmt(record['value_s'])} {time_unit} [{fmt(record['low_s'])}, {fmt(record['high_s'])}]"
    else:
        first = f"= {fmt(record['value_s'])} {time_unit}"
    lines = [first]
    if reference_s is not None:
        lines.append(f"vs {fmt(reference_s)} {time_unit} (reference)")
    return lines


def half_time_mark(result: dict, which: str, curve, *, time_unit: str = "s", symbol: str = "t₁/₂"):
    """What a panel draws for the half-time, card and figure alike: ``{"lo_s", "hi_s", "mid_s", "lines", "rising"}``
    -- the band (the range or the interval; a line where neither), the dotted line (the range's middle, else the
    value), the words (``symbol`` before the first line; the reference's half-time on the second where one is drawn),
    and whether the panel's curve rises (the words then sit at the bottom corner, else the top). None without a
    half-time."""
    record = result.get("half_time")
    if not record:
        return None
    lo = record["low_s"] if record["low_s"] is not None else record["value_s"]
    hi = record["high_s"] if record["high_s"] is not None else record["value_s"]
    if lo is None or hi is None:
        return None
    mid = (lo + hi) / 2 if record["kind"] == "range" or record["value_s"] is None else record["value_s"]
    lines = half_time_words(record, time_unit=time_unit, reference_s=reference_half_time(result["chart"], which))
    lines[0] = f"{symbol} {lines[0]}"
    rising = bool(curve) and float(curve[-1][1]) > float(curve[0][1])
    return {"lo_s": float(lo), "hi_s": float(hi), "mid_s": float(mid), "lines": lines, "rising": rising}


def _text(compare: dict, fit, boot: dict, onset, spread: bool, unit: str) -> dict:
    """The verdict, the before -> end values, the half-time, the flags and the model rows."""
    names = population_names(fit.n_pop)
    change_open = compare["change_undetermined"]
    shape_open = compare["shape_undetermined"]
    one = fit.n_pop == 1
    if fit.change == "none":
        headline = (f"No change shown: no changing model predicts held-out {unit}s better than the constant "
                    f"{'D' if one else 'mixture'} by 2 standard errors.")
    elif one:
        headline = f"The mean D changed ({shape_words(fit.shape, shape_open)})."
    else:
        headline = f"{_CHANGE_WORDS[fit.change][0].upper()}{_CHANGE_WORDS[fit.change][1:]}"
        headline += f" ({shape_words(fit.shape, shape_open)})."
    lines, flags = [headline], []
    interval = boot["interval"]
    curves = boot.get("curves") or {}
    for k, name in enumerate(names):
        d0, d1 = fit.D_before[k], fit.D_end[k]
        f0, f1 = 100.0 * fit.fractions_before[k], 100.0 * fit.fractions_end[k]
        d_end_band = (_band(curves["D"][:, -1:, k])) if len(curves.get("D", ())) else None
        f_end_band = (_band(curves["fractions"][:, -1:, k] * 100.0)) if len(curves.get("fractions", ())) else None
        d_before_ci = interval.get(f"D_before_{k + 1}")
        f_before_ci = interval.get(f"fraction_before_{k + 1}")
        d_moves = fit.change in ("D", "both")
        f_moves = fit.change in ("fractions", "both")
        if d_moves:
            d_text = "D " + _d_words([d0, d1], (d_end_band[0][0], d_end_band[1][0]) if d_end_band is not None
                                     else None)
        else:
            d_text = ("D " + _d_words([d0], tuple(d_before_ci) if d_before_ci else None)
                      + (" throughout" if fit.change != "none" else ""))
        if f_moves:
            f_text = (f"share {f0:.1f} → {f1:.1f} %"
                      + (f" [{f_end_band[0][0]:.1f}, {f_end_band[1][0]:.1f}]" if f_end_band is not None else ""))
        else:
            f_text = (f"share {f0:.1f} %"
                      + (f" [{100 * f_before_ci[0]:.1f}, {100 * f_before_ci[1]:.1f}]" if f_before_ci else "")
                      + (" throughout" if fit.change != "none" else ""))
        line = f"Mean D: {d_text[2:]}" if one else f"{name.capitalize()}: {d_text}, {f_text}"
        if d_moves or f_moves:
            line += " (before → where the data end)"
        lines.append(line)
    if fit.change != "none":
        half_range = compare.get("half_time_range_s")
        half_ci = interval.get("half_time_s")
        if shape_open and half_range:
            lines.append(f"Half-time {half_range[0]:.1f}-{half_range[1]:.1f} s after the induction: the range over "
                         f"the shapes it cannot tell apart ({fit.shape}, {', '.join(shape_open)}).")
        elif fit.half_time_s is not None:
            lines.append(f"Half-time {fit.half_time_s:.1f} s after the induction"
                         + (f" [{half_ci[0]:.1f}, {half_ci[1]:.1f}]" if half_ci else "") + ".")
        if change_open:
            flags.append(f"Held-out {unit}s cannot tell this from "
                         f"{' or '.join(_OPEN_CHANGE_WORDS[c] for c in change_open)}.")
        done = fit.completion
        from .induction import COMPLETE

        if done < COMPLETE:
            flags.append(f"The response is {100 * done:.0f} % complete where the data end. The 'after' values are "
                         f"the shape's extrapolation, so the values above are those where the data end, and the "
                         f"plateau may differ.")
    if boot.get("caveat"):
        flags.append(boot["caveat"] + ".")
    spread_lines = None
    if spread:
        if onset is None:
            spread_lines = ["Onset test between movies: not run, there is no change to time."]
        else:
            spread_lines = [onset["verdict"][0].upper() + onset["verdict"][1:] + ".",
                            f"Against {onset['n_shuffles']} shuffles of whole cells between movies. The test is weak "
                            f"with about 12 movies; onsets between cells are not measurable at this data size and "
                            f"are not tested."]
    rows = []
    recommended = compare["recommended"]
    for row in compare["table"].itertuples():
        rows.append({"model": _model_label(row.change, row.shape or ""), "against_best": float(row.against_best),
                     "z_behind": float(row.z_behind_best), "units_won": int(row.units_won),
                     "chosen": (row.change, row.shape or None) == (recommended[0], recommended[1])})
    notes = [MOBILITY_NOTE] if one else []
    if boot.get("stratified"):
        notes.append(f"Intervals resample {unit}s within each of the {boot['n_strata']} replicates, every replicate "
                     f"keeping its own number of them: the replicates are held fixed, so the bands leave out the "
                     f"variation between replicates. Unit Replicate resamples the replicates themselves.")
    return {"lines": lines, "flags": flags, "notes": notes, "spread": spread_lines, "models": rows,
            "permutation": None, "n_units": int(compare["n_units"]), "unit": unit}


def induction_model_cells(rows: list) -> list:
    """The model table's rows as text: ``[model, against the best, behind by, units won, chosen?]``."""
    out = []
    for r in rows:
        behind = "best" if r["against_best"] == 0 else f"{r['z_behind']:.1f} SE"
        out.append([r["model"], f"{r['against_best']:+.1f}", behind, str(r["units_won"]), bool(r["chosen"])])
    return out


def induction_caption(result: dict) -> str:
    """The card's caption: the clock, the bins drawn, the unit, the populations."""
    s = result["settings"]
    design = s.get("design")
    if s["trigger_frame"] is not None:
        clock = f"induction at frame {s['trigger_frame']}"
    elif design == "sample":
        starts = sorted((s.get("movie_start_s") or {}).values())
        clock = ("each movie's time since its own sample's addition"
                 + (f" ({starts[0] / 60:.0f} to {starts[-1] / 60:.0f} min)" if starts else ""))
    elif s.get("additions"):
        starts = sorted((s.get("movie_start_s") or {}).values())
        clock = ("each movie's time since its replicate's addition"
                 + (f" ({starts[0] / 60:.0f} to {starts[-1] / 60:.0f} min)" if starts else ""))
    else:
        clock = "each movie's own time since the induction"
    bins = f"{s['bin_seconds']:g} s bins drawn" if s["bin_seconds"] else "no bins drawn"
    per_spot = f" · each step's own localisation error ({s['precision'][2]})" if s.get("precision") else ""
    pops = "mean mobility (one D)" if s["n_pop"] == 1 else f"{s['n_pop']} populations"
    return (f"{clock} · {pops} · per {s['unit']} · {bins} · "
            f"{result['n_steps']:,} steps" + per_spot)


def population_hover(series: list, t: float, *, unit: str, time_unit: str = "s") -> str:
    """The value of every population's model curve nearest ``t`` (in ``time_unit`` since the induction)."""
    parts = []
    t_s = t * TIME_UNITS[time_unit]
    for s in series:
        if not s["curve"]:
            continue
        i = int(np.argmin([abs(tt - t_s) for tt, _v in s["curve"]]))
        tt, v = s["curve"][i]
        _t, lo, hi = s["band"][i]
        if unit == "D":
            from .induction import d_at_floor

            # a value on the fit's floor is said, and a band wholly on it left out (10 Nov 2026, `_d_words`)
            value = _fmt_D(v)
            band = None if d_at_floor(lo) and d_at_floor(hi) else f"{lo:.3g}-{hi:.3g}"
        else:
            value, band = f"{v:.1f} %", f"{lo:.1f}-{hi:.1f} %"
        parts.append(f"{s['name']}: {value}" + (f" ({band})" if band else ""))
    return f"{t:.1f} {time_unit} since the induction\n" + "\n".join(parts) if parts else ""
