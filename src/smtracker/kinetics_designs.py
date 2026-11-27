"""The Kinetics section's EXPERIMENT DESIGNS: what "time since the addition" means, what is independent, what is
shuffled, and the lines every card says about what was actually observed. No Qt, no project access.

SPEC: ``docs/kinetics_designs_spec.md`` (approved 7 Nov 2026), grounded in
a methods review ("the review"). The design changes three things, not only the clock:
the clock, the unit every interval and test is computed over, and what the permutation test may deal.

    key          the experimenter's words           clock                                   shuffle
    microscope   Added under the microscope         the trigger frame, or one addition      movies' clocks within a replicate,
                                                    time per replicate and each movie's     or tracks' times within a movie
                                                    camera start
    culture      Added to the culture               one addition time per replicate and     movies' clocks within a replicate
                                                    each movie's camera start (as before)
    sample       Added to each sample               each sample's own addition time and     samples' addition times among the
                                                    its movies' camera starts               samples of a replicate

RULE 0 stays: the replicate (the ``day`` column, the CULTURE the movies or samples came from) is the independent unit
in every design; movies and samples are time points of its response, never replicates.

OLD CONFIGS: a config without a ``design`` key reads as it always did (`design_of`): "culture" for the addition clock,
and "microscope" for the trigger clock (the only design in which a frame of the addition inside a movie exists).

WHAT EVERY DESIGN SAYS ON THE CARD (`design_lines`), the review's demands:
  * the time window observed (first and last movie after the addition),
  * the baseline (movies before the addition, or that there are none),
  * the acquisition per time window, always (`window_qc`: mean track length, median per-spot error, tracks): a trend is
    NAMED, never a refusal, because a real effect can shorten tracks; the censoring-corrected refit the review
    recommends is not yet made, and the card says so,
  * the within-track gate as a STATEMENT, never a test (`within_track_text`, the review's table `GATE_TABLE`).
The within-track test, the per-cell onset and the following of the same cells across movies are NOT built.

THE PER-SAMPLE TABLE (design "sample"): every movie is its own sample by default, with its own typed addition time;
movies may be grouped into one sample by hand. It is saved per condition as ``kinetics_additions.json`` beside
``day_map.json`` (`save_additions`, written on Run only, never on opening). Every movie needs a time: a missing one is
refused by name (`sample_problem`).

THE CULTURE TIMES: the same file also remembers the addition time typed per replicate under the addition clock (the
design "Added to the culture"), in a ``"culture"`` block (`save_culture_times`, written on Run only; a file without
the block reads as before). The dialog's "Same time for every replicate" fills every replicate field of every ticked
condition once (`same_time_fill`); a replicate stays editable after.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

__all__ = [
    "DESIGNS", "DEFAULT_DESIGN", "DESIGN_WORDS", "DESIGN_EXPLANATIONS", "DESIGN_UNITS", "DESIGN_CLOCKS",
    "design_of", "ADDITIONS_FILE", "additions_path", "load_additions", "save_additions", "load_culture_times", "save_culture_times", "same_time_fill",
    "seed_culture_times", "samples_from_rows",
    "sample_problem", "parse_paste", "GATE_SOURCE", "GATE_TABLE", "GATE_MIN_SEGMENT", "gate_row", "window_edges",
    "window_qc", "qc_trend", "within_track_text", "design_lines", "when_words",
]

DESIGNS = ("microscope", "culture", "sample")
#: Every config that names no design reads as this one (the clock the section had before the designs).
DEFAULT_DESIGN = "culture"
DESIGN_WORDS = {"microscope": "Added under the microscope", "culture": "Added to the culture",
                "sample": "Added to each sample"}
#: Two lines under each name in the Run dialog (what is measured from where; which box is independent).
DESIGN_EXPLANATIONS = {
    "microscope": ("The compound reaches the cells while they sit under the objective (flow cell, chamber), so the same "
                   "cells can be imaged before and after. Time runs from the addition, a frame of each movie or one "
                   "clock time per replicate.",
                   "The replicate, the chamber, is the independent unit; its movies are time points of one response."),
    "culture": ("It is added to the culture, which is incubated and mounted; movie 1, 2, 3 of that culture follow. "
                "Time runs from the clock time of the addition to each movie's camera start.",
                "The replicate, the culture, is the independent unit; its movies are time points of one response, "
                "not replicates."),
    "sample": ("Each sample (a slide, a tube, or one movie) is induced on its own, at its own time. Time runs from "
               "each sample's own addition to its movies' camera starts.",
               "The replicate, the culture the samples came from, is the independent unit; samples are time points "
               "within it, unless each came from its own culture."),
}
#: The unit badge of each animation.
DESIGN_UNITS = {"microscope": "chamber", "culture": "culture", "sample": "culture (samples within it)"}
#: The clocks each design offers (`induction_view.CLOCKS`).
DESIGN_CLOCKS = {"microscope": ("trigger", "addition"), "culture": ("addition",), "sample": ("sample",)}


def design_of(config: dict | None) -> str:
    """The design a config means: its ``design`` key, else (old configs) by the clock it names."""
    config = config or {}
    design = config.get("design")
    if design is not None:
        if design not in DESIGNS:
            raise ValueError(f"unknown design {design!r}: choose one of {', '.join(DESIGNS)}")
        return design
    return "microscope" if config.get("clock") == "trigger" else DEFAULT_DESIGN


# --------------------------------------------------------------------------- #
# The per-sample table, beside the day map
# --------------------------------------------------------------------------- #

ADDITIONS_FILE = "kinetics_additions.json"


def additions_path(condition_root) -> Path:
    """Where a condition's per-sample addition table lives: beside its ``day_map.json``."""
    return Path(condition_root) / ADDITIONS_FILE


def _read_file(path: Path) -> dict | None:
    """The saved file as a dict, or None where there is none. A file that does not read is REFUSED by name."""
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError("the file does not hold an object")
        return data
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"{path}: the saved addition times cannot be read ({type(exc).__name__}: {exc})") from exc


def load_additions(condition_root) -> dict | None:
    """``{sample: {"movies": [...], "time": text}}`` saved for the condition, or None where there is no table (no
    file, or a file holding only the culture times, `load_culture_times`). A file that does not read is REFUSED by
    name, never read as empty."""
    path = additions_path(condition_root)
    data = _read_file(path)
    if data is None or "samples" not in data:
        return None
    try:
        return {str(k): {"movies": [str(m) for m in v["movies"]], "time": str(v.get("time", ""))}
                for k, v in data["samples"].items()}
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"{path}: the saved addition times cannot be read ({type(exc).__name__}: {exc})") from exc


def load_culture_times(condition_root) -> dict | None:
    """``{replicate: typed time}`` saved for the design "Added to the culture" (and the microscope's addition clock),
    or None where none was saved (no file, or a file from before 7 Nov 2026 without the block)."""
    path = additions_path(condition_root)
    data = _read_file(path)
    if data is None or "culture" not in data:
        return None
    try:
        return {int(k): str(v) for k, v in data["culture"]["replicates"].items()}
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"{path}: the saved addition times cannot be read ({type(exc).__name__}: {exc})") from exc


def _write_file(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def save_additions(condition_root, samples: dict) -> Path:
    """Write the condition's per-sample table (Run only). Refuses an empty table and a movie in two samples. The
    culture block of the same file (`save_culture_times`) is kept."""
    if not samples:
        raise ValueError("refusing to save an empty table of addition times")
    seen: dict = {}
    for name, entry in samples.items():
        for movie in entry["movies"]:
            if movie in seen:
                raise ValueError(f"movie {movie} is in two samples ({seen[movie]} and {name})")
            seen[movie] = name
    path = additions_path(condition_root)
    payload = _read_file(path) or {}
    payload.update({"version": 1, "written_by": "smtracker.kinetics_designs.save_additions",
                    "samples": {str(k): {"movies": [str(m) for m in v["movies"]], "time": str(v["time"]).strip()}
                                for k, v in samples.items()}})
    return _write_file(path, payload)


def save_culture_times(condition_root, times: dict) -> Path:
    """Write the condition's addition time per replicate (Run only), beside the per-sample table, which is kept.
    ``times`` ``{replicate: typed time}``; blanks are not saved; refuses an empty set."""
    clean = {str(int(k)): str(v).strip() for k, v in times.items() if str(v).strip()}
    if not clean:
        raise ValueError("refusing to save an empty set of addition times")
    path = additions_path(condition_root)
    payload = _read_file(path) or {}
    payload.setdefault("version", 1)
    payload["written_by"] = "smtracker.kinetics_designs"
    payload["culture"] = {"replicates": clean}
    return _write_file(path, payload)


def same_time_fill(same: str, current: dict, ticked, *, only_blank: bool = False) -> dict:
    """What the "Same time for every replicate" field puts in the replicate fields: ``current`` ``{condition:
    {replicate: text}}`` becomes the same dict with ``same`` (stripped) in every replicate of every condition in
    ``ticked``; ``only_blank`` leaves a field that already holds a time (used when a condition is ticked after the
    time was typed). A blank ``same`` changes nothing. Conditions not ticked are returned untouched."""
    same = str(same).strip()
    ticked = set(ticked)
    out: dict = {}
    for key, days in current.items():
        if not same or key not in ticked:
            out[key] = dict(days)
        else:
            out[key] = {d: (text if only_blank and str(text).strip() else same) for d, text in days.items()}
    return out


def seed_culture_times(last: dict | None, saved: dict | None, condition: str) -> dict:
    """The replicate fields' starting text for ``condition``: the last Run's (``last`` ``{condition: {replicate:
    text}}``), else what was saved with the project (``saved``, same shape), else nothing."""
    for source in (last, saved):
        got = (source or {}).get(condition)
        if got:
            return {int(d): str(v) for d, v in got.items()}
    return {}


def samples_from_rows(rows: dict) -> dict:
    """The samples of a table of movies: ``rows`` ``{movie: (sample name, typed time)}``; a blank sample name is the
    movie's own (one movie, one sample, the default). Movies sharing a name are one sample and must agree on the time
    (a blank is filled from the others). RAISES ValueError naming a sample whose movies disagree."""
    out: dict = {}
    for movie, (name, time) in rows.items():
        key = str(name).strip() or str(movie)
        time = str(time).strip()
        entry = out.setdefault(key, {"movies": [], "time": ""})
        entry["movies"].append(str(movie))
        if time:
            if entry["time"] and entry["time"] != time:
                raise ValueError(f"sample {key}: its movies carry two different addition times "
                                 f"({entry['time']} and {time})")
            entry["time"] = time
    return out


def sample_problem(samples_by_condition: dict, movies_by_condition: dict, *, parse) -> str:
    """Why Run must wait under the per-sample clock, or "". ``samples_by_condition`` ``{condition: {sample:
    {"movies", "time"}}}`` for the INCLUDED conditions, ``movies_by_condition`` ``{condition: [movies]}``, ``parse``
    `induction_view.parse_addition` (a date stood in for the check). Every movie of a condition must be in a sample with
    a time that reads; a missing one is named."""
    import datetime as _dt

    if not samples_by_condition:
        return "Include at least one condition."
    for condition, samples in samples_by_condition.items():
        if not movies_by_condition.get(condition):
            return f"No movies are known for {condition}."
        in_sample = {m: name for name, e in samples.items() for m in e["movies"]}
        for movie in movies_by_condition.get(condition, []):
            if movie not in in_sample:
                return f"{condition}: movie {movie} has no addition time."
        for name, entry in samples.items():
            if not str(entry["time"]).strip():
                return f"{condition}: type the time of the addition for {name}."
            if parse(entry["time"], _dt.date(2000, 1, 1)) is None:
                return f"{condition}, {name}: write the time as 10:30 or 2026-10-23 10:30."
    return ""


def parse_paste(text: str, movies: list, *, aliases: dict | None = None) -> tuple:
    """A paste from Excel (TAB-separated lines) as ``({movie: time}, [problems])``. Each line is ``movie<TAB>time``
    (the movie by its analysis name, or by ``aliases`` ``{other name: analysis name}``, as the file was named) or just
    ``time`` (taken in the order of ``movies``). A header line whose first cell is a name nobody has is a problem,
    never skipped silently; blank lines are ignored."""
    aliases = dict(aliases or {})
    known = {str(m): str(m) for m in movies}
    known.update({str(k): str(v) for k, v in aliases.items()})
    lines = [ln for ln in text.replace("\r", "").split("\n") if ln.strip()]
    out: dict = {}
    problems: list = []
    order = [str(m) for m in movies]
    for i, line in enumerate(lines):
        cells = [c.strip() for c in line.split("\t")]
        if len(cells) >= 2 and cells[0] in known:
            out[known[cells[0]]] = cells[1]
        elif len(cells) >= 2:
            problems.append(f"line {i + 1}: {cells[0]!r} is not one of the movies")
        elif i < len(order):
            out[order[i]] = cells[0]
        else:
            problems.append(f"line {i + 1}: more times than movies ({len(order)})")
    return out, problems


# --------------------------------------------------------------------------- #
# The review's gate numbers, in code with their source
# --------------------------------------------------------------------------- #

#: Where the numbers below come from; nothing else is invented here.
GATE_SOURCE = ("methods_time_resolved_smt_merged.md, section 1, 'The gate at real mNG track lengths' "
               "(SEED 20261107; D start 0.5 um2/s; a change in D, not in the bound share)")
#: A segment of fewer steps is not tested: the review's minimum segment.
GATE_MIN_SEGMENT = 4
_FOUR = "four"
#: ``{steps: (candidate change points, crit95, rho for an increase, rho for a decrease)}``; ``rho`` the fold change in D
#: a within-track test detects. 6 steps: no candidate, the test is not defined.
GATE_TABLE = {
    6: (0, None, None, None),
    10: (3, 5.46, 7.91, 16.67),
    12: (5, 6.01, 6.89, 12.29),
    16: (9, 6.86, 5.61, 8.62),
    20: (13, 7.26, 4.72, 6.29),
}


def gate_row(n_steps: float):
    """The review's row for a track of ``n_steps`` steps: the largest tabulated length not above it, as ``(steps,
    candidates, crit95, rho_increase, rho_decrease)``; None below the shortest row that defines the test (10 steps)."""
    rows = [n for n in GATE_TABLE if n <= n_steps and GATE_TABLE[n][0] > 0]
    if not rows:
        return None
    n = max(rows)
    return (n,) + GATE_TABLE[n]


def _fold(rho: float) -> str:
    """A ratio spelled out ("fivefold", "7.9 times"), never a digit before "-fold" (`chart_text.fold_words`)."""
    from .chart_text import fold_words

    return fold_words(rho, 1)


def within_track_text(steps_per_track, change: str = "none", d_before=None, d_end=None) -> str:
    """The within-track gate as a statement, never a test. ``steps_per_track``: the steps of every track; ``change`` the
    fitted change (`induction.CHANGES`) with the populations' D before and where the data end."""
    n = np.asarray(list(steps_per_track), float)
    if not n.size:
        return "Within single tracks: no tracks, so nothing to say."
    median = float(np.median(n))
    need = 2 * GATE_MIN_SEGMENT
    share_long = 100.0 * float(np.mean(n >= 10))
    median_text = f"{median:g}"
    if median < need:
        head = (f"Median track {median_text} steps: with at least {_FOUR} steps on each side no change "
                f"point can be placed (it needs {need}), so a change within single tracks is not testable here.")
    elif gate_row(median) is None:
        head = (f"Median track {median_text} steps: a change point can be placed, but the "
                f"detectable change is tabulated only from 10 steps, so none is quoted here.")
    else:
        row = gate_row(median)
        head = (f"Median track {median_text} steps: a within-track test is defined; at {row[0]} steps it detects D "
                f"rising {_fold(row[3])} or more, or falling {_fold(row[4])} or more (D start 0.5 um2/s).")
    tail = ""
    row10 = (10,) + GATE_TABLE[10] if share_long > 0 else None     # the shortest length the review defines the test at
    if median < 10 and row10 is not None:
        # tracks long enough exist: the review's numbers for them, in the direction of the fitted change
        direction = _direction(change, d_before, d_end)
        if direction == "shares":
            tail = (f" {share_long:.0f} % of tracks have 10 or more steps; the fitted change is in the shares, a "
                    f"different case whose numbers are not tabulated here.")
        elif direction in ("decrease", "increase"):
            rho = row10[4] if direction == "decrease" else row10[3]
            verb = "fall" if direction == "decrease" else "rise"
            tail = (f" {share_long:.0f} % of tracks have 10 or more steps; for those a within-track test would need "
                    f"D to {verb} {_fold(rho)} (at {row10[0]} steps).")
        else:
            tail = (f" {share_long:.0f} % of tracks have 10 or more steps; for those a within-track test would need "
                    f"D to rise {_fold(row10[3])} or fall {_fold(row10[4])} (at {row10[0]} steps).")
    elif median >= need and gate_row(median) is not None:
        direction = _direction(change, d_before, d_end)
        row = gate_row(median)
        if direction in ("decrease", "increase"):
            fitted = _fitted_fold(d_before, d_end)
            rho = row[4] if direction == "decrease" else row[3]
            moved = "fell" if direction == "decrease" else "rose"
            tail = (f" The fitted D {moved} about {_fold(fitted)}; a within-track test needs "
                    f"{_fold(rho)}.")
        elif direction == "shares":
            tail = " The fitted change is in the shares, a different case not tabulated here."
    return head + tail + " No within-track test is built."


def _fitted_fold(d_before, d_end) -> float:
    ratios = [max(a, b) / min(a, b) for a, b in zip(d_before, d_end) if a > 0 and b > 0]
    return max(ratios) if ratios else 1.0


def _direction(change: str, d_before, d_end) -> str:
    """"decrease" / "increase" (the population whose D moved most), "shares" (D did not move), or "" (nothing fitted)."""
    if change in ("D", "both") and d_before is not None and d_end is not None:
        ratios = [(b / a, a, b) for a, b in zip(d_before, d_end) if a > 0 and b > 0]
        if ratios:
            r = max(ratios, key=lambda x: abs(math.log(x[0])))[0]
            return "decrease" if r < 1 else "increase"
    if change == "fractions":
        return "shares"
    return ""


# --------------------------------------------------------------------------- #
# The acquisition per time window
# --------------------------------------------------------------------------- #

#: Windows after the addition (equal in width); the baseline before it is one more window.
QC_WINDOWS = 4
#: A trend is NAMED when a window's value differs from the first window's by this fraction (display only; no test).
QC_TREND_FRACTION = 0.10
#: A window holding fewer tracks is not compared in a trend.
QC_MIN_TRACKS = 20


def when_words(seconds: float) -> str:
    """A time as the card says it: seconds under two minutes, else minutes."""
    s = float(seconds)
    return f"{s:.0f} s" if abs(s) < 120.0 else f"{s / 60.0:.1f} min"


def window_edges(clock, n_after: int = QC_WINDOWS) -> list:
    """Edges (seconds) of the windows the acquisition is read in: a baseline window up to the addition (0) where the
    clock holds both, then ``n_after`` windows of equal width to the last step; all equal where it holds one side."""
    clock = np.asarray(clock, float)
    clock = clock[np.isfinite(clock)]
    if not clock.size:
        raise ValueError("no clock to cut into windows")
    t0, t1 = float(clock.min()), float(clock.max())
    if t1 <= t0:
        return [t0, t0 + 1.0]
    if t0 < 0.0 < t1:
        return [t0, 0.0] + [float(v) for v in np.linspace(0.0, t1, n_after + 1)[1:]]
    return [float(v) for v in np.linspace(t0, t1, n_after + 1)]


def window_qc(steps, edges) -> list:
    """The acquisition per window, ``[{"lo", "hi", "n_tracks", "mean_length", "median_sigma_nm"}]``. A track belongs to
    the window of its FIRST step; its length is its steps plus one, in localisations, counted over the whole table. The
    per-spot error is the median over the window's steps of the mean of the two localisations' errors, in nm; None
    where the tracks carry none. RAISES ``ValueError`` without a ``uid`` column or a clock."""
    from .induction import CLOCK
    from .observables import LOC_SIGMA_STEP_COLUMNS

    for column in ("uid", CLOCK):
        if column not in steps.columns:
            raise ValueError(f"no '{column}' column to read the acquisition per window by")
    edges = [float(e) for e in edges]
    clock = steps[CLOCK].to_numpy(float)
    uid = steps["uid"].to_numpy()
    import pandas as pd

    frame = pd.DataFrame({"uid": uid, "t": clock})
    length = frame.groupby("uid", sort=False)["t"].size() + 1.0
    first = frame.groupby("uid", sort=False)["t"].min()
    have_sigma = all(c in steps.columns for c in LOC_SIGMA_STEP_COLUMNS)
    sigma = None
    if have_sigma:
        sigma = np.nanmean(np.vstack([steps[c].to_numpy(float) for c in LOC_SIGMA_STEP_COLUMNS]), axis=0) * 1000.0
    out = []
    last = len(edges) - 2
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        in_w = (first >= lo) & ((first <= hi) if i == last else (first < hi))
        tracks = first.index[in_w.to_numpy()]
        step_in = (clock >= lo) & ((clock <= hi) if i == last else (clock < hi))
        median_sigma = None
        if sigma is not None:
            values = sigma[step_in & np.isfinite(sigma)]
            median_sigma = float(np.median(values)) if values.size else None
        out.append({"lo": lo, "hi": hi, "n_tracks": int(len(tracks)),
                    "mean_length": float(length.loc[tracks].mean()) if len(tracks) else None,
                    "median_sigma_nm": median_sigma})
    return out


def qc_trend(rows: list, key: str):
    """``(first value, last value, relative change)`` between the first and the last window holding at least
    `QC_MIN_TRACKS` tracks and a value, when the change is at least `QC_TREND_FRACTION`; else None."""
    usable = [r for r in rows if r["n_tracks"] >= QC_MIN_TRACKS and r.get(key) is not None]
    if len(usable) < 2:
        return None
    a, b = usable[0][key], usable[-1][key]
    if a == 0:
        return None
    change = (b - a) / a
    return (a, b, change) if abs(change) >= QC_TREND_FRACTION else None


def _label(row: dict, unit_s: float, unit: str, baseline_first: bool, i: int) -> str:
    if row["hi"] <= 0.0:
        return "before the addition"
    return f"{row['lo'] / unit_s:.1f} to {row['hi'] / unit_s:.1f} {unit}".replace(".0 ", " ")


def _qc_lines(rows: list) -> list:
    span = max(abs(rows[0]["lo"]), abs(rows[-1]["hi"]))
    unit_s, unit = (60.0, "min") if span >= 300.0 else (1.0, "s")
    labels = [_label(r, unit_s, unit, True, i) for i, r in enumerate(rows)]

    def join(values, fmt):
        return " · ".join(f"{lab}: {fmt(v)}" for lab, v in zip(labels, values))

    lines = [
        "Acquisition per time window, mean track length in localisations: "
        + join([r["mean_length"] for r in rows], lambda v: "no tracks" if v is None else f"{v:.1f}") + ".",
    ]
    if any(r["median_sigma_nm"] is not None for r in rows):
        lines.append("Median per-spot error (nm): "
                     + join([r["median_sigma_nm"] for r in rows], lambda v: "none" if v is None else f"{v:.1f}") + ".")
    else:
        lines.append("Median per-spot error: these tracks carry none.")
    lines.append("Tracks per window: " + join([r["n_tracks"] for r in rows], lambda v: f"{v:,}") + ".")
    named = []
    t = qc_trend(rows, "mean_length")
    if t is not None:
        named.append(f"mean track length {'falls' if t[2] < 0 else 'rises'} from {t[0]:.1f} to {t[1]:.1f} "
                     f"localisations ({100 * t[2]:+.0f} %)")
    t = qc_trend(rows, "median_sigma_nm")
    if t is not None:
        named.append(f"the median per-spot error {'falls' if t[2] < 0 else 'rises'} from {t[0]:.1f} to {t[1]:.1f} nm "
                     f"({100 * t[2]:+.0f} %)")
    if named:
        lines.append("A trend over time: " + "; ".join(named) + ". This is noted but does not stop the analysis: a real effect "
                     "can shorten tracks, and a rising error is read by the fit as a rising D. The "
                     "censoring-corrected refit is not yet made.")
    else:
        lines.append("No trend over the windows in track length or per-spot error. The censoring-corrected refit "
                     "(a falling length raises the apparent off-rate) is not yet made.")
    return lines


def _movies_by_state(steps) -> dict:
    """Per movie: its start on the clock (first frame), its last step, and its replicate."""
    from .induction import CLOCK, _movie_table

    names, reps, starts = _movie_table(steps, "day")
    movie = steps["movie"].astype(str).to_numpy()
    t = steps[CLOCK].to_numpy(float)
    last = {m: float(t[movie == m].max()) for m in names}
    return {str(m): (float(s), last[m], str(r)) for m, r, s in zip(names, reps, starts)}


def design_lines(steps, *, design: str, tau_s: float, trigger_frame=None, change: str = "none", d_before=None,
                 d_end=None) -> list:
    """The card's design lines (`module docstring`), in order: the window observed, the baseline, the acquisition per
    window (always), and the within-track statement; for design "microscope" the note on repeated imaging."""
    from .induction import CLOCK

    clock = steps[CLOCK].to_numpy(float)
    if design not in DESIGNS:
        raise ValueError(f"unknown design {design!r}")
    lines: list = [f"Design: {DESIGN_WORDS[design]}. The replicate, the culture, is the independent unit; its "
                   f"movies are time points of one response, not replicates."]
    # THE WINDOW OBSERVED, AND THE BASELINE
    if trigger_frame is not None:
        before = int(np.sum(clock < 0.0))
        lines.append(f"Window observed: the addition falls on frame {int(trigger_frame)} of every movie; the response "
                     f"is followed from it to {when_words(float(clock.max()))} after it.")
        lines.append(f"Baseline: the frames before the addition inside each movie ({before:,} steps)."
                     if before else "Baseline: none, the addition is on a movie's first frames, so the values before "
                     "it come from the model, not from measurements.")
    else:
        info = _movies_by_state(steps)
        after = sorted(s for s, _l, _r in info.values() if s >= 0.0)
        base = {m: v for m, v in info.items() if v[1] < 0.0}
        last = float(clock.max())
        if after:
            first = after[0]
            lines.append(f"Window observed: first movie {when_words(first)} after the addition, last "
                         f"{when_words(last)}; "
                         + ("the response is followed from the addition on."
                            if first <= 2.0 * tau_s else "the response before the first movie is not observed."))
        else:
            lines.append(f"Window observed: no movie begins after the addition; the last step is "
                         f"{when_words(last)} after it.")
        if base:
            per_rep: dict = {}
            for _m, (_s, _l, rep) in base.items():
                per_rep[rep] = per_rep.get(rep, 0) + 1
            replicates = sorted({r for _s, _l, r in info.values()})
            counts = [per_rep.get(r, 0) for r in replicates]
            word = "movie" if len(base) == 1 else "movies"
            tail = {"sample": " (movies recorded before their own sample's addition, untreated or mock-added "
                              "samples)", "culture": "", "microscope": " (the same chamber, before the addition)"}[design]
            lines.append(f"Baseline: {len(base)} {word} before the addition{tail}; per replicate "
                         f"{', '.join(map(str, counts))}."
                         + (" Three or more per culture are advised." if min(counts) < 3 else ""))
        else:
            lines.append("Baseline: no movie was recorded before the addition, so the values before it come from "
                         "the model, not from measurements.")
    if design == "microscope":
        lines.append("The same field may have been imaged repeatedly, so bleaching can shape the windows below. "
                     "Following the same cells across movies (a per-cell onset) is not built yet.")
    elif design == "sample":
        lines.append("Each sample's own addition time sets its movies' clock; the permutation test deals the addition "
                     "times among the samples of a replicate.")
    # THE ACQUISITION PER WINDOW, always
    lines += _qc_lines(window_qc(steps, window_edges(clock)))
    # THE WITHIN-TRACK GATE
    per_track = steps.groupby("uid", sort=False).size().to_numpy()
    lines.append(within_track_text(per_track, change, d_before, d_end))
    return lines
