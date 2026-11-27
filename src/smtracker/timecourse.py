"""How jump distance and diffusion change over experiment time, at two levels.

TWO QUESTIONS, TWO UNITS, AND THEY ARE NOT THE SAME ANALYSIS.

  POPULATION   Does the whole population shift over the course of an acquisition or an
               induction? The unit is the MOVIE, the cell or the replicate day -- never
               the time bin. Binning steps by time and testing across bins is
               pseudoreplication with a clock on it: the bins share cells, tracks and a
               culture. Each unit contributes ONE trajectory and the spread across units
               is the error bar.
  SINGLE TRACK Does one long-lived molecule change behaviour within its own lifetime?
               Only worth asking for labels that survive long enough, so the minimum
               track length is REQUIRED and has no default -- it decides what the
               analysis is looking at.

THE POPULATION LEVEL HAS A MEASURED ARTEFACT RATE, and it is small. Within unperturbed
20-second acquisitions of a lab data set (about 28 movies per condition) the slow
fraction drifted by under a percentage point in each condition, and every 95 % interval
contains zero. Two mechanisms would have produced a spurious rise --
bleaching removes fast molecules preferentially, and detection recall is worse for fast
molecules (26.6 % missed against 23.9 %) -- so the null is not obviously zero and had to
be measured. Between-MOVIE spread is an order of magnitude larger than the trend, so a
single field followed over time is uninformative however clean the perturbation.

BIN WITHIN EACH UNIT'S OWN SPAN, NEVER ON AN ABSOLUTE AXIS. Acquisitions here are 1001
AND 1501 frames, because the crop is defined by a physical state rather than a frame
count. Absolute bins put only the LONG movies in the late bins: the first version of the
measurement above read about twice as large from a biased subset of 18 movies instead of
28, and relative bins halved it.

D IS FITTED ONCE AND ONLY THE POSTERIOR IS BINNED. Refitting per bin lets the two states
move to absorb the very drift being measured, and a late bin rarely has the steps to fit
two populations at all.

THE SINGLE-TRACK LEVEL NEEDS ITS OWN NULL, AND IT IS FREE. A windowed diffusion
coefficient from `w` steps has a relative standard error of roughly 1/sqrt(w) even when
D is perfectly constant -- 32 per cent at w = 10 -- so a trace that wanders is the
expected picture, not a finding. `shuffle_null` re-computes the same trace with the
track's own steps in random order: same steps, same window, no time structure. If the
observed trace is not more structured than that, there is nothing to report.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "TimeCourseError",
    "population_over_time",
    "population_drift",
    "track_timeline",
    "shuffle_null",
]


class TimeCourseError(RuntimeError):
    """Raised when a time course cannot be formed honestly."""


def _relative_time(frame: pd.DataFrame, within: str) -> np.ndarray:
    """Position within each unit's own frame span, in [0, 1]."""
    f = frame.groupby(within, observed=True)["frame"].transform(
        lambda s: (s - s.min()) / max(float(s.max() - s.min()), 1.0))
    return f.to_numpy(float)


def population_over_time(steps: pd.DataFrame, *, weights: str = "w_slow",
                         n_bins: int = 5, unit: str = "movie",
                         min_steps_per_bin: int = 50,
                         absolute: bool = False) -> pd.DataFrame:
    """One trajectory per unit: a soft-weighted quantity against time.

    ``steps`` carries one row per displacement with ``frame``, the unit column, and a
    per-step weight in [0, 1] -- normally the slow-state posterior from a fit made ONCE
    on the whole condition. ``jump_um`` is summarised too where present, because a jump
    distance needs no state model and is the more direct observable.

    A bin with fewer than ``min_steps_per_bin`` steps is returned as NaN, never as zero:
    an empty bin is missing, not a fraction of zero.

    ``absolute=True`` bins on the raw frame axis instead. It is offered because a real
    induction time course may have every field on one clock, and refused silently by
    nothing -- but on acquisitions of unequal length it selects which units reach the
    late bins, so the default is relative.
    """
    for column in ("frame", unit):
        if column not in steps.columns:
            raise TimeCourseError(f"no '{column}' column")
    if weights not in steps.columns:
        raise TimeCourseError(
            f"no '{weights}' column: fit the states once on the whole condition and pass "
            f"the per-step posterior instead of refitting inside each time bin")
    if n_bins < 2:
        raise TimeCourseError("n_bins must be at least 2 for a trajectory")

    frame = steps.copy()
    if absolute:
        f = frame["frame"].to_numpy(float)
        span = (f - f.min()) / max(float(f.max() - f.min()), 1.0)
    else:
        span = _relative_time(frame, unit)
    frame["_bin"] = np.clip((span * n_bins).astype(int), 0, n_bins - 1)

    rows = []
    for key, part in frame.groupby(unit, observed=True):
        for b in range(n_bins):
            block = part[part["_bin"] == b]
            enough = len(block) >= min_steps_per_bin
            row = {unit: key, "bin": b + 1,
                   "t_relative": (b + 0.5) / n_bins,
                   "n_steps": int(len(block)),
                   "ok": bool(enough),
                   "value": float(block[weights].mean()) if enough else np.nan}
            if "jump_um" in block.columns and enough:
                row["median_jump_um"] = float(block["jump_um"].median())
                row["mean_sq_jump_um2"] = float((block["jump_um"] ** 2).mean())
            for carry in ("condition", "day", "protein"):
                if carry in part.columns:
                    row[carry] = part[carry].iloc[0]
            rows.append(row)

    out = pd.DataFrame(rows)
    out.attrs.update({"unit": unit, "n_bins": n_bins, "weights": weights,
                      "absolute": bool(absolute),
                      "min_steps_per_bin": int(min_steps_per_bin)})
    return out


def population_drift(trajectories: pd.DataFrame, *, unit: str | None = None,
                     column: str = "value") -> dict:
    """Last bin minus first, per unit, with an interval formed over UNITS.

    The interval comes from the spread between units, never from the number of steps.
    A unit missing either end bin is excluded and counted, because a drift needs both
    ends of the same unit; comparing one unit's first bin with another's last is not a
    drift.
    """
    unit = unit or trajectories.attrs.get("unit", "movie")
    if unit not in trajectories.columns:
        raise TimeCourseError(f"no '{unit}' column")
    n_bins = int(trajectories["bin"].max())

    deltas, kept, dropped = [], [], []
    for key, part in trajectories.groupby(unit, observed=True):
        first = part[part["bin"] == 1][column]
        last = part[part["bin"] == n_bins][column]
        if first.empty or last.empty or not np.isfinite([first.iloc[0], last.iloc[0]]).all():
            dropped.append(key)
            continue
        deltas.append(float(last.iloc[0] - first.iloc[0]))
        kept.append(key)

    if len(deltas) < 2:
        raise TimeCourseError(
            f"only {len(deltas)} unit(s) have both end bins; a drift interval needs at "
            f"least two units, and {len(dropped)} were excluded for a missing end bin")

    d = np.asarray(deltas, dtype=float)
    sem = float(d.std(ddof=1) / np.sqrt(len(d)))
    mean = float(d.mean())
    return {
        "drift": mean,
        "sem": sem,
        "ci95": (mean - 1.96 * sem, mean + 1.96 * sem),
        "n_units": len(d),
        "n_excluded": len(dropped),
        "excluded": dropped,
        "unit": unit,
        "per_unit": dict(zip(kept, deltas)),
        "caveat": (
            "the interval is the spread between units, not between time bins: bins share "
            "cells, tracks and a culture. Measured on unperturbed acquisitions the "
            "artefact rate is under 1 pp with a roughly +-4 pp interval, so a time course "
            "must beat that and not zero"),
    }


def track_timeline(steps: pd.DataFrame, *, min_localizations: int, window: int,
                   tau_s: float, uid: str = "uid", step: int = 1) -> pd.DataFrame:
    """A sliding-window apparent D and jump distance along each long track.

    ``min_localizations`` HAS NO DEFAULT. It decides what the analysis is about: a
    long-lived organic dye supports a window that a fluorescent protein does not, and a
    threshold chosen by the function would be a threshold nobody owns.

    THE WINDOW IS THE RESOLUTION AND THE NOISE FLOOR AT ONCE. An apparent D from `w`
    squared displacements has a relative standard error near 1/sqrt(w) with D perfectly
    constant, which is reported per row as ``rel_se`` so a wander can be read against it.
    Compare any trace with :func:`shuffle_null` before calling a change real.
    """
    if min_localizations < 3:
        raise TimeCourseError("min_localizations must be at least 3 to give two steps")
    if window < 3:
        raise TimeCourseError(
            "a window below 3 steps has a relative standard error above 58 per cent; "
            "the trace would be noise with a shape")
    if "sq_disp_um2" not in steps.columns:
        raise TimeCourseError("no 'sq_disp_um2' column")
    if tau_s <= 0:
        raise TimeCourseError("tau_s must be positive and must be read, not guessed")

    rows = []
    for key, part in steps.groupby(uid, observed=True):
        part = part.sort_values("frame") if "frame" in part.columns else part
        # A track of n localisations has n-1 steps.
        if len(part) + 1 < min_localizations or len(part) < window:
            continue
        r2 = part["sq_disp_um2"].to_numpy(float)
        frames = (part["frame"].to_numpy(float) if "frame" in part.columns
                  else np.arange(len(part), dtype=float))
        for start in range(0, len(r2) - window + 1, step):
            block = r2[start:start + window]
            d_app = float(block.mean() / (4.0 * tau_s))
            rows.append({
                uid: key,
                "window_start_frame": float(frames[start]),
                "window_mid_frame": float(frames[start:start + window].mean()),
                "n_steps": window,
                "appd_um2_per_s": d_app,
                "median_jump_um": float(np.median(np.sqrt(block))),
                # sqrt(2/w) is the relative s.e. of a mean of w exponential variates;
                # 1/sqrt(w) is the usual shorthand and is the conservative one here.
                "rel_se": float(1.0 / np.sqrt(window)),
            })
    if not rows:
        raise TimeCourseError(
            f"no track reached {min_localizations} localisations with a {window}-step "
            f"window; lower one of them or accept that this label does not support the "
            f"analysis")
    out = pd.DataFrame(rows)
    out.attrs.update({"min_localizations": int(min_localizations),
                      "window": int(window), "tau_s": float(tau_s),
                      "n_tracks": int(out[uid].nunique())})
    return out


def shuffle_null(steps: pd.DataFrame, *, min_localizations: int, window: int,
                 tau_s: float, uid: str = "uid", n_shuffles: int = 20,
                 seed: int = 0) -> dict:
    """How much a windowed trace wanders with the SAME steps in random order.

    This is the chance rate for a single-track time series and it costs nothing: shuffling
    a track's own displacements destroys any time structure while preserving the track's
    length, its diffusion coefficient and its noise exactly. A trace that is no more
    structured than its own shuffle is not evidence of a change in behaviour.

    The statistic is the standard deviation of the windowed apparent D ALONG a track,
    averaged over tracks -- how much the trace moves. Observed against shuffled.
    """
    rng = np.random.default_rng(seed)

    def _spread(frame):
        line = track_timeline(frame, min_localizations=min_localizations,
                              window=window, tau_s=tau_s, uid=uid)
        per_track = line.groupby(uid, observed=True)["appd_um2_per_s"].std(ddof=1)
        return float(np.nanmean(per_track.to_numpy(float))), line

    observed, line = _spread(steps)

    chance = []
    for _ in range(n_shuffles):
        shuffled = steps.copy()
        # WITHIN each track, so the track keeps its own D and its own length.
        shuffled["sq_disp_um2"] = (
            shuffled.groupby(uid, observed=True)["sq_disp_um2"]
            .transform(lambda s: rng.permutation(s.to_numpy())))
        chance.append(_spread(shuffled)[0])

    chance = np.asarray(chance, dtype=float)
    return {
        "observed_spread": observed,
        "chance_spread": float(chance.mean()),
        "chance_sd": float(chance.std(ddof=1)) if len(chance) > 1 else float("nan"),
        "excess": observed - float(chance.mean()),
        "n_tracks": int(line[uid].nunique()),
        "window": int(window),
        "min_localizations": int(min_localizations),
        "n_shuffles": int(n_shuffles),
        "caveat": (
            "a windowed D wanders even when D is constant (relative standard error "
            "about 1/sqrt(window), 32 per cent at 10 steps). Report the excess over this "
            "shuffle, never the observed wander alone"),
    }
