"""Kinetics from a TRIGGERED event: a response curve on a common clock, and a rate.

WHAT THIS IS FOR. The experimenter does something at a known moment -- adds a drug,
switches carbon source, photo-activates -- and the population responds. The readout is
how fast, expressed as a rate constant with an interval, rather than "it went down".

THE CLOCK IS THE WHOLE DESIGN, AND THERE ARE TWO OF THEM.

  TRIGGER CLOCK  x = frames since the trigger. Every molecule is on ONE clock because one
                 external event started the response. This is the only alignment from
                 which a triggered rate constant can be estimated, and it is what
                 `response_curve` builds.
  TRACK CLOCK    x = frames since a track's own first localisation. This is QUALITY
                 CONTROL, not kinetics -- see `lifetime_qc`. It answers whether a single
                 molecule changes over its own lifetime, and how much of any apparent
                 change is survivorship.

WHY THE TRACK CLOCK CANNOT MEASURE A TRIGGERED DECAY. Tracks begin at random times after
the trigger, so re-zeroing each to its own start averages molecules that are at DIFFERENT
points of the same decay. The result is not a wrong rate, it is a flattened curve with no
rate in it: the alignment destroys exactly the structure being measured. `response_curve`
therefore takes no alignment argument at all, and `lifetime_qc` is named for what it is.

NO MINIMUM TRACK LENGTH HERE, AND THE REASON IS STRONGER THAN THE USUAL ONE. A track
contributes one observation at the time it was seen; a five-localisation track is a noisy
point and a thousand of them make a clean curve, so a within-track trajectory is not
needed. MEASURED on four lab groups (`min_length_sweep.csv`), a 15-localisation minimum
keeps a few per cent of tracks, shifts the median per-track D by about a quarter and
inflates the slow fraction by several points -- about twice the cell-bootstrap
half-width. Worse, the selection is CONFOUNDED WITH THE SIGNAL: if D falls after the
trigger then slow molecules stay in focus longer, tracks get longer as the experiment
proceeds, and a fixed minimum admits a larger share of tracks late than early. That can
manufacture a decay from nothing. `min_localizations` exists so a user can impose one
deliberately, defaults to None (no filter) and REPORTS what it kept.

THE UNIT IS THE MOVIE OR THE CELL, NEVER THE TIME BIN. Bins share cells, tracks and a
culture, so an interval formed across bins is pseudoreplication with a clock on it. Every
interval here comes from resampling whole UNITS, and the model is chosen on units held out.
Until 13 Oct 2026 the fit was made per unit and the parameters averaged; SynR showed that a
unit holds too few tracks per bin to pin down an exponential and the average was no estimate
(`fit_response`). Now ONE fit is made with every unit weighing the same -- for binned units,
the fit to the curve averaged over units -- and the units are resampled around it.

THE NULL IS NOT ZERO, AND IT IS ALREADY MEASURED. Within unperturbed 20-second
acquisitions of a lab data set the slow fraction drifted by a few tenths of a percentage point at most, with
intervals that contain zero. Bleaching removes fast molecules preferentially and
detection recall is worse for fast molecules, so a rise was plausible a priori and had to
be measured. `UNTRIGGERED_DRIFT_PP` carries it and `fit_response` reports the fitted
change beside it: a response must beat that band, not zero. It is a LAB REFERENCE measured on
another data set (`UNTRIGGERED_DRIFT_SOURCE`), carried with every fit and named wherever it is
printed, because its conditions are that data set's and not the project's.

THE ACQUISITION LENGTH BOUNDS WHAT IS RESOLVABLE. At tau = 20.47 ms a 1001-frame movie is
20.5 s and a 2001-frame movie 41 s. A half-life much beyond a third of the window cannot
be separated from a linear trend, and one much below the bin width is over before the
first bin ends. `resolvable_halflife_range` returns the window and `fit_response` flags a
constant that falls outside it rather than returning a number the design could not
support.

MODEL SELECTION REPORTS THE ALTERNATIVE. Linear, single-exponential and
exponential-to-plateau are all fitted; BIC recommends and the table is returned, because
"it decayed exponentially" is a claim about the model as much as about the biology.

BINNING IS A CHOICE AND IT IS OFFERED AS ONE. Binning averages away noise and costs time
resolution, and the bin width has no right value -- so `binned=False` places every TRACK
at its own moment and fits the scatter directly, which is what the data actually is. The
bin width is given in SECONDS rather than as a count, because seconds are portable between
acquisitions of different length and a count is not: ten bins is 2.05 s on a 1001-frame
movie and 4.1 s on a 2001-frame one, so the same setting would mean two different
resolutions in one dataset.

UNBINNED PLACES A TRACK AT ITS OWN MIDPOINT, and that is exact only while a track is short
against the response. A track spans time, so a 24-frame track crossing a fast decay is
smeared onto one point. `response_curve` reports `track_span_s` and `fit_response` warns
when the median track approaches the fitted half-life on an UNBINNED curve -- the condition under which the
unbinned estimator stops being the more honest one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from .canonical import without_attrs

__all__ = [
    "KineticsError",
    "NO_SWITCHING_REFUSAL",
    "require_switching",
    "UNTRIGGERED_DRIFT_PP",
    "UNTRIGGERED_DRIFT_SOURCE",
    "RESPONSE_MODELS",
    "response_curve",
    "fit_response",
    "lifetime_qc",
    "resolvable_halflife_range",
    "detect_onsets",
    "align_to_onset",
    "onset_alignment_null",
    "alignment_gain",
]


class KineticsError(RuntimeError):
    """Raised when a kinetic quantity cannot be formed honestly."""


#: WHY A CONDITION WHOSE TRACKS NEVER SWITCH STATE HAS NO KINETICS (ruled 4 Nov 2026: "Kinetics keeps refusing" it,
#: while the HMM maps draw it, `hmm.HMMFit.no_switching`). The triggered curve is the slow state's share of the steps
#: seen at each moment; when no track ever leaves its state that share only says which tracks happened to be seen
#: when, so a change in it is a change in who was imaged, not molecules switching.
NO_SWITCHING_REFUSAL = (
    "no switching within tracks: the HMM fitted to this condition never leaves a state inside a track, so there is "
    "no switching to time and no kinetics to read. The HMM maps still draw this condition, one state per track.")


def require_switching(hmm_fit) -> None:
    """RAISES `KineticsError` (`NO_SWITCHING_REFUSAL`) when ``hmm_fit`` saw no switching within a track
    (`hmm.HMMFit.no_switching`): the one place the Kinetics refusal is decided. Until 4 Nov 2026 `hmm.decode_states`
    refused such a fit for every reader; it decodes it since (each track in one state, for the HMM maps), and Kinetics
    refuses it here instead, by ruling."""
    if bool(getattr(hmm_fit, "no_switching", False)):
        raise KineticsError(NO_SWITCHING_REFUSAL)


#: Measured drift of the slow fraction over unperturbed acquisitions, percentage points,
#: as (point estimate, low, high) per condition. A triggered response must beat this.
#: A LAB REFERENCE, not a measurement of the project in hand (ruled 4 Nov 2026: it stays a
#: constant, since an unperturbed population is expected to hold still, but every text that
#: prints it names where it was measured, `UNTRIGGERED_DRIFT_SOURCE`). The keys are THAT data
#: set's conditions, never the project's.
UNTRIGGERED_DRIFT_PP = {
    "glucose": (0.92, -1.84, 3.68),
    "malate": (-0.08, -3.67, 3.51),
}

#: Where `UNTRIGGERED_DRIFT_PP` was measured (`timecourse`'s head; docs/claude/states_and_kinetics.md,
#: "THE TRACKS DO NOT GET SLOWER OR FASTER WITHIN A MOVIE"): last bin minus first, per movie.
UNTRIGGERED_DRIFT_SOURCE = ("a lab project (glucose and malate, untriggered 20 s acquisitions, "
                            "about 28 movies per condition)")

OBSERVABLES = {
    # column in `steps`            how it is summarised per bin
    "slow_fraction": "w_slow",
    "median_jump_um": "jump_um",
    "appd_um2_per_s": "sq_disp_um2",
}


def _linear(t, a, b):
    return a + b * t


def _exponential(t, a, amp, k):
    """Decay (or growth) all the way to ``a``."""
    return a + amp * np.exp(-k * t)


def _exponential_plateau(t, a, amp, k):
    """Same shape; kept separate so the plateau is a NAMED parameter in the report."""
    return a + amp * np.exp(-k * t)


RESPONSE_MODELS = {
    "linear": (_linear, ("intercept", "slope"), 2),
    "exponential": (_exponential, ("plateau", "amplitude", "rate"), 3),
}


def resolvable_halflife_range(n_frames: int, tau_s: float, n_bins: int) -> dict:
    """What half-lives this acquisition can actually separate.

    Below one bin the response is over before the first point; above about a third of
    the window it is indistinguishable from a straight line. Both bounds are reported
    rather than enforced -- a value outside them is a caveat, not an error.
    """
    if n_frames < 2 or tau_s <= 0 or n_bins < 2:
        raise KineticsError(
            f"need n_frames >= 2, tau_s > 0 and n_bins >= 2; got {n_frames}, {tau_s}, "
            f"{n_bins}")
    window_s = float(n_frames * tau_s)
    bin_s = window_s / n_bins
    return {
        "window_s": window_s,
        "bin_s": bin_s,
        "min_halflife_s": bin_s,
        "max_halflife_s": window_s / 3.0,
        "note": (
            f"this acquisition spans {window_s:.1f} s in {n_bins} bins of {bin_s:.2f} s; "
            f"half-lives below {bin_s:.2f} s are over inside one bin and above "
            f"{window_s / 3.0:.1f} s are not separable from a linear trend"),
    }


def response_curve(steps: pd.DataFrame, *, trigger_frame: int,
                   observable: str = "slow_fraction", unit: str = "movie",
                   tau_s: float, binned: bool = True, n_bins: int | None = None,
                   bin_seconds: float | None = None, min_steps_per_bin: int = 30,
                   min_tracks_per_bin: int = 3,
                   min_localizations: int | None = None) -> pd.DataFrame:
    """One response trajectory per unit, on the TRIGGER clock.

    ``trigger_frame`` has NO DEFAULT: it is a fact about the experiment, not a setting,
    and frame 0 is only one of the cases. Frames before it are dropped and counted --
    a pre-trigger baseline is a separate measurement, not part of the decay.

    ``binned=True`` (the default) bins every STEP at its own frame and averages tracks
    within each bin, so a long track contributes separately to each bin it crosses and
    never votes twice in one. Give EITHER ``bin_seconds`` (preferred: portable between
    acquisitions of different length) or ``n_bins``; giving both is refused, and giving
    neither falls back to ten bins with that stated on the result.

    ``binned=False`` places every TRACK at its own midpoint in time and returns one row
    per track -- the scatter the binned curve is an average of. It costs nothing in time
    resolution and needs no bin width, at the price of much noisier points; the interval
    still comes from the spread between units either way.

    Returns one row per (unit, bin) with the observable, its time in SECONDS since the
    trigger, and the contributing track and step counts. A bin below
    ``min_steps_per_bin`` or ``min_tracks_per_bin`` is NaN, never zero.
    """
    if observable not in OBSERVABLES:
        raise KineticsError(
            f"unknown observable {observable!r}; choose one of "
            f"{sorted(OBSERVABLES)}. Each is a different quantity, so the name is required.")
    source = OBSERVABLES[observable]
    for column in ("frame", "uid", source):
        if column not in steps.columns:
            raise KineticsError(f"missing '{column}' column")
    if unit not in steps.columns:
        raise KineticsError(f"no '{unit}' column to use as the unit")
    if tau_s <= 0:
        raise KineticsError(f"tau_s must be positive, got {tau_s}")
    if not binned and (n_bins is not None or bin_seconds is not None):
        raise KineticsError(
            "binned=False takes no bin width: every track is placed at its own moment, "
            "so n_bins and bin_seconds describe nothing. Drop them, or set binned=True")
    bin_source = "n_bins"
    if binned:
        if n_bins is not None and bin_seconds is not None:
            raise KineticsError(
                "give either bin_seconds or n_bins, not both: they set the same thing "
                "and cannot both apply")
        if bin_seconds is not None:
            if bin_seconds <= 0:
                raise KineticsError(f"bin_seconds must be positive, got {bin_seconds}")
            bin_source = "bin_seconds"
        elif n_bins is None:
            n_bins = 10
            bin_source = "default n_bins=10"
        if n_bins is not None and n_bins < 2:
            raise KineticsError(f"n_bins must be at least 2, got {n_bins}")

    frame = steps.copy()
    n_before = int((frame["frame"] < trigger_frame).sum())
    frame = frame[frame["frame"] >= trigger_frame]
    if frame.empty:
        raise KineticsError(
            f"no steps at or after frame {trigger_frame}; the trigger is after the end "
            f"of the data")

    kept_tracks = None
    if min_localizations is not None:
        if min_localizations < 2:
            raise KineticsError(
                f"min_localizations must be at least 2, got {min_localizations}")
        counts = frame.groupby("uid", observed=True)["frame"].size() + 1
        kept_tracks = counts[counts >= min_localizations].index
        frame = frame[frame["uid"].isin(kept_tracks)]
        if frame.empty:
            raise KineticsError(
                f"no track reaches {min_localizations} localisations after the trigger")

    since = frame["frame"].to_numpy(float) - float(trigger_frame)
    span = float(since.max())
    if span <= 0:
        raise KineticsError("all post-trigger steps sit in one frame; no curve to build")
    frame = frame.assign(since_frames=since)

    def _summarise(part):
        """The observable over a set of steps, TRACK-weighted.

        A long track must not vote once per step: after a trigger that changes D, track
        length itself changes with time, so step-weighting would let the signal reweight
        its own samples.
        """
        by_track = part.groupby("uid", observed=True)[source]
        if observable == "slow_fraction":
            return float(by_track.mean().mean())
        if observable == "median_jump_um":
            return float(by_track.median().median())
        return float(by_track.mean().mean() / (4.0 * tau_s))

    rows = []
    if binned:
        if bin_seconds is not None:
            width = bin_seconds / tau_s                      # seconds -> frames
            n_bins = max(int(np.ceil((span + 1e-9) / width)), 2)
            edges = np.arange(n_bins + 1, dtype=float) * width
        else:
            edges = np.linspace(0.0, span + 1e-9, n_bins + 1)
        frame = frame.assign(bin=np.clip(np.digitize(since, edges), 1, n_bins))
        for (key, b), part in frame.groupby([unit, "bin"], observed=True):
            n_steps = len(part)
            n_tracks = int(part["uid"].nunique())
            ok = n_steps >= min_steps_per_bin and n_tracks >= min_tracks_per_bin
            rows.append({
                unit: key, "bin": int(b),
                "t_s": float(0.5 * (edges[b - 1] + edges[b]) * tau_s),
                "value": _summarise(part) if ok else np.nan,
                "n_steps": n_steps, "n_tracks": n_tracks, "ok": bool(ok),
            })
    else:
        # UNBINNED: one row per track, at its own midpoint in time. `bin` is kept and
        # numbered so the returned schema is identical either way and everything
        # downstream -- fit_response, align_to_onset -- needs no branch.
        for key, part in without_attrs(frame).groupby(unit, observed=True):
            per_track = []
            for uid, block in without_attrs(part).groupby("uid", observed=True):
                n_steps = len(block)
                if n_steps < 1:
                    continue
                per_track.append((float(block["since_frames"].mean() * tau_s),
                                  _summarise(block), n_steps,
                                  float(block["since_frames"].max()
                                        - block["since_frames"].min()) * tau_s))
            per_track.sort()
            for index, (t_s, value, n_steps, _span) in enumerate(per_track, start=1):
                rows.append({unit: key, "bin": index, "t_s": t_s, "value": value,
                             "n_steps": n_steps, "n_tracks": 1, "ok": True})

    out = pd.DataFrame(rows).sort_values([unit, "bin"]).reset_index(drop=True)
    track_spans = (frame.groupby("uid", observed=True)["since_frames"]
                   .agg(lambda v: v.max() - v.min()) * tau_s)
    out.attrs.update({
        "unit": unit, "observable": observable, "trigger_frame": int(trigger_frame),
        "binned": bool(binned),
        "n_bins": (int(n_bins) if binned else None),
        "bin_seconds": (float(bin_seconds) if bin_seconds is not None
                        else (float(span * tau_s / n_bins) if binned else None)),
        "bin_source": (bin_source if binned else "unbinned: one point per track"),
        "tau_s": float(tau_s),
        "track_span_s_median": float(track_spans.median()),
        "track_span_s_p90": float(track_spans.quantile(0.90)),
        "n_steps_before_trigger": n_before,
        "min_localizations": min_localizations,
        "n_tracks_kept": (None if kept_tracks is None else int(len(kept_tracks))),
        "selection_caveat": (
            None if min_localizations is None else
            "a minimum track length is confounded with a change in D: slow molecules "
            "stay in focus longer, so the cut admits a larger share of tracks late than "
            "early. Measured cost at 15 localisations: a few per cent of tracks kept, "
            "slow fraction inflated by several points"),
    })
    return out


#: Unit resamples behind every interval, and how clearly the exponential has to predict held-out
#: units better than the line before it is recommended (the mean difference over its standard error).
N_BOOT = 200
HELDOUT_Z = 2.0


def _unit_series(curve: pd.DataFrame, unit: str) -> list:
    """``[(key, t, y)]``: each unit's usable points (binned means, or one point per track)."""
    out = []
    for key, part in without_attrs(curve).groupby(unit, observed=True):
        part = part[part["ok"] & np.isfinite(part["value"])]
        if len(part):
            out.append((key, part["t_s"].to_numpy(float), part["value"].to_numpy(float)))
    return out


def _fit_pooled(name: str, series: list):
    """ONE fit of model ``name`` to every unit's points, each UNIT weighing the same (its points
    share a weight of 1): for binned units that hold every bin, exactly the least-squares fit to
    the curve averaged over units. The parameters, or None where the fit fails."""
    func, _names, n_par = RESPONSE_MODELS[name]
    t = np.concatenate([s[1] for s in series])
    y = np.concatenate([s[2] for s in series])
    if len(t) < n_par + 1:
        return None
    w = np.concatenate([np.full(len(s[1]), 1.0 / len(s[1])) for s in series])
    sigma = 1.0 / np.sqrt(w)
    try:
        if name == "exponential":
            order = np.argsort(t)
            tail = y[order][-max(1, len(t) // 5):].mean()
            head = y[order][:max(1, len(t) // 5)].mean()
            span = max(float(t.max() - t.min()), 1e-6)
            popt, _ = curve_fit(func, t, y, p0=(tail, head - tail, 3.0 / span), sigma=sigma,
                                bounds=([-np.inf, -np.inf, 1e-4], [np.inf, np.inf, 50.0]),
                                maxfev=20000)
        else:
            popt, _ = curve_fit(func, t, y, sigma=sigma, maxfev=20000)
    except Exception:                                            # noqa: BLE001 - counted
        return None
    return popt if np.all(np.isfinite(popt)) else None


def _heldout(series: list, models: tuple, seed: int) -> dict:
    """Each unit predicted by every model fitted WITHOUT it (leave one out up to 20 units, ten
    folds above): the mean squared error per unit and model."""
    n = len(series)
    order = np.random.default_rng(seed).permutation(n)
    k = n if n <= 20 else 10
    folds = [order[i::k] for i in range(k)]
    errors = {name: np.full(n, np.nan) for name in models}
    for fold in folds:
        train = [series[i] for i in range(n) if i not in set(fold)]
        for name in models:
            popt = _fit_pooled(name, train)
            if popt is None:
                continue
            func = RESPONSE_MODELS[name][0]
            for i in fold:
                _key, t, y = series[i]
                errors[name][i] = float(np.mean((y - func(t, *popt)) ** 2))
    return errors


def fit_response(curve: pd.DataFrame, *, models: tuple[str, ...] = ("linear",
                                                                    "exponential"),
                 n_frames: int | None = None, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """ONE fit to the curve averaged over units; the interval from resampling UNITS; the model
    chosen on units HELD OUT.

    WHY NOT A FIT PER UNIT, AVERAGED (the rule until 13 Oct 2026). A unit holds a few tracks per
    bin -- about 11 per 2 s bin on a SynR movie -- and three exponential parameters are not
    identifiable from that: single movies gave plateaus of -141 and amplitudes of +150, and their
    average was no estimate at all. On SynR's switch (true half-life 4.2 s) the averaged fit read
    1.7 s and its curve left 0-100 %, and on the unswitched control it preferred an exponential
    of 0.4 s (summed per-unit BIC, delta 0.4). The pooled fit with each unit weighing the same
    reads 3.8 s [2.8, 6.1] and calls the control linear (`docs/synthetic_synr.md`).

    THE UNIT IS STILL THE MOVIE OR THE CELL, NEVER THE BIN: the interval comes from refitting
    ``n_boot`` resamples of whole units, so bins sharing a movie never count as independent, and
    the exponential is recommended only when it predicts held-out units better than the line by
    `HELDOUT_Z` standard errors of the per-unit difference. Otherwise the line stands and the
    shape is reported as undetermined.
    """
    unit = curve.attrs.get("unit", "movie")
    if unit not in curve.columns:
        raise KineticsError(f"no '{unit}' column")
    for name in models:
        if name not in RESPONSE_MODELS:
            raise KineticsError(f"unknown model {name!r}; have {sorted(RESPONSE_MODELS)}")
    series = _unit_series(curve, unit)
    if len(series) < 2:
        raise KineticsError(
            f"{len(series)} {unit}(s) with usable points: the interval and the held-out choice "
            f"need at least two")

    point = {name: _fit_pooled(name, series) for name in models}
    if all(v is None for v in point.values()):
        raise KineticsError(
            "no model could be fitted; check that units have enough non-NaN bins "
            "(each model needs more points than parameters)")

    rng = np.random.default_rng(seed)
    draws: dict[str, list] = {name: [] for name in models}
    n_failed = 0
    for _ in range(int(n_boot)):
        pick = [series[i] for i in rng.integers(0, len(series), len(series))]
        for name in models:
            popt = _fit_pooled(name, pick)
            if popt is None:
                n_failed += 1
            else:
                draws[name].append(popt)

    params = {}
    for name in models:
        if point[name] is None:
            continue
        names = RESPONSE_MODELS[name][1]
        boot = np.array(draws[name]) if draws[name] else np.empty((0, len(names)))
        for j, column in enumerate(names):
            values = boot[:, j] if len(boot) else np.array([])
            ok = len(values) >= 2
            params[f"{name}.{column}"] = {
                "mean": float(point[name][j]),
                "sem": float(values.std(ddof=1)) if ok else np.nan,
                "ci95": ((float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))
                         if ok else (np.nan, np.nan)),
                "n_units": len(series), "n_boot": int(len(values))}

    fitted = [name for name in models if point[name] is not None]
    errors = _heldout(series, tuple(fitted), seed)
    heldout = {name: float(np.nanmean(errors[name])) for name in fitted}
    folds_won = {name: 0 for name in fitted}
    margin = None
    if {"linear", "exponential"} <= set(fitted):
        d = errors["linear"] - errors["exponential"]
        d = d[np.isfinite(d)]
        mean_d = float(d.mean()) if len(d) else 0.0
        se_d = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else np.inf
        margin = {"mean": mean_d, "se": se_d,
                  "z": (mean_d / se_d) if se_d and np.isfinite(se_d) and se_d > 0 else 0.0}
        folds_won = {"linear": int(np.sum(d < 0)), "exponential": int(np.sum(d > 0))}
        if margin["z"] >= HELDOUT_Z:
            recommended, inconclusive = "exponential", False
        elif margin["z"] <= -HELDOUT_Z:
            recommended, inconclusive = "linear", False
        else:
            recommended, inconclusive = "linear", True
    else:
        recommended, inconclusive = fitted[0], False

    out = {
        "recommended": recommended,
        "inconclusive": bool(inconclusive),
        "heldout_error": heldout,
        "heldout_margin": margin,
        "folds_won": folds_won,
        "parameters": params,
        "n_units": len(series),
        "n_boot": int(n_boot),
        "n_failed_fits": n_failed,
        "unit": unit,
        "observable": curve.attrs.get("observable"),
        "untriggered_drift_pp": UNTRIGGERED_DRIFT_PP,
        "untriggered_drift_source": UNTRIGGERED_DRIFT_SOURCE,
    }

    rate = params.get("exponential.rate")
    if rate is not None and np.isfinite(rate["mean"]) and rate["mean"] > 0:
        out["halflife_s"] = float(np.log(2.0) / rate["mean"])
        tau_s = curve.attrs.get("tau_s")
        frames = n_frames
        if frames is None and tau_s:
            frames = int(round(curve["t_s"].max() * 2 / tau_s))
        if frames and tau_s:
            # THE TIME RESOLUTION IS NOT ALWAYS A BIN. Unbinned, the finest structure
            # resolvable is set by how long a TRACK lasts, since that is the width of one
            # point -- so the equivalent bin count is the window over the median track
            # span. `attrs.get("n_bins", 10)` also returns None rather than 10 here: the
            # key EXISTS and holds None, which is a different thing from absent.
            n_bins_equiv = curve.attrs.get("n_bins")
            if not n_bins_equiv:
                span = curve.attrs.get("track_span_s_median") or tau_s
                n_bins_equiv = max(int(round(frames * tau_s / max(span, tau_s))), 2)
            window = resolvable_halflife_range(frames, tau_s, int(n_bins_equiv))
            out["resolvable"] = window
            out["halflife_resolvable"] = bool(
                window["min_halflife_s"] <= out["halflife_s"] <= window["max_halflife_s"])
            if not out["halflife_resolvable"]:
                out["halflife_caveat"] = (
                    f"the fitted half-life {out['halflife_s']:.2f} s falls outside what "
                    f"this acquisition can separate ({window['note']})")
    half = out.get("halflife_s")
    span = curve.attrs.get("track_span_s_median")
    # UNBINNED ONLY: a binned curve places every STEP at its own time, so a long track is not smeared onto one moment
    # (3 Nov 2026, reported: the advice to "bin instead" was printed under a Run binned at 2 s)
    if half and span and span > 0.25 * half and not curve.attrs.get("binned", True):
        out["track_span_caveat"] = (
            f"the median track spans {span:.2f} s against a fitted half-life of "
            f"{half:.2f} s, so unbinned tracks smear the decay; bin instead, or "
            f"shorten the tracks by lowering the gap-closing window")
    rates = draws.get("exponential") or []
    if out.get("halflife_s") and len(rates) >= 2:
        half = np.log(2.0) / np.array([r[2] for r in rates])
        out["halflife_ci95"] = (float(np.percentile(half, 2.5)), float(np.percentile(half, 97.5)))
    if out["inconclusive"]:
        out["caveat"] = (
            f"held-out {unit}s do not tell the line from the exponential (difference "
            f"{margin['z']:+.1f} standard errors, {HELDOUT_Z:g} needed): the shape is "
            f"undetermined, and the line is reported as the simpler description")
    return out


def lifetime_qc(steps: pd.DataFrame, *, min_localizations: int, window: int,
                tau_s: float, unit: str = "movie") -> dict:
    """QUALITY CONTROL on the TRACK clock: what happens over a molecule's own lifetime.

    This is deliberately NOT a kinetics function. Tracks start at arbitrary times, so a
    trend here is a property of observation as much as of biology -- and the dominant
    property is survivorship: a track reaching step 15 never left focus and never
    overshot the linking radius, and long tracks are preferentially slow (measured here
    as Spearman(length, D) = -0.11). ``n_alive`` is returned beside the trace for exactly
    that reason: a declining mean over a collapsing sample is the expected picture.

    What DOES survive the caveat is ``excess``, from `timecourse.shuffle_null`: whether a
    track's own D varies more than shuffling its own steps produces. That is switching
    measured per molecule.
    """
    from smtracker.timecourse import shuffle_null, track_timeline

    line = track_timeline(steps, min_localizations=min_localizations, window=window,
                          tau_s=tau_s)
    first = line.groupby("uid")["window_start_frame"].transform("min")
    line = line.assign(since_start=(line["window_mid_frame"] - first).round(0))
    wide = line.pivot_table(index="uid", columns="since_start", values="appd_um2_per_s")
    arr = wide.to_numpy(float)
    n_alive = np.sum(~np.isnan(arr), axis=0).astype(int)

    null = shuffle_null(steps, min_localizations=min_localizations, window=window,
                        tau_s=tau_s)
    total = int(steps["uid"].nunique())
    return {
        "timeline": line,
        "since_start": np.asarray(wide.columns, float),
        "mean_D": np.nanmean(arr, axis=0),
        "n_alive": n_alive,
        "n_tracks": int(len(wide)),
        "n_tracks_total": total,
        "pct_tracks": 100.0 * len(wide) / max(total, 1),
        "observed_spread": null["observed_spread"],
        "chance_spread": null["chance_spread"],
        "excess": null["excess"],
        "chance_sd": null["chance_sd"],
        "beats_shuffle": bool(null["excess"] > 3.0 * null["chance_sd"]),
        "caveat": (
            "the track clock cannot measure a triggered response: tracks begin at "
            "arbitrary times, so re-zeroing flattens the decay. Use `response_curve` "
            "for that. A declining mean is expected from survivorship; only `excess` "
            "is evidence of within-track change"),
    }


# =====================================================================================
# ADAPTIVE ALIGNMENT: when the trigger is common but the RESPONSE is not
# =====================================================================================
#
# THE CASE. A drug is added at one moment, but it reaches each cell at its own time, so
# cells begin responding at different points. Averaging them on the external clock smears
# the transition -- the same failure the track clock has, one level up. Detecting each
# unit's OWN onset and re-aligning recovers the shape, and the spread of onsets is itself
# a measurement of how synchronous the response is.
#
# THE TRAP, AND IT IS SEVERE. Aligning every unit to a feature FOUND IN ITS OWN NOISE
# manufactures that feature in the average. Take pure noise, give each unit its steepest
# drop, line them up on it, and the mean shows a clean step with a plausible rate. This is
# the same mechanism as aligning random walks on their minimum. So the aligned curve is
# never reported alone: `onset_alignment_null` runs the identical detector and identical
# alignment on time-SHUFFLED copies of the same data -- same values, same noise, no time
# structure -- and the honest result is the aligned curve MINUS what alignment produces
# from nothing.
#
# WHAT ALIGNMENT ANSWERS THAT THE POOLED FIT CANNOT. If the units share a rate and differ
# only in latency, aligning sharpens the average and the per-unit rates agree. If the
# rates themselves differ, alignment does NOT sharpen it much and the rate spread stays
# wide. `alignment_gain` reports both, so "is the kinetic the same" is answered with a
# number rather than by looking.


def _changepoint(t: np.ndarray, y: np.ndarray) -> tuple[int, float]:
    """Best split into a flat baseline then a linear response, by residual sum.

    Deliberately the simplest defensible detector: a constant before, a straight line
    after. A more elaborate one would fit the response model being tested, which would
    let the detector choose the answer.
    """
    best, best_rss = None, np.inf
    for k in range(1, len(t) - 2):
        pre = y[:k]
        post_t, post_y = t[k:], y[k:]
        rss = float(np.sum((pre - pre.mean()) ** 2))
        slope, intercept = np.polyfit(post_t, post_y, 1)
        rss += float(np.sum((post_y - (intercept + slope * post_t)) ** 2))
        if rss < best_rss:
            best, best_rss = k, rss
    if best is None:
        raise KineticsError(
            "too few usable bins to locate an onset; a changepoint needs at least four")
    return best, best_rss


def detect_onsets(curve: pd.DataFrame, *, min_bins: int = 5) -> pd.DataFrame:
    """Each unit's own response onset, in seconds since the trigger.

    A unit with fewer than ``min_bins`` usable bins is returned with a NaN onset and
    counted, never dropped silently: "this cell's onset could not be located" and "this
    cell did not respond" are different statements.
    """
    unit = curve.attrs.get("unit", "movie")
    if curve.attrs.get("binned") is False:
        raise KineticsError(
            "onset detection needs a binned curve: a changepoint over one point per "
            "track is fitting a break to single-track noise, and there are thousands of "
            "them per unit. Rebuild with binned=True (bin_seconds is the portable "
            "setting) and detect the onset there")
    rows = []
    for key, part in without_attrs(curve).groupby(unit, observed=True):
        part = part[part["ok"] & np.isfinite(part["value"])].sort_values("t_s")
        if len(part) < min_bins:
            rows.append({unit: key, "onset_s": np.nan, "n_bins": len(part),
                         "rss": np.nan, "reason": f"only {len(part)} usable bins"})
            continue
        t = part["t_s"].to_numpy(float)
        y = part["value"].to_numpy(float)
        k, rss = _changepoint(t, y)
        rows.append({unit: key, "onset_s": float(t[k]), "n_bins": len(part),
                     "rss": rss, "reason": None})
    out = pd.DataFrame(rows)
    out.attrs.update({"unit": unit, "n_located": int(out["onset_s"].notna().sum()),
                      "n_unlocated": int(out["onset_s"].isna().sum())})
    return out


def align_to_onset(curve: pd.DataFrame, onsets: pd.DataFrame) -> pd.DataFrame:
    """Re-express every unit's curve as time since ITS OWN onset."""
    unit = curve.attrs.get("unit", "movie")
    table = onsets.set_index(unit)["onset_s"]
    out = curve.copy()
    shift = out[unit].map(table)
    out["t_aligned_s"] = out["t_s"] - shift
    out = out[np.isfinite(out["t_aligned_s"])]
    if out.empty:
        raise KineticsError("no unit had a locatable onset, so nothing can be aligned")
    out.attrs.update(curve.attrs)
    out.attrs["aligned"] = True
    return out


def _aligned_profile(aligned: pd.DataFrame, edges: np.ndarray) -> np.ndarray:
    idx = np.clip(np.digitize(aligned["t_aligned_s"].to_numpy(float), edges), 1,
                  len(edges) - 1)
    values = aligned["value"].to_numpy(float)
    return np.array([np.nanmean(values[idx == b]) if (idx == b).any() else np.nan
                     for b in range(1, len(edges))])


def onset_alignment_null(curve: pd.DataFrame, *, n_shuffles: int = 200,
                         n_profile_bins: int = 12, seed: int = 0) -> dict:
    """What onset-alignment produces from data with NO time structure.

    Each unit's bin VALUES are permuted in time -- the same numbers, the same spread, the
    order destroyed -- then the same detector and the same alignment are applied. The
    resulting step is what alignment manufactures. The observed aligned step must exceed
    it, and the difference is the finding.
    """
    unit = curve.attrs.get("unit", "movie")
    rng = np.random.default_rng(seed)

    onsets = detect_onsets(curve)
    aligned = align_to_onset(curve, onsets)
    span = float(np.nanmax(np.abs(aligned["t_aligned_s"])))
    edges = np.linspace(-span, span, n_profile_bins + 1)
    observed = _aligned_profile(aligned, edges)
    half = n_profile_bins // 2
    obs_step = float(np.nanmean(observed[:half]) - np.nanmean(observed[half:]))

    steps = []
    for _ in range(n_shuffles):
        shuffled = curve.copy()
        values = shuffled["value"].to_numpy(float).copy()
        for _, index in without_attrs(shuffled).groupby(unit, observed=True).indices.items():
            values[index] = rng.permutation(values[index])
        shuffled["value"] = values
        shuffled.attrs.update(curve.attrs)
        try:
            null_aligned = align_to_onset(shuffled, detect_onsets(shuffled))
        except KineticsError:
            continue
        profile = _aligned_profile(null_aligned, edges)
        steps.append(float(np.nanmean(profile[:half]) - np.nanmean(profile[half:])))

    if len(steps) < 10:
        raise KineticsError(
            f"only {len(steps)} of {n_shuffles} shuffles produced an alignable null; the "
            f"curve is too sparse for this guard to mean anything")
    arr = np.asarray(steps, float)
    chance = float(np.nanmean(arr))
    sd = float(np.nanstd(arr, ddof=1))
    return {
        "observed_step": obs_step,
        "chance_step": chance,
        "chance_sd": sd,
        "excess": obs_step - chance,
        "z": (obs_step - chance) / sd if sd > 0 else np.nan,
        "beats_null": bool(sd > 0 and (obs_step - chance) > 3.0 * sd),
        "n_shuffles": int(len(arr)),
        "profile_edges": edges,
        "observed_profile": observed,
        "onsets": onsets,
        "caveat": (
            "aligning each unit on a feature found in its own noise manufactures that "
            "feature in the average; `chance_step` is how big that artefact is here, and "
            "only `excess` is evidence of a real synchronised response"),
    }


def alignment_gain(curve: pd.DataFrame,
                   *, models: tuple[str, ...] = ("exponential",)) -> dict:
    """Does aligning sharpen the response, and do the per-unit RATES then agree?

    THIS IS THE QUESTION "is the kinetic the same". Two outcomes, and they differ:

      units share a rate, differ in LATENCY   aligning sharpens the mean curve and the
                                              per-unit rate spread falls
      units differ in RATE                    aligning does not sharpen it much and the
                                              rate spread stays wide

    Reported as the ratio of the rate's coefficient of variation after alignment to
    before, plus the onset spread that alignment removed.
    """
    onsets = detect_onsets(curve)
    located = onsets["onset_s"].dropna().to_numpy(float)
    if len(located) < 2:
        raise KineticsError(
            f"only {len(located)} unit(s) have a locatable onset; alignment needs at "
            f"least two to say anything about spread")

    before = fit_response(curve, models=models)
    aligned = align_to_onset(curve, onsets).copy()
    aligned["t_s"] = aligned["t_aligned_s"] - float(np.nanmin(aligned["t_aligned_s"]))
    aligned.attrs.update(curve.attrs)
    after = fit_response(aligned, models=models)

    def cv(fit):
        rate = fit["parameters"].get("exponential.rate")
        if rate is None or not np.isfinite(rate["mean"]) or rate["mean"] == 0:
            return np.nan
        return abs(rate["sem"] * np.sqrt(rate["n_units"]) / rate["mean"])

    cv_before, cv_after = cv(before), cv(after)
    sharpened = (np.isfinite(cv_before) and np.isfinite(cv_after)
                 and cv_after < 0.7 * cv_before)
    return {
        "onset_spread_s": float(np.std(located, ddof=1)),
        "onset_range_s": (float(located.min()), float(located.max())),
        "n_units_aligned": int(len(located)),
        "n_units_unlocated": int(onsets["onset_s"].isna().sum()),
        "rate_cv_before": cv_before,
        "rate_cv_after": cv_after,
        "cv_ratio": (cv_after / cv_before) if (np.isfinite(cv_before)
                                               and cv_before > 0) else np.nan,
        "verdict": "latency" if sharpened else "rate heterogeneity or no response",
        "fit_before": before,
        "fit_after": after,
        "caveat": (
            "run `onset_alignment_null` first: alignment sharpens noise too, so a "
            "reduced rate spread is only meaningful once the aligned step beats what "
            "alignment produces from time-shuffled data"),
    }
