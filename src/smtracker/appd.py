"""appD: apparent-D per short sub-track, fitted as a Gamma mixture.

The second baseline estimator beside ``smtracker.sqd``. Where SQD fits the
distribution of *individual* displacements, appD first collapses each short
sub-track of ``n_steps`` steps to one apparent diffusion coefficient and then
fits the distribution of *those*.

Why a Gamma mixture is the exact model
--------------------------------------
For 2-D Brownian motion each squared displacement is exponential with mean
``4·D·τ``. The mean of ``n`` independent ones is therefore Gamma-distributed,
so with ``appD = mean(r²)/(4τ)``:

    appD  ~  Gamma(shape = n, scale = D / n)

and for a mixture of states::

    p(x) = Σ_k  f_k · Gamma(x; shape = n, scale = D_k / n)

matching the legacy ``gammafit_two.m`` / ``gammaCDF.m`` exactly
(``f·(n/D)ⁿ·x^{n−1}·e^{−n·x/D}/(n−1)!``). No binning, no ECDF: the likelihood
is analytic.

WHICH appD? Two legacy definitions exist: this module implements one
---------------------------------------------------------------------
The legacy tool contains **two different apparent-D estimators**, and they are
not interchangeable:

1. ``getAppD_gui.m`` / ``calcDapp_mle.m``: ``appD = mean(r²) / (4τ)`` over a
   short sub-track. **This module implements that one.** It is the estimator
   whose Gamma-mixture likelihood is exact (see above).
2. ``fitMSDTracks.m`` → ``appDFilter.m`` → ``calcAppD_gamma.m``, the estimator
   the **GUI's appD tab actually uses**: each track gets its own *weighted
   linear MSD fit*, ``D = slope/4``, plus an R² and a Brownian-vs-anomalous
   F-test (``diffusionClassification.m``); tracks are then filtered to
   ``0 ≤ D ≤ 10``, ``R² ≥ r2Lim`` and Brownian-only before the Gamma fit.

Definition 2 is a *fitted* per-track D, not an averaged displacement, and it is
applied to a **selected** subset. On the reference dataset the two differ
substantially (D₂ ≈ 0.76 under definition 1 versus ≈ 0.52–0.60 under
definition 2), so reference numbers produced by the GUI cannot be reproduced by
definition 1. Implementing definition 2 needs the GUI settings that were used
(minimum track length, excluded last lags, the R² cut): until those are known,
this module does not claim to reproduce a GUI appD result.

Note also that definition 2's Gamma shape is only heuristic: a fitted slope is
not a mean of ``n`` exponentials, so the Gamma model is exact for definition 1
and approximate for definition 2.

Trade-off vs SQD, stated plainly
--------------------------------
Averaging ``n`` steps per sub-track sharpens each estimate but blurs states
whose D's are close, and the Gamma width is fixed by ``n`` rather than fitted.
appD is a **cross-check** on the SQD fractions, not a replacement: agreement
between the two is evidence, disagreement is a finding to investigate.

Legacy bugs NOT reproduced (see ``calcAppD_gamma.m``)
----------------------------------------------------
- **All-zero returns.** On failure the legacy sets ``paramEsts = zeros(...)``
  and merely ``disp``s a message, so a silent all-zero "fit" flows onward as if
  it were a result. Here a failed fit raises.
- **BIC computed on ECDF points.** The legacy uses ``n = numel(xData)`` and
  ``L_ = prod(pdf(xData))`` where ``xData`` comes from ``ecdf(x)``: i.e. the
  likelihood is evaluated at the *unique CDF abscissae*, not at the data. Here
  BIC uses the true log-likelihood over every observation.
- **Product of densities.** ``prod(...)`` over thousands of points underflows to
  0, making ``log(L_)`` infinite and BIC meaningless. Here: sum of logs.
- **Silent ``try/catch``** around every start. Here failures are counted and, if
  none succeeds, raised.
- **Best start chosen by R² on the CDF**, mixing estimators: the fit is MLE but
  the winner is picked by a least-squares criterion. Here the best start is the
  one with the highest likelihood, the quantity actually being maximised.
- **Fractions could leave the simplex** (independent bounds). Stick-breaking
  keeps them valid, as in ``sqd``.

Sub-track construction (``mode``)
---------------------------------
``"split"`` (legacy default) cuts each track into consecutive
``n_steps + 1``-point chunks, discarding the remainder. ``"truncate"`` keeps
only the first chunk. Note ``"truncate"`` gives every track exactly one appD
value, so it is **naturally equal-per-track**, immune to the step-count
over-representation described in ``observables.displacement_weights``, whereas
``"split"`` yields more sub-tracks from longer (and so preferentially slower)
tracks and needs ``weighting`` to correct it.
"""

from __future__ import annotations

from .chart_text import unit_word  # the word shown for a grouping unit

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from .canonical import CANONICAL_COLUMNS
from .observables import ObservableError, displacement_weights
from .sqd import (
    FEW_UNITS,
    BOOTSTRAP_UNITS,
    NPopSelection,
    SQDError,
    _effective_n,
    _pack,
    _unpack,
)
from .canonical import without_attrs

__all__ = [
    "TRACK_MIN_STEPS",
    "NOISE_OFFSET_COLUMN",
    "track_appd",
    "AppDError",
    "AppDFit",
    "apparent_D",
    "appd_pdf",
    "appd_cdf",
    "fit_appd",
    "select_n_pop_appd",
    "bootstrap_appd",
    "DEFAULT_N_STEPS",
    "APPD_MODES",
]

#: Legacy ``Nsteps = 4``: sub-tracks of 4 steps (5 localizations).
DEFAULT_N_STEPS = 4
APPD_MODES = ("split", "truncate")

#: The Apparent Diffusion section's per-track apparent D needs at least this many single-frame steps (5 localisations,
#: u-track's own minimum track length).
TRACK_MIN_STEPS = 4

#: PER-SPOT PRECISION (6 Nov 2026): each track's own noise offset, um^2/s -- the mean of its steps' per-axis
#: localisation variance (`precision.step_noise`) over 2 tau, what localisation error adds to its apparent D on
#: average (`appd_gamma`'s module head). Only where `track_appd` is given a calibration.
NOISE_OFFSET_COLUMN = "noise_offset_um2_per_s"


def track_appd(canonical: pd.DataFrame, *, min_steps: int = TRACK_MIN_STEPS, calibration=None) -> pd.DataFrame:
    """ONE apparent D per track: the mean of ALL its single-frame squared jumps over 4 tau (16 Oct 2026, the
    Apparent Diffusion section's one definition, chosen by the experimenter over v2's MSD slope and the 4-step
    pieces).

    Built on `observables.squared_displacements`, the one jump table, so a step bridging a frame gap is never a
    step. ``n_steps`` is how many jumps the value averages: under one Brownian state the value is Gamma(n_steps,
    D / n_steps), the shape `appd_gamma` fits it with, so every step is used and every molecule counted once (a
    4-step piece counts a long track several times; v2's MSD slope drops tracks under 9 points and is not
    gamma-distributed). It is an APPARENT D: localisation error adds about sigma^2 / tau, motion blur takes a
    little, and a molecule that switched state within its track reads in between.

    One row per track with at least ``min_steps`` steps: ``uid, appd_um2_per_s, n_steps, protein, condition, day,
    movie, cell, first_frame``. ``attrs``: ``tau_s``, ``min_steps``, ``n_tracks_too_short``.

    PER-SPOT PRECISION: with a ``calibration`` (`precision.Calibration`, measured on immobile spots, never learned)
    each row also carries `NOISE_OFFSET_COLUMN`, the track's own noise offset, for `appd_gamma`'s ``offset``. A step
    with a spot lacking its precision (a failed spot fit) is LEFT OUT of its track, counted in
    ``attrs["precision_left_out"]``; above `precision.MAX_MISSING_SHARE` the table is refused
    (`precision.usable_steps`). ``attrs["precision_calibration"]``: ``(c_nm, p)``.
    """
    from .observables import squared_displacements

    if min_steps < 1:
        raise AppDError(f"min_steps must be at least 1, got {min_steps}")
    sd = squared_displacements(canonical)
    tau = float(sd.attrs["tau_s"])
    noise, left_out = None, 0
    if calibration is not None:
        from .precision import PrecisionError, step_noise, usable_steps

        try:
            ok = usable_steps(sd)
            left_out = int((~ok).sum())
            if left_out:
                sd = sd.loc[ok]
            noise = step_noise(sd, calibration)
        except PrecisionError as exc:
            raise AppDError(str(exc)) from exc
    plain = sd.copy(deep=False)
    plain.attrs = {}                                   # pandas copies attrs into every group: none to copy
    agg = {"n_steps": ("sq_disp_um2", "size"), "mean_sq": ("sq_disp_um2", "mean"),
           "first_frame": ("frame", "min")}
    if noise is not None:
        plain["step_noise_um2"] = noise
        agg["mean_noise"] = ("step_noise_um2", "mean")
    for column in ("protein", "condition", "day", "movie", "cell"):
        if column in plain.columns:
            agg[column] = (column, "first")
    per = plain.groupby("uid", sort=True, observed=True).agg(**agg).reset_index()
    short = int((per["n_steps"] < min_steps).sum())
    per = per[per["n_steps"] >= min_steps].copy()
    if per.empty:
        raise AppDError(f"no track has {min_steps} single-frame steps; nothing to compute an apparent D from")
    per["appd_um2_per_s"] = per.pop("mean_sq") / (4.0 * tau)
    per["n_steps"] = per["n_steps"].astype(int)
    columns = ["uid", "appd_um2_per_s", "n_steps"] + [c for c in ("protein", "condition", "day", "movie", "cell")
                                                       if c in per.columns] + ["first_frame"]
    if noise is not None:
        # E[r^2] = 4 D tau + 2 noise per step, so a track's mean over 4 tau reads D + mean(noise) / (2 tau)
        per[NOISE_OFFSET_COLUMN] = per.pop("mean_noise") / (2.0 * tau)
        columns.append(NOISE_OFFSET_COLUMN)
    out = per[columns].reset_index(drop=True)
    out.attrs.update(tau_s=tau, min_steps=int(min_steps), n_tracks_too_short=short)
    if calibration is not None:
        out.attrs.update(precision_left_out=left_out,
                         precision_calibration=(float(calibration.c_nm), float(calibration.p)))
    return out
_DEFAULT_STARTS = 50
_PENALTY_PER_POINT = 1e6

_GROUP_COLUMNS = ["protein", "condition", "day", "movie", "cell"]


class AppDError(RuntimeError):
    """Raised when appD values cannot be computed or fitted."""


@dataclass
class AppDFit:
    """An appD Gamma-mixture fit. Field names mirror ``SQDFit`` so the shared
    ``NPopSelection`` machinery (BIC table, analyst's choice) works unchanged."""

    n_pop: int
    D: np.ndarray                      # µm²/s, ascending
    fractions: np.ndarray
    n_steps: int
    tau_lag_s: float
    n_observations: int
    log_likelihood: float
    bic: float
    method: str = "mle"
    weighting: str = "step"
    mode: str = "split"
    n_starts: int = 1
    n_converged_starts: int = 0
    start_spread: dict = field(default_factory=dict)

    @property
    def fractions_percent(self) -> np.ndarray:
        return self.fractions * 100.0

    def summary(self) -> str:
        parts = [
            f"pop{i + 1}: {f * 100:.1f} %, D = {d:.4g} µm²/s"
            for i, (f, d) in enumerate(zip(self.fractions, self.D))
        ]
        return (
            f"appD {self.n_pop}-population Gamma fit "
            f"(n_steps = {self.n_steps}, mode = {self.mode}, "
            f"n = {self.n_observations}): " + "; ".join(parts)
            + f" | BIC = {self.bic:.1f}"
        )

    def to_dict(self) -> dict:
        out = {
            "n_pop": self.n_pop,
            "n_steps": self.n_steps,
            "mode": self.mode,
            "method": self.method,
            "weighting": self.weighting,
            "n_observations": self.n_observations,
            "log_likelihood": self.log_likelihood,
            "bic": self.bic,
        }
        for i, (d, f) in enumerate(zip(self.D, self.fractions), start=1):
            out[f"D_{i}"] = d
            out[f"fraction_{i}"] = f
        return out


def apparent_D(
    canonical: pd.DataFrame,
    *,
    n_steps: int = DEFAULT_N_STEPS,
    mode: str = "split",
    drop_first: bool = False,
    calibration=None,
) -> pd.DataFrame:
    """One apparent D per short sub-track.

    Parameters
    ----------
    canonical
        Canonical table (µm, seconds), from ``canonical.to_canonical``.
    n_steps
        Steps per sub-track (legacy ``Nsteps = 4``, i.e. 5 localizations).
    mode
        ``"split"``: consecutive non-overlapping chunks, remainder discarded
        (legacy default). ``"truncate"``: only the first chunk of each track,
        which makes the result naturally equal-per-track.
    drop_first
        Drop each track's first localization before splitting (the legacy
        ``deleted`` option; the first localization can carry a different
        detection bias).
    calibration
        PER-SPOT PRECISION (a `precision.Calibration`, measured on immobile spots, never learned; 6 Nov 2026, the
        Spatial Map's own apparent-D fit): each sub-track also carries `NOISE_OFFSET_COLUMN`, the mean of its steps'
        per-axis localisation variance (both spots through the calibration) over 2 tau -- what `track_appd` gives a
        whole track. A sub-track with a spot lacking its precision is LEFT OUT and counted
        (``attrs["precision_left_out"]``); above `precision.MAX_MISSING_SHARE` of them, or on a table with no
        per-spot precision at all, refused by name.

    Returns
    -------
    pandas.DataFrame
        One row per sub-track: ``appd_um2_per_s``, ``uid``, ``subtrack``
        (``"<uid>#k"``) and the grouping columns.
    """
    missing = [c for c in CANONICAL_COLUMNS if c not in canonical.columns]
    if missing:
        raise AppDError(f"not a canonical table (missing {missing})")
    if n_steps < 1:
        raise AppDError(f"n_steps must be >= 1, got {n_steps}")
    if mode not in APPD_MODES:
        raise AppDError(f"mode must be one of {APPD_MODES}, got {mode!r}")

    # PER-TRACK D NEEDS ONLY THAT TRACK'S OWN tau. Unlike a pooled model, whose
    # likelihood has one tau in it, every sub-track here is reduced independently to
    # MSD/(4*tau), so a project mixing acquisition rates is fine as long as each
    # track is divided by its own. The canonical `tau_s` column carries it per row;
    # the scalar is only the fallback for tables built before that column existed.
    tau_col = "tau_s" if "tau_s" in canonical.columns else None
    tau_default = None
    if tau_col is None:
        params = canonical.attrs.get("params")
        if params is None:
            raise AppDError(
                "canonical table carries no AcquisitionParams in .attrs['params'] "
                "and no 'tau_s' column; τ is required."
            )
        try:
            tau_default = float(params.tau_s)
        except Exception as exc:                    # mixed-tau project, no column
            raise AppDError(
                f"{exc} Per-track appD can handle mixed rates, but this table has no "
                "'tau_s' column to read them from. Rebuild it with "
                "canonical.to_canonical.") from exc
    n_points = n_steps + 1
    if calibration is not None:
        from .canonical import LOC_SIGMA_COLUMN

        if LOC_SIGMA_COLUMN not in canonical.columns:
            raise AppDError(
                "these tracks carry no per-spot precision: the tracker wrote none (u-track does; TrackMate does when "
                "its spots were fitted, `spotfit`; track tables do not). Per-spot precision cannot be used on them.")
    left_out = 0

    rows = []
    ordered = canonical.sort_values(["uid", "frame"], kind="stable")
    # The loop below iterates `without_attrs(ordered)`: 53.7 s -> 2.1 s on protein A/Malate,
    # identical result (see `canonical.without_attrs`). tau was already read above.
    for uid, track in without_attrs(ordered).groupby("uid", sort=False, observed=True):
        tau = (float(track[tau_col].iloc[0]) if tau_col is not None else tau_default)
        if drop_first:
            track = track.iloc[1:]
        if len(track) < n_points:
            continue
        x = track["x_um"].to_numpy()
        y = track["y_um"].to_numpy()
        frames = track["frame"].to_numpy()
        limit = 1 if mode == "truncate" else len(track) // n_points
        for k in range(limit):
            start = k * n_points
            xs = x[start : start + n_points]
            ys = y[start : start + n_points]
            fs = frames[start : start + n_points]
            # Legacy pairs by row; only accept genuinely consecutive frames so a
            # bridged gap is never treated as one step.
            if np.any(np.diff(fs) != 1):
                continue
            extra = {}
            if calibration is not None:
                # E[r^2] = 4 D tau + 2 noise per step, so the piece's mean over 4 tau reads D + mean(noise) / (2 tau)
                sigma = track[LOC_SIGMA_COLUMN].to_numpy(float)[start : start + n_points]
                if not (np.isfinite(sigma) & (sigma > 0)).all():
                    left_out += 1
                    continue
                var = calibration.true_sigma_um(sigma) ** 2
                extra[NOISE_OFFSET_COLUMN] = float((var[:-1] + var[1:]).mean() / (2.0 * tau))
            squared = np.diff(xs) ** 2 + np.diff(ys) ** 2
            rows.append(
                {
                    "appd_um2_per_s": float(squared.mean() / (4.0 * tau)),
                    "uid": uid,
                    "subtrack": f"{uid}#{k}",
                    **{c: track[c].iloc[0] for c in _GROUP_COLUMNS},
                    **extra,
                }
            )
    if calibration is not None and left_out:
        from .precision import MAX_MISSING_SHARE

        share = left_out / float(left_out + len(rows))
        if share > MAX_MISSING_SHARE:
            raise AppDError(
                f"{left_out} of {left_out + len(rows)} sub-tracks ({100 * share:.1f} %) have a spot without a "
                f"precision, more than the {100 * MAX_MISSING_SHARE:.0f} % a failed fit here and there explains: "
                f"refused, never filled in")

    if not rows:
        raise AppDError(
            f"no sub-track of {n_steps} steps ({n_points} consecutive "
            "localizations) could be formed. Tracks may be too short or gapped."
        )
    out = pd.DataFrame(rows)
    if not np.isfinite(out["appd_um2_per_s"]).all():
        raise AppDError("non-finite apparent D values")
    out.attrs["n_steps"] = n_steps
    out.attrs["mode"] = mode
    out.attrs["tau_s"] = tau
    out.attrs["lag_time_s"] = tau
    if calibration is not None:
        out.attrs["precision_left_out"] = int(left_out)
        out.attrs["precision_calibration"] = (float(calibration.c_nm), float(calibration.p))
    return out


def appd_pdf(x, D, fractions, n_steps: int) -> np.ndarray:
    """Gamma-mixture density: ``Σ f_k Gamma(x; shape=n, scale=D_k/n)``."""
    x = np.asarray(x, dtype=float)
    D = np.asarray(D, dtype=float)
    fractions = np.asarray(fractions, dtype=float)
    if D.shape != fractions.shape:
        raise AppDError(f"D and fractions must match: {D.shape} vs {fractions.shape}")
    if np.any(D <= 0):
        raise AppDError(f"all D must be > 0, got {D}")
    if n_steps < 1:
        raise AppDError(f"n_steps must be >= 1, got {n_steps}")
    if np.any(x < 0):
        raise AppDError("apparent D cannot be negative")
    return np.exp(_log_component_density(x, D, n_steps)) @ fractions


def appd_cdf(x, D, fractions, n_steps: int) -> np.ndarray:
    """Gamma-mixture CDF (for plotting and goodness-of-fit only)."""
    from scipy.stats import gamma as gamma_dist

    x = np.asarray(x, dtype=float)
    D = np.asarray(D, dtype=float)
    fractions = np.asarray(fractions, dtype=float)
    if np.any(D <= 0):
        raise AppDError(f"all D must be > 0, got {D}")
    components = np.column_stack(
        [gamma_dist.cdf(x, a=n_steps, scale=d / n_steps) for d in D]
    )
    return components @ fractions


def _log_component_density(x, D, n_steps: int) -> np.ndarray:
    """``log Gamma(x; shape=n, scale=D_k/n)`` for each component, shape (N, K).

    Computed in log space: the legacy form ``(n/D)^n x^(n-1) e^(-n x/D)/(n-1)!``
    overflows for small D and underflows for large x.
    """
    x = np.asarray(x, dtype=float)[:, None]
    scale = np.asarray(D, dtype=float)[None, :] / n_steps
    safe = np.maximum(x, np.finfo(float).tiny)
    return (
        (n_steps - 1) * np.log(safe)
        - safe / scale
        - n_steps * np.log(scale)
        - gammaln(n_steps)
    )


def fit_appd(
    values,
    *,
    n_pop: int,
    n_steps: int | None = None,
    weighting: str = "step",
    n_starts: int = _DEFAULT_STARTS,
    seed: int | None = 0,
    D_bounds: tuple[float, float] = (1e-4, 10.0),
) -> AppDFit:
    """Fit the Gamma mixture by **maximum likelihood over the data**.

    ``values`` is the table from ``apparent_D`` (``n_steps`` taken from its
    ``.attrs``) or a bare array of apparent-D values (then ``n_steps`` is
    required).

    Raises
    ------
    AppDError
        On bad input or if no start converged. It never returns zeros: the
        legacy's silent all-zero fallback is exactly what this avoids.
    """
    data, n_steps, tau, weights = _extract(values, n_steps, weighting)
    if n_pop < 1:
        raise AppDError(f"n_pop must be >= 1, got {n_pop}")
    if n_starts < 1:
        raise AppDError(f"n_starts must be >= 1, got {n_starts}")
    lo, hi = D_bounds
    if not 0 < lo < hi:
        raise AppDError(f"D_bounds must satisfy 0 < low < high, got {D_bounds}")
    if len(data) < 2 * n_pop:
        raise AppDError(
            f"only {len(data)} apparent-D values for a {n_pop}-population fit; "
            "refusing to fit an underdetermined model."
        )
    if np.ptp(data) <= 0:
        raise AppDError(
            f"all {len(data)} apparent-D values are identical ({data[0]}); there "
            "is no distribution to fit."
        )

    rng = np.random.default_rng(seed)
    starts = _starts(data, n_pop, n_starts, D_bounds, rng)
    lower = np.concatenate([np.full(n_pop, lo), np.zeros(n_pop - 1)])
    upper = np.concatenate([np.full(n_pop, hi), np.ones(n_pop - 1)])
    penalty = _PENALTY_PER_POINT * max(len(data), 1)

    def negative_log_likelihood(theta):
        D, fractions = _unpack(theta, n_pop)
        if np.any(D <= 0):
            return penalty
        value = -_log_likelihood(data, D, fractions, n_steps, weights)
        return penalty if not np.isfinite(value) else value

    results = []
    for theta0 in starts:
        try:
            outcome = minimize(
                negative_log_likelihood, theta0, method="L-BFGS-B",
                bounds=list(zip(lower, upper)),
                options={"maxiter": 5000, "ftol": 1e-12},
            )
        except (ValueError, FloatingPointError):
            continue
        if not outcome.success or not np.all(np.isfinite(outcome.x)):
            continue
        # Best start = highest LIKELIHOOD, the quantity being maximised.
        # (The legacy picked the best start by R² on the CDF instead.)
        results.append((float(outcome.fun), outcome.x))

    if not results:
        raise AppDError(
            f"no start converged for n_pop={n_pop} on {len(data)} apparent-D "
            "values. The model may not be identifiable on this data."
        )

    results.sort(key=lambda cr: cr[0])
    _, best = results[0]
    D, fractions = _unpack(best, n_pop)
    order = np.argsort(D)
    D, fractions = D[order], fractions[order]

    log_likelihood = _log_likelihood(data, D, fractions, n_steps, weights)
    n_free = 2 * n_pop - 1
    # BIC over the DATA, from the true log-likelihood, never over ECDF points,
    # and never from a product of densities that underflows to zero.
    bic = n_free * np.log(_effective_n(data, weights)) - 2.0 * log_likelihood

    return AppDFit(
        n_pop=n_pop,
        D=D,
        fractions=fractions,
        n_steps=n_steps,
        tau_lag_s=tau,
        n_observations=len(data),
        log_likelihood=float(log_likelihood),
        bic=float(bic),
        weighting=weighting,
        mode=str(getattr(values, "attrs", {}).get("mode", "split")),
        n_starts=len(starts),
        n_converged_starts=len(results),
        start_spread=_start_spread(results, n_pop),
    )


def select_n_pop_appd(
    values,
    *,
    max_pop: int = 3,
    min_pop: int = 1,
    n_steps: int | None = None,
    **fit_kwargs,
) -> NPopSelection:
    """BIC across population counts: reuses ``sqd.NPopSelection`` unchanged.

    BIC **recommends**; ``.choose(n, reason=...)`` records the analyst's call.
    """
    if min_pop < 1 or max_pop < min_pop:
        raise AppDError(f"need 1 <= min_pop <= max_pop, got {min_pop}..{max_pop}")
    fits = [
        fit_appd(values, n_pop=n, n_steps=n_steps, **fit_kwargs)
        for n in range(min_pop, max_pop + 1)
    ]
    return NPopSelection(fits=fits)


def bootstrap_appd(
    table: pd.DataFrame,
    *,
    n_pop: int,
    unit: str = "cell",
    n_boot: int = 200,
    seed: int | None = 0,
    **fit_kwargs,
) -> pd.DataFrame:
    """Honest error bars: resample independent units (cells / movies / days)."""
    if unit not in BOOTSTRAP_UNITS:
        raise AppDError(f"unit must be one of {BOOTSTRAP_UNITS}, got {unit!r}")
    if unit not in table.columns:
        raise AppDError(f"appD table has no '{unit}' column")
    if table[unit].isna().any():
        raise AppDError(f"'{unit}' is unassigned for some sub-tracks")
    if n_boot < 1:
        raise AppDError(f"n_boot must be >= 1, got {n_boot}")

    n_steps = table.attrs.get("n_steps")
    groups = [frame for _, frame in table.groupby(unit, observed=True)]
    if len(groups) < 2:
        raise AppDError(
            f"only {len(groups)} distinct {unit_word(unit)}(s); a bootstrap over one unit is "
            "meaningless."
        )

    rng = np.random.default_rng(seed)
    rows, failures = [], 0
    for _ in range(n_boot):
        picked = rng.integers(0, len(groups), size=len(groups))
        data = np.concatenate([groups[i]["appd_um2_per_s"].to_numpy() for i in picked])
        try:
            fit = fit_appd(data, n_pop=n_pop, n_steps=n_steps, **fit_kwargs)
        except AppDError:
            failures += 1
            continue
        rows.append(fit.to_dict())

    if not rows:
        raise AppDError(f"every one of {n_boot} bootstrap replicates failed to fit.")
    out = pd.DataFrame(rows)
    out.attrs["unit"] = unit
    out.attrs["n_units"] = len(groups)
    out.attrs["n_failed"] = failures
    out.attrs["few_units"] = len(groups) < FEW_UNITS
    out.attrs["caveat"] = (
        f"only {len(groups)} {unit_word(unit)}(s): the percentile interval is coarse."
        if len(groups) < FEW_UNITS else ""
    )
    return out


def _extract(values, n_steps, weighting):
    """Accept the appD table or a bare array; return (data, n_steps, tau, weights)."""
    weights = None
    tau = float("nan")
    if isinstance(values, pd.DataFrame):
        if "appd_um2_per_s" not in values.columns:
            raise AppDError("DataFrame has no 'appd_um2_per_s' column")
        data = values["appd_um2_per_s"].to_numpy(dtype=float)
        n_steps = n_steps if n_steps is not None else values.attrs.get("n_steps")
        tau = float(values.attrs.get("tau_s", float("nan")))
        try:
            weights = displacement_weights(values, weighting)
        except ObservableError as exc:
            raise AppDError(str(exc)) from exc
    else:
        data = np.asarray(values, dtype=float).ravel()
        if weighting != "step":
            raise AppDError(
                f"weighting={weighting!r} needs the appD DataFrame (a bare array "
                "carries no track/cell identity)."
            )
    if n_steps is None:
        raise AppDError(
            "n_steps is unknown: pass it, or pass the DataFrame from apparent_D "
            "(which records it). The Gamma shape depends on it."
        )
    if len(data) == 0:
        raise AppDError("no apparent-D values to fit")
    if not np.isfinite(data).all():
        raise AppDError("apparent-D values contain non-finite entries")
    if np.any(data < 0):
        raise AppDError("apparent D cannot be negative")
    return data, int(n_steps), tau, weights


def _log_likelihood(data, D, fractions, n_steps, weights=None) -> float:
    """Sum of log densities, never a product (which underflows to zero)."""
    log_components = _log_component_density(data, D, n_steps)
    with np.errstate(divide="ignore"):
        log_weights = np.log(np.maximum(np.asarray(fractions, dtype=float), 1e-300))
    shifted = log_components + log_weights[None, :]
    peak = shifted.max(axis=1)
    log_density = peak + np.log(np.exp(shifted - peak[:, None]).sum(axis=1))
    if not np.isfinite(log_density).all():
        return -np.inf
    if weights is None:
        return float(log_density.sum())
    w = np.asarray(weights, dtype=float)
    scale = _effective_n(data, weights) / w.sum()
    return float(np.sum(w * log_density) * scale)


def _starts(data, n_pop, n_starts, D_bounds, rng) -> list[np.ndarray]:
    """Quantile-based first start (as the legacy seeds) plus random ones."""
    lo, hi = D_bounds
    quantiles = np.linspace(0.25, 0.75, n_pop) if n_pop > 1 else np.array([0.5])
    seed_D = np.clip(np.quantile(data, quantiles), lo, hi)
    seed_D = np.sort(np.maximum.accumulate(seed_D))
    starts = [_pack(seed_D, np.full(n_pop, 1.0 / n_pop))]
    for _ in range(max(n_starts - 1, 0)):
        D = np.sort(np.exp(rng.uniform(np.log(lo), np.log(hi), size=n_pop)))
        starts.append(_pack(D, rng.dirichlet(np.ones(n_pop))))
    return starts


def _start_spread(results, n_pop) -> dict:
    costs = np.array([c for c, _ in results])
    best = costs.min()
    tolerance = abs(best) * 1e-6 + 1e-12
    near_best = [theta for c, theta in results if c <= best + tolerance]
    D_values = np.array([np.sort(_unpack(t, n_pop)[0]) for t in near_best])
    return {
        "n_near_best": len(near_best),
        "D_relative_range": (
            (D_values.max(axis=0) - D_values.min(axis=0)) / np.abs(D_values.mean(axis=0))
        ).tolist(),
    }
