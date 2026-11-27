"""The jump-distance (JD) fit: a zero-mean normal mixture on SIGNED 1-D displacements.

What this is, and why it is not the SQD fit. The SQD fit models the SQUARED displacement
with a mixture of exponentials. This models the SIGNED one-dimensional displacement with
a mixture of zero-mean normals. **They are different observables and different
functionals of the same tracks.** They are never averaged, never reconciled and never
used to justify one another's settings. A JD fraction and an SQD fraction are two
numbers, not one number measured twice.

The model is the legacy model and it is correct.
``getDispl_gui.m`` lines 33-34 produce signed per-axis displacements, ``diff`` of x and
of y in micrometres. ``gaussianCDF.m`` fits a mixture of normals with the means PINNED AT
ZERO, which is right for a signed displacement of a diffusing molecule. The relation
``D = sigma^2 / (2 tau)`` is the one-dimensional Einstein relation and is right for this
observable. None of that is changed here.

What is changed, each confirmed in the MATLAB source before it was touched:

1. ``fit_ECDF.m`` branches on ``n_pop == 2`` and ``n_pop == 3`` only. A single-population
   call leaves ``p``, ``resid`` and ``J`` undefined and MATLAB raises on the next line.
   Here 1, 2 and 3 all run.
2. ``getDispl_gui.m`` lines 27-29 remove outliers with
   ``l1 = d(:,1) > 5*std(d(:,1)); l2 = d(:,2) > 5*std(d(:,2)); l = ~and(l1,l2)``.
   That is one-sided (a -10 sigma step passes), it is an AND so a step must be extreme in
   BOTH axes at once, and the scale is computed on the uncleaned data so an outlier
   inflates the threshold meant to catch it. Here the rule is symmetric, per axis, and
   uses a robust scale. Both counts are reported, and so is what the legacy rule would
   have removed, because the difference is the point.
3. ``sortp.m`` fills the DERIVED fraction's interval with ``max`` of the fitted ones.
   That invents an interval for a parameter that was never estimated. Here the derived
   fraction carries a stated reason instead.
4. ``sortp.m`` then returns ``(1:end-1)`` after sorting by sigma, so the fraction it
   drops is chosen by sort order rather than by which one was derived. Here every
   fraction is returned and the derived one is identified by index.
5. ``nlparci`` on the ECDF support points, with ``1:5:end`` subsampling live in
   ``fit_ECDF_simultaneously.m`` line 52. Here there is no ``nlparci``: the interval comes
   from a bootstrap over independent units, cells or replicate days. The legacy interval
   was measured on the SQD path to be 90 to 131 times too narrow, and it is the same
   construction.
6. ``initializeVars.m:107`` bounds the two fitted fractions of a three-population fit
   each to ``[1e-3, 1]`` and derives the third as ``1 - alpha - beta``, so the third can
   come out NEGATIVE with no error (``docs/v2_v3_settings_audit.md``, section 2, item 7).
   v3 carried this over until 14 Oct 2026. It is not a corner case: on a normal truncated
   at 2 sigma (lighter tails than any mixture of normals, which is what the detection
   ceiling does to fast steps) the old fit returned fractions ``[1.00, 1.00, -1.00]``.
   Here the fractions are STICK-BREAKING, the parameterisation `smtracker.sqd` uses and
   the same helpers (``sqd._pack`` / ``sqd._unpack``): the fitted parameters are sticks
   ``u_k`` in ``[0, 1]`` and ``f_k = u_k * (1 - f_1 - ... - f_{k-1})``, the last taking
   what remains. Every fraction set ``fit_jd`` returns is therefore >= 0 and sums to 1,
   for 1, 2 and 3 populations, and a set that did not would raise. At two populations
   the stick IS the first fraction, so only the lower bound moved (1e-3 to 0). There is
   no legacy mode for this: a negative fraction is not a result to reproduce.

THE UNIT. The fit itself pools steps, because a CDF needs them pooled. **The INTERVAL
does not.** Every interval here comes from resampling cells or days. Pooled steps are
never an interval's unit anywhere in this project.
"""
from __future__ import annotations

from .chart_text import unit_word  # the word shown for a grouping unit

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import erf
from .canonical import without_attrs
# ONE SOURCE OF TRUTH for the simplex: the stick-breaking pair the SQD, appD and
# simultaneous fits already use. Not a copy.
from .sqd import _pack as _sticks_from_fractions, _unpack as _fractions_from_sticks

__all__ = ["JdError", "SUPPORTED_N_POP", "DERIVED_INTERVAL_REASON",
           "signed_displacements", "remove_outliers", "jd_cdf_model", "fit_jd",
           "jd_by_unit"]

#: 1, 2 and 3. The legacy supports 2 and 3 and crashes on 1; the crash is fixed rather
#: than reproduced, so 1 is available here.
SUPPORTED_N_POP = (1, 2, 3)

#: The ECDF points a per-spot fit is made at (`fit_jd`'s ``noise_um2``), as SQD's Fast ECDF.
PER_SPOT_ECDF_POINTS = 2000

#: Written into the derived fraction's interval cell. Never left empty, never propagated.
DERIVED_INTERVAL_REASON = "not fitted: derived as 1 minus the other fractions"

#: The legacy outlier rule, kept only so its effect can be MEASURED against the corrected
#: one. It is never applied to a reported number.
LEGACY_OUTLIER_RULE = (
    "legacy getDispl_gui.m:27-29: keep unless d_x > 5*std(d_x) AND d_y > 5*std(d_y). "
    "One-sided, joint, and the scale is computed on the uncleaned data.")


class JdError(RuntimeError):
    """Raised when a jump-distance fit cannot be produced honestly."""


def signed_displacements(canonical: pd.DataFrame, *, direction: str = "xy",
                         lag_frames: int = 1, calibration=None) -> pd.DataFrame:
    """Signed per-axis displacements in micrometres, one row per step.

    ``direction`` is ``"x"``, ``"y"`` or ``"xy"``. ``"xy"`` stacks both axes into one
    column, which is what the legacy ``displacement.xy`` field does: for an isotropic
    walk the two axes are the same distribution, so stacking doubles the sample without
    changing the model.

    Only steps between frames exactly ``lag_frames`` apart are used. A gap breaks the
    pair rather than being bridged, so no displacement spans missing time.

    PER-SPOT PRECISION (6 Nov 2026): with a ``calibration`` (`precision.Calibration`) each row also carries
    ``noise_um2``, its two spots' localisation variance on one axis, s_a^2 + s_b^2 through the calibration -- the same
    on any axis, the camera's or the cell's, since the error is isotropic; NaN where a spot has none (the fit leaves
    it out). A table without the tracker's precision is refused.
    """
    if direction not in ("x", "y", "xy"):
        raise JdError(f"direction must be x, y or xy, got {direction!r}")
    for column in ("uid", "frame", "x_um", "y_um"):
        if column not in canonical.columns:
            raise JdError(f"not a canonical table (missing {column!r})")
    if calibration is not None:
        from .canonical import LOC_SIGMA_COLUMN

        if LOC_SIGMA_COLUMN not in canonical.columns:
            raise JdError("these tracks carry no per-spot precision (u-track's do; TrackMate's when its spots were "
                          "fitted; track tables do not)")

    rows = []
    ordered = canonical.sort_values(["uid", "frame"], kind="stable")
    for uid, track in without_attrs(ordered).groupby("uid", sort=False, observed=True):
        frames = track["frame"].to_numpy()
        keep = np.diff(frames) == lag_frames
        if not keep.any():
            continue
        dx = np.diff(track["x_um"].to_numpy())[keep]
        dy = np.diff(track["y_um"].to_numpy())[keep]
        noise = None
        if calibration is not None:
            from .canonical import LOC_SIGMA_COLUMN

            s = calibration.true_sigma_um(track[LOC_SIGMA_COLUMN].to_numpy(dtype=float))
            noise = (s[:-1] ** 2 + s[1:] ** 2)[keep]
        first = track.iloc[0]
        for axis, values in (("x", dx), ("y", dy)):
            if direction in (axis, "xy"):
                for i, v in enumerate(values):
                    row = {"uid": uid, "axis": axis, "displacement_um": float(v),
                           "cell": first.get("cell"), "day": first.get("day"),
                           "condition": first.get("condition")}
                    if noise is not None:
                        row["noise_um2"] = float(noise[i])
                    rows.append(row)
    if not rows:
        raise JdError("no contiguous step at this lag")
    return pd.DataFrame(rows)


def remove_outliers(values, *, n_sigma: float = 5.0) -> dict:
    """Symmetric outlier removal on a robust scale. Returns values and both counts.

    The legacy rule is also evaluated, and its count reported beside this one, because
    "the filter is nearly inert" is a claim that should be a measurement.

    The scale is ``1.4826 * MAD`` about the median, which an outlier cannot inflate. The
    cut is on ``|v - median|``, so both tails are treated alike. The legacy cut is on
    ``v > 5*std`` per axis and requires BOTH axes at once, so it is one-sided, joint, and
    computed on a scale the outlier itself moves.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 3:
        raise JdError(f"only {v.size} finite displacements; refusing to filter")
    median = float(np.median(v))
    mad = float(np.median(np.abs(v - median)))
    scale = 1.4826 * mad
    if scale <= 0:
        scale = float(np.std(v))
    keep = np.abs(v - median) <= n_sigma * scale
    # What the legacy rule would have removed, on the same data, one-sided and on the
    # uncleaned standard deviation. The AND across axes cannot be evaluated on a single
    # stacked column, so this is the UPPER BOUND on the legacy removal: the legacy would
    # remove this many at most, and fewer once the joint condition is applied.
    legacy_upper_bound = int((v > 5.0 * float(np.std(v))).sum())
    # which of the GIVEN values were kept (finite and inside the cut), so a value's companions (its noise) follow it
    given = np.asarray(values, dtype=float).ravel()
    mask = np.zeros(given.size, dtype=bool)
    mask[np.flatnonzero(np.isfinite(given))[keep]] = True
    return {
        "values": v[keep],
        "mask": mask,
        "n_in": int(v.size),
        "n_out": int(keep.sum()),
        "n_removed": int(v.size - keep.sum()),
        "legacy_would_remove_at_most": legacy_upper_bound,
        "legacy_rule": LEGACY_OUTLIER_RULE,
        "rule": f"|v - median| <= {n_sigma} * 1.4826 * MAD, symmetric, robust scale",
        "median_um": median, "robust_scale_um": scale,
    }


def jd_cdf_model(params, x, n_pop: int):
    """``sum_k A_k * Phi(x / sigma_k)``, the legacy zero-mean normal mixture.

    ``params`` is ``[sigma_1..sigma_n, A_1..A_{n-1}]``. The last fraction is DERIVED as
    ``1 - sum(A_1..A_{n-1})``, exactly as ``gaussianCDF.m`` does. This is the model on
    FRACTIONS, for evaluating a result; `fit_jd` fits on sticks (defect 6) and evaluates
    the same mixture through `_mixture_cdf`. A fraction set off the simplex (a negative
    fraction, given or derived) is refused here rather than evaluated, because the
    legacy accepted one and that is the defect.
    """
    params = np.asarray(params, dtype=float)
    sigma = params[:n_pop]
    if n_pop == 1:
        fractions = np.array([1.0])
    else:
        head = params[n_pop:n_pop + n_pop - 1]
        fractions = np.append(head, 1.0 - head.sum())
    if np.any(fractions < -1e-12):
        raise JdError(f"fractions {fractions.tolist()} leave the simplex (a fraction is "
                      "negative); a mixture weight cannot be")
    return _mixture_cdf(sigma, fractions, x)


def _mixture_cdf(sigma, fractions, x, groups=None):
    """``sum_k f_k * Phi(x / sigma_k)``: the one evaluation of the mixture CDF. ``groups``: per-spot precision, the
    jumps' noise (`precision.noise_groups`), population k's spread for a jump of noise s then sqrt(sigma_k^2 + s), the
    CDF averaged over the jumps."""
    sigma = np.asarray(sigma, dtype=float)
    if np.any(sigma <= 0):
        raise JdError("a sigma is not positive; the fit cannot be evaluated")
    x = np.asarray(x, dtype=float)
    total = np.zeros_like(x)
    if groups is not None:
        noise, weights = groups
        for weight, s in zip(fractions, sigma):
            spread = np.sqrt(s * s + np.asarray(noise, dtype=float))[None, :]
            total = total + weight * ((0.5 * (1.0 + erf(x[..., None] / (spread * np.sqrt(2.0))))) @ weights)
        return total
    for weight, s in zip(fractions, sigma):
        total = total + weight * 0.5 * (1.0 + erf(x / (s * np.sqrt(2.0))))
    return total


def fit_jd(displacement_um, *, tau_s: float, n_pop: int, n_starts: int = 8,
           seed: int = 0, n_sigma: float = 5.0, noise_um2=None) -> dict:
    """Fit the zero-mean normal mixture and return D per population plus fractions.

    ``D = sigma^2 / (2 * tau)``, the one-dimensional Einstein relation, which is the
    relation the observable calls for.

    A MULTISTART, not the legacy single hand-placed seed. The legacy seed is a fixed
    triple; a single start cannot be shown to have found the best fit, and this project
    treats that as a defect everywhere else.

    THE FRACTIONS ARE A SIMPLEX (defect 6). The parameters are ``[sigma_1..sigma_n,
    u_1..u_{n-1}]`` with stick-breaking ``u`` in ``[0, 1]``, mapped to fractions by the
    same helper `smtracker.sqd` uses, so every returned fraction is >= 0 and they sum to
    1. The starts are the ones the fraction parameterisation had (equal fractions, the
    same sigma draws from the same seed), only expressed as sticks. The last population's
    fraction is still ``1 - sum(others)`` and is still the one named derived.

    No interval is returned here. An interval on a pooled-step fit would have the same
    construction as the legacy `nlparci`, which was measured on the SQD path to be 90 to
    131 times too narrow. Use `jd_by_unit` and resample cells or days.

    PER-SPOT PRECISION (``noise_um2``, one per displacement: `signed_displacements`' with a calibration; 6 Nov 2026):
    population k's spread for a jump of noise s is sqrt(sigma_k^2 + s), the model CDF averaged over the jumps
    (`precision.noise_groups`), still least squares on the ECDF -- the pooled ECDF is the average of the jumps' own
    CDFs. sigma_k is then the diffusion's alone, D = sigma_k^2 / (2 tau) with the error taken out. A jump without a
    noise (a spot without precision) is left out, counted (``n_without_precision``).
    """
    if n_pop not in SUPPORTED_N_POP:
        raise JdError(f"n_pop must be one of {SUPPORTED_N_POP}, got {n_pop}")
    without = 0
    if noise_um2 is not None:
        noise_um2 = np.asarray(noise_um2, dtype=float).ravel()
        given = np.asarray(displacement_um, dtype=float).ravel()
        if noise_um2.size != given.size:
            raise JdError(f"{noise_um2.size} noises for {given.size} displacements: one each")
        has = np.isfinite(noise_um2) & (noise_um2 >= 0)
        without = int((~has).sum())
        displacement_um, noise_um2 = given[has], noise_um2[has]
    cleaned = remove_outliers(displacement_um, n_sigma=n_sigma)
    v = cleaned["values"]
    if v.size < 20:
        raise JdError(f"only {v.size} displacements after filtering; refusing to fit")
    groups = None
    if noise_um2 is not None:
        from .precision import noise_groups

        groups = noise_groups(noise_um2[cleaned["mask"]])

    order = np.sort(v)
    cdf_y = (np.arange(1, order.size + 1, dtype=float) - 0.5) / order.size
    scale = cleaned["robust_scale_um"]
    # PER SPOT, AT 2000 ECDF POINTS (SQD's Fast ECDF): the per-spot CDF costs one term per noise group at every point,
    # so the fit is made at `PER_SPOT_ECDF_POINTS` evenly spaced points of the ECDF, never on a different sample
    at, at_y = order, cdf_y
    if groups is not None and order.size > PER_SPOT_ECDF_POINTS:
        pick = np.unique(np.linspace(0, order.size - 1, PER_SPOT_ECDF_POINTS).astype(int))
        at, at_y = order[pick], cdf_y[pick]

    def residual(theta):
        sigma, fractions = _fractions_from_sticks(theta, n_pop)
        return _mixture_cdf(sigma, fractions, at, groups) - at_y

    rng = np.random.default_rng(seed)
    # Sticks are bounded to [0, 1], as in sqd._bounds; that bound IS the simplex.
    lower = [1e-4] * n_pop + [0.0] * (n_pop - 1)
    upper = [10.0] * n_pop + [1.0] * (n_pop - 1)
    equal_fractions = np.full(n_pop, 1.0 / n_pop)
    best = None
    for start in range(n_starts):
        if start == 0:
            sigmas = np.linspace(0.3, 1.7, n_pop) * scale
        else:
            sigmas = np.sort(rng.uniform(0.15, 2.5, n_pop)) * scale
        sigmas = np.clip(sigmas, lower[:n_pop], upper[:n_pop])
        seed_vector = list(_sticks_from_fractions(sigmas, equal_fractions))
        try:
            result = least_squares(residual, seed_vector, bounds=(lower, upper),
                                   xtol=1e-10, ftol=1e-10, max_nfev=4000)
        except Exception:                              # pragma: no cover - defensive
            continue
        rss = float(result.fun @ result.fun)
        if best is None or rss < best[0]:
            best = (rss, result)
    if best is None:
        raise JdError("every start failed; no fit is available")
    rss, result = best

    sigma, fractions = _fractions_from_sticks(result.x, n_pop)
    # By construction this cannot fire; if it ever does, the fit is refused, not returned.
    if np.any(fractions < 0) or not np.isclose(fractions.sum(), 1.0, rtol=0, atol=1e-9):
        raise JdError(f"fitted fractions {fractions.tolist()} are off the simplex")
    derived_index = n_pop - 1

    by_sigma = np.argsort(sigma)
    D = sigma ** 2 / (2.0 * float(tau_s))
    return {
        "n_pop": int(n_pop),
        "sigma_um": sigma[by_sigma],
        "D_um2_s": D[by_sigma],
        "fraction": fractions[by_sigma],
        "derived_fraction_index": int(np.flatnonzero(by_sigma == derived_index)[0]),
        "derived_interval_reason": DERIVED_INTERVAL_REASON,
        "residual": rss,
        "n_displacements": int(order.size),
        "tau_s": float(tau_s),
        "n_starts": int(n_starts),
        "filter": {k: cleaned[k] for k in
                   ("n_in", "n_out", "n_removed", "legacy_would_remove_at_most",
                    "rule", "legacy_rule", "median_um", "robust_scale_um")},
        "relation": "D = sigma^2 / (2 tau), the one-dimensional Einstein relation",
        "interval": "none here; resample cells or replicates with jd_by_unit",
        "per_spot": groups is not None, "noise_groups": groups, "n_without_precision": without,
    }


def jd_by_unit(canonical: pd.DataFrame, *, tau_s: float, n_pop: int = 2,
               unit: str = "day", direction: str = "xy", **kwargs) -> pd.DataFrame:
    """One JD fit per independent unit. THIS is what an interval may be built from.

    ``unit`` is ``"day"`` or ``"cell"``. The fit inside a unit pools that unit's steps,
    because a CDF needs them pooled; the SPREAD across units is what carries the
    uncertainty.
    """
    if unit not in ("day", "cell"):
        raise JdError(f"unit must be replicate or cell, got {unit!r}")
    steps = signed_displacements(canonical, direction=direction)
    rows = []
    for key, part in without_attrs(steps).groupby(unit, observed=True):
        try:
            fit = fit_jd(part["displacement_um"].to_numpy(), tau_s=tau_s,
                         n_pop=n_pop, **kwargs)
        except JdError:
            continue
        row = {unit: key, "n_pop": n_pop,
               "n_displacements": fit["n_displacements"],
               "n_tracks": int(part["uid"].nunique())}
        for i in range(n_pop):
            row[f"sigma_{i + 1}_um"] = float(fit["sigma_um"][i])
            row[f"D_{i + 1}_um2_s"] = float(fit["D_um2_s"][i])
            row[f"A_{i + 1}"] = float(fit["fraction"][i])
        row["derived_fraction_index"] = fit["derived_fraction_index"]
        row["unit"] = unit
        rows.append(row)
    if not rows:
        raise JdError(f"no {unit_word(unit)} produced a fit")
    return pd.DataFrame(rows)
