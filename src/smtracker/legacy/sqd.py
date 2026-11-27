"""Item 1: the legacy SQD fit. The old design, reproduced. The old defects, fixed.

THE DESIGN, reproduced exactly and not improved:

* squared displacements at ONE time lag;
* every step in a condition is pooled across every cell and every movie;
* the unit is the step. There is no replicate-day unit and no cell unit. That is the
  legacy design and it is correct here. It is not correct anywhere else in this project;
* the cumulative distribution is MATLAB's ``ecdf`` (``sqdFit_.m:46``,
  ``sqdFit_Simultaneous.m:38``): one point per DISTINCT value plus a first point
  ``(min, 0)``, zeros kept, fitted with a sum of exponentials (`matlab_ecdf`);
* the seeds and bounds are v2's own, read from ``smtracker.sqd.V2_SQD_SEEDS``, the one
  place they are written down: per condition ``initializeVars.m:145-155`` (D >= the
  floor ``0.03^2/(4*tau)``, the lower D <= 3 at two populations), simultaneous
  ``sqdFit_Simultaneous.m:41-57`` (D >= 1e-3, fractions >= 0). There is no multistart;
* the intervals come from the nlparci equivalent: on the FULL residual per condition
  (``sqdFit_.m:86-88``), on EVERY FIFTH residual in the simultaneous fit
  (``sqdFit_Simultaneous.m:67-68``), each as v2 computes it;
* there is no bootstrap, no design effect and no BIC.

One, two and three populations. Not four. SMT_GUI/Help_fcn holds singlePopulation.m,
twoPopulation.m and threePopulation.m only, sqdFit_.m branches on 1, 2 and 3, and
sqdFit_Simultaneous.m switches on 1, 2 and 3 with no case 4. A four-population fit
written here would be a new method carrying an old label.

The model carries no localisation precision term. twoPopulation.m is
``msd = 2*ndim*D*tau`` with ndim = 2, so ``msd = 4*D*tau``, and there is no ``4*sigma^2``
added. Legacy D is therefore APPARENT D. The 30 nm in initializeVars.m line 146 enters
as a lower bound on D, ``0.03^2/(4*tau)``, not as a term in the model.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import t as student_t

from ..sqd import LEGACY_PRECISION_UM, V2_SQD_SEEDS, v2_sqd_start
from .deviations import LEGACY_N_POP

__all__ = ["V2_SQD_SEEDS", "LEGACY_D_LOWER_BOUND_UM", "DERIVED_INTERVAL_REASON",
           "SUBSAMPLING_NOTE", "SIMULTANEOUS_SUBSAMPLE_STEP", "INTERVAL_WARNING",
           "INTERVAL_WARNING_SHORT", "interval_unit_label", "matlab_ecdf",
           "legacy_cdf_model", "nlparci_equivalent", "fit_legacy_sqd", "fit_legacy_sqd_simultaneous", "SqdFitError"]

#: initializeVars.m line 146: ``obj.sqd.lb{1} = 0.03^2/(4*obj.time_intervall)``,
#: commented "Diffusion from localization error = lower bound". 30 nm, as a BOUND on D.
#: The same number as ``smtracker.sqd.LEGACY_PRECISION_UM``, and taken from it.
LEGACY_D_LOWER_BOUND_UM = LEGACY_PRECISION_UM

#: ``sqdFit_Simultaneous.m:67``: ``ix = 1:5:length(resid)``, "reduced to improve speed",
#: and passed straight to nlparci at :68. Every fifth residual, starting with the first.
SIMULTANEOUS_SUBSAMPLE_STEP = 5

#: Printed in the interval cell of the DERIVED fraction. The cell is never left empty,
#: and the interval is never propagated from the Jacobian, because the parameter was not
#: fitted: it is 1 minus the others by construction.
DERIVED_INTERVAL_REASON = "not fitted: derived as 1 minus the other fractions"

#: Reported on every run. Two fits, two behaviours, both reproduced as v2 runs them.
SUBSAMPLING_NOTE = (
    "Per condition, sqdFit_.m computes an interval from the 1:5:end subsampled residual "
    "at lines 81 to 83, then OVERWRITES it at lines 86 to 88 with the full-residual "
    "interval, so the per-condition interval is the FULL residual's, and that is what is "
    "reproduced there. The SIMULTANEOUS fit, sqdFit_Simultaneous.m, which v2 runs by "
    "default for 2 and 3 populations, keeps the subsample: lines 67 to 68 pass every "
    "fifth residual and Jacobian row to nlparci, and that is reproduced there. It "
    "changes the interval only, never a D or a fraction, and makes it wider by about "
    "the square root of 5 than the full residual would.")


#: Printed beside EVERY legacy interval, in the table and on the figure. Measured on
#: seeded data with a known answer, so it is a result and not an opinion.
INTERVAL_WARNING = (
    "THE POINT ESTIMATES ARE SOUND. THE INTERVAL IS NOT. Measured over 200 seeded "
    "datasets with a known answer: the mean fitted value is consistent with no bias, "
    "with any bias bounded to within 0.34 per cent of the true value for D_1, 0.16 per "
    "cent for D_2 and 0.26 per cent for A_1 at 95 per cent. Over the same 200 runs the "
    "reported interval half-width was 90 to 131 times smaller than the actual spread of "
    "the fitted value, and the interval covered the true value in 3 to 4 runs out of "
    "200 where 190 are expected. This is a failure of the INTERVAL ONLY, not of the "
    "estimator. The cause is the design: nlparci on pooled steps treats correlated "
    "observations as independent. BEST CASE: those datasets had independent steps and "
    "no track structure, so real coverage is worse. No corrected interval is offered, "
    "because the project design effect of 2.68 was measured for a different statistic "
    "and is not assumed to transfer. See "
    "legacy_out/tables/legacy_sqd_interval_coverage.csv."
)

#: One short line for the figure. The full sentence lives in the caption and in the
#: table column, so the figure stays readable and nothing is lost.
INTERVAL_WARNING_SHORT = (
    "interval coverage 3 to 4 of 200 seeded runs against 190 expected; point estimates "
    "consistent with no bias")


def interval_unit_label(n_steps: int) -> str:
    """The unit and the population of a legacy interval, in one string.

    Every legacy interval carries this wherever it is printed. The legacy unit is the
    pooled step. Steps inside one track are not independent, so this interval is not a
    replicate-day interval and must never be read as one.
    """
    return (f"unit: pooled step; population: {int(n_steps)} steps pooled over every "
            "cell and every movie in the condition; NOT a replicate-day interval")


class SqdFitError(RuntimeError):
    """Raised when a legacy SQD fit cannot be produced honestly."""


def matlab_ecdf(values):
    """MATLAB's ``[f, x] = ecdf(y)`` for uncensored data, as v2 calls it.

    ``x`` is every DISTINCT value in ascending order with the smallest repeated once in
    front, and ``f`` is ``0`` there and then the share of values at or below each distinct
    value: for ``y = [3 1 2 2]``, ``x = [1 1 2 3]`` and ``f = [0 .25 .75 1]``. So the curve
    v2 fits has one point more than it has distinct values, a first point at ``(min, 0)``,
    and ties merged. NaN is ignored, as ``ecdf`` ignores it; a squared displacement of 0
    is KEPT, as ``ecdf`` keeps it.
    """
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        raise SqdFitError("no finite squared displacement to build a cumulative curve")
    distinct, counts = np.unique(v, return_counts=True)
    f = np.cumsum(counts) / v.size
    return (np.concatenate([[distinct[0]], distinct]),
            np.concatenate([[0.0], f]))


def _usable(values, what: str):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if np.any(v < 0):
        raise SqdFitError(f"{what}: a squared displacement is negative, which cannot be")
    if v.size < 10:
        raise SqdFitError(f"{what}: only {v.size} usable steps; refusing to fit")
    return v


def _clip_seeds(seeds, lower, upper):
    """A start outside its own box, moved onto the box, and the indices that moved.

    MATLAB's lsqnonlin moves an out-of-box start into the box without a word; scipy
    raises instead. The exact interior point MATLAB picks is not reproduced (it cannot be
    checked without MATLAB); the converged fit does not depend on it on this data. The
    move is REPORTED, because a seed that cannot be used was never really a choice.
    """
    asked = list(seeds)
    moved = [min(max(v, lo), hi) for v, lo, hi in zip(asked, lower, upper)]
    return moved, asked, [i for i, (a, b) in enumerate(zip(asked, moved)) if a != b]


def legacy_cdf_model(params, tau_s: float, x, n_pop: int):
    """``P(r^2 < x) = 1 - sum_k A_k exp(-x / (4 D_k tau))``, the legacy form.

    ``params`` is ``[D_1..D_n, A_1..A_{n-1}]``. The last fraction is DERIVED as
    ``1 - sum(A_1..A_{n-1})``, exactly as twoPopulation.m does with ``beta = 1 - alpha``
    and threePopulation.m with ``gamma = 1 - alpha - beta``.
    """
    params = np.asarray(params, dtype=float)
    D = params[:n_pop]
    if n_pop == 1:
        fractions = np.array([1.0])
    else:
        head = params[n_pop:n_pop + n_pop - 1]
        fractions = np.append(head, 1.0 - head.sum())
    x = np.asarray(x, dtype=float)
    msd = 2 * 2 * D * float(tau_s)                 # ndim = 2, so msd = 4 * D * tau
    if np.any(msd <= 0):
        raise SqdFitError("a state mean is not positive; the fit cannot be evaluated")
    total = np.zeros_like(x)
    for weight, mean in zip(fractions, msd):
        total = total + weight * np.exp(-x / mean)
    return 1.0 - total


def nlparci_equivalent(params, residuals, jacobian, *, confidence: float = 0.95):
    """The nlparci equivalent: ``p +/- t(alpha/2, dof) * sqrt(diag(inv(J'J)) * mse)``.

    This is the LEGACY interval and it is reproduced, not replaced. It treats every
    pooled step as an independent observation, which they are not, so it is far too
    narrow. That is a property of the legacy design, and legacy mode shows the legacy
    design. No bootstrap is computed anywhere in this module.
    """
    params = np.asarray(params, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    jacobian = np.asarray(jacobian, dtype=float)
    n, k = jacobian.shape
    dof = max(n - k, 1)
    mse = float(residuals @ residuals) / dof
    try:
        covariance = np.linalg.inv(jacobian.T @ jacobian) * mse
    except np.linalg.LinAlgError:
        return (np.full(k, np.nan), np.full(k, np.nan))
    sigma = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    half = student_t.ppf(0.5 + confidence / 2.0, dof) * sigma
    return (params - half, params + half)


def fit_legacy_sqd(sq_disp_um2, *, tau_s: float, n_pop: int, n_tracks: int,
                   confidence: float = 0.95) -> dict:
    """Fit the pooled squared displacements of one condition. Returns a plain dict.

    ``sq_disp_um2`` is every step in the condition, pooled. Nothing is weighted and
    nothing is grouped, because the legacy unit is the step.

    The two defects are fixed here.

    * Every interval travels with its own parameter through the sort by D. The legacy
      reads intervals from a shared buffer that grows once per condition and is never
      re-sliced, so condition 2 reads at condition 1's offsets.
    * ``n_pop = 1`` runs. It has one parameter, D. It has no fraction and no second D,
      so nothing is indexed past the end of the parameter vector.
    """
    if n_pop not in LEGACY_N_POP:
        raise SqdFitError(
            f"n_pop must be one of {LEGACY_N_POP}, got {n_pop}. The legacy source "
            "supports 1, 2 and 3 populations only, so legacy mode does too. A "
            "four-population fit would be a new method carrying an old label.")
    values = _usable(sq_disp_um2, "legacy SQD")
    cdf_x, cdf_y = matlab_ecdf(values)

    # v2's OWN per-condition seeds and bounds (initializeVars.m:145-155), from the one
    # table. At 3 populations the first seed, 0.01 um^2/s, lies below the floor
    # 0.03^2/(4*tau) = 0.01125 at tau = 20 ms; it is moved onto the bound and reported.
    seeds, lower, upper = v2_sqd_start(n_pop, tau_s, fit="per_condition")
    floor = float(lower[0])
    seeds, seeds_asked, seeds_clipped = _clip_seeds(seeds, lower, upper)

    def residual(p):
        return legacy_cdf_model(p, tau_s, cdf_x, n_pop) - cdf_y

    result = least_squares(residual, seeds, bounds=(lower, upper),
                           xtol=1e-8, ftol=1e-8, max_nfev=3000)
    lo, hi = nlparci_equivalent(result.x, result.fun, result.jac,
                                confidence=confidence)

    D = np.asarray(result.x[:n_pop], dtype=float)
    if n_pop == 1:
        fractions = np.array([1.0])
        frac_lo = np.array([np.nan])
        frac_hi = np.array([np.nan])
        derived_index = 0
    else:
        head = np.asarray(result.x[n_pop:], dtype=float)
        fractions = np.append(head, 1.0 - head.sum())
        frac_lo = np.append(np.asarray(lo[n_pop:]), np.nan)
        frac_hi = np.append(np.asarray(hi[n_pop:]), np.nan)
        derived_index = n_pop - 1

    # WHICH PARAMETERS LANDED ON A BOUND. Both counts are reported, never their
    # difference. A parameter at a bound is not an estimate: the optimiser was stopped
    # by the box, not by the data.
    tol = 1e-9
    at_lower = [i for i, (v, b) in enumerate(zip(result.x, lower)) if v - b <= tol]
    at_upper = [i for i, (v, b) in enumerate(zip(result.x, upper)) if b - v <= tol]

    by_D = np.argsort(D)
    derived_after_sort = int(np.flatnonzero(by_D == derived_index)[0])
    return {
        "n_pop": int(n_pop),
        "D": D[by_D],
        "D_lo": np.asarray(lo[:n_pop])[by_D],
        "D_hi": np.asarray(hi[:n_pop])[by_D],
        "fraction": fractions[by_D],
        "fraction_lo": frac_lo[by_D],
        "fraction_hi": frac_hi[by_D],
        "derived_fraction_index": derived_after_sort,
        "derived_interval_reason": DERIVED_INTERVAL_REASON,
        "residual": float(result.fun @ result.fun),
        "n_steps": int(values.size),
        "n_cdf_points": int(cdf_x.size),
        "n_tracks": int(n_tracks),
        "cdf_x": cdf_x,
        "cdf_y": cdf_y,
        "tau_s": float(tau_s),
        "seeds": list(seeds),
        "seeds_requested": seeds_asked,
        "seed_source": "SMT_GUI/@SMT/initializeVars.m:145-155 (smtracker.sqd.V2_SQD_SEEDS)",
        "seeds_clipped_to_bound": seeds_clipped,
        "seed_clip_note": (
            f"v2's seed {[seeds_asked[i] for i in seeds_clipped]} lies outside its own "
            f"bounds (the D floor 0.03^2/(4*tau) = {floor:.5g} um^2/s here). MATLAB moves "
            "a start into the box silently. The move is reproduced and the indices are "
            "reported." if seeds_clipped else "no seed needed clipping"),
        "D_lower_bound_um2_s": float(floor),
        "n_at_lower_bound": len(at_lower),
        "n_at_upper_bound": len(at_upper),
        "n_parameters": int(len(result.x)),
        "index_at_lower_bound": at_lower,
        "index_at_upper_bound": at_upper,
        "lower_bounds": [float(v) for v in lower],
        "upper_bounds": [float(v) for v in upper],
        "interval_unit": interval_unit_label(int(values.size)),
        "interval_warning": INTERVAL_WARNING,
        "design_effect_applied": (
            "none. The project design effect of 2.68 was measured for a DIFFERENT "
            "statistic on this data and is not assumed to transfer. No corrected "
            "interval is offered."),
        "interval_method": ("nlparci equivalent on the FULL residual vector "
                            "(sqdFit_.m:86-88)"),
        "subsampling_note": SUBSAMPLING_NOTE,
    }


def fit_legacy_sqd_simultaneous(steps_by_unit, *, tau_s: float, n_pop: int,
                                confidence: float = 0.95) -> dict:
    """The legacy SIMULTANEOUS fit: ONE set of D shared across units, fractions per unit.

    This is ``sqdFit_Simultaneous.m``. Its loop over "conditions" is over the units the
    caller supplies; in the legacy that is replicate days of one protein and condition.
    The design exists because **fractions from fits with different D are not the same
    quantity**, so sharing D is what makes per-unit fractions comparable.

    ``steps_by_unit`` maps a unit label to its pooled squared displacements.

    What sharing asserts. A shared D says the units differ in how many molecules are in
    each state and not in how fast either state moves. That is a constraint, not an
    observation. When it is false the shared D lands between the true values, and each
    unit's fraction moves to absorb its own half of the mismatch, in opposite
    directions, so the fraction spread GROWS. A larger fraction difference under sharing
    is therefore not evidence that sharing is right.

    The two defects are fixed here, as in the independent fit: each unit's interval is
    sliced at its own offset rather than from a buffer that grows once per unit, and the
    derived fraction carries a reason instead of a copied interval.
    """
    if n_pop not in LEGACY_N_POP:
        raise SqdFitError(f"n_pop must be one of {LEGACY_N_POP}, got {n_pop}")
    units = list(steps_by_unit)
    if len(units) < 2:
        raise SqdFitError("a simultaneous fit needs at least two units")

    curves, n_steps = {}, {}
    for unit in units:
        v = _usable(steps_by_unit[unit], f"unit {unit!r}")
        curves[unit] = matlab_ecdf(v)
        n_steps[unit] = int(v.size)

    n_free = n_pop - 1
    # v2's SIMULTANEOUS literals (sqdFit_Simultaneous.m:41-57), from the one table: D in
    # [1e-3, 10] -- NOT the floor the per-condition fit uses -- and fractions in [0, 1].
    seeds, lower, upper = v2_sqd_start(n_pop, tau_s, fit="simultaneous",
                                       n_units=len(units))
    seeds, seeds_asked, seeds_clipped = _clip_seeds(seeds, lower, upper)

    def residual(p):
        D = p[:n_pop]
        out = []
        for j, unit in enumerate(units):
            head = p[n_pop + j * n_free: n_pop + (j + 1) * n_free] if n_free else []
            x, y = curves[unit]
            out.append(legacy_cdf_model(np.concatenate([D, head]), tau_s, x, n_pop) - y)
        return np.concatenate(out)

    result = least_squares(residual, seeds, bounds=(lower, upper),
                           xtol=1e-8, ftol=1e-8, max_nfev=6000)
    # THE SUBSAMPLE IS LIVE HERE (sqdFit_Simultaneous.m:67-68): every fifth residual and
    # Jacobian row, starting with the first, go to nlparci. Unlike sqdFit_.m nothing
    # overwrites it, so this is the interval v2 printed. The interval only.
    step = SIMULTANEOUS_SUBSAMPLE_STEP
    lo, hi = nlparci_equivalent(result.x, result.fun[::step], result.jac[::step],
                                confidence=confidence)

    D = np.asarray(result.x[:n_pop], dtype=float)
    by_D = np.argsort(D)
    per_unit = {}
    for j, unit in enumerate(units):
        head = np.asarray(result.x[n_pop + j * n_free: n_pop + (j + 1) * n_free],
                          dtype=float)
        fractions = np.append(head, 1.0 - head.sum()) if n_free else np.array([1.0])
        # EACH UNIT'S INTERVAL AT ITS OWN OFFSET. The legacy reads from a shared buffer
        # that grows once per unit and is never re-sliced, so unit 2 reads unit 1's.
        f_lo = np.append(np.asarray(lo[n_pop + j * n_free: n_pop + (j + 1) * n_free]),
                         np.nan) if n_free else np.array([np.nan])
        f_hi = np.append(np.asarray(hi[n_pop + j * n_free: n_pop + (j + 1) * n_free]),
                         np.nan) if n_free else np.array([np.nan])
        derived = n_pop - 1
        per_unit[unit] = {
            "fraction": fractions[by_D],
            "fraction_lo": f_lo[by_D], "fraction_hi": f_hi[by_D],
            "derived_fraction_index": int(np.flatnonzero(by_D == derived)[0]),
            "n_steps": n_steps[unit],
            "n_cdf_points": int(curves[unit][0].size),
        }
    return {
        "n_pop": int(n_pop), "units": units,
        "D": D[by_D], "D_lo": np.asarray(lo[:n_pop])[by_D],
        "D_hi": np.asarray(hi[:n_pop])[by_D],
        "per_unit": per_unit,
        "residual": float(result.fun @ result.fun),
        "n_residual_points": int(result.fun.size),
        "n_residual_points_in_interval": int(result.fun[::step].size),
        "tau_s": float(tau_s),
        "seeds": list(seeds), "seeds_requested": seeds_asked,
        "seed_source": ("SMT_GUI/@SMT/sqdFit_Simultaneous.m:41-57 "
                        "(smtracker.sqd.V2_SQD_SEEDS)"),
        "seeds_clipped_to_bound": seeds_clipped,
        "lower_bounds": [float(v) for v in lower],
        "upper_bounds": [float(v) for v in upper],
        "derived_interval_reason": DERIVED_INTERVAL_REASON,
        "interval_unit": interval_unit_label(sum(n_steps.values())),
        "interval_method": (
            f"nlparci equivalent on every {step}th residual and Jacobian row, 1:{step}:end "
            "(sqdFit_Simultaneous.m:67-68), as v2 computes it"),
        "subsampling_note": SUBSAMPLING_NOTE,
        "interval_warning": INTERVAL_WARNING,
        "shared": "D is shared across the units; fractions are per unit",
    }
