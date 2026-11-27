"""Aggregation units, the simultaneous fit, and condition tests on the correct N.

The single most dangerous number in this analysis is ``n``. A dataset with 2005
tracks yields ~13 500 displacements, and a test run on those has ``n = 13500``
which is **pseudoreplication**. Steps within a track are correlated, tracks
within a cell share its environment, cells within a day share the culture and
the alignment. The independent unit is the **cell** or the **replicate day**,
not the step, so that is what a comparison must be built on. With 3 replicate
days, ``n = 3``.

This module therefore does three things:

1. **The simultaneous fit**, ``D`` shared across units, fractions fitted per
   unit. This is the legacy "simultaneous" SQD fit (see ``CLAUDE.md``: the loop
   over "conditions" there is over *replicate days* of one protein+condition).
   Sharing ``D`` is what makes per-unit fractions comparable at all: fractions
   from fits with different ``D`` are not the same quantity.
2. **Aggregation over a selectable unit**, pooled / movie / cell / day.
3. **Condition comparison on unit-level values**, with the effect size, an
   interval, and ``n`` stated as the number of units. A permutation test is the
   default because ``n`` is small and normality is not credible at ``n = 3``.

Legacy bug this replaces
------------------------
Fraction error bars were misassigned across conditions/days, and uncertainties
came from ``nlparci`` on correlated CDF points. Here the spread across units is
the uncertainty, and it is attached to the unit it came from.

THE RULE: no SIGNIFICANCE from pooled steps, but distances are fine
-------------------------------------------------------------------
The prohibition is about **the data a p-value is computed from**, not about any
particular test's name. No significance test of any kind (Mann-Whitney, KS,
t-test, permutation, or one invented next year) may be run on pooled
displacements. Naming tests individually would leave the footgun reachable under
the next name, so the guard is **structural**: ``compare_conditions`` accepts
only per-unit summaries and *rejects a raw observable outright*. There is no
code path from pooled steps to a p-value.

Two independent reasons, either sufficient:

1. **The n is wrong.** Pooled steps are pseudoreplicated; measured
   false-positive rate 65 % where 5 % is correct (``tests/test_aggregate.py``,
   using Mann-Whitney as the worked example).
2. **The question is wrong.** Such tests ask whether two distributions are
   *exactly identical*. At n in the thousands they reject on biologically
   trivial differences, so "significant" carries no information about whether
   anything meaningful changed.

**Descriptive distances on pooled data are NOT banned**, and it would cripple
them to pretend otherwise. A distance answers "how far apart", not "is it real",
so no null hypothesis and no ``n`` is involved in the point estimate. Hence:

- **significance** → ``compare_conditions`` (permutation over units);
- **magnitude, model-based** → difference in fitted fractions + bootstrap
  interval (also ``compare_conditions``);
- **magnitude, model-free** → ``wasserstein_effect_size``, computed on the
  **pooled** distributions, which is legitimate and is the point; only its
  *interval* comes from resampling units;
- **is pooling even safe?** → ``poolability``, which asks whether days differ
  from each other as much as conditions do;
- **for the eye** → ``cdf_overlay``, one curve per unit.

The pooled-step p-value failure is kept as a *demonstration* in the test suite,
the justification for the rule, never as a callable function.

Fail loud
---------
- Fewer than 2 units in a group: no comparison is possible, so it raises rather
  than returning a p-value computed from nothing.
- A unit that cannot be fitted is reported, not silently dropped.
- Testing on pooled steps is not offered. There is no argument for it.
"""

from __future__ import annotations

from .chart_text import unit_word  # the word shown for a grouping unit

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .observables import displacement_weights, lag_labels
from .sqd import (
    D_BOUNDS,
    SQDError,
    _bounds,
    _log_likelihood,
    _pack,
    _unpack,
    ecdf_r_squared,
    fit_sqd,
    with_immobile,
    zero_jumps,
)

__all__ = [
    "AggregationError",
    "ComparisonResult",
    "AGGREGATION_UNITS",
    "fit_by_unit",
    "fit_simultaneous",
    "aggregate_fractions",
    "compare_conditions",
    "compare_two_conditions",
    "wasserstein_effect_size",
    "poolability",
    "cdf_overlay",
    "nested_spread",
    "layer_report",
    "LAYERS",
]

#: Selectable aggregation units, coarsest last. ``"pooled"`` is a single unit
#: and therefore carries NO uncertainty: it exists for a point estimate only.
AGGREGATION_UNITS = ("pooled", "movie", "cell", "day")

_MIN_UNITS_FOR_COMPARISON = 2


class AggregationError(RuntimeError):
    """Raised when units cannot be formed, fitted or compared."""


def require_real_days(frame, *, day_column: str = "day") -> int:
    """Refuse day-level statistics when the data has no real day structure.

    A single imaging session carries one ``day`` value for every record. Treating
    that lone day as a replicate (a per-day fit, a day bootstrap, a "n = 1 day"
    caption) FABRICATES a replicate structure the experiment does not have, and
    the resulting interval is anti-conservative for a between-day claim (within-
    session variance is a subset of true biological variance). Pseudo-days (thirds
    of one session) are the same fabrication wearing three hats.

    So every day-level path calls this first: it returns the number of distinct
    days when there are ≥ 2, and RAISES otherwise. The remedy is genuine replicate
    days (bench work), or ``unit="cell"`` with the single-session caveat stated,
    never a day-level number computed from one day. This is the one source of
    truth for that refusal.
    """
    if day_column not in getattr(frame, "columns", ()):
        raise AggregationError(
            f"no {day_column!r} column, so replicate-level statistics cannot be formed")
    n_days = int(frame[day_column].nunique(dropna=True))
    if n_days < 2:
        raise AggregationError(
            f"replicate-level statistics refused: {n_days} distinct replicate in this data. "
            "A single session has no replicate structure to resample. Use unit='cell' "
            "and report a single session, or add replicates.")
    return n_days


@dataclass
class ComparisonResult:
    """A condition comparison built on independent units."""

    condition_a: str
    condition_b: str
    unit: str
    quantity: str
    mean_a: float
    mean_b: float
    difference: float               # b − a
    ci_low: float
    ci_high: float
    p_value: float
    n_a: int
    n_b: int
    test: str
    values_a: np.ndarray
    values_b: np.ndarray

    @property
    def n_units(self) -> int:
        return self.n_a + self.n_b

    def summary(self) -> str:
        return (
            f"{self.quantity}: {self.condition_a} {self.mean_a:.3f} "
            f"(n = {self.n_a} {self.unit}s) vs {self.condition_b} "
            f"{self.mean_b:.3f} (n = {self.n_b}); difference "
            f"{self.difference:+.3f} [{self.ci_low:+.3f}, {self.ci_high:+.3f}], "
            f"p = {self.p_value:.3g} ({self.test}, n is {self.unit}s: NOT steps)"
        )


def fit_by_unit(
    observable: pd.DataFrame,
    *,
    n_pop: int,
    unit: str = "cell",
    shared_D: bool = True,
    min_observations: int = 50,
    group_columns=("condition",),
    **fit_kwargs,
) -> pd.DataFrame:
    """Fit each unit, optionally sharing ``D`` across them (the simultaneous fit).

    Parameters
    ----------
    observable
        Squared displacements from ``observables.squared_displacements``.
    n_pop
        Number of populations. Shared across all units by construction,
        per-unit state counts would make the fractions incomparable.
    unit
        ``"cell"``, ``"movie"``, ``"day"`` or ``"pooled"``.
    shared_D
        ``True`` (default) fits ``D`` once on the pooled data and then fits only
        the fractions per unit: the legacy simultaneous fit, and the only way
        per-unit fractions mean the same thing. ``False`` fits each unit
        independently, which is diagnostic but not directly comparable.
    min_observations
        Units with fewer displacements are reported with ``ok = False`` rather
        than silently dropped.
    group_columns
        Columns carried through to the result (default ``("condition",)``).

    Returns
    -------
    pandas.DataFrame
        One row per unit: the grouping columns, ``n_observations``,
        ``fraction_k``, ``D_k`` and ``ok``.
    """
    if unit not in AGGREGATION_UNITS:
        raise AggregationError(f"unit must be one of {AGGREGATION_UNITS}, got {unit!r}")
    if "sq_disp_um2" not in observable.columns:
        raise AggregationError("observable has no 'sq_disp_um2' column")
    tau_lag_s = observable.attrs.get("lag_time_s")
    if tau_lag_s is None:
        raise AggregationError(
            "observable carries no lag time in .attrs; τ must not be guessed"
        )

    group_columns = [c for c in group_columns if c in observable.columns]

    if unit == "pooled":
        keys = pd.Series(["pooled"] * len(observable), index=observable.index)
    else:
        if unit not in observable.columns:
            raise AggregationError(f"observable has no '{unit}' column")
        if observable[unit].isna().any():
            raise AggregationError(
                f"'{unit}' is unassigned for some displacements; aggregating over "
                "it would silently drop data."
            )
        if unit == "day":
            require_real_days(observable)          # no per-day fit from one session
        keys = observable[unit]

    # Read once, so the pooled fit and every per-unit fit cannot disagree about it.
    weighting = fit_kwargs.get("weighting", "step")
    precision = fit_kwargs.get("precision")
    if precision is not None:
        from .precision import step_noise, usable_steps

        # the share of steps without a precision refused ONCE, on every unit together; a unit (a cell has a few
        # dozen steps) then leaves its own few out, never refusing the Run (7 Nov 2026)
        usable_steps(observable)

    shared = None
    if shared_D:
        # an immobile population of exact-zero jumps in the pooled fit is a D of 0 here, and `_fit_fractions`
        # counts every unit's own zeros into it (`sqd`'s module head)
        pooled = fit_sqd(observable, n_pop=n_pop, **fit_kwargs)
        shared = pooled.D
    elif (observable["sq_disp_um2"] == 0).any() and fit_kwargs.get("zero_jumps", "auto") == "auto":
        # EVERY UNIT FITTED ALONE HOLDS THE POOLED DATA'S POPULATIONS: the exact-zero rule is decided once, on all
        # of it, so a unit without a zero still has the immobile population (at share 0) and fraction_1 is one
        # population in every row
        values = observable["sq_disp_um2"].to_numpy(dtype=float)
        decided = zero_jumps(values, displacement_weights(observable, weighting),
                             lag_labels(observable), weighting=weighting)
        fit_kwargs = {**fit_kwargs, "zero_jumps": "immobile" if decided.immobile else "kept"}

    rows = []
    for key, frame in observable.groupby(keys, observed=True):
        values = frame["sq_disp_um2"].to_numpy(dtype=float)
        row = {unit: key, "n_observations": len(values)}
        for column in group_columns:
            unique = pd.unique(frame[column])
            if len(unique) > 1:
                raise AggregationError(
                    f"{unit_word(unit)} '{key}' spans several {column} values ({list(unique)}); "
                    "units must nest inside groups."
                )
            row[column] = unique[0]

        if len(values) < min_observations:
            row["ok"] = False
            rows.append(row)
            continue

        noise = None
        if precision is not None:
            # PER-SPOT PRECISION: this unit's own steps' noise; a step with a spot without a precision left out
            ok = usable_steps(frame, max_missing=None)
            frame = frame.loc[ok]
            values = frame["sq_disp_um2"].to_numpy(dtype=float)
            noise = step_noise(frame, precision)
        try:
            if shared is not None:
                # THE WEIGHTING MUST REACH THE PER-UNIT FIT TOO, and for a long time it
                # did not. `shared_D=True` fitted D on the pooled data with whatever
                # weighting the caller asked for and then fitted every unit's FRACTIONS
                # on a bare array, i.e. step-weighted, whatever was asked. So
                # `fit_by_unit(shared_D=True, weighting="track")` returned step-weighted
                # fractions sitting on a track-weighted D - two conventions in one row,
                # silently. Measured on one lab condition: the track-weighted and
                # step-weighted fractions differed by several points, and the pooled fit and the mean of
                # the units disagreed by exactly that amount and nothing said why.
                # `_fit_fractions` has taken `sample_weights` all along and
                # `comparison.py` already passes them; this was the one site that did not.
                sample_weights = (None if weighting == "step"
                                  else displacement_weights(frame, weighting))
                fractions = _fit_fractions(values, shared, float(tau_lag_s),
                                           sample_weights, step_noise=noise)
                D = shared
            else:
                # under per-spot precision the unit's TABLE goes in: the fit reads each step's spots from it
                fit = fit_sqd(frame if precision is not None else values, n_pop=n_pop, tau_lag_s=tau_lag_s,
                              **fit_kwargs)
                D, fractions = fit.D, fit.fractions
        except SQDError:
            row["ok"] = False
            rows.append(row)
            continue

        row["ok"] = True
        for i, (d, f) in enumerate(zip(D, fractions), start=1):
            row[f"D_{i}"] = d
            row[f"fraction_{i}"] = f
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        raise AggregationError(f"no {unit_word(unit)} units could be formed")
    out.attrs["unit"] = unit
    out.attrs["shared_D"] = bool(shared_D)
    out.attrs["n_pop"] = n_pop
    out.attrs["n_units_total"] = len(out)
    out.attrs["n_units_ok"] = int(out["ok"].sum())
    out.attrs["n_units_failed"] = int((~out["ok"]).sum())
    return out


def joint_zero_jumps(observable, weighting: str = "step"):
    """The exact-zero decision (`sqd.zero_jumps`) of ONE joint fit over ``observable``'s pooled jumps, or None
    without an exact zero. Decided under the fit's weighting, as the rule reads the share (a uid is one condition's).
    The ONE place the joint fit (`fit_simultaneous`) and the comparison's payloads (`comparison._global_zero_jumps`,
    which carries it so the Whole data set bubbles say the note the other scopes say) make it."""
    values = observable["sq_disp_um2"].to_numpy(float)
    if not (values == 0).any():
        return None
    return zero_jumps(values, displacement_weights(observable, weighting), lag_labels(observable),
                      weighting=weighting)


def fit_simultaneous(
    observable: pd.DataFrame,
    *,
    n_pop: int,
    by: str = "condition",
    n_starts: int = 60,
    seed: int | None = 0,
    D_bounds=D_BOUNDS,
    weighting: str = "step",
    n_jobs: int | None = 1,
    ecdf_max_points: int | None = None,
    gradient: str | None = None,
    precision=None,
) -> pd.DataFrame:
    """Shared ``D`` across GROUPS, fractions per group: the legacy benchmark fit.

    PER-SPOT PRECISION (``precision``, a `precision.Calibration`; 6 Nov 2026, asked: "the spot precision stuff for
    simultanous fits, everywhere its possible"): each step's own noise in its population's mean, 4 D_k tau + 2 noise_i.
    There is then no one curve per group for least squares to fit, so the joint fit is MAXIMUM LIKELIHOOD
    (`_fit_simultaneous_per_spot`): the shared D and each group's fractions, every group's log-likelihood scaled to
    the same total weight, so each group has the equal say in D the joint least squares gives it (below). Steps
    lacking a spot's precision are left out and counted, as `sqd.fit_sqd`'s; lags fitted together and an immobile
    population of exact-zero jumps are refused, as there. The table gains ``attrs["per_spot_precision"]``.

    ``ecdf_max_points`` thins EACH GROUP'S ECDF to about that many points (`observables.fit_ecdf`), as every
    per-condition fit is thinned to the Run's budget; None fits every point. Since 4 Nov 2026 the Run passes its own
    ECDF setting (`comparison._global_joint_D`): the joint fit had fitted every point whatever the setting said, while
    the receipt said "2000 points" -- under Whole data set the D on every bubble.

    ``gradient`` is the optimiser's, as `sqd.fit_sqd`'s: None reads the Run's (`sqd.gradient_mode`, in the calling
    thread, resolved here so the pool's workers are given it); ``"analytic"`` hands every start the EXACT Jacobian
    (`_simultaneous_jacobian`, each group's block of `sqd._cdf_jacobian`), ``"numeric"`` finite differences. Since
    4 Nov 2026 (ruled): the joint fit used finite differences whatever the Run's Optimiser said.

    ``n_jobs`` runs the starts side by side (`parallel.bootstrap_map`; None = every core but one). THE NUMBERS NEVER
    DEPEND ON IT (4 Nov 2026): every start's guess is drawn up front, in the order the one loop drew them, and the
    best is kept in start order, so a pool gives this fit exactly. Measured on a lab project's four conditions under
    Whole data set (3 populations, 50 starts, 80,742 ECDF points): 377 s one start after another.

    This is a **joint** least-squares over every group's ECDF at once, with the
    parameter vector packed exactly as ``CLAUDE.md`` documents:
    ``[D_1..D_n, then per-group fraction blocks]``. That matters, and it is not
    the same as the obvious shortcut:

    - **joint (this)**: each group contributes its own ECDF as a residual
      block, so every group has an equal say in where the shared ``D`` lands;
    - **pool-then-split** (fit ``D`` on the concatenated displacements, then fit
      fractions per group): weights each group by its *step count*, so the
      larger condition effectively sets ``D`` for both.

    On the reference comparison those differ visibly: one condition contributes several times
    the steps of the other, so pooling lets it decide ``D`` and the
    fractions shift by several points. The legacy tool fits jointly; so does
    this.

    Distinct from ``fit_by_unit``, and the distinction matters:

    - ``fit_by_unit`` fits **units** (cells / days) *inside* a group. Units are
      where bootstrap intervals and significance come from.
    - this fits **groups** (conditions / proteins) against one shared ``D``. It
      is the comparison itself, not a source of uncertainty.

    Sharing ``D`` is what makes per-group fractions comparable at all: a
    fraction from a fit with ``D = 0.05`` and one from a fit with ``D = 0.09``
    answer different questions. The cost is that a genuine ``D`` difference is
    absorbed into the fractions, so run the free-``D`` fit alongside (as the
    legacy benchmark does) and report both.
    """
    from .observables import displacement_weights, fit_ecdf, lag_labels

    if by not in observable.columns:
        raise AggregationError(f"observable has no {by!r} column")
    if "sq_disp_um2" not in observable.columns:
        raise AggregationError("observable has no 'sq_disp_um2' column")
    if n_pop < 1:
        raise AggregationError(f"n_pop must be >= 1, got {n_pop}")
    tau_lag_s = observable.attrs.get("lag_time_s")
    if tau_lag_s is None:
        raise AggregationError(
            "observable carries no lag time in .attrs; τ must not be guessed"
        )
    if observable[by].isna().any():
        raise AggregationError(f"{by!r} is unassigned for some displacements")
    if precision is not None:
        return _fit_simultaneous_per_spot(observable, n_pop=n_pop, by=by, n_starts=n_starts, seed=seed,
                                          D_bounds=D_bounds, weighting=weighting, n_jobs=n_jobs,
                                          precision=precision)

    groups = sorted(observable[by].unique())
    # EXACT-ZERO JUMPS (`sqd`'s module head, ruled 4 Nov 2026), decided once on every group's jumps: where they are
    # an immobile population the shared D gains a D of 0, each group's fraction of it is its own zeros' share
    # (counted; a group without zeros holds 0 of it), and the n_pop - 1 shared moving D are fitted to each group's
    # moving jumps.
    zeros = joint_zero_jumps(observable, weighting)
    immobile = zeros is not None and zeros.immobile
    if immobile and n_pop < 2:
        raise AggregationError(f"one population cannot be shared here: {zeros.note} Fit two or more.")
    curves, counts, shares = [], [], []
    for group in groups:
        part = observable.loc[observable[by] == group]
        values = part["sq_disp_um2"].to_numpy(float)
        if values.size == 0:
            raise AggregationError(f"group {group!r} has no displacements")
        # THE WEIGHTING, PER GROUP (14 Oct 2026): this fit ignored it, so Shared D = All
        # Conditions was step-weighted whatever the Weighting setting said. Weights are
        # formed within each group, so no track can be shared between two.
        # Lags 1-4 together: one ECDF per lag within the group (`observables.fit_ecdf`).
        weights, lags = displacement_weights(part, weighting), lag_labels(part)
        if immobile:
            zero = values == 0
            share = zero.mean() if weights is None else float(weights[zero].sum() / weights.sum())
            if zero.all():
                raise AggregationError(f"group {group!r} holds only exact-zero jumps: no moving D to share")
            values, lags = values[~zero], None if lags is None else lags[~zero]
            weights = None if weights is None else weights[~zero]
            shares.append(float(share))
        curves.append(fit_ecdf(values, weights, lags, max_points=ecdf_max_points))
        counts.append(int(part.shape[0]))
    if immobile:
        n_pop = n_pop - 1

    from .parallel import bootstrap_map
    from .sqd import _resolve_gradient

    n_sticks = n_pop - 1
    lo, hi = _bounds(n_pop, D_bounds)
    lower = np.concatenate([lo[:n_pop], np.zeros(n_sticks * len(groups))])
    upper = np.concatenate([hi[:n_pop], np.ones(n_sticks * len(groups))])
    shared = {"curves": curves, "n_pop": n_pop, "tau_lag_s": float(tau_lag_s), "lower": lower, "upper": upper,
              "starts": _simultaneous_guesses(n_pop, len(groups), lo, hi, lower, upper, n_starts, seed),
              "gradient": _resolve_gradient(gradient)}
    outcomes = bootstrap_map(_simultaneous_start, len(shared["starts"]), shared=shared, n_jobs=n_jobs)

    best, best_cost = None, np.inf
    for outcome in outcomes:                     # IN START ORDER: a tie goes to the earlier start, as in the loop
        if outcome is not None and outcome[0] < best_cost:
            best_cost, best = outcome
    if best is None:
        raise AggregationError("no simultaneous fit converged")

    def residuals(theta):
        return _simultaneous_residuals(theta, curves, n_pop, float(tau_lag_s))

    order = np.argsort(best[:n_pop])
    D_moving = np.asarray(best[:n_pop])[order]
    D = with_immobile(D_moving, np.ones(n_pop), 0.0)[0] if immobile else D_moving

    rows = []
    for index, group in enumerate(groups):
        start = n_pop + index * n_sticks
        sticks = best[start:start + n_sticks]
        _, fractions = _unpack(np.concatenate([best[:n_pop], sticks]), n_pop)
        fractions = np.asarray(fractions)[order]
        if immobile:
            fractions = with_immobile(D_moving, fractions, shares[index])[1]
        row = {by: group, "n_observations": counts[index]}
        for i, (d, f) in enumerate(zip(D, fractions), start=1):
            row[f"D_{i}"] = d
            row[f"fraction_{i}"] = f
        rows.append(row)

    out = pd.DataFrame(rows)
    residual = residuals(best)
    observed = np.concatenate([y for _, y in curves])
    total = float(np.sum((observed - observed.mean()) ** 2))
    out.attrs["D"] = D
    out.attrs["r_squared"] = float(1.0 - np.sum(residual ** 2) / total) if total > 0 else float("nan")
    out.attrs["rss"] = float(np.sum(residual ** 2))
    out.attrs["n_pop"] = int(len(D))
    if zeros is not None:
        out.attrs["zero_jumps"] = zeros.to_dict()
    out.attrs["by"] = by
    out.attrs["shared_D"] = True
    out.attrs["n_starts"] = int(n_starts)
    out.attrs["n_observations"] = int(len(observable))
    out.attrs["gradient"] = shared["gradient"]
    return out


def _fit_simultaneous_per_spot(observable, *, n_pop, by, n_starts, seed, D_bounds, weighting, n_jobs, precision):
    """`fit_simultaneous` with each step's own localisation noise: maximum likelihood, D shared, fractions per group
    (the docstring's PER-SPOT PRECISION). The same table and ``attrs`` as the least-squares fit, ``r_squared`` and
    ``rss`` aside (a likelihood has none); ``attrs["log_likelihood"]`` is the scaled sum the fit maximised."""
    from .observables import displacement_weights, lag_labels
    from .parallel import bootstrap_map
    from .precision import PrecisionError, step_noise, usable_steps

    if lag_labels(observable) is not None:
        raise AggregationError("per-spot precision is not built for lags fitted together (they reuse each jump in up to four displacements, so a likelihood over them counts it up to four times); "
                               "fit one lag")
    zeros = joint_zero_jumps(observable, weighting)
    if zeros is not None and zeros.immobile:
        raise AggregationError(f"per-spot precision is not combined with an immobile population of exact-zero jumps "
                               f"({zeros.note}): a jump of exactly 0 has no localisation error to model.")
    tau_lag_s = float(observable.attrs["lag_time_s"])
    groups = sorted(observable[by].unique())
    blocks, counts, left_out = [], [], 0
    for group in groups:
        part = observable.loc[observable[by] == group]
        try:
            ok = usable_steps(part)
            left_out += int((~ok).sum())
            if not ok.all():
                part = part.loc[ok]
            noise = step_noise(part, precision)
        except PrecisionError as exc:
            raise AggregationError(f"group {group!r}: {exc}") from exc
        values = part["sq_disp_um2"].to_numpy(float)
        if values.size < 2 * n_pop:
            raise AggregationError(f"group {group!r} has {values.size} usable displacements for {n_pop} "
                                   f"populations")
        weights = displacement_weights(part, weighting)
        w = np.ones(len(values)) if weights is None else np.asarray(weights, dtype=float)
        blocks.append({"values": values, "noise": noise, "weights": w})
        counts.append(int(len(observable.loc[observable[by] == group])))
    # EACH GROUP THE SAME SAY IN D: `sqd._nll_and_gradient` counts a group's steps at its effective n, so its
    # log-likelihood is scaled by the mean group's effective n over its own
    n_eff = np.array([b["weights"].sum() ** 2 / np.sum(b["weights"] ** 2) for b in blocks])
    for b, n in zip(blocks, n_eff):
        b["scale"] = float(n_eff.mean() / n)
    n_sticks = n_pop - 1
    lo, hi = _bounds(n_pop, D_bounds)
    lower = np.concatenate([lo[:n_pop], np.zeros(n_sticks * len(groups))])
    upper = np.concatenate([hi[:n_pop], np.ones(n_sticks * len(groups))])
    shared = {"blocks": blocks, "n_pop": n_pop, "tau_lag_s": tau_lag_s, "lower": lower, "upper": upper,
              "starts": _simultaneous_guesses(n_pop, len(groups), lo, hi, lower, upper, n_starts, seed)}
    outcomes = bootstrap_map(_simultaneous_per_spot_start, len(shared["starts"]), shared=shared, n_jobs=n_jobs)
    best, best_cost = None, np.inf
    for outcome in outcomes:                     # IN START ORDER: a tie goes to the earlier start
        if outcome is not None and outcome[0] < best_cost:
            best_cost, best = outcome
    if best is None:
        raise AggregationError("no simultaneous fit converged (per-spot precision)")
    order = np.argsort(best[:n_pop])
    D = np.asarray(best[:n_pop])[order]
    rows = []
    for index, group in enumerate(groups):
        start = n_pop + index * n_sticks
        _, fractions = _unpack(np.concatenate([best[:n_pop], best[start:start + n_sticks]]), n_pop)
        fractions = np.asarray(fractions)[order]
        row = {by: group, "n_observations": counts[index]}
        for i, (d, f) in enumerate(zip(D, fractions), start=1):
            row[f"D_{i}"] = d
            row[f"fraction_{i}"] = f
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs.update(D=D, r_squared=float("nan"), rss=float("nan"), log_likelihood=-float(best_cost),
                     n_pop=int(n_pop), by=by, shared_D=True, n_starts=int(n_starts),
                     n_observations=int(len(observable)), gradient="analytic", method="mle",
                     per_spot_precision=True, precision_left_out=int(left_out),
                     precision_calibration=(float(precision.c_nm), float(precision.p)))
    if zeros is not None:
        out.attrs["zero_jumps"] = zeros.to_dict()
    return out


def _simultaneous_per_spot_cost(theta, shared):
    """The joint negative log-likelihood and its exact gradient: each group's `sqd._nll_and_gradient` at the shared D
    and its own sticks, its D part summed over the groups, its sticks' part in its own block."""
    from .sqd import _PENALTY_PER_POINT, _nll_and_gradient

    n_pop, tau = shared["n_pop"], shared["tau_lag_s"]
    n_sticks = n_pop - 1
    D = theta[:n_pop]
    total, grad = 0.0, np.zeros(len(theta))
    for index, b in enumerate(shared["blocks"]):
        start = n_pop + index * n_sticks
        own = np.concatenate([D, theta[start:start + n_sticks]])
        penalty = _PENALTY_PER_POINT * max(len(b["values"]), 1)
        value, g = _nll_and_gradient(own, b["values"], n_pop, tau, b["weights"], 0.0, penalty, b["noise"])
        if value >= penalty:
            return float(_PENALTY_PER_POINT * sum(len(x["values"]) for x in shared["blocks"])), np.zeros(len(theta))
        total += b["scale"] * value
        grad[:n_pop] += b["scale"] * g[:n_pop]
        grad[start:start + n_sticks] += b["scale"] * g[n_pop:]
    return total, grad


def _simultaneous_per_spot_start(index: int, shared: dict):
    """Start ``index`` of the per-spot joint fit: ``(cost, theta)``, or None when the optimiser refused it. Module
    level, so a process pool can run it."""
    from scipy.optimize import minimize

    try:
        result = minimize(_simultaneous_per_spot_cost, shared["starts"][index], args=(shared,), method="L-BFGS-B",
                          jac=True, bounds=list(zip(shared["lower"], shared["upper"])),
                          options={"maxiter": 5000, "ftol": 1e-12})
    except ValueError:                           # pragma: no cover - bad start
        return None
    if not result.success or not np.isfinite(result.fun):
        return None
    return float(result.fun), np.asarray(result.x)


def _simultaneous_guesses(n_pop, n_groups, lo, hi, lower, upper, n_starts, seed) -> list:
    """`fit_simultaneous`'s starting points, IN THE ORDER ITS ONE LOOP DREW THEM from one generator: start 0 the
    D spread evenly in log between the bounds, the others drawn; every start's fractions drawn. Nothing a start does
    draws, so a start given its guess here begins where it began in the loop, in whichever process runs it."""
    n_sticks = n_pop - 1
    rng = np.random.default_rng(seed)
    guesses = []
    for attempt in range(max(1, n_starts)):
        if attempt == 0:
            guess_D = np.geomspace(lo[0] * 5, hi[0] / 5, n_pop)
        else:
            guess_D = np.sort(np.exp(rng.uniform(np.log(lo[:n_pop]), np.log(hi[:n_pop]))))
        theta0 = np.concatenate([guess_D, rng.uniform(0.15, 0.85, n_sticks * n_groups)])
        guesses.append(np.clip(theta0, lower + 1e-12, upper - 1e-12))
    return guesses


def _simultaneous_residuals(theta, curves, n_pop, tau_lag_s):
    """Every group's model CDF less its ECDF, end to end: D shared (``theta[:n_pop]``), each group's fraction
    sticks after them (`CLAUDE.md`'s packing)."""
    from .sqd import sqd_cdf

    n_sticks = n_pop - 1
    D = theta[:n_pop]
    blocks = []
    for index, (x, y) in enumerate(curves):
        start = n_pop + index * n_sticks
        sticks = theta[start:start + n_sticks]
        _, fractions = _unpack(np.concatenate([D, sticks]), n_pop)
        blocks.append(sqd_cdf(x, D, fractions, float(tau_lag_s)) - y)
    return np.concatenate(blocks)


def _simultaneous_jacobian(theta, curves, n_pop, tau_lag_s) -> np.ndarray:
    """The exact Jacobian of `_simultaneous_residuals`: each group's rows are `sqd._cdf_jacobian` of its own model
    (the shared D and its own sticks) -- the D columns from it, its sticks' columns from it, every other group's
    sticks 0. One derivative, the single fit's, so the two never disagree."""
    from .sqd import _cdf_jacobian

    n_sticks = n_pop - 1
    jacobian = np.zeros((sum(len(x) for x, _y in curves), len(theta)))
    row = 0
    for index, (x, _y) in enumerate(curves):
        start = n_pop + index * n_sticks
        block = _cdf_jacobian(np.concatenate([theta[:n_pop], theta[start:start + n_sticks]]), x, n_pop, tau_lag_s)
        jacobian[row:row + len(x), :n_pop] = block[:, :n_pop]
        jacobian[row:row + len(x), start:start + n_sticks] = block[:, n_pop:]
        row += len(x)
    return jacobian


def _simultaneous_start(index: int, shared: dict):
    """Start ``index`` of `fit_simultaneous` (``shared``: its curves, bounds, guesses and gradient): ``(cost,
    theta)``, or None when the optimiser refused the start. Module level, so a process pool can run it."""
    from scipy.optimize import least_squares

    curves, n_pop, tau = shared["curves"], shared["n_pop"], shared["tau_lag_s"]
    jac = "2-point"
    if shared.get("gradient") == "analytic":
        def jac(theta):
            return _simultaneous_jacobian(theta, curves, n_pop, tau)
    try:
        result = least_squares(lambda theta: _simultaneous_residuals(theta, curves, n_pop, tau),
                               shared["starts"][index], jac=jac, bounds=(shared["lower"], shared["upper"]))
    except ValueError:                           # pragma: no cover - bad start
        return None
    return float(result.cost), np.asarray(result.x)


def _fit_fractions(values: np.ndarray, D: np.ndarray, tau_lag_s: float,
                   sample_weights: "np.ndarray | None" = None, step_noise=None) -> np.ndarray:
    """Fit only the mixture weights, with ``D`` held fixed (the simultaneous fit).

    A short closed-form-free EM: with ``D`` fixed the mixture weights have an
    exact EM update, so this converges quickly and cannot wander off to another
    ``D``, which is the entire point of sharing ``D``.

    ``sample_weights`` (one weight per displacement) makes the fit TRACK- or
    CELL-weighted rather than step-weighted: pass ``observables.displacement_weights``
    so a per-replicate fraction matches the track-weighted claims (and, with one
    replicate, reproduces the pooled track-weighted fit). Default ``None`` = the
    step-weighted EM, unchanged.

    A ``D`` of 0 is the immobile population of exact-zero jumps (`sqd`'s module head, ruled 4 Nov 2026): the EM
    against "a point at 0 plus the line" gives every exact zero to it and every moving jump to the others, so its
    fraction is the zeros' (weighted) share, counted, and the others are this EM on the moving jumps alone, scaled
    by the share that moves -- exactly how `sqd.fit_sqd` made the D held here.

    ``step_noise``: PER-SPOT PRECISION (6 Nov 2026, `sqd.fit_sqd`'s ``precision``), each value's own per-axis
    localisation variance: population k's mean for value i is then 4 D_k tau + 2 noise_i, as in the fit that made D.
    """
    from scipy.stats import expon  # local import keeps module import light

    D = np.asarray(D, dtype=float)
    point = D == 0
    if step_noise is not None:
        if point.any():
            raise SQDError("per-spot precision has no immobile population of exact-zero jumps")
        if len(D) == 1:
            return np.array([1.0])
        noise = np.asarray(step_noise, dtype=float)
        means = 2.0 * 2 * D[None, :] * tau_lag_s + 2.0 * noise[:, None]
        densities = np.exp(-np.asarray(values, dtype=float)[:, None] / means) / means
        return _em_weights(densities, sample_weights)
    if point.any():
        if point.sum() > 1 or point.all():
            raise SQDError(f"one immobile population (D 0) beside moving ones, got D = {list(D)}")
        values = np.asarray(values, dtype=float)
        zero = values == 0
        sw = (np.ones(len(values)) if sample_weights is None
              else np.asarray(sample_weights, dtype=float))
        if sw.sum() <= 0:
            raise SQDError("sample weights sum to zero while fitting fractions")
        share = float(sw[zero].sum() / sw.sum())
        out = np.zeros(len(D))
        out[point] = share
        if (~zero).any():
            out[~point] = (1.0 - share) * _fit_fractions(
                values[~zero], D[~point], tau_lag_s,
                None if sample_weights is None else sw[~zero])
        return out

    n_pop = len(D)
    if n_pop == 1:
        return np.array([1.0])
    means = 2.0 * 2 * np.asarray(D, dtype=float) * tau_lag_s      # 4·D·τ
    densities = np.column_stack([expon.pdf(values, scale=m) for m in means])
    return _em_weights(densities, sample_weights)


def _em_weights(densities, sample_weights=None) -> np.ndarray:
    """The mixture weights' EM with every population's density per value held (`_fit_fractions`)."""
    if not np.isfinite(densities).all():
        raise SQDError("non-finite densities while fitting fractions")
    n_pop = densities.shape[1]

    sw = None if sample_weights is None else np.asarray(sample_weights, dtype=float)
    if sw is not None and sw.sum() <= 0:
        raise SQDError("sample weights sum to zero while fitting fractions")

    weights = np.full(n_pop, 1.0 / n_pop)
    for _ in range(500):
        numerator = densities * weights[None, :]
        total = numerator.sum(axis=1, keepdims=True)
        if np.any(total <= 0):
            raise SQDError("degenerate responsibilities while fitting fractions")
        responsibilities = numerator / total
        if sw is None:
            updated = responsibilities.mean(axis=0)
        else:                                    # weighted mean of responsibilities
            updated = (responsibilities * sw[:, None]).sum(axis=0) / sw.sum()
        if np.max(np.abs(updated - weights)) < 1e-10:
            weights = updated
            break
        weights = updated
    return weights / weights.sum()


def aggregate_fractions(
    per_unit: pd.DataFrame,
    *,
    quantity: str = "fraction_1",
    by: str = "condition",
    weighted: bool = False,
) -> pd.DataFrame:
    """Mean ± spread of a per-unit quantity, per condition.

    ``weighted=True`` weights units by their observation count. Unweighted is
    the default: it treats each cell or day as one independent observation,
    which is what makes ``n`` the number of units.
    """
    if quantity not in per_unit.columns:
        raise AggregationError(f"no '{quantity}' column; did the fits succeed?")
    usable = per_unit[per_unit["ok"]] if "ok" in per_unit.columns else per_unit
    if usable.empty:
        raise AggregationError("no unit was fitted successfully")
    if by not in usable.columns:
        raise AggregationError(f"no '{by}' column to group by")

    rows = []
    for key, frame in usable.groupby(by, observed=True):
        values = frame[quantity].to_numpy(dtype=float)
        counts = frame["n_observations"].to_numpy(dtype=float)
        weights = counts if weighted else np.ones_like(counts)
        mean = float(np.average(values, weights=weights))
        n = len(values)
        sd = float(np.std(values, ddof=1)) if n > 1 else float("nan")
        rows.append(
            {
                by: key,
                "n_units": n,
                "mean": mean,
                "sd": sd,
                "sem": sd / np.sqrt(n) if n > 1 else float("nan"),
                "min": float(values.min()),
                "max": float(values.max()),
                "weighted": weighted,
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["unit"] = per_unit.attrs.get("unit")
    out.attrs["quantity"] = quantity
    return out


def compare_conditions(
    per_unit: pd.DataFrame,
    *,
    condition_a: str,
    condition_b: str,
    quantity: str = "fraction_1",
    by: str = "condition",
    test: str = "permutation",
    n_permutations: int = 10000,
    seed: int | None = 0,
    n_boot: int = 10000,
) -> ComparisonResult:
    """Compare two conditions using **unit-level** values.

    ``n`` is the number of independent units, never the number of steps. With
    three replicate days, ``n = 3``, and the result should look as uncertain as
    that implies.

    ``test``
        ``"permutation"`` (default): exact-ish, assumption-free, appropriate at
        small ``n``. ``"ttest"``: Welch's t, offered for familiarity but
        normality is not credible at ``n = 3``.

    The interval on the difference is a bootstrap percentile interval over units.
    """
    _reject_raw_observable(per_unit)
    usable = per_unit[per_unit["ok"]] if "ok" in per_unit.columns else per_unit
    if quantity not in usable.columns:
        raise AggregationError(f"no '{quantity}' column; did the fits succeed?")
    if by not in usable.columns:
        raise AggregationError(f"no '{by}' column to group by")

    values_a = usable.loc[usable[by] == condition_a, quantity].to_numpy(dtype=float)
    values_b = usable.loc[usable[by] == condition_b, quantity].to_numpy(dtype=float)
    for name, values in ((condition_a, values_a), (condition_b, values_b)):
        if len(values) < _MIN_UNITS_FOR_COMPARISON:
            raise AggregationError(
                f"condition '{name}' has {len(values)} usable "
                f"{per_unit.attrs.get('unit', 'unit')}(s); at least "
                f"{_MIN_UNITS_FOR_COMPARISON} are needed. A p-value from fewer "
                "would be meaningless."
            )

    difference = float(values_b.mean() - values_a.mean())
    rng = np.random.default_rng(seed)

    if test == "permutation":
        # DELEGATED, because this was a SECOND COPY of the same calculation and it
        # carried the same defect in a scale-dependent form. The loop here guarded the
        # re-summation with an ABSOLUTE `- 1e-15`; one ULP of a value near 12 is already
        # ~1.8e-15, so the guard held for fractions in [0, 1] and failed for percentages.
        # MEASURED on three replicate days that are perfectly separated (floor 0.100):
        # fractions gave 0.1008, the same numbers as percentages gave **0.0667**, and
        # x1000 gave 0.1008 again: non-monotonic in magnitude, which is the signature of
        # a floating-point artefact rather than of a statistic.
        # `uncertainty.permutation_p` enumerates exactly at these unit counts and uses a
        # RELATIVE tolerance, so it is correct at every scale and needs no n_permutations.
        from .uncertainty import permutation_p as _permutation_p

        p_value = _permutation_p(values_a, values_b, n_perm=n_permutations, seed=seed)
    elif test == "ttest":
        from scipy.stats import ttest_ind

        p_value = float(ttest_ind(values_a, values_b, equal_var=False).pvalue)
    else:
        raise AggregationError(f"test must be 'permutation' or 'ttest', got {test!r}")

    differences = np.empty(n_boot)
    for i in range(n_boot):
        differences[i] = (
            rng.choice(values_b, size=len(values_b), replace=True).mean()
            - rng.choice(values_a, size=len(values_a), replace=True).mean()
        )
    ci_low, ci_high = np.quantile(differences, [0.025, 0.975])

    return ComparisonResult(
        condition_a=condition_a,
        condition_b=condition_b,
        unit=str(per_unit.attrs.get("unit", "unit")),
        quantity=quantity,
        mean_a=float(values_a.mean()),
        mean_b=float(values_b.mean()),
        difference=difference,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=float(p_value),
        n_a=len(values_a),
        n_b=len(values_b),
        test=test,
        values_a=values_a,
        values_b=values_b,
    )


def _reject_raw_observable(frame: pd.DataFrame) -> None:
    """Structural guard: a p-value may never be computed from pooled steps.

    Naming forbidden tests one by one would leave the footgun reachable under
    the next name, so the *data path* is closed instead: if this looks like a
    raw displacement table rather than per-unit summaries, refuse.
    """
    if "sq_disp_um2" in frame.columns:
        raise AggregationError(
            "this looks like a raw displacement table, not per-unit summaries. "
            "Steps are not independent, so a significance test on pooled steps is refused. "
            "Run fit_by_unit(...) first and pass its result; for an effect size without a "
            "test use wasserstein_effect_size."
        )


def wasserstein_effect_size(
    observable: pd.DataFrame,
    *,
    condition_a: str,
    condition_b: str,
    unit: str = "cell",
    by: str = "condition",
    n_boot: int = 2000,
    seed: int | None = 0,
) -> pd.DataFrame:
    """Model-free magnitude: Wasserstein distance between two conditions.

    The fitted-fraction difference is a *model-based* effect size: it presumes
    the mixture model is right. This is the complement: the 1-Wasserstein
    (earth-mover) distance between the two conditions' squared-displacement
    distributions, which assumes no model at all and carries real units (µm²).

    Crucially it is **not a test**. It is a distance, reported with an interval
    obtained by resampling **units** (cells / days), so the uncertainty reflects
    the real ``n``. A distance whose interval includes ~0 is not evidence of a
    difference; a p-value, if wanted, comes from ``compare_conditions``.

    Returns a one-row frame with the distance, its interval, and the unit count.
    """
    from scipy.stats import wasserstein_distance

    for column in ("sq_disp_um2", by, unit):
        if column not in observable.columns:
            raise AggregationError(f"observable has no '{column}' column")
    if observable[unit].isna().any():
        raise AggregationError(f"'{unit}' is unassigned for some displacements")

    frame_a = observable[observable[by] == condition_a]
    frame_b = observable[observable[by] == condition_b]
    for name, frame in ((condition_a, frame_a), (condition_b, frame_b)):
        if frame.empty:
            raise AggregationError(f"condition '{name}' has no displacements")

    units_a = list(frame_a.groupby(unit, observed=True).groups)
    units_b = list(frame_b.groupby(unit, observed=True).groups)
    for name, units in ((condition_a, units_a), (condition_b, units_b)):
        if len(units) < _MIN_UNITS_FOR_COMPARISON:
            raise AggregationError(
                f"condition '{name}' has {len(units)} {unit_word(unit)}(s); at least "
                f"{_MIN_UNITS_FOR_COMPARISON} are needed for an interval."
            )

    values_a = frame_a["sq_disp_um2"].to_numpy(dtype=float)
    values_b = frame_b["sq_disp_um2"].to_numpy(dtype=float)
    distance = float(wasserstein_distance(values_a, values_b))

    groups_a = {k: g["sq_disp_um2"].to_numpy() for k, g in frame_a.groupby(unit, observed=True)}
    groups_b = {k: g["sq_disp_um2"].to_numpy() for k, g in frame_b.groupby(unit, observed=True)}
    rng = np.random.default_rng(seed)
    distances = np.empty(n_boot)
    for i in range(n_boot):
        pick_a = rng.choice(units_a, size=len(units_a), replace=True)
        pick_b = rng.choice(units_b, size=len(units_b), replace=True)
        distances[i] = wasserstein_distance(
            np.concatenate([groups_a[k] for k in pick_a]),
            np.concatenate([groups_b[k] for k in pick_b]),
        )
    ci_low, ci_high = np.quantile(distances, [0.025, 0.975])

    return pd.DataFrame(
        [
            {
                "condition_a": condition_a,
                "condition_b": condition_b,
                "unit": unit,
                "wasserstein_um2": distance,
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "n_units_a": len(units_a),
                "n_units_b": len(units_b),
                "n_displacements": len(values_a) + len(values_b),
                "note": "distance, not a test; interval from resampling units",
            }
        ]
    )


def poolability(
    observable: pd.DataFrame,
    *,
    unit: str = "day",
    by: str = "condition",
    n_pairs: int = 200,
    seed: int | None = 0,
) -> pd.DataFrame:
    """Is pooling units safe? Compare within-condition spread to between-condition.

    The question behind every pooled analysis: **do replicate days differ from
    each other as much as the conditions do?** If they do, pooling days hides a
    batch effect inside the very comparison it is meant to support.

    Method: 1-Wasserstein distance between pairs of units *within* a condition
    (the nuisance spread) against pairs *across* conditions (the signal). Both
    are distances on pooled displacements, which is legitimate: no p-value is
    computed here.

    Returns one row per comparison type with the median and range of distances,
    plus a ``ratio`` (between / within). A ratio near 1 means the conditions are
    no further apart than two arbitrary days, i.e. **do not pool**.
    """
    from scipy.stats import wasserstein_distance

    for column in ("sq_disp_um2", unit, by):
        if column not in observable.columns:
            raise AggregationError(f"observable has no '{column}' column")
    if observable[unit].isna().any():
        raise AggregationError(f"'{unit}' is unassigned for some displacements")
    if unit == "day":
        require_real_days(observable)              # one session has no day-vs-day spread

    groups = {
        (condition, key): frame["sq_disp_um2"].to_numpy(dtype=float)
        for (condition, key), frame in observable.groupby([by, unit], observed=True)
    }
    if len(groups) < 2:
        raise AggregationError(f"need at least 2 {unit_word(unit)}s to assess poolability")

    rng = np.random.default_rng(seed)
    keys = list(groups)
    within, between = [], []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            distance = wasserstein_distance(groups[keys[i]], groups[keys[j]])
            (within if keys[i][0] == keys[j][0] else between).append(distance)

    rows = []
    for label, values in (("within_condition", within), ("between_condition", between)):
        if not values:
            continue
        values = np.asarray(values, dtype=float)
        rows.append(
            {
                "comparison": label,
                "n_pairs": len(values),
                "median_wasserstein_um2": float(np.median(values)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    out = pd.DataFrame(rows)
    if {"within_condition", "between_condition"} <= set(out["comparison"]):
        w = float(out.loc[out.comparison == "within_condition", "median_wasserstein_um2"].iloc[0])
        b = float(out.loc[out.comparison == "between_condition", "median_wasserstein_um2"].iloc[0])
        ratio = b / w if w > 0 else float("inf")
        out.attrs["ratio_between_over_within"] = ratio
        from .chart_text import fold_words

        out.attrs["verdict"] = (
            f"conditions are apart by {fold_words(ratio, 1, of=True)} the {unit_word(unit)}-to-{unit_word(unit)} "
            "spread within a condition: pooling is defensible"
            if ratio >= 3
            else f"conditions are only {fold_words(ratio, 1, of=True)} the {unit_word(unit)}-to-{unit_word(unit)} "
            f"spread: the {unit_word(unit)} effect is comparable to the signal; DO NOT pool"
        )
    else:
        out.attrs["verdict"] = (
            f"only one {by} present, so between-{by} distance cannot be computed; "
            f"the within-{by} spread across {unit_word(unit)}s is reported alone"
        )
    out.attrs["unit"] = unit
    return out


def cdf_overlay(
    observable: pd.DataFrame,
    *,
    unit: str = "cell",
    by: str = "condition",
    n_points: int = 200,
) -> pd.DataFrame:
    """One empirical CDF per unit, on a shared grid, the plot for the eye.

    Numbers hide shape. A per-unit CDF overlay shows immediately whether the
    conditions separate cleanly, whether one outlying cell is driving a
    difference, and whether the spread within a condition swamps the gap between
    conditions: the things a single mean cannot say.

    Returns a long table (``unit``, ``condition``, ``x_um2``, ``cdf``) ready to
    plot without any further computation.
    """
    for column in ("sq_disp_um2", unit):
        if column not in observable.columns:
            raise AggregationError(f"observable has no '{column}' column")
    if n_points < 2:
        raise AggregationError(f"n_points must be >= 2, got {n_points}")

    values = observable["sq_disp_um2"].to_numpy(dtype=float)
    positive = values[values > 0]
    if positive.size == 0:
        raise AggregationError("no positive squared displacements to plot")
    # Log-spaced grid: the interesting structure is at small displacements.
    grid = np.geomspace(positive.min(), np.quantile(positive, 0.999), n_points)

    frames = []
    for key, frame in observable.groupby(unit, observed=True):
        unit_values = np.sort(frame["sq_disp_um2"].to_numpy(dtype=float))
        cdf = np.searchsorted(unit_values, grid, side="right") / len(unit_values)
        row = {unit: key, "x_um2": grid, "cdf": cdf, "n_observations": len(unit_values)}
        if by in frame.columns:
            row[by] = frame[by].iloc[0]
        frames.append(pd.DataFrame(row))
    return pd.concat(frames, ignore_index=True)


def nested_spread(
    per_cell: pd.DataFrame,
    *,
    quantity: str = "fraction_1",
    day_column: str = "day",
) -> pd.DataFrame:
    """How much of the spread is between days versus between cells within a day.

    If between-day variance dominates, the day is the right bootstrap unit and a
    cell-level interval understates the uncertainty. If cells within a day vary
    more than days do, the batch effect is small. This is the check that tells
    you which unit to trust: rather than picking one and hoping.
    """
    usable = per_cell[per_cell["ok"]] if "ok" in per_cell.columns else per_cell
    if quantity not in usable.columns:
        raise AggregationError(f"no '{quantity}' column")
    require_real_days(usable, day_column=day_column)   # nested spread needs ≥2 days

    day_means = usable.groupby(day_column, observed=True)[quantity].mean()
    grand_mean = float(usable[quantity].mean())
    between = float(np.var(day_means.to_numpy(), ddof=1))
    within = float(
        usable.groupby(day_column, observed=True)[quantity]
        .apply(lambda s: np.var(s.to_numpy(), ddof=1) if len(s) > 1 else np.nan)
        .mean()
    )
    total = between + (within if np.isfinite(within) else 0.0)
    return pd.DataFrame(
        [
            {
                "quantity": quantity,
                "grand_mean": grand_mean,
                "n_days": int(usable[day_column].nunique()),
                "n_cells": len(usable),
                "between_day_variance": between,
                "within_day_variance": within,
                "between_day_share": between / total if total > 0 else float("nan"),
                "recommended_unit": (
                    day_column if total > 0 and between / total > 0.5 else "cell"
                ),
            }
        ]
    )


def omnibus_across_conditions(observable, *, conditions, n_pop: int,
                              unit: str = "cell", shared_D: bool = True,
                              quantity: str = "fraction_1", n_permutations: int = 20000,
                              seed: int = 0, **fit_kwargs):
    """Do the conditions differ at all? ONE test on the per-unit fits.

    NOT PAIRWISE, and that is a power decision as much as an honesty one. At three units
    per condition a pairwise permutation test admits 20 splits and cannot reach p < 0.05;
    permuting the same points across FOUR conditions admits 369 600 and reaches 2.7e-6.
    Pairwise discards two conditions to compare two, then pays a multiplicity correction
    for having done it six times.

    The statistic is the between-condition spread of the unit means (a one-way F without
    its distributional assumptions, since at n = 3 normality is not credible). Returns
    ``(per_unit, result, spread)``; ``result`` carries the observed statistic, the p, the
    number of arrangements the design admits, and the smallest p that design could ever
    produce -- so a test that structurally cannot reject says so.
    """
    from math import factorial

    names = list(dict.fromkeys(conditions))
    if len(names) < 2:
        raise ValueError(
            f"an omnibus test needs at least two conditions, got {names}")

    # ACCEPTS A MAPPING TOO, so no caller has to concatenate frames itself. A Qt class
    # doing that broke the thin-Qt rule (widgets.py imports no numpy/pandas/scipy, and an
    # AST test asserts it) -- the concat belongs here, where it is tested.
    if isinstance(observable, dict):
        observable = pd.concat(
            [observable[n].assign(condition=n) for n in names], ignore_index=True)

    # THE DAYS ARE NOT PAIRS, and the day column does not know that. `day` is a
    # WITHIN-GROUP ORDINAL: every condition has its own day 1, and nothing ever tied one
    # condition's day 1 to another's except the shared column name. `fit_by_unit` catches
    # it -- "day '1' spans several condition values" -- because units must NEST inside
    # groups, so the unit key is made unique per condition here rather than pretending
    # the collision is a pairing.
    # .attrs must survive the copy: `fit_by_unit` reads the lag time from there and
    # refuses to guess it, so losing attrs turns a unit-key fix into "tau must not be
    # guessed" three frames down.
    original_attrs = dict(observable.attrs)
    observable = observable.copy()
    observable.attrs.update(original_attrs)
    if unit in observable.columns and "condition" in observable.columns:
        collides = observable.groupby(unit)["condition"].nunique().gt(1).any()
        if collides:
            observable[unit] = (observable["condition"].astype(str) + ":"
                                + observable[unit].astype(str))

    per_unit = fit_by_unit(observable, n_pop=n_pop, unit=unit, shared_D=shared_D,
                           **fit_kwargs)
    if quantity not in per_unit.columns:
        raise ValueError(
            f"{quantity!r} is not a column of the per-unit fits: {list(per_unit.columns)}")

    frame = per_unit[per_unit["condition"].isin(names)]
    values = frame[quantity].to_numpy(dtype=float)
    labels = frame["condition"].to_numpy()
    counts = [int((labels == n).sum()) for n in names]
    if min(counts) < 2:
        raise ValueError(
            f"every condition needs at least two {unit_word(unit)}s; got "
            f"{dict(zip(names, counts))}. One unit gives no spread to compare")

    def statistic(v, l):
        grand = v.mean()
        return sum(int((l == n).sum()) * (v[l == n].mean() - grand) ** 2 for n in names)

    observed = statistic(values, labels)
    rng = np.random.default_rng(seed)
    shuffled = labels.copy()
    n_ge = 0
    for _ in range(int(n_permutations)):
        rng.shuffle(shuffled)
        if statistic(values, shuffled) >= observed:
            n_ge += 1
    # p is never reported as 0: with a finite number of draws the honest statement is an
    # upper bound, which is what (n_ge + 1) / (n + 1) gives.
    p_value = (n_ge + 1) / (int(n_permutations) + 1)

    arrangements = factorial(len(values))
    for c in counts:
        arrangements //= factorial(c)
    result = {
        "statistic": float(observed), "p_value": float(p_value),
        "n_permutations": int(n_permutations), "unit": unit,
        "conditions": names, "n_units": dict(zip(names, counts)),
        "arrangements_the_design_admits": int(arrangements),
        "smallest_p_the_design_can_produce": float(1.0 / arrangements),
        "quantity": quantity, "shared_D": bool(shared_D),
        "note": ("one omnibus test, so there is no multiplicity to correct. Where the "
                 "separation lies is for the eye on the per-unit points; attaching a "
                 "pairwise p after looking is not supported by this test"),
    }

    spread = None
    if unit == "cell" and "day" in per_unit.columns:
        try:
            spread = nested_spread(per_unit, quantity=quantity)
        except Exception:                      # noqa: BLE001 - a diagnostic, not the answer
            spread = None
    return per_unit, result, spread


def achievable_p(counts, *, sided: int = 2) -> dict:
    """The smallest p a permutation test on these group sizes can ever produce.

    A DESIGN question, answerable before any data is fitted.

    Sidedness is not a detail here. Three against three admits C(6,3) = 20 splits, so the
    floor is 1/20 = 0.05 one-sided and 2/20 = 0.100 two-sided -- and a difference between
    two conditions is a two-sided question unless a direction was pre-registered. Reported
    one-sided while quoted two-sided, this exact number reads as "can just reach 0.05"
    when the honest statement is "cannot".

    An omnibus statistic (between-group spread) is naturally ONE-sided, because only large
    values are evidence, so pass ``sided=1`` for it.

    The doubling needs equal groups, and that was missed. Two-sided costs a factor of two
    because the MIRROR arrangement (the one that swaps the group labels) reproduces the
    same ``|difference|``. Swapping labels only yields a valid arrangement when the two
    groups are the same size. At 3 against 4 there is no mirror, so the floor is 1/35 and
    not 2/35, and the old form overstated it twofold. It never bit here because every
    design in this project is balanced (3 replicate days against 3), and it is corrected
    rather than left because an unbalanced comparison is one movie-count away: protein E has
    29 movies against 30. The error ran in the SAFE direction (claiming the design could
    say less than it can), which is why nothing downstream moved.
    """
    from math import factorial

    if sided not in (1, 2):
        raise ValueError(f"sided must be 1 or 2, got {sided!r}")
    counts = [int(c) for c in counts]
    if len(counts) < 2 or min(counts) < 1:
        raise ValueError(f"need at least two groups of at least one unit, got {counts}")
    balanced = len(set(counts)) == 1
    extreme = 2 if (sided == 2 and len(counts) == 2 and balanced) else 1
    total = factorial(sum(counts))
    for c in counts:
        total //= factorial(c)
    smallest = min(extreme / total, 1.0)
    return {"arrangements": int(total), "sided": int(sided),
            "smallest_p": float(smallest),
            "can_reach_0.05": bool(smallest <= 0.05),
            "counts": counts,
            "note": (f"{sided}-sided floor. A two-condition comparison is two-sided "
                     f"unless a direction was pre-registered; an omnibus spread "
                     f"statistic is one-sided because only large values are evidence"
                     + ("" if extreme == 2 else
                        "; groups are unbalanced or this is an omnibus, so there is no "
                        "mirror arrangement and the floor is 1/arrangements"))}


#: Layers a measurement is nested in, outermost first. Steps sit inside tracks, tracks
#: inside cells, cells inside MOVIES, movies inside days.
#:
#: THE MOVIE LAYER WAS MISSING FROM THIS LIST AND IT IS THE LARGEST ONE. A three-level
#: decomposition of the per-cell slow fraction (day, movie, cell) on a lab project put
#: the movie-to-movie share of the variance above the day's in every condition, and
#: above the CELL's in some. A movie is one field of view, so cells inside it share
#: focus, illumination and local density; treating cells as independent ignores that, and
#: it is a larger effect than the shared culture the documented caveat already names. With
#: some 30 movies per condition against 3 days, the MOVIE is the better middle unit:
#: ten times the count of the day without the within-field correlation of the cell.
#: Whether the movie variance is optics or genuine spatial heterogeneity in the culture
#: is NOT settled by this measurement.
LAYERS = ("day", "movie", "cell", "track", "step")


def layer_report(per_cell, *, quantity: str = "fraction_1", day_column: str = "day",
                 movie_column: str = "movie", as_percent: bool = True) -> dict:
    """Spread at each nested layer, in the units the number is quoted in.

    Three levels where the movie column exists: day, movie-within-day, cell-within-movie.
    The movie layer was omitted from the first version and it is the LARGEST one --
    a large share of the variance on a lab project, more than the day's in every
    condition. A movie is one field of view, so cells inside it share focus,
    illumination and local density, and treating them as independent ignores a bigger
    effect than the shared culture the documented caveat names.

    Why it goes on every panel. "n = 3 days" reads as a weakness until the days are
    visibly in agreement, and the shares are what make the choice of unit a decision
    rather than a preference: with some 30 movies against 3 days, the movie is the better
    middle unit -- ten times the count without the within-field correlation of the cell.

    SDs, not variances: 0.00102 cannot be compared by eye to a fraction of 0.14, while
    3.2 pp sits directly beside 14.0 pp.

    The day spread here is the spread of day means of per-CELL values, which is not the
    same as fitting each day's steps as one pool (on a lab project the two differ by about a factor of two):
    pooling weights a big cell more, averaging per-cell values gives every cell equal say.
    The caption says which.
    """
    spread = nested_spread(per_cell, quantity=quantity, day_column=day_column)
    row = spread.iloc[0]
    scale = 100.0 if as_percent else 1.0
    unit = "pp" if as_percent else ""

    usable = per_cell
    if "ok" in usable.columns:
        usable = usable[usable["ok"].astype(bool)]

    between = float(row["between_day_variance"])
    within = float(row["within_day_variance"])
    share = float(row["between_day_share"])
    out = {
        "day_sd": float(np.sqrt(between)) * scale,
        "cell_sd": float(np.sqrt(within)) * scale,
        "between_day_variance": between, "within_day_variance": within,
        "between_day_share": share,
        "n_days": int(usable[day_column].nunique()),
        "n_cells": int(len(usable)),
        "quantity": quantity, "as_percent": bool(as_percent),
    }

    # THE MOVIE LAYER, when the column is there. Split within-day variance into
    # movie-to-movie and cell-within-movie rather than calling all of it "cell".
    #
    # COUNTING MOVIES IS NOT THE TEST -- MOVIES PER DAY IS. Three movies spread one per
    # day gives three distinct names and a movie-within-day variance taken over a SINGLE
    # value, which is NaN; that NaN then propagates through `total` and every share, so a
    # caption reading "movie +/-nan pp (nan %)" is what a user meets. With one movie per
    # day the movie layer is not merely unmeasured, it is CONFOUNDED with the day -- there
    # is no way to tell a movie effect from a day effect -- so the honest report is the
    # two-level one naming the reason, not three levels with a hole in them.
    movies_per_day = 0
    if movie_column in usable.columns and len(usable):
        movies_per_day = int(usable.groupby(day_column)[movie_column].nunique().max())
    if movies_per_day > 1:
        cell_v = float(usable.groupby([day_column, movie_column])[quantity]
                       .var(ddof=1).mean())
        movie_means = (usable.groupby([day_column, movie_column])[quantity]
                       .mean().reset_index())
        movie_v = float(movie_means.groupby(day_column)[quantity].var(ddof=1).mean())
        day_v = float(movie_means.groupby(day_column)[quantity].mean().var(ddof=1))
        # A single day leaves no day spread and a single cell per movie no cell spread;
        # both are legitimately zero-information rather than zero, and a NaN carried into
        # `total` would blank the two layers that ARE measured.
        day_v = day_v if np.isfinite(day_v) else 0.0
        cell_v = cell_v if np.isfinite(cell_v) else 0.0
        total = day_v + movie_v + cell_v
        out.update({
            "movie_sd": float(np.sqrt(movie_v)) * scale,
            "cell_within_movie_sd": float(np.sqrt(cell_v)) * scale,
            "day_share": day_v / total if total else float("nan"),
            "movie_share": movie_v / total if total else float("nan"),
            "cell_share": cell_v / total if total else float("nan"),
            "n_movies": int(usable[movie_column].nunique()),
        })
        out["caption"] = (
            f"replicate ±{out['day_sd']:.1f} {unit} ({100 * out['day_share']:.0f} %)  "
            f"·  movie ±{out['movie_sd']:.1f} {unit} "
            f"({100 * out['movie_share']:.0f} %)  ·  cell ±"
            f"{out['cell_within_movie_sd']:.1f} {unit} "
            f"({100 * out['cell_share']:.0f} %)"
            f"   (n = {out['n_days']} replicates, {out['n_movies']} movies, "
            f"{out['n_cells']} cells; spread of replicate MEANS of per-cell {quantity})")
        out["binding_layer"] = max(
            (("day", out["day_share"]), ("movie", out["movie_share"]),
             ("cell", out["cell_share"])), key=lambda kv: kv[1])[0]
    else:
        reason = ("NO movie column" if movie_column not in usable.columns
                  else f"only {movies_per_day} movie per replicate, so the movie layer is "
                       "CONFOUNDED with the replicate and cannot be separated from it")
        out["movie_layer_folded"] = reason
        out["n_movies"] = (int(usable[movie_column].nunique())
                           if movie_column in usable.columns else 0)
        out["caption"] = (
            f"replicate-to-replicate ±{out['day_sd']:.1f} {unit}  ·  "
            f"cell-to-cell ±{out['cell_sd']:.1f} {unit}  ·  "
            f"between-replicate share {100 * share:.1f} %"
            f"   (n = {out['n_days']} replicates, {out['n_cells']} cells; spread of replicate MEANS "
            f"of per-cell {quantity}; {reason}: the movie layer, the largest one measured "
            f"on real data, is folded into 'cell')")
        out["binding_layer"] = "day" if share > 0.5 else "cell"

    # BOTH HALVES, and the first one was briefly lost when the movie layer was added.
    # The interpretation half says what the movie layer IS; the decision half says what a
    # share MEANS for the choice of unit, which is the reason the number is on screen at
    # all. Replacing the second with the first left a figure nobody could act on.
    out["note"] = (
        "a movie is one field of view, so its cells share focus and illumination; with far "
        "more movies than replicates the movie is usually the better middle unit. "
        "A small between-replicate share means a hierarchical resample gains power; "
        "a large one means the replicate limits the result and more cells cannot help")
    return out
