"""Uncertainty tiers: point / per-unit analytic / bootstrap. One source of truth.

The expensive thing is REFITTING, not resampling. Fit each independent unit
(cell or day) ONCE (which the SuperPlot needs anyway) and the mean, the SD
(biological spread), the SEM/CI (uncertainty of the estimate) and a cell-level
permutation p all fall out for free. So the bootstrap is opt-in, not the default,
and making it opt-in does NOT drop the pseudoreplication guard: that guard is the
cheap per-unit permutation test, not the bootstrap. See the "uncertainty tiers"
section of CLAUDE.md.

Three tiers:
- ``none``: point estimate only, no interval (the instant first look).
- ``analytic``: per-unit mean ± SD/√n (SEM) + a permutation p. The everyday
  default when you want uncertainty. One fit per unit, no bootstrap.
- ``bootstrap``: the full cell/day resample (slow, opt-in, paper numbers).

``escalation_reasons`` names where the analytic interval genuinely misleads, so a
view can auto-offer the bootstrap: near the 0/1 boundary (a symmetric SEM runs off
the edge and hides the skew), for rates (nonlinear, don't average across units),
and for unstable per-unit sets (too few units).

Pure numpy/scipy; SD/SEM/CI are closed-form, the permutation is over unit values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

__all__ = [
    "TIERS",
    "AnalyticSummary",
    "analytic_summary",
    "escalation_reasons",
    "permutation_p",
    "MIN_UNITS",
    "BOUNDARY_MARGIN",
    "T_CRITICAL_95",
    "DESIGN_DAY_RANGE",
    "unit_interval",
    "two_sample_interval",
    "WelchTest",
    "welch_test",
    "EquivalenceTest",
    "equivalence_test",
    "minimum_effect_test",
    "holm",
    "benjamini_hochberg",
    "CORRECTIONS",
    "CORRECTION_NAMES",
    "adjust",
    "pairing_diagnostic",
]

#: The selectable tiers, cheapest first.
TIERS = ("none", "analytic", "bootstrap")

#: Two-sided 95 % Student-t critical values, keyed by DEGREES OF FREEDOM (n - 1).
#:
#: THIS IS NOT THE COMPUTE PATH. :func:`unit_interval` calls ``scipy.stats.t.ppf``, so a
#: missing key is impossible by construction. The table is kept for exactly two jobs:
#:
#: 1. the deposit figure script (``scripts/manuscript_figures.py``) is FORBIDDEN by
#:    ``tests/test_deposit.py::TestFirewall`` from importing ``smtracker`` or ``scipy``,
#:    so it must carry a literal table; this one is what its copy is diffed against;
#: 2. it documents the values, and a test asserts it still matches scipy at every key,
#:    so the reference cannot drift from the truth either.
#:
#: WHY THE TABLE STOPPED BEING THE COMPUTE PATH (AP1b.2). The overnight defect was a
#: missing key falling back to the normal 1.96, 24-38 % too narrow. Four copies held
#: only df <= 2 and seven of eleven fell back at df >= 6. It never fired, because every
#: MSD group has exactly 3 days. But ``docs/design_for_next_batch.md`` recommends 5-13
#: replicate days, i.e. df up to 12: the batch designed to FIX the 2-df problem is
#: precisely the batch that would have made this defect live, and it would have
#: manufactured significance rather than earned it. Widening the table would have moved
#: the same class of defect one batch further out, so the table was removed from the
#: compute path instead.
T_CRITICAL_95 = {
    1: 12.7062, 2: 4.3027, 3: 3.1824, 4: 2.7764, 5: 2.5706,
    6: 2.4469, 7: 2.3646, 8: 2.3060, 9: 2.2622, 10: 2.2281,
    11: 2.2010, 12: 2.1788, 13: 2.1604, 14: 2.1448, 15: 2.1314,
    16: 2.1199, 17: 2.1098, 18: 2.1009, 19: 2.0930, 20: 2.0860,
    21: 2.0796, 22: 2.0739, 23: 2.0687, 24: 2.0639, 25: 2.0595,
    26: 2.0555, 27: 2.0518, 28: 2.0484, 29: 2.0452, 30: 2.0423,
}

#: The replicate-day range ``design_for_next_batch.md`` recommends. Any table that
#: claims to serve this project must cover df = n - 1 across it.
DESIGN_DAY_RANGE = (5, 13)

#: mean ± SD/√n needs at least this many units to mean anything.
MIN_UNITS = 3
#: A fraction within this of 0 or 1 is "near the boundary" (analytic CI misleads).
BOUNDARY_MARGIN = 0.1


def unit_interval(values, *, confidence: float = 0.95):
    """Mean and two-sided t interval over the analysis unit (cells or days).

    One source of truth for the interval this project quotes. Eleven scripts had each
    hard-coded their own copy of the t table, which is fine until one of them is missing
    a key: the multiplier then silently becomes the normal 1.96, which is smaller than
    every t and so NARROWS the interval, and a narrower interval reads as a better
    result. See ``docs/interval_audit.md``.

    The multiplier is COMPUTED (``scipy.stats.t.ppf``), never looked up, so no degree of
    freedom can be missing and no confidence level needs tabulating. scipy is already a
    core dependency of this package.

    ``values`` is one number per unit, never per step or per localisation: the unit is
    the cell or the replicate day, and feeding it correlated within-unit observations is
    the pseudoreplication this project measured at a 65 % false-positive rate.

    Returns ``(mean, lo, hi, n)``. With fewer than two units the interval is ``nan``
    rather than an arbitrary wide default: an interval that cannot be estimated is
    reported as absent, not invented.
    """
    import numpy as np
    from scipy.stats import t as _t

    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence!r}")
    v = np.asarray([x for x in np.asarray(values, dtype=float).ravel()
                    if np.isfinite(x)], dtype=float)
    n = len(v)
    if n < 2:
        return (float(v[0]) if n else float("nan"), float("nan"), float("nan"), n)
    crit = float(_t.ppf(0.5 + confidence / 2.0, n - 1))
    half = crit * v.std(ddof=1) / np.sqrt(n)
    return float(v.mean()), float(v.mean() - half), float(v.mean() + half), n


def two_sample_interval(a, b, *, confidence: float = 0.95):
    """Welch interval for ``mean(b) - mean(a)`` over the analysis unit. UNPAIRED.

    The design correction (P0). ``day`` is a **within-group ordinal, not a date**:
    ``day_map_from_series_blocks`` renumbers the series blocks present *in order of
    occurrence within a group*, so nothing ties one condition's day 1 to the other's
    day 1 except that both columns carry the same name. Differencing them per day was
    D11's shape one level up: two things joined because the label matched.

    So the two conditions are two independent samples of three replicate days each.

    Welch, not pooled-VARIANCE. The per-condition SDs differ by up to 3x here. Pooling assumes an equality of variance we do not have, and it
    would narrow the interval by assuming it.

    This narrows every interval relative to the paired model, i.e. it moves in the
    direction that FLATTERS the effect. The mechanism is degrees of freedom, not data:
    a paired test at n = 3 has df = 2 and t = 4.303, while Welch on 3 vs 3 has df in
    roughly 2.0-4.0 and t nearer 2.8-3.7. Quote that sentence beside the numbers.

    Returns ``(difference, lo, hi, df, n_a, n_b)``. ``df`` is the Welch-Satterthwaite
    value and is NOT an integer; it is returned because a reader will want it.
    Fewer than two units on either side gives ``nan`` bounds rather than an invented
    default. So does NO SPREAD ON EITHER SIDE (every value identical within each sample;
    10 Nov 2026): until then the bounds were the difference itself, a zero-width interval,
    which states a certainty that zero variance in two or three replicates cannot carry --
    `welch_test` already refused the p there for the same reason. ``df`` stays ``inf``.

    `welch_test` adds t and p to exactly this interval: both read `_welch_core`, so the
    interval a figure's source data quotes and the p its bracket prints cannot disagree.
    """
    from scipy.stats import t as _t

    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence!r}")
    x, y, diff, se, df = _welch_core(a, b)
    n_a, n_b = len(x), len(y)
    if n_a < 2 or n_b < 2:
        return (diff, float("nan"), float("nan"), float("nan"), n_a, n_b)
    if se == 0.0:
        return (diff, float("nan"), float("nan"), float("inf"), n_a, n_b)
    half = float(_t.ppf(0.5 + confidence / 2.0, df)) * se
    return (diff, diff - half, diff + half, df, n_a, n_b)


def _welch_core(a, b):
    """``(x, y, mean(y) - mean(x), standard error, Welch-Satterthwaite df)`` over the
    finite values of ``a`` and ``b``. The ONE place the Welch arithmetic lives
    (`two_sample_interval`, `welch_test`). With fewer than two values on either side the
    error and df are ``nan``; with no spread on either side the error is 0 and df ``inf``.
    """
    x = np.asarray([v for v in np.asarray(a, dtype=float).ravel() if np.isfinite(v)])
    y = np.asarray([v for v in np.asarray(b, dtype=float).ravel() if np.isfinite(v)])
    n_a, n_b = len(x), len(y)
    diff = float(_centre(y) - _centre(x)) if n_a and n_b else float("nan")
    if n_a < 2 or n_b < 2:
        return x, y, diff, float("nan"), float("nan")
    va, vb = _spread(x) / n_a, _spread(y) / n_b
    se = float(np.sqrt(va + vb))
    if se == 0.0:
        return x, y, diff, 0.0, float("inf")
    df = float((va + vb) ** 2 / (va ** 2 / (n_a - 1) + vb ** 2 / (n_b - 1)))
    return x, y, diff, se, df


def _constant(v) -> bool:
    return bool(len(v)) and bool(np.all(v == v[0]))


def _centre(v) -> float:
    """The mean, EXACT for a sample whose values are all identical (np.mean of three 0.4s
    is 0.4000000000000001)."""
    return float(v[0]) if _constant(v) else float(v.mean())


def _spread(v) -> float:
    """The sample variance, EXACTLY 0 for a sample whose values are all identical. np.var
    leaves rounding noise there (about 1e-33 for three 0.4s), which made t finite and
    INVENTED a p (0.29 for two identical constant samples) where none exists."""
    return 0.0 if _constant(v) else float(v.var(ddof=1))


@dataclass(frozen=True)
class WelchTest:
    """Welch's two-sample t-test on UNIT-LEVEL values, for ``mean(b) - mean(a)``. UNPAIRED.

    ``difference``, ``ci_low`` and ``ci_high`` are `two_sample_interval`'s, to the bit.
    ``t = difference / SE``, so its sign follows ``b - a`` (scipy's ``ttest_ind(a, b)``
    reports ``a - b``: the same number with the other sign). ``df`` is Welch-Satterthwaite,
    not an integer. ``p`` is two-sided. When no p exists, ``p`` is ``nan`` and ``reason``
    says why -- never an invented number.
    """

    difference: float
    ci_low: float
    ci_high: float
    t: float
    df: float
    p: float
    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    confidence: float = 0.95
    reason: str = ""


def welch_test(a, b, *, confidence: float = 0.95) -> WelchTest:
    """Welch's t-test, two-sided, for ``mean(b) - mean(a)``: TWO INDEPENDENT SAMPLES.

    The test a significance bracket on a figure prints (ruled 13 Oct 2026): one value per
    REPLICATE DAY on each side -- each condition's days one independent sample, never
    matched by day number (RULE 0: a day number is an ordinal within its condition, not a
    date), never per cell or per track (pseudoreplication, measured here at a 65 %
    false-positive rate). Welch rather than pooled variance, for the reason
    `two_sample_interval` gives: the per-condition SDs differ by up to threefold here.

    Dropping the pairing narrows every interval relative to the paired model, through the
    degrees of freedom rather than the data, which is the direction that FLATTERS the
    effect; whoever quotes this p says so beside it.

    No p is invented. Fewer than two finite values on either side, or no spread on either
    side (every value identical within each sample: t is 0/0 when the means agree and
    infinite when they differ, and neither is evidence) give ``p = nan`` and a ``reason``.
    """
    from scipy.stats import t as _t

    diff, lo, hi, df, n_a, n_b = two_sample_interval(a, b, confidence=confidence)
    x, y, _diff, se, _df = _welch_core(a, b)
    mean_a = _centre(x) if n_a else float("nan")
    mean_b = _centre(y) if n_b else float("nan")
    nan = float("nan")
    if n_a < 2 or n_b < 2:
        return WelchTest(diff, lo, hi, nan, nan, nan, n_a, n_b, mean_a, mean_b, confidence,
                         f"Welch's t needs at least two values on each side; there are "
                         f"{n_a} and {n_b}")
    if se == 0.0:
        why = ("every value on both sides is the same: with no spread t is 0/0 and there "
               "is no p" if diff == 0.0 else
               "neither side has any spread (its values are all identical): t is "
               "infinite, and a p from zero variance would be an artefact, not evidence")
        return WelchTest(diff, lo, hi, nan, df, nan, n_a, n_b, mean_a, mean_b, confidence,
                         why)
    t_value = diff / se
    p = float(min(1.0, 2.0 * _t.sf(abs(t_value), df)))
    return WelchTest(diff, lo, hi, float(t_value), df, p, n_a, n_b, mean_a, mean_b,
                     confidence, "")


@dataclass(frozen=True)
class EquivalenceTest:
    """Two one-sided Welch tests (TOST) that ``mean(b) - mean(a)`` lies within ``+-margin``. UNPAIRED.

    ``p`` is the larger of the two one-sided p values: below alpha the two are EQUIVALENT within the margin, which
    a non-significant Welch p never shows. ``ci_low`` / ``ci_high`` are the 1 - 2 alpha interval (90 % at alpha
    0.05), the interval TOST is equivalent to: equivalence is shown exactly when it lies inside the margin. Fields
    as `WelchTest`'s, so the tables read either; ``t`` is the one-sided statistic that gave ``p``."""

    difference: float
    ci_low: float
    ci_high: float
    t: float
    df: float
    p: float
    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    margin: float
    p_lower: float
    p_upper: float
    confidence: float = 0.90
    reason: str = ""


def paired_test(a, b, *, confidence: float = 0.95) -> WelchTest:
    """The paired t-test, two-sided, for ``mean(b - a)``: ``a[i]`` and ``b[i]`` ONE DECLARED PAIR each.

    OPT-IN AND DECLARED, NEVER INFERRED (5 Nov 2026, asked: "optional pair ... in day picker?", "a tick box for
    activate pairing"). Rule 0 stands for everything else: a replicate number is an ordinal within its condition, so
    nothing pairs replicates by number. A pair here is two replicates a PERSON tagged alike on the group board (one
    culture split, one imaging session; `daymap.DayMapping.pairs`), and the caller passes them aligned. The test is
    on the differences: ``t = mean(d) / (sd(d) / sqrt(n))`` on ``n - 1`` degrees of freedom, the interval
    ``mean(d) +- t * sd(d) / sqrt(n)``.

    Fields as `WelchTest`'s (``n_a`` and ``n_b`` are both the number of pairs; ``df`` is ``n - 1``), so the tables
    read either. No p is invented: fewer than two complete pairs, or differences with no spread, give ``p = nan``
    and a ``reason``."""
    from scipy.stats import t as _t

    x = np.asarray(a, dtype=float).ravel()
    y = np.asarray(b, dtype=float).ravel()
    if len(x) != len(y):
        raise ValueError(f"a paired test needs one value of each side per pair, got {len(x)} and {len(y)}")
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    n = int(len(x))
    nan = float("nan")
    d = y - x
    mean_a = _centre(x) if n else nan
    mean_b = _centre(y) if n else nan
    diff = _centre(d) if n else nan
    if n < 2:
        return WelchTest(diff, nan, nan, nan, nan, nan, n, n, mean_a, mean_b, confidence,
                         f"a paired t-test needs at least two declared pairs; there {'is' if n == 1 else 'are'} {n}")
    # differences equal to within rounding (0.4 - 0.3 and 0.5 - 0.4) have no spread: no t from the noise of the sums
    flat = float(np.ptp(d)) <= 1e-12 * max(1.0, float(np.max(np.abs(d))))
    se = 0.0 if flat else float(np.sqrt(_spread(d) / n))
    df = float(n - 1)
    if se == 0.0:
        why = ("every pair differs by exactly the same amount: with no spread t is "
               + ("0/0" if diff == 0.0 else "infinite") + ", and a p from zero variance would be an artefact")
        return WelchTest(diff, nan, nan, nan, df, nan, n, n, mean_a, mean_b, confidence, why)
    half = _t_critical(df, confidence) * se
    t_value = diff / se
    p = float(min(1.0, 2.0 * _t.sf(abs(t_value), df)))
    return WelchTest(diff, diff - half, diff + half, float(t_value), df, p, n, n, mean_a, mean_b, confidence, "")


def equivalence_test(a, b, margin: float, *, alpha: float = 0.05) -> EquivalenceTest:
    """TOST on per-replicate values (asked 24 Oct 2026: "significant different or similar"): is ``mean(b) -
    mean(a)`` inside ``(-margin, +margin)``? The same Welch arithmetic as `welch_test` (`_welch_core`), each
    condition's replicates one independent sample (RULE 0). ``margin`` is in the values' own unit and must be
    chosen BEFORE looking: a margin fitted to the data shows whatever it was fitted to show."""
    from scipy.stats import t as _t

    if not (isinstance(margin, (int, float)) and math.isfinite(margin) and margin > 0):
        raise ValueError(f"the equivalence margin is a positive number, got {margin!r}")
    if not 0.0 < alpha < 0.5:
        raise ValueError(f"alpha must be in (0, 0.5), got {alpha!r}")
    confidence = 1.0 - 2.0 * alpha
    x, y, diff, se, df = _welch_core(a, b)
    n_a, n_b = len(x), len(y)
    mean_a = _centre(x) if n_a else float("nan")
    mean_b = _centre(y) if n_b else float("nan")
    nan = float("nan")
    if n_a < 2 or n_b < 2:
        return EquivalenceTest(diff, nan, nan, nan, nan, nan, n_a, n_b, mean_a, mean_b, float(margin), nan, nan,
                               confidence, f"an equivalence test needs at least two values on each side; there are "
                                           f"{n_a} and {n_b}")
    if se == 0.0:
        return EquivalenceTest(diff, diff, diff, nan, df, nan, n_a, n_b, mean_a, mean_b, float(margin), nan, nan,
                               confidence, "neither side has any spread: an equivalence p from zero variance would "
                                           "be an artefact, not evidence")
    t_lower = (diff + margin) / se                 # H0: diff <= -margin
    t_upper = (diff - margin) / se                 # H0: diff >= +margin
    p_lower = float(_t.sf(t_lower, df))
    p_upper = float(_t.cdf(t_upper, df))
    half = float(_t.ppf(1.0 - alpha, df)) * se
    worst_lower = p_lower >= p_upper
    return EquivalenceTest(diff, diff - half, diff + half, float(t_lower if worst_lower else t_upper), df,
                           float(max(p_lower, p_upper)), n_a, n_b, mean_a, mean_b, float(margin), p_lower, p_upper,
                           confidence, "")


def minimum_effect_test(a, b, margin: float, *, confidence: float = 0.95) -> WelchTest:
    """Do the two samples differ by MORE than ``margin`` -- the minimum-effect test, the mirror of `equivalence_test`.

    H0: ``|mean(b) - mean(a)| <= margin``; H1: the difference lies beyond ``+-margin``. Two one-sided Welch tests,
    each at half the level, ``p = min(1, 2 * min(P(T >= (d - m) / se), P(T <= (d + m) / se)))`` on Welch's df: at a
    margin of 0 it is `welch_test`'s two-sided p, to the bit. Everything else (difference, interval, t, df) is
    `welch_test`'s, and so is a refusal (fewer than two values a side, no spread).

    WHY (asked 23 Nov 2026: "My prof said don't make to many movies or everything will get significant"): under
    Uncertainty Chunks the values are pseudo-replicates of ONE culture, and the standard error of a condition's mean
    falls as the square root of its movies whatever the number of chunks, so with enough movies ANY difference between
    two cultures reads as significant. Against a margin set before looking, a difference smaller than the margin never
    does, however much data there is; a real shift beyond it still does."""
    from scipy.stats import t as _t

    base = welch_test(a, b, confidence=confidence)
    if not math.isfinite(base.p):
        return base
    margin = float(margin)
    if not (math.isfinite(margin) and margin >= 0):
        raise ValueError(f"the margin is a difference of 0 or more, got {margin!r}")
    _x, _y, diff, se, df = _welch_core(a, b)
    above = float(_t.sf((diff - margin) / se, df))
    below = float(_t.cdf((diff + margin) / se, df))
    return replace(base, p=float(min(1.0, 2.0 * min(above, below))))


def holm(pvalues) -> list:
    """Holm's step-down adjustment of ``pvalues``, in their own order.

    The smallest of ``m`` p values is multiplied by ``m``, the next by ``m - 1``, and so
    on; each adjusted value is then raised to at least the one before it (the adjusted p
    never falls as the raw p rises) and capped at 1. Holm controls the family-wise error
    rate under any dependence between the tests and is never less powerful than
    Bonferroni, which is why it is the correction for the pairs a figure compares.

    NaN-SAFE: a ``nan`` (a pair with no test) stays ``nan`` and is NOT counted in ``m``,
    since a test that was not performed is not part of the family. A p outside [0, 1]
    raises: it is a defect upstream, not a number to adjust.
    """
    import math

    p = [float(v) for v in pvalues]
    tested = [i for i, v in enumerate(p) if math.isfinite(v)]
    bad = [p[i] for i in tested if not 0.0 <= p[i] <= 1.0]
    if bad:
        raise ValueError(f"a p value must lie in [0, 1]; got {bad}")
    m = len(tested)
    out = [float("nan")] * len(p)
    running = 0.0
    for rank, i in enumerate(sorted(tested, key=lambda k: p[k])):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out


def benjamini_hochberg(pvalues) -> list:
    """Benjamini and Hochberg's step-up adjustment of ``pvalues``, in their own order (10 Nov 2026: a CHOICE beside
    `holm`, which stays the standard; the lab's own scripts and standing legends use this one).

    The i-th smallest of ``m`` p values is multiplied by ``m / i``; each adjusted value is then lowered to at most
    the one after it (the adjusted p never falls as the raw p rises) and capped at 1. It controls the FALSE DISCOVERY
    RATE, the expected share of false positives among the pairs called different, under independence or positive
    dependence of the tests; `holm` controls the chance of ANY false positive, under any dependence. This adjusted p
    is never larger than Holm's, so it is the more lenient of the two; with one test both leave the p unchanged.

    NaN-SAFE exactly as `holm`: a ``nan`` (a pair with no test) stays ``nan`` and is not counted in ``m``; a p
    outside [0, 1] raises.
    """
    import math

    p = [float(v) for v in pvalues]
    tested = [i for i, v in enumerate(p) if math.isfinite(v)]
    bad = [p[i] for i in tested if not 0.0 <= p[i] <= 1.0]
    if bad:
        raise ValueError(f"a p value must lie in [0, 1]; got {bad}")
    m = len(tested)
    out = [float("nan")] * len(p)
    running = 1.0
    ranked = sorted(tested, key=lambda k: p[k])
    for rank in range(m, 0, -1):                              # from the largest p down
        i = ranked[rank - 1]
        running = min(running, min(1.0, m * p[i] / rank))
        out[i] = running
    return out


#: The corrections for several comparisons on one figure: the function and the name a reader sees. Holm first: it
#: is the standard (ruled 13 Oct 2026) and the one a figure without the choice was made with.
CORRECTIONS = {"holm": holm, "bh": benjamini_hochberg}
CORRECTION_NAMES = {"holm": "Holm", "bh": "Benjamini-Hochberg"}


def adjust(pvalues, correction: str = "holm") -> list:
    """``pvalues`` adjusted by ``correction`` (`CORRECTIONS`: "holm", the standard, or "bh"); an unknown name is
    refused, never read as Holm."""
    if correction not in CORRECTIONS:
        raise ValueError(f"the correction is one of {', '.join(CORRECTIONS)}; got {correction!r}")
    return CORRECTIONS[correction](pvalues)


def pairing_diagnostic(a, b):
    """Did matching on ``day`` remove any shared variance? Checked, not assumed.

    ``SD(b - a)`` against ``sqrt(SD(a)^2 + SD(b)^2)``, the value expected if the two
    series are INDEPENDENT. Below 1 means the matching removed shared variance, so a
    pairing existed; **at or above 1 means it removed none**, which is what "the days
    are not pairs" predicts.

    Requires equal-length series in the order the matching used: it is a diagnostic
    ABOUT a matching, so it needs the matching to describe.

    Returns ``(ratio, sd_difference, sd_independent, n)``.
    """
    import numpy as np

    x = np.asarray(a, dtype=float).ravel()
    y = np.asarray(b, dtype=float).ravel()
    if len(x) != len(y):
        raise ValueError(
            f"pairing_diagnostic compares a matching, so the series must be the same "
            f"length and in the matched order; got {len(x)} and {len(y)}")
    if len(x) < 2:
        return (float("nan"), float("nan"), float("nan"), len(x))
    sd_diff = float((y - x).std(ddof=1))
    sd_indep = float(np.sqrt(x.var(ddof=1) + y.var(ddof=1)))
    ratio = sd_diff / sd_indep if sd_indep > 0 else float("nan")
    return (ratio, sd_diff, sd_indep, len(x))


@dataclass
class AnalyticSummary:
    """Per-unit summary: the point, the spread, the uncertainty, and escalation.

    ``sd`` is the biological SPREAD across units; ``sem``/``ci`` is the UNCERTAINTY
    of the mean: the two are different and must not be conflated on a plot.
    ``escalation`` lists reasons (if any) the analytic interval may mislead and the
    bootstrap is worth reaching for; empty means the analytic tier is trustworthy.
    """

    mean: float
    sd: float
    sem: float
    ci: tuple
    n_units: int
    escalation: list = field(default_factory=list)

    @property
    def should_escalate(self) -> bool:
        return bool(self.escalation)


def analytic_summary(unit_values, *, is_fraction=True, is_rate=False,
                     min_units=MIN_UNITS, confidence=0.95) -> AnalyticSummary:
    """Mean ± SD/√n (SEM) with a t-based CI over per-unit values.

    ``unit_values`` is one number per independent unit (per-cell fractions, per-day
    means, …), NEVER per step. The CI uses Student-t with ``n-1`` df, honest for
    the small unit counts here (3–30). Non-finite values are dropped.
    """
    values = np.asarray(list(unit_values), dtype=float)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n == 0:
        return AnalyticSummary(float("nan"), float("nan"), float("nan"),
                               (float("nan"), float("nan")), 0,
                               ["no units"])
    mean = float(values.mean())
    if n == 1:
        summary = AnalyticSummary(mean, float("nan"), float("nan"), (mean, mean), 1,
                                  [])
    else:
        sd = float(values.std(ddof=1))
        sem = sd / np.sqrt(n)
        crit = _t_critical(n - 1, confidence)
        summary = AnalyticSummary(mean, sd, sem,
                                  (mean - crit * sem, mean + crit * sem), n, [])
    summary.escalation = escalation_reasons(
        values, summary, is_fraction=is_fraction, is_rate=is_rate,
        min_units=min_units)
    return summary


def escalation_reasons(unit_values, summary: AnalyticSummary, *, is_fraction=True,
                       is_rate=False, min_units=MIN_UNITS) -> list:
    """Reasons the analytic interval may mislead → prefer the bootstrap.

    - too few units for a mean ± SD/√n to mean anything;
    - a rate (nonlinear; per-unit averaging misstates it);
    - a fraction near the 0/1 boundary, or whose analytic CI runs off [0, 1]
      (the symmetric interval hides the skew there).
    """
    reasons = []
    n = summary.n_units
    if n < min_units:
        reasons.append(f"only {n} unit(s): analytic interval unreliable")
    if is_rate:
        reasons.append("rates are nonlinear and do not average across units")
    if is_fraction and np.isfinite(summary.mean):
        if summary.mean < BOUNDARY_MARGIN or summary.mean > 1 - BOUNDARY_MARGIN:
            reasons.append("near the 0/1 boundary: symmetric interval misleads")
        lo, hi = summary.ci
        if np.isfinite(lo) and (lo < 0.0 or hi > 1.0):
            reasons.append("analytic CI runs outside [0, 1]")
    return reasons


#: Enumerate every split rather than sampling while there are at most this many. At the
#: unit counts this project actually works with, 3 to 6 replicate days, sometimes a
#: dozen: the exact test is both cheaper and correct, so sampling is the fallback.
_EXACT_SPLIT_LIMIT = 200_000


def permutation_p(a, b, *, n_perm=2000, seed=0) -> float:
    """Two-sided permutation test on the difference of means. UNIT-level input.

    ``a`` and ``b`` are one value per independent unit (cells/days), never pooled
    steps: the pseudoreplication guard.

    Exact where it can be, and that is not a refinement: the sampling version was
    WRONG in the direction that manufactures significance. Shuffling changes the
    ORDER the means are summed in, so recomputing the observed split lands a few ULPs
    either side of ``observed``; when it lands below, ``>= observed`` rejects the very
    permutation that generated the statistic. MEASURED on this project's own data
    (an HMM occupancy, 3 days a side): the sampler returned **0.032** where the exact
    answer is **0.100**, a 3.1x understatement that moved a comparison across 0.05.
    The other five comparisons moved 1.0-1.1x, so the error is not a constant offset,
    it is worst exactly when the groups separate cleanly, which is when a p gets quoted.

    Two fixes, both needed: enumerate every split when there are few enough (the regime
    this project works in), and compare with a relative tolerance so a re-summed
    identical value still counts as "at least as extreme".

    Sampling p is floored at ``1/(n_perm+1)`` so it is never reported as an exact zero;
    the exact p carries no such floor because it is a real proportion.

    FOR THE SMALLEST p a design can produce at all, use ``aggregate.achievable_p``,
    which already owns that calculation and handles the sidedness trap (3 v 3 is
    0.05 one-sided and 0.100 two-sided, and a two-condition comparison is two-sided).
    A second copy was written here and removed: one source of truth.
    """
    from itertools import combinations
    from math import comb

    a = np.asarray(list(a), dtype=float)
    b = np.asarray(list(b), dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    observed = abs(a.mean() - b.mean())
    pool = np.concatenate([a, b])
    na, n = a.size, pool.size
    # Ties and re-summation both land within a few ULPs; without this the observed
    # split can fail its own test.
    tol = 1e-9 * max(1.0, abs(observed))

    if comb(n, na) <= _EXACT_SPLIT_LIMIT:
        hits = total = 0
        everything = np.arange(n)
        for idx in combinations(range(n), na):
            mask = np.zeros(n, dtype=bool)
            mask[list(idx)] = True
            total += 1
            if abs(pool[mask].mean() - pool[~mask].mean()) >= observed - tol:
                hits += 1
        return hits / total

    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        if abs(pool[:na].mean() - pool[na:].mean()) >= observed - tol:
            count += 1
    return (count + 1) / (n_perm + 1)


def _t_critical(df, confidence):
    """Two-sided Student-t critical value. FAILS LOUD (10 Nov 2026): until then any failure in
    scipy returned the normal 1.96 in silence, whatever ``confidence`` was asked and however
    few the degrees of freedom (4.30 at df 2), so an interval could come out less than half
    its width with nothing said. scipy is a dependency; where it fails, the caller hears it."""
    from scipy import stats

    return float(stats.t.ppf(0.5 + confidence / 2, df))
