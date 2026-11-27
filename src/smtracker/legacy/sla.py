"""Legacy item 4: the SLA tab, its confinement classification and its p value.

READ ``SMTracker.m:3171`` (``SLAplots``), ``Help_fcn/dwellCount.m``, ``tauCounter3.m``,
``getConfinedSegments.m``, ``confinedMSDMotion.m``, ``ftestAnomalous.m``,
``diffusionClassification.m``, ``getMSD.m`` and ``ftest/ftest.m`` beside this. Every
statement below was confirmed in those files.

Most of this item already exists. ``smtracker.confinement`` reproduces the moving-ROI
dwell detector faithfully, including the shared one-gap allowance and the
endpoint-preserving masking, AND adds the chance rate the legacy never had. What this
module owes is the SLA tab's own surface: the classification of confined segments, the
p value printed in its title, and the counts the tab never showed.

The P value in the title, which is the thing people ask about. A legacy title reading
``F-test: Brownian (p value: 0.17)`` comes from ``diffusionClassification.m``. It is an
F-test between two NESTED fits of one POOLED MSD curve: a straight line through the origin
region (2 parameters) against a power law (3 parameters). ``p`` is the probability that
the power law's improvement in sum of squares is chance. Below 0.1 the power law is
accepted and the label becomes Super-diffusive or Sub-diffusive according to whether the
fitted exponent exceeds 1; at or above 0.1 the label is Brownian. **So "Brownian" means
"the F-test did not reject a straight line at the 0.1 level", not "this molecule diffuses
freely".** The threshold is hard-coded at 0.1 and is DESIGN, reproduced here.

The classification is on a pooled curve, and this project has already measured why THAT
BENDS. ``ftestAnomalous`` calls ``getMSD``, which pools every segment into one weighted
mean curve, and the test runs on that. An ensemble MSD of a mixture of Brownian states
bends downward from population mixing, and its tail sags further from survivorship, since
long tracks are preferentially the low-D ones. Both are documented in this project at
alpha about 0.69 with no anomalous motion present. **A Sub-diffusive verdict from this
test is therefore predicted by mixing and survivorship alone and is not evidence of
anomalous diffusion.** That is the item's limitation line.

Four defects, each confirmed.

**4.1 The error bar labelled "sem" is scaled by the number of LAGS, not the sample size,
and the right number is in the next column of the same matrix.** ``ftestAnomalous.m:18``
passes ``numel(x)`` as ``N``. ``x`` comes from ``getMSD``, whose own header documents it as
``[dT M STD N]``, so ``numel(x)`` is four times the number of lags. ``N`` is then used only
at ``diffusionClassification.m:50``, ``errorbar(ax,t,msd,weights/sqrt(N),'bo')``, against a
legend that reads ``MSD ± sem``. The weights are the STD column, so the intended quantity
is ``STD/sqrt(n)`` with ``n`` the per-lag count, which is **column 4 of the very matrix
``numel`` was called on**. With thousands of tracks per lag against a few tens of lags the
drawn bar is far too LARGE, which makes the curve look less certain than it is.

**4.2 The F-test treats correlated MSD lags as independent data.**
``diffusionClassification.m:11`` sets ``n = size(t,1)``, the number of lags, and passes it
to ``ftest`` as "# of data". Successive lags of one MSD curve reuse the same
displacements, so they are not independent. Same class as the ``nlparci`` intervals on
correlated CDF points, which this project already retired and reproduces only inside
legacy mode.

**4.3 A second p value is computed and thrown away.** ``diffusionClassification.m:17``
computes ``p = 1 - fcdf(Fstat,1,dfF)`` and never reads it; the decision uses ``p_value``
from ``ftest``. Same pattern as the ``1:5:end`` residual subsampling in ``sqdFit_``, which
is also computed and discarded. No reported number changes either way, and both are
recorded so the next reader does not assume the dead branch ran.

**4.4 The motion type is unassigned when the fitted exponent is exactly 1.**
``diffusionClassification.m:29-33`` branches on ``coeff_a(2) > 1`` and ``< 1`` with no
else, so an exponent of exactly 1 below the threshold leaves ``typeMotion`` undefined and
``typeMotionStr{typeMotion}`` raises. Vanishingly rare in floating point and still a hole;
here it is a named error rather than an index failure.

One thing that is unreachable on this data, stated rather than fixed. ``dwellCount.m:19``
indexes ``Lcut(i_gap-1:i_gap+1,:)``, which would raise for a gap in the first or last row
and handles only the first gap when several exist, since MATLAB's colon takes the first
element of each vector. u-track runs here with ``timeWindow = 1`` and this project has
measured gap loss at exactly zero in every group and band, so ``i_gap`` is empty and the
line is a no-op. It is reproduced, and it is NOT presented as a fixed defect.

One thing that looks like an OFF-BY-ONE and is not. ``confinedMSDMotion.m:15`` slices
``x(2:minLength,:)`` with ``minLength = minSteps + 1``, and ``tauCounter3.m:58`` keeps a
segment when ``tau >= minSteps`` where ``tau = k - 1`` counts POINTS, so the shortest kept
segment has exactly ``minSteps + 1`` points. That reads as a slice one row past the end.
It is not: ``calcMSDTracks_.m:20`` builds ``delays = (0:size(dr2,1))``, so the matrix
carries lag 0 at row 1 and the slice lands exactly on the last row. Recorded under
``checked_not_a_defect``.

What is wrong there is the advice. The catch prints "MSD for confined tracks not possible.
Use a larger number of steps". The reachable failure is too FEW confined segments, and
raising ``minSteps`` makes that strictly worse.

v2's settings, reproduced (checked 14 Oct 2026 against the source; each was different here
before, and each is now v2's):

* **minimum 5 steps** in a confined segment: the SLA tab's list defaults to 5
  (``SMTracker.m:2954-2963``) and passes it to ``confTracks``, whose own default of 3
  (``confTracks.m:17``) the GUI never uses. `V2_MIN_STEPS`.
* **lags 1 to minSteps**: each segment's curve is cut to rows ``2:minSteps+1``
  (``confinedMSDMotion.m:13-15``), lag 0 dropped, and nothing beyond. Pairs by ROW
  (``calcMSDTracks_.m:14-18``).
* **the pooled curve is getMSD's**: a mean weighted by each segment's pair count, its
  weighted SD, and its "degrees of freedom" column (``getMSD.m:57-96``), not a plain mean.
* **the F-test weights each lag by its STANDARD DEVIATION** (``ftestAnomalous.m:17``,
  ``w = x(:,3)``), handed to ``linWfit`` and ``anomalousWfit`` as MATLAB fit weights. That
  is a v2 DEFECT -- a weight must fall as the scatter grows, and this one rises, so the
  noisiest lags count most -- and legacy mode REPRODUCES it, recorded in
  ``legacy_deviations.csv`` as ``defect_reproduced``. ``weights="inverse_variance"``
  gives the corrected weighting, so its effect can be measured rather than argued.
* **the power law is anomalousWfit's**: ``b + K t^alpha`` with ``D = K/4``, upper bounds
  ``b <= 1``, ``alpha <= 2`` and no lower bounds (``anomalousWfit.m:25-30``), seeded as v2
  seeds it -- ``[exp(y) alpha D_alpha]`` handed to the coefficients in the order
  ``(b, K, alpha)`` (``anomalousWfit.m:21``), so the exponent starts at ``D_alpha`` and
  the prefactor at the exponent. Also a defect reproduced and recorded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from ..canonical import without_attrs

__all__ = [
    "SLAError", "LEGACY_P_THRESHOLD", "MOTION_TYPES", "POOLED_CURVE_CAVEAT",
    "GAP_ALLOWANCE_UNREACHABLE", "V2_MIN_STEPS", "V2_ANOMALOUS_UPPER", "WEIGHTINGS",
    "legacy_errorbar_divisor", "pooled_msd",
    "fit_linear_weighted", "fit_anomalous_weighted", "classify_motion",
    "sla_counts",
]


class SLAError(RuntimeError):
    """Raised when a legacy SLA result cannot be produced honestly."""


#: ``diffusionClassification.m:28``. Hard-coded, and DESIGN rather than a defect.
LEGACY_P_THRESHOLD = 0.1

#: ``diffusionClassification.m:44``, in the legacy's own order.
MOTION_TYPES = ("Brownian", "Super-diffusive", "Sub-diffusive")

#: The SLA tab's "minimum steps" list, default 5 (``SMTracker.m:2954-2963``). The 3 in
#: ``confTracks.m:17`` is a function default the GUI overrides, and the 3 in params.mat is
#: never read.
V2_MIN_STEPS = 5

#: ``anomalousWfit.m:30``: ``'Upper',[1,Inf,2]`` for the coefficients ``(b, K, alpha)``,
#: and no ``'Lower'``. So the offset is at most 1 um^2 and the exponent at most 2.
V2_ANOMALOUS_UPPER = (1.0, np.inf, 2.0)

#: ``"v2"`` is ``ftestAnomalous.m:17``, the lag's standard deviation used AS the weight
#: (a defect, reproduced); ``"inverse_variance"`` is ``1/std**2``, the corrected one.
WEIGHTINGS = ("v2", "inverse_variance")

#: The limitation this item is really about. Printed with every classification.
POOLED_CURVE_CAVEAT = (
    "this verdict comes from an F-test on ONE POOLED MSD curve, not from any single "
    "molecule. An ensemble MSD of a mixture of Brownian states bends downward from "
    "population mixing, and its tail sags further from survivorship because long tracks "
    "are preferentially the low-D ones; this project measures an exponent near 0.69 from "
    "those two causes with no anomalous motion present. A sub-diffusive verdict here is "
    "predicted by mixing and survivorship alone and is not evidence of anomalous "
    "diffusion. Brownian means only that a straight line was not rejected at "
    f"p = {LEGACY_P_THRESHOLD}")

#: Stated wherever the gap allowance is reproduced.
#:
#: RENAMED, NOT EXCEPTED. This string is rendered on a figure, and naming the MATLAB file
#: it comes from would put a blocked word on the canvas. The precise file and line live in
#: legacy_deviations.csv, which is the one source of truth for source references and is
#: not a rendered surface. Renaming is the project's standing rule; an exception list would
#: grow until the gate meant nothing.
GAP_ALLOWANCE_UNREACHABLE = (
    "the legacy one-gap allowance in the confinement counter is reproduced and never "
    "fires on this data: u-track runs with timeWindow = 1 and gap loss is measured at "
    "exactly zero in every group and band, so the gap index is empty. This is a "
    "reproduced no-op, not a fixed defect")


def legacy_errorbar_divisor(msd_matrix: np.ndarray) -> dict:
    """The legacy ``N`` for the error bar, and the number it should have been.

    ``ftestAnomalous.m:18`` passes ``numel(x)`` where ``x`` is the ``[dT M STD N]`` matrix
    from ``getMSD``. Both are returned and neither is called the difference; the ratio is
    what the drawn bar is wrong by.
    """
    m = np.asarray(msd_matrix, dtype=float)
    if m.ndim != 2 or m.shape[1] < 3:
        raise SLAError("expected the getMSD matrix [dT, M, STD, N]")
    n_lags = int(m.shape[0])
    legacy = int(m.size)                       # numel(x), all columns
    out = {"legacy_N": legacy, "n_lags": n_lags, "n_columns": int(m.shape[1])}
    if m.shape[1] >= 4:
        per_lag = m[:, 3]
        out["per_lag_n_median"] = float(np.median(per_lag))
        out["intended_N_lives_in_column_4"] = True
        with np.errstate(divide="ignore", invalid="ignore"):
            out["errorbar_too_large_by"] = float(
                np.sqrt(np.median(per_lag) / legacy)) if legacy else float("nan")
    else:
        out["intended_N_lives_in_column_4"] = False
    out["why"] = (
        "numel on an n-by-{c} matrix is {c} times the number of lags, and the per-lag "
        "count the error bar needs is column 4 of that same matrix".format(
            c=int(m.shape[1])))
    return out


def pooled_msd(curves, *, min_curves: int = 1) -> np.ndarray:
    """``getMSD``: one weighted mean MSD curve over many per-segment curves, as v2 pools.

    Each curve is ``[dT, M]``, ``[dT, M, n]`` or ``[dT, M, n, STD]``, with ``n`` the pairs
    behind each lag (column 4 of ``calcMSDTracks_``); without ``n`` every curve weighs 1.
    Returns ``[dT, M, STD, N]`` exactly as ``getMSD.m:57-96`` builds it: ``M`` the mean
    weighted by ``n``, ``STD = sqrt(sw / (sw^2 - ssw) * sum n (m - M)^2)``, and ``N =
    sw^2 / ssw``, "the number of degrees of freedom in the weighted mean" (its header),
    which is the number of curves when every weight is 1. A lag with ONE contributing
    curve has ``sw^2 = ssw``, and v2's ``0/0`` gives NaN there, reproduced; the fit then
    drops that lag, as ``prepareCurveData`` does. ONE curve in all is returned as it is
    (``getMSD.m:13-14``), its own spread as STD.
    """
    curves = [np.asarray(c, dtype=float) for c in curves if len(c)]
    if len(curves) < min_curves:
        raise SLAError(f"{len(curves)} curves, need at least {min_curves}")
    if any(c.ndim != 2 or c.shape[1] < 2 for c in curves):
        raise SLAError("each curve must be [dT, M], [dT, M, n] or [dT, M, n, STD]")
    if len(curves) == 1:                                     # getMSD.m:13-14
        c = curves[0]
        n = c[:, 2] if c.shape[1] > 2 else np.ones(len(c))
        std = c[:, 3] if c.shape[1] > 3 else np.full(len(c), np.nan)
        return np.column_stack([c[:, 0], c[:, 1], std, n])
    delays = np.unique(np.concatenate([c[:, 0] for c in curves]))
    rows = []
    for t in delays:
        m_parts, n_parts = [], []
        for c in curves:
            at = c[:, 0] == t
            m = c[at, 1]
            n = c[at, 2] if c.shape[1] > 2 else np.ones(int(at.sum()))
            ok = np.isfinite(m)                              # getMSD.m:48, NaN skipped
            m_parts.append(m[ok])
            n_parts.append(n[ok])
        m, n = np.concatenate(m_parts), np.concatenate(n_parts)
        if not len(m):
            continue
        sw, ssw = float(n.sum()), float((n ** 2).sum())
        mean = float((n * m).sum() / sw)
        swv = float((n * (m - mean) ** 2).sum())
        denom = sw ** 2 - ssw
        std = float(np.sqrt(sw / denom * swv)) if denom > 0 else float("nan")
        rows.append([t, mean, std, sw ** 2 / ssw])
    if not rows:
        raise SLAError("no lag had a finite value")
    return np.asarray(rows, dtype=float)


def _weighted_lsq(design: np.ndarray, y: np.ndarray, w: np.ndarray):
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(design * sw[:, None], y * sw, rcond=None)
    resid = y - design @ coef
    sse = float((w * resid ** 2).sum())
    return coef, sse


def fit_linear_weighted(t, msd, weights) -> dict:
    """``linWfit``: a weighted straight line, ``D = slope / 4`` for two dimensions."""
    t, msd, w = (np.asarray(a, dtype=float) for a in (t, msd, weights))
    if len(t) < 3:
        raise SLAError("a two-parameter fit needs at least three points")
    coef, sse = _weighted_lsq(np.column_stack([t, np.ones_like(t)]), msd, w)
    return {"D_um2_per_s": float(coef[0] / 4.0), "intercept_um2": float(coef[1]),
            "sse": sse, "n_params": 2, "dfe": int(len(t) - 2)}


def fit_anomalous_weighted(t, msd, weights, *, v2_start: bool = True) -> dict:
    """``anomalousWfit``: ``msd = b + K t**alpha``, ``D = K / 4``, three parameters.

    v2's fit, reproduced: a weighted straight line through ``(log t, log msd)`` gives the
    exponent and ``y``, ``D_alpha = exp(y) Gamma(1 + alpha) / 4``, and the start point
    ``[exp(y) alpha D_alpha]`` is handed to the coefficients in their declared order
    ``(b, K, alpha)`` (``anomalousWfit.m:11-27``) -- so the OFFSET starts at the prefactor,
    the prefactor at the exponent and the exponent at ``D_alpha``. That order is a v2
    defect, reproduced and recorded; ``v2_start=False`` seeds ``(0, exp(y), alpha)``
    instead. Bounds ``b <= 1``, ``alpha <= 2``, none below (``anomalousWfit.m:30``); a
    start outside them is moved onto them. Trust-region least squares at MATLAB fit's
    default tolerances (1e-6, at most 600 evaluations).
    """
    t, msd, w = (np.asarray(a, dtype=float) for a in (t, msd, weights))
    if len(t) < 4:
        raise SLAError("a three-parameter fit needs at least four points")
    if np.any(t <= 0) or np.any(msd <= 0):
        raise SLAError(
            "the power law is seeded from log(t) and log(MSD), and a lag or an MSD at or "
            "below zero has no logarithm (MATLAB would carry on with a complex number)")
    from scipy.optimize import least_squares
    from scipy.special import gamma as gamma_fn

    slope, y0 = _weighted_lsq(np.column_stack([np.log(t), np.ones_like(t)]),
                              np.log(msd), w)[0]
    D_alpha0 = float(np.exp(y0) * gamma_fn(1.0 + slope) / 4.0)
    asked = ([float(np.exp(y0)), float(slope), D_alpha0] if v2_start
             else [0.0, float(np.exp(y0)), float(slope)])
    lower = np.full(3, -np.inf)
    upper = np.asarray(V2_ANOMALOUS_UPPER, dtype=float)
    start = np.minimum(np.asarray(asked), upper)

    def residual(p):
        return np.sqrt(w) * (msd - (p[0] + p[1] * t ** p[2]))

    fit = least_squares(residual, start, bounds=(lower, upper), method="trf",
                        xtol=1e-6, ftol=1e-6, max_nfev=600)
    b, K, alpha = (float(v) for v in fit.x)
    return {"D_alpha": K / 4.0, "alpha": alpha, "intercept_um2": b,
            "sse": float((fit.fun ** 2).sum()), "n_params": 3, "dfe": int(len(t) - 3),
            "start_asked": asked, "start_used": [float(v) for v in start],
            "start_order": ("v2: [exp(y) alpha D_alpha] into (b, K, alpha), "
                            "anomalousWfit.m:21" if v2_start else "(0, exp(y), alpha)"),
            # trust-region stays strictly inside the box, so "at" means within 1e-6.
            "at_upper_bound": [name for name, v, u in zip(("b", "K", "alpha"), fit.x,
                                                           upper)
                               if np.isfinite(u) and v >= u - 1e-6 * max(1.0, abs(u))]}


def classify_motion(msd_matrix: np.ndarray, *, p_threshold: float = LEGACY_P_THRESHOLD,
                    n_independent: int | None = None, weights: str = "v2") -> dict:
    """``diffusionClassification``: the F-test, its p value and its label.

    ``n_independent`` is the sample size the F-test uses. The legacy passes the number of
    MSD LAGS, which are not independent of one another; that is the reproduced design and
    is what ``None`` means here. Supplying a smaller, honest count is offered so the
    difference can be MEASURED rather than argued.

    ``weights`` (`WEIGHTINGS`): ``"v2"`` weights each lag by its STANDARD DEVIATION, as
    ``ftestAnomalous.m:17`` does -- a defect, reproduced, since legacy mode shows v2;
    ``"inverse_variance"`` by ``1/std**2``. Lags whose MSD, lag or weight is not finite are
    dropped first, as MATLAB's ``prepareCurveData`` drops them, and the lag count the
    F-test uses is counted after that (``diffusionClassification.m:11``).
    """
    if weights not in WEIGHTINGS:
        raise SLAError(f"weights must be one of {WEIGHTINGS}, got {weights!r}")
    m = np.asarray(msd_matrix, dtype=float)
    if m.ndim != 2 or m.shape[1] < 3:
        raise SLAError("expected the getMSD matrix [dT, M, STD, N]")
    t, msd, std = m[:, 0], m[:, 1], m[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        w = std.copy() if weights == "v2" else 1.0 / std ** 2
    usable = np.isfinite(t) & np.isfinite(msd) & np.isfinite(w)
    t, msd, w = t[usable], msd[usable], w[usable]
    if np.any(w < 0):
        raise SLAError("a negative weight: the spread column holds a negative value")
    lin = fit_linear_weighted(t, msd, w)
    ano = fit_anomalous_weighted(t, msd, w)

    n = int(len(t)) if n_independent is None else int(n_independent)
    if n <= ano["n_params"]:
        raise SLAError(
            f"{n} data against {ano['n_params']} parameters; the F-test has no degrees "
            f"of freedom left")
    dfe_full = n - ano["n_params"]
    # A RELATIVE FLOOR, NOT A POSITIVE ONE. A noiseless curve leaves both models with a
    # residual of order 1e-26 against data of order 1, and their RATIO is then
    # floating-point dust: the F statistic comes back around 3 and the verdict is
    # whatever the rounding decided. Testing sse > 0 does not catch that, because the
    # dust is positive. The scale to compare against is the weighted total sum of
    # squares, which is what the residual would be if the model explained nothing.
    scale = float((w * (msd - np.average(msd, weights=w)) ** 2).sum())
    # EITHER MODEL AT ROUNDING LEVEL. The power law is fitted at MATLAB fit's tolerance
    # (1e-6, as v2 ran it), so on a perfect straight line it may stop short of rounding
    # while the straight line reaches it exactly; the numerator is then dust as well.
    smallest = min(ano["sse"], lin["sse"])
    if smallest <= 1e-18 * max(scale, 1e-300):
        raise SLAError(
            f"a model fits to within {smallest:.3g} against a weighted total sum of "
            f"squares of {scale:.3g}, so the F statistic is a ratio of rounding error "
            f"and its p value means nothing. A curve with no scatter cannot be "
            f"classified; supply the measured spread.")
    f_stat = ((lin["sse"] - ano["sse"]) / 1.0) / (ano["sse"] / dfe_full)
    f_stat = max(f_stat, 0.0)
    p_value = float(stats.f.sf(f_stat, 1, dfe_full))

    if p_value < p_threshold:
        if ano["alpha"] > 1.0:
            motion = "Super-diffusive"
        elif ano["alpha"] < 1.0:
            motion = "Sub-diffusive"
        else:
            raise SLAError(
                "the fitted exponent is exactly 1 while the F-test rejected a straight "
                "line, which the legacy leaves unassigned (diffusionClassification.m:"
                "29-33 has no else) and which then raises on the label lookup. There is "
                "no honest label for this case, so it is named rather than indexed")
        chosen = ano
    else:
        motion = "Brownian"
        chosen = lin

    return {
        "motion": motion, "p_value": p_value, "f_statistic": float(f_stat),
        "p_threshold": float(p_threshold),
        "n_used_in_test": n, "n_lags": int(len(t)),
        "n_lags_dropped": int((~usable).sum()),
        "n_is_the_lag_count": n_independent is None,
        "weights": weights,
        "weights_note": ("each lag weighted by its standard deviation, as v2 does "
                         "(a defect, reproduced)" if weights == "v2" else
                         "each lag weighted by 1/std^2 (corrected, not v2)"),
        "alpha": ano["alpha"], "D_um2_per_s": lin["D_um2_per_s"],
        "sse_linear": lin["sse"], "sse_anomalous": ano["sse"],
        "chosen_model": "anomalous" if chosen is ano else "linear",
        "title": f"F-test: {motion} (p value: {p_value:.4g})",
        "caveat": POOLED_CURVE_CAVEAT,
        "correlated_lags_note": (
            "the F-test's sample size is the number of MSD lags, and successive lags of "
            "one curve reuse the same displacements, so they are not independent"),
    }


def sla_counts(events: pd.DataFrame, *, n_tracks: int, n_localisations: int) -> dict:
    """The counts the SLA tab never printed beside its classification.

    Both the track count and the localisation count, never their difference, and the
    number of confined segments the curve was actually built from.
    """
    if n_tracks <= 0:
        raise SLAError("a group with no tracks has no SLA result")
    n_segments = int(len(events))
    return {
        "n_tracks": int(n_tracks),
        "n_localisations": int(n_localisations),
        "n_confined_segments": n_segments,
        "n_tracks_with_a_segment": (int(events["uid"].nunique())
                                    if n_segments and "uid" in events.columns else 0),
        "segments_per_track": n_segments / n_tracks,
        "gap_allowance": GAP_ALLOWANCE_UNREACHABLE,
    }


def classify_from_canonical(canonical, *, radius_um: float, min_steps: int = V2_MIN_STEPS,
                            max_lag: int | None = None, min_curves: int = 1,
                            p_threshold: float = LEGACY_P_THRESHOLD,
                            n_independent: int | None = None,
                            weights: str = "v2") -> dict:
    """The catalogue target: a canonical table in, the motion classification out.

    `classify_motion` takes a pooled MSD matrix - the output of two earlier steps - so the
    catalogue entry resolved, passed every signature check, and raised the moment anyone
    clicked it. This runs the chain.

    The curves are the confined segments', not every track's, and that distinction is the
    whole point of this analysis: pooling every track answers a different question with an
    identical-looking answer. `radius_um` therefore has NO DEFAULT - it decides what
    counts as confined, a random walk revisits its own neighbourhood, and at 150 nm and
    D = 0.2 um^2/s 79 per cent of purely FREE localisations are flagged confined.

    The pooled curve's caveat travels with the result rather than sitting in a docstring
    nobody reads at the moment of use: an ensemble curve of a mixture is dominated by the
    fast state and sags in the tail through survivorship, so an alpha below 1 is NOT
    evidence of anomalous diffusion.

    v2's window: ``min_steps`` 5 (the SLA tab's default) and ``max_lag=None`` meaning lags
    1 to ``min_steps`` (``confinedMSDMotion.m:13-15``); curves paired by ROW
    (``calcMSDTracks_``) and pooled by `pooled_msd` with each segment's pair counts.
    """
    import numpy as np

    from ..confinement import confined_segments
    from ..msd import msd_per_track

    segments = confined_segments(canonical, radius_um=radius_um, min_steps=min_steps)
    if len(segments) == 0:
        raise SLAError(
            f"no confined segment at radius {radius_um} um with min_steps {min_steps}, "
            f"so there is no pooled curve to classify. Widen the radius or lower the "
            f"minimum, and report which was used")

    tau = canonical.attrs.get("tau_s")
    if tau is None:
        params = canonical.attrs.get("params")
        if params is None:
            raise SLAError(
                "no frame interval: the canonical table carries neither 'tau_s' nor "
                "AcquisitionParams in .attrs, and every lag time depends on it")
        tau = float(params.frame_interval_ms) / 1000.0

    # confinedMSDMotion.m:13-15: rows 2:minSteps+1, lags 1..minSteps, nothing beyond.
    max_lag = int(min_steps) if max_lag is None else int(max_lag)
    per_track = msd_per_track(segments, max_lag=max_lag, pairing="row")
    curves = []
    for _, part in without_attrs(per_track).groupby("uid", sort=False):
        part = part.sort_values("lag_frames")
        curves.append(np.column_stack([part["lag_frames"].to_numpy(float) * float(tau),
                                       part["msd_um2"].to_numpy(float),
                                       part["n_pairs"].to_numpy(float),
                                       part["std_um2"].to_numpy(float)]))
    out = classify_motion(pooled_msd(curves, min_curves=min_curves),
                          p_threshold=p_threshold, n_independent=n_independent,
                          weights=weights)
    out["n_curves_pooled"] = len(curves)
    out["radius_um"] = float(radius_um)
    out["min_steps"] = int(min_steps)
    out["max_lag"] = int(max_lag)
    out["caveat"] = POOLED_CURVE_CAVEAT
    # A NEGATIVE D IS NOT A SMALL D. The F-test compares two fits, so it can prefer the
    # linear model simply because the anomalous one returned something impossible, and
    # the verdict then reads as evidence for Brownian motion when it is evidence that one
    # fit failed. Measured on a lab project at radius 0.10 um: alpha 2.77 with
    # D = -0.006 um^2/s, reported as "Brownian (p = 1)". Named rather than raised,
    # because reproducing the legacy verdict is this entry's whole purpose.
    fitted_D = float(out.get("D_um2_per_s", 0.0))
    out["fit_is_physical"] = bool(fitted_D > 0.0)
    if fitted_D <= 0.0:
        out["title"] = out.get("title", "") + "  [UNPHYSICAL FIT: D <= 0]"
        out["fit_warning"] = (
            f"the fit returned D = {fitted_D:.4g} um^2/s, which is impossible. The "
            f"F-test may have chosen its model because the alternative failed rather "
            f"than because this one is right; do not read the verdict as evidence.")
    return out
