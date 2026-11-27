"""A nested linear mixed model for two conditions: cells within movies within replicate days.

Asked for (14 Oct 2026): "per day sd welch is good, but can we have more of strict mixed model
statistics as well that looks at the distribution of the data, but not like on v2 there
everything was significant". Ruled: cells within days, a parametric-bootstrap p, offered beside
Welch in the Export dialog.

THE MODEL. For each cell, its slow fraction ``y`` (fitted with the condition's D held fixed):

    y = b0 + b1 * [condition B] + u_day + v_movie + e_cell,
    u ~ N(0, s_day^2),  v ~ N(0, s_movie^2) nested in days,  e ~ N(0, s_cell^2).

THE MOVIE LAYER IS IN IT ON PURPOSE: `aggregate.layer_report` measured it as the LARGEST source of
spread on a lab project (more of the variance than the day), because the cells of one field share
focus, illumination and density. A model of cells in days alone would count them as independent
and be too optimistic -- the v2 failure again, one level up. Rule 0 holds structurally: a day
belongs to ONE condition (days of different conditions share nothing but an integer), so day and
movie labels are qualified by condition before anything is fitted.

WHY v2 MADE EVERYTHING SIGNIFICANT, AND WHY THIS CANNOT: v2 treated every displacement as an
independent sample, so its n was tens of thousands and any difference got a tiny p. Here the
condition effect is judged against the scatter of DAYS and MOVIES; with three days per side the
information about b1 is close to that of the day means, and only where days agree do the cells
add power.

FITTED BY REML (`fit`), natively (no statsmodels: the installer stays lean and the arithmetic is
checked here). The likelihood is in CLOSED FORM (`_core`: Woodbury on the nested compound
symmetry, so no matrix the size of the data is ever formed), evaluated for many variance triples
in one call, which makes the optimiser's gradient a single call and a bootstrap draw a few ms.

THE p IS A PARAMETRIC BOOTSTRAP (`test`): the model is refitted without the condition effect, data
are simulated from that null fit with the observed design B times, and p is the share of
simulated |t| at least the observed one, (k + 1) / (B + 1). Wald or Satterthwaite p values are
too optimistic with three days per condition. `scripts/mixedmodel_calibration.py` measured it on
simulated experiments with day and movie scatter (14 Oct 2026): 3-4 % false positives at alpha
0.05 (Welch on day means 2.5-5 %; every cell independent, v2's way, 40-43 %), power a few points
above Welch's. The table is in docs/claude/sqd_and_statistics.md ("THE STRICTER SECOND TEST").
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

__all__ = ["MixedModelError", "MixedFit", "MixedTest", "fit", "test", "N_BOOT"]

#: Bootstrap draws behind a p; the smallest reportable p is 1 / (N_BOOT + 1).
N_BOOT = 499


class MixedModelError(ValueError):
    """The data cannot carry the model (too few days, a condition missing), named."""


@dataclass
class MixedFit:
    beta: np.ndarray                     # (b0, b1) or (b0,) for the null
    se: np.ndarray
    s_day: float
    s_movie: float
    s_cell: float
    reml: float
    n_cells: int
    n_movies: int
    n_days: int

    @property
    def t(self) -> float:
        return float(self.beta[-1] / self.se[-1]) if len(self.beta) > 1 and self.se[-1] > 0 else 0.0


@dataclass
class MixedTest:
    conditions: tuple                    # (A, B), in the order the rows named them
    mean_a: float                        # the model's mean of A (GLS over its days)
    mean_b: float
    difference: float                    # b1: condition B minus condition A
    se: float                            # REML's model SE: optimistic with few days; p is the bootstrap's
    t: float
    p: float
    n_boot: int
    s_day: float
    s_movie: float
    s_cell: float
    n_cells: tuple                       # (A, B)
    n_movies: tuple
    n_days: tuple
    notes: list = field(default_factory=list)


@dataclass
class _Design:
    """The nesting as integer indices: each cell's movie and day, each movie's day, each day's
    condition (0 or 1), and cells per movie. Everything the closed forms need."""
    cell_movie: np.ndarray
    cell_day: np.ndarray
    movie_day: np.ndarray
    day_b: np.ndarray
    n: np.ndarray
    #: (movies, days) indicator: a per-movie array times it is the per-day sums, for many
    #: variance triples at once (`_core`)
    to_day: np.ndarray

    @property
    def n_days(self) -> int:
        return len(self.day_b)


def _design(groups, b) -> _Design:
    """Days and movies are qualified by condition, so a day holds one condition."""
    day_ix, movie_ix = {}, {}
    cell_day = np.empty(len(groups), dtype=int)
    cell_movie = np.empty(len(groups), dtype=int)
    movie_day = []
    day_b = []
    for i, (d, m) in enumerate(groups):
        if d not in day_ix:
            day_ix[d] = len(day_ix)
            day_b.append(b[i])
        if m not in movie_ix:
            movie_ix[m] = len(movie_ix)
            movie_day.append(day_ix[d])
        cell_day[i], cell_movie[i] = day_ix[d], movie_ix[m]
    movie_day = np.array(movie_day, dtype=int)
    to_day = np.zeros((len(movie_day), len(day_ix)))
    to_day[np.arange(len(movie_day)), movie_day] = 1.0
    return _Design(cell_movie=cell_movie, cell_day=cell_day, movie_day=movie_day,
                   day_b=np.array(day_b, dtype=float),
                   n=np.bincount(cell_movie, minlength=len(movie_ix)).astype(float),
                   to_day=to_day)


def _sums(design: _Design, y):
    """Per movie: sum y and sum y^2 (with the cell counts, all the closed forms need)."""
    m = len(design.n)
    return (np.bincount(design.cell_movie, y, minlength=m),
            np.bincount(design.cell_movie, y * y, minlength=m))


def _core(design: _Design, sums, s2, *, with_condition=True):
    """The REML log-likelihood, the GLS estimate and its covariance at each ROW of ``s2``
    (``(P, 3)``: day, movie, cell variances), in CLOSED FORM and for all rows at once.

    A day's covariance is s_cell^2 I + s_movie^2 (same movie) + s_day^2 (all ones), and every
    cell of a day has the same design row (the day belongs to one condition). Per movie
    (Woodbury on compound symmetry), with c = s_cell^2 + n s_movie^2:
    1'A^-1 1 = n / c, 1'A^-1 y = S / c, y'A^-1 y = (Q - s_movie^2 S^2 / c) / s_cell^2,
    log|A| = (n - 1) log s_cell^2 + log c; then the day term once more with s_day^2. With
    W0, W1, V0, V1 the weighted sums over all days and over condition B's days, X'V^-1 X is
    [[W0, W1], [W1, W1]], so condition A's mean is (V0 - V1) / (W0 - W1) and B's V1 / W1.
    Checked against the dense matrices in `tests/test_mixedmodel.py`. Many rows in one call
    is what makes the optimiser's gradient one call (`_fit_arrays`)."""
    S, Q = sums
    n = design.n
    s2 = np.atleast_2d(np.asarray(s2, dtype=float))
    s2d, s2m, s2e = s2[:, 0:1], s2[:, 1:2], s2[:, 2:3]
    c = s2e + n * s2m
    a = (n / c) @ design.to_day
    by = (S / c) @ design.to_day
    yay = ((Q - s2m * S * S / c) / s2e) @ design.to_day
    k = 1.0 + s2d * a
    logdet = ((n - 1) * np.log(s2e) + np.log(c)).sum(1) + np.log(k).sum(1)
    w, v = a / k, by / k
    yVy = (yay - s2d * by * by / k).sum(1)
    W0, V0 = w.sum(1), v.sum(1)
    if with_condition:
        W1, V1 = w @ design.day_b, v @ design.day_b
        WA = W0 - W1
        mean_a, mean_b = (V0 - V1) / WA, V1 / W1
        beta = np.column_stack([mean_a, mean_b - mean_a])
        quad = yVy - (beta[:, 0] * V0 + beta[:, 1] * V1)
        logdet_x = np.log(W1) + np.log(WA)
        # inverse of [[W0, W1], [W1, W1]]: [[1/WA, -1/WA], [-1/WA, W0/(W1 WA)]]
        cov = np.stack([np.column_stack([1.0 / WA, -1.0 / WA]),
                        np.column_stack([-1.0 / WA, W0 / (W1 * WA)])], axis=1)
    else:
        beta = (V0 / W0)[:, None]
        quad = yVy - V0 * V0 / W0
        logdet_x = np.log(W0)
        cov = (1.0 / W0)[:, None, None]
    p_ = beta.shape[1]
    ll = -0.5 * (logdet + quad + (float(n.sum()) - p_) * np.log(2 * np.pi) + logdet_x)
    return beta, cov, ll


def _gls(design: _Design, sums, s2, *, with_condition=True):
    """`_core` at one variance triple: ``(beta, covariance, REML log-likelihood)``."""
    beta, cov, ll = _core(design, sums, np.asarray(s2, dtype=float)[None, :],
                          with_condition=with_condition)
    return beta[0], cov[0], float(ll[0])


_STARTS = ((-1.0, -1.0, -0.5), (-4.0, -1.0, -0.2), (-1.0, -4.0, -0.2), (-6.0, -6.0, 0.0))
#: The central-difference step of the gradient, in log variance.
_STEP = 1e-5
_SHIFTS = np.vstack([np.zeros(3), _STEP * np.eye(3), -_STEP * np.eye(3)])


def _fit_arrays(design, sums, scale, *, with_condition=True, starts=_STARTS):
    """REML over the log variances (relative to ``scale``), the best of ``starts``. The
    function and its central-difference gradient are ONE `_core` call over seven rows."""
    scale = max(float(scale), 1e-10)

    def value_and_gradient(log_s2):
        with np.errstate(all="ignore"):
            ll = _core(design, sums, np.exp(log_s2 + _SHIFTS) * scale,
                       with_condition=with_condition)[2]
        if not np.all(np.isfinite(ll)):
            return np.inf, np.zeros(3)
        return -ll[0], -(ll[1:4] - ll[4:7]) / (2 * _STEP)

    best = None
    for start in starts:
        res = minimize(value_and_gradient, np.asarray(start, float), jac=True,
                       method="L-BFGS-B", bounds=[(-18, 3)] * 3)
        if best is None or res.fun < best.fun:
            best = res
    s2 = np.exp(best.x) * scale
    beta, cov, ll = _gls(design, sums, s2, with_condition=with_condition)
    return beta, np.sqrt(np.diag(cov)), s2, ll


def _arrays(cells):
    """``cells``: rows of (condition, day, movie, y). Two conditions; labels qualified."""
    conds = sorted({c for c, _d, _m, _y in cells}, key=lambda c: [r[0] for r in cells].index(c))
    if len(conds) != 2:
        raise MixedModelError(f"the model compares two conditions, got {len(conds)}")
    y = np.array([float(v) for _c, _d, _m, v in cells])
    if not np.all(np.isfinite(y)):
        raise MixedModelError("a cell's value is not a number")
    b = np.array([1.0 if c == conds[1] else 0.0 for c, _d, _m, _y in cells])
    groups = [(f"{c}|{d}", f"{c}|{d}|{m}") for c, d, m, _y in cells]
    return conds, y, b, groups


def fit(cells) -> MixedFit:
    """REML fit of the nested model with a condition effect. ``cells``: (condition, day, movie,
    value) rows for exactly two conditions."""
    conds, y, b, groups = _arrays(cells)
    design = _design(groups, b)
    beta, se, s2, ll = _fit_arrays(design, _sums(design, y), np.var(y))
    return MixedFit(beta=beta, se=se, s_day=float(np.sqrt(s2[0])), s_movie=float(np.sqrt(s2[1])),
                    s_cell=float(np.sqrt(s2[2])), reml=float(ll), n_cells=len(y),
                    n_movies=len(design.n), n_days=design.n_days)


def test(cells, *, n_boot: int = N_BOOT, seed: int = 0, min_days: int = 2) -> MixedTest:
    """Condition B minus condition A, its standard error, and a parametric-bootstrap p.

    Refuses (named) when either condition has fewer than ``min_days`` replicate days: the day
    variance and the condition effect cannot then be told apart, and a p would rest on cells."""
    conds, y, b, groups = _arrays(cells)
    design = _design(groups, b)
    per = (int(np.sum(design.day_b == 0)), int(np.sum(design.day_b == 1)))
    short = [c for c, k in zip(conds, per) if k < min_days]
    if short:
        raise MixedModelError(
            f"{', '.join(short)}: fewer than {min_days} replicates; the replicate scatter and the "
            f"condition effect cannot be told apart, and a p from cells alone would be "
            f"pseudoreplication")
    scale = np.var(y)
    sums = _sums(design, y)
    beta, se, s2, _ll = _fit_arrays(design, sums, scale)
    t_obs = abs(beta[1] / se[1]) if se[1] > 0 else 0.0
    beta0, _se0, s2_0, _ll0 = _fit_arrays(design, sums, scale, with_condition=False)
    # EVERY BOOTSTRAP REFIT STARTS AT THE NULL FIT's components (the truth it was drawn from),
    # one start: the draws are close to it, and four starts each would cost four times as much.
    warm = (tuple(np.log(np.maximum(s2_0, 1e-12) / max(scale, 1e-10))),)
    rng = np.random.default_rng(seed)
    sd = np.sqrt(s2_0)
    m_movies = len(design.n)
    exceed = 0
    for _ in range(int(n_boot)):
        sim = (beta0[0] + rng.normal(0.0, sd[0], design.n_days)[design.cell_day]
               + rng.normal(0.0, sd[1], m_movies)[design.cell_movie]
               + rng.normal(0.0, sd[2], len(y)))
        bb, sb, _s, _l = _fit_arrays(design, _sums(design, sim), scale, starts=warm)
        if (abs(bb[1] / sb[1]) if sb[1] > 0 else 0.0) >= t_obs:
            exceed += 1
    p = (exceed + 1) / (int(n_boot) + 1)
    movies = tuple(int(np.sum(design.day_b[design.movie_day] == v)) for v in (0.0, 1.0))
    return MixedTest(conditions=tuple(conds), mean_a=float(beta[0]),
                     mean_b=float(beta[0] + beta[1]), difference=float(beta[1]), se=float(se[1]),
                     t=float(beta[1] / se[1]) if se[1] > 0 else 0.0, p=float(p), n_boot=int(n_boot),
                     s_day=float(np.sqrt(s2[0])), s_movie=float(np.sqrt(s2[1])),
                     s_cell=float(np.sqrt(s2[2])),
                     n_cells=(int(np.sum(b == 0)), int(np.sum(b == 1))), n_movies=movies,
                     n_days=per)
