"""Tests on nested data -- cells in movies in replicates -- beside Welch and the mixed model. Headless.

Asked 24 Oct 2026, after five methods and a shuffle were tried side by side on one user's protein F data ("lets try all
of them u listed plus the shuffle i mentioned"; then "1, good idea. 2. true. 3. we can include if people do more
replicates?"; "Ur proposal sounds good build it"):

* `replicate_shuffle` -- DO THE REPLICATES DIFFER? The spread of the replicate means (each the mean of its movies)
  against the same spread after the movies are dealt out at random into groups of the replicates' own sizes. QC's
  question, not a condition test: a small p says one culture came out unlike the others, which is the scatter every
  condition comparison is judged against. On one lab project it flags the one replicate that sits well
  above the other two; the other conditions and a free mNG set pass.
* `bayes_test` -- a Bayesian hierarchical model on each cell's fraction: cell = condition + replicate + movie + cell
  noise, half-normal priors on the three SDs (`PRIOR_SCALES`, written into every legend), Gibbs for the means and
  effects, random-walk Metropolis on the log SDs, two chains (`RHAT_LIMIT` refuses a run that did not mix). It is
  the one method here that stays sensible with ONE replicate on a side: the replicate SD is learnt from the other
  side's replicates and the prior, and the interval says how little that is. Its p is the two-sided posterior tail
  probability, 2 min(P(diff < 0), P(diff > 0)), read like a p and marked as such.
* `meta_test` -- random-effects meta-regression over replicates: each replicate's step-weighted fraction with its
  own variance (a cluster-robust sandwich over its movies), DerSimonian-Laird between-replicate variance, the
  condition as the moderator, Knapp-Hartung's t on k - 2 df with the variance never below DerSimonian-Laird's.
  Offered from `MIN_REPLICATES_RICH` replicates a side: with two, one df is left and the interval explodes (on
  a 1 v 2 lab comparison it ran to +-160 points).
* `hierarchical_bootstrap` -- replicates resampled within each condition, then movies within each replicate, then
  cells within each movie; the step-weighted fraction of what was drawn; percentile interval and a two-sided p from
  the share of differences across zero. From `MIN_REPLICATES_RICH` a side too: with three only ten distinct replicate
  draws exist (`distinct_draws`), which the result carries as a caveat.

THE INPUT is what the mixed model reads, plus each cell's step count: rows ``(condition, day, movie, value[,
n_steps])``, two conditions, A first, so every difference is B minus A. The fraction of each cell is fitted with
its condition's shared D held (`comparison.condition_replicate_fit`), so cells of one condition are one quantity.
RULE 0: replicate labels are qualified by their condition; replicate 1 of A is never matched to replicate 1 of B.

FAIL LOUD: every refusal is a `NestedError` naming what is missing; nothing returns a made-up p.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = ["NestedError", "NestedTest", "ShuffleTest", "PRIOR_SCALES", "RHAT_LIMIT", "MIN_REPLICATES_RICH",
           "METHOD_NAMES", "replicate_shuffle", "bayes_test", "meta_test", "hierarchical_bootstrap",
           "distinct_draws", "cluster_variance"]

#: Half-normal prior scales on the SD between replicates, between movies and between cells, in FRACTION units
#: (0.10 = 10 percentage points). Weakly informative: the SDs measured on a lab project (24 Oct 2026) fall well
#: inside these scales, so a prior an order wider than the data would allow nothing it needs to rule out.
PRIOR_SCALES = (0.10, 0.10, 0.30)
#: Gelman-Rubin's R-hat on the difference above which the two chains have not met: refused, never reported.
RHAT_LIMIT = 1.05
#: Replicates a side the meta-analysis and the hierarchical bootstrap need (asked: "include if people do more
#: replicates"): below three each has at most one degree of freedom or three distinct replicate draws.
MIN_REPLICATES_RICH = 3
#: The words a legend and a table use for each method.
METHOD_NAMES = {
    "bayes": ("Bayesian hierarchical model on each cell's fraction (cells within movies within replicates; "
              "half-normal priors on the SDs), two-sided posterior tail probability"),
    "meta": ("random-effects meta-regression over replicates (DerSimonian-Laird, Knapp-Hartung t), each "
             "replicate's step-weighted fraction with its movie-clustered variance"),
    "hboot": ("hierarchical bootstrap (replicates, then movies, then cells resampled), step-weighted fraction, "
              "percentile interval and two-sided p"),
}


class NestedError(ValueError):
    """A nested test that cannot be run on these data; the message says why."""


@dataclass(frozen=True)
class NestedTest:
    """Condition B minus condition A by one method. Read through the fields a `uncertainty.WelchTest` has, so the
    comparison tables need no branch per method; ``t`` and ``df`` are NaN where a method has none."""

    method: str
    conditions: tuple
    mean_a: float
    mean_b: float
    difference: float
    ci_low: float
    ci_high: float
    p: float
    n_days: tuple
    n_movies: tuple
    n_cells: tuple
    t: float = float("nan")
    df: float = float("nan")
    confidence: float = 0.95
    caveat: str = ""
    #: method-specific numbers: Bayes ``s_day``, ``s_movie``, ``s_cell`` (posterior medians), ``rhat``, ``draws``,
    #: ``prior``; meta ``tau2``, ``replicate_values``, ``replicate_variances``; hboot ``n_boot``, ``distinct``
    details: dict = field(default_factory=dict)
    reason: str = ""

    @property
    def n_a(self) -> int:
        return int(self.n_days[0])

    @property
    def n_b(self) -> int:
        return int(self.n_days[1])


@dataclass(frozen=True)
class ShuffleTest:
    """`replicate_shuffle`'s answer for one condition. ``spread`` and the null's are SDs of replicate means, in the
    values' own unit; ``p`` the share of shuffles at least as spread, (k + 1) / (n + 1)."""

    p: float
    spread: float
    null_median: float
    null_low: float
    null_high: float
    replicate_means: tuple
    replicates: tuple
    movies_per_replicate: tuple
    n_shuffles: int


# --------------------------------------------------------------------------- the input
def _rows(cells):
    """``(conds, y, w, cond_index, rep_index, movie_index, rep_cond, movie_rep, rep_labels)`` of two conditions."""
    rows = [tuple(r) for r in cells]
    if any(len(r) not in (4, 5) for r in rows):
        raise NestedError("each cell is (condition, replicate, movie, value[, steps])")
    conds = []
    for r in rows:
        if r[0] not in conds:
            conds.append(r[0])
    if len(conds) != 2:
        raise NestedError(f"a test compares two conditions, got {len(conds)}")
    y = np.array([float(r[3]) for r in rows])
    w = np.array([float(r[4]) if len(r) == 5 else 1.0 for r in rows])
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(w)) and np.all(w > 0)):
        raise NestedError("a cell's value or step count is not a positive number")
    reps, movies = {}, {}
    rep_cond, movie_rep, ci, ri, mi = [], [], [], [], []
    for r in rows:
        k = conds.index(r[0])
        rkey = (k, str(r[1]))                       # RULE 0: a replicate belongs to its condition
        if rkey not in reps:
            reps[rkey] = len(reps)
            rep_cond.append(k)
        mkey = (k, str(r[1]), str(r[2]))
        if mkey not in movies:
            movies[mkey] = len(movies)
            movie_rep.append(reps[rkey])
        ci.append(k)
        ri.append(reps[rkey])
        mi.append(movies[mkey])
    labels = [key for key, _ in sorted(reps.items(), key=lambda kv: kv[1])]
    return (conds, y, w, np.array(ci), np.array(ri), np.array(mi), np.array(rep_cond), np.array(movie_rep),
            labels)


def _counts(ci, rep_cond, movie_rep):
    n_days = tuple(int(np.sum(rep_cond == k)) for k in (0, 1))
    n_movies = tuple(int(np.sum(rep_cond[movie_rep] == k)) for k in (0, 1))
    n_cells = tuple(int(np.sum(ci == k)) for k in (0, 1))
    return n_days, n_movies, n_cells


def _need(n_days, least: int, method: str) -> None:
    short = [f"{n} replicate{'s' if n != 1 else ''}" for n in n_days if n < least]
    if short:
        raise NestedError(
            f"{METHOD_NAMES[method].split(' (')[0]} needs {least} replicates on each side; there are "
            f"{n_days[0]} and {n_days[1]}")


def _weighted(y, w) -> float:
    return float(np.sum(w * y) / np.sum(w))


# --------------------------------------------------------------------------- do the replicates differ?
def replicate_shuffle(values, replicates, *, n_shuffles: int = 4999, seed: int = 0) -> ShuffleTest:
    """Do one condition's replicates differ more than random groups of its movies would?

    ``values``: one number per movie (its slow fraction with the condition's D held); ``replicates``: each movie's
    replicate. The statistic is the SD of the replicate means (each replicate the plain mean of its movies); the
    null deals the movies out at random into groups of the replicates' own sizes. A shuffle at least as spread as
    the real one counts, with a relative tolerance so a re-summed identical split is never lost (the permutation
    test once rejected its own split by a few ULPs, `uncertainty.permutation_p`). Needs two replicates with at
    least two movies each."""
    v = np.asarray(values, float)
    lab = np.asarray([str(r) for r in replicates])
    if len(v) != len(lab) or not np.all(np.isfinite(v)):
        raise NestedError("one finite value per movie, with its replicate")
    names = list(dict.fromkeys(lab.tolist()))
    sizes = [int(np.sum(lab == n)) for n in names]
    if len(names) < 2:
        raise NestedError("one replicate: there is nothing to compare it with")
    if min(sizes) < 2:
        raise NestedError("a replicate with a single movie: its mean is that movie, and a shuffle of movies cannot "
                          "tell a replicate from a movie")
    codes = np.array([names.index(x) for x in lab])
    k = len(names)

    def spread(c):
        means = np.bincount(c, weights=v, minlength=k) / np.bincount(c, minlength=k)
        return float(np.std(means, ddof=1)), means

    real, means = spread(codes)
    rng = np.random.default_rng(seed)
    null = np.array([spread(rng.permutation(codes))[0] for _ in range(int(n_shuffles))])
    exceed = int(np.sum(null >= real * (1.0 - 1e-9) - 1e-12))
    p = (exceed + 1) / (int(n_shuffles) + 1)
    lo, med, hi = np.percentile(null, [2.5, 50, 97.5])
    return ShuffleTest(p=float(p), spread=real, null_median=float(med), null_low=float(lo), null_high=float(hi),
                       replicate_means=tuple(float(m) for m in means), replicates=tuple(names),
                       movies_per_replicate=tuple(sizes), n_shuffles=int(n_shuffles))


# --------------------------------------------------------------------------- the Bayesian hierarchical model
def _gibbs(y, ci, ri, mi, n_r, n_m, rng, n_iter, burn, scales):
    """One chain: draws of (mu_A, mu_B, s_rep, s_movie, s_cell) after ``burn``."""
    n = len(y)
    n_ci = np.bincount(ci, minlength=2).astype(float)
    n_ri = np.bincount(ri, minlength=n_r).astype(float)
    n_mi = np.bincount(mi, minlength=n_m).astype(float)
    mu = np.array([np.mean(y[ci == k]) for k in (0, 1)]) + rng.normal(0.0, 0.02, 2)
    a, b = np.zeros(n_r), np.zeros(n_m)
    sd0 = max(float(np.std(y)), 1e-3)
    logsd = np.log(np.array([0.05, 0.05, sd0])) + rng.normal(0.0, 0.3, 3)
    scales = np.asarray(scales, float)

    def log_post(ls, resid):
        ta, tb, s = np.exp(ls)
        ll = (-0.5 * np.dot(a, a) / ta ** 2 - n_r * ls[0] - 0.5 * np.dot(b, b) / tb ** 2 - n_m * ls[1]
              - 0.5 * np.dot(resid, resid) / s ** 2 - n * ls[2])
        return ll - 0.5 * float(np.sum((np.exp(ls) / scales) ** 2)) + float(np.sum(ls))   # half-normal, Jacobian

    keep = np.empty((n_iter - burn, 5))
    for it in range(n_iter):
        ta, tb, s = np.exp(logsd)
        r = y - a[ri] - b[mi]
        mu = np.bincount(ci, weights=r, minlength=2) / n_ci + rng.normal(0.0, 1.0, 2) * s / np.sqrt(n_ci)
        r = y - mu[ci] - b[mi]
        prec = n_ri / s ** 2 + 1.0 / ta ** 2
        a = (np.bincount(ri, weights=r, minlength=n_r) / s ** 2) / prec + rng.normal(0.0, 1.0, n_r) / np.sqrt(prec)
        r = y - mu[ci] - a[ri]
        prec = n_mi / s ** 2 + 1.0 / tb ** 2
        b = (np.bincount(mi, weights=r, minlength=n_m) / s ** 2) / prec + rng.normal(0.0, 1.0, n_m) / np.sqrt(prec)
        resid = y - mu[ci] - a[ri] - b[mi]
        lp = log_post(logsd, resid)
        for _ in range(3):
            prop = logsd + rng.normal(0.0, 0.3, 3)
            lp_new = log_post(prop, resid)
            if math.log(rng.uniform()) < lp_new - lp:
                logsd, lp = prop, lp_new
        if it >= burn:
            keep[it - burn, :2] = mu
            keep[it - burn, 2:] = np.exp(logsd)
    return keep


def _rhat(chains) -> float:
    """Gelman-Rubin's potential scale reduction of one quantity over ``chains`` (equal lengths)."""
    x = np.asarray(chains, float)
    m, n = x.shape
    w = float(np.mean(np.var(x, axis=1, ddof=1)))
    b = n * float(np.var(np.mean(x, axis=1), ddof=1))
    if w <= 0:
        return float("inf") if b > 0 else 1.0
    return math.sqrt(((n - 1) / n * w + b / n) / w)


def bayes_test(cells, *, n_iter: int = 4000, burn: int = 1000, chains: int = 2, seed: int = 0,
               scales=PRIOR_SCALES, confidence: float = 0.95) -> NestedTest:
    """Condition B minus A from the Bayesian hierarchical model (module docstring). Needs at least one replicate
    on each side and three in all, so the replicate SD rests on data and not on its prior alone; refuses a run
    whose chains did not meet (`RHAT_LIMIT`)."""
    conds, y, w, ci, ri, mi, rep_cond, movie_rep, _labels = _rows(cells)
    n_days, n_movies, n_cells = _counts(ci, rep_cond, movie_rep)
    if min(n_days) < 1 or sum(n_days) < 3:
        raise NestedError(
            f"the Bayesian model needs three replicates in all (one side may have one); there are {n_days[0]} and "
            f"{n_days[1]}, so the replicate SD would come from its prior alone")
    if min(n_cells) < 3:
        raise NestedError("the Bayesian model needs at least three cells on each side")
    rng = np.random.default_rng(seed)
    runs = [_gibbs(y, ci, ri, mi, len(rep_cond), len(movie_rep), rng, n_iter, burn, scales) for _ in range(chains)]
    diffs = np.array([r[:, 1] - r[:, 0] for r in runs])
    rhat = _rhat(diffs)
    if not math.isfinite(rhat) or rhat > RHAT_LIMIT:
        raise NestedError(f"the Bayesian model's chains did not meet (R-hat {rhat:.3f} > {RHAT_LIMIT}); "
                          f"no result is reported")
    draws = np.concatenate(runs)
    d = draws[:, 1] - draws[:, 0]
    tail = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(d, [tail, 1.0 - tail])
    n_draws = len(d)
    p = min(1.0, 2.0 * min(int(np.sum(d <= 0)), int(np.sum(d >= 0)) ) / n_draws)
    p = max(p, 1.0 / n_draws)                      # never 0: the draws cannot resolve below one in their number
    sds = np.median(draws[:, 2:], axis=0)
    caveat = ""
    if min(n_days) == 1:
        caveat = ("one side has a single replicate: its scatter is the other side's and the prior's, not its own")
    return NestedTest(method="bayes", conditions=tuple(conds), mean_a=float(np.mean(draws[:, 0])),
                      mean_b=float(np.mean(draws[:, 1])), difference=float(np.mean(d)), ci_low=float(lo),
                      ci_high=float(hi), p=float(p), n_days=n_days, n_movies=n_movies, n_cells=n_cells,
                      confidence=confidence, caveat=caveat,
                      details={"s_day": float(sds[0]), "s_movie": float(sds[1]), "s_cell": float(sds[2]),
                               "rhat": float(rhat), "draws": int(n_draws), "prior": tuple(float(s) for s in scales)})


# --------------------------------------------------------------------------- the meta-analysis
def cluster_variance(y, w, clusters) -> float:
    """The variance of the step-weighted mean ``sum(w y) / sum(w)`` with ``clusters`` (movies) as the independent
    units: the ratio estimator's sandwich, M / (M - 1) sum_m (S_m - theta W_m)^2 / W^2. NaN with one cluster."""
    y, w = np.asarray(y, float), np.asarray(w, float)
    ids = np.unique(np.asarray(clusters), return_inverse=True)[1]
    m = int(ids.max()) + 1 if len(ids) else 0
    if m < 2:
        return float("nan")
    total = float(np.sum(w))
    theta = float(np.sum(w * y)) / total
    s_m = np.bincount(ids, weights=w * y, minlength=m)
    w_m = np.bincount(ids, weights=w, minlength=m)
    return float(m / (m - 1) * np.sum((s_m - theta * w_m) ** 2) / total ** 2)


def meta_test(cells, *, confidence: float = 0.95) -> NestedTest:
    """Condition B minus A by random-effects meta-regression over the replicates (module docstring)."""
    from scipy.stats import t as tdist

    conds, y, w, ci, ri, mi, rep_cond, movie_rep, labels = _rows(cells)
    n_days, n_movies, n_cells = _counts(ci, rep_cond, movie_rep)
    _need(n_days, MIN_REPLICATES_RICH, "meta")
    k = len(rep_cond)
    theta, v = np.empty(k), np.empty(k)
    for r in range(k):
        sel = ri == r
        theta[r] = _weighted(y[sel], w[sel])
        clusters = mi[sel] if len(np.unique(mi[sel])) >= 2 else np.arange(int(sel.sum()))
        v[r] = cluster_variance(y[sel], w[sel], clusters)
        if not math.isfinite(v[r]) or v[r] <= 0:
            raise NestedError(f"replicate {labels[r][1]} of {conds[labels[r][0]]} has no within-replicate variance "
                              f"(one cell, or every cell the same)")
    x = np.column_stack([np.ones(k), rep_cond.astype(float)])
    wf = 1.0 / v
    xtwx = x.T @ (wf[:, None] * x)
    beta_f = np.linalg.solve(xtwx, x.T @ (wf * theta))
    q = float(np.sum(wf * (theta - x @ beta_f) ** 2))
    p_par = 2
    trace = float(np.sum(wf) - np.trace(np.linalg.solve(xtwx, x.T @ ((wf ** 2)[:, None] * x))))
    tau2 = max(0.0, (q - (k - p_par)) / trace) if trace > 0 else 0.0
    ws = 1.0 / (v + tau2)
    xtwx_s = x.T @ (ws[:, None] * x)
    beta = np.linalg.solve(xtwx_s, x.T @ (ws * theta))
    cov = np.linalg.inv(xtwx_s)
    df = k - p_par
    q_hk = float(np.sum(ws * (theta - x @ beta) ** 2)) / df
    scale = max(q_hk, 1.0)                 # never narrower than DerSimonian-Laird's own variance
    se = math.sqrt(scale * cov[1, 1])
    t_value = float(beta[1] / se) if se > 0 else float("nan")
    half = float(tdist.ppf(0.5 + confidence / 2.0, df)) * se
    p = float(min(1.0, 2.0 * tdist.sf(abs(t_value), df))) if math.isfinite(t_value) else float("nan")
    return NestedTest(method="meta", conditions=tuple(conds), mean_a=float(beta[0]), mean_b=float(beta[0] + beta[1]),
                      difference=float(beta[1]), ci_low=float(beta[1] - half), ci_high=float(beta[1] + half),
                      p=p, n_days=n_days, n_movies=n_movies, n_cells=n_cells, t=t_value, df=float(df),
                      confidence=confidence,
                      details={"tau2": float(tau2), "tau": math.sqrt(tau2),
                               "replicate_values": tuple(float(t) for t in theta),
                               "replicate_variances": tuple(float(s) for s in v),
                               "replicate_conditions": tuple(int(c) for c in rep_cond)})


# --------------------------------------------------------------------------- the hierarchical bootstrap
def distinct_draws(n: int) -> int:
    """How many different multisets a resample of ``n`` units with replacement can give: C(2n - 1, n)."""
    return math.comb(2 * n - 1, n)


def hierarchical_bootstrap(cells, *, n_boot: int = 2000, seed: int = 0, confidence: float = 0.95) -> NestedTest:
    """Condition B minus A by resampling replicates, then movies within them, then cells within those (module
    docstring). The statistic is the step-weighted fraction of everything drawn, so a replicate with more steps
    weighs more, as the pooled fit's does."""
    conds, y, w, ci, ri, mi, rep_cond, movie_rep, _labels = _rows(cells)
    n_days, n_movies, n_cells = _counts(ci, rep_cond, movie_rep)
    _need(n_days, MIN_REPLICATES_RICH, "hboot")
    # per condition: [replicate] -> [movie] -> (values, weights)
    tree = []
    for k in (0, 1):
        reps = []
        for r in np.flatnonzero(rep_cond == k):
            movies = []
            for m in np.flatnonzero(movie_rep == r):
                sel = mi == m
                movies.append((y[sel], w[sel]))
            reps.append(movies)
        tree.append(reps)
    rng = np.random.default_rng(seed)
    stats = np.empty((int(n_boot), 2))
    for b in range(int(n_boot)):
        for k in (0, 1):
            sw = swy = 0.0
            reps = tree[k]
            for r in rng.integers(0, len(reps), len(reps)):
                movies = reps[r]
                for m in rng.integers(0, len(movies), len(movies)):
                    yy, ww = movies[m]
                    pick = rng.integers(0, len(yy), len(yy))
                    sw += float(np.sum(ww[pick]))
                    swy += float(np.sum(ww[pick] * yy[pick]))
            stats[b, k] = swy / sw
    d = stats[:, 1] - stats[:, 0]
    tail = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(d, [tail, 1.0 - tail])
    p = min(1.0, (2 * min(int(np.sum(d <= 0)), int(np.sum(d >= 0))) + 1) / (int(n_boot) + 1))
    mean_a = _weighted(y[ci == 0], w[ci == 0])
    mean_b = _weighted(y[ci == 1], w[ci == 1])
    distinct = tuple(distinct_draws(n) for n in n_days)
    caveat = ""
    if min(distinct) < 100:
        caveat = (f"coarse: {min(n_days)} replicates give only {min(distinct)} distinct replicate draws, so the "
                  f"interval and p move in steps")
    return NestedTest(method="hboot", conditions=tuple(conds), mean_a=mean_a, mean_b=mean_b,
                      difference=mean_b - mean_a, ci_low=float(lo), ci_high=float(hi), p=float(p), n_days=n_days,
                      n_movies=n_movies, n_cells=n_cells, confidence=confidence, caveat=caveat,
                      details={"n_boot": int(n_boot), "distinct": distinct})
