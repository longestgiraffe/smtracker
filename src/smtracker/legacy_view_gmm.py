"""The legacy window's GMM tab numbers: SMTracker 2.0's Gaussian mixture model of the jumps (`SMTracker.m:623-753`
and the callbacks at `:3676-4725`; `@SMT/fit_ECDF_simultaneously.m`, `@SMT/getDispl_gui.m`, `Help_fcn/gaussianCDF.m`,
`Help_fcn/sortp.m`; manual Fig. 10 and 11, pp. 16-17).

2.0's method, kept:
- the displacements are the one-frame steps in um of every track in a cell whose length lies in "Cell Size in um
  (Min,Max)", on the camera's x and y or, "projected trajectories", along and across the cell; "xy" stacks x then y.
  2.0's outlier rule is kept as it was (`getDispl_gui.m:27-29`: a step is dropped only when BOTH its x and its y
  exceed five standard deviations, one-sided; it is nearly inert);
- a single, double or triple zero-mean Gaussian mixture is fitted to the ECDF of those steps by least squares, ALL
  conditions TOGETHER: one set of sigmas shared by the conditions, each condition its own fractions (`chisqr2`,
  `chisqr3`); "fix pair" shares them across the reference and the comparison only. D = sigma^2 / (2 tau);
- the "+-" is 2.0's: the fit is repeated on ten folds of the ECDF points (the points `i:10:end` held out in turn)
  and the interval is 1.96 times the standard deviation of the ten fits (not a confidence interval of the fit),
  D's `sigma * sd / tau`; the ten held-out mean squared errors choose "Best Model k Cross-Val" (the model that wins
  most folds, ties by the mean);
- BIC = n ln(RSS / n) + (2k - 1) ln n on each condition's ECDF (n its points), the lowest chosen, marked (*) when
  the next simpler model is within 5 per cent (`BIC`); the Kolmogorov-Smirnov test against 1.36 / sqrt(N) with
  p = 1 - kolmcdf(D sqrt(N)), R squared on the ECDF; the significance rows are 2.0's (`legacy_view.significance_rows`
  on the pooled steps).

KNOWN DEVIATIONS from 2.0 (each on purpose; the tab's Console says them):
- the SINGLE fit is made per condition (one population has nothing to share). 2.0 fitted condition i together with
  conditions 1 to i-1 (it handed the fit the conditions read so far), so its single D for the second condition
  was a pooled one.
- 2.0 reported the LAST fold's parameters as the fit. Here the fit is the multistart optimum on the whole ECDF and
  the ten folds give the interval, each started at 2.0's own start and stopped at the solver's default tolerance, as
  2.0 ran them. A fold holds out a tenth of evenly spread points, so the folds differ little and the "+-" comes out tiny
  (a thousandth of D or less): 2.0's interval is not a confidence interval of the fit, and is shown as 2.0 showed it.
- fractions are STICK-BREAKING (as `smtracker.jd`): 2.0's third fraction was `1 - alpha - beta` and could come out
  negative with no error. The derived fraction's "+-" is its own fold spread (2.0 used the larger or the mean of
  the others).
- the single fit's resnorm for the BIC is the condition's own residual sum of squares (2.0 used the last fold's pooled
  one, which cannot be compared with the double and triple fits').
- the fits use at most 2000 evenly spread points of each ECDF (the ECDF's quantiles, so the same curve), as SQD's
  Fast ECDF; the table's R squared, KS and BIC use every point.

INTERFACES
- `gmm_steps(project, condition, projected, cell_size)` -> ``{"x", "y", "xy"}`` arrays of steps (um).
- `gmm_run(project, ...)` -> the run payload; `gmm_table(run, blocks)`, `gmm_pdf(run, condition, n_pop, x)`,
  `bubble_rows(run)`, `qq_points(x, y)`, `pp_points(run, condition, n_pop)`, `histogram(values, bin_width, xlim)`.
"""
from __future__ import annotations

import math

import numpy as np

from .legacy_view import LegacyViewError, significance_rows

__all__ = ["DIRECTIONS", "FIT_NAMES", "AXES_DEFAULTS", "DEFAULT_CELL_SIZE", "N_FOLDS", "gmm_steps", "ecdf", "gmm_run",
           "gmm_table", "gmm_pdf", "bubble_rows", "qq_points", "pp_points", "histogram", "population_names"]

DIRECTIONS = ("xy", "x", "y")
FIT_NAMES = {1: "Single fit", 2: "Double fit", 3: "Triple fit"}
#: `SMTracker.m:667-706`: min X -1, max X 1, max Y 10, BinWidth 0.025, grid 5; `SMT.m:112`: cell size [0, Inf].
AXES_DEFAULTS = {"min X": -1.0, "max X": 1.0, "min Y": 0.0, "max Y": 10.0, "BinWidth": 0.025, "grid": 5}
DEFAULT_CELL_SIZE = (0.0, math.inf)
#: `fit_ECDF_simultaneously.m:151`: ten folds; the points `i_k:k:end` of each condition's ECDF are held out.
N_FOLDS = 10
#: the lower bound of every sigma (`initializeVars.m:99`): 30 nm, the localisation error.
SIGMA_LOWER_UM = 0.03
#: ECDF points a fit uses (evenly spread quantiles); the statistics use every point.
FIT_POINTS = 2000
#: `GMM_table`'s rows (2.0's words).
POP_WORDS = {2: ("Static", "Mobile"), 3: ("Static", "Slow-mobile", "Mobile")}
#: `fit_ECDF_simultaneously.m`: the interval is 1.96 standard deviations of the fold fits.
INTERVAL_SD = 1.96


def population_names(n_pop: int) -> tuple:
    return POP_WORDS.get(n_pop, ("",))


# --------------------------------------------------------------------------------------------- #
# The steps (`getDispl_gui.m`)
# --------------------------------------------------------------------------------------------- #

def gmm_steps(project, condition: str, *, projected: bool = False, cell_size=DEFAULT_CELL_SIZE) -> dict:
    """``{"x": dx, "y": dy, "xy": stacked}`` in um for the condition's one-frame steps, 2.0's way: tracks of cells
    whose length (um) lies in ``cell_size`` (both limits inclusive), consecutive localisations only (a gap breaks
    the step), the camera's x, y or, ``projected``, the position along and across the cell; the outlier rule of
    `getDispl_gui.m:27-29` (see the module head). RAISES when no step is left."""
    lo, hi = float(cell_size[0]), float(cell_size[1])
    if lo < 0 or hi <= lo:
        raise LegacyViewError(f"Cell Size {lo:g} to {hi:g} um: the minimum must be below the maximum")
    default_window = lo <= 0 and not math.isfinite(hi)
    try:
        table = project.projected(condition) if (projected or not default_window) else project.condition(condition)
    except Exception as exc:                               # noqa: BLE001 - said by name, never swallowed
        raise LegacyViewError(f"{condition}: no cell coordinates ({exc})") from exc
    if not default_window:
        geometry = table.attrs.get("geometry") or {}
        length = np.array([geometry[k].length_um if k in geometry else np.nan
                           for k in table["cell"].astype(str).to_numpy()])
        table = table[(length >= lo) & (length <= hi)]
    if table.empty:
        raise LegacyViewError(f"{condition}: no track lies in a cell of {lo:g} to {hi:g} um")
    x_col, y_col = ("L_um", "D_um") if projected else ("x_um", "y_um")
    ordered = table.sort_values(["uid", "frame"], kind="stable")
    uid = ordered["uid"].to_numpy()
    frame = ordered["frame"].to_numpy()
    same = (uid[1:] == uid[:-1]) & (np.diff(frame) == 1)
    dx = np.diff(ordered[x_col].to_numpy(dtype=float))[same]
    dy = np.diff(ordered[y_col].to_numpy(dtype=float))[same]
    if len(dx) < 20:
        raise LegacyViewError(f"{condition}: only {len(dx)} step(s) for the mixture model")
    drop = (dx > 5 * np.std(dx, ddof=1)) & (dy > 5 * np.std(dy, ddof=1))        # `l = ~and(l1, l2)`
    dx, dy = dx[~drop], dy[~drop]
    return {"x": dx, "y": dy, "xy": np.concatenate([dx, dy])}


def ecdf(values) -> tuple:
    """MATLAB's ``[f, x] = ecdf(v)``: the unique sorted values with the empirical CDF, the first point repeated at
    zero (so ``n_unique + 1`` points)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    unique, counts = np.unique(v, return_counts=True)
    x = np.concatenate([unique[:1], unique])
    y = np.concatenate([[0.0], np.cumsum(counts) / v.size])
    return x, y


# --------------------------------------------------------------------------------------------- #
# The fit (`fit_ECDF_simultaneously.m`, `gaussianCDF.m`, `sortp.m`)
# --------------------------------------------------------------------------------------------- #

def _fractions(sticks) -> np.ndarray:
    """Stick-breaking: ``n - 1`` sticks to ``n`` fractions (>= 0, summing to one)."""
    out, rest = [], 1.0
    for u in sticks:
        out.append(u * rest)
        rest *= 1.0 - u
    out.append(rest)
    return np.array(out)


def _unpack(theta, n_pop: int, n_cond: int) -> tuple:
    """``(sigma sorted ascending, fractions (n_cond, n_pop) in that order)`` from the parameter vector
    ``[sigma_1..n, sticks of condition 1, sticks of condition 2, ...]``."""
    theta = np.asarray(theta, dtype=float)
    sigma = theta[:n_pop]
    fractions = np.array([_fractions(theta[n_pop + c * (n_pop - 1): n_pop + (c + 1) * (n_pop - 1)])
                          for c in range(n_cond)])
    order = np.argsort(sigma)
    return sigma[order], fractions[:, order]


def _mix_cdf(sigma, fractions, x) -> np.ndarray:
    from .jd import _mixture_cdf

    return _mixture_cdf(np.asarray(sigma, float), np.asarray(fractions, float), np.asarray(x, float))


def _bounds(n_pop: int, n_cond: int, tau: float) -> tuple:
    """`initializeVars.m:98-116`: sigma from 30 nm; the ceilings sqrt(2 tau 10) (and sqrt(2 tau 3) for a double fit's
    first sigma), 10 and 3 um^2/s; the sticks (2.0's fractions) from 1e-3 to 1."""
    top = math.sqrt(2.0 * tau * 10.0)
    upper_sigma = [top] * n_pop
    if n_pop == 2:
        upper_sigma[0] = math.sqrt(2.0 * tau * 3.0)
    lower = [SIGMA_LOWER_UM] * n_pop + [1e-3] * (n_cond * (n_pop - 1))
    upper = upper_sigma + [1.0] * (n_cond * (n_pop - 1))
    return np.array(lower), np.array(upper)


def _starts(n_pop: int, n_cond: int, tau: float, n_starts: int, seed: int) -> list:
    """2.0's start (`initializeVars.m:98-106`: sigmas 0.5 / 0.04, 0.4 / 0.01, 0.04, 0.4; fractions one half) first,
    then random ones: a single hand-placed start cannot show it found the best fit."""
    lower, upper = _bounds(n_pop, n_cond, tau)
    p0_sigma = {1: [0.5], 2: [0.04, 0.4], 3: [0.01, 0.04, 0.4]}[n_pop]
    first = np.array(p0_sigma + [0.5] * (n_cond * (n_pop - 1)))
    out = [np.clip(first, lower, upper)]
    rng = np.random.default_rng(seed)
    for _ in range(max(0, n_starts - 1)):
        sigma = np.sort(np.exp(rng.uniform(np.log(lower[:n_pop]), np.log(upper[:n_pop]))))
        sticks = rng.uniform(0.1, 0.9, n_cond * (n_pop - 1))
        out.append(np.clip(np.concatenate([sigma, sticks]), lower, upper))
    return out


def _thin(x, y, limit: int) -> tuple:
    if len(x) <= limit:
        return x, y
    pick = np.unique(np.linspace(0, len(x) - 1, limit).astype(int))
    return x[pick], y[pick]


def _fit_group(ecdfs: list, tau: float, n_pop: int, *, n_starts: int, seed: int) -> dict:
    """One simultaneous fit of ``ecdfs`` (a list of ``(x, y)``): the optimum on the whole ECDF (multistart), then the
    ten folds (started at the optimum) for the interval and the held-out errors."""
    from scipy.optimize import least_squares

    n_cond = len(ecdfs)
    lower, upper = _bounds(n_pop, n_cond, tau)

    def residuals(theta, sets):
        sigma, fractions = _unpack(theta, n_pop, n_cond)
        return np.concatenate([y - _mix_cdf(sigma, fractions[c], x) for c, (x, y) in enumerate(sets)])

    thinned = [_thin(x, y, FIT_POINTS) for x, y in ecdfs]
    best = None
    for start in _starts(n_pop, n_cond, tau, n_starts, seed):
        try:
            result = least_squares(residuals, start, bounds=(lower, upper), args=(thinned,), xtol=1e-10, ftol=1e-10,
                                   max_nfev=2000)
        except (ValueError, FloatingPointError):
            continue
        rss = float(result.fun @ result.fun)
        if best is None or rss < best[0]:
            best = (rss, result.x)
    if best is None:
        raise LegacyViewError(f"{n_pop}-population fit: every start failed")
    theta = best[1]
    sigma, fractions = _unpack(theta, n_pop, n_cond)

    fold_start = _starts(n_pop, n_cond, tau, 1, seed)[0]
    fold_sigma, fold_frac = [], []
    test_sse = np.zeros(N_FOLDS)
    test_n = np.zeros(N_FOLDS)
    for fold in range(N_FOLDS):
        train, test = [], []
        for x, y in ecdfs:
            held = np.zeros(len(x), dtype=bool)
            held[fold::N_FOLDS] = True
            train.append(_thin(x[~held], y[~held], FIT_POINTS))
            test.append((x[held], y[held]))
        try:
            # as 2.0 ran each fold: from ITS start, at the solver's default tolerance (1e-6)
            result = least_squares(residuals, fold_start, bounds=(lower, upper), args=(train,), xtol=1e-6, ftol=1e-6,
                                   max_nfev=2000)
        except (ValueError, FloatingPointError) as exc:
            raise LegacyViewError(f"{n_pop}-population fit, fold {fold + 1}: {exc}") from exc
        s_f, f_f = _unpack(result.x, n_pop, n_cond)
        fold_sigma.append(s_f)
        fold_frac.append(f_f)
        res = residuals(result.x, test)
        test_sse[fold] = float(res @ res)
        test_n[fold] = len(res)
    fold_sigma, fold_frac = np.array(fold_sigma), np.array(fold_frac)
    return {"sigma": sigma, "fractions": fractions,
            "sigma_sd": INTERVAL_SD * fold_sigma.std(axis=0, ddof=1),
            "fraction_sd": INTERVAL_SD * fold_frac.std(axis=0, ddof=1),     # (n_cond, n_pop)
            "test_sse": test_sse, "test_n": test_n}


def _gof(x, y, cdf) -> dict:
    """`goF`: the Kolmogorov-Smirnov test as 2.0 coded it (``N`` the ECDF's length, ``i / N`` against the model)."""
    from scipy import stats

    n = len(x)
    f = cdf(x)
    i = np.arange(1, n + 1)
    d = max(float(np.max(i / n - f)), float(np.max(f - (i - 1) / n)))
    return {"D": d, "N": int(n), "rejected": bool(d > 1.36 / math.sqrt(n)),
            "p_value": float(stats.kstwobign.sf(d * math.sqrt(n)))}


def _r2(y, y_pred) -> float:
    sst = float(((y - y.mean()) ** 2).sum())
    return float(1.0 - ((y - y_pred) ** 2).sum() / sst) if sst > 0 else float("nan")


def _fit_result(conditions, ecdfs, tau, n_pop, group, *, n_starts, seed) -> dict:
    """The per-condition numbers of one fitted model over ``conditions`` (the fitted group)."""
    fit = _fit_group([ecdfs[c] for c in conditions], tau, n_pop, n_starts=n_starts, seed=seed)
    D = fit["sigma"] ** 2 / (2.0 * tau)
    per = {}
    for i, c in enumerate(conditions):
        x, y = ecdfs[c]
        fractions = fit["fractions"][i]
        cdf = (lambda xx, f=fractions, s=fit["sigma"]: _mix_cdf(s, f, xx))
        pred = cdf(x)
        per[c] = {"fractions": fractions, "fraction_sd": fit["fraction_sd"][i],
                  "resnorm": float(((pred - y) ** 2).sum()), "r2": _r2(y, pred), "n_ecdf": int(len(y)),
                  "ks": _gof(x, y, cdf)}
    return {"n_pop": n_pop, "conditions": list(conditions), "sigma": fit["sigma"], "sigma_sd": fit["sigma_sd"],
            "D": D, "D_sd": fit["sigma_sd"] * fit["sigma"] / tau, "per": per,
            "test_sse": fit["test_sse"], "test_n": fit["test_n"], "pair_fixed": bool(group)}


def _single_fits(conditions, ecdfs, tau, *, n_starts, seed) -> dict:
    """The single fit: one sigma per condition, each fitted alone (see the module head)."""
    parts = [_fit_result([c], ecdfs, tau, 1, False, n_starts=n_starts, seed=seed) for c in conditions]
    per, sigma, sigma_sd, D, D_sd = {}, [], [], [], []
    for c, part in zip(conditions, parts):
        per[c] = dict(part["per"][c], sigma=float(part["sigma"][0]), sigma_sd=float(part["sigma_sd"][0]),
                      D=float(part["D"][0]), D_sd=float(part["D_sd"][0]))
    return {"n_pop": 1, "conditions": list(conditions), "per": per, "pair_fixed": False,
            "test_sse": np.sum([p["test_sse"] for p in parts], axis=0),
            "test_n": np.sum([p["test_n"] for p in parts], axis=0)}


def _bic_rows(fits: dict, conditions) -> tuple:
    """2.0's `BIC` and `best_CV` where all three models were fitted: ``({condition: (bic list, text)}, cv text)``."""
    if not all(k in fits for k in (1, 2, 3)):
        return {}, ""
    out = {}
    for c in conditions:
        if not all(c in fits[k]["per"] for k in (1, 2, 3)):
            continue
        bic = []
        for k in (1, 2, 3):
            p = fits[k]["per"][c]
            n = p["n_ecdf"]
            bic.append(n * math.log(max(p["resnorm"], 1e-300) / n) + (2 * k - 1) * math.log(n))
        best = int(np.argmin(bic)) + 1
        star = ""
        if best != 1 and bic[best - 1] / bic[best - 2] - 1 < 0.05:
            star = " (*)"
        out[c] = (bic, f"{FIT_NAMES[best]}{star}", best)
    cv = ""
    sse = np.array([fits[k]["test_sse"] / np.maximum(fits[k]["test_n"], 1) for k in (1, 2, 3)]).T      # (folds, 3)
    wins = np.bincount(sse.argmin(axis=1), minlength=3)
    top = np.flatnonzero(wins == wins.max())
    chosen = int(top[0]) + 1 if len(top) == 1 else int(top[np.argmin(sse[:, top].mean(axis=0))]) + 1
    cv = f"{FIT_NAMES[chosen]} "
    return out, cv


def gmm_run(project, *, reference: str | None = None, comparison: str | None = None, projected: bool = False,
            fix_pair: bool = False, fits=(2,), cell_size=DEFAULT_CELL_SIZE, direction: str = "xy",
            n_starts: int = 6, seed: int = 0, steps: dict | None = None) -> dict:
    """2.0's GMM tab's ``Run & Plot``. ``steps``: ``gmm_steps`` already made for these settings, if the caller has. ``fits``: the model sizes ticked (1, 2, 3); ``direction`` "xy" (the camera
    plane, or the cell's two axes stacked) or "x" / "y" (the projected cell's long / short axis, or the camera's
    when not projected); ``fix_pair`` fits the double and triple models on the reference and comparison alone.

    Returns ``conditions``, ``reference``, ``comparison``, ``direction``, ``projected``, ``tau_s``, ``steps``
    (``{condition: {"x", "y", "xy"}}``), ``samples`` (the chosen direction), ``fits`` (``{n_pop: fit}``; a fit has
    ``sigma``, ``D``, ``D_sd`` and ``per``: per condition ``fractions``, ``fraction_sd``, ``resnorm``, ``r2``,
    ``ks``), ``bic`` (``{condition: (bic, text, best)}``), ``cv`` (the cross-validation choice) and
    ``significance`` (2.0's rows)."""
    if direction not in DIRECTIONS:
        raise LegacyViewError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    fits = tuple(sorted(set(int(k) for k in fits)))
    if not fits or not set(fits) <= {1, 2, 3}:
        raise LegacyViewError("tick at least one of the single, double and triple fits")
    conditions = list(project.conditions)
    reference = reference or conditions[0]
    comparison = comparison or (conditions[1] if len(conditions) > 1 else conditions[0])
    if fix_pair and len(conditions) < 3:
        fix_pair = False                                  # 2.0 enables it only for more than two conditions
    tau = float(project.params.tau_s)
    steps = steps or {c: gmm_steps(project, c, projected=projected, cell_size=cell_size) for c in conditions}
    samples = {c: steps[c][direction] for c in conditions}
    ecdfs = {c: ecdf(samples[c]) for c in conditions}
    group = [reference, comparison] if (fix_pair and reference != comparison) else conditions
    results = {}
    for k in fits:
        if k == 1:
            results[1] = _single_fits(conditions, ecdfs, tau, n_starts=n_starts, seed=seed)
        else:
            results[k] = _fit_result(group, ecdfs, tau, k, fix_pair, n_starts=n_starts, seed=seed)
    bic, cv = _bic_rows(results, conditions)
    return {"conditions": conditions, "reference": reference, "comparison": comparison, "direction": direction,
            "projected": bool(projected), "fix_pair": bool(fix_pair), "tau_s": tau, "steps": steps,
            "samples": samples, "fits": results, "bic": bic, "cv": cv, "cell_size": tuple(cell_size),
            "significance": significance_rows(samples) if len(conditions) > 1 else []}


# --------------------------------------------------------------------------------------------- #
# The table (`GMM_table`)
# --------------------------------------------------------------------------------------------- #

def _num2str(value: float, digits: int = 2) -> str:
    return "NaN" if not math.isfinite(value) else f"{value:.{digits}g}"


def _with_sd(value: float, sd: float) -> str:
    return f"{_num2str(value)} ± {_num2str(sd)}"


UNITS = "µm²/s"


def _block_rows(run: dict | None, conditions: list, header: str) -> list:
    """One direction's block of 2.0's table: header, single, double and triple fits with their statistics and the
    best models; empty cells where the model was not fitted (or the condition not in the fitted pair)."""
    rows = [[header] + [""] * len(conditions)]

    def put(label, fn, k, *, only=None):
        cells = []
        for c in conditions:
            fit = run["fits"].get(k) if run else None
            cells.append(fn(fit, c) if fit and c in fit["per"] else "")
        rows.append([label] + cells)

    stats = lambda fit, c: fit["per"][c]
    rows.append(["Single fit"] + [""] * len(conditions))
    put(f"  D ± sd [{UNITS}]", lambda f, c: _with_sd(f["per"][c]["D"], f["per"][c]["D_sd"]), 1)
    put("  K-S GoF test", lambda f, c: "Rejected" if stats(f, c)["ks"]["rejected"] else "Accepted", 1)
    put("  P-Value", lambda f, c: _num2str(stats(f, c)["ks"]["p_value"]), 1)
    put("  R-Squared", lambda f, c: _num2str(stats(f, c)["r2"], 3), 1)
    for k, title in ((2, "Double fit"), (3, "Triple fit")):
        rows.append([title] + [""] * len(conditions))
        words = POP_WORDS[k]
        for j, w in enumerate(words):
            put(f"  {w} D ± sd [{UNITS}]", (lambda f, c, j=j: _with_sd(f["D"][j], f["D_sd"][j])), k)
        for j, w in enumerate(words):
            put(f"  {w} Fraction ± sd [%]",
                (lambda f, c, j=j: _with_sd(100 * f["per"][c]["fractions"][j], 100 * f["per"][c]["fraction_sd"][j])), k)
        put("  K-S GoF test", lambda f, c: "Rejected" if stats(f, c)["ks"]["rejected"] else "Accepted", k)
        put("  P-Value", lambda f, c: _num2str(stats(f, c)["ks"]["p_value"]), k)
        put("  R-Squared", lambda f, c: _num2str(stats(f, c)["r2"], 3), k)
    rows.append(["Best Model"] + [(run["bic"][c][1] if run and c in run["bic"] else "") for c in conditions])
    rows.append(["Best Model k Cross-Val"] + [(run["cv"] if run and run["bic"] and c in run["bic"] else "")
                                              for c in conditions])
    return rows


def gmm_table(run: dict, blocks: dict | None = None) -> list:
    """2.0's "Results of gaussian mixture modeling": the significance rows (empty for one condition), then three
    blocks "Direction: XY", "Direction:  X", "Direction:  Y". ``blocks``: ``{direction: run}`` for the directions
    run so far in this tab (as 2.0 kept them); the current ``run`` fills its own. Rows of strings, the first column
    their names, one column per condition."""
    conditions = run["conditions"]
    runs = dict(blocks or {})
    runs[run["direction"]] = run
    rows = [list(r) for r in run["significance"]]
    for d, header in (("xy", "Direction: XY"), ("x", "Direction:  X"), ("y", "Direction:  Y")):
        rows.extend(_block_rows(runs.get(d), conditions, header))
    return rows


# --------------------------------------------------------------------------------------------- #
# What the panels draw
# --------------------------------------------------------------------------------------------- #

def histogram(values, bin_width: float, xlim: tuple | None = None) -> dict:
    """MATLAB's ``histogram(v, 'BinWidth', w, 'Normalization', 'pdf')``: edges at multiples of ``w`` covering the
    data (or, given ``xlim``, only what the axis shows), heights ``count / (N w)`` with N the steps counted."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if bin_width <= 0:
        raise LegacyViewError("BinWidth must be positive")
    lo, hi = (v.min(), v.max())
    edges = np.arange(math.floor(lo / bin_width), math.ceil(hi / bin_width) + 1) * bin_width
    counts, edges = np.histogram(v, bins=edges)
    return {"edges": edges, "density": counts / (v.size * bin_width), "n": int(v.size)}


def gmm_pdf(run: dict, condition: str, n_pop: int, x) -> dict:
    """The fitted curves 2.0's `plotGMMFit` drew for ``condition``: ``total`` and ``components`` (each fraction
    times its normal density, the static one first), on ``x``."""
    from scipy import stats

    fit = run["fits"][n_pop]
    x = np.asarray(x, dtype=float)
    if n_pop == 1:
        sigma = [fit["per"][condition]["sigma"]]
        fractions = np.array([1.0])
    else:
        sigma, fractions = fit["sigma"], fit["per"][condition]["fractions"]
    comps = [float(f) * stats.norm.pdf(x, 0.0, s) for f, s in zip(fractions, sigma)]
    return {"total": np.sum(comps, axis=0), "components": comps}


def bubble_rows(run: dict) -> dict:
    """The bubble plot and the crossed error bars' numbers, per ticked model ``{n_pop: [(condition, D list, D sd
    list, fraction % list, fraction sd % list)]}``, the conditions of the fitted group in order (2.0's `bubbleplotGMM_`:
    the single fit is one bubble of 100 per cent, no spread)."""
    out = {}
    for k, fit in run["fits"].items():
        rows = []
        for c in fit["conditions"]:
            p = fit["per"][c]
            if k == 1:
                rows.append((c, [p["D"]], [p["D_sd"]], [100.0], [0.0]))
            else:
                rows.append((c, list(fit["D"]), list(fit["D_sd"]), list(100 * p["fractions"]),
                             list(100 * p["fraction_sd"])))
        out[k] = rows
    return out


def qq_points(x, y, max_points: int = 2000) -> dict:
    """MATLAB's two-sample ``qqplot(x, y)``: the quantiles of ``x`` against those of ``y`` at the plotting positions
    of the smaller sample, and the line through the quartile pairs. At most ``max_points`` are returned (evenly
    spread quantiles; a drawing limit only)."""
    x = np.sort(np.asarray(x, dtype=float)[np.isfinite(x)])
    y = np.sort(np.asarray(y, dtype=float)[np.isfinite(y)])
    n = min(len(x), len(y), int(max_points))
    p = (np.arange(n) + 0.5) / n
    qx, qy = np.quantile(x, p), np.quantile(y, p)
    q = np.array([0.25, 0.75])
    (x1, x3), (y1, y3) = np.quantile(x, q), np.quantile(y, q)
    slope = (y3 - y1) / (x3 - x1) if x3 != x1 else float("nan")
    return {"x": qx, "y": qy, "line": (float(x1), float(y1), float(slope))}


def pp_points(run: dict, condition: str, n_pop: int) -> dict:
    """`ppplot`: the observed ECDF against the fitted CDF at ECDF points reduced to about 100 (`reduce`), the least
    squares line through them and its slope, intercept and R squared."""
    x, y = ecdf(run["samples"][condition])
    fit = run["fits"][n_pop]
    if n_pop == 1:
        pred = _mix_cdf([fit["per"][condition]["sigma"]], [1.0], x)
    else:
        pred = _mix_cdf(fit["sigma"], fit["per"][condition]["fractions"], x)
    design = np.column_stack([np.ones(len(y)), y])
    b = np.linalg.lstsq(design, pred, rcond=None)[0]
    calc = design @ b
    rsq = _r2(pred, calc)
    jump = max(len(y) // 100, 1)
    pick = np.arange(0, len(y), jump)
    return {"observed": y[pick], "predicted": pred[pick], "line": calc[pick], "intercept": float(b[0]),
            "slope": float(b[1]), "r2": rsq}
