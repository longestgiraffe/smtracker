"""The legacy window's APPD tab: SMTracker 2.0's apparent-diffusion tab, computed the old way.

Ruled 26 Nov 2026 (`docs/legacy_window_plan.md`). 2.0's tab (`SMTracker.m:881-955`, callbacks `APPD`, `AppDFit`,
`AppDTable`, `bubbleplotAPPD`, `bestModel`; `Help_fcn/AppDFit_.m`, `AppDFit_simul.m`, `fit_appD_simultaneously.m`,
`calcAppD_gamma.m`, `calcAppD_gau.m`, `AppDPlot.m`, `gammaCDF.m`, `sortParams_*.m`; `@SMT/appDFilter.m`,
`plotSpeedmap.m`; manual Figs 14 and 15). Headless: `smtracker.ui.legacy.tab_appd` only lays out and draws.

WHERE THE VALUES COME FROM, as 2.0's tab took them: the apparent D of each TRACK is its per-track MSD fit from the MSD
tab (`fitMSDTracks.m`, xy, "specific tau" and the method and last-lag box of that tab: `validateinstD` ran the MSD for
every condition with those controls when the APPD tab opened); `appDFilter` keeps the Brownian tracks (type 1) with
``0 <= D <= 10`` and ``R^2 >= `` the box (default 0.5). The Gamma shape is the specific tau minus one
(`steps = str2double(instD{i,1}{2}) - 1`). The four-step-piece estimator of `getAppD_gui.m` (`appd.apparent_D(mode=
"split")`) is the OTHER 2.0 route, the one the reference reproduction (`tests/test_reference_appd.py`) validates; the
tab did not use it, and `step_pieces` offers it as an option beside the tab's own route.

THE FITS (`AppDFit`): 1, 2 and 3 populations, GAMMA or GAUSSIAN.
- One at a time ("Simultaneous" off, and always at one population): maximum likelihood per condition. GAMMA is
  `appd.fit_appd` (the fit validated against 2.0's own `calcAppD_gamma` to 1e-4), the start chosen by likelihood where
  2.0 chose among 21 random starts by the R^2 of the ECDF; GAUSSIAN is a normal mixture by the same optimiser.
  The reported plus-minus is 2.0's: half of a 95 % normal interval from the numerical Hessian (`mle`'s `pci`), the
  last fraction's the mean of the others' (`sortParams_*`).
- Simultaneous (default, from two populations): `lsqnonlin` on the ECDF residuals of every condition at once, one set
  of D (and sigma) for all, a fraction set per condition (`fit_appD_simultaneously`); the interval is `nlparci` on the
  full residual (`legacy.sqd.nlparci_equivalent`, `ix = 1:1:end`). 2.0 started once from `rand`; here the best of five
  seeded starts, so a run is reproducible. NOT the pseudo-replication-safe interval: the legacy design is the point.
- R^2: the ECDF's, `r2.m`. BIC: 2.0's `log(n) k - 2 log(prod(pdf(xData)))` over the ECDF points, which overflows or
  underflows for thousands of points (the module `legacy_bic` records why). Where it is not finite at every count, "Best
  Model" is the first of the ties (MATLAB's `min`), as 2.0's was, and `bic_finite` says so; nothing is repaired.
- The bubble plot of "best" draws THREE populations (`bubbleplotAPPD`'s `otherwise`), 2.0's behaviour, kept.

THE SPEED MAPS (`plotSpeedmap.m`): the tracks whose per-track D lies between two limits (and whose R^2 passes), drawn on
the model cell (length 1, width 1; tracks normalised by their cell's length and width), as lines and as a heat map
(a 20 x 20 histogram cubic-interpolated 32-fold, normalised by the number of points, jet), points outside the model
contour dropped (`inCellContour`). `legacy.speedmap.contour_trim` counts what the trim removed.

INTERFACES: `appd_values`, `appd_run`, `appd_table`, `bubble_data`, `curves_for`, `refit_condition`, `speedmap`,
`model_cell_contour`, `step_pieces`.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.optimize import least_squares, minimize

from .legacy_view import LegacyViewError, significance_rows

__all__ = ["APPD_TYPES", "APPD_POPS", "DEFAULT_R2_LIMIT", "DEFAULT_BIN_WIDTH", "D_WINDOW", "BEST_LABELS",
           "appd_values", "appd_run", "appd_table", "bubble_data", "curves_for", "refit_condition",
           "step_pieces", "speedmap", "model_cell_contour", "v2_bic", "ecdf_r2"]

APPD_TYPES = ("GAMMA", "GAUSSIAN")
APPD_POPS = ("1", "2", "3", "best")
#: `initializeVars.m:189` and `SMTracker.m:924`.
DEFAULT_R2_LIMIT = 0.5
DEFAULT_BIN_WIDTH = 0.05
#: `appDFilter.m:18-19`: the D window a track must lie in (um^2/s).
D_WINDOW = (0.0, 10.0)
#: `bestModel`: the Best Model row's words (a trailing blank, as 2.0's strings had).
BEST_LABELS = {1: "Single Fit ", 2: "Double Fit ", 3: "Triple Fit "}
#: Starts for the simultaneous least-squares fit (2.0: one, from `rand`).
SIMULTANEOUS_STARTS = 5
_PARAMS_PER_POP = {"GAMMA": {1: 1, 2: 3, 3: 5}, "GAUSSIAN": {1: 2, 2: 5, 3: 8}}


# --------------------------------------------------------------------------------------------- #
# The values
# --------------------------------------------------------------------------------------------- #

def appd_values(tracks: dict, r2_lim: float) -> np.ndarray:
    """`appDFilter.m` on one condition's per-track fits (`legacy_view_msd.msd_tracks`): Brownian tracks (type 1)
    with ``0 <= D <= 10`` and ``R^2 >= r2_lim``. RAISES on an R^2 outside 0 to 1 ('Not a proper R^2 value')."""
    if not 0.0 <= r2_lim <= 1.0:
        raise LegacyViewError(f"Not a proper R² value: {r2_lim} (between 0 and 1)")
    D = tracks["D"]
    keep = (D >= D_WINDOW[0]) & (D <= D_WINDOW[1]) & (tracks["r2"] >= r2_lim) & (tracks["type"] == 1)
    return D[keep]


def step_pieces(project, condition: str, *, n_steps: int = 4) -> np.ndarray:
    """The OTHER 2.0 route (`getAppD_gui.m`): the mean squared jump over each four-step piece of every track, over 4
    tau. `appd.apparent_D(mode="split")`, the estimator `tests/test_reference_appd.py` validates against MATLAB."""
    from .appd import apparent_D

    table = project.condition(condition)
    return apparent_D(table, n_steps=n_steps, mode="split")["appd_um2_per_s"].to_numpy(dtype=float)


# --------------------------------------------------------------------------------------------- #
# The models
# --------------------------------------------------------------------------------------------- #

def _component_pdf(kind: str, x, centre, spread, steps: int):
    x = np.asarray(x, dtype=float)
    if kind == "GAMMA":                                    # gammafit_one.m: Gamma(steps, D / steps)
        return stats.gamma.pdf(x, a=steps, scale=centre / steps)
    return stats.norm.pdf(x, centre, spread)


def _component_cdf(kind: str, x, centre, spread, steps: int):
    x = np.asarray(x, dtype=float)
    if kind == "GAMMA":                                    # gammaCDF.m
        return stats.gamma.cdf(x, a=steps, scale=centre / steps)
    return stats.norm.cdf(x, centre, spread)


def _mixture(fun, kind, x, fractions, centres, spreads, steps):
    total = np.zeros_like(np.asarray(x, dtype=float))
    for f, c, s in zip(fractions, centres, spreads):
        total = total + f * fun(kind, x, c, s, steps)
    return total


def ecdf_r2(y_data, y_pred) -> float:
    """`r2.m`: 1 - SSE / SST."""
    sst = float(((y_data - y_data.mean()) ** 2).sum())
    return float(1.0 - ((y_data - y_pred) ** 2).sum() / sst) if sst > 0 else float("nan")


def v2_bic(pdf_at_ecdf_x, n_ecdf: int, k: int) -> float:
    """2.0's `log(n) * k - 2 * log(prod(pdf(xData)))`, in double arithmetic as MATLAB evaluated it: a product that
    overflows is Inf (BIC -Inf), one that underflows is 0 (BIC +Inf). Nothing is repaired."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        product = float(np.prod(np.asarray(pdf_at_ecdf_x, dtype=float)))
        return float(np.log(n_ecdf) * k - 2.0 * np.log(product))


def _sorted_fit(kind, n_pop, fractions, centres, spreads, sd_fraction, sd_centre, sd_spread):
    """`sortParams_*`: ascending by the centre (D or mu), the last fraction's sd the mean of the others'."""
    order = np.argsort(centres)
    if n_pop > 1:
        sd_frac = np.append(sd_fraction[: n_pop - 1], np.mean(sd_fraction[: n_pop - 1]))
    else:
        sd_frac = np.zeros(1)
    return {"fraction": np.asarray(fractions)[order], "fraction_sd": sd_frac[order],
            "D": np.asarray(centres)[order], "D_sd": np.asarray(sd_centre)[order],
            "sigma": (np.asarray(spreads)[order] if kind == "GAUSSIAN" else None),
            "sigma_sd": (np.asarray(sd_spread)[order] if kind == "GAUSSIAN" else None)}


def _hessian_half_width(nll, theta, lower, upper) -> np.ndarray:
    """Half of a 95 % normal interval from the numerical Hessian of the negative log likelihood (`mle`'s `pci`):
    ``1.96 sqrt(diag(inv(H)))``; NaN where the Hessian is not positive definite."""
    theta = np.asarray(theta, dtype=float)
    k = len(theta)
    h = 1e-4 * np.maximum(np.abs(theta), 1e-2)
    H = np.zeros((k, k))
    f0 = nll(theta)
    for i in range(k):
        for j in range(i, k):
            ei, ej = np.zeros(k), np.zeros(k)
            ei[i], ej[j] = h[i], h[j]
            if i == j:
                H[i, i] = (nll(theta + ei) - 2 * f0 + nll(theta - ei)) / h[i] ** 2
            else:
                H[i, j] = H[j, i] = (nll(theta + ei + ej) - nll(theta + ei - ej)
                                     - nll(theta - ei + ej) + nll(theta - ei - ej)) / (4 * h[i] * h[j])
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return np.full(k, np.nan)
    diag = np.diag(cov)
    out = np.where(diag > 0, 1.96 * np.sqrt(np.abs(diag)), np.nan)
    return out


def _gaussian_mle(x, n_pop, seed):
    """A normal mixture by maximum likelihood (`calcAppD_gau.m`'s model): parameters ``[fractions(n-1), mu(n),
    sigma(n)]``, mu >= 0 and sigma >= 1e-3 (the likelihood is unbounded as sigma falls onto a value), the start 2.0's
    (quantiles, a common sigma) plus seeded random ones, the best likelihood kept."""
    rng = np.random.default_rng(seed)
    q = np.quantile(x, np.linspace(0.25, 0.75, n_pop) if n_pop > 1 else [0.5])
    sig0 = float(np.sqrt(max(np.var(x) - 0.25 * (np.ptp(q) ** 2 if n_pop > 1 else 0.0), 1e-6)))
    starts = [np.concatenate([np.full(n_pop - 1, 1.0 / n_pop), q, np.full(n_pop, sig0)])]
    scale = float(np.max(x))
    for _ in range(20):
        starts.append(np.concatenate([rng.dirichlet(np.ones(n_pop))[: n_pop - 1], rng.uniform(0, scale, n_pop),
                                      rng.uniform(0.01, 0.5 * scale + 0.05, n_pop)]))
    lower = np.concatenate([np.full(n_pop - 1, 1e-3), np.zeros(n_pop), np.full(n_pop, 1e-3)])
    upper = np.concatenate([np.ones(n_pop - 1), np.full(n_pop, np.inf), np.full(n_pop, np.inf)])

    def split(theta):
        f = theta[: n_pop - 1]
        return np.append(f, 1.0 - f.sum()), theta[n_pop - 1:2 * n_pop - 1], theta[2 * n_pop - 1:]

    def nll(theta):
        f, mu, sig = split(theta)
        if np.any(f <= 0) or np.any(sig <= 0):
            return 1e30
        density = _mixture(lambda k, xx, c, s, st: stats.norm.pdf(xx, c, s), "GAUSSIAN", x, f, mu, sig, 0)
        with np.errstate(divide="ignore"):
            value = -np.sum(np.log(density))
        return value if np.isfinite(value) else 1e30

    best = None
    for start in starts:
        try:
            r = minimize(nll, start, method="L-BFGS-B", bounds=list(zip(lower, upper)),
                         options={"maxiter": 3000, "ftol": 1e-12})
        except (ValueError, FloatingPointError):
            continue
        if np.all(np.isfinite(r.x)) and (best is None or r.fun < best.fun):
            best = r
    if best is None or best.fun >= 1e29:
        raise LegacyViewError(f"no start converged for a {n_pop}-population Gaussian fit")
    f, mu, sig = split(best.x)
    half = _hessian_half_width(nll, best.x, lower, upper)
    return f, mu, sig, half[: n_pop - 1], half[n_pop - 1:2 * n_pop - 1], half[2 * n_pop - 1:], float(-best.fun)


def _fit_one(x, steps: int, n_pop: int, kind: str, seed: int) -> dict:
    """`AppDFit_` for one condition: maximum likelihood (see the module head)."""
    x = np.asarray(x, dtype=float)
    if kind == "GAMMA":
        from .appd import AppDError, fit_appd

        try:
            fit = fit_appd(x, n_pop=n_pop, n_steps=steps, n_starts=24, seed=seed)
        except AppDError as exc:
            raise LegacyViewError(f"{n_pop} population(s): {exc}") from exc
        f, D = np.asarray(fit.fractions), np.asarray(fit.D)
        order = np.argsort(D)
        f, D = f[order], D[order]
        if n_pop == 1:
            half = _hessian_half_width(lambda t: -np.sum(stats.gamma.logpdf(x, a=steps, scale=t[0] / steps)), D, [0],
                                       [np.inf])
            sd_f, sd_D = np.zeros(0), half
        else:
            def nll(theta):
                fr = np.append(theta[: n_pop - 1], 1.0 - np.sum(theta[: n_pop - 1]))
                if np.any(fr <= 0) or np.any(theta[n_pop - 1:] <= 0):
                    return 1e30
                dens = sum(fi * stats.gamma.pdf(x, a=steps, scale=di / steps) for fi, di in zip(fr, theta[n_pop - 1:]))
                with np.errstate(divide="ignore"):
                    value = -np.sum(np.log(dens))
                return value if np.isfinite(value) else 1e30

            theta = np.concatenate([f[: n_pop - 1], D])
            half = _hessian_half_width(nll, theta, None, None)
            sd_f, sd_D = half[: n_pop - 1], half[n_pop - 1:]
        out = _sorted_fit(kind, n_pop, f, D, None, sd_f, sd_D, None)
        sigma = None
    else:
        f, mu, sig, sd_f, sd_mu, sd_sig, _ = _gaussian_mle(x, n_pop, seed)
        out = _sorted_fit(kind, n_pop, f, mu, sig, sd_f, sd_mu, sd_sig)
    return _finish(out, x, steps, n_pop, kind)


def _finish(out: dict, x, steps: int, n_pop: int, kind: str) -> dict:
    """R^2 of the ECDF and 2.0's BIC for a fit, and the n_pop / kind it is."""
    from .legacy.sqd import matlab_ecdf

    x_data, y_data = matlab_ecdf(x)
    spreads = out["sigma"] if out["sigma"] is not None else [None] * n_pop
    y_pred = _mixture(_component_cdf, kind, x_data, out["fraction"], out["D"], spreads, steps)
    pdf = _mixture(_component_pdf, kind, x_data, out["fraction"], out["D"], spreads, steps)
    out.update({"n_pop": n_pop, "kind": kind, "steps": steps, "r2": ecdf_r2(y_data, y_pred),
                "bic": v2_bic(pdf, len(x_data), _PARAMS_PER_POP[kind][n_pop]), "n": int(len(x))})
    return out


def _fit_simultaneous(xs: dict, steps: int, n_pop: int, kind: str, seed: int) -> dict:
    """`AppDFit_simul` / `fit_appD_simultaneously`: ECDF least squares over every condition at once (module head)."""
    from .legacy.sqd import matlab_ecdf, nlparci_equivalent

    names = list(xs)
    n_cond = len(names)
    ecdfs = {c: matlab_ecdf(xs[c]) for c in names}
    n_frac = (n_pop - 1) * n_cond
    n_centres = n_pop
    n_spread = n_pop if kind == "GAUSSIAN" else 0
    size = n_frac + n_centres + n_spread
    lower = np.full(size, 1e-3)
    upper = np.concatenate([np.ones(n_frac), np.full(n_centres + n_spread, 10.0)])

    def unpack(p, i):
        fr = p[(n_pop - 1) * i:(n_pop - 1) * (i + 1)]
        return (np.append(fr, 1.0 - fr.sum()), p[n_frac:n_frac + n_centres],
                (p[n_frac + n_centres:] if kind == "GAUSSIAN" else [None] * n_pop))

    def residual(p):
        parts = []
        for i, c in enumerate(names):
            f, centres, spreads = unpack(p, i)
            xd, yd = ecdfs[c]
            parts.append(yd - _mixture(_component_cdf, kind, xd, f, centres, spreads, steps))
        return np.concatenate(parts)

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(SIMULTANEOUS_STARTS):
        p0 = np.concatenate([rng.random(n_frac), rng.random(n_centres), rng.random(n_spread)])
        p0 = np.clip(p0, lower + 1e-6, upper - 1e-6)
        try:
            r = least_squares(residual, p0, bounds=(lower, upper), method="trf", xtol=1e-10, ftol=1e-10, max_nfev=400)
        except (ValueError, FloatingPointError):
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        raise LegacyViewError(f"the simultaneous {n_pop}-population fit did not run")
    if np.linalg.matrix_rank(best.jac) < best.jac.shape[1]:
        half = np.full(size, np.nan)                       # 'Jacobian is ill-conditioned': sd = NaN
    else:
        lo, hi = nlparci_equivalent(best.x, best.fun, best.jac)
        half = (hi - lo) / 2.0
    fits = {}
    for i, c in enumerate(names):
        f, centres, spreads = unpack(best.x, i)
        sd_f = half[(n_pop - 1) * i:(n_pop - 1) * (i + 1)]
        sd_c = half[n_frac:n_frac + n_centres]
        sd_s = half[n_frac + n_centres:] if kind == "GAUSSIAN" else None
        out = _sorted_fit(kind, n_pop, f, centres, (np.asarray(spreads) if kind == "GAUSSIAN" else None), sd_f, sd_c,
                          sd_s)
        fits[c] = _finish(out, xs[c], steps, n_pop, kind)
        fits[c]["simultaneous"] = True
    return fits


# --------------------------------------------------------------------------------------------- #
# The tab's Run
# --------------------------------------------------------------------------------------------- #

def _best(bics: dict, conditions) -> tuple:
    """`bestModel`: per condition the count of the smallest BIC (MATLAB's `min`: the first of ties, NaN ignored,
    count 1 where all are NaN), with 2.0's "(*)" where the two counts' BICs are within 5 %."""
    best, label = {}, {}
    for c in conditions:
        b = np.array([bics[n][c] for n in (1, 2, 3)], dtype=float)
        finite = ~np.isnan(b)
        kk = int(np.argmin(np.where(finite, b, np.inf))) + 1 if finite.any() else 1
        mark = ""
        if kk != 1:
            with np.errstate(invalid="ignore", divide="ignore"):
                if abs(b[kk - 1] / b[kk - 2] - 1.0) < 0.05:
                    mark = "(*)"
        best[c] = kk
        label[c] = BEST_LABELS[kk] + mark
    return best, label


def appd_run(project, msd_run: dict, *, kind: str = "GAMMA", simultaneous: bool = True,
             r2_lim: float = DEFAULT_R2_LIMIT, seed: int = 0, values_by_condition: dict | None = None) -> dict:
    """2.0's "Run best fit" on the conditions of ``msd_run`` (`legacy_view_msd.msd_run`, whose settings the values come
    from; its per-track fits are cached in it). ``simultaneous`` is forced off for one condition (2.0 disabled the box).
    ``values_by_condition`` replaces the per-track values (the four-step-piece route, `step_pieces`); the Gamma shape is
    then 4. RAISES `LegacyViewError` naming a condition with too few values or a failed fit."""
    from .legacy_view_msd import msd_tracks

    if kind not in APPD_TYPES:
        raise LegacyViewError(f"fit type must be one of {APPD_TYPES}, got {kind!r}")
    conditions = list(msd_run["conditions"])
    steps = int(msd_run["settings"]["specific_tau"]) - 1
    values, dropped = {}, {}
    for c in conditions:
        if values_by_condition is not None:
            raw = np.asarray(values_by_condition[c], dtype=float)
            steps = 4
        else:
            raw = appd_values(msd_tracks(project, msd_run, c), r2_lim)
        positive = raw[np.isfinite(raw) & (raw > 0)]
        dropped[c] = int(len(raw) - len(positive))
        if len(positive) < 10:
            raise LegacyViewError(f"{c}: {len(positive)} apparent-D values pass the filter (R² >= {r2_lim}); "
                                  f"lower the minimum R² or the specific τ")
        values[c] = positive
    simultaneous = bool(simultaneous) and len(conditions) > 1
    fits: dict = {1: {}, 2: {}, 3: {}}
    for n_pop in (1, 2, 3):
        if simultaneous and n_pop > 1:
            fits[n_pop] = _fit_simultaneous(values, steps, n_pop, kind, seed)
        else:
            for c in conditions:
                fits[n_pop][c] = _fit_one(values[c], steps, n_pop, kind, seed)
                fits[n_pop][c]["simultaneous"] = False
    bics = {n: {c: fits[n][c]["bic"] for c in conditions} for n in (1, 2, 3)}
    best, label = _best(bics, conditions)
    finite = all(np.isfinite(bics[n][c]) for n in (1, 2, 3) for c in conditions)
    return {"kind": kind, "conditions": conditions, "values": values, "steps": steps, "r2_lim": float(r2_lim),
            "simultaneous": simultaneous, "fits": fits, "best": best, "best_label": label, "bic_finite": bool(finite),
            "significance": significance_rows(values), "dropped_zero": dropped, "seed": int(seed),
            "msd_settings": dict(msd_run["settings"])}


def refit_condition(run: dict, condition: str, n_pop: int, *, seed: int) -> dict:
    """"Redo fit for <condition>" (`redofit`, available only when not simultaneous): this condition at ``n_pop`` again
    from new starts, then the Best Model row recomputed. Mutates and returns ``run``."""
    if run["simultaneous"]:
        raise LegacyViewError("Not available when simultaneous fit is selected")
    run["fits"][n_pop][condition] = _fit_one(run["values"][condition], run["steps"], n_pop, run["kind"], seed)
    run["fits"][n_pop][condition]["simultaneous"] = False
    bics = {n: {c: run["fits"][n][c]["bic"] for c in run["conditions"]} for n in (1, 2, 3)}
    run["best"], run["best_label"] = _best(bics, run["conditions"])
    return run


def _fit_for(run: dict, condition: str, n_pop_choice) -> dict:
    n = run["best"][condition] if n_pop_choice == "best" else int(n_pop_choice)
    return run["fits"][n][condition]


def appd_table(runs: dict, kind: str, n_pop_choice: str) -> tuple:
    """`AppDTable`: ``(column names, rows)``: the significance block (`significance_rows`, on the values of the shown
    run), then a GAMMA block and a GAUSSIAN block (each 'Pop k ± sd [%]', 'D_k ± sd', R², Best Model); the block of a
    type not run yet stays blank, as 2.0's persistent table did. ``n_pop_choice`` "1", "2", "3" or "best"."""
    run = runs[kind]
    conditions = run["conditions"]
    names = [" "] + list(conditions)
    rows = [list(r) for r in run["significance"]]

    def pct(v, s):
        return f"{100 * v:2.1f} ± {100 * s:2.1f}"

    for block in APPD_TYPES:
        rows.append([block] + [""] * len(conditions))
        have = runs.get(block)
        pop_rows = [[f"Pop {k} ± sd [%]"] + [""] * len(conditions) for k in (1, 2, 3)]
        d_rows = [[f"D{k} ± sd [µm²/s]"] + [""] * len(conditions) for k in (1, 2, 3)]
        r2_row = ["R²"] + [""] * len(conditions)
        best_row = ["Best Model"] + [""] * len(conditions)
        if have is not None:
            for i, c in enumerate(conditions, start=1):
                fit = _fit_for(have, c, n_pop_choice)
                for k in range(fit["n_pop"]):
                    pop_rows[k][i] = pct(fit["fraction"][k], fit["fraction_sd"][k])
                    d_rows[k][i] = f"{fit['D'][k]:1.3f} ± {fit['D_sd'][k]:1.3f}"
                r2_row[i] = f"{fit['r2']:1.3f}"
                best_row[i] = have["best_label"][c]
        rows.extend(pop_rows + d_rows + [r2_row, best_row])
    return names, rows


def bubble_data(run: dict, n_pop_choice: str) -> dict:
    """`bubbleplotAPPD`: the populations of every condition at one count: ``a`` the shares in per cent (<= 0 shown as
    1e-5, said), ``a_sd``, ``y`` the D, ``y_sd``; "best" draws THREE (2.0's `otherwise`, kept)."""
    n = 3 if n_pop_choice == "best" else int(n_pop_choice)
    conditions = run["conditions"]
    a, a_sd, y, y_sd, cond = [], [], [], [], []
    for i, c in enumerate(conditions):
        fit = run["fits"][n][c]
        for k in range(n):
            a.append(100.0 * float(fit["fraction"][k]))
            a_sd.append(100.0 * float(fit["fraction_sd"][k]))
            y.append(float(fit["D"][k]))
            y_sd.append(float(fit["D_sd"][k]))
            cond.append(i)
    return {"n_pop": n, "a": np.array(a), "a_sd": np.array(a_sd), "y": np.array(y), "y_sd": np.array(y_sd),
            "condition_index": np.array(cond), "any_nonpositive": bool(np.any(np.array(a) <= 0))}


def curves_for(run: dict, condition: str, n_pop_choice: str, x_grid) -> dict:
    """`AppDPlot`'s curves on ``x_grid``: each population's ``fraction * pdf`` and their sum, for the count shown."""
    fit = _fit_for(run, condition, n_pop_choice)
    x_grid = np.asarray(x_grid, dtype=float)
    spreads = fit["sigma"] if fit["sigma"] is not None else [None] * fit["n_pop"]
    parts = [f * _component_pdf(run["kind"], x_grid, c, s, run["steps"])
             for f, c, s in zip(fit["fraction"], fit["D"], spreads)]
    return {"n_pop": fit["n_pop"], "parts": parts, "sum": np.sum(parts, axis=0)}


# --------------------------------------------------------------------------------------------- #
# The speed maps
# --------------------------------------------------------------------------------------------- #

def model_cell_contour(n: int = 96) -> np.ndarray:
    """A model unit cell like 2.0's `modelCell.mat` (`SMT.m:171`): x from 0 to 1 along the length, y from -0.5 to 0.5
    across the width, the caps half-ellipses (a cell three times as long as wide: semi-axes 1/6 and 1/2)."""
    cap = 1.0 / 6.0
    theta = np.linspace(-np.pi / 2, np.pi / 2, n // 4 | 1)             # an odd count: the tip itself is a point
    right = np.column_stack([1 - cap + cap * np.cos(theta), 0.5 * np.sin(theta)])
    top = np.column_stack([np.linspace(1 - cap, cap, n // 4), np.full(n // 4, 0.5)])
    left = np.column_stack([cap - cap * np.cos(theta), -0.5 * np.sin(theta)])
    bottom = np.column_stack([np.linspace(cap, 1 - cap, n // 4), np.full(n // 4, -0.5)])
    return np.vstack([bottom, right, top, left])


def speedmap(project, msd_run: dict, run_or_none, condition: str, lower: float, upper: float,
             r2_lim: float = DEFAULT_R2_LIMIT) -> dict:
    """`plotSpeedmap.m` for ``condition``: the tracks whose per-track D (the MSD fit, `legacy_view_msd.msd_tracks`) lies
    in ``[lower, upper]`` and whose R^2 >= ``r2_lim``, positions normalised by the cell's length and width, those
    outside the model contour dropped. Returns the tracks (``{uid: (n, 2)}``), the trim counts
    (`legacy.speedmap.contour_trim`), the heat map (20 x 20 histogram, 32-fold cubic interpolation, per point) and the
    labels. RAISES where no track lies in the window."""
    from matplotlib.path import Path
    from scipy.ndimage import zoom

    from .canonical import without_attrs
    from .legacy.speedmap import contour_trim
    from .legacy_view_msd import msd_tracks

    if not lower < upper:
        raise LegacyViewError(f"the minimum D ({lower}) must be below the maximum D ({upper})")
    tracks = msd_tracks(project, msd_run, condition)
    keep = (tracks["r2"] >= r2_lim) & (tracks["D"] >= lower) & (tracks["D"] <= upper)
    if not keep.any():
        raise LegacyViewError(f"{condition}: no track has an apparent D between {lower:g} and {upper:g} "
                              f"with R² >= {r2_lim:g}")
    uids = [u for u, k in zip(tracks["uid"], keep) if k]
    projected = without_attrs(project.projected(condition))
    projected = projected[projected["uid"].isin(uids)].sort_values(["uid", "frame"])
    positions = {}
    for uid, part in projected.groupby("uid", sort=False, observed=True):
        positions[uid] = np.column_stack([part["l_norm"].to_numpy(dtype=float),
                                          part["d_norm"].to_numpy(dtype=float) / 2.0])
    if not positions:
        raise LegacyViewError(f"{condition}: the selected tracks have no cell coordinates")
    contour = model_cell_contour()
    path = Path(np.vstack([contour, contour[:1]]))
    trim = contour_trim(positions, lambda xy: path.contains_points(xy, radius=-1e-9) | path.contains_points(xy))
    kept = trim["kept"]
    stacked = np.vstack([np.vstack([xy, [np.nan, np.nan]]) for xy in kept.values()])
    pts = stacked[~np.isnan(stacked[:, 0])]
    n_rows = len(stacked)                                  # 2.0 counts the separator rows too (size(normTracks,1))
    hist, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=[np.linspace(-0.1, 1.1, 21), np.linspace(-0.6, 0.6, 21)])
    heat = zoom(hist, 32, order=3) / n_rows
    return {"condition": condition, "lower": float(lower), "upper": float(upper), "r2_lim": float(r2_lim),
            "n_tracks": len(kept), "tracks": kept, "stacked": stacked, "trim": {k: v for k, v in trim.items()
                                                                              if k != "kept"},
            "heat": np.clip(heat, 0.0, None), "contour": contour, "min_length": tracks["settings"]["min_length"],
            "n_label": n_rows}
