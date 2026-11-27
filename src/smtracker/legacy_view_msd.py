"""The legacy window's MSD tab: SMTracker 2.0's mean-squared-displacement tab, computed the old way.

Ruled 26 Nov 2026 (`docs/legacy_window_plan.md`). 2.0's tab (`SMTracker.m:409-453`, callbacks `MSD`, `fcnMSD`, `estLocEr`;
`@SMT/calcMSD_gui.m`, `getMeanMSD_gui.m`, `getFitMSD_gui.m`, `fillMSDtable.m`, `fitMSDTracks.m`, `plotMSDTracks.m`;
`Help_fcn/linWfit.m`, `anomalousWfit.m`, `diffusionClassification.m`, `ftest.m`; manual Figs 7 and 8). Headless, the
thin-Qt rule: `smtracker.ui.legacy.tab_msd` only lays out and draws. Never shown beside an SMTracker 3 number.

WHAT 2.0 DID, kept
- The ensemble curve (`getMSD`): every track's own time-averaged MSD (all pairs, lag 0 first), pooled per lag as a mean
  weighted by the pairs behind each point, its weighted spread and its "degrees of freedom" column; the plot draws
  ``std / sqrt(N)`` as the error bar. "specific tau" (the default, 9) keeps only tracks of at least that many rows and
  cuts every curve to them (`calcMSD_gui.m`); "all tau" keeps every track whole.
- The ensemble fit (`getFitMSD_gui.m`): an UNWEIGHTED straight line through lags 1 to the "Linear Fit" box (clamped to
  the curve; at 9 and 9 it runs past the cut curve and the catch fits lags 1 to 8, every time), ``D = slope / 4`` for
  x and y as well (a one-axis D reads half; 2.0's own label, kept, said in the Console), R^2 unweighted.
- The per-track fits (`fitMSDTracks.m`, always direction xy): tracks of at least `min_length` rows, lag 0 and NaN rows
  dropped, weights ``1/std`` of each lag's squared displacements, ``linWfit`` replacing an infinite weight (a lag on
  one pair) by ``1 / sum(1/w)``, then the last lag dropped ("Exclude last timelag", on). Linear: a weighted line, D =
  slope / 4. Brownian vs Anomalous: the F-test (`ftest`, 2 against 3 parameters, p < 0.1 chooses the power law,
  super- or sub-diffusive by the sign of alpha - 1). Anomalous free exponent: the power law alone, Brownian where
  |alpha - 1| < 0.1. `legacy.sla`'s weighted fits do the arithmetic.
- The localisation-error panel (`plotMSDTracks.m`): the ticked tracks (R^2 > 0.99 and a positive intercept, 2.0's
  suggestion) drawn grey with D = 0.1, 1 and 3 guide lines, a histogram of ``sqrt(|intercept / 4|)`` in 0.02 um bins,
  its mean the panel's title.

WHAT IS NOT KEPT (each said where it shows)
- The anomalous window's last panel is titled "Diffusion Histogram for the brownian tracks" and in 2.0 draws the ALPHA of
  the tracks with alpha - 1 < 0.1; here it draws the D of the Brownian tracks, which is what its title says.
- The heat map of D against alpha: 2.0 interpolated a 5 x 5 histogram and transposed it into the grid; here the grid
  is oriented as the axes are.
- Tracks whose fit does not exist (fewer than four usable lags for the power law, a lag at or below zero) are counted
  and said, never dropped silently; 2.0 stopped on an error dialog.

INTERFACES
- `msd_run(project, ...)`: the tab's Run. `msd_table(run)`: 2.0's results table. `msd_tracks(project, run, condition)`:
  the per-track fits and the localisation-error panel's numbers (xy). `anomalous_summary(tracks)`: the Anomalous Analysis
  window's numbers. `appd_settings(run)`: what the APPD tab reads from this tab (2.0 read the same controls).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .legacy_view import LegacyViewError

__all__ = ["MSD_METHODS", "MSD_DIRECTIONS", "DEFAULT_SPECIFIC_TAU", "DEFAULT_LINEAR_FIT", "LOC_ERR_BIN_UM",
           "FP_DIFFUSION", "MOTION_LABELS", "TICK_R2", "LINEAR_TOL", "F_TEST_ALPHA", "default_min_length",
           "ensemble_curve", "fit_ensemble", "fit_track", "msd_run", "msd_table", "msd_tracks",
           "anomalous_summary", "localisation_error", "appd_settings"]

#: The tab's "Choose MSD method" radio buttons (`SMTracker.m:482-493`).
MSD_METHODS = ("Linear", "Brownian vs Anomalous", "Anomalous free exponent")
#: "Choose direction" (`SMTracker.m:456-472`); the camera's x and y, not the cell's axes.
MSD_DIRECTIONS = ("xy", "x", "y", "all")
DEFAULT_SPECIFIC_TAU = 9
DEFAULT_LINEAR_FIT = 9
#: `plotMSDTracks.m`: the histogram's BinWidth (um), the free fluorophore's D (`FP_diffusion`, um^2/s).
LOC_ERR_BIN_UM = 0.02
FP_DIFFUSION = 3.0
#: `diffusionClassification.m:44`'s order is Brownian, Super, Sub; the bar plot's own names (`fitMSDTracks.m:129`).
MOTION_LABELS = {1: "Brownian", 2: "Superdiffusive", 3: "Subdiffusive"}
#: 2.0's suggestion for the localisation-error panel: R^2 above this and a positive intercept (`check_`).
TICK_R2 = 0.99
#: `anomalousWfit.m:5`: alpha within this of 1 counts as Brownian when the exponent is free.
LINEAR_TOL = 0.1
#: `diffusionClassification.m:28`: the F-test rejects the straight line below this.
F_TEST_ALPHA = 0.1


def default_min_length(tau_mode: str, method: str, specific_tau: int = DEFAULT_SPECIFIC_TAU) -> int:
    """The "Minimum tau" box as `popMSD` sets it: the specific tau itself in "specific" mode, else 5 for Linear and
    9 for the other methods (`SMTracker.m:3289-3304`)."""
    if tau_mode == "specific":
        return int(specific_tau)
    return 5 if method == "Linear" else 9


# --------------------------------------------------------------------------------------------- #
# The curves
# --------------------------------------------------------------------------------------------- #

def _direction_table(table: pd.DataFrame, direction: str) -> pd.DataFrame:
    """The table whose ``x_um`` is the chosen axis and ``y_um`` zero (2.0 removed the other column,
    `calcMSD_gui.m:62-70`), so one squared-displacement routine serves xy, x and y."""
    if direction == "xy":
        return table
    out = table.copy()
    if direction == "y":
        out["x_um"] = out["y_um"].to_numpy(dtype=float)
    out["y_um"] = 0.0
    return out


def _curves(table: pd.DataFrame, direction: str, tau_mode: str, specific_tau: int) -> pd.DataFrame:
    """Every kept track's MSD per lag (`calcMSDTracks_.m`: every pair of rows ``lag`` apart, lag 0 left out), one row per
    track and lag: ``uid``, ``lag_frames``, ``lag_time_s``, ``msd_um2``, ``std_um2`` (the spread of that lag's squared
    displacements, `nanstd`: 0 for one pair) and ``n_pairs``. "specific" keeps tracks of at least ``specific_tau`` rows
    and stops at lag ``specific_tau - 1``; "all" keeps every track to its last lag. Pairs are taken by ROW, as 2.0's
    were; `msd.msd_per_track` (pairing "row") gives the same numbers, one lag at a time, too slowly for "all"."""
    from .canonical import without_attrs

    plain = without_attrs(table).sort_values(["uid", "frame"], kind="stable")
    lengths = plain.groupby("uid", observed=True).size()
    if tau_mode == "specific":
        keep = lengths.index[lengths >= specific_tau]
        longest = int(specific_tau) - 1
    else:
        keep = lengths.index
        longest = int(lengths.max()) - 1 if len(lengths) else 0
    if len(keep) == 0:
        raise LegacyViewError(f"no track reaches {specific_tau} localisations (specific tau {specific_tau})")
    sub = _direction_table(plain[plain["uid"].isin(keep)], direction)
    codes, uniques = pd.factorize(sub["uid"], sort=False)
    x = sub["x_um"].to_numpy(dtype=float)
    y = sub["y_um"].to_numpy(dtype=float)
    tau = (sub.groupby(codes)["tau_s"].first().to_numpy(dtype=float) if "tau_s" in sub.columns
           else np.full(len(uniques), float(table.attrs["params"].tau_s)))
    n_tracks = len(uniques)
    frames = []
    for lag in range(1, max(longest, 1) + 1):
        i = np.flatnonzero(codes[:-lag] == codes[lag:])
        if not len(i):
            break
        d2 = (x[i + lag] - x[i]) ** 2 + (y[i + lag] - y[i]) ** 2
        c = codes[i]
        n = np.bincount(c, minlength=n_tracks)
        mean = np.bincount(c, weights=d2, minlength=n_tracks) / np.maximum(n, 1)
        sq = np.bincount(c, weights=(d2 - mean[c]) ** 2, minlength=n_tracks)
        std = np.sqrt(np.where(n > 1, sq / np.maximum(n - 1, 1), 0.0))
        have = np.flatnonzero(n > 0)
        frames.append(pd.DataFrame({"uid": uniques[have], "lag_frames": lag, "lag_time_s": lag * tau[have],
                                    "msd_um2": mean[have], "std_um2": std[have], "n_pairs": n[have]}))
    if not frames:
        raise LegacyViewError("no MSD point could be computed: no track has two localisations")
    return pd.concat(frames, ignore_index=True)


def ensemble_curve(curves: pd.DataFrame) -> np.ndarray:
    """`getMSD.m`: ``[dT, mean, std, N]`` per lag (lag 0 omitted: the tab plots and fits rows 2 on). The mean is
    weighted by each track's pairs at that lag, the spread is `getMSD`'s weighted unbiased estimator and N its
    "degrees of freedom" ``sum(n)^2 / sum(n^2)``. A lag with one contributing track has no spread (v2's 0/0: NaN);
    ONE track in all is returned as it is (`getMSD.m:13-14`). Vectorised; `legacy.sla.pooled_msd` is the same
    arithmetic one curve at a time."""
    if curves.empty:
        raise LegacyViewError("no MSD curve to pool")
    rows = []
    n_tracks = curves["uid"].nunique()
    for lag, part in curves.groupby("lag_frames", sort=True):
        m = part["msd_um2"].to_numpy(dtype=float)
        n = part["n_pairs"].to_numpy(dtype=float)
        ok = np.isfinite(m)
        m, n = m[ok], n[ok]
        if not len(m):
            continue
        t = float(part["lag_time_s"].iloc[0])
        sw, ssw = float(n.sum()), float((n ** 2).sum())
        mean = float((n * m).sum() / sw)
        if n_tracks == 1:
            std = float(np.nan_to_num(part["std_um2"].iloc[0], nan=0.0))
            rows.append([t, mean, std, float(n[0])])
            continue
        swv = float((n * (m - mean) ** 2).sum())
        denom = sw ** 2 - ssw
        std = float(np.sqrt(sw / denom * swv)) if denom > 0 else float("nan")
        rows.append([t, mean, std, sw ** 2 / ssw])
    if not rows:
        raise LegacyViewError("no lag had a finite MSD")
    return np.asarray(rows, dtype=float)


def _fit_window(n_available: int, linear_fit: int, min_length: int | None) -> tuple:
    """Lags 1..n of the curve the ensemble line is fitted over (`getFitMSD_gui.m:8-14`): the box's value, clamped to
    the specific tau (`validate`, `SMTracker.m:2410`); where that runs past the curve, the whole curve."""
    box = int(linear_fit) if min_length is None else min(int(linear_fit), int(min_length))
    fell_back = box > n_available
    return (n_available if fell_back else box), fell_back, box


def fit_ensemble(curve: np.ndarray, linear_fit: int, min_length: int | None = None) -> dict:
    """`getFitMSD_gui.m`: the unweighted line ``y = m x + b`` through the pooled curve's lags 1 to the box (see
    `_fit_window`); D = m / 4 (2.0's, for every direction), R^2 = 1 - SSE / SST of that ordinary fit."""
    n_use, fell_back, box = _fit_window(len(curve), linear_fit, min_length)
    part = curve[:n_use]
    ok = np.isfinite(part[:, 0]) & np.isfinite(part[:, 1])
    t, y = part[ok, 0], part[ok, 1]
    if len(t) < 2:
        raise LegacyViewError(f"the ensemble line needs two lags, the curve has {len(t)}")
    slope, intercept = np.polyfit(t, y, 1)
    resid = y - (slope * t + intercept)
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = float(1.0 - (resid ** 2).sum() / sst) if sst > 0 else float("nan")
    return {"slope": float(slope), "intercept": float(intercept), "D": float(slope / 4.0), "r2": r2,
            "n_lags": int(len(t)), "fell_back": bool(fell_back), "linear_fit_box": int(box)}


# --------------------------------------------------------------------------------------------- #
# One track's fit
# --------------------------------------------------------------------------------------------- #

def _weighted_r2(y, y_hat, w) -> float:
    total = float((w * (y - np.average(y, weights=w)) ** 2).sum())
    return float(1.0 - (w * (y - y_hat) ** 2).sum() / total) if total > 0 else float("nan")


def _lin(t, y, w) -> dict:
    """`linWfit.m`: a weighted line, ``[D, 1, intercept, R^2, 1]``."""
    from .legacy.sla import fit_linear_weighted

    fit = fit_linear_weighted(t, y, w)
    slope = fit["D_um2_per_s"] * 4.0
    r2 = _weighted_r2(y, slope * t + fit["intercept_um2"], w)
    return {"coeff": [fit["D_um2_per_s"], 1.0, fit["intercept_um2"], r2, 1], "sse": fit["sse"]}


def _ano_fit(t, y, w) -> dict:
    """`legacy.sla.fit_anomalous_weighted`'s fit (v2's start, bounds ``b <= 1`` and ``alpha <= 2``, the trust-region
    method at MATLAB fit's tolerances), with the Jacobian written out instead of differenced: the same problem and the
    same answer to the solver's tolerance, several times faster, which matters over thousands of tracks
    (`tests/test_legacy_msd.py` holds the two together)."""
    from scipy.optimize import least_squares
    from scipy.special import gamma as gamma_fn

    from .legacy.sla import V2_ANOMALOUS_UPPER, SLAError, _weighted_lsq

    if len(t) < 4:
        raise SLAError("a three-parameter fit needs at least four points")
    if np.any(t <= 0) or np.any(y <= 0):
        raise SLAError("the power law is seeded from log(t) and log(MSD)")
    slope, y0 = _weighted_lsq(np.column_stack([np.log(t), np.ones_like(t)]), np.log(y), w)[0]
    d_alpha0 = float(np.exp(y0) * gamma_fn(1.0 + slope) / 4.0)
    upper = np.asarray(V2_ANOMALOUS_UPPER, dtype=float)
    start = np.minimum(np.array([np.exp(y0), slope, d_alpha0]), upper)      # v2's order into (b, K, alpha)
    sw = np.sqrt(w)
    log_t = np.log(t)

    def residual(p):
        return sw * (y - (p[0] + p[1] * t ** p[2]))

    def jacobian(p):
        power = t ** p[2]
        return -sw[:, None] * np.column_stack([np.ones_like(t), power, p[1] * power * log_t])

    fit = least_squares(residual, start, jac=jacobian, bounds=(np.full(3, -np.inf), upper), method="trf",
                        xtol=1e-6, ftol=1e-6, max_nfev=600)
    b, k, alpha = (float(v) for v in fit.x)
    return {"D_alpha": k / 4.0, "alpha": alpha, "intercept_um2": b, "sse": float((fit.fun ** 2).sum())}


def _ano(t, y, w) -> dict:
    """`anomalousWfit.m`: ``b + K t^alpha``, ``[K / 4, alpha, b, R^2, type]``, type 1 where |alpha - 1| < 0.1."""
    fit = _ano_fit(t, y, w)
    k, alpha, b = fit["D_alpha"] * 4.0, fit["alpha"], fit["intercept_um2"]
    r2 = _weighted_r2(y, b + k * t ** alpha, w)
    kind = 1 if abs(alpha - 1.0) < LINEAR_TOL else (2 if alpha > 1.0 else 3)
    return {"coeff": [fit["D_alpha"], alpha, b, r2, kind], "sse": fit["sse"]}


def _f_test_p(n: int, sse_restricted: float, sse_full: float, np_restricted: int = 2, np_full: int = 3) -> float:
    """`ftest.m`: 1 where the bigger model does not fit better, else the F distribution's upper tail."""
    if sse_full >= sse_restricted:
        return 1.0
    df1, df2 = np_full - np_restricted, n - np_full
    if df2 <= 0:
        return 1.0
    f_stat = df2 * (sse_restricted - sse_full) / (df1 * sse_full)
    return float(stats.f.sf(f_stat, df1, df2))


def fit_track(t, y, std, *, method: str, exclude_last: bool = True) -> dict | None:
    """One track as `fitMSDTracks.m` fits it, from its lags 1.. (``t``, ``y`` the MSD, ``std`` the spread of each lag's
    squared displacements). Returns ``{"coeff": [D, alpha, ycut, R^2, type], "p": F-test p or None}``, or None where
    this track has no fit (the caller counts it). RAISES on an unknown method."""
    from .legacy.sla import SLAError

    if method not in MSD_METHODS:
        raise LegacyViewError(f"method must be one of {MSD_METHODS}, got {method!r}")
    t, y, std = (np.asarray(a, dtype=float) for a in (t, y, std))
    live = np.isfinite(y)                                  # `y(2:end)` of the non-NaN rows
    t, y, std = t[live], y[live], np.nan_to_num(std[live], nan=0.0)
    with np.errstate(divide="ignore"):
        w = 1.0 / std
    finite = np.isfinite(w)
    if not finite.any():
        return None
    w_lin = w.copy()
    w_lin[~finite] = 1.0 / np.sum(1.0 / w[finite])         # linWfit.m:5-8, over the whole vector
    cut = len(t) - (1 if exclude_last else 0)
    if cut < 3:
        return None
    tl, yl, wl = t[:cut], y[:cut], w_lin[:cut]
    ta, ya, wa = t[:cut], y[:cut], w[:cut]
    keep = np.isfinite(wa)                                 # prepareCurveData drops an infinite weight
    ta, ya, wa = ta[keep], ya[keep], wa[keep]
    try:
        if method == "Linear":
            return {"coeff": _lin(tl, yl, wl)["coeff"], "p": None}
        ano = _ano(ta, ya, wa)
        if method == "Anomalous free exponent":
            return {"coeff": ano["coeff"], "p": None}
        lin = _lin(tl, yl, wl)
    except SLAError:
        return None
    p = _f_test_p(len(tl), lin["sse"], ano["sse"])
    if p < F_TEST_ALPHA:
        alpha = ano["coeff"][1]
        if alpha == 1.0:
            return None                                     # 2.0 leaves the type undefined and stops
        coeff = list(ano["coeff"])
        coeff[4] = 2 if alpha > 1.0 else 3
    else:
        coeff = lin["coeff"]
    return {"coeff": coeff, "p": p}


# --------------------------------------------------------------------------------------------- #
# The tab's Run
# --------------------------------------------------------------------------------------------- #

def msd_run(project, conditions=None, *, tau_mode: str = "specific", specific_tau: int = DEFAULT_SPECIFIC_TAU,
            linear_fit: int = DEFAULT_LINEAR_FIT, direction: str = "xy", method: str = "Linear",
            min_length: int | None = None, exclude_last: bool = True) -> dict:
    """2.0's "Plot curves": the ensemble MSD of EVERY condition in the chosen direction(s), fitted as `getFitMSD_gui`
    fits it, and the table. ``tau_mode`` "specific" (default) or "all". The per-track fits (the localisation-error
    panel, the anomalous window, the APPD tab) are `msd_tracks`, on demand, for one condition.

    RAISES `LegacyViewError` naming a condition with no track long enough."""
    if tau_mode not in ("specific", "all"):
        raise LegacyViewError(f"tau mode must be 'specific' or 'all', got {tau_mode!r}")
    if direction not in MSD_DIRECTIONS:
        raise LegacyViewError(f"direction must be one of {MSD_DIRECTIONS}, got {direction!r}")
    if method not in MSD_METHODS:
        raise LegacyViewError(f"method must be one of {MSD_METHODS}, got {method!r}")
    if specific_tau < 3 or linear_fit < 2:
        raise LegacyViewError(f"specific tau {specific_tau} and linear fit {linear_fit}: a line needs at least 2 lags")
    conditions = list(conditions or project.conditions)
    directions = ["xy", "x", "y"] if direction == "all" else [direction]
    min_len = default_min_length(tau_mode, method, specific_tau) if min_length is None else int(min_length)
    clamp = int(specific_tau) if tau_mode == "specific" else None
    ensemble: dict = {}
    keep: dict = {}
    for condition in conditions:
        table = project.condition(condition)
        ensemble[condition] = {}
        for d in directions:
            try:
                curves = _curves(table, d, tau_mode, int(specific_tau))
            except LegacyViewError as exc:
                raise LegacyViewError(f"{condition}: {exc}") from exc
            curve = ensemble_curve(curves)
            fit = fit_ensemble(curve, linear_fit, clamp)
            ensemble[condition][d] = {"curve": curve, "fit": fit, "n_tracks": int(curves["uid"].nunique())}
            if d == "xy":
                keep[condition] = curves
    settings = {"tau_mode": tau_mode, "specific_tau": int(specific_tau), "linear_fit": int(linear_fit),
                "direction": direction, "method": method, "min_length": min_len, "exclude_last": bool(exclude_last)}
    return {"conditions": conditions, "directions": directions, "settings": settings, "ensemble": ensemble,
            "tau_s": float(project.params.tau_s), "_curves_xy": keep}


def appd_settings(run: dict) -> dict:
    """What the APPD tab takes from this tab (`validateinstD`, `SMTracker.m:2312-2345`): "specific tau", xy, the MSD
    method, the minimum number of lags and whether the last lag is excluded."""
    s = run["settings"]
    return {"specific_tau": s["specific_tau"], "method": s["method"], "exclude_last": s["exclude_last"]}


def msd_table(run: dict) -> tuple:
    """2.0's "MSD calculation results" (`fillMSDtable.m`): ``(column names, rows)``. Rows: direction, specific tau,
    #Tracks, then D and R per direction (D = slope / 4, rounded to three decimals as 2.0 rounded)."""
    conditions, s = run["conditions"], run["settings"]
    names = [""] + list(conditions)
    rows = [["direction"] + [s["direction"]] * len(conditions),
            ["specific τ"] + [str(s["specific_tau"])] * len(conditions)]
    first = run["directions"][0]
    rows.append(["#Tracks"] + [str(run["ensemble"][c][first]["n_tracks"]) for c in conditions])
    for d in run["directions"]:
        label = "D xy" if d == "xy" else f"D {d}"
        rows.append([f"{label} [µm²/s]"] + [f"{run['ensemble'][c][d]['fit']['D']:.3f}" for c in conditions])
        rows.append([f"R {d}" if len(run["directions"]) > 1 else "R"]
                    + [f"{run['ensemble'][c][d]['fit']['r2']:.3f}" for c in conditions])
    return names, rows


# --------------------------------------------------------------------------------------------- #
# Per track: the localisation-error panel, the anomalous window, the APPD tab's values
# --------------------------------------------------------------------------------------------- #

def _fit_chunk(chunk, _shared) -> list:
    """Pool worker: ``chunk`` is ``(method, exclude_last, [(t, msd, std), ...])``; one fit (or None) per track."""
    method, exclude_last, items = chunk
    return [fit_track(t, y, std, method=method, exclude_last=exclude_last) for t, y, std in items]


#: Below this many tracks, or for the Linear method (a closed form), the fits run in this process; the power law is
#: a nonlinear fit that takes up to a second on a flat track, and a pool earns its start-up over a few hundred.
_POOL_MIN_TRACKS = 300


def _fit_all(items: list, method: str, exclude_last: bool) -> list:
    """`fit_track` for every ``(t, msd, std)``, in order; the power-law methods fan out over the cores
    (`smtracker.parallel.condition_map`), the same fits as one at a time."""
    if method == "Linear" or len(items) < _POOL_MIN_TRACKS:
        return [fit_track(t, y, std, method=method, exclude_last=exclude_last) for t, y, std in items]
    from . import parallel

    n_chunks = max(1, parallel.n_workers(None) * 4)
    size = -(-len(items) // n_chunks)
    chunks = [(method, exclude_last, items[i:i + size]) for i in range(0, len(items), size)]
    out: list = []
    for part in parallel.condition_map(_fit_chunk, chunks, shared={}, n_jobs=None):
        out.extend(part)
    return out


def msd_tracks(project, run: dict, condition: str) -> dict:
    """`fitMSDTracks.m` for ``condition`` at the run's settings (always xy): one fit per track of at least the
    run's minimum length.

    Returns arrays over the fitted tracks: ``D``, ``alpha``, ``ycut``, ``r2``, ``type`` (1 Brownian, 2 super-, 3
    sub-diffusive), ``p`` (the F-test's, NaN unless Brownian vs Anomalous), ``length`` (localisations), ``movie``,
    ``cell``, ``track`` (2.0's numbers within the condition), ``ticked`` (R^2 > 0.99 and a positive intercept),
    ``curves`` (each track's ``(t, msd)`` from lag 1, the localisation-error panel's grey lines) and the counts:
    ``n_candidates`` (tracks of at least the minimum), ``n_unfitted``."""
    from .canonical import without_attrs

    s = run["settings"]
    if condition not in run["conditions"]:
        raise LegacyViewError(f"no condition {condition!r} in this run")
    cached = run.setdefault("_tracks", {})
    if condition in cached:
        return cached[condition]
    curves = run["_curves_xy"].get(condition)
    if curves is None:
        curves = _curves(project.condition(condition), "xy", s["tau_mode"], s["specific_tau"])
    table = without_attrs(project.condition(condition))
    lengths = table.groupby("uid", observed=True).size()
    info = table.groupby("uid", observed=True).agg(movie=("movie", "first"), cell=("cell", "first"))
    movie_order = {m: i + 1 for i, m in enumerate(sorted(table["movie"].astype(str).unique()))}
    min_len = s["min_length"]
    candidates = [u for u in curves["uid"].unique() if lengths.loc[u] >= min_len]
    if not candidates:
        raise LegacyViewError(f"{condition}: no track of at least {min_len} localisations")
    rows, kept_curves, kept_uids = [], [], []
    n_unfitted = 0
    grouped = {u: g.sort_values("lag_frames") for u, g in curves.groupby("uid", sort=False)}
    items = [(grouped[u]["lag_time_s"].to_numpy(), grouped[u]["msd_um2"].to_numpy(), grouped[u]["std_um2"].to_numpy())
             for u in candidates]
    fits = _fit_all(items, s["method"], s["exclude_last"])
    for number, (uid, item, fit) in enumerate(zip(candidates, items, fits), start=1):
        if fit is None:
            n_unfitted += 1
            continue
        D, alpha, ycut, r2, kind = fit["coeff"]
        try:
            cell = int(info.loc[uid, "cell"])
        except (TypeError, ValueError):
            cell = 0
        rows.append((D, alpha, ycut, r2, kind, np.nan if fit["p"] is None else fit["p"], int(lengths.loc[uid]),
                     movie_order[str(info.loc[uid, "movie"])], cell, number))
        kept_uids.append(uid)
        kept_curves.append((item[0].astype(float), item[1].astype(float)))
    if not rows:
        raise LegacyViewError(f"{condition}: no track produced a fit ({n_unfitted} candidates)")
    a = np.asarray(rows, dtype=float)
    out = {"condition": condition, "uid": kept_uids, "D": a[:, 0], "alpha": a[:, 1], "ycut": a[:, 2], "r2": a[:, 3],
            "type": a[:, 4].astype(int), "p": a[:, 5], "length": a[:, 6].astype(int), "movie": a[:, 7].astype(int),
            "cell": a[:, 8].astype(int), "track": a[:, 9].astype(int),
            "ticked": (a[:, 3] > TICK_R2) & (a[:, 2] > 0), "curves": kept_curves,
            "n_candidates": len(candidates), "n_unfitted": n_unfitted, "settings": dict(s)}
    cached[condition] = out
    return out


def localisation_error(tracks: dict, ticked=None) -> dict:
    """`plotMSDTracks.m`: ``sqrt(|intercept / 4|)`` (um) for the ticked tracks, 0.02 um bins, the mean in nm (the
    panel's title). ``ticked`` defaults to 2.0's suggestion. RAISES where none is ticked."""
    mask = tracks["ticked"] if ticked is None else np.asarray(ticked, dtype=bool)
    if not mask.any():
        raise LegacyViewError("no track is ticked: tick tracks for the localisation error")
    err = np.sqrt(np.abs(tracks["ycut"][mask] / 4.0))
    top = float(err.max())
    edges = np.arange(0.0, top + LOC_ERR_BIN_UM, LOC_ERR_BIN_UM)
    if len(edges) < 2:
        edges = np.array([0.0, LOC_ERR_BIN_UM])
    counts, edges = np.histogram(err, bins=edges)
    return {"error_um": err, "mean_nm": float(1000.0 * err.mean()), "counts": counts, "edges": edges,
            "n": int(mask.sum())}


def anomalous_summary(tracks: dict) -> dict:
    """The numbers of 2.0's Anomalous Analysis window (`fitMSDTracks.m:90-190`, manual Fig. 8): the alpha and R^2
    histograms (probability, bins of 0.1), the share of each type of motion, the (D, alpha) pairs with a 5 x 5
    histogram taken to a 25 x 25 grid by cubic interpolation and a density for the scatter, and the D of the Brownian
    tracks. Meant for the two anomalous methods; under Linear every track is Brownian with alpha 1 and the caller says
    so instead."""
    from scipy.ndimage import zoom

    alpha, r2, D, kind = tracks["alpha"], tracks["r2"], tracks["D"], tracks["type"]
    n = len(alpha)

    def prob_hist(values, width, lo=None, hi=None):
        lo = np.floor(np.nanmin(values) / width) * width if lo is None else lo
        hi = max(np.ceil(np.nanmax(values) / width) * width, lo + width) if hi is None else hi
        edges = np.arange(lo, hi + width / 2, width)
        counts, edges = np.histogram(values, bins=edges)
        return counts / max(len(values), 1), edges

    share = {label: float(np.mean(kind == k)) for k, label in MOTION_LABELS.items()}
    nb = 5
    x_lo, x_hi = -1.0, float(np.max(D))
    y_lo, y_hi = float(np.min(alpha)), float(np.max(alpha))
    if x_hi <= x_lo:
        x_hi = x_lo + 1.0
    if y_hi <= y_lo:
        y_hi = y_lo + 1.0
    hist, xe, ye = np.histogram2d(D, alpha, bins=nb, range=[[min(x_lo, float(np.min(D))), x_hi], [y_lo, y_hi]])
    fine = zoom(hist.T, nb, order=3)                       # (alpha rows, D columns), 5 x 5 to 25 x 25
    fine = np.clip(fine, 0.0, None)
    # density of each point on a 25 x 25 grid, `dscatter`'s idea (a smoothed 2-D histogram looked up at each point)
    dens_h, dxe, dye = np.histogram2d(D, alpha, bins=25)
    ix = np.clip(np.digitize(D, dxe[1:-1]), 0, 24)
    iy = np.clip(np.digitize(alpha, dye[1:-1]), 0, 24)
    density = dens_h[ix, iy]
    brown_D = D[kind == 1]
    return {"n": int(n), "alpha_hist": prob_hist(alpha, 0.1, 0.0, 2.0), "r2_hist": prob_hist(r2, 0.1),
            "share": share, "min_length": tracks["settings"]["min_length"],
            "D": D, "alpha": alpha, "heat": {"grid": fine, "x_edges": (xe[0], xe[-1]), "y_edges": (ye[0], ye[-1])},
            "density": density, "brownian_D": brown_D,
            "brownian_D_hist": (prob_hist(brown_D, 0.05) if len(brown_D) else None)}
