"""The legacy window's numbers: SMTracker 2.0's tabs, computed the old way on the loaded project.

Ruled 26 Nov 2026 ("we need a button to switch to legacy mode. reproducing as much as possible the old style and
overview"; `docs/legacy_window_plan.md`). The window (`smtracker.ui.legacy`) shows SMTracker 2.0's eight tabs on the
conditions the analysis workspace has loaded, with the day map, cell filter and exclusions the workspace uses. This
module holds every number it draws, one function per tab, headless (the thin-Qt rule); the window only draws.

What it is not: a second analysis. Every number here is the OLD program's way of computing it (the ported code in
`smtracker.legacy`, or a v3 function in its v2 mode), shown in a window of its own and never beside a v3 number. No
number from here enters a paper's deposit. The window may use 2.0's own words (free, mixed behaviour, dwelling,
static, mobile); everywhere else the neutral-labels gate holds.

INTERFACES
- `TABS`: 2.0's tab order. `STATS_ROWS`: the Import tab's Statistics rows, 2.0's own words.
- `import_statistics(project, conditions)`: 2.0's Statistics table (`@SMT/fillStatsTable_gui.m` over
  `@SMT/getCellStats.m`), one column per condition.
- `cell_lengths(project, condition)`: every outlined cell's length (um), the boxplot's data.
- `overlay(project, condition, movie, background)`: one movie's image, outlines and tracks (the Import tab's top
  panel), from the Images & tracks page's own payload.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

__all__ = ["TABS", "STATS_ROWS", "LegacyViewError", "import_statistics", "cell_lengths", "overlay",
           "condition_movies", "significance_rows", "SQD_LAG_CHOICES", "SQD_N_POPS", "sqd_steps",
           "gmm_best_n_pop", "sqd_run", "sqd_curve", "sqd_table"]

#: SMTracker 2.0's tabs, in its order (`SMTracker.m:147-154`).
TABS = ("Import", "SLA", "MSD", "Clustering", "GMM", "SQD", "APPD", "SDA")

#: The Import tab's Statistics rows, as 2.0 named them (`@SMT/fillStatsTable_gui.m:30-37`).
STATS_ROWS = ("# movies", "# cells", "# cells with tracks", "av. cell length [µm]", "# tracks",
              "av. lifetime [frames / s]", "#tracks/cell", "#steps/cell")


class LegacyViewError(RuntimeError):
    """A legacy number that cannot be computed on this project, said in words."""


def _round_sig(value: float, digits: int) -> float:
    """MATLAB's ``round(x, n, 'significant')``."""
    if not math.isfinite(value) or value == 0:
        return value
    return round(value, digits - 1 - int(math.floor(math.log10(abs(value)))))


def _num2str(value: float, digits: int) -> str:
    """MATLAB's ``num2str(x, n)``: ``n`` significant digits, no trailing zeros."""
    if not math.isfinite(value):
        return "NaN"
    return f"{value:.{digits}g}"


def _statistics_column(project, condition: str) -> list:
    """One condition's column of 2.0's Statistics table, row by row (`STATS_ROWS`).

    2.0's definitions (`@SMT/getCellStats.m`), kept: a track's lifetime is its number of localisations, and in
    seconds that number times the frame interval; tracks per cell and steps per cell are computed per MOVIE (over
    the movie's cells WITH tracks) and then averaged over the movies that have any; the cell length is the mean over
    every outlined cell with a measurable length, tracks or not."""
    from .images_view import ImagesError, _condition, _condition_cells
    from .qc_view import cell_lengths as qc_cell_lengths

    try:
        cells, _notes = _condition_cells(project, condition)
        data = _condition(project, condition)
    except ImagesError as exc:
        raise LegacyViewError(f"{condition}: {exc}") from exc
    from .canonical import without_attrs

    table = without_attrs(data["canonical"])
    per_track = table.groupby("uid", observed=True).agg(movie=("movie", "first"), n_loc=("uid", "size"),
                                                         tau=("tau_s", "median"))
    lifetimes = per_track["n_loc"].to_numpy(dtype=float)
    tau = float(np.nanmedian(per_track["tau"])) if len(per_track) else float("nan")
    with_tracks = cells[cells["n_tracks"] > 0]
    densities, steps = [], []
    for movie, group in with_tracks.groupby("movie", observed=True):
        tracks_here = per_track[per_track["movie"].astype(str) == str(movie)]
        if group.empty:
            continue
        densities.append(len(tracks_here) / len(group))
        steps.append(float((tracks_here["n_loc"] - 1).sum()) / len(group))
    lengths = qc_cell_lengths(project, condition)["lengths"]
    mean_life = float(lifetimes.mean()) if len(lifetimes) else float("nan")
    return [
        int(cells["movie"].nunique()),
        int(len(cells)),
        int(len(with_tracks)),
        _round_sig(float(np.mean(lengths)), 3) if lengths else float("nan"),
        int(len(per_track)),
        f"{_num2str(mean_life, 2)} / {_num2str(mean_life * tau, 2)}",
        float(np.mean(densities)) if densities else float("nan"),
        float(np.mean(steps)) if steps else float("nan"),
    ]


def import_statistics(project, conditions=None) -> pd.DataFrame:
    """2.0's Statistics table: `STATS_ROWS` down, one column per condition (in ``conditions``' order, default the
    project's). RAISES naming a condition that has no outlined cell in a tracked movie."""
    conditions = list(conditions or project.conditions)
    columns = {condition: _statistics_column(project, condition) for condition in conditions}
    return pd.DataFrame(columns, index=list(STATS_ROWS))


def cell_lengths(project, condition: str) -> np.ndarray:
    """Every outlined cell's length along its centreline (um), the Import tab's boxplot (2.0 drew every cell's
    length, tracks or not)."""
    from .qc_view import cell_lengths as qc_cell_lengths

    return np.asarray(qc_cell_lengths(project, condition)["lengths"], dtype=float)


def condition_movies(project, condition: str) -> list:
    """The condition's movies by their own names, in order (the Import tab's movie picker)."""
    from .images_view import condition_movies as movies

    return movies(project, condition)


def overlay(project, condition: str, movie: str | None = None, *, background: str = "phase") -> dict:
    """One movie as 2.0's "Overlay cells and tracks on images" panel draws it: the image (8-bit grey, stretched),
    the outlines and the tracks, in pixels. The Images & tracks page's own payload (`images_view.movie_payload`), so
    the two never disagree about where a track is. ``background`` "phase", "sum" or "none" (2.0's White)."""
    from .images_view import ImagesError, movie_payload

    try:
        payload = movie_payload(project, condition, movie, background="phase" if background == "none" else background)
    except ImagesError as exc:
        raise LegacyViewError(f"{condition}: {exc}") from exc
    if background == "none":
        payload = dict(payload, image=None, image_error=None)
    return payload


# --------------------------------------------------------------------------------------------- #
# Shared by the fitting tabs: 2.0's significance rows (`SMTracker.m:6151-6208`)
# --------------------------------------------------------------------------------------------- #

#: 2.0's three levels (`significanceTable`): *** below 0.01, ** below 0.05, * below 0.1.
SIGNIFICANCE_ALPHA = (0.1, 0.05, 0.01)


def significance_rows(samples: dict) -> list:
    """2.0's "significantly different?" block for ``{condition: values}``: a header row, a row naming the
    conditions, then one row per condition with each later condition's verdict. 2.0's test, kept: Levene's test on
    the absolute deviations from the mean; where it does not reject equal variances (p >= 0.1), Student's t-test's p
    ("(tt)"), otherwise Levene's own p ("(lv)"); stars at 0.1 / 0.05 / 0.01. On POOLED steps, as 2.0 tested: the old
    program's verdict, shown as such. Empty for one condition."""
    from scipy import stats

    names = list(samples)
    n = len(names)
    if n < 2:
        return []
    rows = [["significantly different?"] + [""] * n, ["stars / p-value"] + names]
    for i, a in enumerate(names):
        row = [a] + ["-"] * n
        x1 = np.asarray(samples[a], dtype=float)
        for j in range(i + 1, n):
            x2 = np.asarray(samples[names[j]], dtype=float)
            p = float(stats.levene(x1, x2, center="mean").pvalue)
            tag = "(lv)"
            if p >= SIGNIFICANCE_ALPHA[0]:
                p = float(stats.ttest_ind(x1, x2).pvalue)
                tag = "(tt)"
            stars = ("***" if p < SIGNIFICANCE_ALPHA[2] else "**" if p < SIGNIFICANCE_ALPHA[1]
                     else "*" if p < SIGNIFICANCE_ALPHA[0] else "ns")
            row[j + 1] = f"{tag} {stars} / {p:.4g}"
        rows.append(row)
    return rows


# --------------------------------------------------------------------------------------------- #
# SQD (`SMTracker.m:756-868` and `sqdFit`, `@SMT/sqdFit_.m`, `@SMT/sqdFit_Simultaneous.m`)
# --------------------------------------------------------------------------------------------- #

#: `sqdFit_.m:30`: a track of Nsteps = 4 localisations or fewer is dropped, so a track needs at least 5.
SQD_MIN_LOCALISATIONS = 5
#: 2.0's "#timelags" popup: one lag, or "1-4 together" (`currentmTlag == 5`).
SQD_LAG_CHOICES = ("1", "2", "3", "4", "1-4 together")
SQD_N_POPS = (1, 2, 3)


def sqd_steps(project, condition: str, lag: int) -> tuple:
    """``(squared displacements um^2, n_tracks)`` as `sqdFit_.m` collects them: every track of at least
    `SQD_MIN_LOCALISATIONS`, every pair ``lag`` rows apart (MATLAB's row pairing), pooled over the condition."""
    from .observables import squared_displacements

    table = project.condition(condition)
    counts = table.groupby("uid", observed=True).size()
    keep = counts[counts >= SQD_MIN_LOCALISATIONS].index
    table = table[table["uid"].isin(keep)]
    steps = squared_displacements(table, lag_frames=int(lag), pairing="row")
    column = "sq_disp_um2" if "sq_disp_um2" in steps.columns else steps.columns[-1]
    return steps[column].to_numpy(dtype=float), int(len(keep))


def _cdf_model(fit: dict, x, tau_lag: float):
    """``1 - sum_k f_k exp(-x / (4 D_k tau))`` at ``x`` (2.0's `twoPopulation.m` family)."""
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    for D, f in zip(fit["D"], fit["fraction"]):
        out -= f * np.exp(-x / (4.0 * D * tau_lag))
    return out


def _r2(y, y_pred) -> float:
    y, y_pred = np.asarray(y, float), np.asarray(y_pred, float)
    total = float(((y - y.mean()) ** 2).sum())
    return float(1.0 - ((y - y_pred) ** 2).sum() / total) if total > 0 else float("nan")


def gmm_best_n_pop(project, condition: str, tau_s: float) -> tuple:
    """``(best population count, {n: BIC})`` as 2.0's "Automatically" chose it (`SMTracker.m` `BIC`): the normal
    mixtures fitted to the ECDF of the stacked x and y jumps at one frame, 1 to 3 populations, BIC = n ln(RSS/n) +
    k ln(n) with n the jumps and k the fitted parameters (n sigmas and n - 1 fractions)."""
    from .jd import JdError, fit_jd, signed_displacements

    jumps = signed_displacements(project.condition(condition), direction="xy")["displacement_um"].to_numpy()
    bic = {}
    for n_pop in SQD_N_POPS:
        try:
            fit = fit_jd(jumps, tau_s=tau_s, n_pop=n_pop)
        except JdError:
            continue
        n = max(int(fit["n_displacements"]), 1)
        rss = max(float(fit["residual"]), 1e-300)
        bic[n_pop] = n * math.log(rss / n) + (2 * n_pop - 1) * math.log(n)
    if not bic:
        raise LegacyViewError(f"{condition}: no jump-distance fit for the automatic choice")
    return min(bic, key=bic.get), bic


def sqd_run(project, conditions=None, *, n_pop: int | None = None, simultaneous: bool = True,
            lag_choice: str = "1") -> dict:
    """2.0's SQD tab's Run & Plot. ``n_pop`` None is "Automatically" (2.0 took the first condition's BIC choice,
    `data.gmm.BIC(1).bestNpop_xy`, and so does this); ``lag_choice`` one of `SQD_LAG_CHOICES`. Every condition is
    fitted alone at 1, 2 and 3 populations (`legacy.sqd.fit_legacy_sqd`, v2's seeds and bounds, the nlparci
    interval); with ``simultaneous`` and more than one population the shared-D fit across the conditions replaces
    the D and fractions (`fit_legacy_sqd_simultaneous`, as `sqdFit` did). R^2 at lags 1 to 4 at the chosen count,
    the significance rows on the shown lag's steps, and the best model per condition."""
    from .legacy.sqd import SqdFitError, fit_legacy_sqd, fit_legacy_sqd_simultaneous

    conditions = list(conditions or project.conditions)
    if lag_choice not in SQD_LAG_CHOICES:
        raise LegacyViewError(f"time lag must be one of {SQD_LAG_CHOICES}, got {lag_choice!r}")
    tau = float(project.params.tau_s)
    lags = (1, 2, 3, 4) if lag_choice == "1-4 together" else (int(lag_choice),)
    best, bics = {}, {}
    # the BIC choice costs three multistart jump-distance fits a condition: made only for "Automatically", as 2.0
    # filled its "Best Model" row only once the GMM fits had run
    for condition in conditions:
        if n_pop:
            best[condition], bics[condition] = None, {}
            continue
        try:
            best[condition], bics[condition] = gmm_best_n_pop(project, condition, tau)
        except LegacyViewError:
            best[condition], bics[condition] = None, {}
    chosen = int(n_pop) if n_pop else (best.get(conditions[0]) or 2)
    steps = {lag: {c: sqd_steps(project, c, lag) for c in conditions} for lag in (1, 2, 3, 4)}
    fits: dict = {}
    for lag in lags:
        fits[lag] = {}
        for k in SQD_N_POPS:
            fits[lag][k] = {}
            for condition in conditions:
                values, n_tracks = steps[lag][condition]
                try:
                    fits[lag][k][condition] = fit_legacy_sqd(values, tau_s=tau * lag, n_pop=k, n_tracks=n_tracks)
                except SqdFitError as exc:
                    raise LegacyViewError(f"{condition}, {k} population(s), lag {lag}: {exc}") from exc
            if simultaneous and k > 1 and len(conditions) > 1:
                shared = fit_legacy_sqd_simultaneous({c: steps[lag][c][0] for c in conditions}, tau_s=tau * lag,
                                                     n_pop=k)
                for condition in conditions:
                    own = fits[lag][k][condition]
                    unit = shared["per_unit"][condition]
                    fits[lag][k][condition] = dict(own, D=shared["D"], D_lo=shared["D_lo"], D_hi=shared["D_hi"],
                                                   fraction=unit["fraction"], fraction_lo=unit["fraction_lo"],
                                                   fraction_hi=unit["fraction_hi"],
                                                   derived_fraction_index=unit["derived_fraction_index"],
                                                   simultaneous=True)
    shown = lags[0]
    r2_lags: dict = {}
    for condition in conditions:
        r2_lags[condition] = {}
        for lag in (1, 2, 3, 4):
            fit = fits.get(lag, {}).get(chosen, {}).get(condition)
            if fit is None:
                values, n_tracks = steps[lag][condition]
                fit = fit_legacy_sqd(values, tau_s=tau * lag, n_pop=chosen, n_tracks=n_tracks)
            r2_lags[condition][lag] = _r2(fit["cdf_y"], _cdf_model(fit, fit["cdf_x"], tau * lag))
    pearson = {}
    for condition in conditions:
        fit = fits[shown][chosen][condition]
        y, y_pred = fit["cdf_y"], _cdf_model(fit, fit["cdf_x"], tau * shown)
        pearson[condition] = float(np.corrcoef(y, y_pred)[0, 1] ** 2) if len(y) > 2 else float("nan")
    return {"conditions": conditions, "tau_s": tau, "lags": list(lags), "lag_choice": lag_choice,
            "n_pop": chosen, "automatic": n_pop is None, "simultaneous": bool(simultaneous),
            "best": best, "bic": bics, "fits": fits, "r2": r2_lags, "pearson_r2": pearson,
            "significance": {lag: significance_rows({c: steps[lag][c][0] for c in conditions}) for lag in lags},
            "n_tracks": {c: steps[1][c][1] for c in conditions},
            # the single-step distances, for the Jump Distance & GoF window (`plotPDF`, `ppplot_jd`)
            "r_um": {c: np.sqrt(steps[1][c][0]) for c in conditions},
            "n_steps": {lag: {c: int(len(steps[lag][c][0])) for c in conditions} for lag in lags}}


def sqd_curve(run: dict, condition: str, n_pop: int, lag: int, x) -> np.ndarray:
    """The fitted CDF of ``condition`` at ``n_pop`` and ``lag`` at ``x`` (the CDF panel's lines)."""
    return _cdf_model(run["fits"][lag][n_pop][condition], x, run["tau_s"] * lag)


def _half_widths(fit: dict, key: str) -> np.ndarray:
    lo, hi = np.asarray(fit[f"{key}_lo"], float), np.asarray(fit[f"{key}_hi"], float)
    return (hi - lo) / 2.0


def sqd_table(run: dict, lag: int | None = None) -> list:
    """2.0's "SQD Analysis Results" (`prepareSQDTable`), rows of strings: the significance block, "# Pops.", each
    population's share [%] and D [um^2/s] with their +- (the nlparci half-width; a derived share has none), the
    Pearson R^2 and R^2 at 1 to 4 frames, and the best model by BIC."""
    lag = run["lags"][0] if lag is None else lag
    conditions, n_pop = run["conditions"], run["n_pop"]
    rows = [list(r) for r in run["significance"].get(lag, [])]
    rows.append(["# Pops."] + [str(n_pop)] * len(conditions))
    fits = run["fits"][lag][n_pop]
    for k in range(n_pop):
        cells = []
        for c in conditions:
            fit = fits[c]
            half = _half_widths(fit, "fraction")[k]
            value = f"{100 * fit['fraction'][k]:.3g}"
            cells.append(value if not np.isfinite(half) else f"{value} ± {round(100 * half, 3):g}")
        rows.append([f"pop{k + 1} [%]"] + cells)
    for k in range(n_pop):
        cells = []
        for c in conditions:
            fit = fits[c]
            half = _half_widths(fit, "D")[k]
            value = f"{fit['D'][k]:.3g}"
            cells.append(value if not np.isfinite(half) else f"{value} ± {round(half, 3):g}")
        rows.append([f"D{k + 1} [µm²/s]"] + cells)
    rows.append(["Pearson R-Squared"] + [f"{run['pearson_r2'][c]:.3g}" for c in conditions])
    for k in (1, 2, 3, 4):
        rows.append([f"R²({k} frame{'s' * (k > 1)})"] + [f"{run['r2'][c][k]:.3g}" for c in conditions])
    rows.append(["Best Model"] + [str(run["best"].get(c) or "-") for c in conditions])
    return rows


# --------------------------------------------------------------------------------------------- #
# The Track Explorer (2.0's second window, `SMTracker.m:1054-1095`; `@SMT/SMTracker_explorer.m`,
# `track_movement.m`, `trackDistance.m`, `trackIntensity.m`, `cellMSD.m`)
# --------------------------------------------------------------------------------------------- #

#: 2.0's confinement circle and line: 3 x the slower population's sigma of the two-population jump fit on the
#: condition's x and y jumps (`obj.gmm.doublefit.nproj.xy.coeff(1)*3`).
CONFINEMENT_SIGMAS = 3.0


def confinement_radius_um(project, condition: str) -> float:
    """2.0's confinement radius for ``condition`` (um): `CONFINEMENT_SIGMAS` x the slower sigma of the
    two-population jump-distance fit on the stacked x and y jumps at one frame."""
    from .jd import JdError, fit_jd, signed_displacements

    jumps = signed_displacements(project.condition(condition), direction="xy")["displacement_um"].to_numpy()
    try:
        fit = fit_jd(jumps, tau_s=float(project.params.tau_s), n_pop=2)
    except JdError as exc:
        raise LegacyViewError(f"{condition}: no two-population jump fit for the confinement circle ({exc})") from exc
    return CONFINEMENT_SIGMAS * float(fit["sigma_um"][0])


def _track_msd(xy: np.ndarray) -> np.ndarray:
    """One track's MSD at lags 1 .. n-1 (every pair, `cellMSD.m`'s bulk sum)."""
    n = len(xy)
    out = np.full(max(n - 1, 0), np.nan)
    for lag in range(1, n):
        d = xy[lag:] - xy[:-lag]
        out[lag - 1] = float(np.nanmean((d ** 2).sum(axis=1)))
    return out


def _original_movie(project, condition: str, qualified: str) -> str:
    """A movie's own name from the project's qualified one (``<condition>_series001`` -> ``series001``), as
    `trackviewer._original_names` reads it: the project's record, else the condition prefix taken off."""
    source = getattr(project, "movie_source", None) or {}
    if qualified in source:
        return str(source[qualified][1])
    prefix = f"{condition}_"
    return qualified[len(prefix):] if qualified.startswith(prefix) else qualified


def qualified_uid(project, condition: str, uid) -> str:
    """The project's own name for a track the overlay names by its movie's original name (``series001:15.1`` ->
    ``<condition>_series001:15.1``); a name already the project's is returned as it is. RAISES for neither."""
    uid = str(uid)
    movies = [str(m) for m in project.condition(condition)["movie"].astype(str).unique()]
    movie, _sep, rest = uid.rpartition(":")
    if movie in movies:
        return uid
    for qualified in movies:
        if _original_movie(project, condition, qualified) == movie:
            return f"{qualified}:{rest}"
    raise LegacyViewError(f"{condition}: no track {uid!r}")


def track_explorer(project, condition: str, uid) -> dict:
    """Everything the Track Explorer draws for one track: its localisations relative to its start (track
    movement), its distance from the start and its step lengths over time (distance to origin), its intensity over
    time normalised to its maximum (2.0's ``int_/max(int_)``), the MSD of every track in its cell and their mean (MSD
    per cell), its cell, movie and frames. RAISES naming a track the condition does not hold."""
    from .canonical import without_attrs

    uid = qualified_uid(project, condition, uid)
    table = without_attrs(project.condition(condition))
    rows = table[table["uid"].astype(str) == uid].sort_values("frame")
    if rows.empty:
        raise LegacyViewError(f"{condition}: no track {uid!r}")
    tau = float(project.params.tau_s)
    xy = rows[["x_um", "y_um"]].to_numpy(dtype=float)
    rel = xy - xy[0]
    steps = np.concatenate([[0.0], np.sqrt((np.diff(xy, axis=0) ** 2).sum(axis=1))])
    time = (rows["frame"].to_numpy(dtype=float) - float(rows["frame"].iloc[0])) * tau
    amp = rows["amp"].to_numpy(dtype=float) if "amp" in rows.columns else np.full(len(rows), np.nan)
    peak = np.nanmax(amp) if np.isfinite(amp).any() else np.nan
    intensity = amp / peak if np.isfinite(peak) and peak > 0 else np.full(len(rows), np.nan)
    cell = rows["cell"].iloc[0]
    cell = None if cell is None or cell != cell else str(cell)
    msd_curves = []
    if cell is not None:
        for _uid, part in table[table["cell"].astype(str) == cell].groupby("uid", observed=True):
            part = part.sort_values("frame")
            curve = _track_msd(part[["x_um", "y_um"]].to_numpy(dtype=float))
            if len(curve):
                msd_curves.append(curve)
    longest = max((len(c) for c in msd_curves), default=0)
    stacked = np.full((len(msd_curves), longest), np.nan)
    for i, curve in enumerate(msd_curves):
        stacked[i, :len(curve)] = curve
    with np.errstate(all="ignore"):
        mean = np.nanmean(stacked, axis=0) if len(msd_curves) else np.array([])
    movie = str(rows["movie"].iloc[0])
    return {"condition": condition, "uid": uid, "cell": cell, "movie": movie,
            "movie_original": _original_movie(project, condition, movie),
            "frames": rows["frame"].astype(int).tolist(), "tau_s": tau,
            "relative_um": rel, "origin_distance_um": np.sqrt((rel ** 2).sum(axis=1)), "step_um": steps,
            "time_s": time, "intensity": intensity, "msd_curves": msd_curves, "msd_mean": mean,
            "msd_lags_s": np.arange(1, longest + 1) * tau}


def msd_per_cell_table(explorer: dict) -> pd.DataFrame:
    """2.0's "Save MSD in csv...": the cell's tracks' MSD, one column per track and the mean, by lag (s)."""
    out = pd.DataFrame({"lag_s": explorer["msd_lags_s"]})
    for i, curve in enumerate(explorer["msd_curves"], start=1):
        column = np.full(len(out), np.nan)
        column[:len(curve)] = curve
        out[f"track_{i}_um2"] = column
    out["mean_um2"] = explorer["msd_mean"]
    return out


# --------------------------------------------------------------------------------------------- #
# SQD's "Jump Distance Analisys & GoF" window (`SMTracker.m` `plotPDF`, `ppplot_jd`; manual Fig. 13)
# --------------------------------------------------------------------------------------------- #

#: The histogram's bins for the single-step distances (2.0's `plotPDF`: equal bins over the range).
JD_BINS = 40


def jd_gof(run: dict, condition: str, n_pop: int, *, bins: int = JD_BINS) -> dict:
    """One condition's single-step distances against the SQD fit at one frame, as 2.0's window drew them: the
    histogram as probabilities per bin, the fit's jump-distance density (Weimann et al. 2013: each population
    ``f (r / (2 D tau)) exp(-r^2 / (4 D tau))``) integrated over each bin, per population and summed, and the P-P
    plot (the observed cumulative share against the fit's, with a straight line fitted through it: slope, intercept
    and R^2). Only at one frame, as 2.0 showed it."""
    if 1 not in run["fits"]:
        raise LegacyViewError("the jump distances are drawn for a fit at one frame (2.0 showed them only then)")
    r = np.sort(np.asarray(run["r_um"][condition], dtype=float))
    fit = run["fits"][1][n_pop][condition]
    tau = float(run["tau_s"])
    edges = np.linspace(0.0, float(r.max()) if len(r) else 1.0, bins + 1)
    counts, _ = np.histogram(r, bins=edges)
    prob = counts / max(len(r), 1)

    def cdf_of(D, x):
        return 1.0 - np.exp(-(np.asarray(x) ** 2) / (4.0 * D * tau))

    parts = [f * np.diff(cdf_of(D, edges)) for D, f in zip(fit["D"], fit["fraction"])]
    total = np.sum(parts, axis=0)
    observed = (np.arange(1, len(r) + 1)) / max(len(r), 1)
    predicted = np.zeros_like(r)
    for D, f in zip(fit["D"], fit["fraction"]):
        predicted += f * cdf_of(D, r)
    if len(r) > 2:
        slope, intercept = np.polyfit(observed, predicted, 1)
        r2 = float(np.corrcoef(observed, predicted)[0, 1] ** 2)
    else:
        slope = intercept = r2 = float("nan")
    return {"condition": condition, "n_pop": n_pop, "n": int(len(r)), "edges": edges, "probability": prob,
            "model": total, "components": parts, "observed": observed, "predicted": predicted,
            "slope": float(slope), "intercept": float(intercept), "r2": r2}


# --------------------------------------------------------------------------------------------- #
# What the Import tab and the Track Explorer used to shape inside their Qt classes (thin-Qt rules 1 and 2)
# --------------------------------------------------------------------------------------------- #

def table_text(value) -> str:
    """A cell of 2.0's uitable: integers as they are, other numbers in MATLAB's short format."""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    value = float(value)
    if not np.isfinite(value):
        return "NaN"
    return f"{value:.5g}"


def statistics_table(project, conditions=None) -> dict:
    """`import_statistics` as plain lists: ``{"columns": [condition, ...], "rows": [(row name, [text per condition])]}``,
    each cell through `table_text`."""
    table = import_statistics(project, conditions)
    return {"columns": [str(c) for c in table.columns],
            "rows": [(str(row), [table_text(table.at[row, c]) for c in table.columns]) for row in table.index]}


def track_lengths(project, condition: str) -> list:
    """Every track's length in frames (localisations), the Import tab's "length of tracks" histogram."""
    from .canonical import without_attrs

    table = without_attrs(project.condition(condition))
    return [int(n) for n in table.groupby("uid", observed=True).size().to_numpy()]


def nearest_track(payload: dict, x: float, y: float, max_px: float):
    """The index of the overlay track with a localisation nearest ``(x, y)`` and within ``max_px``, else None
    (2.0's click-to-pick on the overlay)."""
    best, best_d = None, max_px
    for i, track in enumerate(payload["tracks"]):
        pts = np.asarray(track["points"], dtype=float)
        if not len(pts):
            continue
        d = float(np.min(np.hypot(pts[:, 0] - x, pts[:, 1] - y)))
        if d < best_d:
            best, best_d = i, d
    return best


def track_in_pixels(project, condition: str, uid: str) -> dict:
    """One track on the camera's pixel grid, in frame order, for the explorer's movie panel:
    ``{"x": [...], "y": [...], "frame": [...]}`` (pixel centre at 0, the viewer's convention)."""
    from .canonical import without_attrs
    from .viewer import PIXEL_ORIGIN_OFFSET

    per_px = float(project.params.um_per_pixel)
    rows = without_attrs(project.condition(condition))
    rows = rows[rows["uid"].astype(str) == uid].sort_values("frame")
    xs = rows["x_um"].to_numpy() / per_px - PIXEL_ORIGIN_OFFSET
    ys = rows["y_um"].to_numpy() / per_px - PIXEL_ORIGIN_OFFSET
    return {"x": [float(v) for v in xs], "y": [float(v) for v in ys],
            "frame": [int(v) for v in rows["frame"].to_numpy()]}


def save_msd_per_cell(explorer: dict, path) -> None:
    """2.0's "Save MSD in csv...": `msd_per_cell_table` written to ``path``."""
    msd_per_cell_table(explorer).to_csv(path, index=False)
