"""The legacy window's SLA tab: SMTracker 2.0's Stationary Localization Analysis, computed the old way.

Ruled 26 Nov 2026 (`docs/legacy_window_plan.md`). 2.0's tab (`SMTracker.m:292-408`, callbacks `SLA`, `SLAplots`,
`confinedMap`, `confinedHeatmap`; `@SMT/confTracks.m`, `@SMT/dwell_tracks.m`, `@SMT/fillSLATable_gui.m`,
`Help_fcn/confinedMSDMotion.m`, `Help_fcn/nTrans.m`, `Help_fcn/plotHeatmap.m`; manual Fig. 6, p. 10-12) takes a
condition, a confinement radius and a minimum number of steps, and shows either the "confined tracks" (every track on
a standard 1 x 3 um cell, free blue, mixed behaviour green, dwelling segments red; the transitions between free and
confined; the MSD of the confined segments with its F-test) or the "dwell times" (the survival curve of the dwell
times with a one- and a two-component fit; the dwell events per track length), and fills a table with a column per
condition. Everything is headless here and the Qt tab (`ui.legacy.tab_sla`) only draws it. The numbers are the old
program's: the detector is `smtracker.confinement` (the moving-ROI counter of `dwellCount2.m`, its one-gap quirk
included), the MSD classification is `smtracker.legacy.sla` (`classify_motion`, the F-test of `diffusionClassification`),
and the old words (free, mixed behaviour, dwelling, static, mobile) are 2.0's, allowed in this window only.

What 2.0 did that is kept (the window shows the old program), each recorded where it is computed:
- the radius choice is 2.0's popup; 2.0 offered the radius of the GMM, SQD and APPD fits beside a custom one, which
  needs those tabs' results first. The window offers the custom radius, whose default is 2.0's (one camera pixel,
  `initializeVars.m:18`), editable, and the radius the SQD tab's last fit implies is not offered (a recorded deviation).
- tracks are drawn on the 3 x 1 um model cell (`modelCell.mat`, embedded: `legacy_view_sda.model_cell_contour`) in
  coordinates normalised by each cell's length and width, and a localisation outside that outline is not drawn
  (`prepareToplotAllTracks`, `inCellContour`);
- the heat map divides by the row count INCLUDING the NaN rows that separate the tracks (`plotHeatmap.m:17` on the
  padded array), then up-scales 20 nodes to 609 by cubic interpolation, the old look (`interp2(v, 5, 'cubic')`);
- the transitions counter is `confinement.track_transitions` in its legacy (non-strict) mode;
- the F-test's error bars are `std / sqrt(numel(matrix))`, 2.0's divisor, which is four times the number of lags
  (`legacy.sla.legacy_errorbar_divisor`): a defect, reproduced because the window shows what 2.0 drew;
- the dwell fits' ± are half a 95 % interval from the fit's covariance (MATLAB `confint`).

Where this window departs from 2.0, and why:
- "Dwell min time" is shown in milliseconds as its label says; 2.0 put seconds under a "[ms]" label;
- the two-component dwell fit is returned ordered (the shorter time first); 2.0 left the order to the optimiser;
- a condition whose confined segments cannot be fitted reports it in words (`msd_error`) where 2.0 printed a warning
  and drew nothing.

INTERFACES
- `RADIUS_CHOICE_CUSTOM`, `MIN_STEPS_CHOICES`, `DEFAULT_MIN_STEPS`, `TABLE_ROWS`.
- `default_radius_nm(project)`: 2.0's custom radius, one camera pixel in nm.
- `sla_run(project, condition, radius_um, min_steps, mode)`: everything the tab draws for one condition.
- `sla_table_column(run)`: one condition's 13 table values (`TABLE_ROWS` order).
- `sla_table(columns, dwell_times)`: the table (2.0's rows, a column per condition, `significance_rows` appended once
  every condition has dwell times).
- `heatmap(run, show)`: the confinement heat map (and the x / y distributions) of the ticked sets.
- `fit_dwell_times(steps, tau_s, min_steps)`: 2.0's one- and two-exponential survival fits.
- `per_movie_tracks(project, condition, run)`: the "All Movies" gallery's data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .legacy_view import LegacyViewError, significance_rows

__all__ = ["RADIUS_CHOICE_CUSTOM", "MIN_STEPS_CHOICES", "DEFAULT_MIN_STEPS", "TABLE_ROWS", "default_radius_nm",
           "sla_run", "sla_table_column", "sla_table", "heatmap", "fit_dwell_times", "per_movie_tracks",
           "interp_cubic", "distribution_hist"]

#: 2.0's radius popup entry that this window offers (`SMTracker.m:2938`: "CUSTOM : 106 nm").
RADIUS_CHOICE_CUSTOM = "CUSTOM"
#: 2.0's "minimum # of steps" popup (`popPopUpMinStepsDwell`, `SMTracker.m:2954-2963`): 1 to 50, default 5.
MIN_STEPS_CHOICES = tuple(range(1, 51))
DEFAULT_MIN_STEPS = 5
#: The standard cell's axes, in units of cell length and width (`confTracks.m:6-7`, 3 um long).
CELL_LENGTH_UM = 3.0
MARGIN = 0.1

#: The table's rows, 2.0's words (`fillSLATable_gui.m:18-27`); the two dwell-time-fit rows are filled only after the
#: dwell times have been run for the condition.
TABLE_ROWS = (
    "Dwell radius cut [nm]",
    "Dwell min time [ms]",
    "Steps distribution free vs confined [%]",
    "static tracks [%]",
    "mobile tracks [%]",
    "  free [%]",
    "  mixed behaviour [%]",
    "average residence time [s]",
    "τ (1-comp.) [s]",
    "τ1 (2-comp.) [s]",
    "Fraction τ1 [%]",
    "τ2 (2-comp.) [s]",
    "Fraction τ2 [%]",
)


def default_radius_nm(project) -> float:
    """2.0's custom radius: one camera pixel (`initializeVars.m:18`, ``obj.pixel_size / 1e3`` um), in nm."""
    return float(project.params.pixel_size_nm)


def _normalised_table(project, condition: str) -> pd.DataFrame:
    """The condition's localisations, one row each, sorted by track and frame, with 2.0's normalised coordinates:
    the position along the cell over the cell's length (``nx``) and across it over the cell's width (``ny``), as
    `confTracks.m:62` builds them (`projTracks` over `normCell`). A cell the projection cannot measure carries no
    track here (2.0 gave its tracks NaN placeholders)."""
    try:
        table = project.projected(condition)
    except Exception as exc:                         # noqa: BLE001 - said by name, never an empty result
        raise LegacyViewError(f"{condition}: the tracks could not be placed on their cells ({exc})") from exc
    geometry = table.attrs.get("geometry") or {}
    if not geometry:
        raise LegacyViewError(f"{condition}: no cell has an outline with a centreline, so no track can be drawn on "
                              f"the standard cell")
    ordered = table.sort_values(["uid", "frame"], kind="stable")
    cells = ordered["cell"].astype(str)
    length = cells.map({k: g.length_um for k, g in geometry.items()}).to_numpy(dtype=float)
    width = cells.map({k: g.max_width_um for k, g in geometry.items()}).to_numpy(dtype=float)
    out = ordered.copy()
    out["nx"] = ordered["L_um"].to_numpy(dtype=float) / length
    out["ny"] = ordered["D_um"].to_numpy(dtype=float) / width
    out = out[np.isfinite(out["nx"]) & np.isfinite(out["ny"])]
    if out.empty:
        raise LegacyViewError(f"{condition}: no localisation could be expressed in its cell's coordinates")
    out.attrs.update(table.attrs)
    return out


def _pad(pieces: list) -> np.ndarray:
    """2.0's `prepareToplotAllTracks`: every piece followed by a NaN row, one array, and a point outside the model
    cell dropped (`inCellContour(.., allowGap=1)`: a NaN stays)."""
    from .legacy_view_sda import inside_model_cell

    if not pieces:
        return np.empty((0, 2))
    padded = np.concatenate([np.vstack([p, [np.nan, np.nan]]) for p in pieces])
    finite = np.isfinite(padded[:, 0])
    keep = ~finite
    keep[finite] = inside_model_cell(padded[finite, 0], padded[finite, 1])
    return padded[keep]


def sla_run(project, condition: str, radius_um: float, min_steps: int, mode: str = "confined") -> dict:
    """Everything 2.0's SLA tab shows for ONE condition at one radius and minimum number of steps.

    ``mode`` "confined" (the "confined tracks" radio: map, transitions, the MSD of the confined segments) or "dwell"
    (the "dwell times" radio: the survival curve with its fits, the dwell events per track length). The detection and
    the table numbers are the same in both, as `confTracks.m` and `dwell_tracks.m` store the same fields."""
    from . import confinement
    from .legacy.sla import SLAError

    if mode not in ("confined", "dwell"):
        raise LegacyViewError(f"mode must be 'confined' or 'dwell', not {mode!r}")
    if not radius_um > 0:
        raise LegacyViewError("the confinement radius must be positive")
    table = _normalised_table(project, condition)
    tau_s = float(project.params.frame_interval_ms) / 1000.0
    index = confinement.dwell_index(table, radius_um=radius_um, min_steps=min_steps)
    events = confinement.confinement_events(table, radius_um=radius_um, min_steps=min_steps, dwells=index)
    uids = list(events["uid"])
    n_tracks = len(uids)
    from .canonical import without_attrs

    positions = {u: g for u, g in without_attrs(table).groupby("uid", sort=False, observed=True)}

    free, mixed, segments = [], [], []
    lengths, n_events, all_steps = [], [], []
    for uid in uids:
        track = positions[uid]
        xy = track[["nx", "ny"]].to_numpy(dtype=float)
        taus, idx = index[uid]
        lengths.append(len(xy))
        n_events.append(len(taus))
        all_steps.extend(int(t) for t in taus)
        if taus:
            mixed.append(xy)
            for piece in idx:
                segments.append(xy[np.asarray(piece, dtype=int)])
        else:
            free.append(xy)
    n_localisations = int(len(table))
    n_mixed = len(mixed)
    classes = events["track_class"]
    n_static = int((classes == "confined").sum())

    transitions = confinement.transitions(table, radius_um=radius_um, min_steps=min_steps, dwells=index)
    by_length = transitions.groupby("n_localisations")
    trans = {"length": by_length.size().index.to_numpy(dtype=float),
             "free_to_confined": by_length["n_free_to_confined"].mean().to_numpy(dtype=float),
             "confined_to_free": by_length["n_confined_to_free"].mean().to_numpy(dtype=float),
             "total": int(transitions["n_transitions"].sum())}

    track_lengths = np.asarray(lengths, dtype=float)
    ev = np.asarray(n_events, dtype=float)
    v = np.unique(track_lengths)
    means = np.array([ev[track_lengths == k].mean() for k in v])
    sems = np.array([ev[track_lengths == k].std(ddof=1) / np.sqrt((track_lengths == k).sum())
                     if (track_lengths == k).sum() > 1 else np.nan for k in v])
    dwell_events = {"length": v, "mean": means, "sem": sems, "total": int(ev.sum())}

    steps = np.asarray(all_steps, dtype=int)
    run = {
        "condition": condition, "radius_um": float(radius_um), "min_steps": int(min_steps), "mode": mode,
        "tau_s": tau_s, "n_tracks": int(n_tracks), "n_localisations": n_localisations,
        "free_xy": _pad(free), "mixed_xy": _pad(mixed), "segments_xy": _pad(segments),
        "n_free": int(n_tracks - n_mixed), "n_mixed_incl_static": int(n_mixed), "n_static": n_static,
        "confined_points": float(events["n_confined_points"].sum() / max(int(events["n_localizations"].sum()), 1)),
        "dwell_steps": steps, "dwell_times_s": steps * tau_s,
        "transitions": trans, "dwell_events": dwell_events,
        "msd": None, "msd_error": None, "dwell_fit": None, "dwell_fit_error": None,
        "n_tracks_with_min_steps": int((track_lengths > min_steps).sum()),
    }
    if mode == "confined":
        try:
            run["msd"] = _confined_msd(table, radius_um, min_steps, index, tau_s)
        except (SLAError, confinement.ConfinementError) as exc:
            run["msd_error"] = ("MSD for confined tracks not possible: " + str(exc))
    else:
        if not len(steps):
            run["dwell_fit_error"] = (f"no dwell event of at least {min_steps} steps at a radius of "
                                      f"{radius_um * 1000:.0f} nm, so there is no survival curve to fit")
        else:
            run["dwell_fit"] = fit_dwell_times(steps, tau_s, min_steps)
    return run


def _confined_msd(table, radius_um, min_steps, index, tau_s) -> dict:
    """`confinedMSDMotion.m` over `ftestAnomalous.m`: the MSD of the confined segments (lags 1 to the minimum number
    of steps), pooled by `getMSD`, then the two fits and the F-test, with what the panel draws."""
    from . import confinement
    from .canonical import without_attrs
    from .legacy import sla
    from .msd import msd_per_track

    segments = confinement.confined_segments(table, radius_um=radius_um, min_steps=min_steps, dwells=index)
    per_track = msd_per_track(segments, max_lag=int(min_steps), pairing="row")
    curves = []
    for _, part in without_attrs(per_track).groupby("uid", sort=False):
        part = part.sort_values("lag_frames")
        curves.append(np.column_stack([part["lag_frames"].to_numpy(float) * tau_s, part["msd_um2"].to_numpy(float),
                                       part["n_pairs"].to_numpy(float), part["std_um2"].to_numpy(float)]))
    matrix = sla.pooled_msd(curves)
    verdict = sla.classify_motion(matrix)
    t, msd, std = matrix[:, 0], matrix[:, 1], matrix[:, 2]
    usable = np.isfinite(t) & np.isfinite(msd) & np.isfinite(std)
    lin = sla.fit_linear_weighted(t[usable], msd[usable], std[usable])
    ano = sla.fit_anomalous_weighted(t[usable], msd[usable], std[usable])
    divisor = float(np.sqrt(matrix.size))                           # numel(x): 2.0's N, four times the lags
    fine = np.arange(t[usable][0], t[usable][-1] + 1e-12, 0.01) if t[usable][-1] > t[usable][0] else t[usable]
    return {"t": t[usable], "msd": msd[usable], "err": std[usable] / divisor, "fine_t": fine,
            "linear_t": t[usable], "linear": lin["D_um2_per_s"] * 4.0 * t[usable] + lin["intercept_um2"],
            "anomalous": ano["intercept_um2"] + 4.0 * ano["D_alpha"] * fine ** ano["alpha"],
            "D": float(lin["D_um2_per_s"]), "alpha": float(ano["alpha"]), "D_alpha": float(ano["D_alpha"]),
            "p_value": float(verdict["p_value"]), "motion": verdict["motion"], "title": verdict["title"],
            "n_segments": int(segments.attrs["n_segments"])}


# --------------------------------------------------------------------------------------------- #
# The dwell-time fits (`@SMT/dwell_tracks.m`)
# --------------------------------------------------------------------------------------------- #

def _half_interval(pcov, popt, dfe: int) -> np.ndarray:
    """MATLAB's ``abs(diff(confint(fit))) / 2``: half a 95 % interval from the fit's covariance."""
    from scipy import stats

    if dfe < 1 or not np.all(np.isfinite(pcov)):
        return np.full(len(popt), np.nan)
    return stats.t.ppf(0.975, dfe) * np.sqrt(np.diag(pcov))


def fit_dwell_times(steps, tau_s: float, min_steps: int) -> dict:
    """2.0's survival curve of the dwell times and its two fits (`dwell_tracks.m:56-80`, 193-262).

    ``y = [1, 1 - cdf]`` of the dwell times histogrammed with a bin of one frame interval, at ``x = ((0:n-1) +
    (minSteps - 1)) * tau``; the single fit ``exp(-(t - t0) / (T1 - t0))`` with ``t0 = (minSteps - 1) tau`` and
    ``T1 >= t0``; the double fit ``a exp(-(t-t0)/(T1-t0)) + (1-a) exp(-(t-t0)/(T2-t0))``, the best of several starts
    by RMSE, the shorter time first. Both with half-95 % intervals from the covariance. A fit that cannot be made is
    reported in ``single_error`` / ``double_error``, never as a zero."""
    from scipy.optimize import curve_fit

    steps = np.asarray(steps, dtype=int)
    if not len(steps):
        raise LegacyViewError("no dwell event to build a survival curve from")
    first = int(steps.min())
    counts = np.bincount(steps - first)
    cdf = np.cumsum(counts) / counts.sum()
    y = np.concatenate([[1.0], 1.0 - cdf])
    x = (np.arange(len(y)) + (min_steps - 1)) * tau_s
    t0 = (min_steps - 1) * tau_s
    out = {"x": x, "y": y, "t0": float(t0), "n_events": int(len(steps)), "tau_s": float(tau_s),
           "single": None, "double": None, "single_error": None, "double_error": None}

    def single(t, T1):
        return np.exp(-(t - t0) / (T1 - t0))

    def double(t, a, T1, T2):
        return a * np.exp(-(t - t0) / (T1 - t0)) + (1.0 - a) * np.exp(-(t - t0) / (T2 - t0))

    eps = 1e-9
    mean_excess = max(float(np.mean(steps * tau_s)) - t0, tau_s)
    try:
        popt, pcov = curve_fit(single, x, y, p0=[t0 + mean_excess], bounds=([t0 + eps], [np.inf]), maxfev=20000)
        out["single"] = {"Tau_1": float(popt[0]), "sd": float(_half_interval(pcov, popt, len(x) - 1)[0])}
    except Exception as exc:                          # noqa: BLE001 - reported on the figure and the Console
        out["single_error"] = f"the one-component fit failed: {type(exc).__name__}: {exc}"
    rng = np.random.default_rng(0)
    best = None
    starts = [[a0, t0 + m * mean_excess, t0 + m2 * mean_excess]
              for a0 in (0.3, 0.5, 0.7) for m in (0.3, 0.6, 1.0) for m2 in (1.5, 3.0)]
    starts += [[0.5, t0 + rng.uniform(0.1, 2) * mean_excess, t0 + rng.uniform(1, 5) * mean_excess] for _ in range(6)]
    for p0 in starts:
        try:
            popt, pcov = curve_fit(double, x, y, p0=p0, bounds=([0.0, t0 + eps, t0 + eps], [1.0, np.inf, np.inf]),
                                   maxfev=20000)
        except Exception:                             # noqa: BLE001 - one start failing is part of a multistart
            continue
        rmse = float(np.sqrt(np.mean((double(x, *popt) - y) ** 2)))
        if best is None or rmse < best[0]:
            best = (rmse, popt, pcov)
    if best is None or len(x) < 4:
        out["double_error"] = ("the two-component fit found no solution (there is probably not enough data; try "
                               "less strict conditions)")
    else:
        _, popt, pcov = best
        a, T1, T2 = (float(v) for v in popt)
        sd = _half_interval(pcov, popt, len(x) - 3)
        sd = np.where(np.isfinite(sd), sd, 0.0)       # 2.0 sets an undefined interval to 0 (`dwell_tracks.m:239`)
        if T1 > T2:
            a, T1, T2 = 1.0 - a, T2, T1
            sd = np.array([sd[0], sd[2], sd[1]])
        out["double"] = {"a": a, "Tau_1": T1, "Tau_2": T2, "sd": [float(v) for v in sd]}
    fine = np.arange(0.0, float(x.max()) + 1e-12, 0.001)
    if out["single"]:
        out["single"]["curve"] = single(fine, out["single"]["Tau_1"])
        s = out["single"]
        out["single"]["upper"] = single(fine, s["Tau_1"] + s["sd"]) if np.isfinite(s["sd"]) else None
        out["single"]["lower"] = single(fine, max(s["Tau_1"] - s["sd"], t0 + eps)) if np.isfinite(s["sd"]) else None
    if out["double"]:
        d = out["double"]
        d["curve"] = double(fine, d["a"], d["Tau_1"], d["Tau_2"])
    out["fine"] = fine
    return out


# --------------------------------------------------------------------------------------------- #
# The table (`@SMT/fillSLATable_gui.m`)
# --------------------------------------------------------------------------------------------- #

def _num2str(value: float, digits: int) -> str:
    from .legacy_view import _num2str as num2str

    return num2str(value, digits)


def sla_table_column(run: dict, previous: list | None = None) -> list:
    """One condition's column of 2.0's table (`TABLE_ROWS` order). The dwell-time rows stay empty until the "dwell
    times" run has been made for the condition, as in 2.0, and a later "confined tracks" run leaves them as they were
    (``previous``: the column before, `fillSLATable_gui.m` writes only the rows it has numbers for)."""
    n = max(run["n_tracks"], 1)
    steps_free_confined = (f"{_num2str((1.0 - run['confined_points']) * 100, 2)} / "
                           f"{_num2str(run['confined_points'] * 100, 2)}")
    static = run["n_static"] / n
    free = run["n_free"] / n
    mixed = run["n_mixed_incl_static"] / n - static
    times = np.asarray(run["dwell_times_s"], dtype=float)
    column = [round(run["radius_um"] * 1000.0, 6),
              round(float(times.min()) * 1000.0, 3) if len(times) else "",
              steps_free_confined, round(static * 100, 1), round((1 - static) * 100, 1), round(free * 100, 1),
              round(mixed * 100, 1)] + (list(previous[7:13]) if previous else [""] * 6)
    fit = run.get("dwell_fit")
    if fit and len(times):
        sem = float(times.std(ddof=1) / np.sqrt(len(times))) if len(times) > 1 else float("nan")
        column[7] = f"{times.mean():.3f} ± {sem:.3f} s"
        if fit["single"]:
            column[8] = f"{_num2str(fit['single']['Tau_1'], 2)} ± {_num2str(fit['single']['sd'], 2)} s"
        if fit["double"]:
            d = fit["double"]
            column[9] = f"{_num2str(d['Tau_1'], 2)} ± {_num2str(d['sd'][1], 2)} s"
            column[10] = f"{_num2str(d['a'] * 100, 3)} ± {_num2str(100 * d['sd'][0], 3)} %"
            column[11] = f"{_num2str(d['Tau_2'], 2)} ± {_num2str(d['sd'][2], 2)} s"
            column[12] = f"{_num2str((1 - d['a']) * 100, 3)} ± {_num2str(100 * d['sd'][0], 3)} %"
    return column


def sla_table(columns: dict, dwell_times: dict) -> list:
    """The table as rows: ``[label, value per condition ...]`` then, once every condition has dwell times, 2.0's
    "significantly different?" block (`significance_rows` on the dwell times, `SMTracker.m:3191-3197`)."""
    names = list(columns)
    rows = [[label] + [columns[c][i] for c in names] for i, label in enumerate(TABLE_ROWS)]
    if len(names) > 1 and all(c in dwell_times and len(dwell_times[c]) for c in names):
        rows.extend(significance_rows({c: dwell_times[c] for c in names}))
    return rows


# --------------------------------------------------------------------------------------------- #
# The heat map (`confinedHeatmap`, `plotHeatmap.m` typeMotion 5) and the distributions (`barplotDistribution`)
# --------------------------------------------------------------------------------------------- #

def interp_cubic(v: np.ndarray, refinements: int = 5) -> np.ndarray:
    """``interp2(v, k, 'cubic')``: cubic convolution (Keys, a = -0.5) on a uniform grid, each pass inserting a node
    between every pair, so n nodes become ``(n - 1) 2^k + 1``. The border is extrapolated linearly."""
    v = np.asarray(v, dtype=float)
    factor = 2 ** int(refinements)

    def along(a: np.ndarray, axis: int) -> np.ndarray:
        a = np.moveaxis(a, axis, 0)
        n = a.shape[0]
        padded = np.concatenate([2 * a[:1] - a[1:2], a, 2 * a[-1:] - a[-2:-1]], axis=0)
        s = np.arange(0, (n - 1) * factor + 1) / factor
        i = np.minimum(np.floor(s).astype(int), n - 2)
        f = s - i
        # Keys weights for the four nodes i-1, i, i+1, i+2 (padded index shifted by one)
        w = np.stack([-0.5 * f ** 3 + f ** 2 - 0.5 * f, 1.5 * f ** 3 - 2.5 * f ** 2 + 1,
                      -1.5 * f ** 3 + 2 * f ** 2 + 0.5 * f, 0.5 * f ** 3 - 0.5 * f ** 2])
        last = padded.shape[0] - 1
        out = sum(w[k].reshape((-1,) + (1,) * (a.ndim - 1)) * padded[np.minimum(i + k, last)] for k in range(4))
        return np.moveaxis(out, 0, axis)

    return along(along(v, 0), 1)


def distribution_hist(values: np.ndarray, n_bins: int) -> dict:
    """``histogram(ax, v, n, 'normalization', 'probability')``: ``n_bins`` equal bins over the data's range, each
    bar the share of the points in it."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        raise LegacyViewError("no point to build a distribution from")
    counts, edges = np.histogram(values, bins=max(int(n_bins), 1))
    return {"edges": edges, "probability": counts / counts.sum()}


def heatmap(run: dict, show=("free", "mixed", "dwelling"), *, n_bins: int = 20) -> dict:
    """What `confinedHeatmap` draws for the ticked sets: ``{"x", "y", "image", "vmax", "title", "points", ...}``
    with the image the up-scaled location density (jet, from 0 to its maximum) over x in [-0.1, 1.1], y in
    [-0.6, 0.6], plus the two distributions (`barplotDistribution`, source 1) of the same points."""
    keys = {"free": "free_xy", "mixed": "mixed_xy", "dwelling": "segments_xy"}
    words = {"free": " +free tracks", "mixed": " +mixed-behaviour tracks", "dwelling": " +dwelling sub-tracks"}
    shown = [k for k in ("free", "mixed", "dwelling") if k in show]
    if not shown:
        raise LegacyViewError("tick at least one of free, mixed behaviour and dwelling to draw the heat map")
    points = np.vstack([[np.nan, np.nan]] + [run[keys[k]] for k in shown])
    finite = points[np.isfinite(points[:, 0])]
    if not len(finite):
        raise LegacyViewError("none of the ticked sets holds a localisation inside the model cell")
    ex = np.linspace(-0.1, 1.1, 20)
    ey = np.linspace(-0.6, 0.6, 20)
    counts, _, _ = np.histogram2d(finite[:, 0], finite[:, 1], bins=[ex, ey])
    v = np.zeros((20, 20))
    v[:19, :19] = counts                              # hist3's 20th bin is open-ended (nothing lies there)
    density = interp_cubic(v, 5) / len(points)        # `/size(nTracks,1)`: the NaN rows count (plotHeatmap.m:17)
    cx = np.concatenate([(ex[:-1] + ex[1:]) / 2, [ex[-1] + (ex[1] - ex[0]) / 2]])
    cy = np.concatenate([(ey[:-1] + ey[1:]) / 2, [ey[-1] + (ey[1] - ey[0]) / 2]])
    return {"x": np.linspace(cx[0], cx[-1], density.shape[0]), "y": np.linspace(cy[0], cy[-1], density.shape[1]),
            "image": density.T, "vmax": float(density.max()), "points": finite,
            "title": "".join(words[k] for k in shown), "n_points_in_divisor": int(len(points)),
            "x_distribution": distribution_hist(finite[:, 0], n_bins),
            "y_distribution": distribution_hist(finite[:, 1], n_bins)}


# --------------------------------------------------------------------------------------------- #
# "All Movies" (`@SMT/confTracksperCell.m`)
# --------------------------------------------------------------------------------------------- #

def per_movie_tracks(project, condition: str, run: dict) -> list:
    """One entry per movie of the condition for 2.0's "All Movies" box: the tracks of every cell in that movie in
    micrometres (all in blue, the mixed-behaviour tracks in green, their dwelling segments red), and the cell
    outlines. Detected at the run's radius and minimum steps; the figures are drawn by the tab."""
    from . import confinement
    from .canonical import without_attrs

    table = project.condition(condition).sort_values(["uid", "frame"], kind="stable")
    index = confinement.dwell_index(table, radius_um=run["radius_um"], min_steps=run["min_steps"])
    movies = []
    for movie, part in without_attrs(table).groupby("movie", sort=False, observed=True):
        all_xy, mixed_xy, seg_xy = [], [], []
        for uid, track in part.groupby("uid", sort=False, observed=True):
            xy = track[["x_um", "y_um"]].to_numpy(dtype=float)
            taus, idx = index[uid]
            all_xy.append(xy)
            if taus:
                mixed_xy.append(xy)
                seg_xy.extend(xy[np.asarray(p, dtype=int)] for p in idx)

        def join(pieces):
            return np.concatenate([np.vstack([p, [np.nan, np.nan]]) for p in pieces]) if pieces else np.empty((0, 2))

        um = float(project.params.um_per_pixel)
        outlines = [np.asarray(m.polygon, dtype=float) * um for m in (project.meshes.get(str(movie)) or [])]
        movies.append({"movie": str(movie), "all": join(all_xy), "mixed": join(mixed_xy), "segments": join(seg_xy),
                       "n_tracks": len(all_xy), "outlines": outlines})
    if not movies:
        raise LegacyViewError(f"{condition}: no movie holds a track")
    return movies
