"""The MSD view's numbers and sentences, with no Qt anywhere in it.

`ui/workspace/msd_chart.py` is the Qt shell; this is what it draws from. They are split for
the Thin Qt rule (`tests/test_ui_thin_qt.py`): a wrong number must fail in a headless test,
not look subtly wrong on screen, and a process-pool worker must import this without Qt.

WHAT THE SECTION IS. v2's MSD tab, kept and improved, rebuilt on 16 Oct 2026 in the Apparent
Diffusion section's pattern (asked: "msd is still in a weird shape with lots of text"; the
section's entry is at the end of this module). `compute_msd_view` gives one condition's
ensemble MSD curve, the fit over its EARLY lags only, the interval over independent units,
the intercept's sigma beside the tracker's own, the exponent alpha, and -- on request -- what
v2 printed for the same data, so the difference is measured rather than asserted;
`msd_condition` adds the section's checks. They return plain lists, floats and strings, the
only things the Qt layer reads; the pages carry numbers in tables and captions, and the
explanations once per page (`PAGE_NOTES`).

THE RULES IT KEEPS, each from CLAUDE.md:

- AN ENSEMBLE SLOPE IS NOT A POPULATION'S D. A mixture's MSD is dominated by its fastest
  state; the populations and their fractions are the SQD section's. The Curves page says so.
- ONLY THE EARLY LAGS ARE FITTED (default 4), weighted by pairs; the tail is drawn, hollow,
  and never fitted. Asking for more lags than the curve has is refused by the spine.
- THE INTERCEPT IS NOT SIGMA. For a free molecule imaged continuously it is
  2*ndim*(sigma^2 - D*tau/3): motion blur pulls it DOWN, the cell wall pushes it up, and it
  grows with the fit window. It was called an upper bound until 16 Oct 2026; that held only
  while the wall outweighed blur (the MSD study: at D 0.56 um^2/s and 20 ms blur is about
  13 times 4*sigma^2 at 17 nm). The tracker's own sigma sits beside it with its provenance.
- ALPHA BELOW 1 IS NOT ANOMALOUS DIFFUSION here: the localisation offset, the wall and
  survivorship all pull it down with no anomalous motion present.
- THE INTERVAL IS OVER INDEPENDENT UNITS, never the regression's: the spread over replicate
  DAYS (mean +- SD), or 200 MOVIE or CELL resamples on a single session only -- with two or
  more days movies and cells are not independent and their interval comes out too narrow
  (Report A, the same gate the SQD section keeps). Movies (19 Oct 2026): cells in one movie
  share its focus, illumination and field, so the movie is the coarser single-session unit.
- DIRECTION: the camera plane, or the cell's own axes (v2's x / y were these). One axis is
  one dimension, D = slope / 2; v2 divided by 4 there too, which halved it.

NO STATISTIC BETWEEN CONDITIONS: each condition is computed on its own and drawn beside the
others. A condition that fails is a card that says why, never an absence.

THE REPLICATE DAYS ON THE CURVE (17 Oct 2026, asked: "error bars on msd curve when u use day? clicking on one
showing all three of the same condition?"): every condition with two or more days carries each day's own ensemble
curve and fit (`_day_curves`, the one per-day computation; the D spread over days is read from it). Under
Uncertainty = Days each pooled point's bar is the SD of the days' MSD at that lag, centred on the pooled point as
the table's D carries the SD of the days' D; otherwise the bar stays the SEM over tracks, said as indicative (tracks
within a day are not independent). `chart["bars"]` / `chart["bar_words"]`: the card, the overview and the export
draw the same bars. Under Uncertainty = Movies or Cells (one session) each bar is the 95 % interval
of the resampled curves at that lag (`msd.bootstrap_msd` keeps each resample's curve). A click on a curve shows that condition's day curves.

MSD PER CELL (17 Oct 2026, v2's "Save MSD in csv..."): each cell's ensemble MSD, the same
`msd.ensemble_msd` over that cell's tracks at the Run's lags, weighting and direction
(`result["per_cell"]`, `per_cell_csv`). v2 saved one selected cell at a time (delays, mean,
SD, n); here every cell of every condition in one file, with its day and movie.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = [
    "DEFAULTS", "DIRECTIONS", "DIRECTION_LABELS", "UNCERTAINTY_LABELS", "UNCERTAINTY_MODES",
    "PAGE_NOTES", "SECTION_CURVE", "SECTION_DEFAULTS", "SECTION_DIRECTIONS", "SECTION_DIRECTION_LABELS",
    "SECTION_HELP", "SECTION_LAGS", "SECTION_UNCERTAINTY", "SECTION_UNCERTAINTY_LABELS", "SECTION_WEIGHTING",
    "SECTION_WEIGHTING_LABELS", "compute_msd_view", "curve_hover", "curves_table", "histogram_hover",
    "LENGTH_MIN_TRACKS", "length_check", "msd_condition", "overlay_hover", "overlay_range", "section_config", "short_caption",
    "v2_rows", "window_payload",
]

#: The camera plane, or the cell's own axes (`msd.MSD_DIRECTIONS`).
DIRECTIONS = ("xy", "along", "across")
DIRECTION_LABELS = ("Camera plane", "Along the cell", "Across the cell")
#: What each Uncertainty choice draws (the SQD section's, same meaning).
UNCERTAINTY_MODES = ("none", "day", "cell", "movie")
UNCERTAINTY_LABELS = ("None", "Replicates", "Cells", "Movies")
#: The single-session choices: a percentile interval from resampling this unit.
RESAMPLED_UNITS = ("movie", "cell")
#: A resampled unit, as a sentence counts it.
UNIT_WORDS = {"movie": "movies", "cell": "cells"}

#: The dialog's starting values. The fit window is the measured one (4 lags; CLAUDE.md, the
#: msd module), the curve runs to 10, the v2 comparison uses v2's own minimum of 9.
DEFAULTS = {"n_lags": 4, "max_lag": 10, "weighted": True, "direction": "xy",
            "uncertainty": "day", "legacy": False, "min_length": 9}

#: Movie or cell resamples for the single-session interval.
N_BOOT = 200
#: Per-track curves drawn pale behind the ensemble -- enough to show the spread, few
#: enough to stay legible and cheap to paint.
TRACK_SAMPLE = 120
#: v2's histogram bin, 0.02 um (`plotMSDTracks.m`), here in nm.
HIST_BIN_NM = 20.0
#: The longest curve the dialog accepts; past it nearly every lag rests on a handful of tracks.
MAX_LAG_LIMIT = 50

_SIGMA_SOURCE = {
    "tracker_reported": "reported by the tracker (a fit's formal uncertainty, so a lower "
                        "bound)",
    "theoretical": "derived from the optics, not measured",
    "configured": "set by hand",
    "measured": "measured",
}


# --------------------------------------------------------------------------- #
# Compute: one condition, on the process pool
# --------------------------------------------------------------------------- #


def compute_msd_view(project, condition: str, *, n_lags: int, max_lag: int, weighted: bool,
                     direction: str, uncertainty: str, legacy: bool,
                     min_length: int = DEFAULTS["min_length"]) -> dict:
    """One condition's MSD: the curve, the early-lag fit, its interval and the words.

    Picklable and run on the process pool. An expected refusal comes back as
    ``{"condition": ..., "error": "<Type>: <message>"}`` naming what refused; anything
    else raises.
    """
    from .canonical import without_attrs
    from .msd import MSDError, ensemble_msd, fit_msd, msd_per_track
    from .observables import ObservableError

    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    if uncertainty not in UNCERTAINTY_MODES:
        raise ValueError(f"uncertainty must be one of {UNCERTAINTY_MODES}, got {uncertainty!r}")
    settings = {"n_lags": int(n_lags), "max_lag": int(max_lag), "weighted": bool(weighted),
                "direction": direction, "uncertainty": uncertainty, "legacy": bool(legacy),
                "min_length": int(min_length)}
    try:
        canonical = project.condition(condition)
        projection = None
        if direction == "xy":
            table = canonical
        else:
            table = project.projected(condition)
            projection = {"kept": int(len(table)), "total": int(len(canonical)),
                          "tracks_kept": int(table["uid"].nunique()),
                          "tracks_total": int(canonical["uid"].nunique())}
        per_track = msd_per_track(table, max_lag=int(max_lag), direction=direction)
        ensemble = ensemble_msd(per_track, weighted=weighted)
        fit = fit_msd(ensemble, n_lags=int(n_lags), weighted=weighted)
    except (MSDError, ObservableError, ValueError) as exc:
        return {"condition": condition, "settings": settings,
                "error": f"{type(exc).__name__}: {exc}"}
    try:
        alpha = fit_msd(ensemble, n_lags=int(n_lags), weighted=weighted,
                        anomalous=True).alpha
    except MSDError:
        alpha = None

    sigma = fit.localization_sigma_um
    result = {
        "condition": condition,
        "settings": settings,
        "projection": projection,
        "n_tracks": int(table["uid"].nunique()),
        "n_localisations": int(len(table)),
        "fit": {"D": float(fit.D_um2_per_s), "intercept": float(fit.intercept_um2),
                "sigma_nm": None if sigma is None else float(sigma * 1000.0),
                "alpha": None if alpha is None else float(alpha),
                "r2": float(fit.r_squared), "n_lags": int(fit.n_lags_used),
                "ndim": int(fit.ndim)},
        "tau_s": float(ensemble["lag_time_s"].min()),
        "tracker_sigma": _tracker_sigma(project),
    }
    days = _day_curves(per_track, settings)
    result["days"] = days
    result["interval"] = _interval(table, per_track, settings, uncertainty, days)
    curve = without_attrs(ensemble).sort_values("lag_frames")
    result["curve"] = [(float(r.lag_time_s), float(r.msd_um2),
                        float(r.sem_um2) if np.isfinite(r.sem_um2) else 0.0, int(r.n_tracks))
                       for r in curve.itertuples()]
    result["per_cell"] = _per_cell(table, per_track, weighted)
    result["legacy"] = _legacy(table, per_track, settings) if legacy else None
    result["chart"] = _chart_payload(result, without_attrs(per_track))
    result["histogram"] = (None if result["legacy"] is None
                           else result["legacy"].pop("histogram", None))
    return result


def _per_cell(table, per_track, weighted: bool) -> list:
    """Each cell's ensemble MSD (`msd.ensemble_msd` over the cell's tracks, the Run's weighting): rows of
    ``[day, movie, cell, delay_s, mean_um2, std_um2, n_tracks, n_pairs]``, by cell then lag. Tracks outside every
    outline (no cell) are left out."""
    from .canonical import without_attrs
    from .msd import MSDError, ensemble_msd

    where = without_attrs(table)[["uid", "day", "movie", "cell"]].drop_duplicates("uid")
    tracks = without_attrs(per_track).drop(columns=[c for c in ("day", "movie", "cell") if c in per_track.columns])
    tracks = tracks.merge(where, on="uid", how="left")
    tracks = tracks[tracks["cell"].notna()]
    rows = []
    for (day, movie, cell), part in tracks.groupby(["day", "movie", "cell"], sort=True, observed=True):
        try:
            curve = ensemble_msd(part, weighted=weighted)
        except MSDError:
            continue
        for r in curve.sort_values("lag_frames").itertuples():
            rows.append([_plain(day), str(movie), str(cell), float(r.lag_time_s), float(r.msd_um2),
                         float(r.std_um2), int(r.n_tracks), int(r.n_pairs)])
    return rows


def _plain(value):
    """A day as it reads in a file: an integer day as an integer."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    return int(f) if f.is_integer() else f


#: The columns of `per_cell_csv`, v2's four (delays, mean_msd, std_msd, n_msd) with where each cell is. The replicate
#: column is headed "replicate" (ruled 4 Nov 2026, D9: a user reads "replicate", never "day"; a file a user opens is
#: read as a user reads): the CSV only -- the data's own column stays ``day`` by the 24 Oct 2026 ruling.
PER_CELL_COLUMNS = ("condition", "replicate", "movie", "cell", "delay_s", "mean_msd_um2", "std_msd_um2", "n_tracks",
                    "n_pairs")


def per_cell_csv(results: dict, conditions: list) -> str:
    """Every cell's MSD in one CSV (comma-separated, "." decimals): `PER_CELL_COLUMNS`, a row per cell and lag, the
    conditions in ``conditions`` order. Empty of rows when no condition has cells."""
    import csv
    import io

    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(PER_CELL_COLUMNS)
    for c in conditions:
        res = results.get(c) or {}
        for row in res.get("per_cell") or []:
            day, movie, cell, delay, mean, std, n, pairs = row
            writer.writerow([c, day, movie, cell, f"{delay:.6g}", f"{mean:.6g}", f"{std:.6g}", n, pairs])
    return out.getvalue()


def has_per_cell(results: dict) -> bool:
    """Whether any condition carries a cell's MSD (the button's state)."""
    return any((r or {}).get("per_cell") for r in results.values())


def _tracker_sigma(project) -> dict:
    from .spatial_map_view import sigma_stamp

    stamp = sigma_stamp(project)
    return {"nm": stamp["sigma_nm"], "provenance": stamp["sigma_provenance"]}


def _day_curves(per_track, settings: dict) -> list:
    """Each replicate day's own ensemble MSD and early-lag fit, the Run's settings (`msd.ensemble_msd`, `fit_msd`):
    ``[{"name", "curve": [(t, msd, n_tracks)], "D", "intercept", "line", "error"}]``; ``[]`` with fewer than two
    days. THE ONE per-day computation: the day spread of D (`_interval`) and the day bars read it."""
    from .canonical import without_attrs
    from .msd import MSDError, ensemble_msd, fit_msd

    if "day" not in per_track.columns:
        return []
    keys = sorted(per_track["day"].dropna().unique())
    if len(keys) < 2:
        return []
    out = []
    for day in keys:
        rows = per_track[per_track["day"] == day]
        entry = {"name": _day_name(day), "curve": [], "D": None, "intercept": None, "line": None, "error": None}
        try:
            ens = ensemble_msd(rows, weighted=settings["weighted"])
            entry["curve"] = [(float(r.lag_time_s), float(r.msd_um2), int(r.n_tracks))
                              for r in without_attrs(ens).sort_values("lag_frames").itertuples()]
            fit = fit_msd(ens, n_lags=settings["n_lags"], weighted=settings["weighted"],
                          ndim=int(per_track.attrs["ndim"]))
        except MSDError as exc:
            entry["error"] = str(exc)
            out.append(entry)
            continue
        t_fit = entry["curve"][min(int(fit.n_lags_used), len(entry["curve"])) - 1][0]
        k = float(fit.D_um2_per_s) * 2 * int(fit.ndim)
        entry.update(D=float(fit.D_um2_per_s), intercept=float(fit.intercept_um2),
                     line=[(0.0, float(fit.intercept_um2)), (t_fit, float(fit.intercept_um2) + k * t_fit)])
        out.append(entry)
    return out


def _interval(table, per_track, settings: dict, mode: str, day_curves=None) -> dict | None:
    """The spread over independent units, per the Uncertainty setting."""
    from .msd import MSDError, bootstrap_msd

    if mode == "none":
        return None
    days = sorted(table["day"].dropna().unique()) if "day" in table.columns else []
    if mode == "day":
        if len(days) < 2:
            return {"mode": "day", "note": (
                f"{len(days)} replicate: there is no spread between replicates to show. Choose Movies "
                f"or Cells for an interval over this session's movies or cells.")}
        if day_curves is None:
            day_curves = _day_curves(per_track, settings)
        values = [(d["name"], d["D"]) for d in day_curves if d["D"] is not None]
        failed = [(d["name"], d["error"]) for d in day_curves if d["D"] is None]
        out = {"mode": "day", "values": values, "failed": failed}
        if len(values) >= 2:
            d = np.array([v for _k, v in values])
            out.update(mean=float(d.mean()), sd=float(d.std(ddof=1)), n=len(values))
        return out
    # movies or cells, one session
    unit, words = mode, UNIT_WORDS[mode]
    if len(days) >= 2:
        return {"mode": unit, "withheld": (
            f"No {unit} interval: this condition holds {len(days)} replicates, {words} "
            f"within a replicate are not independent, so a {unit} interval would be too "
            f"narrow. Choose Replicates.")}
    try:
        boot = bootstrap_msd(table, unit=unit, n_boot=N_BOOT, n_lags=settings["n_lags"],
                             max_lag=settings["max_lag"], weighted=settings["weighted"],
                             direction=settings["direction"], per_track=per_track)
    except MSDError as exc:
        return {"mode": unit, "withheld": f"No {unit} interval: {exc}"}
    lo, hi = (float(v) for v in boot["D_um2_per_s"].quantile([0.025, 0.975]))
    # each lag's 95 % interval over the cell resamples (the curve's bars), where most resamples reach that lag
    curves = np.asarray(boot.attrs["curves"], dtype=float)
    reached = np.isfinite(curves).mean(axis=0) >= 0.95
    band = [(float(t), float(np.nanpercentile(curves[:, i], 2.5)), float(np.nanpercentile(curves[:, i], 97.5)))
            for i, t in enumerate(boot.attrs["curve_lag_s"]) if reached[i]]
    return {"mode": unit, "lo": lo, "hi": hi, "n_units": int(boot.attrs["n_units"]),
            "n_boot": int(len(boot)), "caveat": boot.attrs.get("caveat", ""), "curve_band": band}


def _day_name(day) -> str:
    try:
        return f"replicate {int(day)}"
    except (TypeError, ValueError):
        return f"replicate {day}"


def _legacy(table, per_track, settings: dict) -> dict:
    """What v2's MSD tab printed for this data, and the pieces of the difference."""
    from .msd import (
        MSDError,
        ensemble_msd,
        fit_msd,
        legacy_ensemble_window,
        msd_per_track,
        per_track_fits,
    )

    min_length = settings["min_length"]
    out: dict = {"min_length": min_length}
    lengths = table.groupby("uid", observed=True).size()
    long_uids = lengths.index[lengths >= min_length]
    out["tracks_long"] = int(len(long_uids))
    out["tracks_total"] = int(len(lengths))
    # v2's ensemble: only tracks of >= min_length, each CUT to lags 1..min_length-1
    # ("specific tau", calcMSD_gui.m:102-104), the pair-weighted mean (getMSD) fitted
    # UNWEIGHTED over the "Linear Fit" lags -- which at the default 9 and 9 runs past the
    # cut curve, so v2's try/catch fits lags 1-8 (getFitMSD_gui.m:8-14), every time.
    window = legacy_ensemble_window(min_length)
    try:
        sub = table[table["uid"].isin(long_uids)]
        long_curves = msd_per_track(sub, max_lag=max(window["curve_lags"],
                                                     settings["n_lags"]),
                                    direction=settings["direction"])
        ndim = int(long_curves.attrs["ndim"])
        curve = ensemble_msd(
            long_curves[long_curves["lag_frames"] <= window["curve_lags"]], weighted=True)
        used = min(window["n_lags"], len(curve))
        v2 = fit_msd(curve, n_lags=used, weighted=False, ndim=ndim)
        out["ensemble"] = {"D": float(v2.D_um2_per_s), "intercept": float(v2.intercept_um2),
                           "n_lags": int(used), "fell_back": bool(window["fell_back"]),
                           "linear_fit_box": int(window["linear_fit_box"]),
                           # v2 divided every slope by 4, one axis too (fillMSDtable.m:50)
                           "D_printed": float(v2.D_um2_per_s * v2.ndim / 2.0)}
        same_window = fit_msd(ensemble_msd(long_curves, weighted=settings["weighted"]),
                              n_lags=settings["n_lags"], weighted=settings["weighted"],
                              ndim=ndim)
        out["track_set_D"] = float(same_window.D_um2_per_s)
    except MSDError as exc:
        out["ensemble_error"] = str(exc)
    try:
        # "specific tau" (v2's default): each track cut to lags 1..min_length-1, the last
        # dropped, so lags 1-7 at 9 (calcMSD_gui.m:102-104, linWfit.m:6).
        fits = per_track_fits(table, min_length=min_length, specific_tau=True,
                              direction=settings["direction"])
    except MSDError as exc:
        out["per_track_error"] = str(exc)
        return out
    b = fits["intercept_um2"].to_numpy(dtype=float)
    ticked = fits["ticked"].to_numpy(dtype=bool)
    # v2's own formula, sqrt(|b| / 4), on one axis too: this block is what v2 PRINTED.
    folded = np.sqrt(np.abs(b) / 4.0) * 1000.0
    out["per_track"] = {
        "n_fitted": int(len(fits)),
        "n_ticked": int(ticked.sum()),
        "panel_nm": float(folded[ticked].mean()) if ticked.any() else None,
        "all_nm": float(folded.mean()),
        "negative_share": float(np.mean(b < 0)),
        "signs_kept_nm": (float(np.sqrt(b.mean() / 4.0) * 1000.0)
                          if b.mean() > 0 else None),
        "one_axis": int(fits.attrs["ndim"]) == 1,
        "lags_fitted": fits.attrs["lags_fitted"],
        "median_D": float(np.median(fits["D_um2_per_s"])),
    }
    out["histogram"] = _histogram(np.sign(b) * folded)
    return out


def _histogram(signed_nm: np.ndarray) -> dict:
    """v2's 20 nm bins over the per-track sigma, the negative intercepts kept on the left
    (sign times the root of their magnitude) where v2's abs() folded them onto the right."""
    values = signed_nm[np.isfinite(signed_nm)]
    lo_v, hi_v = np.percentile(values, [1, 99]) if values.size else (0.0, 0.0)
    lo = max(-300.0, math.floor(lo_v / HIST_BIN_NM) * HIST_BIN_NM)
    hi = min(400.0, math.ceil(hi_v / HIST_BIN_NM) * HIST_BIN_NM)
    if hi <= lo:
        hi = lo + HIST_BIN_NM
    edges = np.arange(lo, hi + HIST_BIN_NM / 2, HIST_BIN_NM)
    clipped = np.clip(values, lo, hi - 1e-9)
    counts, _ = np.histogram(clipped, bins=edges)
    beyond = int(np.sum((values < lo) | (values >= hi)))
    bins = [(float(a), float(b), int(c), bool(b <= 0)) for a, b, c in
            zip(edges[:-1], edges[1:], counts)]
    return {"bins": bins, "n": int(values.size), "beyond": beyond,
            "y_max": int(counts.max()) if counts.size else 0}


def _chart_payload(result: dict, per_track) -> dict:
    """Everything `MSDChart` draws, as plain Python, in um^2 and seconds."""
    curve = result["curve"]
    fit = result["fit"]
    n_fit = min(fit["n_lags"], len(curve))
    slope = fit["D"] * 2 * fit["ndim"]
    t_fit = curve[n_fit - 1][0]
    t_max = curve[-1][0]
    line = [(0.0, fit["intercept"]), (t_fit, fit["intercept"] + slope * t_fit)]
    extension = ([(t_fit, line[1][1]), (t_max, fit["intercept"] + slope * t_max)]
                 if t_max > t_fit else [])
    legacy_fit = None
    ens = (result.get("legacy") or {}).get("ensemble")
    if ens is not None:
        k = ens["D"] * 2 * fit["ndim"]
        legacy_fit = [(0.0, ens["intercept"]), (t_max, ens["intercept"] + k * t_max)]
    # A deterministic sample of per-track curves, evenly through the sorted track list.
    tracks = []
    by_uid = {uid: part for uid, part in per_track.groupby("uid", sort=True, observed=True)
              if len(part) >= 3}
    uids = sorted(by_uid)
    if uids:
        step = max(1, len(uids) // TRACK_SAMPLE)
        for uid in uids[::step][:TRACK_SAMPLE]:
            part = by_uid[uid].sort_values("lag_frames")
            tracks.append([(float(t), float(m)) for t, m in
                           zip(part["lag_time_s"], part["msd_um2"])])
    bars, bar_words = _bars(result)
    days = [{"name": d["name"], "curve": [(t, m) for t, m, _n in d["curve"]], "line": d["line"], "D": d["D"]}
            for d in result.get("days") or [] if d["curve"]]
    top = max(hi for _t, _lo, hi in bars) if bars else max(m for _t, m, _s, _n in curve)
    top = max(top, *(y for _t, y in line + extension),
              *(m for d in days for _t, m in d["curve"]))
    bottom = min(0.0, min(lo for _t, lo, _hi in bars) if bars else 0.0, fit["intercept"])
    return {"curve": curve, "n_fit": int(n_fit), "fit": line, "extension": extension,
            "legacy_fit": legacy_fit, "tracks": tracks, "t_max": float(t_max),
            "y_range": (float(bottom), float(top)), "ndim": fit["ndim"],
            "direction": result["settings"]["direction"],
            "bars": bars, "bar_words": bar_words, "days": days}


def _bars(result: dict) -> tuple:
    """Each pooled point's bar, ``[(t, low, high)]``, and what it is: under Uncertainty = Days with two or more
    days, the SD of the days' MSD at that lag around the pooled point (a lag fewer than two days reach has none);
    otherwise the SEM over tracks, indicative only."""
    curve = result["curve"]
    days = [d for d in result.get("days") or [] if d["curve"]]
    interval = result.get("interval") or {}
    unit = result["settings"]["uncertainty"]
    if unit in RESAMPLED_UNITS and interval.get("curve_band"):
        at = {round(t, 9): (lo, hi) for t, lo, hi in interval["curve_band"]}
        bars = [(t, *at[round(t, 9)]) for t, _m, _s, _n in curve if round(t, 9) in at]
        return bars, f"95 % interval over {interval['n_units']} {UNIT_WORDS[unit]} (resampled)"
    if result["settings"]["uncertainty"] == "day" and len(days) >= 2:
        bars = []
        for t, m, _s, _n in curve:
            values = [dm for d in days for dt, dm, _dn in d["curve"] if abs(dt - t) <= 1e-9 * max(1.0, t)]
            if len(values) >= 2:
                sd = float(np.std(values, ddof=1))
                bars.append((t, m - sd, m + sd))
        return bars, f"± SD over {len(days)} replicates"
    return ([(t, m - s, m + s) for t, m, s, _n in curve if s > 0],
            "± SEM over tracks (indicative: tracks within a replicate are not independent)")


# --------------------------------------------------------------------------- #
# The words, as pure functions of the result (checked without Qt)
# --------------------------------------------------------------------------- #


def _d(value: float) -> str:
    return f"{value:.3g}"


# --------------------------------------------------------------------------- #
# Hover text
# --------------------------------------------------------------------------- #


def curve_hover(chart: dict, t: float) -> str | None:
    """The nearest lag of one condition's curve, at ``t`` seconds."""
    curve = chart.get("curve") or []
    if not curve:
        return None
    i = min(range(len(curve)), key=lambda k: abs(curve[k][0] - t))
    at, msd, sem, n = curve[i]
    state = "fitted" if i < chart["n_fit"] else "drawn, not fitted"
    bar = next(((lo, hi) for bt, lo, hi in chart.get("bars") or [] if abs(bt - at) <= 1e-9 * max(1.0, at)), None)
    words = chart.get("bar_words") or "± SEM over tracks"
    spread = (f"{words.split(' (')[0]}: {(bar[1] - bar[0]) / 2:.2g} µm²" if bar else "no bar at this lag")
    text = f"lag {i + 1} · {at * 1000:.0f} ms ({state})\nMSD {msd:.4g} µm², {spread}\n{n:,} tracks"
    if chart.get("days"):
        text += "\nclick: the replicates"
    return text


def days_hover(chart: dict, t: float) -> str | None:
    """Each replicate day's MSD at the lag nearest ``t`` (the click view of one condition)."""
    rows = []
    for d in chart.get("days") or []:
        if not d["curve"]:
            continue
        i = min(range(len(d["curve"])), key=lambda k: abs(d["curve"][k][0] - t))
        rows.append((d["curve"][i][0], f"{d['name']}: {d['curve'][i][1]:.4g} µm²"
                     + (f" (D {d['D']:.3g})" if d.get("D") is not None else "")))
    if not rows:
        return None
    return f"{rows[0][0] * 1000:.0f} ms\n" + "\n".join(text for _t, text in rows) + "\nclick: back"


def overlay_hover(entries: list, t: float) -> str | None:
    """Every drawn condition at the lag nearest ``t``: ``entries`` is
    ``[(name, chart), ...]``."""
    rows = []
    for name, chart in entries:
        curve = chart.get("curve") or []
        if not curve:
            continue
        i = min(range(len(curve)), key=lambda k: abs(curve[k][0] - t))
        rows.append((curve[i][0], f"{name}: {curve[i][1]:.4g} µm²"))
    if not rows:
        return None
    return f"{rows[0][0] * 1000:.0f} ms\n" + "\n".join(text for _t, text in rows)


def overlay_range(charts: list) -> tuple | None:
    """The shared axes for several conditions' curves: ``(t_max, y_lo, y_hi)``."""
    drawn = [c for c in charts if c and c.get("curve")]
    if not drawn:
        return None
    return (max(c["t_max"] for c in drawn), min(c["y_range"][0] for c in drawn),
            max(c["y_range"][1] for c in drawn))


def histogram_hover(histogram: dict, value_nm: float) -> str | None:
    for lo, hi, count, negative in histogram["bins"]:
        if lo <= value_nm < hi:
            sign = " (negative intercepts: SMTracker 2 folds these onto the right)" if negative else ""
            return f"{lo:.0f} to {hi:.0f} nm\n{count:,} tracks{sign}"
    return None




# =====================================================================================================================
# THE MSD SECTION, REBUILT (16 Oct 2026)
# =====================================================================================================================
#
# Asked: "msd is still in a weird shape with lots of text". The Apparent Diffusion section's pattern: ONE computation
# per condition (`msd_condition`, a pool job each), four pages behind a strip that draw from it -- Curves (every
# condition's ensemble on shared axes and one table of the numbers), Per condition (each condition's curve, nothing
# under it but its numbers in the caption), Checks (what the fit window and the track length do to D), v2 (what v2
# printed for the same data, as a table, with its sigma histogram) -- and a Run dialog of fixed choices. The prose the
# cards used to carry is cut, or moved into one "How to read" per page. Every number still comes from
# `compute_msd_view`; this adds the checks and the page helpers.

#: The Run's choices.
SECTION_LAGS = (3, 4, 5, 6)
SECTION_CURVE = (10, 20)
SECTION_WEIGHTING = ("pairs", "tracks")
SECTION_WEIGHTING_LABELS = ("By pairs", "Each track once")
SECTION_DIRECTIONS = DIRECTIONS
SECTION_DIRECTION_LABELS = DIRECTION_LABELS
SECTION_UNCERTAINTY = ("day", "movie", "cell", "none")
SECTION_UNCERTAINTY_LABELS = ("Replicates", "Movies", "Cells", "None")
#: The defaults: the measured fit window (4 lags), the curve to lag 10, pairs (as before), the camera plane, Days.
SECTION_DEFAULTS = {"n_lags": 4, "max_lag": 10, "weighting": "pairs", "direction": "xy", "uncertainty": "day"}
#: The fit windows the Checks page compares (lags fitted), and v2's minimum track length, fixed at v2's default.
WINDOW_LAGS = (2, 3, 4, 5, 6, 8)
V2_MIN_LENGTH = 9
#: A track-length bin with fewer tracks than this is drawn faded and never quoted.
LENGTH_MIN_TRACKS = 10

SECTION_HELP = {
    "n_lags": ("Fit the first lags only.",
               "The line is fitted to the first lags; the rest of the curve is drawn hollow and not fitted. Later "
               "lags rest on fewer, slower tracks and the cell wall bends the curve, so a longer window gives a "
               "lower D. Use at least 3 lags."),
    "max_lag": ("How far the curve is drawn.",
                "The curve is drawn this far but not fitted, so you can see it bend at the cell wall."),
    "weighting": ("How the tracks are averaged at each lag.",
                  "By pairs: a track counts by its number of displacement pairs at that lag, as in the SMTracker 2 "
                  "ensemble average, so the mix shifts towards long, slow tracks across the fitted lags. Each "
                  "track once: every track counts the same at every lag. The Checks page shows both."),
    "direction": ("The camera plane, or the cell's own axes.",
                  "Camera plane: x and y of the image, two dimensions, D = slope / 4. Along / across: each step "
                  "is projected onto its cell's centreline, one dimension, so D = slope / 2 (SMTracker 2 divided "
                  "by 4 here, which halved D). Across a rod about a micron wide the curve flattens within a "
                  "few lags because of the wall."),
    "uncertainty": ("Spread over independent units, never the fit's own error.",
                    "Replicates: each replicate is fitted alone; mean ± SD over replicates (a spread, not a "
                    "confidence interval). Replicates are not paired across conditions. Movies, Cells: a 95 % "
                    "interval from 200 resamples, for a single session only; the movie interval is wider because "
                    "cells in a movie share focus and field. None: the pooled fit."),
}

#: What each page says once, behind its "How to read" (the prose that used to sit under every card).
PAGE_NOTES = {
    "curves": ("One slope over every population at once",
               "D here is one slope through the ensemble curve, a mixture of every state dominated by the fastest. "
               "It is a cross-check, never a population's D; the populations and their fractions come from the SQD "
               "section. Filled points were fitted, hollow ones only drawn. With Replicates, mean ± SD is a "
               "spread, not a confidence interval."),
    "conditions": ("Pale lines are single tracks",
                   "Behind each ensemble curve is a sample of up to 120 single tracks. Hover a point for its lag, "
                   "its spread over tracks and how many tracks reach it. Later lags rest on fewer, longer tracks."),
    "checks": ("Why the fit window and track length move D",
               "Fit window: D from the first 2 to 8 lags, by pairs and with each track once. Pair weighting shifts "
               "the mix towards long, slow tracks (survivorship) and the wall bends every curve, so a D that falls "
               "with the window does not mean slower molecules. Track length: D from each length class on its own. "
               "The intercept is not sigma; for a free molecule it is 2·ndim·(σ² - D·τ/3)."),
    "v2": ("What v2 printed, and why it differs",
           "The SMTracker 2 MSD tab fitted only tracks of at least 9 localisations, each cut to lags 1-8, all 8 "
           "fitted unweighted. Its localisation-error panel fitted a line per track over lags 1-7 and averaged "
           "√|b/4|, which folds negative intercepts onto positive values."),
}


def section_config(n_lags: int = 4, max_lag: int = 10, weighting: str = "pairs", direction: str = "xy",
                   uncertainty: str = "day") -> dict:
    """The committed Run settings. RAISES on a value the dialog cannot offer."""
    if int(n_lags) not in SECTION_LAGS:
        raise ValueError(f"n_lags must be one of {SECTION_LAGS}, got {n_lags!r}")
    if int(max_lag) not in SECTION_CURVE:
        raise ValueError(f"max_lag must be one of {SECTION_CURVE}, got {max_lag!r}")
    if weighting not in SECTION_WEIGHTING:
        raise ValueError(f"weighting must be one of {SECTION_WEIGHTING}, got {weighting!r}")
    if direction not in SECTION_DIRECTIONS:
        raise ValueError(f"direction must be one of {SECTION_DIRECTIONS}, got {direction!r}")
    if uncertainty not in SECTION_UNCERTAINTY:
        raise ValueError(f"uncertainty must be one of {SECTION_UNCERTAINTY}, got {uncertainty!r}")
    return {"n_lags": int(n_lags), "max_lag": int(max_lag), "weighting": weighting, "direction": direction,
            "uncertainty": uncertainty}


def msd_condition(project, condition: str, *, n_lags: int = 4, max_lag: int = 10, weighting: str = "pairs",
                  direction: str = "xy", uncertainty: str = "day") -> dict:
    """Everything the MSD section draws for ONE condition, as plain data (picklable; one pool job per condition):
    `compute_msd_view`'s result -- with v2's numbers always, at v2's own minimum length -- plus ``"checks"``: D
    against the fit window both ways (``"window"``) and D per track-length class (``"length"``). An expected refusal
    comes back as ``{"condition", "settings", "error"}``, as `compute_msd_view`'s does."""
    settings = section_config(n_lags, max_lag, weighting, direction, uncertainty)
    result = compute_msd_view(project, condition, n_lags=settings["n_lags"], max_lag=settings["max_lag"],
                              weighted=weighting == "pairs", direction=direction, uncertainty=uncertainty,
                              legacy=True, min_length=V2_MIN_LENGTH)
    result["section"] = settings
    if "error" in result:
        return result
    # v2's fit is drawn on the v2 page only (as its numbers are), never on every condition's curve
    result["chart"]["legacy_fit"] = None
    result["checks"] = _checks(project, condition, settings)
    return result


def _checks(project, condition: str, settings: dict) -> dict:
    """D against the number of lags fitted (by pairs, and each track once), and D from the early lags of each
    track-length class on its own."""
    from .appd_view import LENGTH_BINS
    from .canonical import without_attrs
    from .msd import MSDError, ensemble_msd, fit_msd, msd_per_track

    table = (project.condition(condition) if settings["direction"] == "xy" else project.projected(condition))
    longest = max(WINDOW_LAGS)
    per_track = msd_per_track(table, max_lag=longest, direction=settings["direction"])
    ndim = int(per_track.attrs["ndim"])
    window = {"lags": [], "pairs": [], "tracks": []}
    curves = {"pairs": ensemble_msd(per_track, weighted=True), "tracks": ensemble_msd(per_track, weighted=False)}
    for n in WINDOW_LAGS:
        window["lags"].append(n)
        for key, curve in curves.items():
            try:
                fit = fit_msd(curve, n_lags=n, weighted=key == "pairs", ndim=ndim)
                window[key].append(float(fit.D_um2_per_s))
            except MSDError:
                window[key].append(None)
    steps = without_attrs(table).groupby("uid", observed=True).size() - 1
    plain = without_attrs(per_track)
    length = []
    for low, high in LENGTH_BINS:
        chosen = steps.index[(steps >= low) & ((steps <= high) if high is not None else True)]
        entry = {"low": int(low), "high": None if high is None else int(high), "n_tracks": int(len(chosen)),
                 "D": None}
        if len(chosen):
            rows = plain[plain["uid"].isin(chosen)]
            n_fit = min(settings["n_lags"], int(low))
            try:
                fit = fit_msd(ensemble_msd(rows, weighted=settings["weighting"] == "pairs"), n_lags=n_fit,
                              weighted=settings["weighting"] == "pairs", ndim=ndim)
                entry["D"], entry["n_lags"] = float(fit.D_um2_per_s), n_fit
            except MSDError as exc:
                entry["why"] = str(exc)
        length.append(entry)
    return {"window": window, "length": length, "ndim": ndim}


# -- the pages ----------------------------------------------------------------------------------------------------- #

def _spread_text(interval: dict | None) -> str:
    if not interval:
        return "–"
    if "withheld" in interval or "note" in interval:
        return "withheld" if "withheld" in interval else "one replicate"
    if interval["mode"] == "day":
        return f"± {_d(interval['sd'])} ({interval['n']} replicates)" if "sd" in interval else "too few replicates"
    return (f"[{_d(interval['lo'])} – {_d(interval['hi'])}] ({interval['n_units']} "
            f"{UNIT_WORDS.get(interval['mode'], 'units')})")


def short_caption(result: dict) -> str:
    """A Per condition card's caption: the tracks, D and its spread -- nothing else."""
    spread = _spread_text(result.get("interval"))
    return (f"{result['n_tracks']:,} tracks · D {_d(result['fit']['D'])} µm²/s"
            + ("" if spread == "–" else f" {spread}"))


def curves_table(results: dict, conditions: list, labels: dict) -> dict:
    """The Curves page's table: one row per condition -- tracks, D, its spread, the intercept -- and the line
    under it that says what the numbers are fitted from."""
    columns = ["Condition", "Tracks", "D (µm²/s)", "Spread", "Intercept (µm²)"]
    rows, settings = [], None
    for c in conditions:
        r = results.get(c)
        name = labels.get(c, c)
        if r is None:
            continue
        if "error" in r:
            rows.append([name, "–", "failed", "", ""])
            continue
        settings = r["settings"]
        rows.append([name, f"{r['n_tracks']:,}", _d(r["fit"]["D"]), _spread_text(r.get("interval")),
                     f"{r['fit']['intercept']:.3g}"])
    caption = ""
    if settings is not None:
        where = {"xy": "camera plane", "along": "along the cell", "across": "across the cell"}[settings["direction"]]
        weighting = "by pairs" if settings["weighted"] else "each track once"
        caption = (f"{where} · first {settings['n_lags']} lags fitted, {weighting} · curve to lag "
                   f"{settings['max_lag']}")
    return {"columns": columns, "rows": rows, "caption": caption}


def window_payload(result: dict) -> dict | None:
    """The Checks page's fit-window chart: D against the lags fitted, by pairs and each track once, the Run's
    window marked; and the sentence that reads it."""
    checks = result.get("checks")
    if not checks:
        return None
    w = checks["window"]
    values = [v for v in w["pairs"] + w["tracks"] if v is not None]
    if not values:
        return None
    chosen = result["settings"]["n_lags"]
    text = ""
    first = next((i for i, n in enumerate(w["lags"]) if w["pairs"][i] is not None), None)
    last = next((i for i in range(len(w["lags"]) - 1, -1, -1) if w["pairs"][i] is not None), None)
    if first is not None and last is not None and last > first:
        change = (w["pairs"][last] - w["pairs"][first]) / w["pairs"][first] * 100.0
        text = (f"by pairs, D {_d(w['pairs'][first])} from {w['lags'][first]} lags, {_d(w['pairs'][last])} from "
                f"{w['lags'][last]} ({change:+.0f} %)")
        t_first, t_last = w["tracks"][first], w["tracks"][last]
        if t_first is not None and t_last is not None:
            text += f"; each track once, {(t_last - t_first) / t_first * 100.0:+.0f} %"
    return {"lags": list(w["lags"]), "pairs": list(w["pairs"]), "tracks": list(w["tracks"]), "chosen": chosen,
            "y_max": max(values) * 1.12, "text": text}


def length_check(result: dict) -> dict | None:
    """The Checks page's track-length chart: D from the early lags of each track-length class (bins under
    `LENGTH_MIN_TRACKS` tracks faded, never quoted), and the sentence that reads it."""
    checks = result.get("checks")
    if not checks:
        return None
    bins = []
    for b in checks["length"]:
        if not b["n_tracks"]:
            continue                                    # a class with no track draws nothing
        name = f"{b['low']}–{b['high']}" if b["high"] is not None else f"{b['low']}+"
        bins.append({"label": name, "n": b["n_tracks"], "D": b["D"], "few": b["n_tracks"] < LENGTH_MIN_TRACKS})
    quoted = [b for b in bins if not b["few"] and b["D"] is not None]
    if len(quoted) >= 2:
        a, z = quoted[0], quoted[-1]
        text = f"D {_d(a['D'])} in {a['label']}-step tracks, {_d(z['D'])} in {z['label']} ({z['n']:,} tracks)"
    else:
        text = "too few long tracks to compare lengths"
    drawn = [b["D"] for b in bins if b["D"] is not None]
    return {"bins": bins, "y_max": (max(drawn) * 1.15) if drawn else 1.0, "text": text}


#: The project sigma's source, in the v2 table's few words.
_SIGMA_SHORT = {"tracker_reported": "the tracker's fit", "theoretical": "from the optics", "configured": "set by hand",
                "measured": "measured"}


def _change(value: float, against: float) -> str:
    change = (value - against) / against * 100.0
    return f"{change:+.1f} %" if abs(change) < 10 else f"{change:+.0f} %"


def v2_rows(result: dict) -> list:
    """The v2 page's table for one condition, in few words (the page's "How to read" says how v2 fitted):
    ``[(what, value), ...]`` -- this page's D, v2's ensemble fit and its tracks at this page's window, and v2's
    localisation-error panel."""
    leg = result.get("legacy") or {}
    fit = result["fit"]
    rows = [("This page", f"{_d(fit['D'])} µm²/s ({fit['n_lags']} lags, {result['n_tracks']:,} tracks)")]
    if "ensemble" in leg:
        e = leg["ensemble"]
        rows.append(("v2's ensemble fit", f"{_d(e['D'])} µm²/s ({_change(e['D'], fit['D'])}, "
                                          f"{leg['tracks_long']:,} tracks)"))
        if fit["ndim"] == 1:
            rows.append(("v2 printed", f"{_d(e['D_printed'])} µm²/s (one axis ÷ 4)"))
        if "track_set_D" in leg:
            rows.append((f"v2's tracks, {fit['n_lags']} lags", f"{_d(leg['track_set_D'])} µm²/s"))
    elif "ensemble_error" in leg:
        rows.append(("v2's ensemble fit", "not formed"))
    if "per_track" in leg:
        p = leg["per_track"]
        rows.append(("v2's σ panel", "none ticked" if p["panel_nm"] is None
                     else f"{p['panel_nm']:.1f} nm ({p['n_ticked']:,} ticked)"))
        rows.append(("every fitted track", f"{p['all_nm']:.1f} nm ({p['n_fitted']:,})"))
        rows.append(("negative intercepts", f"{p['negative_share'] * 100:.1f} % (folded positive)"))
    elif "per_track_error" in leg:
        rows.append(("v2's per-track fits", "not formed"))
    tracker = result.get("tracker_sigma") or {}
    if tracker.get("nm") is not None:
        rows.append(("σ of this project", f"{tracker['nm']:.1f} nm ({_SIGMA_SHORT.get(tracker['provenance'], 'stated')})"))
    return rows
