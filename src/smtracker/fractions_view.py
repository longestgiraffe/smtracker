"""The SuperPlots' numbers, groupings and table rows, with no Qt anywhere in it.

`ui/workspace/fractions_chart.py` is the Qt shell (the two SuperPlot charts and their cards); this is
what it draws from. They are split because of the Thin Qt rule (docs/gui_tool_views.md;
`tests/test_ui_thin_qt.py` walks the whole GUI layer): a wrong number must fail in a headless test,
not look subtly wrong on screen, and a process-pool worker should import this module without importing
Qt.

`compute_fractions_checks` is picklable and runs on the process pool, one condition at a time. It
calls the SAME analysis functions the legacy controllers call (`sqd.fit_sqd`/`bootstrap_sqd`,
`hmm.fit_hmm`/`step_sequences`, `qc.track_summary`), then hands the pure builders below plain
lists, floats and strings, which are the only things the Qt layer reads. The builders take no fit, so
each rule can be tested with a hand-made table.

The rules kept here (they used to live inside the chart classes):
  * a condition shows replicates as separate columns only when it HAS two or more replicate days
    (`has_real_days`); otherwise every cell is one "cells" column;
  * a condition whose HMM will not converge draws per-cell apparent D in its OWN units, never forced
    onto a 0-100 % axis, and has no SQD reference band;
  * the distribution drops every non-positive D before taking logs, and a degenerate range is widened
    by half a decade on either side;
  * the jitter is deterministic (`_JITTER_SEED`, one uniform draw per plotted point in plotting
    order), so the same data always draws the same picture.
Values arrive in DISPLAY units (percent for the HMM fraction, um2/s otherwise).
"""
from __future__ import annotations

import math

import numpy as np

__all__ = [
    "compute_fractions_checks",
    "distribution_payload",
    "distribution_table",
    "fraction_payload",
    "fraction_table",
    "has_real_days",
]

_JITTER_SEED = 0
_JITTER_HALF_WIDTH = 0.18


def has_real_days(days) -> bool:
    """More than one distinct, non-missing replicate day."""
    return len({d for d in days if d is not None}) >= 2


def _jitter(rng, n: int) -> list[float]:
    return [float(rng.uniform(-_JITTER_HALF_WIDTH, _JITTER_HALF_WIDTH)) for _ in range(n)]


# --------------------------------------------------------------------------- #
# Chart payloads
# --------------------------------------------------------------------------- #


def fraction_payload(per_cell, frac_col: str, hmm_ok: bool, sqd_pct: float,
                     sqd_ci) -> dict:
    """The per-cell fraction SuperPlot: points by replicate day with their day means, and the
    SQD cell-weighted CI as a reference band (HMM case only)."""
    scale = 100.0 if hmm_ok else 1.0
    values = per_cell[frac_col].to_numpy(dtype=float) * scale
    days = sorted(per_cell["day"].unique())
    by_day = has_real_days(days)

    if hmm_ok:
        lo, hi = 0.0, 100.0
        band = [float(v) for v in sqd_ci]  # already in percent
        overall = float(sqd_pct)
    else:
        if len(values):
            span = float(np.nanmax(values)) - float(np.nanmin(values))
            lo = max(0.0, float(np.nanmin(values)) - 0.1 * span)
            hi = float(np.nanmax(values)) + 0.1 * span
        else:
            lo, hi = 0.0, 1.0
        band = None
        overall = None

    rng = np.random.default_rng(_JITTER_SEED)
    groups = []
    if by_day:
        for index, day in enumerate(days):
            vals = per_cell.loc[per_cell["day"] == day, frac_col].to_numpy(dtype=float) * scale
            groups.append({
                "label": f"Replicate {int(day)}", "colour_index": index,
                "values": [float(v) for v in vals], "jitter": _jitter(rng, len(vals)),
                "mean": float(vals.mean()) if len(vals) else None,
            })
    else:
        groups.append({
            "label": "cells", "colour_index": None,
            "values": [float(v) for v in values], "jitter": _jitter(rng, len(values)),
            "mean": None,
        })
    return {
        "y_label": "Slow fraction (%)" if hmm_ok else "Apparent D (µm²/s)",
        "lo": lo, "hi": hi, "band": band, "overall": overall,
        "by_day": by_day, "groups": groups,
    }


def distribution_payload(track_points) -> dict:
    """The per-track apparent-D SuperPlot: a pure strip on a log axis, scattered by replicate day."""
    appd = track_points["appd_um2_per_s"].to_numpy(dtype=float)
    positive = appd[appd > 0]
    days = sorted(track_points["day"].unique())
    by_day = has_real_days(days)
    if not len(positive):
        return {"empty": True, "log_lo": None, "log_hi": None, "by_day": by_day, "groups": []}

    lo, hi = math.log10(positive.min()), math.log10(positive.max())
    if hi <= lo:
        lo, hi = lo - 0.5, hi + 0.5

    rng = np.random.default_rng(_JITTER_SEED)
    groups = []
    if by_day:
        for index, day in enumerate(days):
            vals = track_points.loc[track_points["day"] == day,
                                    "appd_um2_per_s"].to_numpy(dtype=float)
            vals = vals[vals > 0]
            groups.append({"label": f"Replicate {int(day)}", "colour_index": index,
                           "values": [float(v) for v in vals], "jitter": _jitter(rng, len(vals))})
    else:
        groups.append({"label": "cells", "colour_index": None,
                       "values": [float(v) for v in positive],
                       "jitter": _jitter(rng, len(positive))})
    return {"empty": False, "log_lo": lo, "log_hi": hi, "by_day": by_day, "groups": groups}


# --------------------------------------------------------------------------- #
# Table rows (the accessible equivalent of each chart)
# --------------------------------------------------------------------------- #


def fraction_table(per_cell, frac_col: str, hmm_ok: bool) -> dict:
    """One row per CELL, in the same units the chart draws, so the value column's name can never
    disagree with the y-axis label beside it."""
    by_day = has_real_days(per_cell["day"])
    value_label = "Slow fraction (%)" if hmm_ok else "Apparent D (µm²/s)"
    columns = ["Cell", *(["Replicate"] if by_day else []), value_label]
    ordered = per_cell.sort_values(["day", "cell"] if by_day else ["cell"])
    rows = []
    for _, row in ordered.iterrows():
        value = row[frac_col]
        value_text = f"{100 * value:.1f}%" if hmm_ok else f"{value:.3g}"
        # `cell` is NOT always numeric: real projects carry string ids ("c0", "c1", ...).
        cell_row = [str(row["cell"])]
        if by_day:
            cell_row.append(f"Replicate {int(row['day'])}")
        cell_row.append(value_text)
        rows.append(cell_row)
    return {"columns": columns, "rows": rows, "caption": f"{len(rows)} cell(s)."}


def distribution_table(track_points) -> dict:
    """Summarised PER REPLICATE (median and IQR of per-track apparent D), deliberately not one row
    per track: the table builds a real widget per row and a condition has thousands of tracks."""
    vals = track_points["appd_um2_per_s"]
    by_day = has_real_days(track_points["day"])
    columns = ["Replicate", "Tracks", "Median D (µm²/s)", "IQR (µm²/s)"]
    if by_day:
        groups = [(f"Replicate {int(day)}", track_points[track_points["day"] == day])
                  for day in sorted(track_points["day"].dropna().unique())]
    else:
        groups = [("cells", track_points)]
    rows = []
    for name, sub in groups:
        sub_vals = sub["appd_um2_per_s"]
        sub_vals = sub_vals[sub_vals > 0]
        if len(sub_vals) == 0:
            rows.append([name, "0", "–", "–"])
            continue
        median = sub_vals.median()
        q1, q3 = sub_vals.quantile([0.25, 0.75])
        rows.append([name, str(len(sub_vals)), f"{median:.3g}", f"{q1:.3g}–{q3:.3g}"])
    return {"columns": columns, "rows": rows,
            "caption": f"{int((vals > 0).sum())} track(s) total."}


# --------------------------------------------------------------------------- #
# Compute: one condition, on the process pool
# --------------------------------------------------------------------------- #


def compute_fractions_checks(project, condition: str) -> dict:
    """One condition's SuperPlots. NOT CHEAP (a real SQD fit, its cell bootstrap, an HMM fit), so it
    runs off the GUI thread on its own Run press. Returns plain data only."""
    from .hmm import HMMError, fit_hmm, step_sequences
    from .observables import squared_displacements
    from .qc import track_summary
    from .sqd import bootstrap_sqd, fit_sqd

    part = project.condition(condition)
    tau = project.params.tau_s

    obs = squared_displacements(part)
    sqd_cell = fit_sqd(obs, n_pop=2, weighting="cell")
    sqd_cell_boot = bootstrap_sqd(obs, n_pop=2, unit="cell", n_boot=50, weighting="cell")
    sqd_cell_pct = 100 * float(sqd_cell.fractions[0])
    sqd_cell_ci = tuple(100 * float(v) for v in
                        sqd_cell_boot["fraction_1"].quantile([0.025, 0.975]))

    try:
        hmm_fit = fit_hmm(step_sequences(part), n_states=2, tau_s=tau, n_starts=6)
        summary = track_summary(part, hmm_fit=hmm_fit)
        frac_col, hmm_ok = "slow_fraction", True
    except HMMError:
        summary = track_summary(part)
        frac_col, hmm_ok = "appd_um2_per_s", False

    per_cell = (summary.groupby("cell", observed=True)
                .agg(day=("day", "first"), **{frac_col: (frac_col, "mean")})
                .reset_index())
    track_points = summary[["uid", "day", "appd_um2_per_s"]].dropna()

    return {
        "condition": condition,
        "hmm_ok": hmm_ok,
        "n_tracks": int(summary.attrs["n_tracks"]),
        "n_cells": int(per_cell.shape[0]),
        "fraction": fraction_payload(per_cell, frac_col, hmm_ok, sqd_cell_pct, sqd_cell_ci),
        "distribution": distribution_payload(track_points),
        "fraction_table": fraction_table(per_cell, frac_col, hmm_ok),
        "distribution_table": distribution_table(track_points),
    }
