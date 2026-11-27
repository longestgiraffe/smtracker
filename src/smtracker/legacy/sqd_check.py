"""The seeded ground-truth check for the legacy SQD fit, and its deposited table.

Why this exists. The legacy interval is computed by the nlparci equivalent on POOLED
STEPS. Steps inside one track are not independent, so the interval is far too narrow.
That is a property of the legacy design and legacy mode reproduces it without change.

An assertion that an interval is "too narrow" is an opinion. A run on data with a KNOWN
answer turns it into a measurement. This module builds a two-population dataset with
stated true values, runs the legacy fit on it, and deposits one row per parameter with a
``covers_truth`` column.

No correction is offered. The project design effect of 2.68 was measured for a different
statistic on this data. It is not assumed to transfer, so no widened interval appears
here or anywhere else in legacy mode.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .guard import legacy_path
from .sqd import fit_legacy_sqd, interval_unit_label

__all__ = ["TRUE_D", "TRUE_FRACTION", "SEED", "N_STEPS", "TAU_S", "N_REPLICATES",
           "COVERAGE_CAVEAT",
           "ground_truth_check", "write_ground_truth_table", "coverage_simulation",
           "write_coverage_table"]

#: The known answer. Two populations, well separated, so nothing rests on resolving a
#: state near the floor.
TRUE_D = (0.05, 0.80)
TRUE_FRACTION = (0.30, 0.70)
SEED = 0
N_STEPS = 20000
TAU_S = 0.02
N_TRACKS = 4000

#: Datasets in the coverage simulation. One seeded run cannot separate an unlucky draw
#: from a broken interval. 200 can.
N_REPLICATES = 200

#: Attached to every row of the coverage table, permanently. The simulation is the BEST
#: CASE for the legacy interval, and the number it produces is therefore a floor on how
#: bad the real behaviour is, not an estimate of it.
COVERAGE_CAVEAT = (
    "BEST CASE FOR THE LEGACY INTERVAL. These 200 datasets were generated with "
    "INDEPENDENT steps and NO TRACK STRUCTURE: every squared displacement is an "
    "independent draw from the exact model the legacy fit assumes. Real steps inside one "
    "track are correlated, and real tracks inside one cell share an environment, so the "
    "real spread of the fitted value is WIDER than the spread measured here and the real "
    "coverage is WORSE. The ratio and the coverage count below are therefore bounds on "
    "the best the legacy interval can do, not estimates of what it does on real data.")


def ground_truth_check(*, seed: int = SEED, n_steps: int = N_STEPS,
                       tau_s: float = TAU_S) -> tuple[pd.DataFrame, dict]:
    """Run the legacy fit on seeded data. Return the per-parameter table and the fit.

    The same run produces the fit and the table, so the interval in the table is the
    interval the fit reported. There is no second run to reconcile.
    """
    rng = np.random.default_rng(seed)
    is_low = rng.random(n_steps) < TRUE_FRACTION[0]
    D = np.where(is_low, TRUE_D[0], TRUE_D[1])
    # An exponential squared displacement with mean 4*D*tau is exactly what the legacy
    # model assumes, so the fit is given the best case it can be given.
    sq_disp = rng.exponential(4 * D * tau_s)

    fit = fit_legacy_sqd(sq_disp, tau_s=tau_s, n_pop=2, n_tracks=N_TRACKS)

    rows = []
    for i, truth in enumerate(sorted(TRUE_D)):
        rows.append({
            "parameter": f"D_{i + 1}", "unit_of_parameter": "um^2/s",
            "true_value": truth, "fitted_value": float(fit["D"][i]),
            "interval_lo": float(fit["D_lo"][i]), "interval_hi": float(fit["D_hi"][i]),
        })
    order = np.argsort(TRUE_D)
    for i, j in enumerate(order):
        truth = TRUE_FRACTION[j]
        lo, hi = float(fit["fraction_lo"][i]), float(fit["fraction_hi"][i])
        rows.append({
            "parameter": f"A_{i + 1}", "unit_of_parameter": "fraction",
            "true_value": truth, "fitted_value": float(fit["fraction"][i]),
            "interval_lo": lo, "interval_hi": hi,
        })

    table = pd.DataFrame(rows)
    covers = []
    for r in table.itertuples():
        if not np.isfinite(r.interval_lo) or not np.isfinite(r.interval_hi):
            covers.append(pd.NA)      # the derived fraction has no fitted interval
        else:
            covers.append(bool(r.interval_lo <= r.true_value <= r.interval_hi))
    table["covers_truth"] = covers
    table["interval_width"] = table["interval_hi"] - table["interval_lo"]
    table["distance_from_truth"] = (table["fitted_value"] - table["true_value"]).abs()
    table["widths_from_truth"] = (table["distance_from_truth"]
                                  / table["interval_width"].replace(0, np.nan))
    table["n_steps"] = fit["n_steps"]
    table["unit"] = interval_unit_label(fit["n_steps"])
    table["interval_method"] = fit["interval_method"]
    table["design_effect_applied"] = fit["design_effect_applied"]
    return table, fit


def write_ground_truth_table(root=None) -> pd.DataFrame:
    """Deposit ``legacy_out/tables/legacy_sqd_ground_truth.csv`` and return it."""
    table, _ = ground_truth_check()
    table.to_csv(legacy_path("tables", "legacy_sqd_ground_truth.csv", root=root),
                 index=False)
    return table


def _one_dataset(seed: int, n_steps: int, tau_s: float):
    rng = np.random.default_rng(seed)
    is_low = rng.random(n_steps) < TRUE_FRACTION[0]
    D = np.where(is_low, TRUE_D[0], TRUE_D[1])
    return rng.exponential(4 * D * tau_s)


def coverage_simulation(*, n_replicates: int = N_REPLICATES, n_steps: int = N_STEPS,
                        tau_s: float = TAU_S, confidence: float = 0.95) -> pd.DataFrame:
    """Fit ``n_replicates`` seeded datasets and measure what the legacy interval does.

    This is the headline result, not the single seeded run. One dataset cannot tell an
    unlucky draw from a broken interval, because a 95 per cent interval is expected to
    miss about 1 time in 20 by construction.

    Three numbers per parameter, and the third is the one that matters:

    * the spread of the fitted value across the replicates. This is the real sampling
      error of the estimator.
    * the mean HALF-WIDTH the legacy interval reports.
    * their ratio. A well calibrated interval half-width is about twice the standard
      error for a 95 per cent interval, so a ratio near 2 is right and a ratio far below
      2 means the interval is too narrow by that factor.

    The point estimates are not under test here and are not in question.
    """
    fitted = {"D_1": [], "D_2": [], "A_1": []}
    half = {"D_1": [], "D_2": [], "A_1": []}
    covered = {"D_1": 0, "D_2": 0, "A_1": 0}
    truth = {"D_1": min(TRUE_D), "D_2": max(TRUE_D),
             "A_1": TRUE_FRACTION[int(np.argmin(TRUE_D))]}

    for k in range(n_replicates):
        fit = fit_legacy_sqd(_one_dataset(1000 + k, n_steps, tau_s), tau_s=tau_s,
                             n_pop=2, n_tracks=N_TRACKS, confidence=confidence)
        values = {"D_1": (fit["D"][0], fit["D_lo"][0], fit["D_hi"][0]),
                  "D_2": (fit["D"][1], fit["D_lo"][1], fit["D_hi"][1]),
                  "A_1": (fit["fraction"][0], fit["fraction_lo"][0],
                          fit["fraction_hi"][0])}
        for name, (value, lo, hi) in values.items():
            fitted[name].append(float(value))
            half[name].append((float(hi) - float(lo)) / 2.0)
            if lo <= truth[name] <= hi:
                covered[name] += 1

    rows = []
    for name in ("D_1", "D_2", "A_1"):
        sd = float(np.std(fitted[name], ddof=1))
        mean_half = float(np.mean(half[name]))
        mean_fitted = float(np.mean(fitted[name]))
        # THE BIAS CARRIES ITS OWN INTERVAL. "Unbiased" without one is a claim about a
        # mean estimated from a finite number of replicates, and it is not available.
        se_of_mean = sd / np.sqrt(n_replicates)
        bias = mean_fitted - truth[name]
        bias_half = 1.96 * se_of_mean
        rows.append({
            "parameter": name,
            "true_value": truth[name],
            "mean_fitted_value": mean_fitted,
            "se_of_mean_fitted": float(se_of_mean),
            "bias": float(bias),
            "bias_lo": float(bias - bias_half),
            "bias_hi": float(bias + bias_half),
            "bias_consistent_with_zero": bool(abs(bias) <= bias_half),
            "bias_bounded_within_percent_of_true": float(
                100.0 * bias_half / truth[name]) if truth[name] else np.nan,
            "sd_fitted_across_replicates": sd,
            "mean_reported_interval_half_width": mean_half,
            "ratio_half_width_over_sd": (mean_half / sd) if sd > 0 else np.nan,
            "ratio_expected_for_95_percent": 1.96,
            "coverage_count": covered[name],
            "n_replicates": int(n_replicates),
            "coverage_fraction": covered[name] / float(n_replicates),
            "nominal_coverage": confidence,
            "expected_count_at_nominal": round(confidence * n_replicates),
            "n_steps_per_replicate": int(n_steps),
            "unit": interval_unit_label(n_steps),
            "caveat": COVERAGE_CAVEAT,
        })
    return pd.DataFrame(rows)


def write_coverage_table(root=None, **kwargs) -> pd.DataFrame:
    """Deposit ``legacy_out/tables/legacy_sqd_interval_coverage.csv``."""
    table = coverage_simulation(**kwargs)
    table.to_csv(legacy_path("tables", "legacy_sqd_interval_coverage.csv", root=root),
                 index=False)
    return table
