"""Item 1 output: the legacy SQD table, the legacy SQD figure, and the day diagnostic.

Every interval names its unit and its population in the same place. The legacy unit is
the pooled step. A column ``interval_unit`` carries it on every table row, and the same
string is printed on every figure. It is never a replicate-day interval.

The day diagnostic is a diagnostic. The same legacy fit is run separately on each
replicate day, so the spread between days can be seen next to an interval that does not
contain it. It is deposited as a table and it is deliberately kept OFF the legacy figure,
because the legacy figure shows the legacy design and the legacy design has no day unit.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .guard import assert_neutral, legacy_path, legacy_title
from .sqd import (INTERVAL_WARNING, INTERVAL_WARNING_SHORT,
                  SUBSAMPLING_NOTE,
                  DERIVED_INTERVAL_REASON, fit_legacy_sqd, interval_unit_label,
                  legacy_cdf_model)
from .deviations import LEGACY_N_POP
from .limitations import limitation_line

__all__ = ["sqd_rows", "write_sqd_table", "day_diagnostic", "draw_sqd_figure"]


def sqd_rows(fit: dict, *, condition: str) -> dict:
    """One deposited row: every D, every fraction, every interval limit, and the unit."""
    row = {
        "condition": condition,
        "n_pop": fit["n_pop"],
        "n_steps": fit["n_steps"],
        "n_tracks": fit["n_tracks"],
        "residual": fit["residual"],
        "tau_s": fit["tau_s"],
    }
    for i in range(fit["n_pop"]):
        row[f"D_{i + 1}"] = float(fit["D"][i])
        row[f"D_{i + 1}_lo"] = float(fit["D_lo"][i])
        row[f"D_{i + 1}_hi"] = float(fit["D_hi"][i])
        row[f"A_{i + 1}"] = float(fit["fraction"][i])
        derived = (i == fit["derived_fraction_index"])
        # THE DERIVED FRACTION'S INTERVAL CELL IS NEVER EMPTY and is never propagated
        # from the Jacobian. It carries a short reason instead.
        row[f"A_{i + 1}_lo"] = (DERIVED_INTERVAL_REASON if derived
                                else float(fit["fraction_lo"][i]))
        row[f"A_{i + 1}_hi"] = (DERIVED_INTERVAL_REASON if derived
                                else float(fit["fraction_hi"][i]))
    row["n_at_lower_bound"] = fit["n_at_lower_bound"]
    row["n_at_upper_bound"] = fit["n_at_upper_bound"]
    row["n_parameters"] = fit["n_parameters"]
    row["lower_bounds"] = json.dumps(fit["lower_bounds"])
    row["upper_bounds"] = json.dumps(fit["upper_bounds"])
    row["interval_unit"] = fit["interval_unit"]
    row["interval_method"] = fit["interval_method"]
    row["interval_warning"] = fit["interval_warning"]
    row["design_effect_applied"] = fit["design_effect_applied"]
    row["seeds"] = json.dumps(fit["seeds"])
    row["seeds_clipped_to_bound"] = json.dumps(fit["seeds_clipped_to_bound"])
    return row


def write_sqd_table(fits_by_condition: dict, root=None) -> pd.DataFrame:
    """Deposit ``legacy_out/tables/legacy_sqd_fit.csv``. One row per condition per n_pop."""
    rows = []
    for condition, per_pop in fits_by_condition.items():
        for n_pop in sorted(per_pop):
            rows.append(sqd_rows(per_pop[n_pop], condition=condition))
    table = pd.DataFrame(rows)
    table.to_csv(legacy_path("tables", "legacy_sqd_fit.csv", root=root), index=False)
    return table


def day_diagnostic(steps_by_day: dict, *, condition: str, tau_s: float,
                   n_pop: int = 2) -> pd.DataFrame:
    """The same legacy fit, run separately on each replicate day. A diagnostic only.

    ``steps_by_day`` maps a day to ``(squared displacements, n tracks)``. The legacy
    design has no day unit, so this never enters the legacy figure and never replaces the
    pooled fit. It exists so the between-day spread can be read next to an interval that
    does not contain it.
    """
    rows = []
    for day in sorted(steps_by_day):
        values, n_tracks = steps_by_day[day]
        fit = fit_legacy_sqd(values, tau_s=tau_s, n_pop=n_pop, n_tracks=n_tracks)
        row = {"condition": condition, "day": int(day), "n_pop": n_pop,
               "n_steps": fit["n_steps"], "n_tracks": fit["n_tracks"]}
        for i in range(n_pop):
            row[f"D_{i + 1}"] = float(fit["D"][i])
            row[f"A_{i + 1}"] = float(fit["fraction"][i])
        row["is_diagnostic_only"] = True
        row["why"] = ("the legacy design has no replicate-day unit; this table exists "
                      "to show the between-day spread beside a pooled-step interval "
                      "that does not contain it")
        rows.append(row)
    return pd.DataFrame(rows)


def write_day_diagnostic(frames, root=None) -> pd.DataFrame:
    table = pd.concat(frames, ignore_index=True) if not isinstance(frames, pd.DataFrame) \
        else frames
    table.to_csv(legacy_path("tables", "legacy_sqd_day_diagnostic.csv", root=root),
                 index=False)
    return table


def draw_sqd_figure(fit: dict, *, condition: str, root=None):
    """The cumulative histogram with the fitted curve on top.

    N steps and N tracks are printed and BOTH are named. The interval's unit and
    population are printed in the same place as the interval itself.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, y = fit["cdf_x"], fit["cdf_y"]
    curve = legacy_cdf_model(
        np.concatenate([fit["D"], fit["fraction"][:-1]]) if fit["n_pop"] > 1
        else fit["D"], fit["tau_s"], x, fit["n_pop"])

    fig = plt.figure(figsize=(120 / 25.4, 95 / 25.4))
    ax = fig.add_axes((0.14, 0.40, 0.82, 0.52))
    ax.plot(x, y, color="#0b0b0b", lw=1.0, label="cumulative, measured")
    ax.plot(x, curve, color="#D47B59", lw=1.2, ls="--", label="legacy fit")
    ax.set_xscale("log")
    ax.set_xlabel("squared displacement (um^2)", fontsize=7)
    ax.set_ylabel("cumulative probability", fontsize=7)
    ax.set_title(legacy_title(f"SQD fit, {condition}, {fit['n_pop']} populations"),
                 loc="left", fontsize=8)
    ax.tick_params(labelsize=6)
    ax.legend(frameon=False, fontsize=6)

    lines = [
        f"N tracks {fit['n_tracks']}   N steps {fit['n_steps']}   "
        f"tau {fit['tau_s']} s",
        "  ".join(f"D_{i + 1} {fit['D'][i]:.4f}  A_{i + 1} {fit['fraction'][i]:.3f}"
                  for i in range(fit["n_pop"])),
        interval_unit_label(fit["n_steps"]),
        f"parameters stopped at a fit limit: {fit['n_at_lower_bound']} at the lower "
        f"limit, {fit['n_at_upper_bound']} at the upper limit, of "
        f"{fit['n_parameters']}",
        # THE LIMITATION LINE, one source, also written to the provenance.
        limitation_line("sqd"),
    ]
    for i, text in enumerate(lines):
        fig.text(0.02, 0.30 - 0.055 * i, text, fontsize=4.8, color="#52514e",
                 wrap=True)
    assert_neutral(fig)
    # The FULL sentence goes in a caption file beside the image, so the figure carries
    # one short line and nothing is lost.
    caption = legacy_path("figures", f"legacy_sqd_{condition.lower()}_caption.txt",
                          root=root)
    blocks = [
        legacy_title(f"SQD fit, {condition}, {fit['n_pop']} populations"),
        interval_unit_label(fit["n_steps"]),
        INTERVAL_WARNING,
        SUBSAMPLING_NOTE,
    ]
    caption.write_text((chr(10) * 2).join(blocks) + chr(10), encoding="utf-8")
    path = legacy_path("figures", f"legacy_sqd_{condition.lower()}.png", root=root)
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    return path
