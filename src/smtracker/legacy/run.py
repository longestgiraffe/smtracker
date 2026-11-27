"""The legacy-mode entry point. One switch, one output directory.

Each item is a separate function so the CLI can run one or all of them, and so each has
its own unit test. Nothing here computes: the items live in their own modules and this
file only dispatches, deposits and reports.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .deviations import LEGACY_N_POP, STATUSES, write_deviations
from .guard import LEGACY_ROOT, legacy_path
from .limitations import LIMITATION_LINES
from .loader import DAY_MAP_AUTHORITY, load_legacy_condition
from .sqd import INTERVAL_WARNING, SUBSAMPLING_NOTE, fit_legacy_sqd
from .sqd_check import COVERAGE_CAVEAT, write_coverage_table, write_ground_truth_table
from .sqd_report import (day_diagnostic, draw_sqd_figure, write_day_diagnostic,
                         write_sqd_table)

__all__ = ["run_legacy", "run_sqd", "LEGACY_ITEMS", "discover_groups"]

#: The legacy set has its own count assertion, separate from the manuscript one.
LEGACY_ITEMS = ("sqd", "speedmap", "sda", "sla")

def discover_groups(root) -> list:
    """``[(protein folder, condition folder)]`` under ``root``, laid out as
    ``<root>/<protein>/<condition>/.../TrackingPackage``: every protein folder, and in it every
    condition folder that holds a u-track project somewhere below it, both in name order. Read
    from the folder, never from a list of names written here. Hidden folders and ``backups``
    are skipped. RAISES when nothing is found, naming the layout it looked for."""
    root = Path(root)

    def folders(where: Path) -> list:
        return sorted((p for p in where.iterdir() if p.is_dir() and not p.name.startswith(".")
                       and p.name.lower() != "backups"), key=lambda p: p.name)

    groups = [(protein.name, condition.name) for protein in folders(root)
              for condition in folders(protein)
              if next(condition.rglob("TrackingPackage"), None) is not None]
    if not groups:
        raise ValueError(f"{root}: no <protein>/<condition> folder holding a u-track project "
                         f"(a TrackingPackage folder) was found.")
    return groups


def run_sqd(root, *, groups=None, out_root=None) -> dict:
    """Item 1 on real data. Deposits the fit table, the day diagnostic, the filter
    inventory, one figure per group, and the provenance. ``groups``: the
    ``(protein, condition)`` folders to fit; by default every one under ``root``
    (`discover_groups`)."""
    root = Path(root)
    fits, inventories, day_frames, figures = {}, [], [], {}
    for protein, condition in (discover_groups(root) if groups is None else groups):
        enzyme = protein.split("-")[0]
        cond_root = root / protein / condition
        if not cond_root.is_dir():
            print(f"  {enzyme}/{condition}: not present, SKIPPED and named")
            continue
        data = load_legacy_condition(cond_root, condition=condition)
        label = f"{enzyme}_{condition}"
        fits[label] = {}
        for n_pop in LEGACY_N_POP:
            fits[label][n_pop] = fit_legacy_sqd(
                data["sq_disp_um2"], tau_s=data["tau_s"], n_pop=n_pop,
                n_tracks=data["n_tracks"])
        inv = data["filter_inventory"].copy()
        inv.insert(0, "condition", condition)
        inv.insert(0, "enzyme", enzyme)
        inventories.append(inv)
        day_frames.append(day_diagnostic(data["per_day"], condition=label,
                                         tau_s=data["tau_s"], n_pop=2))
        figures[label] = str(draw_sqd_figure(fits[label][2], condition=label,
                                             root=out_root))
        two = fits[label][2]
        print(f"  {enzyme:5} {condition:8} tracks {data['n_tracks']:5} "
              f"steps {data['n_steps']:6} cells {data['n_cells']:4}  "
              f"D {two['D'][0]:.4f} {two['D'][1]:.4f}  "
              f"A {two['fraction'][0]:.3f} {two['fraction'][1]:.3f}  "
              f"at fit limit {two['n_at_lower_bound']} lower "
              f"{two['n_at_upper_bound']} upper of {two['n_parameters']}")

    table = write_sqd_table(fits, root=out_root)
    inventory = pd.concat(inventories, ignore_index=True)
    inventory.to_csv(legacy_path("tables", "legacy_filter_inventory.csv",
                                 root=out_root), index=False)
    write_day_diagnostic(pd.concat(day_frames, ignore_index=True), root=out_root)

    limits = table.groupby(["condition", "n_pop"])[
        ["n_at_lower_bound", "n_at_upper_bound", "n_parameters"]].sum().reset_index()
    limits.to_csv(legacy_path("tables", "legacy_sqd_fit_limits.csv", root=out_root),
                  index=False)
    return {"table": table, "inventory": inventory, "limits": limits,
            "figures": figures}


def run_legacy(args) -> int:
    """Run legacy mode. Returns a process exit code."""
    frame = write_deviations()
    counts = frame["status"].value_counts()
    print(f"LEGACY MODE. Output directory: {LEGACY_ROOT.resolve()}")
    # EACH STATUS ITS OWN COUNT. They are different kinds of row and are never summed.
    print("  deviations: " + ", ".join(
        f"{int(counts.get(status, 0))} {status.replace('_', ' ')}"
        for status in STATUSES) + ". Every count, never a sum.")

    ground = write_ground_truth_table()
    coverage = write_coverage_table()
    fitted = ground[ground["covers_truth"].notna()]
    print(f"  seeded ground truth: interval covered "
          f"{int(fitted['covers_truth'].sum())} of {len(fitted)} fitted parameters")
    for r in coverage.itertuples():
        print(f"    {r.parameter}: half-width / sd = {r.ratio_half_width_over_sd:.3f} "
              f"(1.96 expected), coverage {r.coverage_count}/{r.n_replicates}")

    root = getattr(args, "root", None)
    if root is None:
        print("  no --root given, so the real-data run is SKIPPED and named. "
              "Pass --root to run it.")
        return 0

    result = run_sqd(root)
    (Path(legacy_path("legacy_sqd_provenance.json"))).write_text(json.dumps({
        "what_this_is": "a display of the OLD method over the CURRENT data. Not a "
                        "second analysis. No number here may enter the manuscript "
                        "deposit.",
        "unit": "the POOLED STEP. There is no replicate-day unit and no cell unit. That "
                "is the legacy design and it is correct here. It is not correct "
                "anywhere else in this project.",
        "n_pop_supported": list(LEGACY_N_POP),
        "why_not_four": "the source supports 1, 2 and 3. A four-population fit written "
                        "now would be a new method carrying an old label.",
        "day_map": DAY_MAP_AUTHORITY,
        "interval": INTERVAL_WARNING,
        "coverage_caveat": COVERAGE_CAVEAT,
        "subsampling": SUBSAMPLING_NOTE,
        "no_comparison": "legacy D values are NEVER placed beside an HMM state, a saSPT "
                         "occupancy or a threshold count, in a table, a figure or a "
                         "sentence. The two output directories keep them apart.",
        "limitation_line": LIMITATION_LINES["sqd"],
        "limitation_policy": (
            "version 3 fixes the defects and adds what the old tool could not do. The old output stays available because people need to recognise their own figures. Supported, reproduced faithfully in design, NOT recommended for a claim. The line above is printed on every legacy figure so that distinction appears at the point of use."),
        "figures": result["figures"],
    }, indent=2), encoding="utf-8")
    print(f"{chr(10)}  fit limits per condition and n_pop: "
          f"{legacy_path('tables', 'legacy_sqd_fit_limits.csv', make_parent=False)}")
    return 0
