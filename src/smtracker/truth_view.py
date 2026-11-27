"""A simulated set's truth beside the analysis (25 Nov 2026, step 5 of `docs/generator_plan.md`).

A project built by the generator (`generator.build`) carries its truth in ``<set>/truth/truth.json``: what was
declared and what was counted, per condition and replicate. When such a project is loaded, the workspace's header
menu offers "Compare with the simulation's truth": one row per condition and population with the declared D and
share, the share COUNTED in the simulation (the value an estimate is compared with), and the last SQD Run's fitted D
and share beside them, matched slowest to slowest. Nothing is refitted here: the fit is the Run's (`bubble_view`
`BubblePlotData`), the truth is the file's.

THE RULES SAID ON THE TABLE: an estimate is compared with the COUNTED share, never the declared one (the counted share
is what the simulated molecules actually did); a fitted D is an apparent D (a bound population reads the localisation
floor, a fast one is shortened by the cell wall and the linking radius); where the fit has a different number of
populations than the truth, the rows are not matched and the table says so.

No Qt; plain dicts and strings out.
"""
from __future__ import annotations

import json
from pathlib import Path

__all__ = ["find_truth", "fitted_from_plot", "truth_table", "TRUTH_NOTES"]

TRUTH_NOTES = (
    "Compare a fitted share with the COUNTED share: what the simulated molecules did, step by step. The declared "
    "share is what was asked for.",
    "A fitted D is an apparent D: a bound population reads the localisation floor, and a fast one is shortened by "
    "the cell wall and the linking radius.",
    "Fitted values are the last SQD Run's, matched slowest to slowest.",
)


def find_truth(condition_roots) -> tuple | None:
    """``(truth, path)`` of the generator's truth beside the loaded conditions: ``truth/truth.json`` in the folder
    that holds them, written by `generator`; None for any other project (real data, the paper's sets)."""
    parents = {Path(p).resolve().parent for p in condition_roots}
    for parent in sorted(parents):
        path = parent / "truth" / "truth.json"
        if not path.is_file():
            continue
        try:
            truth = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if truth.get("written_by") == "smtracker.generator" and isinstance(truth.get("conditions"), dict):
            return truth, path
    return None


def fitted_from_plot(plot) -> dict:
    """``{condition: [(D, fraction), ...] slowest first}`` from a Run's `bubble_view.BubblePlotData` (the first
    series only, as drawn without an overlay)."""
    if plot is None:
        return {}
    out: dict = {}
    for point in getattr(plot, "points", []):
        if getattr(point, "style_index", 0) != 0:
            continue
        name = plot.conditions[point.column]
        out.setdefault(str(name), []).append((int(point.population), float(point.d), float(point.fraction)))
    return {name: [(d, f) for _p, d, f in sorted(rows)] for name, rows in out.items()}


def _pct(value) -> str:
    return "" if value is None else f"{100 * float(value):.1f} %"


def _d(value) -> str:
    return "" if value is None else f"{float(value):.3g}"


def truth_table(truth: dict, fitted: dict | None = None) -> dict:
    """``{"header": [...], "rows": [[...]], "notes": [...]}``: per condition and population, its name, place,
    declared D and share, counted share (of steps), and the fitted D and share where the Run fitted the same number
    of populations. A condition the Run did not fit, or fitted with another count, says so in ``notes``."""
    from .generator import _REGION_WORDS

    fitted = fitted or {}
    unfitted, other_count, changing = [], [], []
    header = ["Condition", "Population", "Where", "D declared", "Share declared", "Share counted",
              "D fitted", "Share fitted"]
    rows, notes = [], list(TRUTH_NOTES)
    for name, cond in truth["conditions"].items():
        pops = cond["populations"]
        declared = cond.get("stationary_declared") or [p["share"] for p in pops]
        counted = cond.get("occupancy_steps") or [None] * len(pops)
        order = sorted(range(len(pops)), key=lambda i: (float(pops[i]["D_um2_s"]), i))
        fit = fitted.get(name)
        matched = fit is not None and len(fit) == len(pops)
        if fit is None:
            unfitted.append(name)
        elif not matched:
            other_count.append(f"{name}: the Run fitted {len(fit)} population{'s' if len(fit) != 1 else ''}, the "
                               f"truth has {len(pops)}; compare them by hand.")
        for rank, i in enumerate(order):
            p = pops[i]
            d_fit, f_fit = fit[rank] if matched else (None, None)
            rows.append([name, p.get("name") or f"population {i + 1}", _REGION_WORDS.get(p["where"], p["where"]),
                         _d(p["D_um2_s"]),
                         _pct(declared[i]), _pct(counted[i]), _d(d_fit), _pct(f_fit)])
        kin = cond.get("kinetics")
        if kin and kin.get("after_populations") is not None:
            changing.append(name)
    if unfitted:
        notes.append(("No SQD Run yet" if len(unfitted) == len(truth["conditions"]) else
                      "Not in the last SQD Run: " + ", ".join(unfitted)) + ", so no fitted values.")
    notes += other_count
    if changing:
        notes.append(f"{', '.join(changing)} change{'s' if len(changing) == 1 else ''} after an addition: the counted "
                     f"share here is over the whole course; truth/<condition>_kinetics.csv holds it per 10 s.")
    return {"header": header, "rows": rows, "notes": notes}
