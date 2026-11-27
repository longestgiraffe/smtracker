"""Distances from tracks to foci the experimenter marks, headless (v2's Active Tools > Distances measurements,
`SMTracker.m` ``onDistance`` / ``getDistances`` / ``distanceHistogram``; added 17 Oct 2026).

WHAT v2 DID. Cell by cell, the user clicked points of interest on the image (``ginput``: a focus of another protein, a
pole, a septum); for every localisation of every track in that cell the distance to the NEAREST focus, in um; all
cells' distances of a condition pooled into one probability histogram, "Distances from trajectories to the nearest
focus (N = ...)".

WHAT IS DONE HERE, and where it differs:

- The same distance: every localisation of the cell's tracks (the analysis's tracks: the cell filter, exclusions and
  the track cap applied) to its nearest focus, um. Foci are kept per condition, movie and cell, in image pixels.
- CHANCE BESIDE IT (new): a histogram alone cannot say whether molecules sit NEAR a focus or merely in a cell that is
  small around it -- in a 1 um wide cell every point is within half a micron of a focus on the axis. So the same
  foci are measured from points spread UNIFORMLY over the same cell's outline (`CHANCE_PER_LOCALISATION` per
  localisation, a fixed seed), and the card reads observed against chance.
- THE UNIT IS THE CELL, then the day (RULE 0): each marked cell's median distance minus its own chance median (the
  EXCESS; negative = closer to the foci than chance), the mean over cells, and per replicate day the mean over that
  day's cells; with two or more days, mean +- SD over days. The pooled histogram is a picture, never the test unit.

Plain data; `ui.workspace.distances_dialog` draws it.
"""
from __future__ import annotations

import math

__all__ = ["DistancesError", "BIN_UM", "CHANCE_PER_LOCALISATION", "foci_count", "nearest_focus_um",
           "chance_points", "distance_payload"]

#: The histogram's bin width, um (a localisation precision of about 30 nm and a 106 nm pixel: 50 nm bins).
BIN_UM = 0.05
#: Uniform points drawn in the outline per localisation measured, for the chance distances.
CHANCE_PER_LOCALISATION = 20
SEED = 0


class DistancesError(RuntimeError):
    """Raised when no distance can be measured."""


def foci_count(foci: dict) -> tuple:
    """``(n_foci, n_cells)`` over ``{condition: {movie: {cell: [[x, y], ...]}}}``."""
    n_foci = n_cells = 0
    for movies in (foci or {}).values():
        for cells in movies.values():
            for points in cells.values():
                if points:
                    n_cells += 1
                    n_foci += len(points)
    return n_foci, n_cells


def nearest_focus_um(points_px, foci_px, um_per_px: float):
    """Each point's distance to its nearest focus, um (points and foci in image pixels)."""
    import numpy as np

    p = np.asarray(points_px, dtype=float).reshape(-1, 2)
    f = np.asarray(foci_px, dtype=float).reshape(-1, 2)
    if not len(f):
        raise DistancesError("no focus to measure to")
    d = np.sqrt(((p[:, None, :] - f[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    return d * float(um_per_px)


def chance_points(polygon_px, n: int, seed: int = SEED):
    """``n`` points uniformly inside the outline (rejection from its bounding box), image pixels."""
    import numpy as np
    from matplotlib.path import Path as MplPath

    poly = np.asarray(polygon_px, dtype=float).reshape(-1, 2)
    if len(poly) < 3:
        raise DistancesError("an outline needs three points")
    path = MplPath(poly)
    lo, hi = poly.min(axis=0), poly.max(axis=0)
    rng = np.random.default_rng(seed)
    out = []
    need = int(n)
    for _ in range(200):
        if need <= 0:
            break
        cand = rng.uniform(lo, hi, size=(max(64, 3 * need), 2))
        inside = cand[path.contains_points(cand)]
        out.append(inside[:need])
        need -= len(out[-1])
    if need > 0:
        raise DistancesError("the outline encloses no area")
    return np.vstack(out)


def _day_name(day) -> str:
    try:
        return f"replicate {int(day)}"
    except (TypeError, ValueError):
        return f"replicate {day}"


def distance_payload(project, foci: dict, *, labels: dict | None = None) -> dict:
    """Every condition with marked cells: observed and chance histograms (probability per bin), the counts, the
    per-cell excess and its mean, per day and over days. ``foci`` is ``{condition: {movie: {cell: [[x, y], ...]}}}``
    in the image pixels Images & tracks draws (0-based, `images_view`)."""
    import numpy as np

    from .canonical import without_attrs
    from .images_view import _condition

    labels = labels or {}
    conditions = []
    for condition, movies in (foci or {}).items():
        marked = {(m, c): pts for m, cells in movies.items() for c, pts in cells.items() if pts}
        if not marked:
            continue
        data = _condition(project, condition)
        per_px = float(data["params"].um_per_pixel)
        table = without_attrs(data["canonical"])
        outlines = {f"{movie}:{mesh.cell_id}": [[float(x) - 1.0, float(y) - 1.0] for x, y in mesh.polygon]
                    for movie, meshes in data["meshes"].items() for mesh in meshes}
        observed, chance, cells, skipped = [], [], [], []
        for k, ((movie, cell), pts) in enumerate(sorted(marked.items())):
            rows = table[(table["movie"].astype(str) == movie) & (table["cell"].astype(str) == cell)]
            if rows.empty:
                skipped.append(f"{cell}: no tracks")
                continue
            points = np.column_stack([rows["x_um"].to_numpy() / per_px - 1.0, rows["y_um"].to_numpy() / per_px - 1.0])
            obs = nearest_focus_um(points, pts, per_px)
            if cell in outlines:
                null = nearest_focus_um(chance_points(outlines[cell], CHANCE_PER_LOCALISATION * len(points),
                                                      seed=SEED + k), pts, per_px)
            else:
                null = np.array([])
                skipped.append(f"{cell}: no outline, so no chance distances")
            day = rows["day"].iloc[0] if "day" in rows.columns else None
            observed.append(obs)
            chance.append(null)
            cells.append({"cell": cell, "movie": movie, "day": day, "n_foci": len(pts), "n": int(len(obs)),
                          "median": float(np.median(obs)),
                          "chance_median": float(np.median(null)) if len(null) else None})
        if not cells:
            conditions.append({"key": condition, "label": str(labels.get(condition, condition)), "n_cells": 0,
                               "skipped": skipped})
            continue
        obs_all = np.concatenate(observed)
        null_all = np.concatenate([c for c in chance if len(c)]) if any(len(c) for c in chance) else np.array([])
        excess = [c["median"] - c["chance_median"] for c in cells if c["chance_median"] is not None]
        by_day = {}
        for c in cells:
            if c["chance_median"] is not None and c["day"] is not None and c["day"] == c["day"]:
                by_day.setdefault(c["day"], []).append(c["median"] - c["chance_median"])
        days = [{"day": _day_name(d), "n_cells": len(v), "excess": float(np.mean(v))}
                for d, v in sorted(by_day.items())]
        conditions.append({
            "key": condition, "label": str(labels.get(condition, condition)), "n_cells": len(cells),
            "n_localisations": int(len(obs_all)), "n_foci": int(sum(c["n_foci"] for c in cells)),
            "observed": obs_all.tolist(), "chance": null_all.tolist(),
            "median": float(np.median(obs_all)), "chance_median": float(np.median(null_all)) if len(null_all) else None,
            "excess_mean": float(np.mean(excess)) if excess else None,
            "excess_sd": float(np.std(excess, ddof=1)) if len(excess) > 1 else None,
            "days": days,
            "day_spread": ({"mean": float(np.mean([d["excess"] for d in days])),
                            "sd": float(np.std([d["excess"] for d in days], ddof=1)), "n_days": len(days)}
                           if len(days) >= 2 else None),
            "cells": cells, "skipped": skipped})
    measured = [c for c in conditions if c.get("n_cells")]
    if not measured:
        raise DistancesError("no marked cell has tracks. Mark foci in a cell that has tracks (Images & tracks, "
                             "zoom to a cell, switch on Mark foci and click).")
    top = max(max(c["observed"]) for c in measured)
    top = max(top, max((max(c["chance"]) for c in measured if c["chance"]), default=0.0))
    n_bins = max(4, int(math.ceil(top / BIN_UM)))
    edges = [BIN_UM * i for i in range(n_bins + 1)]
    for c in measured:
        c["hist"] = _probability(c["observed"], edges)
        c["chance_hist"] = _probability(c["chance"], edges) if c["chance"] else None
    columns = ["Condition", "Cells", "Foci", "Localisations", "Median (µm)", "Chance median (µm)",
               "Excess per cell (µm)", "Over replicates (µm)"]
    rows = []
    for c in conditions:
        if not c.get("n_cells"):
            rows.append([c["label"], "0", "", "", "", "", "", ""])
            continue
        ex = "" if c["excess_mean"] is None else (
            f"{c['excess_mean']:+.3f}" + ("" if c["excess_sd"] is None else f" ± {c['excess_sd']:.3f} SD"))
        sp = c["day_spread"]
        rows.append([c["label"], str(c["n_cells"]), str(c["n_foci"]), f"{c['n_localisations']:,}",
                     f"{c['median']:.3f}", "" if c["chance_median"] is None else f"{c['chance_median']:.3f}", ex,
                     "" if sp is None else f"{sp['mean']:+.3f} ± {sp['sd']:.3f} ({sp['n_days']} replicates)"])
    return {"edges": edges, "conditions": conditions, "columns": columns, "rows": rows,
            "note": ("Distance from each localisation of a marked cell's tracks to its nearest focus; dashed: uniform "
                     "points in the same outline. Excess is a cell's median minus its chance median (negative means "
                     "closer to the foci than chance), averaged over cells, with the SD over replicates.")}


def _probability(values, edges) -> list:
    import numpy as np

    v = np.asarray(values, dtype=float)
    counts, _ = np.histogram(np.clip(v, edges[0], edges[-1]), bins=edges)
    return (counts / max(1, len(v))).tolist()
