"""The legacy data loader, and the filter inventory it deposits.

The ingest path is imported READ-ONLY. Nothing here modifies it. Legacy mode is a
display of the old method over the current data; it is not a second ingest.

The day map is passed explicitly and never defaulted. ``load_utrack_project`` derives a
day from the absolute series decade, ``(series - 1) // 10 + 1``. Two groups skip
decades, so that default disagrees with the lab's own naming rule:

* one group has series 001-010, 011-020 and 031-040, so the default gives days
  1, 2 and 4 where the naming rule gives 1, 2 and 3;
* another group has series 001-009, 031-040 and 041-050, so the default gives 1, 4 and 5.

This is open item B12. The POOLED legacy fit is unaffected, because pooling ignores the
day entirely. The day diagnostic is affected, which is why the map is passed and the
choice is recorded in the provenance.

The filter inventory reports two counts per filter, never their difference, in the order
the filters are applied.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..canonical import to_canonical
from ..grouping import (UTRACK_GLOB, day_map_from_series_blocks,
                        discover_utrack_project, load_utrack_project)
from ..cellmesh import assign_cells, read_project_cell_meshes
from ..observables import squared_displacements
from ..utrack_meta import acquisition_params_from_project

__all__ = ["DAY_MAP_AUTHORITY", "load_legacy_condition", "LegacyLoadError"]

#: Recorded in every provenance file this module writes.
DAY_MAP_AUTHORITY = (
    "grouping.day_map_from_series_blocks, the lab's own series-naming rule: one decade "
    "of series numbers is one replicate day, and the decades PRESENT are numbered in "
    "order of occurrence. NOT load_utrack_project's default of (series - 1) // 10 + 1, "
    "which numbers by absolute decade and disagrees for one group (days 1, 2, 4) and "
    "another group (days 1, 4, 5). Open item B12. The pooled fit is unaffected; the day "
    "diagnostic is not.")


class LegacyLoadError(RuntimeError):
    """Raised when a legacy condition cannot be loaded honestly."""


def load_legacy_condition(condition_root, *, condition: str, min_localisations: int = 5,
                          lag_frames: int = 1) -> dict:
    """Load one condition the legacy way. Returns steps, tracks, and the inventory.

    The legacy analysis keeps only tracks lying entirely inside a segmented cell
    outline (``cellTracks.m``: ``if all(points_in)``), so that filter is applied here and
    counted. Every count is reported as a pair, N in and N out.
    """
    from pathlib import Path

    root = Path(condition_root)
    movies = [str(m) for m, _ in discover_utrack_project(str(root), pattern=UTRACK_GLOB)]
    if not movies:
        raise LegacyLoadError(f"no u-track movies under {root}")
    day_map, spans = day_map_from_series_blocks(movies)

    raw = load_utrack_project(root, condition=condition, day=dict(day_map))
    params = acquisition_params_from_project(root)
    meshes = read_project_cell_meshes(root, allow_unsegmented=True)

    inventory = []

    def _n_tracks(frame):
        """Count tracks whichever key the stage carries. The RAW table keys a track by
        movie plus track_id; `uid` only exists after canonicalisation. Counting `uid` on
        a raw frame raises, and counting `track_id` alone would merge tracks that share
        an id across movies."""
        if "uid" in frame.columns:
            return int(frame["uid"].nunique())
        return int(frame.groupby(["movie", "track_id"], observed=True).ngroups)

    def record(name, frame_in, frame_out, note):
        inventory.append({
            "order": len(inventory) + 1, "filter": name,
            "tracks_in": _n_tracks(frame_in),
            "tracks_out": _n_tracks(frame_out),
            "localisations_in": int(len(frame_in)),
            "localisations_out": int(len(frame_out)),
            "note": note,
        })

    assigned = assign_cells(raw, meshes, drop_unassigned=True, require_all_inside=True)
    record("cell contour filter", raw, assigned,
           "cellTracks.m keeps a track only if EVERY localisation lies inside a "
           "segmented outline; a track that strays once is dropped whole")

    canon = to_canonical(assigned, params)
    canon = canon[canon["cell"].notna()]
    record("canonical conversion", assigned, canon,
           "pixels to um and frames to seconds; rows without a cell are dropped")

    counts = canon.groupby("uid").size()
    keep = set(counts[counts >= min_localisations].index)
    long_enough = canon[canon["uid"].isin(keep)]
    record(f"minimum track length, {min_localisations} localisations", canon,
           long_enough,
           "u-track already applies minTrackLen 5, so this is expected to be a no-op "
           "on this data and is kept as a stated safety net")

    steps = squared_displacements(long_enough, lag_frames=lag_frames)
    column = "sq_disp_um2" if "sq_disp_um2" in steps.columns else steps.columns[-1]

    per_day = {}
    for day, part in steps.groupby("day"):
        values = part[column].to_numpy(dtype=float)
        per_day[int(day)] = (values, int(part["uid"].nunique()))

    return {
        "condition": condition,
        "sq_disp_um2": steps[column].to_numpy(dtype=float),
        "n_tracks": int(long_enough["uid"].nunique()),
        "n_steps": int(len(steps)),
        "n_cells": int(long_enough["cell"].nunique()),
        "tau_s": float(params.tau_s),
        "per_day": per_day,
        "day_map": dict(day_map),
        "day_spans": spans,
        "day_map_authority": DAY_MAP_AUTHORITY,
        "filter_inventory": pd.DataFrame(inventory),
        "min_localisations": int(min_localisations),
        "lag_frames": int(lag_frames),
    }
