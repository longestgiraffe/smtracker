"""Spatial SLA: where confined molecules sit, in the capsule the other map methods draw.

v2's Stationary Localisation Analysis tab drew a "Confined Map" (`confTracks.m`, and
`confinedHeatmap` in `SMTracker.m`): every track of the condition laid on a model cell in
THREE COLOURS -- free tracks in MATLAB's default line blue (no colour given to `plot`, so
the axes' first ColorOrder entry, [0 0.447 0.741]), mixed-behaviour tracks ``'g-'`` (pure
green) and the dwelling segments ``'r'`` (pure red) -- and a "Heat map" view
(`plotHeatmap`, typeMotion 5, ``jet(1024)``) of the localisations of whichever of the
three were ticked. Asked for (13 Oct 2026): "the sla maps in spatial maps with adding the
color code of the matlab version". `SLA_COLOURS` is that code; the Spatial Map section draws
the three sets as density maps (its own maps, NOT `plotHeatmap`'s interp2 upscaling, a
recorded defect) and, in its Tracks view, as lines in those colours.

THE THREE SETS ARE v2's, AND THEY OVERLAP BY DESIGN:
- **free**: tracks with no dwell event (`confinement.confinement_events` class "free");
- **mixed**: every track holding a dwell of at least ``min_steps`` -- v2's ``mixed`` is
  ``tau >= minSteps`` (`confTracks.m`), so the fully confined tracks are IN it (this
  project's "mixed" + "confined" classes);
- **dwelling**: the localisations inside a dwell event, a SUBSET of the mixed tracks'
  localisations. v2 would not draw mixed and dwelling at once (`showConfinedTracks`
  unticks one when the other is ticked); here each is its own panel.

THE DETECTOR IS `smtracker.confinement`, UNCHANGED (the moving ROI of `dwellCount2.m`,
its one-gap quirk included), and the radius and the minimum dwell are choices, not
measurements: a free random walk revisits its own neighbourhood, so the detector flags
"confinement" on purely Brownian tracks. `spatial_map_view.compute_spatial_map_sla` runs
the identical detector on matched free tracks and every caption carries that chance rate.

No edit to `confinement`, `spatialsqd` or `spatialappd`: this module only imports them.
The geometry config is `spatialappd.SpatialAppDConfig` -- the same capsule, bins and long
axis, nothing method-specific -- reused rather than copied.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .canonical import without_attrs
from .confinement import label_confinement
from .spatialappd import SpatialAppDConfig
from .spatialsqd import (
    _check_weighting,
    contributor_labels,
    contributor_neff_grids,
    fold_counts,
    local_half_width,
    long_axis_x,
    row_cell_keys,
    step_block,
    track_weights,
)

__all__ = ["SLA_SETS", "SLA_TITLES", "SLA_COLOURS", "label_sla", "maps_from_labelled_sla",
           "spatial_sla_map", "track_polylines", "V2_RADIUS_RULE", "V2_MIN_STEPS"]

#: The three sets, in v2's checkbox order.
SLA_SETS = ("free", "mixed", "dwelling")
SLA_TITLES = {"free": "free tracks", "mixed": "mixed-behaviour tracks",
              "dwelling": "dwelling segments"}
#: v2's colours (`confTracks.m`): `plot` with no colour takes MATLAB's first default line
#: colour, [0 0.4470 0.7410]; ``'g-'`` is [0 1 0]; ``'r'`` is [1 0 0].
SLA_COLOURS = {"free": "0072BD", "mixed": "00FF00", "dwelling": "FF0000"}

#: v2's defaults: the "CUSTOM" radius starts at the pixel size (`initializeVars.m`,
#: ``customCut = pixel_size/1e3``) and the dwell list at 5 steps (`popPopUpMinStepsDwell`).
V2_RADIUS_RULE = "the camera pixel"
V2_MIN_STEPS = 5


def label_sla(projected: pd.DataFrame, *, radius_um: float, min_steps: int,
              gap_mode: str = "legacy") -> pd.DataFrame:
    """``projected`` (sorted by track and frame) with ``free``, ``mixed`` and
    ``dwelling`` booleans per localisation, from `confinement.label_confinement`.
    ``.attrs["confinement"]`` carries the detector's settings and the dataset fractions."""
    labelled = label_confinement(projected, radius_um=radius_um, min_steps=min_steps,
                                 gap_mode=gap_mode)
    klass = labelled["track_class"].to_numpy()
    labelled["free"] = klass == "free"
    labelled["mixed"] = (klass == "mixed") | (klass == "confined")
    labelled["dwelling"] = labelled["confined"].to_numpy(bool)
    return labelled


def _capsule_xy(labelled: pd.DataFrame, cfg, projected=None):
    """Every row's position on the drawn capsule (µm), and which rows are inside it."""
    l_norm = labelled["l_norm"].to_numpy(float)
    d_norm = labelled["d_norm"].to_numpy(float)
    keep = np.isfinite(d_norm) & (np.abs(d_norm) <= 1.0) & np.isfinite(l_norm)
    source = labelled if projected is None else projected
    cells = (row_cell_keys(labelled, source)[keep]
             if getattr(cfg, "long_axis", "normalised") != "normalised" else None)
    x = long_axis_x(l_norm[keep], cfg, cells=cells,
                    geometry=source.attrs.get("geometry"))
    y = d_norm[keep] * local_half_width(x, cfg)
    return x, y, keep


def track_polylines(labelled: pd.DataFrame, cfg, *, projected=None,
                    decimals: int = 4) -> dict:
    """``{set: [[(u, v), ...], ...]}``: each set's tracks as polylines in BOX FRACTIONS of
    the drawn capsule (u along it from the left pole, v from the bottom wall), what the
    Tracks view paints. A dwelling segment is a run of consecutive dwelling localisations
    of one track; a polyline is broken where a localisation lies outside the capsule, so a
    line never jumps across the cell. Drawn unfolded: a mirrored map folds its counts, and
    a folded track would cross itself."""
    x, y, keep = _capsule_xy(labelled, cfg, projected)
    rows = without_attrs(labelled)
    u = np.full(len(rows), np.nan)
    v = np.full(len(rows), np.nan)
    u[keep] = x / cfg.cell_length_um
    v[keep] = (y + cfg.cell_width_um / 2.0) / cfg.cell_width_um
    uid = rows["uid"].to_numpy()
    starts = np.flatnonzero(np.r_[True, uid[1:] != uid[:-1]])
    ends = np.r_[starts[1:], len(uid)]
    out = {name: [] for name in SLA_SETS}
    flags = {name: rows[name].to_numpy(bool) for name in SLA_SETS}

    def runs(mask, lo, hi):
        i = lo
        while i < hi:
            if not mask[i]:
                i += 1
                continue
            j = i
            while j < hi and mask[j]:
                j += 1
            yield i, j
            i = j

    for lo, hi in zip(starts, ends):
        for name in SLA_SETS:
            mask = flags[name] & np.isfinite(u)
            for a, b in runs(mask, lo, hi):
                if b - a >= 2:
                    out[name].append([(round(float(u[k]), decimals),
                                       round(float(v[k]), decimals)) for k in range(a, b)])
    return out


def maps_from_labelled_sla(labelled: pd.DataFrame, cfg, *, mirrored: bool = False,
                           weighting: str = "steps", with_steps: bool = False,
                           projected=None) -> dict:
    """Per-set localisation-density grids of one labelled table (`label_sla`) -- the SLA
    analogue of `spatialappd.maps_from_projected_appd`, in its output shape, so
    `spatial_map_view.finish_spatial_panels` and the day pages take it unchanged.

    Weights are hard 0/1 per localisation (a localisation is in a set or not); with
    ``weighting="track"`` each localisation weighs 1 / its track's localisations. Adds
    ``"tracks"`` (`track_polylines`) for the Tracks view."""
    _check_weighting(weighting)
    projected = labelled if projected is None else projected
    x, y, keep = _capsule_xy(labelled, cfg, projected)
    per_track = track_weights(labelled)[keep] if weighting == "track" else None
    tracks, cells = contributor_labels(labelled, projected)
    tracks, cells = tracks[keep], cells[keep]
    xe, ye = cfg.edges()

    grids, weights, step_grids, step_weights, drawn_by_label = {}, {}, {}, {}, {}
    neff_track, neff_cell = {}, {}
    for name in SLA_SETS:
        weight = labelled[name].to_numpy(bool)[keep].astype(float)
        drawn_by_label[name] = weight if per_track is None else weight * per_track
        counts = np.histogram2d(x, y, bins=[xe, ye], weights=weight)[0].T
        if mirrored:
            counts = fold_counts(counts)
        neff_track[name], neff_cell[name] = contributor_neff_grids(
            x, y, weight, tracks, cells, xe, ye, mirrored=mirrored)
        step_grids[name] = counts
        step_weights[name] = float(weight.sum())
        if per_track is None:
            grids[name], weights[name] = counts, step_weights[name]
        else:
            drawn = weight * per_track
            tcounts = np.histogram2d(x, y, bins=[xe, ye], weights=drawn)[0].T
            if mirrored:
                tcounts = fold_counts(tcounts)
            grids[name], weights[name] = tcounts, float(drawn.sum())

    out = {
        "grids": grids, "weights": weights,
        "step_grids": step_grids, "step_weights": step_weights,
        "neff_track": neff_track, "neff_cell": neff_cell,
        "weighting": weighting, "labels": {k + 1: n for k, n in enumerate(SLA_SETS)},
        "mirrored": mirrored, "n_localisations": int(keep.sum()),
        "weight_unit": "tracks" if weighting == "track" else "localisations",
        "tracks": track_polylines(labelled, cfg, projected=projected),
    }
    if with_steps:
        out["steps"] = step_block(labelled.iloc[np.flatnonzero(keep)], x, y,
                                  drawn_by_label, cfg)
    return out


def spatial_sla_map(projected: pd.DataFrame, *, bin_long_nm: float, bin_short_nm: float,
                    smooth_bins: float, cell_length_um: float, cell_width_um: float,
                    radius_um: float, min_steps: int, mirrored: bool = False,
                    long_axis: str = "normalised") -> dict:
    """The GUI-facing entry point: label once, then draw. Returns the maps plus
    ``"labelled"`` (the day pages draw from it, so nothing is detected twice) and
    ``"config"``."""
    def even(value: float) -> int:
        count = round(value)
        return max(2, count + (count % 2))

    cfg = SpatialAppDConfig(
        cell_length_um=cell_length_um, cell_width_um=cell_width_um,
        n_bins_long=even(cell_length_um * 1000.0 / bin_long_nm),
        n_bins_short=even(cell_width_um * 1000.0 / bin_short_nm),
        smooth_bins=smooth_bins, long_axis=long_axis)
    labelled = label_sla(projected, radius_um=radius_um, min_steps=min_steps)
    out = maps_from_labelled_sla(labelled, cfg, mirrored=mirrored, projected=projected)
    out["config"] = cfg
    out["labelled"] = labelled
    return out
