"""Spatial appD: where a per-track apparent-diffusion population sits inside
the cell, in the SAME capsule frame `spatialsqd` draws -- so the Map
Configuration dialog can offer this as a second computational method over
the one existing grid/matrix, rather than a second kind of picture.

**No edit to `smtracker.maps`, `smtracker.appd`, or `smtracker.spatialsqd`.**
This module only IMPORTS from them (read-only, the same way `plots.py`
already does), by explicit instruction: those modules are shared with a
colleague and stay untouched by this feature. `local_half_width` and
`fold_counts` are reused from `spatialsqd` rather than re-derived -- one
source of truth for the capsule geometry.

**The split is a fixed physical threshold, not a per-condition quantile.**
`maps.labels_from_appd` already exists and is already used elsewhere
(`comparison.spatial_maps(source="appd")`), but it splits each condition at
its OWN median track D -- so "slow" in one condition can be a different D
range than "slow" in another, which is not the same kind of claim SQD's
populations make (an absolute posterior over a fixed pair of D's). This
module instead reuses `D_THRESHOLD_UM2_S = 0.15`, the same value already
established in `smtracker/spatial/populations.py` for calling a molecule
"slow" -- a documented reuse of an existing convention, not a new number.

**No step-position choice, and that is a fact about the estimator, not a
simplification.** `apparent_D` is a per-TRACK scalar; the labelled table is
therefore one row per LOCALISATION, each already carrying its own
`l_norm`/`d_norm` -- there are no two endpoints to choose between the way a
jump has. This is exactly the case `spatialsqd.maps_from_projected`'s own
`step_position="A_start"` branch already handles (it reads `l_norm`/`d_norm`
straight off the table with no re-projection), so the same geometry line
applies here unconditionally.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .spatialsqd import (
    _check_long_axis,
    axis_profiles,
    _check_weighting,
    contributor_labels,
    contributor_neff_grids,
    fold_counts,
    local_half_width,
    long_axis_x,
    row_cell_keys,
    step_block,
    track_weights,
    upper_envelope,
)

__all__ = [
    "D_THRESHOLD_UM2_S",
    "APPD_POPULATION_NAMES",
    "SpatialAppDConfig",
    "appd_crossings",
    "fit_appd_split",
    "maps_from_projected_appd",
    "spatial_appd_map",
]

#: The panel name of each fitted population, slowest first, by population count.
APPD_POPULATION_NAMES = {2: ("slow", "fast"), 3: ("slow", "middle", "fast")}

#: `smtracker/spatial/populations.py`'s own `edges=(0.0, 0.15, np.inf)` --
#: restated here as a plain float rather than imported, since importing
#: across the two spatial systems for one shared constant would be a new
#: coupling this feature does not need. Documented so the two cannot quietly
#: drift apart without someone reading both files.
D_THRESHOLD_UM2_S = 0.15

STATE_TITLE = {"slow": "slow (D < 0.15 µm²/s)",
              "fast": "fast (D ≥ 0.15 µm²/s)"}


@dataclass(frozen=True)
class SpatialAppDConfig:
    """The geometry `spatialsqd.SpatialSQDConfig` also carries -- every field
    required, no defaults, for the same reason: a bin is a resolution claim.

    No `pole_margin`/`n_profile_bins`/`n_pop`/`method`/`weighting`: this
    module builds no radial profile (the GUI payload never reads one) and
    fits no mixture model, so those fields would be dead weight copied from
    a sibling that does something this one does not.
    """

    cell_length_um: float
    cell_width_um: float
    n_bins_long: int
    n_bins_short: int
    smooth_bins: float
    #: How each cell's long axis is laid onto the drawn capsule (`spatialsqd.long_axis_x`).
    long_axis: str = "normalised"

    def __post_init__(self) -> None:
        _check_long_axis(self.long_axis, self)
        if self.n_bins_long % 2 or self.n_bins_short % 2:
            raise ValueError(
                "n_bins_long and n_bins_short must be even: folding pairs each bin with its reflection, "
                "and an odd count leaves a centre row paired with itself "
                f"(got {self.n_bins_long}x{self.n_bins_short})"
            )

    @property
    def bin_long_um(self) -> float:
        return self.cell_length_um / self.n_bins_long

    @property
    def bin_short_um(self) -> float:
        return self.cell_width_um / self.n_bins_short

    def edges(self) -> tuple[np.ndarray, np.ndarray]:
        half = self.cell_width_um / 2.0
        return (np.linspace(0.0, self.cell_length_um, self.n_bins_long + 1),
                np.linspace(-half, half, self.n_bins_short + 1))

    def centres(self) -> tuple[np.ndarray, np.ndarray]:
        xe, ye = self.edges()
        return 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])


def appd_crossings(D, fractions, n_steps: int) -> dict:
    """Where a fitted appD mixture's populations CROSS: the old program's own split.

    The legacy speed map took a "Minimum D" and "Maximum D" typed by hand, and its
    on-screen hint said how to choose them: "Set the boundaries according to the
    intersection of the individual populations". This computes those intersections
    instead of reading them off a plot.

    Each population is ``f_k * Gamma(x; n_steps, D_k / n_steps)``, and the components
    share their shape, so the ``x^(n-1)`` and ``Gamma(n)`` terms cancel between any two
    and their log ratio is a straight line in x. Two neighbours therefore cross ONCE, in
    closed form,

        x = [ln(f_i / f_j) + n ln(D_j / D_i)] / [n (1/D_i - 1/D_j)],

    and the most-likely population at every x is the upper envelope of those lines.
    A cut is exactly where the posterior of the two neighbours is 0.5 -- the appD
    analogue of SQD's hard assignment.

    A population can be the most likely NOWHERE (a small middle component of three, or
    one whose region lies below zero): it then gets no window, and is named in
    ``never_most_likely`` so its empty map is explained, not mistaken for an absence.

    Returns ``{"cuts": [...] ascending, "windows": [population index per window,
    slowest first], "never_most_likely": [population index, ...]}``, population
    indices counted by ascending D.

    The immobile population of exact-zero tracks (a D of 0, `appd_gamma`'s module head,
    ruled 4 Nov 2026) owns the apparent D of exactly 0 and nothing else: its window comes
    first, ended by a cut at 0, and the moving populations' cuts follow (`spatialsqd.
    sqd_crossings`' rule). A value exactly on that cut is the immobile population's
    (`_window_of`).
    """
    D = np.asarray(D, dtype=float).ravel()
    f = np.asarray(fractions, dtype=float).ravel()
    if D.size != f.size or D.size < 2:
        raise ValueError("need at least two populations, one fraction each")
    order = np.argsort(D)
    if D[order[0]] == 0:
        if D.size == 2:
            return {"cuts": [0.0], "windows": [0, 1], "never_most_likely": []}
        moving = appd_crossings(D[order[1:]], f[order[1:]], n_steps)
        return {"cuts": [0.0, *moving["cuts"]], "windows": [0, *(k + 1 for k in moving["windows"])],
                "never_most_likely": [k + 1 for k in moving["never_most_likely"]]}
    if np.any(D <= 0) or np.any(f <= 0):
        raise ValueError(f"D and fractions must be positive, got D={D}, f={f}")
    order = np.argsort(D)
    n = float(n_steps)
    # ln(f_k Gamma(x; n, D_k/n)) = const(x) + intercept_k + slope_k * x; the shared
    # envelope (`spatialsqd.upper_envelope`) does the rest, as for SQD.
    return upper_envelope(np.log(f[order]) - n * np.log(D[order]), -n / D[order])


def split_values(projected: pd.DataFrame, n_steps: int = 4) -> tuple:
    """``(values, tau_s)``: the sub-track apparent D's the map's fit and its Best are made on (`appd.apparent_D`,
    sub-tracks of ``n_steps``), and the frame interval they were divided by -- what the exact-zero rule's lattice
    test needs (`appd_gamma.zero_tracks`)."""
    values, tau, _offsets = split_values_with_offsets(projected, n_steps)
    return values, tau


def split_values_with_offsets(projected: pd.DataFrame, n_steps: int = 4, calibration=None) -> tuple:
    """`split_values` and, with a per-spot ``calibration`` (6 Nov 2026, the map's own fit Per spot), each sub-track's
    own noise offset (`appd.NOISE_OFFSET_COLUMN`, `appd_gamma`'s ``offset``): ``(values, tau_s, offsets)``, the
    offsets None without one."""
    from .appd import NOISE_OFFSET_COLUMN, apparent_D

    table = apparent_D(projected, n_steps=n_steps, calibration=calibration)
    offsets = None if calibration is None else table[NOISE_OFFSET_COLUMN].to_numpy(float)
    return table["appd_um2_per_s"].to_numpy(float), table.attrs.get("tau_s"), offsets


def _shifted(D, offsets) -> np.ndarray:
    """Where the populations cross on the RAW apparent D a track is labelled by, under per-spot precision: each
    moving population's D plus the median noise offset (the section's histogram's rule, `appd_view.appd_condition`);
    an immobile population (D 0) stays at 0. Unchanged without offsets."""
    D = np.asarray(D, dtype=float)
    if offsets is None:
        return D
    return np.where(D > 0, D + float(np.median(offsets)), D)


def fit_appd_split(projected: pd.DataFrame, *, n_pop: int, n_steps: int = 4, calibration=None) -> dict:
    """Fit the appD gamma mixture on this table's sub-track apparent D's and return
    where its populations cross -- the SAME fit the Apparent D bubble plot draws
    (`appd_gamma.fit_appd_gamma`, sub-tracks of `n_steps`), so the map and that plot
    cannot disagree about what the populations are.

    EXACT-ZERO SUB-TRACKS (ruled 4 Nov 2026, `appd_gamma`'s module head): the fit's own
    rule -- where they are an immobile population the first population is it (D 0, its share
    counted), its window ends at a cut at 0 (`appd_crossings`), and the split carries the
    decision (``zero_jumps``) for the map to say.

    ``calibration`` (6 Nov 2026, the map's own fit Per spot): each sub-track's own noise offset in the mixture
    (`appd_gamma`'s ``offset``), so D has the localisation error taken out; the cuts where the populations cross on
    the raw values a track is labelled by, at the median offset (`_shifted`). ``"per_spot"`` says so."""
    from .appd_gamma import fit_appd_gamma

    if n_pop not in APPD_POPULATION_NAMES:
        raise ValueError(f"n_pop must be one of {tuple(APPD_POPULATION_NAMES)}, got {n_pop}")
    values, tau, offsets = split_values_with_offsets(projected, n_steps, calibration)
    fit = fit_appd_gamma(values, n_steps=n_steps, n_pop=n_pop, tau_s=tau, noun="sub-track", offset=offsets)
    D = np.asarray(fit.D, dtype=float)
    fractions = np.asarray(fit.fraction, dtype=float)
    order = np.argsort(D)
    split = appd_crossings(_shifted(D, offsets), fractions, n_steps)
    zeros = getattr(fit, "zero_tracks", None)
    return {"D": D[order].tolist(), "fractions": fractions[order].tolist(),
            "n_pop": int(n_pop), "n_steps": int(n_steps), "bic": float(fit.bic), **split,
            **({"zero_jumps": zeros.to_dict()} if zeros is not None else {}),
            **({"per_spot": True} if offsets is not None else {})}


def _window_of(cuts, values) -> np.ndarray:
    """Each value's window among ascending ``cuts`` (a value on a cut goes to the window on its right), except an
    apparent D of exactly 0 against a cut at 0: that is the immobile population's window, the first
    (`appd_crossings`)."""
    values = np.asarray(values, dtype=float)
    window = np.searchsorted(np.asarray(cuts, dtype=float), values, side="right")
    if len(cuts) and float(cuts[0]) == 0:
        window = np.where(values == 0, 0, window)
    return window


def split_at_D(projected: pd.DataFrame, D, *, min_steps: int = 4, calibration=None) -> dict:
    """The fitted split with the populations' D HELD at ``D`` (the Apparent Diffusion section's, `map_fit_source`; 4
    Nov 2026): the section's own estimator on this table -- one apparent D per track over all its steps
    (`appd.track_appd`, tracks of ``min_steps`` or more), each its own gamma shape -- with only the fractions fitted
    (`appd_gamma.fractions_at`), cut where the populations cross at the median track's step count, as the section's
    histogram cuts them (`appd_view.appd_condition`). The same keys as `fit_appd_split`'s.

    ``calibration`` (6 Nov 2026): the section's fit was PER SPOT -- each track's own noise offset in the fractions
    (`appd_gamma`'s ``offset``), and the cuts where the populations cross for the median track's offset, as the
    section's histogram draws them."""
    from .appd import NOISE_OFFSET_COLUMN, track_appd
    from .appd_gamma import fractions_at

    D = np.sort(np.asarray(D, dtype=float).ravel())
    if len(D) not in APPD_POPULATION_NAMES:
        raise ValueError(f"an apparent-D map splits at {tuple(APPD_POPULATION_NAMES)} populations, got {len(D)}")
    tracks = track_appd(projected, min_steps=int(min_steps), calibration=calibration)
    x = tracks["appd_um2_per_s"].to_numpy(float)
    n = tracks["n_steps"].to_numpy(float)
    o = None if calibration is None else tracks[NOISE_OFFSET_COLUMN].to_numpy(float)
    fractions = fractions_at(x, n, D, offset=o)
    if fractions is None:
        raise ValueError("no track's apparent D can be read at the section's D")
    cut_n = int(np.median(n))
    split = appd_crossings(_shifted(D, o), np.maximum(np.asarray(fractions, dtype=float), 1e-12), cut_n)
    return {"D": D.tolist(), "fractions": [float(f) for f in fractions], "n_pop": int(len(D)), "n_steps": cut_n,
            "bic": None, **split, **({"per_spot": True} if o is not None else {})}


def _labels_from_appd_threshold(projected: pd.DataFrame, *,
                                d_threshold_um2_s: float, n_steps: int = 4,
                                mode: str = "split") -> pd.DataFrame:
    """Per-LOCALISATION labels, split at a fixed per-track apparent D -- one cut, two
    populations (`_labels_from_appd_cuts`, which is the one implementation)."""
    out = _labels_from_appd_cuts(projected, cuts=[d_threshold_um2_s], windows=[0, 1],
                                 n_steps=n_steps, mode=mode)
    out.attrs["label_source"] = "appd_threshold"
    out.attrs["d_threshold_um2_s"] = d_threshold_um2_s
    return out


def _labels_from_appd_cuts(projected: pd.DataFrame, *, cuts, windows, n_steps: int = 4,
                           mode: str = "split") -> pd.DataFrame:
    """Per-LOCALISATION labels: each track's mean apparent D placed among `cuts`.

    `windows[i]` is the population (0-based, by ascending D) that owns the i-th window
    between cuts; the state written is that population + 1. Built from
    `appd.apparent_D` directly, exactly as `maps.labels_from_appd` does, so this is the
    same estimator under a different cut rather than a new one.
    """
    from .appd import apparent_D  # existing, unmodified, read-only

    cuts = [float(c) for c in cuts]
    if len(windows) != len(cuts) + 1 or cuts != sorted(cuts):
        raise ValueError(f"{len(cuts)} ascending cuts need {len(cuts) + 1} windows, "
                         f"got cuts={cuts}, windows={windows}")
    table = apparent_D(projected, n_steps=n_steps, mode=mode)
    if table.empty:
        raise ValueError("no sub-track yielded an apparent D")
    per_track = table.groupby("uid")["appd_um2_per_s"].mean()
    window = _window_of(cuts, per_track.to_numpy(float))
    state = pd.Series(np.asarray(windows)[window] + 1, index=per_track.index)
    out = projected.copy()
    out["state"] = out["uid"].map(state)
    out = out[out["state"].notna()].copy()
    out.attrs["label_source"] = "appd_cuts"
    out.attrs["appd_cuts"] = cuts
    return out


def _labels_from_appd_ranges(projected: pd.DataFrame, *, ranges, n_steps: int = 4) -> pd.DataFrame:
    """Per-LOCALISATION labels from apparent-D ranges set BY HAND (1 Nov 2026): each track's mean apparent D -- the
    quantity `_labels_from_appd_cuts` places among fitted cuts -- in the range ``[low, high)`` (µm²/s, ascending,
    non-overlapping: `spatialsqd.check_range_values`) that holds it; a track in no range is dropped and counted
    (``attrs["n_outside_tracks"]``)."""
    from .appd import apparent_D  # existing, unmodified, read-only

    table = apparent_D(projected, n_steps=n_steps)
    if table.empty:
        raise ValueError("no sub-track yielded an apparent D")
    per_track = table.groupby("uid")["appd_um2_per_s"].mean()
    values = per_track.to_numpy(float)
    state = np.full(len(values), np.nan)
    for index, (lo, hi) in enumerate(ranges):
        state[(values >= lo) & (values < hi)] = index + 1
    states = pd.Series(state, index=per_track.index)
    out = projected.copy()
    out["state"] = out["uid"].map(states)
    out = out[out["state"].notna()].copy()
    out.attrs.update(projected.attrs)
    out.attrs["label_source"] = "appd_ranges"
    out.attrs["appd_ranges"] = [list(r) for r in ranges]
    out.attrs["n_outside_tracks"] = int(np.isnan(state).sum())
    return out


def maps_from_projected_appd(projected: pd.DataFrame, cfg: SpatialAppDConfig, *,
                             d_threshold_um2_s: float = D_THRESHOLD_UM2_S,
                             n_steps: int = 4, mirrored: bool = False,
                             weighting: str = "steps", with_steps: bool = False,
                             split: dict | None = None, ranges=None, all_tracks: bool = False) -> dict:
    """Per-population localisation-density grids for ONE already-loaded,
    projected table -- the appD analogue of `spatialsqd.maps_from_projected`.

    Every localisation belongs to exactly ONE population (its track's own
    mean apparent D against the fixed threshold), so the weight is a hard
    0/1 indicator throughout -- there is no posterior to split it with the
    way SQD's soft assignment does.

    `weighting="track"` makes every track count once (each localisation weighs
    1 / its track's localisations); `step_grids`/`step_weights` keep the plain
    localisation counts. `with_steps=True` adds `"steps"`, the per-localisation
    record of which track went into which bin (`spatialsqd.step_block`).
    """
    _check_weighting(weighting)
    # `split` (from `fit_appd_split`) is the fitted rule: cut where the populations
    # cross, one panel per population. Without it, the fixed threshold, two panels.
    # `ranges` (µm²/s, set BY HAND): one panel per range, nothing fitted.
    if ranges is not None:
        from .spatialsqd import RANGE_KINDS, check_range_values

        ranges = check_range_values(ranges, unit="um2/s")
        labelled = _labels_from_appd_ranges(projected, ranges=ranges, n_steps=n_steps)
        names = RANGE_KINDS[:len(ranges)]
    elif split is None:
        labelled = _labels_from_appd_threshold(
            projected, d_threshold_um2_s=d_threshold_um2_s, n_steps=n_steps)
        names = APPD_POPULATION_NAMES[2]
    else:
        labelled = _labels_from_appd_cuts(projected, cuts=split["cuts"],
                                          windows=split["windows"], n_steps=n_steps)
        names = APPD_POPULATION_NAMES[int(split["n_pop"])]

    l_norm = labelled["l_norm"].to_numpy(float)
    d_norm = labelled["d_norm"].to_numpy(float)
    state = labelled["state"].to_numpy()
    per_track = track_weights(labelled) if weighting == "track" else None

    xe, ye = cfg.edges()
    keep = np.isfinite(d_norm) & (np.abs(d_norm) <= 1.0) & np.isfinite(l_norm)
    cells = (row_cell_keys(labelled, projected)[keep]
             if getattr(cfg, "long_axis", "normalised") != "normalised" else None)
    x = long_axis_x(l_norm[keep], cfg, cells=cells,
                    geometry=projected.attrs.get("geometry"))
    y = d_norm[keep] * local_half_width(x, cfg)
    state = state[keep]
    if per_track is not None:
        per_track = per_track[keep]
    # appD rows ARE the projected rows (localisations), so the cell is on the row
    tracks, cells = contributor_labels(labelled, labelled)
    tracks, cells = tracks[keep], cells[keep]

    labels = {k + 1: name for k, name in enumerate(names)}
    grids, weights, step_grids, step_weights, drawn_by_label = {}, {}, {}, {}, {}
    neff_track, neff_cell = {}, {}
    # EVERY TRACK (3 Nov 2026, asked: "the all jumps is missing" for the apparent-D and HMM maps): every
    # localisation at weight 1, the map before any split -- ``"jumps"``, the column SQD's map of every jump fills
    columns = ([(0, "jumps")] if all_tracks else []) + list(labels.items())
    for index, name in columns:
        weight = np.ones(len(state)) if index == 0 else (state == index).astype(float)
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
        "grids": grids,
        "weights": weights,
        "step_grids": step_grids,
        "step_weights": step_weights,
        "neff_track": neff_track,
        "neff_cell": neff_cell,
        "weighting": weighting,
        "labels": labels,
        "mirrored": mirrored,
        "d_threshold_um2_s": None if split is not None or ranges is not None else d_threshold_um2_s,
        "split": split,
        "ranges": None if ranges is None else [list(r) for r in ranges],
        "n_outside_tracks": int(labelled.attrs.get("n_outside_tracks", 0)),
        "n_localisations": int(keep.sum()),
        "weight_unit": "tracks" if weighting == "track" else "localisations",
    }
    # where each population sits along and across the cell (`spatialsqd.axis_profiles`)
    out["profiles"] = axis_profiles(x, y, drawn_by_label, cfg)
    if with_steps:
        out["steps"] = step_block(labelled.iloc[np.flatnonzero(keep)], x, y,
                                  drawn_by_label, cfg)
    return out


def spatial_appd_map(projected: pd.DataFrame, *, bin_long_nm: float,
                     bin_short_nm: float, smooth_bins: float,
                     cell_length_um: float, cell_width_um: float,
                     mirrored: bool = False, n_steps: int = 4,
                     d_threshold_um2_s: float = D_THRESHOLD_UM2_S,
                     split: dict | None = None, long_axis: str = "normalised", ranges=None,
                     all_tracks: bool = False) -> dict:
    """The GUI-facing entry point: one projected table in, localisation-
    density grids out. Mirrors `spatialsqd.spatial_sqd_map`'s own shape --
    the bin is given in nanometres for the same reason (it is compared
    against the camera pixel and the localisation precision), and rounded to
    an even bin count since folding pairs each bin with its reflection.
    """
    def even(value: float) -> int:
        count = round(value)
        return max(2, count + (count % 2))

    cfg = SpatialAppDConfig(
        cell_length_um=cell_length_um, cell_width_um=cell_width_um,
        n_bins_long=even(cell_length_um * 1000.0 / bin_long_nm),
        n_bins_short=even(cell_width_um * 1000.0 / bin_short_nm),
        smooth_bins=smooth_bins, long_axis=long_axis)
    out = maps_from_projected_appd(
        projected, cfg, d_threshold_um2_s=d_threshold_um2_s, n_steps=n_steps,
        mirrored=mirrored, split=split, ranges=ranges, all_tracks=all_tracks)
    out["config"] = cfg
    return out
