"""Spatial HMM: where each fitted HMM state sits inside the cell, in the
SAME capsule frame `spatialsqd` draws -- so the Map Configuration dialog's
third computational method slots into the one existing grid/matrix rather
than needing a new kind of picture.

**No edit to `smtracker.hmm`, `smtracker.maps`, or `smtracker.spatialsqd`.**
This module only IMPORTS from them (read-only, the same posture
`spatialappd.py` already established), by the same explicit instruction:
those modules are shared with a colleague and stay untouched by this
feature. `local_half_width`, `fold_counts` and `_step_coordinates` are
reused from `spatialsqd` rather than re-derived -- one source of truth for
the capsule geometry and for the (non-trivial) step-position re-projection.
`_step_coordinates` is not in `spatialsqd.__all__`, which is noted rather
than hidden: it is the one import here that is not from the module's
declared public surface, kept anyway because re-deriving a correct
centreline re-projection would duplicate real complexity for no reason.

**ANY STATE COUNT SINCE 3 NOV 2026, AND THE MAP OF EVERY JUMP** (asked: "the HMM maps, they have less options
than the SQD maps", "the all jumps is missing"). `hmm.decode_states` now keeps every state's posterior
(``p_state_k``), so the soft weight of 1, 2 or 3 states is read directly, and ``all_jumps`` adds the map of
every step at weight 1 (``"jumps"``, as SQD's), placed exactly as the states are. One state draws that map
alone. The paragraph below is how it was until then.

**Fixed at two states, and that is not an arbitrary restriction.**
`hmm.decode_states` returns only the WINNING state and its own confidence
per step (`state`, `posterior`), not a full per-state posterior column.
At exactly two states the complete soft weight is recoverable from that
pair (a 2-state posterior sums to 1); at three or more it is not,
without re-deriving the decode ourselves. Every production caller of
`hmm.fit_hmm` in this codebase already uses `n_states=2`, and the SQD
spatial method already hardcodes `n_pop=2` at the GUI layer rather than
exposing it as a dialog control -- so fixing two states here matches both
the existing UI precedent and the only regime this map can be honest in.

**"Slow"/"fast", not "short"/"long".** SQD's "short"/"long" describes
what ITS OWN hard-assignment mechanism does (a threshold on jump length,
because the SQD posterior is monotonic in r² at two populations). HMM is a
genuine temporal state model with real dwell times and transition rates --
its two states ARE slow/bound and fast/free in the ordinary sense, and this
project's own GUI code already names them that way (`dynamics_chart.py`:
"0 = slow/bound, 1 = fast/free"). Reusing `spatialappd`'s `"slow"`/`"fast"`
KEYS is harmless: payloads from different methods are already kept apart
by `method` being the first element of `spatial_map_chart._settings_key`.

**No Viterbi anywhere in this codebase.** `decode_states`'s "hard" state is
a per-step ARGMAX of the marginal posterior (forward-backward gamma), not a
Viterbi max-probability path -- structurally the same shape as SQD's own
hard assignment (argmax of a posterior), just computed by a temporal decode
instead of a static mixture density. So "soft" vs "hard" here means exactly
what it means for SQD: posterior-weighted vs. posterior-argmax.

**A fitted chain with no within-track switching is DRAWN, and said** (ruled
4 Nov 2026: the refusal is lifted). `decode_states` refused it while the
stationary distribution was the first step's prior, a point mass that decided
every step and contradicted the fit's own occupancy; under the fitted initial
distribution each track is decoded in one state for its whole length, so the
maps place slow and fast TRACKS, as the apparent-D maps do, and the payload's
``no_switching`` makes the map say so (`spatial_map_view.no_switching_warning`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .spatialsqd import (
    _check_long_axis,
    axis_profiles,
    _check_weighting,
    _step_coordinates,
    contributor_labels,
    contributor_neff_grids,
    fold_counts,
    local_half_width,
    long_axis_x,
    row_cell_keys,
    step_block,
    track_weights,
)

__all__ = [
    "HMM_STATE_NAMES",
    "SpatialHMMConfig",
    "maps_from_projected_hmm",
    "spatial_hmm_map",
]

#: Each state count's map names, slowest first (SQD's and apparent diffusion's words for the same places).
HMM_STATE_NAMES = {1: (), 2: ("slow", "fast"), 3: ("slow", "middle", "fast")}
#: The map of every jump (`spatial_map_view.ALL_JUMPS_KIND`).
ALL_JUMPS = "jumps"


@dataclass(frozen=True)
class SpatialHMMConfig:
    """The geometry `spatialsqd.SpatialSQDConfig`/`spatialappd.SpatialAppDConfig`
    also carry -- every field required, no defaults, for the same reason: a
    bin is a resolution claim.

    No `n_states` field: fixed at 2 (see the module docstring). No
    `pole_margin`/`n_profile_bins`: no radial profile is built here, the GUI
    payload never reads one.
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


def maps_from_projected_hmm(projected: pd.DataFrame, fit, cfg: SpatialHMMConfig, *,
                            assignment: str = "soft",
                            step_position: str = "A_start",
                            mirrored: bool = False, meshes=None,
                            min_posterior: float = 0.0,
                            weighting: str = "steps", with_steps: bool = False,
                            all_jumps: bool = False) -> dict:
    """Per-state step-density grids for ONE already-loaded, projected table
    and one already-FITTED `HMMFit` -- the HMM analogue of
    `spatialsqd.maps_from_projected`.

    `fit` is fitted by the CALLER (`spatial_hmm_map`, below), exactly as
    `spatialsqd.maps_from_projected` takes a fitted `fit_sqd` result rather
    than fitting itself -- this function is the reusable in-memory core.

    `weighting` as in `spatialsqd.maps_from_projected`: `"track"` makes every
    track count once; `step_grids`/`step_weights` always hold the plain counts.
    `with_steps=True` adds `"steps"`, the per-step record of which track went into
    which bin (`spatialsqd.step_block`).
    """
    from . import maps as maps_module

    _check_weighting(weighting)

    if assignment not in ("hard", "soft"):
        raise ValueError(f"assignment must be 'hard' or 'soft', got {assignment!r}")
    if step_position not in ("A_start", "B_midpoint", "C_end"):
        raise ValueError(
            f"step_position must be one of A_start/B_midpoint/C_end, got "
            f"{step_position!r}")

    labelled = maps_module.labels_from_hmm(projected, fit, min_posterior=min_posterior)

    if step_position != "A_start":
        if meshes is None:
            raise ValueError(
                f"step_position={step_position!r} re-projects each step's "
                f"midpoint or end through the cell centreline, which needs "
                f"the meshes. Pass meshes=, or use 'A_start', which the "
                f"labelled table already carries.")
        coords = _step_coordinates(projected, labelled, meshes, lag_frames=1)[
            step_position]
        l_norm = coords["l_norm"].to_numpy(float)
        d_norm = coords["d_norm"].to_numpy(float)
    else:
        l_norm = labelled["l_norm"].to_numpy(float)
        d_norm = labelled["d_norm"].to_numpy(float)

    # BY CONSTRUCTION, never by matching label text -- `fit_hmm` already
    # sorts `D` ascending before returning, but computing the order here rather
    # than trusting it is the same defensive habit `spatialsqd`'s own compute
    # wrapper keeps.
    D = np.asarray(fit.D, dtype=float)
    order = np.argsort(D)
    n_states = int(fit.n_states)
    if n_states not in HMM_STATE_NAMES:
        raise ValueError(f"an HMM map draws 1 to 3 states, got {n_states}")
    names = HMM_STATE_NAMES[n_states]
    state = labelled["state"].to_numpy()
    # each state's soft weight is its own posterior (`decode_states`' p_state_k); hard, the winner's
    soft = {name: (labelled[f"p_state_{int(order[i]) + 1}"].to_numpy(float)
                   if f"p_state_{int(order[i]) + 1}" in labelled.columns
                   else (state == int(order[i]) + 1).astype(float))
            for i, name in enumerate(names)}
    hard = {name: (state == int(order[i]) + 1).astype(float) for i, name in enumerate(names)}

    xe, ye = cfg.edges()
    keep = np.isfinite(d_norm) & (np.abs(d_norm) <= 1.0) & np.isfinite(l_norm)
    cells = (row_cell_keys(labelled, projected)[keep]
             if getattr(cfg, "long_axis", "normalised") != "normalised" else None)
    x = long_axis_x(l_norm[keep], cfg, cells=cells,
                    geometry=projected.attrs.get("geometry"))
    y = d_norm[keep] * local_half_width(x, cfg)

    chosen = hard if assignment == "hard" else soft
    weights_by_name = {name: w[keep] for name, w in chosen.items()}
    if all_jumps or not names:
        # EVERY JUMP, weight 1, placed as the states are: the map a one-state run draws, beside every split
        weights_by_name = {ALL_JUMPS: np.ones(int(keep.sum())), **weights_by_name}

    per_track = track_weights(labelled)[keep] if weighting == "track" else None
    tracks, cells = contributor_labels(labelled, projected)
    tracks, cells = tracks[keep], cells[keep]
    grids, weights, step_grids, step_weights, drawn_by_label = {}, {}, {}, {}, {}
    neff_track, neff_cell = {}, {}
    for name, weight in weights_by_name.items():
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

    if weighting == "track":
        weight_unit = "weighted tracks" if assignment == "soft" else "tracks"
    else:
        weight_unit = "weighted steps" if assignment == "soft" else "steps"
    out = {
        "grids": grids,
        "weights": weights,
        "step_grids": step_grids,
        "step_weights": step_weights,
        "neff_track": neff_track,
        "neff_cell": neff_cell,
        "weighting": weighting,
        "labels": {int(order[i]) + 1: name for i, name in enumerate(names)},
        "assignment": assignment,
        "step_position": step_position,
        "mirrored": mirrored,
        "n_states": fit.n_states,
        "D": D,
        "occupancy": np.asarray(fit.occupancy, dtype=float),
        "fraction_kept": float(labelled.attrs.get("fraction_kept", 1.0)),
        "weight_unit": weight_unit,
    }
    # where each population sits along and across the cell (`spatialsqd.axis_profiles`)
    out["profiles"] = axis_profiles(x, y, drawn_by_label, cfg)
    if with_steps:
        out["steps"] = step_block(labelled.iloc[np.flatnonzero(keep)], x, y,
                                  drawn_by_label, cfg)
    return out


def spatial_hmm_map(projected: pd.DataFrame, *, tau_s: float, bin_long_nm: float,
                    bin_short_nm: float, smooth_bins: float,
                    cell_length_um: float, cell_width_um: float,
                    assignment: str = "soft", step_position: str = "A_start",
                    mirrored: bool = False, meshes=None,
                    n_starts: int | None = None,
                    min_posterior: float = 0.0,
                    long_axis: str = "normalised", n_states: int = 2,
                    all_jumps: bool = False, fixed_D=None,
                    localization_sigma_um: float = 0.0, calibration=None) -> dict:
    """The GUI-facing entry point: one projected table in, per-state
    step-density grids out. Mirrors `spatialsqd.spatial_sqd_map`'s own
    shape -- fits internally (a panel can supply a table and scalar
    controls and nothing else), the bin is given in nanometres for the same
    reason (compared against the camera pixel and the localisation
    precision), rounded to an even bin count since folding pairs each bin
    with its reflection.

    `n_starts=None` resolves to `hmm.default_starts(2)` -- the CURRENT
    measured multistart requirement, not the larger counts this codebase's
    older GUI call sites use (they predate that measurement).

    The fit is returned as `fit` so a caller can draw further maps (e.g. per
    replicate day) under the SAME fit instead of refitting.

    ``fixed_D`` and ``localization_sigma_um`` (4 Nov 2026, `map_fit_source`): the states' D HELD, the HMM section's,
    and its sigma, so only the transitions and the initial distribution are fitted on this table (`hmm.fit_hmm`'s
    ``fixed_D``); ``n_states`` must be their count. ``calibration`` (6 Nov 2026): the section's sigma was Per spot --
    each step's own noise in the fit, and in the decoding (`hmm.decode_states` reads the fit's own calibration).
    """
    from .hmm import fit_hmm, step_sequences

    def even(value: float) -> int:
        count = round(value)
        return max(2, count + (count % 2))

    cfg = SpatialHMMConfig(
        cell_length_um=cell_length_um, cell_width_um=cell_width_um,
        n_bins_long=even(cell_length_um * 1000.0 / bin_long_nm),
        n_bins_short=even(cell_width_um * 1000.0 / bin_short_nm),
        smooth_bins=smooth_bins, long_axis=long_axis)
    if calibration is not None:
        sequences, noises = step_sequences(projected, calibration=calibration)
        fit = fit_hmm(sequences, n_states=int(n_states), tau_s=tau_s, n_starts=n_starts, fixed_D=fixed_D,
                      noise_sequences=noises, calibration=calibration)
    else:
        fit = fit_hmm(step_sequences(projected), n_states=int(n_states), tau_s=tau_s,
                      n_starts=n_starts, fixed_D=fixed_D, localization_sigma_um=float(localization_sigma_um))
    out = maps_from_projected_hmm(
        projected, fit, cfg, assignment=assignment, step_position=step_position,
        mirrored=mirrored, meshes=meshes, min_posterior=min_posterior, all_jumps=all_jumps)
    out["config"] = cfg
    out["fit"] = fit
    return out
