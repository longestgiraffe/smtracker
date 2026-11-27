"""The HMM map's Turning back (22 Nov 2026): where in the cell each HMM state turns back beyond chance.

Asked: "what about the angle map we planned where u see how tracks change in their angle? ... HMM? appd?", then
"both ja" (the HMM section's Jump angles page, `hmm_angles`, then the map). The Spatial Map's HMM method gains a
choice, Show: Steps (the maps as before) or **Turning back** (these): the same states, the same fit (its own, or the
HMM section's last Run's D, `map_fit_source`), the same capsule, columns and replicate rows.

WHAT A BIN READS. Of the pairs of steps whose turn lies in the bin (the vertex, the localisation the two steps share)
and whose angle is within 30 degrees of forward or straight back (`hmm_angles.WINDOW_DEG`, the page's windows), the
share that turned back: 0.5 is no preference. That is Show: **Turning back**, the data alone (asked the same day: "make
comparison with simulation optional, and also just show pure data"). Show: **Beyond chance** makes the chance too, at
the SAME pairs and weights -- free molecules started at each pair's turn (`hmm_angles.chance_by_state`, the page's own
chance) -- and draws observed minus chance: 0 is a free molecule there. A pair counts for a
state at its two-slice posterior (`hmm.decode_states`' ``p_stay_k``); All jumps is every pair not across a switch,
its chance each pair's states' chances at their weights.

WHY A DIFFERENCE MAY BE MAPPED HERE when the observed-over-null map was withdrawn (`docs/claude/spatial_maps.md`): that
map divided two separately simulated sets of positions whose baselines differed, so the offset entered twice. Here the
chance is made AT the observed pairs, so the two maps share every position, weight and bin; only the angles differ.

THE BINS. A share needs many pairs, so the maps are binned `spatialsda.VALUE_BIN_FACTOR` times coarser and written
back onto the fine bins (SDA's value maps' rule, `spatialsda._spread`), and a bin needs `spatialsda.VALUE_N_MIN` pairs
in the windows; the observed and chance maps are smoothed and masked alike (`spatial_map_view.finish_turning_panels`).

INTERFACES: `window_columns` (each pair's windows, observed and chance), `maps_from_pairs` (one table of pairs in, the
grids out in `spatialsda.maps_from_projected_sda`'s shape: ``sums`` and ``grids`` per kind, ``<kind>:chance`` beside
each). No Qt; imports with Qt blocked.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .hmm_angles import WINDOW_DEG

__all__ = ["CHANCE_SUFFIX", "UNIT", "COUNT_UNIT", "window_columns", "maps_from_pairs"]

#: The chance map of a kind is drawn under ``<kind>:chance`` and folded into the kind's own panel.
CHANCE_SUFFIX = ":chance"
#: What a Turning back panel's colour bar is in, and what its evidence counts.
UNIT = "turning back beyond chance"
COUNT_UNIT = "pairs"


def window_columns(pairs: pd.DataFrame, chance, n_states: int, *, window_deg: float = WINDOW_DEG) -> pd.DataFrame:
    """``pairs`` with each pair's windows: ``o_win`` / ``o_back`` (its own angle within ``window_deg`` of 0 or 180, and
    of 180), and -- with ``chance`` -- per state ``c_win_k`` / ``c_back_k``, the share of its free molecules that did
    the same (0 where it was not walked for that state, whose weight is then negligible)."""
    out = pairs.copy()
    angle = out["angle_deg"].to_numpy(dtype=float)
    out["o_back"] = (angle >= 180.0 - window_deg).astype(float)
    out["o_win"] = ((angle <= window_deg) | (angle >= 180.0 - window_deg)).astype(float)
    for k in range(int(n_states) if chance is not None else 0):
        drawn = np.asarray(chance[k], dtype=float)
        walked = np.isfinite(drawn)
        back = (drawn >= 180.0 - window_deg) & walked
        win = ((drawn <= window_deg) | (drawn >= 180.0 - window_deg)) & walked
        n = np.maximum(walked.sum(axis=1), 1)
        out[f"c_back_{k + 1}"] = back.sum(axis=1) / n
        out[f"c_win_{k + 1}"] = win.sum(axis=1) / n
    return out


def _weights_for(pairs: pd.DataFrame, kind: str, kinds: list, n_states: int) -> tuple:
    """``(weight, chance_win, chance_back)`` of every pair for ``kind``: a state's own, or All jumps (``"jumps"``: every
    pair not across a switch, its chance its states' at their weights)."""
    states = [k for k in kinds if k != "jumps"]
    with_chance = "c_win_1" in pairs.columns
    if not with_chance:
        if kind == "jumps":
            w = sum(pairs[f"w_{k + 1}"].to_numpy(dtype=float) for k in range(int(n_states)))
        else:
            w = pairs[f"w_{states.index(kind) + 1}"].to_numpy(dtype=float)
        return np.asarray(w, dtype=float), None, None
    if kind == "jumps":
        w = np.zeros(len(pairs))
        c_win = np.zeros(len(pairs))
        c_back = np.zeros(len(pairs))
        for k in range(int(n_states)):
            wk = pairs[f"w_{k + 1}"].to_numpy(dtype=float)
            w += wk
            c_win += wk * pairs[f"c_win_{k + 1}"].to_numpy(dtype=float)
            c_back += wk * pairs[f"c_back_{k + 1}"].to_numpy(dtype=float)
        safe = np.where(w > 0, w, 1.0)
        return w, c_win / safe, c_back / safe
    k = states.index(kind) + 1
    return (pairs[f"w_{k}"].to_numpy(dtype=float), pairs[f"c_win_{k}"].to_numpy(dtype=float),
            pairs[f"c_back_{k}"].to_numpy(dtype=float))


def maps_from_pairs(pairs: pd.DataFrame, kinds: list, cfg, projected: pd.DataFrame, *, n_states: int,
                    mirrored: bool = False, weighting: str = "steps") -> dict:
    """The grids of every kind (the HMM map's, slowest first, All jumps before them) and its chance, from
    `window_columns`' pairs: ``sums`` (weighted pairs that turned back) and ``grids`` (weighted pairs in the windows)
    per kind and per ``<kind>:chance``, ``step_grids`` the plain pairs in the windows (the evidence, the observed
    pairs' for both, so the two are masked alike), the n_eff grids, all on the coarse value bins written back onto
    the fine ones. ``weighting="track"``: every track one vote per map."""
    from .spatialsda import VALUE_BIN_FACTOR, _capsule_xy, _spread, value_bin_factor, value_grids
    from .spatialsqd import _check_weighting, contributor_labels, contributor_neff_grids, fold_counts, track_weights

    _check_weighting(weighting)
    xe, ye = cfg.edges()
    factor = value_bin_factor(cfg.n_bins_long, cfg.n_bins_short, VALUE_BIN_FACTOR)
    kx, ky = (xe[::factor], ye[::factor]) if factor > 1 else (xe, ye)
    table = pairs.reset_index(drop=True)
    if len(table):
        x, y, keep = _capsule_xy(table, table["l_norm"].to_numpy(float), table["d_norm"].to_numpy(float), cfg,
                                 projected)
        tracks, cells = contributor_labels(table, projected)
        tracks, cells = tracks[keep], cells[keep]
        vote = track_weights(table)[keep] if weighting == "track" else np.ones(int(keep.sum()))
    else:
        x = y = vote = np.empty(0)
        keep = np.zeros(0, dtype=bool)
        tracks = cells = np.empty(0, dtype=str)
    o_win = table["o_win"].to_numpy(float)[keep] if len(table) else np.empty(0)
    o_back = table["o_back"].to_numpy(float)[keep] if len(table) else np.empty(0)
    sums, grids, step_grids, step_weights, weights, neff_track, neff_cell = {}, {}, {}, {}, {}, {}, {}
    evidence = np.histogram2d(x, y, bins=[kx, ky], weights=o_win)[0].T
    if mirrored:
        evidence = fold_counts(evidence)
    n_track, n_cell = contributor_neff_grids(x, y, o_win, tracks, cells, kx, ky, mirrored=mirrored)
    for kind in kinds:
        w, c_win, c_back = _weights_for(table, kind, kinds, n_states) if len(table) else (np.empty(0),) * 3
        w = w[keep] * vote if len(table) else w
        sets = [(kind, o_win, o_back)]
        if c_win is not None:
            # AGAINST CHANCE (the map's Show: Beyond chance): the chance's grids beside the kind's own
            sets.append((kind + CHANCE_SUFFIX, c_win[keep] if len(table) else c_win,
                         c_back[keep] if len(table) else c_back))
        for name, win, back in sets:
            # a share: the weighted pairs that turned back over those in the windows, pooled over a fold's reflections
            coarse_sums, _all = value_grids(x, y, back, w, kx, ky, mirrored=mirrored)     # sum of w x turned back
            coarse_win, _all = value_grids(x, y, win, w, kx, ky, mirrored=mirrored)       # sum of w x in a window
            sums[name] = _spread(coarse_sums, factor)
            grids[name] = _spread(coarse_win, factor)
            step_grids[name] = _spread(evidence, factor)
            step_weights[name] = float(o_win.sum())
            weights[name] = float((w * win).sum())
            neff_track[name], neff_cell[name] = _spread(n_track, factor), _spread(n_cell, factor)
    return {"grids": grids, "weights": weights, "step_grids": step_grids, "step_weights": step_weights,
            "sums": sums, "neff_track": neff_track, "neff_cell": neff_cell, "mirrored": bool(mirrored),
            "weighting": weighting, "value_bin_factor": int(factor), "n_pairs": int(len(table))}
