"""Legacy item 2: the appD speed maps, and the counts the legacy figure never printed.

READ ``SMT_GUI/@SMT/plotSpeedmap.m`` beside this. Every defect below was confirmed in
that source before it was changed.

THE DESIGN, reproduced: the apparent D of each track comes from its MSD curve; tracks
with ``R^2 >= r2Lim`` are kept; tracks with ``lb <= D <= ub`` are kept and drawn as one
panel per window; coordinates are normalised by the cell size; the model cell contour
filter is applied; the track map and the heat map use the legacy layout.

The three defects, and one option added.

**3a. The N label counts localisations plus tracks.** Line 182 prints
``size(normTracks,1)``. Line 100 first flattens the cell array and appends one
``[NaN NaN]`` padding row per track, so the printed number is neither a track count nor a
localisation count. Two labels are printed here instead, and both are named.

**3b. The contour filter trims INSIDE a track and says nothing.** ``inCellContour.m``
returns ``[xq(in), yq(in)]``, a SHORTENED track; a track is dropped only when it becomes
empty. So the apparent D comes from the UNTRIMMED MSD table while the drawn positions are
trimmed. Three counts are printed here: tracks partially trimmed, localisations removed
from surviving tracks, and tracks that became empty. **Direction: trimming removes the
outermost localisations first, so it moves the drawn positions toward the cell centre.**

**3c. The colour scale is per panel.** Line 184 calls ``caxis(ax2,'auto')`` after every
draw, so the scale comes from that panel's own maximum and any limit set earlier is
discarded. Two panels can never be compared. ``--speedmap-clim LO HI`` applies one scale
to both; the default stays auto so the default looks like the legacy figure. The applied
limits are printed in both modes.

The minimum track length is the largest effect on this figure. The MSD tab's
``minLength`` is 9 (``SMTracker.m:508``) while u-track supplies 5, so 9 binds and the map
is drawn from about one track in five. The census reports it per group, because the
fraction kept differs between groups and between carbon sources, and that difference is
what makes two panels non-comparable.

The filter is not neutral in D. Long tracks are preferentially the low-D molecules: a
high-D molecule leaves the focal plane sooner and its links fail more often. So the
minimum selects toward low-D, and the contour trim moves drawn positions inward. **Both
effects push the same way**, and together they make a central low-D concentration look
stronger than the data support.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .guard import legacy_path
from ..canonical import without_attrs

__all__ = ["SpeedmapError", "LEGACY_MIN_LENGTH", "UPSTREAM_MIN_LENGTH",
           "TWO_EFFECTS_SAME_DIRECTION", "track_census", "write_track_census",
           "per_track_appd", "contour_trim", "n_label_counts"]


class SpeedmapError(RuntimeError):
    """Raised when a legacy speed map cannot be produced honestly."""


#: SMTracker.m:508, the MSD tab's minLength edit box, default 9.
LEGACY_MIN_LENGTH = 9

#: u-track's own minTrackLen, confirmed empirically: the shortest track in this data is
#: exactly 5 localisations. So 9 binds and 5 does not.
UPSTREAM_MIN_LENGTH = 5

#: Printed with the census and in the provenance. Two effects, one direction.
TWO_EFFECTS_SAME_DIRECTION = (
    "two effects act on this figure and both point the same way: the minimum of 9 "
    "localisations selects toward low-D molecules, because a high-D molecule leaves the "
    "focal plane sooner and its links fail more often; and the contour trim removes the "
    "outermost localisations first, so it moves drawn positions toward the cell centre. "
    "Together they make a central low-D concentration look stronger than the data "
    "support.")


def track_census(canonical: pd.DataFrame, *, enzyme: str, condition: str,
                 min_length: int = LEGACY_MIN_LENGTH) -> dict:
    """Per-group track census: how much of the data the minimum length removes.

    Both counts, never the difference. The fraction kept is reported per group because it
    differs between groups and between carbon sources, and a limitation line that carries
    a hard-coded fraction would drift from its own figure.
    """
    if "uid" not in canonical.columns:
        raise SpeedmapError("not a canonical table (no 'uid')")
    lengths = canonical.groupby("uid").size()
    if lengths.empty:
        raise SpeedmapError(f"{enzyme}/{condition}: no tracks")
    kept = int((lengths >= min_length).sum())
    total = int(lengths.size)
    return {
        "enzyme": enzyme, "condition": condition,
        "median_track_length": float(lengths.median()),
        "n_tracks_in": total,
        "n_tracks_out": kept,
        "n_tracks_below_minimum": int(total - kept),
        "fraction_kept": kept / total,
        "percent_kept": 100.0 * kept / total,
        "legacy_min_length": int(min_length),
        "upstream_min_length": UPSTREAM_MIN_LENGTH,
        "which_binds": ("the legacy minimum of "
                        f"{min_length}; u-track supplies {UPSTREAM_MIN_LENGTH}, so the "
                        "upstream value is not the binding one on this figure"),
        "localisations_in": int(len(canonical)),
        "localisations_out": int(canonical[canonical["uid"].isin(
            lengths[lengths >= min_length].index)].shape[0]),
        "two_effects": TWO_EFFECTS_SAME_DIRECTION,
    }


def write_track_census(rows, root=None) -> pd.DataFrame:
    """Deposit ``legacy_out/tables/legacy_appd_track_census.csv``.

    Adds the between-condition comparison, because if one carbon source loses more of its
    tracks than the other then the two panels are not drawn from comparable track sets,
    and that has to be visible rather than derivable.
    """
    table = pd.DataFrame(list(rows))
    if {"enzyme", "condition"} <= set(table.columns):
        wide = table.pivot_table(index="enzyme", columns="condition",
                                 values="percent_kept")
        if {"Glucose", "Malate"} <= set(wide.columns):
            gap = (wide["Glucose"] - wide["Malate"]).rename("percent_kept_gap")
            table = table.merge(gap.reset_index(), on="enzyme", how="left")
            table["panels_comparable"] = table["percent_kept_gap"].abs() < 5.0
            table["comparability_note"] = np.where(
                table["percent_kept_gap"].abs() < 5.0,
                "the two panels keep a similar share of their tracks",
                "THE TWO PANELS ARE NOT DRAWN FROM COMPARABLE TRACK SETS: one carbon "
                "source loses more of its tracks than the other")
    table.to_csv(legacy_path("tables", "legacy_appd_track_census.csv", root=root),
                 index=False)
    return table


def per_track_appd(canonical: pd.DataFrame, *, max_lag: int = 4,
                   min_length: int = LEGACY_MIN_LENGTH) -> pd.DataFrame:
    """One apparent D and one R-squared per TRACK, from that track's own MSD curve.

    This is ``fitMSDTracks.m``: a weighted straight line through the first ``max_lag``
    lags of each track's MSD, with ``D = slope / 4`` for two dimensions. The legacy
    speed map colours a track by this D and filters on this R-squared.

    Tracks shorter than ``min_length`` are excluded here, which is where the legacy
    ``minLength`` of 9 bites.
    """
    from ..msd import msd_per_track

    lengths = canonical.groupby("uid").size()
    keep = set(lengths[lengths >= min_length].index)
    if not keep:
        raise SpeedmapError(f"no track reaches {min_length} localisations")
    subset = canonical[canonical["uid"].isin(keep)]
    curves = msd_per_track(subset, max_lag=max_lag)

    rows = []
    for uid, part in without_attrs(curves).groupby("uid", observed=True):
        part = part[part["lag_frames"] <= max_lag].sort_values("lag_frames")
        if len(part) < 2:
            continue
        t = part["lag_frames"].to_numpy(dtype=float) * float(
            subset["tau_s"].iloc[0] if "tau_s" in subset.columns else 1.0)
        y = part["msd_um2"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(t, y, 1)
        predicted = slope * t + intercept
        ss_res = float(((y - predicted) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        rows.append({
            "uid": uid,
            "appd_um2_per_s": float(slope / 4.0),      # 2 dimensions
            "intercept_um2": float(intercept),
            "r_squared": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
            "n_lags": int(len(part)),
            "n_localisations": int(lengths.loc[uid]),
        })
    if not rows:
        raise SpeedmapError("no track produced a fit")
    return pd.DataFrame(rows)


def contour_trim(positions_by_track, inside) -> dict:
    """Apply the legacy contour filter and report the THREE counts it never printed.

    ``inside(xy)`` returns a boolean mask. ``inCellContour.m`` keeps the localisations
    inside the outline and returns a SHORTENED track; a track is dropped only when it
    becomes empty. So a track can be partially trimmed and still be drawn, and its colour
    still comes from the UNTRIMMED MSD table.

    DIRECTION, stated with the counts: trimming removes the outermost localisations
    first, so it moves the drawn positions toward the cell centre.
    """
    tracks_in = len(positions_by_track)
    localisations_in = int(sum(len(v) for v in positions_by_track.values()))
    partially_trimmed = 0
    removed_from_survivors = 0
    became_empty = 0
    kept = {}
    for uid, xy in positions_by_track.items():
        xy = np.asarray(xy, dtype=float)
        mask = np.asarray(inside(xy), dtype=bool)
        n_removed = int((~mask).sum())
        if not mask.any():
            became_empty += 1
            continue
        if n_removed:
            partially_trimmed += 1
            removed_from_survivors += n_removed
        kept[uid] = xy[mask]
    return {
        "kept": kept,
        "tracks_in": tracks_in,
        "tracks_out": len(kept),
        "localisations_in": localisations_in,
        "localisations_out": int(sum(len(v) for v in kept.values())),
        "tracks_partially_trimmed": partially_trimmed,
        "localisations_removed_from_surviving_tracks": removed_from_survivors,
        "tracks_that_became_empty": became_empty,
        "direction": ("trimming removes the outermost localisations first, so it moves "
                      "the drawn positions toward the cell centre"),
        "colour_source": ("the apparent D comes from the UNTRIMMED MSD table; only the "
                          "drawn positions are trimmed"),
    }


def n_label_counts(positions_by_track) -> dict:
    """The N label, fixed. Two named counts instead of one meaningless number.

    ``plotSpeedmap.m:182`` prints ``size(normTracks,1)`` AFTER line 100 has flattened the
    cell array and appended one ``[NaN NaN]`` padding row per track. The printed value is
    therefore localisations PLUS tracks, which is neither quantity. The legacy value is
    reproduced here as ``legacy_N_label`` so the difference is visible, and never used as
    either count.
    """
    n_tracks = len(positions_by_track)
    n_localisations = int(sum(len(v) for v in positions_by_track.values()))
    return {
        "n_tracks": n_tracks,
        "n_localisations": n_localisations,
        "legacy_N_label": n_localisations + n_tracks,
        "why_the_legacy_label_is_neither": (
            "plotSpeedmap.m:182 prints size(normTracks,1) after one [NaN NaN] padding "
            "row per track has been appended at line 100, so it is localisations plus "
            "tracks"),
    }
