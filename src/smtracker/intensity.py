"""The per-localisation amplitude u-track already records, and what it will not support.

The legacy intensity pipeline cannot be run on this project, and that is a property of THE
DATA rather than of the conversion. `SMT_GUI/@SMT/spotIntensity.m` integrates a 6-pixel
inner ROI against a 12-pixel outer one, frame by frame, out of the raw movie: it calls
`bkgInst` over frames 1 to 20 with the laser off, `firstFrames` for the initial integrated
intensity, and `unevenIllu` for a per-pixel illumination ratio. All of that needs the
1001-frame stack. The reference project ships `phase/` and `sum/` only, both 126 x 130, so
`spotIntensity`, `BleachingStep`, `bkgCell`, `bkgSub`, `firstFrames`, `cropIntensity` and
`unevenIllumination` have no input here. They stay `planned` in the inventory, blocked on
data rather than on effort, and nothing in this module pretends to stand in for them.

What is available is a different quantity with a different name. u-track's
`tracksCoordAmpCG` carries a fitted Gaussian AMPLITUDE per localisation, which
`smtracker.ingest` already reads into the `amp` column and nothing has used since. It is
not an integrated intensity: no background is subtracted here, no illumination correction
is applied, and the value is the peak height of the fitted PSF rather than the photon
count under it. **Never present it as the legacy's intensity, and never convert one into
the other.**

Measured on the reference data: `amp` is
finite everywhere, spans 0.0036 to 0.366, and varies within a track with a median
coefficient of variation of 0.384. **But the change from a track's first localisation to
its last is symmetric about zero: the share of tracks that decrease is within a point of
the 50 per cent expected under no trend.** So there is no net photobleaching signature over the
lifetime of a track here, and a bleaching-step analysis is not supported by this column on
this data. The median track is 6 to 7 localisations, about 130 ms, which is a short window
in which to catch a step; that is a reason the signature may be absent, not evidence that
it is present.

The unit is the cell or the replicate day. Amplitudes within a track are correlated and
tracks within a cell share its illumination and its focus, so an interval or a test is
built over units. `amp_by_unit` returns one number per unit; nothing here computes a p
value on pooled localisations.
"""
from __future__ import annotations

from .chart_text import unit_word  # the word shown for a grouping unit

import numpy as np
import pandas as pd
from .canonical import without_attrs

__all__ = ["IntensityError", "AMP_IS_NOT_INTENSITY", "LEGACY_PIPELINE_BLOCKED",
           "amp_per_track", "bleaching_trend", "amp_by_unit", "amp_versus_D"]


class IntensityError(RuntimeError):
    """Raised when an amplitude result cannot be produced honestly."""


AMP_IS_NOT_INTENSITY = (
    "amp is u-track's fitted PSF peak height per localisation, not an integrated intensity "
    "(no background or illumination correction). It does not convert into the legacy "
    "spotIntensity figure")

LEGACY_PIPELINE_BLOCKED = (
    "the legacy intensity pipeline (spotIntensity.m) needs the raw 1001-frame movie; "
    "this project ships phase/ and sum/ only. It is blocked on data, not on effort")


def amp_per_track(table: pd.DataFrame, *, amp_col: str = "amp",
                  min_localisations: int = 3) -> pd.DataFrame:
    """One row per track: the amplitude's level, its spread and its trend.

    ``min_localisations`` defaults to 3, matching ``spotIntensity.m:4``, because a trend
    through two points is a line through two points. Tracks below it are counted and
    excluded rather than dropped silently.
    """
    for column in ("uid", "frame", amp_col):
        if column not in table.columns:
            raise IntensityError(f"missing column {column!r}")
    if min_localisations < 3:
        raise IntensityError(
            "a trend needs at least three localisations; through two points it is a line "
            "through two points and its slope carries no information about scatter")

    rows, thin = [], 0
    for uid, part in without_attrs(table).groupby("uid", observed=True, sort=False):
        part = part.sort_values("frame")
        a = part[amp_col].to_numpy(dtype=float)
        a = a[np.isfinite(a)]
        if len(a) < min_localisations:
            thin += 1
            continue
        frames = part["frame"].to_numpy(dtype=float)[:len(a)]
        slope = float(np.polyfit(frames - frames[0], a, 1)[0]) if len(a) > 2 else np.nan
        rows.append({
            "uid": uid, "n_localisations": int(len(a)),
            "amp_mean": float(a.mean()), "amp_median": float(np.median(a)),
            "amp_first": float(a[0]), "amp_last": float(a[-1]),
            "amp_last_minus_first": float(a[-1] - a[0]),
            "amp_cv": float(a.std(ddof=1) / a.mean()) if a.mean() else np.nan,
            "amp_slope_per_frame": slope,
        })
    if not rows:
        raise IntensityError(
            f"no track reached {min_localisations} localisations with a finite amplitude")
    out = pd.DataFrame(rows)
    out.attrs["n_tracks_below_minimum"] = thin
    out.attrs["min_localisations"] = int(min_localisations)
    out.attrs["amp_is_not_intensity"] = AMP_IS_NOT_INTENSITY
    return out


def bleaching_trend(per_track: pd.DataFrame) -> dict:
    """Is there a net decrease in amplitude over a track's lifetime?

    The comparison is against one half, not against zero. Under no trend the sign of the
    change is a coin flip, so the fraction of tracks that decrease is the statistic and
    0.5 is its null. Reporting a median change alone would leave the reader without
    anything to compare it to, which is the failure this project keeps finding.

    A track is short here, about 130 ms at the median, so an ABSENT signature is weak
    evidence that no bleaching occurs and is reported as such rather than as a negative
    result.
    """
    if per_track.empty:
        raise IntensityError("no tracks")
    change = per_track["amp_last_minus_first"].to_numpy(dtype=float)
    change = change[np.isfinite(change)]
    if not len(change):
        raise IntensityError("no finite amplitude change")
    n = int(len(change))
    n_down = int((change < 0).sum())
    fraction = n_down / n
    # A binomial interval on TRACKS, which are the units here; the localisations inside
    # one track are not independent of each other.
    se = np.sqrt(0.25 / n)
    return {
        "n_tracks": n,
        "n_tracks_decreasing": n_down,
        "n_tracks_increasing": int((change > 0).sum()),
        "fraction_decreasing": fraction,
        "fraction_under_no_trend": 0.5,
        "interval_95": (fraction - 1.96 * se, fraction + 1.96 * se),
        "median_change": float(np.median(change)),
        "consistent_with_no_trend": bool(abs(fraction - 0.5) <= 1.96 * se),
        "median_track_localisations": float(per_track["n_localisations"].median()),
        "caveat": (
            "the observation window is short, so an absent bleaching signature is weak "
            "evidence that no bleaching occurs rather than evidence that it does not"),
        "amp_is_not_intensity": AMP_IS_NOT_INTENSITY,
    }


def amp_by_unit(table: pd.DataFrame, per_track: pd.DataFrame, *, unit: str = "cell",
                min_tracks: int = 5) -> pd.DataFrame:
    """One amplitude summary per unit, so an interval is built over units.

    Amplitudes within a track are correlated and tracks within a cell share its
    illumination and its focus, so the pooled localisation is never the unit.
    """
    if unit not in table.columns:
        raise IntensityError(
            f"no {unit!r} column. The unit is the cell or the replicate, not the "
            f"pooled localisation.")
    key = table.groupby("uid", observed=True)[unit].first()
    joined = per_track.assign(**{unit: per_track["uid"].map(key)})
    if joined[unit].isna().all():
        raise IntensityError(f"no track could be assigned a {unit_word(unit)}")

    rows, thin = [], []
    for value, part in joined.dropna(subset=[unit]).groupby(unit, observed=True):
        if len(part) < min_tracks:
            thin.append({"unit": value, "n_tracks": int(len(part))})
            continue
        rows.append({"unit": value, "n_tracks": int(len(part)),
                     "amp_mean": float(part["amp_mean"].mean()),
                     "amp_median": float(part["amp_median"].median()),
                     "fraction_decreasing": float((part["amp_last_minus_first"] < 0).mean())})
    if not rows:
        raise IntensityError(f"no {unit_word(unit)} reached {min_tracks} tracks")
    out = pd.DataFrame(rows)
    out.attrs["unit"] = unit
    out.attrs["n_units_below_min_tracks"] = len(thin)
    out.attrs["units_below_min_tracks"] = thin
    return out


def amp_versus_D(per_track: pd.DataFrame, appd: pd.DataFrame, *,
                 amp_col: str = "amp_median",
                 d_col: str = "appd_um2_per_s") -> dict:
    """Does a brighter spot move differently? Spearman, on TRACKS.

    A track is the unit here, which is defensible because each contributes one amplitude
    and one apparent D. It is still not the cell, so the correlation is DESCRIPTIVE and
    its interval must come from resampling cells or days.

    A positive result has at least three readings and NONE of them is settled by THIS
    number: more molecules in one spot, a brighter fluorophore state, or a spot nearer the
    focal plane, which is also where the apparent D is least truncated. The third is a
    measurement artefact and would produce the same correlation.
    """
    from scipy import stats

    # A NON-UNIQUE uid MULTIPLIES ROWS AND INFLATES n. Measured on a lab project: a track
    # table merged to about 1.4 times as many rows because the apparent-D frame carried more than one
    # row per track. The correlation is then computed on repeated tracks and its p value
    # is wrong by whatever the duplication factor happens to be. An inner merge hides
    # this, because the result still looks like a table of tracks.
    for name, frame in (("per_track", per_track), ("appd", appd)):
        duplicated = int(frame["uid"].duplicated().sum())
        if duplicated:
            raise IntensityError(
                f"{name} carries {duplicated} repeated uid(s) out of {len(frame)} rows. "
                f"One row per track is required: merging a repeated uid multiplies rows "
                f"and inflates the sample size the correlation is computed on.")
    merged = per_track.merge(appd[["uid", d_col]], on="uid", how="inner")
    merged = merged[np.isfinite(merged[amp_col]) & np.isfinite(merged[d_col])]
    if len(merged) < 10:
        raise IntensityError(
            f"{len(merged)} tracks carry both an amplitude and an apparent D; that is too "
            f"few to describe a relationship")
    rho, p = stats.spearmanr(merged[amp_col], merged[d_col])
    return {
        "spearman_rho": float(rho),
        "p_value_on_tracks": float(p),
        "n_tracks": int(len(merged)),
        "unit": "track",
        "interval_note": (
            "the p value is computed on TRACKS, which are not independent within a cell. "
            "Treat rho as descriptive and take its interval by resampling cells or replicates"),
        "confounds": (
            "a positive correlation reads at least three ways: more molecules per spot, a "
            "brighter fluorophore state, or a spot nearer the focal plane, where the "
            "apparent D is also least truncated. The third is a measurement artefact and "
            "produces the same sign"),
        "amp_is_not_intensity": AMP_IS_NOT_INTENSITY,
    }
