"""Step orientation in the cell frame, with the null a rod-shaped cell demands.

Converted from `SMT_GUI/Help_fcn/bias/` (Marco Mauri, 2018): `orientation_radians.m`,
`compute_distances.m`, `plot_apparent_orientation.m`,
`plot_directionality_histogram.m`. Read those beside this.

The legacy measures axial anisotropy, not directionality, and its name says otherwise.
`orientation_radians.m:19` is::

    atan((wy*abs(track(p,3)-track(p+1,3))) ./ (wx*abs(track(p,2)-track(p+1,2))))

Both components are wrapped in `abs`, so every step is folded into the first quadrant and
the angle lands in [0, pi/2]. **A step toward the pole and a step toward midcell get the
same angle; so do a step toward the membrane and a step toward the axis.** Direction is
exactly what the fold destroys. What survives is how much a step lies ALONG the cell
against ACROSS it, which is an axial quantity and a weaker claim.

Both are computed here and they are named apart. `orientation` is the signed angle from
`atan2`, in (-pi, pi], which can answer a directional question. `axial_angle` is the
legacy's folded quantity, in [0, pi/2], reproduced so the old histogram can be recovered
and so the difference between the two is measurable rather than argued.

The frame is the CELL'S, and that part of the legacy is correct. `plot_apparent_orientation`
passes `obj.getTracks(true)`, and `getTracks.m:14` selects `projTracks`, so the
coordinates are already projected onto the cell centreline: the first axis runs along the
cell, the second across it. An earlier reading of this module suspected camera coordinates
and was wrong. `rotateTracks.m` exists and is called by nothing.

`atan` is not `atan2`, and the difference is not only the sign. At `dL = 0` the legacy
divides by zero; MATLAB returns `pi/2`, which happens to be right. At `dL = dD = 0` it
returns `NaN` from `atan(0/0)`, and a zero-length step has no orientation, so that case is
excluded here rather than propagated.

A ROD-SHAPED cell produces axial anisotropy with no biology in it. A cell about three
times longer than it is wide has more room along its axis, so a confined random walk takes
more long steps along than across, and the walls reflect motion back along the axis. Any
observed anisotropy must therefore be read against the rate a free confined walk produces,
exactly as a confined fraction is read against its chance rate. `axial_null` runs the
identical statistic on matched free Brownian tracks inside the same geometry and reports
observed, chance and excess. There is no default D: it must come from the analysis.

The unit is the cell or the replicate day, never the pooled step. Steps within a track are
correlated and tracks within a cell share its environment. `by_unit` returns one number per
unit so an interval and a test can be built on units; nothing here computes a p value on
pooled steps, and `axial_order` refuses a frame of pooled steps for that purpose.
"""
from __future__ import annotations

from .chart_text import unit_word  # the word shown for a grouping unit

import numpy as np
import pandas as pd
from .canonical import without_attrs

__all__ = ["DirectionalityError", "step_vectors", "orientation", "axial_angle",
           "axial_order", "by_unit", "axial_null", "LEGACY_FOLD_NOTE"]


class DirectionalityError(RuntimeError):
    """Raised when an orientation result cannot be produced honestly."""


LEGACY_FOLD_NOTE = (
    "the legacy takes the absolute value of both components before the arctangent "
    "(orientation_radians.m:19), folding every step into the first quadrant. Steps toward "
    "the pole and toward midcell then carry the same angle, so the quantity is axial "
    "anisotropy and not directionality. Both are reported here")


def step_vectors(canonical: pd.DataFrame, *, along: str = "L_um", across: str = "D_um",
                 lag: int = 1, min_step_um: float = 0.0) -> pd.DataFrame:
    """One row per step: its displacement along and across the cell axis.

    Coordinates must already be in the cell frame, which is what `cellcoords.project_tracks`
    supplies. A zero-length step is dropped rather than given an angle, since it has none;
    the count dropped is reported on `.attrs`.
    """
    for column in ("uid", along, across):
        if column not in canonical.columns:
            raise DirectionalityError(f"missing column {column!r}")
    if lag < 1:
        raise DirectionalityError("the lag must be at least one frame")

    rows = []
    for uid, part in without_attrs(canonical).groupby("uid", observed=True, sort=False):
        part = part.sort_values("frame") if "frame" in part.columns else part
        a = part[along].to_numpy(dtype=float)
        b = part[across].to_numpy(dtype=float)
        if len(a) <= lag:
            continue
        frame = (part["frame"].to_numpy()[:-lag] if "frame" in part.columns
                 else np.arange(len(a) - lag))
        row = {"uid": uid, "frame": frame,
               "d_along_um": a[lag:] - a[:-lag], "d_across_um": b[lag:] - b[:-lag]}
        for extra in ("cell", "day", "movie", "condition"):
            if extra in part.columns:
                row[extra] = part[extra].to_numpy()[:-lag]
        rows.append(pd.DataFrame(row))
    if not rows:
        raise DirectionalityError(f"no track has more than {lag} localisation(s)")

    out = pd.concat(rows, ignore_index=True)
    out["step_um"] = np.hypot(out["d_along_um"], out["d_across_um"])
    n_before = len(out)
    out = out[out["step_um"] > max(min_step_um, 0.0)].reset_index(drop=True)
    if out.empty:
        raise DirectionalityError(
            f"every one of {n_before} steps is at or below {min_step_um} um; a step of "
            f"zero length has no orientation")
    out.attrs["n_steps_before_length_filter"] = int(n_before)
    out.attrs["n_steps_kept"] = int(len(out))
    out.attrs["n_steps_without_an_orientation"] = int(n_before - len(out))
    out.attrs["min_step_um"] = float(min_step_um)
    return out


def orientation(steps: pd.DataFrame) -> np.ndarray:
    """The SIGNED angle of each step from the cell's long axis, in (-pi, pi].

    ``atan2`` keeps the quadrant, so a step toward one pole is distinguishable from a step
    toward the other, and a step toward the membrane from one toward the axis. This is the
    quantity a directional question needs and the one the legacy fold destroys.
    """
    return np.arctan2(steps["d_across_um"].to_numpy(dtype=float),
                      steps["d_along_um"].to_numpy(dtype=float))


def axial_angle(steps: pd.DataFrame) -> np.ndarray:
    """The legacy's folded angle, in [0, pi/2]. ``orientation_radians.m`` reproduced.

    Zero is a step lying exactly along the cell, pi/2 exactly across it. Reproduced so the
    old histogram can be recovered and so the gap to the signed angle is measurable.
    """
    return np.arctan2(np.abs(steps["d_across_um"].to_numpy(dtype=float)),
                      np.abs(steps["d_along_um"].to_numpy(dtype=float)))


def axial_order(steps: pd.DataFrame) -> float:
    """One number for how axial the steps are: the mean of ``cos(2*theta)``.

    +1 is every step exactly along the cell, -1 exactly across it, 0 no axial preference.
    The doubled angle is what makes it AXIAL: a step and its reverse contribute equally,
    which is the honest summary of a quantity whose sign was never resolvable.

    A descriptive statistic, not a test. It carries no p value and no interval, because
    pooled steps are not independent. Build the interval over units with ``by_unit``.
    """
    if "sq_disp_um2" in steps.columns:
        raise DirectionalityError(
            "this frame carries pooled squared displacements. An orientation summary on "
            "pooled steps is descriptive only; an interval or a test must be built over "
            "cells or replicates with by_unit.")
    if steps.empty:
        raise DirectionalityError("no steps")
    return float(np.mean(np.cos(2.0 * axial_angle(steps))))


def by_unit(steps: pd.DataFrame, *, unit: str = "cell",
            min_steps: int = 20) -> pd.DataFrame:
    """One axial order per unit, so an interval can be built over units.

    Units below ``min_steps`` are reported and excluded rather than dropped silently: an
    axial order from a handful of steps is noise, and a unit that contributed nothing must
    be visible.
    """
    if unit not in steps.columns:
        raise DirectionalityError(
            f"no {unit!r} column. The unit is the cell or the replicate, never the "
            f"pooled step.")
    rows, thin = [], []
    for key, part in without_attrs(steps).groupby(unit, observed=True, sort=True):
        if len(part) < min_steps:
            thin.append({"unit": key, "n_steps": int(len(part))})
            continue
        rows.append({"unit": key, "n_steps": int(len(part)),
                     "axial_order": axial_order(part),
                     "median_step_um": float(part["step_um"].median())})
    if not rows:
        raise DirectionalityError(
            f"no {unit_word(unit)} reached {min_steps} steps; there is nothing to build an interval "
            f"over")
    out = pd.DataFrame(rows)
    out.attrs["unit"] = unit
    out.attrs["min_steps"] = int(min_steps)
    out.attrs["n_units_below_min_steps"] = len(thin)
    out.attrs["units_below_min_steps"] = thin
    return out


def axial_null(steps: pd.DataFrame, *, D_um2_per_s: float, tau_s: float,
               cell_length_um: float, cell_width_um: float, n_tracks: int = 2000,
               seed: int = 0) -> dict:
    """The axial order a FREE confined walk produces in the same cell, and the excess.

    A rod-shaped cell has more room along its axis than across it and reflects motion back
    along it, so a walk with no preference of its own still comes out axial. Without this
    number an observed axial order means nothing, exactly as a confined fraction means
    nothing without its chance rate.

    ``D_um2_per_s`` has no default. It must come from the analysis, not from this
    function's imagination.
    """
    if D_um2_per_s <= 0 or tau_s <= 0:
        raise DirectionalityError("D and tau must be positive and must come from the "
                                  "analysis")
    if cell_length_um <= 0 or cell_width_um <= 0:
        raise DirectionalityError("the cell dimensions must be positive")

    rng = np.random.default_rng(seed)
    lengths = (steps.groupby("uid").size().to_numpy()
               if "uid" in steps.columns else np.array([], dtype=int))
    n_steps = (rng.choice(lengths, size=n_tracks, replace=True)
               if len(lengths) else np.full(n_tracks, 5, dtype=int))
    n_steps = np.clip(np.asarray(n_steps, dtype=int), 1, None)

    sigma = np.sqrt(2.0 * D_um2_per_s * tau_s)
    half = np.array([cell_length_um, cell_width_um], dtype=float) / 2.0

    # VECTORISED, AND THE REFLECTION IS A TRIANGLE WAVE RATHER THAN A LOOP. Bouncing a
    # position back one wall at a time is correct but costs a Python iteration per step,
    # which on real track-length distributions is millions of them. Folding the CUMULATIVE
    # position through a triangle wave of period 4h is the standard construction of a
    # reflected random walk and handles an excursion of any size in one operation.
    width = int(n_steps.max()) + 1
    starts = rng.uniform(-half, half, size=(len(n_steps), 2))
    increments = rng.normal(0.0, sigma, size=(len(n_steps), width, 2))
    increments[:, 0, :] = 0.0
    walk = starts[:, None, :] + np.cumsum(increments, axis=1)

    period = 4.0 * half
    folded = np.mod(walk + half, period)
    walk = half - np.abs(folded - 2.0 * half)

    # Only the steps each track actually has; the rest of the padded width is dropped.
    step = np.diff(walk, axis=1)
    valid = np.arange(width - 1)[None, :] < n_steps[:, None]
    d_along = step[:, :, 0][valid]
    d_across = step[:, :, 1][valid]

    null_steps = pd.DataFrame({"d_along_um": d_along, "d_across_um": d_across})
    null_steps["step_um"] = np.hypot(null_steps["d_along_um"],
                                     null_steps["d_across_um"])
    null_steps = null_steps[null_steps["step_um"] > 0]
    observed = axial_order(steps)
    chance = axial_order(null_steps)
    return {
        "observed": observed,
        "chance": chance,
        "excess": observed - chance,
        "n_steps_observed": int(len(steps)),
        "n_steps_null": int(len(null_steps)),
        "D_um2_per_s": float(D_um2_per_s),
        "tau_s": float(tau_s),
        "cell_length_um": float(cell_length_um),
        "cell_width_um": float(cell_width_um),
        "aspect_ratio": float(cell_length_um / cell_width_um),
        "sigma_step_um": float(sigma),
        "caveat": (
            "a rod-shaped cell produces axial anisotropy with no biology in it: there is "
            "more room along the axis and the walls reflect motion back along it. Quote "
            "the excess with its chance rate and never the observed value alone"),
        "fold_note": LEGACY_FOLD_NOTE,
    }
