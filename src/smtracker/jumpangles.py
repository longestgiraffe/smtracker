"""Angles between consecutive jumps, with the chance rate they are meaningless without.

WHAT THIS MEASURES. For two consecutive displacements of one track, the angle between
them: 0 means the molecule carried straight on, 180 means it reversed. A free Brownian
walk in unbounded space gives a flat distribution, so any excess at 180 is "compact"
motion -- the molecule revisiting where it has been. The quantity and the polar
histogram follow TrackIt (Kuhn et al., Sci Rep 2021, 11:9465), which follows Burov et
al. (PNAS 2013).

IT REPLACES THE LEGACY DIRECTIONALITY ANALYSIS RATHER THAN ADDING TO IT.
`orientation_radians.m` takes `atan(|dy|/|dx|)` -- the absolute value on BOTH components
-- so every step lands in the first quadrant and a step toward the pole is
indistinguishable from a step toward midcell. What survives there is axial anisotropy,
not direction. An angle BETWEEN consecutive jumps needs no such fold.

TWO THINGS PRODUCE A BACKWARD BIAS WITH NO BIOLOGY IN THEM, AND IN A ROD THEY ARE BOTH
LARGE. They are reported as ONE null, not two, because they interact -- a molecule
reflecting off a cell wall is also localised with error, and two analytic curves drawn on
one axis invite a reader to add them.

  CELL GEOMETRY   TrackIt measures in a mammalian nucleus, where the boundary is far away
                  compared with one step. In a rod at tau = 20.47 ms the RMS 2-D step is
                  76 nm for D = 0.07 and 243 nm for D = 0.72 -- 15 % and 48 % of a 0.5 um
                  half-width. A wall reflects motion back, which is a 180-degree jump
                  angle, and it does so FAR more for the fast state.
  LOCALISATION    Consecutive displacements SHARE a localisation, so their errors are
  ERROR           anti-correlated with covariance -sigma^2. The step autocorrelation is
                  -sigma^2 / (2 D tau + 2 sigma^2): at sigma = 17 nm that is -0.084 for
                  D = 0.07 and -0.010 for D = 0.72, a factor of 8.7. **The slow state
                  looks nearly nine times more backward-biased than the fast state from
                  noise alone**, which is exactly the comparison a reader wants to make.

So: never the histogram alone, and never a slow-against-fast comparison without the null
for each. `angle_null` refuses to default D or sigma for the same reason `confinement_null`
refuses to default D -- a chance rate computed from the function's imagination is not a
chance rate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "JumpAngleError",
    "jump_angles",
    "angle_histogram",
    "asymmetry_ratio",
    "angle_null",
    "angles_by_unit",
]

#: Half-width of the forward and backward windows used by :func:`asymmetry_ratio`,
#: in degrees. 30 is TrackIt's own convention and is stated on every result rather
#: than hidden, because the ratio moves with it.
DEFAULT_WINDOW_DEG = 30.0

#: Bins of the polar histogram over [0, 180]. A jump angle has no sign that survives
#: -- clockwise and anticlockwise are the same event with the cell flipped -- so the
#: histogram is over the UNSIGNED angle and spans half a turn, not a full one.
DEFAULT_N_BINS = 18


class JumpAngleError(RuntimeError):
    """Raised when angles cannot be computed honestly."""


def jump_angles(steps: pd.DataFrame, *, along: str = "d_along_um",
                across: str = "d_across_um", uid: str = "uid",
                min_step_um: float = 0.0) -> pd.DataFrame:
    """The angle between each consecutive pair of steps within a track, in degrees.

    ``steps`` is one row per step, in track order, as produced by
    ``directionality.step_vectors``. Pairs are formed WITHIN a track only; a pair
    spanning a frame gap is dropped, because the second step did not follow the first.

    ``min_step_um`` drops pairs in which either step is shorter than the threshold. A
    very short step has an ill-determined direction -- its angle is dominated by
    localisation error -- so including it adds noise at 180 degrees specifically. The
    default of 0 keeps everything and lets the null carry the bias instead, which is the
    more honest arrangement; raise it only as a sensitivity check.
    """
    for column in (along, across):
        if column not in steps.columns:
            raise JumpAngleError(
                f"no '{column}' column; pass the output of "
                f"directionality.step_vectors, which carries the step's components "
                f"along and across the cell axis")
    if uid not in steps.columns:
        raise JumpAngleError(f"no '{uid}' column: pairs must be formed within a track")

    frame = steps.reset_index(drop=True)
    v = frame[[along, across]].to_numpy(float)
    track = frame[uid].to_numpy()

    # CONSECUTIVE WITHIN ONE TRACK. Without this the last step of one molecule pairs
    # with the first of the next, which is not an angle of anything.
    same = track[:-1] == track[1:]
    if "frame" in frame.columns:
        # AND CONTIGUOUS IN TIME. A pair spanning a gap is not consecutive motion; the
        # molecule had two frames to turn around in. u-track with timeWindow=1 produces
        # no gaps, but TrackMate closes them, so this cannot be left to the tracker.
        f = frame["frame"].to_numpy(float)
        lag = frame["lag_frames"].to_numpy(float) if "lag_frames" in frame else 1.0
        same &= np.isclose(f[1:] - f[:-1], lag if np.isscalar(lag) else lag[:-1])

    a, b = v[:-1][same], v[1:][same]
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    keep = (na > min_step_um) & (nb > min_step_um) & (na > 0) & (nb > 0)
    a, b, na, nb = a[keep], b[keep], na[keep], nb[keep]
    if not len(a):
        raise JumpAngleError(
            "no consecutive step pairs survived; are the steps in track order, and does "
            "min_step_um exceed the step lengths?")

    cos = np.clip((a * b).sum(axis=1) / (na * nb), -1.0, 1.0)
    angle = np.degrees(np.arccos(cos))

    out = pd.DataFrame({"angle_deg": angle,
                        "step_prev_um": na, "step_next_um": nb})
    for carry in (uid, "cell", "movie", "day", "condition", "protein"):
        if carry in frame.columns:
            out[carry] = frame[carry].to_numpy()[:-1][same][keep]

    # THE ANGLE'S POSITION IS THE VERTEX, WHICH IS THE MIDDLE OF THREE LOCALISATIONS.
    # Steps i and i+1 run p_i -> p_{i+1} -> p_{i+2}, so the turn happens AT p_{i+1} --
    # the start of the SECOND step, not the first. Taking the first step's position would
    # place every turn one step upstream of where it occurred, which on a 64 nm step is
    # 64 nm of systematic error in exactly the quantity a spatial map of turns is for.
    # `step_vectors` labels a step by its start, so the vertex columns come from [1:].
    for position in ("l_norm", "d_norm", "L_um", "D_um", "x_um", "y_um"):
        if position in frame.columns:
            out[position] = frame[position].to_numpy()[1:][same][keep]
    if "frame" in frame.columns:
        out["vertex_frame"] = frame["frame"].to_numpy()[1:][same][keep]
    out.attrs["min_step_um"] = float(min_step_um)
    out.attrs["position_convention"] = "vertex: the shared localisation of the two steps"
    return out


def angle_histogram(angles, *, n_bins: int = DEFAULT_N_BINS) -> pd.DataFrame:
    """Normalised angle distribution over [0, 180], ready to draw.

    Returned as a density (bin heights sum to 1 when multiplied by the bin width) so an
    observed set and a null with different n can be drawn on one axis.
    """
    a = np.asarray(angles["angle_deg"] if isinstance(angles, pd.DataFrame) else angles,
                   dtype=float)
    if not len(a):
        raise JumpAngleError("no angles to bin")
    edges = np.linspace(0.0, 180.0, n_bins + 1)
    counts, _ = np.histogram(a, bins=edges)
    width = edges[1] - edges[0]
    return pd.DataFrame({
        "angle_deg": 0.5 * (edges[1:] + edges[:-1]),
        "count": counts,
        "density": counts / (counts.sum() * width),
    })


def asymmetry_ratio(angles, *, window_deg: float = DEFAULT_WINDOW_DEG) -> dict:
    """f(180 +- w) / f(0 +- w): backward over forward.

    1 means no preference. Above 1 means the molecule reverses more often than it carries
    on, which is what confinement, a wall, and localisation error all produce. The window
    is returned with the number because the ratio moves with it.
    """
    a = np.asarray(angles["angle_deg"] if isinstance(angles, pd.DataFrame) else angles,
                   dtype=float)
    if not len(a):
        raise JumpAngleError("no angles")
    forward = int((a <= window_deg).sum())
    backward = int((a >= 180.0 - window_deg).sum())
    if forward == 0:
        raise JumpAngleError(
            f"no steps within {window_deg:g} degrees of forward, so the ratio is "
            f"undefined, not infinite ({backward} backward of {len(a)})")
    return {
        "ratio": backward / forward,
        "n_forward": forward,
        "n_backward": backward,
        "n_angles": int(len(a)),
        "window_deg": float(window_deg),
    }


def angle_null(steps: pd.DataFrame, *, D_um2_per_s: float, tau_s: float,
               localization_sigma_um: float, cell_length_um: float,
               cell_width_um: float, n_tracks: int = 2000, seed: int = 0,
               n_bins: int = DEFAULT_N_BINS,
               window_deg: float = DEFAULT_WINDOW_DEG) -> dict:
    """The angle distribution a FREE walk produces in this cell, at this noise level.

    ONE NULL, NOT TWO. Cell geometry and localisation error both bias angles backward and
    they are not additive: a molecule reflecting off a wall is also localised with error,
    and the second effect is applied to the positions the first produced. Simulating both
    together is the only arrangement that gives a curve a reader can subtract.

    NOTHING HERE HAS A DEFAULT. ``D`` must come from the analysis; the null's whole value
    is that it is matched to the data, and a guessed D makes it a measurement of the
    guess. ``localization_sigma_um`` is likewise required -- pass 0 deliberately to
    isolate the geometry term, never by omission.

    The walk is reflected off the cell walls with a triangle wave, the same construction
    `directionality.axial_null` uses, then each POSITION is perturbed independently by
    ``localization_sigma_um``. Perturbing positions rather than steps is what creates the
    -sigma^2 covariance between consecutive displacements; adding noise to steps directly
    would make them independent and would silently drop the whole second effect.
    """
    if D_um2_per_s <= 0:
        raise JumpAngleError("D must be positive and must come from the analysis")
    if tau_s <= 0:
        raise JumpAngleError("tau_s must be positive")
    if localization_sigma_um < 0:
        raise JumpAngleError("localization_sigma_um cannot be negative")
    if not np.isfinite([cell_length_um, cell_width_um]).all():
        # A NON-FINITE CELL IS A REFUSAL, NOT AN OverflowError FROM INSIDE numpy.
        # This project has at least one mesh whose fitted length comes back inf, and a
        # median taken over it is NaN; without this the failure surfaces six frames down
        # in `rng.uniform` as "Range exceeds valid bounds", which names nothing.
        raise JumpAngleError(
            f"cell dimensions are not finite ({cell_length_um}, {cell_width_um}); a mesh "
            f"with a non-finite fitted length poisons a median; take it over the finite "
            f"cells only")
    if cell_length_um <= 0 or cell_width_um <= 0:
        raise JumpAngleError("cell dimensions must be positive")

    rng = np.random.default_rng(seed)
    observed = jump_angles(steps) if "angle_deg" not in steps.columns else steps

    # Track lengths matched to the data, so the null has the same pair budget per track.
    if "uid" in steps.columns:
        lengths = steps.groupby("uid", observed=True).size().to_numpy()
    else:
        lengths = np.array([max(len(steps) // max(n_tracks, 1), 2)])
    n_steps = rng.choice(np.clip(lengths, 2, None), size=n_tracks, replace=True)

    step_sigma = np.sqrt(2.0 * D_um2_per_s * tau_s)
    half = np.array([cell_length_um, cell_width_um], dtype=float) / 2.0
    width = int(n_steps.max()) + 1

    starts = rng.uniform(-half, half, size=(n_tracks, 2))
    increments = rng.normal(0.0, step_sigma, size=(n_tracks, width, 2))
    increments[:, 0, :] = 0.0
    walk = starts[:, None, :] + np.cumsum(increments, axis=1)

    period = 4.0 * half
    folded = np.mod(walk + half, period)
    walk = half - np.abs(folded - 2.0 * half)

    # THE NOISE GOES ON THE POSITIONS. This is the whole point: it makes consecutive
    # displacements share an error term and therefore anti-correlate.
    if localization_sigma_um > 0:
        walk = walk + rng.normal(0.0, localization_sigma_um, size=walk.shape)

    step = np.diff(walk, axis=1)
    valid = np.arange(width - 1)[None, :] < n_steps[:, None]
    uid = np.repeat(np.arange(n_tracks), width - 1).reshape(n_tracks, width - 1)
    null_steps = pd.DataFrame({
        "uid": uid[valid],
        "d_along_um": step[:, :, 0][valid],
        "d_across_um": step[:, :, 1][valid],
    })
    # THE NULL'S OWN VERTEX POSITIONS, so a spatial map of turns has a comparator.
    # Without these the observed map can only be read against a uniform expectation, and
    # a free walk in a rod does NOT turn back uniformly -- it turns back at the walls,
    # which is the whole hypothesis such a map is meant to test.
    vertex = walk[:, 1:-1, :]
    lnorm = (vertex[:, :, 0] + half[0]) / (2.0 * half[0])
    dnorm = vertex[:, :, 1] / half[1]
    pad = np.full((n_tracks, 1), np.nan)
    null_steps["l_norm"] = np.concatenate([lnorm, pad], axis=1)[valid]
    null_steps["d_norm"] = np.concatenate([dnorm, pad], axis=1)[valid]
    null_angles = jump_angles(null_steps)

    obs_hist = angle_histogram(observed, n_bins=n_bins)
    null_hist = angle_histogram(null_angles, n_bins=n_bins)
    obs_ratio = asymmetry_ratio(observed, window_deg=window_deg)
    null_ratio = asymmetry_ratio(null_angles, window_deg=window_deg)

    # The analytic step autocorrelation, for the record: it is what makes the slow state
    # look backward-biased, and it is checkable against the simulation.
    predicted = (-localization_sigma_um ** 2
                 / (2.0 * D_um2_per_s * tau_s + 2.0 * localization_sigma_um ** 2))

    return {
        "null_angles": null_angles,
        "observed_hist": obs_hist,
        "null_hist": null_hist,
        "observed_ratio": obs_ratio["ratio"],
        "null_ratio": null_ratio["ratio"],
        "excess_ratio": obs_ratio["ratio"] - null_ratio["ratio"],
        "n_angles_observed": obs_ratio["n_angles"],
        "n_angles_null": null_ratio["n_angles"],
        "D_um2_per_s": float(D_um2_per_s),
        "tau_s": float(tau_s),
        "localization_sigma_um": float(localization_sigma_um),
        "cell_length_um": float(cell_length_um),
        "cell_width_um": float(cell_width_um),
        "rms_step_um": float(np.sqrt(4.0 * D_um2_per_s * tau_s)),
        "step_as_fraction_of_half_width": float(
            np.sqrt(4.0 * D_um2_per_s * tau_s) / (cell_width_um / 2.0)),
        "predicted_step_autocorrelation": float(predicted),
        "window_deg": float(window_deg),
        "caveat": (
            "a rod-shaped cell and a shared localisation both bias jump angles backward "
            "without biology. Quote the excess over this matched null, never the observed "
            "ratio alone; a slow and a fast population each need their own null"),
    }


def angles_by_unit(angles: pd.DataFrame, *, unit: str = "cell",
                   min_angles: int = 20,
                   window_deg: float = DEFAULT_WINDOW_DEG) -> pd.DataFrame:
    """One asymmetry ratio per independent unit, so an interval can be built over units.

    Angles within a track are not independent -- consecutive pairs share a step -- and
    tracks within a cell share an environment, so the unit is the cell or the replicate
    day and never the angle. A unit with too few angles is returned with ``ok = False``
    rather than dropped.
    """
    if unit not in angles.columns:
        raise JumpAngleError(f"no '{unit}' column to form units from")
    rows = []
    for key, part in angles.groupby(unit, observed=True):
        row = {unit: key, "n_angles": int(len(part)), "ok": len(part) >= min_angles}
        if row["ok"]:
            try:
                row["ratio"] = asymmetry_ratio(part, window_deg=window_deg)["ratio"]
            except JumpAngleError:
                row["ok"] = False
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs["window_deg"] = float(window_deg)
    out.attrs["min_angles"] = int(min_angles)
    return out
