"""The legacy window's SDA tab: SMTracker 2.0's Spatial Distribution Analysis, computed the old way.

Ruled 26 Nov 2026 (`docs/legacy_window_plan.md`). 2.0's tab (`SMTracker.m:941-1008`, callbacks `SDA`, `listDatasource`,
`listTypeotion`, `listCellsize`, `binwidthChng`, `individual_heatmap`, `reset_heatmaps`, `barplotDistribution`;
`Help_fcn/getNormalizeTracks.m`, `normalizeTracks.m`, `getDynamics.m`, `getAnomalousTracks.m`, `plotHeatmap.m`,
`mirrorPoints.m`, `bin3d.m`; manual Fig. 16, p. 21-22) projects every track of a condition onto a standard 1 x 3 um cell
and shows where the molecules are (spot location), or how fast, how fast changing and how diffusive they are
(velocity, acceleration, diffusion) in each part of it: a heat map, the same with the cell mirrored about both its
axes, and the two distributions along the cell's axes. Headless here; `ui.legacy.tab_sda` draws it.

What is kept as 2.0 had it (the window shows the old program):
- the model cell is 2.0's `modelCell.mat` contour, a capsule 1 long and 1 wide drawn 3 : 1; a localisation outside it
  is not used (`inCellContour`); positions are normalised by each cell's length (along) and width (across);
- the location map is a 20 x 20 `hist3` on x in [-0.1, 1.1], y in [-0.6, 0.6], up-scaled by cubic interpolation from 20
  to 609 nodes and divided by the number of points; mirrored it is the MAXIMUM of the four flipped images
  (`plotHeatmap.m:15-25`: the old look, the defects the legacy package records as 3.1 to 3.3 are 2.0's and stay);
- the dynamics maps are `bin3d`'s 30 x 10 mean per bin over the DATA's own range, one bin per cell of the
  `pcolor` (an edge point lands in both bins it touches, `bin3d.m:12-13`), the mirrored one on the four-fold
  replicated points (`mirrorPoints.m`); each point is the MIDPOINT of its step (`getDynamics.m:8`);
- 2.0's velocity is ``|dx + dy| / tau`` and its acceleration ``|dx + dy| / tau^2`` (the SUM of the two displacement
  components, not their length: `getDynamics.m:11-12`), the diffusion ``(dx^2 + dy^2) / (4 tau)``; shown as 2.0
  computed them, so a "velocity" here is not a speed;
- a motion type (Brownian, Super-diffusive, Sub-diffusive) selects the tracks 2.0's MSD tab classified one by one with
  `diffusionClassification` (tracks of at least 9 localisations, the MSD tab's default, the last lag left out, the
  weights 1 / std, `fitMSDTracks.m`): `track_motion_types` runs that classification (`smtracker.legacy.sla`), and a
  track it cannot classify (no logarithm of a non-positive MSD) is left out and counted;
- cell sizes are 2.0's percentile groups of the cell lengths: small up to the 25th percentile, medium between the
  25th and 75th, large above (`popPopUpCellLength`, `getTracksByCellsize`);
- the colour limits only ever grow and persist between draws (`plotHeatmap.m:65-70`), one per source, until
  "Reset CB limits"; here the two maps of one draw share ONE limit, so the one colour bar the screen shows holds for
  both (2.0 raised it between the two draws, leaving the first on the old scale).

Where this window departs from 2.0, and why:
- 3D: 2.0 drew `dscatter`'s surface for the location and a flat `pcolor` seen from an angle for the dynamics; both are
  drawn here as a surface over the same grid (the value as height), which is what the 3D view is for;
- "Individual cells" asks for a folder and writes there (`legacy_` in every name); 2.0 wrote into the condition's
  own folder, which this window never does.

INTERFACES
- `SOURCES`, `MOTION_TYPES`, `SIZE_CHOICES`, `model_cell_contour()`, `inside_model_cell(x, y)`.
- `size_labels(project, condition)`, `track_motion_types(project, condition)`.
- `sda_run(project, condition, source, motion, size, motion_types=None)`: the points (and values) a draw uses.
- `location_map(x, y, mirror)` / `dynamics_map(x, y, z, mirror)`: the two kinds of heat map.
- `location_distribution(x, y, n_bins)` / `dynamics_distribution(x, y, z)`: `barplotDistribution`.
- `grow_clim(clim, *maxima)`, `individual_cells(project, condition)`.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .legacy_view import LegacyViewError
from .legacy_view_sla import distribution_hist, interp_cubic

__all__ = ["SOURCES", "MOTION_TYPES", "SIZE_CHOICES", "SOURCE_UNITS", "MIN_LENGTH_FOR_MOTION", "model_cell_contour",
           "inside_model_cell", "size_labels", "track_motion_types", "sda_run", "location_map", "dynamics_map",
           "location_distribution", "dynamics_distribution", "grow_clim", "individual_cells", "mirror_points"]

#: 2.0's popups (`SMTracker.m:965-967`).
SOURCES = ("spot location", "velocity", "acceleration", "diffusion")
MOTION_TYPES = ("Brownian", "Super-diffusive", "Sub-diffusive", "All")
SIZE_CHOICES = ("all sizes", "small cells", "medium cells", "large cells")
#: The axis labels `barplotDistribution` gives each dynamics source (mathtext).
SOURCE_UNITS = {"velocity": r"Average Velocity [$\mu$m s$^{-1}$]",
                "acceleration": r"Average Acceleration [$\mu$m s$^{-2}$]",
                "diffusion": r"Average Diffusion [$\mu$m$^{2}$ s$^{-1}$]"}
#: The MSD tab's minimum track length for the anomalous analysis (`popMSD`, `SMTracker.m:3465`) and the percentiles
#: that split the cells by length.
MIN_LENGTH_FOR_MOTION = 9
SIZE_PERCENTILES = (25.0, 75.0)

#: The lower half of 2.0's `modelCell.mat` contour (`SMT_GUI/Help_fcn/config`, 96 points, x along 0 to 1, y across
#: -0.5 to 0.5); the upper half is its mirror image about y = 0.
_MODEL_LOWER = (
    (0.0, 0.0), (0.0033, -0.0997), (0.0108, -0.1945), (0.0225, -0.2812), (0.0382, -0.3562), (0.0573, -0.4158),
    (0.0788, -0.458), (0.1018, -0.4838), (0.1255, -0.4962), (0.1492, -0.5), (0.1731, -0.5), (0.1969, -0.5),
    (0.2208, -0.5), (0.2446, -0.5), (0.2684, -0.5), (0.2923, -0.5), (0.3161, -0.5), (0.3399, -0.5), (0.3638, -0.5),
    (0.3876, -0.5), (0.4114, -0.5), (0.4353, -0.5), (0.4591, -0.5), (0.4829, -0.5), (0.5068, -0.5), (0.5306, -0.5),
    (0.5544, -0.5), (0.5783, -0.5), (0.6021, -0.5), (0.6259, -0.5), (0.6498, -0.5), (0.6736, -0.5), (0.6975, -0.5),
    (0.7213, -0.5), (0.7451, -0.5), (0.769, -0.5), (0.7928, -0.5), (0.8166, -0.5), (0.8405, -0.5), (0.8642, -0.4988),
    (0.888, -0.4912), (0.9114, -0.4726), (0.9338, -0.4391), (0.9543, -0.3886), (0.9755, -0.3218), (0.9862, -0.2408),
    (0.9954, -0.1479), (1.0, -0.0498))


def model_cell_contour() -> np.ndarray:
    """2.0's model cell outline, a closed ``(96, 2)`` polygon in units of the cell's length and width."""
    lower = np.asarray(_MODEL_LOWER, dtype=float)
    upper = lower[::-1] * np.array([1.0, -1.0])
    return np.vstack([lower, upper])


def inside_model_cell(x, y) -> np.ndarray:
    """``inpolygon(x, y, contour)``: True where a point lies in (or on) the model cell."""
    from matplotlib.path import Path

    pts = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
    path = Path(model_cell_contour())
    return path.contains_points(pts) | path.contains_points(pts, radius=1e-9) | path.contains_points(pts, radius=-1e-9)


def mirror_points(xy: np.ndarray) -> np.ndarray:
    """`mirrorPoints.m`: every point, its reflection across the long axis, across the short axis and across both."""
    xy = np.asarray(xy, dtype=float)
    x, y = xy[:, 0], xy[:, 1]
    return np.vstack([np.column_stack([x, y]), np.column_stack([x, -y]), np.column_stack([1.0 - x, y]),
                      np.column_stack([1.0 - x, -y])])


# --------------------------------------------------------------------------------------------- #
# The tracks on the standard cell
# --------------------------------------------------------------------------------------------- #

def _cells(project, condition: str):
    """``(table, geometry)``: the condition's localisations in 2.0's normalised coordinates and each cell's measures."""
    from .legacy_view_sla import _normalised_table

    table = _normalised_table(project, condition)
    return table, table.attrs.get("geometry") or {}


def _length_limits(project, condition: str) -> tuple:
    """2.0's small / medium / large limits (`popPopUpCellLength`): the 25th and 75th percentile of the lengths of
    every outlined cell of the condition, in um."""
    from . import legacy_view

    lengths = legacy_view.cell_lengths(project, condition)
    if not len(lengths):
        raise LegacyViewError(f"{condition}: no outlined cell has a measurable length")
    lo, hi = np.percentile(lengths, SIZE_PERCENTILES)
    return float(lo), float(hi), lengths


def size_labels(project, condition: str) -> list:
    """The cell-size popup's four entries with their limits, as `popPopUpCellLength` words them."""
    lo, hi, _ = _length_limits(project, condition)
    return ["all sizes", f"small cells (< {lo:.3g} µm)", f"medium cells (between {lo:.3g} and {hi:.3g} µm)",
            f"large cells (> {hi:.3g} µm)"]


def _cell_in_size(length_um: float, size: int, lo: float, hi: float) -> bool:
    if size == 0:
        return True
    if size == 1:
        return length_um <= lo
    if size == 2:
        return lo < length_um <= hi
    return length_um > hi


def track_motion_types(project, condition: str) -> tuple:
    """2.0's per-track motion type for the condition (1 Brownian, 2 Super-diffusive, 3 Sub-diffusive), as the MSD
    tab's anomalous analysis assigns it (`fitMSDTracks.m`, case 2): each track's own MSD at lags 1 to 8 (the 9-point
    tracks of "specific tau"), the last lag left out, weights 1 / std (a lag on one pair takes `linWfit`'s
    replacement), the F-test of `diffusionClassification`. Returns ``(types, n_unclassified)``: ``{uid: type}`` for the
    tracks of at least 9 localisations and the number of those that could not be classified (and so are never drawn
    under a motion type)."""
    from .canonical import without_attrs
    from .msd import msd_per_track
    from .parallel import condition_map

    table = project.condition(condition)
    lengths = without_attrs(table).groupby("uid", observed=True).size()
    keep = lengths.index[lengths >= MIN_LENGTH_FOR_MOTION]
    if not len(keep):
        raise LegacyViewError(f"{condition}: no track reaches {MIN_LENGTH_FOR_MOTION} localisations, so none can be "
                              f"classified as Brownian, super- or sub-diffusive")
    curves = msd_per_track(table[table["uid"].isin(keep)], max_lag=MIN_LENGTH_FOR_MOTION - 1, pairing="frame")
    items = []
    for uid, part in without_attrs(curves).groupby("uid", sort=False, observed=True):
        part = part.sort_values("lag_frames")
        t = part["lag_time_s"].to_numpy(dtype=float)
        y = part["msd_um2"].to_numpy(dtype=float)
        std = np.nan_to_num(part["std_um2"].to_numpy(dtype=float), nan=0.0)
        with np.errstate(divide="ignore"):
            w = 1.0 / std
        finite = np.isfinite(w)
        if not finite.all() and finite.any():
            w[~finite] = 1.0 / np.sum(1.0 / w[finite])
        items.append((uid, t[:-1], y[:-1], w[:-1]))                  # excludeLast
    # the fit of each track is a three-parameter trust-region fit, 0.1 s each: a pool of workers does the tracks
    chunks = [items[i:i + _CHUNK] for i in range(0, len(items), _CHUNK)]
    types, failed = {}, 0
    for chunk in condition_map(_classify_chunk, chunks, shared={}, n_jobs=None):
        for uid, code in chunk:
            if code is None:
                failed += 1
            else:
                types[uid] = code
    return types, failed


#: Tracks to a worker's job in `track_motion_types`.
_CHUNK = 60
_MOTION_CODE = {"Brownian": 1, "Super-diffusive": 2, "Sub-diffusive": 3}


def _classify_chunk(chunk, _shared) -> list:                 # pragma: no cover - runs in a pool worker too
    """`[(uid, code or None)]` for a chunk of ``(uid, t, msd, weight)``: `diffusionClassification` on each track's own
    curve; None where the F-test cannot be made (fewer than four usable lags, a non-positive MSD)."""
    from .legacy import sla

    out = []
    for uid, t, y, w in chunk:
        ok = np.isfinite(t) & np.isfinite(y) & np.isfinite(w) & (w > 0)
        if ok.sum() < 4:
            out.append((uid, None))
            continue
        try:
            verdict = sla.classify_motion(np.column_stack([t[ok], y[ok], w[ok]]), weights="v2")
        except sla.SLAError:
            out.append((uid, None))
            continue
        out.append((uid, _MOTION_CODE[verdict["motion"]]))
    return out


def _selected(table, geometry, project, condition, *, motion: int, size: int, motion_types):
    """``(selected rows, n tracks, n cells)`` as `getNormalizeTracks.m` picks them: with "All" the tracks of the cells
    of the chosen size, otherwise the tracks classified as that motion (every size)."""
    lo, hi, lengths = _length_limits(project, condition)
    if motion != 3:                                            # 0 Brownian, 1 Super, 2 Sub, 3 All
        if motion_types is None:
            raise LegacyViewError("the motion types of the tracks are needed for this choice")
        want = {0: 1, 1: 2, 2: 3}[motion]
        uids = {u for u, t in motion_types[0].items() if t == want}
        chosen = table[table["uid"].isin(uids)]
    elif size:
        cells = {k for k, g in geometry.items() if _cell_in_size(g.length_um, size, lo, hi)}
        chosen = table[table["cell"].astype(str).isin(cells)]
    else:
        chosen = table
    n_cells = {0: len(lengths), 1: int((lengths <= lo).sum()), 2: int(((lengths > lo) & (lengths <= hi)).sum()),
               3: int((lengths > hi).sum())}[size if motion == 3 else 0]
    return chosen, int(chosen["uid"].nunique()), n_cells


def sda_run(project, condition: str, source: int, motion: int = 3, size: int = 0, motion_types=None) -> dict:
    """The points one SDA draw uses. ``source`` indexes `SOURCES`, ``motion`` `MOTION_TYPES`, ``size``
    `SIZE_CHOICES`. ``motion_types``: `track_motion_types`' result, needed for a motion other than "All". The points
    are inside the model cell only; ``z`` is None for the location and one value per point for the dynamics."""
    if source not in range(4) or motion not in range(4) or size not in range(4):
        raise LegacyViewError("source, motion type and cell size must index their popups")
    table, geometry = _cells(project, condition)
    if source == 0:
        chosen, n_tracks, n_cells = _selected(table, geometry, project, condition, motion=motion, size=size,
                                              motion_types=motion_types)
        x = chosen["nx"].to_numpy(dtype=float)
        y = chosen["ny"].to_numpy(dtype=float)
        inside = inside_model_cell(x, y)
        label = MOTION_TYPES[motion]
        size_word = SIZE_CHOICES[size] if motion == 3 else SIZE_CHOICES[0]
        return {"source": source, "x": x[inside], "y": y[inside], "z": None, "n_tracks": n_tracks, "n_cells": n_cells,
                "title": f"Heat map of {SOURCES[0]} - {label} tracks. ({n_tracks}) in {n_cells} {size_word}",
                "n_unclassified": (motion_types[1] if motion_types is not None and motion != 3 else 0)}
    tau = float(project.params.frame_interval_ms) / 1000.0
    xs, ys, zs = [], [], []
    from .canonical import without_attrs
    from .legacy.sda import midpoint_positions

    for _, track in without_attrs(table).groupby("uid", sort=False, observed=True):
        um = track[["x_um", "y_um"]].to_numpy(dtype=float)
        if len(um) < 2:
            continue
        mid = midpoint_positions(track[["nx", "ny"]].to_numpy(dtype=float))
        step = um[:-1] - um[1:]
        if source == 1:
            value = np.abs(step.sum(axis=1)) / tau
        elif source == 2:
            value = np.abs(step.sum(axis=1)) / tau ** 2
        else:
            value = (step ** 2).sum(axis=1) / (4.0 * tau)
        xs.append(mid[:, 0]); ys.append(mid[:, 1]); zs.append(value)
    if not xs:
        raise LegacyViewError(f"{condition}: no track has two localisations, so no step has a velocity or a diffusion")
    x, y, z = np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)
    keep = inside_model_cell(x, y) & np.isfinite(z)
    return {"source": source, "x": x[keep], "y": y[keep], "z": z[keep], "n_tracks": int(table["uid"].nunique()),
            "n_cells": len(geometry), "title": f"Heat map of {SOURCES[source]} - All tracks", "n_unclassified": 0}


# --------------------------------------------------------------------------------------------- #
# The maps
# --------------------------------------------------------------------------------------------- #

def location_map(x, y, mirror: bool = False) -> dict:
    """`plotHeatmap.m:12-25`: the location density, 20 x 20 `hist3` up-scaled to 609 x 609 and divided by the number of
    points; mirrored, the maximum of the image and its three flips. ``{"x", "y", "image" (rows y), "vmax"}``."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if not len(x):
        raise LegacyViewError("No tracks found with those characteristics")
    ex, ey = np.linspace(-0.1, 1.1, 20), np.linspace(-0.6, 0.6, 20)
    counts, _, _ = np.histogram2d(x, y, bins=[ex, ey])
    v = np.zeros((20, 20))
    v[:19, :19] = counts
    vq = interp_cubic(v, 5) / len(x)
    if mirror:
        vq = np.maximum.reduce([vq, vq[::-1, :], vq[:, ::-1], vq[::-1, ::-1]])
    half = (ex[1] - ex[0]) / 2
    cx = np.concatenate([(ex[:-1] + ex[1:]) / 2, [ex[-1] + half]])
    half = (ey[1] - ey[0]) / 2
    cy = np.concatenate([(ey[:-1] + ey[1:]) / 2, [ey[-1] + half]])
    return {"x": np.linspace(cx[0], cx[-1], vq.shape[0]), "y": np.linspace(cy[0], cy[-1], vq.shape[1]),
            "image": vq.T, "vmax": float(vq.max()), "kind": "location"}


def dynamics_map(x, y, z, mirror: bool = False, n_bins: tuple = (30, 10)) -> dict:
    """`bin3d.m`: the mean of ``z`` in ``n_bins`` bins over the points' own range, a point on an edge in both bins
    that touch it; mirrored on the four-fold replicated points. ``{"x_edges", "y_edges", "mean" (rows y), "vmax"}``."""
    x, y, z = (np.asarray(a, dtype=float) for a in (x, y, z))
    if not len(x):
        raise LegacyViewError("No tracks found with those characteristics")
    if mirror:
        xy = mirror_points(np.column_stack([x, y]))
        x, y, z = xy[:, 0], xy[:, 1], np.tile(z, 4)
    nx, ny = n_bins
    xb = np.linspace(x.min(), x.max(), nx + 1)
    yb = np.linspace(y.min(), y.max(), ny + 1)
    sx, sy = (x.max() - x.min()) / nx, (y.max() - y.min()) / ny
    in_x = (xb[:, None] <= x[None, :]) & (x[None, :] <= xb[:, None] + sx)          # (nx + 1, n)
    in_y = (yb[:, None] <= y[None, :]) & (y[None, :] <= yb[:, None] + sy)          # (ny + 1, n)
    finite = np.isfinite(z)
    zf = np.where(finite, z, 0.0)
    sums = in_x.astype(float) @ (in_y.astype(float) * zf[None, :]).T
    counts = in_x.astype(float) @ (in_y.astype(float) * finite[None, :]).T
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    drawn = mean[:nx, :ny].T                                                       # pcolor draws nx x ny faces
    return {"x_edges": xb, "y_edges": yb, "mean": drawn, "vmax": float(np.nanmax(drawn)), "kind": "dynamics"}


def location_distribution(x, y, n_bins: int = 20) -> dict:
    """`barplotDistribution`, source 1: the two histograms as shares of the points."""
    return {"x": distribution_hist(x, n_bins), "y": distribution_hist(y, n_bins)}


def dynamics_distribution(x, y, z) -> dict:
    """`barplotDistribution`, dynamics sources: the points' value interpolated linearly onto a 51 x 17 grid over x in
    [0, 1], y in [-0.5, 0.5] (`griddata`, NaN outside the points' hull), then along each axis the median of the other
    and the standard deviation above it."""
    from scipy.interpolate import griddata

    xq, yq = np.meshgrid(np.arange(0.0, 1.0 + 1e-9, 0.02), np.arange(-0.5, 0.5 + 1e-9, 0.06))
    vq = griddata(np.column_stack([x, y]), z, (xq, yq), method="linear")

    def reduce(grid):
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)          # an empty column is NaN, as in MATLAB
            med = np.nanmedian(grid, axis=0)
            sd = np.nanstd(grid, axis=0, ddof=1)
        return med, sd

    mx, sx = reduce(vq)
    my, sy = reduce(vq.T)
    if not (np.isfinite(mx).any() and np.isfinite(my).any()):
        raise LegacyViewError("the interpolated map is empty: too few points to span a surface")
    return {"x_median": mx, "x_std": sx, "y_median": my, "y_std": sy}


def grow_clim(clim, *maxima) -> list:
    """`plotHeatmap.m:65-70`: the colour limit is ``[0, max]`` and only ever raised."""
    top = max([float(clim[1])] + [float(m) for m in maxima if np.isfinite(m)])
    return [0.0, top]


# --------------------------------------------------------------------------------------------- #
# "Individual cells"
# --------------------------------------------------------------------------------------------- #

def individual_cells(project, condition: str) -> list:
    """`showIndividualHeatmap`: for every outlined cell that has tracks, its spot locations in the standard cell,
    ``[{"movie", "cell", "x", "y", "n_tracks"}]`` (spot location only, 2.0's own restriction)."""
    table, _ = _cells(project, condition)
    out = []
    for cell, part in table.groupby("cell", sort=False, observed=True):
        x, y = part["nx"].to_numpy(dtype=float), part["ny"].to_numpy(dtype=float)
        inside = inside_model_cell(x, y)
        if not inside.any():
            continue
        out.append({"movie": str(part["movie"].iloc[0]), "cell": str(cell), "x": x[inside], "y": y[inside],
                    "n_tracks": int(part["uid"].nunique())})
    if not out:
        raise LegacyViewError(f"{condition}: no cell holds a track")
    return out
