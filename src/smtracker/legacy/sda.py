"""Legacy item 3: the SDA maps, on the legacy's own MIDPOINT convention.

READ ``SMT_GUI/Help_fcn/plotHeatmap.m``, ``getDynamics.m``, ``getNormalizeTracks.m``,
``normalizeTracks.m``, ``mirrorPoints.m`` and ``bin3d.m`` beside this. Every statement
below was confirmed in those files before anything here was written.

THE DESIGN, reproduced. Positions are normalised per cell, x by the cell LENGTH and y by
the cell WIDTH, so the drawn cell is a fixed rectangle and cells of every size are pooled
into one picture. The grid spans ``x in [-0.1, 1.1]`` and ``y in [-0.6, 0.6]``, the
overhang being where the pole extrapolation puts molecules projected past the end cap.
Two data sources are reproduced: the LOCATION map (a count per bin) and the DIFFUSION map
(a per-jump apparent D averaged per bin). Folding about both axes is offered because the
legacy offers it.

The step position is the midpoint, and that is the LEGACY'S choice, not an INSTRUCTION'S.
``getDynamics.m:8`` is ``middlePoint = @(track) (diff(track(:,2:3))/2 +
track(1:end-1,2:3))``. Each jump contributes ONE point, at the midpoint of the two
localisations it spans. There is no line rasterisation and no notion of the pixels a jump
crosses. Version 3 assigns a step to its START (``hmm.decode_states``). **The two
conventions read different radial contrasts off the same tracks, for a reason that is a
convention rather than biology**, and that sentence is the item's limitation line.

The size of the gap is measured per group, never imported. This project has a measured
pair, 3.90x against 2.70x, but it belongs to the HMM per-STEP state map, which is a
different quantity on different labels from this map's per-JUMP apparent D. Carrying it
here would move a number between analyses. ``convention_contrast`` runs both conventions
through ``sda_points`` on the group being drawn and fills the limitation line from that.

Five defects, each confirmed in the source.

**3.1 The location map is up-scaled about thirty-twofold per axis and the added detail is
empty.** ``plotHeatmap.m:16-17`` builds a 20-node histogram and then calls
``interp2(v, 5, 'cubic')``. Five refinements take 20 nodes to 609, so every displayed
cell is 1/32 of a data bin and there are 1024 displayed cells per bin. **The short-axis
data bin is already below one camera pixel before any of that**: the grid spans 1.2 cell
widths in 19 intervals, so for a cell about 0.9 um wide the bin is about 57 nm against a
106 nm pixel. ``assert_bin_resolvable`` refuses it and says by how much.

**3.2 Cubic interpolation of counts produces negative densities.** Cubic convolution
overshoots at an edge, a count cannot be negative, and the colour floor is pinned at zero
(``plotHeatmap.m:67``), so the overshoot is clipped and reads as an empty bin.

**3.3 The fold takes the MAXIMUM of four reflections.** ``plotHeatmap.m:20-24`` stacks the
image with its three flips and takes ``max(vqm,[],3)``. The maximum of four noisy
estimates of one quantity is biased upward, and biased most where counts are lowest. A
fold that pools counts is unbiased. ``fold_grid`` offers both and ``fold_inflation``
measures the gap on the data being drawn.

**3.4 The same tab folds two different ways and the figure does not say which.** The
location path maxes four reflected IMAGES (above); the dynamics path replicates the
reflected DATA fourfold (``plotHeatmap.m:84-85`` with ``mirrorPoints.m``) and bins them
together, which is a pooling fold. One word, two operations.

**3.5 The colour limit only ever grows, and it persists between panels.**
``plotHeatmap.m:65-70`` raises ``params.heatmapCaxis`` when a panel exceeds it and never
lowers it, and panels already drawn are not redrawn. The scale a panel is shown on
therefore depends on the order the panels were clicked in that session.

Two things that look like defects and are not. Recorded because a check that was made and
came back clean is worth as much as one that did not, and because the next reader will
otherwise make them again.

* ``bin3d.m`` builds ``Nx+1`` by ``Ny+1`` bins whose last row and column are degenerate,
  holding only the single maximum point; ``pcolor`` then draws ``Nx`` by ``Ny`` faces and
  discards exactly those. The face colour comes from the bin at its lower-left corner,
  which matches the half-open-below, closed-above binning above it. The two behaviours
  agree, so nothing wrong is drawn.
* ``normalizeTracks.m`` does NOT append a padding row per track, so the location map's
  divisor ``size(nTracks,1)`` is a genuine localisation count. This is the same expression
  that IS wrong in ``plotSpeedmap.m:182``, where line 100 pads first. Same code, different
  verdict, because the two callers build their input differently.

One thing that is mislabelled and unused. ``bin3d.m:18`` assigns ``nanstd`` to a variable
named ``sem`` with the division commented out. Nothing in ``plotHeatmap`` reads it, so no
figure is affected, and it is recorded rather than fixed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from ..canonical import without_attrs

__all__ = [
    "SDAError", "LEGACY_N_NODES", "LEGACY_INTERP_REFINEMENTS", "LEGACY_X_SPAN",
    "LEGACY_Y_SPAN", "LEGACY_DYNAMICS_BINS", "MIDPOINT_CONVENTION_NOTE",
    "interp_grid_nodes", "upscale_factor", "bin_size_nm", "assert_bin_resolvable",
    "legacy_bin_report", "midpoint_positions", "jump_diffusion", "sda_points",
    "convention_contrast",
    "location_grid", "value_grid", "fold_grid", "fold_inflation", "sda_provenance",
]


class SDAError(RuntimeError):
    """Raised when a legacy SDA map cannot be produced honestly."""


#: ``plotHeatmap.m:16``, ``linspace(-0.1, 1.1, N)`` with ``N = 20``.
LEGACY_N_NODES = 20

#: ``plotHeatmap.m:17``, ``interp2(v, 5, 'cubic')``.
LEGACY_INTERP_REFINEMENTS = 5

#: The drawn extent, in units of cell length and cell width. The overhang past 0 and 1 is
#: where the pole extrapolation puts molecules projected beyond the end cap.
LEGACY_X_SPAN = (-0.1, 1.1)
LEGACY_Y_SPAN = (-0.6, 0.6)

#: ``plotHeatmap.m:95``, ``bin3d(ax, x, y, v, 30, 10)``.
LEGACY_DYNAMICS_BINS = (30, 10)

#: Printed with every SDA map and carried into the provenance.
#:
#: THE NUMBERS ARE NOT IN THIS STRING, AND THAT IS DELIBERATE. This project has a measured
#: pair, 3.90x against 2.70x, but it was measured on the HMM per-STEP state map, which is
#: a different quantity from this map's per-JUMP apparent D on different labels. Quoting
#: it here would move a number between analyses, which is the drift the limitation lines
#: exist to prevent. ``convention_contrast`` measures the pair on the group being drawn.
MIDPOINT_CONVENTION_NOTE = (
    "the legacy assigns each step to the MIDPOINT of the two localisations it spans "
    "(getDynamics.m:8), where version 3 assigns it to the step's start. The two "
    "conventions read different radial contrasts off the same tracks, because a "
    "long step starting at the membrane can only move inward while its midpoint does "
    "not, so a legacy map reads its contrast for a reason that is a convention rather "
    "than biology. The size of the gap is measured per group by convention_contrast and "
    "is never carried over from another analysis")


def convention_contrast(canonical: pd.DataFrame, *, low_d_threshold: float,
                        n_shells: int = 5, tau_s: float | None = None,
                        l_col: str = "l_norm", d_col: str = "d_norm") -> dict:
    """Radial contrast of the low-D label under BOTH step-position conventions.

    Same tracks, same threshold, same shells; only the position assigned to each jump
    differs. The contrast is the low-D share in the outermost shell over the low-D share
    in the innermost, so 1.0 is no radial preference.

    This is what fills the item's limitation line. It is measured on the group being
    drawn, because the two conventions do not differ by a fixed factor: the gap depends
    on the step-length distribution, which differs between groups and between carbon
    sources.
    """
    if low_d_threshold <= 0:
        raise SDAError("the low-D threshold must be positive and must come from the "
                       "analysis, not from this function's imagination")
    out = {"low_d_threshold_um2_per_s": float(low_d_threshold), "n_shells": int(n_shells)}
    edges = np.linspace(0.0, 1.0, int(n_shells) + 1)
    for name in ("midpoint", "start"):
        pts = sda_points(canonical, l_col=l_col, d_col=d_col, tau_s=tau_s,
                         step_position=name)
        radius = np.abs(pts["d_mid"].to_numpy(dtype=float))
        low = (pts["appd_um2_per_s"].to_numpy(dtype=float) < low_d_threshold)
        shell = np.digitize(radius, edges[1:-1])
        share, counts = [], []
        for s_i in range(int(n_shells)):
            m = shell == s_i
            counts.append(int(m.sum()))
            share.append(float(low[m].mean()) if m.any() else np.nan)
        out[f"{name}_shell_low_d_share"] = share
        out[f"{name}_shell_n"] = counts
        inner, outer = share[0], share[-1]
        out[f"{name}_contrast"] = (float(outer / inner)
                                   if inner and np.isfinite(inner) and np.isfinite(outer)
                                   else float("nan"))
        out[f"{name}_n_jumps"] = int(len(pts))
    if out["midpoint_n_jumps"] != out["start_n_jumps"]:
        raise SDAError("the two conventions produced different jump counts; they must "
                       "describe the same jumps or this is not a comparison between "
                       "conventions")
    out["convention_note"] = MIDPOINT_CONVENTION_NOTE
    return out


# --------------------------------------------------------------- the up-scale guard
def interp_grid_nodes(n_nodes: int = LEGACY_N_NODES,
                      refinements: int = LEGACY_INTERP_REFINEMENTS) -> int:
    """Nodes after ``interp2(v, k)``: each pass inserts one node between every pair."""
    if n_nodes < 2:
        raise SDAError("a grid needs at least two nodes")
    if refinements < 0:
        raise SDAError("refinements cannot be negative")
    return (n_nodes - 1) * (2 ** refinements) + 1


def upscale_factor(n_nodes: int = LEGACY_N_NODES,
                   refinements: int = LEGACY_INTERP_REFINEMENTS) -> int:
    """How many displayed cells span one data bin, per axis. Computed, never written."""
    return (interp_grid_nodes(n_nodes, refinements) - 1) // (n_nodes - 1)


def bin_size_nm(cell_extent_um: float, span: tuple = LEGACY_X_SPAN,
                n_nodes: int = LEGACY_N_NODES) -> float:
    """The data bin size in nanometres, for a cell of the given extent along that axis.

    The grid is in normalised units, so a bin's physical size depends on the cell. Nodes
    are edges, so ``n_nodes`` of them bound ``n_nodes - 1`` intervals.
    """
    if cell_extent_um <= 0:
        raise SDAError("a cell extent must be positive")
    return (span[1] - span[0]) / (n_nodes - 1) * cell_extent_um * 1000.0


def assert_bin_resolvable(bin_nm: float, pixel_nm: float, *, what: str) -> float:
    """Refuse a bin smaller than one camera pixel. Returns the bin size in pixels.

    A bin finer than the detector samples cannot hold information the detector did not
    record, so the extra bins are interpolation presented at the same weight as data.
    This is the guard the legacy up-scaling has no equivalent of.
    """
    if pixel_nm <= 0:
        raise SDAError("the camera pixel size must be positive")
    in_pixels = float(bin_nm) / float(pixel_nm)
    if in_pixels < 1.0:
        raise SDAError(
            f"{what}: the bin is {bin_nm:.1f} nm, which is {in_pixels:.2f} of one "
            f"{pixel_nm:.1f} nm camera pixel. A bin finer than the detector samples "
            f"cannot carry information the detector did not record, so every feature "
            f"below the pixel is interpolation drawn at the same weight as data. Use a "
            f"coarser grid, or pass a larger cell extent if this one is wrong.")
    return in_pixels


def legacy_bin_report(cell_length_um: float, cell_width_um: float,
                      pixel_nm: float) -> dict:
    """What the legacy location grid would resolve, per axis, before and after interp2.

    Both counts are reported and never the difference, and the verdict per axis is the
    guard's own answer rather than a restatement of it.
    """
    factor = upscale_factor()
    out = {"pixel_nm": float(pixel_nm),
           "n_nodes": LEGACY_N_NODES,
           "interp_refinements": LEGACY_INTERP_REFINEMENTS,
           "interp_nodes": interp_grid_nodes(),
           "upscale_factor_per_axis": factor,
           "displayed_cells_per_data_bin": factor * factor,
           "cell_length_um": float(cell_length_um),
           "cell_width_um": float(cell_width_um)}
    for axis, extent, span in (("long", cell_length_um, LEGACY_X_SPAN),
                               ("short", cell_width_um, LEGACY_Y_SPAN)):
        data_nm = bin_size_nm(extent, span)
        out[f"{axis}_axis_data_bin_nm"] = data_nm
        out[f"{axis}_axis_data_bin_px"] = data_nm / pixel_nm
        out[f"{axis}_axis_displayed_bin_nm"] = data_nm / factor
        out[f"{axis}_axis_displayed_bin_px"] = data_nm / factor / pixel_nm
        try:
            assert_bin_resolvable(data_nm, pixel_nm, what=f"{axis} axis, before interp2")
            out[f"{axis}_axis_data_bin_resolvable"] = True
        except SDAError:
            out[f"{axis}_axis_data_bin_resolvable"] = False
        out[f"{axis}_axis_displayed_bin_resolvable"] = (data_nm / factor) >= pixel_nm
    return out


# ------------------------------------------------------------ the midpoint convention
def midpoint_positions(xy: np.ndarray) -> np.ndarray:
    """``getDynamics.m:8``: one point per jump, at the midpoint of the two it spans.

    ``diff(track)/2 + track(1:end-1)`` is the midpoint written the legacy's way. A track
    of n localisations yields n-1 points, one fewer than the track has positions.
    """
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise SDAError("expected an (n, 2) array of positions")
    if len(xy) < 2:
        return np.empty((0, 2), dtype=float)
    return np.diff(xy, axis=0) / 2.0 + xy[:-1]


def jump_diffusion(xy_um: np.ndarray, tau_s: float, lag: int = 1) -> np.ndarray:
    """``getDynamics.m:11``: the per-jump apparent D, squared displacement over 4 tau.

    Two dimensions, no offset removed and no localisation term, so this is an APPARENT D
    exactly as the legacy computes it.
    """
    xy_um = np.asarray(xy_um, dtype=float)
    if tau_s <= 0:
        raise SDAError("tau must be positive")
    if lag < 1:
        raise SDAError("the lag must be at least one frame")
    if len(xy_um) <= lag:
        return np.empty(0, dtype=float)
    step = xy_um[lag:] - xy_um[:-lag]
    return (step ** 2).sum(axis=1) / (4.0 * tau_s)


def sda_points(canonical: pd.DataFrame, *, l_col: str = "l_norm",
               d_col: str = "d_norm", tau_s: float | None = None,
               lag: int = 1, step_position: str = "midpoint") -> pd.DataFrame:
    """One row per JUMP: its position in normalised cell coordinates and its apparent D.

    ``step_position`` is ``"midpoint"`` (the legacy, ``getDynamics.m:8``) or ``"start"``
    (version 3, ``hmm.decode_states``). Both conventions run through this one function.
    An earlier version built the start positions in a second loop and attached them to
    the midpoint jumps by position; the two loops iterated the groups in different
    orders (sorted against first-appearance, so ``t0, t1, t10, t100`` against
    ``t0, t1, t2, t3``) and every value landed on the wrong jump. A comparison between
    conventions is worthless if the two sides do not describe the same jumps, so there
    is one loop and the convention is a branch inside it.

    The positions are taken from the projected columns, so the normalisation by cell
    length and width has already happened upstream. The value is computed from the
    micrometre coordinates, because an apparent D from normalised coordinates would be in
    units of squared cell lengths.
    """
    needed = {l_col, d_col, "uid", "x_um", "y_um"}
    missing = needed - set(canonical.columns)
    if missing:
        raise SDAError(f"missing columns {sorted(missing)}")
    if step_position not in ("midpoint", "start"):
        raise SDAError(
            f"unknown step position {step_position!r}: expected 'midpoint' (the legacy) "
            f"or 'start' (version 3)")
    if tau_s is None:
        if "tau_s" not in canonical.columns:
            raise SDAError(
                "tau is not in the table and was not supplied. The acquisition interval "
                "is read from movieData.mat, never guessed.")
        tau_s = float(canonical["tau_s"].iloc[0])

    rows = []
    for uid, part in without_attrs(canonical).groupby("uid", observed=True, sort=False):
        part = part.sort_values("frame") if "frame" in part.columns else part
        norm = part[[l_col, d_col]].to_numpy(dtype=float)
        um = part[["x_um", "y_um"]].to_numpy(dtype=float)
        if len(norm) <= lag:
            continue
        if step_position == "start":
            mid = norm[:-lag]
        elif lag == 1:
            mid = midpoint_positions(norm)
        else:
            mid = (norm[lag:] + norm[:-lag]) / 2.0
        value = jump_diffusion(um, tau_s, lag=lag)
        if len(mid) != len(value):
            raise SDAError(
                f"{uid}: {len(mid)} midpoints against {len(value)} values. The position "
                f"and the value must come from the same jump or the map is misaligned.")
        frame = (part["frame"].to_numpy()[:-lag] if "frame" in part.columns
                 else np.arange(len(value)))
        rows.append(pd.DataFrame({"uid": uid, "frame": frame, "l_mid": mid[:, 0],
                                  "d_mid": mid[:, 1], "appd_um2_per_s": value}))
    if not rows:
        raise SDAError(f"no track has more than {lag} localisation(s); no jump exists")
    out = pd.concat(rows, ignore_index=True)
    out.attrs["convention"] = MIDPOINT_CONVENTION_NOTE
    out.attrs["step_position"] = step_position
    out.attrs["lag_frames"] = lag
    out.attrs["tau_s"] = float(tau_s)
    return out


# ------------------------------------------------------------------------- the grids
def _edges(span, n_bins):
    return np.linspace(span[0], span[1], int(n_bins) + 1)


def location_grid(points: pd.DataFrame, *, n_bins=(30, 10), l_col="l_mid",
                  d_col="d_mid") -> dict:
    """The location map: a count per bin, and nothing interpolated.

    The legacy divides by the localisation count and then up-scales, which makes the
    displayed surface neither a count nor a probability. Both counts are kept here and the
    density is derived from them, so either can be read.
    """
    if points.empty:
        raise SDAError("no points to bin")
    x = points[l_col].to_numpy(dtype=float)
    y = points[d_col].to_numpy(dtype=float)
    xe, ye = _edges(LEGACY_X_SPAN, n_bins[0]), _edges(LEGACY_Y_SPAN, n_bins[1])
    counts, _, _ = np.histogram2d(x, y, bins=[xe, ye])
    inside = int(counts.sum())
    return {"counts": counts, "x_edges": xe, "y_edges": ye,
            "n_points_in": int(len(x)), "n_points_binned": inside,
            "n_points_outside_grid": int(len(x) - inside),
            "density": counts / inside if inside else counts,
            "n_bins": tuple(int(b) for b in n_bins)}


def value_grid(points: pd.DataFrame, *, value_col="appd_um2_per_s",
               n_bins=LEGACY_DYNAMICS_BINS, min_count: int = 0,
               l_col="l_mid", d_col="d_mid") -> dict:
    """The diffusion map: the mean value per bin, with a reportable minimum count.

    ``min_count`` defaults to 0, which is the legacy behaviour: a bin holding one jump is
    drawn at the same weight as a bin holding five hundred. Raising it masks the sparse
    bins, and the count masked is reported either way so the default is visible rather
    than assumed.
    """
    if points.empty:
        raise SDAError("no points to bin")
    if min_count < 0:
        raise SDAError("min_count cannot be negative")
    x = points[l_col].to_numpy(dtype=float)
    y = points[d_col].to_numpy(dtype=float)
    v = points[value_col].to_numpy(dtype=float)
    xe, ye = _edges(LEGACY_X_SPAN, n_bins[0]), _edges(LEGACY_Y_SPAN, n_bins[1])
    counts, _, _ = np.histogram2d(x, y, bins=[xe, ye])
    totals, _, _ = np.histogram2d(x, y, bins=[xe, ye], weights=v)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)
    occupied = int((counts > 0).sum())
    sparse = int(((counts > 0) & (counts < max(min_count, 1))).sum()) if min_count else 0
    masked = mean.copy()
    if min_count:
        masked[counts < min_count] = np.nan
    return {"mean": masked, "counts": counts, "x_edges": xe, "y_edges": ye,
            "n_bins": tuple(int(b) for b in n_bins), "min_count": int(min_count),
            "n_bins_occupied": occupied,
            "n_bins_drawn": int(np.isfinite(masked).sum()),
            "n_bins_below_min_count": sparse,
            "n_bins_holding_one_point": int((counts == 1).sum()),
            "n_points_binned": int(counts.sum()),
            "sparse_note": (
                "min_count is 0, the legacy behaviour: a bin holding one jump is drawn "
                "at the same weight as the fullest bin" if not min_count else
                f"bins below {min_count} points are masked")}


def fold_grid(grid: np.ndarray, how: str = "sum") -> np.ndarray:
    """Fold about both axes. ``how='max'`` is the legacy location path, ``'sum'`` pools.

    The legacy stacks the image with its three flips and takes the maximum
    (``plotHeatmap.m:20-24``). The maximum of four noisy estimates of one quantity is
    biased upward. Pooling is unbiased, and is what the dynamics path effectively does by
    replicating the data.
    """
    grid = np.asarray(grid, dtype=float)
    stack = np.stack([grid, grid[::-1, :], grid[:, ::-1], grid[::-1, ::-1]])
    if how == "max":
        return np.nanmax(stack, axis=0)
    if how == "sum":
        return np.nansum(stack, axis=0)
    if how == "mean":
        return np.nanmean(stack, axis=0)
    raise SDAError(f"unknown fold {how!r}; expected 'sum', 'mean' or 'max'")


def fold_inflation(grid: np.ndarray) -> dict:
    """How much the legacy maximum fold raises the map above a pooling fold.

    Measured on the grid being drawn, never asserted. The comparison is against the MEAN
    fold, since the maximum and the mean are both per-bin summaries of the same four
    reflected values while the sum is four times larger by construction.

    The sparse and full halves are split on the folded mean, not on the unfolded bin.
    The four bins of a fold orbit share one maximum and one mean, so their ratio is one
    number; classifying it by whichever member happened to be read is arbitrary, and on a
    grid with a gradient it measures the gradient rather than the counts. The folded mean
    is the orbit's own occupancy and is the same for every member.
    """
    grid = np.asarray(grid, dtype=float)
    by_max = fold_grid(grid, "max")
    by_mean = fold_grid(grid, "mean")
    occupied = by_mean > 0
    if not occupied.any():
        raise SDAError("the grid is empty; there is nothing to compare")
    ratio = by_max[occupied] / by_mean[occupied]
    occupancy = by_mean[occupied]
    sparse = occupancy <= np.median(occupancy)
    return {
        "median_ratio_max_over_mean": float(np.median(ratio)),
        "max_ratio_max_over_mean": float(np.max(ratio)),
        "n_bins_compared": int(occupied.sum()),
        "n_bins_raised_by_the_max_fold": int((ratio > 1.0).sum()),
        "median_ratio_in_sparser_half": (
            float(np.median(ratio[sparse])) if sparse.any() else float("nan")),
        "median_ratio_in_fuller_half": (
            float(np.median(ratio[~sparse])) if (~sparse).any() else float("nan")),
        "direction": (
            "the maximum of four reflections is biased upward, and biased most where "
            "counts are lowest, so the legacy fold inflates the sparse bins hardest"),
    }


def sda_provenance(*, enzyme: str, condition: str, points: pd.DataFrame,
                   grid: dict, bins: dict, limitation: str) -> dict:
    """Everything a reader needs to know what the map is, in one record.

    N tracks, N localisations and the bin size in nanometres are all here and are all
    printed on the figure, because the legacy printed one number that was none of them.

    The localisation count is DERIVED from the jump count, which is only valid at lag 1
    and on contiguous tracks. u-track runs here with ``timeWindow = 1``, so tracks are
    contiguous and gap loss is exactly zero in every group; at any other lag the count is
    reported as unavailable rather than guessed.
    """
    lag = int(points.attrs.get("lag_frames", 1))
    n_tracks = int(points["uid"].nunique())
    return {
        "enzyme": enzyme,
        "condition": condition,
        "n_tracks": n_tracks,
        "n_jumps": int(len(points)),
        "n_localisations": (int(len(points) + n_tracks) if lag == 1 else None),
        "n_localisations_note": (
            "jumps plus tracks, valid because tracks are contiguous at lag 1"
            if lag == 1 else
            f"not derivable at lag {lag}; count it from the canonical table"),
        "lag_frames": lag,
        "tau_s": float(points.attrs.get("tau_s", float("nan"))),
        "step_position": "midpoint",
        "convention_note": MIDPOINT_CONVENTION_NOTE,
        "n_bins_long_axis": grid["n_bins"][0],
        "n_bins_short_axis": grid["n_bins"][1],
        "long_axis_bin_nm": bins.get("long_axis_data_bin_nm"),
        "short_axis_bin_nm": bins.get("short_axis_data_bin_nm"),
        "interpolation": "none",
        "legacy_interpolation": (
            f"interp2 with {LEGACY_INTERP_REFINEMENTS} refinements, "
            f"{upscale_factor()} displayed cells per data bin per axis"),
        "limitation": limitation,
    }
