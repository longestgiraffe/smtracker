"""Spatial SDA: where molecules are, and how fast they move there -- v2's Spatial
Distribution Analysis maps, built as v3's own method on the Spatial Map's capsule.

Asked for (13 Oct 2026): "please add the SDA panel now with localisation, speed, diffusion
constant, acceleration, with same color map like v2."

WHAT v2 DREW, read in ``SMT_GUI/Help_fcn/getDynamics.m``, ``plotHeatmap.m``,
``mirrorPoints.m`` and ``bin3d.m`` before this was written (the legacy reproduction, with
its five recorded defects, is `smtracker.legacy.sda`; this module is NOT that port):

- Location: ``hist3`` of the localisations divided by their number (a probability per
  bin), up-scaled by ``interp2(v, 5, 'cubic')`` and, mirrored, the MAXIMUM of four flips.
- Velocity, acceleration, diffusion: one value per jump, placed at the jump's MIDPOINT
  (``getDynamics.m:8``), averaged per bin on a 30 x 10 grid (``bin3d``), mirrored by
  replicating the data fourfold (a pooling fold). Colour: ``flip(hot(1024))``.

WHAT IS DRAWN HERE, four maps (`SDA_KINDS`), each on the section's own capsule and bins:

- ``location`` -- every localisation, binned; the density pipeline the other methods use
  turns it into a probability per bin (localisations in the bin over the condition's).
- ``speed`` -- per STEP, the Euclidean step length over tau, ``|dr| / tau`` (um/s), the
  mean per bin.
- ``diffusion`` -- per step, ``|dr|^2 / (4 tau)`` (um^2/s), the mean per bin: the
  per-jump apparent D, computed by `legacy.sda.jump_diffusion` (the definitions match).
- ``acceleration`` -- per PAIR of consecutive steps of one track, ``|dr_(k+1) - dr_k| /
  tau^2`` (um/s^2), the mean per bin, placed at the localisation the two steps share.

A STEP is the pair `observables.squared_displacements` makes: two localisations of one
track exactly one frame apart (its ``pairing="frame"``); a frame gap breaks a track's
steps and is never bridged. Tested row for row against that function, so the speed and D
maps describe exactly the steps the SQD fit sees. An acceleration needs two such steps
in a row, three consecutive frames.

THE DIVERGENCES FROM v2, each a defect or a convention, recorded in
``docs/v2_to_v3_map.md`` section 8 and ``docs/claude/spatial_maps.md``:

1. SPEED. v2's ``abs(sum(dr, 2)) * px / tau`` is ``|dx + dy| / tau``: the SUM of the two
   components, not the step length, so a diagonal step against the grain reads zero.
   Here ``|dr| / tau``.
2. ACCELERATION. v2's is the same ``|dx + dy|`` divided by tau squared: its acceleration
   map is its speed map rescaled by ``1 / tau`` and never looks at a CHANGE of velocity.
   Here the change of displacement between consecutive steps.
3. STEP POSITION. v2 places a jump at its midpoint; here the section's own "Jump drawn
   at" setting places speed and D (start by default, the SQD and HMM maps' default), and
   the caption names it. Acceleration sits at the shared localisation whatever that says.
4. LOCATION. No ``interp2`` upscaling (defects 3.1 and 3.2) and no maximum fold (3.3):
   the location map is the section's density map, folded by pooling.
5. COLOUR LIMITS. Decided over the whole grid before drawing, never v2's ratchet (3.5).
6. A MEAN OF A HANDFUL IS NOISE. v2's ``bin3d`` averages whatever falls in a bin, one
   jump included. Here a value bin needs `VALUE_N_MIN` steps (`render`'s value-map
   minimum, one source of truth) or the Run's own minimum if higher, and is grey below.
7. FOLD. Sums AND counts are pooled over a bin's four reflections and then divided:
   never four means averaged, never a maximum. v2's dynamics fold replicated the data,
   which also pools, so the mean is the same; the location fold is where v2 differs.
8. COARSER BINS FOR THE MEANS (13 Oct 2026; asked, after "smoothing only smoothes the
   first map": "what else can we do" -- "lets do this"). A mean per bin needs far more
   steps than a count does, so at the localisation map's bin size the three value maps
   were mostly grey or blotchy. They are binned `VALUE_BIN_FACTOR` x `VALUE_BIN_FACTOR`
   times coarser -- on every other edge of the same grid, so each large bin is exactly
   four small ones -- and each large bin's sum and count are written back onto its small
   bins. The mean is then EXACTLY the large bin's; every grid keeps one shape, so the
   masking, smoothing, colour limits, export and day pages are unchanged; the caption
   gives the value maps' own bin size.
9. ADAPTIVE SMOOTHING FOR THE MEANS (13 Oct 2026, asked: "adaptive smoothing for the
   value maps" -- "go"). v2 never smoothed. With "Adaptive smoothing" on (the Run's
   default), a value bin keeps the fixed kernel h0 (`max(sigma, half a value bin)`) where
   its kernel already collects `VALUE_N_MIN` steps' worth, and only where it does not is
   that bin's kernel widened along `ADAPTIVE_WIDTHS` (h0 x 1, 1.5, 2, 3) to the first
   width that does, never past `ADAPTIVE_CAP`; a bin still short at the cap is grey. A
   STEP'S WORTH is the kernel's EFFECTIVE count, ``(sum k)^2 / sum k^2`` over the steps
   the kernel weighs (the participation ratio the n_eff view uses for tracks, here over
   steps): the number of steps whose plain mean would be as precise as the kernel's
   weighted one. It is a bin's own count when the kernel is a point, so the minimum keeps
   its meaning (a mean of 20 steps), and it grows with the width, so widening can reach
   it. The kernel's plain weighted count ``G_h(count)`` cannot: it is an AVERAGE count
   per bin, which a wider kernel over a thin region does not raise. Off, the value maps
   are exactly divergence 8's. Computed in `spatial_map_view.finish_value_panels`.

EVERY VALUE IS APPARENT. Speed and D are truncated by the tracker's linking radius (a
molecule that moves further between frames is not linked: the detection ceiling, see
``docs/claude/linking_and_limits.md``) and floored by localisation error (a molecule that
does not move still reads ``sigma^2 / tau`` of apparent D, see ``sigma_and_floor.md``).
ACCELERATION IS NOT A FORCE: for a diffusing molecule consecutive steps are independent,
so their difference is about sqrt(2) steps long and its map follows D and the
localisation noise, not anything pushing the molecule. `apparent_limits` computes the
bounds from the acquisition and every caption carries them. No statistic between
conditions is computed anywhere (Rule 0).

INTERFACES: `step_table`, `pair_table` (the per-step and per-pair values),
`value_grids` (sums and counts on the grid, folded by pooling), `maps_from_projected_sda`
(one table in, the four maps' grids out, in `spatialsla.maps_from_labelled_sla`'s shape
plus ``sums``), `spatial_sda_map` (the GUI's entry), `apparent_limits`. The masking, the
smoothing and the colour limits are `spatial_map_view`'s (`finish_sda_panels`), as for
every other method. No Qt anywhere; imports with Qt blocked (`tests/test_headless_spine`).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .canonical import without_attrs
from .legacy.sda import jump_diffusion
from .render import DEFAULT_N_MIN
from .spatialappd import SpatialAppDConfig
from .spatialsqd import (
    MANUSCRIPT_POLE_MARGIN,
    PROFILE_BINS_LONG,
    STEP_POSITIONS,
    _check_weighting,
    axis_profiles,
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

__all__ = ["SDA_KINDS", "VALUE_KINDS", "SDA_TITLES", "SDA_UNITS", "SDA_COUNTS",
           "VALUE_N_MIN", "VALUE_BIN_FACTOR", "ADAPTIVE_WIDTHS", "ADAPTIVE_CAP",
           "value_bin_factor", "step_table", "pair_table", "value_grids",
           "maps_from_projected_sda", "spatial_sda_map", "apparent_limits",
           "SDA_PROFILE_BINS", "SDA_COUNT_NAMES", "value_profile", "sda_profiles"]

#: The four maps, in the order they are drawn.
SDA_KINDS = ("location", "speed", "diffusion", "acceleration")
#: The maps that are a MEAN per bin (the first is a probability).
VALUE_KINDS = SDA_KINDS[1:]
#: Each map's column title.
SDA_TITLES = {"location": "Localisations", "speed": "Speed",
              "diffusion": "Diffusion (apparent D)", "acceleration": "Acceleration"}
#: Each map's unit, on its colour bar and in the export.
SDA_UNITS = {"location": "probability / bin", "speed": "µm/s", "diffusion": "µm²/s",
             "acceleration": "µm/s²"}
#: What each map counts per bin, the evidence its masking reads.
SDA_COUNTS = {"location": "localisations", "speed": "steps", "diffusion": "steps",
              "acceleration": "step pairs"}
#: The fewest steps (or step pairs) a value bin may be averaged from. A mean of a few
#: per-step values is noise: one step's apparent D is an exponential draw, so its spread
#: is its own size, and a bin of five reads +-45 %. `render.DEFAULT_N_MIN["value"]`, the
#: value-map minimum this project already uses, so it is stated once.
VALUE_N_MIN = int(DEFAULT_N_MIN["value"])


#: The value maps' bins, per side, in the localisation map's (divergence 8 above): 2 x 2.
VALUE_BIN_FACTOR = 2

#: Adaptive smoothing (divergence 9): the kernel widths a value bin may take, in units of
#: h0 = ``max(sigma, half a value bin)``, tried in order; a bin takes the FIRST whose
#: kernel collects `VALUE_N_MIN` steps' worth. A short ladder rather than a continuous
#: width: every width is one pair of Gaussian filters over the whole grid, computed once.
ADAPTIVE_WIDTHS = (1.0, 1.5, 2.0, 3.0)
#: The widest a value bin's kernel may grow, in h0. WHY THREE: at the usual h0 (half a
#: value bin) a kernel of 3 h0 has a standard deviation of one and a half value bins, so
#: about 93 % of its weight already lies OUTSIDE the bin it is drawn in -- the value is
#: its neighbourhood's (about the 3 x 3 value bins around it), not its own. At 100 nm bins
#: that is h = 300 nm on a cell about 1 um wide: wider, and a bin at one wall averages the
#: other wall, a pole bin the cell body. A bin that needs more is grey, never painted
#: with the cell's average. A decision to confirm, not a measurement.
ADAPTIVE_CAP = ADAPTIVE_WIDTHS[-1]


def value_bin_factor(n_bins_long: int, n_bins_short: int,
                     factor: int = VALUE_BIN_FACTOR) -> int:
    """The factor a grid of ``n_bins_long`` x ``n_bins_short`` can take: ``factor`` where it
    divides both (the GUI's grids are even, so 2 always does), else 1 -- a large bin is
    only ever a whole number of small ones."""
    factor = int(factor)
    if factor < 1:
        raise ValueError(f"the value maps' bin factor must be at least 1, got {factor}")
    return factor if n_bins_long % factor == 0 and n_bins_short % factor == 0 else 1


def _spread(grid: np.ndarray, factor: int) -> np.ndarray:
    """A coarse grid written back onto the fine one: each large bin onto its small bins."""
    if factor == 1:
        return grid
    return np.repeat(np.repeat(grid, factor, axis=0), factor, axis=1)


def _sorted_rows(projected: pd.DataFrame) -> pd.DataFrame:
    need = {"uid", "frame", "x_um", "y_um"}
    missing = need - set(projected.columns)
    if missing:
        raise ValueError(f"the SDA maps need the columns {sorted(missing)}")
    if len(projected) == 0:
        raise ValueError("no localisations to map")
    # ONE sort, the same `squared_displacements` makes; never a slice per track.
    return without_attrs(projected).sort_values(["uid", "frame"], kind="stable")


def _carried(rows: pd.DataFrame, index: np.ndarray) -> dict:
    """The columns a step or pair row inherits from the localisation at `index`."""
    out = {}
    for column in ("condition", "day", "movie", "cell", "l_norm", "d_norm"):
        if column in rows.columns:
            out[column] = rows[column].to_numpy()[index]
    return out


def step_table(projected: pd.DataFrame, tau_s: float) -> pd.DataFrame:
    """One row per STEP: two localisations of one track exactly one frame apart.

    Columns: ``uid``, ``frame`` (the step's first frame, as `squared_displacements`
    gives it), ``dx_um``, ``dy_um``, ``step_um`` (``|dr|``), ``speed_um_s``
    (``|dr| / tau``), ``diffusion_um2_s`` (``|dr|^2 / (4 tau)``, `legacy.sda.
    jump_diffusion`), the START localisation's ``l_norm``/``d_norm`` and, where the table
    has them, ``condition``/``day``/``movie``/``cell``. A frame gap breaks the track:
    no step spans it."""
    tau_s = float(tau_s)
    if not tau_s > 0:
        raise ValueError(f"tau must be positive, got {tau_s}")
    rows = _sorted_rows(projected)
    uid = rows["uid"].to_numpy()
    frame = rows["frame"].to_numpy()
    xy = rows[["x_um", "y_um"]].to_numpy(float)
    same = (uid[1:] == uid[:-1]) & (np.diff(frame.astype(float)) == 1)
    first = np.flatnonzero(same)
    d = xy[first + 1] - xy[first]
    step = np.hypot(d[:, 0], d[:, 1])
    out = pd.DataFrame({"uid": uid[first], "frame": frame[first],
                        "dx_um": d[:, 0], "dy_um": d[:, 1], "step_um": step,
                        "speed_um_s": step / tau_s,
                        "diffusion_um2_s": jump_diffusion(xy, tau_s)[first],
                        **_carried(rows, first)})
    return out.reset_index(drop=True)


def pair_table(projected: pd.DataFrame, tau_s: float) -> pd.DataFrame:
    """One row per PAIR of consecutive steps of one track (three consecutive frames):
    ``acceleration_um_s2`` = ``|dr_(k+1) - dr_k| / tau^2``, placed at the localisation the
    two steps share, whose ``uid``, ``frame``, ``l_norm``/``d_norm`` and grouping columns
    the row carries. Not a force: see the module head."""
    tau_s = float(tau_s)
    if not tau_s > 0:
        raise ValueError(f"tau must be positive, got {tau_s}")
    rows = _sorted_rows(projected)
    uid = rows["uid"].to_numpy()
    frame = rows["frame"].to_numpy()
    xy = rows[["x_um", "y_um"]].to_numpy(float)
    same = (uid[1:] == uid[:-1]) & (np.diff(frame.astype(float)) == 1)
    first = np.flatnonzero(same[:-1] & same[1:])      # steps first->first+1->first+2
    middle = first + 1
    change = (xy[first + 2] - xy[middle]) - (xy[middle] - xy[first])
    out = pd.DataFrame({"uid": uid[middle], "frame": frame[middle],
                        "acceleration_um_s2": np.hypot(change[:, 0], change[:, 1])
                        / tau_s ** 2,
                        **_carried(rows, middle)})
    return out.reset_index(drop=True)


def value_grids(x, y, values, weights, xe, ye, *, mirrored: bool) -> tuple:
    """``(sums, counts)`` on the grid (rows = y, columns = x, as the other maps): the
    weighted sum of ``values`` and the summed ``weights`` per bin. FOLDED BY POOLING:
    sums and counts are each summed over a bin's four reflections (`spatialsqd.
    fold_counts`), so the folded mean is (sum over the orbit) / (count over the orbit) --
    never the mean of four means, never their maximum."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(weights, dtype=float)
    v = np.asarray(values, dtype=float)
    if not (len(x) == len(y) == len(w) == len(v)):
        raise ValueError("positions, values and weights must be the same length")
    if not np.isfinite(v).all():
        raise ValueError("a value to average is not finite")
    sums = np.histogram2d(x, y, bins=[xe, ye], weights=w * v)[0].T
    counts = np.histogram2d(x, y, bins=[xe, ye], weights=w)[0].T
    if mirrored:
        sums, counts = fold_counts(sums), fold_counts(counts)
    return sums, counts


def _capsule_xy(table: pd.DataFrame, l_norm, d_norm, cfg, projected) -> tuple:
    """Every row's position on the drawn capsule (um) and which rows are inside it."""
    l_norm = np.asarray(l_norm, dtype=float)
    d_norm = np.asarray(d_norm, dtype=float)
    keep = np.isfinite(d_norm) & (np.abs(d_norm) <= 1.0) & np.isfinite(l_norm)
    cells = (row_cell_keys(table, projected)[keep]
             if getattr(cfg, "long_axis", "normalised") != "normalised" else None)
    x = long_axis_x(l_norm[keep], cfg, cells=cells, geometry=projected.attrs.get("geometry"))
    y = d_norm[keep] * local_half_width(x, cfg)
    return x, y, keep


#: THE x AND y GRAPHS BESIDE THE MAPS (23 Nov 2026, asked: "u rememver in smtracker v2 in sla or sda ... there were
#: like x and y probablity graphs for the speed map, d coeff map, loclisation and acceleration maps. we should put them
#: in too"). v2 (`SMTracker.m`, ``barplotDistribution``) drew, beside each SDA map, a histogram of the normalised long-
#: and short-axis positions (20 bins, "probability") for the localisations, and for speed, acceleration and D the median
#: of a ``griddata`` interpolation of the values per grid column, an SD over the interpolated grid as its error bar.
#: Here the profiles of every other method (`spatialsqd.axis_profiles`: along = distance from the nearer pole, across =
#: the signed relative position in the cylinder), the localisations as a density (area 1, beside the even reference)
#: and each value map as the MEAN of the measured values per bin, never interpolated; a bin with fewer than
#: `VALUE_N_MIN` rows has no value. Across bins: 20, as v2's.
SDA_PROFILE_BINS = 20
#: What a profile's rows are, by map (the table's count column).
SDA_COUNT_NAMES = {"location": "Localisations", "speed": "Steps", "diffusion": "Steps",
                   "acceleration": "Step pairs"}


def value_profile(x, y, values, weights, cfg, *, n_short: int = SDA_PROFILE_BINS, n_min: int = VALUE_N_MIN,
                  unit: str = "") -> dict:
    """One value map's profile along and across the cell: the (weighted) mean value per bin, None where fewer than
    ``n_min`` rows fall, with the rows per bin; along = distance from the nearer pole (0 to half the length, as
    `axis_profiles`), across = the signed relative short-axis position over [-1, 1] in the cylinder only (the caps,
    `MANUSCRIPT_POLE_MARGIN` of the length at each end, left out). Plus the mean over every row, in the caps and in the
    cylinder (the part of the length between the caps)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(v) & np.isfinite(w)
    x, y, v, w = x[ok], y[ok], v[ok], w[ok]
    length, cap = float(cfg.cell_length_um), float(cfg.cell_width_um) / 2.0
    half = length / 2.0
    s = np.clip(np.minimum(x, length - x), 0.0, half)
    hw = local_half_width(x, cfg)
    d_norm = np.divide(y, hw, out=np.zeros_like(y), where=hw > 0)
    l_norm = x / length if length > 0 else np.zeros_like(x)

    def binned(pos, edges, keep):
        sums = np.histogram(pos[keep], bins=edges, weights=(w * v)[keep])[0]
        drawn = np.histogram(pos[keep], bins=edges, weights=w[keep])[0]
        rows = np.histogram(pos[keep], bins=edges)[0]
        means = [float(a / b) if n >= n_min and b > 0 else None for a, b, n in zip(sums, drawn, rows)]
        return means, rows.astype(float).tolist()

    every = np.ones(len(x), dtype=bool)
    long_vals, long_rows = binned(s, np.linspace(0.0, half, PROFILE_BINS_LONG + 1), every)
    cylinder = (l_norm > MANUSCRIPT_POLE_MARGIN) & (l_norm < 1.0 - MANUSCRIPT_POLE_MARGIN) & (np.abs(d_norm) <= 1.0)
    short_vals, short_rows = binned(d_norm, np.linspace(-1.0, 1.0, int(n_short) + 1), cylinder)

    def mean(keep):
        total = float(w[keep].sum())
        return float(np.sum((w * v)[keep]) / total) if total > 0 and int(keep.sum()) >= n_min else None

    in_caps = s <= cap
    return {"weight": float(w.sum()), "rows": int(len(x)), "unit": str(unit),
            "long": long_vals, "long_counts": long_rows, "short": short_vals, "short_counts": short_rows,
            "mean": mean(every), "mean_caps": mean(in_caps), "mean_cylinder": mean(~in_caps)}


def sda_profiles(rows_by_kind: dict, cfg) -> dict:
    """Every SDA map's profile along and across the cell, in `axis_profiles`' shape (one entry per map under
    ``"populations"``): ``rows_by_kind`` maps a kind to ``(x, y, values, weights)`` on the drawn capsule, ``values``
    None for the localisations (a density, `axis_profiles`) and the per-row value for the others (`value_profile`)."""
    out = axis_profiles(np.empty(0), np.empty(0), {}, cfg, n_short=SDA_PROFILE_BINS)
    pops = {}
    for kind, (x, y, values, weights) in rows_by_kind.items():
        if values is None:
            pops[kind] = axis_profiles(x, y, {kind: weights}, cfg, n_short=SDA_PROFILE_BINS)["populations"][kind]
        else:
            pops[kind] = value_profile(x, y, values, weights, cfg, unit=SDA_UNITS[kind])
    out["populations"] = pops
    out["count_names"] = dict(SDA_COUNT_NAMES)
    out["sda"] = True
    return out


def maps_from_projected_sda(projected: pd.DataFrame, cfg, *, tau_s: float,
                            step_position: str = "A_start", mirrored: bool = False,
                            weighting: str = "steps", meshes=None,
                            with_steps: bool = False,
                            value_factor: int = VALUE_BIN_FACTOR) -> dict:
    """The four maps' grids for ONE projected table, in `spatialsla.maps_from_labelled_sla`'s
    shape (``grids``/``weights`` drawn, ``step_grids``/``step_weights`` the plain counts the
    masking reads, ``neff_track``/``neff_cell``) plus ``sums``: each value map's drawn sum
    of values per bin, so `spatial_map_view.finish_sda_panels` divides pooled sums by pooled
    counts. ``n_rows`` is how many localisations, steps and step pairs each map binned.

    ``step_position`` places speed and D: ``"A_start"`` at the step's first localisation,
    ``"B_midpoint"``/``"C_end"`` re-projected through the cell's centreline exactly as the
    SQD map does (`spatialsqd._step_coordinates`, which needs ``meshes``). Acceleration
    always sits at the shared localisation. ``weighting="track"`` gives every track one
    vote per map (each row weighs 1 / its track's rows in that map), in the sum and the
    count alike; the masking still reads the plain counts.

    ``with_steps=True`` adds ``"steps"`` (`spatialsqd.step_block`): every binned row of all
    four maps, each map's drawn weight on its own rows and zero on the others', so a
    clicked spot on a day's map can say which tracks it is made of.

    The value maps are binned ``value_factor`` times coarser per side (`value_bin_factor`,
    divergence 8): their sums, counts and n_eff grids are made on every
    ``value_factor``-th edge and written back onto the fine bins, so every grid keeps the
    location map's shape. ``"value_bin_factor"`` says what was used. The step record keeps
    each row's SMALL bin; a click on a value map is widened to its large bin by
    `spatial_map_view.spot_contributors`, which reads the factor from the payload."""
    _check_weighting(weighting)
    if step_position not in STEP_POSITIONS:
        raise ValueError(f"step_position must be one of {STEP_POSITIONS}, "
                         f"got {step_position!r}")
    locs = without_attrs(projected).reset_index(drop=True)
    steps = step_table(projected, tau_s)
    pairs = pair_table(projected, tau_s)
    if steps.empty:
        raise ValueError("no track has two localisations one frame apart: there is no "
                         "step to measure a speed from")
    if step_position == "A_start":
        step_l = steps["l_norm"].to_numpy(float)
        step_d = steps["d_norm"].to_numpy(float)
    else:
        if meshes is None:
            raise ValueError(
                f"drawing a step at its {step_position.split('_', 1)[1]} re-projects that "
                f"point through the cell's centreline, which needs the cell outlines; "
                f"draw it at its start, which the projected table already carries")
        coords = _step_coordinates(projected, steps, meshes, 1)[step_position]
        step_l = coords["l_norm"].to_numpy(float)
        step_d = coords["d_norm"].to_numpy(float)

    sources = {
        "location": (locs, locs["l_norm"].to_numpy(float), locs["d_norm"].to_numpy(float),
                     None),
        "speed": (steps, step_l, step_d, steps["speed_um_s"].to_numpy(float)),
        "diffusion": (steps, step_l, step_d, steps["diffusion_um2_s"].to_numpy(float)),
        "acceleration": (pairs, pairs["l_norm"].to_numpy(float) if len(pairs) else
                         np.empty(0), pairs["d_norm"].to_numpy(float) if len(pairs) else
                         np.empty(0), pairs["acceleration_um_s2"].to_numpy(float)),
    }
    xe, ye = cfg.edges()
    factor = value_bin_factor(cfg.n_bins_long, cfg.n_bins_short, value_factor)
    coarse_edges = (xe[::factor], ye[::factor])
    grids, weights, step_grids, step_weights, sums = {}, {}, {}, {}, {}
    neff_track, neff_cell, n_rows = {}, {}, {}
    record_rows, record_x, record_y, record_w = [], [], [], {k: [] for k in SDA_KINDS}
    profile_rows = {}
    for kind in SDA_KINDS:
        table, l_norm, d_norm, values = sources[kind]
        if len(table):
            x, y, keep = _capsule_xy(table, l_norm, d_norm, cfg, projected)
            tracks, cells = contributor_labels(table, projected)
            tracks, cells = tracks[keep], cells[keep]
            plain = np.ones(int(keep.sum()))
            drawn = track_weights(table)[keep] if weighting == "track" else plain
        else:
            x = y = plain = drawn = np.empty(0)
            keep = np.zeros(0, dtype=bool)
            tracks = cells = np.empty(0, dtype=str)
        # a value map on the coarse edges, written back onto the fine bins (`_spread`)
        f = factor if values is not None else 1
        kx, ky = coarse_edges if f > 1 else (xe, ye)
        counts = np.histogram2d(x, y, bins=[kx, ky], weights=plain)[0].T
        if mirrored:
            counts = fold_counts(counts)
        step_grids[kind] = _spread(counts, f)
        step_weights[kind] = float(plain.sum())
        n_rows[kind] = int(len(x))
        if values is None:
            drawn_counts = np.histogram2d(x, y, bins=[xe, ye], weights=drawn)[0].T
            grids[kind] = fold_counts(drawn_counts) if mirrored else drawn_counts
        else:
            coarse_sums, coarse_counts = value_grids(x, y, values[keep], drawn, kx, ky,
                                                     mirrored=mirrored)
            sums[kind], grids[kind] = _spread(coarse_sums, f), _spread(coarse_counts, f)
        weights[kind] = float(drawn.sum())
        n_track, n_cell = contributor_neff_grids(x, y, plain, tracks, cells, kx, ky,
                                                 mirrored=mirrored)
        neff_track[kind], neff_cell[kind] = _spread(n_track, f), _spread(n_cell, f)
        # the x and y graphs beside the map (`sda_profiles`): the same rows, the same weights
        profile_rows[kind] = (x, y, None if values is None else values[keep], drawn)
        if with_steps and len(x):
            own = table.iloc[np.flatnonzero(keep)]
            record_rows.append(own[[c for c in ("uid", "day", "movie", "cell")
                                    if c in own.columns]])
            record_x.append(x)
            record_y.append(y)
            for other in SDA_KINDS:
                record_w[other].append(drawn if other == kind else np.zeros(len(x)))

    out = {"grids": grids, "weights": weights, "step_grids": step_grids,
           "step_weights": step_weights, "sums": sums, "neff_track": neff_track,
           "neff_cell": neff_cell, "n_rows": n_rows, "weighting": weighting,
           "mirrored": bool(mirrored), "step_position": step_position,
           "tau_s": float(tau_s), "labels": {k + 1: n for k, n in enumerate(SDA_KINDS)},
           "value_bin_factor": int(factor), "profiles": sda_profiles(profile_rows, cfg)}
    if with_steps and record_rows:
        rows = pd.concat(record_rows, ignore_index=True)
        out["steps"] = step_block(rows, np.concatenate(record_x), np.concatenate(record_y),
                                  {k: np.concatenate(v) for k, v in record_w.items()}, cfg)
    return out


def spatial_sda_map(projected: pd.DataFrame, *, tau_s: float, bin_long_nm: float,
                    bin_short_nm: float, smooth_bins: float, cell_length_um: float,
                    cell_width_um: float, step_position: str = "A_start",
                    mirrored: bool = False, weighting: str = "steps", meshes=None,
                    long_axis: str = "normalised") -> dict:
    """The GUI-facing entry point: the grid from the bin size (even counts, since folding
    pairs each bin with its reflection), then `maps_from_projected_sda`. Returns the maps
    plus ``"config"``. Nothing is fitted, so a day page draws from the same rule by
    construction."""
    def even(value: float) -> int:
        count = round(value)
        return max(2, count + (count % 2))

    cfg = SpatialAppDConfig(
        cell_length_um=cell_length_um, cell_width_um=cell_width_um,
        n_bins_long=even(cell_length_um * 1000.0 / bin_long_nm),
        n_bins_short=even(cell_width_um * 1000.0 / bin_short_nm),
        smooth_bins=smooth_bins, long_axis=long_axis)
    out = maps_from_projected_sda(projected, cfg, tau_s=tau_s, step_position=step_position,
                                  mirrored=mirrored, weighting=weighting, meshes=meshes)
    out["config"] = cfg
    return out


def apparent_limits(*, tau_s: float, sigma_nm=None, search_radius_um=None) -> dict:
    """What the speed, D and acceleration maps CAN read, from the acquisition: the floor a
    molecule that does not move reads from localisation error alone, and the ceiling the
    tracker's linking radius sets. Every number is computed, None where its input is not
    recorded (never assumed).

    A molecule that does not move: each localisation is off by sigma per axis, so a step
    is ``Rayleigh(sigma * sqrt(2))`` long -- mean ``sigma * sqrt(pi)`` -- its apparent D
    has mean ``4 sigma^2 / (4 tau) = sigma^2 / tau`` (`canonical.AcquisitionParams.
    D_static_error_offset`, the comparator for an APPARENT D), and the change between two
    consecutive steps (``e1 - 2 e2 + e3``) has mean length ``sigma * sqrt(3 pi)``. A
    molecule further than the linking radius between two frames is not linked: speed
    cannot exceed ``r / tau`` and D ``r^2 / (4 tau)`` (`detection_ceiling_D`)."""
    tau_s = float(tau_s)
    if not tau_s > 0:
        raise ValueError(f"tau must be positive, got {tau_s}")
    out = {"tau_s": tau_s, "sigma_nm": None if sigma_nm is None else float(sigma_nm),
           "floor_speed_um_s": None, "floor_D_um2_s": None, "floor_acceleration_um_s2": None,
           "search_radius_um": None if search_radius_um is None else float(search_radius_um),
           "ceiling_speed_um_s": None, "ceiling_D_um2_s": None}
    if sigma_nm is not None and float(sigma_nm) > 0:
        s = float(sigma_nm) / 1000.0
        out["floor_speed_um_s"] = s * math.sqrt(math.pi) / tau_s
        out["floor_D_um2_s"] = s ** 2 / tau_s
        out["floor_acceleration_um_s2"] = s * math.sqrt(3.0 * math.pi) / tau_s ** 2
    if search_radius_um is not None and float(search_radius_um) > 0:
        r = float(search_radius_um)
        out["ceiling_speed_um_s"] = r / tau_s
        out["ceiling_D_um2_s"] = r ** 2 / (4.0 * tau_s)
    return out
