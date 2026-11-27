"""Spatial SQD: where the two SQD populations sit inside the cell.

One source of truth for the manuscript's spatial panel and for the GUI tab, so the
figure a reviewer sees and the figure the tool draws cannot diverge (same rule as
`statemaps`). Nothing here is computed in a Qt class and nothing imports Qt.

The estimator is the one already in use. Per-step labels come from `maps.labels_from_sqd`
under a 2-population `fit_sqd`, so this introduces NO new method: "short jump" is a step
the fitted slow population owns, "long jump" one the fast population owns. Per-step, not
per-track, so it is not switching-blurred the way `appd` is.

Two stages, deliberately split (the deposit firewall's shape):

    compute  reads the raw project and writes tidy tables
    draw     reads ONLY those tables and writes the figures

so a colleague with the tables can redraw every panel without the raw data, without
a data drive, and without refitting anything.

The bin is a resolution claim, so it carries no default. `SpatialSQDConfig` requires
every field. The trap it exists to avoid is the one this project already recorded for
the legacy location map (`plotHeatmap.m` interpolating a 20-node grid to 609): a display
finer than the measurement reads as structure that was never resolved. Here the floor is
the localisation precision, not the camera pixel (a fitted centroid beats a pixel) and
σ is unmeasured in this project at 30-76 nm. A 98-100 nm bin therefore sits above every
candidate σ and above 0.9 camera pixels; the smoothing bandwidth is stated in bins so it
can be read against that.

Folding sums counts, and it is the honest default for asymmetry. Which pole is l=0 and
which side is d>0 are arbitrary, so unfolded asymmetry is almost always noise. `fold_counts`
sums each bin's four-element reflection orbit and mirrors the result back over the whole
cell, so a folded panel is 4-fold symmetric BY CONSTRUCTION and must never be read as
evidence of symmetry.

The colour scale is a choice with a cost, named on the figure either way:
  "panel"  each cell scaled to its own maximum. Every panel is readable; densities are
           NOT comparable between panels.
  "shared" every panel a probability density on one scale. Comparable, at the price of a
           protein with a tight hot spot pushing the others dark - which is exactly what
           made the first drafts of this figure unreadable.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "ALL_JUMPS_LABEL",
    "SpatialSQDConfig",
    "capsule_outline",
    "local_half_width",
    "fold_counts",
    "contributor_labels",
    "contributor_neff_grids",
    "step_block",
    "LONG_AXES",
    "long_axis_x",
    "row_cell_keys",
    "upper_envelope",
    "sqd_crossings",
    "compute_condition",
    "write_tables",
    "read_tables",
    "draw_profiles",
    "draw_maps",
    "draw_convention_comparison",
    "draw_condition_matrix",
    "STATE_TITLE",
    "STEP_POSITIONS",
    "DEFAULT_CMAP",
    "maps_from_projected",
    "spatial_sqd_map",
    "population_labels",
    "RANGE_KINDS",
    "check_range_values",
    "maps_from_ranges",
]

#: What the two jump classes are CALLED on a figure.
#:
#: "short"/"long", never "bound"/"free". At n_pop = 2 the SQD posterior is monotonic in
#: r-squared, so the two classes are separated by jump LENGTH; the manuscript layer uses
#: "slow"/"fast" where it is talking about the fitted populations under soft assignment,
#: and this module stays with the more conservative pair that describes what was drawn.
STATE_TITLE = {"short": "short jumps", "long": "long jumps"}


# The reference curve for a population spread evenly through the cell VOLUME. Projected
# to two dimensions it is NOT flat: a chord through a cylinder is longest at the axis.
_UNIFORM_PEAK = 2.0 / np.pi

#: THE DEFAULT IS jet, RULED BY THE EXPERIMENTER'S GROUP. The cost is recorded here so the
#: choice stays an informed one rather than an inherited one, and so nobody re-derives it.
#: Measured on the colormap itself, not argued: jet's L* is NOT monotonic - 108 of its 255
#: STEPS RUN BACKWARDS over 256 entries (the definition matters: counting how many times
#: the ramp TURNS AROUND instead gives 3, and the two statistics rank jet against turbo in
#: OPPOSITE directions - see CLAUDE.md), rising to about 97 mid-ramp and falling back - so
#: two different
#: densities share a brightness and a smooth gradient grows a visible edge where the data
#: has no boundary. Simulated with the Machado (2009) matrices in linear RGB, its high end
#: reads DARKER than its middle under deuteranopia and protanopia, i.e. the map inverts
#: for roughly 6 per cent of male readers.
#:
#: `cubehelix` runs L* 0 -> 100 monotonically and survives every dichromacy. `turbo` does
#: NOT, despite looking like the fix: 129 backward steps, MORE of the ramp than jet's 108,
#: though only individually smaller - by the turning-point count turbo is the better of the
#: two (1 against 3). Neither is monotonic, which is the point. Any of them is one `--cmap` away and `map_scale.csv` is unaffected either way.
#: Continuity with the v2 speed maps is a real reason to keep jet; it is not a perceptual
#: one, and the two should not be confused.
DEFAULT_CMAP = "jet"

SHORT_COLOUR = "#1f5fbf"
LONG_COLOUR = "#c81e1e"
REFERENCE_COLOUR = "0.45"


#: How a cell's long axis is laid onto the drawn capsule (`long_axis_x`).
LONG_AXES = ("normalised", "pole_anchored")


def _check_long_axis(value: str, cfg) -> None:
    if value not in LONG_AXES:
        raise ValueError(f"long_axis must be one of {LONG_AXES}, got {value!r}")
    if value == "pole_anchored" and cfg.cell_length_um <= cfg.cell_width_um:
        raise ValueError(
            f"a pole-anchored long axis needs a drawn capsule longer than it is wide "
            f"(got {cfg.cell_length_um} x {cfg.cell_width_um} um): it has no cylinder "
            f"between its caps to stretch")


@dataclass(frozen=True)
class SpatialSQDConfig:
    """Every field is required but one. See the module head for why each one costs
    something.

    `long_axis` defaults to "normalised" because that is what every table already
    deposited was drawn with, and `read_tables` rebuilds this from a deposited
    `config.json` that has no such key.
    """

    cell_length_um: float
    cell_width_um: float
    n_bins_long: int
    n_bins_short: int
    smooth_bins: float
    pole_margin: float
    n_profile_bins: int
    n_pop: int
    method: str
    weighting: str
    long_axis: str = "normalised"

    def __post_init__(self) -> None:
        _check_long_axis(self.long_axis, self)
        if self.n_bins_long % 2 or self.n_bins_short % 2:
            raise ValueError(
                "n_bins_long and n_bins_short must be even: folding pairs each bin with its reflection, "
                "and an odd count leaves a centre row paired with itself "
                f"(got {self.n_bins_long}x{self.n_bins_short})"
            )
        if not 0.0 <= self.pole_margin < 0.5:
            raise ValueError(f"pole_margin must be in [0, 0.5), got {self.pole_margin}")
        # ONE POPULATION IS A MAP (1 Nov 2026): every jump in one panel, nothing separated -- what a protein of one
        # state has (`spatial_map_view.MAP_N_POPS_BY_METHOD`). It used to be refused ("a map that separates nothing
        # is not a map"), which left a two-population fit of one state as the only way to draw it, and that fit
        # hands every jump to one of two identical populations. The DEPOSIT keeps its own two (`compute_condition`).
        if self.n_pop < 1:
            raise ValueError(f"a jump-density map needs at least one population; got n_pop={self.n_pop}")

    @property
    def bin_long_um(self) -> float:
        return self.cell_length_um / self.n_bins_long

    @property
    def bin_short_um(self) -> float:
        return self.cell_width_um / self.n_bins_short

    def bin_report(self, pixel_size_um: float) -> str:
        return (
            f"bin {self.bin_long_um * 1000:.0f} x {self.bin_short_um * 1000:.0f} nm "
            f"({self.bin_long_um / pixel_size_um:.2f} x "
            f"{self.bin_short_um / pixel_size_um:.2f} camera px), "
            f"smoothing {self.smooth_bins:.2f} bins"
        )

    def edges(self) -> tuple[np.ndarray, np.ndarray]:
        half = self.cell_width_um / 2.0
        return (np.linspace(0.0, self.cell_length_um, self.n_bins_long + 1),
                np.linspace(-half, half, self.n_bins_short + 1))

    def centres(self) -> tuple[np.ndarray, np.ndarray]:
        xe, ye = self.edges()
        return 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])


def row_cell_keys(labelled: pd.DataFrame, projected: pd.DataFrame) -> np.ndarray:
    """The `cell` key of each row of `labelled`, the key `cellcoords.project_tracks`
    files its `.attrs["geometry"]` under. Taken from the row when it carries one (a
    localisation), otherwise through the row's track (a step), since a track lies in
    one cell."""
    if "cell" in labelled.columns:
        return labelled["cell"].astype(str).to_numpy()
    if "uid" not in labelled.columns or "cell" not in projected.columns:
        raise ValueError("a pole-anchored long axis needs each row's cell: give the "
                         "rows a 'cell' column, or a 'uid' the projected table maps")
    first = projected.drop_duplicates("uid").set_index("uid")["cell"].astype(str)
    return labelled["uid"].map(first).astype(str).to_numpy()


def long_axis_x(l_norm, cfg, *, cells=None, geometry=None) -> np.ndarray:
    """Where each position lands along the DRAWN capsule, in um from its left pole.

    `"normalised"` (the old program's way, and the default): `l_norm` times the drawn
    length. Every cell is stretched end to end onto one capsule, so a bin is a FRACTION
    of each cell, and over a range of lengths a fixed-size structure smears: the cap of
    a 2 um cell is a quarter of its length at each end, of a 6 um cell a twelfth.

    `"pole_anchored"`: each cell's two caps land on the drawn caps and only the cylinder
    between them is stretched. A cap is scaled by that cell's own cap radius (half its
    widest rib) onto the drawn one, the same per-cell width scaling `d_norm` already
    applies across the cell, so near a pole a bin is the same distance from the tip in
    every cell; the cylinder is laid on as a fraction of itself, so midcell is midcell.
    A cell no longer than it is wide has no cylinder and is laid on normalised.
    `cells` (row-aligned keys, `row_cell_keys`) and `geometry` (the projected table's
    `.attrs["geometry"]`) are required here, and a row whose cell has no geometry
    RAISES rather than falling back to the other convention unsaid.
    """
    l_norm = np.asarray(l_norm, dtype=float)
    long_axis = getattr(cfg, "long_axis", "normalised")
    if long_axis == "normalised":
        return l_norm * cfg.cell_length_um
    if long_axis != "pole_anchored":
        raise ValueError(f"long_axis must be one of {LONG_AXES}, got {long_axis!r}")
    if cells is None or not geometry:
        raise ValueError("a pole-anchored long axis needs each row's cell and the cells' "
                         "geometry (the projected table's .attrs['geometry'])")
    keys = np.asarray(cells).astype(str)
    if len(keys) != len(l_norm):
        raise ValueError(f"{len(keys)} cell keys for {len(l_norm)} positions")
    if not len(keys):
        return l_norm * cfg.cell_length_um
    names, code = np.unique(keys, return_inverse=True)
    length = np.array([getattr(geometry.get(n), "length_um", np.nan) for n in names],
                      dtype=float)
    cap = np.array([getattr(geometry.get(n), "max_width_um", np.nan) for n in names],
                   dtype=float) / 2.0
    unknown = ~(np.isfinite(length) & np.isfinite(cap) & (cap > 0))
    if unknown.any():
        raise ValueError(f"no usable geometry for cell(s) {list(names[unknown][:5])}; "
                         "the rows and the projection disagree")
    length, cap = length[code], cap[code]
    s = l_norm * length
    model_cap = cfg.cell_width_um / 2.0
    model_cyl = cfg.cell_length_um - 2.0 * model_cap
    cylinder = length - 2.0 * cap
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = model_cap / cap
        x = np.where(s <= cap, s * scale,
                     np.where(s >= length - cap,
                              cfg.cell_length_um - (length - s) * scale,
                              model_cap + (s - cap) / cylinder * model_cyl))
    return np.where(cylinder > 0, x, l_norm * cfg.cell_length_um)


def local_half_width(x: np.ndarray, cfg: SpatialSQDConfig) -> np.ndarray:
    """Half-width of the spherocylinder at long-axis position `x`, in µm.

    `d_norm` is already scaled by each cell's OWN local half-width, so mapping
    `d_norm * local_half_width(x)` onto one median capsule places polar molecules inside
    the caps and truncates nothing. Pooling in absolute µm instead was wrong and it
    showed: cells run 2.2-4.6 µm, so the long ones were cut and the right edge became a
    mixture of "cell end" and "mid-cell".
    """
    cap = cfg.cell_width_um / 2.0
    x = np.asarray(x, dtype=float)
    w = np.full(x.shape, cap, dtype=float)
    left = x < cap
    right = x > cfg.cell_length_um - cap
    w[left] = np.sqrt(np.clip(cap ** 2 - (x[left] - cap) ** 2, 0.0, None))
    w[right] = np.sqrt(
        np.clip(cap ** 2 - (x[right] - (cfg.cell_length_um - cap)) ** 2, 0.0, None))
    return w


def capsule_outline(cfg: SpatialSQDConfig, n: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """The capsule as a closed analytic curve.

    Contouring the boolean bin mask instead traced the bin grid - a staircase at the caps,
    and nothing along the straight walls, where the mask edge IS the array edge and
    `contour` draws nothing. The shape is closed-form, so it is drawn closed-form.
    """
    cap = cfg.cell_width_um / 2.0
    t = np.linspace(0.0, 2.0 * np.pi, n)
    cx = np.where(np.cos(t) >= 0.0, cfg.cell_length_um - cap, cap)
    return cx + cap * np.cos(t), cap * np.sin(t)


def fold_counts(counts: np.ndarray) -> np.ndarray:
    """Sum each bin's four-fold reflection orbit and mirror it back over the whole cell.

    SUMS, never maxes: the legacy fold takes `max` of four reflections, which is biased
    upward and hardest where counts are lowest (measured median 1.16-1.28 across the
    twelve groups). The result is 4-fold symmetric BY CONSTRUCTION, so symmetry in a
    folded panel is not a finding.
    """
    c = np.asarray(counts, dtype=float)
    return c + c[::-1, :] + c[:, ::-1] + c[::-1, ::-1]


#: The along-the-cell profile's bins: distance from the nearer pole, 0 to half the drawn length.
PROFILE_BINS_LONG = 24
#: The manuscript's short-axis profile (`manuscript/sqdmap`, `tables/config.json`): 60 bins of the SIGNED short-axis
#: position over [-1, 1], from the jumps in the cylinder only -- the caps (this share of the length at each end) are
#: left out, so a cap's curvature never reads as the membrane. The SQD map's own config carries the same two values;
#: the apparent-D and HMM maps' configs carry none, so these are the card's for every method.
MANUSCRIPT_POLE_MARGIN = 0.18
MANUSCRIPT_PROFILE_BINS = 60


def short_axis_profile(d_norm, l_norm, weight, *, pole_margin: float = MANUSCRIPT_POLE_MARGIN,
                       n_bins: int = MANUSCRIPT_PROFILE_BINS) -> dict:
    """THE short-axis profile, once: the manuscript's (`profiles.csv`, `panels.csv`) and the Spatial Map card's.

    The weighted density of the SIGNED ``d_norm`` over [-1, 1] in ``n_bins`` bins, area 1, from the rows in the
    cylinder (``pole_margin < l_norm < 1 - pole_margin``) with ``|d_norm| <= 1``; the weighted counts per bin; the
    mean ``|d_norm|`` (SP2's measure); the weight behind them; and the even-through-the-volume reference
    (`_project_uniform`, mean |d_norm| 4/(3 pi))."""
    d_norm = np.asarray(d_norm, dtype=float)
    l_norm = np.asarray(l_norm, dtype=float)
    weight = np.asarray(weight, dtype=float)
    in_cylinder = (l_norm > pole_margin) & (l_norm < 1.0 - pole_margin)
    usable = in_cylinder & np.isfinite(d_norm) & (np.abs(d_norm) <= 1.0)
    u = d_norm[usable]
    w = weight[usable]
    edges = np.linspace(-1.0, 1.0, n_bins + 1)
    centres = 0.5 * (edges[1:] + edges[:-1])
    counts = np.histogram(u, bins=edges, weights=w)[0]
    area = counts.sum() * (edges[1] - edges[0])
    density = counts / area if area > 0 else counts
    return {"edges": edges, "centres": centres, "counts": counts, "density": density,
            "mean_abs_d": float(np.average(np.abs(u), weights=w)) if w.sum() > 0 else float("nan"),
            "weight": float(w.sum()), "reference": _project_uniform(centres)}


def axis_profiles(x, y, weights_by_label: dict, cfg, *, n_short: int | None = None) -> dict:
    """WHERE EACH POPULATION SITS ALONG AND ACROSS THE CELL, from the drawn rows themselves (asked 17 Oct 2026: "a
    distribution graph ... to measure distance from poles like the manuscript scripts (x and y direction)"; made the
    manuscript's the same day: "can we not do it more like it is in manuscript version?").

    ``x``/``y`` are the rows' positions on the drawn capsule (um; `long_axis_x`, ``d_norm * local_half_width``) and
    ``weights_by_label`` each population's weight per row, exactly what its map was binned with. Per population:

    * ACROSS: the manuscript's short-axis profile (`short_axis_profile`: the SIGNED d_norm over [-1, 1], 60 bins, the
      cylinder of the drawn capsule only), beside the even-through-the-volume reference (2/pi) sqrt(1 - u^2), and its
      mean |d_norm| (SP2's measure, 0.424 for an even population);
    * ALONG: the density of the distance from the NEARER pole (both poles folded: which is which is arbitrary),
      beside the even reference (the cross-section area), its mean and the share in the caps.

    Each density has area 1; the weighted counts per bin ride along (the bars' readout), and so does the drawn
    geometry (the bands a bar marks on the maps, `spatial_map_view.band_polygons`). ``n_short`` overrides the
    across bins (SDA's profiles use `spatialsda.SDA_PROFILE_BINS`: a mean per bin needs coarser ones)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    length, cap = float(cfg.cell_length_um), float(cfg.cell_width_um) / 2.0
    half = length / 2.0
    margin = float(getattr(cfg, "pole_margin", MANUSCRIPT_POLE_MARGIN))
    n_short = int(n_short if n_short is not None else getattr(cfg, "n_profile_bins", MANUSCRIPT_PROFILE_BINS))
    s = np.clip(np.minimum(x, length - x), 0.0, half)
    hw = local_half_width(x, cfg)
    d_norm = np.divide(y, hw, out=np.zeros_like(y), where=hw > 0)
    l_norm = x / length if length > 0 else np.zeros_like(x)
    long_edges = np.linspace(0.0, half, PROFILE_BINS_LONG + 1)
    long_c = 0.5 * (long_edges[1:] + long_edges[:-1])
    fine = np.linspace(0.0, half, 4001)
    area = local_half_width(fine, cfg) ** 2
    norm = np.trapezoid(area, fine)
    long_ref = local_half_width(long_c, cfg) ** 2 / norm
    in_caps_even = float(np.trapezoid(area[fine <= cap], fine[fine <= cap]) / norm)
    empty = short_axis_profile([], [], [], pole_margin=margin, n_bins=n_short)
    populations = {}
    for label, w in weights_by_label.items():
        w = np.asarray(w, dtype=float)
        total = float(w.sum())
        if total <= 0:
            populations[label] = {"weight": 0.0, "long": [0.0] * PROFILE_BINS_LONG,
                                  "long_counts": [0.0] * PROFILE_BINS_LONG, "short": [0.0] * n_short,
                                  "short_counts": [0.0] * n_short, "short_weight": 0.0, "mean_abs_d": None,
                                  "mean_from_pole_um": None, "in_caps": None}
            continue
        long_counts = np.histogram(s, bins=long_edges, weights=w)[0]
        across = short_axis_profile(d_norm, l_norm, w, pole_margin=margin, n_bins=n_short)
        populations[label] = {
            "weight": total, "long": (long_counts / (total * np.diff(long_edges))).tolist(),
            "long_counts": long_counts.tolist(),
            "short": across["density"].tolist(), "short_counts": across["counts"].tolist(),
            "short_weight": across["weight"],
            "mean_abs_d": None if not np.isfinite(across["mean_abs_d"]) else across["mean_abs_d"],
            "mean_from_pole_um": float(np.sum(w * s) / total),
            "in_caps": float(np.sum(w[s <= cap]) / total)}
    return {"long_centres": long_c.tolist(), "short_centres": empty["centres"].tolist(),
            "long_edges": long_edges.tolist(), "short_edges": empty["edges"].tolist(),
            "long_reference": long_ref.tolist(), "short_reference": empty["reference"].tolist(),
            "cap_um": cap, "half_length_um": half, "in_caps_even": in_caps_even,
            "mean_abs_d_even": float(4.0 / (3.0 * np.pi)), "populations": populations,
            "geometry": {"cell_length_um": length, "cell_width_um": float(cfg.cell_width_um),
                         "pole_margin": margin}}


def _project_uniform(u: np.ndarray) -> np.ndarray:
    """Short-axis density of a population spread evenly through the cell VOLUME."""
    u = np.asarray(u, dtype=float)
    return _UNIFORM_PEAK * np.sqrt(np.clip(1.0 - u ** 2, 0.0, None))


#: A jump spans two localisations, so "where it happened" is a CONVENTION, not a
#: measurement, and this project has measured what the choice is worth: on the legacy SDA
#: maps the midpoint convention gives a low-D radial contrast of 2.92 against the start's
#: 1.63, median over twelve groups, midpoint exceeding start in 12 of 12. The mechanism is
#: geometric - a long step STARTING near the membrane can only travel inward, so the start
#: convention places long steps peripherally and the midpoint convention pulls them in.
#: All three are computed so the figure can quote the range instead of one number.
STEP_POSITIONS = ("A_start", "B_midpoint", "C_end")


def _step_coordinates(in_cell: pd.DataFrame, labelled: pd.DataFrame, meshes,
                      lag_frames: int) -> dict[str, pd.DataFrame]:
    """Cell coordinates of each step under all three conventions, row-aligned to `labelled`.

    The midpoint and end are RE-PROJECTED through the centreline rather than averaged in
    `l_norm`/`d_norm`. Averaging normalised coordinates would be wrong near the caps,
    where `d_norm` is scaled by the LOCAL half-width, so the same Cartesian midpoint maps
    to different `d_norm` depending on which endpoint's half-width you divide by.
    """
    from smtracker.cellcoords import project_tracks

    coords = {"A_start": labelled[["l_norm", "d_norm"]].reset_index(drop=True)}
    indexed = in_cell.set_index(["uid", "frame"])
    uid = labelled["uid"].to_numpy()
    frame = labelled["frame"].to_numpy()
    names = ["uid", "frame"]
    start = indexed.reindex(pd.MultiIndex.from_arrays([uid, frame], names=names))
    end = indexed.reindex(
        pd.MultiIndex.from_arrays([uid, frame + lag_frames], names=names))
    if end[["x_um", "y_um"]].isna().to_numpy().any():
        raise ValueError(
            "a step has no localisation at frame+lag; squared_displacements was built "
            "with pairing='frame' so every step should be contiguous"
        )

    for name, weight_end in (("B_midpoint", 0.5), ("C_end", 1.0)):
        moved = start.reset_index()
        moved["x_um"] = ((1.0 - weight_end) * start["x_um"].to_numpy()
                         + weight_end * end["x_um"].to_numpy())
        moved["y_um"] = ((1.0 - weight_end) * start["y_um"].to_numpy()
                         + weight_end * end["y_um"].to_numpy())
        moved.attrs = dict(in_cell.attrs)
        reprojected = project_tracks(moved, meshes, require_centerline=False)
        coords[name] = reprojected[["l_norm", "d_norm"]].reset_index(drop=True)
    return coords


def compute_condition(root, condition: str, day_map: dict, cfg: SpatialSQDConfig,
                      protein: str, params=None, skip_unsegmented: bool = False,
                      assignment: str = "hard") -> dict[str, pd.DataFrame]:
    """Load one protein x condition and return the three tidy tables.

    Imported lazily so `draw` never pulls the analysis stack in - that is the firewall.
    """
    from smtracker.grouping import load_utrack_project
    from smtracker.cellmesh import read_project_cell_meshes, assign_cells
    from smtracker.canonical import to_canonical
    from smtracker.utrack_meta import acquisition_params_from_project
    from smtracker.cellcoords import project_tracks
    from smtracker.observables import squared_displacements
    from smtracker.sqd import fit_sqd
    from smtracker import maps

    root = Path(root)
    raw = load_utrack_project(root, condition=condition, day=day_map)
    meshes = read_project_cell_meshes(root)

    # A MOVIE WITH NO OUTLINES IS DROPPED BY NAME, NEVER SILENTLY. `assign_cells` refuses
    # to mix segmented and unsegmented movies, and it is right to - the cell filter is
    # what makes a track's position meaningful, so an unsegmented movie would contribute
    # tracks that no cell filter ever saw. Dropping it is a data decision, so the movies
    # are listed in the returned table rather than merely excluded.
    dropped: list[str] = []
    if skip_unsegmented:
        have = set(meshes)
        present = sorted(set(raw["movie"].astype(str)))
        dropped = [m for m in present
                   if m not in have and m.split(":")[-1] not in have]
        if dropped:
            keep = ~raw["movie"].astype(str).isin(dropped)
            raw = raw.loc[keep].copy()
            if raw.empty:
                raise ValueError(
                    f"{protein}/{condition}: every movie lacks cell outlines")
    # WHY AN OVERRIDE EXISTS AT ALL, given that this project reads acquisition settings
    # and never guesses them. Some u-track projects were saved without `timeInterval_`,
    # and `acquisition_params_from_project` rightly refuses those. The SQD mixture is
    # SCALE-EQUIVARIANT - verified: scaling every displacement by k leaves the fractions
    # and the per-step posteriors identical to 3e-10 while D scales by exactly k^2 - so
    # the maps and the labels do not depend on the calibration and only D does. A caller
    # who supplies params therefore unblocks the map without being able to fake a D: the
    # source is stamped on every row, and a supplied one says so.
    if params is None:
        params = acquisition_params_from_project(root)
        acquisition_source = "movieData"
    else:
        acquisition_source = "supplied"
    canon = to_canonical(assign_cells(raw, meshes, drop_unassigned=False), params)
    in_cell = canon[canon["cell"].notna()].copy()
    proj = project_tracks(in_cell, meshes, require_centerline=False)
    proj = proj[proj["l_norm"].notna()].copy()

    fit = fit_sqd(squared_displacements(proj), n_pop=cfg.n_pop, method=cfg.method,
                  weighting=cfg.weighting)
    if fit.immobile:
        # the deposit's short / long and its closed-form cliff are two MOVING populations; exact-zero jumps (`sqd`'s
        # module head) are a synthetic truth's, never a u-track project's, and are not deposited
        raise ValueError(f"{protein}/{condition}: {fit.zero_jumps.note} The deposited tables hold two moving "
                         f"populations; map this condition in the app's Spatial Map instead.")
    D = np.asarray(fit.D, dtype=float)
    slow_index = int(np.argmin(D))
    fractions = np.asarray(fit.fractions, dtype=float).ravel()
    slow_fraction = float(fractions[slow_index])

    if assignment not in ("hard", "soft"):
        raise ValueError(f"assignment must be 'hard' or 'soft', got {assignment!r}")
    if cfg.n_pop != 2:
        raise ValueError(
            f"the deposited tables use 'short' and 'long' columns, which fit two populations; "
            f"got n_pop={cfg.n_pop}. The Spatial Map can draw more, but a {cfg.n_pop}-population deposit "
            f"needs those columns named first.")

    labelled = maps.labels_from_sqd(proj, fit)

    # HARD ASSIGNMENT HAS A CLIFF AND THE CLIFF IS COMPUTABLE IN ADVANCE.
    # For a two-exponential mixture the slow state's posterior is largest at r^2 = 0,
    # where it equals (f_s/D_s) / (f_s/D_s + f_f/D_f). If that is at or below 0.5 the
    # slow state is never the argmax for ANY step, so a hard-assigned short-jump map
    # cannot exist - not because the population is absent, but because it is never the
    # single most likely explanation - the panel is simply blank, and nothing warns you.
    # Measured across all 23 deposited groups, every one clears 0.5, but the margin
    # tracks occupancy so the low-occupancy groups sit against the edge and draw a map at
    # barely better than a coin flip, while high-occupancy groups clear it by a wide
    # margin. A group BELOW the cliff was seen in another lab project, whose table is not among
    # these deposits, so the possibility is real and that specific value is unverified
    # here. Which is why this is computed and deposited per group rather than assumed.
    means = 4.0 * D * fit.tau_lag_s
    weights_slow = (slow_fraction / means[slow_index])
    weights_fast = ((1.0 - slow_fraction) / means[1 - slow_index])
    max_posterior_slow = float(weights_slow / (weights_slow + weights_fast))

    values = labelled["sq_disp_um2"].to_numpy(float)
    fractions_all = np.asarray(fit.fractions, dtype=float).ravel()
    density = fractions_all[None, :] / means[None, :] * np.exp(
        -values[:, None] / means[None, :])
    posterior = density / density.sum(axis=1, keepdims=True)
    soft_weight = {"short": posterior[:, slow_index],
                   "long": posterior[:, 1 - slow_index]}

    states = sorted(labelled["state"].unique())
    if assignment == "hard" and len(states) != 2:
        raise ValueError(
            f"{protein}/{condition}: the two-population fit gives only {states} under "
            f"hard assignment, because the slow state's best-case posterior is "
            f"{max_posterior_slow:.3f} (<= 0.5), so argmax never selects it. "
            f"Use assignment='soft' to weight each step by its posterior instead."
        )
    all_states = sorted(np.unique(np.argmax(posterior, axis=1) + 1)) if states else []
    state_of = {"short": slow_index + 1, "long": (1 - slow_index) + 1}

    coordinates = _step_coordinates(in_cell, labelled, meshes, lag_frames=1)
    xe, ye = cfg.edges()
    xc, yc = cfg.centres()
    iy, ix = np.meshgrid(np.arange(cfg.n_bins_short), np.arange(cfg.n_bins_long),
                         indexing="ij")
    state_column = labelled["state"].to_numpy()
    # who contributed each step, for the per-bin n_eff beside every count (handover
    # §4a.4 step 1: written into the deposited table, no figure change)
    step_tracks, step_cells = contributor_labels(labelled, in_cell)

    panels, profiles, bins = [], [], []
    for position, frame in coordinates.items():
        l_norm = frame["l_norm"].to_numpy(float)
        d_norm = frame["d_norm"].to_numpy(float)
        for kind, state in state_of.items():
            if assignment == "hard":
                weight = (state_column == state).astype(float)
            else:
                weight = soft_weight[kind]
            # THE ONE short-axis profile (`short_axis_profile`), the Spatial Map card's too
            prof = short_axis_profile(d_norm, l_norm, weight, pole_margin=cfg.pole_margin,
                                      n_bins=cfg.n_profile_bins)
            profiles.append(pd.DataFrame({
                "condition": condition, "protein": protein, "kind": kind,
                "step_position": position,
                "d_norm": prof["centres"], "density": prof["density"],
                "even_distribution": prof["reference"]}))

            keep = np.isfinite(d_norm) & (np.abs(d_norm) <= 1.0)
            x = l_norm[keep] * cfg.cell_length_um
            y = d_norm[keep] * local_half_width(x, cfg)
            counts = np.histogram2d(x, y, bins=[xe, ye],
                                    weights=weight[keep])[0].T
            neff_t, neff_c = contributor_neff_grids(
                x, y, weight[keep], step_tracks[keep], step_cells[keep], xe, ye,
                mirrored=False)
            bins.append(pd.DataFrame({
                "condition": condition, "protein": protein, "kind": kind,
                "step_position": position,
                "ix": ix.ravel(), "iy": iy.ravel(),
                "x_um": xc[ix.ravel()], "y_um": yc[iy.ravel()],
                "count": counts.ravel(),
                "n_eff_track": neff_t.ravel(), "n_eff_cell": neff_c.ravel()}))

            panels.append({
                "condition": condition, "protein": protein, "kind": kind,
                "step_position": position,
                "n_steps_cylinder": prof["weight"],
                "n_steps_whole_cell": float(weight[keep].sum()),
                "assignment": assignment,
                "max_posterior_slow": max_posterior_slow,
                "hard_assignment_possible": max_posterior_slow > 0.5,
                "slow_fraction": slow_fraction,
                "D_slow_um2_s": float(D[slow_index]),
                "D_fast_um2_s": float(D[1 - slow_index]),
                "mean_abs_d_norm": prof["mean_abs_d"],
                "uniform_in_volume_mean_abs_d": 4.0 / (3.0 * np.pi),
                "pixel_size_um": float(params.pixel_size_nm) / 1000.0,
                "frame_interval_ms": float(params.frame_interval_ms),
                "acquisition_source": acquisition_source,
                "n_cells": int(in_cell["cell"].nunique()),
                "n_tracks": int(in_cell["uid"].nunique()),
                "movies_dropped_unsegmented": ";".join(dropped),
                "n_movies_dropped_unsegmented": len(dropped),
            })

    return {"panels": pd.DataFrame(panels),
            "profiles": pd.concat(profiles, ignore_index=True),
            "map_counts": pd.concat(bins, ignore_index=True)}


def write_tables(tables: dict[str, pd.DataFrame], cfg: SpatialSQDConfig,
                 out_dir) -> dict[str, Path]:
    """Write the tidy tables plus the config that describes their geometry."""
    import json

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, frame in tables.items():
        path = out_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        written[name] = path
    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    written["config"] = config_path
    return written


def read_tables(table_dir) -> tuple[dict[str, pd.DataFrame], SpatialSQDConfig]:
    """The drawing side's ONLY input. Opens no raw data and imports no analysis module."""
    import json

    table_dir = Path(table_dir)
    cfg = SpatialSQDConfig(**json.loads((table_dir / "config.json").read_text("utf-8")))
    tables = {name: pd.read_csv(table_dir / f"{name}.csv")
              for name in ("panels", "profiles", "map_counts")}
    return tables, cfg


def _grid_from_counts(frame: pd.DataFrame, cfg: SpatialSQDConfig) -> np.ndarray:
    grid = np.zeros((cfg.n_bins_short, cfg.n_bins_long), dtype=float)
    grid[frame["iy"].to_numpy(int), frame["ix"].to_numpy(int)] = frame["count"].to_numpy(float)
    return grid


def draw_profiles(tables: dict[str, pd.DataFrame], cfg: SpatialSQDConfig,
                  condition: str, proteins: list[str], *, step_position: str = "A_start"):
    """Short-axis distributions, short over long, one legend per row."""
    import matplotlib.pyplot as plt

    profiles = tables["profiles"]
    n = len(proteins)
    fig, axes = plt.subplots(2, n, figsize=(2.45 * n, 5.0), constrained_layout=True,
                             sharex=True, sharey=True, squeeze=False)
    for column, protein in enumerate(proteins):
        for row, (kind, colour) in enumerate((("short", SHORT_COLOUR),
                                              ("long", LONG_COLOUR))):
            ax = axes[row, column]
            sub = profiles[(profiles.condition == condition)
                           & (profiles.protein == protein)
                           & (profiles.kind == kind)
                           & (profiles.step_position == step_position)]
            if sub.empty:
                ax.set_axis_off()
                continue
            ax.plot(sub["d_norm"], sub["even_distribution"], "--",
                    color=REFERENCE_COLOUR, lw=1.2, label="even distribution")
            ax.plot(sub["d_norm"], sub["density"], color=colour, lw=1.4,
                    label=f"{kind} jumps")
            ax.set_xlim(-1, 1)
            ax.set_ylim(0, 1.35)
            ax.set_title(protein, loc="left", fontsize=9)
            if column == 0:
                ax.legend(fontsize=7, loc="lower center", framealpha=0.9)
                ax.set_ylabel("probability density")
            if row == 1:
                ax.set_xlabel("relative short-axis position")
    return fig


def draw_maps(tables: dict[str, pd.DataFrame], cfg: SpatialSQDConfig, condition: str,
              proteins: list[str], kind: str, *, mirrored: bool, scale: str,
              step_position: str = "A_start", cmap: str = DEFAULT_CMAP, n_columns: int = 2):
    """One capsule per protein.

    `scale="panel"` normalises each panel to its own maximum; `scale="shared"` turns each
    map into a probability density and puts them all on one scale. The choice is printed
    on the colour bar, because an unlabelled bar over per-panel scaling silently invites
    a between-panel comparison the figure cannot support.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import PathPatch
    from scipy.ndimage import gaussian_filter

    if scale not in ("panel", "shared"):
        raise ValueError(f"scale must be 'panel' or 'shared', got {scale!r}")

    counts = tables["map_counts"]
    cap_x, cap_y = capsule_outline(cfg)
    outline = np.column_stack([cap_x, cap_y])
    half = cfg.cell_width_um / 2.0
    xc, yc = cfg.centres()
    XX, YY = np.meshgrid(xc, yc)
    inside = np.abs(YY) <= local_half_width(XX, cfg)

    grids = {}
    for protein in proteins:
        sub = counts[(counts.condition == condition) & (counts.protein == protein)
                     & (counts.kind == kind)
                     & (counts.step_position == step_position)]
        if sub.empty:
            continue
        grid = _grid_from_counts(sub, cfg)
        if mirrored:
            grid = fold_counts(grid)
        if scale == "shared":
            total = grid.sum()
            if total > 0:
                grid = grid / total
        grids[protein] = gaussian_filter(grid, sigma=cfg.smooth_bins)

    if not grids:
        raise ValueError(f"no {kind} map for {condition} at {step_position}")
    shared_max = max(float(np.nanpercentile(g[inside], 99)) for g in grids.values())

    n_rows = int(np.ceil(len(proteins) / n_columns))
    fig, axes = plt.subplots(n_rows, n_columns, squeeze=False,
                             figsize=(5.5 * n_columns, 1.55 * n_rows),
                             constrained_layout=True)
    flat = axes.ravel()
    image = None
    for index, protein in enumerate(proteins):
        ax = flat[index]
        if protein not in grids:
            ax.set_axis_off()
            continue
        grid = grids[protein]
        vmax = shared_max if scale == "shared" else float(np.nanpercentile(grid[inside], 99))
        image = ax.imshow(grid, origin="lower", cmap=cmap, vmin=0.0, vmax=vmax,
                          extent=[0.0, cfg.cell_length_um, -half, half],
                          interpolation="bilinear", aspect="equal")
        # CLIPPED, NOT MASKED. Dropping bins whose centre falls outside left a staircase
        # of white inside the smooth outline AND discarded the outermost bins, which hold
        # the steps nearest the membrane - the part of the distribution the figure is
        # about. Clipping cuts at vector resolution and resamples nothing.
        clip = PathPatch(MplPath(outline), transform=ax.transData, facecolor="none", lw=0)
        ax.add_patch(clip)
        image.set_clip_path(clip)
        ax.plot(cap_x, cap_y, color="0.15", lw=1.3,
                path_effects=[pe.withStroke(linewidth=3.0, foreground="white")])
        ax.set_xlim(-0.05, cfg.cell_length_um + 0.05)
        ax.set_ylim(-half - 0.05, half + 0.05)
        ax.set_title(protein, loc="left", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    for extra in range(len(proteins), n_rows * n_columns):
        flat[extra].set_axis_off()

    bar = fig.colorbar(image, ax=axes, orientation="horizontal", fraction=0.035,
                       pad=0.01, shrink=0.4)
    bar.set_ticks([0.0, image.get_clim()[1]])
    bar.set_ticklabels(["low", "high"])
    bar.ax.tick_params(length=0, labelsize=8)
    bar.set_label(
        f"density of {kind} jumps  "
        f"({'each cell on its own scale' if scale == 'panel' else 'one scale, all cells'})",
        fontsize=8)
    return fig


def _prepare_grid(counts: pd.DataFrame, cfg: SpatialSQDConfig, condition: str,
                  protein: str, kind: str, step_position: str, *, mirrored: bool,
                  normalise: bool):
    from scipy.ndimage import gaussian_filter

    sub = counts[(counts.condition == condition) & (counts.protein == protein)
                 & (counts.kind == kind) & (counts.step_position == step_position)]
    if sub.empty:
        return None
    grid = _grid_from_counts(sub, cfg)
    if mirrored:
        grid = fold_counts(grid)
    if normalise:
        total = grid.sum()
        if total > 0:
            grid = grid / total
    return gaussian_filter(grid, sigma=cfg.smooth_bins)


def draw_condition_matrix(tables: dict[str, pd.DataFrame], cfg: SpatialSQDConfig, *,
                          protein: str, conditions: list[str],
                          kinds: tuple[str, ...] = ("short", "long"),
                          mirrored: bool, step_position: str = "A_start",
                          cmap: str = DEFAULT_CMAP, annotate_n: bool = True,
                          conditions_as: str = "rows", scale: str = "panel"):
    """EVERY condition of one protein in ONE figure: conditions against jump classes.

    `draw_maps` lays out PROTEINS for one condition, which is the right shape when
    several enzymes are compared side by side. A single protein measured under five
    conditions is the other shape, and drawing it with `draw_maps` gives five separate
    one-panel figures, which is what pyk produced, twenty files that a reader has to
    hold in their head to compare.

    `scale="panel"` is the default because it is what every other map here does.
    `draw_maps` defaults to it, and the manuscript's TCA overview takes `vmax` from each
    panel's own grid at all three of its call sites and normalises nothing. The first
    version of this function shipped `column` scaling instead, which changed BOTH the
    normalisation and the ceiling, so the same data drawn as a matrix and as separate
    panels would not have matched, and nobody had asked for that. Caught by the
    experimenter.

    The other two are available and each answers a different question, so the choice is
    the caller's and the colour bar always prints which was used:

    * ``"panel"``: every capsule on its own maximum. Shows the SHAPE of each map and
      supports no comparison of level between panels.
    * ``"column"``: one ceiling per jump class, over every condition drawn. This is
      what makes conditions comparable to each other, and it is deliberately NOT shared
      across columns: short and long jumps are different populations whose densities
      here differ by roughly fivefold, so one scale over both would flatten the sparser
      into a wash. Each column then gets its own bar.
    * ``"shared"``: one ceiling over everything, after turning each map into a
      probability density. Honest and usually unreadable here, for the reason above.

    N goes BELOW each capsule rather than on it: written inside, its legibility would
    depend on the colormap, and a number that is sometimes unreadable is worse than one
    always in the same place. Under soft assignment it is a sum of weights, so it says
    "weighted steps" rather than pretending to be a count.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import PathPatch

    if conditions_as not in ("rows", "columns"):
        raise ValueError(f"conditions_as must be 'rows' or 'columns', "
                         f"got {conditions_as!r}")
    if scale not in ("panel", "column", "shared"):
        raise ValueError(f"scale must be 'panel', 'column' or 'shared', got {scale!r}")
    if not conditions:
        raise ValueError("no conditions given")
    for kind in kinds:
        if kind not in ("short", "long"):
            raise ValueError(f"kind must be 'short' or 'long', got {kind!r}")

    counts = tables["map_counts"]
    panels = tables.get("panels")
    cap_x, cap_y = capsule_outline(cfg)
    outline = np.column_stack([cap_x, cap_y])
    half = cfg.cell_width_um / 2.0
    xc, yc = cfg.centres()
    XX, YY = np.meshgrid(xc, yc)
    inside = np.abs(YY) <= local_half_width(XX, cfg)

    grids: dict[tuple[str, str], np.ndarray] = {}
    for condition in conditions:
        for kind in kinds:
            # NORMALISE ONLY WHERE A SHARED CEILING NEEDS IT. `panel` and `column` draw
            # the smoothed COUNTS, exactly as `draw_maps` and the TCA overview do, so a
            # matrix and a separate panel of the same data are the same picture.
            grid = _prepare_grid(counts, cfg, condition, protein, kind, step_position,
                                 mirrored=mirrored, normalise=scale == "shared")
            if grid is not None:
                grids[(condition, kind)] = grid
    if not grids:
        raise ValueError(f"no map for {protein} at {step_position}")

    def ceiling_of(grid) -> float:
        return float(np.nanpercentile(grid[inside], 99))

    ceiling: dict[tuple[str, str], float] = {}
    if scale == "panel":
        ceiling = {key: ceiling_of(g) for key, g in grids.items()}
    elif scale == "column":
        for kind in kinds:
            present = [g for (_c, k), g in grids.items() if k == kind]
            top = max(ceiling_of(g) for g in present) if present else 1.0
            ceiling.update({key: top for key in grids if key[1] == kind})
    else:
        top = max(ceiling_of(g) for g in grids.values())
        ceiling = dict.fromkeys(grids, top)

    by_row = conditions_as == "rows"
    n_rows = len(conditions) if by_row else len(kinds)
    n_cols = len(kinds) if by_row else len(conditions)

    # THE PANEL IS SIZED FROM THE DATA BOX, NOT FROM THE CAPSULE. Annotating N extends
    # the y limits below the cell; with aspect="equal" the drawn capsule then shrinks
    # horizontally while the axes keeps its width, and the capsules float apart with
    # white space between them. Deriving the width from the box actually plotted removes
    # the slack at source instead of nudging padding afterwards.
    below = 0.34 if annotate_n else 0.06
    row_height = 1.30
    data_w = cfg.cell_length_um + 0.10
    data_h = cfg.cell_width_um + 0.06 + below
    panel_width = row_height * (data_w / data_h)
    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False,
                             figsize=(panel_width * n_cols + 1.6,
                                      row_height * n_rows + 1.3),
                             constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.03, wspace=0.02)

    images: dict[str, object] = {}
    for ci, condition in enumerate(conditions):
        for ki, kind in enumerate(kinds):
            row, column = (ci, ki) if by_row else (ki, ci)
            ax = axes[row, column]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_anchor("C")
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xlim(-0.05, cfg.cell_length_um + 0.05)
            ax.set_ylim(-half - below, half + 0.06)

            # Labels are written from the ROW and COLUMN, never from a panel that may be
            # absent, otherwise a condition with no map leaves its row anonymous.
            if by_row and column == 0:
                ax.set_ylabel(condition, fontsize=11, rotation=0, ha="right",
                              va="center")
            if by_row and row == 0:
                ax.set_title(STATE_TITLE[kind], fontsize=11)
            if not by_row and column == 0:
                ax.set_ylabel(STATE_TITLE[kind], fontsize=11, rotation=0, ha="right",
                              va="center")
            if not by_row and row == 0:
                ax.set_title(condition, fontsize=11)

            grid = grids.get((condition, kind))
            if grid is None:
                ax.plot(cap_x, cap_y, color="0.75", lw=0.9, ls="--")
                ax.text(cfg.cell_length_um / 2.0, 0.0, "no map",
                        ha="center", va="center", fontsize=8, color="0.45")
                continue

            image = ax.imshow(grid, origin="lower", cmap=cmap, vmin=0.0,
                              vmax=ceiling[(condition, kind)],
                              extent=[0.0, cfg.cell_length_um, -half, half],
                              interpolation="bilinear", aspect="equal")
            # CLIPPED, NOT MASKED, as in `draw_maps`: dropping bins whose centre falls
            # outside leaves a staircase of white inside a smooth outline AND discards
            # the outermost bins, which hold the steps nearest the membrane.
            clip = PathPatch(MplPath(outline), transform=ax.transData,
                             facecolor="none", lw=0)
            ax.add_patch(clip)
            image.set_clip_path(clip)
            ax.plot(cap_x, cap_y, color="0.15", lw=1.2,
                    path_effects=[pe.withStroke(linewidth=2.6, foreground="white")])
            images[kind] = image

            if annotate_n and panels is not None:
                match = panels[(panels.condition == condition)
                               & (panels.protein == protein)
                               & (panels.kind == kind)
                               & (panels.step_position == step_position)]
                if len(match):
                    value = float(match.iloc[0].n_steps_whole_cell)
                    soft = str(match.iloc[0].get("assignment", "hard")) == "soft"
                    text = (f"N = {value:,.0f} weighted steps" if soft
                            else f"N = {value:,.0f} steps")
                else:
                    text = "N = 0 steps"
                ax.text(cfg.cell_length_um / 2.0, -half - 0.12, text, ha="center",
                        va="top", fontsize=7.5, color="0.2")

    # ONE BAR PER JUMP CLASS, AND THE BAR SAYS WHAT IT SUPPORTS. Under `panel` the
    # ceiling differs from capsule to capsule, so the bar carries no numbers and says so
    #, an unlabelled bar over per-panel scaling silently invites a between-panel
    # comparison the figure cannot support, which is the same warning `draw_maps`
    # carries. Two bars rather than one because the columns never share a scale.
    # SHORT ENOUGH TO FIT UNDER ONE COLUMN. Two bars sit side by side here, so a label
    # written at the length `draw_maps` uses runs into its neighbour and both become
    # unreadable, which is worse than the terse version, since an unlabelled-looking
    # bar is exactly what invites the between-panel comparison this scaling cannot
    # support. The full sentence belongs in the caption.
    meaning = {
        "panel": "own scale per cell: shape, not level",
        "column": "one scale down this column",
        "shared": "one scale, whole figure (density)",
    }[scale]
    for kind in kinds:
        if kind not in images:
            continue
        ki = list(kinds).index(kind)
        column_axes = [axes[ci, ki] if by_row else axes[ki, ci]
                       for ci in range(len(conditions))]
        bar = fig.colorbar(images[kind], ax=column_axes, orientation="horizontal",
                           fraction=0.045, pad=0.02, shrink=0.75)
        bar.set_ticks([0.0, images[kind].get_clim()[1]])
        bar.set_ticklabels(["low", "high"])
        bar.ax.tick_params(length=0, labelsize=7.5)
        bar.set_label(f"{STATE_TITLE[kind]}: {meaning}", fontsize=7.5)
    return fig


def draw_convention_comparison(tables: dict[str, pd.DataFrame], cfg: SpatialSQDConfig,
                               condition: str, proteins: list[str], kind: str, *,
                               mirrored: bool, cmap: str = DEFAULT_CMAP):
    """The same jumps drawn at A (start), B (midpoint) and C (end).

    The point is that NONE of the three is the TRUE one. A jump is a displacement; it
    occupied all three positions. The columns differ because a long step starting near
    the membrane can only travel inward, so A pushes long steps out and C pulls them in,
    with B between. Read the SPREAD across columns as the convention's cost, and quote a
    contrast only with the convention that produced it.

    Every panel is on its OWN maximum, so this figure compares ARRANGEMENT across
    conventions and never density.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import PathPatch

    counts = tables["map_counts"]
    cap_x, cap_y = capsule_outline(cfg)
    outline = np.column_stack([cap_x, cap_y])
    half = cfg.cell_width_um / 2.0
    xc, yc = cfg.centres()
    XX, YY = np.meshgrid(xc, yc)
    inside = np.abs(YY) <= local_half_width(XX, cfg)

    n_rows = len(proteins)
    fig, axes = plt.subplots(n_rows, len(STEP_POSITIONS), squeeze=False,
                             figsize=(4.2 * len(STEP_POSITIONS), 1.5 * n_rows),
                             constrained_layout=True)
    for row, protein in enumerate(proteins):
        for column, position in enumerate(STEP_POSITIONS):
            ax = axes[row, column]
            grid = _prepare_grid(counts, cfg, condition, protein, kind, position,
                                 mirrored=mirrored, normalise=False)
            if grid is None:
                ax.set_axis_off()
                continue
            image = ax.imshow(grid, origin="lower", cmap=cmap, vmin=0.0,
                              vmax=float(np.nanpercentile(grid[inside], 99)),
                              extent=[0.0, cfg.cell_length_um, -half, half],
                              interpolation="bilinear", aspect="equal")
            clip = PathPatch(MplPath(outline), transform=ax.transData,
                             facecolor="none", lw=0)
            ax.add_patch(clip)
            image.set_clip_path(clip)
            ax.plot(cap_x, cap_y, color="0.15", lw=1.3,
                path_effects=[pe.withStroke(linewidth=3.0, foreground="white")])
            ax.set_xlim(-0.05, cfg.cell_length_um + 0.05)
            ax.set_ylim(-half - 0.05, half + 0.05)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(position.replace("_", "  "), loc="left", fontsize=9)
            if column == 0:
                ax.set_ylabel(protein, fontsize=9, rotation=0, ha="right", va="center")
    return fig


def convention_contrast(tables: dict[str, pd.DataFrame], cfg: SpatialSQDConfig,
                        n_shells: int = 5) -> pd.DataFrame:
    """Outer-shell over inner-shell density, per group and per convention.

    Reported as an enrichment, because the raw ratio has a neutral value that is not 1.
    The projected density of a population spread evenly through the VOLUME falls to zero
    at the membrane, so a uniform population gives an outer/inner ratio of about 0.39 on
    five shells. Quoting the raw 0.12 without that reference reads as "strongly depleted"
    when the honest statement is "about three times more central than uniform". Both
    columns are returned so neither can be reconstructed wrongly from the other.
    """
    profiles = tables["profiles"]
    edges = np.linspace(0.0, 1.0, n_shells + 1)
    rows = []
    for keys, group in profiles.groupby(["condition", "protein", "kind",
                                         "step_position"], sort=True):
        u = np.abs(group["d_norm"].to_numpy(float))
        density = group["density"].to_numpy(float)
        reference = group["even_distribution"].to_numpy(float)
        shell = np.clip(np.digitize(u, edges[1:-1]), 0, n_shells - 1)
        observed = np.array([density[shell == k].mean() if (shell == k).any() else np.nan
                             for k in range(n_shells)])
        expected = np.array([reference[shell == k].mean() if (shell == k).any() else np.nan
                             for k in range(n_shells)])
        enrichment = observed / expected
        rows.append({
            "condition": keys[0], "protein": keys[1], "kind": keys[2],
            "step_position": keys[3],
            "inner_shell_density": observed[0], "outer_shell_density": observed[-1],
            "raw_outer_over_inner": observed[-1] / observed[0],
            "uniform_outer_over_inner": expected[-1] / expected[0],
            "enrichment_inner": enrichment[0], "enrichment_outer": enrichment[-1],
            "enrichment_outer_over_inner": enrichment[-1] / enrichment[0],
        })
    return pd.DataFrame(rows)


# =====================================================================================
# THE GUI ENTRY POINT
#
# `compute_condition` takes a PROJECT ROOT because the manuscript pipeline starts from
# one. The browser does not: it already holds a loaded, projected canonical table, and
# asking it for a path would mean re-reading data it has in memory. So the tool calls
# this instead - a plain function over what the GUI already has, with no Qt in sight and
# no file system underneath it.
#
# SOFT IS THE DEFAULT HERE, unlike `compute_condition`'s hard, and that is deliberate:
# a user clicking a button gets no caption, and hard assignment can hand them an empty
# panel for a group whose slow state is real but never the argmax. Soft has no cliff.
# =====================================================================================

#: EVERY JUMP ONCE, beside the populations (3 Nov 2026, asked: "all jumps for 2 pop shows basically all jumps together
#: (as 1 pop) in the same field"): the label of the map a split run adds with ``all_jumps=True`` -- weight 1 on every
#: jump, whatever the fit or the ranges, so it is the map a one-population run of the same jumps draws (one
#: population's posterior is 1). A fitted split's populations sum to it (soft and hard alike); a split by hand's ranges
#: may leave jumps out of every range, never out of this one.
ALL_JUMPS_LABEL = "every jump"


#: What the tool calls the populations. "short"/"long" only at two populations, where it
#: is literally true; numbered above that, because "medium" asserts a real intermediate
#: STATE where this project's own record says the middle component may be measurement
#: heterogeneity, and it stops scaling at four.
def population_labels(n_pop: int, D=None) -> tuple[str, ...]:
    if n_pop == 1:
        # one population: every jump, nothing split (`spatial_map_view.MAP_N_POPS_BY_METHOD`, 1 Nov 2026)
        names = ("all jumps",)
    elif n_pop == 2:
        names = ("short jumps", "long jumps")
    else:
        names = tuple(f"population {i + 1}" for i in range(n_pop))
    if D is None:
        return names
    order = np.argsort(np.asarray(D, dtype=float))
    out = list(names)
    for rank, index in enumerate(order):
        d = float(np.asarray(D)[index])
        # the immobile population (`sqd`'s module head) is named for what it is, the jumps of length 0
        out[index] = "exact-zero jumps (D = 0)" if d == 0 else f"{names[rank]} (D = {d:.3g})"
    return tuple(out)


WEIGHTINGS = ("steps", "track")


def track_weights(labelled: pd.DataFrame, unit_col: str = "uid") -> np.ndarray:
    """`1 / (rows of that track)` for every row of `labelled`, aligned to it.

    The "every track counts once" factor: a track's rows then sum to 1, however
    long it is. Counted over the rows HANDED IN (the table that is drawn), so a
    track the cutoff shortened or a step the map later drops still divides by the
    rows this table holds for it.
    """
    if unit_col not in labelled.columns:
        raise ValueError(f"track weighting needs a '{unit_col}' column")
    counts = labelled.groupby(unit_col, sort=False)[unit_col].transform("size")
    return 1.0 / counts.to_numpy(float)


def contributor_labels(labelled: pd.DataFrame, projected: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """`(tracks, cells)`: who contributed each row of `labelled`, aligned to it -- the
    two units n_eff is computed in (handover §4a.2: by track AND by cell).

    The cell comes from `projected` through the row's track, so a table that does not
    carry `cell` itself (the per-step tables) still gets it, qualified by movie because
    cell numbers restart in every movie. A track with no cell counts as a cell of its
    own: it is one contributor, and merging every unassigned track into one "cell"
    would invent a dominant bacterium that does not exist.
    """
    if "uid" not in labelled.columns:
        raise ValueError("n_eff needs a 'uid' column to know which track each row came from")
    tracks = labelled["uid"].astype(str).to_numpy()
    if "cell" in projected.columns:
        first = projected.drop_duplicates("uid").set_index("uid")
        key = first["cell"].astype(str)
        if "movie" in first.columns:
            key = first["movie"].astype(str) + ":" + key
        cells = labelled["uid"].map(key.where(first["cell"].notna()))
        missing = cells.isna().to_numpy()
        cells = cells.astype(str).to_numpy()
    else:
        missing = np.ones(len(labelled), dtype=bool)
        cells = np.empty(len(labelled), dtype=object)
    cells = np.where(missing, np.char.add("track:", tracks.astype(str)), cells).astype(str)
    return tracks, cells


def contributor_neff_grids(x, y, weight, tracks, cells, xe, ye, *,
                           mirrored: bool) -> tuple[np.ndarray, np.ndarray]:
    """Per-bin n_eff by track and by cell, on the map's own grid (rows = y, columns = x,
    as `np.histogram2d(...)[0].T`), for one population's weighted rows.

    `weight` is the population's PLAIN weight per row -- the posterior under soft
    assignment, 0/1 under hard -- never the by-track weighting: weighting changes what a
    bin SHOWS, not how many contributors built it. Binning matches `np.histogram2d`
    exactly (last edge inclusive, outside dropped). Mirrored, each contributor's weight
    is pooled across a bin's four reflections BEFORE the formula -- the same orbit
    `fold_counts` sums -- and the result is written back to all four. The grouping
    itself is `smtracker.spatial.maps.contributor_sums`, the one shared function.
    """
    from .spatial.maps import contributor_sums, effective_contributors

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(weight, dtype=float)
    nx, ny = len(xe) - 1, len(ye) - 1
    ok = (x >= xe[0]) & (x <= xe[-1]) & (y >= ye[0]) & (y <= ye[-1]) & (w > 0)
    ix = np.clip(np.searchsorted(xe, x[ok], side="right") - 1, 0, nx - 1)
    iy = np.clip(np.searchsorted(ye, y[ok], side="right") - 1, 0, ny - 1)
    if mirrored:
        ix = np.minimum(ix, nx - 1 - ix)
        iy = np.minimum(iy, ny - 1 - iy)
    bin_id = iy * nx + ix
    out = []
    for who in (np.asarray(tracks)[ok], np.asarray(cells)[ok]):
        s1, s2 = contributor_sums(bin_id, who, w[ok], n_bins=nx * ny)
        grid = effective_contributors(s1, s2).reshape(ny, nx)
        if mirrored:
            rows = np.minimum(np.arange(ny), ny - 1 - np.arange(ny))
            cols = np.minimum(np.arange(nx), nx - 1 - np.arange(nx))
            grid = grid[np.ix_(rows, cols)]
        out.append(grid)
    return out[0], out[1]
def upper_envelope(intercepts, slopes) -> dict:
    """Which of several straight lines is HIGHEST at each x > 0, and where it changes.

    The shared core of every "cut where the fitted populations cross" in the maps. For
    a mixture whose components share a shape, each component's log weighted density is
    `const(x) + intercept + slope * x` in the right variable (r² for SQD, apparent D for
    appD), so the most likely component at x is the top line, and the cuts are where the
    top line changes. Lines must come sorted by ascending slope (= ascending D). A line
    that is never on top for x > 0 gets no window.

    Returns `{"cuts": [x, ...] ascending, "windows": [line index per window],
    "never_most_likely": [line index, ...]}`.
    """
    a = [float(v) for v in intercepts]
    b = [float(v) for v in slopes]

    def cross(i, j):
        return (a[i] - a[j]) / (b[j] - b[i])

    hull: list[int] = []
    for k in range(len(a)):
        while len(hull) >= 2 and cross(hull[-2], k) <= cross(hull[-2], hull[-1]):
            hull.pop()
        hull.append(k)
    while len(hull) >= 2 and cross(hull[0], hull[1]) <= 0.0:
        hull.pop(0)
    return {"cuts": [cross(hull[i], hull[i + 1]) for i in range(len(hull) - 1)],
            "windows": list(hull),
            "never_most_likely": [k for k in range(len(a)) if k not in hull]}


def sqd_crossings(D, fractions, tau_s: float) -> dict:
    """Where an SQD mixture's populations cross, as JUMP LENGTHS (µm): the hard
    assignment's cuts. Each population is `f_k/m_k exp(-r²/m_k)` in r², `m_k = 4 D_k τ`,
    so in r² the log densities are straight lines (`upper_envelope`). At two populations
    this is the single "shorter than x is short" threshold; at three the middle
    population owns a BAND between two cuts, or nothing. Population indices count by
    ascending D.

    The immobile population of exact-zero jumps (a D of 0, `sqd`'s module head) owns the jumps of length 0 and
    nothing else: its window comes first, ended by a cut at 0, and the moving populations' cuts follow."""
    D = np.asarray(D, dtype=float).ravel()
    f = np.asarray(fractions, dtype=float).ravel()
    order = np.argsort(D)
    if D[order[0]] == 0 and len(D) > 1:
        moving = sqd_crossings(D[order[1:]], f[order[1:]], tau_s)
        return {"cuts": [0.0, *moving["cuts"]], "windows": [0, *(k + 1 for k in moving["windows"])],
                "never_most_likely": [k + 1 for k in moving["never_most_likely"]],
                "cuts_um": [0.0, *moving["cuts_um"]]}
    m = 4.0 * D[order] * float(tau_s)
    env = upper_envelope(np.log(f[order] / m), -1.0 / m)
    return {**env, "cuts_um": [float(np.sqrt(x)) for x in env["cuts"]]}


def step_block(rows: pd.DataFrame, x, y, drawn: dict, cfg) -> dict:
    """WHICH TRACK EACH BINNED JUMP CAME FROM: the record behind a map, row by row.

    `rows` are the drawn rows (one per jump, or per localisation for appD), aligned
    with `x`/`y` in µm, and `drawn` maps each population to the weight that went into
    its DRAWN grid (posterior, 0/1, times the track weight when weighting by track).
    Returns numpy arrays: `uid`, the bin indices `ix`/`iy`, `w` per population, and
    `day`/`movie`/`cell` where the rows carry them.

    The bin is np.histogram2d's own: a value on an interior edge goes to the bin on
    its right, one on the last edge to the last bin, and anything outside the edges is
    dropped -- so these rows, summed per bin, rebuild the unfolded grid exactly
    (tested). It exists so a map can say which tracks a spot is made of (asked for:
    "identify that a spot is made by only 1 track"); `spatial_map_view.spot_contributors`
    reads it.
    """
    xe, ye = cfg.edges()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ix = np.searchsorted(xe, x, side="right") - 1
    iy = np.searchsorted(ye, y, side="right") - 1
    ix = np.where(x == xe[-1], len(xe) - 2, ix)
    iy = np.where(y == ye[-1], len(ye) - 2, iy)
    ok = (ix >= 0) & (ix < len(xe) - 1) & (iy >= 0) & (iy < len(ye) - 1)
    out = {"uid": rows["uid"].to_numpy()[ok], "ix": ix[ok], "iy": iy[ok],
           "w": {key: np.asarray(v, dtype=float)[ok] for key, v in drawn.items()}}
    for col in ("day", "movie", "cell"):
        if col in rows.columns:
            out[col] = rows[col].to_numpy()[ok]
    return out


def _check_weighting(weighting: str) -> None:
    if weighting not in WEIGHTINGS:
        raise ValueError(f"weighting must be one of {WEIGHTINGS}, got {weighting!r}")


def maps_from_projected(projected: pd.DataFrame, fit, cfg: SpatialSQDConfig, *,
                        assignment: str = "soft", step_position: str = "A_start",
                        mirrored: bool = False, meshes=None,
                        lag_frames: int = 1, weighting: str = "steps",
                        with_steps: bool = False, all_jumps: bool = False) -> dict[str, object]:
    """Per-population jump-density grids for ONE already-loaded, projected table.

    Returns a dict with one grid per population plus the diagnostics the panel must show:
    the summed weight behind each map, and whether hard assignment is even possible.

    At more than two populations the posterior is not monotonic in jump length. For two,
    hard assignment is exactly a threshold - "shorter than x nm" - and the panel may say
    so. For three or more the middle component's posterior peaks at an INTERMEDIATE
    length, so hard assignment selects a BAND, not a cut, and calling it a threshold
    would be wrong. `hard_rule` names which applies.

    `weighting="track"` multiplies every step's weight by `track_weights` (each
    track counts once). `step_grids`/`step_weights` are always the unmultiplied
    step counts -- what a minimum-per-bin mask must read, since it is about how
    much evidence a bin holds -- and are the SAME objects as `grids`/`weights`
    under the default `"steps"`.

    `with_steps=True` adds `"steps"`, the per-jump record of which track went into
    which bin with what drawn weight (`step_block`).

    `all_jumps=True` adds one more map, every jump at weight 1 (`ALL_JUMPS_LABEL`), when the fit has two or more
    populations; one population IS every jump. It is listed after the populations in ``labels``, so a population's
    index into ``labels`` is unchanged.
    """
    from . import maps as maps_module

    _check_weighting(weighting)
    if assignment not in ("hard", "soft"):
        raise ValueError(f"assignment must be 'hard' or 'soft', got {assignment!r}")
    if step_position not in STEP_POSITIONS:
        raise ValueError(f"step_position must be one of {STEP_POSITIONS}")

    from .sqd import posteriors

    D = np.asarray(fit.D, dtype=float)
    fractions = np.asarray(fit.fractions, dtype=float).ravel()
    n_pop = len(D)
    labelled = maps_module.labels_from_sqd(projected, fit, lag_frames=lag_frames)

    means = 4.0 * D * fit.tau_lag_s
    values = labelled["sq_disp_um2"].to_numpy(float)
    # `sqd.posteriors`, the one formula: an exact-zero jump is the immobile population's (D 0, `sqd`'s module head);
    # a per-spot fit's each step with its own noise (`maps.per_spot_steps`; ``labelled`` already holds only the steps
    # that carry it)
    _steps, noise = maps_module.per_spot_steps(labelled, fit)
    posterior = posteriors(values, D, fractions, fit.tau_lag_s, step_noise=noise)
    immobile = bool(np.any(D == 0))
    order = np.argsort(D)
    labels = population_labels(n_pop, D)
    if assignment == "hard":
        weights = [(np.argmax(posterior, axis=1) == index).astype(float) for index in range(n_pop)]
    else:
        weights = [posterior[:, index] for index in range(n_pop)]
    if all_jumps and n_pop > 1:
        weights, labels = [*weights, np.ones(len(labelled))], (*labels, ALL_JUMPS_LABEL)
    out = _jump_maps(projected, labelled, weights, labels, cfg, step_position=step_position,
                     mirrored=mirrored, meshes=meshes, lag_frames=lag_frames, weighting=weighting,
                     with_steps=with_steps)

    # Can hard assignment give every population a map at all? For two the answer is the
    # closed form; above two, ask the posteriors directly.
    if n_pop == 1:
        # ONE population: every jump in its one map, under soft and hard alike
        best_slow, hard_possible, hard_rule = 1.0, True, "one population, nothing split"
    elif immobile:
        # the immobile population owns exactly the exact-zero jumps, soft and hard alike; the moving ones split the rest
        won = set(np.unique(np.argmax(posterior, axis=1)))
        best_slow = float(posterior[:, order[0]].max())
        hard_possible = len(won) == n_pop
        hard_rule = ("exact-zero jumps against the rest" if n_pop == 2
                     else "exact-zero jumps, then " + ("a threshold on jump length" if n_pop == 3
                                                       else "bands of jump length"))
    elif n_pop == 2:
        slow = int(order[0])
        a = fractions[slow] / means[slow]
        b = fractions[1 - slow] / means[1 - slow]
        best_slow = float(a / (a + b))
        hard_possible = best_slow > 0.5
        hard_rule = "threshold on jump length"
    else:
        won = set(np.unique(np.argmax(posterior, axis=1)))
        best_slow = float(posterior[:, order[0]].max())
        hard_possible = len(won) == n_pop
        hard_rule = "bands of jump length"
    # Which populations hard assignment can never draw -- named so an empty panel is
    # explained, never mistaken for an absence of jumps.
    winners = set(np.unique(np.argmax(posterior, axis=1)).tolist())
    out.update({
        "assignment": assignment,
        "n_pop": n_pop,
        "D": D,
        "fractions": fractions,
        "max_posterior_slowest": best_slow,
        "hard_assignment_possible": hard_possible,
        "hard_rule": hard_rule,
        "never_most_likely": [labels[i] for i in range(n_pop) if i not in winners],
    })
    if weighting == "track":
        out["weight_unit"] = "weighted tracks" if assignment == "soft" else "tracks"
    else:
        out["weight_unit"] = "weighted steps" if assignment == "soft" else "steps"
    return out


#: A MANUAL SPLIT's map kinds, one per range by ascending value (`maps_from_ranges`, `spatialappd`; 1 Nov 2026). The
#: panel's TITLE is its range ("0-150 nm", "0.1-1 um2/s", `spatial_map_view.range_title`); the kind only keys it.
RANGE_KINDS = ("range1", "range2", "range3", "range4")


def check_range_values(ranges, *, unit: str) -> tuple:
    """Up to four ``(low, high)`` ranges, ascending, each low < high, none below 0 and none overlapping (every jump or
    track goes to one map at most), as a tuple of float pairs. Raises `ValueError` saying which rule a range breaks;
    ``unit`` names the values in the message."""
    out = []
    for item in ranges or ():
        lo, hi = (float(v) for v in item)
        if not (np.isfinite(lo) and np.isfinite(hi)) or lo < 0.0 or hi <= lo:
            raise ValueError(f"a range runs from a lower to a higher value, neither below 0; got {lo:g}-{hi:g} {unit}")
        out.append((lo, hi))
    if not 1 <= len(out) <= len(RANGE_KINDS):
        raise ValueError(f"a manual split takes 1 to {len(RANGE_KINDS)} ranges, got {len(out)}")
    out.sort()
    for (a_lo, a_hi), (b_lo, b_hi) in zip(out, out[1:]):
        if b_lo < a_hi:
            raise ValueError(f"the ranges {a_lo:g}-{a_hi:g} and {b_lo:g}-{b_hi:g} {unit} overlap: every jump or track "
                             f"goes to one map, so a range starts where the one before it ends or later")
    return tuple(out)


def maps_from_ranges(projected: pd.DataFrame, ranges_um, cfg: SpatialSQDConfig, *,
                     step_position: str = "A_start", mirrored: bool = False, meshes=None,
                     lag_frames: int = 1, weighting: str = "steps",
                     with_steps: bool = False, all_jumps: bool = False) -> dict[str, object]:
    """Jump-density grids for jump-LENGTH ranges set by hand: one map per range, every jump in the range its length
    falls in (asked 1 Nov 2026: "manually set ... the jump distance for SQD maps so i can manually bin the jumps").
    Nothing is fitted, so no posterior and no cliff: a range is a hard bin, ``[low, high)`` in µm; a jump in no range
    is drawn nowhere and counted (``n_outside_ranges``). The same jumps, positions and binning as `maps_from_projected`
    (`_jump_maps`), one lag at a time: lags 1-4 together divide each jump by its lag, which no length range can read.
    `all_jumps=True` adds the map of every jump, in a range or not (`ALL_JUMPS_LABEL`), after the ranges.
    """
    from .observables import displacements_at

    ranges_um = check_range_values(ranges_um, unit="um")
    if lag_frames == "1-4":
        raise ValueError("A manual split cuts jump lengths at one time lag. "
                         "Pick one lag instead of 1-4 together.")
    steps = displacements_at(projected, lag_frames)
    missing = [c for c in ("l_norm", "d_norm") if c not in steps.columns]
    if missing:
        raise ValueError(f"the jumps lack {missing}; project the table first")
    labelled = steps.copy()
    labelled.attrs.update(projected.attrs)
    labelled.attrs["label_source"] = "ranges"
    length = np.sqrt(labelled["sq_disp_um2"].to_numpy(float))
    weights = [((length >= lo) & (length < hi)).astype(float) for lo, hi in ranges_um]
    labels = RANGE_KINDS[:len(ranges_um)]
    in_any = np.sum(weights, axis=0) > 0
    if all_jumps:
        weights, labels = [*weights, np.ones(len(labelled))], (*labels, ALL_JUMPS_LABEL)
    out = _jump_maps(projected, labelled, weights, labels, cfg, step_position=step_position,
                     mirrored=mirrored, meshes=meshes, lag_frames=lag_frames, weighting=weighting,
                     with_steps=with_steps)
    inside = np.isfinite(labelled["d_norm"].to_numpy(float))
    inside &= np.abs(labelled["d_norm"].to_numpy(float)) <= 1.0
    out.update({
        "assignment": "hard",
        "n_pop": len(ranges_um),
        "D": np.zeros(0),
        "fractions": np.zeros(0),
        "max_posterior_slowest": 1.0,
        "hard_assignment_possible": True,
        "hard_rule": "jump-length ranges set by hand",
        "never_most_likely": [],
        "ranges_um": [list(r) for r in ranges_um],
        "n_outside_ranges": int((inside & ~in_any).sum()),
        "weight_unit": "tracks" if weighting == "track" else "steps",
    })
    return out


def _jump_maps(projected: pd.DataFrame, labelled: pd.DataFrame, weights, labels, cfg, *,
               step_position: str, mirrored: bool, meshes, lag_frames, weighting: str,
               with_steps: bool) -> dict[str, object]:
    """The binning half of every jump map: ``weights[i]`` (one per row of ``labelled``) of each population ``labels[i]``
    placed in the cell frame and binned -- the posteriors of a fit (`maps_from_projected`) or the 0/1 of a range set by
    hand (`maps_from_ranges`). One function, so a manual map and a fitted one are binned the same way.

    `weighting="track"` multiplies every step's weight by `track_weights` (each track counts once).
    `step_grids`/`step_weights` are always the unmultiplied step counts -- what a minimum-per-bin mask must read,
    since it is about how much evidence a bin holds -- and are the SAME objects as `grids`/`weights` under the
    default `"steps"`. `with_steps=True` adds `"steps"`, the per-jump record of which track went into which bin with
    what drawn weight (`step_block`).
    """
    if step_position != "A_start":
        if meshes is None:
            raise ValueError(
                f"step_position={step_position!r} re-projects each step's midpoint or end "
                f"through the cell centreline, which needs the meshes. Pass meshes=, or "
                f"use 'A_start', which the projected table already carries.")
        # each jump's own lag (lags 1-4 together hold four), so its end is the right localisation
        step_lags = (labelled["lag_frames"].to_numpy() if labelled.attrs.get("joint_lags") else lag_frames)
        coords = _step_coordinates(projected, labelled, meshes, step_lags)[step_position]
        l_norm = coords["l_norm"].to_numpy(float)
        d_norm = coords["d_norm"].to_numpy(float)
    else:
        l_norm = labelled["l_norm"].to_numpy(float)
        d_norm = labelled["d_norm"].to_numpy(float)

    xe, ye = cfg.edges()
    keep = np.isfinite(d_norm) & (np.abs(d_norm) <= 1.0) & np.isfinite(l_norm)
    cells = (row_cell_keys(labelled, projected)[keep]
             if getattr(cfg, "long_axis", "normalised") != "normalised" else None)
    x = long_axis_x(l_norm[keep], cfg, cells=cells,
                    geometry=projected.attrs.get("geometry"))
    y = d_norm[keep] * local_half_width(x, cfg)

    per_track = track_weights(labelled) if weighting == "track" else None
    tracks, cells = contributor_labels(labelled, projected)
    tracks, cells = tracks[keep], cells[keep]
    population_weights = weights
    grids, weights, step_grids, step_weights, drawn_by_label = {}, {}, {}, {}, {}
    neff_track, neff_cell = {}, {}
    for index in range(len(labels)):
        weight = np.asarray(population_weights[index], dtype=float)
        drawn_by_label[labels[index]] = (
            weight[keep] if per_track is None else (weight * per_track)[keep])
        counts = np.histogram2d(x, y, bins=[xe, ye], weights=weight[keep])[0].T
        if mirrored:
            counts = fold_counts(counts)
        neff_track[labels[index]], neff_cell[labels[index]] = contributor_neff_grids(
            x, y, weight[keep], tracks, cells, xe, ye, mirrored=mirrored)
        step_grids[labels[index]] = counts
        step_weights[labels[index]] = float(weight[keep].sum())
        if per_track is None:
            grids[labels[index]] = counts
            weights[labels[index]] = step_weights[labels[index]]
        else:
            drawn = (weight * per_track)[keep]
            tcounts = np.histogram2d(x, y, bins=[xe, ye], weights=drawn)[0].T
            if mirrored:
                tcounts = fold_counts(tcounts)
            grids[labels[index]] = tcounts
            weights[labels[index]] = float(drawn.sum())

    out = {
        "grids": grids,
        "weights": weights,
        "step_grids": step_grids,
        "step_weights": step_weights,
        # per-bin n_eff of each population's plain (unweighted-by-track) jumps
        "neff_track": neff_track,
        "neff_cell": neff_cell,
        "weighting": weighting,
        "labels": labels,
        "step_position": step_position,
        "mirrored": mirrored,
    }
    # where each population sits along and across the cell, from the rows its map was binned from
    out["profiles"] = axis_profiles(x, y, drawn_by_label, cfg)
    if with_steps:
        out["steps"] = step_block(labelled.iloc[np.flatnonzero(keep)], x, y,
                                  drawn_by_label, cfg)
    return out


def fit_at_fixed_D(observable, D, *, weighting: str = "step", calibration=None):
    """An SQD fit with the populations' D HELD at ``D`` (a section's fit, `map_fit_source`) and only the fractions
    fitted, on ``observable`` (`observables.displacements_at`'s table) at ``weighting`` ("step": every jump once,
    "track": every track once): `aggregate._fit_fractions`, the EM the SQD section fits each replicate's fractions
    with at a shared D, so the map's fractions are the section's estimator on the map's own jumps. A D of 0 is the
    immobile population of exact-zero jumps, counted as there. Returned as an `sqd.SQDFit` (``method`` "fixed_D",
    its BIC counting only the fractions as free) so every map function reads it as any fit.

    ``calibration`` (a `precision.Calibration`; 6 Nov 2026): the section's fit was PER SPOT, so the fractions are
    fitted with each jump's own noise (the steps lacking a spot's precision left out), and the fit carries the
    calibration for every map that labels its steps (`maps.per_spot_steps`)."""
    from .aggregate import _fit_fractions
    from .sqd import SQDFit, _effective_n, _extract, _log_likelihood

    noise, left_out = None, 0
    if calibration is not None:
        from .precision import PrecisionError, step_noise, usable_steps

        try:
            ok = usable_steps(observable)
            left_out = int((~ok).sum())
            if left_out:
                kept = observable.loc[ok]
                kept.attrs = dict(observable.attrs)
                observable = kept
            noise = step_noise(observable, calibration)
        except PrecisionError as exc:
            raise ValueError(str(exc)) from exc
    values, tau, weights = _extract(observable, None, weighting)
    D = np.sort(np.asarray(D, dtype=float).ravel())
    if not len(D) or not np.all(np.isfinite(D)) or np.any(D < 0):
        raise ValueError(f"fixed D must be finite and not negative, got {D.tolist()}")
    fractions = np.asarray(_fit_fractions(values, D, tau, sample_weights=weights, step_noise=noise), dtype=float)
    log_likelihood = _log_likelihood(values, D, fractions, tau, weights, step_noise=noise)
    n_effective = _effective_n(values, weights)
    bic = (len(D) - 1) * np.log(n_effective) - 2.0 * log_likelihood
    joint = observable.attrs.get("joint_lags") if isinstance(observable, pd.DataFrame) else None
    return SQDFit(n_pop=len(D), D=D, fractions=fractions, tau_lag_s=float(tau), n_observations=len(values),
                  method="fixed_D", log_likelihood=float(log_likelihood), bic=float(bic), starts_strategy="fixed_D",
                  weighting=weighting, n_starts=0, n_converged_starts=0,
                  joint_lags=tuple(joint) if joint else None,
                  per_spot_precision=noise is not None,
                  precision_calibration=(None if calibration is None
                                         else (float(calibration.c_nm), float(calibration.p))),
                  precision_left_out=left_out)


def spatial_sqd_map(projected: pd.DataFrame, *, n_pop: int, assignment: str = "soft",
                    step_position: str = "A_start", mirrored: bool = False,
                    bin_long_nm: float, bin_short_nm: float, smooth_bins: float,
                    cell_length_um: float, cell_width_um: float,
                    method: str = "ecdf_lsq", weighting: str = "track",
                    meshes=None, long_axis: str = "normalised",
                    lag_frames=1, ranges_um=None, all_jumps: bool = False,
                    fixed_D=None, calibration=None) -> dict[str, object]:
    """The catalogue target: one projected table in, jump-density grids out.

    `maps_from_projected` needs a FITTED object as its second argument, and no input kind
    describes that - a catalogue entry pointing at it would resolve and then raise, which
    is exactly what `INPUT_FIRST_ARG` exists to catch. So the panel calls this instead: it
    takes only what a panel can supply (a table and scalar controls), fits internally, and
    returns the same dict.

    The bin is given in nanometres because that is the unit the choice is made in - it is
    compared against the camera pixel (106 nm here) and the localisation precision (30-76,
    unmeasured). Bin COUNTS are derived from it and rounded to even, since folding pairs
    each bin with its reflection.

    The fit is returned as `fit` so a caller can draw further maps (e.g. per
    replicate day) under the SAME fit instead of refitting.

    ``fixed_D`` (µm²/s, one per population; 4 Nov 2026, `map_fit_source`): the populations' D HELD, a section's fit's,
    and only the fractions fitted on this table, at ``weighting`` (`fit_at_fixed_D`); ``n_pop`` must be their count.
    ``calibration`` (6 Nov 2026): a per-spot section fit's, every fit and label of the map with each jump's own noise.
    """
    from .observables import JOINT, displacements_at
    from .sqd import fit_sqd

    def even(value: float) -> int:
        count = int(round(value))
        return max(2, count + (count % 2))

    lag = lag_frames if lag_frames == JOINT else int(lag_frames)
    cfg = SpatialSQDConfig(
        cell_length_um=cell_length_um, cell_width_um=cell_width_um,
        n_bins_long=even(cell_length_um * 1000.0 / bin_long_nm),
        n_bins_short=even(cell_width_um * 1000.0 / bin_short_nm),
        smooth_bins=smooth_bins, pole_margin=0.18, n_profile_bins=60,
        n_pop=n_pop, method=method, weighting=weighting, long_axis=long_axis)
    # `lag_frames`: the jumps that many frames apart, for the fit AND the labels (asked 16 Oct 2026, "Lags also
    # in sqd maps as option") -- the same lag as the SQD section's Time lag, so a map can be read beside it; "1-4"
    # is lags 1-4 together (17 Oct 2026): ONE fit to all four, every lag's jumps labelled under it and drawn.
    # `ranges_um`: a split BY HAND (`maps_from_ranges`), one map per jump-length range; nothing is fitted
    if ranges_um is not None:
        out = maps_from_ranges(projected, ranges_um, cfg, step_position=step_position, mirrored=mirrored,
                               meshes=meshes, lag_frames=lag, all_jumps=all_jumps)
        out["config"] = cfg
        out["fit"] = None
        return out
    if fixed_D is not None:
        if len(fixed_D) != int(n_pop):
            raise ValueError(f"{len(fixed_D)} fixed D values for a {n_pop}-population map")
        fit = fit_at_fixed_D(displacements_at(projected, lag), fixed_D, weighting=weighting, calibration=calibration)
    elif calibration is not None:
        fit = fit_sqd(displacements_at(projected, lag), n_pop=n_pop, method="mle", weighting=weighting,
                      precision=calibration)
    else:
        fit = fit_sqd(displacements_at(projected, lag), n_pop=n_pop, method=method, weighting=weighting)
    out = maps_from_projected(projected, fit, cfg, assignment=assignment,
                              step_position=step_position, mirrored=mirrored,
                              meshes=meshes, lag_frames=lag, all_jumps=all_jumps)
    out["config"] = cfg
    out["fit"] = fit
    return out
