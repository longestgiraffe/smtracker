"""AW5: per-step state maps as a PACKAGE function, and the tables they deposit.

What this is for. The manuscript script grew the whole pipeline inline: load a
condition, assign cells, project onto the centreline, decode each replicate day under
its already-fitted HMM, bin, and turn the bins into a table. A GUI button cannot call a
script (ON2), and a second implementation of any of that is how two day vocabularies
came to exist (D11). So the computation lives here, returns plain arrays and tables, and
the script becomes display only.

What it is not.

* **No Qt.** Nothing here imports or touches a Qt class; the view is a later, thin
  shell over these functions.
* **No cross-condition comparison (Amendment 4).** There is deliberately no function
  here that takes two conditions. The tool can draw glucose and it can draw malate; it
  does not draw the difference. That layer stays in the firewalled figure script,
  reading deposited tables only.
* **No fitting.** The HMM is RECONSTRUCTED from already-deposited per-(group × day)
  parameters. If a day has no fitted row, this raises rather than skipping it.

The day map is required and has no default. ``day_map`` is a keyword with no default
value, because the whole D11 defect was a *default* day rule silently disagreeing with
the one that wrote the fitted table. The caller states its authority;
``grouping.day_map_from_series_blocks`` is the lab's rule and is what to pass.

Every choosable parameter is a column, not a footnote. ``StateMapConfig`` carries them
as a frozen set so they travel together, and ``enrichment_bin_table`` writes every one
of them onto every row. A later question is answered by filtering the deposit, not by
re-running with terabytes of raw microscopy to hand.

N in and N out are separate columns at every filtering step (AX5). No delta stands
alone; no percentage stands alone. Two mechanisms that share a symptom are two columns.

Invariants
----------
- Nothing is refitted; ``per_day_fits`` is read, never derived.
- Every day present in the data must have a fitted row, and vice versa: otherwise the
  map would be built from a subset of days and render as a normal panel (D11).
- A cell that loses every localisation must have a NAMED mechanism.
- Coordinates come from ``cellcoords``; this module adds no geometry.
- The undersampling verdict is REPORTED, never acted on here: what to do with a sparse
  panel (draw and mark it, or withhold it) is a display decision and lives with the
  display.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .canonical import to_canonical
from .cellcoords import project_tracks
from .cellmesh import assign_cells, read_project_cell_meshes
from .grouping import UTRACK_GLOB, discover_utrack_project, load_utrack_project
from .hmm import HMMFit
from .maps import fold_map, labels_from_hmm
from .render import enrichment_grid, map_grid, robust_color_limits
from .utrack_meta import acquisition_params_from_project

__all__ = [
    "StateMapError",
    "StateMapConfig",
    "StateMapResult",
    "decode_state_map",
    "state_grids",
    "enrichment_bin_table",
    "shell_additive_pp_on_map",
    "undersampling_verdict",
    "write_state_map_tables",
]


class StateMapError(RuntimeError):
    """Raised when a state map cannot be built honestly."""


@dataclass(frozen=True)
class StateMapConfig:
    """The choices that decide what a state map contains. **None has a default.**

    A default here would make the choice invisible at the call site, which is the
    failure mode the whole robustness inventory exists to prevent. They are frozen and
    passed as a unit so they can be recorded in provenance together: a map quoted
    without them is not reproducible.

    ``bins_l`` / ``bins_d``
        Grid shape. The PHYSICAL bin size in µm follows from these and the cell
        geometry, and is reported by the grid, never assumed.
    ``sigma_um``
        Localisation precision, which sets the smoothing bandwidth. **Unmeasured for
        this project** (30–76 nm standing range): it is an input, not a measurement.
    ``min_posterior``
        Posterior floor for keeping a decoded step. Raising it trades coverage for
        confidence; the fraction kept is reported.
    ``n_min``
        Minimum count for a bin to be defined. ``None`` uses the renderer's own
        per-kind default. It counts independent visits, not localisations.
    ``fold``
        Apply the cell's symmetries before binning. Which pole is ``l = 0`` and which
        side is ``d > 0`` are arbitrary, so an unfolded asymmetry is almost always
        noise, but folding is a pooling operation and is stated, not assumed.
    ``shell_d_norm``
        The shell cut on the map's own folded ``|d_norm|``. NOT the pre-registered
        spherocylinder ``r_norm`` statistic: a different pipeline, a different
        coordinate, a different functional. They are never reconciled.
    ``n_panel_min`` / ``masked_fraction_max``
        The two limbs of the undersampling verdict. The spec specifies these in kind
        but not in number, so the caller states them.
    ``step_convention``
        Which end of a step carries its position. ``"start"`` is the shipped,
        conservative one; the midpoint convention reads a larger contrast.
    """

    bins_l: int
    bins_d: int
    sigma_um: float
    min_posterior: float
    n_min: int | None
    fold: bool
    shell_d_norm: float
    n_panel_min: int
    masked_fraction_max: float
    step_convention: str

    def __post_init__(self) -> None:
        if self.bins_l < 1 or self.bins_d < 1:
            raise StateMapError(f"bins must be >= 1, got {self.bins_l}x{self.bins_d}")
        if self.sigma_um <= 0:
            raise StateMapError(
                f"sigma_um must be > 0, got {self.sigma_um}: the smoothing scale is "
                "tied to the localisation precision and must not be guessed")
        if not 0.0 <= self.min_posterior <= 1.0:
            raise StateMapError(f"min_posterior must be in [0, 1], got "
                                f"{self.min_posterior}")
        if not 0.0 <= self.shell_d_norm <= 1.0:
            raise StateMapError(f"shell_d_norm must be in [0, 1], got "
                                f"{self.shell_d_norm}")
        if not 0.0 < self.masked_fraction_max <= 1.0:
            raise StateMapError(f"masked_fraction_max must be in (0, 1], got "
                                f"{self.masked_fraction_max}")
        if self.n_panel_min < 0:
            raise StateMapError(f"n_panel_min must be >= 0, got {self.n_panel_min}")
        if self.step_convention not in ("start", "midpoint"):
            raise StateMapError(
                f"step_convention must be 'start' or 'midpoint', got "
                f"{self.step_convention!r}. 'start' is what hmm.decode_states does; "
                "nothing else is implemented, so anything else would be a silent lie "
                "about which end of the step carries the position.")

    def as_columns(self) -> dict:
        """The config as deposit columns. Every choice, spelled out."""
        out = {f"cfg_{k}": v for k, v in asdict(self).items()}
        out["cfg_n_min_provenance"] = "unknown"      # it has never been derived
        out["cfg_shell_statistic"] = (
            "shell additive on the map's own folded |d_norm|: NOT the pre-registered "
            "spherocylinder r_norm statistic; different functionals, never reconciled")
        return out


@dataclass(frozen=True)
class StateMapResult:
    """Decoded steps for one (enzyme × condition), plus how much was lost getting there.

    ``labelled`` is UNFOLDED: folding is a display/binning decision and the caller may
    want the scatter. ``folded`` applies the symmetries when the config asks for it.
    """

    enzyme: str
    condition: str
    labelled: pd.DataFrame
    day_spans: dict
    cell_length_um: float
    cell_half_width_um: float
    loss: dict
    config: StateMapConfig

    @property
    def folded(self) -> pd.DataFrame:
        if not self.config.fold:
            return self.labelled
        return fold_map(self.labelled)


def _reconstruct_fit(row, tau_s: float) -> HMMFit:
    """Rebuild a fitted 2-state model from its deposited parameters. No fitting."""
    D = np.array([float(row.D_lo), float(row.D_hi)])
    a01, a10 = float(row.k_lo_to_hi) * tau_s, float(row.k_hi_to_lo) * tau_s
    A = np.array([[1.0 - a01, a01], [a10, 1.0 - a10]])
    stationary = np.array([a10 / (a01 + a10), a01 / (a01 + a10)])
    occ = float(row.occ_lowD)
    return HMMFit(n_states=2, D=D, transition_matrix=A,
                  occupancy=np.array([occ, 1.0 - occ]), stationary=stationary,
                  tau_s=tau_s, n_steps=0, n_sequences=0,
                  log_likelihood=float("nan"), bic=float("nan"),
                  localization_sigma_um=0.0)


def decode_state_map(
    condition_root,
    *,
    enzyme: str,
    condition: str,
    per_day_fits: pd.DataFrame,
    config: StateMapConfig,
    day_map,
) -> StateMapResult:
    """One condition of a u-track project → per-STEP state labels in cell coordinates.

    ``per_day_fits`` must carry ``enzyme, condition, day, D_lo, D_hi, k_lo_to_hi,
    k_hi_to_lo, occ_lowD``. Nothing is refitted.

    ``day_map`` is REQUIRED and has no default: see the module docstring.
    """
    if not enzyme or not str(enzyme).strip():
        raise StateMapError(
            "enzyme must be a non-empty label: it is a grouping key, and a constant "
            "there silently merges different proteins")
    if day_map is None:
        raise StateMapError(
            "day_map is required and has no default: the default series->replicate rule can "
            "disagree with the one that wrote per_day_fits and drop replicates silently. "
            "Pass grouping.day_map_from_series_blocks(movies)[0].")

    root = Path(condition_root)
    meshes = read_project_cell_meshes(root, allow_unsegmented=True)
    movies = [str(m) for m, _ in discover_utrack_project(str(root), pattern=UTRACK_GLOB)]
    uncovered = [m for m in movies if m not in day_map]
    if uncovered:
        raise StateMapError(
            f"{enzyme}/{condition}: day_map does not cover {uncovered[:5]} "
            f"({len(uncovered)} movie(s)). A partial replicate map would drop them silently.")

    raw = load_utrack_project(root, condition=condition, day=dict(day_map))
    assigned = assign_cells(raw, meshes, drop_unassigned=True, require_all_inside=True)
    params = acquisition_params_from_project(root)
    canon = to_canonical(assigned, params)
    canon = canon[canon["cell"].notna()]

    # --- projection, and the TWO losses that share one symptom (AX3) ---------------
    # A NaN cell coordinate has two causes and they must not be reported as one number:
    #   (1) a RIBLESS cell, an outline with no mesh ribs, so no centreline exists and
    #       the whole cell cannot be projected;
    #   (2) an undefined coordinate inside an otherwise-projected cell, the local
    #       half-width is zero there, so d_norm has no value.
    # project_tracks(require_centerline=False) does NOT drop the rows it skips; it
    # leaves NaN coordinates, so counting unique cells on its output reports zero loss
    # however many cells were skipped (D10). The rows are dropped explicitly.
    ribless = {f"{movie}:{c.cell_id}" for movie, cs in meshes.items() for c in cs
               if not c.has_centerline}
    # THIRD MECHANISM, found on a lab project (one cell of one movie): a cell that HAS ribs but
    # whose rib coordinates are not finite. Oufti wrote inf into the mesh, giving an
    # infinite centreline length and an infinite half-width, so every localisation in
    # that cell projects to NaN and the whole cell drops. Distinct from a ribless cell
    # (no mesh at all) and from an undefined coordinate (a zero local half-width at one
    # point). Named and counted rather than absorbed into either.
    nonfinite = set()
    for movie, cs in meshes.items():
        for c in cs:
            if c.has_centerline and c.mesh is not None:
                if not np.isfinite(np.asarray(c.mesh, dtype=float)).all():
                    nonfinite.add(f"{movie}:{c.cell_id}")
    n_cells_in, n_loc_in = int(canon["cell"].nunique()), int(len(canon))
    projected_all = project_tracks(canon, meshes, require_centerline=False)
    defined = projected_all["l_norm"].notna() & projected_all["d_norm"].notna()
    projected, lost = projected_all[defined], projected_all[~defined]
    if len(projected) + len(lost) != n_loc_in:
        raise StateMapError(
            f"{enzyme}/{condition}: localisation reconciliation failed "
            f"({len(projected)} + {len(lost)} != {n_loc_in})")
    unexplained = (set(canon["cell"].astype(str))
                   - set(projected["cell"].astype(str))) - ribless - nonfinite
    if unexplained:
        raise StateMapError(
            f"{enzyme}/{condition}: {len(unexplained)} cell(s) lost every localisation "
            f"with no stated cause (not ribless): {sorted(unexplained)[:5]}. A whole "
            "cell vanishing needs a named mechanism, not a delta.")
    lost_ribless = int(lost["cell"].astype(str).isin(ribless).sum())
    lost_nonfinite = int(lost["cell"].astype(str).isin(nonfinite).sum())
    loss = {
        "cells_with_an_outline": int(sum(len(cs) for cs in meshes.values())),
        "cells_without_mesh_ribs": len(ribless),
        "cells_with_a_non_finite_mesh": len(nonfinite),
        "cells_assigned_in": n_cells_in,
        "cells_with_coordinates_out": int(projected["cell"].nunique()) if len(projected) else 0,
        "localizations_assigned_in": n_loc_in,
        "localizations_with_coordinates_out": int(len(projected)),
        "localizations_lost_to_ribless_cell": lost_ribless,
        "localizations_lost_to_undefined_coordinate":
            int(len(lost)) - lost_ribless - lost_nonfinite,
        "localizations_lost_to_non_finite_mesh": int(
            lost["cell"].astype(str).isin(nonfinite).sum()),
        "ribless_rule": "INSTRUCTED, not derived",
        "non_finite_mesh_rule": "derived (Oufti wrote inf/NaN rib coordinates)",
        "undefined_coordinate_rule": "derived (zero local half-width)",
    }

    # --- the two day vocabularies must be identical, or this stops (D11) -----------
    fits = per_day_fits[(per_day_fits["enzyme"] == enzyme)
                        & (per_day_fits["condition"] == condition)]
    have = {int(d) for d in projected["day"].unique()}
    fitted = {int(d) for d in fits["day"]}
    if have != fitted:
        raise StateMapError(
            f"{enzyme}/{condition}: the data carry replicates {sorted(have)} but the fitted "
            f"table carries {sorted(fitted)}. These are two different replicate numberings, "
            "and joining them would silently drop the replicates that do not match. Both "
            "must come from the same authority.")

    tau_s = float(canon["tau_s"].iloc[0])
    frames = []
    for day, group in projected.groupby("day"):
        row = fits[fits["day"] == int(day)].iloc[0]
        try:
            labelled = labels_from_hmm(group, _reconstruct_fit(row, tau_s),
                                       min_posterior=config.min_posterior)
        except Exception as exc:            # never silent: a lost day is a result
            raise StateMapError(
                f"{enzyme}/{condition} replicate {int(day)}: decoding produced no labelled "
                f"steps ({exc}). Refusing to build a map from a subset of replicates without "
                "saying so.") from exc
        frames.append(labelled.assign(day=int(day)))

    # Cell geometry must be captured BEFORE attrs are dropped: the renderer needs a bin
    # size in µm and will not invent one. pandas compares .attrs elementwise on concat
    # and raises on the numpy arrays inside CellGeometry, and no map reads them.
    geometry = {}
    for frame in frames:
        geometry.update(frame.attrs.get("geometry", {}) or {})
    lengths = [float(g.length_um) for g in geometry.values()]
    halves = [float(g.max_width_um) / 2.0 for g in geometry.values()]
    if not lengths:
        raise StateMapError(f"{enzyme}/{condition}: no cell geometry survived, so a "
                            "bin size in µm cannot be stated")
    for frame in frames:
        frame.attrs = {}

    _, spans = _day_spans(movies, day_map)
    return StateMapResult(
        enzyme=enzyme, condition=condition,
        labelled=pd.concat(frames, ignore_index=True),
        day_spans=spans,
        cell_length_um=float(np.median(lengths)),
        cell_half_width_um=float(np.median(halves)),
        loss=loss, config=config)


def _day_spans(movies, day_map):
    """``{day: "seriesNNN-NNN"}`` for the mapping actually used. Recorded, not assumed."""
    from .grouping import series_number

    spans: dict[int, tuple[int, int]] = {}
    for movie in movies:
        day = int(day_map[movie])
        n = series_number(movie)
        lo, hi = spans.get(day, (10 ** 9, 0))
        spans[day] = (min(lo, n), max(hi, n))
    return spans, {d: f"series{lo:03d}-{hi:03d}" for d, (lo, hi) in sorted(spans.items())}


def state_grids(result: StateMapResult, *, states=(1, 2)) -> dict:
    """``{state: MapGrid}`` plus ``{"enrichment": MapGrid}``. Plain arrays, no display.

    The grids carry their own bin size, bandwidth, mask count and caption; the caller
    decides colours and whether to draw. Colour LIMITS are not set here: they must be
    computed over the whole panel set before anything is rendered, which only the
    caller can do.
    """
    table = result.folded
    common = dict(bins=(result.config.bins_l, result.config.bins_d),
                  sigma_um=result.config.sigma_um,
                  n_min=result.config.n_min,
                  cell_length_um=result.cell_length_um,
                  cell_half_width_um=result.cell_half_width_um)
    grids = {state: map_grid(table, state=state, **common) for state in states}
    grids["enrichment"] = enrichment_grid(table, state=1, **common)
    return grids


def undersampling_verdict(grid, config: StateMapConfig, *, masked_max=None):
    """``(caveat, rule, value, threshold)``: REPORTED, never acted on here.

    What to do with a sparse panel: draw it and mark it, or withhold it: is a display
    decision and belongs with the display. This function only says which limb fired and
    against what, so that the verdict is identical however the caller renders it.
    """
    limit = config.masked_fraction_max if masked_max is None else float(masked_max)
    n = int(grid.n_localizations)
    total = int(np.asarray(grid.values).size)
    fraction = (int(grid.n_bins_masked) / total) if total else 0.0
    if n < config.n_panel_min:
        return (f"n = {n}: sparse", "total_N", float(n), float(config.n_panel_min))
    if fraction > limit:
        return (f"n = {n} · {100 * fraction:.0f} % of bins under n_min: sparse",
                "masked_fraction", float(fraction), limit)
    return (None, "none", float(fraction), limit)


def shell_additive_pp_on_map(folded: pd.DataFrame, *, shell_d_norm: float) -> dict:
    """Low-D share in the shell minus low-D share overall, in percentage points.

    Reported under BOTH weightings, because a difference that appears under one and not
    the other is a weighting artefact rather than a property of the map: step-weighted
    (every decoded step counts once) and track-weighted (every track counts once,
    ``w = 1/its number of steps``), so a long track cannot speak for a hundred short
    ones. Every ratio carries its denominator count.

    NOT the pre-registered spherocylinder ``r_norm`` shell statistic, see
    ``StateMapConfig.shell_d_norm``.
    """
    for column in ("d_norm", "state", "uid"):
        if column not in folded.columns:
            raise StateMapError(f"missing '{column}'; label and fold the table first")
    d = folded["d_norm"].to_numpy(dtype=float)
    low = folded["state"].to_numpy() == 1
    shell = d >= float(shell_d_norm)
    per_track = folded.groupby("uid")["state"].transform("size").to_numpy(dtype=float)
    out = {"n_steps": int(len(folded)), "n_steps_shell": int(shell.sum()),
           "n_tracks": int(folded["uid"].nunique()),
           "shell_d_norm": float(shell_d_norm)}
    for name, weight in (("step", np.ones(len(folded))), ("track", 1.0 / per_track)):
        total, total_shell = weight.sum(), weight[shell].sum()
        share_all = float((weight * low).sum() / total) if total else float("nan")
        share_shell = (float((weight[shell] * low[shell]).sum() / total_shell)
                       if total_shell else float("nan"))
        out[f"low_D_share_all_{name}"] = share_all
        out[f"low_D_share_shell_{name}"] = share_shell
        out[f"additive_pp_{name}"] = 100.0 * (share_shell - share_all)
    return out


def enrichment_bin_table(grid, *, enzyme: str, condition: str, day, scope: str,
                         config: StateMapConfig, extra=None) -> pd.DataFrame:
    """One row per BIN, with every choosable parameter as a column.

    ``day`` is present so the table can be re-analysed on the **replicate day**, which
    is this project's analysis unit. A table that can only be pooled across days is a
    table that invites the wrong test.
    """
    values = np.asarray(grid.values, dtype=float)
    counts = np.asarray(grid.counts, dtype=float)
    n_d, n_l = values.shape
    l_centres = 0.5 * (grid.l_edges[:-1] + grid.l_edges[1:])
    d_centres = 0.5 * (grid.d_edges[:-1] + grid.d_edges[1:])
    di, li = np.meshgrid(np.arange(n_d), np.arange(n_l), indexing="ij")
    table = pd.DataFrame({
        "enzyme": enzyme, "condition": condition, "day": day, "scope": scope,
        "l_bin": li.ravel(), "d_bin": di.ravel(),
        "l_center_norm": l_centres[li.ravel()],
        "d_center_norm": d_centres[di.ravel()],
        "enrichment_low_D": values.ravel(),
        "n_localizations_bin": counts.ravel(),
        "masked": ~np.isfinite(values.ravel()),
        "n_localizations_panel": int(grid.n_localizations),
        "n_bins_masked_panel": int(grid.n_bins_masked),
        "bin_l_um": grid.bin_um[0], "bin_d_um": grid.bin_um[1],
        "bandwidth_h_um": grid.bandwidth_um,
        "n_min_effective": grid.n_min,
        "folded_effective": grid.folded, "fold_op": grid.pooling,
        "null": grid.null,
        "state_label_source": grid.label_source or "hmm posterior decoding",
        "unit_of_analysis": "replicate",
        **config.as_columns(),
    })
    for key, value in (extra or {}).items():
        table[key] = value
    return table


def write_state_map_tables(results, *, out_dir, expected_panels=None,
                           extra_columns=None) -> dict:
    """Deposit the per-bin enrichment table and the loss table. Returns their paths.

    ``expected_panels`` is asserted against the panels actually written, because the
    day-map defect surfaced as 37 panels where 40 were expected and a human had to do
    that arithmetic. Both counts travel with the table as columns.
    """
    results = list(results)
    if not results:
        raise StateMapError("no results to deposit: an empty deposit reads as "
                            "'no molecules', which is not what happened")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bin_rows, loss_rows, panels = [], [], 0
    for result in results:
        config = result.config
        folded = result.folded
        common = dict(bins=(config.bins_l, config.bins_d), sigma_um=config.sigma_um,
                      n_min=config.n_min, cell_length_um=result.cell_length_um,
                      cell_half_width_um=result.cell_half_width_um)
        pooled = enrichment_grid(folded, state=1, **common)
        bin_rows.append(enrichment_bin_table(
            pooled, enzyme=result.enzyme, condition=result.condition, day=pd.NA,
            scope="pooled over replicates", config=config,
            extra={"day_spans": str(result.day_spans)}))
        panels += 1
        for day, subset in folded.groupby("day"):
            subset = subset.copy()
            subset.attrs["folded"] = bool(config.fold)
            bin_rows.append(enrichment_bin_table(
                enrichment_grid(subset, state=1, **common),
                enzyme=result.enzyme, condition=result.condition, day=int(day),
                scope="single replicate", config=config,
                extra={"day_spans": str(result.day_spans)}))
            panels += 1
        loss_rows.append({"enzyme": result.enzyme, "condition": result.condition,
                          **result.loss, **config.as_columns()})

    bins_table = pd.concat(bin_rows, ignore_index=True)
    n_bins = results[0].config.bins_l * results[0].config.bins_d
    if expected_panels is not None and panels != int(expected_panels):
        raise StateMapError(
            f"{panels} panels deposited where {int(expected_panels)} were expected. "
            "This is the signature of a silently truncated set that renders as a "
            "normal table. Nothing is written.")
    if len(bins_table) != panels * n_bins:
        raise StateMapError(f"{len(bins_table)} rows for {panels} panels x {n_bins} "
                            "bins: the deposit does not reconcile")
    bins_table = bins_table.assign(
        panels_expected=int(expected_panels) if expected_panels is not None else panels,
        panels_actual=panels, bins_per_panel=n_bins,
        rows_actual=len(bins_table), rows_expected=panels * n_bins,
        reconciled=True)
    for key, value in (extra_columns or {}).items():
        if key in bins_table.columns:
            raise StateMapError(
                f"extra column {key!r} would overwrite one the deposit already "
                "defines: pick another name rather than shadowing it")
        bins_table[key] = value

    paths = {"bins": out_dir / "spatial_enrichment_bins.csv",
             "loss": out_dir / "spatial_state_map_loss.csv"}
    bins_table.to_csv(paths["bins"], index=False)
    pd.DataFrame(loss_rows).to_csv(paths["loss"], index=False)
    return {k: str(v) for k, v in paths.items()}
