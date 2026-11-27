"""Reading + validating the cell geometry table for ``smtracker.spatial``.

``read_cell_table`` accepts a CSV, a MATLAB ``.mat`` (v7 via ``scipy.io.loadmat``
and v7.3/HDF5 via ``mat73`` or ``h5py``), or an already-loaded DataFrame, and
returns ONE canonical DataFrame with exactly the geometry columns the rest of the
module needs. It fails LOUD, listing the offending rows, rather than silently
dropping a bad cell: a wrong baseline downstream is worse than a stop here.

Export / provenance / ``validate_schema`` live here too (added with Priority 3).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import geometry as _g
from .coords import to_cell_frame
from .geometry import LENGTH_RANGE_UM, WIDTH_RANGE_UM
from .populations import (
    APPARENT_D_FORMULA,
    APPARENT_D_QUALIFIED_NAME,
    assign_populations,
    per_track_apparent_D,
)

__all__ = [
    "CellTableError",
    "SchemaError",
    "REQUIRED_CELL_COLUMNS",
    "OPTIONAL_CELL_COLUMNS",
    "SCHEMA",
    "SOURCE_VALUES",
    "read_cell_table",
    "build_tables",
    "validate_schema",
    "export",
]


class SchemaError(ValueError):
    """Raised when a table is missing a schema-required column."""


#: The three ``source`` values every table may carry.
SOURCE_VALUES = ("observed", "null_uniform", "null_kinetic")

_IDENT = ("source", "dataset", "enzyme", "condition", "replicate", "session",
          "cell_id", "track_id")

#: The contract. ``validate_schema`` requires these columns are present (extra
#: columns, e.g. ``null_rep`` on the nulls: are allowed). Reproduced in SCHEMA.md.
SCHEMA = {
    "localisations": _IDENT + (
        "frame", "x_um", "y_um", "u_um", "v_um", "l_signed", "l_abs",
        "halfwidth_um", "r_norm", "r_norm_sigma", "d_pole_um", "d_membrane_um",
        "in_cap", "cell_length_um", "cell_width_um", "track_D_um2s",
        "track_D_offset_removed_um2s", "population", "n_loc_track", "proj_weight"),
    "steps": _IDENT + (
        "frame_from", "frame_to", "dt_s", "gap", "du_um", "dv_um", "abs_du_um",
        "abs_dv_um", "step_len_um", "dr2_um2", "u_mid", "v_mid", "l_abs_mid",
        "r_norm_mid", "halfwidth_mid_um", "proj_weight_mid", "track_D_um2s",
        "track_D_offset_removed_um2s", "population"),
    "tracks": _IDENT + (
        "n_loc", "D_app_um2s", "D_offset_removed_um2s", "population", "u_centroid",
        "v_centroid", "l_abs_centroid", "r_norm_centroid", "axial_span_um",
        "radial_span_um", "radius_gyration_um", "n_gaps"),
    "cells": ("source", "dataset", "enzyme", "condition", "replicate", "session",
              "cell_id", "length_um", "width_um", "angle_rad", "centroid_x",
              "centroid_y", "n_tracks", "n_loc", "projected_area_um2",
              "volume_um3", "length_tertile"),
}
_EPS = 1e-12


class CellTableError(ValueError):
    """Raised when a cell geometry table is unreadable or fails validation."""


#: Every cell row must carry these. ``angle`` units are the CALLER's business
#: (stated to ``to_cell_frame`` as ``angle_units``); this reader never guesses them.
REQUIRED_CELL_COLUMNS = (
    "cell_id", "centroid_x", "centroid_y", "angle", "length_um", "width_um",
)
#: Drawing-only; never used for coordinates.
OPTIONAL_CELL_COLUMNS = ("outline_x", "outline_y")


# --------------------------------------------------------------------------- #
# .mat readers (v7 via scipy, v7.3/HDF5 via mat73 or h5py)
# --------------------------------------------------------------------------- #


def _mat_to_flat_dict(path: Path) -> dict:
    """Load a ``.mat`` into ``{name: value}``, transparently across v7 and v7.3.

    Returns the raw variable/struct-field map; column extraction happens above.
    """
    try:
        import scipy.io as sio

        raw = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
        return {k: v for k, v in raw.items() if not k.startswith("__")}
    except NotImplementedError:
        pass                                              # v7.3 → HDF5, fall through
    try:
        import mat73

        return dict(mat73.loadmat(str(path)))
    except ImportError:
        pass
    try:
        import h5py

        out = {}
        with h5py.File(str(path), "r") as f:
            for key in f.keys():
                out[key] = np.asarray(f[key]).squeeze()
        return out
    except ImportError as exc:                            # pragma: no cover - env
        raise CellTableError(
            f"{path} is a v7.3 (HDF5) .mat but neither mat73 nor h5py is installed"
        ) from exc


def _column_from_container(container, name):
    """Pull a 1-D array for ``name`` from a loaded dict / MATLAB struct object."""
    if isinstance(container, dict):
        if name in container:
            return np.atleast_1d(np.asarray(container[name]).squeeze())
        return None
    val = getattr(container, name, None)                  # scipy struct object
    return None if val is None else np.atleast_1d(np.asarray(val).squeeze())


def _mat_to_frame(path: Path) -> pd.DataFrame:
    """Build a cell DataFrame from a ``.mat``.

    Accepts either top-level variables named like the columns, or a SINGLE struct
    whose fields are named like the columns. Fails loud, listing what it found.
    """
    flat = _mat_to_flat_dict(path)
    wanted = list(REQUIRED_CELL_COLUMNS) + list(OPTIONAL_CELL_COLUMNS)

    # 1) top-level variables named like the columns
    cols = {c: _column_from_container(flat, c) for c in wanted}
    if all(cols[c] is not None for c in REQUIRED_CELL_COLUMNS):
        return _assemble_mat_frame(cols)

    # 2) a single struct whose fields are the columns
    for key, val in flat.items():
        if isinstance(val, np.ndarray) and val.dtype != object:
            continue
        cand = {c: _column_from_container(val, c) for c in wanted}
        if all(cand[c] is not None for c in REQUIRED_CELL_COLUMNS):
            return _assemble_mat_frame(cand)

    raise CellTableError(
        f"{path}: could not find the required cell columns "
        f"{REQUIRED_CELL_COLUMNS} as top-level variables or as fields of one "
        f"struct. Found top-level keys: {sorted(flat)}"
    )


def _assemble_mat_frame(cols: dict) -> pd.DataFrame:
    frame = pd.DataFrame({c: cols[c] for c in REQUIRED_CELL_COLUMNS})
    for c in OPTIONAL_CELL_COLUMNS:                       # outlines: keep per-cell arrays
        if cols.get(c) is not None:
            frame[c] = list(cols[c])
    return frame


# --------------------------------------------------------------------------- #
# read_cell_table + validation
# --------------------------------------------------------------------------- #


def read_cell_table(source, *, known_cell_ids=None) -> pd.DataFrame:
    """Read + validate a cell geometry table into one canonical DataFrame.

    Parameters
    ----------
    source : str | pathlib.Path | pandas.DataFrame
        A ``.csv``, a ``.mat`` (v7 or v7.3/HDF5), or an already-loaded DataFrame.
    known_cell_ids : iterable, optional
        ``cell_id`` values that appear in the localisation table. When given,
        EVERY one must exist in the geometry table (a localisation whose cell is
        unknown has no coordinate frame), else the reader raises listing the
        missing ids.

    Returns
    -------
    pandas.DataFrame
        Columns ``cell_id, centroid_x, centroid_y, angle, length_um, width_um``
        (+ ``outline_x/outline_y`` if present). ``angle`` is in the CALLER's units,
        unchanged; unit conversion happens in ``to_cell_frame(angle_units=...)``.

    Raises
    ------
    CellTableError
        On a missing column, a duplicate ``cell_id``, any geometrically
        impossible / implausible row (positive, ``0.5–10 µm`` length,
        ``0.3–2 µm`` width, ``length ≥ width``), or an uncovered ``cell_id``,
        always with the offending rows listed.
    """
    if isinstance(source, pd.DataFrame):
        frame = source.copy()
    else:
        path = Path(source)
        if not path.exists():
            raise CellTableError(f"cell table not found: {path}")
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
        elif path.suffix.lower() == ".mat":
            frame = _mat_to_frame(path)
        else:
            raise CellTableError(
                f"unsupported cell-table type {path.suffix!r}; use .csv or .mat")

    missing = [c for c in REQUIRED_CELL_COLUMNS if c not in frame.columns]
    if missing:
        raise CellTableError(
            f"cell table missing required column(s) {missing}; "
            f"has {list(frame.columns)}")

    keep = list(REQUIRED_CELL_COLUMNS) + [c for c in OPTIONAL_CELL_COLUMNS
                                          if c in frame.columns]
    frame = frame[keep].reset_index(drop=True)
    for c in ("centroid_x", "centroid_y", "angle", "length_um", "width_um"):
        frame[c] = pd.to_numeric(frame[c], errors="coerce")

    _validate_cell_frame(frame, known_cell_ids=known_cell_ids)
    return frame


def _offending(frame: pd.DataFrame, mask, reason: str) -> str:
    rows = frame.loc[mask, ["cell_id", "length_um", "width_um"]]
    return f"{reason} ({int(mask.sum())} row(s)):\n{rows.to_string(index=False)}"


def _validate_cell_frame(frame: pd.DataFrame, *, known_cell_ids=None) -> None:
    problems: list[str] = []

    dup = frame["cell_id"].duplicated(keep=False)
    if dup.any():
        ids = sorted(frame.loc[dup, "cell_id"].unique().tolist())
        problems.append(f"duplicate cell_id(s): {ids}")

    nonfinite = ~np.isfinite(frame[["length_um", "width_um"]].to_numpy()).all(axis=1)
    if nonfinite.any():
        problems.append(_offending(frame, nonfinite, "non-finite length/width"))

    llo, lhi = LENGTH_RANGE_UM
    wlo, whi = WIDTH_RANGE_UM
    bad_len = ~frame["length_um"].between(llo, lhi)
    bad_wid = ~frame["width_um"].between(wlo, whi)
    if bad_len.any():
        problems.append(_offending(
            frame, bad_len, f"length_um outside {llo}–{lhi} µm"))
    if bad_wid.any():
        problems.append(_offending(
            frame, bad_wid, f"width_um outside {wlo}–{whi} µm"))
    short = frame["length_um"] < frame["width_um"]
    if short.any():
        problems.append(_offending(
            frame, short, "length_um < width_um (impossible spherocylinder)"))

    if known_cell_ids is not None:
        have = set(frame["cell_id"].tolist())
        missing = sorted(set(known_cell_ids) - have)
        if missing:
            problems.append(
                f"{len(missing)} cell_id(s) in the localisation table are absent "
                f"from the geometry table: {missing[:20]}"
                + (" …" if len(missing) > 20 else ""))

    if problems:
        raise CellTableError(
            "cell geometry table failed validation:\n\n" + "\n\n".join(problems))


# --------------------------------------------------------------------------- #
# Table builders, the four observed/null tables (same schema, different source)
# --------------------------------------------------------------------------- #

_LOC_INPUT = ("track_id", "frame", "x_um", "y_um", "cell_id", "enzyme",
              "condition", "replicate", "session")


def _midpoint_geometry(u_mid, v_mid, L, W):
    """(l_abs_mid, r_norm_mid, halfwidth_mid_um, proj_weight_mid) at a step midpoint."""
    h = _g.half_width(u_mid, L, W)
    return (
        np.abs(u_mid) / (L / 2.0),
        np.clip(np.abs(v_mid) / np.maximum(h, _EPS), 0.0, 1.0),
        h,
        _g.chord(u_mid, v_mid, L, W) / _g.mean_chord_um(L, W),
    )


def _build_tracks(coords, *, ident_cols):
    """One row per track from its localisations (µm). Centroid fields are
    descriptive data: see the caution: never build a per-track-centroid spatial
    STATISTIC from them (a fast track's centroid regresses toward midcell, an
    effect out of nothing). Step-level abs_du_um/abs_dv_um are the safe channel."""
    grp = coords.sort_values(["track_id", "frame"]).groupby("track_id", sort=False)
    rows = []
    for tid, tr in grp:
        u, v = tr["u_um"].to_numpy(), tr["v_um"].to_numpy()
        L = float(tr["cell_length_um"].iloc[0])
        W = float(tr["cell_width_um"].iloc[0])
        um, vm = float(u.mean()), float(v.mean())
        h_c = float(_g.half_width(um, L, W))
        n_gaps = int((np.diff(np.sort(tr["frame"].to_numpy())) != 1).sum())
        row = {c: tr[c].iloc[0] for c in ident_cols if c in tr}
        row.update({
            "track_id": tid,
            "n_loc": int(len(tr)),
            "u_centroid": um, "v_centroid": vm,
            "l_abs_centroid": abs(um) / (L / 2.0),
            "r_norm_centroid": float(np.clip(abs(vm) / max(h_c, _EPS), 0.0, 1.0)),
            "axial_span_um": float(u.max() - u.min()),
            "radial_span_um": float(v.max() - v.min()),
            "radius_gyration_um": float(
                np.sqrt(((u - um) ** 2 + (v - vm) ** 2).mean())),
            "n_gaps": n_gaps,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def _build_steps(coords, *, tau_s, d_offset, keep_gaps, ident_cols):
    """One row per within-track consecutive pair (µm, s). Only frame-adjacent steps
    unless ``keep_gaps``; gapped steps carry the correct dt_s and ``gap=True``."""
    df = coords.sort_values(["track_id", "frame"]).reset_index(drop=True)
    nxt = df.groupby("track_id", sort=False).shift(-1)
    valid = nxt["frame"].notna().to_numpy()
    cur, nx = df[valid], nxt[valid]
    frame_from = cur["frame"].to_numpy()
    frame_to = nx["frame"].to_numpy()
    du = nx["u_um"].to_numpy() - cur["u_um"].to_numpy()
    dv = nx["v_um"].to_numpy() - cur["v_um"].to_numpy()
    u_mid = 0.5 * (cur["u_um"].to_numpy() + nx["u_um"].to_numpy())
    v_mid = 0.5 * (cur["v_um"].to_numpy() + nx["v_um"].to_numpy())
    L, W = cur["cell_length_um"].to_numpy(), cur["cell_width_um"].to_numpy()
    l_abs_mid, r_norm_mid, hw_mid, pw_mid = _midpoint_geometry(u_mid, v_mid, L, W)

    out = pd.DataFrame({c: cur[c].to_numpy() for c in ident_cols if c in cur})
    out["frame_from"] = frame_from
    out["frame_to"] = frame_to
    out["dt_s"] = (frame_to - frame_from) * tau_s
    out["gap"] = (frame_to - frame_from) != 1
    out["du_um"], out["dv_um"] = du, dv
    out["abs_du_um"], out["abs_dv_um"] = np.abs(du), np.abs(dv)
    out["dr2_um2"] = du ** 2 + dv ** 2
    out["step_len_um"] = np.sqrt(out["dr2_um2"].to_numpy())
    out["u_mid"], out["v_mid"] = u_mid, v_mid
    out["l_abs_mid"], out["r_norm_mid"] = l_abs_mid, r_norm_mid
    out["halfwidth_mid_um"], out["proj_weight_mid"] = hw_mid, pw_mid
    out["track_D_um2s"] = cur["track_D_um2s"].to_numpy()
    out["track_D_offset_removed_um2s"] = cur["track_D_um2s"].to_numpy() - d_offset
    out["population"] = cur["population"].to_numpy()
    if not keep_gaps:
        out = out[~out["gap"].to_numpy()]
    return out.reset_index(drop=True)


def _build_cells(cells, coords_surv, *, source, dataset, angle_units, run_ident):
    """One row per cell per source, complete even for cells with n_tracks = 0."""
    from .coords import _angle_rad

    per_cell = coords_surv.groupby("cell_id")
    n_loc = per_cell.size()
    n_tracks = per_cell["track_id"].nunique()
    L, W = cells["length_um"].to_numpy(), cells["width_um"].to_numpy()
    try:
        tert = pd.qcut(cells["length_um"], 3, labels=[1, 2, 3]).astype("Int64")
    except (ValueError, IndexError):
        tert = pd.Series([pd.NA] * len(cells), index=cells.index, dtype="Int64")

    out = pd.DataFrame({
        "source": source, "dataset": dataset,
        "enzyme": run_ident.get("enzyme"), "condition": run_ident.get("condition"),
        "replicate": run_ident.get("replicate"), "session": run_ident.get("session"),
        "cell_id": cells["cell_id"].to_numpy(),
        "length_um": L, "width_um": W,
        "angle_rad": _angle_rad(cells["angle"].to_numpy(), angle_units),
        "centroid_x": cells["centroid_x"].to_numpy(),
        "centroid_y": cells["centroid_y"].to_numpy(),
        "projected_area_um2": _g.projected_area_um2(L, W),
        "volume_um3": _g.volume_um3(L, W),
        "length_tertile": tert.to_numpy(),
    })
    out["n_tracks"] = out["cell_id"].map(n_tracks).fillna(0).astype(int)
    out["n_loc"] = out["cell_id"].map(n_loc).fillna(0).astype(int)
    if len(coords_surv):
        first = coords_surv.groupby("cell_id")[
            ["enzyme", "condition", "replicate", "session"]].first()
        for c in ("enzyme", "condition", "replicate", "session"):
            mapped = out["cell_id"].map(first[c])
            out[c] = mapped.where(mapped.notna(), out[c])
    return out[list(SCHEMA["cells"])]


def build_tables(locs: pd.DataFrame, cells: pd.DataFrame, *, dataset: str,
                 sigma_um: float, tau_s: float, edges=(0.0, 0.15, np.inf),
                 min_localisations: int = 5, angle_units: str = "rad",
                 source: str = "observed", n_steps: int = 4,
                 keep_gaps: bool = False, extra_columns=None) -> dict:
    """The four schema tables (``localisations/steps/tracks/cells``) for one ``source``.

    ``sigma_um`` and ``tau_s`` are REQUIRED (µm, s). ``sigma_um`` sets
    ``r_norm_sigma`` and the offset ``d_offset = sigma_um²/tau_s`` subtracted to form
    the ``*_offset_removed_*`` columns (NOT clipped at zero). ``dataset`` is broadcast
    into the ``dataset`` column; ``extra_columns`` (e.g. ``{'null_rep': i}``) are
    broadcast onto every table. ``edges`` live on the apparent scale.
    """
    if sigma_um is None or sigma_um <= 0:
        raise SchemaError(f"sigma_um must be > 0 (required), got {sigma_um!r}")
    if tau_s is None or tau_s <= 0:
        raise SchemaError(f"tau_s must be > 0 (required), got {tau_s!r}")
    missing = [c for c in _LOC_INPUT if c not in locs.columns]
    if missing:
        raise SchemaError(f"localisation table missing {missing}")
    if source not in SOURCE_VALUES:
        raise SchemaError(f"source must be one of {SOURCE_VALUES}, got {source!r}")

    d_offset = sigma_um ** 2 / tau_s
    coords = to_cell_frame(locs, cells, angle_units=angle_units)
    coords["dataset"] = dataset

    D = per_track_apparent_D(locs, tau_s=tau_s, n_steps=n_steps)
    n_loc_track_all = locs.groupby("track_id").size()

    ident_track = ["dataset", "enzyme", "condition", "replicate", "session", "cell_id"]
    tracks = _build_tracks(coords, ident_cols=ident_track)
    tracks["source"] = source
    tracks["D_app_um2s"] = tracks["track_id"].map(D)
    tracks["D_offset_removed_um2s"] = tracks["D_app_um2s"] - d_offset

    labelled = assign_populations(tracks, edges=edges,
                                  min_localisations=min_localisations)
    surv_ids = set(labelled["track_id"])
    pop_of = labelled.set_index("track_id")["population"]

    surv = coords[coords["track_id"].isin(surv_ids)].copy()
    surv["source"] = source
    surv["r_norm_sigma"] = sigma_um / np.maximum(surv["halfwidth_um"].to_numpy(), _EPS)
    surv["track_D_um2s"] = surv["track_id"].map(D)
    surv["track_D_offset_removed_um2s"] = surv["track_D_um2s"].to_numpy() - d_offset
    surv["population"] = surv["track_id"].map(pop_of)
    surv["n_loc_track"] = surv["track_id"].map(n_loc_track_all)

    steps = _build_steps(surv, tau_s=tau_s, d_offset=d_offset,
                         keep_gaps=keep_gaps, ident_cols=_IDENT)

    run_ident = {c: (locs[c].iloc[0] if locs[c].nunique() == 1 else None)
                 for c in ("enzyme", "condition", "replicate", "session")}
    cells_tbl = _build_cells(cells, surv, source=source, dataset=dataset,
                             angle_units=angle_units, run_ident=run_ident)

    out = {
        "localisations": surv.reindex(columns=list(SCHEMA["localisations"])
                                      ).reset_index(drop=True),
        "steps": steps.reindex(columns=list(SCHEMA["steps"])).reset_index(drop=True),
        "tracks": labelled.reindex(columns=list(SCHEMA["tracks"])
                                   ).reset_index(drop=True),
        "cells": cells_tbl.reset_index(drop=True),
    }
    if extra_columns:
        for df in out.values():
            for col, val in extra_columns.items():
                df[col] = val
    counts = {"n_loc_in": int(len(locs)), "n_tracks_kept": int(len(labelled)),
              "n_loc_kept": int(len(out["localisations"])),
              "d_offset_um2s": float(d_offset)}
    for df in out.values():
        df.attrs.update(counts)
    out["_counts"] = counts
    return out


# --------------------------------------------------------------------------- #
# validate_schema: any downstream reader can call this
# --------------------------------------------------------------------------- #


def validate_schema(df: pd.DataFrame, kind: str) -> None:
    """Raise :class:`SchemaError` if ``df`` lacks a required column for ``kind``.

    ``kind`` ∈ ``localisations|steps|tracks|cells``. Extra columns (e.g.
    ``null_rep``) are allowed; ``source`` must be one of :data:`SOURCE_VALUES`.
    """
    if kind not in SCHEMA:
        raise SchemaError(f"unknown table kind {kind!r}; expected {list(SCHEMA)}")
    missing = [c for c in SCHEMA[kind] if c not in df.columns]
    if missing:
        raise SchemaError(f"{kind} table missing required column(s) {missing}")
    if "source" in df.columns and len(df):
        bad = set(pd.unique(df["source"])) - set(SOURCE_VALUES)
        if bad:
            raise SchemaError(f"{kind}: unknown source value(s) {sorted(bad)}")


# --------------------------------------------------------------------------- #
# export + provenance
# --------------------------------------------------------------------------- #

_TABLE_KINDS = ("localisations", "steps", "tracks", "cells")


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str | None:
    import subprocess

    from ..childproc import NO_WINDOW

    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"],
                             cwd=Path(__file__).resolve().parents[3],
                             capture_output=True, text=True, timeout=10, creationflags=NO_WINDOW)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _package_version() -> str:
    from importlib import metadata

    try:
        return metadata.version("smtracker")
    except Exception:
        return "unknown"


def _write_table(df: pd.DataFrame, path: Path) -> dict:
    df.to_csv(path, index=False)
    info = {"csv": path.name, "n_rows": int(len(df)), "sha256": _sha256(path)}
    try:
        import pyarrow  # noqa: F401

        pq = path.with_suffix(".parquet")
        df.to_parquet(pq, index=False)
        info["parquet"] = pq.name
    except Exception:
        info["parquet"] = None
    return info


def export(locs: pd.DataFrame, cells: pd.DataFrame, outdir, *, dataset: str,
           edges=(0.0, 0.15, np.inf), sigma_um: float, tau_s: float,
           sigma_provenance: str, sigma_basis: str = "supplied",
           angle_units: str = "rad", min_localisations: int = 5,
           nulls=("uniform", "kinetic"), null_params=None, seed: int = 0,
           input_paths=None) -> dict:
    """Write ``localisations/steps/tracks/cells`` (observed + requested nulls) + a
    ``provenance.json`` into ``outdir``. Parquet is written alongside each CSV when
    ``pyarrow`` is available. ``map_bins.csv`` is written by the map functions.

    Returns the provenance dict. No analysis script should need to ask what a run's
    parameters were: everything is on disk.

    ``null_params['kinetic']`` MUST carry the MEASURED per-condition ``k_on``/``k_off``
    (1/s); ``null_kinetic`` has no rate defaults, because a null built on another
    condition's rates encodes the wrong confinement.

    ``sigma_provenance`` is REQUIRED and has no default, one of ``theoretical`` /
    ``configured`` / ``measured`` (``AcquisitionParams.SIGMA_SOURCES``). A σ without
    its provenance is not a σ: the same number means different things depending on
    whether it came from the optics, from a lab setting, or from a stationary
    reference, and only the last is a measurement. Pass
    ``**params.sigma_stamp()``-derived values rather than assembling them by hand.
    """
    from ..canonical import AcquisitionParams

    if sigma_provenance not in AcquisitionParams.SIGMA_SOURCES:
        raise SchemaError(
            f"sigma_provenance must be one of {AcquisitionParams.SIGMA_SOURCES}, got "
            f"{sigma_provenance!r}. It has no default: a stored σ whose provenance is "
            "unknown cannot be interpreted later.")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    frames = {k: [] for k in _TABLE_KINDS}
    counts = {}
    obs = build_tables(locs, cells, dataset=dataset, sigma_um=sigma_um, tau_s=tau_s,
                       edges=edges, min_localisations=min_localisations,
                       angle_units=angle_units, source="observed")
    counts["observed"] = obs.pop("_counts")
    for k in _TABLE_KINDS:
        frames[k].append(obs[k])

    null_params = dict(null_params or {})
    if "uniform" in nulls:
        from .nulls import null_uniform

        nu = null_uniform(locs, cells, dataset=dataset, sigma_um=sigma_um,
                          tau_s=tau_s, edges=edges, angle_units=angle_units,
                          min_localisations=min_localisations,
                          seed=seed, **null_params.get("uniform", {}))
        for k in _TABLE_KINDS:
            frames[k].append(nu[k])
        counts["null_uniform"] = nu.get("_counts", {})
    if "kinetic" in nulls:
        from .nulls import null_kinetic

        nk = null_kinetic(cells, locs, dataset=dataset, sigma_um=sigma_um,
                          tau_s=tau_s, edges=edges, angle_units=angle_units,
                          min_localisations=min_localisations,
                          seed=seed, **null_params.get("kinetic", {}))
        for k in _TABLE_KINDS:
            frames[k].append(nk[k])
        counts["null_kinetic"] = nk.get("_counts", {})

    written = {}
    for k in _TABLE_KINDS:
        combined = pd.concat(frames[k], ignore_index=True)
        validate_schema(combined, k)
        written[k] = _write_table(combined, outdir / f"{k}.csv")

    prov = {
        "package_version": _package_version(),
        "git_commit": _git_commit(),
        "dataset": dataset,
        "edges": list(edges),
        "sigma_um": float(sigma_um),
        "sigma_provenance": sigma_provenance,
        "sigma_basis": sigma_basis,
        "tau_s": float(tau_s),
        "d_offset_um2s": float(sigma_um ** 2 / tau_s),
        "min_localisations": int(min_localisations),
        "angle_units": angle_units,
        "nulls": list(nulls),
        "null_params": null_params,
        "seed": int(seed),
        "d_estimator": {"name": APPARENT_D_QUALIFIED_NAME,
                        "formula": APPARENT_D_FORMULA},
        "counts": counts,
        "inputs": {name: {"path": str(p), "sha256": _sha256(Path(p))}
                   for name, p in (input_paths or {}).items()
                   if Path(p).exists()},
        "outputs": written,
        "sources": list(SOURCE_VALUES),
    }
    import json

    (outdir / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))
    return prov


#: Fields a stored analysis MUST name for its σ to be interpretable.
REQUIRED_SIGMA_FIELDS = ("sigma_um", "sigma_provenance")


def read_provenance(path) -> dict:
    """Load a run's ``provenance.json``, FAILING when its σ is not fully named.

    Assuming ``theoretical`` for a file that does not say so is exactly the silent
    default the provenance state exists to prevent, an old export made under a
    lab-configured σ would come back reading as though it had used the optics. So a
    provenance file missing ``sigma_um`` or ``sigma_provenance``, or carrying a
    provenance outside :data:`AcquisitionParams.SIGMA_SOURCES`, raises rather than
    loading with a guess.
    """
    import json

    from ..canonical import AcquisitionParams

    path = Path(path)
    if path.is_dir():
        path = path / "provenance.json"
    if not path.is_file():
        raise SchemaError(f"no provenance.json at {path}")
    try:
        prov = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise SchemaError(f"{path} is not valid JSON: {exc}") from exc

    missing = [f for f in REQUIRED_SIGMA_FIELDS if prov.get(f) is None]
    if missing:
        raise SchemaError(
            f"{path} does not name {' and '.join(missing)}. A saved analysis whose σ "
            "or σ provenance is unknown does not load: defaulting it to 'theoretical' "
            "would silently reinterpret a run made under a configured or measured σ. "
            "Re-export the run, or add the fields by hand if you know what they were.")
    if prov["sigma_provenance"] not in AcquisitionParams.SIGMA_SOURCES:
        raise SchemaError(
            f"{path} carries sigma_provenance={prov['sigma_provenance']!r}, not one of "
            f"{AcquisitionParams.SIGMA_SOURCES}.")
    return prov
