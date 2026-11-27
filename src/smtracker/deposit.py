"""The manuscript export contract: two tidy tables per (enzyme × condition).

THE FIREWALL
------------
Analysis code never draws; figure code never touches raw data. This module is the
analysis side of that line: it writes tables complete enough that
``scripts/manuscript_figures.py`` can rebuild every panel **without importing
smtracker and without opening a single ``.mat``**. Get the firewall right and σ,
erosion and "which enzymes qualify for an interval" stop being blockers and become
command-line arguments.

The rule that makes it work
---------------------------
**Every parameter that could have been chosen differently is a COLUMN.** If it is a
column, a later question is answered by filtering. If it is a footnote, the question
is answered by re-running. So σ, its provenance and its basis, the D threshold, the
localisation floor, the erosion, the shell cut and the acquisition constants all ride
on every row: repeated, denormalised, and worth the bytes.

One row per track: structurally, not by convention
---------------------------------------------------
The track table is one row per track and :func:`validate_deposit` refuses anything
else. Per-localisation rows are the legacy failure mode (a long track then counts many
times in every downstream sum), and the fix is to make that shape impossible
downstream rather than to remember not to write it.

The localisation-level statistic is still exactly recoverable, which is the whole
design: every localisation of a track carries the same weight ``w = 1/n_loc`` and the
same label, so carrying ``n_loc_shell`` per track reproduces the weighted sums to
machine precision::

    additive_pp = 100 * ( Σ w·n_loc_shell·[low] / Σ w·n_loc_shell
                        − Σ w·n_loc·[low]      / Σ w·n_loc )

with ``w = 1/n_loc``, i.e. no raw data and no re-derivation needed downstream.
:func:`shell_additive_pp` is that formula, once, so the figure script and any check
can never disagree about it.

CSV is the archival copy; Parquet is written alongside for speed. **CSV wins on
disagreement**: it is the copy that will still open in twenty years.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "DepositError",
    "TRACK_COLUMNS",
    "CELL_COLUMNS",
    "PROVENANCE_COLUMNS",
    "build_deposit",
    "write_deposit",
    "validate_deposit",
    "shell_additive_pp",
    "low_D_fraction",
    "D_LABELS",
    "REGIONS",
]


class DepositError(RuntimeError):
    """Raised when an export would violate the contract."""


#: Labels. Gate-safe vocabulary: low_D / high_D, shell / interior, never a word
#: from ``compare_plots.FORBIDDEN_WORDS``.
D_LABELS = ("low_D", "high_D")
REGIONS = ("shell", "interior")

#: Every parameter that could have been chosen differently, carried on every row of
#: both tables. Repeated per row on purpose: a deposited table must be answerable on
#: its own, without a sidecar anyone can lose.
PROVENANCE_COLUMNS = [
    "d_threshold", "d_correction_formula", "blur_R",
    "sigma_um", "sigma_provenance", "sigma_basis", "localisation_floor_D",
    "shell_r_norm", "erosion_px", "min_localisations",
    "pixel_nm", "frame_interval_s", "na", "emission_nm",
    "smtracker_version", "config_hash",
]

#: One row per TRACK. Never per localisation.
TRACK_COLUMNS = [
    "enzyme", "condition", "replicate_day", "series", "cell_id", "track_id",
    "n_loc", "w",
    "D_corrected", "D_uncorrected",
    "d_label",
    "r_norm", "n_loc_shell", "frac_loc_shell", "region",
] + PROVENANCE_COLUMNS

#: One row per CELL, the level every diagnostic and every bootstrap unit reads.
CELL_COLUMNS = [
    "enzyme", "condition", "replicate_day", "series", "cell_id",
    "n_tracks", "n_low_D", "n_high_D",
    "shell_fraction_low_D", "shell_fraction_high_D",
    "mask_area_um2", "contour_n_points",
] + PROVENANCE_COLUMNS


def _series_of(cell_id: str) -> str:
    """``"<movie>:<cell>"`` → the movie/series token. Identity for anything else."""
    return str(cell_id).split(":")[0]


def _config_hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def build_deposit(localisations: pd.DataFrame, cells: pd.DataFrame, *,
                  params, sigma_um: float, d_threshold: float,
                  shell_r_norm: float, erosion_px: float,
                  min_localisations: int = 5,
                  day_of=None, contour_points=None,
                  blur_R: float | None = None,
                  na: float | None = None, emission_nm: float | None = None,
                  enzyme: str | None = None, condition: str | None = None) -> dict:
    """Collapse the spatial tables into the two manuscript tables.

    Parameters
    ----------
    localisations
        ``smtracker.spatial.io.build_tables(...)["localisations"]`` for ONE
        (enzyme × condition), observed source only.
    cells
        the matching ``["cells"]`` table (one row per cell, complete).
    params
        :class:`~smtracker.canonical.AcquisitionParams`, σ provenance and the
        acquisition constants come from here, never from an argument, so the
        deposited σ and the analysed σ cannot disagree.
    sigma_um
        the σ the ANALYSIS ran with, µm. Checked against ``params``: a mismatch
        raises, because a table stamped with one σ and computed under another is
        worse than no stamp at all.
    day_of
        optional mapping ``cell_id -> replicate_day``; else read from the
        ``session`` column.
    contour_points
        optional mapping ``cell_id -> n polygon vertices`` (a segmentation
        diagnostic; round 12 showed contour sampling drives width degeneracy).

    Returns
    -------
    dict with ``"tracks"`` and ``"cells"`` frames, already
    :func:`validate_deposit`-clean.
    """
    from .msd import BERGLUND_R_CONTINUOUS, berglund_correct_apparent_D

    if localisations.empty:
        raise DepositError("no localisations to deposit: refusing to write an "
                           "empty table that would read as 'no molecules'")
    stamp = params.sigma_stamp()
    if not np.isclose(stamp["sigma_nm"] / 1000.0, float(sigma_um), rtol=1e-9,
                      atol=1e-12):
        raise DepositError(
            f"σ mismatch: the analysis ran at {sigma_um * 1000:.4g} nm but params "
            f"carry {stamp['sigma_nm']:.4g} nm ({stamp['sigma_provenance']}). A table "
            "stamped with one σ and computed under another is worse than no stamp.")

    blur_R = BERGLUND_R_CONTINUOUS if blur_R is None else float(blur_R)
    tau_s = float(params.tau_s)
    obs = localisations[localisations.get("source", "observed") == "observed"] \
        if "source" in localisations else localisations

    shell = obs["r_norm"].to_numpy() > float(shell_r_norm)
    g = obs.assign(_shell=shell).groupby("track_id", sort=True)
    tracks = pd.DataFrame({
        "track_id": g.size().index,
        "n_loc": g.size().to_numpy(),
        "n_loc_shell": g["_shell"].sum().to_numpy().astype(int),
        "D_uncorrected": g["track_D_um2s"].first().to_numpy(),
        "r_norm": g["r_norm"].median().to_numpy(),
        "cell_id": g["cell_id"].first().to_numpy(),
        "enzyme": g["enzyme"].first().to_numpy(),
        "condition": g["condition"].first().to_numpy(),
        "session": g["session"].first().to_numpy(),
    })
    tracks["w"] = 1.0 / tracks["n_loc"]
    tracks["frac_loc_shell"] = tracks["n_loc_shell"] / tracks["n_loc"]
    tracks["D_corrected"] = berglund_correct_apparent_D(
        tracks["D_uncorrected"].to_numpy(), tau_s, sigma_um=float(sigma_um), R=blur_R)
    # the LABEL is on the uncorrected (apparent) scale: that is the pre-registered
    # basis, and correcting first would move a pre-registered boundary silently.
    tracks["d_label"] = np.where(tracks["D_uncorrected"] < float(d_threshold),
                                 D_LABELS[0], D_LABELS[1])
    # a track is `shell` if it VISITS the shell, the same "independent visit" rule
    # N_min counts on. It is not a centroid, which regresses to midcell for fast tracks.
    tracks["region"] = np.where(tracks["n_loc_shell"] > 0, REGIONS[0], REGIONS[1])
    tracks["series"] = [_series_of(c) for c in tracks["cell_id"]]
    tracks["replicate_day"] = (
        tracks["cell_id"].map(day_of) if day_of is not None else tracks["session"])
    if enzyme is not None:
        tracks["enzyme"] = enzyme
    if condition is not None:
        tracks["condition"] = condition

    prov = {
        "d_threshold": float(d_threshold),
        "d_correction_formula": "D_true = (D_app - sigma^2/tau) / (1 - 2R)",
        "blur_R": blur_R,
        "sigma_um": float(sigma_um),
        "sigma_provenance": stamp["sigma_provenance"],
        "sigma_basis": stamp["sigma_basis"],
        "localisation_floor_D": float(params.localization_floor_D),
        "shell_r_norm": float(shell_r_norm),
        "erosion_px": float(erosion_px),
        "min_localisations": int(min_localisations),
        "pixel_nm": float(params.pixel_size_nm),
        "frame_interval_s": tau_s,
        "na": na,
        "emission_nm": emission_nm,
        "smtracker_version": _version(),
    }
    prov["config_hash"] = _config_hash(prov)
    for k, v in prov.items():
        tracks[k] = v

    # ---- the cell table: one row per cell, the unit every diagnostic reads ----
    low = tracks["d_label"].to_numpy() == D_LABELS[0]
    per = tracks.assign(_low=low)
    by_cell = per.groupby("cell_id", sort=True)

    def _shell_fraction(sub):
        """Mean per-track shell fraction within a labelled group (w-consistent:
        each track counts once, which is the point of the whole table)."""
        return float(sub["frac_loc_shell"].mean()) if len(sub) else np.nan

    rows = []
    for cid, sub in by_cell:
        lo, hi = sub[sub["_low"]], sub[~sub["_low"]]
        rows.append({
            "cell_id": cid,
            "enzyme": sub["enzyme"].iloc[0],
            "condition": sub["condition"].iloc[0],
            "replicate_day": sub["replicate_day"].iloc[0],
            "series": sub["series"].iloc[0],
            "n_tracks": int(len(sub)),
            "n_low_D": int(len(lo)),
            "n_high_D": int(len(hi)),
            "shell_fraction_low_D": _shell_fraction(lo),
            "shell_fraction_high_D": _shell_fraction(hi),
        })
    cell_tab = pd.DataFrame(rows)
    geo = cells.drop_duplicates("cell_id").set_index("cell_id")
    cell_tab["mask_area_um2"] = cell_tab["cell_id"].map(
        geo["projected_area_um2"] if "projected_area_um2" in geo else pd.Series(dtype=float))
    cell_tab["contour_n_points"] = (
        cell_tab["cell_id"].map(contour_points) if contour_points is not None
        else pd.Series([pd.NA] * len(cell_tab), dtype="object"))
    for k, v in prov.items():
        cell_tab[k] = v

    out = {"tracks": tracks[TRACK_COLUMNS].sort_values(
               ["enzyme", "condition", "replicate_day", "series", "cell_id",
                "track_id"], kind="stable").reset_index(drop=True),
           "cells": cell_tab[CELL_COLUMNS].sort_values(
               ["enzyme", "condition", "replicate_day", "series", "cell_id"],
               kind="stable").reset_index(drop=True)}
    validate_deposit(out["tracks"], "tracks")
    validate_deposit(out["cells"], "cells")
    return out


def validate_deposit(df: pd.DataFrame, kind: str) -> None:
    """Check the contract. One row per track / per cell is enforced, not assumed."""
    required = {"tracks": TRACK_COLUMNS, "cells": CELL_COLUMNS}[kind]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DepositError(f"{kind} table is missing columns {missing}")
    key = (["enzyme", "condition", "cell_id", "track_id"] if kind == "tracks"
           else ["enzyme", "condition", "cell_id"])
    dup = df.duplicated(key)
    if dup.any():
        raise DepositError(
            f"{kind} table has {int(dup.sum())} duplicate rows on {key}: it must be "
            "one row per "
            + ("track. Per-localisation rows are the legacy failure mode: a long "
               "track would count many times in every downstream sum."
               if kind == "tracks" else "cell."))
    if kind == "tracks":
        bad = set(df["d_label"].unique()) - set(D_LABELS)
        if bad:
            raise DepositError(f"unknown d_label(s) {sorted(bad)}; expected {D_LABELS}")
        bad = set(df["region"].unique()) - set(REGIONS)
        if bad:
            raise DepositError(f"unknown region(s) {sorted(bad)}; expected {REGIONS}")
        if not np.allclose(df["w"], 1.0 / df["n_loc"]):
            raise DepositError("w must equal 1/n_loc: it is precomputed so that no "
                               "one downstream re-derives it differently")


def shell_additive_pp(tracks: pd.DataFrame) -> float:
    """The primary endpoint, in percentage points, from the track table alone.

    ``low_D`` share of the shell minus its share overall, weighted by ``w = 1/n_loc``
    so a track counts once however long it is. Exactly the localisation-level
    statistic, because every localisation of a track shares that track's weight and
    label, which is why ``n_loc_shell`` is carried. ONE implementation, used by the
    figure script and by every check.
    """
    w = tracks["w"].to_numpy()
    n_shell = tracks["n_loc_shell"].to_numpy()
    n_loc = tracks["n_loc"].to_numpy()
    low = (tracks["d_label"].to_numpy() == D_LABELS[0]).astype(float)
    ws, wa = w * n_shell, w * n_loc
    if ws.sum() <= 0 or wa.sum() <= 0:
        return float("nan")
    return 100.0 * ((ws * low).sum() / ws.sum() - (wa * low).sum() / wa.sum())


def low_D_fraction(tracks: pd.DataFrame) -> float:
    """``low_D`` share of tracks, w-weighted (each track counts once)."""
    w = tracks["w"].to_numpy() * tracks["n_loc"].to_numpy()
    low = (tracks["d_label"].to_numpy() == D_LABELS[0]).astype(float)
    return float((w * low).sum() / w.sum()) if w.sum() else float("nan")


def write_deposit(tables: dict, outdir, *, stem: str) -> dict:
    """Write ``<stem>_tracks`` / ``<stem>_cells`` as CSV (archival) + Parquet (fast).

    Returns ``{name: {"csv": path, "sha256": ..., "rows": n, "parquet": path|None}}``.
    Parquet is best-effort; **CSV wins on disagreement** and is always written.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = {}
    for kind in ("tracks", "cells"):
        df = tables[kind]
        validate_deposit(df, kind)
        csv = outdir / f"{stem}_{kind}.csv"
        df.to_csv(csv, index=False)
        info = {"csv": str(csv), "rows": int(len(df)),
                "sha256": hashlib.sha256(csv.read_bytes()).hexdigest()}
        try:
            pq = outdir / f"{stem}_{kind}.parquet"
            df.to_parquet(pq, index=False)
            info["parquet"] = str(pq)
        except Exception:            # pyarrow absent / unsupported dtype, CSV stands
            info["parquet"] = None
        written[kind] = info
    return written


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("smtracker")
    except Exception:
        return "unknown"
