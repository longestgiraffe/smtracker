"""Tracks out of SMTracker, for tools that fit them elsewhere (agreed 17 Oct 2026: "vbSPT export").

vbSPT (v2's File > Export > vbSPT, `SMT_GUI/@SMT/exp2SMTrack.m:60-66`)
-----------------------------------------------------------------------
v2 writes one ``vbSPT<condition>.mat`` per condition holding ONE variable, ``finalTraj``: a 1 x N cell array, one cell
per trajectory, each an n x 2 matrix ``[x y]`` in NANOMETRES, every movie and cell of the condition pooled. No time
column and no run-input file, so vbSPT's ``timestep`` is set by hand. Written the same way here, with three
corrections and two additions:

- A track is SPLIT AT EVERY FRAME GAP, since vbSPT assumes one time step between rows; v2 split at NaN rows but
  dropped the last point of each piece and duplicated pieces of a track with two or more gaps (`getRidnan.m:13-28`).
  Here each contiguous run of frames is one trajectory, every point kept; a piece of one point is dropped (no step).
- The tracks are the ANALYSIS's tracks: the canonical table the fits read (the track-length filter, the cell
  filter, exclusions and the track cap as the workspace applies them), never the tracker's raw output.
- Beside ``finalTraj``, ``day``, ``movie`` and ``cell`` per trajectory (vbSPT reads only the field it is told to), so
  replicate days can be fitted apart (RULE 0: days are independent), and a plain-text note with the time step, the
  length unit and the counts.

SMMTrack (v2's File > Export > SMMTrack, `exp2SMTrack.m:45-58`; added 17 Oct 2026)
---------------------------------------------------------------------------------
For Bernhard Schmidt's SMMTrack (track heat maps, dwell times). v2 writes one ``SMMTrack_<protein>_<condition>.csv``
per condition, NO header, six numeric columns ``Nr TrackID PointID x y t``: a running row number, the track's number in
the condition (1..N, movies then cells then tracks), the point's number in the track (1..n), x and y in MICROMETRES
(pixels x pixel size / 1000) and t in SECONDS FROM THE TRACK'S FIRST POINT (`cellTracks.m:53`: ``0:tau:(n-1)*tau``,
one row a frame, NaN rows where u-track closed a gap). Written the same here, from the analysis's tracks, with:

- A gap is NOT written as a NaN row: the point after a gap carries its own time, ``(frame - first frame) * tau``, which
  is what v2's NaN rows encoded (a missing row never meant a missing time step in SMMTrack's own counting, so a
  consumer that steps through rows should read t).
- Beside the file, ``..._tracks.csv`` (with a header) gives each TrackID its replicate ``day``, ``movie``, ``cell`` and
  the analysis's ``uid`` (RULE 0: days are independent, so a consumer can keep them apart), and a note with the
  columns, units and counts.

Invariants: coordinates come from ``x_um``/``y_um`` (µm, `canonical`) times 1000 for vbSPT and as they are for
SMMTrack; the time step is the table's ``tau_s``; nothing here fits or filters beyond splitting at gaps (vbSPT).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["TrackExportError", "vbspt_trajectories", "write_vbspt", "vbspt_file_name", "smmtrack_rows",
           "smmtrack_file_name", "write_smmtrack", "SMMTRACK_COLUMNS"]

#: v2's six SMMTrack columns, in order (no header in the file; the note names them).
SMMTRACK_COLUMNS = ("Nr", "TrackID", "PointID", "x_um", "y_um", "t_s")


class TrackExportError(RuntimeError):
    """Raised when tracks cannot be exported."""


def vbspt_trajectories(table: pd.DataFrame) -> dict:
    """The trajectories of ``table`` as vbSPT reads them: ``{"traj": [n x 2 arrays, nm], "day": [...], "movie":
    [...], "cell": [...], "uid": [...], "tau_s": float, "n_tracks": int, "n_split": int}`` -- each contiguous run of
    frames of a track one trajectory (a track with a gap is split; a piece of one point dropped)."""
    needed = {"uid", "frame", "x_um", "y_um"}
    missing = needed - set(table.columns)
    if missing:
        raise TrackExportError(f"not a canonical track table (missing {sorted(missing)})")
    if not len(table):
        raise TrackExportError("there are no tracks to export")
    tau = table["tau_s"].dropna().unique() if "tau_s" in table.columns else []
    if len(tau) != 1:
        raise TrackExportError(f"the tracks need one frame interval to export for vbSPT, found {list(tau)}")
    out = {"traj": [], "day": [], "movie": [], "cell": [], "uid": [], "tau_s": float(tau[0])}
    split = 0
    ordered = table.sort_values(["uid", "frame"], kind="stable")
    for uid, track in ordered.groupby("uid", sort=False, observed=True):
        frames = track["frame"].to_numpy()
        xy = 1000.0 * track[["x_um", "y_um"]].to_numpy(dtype=float)
        breaks = np.flatnonzero(np.diff(frames) != 1) + 1
        pieces = np.split(np.arange(len(frames)), breaks)
        split += len(pieces) > 1
        first = track.iloc[0]
        for piece in pieces:
            if len(piece) < 2:
                continue
            out["traj"].append(xy[piece])
            out["day"].append(_plain(first.get("day")))
            out["movie"].append(str(first.get("movie", "")))
            out["cell"].append(str(first.get("cell", "")))
            out["uid"].append(str(uid))
    if not out["traj"]:
        raise TrackExportError("no track has two consecutive frames to export")
    out["n_tracks"] = int(ordered["uid"].nunique())
    out["n_split"] = int(split)
    return out


def _plain(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def vbspt_file_name(condition: str) -> str:
    """v2's file name, ``vbSPT<condition>.mat``, with characters a file name cannot hold replaced."""
    return "vbSPT" + re.sub(r'[<>:"/\\|?*\s]+', "_", str(condition)).strip("_") + ".mat"


def write_vbspt(table: pd.DataFrame, folder, condition: str, *, label: str | None = None) -> dict:
    """Write ``table``'s trajectories for vbSPT into ``folder``: ``vbSPT<condition>.mat`` (``finalTraj`` plus
    ``day``, ``movie``, ``cell``) and a note beside it. Returns ``{"mat": path, "note": path, "n_traj": int, ...}``."""
    from scipy.io import savemat

    folder = Path(folder)
    if not folder.is_dir():
        raise TrackExportError(f"no such folder: {folder}")
    got = vbspt_trajectories(table)
    n = len(got["traj"])
    final = np.empty((1, n), dtype=object)
    for i, xy in enumerate(got["traj"]):
        final[0, i] = xy
    movie = np.empty((1, n), dtype=object)
    cell = np.empty((1, n), dtype=object)
    for i in range(n):
        movie[0, i] = got["movie"][i]
        cell[0, i] = got["cell"][i]
    mat = folder / vbspt_file_name(condition)
    savemat(mat, {"finalTraj": final, "day": np.asarray(got["day"], dtype=float).reshape(1, n),
                  "movie": movie, "cell": cell}, do_compression=True)
    lengths = np.array([len(t) for t in got["traj"]])
    days = sorted({d for d in got["day"] if np.isfinite(d)})
    note = mat.with_suffix(".txt")
    note.write_text(
        f"vbSPT input exported by SMTracker for condition {label or condition}\n"
        f"file: {mat.name}\n"
        "trajectoryfield: finalTraj (1 x N cell, each n x 2 [x y])\n"
        "length unit: nm\n"
        f"timestep: {got['tau_s']:g} s (set this in the vbSPT run input)\n"
        "dim: 2\n"
        f"trajectories: {n} from {got['n_tracks']} tracks "
        f"({got['n_split']} tracks split at frame gaps; pieces of one point dropped)\n"
        f"points per trajectory: median {int(np.median(lengths))}, min {int(lengths.min())}, max {int(lengths.max())}\n"
        f"replicates: {', '.join(f'{d:g}' for d in days) or 'none recorded'} "
        "(variable 'day' per trajectory; replicates are independent, fit them apart)\n"
        "also per trajectory: 'movie', 'cell'\n"
        "tracks: as analysed (track-length filter, cell filter, exclusions, track cap)\n",
        encoding="utf-8")
    return {"mat": mat, "note": note, "n_traj": n, "n_tracks": got["n_tracks"], "n_split": got["n_split"]}


def _safe(name) -> str:
    return re.sub(r'[<>:"/\\|?*\s]+', "_", str(name)).strip("_")


def smmtrack_file_name(condition: str, protein: str | None = None) -> str:
    """v2's file name, ``SMMTrack_<protein>_<condition>.csv`` (``SMMTrack_<condition>.csv`` with no protein), with
    characters a file name cannot hold replaced."""
    parts = ["SMMTrack"] + ([_safe(protein)] if protein and _safe(protein) else []) + [_safe(condition)]
    return "_".join(parts) + ".csv"


def smmtrack_rows(table: pd.DataFrame) -> tuple:
    """``table``'s tracks as SMMTrack rows: ``(rows, tracks)`` -- ``rows`` an n x 6 float array in
    `SMMTRACK_COLUMNS` order, ``tracks`` one row per TrackID (``TrackID, day, movie, cell, uid, n_points``). Tracks in
    v2's order: by movie, then cell, then the track's first frame; every point kept, t from the track's first point."""
    needed = {"uid", "frame", "x_um", "y_um"}
    missing = needed - set(table.columns)
    if missing:
        raise TrackExportError(f"not a canonical track table (missing {sorted(missing)})")
    if not len(table):
        raise TrackExportError("there are no tracks to export")
    tau = table["tau_s"].dropna().unique() if "tau_s" in table.columns else []
    if len(tau) != 1:
        raise TrackExportError(f"the tracks need one frame interval to export for SMMTrack, found {list(tau)}")
    tau = float(tau[0])
    plain = pd.DataFrame({c: table[c].to_numpy() for c in table.columns})          # no attrs, no categories
    for c in ("movie", "cell"):
        plain[c] = plain[c].astype(str) if c in plain.columns else ""
    first = plain.groupby("uid", sort=False)["frame"].transform("min")
    plain = plain.assign(_first=first).sort_values(["movie", "cell", "_first", "uid", "frame"], kind="stable")
    blocks, tracks = [], []
    for k, (uid, track) in enumerate(plain.groupby("uid", sort=False), start=1):
        frames = track["frame"].to_numpy(dtype=float)
        n = len(frames)
        blocks.append(np.column_stack([np.full(n, k), np.arange(1, n + 1), track["x_um"].to_numpy(dtype=float),
                                       track["y_um"].to_numpy(dtype=float), (frames - frames[0]) * tau]))
        row = track.iloc[0]
        tracks.append({"TrackID": k, "day": _plain(row.get("day")), "movie": str(row.get("movie", "")),
                       "cell": str(row.get("cell", "")), "uid": str(uid), "n_points": n})
    body = np.vstack(blocks)
    rows = np.column_stack([np.arange(1, len(body) + 1), body])
    return rows, pd.DataFrame(tracks)


def write_smmtrack(table: pd.DataFrame, folder, condition: str, *, protein: str | None = None,
                   label: str | None = None) -> dict:
    """Write ``table``'s tracks for SMMTrack into ``folder``: v2's headerless six-column CSV, the per-track sidecar
    ``..._tracks.csv`` and a note. Returns ``{"csv", "tracks", "note", "n_tracks", "n_points"}``."""
    folder = Path(folder)
    if not folder.is_dir():
        raise TrackExportError(f"no such folder: {folder}")
    rows, tracks = smmtrack_rows(table)
    path = folder / smmtrack_file_name(condition, protein)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        for r in rows:
            fh.write(f"{int(r[0])},{int(r[1])},{int(r[2])},{r[3]:.6f},{r[4]:.6f},{r[5]:.6g}\n")
    side = path.with_name(path.stem + "_tracks.csv")
    tracks.to_csv(side, index=False)
    days = sorted({d for d in tracks["day"] if np.isfinite(d)})
    note = path.with_suffix(".txt")
    tau = float(table["tau_s"].dropna().iloc[0])
    note.write_text(
        f"SMMTrack input exported by SMTracker for condition {label or condition}\n"
        f"file: {path.name} (no header, comma-separated, '.' decimals)\n"
        "columns: Nr, TrackID, PointID, x (um), y (um), t (s from the track's first point)\n"
        f"frame interval: {tau:g} s; a frame gap is not a row: the point after it carries its own t\n"
        f"tracks: {len(tracks)}, points: {len(rows)}\n"
        f"per track (TrackID -> replicate, movie, cell, uid): {side.name}\n"
        f"replicates: {', '.join(f'{d:g}' for d in days) or 'none recorded'} "
        "(replicates are independent: keep them apart)\n"
        "tracks: as analysed (track-length filter, cell filter, exclusions, track cap)\n",
        encoding="utf-8")
    return {"csv": path, "tracks": side, "note": note, "n_tracks": int(len(tracks)), "n_points": int(len(rows))}
