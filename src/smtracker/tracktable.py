"""Track TABLES from other tracking tools: one reader, several dialects, the same raw table out.

Asked for (13 Oct 2026): "which other tracker data support can we implement? should not be too
hard and test heavy". u-track (`ingest`) and TrackMate XML (`trackmate`) stay as they are; every
other tool that writes its tracks as a table comes in here, from
``<condition>/track_tables/<movie>.<csv|tsv|txt>`` -- a folder of its own, since u-track's
results already live in a folder named ``tracks`` (``<movie>/TrackingPackage/tracks/``).

THE DIALECTS, each from its tool's own documentation or its reader in Spot-On, never assumed:

==============  ==========================================  =======  =====================================
dialect         columns                                     units    from
==============  ==========================================  =======  =====================================
trackmate_csv   TRACK_ID, POSITION_X, POSITION_Y, FRAME     stated   TrackMate's spot export; its units row
                (+ TrackMate's extra header rows)                    says pixel or micron
trackpy         particle, x, y, frame                       px       trackpy's linking output
mosaic          Trajectory, x, y, Frame                     px       Spot-On `read_mosaic` (MOSAIC Suite)
trajectory      trajectory, x, y, frame (+ t, index)        µm       TrackIt's CSV export (gridlib
                                                                     `read_track_file_csv`) and Spot-On's
                                                                     own CSV (`read_csv`): the same table
evalspt         no header, tab: x, y, frame, trajectory     declare  Spot-On `read_evalspt` (SLIMfast /
                                                                     evalSPT); units not stated there
==============  ==========================================  =======  =====================================

UNITS ARE READ OR DECLARED, NEVER GUESSED: a wrong unit moves every D by the square of the factor.
Where a dialect does not fix its units (evalSPT; TrackMate when its units row is absent), the
folder declares them in ``track_tables/format.json`` (``{"units": "um"}``, or a whole custom
dialect: ``{"columns": {"track": ..., "frame": ..., "x": ..., "y": ...}, "units": "nm",
"frame_origin": 1}``), and a table with neither is refused, named. A folder of tables with no
movies (so no preparation run and no ``crops.json``) declares the calibration there too:
``{"pixel_size_nm": 106, "frame_interval_ms": 20}`` (`project._trackmate_acquisition_params`).
Positions leave here in
PIXELS (the project's pixel size converts µm and nm, and is required for them), frames
1-based like u-track's, which only the trigger clock and the crop ranges see -- every lag is a
frame difference. A frame origin no source documents is 1 for tables from MATLAB tools
(TrackIt, evalSPT) and 0 for the Python and Java ones, said in ``attrs["frame_origin"]``.

Fails loud on: a header matching no dialect, a duplicate (track, frame), a coordinate that is not
a number, and µm or nm without a pixel size. Untracked spots (TrackMate's ``TRACK_ID`` "None")
are dropped and counted.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["TrackTableError", "Dialect", "DIALECTS", "TABLE_FOLDER", "TABLE_GLOB", "FORMAT_FILE",
           "TABLE_SUFFIXES", "detect_dialect", "read_track_table", "folder_format"]

#: Where a condition keeps tables from other trackers, and the declaration beside them.
TABLE_FOLDER = "track_tables"
TABLE_GLOB = f"**/{TABLE_FOLDER}/*"
TABLE_SUFFIXES = (".csv", ".tsv", ".txt")
FORMAT_FILE = "format.json"
#: Units a table may be in, and nanometres per unit (pixels are converted by the pixel size).
UNITS = {"px": None, "um": 1000.0, "nm": 1.0}
_UNIT_WORDS = {"pixel": "px", "pixels": "px", "px": "px", "micron": "um", "microns": "um",
               "um": "um", "µm": "um", "micrometer": "um", "nm": "nm", "nanometer": "nm"}


class TrackTableError(RuntimeError):
    """A track table that cannot be read as it stands, named with the reason."""


@dataclass(frozen=True)
class Dialect:
    """How one tool writes its tracks: the column of each role (a name, or a position in a table
    with no header), the separator, the units (None: stated in the file or declared), and the
    first frame's number."""

    name: str
    columns: dict
    units: str | None
    frame_origin: int
    header: bool = True
    sep: str = ","
    source: str = ""
    #: Track ids meaning "not in a track" (TrackMate exports its lone spots with "None").
    untracked: tuple = field(default=())


DIALECTS = {
    "trackmate_csv": Dialect(
        "trackmate_csv", {"track": "TRACK_ID", "frame": "FRAME", "x": "POSITION_X", "y": "POSITION_Y"},
        units=None, frame_origin=0, untracked=("None", "", "nan"),
        source="TrackMate's spot table export; units from its units row"),
    "trackpy": Dialect(
        "trackpy", {"track": "particle", "frame": "frame", "x": "x", "y": "y"},
        units="px", frame_origin=0, source="trackpy link() output"),
    "mosaic": Dialect(
        "mosaic", {"track": "Trajectory", "frame": "Frame", "x": "x", "y": "y"},
        units="px", frame_origin=0, source="MOSAIC ParticleTracker CSV (Spot-On read_mosaic)"),
    "trajectory": Dialect(
        "trajectory", {"track": "trajectory", "frame": "frame", "x": "x", "y": "y"},
        units="um", frame_origin=1,
        source="TrackIt CSV export (gridlib read_track_file_csv) and Spot-On's CSV (read_csv)"),
    "evalspt": Dialect(
        "evalspt", {"x": 0, "y": 1, "frame": 2, "track": 3}, units=None, frame_origin=1,
        header=False, sep="\t", source="evalSPT / SLIMfast (Spot-On read_evalspt); units declared"),
}


def _header(path: Path) -> list:
    with open(path, "r", encoding="utf-8-sig", errors="replace") as handle:
        first = handle.readline().rstrip("\r\n")
    sep = "\t" if "\t" in first else ","
    return [c.strip().strip('"') for c in first.split(sep)], sep


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def folder_format(folder: Path) -> dict:
    """The folder's ``format.json`` declaration, or ``{}``. A file that is not valid JSON, or names
    an unknown unit or dialect, is refused rather than ignored."""
    path = Path(folder) / FORMAT_FILE
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TrackTableError(f"{path}: not a readable declaration ({exc})") from exc
    if not isinstance(data, dict):
        raise TrackTableError(f"{path}: a declaration is a JSON object")
    if "units" in data and data["units"] not in UNITS:
        raise TrackTableError(f"{path}: units {data['units']!r} is not one of {sorted(UNITS)}")
    if "dialect" in data and data["dialect"] not in DIALECTS:
        raise TrackTableError(f"{path}: dialect {data['dialect']!r} is not one of {sorted(DIALECTS)}")
    for key in ("pixel_size_nm", "frame_interval_ms"):
        if key in data:
            try:
                ok = float(data[key]) > 0
            except (TypeError, ValueError):
                ok = False
            if not ok:
                raise TrackTableError(f"{path}: {key} must be a positive number, got {data[key]!r}")
    return data


def detect_dialect(path: str | Path, declared: dict | None = None) -> Dialect:
    """The dialect of a table, from its header (or the folder's declaration): never from its name."""
    path = Path(path)
    declared = declared or {}
    if "columns" in declared:
        cols = declared["columns"]
        missing = [r for r in ("track", "frame", "x", "y") if r not in cols]
        if missing:
            raise TrackTableError(f"{path.parent / FORMAT_FILE}: columns lack {missing}")
        header = all(isinstance(v, str) for v in cols.values())
        return Dialect("declared", dict(cols), units=declared.get("units"),
                       frame_origin=int(declared.get("frame_origin", 1)), header=header,
                       sep=declared.get("sep", "\t" if not header else ","),
                       source=f"declared in {FORMAT_FILE}")
    if "dialect" in declared:
        return DIALECTS[declared["dialect"]]
    names, sep = _header(path)
    lowered = {n.lower() for n in names}
    if all(_is_number(n) for n in names if n) and len([n for n in names if n]) >= 4:
        if sep == "\t" and len([n for n in names if n]) == 4:
            return DIALECTS["evalspt"]
        raise TrackTableError(
            f"{path}: a table with no header and {len(names)} columns matches no known dialect; "
            f"declare its columns in {TABLE_FOLDER}/{FORMAT_FILE}")
    if {"track_id", "position_x", "position_y", "frame"} <= lowered:
        return DIALECTS["trackmate_csv"]
    if {"particle", "x", "y", "frame"} <= lowered:
        return DIALECTS["trackpy"]
    if {"Trajectory", "x", "y", "Frame"} <= set(names):
        return DIALECTS["mosaic"]
    if {"trajectory", "x", "y", "frame"} <= set(names):
        return DIALECTS["trajectory"]
    raise TrackTableError(
        f"{path}: its header ({', '.join(names[:8])}) matches no known dialect "
        f"({', '.join(sorted(DIALECTS))}); declare its columns and units in "
        f"{TABLE_FOLDER}/{FORMAT_FILE}")


def _trackmate_units(path: Path) -> str | None:
    """TrackMate's export carries three rows under the header (a name, a short name, the units):
    the units row of POSITION_X, when there is one."""
    table = pd.read_csv(path, nrows=4, dtype=str, keep_default_na=False)
    lookup = {str(c).strip().upper(): c for c in table.columns}
    column = lookup.get("POSITION_X")
    if column is None:
        return None
    for value in table[column].tolist()[:3]:
        word = _UNIT_WORDS.get(str(value).strip().lower())
        if word:
            return word
    return None


def read_track_table(path: str | Path, *, movie: str | None = None,
                     pixel_size_nm: float | None = None, declared: dict | None = None) -> pd.DataFrame:
    """One table as the shared raw table (`ingest.RAW_COLUMNS`): pixels, 1-based frames, one
    ``track_id`` per track; ``attrs``: the dialect, the units read and how they were known, the
    frame origin, and the untracked spots dropped."""
    from .ingest import RAW_COLUMNS

    path = Path(path)
    movie = movie if movie is not None else path.stem
    declared = declared if declared is not None else folder_format(path.parent)
    dialect = detect_dialect(path, declared)
    try:
        if dialect.header:
            table = pd.read_csv(path, sep=None if dialect.name == "declared" else dialect.sep,
                                engine="python", dtype=str, keep_default_na=False,
                                encoding="utf-8-sig")
        else:
            table = pd.read_csv(path, sep=dialect.sep, header=None, dtype=str,
                                keep_default_na=False, encoding="utf-8-sig")
    except Exception as exc:                                     # noqa: BLE001 - named
        raise TrackTableError(f"{path}: cannot be read as a table ({exc})") from exc

    def column(role):
        key = dialect.columns[role]
        if isinstance(key, int):
            if key >= table.shape[1]:
                raise TrackTableError(f"{path}: no column {key} for {role}")
            return table.iloc[:, key]
        lookup = {str(c).strip(): c for c in table.columns}
        lookup.update({str(c).strip().upper(): c for c in table.columns})
        hit = lookup.get(key) or lookup.get(str(key).upper())
        if hit is None:
            raise TrackTableError(f"{path}: no column {key!r} for {role}")
        return table[hit]

    track = column("track").astype(str).str.strip()
    frame = pd.to_numeric(column("frame"), errors="coerce")
    x = pd.to_numeric(column("x"), errors="coerce")
    y = pd.to_numeric(column("y"), errors="coerce")
    # TrackMate's three descriptive rows read as text: rows with no number in FRAME are not spots
    rows = frame.notna()
    if dialect.name == "trackmate_csv":
        rows &= ~track.isin(dialect.untracked)
    untracked = int((frame.notna() & ~rows).sum())
    if not rows.any():
        out = pd.DataFrame(columns=RAW_COLUMNS)
        out.attrs.update({"source": f"table:{dialect.name}", "untracked_spots": untracked})
        return out
    if x[rows].isna().any() or y[rows].isna().any():
        bad = int((x[rows].isna() | y[rows].isna()).sum())
        raise TrackTableError(f"{path}: {bad} row(s) whose position is not a number")

    units = declared.get("units") or dialect.units
    how = "declared" if declared.get("units") else ("the dialect" if dialect.units else None)
    if units is None and dialect.name == "trackmate_csv":
        units, how = _trackmate_units(path), "its units row"
    if units is None:
        raise TrackTableError(
            f"{path}: the {dialect.name} table does not say whether its positions are pixels, µm "
            f"or nm, and a wrong unit changes every D by the square of the factor. Declare it in "
            f"{path.parent / FORMAT_FILE}, e.g. {{\"units\": \"um\"}}")
    scale = UNITS[units]
    if scale is not None:
        if not pixel_size_nm:
            raise TrackTableError(
                f"{path}: positions in {units} need the pixel size to become pixels; it comes from "
                f"crops.json or the project's pixel size, and none was given")
        to_px = scale / float(pixel_size_nm)
    else:
        to_px = 1.0

    origin = int(declared.get("frame_origin", dialect.frame_origin))
    out = pd.DataFrame({
        "movie": movie,
        "track_id": track[rows].to_numpy(),
        "frame": (frame[rows].to_numpy(float) - origin + 1).round().astype(int),
        "x": x[rows].to_numpy(float) * to_px,
        "y": y[rows].to_numpy(float) * to_px,
        "amp": np.nan, "dx": np.nan, "dy": np.nan,
    })
    twice = out.duplicated(["track_id", "frame"])
    if twice.any():
        first = out.loc[twice].iloc[0]
        raise TrackTableError(
            f"{path}: track {first['track_id']} holds frame {int(first['frame'])} twice "
            f"({int(twice.sum())} such row(s)); a track is one position per frame")
    out = out.sort_values(["track_id", "frame"], kind="stable").reset_index(drop=True)
    out = out[RAW_COLUMNS]
    out.attrs.update({"source": f"table:{dialect.name}", "dialect_source": dialect.source,
                      "table_units": units, "units_from": how, "frame_origin": origin,
                      "untracked_spots": untracked})
    return out
