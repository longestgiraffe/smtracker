"""One interface over both trackers: u-track and TrackMate, same table out.

Everything downstream of ingest is **source-agnostic**: grouping, the canonical
table, observables, every fit and every map consume ``ingest.RAW_COLUMNS`` and
never ask which tracker produced it. This module is the single place that knows
the difference.

Design commitment
-----------------
u-track is **canonical and permanent**: the reference data is u-track and must stay
readable forever. TrackMate is added *alongside* it, never as a replacement.
Adding a third tracker later means adding a reader here and changing nothing
else.

Both readers emit **pixels and frame numbers**, with no unit conversion and no
length filter, because those live exactly once downstream
(``smtracker.canonical``).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .ingest import IngestError, read_utrack_mat
from .trackmate import TrackMateError, read_trackmate_csv, read_trackmate_xml

__all__ = ["ReaderError", "SOURCES", "detect_source", "read_tracks"]

#: Tracker formats understood here.
SOURCES = ("utrack", "trackmate_xml", "trackmate_csv")


class ReaderError(RuntimeError):
    """Raised when a track file's format cannot be determined or read."""


def detect_source(path: str | Path) -> str:
    """Identify the tracker format from the file itself, not just its name.

    ``.mat`` → u-track. ``.xml`` → TrackMate (verified by looking for a
    ``<TrackMate>``/``<Model>`` root rather than trusting the extension).
    ``.csv``/``.txt`` → a TrackMate spot table.
    """
    path = Path(path)
    if not path.exists():
        raise ReaderError(f"track file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".mat":
        return "utrack"
    if suffix in (".csv", ".txt", ".tsv"):
        return "trackmate_csv"
    if suffix == ".xml":
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                head = handle.read(4096)
        except OSError as exc:
            raise ReaderError(f"{path}: cannot be read: {exc}") from exc
        if "TrackMate" in head or "<Model" in head:
            return "trackmate_xml"
        raise ReaderError(
            f"{path}: XML without a TrackMate/Model root: unrecognised format."
        )
    raise ReaderError(
        f"{path}: unrecognised extension {suffix!r}. Understood: .mat (u-track), "
        ".xml (TrackMate), .csv (TrackMate spot table)."
    )


def read_tracks(
    path: str | Path,
    *,
    source: str | None = None,
    movie: str | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Read any supported tracker output into the shared raw table.

    Parameters
    ----------
    path
        The tracker output file.
    source
        Force a format instead of detecting it. One of ``SOURCES``.
    movie
        Identifier for the ``movie`` column; defaults to the file stem.
    **kwargs
        Passed to the underlying reader: e.g. ``var_name`` for u-track, or
        ``pixel_size_nm`` / ``filtered_only`` for TrackMate.

    Returns
    -------
    pandas.DataFrame
        ``ingest.RAW_COLUMNS`` in **pixels** and frame numbers, with
        ``.attrs["source"]`` recording which reader produced it.
    """
    source = source or detect_source(path)
    if source not in SOURCES:
        raise ReaderError(f"source must be one of {SOURCES}, got {source!r}")

    readers = {
        "utrack": read_utrack_mat,
        "trackmate_xml": read_trackmate_xml,
        "trackmate_csv": read_trackmate_csv,
    }
    try:
        table = readers[source](path, movie=movie, **kwargs)
    except (IngestError, TrackMateError) as exc:
        raise ReaderError(f"{path}: {source} reader failed: {exc}") from exc

    table.attrs.setdefault("source", source)
    return table
