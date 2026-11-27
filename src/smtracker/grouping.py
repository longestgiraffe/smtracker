"""Grouping engine: assign every track to condition / day / movie / cell.

The ingest adapters (``smtracker.ingest``) return a *raw* table with empty
``condition, day, cell`` placeholders. This module fills them in, and nothing
else: no units, no filtering (those live in ``smtracker.canonical``).

Invariants
----------
- ``movie`` is set by ingest (one movie = one tracker output file, e.g.
  ``series003``). It is never invented here.
- **Series → day** default mapping: series 001–010 = day 1, 011–020 = day 2, …
  i.e. ``day = (series - 1) // series_per_day + 1``. Fully editable: pass an
  explicit ``day_map`` (series number → day) or a per-movie mapping.
  **The default REFUSES where it is not demonstrably right.** It numbers a day by
  its *absolute* block of series numbers, whereas the lab's naming authority
  numbers the blocks that are *present*, in order. Those agree only when the
  blocks present are contiguous from the first; when they are not (series
  001–020 + 031–040 → default days 1, 2, **4**) the two numberings disagree and
  joining tables built each way silently drops the days that do not match. So
  the default raises in exactly that case and asks for an explicit mapping,
  see ``_refuse_non_contiguous_blocks``.
- **Conditions are unlimited**: no 7-condition ceiling (a legacy MATLAB bug).
  ``ConditionSet`` carries display order and colors; it does not restrict how
  many conditions exist.
- **Fail loud**: a movie that no mapping covers raises, rather than silently
  landing in an ``NaN``/"unknown" bucket that would quietly corrupt a fit.
- Grouping does not touch localizations: row count and coordinates are
  unchanged; only the three grouping columns are written.

Why day matters
---------------
Per ``CLAUDE.md``, the "simultaneous" SQD fit shares D across **replicate days**
of the same protein+condition and fits per-day fractions; bootstrap error bars
are taken over independent units (cells / days). Day assignment is therefore a
scientific input, not cosmetics: hence it is explicit and inspectable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from .ingest import (EmptyTrackFileError, GROUPING_PLACEHOLDERS, IngestError,
                     read_utrack_mat)

__all__ = [
    "GroupingError",
    "ConditionSet",
    "series_number",
    "day_from_series",
    "day_map_from_series_blocks",
    "day_maps_from_series_naming",
    "assign_groups",
    "discover_utrack_project",
    "load_utrack_project",
    "TRACKMATE_GLOB",
    "sources_present",
    "discover_trackmate_project",
    "load_trackmate_project",
    "PROJECT_SOURCES",
    "detect_project_source",
    "grouping_summary",
]

_SERIES_RE = re.compile(r"(\d+)")

#: u-track project layout. What is fixed is the *tail*:
#: ``<movie>/TrackingPackage/tracks/*.mat``. What is above ``<movie>`` is not,
#: u-track is often run in whatever folder the data happens to sit in, and that
#: folder is frequently renamed or dropped when the project is moved. So the glob
#: is recursive and ``movie`` is taken from the folder three levels above the
#: ``.mat``; nothing depends on the names of the folders above it.
UTRACK_GLOB = "**/TrackingPackage/tracks/*.mat"

#: Never treat u-track's own ``backups/`` copies as data.
_EXCLUDED_PARTS = frozenset({"backups"})
#: The longest folder path Windows lists for a program that has not opted in to long
#: paths (MAX_PATH 260, less the 12 a folder must leave for an 8.3 file name).
LONGEST_LISTABLE_FOLDER = 248


def _find_tails(root: Path, pattern: str) -> list[Path]:
    """Every file under ``root`` matching a ``**/<folder>[/<folder>]/*.<ext>`` pattern,
    found by ONE walk that also notices the folders it could not list.

    **A FOLDER WHOSE PATH IS TOO LONG FOR WINDOWS VANISHES FROM A GLOB WITHOUT A WORD.**
    Found 12 Oct 2026 on a scratch copy: every result path was over 260 characters and
    `glob` returned nothing, so a count read 0 detections everywhere. In a project that is
    a movie missing from the analysis with nothing saying so. The walk replaces the glob
    (one pass either way) and REFUSES by name when a folder could not be listed for its
    length. Other listing errors are ignored, as the glob ignored them. A pattern of
    another shape falls back to `glob`.
    """
    import fnmatch
    import os

    parts = pattern.split("/")
    if not (len(parts) in (3, 4) and parts[0] == "**" and "*" not in "".join(parts[1:-1])):
        return sorted(root.glob(pattern))
    tail = [x.lower() for x in parts[1:-1]]
    name_pattern = parts[-1].lower()
    excluded = {x.lower() for x in _EXCLUDED_PARTS}
    too_long: list[str] = []

    def noted(error):
        name = getattr(error, "filename", None) or ""
        if len(str(name)) >= LONGEST_LISTABLE_FOLDER:
            too_long.append(str(name))

    hits = []
    for folder, dirs, files in os.walk(root, onerror=noted):
        dirs[:] = [d for d in dirs if d.lower() not in excluded]
        here = Path(folder)
        names = [here.name.lower()] if len(tail) == 1 else [here.parent.name.lower(),
                                                             here.name.lower()]
        if names == tail:
            hits += [here / f for f in files if fnmatch.fnmatch(f.lower(), name_pattern)]
    if too_long:
        shown = "; ".join(too_long[:3]) + (" ..." if len(too_long) > 3 else "")
        raise GroupingError(
            f"{len(too_long)} folder(s) under {root} could not be read because their path "
            f"is longer than Windows allows ({LONGEST_LISTABLE_FOLDER} characters for a "
            f"folder): {shown}. The movies inside would be missing. Move "
            f"the project to a shorter folder, or turn on long paths in Windows.")
    return sorted(hits)


#: macOS AppleDouble sidecars (._name) shadow every file when a Mac copies to a
#: non-HFS volume. They carry the same extension, so a *.mat glob finds them, and
#: they are never data, skipped at discovery like backups/.


class GroupingError(RuntimeError):
    """Raised when tracks cannot be assigned to condition / day / cell."""


@dataclass
class ConditionSet:
    """Ordered, recolorable set of conditions (unlimited count).

    Display concern only: it fixes plot/report order and colors. Analysis never
    depends on how many conditions there are.
    """

    names: list[str]
    colors: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(set(self.names)) != len(self.names):
            raise GroupingError(f"duplicate condition names: {self.names}")
        unknown = set(self.colors) - set(self.names)
        if unknown:
            raise GroupingError(f"colors given for unknown conditions: {sorted(unknown)}")

    def rename(self, old: str, new: str) -> "ConditionSet":
        """Return a copy with ``old`` renamed to ``new`` (order and color kept)."""
        if old not in self.names:
            raise GroupingError(f"unknown condition '{old}'; have {self.names}")
        names = [new if n == old else n for n in self.names]
        colors = {(new if k == old else k): v for k, v in self.colors.items()}
        return ConditionSet(names, colors)

    def reorder(self, order: Iterable[str]) -> "ConditionSet":
        """Return a copy in the given order (must be a permutation of ``names``)."""
        order = list(order)
        if sorted(order) != sorted(self.names):
            raise GroupingError(
                f"reorder must be a permutation of {self.names}, got {order}"
            )
        return ConditionSet(order, dict(self.colors))

    def recolor(self, colors: Mapping[str, str]) -> "ConditionSet":
        """Return a copy with ``colors`` merged in."""
        return ConditionSet(list(self.names), {**self.colors, **dict(colors)})

    def as_categorical(self, series: pd.Series) -> pd.Series:
        """Cast a condition column to an ordered categorical in this order."""
        missing = set(series.dropna().unique()) - set(self.names)
        if missing:
            raise GroupingError(
                f"conditions present in data but not in the set: {sorted(missing)}"
            )
        return pd.Categorical(series, categories=self.names, ordered=True)


def series_number(movie: str) -> int:
    """Extract the series number from a movie name (``"series003"`` → ``3``).

    Uses the first run of digits. Fails loud if the name holds none: guessing
    would silently mis-assign replicate days.
    """
    m = _SERIES_RE.search(str(movie))
    if m is None:
        raise GroupingError(
            f"cannot read a series number from movie name '{movie}'; "
            "pass an explicit replicate mapping instead."
        )
    return int(m.group(1))


def day_from_series(series: int, series_per_day: int = 10) -> int:
    """Default series→day rule: 001–010 = day 1, 011–020 = day 2, …"""
    if series < 1:
        raise GroupingError(f"series number must be >= 1, got {series}")
    if series_per_day < 1:
        raise GroupingError(f"series_per_day must be >= 1, got {series_per_day}")
    return (series - 1) // series_per_day + 1


def day_map_from_series_blocks(movies, series_per_day: int = 10):
    """``{movie: day}`` and ``{day: "seriesNNN-NNN"}`` from the series numbering.

    The lab's own convention, per the experimenter: series 001-010 are replicate day 1,
    011-020 day 2, and a third block may be 021-030 or 031-040. So the rule is "one
    block of ``series_per_day`` series numbers = one replicate day", and **the blocks
    that are PRESENT are numbered in the order they occur**, which handles a missing
    block without a special case and without assuming a fixed range.

    This is the rule ``scripts/msd_batch.day_map_from_series_naming`` has always
    applied; it lives here now because a *package* function cannot import from
    ``scripts/``, and AW5 needs one. ``msd_batch`` delegates to this, so there is still
    exactly one implementation. Verified identical on all ten real groups.

    Chosen over any timestamp deliberately. The naming is what the experimenter
    controls and records; a file's mtime is a proxy that a copy or a restore destroys,
    and ``movieData``'s ``createTime_`` is the PROCESSING date (it merges sessions
    processed together and splits sessions processed twice, and disagrees with
    acquisition in 2 of 10 groups here). The naming is stable under both.

    Contrast ``day_from_series``, which numbers a day by its ABSOLUTE block. The two
    agree only when the blocks present are contiguous from the first, see
    ``_refuse_non_contiguous_blocks`` for what happened when they did not.
    """
    if series_per_day < 1:
        raise GroupingError(f"series_per_day must be >= 1, got {series_per_day}")
    movies = [str(m) for m in movies]
    if not movies:
        raise GroupingError("no movies to map to replicates")
    numbers = {m: series_number(m) for m in movies}
    blocks = {m: (n - 1) // series_per_day for m, n in numbers.items()}
    order = {b: i + 1 for i, b in enumerate(sorted(set(blocks.values())))}
    movie_to_day = {m: order[b] for m, b in blocks.items()}
    spans: dict[int, tuple[int, int]] = {}
    for movie, day in movie_to_day.items():
        lo, hi = spans.get(day, (10 ** 9, 0))
        spans[day] = (min(lo, numbers[movie]), max(hi, numbers[movie]))
    day_to_span = {d: f"series{lo:03d}-{hi:03d}" for d, (lo, hi) in sorted(spans.items())}
    return movie_to_day, day_to_span


def _refuse_non_contiguous_blocks(day_by_movie: Mapping[str, int],
                                  series_per_day: int) -> None:
    """Refuse the DEFAULT series→day rule where it disagrees with the naming authority.

    The defect this exists for (D11). The default numbers a day by its ABSOLUTE block
    of series numbers: ``day = (series - 1) // series_per_day + 1``. The lab's naming
    authority (``day_map_from_series_naming``) numbers the blocks that are PRESENT, in
    the order they occur. The two agree exactly when the blocks present are contiguous
    from the first one, and silently disagree when they are not:

        protein C/Glucose   series 001-020, 031-040  ->  authority 1, 2, 3   default 1, 2, 4
        protein A/Malate     series 001-009, 031-050  ->  authority 1, 2, 3   default 1, 4, 5

    Joining a table numbered one way against a table numbered the other drops the days
    that do not match. When that happened, **protein A/Malate's maps were built from one
    replicate day of three** and rendered as a normal three-day panel.

    So the default is kept where it is demonstrably right and REFUSES where it is not.
    It cannot silently renumber instead: that would change the meaning of ``day`` for
    every existing caller, and a caller that wants the authority's numbering should say
    so, because only the caller knows which table it will be joined against.

    Not reached when the caller passes ``day=`` explicitly, an explicit mapping IS a
    stated authority, which is the whole point.
    """
    present = sorted(set(day_by_movie.values()))
    if present == list(range(1, len(present) + 1)):
        return
    missing = [d for d in range(present[0], present[-1] + 1) if d not in present]
    lo = (present[0] - 1) * series_per_day + 1
    raise GroupingError(
        f"the default series->replicate rule gives replicates {present} for these movies, which is "
        f"not a contiguous run starting at replicate 1 "
        + (f"(missing {missing}; " if missing else "(")
        + f"first series block starts at series{lo:03d}). The default numbers replicates by "
        f"absolute blocks of {series_per_day} series, so joining the two tables would drop "
        "replicates that do not match. Pass an explicit replicate= mapping that numbers "
        "the blocks present in the order they occur."
    )


def assign_groups(
    df: pd.DataFrame,
    *,
    condition: str | Mapping[str, str],
    protein: str | Mapping[str, str] | None = None,
    day: int | Mapping[str, int] | None = None,
    series_per_day: int = 10,
    cell: str | Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Fill ``protein``, ``condition``, ``day`` and ``cell`` on a raw track table.

    Parameters
    ----------
    df
        Raw table from an ingest adapter (must have a ``movie`` column).
    condition
        A single condition name for every movie, or a ``movie -> condition``
        mapping. Every movie present must be covered (fail loud otherwise).
    protein
        The molecular species, as a name or ``movie -> protein`` mapping.
        Optional but strongly recommended when several proteins will be analysed
        together: the number of diffusive states is a property of the molecule,
        so fits are grouped per protein (``smtracker.compare``). Left ``<NA>``
        when a dataset holds a single, unnamed protein.
    day
        ``None`` (default) → derive from the series number in the movie name
        using ``series_per_day``. Or a single day number, or a
        ``movie -> day`` mapping.
    series_per_day
        Series per replicate day for the default rule (default 10).
    cell
        Optional single cell id or ``movie -> cell`` mapping. Left as ``<NA>``
        when unknown: cell assignment normally comes from segmentation, which
        is not wired up yet. Downstream code that needs cells must check.

    Returns
    -------
    pandas.DataFrame
        A copy with the grouping columns filled; all other columns untouched.
    """
    if "movie" not in df.columns:
        raise GroupingError("raw table has no 'movie' column; came from where?")
    out = df.copy()
    movies = list(pd.unique(out["movie"]))

    out["condition"] = _resolve(out["movie"], condition, movies, "condition")

    if protein is None:
        out["protein"] = pd.Series([pd.NA] * len(out), dtype="object", index=out.index)
    else:
        out["protein"] = _resolve(out["movie"], protein, movies, "protein")

    if day is None:
        day_by_movie = {m: day_from_series(series_number(m), series_per_day) for m in movies}
        _refuse_non_contiguous_blocks(day_by_movie, series_per_day)
        out["day"] = out["movie"].map(day_by_movie)
    else:
        out["day"] = _resolve(out["movie"], day, movies, "day")
    out["day"] = out["day"].astype("int64")

    if cell is None:
        out["cell"] = pd.Series([pd.NA] * len(out), dtype="object", index=out.index)
    else:
        out["cell"] = _resolve(out["movie"], cell, movies, "cell")

    _check_filled(out, ["condition", "day"])
    return out


def _resolve(movie_col: pd.Series, spec, movies: list, what: str) -> pd.Series:
    """Broadcast a scalar, or map a ``movie -> value`` mapping (fail loud on gaps)."""
    if isinstance(spec, Mapping):
        missing = [m for m in movies if m not in spec]
        if missing:
            raise GroupingError(
                f"{what} mapping does not cover these movies: {missing}. "
                "Every movie must be assigned explicitly."
            )
        return movie_col.map(dict(spec))
    return pd.Series([spec] * len(movie_col), index=movie_col.index)


def _check_filled(df: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        if df[col].isna().any():
            bad = sorted(pd.unique(df.loc[df[col].isna(), "movie"]))
            raise GroupingError(f"'{col}' unassigned for movies: {bad}")


def _from_day_folders(root, finder, pattern: str, what: str):
    """``finder`` over each DAY FOLDER of ``root`` (`dayfolders.day_folders`), each movie named for its folder
    (``series001 (rep1)``), or None when ``root`` is not a day-folder condition. A day folder without this tracker's
    output while another has it is REFUSED by name: loading the days that have it would drop a replicate day without
    a word."""
    from .dayfolders import day_folders, movie_name

    folders = day_folders(root)
    if folders is None:
        return None
    out, missing = [], []
    for folder in folders:
        try:
            found = finder(folder, pattern=pattern)
        except GroupingError:
            missing.append(folder.name)
            continue
        out += [(movie_name(movie, folder.name), path) for movie, path in found]
    if not out:
        raise GroupingError(f"no {what} in any replicate folder of {root} "
                            f"({', '.join(f.name for f in folders)})")
    if missing:
        raise GroupingError(
            f"{root} keeps its replicates in sub-folders, and {', '.join(missing)} hold(s) no {what} while "
            f"the others do. Track every replicate the same way, or move that folder out of the condition.")
    return sorted(out, key=lambda mp: mp[0])


def _day_folder_days(root, day, found):
    """The day argument a loader uses: the caller's, else a day-folder condition's own (the k-th folder is day k),
    else None (the series rule)."""
    if day is not None:
        return day
    from .dayfolders import day_map

    try:
        return day_map(root, [movie for movie, _path in found])
    except ValueError as exc:
        raise GroupingError(str(exc)) from None


def discover_utrack_project(
    root: str | Path, *, pattern: str = UTRACK_GLOB
) -> list[tuple[str, Path]]:
    """Find u-track result files under a project folder.

    Searches recursively, so it does not care how the movie folders are nested
    or what the folders above them are called. Returns ``[(movie, path), ...]``
    sorted by movie name, where ``movie`` is the movie folder name (e.g.
    ``"series003"``): the folder holding ``TrackingPackage``. u-track's own
    ``backups/`` copies are ignored. Fails loud if nothing matches or if two
    movie folders share a name: an empty or ambiguous project is a mistake,
    not a valid dataset.
    """
    root = Path(root)
    if not root.is_dir():
        raise GroupingError(f"project folder not found: {root}")
    # REPLICATE DAYS AS SUB-FOLDERS that start their movie names again (`dayfolders`): their movies, by day folder
    in_days = _from_day_folders(root, discover_utrack_project, pattern, "u-track results")
    if in_days is not None:
        return in_days
    from .ingest import is_sidecar

    hits = [
        p for p in _find_tails(root, pattern)
        if not _EXCLUDED_PARTS.intersection(p.parts) and not is_sidecar(p)
    ]
    if not hits:
        raise GroupingError(
            f"no u-track results under {root} matching '{pattern}'. Expected "
            "<movie>/TrackingPackage/tracks/*.mat somewhere beneath it."
        )
    # .../<movie>/TrackingPackage/tracks/<file>.mat → movie folder is 3 levels up
    out = [(p.parents[2].name, p) for p in hits]

    seen: dict[str, Path] = {}
    for movie, path in out:
        if movie in seen:
            raise GroupingError(
                f"two movie folders are both named '{movie}': {seen[movie]} and "
                f"{path}. Movie names must be unique within a project."
            )
        seen[movie] = path
    return sorted(out, key=lambda mp: mp[0])


def load_utrack_project(
    root: str | Path,
    *,
    condition: str | Mapping[str, str],
    protein: str | Mapping[str, str] | None = None,
    day: int | Mapping[str, int] | None = None,
    series_per_day: int = 10,
    cell: str | Mapping[str, str] | None = None,
    pattern: str = UTRACK_GLOB,
    on_movie=None,
) -> pd.DataFrame:
    """Read every u-track result under ``root`` and assign grouping.

    Convenience over ``discover_utrack_project`` + ``read_utrack_mat`` +
    ``assign_groups``. Still raw: pixels, frames, no length filter.

    ``on_movie`` (optional) is called ``on_movie(movie, done, total)`` after each
    file, for a window that has to say what it is doing. THIS loop is where the
    seconds are -- one ``.mat`` per movie -- and it is the only place with a
    count to report: a caller watching from outside sees one condition begin and
    end, which on a single-condition project is a progress line that says
    "1 of 1" and then nothing for six seconds. Never fatal; a handler that
    raises is a window closing mid-load, not a reason to lose the read.
    """
    found = discover_utrack_project(root, pattern=pattern)
    day = _day_folder_days(root, day, found)
    frames, empty = [], []
    for done, (movie, path) in enumerate(found, start=1):
        if on_movie is not None:
            try:
                on_movie(movie, done, len(found))
            except Exception:                          # noqa: BLE001 - progress
                pass
        try:
            frames.append(read_utrack_mat(path, movie=movie))
        except EmptyTrackFileError:
            # The tracker produced no tracks for this movie. That is DATA: one bad
            # field of view must not kill a 40-movie project, but it is never
            # silent: the movies are recorded on the frame and reported by
            # `empty_movies`, because a condition that quietly lost a third of its
            # movies is a different experiment from the one the analyst ran.
            empty.append(movie)
    if not frames:
        raise IngestError(
            f"{root}: every discovered movie is empty: the tracker produced no "
            f"tracks anywhere ({len(empty)} movies). Nothing to analyse.")
    raw = pd.concat(frames, ignore_index=True)
    out = assign_groups(
        raw, condition=condition, protein=protein, day=day,
        series_per_day=series_per_day, cell=cell,
    )
    out.attrs["empty_movies"] = sorted(empty)
    out.attrs["n_movies_read"] = len(frames)
    return out


#: TrackMate project layout. `trackmate_run.run_batch` writes ONE XML PER MOVIE into a
#: flat folder, named after the movie (`<output_dir>/<movie>.xml`), and `tracking_gui`
#: puts that folder at `<project>/trackmate`. So unlike u-track: where the movie is the
#: folder three levels above the file: here the movie IS the file stem.
#:
#: The glob is recursive for the same reason u-track's is: a project may hold one
#: `trackmate/` folder per condition, and the folders above carry no meaning.
TRACKMATE_GLOB = "**/trackmate/*.xml"


def discover_trackmate_project(
    root: str | Path, *, pattern: str = TRACKMATE_GLOB
) -> list[tuple[str, Path]]:
    """Find TrackMate XMLs under a project folder.

    The mirror of :func:`discover_utrack_project`, and deliberately the same shape:
    ``[(movie, path), ...]`` sorted by movie, failing loud on an empty project and on
    two movies sharing a name. Everything downstream is source-agnostic, so the only
    thing that may differ between the two trackers is how the files are found.

    ``movie`` is the XML's STEM, because that is what `run_batch` names it after. Taking
    it from the parent folder instead would give every movie in a batch the name
    ``trackmate``.
    """
    root = Path(root)
    if not root.is_dir():
        raise GroupingError(f"project folder not found: {root}")
    in_days = _from_day_folders(root, discover_trackmate_project, pattern, "TrackMate results")
    if in_days is not None:
        return in_days
    from .ingest import is_sidecar

    hits = [
        p for p in _find_tails(root, pattern)
        if not _EXCLUDED_PARTS.intersection(p.parts) and not is_sidecar(p)
    ]
    if not hits:
        raise GroupingError(
            f"no TrackMate results under {root} matching '{pattern}'. Expected "
            "<...>/trackmate/<movie>.xml somewhere beneath it: the layout "
            "`trackmate_run.run_batch` writes."
        )
    # UNDER MOVIE NAMES (`rawmovie.movie_name`): an XML tracked from a MetaMorph file is
    # `series001_YFPLaser.xml` while its outline is `series001.mat`.
    from .rawmovie import claimed_twice, condition_folder_of, movie_name

    # ONE MOVIE TRACKED TWICE IS REFUSED BY NAME. A run under the old naming leaves
    # `series001_YFPLaser.xml` beside the new `series001.xml`; kept apart (two files never
    # merge) the old one reads as an extra movie with no outline, which is how it was
    # reported: "it doesn't find the correct mesh for the raw stk file".
    by_folder: dict[Path, list[Path]] = {}
    for p in hits:
        by_folder.setdefault(p.parent, []).append(p)
    twice = []
    for folder, files in by_folder.items():
        for name, stems in claimed_twice([p.stem for p in files],
                                         condition_folder_of(files[0])).items():
            twice.append(f"{folder}: {', '.join(s + '.xml' for s in stems)} are all "
                         f"movie '{name}'")
    if twice:
        raise GroupingError(
            "the same movie was tracked more than once: " + "; ".join(twice[:4])
            + ". Keep one track file per movie: move the others into a 'backups' "
              "folder beside them (trackmate/backups/), which is never read.")

    out = [(movie_name(p), p) for p in hits]

    seen: dict[str, Path] = {}
    for movie, path in out:
        if movie in seen:
            raise GroupingError(
                f"two TrackMate XMLs are both named '{movie}': {seen[movie]} and "
                f"{path}. Movie names must be unique within a project."
            )
        seen[movie] = path
    return sorted(out, key=lambda mp: mp[0])


def load_trackmate_project(
    root: str | Path,
    *,
    condition: str | Mapping[str, str],
    protein: str | Mapping[str, str] | None = None,
    day: int | Mapping[str, int] | None = None,
    series_per_day: int = 10,
    cell: str | Mapping[str, str] | None = None,
    pattern: str = TRACKMATE_GLOB,
    pixel_size_nm: float | None = None,
    filtered_only: bool = True,
) -> pd.DataFrame:
    """Read every TrackMate XML under ``root`` and assign grouping.

    The second tracker reaching the analysis spine. `trackmate_run` produced XML and
    `trackmate.read_trackmate_xml` read it, but no PROJECT loader existed, so the only
    route from a tracking run to an analysis was a hand-written script: the launcher
    offered a Tracking stage whose output the Analysis stage could not open.

    Returns exactly what :func:`load_utrack_project` returns, raw pixels and frame
    numbers, no unit conversion and no length filter, with grouping assigned. Everything
    downstream consumes ``ingest.RAW_COLUMNS`` and never asks which tracker produced it.

    ``pixel_size_nm`` is passed through and NOT guessed. SMTracker's own runs force the
    calibration to 1 pixel = 1 unit before configuring anything, so those files need no
    conversion; an XML from elsewhere may be µm-calibrated, and a wrong pixel size
    rescales every D by its square (the 106-vs-160 nm trap costs a factor 2.28).
    """
    from .trackmate import read_trackmate_xml

    frames, empty, unreadable = [], [], []
    #: movie -> its XML's linking distance in pixels (`trackmate.linking_radius_px`), the condition's search radius
    linking: dict = {}
    found = discover_trackmate_project(root, pattern=pattern)
    day = _day_folder_days(root, day, found)
    for movie, path in found:
        try:
            frame = read_trackmate_xml(
                path, movie=movie, pixel_size_nm=pixel_size_nm,
                filtered_only=filtered_only,
            )
            linking[str(movie)] = frame.attrs.get("linking_radius_px")
        except EmptyTrackFileError:
            empty.append(movie)
            continue
        except Exception as exc:                    # noqa: BLE001 - named, never hidden
            # A CORRUPT FILE IS NOT AN EMPTY ONE. u-track's loader lets an empty movie
            # through as data and kills the load on anything else; the same rule here,
            # except that an XML can also be truncated mid-write by an interrupted Fiji
            # run, which looks like a project that merely tracked badly.
            unreadable.append(f"{movie}: {type(exc).__name__}: {exc}")
            continue
        if frame.empty:
            # TrackMate reports "no tracks" by producing a file with none in it rather
            # than by failing, so the empty case arrives as an empty frame and must be
            # recorded in the same place u-track's EmptyTrackFileError is.
            empty.append(movie)
            continue
        frames.append(frame)

    if unreadable:
        raise IngestError(
            f"{root}: {len(unreadable)} TrackMate XML(s) could not be read: "
            + "; ".join(unreadable[:5])
            + (" ..." if len(unreadable) > 5 else "")
        )
    if not frames:
        raise IngestError(
            f"{root}: every discovered movie is empty: TrackMate produced no tracks "
            f"anywhere ({len(empty)} movies). Nothing to analyse.")

    raw = pd.concat(frames, ignore_index=True)
    out = assign_groups(
        raw, condition=condition, protein=protein, day=day,
        series_per_day=series_per_day, cell=cell,
    )
    out.attrs["empty_movies"] = sorted(empty)
    out.attrs["n_movies_read"] = len(frames)
    out.attrs["source"] = "trackmate_xml"
    # each movie's linking distance (pixels, None where its XML records none): `project.load_project` makes it the
    # condition's search radius (ruled 4 Nov 2026)
    out.attrs["linking_radius_px_by_movie"] = linking
    return out


#: Track tables from other tools (`smtracker.tracktable`): ``<condition>/track_tables/<movie>.csv``.
from .tracktable import TABLE_GLOB, TABLE_SUFFIXES  # noqa: E402


def discover_table_project(
    root: str | Path, *, pattern: str = TABLE_GLOB
) -> list[tuple[str, Path]]:
    """Find track tables under a project folder: ``[(movie, path), ...]`` sorted by movie, the
    mirror of `discover_trackmate_project` -- failing loud on an empty project and on two tables
    that are one movie. ``format.json`` and anything not a table are not movies."""
    root = Path(root)
    if not root.is_dir():
        raise GroupingError(f"project folder not found: {root}")
    in_days = _from_day_folders(root, discover_table_project, pattern, "track tables")
    if in_days is not None:
        return in_days
    from .ingest import is_sidecar
    from .rawmovie import movie_name

    hits = [p for p in _find_tails(root, pattern)
            if p.is_file() and p.suffix.lower() in TABLE_SUFFIXES
            and not _EXCLUDED_PARTS.intersection(p.parts) and not is_sidecar(p)]
    if not hits:
        raise GroupingError(
            f"no track tables under {root}: expected <...>/track_tables/<movie>.csv (or .tsv, "
            f".txt) somewhere beneath it.")
    out = [(movie_name(p), p) for p in hits]
    seen: dict[str, Path] = {}
    for movie, path in out:
        if movie in seen:
            raise GroupingError(
                f"two track tables are both movie '{movie}': {seen[movie]} and {path}. Keep one "
                f"per movie; others go in a 'backups' folder beside them, which is never read.")
        seen[movie] = path
    return sorted(out, key=lambda mp: mp[0])


def load_table_project(
    root: str | Path,
    *,
    condition: str | Mapping[str, str],
    protein: str | Mapping[str, str] | None = None,
    day: int | Mapping[str, int] | None = None,
    series_per_day: int = 10,
    cell: str | Mapping[str, str] | None = None,
    pattern: str = TABLE_GLOB,
    pixel_size_nm: float | None = None,
) -> pd.DataFrame:
    """Read every track table under ``root`` and assign grouping: exactly what
    `load_trackmate_project` returns (raw pixels and frame numbers, no unit conversion beyond
    the table's own units, no length filter). A table that cannot be read stops the load,
    named; an empty one is recorded as an empty movie."""
    from .tracktable import TrackTableError, read_track_table

    frames, empty, unreadable, dialects, untracked = [], [], [], {}, 0
    found = discover_table_project(root, pattern=pattern)
    day = _day_folder_days(root, day, found)
    for movie, path in found:
        try:
            frame = read_track_table(path, movie=movie, pixel_size_nm=pixel_size_nm)
        except TrackTableError as exc:
            unreadable.append(f"{movie}: {exc}")
            continue
        untracked += int(frame.attrs.get("untracked_spots", 0))
        if frame.empty:
            empty.append(movie)
            continue
        dialects[movie] = frame.attrs.get("source")
        frames.append(frame)
    if unreadable:
        raise IngestError(
            f"{root}: {len(unreadable)} track table(s) could not be read -- "
            + "; ".join(unreadable[:3]) + (" ..." if len(unreadable) > 3 else ""))
    if not frames:
        raise IngestError(f"{root}: every track table is empty ({len(empty)} movies). "
                          f"Nothing to analyse.")
    raw = pd.concat(frames, ignore_index=True)
    out = assign_groups(raw, condition=condition, protein=protein, day=day,
                        series_per_day=series_per_day, cell=cell)
    out.attrs["empty_movies"] = sorted(empty)
    out.attrs["n_movies_read"] = len(frames)
    out.attrs["source"] = "table"
    out.attrs["table_dialects"] = dialects
    out.attrs["untracked_spots"] = untracked
    return out


#: How a project folder announces which tracker made it. The tails are disjoint, so
#: this is a look rather than a guess. "table" is a track table from any other tool
#: (`smtracker.tracktable`), in its own folder.
PROJECT_SOURCES = ("utrack", "trackmate", "table")


def sources_present(root) -> dict:
    """``{"utrack": n, "trackmate": n}`` for the trackers that left files under `root`.

    The counting, without the refusal. The function below is right to refuse a folder
    tracked twice: two datasets with one name, differing by about 1 pp in fractions and
    10-18 per cent in D, but a refusal is not the only honest answer to a choice. A
    library with no user in front of it must refuse; a window can ask, which is what the
    analysis door now does. Both need the same counting, so it lives here once.

    Empty when nothing is under `root`; raises only when the folder does not exist.
    """
    from pathlib import Path

    from .ingest import is_sidecar

    root = Path(root)
    if not root.is_dir():
        raise GroupingError(f"project folder not found: {root}")
    found = {}
    for name, pattern in (("utrack", UTRACK_GLOB), ("trackmate", TRACKMATE_GLOB),
                          ("table", TABLE_GLOB)):
        hits = [p for p in root.glob(pattern)
                if not _EXCLUDED_PARTS.intersection(p.parts) and not is_sidecar(p)
                and (name != "table" or p.suffix.lower() in TABLE_SUFFIXES)]
        if hits:
            found[name] = len(hits)
    return found


def detect_project_source(root: str | Path) -> str:
    """Say which tracker's output a folder holds, or refuse.

    Refuses when both are present, by design. A folder tracked twice holds two datasets
    with one name, and this project has already recorded what that costs: the same movie
    read through u-track and through TrackMate gives fractions 0.8-1.5 pp apart and D
    values 10-18 per cent apart, so silently preferring one would change reported numbers
    with nothing on screen to say why. The caller must pass ``source=`` and say which.
    """
    found = sources_present(root)
    if not found:
        raise GroupingError(
            f"no tracker output under {root}. Expected "
            f"<movie>/TrackingPackage/tracks/*.mat (u-track), "
            f"<...>/trackmate/<movie>.xml (TrackMate) or <...>/track_tables/<movie>.csv "
            f"(another tool's track table) somewhere beneath it."
        )
    if len(found) > 1:
        counts = ", ".join(f"{k}: {v} file(s)" for k, v in sorted(found.items()))
        raise GroupingError(
            f"{root} holds output from both trackers ({counts}). They disagree: on the "
            f"same movies the fractions differ by about 1 pp and D by 10-18 per cent, so "
            f"none is chosen for you. Pass source='utrack' or source='trackmate'."
        )
    return next(iter(found))


def empty_movies(df: pd.DataFrame) -> list[str]:
    """Movies the tracker produced NO tracks for, recorded at load.

    Read this before comparing conditions: an empty movie contributes nothing to
    any statistic, so a condition that lost several of them has a smaller effective
    n than its movie count suggests, and the day-unit interval is computed on fewer
    movies than the folder listing implies.
    """
    return list(df.attrs.get("empty_movies", []))


def grouping_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per (condition, day, movie): track and localization counts.

    The first thing to look at after grouping: it makes a wrong series→day
    mapping obvious before any fit is run.
    """
    missing = [c for c in ["condition", "day", "movie"] if c not in df.columns]
    if missing:
        raise GroupingError(f"table is missing columns {missing}")
    if df[["condition", "day"]].isna().any().any():
        raise GroupingError("table still has unassigned condition/replicate; run assign_groups")
    return (
        df.groupby(["condition", "day", "movie"], observed=True)
        .agg(n_tracks=("track_id", "nunique"), n_localizations=("frame", "size"))
        .reset_index()
    )


def day_maps_from_series_naming(root, *, series_per_day: int = 10,
                                sources: dict | None = None) -> dict:
    """``{condition: {movie: day}}`` for every condition folder under ``root``.

    The rule was in the package and no caller could reach it. `day_map_from_series_blocks`
    is the lab's naming authority -- one block of series numbers is one replicate day, and
    the blocks PRESENT are numbered in the order they occur -- but every entry point
    except `scripts/msd_batch` fell to the DEFAULT rule, which numbers a day by its
    ABSOLUTE block. The two agree only when the blocks run contiguously from the first,
    and on this project they do not: protein A/Malate is series001-009, 031-040, 041-050, which
    the default calls days [1, 4, 5] and rightly refuses. The guard then printed an
    instruction naming a function the CLI had no way to call.

    Measured on protein A, all four conditions:

        Glu + Mal  30 movies  001-010 / 011-020 / 021-030
        Glucose    30 movies  001-010 / 011-020 / 021-030
        LB         30 movies  001-010 / 011-020 / 021-030
        Malate     29 movies  001-009 / 031-040 / 041-050

    so one rule covers both shapes without a special case.

    ``sources`` ``{condition: "utrack" | "trackmate"}`` numbers a condition's days from THE
    TRACKER THAT WILL BE READ. Without it u-track's movies are numbered when both are
    present -- and a condition tracked twice may hold DIFFERENT movies per tracker.
    Reported on a user's `B subtilis 3610 protein A mNG + protein B/Malate` folder: TrackMate tracked
    series011-016 and 021-040, u-track (in `tif 2/`) only 021-030; the days were numbered
    from u-track's ten, TrackMate was the tracker chosen, and the load refused "day mapping
    does not cover these movies: series011 ...". Deleting the cache could not help: the
    mapping is derived afresh on every open.

    ``series_per_day`` is 10 by default because that is this lab's convention; some use 8,
    which is why it is a parameter and why the loader prints the spans it derived rather
    than applying them silently. A day numbering that disagrees with the naming authority
    is the defect this exists to prevent, so it must be visible on every run.
    """
    from .project import discover_conditions

    out = {}
    for name, path in discover_conditions(root).items():
        # EITHER TRACKER. This read u-track only, and `discover_utrack_project` RAISES
        # where there is none rather than returning empty -- so the `if not movies`
        # below was unreachable and a TrackMate project raised out of the whole
        # derivation. The caller catches that and falls back to the DEFAULT rule, which
        # then refuses any project whose series do not start at 001: a folder tracked
        # here, by this tool, could not be opened by it. Measured on D:/data/project,
        # series 021-030, where the default calls the single block "day 3" and refuses.
        movies = []
        finders = {"utrack": (discover_utrack_project, UTRACK_GLOB),
                   "trackmate": (discover_trackmate_project, TRACKMATE_GLOB),
                   "table": (discover_table_project, TABLE_GLOB)}
        chosen = (sources or {}).get(name)
        order = ([finders[chosen]] if chosen in finders
                 else [finders["utrack"], finders["trackmate"], finders["table"]])
        for finder, pattern in order:
            try:
                movies = [str(m) for m, _ in finder(str(path), pattern=pattern)]
            except Exception:                             # noqa: BLE001 - try the next
                movies = []
            if movies:
                break
        if not movies:
            continue
        # REPLICATE DAYS AS SUB-FOLDERS (`dayfolders`): the folder is the day, and its name is the span shown
        from .dayfolders import day_map, replicate_folders

        by_folder = day_map(path, movies)
        if by_folder is not None:
            movie_to_day = by_folder
            spans = {k: f.name for k, f in enumerate(replicate_folders(path), start=1)}
        else:
            movie_to_day, spans = day_map_from_series_blocks(
                movies, series_per_day=series_per_day)
        out[name] = movie_to_day
        out.setdefault("__spans__", {})[name] = spans
    return out
