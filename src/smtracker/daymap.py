"""Replicate-day mapping, assign every movie to its imaging day, HONESTLY.

The independent unit for uncertainty is the replicate DAY. A misassigned movie
corrupts between-day variance (the exact quantity we add days to measure) and
nothing downstream can catch it. So this module refuses the two easy-but-wrong
moves: it never splits movies into equal blocks (1–10, 11–20), and it never
silently guesses when the evidence is missing.

Layouts supported, in PRECEDENCE order (explicit structure beats inference):
1. **Subfolders ``r1``..``rx``**, one folder per replicate day. If present, they
   ARE the mapping; the folder name is the day.
2. **Timestamp clustering**, read each movie's acquisition time from its
   ``movieData.mat`` and cluster on the GAPS: within-session movies are minutes
   apart, replicate days are hours/dates apart. This is the DEFAULT proposal.
3. **Manual**, an explicit series→day table (e.g. from the UI), always validated
   against the timestamps below.

Equal-block splitting is NOT a default and is not offered as one. The reference
session had unequal movie counts per condition, so equal blocks would already have been
wrong.

A replicate day is a BIOLOGICAL unit (a preculture / independent experiment), NOT
a calendar date. Two replicate days can be imaged on the SAME date (e.g. two
precultures run morning and afternoon): confirmed in the real protein D data. So
this clusters on time gaps, never on the calendar date, and two proposed days
sharing a date is legitimate: it is surfaced as a thing to confirm, never as an
error, and nothing here should ever collapse same-date sessions into one day.

Validation runs on ANY mapping (proposed or hand-entered):
- every movie assigned exactly once (unassigned / double-assigned → error);
- each day's timestamp span ≤ ``max_span_hours`` (a day mixing sessions → warn);
- no two days overlap in time (a real day split across labels → warn);
- warnings REQUIRE explicit confirmation: they are never silently accepted.

Timestamps come from ``movieData.mat``: ``acquisitionDate_`` (the true value)
when u-track recorded it, else ``createTime_`` (the import time, a PROXY, flagged
as such). Missing/unreadable → said explicitly, and the caller falls back to
manual with a banner. Nothing here reads the wall clock; only stored values.

The resolved mapping (source + the actual series→day table + per-day date/span)
is stamped into provenance so a figure records which mapping produced it.
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "DAY_MAP_FILE",
    "DAY_SOURCES",
    "day_map_path",
    "load_day_map",
    "extend_mapping",
    "save_day_map",
    "day_map_to_save",
    "DayMapError",
    "MovieStamp",
    "DayGroup",
    "DayMapping",
    "read_movie_stamps",
    "ACQUISITION_SOURCES",
    "read_acquisition_stamps",
    "find_replicate_subfolders",
    "propose_from_timestamps",
    "mapping_from_subfolders",
    "mapping_from_ranges",
    "mapping_from_assignments",
    "series_number",
    "series_of_movies",
    "SERIES_RULES",
    "mapping_from_series_rule",
    "parse_range_spec",
    "tau_warnings",
    "validate_mapping",
    "resolve_day_mapping",
    "discover_movie_paths",
    "resolve_for_condition",
]

#: The day-source enum, WHERE a mapping's day labels came from, in precedence
#: order. ``replicate_dir`` (r1..rx folders) is the PRIMARY, authoritative source
#: when present; ``movie_timestamp`` (movieData createTime_/acquisitionDate_) is the
#: fallback proposal AND, when a replicate_dir mapping exists, a CROSS-CHECK that
#: warns on disagreement rather than overriding; ``manual`` is an explicit table.
#: A FOURTH STATE RATHER THAN REUSING ONE. "remembered" is a mapping read back from
#: `day_map.json`, and it is not "manual": what was confirmed once may itself have been
#: derived, and a later reader needs to know the mapping came from disk rather than from
#: this session's data. The same argument as `AcquisitionParams`' fourth provenance
#: state, and for the same reason -- a provenance that lies is worse than a coarse one.
DAY_SOURCES = ("replicate_dir", "movie_timestamp", "manual", "remembered")

#: Default gap (hours) above which two consecutive movies are different sessions.
#: MEASURED on real data: within-session movies are ~0.2 min apart, between-session
#: gaps are ≥178 min, a ~1000x separation, so any threshold in that wide empty
#: middle works. 1 h is chosen deliberately LOW: under-splitting (merging two days)
#: corrupts between-day variance: the exact quantity days are added to measure,
#: whereas over-splitting is visible and mergeable in the confirmation table. A
#: real 3 h default once merged two Malate sessions 2.975 h apart. The proposal is
#: always shown for confirmation, so this is a starting point, not a silent choice.
DEFAULT_GAP_HOURS = 1.0
#: A single day's movies must fall within this span; more → sessions likely mixed.
DEFAULT_MAX_SPAN_HOURS = 6.0

_SERIES_RE = re.compile(r"(\d+)")
_REPLICATE_RE = re.compile(r"^r(\d+)$", re.IGNORECASE)


class DayMapError(RuntimeError):
    """Raised when a day mapping cannot be formed or is internally invalid."""


@dataclass(frozen=True)
class MovieStamp:
    """One movie's acquisition timestamp and where it came from."""

    movie: str
    timestamp: _dt.datetime | None
    source: str            # "raw_acquisition" | "acquisitionDate_" | "createTime_" | "unreadable"

    @property
    def series(self) -> int | None:
        m = _SERIES_RE.search(self.movie)
        return int(m.group(1)) if m else None


@dataclass
class DayGroup:
    """One replicate day: its movies and their time envelope."""

    day: int
    movies: list[str]
    date: str | None                       # ISO date of the first movie, if timed
    start: _dt.datetime | None
    end: _dt.datetime | None

    @property
    def span_hours(self) -> float | None:
        if self.start is None or self.end is None:
            return None
        return (self.end - self.start).total_seconds() / 3600.0


@dataclass
class DayMapping:
    """A resolved movie→day mapping, its provenance, and its verdict.

    ``requires_confirmation`` is True whenever ANY warning fired, the caller must
    surface it and get explicit sign-off, never accept silently. ``errors`` (a
    movie unassigned or assigned twice) are hard: the mapping is not usable.
    """

    movie_to_day: dict[str, int]
    source: str                            # one of DAY_SOURCES
    groups: list[DayGroup] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    timestamp_source: str | None = None    # which field, or a "missing" note
    #: THE GROUP BOARD'S HOLD-OUTS, REMEMBERED (tester D11, ruled 4 Nov 2026): the series held out ("Held out") and
    #: the series left without a group ("No group"), each still in ``movie_to_day`` with its own replicate -- the
    #: loader refuses a mapping that leaves a movie out, and re-ticking a held-out series brings its group back. Not
    #: in the fit: `included` is the mapping without them. Read from and written to ``day_map.json``
    #: (``"held_out"``, ``"no_group"``); a file without the fields holds none, as before.
    held_out: list[str] = field(default_factory=list)
    no_group: list[str] = field(default_factory=list)
    #: The source ``day_map.json`` was WRITTEN with ("manual", "synthetic", ...), kept by `load_day_map`, which sets
    #: ``source`` to "remembered": a save that changes only the hold-outs keeps it (`day_map_to_save`).
    saved_source: str | None = None
    #: REPLICATES DECLARED AS PAIRS on the group board (5 Nov 2026, asked: "optional pair ... in day picker?", "a tick
    #: box for activate pairing in day picker"): ``{replicate number in movie_to_day: tag}``, a replicate of another
    #: condition carrying the same tag being its partner (one culture split, one imaging session). Only ever set by a
    #: person, never derived from replicate numbers (Rule 0); empty unless "Pair replicates" was ticked. Read from and
    #: written to ``day_map.json`` (``"pairs"``); a file without the field pairs nothing, as before.
    pairs: dict[int, str] = field(default_factory=dict)

    @property
    def requires_confirmation(self) -> bool:
        return bool(self.warnings)

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def n_days(self) -> int:
        return len(set(self.movie_to_day.values()))

    def included(self) -> dict[str, int]:
        """``movie_to_day`` without the series held out or left without a group on the group board: what the fit
        reads, its replicates numbered 1..k again in order, as the board numbers the groups still holding a series
        (`ui.onboarding.adapt.day_maps_from`). A script that loads with the saved mapping and wants the board's
        hold-outs applies it with `project.without_held_out` (the loader itself reads every movie, so the board can
        show a held-out one again)."""
        kept, renumber = self._kept()
        return {m: renumber[d] for m, d in kept.items()}

    def _kept(self) -> tuple:
        """``(movie_to_day without the hold-outs, {saved replicate: replicate number in the fit})``."""
        out = set(self.held_out) | set(self.no_group)
        kept = {m: int(d) for m, d in self.movie_to_day.items() if m not in out}
        return kept, {old: i + 1 for i, old in enumerate(sorted(set(kept.values())))}

    def included_pairs(self) -> dict[int, str]:
        """``pairs`` keyed by the replicate numbers the fit reads (`included`'s renumbering): a replicate held out
        whole has no partner any more."""
        _kept, renumber = self._kept()
        return {renumber[int(d)]: str(tag) for d, tag in (self.pairs or {}).items() if int(d) in renumber}

    def provenance_dict(self) -> dict:
        """The record stamped into provenance (source, the full series→day table,
        and per-day date/span) so a figure records which mapping produced it."""
        return {
            "source": self.source,
            "timestamp_source": self.timestamp_source,
            "n_days": self.n_days,
            "series_to_day": {m: d for m, d in sorted(self.movie_to_day.items())},
            "per_day": [
                {"day": g.day, "date": g.date,
                 "span_hours": None if g.span_hours is None else round(g.span_hours, 3),
                 "movies": list(g.movies)}
                for g in self.groups
            ],
            "warnings": list(self.warnings),
        }


# --------------------------------------------------------------------------- #
# Timestamp reading (movieData.mat: acquisitionDate_ preferred, createTime_ proxy)
# --------------------------------------------------------------------------- #


def _datevec_to_dt(value) -> _dt.datetime | None:
    """A MATLAB datevec ``[Y, M, D, h, m, s]`` → ``datetime``, or None.

    Parses a STORED 6-element vector; it never reads the current clock.
    """
    try:
        import numpy as np

        arr = np.asarray(value, dtype=float).ravel()
    except Exception:
        return None
    if arr.size < 6 or not np.isfinite(arr[:6]).all():
        return None
    y, mo, d, h, mi, s = arr[:6]
    if y < 1970 or not (1 <= mo <= 12) or not (1 <= d <= 31):
        return None
    sec = int(s)
    micro = int(round((s - sec) * 1_000_000))
    try:
        return _dt.datetime(int(y), int(mo), int(d), int(h), int(mi), sec, micro)
    except ValueError:
        return None


def _stamp_from_movie_data(mat_path: Path) -> tuple[_dt.datetime | None, str]:
    """(timestamp, source_field) for one ``movieData.mat``.

    Prefers ``acquisitionDate_`` (the true acquisition time u-track may record),
    then ``createTime_`` (the import time: a PROXY, flagged). Returns
    ``(None, "unreadable")`` if the file cannot be decoded, never a guess.
    """
    try:
        from .utrack_meta import read_mcos_objects

        objects = read_mcos_objects(mat_path)
    except Exception:
        return None, "unreadable"
    movie_data = next((o for o in objects
                       if getattr(o, "class_name", None) == "MovieData"), None)
    if movie_data is None:
        return None, "unreadable"
    props = getattr(movie_data, "props", {})
    for field_name in ("acquisitionDate_", "createTime_"):
        ts = _datevec_to_dt(props.get(field_name))
        if ts is not None:
            return ts, field_name
    return None, "unreadable"


def read_movie_stamps(movie_paths: dict[str, Path]) -> list[MovieStamp]:
    """Read a timestamp per movie from its ``movieData.mat``.

    ``movie_paths`` maps a movie NAME to its movie folder (the one holding
    ``movieData.mat`` and ``TrackingPackage/``). Unreadable movies come back with
    ``timestamp=None`` and ``source="unreadable"`` so the caller can say so and
    fall back to manual, never silently drop or guess them.
    """
    stamps = []
    for movie, folder in movie_paths.items():
        mat = Path(folder) / "movieData.mat"
        if mat.exists():
            ts, source = _stamp_from_movie_data(mat)
        else:
            ts, source = None, "unreadable"
        stamps.append(MovieStamp(movie=movie, timestamp=ts, source=source))
    return stamps


#: The stamp sources that record when a movie was TAKEN. ``createTime_`` is not one:
#: it is when u-track imported the movie, and an import session can hold movies
#: imaged months apart (see `read_acquisition_stamps`).
ACQUISITION_SOURCES = frozenset({"raw_acquisition", "acquisitionDate_"})


def read_acquisition_stamps(condition_root, movie_paths: dict[str, Path]) -> list[MovieStamp]:
    """One stamp per movie, the instrument's own acquisition time wherever it can be read.

    The raw acquisition beside the condition (``raw/<movie>.nd2`` or ``.stk``) is read
    first, as ``"raw_acquisition"``; a movie without one, or whose file records no date,
    keeps its ``movieData.mat`` stamp (`read_movie_stamps`).

    WHY THE RAW FILE OUTRANKS movieData.mat. Where u-track was given no acquisition date,
    `movieData.mat` holds ``createTime_``, the IMPORT time, and an import session is not
    an imaging session. Measured on protein A/Malate: all 29 movies were imported on 9 March
    2026, series041-050 at 07:36 and series001-009 at 08:18, so the one-hour gap rule
    joined them into one day -- while the `.nd2` files date them 25.02.2026 and
    25.06.2025, eight months apart. The raw read costs ~15 ms per movie.

    A raw file that cannot be read is not an error here: the movie keeps its
    movieData.mat stamp, whose source still says what it is.
    """
    from .rawmovie import ACQUISITION_SUFFIXES, RawMovieError, acquisition_time

    from .dayfolders import on_disk

    stamps = {s.movie: s for s in read_movie_stamps(movie_paths)}
    listed: dict = {}

    def raw_file(movie):
        """The movie's raw acquisition: ``raw/<movie>`` in its own folder (a day folder's, for ``series001 (rep1)``)."""
        folder, name = on_disk(condition_root, movie)
        raw = Path(folder) / "raw"
        if raw not in listed:
            listed[raw] = ({p.stem: p for p in sorted(raw.iterdir())
                            if p.is_file() and p.suffix.lower() in ACQUISITION_SUFFIXES}
                           if raw.is_dir() else {})
        return listed[raw].get(name)

    if condition_root:
        for movie in stamps:
            path = raw_file(movie)
            if path is None:
                continue
            try:
                when = acquisition_time(path)
            except RawMovieError:
                continue
            if when is not None:
                stamps[movie] = MovieStamp(movie=movie, timestamp=when,
                                           source="raw_acquisition")
    return list(stamps.values())


# --------------------------------------------------------------------------- #
# Layout 1, replicate subfolders r1..rx
# --------------------------------------------------------------------------- #


def find_replicate_subfolders(root: Path) -> dict[int, Path] | None:
    """``{day: folder}`` for ``r1``..``rx`` subfolders of ``root``, or None.

    Case-insensitive ``r<number>``. Returns None (not an error) when the layout is
    absent, so the caller falls through to timestamp clustering.
    """
    root = Path(root)
    if not root.is_dir():
        return None
    found = {}
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        m = _REPLICATE_RE.match(child.name)
        if m:
            found[int(m.group(1))] = child
    return found or None


def mapping_from_subfolders(replicate_dirs: dict[int, Path],
                            movies_in: "callable") -> DayMapping:
    """Build the mapping from ``r1``..``rx`` folders, the folder name IS the day.

    ``movies_in(folder)`` returns the movie names discovered under a replicate
    folder (injected so this stays pure and testable without a real tree).
    """
    movie_to_day: dict[str, int] = {}
    groups = []
    for day in sorted(replicate_dirs):
        movies = sorted(movies_in(replicate_dirs[day]))
        for movie in movies:
            if movie in movie_to_day:
                raise DayMapError(
                    f"movie {movie!r} appears under two replicate folders")
            movie_to_day[movie] = day
        groups.append(DayGroup(day=day, movies=movies, date=None, start=None, end=None))
    if not movie_to_day:
        raise DayMapError("replicate folders r1..rx contain no movies")
    return DayMapping(movie_to_day=movie_to_day, source="replicate_dir", groups=groups)


# --------------------------------------------------------------------------- #
# Layout 2, timestamp clustering (the DEFAULT proposal)
# --------------------------------------------------------------------------- #


def propose_from_timestamps(stamps: list[MovieStamp], *,
                            gap_hours: float = DEFAULT_GAP_HOURS) -> DayMapping:
    """Cluster movies into days by time gaps: the default, never equal blocks.

    Movies are sorted by timestamp; a gap larger than ``gap_hours`` to the previous
    movie starts a new day. Days are numbered chronologically. Movies with no
    readable timestamp are reported (error) rather than placed by guesswork.
    """
    timed = [s for s in stamps if s.timestamp is not None]
    untimed = [s for s in stamps if s.timestamp is None]

    mapping = DayMapping(movie_to_day={}, source="movie_timestamp")
    sources = {s.source for s in timed}
    mapping.timestamp_source = (
        "+".join(sorted(sources)) if sources else "none")
    if any(s == "createTime_" for s in sources):
        mapping.warnings.append(
            "timestamps use createTime_ (u-track import time), which only approximates "
            "the acquisition time: check the replicate grouping against your records")

    if not timed:
        mapping.errors.append(
            "no readable acquisition timestamps: cannot propose replicates; enter the "
            "mapping by hand (see the banner)")
        return mapping

    timed.sort(key=lambda s: s.timestamp)
    clusters: list[list[MovieStamp]] = [[timed[0]]]
    for prev, cur in zip(timed, timed[1:]):
        gap_h = (cur.timestamp - prev.timestamp).total_seconds() / 3600.0
        (clusters.append([cur]) if gap_h > gap_hours else clusters[-1].append(cur))

    for day, cluster in enumerate(clusters, start=1):
        movies = [s.movie for s in cluster]
        start, end = cluster[0].timestamp, cluster[-1].timestamp
        for movie in movies:
            mapping.movie_to_day[movie] = day
        mapping.groups.append(DayGroup(
            day=day, movies=sorted(movies), date=start.date().isoformat(),
            start=start, end=end))

    # Two proposed days on the SAME calendar date are legitimate, separate
    # replicates (different precultures) can be imaged the same day. That is NOT
    # an error, but it IS the case to confirm deliberately rather than assume, so
    # name it. A replicate day is a biological unit, never a calendar date.
    from collections import Counter

    date_counts = Counter(g.date for g in mapping.groups)
    for date, count in date_counts.items():
        if count > 1:
            shared = [g.day for g in mapping.groups if g.date == date]
            mapping.warnings.append(
                f"replicates {shared} share the date {date}. Check that they are separate "
                "replicates (for example different precultures imaged on the same date) "
                "and not one session split in two.")

    if untimed:
        mapping.errors.append(
            "movies without a readable timestamp (assign them manually): "
            + ", ".join(sorted(s.movie for s in untimed)))
    return mapping


# --------------------------------------------------------------------------- #
# Manual, parse an editable range/day table, then validate
# --------------------------------------------------------------------------- #


def parse_range_spec(spec: str) -> set[int]:
    """Parse ``"1-10, 12, 15-17"`` → the set of series numbers. Fails loud on junk."""
    out: set[int] = set()
    for chunk in str(spec).replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, _, b = chunk.partition("-")
            try:
                lo, hi = int(a), int(b)
            except ValueError:
                raise DayMapError(f"bad range {chunk!r} in {spec!r}")
            if lo > hi:
                raise DayMapError(f"range {chunk!r} is descending")
            out.update(range(lo, hi + 1))
        else:
            try:
                out.add(int(chunk))
            except ValueError:
                raise DayMapError(f"bad series number {chunk!r} in {spec!r}")
    return out


def mapping_from_ranges(day_to_spec: dict[int, str], movies: list[str]) -> DayMapping:
    """Build a manual mapping from a ``{day: "1-10"}`` table over the real movies.

    Catches double-assignment (a series in two days) as an error; movies whose
    series falls in no range are reported as unassigned. Series numbers come from
    the movie names, so this ties the human table back to the actual movies.
    """
    series_of = {}
    for movie in movies:
        m = _SERIES_RE.search(movie)
        if m is None:
            raise DayMapError(f"cannot read a series number from movie {movie!r}")
        series_of[movie] = int(m.group(1))

    series_to_day: dict[int, int] = {}
    errors: list[str] = []
    for day, spec in day_to_spec.items():
        for series in parse_range_spec(spec):
            if series in series_to_day and series_to_day[series] != day:
                errors.append(
                    f"series {series} assigned to both replicate {series_to_day[series]} "
                    f"and replicate {day}")
            series_to_day[series] = day

    movie_to_day = {}
    unassigned = []
    for movie, series in series_of.items():
        if series in series_to_day:
            movie_to_day[movie] = series_to_day[series]
        else:
            unassigned.append(movie)
    if unassigned:
        errors.append("unassigned movies: " + ", ".join(sorted(unassigned)))

    groups = []
    for day in sorted(set(movie_to_day.values())):
        groups.append(DayGroup(
            day=day, movies=sorted(m for m, d in movie_to_day.items() if d == day),
            date=None, start=None, end=None))
    return DayMapping(movie_to_day=movie_to_day, source="manual", groups=groups,
                      errors=errors)


# --------------------------------------------------------------------------- #
# Validation, cross-check ANY mapping against the timestamps
# --------------------------------------------------------------------------- #


def validate_mapping(movie_to_day: dict[str, int], stamps: list[MovieStamp], *,
                     max_span_hours: float = DEFAULT_MAX_SPAN_HOURS
                     ) -> tuple[list[str], list[str]]:
    """(warnings, errors) for a mapping cross-checked against the timestamps.

    ERRORS (hard): a movie in the project left unassigned, or a movie assigned
    that is not in the project. WARNINGS (need confirmation): a day whose movies
    span more than ``max_span_hours`` (sessions likely mixed), or two days that
    overlap in time (a real day split across labels). A mapping with no timestamps
    to check against gets a warning that it could not be verified.
    """
    warnings: list[str] = []
    errors: list[str] = []

    project_movies = {s.movie for s in stamps}
    assigned = set(movie_to_day)
    missing = project_movies - assigned
    extra = assigned - project_movies
    if missing:
        errors.append("movies not assigned to any replicate: "
                      + ", ".join(sorted(missing)))
    if extra:
        errors.append("mapping names movies not in the project: "
                      + ", ".join(sorted(extra)))

    ts_of = {s.movie: s.timestamp for s in stamps if s.timestamp is not None}
    if not ts_of:
        warnings.append(
            "no timestamps available to cross-check this mapping: it is accepted "
            "unverified; check it against your records")
        return warnings, errors

    day_times: dict[int, list[_dt.datetime]] = {}
    for movie, day in movie_to_day.items():
        if movie in ts_of:
            day_times.setdefault(day, []).append(ts_of[movie])

    envelopes = {}
    for day, times in day_times.items():
        lo, hi = min(times), max(times)
        envelopes[day] = (lo, hi)
        span_h = (hi - lo).total_seconds() / 3600.0
        if span_h > max_span_hours:
            warnings.append(
                f"replicate {day} spans {span_h:.1f} h (> {max_span_hours:g} h): it may "
                "mix movies from different sessions")

    ordered = sorted(envelopes)
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            a, b = envelopes[ordered[i]], envelopes[ordered[j]]
            if a[0] <= b[1] and b[0] <= a[1]:
                warnings.append(
                    f"replicates {ordered[i]} and {ordered[j]} overlap in time: a single "
                    "replicate may be split across two labels")
    return warnings, errors


# --------------------------------------------------------------------------- #
# Resolve, precedence: subfolder > timestamp > manual override
# --------------------------------------------------------------------------- #


def _induced_partition(movie_to_day: dict[str, int], movies) -> frozenset:
    """The unordered partition a mapping induces over ``movies``, day LABELS and
    ordering discarded, so two mappings that group the same movies together match
    even if they number the days differently."""
    groups: dict[int, set] = {}
    for m in movies:
        if m in movie_to_day:
            groups.setdefault(movie_to_day[m], set()).add(m)
    return frozenset(frozenset(g) for g in groups.values())


def _timestamp_cross_check(mapping: DayMapping, stamps: list[MovieStamp], *,
                           gap_hours: float) -> None:
    """Cross-check a replicate_dir mapping against the timestamp clustering and WARN
    (never override) on disagreement: the folders are the source of truth; the
    timestamps only flag that they might be mislabelled."""
    timed = [s for s in stamps if s.timestamp is not None]
    if not timed:
        mapping.warnings.append(
            "no timestamps to cross-check the r1..rx folder mapping against: "
            "accepted on the folders alone; check it against your records")
        return
    proposal = propose_from_timestamps(timed, gap_hours=gap_hours)
    movies = set(mapping.movie_to_day)
    if _induced_partition(mapping.movie_to_day, movies) != \
            _induced_partition(proposal.movie_to_day, movies):
        mapping.warnings.append(
            "the r1..rx folder grouping disagrees with the grouping by movie timestamps. "
            "The folders are used. Check that no movie sits in the wrong replicate folder.")


def resolve_day_mapping(*, movie_paths: dict[str, Path],
                        replicate_dirs: dict[int, Path] | None = None,
                        movies_in=None,
                        override: dict[int, str] | None = None,
                        gap_hours: float = DEFAULT_GAP_HOURS,
                        max_span_hours: float = DEFAULT_MAX_SPAN_HOURS) -> DayMapping:
    """Resolve the day mapping by PRECEDENCE and validate it against timestamps.

    - ``override`` (a ``{day: "1-10"}`` table from the UI) wins if given (the
      human said so) but is still validated and can raise warnings.
    - else ``replicate_dirs`` (``r1``..``rx``) if present, explicit structure.
    - else the timestamp-clustered proposal (the default).

    Every path returns a :class:`DayMapping` carrying its warnings/errors and
    ``requires_confirmation``; the caller decides whether to proceed.
    """
    stamps = read_movie_stamps(movie_paths)

    if override is not None:
        mapping = mapping_from_ranges(override, list(movie_paths))
    elif replicate_dirs:
        if movies_in is None:
            raise DayMapError("subfolder layout needs a movies_in(folder) callable")
        mapping = mapping_from_subfolders(replicate_dirs, movies_in)
        # r1..rx are AUTHORITATIVE; the timestamps only cross-check and warn.
        _timestamp_cross_check(mapping, stamps, gap_hours=gap_hours)
    else:
        return propose_from_timestamps(stamps, gap_hours=gap_hours)

    warnings, errors = validate_mapping(
        mapping.movie_to_day, stamps, max_span_hours=max_span_hours)
    mapping.warnings.extend(warnings)
    mapping.errors.extend(errors)
    if any(s.source == "createTime_" for s in stamps if s.timestamp is not None):
        mapping.timestamp_source = "createTime_"
    elif any(s.timestamp is not None for s in stamps):
        mapping.timestamp_source = "acquisitionDate_"
    else:
        mapping.timestamp_source = "none"
    return mapping


# --------------------------------------------------------------------------- #
# Project discovery, connect the pure logic above to a real u-track tree
# --------------------------------------------------------------------------- #


def discover_movie_paths(root: Path, *, pattern: str | None = None) -> dict[str, Path]:
    """``{movie_name: movie_folder}`` for every u-track movie under ``root``.

    The movie folder is the one holding ``movieData.mat`` and ``TrackingPackage/``
    two levels above the tracks ``.mat`` (``tracks/`` -> ``TrackingPackage/`` ->
    movie). Uses the same discovery globs as grouping so the movie names match.

    EITHER TRACKER. This read u-track only, and `discover_utrack_project` RAISES where
    there is none -- so on a TrackMate project the whole day picker was unreachable: the
    caller caught the error, skipped the condition, and the series rule was applied in
    silence. Reported as "it just opened without asking me first", which is the same
    fault as the day-map derivation had one level up, found separately because each
    entry point does its own discovery.

    A TrackMate movie has no movie FOLDER -- its tracks are `trackmate/<movie>.xml` --
    so the condition root stands in. Nothing here reads that path except the timestamp
    probe, which returns "no stamp" for a folder with no `movieData.mat`, and a missing
    timestamp is already an ordinary state.
    """
    from .grouping import (TRACKMATE_GLOB, UTRACK_GLOB, discover_trackmate_project,
                           discover_utrack_project)

    out: dict[str, Path] = {}
    try:
        for movie, tracks_path in discover_utrack_project(
                root, pattern=pattern or UTRACK_GLOB):
            out[str(movie)] = Path(tracks_path).parents[2]
    except Exception:                                     # noqa: BLE001 - try the next
        out = {}
    if out:
        return out
    try:
        for movie, _xml in discover_trackmate_project(
                root, pattern=pattern or TRACKMATE_GLOB):
            out[str(movie)] = Path(root)
    except Exception:                                     # noqa: BLE001 - try the next
        out = {}
    if out:
        return out
    # TRACK TABLES (`tracktable`), the third tracker: no movie folder either.
    from .grouping import discover_table_project
    from .tracktable import TABLE_GLOB

    try:
        for movie, _table in discover_table_project(root, pattern=pattern or TABLE_GLOB):
            out[str(movie)] = Path(root)
    except Exception:                                     # noqa: BLE001 - none either
        return {}
    return out


def resolve_for_condition(condition_root: Path, *,
                          override: dict[int, str] | None = None,
                          gap_hours: float = DEFAULT_GAP_HOURS,
                          max_span_hours: float = DEFAULT_MAX_SPAN_HOURS) -> DayMapping:
    """Resolve one condition folder's day mapping from a real u-track tree.

    Discovers the movies + their ``movieData.mat`` timestamps and the optional
    ``r1``..``rx`` subfolders, then applies the precedence in
    :func:`resolve_day_mapping`. This is what the UI calls to pre-fill its table.
    """
    from .dayfolders import day_folders, movie_name, replicate_folders

    condition_root = Path(condition_root)
    movie_paths = discover_movie_paths(condition_root)
    # r1..rx, else day folders of either `dayfolders` layout (`rep1`.., `LB-1`..): the folder is the day, as the
    # derived mapping has it (`grouping.day_maps_from_series_naming`)
    replicate_dirs = find_replicate_subfolders(condition_root) or {
        k: folder for k, folder in enumerate(replicate_folders(condition_root) or [], start=1)} or None
    # r1..rx that START THEIR MOVIE NAMES AGAIN (`dayfolders`): their movies by their day-folder names, as loaded
    named_by_folder = day_folders(condition_root) is not None

    def movies_in(folder):
        found = list(discover_movie_paths(folder))
        return [movie_name(m, Path(folder).name) for m in found] if named_by_folder else found

    return resolve_day_mapping(
        movie_paths=movie_paths, replicate_dirs=replicate_dirs,
        movies_in=movies_in, override=override, gap_hours=gap_hours,
        max_span_hours=max_span_hours)


# --------------------------------------------------------------------------- #
# Series-number rules and per-movie assignment.
#
# Two things the range table cannot express, both of which the experimenter needs:
#   * a ONE-CLICK rule for the lab's own convention, so the ranges do not have to
#     be retyped for every condition (and mistyped for one of them);
#   * a PER-MOVIE assignment, which is the only form that handles a scattered or
#     one-off day: a range spec cannot say "series 7 belongs to day 2" without
#     the surrounding block agreeing.
#
# Both live here, not in the dialog: the Qt shell computes nothing.
# --------------------------------------------------------------------------- #

#: Where a confirmed mapping is remembered, beside the condition it describes rather
#: than in one project-wide file. A condition is the unit that moves: this project's own
#: layout rule is that the folders ABOVE a movie carry no meaning, so a mapping filed by
#: the parent's name would be lost the first time a folder was renamed or moved.
DAY_MAP_FILE = "day_map.json"


def day_map_path(condition_root) -> Path:
    """Where `condition_root`'s remembered mapping lives."""
    return Path(condition_root) / DAY_MAP_FILE


def save_day_map(condition_root, mapping: "DayMapping", *,
                 series_per_day: int | None = None) -> Path:
    """Write a confirmed mapping so the next session does not ask again.

    ASKED FOR: "the day picker should save the day naming convention somewhere ... so i
    dont need to setup the naming every single time."

    THE MAPPING, NOT THE RULE. What is recorded is which movie belongs to which day,
    because that is what the analysis consumes and what the user actually confirmed. A
    rule ("ten series to a day") would have to be re-applied to whatever movies are
    present later, and a movie added afterwards would silently change days already
    reported -- the mapping is the thing that must not drift.

    A mapping carrying ERRORS is refused: a movie unassigned or assigned twice is not a
    mapping, and remembering one would hand the same fault back every session.
    """
    import json

    mapping_errors = list(getattr(mapping, "errors", ()) or ())
    if mapping_errors:
        raise DayMapError(
            "refusing to remember a mapping with errors: " + "; ".join(mapping_errors))
    if not mapping.movie_to_day:
        raise DayMapError("refusing to remember an empty mapping")
    path = day_map_path(condition_root)
    payload = {
        "movie_to_day": {str(k): int(v) for k, v in mapping.movie_to_day.items()},
        "source": mapping.source,
        "series_per_day": series_per_day,
        # KEPT so a later reader can say what was accepted and what was merely not an
        # error. A warning that vanished on the round trip would read as a clean map.
        "warnings": list(mapping.warnings or ()),
        "written_by": "smtracker.daymap.save_day_map",
    }
    # THE GROUP BOARD'S HOLD-OUTS (tester D11, ruled 4 Nov 2026), written only when there are any, so a file with none
    # has the shape it always had; each names a movie of the mapping
    for key in ("held_out", "no_group"):
        names = sorted({str(m) for m in (getattr(mapping, key, None) or ())})
        strays = [m for m in names if m not in payload["movie_to_day"]]
        if strays:
            raise DayMapError(f"refusing to remember {key.replace('_', ' ')} series the mapping does not list: "
                              + ", ".join(strays))
        if names:
            payload[key] = names
    # THE DECLARED PAIRS (5 Nov 2026), written only when there are any; each names a replicate of the mapping
    pairs = {int(d): str(tag) for d, tag in (getattr(mapping, "pairs", None) or {}).items()}
    if pairs:
        problem = pair_problem(pairs, set(payload["movie_to_day"].values()))
        if problem:
            raise DayMapError(f"refusing to remember the pairs: {problem}")
        payload["pairs"] = {str(d): pairs[d] for d in sorted(pairs)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


#: A pair's tag: one to three letters or digits, as the board offers them (A, B, ...).
PAIR_TAG_MAX = 3


def pair_problem(pairs: dict, replicates) -> str:
    """Why ``pairs`` (``{replicate: tag}``) cannot be remembered for a condition holding ``replicates``, or "": each
    key a replicate of the condition, each tag a short name, and no tag on two replicates of one condition (a
    replicate has one partner per other condition)."""
    replicates = {int(r) for r in replicates}
    strays = sorted(int(d) for d in pairs if int(d) not in replicates)
    if strays:
        return "no replicate " + ", ".join(str(d) for d in strays)
    bad = [t for t in pairs.values() if not (isinstance(t, str) and 0 < len(t.strip()) <= PAIR_TAG_MAX
                                             and t.strip().isalnum())]
    if bad:
        return "a tag is one to three letters or digits; these are not: " + ", ".join(repr(t) for t in bad)
    seen: dict = {}
    for d, tag in pairs.items():
        seen.setdefault(tag.strip(), []).append(int(d))
    twice = {tag: ds for tag, ds in seen.items() if len(ds) > 1}
    if twice:
        return "; ".join(f"tag {tag} on replicates {', '.join(str(d) for d in sorted(ds))} of one condition"
                         for tag, ds in twice.items())
    return ""


def day_map_to_save(confirmed: dict, loaded: dict, on_disk: dict | None, *,
                    derived_source: str | None = None, left_out: dict | None = None,
                    on_disk_left_out: dict | None = None, on_disk_source: str | None = None,
                    pairs: dict | None = None, on_disk_pairs: dict | None = None) -> str | None:
    """What the analysis door writes for a condition when Open workspace confirms its grouping: the SOURCE to save
    the mapping with, or None when nothing should be written.

    ``confirmed`` is the board's complete mapping, ``loaded`` the one the project was opened with, ``on_disk`` the
    ``movie_to_day`` already in `day_map.json` (None when there is none, or it cannot be read). Found 3 Nov 2026
    (tester B10): every open rewrote every condition's file as "manual", so a synthetic set's provenance
    ("synthetic", written by its builder) was lost and the door wrote into the data folders with nothing changed.
    ``left_out`` and ``on_disk_left_out`` are the board's and the file's hold-outs, ``{"held_out": [...],
    "no_group": [...]}`` (tester D11, 4 Nov 2026); ``on_disk_source`` the source the file was written with
    (`DayMapping.saved_source`). ``pairs`` and ``on_disk_pairs`` the board's and the file's declared pairs
    (``{replicate: tag}``, 5 Nov 2026), compared like the hold-outs.

    * the file already holds this mapping, these hold-outs and these pairs: None, the file and its provenance stay as
      written;
    * the board changed the mapping: "manual", a person's arrangement;
    * the board changed only the hold-outs or the pairs: the file's own source, since the replicates are still its
      own (a synthetic set's stays "synthetic"), "manual" when unknown;
    * no file yet, nothing changed: the derived proposal confirmed, saved with the source it was derived from
      (``derived_source``, one of `DAY_SOURCES` other than "remembered") or "manual" when that is unknown."""
    def norm(mapping):
        return {str(k): int(v) for k, v in (mapping or {}).items()}

    def held(value):
        return {key: sorted({str(m) for m in (value or {}).get(key, ())}) for key in ("held_out", "no_group")}

    def paired(value):
        return {int(d): str(t) for d, t in (value or {}).items()}

    same_days = on_disk is not None and norm(on_disk) == norm(confirmed)
    if same_days and held(on_disk_left_out) == held(left_out) and paired(on_disk_pairs) == paired(pairs):
        return None
    if norm(confirmed) != norm(loaded):
        return "manual"
    if same_days:
        return on_disk_source or "manual"
    if derived_source in DAY_SOURCES and derived_source != "remembered":
        return derived_source
    return "manual"


def load_day_map(condition_root):
    """The remembered mapping for a condition, or ``None`` if there is none.

    Returns ``None`` rather than raising for an absent file -- not having one is the
    ordinary state -- but a file that EXISTS and cannot be read raises, because silently
    re-deriving over a mapping somebody confirmed is how two day numberings come to
    describe one project, which is the defect the day guard exists for.
    """
    import json

    path = day_map_path(condition_root)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise DayMapError(f"{path}: not readable JSON ({exc})") from exc
    raw = payload.get("movie_to_day")
    if not isinstance(raw, dict) or not raw:
        raise DayMapError(f"{path}: holds no movie_to_day mapping")
    try:
        movie_to_day = {str(k): int(v) for k, v in raw.items()}
    except (TypeError, ValueError) as exc:
        raise DayMapError(f"{path}: a replicate is not a whole number ({exc})") from exc
    mapping = mapping_from_assignments(movie_to_day)
    # THE SOURCE SAYS IT WAS REMEMBERED, not re-derived. A run that reports "series
    # naming" for a mapping a person typed would misdescribe its own provenance.
    mapping.source = "remembered"
    written = payload.get("source")
    mapping.saved_source = str(written) if isinstance(written, str) and written else None
    saved_warnings = payload.get("warnings")
    if isinstance(saved_warnings, list):
        for text in saved_warnings:
            if text not in mapping.warnings:
                mapping.warnings.append(str(text))
    # THE GROUP BOARD'S HOLD-OUTS (D11): absent in a file written before 4 Nov 2026, which then holds none
    for key in ("held_out", "no_group"):
        names = payload.get(key, [])
        if not isinstance(names, list) or not all(isinstance(m, str) for m in names):
            raise DayMapError(f"{path}: {key!r} is not a list of series names")
        strays = [m for m in names if m not in movie_to_day]
        if strays:
            raise DayMapError(f"{path}: {key!r} names series the mapping does not list: {', '.join(strays)}")
        setattr(mapping, key, sorted(set(names)))
    # THE DECLARED PAIRS (5 Nov 2026): absent in a file written before, which then pairs nothing
    raw_pairs = payload.get("pairs", {})
    if not isinstance(raw_pairs, dict):
        raise DayMapError(f"{path}: 'pairs' is not a mapping of replicate to tag")
    try:
        pairs = {int(d): str(t).strip() for d, t in raw_pairs.items()}
    except (TypeError, ValueError):
        raise DayMapError(f"{path}: 'pairs' names something that is not a replicate number") from None
    problem = pair_problem(pairs, set(movie_to_day.values())) if pairs else ""
    if problem:
        raise DayMapError(f"{path}: 'pairs': {problem}")
    # the file's replicate numbers, renumbered 1..k as the mapping itself was (`mapping_from_assignments`)
    renumber = {old: i + 1 for i, old in enumerate(sorted(set(movie_to_day.values())))}
    mapping.pairs = {renumber[d]: t for d, t in pairs.items()}
    return mapping


def extend_mapping(remembered: "DayMapping", derived: dict) -> tuple:
    """``(mapping, note)``: a REMEMBERED mapping given a day for every movie it does not
    list, or ``(remembered, "")`` when it lists them all.

    Asked: "what happens if i cached a folder, then i add a new replicate day or a new
    condition?" The cache is keyed on the files and re-reads, but the remembered
    ``day_map.json`` named only the movies present when it was saved, and the loader
    rightly refuses a mapping that leaves a movie out -- so adding a day made the folder
    unopenable until the file was deleted. Now each new movie is placed by the naming
    authority (``derived``, `grouping.day_maps_from_series_naming`): one that falls in a
    series block whose remembered movies all share a day JOINS that day (a movie added to
    an existing session), and a block with no remembered movie becomes a NEW day after the
    last (a new replicate day). The note says which, and is attached as a WARNING, so the
    mapping asks for confirmation and is not written back until it is confirmed.
    """
    known = dict(remembered.movie_to_day)
    derived = {str(m): int(d) for m, d in dict(getattr(derived, "movie_to_day",
                                                        derived)).items()}
    new = sorted((m for m in derived if m not in known), key=_series_or_name)
    if not new:
        return remembered, ""
    old_days_by_block = {}
    for movie, day in known.items():
        if movie in derived:
            old_days_by_block.setdefault(derived[movie], set()).add(day)
    assigned = dict(known)
    next_day = max(known.values(), default=0)
    fresh = {}
    joined, opened = [], []
    for movie in new:
        block = derived[movie]
        days = old_days_by_block.get(block)
        if days and len(days) == 1:
            assigned[movie] = next(iter(days))
            joined.append(movie)
            continue
        if block not in fresh:
            next_day += 1
            fresh[block] = next_day
        assigned[movie] = fresh[block]
        opened.append(movie)
    mapping = mapping_from_assignments(assigned)
    mapping.source = "remembered"
    mapping.warnings = list(remembered.warnings)
    # the hold-outs travel with the mapping; a movie added since was never held out
    mapping.held_out, mapping.no_group = list(remembered.held_out), list(remembered.no_group)
    mapping.saved_source = remembered.saved_source
    mapping.pairs = dict(remembered.pairs)
    parts = []
    if opened:
        days = sorted({mapping.movie_to_day[m] for m in opened})
        parts.append(f"{_span_text(opened)} {'are' if len(opened) > 1 else 'is'} new and "
                     f"became replicate{'s' if len(days) > 1 else ''} "
                     f"{', '.join(str(d) for d in days)} (their own series block)")
    if joined:
        parts.append(f"{_span_text(joined)} joined the replicate of {'their' if len(joined) > 1 else 'its'} "
                     "series block")
    note = ("the saved replicate mapping did not list every movie: " + "; ".join(parts)
            + ". Check the replicates before relying on them; confirming saves them.")
    mapping.warnings.append(note)
    return mapping, note


def _series_or_name(movie: str):
    try:
        return (0, series_number(movie), str(movie))
    except DayMapError:
        return (1, 0, str(movie))


def _span_text(movies) -> str:
    movies = list(movies)
    return movies[0] if len(movies) == 1 else f"{movies[0]}-{movies[-1]} ({len(movies)} movies)"


def series_number(movie: str) -> int:
    """The series number in a movie name. Raises rather than guessing."""
    m = _SERIES_RE.search(str(movie))
    if m is None:
        raise DayMapError(
            f"cannot read a series number from movie {movie!r}: refusing to guess "
            "its replicate")
    return int(m.group(1))


def series_of_movies(movies) -> dict[str, int]:
    """``{movie: series_number}`` for a list of movies."""
    return {m: series_number(m) for m in movies}


def _rule_decade(series: list[int]) -> dict[int, int]:
    """One DECADE of series numbers = one replicate day, decades numbered in order.

    The lab convention: 001-010 is day 1, 011-020 day 2, and the third block is
    whatever decade comes next: 021-030 or 031-040. Numbering the decades that are
    PRESENT (rather than dividing by ten) is what makes the 031-040 case need no
    special handling, and it never invents an empty day.
    """
    decades = sorted({(s - 1) // 10 for s in series})
    order = {d: i + 1 for i, d in enumerate(decades)}
    return {s: order[(s - 1) // 10] for s in series}


def _rule_fixed_block(series: list[int], n: int) -> dict[int, int]:
    """``n`` CONSECUTIVE series per day, by position in the sorted list.

    Differs from the decade rule when the numbering has gaps: this one counts
    movies, the decade rule reads the number. Offered because which is right is a
    fact about how the experiment was recorded, not something to infer.
    """
    if n < 1:
        raise DayMapError(f"series per replicate must be >= 1, got {n}")
    return {s: i // n + 1 for i, s in enumerate(sorted(series))}


#: The one-click rules. Name → (label, function of the series numbers).
SERIES_RULES = {
    "decade": ("One decade of series numbers = one replicate (001-010, 011-020, 031-040…)",
               _rule_decade),
    "block10": ("Every 10 movies in series order = one replicate", 
                lambda s: _rule_fixed_block(s, 10)),
    "block5": ("Every 5 movies in series order = one replicate",
               lambda s: _rule_fixed_block(s, 5)),
}


def mapping_from_series_rule(movies, rule: str = "decade", *,
                             n_per_day: int | None = None,
                             stamps=None) -> "DayMapping":
    """Apply a series-number rule to real movies → a manual ``DayMapping``.

    ``rule`` is a key of :data:`SERIES_RULES`, or ``"block"`` with ``n_per_day``
    for an arbitrary block size. The result is ``source="manual"``: a rule is the
    experimenter asserting the structure, not the file telling us, so it is
    labelled as their assertion and still goes through the same validation.
    """
    movies = list(movies)
    series_of = series_of_movies(movies)
    if rule == "block":
        if n_per_day is None:
            raise DayMapError("rule 'block' needs n_per_day")
        series_to_day = _rule_fixed_block(list(series_of.values()), int(n_per_day))
    else:
        if rule not in SERIES_RULES:
            raise DayMapError(
                f"unknown rule {rule!r}; known: {sorted(SERIES_RULES)} (or 'block' "
                "with n_per_day)")
        series_to_day = SERIES_RULES[rule][1](list(series_of.values()))
    return mapping_from_assignments({m: series_to_day[s]
                                     for m, s in series_of.items()}, stamps=stamps)


def mapping_from_assignments(movie_to_day: dict, *, stamps=None) -> "DayMapping":
    """A ``DayMapping`` from an explicit ``{movie: day}``, the most general form.

    Every other constructor is a shortcut to this one. It is what makes a scattered
    or one-off assignment possible: a range spec cannot say "series 7 is day 2"
    unless its neighbours agree, and sometimes they do not.

    Days are RENUMBERED to 1..k in ascending order of the labels given, so deleting
    a day never leaves a hole and the numbering always means "first, second, third".
    Empty days cannot exist by construction.
    """
    if not movie_to_day:
        raise DayMapError("no movies assigned: refusing to build an empty mapping")
    labels = sorted({int(d) for d in movie_to_day.values()})
    renumber = {old: i + 1 for i, old in enumerate(labels)}
    assigned = {str(m): renumber[int(d)] for m, d in movie_to_day.items()}

    times = {s.movie: s.timestamp for s in (stamps or []) if s.timestamp is not None}
    groups = []
    for day in sorted(set(assigned.values())):
        members = sorted((m for m, d in assigned.items() if d == day),
                         key=series_number)
        ts = sorted(t for m, t in times.items() if m in set(members))
        groups.append(DayGroup(
            day=day, movies=members,
            date=ts[0].date().isoformat() if ts else None,
            start=ts[0] if ts else None, end=ts[-1] if ts else None))
    return DayMapping(movie_to_day=assigned, source="manual", groups=groups)


def tau_warnings(movie_to_day: dict, tau_by_movie: dict) -> list[str]:
    """Warn when a replicate DAY mixes acquisition rates. Pure.

    Mixing frame intervals ACROSS days is a stratum to report; mixing them WITHIN a
    day is fatal, because the day is the analysis unit and its movies are pooled
    into one number. Every D scales with 1/tau, so a day holding both 20.3 ms and
    30 ms movies produces a mean that is a blend of two different measurements, and
    nothing downstream can detect it.

    Returns one warning per offending day, naming the rates and the movie counts, so
    the fix (split the day at the rate change, or drop the odd movies) is obvious
    from the message.
    """
    if not tau_by_movie:
        return []
    by_day: dict[int, dict[float, int]] = {}
    for movie, day in movie_to_day.items():
        ms = tau_by_movie.get(movie)
        if ms is None:
            continue
        by_day.setdefault(int(day), {}).setdefault(round(float(ms), 6), 0)
        by_day[int(day)][round(float(ms), 6)] += 1
    out = []
    for day in sorted(by_day):
        rates = by_day[day]
        if len(rates) > 1:
            detail = ", ".join(f"{ms:g} ms x{n}" for ms, n in sorted(rates.items()))
            worst = max(rates) / min(rates)
            from .chart_text import fold_words

            out.append(
                f"replicate {day} mixes frame intervals ({detail}): every D scales with "
                f"1/tau, so pooling them is wrong by up to {fold_words(worst, 3)}. Split this "
                "replicate at the rate change, or exclude the odd movies.")
    return out
