"""A condition whose replicate DAYS are sub-folders that start their movie names again.

`cond/rep1`, `rep2`, `rep3` (and `cond/1`, `/2`, `/3`): each day folder holds its own
outlines and `tif/series001 ...`, and every day starts again at series001. Asked 16 Oct 2026 ("sure", to loading
such a folder as ONE condition with its days). Before, it could not be loaded at all: discovery refuses two movies of
one name, rightly, since a movie's name is its identity everywhere downstream.

THE RULE (`day_folders`); all of it must hold, or the folder is what it was before:
- two or more of its sub-folders hold movies, and EVERY one of those is NAMED AS A DAY of this folder: a number,
  after the folder's own name where it carries it (`1`, `rep1` in `cond`, `M2c` in `M`, `day 2`, `r3`), or a DATE
  written out in full (`10.03.2024`, `2025-03-10`; days then in date order; never a bare `240101`);
- no one of them holds a movie name twice (a dated folder holding several conditions is not a day);
- the movie names REPEAT across them. With unique names the folder loads as one condition (its days from its
  numbered folders, NUMBERED ON below, else the series naming), so nothing that loaded before changes.
A folder of CONDITIONS (one user's `protein/cond_LB`, `protein/cond_MM-fructose`) fails the naming test and stays a group
of conditions. The test leans that way on purpose: different media merged as days would be wrong without a sign,
while a day folder taken for a condition shows its name.

THE NAMES. A movie becomes ``series001 (rep1)`` (`movie_name`): unique, readable, its series number still the first
digits (`grouping.series_number`), and without ``:`` (a cell is ``<movie>:<n>``). `on_disk` turns it back into
(``.../cond/rep1``, ``series001``) for whatever opens a file; `Project.movie_source` keeps the CONDITION's folder, which
is what every screen compares to tell a condition's movies apart.

THE DAYS. Day k is the k-th folder in the order of its number (`day_map`); the Confirm screen's group board shows it
and can change it, as for any condition.

NUMBERED ON (`numbered_folders`, asked 20 Oct 2026 for `strain/cond-LB-rif-1`, `-2`,
`-3`, `cond/strain/LB/LB-1`.. and a strain's triplicates: "loads the days like they would be together
sitting in one folder"). The days are sibling folders under ONE shared name, numbered 1, 2, 3, each with its own
outlines and `tif/`, and the series numbers run ON across them (series001-010, 011-020, 021-030), so no name
repeats and nothing is renamed. Discovery already took such a folder as one condition one level down, but its
outlines were read from the FIRST `cell_meshes` found, so days 2 and 3 were refused as unsegmented ("no cell
outlines for movies: series011 ..."), and opened directly each day folder came back as its own condition. The
rule, all of which must hold:
- the folders holding movies are siblings whose names, the name they share taken off, are the numbers 1..n
  (`cond-LB-rif-1` .. `-3`, `LB-1` .. `-3`, `3610-strain-1` .. `-3`); two or more of them;
- they may sit below folders that only WRAP them, one sub-folder each (`experiment/20240101/...`,
  `cond-LB/strain-mNG/cond-LB-Triplicates/...`);
- no movie name repeats across them (that is `day_folders`' layout) and none lies outside them.
Day k is the folder numbered k, whatever its series say; `on_disk` finds a movie's own folder for the openers.
Numbers that do not start at 1 or skip one (`IPTG-10`, `IPTG-100`) are not days: such folders read as before.

A FOLDER DESIGN DECLARES THEM (`folderdesign`, 24 Oct 2026): where the project's design makes a folder a condition
whose next level is the replicate, its sub-folders holding movies ARE its replicates, in the design's replicate order,
whatever they are called -- a bare ``20240101`` included, which the rules above refuse. Repeating movie names make
them `day_folders` (the movies renamed ``series001 (20240101)``), unique ones `numbered_folders`. Without a design
file nothing here changes.

Headless and light: no pandas, no Qt. Read by `grouping` (discovery, the default days, the day authority),
`cellmesh` (outlines), `project` (conditions, the cache key), `daymap` and the file openers (`trackclip`,
`images_view`, `viewer`).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

__all__ = ["day_folders", "day_map", "day_number", "has_movies", "movie_name", "movie_names", "namer",
           "numbered_folders", "on_disk", "replicate_folders", "split_movie"]

#: What a day folder's name says once the condition's own name is taken off its front.
_DAY_NAME = re.compile(r"^(?:day|tag|replicate|rep|r|d)?[\s_-]*(\d{1,3})[a-z]?$", re.IGNORECASE)
#: A day folder named by its DATE, written out in full: ``10.03.2024`` (or ``10.3.25``), ``2025-03-10``. Never a
#: bare run of digits (a user's ``240101`` parents hold conditions, not days). Asked 20 Oct 2026 for
#: ``D:/data/project/condition/21.03.2024``, ``19.03.2024``, ``21.03.2024``.
_DOTTED_DATE = re.compile(r"^(\d{1,2})\.(\d{1,2})\.(\d{4}|\d{2})$")
_ISO_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")


def _date_key(name: str) -> int | None:
    """``yyyymmdd`` for a folder named by a date in full (`_DOTTED_DATE`, `_ISO_DATE`), else None."""
    text = str(name).strip()
    match = _DOTTED_DATE.match(text)
    if match:
        day, month, year = (int(g) for g in match.groups())
        year += 2000 if year < 100 else 0
    else:
        match = _ISO_DATE.match(text)
        if not match:
            return None
        year, month, day = (int(g) for g in match.groups())
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    return year * 10000 + month * 100 + day
#: ``series001 (rep1)``: a movie and the day folder it came from.
_MOVIE = re.compile(r"^(?P<movie>.+) \((?P<folder>[^()]+)\)$")
#: Never movies: u-track's own copies.
_SKIP = frozenset({"backups"})


def _table_kinds():
    from .tracktable import TABLE_FOLDER, TABLE_SUFFIXES

    return TABLE_FOLDER, TABLE_SUFFIXES


def movie_names(folder, *, utrack: bool = True, trackmate: bool = True, table: bool = True,
                first_only: bool = False) -> list:
    """The movies under ``folder`` by name: u-track movie folders (never entered: their frames are thousands of
    files), TrackMate XMLs and track tables. ``first_only`` stops at the first one found."""
    table_folder, table_suffixes = _table_kinds()
    names = []
    for here, subs, files in os.walk(folder):
        subs[:] = [d for d in subs if d not in _SKIP and not d.startswith(".")]
        if utrack and "TrackingPackage" in subs:
            names.append(os.path.basename(here))
            subs[:] = []
        else:
            base = os.path.basename(here)
            if trackmate and base == "trackmate":
                names += [f[:-4] for f in files if f.lower().endswith(".xml")]
            if table and base == table_folder:
                names += [os.path.splitext(f)[0] for f in files
                          if os.path.splitext(f)[1].lower() in table_suffixes]
        if first_only and names:
            break
    return names


def has_movies(folder) -> bool:
    return bool(movie_names(folder, first_only=True))


def day_number(name: str, parent: str) -> int | None:
    """The day a sub-folder's NAME gives, inside a folder named ``parent``; None when it names no day. A folder
    named by a date gives ``yyyymmdd``, so dated days sort in the order they were recorded."""
    dated = _date_key(name)
    if dated is not None:
        return dated
    rest = str(name).strip()
    p = str(parent).strip()
    if p and len(rest) > len(p) and rest.lower().startswith(p.lower()):
        rest = rest[len(p):]
    match = _DAY_NAME.match(rest.strip(" _-"))
    return int(match.group(1)) if match else None


def _declared(root):
    """The replicate folders a folder design declares for ``root`` (`folderdesign.declared_replicates`), or None."""
    from .folderdesign import declared_replicates

    return declared_replicates(root)


def _names_repeat(folders) -> bool:
    seen = set()
    for folder in folders:
        names = set(movie_names(folder))
        if seen & names:
            return True
        seen |= names
    return False


def day_folders(root) -> list | None:
    """``root``'s day folders in day order when it is ONE condition whose replicate days are sub-folders reusing
    movie names (the module's rule, or a folder design's declaration), else None."""
    root = Path(root)
    declared = _declared(root)
    if declared is not None:
        return declared if _names_repeat(declared) else None
    try:
        kids = sorted(c for c in root.iterdir()
                      if c.is_dir() and c.name not in _SKIP and not c.name.startswith("."))
    except OSError:
        return None
    days, seen, repeated = [], set(), False
    for kid in kids:
        number = day_number(kid.name, root.name)
        if number is None:
            if has_movies(kid):
                return None                         # a folder of movies that is not a day: not this layout
            continue                                # raw/, phase/, ...: no movies, no say
        listed = movie_names(kid)
        names = set(listed)
        if not names:
            continue
        if len(listed) != len(names):
            return None                         # one "day" holding a name twice is several conditions, not a day
        repeated = repeated or bool(seen & names)
        seen |= names
        days.append((number, kid.name, kid))
    if len(days) < 2 or not repeated:
        return None
    return [kid for _n, _name, kid in sorted(days)]


#: How many folders may only WRAP the numbered ones (`cond-LB/strain-mNG/cond-LB-Triplicates/...` is two).
_MAX_WRAP = 4


def _subfolders(folder: Path) -> list:
    try:
        return sorted(c for c in folder.iterdir()
                      if c.is_dir() and c.name not in _SKIP and not c.name.startswith("."))
    except OSError:
        return []


def _numbers(names) -> list | None:
    """Each name's number once the name they all share is taken off its front, when the numbers are 1..n; else
    None. The shared part never ends inside a number (`rep9` and `rep10` share `rep`, not `rep1`)."""
    stem = os.path.commonprefix([str(n).lower() for n in names]).rstrip("0123456789")
    numbers = []
    for name in names:
        match = _DAY_NAME.match(str(name)[len(stem):].strip(" _-"))
        if match is None:
            return None
        numbers.append(int(match.group(1)))
    return numbers if sorted(numbers) == list(range(1, len(numbers) + 1)) else None


def numbered_folders(root) -> list | None:
    """``root``'s replicate folders in day order when it is ONE condition whose days are sibling folders numbered
    1..n under one shared name, its movies numbered ON across them (the module's NUMBERED ON rule, or a folder
    design's declaration with unique movie names), else None."""
    declared = _declared(Path(root))
    if declared is not None:
        return None if _names_repeat(declared) else declared
    here = Path(root)
    for _ in range(_MAX_WRAP + 1):
        kids = _subfolders(here)
        if any((kid / "TrackingPackage").is_dir() for kid in kids):
            return None                                 # movies right here: a folder of movies, not of days
        holding = [kid for kid in kids if has_movies(kid)]
        if len(holding) != 1:
            break
        here = holding[0]                               # a folder that only wraps the next level down
    else:
        return None
    if len(holding) < 2:
        return None
    numbers = _numbers([kid.name for kid in holding])
    if numbers is None:
        return None
    seen = set()
    for kid in holding:
        names = set(movie_names(kid))
        if seen & names:
            return None                                 # a name twice: `day_folders`' layout, or two conditions
        seen |= names
    return [kid for _n, kid in sorted(zip(numbers, holding))]


def replicate_folders(root) -> list | None:
    """``root``'s replicate-day folders in day order, of either layout (`day_folders`, `numbered_folders`)."""
    return day_folders(root) or numbered_folders(root)


#: ``{root: (its mtime, {movie: the numbered folder holding it})}``. `on_disk` is asked once per image a screen
#: draws, and finding the folders walks the condition.
_HOMES: dict = {}


def _homes(root: Path, *, fresh: bool = False) -> dict:
    try:
        stamp = root.stat().st_mtime_ns
    except OSError:
        return {}
    hit = None if fresh else _HOMES.get(str(root))
    if hit is not None and hit[0] == stamp:
        return hit[1]
    homes = {name: folder for folder in (numbered_folders(root) or []) for name in movie_names(folder)}
    _HOMES[str(root)] = (stamp, homes)
    return homes


def namer(root):
    """``name(movie_folder) -> str``: the name the condition at ``root`` knows a movie folder by --
    ``series001 (19.03.2024)`` inside a day folder that reuses movie names (`day_folders`), else the folder's own
    name. For every reader that walks the movie folders itself (`utrack_meta.read_project_metadata`, the camera
    files, the recorded intervals), so their names are the loaders' (found 20 Oct 2026: the camera's intervals and
    "leave out" used ``series002`` where the condition has two of them)."""
    root = Path(root)
    days = {f.name for f in (day_folders(root) or [])}

    def name(movie_folder) -> str:
        movie_folder = Path(movie_folder)
        if days:
            try:
                first = movie_folder.relative_to(root).parts[0]
            except ValueError:
                first = None
            if first in days:
                return movie_name(movie_folder.name, first)
        return movie_folder.name

    return name


def movie_name(movie: str, folder: str) -> str:
    """A day folder's movie as the condition knows it: ``series001 (rep1)``."""
    return f"{movie} ({folder})"


def split_movie(name: str) -> tuple | None:
    """``(movie, folder)`` from ``series001 (rep1)``, or None for a plain name."""
    match = _MOVIE.match(str(name))
    return (match.group("movie"), match.group("folder")) if match else None


def on_disk(root, movie: str) -> tuple:
    """``(folder, movie)`` where a movie's files are: ``(root/rep1, "series001")`` for ``series001 (rep1)`` when that
    day folder exists, ``(.../LB/LB-2, "series011")`` for a movie of a numbered day folder (`numbered_folders`),
    else ``(root, movie)`` unchanged."""
    parts = split_movie(movie)
    if parts is not None:
        folder = Path(root) / parts[1]
        if folder.is_dir():
            return folder, parts[0]
    homes = _homes(Path(root))
    home = homes.get(str(movie))
    if home is None and homes:
        home = _homes(Path(root), fresh=True).get(str(movie))     # a day folder added below a wrapping folder
    if home is not None and home.is_dir():
        return home, str(movie)
    return Path(root), str(movie)


def day_map(root, movies) -> dict | None:
    """``{movie: day}`` for a day-folder condition's movies (``series001 (rep1)`` -> rep1's place in day order; a
    numbered day folder's ``series011`` -> its folder's number), or None when ``root`` is neither. Refuses a movie it
    cannot place, rather than leaving it a day by guesswork."""
    folders = day_folders(root)
    if folders is None:
        numbered = numbered_folders(root)
        if numbered is None:
            return None
        rank = {name: k for k, folder in enumerate(numbered, start=1) for name in movie_names(folder)}
        missing = [str(m) for m in movies if str(m) not in rank]
        if missing:
            raise ValueError(f"{missing[0]!r} is not in any replicate folder of {root} "
                             f"({', '.join(f.name for f in numbered)})")
        return {str(m): rank[str(m)] for m in movies}
    rank = {f.name: k for k, f in enumerate(folders, start=1)}
    out = {}
    for movie in movies:
        parts = split_movie(movie)
        if parts is None or parts[1] not in rank:
            raise ValueError(f"{movie!r} is not in any replicate folder of {root}")
        out[str(movie)] = rank[parts[1]]
    return out
