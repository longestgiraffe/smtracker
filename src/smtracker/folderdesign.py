"""A project's FOLDER DESIGN: what each folder level between the project folder and the movies means. Headless.

Asked 24 Oct 2026: "Is it possible to build a builder for folder structures so other people can keep their
experimental folder designs? Their own design can be saved. Also grouping in the sqd panel etc could be possible.
For example protein A - glucose, malate / protein B - glucose, malate / Or the other way, glucose - protein A, protein B". Chosen: any
named factors, saved both with the project and in a personal library, for every comparison chart.

A DESIGN is the levels top-down, each with a role (`ROLES`):
* ``factor`` with a name the user gives it (protein, medium, strain, treatment, time point ...): the folder's name is
  that factor's value, and a condition is one combination of factor values;
* ``replicate``: each folder is one replicate (an independent culture) of the condition above it -- however it is
  named (a bare date like ``20240101``, which the automatic rules do not read as a day, included);
* ``skip``: a folder that only wraps the next level (a folder such as ``analysis``, a date that names nothing).
The movies sit below the last level (in ``tif/<movie>``, directly, or in ``trackmate/`` / ``track_tables/``):
`movie_holders` finds them, and every one must sit as deep as the design has levels.

WHAT IT CHANGES. Without a design file nothing changes anywhere. With one (`DESIGN_FILE` in the project folder):
* `project.discover_conditions` takes the conditions from it (`apply`): one per combination of factor values,
  named by its values in level order ("protein A glucose"), each with its folder. This loads three layouts that loaded
  wrongly before: media that number their movies on (they pooled into one condition), replicate folders named by
  bare dates (refused), and a replicate level the automatic rules did not recognise (refused or mis-split);
* `dayfolders` reads a declared replicate level as the condition's replicate folders (`declared_replicates`), so
  every reader of a condition's movies -- outlines, camera files, the file openers -- sees the same replicates;
* `Project.factors` / `Project.factor_names` carry each condition's factor values, for the charts' grouping.

WHAT IT REFUSES, by name (`DesignResult.problems`): movies at another depth than the design's, two folders that
would make one condition, a replicate level that is not right below the last factor (a replicate ABOVE a factor
puts one condition in several folders, which the loader does not read yet), factor names missing, repeated or
reserved, a condition key two value combinations would share.

RULE 0 is untouched: a replicate belongs to its condition; replicate 1 of one condition shares nothing with
replicate 1 of another, whatever their folder names.

Saved as JSON (`write` / `read`, the project's copy) and in a personal library (`library`, `remember`, `forget`,
`matching`: the saved designs that fit a folder), under ``%APPDATA%/SMTracker/designs.json`` or `LIBRARY_ENV`.

Light: no pandas, no Qt (`dayfolders` imports it).
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["DESIGN_FILE", "ROLES", "RESERVED_NAMES", "LIBRARY_ENV", "DesignError", "Level", "FolderDesign",
           "DesignCondition", "DesignResult", "movie_holders", "propose", "apply", "read", "write", "fingerprint",
           "library_path", "library", "remember", "forget", "matching", "declared_replicates", "forget_declared",
           "replicate_order"]

DESIGN_FILE = "smtracker_design.json"
ROLES = ("factor", "replicate", "skip")
#: Names a factor may not take: the canonical table's own columns and the replicate.
RESERVED_NAMES = frozenset({"condition", "replicate", "day", "movie", "cell", "uid", "track", "frame"})
LIBRARY_ENV = "SMTRACKER_DESIGNS"
#: How far below the project folder movies are looked for, and how far up a folder looks for its design.
MAX_DEPTH = 8
_SKIP_DIRS = frozenset({"backups"})
#: Folder names that hold a condition's own files, never a level of the design.
_OWN_FOLDERS = frozenset({"tif", "raw", "phase", "sum", "epi", "cell_meshes", "cell_meshes_preview", "bleaching",
                          "trackmate", "track_tables", "extra"})
#: Words that name a growth medium or carbon source, for `propose`'s guess at a factor's name.
_MEDIA = re.compile(r"^(glu|gluc|glucose|mal|malate|lb|succ|succinate|glycerol|fru|fructose|gm|mm|m9|cit|citrate|"
                    r"pyr|pyruvate|ace|acetate|xyl|xylose|rich|minimal|sgm|smm|smtm|sgmtm)\b", re.IGNORECASE)
_PROTEIN = re.compile(r"(mng|gfp|yfp|mcherry|halo|snap|mscarlet|pa[- ]?jf)", re.IGNORECASE)


class DesignError(ValueError):
    """A design that cannot be used as written (a broken file, an unknown role); the message says why."""


@dataclass(frozen=True)
class Level:
    """One folder level: its role (`ROLES`) and, for a factor, the factor's name."""

    role: str
    name: str = ""

    def to_dict(self) -> dict:
        return {"role": self.role, "name": self.name} if self.role == "factor" else {"role": self.role}


@dataclass(frozen=True)
class FolderDesign:
    """The levels top-down, from the project folder to the folder above the movies."""

    levels: tuple
    #: what the design is called in the library ("" for a project's own)
    name: str = ""

    @property
    def factor_names(self) -> tuple:
        return tuple(level.name for level in self.levels if level.role == "factor")

    @property
    def replicate_index(self) -> int | None:
        found = [i for i, level in enumerate(self.levels) if level.role == "replicate"]
        return found[0] if found else None

    def shape_problems(self) -> list:
        """What is wrong with the design itself, whatever folder it is laid on."""
        problems = []
        roles = [level.role for level in self.levels]
        unknown = sorted({r for r in roles if r not in ROLES})
        if unknown:
            problems.append(f"unknown role {', '.join(unknown)}: a level is a factor, the replicate, or skipped")
        if not self.levels:
            problems.append("a design needs at least one level")
            return problems
        names = self.factor_names
        if not names:
            problems.append("a design needs at least one factor: a condition is a combination of factor values")
        blank = [i + 1 for i, level in enumerate(self.levels) if level.role == "factor" and not level.name.strip()]
        if blank:
            problems.append(f"level {', '.join(map(str, blank))} is a factor without a name")
        lowered = [n.strip().lower() for n in names if n.strip()]
        twice = sorted({n for n in lowered if lowered.count(n) > 1})
        if twice:
            problems.append(f"two levels are called {', '.join(twice)}: every factor needs its own name")
        reserved = sorted(set(lowered) & RESERVED_NAMES)
        if reserved:
            problems.append(f"{', '.join(reserved)} cannot name a factor: the analysis uses that word itself")
        if roles.count("replicate") > 1:
            problems.append("only one level can be the replicate")
        rep = self.replicate_index
        if rep is not None and names:
            last = max(i for i, level in enumerate(self.levels) if level.role == "factor")
            if rep < last:
                problems.append("the replicate level sits above a factor, so one condition would lie in several "
                                "folders, which SMTracker cannot read. Put the replicate right below the "
                                "last factor, or leave it out and let the replicates be found inside each condition")
            elif rep != last + 1:
                problems.append("the replicate level must be right below the last factor; a skipped folder may "
                                "follow it but not come before it")
        return problems

    def to_dict(self) -> dict:
        return {"version": 1, "name": self.name, "levels": [level.to_dict() for level in self.levels]}

    @classmethod
    def from_dict(cls, data) -> "FolderDesign":
        if not isinstance(data, dict) or not isinstance(data.get("levels"), list):
            raise DesignError("a design is a JSON object with a list of levels")
        levels = []
        for item in data["levels"]:
            if not isinstance(item, dict) or item.get("role") not in ROLES:
                raise DesignError(f"a level is a factor, the replicate or skipped, got {item!r}")
            levels.append(Level(str(item["role"]), str(item.get("name") or "").strip()))
        return cls(levels=tuple(levels), name=str(data.get("name") or ""))

    def describe(self) -> str:
        """``protein / medium / replicate`` -- the levels in words."""
        return " / ".join(level.name if level.role == "factor" else
                          ("replicate" if level.role == "replicate" else "(skipped)") for level in self.levels)


@dataclass(frozen=True)
class DesignCondition:
    """One condition a design finds: its key, its factor values in level order, its folder and its declared
    replicate folders (empty when the design has no replicate level: the automatic rules then find them)."""

    key: str
    values: tuple
    folder: Path
    replicates: tuple = ()


@dataclass
class DesignResult:
    design: FolderDesign
    root: Path
    conditions: list = field(default_factory=list)
    problems: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems and bool(self.conditions)

    def available(self) -> dict:
        """``{condition key: folder}``, what `project.discover_conditions` returns."""
        return {c.key: c.folder for c in self.conditions}

    def factors(self) -> dict:
        """``{condition key: {factor name: value}}``."""
        names = self.design.factor_names
        return {c.key: dict(zip(names, c.values)) for c in self.conditions}


# --------------------------------------------------------------------------- the folder tree
def _table_kinds():
    from .tracktable import TABLE_FOLDER, TABLE_SUFFIXES

    return TABLE_FOLDER, TABLE_SUFFIXES


def movie_holders(root) -> list:
    """Each folder that holds movies, as its path below ``root`` (a tuple of folder names): the parent of u-track
    movie folders (``tif/`` counted as the condition's own, so ``<condition>/tif/<movie>`` gives ``<condition>``),
    of a ``trackmate/`` folder of XMLs, or of a ``track_tables/`` folder. ``()`` is ``root`` itself."""
    root = Path(root)
    table_folder, table_suffixes = _table_kinds()
    found = set()
    for here, subs, files in os.walk(root):
        depth = len(Path(here).relative_to(root).parts)
        subs[:] = [d for d in subs if d not in _SKIP_DIRS and not d.startswith(".")]
        if depth > MAX_DEPTH + 2:
            subs[:] = []
            continue
        base = Path(here)
        holder = None
        if "TrackingPackage" in subs:
            holder = base.parent
            subs[:] = []                       # never into a movie: its frames are thousands of files
        elif base.name == "trackmate" and any(f.lower().endswith(".xml") for f in files):
            holder = base.parent
        elif base.name == table_folder and any(os.path.splitext(f)[1].lower() in table_suffixes for f in files):
            holder = base.parent
        if holder is None:
            continue
        if holder.name.lower() == "tif" and holder != root:
            holder = holder.parent
        try:
            found.add(holder.relative_to(root).parts)
        except ValueError:                     # pragma: no cover - the walk stays below root
            continue
    return sorted(found)


_BARE_DATE = re.compile(r"^(20\d{2})(\d{2})(\d{2})$")


def _date_value(name: str) -> int | None:
    """``yyyymmdd`` for a folder named by a date: the forms `dayfolders` reads, and a bare ``20240101``."""
    from .dayfolders import _date_key

    dated = _date_key(name)
    if dated is not None:
        return dated
    match = _BARE_DATE.match(str(name).strip())
    if match:
        year, month, day = (int(g) for g in match.groups())
        try:
            datetime.date(year, month, day)
        except ValueError:
            return None
        return year * 10000 + month * 100 + day
    return None


def replicate_order(names, parent: str = "") -> list:
    """``names`` in replicate order: by date where every one is a date, else by the number each carries (``1``,
    ``rep2``, ``rep3``), else by name. Replicate k is the k-th."""
    from .dayfolders import day_number

    names = list(names)
    dates = [_date_value(n) for n in names]
    if names and all(d is not None for d in dates):
        return [n for _d, n in sorted(zip(dates, names))]
    numbers = [day_number(n, parent) for n in names]
    if names and all(n is not None for n in numbers) and len(set(numbers)) == len(numbers):
        return [n for _k, n in sorted(zip(numbers, names))]
    return sorted(names)


# --------------------------------------------------------------------------- proposing and applying
def _looks_like_replicates(names, parents) -> bool:
    from .dayfolders import day_number

    names = list(names)
    if len(set(names)) < 2:
        return False
    for name, parent in zip(names, parents):
        if _date_value(name) is None and day_number(name, parent) is None:
            return False
    return True


def _guess_name(values, k: int, taken) -> str:
    values = [str(v) for v in values]
    if values and all(_MEDIA.match(v) for v in values):
        guess = "medium"
    elif values and sum(bool(_PROTEIN.search(v)) for v in values) >= max(1, len(values) // 2):
        guess = "protein"
    else:
        guess = f"factor {k}"
    while guess in taken:
        guess = f"factor {k}" if not guess.startswith("factor") else f"factor {k + 1}"
        k += 1
    return guess


def propose(root, *, holders=None) -> FolderDesign | None:
    """A first design for the folders under ``root``, to be checked and named by the user: None when the movies do
    not all sit at one depth below it, or sit in ``root`` itself (one condition: no design is needed).

    Per level: every folder named as a replicate (a number, after its parent's name where it carries it, or a date,
    bare ``20240101`` included) makes the replicate level, if it is the deepest level that varies; a level with one
    name everywhere is skipped (a wrapper); every other level is a factor, called "medium" when its names are growth
    media, "protein" when they name fluorescent fusions, else "factor k"."""
    holders = movie_holders(root) if holders is None else list(holders)
    depths = {len(p) for p in holders}
    if not holders or len(depths) != 1 or 0 in depths:
        return None
    depth = depths.pop()
    levels = []
    taken = set()
    for i in range(depth):
        names = [p[i] for p in holders]
        parents = [p[i - 1] if i else Path(root).name for p in holders]
        distinct = sorted(set(names))
        deeper_varies = any(len({p[j] for p in holders}) > 1 for j in range(i + 1, depth))
        if len(distinct) == 1:
            levels.append(Level("skip"))
        elif not deeper_varies and _looks_like_replicates(names, parents) and any(
                lv.role == "factor" for lv in levels):
            levels.append(Level("replicate"))
        else:
            name = _guess_name(distinct, sum(lv.role == "factor" for lv in levels) + 1, taken)
            taken.add(name)
            levels.append(Level("factor", name))
    design = FolderDesign(tuple(levels))
    return design if not design.shape_problems() else None


def apply(design: FolderDesign, root, *, holders=None) -> DesignResult:
    """The conditions ``design`` finds under ``root``, or the problems that stop it, by name (module docstring).
    ``holders``: `movie_holders(root)` already listed (a page previewing every change reuses one walk)."""
    root = Path(root)
    result = DesignResult(design=design, root=root, problems=list(design.shape_problems()))
    if result.problems:
        return result
    holders = movie_holders(root) if holders is None else list(holders)
    if not holders:
        result.problems.append(f"no movies under {root}")
        return result
    n = len(design.levels)
    wrong = [p for p in holders if len(p) != n]
    for p in wrong[:5]:
        where = "/".join(p) if p else "the project folder itself"
        result.problems.append(f"{where} holds movies {len(p)} folder{'s' if len(p) != 1 else ''} deep; "
                               f"the design has {n} level{'s' if n != 1 else ''}")
    if len(wrong) > 5:
        result.problems.append(f"... and {len(wrong) - 5} more folders at another depth")
    if wrong:
        return result
    factor_at = [i for i, level in enumerate(design.levels) if level.role == "factor"]
    last = max(factor_at)
    rep = design.replicate_index
    by_values: dict = {}
    for parts in holders:
        values = tuple(parts[i] for i in factor_at)
        folder = root.joinpath(*parts[:last + 1])
        entry = by_values.setdefault(values, {"folders": set(), "replicates": set()})
        entry["folders"].add(folder)
        if rep is not None:
            entry["replicates"].add(parts[rep])
    keys: dict = {}
    for values in sorted(by_values):
        entry = by_values[values]
        if len(entry["folders"]) > 1:
            shown = ", ".join(sorted(str(f.relative_to(root)) for f in entry["folders"]))
            result.problems.append(f"{' / '.join(values)} would be one condition in several folders ({shown}): "
                                   f"they differ only in a skipped level; make that level a factor or the replicate")
            continue
        key = " ".join(values)
        if key in keys:
            result.problems.append(f"{' / '.join(values)} and {' / '.join(keys[key])} would both be called {key!r}")
            continue
        keys[key] = values
        folder = next(iter(entry["folders"]))
        replicates = tuple(folder / name for name in replicate_order(entry["replicates"], folder.name))
        result.conditions.append(DesignCondition(key=key, values=values, folder=folder, replicates=replicates))
    if result.problems:
        result.conditions = []
    return result


# --------------------------------------------------------------------------- saved designs
def read(root) -> FolderDesign | None:
    """The design saved in ``root`` (`DESIGN_FILE`), or None when there is none. A file that cannot be read as a
    design raises `DesignError` naming it: a broken design must not load the project another way silently."""
    path = Path(root) / DESIGN_FILE
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DesignError(f"{path} is not a readable folder design ({exc})") from None
    try:
        return FolderDesign.from_dict(data)
    except DesignError as exc:
        raise DesignError(f"{path}: {exc}") from None


def write(design: FolderDesign, root, *, by: str = "") -> Path:
    """Save ``design`` as the project's own (`DESIGN_FILE` in ``root``), after checking that it fits."""
    problems = apply(design, root).problems
    if problems:
        raise DesignError("the design does not fit this folder: " + "; ".join(problems))
    data = design.to_dict()
    data["saved_on"] = datetime.date.today().isoformat()
    if by:
        data["saved_by"] = by
    path = Path(root) / DESIGN_FILE
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    forget_declared()
    return path


def fingerprint(design: FolderDesign | None) -> str:
    """A short hash of what a design changes in a load (its levels), for cache keys; "" for none."""
    if design is None:
        return ""
    blob = json.dumps([level.to_dict() for level in design.levels], sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def library_path() -> Path:
    """The personal library: `LIBRARY_ENV` when set (tests, a shared lab folder), else the user's settings."""
    override = os.environ.get(LIBRARY_ENV)
    if override:
        return Path(override).expanduser()
    if os.name == "nt" and os.environ.get("APPDATA"):
        return Path(os.environ["APPDATA"]) / "SMTracker" / "designs.json"
    base = os.environ.get("XDG_CONFIG_HOME")
    return (Path(base).expanduser() if base else Path.home() / ".config") / "smtracker" / "designs.json"


def library() -> dict:
    """``{name: FolderDesign}`` saved in the personal library, in the order saved; {} when there is none. A broken
    library raises `DesignError` (it is the user's own file, and a silent empty library would lose it)."""
    path = library_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DesignError(f"{path} is not a readable design library ({exc})") from None
    out = {}
    for item in data.get("designs", []) if isinstance(data, dict) else []:
        design = FolderDesign.from_dict(item)
        out[design.name] = design
    return out


def remember(name: str, design: FolderDesign) -> Path:
    """Save ``design`` in the personal library under ``name`` (replacing one of that name)."""
    name = str(name).strip()
    if not name:
        raise DesignError("a design in the library needs a name")
    problems = design.shape_problems()
    if problems:
        raise DesignError("; ".join(problems))
    saved = library()
    saved[name] = FolderDesign(design.levels, name=name)
    path = library_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": 1, "designs": [d.to_dict() for d in saved.values()]}, indent=1),
                    encoding="utf-8")
    return path


def forget(name: str) -> None:
    saved = library()
    if name not in saved:
        raise DesignError(f"no design called {name!r} in the library")
    del saved[name]
    library_path().write_text(json.dumps({"version": 1, "designs": [d.to_dict() for d in saved.values()]},
                                         indent=1), encoding="utf-8")


def matching(root, *, holders=None) -> list:
    """``[(name, FolderDesign)]``: the library's designs that fit ``root`` without a problem, in library order."""
    holders = movie_holders(root) if holders is None else list(holders)
    return [(name, design) for name, design in library().items() if apply(design, root, holders=holders).ok]


# --------------------------------------------------------------------------- declared replicates, for dayfolders
#: ``{folder: (design file, its mtime, replicate folders or None)}``. `dayfolders` asks once per image a screen draws.
_DECLARED: dict = {}


def forget_declared() -> None:
    """Drop what `declared_replicates` remembered (a design was saved or removed)."""
    _DECLARED.clear()


def _design_above(folder: Path):
    """``(design file, the folder holding it)`` for the nearest design at or above ``folder``, or None."""
    here = folder
    for _ in range(MAX_DEPTH + 1):
        candidate = here / DESIGN_FILE
        if candidate.is_file():
            return candidate, here
        if here.parent == here:
            return None
        here = here.parent
    return None


def declared_replicates(folder) -> list | None:
    """``folder``'s replicate folders in replicate order when a design above it (or in it) makes ``folder`` a
    condition whose next level is the replicate; else None (the automatic rules apply). A broken design raises."""
    folder = Path(folder)
    key = str(folder)
    hit = _DECLARED.get(key)
    if hit is not None:
        path, stamp, value = hit
        try:
            if path is None or path.stat().st_mtime_ns == stamp:
                return value
        except OSError:
            pass
    found = _design_above(folder)
    if found is None:
        _DECLARED[key] = (None, 0, None)
        return None
    path, root = found
    design = read(root)
    value = None
    rep = design.replicate_index if design is not None else None
    if rep is not None and not design.shape_problems():
        try:
            depth = len(folder.relative_to(root).parts)
        except ValueError:
            depth = -1
        if depth == rep:
            from .dayfolders import has_movies

            try:
                kids = [c for c in folder.iterdir() if c.is_dir() and c.name not in _SKIP_DIRS
                        and not c.name.startswith(".") and c.name not in _OWN_FOLDERS and has_movies(c)]
            except OSError:
                kids = []
            if kids:
                order = replicate_order([c.name for c in kids], folder.name)
                value = [folder / name for name in order]
    _DECLARED[key] = (path, path.stat().st_mtime_ns, value)
    return value
