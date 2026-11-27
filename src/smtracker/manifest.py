"""Project manifest: content-addressed units for incremental, growing projects.

Adding a movie should cost O(new data), not a full re-fit. That requires knowing,
cheaply, what is on disk, what has already been fitted, and what changed. This
module scans a project into a **content-addressed manifest** (each movie carries
a hash of its tracking result), diffs a rescan against a stored manifest, and
classifies every unit as ingested / new-unfitted / stale.

Grouping is from the folder hierarchy: the source of truth
-----------------------------------------------------------
The folder tree, not u-track's internal metadata, defines the study structure. A
study may be laid out ``<root>/protein/condition/…`` or
``<root>/experiment/protein/…``; the mapping from **nesting level → role** is
therefore configurable (``roles=("protein", "condition")`` etc.). Two rules hold
regardless:

- **Unit identity is the full folder path** to a movie: two ``series001`` under
  different conditions are different units, always.
- **The comparison axis is the level the user varies**, whichever role differs
  across the units being compared. The manifest records every level so any axis
  can be chosen later without re-scanning.

Why content-addressed
---------------------
A movie's fingerprint is a hash of its tracking-result bytes. So "did this movie
change?" is answered without re-ingesting it, and re-running u-track on one movie
marks exactly that movie stale: its neighbours stay cached.

Invariants
----------
- ``backups/`` is skipped (as everywhere else).
- A movie whose ``.mat`` is unreadable is recorded with a ``None`` hash and
  flagged, never silently dropped.
- The manifest round-trips through JSON so it can be stored beside a cache.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from .grouping import UTRACK_GLOB

__all__ = [
    "ManifestError",
    "MovieUnit",
    "scan",
    "diff",
    "classify",
    "derive_roles",
]


class ManifestError(RuntimeError):
    """Raised when a project cannot be scanned into a manifest."""


#: The fixed tail of every movie path: ``<movie>/TrackingPackage/tracks/*.mat``.
#: Everything *above* ``<movie>`` is grouping structure.
_TAIL_DEPTH = 3   # tracks-file → tracks → TrackingPackage → <movie>


def _movie_dir(mat_path: Path) -> Path:
    return mat_path.parents[_TAIL_DEPTH - 1]


def _content_hash(mat_path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with open(mat_path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def derive_roles(levels: list[str], roles) -> dict:
    """Map folder levels above a movie to named roles.

    ``levels`` are the path parts from ``root`` down to (but excluding) the movie
    folder. ``roles`` names them from the OUTERMOST in. Extra inner levels beyond
    the named roles are recorded as ``extra_0``, ``extra_1`` … so nothing is
    lost; fewer levels than roles is an error (the layout does not match).
    """
    roles = list(roles)
    if len(levels) < len(roles):
        raise ManifestError(
            f"path has {len(levels)} grouping level(s) {levels} but {len(roles)} "
            f"role(s) {roles} were requested: the folder layout does not match"
        )
    assigned = {role: levels[i] for i, role in enumerate(roles)}
    for j, extra in enumerate(levels[len(roles):]):
        assigned[f"extra_{j}"] = extra
    return assigned


@dataclass(frozen=True)
class MovieUnit:
    """One movie on disk: its path identity, its roles, and its content hash."""

    path: str                    # full path identity (relative to root)
    movie: str                   # the movie folder name
    roles: dict                  # role → value, from the folder hierarchy
    content_hash: str | None     # sha256 of the tracking result, or None if bad

    @property
    def ok(self) -> bool:
        return self.content_hash is not None

    def to_dict(self) -> dict:
        return {
            "path": self.path, "movie": self.movie, "roles": self.roles,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MovieUnit":
        return cls(data["path"], data["movie"], data["roles"], data["content_hash"])


def scan(root, *, roles=("condition",), pattern: str = UTRACK_GLOB) -> list[MovieUnit]:
    """Scan a project into a content-addressed manifest, one entry per movie.

    ``roles`` names the folder levels above each movie from the outermost in.
    Default ``("condition",)`` matches the ``<root>/<condition>/…`` layout the
    project loader uses; pass e.g. ``("protein", "condition")`` for a deeper
    study. Units are sorted by path for a stable manifest.
    """
    root = Path(root)
    if not root.is_dir():
        raise ManifestError(f"project root not found: {root}")

    units = []
    for mat in sorted(root.glob(pattern)):
        if "backups" in mat.parts:
            continue
        movie_dir = _movie_dir(mat)
        levels = list(movie_dir.relative_to(root).parts[:-1])
        units.append(MovieUnit(
            path=str(movie_dir.relative_to(root)).replace("\\", "/"),
            movie=movie_dir.name,
            roles=derive_roles(levels, roles),
            content_hash=_content_hash(mat),
        ))
    if not units:
        raise ManifestError(
            f"no movie ({pattern!r}) found under {root}: not a u-track project?"
        )
    return units


@dataclass(frozen=True)
class ManifestDiff:
    """What a rescan changed relative to a stored manifest."""

    added: list = field(default_factory=list)       # new movies (path, unit)
    removed: list = field(default_factory=list)      # gone from disk (path)
    changed: list = field(default_factory=list)      # data changed (path, unit)
    unchanged: list = field(default_factory=list)    # same content (path)

    @property
    def is_empty(self) -> bool:
        return not (self.added or self.removed or self.changed)

    def summary(self) -> str:
        return (f"{len(self.added)} added, {len(self.changed)} changed, "
                f"{len(self.removed)} removed, {len(self.unchanged)} unchanged")

    def affected_paths(self) -> list:
        """Paths whose fit must be (re)computed: added and changed only.

        This is the O(new data) set: an aggregate over a role need only refit
        these units and reuse cached fits for the ``unchanged`` ones.
        """
        return [u.path for u in self.added] + [u.path for u in self.changed]


def diff(old: list[MovieUnit], new: list[MovieUnit]) -> ManifestDiff:
    """Rescan-and-diff: what changed between a stored manifest and a fresh scan."""
    old_by_path = {u.path: u for u in old}
    new_by_path = {u.path: u for u in new}

    added = [u for p, u in new_by_path.items() if p not in old_by_path]
    removed = [p for p in old_by_path if p not in new_by_path]
    changed, unchanged = [], []
    for path in new_by_path.keys() & old_by_path.keys():
        if new_by_path[path].content_hash != old_by_path[path].content_hash:
            changed.append(new_by_path[path])
        else:
            unchanged.append(path)
    return ManifestDiff(added=added, removed=removed, changed=changed,
                        unchanged=unchanged)


def classify(manifest: list[MovieUnit], fitted_hashes) -> dict:
    """Split the current manifest into ingested / new-unfitted / stale.

    ``fitted_hashes`` maps ``path → content_hash`` for units that have a cached
    fit. A unit is **ingested** if its current hash matches the fitted one,
    **stale** if it has a fitted entry but the hash moved, and **new-unfitted**
    if it has no fitted entry at all. This is the report the UI shows for a
    growing project: what is ready, what changed, and what is waiting.
    """
    ingested, new_unfitted, stale, unreadable = [], [], [], []
    for unit in manifest:
        if not unit.ok:
            unreadable.append(unit.path)
        elif unit.path not in fitted_hashes:
            new_unfitted.append(unit.path)
        elif fitted_hashes[unit.path] == unit.content_hash:
            ingested.append(unit.path)
        else:
            stale.append(unit.path)
    return {
        "ingested": ingested,
        "new_unfitted": new_unfitted,
        "stale": stale,
        "unreadable": unreadable,
    }


def to_json(manifest: list[MovieUnit], path) -> Path:
    path = Path(path)
    path.write_text(json.dumps([u.to_dict() for u in manifest], indent=2), "utf-8")
    return path


def from_json(path) -> list[MovieUnit]:
    data = json.loads(Path(path).read_text("utf-8"))
    return [MovieUnit.from_dict(d) for d in data]
