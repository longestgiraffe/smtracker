"""Persistence cache: keyed by the provenance stamp, refuses stale loudly.

The problem
-----------
Re-ingesting a project and re-running a bootstrap costs minutes each time, even
when nothing changed. This caches (a) parsed track tables and (b) fit/result
objects, so reopening an unchanged project loads instantly, and toggling a
setting back to one already computed returns the stored result.

The key is the provenance
-------------------------
A cache entry is keyed by ``provenance.key()``, the SHA-256 over the
reproducibility-relevant material (input data + settings + seeds + code). Two
runs with the same inputs share a key and hit; changing any of data / settings /
seed / code-commit changes the key and misses. This is why provenance was built
first: the cache is a thin store over that key.

Refuse stale, LOUDLY: with one graded exception
------------------------------------------------
A miss is not silent: ``load_or_raise`` names what differs from the closest
stored entry. Most mismatches (data, settings, seed, code commit) are **hard
misses**: recompute, no question. One class is softer: a **version-only**
mismatch, where everything matches except a load-bearing package version
(``numpy 2.5.1 → 2.5.2``). That *could* change a number, so it is not a silent
hit, but blindly refusing it forces an hours-long refit for a patch bump that
usually changes nothing. So it is a *named, overridable* stale: the caller is
told exactly which package moved and may ``reuse_on_version_mismatch=True`` to
take the stored result anyway (the reused result keeps its original stamp, so
the provenance never lies about what produced it).

Layout
------
``<cache_root>/<kind>/<key>.<ext>`` plus ``<key>.provenance.json`` beside each,
so any cached artefact is self-describing on disk. Tables are parquet (falling
back to pickle), result objects are pickle, and a human-readable CSV export
carries the provenance header.

Invariants
----------
- Nothing is stored without a provenance stamp.
- A corrupt or unreadable entry is treated as a miss, not a crash.
- The cache never fabricates provenance: a reused-on-override result carries the
  stamp it was computed with, plus a note that it was reused.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .provenance import Provenance

__all__ = [
    "CacheError",
    "StaleCache",
    "VersionMismatch",
    "ResultCache",
    "default_cache_root",
    "kind_folder",
]

#: The longest path Windows opens without long-path support: MAX_PATH (260) less the terminating NUL.
WINDOWS_PATH_LIMIT = 259
#: The longest name a cache writes inside a kind's folder: a 64-character key and ``.provenance.json``.
LONGEST_ENTRY_NAME = 64 + len(".provenance.json")
#: The longest name one folder may have (NTFS, FAT32).
LONGEST_FOLDER_NAME = 255


def kind_folder(root, kind: str) -> str:
    """The folder a cache keeps ``kind``'s entries in, under ``root``: the kind with every character but letters,
    digits and ``._-`` made ``_``, AS IT ALWAYS WAS, wherever its longest entry
    (``<root>/<folder>/<key>.provenance.json``) stays inside `WINDOWS_PATH_LIMIT` and the folder inside
    `LONGEST_FOLDER_NAME`. A longer one keeps as much of its head as fits and ends in ``_`` and the first 16 hex
    characters of the SHA-256 of the whole kind: stable, so a Run finds again what the last one stored, and as
    distinct as the kinds.

    WHY (4 Nov 2026, ruled): the shared-D folder is ``cmp_global_D_`` and every condition's name, so a Run over many
    long names wrote past the limit; the store swallows the error (`warmcache.WarmCache._store`, best effort), and
    every read then fitted the joint again, with no word -- found when a scratch root of 152 characters left
    a lab project's four-condition entry unwritten and the payload loop refitted it. A root too long for even the hash
    gets the hash alone, and its entries may still not fit."""
    safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in kind)
    room = min(WINDOWS_PATH_LIMIT - len(str(root)) - 2 - LONGEST_ENTRY_NAME, LONGEST_FOLDER_NAME)
    if len(safe) <= room:
        return safe
    digest = hashlib.sha256(kind.encode("utf-8")).hexdigest()[:16]
    head = room - len(digest) - 1
    return f"{safe[:head]}_{digest}" if head > 0 else digest


class CacheError(RuntimeError):
    """Raised when the cache cannot honestly serve or store a result."""


class StaleCache(CacheError):
    """No entry matches; carries what differs from the closest stored stamp."""

    def __init__(self, message: str, *, differences=None):
        super().__init__(message)
        self.differences = differences or []


class VersionMismatch(StaleCache):
    """Everything matches except a load-bearing package version, overridable."""

    def __init__(self, message: str, *, changed_versions, stored_key: str):
        super().__init__(message, differences=[message])
        self.changed_versions = changed_versions
        self.stored_key = stored_key


def default_cache_root() -> Path:
    """Per-user cache directory (``platformdirs``), created on demand."""
    try:
        from platformdirs import user_cache_dir

        root = Path(user_cache_dir("smtracker", "smtracker"))
    except Exception:                                  # pragma: no cover - fallback
        root = Path.home() / ".smtracker_cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


@dataclass
class ResultCache:
    """A provenance-keyed store for tables and result objects."""

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    # -- paths ------------------------------------------------------------ #

    def _dir(self, kind: str) -> Path:
        # Sanitize: a kind becomes a directory name, and Windows rejects ':' and
        # friends. Keep it filesystem-safe so no caller can craft a bad path -- and
        # inside Windows' path limit (`kind_folder`, 4 Nov 2026).
        d = self.root / kind_folder(self.root, kind)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _provenance_path(self, kind: str, key: str) -> Path:
        return self._dir(kind) / f"{key}.provenance.json"

    def _index(self, kind: str) -> list[Provenance]:
        """Every stored provenance for a kind (for closest-match diagnostics)."""
        out = []
        for path in self._dir(kind).glob("*.provenance.json"):
            try:
                out.append(Provenance.from_dict(json.loads(path.read_text("utf-8"))))
            except (OSError, ValueError, KeyError):
                continue
        return out

    def has(self, kind: str, provenance: Provenance) -> bool:
        return self._provenance_path(kind, provenance.key()).is_file()

    # -- store ------------------------------------------------------------ #

    def store_table(self, provenance: Provenance, table: pd.DataFrame) -> str:
        """Persist a parsed track table (skip re-ingest next time)."""
        key = provenance.key()
        path = self._dir("table") / f"{key}.parquet"
        try:
            table.to_parquet(path)
        except Exception:                              # pragma: no cover - engine
            path = self._dir("table") / f"{key}.pkl"
            table.to_pickle(path)
        provenance.write_beside(self._dir("table") / key)
        return key

    def store_result(self, kind: str, provenance: Provenance, obj) -> str:
        """Persist any picklable result object (skip recompute next time)."""
        key = provenance.key()
        with open(self._dir(kind) / f"{key}.pkl", "wb") as handle:
            pickle.dump(obj, handle)
        provenance.write_beside(self._dir(kind) / key)
        return key

    def export_csv(self, kind: str, provenance: Provenance, frame: pd.DataFrame) -> Path:
        """A human-readable CSV carrying the provenance header, for sharing."""
        path = self._dir(kind) / f"{provenance.key()}.csv"
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(provenance.csv_header())
            frame.to_csv(handle, index=False)
        provenance.write_beside(path)
        return path

    # -- load ------------------------------------------------------------- #

    def load_table(self, provenance: Provenance, *, reuse_on_version_mismatch=False):
        key = self._resolve(
            "table", provenance, reuse_on_version_mismatch=reuse_on_version_mismatch
        )
        parquet = self._dir("table") / f"{key}.parquet"
        pkl = self._dir("table") / f"{key}.pkl"
        try:
            if parquet.is_file():
                return pd.read_parquet(parquet)
            return pd.read_pickle(pkl)
        except (OSError, ValueError) as exc:
            raise StaleCache(f"cached table is unreadable ({exc}); recompute") from exc

    def load_result(self, kind: str, provenance: Provenance, *,
                    reuse_on_version_mismatch=False):
        key = self._resolve(
            kind, provenance, reuse_on_version_mismatch=reuse_on_version_mismatch
        )
        try:
            with open(self._dir(kind) / f"{key}.pkl", "rb") as handle:
                return pickle.load(handle)
        except (OSError, pickle.UnpicklingError) as exc:
            raise StaleCache(f"cached result is unreadable ({exc}); recompute") from exc

    # -- memoization ------------------------------------------------------ #

    def get_or_compute(self, kind: str, provenance: Provenance, compute, *,
                       reuse_on_version_mismatch: bool = False):
        """Return the cached result, or compute + store it. Fit-level memoize.

        This is what makes toggling a setting back to one already computed
        instant: ``n_pop`` 3 → 2 → 3 recomputes 2 once, then the return to 3 hits
        the stored fit. ``compute`` is a zero-argument callable run only on a
        miss. A **hard** miss recomputes; a version-only mismatch is surfaced as
        ``VersionMismatch`` unless ``reuse_on_version_mismatch`` is set, exactly
        as ``load_result``: memoization must not silently ignore a version bump.
        """
        try:
            return self.load_result(
                kind, provenance,
                reuse_on_version_mismatch=reuse_on_version_mismatch,
            )
        except VersionMismatch:
            raise                                       # caller decides; never silent
        except StaleCache:
            obj = compute()
            self.store_result(kind, provenance, obj)
            return obj

    # -- the matcher ------------------------------------------------------ #

    def _resolve(self, kind: str, provenance: Provenance, *,
                 reuse_on_version_mismatch: bool) -> str:
        """The key to load, or a graded miss explaining what changed."""
        if self.has(kind, provenance):
            return provenance.key()

        want = provenance.key_material()
        version_prefix = "code.load_bearing_versions."
        version_candidate = None
        for stored in self._index(kind):
            stored_material = stored.key_material()
            diff = _diff_material(want, stored_material)
            # A version-only mismatch: every differing path is a load-bearing
            # package version, and nothing else moved.
            if diff and all(p.startswith(version_prefix) for p in diff):
                changed = _version_changes(
                    want["code"]["load_bearing_versions"],
                    stored_material["code"]["load_bearing_versions"],
                )
                version_candidate = (stored, changed)

        if version_candidate is not None:
            stored, changed = version_candidate
            changes = ", ".join(
                f"{name} {old}->{new}" for name, (old, new) in changed.items()
            )
            message = (
                f"cache miss: load-bearing package changed ({changes}): "
                "recompute, or reuse anyway (may change the number)"
            )
            if reuse_on_version_mismatch:
                return stored.key()
            raise VersionMismatch(
                message, changed_versions=changed, stored_key=stored.key()
            )

        raise StaleCache(
            f"no cached {kind} for this provenance (key {provenance.key()[:12]}); "
            "inputs, settings, seed or code changed: recompute"
        )


def _diff_material(want: dict, have: dict, prefix: str = "") -> list[str]:
    """Dotted paths where two key-material dicts differ."""
    paths = []
    for key in set(want) | set(have):
        path = f"{prefix}{key}"
        a, b = want.get(key), have.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            paths.extend(_diff_material(a, b, prefix=f"{path}."))
        elif a != b:
            paths.append(path)
    return paths


def _version_changes(want: dict, have: dict) -> dict:
    """``{package: (stored, wanted)}`` for versions that differ."""
    return {
        name: (have.get(name), want.get(name))
        for name in set(want) | set(have)
        if want.get(name) != have.get(name)
    }
