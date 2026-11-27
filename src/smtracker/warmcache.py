"""Committed warm cache: precomputed view results so ``--browse`` is instant.

The biggest fast-UI-iteration win (layer 1): compute the standard sample's view
results ONCE, commit them to the repo, and serve them on launch so the browser
opens with zero fitting. A miss (a different project, or a genuine change) falls
through to a normal compute.

Why a separate cache, keyed differently
---------------------------------------
The per-user ``ResultCache`` keys on ``provenance.key()``, which includes the git
commit: correct for a live cache (a code change should miss), but fatal for a
*committed* cache: the very next commit would invalidate every entry. So the warm
cache keys on ``provenance.content_key()`` instead, which drops the commit and
keeps data + settings + seeds + load-bearing package versions. Identical inputs
therefore hit across ordinary commits.

Honesty
-------
- Every entry stores its FULL provenance stamp beside it (recording the real
  commit it was computed at), so a served result is never anonymous.
- A structurally-incompatible entry (a changed result class → unpickling fails)
  is treated as a miss and recomputed: the warm cache can never return a broken
  object. The residual gap (a value-only fit change under an unchanged class) is
  covered by ``--refresh`` and by regenerating the bundle in the same commit.
- The data fingerprint is in the content key, so pointing the warm cache at a
  DIFFERENT project simply misses everywhere: it can only ever hit its own data.
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

from .cache import kind_folder
from .provenance import Provenance

__all__ = ["WarmCache", "warm_root_for", "default_warm_root"]

#: The warm cache lives IN the project folder under this name, so it travels with
#: the data, carry the SSD to another machine and the precompute comes along.
WARM_DIRNAME = ".smtracker_warm"


def warm_root_for(project_root) -> Path:
    """Where a project's warm cache lives: ``<project_root>/.smtracker_warm``.

    Keeping it beside the data (not in the repo, not in a per-user dir) means the
    precompute is portable: move or copy the project folder and every view stays
    a cache hit. The dir is hidden and self-contained; a project with no warm
    cache simply computes on first open and populates it.
    """
    return Path(project_root) / WARM_DIRNAME


#: Where the per-user store lives instead, when set: every test gets its own (`tests/conftest.py`), so no test reads
#: a result another test, or another version of the code, kept in the user's real store.
CACHE_DIR_ENV = "SMTRACKER_CACHE_DIR"


def default_warm_root() -> Path:
    """A per-user fallback for when no project root is available (e.g. a bare
    ``ComparisonBrowser`` in a test). Real launches use ``warm_root_for`` so the
    cache sits beside the data. ``SMTRACKER_CACHE_DIR`` set: under it instead."""
    override = os.environ.get(CACHE_DIR_ENV)
    if override:
        return Path(override) / "warm"
    try:
        from platformdirs import user_cache_dir

        return Path(user_cache_dir("smtracker", "smtracker")) / "warm"
    except Exception:                                  # pragma: no cover - fallback
        return Path.home() / ".smtracker_cache" / "warm"


@dataclass
class WarmCache:
    """A content-keyed store with the same ``get_or_compute`` shape ResultCache
    has, so the comparison controllers use either interchangeably.

    ``refresh=True`` (``--refresh``) forces every ``get_or_compute`` to recompute
    and overwrite: the way to rebuild the bundle after the data or the fit logic
    changes.

    ``condition_roots`` (condition → its own folder) routes PER-CONDITION results
    (a fit, a bootstrap, a saSPT run) into ``<condition_folder>/.smtracker_warm``
    via ``for_condition``, so those results travel with the condition folder,
    drop the folder into another experiment and its cache comes along. The
    cross-condition view assembly (which spans the selected conditions) still
    lives at the project ``root``.
    """

    root: Path
    refresh: bool = False
    condition_roots: dict | None = None

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def for_condition(self, condition: str) -> "WarmCache":
        """A cache rooted in ``condition``'s OWN folder, so its per-condition
        results are stored beside that condition's data. Falls back to the main
        root when the condition's folder is unknown."""
        if self.condition_roots and condition in self.condition_roots:
            return WarmCache(warm_root_for(self.condition_roots[condition]),
                             refresh=self.refresh)
        return self

    def _dir(self, kind: str) -> Path:
        # filesystem-safe and inside Windows' path limit, a long kind shortened with a stable hash (`kind_folder`)
        d = self.root / kind_folder(self.root, kind)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _path(self, kind: str, provenance: Provenance) -> Path:
        return self._dir(kind) / f"{provenance.content_key()}.pkl"

    def has(self, kind: str, provenance: Provenance) -> bool:
        return self._path(kind, provenance).is_file()

    def get_or_compute(self, kind: str, provenance: Provenance, compute):
        """Return the warm result, or compute + store it. Same contract as
        ``ResultCache.get_or_compute`` so views need not know which cache they got.
        """
        path = self._path(kind, provenance)
        if not self.refresh and path.is_file():
            try:
                with open(path, "rb") as handle:
                    return pickle.load(handle)
            except (OSError, pickle.UnpicklingError, AttributeError, ModuleNotFoundError):
                # Structurally incompatible / unreadable → treat as a miss and
                # recompute, never return a broken object.
                pass
        obj = compute()
        self._store(kind, provenance, obj, path)
        return obj

    def _store(self, kind, provenance, obj, path) -> None:
        try:
            with open(path, "wb") as handle:
                pickle.dump(obj, handle)
            # the FULL stamp beside it (records the real commit), honesty
            stamp = path.with_suffix(".provenance.json")
            stamp.write_text(json.dumps(provenance.to_dict(), indent=2, default=str),
                             encoding="utf-8")
        except (OSError, pickle.PicklingError):            # pragma: no cover - best effort
            pass

    def clear(self) -> int:
        """Delete every stored entry (a hard reset). Returns how many were removed."""
        removed = 0
        for path in self.root.rglob("*.pkl"):
            try:
                path.unlink()
                removed += 1
            except OSError:                                # pragma: no cover
                pass
        return removed
