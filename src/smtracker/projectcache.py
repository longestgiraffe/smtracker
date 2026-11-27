"""A cache keyed by a PROJECT OBJECT, safe against ``id()`` reuse.

Two caches here were keyed on ``id(project)`` (the Images section's per-condition reads and
the track viewer's). CPython reuses an id as soon as the object it named is freed, so a NEW
project could be served an OLD one's entry: found when a test's fresh project read
another's frame record, and the same mechanism would hand a project rebuilt with a
track-length cap the counts of the uncapped one it replaced. Every entry now remembers its
project (weakly, where the object allows it) and is served only to that very object.

Headless and Qt-free.
"""
from __future__ import annotations

import weakref

__all__ = ["ProjectCache"]

_MISS = object()


class ProjectCache:
    """``get`` / ``put`` / ``forget`` by ``(project, key)``; an entry made for one project
    object is never served to another, whatever their ids."""

    def __init__(self) -> None:
        self._store: dict = {}

    @staticmethod
    def _ref(project):
        try:
            return weakref.ref(project)
        except TypeError:                     # an object without weakref support: hold it
            return lambda: project

    def get(self, project, key, default=None):
        hit = self._store.get((id(project), key), _MISS)
        if hit is _MISS or hit[0]() is not project:
            return default
        return hit[1]

    def __contains__(self, item) -> bool:
        project, key = item
        return self.get(project, key, _MISS) is not _MISS

    def put(self, project, key, value):
        self._store[(id(project), key)] = (self._ref(project), value)
        return value

    def forget(self, project=None) -> None:
        """Drop every entry, or one project's."""
        if project is None:
            self._store.clear()
            return
        for k in [k for k in self._store if k[0] == id(project)]:
            del self._store[k]

    def clear(self) -> None:
        self._store.clear()
