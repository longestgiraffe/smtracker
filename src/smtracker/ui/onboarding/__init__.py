"""The onboarding flow: everything the application shows before a project is open.

A flow rather than an analysis, which is why it is a package of its own and the
nine analysis tabs are not. `smtracker.analyses` is a REGISTRY — the tabs are
data, and the panel builds its controls from them — so a package per tab would
hand-build what that registry generates and rot the first time a tab is renamed.
These screens are the opposite: they own their own state, their own order, and a
decision (which stage, which folder) that no catalogue entry describes.

`stages` is the first window. Nothing here computes; the screens return a choice
and `app.main` acts on it, so the routing stays testable without Qt.
"""
from __future__ import annotations

__all__ = ["confirm", "home", "scan", "stages", "window"]
