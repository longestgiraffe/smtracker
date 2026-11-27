"""Drill-down navigation, the functional gesture, wired end to end.

Click a condition → its replicate view → a cell → its track list → a track → the
inspector. This is the navigation *state machine* behind that gesture: a
breadcrumb stack with back, and a ``drill`` resolver that, given the current
level and what was clicked, returns the descriptor of the next view. The Qt
click-handlers are a thin layer over this, so the *gesture* is tested here, and
smoothness/feel is left for later.

The terminal level is always the single-track inspector / viewer. Each descriptor
names the view and the filter that produces it (a condition, a cell, a uid), so a
Qt shell can build the panel and a headless test can assert the path.

Reuse of cached fits (items 2/4) belongs to the *views*, not this state machine,
the resolver only decides where a click goes, so it stays pure and instant.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["NavigationError", "View", "Breadcrumb", "drill", "LEVELS"]

#: The drill order. Each click descends one level; the last is terminal.
LEVELS = ["conditions", "replicates", "cells", "tracks", "inspector"]


class NavigationError(RuntimeError):
    """Raised on an illegal navigation."""


@dataclass(frozen=True)
class View:
    """A view to render: its level, the filter that produced it, and a label."""

    level: str
    selection: dict            # e.g. {"condition": "malate"} or {"uid": "…"}
    label: str

    @property
    def is_terminal(self) -> bool:
        return self.level == LEVELS[-1]


def drill(current: View, clicked: dict) -> View:
    """Resolve a click on ``current`` into the next view down.

    ``clicked`` names what was selected (``{"condition": …}``, ``{"cell": …}``,
    ``{"uid": …}``). The selection accumulates, so the terminal view carries the
    full path (condition → cell → uid). Clicking past the terminal level raises.
    """
    if current.is_terminal:
        raise NavigationError("already at the track inspector: nothing deeper")
    index = LEVELS.index(current.level)
    next_level = LEVELS[index + 1]
    selection = {**current.selection, **clicked}

    label_bits = []
    for key in ("condition", "day", "cell", "uid"):
        if key in selection:
            label_bits.append(f"{key}={selection[key]}")
    label = f"{next_level}: " + ", ".join(label_bits) if label_bits else next_level
    return View(level=next_level, selection=selection, label=label)


@dataclass
class Breadcrumb:
    """The navigation stack: push on drill-down, pop on back."""

    stack: list = field(default_factory=lambda: [
        View(level="conditions", selection={}, label="all conditions")
    ])

    @property
    def current(self) -> View:
        return self.stack[-1]

    @property
    def can_go_back(self) -> bool:
        return len(self.stack) > 1

    def drill_to(self, clicked: dict) -> View:
        """Click something in the current view; descend and record the crumb."""
        nxt = drill(self.current, clicked)
        self.stack.append(nxt)
        return nxt

    def back(self) -> View:
        """Pop to the previous view. Raises at the root."""
        if not self.can_go_back:
            raise NavigationError("already at the top level")
        self.stack.pop()
        return self.current

    def to_root(self) -> View:
        """Jump back to the top."""
        self.stack = self.stack[:1]
        return self.current

    def trail(self) -> list:
        """The breadcrumb labels, root → current."""
        return [v.label for v in self.stack]
