"""THE ORDER THE CONDITIONS ARE DRAWN IN, set once in the workspace and followed everywhere (3 Nov 2026). Headless: no
Qt, no numpy.

Asked: "or we can set order in the visiiblity screens, change order of how samples are arranged. not only sqd bubbles,
but everywhere possible" -- e.g. a synthetic condition left of the measured one it models, without renaming folders.
The order is a list of condition KEYS; every section's cards, rows, columns and tiles, and every figure exported from
them, follow it (`WorkspaceWindow._apply_condition_order`).

* `apply` -- the conditions in a set order: the ones it names first, in its order, then any it does not know (a
  condition added since) in the order they arrived;
* `moved` -- one condition one place earlier or later; `move` the same under a folder design's nesting (inside its
  group a condition swaps with its neighbour and every other group keeps its own order, at the group's edge the whole
  group moves one place), which is what the Conditions tool's arrows do;
* `sort_by` -- any sequence of items sorted by their condition in an order (a chart's points, a table's rows);
* `arranged` -- the order through the folder design's nesting (`factor_layout.nested`): the nesting still groups,
  and since it ranks each factor's values by their FIRST appearance (an inner factor's within its own group), the set
  order decides which group comes first and the order inside EACH group (ruled 4 Nov 2026, D14: each group its own;
  one list of conditions holds them all, so an order saved before reads as it did);
* `remember` / `recall` -- per project, in the window settings, as the condition colours and the nesting are.

DISPLAY ONLY: the order changes where a condition is drawn and nothing else. A fit is per condition (Rule 0), no cache
is keyed on the order, and changing it never refits or recomputes.
"""
from __future__ import annotations

__all__ = ["SETTINGS_KEY", "apply", "moved", "move", "sort_by", "arranged", "remember", "recall", "project_key"]

#: The window-settings key (`ui.theme.remember`) the orders live under, one per project key.
SETTINGS_KEY = "condition_order"


def project_key(roots) -> str | None:
    """The key a project's order (and its other workspace memories) is kept under: its condition folders, resolved,
    sorted, joined by "|" -- the same at the door (`ui.onboarding`, the Confirm screen's tabs) and in the workspace
    (`WorkspaceWindow._project_key`), so an order set at the door is the one the workspace opens with. None without
    roots or where a path cannot be resolved."""
    from pathlib import Path

    try:
        key = "|".join(sorted(str(Path(r).resolve()) for r in roots or () if str(r)))
    except (TypeError, OSError):
        return None
    return key or None


def apply(keys, order) -> list:
    """``keys`` in ``order``: those ``order`` names first, in its order, then the rest in the order they came."""
    keys = list(keys)
    rank = {str(k): i for i, k in enumerate(order or ())}
    known = sorted((k for k in keys if str(k) in rank), key=lambda k: rank[str(k)])
    return known + [k for k in keys if str(k) not in rank]


def moved(order, key, step: int) -> list:
    """``order`` with ``key`` moved ``step`` places (negative: earlier), held at the ends; unchanged for an unknown
    key."""
    order = list(order)
    if key not in order:
        return order
    i = order.index(key)
    j = max(0, min(len(order) - 1, i + int(step)))
    order.insert(j, order.pop(i))
    return order


def move(keys, key, step: int, factors=None, nesting=()) -> list:
    """The order after an arrow: ``key`` ``step`` places, then `arranged` through the nesting. With a folder design
    (two or more factors in ``nesting``) the groups stay together and EACH GROUP KEEPS ITS OWN ORDER (ruled 4 Nov
    2026, tester D14: an arrow on one condition also moved a sibling's, one inner order shared by every group): a
    condition swaps with its neighbour inside its group, the conditions of every other group staying where they
    are; at the group's edge the WHOLE GROUP moves one place among the groups beside it (a move the nesting would
    otherwise sort straight back), and at the edge of those, the group above it, and so on. Unchanged at the ends."""
    keys = list(keys)
    nesting = tuple(nesting or ())
    if key not in keys:
        return keys
    if not factors or len(nesting) < 2:
        return moved(keys, key, step)
    keys = arranged(keys, None, factors, nesting)

    def prefix(k, depth):
        """The group ``k`` sits in at ``depth``: its values of the first ``depth`` factors."""
        return tuple((factors.get(k) or {}).get(name) for name in nesting[:depth])

    deepest = len(nesting) - 1
    i = keys.index(key)
    j = i + int(step)
    if 0 <= j < len(keys) and prefix(keys[j], deepest) == prefix(key, deepest):
        return arranged(moved(keys, key, step), None, factors, nesting)
    direction = 1 if step > 0 else -1
    for depth in range(deepest, 0, -1):
        mine = prefix(key, depth)
        parent = mine[:-1]

        def sibling(k, depth=depth, parent=parent):
            return prefix(k, depth)[:-1] == parent

        blocks = list(dict.fromkeys(prefix(k, depth) for k in keys if sibling(k)))
        b = blocks.index(mine)
        c = b + direction
        if not 0 <= c < len(blocks):
            continue                          # at the edge of the group above too: move that one
        blocks[b], blocks[c] = blocks[c], blocks[b]
        rank = {block: n for n, block in enumerate(blocks)}
        members = iter(sorted((k for k in keys if sibling(k)), key=lambda k, d=depth: rank[prefix(k, d)]))
        return arranged([next(members) if sibling(k) else k for k in keys], None, factors, nesting)
    return keys


def sort_by(items, order, condition=lambda item: item) -> list:
    """``items`` sorted (stably) by their condition's place in ``order``; an item whose condition ``order`` does not
    name keeps its place after the rest."""
    items = list(items)
    rank = {str(k): i for i, k in enumerate(order or ())}
    far = len(rank)
    return sorted(items, key=lambda item: rank.get(str(condition(item)), far))


def arranged(keys, order, factors=None, nesting=()) -> list:
    """``keys`` in the set ``order`` and then through the folder design's nesting (`factor_layout.nested`), which keeps
    the groups together; without two factors, the set order alone."""
    from .factor_layout import nested

    ordered = apply(keys, order)
    return nested(ordered, factors or {}, nesting) if factors and len(tuple(nesting or ())) >= 2 else ordered


def remember(project_key, order) -> None:
    """Keep ``order`` for the project (`condition_colours.project_key`, the window's `_project_key`) in the window
    settings. A failed write is warned about, never fatal: the order is a convenience."""
    import warnings

    from .ui import theme

    if not project_key:
        return
    stored = theme.remembered(SETTINGS_KEY)
    stored = dict(stored) if isinstance(stored, dict) else {}
    stored[str(project_key)] = [str(k) for k in order]
    try:
        theme.remember(SETTINGS_KEY, stored)
    except OSError as exc:
        warnings.warn(f"could not remember the condition order: {exc}", stacklevel=2)


def recall(project_key, keys) -> list:
    """``keys`` in the order remembered for the project (`apply`: a condition added since keeps its place after the
    rest), else as they came."""
    from .ui import theme

    keys = list(keys)
    if not project_key:
        return keys
    stored = theme.remembered(SETTINGS_KEY)
    stored = (stored if isinstance(stored, dict) else {}).get(str(project_key))
    return apply(keys, stored) if stored else keys
