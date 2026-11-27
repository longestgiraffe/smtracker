"""Conditions grouped by their factors: the nesting order, the column order and the grouped column positions every
comparison chart draws (phases 3 and 4 of the folder designs, 24 Oct 2026). Headless: no Qt, no numpy.

Asked: "grouping in the sqd panel etc could be possible. For example protein A - glucose, malate / protein B - glucose, malate /
Or the other way, glucose - protein A, protein B". A project with a folder design (`folderdesign`) carries each condition's factor
values (`Project.factors`); the NESTING is an order of the factor names -- ``("protein", "medium")`` puts each
protein's media side by side under the protein, ``("medium", "protein")`` each medium's proteins under the medium.

* `orders` -- the nestings offered (every order of up to three factors), `order_words` their names;
* `nested` -- the conditions in nesting order: by the outer factor, then the next, each factor's values in the order
  they first appear among the conditions as given (so a folder's own order is kept, never alphabetised away), the
  outer factor's among all of them and every inner one's WITHIN ITS GROUP: each group keeps its own order (ruled
  4 Nov 2026, tester D14: "protein B glucose" moved up also moved "protein B malate" before "protein A malate", one inner order shared
  by every group);
* `columns` -- each condition's x position with a GAP between outer groups, the groups' spans and headers, and each
  condition's label under its column (the inner values; the outer one is the header). Without two factors it is the
  plain layout (x = index, no groups), so a chart that always asks for it draws exactly as before;
* `remember` / `recall` -- the nesting chosen, per project, in the window settings (as the condition colours).

THE ORDER CHANGES NO NUMBER: a fit is per condition, and Rule 0 makes conditions independent whatever stands beside
them. Only where a condition is drawn moves.
"""
from __future__ import annotations

from itertools import permutations

__all__ = ["GAP", "orders", "order_words", "nested", "columns", "remember", "recall"]

#: The space between two outer groups, in columns.
GAP = 0.6
#: The settings key the nesting is remembered under (`ui.theme.remember`).
_SETTINGS_KEY = "factor_nesting"


def orders(names) -> list:
    """Every nesting of ``names`` (the design's factor names), the design's own order first; [] for fewer than two."""
    names = tuple(names)
    if len(names) < 2:
        return []
    return [names] + [p for p in permutations(names) if p != names]


def order_words(order) -> str:
    """``"protein → medium"``: outer first."""
    return " → ".join(order)


def _prefix(factors, condition, order, depth) -> tuple:
    """``condition``'s values of the first ``depth`` factors of ``order``: the group it sits in at that depth."""
    return tuple(factors[condition][name] for name in order[:depth])


def nested(conditions, factors, order) -> list:
    """``conditions`` in nesting order (module docstring); unchanged without a usable order. A condition without
    factors keeps its place after those that have them.

    EACH GROUP ITS OWN ORDER (ruled 4 Nov 2026, D14). A factor's value is ranked by where it first appears among the
    conditions of ITS GROUP (those sharing every value above it), not among all conditions: under medium -> protein,
    "protein B glucose" before "protein A glucose" puts protein B first in glucose only, and malate keeps its own order. The outer
    factor's group is all of them, as before. A list already nested (every group in one inner order) is unchanged by
    this, so an order saved before reads as it was drawn."""
    conditions = list(conditions)
    order = tuple(order or ())
    if len(order) < 2 or not factors:
        return conditions
    with_factors = [c for c in conditions if all(name in (factors.get(c) or {}) for name in order)]
    rest = [c for c in conditions if c not in with_factors]
    first: dict = {}                       # (depth, the group above, this value) -> where it first appears
    for position, c in enumerate(with_factors):
        for depth, name in enumerate(order):
            first.setdefault((depth, _prefix(factors, c, order, depth), factors[c][name]), position)
    key = {c: tuple(first[(depth, _prefix(factors, c, order, depth), factors[c][name])]
                    for depth, name in enumerate(order)) for c in with_factors}
    return sorted(with_factors, key=lambda c: key[c]) + rest


def columns(conditions, factors=None, order=None, *, gap: float = GAP) -> dict:
    """Where each of ``conditions`` (already in drawing order) stands: ``{"x": [x per condition], "span": (lo, hi),
    "groups": [{"label", "first", "last", "x0", "x1"}], "labels": [label per condition], "outer": factor name or
    ""}``. With two or more factors in ``order`` the conditions sharing an outer value form a group (``first`` /
    ``last`` are indices into ``conditions``) with ``gap`` columns between groups, and each label is the inner values;
    otherwise x is the index and the labels are the conditions themselves."""
    conditions = list(conditions)
    order = tuple(order or ())
    factors = factors or {}
    plain = {"x": [float(i) for i in range(len(conditions))], "span": (-0.5, len(conditions) - 0.5),
             "groups": [], "labels": list(conditions), "outer": ""}
    if len(order) < 2 or not conditions or not all(
            all(name in (factors.get(c) or {}) for name in order) for c in conditions):
        return plain
    outer = order[0]
    xs, groups, labels = [], [], []
    x = 0.0
    previous = object()
    for i, c in enumerate(conditions):
        value = factors[c][outer]
        if value != previous:
            if groups:
                x += gap
            groups.append({"label": str(value), "first": i, "last": i, "x0": x, "x1": x})
            previous = value
        groups[-1]["last"] = i
        groups[-1]["x1"] = x
        xs.append(x)
        labels.append(" ".join(str(factors[c][name]) for name in order[1:]))
        x += 1.0
    # a value that comes back after another (an order the caller did not nest) is not one group: say plainly
    seen = [g["label"] for g in groups]
    if len(seen) != len(set(seen)):
        return plain
    return {"x": xs, "span": (-0.5, xs[-1] + 0.5), "groups": groups, "labels": labels, "outer": outer}


def remember(project_key, order) -> None:
    """Keep the nesting chosen for the project (`condition_colours.project_key`) in the window settings, as the
    condition colours are kept. A failed write is warned about, never fatal."""
    import warnings

    from .ui import theme

    if not project_key:
        return
    stored = theme.remembered(_SETTINGS_KEY)
    stored = dict(stored) if isinstance(stored, dict) else {}
    stored[str(project_key)] = list(order)
    try:
        theme.remember(_SETTINGS_KEY, stored)
    except OSError as exc:
        warnings.warn(f"could not remember the nesting: {exc}", stacklevel=2)


def recall(project_key, names) -> tuple:
    """The nesting remembered for the project when it is still an order of ``names``, else ``names`` itself."""
    from .ui import theme

    names = tuple(names)
    if not project_key:
        return names
    stored = theme.remembered(_SETTINGS_KEY)
    stored = (stored if isinstance(stored, dict) else {}).get(str(project_key))
    if stored and sorted(stored) == sorted(names) and len(stored) == len(names):
        return tuple(stored)
    return names
