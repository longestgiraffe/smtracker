"""Diff engine: what changed between two runs, and how each number moved.

Two questions this answers, both from the provenance stamps that every result now
carries (items 0/2):

1. **What differs between two runs?** ``compare_provenance`` reports the changed
   settings, seeds, code/version, and the exact units added / removed / changed,
   the "why is this number different" answer, before you even look at the number.
2. **How did the numbers move?** ``compare_numbers`` pairs the reported
   quantities and gives Δ (and, where a bootstrap CI is present, whether the
   change clears it: a shift inside both CIs is not yet a real difference).

``compare_conditions`` is the same machinery pointed sideways: given one result
holding several conditions, tabulate a chosen quantity across two of them with
their intervals.

Headless and testable; the visual diff view is deliberately left for later: this
is the numeric core it will sit on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .provenance import Provenance

__all__ = [
    "DiffError",
    "ProvenanceDiff",
    "compare_provenance",
    "compare_numbers",
    "compare_conditions",
]


class DiffError(RuntimeError):
    """Raised when two results cannot be compared."""


@dataclass(frozen=True)
class ProvenanceDiff:
    """What changed between two provenance stamps."""

    settings_changed: dict = field(default_factory=dict)   # key -> (a, b)
    seeds_changed: dict = field(default_factory=dict)
    code_changed: dict = field(default_factory=dict)        # commit, versions
    version_changed: dict = field(default_factory=dict)     # package -> (a, b)
    units_added: list = field(default_factory=list)
    units_removed: list = field(default_factory=list)
    units_changed: list = field(default_factory=list)       # same id, hash moved
    pixel_or_tau_changed: dict = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not any([
            self.settings_changed, self.seeds_changed, self.code_changed,
            self.version_changed, self.units_added, self.units_removed,
            self.units_changed, self.pixel_or_tau_changed,
        ])

    def summary(self) -> str:
        if self.is_empty:
            return "identical provenance: same result expected"
        parts = []
        if self.settings_changed:
            parts.append("settings: " + ", ".join(
                f"{k} {a!r}->{b!r}" for k, (a, b) in self.settings_changed.items()))
        if self.seeds_changed:
            parts.append("seeds: " + ", ".join(
                f"{k} {a}->{b}" for k, (a, b) in self.seeds_changed.items()))
        if self.version_changed:
            parts.append("versions: " + ", ".join(
                f"{k} {a}->{b}" for k, (a, b) in self.version_changed.items()))
        if self.code_changed.get("commit"):
            a, b = self.code_changed["commit"]
            parts.append(f"commit {str(a)[:8]}->{str(b)[:8]}")
        if self.units_added:
            parts.append(f"{len(self.units_added)} unit(s) added")
        if self.units_removed:
            parts.append(f"{len(self.units_removed)} unit(s) removed")
        if self.units_changed:
            parts.append(f"{len(self.units_changed)} unit(s) data changed")
        if self.pixel_or_tau_changed:
            parts.append("ACQUISITION CHANGED: " + ", ".join(
                f"{k} {a}->{b}" for k, (a, b) in self.pixel_or_tau_changed.items()))
        return " | ".join(parts)


def _dict_delta(a: dict, b: dict) -> dict:
    a, b = a or {}, b or {}
    return {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}


def compare_provenance(a: Provenance, b: Provenance) -> ProvenanceDiff:
    """Structured diff of two provenance stamps: the 'what changed' report."""
    if not isinstance(a, Provenance) or not isinstance(b, Provenance):
        raise DiffError("compare_provenance needs two Provenance objects")

    code_a, code_b = a.code, b.code
    version_changed = _dict_delta(
        code_a.get("load_bearing_versions", {}),
        code_b.get("load_bearing_versions", {}),
    )
    code_changed = {}
    if code_a.get("commit") != code_b.get("commit"):
        code_changed["commit"] = (code_a.get("commit"), code_b.get("commit"))

    units_a = a.inputs.get("units", {})
    units_b = b.inputs.get("units", {})
    units_added = sorted(set(units_b) - set(units_a))
    units_removed = sorted(set(units_a) - set(units_b))
    units_changed = sorted(
        p for p in set(units_a) & set(units_b) if units_a[p] != units_b[p]
    )

    acquisition = {}
    for field_name in ("pixel_size_nm", "frame_interval_ms"):
        va, vb = a.inputs.get(field_name), b.inputs.get(field_name)
        if va != vb:
            acquisition[field_name] = (va, vb)

    return ProvenanceDiff(
        settings_changed=_dict_delta(a.settings, b.settings),
        seeds_changed=_dict_delta(a.seeds, b.seeds),
        code_changed=code_changed,
        version_changed=version_changed,
        units_added=units_added,
        units_removed=units_removed,
        units_changed=units_changed,
        pixel_or_tau_changed=acquisition,
    )


def _as_number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compare_numbers(a: dict, b: dict, *, keys=None, ci: dict | None = None):
    """Pair reported quantities and report Δ, with CI-clearance where available.

    ``a`` and ``b`` are flat ``{name: value}`` dicts. ``ci`` optionally maps a
    name to ``{"a": (lo, hi), "b": (lo, hi)}``; when present, ``clears_ci`` says
    whether ``b`` falls outside ``a``'s interval AND vice versa, a shift that
    sits inside both CIs is not yet a real difference. Returns a list of row
    dicts (kept as plain data so any caller can tabulate it).
    """
    shared = keys if keys is not None else sorted(set(a) & set(b))
    rows = []
    for name in shared:
        va, vb = _as_number(a.get(name)), _as_number(b.get(name))
        if va is None or vb is None:
            continue
        row = {"quantity": name, "a": va, "b": vb, "delta": vb - va,
               "pct_change": (100 * (vb - va) / va) if va else None}
        if ci and name in ci:
            ca, cb = ci[name].get("a"), ci[name].get("b")
            outside_a = ca is not None and not (ca[0] <= vb <= ca[1])
            outside_b = cb is not None and not (cb[0] <= va <= cb[1])
            row["clears_ci"] = bool(outside_a and outside_b)
            row["ci_a"] = ca
            row["ci_b"] = cb
        rows.append(row)
    if not rows:
        raise DiffError("no shared numeric quantity to compare")
    return rows


def compare_conditions(per_condition: dict, cond_a: str, cond_b: str, *,
                       quantity: str, ci_key: str | None = None):
    """Tabulate one quantity across two conditions of a single result.

    ``per_condition`` maps a condition name to its numbers dict. Returns the two
    values, their difference, and (if ``ci_key`` names a ``[lo, hi]`` field in
    each) whether the difference clears both intervals.
    """
    for cond in (cond_a, cond_b):
        if cond not in per_condition:
            raise DiffError(f"no condition {cond!r} in the result")
    a, b = per_condition[cond_a], per_condition[cond_b]
    if quantity not in a or quantity not in b:
        raise DiffError(f"quantity {quantity!r} missing from a condition")
    va, vb = _as_number(a[quantity]), _as_number(b[quantity])
    out = {"quantity": quantity, cond_a: va, cond_b: vb, "difference": vb - va}
    if ci_key and ci_key in a and ci_key in b:
        ca, cb = a[ci_key], b[ci_key]
        out["ci_a"] = ca
        out["ci_b"] = cb
        # Disjoint intervals => the difference is real at this level.
        out["intervals_disjoint"] = bool(ca[1] < cb[0] or cb[1] < ca[0])
    return out
