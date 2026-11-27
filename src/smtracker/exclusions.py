"""Excluding movies, cells or tracks: explicitly, reversibly, and on the record.

Why this is a module and not a filter argument
----------------------------------------------
QC flags a unit; a human decides whether it is an artifact. Both halves matter,
and the decision is a *scientific* one, so it must be:

- **explicit**: nothing is ever dropped because a rule said so
  (``smtracker.qc`` flags, this excludes, and only a caller connects them);
- **recorded**, every exclusion carries a written reason, and applying a set
  records exactly what it removed, so a result can always be traced back to the
  data it was computed from;
- **reversible**: the set is a small JSON file, so "what does the number look
  like without series004?" and "…with it back?" are both one run.

That is the same discipline as ``NPopSelection.choose(n, reason=...)``: the tool
does not stop you overriding it, it stops you overriding it *silently*.

Granularity
-----------
``movie``, ``cell`` and ``uid`` (a single track). Cells are named
``<movie>:<cell_id>``, so excluding a movie also removes its cells: that is
handled explicitly rather than relying on string prefixes.

Invariants
----------
- Every exclusion needs a non-empty reason. An unexplained exclusion is exactly
  the thing that becomes unreproducible three months later.
- Applying a set that would remove **everything** fails loud.
- The result records ``n_tracks_removed`` / ``n_localizations_removed`` and the
  full set, on ``.attrs``.
- Pure pandas. No Qt, no plotting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

__all__ = [
    "ExclusionError",
    "KINDS",
    "Exclusion",
    "ExclusionSet",
    "apply_exclusions",
    "proposed_from_flags",
]

#: What can be excluded. Ordered coarse → fine.
KINDS = ("movie", "cell", "uid")


class ExclusionError(RuntimeError):
    """Raised when an exclusion is malformed or would remove everything."""


@dataclass(frozen=True)
class Exclusion:
    """One excluded unit, with the reason it was excluded."""

    kind: str
    key: str
    reason: str

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ExclusionError(f"kind must be one of {KINDS}, got {self.kind!r}")
        if not str(self.key).strip():
            raise ExclusionError("an exclusion needs a key")
        if not str(self.reason).strip():
            raise ExclusionError(
                f"excluding {self.kind} {self.key!r} needs a written reason: an "
                "unexplained exclusion is unreproducible later"
            )

    def to_dict(self) -> dict:
        return {"kind": self.kind, "key": self.key, "reason": self.reason}

    @classmethod
    def parse(cls, text: str, *, reason: str = "") -> "Exclusion":
        """Parse ``kind:key`` (as given on a command line)."""
        kind, separator, key = str(text).partition(":")
        if not separator:
            raise ExclusionError(
                f"expected '<kind>:<key>', got {text!r} (kinds: {', '.join(KINDS)})"
            )
        # A cell key is itself 'movie:cell', so the split must keep the rest.
        return cls(kind=kind.strip(), key=key.strip(), reason=reason)


@dataclass
class ExclusionSet:
    """A reviewable, saveable list of exclusions."""

    items: list[Exclusion] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def add(self, kind: str, key: str, reason: str) -> "ExclusionSet":
        self.items.append(Exclusion(kind=kind, key=key, reason=reason))
        return self

    def keys(self, kind: str) -> set[str]:
        return {item.key for item in self.items if item.kind == kind}

    def summary(self) -> str:
        if not self.items:
            return "no exclusions"
        return "; ".join(f"{i.kind} {i.key} ({i.reason})" for i in self.items)

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(
            json.dumps([i.to_dict() for i in self.items], indent=2), encoding="utf-8"
        )
        return path

    @classmethod
    def from_json(cls, path: str | Path) -> "ExclusionSet":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ExclusionError(f"{path} does not hold a list of exclusions")
        return cls([Exclusion(**entry) for entry in data])


def _conditions_touched(canonical: pd.DataFrame, exclusions: ExclusionSet) -> list:
    """Per item of ``exclusions``, in order, ``{condition: tracks it removes there}`` on ``canonical`` (empty where the
    table has no condition column or the item matches nothing)."""
    out = []
    for item in exclusions:
        if "condition" not in canonical.columns or item.kind not in canonical.columns:
            out.append({})
            continue
        hit = canonical.loc[canonical[item.kind].astype(str) == str(item.key), ["condition", "uid"]]
        out.append({str(c): int(g["uid"].nunique()) for c, g in hit.groupby("condition", observed=True)})
    return out


def apply_exclusions(canonical: pd.DataFrame, exclusions: ExclusionSet) -> pd.DataFrame:
    """Remove the excluded units, recording exactly what went.

    The returned table carries ``attrs["exclusions"]`` (the set, as dicts) and
    the counts removed, so any downstream result can be traced back to the data
    it was actually computed from.
    """
    if not isinstance(exclusions, ExclusionSet):
        raise ExclusionError("expected an ExclusionSet")
    for column in ("movie", "cell", "uid"):
        if column not in canonical.columns:
            raise ExclusionError(f"table has no {column!r} column")

    keep = pd.Series(True, index=canonical.index)
    for kind in KINDS:
        keys = exclusions.keys(kind)
        if keys:
            keep &= ~canonical[kind].astype(str).isin(keys)

    out = canonical[keep].copy()
    n_before = canonical["uid"].nunique()
    n_after = out["uid"].nunique() if len(out) else 0
    if len(exclusions) and n_after == 0:
        raise ExclusionError(
            f"these exclusions remove every track ({n_before} → 0). "
            f"Set: {exclusions.summary()}"
        )

    out.attrs.update(canonical.attrs)
    out.attrs["exclusions"] = [i.to_dict() for i in exclusions]
    # WHERE EACH ONE TOOK DATA FROM (4 Nov 2026): per item, the tracks it removed in each condition, read on the
    # table BEFORE removal (afterwards the unit is gone and nothing says whose it was). QC's "Contributes nothing"
    # lists an exclusion under these conditions only; it once listed every exclusion under every condition.
    out.attrs["exclusion_conditions"] = _conditions_touched(canonical, exclusions)
    out.attrs["n_tracks_before"] = int(n_before)
    out.attrs["n_tracks_removed"] = int(n_before - n_after)
    out.attrs["n_localizations_before"] = int(len(canonical))
    out.attrs["n_localizations_removed"] = int(len(canonical) - len(out))
    out.attrs["exclusion_summary"] = (
        f"{n_before - n_after} of {n_before} tracks and "
        f"{len(canonical) - len(out)} of {len(canonical)} localizations removed | "
        f"{exclusions.summary()}"
    )
    return out


def proposed_from_flags(
    units: pd.DataFrame, *, reason: str | None = None
) -> ExclusionSet:
    """Turn QC flags into a **proposal**, never applied automatically.

    ``units`` is a ``qc.unit_summary`` table. The returned set is something to
    read, edit and decide on: a flag means "look at this", and whether it is an
    artifact or real biology is not a question a threshold can answer.
    """
    if "flagged" not in units.columns:
        raise ExclusionError("not a unit_summary table (no 'flagged' column)")
    unit = units.attrs.get("unit")
    if unit not in KINDS:
        raise ExclusionError(f"unit_summary has no usable unit (got {unit!r})")

    flagged_on = units.attrs.get("flagged_on", "a QC readout")
    proposal = ExclusionSet()
    for _, row in units[units["flagged"]].iterrows():
        value = row.get(flagged_on)
        detail = (
            f"{flagged_on} = {value:.4g} vs median "
            f"{units.attrs.get('median', float('nan')):.4g}"
            if isinstance(value, float) else f"flagged on {flagged_on}"
        )
        proposal.add(
            unit, str(row[unit]),
            reason or f"QC outlier: {detail} (proposed: review before applying)",
        )
    return proposal
