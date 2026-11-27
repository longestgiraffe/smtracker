"""Which colour each condition is drawn in: ONE model for Confirm Project and the workspace.

**Why this exists.** Confirm Project and the workspace once each computed condition
colours their own way, and they disagreed on the same palette: Confirm Project read the
palette's raw hex values in declared order, the workspace read `hcl.series(n)` (fitted to
the page and SORTED BY LIGHTNESS). A choice made in Confirm Project was then thrown away at
the hand-over, so picking Tol muted there changed nothing in the workspace. Both windows
now read a condition's colour only through :class:`ConditionColours`.

**A condition holds a SLOT, never a hex** (see `hcl.slots`). The colour is
``hcl.slots(palette, dark=)[slot]``: fitted to the page, in the palette's declared order,
so a condition keeps its colour when another condition is added -- which `hcl.series`,
being sorted by lightness, does not guarantee. Taking a slot another condition holds is a
SWAP, so two conditions never share a colour.

**A palette is SUPPORTED when it has a colour for every condition.** Okabe-Ito has 7,
Tol muted 9. An unsupported palette is offered disabled, never chosen. With more
conditions than ANY palette holds, the largest palette is used and the conditions past its
end get GENERATED colours: since 6 Nov 2026 each the page-fitted colour FARTHEST (CIEDE2000) from every colour
already drawn (`palettes.distinct`; golden-angle hues before: the closest pair, CIEDE2000, 6.9 at 20 conditions and
3.7 at 30 on Petroff 10, now 13.0 and 11.3). It never raises and never repeats a colour.

**THE SAFEGUARD** (6 Nov 2026, asked: "what happens if data set loads with more conditions than the palettes have
colors? we should add a safeguard for that"): `crowding` / `crowding_text` say so where colours are chosen (the
Confirm screen, the workspace's Conditions tool) -- how many colours are generated, and the closest pair of the
colours drawn when it is under `palettes.DISTINCT_FLOOR`, with what tells them apart (each chart names its
conditions; a folder design's Colour by a factor gives each group one hue).

**A COLOUR BY HAND** (`set_custom`, 6 Nov 2026, "or to manually adjust some colors"): any condition may take a colour
of its own over its palette slot, fitted to the page like every other, kept with the project's choice; its slot is
kept underneath, so "Use the palette's colour" gives it back. **PALETTES THE USER MADE** (`smtracker.palettes`) are
in `hcl.PALETTES` like the built-in ones (`palettes.ensure_registered`, called wherever palettes are listed or read).

**COLOURED BY A FACTOR** (`colour_by`, 25 Oct 2026, asked: "same colour different shade, user can define"): in a
project with a folder design (`folderdesign`), the conditions can take one HUE per value of a factor -- glucose one
colour, malate another -- each condition of that value in its own SHADE of it (`figstyle.day_shade`, darkest
first). The hues are slots per VALUE (`value_slots`), picked and swapped like a condition's; each condition keeps
its own slot underneath, so going back to "Condition" restores every colour it had.

Qt-free: imported with Qt blocked, like `onboarding_model`.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field

from .ui.theme import hcl

__all__ = ["ConditionColours", "supported_palettes", "unsupported_palettes",
           "default_palette", "palette_slots", "extra_colours", "project_key", "recall", "remember", "PAGE_COLOURS",
           "crowding_text"]

#: The `ui.json` key the per-project choices live under (per user, never beside the data).
SETTINGS_KEY = "condition_colours"
#: The export payload's key for `ConditionColours.page_variants` (read by `figure_export.page_coloured`).
PAGE_COLOURS = "page_colours"
#: How many shades of a condition's colour an export may meet (`ConditionColours.page_variants`): fits side by side
#: (up to five models) and replicates.
PAGE_SHADES = 8


def _user_palettes() -> None:
    """The palettes the user made, in `hcl.PALETTES` (`palettes.ensure_registered`); never fatal."""
    try:
        from .palettes import ensure_registered

        ensure_registered()
    except Exception:                                   # noqa: BLE001 - a broken settings file must not stop a window
        pass


def palette_size(key: str) -> int:
    return len(hcl.PALETTES[key].order)


def supported_palettes(n: int) -> list[str]:
    """The palettes with a colour for each of ``n`` conditions, in declared order (the user's own after the
    built-in ones)."""
    _user_palettes()
    return [k for k in hcl.PALETTES if palette_size(k) >= n]


def unsupported_palettes(n: int) -> list[str]:
    """The palettes too small for ``n`` conditions -- shown, never selectable."""
    _user_palettes()
    return [k for k in hcl.PALETTES if palette_size(k) < n]


def default_palette(n: int) -> str:
    """The default where it fits; otherwise the first BUILT-IN palette that does; with more conditions than any
    built-in palette holds, the LARGEST one (extras are generated)."""
    from .palettes import is_user

    supported = [k for k in supported_palettes(n) if not is_user(k)]
    if hcl.DEFAULT_PALETTE in supported:
        return hcl.DEFAULT_PALETTE
    if supported:
        return supported[0]
    return max((k for k in hcl.PALETTES if not is_user(k)), key=palette_size)


def palette_slots(key: str, *, dark: bool = False) -> tuple:
    """The whole palette ``key``, fitted to the page, in its declared order (`hcl.slots`): what a colour that is not a
    condition's (a Clustering cluster) is drawn from."""
    return hcl.slots(key, dark=dark)


def extra_colours(count: int, *, taken, dark: bool) -> list[str]:
    """``count`` colours, each the page-fitted colour farthest (CIEDE2000) from every colour taken and chosen before it
    (`palettes.distinct`), fitted to the page; cached, since every condition's colour reads its palette's pool."""
    return list(_extra(int(count), tuple(c.lower() for c in taken), bool(dark)))


@functools.lru_cache(maxsize=64)
def _extra(count: int, taken: tuple, dark: bool) -> tuple:
    from .palettes import distinct

    return tuple(distinct(count, taken=taken, dark=dark))


def crowding_text(model: "ConditionColours", names=None, *, dark: bool = False) -> str | None:
    """The safeguard's words, or None where every condition has a palette colour and none is hard to tell apart:
    how many colours are generated past the palette's end, and the closest pair drawn when it is under
    `palettes.DISTINCT_FLOOR`. ``names``: condition -> the name shown."""
    found = model.crowding(dark=dark)
    if found is None:
        return None
    names = dict(names or {})
    parts = []
    if found["generated"]:
        label = hcl.PALETTES[model.palette].label
        parts.append(f"{len(model.conditions)} conditions, {palette_size(model.palette)} colours in {label}: "
                     f"{found['generated']} more were made, each as far from the others as can be")
    if found["close"]:
        d, a, b = found["close"]
        parts.append(f"{names.get(a, a)} and {names.get(b, b)} are hard to tell apart by colour (\u0394E {d:.1f}): "
                     f"each chart names its conditions; give one a colour by hand, or colour by a factor")
    return ". ".join(parts) + "." if parts else None


@dataclass
class ConditionColours:
    """The palette, and which slot each condition holds."""

    conditions: list[str]
    palette: str = ""
    #: condition -> slot index. Every condition holds exactly one, all different.
    slots: dict[str, int] = field(default_factory=dict)
    #: COLOURED BY THIS FACTOR ("" = each condition its own slot) and each value's slot, all different
    by: str = ""
    value_slots: dict[str, int] = field(default_factory=dict)
    #: condition -> (its value of `by`, its shade, how many shades), set by `colour_by`; never stored
    shading: dict = field(default_factory=dict, repr=False, compare=False)
    #: `by` was CHOSEN by the user (remembered), not the default that follows the nesting's inner factor
    by_chosen: bool = field(default=False, compare=False)
    #: condition -> a colour chosen BY HAND (``#rrggbb``), over its slot (`set_custom`); fitted to the page as drawn
    custom: dict = field(default_factory=dict)
    #: a SAVED choice kept although its palette is now too small (`from_json`): never reset here
    keep: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self):
        _user_palettes()
        self.conditions = list(self.conditions)
        if self.palette not in hcl.PALETTES or (
                not self.keep and self.palette not in supported_palettes(len(self.conditions))
                and supported_palettes(len(self.conditions))):
            self.palette = default_palette(len(self.conditions))
        if not self._slots_valid(self.slots):
            self.slots = {c: i for i, c in enumerate(self.conditions)}

    # -- reading ---------------------------------------------------------- #
    def pool(self, *, dark: bool = False, palette: str | None = None) -> list[str]:
        """Every slot's colour, in slot order: the palette's own (fitted to the page),
        then generated ones if there are more conditions than it holds."""
        base = list(hcl.slots(palette or self.palette, dark=dark))
        missing = len(self.conditions) - len(base)
        if missing > 0:
            base += extra_colours(missing, taken=base, dark=dark)
        return base

    def colour(self, condition: str, *, dark: bool = False) -> str | None:
        if condition in self.custom:
            return hcl.fit_to_ground(self.custom[condition], hcl.GROUNDS["dark" if dark else "light"])
        if self.by and condition in self.shading:
            from .figstyle import day_shade

            value, shade, shades = self.shading[condition]
            slot = self.value_slots.get(value)
            if slot is not None:
                return day_shade(self.pool(dark=dark)[slot], shade, shades)
        slot = self.slots.get(condition)
        if slot is None:
            return None
        return self.pool(dark=dark)[slot]

    def colours(self, *, dark: bool = False) -> dict[str, str]:
        return {c: self.colour(c, dark=dark) for c in self.slots}

    def page_variants(self, *, dark: bool, shades: int = PAGE_SHADES) -> dict[str, list[str]]:
        """Every condition colour the screen hands out in its theme (``dark``), lower-case, to ``[light, dark]``: the
        condition's colour fitted to a white page and to a dark one, and each of its shades (`figstyle.day_shade`, up
        to ``shades`` of them: fits side by side, replicates) likewise. An EXPORT draws from this, by its page
        (`figure_export.page_coloured`), so a figure's colours follow the FIGURE's page and never the screen's theme
        (found by the overnight tester, 4 Nov 2026: in the dark theme an olive condition printed yellow on white)."""
        from .figstyle import day_shade

        out: dict[str, list[str]] = {}
        for condition in self.slots:
            seen, light, deep = self.colour(condition, dark=dark), self.colour(condition), self.colour(condition, dark=True)
            if not (seen and light and deep):
                continue
            for n in range(1, int(shades) + 1):
                for i in range(n):
                    out.setdefault(day_shade(seen, i, n).lower(), [day_shade(light, i, n), day_shade(deep, i, n)])
        return out

    # -- coloured by a factor (folder designs) ----------------------------- #
    def colour_by(self, factors: dict, name: str, shade_order=()) -> None:
        """Colour by factor ``name`` (``""`` back to one colour per condition): one hue per value, in the order
        values first appear among the conditions, a value's held slot kept; each condition of a value a shade of it,
        in the order of ``shade_order`` (the other factors, outer first) and then the conditions' own order."""
        self.by = str(name or "")
        self.shading = {}
        if not self.by:
            return
        values = []
        for c in self.conditions:
            value = (factors.get(c) or {}).get(self.by)
            if value is not None and str(value) not in values:
                values.append(str(value))
        size = max(len(self.pool()), len(values))
        kept = {v: s for v, s in self.value_slots.items() if v in values and 0 <= s < size}
        if len(set(kept.values())) != len(kept):
            kept = {}
        free = [s for s in range(size) if s not in kept.values()]
        self.value_slots = {v: kept[v] if v in kept else free.pop(0) for v in values}
        members: dict = {}
        for c in self.conditions:
            value = (factors.get(c) or {}).get(self.by)
            if value is not None:
                members.setdefault(str(value), []).append(c)

        # each shading factor's values in the order they first appear (the folder's order, as `factor_layout.nested`)
        firsts = {n: {} for n in shade_order}
        for c in self.conditions:
            for n in shade_order:
                v = str((factors.get(c) or {}).get(n, ""))
                firsts[n].setdefault(v, len(firsts[n]))

        def rank(c):
            return tuple(firsts[n][str((factors.get(c) or {}).get(n, ""))] for n in shade_order)

        for value, cs in members.items():
            ordered = sorted(cs, key=lambda c: (rank(c), self.conditions.index(c))) if shade_order else cs
            for k, c in enumerate(ordered):
                self.shading[c] = (value, k, len(ordered))

    def crowding(self, *, dark: bool = False) -> dict | None:
        """``{"generated": how many colours past the palette's end, "close": (ΔE, a, b) or None}`` -- the closest pair
        of the colours drawn when under `palettes.DISTINCT_FLOOR` -- or None where neither applies."""
        from .palettes import DISTINCT_FLOOR, closest

        generated = max(0, len(self.conditions) - palette_size(self.palette))
        drawn = [(c, self.colour(c, dark=dark)) for c in self.conditions]
        drawn = [(c, v) for c, v in drawn if v]
        d, i, j = closest([v for _c, v in drawn])
        close = (d, drawn[i][0], drawn[j][0]) if i >= 0 and d < DISTINCT_FLOOR else None
        if not generated and close is None:
            return None
        return {"generated": generated, "close": close}

    def set_custom(self, condition: str, colour: str | None) -> bool:
        """``condition`` drawn in ``colour`` (``#rrggbb``) whatever its slot, or back to its slot's (None). True if
        anything changed."""
        import re

        if condition not in self.slots:
            return False
        if colour is None:
            return self.custom.pop(condition, None) is not None
        colour = str(colour).strip().lower()
        if not re.fullmatch(r"#[0-9a-f]{6}", colour):
            raise ValueError(f"not a colour: {colour!r}")
        if self.custom.get(condition) == colour:
            return False
        self.custom[condition] = colour
        return True

    def value_of(self, condition: str) -> str | None:
        """The value of the colouring factor ``condition`` holds, or None when coloured per condition."""
        entry = self.shading.get(condition) if self.by else None
        return entry[0] if entry else None

    def holder_of(self, slot: int) -> str | None:
        return next((c for c, s in self.slots.items() if s == slot), None)

    def supported(self) -> list[str]:
        return supported_palettes(len(self.conditions))

    def unsupported(self) -> list[str]:
        return unsupported_palettes(len(self.conditions))

    # -- changing --------------------------------------------------------- #
    def assign(self, condition: str, slot: int) -> bool:
        """Give ``condition`` that slot, SWAPPING with whoever held it. True if anything moved. Coloured by a
        factor, the slot goes to the condition's VALUE (every condition of it), swapping with the value that held it."""
        value = self.value_of(condition)
        if value is not None:
            if self.value_slots.get(value) == slot:
                return False
            if not 0 <= slot < len(self.pool()):
                raise ValueError(f"slot {slot} is outside the palette")
            holder = next((v for v, s in self.value_slots.items() if s == slot), None)
            if holder is not None:
                self.value_slots[holder] = self.value_slots[value]
            self.value_slots[value] = slot
            return True
        if condition not in self.slots:
            return False
        if self.slots[condition] == slot:
            # its own palette colour picked again: a colour by hand gives way to it (`set_custom`)
            return self.custom.pop(condition, None) is not None
        if not 0 <= slot < len(self.pool()):
            raise ValueError(f"slot {slot} is outside the palette")
        self.custom.pop(condition, None)
        holder = self.holder_of(slot)
        if holder is not None:
            self.slots[holder] = self.slots[condition]
        self.slots[condition] = slot
        return True

    def set_palette(self, key: str) -> bool:
        """Switch palette and reset every condition to its own slot. Refuses an
        unsupported palette (False) unless no palette is supported at all."""
        if key == self.palette or key not in hcl.PALETTES:
            return False
        if self.supported() and key not in self.supported():
            return False
        self.palette = key
        self.slots = {c: i for i, c in enumerate(self.conditions)}
        self.value_slots = {v: i for i, v in enumerate(self.value_slots)}
        return True

    # -- persistence ------------------------------------------------------ #
    def to_json(self) -> dict:
        out = {"palette": self.palette, "slots": dict(self.slots)}
        if self.by or self.value_slots:
            out.update(by=self.by, value_slots=dict(self.value_slots))
        if self.by_chosen:
            out["by_chosen"] = True
        if self.custom:
            out["custom"] = dict(self.custom)
        return out

    @classmethod
    def from_json(cls, conditions, data) -> "ConditionColours":
        """A remembered choice, or the default where it no longer fits (a condition
        gone or added, a palette now unsupported, slots out of range)."""
        conditions = list(conditions)
        try:
            palette = str(data["palette"])
            slots = {str(k): int(v) for k, v in dict(data["slots"]).items()}
        except (KeyError, TypeError, ValueError):
            return cls(conditions)
        _user_palettes()
        # A PALETTE NOW TOO SMALL is replaced by the default where a palette of the size there was before Kelly 20
        # (6 Nov 2026) fits; a choice saved with more conditions than any palette then held keeps its palette and
        # its generated colours past the end -- a palette added later never recolours a project in silence
        fits_before = [k for k in supported_palettes(len(conditions)) if palette_size(k) <= hcl.LARGEST_BEFORE_KELLY]
        if palette not in supported_palettes(len(conditions)) and fits_before:
            return cls(conditions)
        if palette not in hcl.PALETTES:
            return cls(conditions)
        made = cls(conditions, palette=palette, slots=slots, keep=True)
        try:
            made.custom = {str(k): str(v).lower() for k, v in dict(data.get("custom") or {}).items()
                           if str(k) in made.slots}
        except (TypeError, ValueError, AttributeError):
            made.custom = {}
        try:
            made.by = str(data.get("by") or "")
            made.value_slots = {str(k): int(v) for k, v in dict(data.get("value_slots") or {}).items()}
            made.by_chosen = bool(data.get("by_chosen", False))
        except (TypeError, ValueError, AttributeError):
            made.by, made.value_slots, made.by_chosen = "", {}, False
        return made

    def copy(self) -> "ConditionColours":
        made = ConditionColours(list(self.conditions), self.palette, dict(self.slots), keep=True)
        made.by, made.value_slots = self.by, dict(self.value_slots)
        made.shading = dict(self.shading)
        made.by_chosen = self.by_chosen
        made.custom = dict(self.custom)
        return made

    def _slots_valid(self, slots) -> bool:
        if set(slots) != set(self.conditions):
            return False
        values = list(slots.values())
        size = max(len(self.conditions), palette_size(self.palette))
        return len(set(values)) == len(values) and all(0 <= v < size for v in values)


# -- remembered per project ------------------------------------------------ #
def project_key(roots) -> str | None:
    """The folder a project is remembered by: the common parent of its condition roots
    (the folder `open_workspace` titles the window with). None with no roots."""
    import os
    from pathlib import Path

    paths = [str(Path(r).resolve().parent) for r in roots]
    if not paths:
        return None
    return os.path.commonpath(paths)


def recall(key: str | None, conditions) -> ConditionColours:
    """The choice remembered for ``key``, or the default where none fits."""
    from .ui import theme

    stored = (theme.remembered(SETTINGS_KEY) or {}) if key else {}
    entry = stored.get(key) if isinstance(stored, dict) else None
    if entry is None:
        return ConditionColours(list(conditions))
    return ConditionColours.from_json(conditions, entry)


def remember(key: str | None, colours: ConditionColours) -> None:
    """Save ``colours`` for ``key``, keeping every other project's choice. A failed
    write is warned about, never fatal: a colour preference must not stop a window."""
    import warnings

    from .ui import theme

    if not key:
        return
    stored = theme.remembered(SETTINGS_KEY)
    stored = dict(stored) if isinstance(stored, dict) else {}
    stored[key] = colours.to_json()
    try:
        theme.remember(SETTINGS_KEY, stored)
    except OSError as exc:
        warnings.warn(f"could not remember condition colours: {exc}", stacklevel=2)
