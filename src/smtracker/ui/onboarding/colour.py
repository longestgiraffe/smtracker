"""Which colour each condition holds, and the palette they are drawn from.

**A palette is a SET, not a list of options.** Choosing one resets every
condition to its own slot, because the guarantee a colour-vision-safe palette
makes is about the whole set at once: Okabe-Ito's separation is a property of
its seven colours together, and picking three of them out of two palettes
throws that away while still looking deliberate.

**Slots, not colours.** A condition holds a slot INDEX, and the colour is read
from the palette. That is what makes switching palettes a one-line operation and
what makes "taking a colour another condition holds" a SWAP rather than a
collision -- two conditions cannot end up on the same hue, because a slot has
exactly one holder.

**The rules live in `smtracker.condition_colours`**, which the workspace reads
too: this is the Qt wrapper Confirm Project edits, so the colours shown here are
exactly the ones the workspace draws (both page-fitted, both in declared order).
"""

from __future__ import annotations

from qtpy.QtCore import QObject, Signal

from ...condition_colours import ConditionColours
from ..theme import hcl

__all__ = ["ColourChoice"]


class ColourChoice(QObject):
    """The palette, and one slot per condition -- a `ConditionColours` with a signal."""

    changed = Signal()

    def __init__(self, names: list[str], palette: str | None = None, *,
                 model: ConditionColours | None = None, dark: bool = False,
                 parent=None):
        super().__init__(parent)
        self.names = list(names)
        self.dark = dark
        self.model = (model.copy() if model is not None
                      else ConditionColours(self.names, palette or ""))

    @property
    def palette(self) -> str:
        return self.model.palette

    @property
    def slots(self) -> list[int]:
        """condition index -> slot index."""
        return [self.model.slots[n] for n in self.names]

    def colours(self, key: str | None = None) -> list[str]:
        """Every slot's colour, in the palette's declared order, fitted to the page --
        the SAME values the workspace draws with."""
        return self.model.pool(dark=self.dark, palette=key or self.palette)

    def colour_names(self, key: str | None = None) -> list[str]:
        """The colour NAMES, for a picker that can say "Vermillion" instead of
        "#D55E00". Colours generated past the palette's end are numbered.

        NOT ``names`` -- that attribute already holds the CONDITION names.
        """
        order = list(hcl.PALETTES[key or self.palette].order)
        extra = len(self.colours(key)) - len(order)
        return order + [f"colour {len(order) + i + 1}" for i in range(max(0, extra))]

    def colour_of(self, index: int) -> str:
        """Condition ``index``'s colour as drawn: its slot's, or the one chosen for it by hand (`set_custom`)."""
        return self.model.colour(self.names[index], dark=self.dark) or self.colours()[self.slots[index]]

    def set_custom(self, index: int, colour: str | None) -> None:
        """Condition ``index`` in ``colour`` by hand (None: back to its palette colour)."""
        if self.model.set_custom(self.names[index], colour):
            self.changed.emit()

    def holder_of(self, slot: int) -> str | None:
        """Which condition currently holds ``slot``, if any."""
        return self.model.holder_of(slot)

    def assign(self, index: int, slot: int) -> None:
        """Give ``index`` that slot, SWAPPING with whoever held it."""
        if self.model.assign(self.names[index], slot):
            self.changed.emit()

    def set_palette(self, key: str) -> None:
        """Switch palette; an unsupported one is refused (the picker disables it)."""
        if self.model.set_palette(key):
            self.changed.emit()

    def supported(self, key: str) -> bool:
        return key in self.model.supported()

    def palette_swatches(self, key: str | None = None) -> list[str]:
        """The first five colours of a palette — what the picker previews."""
        return self.colours(key)[:5]
