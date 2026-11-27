"""shadcn/ui's token table, transcribed — and why transcription is correct here.

Resolving tokens from a published payload at runtime is the better habit, since
it leaves no second copy to drift -- and it is not available here. shadcn has no
such payload: its tokens ship as ~30 lines of CSS custom properties that the CLI
writes into *your* ``globals.css``, and which you then own and edit. There is no
upstream to read. So these values are literal, and the note below records
exactly which shadcn artefact they came from.

Source: the ``zinc`` base colour of shadcn/ui (``new-york`` style), as emitted
by ``npx shadcn@latest init``, converted from its ``oklch()`` form to hex.
Captured 2026-08-26 against shadcn 4.19.0.

**The palette is neutral on purpose, and that is a real design position.**
``--primary`` is zinc-900, not a brand hue: shadcn expects colour to arrive
from content, not from chrome. That is why the Okabe-Ito condition colours
below are left exactly as the analysis chose them — they encode DATA, and a
design system with no opinion about your data cannot overrule them.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import hcl

__all__ = ["CONDITIONS", "DayRamp", "Tokens", "check_frozen", "day_ramp"]

#: Okabe-Ito, the colour-blind-safe qualitative palette the analysis uses for
#: conditions. NOT a shadcn token and deliberately not converted to one.
CONDITIONS = {
    "Glucose": "#0072B2",
    "Malate": "#D55E00",
    "Succinate": "#009E73",
}

#: **The ramp INVERTS between schemes, and that is not cosmetic.** Lightness
#: runs 94 -> 40 on light but 26 -> 78 on dark, so D1 is the palest chip on
#: white and the darkest on black. Shipping only a light table looked fine until
#: dark mode was rendered, where pale chips sat on near-black cards as the
#: brightest thing on screen: the ordinal DIRECTION survived, but the emphasis
#: inverted with it.
_SPAN = {"light": (94.0, 40.0), "dark": (26.0, 78.0)}


class DayRamp:
    """The seven-ish day tints, derived from the CONDITION's own colour.

    Days are ORDINAL -- D3 comes after D2 -- so they take a sequential ramp, one
    hue at varying lightness. Conditions are NOMINAL, so they take distinct
    hues. Deriving the ramp from the condition's colour is what keeps those two
    systems from colliding: whatever hue a condition holds, its days are that
    hue and nothing else. It also means a colour change REACHES the day tags,
    which is the point -- a tag on the old hue would be claiming a condition it
    no longer matches.

    Chroma tapers with lightness rather than staying flat: a pale tint at full
    chroma reads as a wash of colour instead of a quiet tint, and the light end
    of this ramp carries small text.

    DERIVED at runtime, not frozen. An earlier version kept a table per
    (scheme, condition), which was fine until the colour picker arrived -- a
    frozen table cannot answer for a hue the user just chose. ``_FROZEN`` below
    pins the derivation to the values that table held.
    """

    def __init__(self, scheme: str, base: str, count: int = 7):
        self.count = max(1, count)
        self._lo, self._hi = _SPAN[scheme]
        self.adopt(base)

    def adopt(self, base: str) -> None:
        """Become the ramp for ``base``, IN PLACE.

        Mutating rather than returning a new ramp, because a ramp is held by
        every lane, the day strip and the table at once. Handing out a
        replacement means updating each of those, and forgetting one is
        invisible until something rebuilds from the stale copy.
        """
        self.base = base
        _lum, chroma, hue = hcl._hex_to_hcl(base)
        self.hue, self._chroma = hue, chroma

    def _at(self, index: int) -> tuple[float, float]:
        span = max(1, self.count - 1)
        step = min(max(index - 1, 0), span) / span
        lum = self._lo + (self._hi - self._lo) * step
        # 0.30 -> 1.0 of the condition's chroma across the ramp.
        return lum, self._chroma * (0.30 + 0.70 * step)

    def fill(self, index: int) -> str:
        lum, chroma = self._at(index)
        return hcl.to_hex(lum, chroma, self.hue)

    def ink(self, index: int) -> str:
        """Whichever end of the ramp the fill can actually carry."""
        fill = self.fill(index)
        dark = hcl.to_hex(22.0, min(self._chroma, 30.0), self.hue)
        light = "#ffffff"
        return (dark if hcl.contrast(fill, dark) >= hcl.contrast(fill, light)
                else light)

    def tables(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """``(fills, inks)`` for every day, which is what the widgets take."""
        span = range(1, self.count + 1)
        return (tuple(self.fill(i) for i in span),
                tuple(self.ink(i) for i in span))


def day_ramp(scheme: str = "light", base: str = "#0072B2",
             count: int = 7) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``(fills, inks)`` for ``count`` days of a condition coloured ``base``."""
    return DayRamp(scheme, base, count).tables()


#: A REGRESSION PIN, not a lookup. These are the values the frozen table held
#: before the ramp was derived at runtime; ``check_frozen`` re-derives them and
#: compares, so the port of the colour maths is proved rather than trusted.
_FROZEN = {
    ("light", "#0072B2", 7): ("#e1efff", "#0b3757"),
    ("light", "#D55E00", 8): ("#ffe3d8", "#522a1a"),
    ("light", "#009E73", 6): ("#d8f4e7", "#004025"),
    ("dark", "#0072B2", 7): ("#2c3f55", "#ffffff"),
    ("dark", "#D55E00", 8): ("#5d3222", "#ffffff"),
    ("dark", "#009E73", 6): ("#214436", "#ffffff"),
}


def check_frozen() -> list[str]:
    """Re-derive every pinned ramp and report any that moved. Empty is good."""
    bad = []
    for (scheme, base, count), (fill, ink) in _FROZEN.items():
        fills, inks = day_ramp(scheme, base, count)
        if (fills[0], inks[0]) != (fill, ink):
            bad.append(f"{scheme} {base} x{count}: "
                       f"{fills[0]}/{inks[0]} != {fill}/{ink}")
    return bad


@dataclass(frozen=True)
class Tokens:
    """One colour scheme's worth of shadcn variables.

    Named after the CSS custom properties rather than after a role vocabulary
    of our own, so a value here can be checked against shadcn's docs by
    searching for the same string. ``--card`` and ``--background`` are both
    pure white in the light scheme; that is not a mistake to tidy up, it is
    what makes shadcn's cards read as bordered regions rather than as raised
    surfaces.
    """

    color_scheme: str = "light"

    def __post_init__(self):
        if self.color_scheme not in ("light", "dark"):
            raise ValueError(f"unknown scheme {self.color_scheme!r}")

    @property
    def _dark(self) -> bool:
        return self.color_scheme == "dark"

    # -- surfaces ---------------------------------------------------------- #
    @property
    def background(self) -> str:
        return "#09090B" if self._dark else "#FFFFFF"

    @property
    def card(self) -> str:
        return "#18181B" if self._dark else "#FFFFFF"

    @property
    def muted(self) -> str:
        """Trays, progress tracks, quiet fills. shadcn's zinc-100 / zinc-800."""
        return "#27272A" if self._dark else "#F4F4F5"

    @property
    def subtle(self) -> str:
        """A half-step under ``background`` for panel wells.

        NOT a shadcn token — shadcn would use ``bg-muted/50``, which Qt has no
        equivalent for without compositing. Named distinctly so nobody hunts
        for it in shadcn's docs.
        """
        return "#131316" if self._dark else "#FAFAFA"

    @property
    def popover(self) -> str:
        """``--popover``. Equal to ``--card`` in the stock theme, kept separate
        because a menu floating over a card is exactly where the two would need
        to diverge if anyone ever re-themed them."""
        return "#18181B" if self._dark else "#FFFFFF"

    @property
    def accent(self) -> str:
        """``--accent``: the hover fill on a menu item or a ghost button."""
        return "#27272A" if self._dark else "#F4F4F5"

    # -- the sidebar ------------------------------------------------------- #
    # shadcn gives the sidebar its OWN eight variables rather than reusing the
    # page's, so a sidebar can be re-themed without touching anything else.
    # Every one of them lands on a value this port already has, which is worth
    # recording: the sidebar is not a new palette, it is the same ramp used at
    # different rungs. They are still spelled out under shadcn's names, per this
    # class's rule that a value here can be found by searching shadcn's docs.
    #
    # One honest difference: the registry's stock `globals.css` is NEUTRAL
    # (`oklch(... 0 0)`) while this port is zinc throughout, a decision made
    # long before the sidebar existed. The LIGHTNESSES match rung for rung
    # (0.985 / 0.97 / 0.922 / 0.205); only the hue follows the port.
    @property
    def sidebar(self) -> str:
        """``--sidebar``: 0.985, a half-step under the page. Equals `subtle`."""
        return "#18181B" if self._dark else "#FAFAFA"

    @property
    def sidebar_foreground(self) -> str:
        return "#FAFAFA" if self._dark else "#09090B"

    @property
    def sidebar_accent(self) -> str:
        """``--sidebar-accent``: the hover and active fill on a nav item."""
        return "#27272A" if self._dark else "#F4F4F5"

    @property
    def sidebar_accent_foreground(self) -> str:
        return "#FAFAFA" if self._dark else "#18181B"

    @property
    def sidebar_border(self) -> str:
        """The rail's edge, and the hairline down a nested nav group."""
        return "#27272A" if self._dark else "#E4E4E7"

    @property
    def sidebar_primary(self) -> str:
        """The workspace tile's fill -- the one saturated-dark thing up there."""
        return "#FAFAFA" if self._dark else "#18181B"

    @property
    def sidebar_primary_foreground(self) -> str:
        return "#18181B" if self._dark else "#FAFAFA"

    # -- lines and ink ----------------------------------------------------- #
    @property
    def border(self) -> str:
        return "#27272A" if self._dark else "#E4E4E7"

    @property
    def input(self) -> str:
        """shadcn keeps ``--input`` separate from ``--border`` so control
        outlines can diverge from dividers. They are equal in the stock theme;
        kept separate anyway, because collapsing them here would silently
        delete that affordance the first time someone re-themed it."""
        return "#3F3F46" if self._dark else "#E4E4E7"

    @property
    def ring(self) -> str:
        """``--ring``, the focus/hover halo -- a mid grey in BOTH schemes.

        Distinct from `--border` and `--input`, which it is often mistaken for:
        those two are equal in the stock zinc theme and this one is not. From
        the shipped theme, `oklch(0.708 0 0)` light and `oklch(0.556 0 0)` dark.
        """
        return "#8A8A8A" if self._dark else "#B5B5B5"

    @property
    def foreground(self) -> str:
        return "#FAFAFA" if self._dark else "#09090B"

    @property
    def muted_foreground(self) -> str:
        return "#A1A1AA" if self._dark else "#71717A"

    # -- the one filled thing ---------------------------------------------- #
    @property
    def primary(self) -> str:
        return "#FAFAFA" if self._dark else "#18181B"

    @property
    def primary_foreground(self) -> str:
        return "#18181B" if self._dark else "#FAFAFA"

    # -- geometry ---------------------------------------------------------- #
    #: shadcn's ``--radius`` is **0.625rem = 10 px**, and the scale around it is
    #: MULTIPLICATIVE, not the ``radius ± Npx`` arithmetic it used to be. Read
    #: off the shipped theme (``ui.shadcn.com/r/colors/zinc.json``) and the
    #: theming docs::
    #:
    #:     --radius:    0.625rem          10 px
    #:     --radius-sm: calc(--radius * 0.6)   6 px
    #:     --radius-md: calc(--radius * 0.8)   8 px
    #:     --radius-lg: var(--radius)         10 px
    #:     --radius-xl: calc(--radius * 1.4)  14 px
    #:
    #: This whole scale used to be derived from 0.5rem, which made every corner
    #: in the window one step too tight -- and cards, which the registry marks
    #: ``rounded-xl``, were drawn at 8 px against shadcn's 14.
    radius_sm: float = 6.0
    radius_md: float = 8.0
    radius_lg: float = 10.0
    radius_xl: float = 14.0

    #: Which token each component takes, from the registry itself
    #: (``shadcn view @shadcn/<item>``)::
    #:
    #:     Card                                  rounded-xl    14
    #:     Button / Input / Popover / menu body   rounded-md     8
    #:     DropdownMenuItem                       rounded-sm     6
    #:     DialogContent                          rounded-lg    10
    #:     Badge / Progress / scrollbar thumb     rounded-full
    #:
    #: -- shadow ------------------------------------------------------------ #
    #: Tailwind v4's scale, read out of ``tailwindcss/theme.css`` rather than
    #: remembered, because v4 RENAMED the whole ladder: what v3 called
    #: ``shadow-sm`` is v4's ``shadow-xs``, and v3's plain ``shadow`` is v4's
    #: ``shadow-sm``. Ours was the v3 value under the v4 name, i.e. a full step
    #: too tight AND a single layer::
    #:
    #:     --shadow-xs: 0 1px 2px 0 rgb(0 0 0 / 0.05)          <- what we had
    #:     --shadow-sm: 0 1px 3px 0 rgb(0 0 0 / 0.1),
    #:                  0 1px 2px -1px rgb(0 0 0 / 0.1)        <- Card
    #:     --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1),
    #:                  0 2px 4px -2px rgb(0 0 0 / 0.1)        <- menus
    #:     --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1),
    #:                  0 4px 6px -4px rgb(0 0 0 / 0.1)        <- dialogs
    #:
    #: THE SECOND LAYER IS THE POINT. A Tailwind shadow is a wide soft wash
    #: plus a tight contact shadow pulled in by a negative spread; dropping it
    #: leaves one even blur that reads as blunt no matter what alpha it gets.
    #:
    #: Each layer is ``(dx, dy, blur, spread, alpha)`` -- spread included,
    #: since every second layer here has one.
    #: ``shadow-xs``. NOT a smaller shadow-sm -- it is the one the Button's
    #: ``outline`` variant actually asks for, and the only single-layer step on
    #: the ladder. A filled ``default`` button asks for no shadow at all.
    shadow_xs: tuple = ((0.0, 1.0, 2.0, 0.0, 0.05),)

    shadow_sm: tuple = ((0.0, 1.0, 3.0, 0.0, 0.10),
                        (0.0, 1.0, 2.0, -1.0, 0.10))
    shadow_md: tuple = ((0.0, 4.0, 6.0, -1.0, 0.10),
                        (0.0, 2.0, 4.0, -2.0, 0.10))
    shadow_lg: tuple = ((0.0, 10.0, 15.0, -3.0, 0.10),
                        (0.0, 4.0, 6.0, -4.0, 0.10))

    #: ``DialogOverlay`` is ``bg-black/50``.
    overlay_alpha: float = 0.5

    #: Control heights. shadcn's ``h-9`` default and ``h-8`` small.
    control_height: int = 36
    control_height_sm: int = 32
