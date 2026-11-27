"""The series palette: Okabe-Ito, fitted to the page, plus the HCL that fits it.

**Why a whole colour space lives in here.** The published palette arrives as hex
and its LUMINANCE has to be moved without touching its hue or chroma — doing
that in RGB shifts the hue, which is the one property carrying the
discrimination. So this converts properly, both ways: polar CIELUV (which is
what HCL *is*) <-> CIELUV <-> CIE XYZ <-> linear sRGB <-> gamma-encoded sRGB, at
the D65 white point.

**The palette is Okabe-Ito, not an invention — and getting there took three
tries.** The first was an off-the-shelf categorical viz set. The second was a
searched HCL sequential ramp, which optimised the colour-blindness metric and
was **wrong**: it scored 25.1 normal-vision CIEDE2000 between its four steps
against Okabe-Ito's 38.4, ran L* 16 to 60 so every colour was dark, and stayed
inside one blue-green-olive hue family. It was reported, correctly, as "all dark
and similar to each other" — by a reader with normal colour vision.

The mistake was in the objective, not the arithmetic. Maximising separation
under *simulated dichromacy* subject to a page-contrast floor drove the search
into a narrow hue band (blue-to-yellow is the axis red-green deficiency
preserves) and down to low luminance (on a white page, darker is more contrast).
Both moves are locally right, and together they produce a set nobody can tell
apart.

So this ships **Okabe & Ito's palette** (2008) — the set the field settled on for
exactly this job, and the one R exposes as ``palette.colors("Okabe-Ito")``. It is
QUALITATIVE: the discriminating work is done by hue, across the whole wheel,
which is what buys 38.4 at a glance. Its colour-blind safety was designed in by
people who tested it on readers, rather than derived here from a simulation.

**Two things are done to it, and both are measured rather than assumed:**

* **Luminance is fitted to the page.** Okabe-Ito is specified with no background
  in mind, and several of its colours miss 3 : 1 against white — orange is
  L* 71, contrast 2.25, which is fine on a mid-grey and invisible as a 2 px line
  here. Each colour's L* is walked, hue and chroma held, until it clears. That
  costs nothing in separation, because hue is carrying it.
* **The chosen colours are ordered by luminance.** The data are ordered
  (conditions are a protocol sequence, replicate days are a time order), so the
  palette is handed out light-to-dark and the order reading survives. Nothing
  DEPENDS on it any more, which is the difference from the ramp this replaced.

Every number quoted above was MEASURED — normal-vision separation, deuteranopia,
protanopia, tritanopia, page contrast, gamut — against an independent
reimplementation of the dichromat simulation, so none of them is checked by the
arithmetic that produced it.




---

**A whole colour space rather than a frozen table**, because the day ramp needs
LIVE derivation once a colour picker exists: a table cannot answer for a hue the
user just chose. Keeping only the two conversions ``DayRamp`` calls would have
meant re-deriving CIELUV by hand, which is the kind of thing that is subtly
wrong and stays subtly wrong. Nothing outside the standard library is imported
here -- ``math``, ``functools`` and ``dataclasses``, and no Qt.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

__all__ = [
    "GROUNDS",
    "MIN_CONTRAST",
    "MIN_NORMAL_DELTA_E",
    "MIN_REDGREEN_DELTA_E",
    "OKABE_ITO",
    "SERIES_ORDER",
    "contrast",
    "fit_to_ground",
    "relative_luminance",
    "series",
    "to_hex",
]

#: D65, the white point sRGB is defined against. Y is 100 by CIE convention.
_XN, _YN, _ZN = 95.047, 100.0, 108.883

#: sRGB's own transfer function constants. Not 2.2 — sRGB is a linear toe with a
#: 2.4 power above it, and a plain 2.2 gamma shifts every dark step.
_LINEAR_CUT, _LINEAR_SLOPE = 0.0031308, 12.92
_GAMMA, _GAMMA_A = 2.4, 0.055


def _uv_white() -> tuple[float, float]:
    d = _XN + 15.0 * _YN + 3.0 * _ZN
    return 4.0 * _XN / d, 9.0 * _YN / d


def _luv_to_xyz(lum: float, u: float, v: float) -> tuple[float, float, float]:
    if lum <= 0.0:
        return 0.0, 0.0, 0.0
    un, vn = _uv_white()
    up = u / (13.0 * lum) + un
    vp = v / (13.0 * lum) + vn
    # The CIE two-branch inverse. The linear branch below L* = 8 is what keeps
    # the darkest colours from collapsing to black.
    y = (_YN * ((lum + 16.0) / 116.0) ** 3 if lum > 8.0
         else _YN * lum / 903.2962962962963)
    if abs(vp) < 1e-12:
        return 0.0, y, 0.0
    x = 9.0 * y * up / (4.0 * vp)
    z = y * (12.0 - 3.0 * up - 20.0 * vp) / (4.0 * vp)
    return x, y, z


def _gamma_encode(c: float) -> float:
    if c <= _LINEAR_CUT:
        return _LINEAR_SLOPE * c
    return (1.0 + _GAMMA_A) * (c ** (1.0 / _GAMMA)) - _GAMMA_A


def _gamma_decode(c: float) -> float:
    if c <= 0.04045:
        return c / _LINEAR_SLOPE
    return ((c + _GAMMA_A) / (1.0 + _GAMMA_A)) ** _GAMMA


def _linear(value: str) -> tuple[float, float, float]:
    return tuple(_gamma_decode(int(value[i:i + 2], 16) / 255.0)
                 for i in (1, 3, 5))


def to_hex(lum: float, chroma: float, hue: float) -> str:
    """One HCL triple as an sRGB hex string, gamut-clipped.

    ``fixup`` in R's sense: a colour outside sRGB is clamped per channel rather
    than returned as ``NA``. Clamping is SILENT and a clamped colour is not the
    hue that was asked for, which is why :func:`fit_to_ground` checks the
    round-trip rather than trusting this.
    """
    rad = math.radians(hue)
    x, y, z = _luv_to_xyz(lum, chroma * math.cos(rad), chroma * math.sin(rad))
    x, y, z = x / 100.0, y / 100.0, z / 100.0
    # sRGB's own matrix, D65, written out so the space this lands in is visible.
    lin = (3.240479 * x - 1.537150 * y - 0.498535 * z,
           -0.969256 * x + 1.875992 * y + 0.041556 * z,
           0.055648 * x - 0.204043 * y + 1.057311 * z)
    out = [max(0, min(255, int(round(_gamma_encode(max(0.0, min(1.0, c)))
                                     * 255.0)))) for c in lin]
    return "#{:02x}{:02x}{:02x}".format(*out)


def _hex_to_hcl(value: str) -> tuple[float, float, float]:
    """sRGB hex back to polar CIELUV — the inverse of :func:`to_hex`."""
    r, g, b = _linear(value)
    x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) * 100.0
    y = (0.2126729 * r + 0.7151522 * g + 0.0721750 * b) * 100.0
    z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) * 100.0
    denom = x + 15.0 * y + 3.0 * z
    if denom <= 0.0:
        return 0.0, 0.0, 0.0
    up, vp = 4.0 * x / denom, 9.0 * y / denom
    un, vn = _uv_white()
    ratio = y / _YN
    lum = (116.0 * ratio ** (1 / 3) - 16.0 if ratio > 0.008856
           else 903.2962962962963 * ratio)
    u, v = 13.0 * lum * (up - un), 13.0 * lum * (vp - vn)
    return lum, math.hypot(u, v), math.degrees(math.atan2(v, u)) % 360.0


def relative_luminance(value: str) -> float:
    """WCAG relative luminance. Not L* — this is the one contrast is defined on."""
    r, g, b = _linear(value)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    """WCAG 2.1 contrast ratio between two hex colours."""
    la, lb = relative_luminance(a), relative_luminance(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


# --------------------------------------------------------------------------- #
# The palette
# --------------------------------------------------------------------------- #

#: Okabe & Ito (2008), verbatim. The names are theirs. Their black is omitted:
#: this design already spends near-black on its ink, and a series
#: indistinguishable from the axis labels is a series nobody reads as data.
OKABE_ITO = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "bluish green": "#009E73",
    "reddish purple": "#CC79A7",
    "sky blue": "#56B4E9",
    "yellow": "#F0E442",
    "orange": "#E69F00",
}

#: The order colours are taken in as ``n`` grows.
#:
#: **Prefix-stable on purpose.** Colour is indexed by a condition's position in
#: the catalogue, never by rank among the visible (see
#: ``viewdata.condition_series``), so adding a fourth series must not re-colour
#: the first three — a figure that changed colour when a condition was switched
#: on could not be compared with a screenshot of itself. Measured: this ordering
#: puts the most-separated triple first, which is this design's common case.
SERIES_ORDER = ("blue", "vermillion", "bluish green", "reddish purple",
                "sky blue", "yellow", "orange")

#: Paul Tol's *muted* qualitative scheme (SRON technical note), verbatim. The
#: names are shortened from his; his own light grey, reserved for "bad data", is
#: omitted for the reason Okabe-Ito's black is.
TOL_MUTED = {
    "indigo": "#332288",
    "rose": "#CC6677",
    "green": "#117733",
    "cyan": "#88CCEE",
    "sand": "#DDCC77",
    "teal": "#44AA99",
    "olive": "#999933",
    "purple": "#AA4499",
    "wine": "#882255",
}

#: Tol's order, REORDERED — and the reordering is the whole of the work here.
#:
#: Tol publishes the scheme as a set, not as a take-the-first-n sequence, so the
#: order had to be chosen and then MEASURED. Every 4-permutation of the nine was
#: put through the same three gates below at n = 3 and n = 4 on both grounds;
#: four distinct sets pass and this is the best of them. It scores **17.6**
#: red-green against Okabe-Ito's 12.3 and 31.4 for normal vision against its
#: 37.0 — better on the deficiency the floor exists for, slightly tighter for
#: normal vision, both clear. Slots past the fourth are NOT gate-checked, which
#: is equally true of Okabe-Ito's (see ``MIN_REDGREEN_DELTA_E``).
TOL_ORDER = ("indigo", "rose", "green", "cyan", "sand", "teal", "olive",
             "purple", "wine")

# --------------------------------------------------------------------------- #
# Three more, added 28 Oct 2026 (asked: "find 2-3 new color palettes fitting and add it"). Each published, each
# chosen and ordered as Tol muted was: every 4-permutation through the gates below at n = 3 and n = 4 on both
# grounds, the best red-green worst pair first. MEASURED with Machado (2009) severity-1 deuteranopia and protanopia
# and CIEDE2000 (skimage), a measurement that gives the shipped two their documented numbers (Okabe-Ito 12.2
# red-green / 37.0 normal, Tol muted 17.6 / 31.4). Seven published sets were measured and FAILED at every order,
# once fitted to the page: Tol bright (best 23.7 normal), IBM's (15.9), Dark2 (4.9 red-green), Set1, Tableau 10,
# seaborn's colorblind (10.6 red-green) and Petroff's six-colour set (24.8 normal) -- their light colours darken to
# 3 : 1 on white and close up on the others. Carto Safe passed but is Tol muted with two more colours.
# --------------------------------------------------------------------------- #

#: Paul Tol's *vibrant* qualitative scheme (SRON technical note), verbatim; his grey omitted, as for muted. 24 orders
#: pass; this one scores **15.8** red-green and **30.3** normal (tritanopia 7.8).
TOL_VIBRANT = {
    "blue": "#0077BB",
    "cyan": "#33BBEE",
    "teal": "#009988",
    "orange": "#EE7733",
    "red": "#CC3311",
    "magenta": "#EE3377",
}
TOL_VIBRANT_ORDER = ("blue", "teal", "orange", "magenta", "cyan", "red")

#: Petroff's accessible colour sequences (M. A. Petroff 2021, "Accessible Color Sequences for Data Visualization",
#: arXiv:2107.02270; Matplotlib's ``petroff8`` / ``petroff10`` styles, checked against them): picked by a survey of
#: readers with and without a colour-vision deficiency, so they read as modern figures rather than a
#: colour-blindness tool. Verbatim; the names are ours. Their greys are kept, LAST: a grey series reads as "no data"
#: beside coloured ones, so it is taken only when everything else is.
#: Eight colours: 48 orders pass; this one scores **14.5** red-green and **33.3** normal (tritanopia 18.3).
PETROFF_8 = {
    "blue": "#1845FB",
    "orange": "#FF5E02",
    "red": "#C91F16",
    "magenta": "#C849A9",
    "khaki": "#ADAD7D",
    "sky": "#86C8DD",
    "light blue": "#578DFF",
    "grey": "#656364",
}
PETROFF_8_ORDER = ("red", "magenta", "khaki", "light blue", "sky", "orange", "blue", "grey")

#: Ten colours, the largest palette here: 24 orders pass; this one scores **13.2** red-green and **34.2** normal
#: (tritanopia 25.9). Ten conditions get a palette colour each instead of generated ones.
PETROFF_10 = {
    "blue": "#3F90DA",
    "orange": "#FFA90E",
    "red": "#BD1F01",
    "grey": "#94A4A2",
    "purple": "#832DB6",
    "brown": "#A96B59",
    "dark orange": "#E76300",
    "khaki": "#B9AC70",
    "slate": "#717581",
    "cyan": "#92DADD",
}
PETROFF_10_ORDER = ("blue", "red", "purple", "khaki", "cyan", "brown", "dark orange", "orange", "grey", "slate")

#: Kelly's colours of maximum contrast (K. L. Kelly 1965, "Twenty-two colors of maximum contrast", Color Engineering
#: 3(6): 26-27), as the ISCC-NBS centroid colours they are usually given by; white and black left out, so TWENTY: the
#: largest palette here (6 Nov 2026, asked "it would be good to expand our current palettes"), so up to twenty
#: conditions get a palette colour each instead of generated ones. Its order is the one `palettes.best_order` finds:
#: **37.0** normal and **18.8** red-green over the first three and four, the widest margin over the gates of any order;
#: its closest pair over all twenty, fitted to the light page, is 8.3. Tol bright, Tol light and Tableau 10 were
#: measured the same evening and left out: fitted to the page, none clears the normal-vision gate in any order
#: (23.7, 28.7 and 23.9 at best).
KELLY_20 = {
    "vivid yellow": "#F3C300",
    "strong purple": "#875692",
    "vivid orange": "#F38400",
    "very light blue": "#A1CAF1",
    "vivid red": "#BE0032",
    "greyish yellow": "#C2B280",
    "medium grey": "#848482",
    "vivid green": "#008856",
    "strong purplish pink": "#E68FAC",
    "strong blue": "#0067A5",
    "strong yellowish pink": "#F99379",
    "strong violet": "#604E97",
    "vivid orange yellow": "#F6A600",
    "strong purplish red": "#B3446C",
    "vivid greenish yellow": "#DCD300",
    "strong reddish brown": "#882D17",
    "vivid yellowish green": "#8DB600",
    "deep yellowish brown": "#654522",
    "vivid reddish orange": "#E25822",
    "dark olive green": "#2B3D26",
}
KELLY_20_ORDER = ("strong purplish pink", "deep yellowish brown", "vivid yellowish green", "strong blue",
                  "vivid orange", "vivid red", "medium grey", "dark olive green", "vivid green", "strong purple",
                  "very light blue", "vivid yellow", "strong yellowish pink", "strong purplish red",
                  "strong reddish brown", "greyish yellow", "vivid greenish yellow", "strong violet",
                  "vivid reddish orange", "vivid orange yellow")
#: ORCHID 11 (7 Nov 2026, asked with a "Scientific Color Inspiration" sheet of the Nature cover of 13 June 2024: "i
#: want this color scheme: we can add some more colors like this to the mix. like yellowish brownish, reddish etc"):
#: the sheet's six data colours (its seventh, #DDE6F3, is a background) and five muted ones chosen beside them,
#: olive yellow, brick red-brown, teal, peach and brown, each the best of 160 samples of lightness and chroma around
#: its hue by `palettes.best_order`'s margin over the gates. Its order is best_order's: **31.2** normal and **15.4**
#: red-green over the first three and four. Measured, fitted to the light page: the cover's blue, lilac and pink are
#: close to one another for red-green colour-blind readers (ΔE 2.3-4.3 simulated; on the dark page blue / lilac
#: 0.5), and lilac / pink also in normal vision (9.7, the closest pair of all eleven), so none of them is among the
#: first four, which the gates hold, and pink and lilac stand 5th and 11th; past four conditions pink and blue (5th,
#: 6th) are weak for those readers. Declared after Kelly 20, so no default changes
#: (`condition_colours.default_palette` takes the first that fits).
ORCHID_11 = {
    "purple": "#6A5A8F",
    "olive yellow": "#B7B251",
    "brick": "#9E5C43",
    "teal": "#539B9A",
    "pink": "#D08AB6",
    "blue": "#7FA4C8",
    "peach": "#E1C1A6",
    "dark purple": "#3F345B",
    "brown": "#856236",
    "green": "#8FB79B",
    "lilac": "#B59BC8",
}
ORCHID_11_ORDER = ("purple", "olive yellow", "brick", "teal", "pink", "blue", "peach", "dark purple", "brown", "green",
                   "lilac")

#: The largest palette before Kelly 20 was added: a project saved with more conditions than this kept its palette
#: and generated colours past its end, and keeps them (`condition_colours.ConditionColours.from_json`).
LARGEST_BEFORE_KELLY = 10


@dataclass(frozen=True)
class Palette:
    """One named set of series colours, and the order its slots are in.

    **A workspace stores a SLOT, never a hex** (see
    ``appdata.Workspace.condition_slots``), and this is what a slot indexes. Two
    consequences, both deliberate: a condition cannot hold a colour that is not
    in the palette, so the gates below stay true of anything on screen; and
    switching palette re-colours every condition at once instead of leaving a
    workspace half in one scheme and half in another.
    """

    key: str
    label: str
    colors: dict
    order: tuple
    source: str

    def __len__(self) -> int:
        return len(self.order)


#: Every palette a workspace may choose. Adding one here is the whole of adding
#: a palette, but it inherits the bar: an entry is only finished once it has
#: been through the same separation and contrast gates the first two cleared.
PALETTES = {
    p.key: p for p in (
        Palette("okabe-ito", "Okabe-Ito", OKABE_ITO, SERIES_ORDER,
                "Okabe & Ito (2008)"),
        Palette("tol-muted", "Tol muted", TOL_MUTED, TOL_ORDER,
                "Tol, SRON technical note"),
        Palette("tol-vibrant", "Tol vibrant", TOL_VIBRANT, TOL_VIBRANT_ORDER,
                "Tol, SRON technical note"),
        Palette("petroff-8", "Petroff 8", PETROFF_8, PETROFF_8_ORDER,
                "Petroff (2021), arXiv:2107.02270"),
        Palette("petroff-10", "Petroff 10", PETROFF_10, PETROFF_10_ORDER,
                "Petroff (2021), arXiv:2107.02270"),
        Palette("kelly-20", "Kelly 20", KELLY_20, KELLY_20_ORDER,
                "Kelly (1965), Color Engineering 3(6)"),
        Palette("orchid-11", "Orchid 11", ORCHID_11, ORCHID_11_ORDER,
                "Nature cover (2024): six colours and five muted ones measured beside them"),
    )
}

def register_palette(palette: "Palette") -> None:
    """Add (or replace) a palette made by the user (`smtracker.palettes`): every list and lookup then has it, and the
    page-fitted colours are measured afresh."""
    PALETTES[palette.key] = palette
    slots.cache_clear()
    series.cache_clear()


def unregister_palette(key: str) -> None:
    """Take a user's palette out (a built-in one is never taken out)."""
    if str(key).startswith("user-") and key in PALETTES:
        del PALETTES[key]
        slots.cache_clear()
        series.cache_clear()


#: The one a workspace has until somebody chooses otherwise. Okabe-Ito, because
#: it is the palette this field actually reads: it is the reference every
#: colour-vision figure guide names, and a prototype that opened on a second
#: choice would be making a claim it has no reason to make.
DEFAULT_PALETTE = "okabe-ito"


#: What each colour scheme's page is, for the luminance fit.
GROUNDS = {"light": "#ffffff", "dark": "#111111"}

#: The floor every colour must clear against the page it is drawn on. WCAG 2.1
#: 1.4.11 for non-text content — a curve, a ring or a bubble stroke IS the
#: content here, so it is the applicable number rather than the 4.5 : 1 text one.
MIN_CONTRAST = 3.0

#: The floor every PAIR must clear from every other under simulated deuteranopia
#: AND protanopia, at the lengths this design draws.
#:
#: 12 is not a standard — there is none — it is **Okabe-Ito's own worst pair at
#: four colours**, measured. The floor is not there to beat the reference; it is
#: there to catch a future edit that falls below it.
MIN_REDGREEN_DELTA_E = 12.0

#: And the floor for NORMAL vision, which is the one the searched ramp failed
#: while passing everything else. 30 sits under Okabe-Ito's measured 38.4 and
#: well over the 25.1 that was reported as unreadable.
MIN_NORMAL_DELTA_E = 30.0


@functools.lru_cache(maxsize=64)
def fit_to_ground(value: str, ground: str,
                  target: float = MIN_CONTRAST + 0.05) -> str:
    """Walk a colour's LUMINANCE until it clears ``target`` against ``ground``.

    Hue and chroma are held; only L* moves, one step at a time, DOWN on a light
    page and UP on a dark one. A step that falls out of the sRGB gamut is
    skipped rather than clamped — :func:`to_hex` clamps silently, and a clamped
    colour is no longer the hue being preserved.

    Returns the input unchanged when it already clears, which is most of the
    palette on the dark scheme: Okabe-Ito is a fairly light set and ``#111111``
    is a very dark page.
    """
    lum, chroma, hue = _hex_to_hcl(value)
    if contrast(value, ground) >= target:
        return value.lower()
    step = -1.0 if relative_luminance(ground) > 0.5 else 1.0
    best = value.lower()
    for _ in range(140):
        lum += step
        if not 0.0 <= lum <= 100.0:
            break
        candidate = to_hex(lum, chroma, hue)
        if abs(_hex_to_hcl(candidate)[0] - lum) > 2.5:
            continue                    # gamut-clipped: not this hue any more
        best = candidate
        if contrast(candidate, ground) >= target:
            return candidate
    return best


@functools.lru_cache(maxsize=64)
def slots(palette: str = DEFAULT_PALETTE, *,
          dark: bool = False) -> tuple[str, ...]:
    """The WHOLE palette, fitted to the page, in its declared order.

    **This is what a stored slot number means, and it is deliberately not
    ``series`` below.** ``series`` sorts by lightness, which is right for "give
    me n colours to draw with" and wrong for a named position: ``series(4)[3]``
    and ``series(5)[3]`` are different colours, because the sort inserts the
    fifth in the middle. A workspace that recorded "malate is slot 3" would then
    have malate change colour when a fourth condition appeared.

    Declared order is stable by construction, so slot 3 is slot 3 whatever else
    is on screen. The first three of Okabe-Ito's declared order are also exactly
    ``series(3)``, so the default assignment reproduces the colours this design
    already shipped rather than quietly renaming them.
    """
    p = PALETTES[palette]
    ground = GROUNDS["dark" if dark else "light"]
    return tuple(fit_to_ground(p.colors[k], ground) for k in p.order)


@functools.lru_cache(maxsize=64)
def series(n: int, *, palette: str = DEFAULT_PALETTE,
           dark: bool = False) -> tuple[str, ...]:
    """``n`` series colours, fitted to the page, ordered by lightness.

    Cached, and that is not premature: ``Tokens.viz`` is a property with no memo
    of its own and it is read inside chart paint code, so an uncached build
    would run the fit on every repaint of every figure.

    Raises past the end of the palette rather than wrapping round — two series
    in one colour is worse than a loud failure.
    """
    if n <= 0:
        return ()
    p = PALETTES[palette]
    if n > len(p.order):
        raise ValueError(
            f"{n} series asked for, {p.label} has {len(p.order)} "
            f"({', '.join(p.order)}); wrapping round would give two "
            f"series the same colour")
    return tuple(sorted(slots(palette, dark=dark)[:n], key=relative_luminance))
