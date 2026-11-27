"""How the populations of "Populations over time" are told apart: one rule for the card and the exported figure.

Headless (no Qt): `ui/workspace/kinetics_chart.py` (the card) and `figure_export` (print, the every-condition figure)
both read `population_styles`, so the two can never disagree (Rule 5).

WHY (7 Nov 2026, the experimenter: "the two populations' colours are too similar to read"). The card mixed the faster
populations only 40 % toward the muted ink and the figure used `figstyle.day_shade`'s 30 % darker / 40 % lighter: slow
and fast were one hue at nearly one lightness. Two channels now, so neither has to carry the whole job:

* LIGHTNESS (and a little chroma): the slowest population keeps the CONDITION'S colour (the one every section means
  by that condition); each faster one is the colour's own hue at the lightness and chroma share, among those inside
  sRGB with at least `MIN_CONTRAST` against the ground, that makes the SMALLEST distance between any two populations
  (CIEDE2000, normal vision and simulated deuteranopia and protanopia) largest. Searched, not stepped, because a
  fixed step is no use to a saturated mid-tone colour that has no room on either side.
* SHAPE: the model line's dash pattern and the marker of the per-bin fits and of each replicate's fit: slowest solid
  with circles, then dash-dot with squares, then dotted with triangles. The REFERENCE curve is a plain long dash
  (`REFERENCE_DASH`) in the NEUTRAL grey, which no population is (8 Nov 2026: in the population's ink its dash read as
  the fast population's dash-dot); bands stay shaded, the induction rule stays a dashed muted line.

MEASURED (CIEDE2000, `palettes.lab`, normal vision and simulated deuteranopia / protanopia) in
`tests/test_population_style.py`; the numbers are in changelog.md.
"""
from __future__ import annotations

import functools
import itertools

__all__ = ["MIN_CONTRAST", "REFERENCE_DASH", "SHAPES", "GROUND_LIGHT", "GROUND_DARK", "population_styles",
           "population_inks", "separation"]

#: The inks searched: lightness (CIELUV L*) steps, and the share of the colour's chroma kept.
_LIGHTNESS = tuple(float(v) for v in range(14, 97, 3))
_CHROMA = (1.0, 0.6, 0.3)
#: An ink must keep this contrast with its ground (WCAG's 3:1 for graphics is the aim, 2.2 the floor for the pale ones,
#: whose line style and marker do the rest of the telling-apart).
MIN_CONTRAST = 2.2
#: The cards' grounds (the theme's card colours) and the figures' pages.
GROUND_LIGHT = "#ffffff"
GROUND_DARK = "#18181b"
#: The reference curve's pattern (on / off, in line widths): one long dash, never a population's own, drawn grey.
REFERENCE_DASH = (6.0, 3.0)
#: Per population, slowest first: the model line's pattern (``None`` solid; on / off lengths in line widths) and the
#: marker of the per-bin and per-replicate fits.
SHAPES = (
    {"dash": None, "marker": "circle"},
    {"dash": (5.0, 1.6, 1.0, 1.6), "marker": "square"},
    {"dash": (1.0, 2.0), "marker": "triangle"},
)


def _candidates(colour: str, ground: str) -> list:
    """The inks a faster population may take: the colour's hue at the searched lightnesses and chroma shares, those
    inside sRGB (a clipped colour is another hue) with the contrast `MIN_CONTRAST` against ``ground``."""
    from .ui.theme import hcl

    _lum, chroma, hue = hcl._hex_to_hcl(colour)
    out = []
    for target in _LIGHTNESS:
        for share in _CHROMA:
            candidate = hcl.to_hex(target, chroma * share, hue)
            if abs(hcl._hex_to_hcl(candidate)[0] - target) > 2.5:
                continue
            if hcl.contrast(candidate, ground) >= MIN_CONTRAST and candidate not in out:
                out.append(candidate)
    return out


def separation(inks) -> float:
    """The smallest CIEDE2000 between any two of ``inks`` seen with normal vision, deuteranopia or protanopia."""
    import numpy as np

    from . import palettes

    if len(inks) < 2:
        return float("inf")
    i, j = np.triu_indices(len(inks), 1)
    worst = float("inf")
    for sim in (None, "deut", "prot"):
        labs = palettes.lab(list(inks), sim)
        worst = min(worst, float(palettes.ciede2000(labs[i], labs[j]).min()))
    return worst


@functools.lru_cache(maxsize=512)
def _inks(colour: str, n: int, dark: bool) -> tuple:
    """The slowest keeps ``colour``; the others are the candidates that make the SMALLEST distance between any two
    populations (`separation`) largest, a tie going to the first found."""
    import numpy as np

    from . import palettes

    colour = colour.lower()
    if n <= 1:
        return (colour,)
    pool = [colour] + [c for c in _candidates(colour, GROUND_DARK if dark else GROUND_LIGHT) if c != colour]
    # every pair's distance, the worst over the three ways of seeing, once
    m = len(pool)
    dist = np.full((m, m), np.inf)
    for sim in (None, "deut", "prot"):
        labs = palettes.lab(pool, sim)
        dist = np.minimum(dist, palettes.ciede2000(labs[:, None, :], labs[None, :, :]))
    best, best_score = None, -1.0
    for rest in itertools.combinations(range(1, m), n - 1):
        idx = (0,) + rest
        score = min(dist[a, b] for a, b in itertools.combinations(idx, 2))
        if score > best_score + 1e-9:
            best, best_score = idx, score
    return tuple(pool[i] for i in best)


def population_inks(colour: str, n: int, *, dark: bool = False) -> list:
    """The ``n`` populations' inks, slowest first: the condition's ``colour``, then the same hue at other
    lightnesses (see the module head). ``dark``: drawn on a dark ground."""
    return list(_inks(colour, int(n), bool(dark)))


def population_styles(colour: str, n: int, *, dark: bool = False) -> list:
    """``[{"ink", "dash", "marker"}]`` for ``n`` populations, slowest first (`SHAPES`; three at most are told apart)."""
    if n > len(SHAPES):
        raise ValueError(f"{n} populations: the style tells {len(SHAPES)} apart")
    return [dict(shape, ink=ink) for shape, ink in zip(SHAPES, population_inks(colour, n, dark=dark))]
