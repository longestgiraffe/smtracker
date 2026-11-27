"""The SQD section's Jump distances page: how the 1-, 2- and 3-population fits compare, and
how well each fits -- headless, plain payloads; `ui/workspace/jumps_chart.py` paints them.

Asked for (13 Oct 2026): "the sqd tab needs the jump distance histograms, where u can compare
how different pop fits compare. Also some way to show the goodness of fit that is better
than v2 version".

WHAT v2 SHOWED, read from `SMTracker.m` (`plotCDF`, the "Single step distances" figure):
the ECDF of r² with the fitted curves and their residuals (model minus ECDF), a jump
histogram with the chosen fit and its raw residuals, and an F-TEST between nested models on
the histogram's bins, printed as p-values. That F-test treats every bin of POOLED STEPS as an
independent observation -- steps within a track and tracks within a cell are not -- and its
answer moves with the bin width. It is not carried over (this project's pseudoreplication
rule: no significance test on pooled steps, under any name).

WHAT THIS PAGE SHOWS INSTEAD, each one a different question:
- THE HISTOGRAM against every fit, the EXPECTED COUNT of a bin computed exactly (the model's
  CDF at the bin's two edges, never its density at the centre), with the selected fit's
  components; under it the PEARSON RESIDUALS (observed - expected) / sqrt(expected), on
  which the counting noise alone is about +-1, so a misfit stands out of a visible band.
- THE ECDF RESIDUALS for every fit on one axis, exact and unbinned.
- THE TABLE: BIC from MAXIMUM-LIKELIHOOD fits (the state count's own rule; never from ECDF
  points, a recorded legacy defect), the largest ECDF gap (Kolmogorov distance, DESCRIPTIVE:
  no p-value), and
- CROSS-VALIDATION OVER INDEPENDENT UNITS: fit on most cells (or days), score the exact
  likelihood of the held-out ones. An extra population that fits noise or one cell's
  peculiarity buys nothing on units it never saw; a real one does. Days is the stricter
  test (a cell shares its day with the training cells); every fold's winner is counted.

IN TWO PASSES (13 Oct 2026: "jump distance takes long to load"). The maximum-likelihood fits
and the folds are nearly all of the time -- on the reference's 13 511 jumps 47 s and 16 s
against 3 s for the curves, and a condition of 40 000 jumps takes minutes. So a run first
asks for ``quality=False`` (histogram, curves, residuals, ECDF gap; BIC and held-out marked
pending) for every condition, then for the whole view, which replaces it. Nothing is cut:
the multistart counts are the measured ones (`sqd.MLE_STARTS_BY_N_POP`).

The displayed curves are the SQD section's own estimator (ECDF least squares, step-weighted,
2000 evaluation points), so they are the fits the bubble chart draws; the BIC column's are
maximum-likelihood fits and can differ slightly, which the caption says.
NO STATISTIC BETWEEN CONDITIONS.

WHAT A CARD DRAWS (26 Oct 2026, asked: "if 1 or 2 or 3 or 4 pop selected -> u only see this fit. right next to it
"all" where all curves shown with different colors", and "the histogram color can be same color as the protein in
the bubble"): the bars in the condition's colour, then ONE fit -- its curve, its components, its Pearson and ECDF
residuals -- or SEVERAL, each in its own colour (`FIT_COLOURS`) and v2's line style (their components since
30 Oct 2026 too, below), their Pearson and ECDF residuals as lines in the same colours.

SHOW IS A SET OF FITS (30 Oct 2026, asked: "instead of 1 2 3 all, just make it clickable and the one u click
appears, when u click again it disappears, so u can click 1 2 3 and all appear, then u click 3 again and 3
disappears and 1 and 2 still visible", and "make residuals and ecdf according how many shown of course"). Each
count is a toggle; there is no All pill any more (every count on IS all). A Show value is a tuple of counts,
ascending. `show_choices` names the pills, `default_show` where a card starts (one fit on: BIC's, else two),
`toggle_show` what a click does -- THE LAST FIT ON STAYS ON (a click on it changes nothing: a histogram with no fit
is not a choice) -- and `drawn_fits` what is drawn, the residual plots included (`in_colour`: two or more). Values
stored before (an export remembered per chart, `figure_export.FigureSpec.chosen_fit`) are read by `show_counts` /
`as_shown`: an old count n is ``(n,)``, the old All (`LEGACY_ALL`) every count made. The page's "All conditions"
view and its export draw every condition's histogram so, side by side.

COMPONENTS AND THE ALL CONDITIONS LAYOUT (30 Oct 2026, asked: "its a shame that the component lines are not visible
anymore when i do 1 and 2 pop fit together. maybe we can make this optional, (show component fits or something)",
and "in the all conditions window, option ( ) show in one plot (like it is now). or (show one plot for each
fit/population where 1 pop and 2 pop etc get an own plot. user can choose that conditions or populations go from up
to down or left to right"). A card's Components toggle (ON to start) draws the components of the fits shown:
`components_drawn` says whose -- each fit shown with more than one population (a one-population fit's one
component is the fit itself: no line, no key entry); one fit's in shades of the condition's colour as before,
several fits' in each fit's colour, thinner and dashed. The All conditions view draws one plot per condition (every
shown fit in it, as before) or one plot per FIT (`OVERVIEW_LAYOUTS`), the conditions down the rows or across
(`OVERVIEW_ORDERS`); `overview_tiles` places every tile, for the screen and the export alike, and in a plot per fit
each condition's tiles share one y range (`histogram_top`).

THE KEY'S PLACE, AND THE TILES' SIZE (30 Oct 2026, asked: "do u think we can make these cards more interactable? like
dragging the legend when it blocks the view, or increasing size of these cards in a way if its too small etc?"). A
key starts in the corner of the plot where it covers the least of what is drawn (`legend_corner`, scored by
`legend_overlap`: the share of the key over bars plus, weighted `LEGEND_CURVE_WEIGHT`, the share of its width each
curve runs through it); dragged, it stays where it was put (a PLACE: its rectangle as fractions of the plot box, y
up, `legend_rect_at` re-fits it when its size changes and `clamp_place` keeps it inside), and the export puts its
own key there (`legend_anchor`: the nearer edges held). In the All conditions view a tile can span the grid
(`place_wide`). All of it is display: no number moves.

THE Y AXIS AND THE RANGES (30 Oct 2026, asked: "lets make all optional with textbook (check first) as standard.
rescale whenever u change setting. make y and x axis changeable. make a button scale y axis for all"). CHECKED:
SMTracker 2 (`SMTracker.m`, plotPDF) draws ``histogram(..., 'Normalization', 'probability')`` -- probability per
bin, the fit as the density times the bin width -- every condition on one shared y; Spot-On integrates its model
over each bin "to obtain a normalized distribution". So the standard is PROBABILITY PER BIN (`DEFAULT_Y_MODE`); Counts
(the jumps per bin, as before) and Density (per nm, area 1) are the other two (`Y_MODES`). `in_units` scales the
bars and the curves by ONE divisor (`y_divisor`: the jumps binned, times the bin width for Density), so the picture
is the counts picture rescaled and every comparison in it is unchanged; the Pearson strip stays in counts (a test
statistic) and the ECDF strip is untouched. `y_label` names the unit with the real bin width. A typed range
(`parse_axes`) stays until Auto; a change of unit drops a typed y range (the numbers would mean something else).
"Same y scale" puts every chart of one unit on the largest automatic range among them (`shared_tops`, v2's).

THE BIN IS 10 nm (30 Oct 2026, asked: "switch the jump-distance histogram to a FIXED 10 nm bin ... and the y axis label
names it everywhere"): SMTracker 2's ``BinWidth = 0.01`` µm and Spot-On's default (`BIN_NM`, `histogram_edges`), from 0
to the first bin edge past the 99.5th percentile, so every card, tile and figure reads "Probability (per 10 nm bin)".
Before, 50 bins up to that percentile gave each condition a bin of its own width and a figure of several conditions
"per bin". Display only: the fits are made on the jumps; the expected counts and Pearson residuals per bin follow
the bins.
"""
from __future__ import annotations

import bisect
import math
import numbers

__all__ = ["CV_UNITS", "CV_LABELS", "DEFAULTS", "MAX_POP", "BIN_NM", "histogram_edges", "compute_jump_view",
           "card_caption", "table_rows", "jumps_config", "config_problem", "SETTING_HELP",
           "setting_help", "N_FOLDS", "verdict_line", "READING_NOTES", "LEGACY_ALL", "FIT_COLOURS",
           "FIT_STYLE_WORDS", "show_choices", "default_show", "drawn_fits", "show_words", "show_counts",
           "as_shown", "toggle_show", "in_colour", "components_drawn", "histogram_top", "OVERVIEW_LAYOUTS",
           "OVERVIEW_LAYOUT_LABELS", "OVERVIEW_ORDERS", "OVERVIEW_ORDER_LABELS", "overview_tiles",
           "LEGEND_CORNERS", "LEGEND_CURVE_WEIGHT", "LEGEND_MARGIN", "plot_fractions", "corner_rect",
           "legend_overlap", "legend_corner", "legend_rect_at", "clamp_place", "legend_anchor", "place_wide",
           "Y_MODES", "Y_MODE_LABELS", "DEFAULT_Y_MODE", "as_y_mode", "bin_width", "y_divisor", "y_label",
           "in_units", "parse_axes", "check_axes", "shared_tops", "pearson_span", "ecdf_span", "residual_end",
           "residual_spans", "RESIDUAL_PLOTS",
           "MAX_POP_CHOICES", "FIT_COLOURS_DARK", "FIT_COLOUR_WORDS", "fit_colour"]

MAX_POP = 3
#: The Run's "Fit up to" choice (26 Oct 2026: the Show pills then offer 1-4). Four is OPT-IN, as in the SQD section's
#: Best: on the reference project (13 511 jumps) the full pass took 234 s at four against 53 s at three, and BIC and
#: the held-out cells still preferred three.
MAX_POP_CHOICES = (3, 4)
#: THE HISTOGRAM'S BIN, in nm (30 Oct 2026, asked: "switch the jump-distance histogram to a FIXED 10 nm bin"):
#: SMTracker 2's ``BinWidth = 0.01`` µm (`SMTracker.m`, plotPDF) and Spot-On's default, so "probability per bin" -- the
#: field's textbook y axis -- means one thing on every card, condition and figure. The bins run from 0 up to the first
#: multiple of it at or past the 99.5th percentile of the jump length (`histogram_edges`). Before, 50 equal bins up to
#: that percentile made each condition's bin its own width (7.2 nm here, 10.4 there), and a figure of several
#: conditions could only say "per bin". A display choice: the fits are made on the jumps, never on the bins; the
#: expected counts and the Pearson residuals per bin follow the bins.
BIN_NM = 10.0
#: Folds over cells; days are always left out one at a time.
N_FOLDS = 5
CV_UNITS = ("day", "cell")
CV_LABELS = ("Replicates", "Cells")
DEFAULTS = {"cv_unit": "day", "max_pop": MAX_POP}
#: ECDF residual points (evenly spaced in probability).
ECDF_POINTS = 300
#: What the Show switch's last pill, All, stored before 30 Oct 2026 (a single-choice switch then): read as every count
#: made (`show_counts`). Nothing offers it any more -- every count on is all.
LEGACY_ALL = "all"
#: Each fit's colour when all are drawn, with v2's line style besides, so a grey print still tells them apart. One
#: fit alone is drawn in the ink, its components in the condition's colour.
#: NOT FROM ANY CONDITION PALETTE (26 Oct 2026, asked: "the blue for histogram and 2 pop fit is too similar. choose
#: colors that are not in TOL or okabe ito for fit of histogram"): the bars wear the condition's Okabe-Ito or Tol
#: colour, so a fit in the same palette could match its own bars (Okabe-Ito's blue 2-population line on the blue
#: condition's bars). CHOSEN BY MEASUREMENT (CIELAB): four families (slate, maroon, olive green, orchid; no blue,
#: chromatic hues at least 50 degrees apart), each at least 20 dE from every Okabe-Ito and Paul Tol colour (bright,
#: vibrant, muted, light, high- and medium-contrast), at least 42 dE from any condition's pale bar, and at least 57
#: apart from each other -- 29-30 under simulated deuteranopia and protanopia (Machado 2009).
FIT_COLOURS = {1: "#465258", 2: "#6E070E", 3: "#5C830B", 4: "#AD36BD"}
#: The same hues lightened for a dark ground (the dark theme, a dark slide): at least 40 dE apart.
FIT_COLOURS_DARK = {1: "#9BA8AF", 2: "#E29185", 3: "#97BD4B", 4: "#EE75FC"}
#: Their names, for legends in words.
FIT_COLOUR_WORDS = {1: "slate grey", 2: "maroon", 3: "olive green", 4: "orchid"}


def fit_colour(k: int, dark: bool = False) -> str:
    """Fit ``k``'s colour under All, on a light or a dark ground."""
    return (FIT_COLOURS_DARK if dark else FIT_COLOURS).get(int(k), "#808080")
#: v2's line styles (`plotCDF`: ':', '-', '--'), a fourth population dash-dot.
FIT_STYLE_WORDS = {1: "dotted", 2: "solid", 3: "dashed", 4: "dash-dot"}


def _counts(fits) -> list:
    return sorted(int(k) for k in fits)


def show_choices(fits) -> tuple:
    """The Show pills, ``((1, "1"), (2, "2"), ...)``: the counts that were fitted, each a toggle of its own (no All:
    every count on is all)."""
    return tuple((k, str(k)) for k in _counts(fits))


def default_show(result: dict) -> tuple:
    """Where a card's Show starts: ONE fit on -- the count BIC prefers, else two (the manuscript's), else the fewest
    fitted."""
    made = _counts(result.get("fits") or {})
    for k in (result.get("bic_best"), 2):
        if k is not None and int(k) in made:
            return (int(k),)
    return (made[0],) if made else (2,)


def show_counts(show, fits=None) -> tuple:
    """The counts a Show value NAMES, ascending, fitted or not: a set, list or tuple of counts (today's), an old
    single count ``n`` -> ``(n,)``, the old All (`LEGACY_ALL`) -> every count in ``fits``; None -> ``()``."""
    if show is None:
        return ()
    if isinstance(show, str):
        if show == LEGACY_ALL:
            return tuple(_counts(fits or {}))
        raise ValueError(f"a Show value is population counts, not {show!r}")
    if isinstance(show, numbers.Number):
        return (int(show),)
    return tuple(sorted({int(k) for k in show}))


def as_shown(show, fits) -> tuple:
    """`show_counts` restricted to the counts that were fitted: what of ``show`` a card with these ``fits`` can
    draw (possibly nothing; the caller decides what that means)."""
    made = set(_counts(fits))
    return tuple(k for k in show_counts(show, fits) if k in made)


def toggle_show(show, k, fits) -> tuple:
    """Show after a click on count ``k``: on if it was off, off if it was on -- except the LAST ONE ON, which stays
    on (a click on it changes nothing: a histogram with no fit drawn is not a choice). A count not fitted is not
    turned on."""
    on = set(as_shown(show, fits))
    k = int(k)
    if k in on:
        if len(on) > 1:
            on.discard(k)
    elif k in set(_counts(fits)):
        on.add(k)
    return tuple(sorted(on))


def drawn_fits(fits, show) -> list:
    """The counts drawn for ``show`` (any Show value), ascending: those of it that were fitted. When none of them
    was, the one nearest the first named, never nothing. The histogram's curves and both residual plots draw
    exactly these."""
    made = _counts(fits)
    if not made:
        return []
    on = list(as_shown(show, fits))
    if on:
        return on
    named = show_counts(show, fits)
    k = named[0] if named else 2
    return [min(made, key=lambda m: (abs(m - k), m))]


def in_colour(drawn) -> bool:
    """Whether the fits ``drawn`` are told apart by colour: two or more, each in its own colour and v2's line style,
    without components. One fit alone is drawn in the ink with its components."""
    return len(list(drawn)) > 1


def show_words(show) -> str:
    """``"2 populations"``, ``"1 and 3 populations"``, ``"1, 2 and 3 populations"`` (any Show value; the old All
    reads ``"every fit"``)."""
    if isinstance(show, str) and show == LEGACY_ALL:
        return "every fit"
    counts = show_counts(show)
    if not counts:
        return "no fit"
    if len(counts) == 1:
        return f"{counts[0]} population{'' if counts[0] == 1 else 's'}"
    return f"{', '.join(str(k) for k in counts[:-1])} and {counts[-1]} populations"


# --------------------------------------------------------------------------- #
# The y axis's unit, and the ranges a user types
# --------------------------------------------------------------------------- #

#: The y axis's units: probability per bin (SMTracker 2's and Spot-On's, the standard), the jumps per bin, and the
#: probability density per nm (area 1).
Y_MODES = ("probability", "counts", "density")
Y_MODE_LABELS = ("Probability", "Counts", "Density")
DEFAULT_Y_MODE = "probability"


def as_y_mode(value) -> str:
    """A y unit: ``value`` when it is one (`Y_MODES`); None -- a chart, an export or a session from before the choice
    existed -- is the standard, `DEFAULT_Y_MODE`. Anything else is refused by name."""
    if value is None:
        return DEFAULT_Y_MODE
    if value not in Y_MODES:
        raise ValueError(f"the y axis is one of {', '.join(Y_MODES)}, not {value!r}")
    return value


def histogram_edges(top_nm: float) -> list:
    """The jump histogram's bin edges in nm: `BIN_NM` wide from 0 up to the first multiple of it at or past
    ``top_nm`` (the 99.5th percentile of the jump length), at least one bin."""
    n = max(1, math.ceil(round(float(top_nm) / BIN_NM, 9)))
    return [BIN_NM * i for i in range(n + 1)]


def bin_width(edges) -> float:
    """The histogram's bin width in nm (its bins are equal: `BIN_NM` since 30 Oct 2026; a result made before has 50
    bins up to the 99.5th percentile, its own width)."""
    return (float(edges[-1]) - float(edges[0])) / max(1, len(edges) - 1)


def y_divisor(mode: str, observed, edges) -> float:
    """What the counts are divided by in ``mode``: 1 (Counts), the jumps binned (Probability: the bars sum to 1), or
    the jumps binned times the bin width in nm (Density: the bars' area is 1). The histogram stops at the first bin
    edge past the 99.5th percentile, so "binned" is at least 99.5 % of the condition's jumps."""
    mode = as_y_mode(mode)
    if mode == "counts":
        return 1.0
    n = float(sum(observed)) or 1.0
    return n if mode == "probability" else n * bin_width(edges)


def y_label(mode: str, widths=()) -> str:
    """The y axis's title: "Probability (per 10 nm bin)" -- the bin width when every histogram shares one (`BIN_NM`
    for every result made since 30 Oct 2026, and when no histogram is given), else "per bin" (results made before,
    each its own width) --, "Jumps per bin", or "Probability density (1/nm)"."""
    from .chart_text import JUMPS_COUNT_LABEL

    mode = as_y_mode(mode)
    if mode == "counts":
        return JUMPS_COUNT_LABEL
    if mode == "density":
        return "Probability density (1/nm)"
    distinct = sorted({round(float(w), 1) for w in widths}) or [BIN_NM]
    return f"Probability (per {distinct[0]:g} nm bin)" if len(distinct) == 1 else "Probability (per bin)"


def in_units(mode: str, observed, fits, edges) -> tuple:
    """``(bars, fits)`` in ``mode``: the bars and every fit's expected counts and components divided by ONE divisor
    (`y_divisor`), so the fits sit on the bars exactly as in counts. A fit's other numbers (Pearson, ECDF residuals,
    D, fractions) are left as they are."""
    d = y_divisor(mode, observed, edges)
    bars = [float(c) / d for c in observed]
    scaled = {}
    for k, fit in fits.items():
        out = dict(fit)
        if "expected" in fit:
            out["expected"] = [float(v) / d for v in fit["expected"]]
        if "components" in fit:
            out["components"] = [[float(v) / d for v in comp] for comp in fit["components"]]
        scaled[k] = out
    return bars, scaled


def check_axes(axes) -> dict:
    """``{"x": (from, to) or None, "y": ...}`` checked by the one rule every chart's Axes uses (`axes_view.check`):
    an end may be None (automatic); from below to; neither axis below 0 (a jump distance, a count, a probability and
    a density are never negative). Refused by name."""
    from .axes_view import axis_rows, check

    return check(axis_rows("jumps"), axes)


def parse_axes(x_from="", x_to="", y_from="", y_to="") -> dict:
    """The Axes popover's four fields as ranges (`check_axes`; `axes_view.parse`): empty is automatic, "0,5" reads
    as 0.5."""
    from .axes_view import axis_rows, parse

    return parse(axis_rows("jumps"), {("x", "from"): x_from, ("x", "to"): x_to,
                                      ("y", "from"): y_from, ("y", "to"): y_to})


def shared_tops(tops: dict) -> dict:
    """"Same y scale": ``tops`` ``{chart: (unit, its automatic top)}`` -> ``{chart: the largest top among the charts
    of its unit}`` (v2 put every condition on one y; charts in different units are never mixed)."""
    best = {}
    for unit, top in tops.values():
        best[unit] = max(best.get(unit, 0.0), float(top))
    return {chart: best[unit] for chart, (unit, _top) in tops.items()}


def components_drawn(fits, drawn, on: bool = True) -> list:
    """The fits whose COMPONENTS are drawn, ascending: none with the Components toggle off; otherwise each fit drawn
    -- one alone (as before 30 Oct 2026) or several -- that has more than one population. A one-population fit's one
    component IS the fit: drawn again it would only thicken that line, and a key naming it would name a line that is
    not there, so it has none (one fit of one population draws no component and its key says none)."""
    if not on:
        return []
    return [int(k) for k in drawn if len((fits.get(int(k)) or {}).get("components") or ()) > 1]


def histogram_top(observed, fits, counts) -> float:
    """The highest a histogram of ``observed`` jumps with the fits ``counts`` reaches (bars and expected counts, in
    whatever unit they are given: `in_units`): its y axis's end before the margin; 1 when nothing is above 0. One
    value per condition is what the plots of one condition share."""
    top = float(max([*observed, *(v for k in counts for v in fits[k]["expected"])], default=0.0))
    return top if top > 0 else 1.0


#: THE RESIDUAL PLOTS, in the order they stand under a histogram: the Pearson residual per bin, then the ECDF
#: residual. A card draws both; the All conditions view each on its own box, off to start (12 Nov 2026).
RESIDUAL_PLOTS = ("pearson", "ecdf")


def pearson_span(fits, drawn) -> float:
    """How far a Pearson residual plot of the fits ``drawn`` reaches either side of 0: 1.1 times the largest, at least
    3 so the +-2 band shows. The screen's plots and the exported figures read this one rule."""
    return max(3.0, max((abs(v) for k in drawn for v in fits[k]["pearson"]), default=0.0) * 1.1)


def ecdf_span(fits, drawn) -> float:
    """How far an ECDF residual plot of the fits ``drawn`` reaches either side of 0: 1.1 times the largest, at least
    0.005."""
    return max(0.005, max((abs(v) for k in drawn for _x, v in fits[k]["ecdf_residual"]), default=0.0) * 1.1)


def residual_end(fits, drawn) -> float:
    """The longest jump an ECDF residual plot of the fits ``drawn`` reaches, nm (its points run to the 99.8th
    percentile, a little past the histogram's last bin); 0 without a point."""
    return max((float(x) for k in drawn for x, _v in fits[k]["ecdf_residual"]), default=0.0)


def residual_spans(charts) -> dict:
    """"Same y scale" over residual plots: ``charts`` ``[(fits, drawn)]`` -> ``{"pearson": span, "ecdf": span}``,
    the largest of each kind among them (`pearson_span`, `ecdf_span`), so every plot of a kind reads on one range."""
    charts = list(charts)
    return {"pearson": max((pearson_span(f, d) for f, d in charts), default=3.0),
            "ecdf": max((ecdf_span(f, d) for f, d in charts), default=0.005)}


#: The All conditions view's layouts: one plot per condition (every fit its card shows in it), or one plot per fit.
OVERVIEW_LAYOUTS = ("condition", "fit")
OVERVIEW_LAYOUT_LABELS = ("One plot per condition", "One plot per fit")
#: In one plot per fit: the conditions down the rows (the fits across), or across the columns (the fits down).
OVERVIEW_ORDERS = ("down", "across")
OVERVIEW_ORDER_LABELS = ("Conditions down", "Conditions across")


def overview_tiles(cards, layout: str = "condition", order: str = "down") -> dict:
    """Where every plot of the All conditions view goes -- the screen's tiles and the export's panels, one rule.

    ``cards``: ``[(key, show, fits), ...]`` in the page's order (each condition's Show and its fits made). Returns
    ``{"layout", "order", "rows", "columns", "counts", "tiles": [{"key", "show", "row", "column"}, ...]}``:

    * ``"condition"`` (one plot per condition, as before): a tile per condition with every fit it shows
      (`drawn_fits`); ``rows`` / ``columns`` / ``row`` / ``column`` None -- the screen flows them, the export balances
      them across its width;
    * ``"fit"`` (one plot per fit): a tile per condition AND fit it shows, that fit alone, on a grid: the conditions
      down the rows and the fits (``counts``, every count some condition shows) across the columns (``"down"``), or
      the other way (``"across"``). A fit a condition does not show leaves its cell empty.

    An unknown layout or order is refused by name."""
    if layout not in OVERVIEW_LAYOUTS:
        raise ValueError(f"layout must be one of {OVERVIEW_LAYOUTS}, got {layout!r}")
    if order not in OVERVIEW_ORDERS:
        raise ValueError(f"order must be one of {OVERVIEW_ORDERS}, got {order!r}")
    drawn = [(key, tuple(drawn_fits(fits, show))) for key, show, fits in cards]
    counts = sorted({k for _key, shown in drawn for k in shown})
    if layout == "condition":
        tiles = [{"key": key, "show": shown, "row": None, "column": None} for key, shown in drawn if shown]
        return {"layout": layout, "order": order, "rows": None, "columns": None, "counts": counts, "tiles": tiles}
    tiles = []
    for c, (key, shown) in enumerate(drawn):
        for k in shown:
            f = counts.index(k)
            row, column = (c, f) if order == "down" else (f, c)
            tiles.append({"key": key, "show": (k,), "row": row, "column": column})
    conditions = len(drawn)
    rows, columns = (conditions, len(counts)) if order == "down" else (len(counts), conditions)
    return {"layout": layout, "order": order, "rows": rows, "columns": columns, "counts": counts, "tiles": tiles}


# --------------------------------------------------------------------------- #
# Where the key goes (fractions of the plot box, x right and y UP, as matplotlib's axes fractions)
# --------------------------------------------------------------------------- #

#: The four places a key starts in, in the order a tie goes (matplotlib's "best" tries upper right first too).
LEGEND_CORNERS = ("upper right", "upper left", "lower right", "lower left")
#: What a curve under the key counts against bars under it: a curve crossing the key's whole width counts 1.5 times
#: a key wholly over bars -- the curves are what a card compares, the bars are many and read around it.
LEGEND_CURVE_WEIGHT = 1.5
#: The key's inset from the plot box's edges, a fraction of the box.
LEGEND_MARGIN = 0.02
#: Sampling the key's rectangle: points across and up for the bars, points across for each curve.
_BAR_SAMPLES = (20, 10)
_CURVE_SAMPLES = 40


def plot_fractions(edges, observed, series, x_lo: float, x_hi: float, y_lo: float, y_hi: float) -> tuple:
    """``(curves, bars)`` in fractions of the plot box: each of ``series`` (values at the bins' centres) as
    ``[(x, y), ...]``, and each bar as ``(x0, x1, top)`` -- what `legend_overlap` measures a key against."""
    def fx(x):
        return (x - x_lo) / (x_hi - x_lo) if x_hi > x_lo else 0.0

    def fy(v):
        return (v - y_lo) / (y_hi - y_lo) if y_hi > y_lo else 0.0

    centres = [(a + b) / 2.0 for a, b in zip(edges, edges[1:])]
    curves = [[(fx(x), fy(v)) for x, v in zip(centres, values)] for values in series]
    bars = [(fx(a), fx(b), fy(c)) for a, b, c in zip(edges, edges[1:], observed)]
    return curves, bars


def corner_rect(corner: str, size, margin: float = LEGEND_MARGIN) -> tuple:
    """The key's rectangle ``(x0, y0, x1, y1)`` in ``corner`` (`LEGEND_CORNERS`), ``size`` ``(w, h)`` fractions."""
    if corner not in LEGEND_CORNERS:
        raise ValueError(f"corner must be one of {LEGEND_CORNERS}, got {corner!r}")
    w, h = float(size[0]), float(size[1])
    x0 = margin if "left" in corner else 1.0 - margin - w
    y0 = margin if "lower" in corner else 1.0 - margin - h
    return (x0, y0, x0 + w, y0 + h)


def _bar_top(bars, x: float) -> float:
    starts = [b[0] for b in bars]
    i = bisect.bisect_right(starts, x) - 1
    if 0 <= i < len(bars) and bars[i][0] <= x <= bars[i][1]:
        return bars[i][2]
    return 0.0


def _curve_at(curve, x: float):
    xs = [pt[0] for pt in curve]
    if not curve or x < xs[0] or x > xs[-1]:
        return None
    i = bisect.bisect_left(xs, x)
    if i == 0:
        return curve[0][1]
    (xa, ya), (xb, yb) = curve[i - 1], curve[i]
    return ya if xb == xa else ya + (yb - ya) * (x - xa) / (xb - xa)


def legend_overlap(rect, curves, bars) -> float:
    """How much of the drawing a key at ``rect`` covers: the share of its area over bars (0-1, sampled on a grid),
    plus `LEGEND_CURVE_WEIGHT` times, for each curve, the share of its width the curve runs through it."""
    x0, y0, x1, y1 = rect
    nx, ny = _BAR_SAMPLES
    under = 0
    for i in range(nx):
        x = x0 + (i + 0.5) / nx * (x1 - x0)
        top = _bar_top(bars, x)
        under += sum(1 for j in range(ny) if y0 + (j + 0.5) / ny * (y1 - y0) < top)
    crossing = 0.0
    for curve in curves:
        inside = 0
        for i in range(_CURVE_SAMPLES):
            y = _curve_at(curve, x0 + (i + 0.5) / _CURVE_SAMPLES * (x1 - x0))
            if y is not None and y0 <= y <= y1:
                inside += 1
        crossing += inside / _CURVE_SAMPLES
    return under / (nx * ny) + LEGEND_CURVE_WEIGHT * crossing


def legend_corner(curves, bars, size, margin: float = LEGEND_MARGIN) -> str:
    """The corner (`LEGEND_CORNERS`) where a key of ``size`` covers the least (`legend_overlap`); a tie goes to the
    earlier corner."""
    scores = {c: legend_overlap(corner_rect(c, size, margin), curves, bars) for c in LEGEND_CORNERS}
    return min(LEGEND_CORNERS, key=lambda c: (round(scores[c], 9), LEGEND_CORNERS.index(c)))


def clamp_place(rect) -> tuple:
    """``rect`` moved (never resized) so it lies inside the plot box; a key wider or taller than the box sits at
    its left or bottom edge."""
    x0, y0, x1, y1 = (float(v) for v in rect)
    w, h = x1 - x0, y1 - y0
    x0 = min(max(0.0, x0), max(0.0, 1.0 - w))
    y0 = min(max(0.0, y0), max(0.0, 1.0 - h))
    return (x0, y0, x0 + w, y0 + h)


def legend_rect_at(place, size) -> tuple:
    """A key of ``size`` ``(w, h)`` at a stored ``place`` (the rectangle it was dragged to): the edges nearer the
    plot's sides held -- the left edge for a key in the left half, the right edge in the right half, the top edge
    in the upper half, the bottom in the lower -- so a key that grows or shrinks (another width, other fits) stays
    by the side it was put at; then inside the box (`clamp_place`)."""
    x0, y0, x1, y1 = (float(v) for v in place)
    w, h = float(size[0]), float(size[1])
    nx0 = x0 if (x0 + x1) / 2.0 < 0.5 else x1 - w
    ny0 = y0 if (y0 + y1) / 2.0 < 0.5 else y1 - h
    return clamp_place((nx0, ny0, nx0 + w, ny0 + h))


def legend_anchor(place) -> tuple:
    """``(loc, (x, y))`` for matplotlib (``legend(loc=..., bbox_to_anchor=...)`` in axes fractions): the stored
    ``place``'s nearer edges, as `legend_rect_at` holds them -- so the export's key, of another size, sits where the
    card's was put."""
    x0, y0, x1, y1 = (float(v) for v in place)
    left, lower = (x0 + x1) / 2.0 < 0.5, (y0 + y1) / 2.0 < 0.5
    return (f"{'lower' if lower else 'upper'} {'left' if left else 'right'}",
            (x0 if left else x1, y0 if lower else y1))


def place_wide(tiles, wide, columns: int) -> list:
    """One plot per fit's grid with some tiles spanning it: ``tiles`` ``[(key, row, column), ...]``, ``wide`` the
    keys to span; returns ``[(key, row, column, row span, column span), ...]`` -- each original row's other tiles in
    a row, then each wide tile of that row in a row of its own across every column."""
    wide = set(wide)
    out, row_out = [], 0
    for row in sorted({r for _k, r, _c in tiles}):
        normal = [(k, c) for k, r, c in tiles if r == row and k not in wide]
        if normal:
            out += [(k, row_out, c, 1, 1) for k, c in normal]
            row_out += 1
        for k, r, _c in tiles:
            if r == row and k in wide:
                out.append((k, row_out, 0, 1, max(1, int(columns))))
                row_out += 1
    return out


def _fits(sd, max_pop: int, *, mle_too: bool = True, calibration=None):
    from .sqd import SQDError, fit_sqd

    shown, mle = {}, {}
    if calibration is not None:
        # PER SPOT (6 Nov 2026): one maximum-likelihood fit per count, each jump its own localisation error -- the
        # curves and the BIC alike (no one ECDF curve to fit), as the section's Run makes them
        for n in range(1, max_pop + 1):
            try:
                shown[n] = fit_sqd(sd, n_pop=n, method="mle", precision=calibration)
            except SQDError:
                continue
        return shown, (dict(shown) if mle_too else {})
    for n in range(1, max_pop + 1):
        try:
            shown[n] = fit_sqd(sd, n_pop=n, ecdf_max_points=2000)
        except SQDError:
            continue
        if not mle_too:
            continue
        try:
            mle[n] = fit_sqd(sd, n_pop=n, method="mle")
        except SQDError:
            pass
    return shown, mle


def _noise_groups(noise):
    """`precision.noise_groups`: the one grouping of the jumps' noise."""
    from .precision import noise_groups

    return noise_groups(noise)


def _per_spot_cdf(x, D, fractions, tau, groups):
    """P(r^2 <= x) under a per-spot fit over the jumps it was fitted to: population k's mean for a jump of noise s is
    4 D_k tau + 2 s (`sqd._means`), averaged over the jumps' noise (`precision.noise_groups`)."""
    import numpy as np

    noise, weights = groups
    x = np.asarray(x, dtype=float)[:, None]
    total = np.zeros(x.shape[0])
    for d, f in zip(D, fractions):
        m = 4.0 * float(d) * tau + 2.0 * noise[None, :]
        total += float(f) * ((1.0 - np.exp(-x / m)) @ weights)
    return total


def _ks_distance(sorted_r2, fit, groups=None) -> float:
    import numpy as np

    from .sqd import sqd_cdf

    n = len(sorted_r2)
    if groups is not None:
        F = _per_spot_cdf(sorted_r2, fit.D, fit.fractions, fit.tau_lag_s, groups)
        i = np.arange(1, n + 1)
        return float(max(np.max(i / n - F), np.max(F - (i - 1) / n)))
    if not getattr(fit, "immobile", False):
        F = sqd_cdf(sorted_r2, fit.D, fit.fractions, fit.tau_lag_s,
                    localization_sigma_um=fit.localization_sigma_um)
        i = np.arange(1, n + 1)
        return float(max(np.max(i / n - F), np.max(F - (i - 1) / n)))
    # THE IMMOBILE POPULATION IS A STEP AT 0 (exact-zero jumps, `sqd`'s module head): the ECDF above the model's
    # value AT a jump (`inclusive`, <=), below it just under the jump (strict, <), so the zeros' step is met by the
    # model's own and is no gap. The moving populations are continuous: there the two sides are one.
    F_at = sqd_cdf(sorted_r2, fit.D, fit.fractions, fit.tau_lag_s,
                   localization_sigma_um=fit.localization_sigma_um, immobile=True, inclusive=True)
    F_below = sqd_cdf(sorted_r2, fit.D, fit.fractions, fit.tau_lag_s,
                      localization_sigma_um=fit.localization_sigma_um, immobile=True)
    last = np.searchsorted(sorted_r2, sorted_r2, side="right")       # the ECDF at each value, ties included
    first = np.searchsorted(sorted_r2, sorted_r2, side="left")       # and just under it
    return float(max(np.max(last / n - F_at), np.max(F_below - first / n)))


def _cross_validate(sd, unit: str, max_pop: int, zero_rule: str = "auto", calibration=None) -> dict:
    """Held-out log-likelihood per state count, over folds of independent units. ``zero_rule`` is the whole
    condition's rule for exact-zero jumps (`sqd.zero_rule_of`), so every fold's fit holds its populations.
    ``calibration``: per-spot precision, every fold fitted and scored with each jump's own noise."""
    import numpy as np

    from .sqd import SQDError, fit_sqd, log_likelihood_of

    if unit not in sd.columns:
        return {"unit": unit, "note": f"the steps carry no {unit} column"}
    units = sorted(str(u) for u in sd[unit].dropna().unique())
    if unit == "day":
        if len(units) < 2:
            return {"unit": unit, "note": (
                f"{len(units)} replicate: nothing to hold out. Choose Cells for folds "
                f"over this session's cells.")}
        folds = [[u] for u in units]
    else:
        if len(units) < 4:
            return {"unit": unit, "note": f"{len(units)} cells: too few to fold."}
        order = list(units)
        np.random.default_rng(0).shuffle(order)
        k = min(N_FOLDS, len(order))
        folds = [order[i::k] for i in range(k)]
    keys = sd[unit].astype(str).to_numpy()
    r2 = sd["sq_disp_um2"].to_numpy(float)
    noise = None
    if calibration is not None:
        from .precision import step_noise

        noise = step_noise(sd, calibration)
    per_fold = []
    for fold in folds:
        test = np.isin(keys, fold)
        if test.sum() == 0 or (~test).sum() == 0:
            continue
        train = sd[~test]
        scores = {}
        for n in range(1, max_pop + 1):
            try:
                fit = (fit_sqd(train, n_pop=n, method="mle", precision=calibration) if calibration is not None
                       else fit_sqd(train, n_pop=n, ecdf_max_points=2000, zero_jumps=zero_rule))
            except SQDError:
                continue
            scores[n] = log_likelihood_of(r2[test], fit, step_noise_um2=None if noise is None else noise[test])
        if scores:
            per_fold.append({"units": fold, "n_test": int(test.sum()), "scores": scores})
    if not per_fold:
        return {"unit": unit, "note": "no fold could be fitted"}
    pops = sorted(set.intersection(*(set(f["scores"]) for f in per_fold)))
    total = {n: sum(f["scores"][n] for f in per_fold) for n in pops}
    n_test = sum(f["n_test"] for f in per_fold)
    best = max(total.values())
    wins = {n: 0 for n in pops}
    for f in per_fold:
        wins[max(pops, key=lambda n: f["scores"][n])] += 1
    return {"unit": unit, "n_folds": len(per_fold), "n_units": len(units),
            "n_test": n_test,
            "delta_total": {n: float(total[n] - best) for n in pops},
            "delta_per_step": {n: float((total[n] - best) / n_test) for n in pops},
            "wins": wins, "best": max(pops, key=lambda n: total[n])}


def compute_jump_view(project, condition: str, *, cv_unit: str, max_pop: int = MAX_POP,
                      quality: bool = True, gradient: str = "numeric", lag: int = 1, precision=None) -> dict:
    """One condition's jump distances against every fit, and each fit's quality.

    ``quality=False`` is the quick first pass: no maximum-likelihood fits and no folds, and
    the result says so (``"pending": True``; BIC and held-out read as computing).

    ``gradient`` is the SQD Compute dialog's **Optimiser** (`sqd.GRADIENTS`), so every fit
    on this page, the folds included, is made the way the section's are: "numeric"
    (Standard) or "analytic" (Fast, the same fits about twofold quicker at 3 populations,
    measured 15 Oct 2026; this page's slowness is what it was built for). A Fast result
    says so in its settings.

    ``lag`` is the dialog's **Time lag** in frames (`comparison.SQD_LAGS`): the jumps are then that many frames
    long, as the section's fits are; a lag other than 1 is in the settings.

    ``precision`` is the SQD Run's per-spot calibration ``(c_nm, p, source)`` for this condition (Localisation error
    Per spot; None for a condition whose tracks carry none): every fit on the page is then the per-spot maximum
    likelihood, the curves each jump's own model summed over the jumps, the folds scored with each jump's noise; the
    steps lacking a spot's precision are left out, as the section's fits leave them.

    Picklable and run on the process pool. An expected refusal comes back as
    ``{"condition": ..., "error": ...}``; anything else raises."""
    from .sqd import gradient_mode

    with gradient_mode(gradient):
        result = _compute_jump_view(project, condition, cv_unit=cv_unit, max_pop=max_pop,
                                    quality=quality, lag=int(lag), precision=precision)
    if precision is not None and "error" not in result:
        result["settings"] = {**result["settings"], "precision": [float(precision[0]), float(precision[1]),
                                                                   str(precision[2])]}
    if gradient != "numeric":
        result["settings"] = {**result["settings"], "optimiser": gradient}
    if int(lag) != 1:
        result["settings"] = {**result["settings"], "lag": int(lag)}
    return result


def _compute_jump_view(project, condition: str, *, cv_unit: str, max_pop: int,
                       quality: bool, lag: int = 1, precision=None) -> dict:
    """`compute_jump_view` inside the chosen optimiser."""
    import numpy as np

    from .observables import ObservableError, squared_displacements
    from .sqd import SQDError, sqd_cdf, zero_rule_of

    if cv_unit not in CV_UNITS:
        raise ValueError(f"cv_unit must be one of {CV_UNITS}, got {cv_unit!r}")
    settings = {"cv_unit": cv_unit, "max_pop": int(max_pop)}
    calibration, groups = None, None
    try:
        sd = squared_displacements(project.condition(condition), lag_frames=lag)
        if precision is not None:
            from .precision import Calibration, PrecisionError, step_noise, usable_steps

            calibration = Calibration(c_nm=float(precision[0]), p=float(precision[1]), source=str(precision[2]))
            try:
                ok = usable_steps(sd)
                if not ok.all():
                    kept = sd.loc[ok]
                    kept.attrs = dict(sd.attrs)
                    sd = kept
                groups = _noise_groups(step_noise(sd, calibration))
            except PrecisionError as exc:
                raise SQDError(str(exc)) from exc
        shown, mle = _fits(sd, int(max_pop), mle_too=bool(quality), calibration=calibration)
        if not shown:
            raise SQDError("no population count could be fitted")
    except (SQDError, ObservableError) as exc:
        return {"condition": condition, "settings": settings,
                "error": f"{type(exc).__name__}: {exc}"}

    r2 = np.sort(sd["sq_disp_um2"].to_numpy(float))
    n = len(r2)
    r_nm = np.sqrt(r2) * 1000.0
    # FIXED 10 nm BINS (`BIN_NM`) up to the 99.5th percentile; the fits above are on the jumps, not on these
    edges = np.asarray(histogram_edges(float(np.quantile(r_nm, 0.995))))
    observed = np.histogram(r_nm, bins=edges)[0]
    lo2, hi2 = (edges[:-1] / 1000.0) ** 2, (edges[1:] / 1000.0) ** 2

    def cdf(points, D, fractions, fit, still, inclusive=False):
        # per spot: each jump's own model, averaged over the jumps (`_per_spot_cdf`); else the fit's one curve
        if groups is not None:
            return _per_spot_cdf(points, D, fractions, fit.tau_lag_s, groups)
        return sqd_cdf(points, D, fractions, fit.tau_lag_s, immobile=still, inclusive=inclusive)

    fits = {}
    for k, fit in shown.items():
        # the immobile population of exact-zero jumps (`sqd`'s module head) is a step at 0: in the first bin, [0, 10)
        still = bool(getattr(fit, "immobile", False))
        cdf_hi = cdf(hi2, fit.D, fit.fractions, fit, still)
        cdf_lo = cdf(lo2, fit.D, fit.fractions, fit, still)
        expected = n * (cdf_hi - cdf_lo)
        pearson = np.where(expected > 0, (observed - expected) / np.sqrt(np.maximum(expected, 1e-12)),
                           0.0)
        components = []
        for D, f in zip(fit.D, fit.fractions):
            c_hi = cdf(hi2, [D], [1.0], fit, still)
            c_lo = cdf(lo2, [D], [1.0], fit, still)
            components.append([float(v) for v in n * f * (c_hi - c_lo)])
        q = np.linspace(0.5 / ECDF_POINTS, 1 - 0.5 / ECDF_POINTS, ECDF_POINTS)
        x = np.quantile(r2, q)
        ecdf = np.searchsorted(r2, x, side="right") / n
        # on the ECDF's side (<=), so the zeros' step is the model's too
        model = cdf(x, fit.D, fit.fractions, fit, still, inclusive=True)
        fits[k] = {
            "D": [float(d) for d in fit.D],
            "fractions": [float(f) for f in fit.fractions],
            "expected": [float(v) for v in expected],
            "components": components,
            "pearson": [float(v) for v in pearson],
            "ecdf_residual": [(float(math.sqrt(xx) * 1000.0), float(m - e))
                              for xx, m, e in zip(x, model, ecdf)],
            "ks": _ks_distance(r2, fit, groups),
            "bic": float(mle[k].bic) if k in mle else None,
            "bic_D": [float(d) for d in mle[k].D] if k in mle else None,
            "bic_fractions": [float(f) for f in mle[k].fractions] if k in mle else None,
        }
    bics = {k: v["bic"] for k, v in fits.items() if v["bic"] is not None}
    best_bic = min(bics.values()) if bics else None
    for k, v in fits.items():
        v["delta_bic"] = None if v["bic"] is None else float(v["bic"] - best_bic)
    result = {
        "condition": condition,
        "settings": settings,
        "n_steps": int(n),
        "n_tracks": int(sd["uid"].nunique()),
        "tau_s": float(next(iter(shown.values())).tau_lag_s),
        "edges_nm": [float(e) for e in edges],
        "observed": [int(c) for c in observed],
        "fits": fits,
        "bic_best": None if not bics else min(bics, key=bics.get),
        "cv": (_cross_validate(sd, cv_unit, int(max_pop), zero_rule_of(next(iter(shown.values()))),
                               calibration=calibration)
               if quality else {"unit": cv_unit}),
        "pending": not quality,
    }
    zeros = next(iter(shown.values())).zero_jumps
    if zeros is not None:
        result["zero_jumps"] = zeros.to_dict()
    result["table"] = table_rows(result)
    return result


# --------------------------------------------------------------------------- #
# The words
# --------------------------------------------------------------------------- #


def card_caption(result: dict) -> str:
    if (result.get("settings") or {}).get("precision"):
        return (f"{result['n_steps']:,} jumps · {result['n_tracks']:,} tracks · τ "
                f"{1000 * result['tau_s']:.1f} ms · curves and BIC: per-spot maximum-likelihood fits, each jump "
                f"with its own localisation error ({result['settings']['precision'][2]})")
    return (f"{result['n_steps']:,} jumps · {result['n_tracks']:,} tracks · τ "
            f"{1000 * result['tau_s']:.1f} ms · curves: the SQD section's ECDF fit "
            f"(step-weighted); BIC: maximum-likelihood fits")


def table_rows(result: dict) -> dict:
    """The fit-quality table: ``{"columns": [...], "rows": [[...]], "notes": [...]}``."""
    cv = result.get("cv") or {}
    unit = {"day": "replicates", "cell": "cells"}.get(cv.get("unit"), "units")
    columns = ["Populations", "D (µm²/s)", "Fractions", "ΔBIC", "Largest ECDF gap",
               f"Held-out ΔlogL ({unit})", "Folds won"]
    pending = bool(result.get("pending"))
    wait = "computing…"
    rows = []
    for k, fit in sorted(result["fits"].items()):
        held = cv.get("delta_total", {}).get(k)
        rows.append([
            str(k),
            " / ".join(f"{d:#.3g}" for d in fit["D"]),
            " / ".join(f"{100 * f:.1f}%" for f in fit["fractions"]),
            wait if pending else "–" if fit["delta_bic"] is None else f"{fit['delta_bic']:.1f}",
            f"{100 * fit['ks']:.2f} pp",
            wait if pending else "–" if held is None else f"{held:.1f}",
            wait if pending else "–" if not cv.get("wins")
            else f"{cv['wins'].get(k, 0)} of {cv['n_folds']}",
        ])
    notes = []
    if pending:
        notes.append("BIC (maximum-likelihood fits) and the held-out fits are still being "
                     "computed and will fill in here.")
    if result.get("bic_best") is not None:
        notes.append(f"BIC prefers {result['bic_best']} population"
                     f"{'' if result['bic_best'] == 1 else 's'} (lowest; a difference under 2 "
                     f"is inconclusive).")
    if cv.get("note"):
        notes.append(f"Cross-validation: {cv['note']}")
    elif cv.get("n_folds"):
        notes.append(f"Cross-validation over {cv['n_units']} {unit} in {cv['n_folds']} folds: "
                     f"fitted without each fold and scored on it. The held-out log-likelihood "
                     f"is best at {cv['best']} population{'' if cv['best'] == 1 else 's'} "
                     f"(0 = best)."
                     + (" Cells share their replicates with the training cells, so Replicates is the "
                        "stricter test." if cv["unit"] == "cell" else ""))
    notes.append("The largest ECDF gap is the Kolmogorov distance. It is descriptive: no p-value is "
                 "given, because pooled jumps are not independent observations.")
    zeros = result.get("zero_jumps")
    if zeros and zeros.get("note"):
        # exact-zero jumps (`sqd.zero_jumps`): what every fit on this card did with them, and why 1 is missing
        notes.append(zeros["note"] + (" One population cannot hold them beside the moving ones, so the table "
                                      "starts at two." if zeros.get("immobile") else ""))
    return {"columns": columns, "rows": rows, "notes": notes}


def verdict_line(result: dict) -> str:
    """The card's one line under its table: what BIC and the held-out units each prefer.
    The explanations the cards used to repeat are `READING_NOTES`, said once."""
    if result.get("pending"):
        return "BIC and the held-out fits: computing…"
    parts = []
    best = result.get("bic_best")
    if best is not None:
        deltas = sorted(v["delta_bic"] for v in result["fits"].values()
                        if v.get("delta_bic") is not None)
        part = f"BIC prefers {best} population{'' if best == 1 else 's'}"
        if len(deltas) > 1 and deltas[1] < 2:
            part += f" (inconclusive: the runner-up is {deltas[1]:.1f} behind)"
        parts.append(part)
    cv = result.get("cv") or {}
    unit = {"day": "replicates", "cell": "cells"}.get(cv.get("unit"), "units")
    if cv.get("note"):
        parts.append(f"held out: {cv['note'].split(':')[0]}")
    elif cv.get("n_folds"):
        k = cv["best"]
        parts.append(f"held-out {unit} prefer {k} population{'' if k == 1 else 's'} "
                     f"({cv['wins'].get(k, 0)} of {cv['n_folds']} folds won)")
    return " · ".join(parts)


#: What every card used to repeat, said once on the page (the "How to read" hover card).
READING_NOTES = (
    "Bars: the jumps in each bin, in the condition's colour, in the unit chosen under Y axis. Show turns each fit "
    "on or off (at least one stays on). One fit is drawn in black; two or more use their own colours and "
    "SMTracker 2's line styles (1 population dotted, 2 solid, 3 dashed, 4 dash-dot), and the residuals follow the "
    "same fits. Components draws each fit's populations. Y axis: Probability (the bars sum to 1, as in SMTracker 2 "
    "and Spot-On), Counts (jumps per bin) or Density (per nm, area 1); only the scale changes. Same y scale puts "
    "every chart of one unit on the largest range. The curves are the SQD section's fit (ECDF least squares, "
    "step-weighted).\n\n"
    "Pearson residuals: (observed - expected) / sqrt(expected). Jumps within a track are not independent, so read "
    "the pattern (a run of one sign marks a misfit), not the size. Bins beyond ±2 are red.\n\n"
    "ECDF residuals: each shown fit's model CDF minus the data's, unbinned.\n\n"
    "ΔBIC: from maximum-likelihood fits, so it can differ slightly from the curves. Lowest is preferred; a "
    "difference under 2 is inconclusive.\n\n"
    "Largest ECDF gap: the Kolmogorov distance, descriptive only. No p-value is given because pooled jumps "
    "are not independent observations.\n\n"
    "Held-out ΔlogL: each replicate (or a fifth of the cells) is left out in turn, the fits are made on the "
    "rest and scored on the held-out part; 0 marks the best. Cells share their replicate with the training "
    "cells, so Replicates is the stricter test.")


SETTING_HELP = {
    "max_pop": ("How many populations are fitted.",
                "Up to 3 fits 1, 2 and 3 populations, as Best does in the SQD section. Up to 4 adds a "
                "fourth, which takes about four times as long; Show then offers 4."),
    "cv": ("What is held out to test each fit.",
           "Replicates: each replicate is left out in turn, the fits are made on the others and "
           "scored on it. Cells: five folds of this condition's cells, for single-session data. Cells share "
           "their replicates with the training cells, so this is the gentler test."),
}


def setting_help(key: str) -> tuple[str, str]:
    return SETTING_HELP[key]


def config_problem(included: dict) -> str:
    if not any(included.values()):
        return "Pick at least one condition."
    return ""


def jumps_config(included: dict, cv_unit: str, max_pop: int = MAX_POP) -> dict:
    problem = config_problem(included)
    if problem:
        raise ValueError(problem)
    if cv_unit not in CV_UNITS:
        raise ValueError(f"cv_unit must be one of {CV_UNITS}")
    if int(max_pop) not in MAX_POP_CHOICES:
        raise ValueError(f"max_pop must be one of {MAX_POP_CHOICES}")
    return {"conditions": [k for k, on in included.items() if on], "cv_unit": cv_unit,
            "max_pop": int(max_pop)}
