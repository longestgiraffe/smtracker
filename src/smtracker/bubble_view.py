"""The SQD bubble view's data, rules and sentences, with no Qt anywhere in it.

`ui/workspace/bubble_chart.py` is the Qt shell (the bubble chart, its tooltip, drawer, card and the replicate detail);
this is what it draws from. They are split because of the Thin Qt rule (docs/gui_tool_views.md; `tests/test_ui_thin_qt.py`
walks the whole GUI layer): a wrong number must fail in a headless test, not look subtly wrong on screen, and the
process-pool callables must import without Qt.

What lives here:
  * `warm_condition` and `compute_bubble_view`, the two callables the Run submits to the process pool (picklable, no Qt),
    with `progress_text`, the line a running Run shows;
  * `BubblePoint` and `BubblePlotData`, the plain-field dataclasses the chart paints (`BubbleChart` reads their
    attributes in a dozen places and four other charts build them, so they stay dataclasses; every field is a plain
    float, int, string, list or tuple);
  * THE SAMPLE UNDER EACH COLUMN (`BubblePlotData.samples`, `sample_text`, `sample_marks`, `sample_lines`): "3
    replicates · 1,271 tracks", read from the fit's payload, drawn by the screen chart and the exported figure alike;
    where the figure's conditions were fitted with D forced differently (`BubblePlotData.d_scopes`, `mixed_scopes`,
    1 Nov 2026) it ends with how ("· same D across replicates" / "· D free per replicate",
    `workspace_view.SCOPE_SAMPLE_WORDS`), and so do the caption (`scopes_caption`) and the table's (`shared_d_words`);
    one Force same D across for every condition changes nothing. Every word for that setting is
    `workspace_view`'s (`scope_words`, `SHARED_D_PILLS`, `ARRANGEMENT_NAMES`), read here, never retyped;
  * the sentences and rules: `fraction_text` (ONE formatter for a fraction and its spread), `display_series`,
    `spread_caption`, `at_best_cap`, and the data table's rows, keys and facets (`table_rows`, `point_key`,
    `condition_facets`, `series_facets`) and the replicate detail's (`replicate_table_rows`, `shared_d_caption`).

The Uncertainty setting (`UNCERTAINTY_MODES`) changes the POINT and the SPREAD together, so an interval is never drawn
beside a number it was not computed for; see its comment below.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# THE WORDS FOR FORCE SAME D ACROSS, spelled once in `workspace_view` (ruled 1 Nov 2026) and read here
from .workspace_view import (ARRANGEMENT_NAMES, SCOPE_SAMPLE_WORDS, SHARED_D_NAME, SHARED_D_PILLS, scope_words,
                             setting_words)

__all__ = [
    "UNCERTAINTY_MODES",
    "chunk_words",
    "regrouped",
    "reordered",
    "at_model",
    "column_colours",
    "model_label",
    "models_words",
    "fit_conditions",
    "fold_best",
    "folded_names",
    "folded_words",
    "side_by_side",
    "side_by_side_key",
    "BubblePlotData",
    "BubblePoint",
    "at_best_cap",
    "compute_bubble_view",
    "condition_facets",
    "copy_rows",
    "sqd_receipt",
    "display_series",
    "series_display_names",
    "fraction_text",
    "stacked_fraction",
    "point_key",
    "progress_text",
    "sample_text",
    "sample_marks",
    "sample_lines",
    "column_condition",
    "SAMPLE_UNITS",
    "SCOPE_SAMPLE_WORDS",
    "scope_words",
    "mixed_scopes",
    "scopes_caption",
    "shared_d_words",
    "replicate_table_rows",
    "shows_pairs",
    "series_facets",
    "shared_d_caption",
    "spread_caption",
    "table_rows",
    "warm_condition",
]


def display_series(name: str) -> str:
    """A series' name for DISPLAY (legend, table, caption) -- just the fit
    mode, capitalised. The population count that `compare.bubble_series`
    bakes into its own `"series"` column ("independent, 2 pop") is dropped
    here: this dialog's Overlay control has ONE "Populations" setting for
    every series shown at once (`bubble_n_pops` is a single value, not
    per-series), so every series on screen already carries the same count,
    and the bubbles themselves already show it structurally (same y-axis, same
    column) -- printing it a second time is not information, it is noise.

    The RAW name (with the count) stays exactly as `bubble_series` wrote it
    everywhere this module does a LOOKUP against it (`series_names.index`,
    `overlay["series"] == name`, ...); only the text shown to a reader goes
    through this. If `compare.bubble_series` is ever asked for several
    DIFFERENT population counts in one overlay, the count stops being
    redundant and belongs back in the label -- not a case this dialog can
    currently produce, so not handled here.

    THE ARRANGEMENT'S NAME (`workspace_view.ARRANGEMENT_NAMES`, 1 Nov 2026): "Same D across replicates" /
    "Same D across the whole data set", "Independent" / "Simultaneous" before, which clashed once "Replicates" named
    the simultaneous fit of a condition's replicates.
    """
    mode = name.split(",", 1)[0].strip()
    return ARRANGEMENT_NAMES.get(mode, mode.capitalize())


def series_display_names(names) -> list:
    """The overlay's series as a reader sees them, named by WHAT DIFFERS between them: the arrangement when every
    series has one count ("Same D across replicates" / "Same D across the whole data set", `display_series`), the
    count when every series has one
    arrangement ("2 populations" / "3 populations": the Overlay's "2 and 3"), both when both differ. A count is
    never dropped where it tells two series apart -- 38 % of one of two states and 38 % of one of three are
    different quantities."""
    parsed = []
    for name in names:
        mode, _sep, count = str(name).partition(",")
        parsed.append((mode.strip(), count.replace("pop", "").strip()))
    modes = {m for m, _c in parsed}
    counts = {c for _m, c in parsed}
    if len(counts) <= 1:
        return [display_series(n) for n in names]
    if len(modes) == 1:
        return [f"{c} populations" for _m, c in parsed]
    return [f"{display_series(m)}, {c} populations" for m, c in parsed]


@dataclass
class BubblePoint:
    """One drawn bubble: one population, of one series, in one condition's column."""

    column: int             # index into BubblePlotData.conditions
    x_offset: float          # series offset within the column (0.0 outside overlay)
    d: float                 # µm²/s, apparent (never corrected here)
    fraction: float
    fraction_sd: float
    n_replicates: int
    population: int          # 1-based, slowest -> fastest
    style_index: int         # into _SERIES_STYLES; 0 outside overlay
    #: WHAT THE SPREAD MEANS, carried on the point so no reader of it has to guess:
    #: "sd" = SD over replicate days (a spread, not a confidence interval), "ci" = the
    #: 95 % interval from resampling movies or cells (`interval` holds it), "none" = a pooled
    #: point with no interval. See `fraction_text`.
    spread: str = "sd"
    interval: tuple | None = None
    #: D's own spread where the Run has one (the crossed plot's vertical bar): the SD of each replicate day's own D
    #: under Force same D across = Don't force anywhere, or the 95 % interval from resampling movies or cells under
    #: Uncertainty = Movies or Cells. Zero /
    #: None where D is shared across days by construction.
    d_sd: float = 0.0
    d_interval: tuple | None = None
    #: Where the drawn D sits on the bounds the fit was made within (`sqd.at_bound`: "lower", "upper" or ""): such a
    #: D is not resolved, and the table and the hover say so (`d_text`). Set by the SQD Run (`_mark_bounds`).
    d_bound: str = ""


def d_text(pt: BubblePoint, unit: str = "") -> str:
    """A drawn point's D as the table and the hover print it: "not resolved" where it sits on the fit's bound
    (`BubblePoint.d_bound`; the per-spot audit, 7 Nov 2026), never the bound as if it were a value."""
    return "not resolved" if getattr(pt, "d_bound", "") else f"{pt.d:.3g}{unit}"


def _mark_bounds(points) -> None:
    """Each drawn SQD point whose D sits on the bounds every fit of the Run is made within (`sqd.D_BOUNDS`):
    ``d_bound`` set (`sqd.at_bound`). The drawn D is the fit's own (or its replicates' mean, on the bound only where
    every replicate's is); the immobile population's D of 0 is never on one."""
    from .sqd import at_bound

    for pt in points:
        pt.d_bound = at_bound([pt.d])[0] if pt.d > 0 else ""


def fraction_text(pt: BubblePoint, *, compact: bool = False) -> str:
    """A point's fraction with its spread, as every surface shows it.

    ONE FORMATTER, because this string was written out three times -- the bubble's
    own label, the hover tooltip and the data table -- and three copies of a format
    is three places for "± SD" to be printed beside a number that no longer has one.
    ``compact`` drops the spaces around "±" for the label drawn on the bubble itself ("73±3%"); an interval keeps
    its space in either form ("67% [63–71]", ruled 4 Nov 2026, D16: "67%[63–71]" read as one token).
    """
    gap = "" if compact else " "
    pct = 100.0 * pt.fraction
    if pt.spread == "ci" and pt.interval is not None:
        lo, hi = pt.interval
        return f"{pct:.0f}% [{100 * lo:.0f}–{100 * hi:.0f}]"
    # a ONE-population fit is 100 % by definition: "± 0" would read as a measured agreement between replicates
    if pt.spread == "sd" and pt.n_replicates >= 2 and pt.fraction < 1.0:
        return f"{pct:.0f}{gap}±{gap}{100 * pt.fraction_sd:.0f}%"
    return f"{pct:.0f}%"


def stacked_fraction(text: str) -> str:
    """A `fraction_text` (compact) with its spread on a second line, for a number wider than its column in print
    (`figure_export._number_style`): "73±3%" -> "73%" over "±3", "73% [60–80]" -> "73%" over "[60–80]"; a number with
    no spread is unchanged."""
    text = str(text)
    if "±" in text:
        value, spread = text.split("±", 1)
        return f"{value.strip()}%\n±{spread.strip().rstrip('%')}"
    if "[" in text:
        value, spread = text.split("[", 1)
        return f"{value.strip()}\n[{spread.strip()}"
    return text


@dataclass
class BubblePlotData:
    """Everything `BubbleChart` needs to paint one Run -- built once, off the
    GUI thread, by `compute_bubble_view`; the chart itself does no fitting."""

    conditions: list
    points: list
    errors: dict                       # column index -> why that column is empty
    series_labels: list                # [] outside an overlay
    scope: str
    fraction_weighting: str
    #: The Uncertainty setting the points were drawn under (`UNCERTAINTY_MODES`); "" for a
    #: chart built elsewhere (the apparent-D bubbles), which says nothing about days.
    uncertainty: str = ""
    #: THE PER-DAY VALUES BEHIND EACH DRAWN MEAN, for a significance bracket on an exported
    #: figure (ruled 13 Oct 2026: Welch on per-day values, `significance`). Filled ONLY when
    #: the drawn point IS the mean of these values -- Uncertainty = Days, one arrangement --
    #: so a test can never be run on numbers the figure does not show. Per fitted condition:
    #: ``{"unit": "day" | "cell", "replicates": [day labels], "fractions": [[population 1,
    #: 2, ...] per day], "D": [per population]}``; unit "cell" is one session (one value).
    unit_values: dict = field(default_factory=dict)
    #: The run's D threshold (µm²/s): a significance bracket can compare the share of molecules
    #: in states below it, which means the same at any state count. None where there is none.
    threshold_um2_s: float | None = None
    #: The time lag the fits used, in frames, and in ms (`ComparisonSpec.sqd_lag`): the caption and the
    #: exported legend name it whenever it is not 1, since D is then the apparent D at that lag.
    lag_frames: "int | str" = 1
    lag_ms: float | None = None
    #: PER-SPOT PRECISION (`ComparisonSpec.sqd_precision`, 6 Nov 2026): the calibration's words when the Run fitted
    #: each step's own localisation error, "" otherwise. Said in the caption and the exported legend.
    precision: str = ""
    #: What a reader must check before trusting a drawn fit (`fit_warnings`): a population under 5 %, or two
    #: populations within a factor of two in D. Said on screen, in the caption and in the exported legend.
    warnings: list = field(default_factory=list)
    #: RANDOM CHUNKS IN PLACE OF THE REPLICATES (Uncertainty "Chunks", `comparison.PSEUDO_CHUNKS`): ``{"k", "seed",
    #: "splits", "by": {condition: "movie" | "cell"}, "real_replicates": {condition: n}}``; empty otherwise. The
    #: points are drawn as under "Replicates" (``uncertainty`` is "day"); every sentence about them reads this.
    pseudo: dict = field(default_factory=dict)
    #: WHERE EACH COLUMN STANDS when the conditions are grouped by their factors (`factor_layout.columns`, folder
    #: designs phase 4): ``{"x", "span", "groups", "labels", "outer"}``. Empty is the plain layout (column k at k,
    #: each labelled by its condition), so a chart built without factors draws exactly as before.
    layout: dict = field(default_factory=dict)
    #: SHARED D WITHIN A GROUP (scope "group"): the factor the groups are values of, and each condition's group;
    #: empty under every other scope. Every sentence about the sharing reads these (`scope_words`).
    sharing_factor: str = ""
    sharing_groups: dict = field(default_factory=dict)
    #: FITS SIDE BY SIDE (`ComparisonSpec.side_by_side`, 28 Oct 2026): per column ``{"condition", "model", "n_pop",
    #: "label", "shade", "shades"}`` -- the condition the column is a fit of, the model it was fitted with ("best",
    #: "1".."4", "saspt"), the populations it holds (None when the fit failed), its label under the axis, and its
    #: shade of the condition's colour -- with ``conditions`` holding each column's own name (`side_by_side_key`).
    #: Empty under every other Run, where each column is a condition at its own model. A Best column that stands for a
    #: count also picked (`fold_best`, 1 Nov 2026) carries ``"also"`` (those counts) and ``"folded"`` (the names their
    #: own columns would have had, `folded_names`).
    fits: list = field(default_factory=list)
    #: THE SAMPLE UNDER EACH COLUMN (30 Oct 2026, asked: '"n" on the SQD bubble plots ... under each column or group,
    #: the sample: "3 replicates · 1,271 tracks"'): per CONDITION ``{"unit", "n", "tracks"}`` -- ``unit`` what the
    #: drawn spread is over ("day" replicates, "chunk", "movie", "cell") or "pooled" (no spread: ``n`` the replicates
    #: pooled), ``n`` how many, ``tracks`` the tracks fitted -- READ FROM THE FIT'S PAYLOAD (`condition_replicate_fit`'s
    #: ``pooled`` and ``replicates``; the apparent-D result's own counts), never counted again here. Keyed by
    #: condition, so fits side by side (one sample, several models) and a regrouped order read it unchanged. Empty for
    #: a Run made before it existed: nothing is drawn (`sample_marks`).
    samples: dict = field(default_factory=dict)
    #: HOW EACH CONDITION'S D WAS FITTED (1 Nov 2026, a Shared D per row of the Compute dialog's Conditions page):
    #: ``{condition: scope}`` ("condition", "replicate", "group", "global"), READ FROM EACH FITTED COLUMN'S PAYLOAD
    #: (`BubbleColumn.scope`), every fitted condition, keyed by condition like ``samples``. Nothing reads it directly:
    #: `mixed_scopes` says whether the figure's conditions differ in it, and only then does the sample line, the
    #: caption, the legend or a bracket say anything. Empty for an overlay and for a Run made before it existed.
    d_scopes: dict = field(default_factory=dict)
    #: A COUNT SET BY HAND WHERE BEST (BIC) CHOOSES FEWER (`forced_count_warning`, 4 Nov 2026), per fitted condition:
    #: ``"<condition>: Best (BIC) chooses ..."``. Part of ``warnings`` as well; kept apart so fits side by side, which
    #: read their warnings off the joined points again, carry them over.
    count_warnings: list = field(default_factory=list)

    def __setstate__(self, state) -> None:
        """A Run kept in a SESSION FILE from before a field existed (`pseudo`, 24 Oct 2026; `layout`, 25 Oct) reads
        that field as its default. A pickle restores only what was saved, and a field with a factory default is no
        class attribute: without this, opening such a session raised on the crossed view's ``data.pseudo``."""
        from dataclasses import MISSING, fields

        state = dict(state)
        for f in fields(type(self)):
            if f.name not in state:
                if f.default is not MISSING:
                    state[f.name] = f.default
                elif f.default_factory is not MISSING:
                    state[f.name] = f.default_factory()
        self.__dict__.update(state)

    def column_x(self, column: int) -> float:
        """Column ``column``'s x position: the grouped layout's where there is one, else the index."""
        xs = self.layout.get("x") if self.layout else None
        return float(xs[column]) if xs and 0 <= column < len(xs) else float(column)

    def column_labels(self) -> list:
        """What each column is labelled with under the axis: the inner factor values when grouped (the outer one is
        the group's header), else the condition names."""
        labels = self.layout.get("labels") if self.layout else None
        return [str(v) for v in labels] if labels else [str(c) for c in self.conditions]

    def x_range(self) -> tuple:
        """The x axis's range: half a column and a little beyond the first and last."""
        if self.layout and self.layout.get("x"):
            return (float(self.layout["x"][0]) - 0.6, float(self.layout["x"][-1]) + 0.6)
        return (-0.6, max(1, len(self.conditions)) - 0.4)


#: The words for a sample's unit, one and several (`sample_text`); a user reads "replicate", never "day".
SAMPLE_UNITS = {"day": ("replicate", "replicates"), "chunk": ("chunk", "chunks"), "movie": ("movie", "movies"),
                "cell": ("cell", "cells")}


#: A column's Force same D across on its sample line, said only when the figure's conditions differ in it
#: (`mixed_scopes`), is `workspace_view.SCOPE_SAMPLE_WORDS` (imported above): "same D across replicates" / "D free per
#: replicate" since the evening of 1 Nov 2026 ("D within condition" / "D per replicate", then "D across replicates" /
#: "D not shared" before).


def mixed_scopes(data) -> dict:
    """``{condition: scope}`` when the figure's conditions were fitted with their D forced DIFFERENTLY, else ``{}`` --
    the one
    test every sentence about it reads (the sample line, the captions, the table and the exported legend, and the
    refused bracket in `significance.compare`). Only where the scope moves what is drawn: under Uncertainty
    Replicates (or Chunks) each bubble is the replicates' mean at the condition's D, shared or each replicate's own;
    under None, Movies or Cells it is the pooled fit, the same fit under either, so there is nothing to tell apart
    (`scopes_caption` says so instead)."""
    scopes = dict(getattr(data, "d_scopes", None) or {}) if data is not None else {}
    if getattr(data, "uncertainty", "") != "day" or len(set(scopes.values())) < 2:
        return {}
    return scopes


def scopes_caption(data) -> str:
    """The caption's words for a Force same D across set per condition ("" when the figure has one): which condition
    was fitted how, and that a bracket between two fitted differently is refused -- or, under a pooled fit, that the
    choice does not move what is drawn."""
    scopes = mixed_scopes(data)
    if scopes:
        listed = "; ".join(f"{c} ({scope_words(s)})" for c, s in scopes.items())
        return (f"   [{SHARED_D_NAME} set per condition: {listed}; written under each column. A significance "
                f"bracket between two conditions fitted differently is refused]")
    every = dict(getattr(data, "d_scopes", None) or {}) if data is not None else {}
    if len(set(every.values())) >= 2:
        return (f"   [{SHARED_D_NAME} was set per condition, but each bubble is the pooled fit of the condition, which "
                f"is the same fit under {SHARED_D_PILLS['condition']} and {SHARED_D_PILLS['replicate']}]")
    return ""


def shared_d_words(data) -> str:
    """Force same D across in words for a whole figure: `scope_words` of its scope, or per condition where they
    differ."""
    scopes = mixed_scopes(data)
    if scopes:
        return "D forced per condition (" + "; ".join(f"{c}, {scope_words(s)}" for c, s in scopes.items()) + ")"
    return scope_words(data.scope, getattr(data, "sharing_factor", ""))


def sample_text(sample, scope: str | None = None) -> str:
    """A column's sample in words: "3 replicates · 1,271 tracks", "8 movies · 1,271 tracks", "22 cells · 1,271
    tracks", "5 chunks · ...", or with no spread drawn "3 replicates pooled · 1,271 tracks" ("1,271 tracks" for one
    session). "" with nothing to say. A count the payload did not hold is left out, never guessed. ``scope``, given
    only where the figure's conditions differ in Force same D across (`mixed_scopes`), adds how its D was fitted: "3
    replicates · 1,271 tracks · D free per replicate" (`SCOPE_SAMPLE_WORDS`)."""
    from .chart_text import TITLE_JOIN

    if not sample:
        return ""
    unit, n, tracks = sample.get("unit"), sample.get("n"), sample.get("tracks")
    parts = []
    if n is not None:
        n = int(n)
        if unit == "pooled":
            if n > 1:
                parts.append(f"{n:,} {SAMPLE_UNITS['day'][1]} pooled")
        elif unit in SAMPLE_UNITS:
            one, many = SAMPLE_UNITS[unit]
            parts.append(f"{n:,} {one if n == 1 else many}")
    if tracks is not None:
        parts.append(f"{int(tracks):,} track{'' if int(tracks) == 1 else 's'}")
    if scope and parts:
        parts.append(SCOPE_SAMPLE_WORDS.get(scope, scope_words(scope)))
    return TITLE_JOIN.join(parts)


def sample_lines(text: str, measure, width: float) -> list:
    """``text`` (`sample_text`) as the lines it is drawn in under a column ``width`` wide (``measure`` the renderer's
    own): one line where it fits, else one part a line ("3 replicates" / "1,271 tracks"). A part is never cut."""
    from .chart_text import TITLE_JOIN

    if not text:
        return []
    if width <= 0 or measure(text) <= width:
        return [text]
    return [part for part in text.split(TITLE_JOIN) if part]


def column_condition(data, column: int) -> str:
    """The condition column ``column`` draws: the column itself, or under fits side by side the one it is a fit of."""
    fits = getattr(data, "fits", None) or []
    if fits and 0 <= column < len(fits):
        return str(fits[column]["condition"])
    return str(data.conditions[column])


def sample_marks(data) -> list:
    """Where the sample is written under the chart, the screen's and the export's one rule: ``[{"x0", "x1", "text",
    "under"}]`` in the chart's x units. Under each COLUMN (``under`` "label": right under its label) -- or, with fits
    side by side, ONCE PER GROUP (``under`` "group": under the condition's header), since its columns are fits of one
    sample. A column with no fit has none; a Run without samples gives ``[]``. Where the figure's conditions differ in
    Force same D across (`mixed_scopes`) each line ends with how its D was fitted ("· D free per replicate")."""
    samples = getattr(data, "samples", None) or {}
    if data is None or not samples:
        return []
    layout = getattr(data, "layout", None) or {}
    errors = getattr(data, "errors", None) or {}
    scopes = mixed_scopes(data)
    if getattr(data, "fits", None) and layout.get("groups"):
        out = []
        for group in layout["groups"]:
            first, last = int(group["first"]), int(group["last"])
            if all(i in errors for i in range(first, last + 1)):
                continue
            condition = column_condition(data, first)
            text = sample_text(samples.get(condition), scopes.get(condition))
            if text:
                out.append({"x0": float(group["x0"]) - 0.4, "x1": float(group["x1"]) + 0.4, "text": text,
                            "under": "group"})
        return out
    out = []
    for i in range(len(data.conditions)):
        if i in errors:
            continue
        condition = column_condition(data, i)
        text = sample_text(samples.get(condition), scopes.get(condition))
        if text:
            x = data.column_x(i)
            out.append({"x0": x - 0.5, "x1": x + 0.5, "text": text, "under": "label"})
    return out


def _sample(payload, spread: str, uncertainty: str) -> dict:
    """One condition's sample from its replicate payload (`condition_replicate_fit`), for the spread its points were
    drawn with: the replicates (or chunks) behind an SD, the movies or cells resampled for an interval, the replicates
    pooled where no spread is drawn; the tracks fitted. {} where the payload failed."""
    if not isinstance(payload, dict) or "error" in payload:
        return {}
    pooled = payload.get("pooled") or {}
    replicates = len(payload.get("replicates") or ()) or None
    if spread == "sd":
        unit, n = ("chunk" if payload.get("pseudo") else "day"), replicates
    elif spread == "ci":
        unit = uncertainty if uncertainty in ("movie", "cell") else "cell"
        n = pooled.get(f"n_{unit}s")
    else:
        unit, n = "pooled", replicates
    return {"unit": unit, "n": n, "tracks": pooled.get("n_tracks")}


def _samples(conditions, points, payloads, uncertainty: str) -> dict:
    """Every fitted condition's sample (`_sample`), the spread read off its drawn points."""
    spreads = {}
    for pt in points:
        spreads.setdefault(int(pt.column), pt.spread)
    return {str(c): _sample((payloads or {}).get(c), spreads[i], uncertainty)
            for i, c in enumerate(conditions) if i in spreads}


def regrouped(data: "BubblePlotData", factors: dict, nesting) -> "BubblePlotData":
    """``data`` with its columns in ``nesting``'s order (`factor_layout.nested`) and that order's grouped layout --
    the same points, moved, nothing refitted (a fit is per condition). Without two factors it is ``data`` with the
    plain layout. An overlay's series keep their offsets within each column."""
    from dataclasses import replace as _replace

    from .factor_layout import columns, nested

    if data.fits:
        # FITS SIDE BY SIDE are grouped by their condition already: the conditions take the nesting's order
        return _side_by_side_order(data, nested(fit_conditions(data), factors or {}, tuple(nesting or ())))
    names = [str(c) for c in data.conditions]
    order = nested(names, factors or {}, tuple(nesting or ()))
    where = {c: i for i, c in enumerate(order)}
    old = {i: c for i, c in enumerate(names)}
    points = [_replace(pt, column=where[old[pt.column]]) for pt in data.points]
    errors = {where[old[i]]: reason for i, reason in (data.errors or {}).items()}
    return _replace(data, conditions=list(order), points=points, errors=errors,
                    layout=columns(order, factors or {}, tuple(nesting or ())))


def reordered(data: "BubblePlotData", order) -> "BubblePlotData":
    """``data`` with its columns in the set condition ``order`` (`condition_order`, 3 Nov 2026): the same points,
    moved, nothing refitted. Fits side by side keep each condition's columns together. A grouped layout is left for
    `regrouped` to lay out again (the window's `_grouped` follows this); a plain one stays plain."""
    from dataclasses import replace as _replace

    from .condition_order import apply

    if data is None:
        return data
    if data.fits:
        return _side_by_side_order(data, apply(fit_conditions(data), order))
    names = [str(c) for c in data.conditions]
    new = apply(names, order)
    if new == names:
        return data
    where = {c: i for i, c in enumerate(new)}
    old = {i: c for i, c in enumerate(names)}
    points = [_replace(pt, column=where[old[pt.column]]) for pt in data.points]
    errors = {where[old[i]]: reason for i, reason in (data.errors or {}).items()}
    return _replace(data, conditions=list(new), points=points, errors=errors,
                    layout=data.layout if (data.layout or {}).get("groups") else {})


def chunk_words(pseudo: dict, *, short: bool = False) -> str:
    """What a chunk spread is, in words: "5 random chunks of each condition's movies (pseudo-replicates within one
    culture ...)". ONE sentence for the caption, the legend, the source data and the brackets."""
    if not pseudo:
        return ""
    kinds = sorted(set((pseudo.get("by") or {}).values())) or ["movie"]
    units = " or ".join("movies" if k == "movie" else "cells" for k in kinds)
    # each condition its own count since 23 Nov 2026 (`comparison.pseudo_chunk_count`: at least 3 a chunk)
    counts = sorted(set(int(v) for v in (pseudo.get("ks") or {}).values())) or [int(pseudo["k"])]
    number = str(counts[0]) if len(counts) == 1 else f"{counts[0]}-{counts[-1]}"
    head = f"{number} random chunks of each condition's {units}"
    if short:
        return head
    return (f"{head} (pseudo-replicates within one culture, deal {pseudo.get('seed', 0)}: they show how precisely "
            f"these cells were measured and say nothing about another culture)")


#: The Uncertainty setting's three values, and what each one draws. The POINT and
#: the SPREAD change together, so an interval is never drawn beside a number it was
#: not computed for:
#:
#: * ``"none"`` -- every replicate day pooled into one fit (series001-series030 as one
#:   set), drawn as it is, no interval.
#: * ``"day"``  -- the legacy simultaneous design: D shared, fractions fit per
#:   replicate day, their MEAN ± SD over days. The SD is a spread between sessions,
#:   not a confidence interval.
#: * ``"cell"`` -- the pooled fit with a 95 % interval from resampling cells. Only on
#:   a SINGLE session: with two or more replicate days the day is the unit, cells
#:   within a day are not independent, and a cell interval comes out too narrow
#:   (Report A, `docs/uncertainty_tier_audit.md`). There the pooled point is drawn
#:   with no interval and the caption says why.
#: * ``"movie"`` -- the same with MOVIES resampled (19 Oct 2026). Cells in one movie
#:   share its focus, illumination and field, so on SynR the cell interval held the
#:   population value in only 80 % of sessions (`docs/synthetic_synr.md`); the movie is
#:   the coarser unit. Same single-session gate.
#: * ``"chunk"`` -- RANDOM CHUNKS as pseudo-replicates (24 Oct 2026, `comparison.PSEUDO_CHUNKS`): drawn exactly as
#:   "day" (the spec's ``pseudo_replicates`` makes the chunks the replicates), and said to be chunks everywhere.
UNCERTAINTY_MODES = ("none", "day", "movie", "cell", "chunk")
#: The single-session choices: a percentile interval from resampling this unit.
RESAMPLED_UNITS = ("movie", "cell")
#: A resampled unit, as a sentence counts it.
UNIT_WORDS = {"movie": "movies", "cell": "cells"}


def warm_condition(project, spec, condition, *, uncertainty: str, cache) -> str:
    """Everything one condition contributes to an SQD Run, computed INTO `cache`.

    A process-pool job per condition, so conditions fit side by side and each one's
    completion can be reported -- the Run used to be one opaque job, and "computing"
    was all it could say. `compute_bubble_view` then reads every piece back as a
    cache hit. Returns the condition, so the caller can say which one finished.

    The costly part is not the ECDF: with the model "Best" the state count is chosen
    by BIC over one to four populations with the MLE estimator, at the start counts
    measured as necessary. Measured on the reference project: 229 of 234 s.
    """
    from . import comparison as cmp
    from . import comparison_controller as cctrl
    from .sqd import gradient_mode

    if getattr(spec, "side_by_side", ()):
        # FITS SIDE BY SIDE: this condition at every model ticked, each the Run's own fit with the model locked to it
        for model in spec.side_by_side:
            fitted = fitted_at(spec, model)
            if str(condition) in fitted:
                warm_condition(project, at_model(spec, model, fitted), condition, uncertainty=uncertainty,
                               cache=cache)
        return condition
    # THE OPTIMISER THE RUN CHOSE (`ComparisonSpec.sqd_gradient`), for this job only: a pool
    # worker serves other sections next.
    with gradient_mode(getattr(spec, "sqd_gradient", "numeric")):
        payload = cctrl.replicate_fit_payload(project, spec, condition, cache=cache,
                                              compute=True)
        # payload "unit" "cell" means ONE session (fewer than two replicate days)
        if uncertainty in RESAMPLED_UNITS and payload.get("unit") == "cell":
            cmp._bubble_ci(project, condition, spec, int(payload["n_pop"]), cache,
                           unit=uncertainty)
    return condition


def progress_text(done: int, total: int, finished, elapsed_s: float, *,
                  choosing_count: bool) -> str:
    """The line a running SQD Run shows, e.g. "1 of 2 conditions fitted (glucose) ·
    1:52". Names what finished, so a long wait is visibly moving; says when the
    state count is being chosen, since that -- not the fit -- is the slow part."""
    minutes, seconds = divmod(int(elapsed_s), 60)
    clock = f"{minutes}:{seconds:02d}"
    if total <= 0:
        return f"fitting · {clock}"
    parts = [f"{done} of {total} condition{'s' if total != 1 else ''} fitted"]
    if finished:
        parts[0] += f" ({', '.join(finished)})"
    if done >= total:
        parts[0] = "assembling the figure"
    elif choosing_count:
        parts.append("choosing the state count by BIC (slow, cached afterwards)")
    return " · ".join(parts + [clock])


def compute_bubble_view(project, spec, *, uncertainty: str = "day", cache=None):
    """See `_compute_bubble_view`; every fit inside uses the optimiser the Run chose. With fits side by side
    (`ComparisonSpec.side_by_side`) it is made once per model ticked, the model locked to it, and the figures are
    joined into one (`side_by_side`): every column is exactly the figure that model would draw alone."""
    from .sqd import gradient_mode

    with gradient_mode(getattr(spec, "sqd_gradient", "numeric")):
        models = tuple(getattr(spec, "side_by_side", ()) or ())
        if models:
            runs = []
            for model in models:
                fitted = fitted_at(spec, model)
                if fitted:
                    runs.append((model, *_compute_bubble_view(project, at_model(spec, model, fitted),
                                                              uncertainty=uncertainty, cache=cache)))
            return side_by_side(runs, order=[str(c) for c in spec.conditions], shown=shown_fits(spec))
        return _compute_bubble_view(project, spec, uncertainty=uncertainty, cache=cache)


# -- fits side by side (`ComparisonSpec.side_by_side`, asked 28 Oct 2026: "add multiple fits next to each other,
# should look like group with 1 pop and 2pop ... they should have same colour with different shades"; the models
# ticked on the Compute dialog's Conditions page since 30 Oct: "just tick boxes of what to fit and show") ---------- #

#: A model as a column label says it: a count by itself, Best and saSPT with the count they chose.
_MODEL_WORDS = {"best": "Best", "saspt": "saSPT"}


def at_model(spec, model, conditions=None):
    """``spec`` with every condition locked to ``model`` ("best", "1".."4", "saspt") and no fits side by side: one
    model's Run -- of ``conditions`` (`fitted_at`) when given, else of all of them."""
    from dataclasses import replace as _replace

    kw = {}
    if conditions is not None:
        keep = {str(c) for c in conditions}
        kw["conditions"] = [c for c in spec.conditions if str(c) in keep]
    return _replace(spec, lock_model=str(model), side_by_side=(), side_by_side_shown=None, **kw)


#: The D sharing under which each condition is fitted on its own, so a model's Run can leave out the conditions that
#: do not show it and every fit it makes is the one it would make with them in (`workspace_view.SCOPE_ARRANGEMENT`).
#: `comparison.CONDITION_SCOPES`, the scopes a condition may also have as its own (`ComparisonSpec.scope_for`): a Run
#: mixing the two is still apart. Spelled here because this module imports nothing of the fit at its head; held equal
#: by `tests/test_shared_d_per_condition.py`.
APART_SCOPES = ("condition", "replicate")


def shown_fits(spec) -> dict:
    """``{condition: the models it shows}`` under fits side by side (`ComparisonSpec.side_by_side_shown`; a condition
    not named shows every model ticked), in the spec's condition order."""
    models = tuple(getattr(spec, "side_by_side", ()) or ())
    chosen = getattr(spec, "side_by_side_shown", None) or {}
    return {str(c): tuple(chosen.get(str(c), models)) for c in spec.conditions}


def fitted_at(spec, model) -> list:
    """The conditions ``model``'s Run fits: those that show it, when each condition is fitted on its own
    (`APART_SCOPES`); every condition when D is shared across them, since leaving one out would change the shared D
    of the rest -- the columns not shown are then fitted and not drawn."""
    shown = shown_fits(spec)
    if getattr(spec, "d_sharing_scope", "condition") in APART_SCOPES:
        return [c for c, models in shown.items() if str(model) in models]
    return list(shown)


def model_label(model, n_pop=None) -> str:
    """A side-by-side column's label under the axis (its condition is the group's header): "2-pop", or "Best: 3-pop"
    / "saSPT: 2-pop" for a model that chooses its count ("Best" alone when its fit failed)."""
    model = str(model)
    if model.isdigit():
        return f"{model}-pop"
    word = _MODEL_WORDS.get(model, model)
    return f"{word}: {int(n_pop)}-pop" if n_pop else word


def models_words(models) -> str:
    """The models side by side in a sentence: "1 and 2 populations", "Best (the count BIC prefers) and 2 populations",
    "saSPT's count, 1 and 3 populations"."""
    models = [str(m) for m in models]
    counts = [m for m in models if m.isdigit()]
    parts = (["Best (the count BIC prefers)"] if "best" in models else []) + (
        ["saSPT's count"] if "saspt" in models else [])
    if counts:
        parts.append((", ".join(counts[:-1]) + " and " + counts[-1] if len(counts) > 1 else counts[0])
                     + " population" + ("s" if len(counts) > 1 or counts[0] != "1" else ""))
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def side_by_side_key(condition, label: str) -> str:
    """A side-by-side column's NAME, what `BubblePlotData.conditions` holds for it and every table, tooltip and
    receipt shows: "glucose (2-pop)", "glucose (Best: 3-pop)"."""
    return f"{condition} ({label})"


def fit_conditions(data) -> list:
    """The conditions a figure's columns are fits of, in its order: the columns themselves, or under fits side by
    side each condition once."""
    if data is None:
        return []
    fits = getattr(data, "fits", None)
    if not fits:
        return [str(c) for c in data.conditions]
    return list(dict.fromkeys(str(f["condition"]) for f in fits))


def column_colours(data, colours: dict) -> dict:
    """Each column's colour, by its name: the condition's own from ``colours`` (condition -> hex), or under fits side
    by side a SHADE of it per model (the model's place among those lit, in `MODEL_CHOICES` order, darkest first: one
    model wears one shade in every condition) -- the rule the replicate days are shaded by (`figstyle.day_shade`), so
    one condition keeps one hue everywhere."""
    out = dict(colours)
    fits = getattr(data, "fits", None) if data is not None else None
    if not fits:
        return out
    from .figstyle import day_shade

    for name, fit in zip(data.conditions, fits):
        base = colours.get(fit["condition"])
        if base:
            out[str(name)] = day_shade(str(base), int(fit["shade"]), int(fit["shades"]))
    return out


def _fit_model(fit) -> str:
    """A column's model; a figure from 28-29 Oct recorded only its count."""
    return str(fit.get("model") or fit["n_pop"])


def _side_by_side_layout(conditions, fits) -> dict:
    """The grouped layout (`factor_layout.columns`' shape): each condition a group of its columns (``fits``, in
    drawing order), groups a gap apart, the condition as the group's header and each model's label under its
    column."""
    from .factor_layout import GAP

    xs, groups, labels = [], [], []
    x = 0.0
    for ci, condition in enumerate(conditions):
        mine = [f for f in fits if str(f["condition"]) == str(condition)]
        if ci:
            x += GAP
        first = len(xs)
        for fit in mine:
            xs.append(x)
            labels.append(fit.get("label") or model_label(_fit_model(fit), fit.get("n_pop")))
            x += 1.0
        groups.append({"label": str(condition), "first": first, "last": len(xs) - 1, "x0": xs[first],
                       "x1": xs[-1]})
    return {"x": xs, "span": (-0.5, (xs[-1] if xs else 0.0) + 0.5), "groups": groups, "labels": labels,
            "outer": ""}


def _side_by_side_order(data: "BubblePlotData", order) -> "BubblePlotData":
    """``data`` (fits side by side) with its condition groups in ``order``: the same points, moved."""
    from dataclasses import replace as _replace

    present = list(dict.fromkeys(str(f["condition"]) for f in data.fits))
    order = [c for c in (str(c) for c in order) if c in present]
    order += [c for c in present if c not in order]
    olds =[old for condition in order for old, fit in enumerate(data.fits) if str(fit["condition"]) == condition]
    where = {old: new for new, old in enumerate(olds)}
    names, fits = [None] * len(data.fits), [None] * len(data.fits)
    for old, new in where.items():
        names[new], fits[new] = data.conditions[old], data.fits[old]
    return _replace(data, conditions=names, fits=fits,
                    points=[_replace(pt, column=where[pt.column]) for pt in data.points],
                    errors={where[i]: reason for i, reason in (data.errors or {}).items()},
                    layout=_side_by_side_layout(order, fits))


#: Where a model stands among columns of ONE count (`fit_group_order`): the count picked, then Best, then saSPT.
_SAME_COUNT_RANK = {"best": 1, "saspt": 2}


def fit_group_order(models, mine, n_pops) -> list:
    """One condition's columns side by side in the order of their POPULATION COUNT (asked 1 Nov 2026: "instead of
    showing in the sqd fit group best as first fit, we should still do order: 1 pop, 2 pop, best: 3 pop fit etc, else
    it gets confusing"): a model that chooses its count (Best, saSPT) stands at the count it chose, after a count
    picked as itself; a fit that failed (no count) last, in the ticked order. ``mine`` are indices into ``models``,
    ``n_pops`` each one's count."""
    models = [str(m) for m in models]
    return sorted(mine, key=lambda j: (n_pops.get(j) is None, n_pops.get(j) or 0,
                                       _SAME_COUNT_RANK.get(models[j], 0), list(mine).index(j)))


def fold_best(models, mine, n_pops) -> tuple:
    """BEST ONCE (asked 1 Nov 2026: "fitting for example BEST and 1 pop -> u get twice 1 pop fit of the simulated data.
    if best = something that was already chosen -> only print it once as best"): one condition's columns side by side,
    ``mine`` (indices into ``models``, the models it shows), without a fixed count that its Best column CHOSE --
    ``n_pops`` the count each column's fit holds, None where it failed. Returns ``(kept, folded)``, ``folded`` = ``{the
    Best column: [the counts it stands for]}``. The two are one fit: Best resolves its count and fits at it exactly as
    the count's own Run does (`comparison.fit_condition`; under Whole data set Best is one count for every condition
    and the joint fit is made at it either way), so nothing is lost. A count Best did not choose, or a Best that
    failed, folds nothing."""
    models = [str(m) for m in models]
    if "best" not in models:
        return list(mine), {}
    best = models.index("best")
    chosen = n_pops.get(best) if best in mine else None
    if not chosen:
        return list(mine), {}
    same = [j for j in mine if j != best and models[j] == str(int(chosen))]
    return [j for j in mine if j not in same], ({best: same} if same else {})


def folded_words(fits) -> str:
    """What `fold_best` did, for the caption and the exported legend ("" when nothing): "Best (BIC) chose 1 population
    for mNG, D2 simulated, a count also picked: drawn once, as Best"."""
    by: dict = {}
    for fit in fits or ():
        if fit.get("also"):
            by.setdefault(int(fit["n_pop"]), []).append(str(fit["condition"]))
    if not by:
        return ""
    parts = [f"{n} population{'' if n == 1 else 's'} for {'; '.join(cs)}" for n, cs in sorted(by.items())]
    several = len(by) > 1 or any(len(cs) > 1 for cs in by.values())
    return (f"Best (BIC) chose {' and '.join(parts)}, {'counts' if len(by) > 1 else 'a count'} also picked: "
            + ("each drawn once, as Best" if several else "drawn once, as Best"))


def folded_names(data) -> dict:
    """``{a column's name before `fold_best`: the Best column that stands for it}``: what a significance pair picked
    on a figure from before names now (`significance.compare` reads it). Empty unless a count was folded."""
    out = {}
    for name, fit in zip(getattr(data, "conditions", None) or [], getattr(data, "fits", None) or []):
        for old in fit.get("folded") or ():
            out[str(old)] = str(name)
    return out


def side_by_side(runs, *, order=None, shown=None, head: str | None = None) -> tuple:
    """One figure from one Run per model, ``runs`` = ``[(model, plot, caption, payloads), ...]`` in the ticked order:
    each condition a group of columns, one per model it shows (``shown``, `shown_fits`: ``{condition: models}``,
    every model when None), every column that model's own fit unchanged (points, spread, the per-replicate values a
    bracket tests, the replicate page a click opens), named `side_by_side_key`. ``order`` is the conditions' order
    (the runs' own when None: a model's Run may fit only the conditions that show it, `fitted_at`). ``head`` begins
    the caption of a figure with no D threshold (the apparent-D bubbles, `appd_fits.bubble_plot`, 3 Nov 2026), whose
    columns then carry no below-threshold share. Returns ``(plot, caption, payloads)`` as `compute_bubble_view` does."""
    from dataclasses import replace as _replace

    from .comparison import below_populations, slow_fraction_below

    if not runs:
        raise ValueError("fits side by side: no model's Run to join (no condition shows any)")
    models = [str(m) for m, *_rest in runs]
    first = runs[0][1]
    if order is None:
        order = list(dict.fromkeys(str(c) for _m, plot, *_r in runs for c in plot.conditions))
    conditions = [str(c) for c in order]
    has = {(str(c), j) for j, (_m, plot, *_r) in enumerate(runs) for c in plot.conditions}
    # each column's population count, from its own fit (Best and saSPT choose theirs per condition)
    n_pops = {}
    for j, (_model, plot, _caption, _payloads) in enumerate(runs):
        for pt in plot.points:
            key = (str(plot.conditions[pt.column]), j)
            n_pops[key] = max(n_pops.get(key, 0), int(pt.population))
    names, fits, where = [], [], {}
    for condition in conditions:
        wanted = set(models if shown is None else shown.get(condition, models))
        mine = [j for j, model in enumerate(models) if model in wanted and (condition, j) in has]
        # BEST ONCE (`fold_best`): a count also picked, which Best chose here, is Best's column alone
        mine, folded = fold_best(models, mine, {j: n_pops.get((condition, j)) for j in mine})
        # BY POPULATION COUNT (`fit_group_order`): 1-pop, 2-pop, Best: 3-pop -- never Best first. The SHADE stays
        # the model's (its place in the ticked order), so one model wears one shade in every condition
        shade_of = {j: k for k, j in enumerate(mine)}
        mine = fit_group_order(models, mine, {j: n_pops.get((condition, j)) for j in mine})
        for j in mine:
            shade = shade_of[j]
            n_pop = n_pops.get((condition, j))
            label = model_label(models[j], n_pop)
            where[(condition, j)] = len(names)
            names.append(side_by_side_key(condition, label))
            fit = {"condition": condition, "model": models[j], "n_pop": n_pop, "label": label, "shade": shade,
                   "shades": len(mine)}
            if folded.get(j):
                # the counts this column also stands for, and the names their own columns would have had
                fit["also"] = [models[k] for k in folded[j]]
                fit["folded"] = [side_by_side_key(condition, model_label(models[k], n_pops.get((condition, k))))
                                 for k in folded[j]]
            fits.append(fit)
    points, errors, unit_values, payloads, samples, d_scopes = [], {}, {}, {}, {}, {}
    pseudo = {k: (dict(v) if isinstance(v, dict) else v) for k, v in (first.pseudo or {}).items()}
    for j, (_model, plot, _caption, run_payloads) in enumerate(runs):
        column = {i: where.get((str(c), j)) for i, c in enumerate(plot.conditions)}
        points += [_replace(pt, column=column[pt.column]) for pt in plot.points if column[pt.column] is not None]
        errors.update({column[i]: reason for i, reason in (plot.errors or {}).items() if column.get(i) is not None})
        for condition, entry in (plot.unit_values or {}).items():
            if (str(condition), j) in where:
                unit_values[names[where[(str(condition), j)]]] = entry
        for condition, payload in (run_payloads or {}).items():
            if (str(condition), j) in where:
                payloads[names[where[(str(condition), j)]]] = payload
        # per CONDITION, the same in every model's Run that fits it: the first one that does says it
        for condition, sample in (getattr(plot, "samples", None) or {}).items():
            samples.setdefault(str(condition), sample)
        # and its Shared D: a model's Run may hold only the conditions showing it (`fitted_at`), so whether the
        # figure's conditions differ is read off the joined figure, never off one model's Run
        for condition, scope in (getattr(plot, "d_scopes", None) or {}).items():
            d_scopes.setdefault(str(condition), scope)
        for part in ("by", "real_replicates", "withheld"):
            if isinstance((plot.pseudo or {}).get(part), dict):
                pseudo.setdefault(part, {})
                for condition, value in plot.pseudo[part].items():
                    pseudo[part].setdefault(condition, value)
    points.sort(key=lambda pt: (pt.column, pt.population))
    groups = [c for c in conditions if any(f["condition"] == c for f in fits)]
    plot = _replace(first, conditions=names, points=points, errors=dict(sorted(errors.items())),
                    series_labels=[], unit_values=unit_values, fits=fits, samples=samples, pseudo=pseudo,
                    layout=_side_by_side_layout(groups, fits),
                    d_scopes={c: d_scopes[c] for c in groups if c in d_scopes})
    # each model's own counts set by hand where Best chooses fewer (`forced_count_warning`), once each
    plot.count_warnings = list(dict.fromkeys(w for _m, run, _c, _p in runs for w in (getattr(run, "count_warnings", None)
                                                                                    or ())))
    plot.warnings = fit_warnings(plot.points, plot.conditions) + group_warnings(payloads) + plot.count_warnings
    plot.warnings += (zero_jump_notes(payloads) + immobile_mix_warnings(payloads) + bound_notes(payloads)
                      + radius_notes(payloads))
    _mark_bounds(plot.points)

    bits = []
    threshold = first.threshold_um2_s
    for column, name in enumerate(names):
        pts = sorted((pt for pt in points if pt.column == column), key=lambda pt: pt.population)
        if pts and threshold is not None:
            slow = slow_fraction_below([pt.d for pt in pts], [pt.fraction for pt in pts], threshold)
            which = below_populations([pt.d for pt in pts], threshold)
            bits.append(f"{name} below-threshold {100 * slow:.1f}% ({which})")
    withheld = (list(dict.fromkeys(fits[pt.column]["condition"] for pt in points if pt.spread == "none"))
                if first.uncertainty in RESAMPLED_UNITS else [])
    fewer = [] if shown is None else [
        f"{c} ({models_words([m for m in models if m in shown.get(c, models)])} only)" for c in groups
        if set(shown.get(c, models)) != set(models)]
    lead = f"SQD populations @ D<{threshold:g}" if threshold is not None else (head or "Populations")
    caption = (f"{lead}, every condition fitted with {models_words(models)} "
               f"side by side" + (f", except {', '.join(fewer)}" if fewer else "")
               + " (a comparison of models: each column is the fit with that model; two columns of one condition are "
               "one set of replicates, never two samples)" + (" - " + ", ".join(bits) if bits else "")
               + f"   [{spread_caption(first.uncertainty, withheld, first.scope, first.pseudo, plot.points, _resamples())}]"
               + lag_caption(first.lag_frames, first.lag_ms) + precision_caption(first.precision)
               + warnings_caption(plot.warnings)
               + scopes_caption(plot) + (f"   [{folded_words(fits)}]" if folded_words(fits) else ""))
    if first.scope == "group":
        caption += f"   [{sharing_caption(first.sharing_factor, first.sharing_groups)}]"
    return plot, caption, payloads


def _compute_bubble_view(project, spec, *, uncertainty: str = "day", cache=None):
    """The SQD bubble view, as native plot data + a caption -- the same two
    things `comparison_controller.sqd_view` returns, except the first is data
    for `BubbleChart` rather than a matplotlib-drawing closure.

    Calls the SAME assembly functions `sqd_view` calls internally
    (`assemble_view`, `_overlay_table`) rather than calling `sqd_view` itself,
    so the fit runs exactly once here -- calling both would mean fitting
    twice for one Run press.

    ``uncertainty`` is one of `UNCERTAINTY_MODES`. It chose nothing before: the
    setting only switched a cell bootstrap on, which the multi-day gate then
    withheld, and nothing in this window ever read its result -- so the only
    spread ever drawn was the SD over days, whatever was picked.

    **ALSO returns a THIRD value: `payloads`, `condition ->
    comparison_controller.replicate_fit_payload(..., compute=True)`**, so a
    click on a bubble is instant; `condition_replicate_fit` can run a real
    multistart `fit_sqd` under `d_sharing_scope == "replicate"`.
    """
    from . import comparison_controller as cctrl
    from .aggregate import AggregationError
    from .comparison import ComparisonError, below_populations, slow_fraction_below
    from .observables import ObservableError
    from .sqd import SQDError

    if uncertainty not in UNCERTAINTY_MODES:
        raise ValueError(f"uncertainty must be one of {UNCERTAINTY_MODES}, "
                         f"got {uncertainty!r}")
    chunked = uncertainty == "chunk"
    if chunked != bool(getattr(spec, "pseudo_replicates", 0)):
        raise ValueError("Uncertainty 'chunk' and the spec's pseudo_replicates go together: the chunks are what "
                         "the replicate fits are made on")
    if chunked:
        uncertainty = "day"                  # drawn as the replicates are; `BubblePlotData.pseudo` says chunks
    resampled = uncertainty in RESAMPLED_UNITS
    data = cctrl.assemble_view(project, spec, "sqd", cache=cache, with_ci=resampled,
                               ci_unit=uncertainty if resampled else "cell")
    overlay, overlay_note = cctrl._overlay_table(project, spec)
    if chunked and overlay is not None:
        # The overlay summarises each arrangement over the REAL replicates (`compare.bubble_series`): drawn beside
        # chunks it would label replicate means as chunk means.
        raise ValueError("Uncertainty 'Chunks' draws one fit arrangement: switch the overlay off")

    payloads: dict[str, dict] = {}
    # The replicate payload is what a bubble click opens. A condition that cannot make one HAS AN ERROR ENTRY naming
    # why, ``{"error": "SQDError: ..."}``, which the detail page shows -- but only for the failures the spine names (a
    # fit it refuses, displacements it cannot form, a comparison or an aggregation it cannot build). Any other failure
    # is a bug and is raised. It once swallowed ANY error, and then skipped even the named ones without a word, so a
    # click opened a page that could not say why it was empty.
    for condition in spec.conditions:
        try:
            # WITH THE CACHE, so this is a read of what the bubble fit just made.
            # Without one it fitted every condition a SECOND time, the state-count
            # choice included: measured, 4 BIC selections for 2 conditions.
            payloads[condition] = cctrl.replicate_fit_payload(
                project, spec, condition, cache=cache, compute=True)
        except (SQDError, ObservableError, ComparisonError, AggregationError) as exc:
            payloads[condition] = {"error": f"{type(exc).__name__}: {exc}"}
    scope = data.get("d_sharing_scope", "condition")
    weighting = data.get("fraction_weighting", "track")
    threshold = spec.threshold_um2_s

    if overlay is None:
        columns = data["columns"]
        conditions = [col.condition for col in columns]
        points: list[BubblePoint] = []
        errors: dict[int, str] = {}
        unit_values: dict[str, dict] = {}
        pseudo_by: dict[str, str] = {}
        pseudo_ks: dict[str, int] = {}
        pseudo_real: dict[str, int] = {}
        pseudo_held: dict[str, str] = {}
        bits, withheld = [], []
        for i, col in enumerate(columns):
            if not col.D:
                errors[i] = getattr(col, "error", "not fit")
                continue
            n_rep = int(getattr(col, "n_replicates", 1))
            # CHUNKS WITHHELD on a condition with replicates (`comparison.chunks_withheld`, ruled 4 Nov 2026): drawn
            # as its pooled fit with no spread, as a withheld Movies or Cells interval is; the reason travels in
            # `BubblePlotData.pseudo["withheld"]` for the caption and the brackets
            held = (payloads.get(col.condition) or {}).get("chunks_withheld") if chunked else None
            if held:
                pseudo_held[col.condition] = held
            if uncertainty == "day" and not held:
                Ds = list(col.D)
                fractions = list(col.fraction_mean or col.fractions)
                sds = list(col.fraction_sd or [0.0] * len(Ds))
                intervals = [None] * len(Ds)
                spread = "sd"
                d_sds = list(getattr(col, "D_sd", None) or [0.0] * len(Ds))
                d_intervals = [None] * len(Ds)
                per_day = [[float(f) for f in day]
                           for day in (getattr(col, "per_replicate_fractions", None) or [])]
                if per_day:
                    # The values whose MEAN is the bubble just drawn (`_column_from_payload`),
                    # kept for a significance bracket: never a pooled or per-cell number.
                    chunks = (payloads.get(col.condition) or {}).get("pseudo")
                    if chunks:
                        pseudo_by[col.condition] = chunks["by"]
                        pseudo_ks[col.condition] = int(chunks.get("k") or spec.pseudo_replicates)
                        pseudo_real[col.condition] = int(chunks["real_replicates"])
                    unit_values[col.condition] = {
                        "unit": ("chunk" if chunks else "day") if len(per_day) >= 2 else "cell",
                        # every other deal of the chunks, for how far a p moves with the deal
                        **({"splits": chunks["splits"]} if chunks else {}),
                        "replicates": _replicate_labels(payloads.get(col.condition),
                                                        len(per_day)),
                        "fractions": per_day, "D": [float(d) for d in Ds],
                        # each day's share below the D threshold, as the fit computed it
                        "below": _below_values(payloads.get(col.condition), per_day, Ds,
                                               threshold),
                        # each cell's fractions at the same D: the nested mixed model's data
                        "cells": getattr(col, "per_cell", None) or [],
                        "cells_left_out": int(getattr(col, "cells_left_out", 0) or 0)}
            else:
                # THE POOLED FIT, never a silent fall-back to the day mean: a day mean
                # drawn under "None" would carry the wrong label.
                if not col.pooled_fractions or not col.pooled_D:
                    errors[i] = "no pooled fit in the payload"
                    continue
                Ds = list(col.pooled_D)
                fractions = list(col.pooled_fractions)
                sds = [0.0] * len(Ds)
                ci = getattr(col, "fraction_ci", None) if resampled else None
                intervals = list(ci) if ci else [None] * len(Ds)
                spread = "ci" if ci else "none"
                d_sds = [0.0] * len(Ds)
                d_ci = getattr(col, "D_ci", None) if (resampled and ci) else None
                d_intervals = list(d_ci) if d_ci else [None] * len(Ds)
                if (resampled and not ci) or held:
                    withheld.append(col.condition)
            for k, D in enumerate(Ds):
                points.append(BubblePoint(
                    # a D of 0 is the immobile population of exact zeros, drawn at its "0" tick (`chart_text.log_drawn`)
                    column=i, x_offset=0.0, d=(0.0 if float(D) == 0 else max(float(D), 1e-6)),
                    fraction=float(fractions[k]),
                    fraction_sd=float(sds[k]) if k < len(sds) else 0.0,
                    n_replicates=n_rep, population=k + 1, style_index=0,
                    spread=spread,
                    interval=intervals[k] if k < len(intervals) else None,
                    d_sd=float(d_sds[k]) if k < len(d_sds) and d_sds[k] is not None else 0.0,
                    d_interval=d_intervals[k] if k < len(d_intervals) else None))
            slow = slow_fraction_below(Ds, fractions, threshold)
            bits.append(f"{col.condition}: {col.n_pop}-pop, below-threshold "
                        f"{100 * slow:.1f}% ({below_populations(Ds, threshold)})")
        lag, lag_ms = _lag(project, spec)
        plot = BubblePlotData(
            precision=precision_of(spec),
            conditions=conditions, points=points, errors=errors,
            series_labels=[], scope=scope, fraction_weighting=weighting,
            uncertainty=uncertainty, unit_values=unit_values,
            threshold_um2_s=float(threshold), lag_frames=lag, lag_ms=lag_ms,
            samples=_samples(conditions, points, payloads, uncertainty),
            # each fitted column's Shared D, as its payload was fitted (a row's own, or the Run's)
            d_scopes={str(col.condition): str(col.scope) for i, col in enumerate(columns)
                      if col.D and i not in errors})
        if chunked:
            from .comparison import PSEUDO_SEED, PSEUDO_SPLITS

            plot.pseudo = {"k": int(spec.pseudo_replicates), "seed": PSEUDO_SEED, "splits": PSEUDO_SPLITS,
                           "by": pseudo_by, "ks": pseudo_ks, "real_replicates": pseudo_real,
                           "withheld": pseudo_held}
        if scope == "group":
            plot.sharing_factor = str(data.get("d_sharing_factor") or "")
            plot.sharing_groups = dict(data.get("d_sharing_groups") or {})
        # A COUNT SET BY HAND WHERE BEST (BIC) CHOOSES FEWER (ruled 4 Nov 2026): said, the count kept
        plot.count_warnings = cctrl.count_warnings(project, spec, [c for i, c in enumerate(conditions)
                                                                   if i not in errors], cache=cache)
        plot.warnings = fit_warnings(plot.points, plot.conditions) + group_warnings(payloads) + plot.count_warnings
        plot.warnings += (zero_jump_notes(payloads) + immobile_mix_warnings(payloads) + bound_notes(payloads)
                          + radius_notes(payloads))
        _mark_bounds(plot.points)
        caption = (f"SQD populations @ D<{threshold:g} - {', '.join(bits)}"
                   f"   [{spread_caption(uncertainty, withheld, scope, plot.pseudo, plot.points, _resamples(spec))}]"
                   + lag_caption(lag, lag_ms)
                   + precision_caption(plot.precision) + warnings_caption(plot.warnings) + scopes_caption(plot))
        if scope == "group":
            caption += f"   [{sharing_caption(plot.sharing_factor, plot.sharing_groups)}]"
        at_cap = at_best_cap(spec, columns)
        if at_cap:
            caption += (f"   [Best reached its limit of {spec.best_max_pop} for "
                        f"{', '.join(at_cap)}: a larger count was not tested. Tick "
                        f"'Best (BIC) also tries 4 populations' (Compute, Fits page) to test it]")
        if overlay_note:
            caption += f"   [{overlay_note}]"
        return plot, caption, payloads

    # IN THE RUN's ORDER (the Conditions tool's, `condition_order`), as every other figure: the overlay table comes
    # sorted by name, and its columns were drawn alphabetically (3 Nov 2026, reported)
    from .condition_order import apply as order_apply

    conditions = order_apply(list(dict.fromkeys(overlay["condition"])), [str(c) for c in spec.conditions])
    series_names = list(dict.fromkeys(overlay["series"]))
    cond_index = {c: i for i, c in enumerate(conditions)}
    n_series = len(series_names)
    span = 0.0 if n_series == 1 else 0.34
    offsets = {
        name: (0.0 if n_series == 1 else -span + 2 * span * si / (n_series - 1))
        for si, name in enumerate(series_names)}
    points = []
    for row in overlay.itertuples():
        sd = getattr(row, "fraction_sd", float("nan"))
        points.append(BubblePoint(
            column=cond_index[row.condition], x_offset=offsets[row.series],
            # a D of 0 is the immobile population of exact-zero jumps, drawn at its "0" tick (`chart_text.log_drawn`)
            d=(0.0 if float(row.D_um2_per_s) == 0 else max(float(row.D_um2_per_s), 1e-6)), fraction=float(row.fraction),
            fraction_sd=(0.0 if math.isnan(sd) else float(sd)),
            n_replicates=int(row.n_units), population=int(row.population),
            style_index=series_names.index(row.series)))
    lag, lag_ms = _lag(project, spec)
    # every series is a mean over the replicates, whatever the Uncertainty setting (said in the caption below)
    plot = BubblePlotData(
        conditions=conditions, points=points, errors={},
        series_labels=series_display_names(series_names),
        scope=scope, fraction_weighting=weighting, uncertainty=uncertainty, lag_frames=lag, lag_ms=lag_ms,
        samples={str(c): _sample(payloads.get(c), "sd", "day") for c in conditions})
    plot.warnings = (fit_warnings(plot.points, plot.conditions, plot.series_labels) + bound_notes(payloads)
                     + radius_notes(payloads))
    _mark_bounds(plot.points)

    slow = overlay[overlay["population"] == 1]
    shown = dict(zip(series_names, series_display_names(series_names)))
    bits = "; ".join(
        f"{shown[name]}: " + ", ".join(
            f"{r.condition} {100.0 * r.fraction:.1f}%"
            for r in sorted(slow[slow["series"] == name].itertuples(),
                            key=lambda r: cond_index.get(r.condition, len(cond_index))))
        for name in series_names)
    # THE COMPARISON IS ALWAYS PER DAY, and says so. Each series is its own fit
    # summarised over replicate days; there is no pooled or cell-interval version of
    # it, so the Uncertainty setting cannot apply here and must not appear to.
    bits += ("   [each series: mean over replicates ± SD, whatever the "
             "Uncertainty setting]")
    if len(set(overlay["n_pop"])) > 1:
        bits += ("   [every condition at " + " and at ".join(str(n) for n in sorted(set(overlay["n_pop"])))
                 + " populations, a comparison of models: the Fits page's per-condition choice is not "
                   "used here]")
    if overlay_note:
        bits += f"   [{overlay_note}]"
    return plot, f"SQD populations, fits overlaid - {bits}", payloads


def _below_values(payload, per_day, Ds, threshold) -> list:
    """Each day's share below the D threshold: the replicate fit's own ``slow_fraction`` (at
    that day's D under the replicate scope), else from the shared D."""
    from .comparison import slow_fraction_below

    reps = (payload or {}).get("replicates") if isinstance(payload, dict) else None
    if reps and len(reps) == len(per_day) and all("slow_fraction" in r for r in reps):
        return [float(r["slow_fraction"]) for r in reps]
    return [slow_fraction_below(Ds, day, threshold) for day in per_day]


def _replicate_labels(payload, n: int) -> list:
    """The day labels of a condition's ``n`` replicates, from its replicate payload (the same
    sorted days `condition_replicate_fit` fitted), else 1..n. Labels only: a day number is
    an ordinal WITHIN its condition and is never matched to another condition's."""
    reps = (payload or {}).get("replicates") if isinstance(payload, dict) else None
    if reps and len(reps) == n:
        return [rep.get("replicate") for rep in reps]
    return list(range(1, n + 1))


def at_best_cap(spec, columns) -> list:
    """Conditions where "Best" chose the LARGEST count it was allowed to.

    That is the one case where the cap may have decided the answer rather than the
    data: had a larger count been allowed, BIC might have preferred it. Anywhere below the cap the answer would not have changed.
    """
    cap = int(getattr(spec, "best_max_pop", 4))
    if cap >= 4:
        return []
    return [col.condition for col in columns
            if col.D and spec.model_for(col.condition) == "best"
            and int(col.n_pop) == cap]


#: A fitted population this small is not kept: fit one population fewer (ruled 17 Oct 2026: "if we have a small pop
#: of less than 5 percent we switch to a lower powered fit"). Its D is poorly determined, and with D shared by a
#: condition's days that error moves every day together (SynR: 11.7 % false positives at 4 %).
SMALL_POPULATION = 0.05
#: Two populations whose D are closer than this factor are poorly separated by a mixture of exponentials: the split
#: between them is not well determined (asked 17 Oct 2026; "within a factor of 2").
SIMILAR_D_RATIO = 2.0


# `scope_words` -- Force same D across in words, for every caption, table and receipt that names it -- is
# `workspace_view.scope_words`, imported above: the ONE spelling ("same D across replicates", "same D across
# conditions, per protein", "same D across the whole data set", "D not forced: each replicate its own"; ruled 1 Nov
# 2026).


#: Under Force same D across Conditions, a condition fitted ALONE whose D differs from the group's shared D by this
#: factor or more, population for population at the group's count, is named (`group_warnings`): sharing may be moving a
#: real change in D into its fractions. A heuristic, said as one; a likelihood test would call any difference real
#: at these numbers of displacements.
GROUP_D_RATIO = 1.5


def group_warnings(payloads) -> list:
    """Under Force same D across Conditions: each condition whose own D (fitted alone at the group's count, the
    payload's ``own_D``) is `GROUP_D_RATIO`-fold or more from the shared D, with what to do -- compare with
    Replicates."""
    from .chart_text import fold_words

    out = []
    for condition, payload in (payloads or {}).items():
        if not isinstance(payload, dict) or not payload.get("own_D") or not payload.get("shared_D"):
            continue
        shared, own = sorted(payload["shared_D"]), sorted(payload["own_D"])
        group = payload.get("group") or {}
        # EACH OWN D AGAINST THE SHARED D NEAREST IT (on a log scale), never by index (found by the overnight tester,
        # 4 Nov 2026: a free protein's own populations both sit at the fast D, and its "population 1" at 0.816 was
        # set against the shared SLOW D, 0.0516 -- "15.8-fold" -- while it sat on the shared fast one, 0.796)
        worst = None
        for o in own:
            if o <= 0 or not any(s > 0 for s in shared):
                continue
            k, s = min(((i, s) for i, s in enumerate(shared, start=1) if s > 0),
                       key=lambda pair: abs(math.log(pair[1] / o)))
            ratio = max(s, o) / min(s, o)
            if ratio >= GROUP_D_RATIO and (worst is None or ratio > worst[0]):
                worst = (ratio, k, s, o)
        if worst is None:
            continue
        ratio, k, s, o = worst
        # the whole data set is no group (Force same D across: Whole data set records neither a factor nor a name)
        where = " ".join(str(v) for v in (group.get("factor"), group.get("name")) if v) or "the whole data set"
        out.append(f"{condition}: fitted alone, population {k} has D {o:.3g} µm²/s against {s:.3g} shared by "
                   f"{where} ({fold_words(ratio, 1)} apart); sharing may move a change in D into the fractions. "
                   f"Compare with {setting_words('condition')}")
    return out


def fit_checks(points, conditions, series_labels=()) -> list:
    """The "Check the fit" rule as rows, ``[{"column", "style", "name", "populations", "reason"}]``: per condition (and
    series), a population holding under `SMALL_POPULATION`, and neighbouring populations whose D are within
    `SIMILAR_D_RATIO`. ``populations`` are the population numbers the flag is about (one, or the two of a pair).
    ``points`` are `BubblePoint`s or their plain dicts (the SQD and the apparent-D bubbles alike). ONE rule:
    `fit_warnings` words it for the chart, `significance.bracket_warnings` for the brackets standing on a flagged
    population (ruled 4 Nov 2026)."""
    from .chart_text import fold_words

    groups: dict = {}
    for p in points:
        get = p.get if isinstance(p, dict) else (lambda k, p=p: getattr(p, k))
        groups.setdefault((int(get("column")), int(get("style_index") or 0)), []).append(
            (int(get("population")), float(get("d")), float(get("fraction"))))
    labels = list(series_labels or ())
    out = []
    for (column, style), pops in sorted(groups.items()):
        if len(pops) < 2:
            continue
        name = str(conditions[column]) + (f" ({labels[style]})" if len(labels) > 1 and style < len(labels) else "")
        pops.sort(key=lambda r: r[1])
        for k, d, f in pops:
            # the immobile population of exact-zero jumps (D 0) is counted, not fitted: fitting fewer cannot drop it
            if f < SMALL_POPULATION and d != 0:
                out.append({"column": column, "style": style, "name": name, "populations": (k,),
                            "reason": f"population {k} holds {100 * f:.1f} %, under {100 * SMALL_POPULATION:.0f} %"})
        for (k1, d1, _f1), (k2, d2, _f2) in zip(pops, pops[1:]):
            if d1 > 0 and d2 / d1 < SIMILAR_D_RATIO:
                out.append({"column": column, "style": style, "name": name, "populations": (k1, k2),
                            "reason": f"populations {k1} and {k2} have similar D ({d1:.3g} and {d2:.3g} µm²/s, under "
                                      f"{fold_words(SIMILAR_D_RATIO)} apart)"})
    return out


def zero_jump_notes(payloads) -> list:
    """Each condition whose fit met jumps of length exactly 0 (`sqd.zero_jumps`, the payload's ``zero_jumps``), with
    what was done: an immobile population counted beside the fitted ones, or (coordinates on a lattice) the zeros kept
    in the fit as any jump. Shown with the fit's warnings, so the figure and its caption say it too. The apparent-D
    bubbles carry the same (`appd_fits`, a gamma fit's exact-zero tracks); a joint fit's one decision, shared by its
    conditions, is said once with all of them named."""
    notes: dict = {}
    joint: set = set()
    for condition, payload in (payloads or {}).items():
        zeros = payload.get("zero_jumps") if isinstance(payload, dict) else None
        if zeros and zeros.get("note"):
            notes.setdefault(str(zeros["note"]), []).append(str(condition))
            if payload.get("scope") in ("global", "group"):
                joint.add(str(zeros["note"]))
    # a joint fit's decision counts its group's pooled tracks: said as theirs together, not as each one's
    return [f"{', '.join(conditions)}{' (fitted together)' if note in joint and len(conditions) > 1 else ''}: {note}"
            for note, conditions in notes.items()]


def bound_notes(payloads) -> list:
    """EACH CONDITION WHOSE FITTED D SITS ON THE FIT'S BOUND (7 Nov 2026, the per-spot audit's finding 1: per spot,
    SynR's bound population sat on the lower bound, 0.0001 µm²/s, in every condition and replicate). Such a D is where
    the optimiser stopped, not a value the data chose: not resolved. Its share stands (at the bound 4Dτ is about 8 nm²
    against a noise term of thousands). Read off each column's payload (`comparison.condition_replicate_fit`): each
    replicate's own D under Don't force anywhere, else the shared or pooled D, against `sqd.D_BOUNDS`, the bounds every
    fit of the Run is made within. Shown with the fit's warnings, so the chart, its caption and the exported legend
    say it; conditions with the same note are named together."""
    from .sqd import D_BOUNDS, at_bound

    notes: dict = {}
    for condition, payload in (payloads or {}).items():
        if not isinstance(payload, dict) or "error" in payload:
            continue
        reps = [r["D"] for r in payload.get("replicates") or () if isinstance(r, dict) and r.get("D")]
        if payload.get("scope") == "replicate" and reps:
            sides = [at_bound(d) for d in reps]
            for k in range(max(len(s) for s in sides)):
                for side in ("lower", "upper"):
                    hit = sum(1 for s in sides if k < len(s) and s[k] == side)
                    if hit:
                        where = "every replicate" if hit == len(sides) else f"{hit} of {len(sides)} replicates"
                        notes.setdefault((k + 1, side, where), []).append(str(condition))
        else:
            D = payload.get("shared_D") or (payload.get("pooled") or {}).get("D") or []
            for k, side in enumerate(at_bound(D)):
                if side:
                    notes.setdefault((k + 1, side, ""), []).append(str(condition))
    out = []
    for (k, side, where), conditions in notes.items():
        edge, word = (D_BOUNDS[0], "under") if side == "lower" else (D_BOUNDS[1], "over")
        out.append(f"{'; '.join(conditions)}: population {k}'s D sits on the fit's {side} bound ({edge:g} µm²/s"
                   + (f", in {where}" if where else "") + f"): not resolved, so read it as {word} {edge:g}, not as a "
                   "value; its share stands")
    return out


#: A population whose D the linking radius reads at least this much low is named (`radius_notes`).
RADIUS_NOTE_SHARE = 0.05


def radius_notes(payloads) -> list:
    """THE LINKING RADIUS CUTS THE FAST STEPS, SAID (10 Nov 2026, the per-spot audit's finding 6). The SQD likelihood
    is not normalised for steps past the radius, which break the track, so a fast population's D reads low; where it
    reads at least `RADIUS_NOTE_SHARE` low, the population is named with the D its steps would have
    (`trackmate_run.linking_corrected_D`, from the condition's own radius, pixel size and frame interval: the
    payload's ``linking``), or "at least" the D where it is at what the radius can read at all. No fit changes."""
    from .trackmate_run import linking_corrected_D

    out = []
    for condition, payload in (payloads or {}).items():
        if not isinstance(payload, dict) or "error" in payload or not payload.get("linking"):
            continue
        link = payload["linking"]
        D = payload.get("shared_D") or (payload.get("pooled") or {}).get("D") or []
        for k, d in enumerate(D):
            d = float(d)
            if not d > 0:
                continue
            true, kind = linking_corrected_D(d, radius_px=link["radius_px"], tau_s=link["tau_s"],
                                             pixel_size_nm=link["pixel_size_nm"])
            radius = f"{link['radius_px']:g} px"
            if kind == "bound":
                out.append(f"{condition}: population {k + 1}'s D ({d:.3g} µm²/s) is at what a {radius} linking "
                           f"radius can read at all: read it as at least {d:.3g}, not as a value")
            elif true / d - 1.0 >= RADIUS_NOTE_SHARE:
                out.append(f"{condition}: population {k + 1}'s D reads about {100 * (1 - d / true):.0f} % low "
                           f"({d:.3g} against about {true:.3g} µm²/s): steps past the {radius} linking radius break "
                           "the track and the fit does not model it; its share stands")
    return out


def immobile_mix_warnings(payloads, noun: str = "jumps") -> list:
    """FORCE SAME D ACROSS CONDITIONS OR THE WHOLE DATA SET, JOINING CONDITIONS WITH EXACT ZEROS TO CONDITIONS WITHOUT
    (ruled 4 Nov 2026: "A WARNING, not a refusal ... name those conditions and say their slowest population is empty
    by construction, on screen and in the exported legend"). A joint fit holds the immobile population of exact zeros
    (D 0, `sqd.zero_jumps`, `appd_gamma.zero_tracks`) for every condition it joins, each at its own share: a truth
    condition's is its bound share, a measured condition's 0 -- by construction, not by measurement -- so its "3
    populations" are 2 that move. Read off each column's payload (the SQD replicate fit's, `comparison.
    condition_replicate_fit`; the apparent-D design, `appd_fits`): its shared D holds a 0, and its pooled fraction of
    that population is 0 (none of its exact zeros) or not. One warning per joined group, every such condition named.
    ``noun``: "jumps" (SQD) or "tracks" (apparent D)."""
    from .workspace_view import setting_words

    groups: dict = {}
    for name, payload in (payloads or {}).items():
        if not isinstance(payload, dict) or payload.get("scope") not in ("global", "group"):
            continue
        shared = [float(d) for d in payload.get("shared_D") or ()]
        pooled = payload.get("pooled") or {}
        fractions = pooled.get("fractions", pooled.get("fraction"))
        if not shared or 0.0 not in shared or not fractions or len(fractions) != len(shared):
            continue
        group = payload.get("group") or {}
        members = list(group.get("with") or group.get("members") or ())
        if payload["scope"] == "group" and len(members) < 2:
            continue                                 # alone in its group: its own D, nothing joined
        key = (payload["scope"], str(group.get("factor") or ""), str(group.get("name") or ""))
        side = "without" if float(fractions[shared.index(0.0)]) == 0 else "with"
        groups.setdefault(key, {"with": [], "without": []})[side].append(str(name))
    out = []
    for (scope, factor, group_name), sides in groups.items():
        if not (sides["with"] and sides["without"]):
            continue
        where = setting_words(scope, factor=factor) + (f" ({group_name})" if scope == "group" and group_name else "")
        without = ", ".join(sides["without"])
        one = len(sides["without"]) == 1
        out.append(f"{where} joins conditions with exact-zero {noun} ({', '.join(sides['with'])}) and without "
                   f"({without}): the shared fit holds an immobile population (D 0) for every one of them, so in "
                   f"{without} the slowest population is empty by construction (0 %), not measured, and "
                   f"{'its' if one else 'their'} populations that move are one fewer than the count. Compare with "
                   f"{setting_words('condition')}")
    return out


def fit_warnings(points, conditions, series_labels=()) -> list:
    """What to check before trusting a drawn fit (`fit_checks`), in words. Each warning ends with what to do: fit one
    population fewer."""
    return [f"{c['name']}: {c['reason']}; fit one population fewer"
            for c in fit_checks(points, conditions, series_labels)]


def bic_delta_text(delta: float) -> str:
    """A BIC difference as every sentence prints it: "37.4", or "1,520" from 100 up."""
    return f"{delta:,.0f}" if delta >= 100 else f"{delta:.1f}"


def forced_count_warning(n_set, bic, noun: str = "population") -> str | None:
    """A COUNT SET BY HAND WHERE BEST (BIC) CHOOSES FEWER (ruled 4 Nov 2026, the overnight tester's round 3: a forced
    2-state HMM map of a one-state protein drew an 8 % state threefold slower and nothing said so): "Best (BIC)
    chooses 1 state here, BIC preferring 1 over 2 by ΔBIC = 7.6; kept at the 2 you set" -- or None where Best
    chooses as many or more, or there are no scores. ``bic``: ``[[n, BIC], ...]`` over the counts Best compares
    (the same counts its own choice is made among); ``noun`` "population" or "state". THE USER'S COUNT IS NEVER
    OVERRIDDEN: this only says so. ONE sentence for the SQD, apparent-D and HMM maps and sections."""
    scores = {int(n): float(b) for n, b in (bic or ())}
    if n_set is None or int(n_set) not in scores or len(scores) < 2:
        return None
    n = int(n_set)
    best = min(scores, key=lambda m: scores[m])
    if best >= n:
        return None
    word = noun if best == 1 else f"{noun}s"
    return (f"Best (BIC) chooses {best} {word} here, BIC preferring {best} over {n} by ΔBIC = "
            f"{bic_delta_text(scores[n] - scores[best])}; kept at the {n} you set")


def forced_count_scores(n_set, counts, select, own_bic=None) -> list | None:
    """THE CHECK BESIDE A COUNT SET BY HAND, exact and no wider than it must be (4 Nov 2026): ``[[n, BIC], ...]`` for
    `forced_count_warning`, or None where none of ``counts`` is fewer than ``n_set``. ONE rule for the SQD, apparent-D
    and HMM maps and the SQD and HMM sections.

    ``counts``: the counts Best chooses among (its range; the count set is added), consecutive. ``select(lo, hi)``:
    ``[[n, BIC], ...]`` for the counts ``lo..hi``, each fitted EXACTLY as Best's own selection fits it (the same data,
    likelihood, weighting, starts and seed: every count's fit is independent of which others are fitted, so a score
    here is the score in Best's own table). ``own_bic``: the set count's BIC where the fit drawn at it IS Best's fit
    at that count (the HMM's `fit_hmm`, apparent D's gamma mixture); None where it is not (an SQD map is drawn from
    least squares on the ECDF, the SQD bubbles from their own fit), and the set count is then fitted with the counts
    below it.

    Best chooses FEWER only if a count below beats the set count AND every count above it. So the counts below are
    fitted first, and the counts above only where one below wins: a count Best agrees with, or would raise, costs the
    counts below alone (one population or state at a set two: nearly free), and only a warning costs Best's whole
    selection. Never a sentence naming a count Best would not choose, and never a sentence missed."""
    n = int(n_set)
    counts = sorted({int(c) for c in counts} | {n})
    below, above = [c for c in counts if c < n], [c for c in counts if c > n]
    if not below:
        return None
    scores = {int(m): float(b) for m, b in select(below[0], n if own_bic is None else below[-1])}
    if own_bic is not None:
        scores[n] = float(own_bic)
    if above and n in scores and min(scores, key=scores.get) < n:
        scores.update({int(m): float(b) for m, b in select(above[0], above[-1])})
    return [[m, scores[m]] for m in sorted(scores)]


def warning_line(warnings) -> str:
    """The flag line under a bubble chart: "Check the fit: a · b", or "" when there is nothing to check. The SQD
    section's fit warnings and, beside them, the brackets standing on a flagged population (`significance
    .bracket_warnings`)."""
    warnings = [str(w) for w in warnings or () if w]
    return "Check the fit: " + " · ".join(warnings) if warnings else ""


def warnings_caption(warnings) -> str:
    """The warnings, for a caption ("" when there are none)."""
    return "".join(f"   [check the fit: {w}]" for w in warnings or ())


def _lag(project, spec) -> tuple:
    """``(lag in frames, lag in ms or None)`` for a Run's spec (`ComparisonSpec.sqd_lag`); lags 1-4 together are
    ``("1-4", None)``."""
    from .comparison import JOINT_LAG

    if getattr(spec, "sqd_lag", 1) == JOINT_LAG:
        return JOINT_LAG, None
    lag = int(getattr(spec, "sqd_lag", 1))
    params = getattr(project, "params", None)
    interval = getattr(params, "frame_interval_ms", None)
    return lag, (None if interval is None else float(interval) * lag)


#: The crossed plot's marker per population, slowest first (colour is the condition).
CROSSED_MARKERS = ("circle", "square", "diamond", "triangle")


#: Room left of the lowest and right of the highest fraction on the crossed plot, in percentage points: a marker at
#: 0 or 100 % is drawn whole inside the plot.
CROSSED_PAD = 4.0


def crossed_payload(data: "BubblePlotData", labels: dict | None = None) -> dict:
    """v2's crossed plot of the SQD populations (`SMTracker.m:5019-5073`): each population of each condition one
    point, fraction (%) across and D (µm²/s, log) up, with a bar each way WHERE THE RUN HAS A SPREAD -- across, the
    fraction's (SD over days, or the cells' 95 % interval); up, D's own (the SD of each day's D under Don't force
    anywhere, the
    cells' interval under Uncertainty = Cells), none where D is shared across days. ``{"available": False, "why"}``
    with an overlay (one arrangement only) or before a fit."""
    from .chart_text import D_AXIS_LABEL, log_drawn, zero_d_stand_in, zero_d_ticks

    labels = labels or {}
    if data is None or not data.points:
        return {"available": False, "why": "Run to draw the populations."}
    if data.series_labels:
        return {"available": False, "why": "The fraction-against-D view draws one arrangement: turn Overlay off."}
    # a D of 0 (the immobile population of exact-zero jumps) is drawn at a "0" tick under the rest: "y" keeps the D,
    # the drawers place it with `chart_text.log_drawn` at ``zero_d``
    zero_d = zero_d_stand_in(pt.d for pt in data.points)
    points, x_hi, x_lo, ys = [], 0.0, 100.0, []
    for pt in data.points:
        x = 100.0 * pt.fraction
        if pt.spread == "ci" and pt.interval is not None:
            x_bar = (100.0 * pt.interval[0], 100.0 * pt.interval[1])
        elif pt.spread == "sd" and pt.n_replicates >= 2 and pt.fraction_sd > 0:
            x_bar = (max(0.0, x - 100.0 * pt.fraction_sd), min(100.0, x + 100.0 * pt.fraction_sd))
        else:
            x_bar = None
        if pt.d == 0:
            y_bar = None                                        # D 0 exactly: counted, never fitted
        elif pt.d_interval is not None:
            y_bar = (max(float(pt.d_interval[0]), 1e-6), float(pt.d_interval[1]))
        elif pt.d_sd > 0 and pt.n_replicates >= 2:
            y_bar = (max(pt.d - pt.d_sd, pt.d / 10.0), pt.d + pt.d_sd)
        else:
            y_bar = None
        cond = data.conditions[pt.column]
        points.append({"condition": cond, "name": str(labels.get(cond, cond)), "population": int(pt.population),
                       "marker": CROSSED_MARKERS[(pt.population - 1) % len(CROSSED_MARKERS)],
                       "x": x, "x_bar": x_bar, "y": float(pt.d), "y_bar": y_bar})
        x_hi = max(x_hi, x_bar[1] if x_bar else x)
        x_lo = min(x_lo, x_bar[0] if x_bar else x)
        ys.extend([log_drawn(pt.d, zero_d), *(y_bar or ())])
    lo, hi = math.log10(min(ys)), math.log10(max(ys))
    lo, hi = math.floor((lo - 0.15) * 2) / 2, math.ceil((hi + 0.15) * 2) / 2
    y_ticks = [10 ** e * m for e in range(int(math.floor(lo)), int(math.ceil(hi)) + 1) for m in (1, 3)
               if lo <= math.log10(10 ** e * m) <= hi]
    labelled = zero_d_ticks(y_ticks, zero_d)
    top = min(100.0, 10.0 * math.ceil((x_hi + 4.0) / 10.0))
    n_max = max(p["population"] for p in points)
    if data.uncertainty in RESAMPLED_UNITS:
        across = (f"the 95 % interval from resampling {UNIT_WORDS[data.uncertainty]}"
                  if any(p["x_bar"] for p in points) else "none (pooled)")
    elif data.uncertainty == "day":
        across = (f"± SD over {chunk_words(data.pseudo, short=True)} (pseudo-replicates)" if data.pseudo
                  else "± SD over replicates")
    else:
        across = "none (every replicate pooled)"
    if any(p["y_bar"] for p in points):
        up = (f"the 95 % interval from resampling {UNIT_WORDS[data.uncertainty]}"
              if data.uncertainty in RESAMPLED_UNITS else "± SD of each replicate's own D")
        if mixed_scopes(data) and data.uncertainty not in RESAMPLED_UNITS:
            # A SHARED D PER CONDITION: only the conditions whose replicates each have their own D carry the bar
            up += (f", for the conditions with {scope_words('replicate')}; none for those with the "
                   f"{scope_words('condition')} (one value)")
    elif data.scope in ("condition", "global", "group") and data.uncertainty == "day":
        up = "none: D is shared across a condition's replicates, one value" + (
            " for all conditions" if data.scope == "global" else
            f" for all conditions of each {data.sharing_factor or 'group'}" if data.scope == "group" else "")
    else:
        up = "none"
    return {"available": True, "points": points,
            "legend_conditions": [(c, str(labels.get(c, c))) for c in dict.fromkeys(p["condition"] for p in points)],
            "legend_populations": [(CROSSED_MARKERS[(k - 1) % len(CROSSED_MARKERS)],
                                    f"population {k}" + (" (slowest)" if k == 1 and n_max > 1 else ""))
                                   for k in range(1, n_max + 1)],
            # A POINT AT 100 % (a one-population fit) or 0 % sat ON the plot's edge, half outside it (3 Nov 2026,
            # reported): the range reaches past the data by CROSSED_PAD points; the ticks stay 0 to the top
            "x_range": (min(0.0, x_lo - CROSSED_PAD), max(top, x_hi + CROSSED_PAD)),
            "x_ticks": [v for v in range(0, int(top) + 1, 10 if top <= 60 else 20)],
            "y_range": (10 ** lo, 10 ** hi), "y_ticks": [v for v, _label in labelled],
            "y_tick_labels": [label for _v, label in labelled], "zero_d": zero_d,
            "x_label": "fraction (%)", "y_label": D_AXIS_LABEL,
            "caption": f"Horizontal bars: {across}. Vertical bars: {up}.",
            "columns": ["Condition", "Population", "Fraction (%)", "Fraction from", "Fraction to", "D (µm²/s)",
                        "D from", "D to"]}


def sharing_caption(factor: str, groups: dict) -> str:
    """Which conditions share a D under Force same D across Conditions, for a caption: "Same D across conditions,
    per protein: protein A (protein A glucose, protein A malate); protein B (protein B glucose)"."""
    members: dict = {}
    for condition, group in (groups or {}).items():
        members.setdefault(str(group), []).append(str(condition))
    listed = "; ".join(f"{g} ({', '.join(cs)})" for g, cs in members.items())
    head = scope_words("group", factor)
    head = head[:1].upper() + head[1:]
    return f"{head}: {listed}" if listed else head


def precision_of(spec) -> str:
    """The calibration's words when the Run fitted each step's own localisation error (`ComparisonSpec.sqd_precision`),
    else ""."""
    precision = getattr(spec, "sqd_precision", None)
    if precision is None:
        return ""
    off = tuple(getattr(spec, "sqd_precision_off", ()) or ())
    return str(precision[2]) + (f"; not modelled for {', '.join(off)}: their tracks carry no per-spot precision (a "
                                f"track table; a truth condition has no localisation error)" if off else "")


#: WHAT A PER-SPOT D IS, in every caption, table and legend that says it (7 Nov 2026, the per-spot audit's finding 3;
#: the experimenter: "relabel"). The noise model takes the localisation error out; motion blur stays in: the Berglund correction,
#: which would rescale by 1 / (1 - 2R), is refused beside per spot (it would take the localisation error off twice),
#: so a moving population's D is still apparent. Was "D corrected, not apparent", which said more than was done.
PER_SPOT_D_IS = "corrected for localisation error, not for motion blur"


def precision_caption(precision: str) -> str:
    """What per-spot precision means, for a caption ("" without it, the default)."""
    if not precision:
        return ""
    return ("   [localisation error per spot: each step's own, from its two spots' precision through a calibration "
            f"measured on immobile spots ({precision}); maximum likelihood; D {PER_SPOT_D_IS}]")


def lag_caption(lag, lag_ms: float | None) -> str:
    """What a lag other than 1 means, for a caption ("" at lag 1, the default)."""
    from .comparison import JOINT_LAG

    if lag == JOINT_LAG:
        return ("   [time lags 1-4 together (SMTracker 2): one fit to the displacements 1 to 4 frames apart, "
                "D and fractions "
                "shared across the lags, the count chosen at lag 1; localisation error is not modelled, so a "
                "population at the floor reads lower than at lag 1]")
    if int(lag) == 1:
        return ""
    when = "" if lag_ms is None else f" ({lag_ms:g} ms)"
    return (f"   [time lag {lag} frames{when}: displacements {lag} frames apart, so D is the apparent D at that lag; "
            f"localisation error alone reads {lag} times lower than at lag 1]")


def spread_caption(uncertainty: str, withheld: list, scope: str, pseudo: dict | None = None,
                   points=None, resamples: int | None = None) -> str:
    """What the spread on this figure MEANS, for the caption -- so a cropped figure
    still says whether its ± is a spread between days, an interval, or nothing. ``points``
    (the drawn `BubblePoint`s): under Replicates with no point carrying a ± -- every condition
    imaged in one replicate -- the caption says there is none rather than define one (found by
    the overnight tester, 4 Nov 2026)."""
    if uncertainty == "none":
        return "pooled: every replicate as one set, no interval"
    held = list((pseudo or {}).get("withheld") or {})
    if uncertainty == "day" and pseudo and held:
        # CHUNKS WITHHELD where a condition has replicates (`comparison.chunks_withheld`): those are drawn pooled
        from .comparison import CHUNKS_WITHHELD

        dealt = points is None or any(p.spread == "sd" and p.n_replicates >= 2 for p in points)
        lead = f"± = SD over {chunk_words(pseudo)}" if dealt else "no ±"
        return f"{lead}; {', '.join(held)}: {CHUNKS_WITHHELD}, drawn as the pooled fit"
    if (uncertainty == "day" and points is not None and points
            and not any(p.spread == "sd" and p.n_replicates >= 2 for p in points)):
        return ("no ±: every condition holds one replicate, so there is no spread between replicates "
                "(Movies or Cells give an interval over this session)")
    if uncertainty == "day" and pseudo:
        return f"± = SD over {chunk_words(pseudo)}"
    if uncertainty == "day":
        return "± = SD over replicates (a spread between replicates, not a CI)"
    words = UNIT_WORDS.get(uncertainty, "cells")
    # how many resamples, said (10 Nov 2026): the SQD section's and the apparent D's, 200 each
    times = f" {resamples:,} times" if resamples else ""
    note = f"[ ] = 95 % interval, {words} resampled{times}, around the pooled fit"
    if withheld:
        why = ("not computed with D shared across conditions" if scope in ("global", "group")
               else "withheld: two or more replicates make the replicate the unit, "
                    f"and {words} within a replicate are not independent")
        note += f"; {', '.join(withheld)}: {why}"
    return note


def _resamples(spec=None) -> int:
    """The SQD section's single-session resample count, for its caption (`comparison.SINGLE_SESSION_RESAMPLES`)."""
    from .comparison import SINGLE_SESSION_RESAMPLES

    return spec.interval_resamples() if spec is not None else SINGLE_SESSION_RESAMPLES


def point_key(data: BubblePlotData, pt: BubblePoint) -> tuple:
    """`(condition, series, population)` -- identifies a drawn `BubblePoint`
    as a table row, independent of whatever the CURRENT filter selection
    happens to hide. The one shape `table_rows` (building each row's key)
    and `BubbleCard._on_chart_hover` (matching a hovered bubble back to a
    row) both use, so a hover can never look for a key `table_rows` would
    never have produced.
    """
    condition = data.conditions[pt.column]
    series = data.series_labels[pt.style_index] if data.series_labels else None
    return (condition, series, pt.population)


def table_rows(data: BubblePlotData, *, condition_filter: set | None = None,
                series_filter: set | None = None):
    """`(columns, rows, keys, caption)` for `widgets.table.TableBody` --
    built from the SAME `BubblePlotData` the chart paints (one source of
    truth), so the table can never disagree with the bubbles it lists.

    One row per drawn population (plus one per errored/unfittable column,
    named by its own reason rather than omitted -- a missing row there would
    read as "this condition has no populations" instead of "this condition
    could not be fit"). `condition_filter`/`series_filter` are the
    `FacetedFilter` selections: an EMPTY set (or `None`) means no filter on
    that column, matching `widgets.table.FacetedFilter`'s own "unchecked =
    show everything" rule -- never "everything checked".

    Filters ROWS only. The chart is untouched: `WorkspaceWindow.status`'s
    per-condition fraction numbers, and the bubbles themselves, keep showing
    every condition/series that was actually fit, whether or not the table
    is currently narrowed to a subset of its own rows.

    `keys[i]` is `point_key(data, pt)` for a population row, or `None` for
    an error row -- an error names a condition that fit nothing, so no
    bubble, and therefore no hover, can ever point at one.
    """
    has_series = bool(data.series_labels)
    columns = (["Condition", "Series", "Population", "D (µm²/s)", "Fraction", "n"]
               if has_series else
               ["Condition", "Population", "D (µm²/s)", "Fraction", "n"])
    rows = []
    keys = []
    for pt in data.points:
        condition = data.conditions[pt.column]
        series = data.series_labels[pt.style_index] if has_series else None
        if condition_filter and condition not in condition_filter:
            continue
        if series_filter and has_series and series not in series_filter:
            continue
        pct = fraction_text(pt)
        cells = [condition]
        if has_series:
            cells.append(series)
        cells += [str(pt.population), d_text(pt), pct, str(pt.n_replicates)]
        rows.append(cells)
        keys.append(point_key(data, pt))
    for i, reason in data.errors.items():
        condition = data.conditions[i]
        if condition_filter and condition not in condition_filter:
            continue
        cells = [condition]
        if has_series:
            cells.append("–")
        cells += ["–", "–", str(reason), "–"]
        rows.append(cells)
        keys.append(None)
    caption = (f"One row per drawn population; {shared_d_words(data)}; "
              f"fraction weighting: {data.fraction_weighting}; n = the "
              "independent units (replicates, or cells where a session "
              "has only one) the fraction was averaged over."
              + ((" Time lags 1-4 together (SMTracker 2): D and fractions shared across the four lags."
                  if data.lag_frames == "1-4" else
                  f" Time lag: {data.lag_frames} frames; D is the apparent D at that lag.")
                 if getattr(data, "lag_frames", 1) != 1 else "")
              + (" Localisation error per spot (each step's own, through a calibration measured on immobile "
                 f"spots); D {PER_SPOT_D_IS}." if getattr(data, "precision", "") else ""))
    return columns, rows, keys, caption


def copy_rows(data: BubblePlotData, *, condition_filter: set | None = None,
              series_filter: set | None = None) -> tuple[list, list]:
    """The SQD data table for the clipboard (`table_copy.excel_text`): the rows `table_rows` lists, under the same
    filters, at FULL precision rather than as displayed ("38 ± 4%" is rounded for reading): D in um2/s, the
    fraction and its spread in per cent, each in its own column; an unfittable condition carries its reason."""
    has_series = bool(data.series_labels)
    shown = [pt for pt in data.points
             if not (condition_filter and data.conditions[pt.column] not in condition_filter)
             and not (series_filter and has_series and data.series_labels[pt.style_index] not in series_filter)]
    any_sd = any(pt.spread == "sd" and pt.n_replicates >= 2 for pt in shown)
    any_ci = any(pt.spread == "ci" and pt.interval is not None for pt in shown)
    # THE DAYS BEHIND THE MEAN (asked 15 Oct 2026: "day 1, day 2, day 3 would be printed as well"): each day's
    # fraction, from `unit_values`, which holds exactly the values a drawn mean is made of (Uncertainty = Days)
    per_day = {c: u for c, u in (data.unit_values or {}).items() if u.get("unit") == "day" and u.get("fractions")}
    n_day_cols = max((len(u["fractions"]) for u in per_day.values()), default=0) if not has_series else 0
    columns = (["Condition"] + (["Series"] if has_series else []) + ["Population", "D (µm²/s)", "Fraction (%)"]
               + (["Fraction SD (%)"] if any_sd else []) + (["Fraction CI low (%)", "Fraction CI high (%)"]
                                                            if any_ci else []) + ["n"]
               + [f"Replicate {k + 1} (%)" for k in range(n_day_cols)])
    rows = []
    for pt in shown:
        row = [data.conditions[pt.column]] + ([data.series_labels[pt.style_index]] if has_series else [])
        row += [int(pt.population), d_text(pt) if getattr(pt, "d_bound", "") else float(pt.d),
                100.0 * float(pt.fraction)]
        if any_sd:
            row.append(100.0 * float(pt.fraction_sd) if pt.spread == "sd" and pt.n_replicates >= 2 else "")
        if any_ci:
            lo, hi = pt.interval if (pt.spread == "ci" and pt.interval is not None) else ("", "")
            row += [100.0 * lo if lo != "" else "", 100.0 * hi if hi != "" else ""]
        row.append(int(pt.n_replicates))
        if n_day_cols:
            days = per_day.get(data.conditions[pt.column], {}).get("fractions") or []
            vals = [100.0 * float(day[pt.population - 1]) for day in days if len(day) >= pt.population]
            row += vals + [""] * (n_day_cols - len(vals))
        rows.append(row)
    for i, reason in data.errors.items():
        condition = data.conditions[i]
        if condition_filter and condition not in condition_filter:
            continue
        row = [condition] + (["–"] if has_series else []) + ["–", "–", str(reason)]
        rows.append(row + [""] * (len(columns) - len(row)))
    return columns, rows


def _day_fractions(payload) -> list:
    """``[(day label, [fraction per population], n_tracks)]`` from a condition's replicate payload: the per-day
    fit the SQD Run made (at the condition's D, or each day's own under the replicate scope)."""
    reps = (payload or {}).get("replicates") if isinstance(payload, dict) else None
    out = []
    for rep in reps or []:
        fr = rep.get("fractions")
        if fr is not None and rep.get("replicate") != "all":
            out.append((rep.get("replicate"), [float(f) for f in fr], rep.get("n_tracks")))
    return out


def sqd_receipt(data: BubblePlotData, payloads: dict, spec, uncertainty: str, *,
                computed_at: str | None = None) -> dict | None:
    """The Metadata panel's receipt for the SQD section (asked 15 Oct 2026: "it would be cool if day 1, day 2,
    day 3 would be printed as well, maybe we can do this in the metatable on the right bar"): the run's settings,
    then per condition one line per replicate day -- that day's fractions, slowest population first -- and the
    total. Nothing is refitted: the days are the replicate payloads the Run already made."""
    from .metadata_view import receipt_from_rows

    if data is None or not data.conditions:
        return None
    ecdf = getattr(spec, "ecdf_subsample", None)
    shared = scope_words(str(getattr(spec, "d_sharing_scope", "")), str(getattr(spec, "d_sharing_factor", "") or ""))
    if getattr(spec, "d_sharing_by_condition", None):
        # A SHARED D PER CONDITION: the Run's settings name each condition's own
        shared = "per condition: " + "; ".join(f"{c}, {scope_words(spec.scope_for(c))}" for c in spec.conditions)
    # FITS SIDE BY SIDE: a column is a condition at one model, so the conditions are counted once and the models
    # named (3 Nov 2026, reported: "4 CONDITIONS" for two conditions at Best and 1 population, no models said)
    fits = list(getattr(data, "fits", None) or [])
    of_condition = ({str(name): str(fit["condition"]) for name, fit in zip(data.conditions, fits)} if fits
                    else {str(name): str(name) for name in data.conditions})
    models_row = []
    if fits:
        by_condition: dict = {}
        for fit in fits:
            by_condition.setdefault(str(fit["condition"]), []).append(_fit_model(fit))
        sets = list(dict.fromkeys(tuple(v) for v in by_condition.values()))
        models_row = [("Models", (models_words(sets[0]) if len(sets) == 1 else "; ".join(
            f"{c}: {models_words(v)}" for c, v in by_condition.items())) + " · side by side, one column each")]
    rows = [("D", shared), *models_row,
            ("Weighting", str(getattr(spec, "fraction_weighting", ""))),
            ("Uncertainty", {"day": "replicates", "movie": "movies", "cell": "cells", "none": "none"}.get(
                uncertainty, str(uncertainty))),
            ("ECDF", "full" if ecdf == 0 else f"{ecdf} points" if ecdf else "fast")]
    days_of = {}
    for condition in data.conditions:
        days = _day_fractions((payloads or {}).get(condition))
        if not days:
            continue
        days_of[condition] = days
        n_pop = len(days[0][1])
        text = "; ".join(f"replicate {d} " + " / ".join(f"{100 * f:.1f}" for f in fr) + " %" for d, fr, _n in days)
        rows.append((condition, f"{n_pop} populations · " + text))
    if not days_of:
        return None
    # each condition's replicates once, whatever number of models it was fitted with
    replicates: dict = {}
    for column, days in days_of.items():
        condition = of_condition.get(column, column)
        replicates[condition] = max(replicates.get(condition, 0), len(days))
    n_conditions, n_days = len(replicates), sum(replicates.values())
    rows.append(("Total", f"{n_conditions} condition{'s' if n_conditions != 1 else ''} · {n_days} replicates"))

    def short(name, value):
        if name not in days_of:
            return None
        out = []
        for d, fr, n_tracks in days_of[name]:
            out.append((f"Replicate {d}", " / ".join(f"{100 * f:.1f}" for f in fr) + " %",
                        f"{n_tracks:,} tracks" if n_tracks is not None else "", name))
        return out

    stamp = ["SMTRACKER", "SQD POPULATIONS",
             f"{n_conditions} CONDITION{'S' if n_conditions != 1 else ''}", computed_at or ""]
    return receipt_from_rows(rows, stamp=stamp, rendering_names=(), short=short,
                             footer="* fractions per replicate, slowest first *")


def condition_facets(data: BubblePlotData) -> list:
    """`[(condition, count), ...]` in first-seen order -- the Condition
    filter's own options, straight from the drawn points (never from the
    workspace's separate Conditions tool: that toggle does not drive this
    chart at all yet, so it has nothing authoritative to say about which
    conditions this TABLE can filter by; the table's own rows are the
    only honest source)."""
    counts: dict = {}
    for pt in data.points:
        counts[data.conditions[pt.column]] = (
            counts.get(data.conditions[pt.column], 0) + 1)
    return [(c, counts[c]) for c in data.conditions if c in counts]


def series_facets(data: BubblePlotData) -> list:
    """`[(series, count), ...]` -- `[]` outside an overlay, matching
    `FacetedFilter.set_options`'s own rule that a column with fewer than two
    distinct values gets no filter button at all."""
    if not data.series_labels:
        return []
    counts: dict = {}
    for pt in data.points:
        s = data.series_labels[pt.style_index]
        counts[s] = counts.get(s, 0) + 1
    seen = list(dict.fromkeys(data.series_labels))
    return [(s, counts[s]) for s in seen if s in counts]


def _pair_tag(tags: dict, replicate) -> str:
    """``replicate``'s pair tag in ``tags`` (``{replicate number: tag}``), "" where it has none."""
    try:
        return str(tags.get(int(replicate), ""))
    except (TypeError, ValueError):
        return ""


def shows_pairs(payload: dict, tags: dict | None) -> bool:
    """Whether the replicate table carries a Pair column: some replicate it lists was given a pair tag on the Groups
    board (``tags``: the condition's ``{replicate number: tag}`` from `project.Project.replicate_pairs`)."""
    return bool(tags) and any(_pair_tag(tags, rep["replicate"]) for rep in payload.get("replicates") or ())


#: The replicate table's words for its Pair column (5 Nov 2026), on screen and in print.
PAIR_COLUMN_NOTE = ("Pair: the tag given on the Groups board; the replicate of another condition with the same tag is "
                    "its partner, read only by the paired test")


def replicate_table_rows(payload: dict, tags: dict | None = None) -> tuple[list[str], list[list[str]]]:
    """`(columns, rows)` for `payload["replicates"]` -- `Replicate, Cells,
    Tracks`, then one `f{k}` column per population, matching the legacy
    inspector's own table exactly (screenshot: per-replicate fractions,
    D shared across the header rather than repeated per row). Where replicates were declared as pairs on the Groups
    board (``tags``, `shows_pairs`; asked 5 Nov 2026), a Pair column after Replicate: each one's tag, "none" without."""
    n_pop = payload["n_pop"]
    paired = shows_pairs(payload, tags)
    columns = ["Replicate"] + (["Pair"] if paired else []) + ["Cells", "Tracks"] + [f"f{k + 1}" for k in range(n_pop)]
    rows = []
    for rep in payload["replicates"]:
        row = [str(rep["replicate"])] + ([_pair_tag(tags, rep["replicate"]) or "none"] if paired else [])
        row += [str(rep["n_cells"]), str(rep["n_tracks"])]
        row += [f"{100 * f:.1f}%" for f in rep["fractions"]]
        rows.append(row)
    return columns, rows


def populations_table(data: BubblePlotData, *, condition_filter: set | None = None,
                      series_filter: set | None = None, labels: dict | None = None) -> dict:
    """The SQD data table as the "table" figure kind draws it: the rows on screen (its
    filters applied), a condition by its shown name."""
    columns, rows, _keys, caption = table_rows(data, condition_filter=condition_filter,
                                               series_filter=series_filter)
    labels = labels or {}
    rows = [[str(labels.get(r[0], r[0]))] + [str(c) for c in r[1:]] for r in rows]
    return {"title": "SQD populations", "columns": list(columns), "rows": rows,
            "caption": caption, "file_name": "sqd_populations_table"}


def replicate_footer(payload: dict, n: int) -> str:
    """The replicate table's footer, on screen and in print: "3 replicates, step weighting, unit = replicate". The unit
    as a user reads it (`chart_text.unit_word`: "day" is a replicate; 3 Nov 2026, reported "unit = day")."""
    from .chart_text import unit_word

    return (f"{n} replicate{'' if n == 1 else 's'}, {payload['weighting']} weighting, "
            f"unit = {unit_word(payload['unit'])}")


def replicate_table(payload: dict, condition: str = "", label: str = "", tags: dict | None = None) -> dict:
    """The replicate-days table as the "table" figure kind draws it: one row per replicate,
    the fractions under "Population k" (the screen's f1, f2 ...), the shared D said; the Pair column where the
    replicates were declared as pairs (``tags``, `replicate_table_rows`)."""
    columns, rows = replicate_table_rows(payload, tags)
    columns = [f"Population {c[1:]}" if c.startswith("f") and c[1:].isdigit() else c
               for c in columns]
    caption = (f"{shared_d_caption(payload)}. {replicate_footer(payload, len(rows))}; each population "
               "column is its fraction in that replicate."
               + (f" {PAIR_COLUMN_NOTE}." if shows_pairs(payload, tags) else ""))
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(condition))
    return {"title": "SQD replicate fits", "condition": label or condition,
            "columns": columns, "rows": rows, "caption": caption,
            "file_name": f"sqd_replicates_table_{safe}"}


def shared_d_caption(payload: dict) -> str:
    """The legacy inspector's own header line, reproduced: shared D across
    replicates when the fit scope shares one (`condition`/`global`), or a
    named admission that it does not when the scope is `replicate` --
    `condition_replicate_fit`'s docstring calls that the ONLY arrangement
    where D itself is not one shared quantity."""
    shared = payload.get("shared_D")
    if shared:
        d_text = ", ".join(f"{d:.4g}" for d in shared)
        group = payload.get("group") or {}
        if payload.get("scope") == "group" and group:
            others = [c for c in group.get("with", ()) if c != payload.get("condition")]
            where = (f"{scope_words('group', group.get('factor') or '')}: {group.get('name', '')}".rstrip()
                     + (f", with {', '.join(others)}" if others else ", alone in its group"))
            return f"shared D = [{d_text}] µm²/s ({where})"
        # the Compute dialog's words for the scope, never its internal value ("condition scope" until 1 Nov 2026)
        return f"shared D = [{d_text}] µm²/s ({scope_words(payload['scope'])})"
    words = scope_words("replicate")
    return words[:1].upper() + words[1:]
