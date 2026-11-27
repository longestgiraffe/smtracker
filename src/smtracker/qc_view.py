"""The QC section's numbers, rules and sentences, with no Qt anywhere in it.

`ui/workspace/qc_chart.py` is the Qt shell (the Overview table, the Comparator, the Detail cards and their
dialogs); this is what it draws from. They are split because of the Thin Qt rule (docs/gui_tool_views.md;
`tests/test_ui_thin_qt.py` walks the whole GUI layer): a wrong number or a wrong sentence must fail in a
headless test, not look subtly wrong on screen, and a process-pool worker should import this module without
importing Qt.

Every number comes from `smtracker.qc` (and, for the brightness check, `smtracker.spot_intensity`); nothing
here edits or depends on the legacy browser layer. This module reports structured numbers, never a paragraph.

What lives here, as the section is moved over step by step:
  * the Overview and the simple Detail cards (step 2): `compute_qc_overview` and the plain blocks it puts in each
    condition's entry (``overview`` is derived from them by `overview_row`; ``length``, ``groups``, ``in_cell``,
    ``neighbour``), the neighbour-share measurement, the sparkline bucketing, the verdict and the card captions
    and tooltips;
  * the per-unit outlier views (step 3): `unit_payload` (one plain block per unit: cell, movie or day), the column
    grouping, the movie runs and shared-prefix label trim, the Inspect dialog's description and readout, and the
    tooltip sentences;
  * the General page's two cards (13 Oct 2026): `fair_comparison` -- the conditions side by side on what biases a
    fraction, each column's rule in `FAIR_COLUMNS` and the values that break it marked -- and
    `contributes_nothing` -- everything in the folders that is not in the numbers, per condition.

RETIRED 13 Oct 2026 (the experimenter: "some info is not needed, while i feel we have urgent info we can
provide"): the Comparator (contamination and spot brightness against a control) and the five extra views of the
Localizations-per-frame card. The spine functions they called (`qc.control_contamination`,
`spot_intensity`) are untouched.
"""
from __future__ import annotations

import math
import os
from bisect import bisect_right

import numpy as np
import pandas as pd

from smtracker.chart_text import unit_word

__all__ = [
    "IN_CELL_NOT_MEASURED",
    "CELL_LENGTH_RATIO",
    "CELL_LENGTH_RULE",
    "FAIR_COLUMNS",
    "cell_length_comparison",
    "cell_length_hover",
    "cell_lengths",
    "reordered",
    "contributes_nothing",
    "fair_comparison",
    "groups_table",
    "TRACK_CAP_ABOUT",
    "track_cap_lines",
    "track_cap_of",
    "QUANTITY",
    "SPARK_BARS",
    "column_groups",
    "column_hover",
    "compute_qc_neighbour_share",
    "compute_qc_overview",
    "group_table",
    "in_cell_caption",
    "in_cell_hover",
    "in_cell_payload",
    "length_hover",
    "length_payload",
    "movie_of",
    "movie_runs",
    "neighbour_band_text",
    "neighbour_bar_text",
    "neighbour_caption",
    "neighbour_payload",
    "outlier_caption",
    "outlier_description",
    "outlier_readout",
    "overview_row",
    "quantity",
    "radius_phrase",
    "short_labels",
    "spark_bars",
    "unit_hover",
    "unit_payload",
    "unit_payloads",
]


# --------------------------------------------------------------------------- #
# The Comparator card and its Configure dialog: rules and sentences
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Overview and the simple Detail cards (step 2)
# --------------------------------------------------------------------------- #

#: The Overview sparkline is bucketed down to this many bars however many steps the longest track has:
#: `LengthDistribution.counts` has one bin per step count, and 40+ bins packed into ~60 px read as one spike and a
#: flat line rather than a shape.
SPARK_BARS = 12
DASH = "–"


def _finite(value) -> bool:
    return value is not None and bool(np.isfinite(value))


def _none_if_nan(value) -> float | None:
    return float(value) if _finite(value) else None


def spark_bars(counts) -> list[float] | None:
    """The Length column's bars: the histogram's own counts, with consecutive bins SUMMED into `SPARK_BARS`
    buckets when there are more (a shape readable at sparkline size, not a resample that could invent a height
    nothing in the data produced). `None` means no histogram."""
    if counts is None:
        return None
    counts = np.asarray(counts, dtype=float)
    if len(counts) > SPARK_BARS:
        edges = np.linspace(0, len(counts), SPARK_BARS + 1).astype(int)
        counts = np.array([counts[edges[i]:edges[i + 1]].sum() for i in range(SPARK_BARS)])
    return [float(c) for c in counts]


def overview_row(entry: dict) -> dict:
    """What one Overview row says, as words: the cells' text, the sparkline bars, and whether the condition is
    flagged. An entry that failed to compute gives its error as the status, with dashes and a zero count."""
    if "error" in entry:
        return {"failed": True, "tracks": DASH, "locs_cells": DASH, "bars": None,
                "in_cell": DASH, "flagged": False, "n_flagged": "0", "status": entry["error"]}
    in_cell = entry.get("in_cell_share_pct")
    length = entry.get("length")
    n_flagged = entry.get("n_flagged", 0)
    low_yield = entry.get("low_yield_days") or []
    # A low-yield day (`qc.flag_low_yield`) flags the condition too; the badge says which
    # kind when it is the only one, and `yield_text` names the days.
    status = ("flagged" if n_flagged else "low yield" if low_yield else "clean")
    return {
        "failed": False,
        "tracks": str(entry.get("n_tracks", DASH)),
        "locs_cells": f"{entry.get('n_localizations', DASH)} / {entry.get('n_cells', DASH)}",
        "bars": spark_bars(length["counts"] if length is not None else None),
        "in_cell": f"{in_cell:.0f}%" if in_cell is not None else "not measured",
        "flagged": bool(n_flagged or low_yield),
        "n_flagged": str(n_flagged),
        "status": status,
        "yield_text": yield_text(low_yield),
    }


def yield_text(days: list) -> str:
    """"low yield: day 2" / "low yield: days 1, 2", or "" (`qc.flag_low_yield`)."""
    if not days:
        return ""
    names = ", ".join(str(d) for d in days)
    return f"low yield: replicate{'s' if len(days) > 1 else ''} {names}"


def low_yield_days(project, condition: str) -> tuple:
    """``(days, note)``: the condition's replicate days whose track yield is low by the QC
    rule (`qc.flag_low_yield` on `qc.sample_size` per day -- the Images Overview's own
    numbers, outlined cells and tracked frames), and the rule's sentence. A condition whose
    cells cannot be counted gives ``([], why)``, never a silent pass."""
    from .images_view import ImagesError, _condition_cells
    from .qc import QCError, flag_low_yield, sample_size

    try:
        cells, _notes = _condition_cells(project, condition)
        judged = flag_low_yield(sample_size(cells, by=("condition", "day")))
    except (ImagesError, QCError, KeyError, ValueError) as exc:
        return [], f"track yield not judged: {exc}"
    days = [d for d, low in zip(judged["day"], judged["low_yield"]) if bool(low)]
    return [int(d) if float(d).is_integer() else d for d in days], judged.attrs.get("note", "")


def length_payload(ld) -> dict:
    """The track-length card: one bin per step count, which bins the filter keeps (a bin is kept when its centre is
    above the ``min_steps`` cut), and the spine's own caption."""
    edges = [float(e) for e in ld.edges]
    counts = [float(c) for c in ld.counts]
    top = getattr(ld, "max_steps", None)
    kept = [ld.min_steps < (edges[i] + edges[i + 1]) / 2.0 <= (top if top is not None
                                                               else float("inf"))
            for i in range(len(counts))]
    return {"counts": counts, "edges": edges, "min_steps": int(ld.min_steps), "kept": kept,
            "max_steps": top, "caption": ld.caption()}


def length_hover(payload: dict, x: float) -> str | None:
    """The tooltip for the bin that holds step value ``x``, or None over an empty or missing bin."""
    edges, counts = payload["edges"], payload["counts"]
    i = bisect_right(edges, x) - 1
    if not 0 <= i < len(counts) or counts[i] <= 0:
        return None
    step = (edges[i] + edges[i + 1]) / 2.0
    top = payload.get("max_steps")
    if step <= payload["min_steps"]:
        kept = "removed by the filter"
    elif top is not None and step > top:
        kept = f"left out: longer than {top} steps"
    else:
        kept = "kept"
    return f"{step:g} steps: {counts[i]:,.0f} tracks ({kept})"


#: What the cap does and what it costs, for the dialog that sets it.
TRACK_CAP_ABOUT = (
    "Tracks longer than this many steps are left out of every analysis in every condition. "
    "Long tracks are often stuck objects, autofluorescent spots or slow molecules, so a cap lowers "
    "slow fractions a little. QC still draws the left-out tracks, marked as removed.")


def track_cap_of(project) -> int | None:
    """The cap ``project`` carries (`canonical.cap_track_length`), or None."""
    cap = getattr(getattr(project, "canonical", None), "attrs", {}).get("track_cap") or {}
    return cap.get("max_steps")


def track_cap_lines(canonical, max_steps: int, order=()) -> dict:
    """The dialog's live preview of a cap on the UNCAPPED table: ``{"ok", "cost",
    "per_condition", "longest"}`` -- what it would leave out, in all and per condition,
    before anything is applied. ``ok`` is False when the cap is below
    `canonical.MIN_CAP_STEPS` or would leave nothing. The conditions in ``order`` (the window's,
    `condition_order`), the rest after them as the table has them (3 Nov 2026, reported: listed in reverse)."""
    from .canonical import MIN_CAP_STEPS, track_cap_cost

    whole = track_cap_cost(canonical, max_steps)
    longest = whole["longest"]
    if max_steps < MIN_CAP_STEPS:
        return {"ok": False, "longest": longest, "per_condition": [],
                "cost": f"A cap below {MIN_CAP_STEPS} steps would leave out every track the "
                        "length filter keeps."}
    if whole["n_tracks_removed"] >= whole["n_tracks"]:
        return {"ok": False, "longest": longest, "per_condition": [],
                "cost": "That would leave out every track."}
    if not whole["n_tracks_removed"]:
        cost = f"Leaves out nothing: the longest track has {longest} steps."
    else:
        cost = (f"Leaves out {whole['n_tracks_removed']:,} of {whole['n_tracks']:,} tracks "
                f"({100.0 * whole['n_tracks_removed'] / whole['n_tracks']:.1f} %) and "
                f"{whole['n_localizations_removed']:,} of {whole['n_localizations']:,} "
                f"localisations ({100.0 * whole['n_localizations_removed'] / max(1, whole['n_localizations']):.1f} %). "
                f"The longest track has {longest} steps.")
    from .condition_order import apply as order_apply

    per = {}
    if "condition" in canonical.columns:
        for name, part in canonical.groupby("condition", observed=True, sort=False):
            one = track_cap_cost(part, max_steps)
            share = 100.0 * one["n_tracks_removed"] / max(1, one["n_tracks"])
            per[str(name)] = (f"{name}: {one['n_tracks_removed']:,} of {one['n_tracks']:,} tracks "
                              f"({share:.1f} %)")
    lines = [per[name] for name in order_apply(list(per), [str(k) for k in order or ()])]
    return {"ok": True, "cost": cost, "per_condition": lines, "longest": longest}


def group_table(df) -> dict:
    """N per group as table text: the condition column is dropped (this page is one condition)."""
    by = [b for b in df.attrs.get("by", ()) if b != "condition"]
    columns = [unit_word(b).capitalize() for b in by] + ["Tracks", "Locs", "Cells"]
    rows = []
    for _, table_row in df.iterrows():
        cells = [str(table_row[k]) for k in by]
        cells += [f"{int(table_row['n_tracks']):,}", f"{int(table_row['n_localizations']):,}",
                  str(int(table_row["n_cells"]))]
        rows.append(cells)
    return {"columns": columns, "rows": rows}


def groups_table(block: dict, condition: str) -> dict:
    """QC's N-per-group table as the "table" figure kind draws it."""
    columns = ["Localisations" if c == "Locs" else c for c in block.get("columns", ())]
    return {"title": "N per group", "condition": condition, "columns": columns,
            "rows": [list(r) for r in block.get("rows", ())],
            "caption": "Tracks, localisations and cells per replicate and movie in the "
                       "analysed table (after the length filter and the cell filter). The cell "
                       "count is the one an interval or a test rests on.",
            "file_name": "n_per_group_" + "".join(c if c.isalnum() or c in "._-" else "_"
                                                  for c in str(condition))}


#: What the "Tracks inside a cell outline" card says where the data carries no per-movie cell-filter counts (4 Nov 2026;
#: it showed `qc.in_cell_share`'s refusal, with `cellmesh.assign_cells`, `canonical.to_canonical` and
#: `use_canonical_cache=False` in it). The phrase "per-movie cell-filter counts" is pinned by the tests.
IN_CELL_NOT_MEASURED = (
    "Not measured: this project's data has no per-movie cell-filter counts, so the number of tracks outside the "
    "cell outlines is unknown. Load the project again to measure it.")


def in_cell_caption(df) -> str:
    """The spine's own note (a low share is a look-at-this, not a verdict), every movie that contributes nothing
    by name, and the edge legend."""
    a = df.attrs
    text = str(a.get("note", "")).strip()
    empty = df[df["tracks_in_cells"] == 0]["movie"].astype(str).tolist()
    if empty:
        text += " CONTRIBUTING NOTHING: " + ", ".join(empty) + "."
    return f"{text} | orange edge = flagged ({int(a.get('n_flagged', 0))} movie(s))"


def in_cell_payload(df) -> dict:
    """Per movie: tracks found, tracks inside an outline, the share inside as a percentage (None where undefined),
    and whether the movie is unlike its condition."""
    df = df.reset_index(drop=True)
    return {
        "names": df["movie"].astype(str).tolist(),
        "found": [float(v) for v in df["tracks_found"].to_numpy(dtype=float)],
        "inside": [float(v) for v in df["tracks_in_cells"].to_numpy(dtype=float)],
        "share": [_none_if_nan(v) for v in df["in_cell_share"].to_numpy(dtype=float) * 100.0],
        "flagged": [bool(v) for v in df["flagged"].to_numpy(dtype=bool)],
        "reasons": [str(v) for v in df["reason"]] if "reason" in df else [""] * len(df),
        "caption": in_cell_caption(df),
    }


def in_cell_hover(payload: dict, i: int) -> str:
    found, inside, share = payload["found"][i], payload["inside"][i], payload["share"][i]
    pct = f"{share:.0f}%" if share is not None else "n/a"
    flag = " \u00b7 unlike its condition" if payload["flagged"][i] else ""
    return (f"{payload['names'][i]}: {found:,.0f} found \u00b7 {inside:,.0f} inside a cell "
            f"({pct}) \u00b7 {found - inside:,.0f} outside{flag}")


def neighbour_caption(df) -> str:
    """The size (median), how many movies are flagged, the radius, and what the number is: a lower bound. The flag
    is relative, so the size has to be printed."""
    a = df.attrs
    return (f"median {a.get('median', float('nan')):.1f}% \u00b7 {a.get('n_flagged', 0)} of "
            f"{a.get('n_movies', len(df))} movies flagged \u00b7 share of localisations with "
            f"another within {a.get('radius_px', 0):g} px in the same frame \u00b7 lower bound: "
            f"only localisations that passed tracking and the cell filter are counted")


def neighbour_payload(df) -> dict:
    """Per movie: the share of localizations with a neighbour inside the search radius, the usual range (``low`` and
    ``high``, None where the flagging could not form one) and the median."""
    a = df.attrs
    return {
        "names": df["movie"].astype(str).tolist(),
        "share": [float(v) for v in df["share_pct"].to_numpy(dtype=float)],
        "flagged": [bool(v) for v in df["flagged"].to_numpy(dtype=bool)],
        "n_localizations": [int(v) for v in df["n_localizations"]],
        "typical": [float(v) for v in df["typical_per_frame"]],
        "busiest": [float(v) for v in df["busiest_per_frame"]],
        "low": _none_if_nan(a.get("low")),
        "high": _none_if_nan(a.get("high")),
        "median": _none_if_nan(a.get("median")),
        "radius_px": float(a.get("radius_px", 0)),
        "k": _none_if_nan(a.get("k")),
        "scale_estimator": str(a.get("scale_estimator", "MAD")),
        "caption": neighbour_caption(df),
    }


def neighbour_band_text(payload: dict) -> str | None:
    """The tooltip for the grey band (where no bar is under the pointer), or None when there is no band."""
    low, high = payload["low"], payload["high"]
    if low is None or high is None:
        return None
    return (f"Most movies fall between {max(low, 0.0):.1f}% and {high:.1f}%. "
            f"Movies outside this range are shown in orange.")


def neighbour_bar_text(payload: dict, i: int) -> str:
    text = (f"{payload['names'][i]}\n{payload['share'][i]:.1f}% of localisations have a "
            f"neighbour within {payload['radius_px']:g} px\n{payload['n_localizations'][i]:,} localisations \u00b7 "
            f"typical frame {payload['typical'][i]:g}, busiest {payload['busiest'][i]:g}")
    return text + ("\nunlike the other movies" if payload["flagged"][i] else "")


def radius_phrase(radius_px) -> str:
    """The search radius as the Inspect dialog's neighbour-share description names it: "5 px", or a
    general phrase when the table recorded none (0 or None), never a guessed number."""
    return f"{radius_px:g} px" if radius_px else "the search radius"


# --------------------------------------------------------------------------- #
# Per-unit outliers (step 3)
# --------------------------------------------------------------------------- #

#: column -> (axis label, name in a sentence).
QUANTITY = {
    "median_appd": ("median apparent D (\u00b5m\u00b2/s)", "median apparent D"),
    "slow_fraction": ("slow fraction", "slow fraction"),
    "median_steps": ("median track length (steps)", "median track length"),
}


def quantity(target: str) -> tuple[str, str]:
    return QUANTITY.get(target, (target, target))


def short_labels(names) -> list[str]:
    """Drop the part of every label that they all share up to an underscore ("Glucose_series001" -> "series001"):
    a page is one condition, so the condition prefix is the same on every label and costs the room the
    distinguishing part needs."""
    names = [str(n) for n in names]
    if len(names) < 2:
        return names
    cut = os.path.commonprefix(names).rfind("_")
    return [n[cut + 1:] for n in names] if cut > 0 else names


def movie_of(name: str) -> str:
    """A cell id is ``<movie>:<cell>``; the movie is what separates groups."""
    return name.rsplit(":", 1)[0]


def column_groups(names: list, unit: str) -> list:
    """``[[column name, [unit indices]]]`` in first-seen order: a cell's column is its movie; a movie or a day is its
    own column."""
    seen: dict = {}
    for i, name in enumerate(names):
        seen.setdefault(movie_of(name) if unit == "cell" else name, []).append(i)
    return [[name, idx] for name, idx in seen.items()]


def movie_runs(movies: list) -> list:
    """``[[movie, start, stop]]``: each run of consecutive units from one movie."""
    runs, start = [], 0
    for k in range(1, len(movies) + 1):
        if k == len(movies) or movies[k] != movies[start]:
            runs.append([movies[start], start, k])
            start = k
    return runs


#: The robust scale `qc.flag_outliers` used, in words (its ``scale_estimator``).
SCALE_WORDS = {"MAD": "median absolute deviations",
               "meanAD (MAD was 0)": "mean absolute deviations (the median one was 0)"}


def outlier_caption(df) -> str:
    """"2 of 22 cells flagged: median apparent D beyond the median \u00b1 3.5 median absolute deviations" -- in words,
    never the column or estimator names (3 Nov 2026, reported: "median \u00b1 3.5\u00b7MAD on median_appd")."""
    from .chart_text import unit_word

    a = df.attrs
    n_flag = int(df["flagged"].sum()) if "flagged" in df else 0
    n = int(a.get("n_units", len(df)))
    unit = unit_word(a.get("unit", "unit"))
    target = a.get("flagged_on")
    what = quantity(target)[1] if target else "the value"
    scale = a.get("scale_estimator", "MAD")
    return (f"{n_flag} of {n} {unit}{'' if n == 1 else 's'} flagged: {what} beyond the median "
            f"\u00b1 {a.get('k', 0):g} {SCALE_WORDS.get(scale, scale)}")


def unit_payload(df) -> dict:
    """One `qc.unit_summary` frame as plain data: each unit's name and value (None where it has none), whether it is
    flagged, the usual range, the median line (only where the flagging had a spread to use), the loading order and
    the lowest-to-highest order (stable, units without a value last), and the movie of each cell."""
    a = df.attrs
    unit = a.get("unit", "unit")
    target = a.get("flagged_on")
    if target is not None and target not in df.columns:
        target = None
    n = len(df)
    names = df[unit].astype(str).tolist() if unit in df.columns else [str(i) for i in range(n)]
    raw = df[target].to_numpy(dtype=float) if target is not None else np.full(n, np.nan)
    values = [float(v) if np.isfinite(v) else None for v in raw]
    median = _none_if_nan(a.get("median"))
    scale = a.get("scale", 0)
    return {
        "unit": unit,
        "target": target,
        "axis_label": quantity(target)[0] if target is not None else None,
        "quantity": quantity(target)[1] if target is not None else "value",
        "names": names,
        "values": values,
        "flagged": [bool(v) for v in df["flagged"].to_numpy(dtype=bool)],
        "low": _none_if_nan(a.get("low")),
        "high": _none_if_nan(a.get("high")),
        "median": median,
        "has_spread": bool(scale and scale > 0 and median is not None),
        "n": n,
        "n_flagged": int(df["flagged"].sum()),
        "order": list(range(n)),
        "order_sorted": sorted(range(n), key=lambda i: float("inf") if values[i] is None else values[i]),
        "movies": [movie_of(nm) for nm in names] if unit == "cell" else None,
        "k": _none_if_nan(a.get("k")),
        "scale_estimator": str(a.get("scale_estimator", "MAD")),
        "title": f"Per-{unit} outliers",
        "caption": outlier_caption(df),
    }


def sorted_axis_title(unit: str) -> str:
    """The x title under units drawn lowest to highest, on screen and on paper."""
    return f"{unit}s, sorted by value"


def unit_payloads(unit_summaries: dict) -> dict:
    """``{unit: payload}`` for each unit that could be formed, and ``{"error": ...}`` for one that could not."""
    return {unit: (got if isinstance(got, dict) else unit_payload(got))
            for unit, got in unit_summaries.items()}


def unit_hover(payload: dict, i: int) -> str:
    value = payload["values"][i]
    if value is None:
        return f"{payload['names'][i]}: no value"
    flag = " \u00b7 flagged" if payload["flagged"][i] else ""
    return f"{payload['names'][i]}: {payload['quantity']} {value:.4g}{flag}"


def column_hover(name: str, n: int) -> str:
    return f"{name}: {n} in this column"


def outlier_description(template: str, unit: str, quantity_name: str) -> str:
    """The Inspect dialog's sentence for a view: the unit and the quantity are filled into its template."""
    word = unit_word(unit)
    grouping = ("with each movie's cells in one column" if unit == "cell"
                else f"one {word} per column")
    return template.format(unit=word, quantity=quantity_name, grouping=grouping)


def outlier_readout(n_flagged: int, n: int, unit: str) -> str:
    word = unit_word(unit)
    return f"{n_flagged:,} of {n:,} {word if n == 1 else word + 's'} flagged"


# --------------------------------------------------------------------------- #
# Per-frame views (step 4)
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Is the comparison fair? / What contributes nothing (the General page, 13 Oct 2026)
# --------------------------------------------------------------------------- #

#: The fair-comparison table's rules, each stated in its column's tooltip (`FAIR_COLUMNS`). A mark is a
#: look-at-this, never a verdict: a difference in track length or yield can be the biology itself.
FAIR_LENGTH_RATIO = 1.25        # longest / shortest median track length across the conditions
FAIR_YIELD_RATIO = 1.5          # highest / lowest tracks per cell per 1000 frames
FAIR_IN_CELL_POINTS = 15.0      # highest - lowest in-cell share, percentage points
FAIR_NEAR_EDGE_PCT = 10.0       # a condition losing this share of its found tracks just outside an outline
FAIR_CROWDING_PCT = 10.0        # a condition with this share of localisations crowded (a linking competitor)
FAIR_MIN_DAYS = 3               # fewer replicate days than this

#: ``(key, heading, rule)``: the table's columns, in order, each with the sentence its heading's tooltip says.
FAIR_COLUMNS = (
    ("tracker", "Tracker",
     "The tracker each condition was read from. Marked where they differ, because u-track and TrackMate give "
     "slightly different fractions and D on the same movies."),
    ("days", "Replicates",
     f"Replicates / movies. Marked under {FAIR_MIN_DAYS} replicates, where a per-replicate interval or Welch test "
     f"is very wide."),
    ("tau", "Interval",
     "The frame interval every D uses. Marked where some movies were rescaled to it from an interval a few per "
     "cent away; the number in brackets is how many. Rescaling is limited by an option (10 % by default)."),
    ("length", "Steps",
     f"Median steps per track. Marked where the longest median is {FAIR_LENGTH_RATIO:g} times the shortest or "
     f"more. Long tracks are slower on average, so step-weighted fractions lean toward the slow state; the "
     f"difference can also be real."),
    ("yield", "Yield",
     f"Tracks per outlined cell per 1000 frames. Marked where the highest is {FAIR_YIELD_RATIO:g} times the "
     f"lowest or more: labelling density changes crowding and linking, and few tracks per cell make per-cell "
     f"numbers noisy."),
    ("in_cell", "In cell",
     f"Share of found tracks inside a cell outline. Marked where conditions differ by "
     f"{FAIR_IN_CELL_POINTS:g} points or more, because the cell filter then removes different molecules."),
    ("near_edge", "Near edge",
     f"Share of found tracks dropped although every point lies within 1 px of an outline. Marked at "
     f"{FAIR_NEAR_EDGE_PCT:g} % or more: tight outlines drop free molecules near the membrane and raise the "
     f"slow fraction. The Outline tolerance option keeps them."),
    ("aligned", "Outlines",
     "Offset of the tracks from their cells' outlines, x and y in pixels, before the cell filter. Evenly spread "
     "molecules give about 0. Marked at half a pixel or more in both, in the same direction: the tracker and the "
     "outlines then count pixels from different origins."),
    ("crowding", "Crowded",
     f"Median share of localisations with another inside the search radius in the same frame. Marked at "
     f"{FAIR_CROWDING_PCT:g} % or more, where links can join two molecules."),
)


def _condition_yield(project, condition: str) -> float | None:
    """Tracks per outlined cell per 1000 tracked frames over the whole condition (`qc.sample_size`, the Images
    Overview's own number); None where the cells cannot be counted (named by `low_yield_days`)."""
    from .images_view import ImagesError, _condition_cells
    from .qc import YIELD_COLUMN, QCError, sample_size

    try:
        cells, _notes = _condition_cells(project, condition)
        table = sample_size(cells, by=("condition",))
    except (ImagesError, QCError, KeyError, ValueError):
        return None
    if table.empty or YIELD_COLUMN not in table:
        return None
    value = float(table[YIELD_COLUMN].iloc[0])
    return value if math.isfinite(value) else None


def _spread_marks(values: dict, *, ratio: float | None = None, points: float | None = None) -> set:
    """The conditions at the extremes when the spread across ``values`` reaches the rule (a ratio of highest to
    lowest, or a difference in points); empty when it does not or fewer than two values exist."""
    known = {k: v for k, v in values.items() if v is not None and math.isfinite(v)}
    if len(known) < 2:
        return set()
    hi, lo = max(known.values()), min(known.values())
    if ratio is not None:
        hit = lo > 0 and hi / lo >= ratio
    else:
        hit = hi - lo >= points
    return {k for k, v in known.items() if hit and v in (hi, lo)}


def fair_comparison(project, per_condition: dict) -> dict:
    """Is the comparison fair? The conditions side by side on what biases a fraction, each column with its rule
    (`FAIR_COLUMNS`) and the values that break it marked. What CANNOT differ -- the pixel size and the
    localisation sigma, one for the project (the loader refuses a mix) -- is said once in the caption.

    ``{"columns": [(heading, rule)], "rows": [{"condition", "cells": [text], "marked": [bool]}], "caption",
    "n_marked"}``; a condition whose QC failed gets dashes."""
    from .cellmesh import alignment_mismatch

    names = [c for c, e in per_condition.items() if "error" not in e]
    facts = {c: per_condition[c].get("fair") or {} for c in names}
    trackers = {c: f.get("tracker") for c, f in facts.items()}
    common = max(set(trackers.values()), key=list(trackers.values()).count) if trackers else None
    marks = {
        "tracker": ({c for c, t in trackers.items() if t != common}
                    if len(set(trackers.values())) > 1 else set()),
        "days": {c for c, f in facts.items() if f.get("n_days", 0) < FAIR_MIN_DAYS},
        "tau": {c for c, f in facts.items() if f.get("tau_rescaled", 0)},
        "length": _spread_marks({c: f.get("median_steps") for c, f in facts.items()}, ratio=FAIR_LENGTH_RATIO),
        "yield": _spread_marks({c: f.get("yield") for c, f in facts.items()}, ratio=FAIR_YIELD_RATIO),
        "in_cell": _spread_marks({c: f.get("in_cell_pct") for c, f in facts.items()},
                                 points=FAIR_IN_CELL_POINTS),
        "near_edge": {c for c, f in facts.items()
                      if (f.get("near_edge_pct") or 0.0) >= FAIR_NEAR_EDGE_PCT},
        "crowding": {c for c, f in facts.items() if (f.get("crowding_pct") or 0.0) >= FAIR_CROWDING_PCT},
        "aligned": {c for c, f in facts.items() if alignment_mismatch(f.get("alignment"))},
    }

    def pct(v):
        return DASH if v is None else f"{v:.0f}%"

    rows = []
    for condition, entry in per_condition.items():
        if condition not in facts:
            rows.append({"condition": condition, "cells": [DASH] * len(FAIR_COLUMNS),
                         "marked": [False] * len(FAIR_COLUMNS)})
            continue
        f = facts[condition]
        rescaled = int(f.get("tau_rescaled", 0))
        text = {
            "tracker": f.get("tracker") or DASH,
            "days": f"{f.get('n_days', 0)} / {f.get('n_movies', 0)}",
            "tau": (f"{f['tau_ms']:g} ms" if f.get("tau_ms") is not None else DASH)
                   + (f" ({rescaled})" if rescaled else ""),
            "length": DASH if f.get("median_steps") is None else f"{f['median_steps']:g}",
            "yield": DASH if f.get("yield") is None else f"{f['yield']:.1f}",
            "in_cell": pct(f.get("in_cell_pct")),
            "near_edge": pct(f.get("near_edge_pct")),
            "crowding": DASH if f.get("crowding_pct") is None else f"{f['crowding_pct']:.1f}%",
            "aligned": (DASH if not f.get("alignment") else
                        f"{f['alignment']['x']:+.1f}, {f['alignment']['y']:+.1f}"),
        }
        rows.append({"condition": condition,
                     "cells": [text[key] for key, _h, _r in FAIR_COLUMNS],
                     "marked": [condition in marks[key] for key, _h, _r in FAIR_COLUMNS]})
    params = getattr(project, "params", None)
    sigma = getattr(params, "localization_precision_nm", None)
    source = str(getattr(params, "localization_sigma_source", "") or "").replace("_", " ")
    one = []
    # PIXEL SIZES WITHIN 1 % ARE LOADED WITH A WARNING (`project._check_params_agree`, ruled 26 Oct 2026): said here
    mixed = (getattr(getattr(project, "canonical", None), "attrs", None) or {}).get("pixel_size_mixed")
    mixed_text = ""
    if mixed and getattr(params, "pixel_size_nm", None):
        from .project import pixel_mix_text

        mixed_text = "Not one pixel size: " + pixel_mix_text(mixed, params.pixel_size_nm) + ". "
    elif getattr(params, "pixel_size_nm", None):
        one.append(f"pixel {params.pixel_size_nm:g} nm")
    # CONDITIONS LINKED AT DIFFERENT SEARCH RADII (tester D10, ruled 4 Nov 2026): the load's own words, said here
    from .project import radius_mix_text

    radii = radius_mix_text(getattr(params, "search_radius_px_by_condition", None) or {})
    if radii:
        mixed_text += radii + ". "
    if sigma:
        one.append(f"localisation precision {sigma:.0f} nm" + (f" ({source})" if source else ""))
    n_marked = sum(sum(r["marked"]) for r in rows)
    caption = (mixed_text + (f"One for every condition: {', '.join(one)}. " if one else "")
               + ("Marked: the conditions differ by more than a column's rule (hover a heading). "
                  "A mark is a reason to look closer." if n_marked else
                  "Nothing differs by more than the columns' rules (hover a heading)."))
    return {"columns": [(h, r) for _k, h, r in FAIR_COLUMNS], "rows": rows, "caption": caption,
            "n_marked": n_marked}


# --------------------------------------------------------------------------- #
# Do the replicates differ? (24 Oct 2026)
# --------------------------------------------------------------------------- #

#: Asked 24 Oct 2026: "what about not a per day replicate bootstrap, but kind of a random bootstrap? all movies are
#: ordered randomly and then variance is calculated?"; built as QC's question ("1, good idea"). A p below this marks
#: the condition: a look-at-this, never a verdict (with twenty conditions one is marked by chance).
REPLICATE_SHUFFLE_P = 0.05
#: A movie with fewer squared displacements than this gives a slow fraction that is mostly noise: left out, counted.
REPLICATE_MIN_STEPS = 50
#: ``(heading, rule)``: the card's columns, each rule on its heading's hover.
REPLICATE_COLUMNS = (
    ("Replicates", "The condition's replicates and the movies each holds. A movie with fewer than "
                   f"{REPLICATE_MIN_STEPS} steps is left out."),
    ("Slow share", "Each replicate's slow fraction: the mean over its movies, from a two-population step-weighted "
                   "SQD fit with one D for the condition."),
    ("Spread", "The SD of the replicate values, in percentage points."),
    ("By chance", "The same SD when the movies are dealt at random into groups of the replicates' sizes "
                  "(4,999 deals): median and 2.5 to 97.5 % range."),
    ("p", f"The share of random deals at least as spread as the real replicates. Marked below "
          f"{REPLICATE_SHUFFLE_P:g}."),
)


def replicate_agreement(canonical) -> dict:
    """Do this condition's replicates differ more than random groups of its movies would? Plain facts for the card
    (`replicates_table`): ``{"replicates", "movies", "means_pct", "spread_pp", "null_median_pp", "null_low_pp",
    "null_high_pp", "p", "D_slow", "left_out"}``, or ``{"note": why}`` when it cannot be asked (one replicate, a
    replicate of a single movie). The fit is QC's own -- two populations, step weighting, D shared -- so the card
    reads the same whatever the SQD section was last run with."""
    from .aggregate import _fit_fractions
    from .nested import NestedError, replicate_shuffle
    from .observables import squared_displacements
    from .sqd import fit_sqd

    inside = canonical[canonical["cell"].notna()] if "cell" in canonical.columns else canonical
    if inside["day"].nunique() < 2:
        return {"note": "one replicate"}
    sd = squared_displacements(inside)
    fit = fit_sqd(sd, n_pop=2, weighting="step", gradient="analytic", ecdf_max_points=2000, n_starts=4)
    D = np.asarray(fit.D, float)
    slow = int(np.argmin(D))
    tau = float(sd.attrs["lag_time_s"])
    values, labels, left_out = [], [], 0
    for (day, _movie), g in sd.groupby(["day", "movie"], sort=True, observed=True):
        if len(g) < REPLICATE_MIN_STEPS:
            left_out += 1
            continue
        values.append(100.0 * float(_fit_fractions(g["sq_disp_um2"].to_numpy(float), D, tau)[slow]))
        labels.append(int(day))
    sizes = {r: labels.count(r) for r in labels}
    if len(sizes) < 2:
        return {"note": "one replicate with movies to compare", "left_out": left_out}
    lone = sorted(r for r, n in sizes.items() if n < 2)
    if lone:
        return {"note": f"replicate {', '.join(map(str, lone))} holds one movie", "left_out": left_out}
    try:
        test = replicate_shuffle(values, labels)
    except NestedError as exc:
        return {"note": str(exc), "left_out": left_out}
    return {"replicates": [int(r) for r in test.replicates], "movies": list(test.movies_per_replicate),
            "means_pct": list(test.replicate_means), "spread_pp": test.spread, "null_median_pp": test.null_median,
            "null_low_pp": test.null_low, "null_high_pp": test.null_high, "p": test.p, "D_slow": float(D[slow]),
            "left_out": left_out}


def replicates_table(per_condition: dict) -> dict:
    """The "Do the replicates differ?" card: one row per condition in `fair_comparison`'s shape (``columns``,
    ``rows`` of ``{"condition", "cells", "marked"}``, ``caption``, ``n_marked``), so the same table draws it."""
    rows, n_marked = [], 0
    for condition, entry in per_condition.items():
        facts = entry.get("replicates") if "error" not in entry else None
        if not facts or "note" in facts or "error" in facts:
            why = DASH if not facts else facts.get("note") or facts.get("error")
            rows.append({"condition": condition, "cells": [why] + [DASH] * (len(REPLICATE_COLUMNS) - 1),
                         "marked": [False] * len(REPLICATE_COLUMNS)})
            continue
        marked = facts["p"] < REPLICATE_SHUFFLE_P
        n_marked += int(marked)
        movies = " / ".join(str(n) for n in facts["movies"])
        left = f", {facts['left_out']} left out" if facts.get("left_out") else ""
        rows.append({"condition": condition,
                     "cells": [f"{len(facts['replicates'])} ({movies} movies{left})",
                               " / ".join(f"{m:.0f}%" for m in facts["means_pct"]),
                               f"{facts['spread_pp']:.1f} pp",
                               f"{facts['null_median_pp']:.1f} pp [{facts['null_low_pp']:.1f}, "
                               f"{facts['null_high_pp']:.1f}]",
                               f"{facts['p']:.3f}"],
                     "marked": [False, False, marked, False, marked]})
    caption = ("A small p means one replicate differs from the others by more than its movies differ among "
               "themselves. Look at a marked condition before averaging its replicates. A mark is a reason to "
               "look closer; with twenty conditions one is marked by chance.")
    return {"columns": list(REPLICATE_COLUMNS), "rows": rows, "caption": caption, "n_marked": n_marked}


#: Longest / shortest median cell length across the conditions that marks both extremes, as the track-length rule.
CELL_LENGTH_RATIO = 1.25
#: The Cell length card's rule, its heading's tooltip.
CELL_LENGTH_RULE = (
    f"Marked where the longest median is {CELL_LENGTH_RATIO:g} times the shortest or more. Cell length changes how "
    f"often a free molecule meets the wall and how many molecules a cell holds. The difference can be real, and a "
    f"merged chain of cells reads as one long cell.")


def cell_lengths(project, condition: str) -> dict:
    """One condition's OUTLINED cells' lengths along the centreline, in um: the cells of its tracked movies, the same
    cells the yield counts (`images_view._condition_cells`), each measured by `cellcoords.cell_geometry` (the
    Spatial Map's cell lengths, one calculation). ``{"lengths": [...], "with_tracks": [bool, ...], "no_centreline":
    n}``; a cell drawn as a contour only (no ribs) has no centreline and is counted, not measured.

    Asked 15 Oct 2026: "would be cool to have cell length distribution in qc"."""
    from .cellcoords import CellCoordError, cell_geometry
    from .images_view import _condition, _condition_cells

    cells, _notes = _condition_cells(project, condition)
    meshes = _condition(project, condition)["meshes"]
    um_per_pixel = float(project.params.um_per_pixel)
    by_key = {f"{movie}:{m.cell_id}": m for movie, group in meshes.items() for m in group}
    lengths, with_tracks, no_centreline = [], [], 0
    for key, n_tracks in zip(cells["cell"], cells["n_tracks"]):
        mesh = by_key.get(str(key))
        if mesh is None or not mesh.has_centerline:
            no_centreline += 1
            continue
        try:
            length = cell_geometry(mesh, um_per_pixel=um_per_pixel).length_um
        except (CellCoordError, ValueError):
            no_centreline += 1
            continue
        if math.isfinite(length):
            lengths.append(float(length))
            with_tracks.append(int(n_tracks) > 0)
    return {"lengths": lengths, "with_tracks": with_tracks, "no_centreline": no_centreline}


def cell_length_by_key(project, keys) -> dict:
    """``{cell key: length along its centreline, um}`` for the cells named in ``keys`` (the table's ``cell`` values,
    ``"<movie>:<cell_id>"``), measured by `cellcoords.cell_geometry` from the project's outlines -- the same
    calculation as `cell_lengths`. A cell with no outline or no centreline is left out (a contour has no axis)."""
    from .cellcoords import CellCoordError, cell_geometry

    wanted = {str(k) for k in keys}
    um_per_pixel = float(project.params.um_per_pixel)
    out = {}
    for movie, group in (getattr(project, "meshes", None) or {}).items():
        for mesh in group:
            key = f"{movie}:{mesh.cell_id}"
            if key not in wanted or not getattr(mesh, "has_centerline", False):
                continue
            try:
                length = cell_geometry(mesh, um_per_pixel=um_per_pixel).length_um
            except (CellCoordError, ValueError):
                continue
            if math.isfinite(length):
                out[key] = float(length)
    return out


def cell_length_comparison(project, conditions) -> dict:
    """The QC General page's Cell length card: per condition the spread of its outlined cells' lengths (5, 25, 50,
    75 and 95th percentiles, the longest, how many cells, how many hold a kept track), on one axis for all. A
    median `CELL_LENGTH_RATIO` times the shortest median or more marks both extremes, as the fair
    table marks track length: cell length sets how often a free molecule meets the wall (SynR: a free protein in a
    0.9 um cell reads a 0.2 um2/s population) and how many molecules a cell holds -- and it can be the biology
    itself (cells grow longer in rich medium). A row that cannot be measured carries its reason.

    ``{"rows": [{"condition", "n", "n_with_tracks", "median", "q25", "q75", "p5", "p95", "longest",
    "median_with_tracks", "marked", "error"}], "axis": (lo, hi), "caption", "rule", "n_marked"}``."""
    from .canonical import CanonicalError
    from .images_view import ImagesError

    rows = []
    for condition in conditions:
        try:
            got = cell_lengths(project, condition)
        except (CanonicalError, ImagesError, KeyError, ValueError, AttributeError) as exc:   # named on its row
            rows.append({"condition": condition, "error": f"not measured: {exc}"})
            continue
        lengths = np.asarray(got["lengths"], dtype=float)
        if not len(lengths):
            why = (f"{got['no_centreline']} outlined cell(s), none with a centreline (contours only)"
                   if got["no_centreline"] else "no outlined cell in a tracked movie")
            rows.append({"condition": condition, "error": why})
            continue
        held = lengths[np.asarray(got["with_tracks"], dtype=bool)]
        p5, q25, med, q75, p95 = (float(v) for v in np.percentile(lengths, [5, 25, 50, 75, 95]))
        rows.append({"condition": condition, "n": int(len(lengths)), "n_with_tracks": int(len(held)),
                     "median": med, "q25": q25, "q75": q75, "p5": p5, "p95": p95,
                     "longest": float(lengths.max()),
                     "median_with_tracks": float(np.median(held)) if len(held) else None,
                     "no_centreline": int(got["no_centreline"]), "error": None})
    measured = [r for r in rows if r.get("error") is None]
    marks = _spread_marks({r["condition"]: r["median"] for r in measured}, ratio=CELL_LENGTH_RATIO)
    for r in rows:
        r["marked"] = r["condition"] in marks
    if measured:
        lo = min(r["p5"] for r in measured)
        hi = max(r["p95"] for r in measured)
        pad = 0.05 * (hi - lo or 1.0)
        axis = (max(0.0, lo - pad), hi + pad)
    else:
        axis = (0.0, 1.0)
    rule = CELL_LENGTH_RULE
    caption = ("Outlined cells of the tracked movies, length along the centreline. Bar: middle half; tick: "
               "median; whiskers: 5 to 95 %. Hover a row for its numbers.")
    return {"rows": rows, "axis": axis, "caption": caption, "rule": rule, "n_marked": len(marks)}


def cell_length_hover(row: dict) -> str:
    """One row's numbers, for its tooltip."""
    if row.get("error"):
        return f"{row['condition']}: {row['error']}"
    text = (f"{row['condition']}: {row['n']} outlined cells, median {row['median']:.2f} um "
            f"(middle half {row['q25']:.2f} to {row['q75']:.2f}; 5 to 95 % {row['p5']:.2f} to {row['p95']:.2f}; "
            f"longest {row['longest']:.2f})")
    if row.get("median_with_tracks") is not None:
        text += f". {row['n_with_tracks']} hold a kept track, median {row['median_with_tracks']:.2f} um"
    if row.get("no_centreline"):
        text += f". {row['no_centreline']} drawn as a contour only, not measured"
    if row.get("marked"):
        text += f". Marked: the medians differ by a factor of {CELL_LENGTH_RATIO:g} or more"
    return text.replace(" um", " \u00b5m")


def contributes_nothing(project, per_condition: dict) -> dict:
    """Everything that is in the folders but not in the numbers, per condition, as plain lines: movies the
    tracker gave no tracks, movies left out for want of outlines, outlines holding no cell, movies the cell
    filter emptied, low-yield days, and the track cap and exclusions in force. ``{"rows": [(condition,
    [line])], "n": total lines, "caption"}``; a condition with nothing to say is left out."""
    canonical = getattr(project, "canonical", None)
    attrs = getattr(canonical, "attrs", {}) or {}
    source = getattr(project, "movie_source", None) or {}
    roots = getattr(project, "roots", None) or {}

    def own(qualified_names, condition):
        """The movies of ``condition`` among merged names (``<condition>_<movie>``), by their own names. Which
        condition a movie is from is read from where it was loaded (`Project.movie_source`) when recorded --
        condition names can prefix each other -- and from the prefix otherwise."""
        root = roots.get(condition)
        prefix = f"{condition}_"
        out = []
        for name in qualified_names or ():
            name = str(name)
            mine = (source[name][0] == root) if (name in source and root is not None) else name.startswith(prefix)
            if mine:
                out.append(name[len(prefix):] if name.startswith(prefix) else name)
        return sorted(out)

    empty = attrs.get("empty_movies_by_condition") or {}
    cap = attrs.get("track_cap") or None
    exclusions = attrs.get("exclusions") or []
    touched = list(attrs.get("exclusion_conditions") or [])

    def _excluded_from(item, condition):
        """A table from before `exclusion_conditions`: the item's own condition if it records one, else its key's
        prefix (movies and cells are qualified ``<condition>_<movie>``). Never "every condition"."""
        if item.get("condition") is not None:
            return str(item["condition"]) == condition
        return str(item.get("key", "")).startswith(f"{condition}_")

    rows = []
    for condition, entry in per_condition.items():
        lines = []
        if empty.get(condition):
            lines.append("no tracks from the tracker: " + ", ".join(empty[condition]))
        dropped = own(attrs.get("dropped_unsegmented"), condition)
        if dropped:
            lines.append("no outlines, left out: " + ", ".join(dropped))
        no_cell = own(attrs.get("unsegmented_movies"), condition)
        if no_cell:
            lines.append("outlines hold no cell, every track dropped: " + ", ".join(no_cell))
        block = entry.get("in_cell") if "error" not in entry else None
        if block:
            emptied = [(n, f) for n, f, k in zip(block["names"], block["found"], block["inside"])
                       if f > 0 and k == 0]
            names = own([n for n, _f in emptied], condition) or [n for n, _f in emptied]
            names = [n for n in names if n not in no_cell]
            if names:
                lines.append("every track outside the outlines: " + ", ".join(names))
        if entry.get("low_yield_days"):
            lines.append(yield_text(entry["low_yield_days"]).replace("low yield", "low track yield"))
        if cap:
            removed = [r for r in cap.get("removed", []) if r.get("condition") == condition]
            lines.append(f"track cap: {len(removed)} track(s) over {cap.get('max_steps')} steps left out")
        # AN EXCLUSION BELONGS TO THE CONDITION ITS UNIT WAS IN (4 Nov 2026, found by the overnight tester: one movie
        # of "protein A, treatment" was listed under every condition): read from where each item removed tracks
        # (`exclusions.apply_exclusions`), or for a table recorded before that, from its key's own condition
        mine = [touched[i][condition] if i < len(touched) else None for i, e in enumerate(exclusions)
                if (condition in touched[i] if i < len(touched) else _excluded_from(e, condition))]
        if mine:
            removed = sum(n for n in mine if n is not None)
            lines.append(f"{len(mine)} exclusion{'' if len(mine) == 1 else 's'} made by hand"
                         + (f": {removed:,} track{'' if removed == 1 else 's'} left out"
                            if any(n is not None for n in mine) else ""))
        if lines:
            rows.append((condition, lines))
    n = sum(len(lines) for _c, lines in rows)
    caption = ("Data in the folders that is not in the numbers, and why. A movie listed here adds nothing to "
               "any fit, so the sample can be smaller than the folder suggests."
               if n else "Every movie contributes tracks, and nothing is capped or excluded.")
    return {"rows": rows, "n": n, "caption": caption}


def compute_qc_neighbour_share(canonical, *, k: float = 3.5):
    """Per movie: the share of localizations that have another localization
    within the tracker's search radius in the SAME frame.

    A measurement, not a model: the whole-field estimate the per-frame panel
    used (`A / (pi r^2)` localizations per frame) assumes molecules spread
    evenly over the image, and they sit in cells covering a few per cent of it,
    which the project's own `trackmate_run.linking_ambiguity` records as
    understating the ambiguity about fivefold. Counting neighbours from the
    positions has no such assumption.

    The search radius and the pixel size come from ``canonical.attrs["params"]``
    and are never defaulted: without both, this raises rather than report a
    number for a radius nobody chose. The radius is the one the table's own
    condition was linked at (`canonical.table_params`; tester D10, 4 Nov 2026: a
    project linked at 5 and 8 px counted both at 5); a table spanning conditions
    linked at different radii raises.

    IT IS A LOWER BOUND. The canonical table holds only localizations that
    survived tracking and the cell filter, so a neighbour that was dropped
    before this point is not counted. An exact figure needs the detections
    before linking.

    One k-d tree serves every (movie, frame) group at once: each group's
    points are shifted along x by a whole number of steps wider than the field
    plus the radius, so points from different frames are never within reach of
    each other and only same-frame pairs are found.

    Movies unlike the rest are flagged with `qc.flag_outliers` (median ± k·MAD,
    the rule every other panel uses), never dropped. Returns a DataFrame with
    one row per movie; ``attrs`` carries the radius and the flagging band.
    """
    from scipy.spatial import cKDTree

    from .qc import flag_outliers

    from .canonical import AcquisitionParams, CanonicalError, table_params

    params = canonical.attrs.get("params")
    if isinstance(params, AcquisitionParams):
        try:
            params = table_params(canonical)          # the condition's own radius
        except CanonicalError as exc:
            raise ValueError(f"neighbour share: {exc}") from exc
    radius_px = getattr(params, "search_radius_px", None)
    pixel_nm = getattr(params, "pixel_size_nm", None)
    if not radius_px or not pixel_nm:
        raise ValueError(
            "The neighbour share needs the search radius and the pixel size, "
            "and the data record "
            f"{'neither' if not radius_px and not pixel_nm else 'only one'} of them")
    radius_um = float(radius_px) * float(pixel_nm) / 1000.0

    table = canonical[["movie", "frame", "x_um", "y_um"]].dropna()
    if table.empty:
        raise ValueError("no localisations to measure")
    group = table.groupby(["movie", "frame"], observed=True, sort=False).ngroup().to_numpy()
    x = table["x_um"].to_numpy(dtype=float)
    y = table["y_um"].to_numpy(dtype=float)
    step = float(np.ptp(x)) + 10.0 * radius_um + 1.0
    tree = cKDTree(np.column_stack([x + group * step, y]))
    pairs = tree.query_pairs(radius_um, output_type="ndarray")
    has_neighbour = np.zeros(len(table), dtype=bool)
    if len(pairs):
        has_neighbour[pairs.ravel()] = True

    movie = table["movie"].astype(str).to_numpy()
    per_frame = (pd.DataFrame({"movie": movie, "frame": table["frame"].to_numpy()})
                 .groupby(["movie", "frame"], sort=False).size())
    frame_stats = per_frame.groupby(level=0).agg(["median", "max"])
    counts = pd.DataFrame({"movie": movie, "with_neighbour": has_neighbour}).groupby(
        "movie", sort=True).agg(n_localizations=("with_neighbour", "size"),
                                with_neighbour=("with_neighbour", "sum"))
    out = counts.join(frame_stats.rename(columns={"median": "typical_per_frame",
                                                  "max": "busiest_per_frame"})).reset_index()
    out["n_frames"] = out["movie"].map(per_frame.groupby(level=0).size())
    out["share_pct"] = 100.0 * out["with_neighbour"] / out["n_localizations"]
    flags = flag_outliers(out["share_pct"].to_numpy(), k=k)
    out["flagged"] = flags.flagged
    out.attrs.update({
        "radius_px": float(radius_px),
        "radius_um": radius_um,
        "k": k,
        "median": flags.median,
        "low": flags.low,
        "high": flags.high,
        "scale": flags.scale,
        "scale_estimator": flags.scale_estimator,
        "n_movies": len(out),
        "n_flagged": int(flags.flagged.sum()),
    })
    return out


def _own_cell_filter_counts(project, condition: str, canonical):
    """``canonical`` with its per-movie cell-filter counts cut down to THIS condition.

    `Project.condition` copies the project table's ``attrs`` whole, and those carry
    the counts for EVERY movie of EVERY condition, so `qc.in_cell_share` on one
    condition's table listed the other conditions' movies too (Malate's on the
    Glucose page) and pooled them into the condition's share. A movie the cell
    filter emptied has no rows, so its condition cannot be read off the table; the
    project records where each qualified movie came from (`movie_source`), and that
    is compared with the condition's own root. Without that record (a hand-built
    project) only movies with rows in this table are kept: a movie is never
    attributed to a condition by guessing.
    """
    by_movie = canonical.attrs.get("cell_filter_by_movie")
    if not by_movie:
        return canonical
    source = getattr(project, "movie_source", None)
    root = getattr(project, "roots", {}).get(condition)
    if source and root is not None:
        mine = {m: v for m, v in by_movie.items()
                if m in source and source[m][0] == root}
    else:
        present = set(canonical["movie"].astype(str))
        mine = {m: v for m, v in by_movie.items() if str(m) in present}
    out = canonical.copy()
    out.attrs = dict(canonical.attrs)
    out.attrs["cell_filter_by_movie"] = mine
    return out


def reordered(result: dict, order) -> dict:
    """A QC result (`compute_qc_overview` and its parts) with every per-condition list in the set condition ``order``
    (`condition_order`, 3 Nov 2026): the Overview's rows, and the rows of "Is the comparison fair?", "Do the replicates
    differ?", the cell lengths and "Contributes nothing". Nothing is measured again."""
    from .condition_order import apply, sort_by

    def condition_of(row):
        return row.get("condition") if isinstance(row, dict) else (row[0] if row else None)

    def rows_sorted(part):
        if isinstance(part, dict) and isinstance(part.get("rows"), list):
            return dict(part, rows=sort_by(part["rows"], order, condition_of))
        return part

    out = dict(result)
    per = result.get("per_condition")
    if isinstance(per, dict):
        out["per_condition"] = {k: per[k] for k in apply(list(per), order)}
    for name in ("fair", "replicates", "cells", "nothing"):
        if name in result:
            out[name] = rows_sorted(result[name])
    return out


def compute_qc_overview(project, spec) -> dict:
    """Every loaded condition's QC facts, in one call.

    Mirrors the shape `catalogue_view` already established and every browser view in this workspace already
    returns: ``{"per_condition": {name: {...}}}``. Each entry carries what the Overview row shows (n_tracks,
    n_localizations, n_cells, median_track_length, n_flagged, in_cell_share_pct), the plain blocks the simple
    Detail cards draw (``length``, ``groups``, ``in_cell``, ``neighbour``, ``units``: one block per outlier unit
    or its error, ``frames``: one block per per-frame view), all plain, computed once rather than twice.

    ``in_cell_share`` raising `QCError` (a canonical table with no per-movie cell-filter counts) is caught PER
    CONDITION and surfaced as ``in_cell_share_error``, never swallowed into a blank panel. A condition that cannot
    be summarised (a missing condition, a table the spine refuses) is named in ``error`` and the rest of the run
    carries on; anything unexpected is raised, not turned into a message.
    """
    from .observables import ObservableError
    from .project import ProjectError
    from .qc import (
        QCError,
        group_counts,
        in_cell_share,
        length_distribution,
        track_summary,
        unit_summary,
    )
    from .sqd import SQDError

    per_condition = {}
    for condition in spec.conditions:
        try:
            canonical = _own_cell_filter_counts(project, condition,
                                                project.condition(condition))
            summary = track_summary(canonical)
            lengths = length_distribution(canonical, summary=summary)
            groups = group_counts(canonical, summary=summary)
            units = unit_summary(canonical, unit="cell", summary=summary)
            # The outlier unit (the original's "outlier unit" box): the same flagging rule over cells, movies or
            # days. Each is its own try, so a unit that cannot be formed (a single-session project has nothing to
            # say per day) is named on its own view instead of blanking the others.
            unit_summaries = {"cell": units}
            for other in ("movie", "day"):
                try:
                    unit_summaries[other] = unit_summary(
                        canonical, unit=other, summary=summary)
                except QCError as exc:
                    unit_summaries[other] = {"error": f"{type(exc).__name__}: {exc}"}

            try:
                in_cell = in_cell_share(canonical)
                in_cell_pct = float(in_cell.attrs["share"] * 100.0)
                in_cell_block = in_cell_payload(in_cell)
                in_cell_error = None
            except QCError as exc:
                in_cell_block = None
                in_cell_pct = None
                # the spine's refusal names its functions and a cache flag; the card says it in the user's words
                in_cell_error = (IN_CELL_NOT_MEASURED if not canonical.attrs.get("cell_filter_by_movie")
                                 else str(exc))

            low_yield, yield_note = low_yield_days(project, condition)

            try:
                neighbour = neighbour_payload(compute_qc_neighbour_share(canonical))
                neighbour_error = None
            except (ValueError, QCError) as exc:   # named on the card, never swallowed
                neighbour = None
                neighbour_error = f"{type(exc).__name__}: {exc}"

            try:
                replicates = replicate_agreement(canonical)
            except (ValueError, QCError, SQDError, ObservableError) as exc:   # refused: named on the card
                replicates = {"error": f"{type(exc).__name__}: {exc}"}

            per_condition[condition] = {
                "replicates": replicates,
                "neighbour": neighbour,
                "neighbour_share_error": neighbour_error,
                "n_tracks": int(summary.attrs["n_tracks"]),
                "n_localizations": int(summary.attrs["n_localizations"]),
                "n_cells": int(summary["cell"].nunique()),
                "median_track_length": float(summary["n_steps"].median()),
                "n_flagged": int(units["flagged"].sum()),
                "low_yield_days": low_yield,
                "yield_note": yield_note,
                "in_cell_share_pct": in_cell_pct,
                "in_cell_share_error": in_cell_error,
                "length": length_payload(lengths),
                "groups": group_table(groups),
                "units": unit_payloads(unit_summaries),
                "in_cell": in_cell_block,
                # what the fair-comparison table reads (`fair_comparison`)
                "fair": _fair_facts(project, condition, canonical, summary, in_cell_pct,
                                    in_cell if in_cell_error is None else None, neighbour),
            }
        except (KeyError, ProjectError, QCError, ValueError) as exc:   # one bad condition, not the whole run
            per_condition[condition] = {"error": str(exc)}
    return {"per_condition": per_condition,
            "fair": fair_comparison(project, per_condition),
            "cells": cell_length_comparison(project, list(spec.conditions)),
            "nothing": contributes_nothing(project, per_condition),
            "replicates": replicates_table(per_condition)}


def _fair_facts(project, condition, canonical, summary, in_cell_pct, in_cell, neighbour) -> dict:
    """One condition's facts for `fair_comparison`, all plain."""
    trackers = canonical.attrs.get("trackers") or {}
    names = {"utrack": "u-track", "trackmate": "TrackMate", "table": "Track table"}
    tracker = trackers.get(condition)
    present = set(canonical["movie"].astype(str))
    rescaled = sum(1 for m in (canonical.attrs.get("tau_rescaled") or {}) if str(m) in present)
    tau_s = getattr(getattr(project, "params", None), "tau_s", None)
    found = int(in_cell.attrs.get("tracks_found", 0)) if in_cell is not None else 0
    near = int(in_cell.attrs.get("tracks_near_edge", 0)) if in_cell is not None else 0
    measured = in_cell is not None and "cell_filter_near_edge_by_movie" in canonical.attrs
    return {
        "tracker": names.get(tracker, tracker),
        "n_days": int(canonical["day"].nunique()) if "day" in canonical else 0,
        "n_movies": int(canonical["movie"].nunique()),
        "tau_ms": None if tau_s is None else round(float(tau_s) * 1000.0, 3),
        "tau_rescaled": int(rescaled),
        "median_steps": float(summary["n_steps"].median()),
        "yield": _condition_yield(project, condition),
        "in_cell_pct": in_cell_pct,
        "near_edge_pct": (100.0 * near / found) if measured and found else None,
        "crowding_pct": None if neighbour is None else neighbour.get("median"),
        "alignment": (canonical.attrs.get("outline_alignment") or {}).get(condition),
    }
