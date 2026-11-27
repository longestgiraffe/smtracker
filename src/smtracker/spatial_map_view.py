"""The Spatial Map's numbers, rules and sentences, with no Qt anywhere in it.

`ui/workspace/spatial_map_chart.py` is the Qt shell (the capsule charts, the grid and matrix cards, the
Run dialog); this is what it draws from. They are split because of the Thin Qt rule
(docs/gui_tool_views.md; `tests/test_ui_thin_qt.py` walks the whole GUI layer): a wrong number or a
wrong sentence must fail in a headless test, not look subtly wrong on screen, and a process-pool worker
should import this module without importing Qt.

**No new estimator.** This is the SAME two-population fit the fractions come from, read per STEP instead
of per condition (`spatialsqd.spatial_sqd_map`), plus apparent diffusion (`spatialappd`) and the HMM
(`spatialhmm`). It reads the project ALREADY LOADED (`project.projected`), never `compute_condition`,
which re-reads the disk and needs a day map covering every movie (see `compute_spatial_map_sqd`).

What lives here:
  * the compute, one condition at a time and picklable (`compute_spatial_map`), returning plain lists,
    floats and strings: the `n_min` mask and the count-weighted, capsule-aware smoothing
    (`finish_spatial_panels`), the robust colour limits, the bandwidth and the localisation sigma with its
    provenance;
  * the rules that decide what a picture means: how colour limits are pooled across conditions
    (`panel_limits`, sparse rule A), when a panel is greyed (rule C), and the hard-assignment cliff
    (`cliff_status`: at two populations hard assignment is a length threshold, a population whose best
    posterior is at or below 0.5 is never the argmax and its map is EMPTY, narrowly above it the map is
    barely better than a coin flip);
  * every sentence printed on the figure or in the Run dialog (`caption_for`, `matrix_caption_rows`,
    `figures_text`, `setting_readout`, `bin_warning`), so the figure and the dialog cannot disagree with
    the numbers beside them;
  * the dialog's decisions: which keys each method's compute takes (`map_config`) and which frame is used
    when the measured one is missing (`resolve_frame`);
  * WHICH D a fitted map uses (4 Nov 2026): its own fit, or with the box "Use the same D as the ... fit" ticked the
    section's (`map_fit_source`, the rule; each compute's ``section_fit``, the D held and only fractions or
    transitions fitted; the payload's ``section_fit``, said in the Fit row and the receipt);
  * THE BIN, one camera pixel by default (ruled 30 Oct 2026; was 100 nm): `resolve_bin_nm` is the one rule
    from a choice ("px" or "nm") to nm, `project_pixel` reads the loaded project's pixel size (the loader's
    single reference when conditions differ by up to 1 %, said in a note), and `compute_spatial_map`
    resolves a pixel bin from the project it holds. A pixel bin with no pixel size is REFUSED by name
    (`BinSizeError`), never drawn at a stand-in. Invariant: every payload's ``bin_nm`` is the nm it was
    drawn at (what `settings_key` compares); ``bin_camera_px`` marks a bin read from the pixel, and only
    then does a caption say "106 nm (1 camera pixel)" (`bin_text`);
  * what a clicked spot on a day's map is made of (`spot_contributors`, `spot_rows`): the tracks and
    cells behind it, through the SAME smoothing kernel the picture is drawn with, so the shares add up to
    the colour that was clicked.

Apparent diffusion is split where its FITTED populations cross (`spatialappd.fit_appd_split`), the old
program's own rule ("set the boundaries according to the intersection of the individual populations"), so
the cut is per condition and each caption names it; two populations give two maps, three give three
(`kinds_for`).

SDA (`smtracker.spatialsda`, `compute_spatial_map_sda`) is the one method whose maps are not all
densities: its location map is a probability per bin, and its speed, D and acceleration maps are a MEAN
per bin in a physical unit. Such a panel says so itself (`quantity`, `unit`), and every rule that paints
or pools a panel reads it (`paint_field`, `limits_key`, `panel_unit`): a mean is never turned into a
density, and two different units never share a colour scale. Its value maps may widen their kernel
bin by bin where steps are few (`adaptive_smoothing`, the Run's "Adaptive smoothing"), and a click on
one lists its whole value bin (`spot_contributors`).
"""
from __future__ import annotations

from .chart_text import unit_word  # the word shown for a grouping unit

import functools

import math

from .colour_ramps import V2_HOT_KEY
from .spatialappd import APPD_POPULATION_NAMES
from .spatialsqd import LONG_AXES, RANGE_KINDS

__all__ = [
    "ASSIGNMENTS", "CLIFF_FLOOR", "CLIFF_NARROW", "DEPOSIT_LENGTH_UM", "DEPOSIT_WIDTH_UM",
    "KINDS_BY_METHOD", "kinds_for", "panel_keys_for", "panel_titles_for", "MAP_N_POPS",
    "N_POP_METHODS",
    "METHODS", "METHOD_LABELS", "PANEL_KEYS_BY_METHOD", "PANEL_MAX_MASKED_FRAC",
    "PER_LOCALISATION_METHODS", "SLA_RADIUS_RANGE_NM", "SLA_DWELL_RANGE",
    "SLA_NULL_REPLICATES", "compute_spatial_map_sla", "sla_chance", "sla_chance_text",
    "sla_null_D", "sla_example",
    "PANEL_N_MIN", "PANEL_TITLES_BY_METHOD", "ROBUST_HI_PCT", "ROBUST_LO_PCT", "SCALE_LEVELS",
    "SETTING_NAMES", "MATRIX_SCALES", "matrix_limits", "matrix_scale_text",
    "matrix_bar_note", "bar_layout", "BAR_LAYOUTS", "HIDE_REASON", "hidden_tracks_text", "scale_range_text", "neff_caption", "STEP_LABELS", "STEP_POSITIONS", "bandwidth_text", "bin_warning", "caption_for",
    "cliff_status", "cliff_warning", "cliff_warning_for", "map_fit_warnings", "map_warning", "map_warning_for",
    "no_switching_warning",
    "MAP_N_POPS_BY_METHOD", "MAP_N_POP_DEFAULT", "map_n_pops", "compute_spatial_map",
    "BEST", "check_n_pop", "drawn_n_pop", "condition_config", "planned_counts", "planned_n_pops", "cell_drawn",
    "row_n_pop", "not_drawn_text", "not_drawn_lines", "empty_cell_lines", "bic_margin_text", "KIND_ORDER", "kinds_for_counts",
    "ALL_JUMPS_KIND", "population_kinds", "drawn_total",
    "panel_titles_for_counts", "panel_keys_for_counts", "population_choice_text",
    "SPLITS", "SPLIT_DEFAULT", "RANGE_UNITS", "MANUAL_RANGES_MAX", "MANUAL_RANGES_DEFAULT", "check_ranges",
    "range_title", "manual_split_text", "split_preview", "range_shares", "ranges_from_fit", "ranges_refusal",
    "split_value_text", "split_readout", "RANGE_KINDS",
    "compute_spatial_map_appd", "compute_spatial_map_hmm", "compute_spatial_map_sqd", "even_bins",
    "figures_text", "is_blank", "map_config", "matrix_caption_rows",
    "matrix_figures_text", "measure_cell_frame", "panel_limits", "resolve_frame", "setting_name",
    "setting_readout", "setting_value_text", "settings_key", "assignment_histogram", "ASSIGNMENT_BINS",
    "RENDERINGS", "SMOOTH_FACTOR", "rendering_text", "smooth_display",
    "DAYS_HINT", "day_label", "RENDERING_ROW",
    "NEFF_SCALE_TOP", "NEFF_TOP_STEPS", "neff_scale_top", "neff_tops", "neff_scale_text", "neff_position", "neff_display_panel", "neff_bar_mark",
    "WEIGHTINGS", "CUTOFF_MODES", "SPLIT_FROM", "MAX_LOCS_MIN", "MAX_LOCS_MAX",
    "NEFF_UNITS", "NEFF_MIN_MIN", "NEFF_MIN_MAX", "masking_field", "neff_pair", "pair_colour_index", "PAIR_KEY", "PAIR_TEXT", "neff_of", "neff_hover_text",
    "apply_track_cutoff", "track_census", "cutoff_cost", "length_histogram", "cutoff_text",
    "population_picture", "N_EFF_EXPLAINED", "N_EFF_ABOUT", "SPOT_TOP_TRACKS", "spot_contributors", "spot_rows",
    "participation",
    "CELL_CLASSES", "CELL_CLASS_LABELS", "LONG_AXES", "LONG_AXIS_LABELS", "apply_cell_class",
    "cell_class_range_text", "cell_class_text", "long_axis_text",
    "AXIS_PICTURE_CELLS", "long_axis_picture", "setting_help",
    "compute_spatial_map_sda", "finish_value_panels", "finish_sda_panels", "paint_field",
    "limits_key", "panel_unit", "has_panel_units", "grid_title", "DEFAULT_CMAP_BY_METHOD",
    "default_cmap_for", "colormap_text", "PER_PANEL_UNIT_TITLE", "own_scale_ends",
    "sda_values_text", "sda_limits_text", "sda_grey_text", "sda_value_bins_text",
    "sda_example", "STEP_METHODS",
    "SDA_EXAMPLE_UM", "SDA_EXAMPLE_KEY", "sda_example_line", "n_min_unit", "bar_number", "ZERO_FLOOR_SHARE", "from_zero",
    "effective_steps", "adaptive_smoothing", "sda_value_smoothing_text", "adaptive_example",
    "adaptive_example_line", "ADAPTIVE_EXAMPLE_KEY", "spot_bin_text",
    "BIN_UNITS", "BIN_UNIT_DEFAULT", "BIN_CAMERA_PX", "BinSizeError", "nm_text", "bin_number",
    "project_pixel", "resolve_bin_nm", "camera_pixel_bin", "bin_text", "bin_choice_text",
    "bin_unit_labels", "dialog_bin", "bin_page_warning",
]

#: The MSD deposit's own capsule, the frame `manuscript/sqdmap`'s panels are drawn in. Offered beside the
#: frame measured from the open project's own outlines (`docs/gui_tool_views.md` lists the map geometry as
#: an open question and names exactly this pair), so both are drawable and the caption says which.
DEPOSIT_LENGTH_UM = 2.94
DEPOSIT_WIDTH_UM = 1.00

#: The render spec's own thresholds for greying a panel, restated from `comparison.SPATIAL_N_PANEL_MIN` /
#: `SPATIAL_MAX_MASKED_FRAC`: two views greying at two thresholds would be worse than either being wrong.
PANEL_N_MIN = 150
PANEL_MAX_MASKED_FRAC = 0.80

#: The spec's robust clip, sparse rule B: the 2nd and 98th percentiles of the unmasked bins.
ROBUST_LO_PCT = 2.0
ROBUST_HI_PCT = 98.0

#: A COLOUR SCALE OF A COUNT, A DENSITY OR A SHARE STARTS AT 0 when its low end is at most this share of its top (30 Oct
#: 2026, asked: a Spatial Map's bar read "9.39e-06" to "0.0105" -- "a scale for counts/density/share starts at 0 when
#: its minimum is within a few % of the maximum's scale"). That low end is the 2nd percentile of the bins mapped
#: (`ROBUST_LO_PCT`), and on a map with many near-empty bins it is a hair above nothing: a number that reads as a
#: measurement and says nothing. Within 5 % of the top the scale is drawn AND labelled from 0 (`from_zero`), so no
#: colour moves by more than 5 % of the bar. A MEAN per bin (SDA's speed, apparent D, acceleration: `paint_field`
#: "value") keeps its own low end, and so does a scale whose low end is further up (a dense map's).
ZERO_FLOOR_SHARE = 0.05

#: How the colour limits are pooled: `population` is sparse rule A (one scale per population, shared across
#: every condition computed at the same settings), `both` one scale for both populations as a density,
#: `panel` each panel on its own.
SCALE_LEVELS = ("population", "both", "panel")

#: Where the hard-assignment cliff sits. At or below `CLIFF_FLOOR` the slow population's best posterior never
#: wins an argmax (its hard map is empty); below `CLIFF_NARROW` it clears the floor only narrowly.
CLIFF_FLOOR = 0.5
CLIFF_NARROW = 0.55

STEP_POSITIONS = ("A_start", "B_midpoint", "C_end")
STEP_LABELS = ("start", "midpoint", "end")
ASSIGNMENTS = ("soft", "hard")
METHODS = ("sqd", "appd", "hmm", "sla", "sda")
METHOD_LABELS = {"sqd": "SQD", "appd": "Apparent Diffusion", "hmm": "HMM (Kinetics)",
                 "sla": "Confinement (SLA)", "sda": "SDA"}
#: The methods that bin per-LOCALISATION rows rather than per-step jumps.
PER_LOCALISATION_METHODS = ("appd", "sla")
#: The methods that place a STEP, so "Jump drawn at" applies (SDA's speed and D maps).
STEP_METHODS = ("sqd", "hmm", "sda")

#: Three population-column vocabularies, one per method. SQD's populations are named for the jump LENGTH that
#: separates them, appD's for its fitted populations; reusing "short"/"long" for appD would be the "two
#: datasets, one name" failure. HMM's states genuinely ARE slow/bound and fast/free, so it reuses appD's
#: "slow"/"fast" KEYS (harmless: `settings_key` starts with `method`) with its own TITLES.
#: These are the TWO-population sets; `kinds_for` gives a three-population appD its middle column.
#: EVERY SQD SPLIT ALSO DRAWS "ALL JUMPS" (3 Nov 2026, asked: "all jumps for 2 pop shows basically all jumps together
#: (as 1 pop) in the same field"): every jump at weight 1 (`spatialsqd.ALL_JUMPS_LABEL`), the map a one-population run
#: draws, so the grid's first column holds every condition on one footing whatever its count. Its kind is the
#: one-population map's own, ``"jumps"`` (`ALL_JUMPS_KIND`); `population_kinds` is a run's kinds without it.
KINDS_BY_METHOD = {"sqd": ("jumps", "short", "long"), "appd": ("jumps", "slow", "fast"),
                   "hmm": ("jumps", "slow", "fast"),
                   "sla": ("free", "mixed", "dwelling"),
                   # `spatialsda.SDA_KINDS`, restated so this module imports no compute at
                   # its top (`tests/test_spatial_sda.py` holds the two equal).
                   "sda": ("location", "speed", "diffusion", "acceleration")}

#: How many populations a map may split into (one map each), and the methods that may.
#: HMM may not: its map rebuilds each step's soft weight from the winning state and that
#: state's own posterior, which holds at exactly two states (`spatialhmm`'s docstring).
#:
#: SQD MAY DRAW ONE (1 Nov 2026). Asked, of free mNG's simulated condition, whose "long jumps" map read n = 0 while
#: its truth drew two: "maybe we need to either handle it that a 1 pop fit only 1 map gets produced, or if forced
#: into two populations, the short and long jumps should both be created?". A protein of ONE state has no short and
#: long jumps: its two-population fit returns the same D twice (free mNG v6 at 5 px, 0.861 and 0.861 um2/s), every
#: jump is then exactly as likely in either, and hard assignment hands every one to the larger share -- the other
#: map empty, by construction. Drawing both under soft assignment would draw the SAME jumps twice, scaled by the two
#: shares, which reads as two populations in one place. So one population draws ONE map, "All jumps", and a
#: two-population map that finds one says so (`map_warning`). Apparent diffusion keeps 2 and 3.
MAP_N_POPS_BY_METHOD = {"sqd": (1, *APPD_POPULATION_NAMES), "appd": tuple(APPD_POPULATION_NAMES),
                        # HMM since 3 Nov 2026 (asked: "the HMM maps, they have less options than the SQD maps"):
                        # `hmm.decode_states` keeps every state's posterior, so 1, 2 or 3 states (`spatialhmm`)
                        "hmm": (1, 2, 3)}
#: Every count some method takes, ascending.
MAP_N_POPS = tuple(sorted({n for counts in MAP_N_POPS_BY_METHOD.values() for n in counts}))
#: The count a run opens on, whichever method: two, every map made before.
MAP_N_POP_DEFAULT = 2


def map_n_pops(method: str) -> tuple:
    """The population counts `method`'s maps take (empty for a method that fits none)."""
    return MAP_N_POPS_BY_METHOD.get(method, ())


#: A COUNT PER CONDITION (1 Nov 2026). Asked: "the condition picker of how many pop i wanna assume for each condition
#: (like in sqd bubble plot, with being able to manually choose condition 1 -> 2, condition 2 -> 3 pop etc. or best,
#: or all etc.". Each condition's maps are fitted at its OWN count (`n_pop_by_condition`), or at the count BIC
#: chooses for it -- "best", the bubble plot's own rule (`comparison.resolve_n_pop`; apparent diffusion's gamma
#: mixture by `appd_gamma.select_n_pop_gamma`), among the counts the map draws. One set of maps per condition: the
#: grid's columns are every panel some condition draws (`kinds_for_counts`), a row's others left empty.
BEST = "best"


def check_n_pop(method: str, n_pop):
    """A population choice for `method`'s maps: one of `map_n_pops(method)`, or `BEST`."""
    if n_pop == BEST:
        return BEST
    if int(n_pop) not in map_n_pops(method):
        raise ValueError(f"n_pop must be one of {map_n_pops(method)} or {BEST!r} for {method}, got {n_pop}")
    return int(n_pop)


def drawn_n_pop(state: dict) -> int:
    """The count the Run dialog's pictures draw: the chosen count, or two (the default) for Best, whose count is
    each condition's own and known only once it is fitted."""
    n_pop = (state or {}).get("n_pop", MAP_N_POP_DEFAULT)
    return MAP_N_POP_DEFAULT if n_pop in (None, BEST) else int(n_pop)


def condition_config(config: dict, condition: str, chosen=None) -> dict:
    """The config ONE condition's job is handed: the Run's, with this condition's own count
    (``n_pop_by_condition``); ``chosen`` (a redraw: the count its Run already chose by BIC) keeps a Best condition at
    the count it was drawn with, so hiding a track never changes how many maps it has."""
    # the box ("Use the same D as the ... fit", `map_fit_source.TICK_KEY`) is the Run's, never a job's: a job takes
    # the section's fit itself (`map_fit_source.job_config`), or none. So is Per spot (`OWN_PRECISION_KEY`): a job is
    # handed its calibration (``"precision"``), or none (a condition whose tracks carry no precision)
    out = {k: v for k, v in config.items()
           if k not in ("n_pop_by_condition", "same_D", "fit_page", "fit_page_name", OWN_PRECISION_KEY)}
    by = config.get("n_pop_by_condition") or {}
    if "n_pop" in out and str(condition) in by:
        out["n_pop"] = by[str(condition)]
    if chosen is not None and out.get("n_pop") == BEST:
        out["n_pop"] = int(chosen)
    return out


def planned_counts(config: dict, conditions) -> tuple:
    """The fixed counts a Run will draw over `conditions` (ascending): the grid's columns before any result lands.
    A Best condition adds its count when its maps arrive (`kinds_for_counts`)."""
    if "n_pop" not in config:
        return ()
    by = config.get("n_pop_by_condition") or {}
    choices = {by.get(str(c), config["n_pop"]) for c in conditions}
    return tuple(sorted(int(n) for n in choices if n != BEST))


# -- A COLUMN A ROW NEVER DRAWS (2 Nov 2026) ------------------------------------------------------------------- #
#
# Reported: "when i fit some 1 pop and some 2 pop sqd map -> for the ones fitted with 1 pop, the 2 maps for short and
# long say waiting even though ofc they never load . other way for the one i choose 2 pop, the all jump map says
# waiting but ofc it doesnt even get cmputed". The grid's columns are every panel SOME row draws (`kinds_for_counts`),
# and every cell without a map said "waiting…", the line for a map still computing. A cell is now one of three:
# its map; WAITING for it (a row still computing at a count that draws that column, or a Best row, whose count is
# known only when it lands); or NOT DRAWN at the row's count -- left empty, and its line says so.


def planned_n_pops(config: dict, conditions) -> dict:
    """Each condition's count as a Run plans it: its own (``n_pop_by_condition``), else the Run's; `BEST` for one
    whose count BIC chooses when it is fitted. Empty for a method that fits no count."""
    if "n_pop" not in config:
        return {}
    by = config.get("n_pop_by_condition") or {}
    out = {}
    for c in conditions:
        n = by.get(str(c), config["n_pop"])
        out[str(c)] = BEST if n == BEST else int(n)
    return out


def cell_drawn(method: str, kind: str, payload: dict | None = None, planned=None, ranges=None) -> bool:
    """Whether one grid row has a map in the column ``kind``. A row whose maps have landed: if they hold one (a
    row's maps hold every panel of its count, an empty one included). A row still computing: if its planned count
    draws that column -- always for a Best row (its count is chosen when it is fitted), a manual split (every row
    the same ranges) and a method that fits no count."""
    if payload is not None:
        return kind in (payload.get("panels") or {})
    if planned in (None, BEST) or ranges or method not in KIND_ORDER:
        return True
    return kind in kinds_for(method, int(planned))


def row_n_pop(payload: dict | None, planned=None):
    """One grid row's count: its landed maps' own, else its planned fixed count; None for a Best row still
    computing (and a row of a method that fits no count)."""
    if payload is not None:
        return payload.get("n_pop")
    return None if planned in (None, BEST) else int(planned)


def bic_margin_text(n_pop, bic) -> str:
    """How clearly BIC chose a Best row's count: "BIC prefers 1 over 2 by ΔBIC = 37.4", against the runner-up among
    the counts it compared (``bic``: ``[[n, BIC], ...]``, a Best map's payload); "" without them."""
    scores = {int(n): float(b) for n, b in (bic or ())}
    if n_pop is None or int(n_pop) not in scores or len(scores) < 2:
        return ""
    from .bubble_view import bic_delta_text

    n = int(n_pop)
    other = min((m for m in scores if m != n), key=lambda m: scores[m])
    return f"BIC prefers {n} over {other} by ΔBIC = {bic_delta_text(scores[other] - scores[n])}"


def not_drawn_lines(n_pop=None, bic=None) -> list:
    """What a grid cell its row's count has no map for says (3 Nov 2026, asked: "for 1 pop fits ... these stay empty.
    or instead of empty, what else could we show?"): ONE POPULATION HAS NO SPLIT, said where the split would stand,
    with how clearly BIC chose that count when Best chose it (`bic_margin_text`). Nothing is drawn in its place: a
    split forced on one population is not the data's, and a split by hand is its own run (`SPLITS`)."""
    if n_pop is None:
        return ["not drawn for this condition"]
    n = int(n_pop)
    head = "One population: no split" if n == 1 else f"not drawn at {n} populations"
    margin = bic_margin_text(n, bic)
    return [head, margin] if margin else [head]


def empty_cell_lines(payload: dict | None, kind: str | None = None, n_pop=None, *, margin: bool = True) -> list:
    """What a grid cell with no map says, on screen and in print alike: a map its row's count DOES draw that is missing
    from maps made before it existed (All jumps beside a split, 3 Nov 2026) says to run the maps again; else
    `not_drawn_lines` at the row's count (``n_pop``, else its payload's), with a Best row's BIC margin unless
    ``margin`` is off (Numbers off in print)."""
    if payload is not None and kind is not None and kind in kinds_for(
            payload.get("method", "sqd"), payload.get("n_pop") or MAP_N_POP_DEFAULT, payload.get("ranges")):
        return ["this map is missing from the last Run: run the maps again"]
    n = n_pop if n_pop is not None else (payload or {}).get("n_pop")
    return not_drawn_lines(n, (payload or {}).get("bic") if margin else None)


def not_drawn_text(n_pop=None, bic=None) -> str:
    """The line under a grid cell its row's count has no map for (`not_drawn_lines`, on one line)."""
    return " · ".join(not_drawn_lines(n_pop, bic))


#: Each method's panel kinds in the grid's order, every count together (`kinds_for_counts`).
KIND_ORDER = {"sqd": ("jumps", "short", "middle", "long"), "appd": ("jumps", "slow", "middle", "fast"),
              "hmm": ("jumps", "slow", "middle", "fast")}
#: A column two counts share is called by its population alone where their titles differ (apparent diffusion's
#: "below the crossing" at two, "below the first crossing" at three).
_SHARED_TITLES = {"slow": "Slow", "middle": "Middle", "fast": "Fast"}


def kinds_for_counts(method: str, counts, ranges=None) -> tuple:
    """The grid's columns when conditions are fitted at different counts: every kind some count draws, in
    `KIND_ORDER`. One count (or none yet) is `kinds_for`; a manual split is its ranges."""
    counts = sorted({int(n) for n in counts or ()})
    if ranges or method not in KIND_ORDER or len(counts) <= 1:
        return kinds_for(method, counts[0] if counts else MAP_N_POP_DEFAULT, ranges)
    present = {k for n in counts for k in kinds_for(method, n)}
    return tuple(k for k in KIND_ORDER[method] if k in present)


def panel_titles_for_counts(method: str, counts, ranges=None) -> tuple:
    """`kinds_for_counts`' column titles: each kind's own title, or its population alone where the counts that draw
    it title it differently."""
    counts = sorted({int(n) for n in counts or ()})
    if ranges or method not in KIND_ORDER or len(counts) <= 1:
        return panel_titles_for(method, counts[0] if counts else MAP_N_POP_DEFAULT, ranges)
    titles: dict = {}
    for n in counts:
        for kind, title in zip(kinds_for(method, n), panel_titles_for(method, n)):
            titles.setdefault(kind, set()).add(title)
    return tuple(next(iter(titles[k])) if len(titles[k]) == 1 else _SHARED_TITLES.get(k, k.capitalize())
                 for k in kinds_for_counts(method, counts))


def panel_keys_for_counts(method: str, counts, ranges=None) -> dict:
    """Display title -> panel kind over `kinds_for_counts`, for the Visibility tool."""
    return dict(zip(panel_titles_for_counts(method, counts, ranges), kinds_for_counts(method, counts, ranges)))


def population_choice_text(payload: dict) -> str:
    """A condition's count as its map says it: "2", or "2 (Best by BIC)"."""
    n = payload.get("n_pop")
    return f"{n} (Best by BIC)" if payload.get("n_pop_choice") == BEST else str(n)


def _calibrated(calibration) -> dict:
    """``{"calibration": ...}`` where the map's own fit is Per spot, else nothing: a plain map calls every Best helper
    exactly as before (6 Nov 2026)."""
    return {} if calibration is None else {"calibration": calibration}


def _best_sqd_count(table, lag_frames, fit_weighting: str, calibration=None) -> int:
    """The count BIC chooses for an SQD map (`_best_sqd_selection`)."""
    return _best_sqd_selection(table, lag_frames, fit_weighting, **_calibrated(calibration))["n_pop"]


def _best_sqd_selection(table, lag_frames, fit_weighting: str, counts=None, calibration=None) -> dict:
    """``{"n_pop", "bic": [[n, BIC], ...]}``: the count BIC chooses for an SQD map and the BIC of every count it
    compared (3 Nov 2026: a cell a one-population row leaves empty says by how much, `bic_margin_text`). The bubble
    plot's own rule (`comparison.resolve_n_pop`'s Best: `sqd.select_n_pop`, maximum likelihood fits from the full
    multistart at the map fit's weighting), among `map_n_pops("sqd")`; lags 1-4 together choose at lag 1, as the SQD
    section does. ``counts`` (consecutive) fits only those, each exactly as in the whole selection (every count's fit
    is its own plan and seed): the check beside a count set by hand (`_forced_selection`). ``calibration`` (the map's
    own fit Per spot, 6 Nov 2026): every count fitted with each jump's own localisation error."""
    from .observables import displacements_at
    from .sqd import select_n_pop

    lag = map_lag(lag_frames)
    counts = tuple(counts or map_n_pops("sqd"))
    selection = select_n_pop(displacements_at(table, 1 if lag == "1-4" else lag), min_pop=min(counts),
                             max_pop=max(counts), method="mle", n_jobs=1, weighting=fit_weighting,
                             **({"precision": calibration} if calibration is not None else {}))
    return {"n_pop": int(selection.recommended.n_pop),
            "bic": [[int(f.n_pop), float(f.bic)] for f in sorted(selection.fits, key=lambda f: f.n_pop)]}


def _best_hmm_selection(table, tau_s: float, counts=None, calibration=None) -> dict:
    """``{"n_pop", "bic": [[n, BIC], ...]}``: the state count BIC chooses for an HMM map (3 Nov 2026), the HMM
    section's own rule (`hmm.select_n_states`), among `map_n_pops("hmm")`, on the map fit's own steps. ``counts``
    (consecutive) fits only those, each the same `fit_hmm` call as in the whole selection -- and as the map's own fit
    (`spatialhmm.spatial_hmm_map`), so the map's BIC is Best's at its count. ``calibration`` (the map's own fit Per
    spot): each step's own noise in every fit, as the map's own."""
    from .hmm import select_n_states, step_sequences

    counts = tuple(counts or map_n_pops("hmm"))
    if calibration is None:
        sequences, extra = step_sequences(table), {}
    else:
        sequences, noises = step_sequences(table, calibration=calibration)
        extra = {"noise_sequences": noises, "calibration": calibration}
    chosen = select_n_states(sequences, min_states=min(counts), max_states=max(counts),
                             tau_s=float(tau_s), **extra)
    return {"n_pop": int(chosen.recommended.n_states),
            "bic": [[int(f.n_states), float(f.bic)] for f in sorted(chosen.fits, key=lambda f: f.n_states)]}


def _best_appd_count(table, calibration=None) -> int:
    """The count BIC chooses for an apparent-diffusion map (`_best_appd_selection`)."""
    return _best_appd_selection(table, **_calibrated(calibration))["n_pop"]


def _best_appd_selection(table, calibration=None) -> dict:
    """``{"n_pop", "bic": [[n, BIC], ...]}``: the count BIC chooses for an apparent-diffusion map -- the gamma mixture
    on the map fit's own sub-track values (`spatialappd.fit_appd_split`'s), `appd_gamma.select_n_pop_gamma`, among
    `map_n_pops("appd")` -- and the BIC of every count that converged."""
    rows = _appd_bic_rows(table, map_n_pops("appd"), **_calibrated(calibration))
    if not rows:
        raise ValueError("no apparent-diffusion population count could be fitted, so Best has nothing to choose")
    return {"n_pop": int(min(rows, key=lambda r: r[1])[0]), "bic": rows}


def _appd_bic_rows(table, counts, calibration=None) -> list:
    """``[[n, BIC], ...]`` of every count in ``counts`` whose gamma mixture converged (`appd_gamma.select_n_pop_gamma`
    on the map fit's own sub-track values, `spatialappd.fit_appd_split`'s: the same fit at a count as the map's; Per
    spot, each sub-track's own noise offset as there)."""
    from .appd_gamma import select_n_pop_gamma
    from .spatialappd import split_values_with_offsets

    # the map fit's own exact-zero rule (`appd_gamma.zero_tracks`, its sub-tracks and their frame interval)
    values, tau, offsets = split_values_with_offsets(table, 4, calibration)
    selection = select_n_pop_gamma(values, n_steps=4, n_pop_values=tuple(counts), tau_s=tau, noun="sub-track",
                                   offset=offsets)
    return [[int(r.n_pop), float(r.bic)] for r in selection.itertuples() if r.bic == r.bic]


#: A COUNT SET BY HAND IS CHECKED AGAINST BEST (ruled 4 Nov 2026, `bubble_view.forced_count_warning`): a map drawn at a
#: count the user picked keeps it, and Best's scores are taken beside it on the same table, so the map can say "Best
#: (BIC) chooses 1 state here ..." where Best would choose fewer. Only where fewer exist among the map's counts, and
#: only as many counts as the answer needs (`bubble_view.forced_count_scores`: the counts below, the counts above only
#: where one below wins), each fitted as Best fits it -- so EXACT for every method (until the evening of 4 Nov 2026 an
#: SQD map scored its own least-squares fits, an approximation, to spare Best's maximum-likelihood fit at every count).
def _forced_selection(method: str, n_pop, select, own_bic=None) -> list | None:
    """The BIC rows beside a map drawn at a count set by hand, or None where none is needed: Best itself, or a count
    with no fewer among `map_n_pops(method)`. ``select(counts)``: the method's `_best_*_selection` (or
    `_appd_bic_rows`) on the map fit's table at those counts; ``own_bic``: the map's own BIC where its fit is Best's
    at the count (HMM, apparent D), else None (SQD: its map is drawn from least squares on the ECDF)."""
    from .bubble_view import forced_count_scores

    if n_pop == BEST:
        return None

    def rows(lo, hi):
        out = select(tuple(range(int(lo), int(hi) + 1)))
        return out["bic"] if isinstance(out, dict) else out

    return forced_count_scores(int(n_pop), map_n_pops(method), rows, own_bic)


#: HOW A MAP SPLITS ITS JUMPS OR TRACKS (1 Nov 2026). "fitted", every map before: the populations are fitted and
#: cut where they cross. "manual": ranges set BY HAND, one map each, nothing fitted -- asked: "manually set the D for
#: appd map and the jump distance for SQD maps so i can manually bin the jumps/tracks and see how it behaves ...
#: range D1 1 to 5, range D2 0.1 to 1 ... add D for more up to 4", SMTracker 2's own way ("u saw the histogram of D
#: and where the populations sit ... and then manually set the D to these values or others"). SQD's ranges are jump
#: LENGTHS in nm (`spatialsqd.maps_from_ranges`), apparent diffusion's each track's mean apparent D in µm²/s
#: (`spatialappd._labels_from_appd_ranges`); ``[low, high)``, ascending, never overlapping, so a jump or track goes to
#: one map or to none (counted). Each map is TITLED by its range (`range_title`), never "short" or "slow".
SPLITS = ("fitted", "manual")
SPLIT_DEFAULT = "fitted"
#: The methods a manual split applies to, and the unit their ranges are typed in.
RANGE_UNITS = {"sqd": "nm", "appd": "µm²/s"}
#: How many ranges a manual split takes (`spatialsqd.RANGE_KINDS`).
MANUAL_RANGES_MAX = len(RANGE_KINDS)
#: The ranges a manual split opens on: two each, the experimenter's own example for apparent D.
MANUAL_RANGES_DEFAULT = {"sqd": ((0.0, 150.0), (150.0, 1000.0)), "appd": ((0.1, 1.0), (1.0, 5.0))}


def check_ranges(method: str, ranges) -> tuple:
    """A manual split's ranges for `method`, checked and ascending (`spatialsqd.check_range_values`, in the method's
    unit); `ValueError` names the rule a range breaks."""
    from .spatialsqd import check_range_values

    if method not in RANGE_UNITS:
        raise ValueError(f"a manual split is for {' and '.join(sorted(RANGE_UNITS))} maps, not {method!r}")
    return check_range_values(ranges, unit=RANGE_UNITS[method])


def _range_number(value: float) -> str:
    """A range end as typed: 150, 0.1, 1.25 -- no trailing zeros, no exponent below a million."""
    text = f"{float(value):.6g}"
    return text if "e" not in text else f"{float(value):g}"


def range_title(method: str, lo: float, hi: float) -> str:
    """A manual map's title: its range in its unit, "0–150 nm", "0.1–1 µm²/s"."""
    return f"{_range_number(lo)}\u2013{_range_number(hi)} {RANGE_UNITS[method]}"


def manual_split_text(payload: dict) -> str:
    """A manual split in words, for the captions and the receipt: its ranges, and how much fell in none."""
    method = payload.get("method", "sqd")
    ranges = payload.get("ranges") or ()
    text = "set by hand: " + ", ".join(range_title(method, lo, hi) for lo, hi in ranges)
    outside = payload.get("n_outside_ranges")
    if outside:
        what = "jump" if method == "sqd" else "track"
        text += f" ({outside:,} {what}{'' if outside == 1 else 's'} in no range, not drawn)"
    return text


#: The preview's histogram (`split_preview`): log-spaced bars, and the quantiles a range's share is read from.
PREVIEW_BINS = 60
PREVIEW_QUANTILES = 2001


def split_preview(project, condition: str, *, method: str, n_pop: int = 2, lag_frames=1,
                  weighting: str = "steps", gradient: str | None = None, section_fit=None) -> dict:
    """What a manual split is set on, for one condition (asked 1 Nov 2026, SMTracker 2's way: "u saw the histogram of
    D and where the populations sit (like in jump distance histogram) and then manually set the D").

    The histogram of exactly the values the ranges cut -- SQD: every jump's LENGTH in nm at the run's lag (the jumps
    `spatialsqd.maps_from_ranges` bins); apparent diffusion: every track's MEAN apparent D in µm²/s
    (`spatialappd._labels_from_appd_ranges`) -- on `PREVIEW_BINS` log-spaced bars, with the populations the FITTED
    split finds on the same data: their D, fractions and where they cross (the fitted maps' cuts), and for SQD each
    population's expected count per bar (the exact bin integral of ``f_k exp(-r^2/m_k)``, ``m_k = 4 D_k tau``). The
    appD populations are fitted on sub-tracks (`fit_appd_split`, the fitted maps' own fit), so they are drawn as
    their D and cuts, not as curves over track means. ``quantiles`` (`PREVIEW_QUANTILES`) let `range_shares` say what
    share of the values a range holds without sending every value to the dialog. Plain lists throughout.

    ``section_fit`` (the box "Use the same D as the ... fit", `map_fit_source`): the populations drawn, and so "From
    the fit"'s cuts, are at the section's D, their fractions fitted here as the map's are -- where the fit can be read
    at this histogram (SQD: the same time lag). ``source`` says which: "section" or "own".
    """
    import numpy as np

    from .sqd import gradient_mode

    if method not in RANGE_UNITS:
        raise ValueError(f"a manual split is for {' and '.join(sorted(RANGE_UNITS))} maps, not {method!r}")
    projected = project.projected(condition)
    curves = None
    held = section_fit
    if held is not None and method == "sqd" and str(map_lag(held.get("lag_frames", 1))) != str(map_lag(lag_frames)):
        held = None                     # D at another lag cannot be read at this lag's jumps
    if held is not None and method == "appd" and int(held["n_pop"]) not in map_n_pops("appd"):
        held = None
    if held is not None:
        n_pop = int(held["n_pop"])
    if n_pop == BEST:
        # the populations BIC chooses for this condition, drawn as any count is (`_best_sqd_count`, `_best_appd_count`),
        # at the SQD section's optimiser (`compute_spatial_map`'s ``gradient``)
        with gradient_mode(gradient):
            n_pop = (_best_sqd_count(projected, lag_frames, SQD_FIT_WEIGHTING[weighting]) if method == "sqd"
                     else _best_appd_count(projected))
    if method == "sqd":
        from .observables import displacements_at
        from .spatialsqd import sqd_crossings
        from .sqd import fit_sqd

        lag = map_lag(lag_frames)
        if lag == "1-4":
            raise ValueError("A manual split cuts jump lengths at one time lag. "
                             "Pick one lag instead of 1-4 together.")
        steps = displacements_at(projected, lag)
        r2 = steps["sq_disp_um2"].to_numpy(float)
        values = 1000.0 * np.sqrt(r2[r2 > 0])
        if held is not None:
            from .spatialsqd import fit_at_fixed_D

            fit = fit_at_fixed_D(steps, held["D"], weighting=str(held.get("weighting") or "step"),
                                 calibration=held_calibration(held))
        else:
            fit = fit_sqd(steps, n_pop=int(n_pop), method="ecdf_lsq", weighting=SQD_FIT_WEIGHTING[weighting])
        order = np.argsort(np.asarray(fit.D, dtype=float))
        D = np.asarray(fit.D, dtype=float)[order]
        f = np.asarray(fit.fractions, dtype=float).ravel()[order]
        tau = float(fit.tau_lag_s)
        # the immobile population's cut at 0 (exact-zero jumps) is no length on this log axis: the moving ones' cuts
        cuts = [1000.0 * c for c in sqd_crossings(D, f, tau)["cuts_um"] if c > 0]
    else:
        from .appd import apparent_D
        from .spatialappd import fit_appd_split, split_at_D

        table = apparent_D(projected, n_steps=4)
        per_track = table.groupby("uid")["appd_um2_per_s"].mean().to_numpy(float)
        values = per_track[per_track > 0]
        fit = (split_at_D(projected, held["D"], min_steps=int(held.get("min_steps", 4)),
                          calibration=held_calibration(held)) if held is not None else
               fit_appd_split(projected, n_pop=int(n_pop) if int(n_pop) in map_n_pops("appd") else 2))
        D, f = np.asarray(fit["D"], dtype=float), np.asarray(fit["fractions"], dtype=float)
        # the immobile population's cut at 0 (exact-zero tracks) is no apparent D on this log axis
        cuts = [c for c in fit["cuts"] if float(c) > 0]
    if values.size == 0:
        raise ValueError(f"{condition} has no {'jump' if method == 'sqd' else 'track'} to draw a histogram of")
    lo = 10.0 ** math.floor(math.log10(float(np.quantile(values, 0.001))))
    hi = 10.0 ** math.ceil(math.log10(float(values.max())))
    edges = np.geomspace(lo, hi, PREVIEW_BINS + 1)
    counts = np.histogram(np.clip(values, lo, hi), bins=edges)[0]
    if method == "sqd":
        a, b = edges[:-1] / 1000.0, edges[1:] / 1000.0
        m = 4.0 * D * tau
        # each population's share of EVERY jump, the zeros included; the immobile population of exact-zero jumps
        # (a D of 0, `sqd`'s module head) has none on this log axis of lengths
        curves = [([0.0] * len(a) if mk == 0 else
                   (r2.size * fk * (np.exp(-a ** 2 / mk) - np.exp(-b ** 2 / mk))).tolist()) for fk, mk in zip(f, m)]
    return {"method": method, "condition": str(condition), "unit": RANGE_UNITS[method], "n": int(values.size),
            "edges": edges.tolist(), "counts": counts.astype(int).tolist(), "curves": curves,
            "D": D.tolist(), "fractions": f.tolist(), "cuts": [float(c) for c in cuts],
            "quantiles": np.quantile(values, np.linspace(0.0, 1.0, PREVIEW_QUANTILES)).tolist(),
            "n_pop": int(len(D)), "lag_frames": map_lag(lag_frames) if method == "sqd" else None,
            "source": "section" if held is not None else "own"}


def range_shares(preview: dict, ranges) -> list:
    """The share of a preview's values (`split_preview`) each range ``[low, high)`` holds, from its quantiles: to
    within 1 / `PREVIEW_QUANTILES`. Pure Python, so the dialog may call it as it is typed."""
    import bisect

    q = (preview or {}).get("quantiles") or []
    if not q:
        return [0.0 for _ in ranges or ()]
    return [(bisect.bisect_left(q, float(hi)) - bisect.bisect_left(q, float(lo))) / len(q)
            for lo, hi in ranges or ()]


def _nice(value: float) -> float:
    """A range end a person would type: two significant figures."""
    if value <= 0:
        return 0.0
    digits = 1 - int(math.floor(math.log10(value)))
    return round(value, digits)


def ranges_from_fit(preview: dict) -> list:
    """Ranges that start where the fitted maps cut (`split_preview`'s ``cuts``, two significant figures): from 0 to the
    first cut, cut to cut, and the last cut to the largest value seen -- the fitted split, as ranges to edit. One
    range, 0 to the largest, where the fit cuts nowhere."""
    top = (preview or {}).get("quantiles") or [0.0]
    ends = [0.0, *[_nice(c) for c in sorted((preview or {}).get("cuts") or ())], _nice(max(top) * 1.05)]
    ends = sorted(set(ends))
    return [[lo, hi] for lo, hi in zip(ends, ends[1:])][:MANUAL_RANGES_MAX]


def ranges_refusal(method: str, split: str, ranges, lag_frames=1) -> str:
    """Why the Run dialog's manual ranges cannot run, or "" (`check_ranges`, and one lag for SQD): said on the Split
    page, and Run refuses (`SpatialMapDialog.refusal`)."""
    if split != "manual" or method not in RANGE_UNITS:
        return ""
    try:
        check_ranges(method, ranges)
    except ValueError as exc:
        return str(exc)[:1].upper() + str(exc)[1:] + "."
    if method == "sqd" and map_lag(lag_frames) == "1-4":
        return ("A manual split cuts jump lengths at one time lag. "
                "Pick one lag under Time lag instead of 1-4 together.")
    return ""


def split_value_text(state: dict) -> str:
    """The Split row's value in the nav: FITTED, or BY HAND and how many ranges."""
    if state.get("split") != "manual":
        return "FITTED"
    n = len(state.get("ranges_typed") or state.get("ranges") or ())
    return f"BY HAND · {n} RANGE{'' if n == 1 else 'S'}"


def split_readout(state: dict) -> str:
    """The line under the Split page's histogram: where the fitted split cuts, or what share of the shown condition's
    values each range holds (`range_shares`) and how much falls in none."""
    method = state.get("method", "sqd")
    preview = state.get("preview") if isinstance(state.get("preview"), dict) else None
    if preview is not None and ("error" in preview or preview.get("pending") or "quantiles" not in preview):
        preview = None                      # still being read, or could not be: the picture says which
    what = "jumps" if method == "sqd" else "tracks"
    unit = RANGE_UNITS.get(method, "")
    if state.get("split") != "manual":
        if preview is None:
            return "Each condition's populations are fitted, and its maps cut where they cross."
        cuts = preview.get("cuts") or []
        where = (": " + ", ".join(f"{_range_number(_nice(c))} {unit}" for c in cuts) if cuts else
                 " nowhere: one population is the most likely at every value")
        if preview.get("source") == "section":
            from .map_fit_source import FIT_NAMES

            return (f"At the {FIT_NAMES.get(method, method)} fit's D: {len(preview.get('D') or ())} populations in "
                    f"{preview['condition']}, cut where they cross{where}. Every condition uses its D "
                    f"from that fit.")
        return (f"Fitted: {len(preview.get('D') or ())} populations in {preview['condition']}, cut where they cross"
                f"{where}. Every condition is fitted and cut on its own.")
    if state.get("split_refusal"):
        return "The ranges cannot run yet; the warning above says why."
    ranges = state.get("ranges") or state.get("ranges_typed") or ()
    if preview is None:
        return "Each range is one map, the same ranges for every condition."
    shares = range_shares(preview, ranges)
    parts = [f"{range_title(method, lo, hi)} {100 * s:.0f} %" for (lo, hi), s in zip(ranges, shares)]
    outside = max(0.0, 1.0 - sum(shares))
    return (f"Of {preview['condition']}'s {preview['n']:,} {what}: " + ", ".join(parts)
            + (f"; {100 * outside:.0f} % in no range, not drawn" if outside >= 0.005 else "") + ".")


def _ranges_key(ranges) -> tuple | None:
    """Ranges as a hashable key part (`settings_key`)."""
    return None if not ranges else tuple((float(lo), float(hi)) for lo, hi in ranges)
#: The SQD map's time lags, in frames (the SQD section's `comparison.SQD_LAG_CHOICES`): jumps that many frames apart
#: are fitted and drawn (asked 16 Oct 2026, "Lags also in sqd maps as option"). 1, the default, is every map before;
#: "1-4" (17 Oct 2026) is v2's "1-4 together": one fit to the four lags, every lag's jumps drawn.
MAP_LAGS = (1, 2, 3, 4, "1-4")


def map_lag(lag):
    """A lag setting as the map pipeline takes it: a number of frames, or "1-4"."""
    return "1-4" if str(lag) == "1-4" else int(lag or 1)
N_POP_METHODS = ("sqd", "appd", "hmm")

#: Display title -> the payload key it draws, for the Visibility tool.
PANEL_KEYS_BY_METHOD = {
    "sqd": {"All jumps": "jumps", "Short jumps": "short", "Long jumps": "long"},
    "appd": {"All tracks": "jumps", "Slow (below the crossing)": "slow", "Fast (above the crossing)": "fast"},
    "hmm": {"All jumps": "jumps", "Slow (HMM state)": "slow", "Fast (HMM state)": "fast"},
    # v2's own three sets and words (`confTracks.m`), in its checkbox order.
    "sla": {"Free tracks": "free", "Mixed-behaviour tracks": "mixed",
            "Dwelling segments": "dwelling"},
    # v2's SDA maps (`spatialsda.SDA_TITLES`), in v2's own order.
    "sda": {"Localisations": "location", "Speed": "speed",
            "Diffusion (apparent D)": "diffusion", "Acceleration": "acceleration"},
}
_APPD3_KEYS = {"All tracks": "jumps", "Slow (below the first crossing)": "slow",
               "Middle (between the crossings)": "middle",
               "Fast (above the second crossing)": "fast"}
_SQD3_KEYS = {"All jumps": "jumps", "Short jumps": "short", "Middle jumps": "middle", "Long jumps": "long"}
#: One SQD population: every jump in one map (`MAP_N_POPS_BY_METHOD`).
_SQD1_KEYS = {"All jumps": "jumps"}
#: The HMM maps at three states, and the map of every jump before them (3 Nov 2026)
_HMM3_KEYS = {"All jumps": "jumps", "Slow (HMM state)": "slow", "Middle (HMM state)": "middle",
              "Fast (HMM state)": "fast"}
#: The apparent-D maps' map of every track: its localisations, not its jumps (`spatialappd`'s ``all_tracks``)
_APPD_ALL = {"All tracks": "jumps"}
#: The map of every jump: one SQD population's own, and the first column of every SQD split.
ALL_JUMPS_KIND = "jumps"


def kinds_for(method: str, n_pop: int = 2, ranges=None) -> tuple:
    """The panel kinds a run draws: one per population, slowest first -- or one per range of a manual split
    (``ranges``, `SPLITS`), lowest first -- and for SQD the map of every jump before them (`ALL_JUMPS_KIND`)."""
    if ranges and method in RANGE_UNITS:
        return (ALL_JUMPS_KIND,) + RANGE_KINDS[:len(ranges)]
    if int(n_pop or 2) == 1 and method in ("sqd", "hmm"):
        return tuple(_SQD1_KEYS.values())
    if int(n_pop or 2) == 3 and method in N_POP_METHODS:
        return tuple(panel_keys_for(method, 3).values())
    return KINDS_BY_METHOD[method]


def population_kinds(method: str, n_pop: int = 2, ranges=None) -> tuple:
    """`kinds_for` without the map of every jump: the populations (or ranges) a run splits into, slowest first. One
    SQD population is its own map of every jump, so it stays."""
    kinds = kinds_for(method, n_pop, ranges)
    return tuple(k for k in kinds if k != ALL_JUMPS_KIND) or kinds


def panel_keys_for(method: str, n_pop: int = 2, ranges=None) -> dict:
    """Display title -> panel kind, for the Visibility tool and the column heads; a manual split's titles are its
    ranges (`range_title`)."""
    if ranges and method in RANGE_UNITS:
        return {**(dict(_SQD1_KEYS) if method == "sqd" else dict(_APPD_ALL)),
                **{range_title(method, lo, hi): kind for kind, (lo, hi) in zip(RANGE_KINDS, ranges)}}
    if int(n_pop or 2) == 1 and method in ("sqd", "hmm"):
        return dict(_SQD1_KEYS)
    if int(n_pop or 2) == 3 and method == "appd":
        return dict(_APPD3_KEYS)
    if int(n_pop or 2) == 3 and method == "hmm":
        return dict(_HMM3_KEYS)
    if int(n_pop or 2) == 3 and method == "sqd":
        return dict(_SQD3_KEYS)
    return PANEL_KEYS_BY_METHOD[method]


def panel_titles_for(method: str, n_pop: int = 2, ranges=None) -> tuple:
    keys = panel_keys_for(method, n_pop, ranges)
    return tuple(next(title for title, k in keys.items() if k == kind)
                 for kind in kinds_for(method, n_pop, ranges))

#: The same titles in `KINDS_BY_METHOD[method]` order, DERIVED rather than written out a second time.
PANEL_TITLES_BY_METHOD: dict[str, tuple] = {
    method: tuple(next(title for title, k in PANEL_KEYS_BY_METHOD[method].items() if k == kind)
                  for kind in kinds)
    for method, kinds in KINDS_BY_METHOD.items()
}

#: The Run dialog's setting names (also the caption's row names: somebody who just set `Jump drawn at` should
#: read `Jump drawn at` back).
SETTING_NAMES = {
    "show": "Show",
    "method": "Computational method",
    "jump": "Jump drawn at",
    "assignment": "Assignment",
    "fold": "Fold",
    "frame": "Cell frame",
    "cells": "Cells",
    "axis": "Long axis",
    "bin": "Bin size",
    "n_min": "Minimum jumps per bin",
    "weighting": "Weighting",
    "cutoff": "Maximum localisations per track",
    "neff": "Effective sample size",
    "populations": "Populations",
    "split": "Split",
    "lag": "Time lag",
    # The dialog's page holding `n_min` and `neff` together: both blank a bin, for different
    # reasons. A PAGE name only -- the caption keeps the two settings' own rows.
    "masking": "Bin masking",
    # Confinement (SLA) only: the moving circle's radius and how long a molecule must stay.
    "radius": "Confinement radius",
    "dwell": "Minimum dwell",
    # SDA only: whether the value maps' kernel widens where steps are few.
    "adaptive": "Adaptive smoothing",
}
#: Apparent diffusion and SLA bin per-LOCALISATION rows, not per-step jumps, so only the
#: row name changes.
_SETTING_NAMES_APPD = {"n_min": "Minimum localisations per bin"}
#: SDA's four maps count three different things per bin (localisations, steps, step
#: pairs), so its minimum names none of them; each map applies it to its own count.
_SETTING_NAMES_SDA = {"n_min": "Minimum per bin"}


def n_min_unit(method: str) -> str:
    """The word beside the Minimum-per-bin field: what that method's maps count."""
    if method in PER_LOCALISATION_METHODS:
        return "localisations"
    return "per map" if method == "sda" else "jumps"


def setting_name(key: str, method: str = "sqd") -> str:
    if method in PER_LOCALISATION_METHODS and key in _SETTING_NAMES_APPD:
        return _SETTING_NAMES_APPD[key]
    if method == "sda" and key in _SETTING_NAMES_SDA:
        return _SETTING_NAMES_SDA[key]
    return SETTING_NAMES[key]


# --------------------------------------------------------------------------- #
# Compute (one condition, from the project already loaded) and the pooling and caption rules
# --------------------------------------------------------------------------- #


#: Which cells a map draws, by length. THIRDS of the cells measured with the frame
#: (`measure_cell_frame`), so each class holds about as many cells as the others.
CELL_CLASSES = ("all", "short", "mid", "long")
CELL_CLASS_LABELS = {"all": "All cells", "short": "Short cells", "mid": "Mid cells",
                     "long": "Long cells"}
LONG_AXIS_LABELS = {"normalised": "Normalised", "pole_anchored": "Pole-anchored"}


def _class_of(lengths, cuts) -> "np.ndarray":
    """Each length's class: short below the first cut, long at or above the second."""
    import numpy as np

    lengths = np.asarray(lengths, dtype=float)
    return np.where(lengths < cuts[0], "short",
                    np.where(lengths >= cuts[1], "long", "mid"))


def measure_cell_frame(project, condition) -> dict | None:
    """The median capsule of THIS condition's own segmented cells, in µm.

    `condition` may be ONE name or a SEQUENCE of them. A sequence pools every
    named condition's cells and takes one median over the lot, which is what
    the Spatial Map SECTION needs: it draws every condition in one grid under
    one colour bar per population, and `settings_key` (rightly) refuses to
    pool panels computed at different settings -- the cell frame among them.
    Measuring each condition separately would therefore give every row its own
    scale and take the grid's whole point away. One frame over all of them is
    also what `compare_plots.draw_spatial` does, and the spread that travels
    with the median is then the honest account of how much that costs.

    Every cell is mapped onto ONE capsule (`d_norm` is already scaled by each
    cell's own local half-width, so nothing is truncated), and that capsule
    has to be some real cell's size. Measuring it beats naming it: this
    project's own record of the legacy location map says an identical number
    across every group is the tell for a fallback, and a default cell width
    printed beside real numbers reads as a measurement.

    CHEAP ENOUGH FOR THE GUI THREAD, and deliberately so -- it runs when the
    Run dialog opens, before anything has been computed. It reads the
    CANONICAL table (a mask over an in-memory frame) and the outlines already
    in hand; it does NOT call `project.projected`, which would cost the
    projection this dialog exists to configure.

    Returns `None` when the frame cannot be measured -- no outlines, or none
    with ribs. `None` is a real answer ("this cannot be measured"), and the
    caller shows the deposit's frame LABELLED as not measured rather than
    silently passing it off as this project's.
    """
    import numpy as np

    from .cellcoords import CellCoordError, cell_geometry

    params = getattr(project, "params", None)
    meshes = getattr(project, "meshes", None)
    if params is None or not meshes:
        return None
    names = [condition] if isinstance(condition, str) else list(condition)
    try:
        um_per_pixel = float(params.um_per_pixel)
        tables = [project.condition(name) for name in names]
    except (AttributeError, KeyError):
        # params without a pixel size, or no such table: "cannot measure it". Any other
        # error is a bug and is raised.
        return None

    flat = (meshes.values() if isinstance(meshes, dict) else [meshes])
    lookup = {f"{m.movie}:{m.cell_id}": m for group in flat for m in group}
    # A cell key is `movie:cell`, and a movie belongs to exactly one
    # condition, so pooling several conditions cannot double-count a cell.
    keys = sorted({str(k) for table in tables
                   for k in table["cell"].dropna().unique()})

    lengths, widths = [], []
    for key in keys:
        mesh = lookup.get(key)
        if mesh is None or not mesh.has_centerline:
            continue
        try:
            g = cell_geometry(mesh, um_per_pixel=um_per_pixel)
        except (CellCoordError, ValueError):
            continue
        if np.isfinite(g.length_um) and np.isfinite(g.max_width_um):
            lengths.append(g.length_um)
            widths.append(g.max_width_um)

    if not lengths:
        return None
    lengths, widths = np.asarray(lengths), np.asarray(widths)
    # THE LENGTH CLASSES, from the same cells and the same pool as the frame: thirds,
    # so each class holds about as many cells. One set of cuts for every condition in
    # the run, as one frame is -- a "long" panel means the same lengths in every row.
    cuts = tuple(float(c) for c in np.percentile(lengths, [100.0 / 3.0, 200.0 / 3.0]))
    which = _class_of(lengths, cuts)
    classes = {}
    for name in CELL_CLASSES[1:]:
        mask = which == name
        classes[name] = None if not mask.any() else {
            "cell_length_um": float(np.median(lengths[mask])),
            "cell_width_um": float(np.median(widths[mask])),
            "n_cells": int(mask.sum()),
            "length_range_um": (float(lengths[mask].min()), float(lengths[mask].max()))}
    return {"cell_length_um": float(np.median(lengths)),
            "cell_width_um": float(np.median(widths)),
            "n_cells": len(lengths),
            # The SPREAD travels with the median, because one capsule stands
            # in for all of them and how well it does is the reader's
            # business: these cells run 2.2-4.6 um in the deposited data.
            "length_range_um": (float(np.min(lengths)), float(np.max(lengths))),
            "width_range_um": (float(np.min(widths)), float(np.max(widths))),
            "length_cuts_um": cuts,
            "classes": classes}


def apply_cell_class(projected, cell_class: str, cuts_um):
    """`(kept_table, stats)`: only the rows of cells in `cell_class`, by each cell's
    own length (the projection's `.attrs["geometry"]`, the length the frame was
    measured from). `"all"` returns the SAME table object, untouched.

    A DRAWING choice, never a fitting one: every compute here fits on the
    condition's tables before this is applied, so a short-cell map and a long-cell
    map are read under the same populations and differ only in where the jumps fell.
    """
    import numpy as np

    if cell_class not in CELL_CLASSES:
        raise ValueError(f"cell_class must be one of {CELL_CLASSES}, got {cell_class!r}")
    total = int(projected["cell"].nunique()) if "cell" in projected.columns else 0
    if cell_class == "all":
        return projected, {"cells_kept": total, "cells_total": total}
    if cuts_um is None or len(cuts_um) != 2:
        raise ValueError("a length class needs the two cuts measured over the cells "
                         "(measure_cell_frame's 'length_cuts_um')")
    geometry = projected.attrs.get("geometry")
    if not geometry:
        raise ValueError("the projected table carries no cell geometry, so its cells "
                         "have no length to class them by")
    keys = projected["cell"].astype(str)
    length = keys.map({k: g.length_um for k, g in geometry.items()}).to_numpy(float)
    mask = np.isfinite(length) & (_class_of(length, cuts_um) == cell_class)
    if not mask.any():
        # The cuts are measured over every condition in the run, so one condition can
        # hold no cell of a class at all. Said by name, never drawn as an empty cell.
        raise ValueError(f"no {cell_class} cells here "
                         f"({cell_class_range_text(cell_class, cuts_um)})")
    kept = projected.loc[mask].copy()
    kept.attrs.update(projected.attrs)
    return kept, {"cells_kept": int(kept["cell"].nunique()), "cells_total": total}


def cell_class_range_text(cell_class: str, cuts_um) -> str:
    """`"under 2.86 µm"`, `"2.86–3.41 µm"`, `"3.41 µm and over"`; "" for all cells."""
    if cell_class == "all" or not cuts_um:
        return ""
    lo, hi = cuts_um
    return {"short": f"under {lo:.2f} µm", "mid": f"{lo:.2f}\u2013{hi:.2f} µm",
            "long": f"{hi:.2f} µm and over"}[cell_class]


def cell_class_text(payload: dict) -> str:
    """The Cells caption value: which cells, their lengths, and how many of them."""
    cell_class = payload.get("cell_class", "all")
    stats = payload.get("class_stats") or {}
    if cell_class == "all":
        return "ALL" + (f" · {stats['cells_total']} cells" if stats.get("cells_total") else "")
    bits = [cell_class.upper(), cell_class_range_text(cell_class, payload.get("class_cuts_um"))]
    if stats.get("cells_total"):
        bits.append(f"{stats['cells_kept']} of {stats['cells_total']} cells")
    bits.append("fit on every cell")
    return " · ".join(b for b in bits if b)


def long_axis_text(payload: dict) -> str:
    """The Long axis caption value."""
    if payload.get("long_axis", "normalised") == "pole_anchored":
        return "POLE-ANCHORED · caps at their own size, the middle stretched"
    return "NORMALISED · every cell stretched to the drawn length"


def robust_limits(grid, measured) -> tuple[float, float]:
    """The spec's sparse rule B: clip to the 2nd and 98th percentiles of the
    UNMASKED bins, so one low-count spike cannot define the top of the scale.

    Both ends, not just the top. The previous version took the 99th percentile
    for the ceiling and left the floor at zero, which is `draw_maps`'
    convention rather than the spec's.
    """
    import numpy as np

    if not measured.any():
        return (0.0, 0.0)
    values = np.asarray(grid)[measured]
    return (float(np.percentile(values, ROBUST_LO_PCT)),
            float(np.percentile(values, ROBUST_HI_PCT)))


def bandwidth_um(sigma_nm, bin_long_um: float, bin_short_um: float) -> float:
    """`h = max(sigma, bin_halfwidth)`, in micrometres.

    `render._resolve_bandwidth`'s formula, restated so this view uses the same
    rule as the module the spec names as its own implementation. The spec's
    governing principle wants a PHYSICAL bandwidth tied to the localisation
    precision; the previous fixed 1.0 bins made h follow the bin instead, so
    dragging the bin slider silently moved the real smoothing scale from 30 nm
    to 400 nm.

    **A missing sigma is not guessed.** `render` refuses outright rather than
    default one. Refusing would leave the map undrawable on a project whose
    params never loaded, so this falls back to the OTHER term of the max --
    half a bin, which is not a claim about sigma at all -- and
    `bandwidth_text` prints "sigma not recorded" beside it.
    """
    half_bin = max(bin_long_um, bin_short_um) / 2.0
    if sigma_nm is None:
        return half_bin
    return max(float(sigma_nm) / 1000.0, half_bin)


def sigma_stamp(project) -> dict:
    """The localisation sigma and WHERE IT CAME FROM, as a pair.

    `AcquisitionParams.sigma_stamp` is this project's one source of truth for
    that pair, and its own docstring is the reason both travel together: "a
    sigma without a provenance state is not a sigma". 17.9 nm read off the
    tracker and 98.8 nm derived from the PSF are both called sigma and are
    not the same claim.

    Returns `None` for both rather than a number when the project cannot
    supply one -- an unexplained default printed beside real numbers reads
    as a measurement.
    """
    params = getattr(project, "params", None)
    try:
        stamp = params.sigma_stamp()
    except AttributeError:
        # no params at all: "cannot state it". Any other error is a bug and is raised.
        return {"sigma_nm": None, "sigma_provenance": None}
    return {"sigma_nm": float(stamp["sigma_nm"]),
            "sigma_provenance": str(stamp["sigma_provenance"])}


def finish_spatial_panels(grids: dict, weights: dict, *, n_min: int,
                           inside, h_um: float, bin_long_um: float,
                           bin_short_um: float, step_grids: dict | None = None,
                           step_weights: dict | None = None,
                           neff: dict | None = None, neff_min: float | None = None,
                           neff_unit: str = "tracks") -> dict:
    """The `n_min` mask and the count-weighted, capsule-aware smoothing --
    the GUI-only post-processing neither `spatialsqd` nor `spatialappd` does
    for itself (see each module's own docstring). Shared by both
    computational methods: this step operates on plain `{kind: grid}` /
    `{kind: weight}` dicts and knows nothing about how those grids were
    labelled, which is what makes it reusable rather than duplicated.

    `step_grids`/`step_weights` are the PLAIN jump (or localisation) counts.
    When the map is weighted by track, `grids`/`weights` hold the drawn,
    track-weighted values and these hold the evidence: the `n_min` mask, the
    panel's `n_jumps` and the sparse rule read them, because "too few jumps in
    this bin" is a statement about data, not about how the jumps are weighted.
    Left out (every by-step caller), they are the drawn grids themselves.

    `neff` is `{kind: {"tracks": grid, "cells": grid}}`, each bin's effective number
    of contributors (handover §4a.2). With `neff_min` set, a bin below it in
    `neff_unit` is masked exactly like a bin below `n_min` -- undefined, grey, out of
    the kernel and out of the colour limits -- and the panel counts the two reasons
    apart (`n_masked_count`, `n_masked_neff`; a bin failing both is counted once, as
    too few jumps). `neff_min=None` is OFF and masks nothing.
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter

    step_grids = grids if step_grids is None else step_grids
    step_weights = weights if step_weights is None else step_weights

    panels: dict[str, dict] = {}
    for kind, raw_grid in grids.items():
        raw = np.asarray(raw_grid, dtype=float)

        # `n_min`, applied exactly as `render.py` applies it: below the
        # threshold a bin is UNDEFINED (not zero), it is taken out of the
        # kernel, and what remains is count-weighted so a hole does not drag
        # its neighbours down. Dividing one Gaussian by the other IS that
        # renormalisation.
        #
        # **CORRECTION, AND THE OLD CLAIM IS LEFT NAMED SO IT CANNOT BE
        # REQUOTED.** This used to say that at `n_min = 0` the arithmetic
        # "reduces EXACTLY to `draw_maps`' own gaussian_filter -- the
        # default path is the deposited picture, not an approximation of
        # it". That was true when it was written and is NOT true any more:
        # enforcing the cell boundary below (the render spec's decision 3,
        # which `draw_maps` does not implement) deliberately parts company
        # with it. MEASURED on one condition at the default bin: 56 of 280
        # in-cell bins differ by more than 1 %, entirely in the band whose
        # kernel reaches past the capsule; the interior is untouched, and
        # the largest difference is at the wall, where `draw_maps` is the
        # one that is wrong (it reads a uniform density as 47 % low there).
        # So the GUI map is spec-compliant and the deposited panels are not,
        # and a GUI panel put beside a manuscript one will differ at the
        # rim. That is a decision, recorded here, not a drift.
        # TWO DIFFERENT MASKS, because they answer two different questions.
        #
        # `sparse` is the USER's mask: bins the run was told to treat as
        # undefined, drawn grey, counted in the caption.
        #
        # `usable` is the KERNEL's: everything the smoothing may see. It
        # additionally drops every bin OUTSIDE THE CAPSULE -- which is the
        # render spec's decision 3, "the estimate must respect the cell
        # boundary ... so density neither leaks out nor is dragged to zero
        # against the wall".
        #
        # `gaussian_filter` reflects at the ARRAY edge, i.e. the bounding
        # rectangle, not at the capsule; the corners the capsule does not
        # reach hold a legitimate zero, and feeding those zeros to the
        # kernel pulls the bins beside them down. MEASURED on a perfectly
        # uniform in-cell density at the default bin: the worst wall bin
        # came back 47 % low and 11 % of in-cell bins more than 5 % low.
        # Excluding them and renormalising returns it flat to the digit.
        #
        # `render.py` never needed this and its own spec says the clip was
        # RETIRED -- correctly, because in its frame `d_norm` is already
        # divided by each cell's local half-width, so the cell IS the
        # rectangle and there is no outside. `spatialsqd`/`spatialappd`
        # re-expand by that half-width onto one median capsule (which
        # truncates nothing, see `local_half_width`), so here there is an
        # outside again, and the boundary rule that frame did not need
        # applies to this one.
        evidence = np.asarray(step_grids[kind], dtype=float)
        sparse = (evidence < n_min) if n_min > 0 else np.zeros(raw.shape, dtype=bool)
        # THE n_eff GUARD: too few CONTRIBUTORS, read against the same `inside`, so
        # one rule covers every bin of every panel. Shown in the cutoff's unit, or
        # by track while the cutoff is off.
        shown_unit = neff_unit if neff_min is not None else "tracks"
        contributors = (np.asarray(neff[kind][shown_unit], dtype=float)
                        if neff is not None else None)
        if neff_min is not None:
            if contributors is None:
                raise ValueError("an n_eff cutoff needs the per-bin n_eff grids")
            few = contributors < float(neff_min)
        else:
            few = np.zeros(raw.shape, dtype=bool)
        masked_count = inside & sparse
        masked_neff = inside & few & ~sparse
        masked = masked_count | masked_neff
        usable = inside & ~(sparse | few)
        # THE KERNEL IS ISOTROPIC IN MICROMETRES, and therefore anisotropic
        # in bins -- the spec's own [impl] note, and what `render` does
        # (`sigma_bins = (h / bin_um[1], h / bin_um[0])`). The two bin edges
        # differ here only by the even-count rounding, but stating it per
        # axis costs nothing and cannot drift.
        sigma_bins = (h_um / bin_short_um, h_um / bin_long_um)
        num = gaussian_filter(np.where(usable, raw, 0.0), sigma=sigma_bins)
        den = gaussian_filter(usable.astype(float), sigma=sigma_bins)
        smooth = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-6)

        # `scale="shared"` turns each map into a probability DENSITY before
        # the limits are taken, which is `draw_maps`' own convention and the
        # only way one panel and another can share one bar: they carry
        # different total weight, so a shared RAW scale would read as "there
        # are fewer of this population", which is a fact about occupancy,
        # not about where they happen.
        # The density normalises over the CELL, not over the bounding box --
        # the out-of-capsule bins carry a smoothed value now (they are
        # clipped away when drawn) and must not enter the total.
        total = smooth[usable].sum()
        density = smooth / total if total > 0 else smooth
        measured = usable
        n_jumps = float(step_weights[kind])
        masked_frac = (masked.sum() / inside.sum()) if inside.any() else 1.0
        # SPARSE RULE C. A panel too thin to map is greyed and labelled, not
        # rendered as confetti -- the spatial analogue of reporting a bound
        # instead of a fake point. The two thresholds are the ones the
        # existing spatial view already uses; see `PANEL_N_MIN`.
        sparse = (not measured.any()
                  or n_jumps < PANEL_N_MIN
                  or masked_frac > PANEL_MAX_MASKED_FRAC)
        panels[kind] = {
            "counts": smooth.tolist(),
            "density": density.tolist(),
            "masked": masked.tolist(),
            "n_jumps": n_jumps,
            # The drawn total: equal to `n_jumps` by step; by track, the
            # (posterior-weighted) number of tracks this map counts.
            "n_weight": float(weights[kind]),
            "n_masked": int(masked.sum()),
            "n_masked_count": int(masked_count.sum()),
            "n_masked_neff": int(masked_neff.sum()),
            # the n_eff display: every in-cell bin's contributors (0 = empty);
            # None outside the cell, never NaN (NaN != NaN breaks every comparison
            # of two payloads, and does not survive a JSON round trip)
            "neff": ([[float(v) if ok else None for v, ok in zip(row, irow)]
                      for row, irow in zip(contributors.tolist(), inside.tolist())]
                     if contributors is not None else None),
            "neff_unit": shown_unit,
            "masked_frac": float(masked_frac),
            "sparse": bool(sparse),
            "limits_counts": robust_limits(smooth, measured),
            "limits_density": robust_limits(density, measured),
        }
    return panels


def _impulse_matrix(n: int, sigma_bins: float):
    """``M[b, i]``: what `scipy.ndimage.gaussian_filter` puts at bin ``b`` along one axis from
    one count at bin ``i`` -- its own truncation and reflecting edge, so a sum through ``M`` is
    that filter exactly (`spot_contributors` reads the kernel the same way)."""
    import numpy as np
    from scipy.ndimage import gaussian_filter1d

    return gaussian_filter1d(np.eye(int(n)), float(sigma_bins), axis=0)


def _value_blocks(grid, factor: int):
    """A value map's per-VALUE-bin grid from its spread fine grid (`spatialsda._spread`),
    refusing a grid that is not whole value bins: the effective count weighs each value
    bin's steps once, and a grid that varies inside a value bin has no such count."""
    import numpy as np

    grid = np.asarray(grid, dtype=float)
    f = int(factor)
    if f < 1 or grid.shape[0] % f or grid.shape[1] % f:
        raise ValueError(f"a {grid.shape[0]} x {grid.shape[1]} grid is not whole "
                         f"{f} x {f} value bins")
    blocks = grid[::f, ::f]
    if f > 1 and not np.array_equal(np.repeat(np.repeat(blocks, f, axis=0), f, axis=1), grid):
        raise ValueError(f"the value map's counts are not whole {f} x {f} value bins: "
                         f"each value bin's count must be written onto all its small bins")
    return blocks


def effective_steps(block_counts, kernel_in, sigma_bins, factor: int = 1):
    """Each fine bin's EFFECTIVE number of steps under the Gaussian of ``sigma_bins``
    (short axis, long axis; in fine bins): ``(sum_i k_i)^2 / sum_i k_i^2`` over every step
    the kernel centred on that bin weighs -- the participation ratio n_eff takes over
    tracks, here over steps. It is the number of steps whose plain mean is as precise as
    the kernel's weighted one: a point kernel gives the bin's own count, a wider one more.

    ``block_counts`` are the plain step counts per VALUE bin (``factor`` x ``factor`` fine
    bins each), ``kernel_in`` the fine bins the kernel may read. A step in value bin ``B``
    weighs, at fine bin ``b``, the kernel summed over ``B``'s readable fine bins -- the very
    weight `finish_value_panels`' ``G(sum) / G(count)`` over the spread grids gives it, so
    this is the effective count of the mean that is drawn, not of some other one.

    Products of small matrices (the kernel is separable; the mask is applied per value
    bin), never a loop over bins: ``factor**2`` products for the sum and ``factor**4`` for
    the sum of squares."""
    import numpy as np

    f = int(factor)
    counts = np.asarray(block_counts, dtype=float)
    k_in = np.asarray(kernel_in, dtype=bool)
    n_short, n_long = k_in.shape
    my = _impulse_matrix(n_short, sigma_bins[0])
    mx = _impulse_matrix(n_long, sigma_bins[1])
    first = np.zeros((n_short, n_long))
    second = np.zeros((n_short, n_long))
    offsets = [(a, c) for a in range(f) for c in range(f)]
    for a, c in offsets:
        m_ac = k_in[a::f, c::f]
        first += my[:, a::f] @ (counts * m_ac) @ mx[:, c::f].T
        for a2, c2 in offsets:
            both = counts * (m_ac & k_in[a2::f, c2::f])
            second += ((my[:, a::f] * my[:, a2::f]) @ both
                       @ (mx[:, c::f] * mx[:, c2::f]).T)
    return np.divide(first ** 2, second, out=np.zeros_like(first), where=second > 0)


def adaptive_smoothing(total, drawn, evidence, kernel_in, *, h0_um: float,
                       bin_long_um: float, bin_short_um: float, floor: float,
                       factor: int = 1, widths=None) -> dict:
    """The value maps' ADAPTIVE kernel (`spatialsda` divergence 9): per fine bin, the first
    width of ``widths`` (in h0; `spatialsda.ADAPTIVE_WIDTHS`) at which the kernel collects
    ``floor`` steps' worth (`effective_steps`, from the plain counts ``evidence``), and the
    count-weighted mean ``G_h(total) / G_h(drawn)`` over ``kernel_in`` at that width.

    Returns ``value`` (each bin's mean at its width; at the widest where none reaches),
    ``index`` (the width's position in ``widths``; -1 where even the widest falls short --
    such a bin is grey, never given a value), ``h_um`` (the width used, the widest where
    none reaches) and ``steps`` (the effective count at that width). At ``widths[0] == 1``
    over bins that all reach, ``value`` is the fixed kernel's bit for bit: the same filter
    on the same arrays.

    A VALUE BIN HOLDING NO STEP OF ITS OWN is -1 too, whatever its kernel collects: its
    mean would be its neighbours' alone, the same borrowed value a bin outside the cell
    is refused (`finish_value_panels`). A widened kernel fills a thin bin; it never paints
    an empty one."""
    import numpy as np
    from scipy.ndimage import gaussian_filter

    if widths is None:
        from .spatialsda import ADAPTIVE_WIDTHS as widths
    widths = tuple(float(w) for w in widths)
    if not widths or any(b <= a for a, b in zip(widths, widths[1:])) or widths[0] <= 0:
        raise ValueError(f"the widths must be positive and rising, got {widths}")
    k_in = np.asarray(kernel_in, dtype=bool)
    blocks = _value_blocks(evidence, factor)
    total = np.asarray(total, dtype=float)
    drawn = np.asarray(drawn, dtype=float)
    shape = k_in.shape
    value = np.zeros(shape)
    steps = np.zeros(shape)
    h_um = np.full(shape, widths[-1] * h0_um)
    index = np.full(shape, -1, dtype=int)
    open_ = np.ones(shape, dtype=bool)
    # A rounding tolerance only: a field of exactly `floor` steps per bin reads
    # 19.999999999 through the filter and must not widen for that.
    target = float(floor) * (1.0 - 1e-9)
    for k, w in enumerate(widths):
        h = w * h0_um
        sigma_bins = (h / bin_short_um, h / bin_long_um)
        num = gaussian_filter(np.where(k_in, total, 0.0), sigma=sigma_bins)
        den = gaussian_filter(np.where(k_in, drawn, 0.0), sigma=sigma_bins)
        mean = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)
        worth = effective_steps(blocks, k_in, sigma_bins, factor)
        take = open_ & (worth >= target)
        last = k == len(widths) - 1
        fill = take | (open_ & last)
        value[fill], steps[fill] = mean[fill], worth[fill]
        h_um[fill] = h
        index[take] = k
        open_ &= ~take
    index[np.asarray(evidence, dtype=float) <= 0] = -1
    return {"value": value, "index": index, "h_um": h_um, "steps": steps,
            "widths": widths}


def finish_value_panels(sums: dict, grids: dict, *, step_grids: dict, weights: dict,
                        step_weights: dict, n_min: int, value_min: int, inside,
                        h_um: float, bin_long_um: float, bin_short_um: float,
                        units: dict, count_units: dict, neff: dict | None = None,
                        neff_min: float | None = None, neff_unit: str = "tracks",
                        adaptive: bool = False, value_factor: int = 1) -> dict:
    """`finish_spatial_panels` for a MEAN-VALUE map (SDA's speed, D and acceleration):
    the same masking, the same kernel, the same sparse rule -- but never a density.

    ``sums`` are each map's drawn sums of values per bin, ``grids`` its drawn weights (the
    divisor), ``step_grids`` the plain counts of steps or step pairs (the evidence), all
    already FOLDED BY POOLING where the map is mirrored (`spatialsda.value_grids`).

    A bin is undefined (grey) when it holds fewer than ``max(n_min, value_min)`` steps --
    a mean of a handful is noise, so a value map has its own floor (`spatialsda.
    VALUE_N_MIN`) under whatever the Run asks for -- or, with the n_eff cutoff on, too few
    contributors. Masked bins are out of the kernel. The map is then COUNT-WEIGHTED
    smoothing of the per-bin mean: ``G(sum) / G(count)`` over the usable bins, at the same
    bandwidth h the density maps use (`render`'s value maps do the same), so a bin with
    more support pulls its neighbours more and a hole drags nothing. The colour limits are
    the robust 2-98 % of the usable bins, in the map's own unit (``limits_value``).

    ``adaptive=True`` (`spatialsda` divergence 9, `adaptive_smoothing`) moves the floor
    from a bin's OWN count to what its kernel collects: each bin keeps ``h_um`` where the
    kernel there is already worth ``max(n_min, value_min)`` steps, takes the first wider
    width of `spatialsda.ADAPTIVE_WIDTHS` that is where it is not, and is grey when even
    `spatialsda.ADAPTIVE_CAP` x ``h_um`` falls short. Every in-cell step then enters the
    kernel (only the n_eff cutoff takes a bin out of it): the floor is applied to the mean,
    not to the bins it is made from. ``value_factor`` is the value bins' size in fine bins
    (`spatialsda.value_bin_factor`), which the effective count needs to weigh each value
    bin's steps once. The panel records each bin's width (``h_nm``) and effective count
    (``steps_eff``), and ``kernel_in``, the bins the kernel read, either way."""
    import numpy as np
    from scipy.ndimage import gaussian_filter

    floor = max(int(n_min), int(value_min), 1)
    panels: dict[str, dict] = {}
    for kind, raw_sum in sums.items():
        total = np.asarray(raw_sum, dtype=float)
        drawn = np.asarray(grids[kind], dtype=float)
        evidence = np.asarray(step_grids[kind], dtype=float)
        shown_unit = neff_unit if neff_min is not None else "tracks"
        contributors = (np.asarray(neff[kind][shown_unit], dtype=float)
                        if neff is not None else None)
        if neff_min is not None:
            if contributors is None:
                raise ValueError("an n_eff cutoff needs the per-bin n_eff grids")
            few = contributors < float(neff_min)
        else:
            few = np.zeros(total.shape, dtype=bool)
        extra: dict = {"adaptive": bool(adaptive), "h0_nm": float(h_um * 1000.0)}
        if adaptive:
            kernel_in = inside & ~few
            chosen = adaptive_smoothing(
                total, drawn, evidence, kernel_in, h0_um=h_um, bin_long_um=bin_long_um,
                bin_short_um=bin_short_um, floor=floor, factor=value_factor)
            sparse_bins = chosen["index"] < 0
            value = chosen["value"]
        else:
            sparse_bins = evidence < floor
            kernel_in = inside & ~(sparse_bins | few)
            sigma_bins = (h_um / bin_short_um, h_um / bin_long_um)
            num = gaussian_filter(np.where(kernel_in, total, 0.0), sigma=sigma_bins)
            den = gaussian_filter(np.where(kernel_in, drawn, 0.0), sigma=sigma_bins)
            value = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)
        masked_count = inside & sparse_bins
        masked_neff = inside & few & ~sparse_bins
        masked = masked_count | masked_neff
        usable = inside & ~masked
        if adaptive:
            widths = chosen["widths"]
            h_nm = chosen["h_um"] * 1000.0
            used = h_nm[usable]
            extra.update({
                "h_cap_nm": float(widths[-1] * h_um * 1000.0),
                # each in-cell bin's kernel: the width it was drawn at, or -- grey for too
                # few steps -- the widest it was allowed; None outside the cell
                # (unrounded: a clicked spot re-uses it as its kernel, `spot_contributors`)
                "h_nm": [[float(v) if ok else None for v, ok in zip(row, irow)]
                         for row, irow in zip(h_nm.tolist(), inside.tolist())],
                "steps_eff": [[round(float(v), 2) if ok else None
                               for v, ok in zip(row, irow)]
                              for row, irow in zip(chosen["steps"].tolist(),
                                                   inside.tolist())],
                "h_range_nm": ([float(used.min()), float(used.max())] if used.size
                               else None),
                "n_widened": int((usable & (chosen["index"] > 0)).sum()),
                "n_drawn": int(usable.sum()),
            })
        extra["kernel_in"] = np.asarray(kernel_in, dtype=bool).tolist()
        n_rows = float(step_weights[kind])
        masked_frac = (masked.sum() / inside.sum()) if inside.any() else 1.0
        sparse = (not usable.any() or n_rows < PANEL_N_MIN
                  or masked_frac > PANEL_MAX_MASKED_FRAC)
        panels[kind] = {
            "quantity": "value",
            "unit": units[kind],
            "count_unit": count_units[kind],
            "value": value.tolist(),
            # A bin whose centre lies outside the capsule is never averaged (it is out of
            # the kernel), so it is drawn undefined too: the sliver of it the capsule shows
            # would otherwise carry a mean borrowed from its neighbours, even masked ones.
            # Counted nowhere: `n_masked` and the sparse rule read the cell's own bins.
            "masked": (masked | ~inside).tolist(),
            "n_jumps": n_rows,
            "n_weight": float(weights[kind]),
            "n_masked": int(masked.sum()),
            "n_masked_count": int(masked_count.sum()),
            "n_masked_neff": int(masked_neff.sum()),
            "neff": ([[float(v) if ok else None for v, ok in zip(row, irow)]
                      for row, irow in zip(contributors.tolist(), inside.tolist())]
                     if contributors is not None else None),
            "neff_unit": shown_unit,
            "masked_frac": float(masked_frac),
            "sparse": bool(sparse),
            "value_min": int(floor),
            "limits_value": robust_limits(value, usable),
            **extra,
        }
    return panels


def finish_sda_panels(grids: dict, weights: dict, *, sums: dict, n_min: int, inside,
                      h_um: float, bin_long_um: float, bin_short_um: float,
                      step_grids: dict, step_weights: dict, neff: dict | None = None,
                      neff_min: float | None = None, neff_unit: str = "tracks",
                      value_h_um: float | None = None, adaptive: bool = False,
                      value_factor: int = 1) -> dict:
    """SDA's four panels from `spatialsda.maps_from_projected_sda`'s output (through
    `_sda_parts`): the LOCATION map through the density pipeline every other method uses
    (`finish_spatial_panels`), marked a probability per bin, and the three value maps
    through `finish_value_panels`. Takes exactly `finish_spatial_panels`' keywords plus
    ``sums`` and ``value_factor`` (both from `_sda_parts`), so `_day_payloads` finishes a
    day the way it finishes the condition.

    ``value_h_um`` is the value maps' own kernel width: the section's rule, ``max(sigma,
    half a bin)``, applied to THEIR bins (`spatialsda.VALUE_BIN_FACTOR` times the
    location map's). At the small bin's width the large bins' edges stayed as steps, in
    Bins and in Smooth alike. None is ``h_um``. ``adaptive`` widens it bin by bin where
    steps are few (`finish_value_panels`); the location map is never touched by it."""
    from .spatialsda import SDA_COUNTS, SDA_KINDS, SDA_UNITS, VALUE_KINDS, VALUE_N_MIN

    common = dict(n_min=n_min, inside=inside, h_um=h_um, bin_long_um=bin_long_um,
                  bin_short_um=bin_short_um, neff_min=neff_min, neff_unit=neff_unit)
    pick = (lambda d, keys: {k: d[k] for k in keys})                    # noqa: E731
    location = finish_spatial_panels(
        pick(grids, ("location",)), pick(weights, ("location",)),
        step_grids=pick(step_grids, ("location",)),
        step_weights=pick(step_weights, ("location",)),
        neff=None if neff is None else pick(neff, ("location",)), **common)
    location["location"].update({"quantity": "probability", "unit": SDA_UNITS["location"],
                                 "count_unit": SDA_COUNTS["location"]})
    values = finish_value_panels(
        pick(sums, VALUE_KINDS), pick(grids, VALUE_KINDS),
        step_grids=pick(step_grids, VALUE_KINDS), weights=pick(weights, VALUE_KINDS),
        step_weights=pick(step_weights, VALUE_KINDS), value_min=VALUE_N_MIN,
        units=SDA_UNITS, count_units=SDA_COUNTS,
        neff=None if neff is None else pick(neff, VALUE_KINDS),
        adaptive=adaptive, value_factor=value_factor,
        **{**common, "h_um": h_um if value_h_um is None else value_h_um})
    both = {**location, **values}
    return {kind: both[kind] for kind in SDA_KINDS}


def _sda_parts(out: dict) -> dict:
    """`finish_sda_panels`' keywords from one `maps_from_projected_sda` output: `_parts`'
    plus the value maps' sums and the size of their bins in the location map's."""
    return {**_parts(out), "sums": dict(out["sums"]),
            "value_factor": int(out.get("value_bin_factor", 1))}


def finish_turning_panels(grids: dict, weights: dict, *, sums: dict, n_min: int, inside, h_um: float,
                          bin_long_um: float, bin_short_um: float, step_grids: dict, step_weights: dict,
                          neff: dict | None = None, neff_min: float | None = None, neff_unit: str = "tracks",
                          value_factor: int = 1) -> dict:
    """The HMM map's Turning back panels (`spatialangles`, 22 Nov 2026) from `spatialangles.maps_from_pairs`' output
    (through `_turning_parts`): each kind's observed share turning back and its chance at the same pairs, both through
    `finish_value_panels` (the same floor, kernel and n_eff mask), then ONE panel per kind, observed minus chance,
    masked where either is. Its colour limits are symmetric about 0 (the robust 2-98 % of the usable bins), so 0 sits
    mid-scale; the observed and chance shares stay in the panel (``observed``, ``chance``) for the receipt and the
    export. Takes `finish_spatial_panels`' keywords plus ``sums`` and ``value_factor``, so `_day_payloads` finishes a
    replicate exactly as its condition."""
    import numpy as np

    from .spatialangles import CHANCE_SUFFIX, COUNT_UNIT, UNIT
    from .spatialsda import VALUE_N_MIN

    names = list(sums)
    against = any(k.endswith(CHANCE_SUFFIX) for k in names)
    both = finish_value_panels(
        sums, grids, step_grids=step_grids, weights=weights, step_weights=step_weights, n_min=n_min,
        value_min=VALUE_N_MIN, inside=inside, h_um=h_um, bin_long_um=bin_long_um, bin_short_um=bin_short_um,
        units={k: "share turning back" for k in names}, count_units={k: COUNT_UNIT for k in names},
        neff=neff, neff_min=neff_min, neff_unit=neff_unit, adaptive=False, value_factor=value_factor)
    if not against:
        # SHOW: TURNING BACK, the data alone: each kind's own share turning back (0.5: no preference)
        for panel in both.values():
            panel["observed"] = panel["value"]
        return both
    panels = {}
    for kind in [k for k in names if not k.endswith(CHANCE_SUFFIX)]:
        seen, free = both[kind], both[kind + CHANCE_SUFFIX]
        masked = np.asarray(seen["masked"], dtype=bool) | np.asarray(free["masked"], dtype=bool)
        value = np.asarray(seen["value"], dtype=float) - np.asarray(free["value"], dtype=float)
        usable = inside & ~masked
        lo, hi = robust_limits(value, usable) if usable.any() else (-0.05, 0.05)
        reach = max(abs(float(lo)), abs(float(hi)), 1e-6)
        panel = dict(seen)
        panel.update({"value": value.tolist(), "masked": masked.tolist(), "unit": UNIT,
                      "observed": seen["value"], "chance": free["value"],
                      "limits_value": (-reach, reach), "n_masked": int((inside & masked).sum()),
                      "masked_frac": float((inside & masked).sum() / inside.sum()) if inside.any() else 1.0})
        panels[kind] = panel
    return panels


def _turning_parts(out: dict) -> dict:
    """`finish_turning_panels`' keywords from one `spatialangles.maps_from_pairs` output."""
    return {**_parts(out), "sums": dict(out["sums"]), "value_factor": int(out.get("value_bin_factor", 1))}


# --------------------------------------------------------------------------- #
# What a panel IS: a density (every method but SDA), a probability, or a mean
# --------------------------------------------------------------------------- #


def paint_field(panel: dict | None, field: str) -> str:
    """The grid a panel is painted from. ``field`` is the grid's own ("density" under a
    shared scale, "counts" per map, `matrix_limits`); an SDA probability map is always its
    density (v2 divided by the localisations), a value map always its MEAN (``"value"``) --
    a mean per bin is not a density, and normalising it would erase its unit."""
    quantity = (panel or {}).get("quantity")
    if quantity == "value":
        return "value"
    if quantity == "probability":
        return "density"
    return field


def limits_key(panel: dict | None, field: str) -> str:
    """The panel's own colour limits for ``field`` (`paint_field`'s)."""
    return f"limits_{paint_field(panel, field)}"


def panel_unit(panel: dict | None, field: str, weight_unit: str = "weighted jumps") -> str:
    """What one panel's colour bar is in: a value map's own unit, a probability per bin,
    or the density / count the grid's scale paints."""
    quantity = (panel or {}).get("quantity")
    if quantity in ("value", "probability"):
        return str(panel["unit"])
    return "density / bin" if field == "density" else f"{weight_unit} / bin"


def has_panel_units(payloads) -> bool:
    """Whether any panel of these payloads carries a unit of its own (SDA): then each
    column's bar names its own unit and no two units share a scale."""
    for payload in payloads or ():
        for panel in ((payload or {}).get("panels") or {}).values():
            if (panel or {}).get("quantity") in ("value", "probability"):
                return True
    return False


def _scale_group(panel: dict | None, field: str) -> tuple:
    """Panels pool colour limits only within one group: one quantity in one unit. Every
    density method's panels are one group, as before; SDA's four maps are four."""
    return (paint_field(panel, field), panel_unit(panel, field))


#: The export's bar title for a grid whose maps each carry their own unit (SDA): each bar
#: then prints its own (µm/s, µm²/s, ...); a title typed in the Export dialog replaces all.
PER_PANEL_UNIT_TITLE = "each map's own unit"


def own_scale_ends(unit: str | None = None) -> tuple:
    """The two end labels of a bar that does not describe one range (each map on its own
    scale, its range printed under it): "low" and "high", the unit beside "high" when the
    column has one of its own."""
    return ("low", f"high, {unit}") if unit else ("low", "high")


def sparse_text(n_jumps: float, measure, width: float) -> str:
    """What a map too sparse to draw says inside its capsule, as wide as ``width`` allows (``measure`` the
    renderer's text width): "n = 58 · too sparse to map" on one line, else on two, else "too sparse" -- never cut at
    both ends (found by the overnight tester, 4 Nov 2026: "≡ 58 — too sparse to m" in the Test page's small capsules).
    The figures line under the capsule says the whole of it either way."""
    n = f"n = {float(n_jumps):,.0f}"
    for text in (f"{n} · too sparse to map", f"{n}\ntoo sparse to map", f"{n}\ntoo sparse", "too sparse"):
        if all(measure(line) <= width for line in text.split("\n")):
            return text
    return "sparse"


def grid_title(method: str) -> str:
    """The grid card's heading."""
    if method in ("sqd", "hmm"):                  # both place steps (an HMM card read "Localisation-density" until 4 Nov 2026)
        return "Jump-density map"
    if method == "sda":
        return "Where molecules are, and how they move there"
    return "Localisation-density map"


#: The colormap each method opens with. jet everywhere by the PI's ruling (`spatialsqd.
#: DEFAULT_CMAP`), except SDA, which opens in v2's own SDA ramp, reversed hot, at the
#: experimenter's request (13 Oct 2026: "with same color map like v2"). Any ramp stays
#: selectable for any method; this is only where each starts.
DEFAULT_CMAP_BY_METHOD = {"sda": V2_HOT_KEY}


def default_cmap_for(method: str) -> str:
    from .spatialsqd import DEFAULT_CMAP

    return DEFAULT_CMAP_BY_METHOD.get(method, DEFAULT_CMAP)


#: The n_eff cutoff's contributor unit (handover §4a.2, "compute it twice") and its range.
#: The value has no criterion yet -- the matched-simulation reference is road step 4 --
#: so every caption calls it PROVISIONAL.
NEFF_UNITS = ("tracks", "cells")
NEFF_MIN_MIN = 2.0
NEFF_MIN_MAX = 20.0


def neff_of(counts) -> float:
    """`(Σc)² / Σc²` for a short list, in plain Python (the pictures' numbers; the maps
    use `smtracker.spatial.maps.effective_contributors`, the same formula)."""
    total = sum(counts)
    square = sum(c * c for c in counts)
    return total * total / square if square else 0.0


def _lcg(seed: int):
    """A tiny deterministic generator for the dialog's pictures: they must not shimmer as a
    slider moves, and they must not depend on numpy (the dialog's side is plain Python)."""
    state = [seed % (2 ** 32)]

    def draw() -> float:
        state[0] = (state[0] * 1664525 + 1013904223) % (2 ** 32)
        return state[0] / 2 ** 32
    return draw


#: The grid the Bin masking field is CALIBRATED at: the median count is ~15 over a
#: 30-column grid, against the reference deposit's ~20 jumps a bin.
MASKING_FIELD_REF_NX = 30
#: Cells the field's ordinary tracks are spread over: many, so an ordinary bin reads about
#: the same by cell as by track, and the one-cell patch is what Cells mode adds.
_MASKING_CELLS = 24


def _split(total: int, parts: int) -> list[int]:
    """`total` jumps over `parts` tracks, as evenly as whole jumps allow."""
    parts = max(1, min(parts, total))
    base, extra = divmod(total, parts)
    return [base + (1 if k < extra else 0) for k in range(parts)]


def _masking_bin(contributors: list[tuple]) -> dict:
    """One bin from its invented contributors `(track, cell, jumps)`: the count and both
    n_effs, COMPUTED -- never typed -- so the picture cannot disagree with the formula."""
    by_track: dict = {}
    by_cell: dict = {}
    for track, cell, jumps in contributors:
        by_track[track] = by_track.get(track, 0) + jumps
        by_cell[cell] = by_cell.get(cell, 0) + jumps
    return {"count": sum(by_track.values()),
            "neff_tracks": neff_of(by_track.values()),
            "neff_cells": neff_of(by_cell.values())}


def _build_masking_field(nx: int, ny: int) -> list[list[dict]]:
    scale = (MASKING_FIELD_REF_NX / max(1, nx)) ** 2
    draw = _lcg(99)
    counts = []
    for iy in range(ny):
        row = []
        for ix in range(nx):
            dx, dy = ix / nx - 0.28, iy / ny - 0.5
            hot = math.exp(-(dx * dx * 26 + dy * dy * 7)) * 55
            row.append(max(0, round((hot + draw() * 20 - 2) * scale)))
        counts.append(row)
    peak = max(max(row) for row in counts)
    cy = ny // 2
    # THE STUCK MOLECULE: a dense spot right of the blob, almost all one track. Dense on
    # purpose -- no sensible `n_min` removes it, which is the point of the second rule.
    sx = min(nx - 1, int(0.72 * nx))
    stuck = {(sx, cy), (min(nx - 1, sx + 1), cy), (sx, max(0, cy - 1))}
    # THE ONE-CELL PATCH: dense bins at the blob's edge built by many tracks, all but one
    # from one cell -- kept when counting tracks, blanked when counting cells.
    px = max(0, int(0.14 * nx))
    patch = {(px, cy), (px, max(0, cy - 1))}
    track = 0
    field = []
    for iy in range(ny):
        row = []
        for ix in range(nx):
            count = counts[iy][ix]
            if (ix, iy) in stuck:
                count = max(8, round(0.85 * peak))
                contributors = [(track, 0, count - 3)] + [
                    (track + 1 + k, 1 + k, 1) for k in range(3)]
            elif (ix, iy) in patch:
                count = max(count, round(0.6 * peak), 6)
                parts = _split(count, max(4, round(count / 2.5)))
                contributors = [(track + k, 0 if k else 1, n) for k, n in enumerate(parts)]
            elif count:
                parts = _split(count, max(1, round(count / 2.5)))
                contributors = [(track + k, int(draw() * _MASKING_CELLS), n)
                                for k, n in enumerate(parts)]
            else:
                contributors = []
            track += len(contributors)
            row.append(_masking_bin(contributors))
        field.append(row)
    return field


#: Built once per grid SHAPE: dragging a threshold must not rebuild the field under it.
_MASKING_FIELD_CACHE: dict = {}


def masking_field(nx: int, ny: int) -> list[list[dict]]:
    """The Bin masking page's cell: per bin, `count`, `neff_tracks` and `neff_cells`.

    INVENTED, and it has to be: before a run there is nothing to show. Each bin is a list
    of made-up contributors and every number comes from it. **The counts scale with the
    BIN, because the jumps do not**: a cell holds the same jumps however it is divided, so
    halving the bin quarters what lands in each, and `Bin size` and the masking rules act
    on one picture. Two features exist to be removed by n_eff and not by the count: a dense
    spot built by one stuck track, and a dense patch built by many tracks of one cell.
    """
    if (nx, ny) not in _MASKING_FIELD_CACHE:
        _MASKING_FIELD_CACHE[(nx, ny)] = _build_masking_field(nx, ny)
    return _MASKING_FIELD_CACHE[(nx, ny)]


def _pass_through(draw, lane: float, inside: int) -> list[tuple[float, float]]:
    """A track crossing the square: one point before it, `inside` points in it, one after."""
    pts = [(-0.07, lane + (draw() - 0.5) * 0.08)]
    for k in range(inside):
        x = 0.14 + (k + 0.5) / inside * 0.72 + (draw() - 0.5) * 0.1
        pts.append((x, min(0.94, max(0.06, lane + (draw() - 0.5) * 0.12))))
    pts.append((1.07, lane + (draw() - 0.5) * 0.08))
    return pts


def _inside(x: float, y: float) -> bool:
    return 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0


def _pair_bin(tracks: list[dict], unit: str) -> dict:
    """A bin of the pair: its tracks, and the count and n_eff COUNTED from the drawn dots."""
    per: dict = {}
    for k, tr in enumerate(tracks):
        n = sum(_inside(x, y) for x, y in tr["points"])
        if n:
            key = k if unit == "tracks" else tr["cell"]
            per[key] = per.get(key, 0) + n
    return {"tracks": tracks, "count": sum(per.values()), "neff": neff_of(per.values())}


def pair_colour_index(bin_: dict, unit: str) -> list[int]:
    """The colour index of each track in a `neff_pair` bin: the contributor n_eff COUNTS.
    By track every track is its own; by cell every track takes its cell's, so the tracks of
    one cell read as one colour -- which is exactly why that bin's cell n_eff is low."""
    if unit not in NEFF_UNITS:
        raise ValueError(f"unit must be one of {NEFF_UNITS}, got {unit!r}")
    if unit == "tracks":
        return list(range(len(bin_["tracks"])))
    return [tr["cell"] for tr in bin_["tracks"]]


#: The pair's key line, per unit: what a colour means follows what n_eff counts.
PAIR_KEY = {
    "tracks": "In the two bins: a line is a track, a dot a jump, a colour a track.",
    "cells": ("In the two bins: a line is a track, a dot a jump, a colour a cell "
              "(one cell can hold several tracks)."),
}


#: What each bin of the pair is, in words -- the n_eff values back the picture (a test holds
#: low < 2 <= high) but are not printed: a reader needs "dominated by one track", not 1.7.
PAIR_TEXT = {
    "low": {"title": "Low n_eff", "tracks": "Dominated by one or a few tracks",
            "cells": "Dominated by one or a few cells"},
    "high": {"title": "High n_eff", "tracks": "Jumps from many different tracks",
             "cells": "Jumps from many different cells"},
}


def neff_pair(unit: str) -> dict:
    """The Bin masking page's two bins, `{"low": bin, "high": bin}`, for one unit.

    Each bin is `tracks` (`{"cell", "points"}` in the square's 0..1 coordinates; points
    outside the square let a line enter and leave it) plus `count` and `neff` counted from
    the points inside. **Low and high hold the same count**: a count cannot tell them
    apart, n_eff can. Tracks: ten short tracks against one track looping inside. Cells: the
    SAME tracks in both bins, coloured by many cells against nearly all one -- by track the
    low bin is still high, which is the case the cell score exists for.
    """
    if unit not in NEFF_UNITS:
        raise ValueError(f"unit must be one of {NEFF_UNITS}, got {unit!r}")
    draw = _lcg(7)
    if unit == "tracks":
        high = [{"cell": k % 5, "points": _pass_through(draw, (k + 0.5) / 10, 2 + k % 2)}
                for k in range(10)]
        loop = [(-0.07, 0.30)]
        x, y = 0.30, 0.40
        for _ in range(19):
            x = min(0.84, max(0.16, x + (draw() - 0.5) * 0.30))
            y = min(0.84, max(0.16, y + (draw() - 0.5) * 0.30))
            loop.append((x, y))
        loop.append((1.07, 0.62))
        low = [{"cell": 0, "points": loop}] + [
            {"cell": 1 + k, "points": _pass_through(draw, lane, 2)}
            for k, lane in enumerate((0.10, 0.52, 0.90))]
        return {"low": _pair_bin(low, unit), "high": _pair_bin(high, unit)}
    paths = [_pass_through(draw, (k + 0.5) / 9, 3 if k < 7 else 2) for k in range(9)]
    many = (0, 1, 2, 3, 4, 5, 6, 0, 1)
    one = (0,) * 8 + (1,)
    return {"low": _pair_bin([{"cell": c, "points": p} for c, p in zip(one, paths)], unit),
            "high": _pair_bin([{"cell": c, "points": p} for c, p in zip(many, paths)], unit)}


def _parts(out: dict, key_for: dict | None = None) -> dict:
    """The keyword arguments `finish_spatial_panels` takes, from one builder's output.
    `key_for` maps panel kinds to the builder's own labels (SQD numbers its
    populations); without it the labels ARE the kinds. One function for the condition
    and every day, so a day page can never be missing what the condition has."""
    pick = (lambda d: {k: d[v] for k, v in key_for.items()}) if key_for else dict
    return {
        "grids": pick(out["grids"]), "weights": pick(out["weights"]),
        "step_grids": pick(out["step_grids"]), "step_weights": pick(out["step_weights"]),
        "neff": {k: {"tracks": t, "cells": c} for (k, t), c in zip(
            pick(out["neff_track"]).items(), pick(out["neff_cell"]).values())},
    }


#: How a map counts: every jump ("steps", what the maps always did) or every track
#: once ("track": each jump weighs 1 / its track's jumps).
WEIGHTINGS = ("steps", "track")
#: The SQD fit behind the map follows the map's weighting (ruled: steps standard, track
#: optional), so a panel never pairs a fit of one convention with counts of the other.
#: By steps it is the legacy step-weighted fit, the one the SQD Run reproduces MATLAB
#: with; the map used to fit track-weighted whatever it was set to.
SQD_FIT_WEIGHTING = {"steps": "step", "track": "track"}
#: What a track-length cutoff removes: the too-long tracks, or every cell holding one.
CUTOFF_MODES = ("tracks", "cells")
#: Which tracks the short/long split (the fit) is learned from when a cutoff is on.
SPLIT_FROM = ("all", "kept")
#: The cutoff's range in localisations per track. Tracks have at least 5 (the
#: tracker's minimum), so a limit of 5 or less would remove every track.
MAX_LOCS_MIN = 6
MAX_LOCS_MAX = 200


def _check_neff(neff_min, neff_unit) -> None:
    if neff_unit not in NEFF_UNITS:
        raise ValueError(f"neff_unit must be one of {NEFF_UNITS}, got {neff_unit!r}")
    if neff_min is not None and not (NEFF_MIN_MIN <= float(neff_min) <= NEFF_MIN_MAX):
        raise ValueError(f"neff_min must be between {NEFF_MIN_MIN} and {NEFF_MIN_MAX}, "
                         f"or None for off; got {neff_min!r}")


def _cutoff_totals(projected) -> dict:
    return {"tracks_total": int(projected["uid"].nunique()),
            "cells_total": int(projected["cell"].nunique()) if "cell" in projected.columns else 0,
            "locs_total": int(len(projected))}


def apply_track_cutoff(projected, max_locs, mode: str):
    """`(kept_table, stats)`: the table with too-long tracks (or their cells) removed.

    A track is too long when it has MORE than `max_locs` localisations in this
    table. `mode="tracks"` removes those tracks; `"cells"` removes every row of
    every cell that holds one. `max_locs=None` is OFF: the SAME table object comes
    back, untouched, so a caller can tell "nothing was cut" by identity. Every
    count in `stats` is a plain int.
    """
    if mode not in CUTOFF_MODES:
        raise ValueError(f"cutoff mode must be one of {CUTOFF_MODES}, got {mode!r}")
    if "uid" not in projected.columns:
        raise ValueError("the track cutoff needs a 'uid' column")
    stats = {"tracks_removed": 0, "cells_removed": 0, "locs_removed": 0,
             **_cutoff_totals(projected)}
    if max_locs is None:
        return projected, stats
    max_locs = int(max_locs)
    if not MAX_LOCS_MIN <= max_locs <= MAX_LOCS_MAX:
        raise ValueError(f"max_locs must be between {MAX_LOCS_MIN} and {MAX_LOCS_MAX} "
                         f"(tracks have at least 5 localisations), got {max_locs}")
    n_locs = projected.groupby("uid", sort=False)["uid"].transform("size")
    too_long = n_locs > max_locs
    if mode == "tracks":
        drop = too_long
    else:
        if "cell" not in projected.columns:
            raise ValueError("the 'Cells + Tracks' cutoff needs a 'cell' column")
        bad = set(projected.loc[too_long, "cell"].dropna().unique())
        # The long track itself always goes, even one with no cell to remove.
        drop = projected["cell"].isin(bad) | too_long
        stats["cells_removed"] = len(bad)
    kept = projected.loc[~drop].copy()
    kept.attrs.update(projected.attrs)
    stats["tracks_removed"] = int(projected.loc[drop, "uid"].nunique())
    stats["locs_removed"] = int(drop.sum())
    return kept, stats


def track_census(projected) -> list:
    """`[[cell, n_localisations], ...]`, one per track, sorted by length -- the
    plain data the dialog's track-length chart and live cost line are drawn from.
    A track with no cell gets `""`."""
    import pandas as pd

    if "uid" not in projected.columns:
        raise ValueError("the track census needs a 'uid' column")
    cells = (projected.groupby("uid", sort=False)["cell"].first()
             if "cell" in projected.columns else None)
    sizes = projected.groupby("uid", sort=False).size()

    def cell_of(uid):
        if cells is None:
            return ""
        value = cells[uid]
        return "" if value is None or pd.isna(value) else str(value)

    rows = [[cell_of(uid), int(n)] for uid, n in sizes.items()]
    return sorted(rows, key=lambda r: r[1])


def cutoff_cost(census, max_locs, mode: str) -> dict:
    """What a cutoff would remove, from the census alone -- so the dialog can say
    it live, before anything runs. The same counts `apply_track_cutoff` reports,
    plus `jumps_removed_pct` (a track of n localisations has n - 1 jumps)."""
    if mode not in CUTOFF_MODES:
        raise ValueError(f"cutoff mode must be one of {CUTOFF_MODES}, got {mode!r}")
    tracks_total = len(census)
    cells_total = len({c for c, _n in census if c != ""})
    locs_total = sum(n for _c, n in census)
    jumps_total = sum(n - 1 for _c, n in census)
    out = {"tracks_removed": 0, "cells_removed": 0, "locs_removed": 0,
           "tracks_total": tracks_total, "cells_total": cells_total, "locs_total": locs_total,
           "jumps_removed_pct": 0.0}
    if max_locs is None or not census:
        return out
    if mode == "tracks":
        removed = [(c, n) for c, n in census if n > max_locs]
    else:
        bad = {c for c, n in census if n > max_locs and c != ""}
        removed = [(c, n) for c, n in census if c in bad or n > max_locs]
        out["cells_removed"] = len(bad)
    out["tracks_removed"] = len(removed)
    out["locs_removed"] = sum(n for _c, n in removed)
    jumps_removed = sum(n - 1 for _c, n in removed)
    out["jumps_removed_pct"] = 100.0 * jumps_removed / jumps_total if jumps_total else 0.0
    return out


def length_histogram(census, *, upto: int = MAX_LOCS_MAX) -> list:
    """Tracks per length, index 0 = 5 localisations; the last bin collects every
    track of `upto` or more."""
    counts = [0] * (upto - 4)
    for _c, n in census:
        counts[min(max(int(n), 5), upto) - 5] += 1
    return counts


def cutoff_text(cost: dict, mode: str, max_locs) -> str:
    """The live cost line under the chart, and the cutoff's caption value."""
    if max_locs is None:
        return "no tracks removed"
    pct = cost["jumps_removed_pct"]
    pct_text = f"{pct:.1f}" if 0 < pct < 1 else f"{pct:.0f}"
    if mode == "cells":
        return (f"removes {cost['cells_removed']:,} of {cost['cells_total']:,} cells "
                f"({cost['tracks_removed']:,} tracks) · {pct_text} % of jumps")
    return (f"removes {cost['tracks_removed']:,} of {cost['tracks_total']:,} tracks · "
            f"{pct_text} % of jumps")


_WEIGHTING_TEXT = {"steps": "BY STEPS · every jump counts",
                   "track": "BY TRACK · every track counts once"}


def _cutoff_caption(payload: dict) -> str:
    if payload.get("max_locs") is None:
        return "OFF"
    stats = payload.get("cutoff_stats") or {}
    what = "tracks" if payload.get("cutoff_mode", "tracks") == "tracks" else "cells + their tracks"
    return (f"≤ {payload['max_locs']} localisations · {what} · removed "
            f"{stats.get('tracks_removed', 0):,} of {stats.get('tracks_total', 0):,} tracks")


_NEFF_UNIT_TEXT = {"tracks": "tracks", "cells": "cells"}


def neff_caption(payload: dict) -> str:
    """The n_eff guard as the caption states it: OFF, or the cutoff with its unit, the
    word PROVISIONAL (no matched-simulation reference yet, handover §4a.3) and, per panel,
    how many bins it masked for too few contributors."""
    if payload.get("neff_min") is None:
        return "OFF"
    # every panel the run drew, in its own order: two populations or three
    masked = " / ".join(str(panel.get("n_masked_neff", 0))
                        for panel in payload.get("panels", {}).values())
    unit = _NEFF_UNIT_TEXT[payload.get("neff_unit", "tracks")]
    return (f"bins worth under {payload['neff_min']:g} {unit} masked · provisional cutoff, not checked "
            f"against a simulation · {masked} bins masked for too few contributors")


#: Under the Spatial Map section's grid: the rows can be clicked.
DAYS_HINT = "Click a condition's maps to see its replicates."


def day_label(day) -> str:
    """A replicate day's row name -- the onboarding screens' own "Day n". `day`
    is a WITHIN-CONDITION ordinal, never a date and never paired across
    conditions (Rule 0)."""
    return f"Replicate {int(day)}"


#: The methods with profiles along and across the cell: those whose maps have populations, and SDA (23 Nov 2026: v2's
#: x and y graphs beside its maps, `spatialsda.sda_profiles`; a value map's profile is its mean value per bin).
PROFILE_METHODS = ("sqd", "appd", "hmm", "sda")


def profiles_by_kind(profiles, key_for: dict) -> dict | None:
    """A builder's `axis_profiles` with its populations under the panel kinds (``key_for``: kind -> the builder's
    own label), or None where the builder made none."""
    if not profiles:
        return None
    pops = profiles["populations"]
    return {**{k: v for k, v in profiles.items() if k != "populations"},
            "populations": {kind: pops[label] for kind, label in key_for.items() if label in pops}}


def _profile_rows(entries, kind: str) -> list:
    """The table under the profiles: per entry its mean |d_norm|, mean distance from the pole and share in the caps
    for population ``kind``, and the even expectation; for a value map (an SDA speed, D or acceleration) its rows and
    its mean over all of them, in the caps and in the cylinder."""
    rows = []
    for name, prof in entries:
        pop = (prof or {}).get("populations", {}).get(kind)
        if pop and pop.get("unit"):
            rows.append([name, f"{pop['rows']:,}", *[_value_text(pop.get(k)) for k in
                                                       ("mean", "mean_caps", "mean_cylinder")]])
            continue
        if not pop or pop.get("mean_abs_d") is None:
            rows.append([name, "–", "–", "–", "–"])
            continue
        rows.append([name, f"{pop['weight']:,.0f}", f"{pop['mean_abs_d']:.3f}",
                     f"{1000 * pop['mean_from_pole_um']:.0f}", f"{100 * pop['in_caps']:.1f}"])
    return rows


def _value_text(value) -> str:
    return "–" if value is None else f"{value:.3g}"


PROFILE_COLUMNS = ["", "Jumps", "Mean |d_norm|", "Mean from pole (nm)", "In the caps (%)"]


def _value_unit(prof: dict, kind: str) -> str | None:
    """The unit of ``kind``'s profile where it is a value map's (an SDA speed, D or acceleration), else None."""
    return ((prof or {}).get("populations", {}).get(kind) or {}).get("unit") or None


def _columns_for(prof: dict, kind: str) -> list:
    """The table's columns for ``kind``: a value map's means in its unit, else the distribution's measures (the count
    named by what the map counts: jumps, or an SDA map's localisations)."""
    unit = _value_unit(prof, kind)
    if unit:
        return list(VALUE_PROFILE_COLUMNS)
    count = ((prof or {}).get("count_names") or {}).get(kind, "Jumps")
    return ["", count, *PROFILE_COLUMNS[2:]]


#: The table of the SDA value maps (speed, D, acceleration), one column set for all three: the unit goes with the map's
#: name ("Speed (µm/s)", `ui.workspace.spatial_profiles`), the count is steps or, for acceleration, step pairs.
VALUE_PROFILE_COLUMNS = ("", "Count", "Mean", "In the caps", "In the cylinder")


def _profile_extras(prof: dict, kind: str) -> dict:
    """What tells a value map's profile from a distribution's: its unit and y label, no even reference, SDA's note
    and heading."""
    unit = _value_unit(prof, kind)
    sda = bool((prof or {}).get("sda"))
    out = {"unit": unit, "y_label": f"mean, {unit}" if unit else "density", "row_name": "Map" if sda else "Population",
           "heading": "Along and across the cell" if sda else "Where the jumps sit",
           "note": SDA_PROFILE_NOTE if sda else PROFILE_NOTE}
    if unit:
        out.update({"long_reference": None, "short_reference": None})
    return out


def profile_card_payload(payloads: dict, conditions: list, labels: dict, kind: str) -> dict | None:
    """The overview's Profiles card for population ``kind``: one line per condition along the cell (distance from
    the nearer pole) and across it (|d_norm|), the even-distribution reference, and the table. None before any
    condition has profiles."""
    have = [(c, payloads[c]["profiles"]) for c in conditions
            if c in payloads and (payloads[c] or {}).get("profiles")]
    if not have:
        return None
    ref = have[0][1]
    lines = [{"key": c, "name": labels.get(c, c), "bold": True, **_line_values(p, kind)} for c, p in have]
    rows = _profile_rows([(labels.get(c, c), p) for c, p in have], kind)
    if not _value_unit(ref, kind):
        rows.append(["even through the cell", "", f"{ref['mean_abs_d_even']:.3f}", "",
                     f"{100 * ref['in_caps_even']:.1f}"])
    return {"lines": lines, **_axes_of(ref), "columns": _columns_for(ref, kind), "rows": rows,
            **_profile_extras(ref, kind)}


def _line_values(prof: dict, kind: str) -> dict:
    """One line's densities and the weighted counts behind them (the export's source data)."""
    pop = (prof or {}).get("populations", {}).get(kind, {})
    return {"long": pop.get("long"), "short": pop.get("short"), "long_counts": pop.get("long_counts"),
            "short_counts": pop.get("short_counts")}


def _axes_of(prof: dict) -> dict:
    """The bins, the even references and the drawn geometry every profile payload carries."""
    return {"long_centres": prof["long_centres"], "short_centres": prof["short_centres"],
            "long_edges": prof.get("long_edges"), "short_edges": prof.get("short_edges"),
            "long_reference": prof["long_reference"], "short_reference": prof["short_reference"],
            "half_length_um": prof["half_length_um"], "cap_um": prof["cap_um"], "geometry": prof.get("geometry")}


def day_profile_payload(payload: dict, kind: str, condition_name: str) -> dict | None:
    """The day page's Profiles card for population ``kind``: the condition's profile (bold) and each replicate day's
    (thin), along and across the cell, with the even reference and the table (a row per day)."""
    prof = (payload or {}).get("profiles")
    if not prof:
        return None
    days = [d for d in payload.get("days") or [] if d.get("profiles")]
    lines = [{"key": "condition", "name": condition_name, "bold": True, **_line_values(prof, kind)}]
    for i, d in enumerate(days):
        lines.append({"key": f"replicate {d['day']}", "name": f"replicate {d['day']}", "bold": False, "day_index": i,
                      "n_days": len(days), **_line_values(d["profiles"], kind)})
    rows = _profile_rows([(condition_name, prof)] + [(f"replicate {d['day']}", d["profiles"]) for d in days], kind)
    if not _value_unit(prof, kind):
        rows.append(["even through the cell", "", f"{prof['mean_abs_d_even']:.3f}", "",
                     f"{100 * prof['in_caps_even']:.1f}"])
    return {"lines": lines, **_axes_of(prof), "columns": _columns_for(prof, kind), "rows": rows,
            **_profile_extras(prof, kind)}


PROFILE_NOTE = ("Across the cell: each jump's relative short-axis position, -1 and 1 at the outline on either "
                "side and 0 on the cell axis. Only jumps in the cylinder count; the caps (18 % of the length at each "
                "end) are left out. Along the cell: each jump's distance from the nearer pole, both poles "
                "pooled. Dashed: a population spread evenly through the cell's volume. It is not flat, because a "
                "chord through a cylinder is longest on its axis (mean |d_norm| 0.424). Each curve has area 1, so "
                "you can compare shapes but not counts. Compare shifts between populations in the same cells; the "
                "outline is not the membrane. The eye beside a name hides its curve.")
#: The SDA maps' x and y graphs (23 Nov 2026, v2's beside its maps; `spatialsda.sda_profiles`).
SDA_PROFILE_NOTE = ("Beside each SDA map, along and across the cell. Along: distance from the nearer pole, both poles "
                    "pooled. Across: the relative short-axis position, -1 and 1 at the outline and 0 on the cell axis, "
                    "from the cylinder only (the caps, 18 % of the length at each end, are left out). Localisations: "
                    "their density (area 1), the dashed curve a population spread evenly through the cell's volume. "
                    "Speed, D and acceleration: the mean of the measured values in each bin, in the map's unit; a bin "
                    "with too few steps has no point. SMTracker 2 drew the median of values interpolated onto a grid "
                    "instead; these are never interpolated. On a replicate page each replicate is its own line. The "
                    "eye beside a name hides its curve.")


def _day_payloads(projected, base: dict, make_maps, kinds_from, *, n_min: int,
                  inside, h_um: float, bin_long_um: float,
                  bin_short_um: float, kind_keys: dict | None = None,
                  cfg=None, neff_min: float | None = None,
                  neff_unit: str = "tracks", finish=None) -> list[dict]:
    """The condition's maps again, once per replicate day, under the condition's
    OWN rule.

    `make_maps(sub)` draws one day's rows with the fit (or threshold) the
    condition was already drawn with, and `kinds_from(out)` maps its
    populations onto the panel kinds exactly as the condition did -- so nothing
    is refitted, and a difference between two day rows is where that day's
    jumps fell, never a different rule. A track lives in one movie and so one
    day, so a day's rows never cut a track. The days PARTITION the condition's
    jumps (tested: their `n_jumps` sum to the condition's).

    Every day carries the condition's settings and fit (`base`), so its colour
    limits pool with the other days' through `panel_limits` like conditions do.

    With `kind_keys` (panel kind -> the population key in `make_maps`' output) and
    `cfg`, `make_maps` is expected to have been asked for its step record, and each
    day also carries `spots`: which track every binned jump came from (`_spot_block`),
    what a click on that day's map reads (`spot_contributors`).

    `finish` is how a day's grids become panels, `finish_spatial_panels` unless the
    method says otherwise (SDA's `finish_sda_panels`): the condition's own finisher, so
    a day is finished exactly as its condition is.
    """
    if "day" not in projected.columns:
        raise ValueError("the projected table has no 'replicate' column, so the map "
                         "cannot be split by replicate")
    finish = finish_spatial_panels if finish is None else finish
    days = sorted(int(d) for d in projected["day"].dropna().unique())
    out = []
    for day in days:
        day_rows = projected[projected["day"] == day]
        made = make_maps(day_rows)
        panels = finish(
            **kinds_from(made),
            n_min=n_min, inside=inside, h_um=h_um,
            bin_long_um=bin_long_um, bin_short_um=bin_short_um,
            neff_min=neff_min, neff_unit=neff_unit)
        entry = {**base, "day": day, "panels": panels}
        if kind_keys is not None and made.get("profiles"):
            entry["profiles"] = profiles_by_kind(made["profiles"], kind_keys)
        if kind_keys is not None and cfg is not None and "steps" in made:
            entry["spots"] = _spot_block(made["steps"], day_rows, kind_keys,
                                         inside, cfg)
        out.append(entry)
    return out


#: How many of a spot's tracks are listed and drawn: the largest first.
SPOT_TOP_TRACKS = 5

#: What n_eff means, printed on the spot card under its numbers. Said in words because the
#: number alone reads like a count, and it is not one: 92 tracks can be worth 2.
N_EFF_EXPLAINED = (
    "n_eff is the number of independent tracks (or cells) the spot is made of: 20 tracks giving 5 % each count as "
    "20, while one track giving 90 % beside many tiny ones counts as about 1. Formula: (sum of shares)² / sum of "
    "squared shares, taken through the smoothing so that it describes the colour you clicked. Resting the pointer "
    "on a bin gives that bin's own n_eff.")


#: What n_eff is, why it is on these maps, and what it costs -- the hover card of the n_eff
#: view and of the Bin masking page. Asked: "nowhere its explained what n_eff is and whats
#: the advantage/disadvantage of it". Plain words, no threshold claimed that was not measured.
N_EFF_ABOUT = (
    "n_eff, the effective number of contributors, is the number of independent tracks (or cells) a bin's jumps "
    "come from: (sum of shares)² / sum of squared shares, where a share is a track's part of the bin. Ten "
    "tracks giving 10 % each count as 10; one track giving 90 % beside many tiny ones counts as about 1.\n\n"
    "What it buys: a bin can hold hundreds of jumps from one molecule, and the colour cannot tell that from many "
    "molecules. The n_eff view shows which parts of a map rest on few. The n_eff cutoff under Bin masking blanks "
    "bins with too few contributors.\n\n"
    "What it costs: masking blanks the thin places, usually the rim and the poles, so read a masked map beside its "
    "unmasked one. n_eff describes and does not test; a high n_eff does not make a difference between maps "
    "significant. A bigger bin size reads higher, so compare n_eff only between maps with the same bins.")


def _spot_block(steps: dict, rows, kind_keys: dict, inside, cfg) -> dict:
    """One day's per-jump record as plain lists, plus each track's path.

    `steps` is `spatialsqd.step_block`'s output for this day's drawn rows; `rows`
    is the day's projected table, from which each track's localisations are laid
    out in the same map frame the jumps were binned in (`x = l_norm * L`,
    `y = d_norm * local half-width`). Only tracks that put a jump in a bin are kept.

    ONE NUMPY PASS, NEVER A GROUPBY. Slicing the table once per track cost ~4 ms a
    track (pandas copies the table's metadata into every slice): 13.2 of 15.9 s on
    protein A/Malate, for a record nobody had clicked yet.
    """
    import numpy as np

    from .spatialsqd import local_half_width, long_axis_x

    uids, first, track = np.unique(steps["uid"].astype(str), return_index=True,
                                   return_inverse=True)
    w = {kind: np.asarray(steps["w"][key], dtype=float).tolist()
         for kind, key in kind_keys.items()}

    uid = rows["uid"].astype(str).to_numpy()
    code = np.searchsorted(uids, uid)
    known = (code < len(uids)) & (uids[np.minimum(code, len(uids) - 1)] == uid)
    frame = (rows["frame"].to_numpy(float) if "frame" in rows.columns
             else np.arange(len(rows), dtype=float))
    code, frame = code[known], frame[known]
    n_locs = np.bincount(code, minlength=len(uids))
    order = np.lexsort((frame, code))
    code, frame = code[order], frame[order]

    def column(name):
        return (rows[name].to_numpy(float)[known][order] if name in rows.columns
                else np.full(len(code), np.nan))

    l_norm, d_norm = column("l_norm"), column("d_norm")
    ok = np.isfinite(l_norm) & np.isfinite(d_norm)
    cells = None
    if getattr(cfg, "long_axis", "normalised") != "normalised":
        cells = rows["cell"].astype(str).to_numpy()[known][order][ok]
    x = long_axis_x(l_norm[ok], cfg, cells=cells, geometry=rows.attrs.get("geometry"))
    y = d_norm[ok] * local_half_width(x, cfg)

    def per_track(values, codes):
        cuts = np.searchsorted(codes, np.arange(1, len(uids)))
        return [part.tolist() for part in np.split(values, cuts)] if len(uids) else []

    xs = per_track(np.round(x, 5), code[ok])
    ys = per_track(np.round(y, 5), code[ok])
    # Each track's own JUMPS, in nm, between consecutive frames only: what the spot
    # page reads to say whether a track moved at all (`spot_page_view.strip_rows`).
    x_um, y_um = column("x_um"), column("y_um")
    same = (code[1:] == code[:-1]) & (np.diff(frame) == 1)
    jumps = np.hypot(np.diff(x_um), np.diff(y_um))[same] * 1000.0
    jumps_nm = per_track(np.round(jumps, 1), code[:-1][same])
    starts = np.searchsorted(code, np.arange(len(uids)))
    ends = np.searchsorted(code, np.arange(len(uids)), side="right") - 1

    movie = steps.get("movie")
    cell = steps.get("cell")
    return {
        "track": track.astype(int).tolist(),
        "ix": np.asarray(steps["ix"], dtype=int).tolist(),
        "iy": np.asarray(steps["iy"], dtype=int).tolist(),
        "w": w,
        "inside": np.asarray(inside, dtype=bool).tolist(),
        "tracks": {
            "uid": [str(u) for u in uids],
            "movie": [None] * len(uids) if movie is None else [str(movie[i]) for i in first],
            "cell": [None] * len(uids) if cell is None else [str(cell[i]) for i in first],
            "n_locs": n_locs.astype(int).tolist(),
            # u-track's own number, the part of `uid` after the movie: what a person
            # looking for the track in u-track or a viewer types.
            "track_id": [str(u).rsplit(":", 1)[-1] for u in uids],
            "first_frame": [int(frame[i]) if n else None for i, n in zip(starts, n_locs)],
            "last_frame": [int(frame[j]) if n else None for j, n in zip(ends, n_locs)],
            "x": xs,
            "y": ys,
            "jumps_nm": jumps_nm,
        },
    }


def participation(contributions) -> dict:
    """How many independent things a total is made of: the participation ratio.

    `n_eff = (sum c)^2 / sum c^2` over the contributors' shares -- the inverse Simpson
    index, `docs/gui_handover_sqd_run.md` section 4a item 2. Twenty equal tracks give 20;
    one track holding most of it gives close to 1, however many others touch it. It
    needs no cut. `top_share` is the largest single contributor's share. Zero
    contributions count for nothing.
    """
    import numpy as np

    c = np.asarray(contributions, dtype=float)
    c = c[c > 0]
    total = float(c.sum())
    if total <= 0:
        return {"total": 0.0, "n": 0, "n_eff": 0.0, "top_share": 0.0}
    return {"total": total, "n": int(c.size),
            "n_eff": float(total ** 2 / (c ** 2).sum()),
            "top_share": float(c.max() / total)}


def spot_contributors(day_payload: dict, kind: str, ix: int, iy: int, *,
                      top: int = SPOT_TOP_TRACKS) -> dict | None:
    """What the spot drawn at bin (`ix`, `iy`) of one day's `kind` map is made of.

    THE SPOT IS WHAT IS DRAWN, NOT THE BIN. The picture is the bin counts smoothed
    by a Gaussian of bandwidth `h` (`finish_spatial_panels`), over the bins that are
    in the cell and not masked, then renormalised -- so a bright spot is fed by the
    jumps in a neighbourhood of bins, and asking "which tracks are in this bin"
    would miss most of it. Each jump is weighted here by exactly the kernel the
    smoothing used (`scipy.ndimage.gaussian_filter1d` on an impulse, so its
    truncation and edge reflection are the smoothing's own), and on a folded map by
    all four of its reflections, the way `fold_counts` pools them. A track's share
    is then its part of the value at that bin, and the shares add up to the drawn
    value (tested to rounding).

    Returns None for a click outside the capsule, and a dict with the spot's
    position, `by_track`/`by_cell` (`participation`), and the `top` tracks with
    their share, movie, cell, length and path in the map frame. On a folded map
    each path is reflected into the clicked quarter, as the map itself is.

    A VALUE MAP (SDA's speed, D and acceleration: a panel whose ``quantity`` is
    ``"value"``) is binned in value bins of ``value_bin_factor`` x ``value_bin_factor``
    small ones (`spatialsda` divergence 8), so a click there lists the WHOLE value bin
    holding the clicked small bin -- the same answer from any of its small bins -- and
    the location map keeps its own bin. Each step then weighs what it weighs in the
    drawn mean: the value map's own kernel (its ``h_nm`` per bin when adaptively
    smoothed, its fixed width otherwise), summed over the fine bins of the step's value
    bin that the kernel read (``kernel_in``), at each small bin of the clicked value bin.
    ``bin_long_nm``/``bin_short_nm`` say the size listed, ``h_range_nm`` the widths.
    A value map's ``value`` is None (a mean is not the sum the shares add up to);
    ``weight`` is that sum, the drawn mean's denominator over the value bin.
    """
    import numpy as np

    spots = day_payload.get("spots")
    if spots is None:
        return None
    inside = np.asarray(spots["inside"], dtype=bool)
    n_short, n_long = inside.shape
    if not (0 <= ix < n_long and 0 <= iy < n_short) or not inside[iy, ix]:
        return None
    panel = day_payload["panels"][kind]
    masked = np.asarray(panel["masked"], dtype=bool)
    usable = inside & ~masked

    length = float(day_payload["cell_length_um"])
    width = float(day_payload["cell_width_um"])
    bin_long, bin_short = length / n_long, width / n_short

    jx = np.asarray(spots["ix"], dtype=int)
    jy = np.asarray(spots["iy"], dtype=int)
    w = np.asarray(spots["w"][kind], dtype=float)
    mirrored = bool(day_payload.get("mirrored"))
    reflections = [(jy, jx)]
    if mirrored:
        reflections += [(n_short - 1 - jy, jx), (jy, n_long - 1 - jx),
                        (n_short - 1 - jy, n_long - 1 - jx)]
    is_value = panel.get("quantity") == "value"
    if is_value:
        f = int(day_payload.get("value_bin_factor") or 1)
        if n_short % f or n_long % f:
            raise ValueError(f"a {n_short} x {n_long} grid is not whole {f} x {f} value bins")
        kernel_in = np.asarray(panel.get("kernel_in", usable), dtype=bool)
        bx0, by0 = (ix // f) * f, (iy // f) * f
        cells = [(by0 + a, bx0 + c) for a in range(f) for c in range(f)
                 if inside[by0 + a, bx0 + c]]
        widths = panel.get("h_nm")
        h0_nm = float(panel.get("h0_nm") or day_payload.get("value_h_nm")
                      or day_payload["h_nm"])
        kernel = np.zeros(len(w))
        weight_of: dict = {}
        used = []
        for ty, tx in cells:
            h_nm = (widths[ty][tx] if widths and widths[ty][tx] is not None else h0_nm)
            used.append(float(h_nm))
            if h_nm not in weight_of:
                my = _impulse_matrix(n_short, h_nm / 1000.0 / bin_short)
                mx = _impulse_matrix(n_long, h_nm / 1000.0 / bin_long)
                weight_of[h_nm] = (my, mx)
            my, mx = weight_of[h_nm]
            # what one step of value bin (By, Bx) weighs at (ty, tx): the kernel over that
            # value bin's fine bins the smoothing read
            per_bin = sum(kernel_in[a::f, c::f] * np.outer(my[ty, a::f], mx[tx, c::f])
                          for a in range(f) for c in range(f))
            for ry, rx in reflections:
                kernel += per_bin[ry // f, rx // f]
        contribution = w * kernel
        den = None
        xc = (bx0 + f / 2.0) * bin_long
        yc = -width / 2.0 + (by0 + f / 2.0) * bin_short
        spot_masked = bool(all(masked[ty, tx] for ty, tx in cells))
        h_range = [min(used), max(used)]
    else:
        f = 1
        h_nm = float(day_payload["h_nm"])
        h_um = h_nm / 1000.0
        # M[b, i]: the smoothing's response at bin b to one count at bin i.
        mx = _impulse_matrix(n_long, h_um / bin_long)
        my = _impulse_matrix(n_short, h_um / bin_short)
        kernel = np.zeros(len(w))
        for ry, rx in reflections:
            kernel += usable[ry, rx] * my[iy, ry] * mx[ix, rx]
        contribution = w * kernel
        den = float((my[iy][:, None] * mx[ix][None, :] * usable).sum())
        xc = (ix + 0.5) * bin_long
        yc = -width / 2.0 + (iy + 0.5) * bin_short
        spot_masked = bool(masked[iy, ix])
        h_range = [h_nm, h_nm]

    tracks = spots["tracks"]
    sigma_nm = day_payload.get("sigma_nm")
    floor_nm = None
    if sigma_nm and float(sigma_nm) > 0:
        from .inspector import floor_band

        floor_nm = 1000.0 * floor_band(float(sigma_nm) / 1000.0, quantile=0.9)
    codes = np.asarray(spots["track"], dtype=int)
    per_track = np.bincount(codes, weights=contribution, minlength=len(tracks["uid"]))
    cell_keys = [f"{m}|{c}" for m, c in zip(tracks["movie"], tracks["cell"])]
    cell_names, cell_first, cell_code = np.unique(
        np.asarray(cell_keys, dtype=str), return_index=True, return_inverse=True)
    per_cell = np.bincount(cell_code, weights=per_track, minlength=len(cell_names))
    top_cell = int(cell_first[int(np.argmax(per_cell))]) if len(per_cell) else None

    total = float(per_track.sum())
    order = np.argsort(-per_track)
    listed = []
    for code in order[:top]:
        if per_track[code] <= 0:
            break
        px = np.asarray(tracks["x"][code], dtype=float)
        py = np.asarray(tracks["y"][code], dtype=float)
        if mirrored:
            px = length / 2.0 + np.sign(xc - length / 2.0) * np.abs(px - length / 2.0)
            py = np.sign(yc) * np.abs(py)
        jumps = np.asarray((tracks.get("jumps_nm") or [[]] * len(tracks["uid"]))[code],
                           dtype=float)
        listed.append({
            "uid": tracks["uid"][code],
            "movie": tracks["movie"][code],
            "cell": _cell_of(tracks, code),
            "n_locs": int(tracks["n_locs"][code]),
            "track_id": (tracks.get("track_id") or [None] * len(tracks["uid"]))[code],
            "first_frame": (tracks.get("first_frame") or [None] * len(tracks["uid"]))[code],
            "last_frame": (tracks.get("last_frame") or [None] * len(tracks["uid"]))[code],
            "n_jumps": int(jumps.size),
            "median_jump_nm": float(np.median(jumps)) if jumps.size else None,
            "within_floor": (None if floor_nm is None or not jumps.size
                             else int((jumps <= floor_nm).sum())),
            "share": float(per_track[code] / total) if total > 0 else 0.0,
            # Its own jumps (nm), what `median_jump_nm` and `within_floor` summarise: the
            # Spatial Map's replicate page draws each one against the immobile band.
            "jumps_nm": [float(j) for j in jumps],
            # In the map frame, as fractions of the capsule's box (0..1 along, 0 at the
            # bottom wall): what the chart draws over the picture.
            "path": [[float(a / length), float(b / width + 0.5)] for a, b in zip(px, py)],
        })
    return {
        "day": day_payload.get("day"),
        "kind": kind,
        "method": day_payload.get("method"),
        "n_pop": day_payload.get("n_pop"),
        "ranges": day_payload.get("ranges"),
        "condition": day_payload.get("condition"),
        "ix": int(ix), "iy": int(iy),
        # the bin listed: the clicked small bin, or on a value map its whole value bin
        # (`bin_ix`/`bin_iy` its first small bin, `bin_factor` small bins a side)
        "bin_ix": int((ix // f) * f), "bin_iy": int((iy // f) * f), "bin_factor": int(f),
        "bin_long_nm": float(bin_long * f * 1000.0),
        "bin_short_nm": float(bin_short * f * 1000.0),
        "value_bin": bool(is_value),
        "h_range_nm": [float(h_range[0]), float(h_range[1])],
        "x_um": float(xc), "y_um": float(yc),
        "masked": spot_masked,
        "mirrored": mirrored,
        "value": None if den is None else (total / den if den > 1e-6 else 0.0),
        "weight": total,
        "by_track": participation(per_track),
        "floor_nm": floor_nm,
        "sigma_nm": None if not sigma_nm else float(sigma_nm),
        "by_cell": participation(per_cell),
        "top_cell": None if top_cell is None else _cell_of(tracks, top_cell),
        # The `top` largest cells and their shares, the cells' counterpart of `tracks`.
        "cells": [{"cell": _cell_of(tracks, int(cell_first[k])),
                   "movie": tracks["movie"][int(cell_first[k])],
                   "share": float(per_cell[k] / total) if total > 0 else 0.0}
                  for k in np.argsort(-per_cell)[:top] if per_cell[k] > 0],
        "tracks": listed,
    }


def _cell_of(tracks: dict, code) -> str | None:
    """A track's cell, or its movie where the table carries no cell column."""
    cell = tracks["cell"][int(code)]
    return cell if cell is not None else tracks["movie"][int(code)]


def _pct(share: float) -> str:
    """A share as a percentage that never reads 0 % for a contributor: a track that
    touches a spot at all shows "<1%", not a zero the eye takes as "not there"."""
    return "<1%" if 0 < share < 0.005 else f"{share:.0%}"


def _short(name, condition) -> str:
    """A cell or movie as the screens name it: the condition prefix stripped
    ("Malate_series005:1" -> "series005:1"), never split on an underscore."""
    from .onboarding_view import short_movie_name

    return "?" if name is None else short_movie_name(name, condition or "")


def spot_bin_text(spot: dict) -> str:
    """The spot card's "Bin" row: the size of the bin whose tracks are listed and the
    kernel they are weighed through. On a value map that is the value bin, a block of
    small ones, whichever of them was clicked."""
    size = f"{spot['bin_long_nm']:.0f} × {spot['bin_short_nm']:.0f} nm"
    lo, hi = spot.get("h_range_nm") or (None, None)
    kernel = ("" if lo is None else f" · through the map's kernel, h {lo:.0f} nm"
              if hi - lo < 0.5 else f" · through the map's kernel, h {lo:.0f}–{hi:.0f} nm")
    f = int(spot.get("bin_factor", 1))
    if spot.get("value_bin") and f > 1:
        return (f"{size}, the whole value bin ({f} × {f} of the localisation map's), "
                f"as this map is binned{kernel}")
    return f"{size}{kernel}"


def spot_rows(spot: dict) -> list[tuple[str, str]]:
    """The readout under a clicked spot, as (name, value) rows. Reports, never judges:
    no n_eff threshold is established for these maps yet (the matched-simulation
    reference in `docs/gui_handover_sqd_run.md` 4a is what would set one)."""
    method = spot.get("method") or "sqd"
    kinds = kinds_for(method, spot.get("n_pop") or 2, spot.get("ranges"))
    title = (panel_titles_for(method, spot.get("n_pop") or 2, spot.get("ranges"))[kinds.index(spot["kind"])]
             if spot["kind"] in kinds else spot["kind"])
    where = (f"{day_label(spot['day'])} · {title} · {spot['x_um']:.2f} µm along, "
             f"{spot['y_um']:+.2f} µm across")
    rows = [("Spot", where + (" · folded, so its four mirror positions pool"
                              if spot["mirrored"] else ""))]
    if spot.get("bin_long_nm"):
        rows.append(("Bin", spot_bin_text(spot)))
    if spot["masked"] and spot.get("value_bin"):
        rows.append(("Grey", "too few steps for a mean, drawn grey; these are the steps "
                             "its kernel reaches"))
    elif spot["masked"]:
        rows.append(("Grey", "below the minimum per bin, drawn grey; these are the "
                             "jumps its smoothing neighbourhood carries"))
    t, c = spot["by_track"], spot["by_cell"]
    if t["n"] == 0:
        rows.append(("Made of", "nothing: no jump of this map reaches this spot"))
        return rows
    rows.append(("Made of", f"{t['n']:,} tracks from {c['n']:,} cells"))
    rows.append(("Spot worth (n_eff)",
                 f"{c['n_eff']:.1f} independent cells · {t['n_eff']:.1f} independent tracks"))
    rows.append(("Largest track", f"{_pct(t['top_share'])} of this spot"))
    rows.append(("Largest cell", f"{_pct(c['top_share'])} of this spot · "
                                 f"{_short(spot['top_cell'], spot['condition'])}"))
    return rows


def held_calibration(fit):
    """The per-spot calibration a section's fit was made with (`map_fit_source`'s ``"precision"``: ``(c_nm, p,
    source)``, 6 Nov 2026), as a `precision.Calibration`; None for a fit without one, or no fit. A map that takes a
    per-spot fit's D labels its steps (or tracks) with the same noise."""
    precision = (fit or {}).get("precision")
    if not precision:
        return None
    from .precision import Calibration

    return Calibration(c_nm=float(precision[0]), p=float(precision[1]), source=str(precision[2]))


#: THE MAP'S OWN FIT PER SPOT (6 Nov 2026; asked "the spot precision stuff ... everywhere its possible"): the Run
#: dialog's Localisation error on the Populations page, shown while the box "Use the same D as the ... fit" is unticked
#: (ticked, the section's fit brings its own calibration, `held_calibration`) and the split is Fitted. A Run config
#: carries it ONLY when set (`map_config(per_spot=True)`), so a map made before keeps its cache entry and its caption;
#: the window resolves the Run's calibration (`workspace_view.run_precision`) and hands each job its
#: ``"precision"`` -- none to a condition whose tracks carry no precision (a truth condition), fitted without it.
OWN_PRECISION_KEY = "per_spot"
#: The dialog's two choices, as the Apparent Diffusion section's Run names them.
OWN_PRECISION_LABELS = ("Not modelled", "Per spot")
OWN_PRECISION_NOTE = ("Per spot: the map's fit uses each jump's own localisation error, from the tracker's precision and a "
                      "calibration measured on immobile spots. A condition whose tracks carry no precision is fitted without "
                      "it.")
OWN_PRECISION_LAG_REFUSAL = ("Localisation error Per spot fits one time lag: lags 1-4 together reuse each jump in "
                             "up to four displacements, so a likelihood over them counts it up to four times. Pick "
                             "one lag.")


def _own_precision(precision, fixed, split):
    """The calibration of the map's OWN fit Per spot (``precision``: ``(c_nm, p, source)``), or None: not asked, a
    section's fit (its own decides) or a split by hand (nothing fitted)."""
    if not precision or fixed is not None or split == "manual":
        return None
    return held_calibration({"precision": precision})


def _own_precision_stamp(own, precision) -> dict:
    """The payload's record of the map's own fit Per spot: ``{"precision": [c_nm, p, source]}``, nothing otherwise."""
    if own is None:
        return {}
    return {"precision": [float(precision[0]), float(precision[1]), str(precision[2])]}


def own_precision_text(payloads) -> str | None:
    """The Fit row's words for a map's own fit Per spot: "each step's own localisation error (per spot, <source>)",
    and the conditions fitted without it named ("not modelled for ..."); None where no map of ``payloads`` was."""
    payloads = [p for p in payloads or () if p]
    with_ = [p for p in payloads if p.get("precision")]
    if not with_:
        return None
    unit = "track" if with_[0].get("method") == "appd" else "step"
    text = f"each {unit}'s own localisation error (per spot, {with_[0]['precision'][2]})"
    without = [str(p.get("condition")) for p in payloads
               if not p.get("precision") and not p.get("section_fit") and p.get("split") != "manual"]
    return text + (f"; not modelled for {', '.join(without)}" if without else "")


def compute_spatial_map_sqd(project, condition: str, *, step_position: str,
                             assignment: str, mirrored: bool, n_min: int,
                             bin_nm: float, cell_length_um: float,
                             cell_width_um: float, frame_source: str,
                             weighting: str = "steps", max_locs=None,
                             cutoff_mode: str = "tracks",
                             split_from: str = "all", n_pop: int = 2,
                             neff_min: float | None = None,
                             neff_unit: str = "tracks",
                             cell_class: str = "all", class_cuts_um=None,
                             long_axis: str = "normalised", lag_frames: int = 1,
                             split: str = SPLIT_DEFAULT, ranges=None, section_fit=None, precision=None) -> dict:
    """One condition's jump-density maps, from the project ALREADY LOADED.

    ``section_fit`` (4 Nov 2026, `map_fit_source`): the SQD section's fit of this condition, its D HELD -- the map
    fits only the fractions at them, at the section's weighting, on its own jumps; no Best, no check of a count set
    by hand (the count is the section's). A split by hand uses no fit and ignores it.

    ``precision`` (``(c_nm, p, source)``; 6 Nov 2026, `OWN_PRECISION_KEY`): the map's OWN fit Per spot -- every fit
    (Best's, the check beside a count set by hand, the map's) by maximum likelihood with each jump's own localisation
    error, and every jump labelled under it (`maps.per_spot_steps`). Ignored beside a section's fit (that fit's own
    calibration decides) and by a split by hand (nothing fitted).

    **`spatial_sqd_map` is the seam, NOT `compute_condition`.** That function
    exists for exactly this caller -- its own docstring calls itself "the
    catalogue target: one projected table in, jump-density grids out ... it
    takes only what a panel can supply (a table and scalar controls), fits
    internally, and returns the same dict". `compute_condition` is the
    SCRIPT's front door: it re-reads the project off disk, and a GUI that
    went through it would be re-doing minutes of work the workspace has
    already done.

    **The first version DID go through it, and the bug that found was not
    the wasted time.** `compute_condition` re-runs the raw loader, which
    needs a day map, and the nearest one to hand -- `project.day_maps` --
    stops being the loader's own mapping the moment the onboarding group
    board is confirmed: `project.regroup` overwrites it with the BOARD's
    mapping, from which a held-out or unplaced series is correctly absent.
    The raw loader reads every movie on disk, filtered or not, and refuses a
    mapping with any gap, so a project whose `series009` has no segmented
    cells failed with "day mapping does not cover these movies:
    ['series009']". `project.py`'s own `day_maps` field and
    `onboarding.window._regrouped` both name that trap in as many words.

    **Reading the loaded table is not merely the fix, it is the correct
    answer.** Filling the gap instead would have put series the user HELD
    OUT on the Confirm screen back into this one map while every other page
    in the workspace excludes them -- two datasets under one name, which is
    the failure this project keeps recording. `project.projected` is the
    same cached table the rest of the workspace reads.

    What this file still adds on top of `spatial_sqd_map`, because
    `spatialsqd` does not do it: the `n_min` mask and the count-weighted
    smoothing -- see `finish_spatial_panels`.

    `days` holds the same maps for each replicate day, drawn under the
    condition's own fit -- one rule for every day, so a difference between day
    rows is where that day's jumps fell, not a different rule. The days
    partition the condition's jumps.
    """
    import numpy as np

    from .spatialsqd import ALL_JUMPS_LABEL, local_half_width, maps_from_projected, maps_from_ranges, spatial_sqd_map

    if split_from not in SPLIT_FROM:
        raise ValueError(f"split_from must be one of {SPLIT_FROM}, got {split_from!r}")
    _check_neff(neff_min, neff_unit)
    if weighting not in SQD_FIT_WEIGHTING:
        raise ValueError(f"weighting must be one of {WEIGHTINGS}, got {weighting!r}")
    fit_weighting = SQD_FIT_WEIGHTING[weighting]
    fixed = section_fit if section_fit is not None and split != "manual" else None
    own = _own_precision(precision, fixed, split)
    if own is not None and map_lag(lag_frames) == "1-4":
        raise ValueError(OWN_PRECISION_LAG_REFUSAL)
    if fixed is not None:
        # THE SECTION'S D, its count and its fractions' weighting (`map_fit_source`)
        n_pop = int(fixed["n_pop"])
        fit_weighting = str(fixed.get("weighting") or fit_weighting)
    # THE CUTOFF FIRST, then the fit on the table `split_from` names: "all" learns
    # the short/long rule from every track (as the bubble plot does) and the cutoff
    # only removes what is drawn; "kept" learns it from the surviving tracks.
    full = project.projected(condition)
    kept, cutoff_stats = apply_track_cutoff(full, max_locs, cutoff_mode)
    fit_table = full if split_from == "all" else kept
    # THE LENGTH CLASS IS DRAWN, NEVER FITTED: the populations are the condition's,
    # so a short-cell and a long-cell map differ only in where the jumps fell.
    projected, class_stats = apply_cell_class(kept, cell_class, class_cuts_um)
    # A SPLIT BY HAND (`SPLITS`): jump-length ranges in nm, one map each, nothing fitted
    manual = check_ranges("sqd", ranges) if split == "manual" else None
    ranges_um = None if manual is None else [(lo / 1000.0, hi / 1000.0) for lo, hi in manual]
    # BEST: the count BIC chooses for THIS condition, on the table the split is learned from (`_best_sqd_count`)
    choice, bic, bic_forced = n_pop, None, None
    if fixed is not None:
        pass                    # the section's count: its own check of a count set by hand is the section's
    elif manual is None and n_pop == BEST:
        best = _best_sqd_selection(fit_table, lag_frames, fit_weighting, **_calibrated(own))
        n_pop, bic = best["n_pop"], best["bic"]
    elif manual is None:
        # Best's own maximum-likelihood fits, the set count's too (the map's least squares is not what Best scores)
        bic_forced = _forced_selection("sqd", n_pop, lambda counts: _best_sqd_selection(fit_table, lag_frames,
                                                                                      fit_weighting, counts,
                                                                                      **_calibrated(own)))
    out = spatial_sqd_map(
        fit_table, n_pop=int(n_pop) if manual is None else len(manual), assignment=assignment,
        step_position=step_position, ranges_um=ranges_um,
        mirrored=mirrored,
        # ONE bin size for both axes. The choice is a single resolution
        # claim -- read against the camera pixel and the localisation
        # precision -- so splitting it into two numbers would invite two
        # answers to one question; `spatial_sqd_map` rounds each axis to an
        # even count from it, since folding pairs each bin with its
        # reflection.
        bin_long_nm=float(bin_nm), bin_short_nm=float(bin_nm),
        # `smooth_bins` is carried on the config for the record only --
        # `maps_from_projected` does no smoothing, so the kernel below is the
        # one that runs, and it is set per axis from a PHYSICAL bandwidth.
        smooth_bins=1.0,
        cell_length_um=cell_length_um, cell_width_um=cell_width_um,
        method="ecdf_lsq", weighting=fit_weighting, meshes=project.meshes,
        long_axis=long_axis, lag_frames=map_lag(lag_frames), all_jumps=True,
        fixed_D=None if fixed is None else fixed["D"],
        calibration=held_calibration(fixed) if fixed is not None else own)
    cfg = out["config"]
    # Redraw under the SAME fit when the drawn table or the map weighting differs
    # from what `spatial_sqd_map` binned. (Its own `weighting` argument is the FIT's
    # weighting, `fit_weighting`; it binned every jump once, whatever the fit.)
    def draw(table, **extra):
        """This run's maps of ``table`` under the condition's own rule: its fit, or its ranges."""
        if manual is not None:
            return maps_from_ranges(table, ranges_um, cfg, step_position=step_position, mirrored=mirrored,
                                    meshes=project.meshes, weighting=weighting, lag_frames=map_lag(lag_frames),
                                    all_jumps=True, **extra)
        return maps_from_projected(table, out["fit"], cfg, assignment=assignment, step_position=step_position,
                                   mirrored=mirrored, meshes=project.meshes, weighting=weighting,
                                   lag_frames=map_lag(lag_frames), all_jumps=True, **extra)

    if projected is not fit_table or weighting != "steps":
        out = {**out, **draw(projected)}

    # BY CONSTRUCTION, never by matching the label text. `population_labels`
    # builds its keys from `argsort(D)`, so the slowest population's key is
    # `labels[order[0]]` whatever it is called -- and this project has
    # already shipped a figure whose panels were swapped because they were
    # taken by sort order instead of by key.
    if manual is not None:
        # a range's map IS its kind (`spatialsqd.RANGE_KINDS`), lowest range first
        kinds = population_kinds("sqd", len(manual), manual)
        key_for = {kind: kind for kind in kinds}
    else:
        order = np.argsort(np.asarray(out["D"], dtype=float))
        kinds = population_kinds("sqd", n_pop)
        key_for = {kind: out["labels"][order[i]] for i, kind in enumerate(kinds)}
    # EVERY JUMP, beside a split (`spatialsqd.ALL_JUMPS_LABEL`): first, as the grid's first column
    if ALL_JUMPS_LABEL in out["labels"]:
        key_for = {ALL_JUMPS_KIND: ALL_JUMPS_LABEL, **key_for}
    kind_of = {label: kind for kind, label in key_for.items()}

    xc, yc = cfg.centres()
    xx, yy = np.meshgrid(xc, yc)
    inside = np.abs(yy) <= local_half_width(xx, cfg)

    bin_long_um = cfg.cell_length_um / cfg.n_bins_long
    bin_short_um = cfg.cell_width_um / cfg.n_bins_short
    stamp = sigma_stamp(project)
    h_um = bandwidth_um(stamp["sigma_nm"], bin_long_um, bin_short_um)

    def by_kind(o):
        return _parts(o, key_for)

    panels = finish_spatial_panels(
        **by_kind(out), n_min=n_min, inside=inside, h_um=h_um,
        bin_long_um=bin_long_um, bin_short_um=bin_short_um,
        neff_min=neff_min, neff_unit=neff_unit)

    payload = {
        "method": "sqd",
        "condition": condition,
        "lag_frames": map_lag(lag_frames),
        "step_position": step_position,
        "assignment": assignment,
        "mirrored": bool(mirrored),
        "n_min": int(n_min),
        "n_bins_inside": int(inside.sum()),
        "cell_length_um": cfg.cell_length_um,
        "cell_width_um": cfg.cell_width_um,
        "n_bins_long": cfg.n_bins_long,
        "n_bins_short": cfg.n_bins_short,
        "smooth_bins": cfg.smooth_bins,
        "bin_nm": bin_number(bin_nm),
        # THE BANDWIDTH IN NANOMETRES, and the sigma it is read against.
        # The render spec's "reported on every figure" list asks for the
        # bandwidth in um plus the sigma it is tied to; the caption used to
        # print "smoothing 1 bins", which is meaningless outside this one
        # grid. `smooth_bins` is a BIN COUNT, so h follows the bin -- see
        # `caption_for`, which prints the derivation rather than letting a
        # reader infer that h came from sigma. It did not.
        "h_nm": float(h_um * 1000.0),
        "panel_n_min": PANEL_N_MIN,
        **stamp,
        #: "measured" (this condition's own outlines) or "deposit" -- the
        #: caption prints it, because the two are different pictures of one
        #: quantity and must not be stacked without saying so.
        "frame_source": frame_source,
        # `n_jumps` always counts JUMPS, whatever the weighting, so the unit is
        # the step unit; how the map counted is `weighting`.
        "weight_unit": "weighted steps" if assignment == "soft" and manual is None else "steps",
        "weighting": weighting,
        "fit_weighting": fit_weighting if manual is None else None,
        # THE SECTION'S FIT the D came from (`map_fit_source`), None where the map fitted its own
        "section_fit": None if fixed is None else dict(fixed),
        # THE MAP'S OWN FIT PER SPOT (`OWN_PRECISION_KEY`): the calibration, ``[c_nm, p, source]``
        **_own_precision_stamp(own, precision),
        "split": SPLIT_DEFAULT if manual is None else "manual",
        "ranges": None if manual is None else [list(r) for r in manual],
        "n_outside_ranges": out.get("n_outside_ranges"),
        # what was asked: a count, or Best (the count BIC chose is ``n_pop``)
        "n_pop_choice": BEST if choice == BEST else str(choice),
        # Best's BIC of every count it compared, ``[[n, BIC], ...]``; None for a count set by hand
        "bic": bic,
        **({"bic_forced": bic_forced} if bic_forced else {}),
        "max_locs": None if max_locs is None else int(max_locs),
        "cutoff_mode": cutoff_mode,
        "split_from": split_from,
        "cutoff_stats": cutoff_stats,
        "neff_min": None if neff_min is None else float(neff_min),
        "neff_unit": neff_unit,
        "cell_class": cell_class,
        "class_cuts_um": None if class_cuts_um is None else tuple(class_cuts_um),
        "class_stats": class_stats,
        "long_axis": long_axis,
        # THE CLIFF, carried so the panel can say so BEFORE you read the
        # picture rather than after. At two populations hard assignment is
        # exactly a threshold on jump length, and a population whose best
        # posterior is at or below 0.5 is never the argmax -- its map is
        # blank, and nothing else warns you.
        "max_posterior_slowest": float(out["max_posterior_slowest"]),
        "hard_assignment_possible": bool(out["hard_assignment_possible"]),
        "n_pop": int(n_pop) if manual is None else len(manual),
        #: Under hard assignment, a population that wins no jump draws an empty map.
        "never_most_likely": [kind_of[label] for label in out.get("never_most_likely", [])],
        "D": [float(v) for v in np.asarray(out["D"], dtype=float)],
        "fractions": [float(v) for v in np.asarray(out["fractions"],
                                                   dtype=float).ravel()],
        "panels": panels,
    }
    zeros = getattr(out.get("fit"), "zero_jumps", None)
    if zeros is not None:
        # exact-zero jumps met by the map's fit, and what was done (`sqd.zero_jumps`): said with the map's warnings
        payload["zero_jumps"] = zeros.to_dict()
    # The SAME `key_for`, from the CONDITION fit -- never recomputed per day.
    payload["days"] = _day_payloads(
        projected, {k: v for k, v in payload.items() if k != "panels"},
        lambda sub: draw(sub, with_steps=True),
        by_kind,
        n_min=n_min, inside=inside, h_um=h_um,
        bin_long_um=bin_long_um, bin_short_um=bin_short_um,
        kind_keys=key_for, cfg=cfg, neff_min=neff_min, neff_unit=neff_unit)
    # Added AFTER the days, so it is not copied into every day: the chart's data.
    payload["track_census"] = track_census(full)
    payload["profiles"] = profiles_by_kind(out.get("profiles"), key_for)
    # THE MAP OF EVERY JUMP COUNTS WHOLE JUMPS, each at weight 1, so its line says "steps" under soft assignment too
    for one in (payload, *payload["days"]):
        if ALL_JUMPS_KIND in one["panels"] and manual is None and assignment == "soft":
            one["panels"][ALL_JUMPS_KIND]["count_unit"] = "steps"
    return payload


def compute_spatial_map_appd(project, condition: str, *, mirrored: bool,
                             n_min: int, bin_nm: float, cell_length_um: float,
                             cell_width_um: float, frame_source: str,
                             weighting: str = "steps", max_locs=None,
                             cutoff_mode: str = "tracks", n_pop: int = 2,
                             neff_min: float | None = None,
                             neff_unit: str = "tracks",
                             cell_class: str = "all", class_cuts_um=None,
                             long_axis: str = "normalised",
                             split: str = SPLIT_DEFAULT, ranges=None, section_fit=None, precision=None) -> dict:
    """One condition's apparent-diffusion localisation-density maps, from the
    project ALREADY LOADED -- the appD analogue of `compute_spatial_map_sqd`.

    Deliberately the SAME signature shape minus `step_position`/`assignment`
    (appD has neither -- see `smtracker.spatialappd`'s own module
    docstring), so `compute_spatial_map`'s dispatch below can hand either
    function exactly the config keys it takes and no others.

    `days` holds the same maps for each replicate day, drawn under the
    condition's own fit -- one rule for every day, so a difference between day
    rows is where that day's jumps fell, not a different rule. The days
    partition the condition's jumps.

    ``section_fit`` (4 Nov 2026, `map_fit_source`): the Apparent Diffusion section's fit of this condition, its D
    HELD -- the split is `spatialappd.split_at_D` (the section's estimator, fractions only) on this condition's
    tracks; no Best, no check of a count set by hand. A split by hand ignores it.

    ``precision`` (``(c_nm, p, source)``; 6 Nov 2026, `OWN_PRECISION_KEY`): the map's OWN fit Per spot -- each
    sub-track's own noise offset in the gamma mixture and in Best's (`spatialappd.fit_appd_split`), the cuts at the
    median offset. Ignored beside a section's fit and by a split by hand, as SQD's.
    """
    import numpy as np

    from .spatialappd import (
        fit_appd_split,
        local_half_width,
        maps_from_projected_appd,
        spatial_appd_map,
        split_at_D,
    )

    # THE SPLIT IS FITTED: the appD gamma mixture on every track of the condition
    # (before any cutoff, as the SQD fit learns from all tracks by default), cut where
    # its populations cross -- the old program's own rule. Each track is then labelled
    # by its mean apparent D.
    _check_neff(neff_min, neff_unit)
    full = project.projected(condition)
    kept, cutoff_stats = apply_track_cutoff(full, max_locs, cutoff_mode)
    projected, class_stats = apply_cell_class(kept, cell_class, class_cuts_um)
    # A SPLIT BY HAND (`SPLITS`): apparent-D ranges in µm²/s, one map each, nothing fitted
    manual = check_ranges("appd", ranges) if split == "manual" else None
    # BEST: the count BIC chooses for THIS condition (`_best_appd_count`), on every track as the split is
    fixed = section_fit if section_fit is not None and manual is None else None
    own = _own_precision(precision, fixed, split)
    choice, bic_forced = n_pop, None
    if fixed is not None:
        # THE SECTION'S D (`map_fit_source`): its count and its estimator, only the fractions fitted here
        n_pop = choice = int(fixed["n_pop"])
        split = split_at_D(full, fixed["D"], min_steps=int(fixed.get("min_steps", 4)),
                           calibration=held_calibration(fixed))
    else:
        if manual is None and n_pop == BEST:
            n_pop = _best_appd_count(full, **_calibrated(own))
        split = fit_appd_split(full, n_pop=int(n_pop), **_calibrated(own)) if manual is None else None
    if manual is None and choice != BEST and fixed is None:
        # the map's own gamma mixture is Best's fit at its count: its BIC is Best's score there
        bic_forced = _forced_selection("appd", n_pop, lambda counts: _appd_bic_rows(full, counts, **_calibrated(own)),
                                       split["bic"])
    out = spatial_appd_map(
        projected, bin_long_nm=float(bin_nm), bin_short_nm=float(bin_nm),
        smooth_bins=1.0, cell_length_um=cell_length_um,
        cell_width_um=cell_width_um, mirrored=mirrored, split=split,
        long_axis=long_axis, ranges=manual, all_tracks=True)
    cfg = out["config"]
    if weighting != "steps":
        out = {**out, **maps_from_projected_appd(
            projected, cfg, mirrored=mirrored, weighting=weighting, split=split, ranges=manual, all_tracks=True)}
    names = kinds_for("appd", n_pop, manual)

    xc, yc = cfg.centres()
    xx, yy = np.meshgrid(xc, yc)
    inside = np.abs(yy) <= local_half_width(xx, cfg)

    bin_long_um = cfg.cell_length_um / cfg.n_bins_long
    bin_short_um = cfg.cell_width_um / cfg.n_bins_short
    stamp = sigma_stamp(project)
    h_um = bandwidth_um(stamp["sigma_nm"], bin_long_um, bin_short_um)

    panels = finish_spatial_panels(
        **_parts(out), n_min=n_min, inside=inside, h_um=h_um,
        bin_long_um=bin_long_um, bin_short_um=bin_short_um,
        neff_min=neff_min, neff_unit=neff_unit)

    payload = {
        "method": "appd",
        "condition": condition,
        "mirrored": bool(mirrored),
        "n_min": int(n_min),
        "n_bins_inside": int(inside.sum()),
        "cell_length_um": cfg.cell_length_um,
        "cell_width_um": cfg.cell_width_um,
        "n_bins_long": cfg.n_bins_long,
        "n_bins_short": cfg.n_bins_short,
        "smooth_bins": cfg.smooth_bins,
        "bin_nm": bin_number(bin_nm),
        "h_nm": float(h_um * 1000.0),
        "panel_n_min": PANEL_N_MIN,
        **stamp,
        "frame_source": frame_source,
        "weight_unit": "localisations",
        "weighting": weighting,
        "max_locs": None if max_locs is None else int(max_locs),
        "cutoff_mode": cutoff_mode,
        "cutoff_stats": cutoff_stats,
        "neff_min": None if neff_min is None else float(neff_min),
        "neff_unit": neff_unit,
        "cell_class": cell_class,
        "class_cuts_um": None if class_cuts_um is None else tuple(class_cuts_um),
        "class_stats": class_stats,
        "long_axis": long_axis,
        "n_pop": int(n_pop) if manual is None else len(manual),
        "appd_D": [] if manual is not None else split["D"],
        "appd_fractions": [] if manual is not None else split["fractions"],
        "crossings_um2_s": None if manual is not None else split["cuts"],
        #: A population that is the most likely at no D gets an empty map, named here.
        "never_most_likely": [] if manual is not None else [population_kinds("appd", n_pop)[k]
                                                            for k in split["never_most_likely"]],
        "split": SPLIT_DEFAULT if manual is None else "manual",
        "ranges": None if manual is None else [list(r) for r in manual],
        "n_outside_ranges": out.get("n_outside_tracks") if manual is not None else None,
        "n_pop_choice": BEST if choice == BEST else str(choice),
        **({"bic_forced": bic_forced} if bic_forced else {}),
        # THE SECTION'S FIT the D came from (`map_fit_source`), None where the map fitted its own
        "section_fit": None if fixed is None else dict(fixed),
        **_own_precision_stamp(own, precision),
        # exact-zero sub-tracks met by the map's fit, and what was done (`appd_gamma.zero_tracks`): said with the
        # map's warnings, as an SQD map's exact-zero jumps are
        **({"zero_jumps": split["zero_jumps"]} if manual is None and split.get("zero_jumps") else {}),
        "panels": panels,
    }
    payload["days"] = _day_payloads(
        projected, {k: v for k, v in payload.items() if k != "panels"},
        lambda sub: maps_from_projected_appd(
            sub, cfg, mirrored=mirrored, weighting=weighting, with_steps=True,
            split=split, ranges=manual, all_tracks=True),
        _parts,
        n_min=n_min, inside=inside, h_um=h_um,
        bin_long_um=bin_long_um, bin_short_um=bin_short_um,
        kind_keys={k: k for k in names}, cfg=cfg,
        neff_min=neff_min, neff_unit=neff_unit)
    # Added AFTER the days, so it is not copied into every day: the chart's data.
    payload["track_census"] = track_census(full)
    payload["profiles"] = profiles_by_kind(out.get("profiles"), {k: k for k in names})
    return payload


def compute_spatial_map_hmm(project, condition: str, *, step_position: str,
                            assignment: str, mirrored: bool, n_min: int,
                            bin_nm: float, cell_length_um: float,
                            cell_width_um: float, frame_source: str,
                            weighting: str = "steps", max_locs=None,
                            cutoff_mode: str = "tracks",
                            split_from: str = "all",
                            neff_min: float | None = None,
                            neff_unit: str = "tracks",
                             cell_class: str = "all", class_cuts_um=None,
                             long_axis: str = "normalised", n_pop=2, section_fit=None, precision=None,
                             hmm_show: str = "steps") -> dict:
    """One condition's HMM-state step-density maps, from the project
    ALREADY LOADED -- the HMM analogue of `compute_spatial_map_sqd`, fixed
    at two states (see `smtracker.spatialhmm`'s own module docstring on
    why). Same signature shape as SQD, not appD: HMM is ALSO per-step with
    a posterior to soften.

    Fits fresh, exactly as `compute_spatial_map_sqd` fits `fit_sqd` fresh
    on every compute -- confirmed this session to cost the same order of
    magnitude (~1-2 s at this project's scale) as SQD's own fit, and this
    app already fits an HMM synchronously per condition for the heavier
    Kinetics/SuperPlots cards through the identical process-pool pattern.

    A condition whose fitted chain has no within-track switching is drawn
    (ruled 4 Nov 2026; it was refused while the decode's prior was the
    stationary distribution): each track is decoded in one state for its whole
    length, the payload's ``no_switching`` is set and the map says so
    (`no_switching_warning`).

    `days` holds the same maps for each replicate day, drawn under the
    condition's own fit -- one rule for every day, so a difference between day
    rows is where that day's jumps fell, not a different rule. The days
    partition the condition's jumps.

    ``section_fit`` (4 Nov 2026, `map_fit_source`): the HMM section's fit of this condition, its states' D HELD and
    its sigma -- only the transitions are fitted here (`hmm.fit_hmm`'s ``fixed_D``); no Best, no check of a count set
    by hand.

    ``precision`` (``(c_nm, p, source)``; 6 Nov 2026, `OWN_PRECISION_KEY`): the map's OWN fit Per spot -- each step's
    own noise in the fit, in Best's and in the decoding (`spatialhmm.spatial_hmm_map`'s ``calibration``). Ignored
    beside a section's fit (its own sigma decides).
    """
    import numpy as np

    from .spatialhmm import local_half_width, maps_from_projected_hmm, spatial_hmm_map

    if split_from not in SPLIT_FROM:
        raise ValueError(f"split_from must be one of {SPLIT_FROM}, got {split_from!r}")
    _check_neff(neff_min, neff_unit)
    full = project.projected(condition)
    kept, cutoff_stats = apply_track_cutoff(full, max_locs, cutoff_mode)
    fit_table = full if split_from == "all" else kept
    projected, class_stats = apply_cell_class(kept, cell_class, class_cuts_um)
    tau_s = project.params.tau_s
    fixed = section_fit
    own = _own_precision(precision, fixed, SPLIT_DEFAULT)
    sigma_um = 0.0
    if fixed is not None:
        # THE SECTION'S STATES (`map_fit_source`): their D held, its sigma, its count
        from .hmm_view import _sigma_um

        n_pop = int(fixed["n_pop"])
        if held_calibration(fixed) is None:
            sigma_um, _words = _sigma_um(project, str(fixed.get("sigma", "off")))
    # THE STATE COUNT (3 Nov 2026: 1, 2 or 3, or BIC's on the fit's own steps -- the SQD map's choices)
    choice = n_pop
    best = None
    if n_pop == BEST:
        best = _best_hmm_selection(fit_table, tau_s, **_calibrated(own))
        n_pop = best["n_pop"]
    n_pop = int(check_n_pop("hmm", n_pop))
    names = kinds_for("hmm", n_pop)
    out = spatial_hmm_map(
        fit_table, tau_s=tau_s, bin_long_nm=float(bin_nm),
        bin_short_nm=float(bin_nm), smooth_bins=1.0,
        cell_length_um=cell_length_um, cell_width_um=cell_width_um,
        assignment=assignment, step_position=step_position, mirrored=mirrored,
        meshes=project.meshes, long_axis=long_axis, n_states=n_pop, all_jumps=True,
        fixed_D=None if fixed is None else fixed["D"], localization_sigma_um=sigma_um,
        calibration=held_calibration(fixed) if fixed is not None else own)
    # the map's own fit is Best's at its count (the same `fit_hmm` call on the same steps): its BIC is Best's score
    bic_forced = None if choice == BEST or fixed is not None else _forced_selection(
        "hmm", n_pop, lambda counts: _best_hmm_selection(fit_table, tau_s, counts, **_calibrated(own)),
        out["fit"].bic)
    cfg = out["config"]
    if projected is not fit_table or weighting != "steps":
        out = {**out, **maps_from_projected_hmm(
            projected, out["fit"], cfg, assignment=assignment,
            step_position=step_position, mirrored=mirrored,
            meshes=project.meshes, weighting=weighting, all_jumps=True)}

    xc, yc = cfg.centres()
    xx, yy = np.meshgrid(xc, yc)
    inside = np.abs(yy) <= local_half_width(xx, cfg)

    bin_long_um = cfg.cell_length_um / cfg.n_bins_long
    bin_short_um = cfg.cell_width_um / cfg.n_bins_short
    stamp = sigma_stamp(project)
    h_um = bandwidth_um(stamp["sigma_nm"], bin_long_um, bin_short_um)

    turning = None
    if hmm_show in ("angles", "turning"):
        # TURNING BACK (22 Nov 2026, `spatialangles`): the same states and fit, each pair of steps at its turn, the
        # share turning back ("angles", the data alone) or that minus chance at the same pairs ("turning", Beyond
        # chance); the step maps are not drawn
        turning = _turning_maps(project, condition, projected, out["fit"], cfg, names, full.attrs.get("geometry"),
                                n_pop=n_pop, mirrored=mirrored, weighting=weighting, chance=hmm_show == "turning")
        panels = finish_turning_panels(
            **_turning_parts(turning["maps"]), n_min=n_min, inside=inside, h_um=h_um,
            bin_long_um=bin_long_um, bin_short_um=bin_short_um, neff_min=neff_min, neff_unit=neff_unit)
    elif hmm_show != "steps":
        raise ValueError(f"hmm_show must be 'steps', 'angles' or 'turning', got {hmm_show!r}")
    else:
        panels = finish_spatial_panels(
            **_parts(out), n_min=n_min, inside=inside, h_um=h_um,
            bin_long_um=bin_long_um, bin_short_um=bin_short_um,
            neff_min=neff_min, neff_unit=neff_unit)
    # A STATE NO STEP WON (4 Nov 2026, found by the overnight tester): under hard assignment its map is empty by
    # construction, and the grid said only "n = 0 -- too sparse to map" (`empty_state_warning` says why)
    never = ([k for k in population_kinds("hmm", n_pop) if float((panels.get(k) or {}).get("n_jumps") or 0.0) == 0.0]
             if assignment == "hard" and n_pop > 1 and turning is None else [])

    payload = {
        "method": "hmm",
        "condition": condition,
        "step_position": step_position,
        "assignment": assignment,
        "mirrored": bool(mirrored),
        "n_min": int(n_min),
        "n_bins_inside": int(inside.sum()),
        "cell_length_um": cfg.cell_length_um,
        "cell_width_um": cfg.cell_width_um,
        "n_bins_long": cfg.n_bins_long,
        "n_bins_short": cfg.n_bins_short,
        "smooth_bins": cfg.smooth_bins,
        "bin_nm": bin_number(bin_nm),
        "h_nm": float(h_um * 1000.0),
        "panel_n_min": PANEL_N_MIN,
        **stamp,
        "frame_source": frame_source,
        "weight_unit": "weighted steps" if assignment == "soft" else "steps",
        "weighting": weighting,
        "max_locs": None if max_locs is None else int(max_locs),
        "cutoff_mode": cutoff_mode,
        "split_from": split_from,
        "cutoff_stats": cutoff_stats,
        "neff_min": None if neff_min is None else float(neff_min),
        "neff_unit": neff_unit,
        "cell_class": cell_class,
        "class_cuts_um": None if class_cuts_um is None else tuple(class_cuts_um),
        "class_stats": class_stats,
        "long_axis": long_axis,
        "D": [float(v) for v in np.asarray(out["D"], dtype=float)],
        "occupancy": [float(v) for v in np.asarray(out["occupancy"], dtype=float)],
        "fraction_kept": float(out["fraction_kept"]),
        "n_pop": int(n_pop),
        "n_pop_choice": BEST if choice == BEST else str(choice),
        **({"bic": best["bic"]} if best is not None else {}),
        **({"bic_forced": bic_forced} if bic_forced else {}),
        "never_most_likely": never,
        # NO SWITCHING WITHIN TRACKS (`HMMFit.no_switching`): each track drawn in one state, said by the map
        "no_switching": bool(out["fit"].no_switching),
        # THE SECTION'S FIT the D came from (`map_fit_source`), None where the map fitted its own
        "section_fit": None if fixed is None else dict(fixed),
        **_own_precision_stamp(own, precision),
        "panels": panels,
    }
    if turning is not None:
        payload.update({"show": hmm_show, "weight_unit": "weighted pairs", "turning": turning["words"]})
        pairs = turning["pairs"]
        payload["days"] = _day_payloads(
            projected, {k: v for k, v in payload.items() if k != "panels"},
            lambda sub: _turning_day(pairs, sub, cfg, projected, names, n_pop, mirrored, weighting),
            _turning_parts, finish=finish_turning_panels,
            n_min=n_min, inside=inside, h_um=h_um, bin_long_um=bin_long_um, bin_short_um=bin_short_um,
            neff_min=neff_min, neff_unit=neff_unit)
        payload["track_census"] = track_census(full)
        payload["profiles"] = None
        return payload
    payload["days"] = _day_payloads(
        projected, {k: v for k, v in payload.items() if k != "panels"},
        lambda sub: maps_from_projected_hmm(
            sub, out["fit"], cfg, assignment=assignment,
            step_position=step_position, mirrored=mirrored, meshes=project.meshes,
            weighting=weighting, with_steps=True, all_jumps=True),
        _parts,
        n_min=n_min, inside=inside, h_um=h_um,
        bin_long_um=bin_long_um, bin_short_um=bin_short_um,
        kind_keys={k: k for k in names}, cfg=cfg,
        neff_min=neff_min, neff_unit=neff_unit)
    # Added AFTER the days, so it is not copied into every day: the chart's data.
    payload["track_census"] = track_census(full)
    payload["profiles"] = profiles_by_kind(out.get("profiles"), {k: k for k in names})
    return payload


def _turning_maps(project, condition: str, projected, fit, cfg, names, geometry, *, n_pop: int, mirrored: bool,
                  weighting: str, chance: bool = True) -> dict:
    """The HMM map's Turning back for one condition (`spatialangles`): every step decoded under ``fit``, the pairs at
    their turns (`hmm_angles.pair_table`), each spot's localisation error (`hmm_angles.noise_for`), the chance at the
    same pairs (`hmm_angles.chance_by_state`, the Jump angles page's own) and the grids. Returns the maps, the pairs
    with their windows (the replicate rows reuse them, never a new chance) and the words a caption needs."""
    from .hmm import decode_states
    from .hmm_angles import cell_words, chance_by_state, exposure_for, linking_for, noise_for, pair_table, state_names
    from .spatialangles import maps_from_pairs, window_columns

    decoded = decode_states(projected, fit)
    pairs = pair_table(projected, decoded, n_pop)
    walked, D_free, noise_words, link_words, exposure_words, cell_text = None, [], "", "", "", ""
    if chance:
        sigma, noise_words = noise_for(project, condition, pairs, fit)
        link_um, link_words = linking_for(project, condition)
        share, exposure_words = exposure_for(project, condition)
        walked, D_free, cell = chance_by_state(pairs, geometry or {}, sigma, fit, link_um=link_um,
                                               exposure=1.0 if share is None else share)
        cell_text = cell_words(cell, state_names(n_pop))
    pairs = window_columns(pairs, walked, n_pop)
    maps = maps_from_pairs(pairs, list(names), cfg, projected, n_states=n_pop, mirrored=mirrored, weighting=weighting)
    words = {"n_pairs": int(len(pairs)), "switch_share": float(pairs["w_switch"].sum() / max(len(pairs), 1)),
             "chance": bool(chance), "D_chance": [float(d) for d in D_free], "noise_words": noise_words,
             "link_words": link_words, "exposure_words": exposure_words, "cell_words": cell_text}
    return {"maps": maps, "pairs": pairs, "words": words}


def _turning_day(pairs, day_rows, cfg, projected, names, n_pop: int, mirrored: bool, weighting: str) -> dict:
    """One replicate's Turning back grids: its own pairs, with the chance already made for the condition."""
    from .spatialangles import maps_from_pairs

    days = set(day_rows["day"].dropna().unique().tolist())
    return maps_from_pairs(pairs[pairs["day"].isin(days)], list(names), cfg, projected, n_states=n_pop,
                           mirrored=mirrored, weighting=weighting)


#: The chance rate's free-Brownian replicates: two keep a map run near the detector's own
#: cost (5.4 s on the reference project's 2005 tracks, against 1.8 s for one detection).
SLA_NULL_REPLICATES = 2
#: The confinement radius's range in the dialog, in nm.
SLA_RADIUS_RANGE_NM = (20, 600)
#: The minimum dwell's range, in steps (v2's list ran 1-50).
SLA_DWELL_RANGE = (1, 30)


def sla_null_D(full) -> float:
    """The free molecule the chance rate simulates: the condition's own fast SQD state
    (two populations, step-weighted, the fast ECDF) -- `confinement_null`'s rule, the D
    "must come from the analysis", never from a default."""
    from .observables import squared_displacements
    from .sqd import fit_sqd

    fit = fit_sqd(squared_displacements(full), n_pop=2, ecdf_max_points=2000)
    return float(fit.D[-1])


def sla_chance(projected, *, radius_um: float, min_steps: int, D_um2_s: float,
               n_replicates: int = SLA_NULL_REPLICATES, seed: int = 0) -> dict:
    """What the identical detector finds on free Brownian tracks matched to these tracks'
    lengths (`simulate.matched_brownian_null`): ``{set: (mean, sd)}`` for the three
    shares the map reports -- tracks free, tracks holding a dwell (v2's "mixed"), and
    localisations dwelling."""
    import numpy as np

    from .confinement import confinement_events
    from .simulate import matched_brownian_null

    draws = {"free": [], "mixed": [], "dwelling": []}
    for replicate in range(int(n_replicates)):
        null = matched_brownian_null(projected, D_um2_s=D_um2_s, sigma_um=0.0,
                                     seed=seed + replicate)
        events = confinement_events(null, radius_um=radius_um, min_steps=min_steps)
        draws["free"].append(events.attrs["fraction_free"])
        draws["mixed"].append(1.0 - events.attrs["fraction_free"])
        draws["dwelling"].append(events.attrs["confined_segment_fraction"])
    return {k: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"))
            for k, v in draws.items()}


def compute_spatial_map_sla(project, condition: str, *, mirrored: bool, n_min: int,
                            bin_nm: float, cell_length_um: float, cell_width_um: float,
                            frame_source: str, radius_nm: float, min_steps: int,
                            weighting: str = "steps", max_locs=None,
                            cutoff_mode: str = "tracks", neff_min: float | None = None,
                            neff_unit: str = "tracks", cell_class: str = "all",
                            class_cuts_um=None, long_axis: str = "normalised") -> dict:
    """One condition's SLA maps -- v2's Confined Map (`smtracker.spatialsla`) -- from the
    project ALREADY LOADED, in the apparent-diffusion map's payload shape plus each
    panel's ``tracks`` (polylines for the Tracks view) and ``ink`` (v2's colour).

    The detector runs ONCE, on the drawn tracks; the day pages draw from those labels, so
    nothing is detected twice. A track's class depends on that track alone, so the cutoff
    and the cell class cannot change it -- they only choose which tracks are drawn.

    THE CHANCE RATE IS PART OF THE PAYLOAD, never optional: the same detector on free
    Brownian tracks at the condition's own fast D (`sla_null_D`), matched to the drawn
    tracks' lengths. A share quoted without it is uninterpretable (`confinement`'s head).
    """
    import numpy as np

    from .spatialappd import local_half_width
    from .spatialsla import SLA_COLOURS, SLA_SETS, maps_from_labelled_sla, spatial_sla_map
    from .spatialsla import track_polylines

    _check_neff(neff_min, neff_unit)
    lo, hi = SLA_RADIUS_RANGE_NM
    if not lo <= float(radius_nm) <= hi:
        raise ValueError(f"the confinement radius must be {lo}-{hi} nm, got {radius_nm}")
    if not SLA_DWELL_RANGE[0] <= int(min_steps) <= SLA_DWELL_RANGE[1]:
        raise ValueError(f"the minimum dwell must be {SLA_DWELL_RANGE[0]}-"
                         f"{SLA_DWELL_RANGE[1]} steps, got {min_steps}")
    radius_um = float(radius_nm) / 1000.0
    full = project.projected(condition)
    kept, cutoff_stats = apply_track_cutoff(full, max_locs, cutoff_mode)
    projected, class_stats = apply_cell_class(kept, cell_class, class_cuts_um)
    out = spatial_sla_map(
        projected, bin_long_nm=float(bin_nm), bin_short_nm=float(bin_nm), smooth_bins=1.0,
        cell_length_um=cell_length_um, cell_width_um=cell_width_um,
        radius_um=radius_um, min_steps=int(min_steps), mirrored=mirrored,
        long_axis=long_axis)
    cfg, labelled = out["config"], out["labelled"]
    if weighting != "steps":
        out = {**out, **maps_from_labelled_sla(labelled, cfg, mirrored=mirrored,
                                               weighting=weighting, projected=projected)}

    xc, yc = cfg.centres()
    xx, yy = np.meshgrid(xc, yc)
    inside = np.abs(yy) <= local_half_width(xx, cfg)
    bin_long_um = cfg.cell_length_um / cfg.n_bins_long
    bin_short_um = cfg.cell_width_um / cfg.n_bins_short
    stamp = sigma_stamp(project)
    h_um = bandwidth_um(stamp["sigma_nm"], bin_long_um, bin_short_um)
    panels = finish_spatial_panels(
        **_parts(out), n_min=n_min, inside=inside, h_um=h_um,
        bin_long_um=bin_long_um, bin_short_um=bin_short_um,
        neff_min=neff_min, neff_unit=neff_unit)
    for kind in SLA_SETS:
        panels[kind]["tracks"] = out["tracks"][kind]
        panels[kind]["ink"] = SLA_COLOURS[kind]

    attrs = labelled.attrs["confinement"]
    n_tracks = max(1, int(attrs["n_tracks"]))
    observed = {"free": float(attrs["fraction_free"]),
                "mixed": 1.0 - float(attrs["fraction_free"]),
                "dwelling": float(attrs["confined_segment_fraction"])}
    null_D = sla_null_D(full)
    chance = sla_chance(projected, radius_um=radius_um, min_steps=int(min_steps),
                        D_um2_s=null_D)
    payload = {
        "method": "sla",
        "condition": condition,
        "mirrored": bool(mirrored),
        "n_min": int(n_min),
        "n_bins_inside": int(inside.sum()),
        "cell_length_um": cfg.cell_length_um,
        "cell_width_um": cfg.cell_width_um,
        "n_bins_long": cfg.n_bins_long,
        "n_bins_short": cfg.n_bins_short,
        "smooth_bins": cfg.smooth_bins,
        "bin_nm": bin_number(bin_nm),
        "h_nm": float(h_um * 1000.0),
        "panel_n_min": PANEL_N_MIN,
        **stamp,
        "frame_source": frame_source,
        "weight_unit": "localisations",
        "weighting": weighting,
        "max_locs": None if max_locs is None else int(max_locs),
        "cutoff_mode": cutoff_mode,
        "cutoff_stats": cutoff_stats,
        "neff_min": None if neff_min is None else float(neff_min),
        "neff_unit": neff_unit,
        "cell_class": cell_class,
        "class_cuts_um": None if class_cuts_um is None else tuple(class_cuts_um),
        "class_stats": class_stats,
        "long_axis": long_axis,
        "radius_nm": float(radius_nm),
        "min_steps": int(min_steps),
        "sla_n_tracks": n_tracks,
        "sla_observed": observed,
        "sla_chance": {k: v[0] for k, v in chance.items()},
        "sla_chance_sd": {k: v[1] for k, v in chance.items()},
        "sla_null_D_um2_s": null_D,
        "sla_null_replicates": SLA_NULL_REPLICATES,
        "panels": panels,
    }
    days = _day_payloads(
        labelled, {k: v for k, v in payload.items() if k != "panels"},
        lambda sub: maps_from_labelled_sla(sub, cfg, mirrored=mirrored,
                                           weighting=weighting, with_steps=True,
                                           projected=projected),
        _parts,
        n_min=n_min, inside=inside, h_um=h_um,
        bin_long_um=bin_long_um, bin_short_um=bin_short_um,
        kind_keys={k: k for k in SLA_SETS}, cfg=cfg,
        neff_min=neff_min, neff_unit=neff_unit)
    for entry in days:
        drawn = track_polylines(labelled[labelled["day"] == entry["day"]], cfg,
                                projected=projected)
        for kind in SLA_SETS:
            entry["panels"][kind]["tracks"] = drawn[kind]
            entry["panels"][kind]["ink"] = SLA_COLOURS[kind]
    payload["days"] = days
    payload["track_census"] = track_census(full)
    return payload


def sla_chance_text(payload: dict) -> str:
    """The shares the map rests on, each beside its chance rate: ``"tracks with a dwell
    48 % (chance 10 %) · dwelling localisations 41 % (chance 6 %) · free Brownian at D =
    0.85 µm²/s, 2 replicates"``."""
    observed, chance = payload.get("sla_observed") or {}, payload.get("sla_chance") or {}
    if not observed:
        return ""

    def share(x: float) -> str:
        # A chance rate under one per cent is not zero, and "0%" would say it is.
        return f"{100 * x:.1f}%" if 0 < x < 0.01 else f"{x:.0%}"

    return (f"tracks with a dwell {share(observed['mixed'])} "
            f"(chance {share(chance['mixed'])}) · "
            f"dwelling localisations {share(observed['dwelling'])} "
            f"(chance {share(chance['dwelling'])}) · chance = the same detector on free "
            f"Brownian tracks at D = {payload['sla_null_D_um2_s']:.2f} µm²/s, "
            f"{payload.get('sla_null_replicates', SLA_NULL_REPLICATES)} replicates")


def compute_spatial_map_sda(project, condition: str, *, step_position: str, mirrored: bool,
                            n_min: int, bin_nm: float, cell_length_um: float,
                            cell_width_um: float, frame_source: str,
                            weighting: str = "steps", max_locs=None,
                            cutoff_mode: str = "tracks", neff_min: float | None = None,
                            neff_unit: str = "tracks", cell_class: str = "all",
                            class_cuts_um=None, long_axis: str = "normalised",
                            value_adaptive: bool = True) -> dict:
    """One condition's SDA maps -- v2's Spatial Distribution Analysis (`smtracker.
    spatialsda`): where the localisations are, and the mean speed, apparent D and
    acceleration of the steps in each bin -- from the project ALREADY LOADED.

    Nothing is fitted, so there is no split to learn and no `split_from`; the cutoff and
    the cell class choose which tracks are drawn, as for every method. Speed and D are
    placed by "Jump drawn at" (``step_position``), acceleration at the localisation its
    two steps share. Every value is apparent: the payload carries the floor and the
    ceiling the acquisition sets (`spatialsda.apparent_limits`), and every caption prints
    them. The days are drawn by the same rule (nothing to refit) and partition the
    condition's localisations, steps and step pairs.

    ``value_adaptive`` is the Run's "Adaptive smoothing" (`spatialsda` divergence 9): the
    value maps' kernel widens bin by bin where steps are few. ONE finisher carries it, so
    the condition, every day page and the export (which draws these panels) take the
    same rule."""
    import numpy as np

    from .spatialappd import local_half_width
    from .spatialsda import (
        ADAPTIVE_CAP,
        ADAPTIVE_WIDTHS,
        SDA_COUNTS,
        VALUE_N_MIN,
        apparent_limits,
        maps_from_projected_sda,
        spatial_sda_map,
    )

    if step_position not in STEP_POSITIONS:
        raise ValueError(f"step_position must be one of {STEP_POSITIONS}, "
                         f"got {step_position!r}")
    _check_neff(neff_min, neff_unit)
    full = project.projected(condition)
    kept, cutoff_stats = apply_track_cutoff(full, max_locs, cutoff_mode)
    projected, class_stats = apply_cell_class(kept, cell_class, class_cuts_um)
    tau_s = float(project.params.tau_s)
    meshes = getattr(project, "meshes", None)
    out = spatial_sda_map(
        projected, tau_s=tau_s, bin_long_nm=float(bin_nm), bin_short_nm=float(bin_nm),
        smooth_bins=1.0, cell_length_um=cell_length_um, cell_width_um=cell_width_um,
        step_position=step_position, mirrored=mirrored, weighting=weighting,
        meshes=meshes, long_axis=long_axis)
    cfg = out["config"]

    xc, yc = cfg.centres()
    xx, yy = np.meshgrid(xc, yc)
    inside = np.abs(yy) <= local_half_width(xx, cfg)
    bin_long_um = cfg.cell_length_um / cfg.n_bins_long
    bin_short_um = cfg.cell_width_um / cfg.n_bins_short
    stamp = sigma_stamp(project)
    h_um = bandwidth_um(stamp["sigma_nm"], bin_long_um, bin_short_um)
    # the value maps' kernel follows THEIR bins (`spatialsda.VALUE_BIN_FACTOR`)
    factor = int(out["value_bin_factor"])
    value_h_um = bandwidth_um(stamp["sigma_nm"], bin_long_um * factor, bin_short_um * factor)
    adaptive = bool(value_adaptive)
    finish = functools.partial(finish_sda_panels, value_h_um=value_h_um, adaptive=adaptive)
    panels = finish(
        **_sda_parts(out), n_min=n_min, inside=inside, h_um=h_um,
        bin_long_um=bin_long_um, bin_short_um=bin_short_um,
        neff_min=neff_min, neff_unit=neff_unit)

    params = project.params
    # THE CONDITION'S OWN SEARCH RADIUS, the one it was linked at (tester D10, 4 Nov 2026), never the first condition's
    radius_px = params.search_radius_px_for(condition)
    radius_um = (None if radius_px is None
                 else float(radius_px) * float(params.um_per_pixel))
    limits = apparent_limits(tau_s=tau_s, sigma_nm=stamp["sigma_nm"],
                             search_radius_um=radius_um)
    payload = {
        "method": "sda",
        "condition": condition,
        "step_position": step_position,
        "mirrored": bool(mirrored),
        "n_min": int(n_min),
        "n_bins_inside": int(inside.sum()),
        "cell_length_um": cfg.cell_length_um,
        "cell_width_um": cfg.cell_width_um,
        "n_bins_long": cfg.n_bins_long,
        "n_bins_short": cfg.n_bins_short,
        "smooth_bins": cfg.smooth_bins,
        "bin_nm": bin_number(bin_nm),
        "h_nm": float(h_um * 1000.0),
        "panel_n_min": PANEL_N_MIN,
        **stamp,
        "frame_source": frame_source,
        # Each panel names what it counts (`count_unit`); this is the step unit the
        # weighting text and the older readers fall back on.
        "weight_unit": "steps",
        "count_units": dict(SDA_COUNTS),
        "weighting": weighting,
        "max_locs": None if max_locs is None else int(max_locs),
        "cutoff_mode": cutoff_mode,
        "cutoff_stats": cutoff_stats,
        "neff_min": None if neff_min is None else float(neff_min),
        "neff_unit": neff_unit,
        "cell_class": cell_class,
        "class_cuts_um": None if class_cuts_um is None else tuple(class_cuts_um),
        "class_stats": class_stats,
        "long_axis": long_axis,
        "tau_s": tau_s,
        "value_n_min": int(VALUE_N_MIN),
        "value_bin_factor": factor,
        "value_h_nm": float(value_h_um * 1000.0),
        "value_adaptive": adaptive,
        "value_widths": list(ADAPTIVE_WIDTHS) if adaptive else [1.0],
        "value_h_cap_nm": (float(value_h_um * ADAPTIVE_CAP * 1000.0) if adaptive
                           else float(value_h_um * 1000.0)),
        "sda_limits": limits,
        "panels": panels,
    }
    payload["days"] = _day_payloads(
        projected, {k: v for k, v in payload.items() if k != "panels"},
        lambda sub: maps_from_projected_sda(
            sub, cfg, tau_s=tau_s, step_position=step_position, mirrored=mirrored,
            weighting=weighting, meshes=meshes, with_steps=True),
        _sda_parts,
        n_min=n_min, inside=inside, h_um=h_um,
        bin_long_um=bin_long_um, bin_short_um=bin_short_um,
        kind_keys={k: k for k in KINDS_BY_METHOD["sda"]}, cfg=cfg,
        neff_min=neff_min, neff_unit=neff_unit, finish=finish)
    payload["track_census"] = track_census(full)
    # Added AFTER the days, so it is not copied into every day (each day has its own): the x and y graphs
    payload["profiles"] = profiles_by_kind(out.get("profiles"), {k: k for k in KINDS_BY_METHOD["sda"]})
    return payload


def _fmt(value: float) -> str:
    """A bound in a caption: three significant figures, never scientific notation."""
    return f"{value:.3g}" if 1e-3 <= abs(value) < 1e4 else f"{value:.2e}"


def sda_values_text(payload: dict) -> str:
    """What each SDA map is, as the caption states it -- including where a step is drawn
    and that v2 drew it elsewhere."""
    step = STEP_LABELS[STEP_POSITIONS.index(payload.get("step_position", "A_start"))]
    return (f"localisations: share of the condition's in each bin · speed |Δr|/τ and "
            f"apparent D |Δr|²/(4τ) per step, drawn at its {step} (SMTracker 2 draws every step "
            f"at its midpoint) · acceleration |Δr₂ - Δr₁|/τ² per pair of consecutive "
            f"steps, drawn at the localisation they share · mean per bin, sums and counts "
            f"pooled{' over the four quarters' if payload.get('mirrored') else ''}")


def sda_limits_text(payload: dict) -> str:
    """The one line every SDA caption carries: the values are APPARENT -- bounded above
    by the tracker's linking radius and below by localisation error -- and acceleration
    is not a force. Every bound computed from this acquisition, none assumed."""
    lim = payload.get("sda_limits") or {}
    sigma = lim.get("sigma_nm")
    if lim.get("ceiling_speed_um_s") is not None:
        top = (f"capped by the tracker's linking radius ({_fmt(lim['search_radius_um'])} µm:"
               f" speed ≤ {_fmt(lim['ceiling_speed_um_s'])} µm/s, D ≤ "
               f"{_fmt(lim['ceiling_D_um2_s'])} µm²/s)")
    else:
        top = "capped by the tracker's linking radius (not recorded for this project)"
    if lim.get("floor_D_um2_s") is not None:
        low = (f"floored by localisation error (σ {sigma:.0f} nm: a molecule that does not "
               f"move reads {_fmt(lim['floor_speed_um_s'])} µm/s, D "
               f"{_fmt(lim['floor_D_um2_s'])} µm²/s, acceleration "
               f"{_fmt(lim['floor_acceleration_um_s2'])} µm/s²)")
    else:
        low = "floored by localisation error (σ not recorded)"
    return (f"APPARENT · speed and D {top} and {low} · acceleration is dominated by "
            f"localisation noise for a diffusing molecule and is not a force")


def sda_value_bins_text(payload: dict) -> str:
    """The value maps' own bin size (`spatialsda.VALUE_BIN_FACTOR`)."""
    f = int(payload.get("value_bin_factor", 1))
    long_nm = payload["cell_length_um"] * 1000.0 / payload["n_bins_long"] * f
    short_nm = payload["cell_width_um"] * 1000.0 / payload["n_bins_short"] * f
    word = "from" if payload.get("value_adaptive") else "at"
    smoothed = (f" · smoothed {word} h {payload['value_h_nm']:.0f} nm"
                if payload.get("value_h_nm") is not None else "")
    if f == 1:
        return (f"speed, D and acceleration on the localisation map's bins, "
                f"{long_nm:.0f} × {short_nm:.0f} nm{smoothed}")
    return (f"speed, D and acceleration on bins {f} × {f} of the localisation map's, "
            f"{long_nm:.0f} × {short_nm:.0f} nm: a mean needs far more steps per bin than "
            f"a count{smoothed} (max(σ, ½ value bin))")


def _value_floor(payload: dict) -> int:
    """The value maps' minimum: the Run's own, or `spatialsda.VALUE_N_MIN` if higher."""
    return max(int(payload.get("n_min", 0)), int(payload.get("value_n_min", 0)), 1)


def sda_value_smoothing_text(payloads) -> str:
    """The "Value smoothing" caption row: the kernel widths the value maps were drawn at,
    pooled over ``payloads`` (the rows shown), and what makes a bin grey under them. With
    adaptive smoothing (`spatialsda` divergence 9) it gives the RANGE actually used, so a
    map whose every bin kept h0 says so rather than advertising widths nobody got."""
    payloads = [p for p in (payloads or ()) if p]
    if not payloads:
        return ""
    base = payloads[0]
    floor = _value_floor(base)
    h0 = float(base.get("value_h_nm") or base.get("h_nm") or 0.0)
    if not base.get("value_adaptive"):
        return (f"FIXED · h {h0:.0f} nm in every bin · grey under {floor} steps of the "
                f"bin's own")
    cap = float(base.get("value_h_cap_nm") or h0)
    lo, hi, widened, drawn = math.inf, -math.inf, 0, 0
    for p in payloads:
        for panel in (p.get("panels") or {}).values():
            span = (panel or {}).get("h_range_nm")
            if span:
                lo, hi = min(lo, span[0]), max(hi, span[1])
            widened += int((panel or {}).get("n_widened", 0))
            drawn += int((panel or {}).get("n_drawn", 0))
    tail = (f"from h0 {h0:.0f} nm up to {cap / h0:g} × h0 = {cap:.0f} nm · grey where even "
            f"{cap:.0f} nm collects fewer than {floor} steps' worth" if h0 > 0 else "")
    if not drawn:
        return f"ADAPTIVE · no value bin drawn · {tail}"
    used = (f"h {lo:.0f}–{hi:.0f} nm, wider where steps are few" if hi > lo + 0.5 else
            f"h {lo:.0f} nm in every bin, none needed wider")
    return (f"ADAPTIVE · {used} ({widened:,} of {drawn:,} drawn value-map bins widened) · "
            f"{tail}")


def sda_grey_text(payload: dict) -> str:
    """What a grey bin means on an SDA map."""
    n_min = int(payload.get("n_min", 0))
    floor = _value_floor(payload)
    if payload.get("value_adaptive"):
        cap = float(payload.get("value_h_cap_nm") or payload.get("value_h_nm") or 0.0)
        first = (f"speed, D and acceleration: fewer than {floor} steps' worth (step pairs "
                 f"for acceleration) even in the widest kernel, h {cap:.0f} nm, because a "
                 f"mean of so few is noise, or no step in the bin itself (a mean borrowed "
                 f"from its neighbours alone is not drawn)")
    else:
        first = (f"speed, D and acceleration: fewer than {floor} steps in the bin (step "
                 f"pairs for acceleration), because a mean of so few is noise")
    parts = [first, (f"localisations: fewer than {n_min}" if n_min > 0
                     else "localisations: no minimum set")]
    if payload.get("neff_min") is not None:
        parts.append("any map: built by too few independent contributors (n_eff)")
    return " · ".join(parts)


def _sda_totals(payloads: list) -> str:
    """The SDA "Total" row: what each map counted, summed over the rows -- three different
    things, never added together."""
    out = []
    for kind, what in (("location", "localisations"), ("speed", "steps"),
                       ("acceleration", "step pairs")):
        total = sum(float(((p.get("panels") or {}).get(kind) or {}).get("n_jumps", 0))
                    for p in payloads)
        out.append(f"{total:,.0f} {what}")
    return " · ".join(out)


#: The written reason every track hidden from the map carries (`exclusions` refuses an exclusion without one).
HIDE_REASON = "hidden from the Spatial Map by hand, on its spot card"


class _WithoutTracks:
    """``project`` with some of ONE condition's tracks hidden (`exclusions.apply_exclusions`, which records what it
    removed and refuses to remove everything). Every map method reads a condition through `projected` (and
    `condition`), so the map's own fit, its bins and its days all go without them; everything else is the
    project's own. Built inside the worker: only the uids cross to it."""

    def __init__(self, project, condition: str, exclusions):
        self._project, self._condition, self._exclusions = project, str(condition), exclusions

    def __getattr__(self, name):
        return getattr(self._project, name)

    def _filtered(self, name, table):
        from .exclusions import apply_exclusions

        return apply_exclusions(table, self._exclusions) if str(name) == self._condition else table

    def condition(self, name):
        return self._filtered(name, self._project.condition(name))

    def projected(self, name):
        return self._filtered(name, self._project.projected(name))


def compute_spatial_map(project, condition: str, *, method: str = "sqd",
                        hidden=None, gradient: str | None = None, **config) -> dict:
    """The picklable process-pool entry point EVERY computational method
    submits -- `window.py` needs no branch of its own, since which method ran
    is a fact of the CONFIG it was handed (`SpatialMapDialog.config()`'s own
    `"method"` key), not of the caller.

    `method="sqd"` by default so a caller predating this dispatch (there are
    none left in this codebase, but a pickled closure from an in-flight job
    at the moment this changed would be one) still resolves to what it always
    meant.

    ``hidden``: uids of this condition's tracks HIDDEN FROM THE MAP by hand (asked 16 Oct 2026: an eye on each
    track of a spot's card, "if u click on it ... the track is excluded from the map"). The map is computed
    without them (`_WithoutTracks`) and says so: ``payload["hidden_tracks"]``, on every day too, read by the
    receipt and the export. Only the map: every other section keeps them. Not a setting that splits the colour
    scale (`settings_key`): the same rule drew it, on fewer tracks.

    A bin chosen in camera pixels (``bin_camera_px``, the Run's standard since 30 Oct 2026) is resolved here
    from THIS project's pixel size (`_resolve_bin_config`), refused by name where none is known, and the
    payload -- every day too -- records the nm it was drawn at (``bin_nm``, which `settings_key` reads) and
    how it was chosen (``bin_camera_px``, ``bin_pixel_nm``, ``bin_pixel_note``), so a caption can say
    "106 nm (1 camera pixel)" and a 1 px map at 106 nm never shares a scale with a 100 nm one.

    ``gradient``: the optimiser of every SQD fit inside -- the SQD section's own (its Compute dialog's Optimiser,
    `sqd.gradient_mode`), handed in by the window as Jump distances' is (3 Nov 2026: "crazy that bic spatial map
    runs for 30 min plus"). Unnamed, the library's default (numeric), which a Best (BIC) map ran at while the SQD
    section ran Fast: choosing the count fits 1, 2 and 3 populations from many starts, and the numeric optimiser
    is about fourfold slower. The count chosen is the same.
    """
    from .sqd import gradient_mode

    with gradient_mode(gradient):
        return _compute_spatial_map(project, condition, method=method, hidden=hidden, **config)


def _compute_spatial_map(project, condition: str, *, method: str, hidden, **config) -> dict:
    """`compute_spatial_map` at the optimiser it set."""
    config, bin_stamp = _resolve_bin_config(project, config)
    hidden = sorted({str(uid) for uid in (hidden or ())})
    if hidden:
        from .exclusions import Exclusion, ExclusionSet

        project = _WithoutTracks(project, condition,
                                 ExclusionSet([Exclusion("uid", uid, HIDE_REASON) for uid in hidden]))
    if method == "appd":
        payload = compute_spatial_map_appd(project, condition, **config)
    elif method == "hmm":
        payload = compute_spatial_map_hmm(project, condition, **config)
    elif method == "sla":
        payload = compute_spatial_map_sla(project, condition, **config)
    elif method == "sda":
        payload = compute_spatial_map_sda(project, condition, **config)
    else:
        payload = compute_spatial_map_sqd(project, condition, **config)
    if bin_stamp:
        payload.update(bin_stamp)
        for day in payload.get("days") or ():
            day.update(bin_stamp)
    if hidden:
        payload["hidden_tracks"] = list(hidden)
        for day in payload.get("days") or ():
            day["hidden_tracks"] = list(hidden)
    return payload


def hidden_tracks_text(payloads, unit: str = "condition") -> str | None:
    """The receipt's line for tracks hidden from the map by hand, or None when there are none: per condition on
    the overview, once on a day page (its days share the condition's)."""
    def count(n):
        return f"{n} track{'' if n == 1 else 's'}"

    hidden = [(p.get("condition"), len(p.get("hidden_tracks") or ())) for p in payloads or ()]
    hidden = [(c, n) for c, n in hidden if n]
    if not hidden:
        return None
    if unit == "day":
        return f"{count(hidden[0][1])} of this condition, by hand on the spot card; the map only"
    return "; ".join(f"{c}: {count(n)}" for c, n in hidden) + " · by hand on the spot card; the map only"


def settings_key(payload: dict) -> tuple:
    """What has to match before two conditions' maps may share a colour scale.

    Sharing across conditions is only meaningful between panels computed the
    same way. A map folded at 100 nm bins and one unfolded at 220 nm are two
    different measurements, and putting them on one bar would invite exactly
    the comparison the scale is supposed to license.

    ``bin_nm`` is the RESOLVED bin, the nm a map was drawn at: a 1 px map is keyed at its pixel's size
    (106 nm here, 160 on another camera), never at "1 px", so it cannot pool with a 100 nm map. How the
    bin was chosen (``bin_camera_px``) is not part of the key: a map typed at 106 nm is the same
    measurement as one read from a 106 nm pixel.
    """
    return tuple(payload.get(k) for k in (
        "method", "step_position", "assignment", "mirrored", "n_min", "bin_nm",
        "cell_length_um", "cell_width_um", "frame_source",
        # How the map counts and what it left out change every number on it.
        "weighting", "max_locs", "cutoff_mode", "split_from", "n_pop",
        # the n_eff mask changes which bins are drawn and so every limit
        "neff_min", "neff_unit",
        # which cells are drawn, and where along the capsule they land
        "cell_class", "class_cuts_um", "long_axis",
        # jumps a different number of frames apart are a different measurement
        "lag_frames",
        # the SLA detector's two choices decide every set
        "radius_nm", "min_steps",
        # SDA: a widened kernel changes which value bins are drawn and what they read
        "value_adaptive",
        # the HMM map's Turning back (`spatialangles`) is another quantity than its steps
        "show")) + (
        # a manual split's ranges ARE its maps: the same ranges or no shared scale
        payload.get("split") or SPLIT_DEFAULT, _ranges_key(payload.get("ranges")),
        # where the D came from (`map_fit_source`): a section's fit, or the map's own
        (payload.get("section_fit") or {}).get("page"))


def from_zero(limits, panel: dict | None = None, field: str = "density"):
    """``limits`` ``(lo, hi)`` from 0 when ``panel`` maps a count, a density or a share (`paint_field` is not "value")
    and ``lo`` is at most `ZERO_FLOOR_SHARE` of ``hi``; otherwise unchanged (None stays None)."""
    if not limits or paint_field(panel, field) == "value":
        return limits
    lo, hi = float(limits[0]), float(limits[1])
    if hi > 0.0 and 0.0 <= lo <= ZERO_FLOOR_SHARE * hi:
        return (0.0, hi)
    return limits


def _from_zero_all(limits: dict, payload: dict | None, field: str) -> dict:
    """Every panel's limits in ``limits`` (``{kind: (lo, hi)}``) through `from_zero`, each with its own panel."""
    panels = (payload or {}).get("panels") or {}
    return {kind: from_zero(lim, panels.get(kind), field) for kind, lim in limits.items()}


def panel_limits(payload: dict, level: str, others=()) -> tuple[dict, str]:
    """The colour limits for each panel, decided over the whole set BEFORE any of it is drawn, a scale of a count,
    a density or a share from 0 when its low end is that close to it (`from_zero`). See `_panel_limits`."""
    limits, field = _panel_limits(payload, level, others)
    return _from_zero_all(limits, payload, field), field


def _panel_limits(payload: dict, level: str, others=()) -> tuple[dict, str]:
    """The colour limits for each panel, decided over the whole set BEFORE any
    of it is drawn.

    `level` is the spec's open question about what "shared" means:

    - `"population"` -- SPARSE RULE A, and the default. One scale per
      population, pooled across every OTHER condition computed at the same
      settings, so short-jumps-in-glucose and short-jumps-in-malate are
      comparable, and a sparse population's spikes can never set the bar for
      an abundant one.
    - `"both"` -- one scale across both populations of this condition. The
      spec warns against exactly this, and the warning is defused here by
      normalising each map to a probability density first, so the abundance
      difference is divided out before the limits are taken.
    - `"panel"` -- each panel on its own.

    A panel flagged sparse contributes NOTHING to the limits. It is not drawn,
    and letting it set a ceiling would be the sparse-spike failure arriving by
    the back door.

    A panel carrying its own unit (SDA) is read in that unit (`limits_key`), and
    "both" pools only panels of one unit (`_scale_group`): a speed and a D never
    share a bar.
    """
    field = "density" if level in ("population", "both") else "counts"

    def usable(p):
        return p and not p.get("sparse")

    def lim(p):
        return p[limits_key(p, field)]

    own = {k: p for k, p in payload.get("panels", {}).items() if usable(p)}
    if level == "panel":
        return {k: lim(p) for k, p in own.items()}, field

    if level == "both":
        groups: dict = {}
        for k, p in own.items():
            groups.setdefault(_scale_group(p, field), []).append(k)
        limits = {}
        for kinds in groups.values():
            pool = [lim(own[k]) for k in kinds]
            one = (min(lo for lo, _ in pool), max(hi for _, hi in pool))
            limits.update(dict.fromkeys(kinds, one))
        return limits, field

    # per population, across every condition drawn the same way
    mine = settings_key(payload)
    sets = [payload] + [o for o in others
                        if o is not payload and settings_key(o) == mine]
    limits = {}
    for kind in own:
        pool = [lim(o["panels"][kind]) for o in sets
                if usable(o.get("panels", {}).get(kind))]
        if pool:
            limits[kind] = (min(lo for lo, _ in pool), max(hi for _, hi in pool))
    return limits, field


#: The grid's COLOUR SCALE, chosen in the Visibility tool beside Bins/Smooth/n_eff. Asked for: "does
#: it make sense to include an option of how to scale the maps? in between condition, in between
#: population, everything on one scale?" Each answers a different question, so each is offered:
#:
#: - `"population"` (the default, and the grid's only scale before): one scale per population, pooled
#:   over every row -- short jumps in glucose and in malate on one scale, long jumps on another.
#: - `"row"`: one scale per condition (per DAY on the day page), shared by its populations.
#: - `"panel"`: each map on its own, in counts -- the old program's look; no comparison by colour.
#: - `"all"`: one scale for every map, both populations and every row -- "are short jumps placed
#:   differently from long ones" by eye.
#:
#: Every SHARED scale compares DENSITIES (each map normalised first), never counts, or a row or a
#: population with more jumps would simply look hotter. A sparse panel never sets a limit, and a row
#: computed at other settings is never pooled with the rest (`settings_key`).
MATRIX_SCALES = ("population", "row", "panel", "all")


def matrix_limits(payloads: dict, level: str) -> tuple[dict, str]:
    """``({row: {kind: (lo, hi)}}, field)`` for every row of the grid, decided over the whole
    set BEFORE any of it is drawn; `field` is what the maps paint ("density", or "counts" when each
    map is on its own). A panel with a unit of its own (SDA) is painted and pooled in it
    (`paint_field`, `limits_key`), and "row" and "all" pool only within one unit: one
    scale for every map means one per unit when the maps are in different units. A scale of a count, a density or
    a share whose low end is within `ZERO_FLOOR_SHARE` of its top starts at 0 (`from_zero`), on screen and in print
    alike, since both read their limits here."""
    limits, field = _matrix_limits(payloads, level)
    return {row: _from_zero_all(lims, (payloads or {}).get(row), field) for row, lims in limits.items()}, field


def _matrix_limits(payloads: dict, level: str) -> tuple[dict, str]:
    """`matrix_limits` before `from_zero`."""
    if level not in MATRIX_SCALES:
        raise ValueError(f"level must be one of {MATRIX_SCALES}, got {level!r}")
    rows = {row: p for row, p in (payloads or {}).items() if p}
    if level == "panel":
        return {row: panel_limits(p, "panel")[0] for row, p in rows.items()}, "counts"
    if level == "row":
        return {row: panel_limits(p, "both")[0] for row, p in rows.items()}, "density"
    if not rows:
        return {}, "density"

    def usable(panel):
        return panel and not panel.get("sparse")

    def lim(panel):
        return panel[limits_key(panel, "density")]

    mine = settings_key(next(iter(rows.values())))
    pooled = {row: p for row, p in rows.items() if settings_key(p) == mine}
    # A row computed at other settings is not pooled: it keeps a scale of its own.
    out = {row: panel_limits(p, "population" if level == "population" else "both")[0]
           for row, p in rows.items() if row not in pooled}
    if level == "population":
        shared = {}
        for kind in {k for p in pooled.values() for k in p.get("panels", {})}:
            pool = [lim(p["panels"][kind]) for p in pooled.values()
                    if usable(p.get("panels", {}).get(kind))]
            if pool:
                shared[kind] = (min(lo for lo, _ in pool), max(hi for _, hi in pool))
        for row, p in pooled.items():
            out[row] = {k: v for k, v in shared.items()
                        if usable(p.get("panels", {}).get(k))}
        return out, "density"
    groups: dict = {}
    for p in pooled.values():
        for panel in p.get("panels", {}).values():
            if usable(panel):
                groups.setdefault(_scale_group(panel, "density"), []).append(lim(panel))
    ones = {group: (min(lo for lo, _ in pool), max(hi for _, hi in pool))
            for group, pool in groups.items()}
    for row, p in pooled.items():
        out[row] = {k: ones[_scale_group(panel, "density")]
                    for k, panel in p.get("panels", {}).items() if usable(panel)}
    return out, "density"


def colormap_text(key: str) -> str:
    """The caption's Colormap value: a built-in's name, or "custom · <name>" for a user's
    ramp (`colour_ramps`), so a map says which ramp drew it."""
    from .colour_ramps import is_custom_key, name_of_key

    if key == V2_HOT_KEY:
        return f"{key} \u00b7 SMTracker 2's SDA ramp (reversed hot, white = low)"
    return f"custom \u00b7 {name_of_key(key)}" if is_custom_key(key) else str(key)


#: The "Colour scale" words when every map carries its own unit (SDA).
_PER_UNIT_SCALE_TEXT = {
    "population": "one per map across {rows}, each in its own unit",
    "row": "each map its own within a {unit}: no two of its maps share a unit",
    "panel": "each map its own \u00b7 no comparison by colour",
    "all": "one per unit across {rows}: different units never share a scale",
}


def matrix_scale_text(level: str, n: int, unit: str = "condition",
                      per_unit: bool = False) -> str:
    """The caption's "Colour scale" value for `level`. ``per_unit``: the maps carry units
    of their own (SDA), so no two of them share a scale and each reads in its unit."""
    rows = f"{n} {unit_word(unit)}{'' if n == 1 else 's'}"
    if per_unit:
        text = (_PER_UNIT_SCALE_TEXT[level].format(rows=rows, unit=unit_word(unit))
                + " \u00b7 localisations as a probability per bin, the rest as a mean per bin")
        return f"{text} \u00b7 clipped {ROBUST_LO_PCT:g}\u2013{ROBUST_HI_PCT:g}%"
    text = {"population": f"one per population across {rows} · as a density",
            "row": f"one per {unit_word(unit)}, shared by its populations · as a density",
            "panel": "each map its own · counts · no comparison by colour",
            "all": f"one for every map, both populations and {rows} · as a density"}[level]
    return f"{text} · clipped {ROBUST_LO_PCT:g}\u2013{ROBUST_HI_PCT:g}%"


#: Where a grid's colour bars stand: "columns" one under each population column, "rows" one to the right of each
#: row, "maps" one under each map (`bar_layout`).
BAR_LAYOUTS = ("columns", "rows", "maps")


def bar_layout(level: str, limits: dict | None = None, units: dict | None = None) -> str:
    """Where the grid's colour bars stand for colour scale ``level`` (`MATRIX_SCALES`): each bar where the maps it
    describes are (asked 16 Oct 2026: "if i choose population, the colour scale [should] be on the bottom and say
    per population; per map every map should get a colour map; per condition it should be right to the maps").

    - "population" and "all": ``"columns"``, one bar under each population column (for "all" they read alike);
    - "panel": ``"maps"``, one under every map;
    - "row": ``"rows"``, one to the right of each row -- unless a row's maps do not share ONE range (``limits``,
      ``{row: {kind: range or top}}``) or one unit (``units``, ``{kind: unit}``), as SDA's maps, each in a unit of
      its own, never do: then ``"maps"``, since one bar beside the row would describe some of them and not others.
    """
    if level not in MATRIX_SCALES:
        raise ValueError(f"level must be one of {MATRIX_SCALES}, got {level!r}")
    if level in ("population", "all"):
        return "columns"
    if level == "panel":
        return "maps"
    if units and len({u for u in units.values()}) > 1:
        return "maps"
    for row in (limits or {}).values():
        ranges = {tuple(v) if isinstance(v, (list, tuple)) else v
                  for v in (row or {}).values() if v is not None}
        if len(ranges) > 1:
            return "maps"
    return "rows"


def matrix_bar_note(level: str, unit: str = "condition", layout: str | None = None) -> str:
    """What a colour bar says it describes, where it stands (`bar_layout`)."""
    layout = layout or bar_layout(level)
    if layout == "maps":
        return "per map"
    if layout == "rows":
        return f"per {unit_word(unit)}"          # "per replicate" on a condition's page, never "per day"
    # SHORT, because the bar's two numbers win the line and a long note is dropped between them
    return {"population": "per population", "all": "one scale · every map"}[level]


def scale_range_text(limits, unit: str | None = None) -> str:
    """A map's own range, printed under it when it does not share the bar's; in the map's
    own unit when it has one (SDA)."""
    if not limits:
        return ""
    lo, hi = limits
    if unit:
        return f"scale {bar_number(lo)}\u2013{bar_number(hi)} {unit}"
    return f"scale {bar_number(lo)}\u2013{bar_number(hi)}"


def bar_number(value) -> str:
    """A number at the end of a colour bar: `chart_text.scale_number`, THE formatter of every colour bar's ends
    (30 Oct 2026; it was this module's for SDA's units alone, the density bars printing ``.3g``, "9.39e-06"). Three
    significant figures in plain decimals, an acceleration of 1234 \u00b5m/s\u00b2 "1,234", a probability of 3.7e-05
    "0.000037", 0 "0"; exponent notation only under 1e-5 or from 1e6."""
    from .chart_text import scale_number

    return scale_number(value)


#: How a grid is PAINTED. Display only: not one number in the payload moves between the two.
RENDERINGS = ("bins", "smooth", "neff", "tracks")

#: Display pixels per bin, per axis, in smooth mode.
SMOOTH_FACTOR = 8


def rendering_text(payload: dict, rendering: str) -> str | None:
    """The caption sentence for a non-default rendering, or None for plain bins.

    A smooth picture looks finer than the data is, so it says so and names the real resolution: the
    bin size still decides what can be resolved, and the interpolation adds no information."""
    if rendering == "bins":
        return None
    if rendering == "tracks":
        return ("tracks drawn as lines, unfolded, in SMTracker 2's colours (free blue, "
                "mixed-behaviour green, dwelling red); the density is not drawn"
                if payload.get("method") == "sla" else
                "the Tracks view draws Confinement (SLA) maps only")
    if rendering == "neff":
        unit = _NEFF_UNIT_TEXT[payload.get("neff_unit", "tracks")
                               if payload.get("neff_min") is not None else "tracks"]
        return (f"colour = n_eff, how many equal {unit} built each bin (log scale from 1, "
                f"one top for every panel shown, named on the bar: the run's "
                f"{100 * NEFF_TOP_QUANTILE:g}th percentile rounded up, at least "
                f"{NEFF_SCALE_TOP:g}; grey = empty; outlined = masked) · not a density")
    if rendering != "smooth":
        raise ValueError(f"rendering must be one of {RENDERINGS}, got {rendering!r}")
    bin_l = payload["cell_length_um"] / payload["n_bins_long"] * 1000
    bin_s = payload["cell_width_um"] / payload["n_bins_short"] * 1000
    return (f"display smoothed (bilinear between bin centres; the data resolution is still "
            f"{bin_l:.0f} × {bin_s:.0f} nm bins)")


#: The grid caption's "Rendering" row, in BOTH modes. Always present and short
#: on purpose: a row that appeared only in Smooth made the caption taller,
#: which could bring in the page's scrollbar and resize every map -- the maps
#: visibly jumped on each switch. Both values fit one line, so switching
#: changes no height. The bin size is already its own row, so Smooth does not
#: repeat it here (the single-sentence caption, `caption_for`, still does).
RENDERING_ROW = {"bins": "BINS · each bin as computed",
                 "smooth": "SMOOTH · bilinear, display only",
                 "neff": "N_EFF · contributors per bin, display only",
                 "tracks": "TRACKS · each set's tracks as lines, in SMTracker 2's colours"}

#: The n_eff view's scale: log, from 1 contributor to a TOP shared by every panel of
#: the run -- one colour means one n_eff on every panel and every condition, since the
#: point of the view is comparing how thin maps are, which a per-panel stretch hides.
#:
#: **THE TOP COMES FROM THE RUN, NEVER BELOW 20.** It was a fixed 20, and on real data
#: the maps were yellow edge to edge: panels of 4 500 to 28 000 weighted jumps put
#: nearly every inside bin far past 20 contributors, so the whole colour range was spent
#: on the rim. Asked: "the scale for n_eff is a bit off since everything nearly is
#: yellow". `neff_scale_top` takes the 95th percentile of every bin's n_eff in the run
#: and rounds it UP to the next of `NEFF_TOP_STEPS`, so a bin of a handful of tracks
#: still reads dark while the inside spreads out; 20 stays the floor, so a thin run
#: still looks thin rather than being stretched to fill the ramp. The bar names the top.
NEFF_SCALE_TOP = 20.0
NEFF_TOP_QUANTILE = 0.95
NEFF_TOP_STEPS = (20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0)


def _neff_top_of(panels) -> float:
    """The top for these panels, pooled: their n_eff's `NEFF_TOP_QUANTILE`, rounded up to
    `NEFF_TOP_STEPS`, never below `NEFF_SCALE_TOP`. Empty bins do not count."""
    values = []
    for panel in panels:
        for row in (panel or {}).get("neff") or ():
            values.extend(float(v) for v in row if v is not None and v > 0)
    if not values:
        return NEFF_SCALE_TOP
    values.sort()
    q = values[min(len(values) - 1, int(math.ceil(NEFF_TOP_QUANTILE * len(values))) - 1)]
    return next((step for step in NEFF_TOP_STEPS if step >= q), NEFF_TOP_STEPS[-1])


def neff_scale_top(payloads) -> float:
    """The n_eff view's top for these payloads, every panel of every one pooled (the
    "All" scale); no n_eff at all gives the floor."""
    return _neff_top_of(panel for payload in payloads or ()
                        for panel in ((payload or {}).get("panels") or {}).values())


def neff_tops(payloads: dict, level: str) -> dict:
    """``{row: {kind: top}}``: the n_eff view's log-scale top for every map, following the
    grid's colour scale (`MATRIX_SCALES`) as the density maps do -- per population across
    rows, per row, per map, or one for all. Asked: "when i choose population, all,
    condition etc for scale ... they dont change in n eff mode". A row computed at other
    settings is never pooled with the rest; it gets a top of its own."""
    if level not in MATRIX_SCALES:
        raise ValueError(f"level must be one of {MATRIX_SCALES}, got {level!r}")
    rows = {row: p for row, p in (payloads or {}).items() if p}

    def panels(p):
        return (p.get("panels") or {})

    if level == "panel":
        return {row: {k: _neff_top_of([pl]) for k, pl in panels(p).items()}
                for row, p in rows.items()}
    if level == "row" or not rows:
        return {row: dict.fromkeys(panels(p), _neff_top_of(panels(p).values()))
                for row, p in rows.items()}
    mine = settings_key(next(iter(rows.values())))
    pooled = {row: p for row, p in rows.items() if settings_key(p) == mine}
    out = {row: dict.fromkeys(panels(p), _neff_top_of(panels(p).values()))
           for row, p in rows.items() if row not in pooled}
    if level == "population":
        kinds = {k for p in pooled.values() for k in panels(p)}
        tops = {k: _neff_top_of(panels(p).get(k) for p in pooled.values()) for k in kinds}
        for row, p in pooled.items():
            out[row] = {k: tops[k] for k in panels(p)}
        return out
    top = _neff_top_of(pl for p in pooled.values() for pl in panels(p).values())
    for row, p in pooled.items():
        out[row] = dict.fromkeys(panels(p), top)
    return out


def neff_scale_text(level: str, n: int, unit: str = "condition") -> str:
    """The caption's "Colour scale" value in the n_eff view."""
    rows = f"{n} {unit_word(unit)}{'' if n == 1 else 's'}"
    shared = {"population": f"one top per population across {rows}",
              "row": f"one top per {unit_word(unit)}, shared by its populations",
              "panel": "each map its own top",
              "all": f"one top for every map and {rows}"}[level]
    return (f"n_eff, log scale from 1 · {shared} (the {100 * NEFF_TOP_QUANTILE:g}th "
            f"percentile rounded up, at least {NEFF_SCALE_TOP:g})")


def neff_position(value: float, top: float = NEFF_SCALE_TOP) -> float:
    """Where an n_eff sits on the log scale: 0 at one contributor, 1 at `top` or more."""
    if value <= 1.0:
        return 0.0
    return min(1.0, math.log(value) / math.log(top))


def neff_display_panel(panel: dict | None, top: float = NEFF_SCALE_TOP) -> dict | None:
    """One panel as the n_eff view paints it, in the shape the chart already draws:
    `neff_pos` (scale positions), `masked` (grey: no contributor -- empty, or outside
    the cell), `outline` (the bins the run MASKED, still coloured here so you can see
    what the cutoff removed). Never sparse-greyed: the view is the diagnostic for
    exactly the thin panels. None when the payload predates n_eff."""
    if not panel or panel.get("neff") is None:
        return None
    pos, grey, outline = [], [], []
    for row, mrow in zip(panel["neff"], panel["masked"]):
        prow, grow, orow = [], [], []
        for v, m in zip(row, mrow):
            empty = v is None or v <= 0
            prow.append(0.0 if empty else neff_position(v, top))
            grow.append(empty)
            orow.append(bool(m) and not empty)
        pos.append(prow)
        grey.append(grow)
        outline.append(orow)
    return {"neff_pos": pos, "masked": grey, "outline": outline,
            "n_jumps": panel["n_jumps"], "sparse": False,
            # carried so a hover reads the same number in this view as in the others
            "neff": panel["neff"], "neff_unit": panel.get("neff_unit"),
            # and, on an adaptively smoothed value map, the kernel that built each bin
            **{k: panel[k] for k in ("h_nm", "steps_eff", "value_min") if k in panel}}


def neff_hover_text(panel: dict | None, ix: int, iy: int) -> str | None:
    """What a bin says when the pointer rests on it: its own n_eff (the panel's `neff`,
    in the panel's unit). Words on demand, never printed. None where there is nothing
    to say -- outside the cell, an empty bin, or a payload from before n_eff.

    On an ADAPTIVELY smoothed value map (`finish_value_panels`) the bin's n_eff counts the
    contributors of its own value bin, while its colour comes from a kernel that may reach
    further; so it also says that kernel's width and how many steps' worth it collected
    -- or, for a grey bin, how few even the widest did."""
    grid = (panel or {}).get("neff")
    if not grid or not (0 <= iy < len(grid) and 0 <= ix < len(grid[0])):
        return None
    value = grid[iy][ix]
    if not value:
        return None
    unit = (panel.get("neff_unit") or "tracks").rstrip("s")
    text = f"{value:.1f} independent {unit}{'' if round(value, 1) == 1.0 else 's'}"
    widths, worth = panel.get("h_nm"), panel.get("steps_eff")
    h = widths[iy][ix] if widths else None
    steps = worth[iy][ix] if worth else None
    if h is None or steps is None:
        return text
    if steps < float(panel.get("value_min", 0)) * (1.0 - 1e-9):
        return f"{text} · grey: {steps:.0f} steps' worth even at h {h:.0f} nm"
    return f"{text} · mean of {steps:.0f} steps' worth at h {h:.0f} nm"


def neff_bar_mark(payload: dict | None, top: float = NEFF_SCALE_TOP) -> float | None:
    """Where the cutoff sits on the n_eff view's colour bar, or None when it is off."""
    if not payload or payload.get("neff_min") is None:
        return None
    return neff_position(float(payload["neff_min"]), top)


def smooth_display(grid: list, masked: list, factor: int = SMOOTH_FACTOR) -> list:
    """The grid resampled `factor` times finer by BILINEAR interpolation, for painting only.

    Returns `n_short * factor` rows of `n_long * factor` values, `None` wherever the pixel lies in a
    masked bin. Rules, each because the legacy `interp2(..., 'cubic')` broke it:

    * **Bilinear, never cubic.** Every output is a convex combination of input bins, so it can never
      leave [min, max] of them -- cubic overshoots below zero and the overshoot reads as an empty bin.
    * **A masked bin is undefined, not zero.** It never contributes to a neighbour (its weight is
      dropped and the rest renormalised) and its own pixels come back `None`, so it stays sharp grey.
    * Row 0 is the BOTTOM of the cell, as in the payload; the output keeps that order.
    """
    import numpy as np

    if factor < 1 or int(factor) != factor:
        raise ValueError(f"factor must be a positive integer, got {factor!r}")
    g = np.asarray(grid, dtype=float)
    m = np.asarray(masked, dtype=bool)
    if g.ndim != 2 or g.shape != m.shape:
        raise ValueError(f"grid {g.shape} and mask {m.shape} must be the same 2-D shape")
    ny, nx = g.shape
    valid = (~m & np.isfinite(g)).astype(float)
    g0 = np.where(valid > 0, g, 0.0)

    def axis(n):
        # Output pixel centres in input-bin units; bin k's centre sits at k.
        # Beyond the first/last centre there is no neighbour, so the coordinate
        # is CLAMPED to [0, n-1] first (the edge keeps its bin's own value).
        c = np.clip((np.arange(n * factor) + 0.5) / factor - 0.5, 0.0, n - 1.0)
        i0 = np.floor(c).astype(int)
        i1 = np.minimum(i0 + 1, n - 1)
        w1 = c - i0
        return i0, i1, 1.0 - w1, w1, np.minimum(np.arange(n * factor) // factor, n - 1)

    y0, y1, wy0, wy1, own_y = axis(ny)
    x0, x1, wx0, wx1, own_x = axis(nx)
    num = np.zeros((ny * factor, nx * factor))
    den = np.zeros_like(num)
    for yi, wy in ((y0, wy0), (y1, wy1)):
        for xi, wx in ((x0, wx0), (x1, wx1)):
            w = wy[:, None] * wx[None, :] * valid[np.ix_(yi, xi)]
            num += w * g0[np.ix_(yi, xi)]
            den += w
    out = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)
    undefined = m[np.ix_(own_y, own_x)] | (den <= 1e-12)
    result = out.tolist()
    for r in np.nonzero(undefined.any(axis=1))[0]:
        row = result[r]
        for c in np.nonzero(undefined[r])[0]:
            row[c] = None
    return result


def caption_for(payload: dict, level: str, cmap: str, n_sets: int = 1,
                rendering: str = "bins") -> str:
    """`render.MapGrid.caption`'s own shape: the numbers that decide what the
    picture means, printed on the picture."""
    if not payload:
        return ""
    method = payload.get("method", "sqd")
    kinds = kinds_for(method, payload.get("n_pop") or 2, payload.get("ranges"))
    bin_l = payload["cell_length_um"] / payload["n_bins_long"] * 1000
    bin_s = payload["cell_width_um"] / payload["n_bins_short"] * 1000
    jumps = drawn_total(payload)
    sparse = sum(1 for p in payload["panels"].values() if p.get("sparse"))
    n_min = payload["n_min"]
    # PER PANEL, against ONE panel's bin count -- the two are separate maps
    # over the same grid, so summing their masked bins and reading that
    # against a single panel's total says 175 of 236 where each panel is
    # really about a third masked.
    masked = " / ".join(str(payload["panels"][k].get("n_masked_count",
                                                     payload["panels"][k]["n_masked"]))
                        for k in kinds if k in payload["panels"])
    n_min_text = ("no minimum, every bin drawn" if n_min <= 0 else
                  f"n_min {n_min} ({masked} of {payload['n_bins_inside']} "
                  f"bins inside the cell masked)")
    frame = ("measured from this condition's own outlines"
             if payload.get("frame_source") == "measured"
             else "a standard capsule shape, not measured here")
    # The spec's sparse rule B asks for robust limits AND for the figure to
    # SAY they are robust -- an unstated clip is indistinguishable from a
    # min-max scale to the reader.
    scale_text = {
        "population": (f"one scale per population across {n_sets} condition"
                       f"{'' if n_sets == 1 else 's'}, as a density"),
        "both": "one scale for both populations, as a density",
        "panel": "each panel on its own counts",
    }[level] + f", clipped {ROBUST_LO_PCT:g}\u2013{ROBUST_HI_PCT:g}%"
    # SQD and HMM both read the jump-position/assignment vocabulary (both
    # are per-STEP, both carry a posterior to soften -- see `spatialhmm`'s
    # own module docstring). appD has neither (a per-track scalar, no jump
    # to threshold or place) and states where its fitted populations cross.
    if method == "appd":
        method_text = "apparent diffusion, " + crossing_text(payload)
    elif method == "sla":
        method_text = (f"confinement (SLA), radius {payload['radius_nm']:g} nm, dwell ≥ "
                       f"{payload['min_steps']} steps · {sla_chance_text(payload)}")
    elif method == "sda":
        method_text = (f"SDA, SMTracker 2's spatial distribution maps · {sda_values_text(payload)} · "
                       f"value smoothing {sda_value_smoothing_text([payload]).lower()} · "
                       f"{sda_limits_text(payload)} · grey: {sda_grey_text(payload)}")
    else:
        step = STEP_LABELS[STEP_POSITIONS.index(payload["step_position"])]
        method_text = f"{payload['assignment']} assignment \u00b7 jump drawn at its {step}"
        if payload.get("split") == "manual":
            method_text = f"jump lengths {manual_split_text(payload)} \u00b7 jump drawn at its {step}"
        if payload.get("fit_weighting"):
            method_text += f" \u00b7 SQD fit {payload['fit_weighting']}-weighted"
        if method == "hmm":
            kept = payload.get("fraction_kept", 1.0)
            method_text += (" \u00b7 HMM fit"
                            + ("" if kept >= 0.999 else
                               f", {kept:.0%} of steps kept at its posterior floor"))
        if method == "hmm" and payload.get("show") in ("angles", "turning"):
            # TURNING BACK (22 Nov 2026, `spatialangles`): no jump is placed and nothing is assigned
            turning = payload.get("turning") or {}
            method_text = (("turning back beyond chance: " if payload["show"] == "turning" else "turning back: ")
                           + "of each state's pairs of steps within 30\u00b0 of forward or back, the share turned back"
                           + (", minus free molecules' at the same places" if payload["show"] == "turning" else "")
                           + " \u00b7 HMM fit"
                           + (f" \u00b7 {int(turning['n_pairs']):,} pairs" if turning.get("n_pairs") else ""))
    shown = rendering_text(payload, rendering)
    extras = []
    if payload.get("weighting") == "track":
        extras.append("every track counted once")
    if payload.get("neff_min") is not None:
        extras.append(f"{setting_name('neff').lower()}: {neff_caption(payload)}")
    if payload.get("max_locs") is not None:
        extras.append(f"track cutoff: {_cutoff_caption(payload)}")
        if payload.get("split_from") == "kept":
            extras.append("short/long split from the kept tracks")
    if payload.get("cell_class", "all") != "all":
        extras.append(f"cells: {cell_class_text(payload).lower()}")
    if payload.get("long_axis") == "pole_anchored":
        extras.append("long axis pole-anchored")
    method_text = " · ".join([method_text, *extras])
    return (
        f"cell {payload['cell_length_um']:.2f} × {payload['cell_width_um']:.2f} µm "
        f"({frame}) · "
        f"{payload['n_bins_long']} × {payload['n_bins_short']} bins "
        f"({bin_l:.0f} × {bin_s:.0f} nm) · "
        + (f"bin {choice} · " if (choice := bin_choice_text(payload)) else "")
        + f"{bandwidth_text(payload)} · "
        f"{n_min_text} · {'folded (counts summed)' if payload['mirrored'] else 'unfolded'} · "
        f"{scale_text} · "
        f"{method_text} · "
        + (f"N {_sda_totals([payload])}" if method == "sda" else
           f"N {jumps:,.0f} {payload.get('weight_unit', 'weighted jumps')}")
        + f" · {cmap}"
        + (f" · {shown}" if shown else "")
        + (f" · {sparse} greyed (under {payload.get('panel_n_min', PANEL_N_MIN)} "
           f"jumps or over {PANEL_MAX_MASKED_FRAC:.0%} masked)" if sparse else "")
    )


def bandwidth_text(payload: dict) -> str:
    """`h` in nanometres, with the RULE that produced it and the sigma beside it.

    The spec asks for "smoothing bandwidth h (um) and the sigma it's tied to".
    It now genuinely is tied to sigma -- `bandwidth_um` is
    `render._resolve_bandwidth`'s own `max(sigma, bin_halfwidth)` -- so the
    caption names that rule rather than the previous "= 1 bin", which was true
    of the old fixed kernel and would now be false.
    """
    h = payload.get("h_nm")
    if h is None:
        return f"smoothing {payload.get('smooth_bins', 1):g} bins"
    sigma, source = payload.get("sigma_nm"), payload.get("sigma_provenance")
    if sigma is None:
        return f"h {h:.0f} nm = \u00bd bin \u00b7 \u03c3 not recorded"
    rule = "\u03c3" if sigma >= h - 1e-9 else "\u00bd bin"
    return (f"h {h:.0f} nm = max(\u03c3, \u00bd bin) = {rule} \u00b7 "
            f"\u03c3 {sigma:.0f} nm ({str(source).replace('_', '-')})")


def matrix_caption_rows(payloads: list, cmap: str, rendering: str = "bins",
                        unit: str = "condition", scale: str = "population") -> list:
    """The caption as (NAME, VALUE) pairs instead of one long sentence.

    Asked for directly, and the reason is legibility rather than taste: the
    run-on version put eleven facts in one paragraph separated by middots, so
    finding "what was the assignment" meant reading the whole thing. There is
    room under the grid for a list, and a list is scannable.

    **The names are `SETTING_NAMES`, the Run dialog's own.** Somebody who
    just set `Jump drawn at` should read `Jump drawn at` back, not "step
    position" -- one vocabulary, in the order the dialog's nav puts them, then
    the facts that are not settings (smoothing, scale, ramp, N).

    Values are CAPITALISED PER VALUE, never by styling the label, and units
    are left alone -- `_value_text`'s own note: upper-casing `100 nm` gives a
    non-unit, and upper-casing the micro sign gives M, so micrometres would
    read as millimetres. This project has shipped that once already.
    """
    if unit not in ("condition", "day"):
        raise ValueError(f"unit must be 'condition' or 'replicate', got {unit!r}")
    if not payloads:
        return []
    row_unit = unit
    base = payloads[0]
    method = base.get("method", "sqd")
    unit = base.get("weight_unit", "weighted jumps")
    bin_l = base["cell_length_um"] / base["n_bins_long"] * 1000
    bin_s = base["cell_width_um"] / base["n_bins_short"] * 1000
    jumps = sum(drawn_total(q) for q in payloads)
    sparse = sum(1 for q in payloads for p in q["panels"].values()
                 if p.get("sparse"))
    n_min = base["n_min"]
    frame = ("measured from each condition's own outlines"
             if base.get("frame_source") == "measured"
             else "a standard capsule shape, not measured here")
    n = len(payloads)
    if rendering not in RENDERING_ROW:
        raise ValueError(f"rendering must be one of {RENDERINGS}, got {rendering!r}")
    # SDA's maps count three different things: each total is said, never their sum.
    total = _sda_totals(payloads) if method == "sda" else f"{jumps:,.0f} {unit}"
    if sparse:
        total += (f" · {sparse} panel{'' if sparse == 1 else 's'} greyed "
                  f"(under {base.get('panel_n_min', PANEL_N_MIN)} "
                  f"{'counted' if method == 'sda' else unit} or "
                  f"over {PANEL_MAX_MASKED_FRAC:.0%} masked)")
    # SQD's "Jump drawn at"/"Assignment" rows do not exist under Apparent
    # Diffusion (see `_KEYS_BY_METHOD`); its own D-threshold fact replaces
    # them, in the same (name, value) shape. HMM keeps both rows (it is
    # ALSO per-step with a posterior to soften) and adds its own fit
    # diagnostic where SQD/appD have nothing analogous.
    if method == "sqd" and base.get("split") == "manual":
        # A SPLIT BY HAND has no assignment: the ranges ARE the maps (`SPLITS`)
        method_rows = [
            (SETTING_NAMES["lag"], lag_text(base.get("lag_frames", 1))),
            (SETTING_NAMES["jump"], STEP_LABELS[
                 STEP_POSITIONS.index(base["step_position"])].upper()),
            ("Split", manual_split_text(base) if row_unit == "day" else "; ".join(
                f"{p.get('condition')}: {manual_split_text(p)}" for p in payloads)),
        ]
    elif method == "sqd":
        method_rows = [
            (SETTING_NAMES["lag"], lag_text(base.get("lag_frames", 1))),
            (SETTING_NAMES["jump"], STEP_LABELS[
                 STEP_POSITIONS.index(base["step_position"])].upper()),
            (SETTING_NAMES["assignment"], str(base["assignment"]).upper()),
        ]
    elif method == "hmm":
        kept = base.get("fraction_kept", 1.0)
        method_rows = [
            (SETTING_NAMES["jump"], STEP_LABELS[
                 STEP_POSITIONS.index(base["step_position"])].upper()),
            (SETTING_NAMES["assignment"], str(base["assignment"]).upper()),
            ("HMM fit",
             "every step kept" if kept >= 0.999
             else f"{kept:.0%} of steps kept at its posterior floor"),
        ]
        if base.get("show") in ("angles", "turning"):
            method_rows = [(SETTING_NAMES["show"], "TURNING BACK BEYOND CHANCE" if base["show"] == "turning"
                            else "TURNING BACK"),
                           ("Placed at", "THE TURN (THE SHARED LOCALISATION)"), method_rows[-1]]
    elif method == "sla":
        method_rows = [
            ("Computational method", "CONFINEMENT (SLA)"),
            (SETTING_NAMES["radius"], f"{base['radius_nm']:g} nm"),
            (SETTING_NAMES["dwell"], f"{base['min_steps']} steps"),
            # The shares against chance, per condition (a day page shares its condition's).
            ("Against chance", sla_chance_text(base) if row_unit == "day" else "; ".join(
                f"{p.get('condition')}: {sla_chance_text(p)}" for p in payloads)),
        ]
    elif method == "sda":
        step = STEP_LABELS[STEP_POSITIONS.index(base["step_position"])]
        method_rows = [
            ("Computational method", "SDA · SMTracker 2's spatial distribution maps"),
            (SETTING_NAMES["jump"], f"{step.upper()} · SMTracker 2 drew every step at its midpoint"),
            ("Maps", sda_values_text(base)),
            # The bounds are the acquisition's: one project, one line.
            ("Apparent", sda_limits_text(base)),
            ("Value bins", sda_value_bins_text(base)),
            ("Value smoothing", sda_value_smoothing_text(payloads)),
            ("Grey bin", sda_grey_text(base)),
        ]
    else:
        method_rows = [
            ("Computational method", "APPARENT DIFFUSION"),
            # A day page's rows share their condition's split, so it is said once.
            ("Split", crossing_text(base) if row_unit == "day" else "; ".join(
                f"{p.get('condition')}: {crossing_text(p)}" for p in payloads)),
        ]
    # EACH CONDITION'S COUNT, where a Run fitted them at different counts or chose them by BIC
    if method in KIND_ORDER and base.get("split") != "manual":
        said = {population_choice_text(p) for p in payloads}
        method_rows.insert(1, (SETTING_NAMES["populations"], population_choice_text(base)
                               if row_unit == "day" or len(said) == 1 else "; ".join(
                                   f"{p.get('condition')}: {population_choice_text(p)}" for p in payloads)))
    return [
        *method_rows,
        (SETTING_NAMES["fold"],
         "ON · counts summed" if base["mirrored"] else "OFF"),
        (SETTING_NAMES["frame"],
         f"{base['cell_length_um']:.2f} × {base['cell_width_um']:.2f} µm · {frame}"),
        (SETTING_NAMES["cells"], cell_class_text(base) if n == 1 else
         " · ".join(b for b in (str(base.get("cell_class", "all")).upper(),
                                cell_class_range_text(base.get("cell_class", "all"),
                                                      base.get("class_cuts_um")))
                    if b)),
        (SETTING_NAMES["axis"], long_axis_text(base)),
        # A bin read from the camera pixel is NAMED so ("106 nm (1 camera pixel)", ruled 30 Oct 2026),
        # before the grid it became; a bin typed in nm (and every map before) keeps the grid alone.
        (SETTING_NAMES["bin"],
         ((f"{bin_text(base)} · " if base.get("bin_camera_px") else "")
          + f"{base['n_bins_long']} × {base['n_bins_short']} bins · "
          f"{bin_l:.0f} × {bin_s:.0f} nm"
          + (f" · {base['bin_pixel_note']}" if base.get("bin_camera_px")
             and base.get("bin_pixel_note") else ""))),
        (setting_name("n_min", method),
         (f"{n_min if n_min > 0 else 'OFF'} on the localisations · the value maps' own "
          f"floor {_value_floor(base)} "
          + ("steps' worth in the kernel" if base.get("value_adaptive") else "steps")
          + f" · of {base['n_bins_inside']} bins inside the cell") if method == "sda" else
         "OFF · every bin drawn" if n_min <= 0
         else f"{n_min} · of {base['n_bins_inside']} bins inside the cell"),
        ("Smoothing", bandwidth_text(base)),
        ("Weighting", _WEIGHTING_TEXT[base.get("weighting", "steps")]),
        (SETTING_NAMES["neff"], neff_caption(base)),
        ("Track cutoff", _cutoff_caption(base)),
        *([("Hidden by hand", hidden)] if (hidden := hidden_tracks_text(payloads, row_unit)) else []),
        *([("Short/long split",
            "KEPT TRACKS" if base.get("split_from") == "kept" else "ALL TRACKS")]
          if method in ("sqd", "hmm") else []),
        ("Colour scale", (neff_scale_text(scale, n, row_unit) if rendering == "neff"
                          else matrix_scale_text(scale, n, row_unit,
                                                 per_unit=has_panel_units(payloads)))),
        ("Colormap", colormap_text(cmap)),
        ("Rendering", RENDERING_ROW[rendering]),
        # The day page's rows share ONE rule -- said, so nobody reads a
        # difference between days as a difference between fits. The overview
        # says its own: each condition (row) is its own fit. WHICH D (4 Nov 2026, `map_fit_source`): a section's
        # fit is named ("D from the SQD fit: Force same D across Conditions, 3 populations"), the map's own said so.
        ("Fit", _fit_row(payloads, row_unit)),
        ("Total", total),
    ]


def _fit_row(payloads: list, row_unit: str) -> str:
    """The caption's Fit row: the section's fit the D came from (`map_fit_source.fit_row_text`), else the map's own
    fit -- "the map's own" where a fitted map drew it, so a reader never takes it for the section's."""
    from .map_fit_source import fit_row_text

    text = fit_row_text(payloads, row_unit)
    if text is not None:
        return text
    base = payloads[0]
    own = base.get("method") in N_POP_METHODS and base.get("split") != "manual"
    words = "the map's own, one per condition" if own else "one per condition"
    per_spot = own_precision_text(payloads) if own else None
    return (words + (f" · {per_spot}" if per_spot else "")
            + (", shared by every replicate" if row_unit == "day" else ""))


def rendering_row_names(method: str = "sqd") -> tuple:
    """The caption rows that describe how the map is DRAWN, not what was computed --
    the Metadata receipt's RENDERING heading. Named through the same vocabulary the
    rows use, so a renamed setting cannot drift out of its group."""
    return (setting_name("n_min", method), SETTING_NAMES["neff"],
            "Colour scale", "Colormap", "Rendering")


#: The method as the receipt's stamp names it.
_RECEIPT_METHOD = {"sqd": "SQD", "hmm": "HMM", "appd": "APPARENT D"}


def spatial_receipt(payloads: list, cmap: str, rendering: str = "bins",
                    unit: str = "condition", scale: str = "population", *,
                    computed_at: str | None = None) -> dict | None:
    """The Metadata receipt for a Spatial Map page: `matrix_caption_rows` (the same
    rows, nothing recomputed) arranged by `metadata_view.receipt_from_rows`. None when
    nothing has been drawn. The stamp names the method, then the scope -- the
    conditions on the overview, the condition and its days on a day page."""
    from .metadata_view import receipt_from_rows

    rows = matrix_caption_rows(payloads, cmap, rendering, unit=unit, scale=scale)
    if not rows:
        return None
    method = payloads[0].get("method", "sqd")
    n = len(payloads)
    if unit == "day":
        # "3 REPLICATES", never "3 DAYS": a user reads replicate (ruled 24 Oct 2026, `chart_text.unit_word`)
        scope = (f"{str(payloads[0].get('condition', '')).upper()} · "
                 f"{n} {_unit_upper(n)}")
    elif n == 1:
        scope = str(payloads[0].get("condition", "")).upper()
    else:
        scope = f"{n} CONDITIONS"
    # No build on the stamp (ruled by the UI designer); the window's title bar names it.
    stamp = ["SMTRACKER", f"SPATIAL MAP · {_RECEIPT_METHOD.get(method, method.upper())}",
             scope, computed_at or ""]
    return receipt_from_rows(rows, stamp=stamp, rendering_names=rendering_row_names(method),
                             short=_receipt_short(payloads, method, scale, unit))


def _unit_upper(n: int) -> str:
    """The replicate unit as the receipt prints it, "REPLICATE" / "REPLICATES" (`chart_text.unit_word`)."""
    from .chart_text import unit_word

    word = unit_word("day").upper()
    return word if n == 1 else word + "S"


#: The receipt's own (shorter) names for rows whose caption name is a sentence.
_RECEIPT_LABELS = {"Computational method": "Method",
                   "Minimum jumps per bin": "Min. jumps/bin",
                   "Minimum localisations per bin": "Min. locs/bin",
                   "Effective sample size": "n_eff cutoff"}
#: A colour scale in one or two words: the caption's sentence goes to the hover.
_RECEIPT_SCALES = {"population": "PER POPULATION", "panel": "PER MAP", "all": "ONE FOR ALL"}


def _split_value(payload: dict) -> str:
    """An Apparent-D split as a receipt value: the crossing(s) alone, in µm²/s -- or the ranges set by hand."""
    if payload.get("split") == "manual":
        return "BY HAND · " + " / ".join(range_title(payload.get("method", "sqd"), lo, hi)
                                         for lo, hi in payload.get("ranges") or ())
    cuts = payload.get("crossings_um2_s")
    if not cuts:
        return "NOT RECORDED"
    text = " / ".join(f"{c:.3g}" for c in cuts) + " µm²/s"
    return text + (" · EMPTY POP." if payload.get("never_most_likely") else "")


def _receipt_short(payloads: list, method: str, scale: str, unit: str = "condition"):
    """The design's short wording per row, as ``short(name, value) -> (label, value)``, or
    a LIST of ``(label, value, detail)`` to itemise a row (the Apparent-D split: one line
    per condition, as a receipt lists items). Rows it does not name keep the caption's
    name and the value up to its first " · "."""
    base = payloads[0]

    def short(name: str, value: str):
        label = _RECEIPT_LABELS.get(name, name)
        if name == "Computational method":
            return label, _RECEIPT_METHOD.get(method, value)
        if name == "Split":
            # A RESULT, not a setting (the UI designer): its own SPLIT section, one row per condition
            # named by the condition; a day page has its one condition's.
            from .metadata_view import SPLIT
            shown = [base] if unit == "day" else payloads
            return [(str(p.get("condition")), _split_value(p), crossing_text(p), SPLIT)
                    for p in shown]
        if name == "HMM fit":
            return label, value.replace(" at its posterior floor", "").upper()
        if name == SETTING_NAMES["bin"]:
            if base.get("bin_camera_px"):
                # the bin as chosen and the grid; the bins' own nm go to the hover
                return label, f"{bin_text(base)} · {base['n_bins_long']} × {base['n_bins_short']}"
            bin_l = base["cell_length_um"] / base["n_bins_long"] * 1000
            bin_s = base["cell_width_um"] / base["n_bins_short"] * 1000
            return label, (f"{base['n_bins_long']} × {base['n_bins_short']} · "
                           f"{bin_l:.0f} × {bin_s:.0f} nm")
        if name == "Smoothing" and base.get("h_nm") is not None:
            sigma = base.get("sigma_nm")
            text = f"h {base['h_nm']:.0f} nm"
            return label, text + (f" · σ {sigma:.0f} nm" if sigma is not None else "")
        if name == "Fit":
            fit = next((p.get("section_fit") for p in payloads if p.get("section_fit")), None)
            if fit is not None:
                from .map_fit_source import fit_short_text

                return label, fit_short_text(fit)
            return label, "1 PER CONDITION"
        if name == "Colour scale":
            if scale == "row":       # a row is a condition on the overview, a day on its page
                return label, f"PER {_unit_upper(1)}" if unit == "day" else "PER CONDITION"
            return label, _RECEIPT_SCALES.get(scale, value.split(" · ")[0])
        return label, value.split(" · ")[0]
    return short


# --------------------------------------------------------------------------- #
# The hard-assignment cliff, said BEFORE the picture is read
# --------------------------------------------------------------------------- #


def cliff_status(payload: dict | None) -> str | None:
    """`"blocked"` when hard assignment cannot draw some population at all, `"narrow"` when it
    clears the cliff by less than `CLIFF_NARROW`, else `None`.

    SQD only, and only at hard assignment: the cliff is a property of the SQD posterior, so appD (no
    posterior) and HMM (its hard state is a decode, not this mixture) never have one, and soft assignment
    degrades smoothly. ONE rule for the single card and the matrix.

    BLOCKED ON EITHER SIDE (1 Nov 2026). It read only the short population's best posterior, so a two-population
    map whose LONG population won no jump -- a fit of one population, its two D's alike, every jump handed to the
    larger share -- drew an empty "Long jumps" panel and said nothing but n = 0 (free mNG's simulated condition).
    `never_most_likely` names every population that won no jump, whichever it is."""
    if not payload or payload.get("method", "sqd") != "sqd" or payload.get("assignment") != "hard":
        return None
    if not payload.get("hard_assignment_possible", True) or payload.get("never_most_likely"):
        return "blocked"
    best = payload.get("max_posterior_slowest")
    if best is not None and best < CLIFF_NARROW:
        return "narrow"
    return None


def _fitted_d_alike(payload: dict | None) -> bool:
    """Whether an SQD map's fitted populations include two whose D are within `bubble_view.SIMILAR_D_RATIO`
    (twofold): the bubbles' own rule for "fit one population fewer", so the map and the bubbles call the same fit
    the same way."""
    from .bubble_view import SIMILAR_D_RATIO

    D = sorted(float(d) for d in (payload or {}).get("D") or ())
    return any(a > 0 and b / a < SIMILAR_D_RATIO for a, b in zip(D, D[1:]))


def _fit_found_fewer(payload: dict | None) -> bool:
    """Whether an SQD map's fit found fewer populations than it was asked for, by the bubbles' two rules
    (`bubble_view.fit_warnings`): a population under `SMALL_POPULATION` (5 %) or two within twofold in D."""
    from .bubble_view import SMALL_POPULATION

    small = any(float(f) < SMALL_POPULATION for f in (payload or {}).get("fractions") or ())
    return small or _fitted_d_alike(payload)


def _empty_not_short(payload: dict | None) -> list:
    """The populations OTHER than the short one that won no jump under hard assignment (`never_most_likely`)."""
    return [k for k in (payload or {}).get("never_most_likely") or [] if k != "short"]


def map_fit_warnings(payload: dict | None) -> list:
    """The fit's own warnings under an SQD map of two or more populations: `bubble_view.fit_warnings`, the SAME rule
    and words the bubbles print (a population under 5 %, neighbours within twofold in D, each ending "fit one
    population fewer"), for this condition. Soft and hard alike: under soft a two-population fit that found one
    draws the SAME jumps twice, scaled by the two shares, which reads as two populations in one place."""
    method = (payload or {}).get("method", "sqd")
    if not payload or method not in ("sqd", "appd", "hmm") or int(payload.get("n_pop") or 2) < 2             or payload.get("split") == "manual":
        return []
    # THE APPARENT-D AND HMM MAPS TOO (3 Nov 2026, found by the overnight tester: free mNG's simulated condition drew
    # every localisation as "Slow" with no word): their own populations and shares, the same rule
    if method == "appd":
        D, f = payload.get("appd_D") or (), payload.get("appd_fractions") or ()
    elif method == "hmm":
        D, f = payload.get("D") or (), payload.get("occupancy") or ()
    else:
        D, f = payload.get("D") or (), payload.get("fractions") or ()
    if len(D) < 2 or len(D) != len(f):
        return []
    from .bubble_view import fit_warnings

    points = [{"column": 0, "style_index": 0, "population": k + 1, "d": float(d), "fraction": float(x)}
              for k, (d, x) in enumerate(sorted(zip(D, f)))]
    return fit_warnings(points, [str(payload.get("condition") or "this condition")])


def map_warning(payload: dict | None) -> str | None:
    """Everything one map must say before it is read: the cliff (`cliff_warning`, SQD) and the fit's own warnings
    (`map_fit_warnings`, SQD, apparent D and HMM), or `None`. An apparent-D map, which splits at 2 or 3 only, says
    where to read a one-population condition: its All tracks map."""
    fits = map_fit_warnings(payload)
    parts = ([cliff_warning(payload), no_switching_warning(payload), empty_state_warning(payload)]
             + [f"Check the fit: {w}." for w in fits] + [forced_count_text(payload)])
    if fits and (payload or {}).get("method") == "appd":
        parts.append(APPD_ONE_POPULATION)
    # exact-zero jumps (`sqd.zero_jumps`): the immobile population counted, or zeros kept on a lattice
    parts.append(((payload or {}).get("zero_jumps") or {}).get("note"))
    text = " ".join(p for p in parts if p)
    return text or None


def forced_count_text(payload: dict | None) -> str | None:
    """A map drawn at a count set by hand where Best (BIC) chooses fewer (`bubble_view.forced_count_warning`, ruled 4
    Nov 2026): "<condition>: Best (BIC) chooses 1 state here, ...; kept at the 2 you set." -- or None. The map is
    never redrawn at Best's count: the user decides."""
    from .bubble_view import forced_count_warning

    if not payload or not payload.get("bic_forced"):
        return None
    noun = "state" if payload.get("method") == "hmm" else "population"
    words = forced_count_warning(payload.get("n_pop"), payload["bic_forced"], noun)
    return f"{payload.get('condition') or 'This condition'}: {words}." if words else None


def no_switching_warning(payload: dict | None) -> str | None:
    """An HMM map whose fit saw no switching within a track (``no_switching``, `HMMFit.no_switching`): each track is
    decoded in one state for its whole length, so the maps say where slow and fast TRACKS are, not where molecules
    switch (ruled 4 Nov 2026: such a map is drawn, no longer refused). `None` otherwise."""
    if not payload or payload.get("method") != "hmm" or not payload.get("no_switching"):
        return None
    return ("No switching within tracks: the fitted chain never leaves a state, so each track is drawn in one state "
            "for its whole length. These maps show where slow and fast tracks lie, as the apparent-D maps do; they "
            "do not show where molecules switch.")


def empty_state_warning(payload: dict | None) -> str | None:
    """An HMM map at hard assignment whose decode made some state the most likely for no step
    (`never_most_likely`): its map is empty by construction, not too sparse (SQD's own are said by
    `cliff_warning`, appD's by `crossing_text`). `None` otherwise."""
    if not payload or payload.get("method") != "hmm" or payload.get("assignment") != "hard":
        return None
    never = list(payload.get("never_most_likely") or ())
    if not never:
        return None
    which = " and ".join(never)
    return (f"Hard assignment draws no step in the {which} state: under the fitted chain no step is most likely "
            f"{'there' if len(never) == 1 else 'in them'}, so {'its map is' if len(never) == 1 else 'their maps are'} "
            f"empty by construction. Re-run with soft assignment.")


# Where an apparent-D map's one-population condition is read: said after its fit warnings, by one map
# (`map_warning`) and by the grid (`map_warning_for`) alike (the grid left it out until 4 Nov 2026).
APPD_ONE_POPULATION = "An apparent-D map splits at 2 or 3 populations: for one, read its All tracks map."


def map_warning_for(payloads: dict, conditions: list, unit: str = "condition") -> str | None:
    """`map_warning` over several conditions: the cliff named per condition (`cliff_warning_for`), then every
    condition's fit warnings. On a replicate page (``unit="day"``) the fit is the condition's, said once."""
    parts = [cliff_warning_for(payloads, conditions, unit=unit)]
    if unit == "day":
        first = next((payloads.get(c) for c in conditions if payloads.get(c) is not None), None)
        parts.append(no_switching_warning(first))
        parts.append(empty_state_warning(first))
        warned = [first] if map_fit_warnings(first) else []
        parts += [f"Check the fit: {w}." for w in map_fit_warnings(first)] + [forced_count_text(first)]
    else:
        warned = [payloads.get(c) for c in conditions if map_fit_warnings(payloads.get(c))]
        for c in conditions:
            still = no_switching_warning(payloads.get(c))
            parts.append(f"{c}: {still}" if still else None)
        for c in conditions:
            empty = empty_state_warning(payloads.get(c))
            parts.append(f"{c}: {empty}" if empty else None)
        for c in conditions:
            parts += [f"Check the fit: {w}." for w in map_fit_warnings(payloads.get(c))]
        parts += [forced_count_text(payloads.get(c)) for c in conditions]
    if any((p or {}).get("method") == "appd" for p in warned):
        parts.append(APPD_ONE_POPULATION)              # once, however many conditions were warned
    # exact-zero jumps or tracks (`sqd.zero_jumps`, `appd_gamma.zero_tracks`): every condition's, as one map says its
    # own; a replicate page's fit is the condition's, said once
    for c in (conditions[:1] if unit == "day" else conditions):
        note = ((payloads.get(c) or {}).get("zero_jumps") or {}).get("note")
        parts.append(f"{c}: {note}" if note and unit != "day" else note)
    text = " ".join(p for p in parts if p)
    return text or None


def cliff_warning(payload: dict | None) -> str | None:
    """The one-condition warning text, or `None` when there is nothing to say."""
    status = cliff_status(payload)
    best = (payload or {}).get("max_posterior_slowest")
    if status == "blocked" and int((payload or {}).get("n_pop") or 2) == 2 and _empty_not_short(payload):
        from .bubble_view import SMALL_POPULATION

        pops = sorted(zip((float(d) for d in (payload or {}).get("D") or ()),
                          (float(f) for f in (payload or {}).get("fractions") or ())))
        if len(pops) == 2 and pops[1][1] < SMALL_POPULATION:
            # the long population holds (almost) nothing: free mNG's simulated condition, 0.0 % at 1.47 um2/s
            return (f"Hard assignment draws no long jump here: the long population holds {100 * pops[1][1]:.1f} % "
                    f"of the fit (D {pops[1][0]:.3g} µm²/s), so no jump is assigned to it and its map is empty. "
                    f"The fit found one population: draw this map at 1 population.")
        said = f" (fitted D {pops[0][0]:.3g} and {pops[1][0]:.3g} µm²/s)" if len(pops) == 2 else ""
        if _fitted_d_alike(payload):
            return (f"Hard assignment draws no long jump here: the two fitted populations have similar D{said}, so "
                    f"every jump goes to the larger share and the long-jump map is empty. The fit found one "
                    f"population: draw this map at 1 population.")
        return (f"Hard assignment draws no long jump here: no observed jump is long enough to be most likely "
                f"long{said}, so that map is empty. Re-run with soft assignment.")
    if status == "blocked" and int((payload or {}).get("n_pop") or 2) > 2:
        never = ", ".join((payload or {}).get("never_most_likely") or []) or "one population"
        return (f"Hard assignment cannot draw every panel here: {never} is the most "
                f"likely for no jump, so its map is empty. At three populations the middle one is a band of "
                f"jump lengths between two cuts. Re-run with soft assignment.")
    if status == "blocked":
        return (f"Hard assignment cannot draw the short-jump panel here: the "
                f"short population's best posterior is {best:.3f}, at or "
                f"below 0.5, so no jump is assigned to it and that map "
                f"is empty by construction. Re-run with soft assignment.")
    if status == "narrow":
        return (f"Hard assignment works only narrowly here: the short population's best posterior is {best:.3f}, barely "
                f"above the 0.5 needed to assign a jump to it, so this map is drawn at little better than "
                f"chance. Soft assignment degrades gradually.")
    return None


def cliff_warning_for(payloads: dict, conditions: list,
                      unit: str = "condition") -> str | None:
    """The same warning over several conditions, naming them. Blocked outranks narrow.

    With `unit="day"` the rows are one condition's replicate days, drawn under
    ONE fit, so the cliff is the same for every day by construction: it is said
    once, as that condition's own warning, rather than listed per day.
    """
    if unit not in ("condition", "day"):
        raise ValueError(f"unit must be 'condition' or 'replicate', got {unit!r}")
    if unit == "day":
        first = next((payloads.get(c) for c in conditions
                      if payloads.get(c) is not None), None)
        return cliff_warning(first) if first is not None else None
    blocked = [c for c in conditions if cliff_status(payloads.get(c)) == "blocked"]
    narrow = [c for c in conditions if cliff_status(payloads.get(c)) == "narrow"]
    # WHICH panel each blocked condition cannot draw: the short one (its best posterior at or below 0.5) or
    # another (a fit of one population at two, or a middle band at three) -- one sentence each, so neither is
    # explained by the other's reason.
    other = [c for c in blocked if _empty_not_short(payloads.get(c))
             and "short" not in ((payloads.get(c) or {}).get("never_most_likely") or [])]
    short = [c for c in blocked if c not in other]
    parts = []
    if short:
        parts.append(f"Hard assignment cannot draw the short-jump panel for "
                     f"{', '.join(short)}: the short population's best posterior "
                     f"is at or below 0.5 there, so no jump is assigned to it "
                     f"and that map is empty by construction. Re-run with soft assignment.")
    if other:
        named = ", ".join(f"{c} ({', '.join(_empty_not_short(payloads.get(c)))} jumps)" for c in other)
        fewer = [c for c in other if _fit_found_fewer(payloads.get(c))]
        advice = (f" The fit found one population fewer for {', '.join(fewer)} (a population under 5 % or two alike "
                  f"D): draw it at one population fewer." if fewer else " Re-run with soft assignment.")
        parts.append(f"Hard assignment cannot draw every panel for {named}: no jump is ever most likely there, so "
                     f"those maps are empty." + advice)
    if parts:
        return " ".join(parts)
    if narrow:
        return (f"Hard assignment works only narrowly for "
                f"{', '.join(narrow)}: the short population's best posterior "
                f"is under 0.55, barely above the 0.5 needed to assign a jump to it, so those maps are "
                f"drawn at little better than chance. Soft assignment degrades gradually.")
    return None


# --------------------------------------------------------------------------- #
# The figures line under each panel
# --------------------------------------------------------------------------- #


def population_picture(method: str, n_pop: int) -> dict:
    """The Populations row's schematic: example populations and where they cross, from
    the SAME functions the maps use (`spatialsqd.sqd_crossings`,
    `spatialappd.appd_crossings`), so the picture cannot cut somewhere the map would not.
    SQD's cuts are jump lengths in nm, appD's apparent D's in µm²/s."""
    three = int(n_pop) == 3
    if method == "sqd":
        from .spatialsqd import sqd_crossings

        D = {1: (0.85,), 3: (0.022, 0.12, 0.92)}.get(int(n_pop), (0.055, 0.85))
        f = {1: (1.0,), 3: (0.17, 0.30, 0.53)}.get(int(n_pop), (0.35, 0.65))
        tau = 0.02
        cuts = [1000.0 * c for c in sqd_crossings(D, f, tau)["cuts_um"]]
        return {"method": "sqd", "D": D, "fractions": f, "tau_s": tau, "cuts": cuts}
    from .spatialappd import appd_crossings

    D = (0.04, 0.22, 1.0) if three else (0.05, 0.85)
    f = (0.30, 0.25, 0.45) if three else (0.35, 0.65)
    return {"method": "appd", "D": D, "fractions": f, "n_steps": 4,
            "cuts": appd_crossings(D, f, 4)["cuts"]}


#: The Assignment picture's bars.
ASSIGNMENT_BINS = 24


def assignment_histogram(n_pop: int = 2, n_bins: int = ASSIGNMENT_BINS) -> dict:
    """The Assignment picture: the JUMP-DISTANCE HISTOGRAM of the example fit (`population_picture`) and how each
    bar is given to the populations (asked 16 Oct 2026: "show a jump distance histogram and how it assigns from
    that").

    Exact for that mixture, nothing drawn by hand. A bar is the share of jumps whose length falls in it,
    ``sum_k f_k [exp(-lo²/m_k) - exp(-hi²/m_k)]`` with ``m_k = 4 D_k τ`` (a 2-D jump's length law). SOFT gives each
    population its own part of the bar -- the posterior `spatialsqd` weights every jump with, summed over the bar.
    HARD gives every jump to the population that is most likely at its length (`spatialsqd.sqd_crossings`, the
    cuts where the curves cross), so a bar is one colour unless a cut falls inside it.

    ``{"edges_nm", "total" (per bar, summing to the share shown), "soft", "hard" (per bar, one share per population
    by ascending D, summing to 1), "cuts_nm", "D", "fractions", "tau_s"}``.
    """
    import numpy as np

    from .spatialsqd import sqd_crossings

    pic = population_picture("sqd", n_pop)
    D = np.asarray(pic["D"], dtype=float)
    f = np.asarray(pic["fractions"], dtype=float)
    tau = float(pic["tau_s"])
    m = 4.0 * D * tau                                            # µm²
    top = math.ceil(1000.0 * math.sqrt(-m.max() * math.log(0.01)) / 50.0) * 50.0 / 1000.0
    edges = np.linspace(0.0, top, int(n_bins) + 1)               # µm
    lo, hi = edges[:-1], edges[1:]

    def between(a, b):
        """Each population's share of all jumps with a length in [a, b] (µm)."""
        return f[:, None] * (np.exp(-np.square(a)[None, :] / m[:, None])
                             - np.exp(-np.square(b)[None, :] / m[:, None]))

    per_pop = between(lo, hi)
    total = per_pop.sum(axis=0)
    soft = per_pop / total
    crossing = sqd_crossings(D, f, tau)
    bounds = [0.0, *crossing["cuts_um"], math.inf]
    hard = np.zeros_like(soft)
    for (a, b), owner in zip(zip(bounds, bounds[1:]), crossing["windows"]):
        part = between(np.clip(lo, a, b), np.clip(hi, a, b)).sum(axis=0)
        hard[owner] += part / total
    # each population's own curve, on the bars' scale (its density times a bar's width): the soft split of a bar
    # is the ratio of the curves at its length, and the hard cut is where they cross
    r = np.linspace(0.0, top, 121)
    width = top / int(n_bins)
    curves = f[:, None] * (2.0 * r[None, :] / m[:, None]) * np.exp(-np.square(r)[None, :] / m[:, None]) * width
    return {"edges_nm": [1000.0 * e for e in edges], "total": [float(t) for t in total],
            "soft": [[float(v) for v in col] for col in soft.T],
            "hard": [[float(v) for v in col] for col in hard.T],
            "curve_nm": [1000.0 * v for v in r], "curves": [[float(v) for v in c] for c in curves],
            "cuts_nm": [1000.0 * c for c in crossing["cuts_um"]],
            "D": tuple(pic["D"]), "fractions": tuple(pic["fractions"]), "tau_s": tau}


#: The Long axis picture's two example cells, (length, width) in um. ILLUSTRATIVE, not
#: anybody's data: the ends of the 2-6 um range the question was asked about.
AXIS_PICTURE_CELLS = ((2.0, 0.9), (5.0, 0.9))
#: The example molecule's distance from each cell's tip, um.
AXIS_PICTURE_MARKER_UM = 0.25
#: Bins per cap and along the cylinder in the picture's outline.
AXIS_PICTURE_BINS = (2, 8)


def long_axis_picture(long_axis: str, model_length_um: float, model_width_um: float) -> dict:
    """The Long axis row's picture: where the drawn outline's bins fall inside a short and
    a long cell, and where one molecule `AXIS_PICTURE_MARKER_UM` from each cell's tip
    lands on the outline. Computed with `spatialsqd.long_axis_x` itself, the function the
    maps use, so the picture cannot lay a cell on differently from a real run; the bin
    edges inside a cell are that map inverted numerically."""
    from types import SimpleNamespace

    import numpy as np

    from .spatialsqd import long_axis_x

    cfg = SimpleNamespace(cell_length_um=float(model_length_um),
                          cell_width_um=float(model_width_um), long_axis=long_axis)
    model_r = cfg.cell_width_um / 2.0
    n_cap, n_cyl = AXIS_PICTURE_BINS
    model_edges = ([model_r * k / n_cap for k in range(n_cap)]
                   + [model_r + (cfg.cell_length_um - 2 * model_r) * k / n_cyl
                      for k in range(n_cyl)]
                   + [cfg.cell_length_um - model_r * (n_cap - k) / n_cap
                      for k in range(n_cap + 1)])
    cells = []
    for length, width in AXIS_PICTURE_CELLS:
        geometry = {"c": SimpleNamespace(length_um=length, max_width_um=width)}
        s = np.linspace(0.0, length, 4001)
        x = long_axis_x(s / length, cfg, cells=["c"] * len(s), geometry=geometry)
        edges = np.interp(model_edges, x, s)
        land = float(long_axis_x([AXIS_PICTURE_MARKER_UM / length], cfg, cells=["c"],
                                 geometry=geometry)[0])
        middle = float(long_axis_x([0.5], cfg, cells=["c"], geometry=geometry)[0])
        cells.append({"length_um": length, "width_um": width,
                      "edges_um": [float(e) for e in edges],
                      "pole_um": float(edges[n_cap]),
                      "far_pole_um": float(length - edges[-1 - n_cap]),
                      "lands_um": land, "midpoint_lands_um": middle})
    return {"long_axis": long_axis, "model_length_um": cfg.cell_length_um,
            "model_width_um": cfg.cell_width_um, "model_edges_um": model_edges,
            "n_cap_bins": n_cap, "marker_um": AXIS_PICTURE_MARKER_UM, "cells": cells}


def setting_help(key: str, state: dict) -> tuple[str, str] | None:
    """The Run dialog's on-demand explanation for a row: `(link text, tooltip text)`, or
    None for a row whose picture and line already say it. Words on demand, never printed:
    asked for after "why not measure from each cell's midpoint, relative to its length",
    which is exactly what one of the two choices already does."""
    if key == "split":
        unit = RANGE_UNITS.get(state.get("method", "sqd"), "")
        what = ("each jump's length" if state.get("method", "sqd") == "sqd"
                else "each track's mean apparent D")
        return ("What a range is", (
            f"Fitted (default): each condition is fitted and its maps are cut where the populations cross.\n\n"
            f"By hand: you set up to {MANUAL_RANGES_MAX} ranges of {what} ({unit}), and each range is one map. "
            f"The same ranges apply to every condition and nothing is fitted. A range includes its low end and "
            f"excludes its high end; ranges may not overlap. "
            f"A {'jump' if unit == 'nm' else 'track'} in no range is not drawn (the caption counts them).\n\n"
            f"The histogram shows the selected condition's values on a log axis. "
            f"\u201cFrom the fit\u201d puts the ranges where the fitted maps would cut."))
    if key == "axis":
        pic = long_axis_picture("normalised", state.get("cell_length_um", DEPOSIT_LENGTH_UM),
                                state.get("cell_width_um", DEPOSIT_WIDTH_UM))
        short, long_ = pic["cells"]
        anchored = long_axis_picture(
            "pole_anchored", pic["model_length_um"], pic["model_width_um"])["cells"][0]
        marker = int(round(pic["marker_um"] * 1000))
        return ("How a cell is laid onto the outline", (
            "Along the cell, every position is measured from the cell's tip as a fraction of "
            "the cell's length. That is the same as measuring from the cell's midpoint "
            "relative to its full length.\n\n"
            "Normalised (as in SMTracker 2): the fraction times the drawn length. Tips "
            "and midpoints line up, but everything is stretched with its cell, so a molecule "
            f"{marker} nm from the tip lands {short['lands_um'] * 1000:.0f} nm in from a "
            f"{short['length_um']:g} µm cell and {long_['lands_um'] * 1000:.0f} nm in from a "
            f"{long_['length_um']:g} µm one.\n\n"
            "Pole-anchored: each cap is placed at its own size and only the middle is "
            "stretched, so tips, caps and midpoints all line up and that molecule "
            f"lands {anchored['lands_um'] * 1000:.0f} nm in from either cell.\n\n"
            "Across the cell both choices are the same: the distance from the centre line "
            "divided by the cell's half-width, 0 on the axis and 1 at the edge."))
    if key in ("masking", "neff"):
        return ("What n_eff is, and what it costs", N_EFF_ABOUT)
    if key == "cells":
        return ("How the classes are made", (
            "Each cell's length is measured along its centre line. The cells of every "
            "condition in this run are pooled and cut into thirds, so “Long” means the same "
            "lengths in every row.\n\n"
            "Every class is drawn on the all-cells outline, so the colours compare bin for bin.\n\n"
            "The populations are still fitted on every cell; a class only chooses which "
            "jumps are drawn."))
    return None


def crossing_text(payload: dict) -> str:
    """Where an appD run was split, in words: the fitted populations' crossings, or the ranges set by hand."""
    if payload.get("split") == "manual":
        return manual_split_text(payload)
    cuts = payload.get("crossings_um2_s")
    never = payload.get("never_most_likely") or []
    if not cuts and never:
        # the populations never cross: one is the most likely at every D (found 3 Nov 2026 by the overnight tester:
        # this said "run the map again" right after a fresh Run)
        return (f"the fitted populations never cross: {', '.join(never)} is the most likely at no D, so its map is "
                f"empty and the fit has found one population")
    if not cuts:
        return "no crossing recorded (run the map again)"
    if float(cuts[0]) == 0:
        # THE IMMOBILE POPULATION OF EXACT-ZERO TRACKS (`spatialappd.appd_crossings`): its window ends at a cut at 0
        moving = [float(c) for c in cuts[1:]]
        text = "the slowest map holds the tracks whose apparent D is 0 (the immobile population)"
        if moving:
            text += (f"; the {payload.get('n_pop', len(cuts) + 1) - 1} that move are split where they cross: "
                     + " and ".join(f"{c:.3g}" for c in moving) + " \u00b5m\u00b2/s")
        if never:
            text += (f" ({', '.join(never)} is the most likely at no D, so its map is "
                     f"empty)")
        return text
    values = " and ".join(f"{c:.3g}" for c in cuts)
    text = (f"split where the {payload.get('n_pop', len(cuts) + 1)} fitted populations "
            f"cross: {values} \u00b5m\u00b2/s")
    top = max((float(d) for d in payload.get("appd_D") or ()), default=None)
    if top and max(float(c) for c in cuts) > 10.0 * top:
        # a crossing far beyond every fitted population (2.65e+03 um2/s on free mNG): no track lies there
        text += "; beyond every fitted population, so the fit found one population"
    if never:
        text += (f" ({', '.join(never)} is the most likely at no D, so its map is "
                 f"empty)")
    return text


def drawn_total(payload: dict) -> float:
    """The jumps (or localisations) one condition's maps were drawn from: its map of every jump where it has one (one
    SQD population, and every SQD split since 3 Nov 2026), else its panels summed -- a split's populations partition
    its jumps -- never the two added, which counted every jump twice."""
    panels = payload.get("panels") or {}
    if ALL_JUMPS_KIND in panels:
        return float(panels[ALL_JUMPS_KIND]["n_jumps"])
    return float(sum(p["n_jumps"] for p in panels.values()))


def figures_text(payload: dict, panel: dict) -> str:
    """`N unit · masked · too sparse`, the facts that decide how much to trust one panel.
    A panel that says what it counts (SDA: localisations, steps, step pairs) is counted
    in that; a value map always has a floor, so its masked count is always said."""
    unit_label = panel.get("count_unit") or payload.get("weight_unit", "weighted jumps")
    guarded = (payload["n_min"] > 0 or payload.get("neff_min") is not None
               or bool(panel.get("value_min")))
    masked = "" if not guarded else f" · {panel['n_masked']} masked"
    if payload.get("neff_min") is not None and panel.get("n_masked_neff"):
        masked += f" ({panel['n_masked_neff']} for n_eff)"
    note = " · too sparse to map" if panel.get("sparse") else ""
    # By track, the map counts tracks; the jump count stays first because it is
    # what the masking and the sparse rule read.
    tracks = (f" · ≈ {panel.get('n_weight', 0):,.0f} tracks"
              if payload.get("weighting") == "track" else "")
    # An adaptively smoothed map's widths are NOT added here: this line is also printed
    # under each exported panel, where it must stay one short line (a slides figure ran
    # off with them). The caption's "Value smoothing" row and each bin's hover say them.
    return f"{panel['n_jumps']:,.0f} {unit_label}{tracks}{masked}{note}"


def matrix_figures_text(payload: dict | None, panel: dict | None, error: str | None, *, drawn: bool = True,
                        n_pop=None, kind: str | None = None) -> str:
    """A matrix cell's line: the error NAMED if that condition raised; for a column its row's count never draws
    (``drawn`` False, `cell_drawn`), that it is not drawn at ``n_pop`` (`not_drawn_text`, with a Best row's BIC
    margin) -- never "waiting…" for a map that is not coming; "waiting…" before its result. A map its row's count
    DOES draw that is missing from maps made before it existed (``kind``: All jumps beside a split, 3 Nov 2026) says
    to run the maps again."""
    if error is not None:
        return error
    if not drawn:
        return " · ".join(empty_cell_lines(payload, kind, n_pop))
    if panel is None:
        return "waiting…"
    return figures_text(payload, panel)


# --------------------------------------------------------------------------- #
# The Run dialog: what each control means, what it costs, and what the run is handed
# --------------------------------------------------------------------------- #


def even_bins(value: float) -> int:
    """`spatial_sqd_map`'s own rounding, so the dialog's bin count is the one the run will really use:
    even, because folding pairs each bin with its reflection."""
    n = round(value)
    return max(2, n + (n % 2))


# -- THE BIN: ONE CAMERA PIXEL BY DEFAULT (ruled 30 Oct 2026) -------------------------------------------- #
#
# Asked: "bin size 100 standard. doesnt it make sense to do bin size standard 1 pixel? so it reads pixel size
# from data set?" ... "maybe 1 px is smartest. u compare same camera anyway". Both lab cameras are ruled at
# 106 nm, so on this lab's data the standard moves from 100 to 106 nm; on another camera it is that camera's
# pixel, READ from the loaded project and never typed. A bin chosen in nm stays one click away (`BIN_UNITS`),
# and every map made before (no `bin_camera_px` on its payload) still reads as the nm it was drawn at.

#: How the Run's bin is chosen: in camera PIXELS of the data set, or typed in nm.
BIN_UNITS = ("px", "nm")
#: THE STANDARD: one camera pixel of the data set (was 100 nm until 30 Oct 2026).
BIN_UNIT_DEFAULT = "px"
#: A bin chosen in camera pixels is this many of them.
BIN_CAMERA_PX = 1


class BinSizeError(ValueError):
    """A bin of camera pixels asked for where the data set records no pixel size. Refused BY NAME: a stand-in
    (100 nm, the old standard) would draw every condition at a size nobody chose and call it a pixel -- and
    an identical number across groups is the tell of exactly that fallback."""


def nm_text(nm) -> str:
    """A bin size in nm as every sentence prints it: ``100``, ``106``, ``107.5`` (u-track writes 106 nm as
    106.000003 in some projects; that is still 106)."""
    return f"{float(nm):.4g}"


def bin_number(nm):
    """A bin in nm as a config and a payload record it: a whole number stays an ``int`` (every bin typed on
    the slider, and every session written before 30 Oct 2026), a pixel's size keeps its decimals, since
    107.5 nm truncated to 107 would record a bin the map was not drawn at."""
    value = float(nm)
    return int(value) if value.is_integer() else value


def project_pixel(project) -> tuple:
    """``(pixel size in nm, note)`` of the LOADED project: the ONE size its every coordinate was converted
    at, ``project.params.pixel_size_nm``. Conditions recorded up to 1 % apart load at the loader's reference
    (the first condition's, `project._check_params_agree`) and leave each one's size in
    ``canonical.attrs["pixel_size_mixed"]``; then ``note`` says so in words, else it is None. Only if the
    loader kept no reference is the median of the recorded sizes used, and the note says that too.
    ``(None, None)`` when nothing records a pixel size -- `resolve_bin_nm` refuses that by name."""
    import statistics

    pixel = getattr(getattr(project, "params", None), "pixel_size_nm", None)
    attrs = getattr(getattr(project, "canonical", None), "attrs", None) or {}
    mixed = {str(k): float(v) for k, v in (attrs.get("pixel_size_mixed") or {}).items()}
    median = pixel is None and bool(mixed)
    if median:
        pixel = statistics.median(mixed.values())
    if pixel is None:
        return None, None
    pixel = float(pixel)
    note = None
    if mixed and max(mixed.values()) > min(mixed.values()):
        spread = 100.0 * (max(mixed.values()) / min(mixed.values()) - 1.0)
        each = ", ".join(f"{name} {nm_text(nm)} nm" for name, nm in mixed.items())
        note = (f"pixel sizes differ by up to {spread:.2f} % ({each}); 1 px is "
                + (f"their median, {nm_text(pixel)} nm" if median else
                   f"{nm_text(pixel)} nm, the size every coordinate was converted at"))
    return pixel, note


def resolve_bin_nm(unit: str, typed_nm, pixel_size_nm, *, pixels: float = BIN_CAMERA_PX) -> float:
    """THE ONE RULE for the bin the maps are drawn at, in nm: ``typed_nm`` for a bin chosen in nm,
    ``pixels`` camera pixels of ``pixel_size_nm`` for one chosen in pixels. A pixel bin with no pixel size to
    read is refused (`BinSizeError`, named), never drawn at a stand-in."""
    if unit not in BIN_UNITS:
        raise ValueError(f"the bin unit must be one of {BIN_UNITS}, got {unit!r}")
    if unit == "nm":
        if typed_nm is None:
            raise ValueError("a bin chosen in nm needs its size in nm")
        return float(typed_nm)
    if pixel_size_nm is None:
        raise BinSizeError(
            f"Bin size {pixels:g} px: no pixel size is known for this data set, so a camera pixel has "
            "no size in nm. Choose the bin in nm, or load the project with its pixel size.")
    pixel = float(pixel_size_nm)
    if not (math.isfinite(pixel) and pixel > 0):
        raise BinSizeError(f"Bin size {pixels:g} px: the data set's pixel size ({pixel_size_nm!r} nm) "
                           "is not a size. Choose the bin in nm.")
    return float(pixels) * pixel


def camera_pixel_bin(project, pixels: float = BIN_CAMERA_PX) -> dict:
    """``pixels`` camera pixels of the loaded project as the maps' bin, by `resolve_bin_nm`:
    ``{"bin_nm", "pixel_size_nm", "note"}`` (``note``: `project_pixel`'s, for mixed pixel sizes)."""
    pixel, note = project_pixel(project)
    nm = resolve_bin_nm("px", None, pixel, pixels=pixels)
    return {"bin_nm": nm, "pixel_size_nm": float(pixel), "note": note}


def _resolve_bin_config(project, config: dict) -> tuple:
    """``(config, stamp)``: a Run's config with its bin in nm, and what the payload records of how it was
    chosen. A bin chosen in camera pixels (``bin_camera_px``) is resolved HERE, from the project the worker
    holds (`camera_pixel_bin`), so the map is drawn at this data set's pixel whatever the dialog last
    saw; a ``bin_nm`` the dialog already resolved must agree, or the run is refused rather than drawn at
    one size and labelled with another. A config without ``bin_camera_px`` (a bin typed in nm, every
    session before 30 Oct 2026) passes through untouched."""
    config = dict(config)
    pixels = config.pop("bin_camera_px", None)
    if not pixels:
        return config, {}
    rule = camera_pixel_bin(project, pixels)
    asked = config.get("bin_nm")
    if asked is not None and not math.isclose(float(asked), rule["bin_nm"], rel_tol=1e-6):
        raise BinSizeError(
            f"Bin size {pixels:g} px: the Run asked for {nm_text(asked)} nm, but this project's pixel is "
            f"{nm_text(rule['pixel_size_nm'])} nm. Open Map Configuration again.")
    config["bin_nm"] = rule["bin_nm"]
    return config, {"bin_camera_px": bin_number(pixels), "bin_pixel_nm": rule["pixel_size_nm"],
                    "bin_pixel_note": rule["note"]}


def bin_text(payload: dict) -> str:
    """The bin AS CHOSEN, in words: ``106 nm (1 camera pixel)`` for one read from the data set's pixel,
    ``100 nm`` for one typed in nm -- and for every map made before 30 Oct 2026, which carries no unit."""
    nm = nm_text(payload["bin_nm"])
    pixels = payload.get("bin_camera_px")
    if not pixels:
        return f"{nm} nm"
    return f"{nm} nm ({pixels:g} camera pixel{'' if float(pixels) == 1 else 's'})"


def bin_choice_text(payload: dict) -> str | None:
    """What a caption adds about the bin beyond the grid it states: `bin_text` and, with the conditions'
    pixel sizes mixed, `project_pixel`'s note -- for a bin read from the camera pixel only. None for a bin
    typed in nm, whose caption stays what it was (the grid and its nm)."""
    if not payload.get("bin_camera_px"):
        return None
    note = payload.get("bin_pixel_note")
    return bin_text(payload) + (f"; {note}" if note else "")


def bin_unit_labels(pixel_size_nm) -> tuple:
    """The Bin size page's two pills: one camera pixel, showing the project's size, and the nm slider."""
    pixel = (f"{BIN_CAMERA_PX:g} px ({nm_text(pixel_size_nm)} nm)" if pixel_size_nm
             else f"{BIN_CAMERA_PX:g} px (no pixel size)")
    return pixel, "In nm"


def dialog_bin(unit: str, typed_nm, pixel_size_nm) -> dict:
    """What the Run dialog's state holds of the bin: ``{"bin_unit", "bin_nm", "bin_refusal"}``. ``bin_nm``
    is `resolve_bin_nm`'s answer, or None with the refusal's sentence in ``bin_refusal`` when a pixel bin has
    no pixel size: the page prints it and Run refuses it (`SpatialMapDialog.refusal`), so the refusal is
    shown, never replaced by a number."""
    try:
        nm = resolve_bin_nm(unit, typed_nm, pixel_size_nm)
    except BinSizeError as exc:
        return {"bin_unit": unit, "bin_nm": None, "bin_refusal": str(exc)}
    return {"bin_unit": unit, "bin_nm": bin_number(nm), "bin_refusal": ""}


def bin_page_warning(state: dict) -> str:
    """The Bin size page's amber line: the refusal when a pixel bin cannot be resolved, else `bin_warning`
    on the bin as resolved (one camera pixel never sits under 0.9 of one)."""
    if state.get("bin_refusal"):
        return "⚠ " + state["bin_refusal"]
    if state.get("bin_nm") is None:
        return ""
    return bin_warning(float(state["bin_nm"]), state.get("pixel_size_nm"))


def bin_warning(bin_nm: float, pixel_nm: float | None) -> str:
    """A WARNING, never a refusal: where the bin is drawn is the user's call, what it costs is not.

    Thresholds are the measurement's: below 30 nm is under every candidate localisation precision
    (30-76 nm), below 76 nm under the upper one, and under 0.9 camera pixels is
    `spatial_sqd_figure.py`'s own floor (0.9, not 1.0, so the shipped 98-100 nm against a 106 nm pixel is
    deliberate rather than marginal and the warning does not fire on it; the standard since 30 Oct 2026 is
    one camera pixel, which never does)."""
    bits = []
    if bin_nm < 30:
        bits.append("below every candidate localisation precision (30–76 nm)")
    elif bin_nm < 76:
        bits.append("below the upper candidate localisation precision (76 nm)")
    if pixel_nm and bin_nm < 0.9 * pixel_nm:
        bits.append(f"under 0.9 camera pixels ({0.9 * pixel_nm:.0f} nm)")
    return ("⚠ " + " · ".join(bits)) if bits else ""


def lag_text(lag) -> str:
    """A time lag as the receipt and the dialog's row say it."""
    if map_lag(lag) == "1-4":
        return "1\u20134 TOGETHER"
    lag = int(lag or 1)
    return f"{lag} FRAME" + ("" if lag == 1 else "S")


def _takes_section_fit(state: dict) -> bool:
    """The Run dialog's state with the box ticked on a fitted split (`map_fit_source.uses_section`): the counts and
    SQD's time lag are the section fit's, so their pages say so."""
    from .map_fit_source import uses_section

    return uses_section(state)


def setting_value_text(key: str, state: dict) -> str:
    """The value each nav row shows: CAPS for the words and the UNIT left exactly as it is (`100 nm` must
    not become `100 NM`, and upper-casing the micro sign gives M, which this project shipped once)."""
    if key == "split":
        return split_value_text(state)
    if key == "show":
        return {"angles": "TURNING BACK", "turning": "BEYOND CHANCE"}.get(state.get("hmm_show"), "STEPS")
    if key == "method":
        return METHOD_LABELS[state["method"]].upper()
    if key == "jump":
        return STEP_LABELS[STEP_POSITIONS.index(state["step_position"])].upper()
    if key == "assignment":
        return str(state["assignment"]).upper()
    if key == "fold":
        return "ON" if state["mirrored"] else "OFF"
    if key == "frame":
        return "MEASURED" if state["frame_source"] == "measured" else "DEPOSIT"
    if key == "cells":
        return str(state.get("cell_class", "all")).upper()
    if key == "axis":
        return ("POLE-ANCHORED" if state.get("long_axis") == "pole_anchored"
                else "NORMALISED")
    if key == "bin":
        if state.get("bin_unit") == "px":
            # the pill's own words, so the nav and the page say one thing
            return bin_unit_labels(state.get("pixel_size_nm"))[0]
        return f"{nm_text(state['bin_nm'])} nm"
    if key == "weighting":
        return "BY TRACK" if state.get("weighting") == "track" else "BY STEPS"
    if key == "cutoff":
        if state.get("max_locs") is None:
            return "OFF"
        return f"≤ {state['max_locs']}" + (" · CELLS" if state.get("cutoff_mode") == "cells" else "")
    if key == "neff":
        if state.get("neff_min") is None:
            return "OFF"
        return (f"< {state['neff_min']:g} · "
                + ("CELLS" if state.get("neff_unit") == "cells" else "TRACKS"))
    if key == "populations" and _takes_section_fit(state):
        from .map_fit_source import FIT_NAMES

        return f"{FIT_NAMES[state['method']].upper()} FIT"
    if key == "lag" and _takes_section_fit(state):
        return "SQD FIT"
    if key == "populations":
        by = set((state.get("n_pop_by_condition") or {}).values())
        if len(by) > 1:
            return "PER CONDITION"
        n_pop = next(iter(by)) if by else state.get("n_pop", 2)
        return "BEST (BIC)" if n_pop == BEST else str(int(n_pop))
    if key == "lag":
        return lag_text(state.get("lag_frames", 1))
    if key == "radius":
        return f"{int(state.get('radius_nm', 0))} nm"
    if key == "dwell":
        return f"≥ {int(state.get('min_steps', 0))} STEPS"
    if key == "adaptive":
        return "ON" if state.get("value_adaptive", True) else "OFF"
    if key == "masking":
        # Both rules read as MINIMUMS, what a bin needs to be drawn.
        count = state["n_min"] > 0
        neff = state.get("neff_min") is not None
        if not (count or neff):
            return "OFF"
        parts = []
        if count:
            # With both on, the count's unit is dropped: the nav column holds one line, and
            # a second line would make this dialog taller than every other page.
            parts.append(f"≥ {state['n_min']}" + ("" if neff else (
                " LOCS" if state.get("method") in PER_LOCALISATION_METHODS
                else "" if state.get("method") == "sda" else " JUMPS")))
        if neff:
            parts.append(f"n_eff ≥ {state['neff_min']:g} "
                         + ("CELLS" if state.get("neff_unit") == "cells" else "TRACKS"))
        return " · ".join(parts)
    return "OFF" if state["n_min"] <= 0 else str(state["n_min"])


def setting_readout(key: str, state: dict) -> str:
    """The live line under each picture in the Run dialog."""
    if key == "show":
        return {"angles": "Each state's pairs of steps at their turn: the share that turned back (0.5: no preference), "
                          "the data alone.",
                "turning": "Each state's pairs of steps at their turn: the share that turned back, minus the share "
                           "free molecules turn back at the same places (0: as free)."}.get(
            state.get("hmm_show"), "Where each state's steps are: their density per bin.")
    if key == "split":
        return split_readout(state)
    if key in ("populations", "lag") and _takes_section_fit(state):
        from .map_fit_source import box_help

        if key == "lag":
            return ("Each condition's jumps at the time lag the SQD fit was made at: its D is the apparent D at that "
                    "lag. Untick “Use the same D as the SQD fit” on the Populations page to choose one.")
        return box_help(state["method"])
    if key == "lag":
        lag = map_lag(state.get("lag_frames", 1))
        if lag == "1-4":
            return ("Jumps 1 to 4 frames apart, fitted together (SMTracker 2's \u201c1-4 together\u201d: one D and one set of "
                    "fractions for all four lags) and every lag's jumps drawn. Localisation error is not modelled, so "
                    "a population at the floor reads lower than at lag 1.")
        if lag == 1:
            return "Jumps between consecutive frames."
        return (f"Jumps {lag} frames apart, fitted and drawn. A diffusing population keeps its D; localisation "
                f"error alone reads {lag} times lower, so a thinning slow map reflects the floor and no extra state.")
    if key == "method":
        return {
            "sqd": "The same two-population fit the populations come from, read per step.",
            "appd": ("A per-track apparent diffusion, split where its fitted "
                     "populations cross, as in SMTracker 2."),
            "hmm": ("A fitted two-state Markov model, read per step, "
                    "with its own dwell times and switching rates."),
            "sla": ("SMTracker 2's Confined Map: where molecules dwell inside a moving circle, "
                    "in SMTracker 2's colours, with the chance rate on free tracks beside it."),
            "sda": ("SMTracker 2's SDA maps: where molecules are, and their mean speed, apparent D "
                    "and acceleration per bin, in SMTracker 2's colours. Apparent values, no fit."),
        }[state["method"]]
    if key == "radius":
        r = int(state.get("radius_nm", 0))
        pixel = state.get("pixel_size_nm")
        bits = [f"A molecule dwells while it stays within {r} nm of one of its own points."]
        if pixel:
            bits.append(f"{r / pixel:.2f} camera pixels (SMTracker 2 uses one pixel).")
        bits.append("A larger circle also flags more of a free walk: read every share "
                    "against its chance rate.")
        return " ".join(bits)
    if key == "dwell":
        n = int(state.get("min_steps", 0))
        tau = state.get("tau_s")
        span = f" ({n * tau * 1000:.0f} ms)" if tau else ""
        return (f"At least {n} consecutive steps{span} inside the circle count as a dwell. "
                f"SMTracker 2's default is 5.")
    if key == "adaptive":
        from .spatialsda import ADAPTIVE_CAP, ADAPTIVE_WIDTHS, VALUE_N_MIN

        floor = max(int(state.get("n_min", 0)), VALUE_N_MIN)
        if state.get("value_adaptive", True):
            ladder = ", ".join(f"{w:g}" for w in ADAPTIVE_WIDTHS[1:])
            return (f"Each value bin starts at h0 = max(σ, ½ value bin) and widens to "
                    f"{ladder} × h0 only where its kernel holds fewer than {floor} steps' "
                    f"worth; a bin still short at {ADAPTIVE_CAP:g} × h0, or with no step of "
                    f"its own, is grey. The localisation map is not affected.")
        return (f"One width, h0 = max(σ, ½ value bin), in every value bin; a bin with fewer "
                f"than {floor} steps of its own is grey.")
    if key == "populations" and (state.get("n_pop") == BEST
                                 or len(set((state.get("n_pop_by_condition") or {}).values())) > 1):
        return ("Each condition uses the count picked on its row, or the count BIC chooses under Best (BIC). "
                "The grid has a column for every map some condition draws; a condition with one population shows a "
                "note in the columns it does not draw.")
    if key == "populations":
        three = drawn_n_pop(state) == 3
        if state.get("method") == "sqd" and drawn_n_pop(state) == 1:
            return ("One population: one map of every jump, nothing split. For a protein of one state, or a "
                    "two-population fit that found one (its two D's alike).")
        if state.get("method") == "sqd":
            return ("Three populations: three maps, each jump shared by its posterior, after All jumps (every "
                    "jump once, as one population draws it). Under Hard the middle one is a band of jump lengths "
                    "and can be empty." if three else
                    "Two populations: the same fit the fractions come from, after All jumps (every jump once, as "
                    "one population draws it). Under Hard, one cut on jump length.")
        return ("Three populations: two cuts, where neighbours cross, and three maps. A "
                "population that is the most likely at no D gets an empty map."
                if three else
                "Two populations: one cut, where they cross. Each track goes to the side "
                "its mean apparent D falls on.")
    if key == "jump":
        text = {
            "A_start": "Counted where the jump began: long jumps read as peripheral.",
            "B_midpoint": "Counted halfway: pulls long jumps toward the axis.",
            "C_end": "Counted where the jump landed.",
        }[state["step_position"]]
        if state.get("method") == "sda":
            text += (" Speed and D only: SMTracker 2 draws them at the midpoint, and acceleration "
                     "sits where its two steps meet.")
        return text
    if key == "assignment":
        three = drawn_n_pop(state) == 3
        if drawn_n_pop(state) == 1:
            return "One population: every jump is in its one map, so Soft and Hard draw the same."
        if state["assignment"] == "soft":
            return ("Each bar is the jumps of one length, shared in the ratio of the fitted "
                    "curves at that length: every jump counts toward every map, in proportion.")
        cuts = assignment_histogram(3 if three else 2)["cuts_nm"]
        where = " and ".join(f"{c:.0f} nm" for c in cuts)
        return (f"Each jump goes wholly to the population most likely at its length: "
                f"{'two cuts' if three else 'one cut'} where the curves cross, the dashed "
                f"line ({where} in this example).")
    if key == "fold":
        return ("The four quarters are summed, so the map is symmetric by construction."
                if state["mirrored"] else "The cloud as it fell.")
    if key == "frame":
        # The chosen capsule prints its own number on itself, so a line here would restate it.
        return ""
    if key == "cells":
        cuts = state.get("class_cuts_um") or state.get("measured_cuts_um")
        if not cuts:
            return "Cell lengths could not be measured here, so every cell is drawn."
        n = state.get("measured_n_cells")
        pool = f" of the {n} cells measured" if n else ""
        return (f"Thirds{pool}: short under {cuts[0]:.2f} µm, long {cuts[1]:.2f} µm and "
                f"over. The populations are still fitted on every cell, so a class "
                f"changes only which jumps are drawn.")
    if key == "axis":
        if state.get("long_axis") == "pole_anchored":
            return ("Each cell's poles keep their own size, scaled by its width like the "
                    "short axis; only the middle is stretched. A pole bin is the same "
                    "distance from the tip in every cell.")
        return ("Every cell is stretched to the drawn length, as in SMTracker 2. A "
                "pole bin is a fraction of each cell, so in a long cell it reaches "
                "further from the tip.")
    if key == "bin":
        value = state["bin_nm"]
        if value is None:
            # a pixel bin with no pixel size: no grid to state; the page's warning line names
            # the refusal (`bin_page_warning`), once
            return ""
        value = float(value)
        nx = even_bins(state["cell_length_um"] * 1000.0 / value)
        ny = even_bins(state["cell_width_um"] * 1000.0 / value)
        bits = [f"{nx} × {ny} bins"]
        if state.get("bin_unit") == "px":
            bits.append(f"{BIN_CAMERA_PX:g} camera pixel = {nm_text(value)} nm, read from the data set")
            if state.get("pixel_note"):
                bits.append(state["pixel_note"])
        elif state.get("pixel_size_nm"):
            bits.append(f"{value / state['pixel_size_nm']:.2f} camera pixels")
        return " · ".join(bits)
    if key == "weighting":
        text = ("Every track counts once: a long track weighs as much as a short one."
                if state.get("weighting") == "track" else
                "Every jump counts: a long track counts once per jump.")
        if state.get("method", "sqd") == "sqd" and _takes_section_fit(state):
            # the fractions at the SQD fit's D are fitted at ITS weighting (`map_fit_source`)
            text += " The fractions at the SQD fit's D follow that fit's own weighting."
        elif state.get("method", "sqd") == "sqd":
            text += (" The SQD fit is track-weighted too." if state.get("weighting") == "track"
                     else " The SQD fit is step-weighted too, as in SMTracker 2.")
        return text
    if key == "cutoff":
        census = state.get("census") or []
        mode = state.get("cutoff_mode", "tracks")
        return cutoff_text(cutoff_cost(census, state.get("max_locs"), mode), mode,
                           state.get("max_locs"))
    if key == "neff":
        if state.get("neff_min") is None:
            return "Every bin is drawn, however few tracks built it."
        unit = _NEFF_UNIT_TEXT[state.get("neff_unit", "tracks")]
        return (f"Bins worth fewer than {state['neff_min']:g} equal {unit} are left blank "
                f"(grey). Provisional: the cutoff is not checked against a simulation.")
    if key == "masking":
        # One line for the page: the plain blanks are the count's, the hatched ones n_eff's.
        parts = []
        if state.get("method") == "sda":
            from .spatialsda import VALUE_N_MIN

            floor = max(int(state["n_min"]), VALUE_N_MIN)
            # ONE line: this page is the dialog's tallest, so a second line here makes the
            # card taller than every other page. Adaptive smoothing's page says the rest.
            parts.append(f"Blank: speed, D and acceleration under {floor} steps"
                         + ("' worth (Adaptive smoothing)" if state.get("value_adaptive", True)
                            else " (a mean of fewer is noise)")
                         + (f"; localisations under {state['n_min']}."
                            if state["n_min"] > 0 else "."))
        elif state["n_min"] > 0:
            what = ("localisations" if state.get("method") in PER_LOCALISATION_METHODS
                    else "jumps")
            parts.append(f"Blank: under {state['n_min']} {what}.")
        if state.get("neff_min") is not None:
            unit = _NEFF_UNIT_TEXT[state.get("neff_unit", "tracks")]
            parts.append(f"Hatched: n_eff under {state['neff_min']:g} {unit} (provisional).")
        return " ".join(parts) or "Every bin is drawn, however few jumps or tracks built it."
    value = state["n_min"]
    unit = "jumps" if state["method"] == "sqd" else "localisations"
    return ("Every bin is drawn." if value <= 0 else
            f"Bins holding fewer than {value} {unit} are left blank.")


def resolve_frame(measured: dict | None, measured_selected: bool) -> tuple[float, float, str]:
    """`(length_um, width_um, source)`. The measured capsule only when it exists AND was chosen; a frame
    that could not be measured is never quietly replaced by the deposit's under a measured label.

    A LENGTH CLASS DOES NOT CHANGE THE FRAME. Each class is drawn on the all-cells capsule, so a class's
    map and the all-cells map have the same bins and their densities compare bin for bin. Drawing each
    class on its own median capsule was tried first: the bin count followed the class, so the density per
    bin did too, and short cells read saturated against long ones for no reason in the data."""
    if measured_selected and measured is not None:
        return measured["cell_length_um"], measured["cell_width_um"], "measured"
    return DEPOSIT_LENGTH_UM, DEPOSIT_WIDTH_UM, "deposit"


def map_config(*, method: str, mirrored: bool, n_min: int, bin_nm: float | None, cell_length_um: float,
               cell_width_um: float, frame_source: str, step_position: str,
               assignment: str, weighting: str = "steps", max_locs=None,
               cutoff_mode: str = "tracks", split_from: str = "all", n_pop: int = 2,
               neff_min: float | None = None, neff_unit: str = "tracks",
               cell_class: str = "all", class_cuts_um=None,
               long_axis: str = "normalised", radius_nm: float | None = None,
               min_steps: int | None = None, value_adaptive: bool = True, lag_frames: int = 1,
               bin_unit: str = "nm", split: str = SPLIT_DEFAULT, ranges=None,
               n_pop_by_condition: dict | None = None, same_D: bool = False,
               fit_page: str | None = None, per_spot: bool = False, hmm_show: str = "steps") -> dict:
    """What `compute_spatial_map` is handed. A method's config carries only the keys its own compute takes:
    step position and assignment exist for SQD and HMM (both per-step with a posterior to soften) and are

    ``same_D`` is the box "Use the same D as the ... fit" (`map_fit_source`), kept for SQD, apparent D and HMM
    only: the RUN's, read by `map_fit_source.plan`, never handed to a job (`condition_config` drops it). Unticked by
    default here, so a script's map fits its own as before; the dialog opens ticked (`map_fit_source.TICK_DEFAULT`).
    OMITTED entirely under apparent diffusion.

    ``per_spot`` (6 Nov 2026, `OWN_PRECISION_KEY`): the map's OWN fit with each jump's (track's) own localisation
    error -- kept only where it applies (SQD, apparent D and HMM, the box unticked, a Fitted split), so a config without
    it is the map as before; SQD's lags 1-4 together are refused beside it.

    THE BIN: ``bin_unit="nm"`` (this function's default, so a script that states its nm keeps it) draws at
    ``bin_nm``; ``"px"`` -- the Run dialog's STANDARD since 30 Oct 2026 -- adds ``bin_camera_px`` and the
    worker resolves it from the project it holds (`_resolve_bin_config`): ``bin_nm`` may then be None, and a
    value the dialog resolved must be that project's."""
    if bin_unit not in BIN_UNITS:
        raise ValueError(f"bin_unit must be one of {BIN_UNITS}, got {bin_unit!r}")
    if bin_nm is None and bin_unit == "nm":
        raise ValueError("a bin chosen in nm needs its size (bin_nm)")
    config = {"method": method, "mirrored": bool(mirrored), "n_min": int(n_min),
              "bin_nm": None if bin_nm is None else bin_number(bin_nm),
              "cell_length_um": cell_length_um,
              "cell_width_um": cell_width_um, "frame_source": frame_source,
              "weighting": weighting,
              "max_locs": None if max_locs is None else int(max_locs),
              "cutoff_mode": cutoff_mode,
              "neff_min": None if neff_min is None else float(neff_min),
              "neff_unit": neff_unit,
              "cell_class": cell_class,
              # The cuts only mean something for a class; "all" carries none, so two
              # all-cell maps measured over different pools still pool.
              "class_cuts_um": (None if cell_class == "all" or class_cuts_um is None
                                else tuple(float(c) for c in class_cuts_um)),
              "long_axis": long_axis}
    if bin_unit == "px":
        config["bin_camera_px"] = BIN_CAMERA_PX
    if cell_class not in CELL_CLASSES:
        raise ValueError(f"cell_class must be one of {CELL_CLASSES}, got {cell_class!r}")
    if cell_class != "all" and config["class_cuts_um"] is None:
        raise ValueError("a length class needs the cuts measured over the cells")
    if long_axis not in LONG_AXES:
        raise ValueError(f"long_axis must be one of {LONG_AXES}, got {long_axis!r}")
    if method == "hmm" and hmm_show in ("angles", "turning"):
        # TURNING BACK (22 Nov 2026, `spatialangles`; "angles" the data alone, "turning" beyond chance): kept only when
        # asked, so a steps map's config is as before
        config["hmm_show"] = hmm_show
    if method in ("sqd", "hmm"):
        config["step_position"] = step_position
        config["assignment"] = assignment
        # appD learns its cut from every track (`compute_spatial_map_appd`).
        config["split_from"] = split_from
    if method == "sqd":
        if map_lag(lag_frames) not in MAP_LAGS:
            raise ValueError(f"lag_frames must be one of {MAP_LAGS}, got {lag_frames}")
        config["lag_frames"] = map_lag(lag_frames)
    if method == "sda":
        # SDA places each step (its speed and D), but fits nothing: no posterior to
        # soften, no split to learn. Its value maps may widen their kernel where steps
        # are few (`spatialsda` divergence 9); ON is the Run's default.
        config["step_position"] = step_position
        config["value_adaptive"] = bool(value_adaptive)
    if method in N_POP_METHODS:
        config["n_pop"] = check_n_pop(method, n_pop)
        # THE SECTION'S D (`map_fit_source`): kept only while ticked, so a config without it is a map's own fit
        if same_D:
            config["same_D"] = True
            # WHICH PAGE of the section (subpages): said only when it is not the section's own
            if fit_page and fit_page != method:
                config["fit_page"] = str(fit_page)
        # THE MAP'S OWN FIT PER SPOT (`OWN_PRECISION_KEY`): only where the map fits its own
        if per_spot and not same_D and not (method in RANGE_UNITS and split == "manual"):
            if method == "sqd" and map_lag(lag_frames) == "1-4":
                raise ValueError(OWN_PRECISION_LAG_REFUSAL)
            config[OWN_PRECISION_KEY] = True
        # EACH CONDITION ITS OWN COUNT, or Best (`condition_config` hands each job its own)
        if n_pop_by_condition:
            config["n_pop_by_condition"] = {str(c): check_n_pop(method, v) for c, v in n_pop_by_condition.items()}
    if method in RANGE_UNITS:
        # FITTED or BY HAND (`SPLITS`): a manual split carries its ranges, checked here so the Run refuses a bad
        # range by name before any job is sent. Only SQD and apparent-D maps split by hand: an HMM map's states are
        # decoded, and its compute takes no split (handing it one failed every HMM map, 3 Nov 2026).
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        config["split"] = split
        if split == "manual":
            config["ranges"] = [list(r) for r in check_ranges(method, ranges)]
            if method == "sqd" and map_lag(lag_frames) == "1-4":
                raise ValueError("a manual split cuts jump lengths at one time lag; lags 1-4 together divide each "
                                 "jump by its lag, so pick one lag")
    if method == "sla":
        # NO DEFAULT: the dialog always holds both (v2's: the pixel, 5 steps).
        if radius_nm is None or min_steps is None:
            raise ValueError("a confinement map needs its radius and minimum dwell")
        config["radius_nm"] = float(radius_nm)
        config["min_steps"] = int(min_steps)
    return config


# --------------------------------------------------------------------------- #
# What the dialog's diagrams share with the run
# --------------------------------------------------------------------------- #


def is_blank(count: float, n_min: int) -> bool:
    """A bin holding fewer than `n_min` is undefined (left blank), never drawn as zero; 0 means off."""
    return n_min > 0 and count < n_min


#: The SLA pictures' example track, in µm along / across a 2.94 x 1.00 µm cell (across from
#: the axis): seven points jittering inside about 40 nm, then six free steps of ~250 nm.
#: INVENTED, and said so; which points dwell is decided by `confinement.track_dwells` at the
#: dialog's radius and dwell, never typed.
SLA_EXAMPLE_UM = (
    (0.62, 0.05), (0.66, 0.03), (0.63, 0.08), (0.67, 0.07), (0.64, 0.02), (0.68, 0.06),
    (0.65, 0.04), (0.92, -0.10), (1.18, 0.12), (1.45, -0.08), (1.72, 0.15), (1.98, -0.05),
    (2.24, 0.10),
)


def sla_example(radius_nm: float, min_steps: int) -> dict:
    """The example track's points and which of them dwell, by the real detector:
    ``{"points": [(along_um, across_um)], "dwelling": [bool per point], "has_dwell"}``."""
    import numpy as np

    from .confinement import track_dwells

    xy = np.asarray(SLA_EXAMPLE_UM, dtype=float)
    _taus, indexes = track_dwells(xy, np.arange(len(xy)), radius_um=float(radius_nm) / 1000.0,
                                  min_steps=int(min_steps))
    flagged = set()
    for idx in indexes:
        flagged.update(int(i) for i in np.asarray(idx).ravel())
    dwelling = [i in flagged for i in range(len(xy))]
    return {"points": [tuple(map(float, p)) for p in SLA_EXAMPLE_UM],
            "dwelling": dwelling, "has_dwell": any(dwelling)}


#: The SDA picture's example track, in µm along / across a 2.94 x 1.00 µm cell (across from
#: the axis): short steps, then long ones, with one turn back. INVENTED, and said so; its
#: speeds and accelerations are computed by `spatialsda` itself at `SDA_EXAMPLE_TAU_S`.
SDA_EXAMPLE_UM = (
    (0.45, 0.10), (0.50, 0.06), (0.53, 0.12), (0.58, 0.08), (0.80, -0.10), (1.10, 0.02),
    (1.42, -0.16), (1.30, 0.10), (1.66, 0.18), (2.02, -0.04), (2.30, 0.12),
)
SDA_EXAMPLE_TAU_S = 0.02
#: The picture's key and its line under the cell.
SDA_EXAMPLE_KEY = ("line width = speed of that step", "ring = acceleration where two steps meet")


def sda_example_line(example: dict) -> str:
    """The line under the SDA picture: the example's own range of speeds, computed."""
    speeds = example["speed"]
    return (f"speed {min(speeds):.0f}–{max(speeds):.0f} µm/s at τ "
            f"{SDA_EXAMPLE_TAU_S * 1000:.0f} ms · each map averages these per bin · "
            f"invented example")


def sda_example() -> dict:
    """The SDA settings' picture, computed by the maps' own functions: the example
    track's ``points``, each step's ``speed`` (µm/s) and ``diffusion`` (µm²/s), and each
    pair of steps' ``acceleration`` (µm/s²) with ``acc_at``, the index of the point the
    two steps share -- where the map puts it. Computed once: the dialog repaints it."""
    if _SDA_EXAMPLE_CACHE:
        return _SDA_EXAMPLE_CACHE[0]
    import pandas as pd

    from .spatialsda import pair_table, step_table

    pts = SDA_EXAMPLE_UM
    table = pd.DataFrame({"uid": "example", "frame": list(range(len(pts))),
                          "x_um": [p[0] for p in pts], "y_um": [p[1] for p in pts],
                          "l_norm": [p[0] / DEPOSIT_LENGTH_UM for p in pts],
                          "d_norm": [p[1] / (DEPOSIT_WIDTH_UM / 2.0) for p in pts]})
    steps = step_table(table, SDA_EXAMPLE_TAU_S)
    pairs = pair_table(table, SDA_EXAMPLE_TAU_S)
    _SDA_EXAMPLE_CACHE.append({
        "points": [tuple(map(float, p)) for p in pts],
        "speed": [float(v) for v in steps["speed_um_s"]],
        "diffusion": [float(v) for v in steps["diffusion_um2_s"]],
        "acceleration": [float(v) for v in pairs["acceleration_um_s2"]],
        "acc_at": [int(f) for f in pairs["frame"]]})
    return _SDA_EXAMPLE_CACHE[0]


_SDA_EXAMPLE_CACHE: list = []


#: The Adaptive smoothing picture's invented field: steps per value bin, a peak at
#: midcell thinning to nothing at the poles and toward the walls. INVENTED, and said so.
ADAPTIVE_EXAMPLE_PEAK = 40.0
ADAPTIVE_EXAMPLE_SPREAD = 0.22
ADAPTIVE_EXAMPLE_KEY = ("grey = h0", "colour = widened to 1.5, 2, 3 × h0 (darker = wider)",
                        "hatched = blank")


def adaptive_example(*, bin_nm: float, cell_length_um: float, cell_width_um: float,
                     adaptive: bool, n_min: int = 0) -> dict:
    """The Adaptive smoothing page's picture, decided by the run's own rule: an INVENTED
    count field on the dialog's grid (value bins `spatialsda.VALUE_BIN_FACTOR` x the bin),
    put through `adaptive_smoothing` -- or, off, through the fixed floor -- exactly as a
    value map is. h0 is half a value bin (the dialog does not know sigma; it is the usual
    case). Returns the fine grid's ``inside``, ``multiple`` (each drawn bin's width in h0,
    None where grey or outside), ``counts`` (steps in each bin's value bin) and the totals
    the line under the picture reads."""
    import numpy as np

    from .spatialappd import SpatialAppDConfig, local_half_width
    from .spatialsda import VALUE_N_MIN, value_bin_factor

    nx = even_bins(cell_length_um * 1000.0 / float(bin_nm))
    ny = even_bins(cell_width_um * 1000.0 / float(bin_nm))
    f = value_bin_factor(nx, ny)
    cfg = SpatialAppDConfig(cell_length_um=cell_length_um, cell_width_um=cell_width_um,
                            n_bins_long=nx, n_bins_short=ny, smooth_bins=1.0)
    xc, yc = cfg.centres()
    xx, yy = np.meshgrid(xc, yc)
    inside = np.abs(yy) <= local_half_width(xx, cfg)
    # the field per VALUE bin, then written onto its small bins as a value map's is
    ux = (np.arange(nx // f) + 0.5) / (nx // f)
    vy = ((np.arange(ny // f) + 0.5) / (ny // f)) * 2.0 - 1.0
    blocks = np.round(ADAPTIVE_EXAMPLE_PEAK
                      * np.exp(-((ux[None, :] - 0.5) / ADAPTIVE_EXAMPLE_SPREAD) ** 2)
                      * (1.0 - 0.6 * vy[:, None] ** 2))
    counts = np.repeat(np.repeat(blocks, f, axis=0), f, axis=1)
    floor = max(int(n_min), VALUE_N_MIN)
    bin_long, bin_short = cell_length_um / nx, cell_width_um / ny
    h0 = max(bin_long, bin_short) * f / 2.0
    if adaptive:
        chosen = adaptive_smoothing(counts, counts, counts, inside, h0_um=h0,
                                    bin_long_um=bin_long, bin_short_um=bin_short,
                                    floor=floor, factor=f)
        grey = inside & (chosen["index"] < 0)
        multiple = np.where(grey, np.nan, chosen["h_um"] / h0)
    else:
        grey = inside & (counts < floor)
        multiple = np.where(grey, np.nan, 1.0)
    drawn = inside & ~grey
    return {"nx": nx, "ny": ny, "factor": f, "floor": floor, "h0_nm": h0 * 1000.0,
            "adaptive": bool(adaptive),
            "inside": inside.tolist(), "counts": counts.tolist(),
            "multiple": [[float(m) if ok else None for m, ok in zip(row, orow)]
                         for row, orow in zip(multiple.tolist(), drawn.tolist())],
            "n_h0": int((drawn & (multiple <= 1.0 + 1e-9)).sum()),
            "n_widened": int((drawn & (multiple > 1.0 + 1e-9)).sum()),
            "n_grey": int(grey.sum())}


def adaptive_example_line(example: dict) -> str:
    """The line under the Adaptive smoothing picture: what the run's rule did to it."""
    if not example["adaptive"]:
        return (f"h0 {example['h0_nm']:.0f} nm · {example['n_h0']} bins drawn · "
                f"{example['n_grey']} blank, under {example['floor']} steps · invented")
    return (f"h0 {example['h0_nm']:.0f} nm · {example['n_h0']} bins kept it · "
            f"{example['n_widened']} widened · {example['n_grey']} blank · invented")

