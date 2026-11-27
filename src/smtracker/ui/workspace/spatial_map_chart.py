"""The SQD map -- WHERE each fitted population's jumps happen, drawn in each
cell's own frame, as a second tab on the Populations condition drill-in.

**No new estimator.** This is the SAME two-population SQD fit the fractions
come from, read per STEP instead of per condition: `spatialsqd.
spatial_sqd_map`, whose own docstring calls it "the catalogue target -- one
projected table in, jump-density grids out", written for a panel that can
supply a table and scalar controls and nothing else. That is this panel.
Nothing in this file computes a fit; it configures one, runs it off the GUI
process, and draws what comes back.

**It reads the project ALREADY LOADED, and that is a correctness
requirement, not an optimisation.** `compute_condition` -- the front door
`scripts/spatial_sqd_figure.py` uses -- re-reads the project off disk, and
therefore needs a day map that covers every movie on disk. After the
onboarding group board is confirmed, `project.day_maps` is no longer that
mapping, and a series the user HELD OUT is correctly absent from it. Going
through that door both failed loudly on the first real project (see
`compute_spatial_map`) and would have drawn this one map from a different
set of series than every other page in the workspace.

**Painted natively, not through `spatialsqd.draw_maps`.** That half is
matplotlib, and this shell has no figure canvas in it anywhere -- the bubble
chart, the Apparent-D charts, the SuperPlots and the Dynamics charts are all
hand-painted from numbers for the same reason (`bubble_chart.py`'s own
module docstring records `window.py` using matplotlib the first time and
being corrected). So COMPUTE is reused as it stands and DRAW is this file. That
split has a real cost and it is paid rather than assumed away: every rule
`draw_maps` enforces about an honest map has to hold here too, and each one
is named at the place it is done.

**What the render spec demands, and where each is honoured**

- *The estimate is separated from the display.* Where a jump is drawn, how
  it is assigned, whether the map is folded and `n_min` are MEASUREMENT:
  they live in the Run dialog and travel in the payload. The colormap is
  DISPLAY: it lives in the Visibility tool and changes nothing but pixels.
  That line is why `n_min` is not a Visibility control -- it masks bins AND
  re-weights the smoothing, so it changes the numbers.
- *Fold sums, never maxes.* `spatialsqd.fold_counts`, reused, not
  reimplemented -- taking the max is the legacy `plotHeatmap` bias this
  project already records.
- *Colour limits over the whole panel set BEFORE drawing.* `_panel_limits`
  makes one pass over both grids before either is painted, and the shared
  limit is the DEFAULT, asked for directly ("one top for both"). Per-panel
  stays available and is not the default: two panels on their own limits
  look equally saturated while standing for different densities.
- *Sparse bins undefined, never zero, and never fed to the kernel.*
  `n_min`, applied as `render.py` applies it -- a masked bin is drawn grey,
  is kept out of the Gaussian, and the rest is count-weighted so a hole does
  not drag its neighbours down. `spatialsqd` has no `n_min` of its own; this
  is the one number the GUI map gains over the deposited one, which is
  exactly why it is a Run-dialog control and is printed in the caption.
- *The capsule outline is analytic.* `spatialsqd.capsule_outline`'s own
  closed form, as a `QPainterPath` -- never contoured off the bin grid,
  which staircases at the caps and draws nothing along the walls.
- *Every panel carries its caption.* Bin, bandwidth, `n_min` with its masked
  count, fold, scale, assignment, step position, N and the colormap --
  `_caption_for`, modelled on `render.MapGrid.caption`.

**The geometry is STATED, not settled here.** `docs/gui_tool_views.md` lists
the map geometry as one of the open questions the GUI must not answer on its
own: the MSD deposit draws 44 x 18 bins on a 2.94 x 1.00 um cell, the
a published two-protein deposit 30 x 10 on 2.879 x 1.06. This view uses
`scripts/spatial_sqd_figure.py`'s own shipped defaults so the picture
matches what that script draws, and prints them under every map so two
pictures of one quantity cannot be stacked without saying so.
"""
from __future__ import annotations

import math

from qtpy.QtCore import (
    QEasingCurve,
    QPauseAnimation,
    QPointF,
    QRectF,
    QSequentialAnimationGroup,
    Qt,
    QTimer,
    QVariantAnimation,
    Signal,
)
from qtpy.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QFontMetrics,
    QFontMetricsF,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
)
from qtpy.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ...factor_layout import columns, nested
from ...spatial_map_view import PROFILE_METHODS, profile_card_payload, sparse_text
from ...spatial_map_view import (
    ASSIGNMENTS,
    CELL_CLASSES,
    OWN_PRECISION_LABELS,
    OWN_PRECISION_LAG_REFUSAL,
    OWN_PRECISION_NOTE,
    LONG_AXES,
    cell_class_range_text,
    long_axis_picture,
    setting_help,
    MAX_LOCS_MAX,
    MAX_LOCS_MIN,
    NEFF_MIN_MAX,
    NEFF_MIN_MIN,
    NEFF_UNITS,
    DAYS_HINT,
    DEPOSIT_LENGTH_UM,
    DEPOSIT_WIDTH_UM,
    MAP_LAGS,
    MAP_N_POPS,
    MAP_N_POP_DEFAULT,
    MAP_N_POPS_BY_METHOD,
    map_n_pops,
    BEST,
    drawn_n_pop,
    kinds_for_counts,
    panel_keys_for_counts,
    panel_titles_for_counts,
    MANUAL_RANGES_DEFAULT,
    MANUAL_RANGES_MAX,
    RANGE_KINDS,
    RANGE_UNITS,
    SPLITS,
    SPLIT_DEFAULT,
    range_shares,
    ranges_from_fit,
    ranges_refusal,
    KINDS_BY_METHOD,
    kinds_for,
    panel_keys_for,
    panel_titles_for,
    METHOD_LABELS,
    METHODS,
    RENDERINGS,
    SLA_DWELL_RANGE,
    SLA_RADIUS_RANGE_NM,
    sla_example,
    N_EFF_ABOUT,
    WEIGHTINGS,
    SCALE_LEVELS,
    SPLIT_FROM,
    STEP_LABELS,
    STEP_POSITIONS,
    caption_for,
    map_warning,
    map_warning_for,
    even_bins,
    figures_text,
    assignment_histogram,
    is_blank,
    map_config,
    matrix_caption_rows,
    spatial_receipt,
    matrix_figures_text,
    cell_drawn,
    row_n_pop,
    neff_bar_mark,
    neff_display_panel,
    MATRIX_SCALES,
    bar_layout,
    matrix_bar_note,
    matrix_limits,
    neff_scale_top,
    neff_tops,
    NEFF_SCALE_TOP,
    masking_field,
    neff_pair,
    pair_colour_index,
    PAIR_KEY,
    PAIR_TEXT,
    neff_hover_text,
    panel_limits,
    resolve_frame,
    setting_name,
    setting_readout,
    setting_value_text,
    length_histogram,
    smooth_display,
    population_picture,
    default_cmap_for,
    grid_title,
    paint_field,
    panel_unit,
    sda_example,
    sda_example_line,
    SDA_EXAMPLE_KEY,
    adaptive_example,
    adaptive_example_line,
    ADAPTIVE_EXAMPLE_KEY,
    n_min_unit,
    bar_number,
    BIN_UNITS,
    BIN_UNIT_DEFAULT,
    bin_page_warning,
    bin_unit_labels,
    dialog_bin,
)
from ...colour_ramps import V2_HOT_KEY, v2_hot_ramp
from ...map_fit_source import METHODS as SAME_D_METHODS
from ...map_fit_source import PAGE_NAMES as SAME_D_PAGE_NAMES
from ...map_fit_source import gone_text as same_d_gone_text
from ...map_fit_source import TICK_DEFAULT as SAME_D_DEFAULT
from ...map_fit_source import box_text as same_d_text
from ...map_fit_source import populations_note as same_d_note
from ...spatialsqd import DEFAULT_CMAP
from ..theme import typography as ty
from ..widgets.controls import Button, FloatField, NumberField, Segmented, Slider
from .ramp_dialog import _combo_css  # the workspace's one combo-box style
from ..widgets.dialogs import CloseButton, OverlayDialog
from .dynamics_chart import _ChartTooltip
from ..widgets.primitives import Card, Checkbox, FitScroll, fit_label, label
from .bubble_chart import CARD_GAP, CARD_PAD
from .qc_chart import _FLAG as _FLAG_INK
from .section_placeholder import SectionPlaceholder
from .subpage_rows import PageChoice
from .toolbar import rem, text_px
from ..widgets.primitives import discard
from ..widgets.settings import InfoLink as _InfoLink
from ..widgets.settings import SettingNav as _SettingNav
from ..widgets.settings import SettingRow as _SettingRow  # noqa: F401

__all__ = ["SpatialMapDialog", "SpatialMapMatrix", "SpatialMapSection"]

#: Sampled from the real colormaps at 32 steps and written down here rather
#: than imported: `matplotlib.cm` would pull matplotlib into the UI, which
#: this shell does nowhere. 32 steps is ample for a 30 x 10 grid -- the BINS
#: quantise this picture, not the ramp.
#:
#: `jet` is the DEFAULT by the PI's ruling, and its cost is recorded rather
#: than quietly fixed: 108 of its 255 steps run BACKWARDS in CIE L*, so two
#: different densities print at the same lightness and it does not survive
#: greyscale. `magma` reversed (light = low) is one of the alternatives
#: already rendered for the manuscript (`manuscript/sqdmap/figures_magma_r/`)
#: and is monotonic in L*. No verdict between them was ever written down, so
#: this offers both and the caption says which was used -- it does not
#: settle it.
CMAP_RAMPS: dict[str, tuple[str, ...]] = {
    "jet": (
        "000080", "0000a4", "0000c8", "0000ed", "0004ff", "0024ff", "0044ff", "0064ff",
        "0088ff", "00a8ff", "00c8ff", "02e8f4", "1fffd7", "39ffbe", "53ffa4", "6dff8a",
        "8aff6d", "a4ff53", "beff39", "d7ff1f", "f4f802", "ffdb00", "ffbd00", "ff9f00",
        "ff7e00", "ff6000", "ff4300", "ff2500", "ed0400", "c80000", "a40000", "800000",
    ),
    "magma": (
        "fcfdbf", "fceeb0", "fde0a1", "fed194", "fec185", "feb27a", "fea36f", "fd9467",
        "fb835f", "f8745c", "f3655c", "ec5860", "e24d66", "d6456c", "ca3e72", "bd3977",
        "ae347b", "a1307e", "942c80", "882781", "792282", "6d1d81", "601880", "54137d",
        "451077", "38106c", "2a115c", "1e1149", "130d34", "0a0822", "030312", "000004",
    ),
}

#: `scripts/spatial_sqd_figure.py`'s own shipped defaults, so this view draws
#: the same picture that script does. 98 x 100 nm bins: above every candidate
#: localisation precision in this project (30-76 nm) and above 0.9 camera
#: pixels, so the map never displays structure finer than the measurement
#: resolves. EVEN counts, because folding pairs each bin with its reflection
#: (`SpatialSQDConfig` refuses an odd one). See the module docstring on why
#: the geometry is stated rather than chosen here.
#: The BIN, in nanometres, because that is the unit the choice is made in:
#: it is read against the camera pixel (106 nm here) and the localisation
#: precision (30-76 nm, unmeasured). `spatial_sqd_map` takes it this way and
#: derives the even bin counts itself. 100 nm on the deposit's capsule gives
#: 30 x 10, which is what `scripts/spatial_sqd_figure.py` draws.
#:
#: The range is bounded BELOW by the measurement rather than by taste: at
#: 30 nm the bin is under a third of a camera pixel and well under every
#: candidate localisation precision, so the map would display structure the
#: data cannot resolve. The floor is a WARNING, not a refusal -- see
#: `SpatialMapDialog._on_bin`.
#:
#: **THE STANDARD IS NO LONGER 100 nm BUT ONE CAMERA PIXEL** (ruled 30 Oct 2026,
#: `spatial_map_view.BIN_UNIT_DEFAULT`): the Bin size page opens on "1 px (106 nm)",
#: read from the data set, and the slider -- the nm choice, one click away -- sits at
#: that pixel. `BIN_NM_DEFAULT` is only where the slider stands before a pixel size is
#: known; a 1 px Run never draws at it (`spatial_map_view.resolve_bin_nm` refuses).
BIN_NM_DEFAULT = 100
BIN_NM_MIN = 30
BIN_NM_MAX = 400

#: `n_min`'s slider range. **0 means OFF -- every bin is drawn, which is what
#: `draw_maps` does today and therefore what the deposited panels show.** It
#: is not "n_min = 0": `render.make_map` REFUSES `n_min < 1` outright, since
#: a threshold of zero masks nothing and is not a threshold. For scale,
#: `render.DEFAULT_N_MIN["density"]` is 5 -- a jump-density bin is merely
#: sparse where a value bin would be noise, which is why the density default
#: is 5 and not the value path's 20.
N_MIN_MAX = 40
N_MIN_DEFAULT = 0

#: `dynamics_chart._WARN_COLOR`, the same amber this shell already uses for
#: a caveat that must be read before the numbers beside it. Restated rather
#: than imported: `dynamics_chart` is a sibling of this file, not a base for
#: it, and one chart reaching into another's private palette is the kind of
#: coupling this workspace has deliberately avoided between chart modules.
_WARN_COLOR = "#c05621"

#: The Run dialog's settings, in nav order: internal key, the name on the
#: row, and ONE LINE saying what the control does. Fifteen words is the
#: house limit -- `compute.CONTROL_HELP`'s own median is fourteen, and this
#: dialog previously ran to fifty-four a control, which is what sent the
#: explanation into pictures instead.
#:
#: **THREE ROW SETS, one per computational method.** "Jump drawn at" and
#: "Assignment" are SQD concepts -- a jump has two endpoints to choose
#: between and a posterior to soften; apparent diffusion is a per-track
#: scalar with neither (see `smtracker.spatialappd`'s own module docstring).
#: So those two rows do not exist at all under Apparent Diffusion rather
#: than being shown disabled -- a row that is always "N/A" answers a
#: question nobody asked instead of not asking it. HMM keeps the full SQD
#: set: it is ALSO per-step (so a step still has two endpoints to place)
#: and ALSO carries a posterior to soften (`smtracker.spatialhmm`'s own
#: module docstring -- the "hard" state there is a posterior argmax too,
#: just from a temporal decode instead of a static mixture density).
_KEYS_BY_METHOD = {
    "sqd": ("method", "populations", "split", "lag", "jump", "assignment", "fold", "frame", "cells",
            "axis", "bin", "masking", "weighting", "cutoff"),
    "appd": ("method", "populations", "split", "fold", "frame", "cells", "axis", "bin", "masking",
             "weighting", "cutoff"),
    # Show (22 Nov 2026, `spatialangles`): where each state's steps are, or where it turns back beyond chance
    "hmm": ("method", "populations", "show", "jump", "assignment", "fold", "frame", "cells", "axis", "bin",
            "masking", "weighting", "cutoff"),
    # Confinement (SLA): v2's two detector choices instead of the populations; a track
    # belongs to its set whole, so there is no jump to place and no posterior to soften.
    "sla": ("method", "radius", "dwell", "fold", "frame", "cells", "axis", "bin", "masking",
            "weighting", "cutoff"),
    # SDA (v2's spatial distribution maps): a step is placed (speed and D), so "Jump drawn
    # at" stays; nothing is fitted, so no populations, assignment or split. Its value maps
    # may widen their kernel where steps are few: "Adaptive smoothing", beside the bins.
    "sda": ("method", "jump", "fold", "frame", "cells", "axis", "bin", "adaptive",
            "masking", "weighting", "cutoff"),
}
#: The HMM map's Show, as the config says it (`spatial_map_view.compute_spatial_map`'s ``hmm_show``) and as the pills
#: read.
HMM_SHOWS = ("steps", "angles", "turning")
HMM_SHOW_LABELS = ("Steps", "Turning back", "Beyond chance")
_SETTING_DESC = {
    "show": ("Steps: where each state's steps are. Turning back: where it turns back (the data alone). Beyond "
             "chance: where it turns back more than a free molecule would at the same place (0 on the map)."),
    "method": "Which fit produces the labels this map colours by.",
    "jump": "A jump is two points. Which end gets counted?",
    "assignment": "Give each jump to one population, or split it between both.",
    "fold": "Add the cell's four quarters on top of each other.",
    "frame": "Every cell is drawn onto one outline. Whose?",
    "cells": "Draw every cell, or only the shortest, middle or longest third.",
    "axis": "Stretch each whole cell to the outline, or keep its poles their own size.",
    "bin": "How fine the grid is. The square is one camera pixel.",
    "masking": "Blank out bins that cannot be trusted: too few jumps, or built by too few "
               "independent tracks or cells.",
    "weighting": "Count every jump (standard), or every track once. The SQD fit follows.",
    "cutoff": "Leave out tracks that are suspiciously long. Drag the line.",
    "populations": "How many populations are fitted. Each gets a map; the dashed "
                   "lines are where the fitted curves cross.",
    "split": "Cut where the fitted populations cross, or set the ranges yourself on the "
             "histogram: each range is one map.",
    "lag": "How many frames apart a jump's two ends are. The fit and the maps follow.",
    "radius": "How close a molecule must stay to one of its own points to be dwelling.",
    "dwell": "How many steps in a row it must stay that close.",
    "adaptive": "Widen the speed, D and acceleration maps' smoothing only where a bin's "
                "kernel holds too few steps for a mean.",
}

#: v2's SLA colours (`spatialsla.SLA_COLOURS`), for the dialog's pictures.
_SLA_FREE_INK, _SLA_MIXED_INK, _SLA_DWELL_INK = "#0072BD", "#00FF00", "#FF0000"

#: The n_eff view's ramp: viridis, sampled at 32 steps like the ramps above. A DIFFERENT
#: look from the density map on purpose -- it is a different quantity (how many
#: contributors, not where the jumps are), and jet is kept for density by the PI's ruling.
#: Perceptually uniform and not red/green, so a colour-blind reader reads it too.
NEFF_RAMP: tuple[str, ...] = (
    "440154", "470d60", "48186a", "482374", "472e7c", "453882", "424186", "3e4a89",
    "3a548c", "365d8d", "32658e", "2e6d8e", "2b758e", "287d8e", "25848e", "228c8d",
    "1f948c", "1e9c89", "20a386", "25ab82", "2eb37c", "3aba76", "48c16e", "58c765",
    "6ccd5a", "7fd34e", "93d741", "a8db34", "c0df25", "d5e21a", "eae51a", "fde725",
)

#: viridis as a third DENSITY ramp too (asked: "we add viridis as well to the colour maps"):
#: the same samples the n_eff view paints with. Perceptually uniform, not red/green.
CMAP_RAMPS["viridis"] = NEFF_RAMP

#: v2's SDA ramp, ``flip(hot(1024))``: white at zero, through yellow and red, to black at
#: the top. COMPUTED from MATLAB's own formula (`colour_ramps.v2_hot_ramp`), not typed,
#: and the SDA method's default (`spatial_map_view.DEFAULT_CMAP_BY_METHOD`).
CMAP_RAMPS[V2_HOT_KEY] = v2_hot_ramp()

#: The user's CUSTOM ramps (`smtracker.colour_ramps`), keyed "custom:<name>", registered by the
#: window from the per-user store. Kept apart from CMAP_RAMPS so a built-in name is never
#: shadowed.
CUSTOM_RAMPS: dict[str, tuple[str, ...]] = {}


def register_custom_ramps(ramps) -> None:
    """Replace the custom ramps the maps can paint with (`colour_ramps.CustomRamp` list)."""
    CUSTOM_RAMPS.clear()
    CUSTOM_RAMPS.update({ramp.key: ramp.ramp() for ramp in ramps})


def ramp_for(key: str) -> tuple[str, ...] | None:
    """The 32 colours `key` names (built-in or custom), or None for a key nothing names."""
    return CMAP_RAMPS.get(key) or CUSTOM_RAMPS.get(key)
_NEFF_UNIT_SHORT = {"tracks": "tracks", "cells": "cells"}


#: The diagrams' own palette. NOT the map's colormap -- see `_SettingArt`.
from ..widgets.settings import ACCENT_INK as _ACCENT_INK  # noqa: E402 - one palette
from ..widgets.settings import CUT_INK as _CUT_INK  # noqa: E402
from ..widgets.settings import LONG_INK as _LONG_INK  # noqa: E402
from ..widgets.settings import SHORT_INK as _SHORT_INK  # noqa: E402
#: The Bin masking pair's CONTRIBUTOR colours -- the one exception to "inks and greys":
#: there a colour IS the contributor n_eff counts (a track under Tracks, a cell under Cells),
#: and the low bins are read off it. Muted, categorical, never the map's jet, no red beside
#: a green. `_contributor_ink` extends it, so any number of contributors stay distinct.
_CONTRIBUTOR_INKS = ("#4e79a7", "#f28e2b", "#b07aa1", "#76b7b2", "#edc948", "#ff9da7",
              "#9c755f", "#bab0ac")
#: The pair's geometry under the Bin masking cell: its key (room for two lines, so the
#: longer Cells key may wrap without the page changing height), each square, the two
#: label lines under a square, and the gap between the cell and the pair.
_PAIR_KEY_H = 44.0
_PAIR_SIDE = 140.0
_PAIR_LABEL_H = 88.0
_PAIR_GAP = 12.0

#: The weighting diagram's worked example (the same one as in
#: `notes/spatial-map-cutoff-decisions.html`): three tracks on a 5 x 3 grid,
#: in a 300 x 180 frame. A and B are short (3 jumps each); C is long (12 jumps,
#: all starting in one bin). By steps a bin holds its jumps; by track each jump
#: weighs 1 / its track's jumps, so every track adds up to 1.
_W_TRACKS = (
    ((30, 30), (92, 24), (86, 92), (108, 104)),
    ((272, 34), (266, 96), (214, 152), (196, 170)),
    ((130, 72), (168, 108), (138, 112), (172, 68), (128, 96), (176, 92), (146, 64),
     (160, 116), (132, 84), (170, 80), (140, 104), (164, 70), (150, 96)),
)
#: bin -> (by-steps value, by-track value, its label in each mode)
_W_BINS = {
    (0, 0): (1, 1 / 3, "1", "⅓"), (1, 0): (1, 1 / 3, "1", "⅓"), (1, 1): (1, 1 / 3, "1", "⅓"),
    (4, 0): (1, 1 / 3, "1", "⅓"), (4, 1): (1, 1 / 3, "1", "⅓"), (3, 2): (1, 1 / 3, "1", "⅓"),
    (2, 1): (12, 1, "12", "1"),
}
_W_TOTALS = (18.0, 3.0)

#: The capsule's true proportion, so the dialog illustrates the cell it is
#: configuring rather than a box of convenient shape.
_ART_ASPECT = DEPOSIT_LENGTH_UM / DEPOSIT_WIDTH_UM


#: One Fold flip, in ms; the picture plays two (across midcell, then across the long
#: axis) when Fold is toggled -- `_SettingArt._animate_fold`.
_FOLD_FLIP_MS = 200

#: How strongly a clicked spot's tracks are drawn while the pointer is on the map: faint
#: enough that the bins under them read (asked for -- the tracks covered the very bins
#: being hovered), full strength again once the pointer leaves for the card.
OVERLAY_HOVER_OPACITY = 0.3

#: One asymmetric cloud for the fold diagram, fixed for the same reason.
_FOLD_CLOUD = tuple(
    (0.16 + 0.30 * ((i * 37 % 61) / 61) + 0.09 * ((i * 17 % 29) / 29),
     -0.46 + 0.34 * ((i * 23 % 43) / 43) + 0.14 * (((i * 11 % 19) / 19) - 0.5))
    for i in range(72)
)

#: The one jump `_draw_jump` has always drawn, as the two points a step
#: spans -- 30%/16% of the capsule box to 60%/52%. Module-level so the
#: landing-cell table below can be computed once rather than re-derived
#: inside the paint method on every repaint.
_JUMP_A_FRAC = (0.30, 0.16)
_JUMP_B_FRAC = (0.60, 0.52)
_JUMP_MID_FRAC = ((_JUMP_A_FRAC[0] + _JUMP_B_FRAC[0]) / 2.0,
                  (_JUMP_A_FRAC[1] + _JUMP_B_FRAC[1]) / 2.0)
#: Deliberately coarse -- six columns, four rows -- the same "schematic, not
#: a data preview" choice every diagram in this class makes; real bins are
#: far finer. `_JUMP_LANDING` is each option's CELL (a floor() into this
#: grid, the same operation a real bin assignment performs at finer
#: resolution) and its CENTRE as a fraction of the box, which is what the
#: animated highlight slides between.
_JUMP_COLS, _JUMP_ROWS = 6, 4


def _jump_landing() -> dict:
    out = {}
    for key, (fx, fy) in zip(STEP_POSITIONS,
                             (_JUMP_A_FRAC, _JUMP_MID_FRAC, _JUMP_B_FRAC)):
        gx = min(_JUMP_COLS - 1, int(fx * _JUMP_COLS))
        gy = min(_JUMP_ROWS - 1, int(fy * _JUMP_ROWS))
        out[key] = ((gx, gy),
                   ((gx + 0.5) / _JUMP_COLS, (gy + 0.5) / _JUMP_ROWS))
    return out


#: `step_position -> ((cell_x, cell_y), (centre_fx, centre_fy))`.
_JUMP_LANDING = _jump_landing()

#: Vertical space BELOW the plate for the "Jump drawn at" time axis, in px. The
#: plate (`_SettingArt.HEIGHT`) is untouched, so every diagram -- the jump's cell
#: included -- is drawn at the same size as before; the widget is this much
#: taller on EVERY page, so switching pages does not move the readout under it.
_JUMP_AXIS_BAND = 44.0

#: The one-shot intro on the "Jump drawn at" page, in ms: hold on the start,
#: slide to the end, hold on the end, slide BACK to the start (asked for: end ->
#: start once it has gone start -> end), hold on the start, then the resting picture.
_JUMP_INTRO_HOLD_MS = 150
_JUMP_INTRO_SLIDE_MS = 450
#: The whole intro, three holds and two slides.
_JUMP_INTRO_MS = 3 * _JUMP_INTRO_HOLD_MS + 2 * _JUMP_INTRO_SLIDE_MS

#: THE "Computational method" PICTURE: one molecule that sits BOUND for six steps and
#: then DIFFUSES away for six, with one odd step in each half -- a long hop while bound
#: (step 3) and a short step while free (step 8). The three methods label exactly
#: those two steps differently, which is the whole of why they draw different maps:
#:
#:   SQD   each jump by its OWN length, so both odd steps flip;
#:   HMM   each step from the whole track, where states persist, so they do not;
#:   appD  ONE apparent D for the whole track, so every step gets one label and the
#:         bound half vanishes from the slow map (the switching-blur this project
#:         measured: appD recovers about half of the SQD condition difference).
#:
#: Points are (along, across) fractions of the capsule's box, 0 at the top wall.
_METHOD_TRACK = (
    (0.200, 0.42), (0.215, 0.54), (0.230, 0.42), (0.245, 0.53), (0.310, 0.38),
    (0.325, 0.50), (0.340, 0.39), (0.440, 0.66), (0.530, 0.36), (0.545, 0.46),
    (0.640, 0.70), (0.730, 0.40), (0.820, 0.64),
)
#: The jump length (in cell widths) that splits short from long in this schematic.
_METHOD_CUT = 0.15
#: The states the chain recovers: bound for six steps, free for six. Hand-given --
#: they are the TRUTH the example was built from, which the HMM reads back because
#: one odd step is cheaper to explain as noise than as two switches.
_METHOD_HMM_STATES = (0,) * 6 + (1,) * 6
#: How each method's picture is captioned under the step strip.
_METHOD_CAPTION = {
    "sqd": "SQD: each jump labelled by its own length",
    "hmm": "HMM: each step labelled from the whole track",
    "appd": "Apparent D: one D for the whole track",
    "sla": "SLA: where the track stays inside one moving circle",
    "sda": "SDA: speed per step, acceleration per pair of steps",
}
#: The intro: a hold, the walk along the track, the colouring, a hold (ms).
_METHOD_INTRO_HOLD_MS = 200
_METHOD_INTRO_MS = 1800
#: Within the intro's 0..1: the walk ends here, the colouring runs to `_COLOUR_END`.
_METHOD_WALK_END = 0.62
_METHOD_COLOUR_END = 0.80


def _method_step_lengths() -> list[float]:
    """Each step's length in the example track, in cell widths (the box is
    `_ART_ASPECT` widths long, so an along-axis fraction is scaled by it)."""
    out = []
    for (u0, v0), (u1, v1) in zip(_METHOD_TRACK, _METHOD_TRACK[1:]):
        out.append(math.hypot((u1 - u0) * _ART_ASPECT, v1 - v0))
    return out


def _method_labels(method: str) -> tuple[int, ...]:
    """0 = slow / short, 1 = fast / long, per step, as `method` labels the example.

    SQD and appD are COMPUTED from the drawn track (a cut on each jump's length, and on
    the track's root-mean-square jump), so the picture cannot claim more than its own
    geometry shows; the HMM's are the states the example was built from."""
    lengths = _method_step_lengths()
    if method == "sqd":
        return tuple(int(r > _METHOD_CUT) for r in lengths)
    if method == "appd":
        msd = sum(r * r for r in lengths) / len(lengths)
        return (int(msd > _METHOD_CUT ** 2),) * len(lengths)
    if method == "hmm":
        return _METHOD_HMM_STATES
    if method in ("sla", "sda"):
        # SLA and SDA draw their own examples (`_SettingArt._draw_sla`, `_draw_sda`); on
        # this track they read every step as one class, so a method switch still animates.
        return (1,) * len(lengths)
    raise ValueError(f"no picture for method {method!r}")

def _capsule_path(box: QRectF) -> QPainterPath:
    """`spatialsqd.capsule_outline`'s closed form, as a `QPainterPath`: two
    semicircular caps joined by straight walls.

    Module-level so the MAP and the Run dialog's own diagrams draw the same
    shape from the same code -- a dialog that illustrated a slightly
    different cell than the one it is configuring would be worse than no
    illustration.
    """
    r = box.height() / 2.0
    path = QPainterPath()
    path.moveTo(box.left() + r, box.top())
    path.lineTo(box.right() - r, box.top())
    path.arcTo(QRectF(box.right() - 2 * r, box.top(), 2 * r, 2 * r), 90, -180)
    path.lineTo(box.left() + r, box.bottom())
    path.arcTo(QRectF(box.left(), box.top(), 2 * r, 2 * r), 270, -180)
    path.closeSubpath()
    return path


def _mix(a: QColor, b: QColor, f: float) -> QColor:
    return QColor(round(a.red() * f + b.red() * (1 - f)),
                  round(a.green() * f + b.green() * (1 - f)),
                  round(a.blue() * f + b.blue() * (1 - f)))


class _CapsuleChart(QWidget):
    """One population's map, inside one analytic capsule.

    `clicked` fires on a left click only when the owner made it clickable
    (`set_clickable`) -- the Spatial Map section's grid does, so a condition's
    maps open its replicate days; nothing else does.

    The outline is `spatialsqd.capsule_outline`'s closed form as a
    `QPainterPath` -- two semicircular caps joined by straight walls --
    rather than a contour of the bin mask, which staircases at the caps and
    draws nothing along the walls (that function's own note). It doubles as
    the clip, so no bin can paint outside the cell.

    A bin is one filled rect drawn OPAQUE. Translucent rects overlapping by
    half a pixel leave a darker seam on every shared edge, which reads as a
    lattice the data does not have; the masked grey is therefore pre-blended
    against the card rather than drawn with alpha.

    **The capsule keeps the CELL's proportions, never the widget's.** A box
    drawn to whatever space is going makes a 2.94 x 1.00 µm cell look stubby
    or stretched, and this map is read for position along exactly those two
    axes.
    """

    clicked = Signal()
    #: A bin was clicked, (ix, iy) in the grid's own indices (row 0 = the bottom
    #: wall), only when `set_spot_clickable` is on -- the day page's maps.
    spot_clicked = Signal(int, int)

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self._clickable = False
        self._spot_clickable = False
        #: The clicked spot's tracks, as box-fraction polylines (largest first),
        #: and the clicked bin -- drawn over the picture, nothing computed here.
        self._overlay: dict | None = None
        #: The pointer is inside the capsule, so the overlay is drawn faded (`OVERLAY_HOVER_OPACITY`).
        self._overlay_hovered = False
        #: A magnifying lens that follows the pointer (`spot_page._Loupe`): the Spatial Map's
        #: replicate page's alone. None everywhere else, so nothing else changes.
        self.loupe = None
        #: Bumped whenever what the map SHOWS changes (panel, rendering, tracks view, overlay,
        #: its hover fading) -- never on a hover outline: the lens keys its magnified picture
        #: on it, so it redraws that picture only when it is out of date.
        self.content_version = 0
        #: The panel on screen, for the hover readout (`neff_hover_text`).
        self._panel: dict | None = None
        #: The hover readout: shadcn's chart tooltip, the one every other chart here
        #: shows on a data hover (`dynamics_chart._ChartTooltip`), not Qt's native one.
        self._tip = _ChartTooltip(t, self, indicator="square")
        self._tip_bin: tuple | None = None
        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("background:transparent;")
        self.setMinimumHeight(90)
        # The capsule's height follows its WIDTH, so the layout is told that
        # rather than handing this widget a tall box it can only centre a
        # short capsule inside. `paintEvent` still fits to whichever of the
        # two is binding, so a card too short to honour this is drawn
        # correctly (smaller), never stretched.
        policy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        self._grid: list[list[float]] | None = None
        self._masked: list[list[bool]] | None = None
        self._sparse = False
        self._n_jumps = 0.0
        self._lo = 0.0
        self._top = 0.0
        self._ramp: tuple[str, ...] = CMAP_RAMPS[DEFAULT_CMAP]
        self._aspect = DEPOSIT_LENGTH_UM / DEPOSIT_WIDTH_UM
        #: SMOOTH mode's picture, built once per `set_panel` rather than per
        #: paint (resampling is arithmetic over ~50k pixels). Masked pixels
        #: are transparent here; the grey rects are drawn over them sharp.
        self._smooth = False
        self._image: QImage | None = None
        #: A grid cell its row's count has no map for (`spatial_map_view.cell_drawn`): nothing is painted, not even
        #: the outline, which would read as an empty map.
        self._absent = False

    def set_absent(self, on: bool) -> None:
        """Paint nothing here: the row's count draws no map in this column (`SpatialMapMatrix._redraw`)."""
        if bool(on) != self._absent:
            self._absent = bool(on)
            self.content_version += 1
            self.update()

    def set_tracks(self, paths, ink: str | None) -> None:
        self.content_version += 1
        self._set_tracks(paths, ink)

    def _set_tracks(self, paths, ink: str | None) -> None:
        """The Tracks view (Confinement (SLA) only): ``paths`` are polylines in BOX
        FRACTIONS (u from the left pole, v from the bottom wall, `spatialsla.
        track_polylines`), painted in ``ink`` instead of the grid. ``None`` leaves the
        Tracks view; an empty list on a panel that has no tracks says why."""
        if paths is None:
            self._tracks = None
        else:
            path = QPainterPath()
            for line in paths:
                if len(line) < 2:
                    continue
                path.moveTo(line[0][0], 1.0 - line[0][1])
                for u, v in line[1:]:
                    path.lineTo(u, 1.0 - v)
            self._tracks = (path, QColor("#" + ink) if ink else None, len(paths))
        self.update()

    def set_panel(self, panel: dict | None, field: str, limits, ramp,
                  aspect: float, smooth: bool = False) -> None:
        self.content_version += 1
        self._tracks = None
        self._panel = panel
        self._grid = None if panel is None else panel[field]
        self._masked = None if panel is None else panel["masked"]
        #: the n_eff view's outlined bins (masked by the run, still coloured)
        self._outline = None if panel is None else panel.get("outline")
        self._sparse = bool(panel and panel.get("sparse"))
        self._n_jumps = float(panel["n_jumps"]) if panel else 0.0
        lo, hi = limits if limits else (0.0, 0.0)
        changed = aspect != self._aspect
        self._lo, self._top, self._ramp, self._aspect = lo, hi, ramp, aspect
        self._smooth = smooth
        self._image = self._build_image() if smooth else None
        if changed:
            # `heightForWidth` just started answering differently, and Qt
            # caches that per layout pass.
            self.updateGeometry()
        self.update()

    def _build_image(self) -> QImage | None:
        """The smooth picture: `spatial_map_view.smooth_display`'s bilinear
        resample, coloured by the ramp INTERPOLATED between its stops (the
        32-step ramp that suits flat bins would band a smooth gradient)."""
        if self._grid is None or self._masked is None or self._top <= self._lo:
            return None
        values = smooth_display(self._grid, self._masked)
        h, w = len(values), len(values[0])
        stops = [QColor("#" + hexcol) for hexcol in self._ramp]
        last = len(stops) - 1
        lut = []
        for i in range(256):
            pos = i / 255.0 * last
            k = min(last - 1, int(pos)) if last > 0 else 0
            f = pos - k
            c = _mix(stops[k + 1], stops[k], f) if last > 0 else stops[0]
            lut.append(c.rgba())
        img = QImage(w, h, QImage.Format_ARGB32)
        img.fill(0)
        for iy, row in enumerate(values):
            # Row 0 is the BOTTOM of the cell; image row 0 is the top.
            y = h - 1 - iy
            for ix, v in enumerate(row):
                if v is None:
                    continue
                f = self._fraction(v)
                img.setPixel(ix, y, lut[round(f * 255)])
        return img

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return max(self.minimumHeight(), round(width / self._aspect))

    def set_clickable(self, on: bool) -> None:
        """Offer a click (and say so with the cursor) -- only where the owner
        has somewhere to go."""
        self._clickable = bool(on)
        self.setCursor(Qt.PointingHandCursor if on else Qt.ArrowCursor)

    def set_spot_clickable(self, on: bool) -> None:
        """Offer a click on a SPOT: the day page's maps, whose payload carries the
        record of which track went into which bin."""
        self._spot_clickable = bool(on)
        if on:
            self.setCursor(Qt.PointingHandCursor)
        elif not self._clickable:
            self.setCursor(Qt.ArrowCursor)

    def set_overlay(self, overlay: dict | None) -> None:
        """`{"paths": [[[u, v], ...], ...], "bin": (ix, iy)}`, or None to clear."""
        self.content_version += 1
        self._overlay = overlay
        self.update()

    def overlay(self) -> dict | None:
        return self._overlay

    def bin_at(self, pos):
        """(ix, iy) of the bin under `pos` (row 0 = bottom wall), or None off the box."""
        if not self._grid:
            return None
        box = self._box()
        if not box.contains(pos):
            return None
        n_short, n_long = len(self._grid), len(self._grid[0])
        ix = min(n_long - 1, int((pos.x() - box.left()) / box.width() * n_long))
        row = min(n_short - 1, int((pos.y() - box.top()) / box.height() * n_short))
        return ix, n_short - 1 - row

    def mouseMoveEvent(self, e) -> None:
        """Resting on a bin says its n_eff -- on demand, never printed."""
        pos = e.position() if hasattr(e, "position") else e.pos()
        self._hover_at(pos)
        if self.loupe is not None:
            self.loupe.follow(self, pos)
        super().mouseMoveEvent(e)

    def _hover_at(self, pos) -> None:
        # The overlay fades only while the pointer is INSIDE the capsule, not the widget's
        # whole box around it (asked for: the corners beside the caps are not the map).
        self._set_overlay_hovered(self._capsule(self._box()).contains(QPointF(pos)))
        spot = self.bin_at(pos)
        text = None if spot is None else neff_hover_text(self._panel, *spot)
        if text:
            self._show_tip(spot, text)
        else:
            self._hide_tip()

    def _rehover(self) -> None:
        """Re-read the hover from where the pointer IS. A click on a spot shows or hides the
        card and the track list under the maps, the page changes height and the map can move
        under a pointer that never moved -- no move event follows, so the outline stayed on
        the bin that USED to be under it (reported: after a second click the hovered bin had
        no outline until the mouse moved)."""
        try:
            if not self.isVisible():
                return
            pos = self.mapFromGlobal(QCursor.pos())
        except RuntimeError:
            return
        if self.rect().contains(pos):
            self._hover_at(QPointF(pos))
            if self.loupe is not None:
                # a click can move the map under a still pointer (the page grows): the lens
                # follows it to where the pointer now is, and stays up
                self.loupe.follow(self, QPointF(pos))
        else:
            self._hide_tip()
            self._set_overlay_hovered(False)

    def _bin_top(self, ix: int, iy: int) -> QPointF:
        """The top centre of bin (ix, iy) (row 0 = bottom wall): where the tooltip points."""
        box = self._box()
        n_short, n_long = len(self._grid), len(self._grid[0])
        row = n_short - 1 - iy
        return QPointF(box.left() + (ix + 0.5) * box.width() / n_long,
                       box.top() + row * box.height() / n_short)

    def _bin_rect(self, ix: int, iy: int) -> QRectF:
        """Bin (ix, iy)'s rectangle on screen (row 0 = bottom wall)."""
        box = self._box()
        n_short, n_long = len(self._grid), len(self._grid[0])
        cw, ch = box.width() / n_long, box.height() / n_short
        return QRectF(box.left() + ix * cw, box.top() + (n_short - 1 - iy) * ch, cw, ch)

    def _update_bin(self, spot) -> None:
        """Repaint just around one bin (its hover outline), or nothing without one."""
        if spot is not None and self._grid:
            self.update(self._bin_rect(*spot).adjusted(-3, -3, 3, 3).toAlignedRect())

    def _fraction(self, value: float) -> float:
        """Where `value` sits on the ramp, 0 to 1 -- the one place it is worked out, for the
        bins, the smooth picture and the hover tooltip's pill.

        A FLAT RANGE (top == bottom: every bin of the map the same, an empty set, or no
        limits at all for a sparse map, which arrive as (0, 0)) puts every bin at the ramp's
        bottom: there is no scale to place it on. Dividing by it raised ZeroDivisionError
        on every paint and every mouse move over the map (reported 13 Oct 2026)."""
        span = self._top - self._lo
        if not span > 0:
            return 0.0
        return min(1.0, max(0.0, (value - self._lo) / span))

    def _ramp_colour(self, value: float) -> QColor:
        """The ramp colour a bin of `value` is painted in, read by `paintEvent` and by the
        hover tooltip's pill."""
        steps = len(self._ramp)
        v = self._fraction(value)
        return QColor("#" + self._ramp[min(steps - 1, round(v * (steps - 1)))])

    def _bin_colour(self, ix: int, iy: int) -> QColor:
        """The colour bin (ix, iy) shows on screen: grey when masked, the smooth picture's
        own pixel at the bin's centre in Smooth, otherwise its ramp colour."""
        card = QColor(self.t.card)
        if self._masked[iy][ix]:
            return _mix(QColor(self.t.border), card, 0.55)
        if self._smooth and self._image is not None:
            n_short, n_long = len(self._grid), len(self._grid[0])
            img = self._image
            px = min(img.width() - 1, int((ix + 0.5) / n_long * img.width()))
            py = min(img.height() - 1, int((n_short - 1 - iy + 0.5) / n_short * img.height()))
            return img.pixelColor(px, py)
        return self._ramp_colour(self._grid[iy][ix])

    def _show_tip(self, spot: tuple, text: str) -> None:
        """Above the hovered bin, clamped to the window, flipped below when there is no
        room -- `dynamics_chart._HoverChart._place`'s rule. Re-placed only when the bin
        changes, so a pointer moving inside one bin does not shake it."""
        try:
            if spot == self._tip_bin and self._tip.isVisible():
                return
            # The hovered bin is OUTLINED as well (asked for: among like-coloured neighbours
            # the swatch said which colour, not which bin); outline and tooltip move together.
            self._update_bin(self._tip_bin)
            self._tip_bin = spot
            self._update_bin(spot)
            # The pill is the hovered bin's own colour, so it says which bin it describes.
            self._tip.set_content(heading=text, rows=(), colour=self._bin_colour(*spot))
            window = self.window()
            if self._tip.parentWidget() is not window:
                self._tip.setParent(window)
            anchor = self.mapTo(window, self._bin_top(*spot).toPoint())
            w, h = self._tip.width(), self._tip.height()
            gap = 6
            x = max(4, min(anchor.x() - w // 2, max(4, window.width() - w - 4)))
            y = anchor.y() - gap - h
            if y < 4:
                y = anchor.y() + round(self._box().height() / len(self._grid)) + gap
            self._tip.move(x, y)
            self._tip.raise_()
            self._tip.show()
        except RuntimeError:
            # the window owns the reparented tip and may tear it down first on close
            return

    def hover_bin(self, ix: int, iy: int) -> None:
        """Hover bin (ix, iy) as if the pointer were on it: its value tooltip and outline,
        and the spot's tracks faded -- for the replicate page, where the arrow keys take the
        pointer's place (asked 14 Oct 2026)."""
        if not self._grid:
            return
        self._set_overlay_hovered(True)
        text = neff_hover_text(self._panel, ix, iy)
        if text:
            self._show_tip((ix, iy), text)
        else:
            self._hide_tip()

    def clear_hover(self) -> None:
        """No hover: the tooltip, the outline and the tracks' fading off."""
        self._hide_tip()
        self._set_overlay_hovered(False)

    def _hide_tip(self) -> None:
        self._update_bin(self._tip_bin)
        self._tip_bin = None
        try:
            self._tip.hide()
        except RuntimeError:
            pass

    def _set_overlay_hovered(self, on: bool) -> None:
        if on != self._overlay_hovered:
            self.content_version += 1
            self._overlay_hovered = on
            if self._overlay:
                self.update()

    def leaveEvent(self, e) -> None:
        self._hide_tip()
        self._set_overlay_hovered(False)
        if self.loupe is not None:
            self.loupe.leave(self)
        super().leaveEvent(e)

    def hideEvent(self, e) -> None:
        self._hide_tip()
        self._set_overlay_hovered(False)
        if self.loupe is not None:
            self.loupe.leave(self)
        super().hideEvent(e)

    def mouseReleaseEvent(self, e) -> None:
        if e.button() == Qt.LeftButton and self._spot_clickable and self._grid:
            box = self._box()
            pos = e.position() if hasattr(e, "position") else e.pos()
            if box.contains(pos):
                n_short, n_long = len(self._grid), len(self._grid[0])
                ix = min(n_long - 1, int((pos.x() - box.left()) / box.width() * n_long))
                row = int((pos.y() - box.top()) / box.height() * n_short)
                self.spot_clicked.emit(ix, n_short - 1 - min(n_short - 1, row))
                # after the page has re-laid itself out around the card the click changed
                QTimer.singleShot(0, self._rehover)
        elif e.button() == Qt.LeftButton and self._clickable:
            self.clicked.emit()
        super().mouseReleaseEvent(e)

    def _capsule(self, box: QRectF) -> QPainterPath:
        return _capsule_path(box)

    def _box(self) -> QRectF:
        """The capsule's box: the cell's proportions, fitted to this widget."""
        w, h = float(self.width()), float(self.height())
        bw = min(w, h * self._aspect)
        bh = bw / self._aspect
        return QRectF((w - bw) / 2.0, (h - bh) / 2.0, bw, bh)

    def _paint_overlay(self, p: QPainter, box: QRectF) -> None:
        """The clicked spot's tracks over the picture: white over a dark halo so
        they read on every ramp, the largest contributor thickest, and a ring on
        the clicked bin."""
        def point(u, v):
            # v = 0 is the BOTTOM wall, as the grid's row 0 is.
            return QPointF(box.left() + u * box.width(), box.bottom() - v * box.height())

        paths = self._overlay.get("paths") or []
        chosen = int(self._overlay.get("selected", 0))
        order = [r for r in range(len(paths) - 1, -1, -1) if r != chosen]
        if 0 <= chosen < len(paths):
            order.append(chosen)
        for rank in order:
            pts = [point(u, v) for u, v in paths[rank]]
            if len(pts) < 2:
                continue
            width = 2.6 if rank == chosen else 1.4
            for colour, extra in (("#111111", 2.0), ("#ffffff", 0.0)):
                pen = QPen(QColor(colour))
                pen.setWidthF(width + extra)
                pen.setCapStyle(Qt.RoundCap)
                pen.setJoinStyle(Qt.RoundJoin)
                p.setPen(pen)
                p.setBrush(Qt.NoBrush)
                for a, b in zip(pts, pts[1:]):
                    p.drawLine(a, b)
        spot = self._overlay.get("bin")
        if spot is not None and self._grid:
            n_short, n_long = len(self._grid), len(self._grid[0])
            c = point((spot[0] + 0.5) / n_long, (spot[1] + 0.5) / n_short)
            r = max(4.0, 0.6 * box.width() / n_long * float(self._overlay.get("span", 1)))
            for colour, width in (("#111111", 3.4), ("#ffffff", 1.6)):
                pen = QPen(QColor(colour))
                pen.setWidthF(width)
                p.setPen(pen)
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(c, r, r)

    def _paint_hover_outline(self, p: QPainter) -> None:
        """The hovered bin's edge: a dark stroke with a white one inside it -- the spot
        overlay's halo-and-white pair, so it reads on every ramp colour -- sharp-cornered
        like the grid, and clipped (by the caller) to the capsule. Drawn last, above the
        overlay. Thicker and two-tone, so it is never taken for the n_eff view's outlines."""
        rect = self._bin_rect(*self._tip_bin)
        p.setBrush(Qt.NoBrush)
        for colour, width, inset in (("#111111", 3.0, 1.5), ("#ffffff", 1.5, 1.5)):
            pen = QPen(QColor(colour))
            pen.setWidthF(width)
            pen.setJoinStyle(Qt.MiterJoin)
            p.setPen(pen)
            p.drawRect(rect.adjusted(inset, inset, -inset, -inset))

    def _paint_overlay_faded(self, p: QPainter, box: QRectF) -> None:
        """The overlay at `OVERLAY_HOVER_OPACITY`, faded as ONE layer: painted whole into
        a transparent image first, so overlapping joints, the halo under the white line and
        crossing tracks do not each darken where a translucent pen would stack them.

        The layer is the size of what `p` paints on (its logical window at its device's
        ratio): the widget itself, or the replicate page's lens drawing the map again."""
        ratio = p.device().devicePixelRatioF()
        area = p.window()
        layer = QImage(max(1, round(area.width() * ratio)), max(1, round(area.height() * ratio)),
                       QImage.Format_ARGB32_Premultiplied)
        layer.setDevicePixelRatio(ratio)
        layer.fill(0)
        q = QPainter(layer)
        q.setRenderHint(QPainter.Antialiasing, True)
        self._paint_overlay(q, box)
        q.end()
        p.save()
        p.setOpacity(OVERLAY_HOVER_OPACITY)
        p.drawImage(QPointF(0, 0), layer)
        p.restore()

    def paintEvent(self, _e) -> None:
        if self._absent:
            return
        p = QPainter(self)
        self.paint_map(p, self._box())
        p.end()

    def paint_map(self, p: QPainter, box: QRectF, *, hover: bool = True,
                  faded: bool | None = None) -> None:
        """Everything the map draws, fitted to `box` -- the widget's own (`paintEvent`), or
        a larger one: the replicate page's lens draws the map AGAIN at its magnification so the
        bins' edges stay sharp, rather than enlarging a snapshot's pixels. `hover` False
        leaves out the hover outline. `faded` draws the spot's tracks faded (None: as the
        widget does, while the pointer is over its capsule)."""
        p.setRenderHint(QPainter.Antialiasing, True)
        path = self._capsule(box)

        p.save()
        # INTERSECTED with a clip the caller already set (the lens's circle); with none set,
        # the capsule alone, as before.
        p.setClipPath(path, Qt.IntersectClip if p.hasClipping() else Qt.ReplaceClip)
        card = QColor(self.t.card)
        p.fillRect(box, card)
        grid, masked = self._grid, self._masked
        tracks = getattr(self, "_tracks", None)
        if tracks is not None:
            # THE TRACKS VIEW: every track of the set as a line in v2's colour, over the
            # card; no grid, no masking (both belong to the density).
            # NOT `path`: that name holds the capsule, drawn as the outline below; reusing it
            # drew the tracks there instead, at one pixel, and the cell had no outline.
            lines, ink, count = tracks
            if ink is not None and count:
                pen = QPen(ink)
                pen.setWidthF(0.8)
                pen.setCosmetic(True)
                pen.setJoinStyle(Qt.RoundJoin)
                p.setPen(pen)
                p.setBrush(Qt.NoBrush)
                p.save()
                p.translate(box.left(), box.top())
                p.scale(box.width(), box.height())
                p.drawPath(lines)
                p.restore()
            grid = None
            self._sparse_text = None
        elif self._sparse:
            # SPARSE RULE C: honest emptiness beats bright noise. The whole
            # panel is greyed and says how little is behind it, rather than
            # drawing a handful of two-jump specks as though they were a map.
            p.fillRect(box, _mix(QColor(self.t.border), card, 0.45))
            grid = None
        if grid is not None and self._top > self._lo:
            n_short, n_long = len(grid), len(grid[0])
            cw, ch = box.width() / n_long, box.height() / n_short
            grey = _mix(QColor(self.t.border), card, 0.55)
            smooth = self._smooth and self._image is not None
            if smooth:
                p.setRenderHint(QPainter.SmoothPixmapTransform, True)
                p.drawImage(box, self._image)
            for iy in range(n_short):
                row, mrow = grid[iy], masked[iy]
                # Row 0 is the BOTTOM of the cell -- the histogram's own
                # origin, and `draw_maps`' `origin="lower"`. Qt's y grows
                # downward, so the rows are laid out bottom-up here.
                y = box.top() + (n_short - 1 - iy) * ch
                for ix in range(n_long):
                    cell = QRectF(box.left() + ix * cw, y, cw + 0.6, ch + 0.6)
                    if mrow[ix]:
                        # UNDEFINED, which is a different statement from
                        # "no jumps happened here". Sharp in smooth mode too.
                        p.fillRect(cell, grey)
                        continue
                    if smooth:
                        continue
                    # EVERY unmasked bin is painted, including an empty
                    # one: the ramp's own bottom end IS "nothing here", and
                    # skipping it would leave the card showing through as a
                    # white hole -- which is how a MASKED bin should read,
                    # and this is the opposite statement. `imshow` paints
                    # the same bins for the same reason.
                    p.fillRect(cell, self._ramp_colour(row[ix]))
            if self._outline:
                # The n_eff view: bins the run MASKED are coloured like the rest
                # (that is the point -- you see how few built them) and ringed.
                ring = QPen(QColor(self.t.foreground))
                ring.setWidthF(1.4)
                p.setPen(ring)
                p.setBrush(Qt.NoBrush)
                for iy, orow in enumerate(self._outline):
                    y = box.top() + (n_short - 1 - iy) * ch
                    for ix, on in enumerate(orow):
                        if on:
                            p.drawRect(QRectF(box.left() + ix * cw + 0.7, y + 0.7,
                                              cw - 1.4, ch - 1.4))
        fade = self._overlay_hovered if faded is None else faded
        if self._overlay and fade:
            self._paint_overlay_faded(p, box)
        elif self._overlay:
            self._paint_overlay(p, box)
        if hover and self._tip_bin is not None and self._grid:
            self._paint_hover_outline(p)
        p.restore()

        if tracks is not None:
            # THE CELL OUTLINE in the Tracks view (asked 13 Oct 2026: "please add cell outline
            # in the SLA maps track view"): the model cell in ink, as v2's confined map drew
            # it (`confTracks.m:121`, 'k-'); the faint border vanished under the tracks.
            pen = QPen(QColor(self.t.foreground))
            pen.setWidthF(1.6)
        else:
            pen = QPen(QColor(self.t.border))
            pen.setWidthF(1.0)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawPath(path)
        if tracks is not None and tracks[1] is None:
            p.setPen(QColor(self.t.muted_foreground))
            p.setFont(ty.ui("xs", "medium", px=text_px(0.75)))
            p.drawText(box, Qt.AlignCenter, "Tracks: Confinement (SLA) maps only")
        elif tracks is None and self._sparse:
            p.setPen(QColor(self.t.muted_foreground))
            font = ty.ui("xs", "medium", px=text_px(0.75))
            p.setFont(font)
            # as much of it as the capsule holds, never cut at both ends (`spatial_map_view.sparse_text`)
            self.sparse_label = sparse_text(self._n_jumps, QFontMetricsF(font).horizontalAdvance, box.width() - 12)
            p.drawText(box, Qt.AlignCenter, self.sparse_label)


class _ColourBar(QWidget):
    """The ramp, from nothing to the panel's own top, with both ends named.

    **An unlabelled bar is worse than none over per-panel scaling**, which is
    `draw_maps`' own reason for printing the scale on it: two panels drawn to
    their own maxima look equally saturated while standing for different
    densities, so a bar with no number on it silently invites the comparison
    the figure cannot support. Hence one bar per panel when the scales are
    per-panel, and ONE bar for the pair when they are shared -- the layout
    itself says which comparison is on offer.
    """

    #: Not `_CapsuleChart`'s 32 steps: this is a continuous strip a few
    #: hundred pixels wide, where 32 flat bands would read as a staircase the
    #: map does not have. The ramp is interpolated between its sampled stops
    #: for the bar only; the MAP stays quantised by its bins.
    STEPS = 192
    BAR_H = 8
    #: A VERTICAL bar's width (`vertical=True`): the strip, then its two numbers beside it -- the bar a grid
    #: row stands beside when its maps share one scale (`spatial_map_view.bar_layout`, "rows").
    VERTICAL_W = 84

    def __init__(self, t, parent=None, *, vertical: bool = False):
        super().__init__(parent)
        self.t = t
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("background:transparent;")
        self._ramp: tuple[str, ...] = CMAP_RAMPS[DEFAULT_CMAP]
        self._lo = 0.0
        self._top = 0.0
        self._unit = ""
        self._note = ""
        self._ends: tuple[str, str] | None = None
        self._mark: float | None = None
        self._metrics_h = text_px(0.75) + 4
        self._vertical = bool(vertical)
        if self._vertical:
            self.setFixedWidth(self.VERTICAL_W)
            self.setMinimumHeight(3 * self._metrics_h)
            self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        else:
            self.setFixedHeight(self.BAR_H + 4 + self._metrics_h)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def is_vertical(self) -> bool:
        return self._vertical

    def set_scale(self, ramp, limits, unit: str, note: str = "", *,
                  ends: tuple[str, str] | None = None, mark: float | None = None) -> None:
        """`ends` replaces the two numeric end labels (the n_eff view's fixed log
        scale reads "1" and "20+ tracks", not its 0..1 positions); `mark` is a
        position in [0, 1] drawn as a tick across the bar (the n_eff cutoff)."""
        lo, hi = limits if limits else (0.0, 0.0)
        self._ramp, self._lo, self._top = ramp, lo, hi
        self._unit, self._note = unit, note
        self._ends, self._mark = ends, mark
        self.update()

    def _colour_at(self, f: float) -> QColor:
        """Linear interpolation between the two nearest sampled stops."""
        ramp = self._ramp
        x = max(0.0, min(1.0, f)) * (len(ramp) - 1)
        lo = int(x)
        hi = min(len(ramp) - 1, lo + 1)
        a, b = QColor("#" + ramp[lo]), QColor("#" + ramp[hi])
        return _mix(b, a, x - lo)

    def _end_texts(self, with_unit: bool = True, width: float | None = None) -> tuple:
        """The two end labels: the ones handed over (`set_scale`'s ``ends``), else the limits by THE formatter of
        every colour bar's ends (`chart_text.scale_ends`, the exported figure's too; ".3g" before, which printed a
        density's low end as "9.39e-06"), shortened where ``width`` px cannot hold both side by side."""
        if self._ends is not None:
            return self._ends
        from ... import chart_text

        unit = f" {self._unit}" if with_unit and self._unit else ""
        fits = None
        if width is not None:
            fm = QFontMetrics(ty.ui("xs", tabular=True, px=text_px(0.75)))
            fits = lambda t: fm.horizontalAdvance(t[0]) + fm.horizontalAdvance(t[1] + unit) + 12 <= width  # noqa: E731
        lo, hi = chart_text.scale_ends(self._lo, self._top, fits)
        return lo, f"{hi}{unit}"

    def _paint_vertical(self, p: QPainter) -> None:
        """The ramp UP the strip, low at the bottom; the top number at the top, the bottom one at the bottom, and
        along the right edge, turned up the bar, what it describes (with the unit when there is room)."""
        w, h = self.width(), self.height()
        strip = float(self.BAR_H)
        for i in range(self.STEPS):
            y1 = h - i * h / self.STEPS
            y0 = h - (i + 1) * h / self.STEPS
            p.fillRect(QRectF(0.0, y0, strip, y1 - y0 + 1.0), self._colour_at(i / (self.STEPS - 1)))
        pen = QPen(QColor(self.t.border))
        pen.setWidthF(1.0)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawRect(QRectF(0.5, 0.5, strip - 1.0, h - 1.0))
        if self._mark is not None:
            y = max(1.0, min(h - 1.0, h - self._mark * h))
            tick = QPen(QColor(self.t.foreground))
            tick.setWidthF(2.0)
            p.setPen(tick)
            p.drawLine(QPointF(-1.0, y), QPointF(strip + 1.0, y))
        p.setFont(ty.ui("xs", tabular=True, px=text_px(0.75)))
        p.setPen(QColor(self.t.muted_foreground))
        fm = QFontMetrics(p.font())
        x = strip + 4.0
        mh = float(self._metrics_h)
        room_w = int(w - x - mh - 2.0)          # the numbers; the note runs up the right edge
        lo_text, hi_text = self._end_texts(with_unit=False)
        p.drawText(QRectF(x, 0.0, room_w, mh), Qt.AlignLeft | Qt.AlignTop,
                   fm.elidedText(hi_text, Qt.ElideRight, room_w))
        p.drawText(QRectF(x, h - mh, room_w, mh), Qt.AlignLeft | Qt.AlignBottom,
                   fm.elidedText(lo_text, Qt.ElideRight, room_w))
        note_font = ty.ui("xs", px=text_px(0.75))
        p.setFont(note_font)
        fm_note = QFontMetrics(note_font)
        room = h - 4.0
        unit = "" if self._ends is not None else self._unit
        for text in ((f"{self._note} · {unit}" if self._note and unit else None), self._note):
            if text and fm_note.horizontalAdvance(text) <= room:
                p.save()
                p.translate(w - mh / 2.0 - 1.0, h / 2.0)
                p.rotate(-90.0)
                p.drawText(QRectF(-room / 2.0, -mh / 2.0, room, mh), Qt.AlignCenter, text)
                p.restore()
                break

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        w = self.width()
        if w <= 0:
            p.end()
            return
        if self._vertical:
            self._paint_vertical(p)
            p.end()
            return
        for i in range(self.STEPS):
            x0 = i * w / self.STEPS
            x1 = (i + 1) * w / self.STEPS
            p.fillRect(QRectF(x0, 0, x1 - x0 + 1.0, float(self.BAR_H)),
                       self._colour_at(i / (self.STEPS - 1)))
        pen = QPen(QColor(self.t.border))
        pen.setWidthF(1.0)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawRect(QRectF(0.5, 0.5, w - 1.0, self.BAR_H - 1.0))
        if self._mark is not None:
            # the cutoff: bins left of it are masked by the run
            x = max(1.0, min(w - 1.0, self._mark * w))
            tick = QPen(QColor(self.t.foreground))
            tick.setWidthF(2.0)
            p.setPen(tick)
            p.drawLine(QPointF(x, -1.0), QPointF(x, self.BAR_H + 1.0))
            p.setPen(pen)

        p.setFont(ty.ui("xs", tabular=True, px=text_px(0.75)))
        p.setPen(QColor(self.t.muted_foreground))
        box = QRectF(0, self.BAR_H + 2, w, self._metrics_h)
        # BOTH ends are labelled, because both are clipped now (sparse rule
        # B is a 2-98 % window, not a ceiling over a zero floor).
        lo_text, hi_text = self._end_texts(width=float(w))
        p.drawText(box, Qt.AlignLeft | Qt.AlignVCenter, lo_text)
        p.drawText(box, Qt.AlignRight | Qt.AlignVCenter, hi_text)
        if self._note:
            # THE ENDS WIN. Three strings share one line here, and the two
            # numbers are the bar's whole job -- a note printed over them
            # turns a scale into an unreadable smear, which is worse than no
            # note at all. Found on the section grid, where the columns are
            # about half the width the single-condition card gives them.
            note_font = ty.ui("xs", px=text_px(0.75))
            fm_ends = QFontMetrics(p.font())
            fm_note = QFontMetrics(note_font)
            # THE NOTE IS CENTRED, so it is the WIDER end that binds, not the
            # sum of the two. `lo + hi + note <= w` passes on a bar whose
            # right label is long and left label short -- and then the note,
            # sitting symmetrically about the middle, runs straight into the
            # long one. Seen on the long-jump column at a narrow window.
            # Centred means the note spans w/2 +- note/2, so each end needs
            # its own width plus a gap inside that.
            widest = max(fm_ends.horizontalAdvance(lo_text),
                         fm_ends.horizontalAdvance(hi_text))
            if 2 * widest + fm_note.horizontalAdvance(self._note) + 24 <= w:
                p.setFont(note_font)
                p.drawText(box, Qt.AlignHCenter | Qt.AlignVCenter, self._note)
        p.end()


def _neff_bar(bar, payload: dict | None, note: str, top: float | None,
              ramp=None) -> None:
    """The n_eff view's colour bar: viridis, the log scale's ends named (its top follows
    the grid's colour scale, `spatial_map_view.neff_tops`), the cutoff as a tick. `top`
    None: the maps under this bar have tops of their own, each printed under its map."""
    unit = _NEFF_UNIT_SHORT[(payload or {}).get("neff_unit", "tracks")
                            if (payload or {}).get("neff_min") is not None else "tracks"]
    if top is None:
        bar.set_scale(ramp or NEFF_RAMP, (0.0, 1.0), "", note, ends=("1", "own top"))
        return
    bar.set_scale(ramp or NEFF_RAMP, (0.0, 1.0), "", note,
                  ends=("1", f"{top:g}+ {unit}"), mark=neff_bar_mark(payload, top))


class SpatialMapGrid(Card):
    """The two maps side by side, the scale they share, and the caption that
    says what was drawn."""

    def __init__(self, t, parent=None):
        super().__init__(t, parent=parent)
        self.t = t
        self._payload: dict | None = None
        #: Every other condition computed so far, so a scale can be shared
        #: across them -- see `_panel_limits` and sparse rule A.
        self._others: list = []
        self._cmap = DEFAULT_CMAP
        #: DISPLAY only, like the colormap: "bins" or "smooth".
        self._rendering = RENDERINGS[0]
        #: SPARSE RULE A is the default: one scale per population, pooled
        #: across conditions. `both` (one top for both populations) is what
        #: was asked for on screen before the rule was implemented and stays
        #: one click away; `panel` scales each on its own.
        self._level = "population"

        outer = QVBoxLayout(self)
        outer.setContentsMargins(CARD_PAD, CARD_PAD, CARD_PAD, CARD_PAD)
        outer.setSpacing(CARD_GAP)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(8)
        self._head_title = label("Jump-density map", ty.ui("sm", "medium"),
                                 t.foreground)
        head.addWidget(self._head_title)
        head.addStretch(1)
        self._scale = Segmented(t, ["Per population", "Both", "Per panel"],
                                current=0, size="small")
        self._scale.changed.connect(self._on_scale)
        head.addWidget(self._scale)
        outer.addLayout(head)

        panels = QHBoxLayout()
        panels.setContentsMargins(0, 0, 0, 0)
        panels.setSpacing(CARD_GAP)
        self._charts: dict[str, _CapsuleChart] = {}
        self._columns: dict[str, QWidget] = {}
        self._headers: dict[str, object] = {}
        self._figures: dict[str, object] = {}
        self._bars: dict[str, _ColourBar] = {}
        # ONE column per UNIQUE kind key across every method, built once --
        # never rebuilt when the method switches, since this widget is a
        # persistent instance reused across `set_payload` calls (unlike
        # `SpatialMapMatrix`, which rebuilds per-run and can afford to).
        # `_redraw` shows only the CURRENT payload's two and hides the
        # other(s), the same `set_panel_visible` pattern this class already
        # had. `appd` and `hmm` deliberately share the KEYS "slow"/"fast"
        # (see `_KINDS_BY_METHOD`'s own comment), so they share these same
        # two physical columns rather than getting four -- only one of the
        # two methods is ever the active payload at a time, and the header
        # TEXT is set fresh per redraw below rather than baked in here, so
        # a shared column never shows a stale title from the other method.
        # KNOWN, ACCEPTED SIMPLIFICATION: `set_panel_visible("slow", on)`
        # therefore also carries over between appd and hmm if the user hides
        # one and later views a condition computed with the other -- only
        # WHICH COLUMNS ARE SHOWN can leak across methods this way, never
        # what is IN them (charts/figures/headers are always set fresh from
        # the live payload). Not worth a full method-qualified key for a
        # single-condition drill-in where switching methods happens through
        # the dialog and a fresh Run, not by casually flipping conditions.
        all_kinds = dict.fromkeys(
            [*(k for method in KINDS_BY_METHOD for n in MAP_N_POPS for k in kinds_for(method, n)),
             *RANGE_KINDS])
        for kind in all_kinds:
            column = QWidget()
            column.setStyleSheet("background:transparent;")
            col_lay = QVBoxLayout(column)
            col_lay.setContentsMargins(0, 0, 0, 0)
            col_lay.setSpacing(4)
            header = label("", ty.ui("xs", "medium"), t.foreground)
            col_lay.addWidget(header)
            chart = _CapsuleChart(t)
            col_lay.addWidget(chart, 1)
            figures = label("", ty.ui("xs", tabular=True), t.muted_foreground)
            figures.setWordWrap(True)          # an error message must not widen the window
            col_lay.addWidget(figures)
            #: ONE bar per panel, shown only when the panels are on their
            #: OWN scales. Under a shared scale these hide and
            #: `self._shared_bar` takes over, so the number of bars on
            #: screen always equals the number of scales in play -- the
            #: layout itself says which comparison is on offer.
            bar = _ColourBar(t)
            col_lay.addWidget(bar)
            panels.addWidget(column, 1)
            self._charts[kind] = chart
            self._columns[kind] = column
            self._headers[kind] = header
            self._figures[kind] = figures
            self._bars[kind] = bar
        # NOT stretched: each capsule asks for the height its width
        # implies (`_CapsuleChart.heightForWidth`), so a taller card should
        # leave the maps alone and put the slack at the bottom rather than
        # floating them in the middle of an empty panel.
        outer.addLayout(panels)

        self._shared_bar = _ColourBar(t)
        outer.addWidget(self._shared_bar)

        #: THE CLIFF, said BEFORE the picture rather than after. At two
        #: populations hard assignment is exactly a threshold on jump
        #: length, and a population whose best posterior is at or below 0.5
        #: is never the argmax -- its panel comes back EMPTY, and an empty
        #: panel reads as "nothing happens here" rather than as "this
        #: assignment cannot draw this population". Hidden whenever it does
        #: not apply, so it is never decoration.
        self._warning = label("", ty.ui("xs", "medium"), _WARN_COLOR)
        self._warning.setWordWrap(True)
        self._warning.hide()
        outer.addWidget(self._warning)

        self._caption = label("", ty.ui("xs"), t.muted_foreground)
        self._caption.setWordWrap(True)
        outer.addWidget(self._caption)

    # -- payload ------------------------------------------------------------ #
    def set_payload(self, payload: dict, others=()) -> None:
        """`others` are the other conditions already computed. Only those
        drawn at identical settings are pooled -- `_panel_limits` checks."""
        self._payload = payload
        self._others = list(others)
        self._redraw()

    def set_colormap(self, name: str) -> None:
        """DISPLAY only -- not one number in the payload moves, which is why
        this is a Visibility control and `n_min` is not."""
        if ramp_for(name) is not None and name != self._cmap:
            self._cmap = name
            self._redraw()

    def colormap(self) -> str:
        return self._cmap

    def set_rendering(self, mode: str) -> None:
        """DISPLAY only -- bins or a bilinear smooth picture of the same bins."""
        if mode not in RENDERINGS:
            raise ValueError(f"rendering must be one of {RENDERINGS}, got {mode!r}")
        if mode != self._rendering:
            self._rendering = mode
            self._redraw()

    def rendering(self) -> str:
        return self._rendering

    def set_panel_visible(self, kind: str, on: bool) -> None:
        """Hide one of the two maps.

        **The colour limits do not move when a panel is hidden.** They are
        computed over the whole panel SET (`_panel_limits`), and a hidden
        panel is still part of that set -- otherwise hiding one map would
        silently rescale the other, so the same colour would mean two
        different densities depending on what else happens to be on screen.
        """
        column = self._columns.get(kind)
        if column is not None:
            # The column carries this panel's own bar, so hiding it takes
            # that bar with it. The SHARED bar is outside the columns and
            # stays, which is correct: it describes a scale both panels are
            # on, and that scale is unchanged by one of them being hidden.
            column.setVisible(on)

    def panel_states(self) -> dict[str, bool]:
        return {k: w.isVisible() for k, w in self._columns.items()}

    def _on_scale(self, index: int) -> None:
        self._level = SCALE_LEVELS[index]
        self._redraw()

    def _redraw(self) -> None:
        payload = self._payload
        if not payload:
            return
        method = payload.get("method", "sqd")
        n_pop = payload.get("n_pop") or 2
        active = kinds_for(method, n_pop, payload.get("ranges"))
        titles = dict(zip(active, panel_titles_for(method, n_pop, payload.get("ranges"))))
        # appD is per-LOCALISATION; SQD and HMM are both per-STEP, so a
        # "jump" exists for both of them the same way.
        self._head_title.setText(
            "Localisation-density map" if method == "appd" else grid_title(method))
        limits, field = panel_limits(payload, self._level, self._others)
        ramp = ramp_for(self._cmap) or CMAP_RAMPS[DEFAULT_CMAP]   # a deleted custom ramp
        aspect = payload["cell_length_um"] / payload["cell_width_um"]
        unit_label = payload.get("weight_unit", "weighted jumps")
        # The unit follows the level, because the two plot different
        # quantities: a shared or per-population scale normalises each map to
        # a density so it can be compared at all; per-panel leaves the
        # weighted counts alone.
        unit = "density / bin" if field == "density" else f"{unit_label} / bin"
        for kind, column in self._columns.items():
            if kind not in active:
                # The OTHER methods' columns -- built once, never rebuilt
                # (see `__init__`), forced hidden here regardless of any
                # earlier `set_panel_visible` on them, since this payload has
                # nothing to draw in them.
                column.setVisible(False)
                continue
            # The header TEXT is set fresh here rather than baked in at
            # construction -- appd and hmm share these same two physical
            # columns (see `__init__`'s own comment), so a stale title from
            # whichever method built the column last would otherwise stick.
            self._headers[kind].setText(titles[kind])
            chart = self._charts[kind]
            panel = payload["panels"].get(kind)
            if self._rendering == "tracks":
                chart.set_panel(panel, paint_field(panel, field), limits.get(kind), ramp,
                                aspect)
                chart.set_tracks(panel.get("tracks", []) if panel else [],
                                 panel.get("ink") if panel else None)
                self._bars[kind].setVisible(False)
            elif self._rendering == "neff":
                # THE n_eff VIEW: a different quantity on its own fixed scale, so
                # the density limits and the scale level do not apply.
                top = neff_scale_top([payload])
                shown = neff_display_panel(panel, top)
                chart.set_panel(shown, "neff_pos", (0.0, 1.0), ramp, aspect)
                _neff_bar(self._bars[kind], payload, "log scale · this condition", top, ramp)
                self._bars[kind].setVisible(shown is not None)
            else:
                chart.set_panel(panel, paint_field(panel, field), limits.get(kind), ramp,
                                aspect, self._rendering == "smooth")
                self._bars[kind].set_scale(ramp, limits.get(kind),
                                           panel_unit(panel, field, unit_label),
                                           "" if self._level == "both" else "own scale")
                self._bars[kind].setVisible(self._level != "both" and bool(panel)
                                            and not panel.get("sparse"))
            if panel is not None:
                self._figures[kind].setText(figures_text(payload, panel))
        self._shared_bar.setVisible(self._level == "both" and bool(limits)
                                    and self._rendering not in ("neff", "tracks"))
        if self._level == "both" and limits and self._rendering != "neff":
            self._shared_bar.set_scale(ramp, next(iter(limits.values())), unit,
                                       "one scale for both panels")
        self._caption.setText(
            caption_for(payload, self._level, self._cmap, len(self._others) + 1,
                        self._rendering))

        # THE CLIFF (see `spatial_map_view.cliff_status`) and the fit's own warnings, said before the picture
        # is read (`map_warning`).
        warning = map_warning(payload)
        self._warning.setText(warning or "")
        self._warning.setVisible(bool(warning))


def _population_inks(n_pop: int) -> list:
    """One ink per population, slowest first: the short ink alone for one population (every jump), short and long
    for two, a middle accent between them for three."""
    if int(n_pop) == 1:
        return [QColor(_SHORT_INK)]
    if int(n_pop) == 2:
        return [QColor(_SHORT_INK), QColor(_LONG_INK)]
    return [QColor(_SHORT_INK), QColor(_ACCENT_INK), QColor(_LONG_INK)]


def _sqd_population_names(n_pop: int) -> tuple:
    """The SQD pictures' legend words, slowest first (`spatial_map_view.kinds_for`'s kinds)."""
    return {1: ("all jumps",), 3: ("short", "middle", "long")}.get(int(n_pop), ("short", "long"))


class _SettingArt(QWidget):
    """The live diagram for whichever setting is open.

    One exception to "nothing here is your data" below: the track cutoff draws
    the REAL distribution of track lengths (from the census the dialog is given),
    because the point of that picture is choosing a number from the data.

    **Drawn in ink and greys, never in the map's own colormap.** These are
    MECHANISM diagrams, not previews: nothing here is your data, and the one
    that could be mistaken for a result -- the sparse-bin grid -- is
    deliberately monochrome so it cannot read as a jet-coloured map. The two
    population colours appear once, for assignment, where the whole point is
    that a jump belongs to one population or is split between two.

    The capsule comes from `_capsule_path`, the same closed form the real
    map draws, so the dialog cannot illustrate a different cell than the one
    it is configuring.
    """

    #: THE DIAGRAM SITS IN A PLATE, and the plate is what aligns.
    #:
    #: A capsule cannot align with a column of text: its outline is a curve
    #: that touches any vertical line at ONE tangent point, so flush-left
    #: against the title and the control it still read as ragged, and
    #: centred it read as adrift. Neither is fixable by moving the capsule,
    #: because the capsule has no straight edge to align WITH. A quiet
    #: rounded plate does: its edges are straight, they sit exactly on the
    #: panel's own margins, and the capsule is then free to be centred
    #: inside it where it belongs.
    #:
    #: Height follows from that: the plate spans the 456px panel, its
    #: padding leaves 428 for the capsule, and 428 / 2.94 = 146.
    PAD = 14
    HEIGHT = 174

    #: The cutoff chart's line was dragged to this limit.
    cutoff_dragged = Signal(int)

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("background:transparent;")
        self.setMinimumHeight(int(self.HEIGHT + _JUMP_AXIS_BAND))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._key = "method"
        self._state: dict = {}
        #: Soft/hard as a CONTINUOUS blend, not a bool -- 0.0 draws exactly
        #: the soft picture, 1.0 exactly the hard one, and `_draw_assignment`
        #: reads this alone, so a value sitting still at either end reproduces
        #: today's two static pictures pixel-for-pixel. Only a real toggle
        #: (see `show_setting`) ever moves it off an endpoint.
        self._hardness = 0.0
        #: `None` until the first `show_setting` call, which is how a real
        #: TOGGLE is told apart from the picture simply being (re)drawn --
        #: `show_setting` fires on every control change in the dialog, most
        #: of which have nothing to do with assignment.
        self._last_assignment: str | None = None
        #: The one animation this widget can have in flight, kept as a live
        #: attribute for the reason `window.py`'s own `_fade_in` already
        #: gives: a collected Python wrapper mid-flight is a real risk for a
        #: RUNNING `QVariantAnimation`. Unlike that one-shot fade, this widget
        #: is built once and lives for the dialog's whole life (`SpatialMapDialog`'s
        #: own "built once and reopened"), so a single named slot is enough --
        #: never more than one such animation exists at a time.
        self._assign_anim: QVariantAnimation | None = None
        #: The highlighted grid cell in `_draw_jump`, as a CONTINUOUS centre
        #: fraction of the box -- see `_JUMP_LANDING`. Animates the same way
        #: `_hardness` does: a value sitting still reproduces one option's
        #: resting picture exactly, and only a real change in
        #: `step_position` ever moves it.
        self._jump_frac = QPointF(*_JUMP_LANDING[STEP_POSITIONS[0]][1])
        self._last_step_position: str | None = None
        self._jump_anim: QVariantAnimation | None = None
        #: "Cell frame"'s own SOLID capsule size, as a CONTINUOUS
        #: (length_um, width_um) -- squeezes/expands between deposit's and
        #: measured's own real size when the selection changes, the same
        #: "a value sitting still reproduces the resting picture exactly"
        #: shape `_hardness`/`_jump_frac` already use. `QPointF(0, 0)` here
        #: is never actually drawn from: `show_setting`'s first call always
        #: sets it directly before any paint happens.
        self._frame_size = QPointF(0.0, 0.0)
        self._last_frame_selected: str | None = None
        self._frame_size_anim: QVariantAnimation | None = None
        #: The intro's PROGRESS along the jump, 0.0 at the start point to 1.0 at the
        #: end, or `None` when no intro is running -- `None` is the resting picture.
        self._intro_p: float | None = None
        #: The running intro, kept alive as an attribute for the same reason as the
        #: other animations here.
        self._intro_anim: QSequentialAnimationGroup | None = None
        #: Whether the intro already played in THIS opening of the dialog.
        #: `SpatialMapDialog.open` resets it.
        self._intro_played = False
        #: The method picture's own intro (`play_method_intro`): its 0..1 progress, or
        #: None when resting; kept apart from the jump intro, since each page has one.
        self._method_p: float | None = None
        self._method_anim: QSequentialAnimationGroup | None = None
        self._method_played = False
        self._last_method: str | None = None

    #: How much taller the picture is on Bin masking: exactly the n_eff pair, under a
    #: full-size cell. The plate's own bottom band (`_JUMP_AXIS_BAND`) is used first.
    MASKING_EXTRA = int(_PAIR_GAP + _PAIR_KEY_H + _PAIR_SIDE + _PAIR_LABEL_H - _JUMP_AXIS_BAND)

    def show_setting(self, key: str, state: dict) -> None:
        height = int(self.HEIGHT + _JUMP_AXIS_BAND
                     + (self.MASKING_EXTRA if key == "masking" else 0))
        if self.minimumHeight() != height:
            self.setMinimumHeight(height)
            self.setMaximumHeight(height)
        if key != "jump":
            self._stop_jump_intro()
        if key != "method":
            self.stop_method_intro()
        assignment = state.get("assignment")
        if self._last_assignment is None:
            # The very first paint -- nothing has been ON SCREEN yet for a
            # toggle to move FROM, so this sets the resting value directly
            # rather than animating from an arbitrary 0.0.
            self._hardness = 1.0 if assignment == "hard" else 0.0
        elif assignment != self._last_assignment:
            self._animate_hardness(assignment == "hard")
        self._last_assignment = assignment

        step = state.get("step_position")
        if self._last_step_position is None:
            self._jump_frac = QPointF(*_JUMP_LANDING.get(
                step, _JUMP_LANDING[STEP_POSITIONS[0]])[1])
        elif step != self._last_step_position and step in _JUMP_LANDING:
            self._stop_jump_intro()
            self._animate_jump(step)
        self._last_step_position = step

        # `frame_selected` is the raw tab pick (`self._frame.current()`),
        # decoupled from whether `measured` data exists yet -- unlike
        # `state["frame_source"]` (`SpatialMapDialog.config()`), which
        # reads "deposit" whenever `measured` is still `None` REGARDLESS of
        # which tab is highlighted. Using the resolved string here would
        # make `set_context` supplying real data after construction look
        # like a click (deposit -> measured with nobody having toggled
        # anything), which is exactly the bug this comment used to describe
        # incorrectly as fixed: the dialog's own `_show(_KEYS[0])` call
        # fires before `set_context` runs, so that first call always sees
        # `measured=None` and `frame_source` resolves to "deposit" no
        # matter that the segmented control defaults to index 0.
        frame_selected = state.get("frame_selected", "deposit")
        frame_target = self._frame_target(state, frame_selected)
        if self._last_frame_selected is None:
            self._frame_size = frame_target
        elif frame_selected != self._last_frame_selected:
            # A real TOGGLE -- the user picked the other option. Animate.
            self._animate_frame_size(frame_target)
        elif (self._frame_size_anim is None
              and (self._frame_size.x(), self._frame_size.y())
              != (frame_target.x(), frame_target.y())):
            # The SAME tab is still selected, but what it resolves TO has
            # changed underneath -- `set_context` supplying (or replacing)
            # `measured` after construction, or reopening for a different
            # condition. Not a click, so not animated: snap straight to the
            # now-correct value rather than squeeze/expand toward data the
            # user never asked to compare against.
            self._frame_size = frame_target
        self._last_frame_selected = frame_selected

        # The jump diagram replays its intro on click; say so with the cursor.
        weighting = state.get("weighting", "steps")
        if getattr(self, "_last_weighting", None) is None:
            self._weight_blend = 1.0 if weighting == "track" else 0.0
        elif weighting != self._last_weighting:
            self._animate_weighting(weighting == "track")
        self._last_weighting = weighting

        folded = bool(state.get("mirrored"))
        if getattr(self, "_last_fold", None) is None:
            self._fold_blend = 1.0 if folded else 0.0
        elif folded != self._last_fold:
            self._animate_fold(folded)
        self._last_fold = folded

        self.setCursor(Qt.PointingHandCursor if key in ("jump", "method") else
                       Qt.SizeHorCursor if key == "cutoff" else Qt.ArrowCursor)
        self._key, self._state = key, dict(state)
        # A different METHOD replays its picture: how a method decides is the walk.
        method = state.get("method")
        if (key == "method" and self._last_method is not None
                and method != self._last_method):
            self.play_method_intro(force=True)
        self._last_method = method
        self.update()

    def _frame_target(self, state: dict, source: str) -> QPointF:
        """The real (length_um, width_um) a "Cell frame" selection is
        heading TO or resting at -- a pure lookup, deposit's own constant
        or this condition's own measured median, never a guess.

        Takes `state` as a plain argument rather than reading
        `self._state`: `show_setting` calls this before that assignment
        happens, and threading the value through here is simpler than
        moving that assignment earlier just for this one reader.
        """
        if source == "measured":
            m = state.get("measured")
            if m:
                return QPointF(m["cell_length_um"], m["cell_width_um"])
        return QPointF(DEPOSIT_LENGTH_UM, DEPOSIT_WIDTH_UM)

    def _animate_frame_size(self, target: QPointF) -> None:
        """The solid capsule squeezing or expanding to the newly selected
        option's own size -- asked for directly, once the number-on-the-
        shape redesign made a plain instant swap feel abrupt.

        Same retarget-not-queue shape as `_animate_hardness`/`_animate_jump`:
        `.stop()` leaves `self._frame_size` at whatever the last tick wrote,
        so clicking measured/deposit/measured in quick succession never
        visibly jumps.
        """
        if self._frame_size_anim is not None:
            self._frame_size_anim.stop()
        anim = QVariantAnimation(self)
        anim.setStartValue(QPointF(self._frame_size))
        anim.setEndValue(target)
        # 220ms -- quick enough to read as a snappy response to a click,
        # long enough that the squeeze/expand is actually visible rather
        # than a blur between two static pictures.
        anim.setDuration(220)
        curve = QEasingCurve(QEasingCurve.BezierSpline)
        curve.addCubicBezierSegment(QPointF(0.4, 0.0), QPointF(0.2, 1.0),
                                    QPointF(1.0, 1.0))
        anim.setEasingCurve(curve)
        anim.valueChanged.connect(self._on_frame_size_tick)
        anim.finished.connect(self._on_frame_size_finished)
        self._frame_size_anim = anim
        anim.start()

    def _on_frame_size_tick(self, value) -> None:
        self._frame_size = value
        self.update()

    def _on_frame_size_finished(self) -> None:
        self._frame_size_anim = None

    def _animate_hardness(self, to_hard: bool) -> None:
        """Soft sharpening into hard, or relaxing back -- see `_hardness`.

        Retargets a RUNNING animation rather than queuing a second one:
        `.stop()` leaves `self._hardness` at whatever `_on_hardness_tick`
        last wrote, since that value is read from the emitted signal rather
        than re-queried afterward, so starting the next leg from
        `self._hardness` cannot introduce a visible jump even under a rapid
        double-toggle.
        """
        target = 1.0 if to_hard else 0.0
        if self._assign_anim is not None:
            self._assign_anim.stop()
        anim = QVariantAnimation(self)
        anim.setStartValue(self._hardness)
        anim.setEndValue(target)
        # 280ms: long enough to watch the bars move, short enough that a
        # user clicking soft/hard/soft in quick succession is not left
        # waiting on it.
        anim.setDuration(280)
        curve = QEasingCurve(QEasingCurve.BezierSpline)
        curve.addCubicBezierSegment(QPointF(0.4, 0.0), QPointF(0.2, 1.0),
                                    QPointF(1.0, 1.0))
        anim.setEasingCurve(curve)
        anim.valueChanged.connect(self._on_hardness_tick)
        anim.finished.connect(self._on_hardness_finished)
        self._assign_anim = anim
        anim.start()

    def _on_hardness_tick(self, value) -> None:
        self._hardness = float(value)
        # Redraws WHATEVER `_key` currently is. If a different setting is on
        # screen the assignment toggle could not have fired in the first
        # place (its own control is only visible while its own row is
        # selected), so this is never a wasted repaint of an unrelated
        # picture -- see `show_setting`'s docstring.
        self.update()

    def _on_hardness_finished(self) -> None:
        self._assign_anim = None

    def _animate_jump(self, to_step: str) -> None:
        """The highlighted cell sliding to a new option -- `_animate_hardness`'s
        own shape, aimed at a `QPointF` instead of a float. PySide6's
        `QVariantAnimation` interpolates `QPointF` natively, so the glow
        travels in a straight line across the grid rather than needing its
        x/y driven as two separate animations.
        """
        target = QPointF(*_JUMP_LANDING[to_step][1])
        if self._jump_anim is not None:
            self._jump_anim.stop()
        anim = QVariantAnimation(self)
        anim.setStartValue(QPointF(self._jump_frac))
        anim.setEndValue(target)
        # Faster than the assignment morph (280ms) on purpose: that one is
        # a shape genuinely changing (a gradient sharpening into a step) and
        # wants to be watched; this is a single cell relocating, and asked
        # for directly to move quicker once it was on screen -- twice now.
        anim.setDuration(100)
        curve = QEasingCurve(QEasingCurve.BezierSpline)
        curve.addCubicBezierSegment(QPointF(0.4, 0.0), QPointF(0.2, 1.0),
                                    QPointF(1.0, 1.0))
        anim.setEasingCurve(curve)
        anim.valueChanged.connect(self._on_jump_tick)
        anim.finished.connect(self._on_jump_finished)
        self._jump_anim = anim
        anim.start()

    def _on_jump_tick(self, value) -> None:
        self._jump_frac = value
        self.update()

    def _on_jump_finished(self) -> None:
        self._jump_anim = None

    def _animate_weighting(self, to_track: bool) -> None:
        """The bins re-colour from "every jump counts" to "every track counts
        once" (or back) -- `_animate_hardness`'s shape, 220 ms."""
        anim = getattr(self, "_weight_anim", None)
        if anim is not None:
            anim.stop()
        anim = QVariantAnimation(self)
        anim.setStartValue(float(getattr(self, "_weight_blend", 0.0)))
        anim.setEndValue(1.0 if to_track else 0.0)
        anim.setDuration(220)
        anim.setEasingCurve(QEasingCurve(QEasingCurve.InOutCubic))
        anim.valueChanged.connect(self._on_weight_tick)
        anim.finished.connect(self._on_weight_finished)
        self._weight_anim = anim
        anim.start()

    def _on_weight_tick(self, value) -> None:
        self._weight_blend = float(value)
        self.update()

    def _on_weight_finished(self) -> None:
        self._weight_anim = None

    def _animate_fold(self, to_on: bool) -> None:
        """Fold flips the cloud into the other quarters, or back -- see `_draw_fold`. One
        linear animation; `_draw_fold` splits it into its two eased flips, so a toggle in
        mid-flip retargets from wherever it is (`_animate_hardness`'s rule). Short on
        purpose: two flips of 200 ms, played only on a toggle."""
        anim = getattr(self, "_fold_anim", None)
        if anim is not None:
            anim.stop()
        anim = QVariantAnimation(self)
        start = float(getattr(self, "_fold_blend", 0.0))
        target = 1.0 if to_on else 0.0
        anim.setStartValue(start)
        anim.setEndValue(target)
        anim.setDuration(max(1, round(_FOLD_FLIP_MS * 2 * abs(target - start))))
        anim.valueChanged.connect(self._on_fold_tick)
        anim.finished.connect(self._on_fold_finished)
        self._fold_anim = anim
        anim.start()

    def _on_fold_tick(self, value) -> None:
        self._fold_blend = float(value)
        self.update()

    def _on_fold_finished(self) -> None:
        self._fold_anim = None

    # -- the cutoff chart's geometry: a LOG length axis, 5 to 200 ------------ #
    def _cutoff_plot(self) -> QRectF:
        return QRectF(self.PAD + 6, self.PAD + 10, self.width() - 2 * self.PAD - 12,
                      self.height() - 2 * self.PAD - 34)

    def _x_for(self, length: float) -> float:
        r = self._cutoff_plot()
        lo, hi = math.log(4.5), math.log(MAX_LOCS_MAX + 0.5)
        return r.left() + (math.log(length) - lo) / (hi - lo) * r.width()

    def _limit_at(self, x: float) -> int:
        r = self._cutoff_plot()
        lo, hi = math.log(4.5), math.log(MAX_LOCS_MAX + 0.5)
        f = min(1.0, max(0.0, (x - r.left()) / max(1.0, r.width())))
        return max(MAX_LOCS_MIN, min(MAX_LOCS_MAX, round(math.exp(lo + f * (hi - lo)) - 0.5)))

    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.LeftButton and self._key == "cutoff":
            self.cutoff_dragged.emit(self._limit_at(e.position().x()))
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e) -> None:
        if (e.buttons() & Qt.LeftButton) and self._key == "cutoff":
            self.cutoff_dragged.emit(self._limit_at(e.position().x()))
        super().mouseMoveEvent(e)

    def reset_intro(self) -> None:
        """Stop any intro and allow it to play again (a new opening)."""
        self.stop_intro()
        self._intro_played = False
        self._method_played = False

    def intro_running(self) -> bool:
        """Whether the jump intro is in flight."""
        return self._intro_anim is not None

    def play_intro(self, *, force: bool = False) -> None:
        """Once per opening unless `force`; only on the jump page and only while visible."""
        if self._key != "jump" or not self.isVisible():
            return
        if self._intro_played and not force:
            return
        self._stop_jump_intro()
        group = QSequentialAnimationGroup(self)
        group.addAnimation(QPauseAnimation(_JUMP_INTRO_HOLD_MS, group))
        # There and back: start -> end, then end -> start (the picture is drawn from
        # the progress alone, so the way back is the same slide run from 1 to 0).
        for start, end in ((0.0, 1.0), (1.0, 0.0)):
            slide = QVariantAnimation(group)
            slide.setStartValue(start)
            slide.setEndValue(end)
            slide.setDuration(_JUMP_INTRO_SLIDE_MS)
            slide.setEasingCurve(QEasingCurve(QEasingCurve.InOutCubic))
            slide.valueChanged.connect(self._on_intro_tick)
            group.addAnimation(slide)
            group.addAnimation(QPauseAnimation(_JUMP_INTRO_HOLD_MS, group))
        group.finished.connect(self._on_intro_finished)
        self._intro_played = True
        self._intro_p = 0.0
        self._intro_anim = group
        self.update()
        group.start()

    def stop_intro(self) -> None:
        """Stop every intro, if running, and return to the resting picture."""
        self._stop_jump_intro()
        self.stop_method_intro()

    # -- the method picture's intro: a walk along the track, then the labels -- #
    def method_intro_running(self) -> bool:
        return self._method_anim is not None

    def play_method_intro(self, *, force: bool = False) -> None:
        """Once per opening unless `force`; only on the method page and only while
        visible -- the same rules as the jump intro."""
        if self._key != "method" or not self.isVisible():
            return
        if self._method_played and not force:
            return
        self.stop_method_intro()
        group = QSequentialAnimationGroup(self)
        group.addAnimation(QPauseAnimation(_METHOD_INTRO_HOLD_MS, group))
        walk = QVariantAnimation(group)
        walk.setStartValue(0.0)
        walk.setEndValue(1.0)
        walk.setDuration(_METHOD_INTRO_MS)
        walk.valueChanged.connect(self._on_method_tick)
        group.addAnimation(walk)
        group.finished.connect(self._on_method_finished)
        self._method_played = True
        self._method_p = 0.0
        self._method_anim = group
        self.update()
        group.start()

    def stop_method_intro(self) -> None:
        if self._method_anim is not None:
            anim = self._method_anim
            self._method_anim = None
            anim.stop()
        if self._method_p is not None:
            self._method_p = None
            self.update()

    def _on_method_tick(self, value) -> None:
        self._method_p = float(value)
        self.update()

    def _on_method_finished(self) -> None:
        self._method_anim = None
        self._method_p = None
        self.update()

    def _stop_jump_intro(self) -> None:
        """Stop the jump intro, if running, and return to the resting picture."""
        if self._intro_anim is not None:
            anim = self._intro_anim
            self._intro_anim = None
            anim.stop()
        self._intro_p = None
        self.update()

    def _on_intro_tick(self, value) -> None:
        """One step of the slide."""
        self._intro_p = float(value)
        self.update()

    def _on_intro_finished(self) -> None:
        """The intro ended; back to the resting picture."""
        self._intro_anim = None
        self._intro_p = None
        self.update()

    def mouseReleaseEvent(self, e) -> None:
        """A click on the jump or method diagram replays its intro."""
        if e.button() == Qt.LeftButton and self._key == "jump":
            self.play_intro(force=True)
        elif e.button() == Qt.LeftButton and self._key == "method":
            self.play_method_intro(force=True)
        super().mouseReleaseEvent(e)

    # -- geometry ----------------------------------------------------------- #
    def _plate(self) -> QRectF:
        """The straight-edged surface the diagram sits on -- see `HEIGHT`."""
        return QRectF(0.0, 0.0, float(self.width()), float(self.HEIGHT))

    def _inner(self) -> QRectF:
        return self._plate().adjusted(self.PAD, self.PAD, -self.PAD, -self.PAD)

    def _capsule_box(self) -> QRectF:
        """The cell, centred on the plate and keeping its own proportion."""
        box = self._inner()
        bw = min(box.width(), box.height() * _ART_ASPECT)
        bh = bw / _ART_ASPECT
        return QRectF(box.center().x() - bw / 2.0, box.center().y() - bh / 2.0,
                      bw, bh)

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        # THE PLATE IS INVISIBLE: no fill, no border, asked for directly
        # after a tone and then a hairline were each tried and rejected. It
        # survives as GEOMETRY only -- `_plate`/`_inner` still give every
        # diagram one inset and one centre, so the six sit alike, and the
        # capsule comes within `PAD` of both margins, which is close enough
        # that its curve no longer reads against them.
        getattr(self, "_draw_" + self._key)(p)
        p.end()

    def _fill_cell(self, p: QPainter, box: QRectF) -> None:
        """A no-op now the plate itself is card-coloured -- kept as the one
        place that would fill a cell, so a future plate tone does not have to
        rediscover which diagrams need it."""

    def _draw_method(self, p: QPainter) -> None:
        """How each method decides a step is slow or fast, on one example track.

        Asked for ("animations explaining ... one illustrating sqd, hmm, appd as
        well"), replacing the blank plate an earlier ruling left here. The track
        and the three labellings are `_METHOD_TRACK` / `_method_labels`: a bound
        half and a free half, each with one odd step, so the methods disagree on
        exactly the steps that show how they decide.

        The intro WALKS the track. Under SQD each step takes its colour as it is
        walked -- a jump is judged on its own. Under the HMM and apparent D the
        walked steps stay grey and colour only once the walk ends, because both
        need the whole track; the HMM then splits it into two runs, apparent D
        paints it one colour. Under the cell, the same steps in time order.
        Population colours as in the assignment picture: slow / short in
        `_SHORT_INK`, fast / long in `_LONG_INK`.
        """
        method = self._state.get("method", "sqd")
        if method == "sla":
            self._draw_sla(p)
            return
        if method == "sda":
            self._draw_sda(p)
            return
        labels = _method_labels(method)
        box = self._capsule_box()
        n = len(labels)
        ink = (QColor(_SHORT_INK), QColor(_LONG_INK))
        grey = QColor(self.t.border)
        intro = self._method_p
        if intro is None:
            walked, colour = float(n), 1.0
        else:
            walked = n * min(1.0, intro / _METHOD_WALK_END)
            colour = min(1.0, max(0.0, (intro - _METHOD_WALK_END)
                                  / (_METHOD_COLOUR_END - _METHOD_WALK_END)))

        def step_colour(i: int) -> QColor:
            if method == "sqd" or intro is None:
                return ink[labels[i]]
            return _mix(ink[labels[i]], grey, colour)

        pts = [QPointF(box.left() + u * box.width(), box.top() + v * box.height())
               for u, v in _METHOD_TRACK]
        p.save()
        p.setClipPath(_capsule_path(box))
        axis = QPen(QColor(self.t.border))
        axis.setWidthF(1.0)
        axis.setDashPattern([2, 4])
        p.setPen(axis)
        p.drawLine(QPointF(box.left(), box.center().y()),
                   QPointF(box.right(), box.center().y()))
        head = pts[0]
        for i in range(n):
            a, b = pts[i], pts[i + 1]
            done = walked - i
            if done <= 0:
                ghost = QPen(grey)
                ghost.setWidthF(1.1)
                ghost.setDashPattern([2, 3])
                p.setPen(ghost)
                p.drawLine(a, b)
                continue
            end = b if done >= 1 else QPointF(a.x() + (b.x() - a.x()) * done,
                                             a.y() + (b.y() - a.y()) * done)
            pen = QPen(step_colour(i))
            pen.setWidthF(2.6)
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            p.drawLine(a, end)
            head = end
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(self.t.foreground))
        for k, point in enumerate(pts):
            if k <= walked:
                p.drawEllipse(point, 1.8, 1.8)
        if intro is not None and walked < n:
            p.drawEllipse(head, 4.6, 4.6)
        p.restore()
        self._outline(p, box)

        # THE SAME STEPS IN TIME ORDER, under the cell (in `_JUMP_AXIS_BAND`).
        top = box.bottom() + 10.0
        gap = 3.0
        cell = (box.width() - gap * (n - 1)) / n
        p.setPen(Qt.NoPen)
        for i in range(n):
            x = box.left() + i * (cell + gap)
            p.setBrush(step_colour(i) if i < walked else grey)
            p.drawRoundedRect(QRectF(x, top, cell, 9.0), 2.0, 2.0)
        p.setFont(ty.ui("xs", px=text_px(0.66)))
        line = QRectF(box.left(), top + 13.0, box.width(), 14.0)
        p.setPen(QColor(self.t.muted_foreground))
        p.drawText(line, Qt.AlignLeft | Qt.AlignVCenter, _METHOD_CAPTION[method])
        # The legend, right-aligned: what the two colours are.
        fm = QFontMetricsF(p.font())
        x = line.right()
        for text, colour in (("fast", ink[1]), ("slow", ink[0])):
            w = fm.horizontalAdvance(text)
            x -= w
            p.setPen(QColor(self.t.muted_foreground))
            p.drawText(QRectF(x, line.top(), w + 1, line.height()),
                       Qt.AlignLeft | Qt.AlignVCenter, text)
            x -= 11.0
            p.setPen(Qt.NoPen)
            p.setBrush(colour)
            p.drawRoundedRect(QRectF(x, line.center().y() - 4.0, 8.0, 8.0), 1.5, 1.5)
            x -= 10.0

    def _draw_populations(self, p: QPainter) -> None:
        """The fitted populations' curves and a dashed cut where neighbours CROSS -- the
        "intersection of the individual populations" the old speed map asked you to type
        in. Apparent diffusion is drawn over apparent D, SQD over jump length (its cuts
        are the hard assignment's). Two or three curves, following the control;
        schematic values, the cuts from the same functions the maps use
        (`spatial_map_view.population_picture`, which calls `spatialappd.appd_crossings`
        and `spatialsqd.sqd_crossings`)."""
        n_pop = drawn_n_pop(self._state)
        if self._state.get("method") == "sqd":
            self._draw_sqd_populations(p, n_pop)
            return
        pic = population_picture("appd", n_pop)
        D, f, n = pic["D"], pic["fractions"], pic["n_steps"]
        inks = ([QColor(_SHORT_INK), QColor(_LONG_INK)] if n_pop == 2 else
                [QColor(_SHORT_INK), QColor(_ACCENT_INK), QColor(_LONG_INK)])
        box = self._inner()
        plot = QRectF(box.left(), box.top() + 6, box.width(), box.height() - 30)
        lo, hi = math.log10(0.005), math.log10(8.0)

        def x_of(d):
            return plot.left() + (math.log10(d) - lo) / (hi - lo) * plot.width()

        def dens(k, d):
            # f_k * Gamma(d; n, D_k / n), shown per log-D (times d) so peaks read fairly.
            scale = D[k] / n
            return f[k] * d ** n / (math.gamma(n) * scale ** n) * math.exp(-d / scale)

        xs = [10 ** (lo + (hi - lo) * i / 240) for i in range(241)]
        top = max(dens(k, d) for k in range(n_pop) for d in xs) or 1.0
        for k in range(n_pop):
            pts = [QPointF(x_of(d), plot.bottom() - dens(k, d) / top * plot.height() * 0.92)
                   for d in xs]
            path = QPainterPath(QPointF(pts[0].x(), plot.bottom()))
            for q in pts:
                path.lineTo(q)
            path.lineTo(QPointF(pts[-1].x(), plot.bottom()))
            fill = QColor(inks[k])
            fill.setAlphaF(0.18)
            p.setPen(Qt.NoPen)
            p.setBrush(fill)
            p.drawPath(path)
            pen = QPen(inks[k])
            pen.setWidthF(1.8)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            for a, b in zip(pts, pts[1:]):
                p.drawLine(a, b)
        cuts = pic["cuts"]
        self._draw_cuts_and_axis(p, plot, x_of, cuts, (0.01, 0.1, 1.0),
                                 "apparent D, µm²/s", "{:.2g}", n_pop,
                                 ("slow", "fast") if n_pop == 2 else ("slow", "middle", "fast"))

    def _draw_split(self, p: QPainter) -> None:
        """The Split page's picture: the shown condition's OWN histogram (`spatial_map_view.split_preview`) on a log
        axis -- SQD's jump lengths, apparent diffusion's per-track mean D -- with the fitted populations where they
        sit (SQD: each one's expected bars as a curve; apparent D: a line at each D) and where the fitted maps cut
        (dashed); under Manual, each range shaded in its map's ink with the share it holds."""
        box = self._inner()
        plot = QRectF(box.left(), box.top() + 16, box.width(), box.height() - 52)
        p.setFont(ty.ui("xs", px=text_px(0.66)))
        pre = self._state.get("preview")
        name = self._state.get("preview_condition") or "the condition"
        if not isinstance(pre, dict) or pre.get("pending") or "error" in pre:
            p.setPen(QColor(self.t.muted_foreground))
            text = (f"The histogram could not be read: {pre['error']}" if isinstance(pre, dict) and "error" in pre
                    else f"Reading the histogram of {name}\u2026" if isinstance(pre, dict)
                    else "Load a condition to see its histogram.")
            p.drawText(plot, Qt.AlignCenter | Qt.TextWordWrap, text)
            return
        edges, counts = pre["edges"], pre["counts"]
        lo, hi = math.log10(edges[0]), math.log10(edges[-1])

        def x_of(v):
            v = min(max(float(v), edges[0]), edges[-1])
            return plot.left() + (math.log10(v) - lo) / (hi - lo) * plot.width()

        curves = pre.get("curves") or []
        top = max([*counts, *(c for curve in curves for c in curve)] or [1]) or 1
        inks = [QColor(_SHORT_INK), QColor(_ACCENT_INK), QColor(_LONG_INK), QColor(_CUT_INK)]
        manual = self._state.get("split_chosen") == "manual"
        ranges = self._state.get("ranges_typed") or []
        if manual:
            shares = range_shares(pre, ranges)
            for index, ((a, b), share) in enumerate(zip(ranges, shares)):
                x0, x1 = x_of(max(a, edges[0])), x_of(min(b, edges[-1]))
                if x1 <= x0:
                    continue
                fill = QColor(inks[index % len(inks)])
                fill.setAlphaF(0.16)
                p.fillRect(QRectF(x0, plot.top(), x1 - x0, plot.height()), fill)
                p.setPen(inks[index % len(inks)])
                p.drawText(QRectF(x0, plot.top() - 14, max(40.0, x1 - x0), 12), Qt.AlignHCenter | Qt.AlignBottom,
                           f"map {index + 1}: {100 * share:.0f} %")
        bar = QColor(self.t.muted_foreground)
        bar.setAlphaF(0.55)
        p.setPen(Qt.NoPen)
        p.setBrush(bar)
        for (a, b), count in zip(zip(edges, edges[1:]), counts):
            if count <= 0:
                continue
            h = count / top * plot.height()
            p.drawRect(QRectF(x_of(a) + 0.3, plot.bottom() - h, max(1.0, x_of(b) - x_of(a) - 0.6), h))
        for k, curve in enumerate(curves):
            pen = QPen(_population_inks(len(curves))[k])
            pen.setWidthF(1.8)
            p.setPen(pen)
            pts = [QPointF(x_of(math.sqrt(a * b)), plot.bottom() - c / top * plot.height())
                   for (a, b), c in zip(zip(edges, edges[1:]), curve)]
            for u, v in zip(pts, pts[1:]):
                p.drawLine(u, v)
        if not curves:
            # apparent D: a line at each fitted population's D (fitted on sub-tracks, so no curve over track means)
            for k, (d, f) in enumerate(zip(pre.get("D") or (), pre.get("fractions") or ())):
                pen = QPen(_population_inks(len(pre["D"]))[k])
                pen.setWidthF(1.8)
                p.setPen(pen)
                x = x_of(d)
                p.drawLine(QPointF(x, plot.top() + 12), QPointF(x, plot.bottom()))
                p.drawText(QRectF(x - 40, plot.top(), 80, 12), Qt.AlignHCenter | Qt.AlignTop,
                           f"{d:.2g} ({100 * f:.0f} %)")
        if not manual:
            for cut in pre.get("cuts") or ():
                pen = QPen(QColor(self.t.foreground))
                pen.setWidthF(1.3)
                pen.setDashPattern([3, 3])
                p.setPen(pen)
                x = x_of(cut)
                p.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
                p.drawText(QRectF(x - 40, plot.top() - 14, 80, 12), Qt.AlignHCenter | Qt.AlignBottom,
                           f"cut {cut:.3g}")
        axis = QPen(QColor(self.t.border))
        axis.setWidthF(1.0)
        p.setPen(axis)
        p.drawLine(QPointF(plot.left(), plot.bottom()), QPointF(plot.right(), plot.bottom()))
        p.setPen(QColor(self.t.muted_foreground))
        for decade in range(math.ceil(lo), math.floor(hi) + 1):
            x = x_of(10.0 ** decade)
            p.drawLine(QPointF(x, plot.bottom()), QPointF(x, plot.bottom() + 3))
            p.drawText(QRectF(x - 30, plot.bottom() + 4, 60, 12), Qt.AlignHCenter | Qt.AlignTop,
                       f"{10.0 ** decade:g}")
        what = ("jump length, nm" if pre.get("method") == "sqd" else "apparent D per track, µm²/s")
        p.drawText(QRectF(plot.left(), plot.bottom() + 18, plot.width(), 12), Qt.AlignHCenter | Qt.AlignTop,
                   f"{name} \u00b7 {pre['n']:,} {'jumps' if pre.get('method') == 'sqd' else 'tracks'} \u00b7 "
                   f"{what} (log scale)")

    def _draw_sqd_populations(self, p: QPainter, n_pop: int) -> None:
        """SQD's populations over JUMP LENGTH (nm, log axis): each a Rayleigh curve
        `f_k (2r/m_k) exp(-r²/m_k)`, `m_k = 4 D_k τ`, cut where neighbours cross."""
        pic = population_picture("sqd", n_pop)
        D, f, tau = pic["D"], pic["fractions"], pic["tau_s"]
        inks = _population_inks(n_pop)
        box = self._inner()
        plot = QRectF(box.left(), box.top() + 6, box.width(), box.height() - 30)
        lo, hi = math.log10(10.0), math.log10(1500.0)

        def x_of(r_nm):
            return plot.left() + (math.log10(r_nm) - lo) / (hi - lo) * plot.width()

        def dens(k, r_nm):
            r = r_nm / 1000.0
            m = 4.0 * D[k] * tau
            # per log-r (times r), so a peak reads at its own scale
            return f[k] * (2.0 * r * r / m) * math.exp(-r * r / m)

        rs = [10 ** (lo + (hi - lo) * i / 240) for i in range(241)]
        top = max(dens(k, r) for k in range(n_pop) for r in rs) or 1.0
        for k in range(n_pop):
            pts = [QPointF(x_of(r), plot.bottom() - dens(k, r) / top * plot.height() * 0.92)
                   for r in rs]
            path = QPainterPath(QPointF(pts[0].x(), plot.bottom()))
            for q in pts:
                path.lineTo(q)
            path.lineTo(QPointF(pts[-1].x(), plot.bottom()))
            fill = QColor(inks[k])
            fill.setAlphaF(0.18)
            p.setPen(Qt.NoPen)
            p.setBrush(fill)
            p.drawPath(path)
            pen = QPen(inks[k])
            pen.setWidthF(1.8)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            for a, b in zip(pts, pts[1:]):
                p.drawLine(a, b)
        cuts = pic["cuts"]
        self._draw_cuts_and_axis(p, plot, x_of, cuts, (30.0, 100.0, 300.0, 1000.0),
                                 "jump length, nm", "{:.0f} nm", n_pop, _sqd_population_names(n_pop))

    def _draw_cuts_and_axis(self, p, plot, x_of, cuts, ticks, axis_title, cut_fmt,
                            n_pop, names) -> None:
        """The dashed cuts, the axis and the one-line legend under both pictures."""
        ink = QColor(self.t.foreground)
        p.setFont(ty.ui("xs", px=text_px(0.66)))
        for cut in cuts:
            pen = QPen(ink)
            pen.setWidthF(1.3)
            pen.setDashPattern([3, 3])
            p.setPen(pen)
            x = x_of(cut)
            p.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
            p.drawText(QRectF(x - 40, plot.top() - 2, 80, 12),
                       Qt.AlignHCenter | Qt.AlignTop, "cut " + cut_fmt.format(cut))
        axis = QPen(QColor(self.t.border))
        axis.setWidthF(1.0)
        p.setPen(axis)
        p.drawLine(QPointF(plot.left(), plot.bottom()), QPointF(plot.right(), plot.bottom()))
        p.setPen(QColor(self.t.muted_foreground))
        for d in ticks:
            x = x_of(d)
            p.drawLine(QPointF(x, plot.bottom()), QPointF(x, plot.bottom() + 3))
            p.drawText(QRectF(x - 30, plot.bottom() + 4, 60, 12),
                       Qt.AlignHCenter | Qt.AlignTop, f"{d:g}")
        # the axis title on the legend's line, right-aligned: on the ticks' line it ran over the last tick ("1000")
        p.drawText(QRectF(plot.left(), plot.bottom() + 16, plot.width(), 12),
                   Qt.AlignRight | Qt.AlignTop, axis_title)
        legend = "  ·  ".join(names)
        p.drawText(QRectF(plot.left(), plot.bottom() + 16, plot.width(), 12),
                   Qt.AlignLeft | Qt.AlignTop,
                   f"one map per population: {legend}")

    def _outline(self, p: QPainter, box: QRectF, colour=None, width=1.0):
        pen = QPen(QColor(colour or self.t.foreground))
        pen.setWidthF(width)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawPath(_capsule_path(box))

    # -- one method per setting --------------------------------------------- #
    def _draw_sla(self, p: QPainter) -> None:
        """The SLA settings' picture: an INVENTED track (`spatial_map_view.SLA_EXAMPLE_UM`)
        whose dwell is found by the real detector at this dialog's radius and dwell
        (`sla_example`), drawn in v2's colours -- the whole track green when it holds a
        dwell (v2's mixed behaviour), blue when it does not (free), the dwelling points and
        steps red -- with the radius as a circle on the dwell. Drag the radius and watch the
        dwell appear, grow and swallow the free steps."""
        radius = float(self._state.get("radius_nm") or 106)
        steps = int(self._state.get("min_steps") or 5)
        example = sla_example(radius, steps)
        box = self._capsule_box()
        length = float(DEPOSIT_LENGTH_UM)
        width = float(DEPOSIT_WIDTH_UM)
        pts = [QPointF(box.left() + a / length * box.width(),
                       box.top() + (0.5 - c / width) * box.height())
               for a, c in example["points"]]
        dwell = example["dwelling"]
        track_ink = QColor(_SLA_MIXED_INK if example["has_dwell"] else _SLA_FREE_INK)
        p.save()
        p.setClipPath(_capsule_path(box))
        for i in range(len(pts) - 1):
            pen = QPen(QColor(_SLA_DWELL_INK) if dwell[i] and dwell[i + 1] else track_ink)
            pen.setWidthF(2.2)
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            p.drawLine(pts[i], pts[i + 1])
        p.setPen(Qt.NoPen)
        for point, flagged in zip(pts, dwell):
            p.setBrush(QColor(_SLA_DWELL_INK) if flagged else QColor(self.t.foreground))
            p.drawEllipse(point, 2.2, 2.2)
        if example["has_dwell"]:
            inside = [q for q, flagged in zip(pts, dwell) if flagged]
            centre = QPointF(sum(q.x() for q in inside) / len(inside),
                             sum(q.y() for q in inside) / len(inside))
            r = radius / 1000.0 / length * box.width()
            ring = QPen(QColor(self.t.muted_foreground))
            ring.setWidthF(1.0)
            ring.setDashPattern([3, 3])
            p.setPen(ring)
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(centre, r, r)
        p.restore()
        self._outline(p, box)
        # The legend under the cell, v2's three colours.
        top = box.bottom() + 12.0
        p.setFont(ty.ui("xs", px=text_px(0.66)))
        fm = QFontMetricsF(p.font())
        x = box.left()
        for text, ink in (("free track", _SLA_FREE_INK),
                          ("mixed-behaviour track", _SLA_MIXED_INK),
                          ("dwelling", _SLA_DWELL_INK)):
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(ink))
            p.drawRoundedRect(QRectF(x, top + 3.0, 8.0, 8.0), 1.5, 1.5)
            x += 12.0
            w = fm.horizontalAdvance(text)
            p.setPen(QColor(self.t.muted_foreground))
            p.drawText(QRectF(x, top, w + 1, 14.0), Qt.AlignLeft | Qt.AlignVCenter, text)
            x += w + 14.0
        verdict = (f"a dwell of {sum(dwell)} points at {radius:g} nm, ≥ {steps} steps"
                   if example["has_dwell"] else
                   f"no dwell at {radius:g} nm and ≥ {steps} steps: a free track")
        p.setPen(QColor(self.t.muted_foreground))
        p.drawText(QRectF(box.left(), top + 16.0, box.width(), 14.0),
                   Qt.AlignLeft | Qt.AlignVCenter, verdict + " · invented example")

    _draw_radius = _draw_sla
    _draw_dwell = _draw_sla

    def _draw_lag(self, p: QPainter) -> None:
        """The time lag: one track's positions in the cell, its frame-to-frame steps faint, and the jumps the run
        uses -- each one ``lag`` frames long, overlapping -- in the accent ink. A mechanism picture."""
        lag = self._state.get("lag_frames", 1)
        lags = (1, 2, 3, 4) if str(lag) == "1-4" else (int(lag),)          # lags 1-4 together: all four
        box = self._capsule_box()
        pts = [QPointF(box.left() + u * box.width(), box.top() + v * box.height()) for u, v in _METHOD_TRACK]
        p.save()
        p.setClipPath(_capsule_path(box))
        faint = QPen(QColor(self.t.border))
        faint.setWidthF(1.2)
        p.setPen(faint)
        for a, b in zip(pts, pts[1:]):
            p.drawLine(a, b)
        jump = QPen(QColor(_ACCENT_INK))
        jump.setWidthF(2.0 if len(lags) == 1 else 1.3)
        p.setPen(jump)
        for lag in lags:
            for i in range(0, len(pts) - lag):
                p.drawLine(pts[i], pts[i + lag])
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(self.t.foreground))
        for q in pts:
            p.drawEllipse(q, 2.4, 2.4)
        p.restore()
        self._outline(p, box)
        p.setFont(ty.ui("xs", px=text_px(0.66)))
        p.setPen(QColor(self.t.muted_foreground))
        n = sum(max(0, len(pts) - each) for each in lags)
        if len(lags) > 1:
            words = f"{len(pts)} positions · {n} jumps of 1 to 4 frames, overlapping"
        elif lags[0] > 1:
            words = f"{len(pts)} positions · {n} jumps of {lags[0]} frames, overlapping"
        else:
            words = f"{len(pts)} positions · {n} jumps of 1 frame"
        p.drawText(QRectF(box.left(), box.bottom() + 12.0, box.width(), 14.0), Qt.AlignLeft | Qt.AlignVCenter, words)

    def _draw_sda(self, p: QPainter) -> None:
        """The SDA method's picture: an INVENTED track (`spatial_map_view.SDA_EXAMPLE_UM`)
        whose speeds and accelerations are computed by the maps' own functions
        (`sda_example`). Each step is drawn as wide as it is fast; a ring sits where two
        steps meet, as large as the change between them -- where the acceleration map puts
        it. In ink and greys, like every mechanism picture here: not a preview of a map."""
        example = sda_example()
        box = self._capsule_box()
        length, width = float(DEPOSIT_LENGTH_UM), float(DEPOSIT_WIDTH_UM)
        pts = [QPointF(box.left() + a / length * box.width(),
                       box.top() + (0.5 - c / width) * box.height())
               for a, c in example["points"]]
        speed, acc = example["speed"], example["acceleration"]
        top_speed = max(speed) or 1.0
        top_acc = max(acc) or 1.0
        ink = QColor(self.t.foreground)
        p.save()
        p.setClipPath(_capsule_path(box))
        for i, v in enumerate(speed):
            pen = QPen(ink)
            pen.setWidthF(1.0 + 4.0 * v / top_speed)
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            p.drawLine(pts[i], pts[i + 1])
        ring = QPen(QColor(self.t.muted_foreground))
        ring.setWidthF(1.2)
        p.setPen(ring)
        p.setBrush(Qt.NoBrush)
        for value, at in zip(acc, example["acc_at"]):
            r = 2.5 + 7.0 * value / top_acc
            p.drawEllipse(pts[at], r, r)
        p.restore()
        self._outline(p, box)
        top = box.bottom() + 12.0
        p.setFont(ty.ui("xs", px=text_px(0.66)))
        p.setPen(QColor(self.t.muted_foreground))
        p.drawText(QRectF(box.left(), top, box.width(), 14.0),
                   Qt.AlignLeft | Qt.AlignVCenter, "  ·  ".join(SDA_EXAMPLE_KEY))
        p.drawText(QRectF(box.left(), top + 16.0, box.width(), 14.0),
                   Qt.AlignLeft | Qt.AlignVCenter, sda_example_line(example))

    def _draw_adaptive(self, p: QPainter) -> None:
        """Adaptive smoothing: an INVENTED field of steps on the dialog's own grid, put
        through the run's own rule (`adaptive_example` -> `adaptive_smoothing`). A bin that
        kept h0 is grey; a widened one is in the dialog's accent ink, darker the wider its
        kernel; a bin too thin even at the widest (or empty) is hatched blank, as it is
        grey on the map. A mechanism picture, never a preview of a map."""
        state = self._state
        example = adaptive_example(
            bin_nm=max(1.0, float(state.get("bin_nm", BIN_NM_DEFAULT))),
            cell_length_um=state.get("cell_length_um", DEPOSIT_LENGTH_UM),
            cell_width_um=state.get("cell_width_um", DEPOSIT_WIDTH_UM),
            adaptive=bool(state.get("value_adaptive", True)),
            n_min=int(state.get("n_min", 0)))
        box = self._capsule_box()
        nx, ny = example["nx"], example["ny"]
        cw, ch = box.width() / nx, box.height() / ny
        hatch = QBrush(QColor(self.t.muted_foreground), Qt.BDiagPattern)
        p.save()
        p.setClipPath(_capsule_path(box))
        p.setPen(Qt.NoPen)
        for iy in range(ny):
            row = ny - 1 - iy                       # row 0 of the grid is the bottom wall
            for ix in range(nx):
                if not example["inside"][iy][ix]:
                    continue
                cell = QRectF(box.left() + ix * cw, box.top() + row * ch, cw + 0.4, ch + 0.4)
                m = example["multiple"][iy][ix]
                if m is None:
                    p.setBrush(QColor(self.t.card))
                    p.drawRect(cell)
                    p.setBrush(hatch)
                elif m <= 1.0 + 1e-9:
                    ink = QColor(self.t.foreground)
                    ink.setAlphaF(0.14)
                    p.setBrush(ink)
                else:
                    ink = QColor(_ACCENT_INK)
                    ink.setAlphaF(min(0.9, 0.2 + 0.35 * (float(m) - 1.0)))
                    p.setBrush(ink)
                p.drawRect(cell)
        # the value bins' edges, so a block of small bins reads as one value bin
        f = example["factor"]
        pen = QPen(QColor(self.t.card))
        pen.setWidthF(1.0)
        p.setPen(pen)
        for k in range(f, nx, f):
            x = box.left() + k * cw
            p.drawLine(QPointF(x, box.top()), QPointF(x, box.bottom()))
        for k in range(f, ny, f):
            y = box.top() + k * ch
            p.drawLine(QPointF(box.left(), y), QPointF(box.right(), y))
        p.restore()
        self._outline(p, box)
        top = box.bottom() + 12.0
        p.setFont(ty.ui("xs", px=text_px(0.66)))
        p.setPen(QColor(self.t.muted_foreground))
        p.drawText(QRectF(box.left(), top, box.width(), 14.0),
                   Qt.AlignLeft | Qt.AlignVCenter, "  ·  ".join(ADAPTIVE_EXAMPLE_KEY))
        p.drawText(QRectF(box.left(), top + 16.0, box.width(), 14.0),
                   Qt.AlignLeft | Qt.AlignVCenter, adaptive_example_line(example))

    def _draw_jump(self, p: QPainter) -> None:
        """The one capsule this has always been, now with a grid on it: a
        long jump near the wall can only travel inward -- which is the whole
        of why the convention moves the answer -- and the grid is what shows
        it moves to a different BIN, not just a different point on the line.

        One illustration, not three: the highlighted cell is a single glow
        that SLIDES across this one grid when the option changes
        (`_animate_jump`), rather than three small capsules side by side.
        Deliberately coarse -- six columns, four rows, from `_JUMP_LANDING`
        -- the same "schematic, not a data preview" every diagram here
        already is; real bins are far finer.

        Under the cell (in `_JUMP_AXIS_BAND`, below the plate, so the cell is
        the same size as every other diagram's), a time axis whose ticks sit
        directly below the three points, so the order start -> end is readable
        without the animation.
        While the intro runs (`_intro_p` is not None) the particle slides from
        start to end and back, and the axis marker follows it.
        """
        box = self._capsule_box()
        self._fill_cell(p, box)
        a = QPointF(box.left() + box.width() * _JUMP_A_FRAC[0],
                    box.top() + box.height() * _JUMP_A_FRAC[1])
        b = QPointF(box.left() + box.width() * _JUMP_B_FRAC[0],
                    box.top() + box.height() * _JUMP_B_FRAC[1])
        mid = QPointF((a.x() + b.x()) / 2.0, (a.y() + b.y()) / 2.0)
        chosen = self._state.get("step_position", STEP_POSITIONS[0])
        intro = self._intro_p

        p.save()
        p.setClipPath(_capsule_path(box))

        cw, ch = box.width() / _JUMP_COLS, box.height() / _JUMP_ROWS
        if intro is None:
            # THE GLOW, first -- under the grid lines and the jump, the way a
            # highlighted cell sits under the graph paper drawn over it.
            glow = QColor(self.t.foreground)
            glow.setAlphaF(0.16)
            p.setPen(Qt.NoPen)
            p.setBrush(glow)
            gx = box.left() + self._jump_frac.x() * box.width() - cw / 2.0
            gy = box.top() + self._jump_frac.y() * box.height() - ch / 2.0
            p.drawRect(QRectF(gx, gy, cw + 0.4, ch + 0.4))

        grid = QPen(QColor(self.t.border))
        grid.setWidthF(0.6)
        p.setPen(grid)
        for k in range(1, _JUMP_COLS):
            x = box.left() + box.width() * k / _JUMP_COLS
            p.drawLine(QPointF(x, box.top()), QPointF(x, box.bottom()))
        for k in range(1, _JUMP_ROWS):
            y = box.top() + box.height() * k / _JUMP_ROWS
            p.drawLine(QPointF(box.left(), y), QPointF(box.right(), y))

        axis = QPen(QColor(self.t.border))
        axis.setWidthF(1.0)
        axis.setDashPattern([2, 4])
        p.setPen(axis)
        p.drawLine(QPointF(box.left(), box.center().y()),
                   QPointF(box.right(), box.center().y()))

        pen = QPen(QColor(self.t.foreground))
        pen.setWidthF(1.6)
        if intro is None:
            p.setPen(pen)
            p.drawLine(a, b)
            for key, point in zip(STEP_POSITIONS, (a, mid, b)):
                on = key == chosen
                if on:
                    p.setBrush(QColor(self.t.foreground))
                    p.setPen(Qt.NoPen)
                    p.drawEllipse(point, 5.5, 5.5)
                else:
                    p.setBrush(QColor(self.t.card))
                    ring = QPen(QColor(self.t.muted_foreground))
                    ring.setWidthF(1.2)
                    p.setPen(ring)
                    p.drawEllipse(point, 3.4, 3.4)
        else:
            h = QPointF(a.x() + (b.x() - a.x()) * intro,
                        a.y() + (b.y() - a.y()) * intro)
            p.setPen(pen)
            p.drawLine(a, h)
            ghost = QPen(QColor(self.t.border))
            ghost.setWidthF(1.2)
            ghost.setDashPattern([2, 3])
            p.setPen(ghost)
            p.drawLine(h, b)
            ring = QPen(QColor(self.t.muted_foreground))
            ring.setWidthF(1.2)
            for point in (a, mid, b):
                p.setBrush(QColor(self.t.card))
                p.setPen(ring)
                p.drawEllipse(point, 3.4, 3.4)
            p.setBrush(QColor(self.t.foreground))
            p.setPen(Qt.NoPen)
            p.drawEllipse(h, 5.5, 5.5)
        p.restore()
        self._outline(p, box)
        self._draw_time_axis(p, box, (a.x(), mid.x(), b.x()), chosen, intro)

    def _draw_time_axis(self, p: QPainter, box: QRectF, xs, chosen: str,
                        intro: float | None) -> None:
        """The time axis under the jump: ticks directly below start, midpoint
        and end, an arrow to the right, and 'one frame (τ)' over the start-end
        span, because start and end are two consecutive frames."""
        y = box.bottom() + 22.0
        x0, x1 = box.left(), box.right()
        xa, xm, xb = xs
        line = QPen(QColor(self.t.border))
        line.setWidthF(1.0)
        p.setPen(line)
        p.setBrush(Qt.NoBrush)
        p.drawLine(QPointF(x0, y), QPointF(x1, y))
        p.drawLine(QPointF(x1, y), QPointF(x1 - 5, y - 3.5))
        p.drawLine(QPointF(x1, y), QPointF(x1 - 5, y + 3.5))

        p.setFont(ty.ui("xs", px=text_px(0.66)))
        p.setPen(QColor(self.t.muted_foreground))
        p.drawText(QRectF(x1 - 60, y - 16, 60, 12),
                   Qt.AlignRight | Qt.AlignVCenter, "time")
        p.drawText(QRectF(xa, y - 16, xb - xa, 12),
                   Qt.AlignHCenter | Qt.AlignVCenter, "one frame (\u03c4)")

        xp = None
        if intro is not None:
            xp = xa + (xb - xa) * intro
            fill = QPen(QColor(self.t.foreground))
            fill.setWidthF(1.6)
            p.setPen(fill)
            p.drawLine(QPointF(xa, y), QPointF(xp, y))

        for key, x, text in zip(STEP_POSITIONS, (xa, xm, xb), STEP_LABELS):
            on = intro is None and key == chosen
            if on:
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(self.t.foreground))
                p.drawEllipse(QPointF(x, y), 4.0, 4.0)
            else:
                ring = QPen(QColor(self.t.muted_foreground))
                ring.setWidthF(1.2)
                p.setPen(ring)
                p.setBrush(QColor(self.t.card))
                p.drawEllipse(QPointF(x, y), 3.0, 3.0)
            p.setFont(ty.ui("xs", "medium" if on else "normal", px=text_px(0.66)))
            p.setPen(QColor(self.t.foreground if on else self.t.muted_foreground))
            p.drawText(QRectF(x - 40, y + 6, 80, 12),
                       Qt.AlignHCenter | Qt.AlignTop, text)

        if xp is not None:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(self.t.foreground))
            p.drawEllipse(QPointF(xp, y), 4.0, 4.0)

    def _draw_show(self, p: QPainter) -> None:
        """Show (22 Nov 2026, `spatialangles`): under Steps three steps of a track, each a dot where it is counted;
        under Turning back two of them meeting at their shared localisation, the angle between them marked, the
        turn back drawn in the slow state's ink and a straight run in muted ink beside it."""
        box = self._inner()
        ink = _population_inks(2)[0]
        muted = QColor(self.t.muted_foreground)
        turning = self._state.get("hmm_show") in ("angles", "turning")
        cy = box.center().y()
        left = QPointF(box.left() + box.width() * 0.12, cy + box.height() * 0.18)
        vertex = QPointF(box.left() + box.width() * 0.42, cy - box.height() * 0.12)
        back = QPointF(box.left() + box.width() * 0.20, cy - box.height() * 0.30)
        straight = [QPointF(box.left() + box.width() * x, cy + box.height() * y)
                    for x, y in ((0.58, 0.20), (0.73, 0.02), (0.88, -0.16))]
        pen = QPen(ink, 2.0)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        if turning:
            p.drawLine(left, vertex)
            p.drawLine(vertex, back)
            arc = QRectF(vertex.x() - 16, vertex.y() - 16, 32, 32)
            p.setPen(QPen(QColor(self.t.foreground), 1.2))
            p.drawArc(arc, 200 * 16, 50 * 16)
            p.setPen(QPen(muted, 1.6))
            p.drawLine(straight[0], straight[1])
            p.drawLine(straight[1], straight[2])
            p.setPen(Qt.NoPen)
            p.setBrush(ink)
            p.drawEllipse(vertex, 4.0, 4.0)
            p.setBrush(muted)
            p.drawEllipse(straight[1], 3.0, 3.0)
        else:
            p.drawLine(left, vertex)
            p.drawLine(vertex, back)
            p.setPen(Qt.NoPen)
            p.setBrush(ink)
            for dot in (left, vertex, back):
                p.drawEllipse(dot, 3.5, 3.5)

    def _draw_assignment(self, p: QPainter) -> None:
        """A JUMP-DISTANCE HISTOGRAM of the example fit and how each bar is given to the populations (asked
        16 Oct 2026: "show a jump distance histogram and how it assigns from that"). Every number comes from
        `spatial_map_view.assignment_histogram`, exact for that mixture: under SOFT a bar is split in the ratio of
        the populations' own curves at its length (drawn over the bars), which is the posterior each jump is
        weighted with; under HARD it goes wholly to the population most likely at its length, so the bars change
        colour at the dashed cut, where the curves cross.

        Morphs rather than snaps: `self._hardness` (see `_SettingArt.__init__`) blends every bar's shares
        linearly, 0.0 being exactly the soft picture and 1.0 exactly the hard one, and the cut's alpha follows the
        same value, so it fades in on the way to hard and out on the way to soft.
        """
        n_pop = drawn_n_pop(self._state)
        pic = assignment_histogram(n_pop if n_pop in (1, 3) else 2)
        inks = _population_inks(n_pop)
        names = _sqd_population_names(n_pop)
        box = self._inner()
        plot = QRectF(box.left(), box.top() + 16, box.width(), box.height() - 46)
        edges, total, h = pic["edges_nm"], pic["total"], self._hardness
        span = edges[-1] or 1.0
        peak = max(total) or 1.0

        def x_of(r_nm):
            return plot.left() + r_nm / span * plot.width()

        def height_of(value):
            return value / peak * plot.height() * 0.94

        p.setPen(Qt.NoPen)
        for i, count in enumerate(total):
            x0, x1 = x_of(edges[i]) + 0.75, x_of(edges[i + 1]) - 0.75
            y = plot.bottom()
            for k in range(len(inks)):
                soft = pic["soft"][i][k]
                share = soft + (pic["hard"][i][k] - soft) * h
                bar = height_of(count * share)
                if bar <= 0.0:
                    continue
                p.setBrush(inks[k])
                p.drawRect(QRectF(x0, y - bar, x1 - x0, bar))
                y -= bar
        # each population's own curve, on the bars' scale, with a halo so it reads over its own colour
        for k, curve in enumerate(pic["curves"]):
            pts = [QPointF(x_of(r), plot.bottom() - height_of(v))
                   for r, v in zip(pic["curve_nm"], curve)]
            for colour, width in ((QColor(getattr(self.t, "card", self.t.background)), 3.6), (inks[k], 1.6)):
                pen = QPen(colour)
                pen.setWidthF(width)
                p.setPen(pen)
                p.setBrush(Qt.NoBrush)
                for a, b in zip(pts, pts[1:]):
                    p.drawLine(a, b)
        p.setFont(ty.ui("xs", px=text_px(0.66)))
        if h > 0.01:
            ink = QColor(self.t.foreground)
            ink.setAlphaF(min(1.0, h))
            for cut in pic["cuts_nm"]:
                pen = QPen(ink)
                pen.setWidthF(1.4)
                pen.setDashPattern([3, 3])
                p.setPen(pen)
                x = x_of(cut)
                p.drawLine(QPointF(x, plot.top() - 2), QPointF(x, plot.bottom()))
                p.drawText(QRectF(x + 3, plot.top() - 14, 90, 12), Qt.AlignLeft | Qt.AlignTop,
                           f"cut {cut:.0f} nm")
        axis = QPen(QColor(self.t.border))
        axis.setWidthF(1.0)
        p.setPen(axis)
        p.drawLine(QPointF(plot.left(), plot.bottom()), QPointF(plot.right(), plot.bottom()))
        p.setPen(QColor(self.t.muted_foreground))
        tick = 200.0 if span > 400.0 else 100.0
        r = 0.0
        while r <= span + 1e-6:
            x = x_of(r)
            p.drawLine(QPointF(x, plot.bottom()), QPointF(x, plot.bottom() + 3))
            p.drawText(QRectF(x - 30, plot.bottom() + 4, 60, 12), Qt.AlignHCenter | Qt.AlignTop, f"{r:g}")
            r += tick
        p.drawText(QRectF(plot.left(), plot.bottom() + 16, plot.width(), 12),
                   Qt.AlignRight | Qt.AlignTop, "jump length, nm")
        # the key, and that these are an example's numbers, not the data's
        x = plot.left()
        for k, name in enumerate(names):
            p.setPen(Qt.NoPen)
            p.setBrush(inks[k])
            p.drawRect(QRectF(x, plot.bottom() + 19, 8, 8))
            p.setPen(QColor(self.t.muted_foreground))
            p.drawText(QRectF(x + 11, plot.bottom() + 16, 60, 12), Qt.AlignLeft | Qt.AlignTop, name)
            x += 13 + p.fontMetrics().horizontalAdvance(name) + 10
        shares = " / ".join(f"{100 * f:.0f} %" for f in pic["fractions"])
        p.drawText(QRectF(plot.left(), box.top() - 2, plot.width(), 12), Qt.AlignRight | Qt.AlignTop,
                   f"example fit, {shares}")

    def _draw_fold(self, p: QPainter) -> None:
        """The same cloud, raw and reflected into all four quarters -- so
        the symmetry folding creates is visibly created, not found. Toggling Fold
        plays the reflection as two flips (`_animate_fold`)."""
        box = self._capsule_box()
        self._fill_cell(p, box)
        p.save()
        p.setClipPath(_capsule_path(box))
        pen = QPen(QColor(self.t.border))
        pen.setWidthF(1.0)
        pen.setDashPattern([2, 4])
        p.setPen(pen)
        p.drawLine(QPointF(box.center().x(), box.top()),
                   QPointF(box.center().x(), box.bottom()))
        p.drawLine(QPointF(box.left(), box.center().y()),
                   QPointF(box.right(), box.center().y()))

        # The two FLIPS, from `_fold_blend` (0 = Off, 1 = On): over its first half the
        # cloud's copy swings across the midcell line, over its second half both swing
        # across the long axis. A copy fades in over the first part of its own flip, so
        # at 0 and 1 the picture is exactly the resting unfolded / folded one.
        blend = float(getattr(self, "_fold_blend",
                              1.0 if self._state.get("mirrored") else 0.0))
        ease = QEasingCurve(QEasingCurve.InOutCubic)
        s1 = ease.valueForProgress(min(1.0, max(0.0, blend * 2.0)))
        s2 = ease.valueForProgress(min(1.0, max(0.0, blend * 2.0 - 1.0)))
        alpha = 0.55
        layers = [(1.0, 1.0, alpha)]                         # (x sign, y sign, alpha)
        if s1 > 0:
            layers.append((1.0 - 2.0 * s1, 1.0, alpha * min(1.0, s1 / 0.4)))
        if s2 > 0:
            fade = min(1.0, s2 / 0.4)
            layers += [(1.0, 1.0 - 2.0 * s2, alpha * fade),
                       (-1.0, 1.0 - 2.0 * s2, alpha * fade)]
        p.setPen(Qt.NoPen)
        for sx, sy, a in layers:
            ink = QColor(self.t.foreground)
            ink.setAlphaF(a)
            p.setBrush(ink)
            for fx, fy in _FOLD_CLOUD:
                p.drawEllipse(QPointF(box.left() + box.width() * (0.5 + sx * fx / 2),
                                      box.top() + box.height() * (0.5 + sy * fy / 2)),
                              2.0, 2.0)
        p.restore()
        self._outline(p, box)

    def _draw_frame(self, p: QPainter) -> None:
        """Both capsules, overlaid at TRUE relative scale -- drawing them the
        same size would hide the only thing this control changes.

        The CHOSEN one prints its own number, and what it means, directly
        on itself -- asked for directly, so the picture states its own
        value rather than leaving that to the caption underneath, which no
        longer needs to repeat it (see `_read_text`'s own "frame" branch,
        now empty for exactly this reason). The other capsule stays a bare
        dashed outline: it is there for scale, not to be read.

        THE CHOSEN CAPSULE SQUEEZES OR EXPANDS rather than snapping when
        the selection changes -- `self._frame_size` (see `__init__`) is the
        CONTINUOUS value actually drawn and printed; a value sitting still
        reproduces either resting picture exactly, and only a real toggle
        (`show_setting`) ever moves it. The other capsule is never
        animated: its own size is a fixed fact the moment it stops being
        the chosen one, nothing about it is changing.

        THE GHOST IS DRAWN IN `muted_foreground`, not `border` -- asked for
        directly after the dashed comparison outline read as too faint to
        see. `border` is this file's own convention for the QUIETEST marks
        (grid lines, axes); a capsule meant to be compared against needs to
        clear that bar.
        """
        measured = self._state.get("measured")
        chosen = self._state.get("frame_source", "deposit")
        outer = self._capsule_box()
        self._fill_cell(p, outer)
        px = outer.width() / DEPOSIT_LENGTH_UM
        cx, cy = outer.center().x(), outer.center().y()
        pairs = [((DEPOSIT_LENGTH_UM, DEPOSIT_WIDTH_UM), chosen == "deposit",
                 "predefined value")]
        if measured:
            sub = (f"median of {measured['n_cells']} cells"
                  if measured.get("n_cells") else "")
            pairs.append(((measured["cell_length_um"], measured["cell_width_um"]),
                          chosen == "measured", sub))
        for (true_length, true_width), on, sub_label in pairs:
            length, width = ((self._frame_size.x(), self._frame_size.y())
                            if on else (true_length, true_width))
            bw, bh = length * px, width * px
            # A COMMON CENTRE, inside the left-aligned outer box: two
            # capsules sharing a left edge would put the whole difference
            # at one end and read as one being longer, not as two sizes.
            box = QRectF(cx - bw / 2.0, cy - bh / 2.0, bw, bh)
            pen = QPen(QColor(self.t.foreground if on else self.t.muted_foreground))
            pen.setWidthF(1.6 if on else 1.3)
            if not on:
                pen.setDashPattern([5, 4])
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawPath(_capsule_path(box))
            if not on:
                continue
            # Inset by THIS capsule's own cap radius, not the outer box's --
            # `_draw_bin`'s own lesson about a capsule's curved ends applied
            # to text here instead of a single square.
            margin = box.height() / 2.0
            number_h, text_gap, label_h = 28.0, 2.0, 18.0
            total = number_h + text_gap + label_h
            top = cy - total / 2.0
            number_rect = QRectF(box.left() + margin, top,
                                 box.width() - 2.0 * margin, number_h)
            p.setFont(ty.ui("base", "semibold", px=text_px(1.05)))
            p.setPen(QColor(self.t.foreground))
            p.drawText(number_rect, Qt.AlignHCenter | Qt.AlignVCenter,
                       f"{length:.2f} × {width:.2f} µm")
            if sub_label:
                label_rect = QRectF(number_rect.left(), number_rect.bottom() + text_gap,
                                    number_rect.width(), label_h)
                p.setFont(ty.ui("xs", px=text_px(0.7)))
                p.setPen(QColor(self.t.muted_foreground))
                p.drawText(label_rect, Qt.AlignHCenter | Qt.AlignVCenter, sub_label)

    def _area(self) -> QRectF:
        """The whole widget inside the plate's padding, the time-axis band included:
        the cells and long-axis pictures stack capsules and need the height."""
        return QRectF(self.PAD, self.PAD, self.width() - 2.0 * self.PAD,
                      self.height() - 2.0 * self.PAD)

    def _draw_cells(self, p: QPainter) -> None:
        """The three length classes, one capsule each at TRUE relative scale (median
        length and width of the class), stacked. The chosen one is solid with its range
        and count, the others dashed for scale; "All" draws all three solid. Without
        measured lengths there are no classes, and the one capsule says so."""
        measured = self._state.get("measured") or {}
        classes = measured.get("classes") or {}
        rows = [(name, classes[name]) for name in CELL_CLASSES[1:] if classes.get(name)]
        if not rows:
            box = self._capsule_box()
            self._outline(p, box, self.t.muted_foreground)
            p.setFont(ty.ui("xs", px=text_px(0.75)))
            p.setPen(QColor(self.t.muted_foreground))
            p.drawText(box, Qt.AlignCenter,
                       "cell lengths not measured here\nevery cell is drawn")
            return
        chosen = self._state.get("cell_class", "all")
        cuts = measured.get("length_cuts_um")
        area = self._area()
        label_w, gap = 190.0, 10.0
        longest = max(c["cell_length_um"] for _, c in rows)
        widest = max(c["cell_width_um"] for _, c in rows)
        px = min((area.width() - label_w - 14.0) / longest,
                 (area.height() - gap * (len(rows) - 1)) / (len(rows) * widest))
        row_h = widest * px
        top = area.top() + (area.height() - len(rows) * row_h - gap * (len(rows) - 1)) / 2
        for i, (name, cls) in enumerate(rows):
            on = chosen in ("all", name)
            ink = QColor(self.t.foreground if on else self.t.muted_foreground)
            cy = top + i * (row_h + gap) + row_h / 2.0
            bw, bh = cls["cell_length_um"] * px, cls["cell_width_um"] * px
            pen = QPen(ink)
            pen.setWidthF(1.6 if on else 1.2)
            if not on:
                pen.setDashPattern([5, 4])
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawPath(_capsule_path(QRectF(area.left(), cy - bh / 2.0, bw, bh)))
            text = QRectF(area.left() + longest * px + 14.0, cy - row_h / 2.0,
                          label_w, row_h)
            p.setFont(ty.ui("sm", "semibold", px=text_px(0.8)) if on
                      else ty.ui("sm", px=text_px(0.8)))
            p.setPen(ink)
            p.drawText(text.adjusted(0, 0, 0, -row_h / 2.0), Qt.AlignLeft | Qt.AlignBottom,
                       f"{name.capitalize()} · {cell_class_range_text(name, cuts)}")
            p.setFont(ty.ui("xs", px=text_px(0.7)))
            p.setPen(QColor(self.t.muted_foreground))
            p.drawText(text.adjusted(0, row_h / 2.0, 0, 0), Qt.AlignLeft | Qt.AlignTop,
                       f"median {cls['cell_length_um']:.2f} µm · {cls['n_cells']} cells")

    def _axis_marker(self, p: QPainter, x: float, y: float, filled: bool) -> None:
        """The example molecule: FILLED for the short cell, a RING for the long one, so
        the two are told apart without colour, and a ring around a dot reads as both
        landing on one place."""
        ink = QColor(self.t.foreground)
        if filled:
            p.setPen(Qt.NoPen)
            p.setBrush(ink)
            p.drawEllipse(QPointF(x, y), 3.2, 3.2)
        else:
            pen = QPen(ink)
            pen.setWidthF(1.4)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(x, y), 5.6, 5.6)

    def _draw_axis(self, p: QPainter) -> None:
        """The drawn outline on top, a short and a long cell under it, all at TRUE relative
        scale (`spatial_map_view.long_axis_picture`, computed with the maps' own
        `long_axis_x`). In each cell: the outline's bin edges where they fall, the part
        the drawn POLES take (filled), the cell's MIDPOINT (dashed) and one molecule the
        same distance from the tip. On the outline: where that molecule lands from each
        cell. Normalised, the two land apart and each pole takes a fraction of its cell;
        pole-anchored, they land together and each pole takes exactly the cell's cap."""
        model_l = float(self._state.get("cell_length_um", DEPOSIT_LENGTH_UM))
        model_w = float(self._state.get("cell_width_um", DEPOSIT_WIDTH_UM))
        pic = long_axis_picture(self._state.get("long_axis", "normalised"), model_l, model_w)
        n_cap = pic["n_cap_bins"]
        rows = [{"length_um": model_l, "width_um": model_w,
                 "edges_um": pic["model_edges_um"],
                 "pole_um": pic["model_edges_um"][n_cap],
                 "far_pole_um": model_l - pic["model_edges_um"][-1 - n_cap]},
                *pic["cells"]]
        area = self._area()
        label_w, gap = 170.0, 12.0
        longest = max(r["length_um"] for r in rows)
        widest = max(r["width_um"] for r in rows)
        px = min((area.width() - label_w - 14.0) / longest,
                 (area.height() - gap * (len(rows) - 1)) / (len(rows) * widest))
        row_h = widest * px
        top = area.top() + (area.height() - len(rows) * row_h - gap * (len(rows) - 1)) / 2.0
        fill = QColor(_ACCENT_INK)
        fill.setAlphaF(0.35)
        dash = QPen(QColor(self.t.muted_foreground))
        dash.setWidthF(1.0)
        dash.setDashPattern([3, 3])
        for i, row in enumerate(rows):
            length, width = row["length_um"], row["width_um"]
            cy = top + i * (row_h + gap) + row_h / 2.0
            box = QRectF(area.left(), cy - width * px / 2.0, length * px, width * px)
            p.save()
            p.setClipPath(_capsule_path(box))
            p.fillRect(QRectF(box.left(), box.top(), row["pole_um"] * px, box.height()), fill)
            p.fillRect(QRectF(box.right() - row["far_pole_um"] * px, box.top(),
                              row["far_pole_um"] * px, box.height()), fill)
            pen = QPen(QColor(self.t.border))
            pen.setWidthF(0.8)
            p.setPen(pen)
            for e in row["edges_um"][1:-1]:
                x = box.left() + e * px
                p.drawLine(QPointF(x, box.top()), QPointF(x, box.bottom()))
            p.setPen(dash)
            mid = box.left() + length / 2.0 * px
            p.drawLine(QPointF(mid, box.top()), QPointF(mid, box.bottom()))
            p.restore()
            self._outline(p, box, width=1.6 if i == 0 else 1.0)
            if i == 0:
                for k, cell in enumerate(pic["cells"]):
                    self._axis_marker(p, box.left() + cell["lands_um"] * px, cy, k == 0)
            else:
                self._axis_marker(p, box.left() + pic["marker_um"] * px, cy, i == 1)
            text = QRectF(area.left() + longest * px + 14.0, cy - row_h / 2.0,
                          label_w, row_h)
            p.setFont(ty.ui("sm", "semibold", px=text_px(0.8)))
            p.setPen(QColor(self.t.foreground))
            title_rect = text.adjusted(0, 0, 0, -row_h / 2.0)
            if i:
                # the row's marker, drawn, in front of its name
                self._axis_marker(p, title_rect.left() + 6.0, title_rect.bottom() - 8.0, i == 1)
                p.setPen(QColor(self.t.foreground))
                title_rect = title_rect.adjusted(16.0, 0, 0, 0)
            p.drawText(title_rect, Qt.AlignLeft | Qt.AlignBottom,
                       "drawn outline" if i == 0 else f"{length:.1f} µm cell")
            p.setFont(ty.ui("xs", px=text_px(0.7)))
            if i == 0:
                a, b = (c["lands_um"] * 1000 for c in pic["cells"])
                p.setPen(QColor(self.t.muted_foreground))
                # in the order of the cells below, whose markers say which is which
                line = (f"both land {a:.0f} nm in" if abs(a - b) < 1.0 else
                        f"they land {a:.0f} and {b:.0f} nm in")
            else:
                p.setPen(QColor(_ACCENT_INK))
                line = f"each pole takes {row['pole_um'] * 1000:.0f} nm"
            p.drawText(text.adjusted(0, row_h / 2.0, 0, 0), Qt.AlignLeft | Qt.AlignTop, line)

    def _draw_bin(self, p: QPainter) -> None:
        """The grid at the chosen size, with ONE camera pixel drawn to the
        same scale beside it -- which is what the number is read against."""
        box = self._capsule_box()
        length = self._state.get("cell_length_um", DEPOSIT_LENGTH_UM)
        width = self._state.get("cell_width_um", DEPOSIT_WIDTH_UM)
        bin_nm = max(1.0, float(self._state.get("bin_nm", BIN_NM_DEFAULT)))
        nx = even_bins(length * 1000.0 / bin_nm)
        ny = even_bins(width * 1000.0 / bin_nm)
        p.save()
        p.setClipPath(_capsule_path(box))
        pen = QPen(QColor(self.t.border))
        pen.setWidthF(0.6)
        p.setPen(pen)
        for k in range(1, nx):
            x = box.left() + box.width() * k / nx
            p.drawLine(QPointF(x, box.top()), QPointF(x, box.bottom()))
        for k in range(1, ny):
            y = box.top() + box.height() * k / ny
            p.drawLine(QPointF(box.left(), y), QPointF(box.right(), y))
        pixel_nm = self._state.get("pixel_size_nm")
        if pixel_nm:
            side = box.width() * float(pixel_nm) / (length * 1000.0)
            pen = QPen(QColor(_ACCENT_INK))
            pen.setWidthF(1.5)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            # Clear of the right-hand CAP: the capsule curves away there
            # and the clip takes the corner off the square. The walls are
            # straight only between the two caps, so it is inset by the cap
            # radius.
            square = QRectF(box.right() - box.height() / 2.0 - side - 6,
                            box.bottom() - side - 7, side, side)
            p.drawRect(square)
            # NAMED WHERE IT SITS. The description already says "the square
            # is one camera pixel" and that was not enough: read against a
            # grid of identical cells, an odd-coloured unlabelled square
            # looks like an artefact rather than a ruler. Reported as "why
            # is there an outlier square".
            p.setFont(ty.ui("xs", "medium", px=text_px(0.66)))
            p.drawText(QRectF(box.left(), square.center().y() - 8,
                              square.left() - box.left() - 5, 16),
                       Qt.AlignRight | Qt.AlignVCenter, "1 camera pixel")
        p.restore()
        self._outline(p, box)

    def _draw_weighting(self, p: QPainter) -> None:
        """The same three tracks counted two ways: every jump (C's bin holds 12 of
        18 votes) or every track once (1 of 3). The bins re-colour between the two
        (`_weight_blend`), and each bin's number is the one for the mode shown."""
        box = self._capsule_box()
        b = float(getattr(self, "_weight_blend", 0.0))
        card = QColor(self.t.card)
        ink = QColor(_LONG_INK)
        cw, ch = box.width() / 5.0, box.height() / 3.0
        top_share = 12.0 / _W_TOTALS[0]
        p.save()
        p.setClipPath(_capsule_path(box))
        p.setPen(Qt.NoPen)
        for (c, r), (v_steps, v_track, _l1, _l2) in _W_BINS.items():
            share = (v_steps / _W_TOTALS[0]) * (1 - b) + (v_track / _W_TOTALS[1]) * b
            p.setBrush(_mix(ink, card, min(1.0, share / top_share)))
            p.drawRect(QRectF(box.left() + c * cw, box.top() + r * ch, cw + 0.5, ch + 0.5))
        grid = QPen(QColor(self.t.border))
        grid.setWidthF(0.6)
        p.setPen(grid)
        for k in range(1, 5):
            x = box.left() + k * cw
            p.drawLine(QPointF(x, box.top()), QPointF(x, box.bottom()))
        for k in range(1, 3):
            y = box.top() + k * ch
            p.drawLine(QPointF(box.left(), y), QPointF(box.right(), y))

        def at(pt):
            return QPointF(box.left() + pt[0] / 300.0 * box.width(),
                           box.top() + pt[1] / 180.0 * box.height())

        for i, track in enumerate(_W_TRACKS):
            pen = QPen(QColor(self.t.foreground if i == 2 else _SHORT_INK))
            pen.setWidthF(1.2 if i == 2 else 1.6)
            p.setPen(pen)
            pts = [at(pt) for pt in track]
            for a, c in zip(pts, pts[1:]):
                p.drawLine(a, c)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(self.t.foreground if i == 2 else _SHORT_INK))
            for pt in pts[:-1]:
                p.drawEllipse(pt, 2.0 if i == 2 else 3.0, 2.0 if i == 2 else 3.0)
        p.setFont(ty.ui("xs", "semibold", px=text_px(0.7)))
        for (c, r), (_v1, _v2, l_steps, l_track) in _W_BINS.items():
            text = l_track if b >= 0.5 else l_steps
            w = 10 + 7 * len(text)
            # Toward the cell's inside in the end column, where the rounded cap
            # would clip a label sitting in the bin's outer corner.
            x = (box.left() + c * cw + 3) if c == 4 else (box.left() + (c + 1) * cw - w - 3)
            tag = QRectF(x, box.top() + (r + 1) * ch - 17, w, 14)
            p.setPen(QPen(QColor(self.t.border)))
            p.setBrush(card)
            p.drawRoundedRect(tag, 4, 4)
            p.setPen(QColor(self.t.foreground))
            p.drawText(tag, Qt.AlignCenter, text)
        p.restore()
        self._outline(p, box)

    def _draw_cutoff(self, p: QPainter) -> None:
        """The REAL track-length distribution, the limit as a line you drag, and
        what it removes shaded. A log axis, because the body of the distribution
        sits at 5-15 localisations while the suspicious tail runs to 200."""
        r = self._cutoff_plot()
        axis = QPen(QColor(self.t.border))
        axis.setWidthF(1.0)
        p.setPen(axis)
        p.drawLine(QPointF(r.left(), r.bottom()), QPointF(r.right(), r.bottom()))
        p.setFont(ty.ui("xs", px=text_px(0.66)))
        p.setPen(QColor(self.t.muted_foreground))
        for tick in (5, 10, 20, 50, 100, 200):
            x = self._x_for(tick)
            p.drawText(QRectF(x - 20, r.bottom() + 4, 40, 14), Qt.AlignHCenter | Qt.AlignTop, str(tick))
        p.drawText(QRectF(r.left(), r.bottom() + 18, r.width(), 14), Qt.AlignHCenter | Qt.AlignTop,
                   "localisations per track (log scale)")
        census = self._state.get("census") or []
        if not census:
            p.drawText(r, Qt.AlignCenter, "Run the map once to see the track lengths.")
            return
        hist = length_histogram(census)
        top = max(hist) or 1
        limit = self._state.get("max_locs")
        on = limit is not None
        if on:
            x_cut = self._x_for(limit + 0.5)
            shade = QColor(_CUT_INK)
            shade.setAlphaF(0.10)
            p.fillRect(QRectF(x_cut, r.top(), r.right() - x_cut, r.height()), shade)
        p.setPen(Qt.NoPen)
        for i, count in enumerate(hist):
            if count == 0:
                continue
            n = i + 5
            x0 = self._x_for(max(4.5, n - 0.5))
            x1 = self._x_for(min(MAX_LOCS_MAX + 0.5, n + 0.5))
            h = max(2.0, count / top * r.height())
            p.setBrush(QColor(_CUT_INK if (on and n > limit) else _SHORT_INK))
            p.drawRect(QRectF(x0 + 0.3, r.bottom() - h, max(1.0, x1 - x0 - 0.6), h))
        if on:
            line = QPen(QColor(_CUT_INK))
            line.setWidthF(2.0)
            p.setPen(line)
            p.drawLine(QPointF(x_cut, r.top()), QPointF(x_cut, r.bottom()))
            p.setBrush(QColor(self.t.card))
            p.drawEllipse(QPointF(x_cut, r.top()), 5.0, 5.0)
            p.setPen(QColor(_CUT_INK))
            p.setFont(ty.ui("xs", "semibold", px=text_px(0.7)))
            anchor_left = x_cut + 90 < r.right()
            label_rect = (QRectF(x_cut + 8, r.top() + 4, 90, 16) if anchor_left
                          else QRectF(x_cut - 98, r.top() + 4, 90, 16))
            p.drawText(label_rect, (Qt.AlignLeft if anchor_left else Qt.AlignRight) | Qt.AlignVCenter,
                       f"cutoff: {limit}")

    def _draw_masking(self, p: QPainter) -> None:
        """Bin masking: one invented cell, thresholded by BOTH rules, plus -- with n_eff on --
        a low and a high n_eff bin. MONOCHROME cell on purpose: it is the one diagram shaped
        like the product's own output, so it must not read as a jet map of your data.

        A bin below the count is plain blank; a bin that passes the count but falls below
        n_eff is HATCHED -- the caption's two reasons, apart, with the count taking a bin
        that fails both (as `finish_spatial_panels` does). Every number in the picture is
        computed from its invented contributors (`masking_field`, `neff_pair`)."""
        state = self._state
        n_min = int(state.get("n_min", 0))
        cut = state.get("neff_min")
        unit = state.get("neff_unit", "tracks")
        # THE SAME GRID `Bin size` IS SET TO: the settings act on one picture.
        length = state.get("cell_length_um", DEPOSIT_LENGTH_UM)
        width = state.get("cell_width_um", DEPOSIT_WIDTH_UM)
        bin_nm = max(1.0, float(state.get("bin_nm", BIN_NM_DEFAULT)))
        nx = even_bins(length * 1000.0 / bin_nm)
        ny = even_bins(width * 1000.0 / bin_nm)
        field = masking_field(nx, ny)
        box, pair_area = self._masking_layout()
        field_max = max(1, max(b["count"] for row in field for b in row))
        card = QColor(self.t.card)
        hatch = QBrush(QColor(self.t.muted_foreground), Qt.BDiagPattern)
        hatch_ground = _mix(QColor(self.t.border), card, 0.5)
        p.save()
        p.setClipPath(_capsule_path(box))
        cw, ch = box.width() / nx, box.height() / ny
        p.setPen(Qt.NoPen)
        for iy in range(ny):
            for ix in range(nx):
                b = field[iy][ix]
                cell = QRectF(box.left() + ix * cw, box.top() + iy * ch, cw + 0.4, ch + 0.4)
                if is_blank(b["count"], n_min):
                    p.setBrush(QColor(self.t.border))
                    p.drawRect(cell)
                elif cut is not None and b["neff_" + unit] < cut:
                    # its own density, faded, under the hatch: there ARE jumps here,
                    # just too few contributors -- the stuck spot reads as dense-but-one
                    ink = QColor(self.t.foreground)
                    ink.setAlphaF(0.04 + 0.36 * (b["count"] / field_max))
                    p.setBrush(hatch_ground)
                    p.drawRect(cell)
                    p.setBrush(ink)
                    p.drawRect(cell)
                    p.setBrush(hatch)
                    p.drawRect(cell)
                else:
                    ink = QColor(self.t.foreground)
                    ink.setAlphaF(0.08 + 0.72 * (b["count"] / field_max))
                    p.setBrush(ink)
                    p.drawRect(cell)
        p.restore()
        self._outline(p, box)
        if pair_area is not None:
            self._draw_neff_pair(p, pair_area, unit)

    def _masking_layout(self) -> tuple[QRectF, QRectF | None]:
        """Where the Bin masking cell and the n_eff pair go. The cell is ALWAYS the capsule
        every other page draws (`_capsule_box`) -- never squeezed to make room (ruled). With
        n_eff on, the pair sits centred BELOW it, in the height this page has over the
        others (`MASKING_EXTRA`)."""
        box = self._capsule_box()
        if self._state.get("neff_min") is None:
            return box, None
        top = box.bottom() + _PAIR_GAP
        pair = QRectF(self.PAD, top, self.width() - 2 * self.PAD,
                      self.height() - self.PAD - top)
        return box, pair

    def _draw_neff_pair(self, p: QPainter, area: QRectF, unit: str) -> None:
        """Two SQUARE bins -- they stand for map bins -- drawn like the n_eff slide: lines
        are tracks, dots are jumps, and a colour is the contributor n_eff counts (a track
        under Tracks, a cell under Cells). Same count in both; under each, its name and what
        it means (`PAIR_TEXT`). FIXED, never tied to the cutoff (ruled): Low is drawn blanked,
        High kept -- a "High n_eff" bin turning blanked as the slider moves would confuse,
        and the slider already acts on the hatching in the cell above."""
        pair = neff_pair(unit)
        muted = QColor(self.t.muted_foreground)
        p.setPen(muted)
        p.setFont(ty.ui("sm"))
        p.drawText(QRectF(area.left(), area.top(), area.width(), _PAIR_KEY_H - 6),
                   Qt.AlignHCenter | Qt.AlignVCenter | Qt.TextWordWrap, PAIR_KEY[unit])
        gap = 64.0
        side = max(10.0, min(_PAIR_SIDE, (area.width() - gap) / 2.0,
                             area.height() - _PAIR_KEY_H - _PAIR_LABEL_H))
        # each bin's words stay inside its own column: the square plus half the gap each side
        # less a margin, so the two descriptions can never meet in the middle
        half = (gap - 16.0) / 2.0
        top = area.top() + _PAIR_KEY_H
        left = area.center().x() - (2 * side + gap) / 2.0
        dot, line = side / 40.0, max(1.3, side / 75.0)
        what = "localisations" if self._state.get("method") == "appd" else "jumps"
        for k, name in enumerate(("low", "high")):
            b = pair[name]
            inks = [_contributor_ink(i) for i in pair_colour_index(b, unit)]
            sq = QRectF(left + k * (side + gap), top, side, side)
            blanked = name == "low"
            frame = QPainterPath()
            frame.addRect(sq)

            def at(pt, sq=sq):
                return QPointF(sq.left() + pt[0] * sq.width(), sq.top() + pt[1] * sq.height())

            for clipped in (False, True):
                p.save()
                if clipped:
                    p.setClipPath(frame)
                for tr, colour in zip(b["tracks"], inks):
                    ink = QColor(colour)
                    if not clipped:
                        ink.setAlphaF(0.28)
                    pen = QPen(ink)
                    pen.setWidthF(line)
                    p.setPen(pen)
                    pts = [at(pt) for pt in tr["points"]]
                    for a, c in zip(pts, pts[1:]):
                        p.drawLine(a, c)
                    if clipped:
                        p.setPen(Qt.NoPen)
                        p.setBrush(QColor(colour))
                        for x, y in tr["points"]:
                            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                                p.drawEllipse(at((x, y)), dot, dot)
                p.restore()
            outline = QPen(QColor(self.t.foreground if blanked else self.t.border))
            outline.setWidthF(1.4)
            if blanked:
                outline.setStyle(Qt.DashLine)
            p.setPen(outline)
            p.setBrush(Qt.NoBrush)
            p.drawPath(frame)
            text = PAIR_TEXT[name]
            p.setFont(ty.ui("sm", "semibold"))
            p.setPen(muted if blanked else QColor(self.t.foreground))
            p.drawText(QRectF(sq.left() - half, sq.bottom() + 4, sq.width() + 2 * half, 20),
                       Qt.AlignCenter, text["title"])
            p.setFont(ty.ui("sm"))
            p.setPen(muted)
            p.drawText(QRectF(sq.left() - half, sq.bottom() + 24, sq.width() + 2 * half, 20),
                       Qt.AlignCenter, f"{b['count']} {what}")
            p.drawText(QRectF(sq.left() - half, sq.bottom() + 44, sq.width() + 2 * half, 42),
                       Qt.AlignHCenter | Qt.AlignTop | Qt.TextWordWrap, text[unit])


def _contributor_ink(index: int) -> str:
    """The colour of contributor `index`: the listed muted inks first, then hues stepped by
    the golden angle at the same muted saturation and lightness -- distinct for ANY number of
    contributors, never wrapping round onto a colour already used."""
    if index < len(_CONTRIBUTOR_INKS):
        return _CONTRIBUTOR_INKS[index]
    hue = (0.61 + (index - len(_CONTRIBUTOR_INKS)) * 0.381966) % 1.0
    return QColor.fromHslF(hue, 0.45, 0.58).name()


def _group_caption(t, text: str):
    """A small heading inside the Run dialog's scrolling body.

    `toolbar._caption`'s own size and weight, restated rather than imported:
    that helper belongs to the tool rail's panels, and a dialog reaching
    into another module's private label style couples the two for the sake
    of two arguments.
    """
    return label(text, ty.ui("xs", "medium", px=text_px(0.75)),
                 t.muted_foreground)


class SpatialMapDialog(OverlayDialog):
    """The map's own Run configuration: one setting at a time, each shown as
    a picture that redraws while you change it.

    **Why it is not six controls stacked with their caveats under them,
    which is what it was.** That version ran to 298 words of prose against a
    house median of fourteen (`compute.CONTROL_HELP`), and the words were
    doing the wrong job: they described the mechanism instead of the choice.
    An information button was tried on paper and rejected for a concrete
    reason -- a popover cannot show a LIVE illustration, so tuning a slider
    becomes drag, click, look, close, drag again. The picture has to be on
    the page for the loop to close.

    **The nav keeps every value on screen.** Plain tabs would hide five
    settings behind the one open and make "what did I set?" a clicking
    exercise; the left column answers it by being read. Selection is the dot
    and the name's ink -- deliberately quiet, since a column of six chips or
    filled rows shouts louder than the picture it sits beside.

    **Show tips collapses the prose and the diagrams**, leaving the six
    controls alone, for somebody who already knows what they do.

    Everything here changes the MEASUREMENT, which is why it sits behind Run
    rather than beside the colormap in the Visibility tool -- and why every
    one of these settings is printed in the finished map's caption. Built on
    `OverlayDialog` and opened from `window.py`, the same call-site
    convention as `ComputeDialog`.
    """

    #: The panel is pinned so switching setting does not resize the card --
    #: a dialog that grows and shrinks under the cursor as you read down the
    #: nav is unusable, and the diagrams are deliberately different heights.
    #: 314 plus `_JUMP_AXIS_BAND` (the illustration grew by that band on every
    #: page to fit the "Jump drawn at" time axis without shrinking its cell),
    #: plus 99 for the track-cutoff page, whose three control rows (type, limit,
    #: short/long split) are the tallest content any page holds. MEASURED as the
    #: smallest value at which every page of every method is one card height
    #: (645 px): at the previous +8 the SQD cutoff page came out 8 px taller than
    #: its other pages and the appD one 39 px taller. (That +8 was itself measured:
    #: at +0 the toggle pages, 48 px control row, came out 4 px taller than the
    #: slider pages, 44 px.) A side effect worth keeping: SQD and appD now open
    #: at the same height, so switching the method no longer resizes the dialog.
    #: Plus 209 for "Bin masking", the tallest page: two control rows and a taller picture --
    #: the full-size cell, never squeezed (ruled), with the n_eff pair under it at 140 px
    #: squares and three lines of words under each (`_SettingArt.MASKING_EXTRA`). MEASURED in
    #: an unclipping window: +209 is the smallest value at which every page of every method
    #: is one card height (867 px); at +208 the other pages came out 1 px shorter. **MEASURE
    #: IN A WINDOW TALLER THAN THE CARD**: an 800 px window caps the card at 800 and every
    #: page then "agrees". +236 since the Bin masking page gained its "What n_eff is, and what it
    #: costs" line (27 px; measured the same way: 235 leaves one page 1 px short).
    PANEL_H = 314 + int(_JUMP_AXIS_BAND) + 99 + 236

    #: The card's width. Wide enough for Bin masking's two columns side by side under every
    #: method -- the widest page, with "Minimum localisations per bin" under apparent D.
    #: MEASURED: under apparent D that page needs 575 px beside 266 px of nav and margins,
    #: i.e. 841; since the settings SCROLL (22 Oct 2026, `body_scroll`) their column also keeps
    #: its scroll bar's 12 px, so 853, and rem(54) = 864 is the whole-rem step above it (was
    #: rem(53) = 848, before that rem(45) = 720). Any less and the card grows to fit on its
    #: own, by method -- so switching the method resized the dialog.
    #: `tests/test_spatial_map_widgets.py` asserts the width holds every method.
    CARD_MIN_W = rem(54)

    run = Signal()
    cancelled = Signal()
    #: The Split page wants a condition's histogram (`spatial_map_view.split_preview`): ``{"method", "condition",
    #: "n_pop", "lag_frames", "weighting"}``. The window computes it off the GUI thread and hands it back with
    #: `set_preview`; nothing here computes.
    preview_wanted = Signal(dict)

    def __init__(self, t, *, content=None, parent=None):
        super().__init__(t, content, parent=parent)
        self.t = t
        self._card.setMinimumWidth(self.CARD_MIN_W)
        #: Filled by `set_context` when the dialog opens on a condition.
        #: `None` until then, and `None` is also what "could not be
        #: measured" looks like -- both mean "do not offer the measured
        #: frame".
        self._measured: dict | None = None
        self._pixel_size_nm: float | None = None
        #: `spatial_map_view.project_pixel`'s note when the conditions' pixel sizes differ.
        self._pixel_note: str | None = None
        self._condition = ""

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(8)
        head.addWidget(label("Map Configuration", ty.ui("base", "semibold"), t.foreground))
        head.addStretch(1)
        self._close_btn = CloseButton(t)
        self._close_btn.clicked.connect(self._on_cancel)
        head.addWidget(self._close_btn)

        self._subtitle = label(
            "<p style='margin:0'>The same two-population fit the populations come from, "
            "read per step. Every setting below is printed on the finished map.</p>"
            "<p style='margin:6px 0 0 0'><b>Note:</b> the diagrams are illustrations, except "
            "Split and the track cutoff, which draw your data.</p>",
            ty.ui("sm"), t.muted_foreground)
        self._subtitle.setTextFormat(Qt.RichText)
        self._subtitle.setWordWrap(True)

        # The title and its description are one header block (shadcn's DialogHeader,
        # gap-1.5 = 6 px); the card's 16 px spacing belongs between blocks, not inside one.
        header = QVBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)
        header.addLayout(head)
        header.addWidget(self._subtitle)
        self.card_lay.addLayout(header)

        # -- the controls, built once and swapped in by `_show` -- #
        #
        # Capitalized here ONLY -- these are clickable tab labels, matching
        # this app's other `Segmented` controls ("Cards"/"Table",
        # "Distribution"/"Bubble Plot"). `_STEP_LABELS`/`_ASSIGNMENTS`
        # themselves stay lowercase: `_caption_for` reuses them verbatim
        # inside the rendered map's own figure caption (line ~790), where
        # lowercase is the correct scientific-figure convention.
        self._method = Segmented(
            t, [METHOD_LABELS[m] for m in METHODS], current=0)
        self._step = Segmented(t, [s.capitalize() for s in STEP_LABELS], current=0)
        self._assignment = Segmented(t, [s.capitalize() for s in ASSIGNMENTS],
                                     current=0)
        #: The HMM map's Show (`spatialangles`): Steps (where each state's steps are), Turning back (the data alone)
        #: or Beyond chance. On the Jump angles section's own map page from 22 Nov 2026; here, the one place, since
        #: 23 Nov 2026, when the Jump angles became a tab of the HMM section ("Spatial Map only").
        self._show_turning = Segmented(t, list(HMM_SHOW_LABELS), current=0)
        self._fold = Segmented(t, ["Off", "On"], current=0)
        self._frame = Segmented(t, ["Measured", "Deposit"], current=0)
        #: Which cells are drawn, by length (thirds of the cells the frame is measured
        #: from), and how each one's long axis is laid onto the outline.
        self._cells = Segmented(t, ["All", "Short", "Mid", "Long"], current=0)
        self._axis = Segmented(t, ["Normalised", "Pole-anchored"], current=0)
        #: The population count (SQD and apparent diffusion): one map each. ONE CONTROL PER METHOD, because the
        #: counts differ (`spatial_map_view.map_n_pops`: SQD 1-3 since 1 Nov 2026, apparent diffusion 2-3) and a
        #: `Segmented` cannot change how many pills it has; each opens on two and keeps its own choice. `_pops` is
        #: the current method's, and the Populations row shows it (`_on_method_changed`).
        self._pops_by_method = {
            method: Segmented(t, [str(n) for n in counts] + ["Best (BIC)"],
                              current=counts.index(MAP_N_POP_DEFAULT))
            for method, counts in MAP_N_POPS_BY_METHOD.items()}
        for method, seg in self._pops_by_method.items():
            seg.changed.connect(lambda i, m=method: self._on_every_count(m, i))
        self._pops = self._pops_by_method[METHODS[0]]
        #: EACH CONDITION ITS OWN COUNT (1 Nov 2026, `spatial_map_view.BEST`, `condition_config`): the Populations
        #: page's tray above is "Every condition" -- a click there sets every row -- and below it one tray per
        #: condition, clickable fields as on the SQD dialog's Fits page. ``{method: {condition: count | "best"}}``;
        #: a condition not in it follows Every condition.
        self._pop_choice: dict = {m: {} for m in MAP_N_POPS_BY_METHOD}
        self._pop_rows: dict = {}
        self._setting_pops = False
        #: THE SECTION'S D (4 Nov 2026, `map_fit_source`): per fitted method, whether its maps take the D of that
        #: section's last Run ("Use the same D as the SQD fit"); TICKED for a new map. The box is on the Populations
        #: page; ticked, the counts there (and SQD's time lag) are the section fit's and greyed.
        self._same_d: dict = {m: bool(SAME_D_DEFAULT) for m in SAME_D_METHODS}
        #: THE MAP'S OWN FIT PER SPOT (6 Nov 2026, `spatial_map_view.OWN_PRECISION_KEY`): per fitted method, whether
        #: its own fit models each jump's localisation error; the row shows while the box is unticked and the split is
        #: Fitted (ticked, the section's fit brings its own). Off for a new map.
        self._per_spot: dict = {m: False for m in SAME_D_METHODS}
        self._setting_precision = False
        #: WHICH PAGE of that section the D comes from (subpages, `smtracker.subpages`): per fitted method the page key
        #: ("sqd", or "sqd~2" for an added page), the pages on offer ``[(key, name)]`` (`set_fit_pages`, from the
        #: window) and the name each key was last known by -- so a page that is deleted is still NAMED.
        self._fit_page: dict = {m: m for m in SAME_D_METHODS}
        self._fit_pages: dict = {m: [(m, SAME_D_PAGE_NAMES[m])] for m in SAME_D_METHODS}
        self._fit_names: dict = {m: {m: SAME_D_PAGE_NAMES[m]} for m in SAME_D_METHODS}
        #: SPLIT (1 Nov 2026, `spatial_map_view.SPLITS`): Fitted, or ranges set BY HAND on the condition's own
        #: histogram, one map per range -- SQD's of jump length (nm), apparent diffusion's of each track's mean
        #: apparent D (µm²/s), each method its own list. The histograms arrive from the window (`preview_wanted`,
        #: `set_preview`), kept per request so going back to a condition draws at once.
        self._split_mode = Segmented(t, ["Fitted", "Manual"], current=SPLITS.index(SPLIT_DEFAULT))
        self._split_mode.changed.connect(self._on_split_changed)
        self._ranges_by_method = {m: [list(r) for r in MANUAL_RANGES_DEFAULT[m]] for m in RANGE_UNITS}
        self._previews: dict = {}
        #: ``[(condition key, its name), ...]`` the histogram can show (`set_context`)
        self._preview_conditions: list = []
        self._lag = Segmented(t, [str(n).replace("-", "\u2013") for n in MAP_LAGS], current=0)
        self._lag.changed.connect(self._on_control_changed)
        self._method.changed.connect(self._on_method_changed)
        for seg in (self._step, self._assignment, self._fold, self._frame,
                    self._cells, self._axis, self._show_turning):
            seg.changed.connect(self._on_control_changed)

        self._bin = Slider(t, BIN_NM_MIN, BIN_NM_MAX, BIN_NM_DEFAULT)
        # THE BIN'S UNIT (30 Oct 2026): one camera pixel of the data set, the standard, or
        # the nm on the slider. Moving the slider by hand chooses nm, as moving the SLA
        # radius by hand keeps it; picking 1 px puts the slider back on the pixel.
        self._bin_unit = Segmented(t, list(bin_unit_labels(None)),
                                   current=BIN_UNITS.index(BIN_UNIT_DEFAULT))
        self._bin_unit.changed.connect(self._on_bin_unit_changed)
        self._setting_bin = False
        self._n_min = Slider(t, 0, N_MIN_MAX, N_MIN_DEFAULT)
        self._bin.changed.connect(self._on_bin_moved)
        self._n_min.changed.connect(self._on_control_changed)
        # Weighting, and the track cutoff with the split it learns from.
        self._weighting = Segmented(t, ["By steps", "By track"], current=0)
        self._weighting.changed.connect(self._on_control_changed)
        self._cutoff = Slider(t, MAX_LOCS_MIN, MAX_LOCS_MAX, 40)
        self._cutoff.changed.connect(self._on_control_changed)
        self._cutoff_mode = Segmented(t, ["Off", "Tracks", "Cells + Tracks"], current=0)
        self._cutoff_mode.changed.connect(self._on_control_changed)
        self._split = Segmented(t, ["All tracks", "Kept tracks"], current=0)
        self._split.changed.connect(self._on_control_changed)
        # The n_eff guard (handover §4a.3): which contributor it counts, and the cutoff.
        self._neff_mode = Segmented(t, ["Off", "Tracks", "Cells"], current=0)
        self._neff_mode.changed.connect(self._on_control_changed)
        self._neff = Slider(t, int(NEFF_MIN_MIN), int(NEFF_MIN_MAX), 3)
        self._neff.changed.connect(self._on_control_changed)
        # Confinement (SLA): v2's defaults, the pixel (106 nm here; `set_context` puts the
        # condition's own pixel in until the radius is moved by hand) and 5 steps.
        self._radius = Slider(t, SLA_RADIUS_RANGE_NM[0], SLA_RADIUS_RANGE_NM[1], 106)
        self._radius_by_hand = False
        self._radius.changed.connect(self._on_radius_changed)
        self._dwell = Slider(t, SLA_DWELL_RANGE[0], SLA_DWELL_RANGE[1], 5)
        self._dwell.changed.connect(self._on_control_changed)
        # SDA: the value maps' adaptive kernel. ON by default (13 Oct 2026; a decision to
        # confirm): off draws every value bin at one width, as before.
        self._adaptive = Segmented(t, ["Off", "On"], current=1)
        self._adaptive.changed.connect(self._on_control_changed)
        #: The census the cutoff chart and its cost line read -- `set_context`.
        self._census: list = []
        self._controls = {
            "method": self._method,
            "populations": self._populations_controls(t),
            "split": self._split_controls(t),
            "lag": self._lag,
            "jump": self._step, "assignment": self._assignment,
            "show": self._show_turning,
            "fold": self._fold, "frame": self._frame,
            "cells": self._cells, "axis": self._axis,
            "bin": self._bin_controls(t),
            "weighting": self._weighting,
            "cutoff": self._cutoff_controls(t),
            "masking": self._masking_controls(t),
            "radius": self._slider_row(t, self._radius, "nm"),
            "dwell": self._slider_row(t, self._dwell, "steps"),
            "adaptive": self._adaptive,
        }

        #: The CURRENT method's row set -- `_KEYS_BY_METHOD["sqd"]` at
        #: construction, rebuilt by `_on_method_changed` thereafter. An
        #: instance attribute rather than the module-level `_KEYS` this
        #: dialog used to read directly, since which rows exist now depends
        #: on which method is picked.
        self._keys = self._rows_for("sqd")

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(rem(1.25))
        self.nav = _SettingNav(t, [setting_name(k) for k in self._keys])
        self.nav.changed.connect(lambda i: self._show(self._keys[i]))
        body.addWidget(self.nav, 0, Qt.AlignTop)

        panel = QWidget()
        panel.setStyleSheet("background:transparent;")
        panel.setMinimumHeight(self.PANEL_H)
        self._panel_lay = QVBoxLayout(panel)
        self._panel_lay.setContentsMargins(0, 0, 0, 0)
        self._panel_lay.setSpacing(rem(0.75))
        self._title = label("", ty.ui("sm", "semibold", px=text_px(0.9375)),
                            t.foreground)
        self._desc = label("", ty.ui("sm"), t.muted_foreground)
        self._desc.setWordWrap(True)
        #: The control sits DIRECTLY above the picture it changes -- the
        #: proximity is what makes the causation legible, and a gap here
        #: was the first thing that read wrong on paper.
        self._control_slot = QWidget()
        self._control_slot.setStyleSheet("background:transparent;")
        self._control_lay = QHBoxLayout(self._control_slot)
        self._control_lay.setContentsMargins(0, rem(0.25), 0, rem(0.25))
        self._control_lay.setSpacing(0)
        self._art = _SettingArt(t)
        self._art.cutoff_dragged.connect(self._on_cutoff_dragged)
        self._read = label("", ty.ui("xs", tabular=True), t.muted_foreground)
        self._read.setWordWrap(True)
        #: The info line for a row whose mechanism needs more than its picture: the
        #: words are a TOOLTIP, never printed (`spatial_map_view.setting_help`).
        self._help = _InfoLink(t)
        self._help.hide()
        self._warn = label("", ty.ui("xs", "medium"), _WARN_COLOR)
        self._warn.setWordWrap(True)
        for w in (self._title, self._desc, self._control_slot, self._art,
                  self._read, self._help, self._warn):
            self._panel_lay.addWidget(w)
        self._panel_lay.addStretch(1)
        body.addWidget(panel, 1)
        # THE SETTINGS SCROLL on a short window, the title and Run stay (found by `scripts/gui_sweep.py`, 22 Oct
        # 2026: the dialog asked for 877 px, so on a 768 px laptop screen Cancel and Run fell off its foot). The
        # Export dialog's column (`primitives.FitScroll`): a tall window shows every setting with no scroll bar.
        body_host = QWidget()
        body_host.setStyleSheet("background:transparent;")
        body_host.setLayout(body)
        self.body_scroll = FitScroll()
        self.body_scroll.setWidget(body_host)
        self.body_scroll.setWidgetResizable(True)
        self.body_scroll.setFrameShape(QScrollArea.NoFrame)
        self.body_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.body_scroll.setStyleSheet("QScrollArea { background:transparent; border:none; }")
        self.card_lay.addWidget(self.body_scroll, 1)

        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.setSpacing(8)
        #: `Checkbox`, not a new switch widget -- this app has no switch, and
        #: inventing one for a single row would put a second on/off language
        #: on screen beside the one every other panel already uses.
        self.tips = Checkbox(t, checked=True)
        self.tips.toggled.connect(self._set_tips)
        footer.addWidget(self.tips)
        footer.addWidget(label("Show tips", ty.ui("sm"), t.muted_foreground))
        footer.addStretch(1)
        cancel = Button(t, "Cancel", variant="outline")
        cancel.clicked.connect(self._on_cancel)
        self.run_btn = Button(t, "Run", variant="default")
        self.run_btn.clicked.connect(self._on_run)
        footer.addWidget(cancel)
        footer.addWidget(self.run_btn)
        self.card_lay.addLayout(footer)

        self._show(self._keys[0])

    # -- construction helpers ----------------------------------------------- #
    def _populations_controls(self, t) -> QWidget:
        """The Populations page: "Every condition" (the method's tray: its counts and Best (BIC)) and, once the
        conditions are known (`set_context`), one tray per condition. A click on Every condition sets every row;
        a row set apart leaves Every condition unlit."""
        box = QWidget()
        box.setStyleSheet("background:transparent;")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        # THE SECTION'S D (`map_fit_source`): the box, its label naming the method's section, and while ticked the
        # line that says the counts below come from it
        same = QWidget()
        same.setStyleSheet("background:transparent;")
        s = QHBoxLayout(same)
        s.setContentsMargins(0, 0, 0, 4)
        s.setSpacing(8)
        self.same_d_box = Checkbox(t, checked=self._same_d.get("sqd", SAME_D_DEFAULT))
        self.same_d_box.toggled.connect(self._on_same_d_toggled)
        s.addWidget(self.same_d_box, 0, Qt.AlignVCenter)
        self.same_d_label = label(same_d_text("sqd"), ty.ui("sm", "medium"), t.foreground)
        s.addWidget(self.same_d_label, 0, Qt.AlignVCenter)
        # WHICH PAGE of the section (subpages): a drop-down of "SQD", "SQD 2" ..., there only while the section has
        # more than one page to take D from; the section's own page by default
        self.fit_page_box = PageChoice(t)
        self.fit_page_box.activated.connect(self._on_fit_page_picked)
        self.fit_page_box.hide()
        s.addWidget(self.fit_page_box, 0, Qt.AlignVCenter)
        s.addStretch(1)
        lay.addWidget(same)
        self._same_d_row = same
        # the page the D was to come from is GONE: said, with a button that switches to the section's own page
        gone = QWidget()
        gone.setStyleSheet("background:transparent;")
        g = QHBoxLayout(gone)
        g.setContentsMargins(0, 0, 0, 4)
        g.setSpacing(8)
        self.fit_gone_note = fit_label("", ty.ui("xs"), _FLAG_INK)
        g.addWidget(self.fit_gone_note, 0)
        self.fit_gone_button = Button(t, "Use SQD", size="sm")
        self.fit_gone_button.clicked.connect(lambda: self.set_fit_page(self.method_key(), self.method_key()))
        g.addWidget(self.fit_gone_button, 0, Qt.AlignVCenter)
        g.addStretch(1)
        lay.addWidget(gone)
        self._fit_gone_row = gone
        gone.hide()
        # a FitLabel, never a plain wrapped label: that would make the page height-for-width (`_rebuild_pop_rows`)
        self.same_d_note = fit_label(same_d_note("sqd"), ty.ui("xs"), t.muted_foreground)
        lay.addWidget(self.same_d_note)
        # THE MAP'S OWN FIT PER SPOT (`spatial_map_view.OWN_PRECISION_KEY`): there while the map fits its own D
        own = QWidget()
        own.setStyleSheet("background:transparent;")
        o = QHBoxLayout(own)
        o.setContentsMargins(0, 0, 0, 0)
        o.setSpacing(12)
        precision_caption = label("Localisation error", ty.ui("sm", "medium"), t.foreground)
        precision_caption.setFixedWidth(rem(11))
        o.addWidget(precision_caption)
        self._precision = Segmented(t, list(OWN_PRECISION_LABELS), current=0)
        self._precision.changed.connect(self._on_precision_changed)
        o.addWidget(self._precision, 0, Qt.AlignVCenter)
        o.addStretch(1)
        lay.addWidget(own)
        self._precision_row = own
        self.precision_note = fit_label(OWN_PRECISION_NOTE, ty.ui("xs"), t.muted_foreground)
        lay.addWidget(self.precision_note)
        self.precision_refusal = fit_label(OWN_PRECISION_LAG_REFUSAL, ty.ui("xs"), _FLAG_INK)
        lay.addWidget(self.precision_refusal)
        every = QWidget()
        every.setStyleSheet("background:transparent;")
        h = QHBoxLayout(every)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(12)
        self._every_caption = label("Every condition", ty.ui("sm", "medium"), t.foreground)
        self._every_caption.setFixedWidth(rem(11))
        h.addWidget(self._every_caption)
        self._every_slot = QHBoxLayout()
        self._every_slot.setContentsMargins(0, 0, 0, 0)
        h.addLayout(self._every_slot)
        h.addStretch(1)
        lay.addWidget(every)
        self._pop_rows_box = QWidget()
        self._pop_rows_box.setStyleSheet("background:transparent;")
        self._pop_rows_lay = QVBoxLayout(self._pop_rows_box)
        self._pop_rows_lay.setContentsMargins(0, 4, 0, 0)
        self._pop_rows_lay.setSpacing(6)
        lay.addWidget(self._pop_rows_box)
        self._place_every_tray()
        box._is_slider_row = True
        return box

    def same_d(self, method: str | None = None) -> bool:
        """Whether ``method``'s maps (the current method's when None) take the section's D (`map_fit_source`)."""
        return bool(self._same_d.get(method or self.method_key(), False))

    def set_same_d(self, method: str, on: bool) -> None:
        """Tick or untick ``method``'s box ("Use the same D as the ... fit"); each fitted method keeps its own."""
        if method not in self._same_d:
            raise ValueError(f"only {SAME_D_METHODS} maps take a section's D, not {method!r}")
        self._same_d[method] = bool(on)
        self._sync_same_d()
        self._refresh()

    # -- which page of the section the D comes from (subpages) ---------------------------------------------------- #
    def fit_page(self, method: str | None = None) -> str:
        """The page key ``method``'s maps take D from (``"sqd"``, ``"sqd~2"``); the section's own by default."""
        method = method or self.method_key()
        return self._fit_page.get(method, method)

    def fit_page_gone(self, method: str | None = None) -> str | None:
        """The name of the page ``method``'s maps were to take D from if it no longer exists, else None."""
        method = method or self.method_key()
        key = self.fit_page(method)
        if key in {k for k, _n in self._fit_pages.get(method, ())}:
            return None
        return self._fit_names.get(method, {}).get(key, key)

    def set_fit_pages(self, method: str, pages) -> None:
        """The pages the section has now, ``[(key, name)]`` in the sidebar's order (the window, on every change). A
        chosen page that is not among them is kept and flagged (`fit_page_gone`), never swapped in silence."""
        if method not in self._fit_page:
            return
        self._fit_pages[method] = [(str(k), str(n)) for k, n in pages]
        self._fit_names[method].update({str(k): str(n) for k, n in pages})
        self._sync_fit_page()

    def set_fit_page(self, method: str, key: str) -> None:
        """Take ``method``'s D from the page ``key``."""
        if method not in self._fit_page:
            raise ValueError(f"only {SAME_D_METHODS} maps take a section's D, not {method!r}")
        self._fit_page[method] = str(key)
        self._sync_fit_page()
        self._refresh()

    def preset_fit(self, method: str, key: str) -> None:
        """"Map this fit" (`map_fit_source.MAP_THIS_FIT`): ``method`` chosen, its box ticked and the page ``key`` (its
        fit) chosen for the D; every other setting as the dialog last had it."""
        if method not in self._fit_page:
            raise ValueError(f"only {SAME_D_METHODS} maps take a section's D, not {method!r}")
        if self.method_key() != method:
            self._method.select(METHODS.index(method))
            self._on_method_changed()
        self.set_same_d(method, True)
        self.set_fit_page(method, key)

    def _on_fit_page_picked(self, index: int) -> None:
        method = self.method_key()
        key = self.fit_page_box.itemData(index)
        if method in self._fit_page and key:
            self.set_fit_page(method, key)

    def _sync_fit_page(self) -> None:
        """The drop-down: the pages on offer, the chosen one selected (a deleted one shown by its old name), and the
        gone row with its button while the chosen page is missing."""
        method = self.method_key()
        fitted = method in self._fit_page
        pages = self._fit_pages.get(method, []) if fitted else []
        gone = self.fit_page_gone(method) if fitted else None
        ticked = fitted and self._same_d.get(method, False)
        box = self.fit_page_box
        box.blockSignals(True)
        box.clear()
        for key, name in pages:
            box.addItem(name, key)
        if gone is not None:
            box.addItem(f"{gone} (deleted)", self.fit_page(method))
        index = box.findData(self.fit_page(method)) if fitted else -1
        box.setCurrentIndex(max(0, index))
        box.blockSignals(False)
        box.setVisible(ticked and (len(pages) > 1 or gone is not None))
        own = SAME_D_PAGE_NAMES.get(method, method)
        self._fit_gone_row.setVisible(ticked and gone is not None)
        if gone is not None:
            self.fit_gone_note.setText(f"\u201c{gone}\u201d was deleted.")
            self.fit_gone_button.set_text(f"Use {own}")

    def per_spot(self, method: str | None = None) -> bool:
        """Whether ``method``'s own fit (the current method's when None) models each jump's localisation error."""
        return bool(self._per_spot.get(method or self.method_key(), False))

    def set_per_spot(self, method: str, on: bool) -> None:
        """Localisation error Per spot (``on``) or Not modelled for ``method``'s own fit; each fitted method its own."""
        if method not in self._per_spot:
            raise ValueError(f"only {SAME_D_METHODS} maps fit populations, not {method!r}")
        self._per_spot[method] = bool(on)
        self._refresh()

    def _on_precision_changed(self, index: int) -> None:
        method = self.method_key()
        if self._setting_precision or method not in self._per_spot:
            return
        self._per_spot[method] = index == OWN_PRECISION_LABELS.index("Per spot")
        self._refresh()

    def _per_spot_applies(self) -> bool:
        """Whether the current method's map fits its own D (a fitted method, the box unticked, the split Fitted): the
        only map the Localisation error choice reaches."""
        method = self.method_key()
        manual = method in RANGE_UNITS and self._split_mode.current() == SPLITS.index("manual")
        return method in self._per_spot and not self._same_d.get(method, False) and not manual

    def _on_same_d_toggled(self, on: bool) -> None:
        method = self.method_key()
        if method in self._same_d:
            self._same_d[method] = bool(on)
        self._sync_same_d()
        self._refresh()

    def _sync_same_d(self) -> None:
        """The box, its label and its note for the current method; ticked on a fitted split, the counts and SQD's
        time lag are greyed (they are the section fit's)."""
        method = self.method_key()
        fitted = method in self._same_d
        on = fitted and self._same_d[method]
        self._same_d_row.setVisible(fitted)
        self._sync_fit_page()
        if fitted:
            self.same_d_box.set_checked(on, animate=False)
            self.same_d_label.setText(same_d_text(method))
            self.same_d_note.setText(same_d_note(method))
        manual = method in RANGE_UNITS and self._split_mode.current() == SPLITS.index("manual")
        held = on and not manual
        self.same_d_note.setVisible(held)
        self._every_caption.setEnabled(not held)
        for seg in (self._pops, *self._pop_rows.values()):
            seg.setEnabled(not held)
        self._lag.setEnabled(not (held and method == "sqd"))
        # the map's own fit Per spot: the row while the map fits its own, its refusal beside lags 1-4 together
        own = self._per_spot_applies()
        self._precision_row.setVisible(own)
        self.precision_note.setVisible(own)
        if fitted:
            self._setting_precision = True
            try:
                self._precision.select(OWN_PRECISION_LABELS.index("Per spot" if self.per_spot(method)
                                                                  else "Not modelled"))
            finally:
                self._setting_precision = False
        self.precision_refusal.setVisible(own and bool(self._precision_refused()))

    def _precision_refused(self) -> str:
        """Why the map's own fit cannot be Per spot as set, or "": SQD's lags 1-4 together."""
        method = self.method_key()
        if (self._per_spot_applies() and self.per_spot(method) and method == "sqd"
                and str(MAP_LAGS[self._lag.current()]) == "1-4"):
            return OWN_PRECISION_LAG_REFUSAL
        return ""

    def _count_method(self) -> str:
        """The method whose counts the Populations page shows: the current one where it fits populations."""
        method = self.method_key()
        return method if method in MAP_N_POPS_BY_METHOD else "sqd"

    def _choices(self, method: str) -> tuple:
        """The counts and Best, in the trays' order."""
        return (*map_n_pops(method), BEST)

    def _place_every_tray(self) -> None:
        """The current method's Every-condition tray in its slot (each method keeps its own), then the rows."""
        while self._every_slot.count():
            item = self._every_slot.takeAt(0)
            if item.widget() is not None:
                item.widget().hide()
                item.widget().setParent(None)
        self._pops = self._pops_by_method[self._count_method()]
        self._every_slot.addWidget(self._pops)
        self._pops.show()
        self._rebuild_pop_rows()

    def _rebuild_pop_rows(self) -> None:
        """One tray per condition (`set_context`'s list) at its own count, or at Every condition's."""
        while self._pop_rows_lay.count():
            item = self._pop_rows_lay.takeAt(0)
            if item.widget() is not None:
                discard(item.widget())
        self._pop_rows = {}
        method = self._count_method()
        choices = self._choices(method)
        for key, name in self._preview_conditions:
            row = QWidget()
            row.setStyleSheet("background:transparent;")
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(12)
            # A FitLabel (its own height, from its text): a plain wrapped label made the page height-for-width, and
            # rows rebuilt while the page was detached (a change of method) came back at a height Qt had cached with
            # them hidden, drawn over each other (found on main, 4 Nov 2026, in this branch's screenshots)
            caption = fit_label(name, ty.ui("sm"), self.t.foreground)
            caption.setFixedWidth(rem(11))
            h.addWidget(caption)
            seg = Segmented(self.t, [str(n) for n in map_n_pops(method)] + ["Best (BIC)"],
                            current=choices.index(self._condition_choice(method, key)))
            seg.changed.connect(lambda i, k=key, m=method: self._on_row_count(m, k, i))
            h.addWidget(seg)
            h.addStretch(1)
            self._pop_rows_lay.addWidget(row)
            self._pop_rows[key] = seg
        self._pop_rows_box.setVisible(bool(self._pop_rows))
        self._sync_every_tray()

    def _every_choice(self, method: str):
        """Every condition's own choice: its lit pill, or the default before any."""
        seg = self._pops_by_method[method]
        index = seg.current()
        choices = self._choices(method)
        return choices[index] if 0 <= index < len(choices) else getattr(self, "_every_last", {}).get(
            method, MAP_N_POP_DEFAULT)

    def _condition_choice(self, method: str, condition: str):
        return self._pop_choice[method].get(condition, self._every_choice(method))

    def _sync_every_tray(self) -> None:
        """Every condition lit at the rows' one choice, unlit while they differ."""
        method = self._count_method()
        if not self._pop_rows:
            return
        picks = {self._condition_choice(method, k) for k in self._pop_rows}
        self._setting_pops = True
        try:
            choices = self._choices(method)
            self._pops_by_method[method].select(choices.index(picks.pop()) if len(picks) == 1 else -1)
        finally:
            self._setting_pops = False

    def _on_every_count(self, method: str, index: int) -> None:
        """A click on Every condition: every row at that choice."""
        if self._setting_pops or index < 0:
            return
        choice = self._choices(method)[index]
        self._every_last = {**getattr(self, "_every_last", {}), method: choice}
        self._pop_choice[method] = {}
        self._setting_pops = True
        try:
            for seg in self._pop_rows.values():
                seg.select(index)
        finally:
            self._setting_pops = False
        self._refresh()

    def _on_row_count(self, method: str, condition: str, index: int) -> None:
        """A click on one condition's tray: that condition's own count."""
        if self._setting_pops or index < 0:
            return
        self._pop_choice[method][condition] = self._choices(method)[index]
        self._sync_every_tray()
        self._refresh()

    def _n_pop_by_condition(self) -> dict | None:
        """Each condition's choice for the run (`spatial_map_view.condition_config`), or None before the
        conditions are known (Every condition alone then decides)."""
        method = self.method_key()
        if method not in MAP_N_POPS_BY_METHOD or not self._pop_rows:
            return None
        return {k: self._condition_choice(method, k) for k in self._pop_rows}

    def _split_controls(self, t) -> QWidget:
        """The Split page's controls: Fitted or Manual, the condition whose histogram is drawn, "From the fit", and
        -- Manual -- the ranges, one row each (low, high, remove), with "Add range" up to `MANUAL_RANGES_MAX`."""
        box = QWidget()
        box.setStyleSheet("background:transparent;")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        top = QWidget()
        top.setStyleSheet("background:transparent;")
        h = QHBoxLayout(top)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(10)
        h.addWidget(self._split_mode)
        h.addStretch(1)
        h.addWidget(label("Histogram of", ty.ui("sm"), t.muted_foreground))
        self._preview_pick = QComboBox()
        self._preview_pick.setFont(ty.ui("sm"))
        self._preview_pick.setStyleSheet(_combo_css(t))
        self._preview_pick.setMinimumWidth(rem(10))
        self._preview_pick.currentIndexChanged.connect(lambda _i: self._refresh())
        h.addWidget(self._preview_pick)
        self._from_fit = Button(t, "From the fit", size="sm")
        self._from_fit.clicked.connect(self._on_ranges_from_fit)
        h.addWidget(self._from_fit)
        lay.addWidget(top)
        self._range_box = QWidget()
        self._range_box.setStyleSheet("background:transparent;")
        self._range_lay = QVBoxLayout(self._range_box)
        self._range_lay.setContentsMargins(0, 0, 0, 0)
        self._range_lay.setSpacing(6)
        lay.addWidget(self._range_box)
        self._add_range = Button(t, "Add range", size="sm", glyph="plus")
        self._add_range.clicked.connect(self._on_add_range)
        add_row = QWidget()
        add_row.setStyleSheet("background:transparent;")
        a = QHBoxLayout(add_row)
        a.setContentsMargins(0, 0, 0, 0)
        a.addWidget(self._add_range)
        a.addStretch(1)
        lay.addWidget(add_row)
        self._add_row = add_row
        self._rebuild_range_rows()
        box._is_slider_row = True
        return box

    def _split_method(self) -> str:
        """The method whose ranges the Split page shows: the current one where it takes a split, else SQD's."""
        method = self.method_key()
        return method if method in RANGE_UNITS else "sqd"

    def _rebuild_range_rows(self) -> None:
        """One row per range of the current method: its low and high ends (typed or stepped, in the method's unit)
        and Remove; shown under Manual only."""
        while self._range_lay.count():
            item = self._range_lay.takeAt(0)
            if item.widget() is not None:
                discard(item.widget())
        method = self._split_method()
        ranges = self._ranges_by_method[method]
        unit = RANGE_UNITS[method]
        nm = method == "sqd"
        for index, (lo, hi) in enumerate(ranges):
            row = QWidget()
            row.setStyleSheet("background:transparent;")
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(8)
            fields = []
            for end, value in enumerate((lo, hi)):
                field = FloatField(self.t, 0.0, 100000.0 if nm else 1000.0, float(value),
                                   step=10.0 if nm else 0.05, decimals=0 if nm else 3,
                                   suffix=f" {unit}", width=150 if nm else 170)
                field.changed.connect(lambda v, i=index, e=end: self._on_range_typed(i, e, v))
                fields.append(field)
            h.addWidget(label(f"Map {index + 1}", ty.ui("sm"), self.t.muted_foreground))
            h.addWidget(fields[0])
            h.addWidget(label("to", ty.ui("sm"), self.t.muted_foreground))
            h.addWidget(fields[1])
            remove = Button(self.t, "Remove", variant="ghost", size="sm")
            remove.clicked.connect(lambda i=index: self._on_remove_range(i))
            remove.setEnabled(len(ranges) > 1)
            h.addWidget(remove)
            h.addStretch(1)
            self._range_lay.addWidget(row)
        self._add_range.setEnabled(len(ranges) < MANUAL_RANGES_MAX)
        manual = self._split_mode.current() == SPLITS.index("manual")
        self._range_box.setVisible(manual)
        self._add_row.setVisible(manual)
        self._from_fit.setVisible(manual)

    def _on_split_changed(self, *_a) -> None:
        self._rebuild_range_rows()
        self._refresh()

    def _on_range_typed(self, index: int, end: int, value: float) -> None:
        ranges = self._ranges_by_method[self._split_method()]
        if 0 <= index < len(ranges):
            ranges[index][end] = float(value)
            self._refresh()

    def _on_remove_range(self, index: int) -> None:
        ranges = self._ranges_by_method[self._split_method()]
        if len(ranges) > 1 and 0 <= index < len(ranges):
            del ranges[index]
            self._rebuild_range_rows()
            self._refresh()

    def _on_add_range(self) -> None:
        """A new range after the last: from its high end, as wide again (SQD) or fivefold (apparent D, a log axis)."""
        method = self._split_method()
        ranges = self._ranges_by_method[method]
        if len(ranges) >= MANUAL_RANGES_MAX:
            return
        start = max(hi for _lo, hi in ranges) if ranges else 0.0
        width = (max(hi - lo for lo, hi in ranges) if ranges else 150.0) if method == "sqd" else None
        ranges.append([start, start + width] if method == "sqd" else [start, max(start, 0.01) * 5.0])
        self._rebuild_range_rows()
        self._refresh()

    def _on_ranges_from_fit(self) -> None:
        """The ranges where the shown condition's fitted maps would cut (`spatial_map_view.ranges_from_fit`)."""
        preview = self._previews.get(self._preview_key())
        if not isinstance(preview, dict) or "quantiles" not in preview:
            return
        self._ranges_by_method[self._split_method()] = ranges_from_fit(preview)
        self._rebuild_range_rows()
        self._refresh()

    def _preview_condition(self) -> str:
        index = self._preview_pick.currentIndex() if hasattr(self, "_preview_pick") else -1
        return self._preview_conditions[index][0] if 0 <= index < len(self._preview_conditions) else ""

    def _preview_key(self) -> tuple:
        """What a histogram depends on: the method, the condition, the fitted populations drawn on it, the lag and
        the fit's weighting."""
        method = self._split_method()
        return (method, self._preview_condition(),
                self._condition_choice(method, self._preview_condition()) if method in self._pop_choice
                else self._n_pop(), str(MAP_LAGS[self._lag.current()]),
                WEIGHTINGS[self._weighting.current()], self.same_d(method),
                self.fit_page(method) if method in self._fit_page else "")

    def _want_preview(self) -> None:
        """Ask the window for the shown condition's histogram, once per request (`preview_wanted`). With the box
        ticked (``same_D``) the window draws it at the section fit's D where it has one."""
        key = self._preview_key()
        if not key[1] or key in self._previews:
            return
        self._previews[key] = {"pending": True}
        method, condition, n_pop, lag, weighting, same, page = key
        self.preview_wanted.emit({"method": method, "condition": condition,
                                  "n_pop": n_pop if n_pop == BEST else int(n_pop),
                                  "lag_frames": lag if lag == "1-4" else int(lag), "weighting": weighting,
                                  "same_D": bool(same), "fit_page": page})

    def set_preview(self, request: dict, preview: dict | None, error: str | None = None) -> None:
        """A histogram the window computed for `request` (`preview_wanted`), or why it could not be."""
        lag = request.get("lag_frames", 1)
        n_pop = request["n_pop"] if request["n_pop"] == BEST else int(request["n_pop"])
        key = (request["method"], request["condition"], n_pop, str(lag), request["weighting"],
               bool(request.get("same_D", False)), str(request.get("fit_page", "")))
        self._previews[key] = {"error": str(error)} if error is not None else dict(preview or {})
        if getattr(self, "_key", None) == "split":
            self._refresh()

    def _cutoff_controls(self, t) -> QWidget:
        """The cutoff page's controls, stacked: its type (Off / Tracks / Cells +
        Tracks), the limit (slider + typed field), and -- for the methods that
        fit -- which tracks the short/long split is learned from."""
        box = QWidget()
        box.setStyleSheet("background:transparent;")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        def row(name, widget, stretch_after=True):
            r = QWidget()
            r.setStyleSheet("background:transparent;")
            h = QHBoxLayout(r)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(12)
            caption = label(name, ty.ui("sm"), t.muted_foreground)
            caption.setFixedWidth(rem(9))
            h.addWidget(caption)
            h.addWidget(widget, 0 if stretch_after else 1)
            if stretch_after:
                h.addStretch(1)
            lay.addWidget(r)
            return r

        row("Type", self._cutoff_mode)
        # The limit on a line of its own, full width: beside a caption, the typed
        # field and a unit, the slider was left about 80 px to move in. The page's
        # title already says what the number limits.
        self._cutoff_limit_row = self._slider_row(t, self._cutoff)
        lay.addWidget(self._cutoff_limit_row)
        self._split_row = row("Short/long split from", self._split)
        box._is_slider_row = True
        return box

    def _masking_controls(self, t) -> QWidget:
        """Bin masking's controls: the two rules that blank a bin, each on its own row with
        its own name. The count, then n_eff: what counts as a contributor (Off / Tracks /
        Cells) and its value. Two settings, one page -- the caption still names them apart.

        **NUMBER FIELDS, NO SLIDERS** (ruled): on Bin size a slider earns its place because
        dragging redraws a grid you can see; here the picture is invented, and dragging it
        would feel like exploring data that is not there. The `Slider`s stay as the hidden
        MODEL behind each field, so `config()` and every caller read what they always read.
        """
        box = QWidget()
        box.setStyleSheet("background:transparent;")
        # TWO COLUMNS, the UI designer's sketch: each rule's name over its field, side by side; the
        # n_eff unit choice sits beside its name. The dialog is as wide as this needs
        # (`SpatialMapDialog.CARD_MIN_W`), so nothing here is squeezed to fit.
        grid = QGridLayout(box)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(32)
        grid.setVerticalSpacing(8)
        self._n_min_heading = label(setting_name("n_min"), ty.ui("sm"), t.foreground)
        grid.addWidget(self._n_min_heading, 0, 0, Qt.AlignLeft | Qt.AlignVCenter)
        self._n_min_unit = label("jumps", ty.ui("sm"), t.muted_foreground)
        self._n_min_row = self._field_row(t, box, self._n_min, self._n_min_unit)
        grid.addWidget(self._n_min_row, 1, 0)
        head = QWidget()
        head.setStyleSheet("background:transparent;")
        h = QHBoxLayout(head)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(12)
        self._neff_heading = label(setting_name("neff"), ty.ui("sm"), t.foreground)
        h.addWidget(self._neff_heading)
        h.addWidget(self._neff_mode)
        h.addStretch(1)
        grid.addWidget(head, 0, 1)
        # names the NUMBER, not the unit: the Off / Tracks / Cells switch beside it does that
        self._neff_unit_label = label("n_eff cutoff", ty.ui("sm"), t.muted_foreground)
        self._neff_limit_row = self._field_row(t, box, self._neff, self._neff_unit_label)
        grid.addWidget(self._neff_limit_row, 1, 1)
        grid.setColumnStretch(1, 1)
        box._is_slider_row = True
        return box

    def _field_row(self, t, box: QWidget, slider, unit: QWidget) -> QWidget:
        """A typed field and its unit, kept ONE value with `slider` (the hidden model), the
        same wiring `_slider_row` gives its own field. The slider is parented to the page
        and hidden, never left parentless: a parentless widget shown later is a window."""
        slider.setParent(box)
        slider.hide()
        row = QWidget()
        row.setStyleSheet("background:transparent;")
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        field = NumberField(t, slider.lo, slider.hi, slider.value(), width=112)
        field.changed.connect(slider.set_value)
        slider.changed.connect(field.setValue)
        lay.addWidget(field)
        lay.addWidget(unit)
        lay.addStretch(1)
        row._field = field
        return row

    def _bin_controls(self, t) -> QWidget:
        """Bin size's controls, stacked: the unit (one camera pixel of the data set, or nm),
        then the nm slider and its typed field on a line of their own, full width -- the
        cutoff page's arrangement. ``_field`` is the slider row's, so a caller typing a bin
        reaches the same field it always did."""
        box = QWidget()
        box.setStyleSheet("background:transparent;")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.addWidget(self._bin_unit)
        head.addStretch(1)
        lay.addLayout(head)
        self._bin_row = self._slider_row(t, self._bin, "nm")
        lay.addWidget(self._bin_row)
        box._is_slider_row = True
        box._field = self._bin_row._field
        return box

    def _slider_row(self, t, slider, unit: str = "") -> QWidget:
        """The slider plus a field to TYPE the same number into.

        Two views of ONE value, so neither owns it: dragging updates the
        field, typing moves the slider, and both stop at the slider's own
        range. Each side only writes when the value actually differs, which is
        what keeps the two from echoing each other forever.
        """
        row = QWidget()
        row.setStyleSheet("background:transparent;")
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)
        lay.addWidget(slider, 1)
        field = NumberField(t, slider.lo, slider.hi, slider.value(), width=112)
        field.changed.connect(slider.set_value)
        slider.changed.connect(field.setValue)
        lay.addWidget(field)
        if unit:
            lay.addWidget(label(unit, ty.ui("sm"), t.muted_foreground))
        row._is_slider_row = True
        row._field = field
        return row

    # -- what the caller sets ----------------------------------------------- #
    def set_context(self, *, measured: dict | None, pixel_size_nm: float | None,
                    condition: str = "", census=None, pixel_note: str | None = None,
                    conditions=None) -> None:
        """What this condition's own data says, read when the dialog opens.

        `measured` is `measure_cell_frame`'s answer or `None`. **`None`
        withdraws the measured option rather than quietly standing the
        deposit's capsule in for it** -- a frame that could not be measured
        and a frame measured to be 2.94 µm are different facts, and this
        project's own record of the legacy location map is that an
        unexplained default printed beside real numbers reads as a
        measurement.
        """
        self._measured = measured
        self._pixel_size_nm = pixel_size_nm
        self._pixel_note = pixel_note
        self._condition = condition
        # "1 px (106 nm)": the pill names THIS data set's pixel, and the slider stands on it
        # while 1 px is chosen (`spatial_map_view.project_pixel` gives both from the project).
        self._bin_unit.set_labels(bin_unit_labels(pixel_size_nm))
        self._snap_bin_to_pixel()
        if pixel_size_nm and not self._radius_by_hand:
            # v2's custom radius starts at one camera pixel.
            self._setting_radius = True
            self._radius.set_value(int(round(float(pixel_size_nm))))
            self._setting_radius = False
        #: `[[cell, n_localisations], ...]` for the loaded condition(s); empty
        #: until one has been run or measured -- the chart says so.
        self._census = list(census or [])
        # THE CONDITIONS THE SPLIT PAGE'S HISTOGRAM CAN SHOW, ``[(key, name), ...]``; every opening reads its
        # histograms afresh (the data may have changed: a track hidden, a condition excluded)
        self._previews = {}
        shown = self._preview_condition()
        self._preview_conditions = [(str(k), str(n)) for k, n in (conditions or ())]
        self._preview_pick.blockSignals(True)
        self._preview_pick.clear()
        self._preview_pick.addItems([name for _k, name in self._preview_conditions])
        keys = [k for k, _n in self._preview_conditions]
        self._preview_pick.setCurrentIndex(keys.index(shown) if shown in keys else (0 if keys else -1))
        self._preview_pick.blockSignals(False)
        # the Populations page's rows: one per condition, each keeping a choice made for it while it is still here
        self._pop_choice = {m: {k: v for k, v in picks.items() if k in keys} for m, picks in self._pop_choice.items()}
        self._rebuild_pop_rows()
        if measured is None and self._frame.current() == 0:
            self._frame.select(1)
        # LENGTH CLASSES NEED MEASURED LENGTHS. Without them the classes have no cuts,
        # so the control is offered only where the cells could be measured.
        if not (measured or {}).get("length_cuts_um"):
            self._cells.select(0)
            self._cells.setEnabled(False)
        else:
            self._cells.setEnabled(True)
        self._refresh()

    def frame_um(self) -> tuple[float, float]:
        length, width, _source = resolve_frame(self._measured, self._frame.current() == 0)
        return length, width

    def _cell_class(self) -> str:
        return CELL_CLASSES[self._cells.current()]

    def method_key(self) -> str:
        return METHODS[self._method.current()]

    def _rows_for(self, method: str) -> list:
        """The rows ``method`` has here (`_KEYS_BY_METHOD`)."""
        return list(_KEYS_BY_METHOD[method])

    def _bin_unit_key(self) -> str:
        return BIN_UNITS[self._bin_unit.current()]

    def _snap_bin_to_pixel(self) -> None:
        """While 1 px is chosen, the slider shows the pixel's nm (the slider's own whole nm;
        the run draws at the pixel's exact size, `spatial_map_view.dialog_bin`)."""
        if self._bin_unit_key() != "px" or not self._pixel_size_nm:
            return
        self._setting_bin = True
        try:
            self._bin.set_value(int(round(float(self._pixel_size_nm))))
        finally:
            self._setting_bin = False

    def _on_bin_unit_changed(self, *_a) -> None:
        self._snap_bin_to_pixel()
        self._refresh()

    def _on_bin_moved(self, *_a) -> None:
        """The slider moved: by hand (dragged, typed) it is a bin in nm, so the unit follows."""
        if not self._setting_bin and self._bin_unit_key() == "px":
            self._bin_unit.select(BIN_UNITS.index("nm"))      # refreshes through its own signal
            return
        self._refresh()

    def refusal(self) -> str:
        """Why Run cannot start, or "": a 1 px bin where no pixel size is known, or manual ranges that cannot run."""
        state = self._controls_state()
        return state["bin_refusal"] or state["split_refusal"] or self._precision_refused()

    def _on_radius_changed(self, *_a) -> None:
        if not getattr(self, "_setting_radius", False):
            self._radius_by_hand = True
        self._on_control_changed()

    def _on_method_changed(self, *_a) -> None:
        """The Computational method row itself changed -- unlike every other
        control, this one changes which OTHER rows exist (`_KEYS_BY_METHOD`),
        so the nav's row SET is rebuilt rather than just re-highlighted.

        Lands on `"method"` again after the rebuild, so the control the user
        just touched stays the one on screen -- an SQD-only row (Jump drawn
        at / Assignment) disappearing out from under whatever was open would
        otherwise leave the panel showing a control that no longer exists.
        """
        self._keys = self._rows_for(self.method_key())
        self._rebuild_range_rows()
        # the Populations page shows THIS method's counts and trays
        self._place_every_tray()
        self.nav.set_items([setting_name(k, self.method_key()) for k in self._keys])
        self._n_min_heading.setText(setting_name("n_min", self.method_key()))
        self._n_min_unit.setText(n_min_unit(self.method_key()))
        self._show(self._keys[0])

    def _n_pop(self):
        """Every condition's choice (a count, or Best); two for a method that fits none."""
        if self.method_key() not in MAP_N_POPS_BY_METHOD:
            return MAP_N_POP_DEFAULT
        return self._every_choice(self.method_key())

    def _controls_state(self) -> dict:
        """Every control as a plain dict, for the view's rules; NOTHING is decided here."""
        length, width, source = resolve_frame(self._measured, self._frame.current() == 0)
        measured = self._measured or {}
        return {
            "method": self.method_key(),
            "step_position": STEP_POSITIONS[self._step.current()],
            "assignment": ASSIGNMENTS[self._assignment.current()],
            "hmm_show": HMM_SHOWS[self._show_turning.current()],
            "mirrored": self._fold.current() == 1,
            "n_min": int(self._n_min.value()),
            # `bin_unit`, `bin_nm` (resolved, or None) and `bin_refusal`: the view's rule
            **dialog_bin(self._bin_unit_key(), int(self._bin.value()), self._pixel_size_nm),
            "pixel_note": self._pixel_note,
            "cell_length_um": length,
            "cell_width_um": width,
            "frame_source": source,
            "pixel_size_nm": self._pixel_size_nm,
            "weighting": WEIGHTINGS[self._weighting.current()],
            "max_locs": None if self._cutoff_mode.current() == 0 else int(self._cutoff.value()),
            "cutoff_mode": "cells" if self._cutoff_mode.current() == 2 else "tracks",
            "split_from": SPLIT_FROM[self._split.current()],
            "neff_min": None if self._neff_mode.current() == 0 else float(self._neff.value()),
            "neff_unit": "cells" if self._neff_mode.current() == 2 else "tracks",
            "n_pop": self._n_pop(),
            "n_pop_by_condition": self._n_pop_by_condition(),
            # the box "Use the same D as the ... fit" (`map_fit_source`): the current fitted method's own
            "same_D": self.same_d(),
            # the map's own fit Per spot (`spatial_map_view.OWN_PRECISION_KEY`): `map_config` keeps it only where the
            # map fits its own; refused beside lags 1-4 together (`refusal`), meanwhile drawn as not modelled
            "per_spot": self.per_spot() and not self._precision_refused(),
            # WHICH PAGE of the section (subpages): only said when it is not the section's own, so a map of the
            # section's own page has the config it always had
            **({"fit_page": self.fit_page()} if self.method_key() in self._fit_page
               and self.fit_page() != self.method_key() else {}),
            "lag_frames": MAP_LAGS[self._lag.current()],
            "cell_class": self._cell_class(),
            "class_cuts_um": measured.get("length_cuts_um"),
            "long_axis": LONG_AXES[self._axis.current()],
            "radius_nm": int(self._radius.value()),
            "min_steps": int(self._dwell.value()),
            "value_adaptive": self._adaptive.current() == 1,
            "census": self._census,
            # What the readouts say about the measured pool; not part of the run.
            "measured_cuts_um": measured.get("length_cuts_um"),
            "measured_n_cells": measured.get("n_cells"),
            **self._split_state(),
        }

    def _split_state(self) -> dict:
        """The Split page's part of the state: the split, the ranges as typed, why they cannot run (the view's rule,
        `ranges_refusal`), and the shown condition's histogram when it has arrived."""
        method = self.method_key()
        manual = method in RANGE_UNITS and self._split_mode.current() == SPLITS.index("manual")
        typed = [list(r) for r in self._ranges_by_method[self._split_method()]]
        split = "manual" if manual else SPLIT_DEFAULT
        return {"split": split, "ranges": typed if manual else None, "ranges_typed": typed,
                "split_refusal": ranges_refusal(method, split, typed, MAP_LAGS[self._lag.current()]),
                "preview": self._previews.get(self._preview_key())}

    #: `_controls_state` keys that describe the dialog, not the run.
    _NOT_CONFIG = ("pixel_size_nm", "census", "measured_cuts_um", "measured_n_cells",
                   "bin_refusal", "pixel_note", "ranges_typed", "split_refusal", "preview")

    def config(self) -> dict:
        state = self._controls_state()
        if state.get("split_refusal"):
            # the Split page says why and Run refuses (`refusal`); meanwhile every other page draws as fitted
            state["split"], state["ranges"] = SPLIT_DEFAULT, None
        return map_config(**{k: v for k, v in state.items() if k not in self._NOT_CONFIG})

    # -- a session's dialog, restored (B8, 4 Nov 2026) ------------------------------------------------------- #
    def set_config(self, config: dict | None) -> None:
        """EVERY CONTROL FROM A MAP CONFIG (`config()`'s, as a Run and a session keep it): the method first (it
        rebuilds the pages), then each setting the config names -- the bin and its unit, the frame, the cells and
        the long axis, the fold, the masking and the n_eff guard, the weighting, the cutoff and what the split is
        learned from, the jump's end and the assignment, the time lag, the populations of every condition, the split
        and its ranges, SLA's radius and dwell, SDA's adaptive smoothing, and the box "Use the same D as the ... fit"
        (a config without it unticked: a map made before kept its own fit). A key the config lacks keeps the control
        as it is. Found by the tester (B8): after Open session the grid came back and this dialog on its defaults,
        so Run drew a different map."""
        c = dict(config or {})
        if not c:
            return
        method = str(c.get("method", self.method_key()))
        if method in METHODS and method != self.method_key():
            self._method.select(METHODS.index(method))

        def pick(seg, values, value) -> None:
            values = list(values)
            if value in values:
                seg.select(values.index(value))

        if "mirrored" in c:
            self._fold.select(1 if c["mirrored"] else 0)
        if "n_min" in c:
            self._n_min.set_value(int(c["n_min"]))
        if c.get("bin_camera_px"):
            pick(self._bin_unit, BIN_UNITS, "px")
        elif c.get("bin_nm") is not None:
            self._setting_bin = True
            try:
                pick(self._bin_unit, BIN_UNITS, "nm")
                self._bin.set_value(int(round(float(c["bin_nm"]))))
            finally:
                self._setting_bin = False
        if "frame_source" in c:
            self._frame.select(0 if c["frame_source"] == "measured" else 1)
        if "cell_class" in c:
            pick(self._cells, CELL_CLASSES, c["cell_class"])
        if "long_axis" in c:
            pick(self._axis, LONG_AXES, c["long_axis"])
        if "weighting" in c:
            pick(self._weighting, WEIGHTINGS, c["weighting"])
        if "max_locs" in c:
            if c["max_locs"] is None:
                self._cutoff_mode.select(0)
            else:
                self._cutoff.set_value(int(c["max_locs"]))
                self._cutoff_mode.select(2 if c.get("cutoff_mode") == "cells" else 1)
        if "neff_min" in c:
            if c["neff_min"] is None:
                self._neff_mode.select(0)
            else:
                self._neff.set_value(int(round(float(c["neff_min"]))))
                self._neff_mode.select(2 if c.get("neff_unit") == "cells" else 1)
        if "split_from" in c:
            pick(self._split, SPLIT_FROM, c["split_from"])
        if "step_position" in c:
            pick(self._step, STEP_POSITIONS, c["step_position"])
        if "assignment" in c:
            pick(self._assignment, ASSIGNMENTS, c["assignment"])
        if method == "hmm" and c.get("hmm_show") in HMM_SHOWS:
            self._show_turning.select(HMM_SHOWS.index(c["hmm_show"]))
        if "lag_frames" in c:
            pick(self._lag, [str(v) for v in MAP_LAGS], str(c["lag_frames"]))
        if "value_adaptive" in c:
            self._adaptive.select(1 if c["value_adaptive"] else 0)
        if "radius_nm" in c:
            self._setting_radius = True
            self._radius.set_value(int(round(float(c["radius_nm"]))))
            self._setting_radius = False
            self._radius_by_hand = True
        if "min_steps" in c:
            self._dwell.set_value(int(c["min_steps"]))
        if method in MAP_N_POPS_BY_METHOD and "n_pop" in c:
            choices = self._choices(method)
            n_pop = c["n_pop"] if c["n_pop"] == BEST else int(c["n_pop"])
            if n_pop in choices:
                self._every_last = {**getattr(self, "_every_last", {}), method: n_pop}
                self._setting_pops = True
                try:
                    self._pops_by_method[method].select(choices.index(n_pop))
                finally:
                    self._setting_pops = False
            # a row set apart is one whose choice differs from Every condition's (`_on_row_count`), so only those
            by = {str(k): (v if v == BEST else int(v)) for k, v in dict(c.get("n_pop_by_condition") or {}).items()}
            self._pop_choice[method] = {k: v for k, v in by.items() if v in choices and v != n_pop}
        if method in self._same_d:
            self._same_d[method] = bool(c.get("same_D", False))
        if method in self._per_spot and not c.get("same_D") and c.get("split") != "manual":
            # only a config whose map fits its own speaks for the choice (`map_config` keeps it nowhere else)
            self._per_spot[method] = bool(c.get("per_spot", False))
        if method in self._fit_page:
            self._fit_page[method] = str(c.get("fit_page") or method)
            self._sync_fit_page()
        if method in RANGE_UNITS:
            if c.get("split") == "manual" and c.get("ranges"):
                self._ranges_by_method[method] = [[float(lo), float(hi)] for lo, hi in c["ranges"]]
            pick(self._split_mode, SPLITS, c.get("split", SPLIT_DEFAULT))
        self._rebuild_range_rows()
        self._rebuild_pop_rows()
        self._refresh()

    def state(self) -> dict:
        """The dialog as a session keeps it: the Run's config (`config()`) and what a config cannot hold -- every
        fitted method's own box, counts and per-condition choices, each split method's typed ranges (kept while
        Fitted is chosen, which a config drops), the cutoff's and the n_eff guard's limits while they are off, SLA's
        radius set by hand and Show tips. Plain data."""
        return {"config": self.config(), "method": self.method_key(),
                "same_d": dict(self._same_d), "per_spot": dict(self._per_spot), "fit_page": dict(self._fit_page),
                "fit_names": {m: dict(v) for m, v in self._fit_names.items()},
                "every": {m: self._every_choice(m) for m in MAP_N_POPS_BY_METHOD},
                "pop_choice": {m: dict(v) for m, v in self._pop_choice.items()},
                "ranges": {m: [list(r) for r in v] for m, v in self._ranges_by_method.items()},
                "split_manual": self._split_mode.current() == SPLITS.index("manual"),
                "cutoff": int(self._cutoff.value()), "neff": int(self._neff.value()),
                "bin": int(self._bin.value()), "radius": int(self._radius.value()),
                "radius_by_hand": bool(self._radius_by_hand), "dwell": int(self._dwell.value()),
                "tips": bool(self.tips.checked)}

    def set_state(self, state: dict | None) -> None:
        """A session's dialog (`state`), every control as it was saved: what only the state holds first, then the
        config, so the config decides wherever both speak."""
        s = dict(state or {})
        if not s:
            return
        for method, on in (s.get("same_d") or {}).items():
            if method in self._same_d:
                self._same_d[method] = bool(on)
        for method, on in (s.get("per_spot") or {}).items():
            if method in self._per_spot:
                self._per_spot[method] = bool(on)
        for method, key in (s.get("fit_page") or {}).items():
            if method in self._fit_page:
                self._fit_page[method] = str(key)
        for method, names in (s.get("fit_names") or {}).items():
            if method in self._fit_names:
                self._fit_names[method].update({str(k): str(v) for k, v in names.items()})
        for method, choice in (s.get("every") or {}).items():
            if method in self._pops_by_method:
                choices = self._choices(method)
                choice = choice if choice == BEST else int(choice)
                if choice in choices:
                    self._every_last = {**getattr(self, "_every_last", {}), method: choice}
                    self._setting_pops = True
                    try:
                        self._pops_by_method[method].select(choices.index(choice))
                    finally:
                        self._setting_pops = False
        for method, picks in (s.get("pop_choice") or {}).items():
            if method in self._pop_choice:
                self._pop_choice[method] = {str(k): v for k, v in dict(picks or {}).items()}
        for method, ranges in (s.get("ranges") or {}).items():
            if method in self._ranges_by_method and ranges:
                self._ranges_by_method[method] = [[float(lo), float(hi)] for lo, hi in ranges]
        for name, slider in (("cutoff", self._cutoff), ("neff", self._neff), ("dwell", self._dwell)):
            if s.get(name) is not None:
                slider.set_value(int(s[name]))
        if s.get("radius") is not None:
            self._setting_radius = True
            self._radius.set_value(int(s["radius"]))
            self._setting_radius = False
        self._radius_by_hand = bool(s.get("radius_by_hand", self._radius_by_hand))
        if s.get("bin") is not None:
            self._setting_bin = True
            try:
                self._bin.set_value(int(s["bin"]))
            finally:
                self._setting_bin = False
        if "tips" in s and bool(s["tips"]) != bool(self.tips.checked):
            self.tips.set_checked(bool(s["tips"]), animate=False)
            self._set_tips(bool(s["tips"]))
        self.set_config(s.get("config"))
        if s.get("split_manual") is not None and self.method_key() in RANGE_UNITS:
            self._split_mode.select(SPLITS.index("manual" if s["split_manual"] else SPLIT_DEFAULT))
            self._rebuild_range_rows()
        self._refresh()

    def _on_cutoff_dragged(self, limit: int) -> None:
        """The chart's line was dragged: move the slider (the typed field follows
        it). Dragging while the cutoff is Off turns it on, in Tracks mode -- the
        line being dragged is the cutoff, so leaving it off would ignore the hand."""
        if self._cutoff_mode.current() == 0:
            self._cutoff_mode.select(1)
        self._cutoff.set_value(limit)

    # -- the value each nav row shows, and the live readout under each picture -- #
    def _value_text(self, key: str) -> str:
        return setting_value_text(key, self._controls_state())

    def _read_text(self, key: str) -> str:
        return setting_readout(key, self._controls_state())

    def _warn_text(self, key: str) -> str:
        """A WARNING (see `spatial_map_view.bin_warning`), or the one refusal the page has: a 1 px
        bin where no pixel size is known (`spatial_map_view.bin_page_warning`)."""
        if key == "split":
            return self._controls_state()["split_refusal"]
        if key != "bin":
            return ""
        return bin_page_warning(self._controls_state())

    # -- showing one setting ------------------------------------------------ #
    def _show(self, key: str) -> None:
        self._key = key
        self._title.setText(setting_name(key, self.method_key()))
        self._desc.setText(_SETTING_DESC[key])
        while self._control_lay.count():
            item = self._control_lay.takeAt(0)
            w = item.widget()
            if w is not None:
                # Hidden BEFORE it is detached, or a queued show from an earlier
                # `_show` in the same turn opens it as a window (`discard`'s note).
                # Kept, not deleted: the control is reused, and shown again below.
                w.hide()
                w.setParent(None)
        control = self._controls[key]
        if getattr(control, "_is_slider_row", False):
            # a slider row: it must take the width, not be pushed aside by a
            # stretch -- `Slider` is Expanding with no minimum, so a stretch
            # beside it collapses the track to nothing and leaves only the
            # number field on screen.
            self._control_lay.addWidget(control, 1)
        else:
            self._control_lay.addWidget(control)
            self._control_lay.addStretch(1)
        control.show()
        self._refresh()
        if key == "jump" and self._card.isVisible():
            self._art.play_intro()
        elif key == "method" and self._card.isVisible():
            self._art.play_method_intro()

    def _on_control_changed(self, *_a) -> None:
        self._refresh()

    def _refresh(self) -> None:
        """Every value, readout and picture, from the controls as they are.

        ONE path for all of it, called from every control's own signal, so
        the nav cannot show one thing while the picture shows another --
        which is the failure a two-level nav makes most visible.
        """
        key = getattr(self, "_key", self._keys[0])
        self._sync_same_d()
        for i, k in enumerate(self._keys):
            self.nav.set_value(i, self._value_text(k))
        self._read.setText(self._read_text(key))
        help_ = setting_help(key, self._controls_state())
        self._help.setText("" if help_ is None else help_[0])
        self._help.set_tip("" if help_ is None else help_[1])
        tips = getattr(self, "tips", None)
        self._help.setVisible(help_ is not None and (tips is None or bool(tips.checked)))
        warn = self._warn_text(key)
        self._warn.setText(warn)
        self._warn.setVisible(bool(warn))
        state = self.config()
        if state.get("bin_nm") is None:
            # a 1 px bin with no pixel size: the page names the refusal and Run refuses it;
            # the illustration keeps the slider's grid rather than no grid
            state["bin_nm"] = self._bin.value()
        state["measured"] = self._measured
        state["pixel_size_nm"] = self._pixel_size_nm
        # The raw tab pick, decoupled from `frame_source` -- see
        # `_SettingArt.show_setting`'s own comment on why the resolved
        # string cannot be used to detect a genuine click.
        state["frame_selected"] = "measured" if self._frame.current() == 0 \
            else "deposit"
        state["census"] = self._census
        # THE SPLIT PAGE draws what was typed, even while it cannot run, on the shown condition's histogram
        state.update({k: v for k, v in self._split_state().items() if k != "ranges"})
        state["split_chosen"] = self._split_state()["split"]
        state["preview_condition"] = dict(self._preview_conditions).get(self._preview_condition(), "")
        if key == "split":
            self._want_preview()
            state["preview"] = self._previews.get(self._preview_key())
        # The limit and the split only mean something with the cutoff on; the
        # split does not exist for a method that fits nothing (appD).
        cutoff_on = self._cutoff_mode.current() != 0
        self._cutoff_limit_row.setEnabled(cutoff_on)
        self._neff_limit_row.setEnabled(self._neff_mode.current() != 0)
        self._split_row.setVisible(self.method_key() in ("sqd", "hmm"))
        self._split_row.setEnabled(cutoff_on)
        self._art.show_setting(key, state)

    def _set_tips(self, on: bool) -> None:
        if not on:
            self._art.stop_intro()
        for w in (self._subtitle, self._desc, self._art, self._read):
            w.setVisible(on)
        self._help.setVisible(on and bool(self._help.text()))
        self._resync_after_resize()

    # -- opening and closing ------------------------------------------------ #
    def open(self) -> None:
        """Every opening gets the jump intro once again."""
        self._art.reset_intro()
        super().open()

    def _on_dialog_anim_finished(self) -> None:
        """Once the card is actually on screen, a dialog that opened on the jump
        page plays its intro -- before this the card is hidden behind the
        opening animation's snapshot, so an intro started earlier would be
        unseen."""
        opening = not self._anim_closing
        super()._on_dialog_anim_finished()
        if opening and self._key == "jump":
            self._art.play_intro()
        elif opening and self._key == "method":
            self._art.play_method_intro()

    def _dismiss(self) -> None:
        self._art.stop_intro()
        super()._dismiss()

    # -- exits -------------------------------------------------------------- #
    def _on_run(self) -> None:
        if self.refusal():
            # REFUSED, ON THE PAGE THAT DECIDES IT: a 1 px bin with no pixel size is never
            # run at a stand-in (`spatial_map_view.BinSizeError`); ranges that overlap or
            # run backwards are never run as anything else (`ranges_refusal`).
            page = "split" if self._controls_state()["split_refusal"] else "bin"
            if page in self._keys:
                self.nav.select(self._keys.index(page))
                self._show(page)
            return
        self._acted = True
        self.run.emit()
        self._dismiss()

    def _on_cancel(self) -> None:
        self._dismiss()

    def _on_dismiss_without_action(self) -> None:
        self.cancelled.emit()


# ---------------------------------------------------------------------------
#  The Spatial Map SECTION -- every condition at once
# ---------------------------------------------------------------------------
#
# WHY A SECTION AND NOT A DEEPER DRILL-IN. The map sat four levels down
# (`SQD > Populations > <condition> > Map`), which was right while it was one
# method belonging to one fit. It stops being right the moment the view is
# meant to host appD and HMM as well, since neither of those is an SQD result.
# `docs/gui_tool_views.md` rules on the ESTIMATOR, the RENDERING and on
# REACHABILITY ("anything not in `SECTIONS` is, in practice, not in the
# application") and says nothing anywhere about where a view sits, so this is
# a decision rather than a compliance change. Its own precedent points the
# same way: it asks for a sidebar section for the QC dashboard on the grounds
# that the view is central, not on the grounds of whose estimator it is.
#
# SQD AND APPARENT DIFFUSION, HMM DEFERRED. `SpatialMapDialog` now carries a
# Computational method row (SQD / Apparent Diffusion), and `compute_spatial_map`
# dispatches on it -- but NOT by routing to `comparison.spatial_maps(source=
# "appd")`, whose `render`-based pipeline draws a DIFFERENT SHAPE (normalised
# by the LOCAL half-width, so the cell IS the rectangle and there is no
# outside). This grid stays in `spatialsqd`'s own capsule frame for both
# methods, matching this file's already-settled rule: "each method draws in
# the frame it computed in, named in the caption" -- so appD gets its own
# `smtracker.spatialappd` module in that same frame instead. Nothing here
# assumed SQD was the only source: the matrix takes payloads, not a fit.
#
# THE PER-CONDITION MAP IS GONE. The SQD drill-in's Map tab sat beside this section for a
# while; it was removed by request, so the map is reached here only.


class SpatialMapMatrix(Card):
    """Every loaded condition's maps in one grid: conditions down the page,
    populations across it.

    **The shape `compare_plots.draw_spatial` already draws**, and the reason
    is not fidelity to the old application. With every condition on screen the
    colour scale's SHARING becomes a fact you can see -- one bar under each
    population column -- instead of a setting somebody has to reason about in
    the abstract. Sparse rule A (one scale per population, pooled across
    conditions) is the DEFAULT. **The other scales are offered too, by ruling**
    ("does it make sense to include an option of how to scale the maps? in
    between condition, in between population, everything on one scale?"): the
    Visibility tool's Colour scale (`spatial_map_view.MATRIX_SCALES`) -- per
    condition, per map, or one for all -- because each answers a different
    question. Shared scales compare densities; a scale that is not shared by a
    column prints its range under each map, and the caption names the scale.

    **One bar per column is only honest because one Run fills the whole
    grid.** `_panel_limits` refuses to pool panels computed at different
    settings, so a grid holding a stale row would be showing a bar that
    describes some of its panels and not others. `SpatialMapSection.begin_run`
    clears the previous results, so that state cannot arise here.

    A condition that has not landed yet is drawn as an EMPTY CAPSULE, not
    omitted: the grid's row count is what says how much is still coming, and a
    row appearing from nowhere reads as a layout jump rather than as progress.

    **Two uses, one grid.** The Spatial Map section builds it `clickable`: a
    condition's maps open its replicate days (`condition_clicked`), and a hint
    under the grid says so. The replicate page (`spot_page.SpotPage`) builds
    it with `row_unit="day"`: its rows are one condition's replicates, and the
    caption and the cliff warning say "replicate" instead of "condition".
    """

    condition_clicked = Signal(str)
    #: A spot on a DAY row's map was clicked: (row key, panel kind, ix, iy).
    spot_clicked = Signal(str, str, int, int)
    #: The settings this grid was drawn with changed (`caption_rows`): the Metadata
    #: receipt, which now holds them, follows.
    caption_changed = Signal()
    #: The head's Export button (`export_payload` is what it draws).
    export_requested = Signal()

    def __init__(self, t, parent=None, *, clickable: bool = False,
                 row_unit: str = "condition"):
        super().__init__(t, parent=parent)
        if row_unit not in ("condition", "day"):
            raise ValueError(f"row_unit must be 'condition' or 'replicate', got {row_unit!r}")
        self.t = t
        self._clickable = bool(clickable)
        self._row_unit = row_unit
        self._cmap = DEFAULT_CMAP
        #: The ramp picked for each method, so SDA opens in v2's ramp and every other
        #: method in jet (`spatial_map_view.default_cmap_for`) until the user picks one,
        #: and a pick made for one method stays with that method.
        self._picked: dict[str, str] = {}
        self._rendering = RENDERINGS[0]
        #: The colour scale (`spatial_map_view.MATRIX_SCALES`), from the Visibility tool.
        self._scale = MATRIX_SCALES[0]
        self._conditions: list[str] = []
        self._payloads: dict[str, dict] = {}
        self._errors: dict[str, str] = {}
        #: condition -> {"label": w, "charts": {kind: chart}, "figures": {kind: w}}
        self._rows: dict[str, dict] = {}
        #: condition key -> the name to print on its row. See `set_conditions`.
        self._labels: dict[str, str] = {}
        self._heads: dict[str, object] = {}
        self._bars: dict[str, _ColourBar] = {}
        #: A bar under every map, and a vertical one beside every row: shown only where the colour scale puts
        #: them (`spatial_map_view.bar_layout`) -- per map, per condition -- the column bars otherwise.
        self._map_bars: dict[str, dict[str, _ColourBar]] = {}
        self._row_bars: dict[str, _ColourBar] = {}
        self._bar_layout = "columns"
        #: Which computational method the CURRENT grid is built for --
        #: `set_method` changes it and forces the next `set_conditions` to
        #: rebuild rather than early-returning (see both methods' own
        #: comments). Kinds/titles/panel-key vocabulary all follow this.
        self._method = "sqd"
        #: The run's population count -- three only for a three-population appD map.
        self._n_pop = 2
        #: A manual split's ranges (`spatial_map_view.SPLITS`): the columns ARE the ranges; None for a fitted split.
        self._ranges = None
        #: Every count the rows are fitted at (`spatial_map_view.kinds_for_counts`): the columns are every panel
        #: some row draws; a Best row's count joins when its maps land (`set_payload`).
        self._counts: tuple = ()
        #: Each row's count as the Run planned it (`spatial_map_view.planned_n_pops`): a column its count never
        #: draws is NOT DRAWN from the start, never "waiting…" (`cell_drawn`); a landed row reads its own maps.
        self._planned: dict = {}
        self._shown: dict[str, bool] = {k: True for k in self.kinds()}
        #: The hover lens every capsule hands its moves to (`set_loupe`), or None.
        self._loupe = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(CARD_PAD, CARD_PAD, CARD_PAD, CARD_PAD)
        outer.setSpacing(CARD_GAP)
        self._head_title = label("Jump-density map", ty.ui("sm", "medium"),
                                 t.foreground)
        from ..widgets import tooltip as _tip
        from .toolbar import ToolButton

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(8)
        head.addWidget(self._head_title)
        # A TIP beside the title (`set_tip`): the Spatial Map's replicate page's alone, hidden here.
        from ..widgets.primitives import icon as _icon

        self._tip_icon = _icon("lightbulb", rem(1), t.muted_foreground)
        from ..widgets.primitives import EndElided

        # ELIDED AT ITS END where the card is narrow, the whole tip on hover (found by the overnight tester, 4 Nov
        # 2026: "Tip: click any spot on a map to see what makes" was cut under the export button at 1366 px)
        self._tip_text = EndElided("", ty.ui("xs"), t.muted_foreground)
        head.addSpacing(rem(0.75))
        head.addWidget(self._tip_icon)
        head.addWidget(self._tip_text, 1)
        self._tip_icon.hide()
        self._tip_text.hide()
        head.addStretch(1)
        self.export_button = ToolButton(t, "export", "download", size=rem(1.75), rest=t.card)
        self.export_button.clicked.connect(lambda _k: self.export_requested.emit())
        self._export_tip = _tip.attach(t, self.export_button,
                                       "Export figure (PDF, SVG, PNG, TIFF)", side="bottom",
                                       top_level=True)
        self.export_button.setEnabled(False)
        head.addWidget(self.export_button)
        outer.addLayout(head)

        body = QWidget()
        body.setStyleSheet("background:transparent;")
        self._grid = QGridLayout(body)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(CARD_GAP)
        self._grid.setVerticalSpacing(4)
        outer.addWidget(body)

        #: The hard-assignment cliff, once for the whole grid and naming the
        #: conditions it applies to -- see `SpatialMapGrid`'s own copy. One
        #: warning per row would repeat the same sentence four times.
        self._warning = label("", ty.ui("xs", "medium"), _WARN_COLOR)
        self._warning.setWordWrap(True)
        self._warning.hide()
        outer.addWidget(self._warning)

        # THE SETTINGS LIST LEFT THE PAGE (ruled by the UI designer: 17-19 rows under every grid
        # "scream for attention" and most readers need them rarely). The rows are still
        # built here, from the same call, and the toolbar's Metadata tool prints them as
        # a receipt (`spatial_map_view.spatial_receipt`); nothing is drawn under the grid.
        self.caption_rows: list = []

        #: What n_eff is and what it costs, on demand, while the grid paints it.
        self._neff_info = _InfoLink(t, "What is n_eff, and what does it cost?", N_EFF_ABOUT)
        self._neff_info.hide()
        outer.addWidget(self._neff_info)

        #: Only on a clickable grid, and only once there is something to click.
        self._hint = label(DAYS_HINT, ty.ui("xs"), t.muted_foreground)
        self._hint.hide()
        outer.addWidget(self._hint)

    # -- structure ---------------------------------------------------------- #
    def kinds(self) -> tuple:
        return kinds_for_counts(self._method, self._counts or (self._n_pop,), self._ranges)

    def panel_titles(self) -> tuple:
        return panel_titles_for_counts(self._method, self._counts or (self._n_pop,), self._ranges)

    def panel_keys(self) -> dict:
        return panel_keys_for_counts(self._method, self._counts or (self._n_pop,), self._ranges)

    def set_method(self, method: str, n_pop: int = 2, ranges=None, counts=None, planned=None) -> None:
        """Which computational method the NEXT `set_conditions` builds for.

        Only sets a flag and drops the memoised condition list -- it does
        not rebuild here. `begin_run` always calls `set_conditions` right
        after this, and that is where the rebuild actually happens (it
        already has the loop that touches every column); a second rebuild
        path here would be the two-sources-of-truth failure this project
        keeps recording. Clearing `self._conditions` is what defeats
        `set_conditions`'s own early-return guard when the CONDITION LIST
        happens to be unchanged between an SQD run and an appD run.
        """
        n_pop = int(n_pop or 2)
        ranges = None if not ranges else [list(r) for r in ranges]
        # before the unchanged-check: a Run with the same counts may give them to other rows (A 1 and B 2, then A 2
        # and B 1), and the grid it keeps must know which
        self._planned = {str(c): n for c, n in (planned or {}).items()}
        # ``counts`` () is a Run whose every row is Best: no column is known until one lands
        counts = (tuple(sorted({int(n) for n in counts})) if counts else
                  ((n_pop,) if counts is None else ()))
        if (method == self._method and n_pop == self._n_pop and ranges == self._ranges
                and counts == self._counts):
            return
        if method != self._method:
            # Each method opens in its own ramp (v2's for SDA, jet for the rest) until
            # one is picked for it; display only, nothing is recomputed.
            self._cmap = self._picked.get(method, default_cmap_for(method))
        self._method = method
        self._n_pop = n_pop
        self._ranges = ranges
        self._counts = counts
        self._conditions = []
        self._shown = {k: True for k in self.kinds()}

    def set_scale(self, level: str) -> None:
        if level not in MATRIX_SCALES:
            raise ValueError(f"scale must be one of {MATRIX_SCALES}, got {level!r}")
        if level != self._scale:
            self._scale = level
            self._redraw()

    def scale(self) -> str:
        return self._scale

    def redraw(self) -> None:
        """Repaint with the current settings -- after a custom ramp's colours changed under
        the same key (`spatial_map_chart.register_custom_ramps`)."""
        self._redraw()

    def set_grouping(self, factors: dict, nesting) -> None:
        """Group the rows by the folder design's factors (`factor_layout`); the grid is rebuilt at its next
        `set_conditions`, and now when it has rows."""
        self._grouping = (dict(factors or {}), tuple(nesting or ()))
        if getattr(self, "_conditions", None):
            self.set_conditions(list(self._conditions), self._labels)

    def set_conditions(self, conditions, labels: dict | None = None) -> None:
        """Rebuild the grid for this list of conditions, in this order.

        `conditions` are the KEYS every payload is filed under; `labels` maps
        a key to what the row should be called. The two are separate because
        they are separate everywhere else in this workspace -- the sidebar and
        the Conditions tool carry `(key, label)` pairs, and a grid that
        labelled its rows with the key would be the only surface in the
        application showing the internal name.

        Rebuilds rather than diffs: the list changes only when a project is
        loaded or the run set is chosen, which is not a per-frame event, and a
        diff here would be the more complex of the two for no gain.
        """
        conditions = [str(c) for c in conditions]
        # GROUPED BY THE FOLDER DESIGN'S FACTORS (`set_grouping`): the rows in the nesting's order, each headed by its
        # inner values, the first of a group by its outer value above them. Ordered BEFORE the unchanged-check, so an
        # unchanged grid is still left alone (a rebuild empties its maps).
        heads = {}
        grouping = getattr(self, "_grouping", ({}, ()))
        factors, nesting = grouping
        if factors and len(nesting) >= 2:
            conditions = nested(conditions, factors, nesting)
            layout = columns(conditions, factors, nesting)
            if layout["groups"]:
                firsts = {g["first"]: g["label"] for g in layout["groups"]}
                heads = {c: (firsts.get(i), layout["labels"][i]) for i, c in enumerate(conditions)}
        if (conditions == self._conditions and self._rows
                and dict(labels or {}) == self._labels and grouping == getattr(self, "_built_grouping", None)):
            return
        self._labels = dict(labels or {})
        self._built_grouping = grouping
        self._conditions = conditions
        self._head_title.setText(grid_title(self._method))
        while self._grid.count():
            item = self._grid.takeAt(0)
            w = item.widget()
            if w is not None:
                discard(w)
        self._rows, self._heads, self._bars = {}, {}, {}
        self._map_bars, self._row_bars = {}, {}
        t = self.t
        n_kinds = len(self.kinds())

        for col, (kind, title) in enumerate(zip(self.kinds(), self.panel_titles()),
                                            start=1):
            head = label(title, ty.ui("xs", "medium"), t.foreground)
            self._grid.addWidget(head, 0, col)
            self._heads[kind] = head

        for i, condition in enumerate(conditions):
            # Three grid rows per condition: the capsules, a colour bar under each map (shown only when each
            # map has a scale of its own), then their figures line; and a vertical bar to the right of the
            # capsules (shown when the row shares one scale). The name sits on the FIRST row, top aligned, so
            # it reads as the row's heading rather than as a caption floating beside the middle of a capsule.
            r = 1 + 3 * i
            if condition in heads:
                outer, inner = heads[condition]
                text = (f"<span style='color:{t.muted_foreground}; font-weight:600'>{outer}</span><br>{inner}"
                        if outer else inner)
                name = label(text, ty.ui("xs", "medium"), t.foreground)
                name.setTextFormat(Qt.RichText)
            else:
                name = label(self._labels.get(condition, condition),
                             ty.ui("xs", "medium"), t.foreground)
            name.setWordWrap(True)
            self._grid.addWidget(name, r, 0, 3, 1, Qt.AlignTop | Qt.AlignLeft)
            charts, figures, map_bars = {}, {}, {}
            for col, kind in enumerate(self.kinds(), start=1):
                chart = _CapsuleChart(t)
                chart.loupe = self._loupe
                chart.clicked.connect(
                    lambda c=condition, k=kind: self._on_row_clicked(c, k))
                chart.spot_clicked.connect(
                    lambda ix, iy, c=condition, k=kind: self.spot_clicked.emit(c, k, ix, iy))
                self._grid.addWidget(chart, r, col)
                map_bar = _ColourBar(t)
                self._grid.addWidget(map_bar, r + 1, col)       # in the layout BEFORE it is hidden
                map_bar.hide()
                fig = label("", ty.ui("xs", tabular=True), t.muted_foreground)
                # WRAPPED: a failed condition prints its whole error here, and an
                # unwrapped line made the window demand 1921 px on a 1920 px screen
                # ("QWindowsWindow::setGeometry: Unable to set geometry").
                fig.setWordWrap(True)
                self._grid.addWidget(fig, r + 2, col)
                charts[kind] = chart
                figures[kind] = fig
                map_bars[kind] = map_bar
            row_bar = _ColourBar(t, vertical=True)
            self._grid.addWidget(row_bar, r, n_kinds + 1)
            row_bar.hide()
            self._rows[condition] = {"label": name, "charts": charts,
                                     "figures": figures}
            self._map_bars[condition] = map_bars
            self._row_bars[condition] = row_bar

        bar_row = 1 + 3 * len(conditions)
        for col, kind in enumerate(self.kinds(), start=1):
            bar = _ColourBar(t)
            self._grid.addWidget(bar, bar_row, col)
            self._bars[kind] = bar

        # The name column takes what it needs; the two map columns split the
        # rest evenly, so the capsules stay the same size as each other
        # whatever the longest condition name happens to be.
        for kind in self.kinds():
            self.set_panel_visible(kind, self._shown[kind])
        self._sync_column_stretch()
        self._redraw()

    def _on_row_clicked(self, condition: str, kind: str | None = None) -> None:
        """Only a row that HAS maps is a way in: one still waiting, or one
        showing its error, has no days to show. ``kind``: the population whose map was clicked, kept
        (`last_kind`) so the day page's profiles open on it."""
        if self._clickable and condition in self._payloads:
            self.last_kind = kind
            self.condition_clicked.emit(condition)

    # -- payloads ----------------------------------------------------------- #
    def clear_payloads(self) -> None:
        self._payloads, self._errors = {}, {}
        self._redraw()

    def set_payload(self, condition: str, payload: dict) -> None:
        self._payloads[condition] = payload
        self._errors.pop(condition, None)
        # A COUNT THE GRID HAS NO COLUMN FOR (a Best row's, known only now): the columns grow, the maps kept
        n = payload.get("n_pop")
        if (n and not self._ranges and payload.get("split") != "manual"
                and payload.get("method", self._method) == self._method and int(n) not in self._counts):
            self._counts = tuple(sorted({*self._counts, int(n)}))
            self._shown = {k: self._shown.get(k, True) for k in self.kinds()}
            conditions, labels = list(self._conditions), dict(self._labels)
            self._conditions = []
            self.set_conditions(conditions, labels)
        # EVERY row is redrawn, not just this one: a new condition widens the
        # pooled colour limits, and a scale that silently went stale is the
        # order-dependent colour bar the rendering spec's decision 5 exists to
        # rule out.
        self._redraw()

    def set_error(self, condition: str, message: str) -> None:
        """A condition that failed is NAMED in its own row, never left blank.

        `catalogue_view`'s own rule -- an empty panel reads as "no data" where
        the truth is "this one raised", and the two want different responses.
        """
        self._errors[condition] = message
        self._payloads.pop(condition, None)
        self._redraw()

    def has_payloads(self) -> bool:
        return bool(self._payloads)

    def receipt(self, *, computed_at: str | None = None):
        """This grid's Metadata receipt (`spatial_map_view.spatial_receipt`), or None
        when nothing is drawn. Same rows as `caption_rows`, same call, nothing refitted."""
        payloads = [self._payloads[c] for c in self._conditions if c in self._payloads]
        return spatial_receipt(payloads, self._cmap, self._rendering,
                               unit=self._row_unit, scale=self._scale,
                               computed_at=computed_at)

    def payload(self, condition: str) -> dict | None:
        return self._payloads.get(condition)

    def set_overlay(self, condition: str, kind: str, overlay: dict | None) -> None:
        """Draw a clicked spot's tracks over one map, clearing every other map's."""
        for key, row in self._rows.items():
            for k, chart in row["charts"].items():
                chart.set_overlay(overlay if (key, k) == (condition, kind) else None)

    def set_tip(self, text: str | None) -> None:
        """A tip beside the title, with a lightbulb; None hides it (the default)."""
        on = bool(text)
        self._tip_text.set_full_text(text or "")
        self._tip_icon.setVisible(on)
        self._tip_text.setVisible(on)

    def set_loupe(self, loupe) -> None:
        """A magnifying lens for every capsule, now and after a rebuild (the replicate page's)."""
        self._loupe = loupe
        for row in self._rows.values():
            for chart in row["charts"].values():
                chart.loupe = loupe

    def chart(self, condition: str, kind: str):
        return self._rows[condition]["charts"][kind]

    def conditions(self) -> list[str]:
        """The row keys, in row order."""
        return list(self._conditions)

    def payloads(self) -> dict:
        return dict(self._payloads)

    def export_payload(self) -> dict | None:
        """What the Export draws: the grid AS ON SCREEN -- its rows and their names, every
        payload and error, the colour scale, the ramp's own colours, the rendering and the
        population columns shown. None before any map has landed."""
        if not self._payloads:
            return None
        return {"rows": list(self._conditions), "labels": dict(self._labels),
                "payloads": dict(self._payloads), "errors": dict(self._errors),
                "scale": self._scale, "rendering": self._rendering, "cmap": self._cmap,
                "ramp": list(ramp_for(self._cmap) or CMAP_RAMPS[DEFAULT_CMAP]),
                "shown": dict(self._shown), "row_unit": self._row_unit,
                "method": self._method, "n_pop": self._n_pop, "ranges": self._ranges,
                "counts": list(self._counts)}

    # -- display ------------------------------------------------------------ #
    def set_colormap(self, name: str) -> None:
        if ramp_for(name) is None:
            return
        self._picked[self._method] = name
        if name != self._cmap:
            self._cmap = name
            self._redraw()

    def colormap(self) -> str:
        return self._cmap

    def set_rendering(self, mode: str) -> None:
        if mode not in RENDERINGS:
            raise ValueError(f"rendering must be one of {RENDERINGS}, got {mode!r}")
        if mode != self._rendering:
            self._rendering = mode
            self._redraw()

    def rendering(self) -> str:
        return self._rendering

    def set_panel_visible(self, kind: str, on: bool) -> None:
        """Hide a whole population COLUMN -- its header, every condition's
        panel and figures, and the one bar underneath.

        As in `SpatialMapGrid`, **the colour limits do not move.** They are
        taken over the whole set before anything is drawn, and a hidden
        column is still part of that set; rescaling on hide would make the
        same colour mean two densities depending on what else is on screen.
        """
        if kind not in self._shown:
            return
        self._shown[kind] = on
        # THE COLUMNS KEEP THEIR WIDTH, and a hidden one CLOSES UP (ruled 4 Nov 2026, D15: "close it on screen
        # too", as the export does). Growing the survivors was tried and reverted: a capsule is HEIGHT-limited once
        # its column is wide enough, so it stopped growing and floated in the middle of its column while the header
        # and the figures line stayed hard left -- the ragged edge this card was corrected for once. So the shown
        # columns keep the width they have with every column in, side by side, and the hidden one's width goes to
        # an empty column after the last (`_sync_column_stretch`): no gap between maps, and every picture lines up
        # with its own label.
        self._sync_column_stretch()
        head = self._heads.get(kind)
        if head is not None:
            head.setVisible(on)
        bar = self._bars.get(kind)
        if bar is not None:
            bar.setVisible(on and bar.property("has_scale") is True)
        for row in self._rows.values():
            row["charts"][kind].setVisible(on)
            row["figures"][kind].setVisible(on)
        for bars in self._map_bars.values():
            map_bar = bars.get(kind)
            if map_bar is not None:
                map_bar.setVisible(on and map_bar.property("has_scale") is True)

    def _sync_column_stretch(self) -> None:
        """The grid's columns: the names (0) take what they need, every SHOWN map column one share, a hidden one none
        (it closes up), the row bars' column (after the maps) none, and an empty column after it one share per hidden
        map, so the shown columns keep the width they have with every column in (`set_panel_visible`). Every other
        column a rebuild with fewer kinds left behind is reset."""
        kinds = self.kinds()
        hidden = sum(1 for k in kinds if not self._shown.get(k, True))
        spare = len(kinds) + 2
        for col in range(max(self._grid.columnCount(), spare + 1)):
            if 1 <= col <= len(kinds):
                stretch = 1 if self._shown.get(kinds[col - 1], True) else 0
            else:
                stretch = hidden if col == spare else 0
            self._grid.setColumnStretch(col, stretch)

    def bar_layout(self) -> str:
        """Where this grid's colour bars stand now (`spatial_map_view.bar_layout`)."""
        return self._bar_layout

    def panel_states(self) -> dict[str, bool]:
        return dict(self._shown)

    def _redraw(self) -> None:
        payloads = [self._payloads[c] for c in self._conditions
                    if c in self._payloads]
        ramp = ramp_for(self._cmap) or CMAP_RAMPS[DEFAULT_CMAP]   # a deleted custom ramp
        aspect = DEPOSIT_LENGTH_UM / DEPOSIT_WIDTH_UM
        # THE COLOUR SCALE, decided over the whole grid BEFORE any of it is drawn
        # (`spatial_map_view.matrix_limits`): per population (sparse rule A, the
        # default), per row, per map, or one for all.
        row_limits, field = matrix_limits(
            {c: self._payloads[c] for c in self._conditions if c in self._payloads},
            self._scale)
        if payloads:
            aspect = payloads[0]["cell_length_um"] / payloads[0]["cell_width_um"]
        weight_unit = (payloads[0].get("weight_unit", "weighted jumps") if payloads
                       else "weighted jumps")
        # EACH COLUMN'S OWN UNIT: a map that carries one (SDA's µm/s, µm²/s, µm/s², the
        # location's probability) is painted and labelled in it, never as a density.
        column_unit, column_own = {}, {}
        for kind in self.kinds():
            first = next((p["panels"][kind] for p in payloads
                          if p.get("panels", {}).get(kind)), None)
            column_unit[kind] = panel_unit(first, field, weight_unit)
            column_own[kind] = bool(first and first.get("quantity") in ("value", "probability"))
        # n_eff tops follow the same colour scale (`spatial_map_view.neff_tops`).
        neff_top = neff_tops(
            {c: self._payloads[c] for c in self._conditions if c in self._payloads},
            self._scale)
        # WHERE THE BARS STAND (`spatial_map_view.bar_layout`): under the columns (per population, or one for
        # all), beside each row (per condition), under each map (per map) -- each bar where its maps are, with
        # its own numbers, so no map's range has to be printed under it any more.
        layout = bar_layout(self._scale, neff_top if self._rendering == "neff" else row_limits,
                            {k: column_unit[k] for k in self.kinds()} if any(column_own.values())
                            else None)
        self._bar_layout = layout
        note = matrix_bar_note(self._scale, self._row_unit, layout)
        bars_on = self._rendering != "tracks"

        for condition, row in self._rows.items():
            payload = self._payloads.get(condition)
            error = self._errors.get(condition)
            planned = self._planned.get(condition)
            for kind in self.kinds():
                panel = None if payload is None else payload["panels"].get(kind)
                # A COLUMN THIS ROW'S COUNT NEVER DRAWS: empty, and its line says so (`cell_drawn`)
                drawn = cell_drawn(self._method, kind, payload, planned, self._ranges)
                row["charts"][kind].set_absent(not drawn)
                if self._rendering == "tracks":
                    row["charts"][kind].set_panel(panel, paint_field(panel, field), None,
                                                  ramp, aspect)
                    row["charts"][kind].set_tracks(
                        panel.get("tracks", []) if panel else [],
                        panel.get("ink") if panel else None)
                elif self._rendering == "neff":
                    top = neff_top.get(condition, {}).get(kind, NEFF_SCALE_TOP)
                    row["charts"][kind].set_panel(neff_display_panel(panel, top), "neff_pos",
                                                  (0.0, 1.0), ramp, aspect)
                else:
                    row["charts"][kind].set_panel(
                        panel, paint_field(panel, field),
                        row_limits.get(condition, {}).get(kind), ramp, aspect,
                        self._rendering == "smooth")
                row["charts"][kind].set_clickable(
                    self._clickable and payload is not None and drawn)
                # Only a DAY row's map can say which tracks make a spot: its payload
                # carries the per-jump record (`spatial_map_view._spot_block`).
                row["charts"][kind].set_spot_clickable(
                    self._row_unit == "day" and payload is not None
                    and "spots" in payload and panel is not None
                    and not panel.get("sparse"))
                row["figures"][kind].setText(matrix_figures_text(
                    payload, panel, error, drawn=drawn, n_pop=row_n_pop(payload, planned), kind=kind))
                # THE BAR UNDER THIS MAP (per map): its own range, its own numbers.
                map_bar = self._map_bars[condition][kind]
                on = (layout == "maps" and bars_on and payload is not None and panel is not None)
                limits = row_limits.get(condition, {}).get(kind)
                if on and self._rendering == "neff":
                    _neff_bar(map_bar, payload, note,
                              neff_top.get(condition, {}).get(kind, NEFF_SCALE_TOP), ramp)
                elif on and limits:
                    lo, hi = limits
                    map_bar.set_scale(ramp, limits, column_unit[kind], note,
                                      ends=((bar_number(lo), f"{bar_number(hi)} {column_unit[kind]}")
                                            if column_own[kind] else None))
                else:
                    on = False
                map_bar.setProperty("has_scale", on)
                map_bar.setVisible(on and self._shown[kind])
            # THE BAR BESIDE THIS ROW (per condition): the one range its maps share.
            row_bar = self._row_bars[condition]
            on = layout == "rows" and bars_on and payload is not None
            shared = next((k for k in self.kinds()
                           if row_limits.get(condition, {}).get(k)), None)
            if on and self._rendering == "neff":
                tops = neff_top.get(condition, {})
                _neff_bar(row_bar, payload, note,
                          next(iter(tops.values()), NEFF_SCALE_TOP), ramp)
            elif on and shared is not None:
                row_bar.set_scale(ramp, row_limits[condition][shared], column_unit[shared], note)
            else:
                on = False
            row_bar.setVisible(on)

        for kind in self.kinds():
            bar = self._bars.get(kind)
            if bar is None:
                continue
            column = [lim[kind] for lim in row_limits.values() if lim.get(kind)]
            got = bool(column) and layout == "columns"
            if self._rendering == "neff":
                tops = {tops_row[kind] for tops_row in neff_top.values() if kind in tops_row}
                got = bool(tops) and layout == "columns"
                if got:
                    _neff_bar(bar, payloads[0], f"log scale · {note}",
                              None if len(tops) > 1 else tops.pop(), ramp)
            elif got:
                # Says what the one scale is pooled over: the conditions on
                # the section's grid, one condition's days on the day page. A map in a
                # unit of its own names it at the top end, its numbers never in
                # exponent notation (`bar_number`).
                lo, hi = column[0]
                bar.set_scale(ramp, column[0], column_unit[kind], note,
                              ends=((bar_number(lo), f"{bar_number(hi)} {column_unit[kind]}")
                                    if column_own[kind] else None))
            bar.setProperty("has_scale", got)
            bar.setVisible(got and self._shown[kind] and self._rendering != "tracks")

        self.caption_rows = matrix_caption_rows(
            payloads, self._cmap, self._rendering, unit=self._row_unit, scale=self._scale)
        self.caption_changed.emit()
        self._hint.setVisible(self._clickable and bool(payloads))
        self._neff_info.setVisible(self._rendering == "neff" and bool(payloads))
        self.export_button.setEnabled(bool(payloads))

        # The cliff, named per condition, and each condition's fit warnings (`spatial_map_view.map_warning_for`).
        warning = map_warning_for(self._payloads, self._conditions,
                                  unit=self._row_unit)
        self._warning.setText(warning or "")
        self._warning.setVisible(bool(warning))


def _scrolled(widget: QWidget) -> QScrollArea:
    """`widget` held at its NATURAL height inside a vertical scroll area -- the
    page shape both the section's grid and the replicate-day page use.

    Natural height, with the slack below: each capsule is ~3:1 and therefore
    WIDTH-limited on a wide page, so stretching the card would only draw a tall
    empty box under the last row.

    SCROLLED, and the row height is NOT the thing that gives way. This project
    runs up to seven conditions, and seven rows of a 3:1 capsule exceed any
    laptop window -- so either the maps shrink with the row count or the page
    scrolls. Shrinking would make the SAME map legible or not depending on how
    many neighbours it has, which is the one thing a comparison grid must not
    do. Horizontal scrolling stays OFF: the columns are what the eye compares
    across, and a comparison you have to scroll sideways to make is not one.
    """
    page = QWidget()
    page.setStyleSheet("background:transparent;")
    page_lay = QVBoxLayout(page)
    page_lay.setContentsMargins(0, 0, 0, 0)
    page_lay.setSpacing(0)
    page_lay.addWidget(widget)
    page_lay.addStretch(1)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QScrollArea.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    scroll.setStyleSheet("QScrollArea { background:transparent; }")
    scroll.setWidget(page)
    return scroll


class SpatialMapSection(QWidget):
    """The Spatial Map sidebar section: a placeholder until the first result,
    then the matrix, filling in as conditions land.

    The same idle -> content shape both existing sections use, and for the
    same reason -- `section_placeholder`'s own module docstring, "THE FIRST
    RUN AND EVERY RUN AFTER IT ARE TWO DIFFERENT MOMENTS". The placeholder is
    a one-way door: once a grid has been drawn, a later failure is reported on
    the grid rather than by returning to an empty tray.

    **It switches to the grid on the FIRST condition, not the last.** A run
    over four conditions is four separate computations, and watching three
    finished maps while the fourth is still running is a far better account of
    what is happening than a spinner that says nothing for the whole wait.

    The Run itself lives in `window.py`, this workspace's convention for every
    Run; `run_requested` is the signal that asks for it.
    """

    run_requested = Signal()
    cancel_requested = Signal()
    #: A condition's maps were clicked -- `window.py` opens its replicate days.
    condition_clicked = Signal(str)
    #: A spot's track was handed to a viewer -- `window.py` opens it.
    track_requested = Signal(str, str, str)
    #: A track hidden from the map, or shown again, on the day page: (condition, uid, hidden).
    track_hidden = Signal(str, str, bool)
    #: An Export: "grid" (every condition) or "days" (one condition's days).
    export_requested = Signal(str)
    #: The notice's Run (the maps' D is out of date): `window.py` runs the maps again at their settings.
    notice_run_requested = Signal()

    def __init__(self, t, parent=None, *, title: str = "Spatial Map", description: str | None = None):
        super().__init__(parent)
        self.t = t
        self.setStyleSheet("background:transparent;")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.placeholder = SectionPlaceholder(
            t, title=title,
            # TWO LINES. `SectionPlaceholder._description` is a fixed
            # `rem(24)` wide with a 44px minimum height, which is exactly two
            # lines -- a longer string is CLIPPED MID-SENTENCE rather than
            # wrapped, which is how the first draft of this read. The cost
            # ("not cheap") belongs on the Run dialog, which is the screen
            # where it can still be acted on.
            description=description or ("Where each population's jumps happen, in each cell's "
                                         "own frame, every condition side by side."),
            icon_name="grid")
        self.placeholder.run_requested.connect(self.run_requested.emit)
        self.placeholder.cancel_requested.connect(self.cancel_requested.emit)

        self.matrix = SpatialMapMatrix(t, clickable=True)
        self.matrix.condition_clicked.connect(self.condition_clicked.emit)
        self.matrix.export_requested.connect(lambda: self.export_requested.emit("grid"))
        # WHERE THE JUMPS SIT along and across the cell, every condition (asked 17 Oct 2026)
        from .spatial_profiles import ProfilesCard

        self.profiles = ProfilesCard(t, "Where the jumps sit · along and across the cell")
        self.profiles.export_requested.connect(lambda: self.export_requested.emit("profiles"))
        overview = QWidget()
        overview.setStyleSheet("background:transparent;")
        overview_lay = QVBoxLayout(overview)
        overview_lay.setContentsMargins(0, 0, 0, 0)
        overview_lay.setSpacing(CARD_GAP)
        overview_lay.addWidget(self.matrix)
        overview_lay.addWidget(self.profiles)
        self.profiles.hide()
        self._overview = _scrolled(overview)
        #: condition -> colour, for the profiles' lines (`begin_run`).
        self._colours: dict = {}
        #: The replicate page (`spot_page.SpotPage`) -- a condition's maps, one row per
        #: replicate, and what makes a clicked spot. The redesign tried behind the rail's Test
        #: button since 13 Oct 2026; the only replicate page since 23 Nov 2026.
        from .spot_page import SpotPage

        self.days = SpotPage(t)
        self.days.track_requested.connect(self.track_requested.emit)
        self.days.hidden_changed.connect(self.track_hidden.emit)
        self.days.matrix.export_requested.connect(lambda: self.export_requested.emit("days"))
        self.days.profiles.export_requested.connect(lambda: self.export_requested.emit("day_profiles"))

        self._pages = QStackedWidget()
        self._pages.addWidget(self.placeholder)
        self._pages.addWidget(self._overview)
        self._pages.addWidget(self.days)
        # THE MAPS' D IS OUT OF DATE (4 Nov 2026, `map_fit_source.stale_text`): the section they took D from has
        # been run again with other settings. Said above the grid with a Run, never redrawn in silence.
        self.notice_row = QWidget()
        self.notice_row.setStyleSheet("background:transparent;")
        n = QHBoxLayout(self.notice_row)
        n.setContentsMargins(0, 0, 0, rem(0.5))
        n.setSpacing(10)
        self.notice_text = fit_label("", ty.ui("sm", "medium"), _WARN_COLOR)
        n.addWidget(self.notice_text, 1)
        self.notice_run = Button(t, "Run", variant="outline", size="sm")
        self.notice_run.clicked.connect(self.notice_run_requested.emit)
        n.addWidget(self.notice_run, 0, Qt.AlignVCenter)
        outer.addWidget(self.notice_row)
        self.notice_row.hide()
        outer.addWidget(self._pages)

        #: Flips True on the first drawn grid and never back -- the
        #: placeholder is a ONE-WAY DOOR (`section_placeholder`'s own module
        #: docstring), so a RE-run empties the rows in place rather than
        #: sending the page back to a tray that says nothing has been
        #: computed. Both other sections take the same line.
        self._ever_shown = False

    # -- the run ------------------------------------------------------------ #
    def begin_run(self, conditions, labels: dict | None = None,
                  method: str = "sqd", n_pop: int = 2, colours: dict | None = None,
                  ranges=None, counts=None, planned=None) -> None:
        """A new Run over CONDITIONS. Drops every previous result. ``planned``: each condition's count
        (`spatial_map_view.planned_n_pops`), so a column a row's count never draws says so from the start.

        Dropping them is what lets the grid show ONE bar per column:
        `_panel_limits` will not pool panels computed at different settings,
        so keeping a row from an earlier configuration would leave the bar
        describing some of the grid and not the rest. Re-running is the only
        way to change a setting, so nothing is lost that cannot be recomputed
        by the press that discarded it.

        `method` BEFORE `set_conditions`: `SpatialMapMatrix.set_method`
        drops the matrix's memoised condition list when the method changes,
        which is what makes the very next `set_conditions` call rebuild the
        grid's column set instead of early-returning on an unchanged
        condition list.
        """
        self.matrix.set_method(method, n_pop, ranges, counts, planned)
        self.matrix.set_conditions(conditions, labels)
        self.matrix.clear_payloads()
        self._colours = dict(colours or {})
        self.profiles.hide()
        self.placeholder.set_running(True)
        # Back to the grid (or the placeholder before the first result): a day
        # page from the previous Run describes maps this Run is replacing.
        self.show_overview()

    def set_running(self, on: bool) -> None:
        self.placeholder.set_running(on)

    def set_notice(self, text: str | None) -> None:
        """The line above the grid with its Run (`map_fit_source.stale_text`); None or "" hides it."""
        self.notice_text.setText(text or "")
        self.notice_row.setVisible(bool(text))

    def notice_shown(self) -> str:
        """The notice's text while it shows, else ""."""
        return self.notice_text.text() if self.notice_row.isVisibleTo(self) else ""

    def set_payload(self, condition: str, payload: dict) -> None:
        self.matrix.set_payload(condition, payload)
        self._ever_shown = True
        self._refresh_profiles()
        self._leave_placeholder()

    def _refresh_profiles(self) -> None:
        """The overview's Profiles card from every condition landed so far, on the population it shows."""
        method = self.matrix._method
        if method not in PROFILE_METHODS:
            self.profiles.hide()
            return
        payloads, conditions, labels = self.matrix.payloads(), list(self.matrix._conditions), dict(self.matrix._labels)
        self.profiles.set_source(lambda k: profile_card_payload(payloads, conditions, labels, k),
                                 self.matrix.kinds(), self.matrix.panel_titles(), self._colours,
                                 select=self.profiles.marked_kind())

    def redrawn(self, condition: str, payload: dict) -> None:
        """One condition's map again, after a track was hidden or shown: the grid's row, and its days where they
        are open."""
        self.matrix.set_payload(condition, payload)
        self.days.refresh_condition(condition, payload)
        self._refresh_profiles()

    def set_error(self, condition: str, message: str) -> None:
        """Report the failure IN THE GRID once there is a grid to report it
        in; before that there is nothing on screen to annotate, so the
        placeholder carries it."""
        self.matrix.set_error(condition, message)
        if self.matrix.has_payloads():
            self._leave_placeholder()

    def _leave_placeholder(self) -> None:
        """Show the grid when a result lands -- but ONLY from the placeholder.
        A row becomes clickable the moment it lands, so somebody can be
        reading one condition's days while the next condition is still
        computing; that arrival must not pull them back to the grid."""
        if self._pages.currentWidget() is self.placeholder:
            self._pages.setCurrentWidget(self._overview)

    def has_result(self) -> bool:
        return self.matrix.has_payloads()

    # -- the replicate-day page ------------------------------------------- #
    def show_days(self, condition: str) -> None:
        """Open `condition`'s replicate days, drawn the way the grid is drawn
        right now: its colormap, rendering and hidden columns are carried over
        BEFORE the page is shown, so the two never disagree on screen."""
        payload = self.matrix.payloads()[condition]
        page = self.days
        # The method FIRST: switching the replicate grid's method puts it in that method's own
        # ramp, which would otherwise undo the ramp carried over just below.
        page.matrix.set_method(payload["method"], payload.get("n_pop") or 2, payload.get("ranges"))
        page.set_colormap(self.matrix.colormap())
        page.set_rendering(self.matrix.rendering())
        page.set_scale(self.matrix.scale())
        page.show_condition(condition, payload, name=self.matrix._labels.get(condition, condition),
                            colour=self._colours.get(condition),
                            kind=getattr(self.matrix, "last_kind", None))
        for kind, on in self.matrix.panel_states().items():
            page.set_panel_visible(kind, on)
        self._pages.setCurrentWidget(page)

    def show_track_message(self, text: str) -> None:
        """What became of a viewer request or a redraw, said on the replicate page."""
        self.days.show_track_message(text)

    def show_overview(self) -> None:
        """The grid -- or the placeholder, while no Run has ever drawn one
        (the one-way door: once the grid has been shown it is never swapped
        back for the placeholder)."""
        self._pages.setCurrentWidget(
            self._overview if self._ever_shown else self.placeholder)

    def showing_days(self) -> str | None:
        """The condition whose days are on screen, or None."""
        current = self._pages.currentWidget()
        return current.condition() if current is self.days else None

    def drillable_conditions(self) -> list[str]:
        """The conditions that have maps, in row order -- the ones whose days
        can be opened (the breadcrumb dropdown's options)."""
        drawn = self.matrix.payloads()
        return [c for c in self.matrix.conditions() if c in drawn]

    # -- the Visibility tool ------------------------------------------------ #
    def method(self) -> str:
        return self.matrix._method

    def visibility_states(self) -> dict:
        """Keyed by DISPLAY TITLE, which is what `VisibilityBody.set_section`'s
        `states` expects."""
        states = self.matrix.panel_states()
        return {title: states[kind] for title, kind in self.matrix.panel_keys().items()}

    # Every display setting reaches BOTH pages: the Visibility tool describes
    # the section, and the day page is part of it.
    def set_panel_visible(self, key: str, on: bool) -> None:
        kind = self.matrix.panel_keys()[key]
        self.matrix.set_panel_visible(kind, on)
        self.days.set_panel_visible(kind, on)

    def set_colormap(self, name: str) -> None:
        self.matrix.set_colormap(name)
        self.days.set_colormap(name)

    def colormap(self) -> str:
        return self.matrix.colormap()

    def set_rendering(self, mode: str) -> None:
        self.matrix.set_rendering(mode)
        self.days.set_rendering(mode)

    def set_scale(self, level: str) -> None:
        self.matrix.set_scale(level)
        self.days.set_scale(level)

    def scale(self) -> str:
        return self.matrix.scale()

    def refresh_colours(self) -> None:
        """Repaint both pages: a custom ramp on screen was edited under its own key."""
        self.matrix.redraw()
        self.days.matrix.redraw()

    def rendering(self) -> str:
        return self.matrix.rendering()
