"""Publication figures from the workspace's own results -- ready to submit, not to fix.

Asked for: an exporter for manuscripts AND slides, "least amount of work in inkscape,
figures should be ready to publish", single panels and a multi-panel composer, and a few
important fonts. PHASE 1 is this foundation and the SQD bubble chart; the other charts
and the composer build on it.

**THE SAME NUMBERS, A PRINT RENDERER.** The workspace paints its charts with Qt for the
screen. A figure is drawn here with matplotlib from the SAME plain payload the screen
chart is given (`BubblePlotData`), with the same text and axis rules (`chart_text`), so
the two may differ in look -- print is sized in millimetres and points -- but never in a
number.

**EXACT SIZE.** A journal asks for a column width in millimetres: the figure is laid out
at that size (`layout="constrained"`) and saved WITHOUT ``bbox_inches="tight"``, which
would trim it to some other size the journal then rescales -- and with it every font.

**TEXT STAYS TEXT.** PDF embeds the font as TrueType (``pdf.fonttype`` 42) and SVG keeps
``<text>`` (``svg.fonttype`` "none"), so a label is still a label in Illustrator or
Inkscape.

**PRESETS** are `figstyle.JOURNAL_PRESETS`, checked against the journals' own guides (see
there): a width is a FRACTION of the journal's full width (1/4, 1/3, 1/2, 2/3, 3/4, Full; `WIDTHS`, 3 Nov 2026), the
journal's stated width where it states one and the full width divided where it does not (Nature's 2/3, never
passed off as a column width it defines), and each journal's own smallest text and largest height are what the
checks hold it to.

**TEXT IS EDITABLE** (asked for: "custom field for axis titles and condition names"): the
axis titles and each condition's displayed name are set in the dialog, defaulting to the
screen's own (`chart_text.D_AXIS_LABEL`). A display name changes the FIGURE only; the
numbers written beside it keep the real condition names.

**FONTS** (`FONTS`): Arial, the journals' default; Helvetica; Inter, the app's own (bundled).
A font that is not installed is substituted -- and the substitution is REPORTED, never
silent, since a figure set in a fallback face is the kind of thing found at proof stage.

**CHECKS BEFORE EXPORT** (`check_figure`): text under the journals' 5 pt minimum, lines
under 0.25 pt, text overlapping text, text running off the figure. Reported, not refused:
the experimenter decides.

**WHAT TRAVELS WITH A FIGURE** (`write_figure`):
* **SOURCE DATA** -- ``<name>.source_data.xlsx`` (asked for, "genius": Nature requires
  source data for every graph) and the same table as ``<name>.csv``: one row per plotted
  point, as NUMBERS (D, the fraction as 0-1, what the spread is and its values, the
  replicate count), under a header naming the figure, the units and what the spread
  means. `write_source_workbook` takes several sheets, so a composed figure gets one sheet
  per panel;
* a DRAFTED legend in words (`bubble_legend` -- what the area, the height and the spread
  mean, and that D is apparent);
* the analysis provenance, and ``<name>.figure.json`` -- the export settings, the font
  actually used, the checks' findings and the files written.
"""
from __future__ import annotations

from .chart_text import unit_word  # the word shown for a grouping unit

import csv
import io
import json
import math
import re
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from fractions import Fraction
from pathlib import Path

from . import chart_text, figstyle
from .condition_colours import PAGE_COLOURS
from .population_style import REFERENCE_DASH, population_styles

__all__ = [
    "DPIS",
    "FONTS",
    "FORMATS",
    "KINDS",
    "MIN_FONT_PT",
    "MIN_LINE_PT",
    "PRESETS",
    "PRESET_LABELS",
    "WIDTHS",
    "WIDTH_LABELS",
    "WIDTH_FRACTIONS",
    "PANEL_WIDTHS",
    "PANEL_WIDTH_LABELS",
    "width_fraction",
    "fraction_mm",
    "compose_places",
    "Built",
    "FigureExportError",
    "FigureSpec",
    "build_figure",
    "bubble_legend",
    "appd_bubble_legend",
    "appd_histogram_legend",
    "appd_cells_legend",
    "appd_checks_legend",
    "kinetics_legend",
    "populations_legend",
    "mock_legend",
    "hmm_legend",
    "hmm_comparison_legend",
    "hmm_angles_legend",
    "hmm_overview_legend",
    "jumps_legend",
    "fit_choices",
    "fit_pressed",
    "show_checked",
    "shown_y_mode",
    "jumps_y_title",
    "y_mode_choices",
    "grid_choices",
    "PICK_KINDS",
    "shown_fit",
    "qc_legend",
    "spatial_legend",
    "cell_image_legend",
    "column_names",
    "default_titles",
    "auto_title",
    "with_title",
    "show_options",
    "scale_bar_auto",
    "parse_length",
    "msd_legend",
    "replicate_legend",
    "write_source_workbook",
    "check_figure",
    "preview_png",
    "resolve_font",
    "with_changes",
    "write_figure",
]

PRESETS = ("nature", "cell", "elife", "slides", "slides_dark", "poster")
PRESET_LABELS = {"nature": "Nature", "cell": "Cell", "elife": "eLife", "slides": "Slides",
                 "slides_dark": "Dark slides", "poster": "Poster"}
#: Presets drawn for a screen or a wall rather than a page: bigger gutters, 16:9-ish panels.
LARGE_PRESETS = ("slides", "slides_dark", "poster")
#: A figure's WIDTH, named as a FRACTION OF THE JOURNAL'S FULL WIDTH (3 Nov 2026, asked: "1 seems to be 1/2 of the full
#: width of the page. 1.5 seems to be 2/3, and 2 full width ... is 1 1.5 and 2 even good naming?", "why not 1/3, 1/2,
#: 2/3, 1", "all scaling to the width of one full figure"): it was "1 column / 1.5 columns / 2 columns", the
#: journals' words, while the Composer's "1" was one cell of its grid, so one number named two widths. The keys are
#: unchanged (a remembered figure reads as before). 1/2 and Full are the journal's own one and two columns, 2/3 its
#: 1.5-column width where it states one (Cell's 114 mm); every other fraction is the full width divided with the
#: Composer's gutter (`fraction_mm`), so a chart exported at 2/3 is the width a 2/3 panel is drawn in the Composer.
#: 1/4 and 3/4 since 5 Nov 2026 (asked: "composer has 1/4 but export does not", "same with 3/4"): the Export dialog
#: offers every width the Composer does (`PANEL_WIDTHS`).
WIDTHS = ("quarter", "third", "single", "onehalf", "threequarter", "double", "custom")
WIDTH_LABELS = {"quarter": "1/4", "third": "1/3", "single": "1/2", "onehalf": "2/3", "threequarter": "3/4",
                "double": "Full", "custom": "Custom"}
#: Each named width's fraction of the full width (`PANEL_WIDTHS`' keys).
WIDTH_FRACTIONS = {"quarter": "1/4", "third": "1/3", "single": "1/2", "onehalf": "2/3", "threequarter": "3/4",
                   "double": "1"}
FONTS = ("Arial", "Helvetica", "Inter")
FORMATS = ("pdf", "svg", "png", "tiff")
DPIS = (300, 600)
#: The journals' floor for any text at final size (Nature and Cell both say 5 pt).
MIN_FONT_PT = 5.0
#: Thinner than this disappears in print.
MIN_LINE_PT = 0.25
_MM = 25.4
_INTER = Path(__file__).resolve().parent / "ui" / "theme" / "fonts" / "Inter[opsz,wght].ttf"
#: When a font is missing, what stands in, in order -- metric look-alikes first.
_STAND_INS = {"Arial": ("Liberation Sans", "Helvetica", "DejaVu Sans"),
              "Helvetica": ("Arial", "Liberation Sans", "DejaVu Sans"),
              "Inter": ("Arial", "Helvetica", "DejaVu Sans")}


class FigureExportError(ValueError):
    """A figure setting that cannot be honoured, with what to change."""


@dataclass(frozen=True)
class FigureSpec:
    """How one figure is exported. Every field is printed into ``.figure.json``."""

    preset: str = "nature"
    width: str = "single"
    #: Used when ``width`` is "custom".
    width_mm: float | None = None
    #: None: the chart's own proportion at this width.
    height_mm: float | None = None
    font: str = "Arial"
    #: None: the preset's size (7 pt for Nature and Cell, 16 pt for slides).
    font_pt: float | None = None
    formats: tuple = ("pdf", "png")
    dpi: int = 600
    #: The fraction printed on (or below) each bubble.
    numbers: bool = True
    #: THE SAMPLE under each column of a bubble chart, "3 replicates · 1,271 tracks" (asked 30 Oct 2026: '"n" ON THE
    #: SQD BUBBLE PLOTS ... optionable to toggle off if people just write in text/legend/caption'): on by default; the
    #: Export dialog's Show box "n" (`show_options`), kept by a composed panel (`PANEL_WORDS`) and remembered as a
    #: style. Its numbers are the fit's payload's (`bubble_view.sample_marks`).
    sample_n: bool = True
    grid: bool = True
    legend: bool = True
    #: The cell image's scale bar, the bar and its length together (and the Spatial Map's).
    scale_bar: bool = True
    #: The Spatial Map's LABELS: the population titles over its columns and the names beside its rows (asked
    #: 16 Oct 2026: "a disable option for the scale bar, color map etc"), for a grid composed under labels of
    #: its own.
    headings: bool = True
    #: Its length in µm, typed; None picks a round length near an eighth of what is shown
    #: (`images_view.scale_bar_um`), which follows the zoom.
    scale_bar_um: float | None = None
    #: The axis titles; None is the chart's own (`default_titles`, the screen's), ""
    #: none at all.
    y_title: str | None = None
    x_title: str | None = None
    #: ``((condition, name shown), ...)``: the figure's own names for conditions.
    names: tuple = ()
    #: SIGNIFICANCE BRACKETS (ruled 13 Oct 2026): OFF unless this figure asks. Welch's t on
    #: the PER-DAY values of the two conditions (never paired by day number, never per cell
    #: or track), the exact p adjusted over the pairs picked (`significance`; Holm unless
    #: ``significance_correction`` says otherwise).
    significance: bool = False
    #: ``((condition A, condition B), ...)``: the pairs picked in the dialog, by their REAL
    #: names; a bracket per pair, the correction over them. They belong to THIS chart.
    significance_pairs: tuple = ()
    #: Whose fraction is compared: 1 is the slowest population; "all" compares every one.
    significance_population: object = 1
    #: Which test (`significance.TESTS`): "welch" (per-replicate values, the standard), "mixed" (each cell's
    #: fraction in a nested mixed model with a parametric-bootstrap p; asked for 14 Oct 2026), or since 24 Oct
    #: 2026 "tost" (equivalence within ``significance_margin``), "bayes", "meta" or "hboot" (`nested`).
    significance_test: str = "welch"
    #: What a bracket says (`significance.LABELS`): stars (the default since 14 Oct 2026,
    #: asked for), the exact p, or both; and the star thresholds on the adjusted p.
    significance_label: str = "stars"
    significance_stars: tuple = (0.05, 0.01, 0.001)
    #: The equivalence test's margin, percentage points of the fraction (``significance_test="tost"``).
    significance_margin: float = 5.0
    #: The correction over the brackets shown (`significance.CORRECTIONS`): "holm", the standard and what every
    #: figure made before 10 Nov 2026 used, or "bh" (Benjamini-Hochberg). Named in the legend and the sheets.
    significance_correction: str = "holm"
    #: A title over the figure (asked for: "change x/y axis label, title ..."), CENTRED over
    #: the plot. None is AUTOMATIC (`auto_title`: the project's name, and the condition's for
    #: a chart of one condition -- asked for 13 Oct 2026: "maybe it tries to automatically
    #: add a title as well (manually editable)"); "" is none; anything else is typed.
    title: str | None = None
    #: Where each axis starts and ends ("... and x y axis SCALE, where it starts and
    #: ends"); None is the chart's own. Only the axes `axis_ranges(kind)` names are set.
    y_min: float | None = None
    y_max: float | None = None
    x_min: float | None = None
    x_max: float | None = None
    #: The jump-distance figures' FITS DRAWN (30 Oct 2026: a card's Show is a set of toggles): a tuple of population
    #: counts -- one alone, with its components; two or more, each in its own colour -- or None to follow the card
    #: (the fits its Show has on, else the one BIC prefers: `_chosen`). The one-fit figures (apparent-D, Signed
    #: jumps) take an int. Read from a figure remembered before: an int ``n`` on a jump figure draws ``(n,)``, and
    #: ``"all"`` (`FIT_ALL`, the old Show's last pill) every fit made.
    chosen_fit: int | tuple | str | None = None
    #: The jump-distance figures' two residual plots, each OPTIONAL (asked 30 Oct 2026: "also make it optional to
    #: export ecdf and residuals in export window and in composer", and later the same day for every jump export: "do
    #: we have ecdf and residual export function? maybe we include it in exporter for every jump distance as an
    #: option? even for each/all condition etc windows"): the Pearson residuals and the ECDF residuals. None is the
    #: FIGURE'S OWN DEFAULT (`RESIDUALS_BY_DEFAULT`: on for a card's figure, as before; off for Export all, whose grid
    #: gets tall with two strips under every plot), True / False the Export dialog's box. Off, the plot is not drawn,
    #: its numbers leave the source data, and the plots left take its room (`_jumps_height`). Offered on both jump
    #: figures (`show_options`); a composed panel keeps them (`PANEL_WORDS`); per chart, never carried to the next
    #: (`remembered_record`).
    pearson_residuals: bool | None = None
    ecdf_residuals: bool | None = None
    #: The jump-distance figures' COMPONENTS (asked 30 Oct 2026: "its a shame that the component lines are not
    #: visible anymore when i do 1 and 2 pop fit together. maybe we can make this optional"): None follows the card's
    #: Components box (each card's, on the overview; on when a payload predates it), True / False the Export dialog's
    #: box for every panel (`_components_on`). Whose are drawn is `jumps_view.components_drawn`.
    components: bool | None = None
    #: The jump-distance figures' Y UNIT (30 Oct 2026, asked: "lets make all optional with textbook (check first) as
    #: standard"): None follows the card (its entry's ``"y_mode"``; an entry, a Composer panel or a remembered figure
    #: from before the choice existed reads as the standard, Probability), else "probability" (per bin, SMTracker 2's
    #: and Spot-On's), "counts" (the jumps per bin) or "density" (per nm, area 1) for every panel
    #: (`jumps_view.Y_MODES`). The Export dialog's Y unit row; a change there drops a typed y range.
    y_mode: str | None = None
    #: The HMM state diagrams' GRID (5 Nov 2026, asked: "Option to order in x times x grid ?"): None, as many
    #: columns as the width holds (`_hmm_layout`), else that many columns, the rows following. The Export
    #: dialog's Grid row (`grid_choices`); a composed panel keeps it (`PANEL_WORDS`); per figure, never carried to
    #: the next chart (`remembered_record`).
    grid_columns: int | None = None
    #: The conditions an "every condition" figure LEAVES OUT (5 Nov 2026, asked: "first i think we need to select which
    #: conditions to take, for example if i just wanna show two of them"): their real names, ticked off beside each
    #: name in the Export dialog (`PICK_KINDS`). Empty: every condition the payload holds. Per figure (the section's
    #: eyes give its start), kept by a composed panel (`PANEL_WORDS`).
    left_out: tuple = ()
    #: The HMM diagrams' rates WITH A ± (5 Nov 2026, asked: "dont show the [ ] in the picture or make it optional.
    #: better a + / - but make it optional as well. this is something to be printed in data table"): off, each rate
    #: alone; on, "7.7 ± 1.2 /s" -- half the 95 % interval over resampled cells or movies, or the SD over the
    #: replicates' own fits (`_hmm_rate_spread`), the key saying which. The interval itself is in the source data
    #: either way. The Export dialog's Show box "± on rates", offered where a spread was computed.
    rate_spread: bool = False
    #: The HMM diagrams' occupancies WITH A ± (5 Nov 2026, asked: "no option to show and hide the +- of the replicates
    #: of HMM population fractions"): off, "38.0%" under a state; on, "38.0 ± 4.1%" -- the SD of the replicates' own
    #: occupancies, or half the 95 % interval over resampled cells or movies (`_hmm_occupancy_spread`), the key saying
    #: which. The Show box "± on occupancy", offered where a spread was computed; the source data hold it either way.
    occupancy_spread: bool = False
    #: Populations over time (asked 8 Nov 2026 for the kinetics figure: "remove first row", "share -> share of slow
    #: fraction?"): the slowest population's D panel (off: the D panels start at the next population, the slowest's
    #: D sitting at the localisation floor), and the other populations' shares (off: the share panel draws the
    #: slowest's share alone, titled "slow share (%)"). On, both, as before; offered with two populations or more.
    slowest_d: bool = True
    other_shares: bool = True
    #: The half-time on Populations over time (8 Nov 2026): a dotted line where the change is half made, on the panels
    #: that changed, and "t1/2 = 3.3-3.4 min" (with the reference's, where one is drawn). A result made before carries
    #: no half-time, so its figure draws as it did.
    half_time: bool = True
    #: The every-condition populations figure with the CONDITIONS AS ROWS (8 Nov 2026, as Fig. 6B/C): a row per
    #: condition, a column per quantity drawn (each population's D, then the share), the condition's name bold in its
    #: colour over its panels, every panel its y title. Off: a column per condition, as before.
    conditions_as_rows: bool = False
    #: The HMM comparison's PAIR LINES (5 Nov 2026, asked: "lines joining paired replicates", off by default): a thin
    #: line through the open circles of replicates declared as pairs on the Groups board (`hmm_view.pair_lines`), the
    #: key and legend saying what it is. The Show box "Pair lines", offered where two drawn conditions share a tag.
    pair_lines: bool = False
    #: The HMM comparison's DWELL PANEL (12 Nov 2026, asked: "can we get for hmm a dwell time and rate +- error. also
    #: for export option to export dwell time as well"): a fifth panel, the slow state's model dwell in ms, after the
    #: rate out it comes from (`hmm_view.DWELL_PANEL`). The Show box "Dwell time", starting as the strip shows it.
    hmm_dwell: bool = False
    #: The Spatial Map's CELLS ON ONE SCALE (5 Nov 2026, asked: "when exporting spatial maps i feel the cells get
    #: downscaled differently depending on the text label size and other stuff? its hard to compare then when cells get
    #: scaled differently", then "maybe the cells should have just always the same length and width depending on the
    #: fit settings, and the text etc gets scaled around it", and, for maps too wide: "keep height and width of total
    #: image so we can put them next to each other in composer", "figure height preserve as well"): mm on paper per µm
    #: of cell. None: the standard, `map_scale_standard` (6 at an 89 mm column); typed, that. Every capsule is drawn at it whatever the names
    #: or the colour bars, the grid centred in the width chosen. Where the columns do not fit that width at the scale,
    #: the capsules are drawn as large as it holds and each row KEEPS the height it has at the scale, so the figure is
    #: the same size either way (`_spatial_layout`); the check line says the scale drawn. A set height never shrinks
    #: them. Remembered for the next figure, kept by a composed panel.
    map_scale: float | None = None

    def problem(self) -> str:
        """Why this spec cannot be exported, or ""."""
        if self.preset not in PRESETS:
            return f"unknown preset {self.preset!r}"
        if self.width not in WIDTHS:
            return f"unknown width {self.width!r}"
        if self.width == "custom" and not (self.width_mm and 20 <= self.width_mm <= 600):
            return "a custom width needs a value between 20 and 600 mm"
        if self.height_mm is not None and not 20 <= self.height_mm <= 600:
            return "the height must be between 20 and 600 mm"
        if self.font not in FONTS:
            return f"unknown font {self.font!r}"
        if self.font_pt is not None and not 4 <= self.font_pt <= 40:
            return "the font size must be between 4 and 40 pt"
        if not self.formats or any(f not in FORMATS for f in self.formats):
            return "choose at least one of PDF, SVG, PNG, TIFF"
        if self.dpi not in DPIS:
            return f"the resolution must be one of {DPIS} dpi"
        if self.scale_bar_um is not None and not 0 < self.scale_bar_um <= 1000:
            return "the scale bar needs a length above 0 µm"
        if self.map_scale is not None and (isinstance(self.map_scale, bool)
                                           or not isinstance(self.map_scale, (int, float))
                                           or not 0 < self.map_scale <= 200):
            return "the map scale needs a value above 0 mm per µm (200 at most)"
        significance = _significance_problem(self)
        if significance:
            return significance
        for axis, lo, hi in (("y", self.y_min, self.y_max), ("x", self.x_min, self.x_max)):
            if lo is not None and hi is not None and lo >= hi:
                return f"the {axis} axis must start below where it ends ({lo:g} is not below {hi:g})"
        if not _fit_value_ok(self.chosen_fit):
            return "the chosen fit must be 1, 2, 3 or 4 populations (one, or several of them), or all"
        if self.components not in (None, True, False):
            return "components must be shown, hidden, or as the card shows them"
        if self.pearson_residuals not in (None, True, False) or self.ecdf_residuals not in (None, True, False):
            return "a residual plot must be shown, hidden, or as the figure draws it by default"
        if self.y_mode is not None and self.y_mode not in ("probability", "counts", "density"):
            return "the y axis is probability, counts or density"
        if self.grid_columns is not None and (isinstance(self.grid_columns, bool)
                                              or not isinstance(self.grid_columns, int)
                                              or not 1 <= self.grid_columns <= GRID_MAX_COLUMNS):
            return f"the grid needs 1 to {GRID_MAX_COLUMNS} columns, or Auto"
        if not isinstance(self.left_out, tuple) or not all(isinstance(v, str) for v in self.left_out):
            return "the conditions left out must be names"
        return ""

    @property
    def preset_values(self) -> dict:
        return figstyle.JOURNAL_PRESETS[self.preset]

    def width_value(self) -> float:
        """The width in mm: a custom one as typed; a fraction the journal states (one column, two, Cell's 1.5) as it
        states it; any other fraction (1/3 everywhere, 2/3 where the journal states no 1.5 columns: Nature, eLife)
        its share of the full width with the Composer's gutter (`fraction_mm`), rounded to 0.1 mm."""
        if self.width == "custom":
            return float(self.width_mm)
        stated = self.preset_values.get(f"{self.width}_mm")
        if stated is not None:
            return float(stated)
        return round(fraction_mm(WIDTH_FRACTIONS[self.width], float(self.preset_values["double_mm"]),
                                 compose_gutters(self)[0]), 1)

    def pt(self) -> float:
        return float(self.font_pt if self.font_pt is not None
                     else self.preset_values["font_pt"])

    def min_pt(self) -> float:
        """The smallest text this journal accepts (Nature 5, Cell 6, eLife 7 pt)."""
        return float(self.preset_values.get("min_font_pt") or MIN_FONT_PT)

    def small_pt(self) -> float:
        """Tick labels, the names under columns, the sample and the numbers on bubbles: a point under the body size.
        It FOLLOWS a typed size (1 Nov 2026, asked: "scaling down text doesnt scale down the text of the condition
        name and replicate = etc, and the fraction sizes"): it was held at the journal's minimum, so a Cell figure set
        at 5 pt kept its names, its sample and its numbers at 6 pt, larger than the body. Text under the journal's
        minimum is then the user's choice and `check_figure` names it; every journal's own size stays at or over it
        (Cell 7 pt, these at 6)."""
        return self.pt() - 1.0

    def shown_name(self, condition) -> str:
        """The condition's name in this figure."""
        for real, shown in self.names:
            if real == condition and str(shown).strip():
                return str(shown)
        return str(condition)

    def line_pt(self) -> float:
        return float(self.preset_values["line_pt"])

    @property
    def dark(self) -> bool:
        """Drawn on a dark page (the dark-slide preset): every colour comes from `_page`."""
        return bool(self.preset_values.get("dark"))

    def default_height_mm(self) -> float:
        width = self.width_value()
        if self.preset in ("slides", "slides_dark"):
            return round(width * 9 / 16, 1)
        return round(max(55.0, self.preset_values["single_mm"] * 0.72), 1)

    def size_mm(self) -> tuple:
        height = self.height_mm if self.height_mm is not None else self.default_height_mm()
        return self.width_value(), float(height)

    def record(self) -> dict:
        """The spec as plain JSON-ready values (``.figure.json``)."""
        data = asdict(self)
        data["formats"] = list(self.formats)
        data["names"] = [list(pair) for pair in self.names]
        data["significance_pairs"] = [list(pair) for pair in self.significance_pairs]
        data["significance_stars"] = [float(t) for t in self.significance_stars]
        data["left_out"] = list(self.left_out)
        if isinstance(self.chosen_fit, tuple):
            data["chosen_fit"] = [int(k) for k in self.chosen_fit]
        # "" has meant "no title" only since titles became automatic: a record without this
        # mark was written when "" was everybody's default (`from_dict`).
        data["title_rule"] = TITLE_RULE
        # the residual plots' None (the figure's own default) since 30 Oct 2026: a record without this mark held
        # the old default, True, for every chart (`from_dict`)
        data["residual_rule"] = RESIDUAL_RULE
        return data

    def remembered_record(self) -> dict:
        """What is remembered for the next figure: the style, not the words. Condition
        names belong to this project and axis titles to this KIND of chart (an MSD figure
        must not open with the SQD chart's "D" title)."""
        data = self.record()
        # The scale bar's length belongs to the VIEW it was typed for (a whole field
        # against one zoomed cell), like the titles to their chart: never carried over.
        # So do the title and where each axis starts and ends, and which fit a jump-distance
        # figure draws (one condition's choice, not the next chart's).
        # The residual plots too (30 Oct 2026): each jump figure has its own default (`RESIDUALS_BY_DEFAULT`), so a
        # card's ticked boxes must not switch them on in the next Export all.
        for words in ("names", "y_title", "x_title", "scale_bar_um", "title", "y_min",
                      "y_max", "x_min", "x_max", "chosen_fit", "components", "y_mode",
                      "pearson_residuals", "ecdf_residuals", "grid_columns", "left_out"):
            data.pop(words)
        # Significance is asked for PER FIGURE: its pairs name this chart's conditions, and a
        # next chart starts with it off (ruled 13 Oct 2026).
        for words in SIGNIFICANCE_WORDS:
            data.pop(words)
        return data

    @classmethod
    def from_dict(cls, data) -> "FigureSpec":
        """A remembered spec; unknown or unusable fields fall back to the defaults."""
        if not isinstance(data, dict):
            return cls()
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        if known.get("title") == "" and data.get("title_rule") != TITLE_RULE:
            # Written before titles were automatic, when "" was the default nobody chose:
            # it starts automatic, like a chart never exported.
            known["title"] = None
        if data.get("residual_rule") != RESIDUAL_RULE:
            # Written before the residual plots had a default per figure (30 Oct 2026), when every record held the old
            # default, True -- a chart's choice or not: read as the figure's own default (`RESIDUALS_BY_DEFAULT`), so
            # an Export all exported before does not start with strips it never had.
            known.pop("pearson_residuals", None)
            known.pop("ecdf_residuals", None)
        if "formats" in known:
            known["formats"] = tuple(known["formats"] or ())
        if isinstance(known.get("chosen_fit"), list):
            # a set of fits travels as a JSON list; an int or "all" (written before 30 Oct 2026) is read as it was
            try:
                known["chosen_fit"] = tuple(int(k) for k in known["chosen_fit"])
            except (TypeError, ValueError):
                known.pop("chosen_fit")
        if "names" in known:
            known["names"] = tuple(tuple(pair) for pair in known["names"] or ()
                                   if isinstance(pair, (list, tuple)) and len(pair) == 2)
        if "significance_stars" in known:
            try:
                known["significance_stars"] = tuple(float(t) for t in
                                                    known["significance_stars"] or ())
            except (TypeError, ValueError):
                known.pop("significance_stars")
        if "left_out" in known:
            known["left_out"] = tuple(str(v) for v in known["left_out"] or () if isinstance(v, str))
        if "significance_pairs" in known:
            known["significance_pairs"] = tuple(
                (str(pair[0]), str(pair[1])) for pair in known["significance_pairs"] or ()
                if isinstance(pair, (list, tuple)) and len(pair) == 2)
        try:
            spec = cls(**known)
        except TypeError:
            return cls()
        return spec if not spec.problem() else cls()


# -- fonts ------------------------------------------------------------------- #
def _installed(family: str) -> bool:
    from matplotlib import font_manager

    try:
        found = font_manager.findfont(font_manager.FontProperties(family=family),
                                      fallback_to_default=False)
    except ValueError:
        return False
    try:
        return font_manager.FontProperties(fname=found).get_name().lower() == family.lower()
    except (OSError, RuntimeError):
        return False


def _register_inter() -> None:
    from matplotlib import font_manager

    if _INTER.is_file() and not _installed("Inter"):
        font_manager.fontManager.addfont(str(_INTER))


def resolve_font(name: str) -> tuple:
    """``(family actually used, note or "")``: a missing font is substituted and SAID."""
    if name == "Inter":
        _register_inter()
    if _installed(name):
        return name, ""
    for stand_in in _STAND_INS.get(name, ()):
        if _installed(stand_in):
            return stand_in, (f"{name} is not installed on this computer; {stand_in} is "
                              f"used instead")
    return "DejaVu Sans", f"{name} is not installed on this computer; DejaVu Sans is used"


def _rc(spec: FigureSpec, family: str) -> dict:
    rc = dict(figstyle.rc_for(spec.preset))
    pt, small, line = spec.pt(), spec.small_pt(), spec.line_pt()
    rc.update({
        "font.family": "sans-serif",
        "font.sans-serif": [family, "DejaVu Sans"],
        "font.size": pt,
        "axes.labelsize": pt,
        "axes.titlesize": pt + 1,
        "xtick.labelsize": small,
        "ytick.labelsize": small,
        "legend.fontsize": small,
        "lines.linewidth": line,
        "axes.linewidth": max(MIN_LINE_PT * 2, line * 0.8),
        "xtick.major.width": max(MIN_LINE_PT * 2, line * 0.8),
        "ytick.major.width": max(MIN_LINE_PT * 2, line * 0.8),
        "xtick.major.size": pt * 0.4,
        "ytick.major.size": pt * 0.4,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": _page(spec)["bg"],
        "axes.unicode_minus": False,
        # math text (the half-time's t1/2) in the figure's own face, not DejaVu
        "mathtext.fontset": "custom", "mathtext.rm": family, "mathtext.it": f"{family}:italic",
        "mathtext.bf": f"{family}:bold",
    })
    if spec.dark:
        page = _page(spec)
        rc.update({"text.color": page["ink"], "axes.labelcolor": page["ink"],
                   "axes.titlecolor": page["ink"], "axes.edgecolor": page["edge"],
                   "xtick.color": page["edge"], "ytick.color": page["edge"],
                   "xtick.labelcolor": page["ink"], "ytick.labelcolor": page["ink"],
                   "legend.labelcolor": page["ink"], "figure.facecolor": page["bg"],
                   "axes.facecolor": page["bg"]})
    return rc


def _fit_samples(marks, spacing: float, size: float, floor: float, family: str) -> tuple:
    """``(size, marks)``: the sample lines under a bubble chart's columns (``[(mark, lines)]``, each mark's room its
    columns' span x ``spacing`` points) fitted in ONE size for every column: ``size`` where every line fits its room,
    else smaller in proportion, down to ``floor``; where a line is still too wide there, every part is broken into
    its words, one a line ("3" / "replicates" / "5,438" / "tracks"), at the size those fit, never under ``floor``. A
    first guess on an estimated room: `_unclash_samples` measures what is drawn."""
    def room(mark) -> float:
        return (float(mark["x1"]) - float(mark["x0"])) * spacing * 0.92

    def worst(items, pt) -> float:
        return max((_text_width_pt(line, pt, family) / room(m) for m, lines in items for line in lines
                    if room(m) > 0), default=0.0)

    over = worst(marks, size)
    if over <= 1.0:
        return size, marks
    smaller = max(floor, size / over)
    if worst(marks, smaller) <= 1.0:
        return smaller, marks
    words = [(m, [w for line in lines for w in line.split(" ") if w]) for m, lines in marks]
    return max(floor, min(size, size / max(worst(words, size), 1e-9))), words


def _text_width_pt(text: str, size_pt: float, family: str, weight: str = "normal") -> float:
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath

    if not text:
        return 0.0
    path = TextPath((0, 0), text, size=size_pt,
                    prop=FontProperties(family=family, weight=weight))
    return float(path.get_extents().width)


# -- the built figure ------------------------------------------------------------ #
@dataclass
class Built:
    """A figure ready to preview or write, and what it was built from."""

    figure: object
    spec: FigureSpec
    kind: str
    family: str
    notes: list
    rc: dict
    table: tuple          # (columns, rows) -- the numbers behind the figure, as numbers
    legend: str           # the drafted figure legend
    title: str
    notes_for_source: tuple = ()   # the header lines above the source-data table
    #: Further workbook sheets: ``((name, title, header lines, columns, rows), ...)``.
    more_sheets: tuple = ()
    #: What the layout itself knows is wrong (a set height the grid cannot fit), named by
    #: `check_figure` before anything it measures.
    layout_findings: tuple = ()
    #: A COMPOSED figure's panels, each ``{"letter", "kind", "title", "table", "notes",
    #: "more", "legend"}``: one workbook sheet and one CSV per panel (`write_figure`).
    parts: tuple = ()

    def size_mm(self) -> tuple:
        return self.spec.size_mm()


def _new_figure(spec: FigureSpec, rc: dict):
    import matplotlib as mpl
    from matplotlib.figure import Figure

    width, height = spec.size_mm()
    with mpl.rc_context(rc):
        fig = Figure(figsize=(width / _MM, height / _MM), layout="constrained")
        fig.get_layout_engine().set(w_pad=2 / 72, h_pad=2 / 72, wspace=0, hspace=0)
    return fig


# -- the SQD bubble chart -------------------------------------------------------- #
def _lighten(hex_colour: str, amount: float) -> str:
    value = hex_colour.lstrip("#")
    rgb = [int(value[i:i + 2], 16) for i in (0, 2, 4)]
    return "#" + "".join(f"{round(255 - amount * (255 - c)):02x}" for c in rgb)


#: The page a figure is drawn on. Every colour that assumed white paper reads from here,
#: so a dark slide is the same figure on another page rather than a second set of charts.
_LIGHT_PAGE = {"bg": "#ffffff", "ink": "#18181b", "muted": "#52525b", "grid": "#e4e4e7",
               "edge": "#4a5568", "masked": "#d4d4d8", "sparse": "#e4e4e7"}
_DARK_PAGE = {"bg": "#111114", "ink": "#f4f4f5", "muted": "#a1a1aa", "grid": "#34343a",
              "edge": "#a1a1aa", "masked": "#3f3f46", "sparse": "#27272a"}


def _page(spec: FigureSpec) -> dict:
    return _DARK_PAGE if spec.dark else _LIGHT_PAGE


def _mix(hex_colour: str, toward: str, amount: float) -> str:
    """``hex_colour`` with ``amount`` of it kept and the rest ``toward``."""
    a = [int(hex_colour.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
    b = [int(toward.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
    return "#" + "".join(f"{round(x * amount + y * (1 - amount)):02x}" for x, y in zip(a, b))


def _strong(spec: FigureSpec, hex_colour: str) -> str:
    """A colour made to stand out from the PAGE: darker on paper, lighter on a dark slide."""
    return _mix(hex_colour, "#ffffff", 0.65) if spec.dark else _darken(hex_colour)


def _pale(spec: FigureSpec, hex_colour: str, amount: float) -> str:
    """A colour faded toward the PAGE (white paper, or a dark slide)."""
    return _mix(hex_colour, _page(spec)["bg"], amount)


def _darken(hex_colour: str, factor: float = 0.74) -> str:
    value = hex_colour.lstrip("#")
    rgb = [int(value[i:i + 2], 16) for i in (0, 2, 4)]
    return "#" + "".join(f"{round(c * factor):02x}" for c in rgb)


def _bubble_plot_pt(spec: FigureSpec, names, legend_above: bool, r_max: float) -> float:
    """The plot's height in points, estimated from the figure's: what the axis range and the
    brackets' anchoring (`_draw_brackets`) both measure against."""
    small, pt = spec.small_pt(), spec.pt()
    lines = max([str(n).count("\n") + 1 for n in names] + [1])
    taken = lines * small * 1.25 + 2.2 * pt + (small * 1.6 if legend_above else 0.0)
    return max(spec.size_mm()[1] / _MM * 72.0 - taken, 4.0 * r_max)


def _bubble_range(ds, r_max: float, spec: FigureSpec, names, legend_above: bool,
                  above_pt: float = 0.0, below_pt: float = 0.0, number_pt: float | None = None,
                  number_top_pt: float = 0.0) -> tuple:
    """The D axis's ends, padded so the LARGEST bubble and a number printed under one fit
    inside the plot at THIS height. A fixed factor (x1.6, the first version) fits the default
    height and clipped the top bubble in a short composed panel: the bubble is sized in
    points, so its room has to be found in points too. ``above_pt`` and ``below_pt`` are the
    significance brackets' room over the topmost population and under the slowest, made
    INSIDE the axes as the screen chart makes it. ``number_pt``: a number's room under its bubble (its height and
    gap; one line at the small size when None). ``number_top_pt``: how far numbers stand over the top bubble, where
    one goes over its bubble (`_draw_columns` measures it, 1 Nov 2026; kept always, it stretched a slide's 16 pt plot
    to six decades)."""
    number = spec.small_pt() * 1.4 if number_pt is None else float(number_pt)
    plot = _bubble_plot_pt(spec, names, legend_above, r_max)
    return chart_text.bubble_axis_range(min(ds), max(ds), plot, r_max + 2.0 + above_pt + float(number_top_pt),
                                        r_max + number + 2.0 + below_pt)


#: The line spacing of a number set on two lines (`_number_style`), in its own font size.
NUMBER_LINE_SPACING = 1.1


def _number_style(texts, room: float, small: float, floor: float, family: str) -> tuple:
    """``(size in points, stacked)``: how EVERY number of a bubble figure is set so the widest fits ``room`` (its
    column), one format for all of them (1 Nov 2026, asked: "the numbers look a bit random moved arond cause it doesnt
    fit anymore ... high quality images with 3-10 conditions"). In this order: one line at the small size; the spread
    on a second line (`bubble_view.stacked_fraction`: "73%" over "±3"); then smaller, in `FIT_STEP_PT` steps down to
    ``floor`` (the journal's minimum, or the typed size where that is under it). Still too wide at the floor, it
    stays there, and `check_figure` names any overlap. A number is never moved beside its bubble for room."""
    from .bubble_view import stacked_fraction

    def widest(values, size):
        return max([_text_width_pt(line, size, family) + 1.0 for v in values if v for line in v.split("\n")]
                   or [0.0])

    if widest(texts, small) <= room:
        return small, False
    stacked = [stacked_fraction(t) for t in texts]
    size = small
    while size - FIT_STEP_PT >= floor - 1e-9 and widest(stacked, size) > room:
        size -= FIT_STEP_PT
    return size, True


def _column_spacing(spec: FigureSpec, n: int, span: float | None = None) -> float:
    """A bubble column's width in points, from the figure's: what a name may span, how large
    a bubble may be before it touches its neighbour, how wide a number may be (`_number_style`) and how wide a
    bracket's p may be. ``span``: the x range's width in column steps where grouped columns stand apart
    (`bubble_view.BubblePlotData.layout`). The plot's width is the figure's less what the D axis takes (its title, its
    tick labels, the layout's pads), never under 82 % of it (1 Nov 2026: a flat 82 % read a double-column plot 13 %
    narrower than it is, and its numbers moved for want of room they had)."""
    width = spec.width_value() / _MM * 72.0
    axis = 1.6 * spec.pt() + 3.0 * spec.small_pt() + 12.0
    return max(0.82 * width, width - axis) / (span if span else max(1, n) + 0.2)


def _grouped(layout) -> bool:
    return bool(layout and layout.get("groups"))


#: The kinds whose columns can be GROUPED (fits side by side, a folder design's nesting), so their Names are the parts.
GROUPED_NAME_KINDS = ("sqd_bubbles", "appd_bubbles", "sqd_crossed")


def grouped_names(data) -> dict | None:
    """How a grouped bubble chart's columns are named (30 Oct 2026, asked: "changing the name in export window of a
    grouped [chart] in sqd bubble plots ... i should change the name of the condition or whatever like measured -> ...
    and the groups"): ``{"groups": [each group header once], "labels": [each column label once], "columns": [(column
    key, its group, its label)], "side_by_side": bool}``, or None for a chart that is not grouped. Under fits side by
    side a group is a condition and a label a model ("1-pop"; the key "<condition> (<label>)",
    `bubble_view.side_by_side_key`); under a folder design a group is the outer factor's value and a label the inner
    ones (`factor_layout.columns`). A label that is also a group header is listed once, with the headers."""
    layout = getattr(data, "layout", None) if data is not None else None
    if not _grouped(layout):
        return None
    group_of = {}
    for group in layout["groups"]:
        for i in range(int(group["first"]), int(group["last"]) + 1):
            group_of[i] = str(group["label"])
    labels = [str(v) for v in layout.get("labels") or ()]
    columns = [(str(c), group_of.get(i, ""), labels[i] if i < len(labels) else str(c))
               for i, c in enumerate(data.conditions)]
    groups = list(dict.fromkeys(str(g["label"]) for g in layout["groups"]))
    return {"groups": groups, "labels": [v for v in dict.fromkeys(labels) if v not in groups],
            "columns": columns, "side_by_side": bool(getattr(data, "fits", None))}


def with_part_names(spec: FigureSpec, data) -> FigureSpec:
    """``spec`` with every column of a grouped chart named from its renamed PARTS: a column whose group header or label
    was renamed is shown as the renamed group with the renamed label -- "measured (1 population)" under fits side by
    side, "protein A glucose" under a folder design -- wherever the figure names a column (the source data's "Shown as", the
    significance brackets' sheets and sentence, the crossed plot's key). A column renamed by its own full key (an
    export made before 30 Oct 2026) keeps that name. The ONE rename route stays ``spec.names``: this only adds pairs."""
    parts = grouped_names(data)
    if parts is None or not spec.names:
        return spec
    named = {str(real) for real, shown in spec.names if str(shown).strip()}
    extra = []
    for key, group, label in parts["columns"]:
        if key in named or (group not in named and label not in named):
            continue
        g, l = spec.shown_name(group), spec.shown_name(label)
        extra.append((key, f"{g} ({l})" if parts["side_by_side"] else f"{g} {l}"))
    return replace(spec, names=tuple(spec.names) + tuple(extra)) if extra else spec


def name_sections(kind: str, payload: dict) -> list:
    """The Export dialog's Names, ``[(heading, [names])]``: a grouped bubble chart's group headers and its column
    labels, each listed once under its heading (`grouped_names`); any other chart its columns (`column_names`), no
    heading. Empty when the chart names nothing."""
    if kind in GROUPED_NAME_KINDS and isinstance(payload, dict):
        parts = grouped_names(payload.get("data"))
        if parts is not None:
            return [(heading, names) for heading, names in (("Group headers", parts["groups"]),
                                                            ("Column labels", parts["labels"])) if names]
    names = column_names(kind, payload)
    return [("", names)] if names else []


def _draw_columns(fig, ax, columns, points, spec: FigureSpec, family: str, *,
                  errors=None, series_labels=(), plan=None, layout=None, samples=None) -> None:
    """Bubbles in columns, in print: the screen chart's encoding, sized in points.

    ``columns`` are the column names (before `FigureSpec.shown_name`); ``points`` are
    ``(column, x_offset, D, fraction, text, colour, style_index)``. The SQD chart and
    its replicate-days page both draw through this, as their screen charts share one
    painter (`bubble_chart.paint_bubble`).

    Height = apparent D (log), AREA = fraction (radius by its square root, the screen's
    rule), colour = condition, the D marked by a tick either side OUTSIDE the bubble,
    the number BELOW every bubble in one ink (beside, as on screen, it reaches the next
    column at print widths).

    ``plan`` (`_significance_plan`) adds the significance brackets AT their population's
    bubbles (`_draw_brackets`): under the slowest, over every other; the D axis makes the
    room over the topmost population and under the slowest, as the screen chart does.

    ``layout`` (`factor_layout.columns`, folder designs): conditions GROUPED by their outer factor stand apart by a
    gap, each labelled by its inner values, with the outer value as a header under its group -- the screen
    chart's arrangement (`bubble_chart.draw_group_headers`).

    ``samples`` (`bubble_view.sample_marks`, when `FigureSpec.sample_n` is on): THE SAMPLE, "3 replicates · 1,271
    tracks", under each column's label -- or, with fits side by side, once under each group's header -- in the muted
    ink, one part a line where the column is too narrow for both (`bubble_view.sample_lines`), as on screen.
    """
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, NullLocator
    from matplotlib.transforms import blended_transform_factory, offset_copy

    from .plots import BUBBLE_SERIES_STYLES

    pt, small = spec.pt(), spec.small_pt()
    n = max(1, len(columns))
    grouped = _grouped(layout)
    xs = [float(x) for x in layout["x"]] if grouped else [float(i) for i in range(n)]
    ax.set_xlim(xs[0] - 0.6, xs[-1] + 0.6)
    # A D OF 0 (the immobile population of exact-zero jumps, `sqd`'s module head) at a "0" tick under the rest, as on
    # screen (`chart_text.zero_d_stand_in`); the brackets read the same drawn D (`_bracket_spots`)
    zero_d = chart_text.zero_d_stand_in(pt_[2] for pt_ in points)
    if zero_d is not None:
        points = [(c, off, chart_text.log_drawn(d, zero_d), *rest) for c, off, d, *rest in points]
    ds = [pt_[2] for pt_ in points] or [0.01, 1.0]
    spacing = _column_spacing(spec, n, (xs[-1] - xs[0] + 1.2) if grouped else None)
    measure = lambda s: _text_width_pt(s, small, family)                   # noqa: E731
    shown = ([spec.shown_name(str(v)) for v in layout["labels"]] if grouped
             else [spec.shown_name(c) for c in columns])
    if grouped:
        # A COLUMN LABEL OF A GROUP ("1-pop", "Best: 3-pop", an inner factor's value) breaks at spaces only and keeps
        # a word whole (it read "1-" over "pop"), in its column's room; and EACH GROUP'S HEADER is wrapped in its
        # group's (the headers ran into each other: "mNG, D2 simulatedmNG, D2 truth"), 1 Nov 2026
        names = ["\n".join(chart_text.wrap_text(str(c), measure, spacing, breaks=chart_text.SPACE_BREAKS,
                                                cut_words=False)) for c in shown]
        headers = ["\n".join(chart_text.wrap_text(
            spec.shown_name(str(group["label"])), measure,
            (float(group["x1"]) - float(group["x0"]) + 1.0) * spacing * 0.95, breaks=chart_text.SPACE_BREAKS,
            cut_words=False)) for group in layout["groups"]]
    else:
        names = ["\n".join(chart_text.wrap_text(str(c), measure, spacing * 0.92)) for c in shown]
        headers = []
    header_rows = max([h.count("\n") + 1 for h in headers] or [0])
    # THE SAMPLE's lines under the labels (or the group headers): each in its column's (or group's) room
    from .bubble_view import sample_lines

    marks = [(m, sample_lines(m["text"], measure, (float(m["x1"]) - float(m["x0"])) * spacing * 0.92))
             for m in (samples or ())]
    # EVERY SAMPLE FITTED TO ITS COLUMN, in one size (7 Nov 2026, found on SynR's eight-column draft at one column's
    # width: "3 replicates" / "5,438 tracks" ran into the next column's): a first guess here (`_fit_samples`: smaller
    # where a part is wider than its room, down to the journal's minimum, then one word a line), and the drawn texts
    # MEASURED once the layout has run (`_unclash_samples`, in `_fit_texts`): the room is only estimated here, and
    # on a narrow figure the estimate reads a column up to a third too wide (30 mm, two columns: 31.7 pt, 24.8 drawn)
    sample_pt, marks = _fit_samples(marks, spacing, small, min(small, spec.min_pt()), family)
    sample_rows = max([len(lines) for _m, lines in marks] or [0])
    if grouped or sample_rows:
        # the group headers and the sample take rows under the labels: counted where the plot's height is estimated
        names = [name + ("\n" * (header_rows + sample_rows) if i == 0 else "")
                 for i, name in enumerate(names)]
    # As large as a column allows; a bubble and its D ticks together never past half a column, so a number fitted to
    # its column (`_number_style`) never meets the next column's ticks; and never wider than a quarter of the plot's
    # height (1 Nov 2026: a slide's 16 pt sized it 64 pt across on a 148 pt plot, two populations touched and the
    # number between them had nowhere to stand; every print preset was under it already)
    r_max = min(2.0 * pt, 0.38 * spacing, 0.12 * _bubble_plot_pt(spec, names, bool(series_labels), 0.0))
    tick_len = min(0.6 * pt, 0.12 * spacing)
    # EVERY NUMBER IN ONE FORMAT, fitted to its column (`_number_style`): one line, else the spread under it, else
    # smaller -- never one number moved aside
    texts = [str(text) if spec.numbers and text else "" for *_head, text, _colour, _style in points]
    number_pt, stacked = _number_style(texts, 0.95 * spacing, small, min(small, spec.min_pt()), family)
    if stacked:
        from .bubble_view import stacked_fraction

        texts = [stacked_fraction(t) for t in texts]
    number_rows = max([t.count("\n") + 1 for t in texts if t] or [1])
    number_h = number_pt * (1.2 + NUMBER_LINE_SPACING * (number_rows - 1))
    number_gap = 0.25 * pt
    band = 0.0                              # the brackets stand inside the axes now
    r_min = min(0.35 * pt, r_max / 3)
    radius = lambda f: r_min + (r_max - r_min) * max(f, 0.0) ** 0.5             # noqa: E731
    import math as _math

    plot_pt = _bubble_plot_pt(spec, names, bool(series_labels), r_max)
    x_left, x_right = xs[0] - 0.6, xs[-1] + 0.6
    sizes = [(max(_text_width_pt(line, number_pt, family) for line in text.split("\n")) + 1.0, number_h)
             if text else (0.0, 0.0) for text in texts]
    over_plot = plan["above_pt"] if plan else 0.0           # the brackets' room over the top, inside the axes

    def laid_out(number_top: float, open_top: bool) -> tuple:
        """The D axis's ends, ``number_top`` points kept over the top bubble for numbers, and every bubble and its
        number placed on it: ``(lo, hi, points per decade, spots, numbers)``. WHERE EACH NUMBER STANDS
        (`chart_text.place_numbers`, the screen's rule): under its bubble, over it only where under hits another
        bubble -- in points, y down from the plot's top, clear of the brackets' room; ``open_top`` lets a number
        stand over the plot, to measure how far one does."""
        lo, hi = _bubble_range(ds, r_max, spec, names, bool(series_labels), over_plot,
                               plan["below_pt"] if plan else 0.0, number_pt=number_h + number_gap,
                               number_top_pt=number_top)
        # A range typed in the Export dialog wins, BEFORE the ticks and grid follow from it.
        lo, hi = (spec.y_min if spec.y_min is not None else lo,
                  spec.y_max if spec.y_max is not None else hi)
        if lo <= 0 or hi <= lo:
            raise FigureExportError("the D axis is logarithmic: it must start above 0 and "
                                    "below where it ends")
        per_decade = plot_pt / max(_math.log10(hi / lo), 1e-9)
        spots = [((xs[c] + off - x_left) * spacing, plot_pt - _math.log10(max(d, 1e-12) / lo) * per_decade,
                  radius(f)) for c, off, d, f, *_rest in points]
        bounds = (0.0, -_math.inf if open_top else over_plot, (x_right - x_left) * spacing, plot_pt)
        return lo, hi, per_decade, spots, chart_text.place_numbers(spots, sizes, gap=number_gap, reach=tick_len,
                                                                   bounds=bounds)

    # ROOM OVER THE TOP BUBBLE ONLY WHERE A NUMBER TAKES IT: placed once with the top open, the axis then grows by
    # how far any number stood over the plot, and they are placed again on it
    lo, hi, per_decade, spots, numbers = laid_out(0.0, True)
    over = max([over_plot - place["rect"][1] for place in numbers if place["rect"] is not None] + [0.0])
    if over > 0.0:
        # never more than a quarter of the plot (more stretches the axis past what its ticks can name), nor past a
        # typed top: a number with no room left stays inside, backed where it stands on a bubble
        room = over + 1.0 if spec.y_max is None and over + 1.0 <= 0.25 * plot_pt else 0.0
        lo, hi, per_decade, spots, numbers = laid_out(room, False)
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    # as many ticks as the plot's height holds at the tick size (`chart_text.log_tick_limit`; 6 Nov 2026: a per-spot
    # slow D at 0.0001 gave sixteen labels in a short panel)
    labelled = chart_text.zero_d_ticks(chart_text.log_ticks(lo, hi, chart_text.log_tick_limit(plot_pt, small)),
                                       zero_d)
    ticks = [v for v, _label in labelled]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.set_yticklabels([label for _v, label in labelled])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylabel(chart_text.D_AXIS_LABEL if spec.y_title is None else spec.y_title)
    if spec.x_title:
        ax.set_xlabel(spec.x_title)
    if spec.grid:
        for v in ticks:
            ax.axhline(v, color=_page(spec)["grid"], lw=max(MIN_LINE_PT, spec.line_pt() * 0.6),
                       zorder=0)

    ax.set_xticks(xs)
    ax.set_xticklabels([name.rstrip("\n") for name in names])
    ax.tick_params(axis="x", length=0)
    lines = max([name.rstrip("\n").count("\n") + 1 for name in names] + [1])
    muted = _page(spec)["muted"]

    def write_samples(chosen, down: float) -> float:
        """The samples ``chosen`` from ``down`` points under the axis; returns the points they take."""
        if not chosen:
            return 0.0
        at = offset_copy(blended_transform_factory(ax.transData, ax.transAxes), fig=fig, units="points", x=0,
                         y=-down)
        for mark, text_lines in chosen:
            drawn = ax.text((float(mark["x0"]) + float(mark["x1"])) / 2.0, 0, "\n".join(text_lines), transform=at,
                            ha="center", va="top", fontsize=sample_pt, color=muted, linespacing=1.15, clip_on=False)
            drawn._smtracker_sample = True                 # measured once laid out (`_unclash_samples`)
        return max(len(text_lines) for _m, text_lines in chosen) * sample_pt * 1.25 + 0.2 * pt

    # under each column's label (`bubble_view.sample_marks`' "label"), before any group header
    taken = write_samples([(m, t) for m, t in marks if m["under"] == "label"], lines * small * 1.25 + 0.35 * pt)
    if grouped:
        down = lines * small * 1.25 + 0.6 * pt + taken       # under the labels (and their samples), in points
        under = offset_copy(blended_transform_factory(ax.transData, ax.transAxes), fig=fig, units="points",
                            x=0, y=-down)
        # with fits side by side, the sample once under its group's header (as many lines as the tallest header)
        write_samples([(m, t) for m, t in marks if m["under"] == "group"],
                      down + 0.25 * pt + small * (1.25 * header_rows + 0.05))
        for group, header in zip(layout["groups"], headers):
            x0, x1 = group["x0"] - 0.4, group["x1"] + 0.4
            ax.add_artist(Line2D([x0, x1], [0, 0], transform=under, color=_page(spec)["grid"],
                                 lw=max(MIN_LINE_PT, spec.line_pt() * 0.8), clip_on=False))
            # a group of fits side by side is headed by its condition, under the name the figure gives it
            ax.text((x0 + x1) / 2.0, 0, header, transform=offset_copy(
                under, fig=fig, units="points", x=0, y=-0.25 * pt), ha="center", va="top", fontsize=small,
                color=_page(spec)["ink"], linespacing=1.15, clip_on=False)
    # the D ticks start AT the bubble's edge (1 Nov 2026, asked: "they are too far away from bubble, move them at the
    # bubble edge"; they stood 0.3 of the text size off)
    tick_w = max(MIN_LINE_PT * 2, spec.line_pt())
    for (column, x_offset, d, fraction, one_line, colour, style_index), text, spot, place in zip(
            points, texts, spots, numbers):
        x = xs[column] + x_offset
        r = radius(fraction)
        style = BUBBLE_SERIES_STYLES[style_index % len(BUBBLE_SERIES_STYLES)]
        solid = style["fill"] >= 1.0
        face = colour if solid else _pale(spec, colour, style["fill"])
        ax.scatter([x], [d], s=(2 * r) ** 2, facecolors=face,
                   edgecolors=_page(spec)["bg"] if solid else colour,
                   linewidths=0.8 if solid else max(0.6, style["linewidth"] * 0.5),
                   linestyles="solid" if solid else style["linestyle"], zorder=3)
        for sign in (-1, 1):
            ax.plot([x], [d], marker="_", ms=tick_len, mew=tick_w,
                    color=_strong(spec, colour), zorder=4,
                    transform=offset_copy(ax.transData, fig=fig, units="points",
                                          x=sign * (r + tick_len / 2), y=0))
        if place["rect"] is None:
            continue
        # EVERY number OUTSIDE its bubble, in one ink (asked for: "move the pop size below
        # the bubble? If the bubble is small it gets moved anyway"): inside some bubbles and
        # under others read as two kinds of number. Below by default; over only where below
        # hits another bubble (1 Nov 2026), never beside. Drawn over every bubble, backed
        # where neither side was clear.
        backing = None if place["clear"] else {"boxstyle": "round,pad=0.12", "fc": _page(spec)["bg"],
                                               "ec": "none", "alpha": 0.85}
        number = ax.annotate(text, (x, d), xytext=(place["x"] - spot[0], spot[1] - place["y"]),
                             textcoords="offset points", ha="center", va="center", multialignment="center",
                             linespacing=NUMBER_LINE_SPACING, bbox=backing, fontsize=number_pt,
                             color=_page(spec)["ink"], zorder=5)
        if text != one_line or number_pt < small - 1e-9:
            # set to fit its column: named by `check_figure`'s note, as every fitted text is (`_fit_notes`)
            number._smtracker_fit_from, number._smtracker_fit_text = float(small), str(one_line)
            number._smtracker_fit_what = "number"
    for column, reason in (errors or {}).items():
        ax.text(xs[column], hi / 1.15, "\n".join(chart_text.wrap_text(
            str(reason), measure, spacing * 0.9)), ha="center", va="top",
            fontsize=small, color=chart_text.FLAG_COLOR)
    if plan:
        # a bracket clears its bubbles' numbers on its side, wherever they stand
        reaches = {(c, d, f): chart_text.number_reach(spot, place)
                   for (c, _off, d, f, *_rest), spot, place in zip(points, spots, numbers)}
        _draw_brackets(fig, ax, plan["brackets"], spec, per_decade=per_decade,
                       reach=lambda s, side: reaches.get(s, (radius(s[2]),) * 2)[side == "below"])
        # A picked pair with no p is named in the checks (`check_figure`), never drawn.
        ax._smtracker_findings = list(plan["findings"])
    if series_labels and spec.legend:
        handles = []
        for index, _label in enumerate(series_labels):
            style = BUBBLE_SERIES_STYLES[index % len(BUBBLE_SERIES_STYLES)]
            solid = style["fill"] >= 1.0
            handles.append(Line2D([], [], marker="o", linestyle="none", markersize=small,
                                  markerfacecolor="#71717a" if solid
                                  else _pale(spec, "#71717a", style["fill"]),
                                  markeredgecolor=_page(spec)["bg"] if solid else "#71717a"))
        # Over the bracket band when there is one (`_points_over_plot`).
        anchor = ({"bbox_to_anchor": (0.5, band), "bbox_transform": blended_transform_factory(
            ax.transAxes, _points_over_plot(fig, ax))} if band else
                  {"bbox_to_anchor": (0.5, 1.0)})
        ax.legend(handles, list(series_labels), loc="lower center", ncol=len(handles),
                  frameon=False, handletextpad=0.3, columnspacing=1.0, **anchor)


def _title(spec: FigureSpec, kind: str, axis: str, payload: dict | None = None) -> str:
    """The axis title in this figure: the spec's when set, else the chart's own."""
    chosen = spec.y_title if axis == "y" else spec.x_title
    if chosen is not None:
        return chosen
    y, x = default_titles(kind, payload)
    return y if axis == "y" else x


def _draw_msd(fig, ax, entries, spec: FigureSpec, family: str, kind: str) -> None:
    """The ensemble MSD in print, from the screen chart's own payload (`msd_view`):
    the mean at each lag with its SEM, FILLED over the lags the fit used and open beyond,
    the fit line over those lags and its dashed extension, and -- for one condition --
    the pale per-track curves the screen draws behind it."""
    pt, small, line = spec.pt(), spec.small_pt(), spec.line_pt()
    single = kind == "msd_condition"
    t_max, y_lo, y_hi = 0.0, 0.0, 0.0
    for entry in entries:
        chart, colour = entry["result"]["chart"], entry["colour"]
        t_max = max(t_max, chart["t_max"])
        y_lo, y_hi = min(y_lo, chart["y_range"][0]), max(y_hi, chart["y_range"][1])
        if single:
            for track in chart.get("tracks") or []:
                ax.plot([a for a, _b in track], [b for _a, b in track],
                        color=_pale(spec, colour, 0.25), lw=max(MIN_LINE_PT, line * 0.5),
                        zorder=1)
        curve, n_fit = chart["curve"], int(chart["n_fit"])
        ts = [c[0] for c in curve]
        ms = [c[1] for c in curve]
        # THE SCREEN'S BARS (`msd_view._bars`): the SD over replicate days under Days, else the SEM over tracks
        bars = chart.get("bars")
        if bars is None:
            bars = [(c[0], c[1] - c[2], c[1] + c[2]) for c in curve if c[2] > 0]
        at = {t: m for t, m in zip(ts, ms)}
        bar_t = [t for t, _lo, _hi in bars if t in at]
        if bar_t:
            ax.errorbar(bar_t, [at[t] for t in bar_t],
                        yerr=[[at[t] - lo for t, lo, _hi in bars if t in at],
                              [hi - at[t] for t, _lo, hi in bars if t in at]],
                        fmt="none", ecolor=colour, elinewidth=max(MIN_LINE_PT * 2, line * 0.8),
                        capsize=pt * 0.18, capthick=max(MIN_LINE_PT * 2, line * 0.8), zorder=2)
        ax.plot(ts[:n_fit], ms[:n_fit], "o", ms=pt * 0.55, color=colour,
                markeredgecolor=_page(spec)["bg"], markeredgewidth=0.4, zorder=4,
                label=spec.shown_name(entry["name"]))
        if len(ts) > n_fit:
            ax.plot(ts[n_fit:], ms[n_fit:], "o", ms=pt * 0.55, markerfacecolor=_page(spec)["bg"],
                    markeredgecolor=colour, markeredgewidth=max(0.6, line * 0.8), zorder=4)
        (x0, y0), (x1, y1) = chart["fit"]
        ax.plot([x0, x1], [y0, y1], "-", color=colour, lw=max(0.6, line * 1.2), zorder=3)
        if chart.get("extension"):
            (x0, y0), (x1, y1) = chart["extension"]
            ax.plot([x0, x1], [y0, y1], "--", color=colour, lw=max(0.5, line), zorder=3)
    ax.set_xlim(0, max(t_max * 1.04, 1e-9))
    span = (y_hi - y_lo) or 1.0
    ax.set_ylim(y_lo, y_hi + span * 0.06)
    if spec.grid:
        ax.grid(axis="y", color=_page(spec)["grid"], lw=max(MIN_LINE_PT, line * 0.6), zorder=0)
        ax.set_axisbelow(True)
    ax.set_ylabel(_title(spec, kind, "y"))
    ax.set_xlabel(_title(spec, kind, "x"))
    keyed = not single and spec.legend and len(entries) > 1
    if spec.numbers:
        for i, entry in enumerate(entries):
            fit = entry["result"]["fit"]
            note = f"D = {fit['D']:.3g} µm²/s"
            if not single and not keyed:
                # No legend to name the colours, so each D names its condition.
                note = f"{spec.shown_name(entry['name'])}: {note}"
            ax.annotate(note, (0.02, 0.97 - 0.075 * i), xycoords="axes fraction",
                        ha="left", va="top", fontsize=small, color=entry["colour"]
                        if not single else _page(spec)["ink"])
    if keyed:
        ax.legend(loc="lower right", frameon=False, handletextpad=0.3)


def _msd_source(entries, spec: FigureSpec) -> tuple:
    # "Tracks", not "Pairs": the curve's fourth value is how many tracks reach the lag (`msd_view`'s
    # ``"curve"``), and the SEM is over those tracks -- both were labelled as pairs until 16 Oct 2026.
    columns = ["Condition", "Shown as", "Lag (s)", "MSD (µm²)", "SEM over tracks (µm²)", "Tracks",
               "Fitted", "Bar from (µm²)", "Bar to (µm²)", "Bar"]
    rows = []
    for entry in entries:
        chart = entry["result"]["chart"]
        bars = {t: (lo, hi) for t, lo, hi in chart.get("bars") or []}
        for i, (lag, msd, sem, tracks) in enumerate(chart["curve"]):
            lo, hi = bars.get(lag, (None, None))
            rows.append([entry["key"], spec.shown_name(entry["name"]), float(lag),
                         float(msd), float(sem), int(tracks), i < int(chart["n_fit"]),
                         lo, hi, chart.get("bar_words", "") if lo is not None else ""])
    return columns, rows


def _msd_day_sheet(entries, spec: FigureSpec) -> tuple:
    """Each replicate day's own curve and D, as a further sheet, or () when no condition has days."""
    rows = []
    for entry in entries:
        for d in entry["result"]["chart"].get("days") or []:
            for t, m in d["curve"]:
                rows.append([spec.shown_name(entry["name"]), d["name"], float(t), float(m), d.get("D")])
    if not rows:
        return ()
    return (("Replicates", "Each replicate's own ensemble MSD and early-lag fit",
             ("Each replicate is fitted on its own with the settings of the run; replicates are independent.",),
             ["Condition", "Replicate", "Lag (s)", "MSD (µm²)", "D of that replicate (µm²/s)"], rows),)


def _msd_fit_line(entry, spec: FigureSpec) -> str:
    result = entry["result"]
    fit, s = result["fit"], result["settings"]
    weighting = "weighted by pairs" if s["weighted"] else "each track once"

    def num(key, fmt):
        # A negative intercept has no σ, and a curve too short has no α: said, not crashed.
        value = fit.get(key)
        return "n/a" if value is None else format(value, fmt)

    text = (f"{spec.shown_name(entry['name'])}: D = {num('D', '.4g')} µm²/s, intercept "
            f"{num('intercept', '.4g')} µm² (σ {num('sigma_nm', '.0f')} nm), "
            f"α = {num('alpha', '.2f')}; first {fit['n_lags']} lags, {weighting}; "
            f"{result['n_tracks']} tracks")
    interval = result.get("interval")
    if interval and interval.get("mode") == "day" and interval.get("values"):
        listed = ", ".join(f"{name} {value:.3g}" for name, value in interval["values"])
        text += f"; D per replicate: {listed}"
        if "sd" in interval:
            text += f" (mean {interval['mean']:.3g} ± SD {interval['sd']:.2g})"
    elif interval and interval.get("mode") in ("cell", "movie") and "lo" in interval:
        text += (f"; 95 % interval over {interval['n_units']} {interval['mode']}s "
                 f"[{interval['lo']:.3g}, {interval['hi']:.3g}]")
    return text + "."


def _msd_source_notes(entries, spec: FigureSpec) -> tuple:
    return ("MSD: mean squared displacement at the lag (µm²), over all position pairs with pair weighting, "
            "otherwise each track once. SEM: over the tracks that reach the lag. Tracks: how many reach it. "
            "Fitted: the lag was in the fit.",
            *(_msd_fit_line(e, spec) for e in entries))


def msd_legend(entries, spec: FigureSpec) -> str:
    """A drafted legend for the MSD figure: what the points, the fit and the intercept are."""
    s = entries[0]["result"]["settings"]
    where = {"xy": "in the camera plane", "along": "along the cell axis",
             "across": "across the cell"}[s["direction"]]
    ndim = entries[0]["result"]["fit"]["ndim"]
    parts = [f"Ensemble mean squared displacement (MSD) against lag time, {where}. Points "
             + ("are the mean over all pairs of positions at each lag" if s["weighted"]
                else "are the mean of the tracks' own MSDs at each lag, each track once")
             + " " + (entries[0]["result"]["chart"].get("bar_words")
                      or "± SEM over tracks (indicative only, since tracks within a replicate are not independent)")
             + f"; filled points are the first {s['n_lags']} lags, fitted "
             + ("weighted by their number of pairs" if s["weighted"] else "with equal weights")
             + f" to MSD = {2 * ndim}·D·t + b (solid line; the dashed line extends it beyond the fit)."]
    parts.append(" ".join(_msd_fit_line(e, spec) for e in entries))
    parts.append(f"For a free molecule imaged continuously b = {2 * ndim}·(σ² − D·τ/3). Motion blur lowers it "
                 "and the cell wall raises it, so σ = √(b/" + str(2 * ndim) + ") is not a measurement of the "
                 "localization error. α < 1 can come from mixed populations, the offset, the cell wall or slow "
                 "long tracks (survivorship) and does not show anomalous diffusion. MSD is a cross-check; the "
                 "population fractions come from the SQD fit.")
    return " ".join(parts)


# -- The Spatial Map's profiles and the appD cell-length card -------------------------------------------------------- #
#: Agreed 17 Oct 2026. Each draws the payload its card paints (`spatial_map_view.profile_card_payload` /
#: `day_profile_payload`; `appd_view.cell_length_payload`).

def _profile_line_colour(line: dict, colours: dict) -> str:
    if line["key"] in colours:
        return colours[line["key"]]
    if "day_index" in line:
        return figstyle.day_shade(colours.get("condition") or QC_DATA, line["day_index"], line["n_days"])
    return colours.get("condition") or QC_DATA


def _profiles_height(payload: dict, spec: FigureSpec) -> float:
    """One row of two plots per population, as the manuscript's figure: about a third of the width a row."""
    default = spec.default_height_mm()
    width = spec.width_value()
    rows = max(1, len(payload.get("panels") or []))
    tallest = spec.preset_values.get("max_height_mm") or 600.0
    per_row = width * (0.3 if spec.preset in LARGE_PRESETS else 0.36)
    # the key, the axis titles under the last row and a figure title, in the preset's own type
    frame = (5.0 * spec.pt() + 2.0 * spec.small_pt()) * 25.4 / 72.0 + (spec.pt() * 0.5 if spec.title else 0.0)
    if spec.preset in LARGE_PRESETS:
        return round(max(default, rows * per_row + frame), 1)
    return round(min(tallest, max(default, rows * per_row + frame)), 1)


def _draw_spatial_profiles(fig, payload: dict, spec: FigureSpec, family: str):
    """Where the jumps sit, in print, as the manuscript's short-axis figure: ONE ROW PER POPULATION, ALONG the cell
    (distance from the nearer pole, nm; the cap's edge dotted) on the left and ACROSS it (the relative short-axis
    position, -1..1, cylinder only) on the right, a line per condition -- or the
    condition bold and its days thin -- the even reference dashed, the y axis shared within a column. The key above.
    Returns the first row's along-the-cell axes."""
    from matplotlib.lines import Line2D

    panels = payload.get("panels") or []
    if not panels:
        raise FigureExportError("no profile to draw yet")
    colours = payload.get("colours") or {}
    page = _page(spec)
    line = spec.line_pt()
    grid = fig.add_gridspec(len(panels), 2)

    def values(seq):
        return [v for v in (seq or []) if v is not None]

    # densities share a y axis within a column; an SDA value map (a mean in its own unit) has its own
    tops = {"long": 1e-9, "short": 1e-9}
    for panel in panels:
        pl = panel["payload"]
        if pl.get("unit"):
            continue
        for key in tops:
            tops[key] = max(tops[key], *values(pl.get(f"{key}_reference")),
                            *[v for ln in pl["lines"] for v in values(ln.get(key))])
    first = None
    for r, panel in enumerate(panels):
        pl = panel["payload"]
        lines = [ln for ln in pl["lines"] if ln.get("long") and ln.get("short")]
        for c, key in enumerate(("long", "short")):
            ax = fig.add_subplot(grid[r, c])
            first = first or ax
            centres = [1000.0 * v for v in pl["long_centres"]] if key == "long" else list(pl["short_centres"])
            if pl.get(f"{key}_reference"):
                ax.plot(centres, pl[f"{key}_reference"], ls=(0, (4.0, 2.5)), color=page["muted"],
                        lw=max(0.6, line), zorder=3)
            if key == "long":
                ax.axvline(1000.0 * pl["cap_um"], ls=":", color=page["muted"], lw=max(MIN_LINE_PT, line * 0.8))
            for ln in sorted(lines, key=lambda ln: bool(ln.get("bold"))):
                # a bin with too few steps has no value: a gap, never a zero
                ax.plot(centres, [float("nan") if v is None else v for v in ln[key]],
                        color=_profile_line_colour(ln, colours),
                        lw=max(0.9, line * 1.6) if ln.get("bold") else max(0.5, line * 0.9), zorder=4)
            top = (max([1e-9, *[v for ln in lines for v in values(ln.get(key))]]) if pl.get("unit")
                   else tops[key])
            ax.set_ylim(0, top * 1.08)
            ax.set_xlim((0, 1000.0 * pl["half_length_um"]) if key == "long" else (-1.0, 1.0))
            if key == "short":
                # five ticks where a column is wide enough for them, else the ends and the axis
                roomy = spec.width_value() >= 120.0 and spec.preset not in LARGE_PRESETS
                ticks = [-1.0, -0.5, 0.0, 0.5, 1.0] if roomy else [-1.0, 0.0, 1.0]
                ax.set_xticks(ticks)
                ax.set_xticklabels([f"{t:g}".replace("-", "−") for t in ticks])
            _qc_grid(ax, spec)
            if c == 0:
                y_label = pl.get("y_label") if pl.get("unit") else _title(spec, 'spatial_profiles', 'y')
                ax.set_ylabel(f"{panel['title']}\n{y_label}")
            else:
                ax.tick_params(labelleft=True)
            if r == len(panels) - 1:
                if key == "long":
                    ax.set_xlabel(_title(spec, "spatial_profiles", "x"))
                else:
                    # ENDING at the axis's right end: centred, it spilled past the page edge on a narrow figure
                    ax.set_xlabel("short-axis position", loc="right")
            else:
                ax.tick_params(labelbottom=False)
    if spec.legend:
        lines = [ln for ln in panels[0]["payload"]["lines"] if ln.get("long")]
        handles = [Line2D([], [], color=_profile_line_colour(ln, colours),
                          lw=max(0.9, line * 1.6) if ln.get("bold") else max(0.5, line * 0.9)) for ln in lines]
        names = [spec.shown_name(ln["name"]) for ln in lines]
        if any(p["payload"].get("long_reference") for p in panels):
            handles.append(Line2D([], [], ls=(0, (4.0, 2.5)), color=page["muted"], lw=max(0.6, line)))
            names.append("even distribution")
        # ITS COLUMN'S HALF OF THE WIDTH (30 Oct 2026): over the left plot but allowed the whole figure's width, the
        # key overhung that plot by up to half the page, so the layout squeezed the left column -- further on every
        # layout run (Nature 1 column 21.0 -> 17.0 mm a plot over eight runs, a slide 16.8 -> 5.3 mm, its tick
        # labels over each other in the file written)
        _legend_above(first, handles, names, spec, family, share=0.5)
    return first


def _spatial_profiles_source(payload: dict, spec: FigureSpec) -> tuple:
    rows, numbers = [], {}
    for panel in payload.get("panels") or []:
        pl = panel["payload"]
        unit = pl.get("unit") or "density"
        for ln in pl["lines"]:
            for key, axis in (("long", "along (nm from the nearer pole)"),
                              ("short", "across (relative short-axis position, cylinder only)")):
                centres = pl[f"{key}_centres"]
                counts = ln.get(f"{key}_counts") or [None] * len(centres)
                reference = pl.get(f"{key}_reference") or [None] * len(centres)
                for i, c in enumerate(centres):
                    if ln.get(key):
                        value = ln[key][i]
                        rows.append([panel["title"], spec.shown_name(ln["name"]), axis,
                                     1000.0 * c if key == "long" else c,
                                     None if value is None else float(value), unit,
                                     None if counts[i] is None else float(counts[i]),
                                     None if reference[i] is None else float(reference[i])])
        # one numbers table per column set: the SDA value maps' means (each its unit with its name) beside the
        # distributions' measures
        title = f"{panel['title']} ({pl['unit']})" if pl.get("unit") else panel["title"]
        numbers.setdefault(tuple(pl["columns"]), []).extend(
            [[title, spec.shown_name(str(r[0])), *r[1:]] for r in pl["rows"]])
    columns = ["Population", "Line", "Axis", "Bin centre", "Value", "Unit", "Rows in the bin (weighted)",
               "Even distribution"]
    tables = []
    for cols, body in numbers.items():
        if cols[2:3] == ("Mean |d_norm|",):
            tables.append(("Numbers", "Mean |d_norm|, mean distance from the pole, share in the caps",
                           ("Mean |d_norm| is measured on the rows in the cylinder; a population spread evenly "
                            "through the volume has 0.424.",),
                           ["Population", *cols], body))
        else:
            tables.append(("Means", "Mean value, in the caps and in the cylinder",
                           ("The mean of the measured values; a bin or part with too few steps has none.",),
                           ["Map", *cols], body))
    return (columns, rows), tuple(tables)


def spatial_profiles_legend(payload: dict) -> str:
    titles = [p["title"].lower() for p in payload.get("panels") or []]
    if any(p["payload"].get("heading") == "Along and across the cell" for p in payload.get("panels") or []):
        return ("The SDA maps along and across the cell, one row per map (" + ", ".join(titles) + "). Left: by "
                "distance from the nearer pole, the dotted line the cap edge. Right: by relative short-axis position "
                "(-1 and 1 are the outline, 0 the cell axis), from the cylinder only (the caps, 18 % of the length at "
                "each end, are left out). Localisations: their density, area 1, beside a population spread evenly "
                "through the cell's volume (dashed). Speed, apparent D and acceleration: the mean of the measured "
                "values in each bin, in the map's unit, with no point where a bin has too few steps; never "
                "interpolated. The values are apparent: capped by the tracker's linking radius and floored by "
                "localisation error. Positions are laid onto one median capsule.")
    return ("Where each population's jumps sit, one row per population ("
            + ", ".join(titles) + "). Right: density of the relative short-axis position (-1 and 1 are the "
            "outline, 0 the cell axis), from jumps in the cylinder only (the caps, 18 % of the length at each end, "
            "are left out). Left: density of the distance from the nearer pole; the dotted line marks the cap "
            "edge. Each curve has area 1. The dashed curve is a population spread evenly through the cell's "
            "volume (mean |d_norm| 0.424); it is not flat, because a chord through a cylinder is longest on its "
            "axis. Positions are laid onto one median capsule. Compare populations within the same cells; the "
            "outline is not the membrane.")


def _draw_appd_cell_length(fig, ax, payload: dict, spec: FigureSpec, family: str) -> None:
    """Slow share against cell length in print: each cell a dot in its condition's colour (area by its tracks),
    each condition's mean per length class a line."""
    from matplotlib.lines import Line2D

    if not payload.get("available"):
        raise FigureExportError(f"no cell-length figure: {payload.get('why', 'not computed')}")
    labels, colours = payload.get("labels") or {}, payload.get("colours") or {}
    page = _page(spec)
    line, small = spec.line_pt(), spec.small_pt()
    scale = small / 6.0
    for c, pts in payload["points"].items():
        colour = colours.get(c) or QC_DATA
        ax.scatter([pt["x"] for pt in pts], [pt["y"] for pt in pts], s=[(scale * (1.0 + 0.5 * pt["n"] ** 0.5)) ** 2
                                                                          for pt in pts],
                   color=colour, alpha=0.45, linewidths=0, zorder=2)
    for c, binned in payload["binned"].items():
        colour = _strong(spec, colours.get(c) or QC_DATA)
        ax.plot([b[0] for b in binned], [b[1] for b in binned], "-o", color=colour, lw=max(0.9, line * 1.6),
                ms=small * 0.5, markerfacecolor=page["bg"], zorder=4)
    lo, hi = payload["x_range"]
    pad = 0.04 * ((hi - lo) or 1.0)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(0, 100)
    ax.set_xlabel(_title(spec, "appd_cell_length", "x"))
    ax.set_ylabel(_title(spec, "appd_cell_length", "y"))
    _qc_grid(ax, spec)
    if spec.legend:
        handles = [Line2D([], [], color=colours.get(c) or QC_DATA, marker="o", ls="-", lw=max(0.9, line * 1.6),
                          ms=small * 0.5) for c in payload["points"]]
        _legend_above(ax, handles, [spec.shown_name(labels.get(c, c)) for c in payload["points"]], spec, family)


def _appd_cell_length_source(payload: dict, spec: FigureSpec) -> tuple:
    labels = payload.get("labels") or {}
    rows = [[c, spec.shown_name(labels.get(c, c)), pt.get("day"), float(pt["x"]), float(pt["y"]), int(pt["n"])]
            for c, pts in payload["points"].items() for pt in pts]
    columns = ["Condition", "Shown as", "Replicate", "Cell length (µm)", "Slow share (%)", "Tracks"]
    table = ("Length-matched", "Each condition as measured and at matched cell length", (payload.get("text", ""),),
             list(payload["columns"]), [[spec.shown_name(labels.get(r[0], r[0])), *r[1:]] for r in payload["rows"]])
    return (columns, rows), (table,)


def appd_cell_length_legend(payload: dict) -> str:
    source = ("the SQD fit (the mean probability of its jumps being in the slowest population)"
              if payload.get("source") == "sqd" else "the apparent-D fit (the mean probability of its tracks being "
              "in the slowest population)")
    return ("Slow share per cell against cell length, from " + source + ". Each dot is one cell, its area set by "
            "its tracks. Lines are each condition's mean per length class (quartiles of all cells). "
            + (payload.get("text", "") or "")
            + ". Length-matched: the class means are weighted by every condition's cells together. The plot is "
            "descriptive, since cells within a replicate are not independent.")


# -- MSD: the Checks and v2 pages --------------------------------------------------------------------------------- #
#: Asked 16 Oct 2026 ("Export for MSD pages"): each draws from the page helper its card paints from
#: (`msd_view.window_payload`, `length_check`, `v2_rows`, the result's ``histogram``).

def _draw_msd_checks(fig, entry: dict, spec: FigureSpec, family: str):
    """One condition's MSD checks in print: D against the number of lags fitted (by pairs in the condition's colour,
    each track once dashed, the Run's window marked) over D from the early lags of each track-length class (a class
    under `msd_view.LENGTH_MIN_TRACKS` tracks faded). Side by side when wide. Returns the window plot's axes."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator

    from .msd_view import length_check, window_payload

    result = entry["result"]
    w, ln = window_payload(result), length_check(result)
    if w is None:
        raise FigureExportError(f"{entry.get('name', 'this condition')} has no MSD checks to draw")
    colour = entry.get("colour") or QC_DATA
    page = _page(spec)
    line, small = spec.line_pt(), spec.small_pt()
    width_mm, height_mm = spec.size_mm()
    wide = width_mm / height_mm >= _APPD_CHECKS_WIDE
    grid = (fig.add_gridspec(1, 2, width_ratios=(1.15, 1.0)) if wide
            else fig.add_gridspec(2, 1, height_ratios=(1.0, 1.0)))
    win, bars = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    ink = _strong(spec, colour)
    win.axvline(w["chosen"], ls=":", color=page["muted"], lw=max(MIN_LINE_PT, line), zorder=1)
    for key, style, c in (("pairs", "-", ink), ("tracks", (0, (4.0, 2.5)), page["ink"])):
        pts = [(n, v) for n, v in zip(w["lags"], w[key]) if v is not None]
        win.plot([n for n, _v in pts], [v for _n, v in pts], ls=style, color=c, lw=max(0.8, line * 1.4),
                 marker="o", ms=small * 0.45, zorder=3)
    values = [v for v in w["pairs"] + w["tracks"] if v is not None]
    win.set_ylim(min(values) * 0.9, w["y_max"])
    win.xaxis.set_major_locator(FixedLocator(w["lags"]))
    win.set_xlabel(_title(spec, "msd_checks", "x"))
    win.set_ylabel(_title(spec, "msd_checks", "y"))
    if spec.legend:
        handles = [Line2D([], [], color=ink, lw=max(0.8, line * 1.4)),
                   Line2D([], [], ls=(0, (4.0, 2.5)), color=page["ink"], lw=max(0.8, line * 1.4)),
                   Line2D([], [], ls=":", color=page["muted"], lw=max(MIN_LINE_PT, line))]
        _legend_above(win, handles, ["by pairs", "each track once", f"the Run's {w['chosen']} lags"], spec, family,
                      share=0.55 if wide else 1.0, title=_keyed_title(spec, entry["name"]))
    bins = (ln or {}).get("bins") or []
    if bins:
        xs = list(range(len(bins)))
        bars.bar(xs, [b["D"] or 0.0 for b in bins], width=0.62, linewidth=0, zorder=2,
                 color=[_pale(spec, colour, 0.3) if b["few"] else colour for b in bins])
        if spec.numbers:
            for x, b in zip(xs, bins):
                if b["D"] is not None:
                    bars.text(x, b["D"] * 1.02, f"{b['n']:,}", ha="center", va="bottom", fontsize=small,
                              color=page["muted"] if b["few"] else page["ink"])
        bars.set_xticks(xs)
        bars.set_xticklabels([b["label"] for b in bins])
        bars.set_xlim(-0.6, len(bins) - 0.4)
        bars.set_ylim(0, ln["y_max"] * (1.12 if spec.numbers else 1.0))
    else:
        bars.set_axis_off()
    bars.tick_params(axis="x", length=0)
    bars.set_ylabel(chart_text.D_LINEAR_LABEL)
    bars.set_xlabel(chart_text.TRACK_STEPS_LABEL)
    for ax in (win, bars):
        _qc_grid(ax, spec)
    return win


def _msd_checks_source(entry: dict, spec: FigureSpec) -> tuple:
    from .msd_view import LENGTH_MIN_TRACKS, length_check, window_payload

    result = entry["result"]
    shown = spec.shown_name(entry["name"])
    w, ln = window_payload(result), length_check(result) or {"bins": []}
    columns = ["Condition", "Shown as", "Lags fitted", "D by pairs (µm²/s)", "D each track once (µm²/s)",
               "The Run's window"]
    rows = [[entry["key"], shown, n, p, t, n == w["chosen"]] for n, p, t in zip(w["lags"], w["pairs"], w["tracks"])]
    length = ("Track length", "D from the early lags of each track-length class",
              (f"A class with fewer than {LENGTH_MIN_TRACKS} tracks is drawn faded and not quoted.",),
              ["Condition", "Shown as", "Track length (steps)", "Tracks", "D (µm²/s)", "Quoted"],
              [[entry["key"], shown, b["label"], int(b["n"]), b["D"], not b["few"]] for b in ln["bins"]])
    return (columns, rows), (length,)


def _msd_checks_notes(entry: dict) -> tuple:
    from .msd_view import length_check, window_payload

    result = entry["result"]
    w, ln = window_payload(result), length_check(result)
    s = result["settings"]
    return ("D: the slope of the ensemble MSD over the first n lags, divided by 2 x the number of dimensions. "
            + ("Weighted by pairs: each lag counts every displacement pair at that lag."
               if s["weighted"] else "Each track counts once.") + " The weighting of the run is the solid line.",
            "Fit window: " + (w["text"] or "too few lags") + ".",
            "Track length: " + ((ln or {}).get("text") or "not formed") + ".",
            "Pair weighting favours long tracks as the lag grows, and long tracks are slow; the cell wall also bends "
            "every curve. A D that falls with the window is not evidence of slower molecules.")


def msd_checks_legend(entry: dict) -> str:
    from .msd_view import window_payload

    w = window_payload(entry["result"])
    return (f"Checks of the ensemble MSD fit of {entry['name']}. Top: D from the first 2 to 8 lags, weighted by "
            "displacement pairs (solid) and with each track counted once (dashed); the dotted line marks the "
            f"{w['chosen']} lags of the run. {w['text'].capitalize() if w['text'] else ''}. Bottom: D from the early "
            "lags of each track-length class alone (faded where there are too few tracks). A D that falls with "
            "the fit window or with track length points to survivorship and confinement, not to slower molecules.")


def _draw_msd_v2(fig, ax, entry: dict, spec: FigureSpec, family: str) -> None:
    """v2's localisation-error panel in print: every track's sigma = sqrt(|b|/4) from its own line over lags 1-7,
    in v2's 20 nm bins, the negative intercepts kept on the left (grey) where v2's abs() folded them right."""
    from matplotlib.patches import Patch

    result = entry["result"]
    hist = result.get("histogram")
    if not hist or not hist.get("bins"):
        raise FigureExportError(f"{entry.get('name', 'this condition')} has no v2 per-track fits to draw")
    colour = entry.get("colour") or QC_DATA
    page = _page(spec)
    grey = _pale(spec, page["muted"], 0.55)
    for a, b, count, negative in hist["bins"]:
        if count:
            ax.bar(a + (b - a) * 0.04, count, width=(b - a) * 0.92, align="edge", linewidth=0,
                   color=grey if negative else colour, zorder=2)
    lo, hi = hist["bins"][0][0], hist["bins"][-1][1]
    if lo < 0 < hi:
        ax.axvline(0.0, ls=(0, (3.0, 2.0)), color=page["ink"], lw=max(MIN_LINE_PT, spec.line_pt()), zorder=3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, max(1, hist["y_max"]) * 1.08)
    ax.set_xlabel(_title(spec, "msd_v2", "x"))
    ax.set_ylabel(_title(spec, "msd_v2", "y"))
    if spec.legend:
        _legend_above(ax, [Patch(color=grey, lw=0), Patch(color=colour, lw=0)],
                      ["negative intercept (v2 folds it right)", "positive"], spec, family,
                      title=_keyed_title(spec, entry["name"]))
    _qc_grid(ax, spec)


def _msd_v2_source(entry: dict, spec: FigureSpec) -> tuple:
    from .msd_view import v2_rows

    result = entry["result"]
    shown = spec.shown_name(entry["name"])
    columns = ["Condition", "Shown as", "Bin from (nm)", "Bin to (nm)", "Tracks", "Negative intercepts"]
    rows = [[entry["key"], shown, float(a), float(b), int(c), bool(neg)]
            for a, b, c, neg in result["histogram"]["bins"]]
    table = ("Beside v2", "What SMTracker 2 printed for this data, beside this page's numbers", (),
             ["Condition", "Shown as", "What", "Value"],
             [[entry["key"], shown, what, value] for what, value in v2_rows(result)])
    return (columns, rows), (table,)


def _msd_v2_notes(entry: dict) -> tuple:
    hist = entry["result"]["histogram"]
    return ("SMTracker 2's localisation-error panel: a line is fitted to each track's MSD over lags 1-7 (1/std "
            "weights, last lag dropped) and sigma = sqrt(|b|/4) of its intercept b. SMTracker 2 took the absolute "
            "value of negative intercepts; here they keep their sign and are drawn left of zero.",
            f"{hist['n']:,} tracks; {hist['beyond']:,} beyond the drawn range are counted in the end bins.",
            "An intercept is not sigma: for a free molecule imaged continuously it is 2·ndim·(σ² − D·τ/3), so "
            "motion blur lowers it and the cell wall raises it.")


def msd_v2_legend(entry: dict) -> str:
    from .msd_view import v2_rows

    rows = dict(v2_rows(entry["result"]))
    parts = [f"SMTracker 2's localisation-error panel for {entry['name']}: per-track sigma = sqrt(|b|/4) from the "
             "intercept b of a line fitted to each track's MSD over lags 1-7, in 20 nm bins. Negative intercepts, "
             "which SMTracker 2 folded to positive, are drawn left of zero (grey)."]
    panel = rows.get("v2's σ panel")
    if panel:
        parts.append(f"v2's panel reads {panel}.")
    parts.append("An intercept is not sigma: motion blur lowers it and the cell wall raises it.")
    return " ".join(parts)


# -- Clustering: the shared clusters' shares per condition (7 Nov 2026) --------------------------------------------- #
def _draw_cluster_shares(fig, ax, payload: dict, spec: FigureSpec, family: str) -> None:
    """Per condition, each shared cluster's share of tracks: one dot per replicate, the mean as a bar and the SD as
    whiskers, in the cluster inks the screen uses (`clustering_view.cluster_inks`: the palette, fitted to the page, slowest first). The data is
    `clustering_compare.shares_payload`, the screen's own."""
    from .clustering_view import cluster_inks

    shares = payload["shares"]
    columns, k = shares["columns"], shares["k"]
    inks = cluster_inks(k, spec.dark)
    if not columns:
        raise FigureExportError("there are no shared-cluster shares to draw")
    width = 0.8 / k
    names = []
    for i, col in enumerate(columns):
        names.append(spec.shown_name(col["name"]))
        for j in range(k):
            ink = inks[j]
            x = i + (j - (k - 1) / 2) * width
            mean = 100.0 * col["mean"][j]
            ax.hlines(mean, x - width * 0.35, x + width * 0.35, color=ink, lw=max(MIN_LINE_PT, spec.line_pt() * 1.6),
                      zorder=3)
            if col["sd"] is not None:
                sd = 100.0 * col["sd"][j]
                ax.errorbar([x], [mean], yerr=[[min(sd, mean)], [sd]], color=ink, lw=max(MIN_LINE_PT, spec.line_pt()),
                            capsize=2.0, zorder=3, fmt="none")
            m = len(col["replicates"])
            xs = [x + (r - (m - 1) / 2) * min(width / 4, 0.06) for r in range(m)]
            ax.scatter(xs, [100.0 * rep["shares"][j] for rep in col["replicates"]], s=max(4.0, spec.line_pt() * 14),
                       color=ink, alpha=0.75, edgecolors=col.get("colour") or "none", linewidths=0.8, zorder=4,
                       label=f"cluster {j + 1}" if i == 0 else None)
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(names)
    ax.set_xlim(-0.6, len(columns) - 0.4)
    ax.set_ylim(bottom=0.0)
    ax.set_ylabel(_title(spec, "clusters_shares", "y"))
    if spec.legend:
        handles, labels = ax.get_legend_handles_labels()
        _legend_above(ax, handles, labels, spec, family, title=None)
    _qc_grid(ax, spec)


def _clusters_shares_source(payload: dict, spec: FigureSpec) -> tuple:
    shares = payload["shares"]
    columns = ["Condition", "Shown as", "Replicate", "Cluster", "Share of tracks (%)", "Tracks in replicate"]
    rows = []
    for col in shares["columns"]:
        shown = spec.shown_name(col["name"])
        for rep in col["replicates"]:
            for j, share in enumerate(rep["shares"], start=1):
                rows.append([col["key"], shown, rep["replicate"], j, 100.0 * float(share), int(rep["n_tracks"])])
    summary_rows = []
    for col in shares["columns"]:
        shown = spec.shown_name(col["name"])
        for j in range(shares["k"]):
            summary_rows.append([col["key"], shown, j + 1, 100.0 * float(col["mean"][j]),
                                 None if col["sd"] is None else 100.0 * float(col["sd"][j]), int(col["n"])])
    table = ("Summary", "Mean and SD over replicates of each cluster's share of tracks, per condition",
             (), ["Condition", "Shown as", "Cluster", "Mean (%)", "SD over replicates (%)", "Replicates"],
             summary_rows)
    s = shares["settings"]
    settings = ("Settings", "The Run that made these shares", (), ["Setting", "Value"],
                [["Lags", s["n_lags"]], ["Feature", s["feature"]], ["Clusters", "shared (centroids fitted once on the "
                 "pooled tracks of every condition)"], ["k", shares["k"]]])
    return (columns, rows), (table, settings)


def clusters_shares_legend(payload: dict) -> str:
    return ("Clustering with shared clusters: k-means centroids are fitted once on the pooled tracks of every "
            "condition and each condition's tracks are assigned to them. Clusters are numbered from the slowest. "
            + payload["shares"]["caption"]
            + " " + " ".join(str(line) for line in payload["card"]["lines"][:1]))


# -- Clustering: a condition's cluster curves (4 Nov 2026, rail review B3) ------------------------------------------ #
def _draw_clusters(fig, ax, entry: dict, spec: FigureSpec, family: str) -> None:
    """The card's chart in print: each cluster's ensemble MSD against lag time, a line and its points, in the
    cluster inks the screen uses (`clustering_view.CLUSTER_INKS`, slowest first)."""
    from .clustering_view import cluster_inks

    curves = entry["result"]["curves"]
    if not curves:
        raise FigureExportError(f"{entry.get('name', 'this condition')} has no cluster curves to draw")
    inks = cluster_inks(len(curves), spec.dark)
    for i, curve in enumerate(curves):
        ink = inks[i]
        ax.plot([t for t, _m in curve], [m for _t, m in curve], color=ink, lw=max(MIN_LINE_PT, spec.line_pt()),
                marker="o", ms=max(2.0, spec.line_pt() * 2.4), zorder=3, label=f"cluster {i + 1}")
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(_title(spec, "clusters_condition", "x"))
    ax.set_ylabel(_title(spec, "clusters_condition", "y"))
    if spec.legend:
        handles, labels = ax.get_legend_handles_labels()
        _legend_above(ax, handles, labels, spec, family, title=_keyed_title(spec, entry["name"]))
    _qc_grid(ax, spec)


def _clusters_source(entry: dict, spec: FigureSpec) -> tuple:
    result = entry["result"]
    shown = spec.shown_name(entry["name"])
    columns = ["Condition", "Shown as", "Cluster", "Lag (s)", "MSD (µm²)"]
    rows = [[entry["key"], shown, i + 1, float(t), float(m)]
            for i, curve in enumerate(result["curves"]) for t, m in curve]
    table = ("Clusters", "Each cluster: its tracks, share, D, alpha and R² of the early-lag fit",
             (), ["Condition", "Shown as", *result["columns"]],
             [[entry["key"], shown, *row] for row in result["rows"]])
    return (columns, rows), (table,)


def clusters_legend(entry: dict) -> str:
    lines = " ".join(str(line) for line in entry["result"]["lines"])
    return (f"Clustering of {entry['name']}: k-means on each track's MSD curve. Each line is one cluster's ensemble "
            f"MSD against lag time, with clusters numbered from the slowest. {lines}")


# -- Apparent Diffusion: the Distributions, Per cell and Checks pages ---------------------------------------------- #
#: Asked 16 Oct 2026 ("build appd export"). Each figure draws from the SAME page helper its card paints from
#: (`appd_view.histogram_payload`, `cells_payload`, `switching_payload`, `length_payload`), so print and screen
#: cannot disagree. The section's first "appd_distribution" kind (violins over whole-track means) went with the
#: first section.

def _appd_fits(result: dict) -> dict:
    """An apparent-D result's CONVERGED fits, keyed by their population count (`appd_view.converged_fits`)."""
    from .appd_view import converged_fits

    return converged_fits(result)


def _appd_shown(entry: dict, spec: FigureSpec) -> tuple:
    """``(the counts drawn -- a tuple, ascending --, how they were chosen)`` for a Distributions figure: the Export
    dialog's Fit row when set, else the card's Show (an int ``"selected"`` from before 3 Nov 2026 reads as that one
    fit), else the Run's own count. A count asked for that did not converge is refused by name. Since 3 Nov 2026 the
    figure draws a SET of fits, as the Jump distances figures do (`_chosen`)."""
    from .appd_view import default_show
    from .jumps_view import as_shown, show_counts

    result = entry["result"]
    fits = _appd_fits(result)
    name = entry.get("name", "this condition")
    if not fits:
        raise FigureExportError(f"{name} has no apparent-D fit to draw")
    if spec.chosen_fit is not None:
        asked = show_counts(spec.chosen_fit, fits)
        missing = [k for k in asked if k not in fits]
        if missing:
            raise FigureExportError(f"{name} has no {_plural(missing[0])} fit (converged: "
                                    f"{', '.join(str(k) for k in sorted(fits))})")
        return asked, "in the Export dialog"
    shown = as_shown(entry.get("selected"), fits)
    if shown:
        return shown, "on the card"
    k = result.get("n_pop")
    if k is not None and int(k) in fits:
        return (int(k),), "by BIC" if result.get("chosen_by") == "best" else "in the Run"
    return default_show(result), "the fewest populations that converged"


def _appd_strips(spec: FigureSpec, kind: str, entries=()) -> list:
    """The residual plots an apparent-D distribution figure draws under its histogram, ``["pearson", "ecdf"]`` or
    fewer: the Export dialog's boxes when set, else -- on a card's figure -- what its card shows (its ``"residuals"``),
    else the figure's own default (`RESIDUALS_BY_DEFAULT`: the card's on, the All conditions figure's off)."""
    out = []
    for name in ("pearson", "ecdf"):
        value = getattr(spec, f"{name}_residuals")
        if value is None:
            card = [e.get("residuals") for e in entries or () if e.get("residuals") is not None]
            on = bool(card[0]) if (card and kind == "appd_histogram") else RESIDUALS_BY_DEFAULT.get(kind, True)
        else:
            on = bool(value)
        if on:
            out.append(name)
    return out


def _appd_pearson_strip(strip, h: dict, spec: FigureSpec, span: float) -> None:
    """THE PEARSON RESIDUALS under an apparent-D histogram (`appd_view.distribution_payload`'s fits), ``span`` either
    side of 0: one fit's as bars, beyond +-2 in the flag colour, over the +-2 band; two or more each a line in its
    colour and v2's style. A bin expecting under one track carries none (`appd_view.PEARSON_MIN_EXPECTED`)."""
    from matplotlib.ticker import FixedLocator

    page, line, lw = _page(spec), spec.line_pt(), _jump_widths(spec)
    edges, drawn = h["edges"], list(h["show"])
    strip.axhspan(-2.0, 2.0, color=_pale(spec, _QC_GREY, 0.16), lw=0, zorder=0.5)
    strip.axhline(0.0, color=page["grid"], lw=max(MIN_LINE_PT, line * 0.6), zorder=1)
    if h["every"]:
        for k in drawn:
            strip.plot(h["centres"], [float("nan") if v is None else v for v in h["fits"][k]["pearson"]],
                       _JUMP_STYLES.get(k, "-"), color=_fit_colour(k, spec), lw=lw["residual"], zorder=2)
    else:
        kept = [(a, b, v) for a, b, v in zip(edges, edges[1:], h["fits"][drawn[0]]["pearson"]) if v is not None]
        if kept:
            strip.bar([a for a, _b, _v in kept], [v for _a, _b, v in kept], width=[b - a for a, b, _v in kept],
                      align="edge", linewidth=0, zorder=2,
                      color=[chart_text.FLAG_COLOR if abs(v) > 2 else _pale(spec, page["ink"], 0.6)
                             for _a, _b, v in kept])
    strip.set_ylim(-span, span)
    strip.yaxis.set_major_locator(FixedLocator(_symmetric_ticks(span, 2.0)))


def _appd_spans(h: dict) -> dict:
    """How far an apparent-D figure's residual plots reach either side of 0: the Pearson at least 3 (the +-2 band
    shows), the ECDF at least 0.005."""
    pearson = [abs(v) for k in h["show"] for v in h["fits"][k]["pearson"] if v is not None]
    ecdf = [abs(v) for k in h["show"] for _x, v in h["fits"][k]["ecdf_residual"]]
    return {"pearson": max(3.0, max(pearson, default=0.0) * 1.1), "ecdf": max(0.005, max(ecdf, default=0.0) * 1.1)}


def _appd_ecdf_fits(h: dict) -> dict:
    """The fits drawn in the shape `_ecdf_strip` reads (``{count: {"ecdf_residual"}}``)."""
    return {k: {"ecdf_residual": h["fits"][k]["ecdf_residual"]} for k in h["show"]}


def _log_bars(edges) -> tuple:
    """``(lefts, widths)`` of histogram bars on a log axis, each a little narrower than its bin (a gap that
    looks the same at every decade)."""
    lefts, widths = [], []
    for a, b in zip(edges, edges[1:]):
        r = b / a
        lefts.append(a * r ** 0.04)
        widths.append(a * r ** 0.96 - a * r ** 0.04)
    return lefts, widths


def _log_x(ax, lo: float, hi: float, sparse: bool = False) -> None:
    """A log x axis over [lo, hi], ticked at whole decades (at 1, 2, 5 per decade when it spans fewer than
    three), the labels plain numbers. ``sparse`` (a narrow plot at a large font: `_narrow_for_font`): whole decades
    whenever two fall inside, so neighbouring labels do not meet."""
    from matplotlib.ticker import FixedLocator, NullLocator

    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ticks = chart_text.log_ticks(lo, hi)
    decades = [v for v in ticks if abs(math.log10(v) - round(math.log10(v))) < 1e-9]
    if len(decades) >= (2 if sparse else 3):
        ticks = decades
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.set_xticklabels([chart_text.tick_label(v) for v in ticks])
    ax.xaxis.set_minor_locator(NullLocator())


def _narrow_for_font(plot_mm: float, spec: FigureSpec) -> bool:
    """Whether a log axis ``plot_mm`` wide is narrow for its tick labels' size: under 8 mm per point of the small
    text (a slide's 85 mm column at 15 pt is; a journal's 55 mm third at 6 pt is not)."""
    return plot_mm / max(1.0, spec.small_pt()) < 8.0


def _keyed_title(spec: FigureSpec, name: str):
    """The condition's shown name over a key -- or None when the figure's title already says it."""
    title = spec.shown_name(name)
    return None if (spec.title and title and title in spec.title) else title


def _appd_marks(h: dict, page: dict) -> list:
    """The vertical marks of an apparent-D histogram that fall inside its bins: where the Run's populations cross
    (only with that fit drawn alone), what an immobile molecule reads, the detection ceiling. Short words for the
    key (the drafted legend says each in full): long ones wrapped beside a slide's plot and ran off its top."""
    lo, hi = h["edges"][0], h["edges"][-1]
    marks = [(cut, (0, (3.0, 2.0)), page["ink"], "cut") for cut in h["cuts"]]
    marks += [(h["immobile"], ":", page["muted"], "immobile"), (h["ceiling"], "-.", page["muted"], "ceiling")]
    return [m for m in marks if m[0] is not None and lo <= m[0] <= hi]


def _draw_appd_curves(ax, h: dict, spec: FigureSpec, colour: str) -> None:
    """The bars and the fits of one apparent-D histogram (`appd_view.distribution_payload`): one fit alone in the
    condition's strong shade with its populations dashed in its shades; two or more each in its own colour and v2's
    line style (`_fit_colour`, `_JUMP_STYLES`), their populations thin and dashed in that colour when Components is
    on. The card's figure and the All conditions figure draw alike."""
    line, lw = spec.line_pt(), _jump_widths(spec)
    lefts, widths = _log_bars(h["edges"])
    ax.bar(lefts, h["observed"], width=widths, align="edge", color=_pale(spec, colour, 0.45), linewidth=0, zorder=1)
    drawn, every = list(h["show"]), h["every"]
    for k in drawn:
        fit = h["fits"][k]
        if every:
            if k in h["parts"]:
                for comp in fit["components_drawn"]:
                    ax.plot(h["centres"], [float("nan") if v is None else v for v in comp], ls=_COMPONENT_DASH,
                            color=_fit_colour(k, spec), lw=lw["component_many"], alpha=0.9, zorder=3)
            ax.plot(h["centres"], fit["expected"], _JUMP_STYLES.get(k, "-"), color=_fit_colour(k, spec),
                    lw=lw["fit"], zorder=4)
            continue
        n_comp = len(fit["components"])
        ax.plot(h["centres"], fit["expected"], "-", color=_strong(spec, colour), lw=max(0.8, line * 1.5), zorder=4)
        if k in h["parts"]:
            # over the sum, so a population that IS the sum where it lies still shows as its own dashes
            for c, comp in enumerate(fit["components_drawn"]):
                ax.plot(h["centres"], [float("nan") if v is None else v for v in comp], ls=(0, (4.0, 2.5)),
                        color=figstyle.day_shade(colour, c, max(1, n_comp)), lw=max(0.6, line * 1.1), zorder=5)


def _draw_appd_histogram(fig, entry: dict, spec: FigureSpec, family: str):
    """One condition's Distributions card in print: tracks per bin on a log apparent-D axis, the fits drawn
    (`_appd_shown`: the card's Show, else the dialog's Fit row; `_draw_appd_curves`), and the marks -- where the
    Run's populations cross (with that fit drawn alone), what an immobile molecule reads, the detection ceiling. The
    key names the marks rather than labels on the plot, which met each other where the marks lie close.

    Since 3 Nov 2026 the fits drawn are a SET, as on the card, and UNDER THE HISTOGRAM its residual plots, each
    optional (`_appd_strips`: the Pearson residual per bin and the ECDF residual, on the same log axis; the lowest
    plot carries the x axis). The curves are the POOLED fit of every track, which the legend says. Returns the
    histogram's axes: the dialog's titles and ranges are its (the x range is shared by every plot)."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    from .appd_view import distribution_payload

    result = entry["result"]
    shown, _how = _appd_shown(entry, spec)
    h = distribution_payload(result, shown, components=_components_on(entry, spec))
    drawn, every = list(h["show"]), h["every"]
    one = drawn[0]
    colour = entry.get("colour") or QC_DATA
    page = _page(spec)
    line, small = spec.line_pt(), spec.small_pt()
    edges = h["edges"]
    lo, hi = edges[0], edges[-1]
    strips = [s for s in _appd_strips(spec, "appd_histogram", [entry]) if s != "ecdf" or h["ecdf_ok"]]
    marks = _appd_marks(h, page)
    fit = h["fits"][one]
    n_comp = len(fit["components"])
    numbers = [f"D {d:.3g} · {100 * f:.1f} %" for d, f in zip(fit["D"], fit["fraction"])]
    handles, labels = [], []
    if spec.legend:
        handles, labels = [Patch(color=_pale(spec, colour, 0.45), lw=0)], ["tracks"]
        if every:
            # the fits drawn and nothing else (each one's populations are said in the legend's words)
            handles += [Line2D([], [], ls=_JUMP_STYLES.get(k, "-"), color=_fit_colour(k, spec),
                               lw=_jump_widths(spec)["fit"]) for k in drawn]
            labels += [f"{_plural(k)} fit" for k in drawn]
        else:
            handles.append(Line2D([], [], color=_strong(spec, colour), lw=max(0.8, line * 1.5)))
            labels.append(f"{_plural(one)} fit" + (f": {numbers[0]}" if spec.numbers and n_comp == 1 else ""))
            if one in h["parts"]:
                handles += [Line2D([], [], ls=(0, (4.0, 2.5)), color=figstyle.day_shade(colour, c, max(1, n_comp)),
                                   lw=max(0.6, line * 1.1)) for c in range(n_comp)]
                labels += [numbers[c] if spec.numbers else f"population {c + 1}" for c in range(n_comp)]
        handles += [Line2D([], [], ls=style, color=c, lw=max(MIN_LINE_PT, line)) for _d, style, c, _w in marks]
        labels += [f"{words} {d:.2g}" if spec.numbers else words for d, _st, _c, words in marks]
    width_mm, height_mm = spec.size_mm()
    title = _keyed_title(spec, entry["name"])
    wide = width_mm / height_mm >= _APPD_CHECKS_WIDE
    # A WIDE figure (a slide, two columns) keeps its key BESIDE the plot: above it, seven entries at a slide's
    # 16 pt took most of the height and the y labels met (found in the first render, 16 Oct 2026). With residual
    # plots it is laid out as the wide jump figure instead (3 Nov 2026): stacked under a slide's histogram, their
    # tick labels met; the histogram on the left, the residual plots stacked on the right, the key in a row of its
    # own over both.
    beside = bool(handles) and wide and not strips
    rows = ["histogram", *strips]
    ratios = [_JUMPS_SHARES[r] for r in rows]
    if strips and wide:
        shares = [_JUMPS_BESIDE[n] for n in strips]
        first = 0
        if handles:
            labels, ncol = _legend_columns(labels, width_mm / _PT_MM * 0.94, spec, family)
            lines = math.ceil(len(labels) / ncol) + (1 if title else 0)
            tallest = max(text.count("\n") + 1 for text in labels)
            key = (lines * tallest * small * 1.55 + small) * _PT_MM
            plot = max(20.0, height_mm - (2.4 * spec.pt() + 2.0 * small) * _PT_MM - 2.0)
            rest = max(10.0, plot - key)
            grid = fig.add_gridspec(1 + len(strips), 2, width_ratios=(1.3, 1.0),
                                    height_ratios=(key, *(rest * r / sum(shares) for r in shares)))
            holder = fig.add_subplot(grid[0, :])
            holder.set_axis_off()
            holder.legend(handles, labels, loc="center", ncol=ncol, frameon=False, handletextpad=0.4,
                          columnspacing=1.0, borderaxespad=0.0, handlelength=1.8,
                          **({"title": title, "title_fontproperties": {"size": small, "weight": "bold"}}
                             if title else {}))
            handles, first = [], 1
        else:
            grid = fig.add_gridspec(len(strips), 2, width_ratios=(1.3, 1.0), height_ratios=shares)
        ax = fig.add_subplot(grid[first:, 0])
        under = {name: fig.add_subplot(grid[first + i, 1], sharex=ax) for i, name in enumerate(strips)}
    elif beside:
        measure = lambda text: _text_width_pt(text, small, family)               # noqa: E731
        whole = width_mm / _PT_MM
        room = whole * 0.38
        labels = ["\n".join(chart_text.wrap_text(text, measure, room - small * 3.4))
                  if measure(text) + small * 3.0 > room else text for text in labels]
        need = max(max(measure(part) for part in text.split("\n")) for text in labels) + small * 3.4
        share = min(0.4, need / whole)
        grid = fig.add_gridspec(len(rows), 2, width_ratios=(1.0 - share, share), height_ratios=ratios)
        ax = fig.add_subplot(grid[0, 0])
        holder = fig.add_subplot(grid[:, 1])
        holder.set_axis_off()
        holder.legend(handles, labels, loc="center left", frameon=False, handletextpad=0.4, borderaxespad=0.0,
                      handlelength=1.8, **({"title": title, "title_fontproperties": {"size": small, "weight": "bold"}}
                                          if title else {}))
        under = {}
    else:
        grid = fig.add_gridspec(len(rows), 1, height_ratios=ratios)
        ax = fig.add_subplot(grid[0, 0])
        under = {name: fig.add_subplot(grid[1 + i, 0], sharex=ax) for i, name in enumerate(strips)}
    _draw_appd_curves(ax, h, spec, colour)
    for d, style, c, _words in marks:
        ax.axvline(d, ls=style, color=c, lw=max(MIN_LINE_PT, line), zorder=6)
    # side by side the residual column is 1/2.3 of the width: its tick labels decide (`_narrow_for_font`)
    _log_x(ax, lo, hi, sparse=_narrow_for_font(width_mm * (1.0 / 2.3 if strips and wide else 1.0), spec))
    ax.set_ylim(0, h["y_max"])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.set_ylabel(_title(spec, "appd_histogram", "y"))
    spans = _appd_spans(h)
    if "pearson" in under:
        _appd_pearson_strip(under["pearson"], h, spec, spans["pearson"])
        under["pearson"].set_ylabel(chart_text.PEARSON_LABEL.replace(" ", "\n", 1))
    if "ecdf" in under:
        _ecdf_strip(under["ecdf"], _appd_ecdf_fits(h), drawn, spec, spans["ecdf"])
        under["ecdf"].set_ylabel(chart_text.ECDF_RESIDUAL_LABEL.replace(" ", "\n", 1))
    # THE X AXIS: under the lowest plot of the stack; side by side, under the histogram and the lowest residual plot,
    # whose narrow column wraps its title rather than running off
    x_title = _title(spec, "appd_histogram", "x")
    side = bool(strips) and wide
    stack = list(under.values()) if side else [ax, *under.values()]
    for upper in stack[:-1]:
        upper.tick_params(labelbottom=False)
    if side:
        ax.set_xlabel(x_title)
        if x_title:
            measure = lambda s: _text_width_pt(s, spec.pt(), family)                  # noqa: E731
            x_title = "\n".join(chart_text.wrap_text(x_title, measure, width_mm / _PT_MM * 0.3, max_lines=2))
    stack[-1].set_xlabel(x_title)
    if handles and not beside:
        _legend_above(ax, handles, labels, spec, family, title=title)
    elif not spec.legend and spec.numbers and not every:
        ax.annotate(f"{_plural(one)}:\n" + "\n".join(numbers), (0.98, 0.96), xycoords="axes fraction",
                    ha="right", va="top", fontsize=spec.small_pt(), color=page["ink"])
    for one_ax in (ax, *under.values()):
        _qc_grid(one_ax, spec)
    if under:
        fig.align_ylabels(list(under.values()) if side else [ax, *under.values()])
    return ax


def _appd_histogram_height(spec: FigureSpec, strips) -> float:
    """An apparent-D distribution figure's own height: the preset's with the histogram alone (as before 3 Nov 2026);
    with residual plots under it taller by most of their share (`_JUMPS_SHARES`, the jump figure's), at most the
    journal's maximum; on a slide or poster 3:4 up to 200 mm wide, as the jump figure."""
    default = spec.default_height_mm()
    if not strips:
        return default
    tallest = spec.preset_values.get("max_height_mm") or 600.0
    if spec.preset in LARGE_PRESETS:
        width = spec.width_value()
        return round(min(tallest, max(default, width * 0.75)), 1) if width <= 200.0 else default
    share = sum(_JUMPS_SHARES[n] for n in ("histogram", *strips)) / _JUMPS_SHARES["histogram"]
    return round(min(tallest, default * (1.0 + (share - 1.0) * 0.8)), 1)


def _appd_quality_columns(result: dict) -> tuple:
    """``(columns, {count: [values]})``: the fit-quality numbers a source sheet adds per count (`appd_view.fit_quality`):
    the largest ECDF gap, the held-out log-likelihood against the best and the folds won; empty without them."""
    quality = result.get("quality") or {}
    if not quality:
        return [], {}
    cv = quality.get("cv") or {}
    unit = {"day": "replicates", "cell": "cells"}.get(cv.get("unit"), "units")
    columns = ["Largest ECDF gap (0-1)", f"Held-out ΔlogL ({unit})", "Folds won", "Folds"]
    values = {}
    for k in _appd_fits(result):
        gap = ((quality.get("fits") or {}).get(k) or {}).get("ks")
        values[k] = [gap, (cv.get("delta_total") or {}).get(k), (cv.get("wins") or {}).get(k),
                     cv.get("n_folds")]
    return columns, values


def _appd_histogram_source(entry: dict, spec: FigureSpec) -> tuple:
    """Per bin: its edges, the tracks in it, and for the fits drawn the expected tracks (one fit: each population and
    their sum; several: each one's sum) and the Pearson residual; every fit made with its BIC and, since 3 Nov 2026,
    its fit quality (the largest ECDF gap, the held-out fits) as a second sheet; the ECDF residuals of the fits drawn
    as a third."""
    from .appd_view import distribution_payload

    result = entry["result"]
    shown, _how = _appd_shown(entry, spec)
    h = distribution_payload(result, shown, components=True)
    drawn = list(h["show"])
    name = spec.shown_name(entry["name"])
    columns = ["Condition", "Shown as", "Bin from (µm²/s)", "Bin to (µm²/s)", "Tracks"]
    per_fit = []
    for k in drawn:
        fit = h["fits"][k]
        if not h["every"] and len(fit["components"]) > 1:
            per_fit += [(f"Expected, population {c + 1} of {k}", comp) for c, comp in enumerate(fit["components"])]
        per_fit.append((f"Expected, the {_plural(k)} fit", fit["expected"]))
        per_fit.append((f"Pearson residual, the {_plural(k)} fit", fit["pearson"]))
    columns += [title for title, _v in per_fit]
    edges = h["edges"]
    rows = [[entry["key"], name, float(edges[i]), float(edges[i + 1]), int(count)]
            + [None if values[i] is None else float(values[i]) for _t, values in per_fit]
            for i, count in enumerate(h["observed"])]
    fits = result.get("fits") or {}
    selection = {int(r["n_pop"]): r for r in (result.get("selection") or {}).get("rows", [])}
    widest = max([len(f["D"]) for f in _appd_fits(result).values()] or [1])
    quality_columns, quality = _appd_quality_columns(result)
    fit_columns = (["Populations"] + [f"D{c + 1} (µm²/s)" for c in range(widest)]
                   + [f"Fraction {c + 1} (0-1)" for c in range(widest)]
                   + ["BIC", "ΔBIC", "Did not converge", "Drawn"] + quality_columns)
    fit_rows = []
    for k in sorted(int(k) for k in fits):
        f = fits.get(k) or fits.get(str(k)) or {}
        D = list(f.get("D") or [])
        fr = list(f.get("fraction") or [])
        row = selection.get(k, {})
        fit_rows.append([k] + [float(v) for v in D] + [None] * (widest - len(D))
                        + [float(v) for v in fr] + [None] * (widest - len(fr))
                        + [row.get("bic"), row.get("delta"), f.get("failed") or row.get("failed") or "",
                           k in drawn]
                        + (quality.get(k) or [None] * len(quality_columns) if quality_columns else []))
    notes = ["The pooled fit of every track (one apparent D per track, each with its own gamma shape).",
             "BIC: n is the number of tracks. A runner-up within 2 of the best is inconclusive. BIC recommends a "
             "count; the choice is yours."]
    if quality_columns:
        from .appd_view import quality_table

        notes += [n for n in quality_table(result)["notes"] if not n.startswith("ΔBIC")]
    fits_sheet = ("Fits", "Every population count fitted, its BIC and its fit quality", tuple(notes), fit_columns,
                  fit_rows)
    more = [fits_sheet]
    if h["ecdf_ok"]:
        ecdf_columns = ["Condition", "Populations", "Apparent D (µm²/s)", "Model CDF - ECDF"]
        ecdf_rows = [[entry["key"], k, float(x), float(v)] for k in drawn for x, v in h["fits"][k]["ecdf_residual"]]
        more.append(("ECDF residuals", "The model's CDF minus the empirical CDF of the per-track apparent D, for the "
                                       "fits drawn", (), ecdf_columns, ecdf_rows))
    return (columns, rows), tuple(more)


def _appd_definition(result: dict) -> str:
    settings = result.get("settings") or {}
    shortest = int(settings.get("min_steps", 4)) + 1
    return (f"Apparent D: one value per track, the mean of its single-frame squared jumps divided by 4τ "
            f"(τ = {1000 * result['tau_s']:.1f} ms), from {result['n_tracks']:,} tracks of at least {shortest} "
            f"points. The mixture is fitted with each track's own gamma shape (set by its number of steps).")


def _appd_histogram_notes(entry: dict, spec: FigureSpec) -> tuple:
    result = entry["result"]
    shown, how = _appd_shown(entry, spec)
    drawn = ("the " + _plural(shown[0]) + " fit") if len(shown) == 1 else ("the " + _shown_words(shown) + " fits")
    notes = [_appd_definition(result),
             f"Drawn: {drawn} ({how}), the pooled fit of every track. Expected: the tracks the fit predicts in a "
             f"bin, the sum over tracks of each track's own gamma probability for the bin."]
    if len(shown) == 1 and shown[0] == result.get("n_pop") and result.get("cuts_um2_s"):
        notes.append("Cut: where the populations cross, so a track there is equally likely in either, at "
                     + ", ".join(f"{c:.3g}" for c in result["cuts_um2_s"])
                     + f" µm²/s for a track of {result.get('cut_n_steps', '?')} steps (it depends on length).")
    notes.append(f"Immobile: the value an immobile molecule reads from localisation error alone, "
                 f"{result['immobile_um2_s']:.3g} µm²/s. Ceiling: the tracker's linking limit, "
                 f"{result['ceiling_um2_s']:.3g} µm²/s; faster molecules are lost or truncated.")
    if result.get("days") and result["days"].get("D"):
        from .appd_view import pooled_note

        notes.append(pooled_note(result).replace("Curves: ", "The curves are "))
    if "ecdf" in _appd_strips(spec, "appd_histogram", [entry]) and not (result.get("quality") or {}).get("fits"):
        notes.append("ECDF residuals are not drawn: this result has none. Run the section again.")
    if result.get("n_too_short"):
        notes.append(f"{result['n_too_short']:,} shorter tracks left out.")
    return tuple(notes)


def _appd_strip_words(strips) -> str:
    """The residual plots under a histogram, in the legend's words ("" when none)."""
    parts = []
    if "pearson" in strips:
        parts.append("the Pearson residual per bin, (observed - expected) / sqrt(expected), about ±1 if every track "
                     "were independent (bins expecting under one track have none; residuals beyond ±2 are red "
                     "when one fit is drawn)")
    if "ecdf" in strips:
        parts.append("the model's cumulative distribution minus the data's, over the per-track values, unbinned")
    return ("Under the histogram: " + "; then ".join(parts) + ".") if parts else ""


def appd_histogram_legend(entry: dict, spec: FigureSpec | None = None) -> str:
    """A drafted legend for one condition's apparent-D distribution: the fits drawn (one, or several in their
    colours), the pooled fit's words, the residual plots under it, BIC and the held-out fits."""
    from .appd_view import bic_text, quality_verdict

    spec = spec or FigureSpec()
    result = entry["result"]
    shown, _how = _appd_shown(entry, spec)
    fits = _appd_fits(result)
    parts = [f"Apparent diffusion coefficients of {entry['name']} on a logarithmic axis. {_appd_definition(result)}"]
    if len(shown) == 1:
        fit = fits[shown[0]]
        comps = "; ".join(f"D {d:.3g} µm²/s, {100 * f:.1f} %" for d, f in zip(fit["D"], fit["fraction"]))
        parts.append(f"Bars: tracks per bin. Lines: the {_plural(shown[0])} gamma mixture fitted to every track "
                     f"pooled ({comps})"
                     + ("; dashed, each population; solid, their sum" if len(fit["D"]) > 1 else "")
                     + ", as expected tracks per bin.")
        parts.append("Vertical lines: where the populations cross, the value an immobile molecule reads from "
                     "localisation error alone, and the tracker's detection ceiling.")
    else:
        lines = "; ".join(f"{_plural(k)}, {_JUMP_STYLE_WORDS.get(k, 'line')} {_fit_colour_word(k)}" for k in shown)
        parts.append(f"Bars: tracks per bin. Lines: the gamma mixtures of {_shown_words(shown)} fitted to every "
                     f"track pooled, as expected tracks per bin ({lines})"
                     + ("; thin dashes, each fit's populations" if _components_on(entry, spec) else "") + ".")
        parts.append("Vertical lines: the value an immobile molecule reads from localisation error alone, and the "
                     "tracker's detection ceiling.")
    strips = [s for s in _appd_strips(spec, "appd_histogram", [entry])
              if s != "ecdf" or (result.get("quality") or {}).get("fits")]
    if strips:
        parts.append(_appd_strip_words(strips))
    top = max(fits) if fits else 3
    parts.append(bic_text(result).replace("BIC:", f"BIC over 1 to {top} populations:") + "; BIC recommends a "
                 "count, the choice is yours.")
    verdict = quality_verdict(result)
    if "held-out" in verdict:
        parts.append(verdict.split(" · ")[-1][0].upper() + verdict.split(" · ")[-1][1:]
                     + ", each replicate (or fold of cells) fitted without its tracks and scored on them.")
    parts.append("Apparent D averages over switching within a track, so its populations are blurrier than the SQD "
                 "fit's, which is the main readout.")
    return " ".join(parts)


def _fit_colour_word(k: int) -> str:
    """Fit ``k``'s colour in words (`jumps_view.FIT_COLOUR_WORDS`)."""
    from .jumps_view import FIT_COLOUR_WORDS

    return FIT_COLOUR_WORDS.get(int(k), "grey")


# -- every condition's apparent-D distribution in one figure (3 Nov 2026) -------------------------------------------
def _appd_overview_entries(payload: dict) -> list:
    """The overview's entries, each with a fit to draw; refused by name otherwise."""
    entries = (payload or {}).get("entries") or []
    if not entries:
        raise FigureExportError("no apparent-D result to draw yet")
    for e in entries:
        if "error" in (e.get("result") or {}) or not _appd_fits(e.get("result") or {}):
            raise FigureExportError(f"{e.get('name', 'a condition')} has no apparent-D fit to draw")
    return entries


def _appd_overview_shown(entry: dict, spec: FigureSpec) -> tuple:
    """The counts one panel of the overview draws: the dialog's Fit row (those of them this condition has), else its
    card's Show (`_appd_shown`)."""
    from .jumps_view import as_shown

    if spec.chosen_fit is not None:
        mine = as_shown(spec.chosen_fit, _appd_fits(entry["result"]))
        if mine:
            return mine
    return _appd_shown(entry, replace(spec, chosen_fit=None))[0]


def _appd_overview_payloads(payload: dict, spec: FigureSpec) -> tuple:
    """``(entries, {key: distribution payload}, strips)``: what each panel draws, and the residual plots every panel
    draws (the ECDF residuals only where every result carries them)."""
    from .appd_view import distribution_payload

    entries = _appd_overview_entries(payload)
    drawn = {e["key"]: distribution_payload(e["result"], _appd_overview_shown(e, spec),
                                            components=_components_on(e, spec)) for e in entries}
    strips = [s for s in _appd_strips(spec, "appd_overview")
              if s != "ecdf" or all(h["ecdf_ok"] for h in drawn.values())]
    return entries, drawn, strips


def _appd_overview_height(payload: dict, spec: FigureSpec) -> float:
    """Each panel about 0.72 times as tall as wide (taller by its residual plots), plus its title, and the key's row
    above them -- balanced across the width as the jump overview's (`_overview_grid`)."""
    width = spec.width_value()
    try:
        _entries, _drawn, strips = _appd_overview_payloads(payload, spec)
        n = len(_entries)
    except (FigureExportError, KeyError, TypeError, ValueError):
        # a payload the figure will refuse by name; its height is never drawn
        n, strips = max(1, len((payload or {}).get("entries") or ())), []
    cols, rows = _overview_grid(n, width)
    # on a slide or poster as tall as wide: a residual plot under a 16 pt panel 0.72 as tall met its own tick labels
    plot = width / cols * (1.0 if strips and spec.preset in LARGE_PRESETS else 0.72)
    if strips:
        plot *= sum(_JUMPS_SHARES[s] for s in ("histogram", *strips)) / _JUMPS_SHARES["histogram"]
    panel = plot + 3.0 * spec.small_pt() * _PT_MM
    key = _overview_key_mm(spec) if spec.legend else 0.0
    tallest = spec.preset_values.get("max_height_mm") or 600.0
    return round(min(tallest, max(40.0, rows * panel + key + 14.0)), 1)


def _draw_appd_overview(fig, payload: dict, spec: FigureSpec, family: str):
    """Every condition's apparent-D distribution in one figure, each panel as its card draws it (its Show, else the
    dialog's Fit row; `_draw_appd_curves`) on ONE log apparent-D axis, each its own y axis (the conditions have
    their own track counts); the residual plots under every panel optional (off by default, `_appd_strips`), one y
    range per row. The key in a row of its own above the panels. Returns the first panel (the dialog's x range is
    shared by all)."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    entries, drawn, strips = _appd_overview_payloads(payload, spec)
    page = _page(spec)
    small, line = spec.small_pt(), spec.line_pt()
    width_mm, height_mm = spec.size_mm()
    cols, rows = _overview_grid(len(entries), width_mm)
    key = _overview_key_mm(spec) if spec.legend else 0.0
    panel = max(10.0, (height_mm - key) / rows)
    per = 1 + len(strips)
    shares = [_JUMPS_SHARES[n] for n in ("histogram", *strips)]
    grid = fig.add_gridspec(rows * per + (1 if key else 0), cols,
                            height_ratios=([key] if key else []) + [panel * s / sum(shares) for s in shares] * rows)
    top_row = 1 if key else 0
    places = [(e, divmod(i, cols)) for i, e in enumerate(entries)]
    lowest = {c: max(r for _e, (r, cc) in places if cc == c) for _e, (_r, c) in places}
    spans = {}
    for e, (r, _c) in places:
        got = spans.setdefault(r, {"pearson": 3.0, "ecdf": 0.005})
        mine = _appd_spans(drawn[e["key"]])
        got["pearson"], got["ecdf"] = max(got["pearson"], mine["pearson"]), max(got["ecdf"], mine["ecdf"])
    one_choice = len({drawn[e["key"]]["show"] for e in entries}) == 1
    first = None
    for e, (r, c) in places:
        ax = fig.add_subplot(grid[top_row + r * per, c], **({} if first is None else {"sharex": first}))
        first = first or ax
        under = {n: fig.add_subplot(grid[top_row + r * per + 1 + i, c], sharex=first) for i, n in enumerate(strips)}
        h = drawn[e["key"]]
        _draw_appd_curves(ax, h, spec, e.get("colour") or QC_DATA)
        for d, style, colour, _words in _appd_marks(h, page):
            ax.axvline(d, ls=style, color=colour, lw=max(MIN_LINE_PT, line), zorder=6)
        ax.set_ylim(0, h["y_max"])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        name = spec.shown_name(e["name"])
        ax.set_title(name if one_choice else f"{name} · {_shown_words(h['show'])}", fontsize=small, loc="left",
                     pad=2.0)
        if c == 0:
            ax.set_ylabel(_title(spec, "appd_overview", "y"))
        if "pearson" in under:
            _appd_pearson_strip(under["pearson"], h, spec, spans[r]["pearson"])
        if "ecdf" in under:
            _ecdf_strip(under["ecdf"], _appd_ecdf_fits(h), list(h["show"]), spec, spans[r]["ecdf"])
        if c == 0:
            for n, words in (("pearson", chart_text.PEARSON_LABEL), ("ecdf", chart_text.ECDF_RESIDUAL_LABEL)):
                if n in under:
                    under[n].set_ylabel(words.replace(" ", "\n", 1))
        stack = [ax, *under.values()]
        for upper in stack[:-1]:
            upper.tick_params(labelbottom=False)
        if r == lowest[c]:
            stack[-1].set_xlabel(_title(spec, "appd_overview", "x"))
        else:
            stack[-1].tick_params(labelbottom=False)
        for one_ax in stack:
            _qc_grid(one_ax, spec)
    _log_x(first, min(h["edges"][0] for h in drawn.values()), max(h["edges"][-1] for h in drawn.values()),
           sparse=_narrow_for_font(width_mm / cols, spec))
    if spec.legend:
        sets = [drawn[e["key"]]["show"] for e in entries]
        coloured = sorted({k for s in sets if len(s) > 1 for k in s})
        singles = sorted({s[0] for s in sets if len(s) == 1})
        handles = [Patch(color=_pale(spec, page["muted"], 0.45), lw=0)]
        labels = ["tracks (each condition's colour)"]
        handles += [Line2D([], [], ls=_JUMP_STYLES.get(k, "-"), color=_fit_colour(k, spec), lw=_jump_widths(spec)["fit"])
                    for k in coloured]
        labels += [f"{_plural(k)} fit" for k in coloured]
        if singles:
            handles.append(Line2D([], [], color=page["ink"], lw=max(0.8, line * 1.5)))
            labels.append(("one fit, as named in its panel" if len(singles) > 1 or coloured
                           else f"{_plural(singles[0])} fit") + " (in the condition's colour)")
        marks = {words: (style, colour) for h in drawn.values() for _d, style, colour, words in _appd_marks(h, page)}
        for words in ("cut", "immobile", "ceiling"):
            if words in marks:
                style, colour = marks[words]
                handles.append(Line2D([], [], ls=style, color=colour, lw=max(MIN_LINE_PT, line)))
                labels.append(words)
        holder = fig.add_subplot(grid[0, :])
        holder.set_axis_off()
        labels, ncol = _legend_columns(labels, width_mm / _PT_MM * 0.94, spec, family)
        holder.legend(handles, labels, loc="center", ncol=ncol, frameon=False, handletextpad=0.4,
                      columnspacing=1.0, borderaxespad=0.0, handlelength=1.8)
    return first


def _appd_overview_source(payload: dict, spec: FigureSpec) -> tuple:
    """Every condition's bins (edges, tracks, and for each count it draws the expected tracks and the Pearson
    residual), then every condition's every fit with its BIC and fit quality."""
    entries, drawn, _strips = _appd_overview_payloads(payload, spec)
    counts = sorted({k for h in drawn.values() for k in h["show"]})
    columns = (["Condition", "Shown as", "Bin from (µm²/s)", "Bin to (µm²/s)", "Tracks"]
               + [f"Expected, the {_plural(k)} fit" for k in counts]
               + [f"Pearson residual, the {_plural(k)} fit" for k in counts])
    rows = []
    for e in entries:
        h = drawn[e["key"]]
        name = spec.shown_name(e["name"])
        for i, count in enumerate(h["observed"]):
            expected = [h["fits"][k]["expected"][i] if k in h["fits"] else None for k in counts]
            pearson = [h["fits"][k]["pearson"][i] if k in h["fits"] else None for k in counts]
            rows.append([e["key"], name, float(h["edges"][i]), float(h["edges"][i + 1]), int(count)]
                        + expected + pearson)
    widest = max(len(f["D"]) for e in entries for f in _appd_fits(e["result"]).values())
    fit_columns = (["Condition", "Populations"] + [f"D{c + 1} (µm²/s)" for c in range(widest)]
                   + [f"Fraction {c + 1} (0-1)" for c in range(widest)] + ["BIC", "ΔBIC", "Drawn"]
                   + ["Largest ECDF gap (0-1)", "Held-out ΔlogL", "Held out over", "Folds won", "Folds"])
    fit_rows = []
    for e in entries:
        result = e["result"]
        selection = {int(r["n_pop"]): r for r in (result.get("selection") or {}).get("rows", [])}
        quality = result.get("quality") or {}
        cv = quality.get("cv") or {}
        for k, f in sorted(_appd_fits(result).items()):
            row = selection.get(k, {})
            fit_rows.append([e["key"], k] + [float(v) for v in f["D"]] + [None] * (widest - len(f["D"]))
                            + [float(v) for v in f["fraction"]] + [None] * (widest - len(f["fraction"]))
                            + [row.get("bic"), row.get("delta"), k in drawn[e["key"]]["show"],
                               ((quality.get("fits") or {}).get(k) or {}).get("ks"),
                               (cv.get("delta_total") or {}).get(k),
                               {"day": "replicates", "cell": "cells"}.get(cv.get("unit"), ""),
                               (cv.get("wins") or {}).get(k), cv.get("n_folds")])
    notes = ("Each condition's pooled fit of every track (one apparent D per track, each with its own gamma "
             "shape). BIC's n is the number of tracks.",)
    return (columns, rows), (("Fits", "Every condition's every population count, its BIC and its fit quality",
                              notes, fit_columns, fit_rows),)


def _appd_overview_notes(payload: dict, spec: FigureSpec) -> tuple:
    entries, drawn, strips = _appd_overview_payloads(payload, spec)
    notes = [f"{spec.shown_name(e['name'])}: {e['result']['n_tracks']:,} tracks, drawn "
             + ("the " + _plural(drawn[e['key']]['show'][0]) + " fit" if len(drawn[e["key"]]["show"]) == 1
                else "the " + _shown_words(drawn[e["key"]]["show"]) + " fits")
             for e in entries]
    notes.append("The curves are each condition's pooled fit of every track. The Populations bubbles may show a D "
                 "shared across the condition's replicates instead.")
    if "ecdf" in _appd_strips(spec, "appd_overview") and "ecdf" not in strips:
        notes.append("ECDF residuals are not drawn: a result has none. Run the section again.")
    return tuple(notes)


def appd_overview_legend(payload: dict, spec: FigureSpec | None = None) -> str:
    """A drafted legend for every condition's apparent-D distribution in one figure."""
    spec = spec or FigureSpec()
    entries, drawn, strips = _appd_overview_payloads(payload, spec)
    names = ", ".join(spec.shown_name(e["name"]) for e in entries)
    sets = {drawn[e["key"]]["show"] for e in entries}
    parts = [f"Apparent diffusion coefficients of {names}, one panel per condition, on a logarithmic axis. "
             f"Apparent D: one value per track, its mean single-frame squared jump divided by 4τ. A gamma "
             f"mixture is fitted to all tracks of a condition pooled, each with its own gamma shape (from its "
             f"number of steps)."]
    if len(sets) == 1 and len(next(iter(sets))) == 1:
        k = next(iter(sets))[0]
        parts.append(f"Bars: tracks per bin, in each condition's colour. Lines: the {_plural(k)} fit as expected "
                     f"tracks per bin; dashed, its populations.")
    else:
        coloured = sorted({k for s in sets if len(s) > 1 for k in s})
        parts.append("Bars: tracks per bin, in each condition's colour. Lines: the fits named in each panel as "
                     "expected tracks per bin"
                     + (" (" + "; ".join(f"{_plural(k)}, {_JUMP_STYLE_WORDS.get(k, 'line')} {_fit_colour_word(k)}"
                                         for k in coloured) + ")" if coloured else "") + ".")
    parts.append("Vertical lines: the value an immobile molecule reads from localisation error alone, the tracker's "
                 "detection ceiling and, where the fit of the run is drawn alone, where its populations cross.")
    if strips:
        parts.append(_appd_strip_words(strips).replace("Under the histogram", "Under each histogram"))
    parts.append("Apparent D averages over switching within a track, so its populations are blurrier than the SQD "
                 "fit's, which is the main readout.")
    return " ".join(parts)


def _draw_appd_cells(fig, ax, payload: dict, spec: FigureSpec, family: str) -> None:
    """The Per cell SuperPlot in print: a column per condition, each cell a dot (area by its tracks, shade by its
    replicate day), each day's mean an open circle, the mean of the day means a bar and its SD a line beside."""
    from matplotlib.lines import Line2D

    cells = payload["cells"]
    columns = cells.get("columns") or []
    if not columns:
        raise FigureExportError("no condition has cells to draw yet")
    labels, colours = payload.get("labels") or {}, payload.get("colours") or {}
    page = _page(spec)
    line, small = spec.line_pt(), spec.small_pt()
    scale = small / 6.0
    # GROUPED BY THE FACTORS (`appd_view.grouped_cells`): the columns at the layout's positions
    grouping = cells.get("layout") or {}
    grouped = bool(grouping.get("groups")) and len(grouping.get("x") or ()) == len(columns)
    X = [float(x) for x in grouping["x"]] if grouped else [float(k) for k in range(len(columns))]
    for k, col in enumerate(columns):
        i = X[k]                                        # the column's position (its index when not grouped)
        base = colours.get(col["condition"]) or QC_DATA
        n_days = max(1, col["n_days"])
        if col.get("why"):
            # one population: no slow share (`appd_view.ONE_POPULATION`), said where the dots would be
            ax.text(i, 50.0, "Not available:\n" + col["why"].replace(", ", ",\n"), ha="center", va="center",
                    fontsize=small, color=page["muted"], zorder=3)
        if col["points"]:
            ax.scatter([i + pt["x"] for pt in col["points"]], [pt["y"] for pt in col["points"]],
                       s=[(2.0 * scale * (0.8 + 0.35 * math.sqrt(pt["n"]))) ** 2 for pt in col["points"]],
                       c=[figstyle.day_shade(base, pt["day_index"], n_days) for pt in col["points"]],
                       alpha=0.55, linewidths=0, zorder=2)
        for m in col["day_means"]:
            ax.scatter([i + 0.34], [m["y"]], s=(2.0 * scale * 2.2) ** 2, facecolors=page["bg"],
                       edgecolors=figstyle.day_shade(base, m["day_index"], n_days),
                       linewidths=max(0.6, line * 1.2), zorder=4)
        if col["mean"] is not None:
            ax.hlines(col["mean"], i - 0.3, i + 0.3, color=page["ink"], lw=max(0.8, line * 1.6), zorder=5)
            if col["sd"] is not None:
                ax.vlines(i + 0.44, col["mean"] - col["sd"], col["mean"] + col["sd"], color=page["ink"],
                          lw=max(0.6, line), zorder=5)
    n = len(columns)
    ax.set_xlim(X[0] - 0.6, X[-1] + 0.6)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    pitch = spec.width_value() / _MM * 72.0 * 0.8 / ((X[-1] - X[0] + 1.0) if grouped else n)
    measure = lambda text: _text_width_pt(text, small, family)                  # noqa: E731
    names = []
    for k, col in enumerate(columns):
        name = (str(grouping["labels"][k]) if grouped
                else spec.shown_name(labels.get(col["condition"], col["condition"])))
        text = "\n".join(chart_text.wrap_text(name, measure, pitch * 0.92, max_lines=3))
        if spec.numbers:
            text += f"\n{col.get('n_cells', len(col['points']))} cells"
        names.append(text)
    ax.set_xticks(X)
    ax.set_xticklabels(names)
    ax.tick_params(axis="x", length=0)
    if grouped:
        from matplotlib.transforms import blended_transform_factory, offset_copy

        pt = spec.pt()
        down = max(name.count("\n") + 1 for name in names) * small * 1.25 + 0.6 * pt
        under = offset_copy(blended_transform_factory(ax.transData, ax.transAxes), fig=fig, units="points", x=0,
                            y=-down)
        for group in grouping["groups"]:
            g0, g1 = group["x0"] - 0.4, group["x1"] + 0.4
            ax.add_artist(Line2D([g0, g1], [0, 0], transform=under, color=page["grid"],
                                 lw=max(MIN_LINE_PT, line * 0.8), clip_on=False))
            ax.text((g0 + g1) / 2.0, 0, str(group["label"]), transform=offset_copy(
                under, fig=fig, units="points", x=0, y=-0.25 * pt), ha="center", va="top", fontsize=small,
                color=page["ink"], clip_on=False)
    ax.set_ylabel(_title(spec, "appd_cells", "y"))
    _qc_grid(ax, spec)
    if spec.legend:
        grey = page["muted"]
        handles = [Line2D([], [], ls="", marker="o", ms=small * 0.75, color=grey, alpha=0.55, mew=0),
                   Line2D([], [], ls="", marker="o", ms=small * 0.75, mfc=page["bg"], mec=grey,
                          mew=max(0.6, line * 1.2)),
                   Line2D([], [], color=page["ink"], lw=max(0.8, line * 1.6))]
        words = ["one cell (area: its tracks)", "one replicate's mean", "mean of the replicates ± SD"]
        _legend_above(ax, handles, words, spec, family)


def _appd_cells_source(payload: dict, spec: FigureSpec) -> tuple:
    """Every cell drawn (its condition, day, movie, cell, tracks and slow share); each day's mean and each
    condition's mean ± SD as further sheets."""
    cells = payload["cells"]
    labels = payload.get("labels") or {}
    rows, days, conds = [], [], []
    for col in cells["columns"]:
        c = col["condition"]
        shown = spec.shown_name(labels.get(c, c))
        for pt in col["points"]:
            rows.append([c, shown, pt.get("day"), pt.get("movie"), pt.get("cell"), int(pt["n"]), float(pt["y"])])
        for m in col["day_means"]:
            days.append([c, shown, m["day"], int(m["n_cells"]), float(m["y"])])
        conds.append([c, shown, len(col["day_means"]), col["mean"], col["sd"], len(col["points"]), col["hidden"],
                      col.get("why", "")])
    columns = ["Condition", "Shown as", "Replicate", "Movie", "Cell", "Tracks", "Slow share (%)"]
    day_sheet = ("Replicates", "Each replicate's mean over its cells",
                 ("A replicate is numbered within its condition. Replicates are independent; none is paired "
                  "with a replicate of another condition.",),
                 ["Condition", "Shown as", "Replicate", "Cells", "Mean slow share (%)"], days)
    cond_sheet = ("Conditions", "The mean of the replicate means and their SD",
                  ("SD over replicates: the spread between replicates, not a confidence interval.",),
                  ["Condition", "Shown as", "Replicates", "Mean of the replicate means (%)", "SD over replicates (%)", "Cells drawn",
                   f"Cells under {cells.get('min_tracks', 3)} tracks, left out", "Not drawn because"], conds)
    return (columns, rows), (day_sheet, cond_sheet)


def _appd_cells_notes(payload: dict) -> tuple:
    cells = payload["cells"]
    hidden = sum(col["hidden"] for col in cells["columns"])
    labels = payload.get("labels") or {}
    single = [str(labels.get(col["condition"], col["condition"])) for col in cells["columns"] if col.get("why")]
    return ("Slow share of a cell: the mean, over its tracks, of each track's probability of belonging to the "
            "slowest population of its condition's fit.",
            *((f"Fitted with one population, so there is no slow share (every cell would read 100 %): "
               f"{', '.join(single)}.",) if single else ()),
            f"Cells with fewer than {cells.get('min_tracks', 3)} tracks are left out ({hidden} in all), because "
            "one or two tracks give a share of 0, 50 or 100 %.",
            "Replicate means: the mean over a replicate's cells. The bar is the mean of the replicate means and "
            "the line beside it is their SD.")


def appd_cells_legend(payload: dict) -> str:
    """A drafted legend for the per-cell slow share."""
    cells = payload["cells"]
    hidden = sum(col["hidden"] for col in cells["columns"])
    return ("Slow share per cell, from the apparent-D fit of each condition (one value per track). Each dot is "
            "one cell, the mean probability of its tracks belonging to the slowest population; its area is set "
            "by the cell's tracks and its shade by its replicate. Open circles: each replicate's mean over its "
            "cells; bar: the mean of the replicate means; line beside it: their SD over replicates (a spread, "
            "not a confidence interval). Replicates are independent within and between conditions."
            + (f" Cells with fewer than {cells.get('min_tracks', 3)} tracks are not drawn ({hidden})." if hidden
               else "")
            + (" " + ", ".join(str((payload.get("labels") or {}).get(c["condition"], c["condition"]))
                               for c in cells["columns"] if c.get("why"))
               + ": fitted with one population, so no slow share is drawn."
               if any(c.get("why") for c in cells["columns"]) else ""))


#: Wider than this (width / height), the two checks stand side by side.
_APPD_CHECKS_WIDE = 1.3


def _appd_checks_height(spec: FigureSpec) -> float:
    """The checks figure's own height: one column stacks the two plots (about as tall as wide, at most 120 mm);
    wider than 120 mm, or on a slide, they stand side by side."""
    default = spec.default_height_mm()
    width = spec.width_value()
    if spec.preset in LARGE_PRESETS:
        return round(max(default, width * 0.6), 1) if width <= 200.0 else default
    tallest = spec.preset_values.get("max_height_mm") or 600.0
    if width > 120.0:
        return round(min(tallest, max(default, width * 0.45)), 1)
    return round(min(tallest, max(default, min(width * 0.95, 120.0))), 1)


def _draw_appd_checks(fig, entry: dict, spec: FigureSpec, family: str):
    """One condition's Checks card in print: the SWITCHING check (tracks per bin on the log apparent-D axis, what
    SQD's track-weighted populations predict if no molecule switched within its track dashed over them, the band
    between the states shaded) and the TRACK-LENGTH check (the slow share per length bin, a bin under
    `appd_view.LENGTH_MIN_TRACKS` tracks faded). Returns the switching plot's axes (the dialog's titles and
    ranges are its)."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    from .appd_view import length_payload, switching_payload

    result = entry["result"]
    sw, ln = switching_payload(result), length_payload(result)
    colour = entry.get("colour") or QC_DATA
    page = _page(spec)
    line, small = spec.line_pt(), spec.small_pt()
    width_mm, height_mm = spec.size_mm()
    wide = width_mm / height_mm >= _APPD_CHECKS_WIDE
    grid = (fig.add_gridspec(1, 2, width_ratios=(1.25, 1.0)) if wide
            else fig.add_gridspec(2, 1, height_ratios=(1.15, 1.0)))
    hist, bars = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    bar_colour = _pale(spec, colour, 0.45)
    if sw["available"]:
        edges = sw["edges"]
        band = sw.get("band")
        band_colour = _pale(spec, page["muted"], 0.2)
        if band:
            hist.axvspan(band[0], band[1], color=band_colour, lw=0, zorder=0.5)
        lefts, widths = _log_bars(edges)
        hist.bar(lefts, sw["observed"], width=widths, align="edge", color=bar_colour, linewidth=0, zorder=1)
        hist.plot(sw["centres"], sw["predicted"], ls=(0, (5.0, 3.0)), color=page["ink"],
                  lw=max(0.8, line * 1.4), zorder=3)
        _log_x(hist, edges[0], edges[-1])
        hist.set_ylim(0, sw["y_max"])
        hist.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        if spec.legend:
            handles = [Patch(color=bar_colour, lw=0),
                       Line2D([], [], ls=(0, (5.0, 3.0)), color=page["ink"], lw=max(0.8, line * 1.4))]
            words = ["tracks", "no switching (SQD's populations)"]
            if band:
                handles.append(Patch(color=band_colour, lw=0))
                words.append("between the states")
            _legend_above(hist, handles, words, spec, family, share=0.55 if wide else 1.0,
                          title=_keyed_title(spec, entry["name"]))
    else:
        hist.set_axis_off()
        hist.text(0.5, 0.5, "\n".join(chart_text.wrap_text(
            f"Switching check not available: {sw['why']}.", lambda t: _text_width_pt(t, small, family),
            width_mm / _PT_MM * (0.5 if wide else 0.9))), transform=hist.transAxes, ha="center", va="center",
            fontsize=small, color=page["muted"])
    hist.set_ylabel(_title(spec, "appd_checks", "y"))
    hist.set_xlabel(_title(spec, "appd_checks", "x"))
    # THE TRACK-LENGTH CHECK
    bins = ln["bins"]
    xs = list(range(len(bins)))
    if ln.get("available", True):
        fills = [_pale(spec, colour, 0.3) if b["few"] else colour for b in bins]
        bars.bar(xs, [b["slow"] for b in bins], width=0.62, color=fills, linewidth=0, zorder=2)
        if spec.numbers:
            for x, b in zip(xs, bins):
                bars.text(x, b["slow"] + 2.0, f"{b['n']:,}", ha="center", va="bottom", fontsize=small,
                          color=page["muted"] if b["few"] else page["ink"])
    else:
        # one population: no slow share by length (`appd_view.ONE_POPULATION`), never a bar at 100 %
        bars.text(0.5, 0.5, f"Not available:\n{ln['why']}", transform=bars.transAxes, ha="center", va="center",
                  fontsize=small, color=page["muted"])
    bars.set_xticks(xs)
    bars.set_xticklabels([b["label"].replace(" steps", "") for b in bins])
    bars.tick_params(axis="x", length=0)
    bars.set_xlim(-0.6, len(bins) - 0.4)
    bars.set_ylim(0, 112 if spec.numbers else 100)
    bars.set_yticks([0, 50, 100])
    bars.set_ylabel(chart_text.SLOW_SHARE_LABEL)
    bars.set_xlabel(chart_text.TRACK_STEPS_LABEL)
    for ax in (hist, bars):
        _qc_grid(ax, spec)
    return hist


def _appd_checks_source(entry: dict, spec: FigureSpec) -> tuple:
    """The switching check per bin (tracks observed, tracks predicted with no switching) and the track-length
    check per length bin as a second sheet."""
    from .appd_view import length_payload, switching_payload

    result = entry["result"]
    shown = spec.shown_name(entry["name"])
    sw, ln = switching_payload(result), length_payload(result)
    length_columns = ["Condition", "Shown as", "Track length", "Tracks", "Slow share (%)",
                      "Median apparent D (µm²/s)", "25th percentile", "75th percentile", "Quoted"]
    length_rows = [[entry["key"], shown, b["label"], int(b["n"]), None if b["slow"] is None else float(b["slow"]),
                    b["median"], b["q25"], b["q75"], not b["few"] and b["slow"] is not None] for b in ln["bins"]]
    from .appd_view import LENGTH_MIN_TRACKS

    length_sheet = ("Track length", "The slow share by track length",
                    (f"A bin with fewer than {LENGTH_MIN_TRACKS} tracks is drawn faded and not quoted.",),
                    length_columns, length_rows)
    if not sw["available"]:
        return (length_columns, length_rows), ()
    edges = sw["edges"]
    columns = ["Condition", "Shown as", "Bin from (µm²/s)", "Bin to (µm²/s)", "Tracks observed",
               "Tracks predicted, no switching"]
    rows = [[entry["key"], shown, float(edges[i]), float(edges[i + 1]), int(o), float(pr)]
            for i, (o, pr) in enumerate(zip(sw["observed"], sw["predicted"]))]
    return (columns, rows), (length_sheet,)


def _appd_checks_notes(entry: dict) -> tuple:
    from .appd_view import length_payload, switching_payload

    result = entry["result"]
    sw, ln = switching_payload(result), length_payload(result)
    notes = [_appd_definition(result)]
    if sw["available"]:
        notes.append("Predicted: the apparent-D histogram the SQD populations (weighted by tracks) give if every "
                     "molecule kept one state for its whole track, each track with its own number of steps. "
                     + sw["text"])
    else:
        notes.append(f"Switching check not available: {sw['why']}.")
    notes.append("Track length: " + ln["text"] + ".")
    return tuple(notes)


def appd_checks_legend(entry: dict) -> str:
    """A drafted legend for one condition's switching and track-length checks."""
    from .appd_view import length_payload, switching_payload

    result = entry["result"]
    sw, ln = switching_payload(result), length_payload(result)
    parts = [f"Checks of the apparent-D populations of {entry['name']}."]
    if sw["available"]:
        parts.append("Switching: tracks per bin of apparent D (bars) against what the SQD populations predict if "
                     "no molecule switched state within its track (dashed). The shaded band lies between the "
                     "states (twice the slow D to half the fast D). " + sw["text"] + " Excess tracks there are "
                     "molecules that switched within their track, or a state in between.")
    if not ln.get("available", True):
        parts.append(f"Track length: {ln['text']}.")
        return " ".join(parts)
    parts.append("Track length: the slow share by the number of steps in a track (bins with few tracks are faded "
                 f"and not quoted); {ln['text']}. Long tracks tend to be slow, because a fast molecule leaves the "
                 "focus or the linking radius sooner.")
    return " ".join(parts)


def _appd_counts(data) -> str:
    """How many populations each condition was fitted with, from the drawn points: "two", or
    "two (Glucose), three (Malate)" when they differ (Best may choose per condition)."""
    words = {1: "one", 2: "two", 3: "three", 4: "four"}
    per = {}
    for p in data.points:
        per[p.column] = max(per.get(p.column, 0), int(p.population))
    counts = sorted(set(per.values()))
    if len(counts) == 1:
        return words.get(counts[0], str(counts[0]))
    return ", ".join(f"{words.get(n, n)} ({data.conditions[c]})" for c, n in sorted(per.items()))


def _chunks_withheld_sentences(data, shown=str) -> list:
    """The legend's sentence for conditions whose Chunks were withheld because they hold replicates
    (`comparison.chunks_withheld`, ruled 4 Nov 2026): drawn as the pooled fit, and why. [] where none were."""
    held = list(((getattr(data, "pseudo", None) or {}).get("withheld") or {}))
    if not held:
        return []
    from .comparison import CHUNKS_WITHHELD

    return [f"{', '.join(shown(c) for c in held)}: {CHUNKS_WITHHELD}; drawn as the pooled fit, no spread."]


def appd_bubble_legend(data, shown=str) -> str:
    """A drafted legend for the gamma-mixture bubble plot of per-track apparent D, from the plot's own points: the
    population count each condition was fitted with, and what the spread is (days, cells or none)."""
    parts = [f"Gamma mixture ({_appd_counts(data)} population(s)) fitted to the apparent diffusion coefficients of "
             "single tracks (one value per track, each with its own gamma shape). Each bubble is one "
             "population: its height is its D (log scale); its area and the number under it are the fraction "
             "of tracks."]
    spreads = {p.spread for p in data.points}
    days = sorted({p.n_replicates for p in data.points if p.spread == "sd"})
    pseudo = getattr(data, "pseudo", None) or {}
    if "sd" in spreads and pseudo:
        from .bubble_view import chunk_words

        parts.append(f"Fractions are the mean ± SD over {chunk_words(pseudo)}; the SD is the spread between "
                     "chunks, not a confidence interval.")
    elif "sd" in spreads:
        span = "-".join(str(d) for d in (days[0], days[-1])) if len(days) > 1 else str(days[0])
        parts.append(f"Fractions are the mean ± SD over replicates (n = {span} replicates per "
                     "condition), with D shared by a condition's replicates. The SD is the spread between "
                     "replicates, not a confidence interval.")
    if "ci" in spreads:
        parts.append("Brackets: a 95 % interval from resampling cells around the pooled fit (one session).")
    if "none" in spreads:
        parts.append("Where no spread is shown, the fraction is from one fit to the condition's "
                     "pooled data.")
    parts += _chunks_withheld_sentences(data, shown)
    if data.errors:
        parts.append("No fit for " + "; ".join(
            f"{shown(data.conditions[i])} ({reason})" for i, reason in data.errors.items()) + ".")
    for warning in getattr(data, "warnings", None) or ():
        parts.append(f"Check the fit: {str(warning).rstrip('.')}.")
    if getattr(data, "precision", ""):
        from .bubble_view import PER_SPOT_D_IS

        parts.append("Each track's own localisation error entered the fit (per-spot precision), taken from its "
                     "spots through a calibration measured on immobile spots (" + data.precision + "), so D is "
                     f"{PER_SPOT_D_IS}. A moving population's D is still apparent.")
    parts.append("Apparent D averages over switching within a track, so these populations "
                 "are blurrier than the SQD fit's, which is the main readout.")
    return " ".join(parts)


def _appd_bubble_notes(data=None) -> tuple:
    count = _appd_counts(data) if data is not None and data.points else "the chosen number of"
    return (f"D: the population's apparent diffusion coefficient, µm²/s, from a gamma mixture "
            f"({count} population(s)) on per-track apparent D (all of a track's steps, its own shape).",
            "Fraction: the population's share of tracks, 0 to 1. With replicates it is the mean over replicates "
            "and its SD (a spread between replicates); with cells it is a 95 % interval from resampling cells; "
            "otherwise it comes from one pooled fit.")


def _draw_kinetics(fig, ax, entry: dict, spec: FigureSpec, kind: str) -> None:
    """The triggered response in print, from the card's own payload
    (`kinetics_view._chart_payload`): binned, every unit a pale curve and the mean across
    units with its SD; unbinned, one dot per track; the recommended fit dashed over either.
    The y axis follows the data, as on screen, so a few points of change stay visible."""
    pt, small, line = spec.pt(), spec.small_pt(), spec.line_pt()
    result, colour = entry["result"], entry["colour"]
    data = result["chart"]
    if data["y_range"] is None:
        raise FigureExportError("this condition's response curve is empty")
    lo, hi = data["y_range"]
    pad = max((hi - lo) * 0.08, 1.0)
    for curve in data["units"]:
        ax.plot([a for a, _b in curve], [b for _a, b in curve], color=_pale(spec, colour, 0.3),
                lw=max(MIN_LINE_PT, line * 0.6), zorder=1)
    if data["mean"]:
        ts = [m[0] for m in data["mean"]]
        ms = [m[1] for m in data["mean"]]
        sds = [m[2] if m[2] is not None else 0.0 for m in data["mean"]]
        ax.errorbar(ts, ms, yerr=sds, fmt="o-", color=colour, ms=pt * 0.5,
                    lw=max(0.6, line * 1.2), elinewidth=max(MIN_LINE_PT * 2, line * 0.8),
                    capsize=0, markeredgecolor=_page(spec)["bg"], markeredgewidth=0.4, zorder=3)
    if data["points"]:
        ax.scatter([a for a, _b in data["points"]], [b for _a, b in data["points"]],
                   s=(0.35 * pt) ** 2, c=colour, alpha=0.45, linewidths=0, zorder=2)
    if data["fit"]:
        ax.plot([a for a, _b in data["fit"]], [b for _a, b in data["fit"]], "--",
                color=_strong(spec, colour), lw=max(0.6, line * 1.2), zorder=4)
    ax.set_xlim(0, max(data["t_max"] * 1.03, 1e-9))   # the last point whole, not halved
    ax.set_ylim(max(0.0, lo - pad), min(100.0, hi + pad))
    if spec.grid:
        ax.grid(axis="y", color=_page(spec)["grid"], lw=max(MIN_LINE_PT, line * 0.6), zorder=0)
        ax.set_axisbelow(True)
    ax.set_ylabel(_title(spec, kind, "y"))
    ax.set_xlabel(_title(spec, kind, "x"))
    if spec.numbers and result.get("change_pp") is not None:
        ax.annotate(f"fitted change {result['change_pp']:+.1f} pp", (0.02, 0.97),
                    xycoords="axes fraction", ha="left", va="top", fontsize=small,
                    color=_page(spec)["ink"])


def _kinetics_source(entry: dict, spec: FigureSpec) -> tuple:
    """The plotted values: the bin means (or, unbinned, every track's point) as the main
    table; each unit's curve and the fitted curve as further sheets."""
    data, name = entry["result"]["chart"], entry["name"]
    shown = spec.shown_name(name)
    sheets = []
    if data["mean"]:
        columns = ["Condition", "Shown as", "Time (s)", "Mean slow fraction (%)", "SD (%)",
                   "Units"]
        rows = [[name, shown, float(t), float(m), None if sd is None else float(sd), int(n)]
                for t, m, sd, n in data["mean"]]
        unit_rows = [[name, i + 1, float(t), float(v)]
                     for i, curve in enumerate(data["units"]) for t, v in curve]
        sheets.append((f"Per {data['unit']}", f"Each {data['unit']}'s curve", (),
                       ["Condition", data["unit"].capitalize(), "Time (s)",
                        "Slow fraction (%)"], unit_rows))
    else:
        columns = ["Condition", "Shown as", "Time (s)", "Slow fraction (%)"]
        rows = [[name, shown, float(t), float(v)] for t, v in data["points"]]
    if data["fit"]:
        sheets.append(("Fitted curve", "The recommended model's curve", (),
                       ["Time (s)", "Fitted slow fraction (%)"],
                       [[float(t), float(v)] for t, v in data["fit"]]))
    return (columns, rows), tuple(sheets)


def _kinetics_source_notes(entry: dict) -> tuple:
    text = entry["result"]["text"]
    models = "; ".join(f"{name}: held-out error {err * 1e4:.2f} pp² ({delta * 1e4:+.2f} against "
                       f"the best, {won} units won){' (recommended)' if best else ''}"
                       for name, err, delta, won, best in text["models"])
    return ("Slow fraction: the share of steps in the slow HMM state, %; time: seconds "
            "since the trigger.", "Models: " + models + ".", *text["parameters"],
            text["change"], *text["flags"])


def kinetics_legend(entry: dict) -> str:
    """A drafted legend for the triggered response."""
    result = entry["result"]
    data, text = result["chart"], result["text"]
    s = result["settings"]
    if data["mean"]:
        how = (f"Pale lines: each {s['unit']}; points: the mean across {s['unit']}s in "
               f"{s['bin_seconds']:g} s bins ± SD.")
    else:
        how = "Each point is one track, placed at the middle of its time span."
    best = next((name for name, _e, _d, _w, rec in text["models"] if rec), None)
    fit = (f" Dashed: the recommended model ({best}), fitted once with every {s['unit']} "
           f"weighing the same, chosen on {s['unit']}s held out; intervals from resampling "
           f"{s['unit']}s." if data["fit"] else " No model curve is drawn: it left 0-100 %.")
    return (f"Fraction of steps in the slow state against time since the trigger (frame "
            f"{s['trigger_frame']}). {how}{fit} " + " ".join(text["parameters"]) + " "
            + text["change"] + (" " + " ".join(text["flags"]) if text["flags"] else ""))


# -- the Kinetics section's "Populations over time" (`induction_view`) --------------- #
def time_divisor(entry: dict) -> float:
    """Seconds per unit of the time axis an entry is drawn in (``entry["time_unit"]``: "s", the default, or "min";
    `induction_view.TIME_UNITS`)."""
    from .induction_view import TIME_UNITS

    unit = entry.get("time_unit") or "s"
    if unit not in TIME_UNITS:
        raise FigureExportError(f"time unit {unit!r}: choose one of {tuple(TIME_UNITS)}")
    return TIME_UNITS[unit]


_MPL_MARKERS = {"circle": "o", "square": "s", "triangle": "^"}


def _population_panels(d_axes, bottom, entry: dict, spec: FigureSpec, *, keyed: bool = False) -> None:
    """One condition's panels drawn onto ``d_axes`` (one D axis PER POPULATION, linear, 8 Nov 2026) and ``bottom`` (share, %): each population's
    model line in a shade of the condition's colour with its 95 % band (whole units resampled), the free per-bin fits
    as hollow circles, each REPLICATE's own per-bin fit as small open circles (the app's convention for a replicate),
    a reference curve dashed (``series["reference"]``, a file the person chose), a dashed rule at the induction. The
    populations are told apart by lightness AND line style and marker (`population_style`).
    Times are divided by the entry's time unit; limits and labels are the caller's."""
    from .induction_view import ALTERNATIVES_LABEL

    colour = entry.get("colour") or QC_DATA
    chart = entry["result"]["chart"]
    page = _page(spec)
    pt, line = spec.pt(), spec.line_pt()
    k_t = time_divisor(entry)
    n = len(chart["D"]["series"])
    # THE POPULATIONS' LOOK is the card's (`population_style`, one rule): the slowest in the condition's colour, the
    # others in other lightnesses AND other line styles and markers
    styles = population_styles(colour, n, dark=spec.dark)
    d_series = chart["D"]["series"]
    drawn = _drawn_d(chart, spec)
    jobs = [(ax, [(k, d_series[k])]) for k, ax in zip(drawn, d_axes)]
    if bottom is not None:                              # the mean mobility has no share panel
        shares = list(enumerate(chart["fractions"]["series"]))
        jobs.append((bottom, shares if spec.other_shares else shares[:1]))
    for ax, members in jobs:
        for k, s in members:
            ink, shape = styles[k]["ink"], styles[k]
            marker = _MPL_MARKERS[shape["marker"]]
            dash = (0, shape["dash"]) if shape["dash"] else "-"
            if s["band"]:
                ax.fill_between([b[0] / k_t for b in s["band"]], [b[1] for b in s["band"]],
                                [b[2] for b in s["band"]], color=ink, alpha=0.18, lw=0, zorder=1)
            reps = s.get("replicates") or []
            if reps:
                ax.scatter([r[1] / k_t for r in reps], [r[2] for r in reps], s=(0.32 * pt) ** 2, facecolors="none",
                           marker=marker, edgecolors=ink, linewidths=max(0.35, line * 0.55), alpha=0.75, zorder=2)
            if s["bins"]:
                ax.scatter([b[0] / k_t for b in s["bins"]], [b[1] for b in s["bins"]], s=(0.5 * pt) ** 2,
                           facecolors="none", marker=marker, edgecolors=ink, linewidths=max(0.4, line * 0.8), zorder=3)
            ref = s.get("reference") or []
            if len(ref) > 1:
                # GREY, a long dash (8 Nov 2026: in the population's ink it read as the fast population's dash-dot)
                ax.plot([c[0] / k_t for c in ref], [c[1] for c in ref], color=page["muted"], lw=max(0.6, line * 1.1),
                        ls=(0, REFERENCE_DASH), zorder=3.5)
            for alt in s.get("alternatives") or ():
                # THE SHAPES THE DATA CANNOT TELL APART: faint, behind the bold chosen curve
                ax.plot([c[0] / k_t for c in alt["curve"]], [c[1] for c in alt["curve"]], color=ink, ls=dash,
                        lw=max(0.5, line * 0.8), alpha=0.4, zorder=3.8)
            ax.plot([c[0] / k_t for c in s["curve"]], [c[1] for c in s["curve"]], color=ink, ls=dash,
                    lw=max(0.6, line * 1.4), label=s["name"], zorder=4)
        if chart["t_range"][0] <= 0.0 <= chart["t_range"][1]:
            ax.axvline(0.0, color=page["muted"], lw=max(MIN_LINE_PT, line * 0.7), ls="--", zorder=0)
        if spec.grid:
            ax.grid(axis="y", color=page["grid"], lw=max(MIN_LINE_PT, line * 0.6), zorder=0)
            ax.set_axisbelow(True)
    # THE HALF-TIME on the panels that changed (8 Nov 2026): the D panels where the D's did, the share panel where the
    # shares did
    from .induction_view import half_time_panels

    if spec.half_time:
        marked = half_time_panels(entry["result"].get("held_out_change"))
        if "D" in marked:
            for ax, k in zip(d_axes, drawn):
                _mark_half_time(ax, entry["result"], "D", d_series[k]["curve"], entry, spec, keyed=keyed)
        if "share" in marked and bottom is not None:
            _mark_half_time(bottom, entry["result"], "share", chart["fractions"]["series"][0]["curve"], entry, spec)
    for ax, s in zip(d_axes, [d_series[k] for k in drawn]):
        if s.get("reference"):
            ax.plot([], [], color=page["muted"], lw=max(0.6, line * 1.1), ls=(0, REFERENCE_DASH), label="Reference")
        if s.get("alternatives"):
            ax.plot([], [], color=page["muted"], lw=max(0.5, line * 0.8), alpha=0.5, label=ALTERNATIVES_LABEL)


#: Room ABOVE the data of a D panel that carries the key, as a share of its range (8 Nov 2026): on SynK course the key
#: at the panel's top lay over the reference's true fast D at 1.0 ("fast" and "Reference" struck through); "best"
#: could not avoid it, a three-column key spanning the panel. Never added to a range typed in the Axes popover.
KEY_ROOM = 0.22


def _with_key_room(lo: float, hi: float) -> tuple:
    """``(lo, hi)`` with `KEY_ROOM` of the range added above, for the key to sit over the data, not on it."""
    return lo, hi + KEY_ROOM * (hi - lo)


def _populations_height(spec: FigureSpec, rows: int = 2) -> float:
    """``rows`` stacked panels (a D panel per population over the share panel): two are about as tall as wide in a
    column, the preset's own beyond; each further row adds half a panel's height."""
    default = spec.default_height_mm()
    width = spec.width_value()
    tallest = spec.preset_values.get("max_height_mm") or 600.0
    more = max(rows - 2, 0) * 0.3 * width
    if spec.preset in LARGE_PRESETS:
        return round(max(default, width * 0.75) + more, 1)
    return round(min(tallest, max(default, width * 0.95) + more), 1)


#: The half-time's symbol in a figure: t with the subscript 1/2, in the figure's face (`_rc`)
HALF_TIME_SYMBOL = r"$\mathit{t}_{1/2}$"


def _mark_half_time(ax, result: dict, which: str, curve, entry: dict, spec: FigureSpec, *,
                    keyed: bool = False) -> bool:
    """The half-time on one panel, as the card draws it (`induction_view.half_time_mark`): a light band over its
    range or interval, a dotted line at its middle, and two short lines of words in the corner the curve has left
    ("t1/2 = 3.3-3.4 min", "vs 2.3 min (reference)"): the bottom right where it rises, the top right where it falls.
    ``keyed``: the panel's key is in its upper right, so a falling curve's words go to the bottom LEFT instead (its
    baseline is high there)."""
    from .induction_view import half_time_mark

    mark = half_time_mark(result, which, curve, time_unit=entry.get("time_unit") or "s", symbol=HALF_TIME_SYMBOL)
    if mark is None:
        return False
    page = _page(spec)
    k_t = time_divisor(entry)
    ax.axvspan(mark["lo_s"] / k_t, mark["hi_s"] / k_t, color=page["muted"], alpha=0.22, lw=0, zorder=1.5)
    ax.axvline(mark["mid_s"] / k_t, color=page["muted"], lw=max(MIN_LINE_PT, spec.line_pt() * 0.8),
               ls=(0, (1.0, 1.4)), zorder=1.6)
    rising = mark["rising"]
    if keyed and not rising:
        x, y, ha, va = 0.01, 0.03, "left", "bottom"
    else:
        x, y, ha, va = 0.99, (0.03 if rising else 0.97), "right", ("bottom" if rising else "top")
    ax.text(x, y, chr(10).join(mark["lines"]), transform=ax.transAxes, fontsize=spec.small_pt(), color=page["muted"],
            ha=ha, va=va, zorder=6, linespacing=1.25)
    return True


def _drawn_d(chart: dict, spec: FigureSpec | None = None) -> list:
    """The populations whose D a populations figure draws, by index: every one, or all but the slowest where
    `FigureSpec.slowest_d` is off and there are two or more."""
    n = len(chart["D"]["series"])
    return list(range(n)) if (spec is None or spec.slowest_d or n < 2) else list(range(1, n))


def _panel_rows(chart: dict, spec: FigureSpec | None = None) -> int:
    """The rows of a populations figure: a D panel per population drawn (`_drawn_d`), and the share panel where one
    is drawn."""
    return len(_drawn_d(chart, spec)) + (1 if chart.get("share_panel", True) else 0)


def _share_label(spec: FigureSpec) -> str:
    """The share panel's title: every population's share, or the slowest's alone (`FigureSpec.other_shares` off)."""
    return chart_text.SHARE_PCT_LABEL if spec.other_shares else chart_text.SLOW_SHARE_LABEL


def _draw_populations(fig, entry: dict, spec: FigureSpec):
    """The populations over time in print, from the card's own payload (`induction_view.compute_induction`): each
    population's D in a panel of its own (linear, its own y range, `induction_view.d_panels`), then the share panel,
    against the time since the induction, one x axis; the model line in a shade of the condition's colour per
    population, its 95 % band from whole units resampled, the free per-bin fits as hollow circles, a dashed rule at
    the induction. Returns the LOWER axes: the dialog's x title and range are its (shared)."""
    from .axes_view import apply_range
    from .induction_view import d_panels

    result, colour = entry["result"], entry.get("colour") or QC_DATA
    chart = result["chart"]
    page = _page(spec)
    small = spec.small_pt()
    drawn = _drawn_d(chart, spec)
    n = len(drawn)
    shares = chart.get("share_panel", True)
    rows = _panel_rows(chart, spec)
    grid = fig.add_gridspec(rows, 1, height_ratios=(1.0,) * rows, hspace=0.12)
    d_axes = []
    for k in range(n):
        d_axes.append(fig.add_subplot(grid[k, 0], sharex=d_axes[0] if d_axes else None))
    bottom = fig.add_subplot(grid[n, 0], sharex=d_axes[0]) if shares else None
    k_t = time_divisor(entry)
    t0, t1 = chart["t_range"][0] / k_t, chart["t_range"][1] / k_t
    typed = entry.get("axes") or {}                   # the card's Axes (30 Oct 2026): each panel's y, as shown
    # the D panels carry the key in their upper right (below): the half-time's words keep to the bottom there
    _population_panels(d_axes, bottom, entry, spec, keyed=spec.legend)
    for ax, panel in zip(d_axes, [d_panels(chart)[k] for k in drawn]):
        lo, hi = apply_range(*panel["y_range"], typed.get("D"), log=False)       # a typed D range holds on every panel
        if spec.legend and typed.get("D") is None:
            lo, hi = _with_key_room(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_ylabel(chart_text.D_LINEAR_LABEL)          # each D panel is linear (`d_panels`), never "log"
    if bottom is not None:
        lo, hi = apply_range(*chart["fractions"]["y_range"], typed.get("share"), log=False)
        bottom.set_ylim(lo, hi)
        bottom.set_ylabel(_share_label(spec))
    d_axes[0].set_xlim(t0, t1)
    lowest = bottom if bottom is not None else d_axes[-1]
    for ax in (*d_axes, *([bottom] if bottom is not None else [])):
        if ax is not lowest:
            ax.tick_params(labelbottom=False)
    lowest.set_xlabel(_title(spec, "kinetics_populations", "x", {"entries": [entry]}))
    if spec.legend:
        for ax in d_axes:                                  # in the room above the data (`KEY_ROOM`)
            ax.legend(loc="upper right", frameon=False, fontsize=small, handlelength=2.6, ncol=3)
    if spec.numbers and result["text"]["lines"]:
        # THE VERDICT ABOVE THE FIRST D PANEL, WRAPPED (26 Oct 2026, SynK through u-track): inside the panel on one
        # line, "No change shown: no changing model predicts held-out movies better than ..." ran off the figure and
        # the layout squeezed the panels into a strip a fifth of its width. Above it, the layout keeps its room.
        family, _note = resolve_font(spec.font)
        measure = lambda s: _text_width_pt(s, small, family)                   # noqa: E731
        lines = chart_text.wrap_text(result["text"]["lines"][0], measure, spec.width_value() / _PT_MM * 0.8,
                                     max_lines=3)
        d_axes[0].set_title("\n".join(lines), loc="left", fontsize=small, color=page["ink"], pad=3.0)
    return lowest


def _populations_source(entry: dict, spec: FigureSpec) -> tuple:
    """The plotted values: every population's model D and share with their bands on the drawn times as the main
    table; the free per-bin fits and the model table as further sheets."""
    result, name = entry["result"], entry["name"]
    shown = spec.shown_name(name)
    chart = result["chart"]
    minutes = time_divisor(entry) != 1.0
    columns = ["Condition", "Shown as", "Population", "Time (s)"] + (["Time (min)"] if minutes else []) + [
        "D (µm²/s)", "D low", "D high", "Share (%)", "Share low", "Share high", "Reference D (µm²/s)",
        "Reference share (%)"]
    rows = []
    for sd, sf in zip(chart["D"]["series"], chart["fractions"]["series"]):
        # a reference value is written at the model's grid time nearest it only when the file has a point there
        # (within half a grid step); the whole reference is in its own sheet
        for (t, d), (_t, dlo, dhi), (_t2, f), (_t3, flo, fhi) in zip(sd["curve"], sd["band"], sf["curve"],
                                                                      sf["band"]):
            rows.append([name, shown, sd["name"], float(t)] + ([float(t) / 60.0] if minutes else [])
                        + [float(d), float(dlo), float(dhi), float(f), float(flo), float(fhi),
                           _reference_at(sd.get("reference"), t, sd["curve"]),
                           _reference_at(sf.get("reference"), t, sf["curve"])])
    bins = [[sd["name"], float(t), float(d), None] for sd in chart["D"]["series"] for t, d in sd["bins"]]
    shares = {(sf["name"], float(t)): float(f) for sf in chart["fractions"]["series"] for t, f in sf["bins"]}
    for row in bins:
        row[3] = shares.get((row[0], row[1]))
    sheets = []
    if bins:
        sheets.append(("Per-bin fits", "Free fits in each time bin. They show the data and are not the estimate.", (),
                       ["Population", "Time (s)", "D (µm²/s)", "Share (%)"], bins))
    replicates = _replicate_rows(chart)
    if replicates:
        sheets.append(("Per-replicate bin fits", "The same free fit in each time bin, made on one replicate's steps "
                       "alone (the open circles of the figure)", (),
                       ["Population", "Replicate", "Time (s)", "D (µm²/s)", "Share (%)"], replicates))
    reference = [[sd["name"], float(t), float(v), None] for sd in chart["D"]["series"]
                 for t, v in (sd.get("reference") or [])]
    shares_ref = [[sf["name"], float(t), None, float(v)] for sf in chart["fractions"]["series"]
                  for t, v in (sf.get("reference") or [])]
    if reference or shares_ref:
        sheets.append(("Reference", "The reference curves as read from "
                       + str((result.get("reference") or {}).get("path", "its file")) + " (D of 0 is not drawn)", (),
                       ["Population", "Time (s)", "D (µm²/s)", "Share (%)"], reference + shares_ref))
    alternatives = [[sd["name"], alt["shape"], float(t), float(v)] for sd in chart["D"]["series"]
                    for alt in (sd.get("alternatives") or ()) for t, v in alt["curve"]]
    if alternatives:
        sheets.append(("Alternative shapes", "The curves of the shapes the data cannot tell from the chosen one "
                       "(drawn faintly behind it)", (), ["Population", "Shape", "Time (s)", "D (µm²/s)"], alternatives))
    models = result["text"]["models"]
    sheets.append(("Models", "Every model scored on units held out", (),
                   ["Model", "Held-out log-likelihood against the best", "SE behind the best", "Units won", "Chosen"],
                   [[m["model"], float(m["against_best"]), float(m["z_behind"]), int(m["units_won"]),
                     "yes" if m["chosen"] else ""] for m in models]))
    block = result.get("mock")
    if block and "welch" in block:
        from .kinetics_mock_view import replicate_table

        table = replicate_table(block, name, block["mock"])
        sheets.append(("Against the mock", "Each replicate's change from before the addition to after it, for the "
                       "treated condition and the mock addition " + str(block["mock"]) + " (two independent "
                       "samples, not matched by replicate number)", tuple(block_lines_for_notes(block)), list(table[0]),
                       [list(r.values()) for r in table]))
    half = result.get("half_time")
    if half:
        from .induction_view import half_time_panels, reference_half_time

        marked = half_time_panels(result.get("held_out_change"))
        refs = [reference_half_time(chart, w) for w in marked]
        sheets.append(("Half-time", "Half-time after the induction: when the change is half made, from the model "
                       "chosen on units held out. Low / high: the range over the shapes the data cannot tell apart "
                       "(kind range) or the 95 % interval over resampled units (kind interval)", (),
                       ["Half-time (s)", "Low (s)", "High (s)", "Kind", "Changed", "Reference half-time (s)"],
                       [[half["value_s"], half["low_s"], half["high_s"], half["kind"], ", ".join(marked),
                         next((v for v in refs if v is not None), None)]]))
    perm = (result.get("permutation") or {}).get("summary")
    if perm:
        sheets.append(("Permutation test", "Time-label permutation test: the gain of the changing model over the "
                       "constant mixture, on the observed clock and on each shuffle of the movies' clocks within "
                       "each replicate", (), ["Clock", "Gain (log-likelihood units)"],
                       [["observed", float(perm["observed_gain"])]]
                       + [[f"shuffle {i + 1}", float(g)] for i, g in enumerate(perm["null_gains"])]))
    return (columns, rows), tuple(sheets)


def _reference_at(reference, t: float, curve) -> float | None:
    """The reference's value at the model grid's time ``t`` where it has a point within half a grid step, else None."""
    if not reference or len(curve) < 2:
        return None
    half = 0.5 * abs(curve[1][0] - curve[0][0])
    best = min(reference, key=lambda r: abs(r[0] - t))
    return float(best[1]) if abs(best[0] - t) <= half else None


def _replicate_rows(chart: dict) -> list:
    """Each replicate's per-bin free fit as rows ``[population, replicate, t, D, share %]`` (D and share of one fit
    on one row)."""
    shares = {(sf["name"], r[0], float(r[1])): float(r[2]) for sf in chart["fractions"]["series"]
              for r in (sf.get("replicates") or [])}
    return [[sd["name"], r[0], float(r[1]), float(r[2]), shares.get((sd["name"], r[0], float(r[1])))]
            for sd in chart["D"]["series"] for r in (sd.get("replicates") or [])]


def block_lines_for_notes(block: dict) -> list:
    """The mock block's words as a sheet's notes."""
    from .kinetics_mock_view import block_lines

    return block_lines(block)


def _populations_notes(entry: dict) -> tuple:
    """The notes in the card's order: with a mock its comparison first, the held-out verdict second
    (`kinetics_mock_view.lead_sentences`, ruled 8 Nov 2026)."""
    from .kinetics_mock_view import lead_sentences

    result = entry["result"]
    text = result["text"]
    return ("D: each population's diffusion coefficient, µm²/s. Share: its fraction of the steps, %. Time: seconds "
            "since the induction. Low / high: the 95 % band of the model refitted on resampled units.",
            *lead_sentences(result), *text["lines"], *(text.get("permutation") or ()), *text["flags"],
            *(text["spread"] or ()), *(text.get("notes") or ()), *(text.get("design") or ()))


def _rows_sentence(spec: FigureSpec | None, *, overview: bool) -> str:
    """What the D and share panels hold, as the figure draws them (`FigureSpec.slowest_d`, `other_shares`,
    `conditions_as_rows`)."""
    across = overview and spec is not None and spec.conditions_as_rows
    where = "column" if across else "row" if overview else "panel"
    d = (f"D (a {where} per population, its own axis)" if overview
         else "Each population's D (one panel each, its own axis)")
    if spec is not None and not spec.slowest_d:
        d = (f"the faster populations' D (a {where} each, its own axis; the slowest population's D, at the "
             f"localisation floor, is not drawn)")
    share = ("the slowest population's share of the steps" if spec is not None and not spec.other_shares
             else "share of the steps")
    if across:
        return f"a row per condition, its name in its colour; {d} and {share} (right {where})"
    return f"{d} and {share} (bottom {where})"


def populations_legend(entry: dict, spec: FigureSpec | None = None) -> str:
    """A drafted legend for the populations over time; with a mock its comparison leads the sentences, the held-out
    verdict second (`kinetics_mock_view.lead_sentences`, ruled 8 Nov 2026)."""
    from .kinetics_mock_view import lead_sentences

    result = entry["result"]
    s, text = result["settings"], result["text"]
    lead = lead_sentences(result)
    if s["trigger_frame"] is not None:
        clock = f"the induction on frame {s['trigger_frame']} of every movie"
    elif s.get("design") == "sample":
        clock = "each movie's own time since its own sample's addition (camera start plus frames cut)"
    elif s.get("additions"):
        clock = "each movie's own time since its replicate's addition (camera start plus frames cut)"
    else:
        clock = "each movie's own time since the induction"
    bins = (f" Open markers: free fits in {s['bin_seconds']:g} s bins; they show the data and are not the estimate."
            if s["bin_seconds"] else "")
    chart = result["chart"]
    if any(sd.get("replicates") for sd in chart["D"]["series"]):
        bins += (" Small open circles (squares, triangles): the same free fit on each replicate's steps alone "
                 "(a replicate with too few steps in a bin has none).")
    reference = (f" Grey dashed lines: reference curves read from {(result.get('reference') or {}).get('path', 'a file')}, "
                 f"not fitted here." if (result.get("reference") or any(sd.get("reference")
                                                                         for sd in chart["D"]["series"])) else "")
    unit = " Time axis in minutes." if time_divisor(entry) != 1.0 else ""
    if (spec is None or spec.half_time) and result.get("half_time"):
        kind = result["half_time"]["kind"]
        unit += (" Dotted line: the half-time after the induction (t1/2), when the change is half made"
                 + (", over the range of the equally good shapes" if kind == "range" else
                    ", with its 95 % interval" if kind == "interval" else "")
                 + (", beside the reference's" if result.get("reference") else "") + ".")
    if any(sd.get("alternatives") for sd in chart["D"]["series"]):
        unit += (" Faint lines behind the bold one: shapes the data cannot tell apart (they predicted held-out "
                 f"{s['unit']}s as well); the bold one comes first in the order exponential, sigmoid, linear.")
    if len(chart["D"]["series"]) > 1:
        unit += (" Populations, slowest first, differ in lightness, line (solid, dash-dot, dotted) and marker "
                 "(circle, square, triangle); the slowest has the condition's colour.")
    if not chart.get("share_panel", True):
        return (f"The mean D (one D for every jump) against the time since the induction ({clock}; dashed "
                f"vertical line). Line: the model chosen on held-out {s['unit']}s, one D changing with time, fitted "
                f"to every step at its own time. Band: 95 % interval of the model refitted on resampled {s['unit']}s."
                f"{bins}{unit} " + " ".join([*lead, *text["lines"]])
                + (" " + " ".join(text.get("permutation") or ()) if text.get("permutation") else "")
                + (" " + " ".join(text["flags"]) if text["flags"] else "")
                + (" " + " ".join(text["design"]) if text.get("design") else ""))
    panels = (_rows_sentence(spec, overview=False)
              if len(chart["D"]["series"]) > 1 else "The population's D (top) and share of the steps (bottom)")
    return (f"{panels[0].upper()}{panels[1:]} against the time since the "
            f"induction ({clock}; dashed vertical line). Lines: the model chosen on held-out {s['unit']}s, one SQD "
            f"mixture fitted to every step at its own time. Bands: 95 % interval of the model refitted on "
            f"resampled {s['unit']}s.{bins}{reference}{unit} " + " ".join([*lead, *text["lines"]])
            + (" " + " ".join(text.get("permutation") or ()) if text.get("permutation") else "")
            + (" " + " ".join(text["flags"]) if text["flags"] else "")
            + (" " + " ".join(text["design"]) if text.get("design") else ""))


# -- Populations over time, every condition in one figure (7 Nov 2026) -------------------- #
def _populations_of(payload) -> int:
    """The populations a populations-over-time payload holds (its first result's), 0 where none is there yet."""
    for e in (payload or {}).get("entries") or ():
        chart = ((e or {}).get("result") or {}).get("chart")
        if chart:
            return len(chart["D"]["series"])
    return 0


def _overview_entries(payload: dict) -> list:
    """The conditions of a "kinetics_overview": the entries with a result, a condition whose job failed named."""
    entries = payload.get("entries") or []
    if not entries:
        raise FigureExportError("there is no populations-over-time result to draw yet")
    bad = [f"{e.get('name', '?')}: {(e.get('result') or {}).get('error', 'no result')}" for e in entries
           if not e.get("result") or "error" in e["result"]]
    if bad:
        raise FigureExportError("conditions with no populations-over-time fit to draw: " + "; ".join(bad))
    units = {e.get("time_unit") or "s" for e in entries}
    if len(units) > 1:
        raise FigureExportError(f"the conditions are in different time units ({', '.join(sorted(units))}); they "
                                f"share one time axis, so export them separately")
    return entries


def _populations_overview_height(payload: dict, spec: FigureSpec) -> float:
    """A row per population's D and the share row, of panels as wide as the figure allows its columns: each row about
    as tall as one column is wide, never taller than the preset's maximum."""
    entries = payload.get("entries") or ()
    n = max(1, len(entries))
    rows = max([_panel_rows(e["result"]["chart"], spec) for e in entries if e.get("result") and "error" not in e["result"]]
               or [2])
    width = spec.width_value()
    tallest = spec.preset_values.get("max_height_mm") or 600.0
    if spec.conditions_as_rows:
        # a row per condition, its panels (a column per quantity) about 0.4 as tall as they are wide
        return round(min(tallest, max(spec.default_height_mm(), 22.0 + 0.4 * n * width / rows)), 1)
    column = width / n
    return round(min(tallest, max(spec.default_height_mm(), 30.0 + 0.8 * rows * column)), 1)


def _draw_populations_overview(fig, payload: dict, spec: FigureSpec):
    """Every condition's populations over time side by side (columns), a row per population's D (its own y range,
    shared by the conditions: the union of theirs) and the share row, ONE time axis (the union of the conditions'
    ranges, the dialog's x range), so the conditions read against each other. Each column is `_population_panels`.
    Returns the lower left axes: the dialog's x title and range are its (shared). The conditions must hold the same
    number of populations (rows are populations)."""
    from .induction_view import d_panels

    entries = _overview_entries(payload)
    counts = {len(e["result"]["chart"]["D"]["series"]) for e in entries}
    if len(counts) > 1:
        raise FigureExportError(f"the conditions hold different numbers of populations ({sorted(counts)}); the rows "
                                f"of one figure are populations, so export the conditions one at a time")
    page = _page(spec)
    small = spec.small_pt()
    k_t = time_divisor(entries[0])
    n = len(entries)
    pops = counts.pop()
    shares = all(e["result"]["chart"].get("share_panel", True) for e in entries)
    t0 = min(e["result"]["chart"]["t_range"][0] for e in entries) / k_t
    t1 = max(e["result"]["chart"]["t_range"][1] for e in entries) / k_t
    drawn = _drawn_d(entries[0]["result"]["chart"], spec)          # every condition holds the same populations
    nd = len(drawn)
    ranges = [d_panels(e["result"]["chart"]) for e in entries]
    d_ranges = [(min(r[k]["y_range"][0] for r in ranges), max(r[k]["y_range"][1] for r in ranges))
                for k in drawn]
    rows = nd + (1 if shares else 0)
    if spec.conditions_as_rows:
        return _overview_as_rows(fig, payload, entries, spec, nd, shares, d_ranges, (t0, t1))
    grid = fig.add_gridspec(rows, n, height_ratios=(1.0,) * rows, hspace=0.12, wspace=0.12)
    first = {}                                         # row -> the first column's axes (the shared ones)
    columns = []
    for j, entry in enumerate(entries):
        col = []
        for r in range(rows):
            ax = fig.add_subplot(grid[r, j], sharex=first.get(0), sharey=first.get(r))
            first.setdefault(r, ax)
            col.append(ax)
        _population_panels(col[:nd], col[nd] if shares else None, entry, spec)
        for r, ax in enumerate(col):
            if r < rows - 1:
                ax.tick_params(labelbottom=False)
            if j:
                ax.tick_params(labelleft=False)
        col[0].set_title(spec.shown_name(entry["name"]), loc="left", fontsize=small, color=page["ink"], pad=3.0)
        columns.append(col)
    # ONE TIME AXIS TITLE across the columns (a column is narrower than the title), under the MIDDLE column: the key
    # takes the figure's bottom (8 Nov 2026), and a figure-wide x title and a bottom key were placed on each other.
    # With an even count it is centred on the gap between the middle two (the grid's wspace, 0.12 of a column).
    mid = columns[(n - 1) // 2][-1]
    mid.set_xlabel(_title(spec, "kinetics_overview", "x", payload), fontsize=spec.pt(), color=page["ink"])
    if n % 2 == 0:
        mid.xaxis.label.set_x(1.0 + 0.12 / 2)
        mid.xaxis.label._smtracker_span = 2.12       # its room: two panels and the gap (`_text_rooms`)
    if spec.legend:
        _overview_key(fig, columns, spec, page)
    first[0].set_xlim(t0, t1)
    for r in range(nd):
        first[r].set_ylim(*d_ranges[r])                    # the key is below the columns, not in a panel
        first[r].set_ylabel(chart_text.D_LINEAR_LABEL)
    if shares:
        first[nd].set_ylim(0.0, 100.0)
        first[nd].set_ylabel(_share_label(spec))
    return first[rows - 1]


def _overview_as_rows(fig, payload: dict, entries: list, spec: FigureSpec, nd: int, shares: bool, d_ranges: list,
                      t_range: tuple):
    """`_draw_populations_overview` with the CONDITIONS AS ROWS (`FigureSpec.conditions_as_rows`, 8 Nov 2026, as
    Fig. 6B/C: "it is not clear that the rows are the different experiments"): a row per condition, a column per
    quantity (each drawn population's D, then the share), one y range per column and one time axis. The condition's
    name stands over each of its panels, bold in its colour; every panel names its quantity (asked: "no unit y axis").
    Returns the lower left axes, as the columns layout does."""
    page = _page(spec)
    n, ncols = len(entries), nd + (1 if shares else 0)
    grid = fig.add_gridspec(n, ncols, hspace=0.12, wspace=0.12)
    first = {}                                         # column -> the first row's axes (the shared ones)
    table = []
    for i, entry in enumerate(entries):
        row = []
        for c in range(ncols):
            ax = fig.add_subplot(grid[i, c], sharex=first.get(0), sharey=first.get(c))
            first.setdefault(c, ax)
            row.append(ax)
        _population_panels(row[:nd], row[nd] if shares else None, entry, spec)
        for c, ax in enumerate(row):
            ax.set_title(spec.shown_name(entry["name"]), loc="left", fontsize=spec.pt(), fontweight="bold",
                         color=entry.get("colour") or page["ink"], pad=3.0)
            ax.set_ylabel(chart_text.D_LINEAR_LABEL if c < nd else _share_label(spec))
            if i < n - 1:
                ax.tick_params(labelbottom=False)
        table.append(row)
    # ONE time title under the bottom row's middle panel (centred on the gap between the middle two when even)
    mid = table[-1][(ncols - 1) // 2]
    mid.set_xlabel(_title(spec, "kinetics_overview", "x", payload), fontsize=spec.pt(), color=page["ink"])
    if ncols % 2 == 0:
        mid.xaxis.label.set_x(1.0 + 0.12 / 2)
        mid.xaxis.label._smtracker_span = 2.12       # its room: two panels and the gap (`_text_rooms`)
    if spec.legend:
        _overview_key(fig, table, spec, page)
    first[0].set_xlim(*t_range)
    for c in range(nd):
        first[c].set_ylim(*d_ranges[c])
    if shares:
        first[nd].set_ylim(0.0, 100.0)
    return table[-1][0]


def _overview_key(fig, columns, spec: FigureSpec, page: dict) -> None:
    """ONE key for every column, BELOW the columns (asked 8 Nov 2026, "legend on the bottom"; above them before, where a
    three-column key in the last column, wider than a column, had spilled left). The entries are every panel's (the
    share panel's too, so a population whose D is not drawn is still named), the populations first, then the reference
    and the shapes. The populations are drawn in ink, since the line style names the population and the column's
    colour names the condition; the reference and the faint shapes keep their own neutral colour."""
    from matplotlib.lines import Line2D

    from .induction_view import ALTERNATIVES_LABEL

    seen = {}
    for col in columns:
        for ax in col:
            for handle, label in zip(*ax.get_legend_handles_labels()):
                seen.setdefault(label, handle)
    if not seen:
        return
    neutral = ("Reference", ALTERNATIVES_LABEL)                     # after the populations, in this order
    seen = dict([(k, v) for k, v in seen.items() if k not in neutral]
                + [(k, seen[k]) for k in neutral if k in seen])
    handles = []
    for label, handle in seen.items():
        key = Line2D([], [])
        key.update_from(handle)
        if label not in neutral:
            key.set_color(page["ink"])
        handles.append(key)
    fig.legend(handles, list(seen), loc="outside lower center", ncol=len(handles), frameon=False,
               fontsize=spec.small_pt(), handlelength=2.6)


def _populations_overview_source(payload: dict, spec: FigureSpec) -> tuple:
    """Every condition's main table stacked, the per-bin, per-replicate, reference, model and permutation sheets
    stacked under a Condition column."""
    entries = _overview_entries(payload)
    columns, rows, sheets = None, [], {}
    for entry in entries:
        (cols, body), more = _populations_source(entry, spec)
        columns = cols
        rows += body
        for title, note, _span, heads, data in more:
            kept = sheets.setdefault(title, [title, note, (), ["Condition"] + list(heads), []])
            kept[4] += [[entry["name"]] + list(r) for r in data]
    return (columns, rows), tuple(tuple(v) for v in sheets.values())


def _populations_overview_notes(payload: dict) -> tuple:
    entries = _overview_entries(payload)
    out = ["D: each population's diffusion coefficient, µm²/s; share: its fraction of the steps, %. One time axis for "
           "every condition and one y range per quantity drawn. Low / high: the 95 % band of each model refitted on "
           "its units resampled."]
    for e in entries:
        lines = e["result"]["text"]["lines"]
        out.append(f"{e['name']}: " + (lines[0] if lines else ""))
        out += [f"{e['name']}: {line}" for line in (e["result"]["text"].get("permutation") or ())[:1]]
        out += [f"{e['name']}: {line}" for line in (e["result"]["text"].get("design") or ())]
    return tuple(out)


def populations_overview_legend(payload: dict, spec: FigureSpec | None = None) -> str:
    """A drafted legend for the populations over time of every condition."""
    entries = _overview_entries(payload)
    first = entries[0]["result"]["settings"]
    reference = " Grey dashed lines: reference curves, not fitted here." if any(
        e["result"].get("reference") for e in entries) else ""
    if (spec is None or spec.half_time) and any(e["result"].get("half_time") for e in entries):
        reference += (" Dotted lines: the half-time after the induction (t1/2) where a change was found, and the "
                      "reference's where one is drawn.")
    per = "column" if spec is not None and spec.conditions_as_rows else "row"
    return (f"Each condition's populations over time: {_rows_sentence(spec, overview=True)} "
            f"against the time since the induction, on one time axis with one y range per {per}. Lines: the model "
            f"chosen on held-out {first['unit']}s. Bands: 95 % interval of the model refitted on resampled "
            f"{first['unit']}s. Circles: free fits per time bin and, small, per replicate.{reference} "
            + " ".join(f"{e['name']}: {e['result']['text']['lines'][0]}" for e in entries
                       if e["result"]["text"]["lines"]))


# -- Against the mock addition, every treated condition (8 Nov 2026) ---------------- #
# The card's MockChart in print (asked 8 Nov 2026, "Add it", for the kinetics figure's verdict panel): ONE PANEL PER
# TREATED CONDITION, each replicate a thin line from its value BEFORE the addition to its value AFTER it, the treated
# condition beside its own mock, in the readout's own unit (µm²/s, %), the replicates' mean as a bold line, and above
# the panel each condition's own mean change and the p of their difference (`kinetics_mock_view.mock_lines`, one
# source with the card; redrawn the same day from change dots, which read as "a change of a change", as Fig. 6D). The
# panels share one y axis. A treated condition whose block holds no comparison (a refusal, an error) is not drawn and
# the legend names it. Each treated condition carries its OWN mock: the after window is its own (Auto starts where ITS
# response is 90 % complete), so the mock's values differ between panels.


def _mock_entries(payload: dict) -> tuple:
    """``(drawn, left)``: the entries whose mock block holds a comparison, and ``(name, reason)`` for the rest.
    Refused by name when nothing can be drawn or the panels' units differ (one y axis)."""
    from .kinetics_mock_view import mock_chart

    drawn, left = [], []
    for e in payload.get("entries") or []:
        result, name = e.get("result") or {}, e.get("name", "?")
        block = result.get("mock") or {}
        if "error" in result:
            left.append((name, str(result["error"])))
        elif "welch" in block:
            drawn.append(e)
        elif block:
            left.append((name, str(block.get("refused") or block.get("error") or "no comparison")))
        else:
            left.append((name, "no mock addition was chosen in its Run"))
    if not drawn:
        raise FigureExportError("no condition read against a mock addition to draw yet"
                                + ("" if not left else " (" + "; ".join(f"{n}: {r}" for n, r in left) + ")"))
    units = {mock_chart(e["result"]["mock"])["unit"] for e in drawn}
    if len(units) > 1:
        raise FigureExportError("the comparisons are in different units (a mean D and a bound share) and share one "
                                "y axis; export one readout at a time")
    return drawn, left


def _mock_names(entry: dict, spec: FigureSpec) -> dict:
    block = entry["result"]["mock"]
    return {"treated": spec.shown_name(entry["name"]),
            "mock": spec.shown_name(entry.get("mock_name") or block["mock"])}


def _draw_mock(fig, payload: dict, spec: FigureSpec):
    """Every treated condition against its mock (the section's comment above). Returns the first panel's axes: the
    dialog's y range and title are its (shared)."""
    from .kinetics_mock_view import LINE_X, mock_lines

    drawn, _left = _mock_entries(payload)
    page = _page(spec)
    pt, small, line = spec.pt(), spec.small_pt(), spec.line_pt()
    family, _note = resolve_font(spec.font)
    measure = lambda s: _text_width_pt(s, small, family)                                   # noqa: E731
    panel_pt = spec.width_value() / _PT_MM / len(drawn)
    lines = []
    for entry in drawn:
        names = _mock_names(entry, spec)
        lines.append(mock_lines(entry["result"]["mock"], names["treated"], names["mock"]))
    top = max(d["y_range"][1] for d in lines)
    grid = fig.add_gridspec(1, len(drawn), wspace=0.12)
    axes = []
    for j, (entry, chart) in enumerate(zip(drawn, lines)):
        ax = fig.add_subplot(grid[0, j], sharey=axes[0] if axes else None)
        axes.append(ax)
        names = _mock_names(entry, spec)
        inks = {"treated": entry.get("colour") or QC_DATA, "mock": entry.get("mock_colour") or page["muted"]}
        for column in chart["columns"]:
            ink = inks[column["role"]]
            x0, x1 = column["x"]
            for _replicate, before, after in column["rows"]:
                ax.plot([x0, x1], [before, after], color=ink, lw=max(MIN_LINE_PT, line * 0.8), alpha=0.55, zorder=2)
                ax.scatter([x0, x1], [before, after], s=(0.45 * pt) ** 2, color=ink, alpha=0.75, lw=0, zorder=3)
            if all(math.isfinite(v) for v in column["mean"]):
                ax.plot([x0, x1], list(column["mean"]), color=ink, lw=max(0.8, line * 2.0), zorder=4,
                        solid_capstyle="round")
        ax.set_xlim(LINE_X["treated"][0] - 0.45, LINE_X["mock"][1] + 0.45)
        ax.set_xticks([*LINE_X["treated"], *LINE_X["mock"]])
        ax.set_xticklabels(["before", "after", "before", "after"], fontsize=small)
        ax.tick_params(axis="x", length=0)
        # each group's name under its two ticks, bold in its colour
        for role in ("treated", "mock"):
            x0, x1 = LINE_X[role]
            ax.annotate("\n".join(chart_text.wrap_text(names[role], measure, 0.45 * panel_pt, max_lines=2)),
                        xy=((x0 + x1) / 2, 0.0), xycoords=("data", "axes fraction"), xytext=(0, -1.9 * small),
                        textcoords="offset points", ha="center", va="top", fontsize=small, color=inks[role],
                        fontweight="bold", annotation_clip=False)
        if spec.grid:
            ax.grid(axis="y", color=page["grid"], lw=max(MIN_LINE_PT, line * 0.6), zorder=0)
            ax.set_axisbelow(True)
        if j:
            ax.tick_params(labelleft=False)
        if spec.numbers and chart["heading"]:
            heading = [w for h in chart["heading"]
                       for w in chart_text.wrap_text(h, measure, 0.95 * panel_pt, max_lines=2)]
            ax.set_title("\n".join(heading), loc="center", fontsize=small, color=page["ink"], pad=3.0,
                         linespacing=1.3)
    axes[0].set_ylim(0.0, top)
    axes[0].set_ylabel(_title(spec, "kinetics_mock", "y", payload))
    return axes[0]


def _mock_source(payload: dict, spec: FigureSpec) -> tuple:
    """Each replicate as drawn (the main table: the panel, the group, its value before and after in the readout's own
    unit, its change as computed and as the heading states it), and one row per comparison (each group's mean change
    and interval, the means its bold line joins, the difference beyond the mock, p)."""
    from .kinetics_mock_view import _pct, mock_chart, mock_lines, replicate_table

    drawn, _left = _mock_entries(payload)
    columns, rows, summary = None, [], []
    for entry in drawn:
        block, names = entry["result"]["mock"], _mock_names(entry, spec)
        mobility = block["readout"] == "mean_D"
        table = replicate_table(block, names["treated"], names["mock"])
        heads = list(table[0])
        computed = next(h for h in heads if h.startswith("change"))
        shown = "Change stated (%)" if mobility else "Change stated (pp)"
        columns = ["Panel", *heads, shown]
        for r in table:
            v = float(r[computed])
            rows.append([names["treated"], *r.values(), (_pct(v) if mobility else v) if math.isfinite(v) else None])
        chart, w = mock_chart(block), block["welch"]
        treated, mock = chart["columns"]
        bold = {c["role"]: c["mean"] for c in mock_lines(block)["columns"]}
        summary.append([names["treated"], names["mock"], block["readout"], block.get("after_from_s"),
                        treated["mean"], treated["lo"], treated["hi"], mock["mean"], mock["lo"], mock["hi"],
                        chart["difference"]["value"], chart["difference"]["lo"], chart["difference"]["hi"],
                        w["p"], w["df"], w["n_b"], w["n_a"], *bold["treated"], *bold["mock"]])
    unit = mock_chart(drawn[0]["result"]["mock"])["unit"]
    value_unit = "µm²/s" if unit == "%" else "%"
    sheet = ("Comparisons", "Each treated condition against its mock: each group's mean change with its 95 % t interval "
             f"over replicates, and the difference beyond the mock (Welch, unpaired), in {unit}. The ends of the bold "
             f"lines are the mean of the replicates' values before and after, in {value_unit}", (),
             ["Treated", "Mock", "Readout", "After window from (s)", f"Treated change ({unit})", "Treated low",
              "Treated high", f"Mock change ({unit})", "Mock low", "Mock high", f"Beyond the mock ({unit})",
              "Beyond low", "Beyond high", "p (Welch)", "Welch df", "Treated replicates", "Mock replicates",
              f"Treated mean before ({value_unit})", f"Treated mean after ({value_unit})",
              f"Mock mean before ({value_unit})", f"Mock mean after ({value_unit})"], summary)
    return (columns, rows), (sheet,)


def _mock_notes(payload: dict) -> tuple:
    drawn, left = _mock_entries(payload)
    from .kinetics_mock_view import export_lines

    out = ["Each replicate's value before the addition and after it (a line joins the two). Treated and mock are two "
           "independent samples of replicates (Welch), not matched by replicate number."]
    for e in drawn:
        block = e["result"]["mock"]
        out += [f"{e['name']}: {s}" for s in export_lines(block, e.get("mock_name") or str(block["mock"]))]
    out += [f"Not drawn: {name}, {reason}." for name, reason in left]
    return tuple(out)


def mock_legend(payload: dict, spec: FigureSpec | None = None) -> str:
    """A drafted legend for the treated conditions against their mock addition."""
    spec = spec or FigureSpec()
    drawn, left = _mock_entries(payload)
    mobility = drawn[0]["result"]["mock"]["readout"] == "mean_D"
    what = ("the mean D (one population fitted to every step of a window; the ratio after over before, as a "
            "percentage)" if mobility else
            "the bound share (the share of the slowest population, with its D held at one fit over both conditions; "
            "after minus before, in percentage points)")
    value = "mean D (µm²/s)" if mobility else "bound share (%)"
    out = (f"Each treated condition against a mock addition imaged on the same schedule. Each replicate's {value} "
           f"before and after the addition is joined by a thin line; the bold line is the mean. Above each panel: "
           f"each condition's mean change in {what}, and whether the two differ (Welch over replicates, two "
           f"independent samples not matched by replicate number; this narrows the interval, which favours an "
           f"effect).")
    for e in drawn:
        block, names = e["result"]["mock"], _mock_names(e, spec)
        out += f" {names['treated']} against {names['mock']}: " + " ".join(
            line.split(" (unpaired")[0] + ("." if " (unpaired" in line else "")
            for line in (block.get("lines") or []) if line.startswith("Beyond the mock"))
        after = block.get("after_why")
        if after:
            out += f" After window {after}."
    if left:
        out += " Not drawn: " + "; ".join(f"{n} ({r})" for n, r in left) + "."
    return out


# -- the HMM's state diagram ------------------------------------------------------- #
def _single_entry(payload: dict, what: str) -> dict:
    """The one condition a per-card figure draws (``payload["entries"][0]``), refused by
    name when the card holds no result or an error."""
    entries = payload.get("entries") or []
    if not entries:
        raise FigureExportError(f"no {what} result to draw yet")
    entry = entries[0]
    result = entry.get("result") or {}
    if "error" in result:
        raise FigureExportError(f"{entry.get('name', 'this condition')} has no {what} fit to "
                                f"draw: {result['error']}")
    return entry


# THE STATE DIAGRAM IN PRINT (redrawn 5 Nov 2026, after the experimenter tried the first "every condition" figure on
# protein F: "the text seems sometimes too big. and its over the arrows and the arrows not visible. also the text for one
# rate in some settings is directly UNDER the title and not at the arrow. its a mess ... pls take a look at how others
# did it in vbspt etc. im free to change the style"). The diagram it replaces stood its states on the screen card's
# triangle and SEARCHED for room for every label; shrunk into a cell, the search ran out of room and left labels over
# the arrows or far from them. Now nothing is searched. The states stand in a ROW, slow to fast (the order
# `hmm_view` names them in), read along D as vbSPT's model diagrams are (Persson et al. 2013), the circle's AREA its
# occupancy. Each pair of neighbours has two straight arrows, the one toward the faster state above the centre line
# and the one back below it, each rate written just over (or under) its own arrow; two states that are not
# neighbours (three or four states) are joined by a bracket over the row, one level each way, the rate on the
# bracket. Each state's name, D and occupancy stand in a block under it. An arrow's WIDTH grows with its rate, as
# vbSPT draws it. Every distance is computed from the text it has to hold, so the layout is known before anything is
# drawn; a diagram too wide for its cell is drawn smaller as a whole, text with it (`_HMM_MIN_SCALE` at the least).
# One scale for the circles and one for the arrows across the whole figure, so conditions side by side compare.

#: The radius of a state holding every step (occupancy 1), in ems of the body text.
_HMM_R_EM = 1.9
#: How far each arrow of a pair runs from the row's centre line, in ems (at least clear of the other's head).
_HMM_ARROW_EM = 0.32
#: An arrow's width at the slowest and at the fastest rate the figure draws, in the preset's line widths.
_HMM_ARROW_LW = (0.9, 1.9)
#: The arrowheads: their length and half width in units of their size, which is this share of the small text's
#: size -- one size for every arrow, whatever its width (heads grown with the width covered the labels).
_HMM_HEAD = (0.55, 0.32, 0.8)
#: The smallest a diagram is drawn to fit its cell, its text with it (5 pt where the body is 7).
_HMM_MIN_SCALE = 0.72


def _hmm_rate_spread(result: dict, i: int, j: int) -> tuple:
    """The ± the rate from state ``i`` to ``j`` carries when asked for (`FigureSpec.rate_spread`): ``(value, what)``
    -- half its 95 % interval over resampled cells or movies, or the SD of the replicates' (chunks') own fits -- or
    ``(None, "")`` where none was computed (no uncertainty, one replicate, the spread withheld)."""
    interval = result.get("interval") or {}
    mode = interval.get("mode")
    if mode in ("cell", "movie") and not interval.get("withheld"):
        band = (interval.get("rates") or {}).get(f"{i}>{j}")
        if band:
            return (float(band[1]) - float(band[0])) / 2.0, f"half the 95 % interval over resampled {mode}s"
    if mode == "day":
        from .hmm_view import unit_words

        n = int(result["fit"]["n_states"])
        values = [float(v["rates"][i][j]) for v in interval.get("values") or ()
                  if int(v.get("n_states", 0)) == n and not v.get("degenerate")]
        if len(values) >= 2:
            mean = sum(values) / len(values)
            sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
            return sd, f"SD over {unit_words(interval)[1]}"
    return None, ""


def _hmm_has_spread(payload) -> bool:
    """Whether any rate of the payload's diagrams has a ± to show (the Show box is offered only then)."""
    for e in (payload or {}).get("entries") or [] if isinstance(payload, dict) else []:
        result = e.get("result") or {}
        for ar in (result.get("diagram") or {}).get("arrows") or ():
            if _hmm_rate_spread(result, int(ar["from"]), int(ar["to"]))[0] is not None:
                return True
    return False


def _hmm_occupancy_spread(result: dict, k: int) -> tuple:
    """The ± state ``k``'s occupancy carries when asked for (`FigureSpec.occupancy_spread`): ``(value, what)``, a
    share (0-1) -- the SD of the replicates' (chunks') own occupancies, as the card and the source data give it, or
    half the 95 % interval over resampled cells or movies -- or ``(None, "")`` where none was computed. One state
    holds every step in every replicate, so it has none."""
    if int(result["fit"]["n_states"]) < 2:
        return None, ""
    interval = result.get("interval") or {}
    mode = interval.get("mode")
    if mode in ("cell", "movie") and not interval.get("withheld"):
        band = (interval.get("occupancy") or [None] * (k + 1))[k]
        if band:
            return (float(band[1]) - float(band[0])) / 2.0, f"half the 95 % interval over resampled {mode}s"
    if mode == "day" and interval.get("occupancy_sd"):
        from .hmm_view import unit_words

        return float(interval["occupancy_sd"][k]), f"SD over {unit_words(interval)[1]}"
    return None, ""


def _hmm_has_occupancy_spread(payload) -> bool:
    """Whether any state of the payload's diagrams has an occupancy ± to show (the Show box is offered only then)."""
    for e in (payload or {}).get("entries") or [] if isinstance(payload, dict) else []:
        result = e.get("result") or {}
        if "fit" in result and "diagram" in result and any(
                _hmm_occupancy_spread(result, k)[0] is not None for k in range(len(result["diagram"]["nodes"]))):
            return True
    return False


def _hmm_block(result: dict, k: int, node: dict, spread: bool) -> list:
    """The lines under state ``k``: its name, D and occupancy as the card writes them (`hmm_view.diagram_payload`),
    the occupancy "38.0 ± 4.1%" with ``spread`` (`_hmm_occupancy_spread`)."""
    lines = str(node["label"]).split("\n")
    if spread and lines[-1].endswith("%"):
        value, _what = _hmm_occupancy_spread(result, k)
        if value is not None:
            lines[-1] = f"{lines[-1][:-1]} ± {100 * value:.1f}%"
    return lines


def _hmm_rate_words(result: dict, i: int, j: int, rate, spread: bool) -> str:
    """A rate as the diagram writes it: "7.73 /s", or with ``spread`` "7.73 ± 1.2 /s" (`_hmm_rate_spread`)."""
    words = f"{float(rate):.3g}"
    if spread:
        value, _what = _hmm_rate_spread(result, i, j)
        if value is not None:
            words += f" ± {value:.2g}"
    return words + " /s"


def _hmm_rate_max(entries: list) -> float:
    """The fastest rate any shown diagram draws: the arrows' widths are on this one scale."""
    rates = [float(a["rate"]) for e in entries for a in e["result"]["diagram"]["arrows"]
             if a.get("rate") is not None and math.isfinite(float(a["rate"]))]
    return max(rates, default=0.0)


def _hmm_chain(entry: dict, spec: FigureSpec, family: str, scale: float = 1.0, floors=None) -> dict:
    """Where everything of ``entry``'s state diagram goes, in mm, the row's centre line at y = 0 and the leftmost ink
    at x = 0: ``{"x", "r", "width", "top", "bottom", "a", "gap", "pairs", "brackets", "blocks", "block_top",
    "small", "scale"}``. ``pairs`` are the neighbours' arrows (``(i, j, text)``, i to j), ``brackets`` the others
    (``{"from", "to", "y", "text", "legs": {state: x}, "label_x"}``), ``blocks`` each state's lines under it.
    ``floors`` (``(left, [gap, ...], right)``, the diagram's own ``"spacing"`` of the others): at least that much room
    left of the first state, between neighbours and right of the last -- so the diagrams stacked in one column of the
    grid put each state over the same state (`_hmm_layout`)."""
    result = entry["result"]
    diagram = result["diagram"]
    nodes = diagram["nodes"]
    n = len(nodes)
    em = spec.pt() * _PT_MM * scale
    small = spec.small_pt() * scale
    th = small * _PT_MM * 1.15                          # one line of the small text
    gap = 0.35 * scale
    ms = _HMM_HEAD[2] * small                          # the arrowheads' size, pt
    head = _HMM_HEAD[1] * ms * _PT_MM                  # an arrowhead's half height, mm
    a = max(_HMM_ARROW_EM * em, head + 0.25 * scale)    # the two arrows' heads never touch
    r = [max(0.5 * scale, _HMM_R_EM * em * math.sqrt(max(float(node["occupancy"]), 0.0))) for node in nodes]

    def wide(text: str) -> float:
        return _text_width_pt(text, small, family) * _PT_MM if text else 0.0

    blocks = [_hmm_block(result, k, node, spec.occupancy_spread) if spec.numbers else [str(node["name"])]
              for k, node in enumerate(nodes)]
    block_w = [max(wide(line) for line in block) for block in blocks]
    rates = {(int(ar["from"]), int(ar["to"])): ar.get("rate") for ar in diagram["arrows"]}

    def text(i: int, j: int) -> str:
        """The rate alone, or with its ± (`FigureSpec.rate_spread`); an interval is the source data's, never drawn."""
        rate = rates.get((i, j))
        if not spec.numbers or rate is None or not math.isfinite(float(rate)):
            return ""
        return _hmm_rate_words(result, i, j, rate, spec.rate_spread)

    labelled = spec.numbers and n >= 2
    # x: each pair of neighbours as far apart as its two labels, a readable arrow and the blocks under them need
    left = max(r[0] + 0.3 * scale, block_w[0] / 2.0)
    gaps = []
    for i in range(n - 1):
        label_w = max(wide(text(i, i + 1)), wide(text(i + 1, i)))
        gaps.append(max(r[i] + r[i + 1] + label_w + 2.0 * scale, r[i] + r[i + 1] + 3.5 * scale,
                        (block_w[i] + block_w[i + 1]) / 2.0 + 1.6 * scale))
    right = max(r[-1] + 0.3 * scale, block_w[-1] / 2.0)
    spacing = (left, list(gaps), right)
    if floors is not None:
        left, right = max(left, floors[0]), max(right, floors[2])
        gaps = [max(g, f) for g, f in zip(gaps, floors[1])]
    x = [left]
    for g in gaps:
        x.append(x[-1] + g)
    width = x[-1] + right
    pairs = [(i, i + 1, text(i, i + 1)) for i in range(n - 1)] + [(i + 1, i, text(i + 1, i)) for i in range(n - 1)]
    # an arrow's label stands clear of its arrowhead and of the head's outline (the widest arrow's stroke; at the
    # first try the labels touched them), the brackets over the row: the nearer two states first, each pair one
    # level each way
    stroke = max(0.5, spec.line_pt() * _HMM_ARROW_LW[1]) * _PT_MM / 2.0
    off = head + stroke + 0.5 * scale
    label_h = (off + th) if labelled else head + stroke
    # the first bracket high enough over the circles that a leg runs straight past its arrowhead's length
    level = max(max(r), a + label_h) + _HMM_HEAD[0] * ms * _PT_MM + 1.6 * scale
    brackets = []
    for span in range(2, n):
        for i in range(n - span):
            for f, t in ((i, i + span), (i + span, i)):
                brackets.append({"from": f, "to": t, "y": level, "text": text(f, t)})
                level += (stroke + gap + th if labelled else stroke) + 0.8 * scale
    # where each bracket's legs stand on a state: those toward a state on its left left of its centre, those toward
    # the right to the right, and the lower a bracket the nearer its partner its leg -- so no two cross at a state
    for i in range(n):
        legs = [(k, (b["to"] if b["from"] == i else b["from"]) > i, b["y"])
                for k, b in enumerate(brackets) if i in (b["from"], b["to"])]
        ordered = (sorted([leg for leg in legs if not leg[1]], key=lambda leg: leg[2])
                   + sorted([leg for leg in legs if leg[1]], key=lambda leg: -leg[2]))
        step = min(1.4 * scale, 1.5 * r[i] / max(len(ordered), 1))
        for pos, (k, _right, _y) in enumerate(ordered):
            brackets[k].setdefault("legs", {})[i] = x[i] + (pos - (len(ordered) - 1) / 2.0) * step
    # each bracket's rate over the middle of its span, moved along it off any higher bracket's leg (four states)
    for k, b in enumerate(brackets):
        lo, hi = sorted(b["legs"].values())
        w = wide(b["text"]) + 0.6 * scale
        crossing = [leg for other in brackets if other["y"] > b["y"] for leg in other["legs"].values()
                    if lo < leg < hi]
        mid = (lo + hi) / 2.0
        tries = [mid + s * d for d in [0.5 * scale * m for m in range(0, 40)] for s in (1, -1)]
        b["label_x"] = next((c for c in tries if lo + w / 2.0 <= c <= hi - w / 2.0
                             and not any(abs(leg - c) < w / 2.0 for leg in crossing)), mid)
    top = (brackets[-1]["y"] + (stroke + gap + th if labelled else stroke) if brackets
           else max(max(r), a + label_h)) + 0.3 * scale
    block_top = -(max(max(r), a + label_h) + 0.8 * scale)
    line_h = small * _PT_MM * 1.2
    bottom = block_top - max(len(block) for block in blocks) * line_h - 0.3 * scale
    return {"x": x, "r": r, "width": width, "top": top, "bottom": bottom, "a": a, "gap": gap, "pairs": pairs,
            "brackets": brackets, "blocks": blocks, "block_top": block_top, "small": small, "scale": scale,
            "head": head, "ms": ms, "off": off, "stroke": stroke, "spacing": spacing}


def _draw_hmm_chain(ax, entry: dict, chain: dict, spec: FigureSpec, rate_max: float) -> None:
    """``entry``'s state diagram as `_hmm_chain` placed it, on ``ax`` in mm (equal aspect, the limits its extent)."""
    from matplotlib.patches import Circle, FancyArrowPatch
    from matplotlib.path import Path as MplPath

    result = entry["result"]
    diagram, colour = result["diagram"], entry.get("colour") or QC_DATA
    page = _page(spec)
    ink, muted = page["ink"], page["muted"]
    x, r, a, gap, small, scale = chain["x"], chain["r"], chain["a"], chain["gap"], chain["small"], chain["scale"]
    rates = {(int(ar["from"]), int(ar["to"])): ar.get("rate") for ar in diagram["arrows"]}
    dashed = bool(diagram.get("not_identifiable"))
    line = spec.line_pt()
    lo_lw, hi_lw = (max(0.5, line * f) for f in _HMM_ARROW_LW)

    def lw(i: int, j: int) -> float:
        rate = rates.get((i, j))
        if rate is None or not rate_max or not math.isfinite(float(rate)):
            return lo_lw
        return lo_lw + (hi_lw - lo_lw) * max(0.0, min(1.0, float(rate) / rate_max))

    def arrow(i: int, j: int, **where):
        return FancyArrowPatch(arrowstyle=f"-|>,head_length={_HMM_HEAD[0]},head_width={_HMM_HEAD[1]}",
                               shrinkA=0, shrinkB=0, mutation_scale=chain["ms"], color=muted, lw=lw(i, j),
                               linestyle=(0, (3.0, 2.0)) if dashed else "-", capstyle="butt",
                               joinstyle="miter", zorder=2, clip_on=False, **where)

    ax.set_axis_off()
    for xi, ri in zip(x, r):
        ax.add_patch(Circle((xi, 0.0), ri, facecolor=_pale(spec, colour, 0.28), edgecolor=colour,
                            lw=max(0.6, line * 1.2), zorder=3, clip_on=False))

    def edge(i: int, y: float, toward: int) -> float:
        """Where an arrow at height ``y`` leaves (or meets) state ``i``'s circle on the side of ``toward``."""
        reach = math.sqrt(max(r[i] ** 2 - y ** 2, 0.0))
        return x[i] + (1 if toward > i else -1) * (reach + 0.3 * scale)

    for i, j, words in chain["pairs"]:
        y = a if j > i else -a
        start, end = edge(i, y, j), edge(j, y, i)
        ax.add_patch(arrow(i, j, posA=(start, y), posB=(end, y)))
        if words:
            off = chain["off"]                         # clear of the arrowhead, over or under its own arrow
            ax.text((start + end) / 2.0, y + (off if y > 0 else -off), words, ha="center",
                    va="bottom" if y > 0 else "top", fontsize=small, color=ink, zorder=5, clip_on=False)
    for b in chain["brackets"]:
        f, t, y = b["from"], b["to"], b["y"]
        xf, xt = b["legs"][f], b["legs"][t]
        yf = math.sqrt(max(r[f] ** 2 - (xf - x[f]) ** 2, 0.0)) + 0.3 * scale
        yt = math.sqrt(max(r[t] ** 2 - (xt - x[t]) ** 2, 0.0)) + 0.3 * scale
        c = max(0.05, min(0.8 * scale, (y - max(yf, yt)) / 2.0, abs(xt - xf) / 4.0))
        sign = 1.0 if xt > xf else -1.0
        verts = [(xf, yf), (xf, y - c), (xf, y), (xf + sign * c, y), (xt - sign * c, y), (xt, y), (xt, y - c),
                 (xt, yt)]
        codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO, MplPath.CURVE3,
                 MplPath.CURVE3, MplPath.LINETO]
        ax.add_patch(arrow(f, t, path=MplPath(verts, codes)))
        if b["text"]:
            ax.text(b["label_x"], y + chain["stroke"] + gap, b["text"], ha="center", va="bottom", fontsize=small,
                    color=ink, zorder=5, clip_on=False)
    for xi, block in zip(x, chain["blocks"]):
        ax.text(xi, chain["block_top"], "\n".join(block), ha="center", va="top", fontsize=small, color=ink,
                linespacing=1.2, zorder=5, clip_on=False)
    ax.set_xlim(0.0, chain["width"])
    ax.set_ylim(chain["bottom"], chain["top"])
    ax.set_aspect("equal", adjustable="box")


def _hmm_names(n: int) -> list:
    from .hmm_view import _names

    return _names(n)


def _hmm_settings_line(result: dict) -> str:
    s = result["settings"]
    fit = result["fit"]
    states = (f"Best by BIC over 1-3 (recommended {fit['n_states']})" if s["states"] == "best"
              else f"{s['states']} (fixed)")
    from .hmm_view import chunks_of

    interval = result.get("interval") or {}
    resampled = "movies" if s["uncertainty"] == "movie" else "cells"
    if s["uncertainty"] == "none":
        spread = "none"
    elif s["uncertainty"] == "chunk" and chunks_of(interval) and interval.get("values"):
        from .bubble_view import chunk_words

        spread = f"chunks (one fit per chunk: {chunk_words(chunks_of(interval))})"
    elif s["uncertainty"] in ("day", "chunk"):
        one, many = ("chunk", "chunks") if s["uncertainty"] == "chunk" else ("replicate", "replicates")
        spread = (f"{many} (one fit per {one}, each independent)"
                  if interval.get("values") else f"{many}: " + str(
                      interval.get("note") or f"no {one} could be fitted"))
    elif interval.get("withheld"):
        spread = f"{resampled}: " + str(interval["withheld"])
    else:
        spread = (f"{resampled} (95 % percentile intervals over {interval.get('n_boot', 0)} "
                  f"resamples of {interval.get('n_units', 0)} {resampled})")
    return (f"Settings: states {states}; uncertainty {spread}; σ correction "
            f"{result['sigma_words']}.")


def _hmm_source(entry: dict, spec: FigureSpec) -> tuple:
    """The states as the main table; the rates, the BIC rows and every replicate day's
    values as further sheets -- each as numbers."""
    result, name = entry["result"], entry["key"]
    shown = spec.shown_name(entry["name"])
    fit = result["fit"]
    n = int(fit["n_states"])
    names = _hmm_names(n)
    interval = result.get("interval") or {}
    cell = interval.get("mode") in ("cell", "movie") and not interval.get("withheld")
    units = "movies" if interval.get("mode") == "movie" else "cells"   # the resampled unit
    day_mean = interval.get("occupancy_mean")
    from .hmm_view import unit_words

    one, many = unit_words(interval)
    from .hmm_view import d_word

    columns = ["Condition", "Shown as", "State", f"D (µm²/s, {d_word(result)})", "Occupancy (0-1)",
               "Initial π (0-1)", "Stationary (0-1)", "Dwell time (s, model-derived)",
               "Left within two frames", f"Occupancy low (95 %, {units})",
               f"Occupancy high (95 %, {units})", f"D low (95 %, {units})", f"D high (95 %, {units})",
               f"Occupancy, mean over {many}", f"Occupancy, SD over {many}"]
    rows = []
    for k in range(n):
        occ_band = (interval.get("occupancy") or [None] * n)[k] if cell else None
        d_band = (interval.get("D") or [None] * n)[k] if cell else None
        rows.append([
            name, shown, names[k], float(fit["D"][k]), float(fit["occupancy"][k]),
            None if fit["initial"] is None else float(fit["initial"][k]),
            float(fit["stationary"][k]),
            None if fit["dwell_s"][k] is None else float(fit["dwell_s"][k]),
            bool(fit["transient"][k]),
            None if not occ_band else float(occ_band[0]),
            None if not occ_band else float(occ_band[1]),
            None if not d_band else float(d_band[0]),
            None if not d_band else float(d_band[1]),
            None if not day_mean else float(day_mean[k]),
            None if not day_mean else float(interval["occupancy_sd"][k])])
    sheets = []
    if n >= 2:
        bands = interval.get("rates", {}) if cell else {}
        stuck = bool(result["diagram"].get("not_identifiable"))
        rate_rows = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                band = bands.get(f"{i}>{j}")
                rate_rows.append([name, names[i], names[j], float(fit["rates"][i][j]),
                                  None if not band else float(band[0]),
                                  None if not band else float(band[1]), stuck])
        notes = ["Rate: per second, the fitted per-step transition probability divided by "
                 f"the frame interval ({1000 * result['tau_s']:.1f} ms). Rates are the measured quantity."]
        if cell:
            notes.append(f"Low / high: the 2.5th and 97.5th percentiles over "
                         f"{interval['n_boot']} resamples of {interval['n_units']} cells "
                         f"(rates are bounded at zero and skewed); {interval['n_degenerate']} "
                         "resamples held no transition out of a state.")
        if stuck:
            notes.append("Not identifiable: the resamples (or the fit) held no transition out of "
                         "a state, so the rate is not measured. Report the occupancy instead.")
        sheets.append(("Rates", "Switching rates between states", tuple(notes),
                       ["Condition", "From", "To", "Rate (/s)", f"Low (95 %, {units})",
                        f"High (95 %, {units})", "Not identifiable"], rate_rows))
    selection = result.get("selection")
    if selection:
        bic_rows = [[int(k), float(b), float(d), int(k) == int(selection["recommended"])]
                    for k, b, d in selection["rows"]]
        notes = ["BIC from the fit of each state count; the lowest is recommended. BIC "
                 "recommends a count; the choice is yours."]
        if selection.get("inconclusive"):
            notes.append(f"Inconclusive: the runner-up is within {selection['margin']:.1f} of "
                         "the best (under 2).")
        sheets.append(("State count (BIC)", "BIC for each state count", tuple(notes),
                       ["States", "BIC", "ΔBIC", "Recommended"], bic_rows))
    values = interval.get("values") or [] if interval.get("mode") == "day" else []
    if values:
        state_rows, rate_rows = [], []
        for v in values:
            m = int(v["n_states"])
            day_names = _hmm_names(m)
            for k in range(m):
                state_rows.append([v["day"], day_names[k], float(v["D"][k]),
                                   float(v["occupancy"][k]),
                                   None if v["initial"] is None else float(v["initial"][k]),
                                   float(v["stationary"][k]),
                                   None if v["dwell_s"][k] is None else float(v["dwell_s"][k]),
                                   int(v["n_tracks"]), int(v["n_steps"])])
            for i in range(m):
                for j in range(m):
                    if i != j:
                        rate_rows.append([v["day"], day_names[i], day_names[j],
                                          float(v["rates"][i][j]), bool(v["degenerate"])])
        failed = tuple(f"{day}: no fit ({why})" for day, why in interval.get("failed", []))
        Many, One = many[0].upper() + many[1:], one[0].upper() + one[1:]
        what = ("Each replicate is fitted on its own. Replicates are independent measurements and are not paired "
                "across conditions." if one == "replicate" else
                "Each chunk is fitted on its own. The chunks are random pseudo-replicates within one culture; they "
                "show how precisely these cells were measured, not where another culture would come out.")
        sheets.append((f"{Many}, states", f"Each {one}'s own fit: the states", (what, *failed),
                       [One, "State", f"D (µm²/s, {d_word(result)})", "Occupancy (0-1)",
                        "Initial π (0-1)", "Stationary (0-1)",
                        "Dwell time (s, model-derived)", "Tracks", "Steps"], state_rows))
        sheets.append((f"{Many}, rates", f"Each {one}'s own fit: the rates",
                       (f"No transition: the chain of that {one} never left a state, so its rates are "
                        "not measured.",),
                       [One, "From", "To", "Rate (/s)", "No transition"], rate_rows))
    return (columns, rows), tuple(sheets)


def _hmm_source_notes(entry: dict) -> tuple:
    from .hmm_view import card_caption

    result = entry["result"]
    notes = [_hmm_settings_line(result), card_caption(result) + ".",
             "Dwell time: τ / (1 − the probability of staying), the model's mean time in the "
             "state (exponential, Markovian). It is an extrapolation and usually longer than the "
             "tracks; the rates are the measured quantity.",
             "Occupancy: the time-average share of observed steps in the state. Tracks of "
             "5-6 steps are too short to mix, so it lies between the initial distribution π and the "
             "stationary mix and is not the stationary distribution implied by the rates.",
             "D: the state's apparent diffusion coefficient, µm²/s. The slow state's D sits at "
             "the localisation floor and the fast state's at the detection ceiling, so they are bounds, "
             "not physical coefficients."]
    from .hmm_view import corrected, per_spot

    if per_spot(result):
        notes[-1] = (f"D: the state's diffusion coefficient with every step's own localisation error modelled "
                     f"(σ correction {result['sigma_words']}), µm²/s. A bound state reads near zero, below the "
                     f"localisation floor.")
    elif corrected(result):
        notes[-1] = (f"D: the state's diffusion coefficient corrected for localisation error (σ correction "
                     f"{result['sigma_words']}), µm²/s. This is a sensitivity check and differs from the apparent D "
                     f"that SQD reports.")
    notes += [text for text, kind in result.get("lines", ()) if kind == "flag"]
    return tuple(notes)


def hmm_legend(entry: dict) -> str:
    """A drafted legend for the HMM state diagram, in the project's reporting order: the
    rates first; dwell times as model extrapolations; occupancy as a time-average between
    π and the stationary mix; D apparent."""
    result = entry["result"]
    fit, interval = result["fit"], result.get("interval") or {}
    n = int(fit["n_states"])
    names = _hmm_names(n)
    parts = [f"Diffusive-state hidden Markov model of {entry['name']}: {n} state"
             f"{'' if n == 1 else 's'} fitted to {fit['n_tracks']:,} tracks "
             f"({fit['n_steps']:,} steps, τ = {1000 * result['tau_s']:.1f} ms)."]
    if n >= 2:
        rates = "; ".join(f"{names[i]} to {names[j]} {fit['rates'][i][j]:.3g} /s"
                          for i in range(n) for j in range(n) if i != j)
        sentence = (f"Arrows give the switching rates between states per second, from the "
                    f"fitted per-step transition probabilities ({rates}); a wider arrow is a faster rate")
        if interval.get("mode") in ("cell", "movie") and interval.get("rates"):
            sentence += (f"; 95 % percentile intervals over {interval['n_boot']} "
                         f"resamples of {interval['n_units']} {interval['mode']}s are in the source data")
        elif interval.get("mode") == "day" and interval.get("values"):
            from .hmm_view import chunks_of

            sentence += (f"; each of the {len(interval['values'])} "
                         + ("random chunks (pseudo-replicates within one culture)" if chunks_of(interval)
                            else "replicates")
                         + " was also fitted on its own (source data)")
        parts.append(sentence + ".")
        if result["diagram"].get("not_identifiable"):
            parts.append("Dashed arrows: the rate is not identifiable because the resamples held no "
                         "transition out of a state. Report the occupancy instead.")
        dwell = ", ".join(f"{names[i]} {'never left' if d is None else f'{1000 * d:.0f} ms'}"
                          for i, d in enumerate(fit["dwell_s"]))
        parts.append(f"Dwell times ({dwell}) are model extrapolations (exponential, "
                     "Markovian) and usually exceed the observed tracks.")
    occ = " / ".join(f"{100 * v:.1f} %" for v in fit["occupancy"])
    sentence = (f"Circle area is the occupancy ({occ}): the time-average share of observed "
                "steps in each state")
    if fit["initial"] is not None and n >= 2:
        sentence += (". For tracks of 5-6 steps it lies between the initial distribution ("
                     + " / ".join(f"{100 * v:.1f} %" for v in fit["initial"])
                     + ") and the stationary mix ("
                     + " / ".join(f"{100 * v:.1f} %" for v in fit["stationary"])
                     + ") and is not the stationary distribution implied by the rates")
    parts.append(sentence + ".")
    from .hmm_view import corrected, per_spot

    if per_spot(result):
        parts.append("D is each state's diffusion coefficient with every step's own localisation error modelled "
                     f"(µm²/s; σ correction {result['sigma_words']}). A bound state reads near zero, below the "
                     "localisation floor.")
    elif corrected(result):
        parts.append("D is each state's diffusion coefficient corrected for localisation error (µm²/s; σ "
                     f"correction {result['sigma_words']}). This is a sensitivity check: a slow state that "
                     "vanishes under it was localisation noise.")
    else:
        parts.append("D is each state's apparent diffusion coefficient (µm²/s; σ correction "
                     f"{result['sigma_words']}). The slow state's D sits at the localisation floor "
                     "and the fast state's at the detection ceiling, so they are bounds, not physical "
                     "coefficients.")
    selection = result.get("selection")
    if selection:
        deltas = ", ".join(f"{k} state{'' if int(k) == 1 else 's'} ΔBIC {d:.1f}" for k, _b, d in selection["rows"])
        counts = [int(k) for k, _b, _d in selection["rows"]]
        parts.append(f"The state count was chosen by BIC over {min(counts)}-{max(counts)} ({deltas}); BIC recommends a "
                     "count, the choice is yours."
                     + (" The call is inconclusive (runner-up within 2)."
                        if selection.get("inconclusive") else ""))
        if int(selection["recommended"]) >= 3:
            parts.append("A third state's D often sits at the localisation floor and may reflect "
                         "measurement heterogeneity instead of a separate population.")
    return " ".join(parts)


# -- the HMM's state diagrams in a figure: one condition, or every condition ---------------- #
# Asked 5 Nov 2026: "we need to be able to export multiple conditions as one image for hmm. with the bubbles, fraction
# sizes, rates, D for all four conditions here", then, having tried it: "first i think we need to select which
# conditions to take, for example if i just wanna show two of them". "hmm_overview" draws the diagrams of the
# conditions the dialog keeps (`FigureSpec.left_out`, ticked off beside each name) in a grid, each named over its
# diagram, one key under them; "hmm_condition" (a card's own Export) is the same figure of one, its name in the key.

#: Between two cells across, and between two rows of cells (mm).
_HMM_OVERVIEW_GAP_MM = (3.0, 2.5)
#: The kinds whose conditions the Export dialog can leave out, a box beside each name (`FigureSpec.left_out`).
PICK_KINDS = ("hmm_overview",)


def _hmm_overview_entries(payload: dict, spec: FigureSpec | None = None) -> list:
    """The entries drawn: the payload's, less those ``spec`` leaves out (`FigureSpec.left_out`), each with a diagram;
    refused by name otherwise."""
    entries = (payload or {}).get("entries") or []
    if not entries:
        raise FigureExportError("no HMM result to draw yet")
    left_out = set(spec.left_out) if spec is not None else set()
    shown = [e for e in entries if e.get("name") not in left_out]
    if not shown:
        raise FigureExportError("every condition is left out: tick at least one beside its name")
    for e in shown:
        result = e.get("result") or {}
        name = e.get("name", "a condition")
        if "error" in result:
            raise FigureExportError(f"{name} has no HMM fit to draw: {result['error']}")
        if "diagram" not in result:
            raise FigureExportError(f"{name} has no HMM fit to draw")
    return shown


def _hmm_key(entries: list, spec: FigureSpec, named: bool) -> str:
    """The key under the diagrams: what a circle, a D and an arrow are, each said once ("hmm_condition": the
    condition's name first). A cell interval is named by its unit, its resample count being each condition's own."""
    from .hmm_view import d_word

    words = []
    for e in entries:
        word = d_word(e["result"])
        if word not in words:
            words.append(word)
    parts = ([spec.shown_name(entries[0]["name"])] if named else []) + [
        "circle area = occupancy", f"D {' or '.join(words)} (µm²/s)"]
    # one "±" part for whatever carries one: rates and occupancies of one fit spread over the same unit, so one phrase
    whats = []
    if spec.occupancy_spread and spec.numbers:
        for e in entries:
            for k in range(len(e["result"]["diagram"]["nodes"])):
                _value, what = _hmm_occupancy_spread(e["result"], k)
                if what and what not in whats:
                    whats.append(what)
    if any(len(e["result"]["diagram"]["arrows"]) for e in entries):
        parts.append("arrows: switching rates per second, wider = faster")
        if spec.rate_spread:
            for e in entries:
                for ar in e["result"]["diagram"]["arrows"]:
                    _value, what = _hmm_rate_spread(e["result"], int(ar["from"]), int(ar["to"]))
                    if what and what not in whats:
                        whats.append(what)
    if whats:
        parts.append("± " + "; ".join(whats))
    if any(e["result"]["diagram"].get("not_identifiable") for e in entries):
        parts.append("dashed: not identifiable")
    return " · ".join(parts)


def _hmm_layout(payload: dict, spec: FigureSpec, family: str, kind: str = "hmm_overview") -> dict:
    """Where the figure's diagrams stand (mm): ``{"entries", "chains", "columns", "rows", "cell_w", "names",
    "name_h", "row_top", "row_bottom", "gap", "key_lines", "key_h", "title_h", "height", "rate_max"}``. Each diagram is
    as wide as its text needs (`_hmm_chain`); Auto puts as many across as hold the widest, then as few as keep the
    rows (`_overview_grid`'s rule), the Grid row a number of columns (`FigureSpec.grid_columns`). A diagram wider
    than its cell, or a figure taller than a typed height, is drawn smaller as a whole, text included, never under
    `_HMM_MIN_SCALE`. Within a row the diagrams' centre lines are level."""
    named = kind == "hmm_overview"
    entries = _hmm_overview_entries(payload, spec if named else None)
    n = len(entries)
    gap_x, gap_y = _HMM_OVERVIEW_GAP_MM
    usable = spec.width_value() - 1.6
    pt, small = spec.pt(), spec.small_pt()
    chains = [_hmm_chain(e, spec, family) for e in entries]
    names = [spec.shown_name(e["name"]) for e in entries] if named else [""] * n
    natural = max(max(c["width"] for c in chains),
                  max(_text_width_pt(s, pt, family) * _PT_MM for s in names) if named else 0.0) + 0.6
    if spec.grid_columns:
        columns = max(1, min(n, int(spec.grid_columns)))
    else:
        columns = max(1, min(n, int((usable + gap_x) // (natural + gap_x))))
        while columns > 1 and math.ceil(n / (columns - 1)) == math.ceil(n / columns):
            columns -= 1
    rows = math.ceil(n / columns)
    cell_w = (usable - gap_x * (columns - 1)) / columns
    measure = lambda s: _text_width_pt(s, small, family)                   # noqa: E731
    # 0.88 of the width: raster text is hinted wider than its outline measures
    key_lines = (chart_text.wrap_text(_hmm_key(entries, spec, not named), measure, usable / _PT_MM * 0.88,
                                      max_lines=3) if spec.legend else [])
    key_h = len(key_lines) * small * 1.25 * _PT_MM + (1.0 if key_lines else 0.0)
    title_h = pt * 1.6 * _PT_MM if spec.title else 0.0

    def place(scale: float) -> dict:
        drawn = [dict(c) for c in (chains if scale >= 1.0 else [_hmm_chain(e, spec, family, scale)
                                                                 for e in entries])]
        # STACKED, EACH STATE OVER THE SAME STATE: the diagrams of one grid column with the same number of states
        # drawn again with the room the widest of them needs at each place (asked for a 4 x 1 grid, 5 Nov 2026)
        if rows > 1:
            for column in range(columns):
                stack = [k for k in range(column, n, columns)]
                for count in {len(drawn[k]["x"]) for k in stack}:
                    same = [k for k in stack if len(drawn[k]["x"]) == count]
                    if len(same) < 2:
                        continue
                    floors = (max(drawn[k]["spacing"][0] for k in same),
                              [max(drawn[k]["spacing"][1][g] for k in same) for g in range(count - 1)],
                              max(drawn[k]["spacing"][2] for k in same))
                    for k in same:
                        drawn[k] = _hmm_chain(entries[k], spec, family, scale, floors)
        wrapped = [chart_text.wrap_text(s, lambda t: _text_width_pt(t, pt, family), cell_w / _PT_MM * 0.95,
                                        max_lines=2) if named else [] for s in names]
        row_top, row_bottom, name_h = [], [], []
        for row in range(rows):
            members = range(row * columns, min(n, (row + 1) * columns))
            # the names, D and occupancies of a row on one line: each block as low as the row's lowest
            line = min(drawn[k]["block_top"] for k in members)
            for k in members:
                drawn[k]["bottom"] -= drawn[k]["block_top"] - line
                drawn[k]["block_top"] = line
            row_top.append(max(drawn[k]["top"] for k in members))
            row_bottom.append(min(drawn[k]["bottom"] for k in members))
            lines = max((len(wrapped[k]) for k in members), default=0)
            name_h.append(lines * pt * 1.25 * _PT_MM + (0.6 if lines else 0.0))
        height = (1.6 + title_h + key_h + gap_y * (rows - 1)
                  + sum(h + t - b for h, t, b in zip(name_h, row_top, row_bottom)))
        return {"chains": drawn, "names": wrapped, "name_h": name_h, "row_top": row_top, "row_bottom": row_bottom,
                "height": height}

    scale = max(_HMM_MIN_SCALE, min(1.0, cell_w / natural))
    laid = place(scale)
    if spec.height_mm is not None and laid["height"] > float(spec.height_mm) and scale > _HMM_MIN_SCALE:
        fixed = 1.6 + title_h + key_h + gap_y * (rows - 1) + sum(laid["name_h"])
        drawn_h = laid["height"] - fixed
        scale = max(_HMM_MIN_SCALE, scale * max(0.0, float(spec.height_mm) - fixed) / max(drawn_h, 1e-9))
        laid = place(scale)
    return {"entries": entries, "columns": columns, "rows": rows, "cell_w": cell_w, "gap": (gap_x, gap_y),
            "key_lines": key_lines, "key_h": key_h, "title_h": title_h, "rate_max": _hmm_rate_max(entries),
            "scale": scale, **laid, "height": round(laid["height"], 1)}


def _hmm_figure_height(payload: dict, spec: FigureSpec, kind: str = "hmm_overview") -> float:
    family, _note = resolve_font(spec.font)
    return _hmm_layout(payload, replace(spec, height_mm=None), family, kind)["height"]


def _draw_hmm_figure(fig, payload: dict, spec: FigureSpec, family: str, kind: str = "hmm_overview"):
    """The state diagrams as `_hmm_layout` placed them: each condition's (`_draw_hmm_chain`) centred in its cell, in
    the entries' order (the section's, `condition_order`) across and then down, its name over it, the key under them
    all. Every diagram is placed in MILLIMETRES (the layout engine off on a whole figure): its text was measured on
    the promise that a millimetre of its axes is one of paper. The grid stands at the figure's bottom, so the room a
    composed panel gives its letter stays free over it. Returns the first diagram's axes."""
    from matplotlib.figure import Figure

    lay = _hmm_layout(payload, spec, family, kind)
    page = _page(spec)
    width_mm, height_mm = spec.size_mm()
    gap_x, gap_y = lay["gap"]
    cell_w = lay["cell_w"]
    y = lay["height"] - 0.8 - lay["title_h"]                    # the next row's top, mm from the figure's bottom
    first = None
    for row in range(lay["rows"]):
        line_y = y - lay["name_h"][row] - lay["row_top"][row]  # the row's centre line
        for k in range(row * lay["columns"], min(len(lay["entries"]), (row + 1) * lay["columns"])):
            entry, chain = lay["entries"][k], lay["chains"][k]
            x0 = 0.8 + (k - row * lay["columns"]) * (cell_w + gap_x)
            # each line of the name on its BASELINE: a top-aligned name stood as high as its tallest letter, so
            # "glucose" and "malate" sat at two heights in one row
            for m, words in enumerate(lay["names"][k]):
                fig.text((x0 + cell_w / 2.0) / width_mm,
                         (y - 0.3 - spec.pt() * _PT_MM * (0.8 + 1.25 * m)) / height_mm, words,
                         ha="center", va="baseline", fontsize=spec.pt(), color=page["ink"])
            left = x0 + (cell_w - chain["width"]) / 2.0
            bottom = line_y + chain["bottom"]
            ax = fig.add_axes((left / width_mm, bottom / height_mm, chain["width"] / width_mm,
                               (chain["top"] - chain["bottom"]) / height_mm))
            _draw_hmm_chain(ax, entry, chain, spec, lay["rate_max"])
            first = first or ax
        y -= lay["name_h"][row] + lay["row_top"][row] - lay["row_bottom"][row] + gap_y
    if lay["key_lines"]:
        fig.text(0.5, (0.8 + lay["key_h"] - 0.6) / height_mm, "\n".join(lay["key_lines"]), ha="center", va="top",
                 fontsize=spec.small_pt(), color=page["muted"], linespacing=1.2)
    if isinstance(fig, Figure):
        fig.set_layout_engine("none")
    return first


def _hmm_overview_source(payload: dict, spec: FigureSpec) -> tuple:
    """Every condition's source data (`_hmm_source`) in one workbook: the states as the main table, each further
    sheet once, its rows every condition's -- named by a Condition column where the card's sheet had none."""
    columns, rows, sheets, order = None, [], {}, []
    for e in _hmm_overview_entries(payload, spec):
        (cols, part), more = _hmm_source(e, spec)
        columns = columns or cols
        rows += part
        for title, caption, notes, s_cols, s_rows in more:
            if s_cols[0] != "Condition":
                s_cols = ["Condition", *s_cols]
                s_rows = [[e["key"], *row] for row in s_rows]
            if title not in sheets:
                sheets[title] = [caption, [], s_cols, []]
                order.append(title)
            sheets[title][1] += [n for n in notes if n not in sheets[title][1]]
            sheets[title][3] += s_rows
    return (columns, rows), tuple((t, sheets[t][0], tuple(sheets[t][1]), sheets[t][2], sheets[t][3]) for t in order)


def _hmm_overview_notes(payload: dict, spec: FigureSpec) -> tuple:
    """The cards' source notes: those every condition shares once, then each condition's own, named."""
    per = [(spec.shown_name(e["name"]), _hmm_source_notes(e)) for e in _hmm_overview_entries(payload, spec)]
    shared = [n for n in per[0][1] if all(n in notes for _name, notes in per)]
    return tuple(shared) + tuple(f"{name}: {n}" for name, notes in per for n in notes if n not in shared)


def hmm_overview_legend(payload: dict, spec: FigureSpec | None = None) -> str:
    """A drafted legend for the state diagrams of every condition: what each condition's fit holds (its tracks, its
    occupancies, its rates), then what a circle, an arrow, a dwell time and a D are, said once."""
    from .hmm_view import corrected, per_spot

    entries = _hmm_overview_entries(payload, spec)
    shown = (spec.shown_name if spec is not None else (lambda n: n))
    taus = {round(1000 * e["result"]["tau_s"], 1) for e in entries}
    parts = [f"Diffusive-state hidden Markov models, one state diagram per condition, in reading order ("
             + ", ".join(shown(e["name"]) for e in entries) + ")"
             + (f"; τ = {taus.pop():.1f} ms." if len(taus) == 1 else ".")]
    for e in entries:
        result = e["result"]
        fit = result["fit"]
        n = int(fit["n_states"])
        names = _hmm_names(n)
        sentence = (f"{shown(e['name'])}: {n} state{'' if n == 1 else 's'}, {fit['n_tracks']:,} tracks "
                    f"({fit['n_steps']:,} steps"
                    + (f", τ = {1000 * result['tau_s']:.1f} ms" if len(taus) > 1 else "") + "); occupancy "
                    + " / ".join(f"{100 * v:.1f} %" for v in fit["occupancy"]))
        if n >= 2:
            sentence += "; " + "; ".join(f"{names[i]} to {names[j]} {fit['rates'][i][j]:.3g} /s"
                                         for i in range(n) for j in range(n) if i != j)
        parts.append(sentence + ".")
    if any(int(e["result"]["fit"]["n_states"]) >= 2 for e in entries):
        parts.append("States stand in a row from slow to fast. Arrows give the switching rates between states "
                     "(per second, from the fitted per-step transition probabilities); a wider arrow is a faster "
                     "rate, on one scale for the figure. Intervals and each replicate's fit are in the source data.")
        if spec is not None and spec.rate_spread:
            whats = []
            for e in entries:
                for ar in e["result"]["diagram"]["arrows"]:
                    _value, what = _hmm_rate_spread(e["result"], int(ar["from"]), int(ar["to"]))
                    if what and what not in whats:
                        whats.append(what)
            if whats:
                parts.append("The ± beside a rate is " + "; ".join(whats) + ".")
        if any(e["result"]["diagram"].get("not_identifiable") for e in entries):
            parts.append("Dashed arrows: the rate is not identifiable because the resamples held no transition out "
                         "of a state. Report the occupancy instead.")
        parts.append("Dwell times (source data) are model extrapolations (exponential, Markovian) and usually "
                     "exceed the observed tracks.")
    parts.append("Circle area is the occupancy: the time-average share of observed steps in each state. For "
                 "tracks of 5-6 steps it lies between the initial distribution and the stationary mix (source data) "
                 "and is not the stationary distribution implied by the rates.")
    if spec is not None and spec.occupancy_spread:
        whats = []
        for e in entries:
            for k in range(len(e["result"]["diagram"]["nodes"])):
                _value, what = _hmm_occupancy_spread(e["result"], k)
                if what and what not in whats:
                    whats.append(what)
        if whats:
            parts.append("The ± beside an occupancy is " + "; ".join(whats) + ".")
    if entries and all(per_spot(e["result"]) for e in entries):
        parts.append("D is each state's diffusion coefficient with every step's own localisation error modelled "
                     "(µm²/s, per-spot precision through a calibration measured on immobile spots). A bound state "
                     "reads near zero, below the localisation floor.")
    elif any(corrected(e["result"]) for e in entries):
        parts.append("D is each state's diffusion coefficient corrected for localisation error (µm²/s). This is a "
                     "sensitivity check: a slow state that vanishes under it was localisation noise.")
    else:
        parts.append("D is each state's apparent diffusion coefficient (µm²/s). The slow state's D sits at the "
                     "localisation floor and the fast state's at the detection ceiling, so they are bounds, not "
                     "physical coefficients.")
    chosen = [e for e in entries if e["result"].get("selection")]
    if chosen:
        parts.append("Where the state count was chosen by BIC (source data), BIC recommends a count and the choice "
                     "is yours.")
        if any(int(e["result"]["selection"]["recommended"]) >= 3 for e in chosen):
            parts.append("A third state's D often sits at the localisation floor and may reflect measurement "
                         "heterogeneity, not a separate population.")
    return " ".join(parts)


# -- the HMM across conditions ------------------------------------------------------ #
#: A panel of the HMM comparison needs at least this many ems of the body text across:
#: four stand in a row when each gets that much, else two by two, else one above another.
_COMPARE_PANEL_EM = 11.0
#: A condition's replicate days are spread sideways by this much each, in columns (the
#: manuscript's 0.055 between three), and over this much at most; the mean and its bar
#: stand this far right of the column's middle.
_DAY_STEP, _DAY_SPAN, _MEAN_AT = 0.07, 0.28, 0.24
#: Where a pooled fit's 95 % cell band reaches either side of its column, in columns.
_BAND_HALF = 0.15


def _comparison_rows(payload) -> list:
    """The conditions the comparison draws, in its order: the first panel's rows (every
    panel holds one row per condition, `hmm_view.comparison_payload`)."""
    panels = (payload or {}).get("panels") or []
    return list(panels[0].get("rows") or []) if panels else []


def _comparison_drawn(row: dict) -> str:
    """How one condition's number is drawn in a panel, in the words the source data use."""
    if row.get("days"):
        return f"{(row.get('unit') or ('replicate', 'replicates'))[1]}, their mean ± 1 SD"
    if row.get("value") is None:
        return "no value"
    if row.get("band") is not None:
        return f"pooled fit, 95 % {row.get('band_unit') or 'cell'} interval"
    return "pooled fit, no spread"


def _comparison_unit(payload: dict) -> tuple:
    """``(one, many)``: what the comparison's open circles are, "replicate(s)" or, under Uncertainty Chunks,
    "chunk(s)" (`hmm_view.unit_words`, carried on each row; the Run's Uncertainty is one for every condition)."""
    for p in payload.get("panels") or []:
        for r in p.get("rows") or []:
            if r.get("days") and r.get("unit"):
                return tuple(r["unit"])
    return ("replicate", "replicates")


def _pair_lines_drawn(payload: dict, spec: FigureSpec) -> bool:
    """Whether the comparison draws its pair lines: asked (`FigureSpec.pair_lines`) and a tag two conditions share."""
    return bool(getattr(spec, "pair_lines", False) and (payload or {}).get("paired"))


def _comparison_key(payload: dict, spec: FigureSpec | None = None) -> list:
    """``[(what, label)]``: one key entry per way a number is drawn somewhere in the figure."""
    rows = [r for p in payload.get("panels") or [] for r in p.get("rows") or []]
    key = []
    if any(r.get("days") for r in rows):
        one, many = _comparison_unit(payload)
        key += [("day", f"{one} (no order)"),
                ("mean", f"mean of the {many} ± 1 SD (spread, not a CI)")]
    banded = {r.get("band_unit") or "cell" for r in rows
              if not r.get("days") and r.get("value") is not None and r.get("band") is not None}
    if banded:
        key.append(("band", f"pooled fit, 95 % {' or '.join(sorted(banded, reverse=True))} interval"))
    if any(not r.get("days") and r.get("value") is not None and r.get("band") is None
           for r in rows):
        key.append(("lone", "pooled fit, no spread"))
    if spec is not None and _pair_lines_drawn(payload, spec):
        key.append(("pair", "replicates declared as a pair"))
    return key


def _titled_unit(title: str, unit: str, measure, width: float) -> list:
    """``title (unit)`` in lines no wider than ``width``, the unit never split: wrapped on
    its own, "(/s)" broke after its slash."""
    lines = chart_text.wrap_text(title, measure, width, max_lines=3)
    unit = f"({unit})"
    if measure(f"{lines[-1]} {unit}") <= width:
        lines[-1] = f"{lines[-1]} {unit}"
    else:
        lines.append(unit)
    return lines


def hmm_comparison_panels(payload: dict, spec: FigureSpec) -> dict:
    """The HMM comparison's payload with the panels ``spec`` asks for: the DWELL PANEL (`FigureSpec.hmm_dwell`, 12 Nov
    2026) put in after the rate out, or taken out; every other panel as the strip drew it. The payload carries the
    dwell panel either way (``"dwell_panel"``, `hmm_view.comparison_payload`), so nothing is recomputed. A payload
    without it (made before) is returned as it is."""
    payload = dict(payload or {})
    panels = list(payload.get("panels") or [])
    has = any(p.get("key") == "dwell" for p in panels)
    want = bool(getattr(spec, "hmm_dwell", False))
    if want and not has and payload.get("dwell_panel"):
        keys = [p.get("key") for p in panels]
        at = keys.index("rate_out") + 1 if "rate_out" in keys else len(panels)
        panels.insert(at, payload["dwell_panel"])
    elif not want and has:
        panels = [p for p in panels if p.get("key") != "dwell"]
    payload["panels"] = panels
    return payload


def _comparison_layout(payload: dict, spec: FigureSpec, family: str) -> dict:
    """How the HMM comparison sits at ``spec``'s width -- the way the jump-distance figure
    chooses wide or stacked, but by what a panel must hold: FOUR IN A ROW when each panel
    gets `_COMPARE_PANEL_EM` ems of the body text and every condition's name fits under its
    column (wrapped at spaces, three lines at most), else TWO BY TWO, else one above another.
    Where no layout lets the names lie flat they stand upright (never cut, never overlapping).

    Returns ``{"columns", "rows", "names" (as drawn), "upright", "titles" (each panel's,
    wrapped), "key", "key_labels", "key_columns", "key_mm", "plot_mm", "gap_mm", "height"}``;
    ``height`` is the figure's own and the composer's natural height at this width."""
    panels = payload.get("panels") or []
    rows_drawn = _comparison_rows(payload)
    grouping = payload.get("layout") or {}
    grouped = bool(grouping.get("groups")) and len(grouping.get("x") or ()) == len(rows_drawn)
    # GROUPED (folder designs): each column named by its inner values, the outer value a header under its group
    names = ([str(v) for v in grouping["labels"]] if grouped
             else [spec.shown_name(r["name"]) for r in rows_drawn])
    n = max(1, len(names)) if not grouped else max(1.0, grouping["x"][-1] - grouping["x"][0] + 1.0)
    pt, small = spec.pt(), spec.small_pt()
    width = spec.width_value()
    em = pt * _PT_MM
    gap = em
    measure = lambda s: _text_width_pt(s, small, family)                   # noqa: E731
    # the y axis's tick labels, their ticks and pads, left of every panel
    ticks = (measure("0.00") + pt * 0.4 + small * 0.8) * _PT_MM

    def plot_w(columns: int) -> float:
        return (width - gap * (columns - 1)) / columns - ticks - 1.0

    def room(columns: int) -> float:
        # pt per column, less a margin: `_text_width_pt` measures the glyphs' outlines, and
        # raster text is hinted about 10 % wider (the HMM diagram's key, measured)
        return plot_w(columns) / n / _PT_MM * 0.85

    def flat(columns: int) -> bool:
        r = room(columns)
        return all(measure(word) <= r for name in names for word in name.split()) \
            and all(len(chart_text.wrap_text(name, measure, r, max_lines=99)) <= 3
                    for name in names)

    # four panels: four in a row or two by two, as before; five (the dwell, 12 Nov 2026): five in a row, else three
    # and two, else two across
    count = max(1, len(panels))
    candidates = [count] + [c for c in (math.ceil(count / 2), 2) if c < count]
    wide_enough = [c for c in dict.fromkeys(candidates) if (width - gap * (c - 1)) / c / em >= _COMPARE_PANEL_EM]
    columns = next((c for c in wide_enough if flat(c)),
                   wide_enough[-1] if wide_enough else 1)
    upright = not flat(columns)
    if upright:
        drawn = names
        names_mm = max([measure(name) for name in names] + [small]) * _PT_MM + 1.0
    else:
        drawn = ["\n".join(chart_text.wrap_text(name, measure, room(columns), max_lines=3))
                 for name in names]
        names_mm = max([d.count("\n") + 1 for d in drawn] + [1]) * small * 1.25 * _PT_MM
    if grouped:
        names_mm += small * 1.9 * _PT_MM                  # the group headers' row
    title_room = plot_w(columns) / _PT_MM * 0.88
    titles = ["\n".join(_titled_unit(p["title"], p["unit"],
                                     lambda s: _text_width_pt(s, pt, family), title_room))
              for p in panels]
    title_mm = (max([t.count("\n") + 1 for t in titles] + [1]) * pt * 1.25 + 6.0) * _PT_MM
    key = _comparison_key(payload, spec) if spec.legend else []
    labels, key_columns = _legend_columns([label for _w, label in key],
                                          width / _PT_MM * 0.94, spec, family) \
        if key else ([], 1)
    key_rows = math.ceil(len(labels) / key_columns) if labels else 0
    tallest = max([text.count("\n") + 1 for text in labels] + [1])
    key_mm = (key_rows * tallest * small * 1.6 + small * 0.8) * _PT_MM if labels else 0.0
    rows = math.ceil(count / columns)
    # three quarters of its width, kept between 8 and 16 lines of the body text: two by two
    # at two columns' width the plots grew to 61 mm and the figure to 163 mm
    plot_mm = min(max(plot_w(columns) * 0.75, 8.0 * em), 16.0 * em)
    row_mm = plot_mm + title_mm + names_mm + (small * 1.2 + 4.0) * _PT_MM + 2.0
    height = rows * row_mm + key_mm + (pt * 1.6 * _PT_MM if spec.title else 0.0) + 2.0
    return {"columns": columns, "rows": rows, "names": drawn, "upright": upright,
            "grouping": grouping if grouped else {},
            "titles": titles, "key": key, "key_labels": labels, "key_columns": key_columns,
            "key_mm": key_mm, "plot_mm": plot_mm, "row_mm": row_mm, "gap_mm": gap,
            "height": round(height, 1)}


def _draw_hmm_comparison(fig, payload: dict, spec: FigureSpec, family: str, marks=()):
    """The HMM section's "Across conditions · the slow state" card in print, drawn from the
    card's own payload (`hmm_view.comparison_payload`) and nothing recomputed: four panels --
    the slow state's occupancy, the rate into it, the rate out of it, its apparent D -- each
    condition a column, named under it, in its own colour. The manuscript's construction
    (`scripts/figure_hmm_four_panel.py`): replicate days as OPEN circles, identical, spread
    sideways by a fixed step (no order implied, none to read), their mean FILLED beside them
    with a +-1 SD bar -- the day-to-day spread; a pooled fit with its 95 % cell interval as
    a pale BAND, a different glyph for a different quantity; a pooled fit alone as a filled
    circle; "n/a" where there is no number (a one-state fit has no rates). Every axis
    starts at zero, unless the card's Axes typed a panel's range (payload ``"axes"``,
    ``{panel key: (from, to)}``, drawn as shown). The key sits in a row of its own over the panels.
    ``marks``: the significance brackets (`_hmm_bracket_plan`), each in its panel's top band, tiered by the screen's
    rule (`significance.bracket_tiers`); the data keep the plot under the band.

    Returns the first panel's axes."""
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.ticker import MaxNLocator

    from .axes_view import apply_range

    panels = payload.get("panels") or []
    if not _comparison_rows(payload):
        raise FigureExportError("no HMM fit to compare yet: the comparison holds no condition")
    lay = _comparison_layout(payload, spec, family)
    columns, n_rows = lay["columns"], lay["rows"]
    page = _page(spec)
    pt, small, line = spec.pt(), spec.small_pt(), spec.line_pt()
    grey = page["muted"]
    stroke = max(0.6, line)
    day_ms, mean_ms = 0.62 * pt, 0.8 * pt
    cap = 0.35 * pt
    panel_mm = (spec.width_value() - lay["gap_mm"] * (columns - 1)) / columns
    widths = [r for c in range(columns)
              for r in ((panel_mm, lay["gap_mm"]) if c < columns - 1 else (panel_mm,))]
    heights = [lay["plot_mm"]] * n_rows
    if lay["key"]:
        # THE KEY IN A GRID OF ITS OWN, over the panels' grid. Sharing one grid with the
        # panels, the key's axes spanned every column, and constrained layout gave every
        # column it "submerged" the widest margins of all (`match_submerged_margins`): 13 mm
        # of nothing between panels at two columns' width.
        outer = fig.add_gridspec(2, 1, height_ratios=(lay["key_mm"],
                                                      lay["row_mm"] * n_rows))
        grid = outer[1].subgridspec(n_rows, len(widths), width_ratios=widths,
                                    height_ratios=heights)
        holder = fig.add_subplot(outer[0])
        holder.set_axis_off()
        handles = []
        for what, _label in lay["key"]:
            if what == "day":
                handles.append(Line2D([], [], marker="o", ls="none", ms=day_ms,
                                      mfc="none", mec=grey, mew=stroke))
            elif what == "mean":
                handles.append((Line2D([], [], marker="|", ls="none", ms=pt * 1.5, mew=stroke,
                                       color=grey),
                                Line2D([], [], marker="o", ls="none", ms=mean_ms, color=grey,
                                       mec=page["bg"], mew=0.4)))
            elif what == "band":
                handles.append((Patch(facecolor=_pale(spec, grey, 0.3), edgecolor="none"),
                                Line2D([], [], marker="o", ls="none", ms=mean_ms, color=grey,
                                       mec=page["bg"], mew=0.4)))
            elif what == "pair":
                handles.append(Line2D([], [], color=grey, lw=max(MIN_LINE_PT, stroke * 0.8)))
            else:
                handles.append(Line2D([], [], marker="o", ls="none", ms=mean_ms, color=grey,
                                      mec=page["bg"], mew=0.4))
        holder.legend(handles, lay["key_labels"], loc="center", ncol=lay["key_columns"],
                      frameon=False, handletextpad=0.4, columnspacing=1.2,
                      borderaxespad=0.0, handlelength=1.6,
                      handler_map={tuple: HandlerTuple(ndivide=1, pad=0.0)})
    else:
        grid = fig.add_gridspec(n_rows, len(widths), width_ratios=widths, height_ratios=heights)
    axes = []
    for index, panel in enumerate(panels):
        r, c = divmod(index, columns)
        ax = fig.add_subplot(grid[r, 2 * c])
        axes.append(ax)
        rows = panel.get("rows") or []
        n = max(1, len(rows))
        tops = []
        for row in rows:
            tops += list(row.get("days") or [])
            if row.get("value") is not None:
                tops.append(row["value"])
            if row.get("mean") is not None and row.get("sd") is not None:
                tops.append(row["mean"] + row["sd"])
            if row.get("band") is not None:
                tops.append(row["band"][1])
        top = max(tops) if tops and max(tops) > 0 else 1.0
        grouping = lay["grouping"]
        X = [float(x) for x in grouping["x"]] if grouping else [float(k) for k in range(len(rows))]
        ax.set_xlim(X[0] - 0.6 if X else -0.6, X[-1] + 0.6 if X else n - 0.4)
        # every axis from zero -- or where the card's Axes put this panel (30 Oct 2026; payload "axes", as shown)
        y_lo, _y_hi = apply_range(0.0, top * 1.1, (payload.get("axes") or {}).get(panel.get("key")))
        ax.set_ylim(y_lo, _y_hi)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        mine = [dict(m) for m in marks if m.get("population") == panel.get("key")]
        if mine:
            # THE BRACKETS' BAND at the panel's top: the data keep the plot under it, the ticks only the data's range
            from matplotlib.transforms import blended_transform_factory

            from . import significance as sig

            x_mm = max(1.0, panel_mm - ((_text_width_pt("0.00", small, family) + pt * 0.4 + small * 0.8) * _PT_MM)) \
                / ((X[-1] + 0.6) - (X[0] - 0.6) if X else 1.0)
            for m in mine:
                m["x0"], m["x1"] = X[m["a"]], X[m["b"]]
            tiers = sig.bracket_tiers(mine, lambda m: (_text_width_pt(m["text"], small, family) * _PT_MM / 2 + 0.6)
                                      / x_mm)
            tier = small * 1.8 / (lay["plot_mm"] / _PT_MM)
            band = min(0.6, (max(tiers) + 1) * tier)
            ticks = [v for v in MaxNLocator(nbins=4, min_n_ticks=3).tick_values(y_lo, _y_hi) if y_lo <= v <= _y_hi]
            ax.set_ylim(y_lo, y_lo + (_y_hi - y_lo) / (1.0 - band))
            ax.set_yticks(ticks)
            at = blended_transform_factory(ax.transData, ax.transAxes)
            ink, width = page["ink"], max(MIN_LINE_PT, line * 0.8)
            for m, level in zip(mine, tiers):
                y = 1.0 - band + (level + 0.3) * tier
                ax.add_artist(Line2D([m["x0"], m["x0"], m["x1"], m["x1"]], [y - 0.25 * tier, y, y, y - 0.25 * tier],
                                     transform=at, color=ink, linewidth=width, solid_capstyle="butt",
                                     clip_on=False, zorder=6))
                ax.text((m["x0"] + m["x1"]) / 2.0, y + 0.05 * tier, m["text"], transform=at, ha="center",
                        va="bottom", fontsize=small, color=ink, clip_on=False, zorder=6)
        ax.set_title(lay["titles"][index], loc="left", fontsize=pt, color=page["ink"])
        ax.set_xticks(X)
        ax.set_xticklabels(lay["names"][:len(rows)], rotation=90 if lay["upright"] else 0,
                           linespacing=1.1)
        ax.tick_params(axis="x", length=0)
        if grouping:
            from matplotlib.transforms import blended_transform_factory, offset_copy

            if lay["upright"]:
                down = max(_text_width_pt(str(v), small, family) for v in lay["names"]) + 0.8 * pt
            else:
                down = max(str(v).count("\n") + 1 for v in lay["names"]) * small * 1.25 + 0.6 * pt
            under = offset_copy(blended_transform_factory(ax.transData, ax.transAxes), fig=fig, units="points",
                                x=0, y=-down)
            for group in grouping["groups"]:
                g0, g1 = group["x0"] - 0.4, group["x1"] + 0.4
                ax.add_artist(Line2D([g0, g1], [0, 0], transform=under, color=page["grid"],
                                     lw=max(MIN_LINE_PT, line * 0.8), clip_on=False))
                ax.text((g0 + g1) / 2.0, 0, str(group["label"]), transform=offset_copy(
                    under, fig=fig, units="points", x=0, y=-0.25 * pt), ha="center", va="top", fontsize=small,
                    color=page["ink"], clip_on=False)
        if spec.grid:
            ax.grid(axis="y", color=page["grid"], lw=max(MIN_LINE_PT, line * 0.6), zorder=0)
            ax.set_axisbelow(True)
        def day_xs(k: int, count: int) -> list:
            # THE DAYS: identical open circles at fixed places either side of the column's middle, so neither a
            # colour nor a position can be read as an order
            span = min(_DAY_SPAN, _DAY_STEP * (count - 1))
            return [X[k] - 0.06 + (span * (j / (count - 1) - 0.5) if count > 1 else 0.0) for j in range(count)]

        if _pair_lines_drawn(payload, spec):
            # THE PAIR LINES (5 Nov 2026): through the replicates declared as pairs, behind the circles
            from .hmm_view import pair_lines

            for _tag, points in pair_lines(rows):
                ax.plot([day_xs(k, len(rows[k]["days"]))[j] for k, j in points],
                        [rows[k]["days"][j] for k, j in points], "-", color=grey, alpha=0.7,
                        lw=max(MIN_LINE_PT, stroke * 0.8), zorder=2.5)
        for k, row in enumerate(rows):
            colour = row.get("colour") or QC_DATA
            days = list(row.get("days") or [])
            if days:
                xs = day_xs(k, len(days))
                ax.plot(xs, days, "o", ms=day_ms, mfc="none", mec=colour,
                        mew=stroke, zorder=3)
                x = X[k] + _MEAN_AT
                ax.errorbar([x], [row["mean"]], yerr=[[row["sd"]], [row["sd"]]], fmt="none",
                            ecolor=colour, elinewidth=stroke, capsize=cap, capthick=stroke,
                            zorder=2)
                ax.plot([x], [row["mean"]], "o", ms=mean_ms, color=colour, mec=page["bg"],
                        mew=0.4, zorder=4)
            elif row.get("value") is not None:
                if row.get("band") is not None:
                    lo, hi = row["band"]
                    ax.add_patch(Rectangle((X[k] - _BAND_HALF, lo), 2 * _BAND_HALF, hi - lo,
                                           facecolor=_pale(spec, colour, 0.35),
                                           edgecolor="none", zorder=1))
                ax.plot([X[k]], [row["value"]], "o", ms=mean_ms, color=colour, mec=page["bg"],
                        mew=0.4, zorder=4)
            else:
                # NO NUMBER (a one-state fit has no rates): said, never an empty column
                ax.annotate("n/a", (X[k], y_lo), xytext=(0, small * 0.3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=small, color=page["muted"])
    return axes[0]


def _comparison_number(value):
    return None if value is None else float(value)


def _hmm_comparison_source(payload: dict, spec: FigureSpec) -> tuple:
    """One row per condition per panel, as numbers in the panel's unit as drawn: the pooled
    fit, the days' mean and SD, the movie or cell band (each condition's spread line names
    which), and every replicate day's value."""
    panels = payload.get("panels") or []
    widest = max([len(r.get("days") or ()) for p in panels for r in p.get("rows") or ()]
                 + [0])
    one, many = _comparison_unit(payload)
    Many, One = many[0].upper() + many[1:], one[0].upper() + one[1:]
    tagged = any(t for p in panels for r in p.get("rows") or () for t in r.get("day_tags") or ())
    columns = (["Panel", "Unit", "Condition", "Shown as", "States in the fit", "Drawn as",
                "Pooled fit", f"Mean over {many}", f"SD over {many} (spread)", Many,
                "Band low (95 %, movies or cells)", "Band high (95 %, movies or cells)",
                f"{Many}, as named"] + (["Pair tags, in value order"] if tagged else [])
               + [f"{One} value {k + 1}" for k in range(widest)])
    rows = []
    for panel in panels:
        for row in panel.get("rows") or ():
            days = [float(v) for v in row.get("days") or ()]
            band = row.get("band")
            rows.append([panel["title"], panel["unit"], row["key"], spec.shown_name(row["name"]),
                         row.get("n_states"), _comparison_drawn(row),
                         _comparison_number(row.get("value")),
                         _comparison_number(row.get("mean")), _comparison_number(row.get("sd")),
                         len(days) or None,
                         None if band is None else float(band[0]),
                         None if band is None else float(band[1]),
                         "; ".join(row.get("day_labels") or ()) or None]
                        + (["; ".join(t or "none" for t in row.get("day_tags") or ()) or None] if tagged else [])
                        + days + [None] * (widest - len(days)))
    return (columns, rows), ()


def _comparison_condition_line(entry: dict) -> str:
    n = entry.get("n_states")
    text = f"{entry['name']}: {n} state{'' if n == 1 else 's'}"
    text += "; " + (str(entry["spread"]).rstrip(".") if entry.get("spread")
                    else "no spread computed (uncertainty None)")
    if entry.get("sigma"):
        text += f"; σ correction {entry['sigma']}"
    if entry.get("n_tracks") is not None:
        text += f"; {int(entry['n_tracks']):,} tracks, {int(entry.get('n_steps') or 0):,} steps"
    if entry.get("tau_s"):
        text += f", τ = {1000 * float(entry['tau_s']):.1f} ms"
    for day, why in entry.get("failed") or ():
        text += f"; {day}: no fit ({why})"
    return text + "."


def _hmm_comparison_notes(payload: dict, tested: bool = False) -> tuple:
    dwell_rows = [r for p in payload.get("panels") or () if p.get("key") == "dwell" for r in p.get("rows") or ()]
    notes = ["One row per condition per panel, in the unit drawn: occupancy in per "
             "cent (0-100), rates per second, D in µm²/s" + (", the dwell in ms." if dwell_rows else "."),
             "The slow state is the state with the lowest apparent D in each condition's own "
             "fit. Rate out: the sum of its rates to every other state. Rate in: Σ s_i "
             "k(i→slow) / Σ s_i over the other states i, weighted by the stationary mix s; "
             "for two states it is the fitted rate.",
             "Pooled fit: one fit of all the condition's tracks. Mean and SD over replicates: taken over "
             "each replicate's own fit. The SD (n − 1) is the spread between replicates, drawn as "
             "the ±1 SD bar; it is not a confidence interval.",
             "Band: the 2.5th and 97.5th percentiles over cell resamples with the whole model "
             "refitted, for a single replicate only (with two or more replicates, cells are not "
             "independent).",
             "Replicate values: each condition's own replicates, named in the column before them. "
             "Replicate k of one condition and replicate k of another are unrelated: every replicate is an "
             "independent measurement and replicates are not paired across conditions by number.",
             "The brackets' statistics are on the Comparisons sheet; nothing else between conditions is "
             "computed." if tested else "No statistic between conditions is computed, here or on the figure."]
    if dwell_rows:
        notes.append("Dwell: the model-derived mean dwell in the slow state, 1 / its rate out, in ms; per replicate, "
                     "1 / that replicate's rate out. Its band (a single replicate) is the 95 % band of the rate out, "
                     "inverted, and is drawn only where it has an upper bound.")
        for row in dwell_rows:
            if row.get("never_left"):
                notes.append(f"Dwell, {row['name']}: {', '.join(row['never_left'])} never left the slow state, so "
                             "it has no dwell and is left out of the mean.")
    if any(t for p in payload.get("panels") or () for r in p.get("rows") or () for t in r.get("day_tags") or ()):
        notes.append("Pair tags: the tag given to each replicate on the Groups board (Pair replicates). "
                     "Replicates of different conditions with the same tag form a pair; the pair lines join them "
                     "and the paired test uses them. 'none': no tag.")
    if _comparison_unit(payload)[0] == "chunk":
        notes.append("Under Uncertainty Chunks the replicates in these notes are random chunks of each condition's "
                     "movies or cells. They are pseudo-replicates within one culture and show how precisely these "
                     "cells were measured, not where another culture would come out.")
    notes += [_comparison_condition_line(c) for c in payload.get("conditions") or ()]
    if payload.get("caption"):
        notes.append("On screen: " + str(payload["caption"]))
    return tuple(notes)


def _t_over_sd(n: int) -> float:
    """How many SDs the 95 % t interval of a mean of ``n`` values reaches either side."""
    from scipy.stats import t

    return float(t.ppf(0.975, n - 1) / math.sqrt(n))


def hmm_comparison_legend(payload: dict, spec: FigureSpec | None = None, tested: bool = False) -> str:
    """A drafted legend for the HMM comparison across conditions: what the slow state and
    the four numbers are, what each glyph is -- the bar is SPREAD, not a confidence interval
    -- and that no statistic between conditions is drawn."""
    spec = spec or FigureSpec()
    rows = [r for p in payload.get("panels") or [] for r in p.get("rows") or []]
    names = ", ".join(spec.shown_name(r["name"]) for r in _comparison_rows(payload))
    conditions = payload.get("conditions") or []
    counts = {}
    for c in conditions:
        counts.setdefault(int(c["n_states"]), []).append(spec.shown_name(c["name"]))
    parts = [f"Diffusive-state hidden Markov model across conditions ({names}), showing the slow "
             "state (the lowest apparent D in each condition's own fit). Panels: its occupancy (time-average "
             "share of observed steps, %), the rate into it and out of it (per second), and its apparent "
             "diffusion coefficient (µm²/s). Conditions are coloured and named under their columns."]
    dwell_rows = [r for p in payload.get("panels") or [] if p.get("key") == "dwell" for r in p.get("rows") or []]
    if dwell_rows:
        parts.append("Dwell: the model-derived mean dwell in the slow state, 1 / its rate out (ms); it is usually "
                     "longer than the tracks, so it is an estimate. Per replicate it is 1 / that replicate's rate "
                     "out; a replicate that never left the state has no dwell and is left out of the mean.")
    if any(n >= 3 for n in counts):
        parts.append("With three states the rate out is the sum of the slow state's exit rates, and the rate in "
                     "is the other states' rates into it weighted by the fitted stationary mix. With two states, "
                     "both are the fitted rates.")
    if len(counts) > 1:
        parts.append("State counts differ (" + "; ".join(
            f"{n} state{'' if n == 1 else 's'}: {', '.join(found)}"
            for n, found in sorted(counts.items()))
            + "). Each column shows the slow state of that condition's own fit.")
    day_rows = [r for r in rows if r.get("days")]
    if day_rows and _comparison_unit(payload)[0] == "chunk":
        sentence = ("Open circles: random chunks of each condition's movies or cells (pseudo-replicates within one "
                    "culture), each fitted on its own and spread sideways in no particular order. Filled circles: "
                    "their mean. Bars: ±1 SD across chunks, the precision within this culture, not where another "
                    "culture would fall")
        parts.append(sentence + ".")
    elif day_rows:
        sentence = ("Open circles: replicates, each fitted on its own and spread sideways "
                    "in no particular order. Filled circles: their mean. Bars: ±1 SD across the "
                    "replicates, the spread between replicates and not a confidence interval")
        sizes = {len(r["days"]) for r in day_rows}
        if len(sizes) == 1:
            k = sizes.pop()
            sentence += (f" (with {k} replicates the 95 % t interval is {_t_over_sd(k):.2f} times "
                         "the SD; bars that do not overlap are not a significance claim)")
        parts.append(sentence + ".")
    if any(not r.get("days") and r.get("value") is not None and r.get("band") is not None
           for r in rows):
        parts.append("A filled circle on a pale band: the pooled fit and its 95 % percentile "
                     "interval over resamples of a single replicate's cells (counts in the "
                     "source data).")
    if any(not r.get("days") and r.get("value") is not None and r.get("band") is None
           for r in rows):
        parts.append("A filled circle alone: the pooled fit, with no spread.")
    if any(not r.get("days") and r.get("value") is None for r in rows):
        parts.append("n/a: no value (a one-state fit has no rates).")
    if _pair_lines_drawn(payload, spec):
        parts.append("Thin grey lines join replicates paired by hand on the Groups board (one culture or "
                     "imaging session serving both conditions); they are shown for reading and are not tested.")
        parts.append("Every axis starts at zero. Apart from the declared pairs, replicates are independent "
                     "measurements and are not paired across conditions by number"
                     + ("." if tested else ", and no statistic between conditions is computed."))
    elif tested:
        parts.append("Every axis starts at zero. Replicates are independent measurements and are not paired "
                     "across conditions by number.")
    else:
        parts.append("Every axis starts at zero. Replicates are independent measurements and are not paired "
                     "across conditions, and no statistic between conditions is computed.")
    corrected = [spec.shown_name(c["name"]) for c in conditions
                 if c.get("sigma") and not str(c["sigma"]).startswith("off")]
    if corrected:
        parts.append(f"D is corrected for localisation error for {', '.join(corrected)} "
                     "(σ correction, a sensitivity check); otherwise D is apparent.")
    parts.append("An apparent D for the slow state sits at the localisation floor, so it is an upper bound and "
                 "not a physical coefficient.")
    return " ".join(parts)


# -- the jump distances ------------------------------------------------------------ #
#: v2's line styles for 1, 2 and 3 populations (`plotCDF`: ':', '-', '--'), the screen's.
_JUMP_STYLES = {1: ":", 2: "-", 3: "--", 4: "-."}
_JUMP_STYLE_WORDS = {1: "dotted", 2: "solid", 3: "dashed", 4: "dash-dot"}
#: What `FigureSpec.chosen_fit` (and a card's Show) held for "every fit" before 30 Oct 2026 (`jumps_view.LEGACY_ALL`):
#: still read, as every fit made; nothing offers it any more. And the overview's Fit row's "as each card shows it"
#: (a figure of every condition whose cards differ).
FIT_ALL = "all"
FIT_AS_CARDS = "cards"
#: The figures whose Fit row TOGGLES fits on and off, as the cards' Show does (30 Oct 2026; `fit_pressed`): the jump
#: figures, and since 3 Nov 2026 the apparent-D distributions (a card's and every condition's). The others draw one
#: fit.
FIT_SET_KINDS = ("jumps_condition", "jumps_overview", "appd_histogram", "appd_overview")
#: The figures with a Y unit row (probability, counts, density per bin): the jump figures alone, whose bins are 10 nm
#: (`y_mode_choices`); an apparent-D histogram's log bins stay counts of tracks.
Y_MODE_KINDS = ("jumps_condition", "jumps_overview")


# -- the HMM section's Jump angles page (22 Nov 2026, `hmm_angles_view.export_payload`) ------------------------- #
#: The strip's height and each condition's row of histograms, mm.
_ANGLES_STRIP_MM = 46.0
_ANGLES_ROW_MM = 24.0


def _angles_chance(payload: dict) -> bool:
    """Whether the figure reads against chance (the page's Against chance), else the data alone."""
    return (payload.get("mode") or (payload.get("strip") or {}).get("mode") or "chance") == "chance"


def _angles_key(payload: dict) -> tuple:
    """The key's entries: the strip's, then the half circles'; chance's only against chance."""
    if _angles_chance(payload):
        return (("replicate", "mean ± SD", "free molecules' range"), ("observed", "chance"))
    return (("replicate", "mean ± SD"), ("observed",))


def _hmm_angles_key_lines(spec: FigureSpec, payload: dict | None = None) -> int:
    """The key's lines: one where its entries fit the width, else the strip's and the half circles' apart."""
    small = spec.small_pt()
    need_pt = sum(len(label) * 0.55 * small + 2.6 * small for group in _angles_key(payload or {}) for label in group)
    return 1 if need_pt * 0.3528 <= spec.width_value() * 0.94 else 2


def _hmm_angles_key_mm(spec: FigureSpec, payload: dict | None = None) -> float:
    return (2.0 + 4.2 * _hmm_angles_key_lines(spec, payload)) if spec.legend else 0.0


def _hmm_angles_height(payload: dict, spec: FigureSpec) -> float:
    """The key, the strip, then one row of half circles a condition."""
    return (6.0 + _hmm_angles_key_mm(spec, payload) + _ANGLES_STRIP_MM
            + _ANGLES_ROW_MM * max(1, len(payload.get("histograms") or ())))


def _draw_hmm_angles(fig, payload: dict, spec: FigureSpec, family: str):
    """The Jump angles page in print, from what it shows and nothing recomputed, in its reading (`_angles_chance`): on
    top turning back across conditions, one panel a state (each replicate an open circle, the mean filled with +-1 SD,
    the reference line at 1 for the data alone, at 0 against chance with the band free molecules read within shaded);
    under it one row a condition, each state's angles as a half circle (0 degrees, forward, on the left; 180, back, on
    the right), chance a dashed outline against chance only. Returns the first panel's axes."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    strip = payload.get("strip") or {}
    panels = strip.get("panels") or []
    hists = payload.get("histograms") or []
    if not panels or not hists:
        raise FigureExportError("no jump angles to draw yet: run the HMM and open its Jump angles page")
    against = _angles_chance(payload)
    page = _page(spec)
    pt, small, line = spec.pt(), spec.small_pt(), spec.line_pt()
    ink, grey = page["ink"], page["muted"]
    stroke = max(0.6, line)
    states = [p["name"] for p in panels]
    # THE KEY IN A ROW OF ITS OWN over everything (drawn over each panel's title, it covered "Slow state")
    key_mm = _hmm_angles_key_mm(spec, payload)
    ratios = ((key_mm,) if key_mm else ()) + (_ANGLES_STRIP_MM, _ANGLES_ROW_MM * len(hists))
    outer = fig.add_gridspec(len(ratios), 1, height_ratios=ratios)
    if key_mm:
        holder = fig.add_subplot(outer[0])
        holder.set_axis_off()
        handles = {"replicate": Line2D([], [], marker="o", ls="none", ms=0.62 * pt, mfc="none", mec=grey, mew=stroke),
                   "mean ± SD": Line2D([], [], marker="o", ls="none", ms=0.8 * pt, color=grey),
                   "free molecules' range": Patch(facecolor=page["grid"], edgecolor="none"),
                   "observed": Patch(facecolor=_pale(spec, grey, 0.6), edgecolor="none"),
                   "chance": Line2D([], [], color=ink, lw=stroke, ls=(0, (3, 2)))}
        labels = [label for group in _angles_key(payload) for label in group]
        holder.legend([handles[k] for k in labels], labels, loc="center", frameon=False, fontsize=small,
                      ncol=len(labels) if _hmm_angles_key_lines(spec, payload) == 1 else 3, handlelength=1.6,
                      columnspacing=1.2, borderaxespad=0.0)
        strip_cell, rows_cell = outer[1], outer[2]
    else:
        strip_cell, rows_cell = outer[0], outer[1]
    top = strip_cell.subgridspec(1, len(states))
    lo, hi = float(strip["lo"]), float(strip["hi"])
    reference = float(strip.get("reference", 0.0 if against else 1.0))
    band = strip.get("resolution")
    first = None
    for i, panel in enumerate(panels):
        ax = fig.add_subplot(top[0, i])
        first = first or ax
        columns = panel["columns"]
        if band is not None:
            ax.axhspan(-float(band), float(band), color=page["grid"], lw=0, zorder=0)
        ax.axhline(reference, color=grey, lw=stroke, zorder=1)
        for j, column in enumerate(columns):
            colour = column["colour"]
            reps = column["replicates"]
            for r, v in enumerate(reps):
                ax.plot(j - 0.15 + (r - (len(reps) - 1) / 2) * 0.07, v, "o", ms=0.62 * pt, mfc="none", mec=colour,
                        mew=stroke, zorder=3)
            if math.isfinite(column["mean"]):
                if math.isfinite(column["sd"]):
                    ax.errorbar(j + 0.15, column["mean"], yerr=column["sd"], fmt="none", ecolor=ink, elinewidth=stroke,
                                capsize=0.35 * pt, capthick=stroke, zorder=2)
                ax.plot(j + 0.15, column["mean"], "o", ms=0.8 * pt, color=colour, mec=page["bg"], mew=0.4, zorder=4)
        ax.set_xlim(-0.6, len(columns) - 0.4)
        ax.set_ylim(lo, hi)
        ax.set_xticks(range(len(columns)))
        ax.set_xticklabels([spec.shown_name(c["label"]) for c in columns])
        ax.set_title(f"{str(panel['name']).capitalize()} state", loc="left")
        if i == 0:
            ax.set_ylabel(strip.get("y_label") or ("Turning back beyond chance" if against else "Turning back"))
        if spec.grid:
            ax.yaxis.grid(True, color=page["grid"], lw=0.4)
            ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    bottom = rows_cell.subgridspec(len(hists), len(states))
    for r, h in enumerate(hists):
        by = {s["name"]: s for s in h["states"]}
        for k, name in enumerate(states):
            state = by.get(name)
            if state is None:
                fig.add_subplot(bottom[r, k]).set_axis_off()
                continue
            # A HALF CIRCLE (asked 22 Nov 2026: "an angular map instead of bars ... something like a half circle", then
            # "0° left and 180° right? back and forward?"): 0 degrees, forward, on the left, 180, back, on the right
            # (zero in the west, clockwise); each bin a wedge as long as its density; chance a dashed outline against
            # chance only; the radius carries no number a reader uses
            ax = fig.add_subplot(bottom[r, k], projection="polar")
            ax.set_theta_zero_location("W")
            ax.set_theta_direction(-1)
            ax.set_thetamin(0)
            ax.set_thetamax(180)
            edges = [math.radians(float(e)) for e in state["edges"]]
            observed = [float(v) if v is not None and math.isfinite(float(v)) else 0.0 for v in state["observed"]]
            ax.bar(edges[:-1], observed, width=[b - a for a, b in zip(edges[:-1], edges[1:])], align="edge",
                   color=_pale(spec, h["colour"], 0.6), lw=0, zorder=1)
            chance = []
            if against and state.get("chance") is not None:
                chance = [float(v) if v is not None and math.isfinite(float(v)) else 0.0 for v in state["chance"]]
                arc_t, arc_r = [], []
                for i, c in enumerate(chance):
                    span = [edges[i] + (edges[i + 1] - edges[i]) * f / 8.0 for f in range(9)]
                    arc_t += span
                    arc_r += [c] * len(span)
                ax.plot(arc_t, arc_r, color=ink, lw=stroke, ls=(0, (3, 2)), zorder=2)
            ax.set_ylim(0, max(observed + chance + [1e-9]) * 1.08)
            ax.set_xticks([0.0, math.pi / 2, math.pi])
            ax.set_xticklabels(["0°", "90°", "180°"])
            ax.xaxis.set_tick_params(pad=1.0)
            ax.set_yticks([])
            ax.grid(True, color=page["grid"], lw=0.4)
            ax.spines["polar"].set_color(page["grid"])
            ax.set_title(f"{spec.shown_name(h['label'])} · {name}", loc="left", fontsize=small)
    return first


def _hmm_angles_source(payload: dict, spec: FigureSpec) -> tuple:
    """One row a condition and state (the replicates' mean and SD of the number shown, the summary line, each
    replicate's value); the Welch tables and every half circle's bins as further sheets."""
    against = _angles_chance(payload)
    shown = "Beyond chance" if against else "Turning back"
    strip = payload.get("strip") or {}
    widest = max([len(c["replicates"]) for p in strip.get("panels") or () for c in p["columns"]] + [0])
    lines = payload.get("lines") or {}
    columns = (["Condition", "Shown as", "State", "Replicates with a value", f"{shown}, mean over replicates",
                f"{shown}, SD over replicates (spread)", "Summary"]
               + [f"Replicate value {k + 1}" for k in range(widest)])
    rows = []
    for panel in strip.get("panels") or ():
        for column in panel["columns"]:
            summary = next((ln for ln in lines.get(column["key"], ()) if ln.startswith(f"{panel['name']}:")), "")
            reps = [float(v) for v in column["replicates"]]
            rows.append([column["key"], spec.shown_name(column["label"]), panel["name"], len(reps),
                         _comparison_number(column["mean"]), _comparison_number(column["sd"]), summary]
                        + reps + [None] * (widest - len(reps)))
    sheets = []
    tables = payload.get("tables") or []
    if tables:
        t_rows = [[t["name"], *row] for t in tables for row in t["rows"]]
        sheets.append(("Comparisons", f"{shown} between conditions",
                       tuple(t["caption"] for t in tables), ["State", *tables[0]["columns"]], t_rows))
    h_rows = []
    for h in payload.get("histograms") or ():
        for state in h["states"]:
            edges = state["edges"]
            chance = state.get("chance") if against else None
            for i in range(len(edges) - 1):
                h_rows.append([h["key"], spec.shown_name(h["label"]), state["name"], float(edges[i]),
                               float(edges[i + 1]), state["observed"][i]]
                              + ([chance[i]] if chance is not None else []))
    sheets.append(("Angles", "Angle densities" + (", observed and chance at the same places" if against else ""),
                   ("Density per degree: each bin's weighted share of the pairs divided by its width.",),
                   ["Condition", "Shown as", "State", "From (°)", "To (°)", "Observed"]
                   + (["Chance"] if against else []), h_rows))
    return (columns, rows), tuple(sheets)


def _hmm_angles_notes(payload: dict) -> tuple:
    notes = ["Turning back: the share of pairs of steps within 30° of 180° over the share within 30° of 0° (1: no "
             "preference)."]
    if _angles_chance(payload):
        notes.append("Beyond chance: that number minus the same for free molecules started at the same places, in the "
                     "same cells, with the state's step size, each spot's localisation error and the camera's blur. "
                     "Free molecules simulated over whole tracks in a capsule read within ±0.05 through the method alone.")
    else:
        notes.append("The data alone, no simulation: a free molecule also turns back near the cell wall and where its "
                     "steps are about as small as the localisation error, and goes forward more when fast.")
    notes += ["Each pair counts for an HMM state with the probability that both of its steps were in that state; pairs "
              "across a switch are left out.",
              "Replicate values: one per replicate, the replicate's pairs only. The SD (n - 1) is the spread between "
              "replicates; it is not a confidence interval. Replicates are independent and never paired across "
              "conditions."]
    for key, lines in (payload.get("notes") or {}).items():
        notes += [f"{key}: {ln}" for ln in lines]
    return tuple(notes)


def hmm_angles_legend(payload: dict, spec: FigureSpec | None = None) -> str:
    """A drafted legend for the Jump angles figure, in its reading."""
    spec = spec or FigureSpec()
    against = _angles_chance(payload)
    entries = payload.get("entries") or []
    names = ", ".join(spec.shown_name(e["label"]) for e in entries)
    states = [p["name"] for p in (payload.get("strip") or {}).get("panels") or ()]
    parts = [f"Jump angles per diffusive state of the hidden Markov model ({names}). The angle is taken between two "
             "consecutive steps of a track at the localisation they share (0°, forward; 180°, turned back); each pair "
             "is weighted by the probability that both steps were in the state, and pairs across a switch are left "
             "out."]
    if against:
        parts += ["Top: turning back (pairs within 30° of 180° over pairs within 30° of 0°) beyond chance, per state "
                  f"({', '.join(states)}); open circles, replicates; filled circle and bar, mean ± SD (a spread, not a "
                  "confidence interval); shaded, the ±0.05 range in which free molecules read through the method alone.",
                  "Chance: free molecules started at the same places in the same cells (straight capsules of each "
                  "cell's length, as wide as the space the fastest state's molecules fill, in 3D), at the D that "
                  "reproduces the state's observed step size, with each localisation's error and blur over the "
                  "camera's exposure, linked through the tracker's linking radius as the tracked pairs were (a pair "
                  "kept only where both steps are shorter). A fast state far past the radius can still read high "
                  "beyond chance: u-track's search radius adapts to each track.",
                  "Below: each condition's angles per state as a half circle (forward on the left, back on the right; "
                  "wedges, the observed density per 10° bin) with chance at the same places (dashed)."]
    else:
        parts += ["Top: turning back (pairs within 30° of 180° over pairs within 30° of 0°; 1, no preference), per "
                  f"state ({', '.join(states)}); open circles, replicates; filled circle and bar, mean ± SD (spread, "
                  "not a confidence interval). No simulation is subtracted: a free molecule also turns back near the "
                  "cell wall and where its steps are as small as the localisation error.",
                  "Below: each condition's angles per state as a half circle (forward on the left, back on the right; "
                  "wedges, the observed density per 10° bin)."]
    unlinked = [spec.shown_name(name) for name in payload.get("unlinked") or ()]
    if against and unlinked:
        parts.append(f"For {', '.join(unlinked)} no linking radius was recorded: every simulated step was linked, so a "
                     "fast state reads too high beyond chance.")
    if payload.get("tables"):
        parts.append("Between conditions: Welch's t-test on the replicates' values (independent samples, never "
                     "paired), Holm-adjusted over the pairs of each state (Comparisons sheet).")
    return " ".join(parts)


def _fit_value_ok(value) -> bool:
    """A `FigureSpec.chosen_fit` that can be honoured: None, a count 1-4, the old "all", or a non-empty set of
    distinct counts 1-4."""
    def count(k):
        return isinstance(k, int) and not isinstance(k, bool) and k in (1, 2, 3, 4)

    if value is None or (isinstance(value, str) and value == FIT_ALL):
        return True
    if isinstance(value, tuple):
        return bool(value) and len(set(value)) == len(value) and all(count(k) for k in value)
    return count(value)


def _one_fit(value, refusal: str):
    """A one-fit figure's `FigureSpec.chosen_fit` as a count, or None (follow the card): a set of one count is that
    count; "all" or several are refused with ``refusal``."""
    if value is None:
        return None
    if (isinstance(value, str) and value == FIT_ALL) or (isinstance(value, tuple) and len(value) != 1):
        raise FigureExportError(refusal)
    return int(value[0]) if isinstance(value, tuple) else int(value)


def _fit_colour(k: int, spec: "FigureSpec") -> str:
    """Fit ``k``'s colour when several are drawn (`jumps_view.fit_colour`: no condition palette's), lighter on a dark
    page."""
    from .jumps_view import fit_colour

    return fit_colour(k, dark=spec.dark)


def _shown_words(k) -> str:
    """``"2 populations"`` or ``"1 and 3 populations"`` (`jumps_view.show_words`)."""
    from .jumps_view import show_words

    return show_words(k)


def _plural(k: int) -> str:
    return f"{k} population{'' if k == 1 else 's'}"


def _jump_fits(result: dict) -> dict:
    """The fits made, keyed by their population count (an int, however it travelled)."""
    return {int(k): v for k, v in result["fits"].items()}


def _jd_fits(entry: dict) -> dict:
    """A Signed jumps entry's fits on its axis, keyed by population count, the failed ones left out."""
    axis = ((entry["result"].get("axes") or {}).get(entry["axis"])) or {}
    return {int(k): v for k, v in (axis.get("fits") or {}).items() if "failed" not in v}


def _jd_chosen(entry: dict, spec: FigureSpec) -> tuple:
    """``(population count, how it was chosen)`` for a Signed jumps card: the Export dialog's, else the card's."""
    fits = _jd_fits(entry)
    if not fits:
        raise FigureExportError(f"{entry.get('name', 'this condition')} has no fit on this axis")
    asked = _one_fit(spec.chosen_fit, "a Signed jumps figure draws one fit; choose a population count")
    if asked is not None:
        if asked not in fits:
            raise FigureExportError(
                f"{entry.get('name', 'this condition')} has no {_plural(asked)} fit "
                f"(made: {', '.join(str(k) for k in sorted(fits))})")
        return asked, "in the Export dialog"
    k = entry.get("selected")
    if k is not None and int(k) in fits:
        return int(k), "on the card"
    return min(fits), "the fewest populations fitted"


def fit_choices(kind: str, payload) -> tuple:
    """``((population count, its words), ...)``: the fits a jump-distance figure can be
    drawn around -- the ones that were made -- or ``()`` for any other kind (the Export
    dialog offers its Fit row only then). On the jump figures (`FIT_SET_KINDS`) each is a
    toggle (`fit_pressed`), the overview's first "Cards"; on the others one choice."""
    if kind not in ("jumps_condition", "jumps_overview", "appd_histogram", "appd_overview", "jd_condition"):
        return ()
    try:
        if kind in ("jumps_overview", "appd_overview"):
            # first "as the cards show them", then every count a condition was fitted at (no All: every one on is)
            fitted = _jump_fits if kind == "jumps_overview" else _appd_fits
            made = sorted({k for e in payload["entries"] for k in fitted(e["result"])})
            if not made:
                return ()
            return ((FIT_AS_CARDS, "Cards"),) + tuple((k, str(k)) for k in made)
        result = payload["entries"][0]["result"]
        if kind == "jd_condition":
            made = sorted(_jd_fits(payload["entries"][0]))
        else:
            made = sorted(_jump_fits(result) if kind == "jumps_condition" else _appd_fits(result))
    except (KeyError, IndexError, TypeError, AttributeError):
        return ()
    if kind in ("jumps_condition", "appd_histogram"):
        # short words: up to four toggles (1-4) in the dialog's row
        return tuple((k, str(k)) for k in made)
    return tuple((k, _plural(k)) for k in made)


def _chosen(entry: dict, spec: FigureSpec) -> tuple:
    """``(the counts drawn -- a tuple, ascending --, how they were chosen)``: the spec's when set (the Export
    dialog's Fit row), else the counts the card's Show has on, else BIC's preference, else the fewest made. A count
    asked for that was not fitted is refused by name. A value stored before 30 Oct 2026 reads as
    `jumps_view.show_counts` reads it: a count ``n`` is ``(n,)``, "all" every fit made."""
    from .jumps_view import as_shown, show_counts

    fits = _jump_fits(entry["result"])
    name = entry.get("name", "this condition")
    if not fits:
        raise FigureExportError(f"{name} has no fit to draw")
    if spec.chosen_fit is not None:
        asked = show_counts(spec.chosen_fit, fits)
        missing = [k for k in asked if k not in fits]
        if missing:
            raise FigureExportError(
                f"{name} has no {_plural(missing[0])} fit (made: {', '.join(str(k) for k in sorted(fits))})")
        return asked, "in the Export dialog"
    shown = as_shown(entry.get("selected"), fits)
    if shown:
        return shown, "on the card"
    k = entry["result"].get("bic_best")
    if k is not None and int(k) in fits:
        return (int(k),), "by BIC"
    return (min(fits),), "the fewest populations fitted"


def shown_fit(kind: str, payload, spec: FigureSpec):
    """What ``spec`` draws for this payload (the dialog's Fit row is lit on it), or None: on a jump figure the
    counts drawn (a tuple) -- the overview's "Cards" (`FIT_AS_CARDS`) while its cards differ --, on a one-fit
    figure the count."""
    try:
        if not fit_choices(kind, payload):
            return None
        if kind in ("jumps_overview", "appd_overview"):
            # the dialog's choice, else the set the cards agree on, else "as the cards show them"
            fitted, chosen = (_jump_fits, _chosen) if kind == "jumps_overview" else (_appd_fits, _appd_shown)
            if spec.chosen_fit is not None:
                from .jumps_view import show_counts

                made = {k: None for e in payload["entries"] for k in fitted(e["result"])}
                return show_counts(spec.chosen_fit, made)
            shown = {chosen(e, spec)[0] for e in payload["entries"]}
            return next(iter(shown)) if len(shown) == 1 else FIT_AS_CARDS
        entry = payload["entries"][0]
        if kind == "jd_condition":
            return _jd_chosen(entry, spec)[0]
        return (_chosen(entry, spec) if kind == "jumps_condition" else _appd_shown(entry, spec))[0]
    except FigureExportError:
        return None


def fit_pressed(kind: str, payload, spec: FigureSpec, key):
    """The `FigureSpec.chosen_fit` after a click on the Fit row's pill ``key``. On a jump figure (`FIT_SET_KINDS`) a
    count TOGGLES, starting from what is drawn (`jumps_view.toggle_show`: the last one on stays on), and the
    overview's "Cards" goes back to each card's own (None); on any other figure the pill is its one fit."""
    from .jumps_view import toggle_show

    if kind not in FIT_SET_KINDS:
        return key
    if key == FIT_AS_CARDS:
        return None
    shown = shown_fit(kind, payload, spec)
    made = {k: None for k, _w in fit_choices(kind, payload) if k != FIT_AS_CARDS}
    return toggle_show(shown if isinstance(shown, tuple) else (), key, made)


#: Wider than this (width / height), the residuals stand BESIDE the histogram, not under it.
_JUMPS_WIDE = 1.3


#: The stacked plots' shares of a one-column jump figure (`_draw_jumps`), and of the right-hand column beside the
#: histogram on a wide one. A residual plot turned off gives its share back (`_jumps_stacked`).
_JUMPS_SHARES = {"histogram": 3.0, "pearson": 1.15, "ecdf": 1.35}
_JUMPS_BESIDE = {"pearson": 1.0, "ecdf": 1.15}
#: What is not plot in a stacked jump figure (the key above, the axis title below), in lines of the body text: kept
#: whole when a residual plot is turned off.
_JUMPS_FIXED_LINES = 4.0


#: The jump figures' LINE WIDTHS in pt, from the preset's line (asked 30 Oct 2026: "i think we can make the lines a
#: bit bigger and better visible"). At Nature's and Cell's 0.75 pt: a fit 1.5 (was 1.05, line x 1.4), one fit's
#: components 0.9 (was 0.75), several fits' components 0.68 and dashed (new), the Pearson strip's lines 1.05 (was
#: 0.75); the ECDF residual is drawn as a fit. On slides (1.5 pt) 3.0, 1.8, 1.35 and 2.1. Each has a floor, never
#: under `MIN_LINE_PT`.
def _jump_widths(spec: FigureSpec) -> dict:
    line = spec.line_pt()
    return {"fit": max(1.1, line * 2.0), "component": max(0.8, line * 1.2),
            "component_many": max(0.6, line * 0.9), "residual": max(0.8, line * 1.4)}


#: Several fits' components: dashed (dash and gap in line widths), in each fit's colour (`jumps_view.components_drawn`).
_COMPONENT_DASH = (0, (3.0, 2.0))


def _y_mode_of(entry: dict, spec: FigureSpec) -> str:
    """A panel's y unit: the Export dialog's when set (`FigureSpec.y_mode`), else the card's (its entry's
    ``"y_mode"``; the standard, Probability, for an entry made before the choice)."""
    from .jumps_view import as_y_mode

    return as_y_mode(spec.y_mode if spec.y_mode is not None else entry.get("y_mode"))


def _overview_y_mode(entries, spec: FigureSpec) -> str:
    """Export all's one y unit: the dialog's, else the cards' when they agree, else the standard."""
    from .jumps_view import DEFAULT_Y_MODE, as_y_mode

    if spec.y_mode is not None:
        return as_y_mode(spec.y_mode)
    modes = {as_y_mode(e.get("y_mode")) for e in entries}
    return next(iter(modes)) if len(modes) == 1 else DEFAULT_Y_MODE


def _unit_words(mode: str, edges) -> str:
    """The y unit in a column header: "probability per 10.4 nm bin", "jumps", "per nm"."""
    from .jumps_view import bin_width

    return {"probability": f"probability per {bin_width(edges):.3g} nm bin", "counts": "jumps",
            "density": "probability density, 1/nm"}[mode]


def _components_on(entry: dict, spec: FigureSpec) -> bool:
    """Whether this panel draws components: the Export dialog's box when it was set (`FigureSpec.components`), else
    the card's (its entry's ``"components"``; on for an entry made before 30 Oct 2026)."""
    return bool(entry.get("components", True)) if spec.components is None else bool(spec.components)


#: WHETHER A JUMP FIGURE DRAWS ITS RESIDUAL PLOTS when the Export dialog has not said (`FigureSpec.pearson_residuals`
#: / `.ecdf_residuals` None): a card's figure both, as the card shows them and as it always has; Export all none --
#: two strips under every plot of a grid make it tall -- until its boxes are ticked (30 Oct 2026).
RESIDUALS_BY_DEFAULT = {"jumps_condition": True, "jumps_overview": False,
                        # the apparent-D distributions likewise (3 Nov 2026): a card's as its card shows them
                        # (`_appd_strips`), every condition's figure none until ticked
                        "appd_histogram": True, "appd_overview": False}


def residuals_on(spec: FigureSpec, kind: str, which: str) -> bool:
    """Whether jump figure ``kind`` draws its ``which`` ("pearson" | "ecdf") residual plot: the Export dialog's box
    when it was set, else the figure's own default (`RESIDUALS_BY_DEFAULT`)."""
    value = getattr(spec, f"{which}_residuals")
    return RESIDUALS_BY_DEFAULT.get(kind, True) if value is None else bool(value)


def _jumps_strips(spec: FigureSpec, kind: str = "jumps_condition") -> list:
    """The residual plots this jump figure draws, in order under (or beside) the histogram: ``["pearson", "ecdf"]``,
    one of them, or ``[]`` (`residuals_on`)."""
    return [name for name in ("pearson", "ecdf") if residuals_on(spec, kind, name)]


def _jumps_stacked(full: float, spec: FigureSpec, floor: float) -> float:
    """A stacked jump figure's height ``full`` (both residual plots) for the plots ``spec`` keeps: the key and the
    axis title keep their room, the plots shrink by the share turned off, never under ``floor``. Both on:
    ``full``, unchanged."""
    strips = _jumps_strips(spec)
    share = (sum(_JUMPS_SHARES[n] for n in ("histogram", *strips)) / sum(_JUMPS_SHARES.values()))
    if share >= 1.0:
        return full
    fixed = _JUMPS_FIXED_LINES * spec.pt() * _PT_MM
    return min(full, max(floor, fixed + (full - fixed) * share))


def _jumps_height(spec: FigureSpec) -> float:
    """The figure's own height. One column: three stacked plots, about as tall as wide (at
    most 130 mm and the journal's maximum), shorter by the share of a residual plot turned
    off (`_jumps_stacked`; the histogram alone is as tall as any chart). Wider than 120 mm:
    the residuals go beside the histogram (`_JUMPS_WIDE`), half as tall as wide. On a slide
    or poster, side by side at 3:4 up to 200 mm wide (a 150 mm slide's 16:9 left three plots
    and their key 84 mm at 16 pt, and their tick labels ran into each other), the preset's
    own beyond."""
    default = spec.default_height_mm()
    width = spec.width_value()
    if spec.preset in LARGE_PRESETS:
        return round(max(default, width * 0.75), 1) if width <= 200.0 else default
    tallest = spec.preset_values.get("max_height_mm") or 600.0
    if width > 120.0:
        return round(min(tallest, max(default, width * 0.5)), 1)
    full = min(tallest, max(default, min(width * 1.05, 130.0)))
    return round(_jumps_stacked(full, spec, default), 1)


def _jumps_size(spec: FigureSpec) -> tuple:
    """``(width, height)`` in mm as the jump figure is drawn: the spec's height, else its own (`_jumps_height`)."""
    width, height = spec.size_mm()
    return width, (height if spec.height_mm is not None else _jumps_height(spec))


def _jumps_wide(spec: FigureSpec) -> bool:
    """Whether the residual plots stand BESIDE the histogram (`_JUMPS_WIDE`); never without one."""
    width, height = _jumps_size(spec)
    return bool(_jumps_strips(spec)) and width / height >= _JUMPS_WIDE


def _jumps_places(spec: FigureSpec) -> dict:
    """Where each plot of the jump figure sits, in the legend's words: Top / Middle / Bottom stacked, Left / Right
    side by side; ``{}`` for the histogram alone."""
    strips = _jumps_strips(spec)
    if not strips:
        return {}
    if _jumps_wide(spec):
        right = ["Right"] if len(strips) == 1 else ["Right, top", "Right, bottom"]
        return {"histogram": "Left", **dict(zip(strips, right))}
    return dict(zip(["histogram", *strips], ["Top", "Bottom"] if len(strips) == 1 else ["Top", "Middle", "Bottom"]))


def _symmetric_ticks(span: float, floor: float) -> list:
    """``[-t, 0, t]`` with ``t`` a round number inside ``span`` (at least ``floor``)."""
    for step in (500, 200, 100, 50, 20, 10, 5, 2, 1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005,
                 0.002, 0.001):
        if step <= span * 0.9 and step >= floor:
            return [-step, 0.0, step]
    return [-floor, 0.0, floor]


def _draw_jumps(fig, entry: dict, spec: FigureSpec, family: str):
    """The jump distances in print, from the card's own payload
    (`jumps_view.compute_jump_view`), as up to three plots over one jump-distance axis:

    * the HISTOGRAM -- observed jumps per bin in the condition's colour (pale), and the fits drawn (`_chosen`: the
      card's Show, else the dialog's Fit row): ONE fit's EXPECTED count per bin in the ink with its components in
      shades of the condition's colour, or two or more each in its own colour and v2's line style (dotted 1, solid
      2, dashed 3, dash-dot 4), no components;
    * the PEARSON residuals per bin of the one fit, beyond +-2 in the flag colour, over the +-2 band counting noise
      alone would mostly stay inside (two or more fits: each one's, as lines);
    * the ECDF residual (model minus empirical CDF), exact and unbinned, of the fit(s) drawn.

    Asked 26 Oct 2026: "if 1 or 2 or 3 or 4 pop selected -> u only see this fit. right next to it "all" where all
    curves shown with different colors", and "the histogram color can be same color as the protein in the
    bubble". Before, every fit was drawn over grey bars, the chosen one strong. Since 30 Oct 2026 the fits drawn
    are a SET, as on the card, and EACH RESIDUAL PLOT IS OPTIONAL (asked: "also make it optional to export ecdf
    and residuals in export window and in composer"; `FigureSpec.pearson_residuals`, `.ecdf_residuals`): a plot
    turned off is not drawn, the ones left take its room, and the lowest plot carries the x axis.

    Stacked in a column, or -- wider than `_JUMPS_WIDE`, with a residual plot to place -- the
    histogram on the left and the residual plots on the right. The key sits ABOVE the plots
    (inside, it lay on the bars and on the one-population curve, which peaks mid-histogram),
    the fit(s) in one column and the one fit's components in the next, under the condition's
    shown name.

    Returns the histogram's axes: the dialog's axis titles and ranges are its (the x range is
    shared by every plot)."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    from .jumps_view import bin_width, components_drawn, histogram_top, in_colour, in_units, y_label

    result = entry["result"]
    fits = _jump_fits(result)
    if not fits:
        raise FigureExportError(f"{entry.get('name', 'this condition')} has no fit to draw")
    # THE Y UNIT (30 Oct 2026): the bars and every curve divided by one divisor (`jumps_view.in_units`), the Pearson
    # and ECDF residuals as they are
    mode = _y_mode_of(entry, spec)
    observed_in, fits = in_units(mode, result["observed"], fits, result["edges_nm"])
    drawn = list(_chosen(entry, spec)[0])
    every = in_colour(drawn)
    one = drawn[0]
    parts = components_drawn(fits, drawn, _components_on(entry, spec))
    colour = entry.get("colour") or QC_DATA
    page = _page(spec)
    pt, small, line = spec.pt(), spec.small_pt(), spec.line_pt()
    edges, observed = result["edges_nm"], observed_in
    centres = [(a + b) / 2.0 for a, b in zip(edges, edges[1:])]
    lw = _jump_widths(spec)
    strong = lw["fit"]
    components = fits[one]["components"] if parts and not every else []
    n_comp = len(components)
    numbers = ([] if every else [f"D {d:.3g} µm²/s · {100 * f:.1f} %"
                                 for d, f in zip(fits[one]["D"], fits[one]["fractions"])])

    def ink(k):
        return _fit_colour(k, spec) if every else page["ink"]

    def style(k):
        return _JUMP_STYLES.get(k, "-") if every else "-"

    title = spec.shown_name(entry["name"])
    if spec.title and title and title in spec.title:
        # the figure's title already names the condition (its automatic one does): the key
        # says it a second time, and right under the title the two lines met
        title = None
    width_mm, height_mm = spec.size_mm()
    strips = _jumps_strips(spec)
    wide = bool(strips) and width_mm / height_mm >= _JUMPS_WIDE
    handles, labels, ncol = [], [], 1
    if spec.legend:
        # THE KEY: the fits drawn and nothing else, as SMTracker 2's (30 Oct 2026: "1 and 2 and 3 etc populations as
        # it is"; the component curves are said in the legend's words). One fit with Numbers: its populations' shares
        # and D's beside its name.
        fit_h = [Line2D([], [], ls=style(k), color=ink(k), lw=strong) for k in drawn]
        fit_l = [_plural(k) for k in drawn]
        if not every and spec.numbers and len(fits[one]["D"]) > 1:
            fit_l = [f"{_plural(one)}: " + "; ".join(numbers)]
        comp_h, comp_l = [], []
        measure = lambda s: _text_width_pt(s, small, family)               # noqa: E731
        room = width_mm / _PT_MM * 0.94
        pair = (max(measure(t) for t in fit_l) + max((measure(t) for t in comp_l), default=0.0)
                + small * 6.6)
        if comp_h and pair <= room:
            rows = max(len(fit_h), len(comp_h))
            blank = Line2D([], [], color="none", lw=0)
            handles = (fit_h + [blank] * (rows - len(fit_h))
                       + comp_h + [blank] * (rows - len(comp_h)))
            labels = (fit_l + [""] * (rows - len(fit_l)) + comp_l + [""] * (rows - len(comp_l)))
            ncol = 2
        else:
            handles, labels = fit_h + comp_h, fit_l + comp_l
            labels, _fitting = _legend_columns(labels, room, spec, family)
    inside = None
    if handles and entry.get("legend_at") is not None:
        # THE KEY WHERE THE CARD'S WAS PUT (30 Oct 2026: dragged on the screen, "dragging the legend when it blocks
        # the view"): inside the histogram, one column, its nearer edges at the card's (`jumps_view.legend_anchor`);
        # the plots take the room the key's row had. Not put anywhere, the key stays above the plots, as before.
        inside = ([h for h, t in zip(handles, labels) if t], [t for t in labels if t])
        # inside the plot a key is narrower: a long label (a fit's numbers) wraps to under half the figure's width
        unwrapped = [" ".join(t.split()) for t in inside[1]]
        inside = (inside[0], _legend_columns(unwrapped, width_mm / _PT_MM * 0.45, spec, family)[0])
        handles, labels = [], []
    if wide:
        # BESIDE: the histogram the whole height on the left, the residual plots stacked on
        # the right, one x axis for all; the key in a row of its own over both (above the
        # histogram alone it pushed the right column down with it).
        shares = [_JUMPS_BESIDE[n] for n in strips]
        if handles:
            rows = math.ceil(len(labels) / ncol) + 1
            tallest = max(text.count("\n") + 1 for text in labels)
            key = (rows * tallest * small * 1.55 + small) * _PT_MM
            plot = max(20.0, height_mm - (2.4 * pt + 2.0 * small) * _PT_MM - 2.0)
            rest = max(10.0, plot - key)
            grid = fig.add_gridspec(1 + len(strips), 2, width_ratios=(1.3, 1.0),
                                    height_ratios=(key, *(rest * r / sum(shares) for r in shares)))
            holder = fig.add_subplot(grid[0, :])
            holder.set_axis_off()
            holder.legend(handles, labels, loc="center", ncol=ncol, frameon=False,
                          handletextpad=0.4, columnspacing=1.0, borderaxespad=0.0,
                          handlelength=1.8, title=title,
                          title_fontproperties={"size": small, "weight": "bold"})
            first = 1
        else:
            grid = fig.add_gridspec(len(strips), 2, width_ratios=(1.3, 1.0), height_ratios=shares)
            first = 0
        hist = fig.add_subplot(grid[first:, 0])
        axes = {n: fig.add_subplot(grid[first + i, 1], sharex=hist) for i, n in enumerate(strips)}
    else:
        grid = fig.add_gridspec(1 + len(strips), 1,
                                height_ratios=[_JUMPS_SHARES[n] for n in ("histogram", *strips)])
        hist = fig.add_subplot(grid[0])
        axes = {n: fig.add_subplot(grid[1 + i], sharex=hist) for i, n in enumerate(strips)}
        if handles:
            _legend_above(hist, handles, labels, spec, family, title=title, ncol=ncol)
    if inside:
        from .jumps_view import legend_anchor

        loc, anchor = legend_anchor(entry["legend_at"])
        hist.legend(*inside, loc=loc, bbox_to_anchor=anchor, ncol=1, frameon=True, framealpha=0.85,
                    facecolor=page["bg"], edgecolor="none", borderaxespad=0.0, borderpad=0.4, handletextpad=0.4,
                    handlelength=1.8, fontsize=small, title=title,
                    title_fontproperties={"size": small, "weight": "bold"})

    # THE HISTOGRAM
    _jump_bars(hist, edges, observed, spec, colour)
    for c, comp in enumerate(components):
        hist.plot(centres, comp, "-", color=figstyle.day_shade(colour, c, n_comp),
                  lw=lw["component"], zorder=2)
    for k in (parts if every else ()):
        for comp in fits[k]["components"]:
            hist.plot(centres, comp, ls=_COMPONENT_DASH, color=ink(k), lw=lw["component_many"], alpha=0.9,
                      zorder=3)
    for k in drawn:
        hist.plot(centres, fits[k]["expected"], style(k), color=ink(k), lw=strong, zorder=4)
    top = histogram_top(observed, fits, drawn)
    if entry.get("y_end") and spec.y_mode in (None, entry.get("y_mode")):
        top = max(top, float(entry["y_end"]))             # "Same y scale" on the cards: the top they share
    hist.set_ylim(0, top * 1.06)
    hist.set_ylabel(spec.y_title if spec.y_title is not None else y_label(mode, [bin_width(edges)]))
    if wide or not strips:
        hist.set_xlabel(_title(spec, "jumps_condition", "x"))
    else:
        hist.tick_params(labelbottom=False)
    hist.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=mode == "counts"))
    if not spec.legend and spec.numbers and not every:
        hist.annotate(f"{_plural(one)}:\n" + "\n".join(numbers), (0.98, 0.96),
                      xycoords="axes fraction", ha="right", va="top", fontsize=small,
                      color=page["ink"])

    # THE RESIDUAL PLOTS (`_pearson_strip`, `_ecdf_strip`: one drawing for the card's figure and Export all's tiles)
    if axes.get("pearson") is not None:
        _pearson_strip(axes["pearson"], edges, fits, drawn, spec, _pearson_span(fits, drawn))
        axes["pearson"].set_ylabel(chart_text.PEARSON_LABEL.replace(" ", "\n", 1))
    ecdf = axes.get("ecdf")
    if ecdf is not None:
        _ecdf_strip(ecdf, fits, drawn, spec, _ecdf_span(fits, drawn))
        ecdf.set_ylabel(chart_text.ECDF_RESIDUAL_LABEL.replace(" ", "\n", 1))

    # THE LOWEST RESIDUAL PLOT carries the x axis (the histogram, when it stands alone or beside)
    if strips:
        x_title = _title(spec, "jumps_condition", "x")
        if wide and x_title:
            # The right column is narrow: its axis title wraps rather than running off.
            measure = lambda s: _text_width_pt(s, pt, family)                  # noqa: E731
            x_title = "\n".join(chart_text.wrap_text(x_title, measure,
                                                     width_mm / _PT_MM * 0.3, max_lines=2))
        for name in strips[:-1]:
            axes[name].tick_params(labelbottom=False)
        axes[strips[-1]].set_xlabel(x_title)
    # the x range: the bins, and the ECDF residuals' points when that plot is drawn
    x_end = max([edges[-1]] + ([x for k in drawn for x, _v in fits[k]["ecdf_residual"]] if ecdf is not None
                               else []))
    hist.set_xlim(0.0, x_end)
    if spec.grid:
        for ax in (hist, *axes.values()):
            ax.grid(axis="y", color=page["grid"], lw=max(MIN_LINE_PT, line * 0.6), zorder=0)
            ax.set_axisbelow(True)
    fig.align_ylabels(list(axes.values()) if wide else [hist, *axes.values()])
    return hist


def _pearson_span(fits, drawn) -> float:
    """How far a Pearson strip of the fits ``drawn`` reaches either side of 0: `jumps_view.pearson_span`, the rule the
    screen's plots read too (12 Nov 2026)."""
    from .jumps_view import pearson_span

    return pearson_span(fits, drawn)


def _ecdf_span(fits, drawn) -> float:
    """How far an ECDF residual strip of the fits ``drawn`` reaches either side of 0: `jumps_view.ecdf_span`."""
    from .jumps_view import ecdf_span

    return ecdf_span(fits, drawn)


def _pearson_strip(strip, edges, fits, drawn, spec: FigureSpec, span: float) -> None:
    """THE PEARSON RESIDUALS per bin under a jump histogram, ``span`` either side of 0: one fit's as bars, beyond +-2 in
    the flag colour, over the +-2 band counting noise alone would mostly stay inside; two or more fits' each as a line
    in its colour and v2's style. The card's figure and Export all's tiles draw it alike."""
    from matplotlib.ticker import FixedLocator

    from .jumps_view import in_colour

    page, line, lw = _page(spec), spec.line_pt(), _jump_widths(spec)
    centres = [(a + b) / 2.0 for a, b in zip(edges, edges[1:])]
    widths = [b - a for a, b in zip(edges, edges[1:])]
    strip.axhspan(-2.0, 2.0, color=_pale(spec, _QC_GREY, 0.16), lw=0, zorder=0.5)
    strip.axhline(0.0, color=page["grid"], lw=max(MIN_LINE_PT, line * 0.6), zorder=1)
    if in_colour(drawn):
        for k in drawn:
            strip.plot(centres, fits[k]["pearson"], _JUMP_STYLES.get(k, "-"), color=_fit_colour(k, spec),
                       lw=lw["residual"], zorder=2)
    else:
        pearson = fits[drawn[0]]["pearson"]
        strip.bar([a + w * 0.04 for a, w in zip(edges, widths)], pearson,
                  width=[w * 0.92 for w in widths], align="edge", linewidth=0, zorder=2,
                  color=[chart_text.FLAG_COLOR if abs(v) > 2 else _pale(spec, page["ink"], 0.6) for v in pearson])
    strip.set_ylim(-span, span)
    strip.yaxis.set_major_locator(FixedLocator(_symmetric_ticks(span, 2.0)))


def _ecdf_strip(strip, fits, drawn, spec: FigureSpec, span: float) -> None:
    """THE ECDF RESIDUALS (model CDF minus the empirical CDF, exact and unbinned) of the fits ``drawn``, ``span`` either
    side of 0: one fit's in the ink, several each in its colour and v2's style."""
    from matplotlib.ticker import FixedLocator

    from .jumps_view import in_colour

    page, line = _page(spec), spec.line_pt()
    every = in_colour(drawn)
    strip.axhline(0.0, color=page["grid"], lw=max(MIN_LINE_PT, line * 0.6), zorder=1)
    for k in drawn:
        pts = fits[k]["ecdf_residual"]
        strip.plot([x for x, _v in pts], [v for _x, v in pts], _JUMP_STYLES.get(k, "-") if every else "-",
                   color=_fit_colour(k, spec) if every else page["ink"], lw=_jump_widths(spec)["fit"], zorder=4)
    strip.set_ylim(-span, span)
    strip.yaxis.set_major_locator(FixedLocator(_symmetric_ticks(span, 0.001)))


#: The jump histogram's bars, one artist per histogram (`_jump_bars`): found by this id.
JUMP_BARS_GID = "jump_bars"


def _jump_bars(ax, edges, observed, spec: FigureSpec, colour: str) -> None:
    """The observed jumps per bin, in the condition's colour (pale), as the card draws them.

    ONE ARTIST, NOT ONE PER BIN (3 Nov 2026): at the 10 nm bin a histogram has about eighty bars, and ``ax.bar``
    made each its own patch, whose extent matplotlib measures curve by curve at every layout pass, check and
    render -- a five-condition "every condition" figure carried 386 of them and was the slowest panel in the
    composer (3.8 s of a 15 s preview for six panels). The same rectangles as one `PolyCollection`, the same
    colour, gap and order, measured in one vectorised step; the axis still starts at zero (its sticky edge)."""
    from matplotlib.collections import PolyCollection

    rectangles = []
    for a, b, height in zip(edges, edges[1:], observed):
        w = b - a
        left, right = a + w * 0.04, a + w * 0.96
        rectangles.append([(left, 0.0), (right, 0.0), (right, height), (left, height)])
    bars = PolyCollection(rectangles, facecolors=_pale(spec, colour, 0.38), edgecolors="none", linewidths=0,
                          zorder=1)
    bars.set_gid(JUMP_BARS_GID)
    bars.sticky_edges.y.append(0.0)
    ax.add_collection(bars, autolim=True)
    ax.autoscale_view()


def jump_bar_heights(ax) -> list:
    """The heights `_jump_bars` drew on ``ax``, bin by bin (for the checks and the tests)."""
    bars = next(c for c in ax.collections if c.get_gid() == JUMP_BARS_GID)
    return [float(max(v[1] for v in path.vertices)) for path in bars.get_paths()]


def _overview_grid(n: int, width_mm: float, min_mm: float = 58.0) -> tuple:
    """``(columns, rows)`` for ``n`` histograms across ``width_mm``: as many columns of at least ``min_mm`` (58 mm, a
    histogram's) as fit, then as few as keep the same number of rows (four in two rows of two, not three and one)."""
    cols = max(1, min(n, int(width_mm // min_mm)))
    rows = math.ceil(n / cols)
    while cols > 1 and math.ceil(n / (cols - 1)) == rows:
        cols -= 1
    return cols, rows


#: The overview's key row, in lines of the small font (`_overview_key_mm`).
_OVERVIEW_KEY_LINES = 2


def _overview_key_mm(spec: FigureSpec, lines: int = _OVERVIEW_KEY_LINES) -> float:
    return (lines * spec.small_pt() * 1.55 + spec.small_pt()) * _PT_MM


def _overview_places(payload: dict, spec: FigureSpec, width_mm: float) -> dict:
    """Where the overview's panels stand: ``{"layout", "order", "rows", "columns", "chosen", "panels": [(entry,
    counts, row, column)]}`` -- `jumps_view.overview_tiles`, the screen's rule, with the payload's ``"layout"`` and
    ``"order"`` (the All conditions view's switches; one plot per condition when absent). One plot per condition:
    balanced across the width (`_overview_grid`); one plot per fit: the tiles' own grid."""
    from .jumps_view import OVERVIEW_LAYOUTS, OVERVIEW_ORDERS, overview_tiles

    layout, order = payload.get("layout") or "condition", payload.get("order") or "down"
    if layout not in OVERVIEW_LAYOUTS or order not in OVERVIEW_ORDERS:
        raise FigureExportError(f"unknown layout of the jump histograms: {layout!r}, {order!r}")
    entries = payload["entries"]
    chosen = {e["key"]: _chosen(e, spec)[0] for e in entries}
    tiles = overview_tiles([(e["key"], chosen[e["key"]], _jump_fits(e["result"])) for e in entries],
                           layout, order)["tiles"]
    by_key = {e["key"]: e for e in entries}
    if layout == "condition":
        # the Grid row's columns where one is chosen (7 Nov 2026, "draw and export as grids for example 4x2 / 2x4"),
        # else balanced across the width
        if spec.grid_columns:
            cols = max(1, min(len(tiles), int(spec.grid_columns)))
            rows = math.ceil(len(tiles) / cols)
        else:
            cols, rows = _overview_grid(len(tiles), width_mm)
        panels = [(by_key[t["key"]], t["show"], i // cols, i % cols) for i, t in enumerate(tiles)]
    else:
        rows = 1 + max(t["row"] for t in tiles)
        cols = 1 + max(t["column"] for t in tiles)
        panels = [(by_key[t["key"]], t["show"], t["row"], t["column"]) for t in tiles]
    return {"layout": layout, "order": order, "rows": rows, "columns": cols, "chosen": chosen, "panels": panels}


def _jumps_overview_height(payload: dict, spec: FigureSpec) -> float:
    """Each panel about 0.72 times as tall as wide, plus its title, and the key's row above them -- the panels as
    many across and down as the layout puts them (`_overview_places`)."""
    width = spec.width_value()
    try:
        places = _overview_places(payload, spec, width)
        cols, rows = places["columns"], places["rows"]
    except (FigureExportError, KeyError, TypeError, ValueError):
        # a payload the figure will refuse by name; its height is never drawn
        cols, rows = _overview_grid(max(1, len(payload.get("entries") or ())), width)
    plot = width / cols * 0.72
    strips = _jumps_strips(spec, "jumps_overview")
    if strips:
        # each tile's residual plots under it, at the card figure's shares of the histogram
        plot *= sum(_JUMPS_SHARES[n] for n in ("histogram", *strips)) / _JUMPS_SHARES["histogram"]
    panel = plot + 3.0 * spec.small_pt() * _PT_MM
    key = _overview_key_mm(spec) if spec.legend else 0.0
    tallest = spec.preset_values.get("max_height_mm") or 600.0
    return round(min(tallest, max(40.0, rows * panel + key + 14.0)), 1)


def _draw_jumps_overview(fig, payload: dict, spec: FigureSpec, family: str):
    """Every condition's jump histogram in one figure, each as its card draws it (`_chosen`: the fits its Show has
    on, else the dialog's Fit row for every panel; its components as its Components box, else the dialog's), on ONE
    x axis. Asked 26 Oct 2026: "a way to print all currently loaded jd plots histogram into one screen and export
    together".

    THE RESIDUAL PLOTS ARE OPTIONAL HERE TOO (30 Oct 2026, asked: "maybe we include it in exporter for every jump
    distance as an option? even for each/all condition etc windows"): the Export dialog's "Pearson residuals" and
    "ECDF residuals" boxes (`FigureSpec.pearson_residuals` / `.ecdf_residuals`, OFF by default here:
    `RESIDUALS_BY_DEFAULT`) put each tile's strips under its histogram, drawn as the card's figure draws them
    (`_pearson_strip`, `_ecdf_strip`), on the histograms' x axis; the strips of one row of tiles share one y range
    (every tile's with "Same y scale"), as a row's histograms read across.

    LAID OUT AS THE PAGE (30 Oct 2026, `_overview_places`): one plot per condition, each its own y axis (the
    conditions have their own jump counts); or one plot per FIT, a condition's fits apart, the conditions down the
    rows or across the columns, a condition's plots on ONE y range (its bars are the same jumps) -- every plot one
    fit in the ink, its components (on) in shades of the condition's colour. Returns the first panel (the dialog's x
    range is shared by all)."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    from .jumps_view import bin_width, components_drawn, histogram_top, in_colour, in_units, shared_tops, y_label

    entries = payload.get("entries") or []
    if not entries:
        raise FigureExportError("no jump-distance result to draw yet")
    for e in entries:
        if "error" in (e.get("result") or {}) or not _jump_fits(e["result"]):
            raise FigureExportError(f"{e.get('name', 'a condition')} has no jump-distance fit to draw")
    # ONE Y UNIT for every panel (30 Oct 2026; `_overview_y_mode`), each condition's bars and curves divided by its
    # own divisor; "Same y scale" (the payload's ``same_y``) puts every panel on the largest top, as on the page
    mode = _overview_y_mode(entries, spec)
    units = {e["key"]: in_units(mode, e["result"]["observed"], _jump_fits(e["result"]), e["result"]["edges_nm"])
             for e in entries}
    typed = (payload.get("axes") or {}).get("y") or (None, None)
    page = _page(spec)
    small, line = spec.small_pt(), spec.line_pt()
    lw = _jump_widths(spec)
    strong = lw["fit"]
    width_mm, height_mm = spec.size_mm()
    places = _overview_places(payload, spec, width_mm)
    chosen, panels, by_fit = places["chosen"], places["panels"], places["layout"] == "fit"
    cols, rows = places["columns"], places["rows"]
    strips = _jumps_strips(spec, "jumps_overview")
    one_choice = len(set(chosen.values())) == 1
    # one plot per fit: a condition's plots share its y range, as on the page; "Same y scale": every panel one
    tops = {e["key"]: histogram_top(units[e["key"]][0], units[e["key"]][1], chosen[e["key"]]) for e in entries}
    if payload.get("same_y"):
        tops = shared_tops({key: (mode, top) for key, top in tops.items()})
    # THE KEY IN A ROW OF ITS OWN over the panels (above the first panel it lay on that panel's title)
    key = _overview_key_mm(spec) if spec.legend else 0.0
    panel = max(10.0, (height_mm - key) / rows)
    # ONE GRID, A TILE'S STRIPS IN ROWS OF THEIR OWN (3 Nov 2026, reported: "2nd row of pearson gets shifted to the
    # right" and "the probability per 10 nm bin y axis title needs two rows?"). Each tile was a sub-grid of its own
    # cell, so the lowest row's x numbers and title came out of ITS plots' height (18 % shorter), its y title no
    # longer fitted and was set smaller or wrapped, and constrained layout -- which shares a column's left margin
    # across the rows of ONE grid only -- moved that row's first plot sideways. In one grid every plot of a row is
    # the same height and every column one left edge.
    per = 1 + len(strips)
    shares = [_JUMPS_SHARES[n] for n in ("histogram", *strips)]
    grid = fig.add_gridspec(rows * per + (1 if key else 0), cols,
                            height_ratios=([key] if key else []) + [panel * s / sum(shares) for s in shares] * rows)
    top_row = 1 if key else 0
    first = None
    lowest = {c: max(r for _e, _k, r, cc in panels if cc == c) for _e, _k, _r, c in panels}
    leftmost = {r: min(c for _e, _k, rr, c in panels if rr == r) for _e, _k, r, _c in panels}
    x_end = max(e["result"]["edges_nm"][-1] for e in entries)
    if "ecdf" in strips:
        # the ECDF residuals run a little past the last bin (their points reach the 99.8th percentile)
        x_end = max([x_end] + [x for e, counts, _r, _c in panels for k in counts
                               for x, _v in _jump_fits(e["result"])[k]["ecdf_residual"]])
    # ONE Y RANGE PER ROW for each residual plot (every tile's under "Same y scale"), as a row's histograms read across
    spans = {}
    for e, counts, r, _c in panels:
        fits_e = _jump_fits(e["result"])
        row = "all" if payload.get("same_y") else r
        got = spans.setdefault(row, {"pearson": 0.0, "ecdf": 0.0})
        got["pearson"] = max(got["pearson"], _pearson_span(fits_e, counts))
        got["ecdf"] = max(got["ecdf"], _ecdf_span(fits_e, counts))
    drawn_parts = {"several": set(), "one": False}
    for e, counts, r, c in panels:
        share = {} if first is None else {"sharex": first}
        ax = fig.add_subplot(grid[top_row + r * per, c], **share)
        first = first or ax
        under = {n: fig.add_subplot(grid[top_row + r * per + 1 + i, c], sharex=first) for i, n in enumerate(strips)}
        result = e["result"]
        bars, fits = units[e["key"]]
        colour = e.get("colour") or QC_DATA
        edges = result["edges_nm"]
        centres = [(a + b) / 2.0 for a, b in zip(edges, edges[1:])]
        _jump_bars(ax, edges, bars, spec, colour)
        drawn = list(counts)
        every = in_colour(drawn)
        parts = components_drawn(fits, drawn, _components_on(e, spec))
        for kk in parts:
            comps = fits[kk]["components"]
            for i, comp in enumerate(comps):
                if every:
                    ax.plot(centres, comp, ls=_COMPONENT_DASH, color=_fit_colour(kk, spec),
                            lw=lw["component_many"], alpha=0.9, zorder=3)
                else:
                    ax.plot(centres, comp, "-", color=figstyle.day_shade(colour, i, len(comps)),
                            lw=lw["component"], zorder=2)
        if every:
            drawn_parts["several"].update(parts)
        else:
            drawn_parts["one"] = drawn_parts["one"] or bool(parts)
        for kk in drawn:
            ax.plot(centres, fits[kk]["expected"], _JUMP_STYLES.get(kk, "-") if every else "-",
                    color=_fit_colour(kk, spec) if every else page["ink"], lw=strong, zorder=4)
        top = tops[e["key"]] if (by_fit or payload.get("same_y")) else histogram_top(bars, fits, drawn)
        lo = 0.0 if typed[0] is None else float(typed[0])
        ax.set_ylim(lo, top * 1.06 if typed[1] is None else float(typed[1]))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=mode == "counts"))
        name = spec.shown_name(e["name"])
        title = (f"{name} · {_plural(drawn[0])}" if by_fit
                 else name if one_choice else f"{name} · {_shown_words(counts)}")
        ax.set_title(title, fontsize=small, loc="left", pad=2.0)
        if c == leftmost[r]:
            ax.set_ylabel(spec.y_title if spec.y_title is not None
                          else y_label(mode, [bin_width(x["result"]["edges_nm"]) for x in entries]))
        # the tile's residual plots under it, the row's y range (`_pearson_strip`, `_ecdf_strip`)
        span = spans["all" if payload.get("same_y") else r]
        if "pearson" in under:
            _pearson_strip(under["pearson"], edges, fits, drawn, spec, span["pearson"])
        if "ecdf" in under:
            _ecdf_strip(under["ecdf"], fits, drawn, spec, span["ecdf"])
        if c == leftmost[r]:
            for n, words in (("pearson", chart_text.PEARSON_LABEL), ("ecdf", chart_text.ECDF_RESIDUAL_LABEL)):
                if n in under:
                    under[n].set_ylabel(words.replace(" ", "\n", 1))
        # the x axis's numbers and title under the tile's lowest plot, in the grid's lowest row of each column
        stack = [ax, *under.values()]
        for upper in stack[:-1]:
            upper.tick_params(labelbottom=False)
        if r == lowest[c]:
            stack[-1].set_xlabel(_title(spec, "jumps_overview", "x"))
        else:
            stack[-1].tick_params(labelbottom=False)
        if spec.grid:
            for one_ax in stack:
                one_ax.grid(axis="y", color=page["grid"], lw=max(MIN_LINE_PT, line * 0.6), zorder=0)
                one_ax.set_axisbelow(True)
    x_typed = (payload.get("axes") or {}).get("x") or (None, None)
    first.set_xlim(0.0 if x_typed[0] is None else float(x_typed[0]), x_end if x_typed[1] is None else float(x_typed[1]))
    if spec.legend:
        handles, labels = [], []
        # the fits drawn and nothing else, as SMTracker 2's key (the components are said in the legend's words): the
        # counts drawn in colour somewhere (a panel with two or more fits on), then the one-fit panels' ink
        shown_sets = [counts for _e, counts, _r, _c in panels]
        coloured = sorted({kk for v in shown_sets if len(v) > 1 for kk in v})
        handles += [Line2D([], [], ls=_JUMP_STYLES.get(kk, "-"), color=_fit_colour(kk, spec), lw=strong)
                    for kk in coloured]
        labels += [_plural(kk) for kk in coloured]
        singles = sorted({v[0] for v in shown_sets if len(v) == 1})
        if singles:
            handles.append(Line2D([], [], color=page["ink"], lw=strong))
            mixed = len(singles) > 1 or bool(coloured)
            labels.append("one fit, as named in its panel" if mixed else _plural(singles[0]))
        holder = fig.add_subplot(grid[0, :])
        holder.set_axis_off()
        room = width_mm / _PT_MM * 0.94
        labels, ncol = _legend_columns(labels, room, spec, family)
        holder.legend(handles, labels, loc="center", ncol=ncol, frameon=False, handletextpad=0.4,
                      columnspacing=1.0, borderaxespad=0.0, handlelength=1.8)
    return first


def _draw_jd(fig, ax, entry: dict, spec: FigureSpec, family: str) -> None:
    """A Signed jumps card in print (`jd_view.card_payload`, the card's own numbers): the signed jumps per bin in
    grey, every fit's expected count in v2's line styles (the chosen one strong), the chosen fit's populations in
    shades of the condition's colour; the key above, titled with the condition and the axis."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    from .jd_view import JD_AXIS_LABELS, card_payload

    chosen, _how = _jd_chosen(entry, spec)
    colour = entry.get("colour") or QC_DATA
    card = card_payload(entry["result"], entry["axis"], chosen, colour)
    if "error" in card:
        raise FigureExportError(card["error"])
    h = card["histogram"]
    fits = {int(k): v for k, v in h["fits"].items()}
    numbers_src = _jd_fits(entry)[chosen]
    page = _page(spec)
    small, line = spec.small_pt(), spec.line_pt()
    strong, weak = max(1.0, line * 1.8), max(0.5, line * 0.9)
    edges = list(h["edges"])
    widths = [b - a for a, b in zip(edges, edges[1:])]
    centres = [(a + b) / 2 for a, b in zip(edges, edges[1:])]
    components = fits[chosen]["components"]
    n_comp = len(components)

    def ink(k):
        return page["ink"] if k == chosen else page["muted"]

    ax.bar([a + w * 0.04 for a, w in zip(edges, widths)], h["observed"], width=[w * 0.92 for w in widths],
           align="edge", color=_pale(spec, _QC_GREY, 0.38), linewidth=0, zorder=1)
    for c, comp in enumerate(components):
        ax.plot(centres, comp, "-", color=figstyle.day_shade(colour, c, n_comp), lw=max(0.6, line), zorder=2)
    for k in sorted(fits):
        ax.plot(centres, fits[k]["expected"], _JUMP_STYLES.get(k, "-"), color=ink(k),
                lw=strong if k == chosen else weak, zorder=4 if k == chosen else 3)
    top = max([*h["observed"], *(v for f in fits.values() for v in f["expected"])] + [1.0])
    ax.set_ylim(0, top * 1.06)
    ax.set_xlim(edges[0], edges[-1])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.set_xlabel(_title(spec, "jd_condition", "x"))
    ax.set_ylabel(_title(spec, "jd_condition", "y"))
    _qc_grid(ax, spec)
    numbers = [f"D {d:.3g} µm²/s · {100 * f:.1f} %" for d, f in zip(numbers_src["D"], numbers_src["fractions"])]
    if spec.legend:
        handles = [Line2D([], [], ls=_JUMP_STYLES.get(k, "-"), color=ink(k), lw=strong if k == chosen else weak)
                   for k in sorted(fits)]
        labels = [_plural(k) for k in sorted(fits)]
        handles += [Line2D([], [], color=figstyle.day_shade(colour, c, n_comp), lw=max(0.6, line))
                    for c in range(n_comp)]
        labels += [numbers[c] if spec.numbers else f"population {c + 1} of {chosen}" for c in range(n_comp)]
        title = spec.shown_name(entry["name"])
        if spec.title and title and title in spec.title:
            title = None
        axis_words = JD_AXIS_LABELS[entry["axis"]].lower()
        _legend_above(ax, handles, labels, spec, family,
                      title=f"{title} · {axis_words}" if title else axis_words)
    elif spec.numbers:
        ax.annotate(f"{_plural(chosen)}:\n" + "\n".join(numbers), (0.98, 0.96), xycoords="axes fraction",
                    ha="right", va="top", fontsize=small, color=page["ink"])


def _jd_source(entry: dict, spec: FigureSpec) -> tuple:
    """Per bin: its edges, the observed jumps, every fit's expected count and the chosen fit's populations; the
    card's table as a second sheet."""
    from .jd_view import JD_AXIS_LABELS, card_payload

    chosen, _how = _jd_chosen(entry, spec)
    card = card_payload(entry["result"], entry["axis"], chosen, entry.get("colour") or QC_DATA)
    h = card["histogram"]
    fits = {int(k): v for k, v in h["fits"].items()}
    order = sorted(fits)
    n_comp = len(fits[chosen]["components"])
    axis = JD_AXIS_LABELS[entry["axis"]]
    columns = (["Condition", "Shown as", "Axis", "Bin from (nm)", "Bin to (nm)", "Observed jumps"]
               + [f"Expected, {_plural(k)}" for k in order]
               + [f"Population {c + 1} of the {chosen}-population fit (expected)" for c in range(n_comp)])
    edges = h["edges"]
    rows = []
    for i, count in enumerate(h["observed"]):
        rows.append([entry["key"], spec.shown_name(entry["name"]), axis, float(edges[i]), float(edges[i + 1]),
                     int(count)] + [float(fits[k]["expected"][i]) for k in order]
                    + [float(fits[chosen]["components"][c][i]) for c in range(n_comp)])
    table = ("Populations", f"{axis}, {_plural(chosen)}", (card["caption"],), list(card["columns"]),
             [list(r) for r in card["rows"]])
    return (columns, rows), (table,)


def jd_legend(entry: dict, spec: FigureSpec) -> str:
    from .jd_view import JD_AXIS_LABELS, card_payload

    chosen, how = _jd_chosen(entry, spec)
    card = card_payload(entry["result"], entry["axis"], chosen, entry.get("colour") or QC_DATA)
    axis = JD_AXIS_LABELS[entry["axis"]].lower()
    where = {"along": "relative to each track's own cell axis, along it",
             "across": "relative to each track's own cell axis, across it",
             "xy": "on the camera's x and y axes, stacked"}[entry["axis"]]
    text = (f"Signed one-dimensional jumps of {spec.shown_name(entry['name'])} ({axis}), each measured {where}; "
            "a jump to one side is negative. Grey: jumps per bin. Lines: expected counts of zero-mean normal "
            "mixtures of 1 (dotted), 2 (solid) and 3 (dashed) populations; the "
            f"{_plural(chosen)} fit ({how}) is drawn strong, with its populations in colour. "
            f"D = sigma^2 / (2 tau). {card['caption']}")
    if entry["axis"] != "xy":
        text += (" A cell is about 1 um wide, so the membrane cuts short the jumps of a fast molecule across it. "
                 "The along / across D ratio therefore reflects confinement; it is not a direction-dependent coefficient.")
    text += (" This is a cross-check of the SQD fractions on another observable and is not averaged with them. Any "
             "spread is the SD over replicates, each fitted on its own.")
    return text


def _jumps_source(entry: dict, spec: FigureSpec) -> tuple:
    """Per bin, as numbers: its edges, the observed count, the DRAWN fits' expected counts and (their plot drawn)
    Pearson residuals, the one fit's components; the drawn fits' ECDF residual points (their plot drawn) and the
    fit-quality table of EVERY fit made, saying which are drawn, as further sheets."""
    result, name = entry["result"], entry["key"]
    shown = spec.shown_name(entry["name"])
    from .jumps_view import components_drawn, in_units

    # in the unit drawn (30 Oct 2026): the bars and the curves as plotted; the raw count of jumps beside them
    mode = _y_mode_of(entry, spec)
    edges = result["edges_nm"]
    bars, fits = in_units(mode, result["observed"], _jump_fits(result), edges)
    unit = "" if mode == "counts" else f" ({_unit_words(mode, edges)})"
    drawn, _how = _chosen(entry, spec)
    order = list(drawn)
    pearson = residuals_on(spec, "jumps_condition", "pearson")
    # the components drawn: the one fit's, or several fits' (each with more than one population)
    comp_of = [(k, c) for k in components_drawn(fits, order, _components_on(entry, spec))
               for c in range(len(fits[k]["components"]))]
    comps = [fits[k]["components"][c] for k, c in comp_of]
    columns = (["Condition", "Shown as", "Bin from (nm)", "Bin to (nm)", "Observed jumps"]
               + ([f"Observed{unit}"] if mode != "counts" else [])
               + [f"Expected, {_plural(k)}{unit}" for k in order]
               + ([f"Pearson residual, {_plural(k)}" for k in order] if pearson else [])
               + [f"Component {c + 1} of the {k}-population fit (expected){unit}" for k, c in comp_of])
    rows = []
    for i, count in enumerate(result["observed"]):
        rows.append([name, shown, float(edges[i]), float(edges[i + 1]), int(count)]
                    + ([float(bars[i])] if mode != "counts" else [])
                    + [float(fits[k]["expected"][i]) for k in order]
                    + ([float(fits[k]["pearson"][i]) for k in order] if pearson else [])
                    + [float(comp[i]) for comp in comps])
    sheets = []
    if residuals_on(spec, "jumps_condition", "ecdf"):
        ecdf_rows = [[k, float(x), float(v)] for k in order for x, v in fits[k]["ecdf_residual"]]
        sheets.append(("ECDF residuals", "Model CDF minus empirical CDF, the fits drawn",
                       ("Evaluated at evenly spaced quantiles of the squared displacement, drawn "
                        "against the jump distance; exact, not binned.",),
                       ["Populations", "Jump distance (nm)", "Model − ECDF (0-1)"], ecdf_rows))
    cv = result.get("cv") or {}
    made = sorted(fits)
    widest = max(len(fits[k]["D"]) for k in made)
    quality_columns = (["Populations"] + [f"D{c + 1} (µm²/s)" for c in range(widest)]
                       + [f"Fraction {c + 1} (0-1)" for c in range(widest)]
                       + ["BIC (maximum likelihood)", "ΔBIC"]
                       + [f"Maximum-likelihood D{c + 1} (µm²/s)" for c in range(widest)]
                       + [f"Maximum-likelihood fraction {c + 1} (0-1)" for c in range(widest)]
                       + ["Largest ECDF gap (0-1)", "Held-out ΔlogL", "Held-out ΔlogL per jump",
                          "Folds won", "Folds", "Drawn in the figure"])
    quality = []
    for k in made:
        f = fits[k]

        def padded(values):
            values = list(values or [])
            return [float(v) for v in values] + [None] * (widest - len(values))

        quality.append([k] + padded(f["D"]) + padded(f["fractions"])
                       + [f.get("bic"), f.get("delta_bic")]
                       + padded(f.get("bic_D")) + padded(f.get("bic_fractions"))
                       + [float(f["ks"]), (cv.get("delta_total") or {}).get(k),
                          (cv.get("delta_per_step") or {}).get(k),
                          (cv.get("wins") or {}).get(k) if cv.get("wins") else None,
                          cv.get("n_folds"), k in drawn])
    notes = list(result.get("table", {}).get("notes", ()))
    notes.append("D and fractions: the drawn curves, from the SQD fit (ECDF least "
                 "squares, step-weighted, 2000 points). BIC and the maximum-likelihood D and "
                 "fractions: maximum-likelihood fits of the same data, which can differ "
                 "slightly.")
    notes.append("No p-value is given for any comparison. Jumps within a track and tracks "
                 "within a cell are not independent, so a test on pooled jumps would claim "
                 "far more evidence than there is, and its result depends on the bin width.")
    sheets.append(("Fit quality", "How well each fit fits", tuple(notes), quality_columns, quality))
    return (columns, rows), tuple(sheets)


#: Said in the notes and the legend of a figure exported from the QUICK pass
#: (`jumps_view.compute_jump_view(quality=False)`: ``"pending"``), so a figure made before
#: BIC and the folds arrived cannot pass for one that had them.
_JUMPS_PENDING = ("BIC (maximum-likelihood fits) and the held-out fits were not finished when this figure "
                  "was exported, so their columns are empty. Export it again once the card has filled "
                  "them in.")


def _residual_words(spec: FigureSpec, kind: str = "jumps_condition") -> str:
    """The residual plots a jump figure draws, in words: "Pearson and ECDF residuals", one of them, or ""."""
    names = {"pearson": "Pearson", "ecdf": "ECDF"}
    strips = [names[n] for n in _jumps_strips(spec, kind)]
    return f"{' and '.join(strips)} residuals" if strips else ""


def _jumps_source_notes(entry: dict, spec: FigureSpec) -> tuple:
    from .jumps_view import card_caption

    from .jumps_view import components_drawn

    result = entry["result"]
    drawn, how = _chosen(entry, spec)
    binned = sum(result["observed"])
    pending = ((_JUMPS_PENDING,) if result.get("pending") else ())
    residuals = _residual_words(spec)
    parts = components_drawn(_jump_fits(result), drawn, _components_on(entry, spec))
    several = (f", the components of the {', '.join(str(k) for k in parts)}-population fit"
               f"{'s' if len(parts) > 1 else ''} dashed in its colour" if parts else ", without components")
    return (card_caption(result) + ".", *pending,
            (f"Drawn: the fits of {_shown_words(drawn)} ({how}), each in its own colour and SMTracker 2's line style "
             f"({_fit_style_words(drawn)}){several}"
             + (f"; their {residuals}." if residuals else "; no residual plot.")
             if len(drawn) > 1 else
             f"Chosen fit: {_plural(drawn[0])} ({how}), the only fit drawn"
             + (", with its components" if parts
                else ", without its components" if len(_jump_fits(result)[drawn[0]]["components"]) > 1 else "")
             + (f", and the one whose {residuals} are shown." if residuals else "; no residual plot."))
            + " This sheet holds the fits drawn; the Fit quality sheet holds every fit made.",
            *(("The key sits inside the histogram where it was put on the card.",)
              if entry.get("legend_at") is not None and spec.legend else ()),
            "Bars: the observed jumps per bin, in the condition's colour.",
            _unit_note(_y_mode_of(entry, spec), result),
            "Expected: the jumps a fit predicts in the bin, n × (CDF at the bin's end − CDF at "
            "its start), an exact integral over the bin.",
            *(("Pearson residual: (observed − expected) / √expected, from the counts whatever the y axis shows. "
               "It would be about ±1 if every jump were independent; jumps within a track are not, so "
               "read the pattern (a run of one sign marks a misfit), not the size.",)
             if residuals_on(spec, "jumps_condition", "pearson") else ()),
            f"The histogram runs to the 99.5th percentile of the jump length: {binned:,} of "
            f"{result['n_steps']:,} jumps are binned.",
            "No p-value is given because pooled jumps are not independent observations.")


def _unit_note(mode: str, result: dict) -> str:
    """What the y axis is, said in the source data's notes: the divisor and the bin width."""
    from .jumps_view import bin_width

    n, w = sum(result["observed"]), bin_width(result["edges_nm"])
    if mode == "counts":
        return "Y axis: Counts, the jumps per bin (curves: the jumps a fit predicts per bin)."
    if mode == "density":
        return (f"Y axis: probability density per nm, the counts and curves divided by the {n:,} jumps binned "
                f"times the {w:.3g} nm bin width (the bars' area is 1). The 'Observed jumps' column is the count.")
    return (f"Y axis: probability per {w:.3g} nm bin, as in SMTracker 2 and Spot-On: the counts and curves "
            f"divided by the {n:,} jumps binned (the bars sum to 1). The 'Observed jumps' column is the count.")


def _fit_style_words(counts=None) -> str:
    """"1 population slate grey, dotted; 2 populations maroon, solid; ..." -- a figure's fits in colour, in words
    (``counts``: those drawn; every one when None)."""
    from .jumps_view import FIT_COLOUR_WORDS

    counts = sorted(FIT_COLOUR_WORDS) if counts is None else sorted(int(k) for k in counts)
    return "; ".join(f"{_plural(k)} {FIT_COLOUR_WORDS[k]}, {_JUMP_STYLE_WORDS[k]}" for k in counts
                     if k in FIT_COLOUR_WORDS)


def _jumps_overview_source(payload: dict, spec: FigureSpec) -> tuple:
    """Per condition and bin: what was drawn, the observed jumps and the DRAWN fits' expected counts (empty where a
    panel does not draw that fit); every fit each condition made as a second sheet, saying which are drawn."""
    from .jumps_view import in_units

    entries = payload["entries"]
    chosen = {e["key"]: _chosen(e, spec)[0] for e in entries}
    counts = sorted({k for v in chosen.values() for k in v})
    mode = _overview_y_mode(entries, spec)
    unit = {"probability": " (probability per bin)", "counts": "", "density": " (probability density, 1/nm)"}[mode]
    pearson = residuals_on(spec, "jumps_overview", "pearson")
    columns = (["Condition", "Shown as", "Drawn", "Bin from (nm)", "Bin to (nm)", "Observed jumps"]
               + ([f"Observed{unit}"] if mode != "counts" else [])
               + [f"Expected, {_plural(k)}{unit}" for k in counts]
               + ([f"Pearson residual, {_plural(k)}" for k in counts] if pearson else []))
    rows, fit_rows, ecdf_rows = [], [], []
    widest = max(len(f["D"]) for e in entries for f in _jump_fits(e["result"]).values())
    for e in entries:
        result = e["result"]
        bars, fits = in_units(mode, result["observed"], _jump_fits(result), result["edges_nm"])
        shown = spec.shown_name(e["name"])
        on = chosen[e["key"]]
        drawn = _shown_words(on)
        edges = result["edges_nm"]
        for i, count in enumerate(result["observed"]):
            rows.append([e["key"], shown, drawn, float(edges[i]), float(edges[i + 1]), int(count)]
                        + ([float(bars[i])] if mode != "counts" else [])
                        + [float(fits[k]["expected"][i]) if k in on else None for k in counts]
                        + ([float(fits[k]["pearson"][i]) if k in on else None for k in counts] if pearson else []))
        if residuals_on(spec, "jumps_overview", "ecdf"):
            ecdf_rows += [[e["key"], shown, k, float(x), float(v)] for k in on for x, v in fits[k]["ecdf_residual"]]
        for k in sorted(fits):
            f = fits[k]
            pad = widest - len(f["D"])
            fit_rows.append([e["key"], shown, k] + [float(d) for d in f["D"]] + [None] * pad
                            + [float(x) for x in f["fractions"]] + [None] * pad
                            + [f.get("delta_bic"), float(f["ks"]), k in on])
    fit_sheet = ("Fits", "Each condition's fits",
                 ("D and fractions: the drawn curves, the SQD section's fit (ECDF least squares, step-weighted). "
                  "ΔBIC: from maximum-likelihood fits of the same data (empty if they were not finished).",),
                 ["Condition", "Shown as", "Populations"] + [f"D{c + 1} (µm²/s)" for c in range(widest)]
                 + [f"Fraction {c + 1} (0-1)" for c in range(widest)]
                 + ["ΔBIC", "Largest ECDF gap (0-1)", "Drawn in its panel"],
                 fit_rows)
    sheets = (fit_sheet,)
    if residuals_on(spec, "jumps_overview", "ecdf"):
        sheets = (("ECDF residuals", "Model CDF minus empirical CDF, the fits drawn",
                   ("Each condition's fits drawn, evaluated at evenly spaced quantiles of the squared displacement "
                    "and plotted against the jump distance (exact, not binned).",),
                   ["Condition", "Shown as", "Populations", "Jump distance (nm)", "Model − ECDF (0-1)"], ecdf_rows),
                  ) + sheets
    return (columns, rows), sheets


def _jumps_overview_notes(payload: dict, spec: FigureSpec) -> tuple:
    entries = payload["entries"]
    pending = any(e["result"].get("pending") for e in entries)
    by_fit = payload.get("layout") == "fit"
    layout = ("One plot per fit, with the conditions "
              f"{'down the rows' if payload.get('order', 'down') == 'down' else 'across the columns'}. All plots "
              "share one x axis, and each condition's plots share one y axis." if by_fit else
              "One plot per condition. All panels share one x axis and each has its own y axis.")
    return ((f"{len(entries)} conditions, each drawn as its card shows it unless the Export dialog's Fit row "
             "set the fits for all (the Drawn column). The Expected columns hold the fits drawn and are empty "
             "where a panel does not draw that fit.",)
            + ((_JUMPS_PENDING,) if pending else ())
            + (layout,
               _overview_unit_note(payload, spec),
               "Bars: the observed jumps per bin, in the condition's colour, up to its 99.5th percentile.",
               "Expected: the jumps a fit predicts in the bin, n × (CDF at the bin's end − CDF at its start), "
               "an exact integral over the bin.")
            + (("Pearson residual: (observed − expected) / √expected per bin, from the counts whatever the y axis "
                "shows. Jumps within a track are not independent, so read the pattern, not the size.",)
               if residuals_on(spec, "jumps_overview", "pearson") else ())
            + ((f"Residual plots under each plot: {_residual_words(spec, 'jumps_overview')}, each row of plots on "
                f"one y range{' (every plot one, Same y scale)' if payload.get('same_y') else ''}.",)
               if _jumps_strips(spec, "jumps_overview") else ("No residual plot.",))
            + ("No p-value is given because pooled jumps are not independent observations.",))


def _overview_unit_note(payload: dict, spec: FigureSpec) -> str:
    """Export all's y unit in words, and whether every panel shares one y range."""
    mode = _overview_y_mode(payload["entries"], spec)
    unit = {"probability": "probability per bin (each condition's counts and curves divided by its jumps binned)",
            "counts": "the jumps per bin", "density": "probability density per nm (area 1 for each condition)"}[mode]
    return (f"Y axis: {unit}. " + ("Every panel has the same y range (Same y scale)." if payload.get("same_y")
                                    else "Each panel has its own y range."))


def jumps_overview_legend(payload: dict, spec: FigureSpec | None = None) -> str:
    """A drafted legend for the figure of every condition's jump histogram."""
    spec = spec or FigureSpec()
    entries = payload["entries"]
    chosen = {e["key"]: _chosen(e, spec)[0] for e in entries}
    names = ", ".join(f"{spec.shown_name(e['name'])} ({e['result']['n_steps']:,} jumps)" for e in entries)
    coloured = sorted({k for v in chosen.values() if len(v) > 1 for k in v})
    if len(set(chosen.values())) == 1:
        k = next(iter(chosen.values()))
        drawn = (f"the {_plural(k[0])} fit (line) with its components in shades of the condition's colour"
                 if len(k) == 1 else f"the fits of {_shown_words(k)} ({_fit_style_words(k)})")
    else:
        drawn = ("the fits named in each panel (one fit: a line with its components in shades of the condition's "
                 "colour"
                 + (f"; several: {_fit_style_words(coloured)}, with dashes for each fit's populations)"
                    if coloured else ")"))
    pending = " " + _JUMPS_PENDING if any(e["result"].get("pending") for e in entries) else ""
    unit = {"probability": "the probability per bin", "counts": "the jumps per bin",
            "density": "the probability density per nm"}[_overview_y_mode(entries, spec)]
    if payload.get("layout") == "fit":
        drawn = ("one fit in each plot, as named over it (line), with its components in shades of the condition's "
                 "colour where they are shown")
        axes = ("One plot per fit, with the conditions "
                f"{'down the rows' if payload.get('order', 'down') == 'down' else 'across the columns'}. One x "
                "axis; each condition's plots share one y axis.")
    else:
        axes = "One x axis, each panel its own y axis."
    if payload.get("same_y"):
        axes = axes.replace("each panel its own y axis", "every panel one y axis").replace(
            "each condition's plots on one y axis", "every plot one y axis")
    strips = _jumps_strips(spec, "jumps_overview")
    under = ""
    if strips:
        words = {"pearson": "the Pearson residuals per bin of the fit(s) drawn, (observed − expected) / √expected, "
                            "over the ±2 band (one fit: bars, with residuals beyond ±2 in orange)",
                 "ecdf": "the model CDF minus the empirical CDF, exact and unbinned"}
        under = (" Under each plot, " + "; then ".join(words[n] for n in strips)
                 + ("; every residual plot has one y range." if payload.get("same_y")
                    else "; the residual plots of a row share one y range.")
                 + (" Jumps within a track are not independent, so a run of one sign marks a misfit; the size is "
                    "not a test." if "pearson" in strips else ""))
    return (f"Jump distances of {names}: bars in each condition's colour, {unit} up to its 99.5th percentile; "
            f"drawn, {drawn}, the fit's share of the bin in the same unit (the SQD section's "
            f"squared-displacement fit: ECDF least squares, step-weighted, integrated over the bin). {axes}{under} "
            "No p-value is given, since pooled jumps are not independent."
            + pending)


def jumps_legend(entry: dict, spec: FigureSpec | None = None) -> str:
    """A drafted legend for the jump-distance figure."""
    spec = spec or FigureSpec()
    result = entry["result"]
    fits = _jump_fits(result)
    drawn, _how = _chosen(entry, spec)
    head = (f"Jump distances of {entry['name']} ({result['n_steps']:,} jumps from "
            f"{result['n_tracks']:,} tracks, τ = {1000 * result['tau_s']:.1f} ms)")
    # each plot's place on the figure (Top / Middle / Bottom, Left / Right), none for the histogram alone
    places = _jumps_places(spec)

    def at(plot: str, text: str) -> str:
        return f"{places[plot]}: {text}" if plot in places else text[0].upper() + text[1:]

    from .jumps_view import bin_width, components_drawn

    with_parts = components_drawn(fits, drawn, _components_on(entry, spec))
    mode = _y_mode_of(entry, spec)
    width = bin_width(result["edges_nm"])
    bars_words = {"counts": "the observed jumps per bin",
                  "probability": f"the probability per {width:.3g} nm bin (the jumps in it over the jumps binned)",
                  "density": "the probability density per nm (area 1)"}[mode]
    curve_words = {"counts": "expected count per bin", "probability": "expected probability per bin",
                   "density": "expected density"}[mode]
    if len(drawn) > 1:
        dashed = (f" Dashed lines in each fit's colour: each fit's populations (the components of the "
                  f"{' and '.join(str(k) for k in with_parts)}-population fit{'s' if len(with_parts) > 1 else ''})."
                  if with_parts else "")
        parts = [f"{head}, against the fits of {_shown_words(drawn)} to the squared-displacement distribution "
                 f"({_fit_style_words(drawn)}; the SQD section's ECDF least-squares fit, step-weighted). "
                 + at("histogram", f"bars in the condition's colour, {bars_words} up to the 99.5th "
                      f"percentile; lines, each fit's {curve_words}, the integral of its "
                      "distribution over the bin.") + dashed]
        if residuals_on(spec, "jumps_condition", "pearson"):
            parts.append(at("pearson", "each fit's Pearson residuals, (observed − expected) / √expected, over the "
                            "±2 band. Jumps within a track are not independent, so a run of one sign marks a "
                            "misfit; the size is not a test."))
        if residuals_on(spec, "jumps_condition", "ecdf"):
            parts.append(at("ecdf", "model CDF minus the empirical CDF for each fit, exact and unbinned."))
    else:
        one = drawn[0]
        f = fits[one]
        comps = "; ".join(f"D {d:.3g} µm²/s, {100 * x:.1f} %"
                          for d, x in zip(f["D"], f["fractions"]))
        parts = [f"{head}, against the {_plural(one)} fit of the squared-displacement distribution (the SQD "
                 "section's ECDF least-squares fit, step-weighted). "
                 + at("histogram", f"bars in the condition's colour, {bars_words} up to the 99.5th "
                      f"percentile; line, the fit's {curve_words}, the integral of its distribution "
                      "over the bin" + ("; shades of the condition's colour, the components of the "
                                        f"{_plural(one)} fit ({comps})." if with_parts else "."))]
        if residuals_on(spec, "jumps_condition", "pearson"):
            parts.append(at("pearson", f"Pearson residuals of the {_plural(one)} fit, (observed − expected) / "
                            "√expected, over the ±2 band with residuals beyond ±2 in orange. Jumps within a track "
                            "are not independent, so a run of one sign marks a misfit; the size is not a test."))
        if residuals_on(spec, "jumps_condition", "ecdf"):
            parts.append(at("ecdf", f"model CDF minus the empirical CDF of the {_plural(one)} fit, exact and "
                            "unbinned."))
    if result.get("pending"):
        parts.append(_JUMPS_PENDING)
    bic = result.get("bic_best")
    if bic is not None:
        deltas = ", ".join(f"{k}: {fits[k]['delta_bic']:.1f}" for k in sorted(fits)
                           if fits[k].get("delta_bic") is not None)
        parts.append(f"BIC (maximum-likelihood fits) prefers {_plural(int(bic))} (ΔBIC {deltas}); "
                     "BIC recommends a count, the choice is yours.")
    cv = result.get("cv") or {}
    unit = {"day": "replicates", "cell": "cells"}.get(cv.get("unit"), "units")
    if cv.get("wins"):
        wins = ", ".join(f"{k}: {cv['wins'].get(k, 0)} of {cv['n_folds']}" for k in sorted(fits)
                         if k in cv["wins"])
        parts.append(f"Fitted without each of {cv['n_folds']} folds of {cv['n_units']} {unit} "
                     f"and scored on it, the held-out likelihood is best at "
                     f"{_plural(int(cv['best']))} (folds won {wins}).")
    elif cv.get("note"):
        parts.append(f"No cross-validation: {cv['note']}")
    parts.append("The largest ECDF gap is descriptive, and no p-value is given because pooled jumps "
                 "are not independent observations. D values are apparent: the slow state's D "
                 "sits at the localisation floor and the fast state's at the detection "
                 "ceiling.")
    return " ".join(parts)


# -- QC ------------------------------------------------------------------------------ #
#: The QC charts' data ink where no condition colour is handed over (the screen's default).
QC_DATA = "#2b6cb0"
#: Grey for what a filter removed or a cell outline did not hold (the screen's muted ink).
_QC_GREY = "#71717a"


def _qc_movie_names(block: dict) -> list:
    """The per-movie charts' labels as the screen prints them: the part every movie shares
    (the condition's own prefix) dropped (`qc_view.short_labels`)."""
    from .qc_view import short_labels

    return short_labels(block["names"])


def _movie_axis(ax, labels: list, spec: FigureSpec) -> None:
    """One label under each movie's bar, turned upright and thinned so two never touch --
    the screen's rule (`_QCChart._x_categories`). Every movie is named in the Source Data,
    whichever are drawn."""
    n = max(1, len(labels))
    pitch = spec.width_value() / _MM * 72.0 * 0.8 / n
    step = max(1, math.ceil(spec.small_pt() * 1.35 / pitch))
    shown = list(range(0, len(labels), step))
    ax.set_xticks(shown)
    ax.set_xticklabels([spec.shown_name(labels[i]) for i in shown], rotation=90)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, n - 0.4)


def _legend_columns(labels, room: float, spec: FigureSpec, family: str) -> tuple:
    """``(labels, columns)``: a key's labels -- each wrapped if it is wider than ``room``
    points (a poster's 24 pt in a narrow composed panel ran off it) -- and as many columns
    (filled down, then across) as ``room`` holds."""
    small = spec.small_pt()
    measure = lambda s: _text_width_pt(s, small, family)                   # noqa: E731
    labels = ["\n".join(chart_text.wrap_text(text, measure, room - small * 3.2))
              if measure(text) + small * 2.8 > room else text for text in labels]
    widths = [max(measure(line) for line in text.split("\n")) + small * 2.8
              for text in labels]
    ncol = len(labels)
    while ncol > 1:
        # Split as matplotlib splits them: the first ``extra`` columns one longer. (Chunks
        # of ceil(n / ncol) did not: 5 labels in 4 columns were judged as 2 + 2 + 1 and drawn
        # as 2 + 1 + 1 + 1, a column wider -- the jump-distance key ran off an 89 mm figure.)
        rows, extra = divmod(len(widths), ncol)
        sizes = [rows + 1] * extra + [rows] * (ncol - extra)
        starts = [sum(sizes[:c]) for c in range(ncol)]
        columns = [widths[s:s + n] for s, n in zip(starts, sizes)]
        if sum(max(c) for c in columns if c) + small * (ncol - 1) <= room:
            break
        ncol -= 1
    return labels, ncol


def _legend_above(ax, handles, labels, spec: FigureSpec, family: str, *,
                  share: float = 1.0, title: str | None = None, ncol: int | None = None) -> None:
    """A legend over the plot in as many columns as the figure's width holds -- or the
    ``share`` of it the plot takes (a plot beside another), or ``ncol`` when the caller has
    laid the key out itself -- under ``title`` in bold."""
    small = spec.small_pt()
    room = spec.width_value() / _MM * 72.0 * 0.94 * share
    labels, fitting = _legend_columns(labels, room, spec, family)
    ncol = fitting if ncol is None else ncol
    extra = ({} if not title else
             {"title": title, "title_fontproperties": {"size": small, "weight": "bold"}})
    ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=ncol,
              frameon=False, handletextpad=0.4, columnspacing=1.0, borderaxespad=0.2,
              handlelength=1.4, **extra)


def _qc_grid(ax, spec: FigureSpec) -> None:
    if spec.grid:
        ax.grid(axis="y", color=_page(spec)["grid"], lw=max(MIN_LINE_PT, spec.line_pt() * 0.6),
                zorder=0)
        ax.set_axisbelow(True)


def _band_text(block: dict) -> str:
    """"median ± 3.5·MAD", or the plain words when the payload carries no k."""
    k = block.get("k")
    scale = block.get("scale_estimator") or "MAD"
    return f"median ± {k:g}·{scale}" if k is not None else "the usual range"


def _draw_qc(fig, ax, kind: str, payload: dict, spec: FigureSpec, family: str) -> tuple:
    """One QC Detail panel in print, from the block the screen card draws
    (`qc_view.compute_qc_overview`'s entry): ``(table, more sheets, legend, notes)``."""
    block, condition = payload["block"], str(payload.get("condition", ""))
    colour = payload.get("colour") or QC_DATA
    draw = {"qc_length": _draw_qc_length, "qc_neighbour": _draw_qc_neighbour,
            "qc_outliers": _draw_qc_outliers, "qc_incell": _draw_qc_incell}[kind]
    draw(ax, block, colour, spec, family, payload)
    table, notes = _qc_source(kind, block, condition, spec)
    return table, (), qc_legend(kind, block, condition), notes


def _length_totals(block: dict) -> tuple:
    kept = sum(c for c, k in zip(block["counts"], block["kept"]) if k)
    total = sum(block["counts"])
    return int(round(kept)), int(round(total - kept)), int(round(total))


def _draw_qc_length(ax, block, colour, spec, family, payload) -> None:
    """Track lengths on a log count axis, kept bins in the condition's colour, removed ones
    grey, the length cut dashed -- `TrackLengthChart`."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import FixedLocator, MaxNLocator, NullLocator

    counts, edges, kept = block["counts"], block["edges"], block["kept"]
    if not counts or edges[-1] <= edges[0]:
        raise FigureExportError("this condition has no track lengths to draw")
    line = spec.line_pt()
    lo = 0.5
    hi = 10.0 ** max(1, math.ceil(math.log10(max(max(counts), 1.0) * 1.0001)))
    removed = _pale(spec, _QC_GREY, 0.45)
    shown = [i for i, c in enumerate(counts) if c > 0]
    ax.bar([(edges[i] + edges[i + 1]) / 2.0 for i in shown],
           [counts[i] - lo for i in shown], bottom=lo,
           width=[(edges[i + 1] - edges[i]) * 0.84 for i in shown],
           color=[colour if kept[i] else removed for i in shown], linewidth=0, zorder=2)
    for step in _length_cuts(block):
        ax.axvline(step + 0.5, color=chart_text.FLAG_COLOR, ls="--", lw=max(0.5, line),
                   zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    decades = [10.0 ** e for e in range(0, int(round(math.log10(hi))) + 1)]
    ax.yaxis.set_major_locator(FixedLocator(decades))
    ax.set_yticklabels([f"{v:,.0f}" for v in decades])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlim(edges[0], edges[-1])
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    _qc_grid(ax, spec)
    ax.set_ylabel(_title(spec, "qc_length", "y"))
    ax.set_xlabel(_title(spec, "qc_length", "x"))
    if spec.legend:
        n_kept, n_removed, total = _length_totals(block)
        kept_text, removed_text = "kept", "removed by the filter"
        if spec.numbers and total:
            kept_text += f" ({n_kept:,})"
            removed_text += f" ({n_removed:,}, {100.0 * n_removed / total:.1f} %)"
        # Above the plot, as on the other QC panels: inside it, the longest tracks' bar
        # can stand exactly where a corner legend would go.
        _legend_above(ax, [Patch(color=colour), Patch(color=removed),
                           Line2D([], [], color=chart_text.FLAG_COLOR, ls="--",
                                  lw=max(0.5, line))],
                      [kept_text, removed_text, _length_cut_words(block)],
                      spec, family)


def _length_cuts(block: dict) -> list:
    """The step counts a dashed line follows: the length filter, and a user's cap."""
    cuts = [block["min_steps"]]
    if block.get("max_steps") is not None:
        cuts.append(block["max_steps"])
    return cuts


def _length_cut_words(block: dict) -> str:
    if block.get("max_steps") is None:
        return f"cut: ≤ {block['min_steps']} steps"
    return f"cuts: ≤ {block['min_steps']} and > {block['max_steps']} steps"


def _draw_qc_neighbour(ax, block, colour, spec, family, payload) -> None:
    """Per movie, the share of localizations with a neighbor inside the search radius, the
    usual range as a band, the median as a line, movies unlike the rest orange --
    `NeighbourShareChart`."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    names = _qc_movie_names(block)
    share, flagged = block["share"], block["flagged"]
    if not names:
        raise FigureExportError("no movie to draw")
    line = spec.line_pt()
    low, high, median = block["low"], block["high"], block["median"]
    top = max([*share, high if high is not None else 0.0, 1.0])
    ax.set_ylim(0, top * 1.08)
    handles, labels = [], []
    if low is not None and high is not None:
        band = _pale(spec, colour, 0.16)
        ax.axhspan(max(low, 0.0), high, color=band, lw=0, zorder=0.5)
        handles.append(Patch(color=band))
        labels.append(f"usual range ({_band_text(block)})")
    ax.bar(range(len(share)), share, width=0.8, linewidth=0, zorder=2,
           color=[chart_text.FLAG_COLOR if f else colour for f in flagged])
    if median is not None:
        ax.axhline(median, color=_pale(spec, _page(spec)["ink"], 0.6), lw=max(0.5, line),
                   zorder=3)
        handles.append(Line2D([], [], color=_pale(spec, _page(spec)["ink"], 0.6),
                              lw=max(0.5, line)))
        labels.append(f"median {median:.1f} %" if spec.numbers else "median")
    if any(flagged):
        handles.append(Patch(color=chart_text.FLAG_COLOR))
        labels.append("unlike the other movies")
    _movie_axis(ax, names, spec)
    _qc_grid(ax, spec)
    ax.set_ylabel(_title(spec, "qc_neighbour", "y"))
    if _title(spec, "qc_neighbour", "x"):
        ax.set_xlabel(_title(spec, "qc_neighbour", "x"))
    if spec.legend and handles:
        _legend_above(ax, handles, labels, spec, family)


def _draw_qc_outliers(ax, block, colour, spec, family, payload) -> None:
    """One dot per unit, lowest to highest, the usual range as a band, the median as a line
    and flagged units larger and orange -- `UnitDotsChart` as the Detail page draws it.
    The axis starts at zero, as on screen: a cut axis would exaggerate the spread this panel
    exists to judge."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    values, flagged = block["values"], block["flagged"]
    finite = [v for v in values if v is not None]
    if block.get("target") is None or not finite:
        raise FigureExportError("no per-unit value to draw")
    pt, line = spec.pt(), spec.line_pt()
    n = int(block["n"])
    low, high = block["low"], block["high"]
    hi_v = max(finite) if high is None else max(max(finite), high)
    lo_v = min(0.0, min(finite))
    ax.set_ylim(lo_v, hi_v + 0.06 * ((hi_v - lo_v) or 1.0))
    ax.set_xlim(-0.8, n - 0.2)
    handles, labels = [], []
    if low is not None and high is not None:
        band = _pale(spec, _QC_GREY, 0.16)
        ax.axhspan(low, high, color=band, lw=0, zorder=0.5)
        handles.append(Patch(color=band))
        labels.append(f"usual range ({_band_text(block)})")
    rank = {i: r for r, i in enumerate(block["order_sorted"])}
    pitch = spec.width_value() / _MM * 72.0 * 0.8 / max(1, n)
    size = min(0.62 * pt, max(1.2, pitch * 0.9))
    plain = [i for i in range(n) if values[i] is not None and not flagged[i]]
    flags = [i for i in range(n) if values[i] is not None and flagged[i]]
    ax.scatter([rank[i] for i in plain], [values[i] for i in plain], s=size ** 2, c=colour,
               linewidths=0, alpha=1.0 if size >= 2.5 else 0.8, zorder=3)
    if flags:
        big = max(size * 1.5, 0.55 * pt)
        ax.scatter([rank[i] for i in flags], [values[i] for i in flags], s=big ** 2,
                   c=chart_text.FLAG_COLOR, edgecolors=_pale(spec, _page(spec)["ink"], 0.6),
                   linewidths=0.4, zorder=4)
        handles.append(Line2D([], [], marker="o", linestyle="none", markersize=0.55 * pt,
                              markerfacecolor=chart_text.FLAG_COLOR,
                              markeredgecolor=_pale(spec, _page(spec)["ink"], 0.6),
                              markeredgewidth=0.4))
        labels.append(f"flagged ({len(flags)} of {n})" if spec.numbers else "flagged")
    if block.get("has_spread"):
        ink = _pale(spec, _page(spec)["ink"], 0.8)
        ax.axhline(block["median"], color=ink, lw=max(0.5, line), zorder=3.5)
        handles.append(Line2D([], [], color=ink, lw=max(0.5, line)))
        labels.append(f"median {block['median']:.3g}" if spec.numbers else "median")
    ax.set_xticks([])
    _qc_grid(ax, spec)
    ax.set_ylabel(_title(spec, "qc_outliers", "y", payload))
    ax.set_xlabel(_title(spec, "qc_outliers", "x", payload))
    if spec.legend and handles:
        _legend_above(ax, handles, labels, spec, family)


def _draw_qc_incell(ax, block, colour, spec, family, payload) -> None:
    """Tracks found per movie, stacked inside / outside a cell outline, a movie unlike its
    condition edged orange, and the share inside on its own 0-100 % axis at the right --
    `InCellShareChart`."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    names = _qc_movie_names(block)
    if not names:
        raise FigureExportError("no movie to draw")
    pt, line = spec.pt(), spec.line_pt()
    found, inside, share, flagged = (block["found"], block["inside"], block["share"],
                                     block["flagged"])
    xs = list(range(len(names)))
    outside_ink = _pale(spec, _QC_GREY, 0.45)
    ax.bar(xs, inside, width=0.8, color=colour, linewidth=0, zorder=2)
    ax.bar(xs, [f - i for f, i in zip(found, inside)], bottom=inside, width=0.8,
           color=outside_ink, linewidth=0, zorder=2)
    marked = [x for x in xs if flagged[x]]
    if marked:
        ax.bar(marked, [max(found[x], 1e-9) for x in marked], width=0.8, fill=False,
               edgecolor=chart_text.FLAG_COLOR, linewidth=max(0.6, line * 1.2), zorder=3)
    ax.set_ylim(0, max(max(found), 1.0) * 1.08)
    _movie_axis(ax, names, spec)
    _qc_grid(ax, spec)
    ax.set_ylabel(_title(spec, "qc_incell", "y"))
    if _title(spec, "qc_incell", "x"):
        ax.set_xlabel(_title(spec, "qc_incell", "x"))
    right = ax.twinx()
    known = [x for x in xs if share[x] is not None]
    ink = _pale(spec, _page(spec)["ink"], 0.8)
    right.plot(known, [share[x] for x in known], "-o", color=ink, lw=max(0.5, line),
               ms=0.32 * pt, zorder=4)
    right.set_ylim(0, 105)
    right.set_yticks([0, 50, 100])
    right.spines["right"].set_visible(True)
    right.set_ylabel(chart_text.QC_INSIDE_LABEL)
    if spec.legend:
        handles = [Patch(color=colour), Patch(color=outside_ink)]
        labels = ["inside a cell", "outside every outline"]
        if marked:
            handles.append(Patch(facecolor="none", edgecolor=chart_text.FLAG_COLOR,
                                 linewidth=max(0.6, line * 1.2)))
            labels.append("unlike its condition")
        handles.append(Line2D([], [], color=ink, marker="o", ms=0.32 * pt, lw=max(0.5, line)))
        total, kept = sum(found), sum(inside)
        labels.append(f"% inside (all movies: {100.0 * kept / total:.0f} %)"
                      if spec.numbers and total else "% inside")
        _legend_above(ax, handles, labels, spec, family)


def _qc_source(kind: str, block: dict, condition: str, spec: FigureSpec) -> tuple:
    """``((columns, rows), header notes)``: what one QC panel drew, as numbers."""
    caption = str(block.get("caption", ""))
    if kind == "qc_length":
        edges = block["edges"]
        rows = [[condition, (edges[i] + edges[i + 1]) / 2.0, int(round(c)),
                 bool(block["kept"][i])]
                for i, c in enumerate(block["counts"])]
        top = block.get("max_steps")
        rule = (f"A track with more than {block['min_steps']} steps is kept; shorter ones "
                "are removed before any analysis." if top is None else
                f"A track with {block['min_steps'] + 1} to {top} steps is kept; shorter ones "
                f"are removed before any analysis, and longer ones are left out by a cap "
                f"you set.")
        return ((["Condition", "Steps per track", "Tracks", "Kept by the filter"], rows),
                (rule, caption))
    names = [str(n) for n in block.get("names", ())]
    short = _qc_movie_names(block) if names else []
    if kind == "qc_neighbour":
        rows = [[condition, names[i], spec.shown_name(short[i]), float(block["share"][i]),
                 int(block["n_localizations"][i]), float(block["typical"][i]),
                 float(block["busiest"][i]), bool(block["flagged"][i])]
                for i in range(len(names))]
        band = ("" if block["low"] is None or block["high"] is None else
                f" = [{max(block['low'], 0.0):.1f}, {block['high']:.1f}] %")
        return ((["Condition", "Movie", "Shown as", "With a neighbor (%)", "Localizations",
                  "Typical per frame", "Busiest per frame", "Flagged"], rows),
                (f"With a neighbor: the share of a movie's localizations with another "
                 f"localization within {block['radius_px']:g} px in the same frame, %.",
                 f"Flagged: outside {_band_text(block)} over this condition's movies"
                 f"{band}.", caption))
    if kind == "qc_outliers":
        unit = str(block["unit"])
        rank = {i: r + 1 for r, i in enumerate(block["order_sorted"])}
        quantity = str(block.get("axis_label") or "value")
        columns = ["Condition", unit_word(unit).capitalize()]
        if block.get("movies"):
            columns.append("Movie")
        columns += [quantity[:1].upper() + quantity[1:], "Flagged", "Position (sorted)"]
        rows = []
        for i in range(int(block["n"])):
            row = [condition, names[i]]
            if block.get("movies"):
                row.append(block["movies"][i])
            row += [block["values"][i], bool(block["flagged"][i]), rank[i]]
            rows.append(row)
        band = ("" if block["low"] is None or block["high"] is None else
                f" = [{block['low']:.4g}, {block['high']:.4g}]")
        return (columns, rows), (f"Flagged: outside {_band_text(block)}{band}. A flag "
                                 "means this unit deserves a look; nothing is removed.", caption)
    rows = [[condition, names[i], spec.shown_name(short[i]), int(round(block["found"][i])),
             int(round(block["inside"][i])), int(round(block["found"][i] - block["inside"][i])),
             block["share"][i], bool(block["flagged"][i]),
             (block.get("reasons") or [""] * len(names))[i]]
            for i in range(len(names))]
    return ((["Condition", "Movie", "Shown as", "Tracks found", "Inside a cell outline",
              "Outside every outline", "Inside (%)", "Flagged", "Why flagged"], rows),
            ("Inside a cell outline: every localization of the track lies inside one "
             "segmented cell. Only those tracks enter the analysis.", caption))


def qc_legend(kind: str, block: dict, condition: str) -> str:
    """A drafted legend for one QC panel."""
    of = f" ({condition})" if condition else ""
    if kind == "qc_length":
        n_kept, n_removed, total = _length_totals(block)
        pct = f" ({100.0 * n_removed / total:.1f} %)" if total else ""
        top = block.get("max_steps")
        if top is None:
            return (f"Track lengths{of}, in steps per track, on a log count axis. Tracks of "
                    f"{block['min_steps']} steps or fewer (grey, left of the dashed line) are "
                    f"removed before analysis: {n_removed:,} of {total:,} tracks{pct}; "
                    f"{n_kept:,} are kept. Long tracks tend to be slow.")
        return (f"Track lengths{of}, in steps per track, on a log count axis. Tracks of "
                f"{block['min_steps']} steps or fewer, and tracks longer than {top} steps (a "
                f"cap you set), are grey and left out of the analysis: "
                f"{n_removed:,} of {total:,} tracks{pct}; {n_kept:,} are kept. Long tracks "
                "tend to be slow, so the cap lowers the slow fraction.")
    if kind == "qc_neighbour":
        n_flag = sum(block["flagged"])
        median = ("" if block["median"] is None else
                  f" The line is the median ({block['median']:.1f} %).")
        return (f"Share of each movie's localizations with another localization within "
                f"{block['radius_px']:g} px in the same frame{of}; the higher it is, the "
                f"more often the tracker could link the wrong molecules. The band is the "
                f"usual range across this condition's movies ({_band_text(block)}).{median} "
                f"Orange: movies outside it ({n_flag} of {len(block['flagged'])}); a condition crowded "
                "throughout shows none. Only localizations that survived tracking and the cell filter "
                "are counted, so the true share is at least this high.")
    if kind == "qc_outliers":
        unit, quantity = block["unit"], block.get("quantity", "value")
        return (f"Per-{unit_word(unit)} {quantity}{of}, one dot per {unit_word(unit)}, sorted from lowest to "
                f"highest. The band is the usual range ({_band_text(block)}) and the line "
                f"the median; orange dots are flagged ({block['n_flagged']} of "
                f"{block['n']}). A flag means that {unit_word(unit)} deserves a look; "
                "nothing is removed.")
    n_flag = sum(block["flagged"])
    total, kept = sum(block["found"]), sum(block["inside"])
    share = f" ({100.0 * kept / total:.0f} %)" if total else ""
    return (f"Tracks found per movie{of}, split into those entirely inside a segmented cell outline "
            f"(coloured) and those outside every outline (grey); the line is the share inside (right axis). "
            f"{kept:,.0f} of {total:,.0f} tracks lie inside{share}; only those enter the analysis. An orange "
            f"edge marks a movie unlike its condition ({n_flag} of {len(block['flagged'])}). A low share can "
            "come from under-segmentation, an extracellular label or a coordinate mismatch.")


# -- the Spatial Map ----------------------------------------------------------------- #
_PT_MM = 25.4 / 72.0
#: A bin the run masked: UNDEFINED, a different statement from "nothing here" (the ramp's
#: low end). Darker than the screen's grey so it survives print against a light ramp end.
_MASKED_GREY = "#d4d4d8"
#: A panel too thin to map (sparse rule C): greyed whole and labelled, never confetti.
_SPARSE_GREY = "#e4e4e7"
_OUTLINE_GREY = "#a1a1aa"
_MUTED_INK = "#52525b"


def _spatial_rows(payload: dict) -> list:
    """The grid's rows in order: every condition (or day) with maps or with an error."""
    return [r for r in payload.get("rows", ())
            if r in payload.get("payloads", {}) or r in payload.get("errors", {})]


def _row_label(payload: dict, row) -> str:
    return str(payload.get("labels", {}).get(row, row))


def _spatial_ranges(payload: dict):
    """A manual split's ranges (`spatial_map_view.SPLITS`), from the export payload or its first condition's map; None
    for a fitted split."""
    if payload.get("ranges"):
        return payload["ranges"]
    for one in (payload.get("payloads") or {}).values():
        if one and one.get("ranges"):
            return one["ranges"]
    return None


def _spatial_counts(payload: dict) -> list:
    """Every count the grid's rows were fitted at (each condition its own since 1 Nov 2026), else the one."""
    counts = payload.get("counts") or [p.get("n_pop") for p in (payload.get("payloads") or {}).values()
                                       if p and p.get("n_pop")]
    return sorted({int(n) for n in counts}) or [int(payload.get("n_pop") or 2)]


def _spatial_kinds(payload: dict) -> list:
    from .spatial_map_view import kinds_for_counts

    shown = payload.get("shown") or {}
    return [k for k in kinds_for_counts(payload["method"], _spatial_counts(payload), _spatial_ranges(payload))
            if shown.get(k, True)]


def _spatial_present(payload: dict) -> dict:
    return {r: payload["payloads"][r] for r in _spatial_rows(payload)
            if r in payload.get("payloads", {})}


def _sla_tracks(payload: dict) -> bool:
    """The grid is Confinement (SLA) in the Tracks view: each set's tracks are DRAWN, as
    lines in v2's colours, not its density (any other method in that view draws its bins,
    as the screen says it cannot draw tracks)."""
    return payload.get("rendering") == "tracks" and payload.get("method") == "sla"


def _spatial_unit(payload: dict) -> str:
    """The colour bar's title, as the screen's bar names its unit ("" in the SLA Tracks
    view, which has no bar)."""
    from .spatial_map_view import PER_PANEL_UNIT_TITLE, has_panel_units, matrix_limits

    if _sla_tracks(payload):
        return ""
    present = _spatial_present(payload) if payload.get("payloads") else {}
    if payload.get("rendering") == "neff":
        return f"n_eff ({_neff_unit(next(iter(present.values()), {}))}), log scale"
    if has_panel_units(present.values()):
        # SDA: each bar names its own unit (`_draw_spatial`); a typed title replaces all.
        return PER_PANEL_UNIT_TITLE
    _limits, field = matrix_limits(present, payload.get("scale", "population"))
    if field == "density":
        return "density / bin"
    base = next(iter(present.values()), {})
    return f"{base.get('weight_unit', 'weighted jumps')} / bin"


def _spatial_bar_layout(payload: dict) -> str:
    """Where the grid's colour bars stand, as on screen (`spatial_map_view.bar_layout`): "columns", "rows" (beside
    each row), "maps" (under each map), or "none" where the tracks are drawn instead of a density."""
    from .spatial_map_view import bar_layout, matrix_limits, neff_tops

    if _sla_tracks(payload):
        return "none"
    present = _spatial_present(payload)
    scale, rendering = payload.get("scale", "population"), payload.get("rendering", "bins")
    limits, _field = matrix_limits(present, scale)
    units = {}
    for kind in _spatial_kinds(payload):
        first = next((q["panels"][kind] for q in present.values() if q.get("panels", {}).get(kind)), None)
        if first and first.get("quantity") in ("value", "probability"):
            units[kind] = first.get("unit")
    return bar_layout(scale, neff_tops(present, scale) if rendering == "neff" else limits,
                      {k: units.get(k) for k in _spatial_kinds(payload)} if units else None)


def _neff_unit(base: dict) -> str:
    """The contributor the n_eff counts, as the screen's bar names it (`_neff_bar`)."""
    return str(base.get("neff_unit") or "tracks") if base.get("neff_min") is not None else "tracks"


def _ramp_cmap(ramp, bad: str = _MASKED_GREY):
    """The screen ramp's own colours (`spatial_map_chart.ramp_for`), interpolated: the same
    picture whatever the ramp, a user's custom ones included."""
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(
        "ramp", ["#" + str(h).lstrip("#") for h in ramp], N=256)
    return cmap.with_extremes(bad=bad)


def _capsule(length: float, width: float, transform, **kw):
    """`spatialsqd.capsule_outline`'s shape: two semicircular caps joined by straight walls."""
    from matplotlib.patches import FancyBboxPatch

    return FancyBboxPatch((0.0, 0.0), length, width, transform=transform,
                          boxstyle=f"round,pad=0,rounding_size={width / 2.0}", **kw)


def _in_capsule(x: float, y: float, length: float, width: float) -> bool:
    r = width / 2.0
    cx = min(max(x, r), length - r)
    return (x - cx) ** 2 + (y - r) ** 2 <= r * r + 1e-12


def _spatial_map_ends(payload: dict, row, kind) -> tuple | None:
    """The two numbers under one map's own bar: its colour range (in its own unit's way, `bar_number`), or the
    n_eff view's log scale up to the map's top."""
    from .spatial_map_view import NEFF_SCALE_TOP, matrix_limits, neff_tops

    present = _spatial_present(payload)
    scale = payload.get("scale", "population")
    if payload.get("rendering") == "neff":
        top = neff_tops(present, scale).get(row, {}).get(kind, NEFF_SCALE_TOP)
        return "1", f"{top:g}+"
    limits, _field = matrix_limits(present, scale)
    got = limits.get(row, {}).get(kind)
    if not got:
        return None
    return float(got[0]), float(got[1])          # numbers: the drawing fits them to the bar


def _spatial_row_ends(payload: dict, row=None):
    """The two numbers beside a row's bar (its maps share one range): for ``row``, or, without one, every row's
    (for the width the bars need)."""
    rows = [row] if row is not None else _spatial_rows(payload)
    out = []
    for r in rows:
        ends = next((e for k in _spatial_kinds(payload)
                     if (e := _spatial_map_ends(payload, r, k)) is not None), None)
        if ends is not None:
            out.append(ends)
    return out


def _spatial_cells(payload: dict, spec: FigureSpec) -> dict:
    """Each map's line under it: its figures (numbers on). A map on a scale of its own no longer prints its range
    here: its bar, beside its row or under it (`_spatial_bar_layout`), carries the numbers, as on screen."""
    from .spatial_map_view import matrix_figures_text

    present = _spatial_present(payload)
    out = {}
    for row in _spatial_rows(payload):
        p = present.get(row)
        for kind in _spatial_kinds(payload):
            panel = None if p is None else p["panels"].get(kind)
            out[(row, kind)] = (matrix_figures_text(p, panel, None)
                                if spec.numbers and panel is not None else "")
    return out


#: The kinds the Export dialog offers a Map scale for (`FigureSpec.map_scale`).
MAP_SCALE_KINDS = ("spatial_maps",)
#: The cells' size on paper when no map scale is typed, mm per µm of cell, at a journal column of 89 mm (Nature's): a
#: 3 µm cell 18 mm long, so a grid of three columns with its names and colour bars fits one column
#: (`FigureSpec.map_scale`). A preset's own standard is in proportion to its column (`map_scale_standard`).
MAP_SCALE_STANDARD = 6.0


def map_scale_standard(spec: FigureSpec) -> float:
    """The map scale (mm per µm) a figure draws when none is typed: `MAP_SCALE_STANDARD` at an 89 mm column, in
    proportion to the preset's own one column (Cell 5.73, slides 10.11, poster 16.85), so the cells keep their share
    of a column whatever the medium."""
    return round(MAP_SCALE_STANDARD * float(spec.preset_values["single_mm"]) / 89.0, 2)


def _spatial_layout(payload: dict, spec: FigureSpec, family: str, top: float = 1.0) -> dict:
    """Every size in the grid, in mm, in the width chosen: a name column, one column per population, each capsule at
    the CELL's proportions and at ONE SCALE, the map scale's mm per µm (`FigureSpec.map_scale`), never the room the
    names or the bars leave; the grid centred in the width. Columns the width cannot hold at the scale are drawn as
    large as it holds, each row as tall as at the scale (``"row_pad"`` under it), so the figure keeps its size."""
    from .spatial_map_view import kinds_for_counts, panel_titles_for_counts

    rows, kinds = _spatial_rows(payload), _spatial_kinds(payload)
    present = _spatial_present(payload)
    if not rows or not kinds or not present:
        raise FigureExportError("no map to draw yet: run the maps first")
    base = next(iter(present.values()))
    length, width = float(base["cell_length_um"]), float(base["cell_width_um"])
    pt, small = spec.pt(), spec.small_pt()
    total = spec.width_value()
    measure = lambda s: _text_width_pt(s, pt, family)                      # noqa: E731
    measure_small = lambda s: _text_width_pt(s, small, family)             # noqa: E731
    names = [spec.shown_name(_row_label(payload, r)) for r in rows]
    # Ink extents run a little under the printed advance: measured with room to spare.
    label_w = (min(max(measure(n) for n in names) * _PT_MM * 1.08 + 2.5, 0.28 * total)
               if spec.headings else 0.5)
    name_lines = [chart_text.wrap_text(n, measure, (label_w - 2.5) / _PT_MM / 1.08)
                  for n in names]
    gap, right = 2.5, 1.0
    placement = _spatial_bar_layout(payload) if spec.legend else "none"
    # A BAR BESIDE EACH ROW (per condition) takes a column on the right: the strip and its numbers.
    row_bar_w = 0.0
    if placement == "rows":
        from .spatial_map_view import bar_number

        ends = [t if isinstance(t, str) else bar_number(t)
                for pair in _spatial_row_ends(payload) for t in pair]
        row_bar_w = 1.2 + 1.8 + 0.8 + max(measure_small(t) for t in ends) * _PT_MM * 1.05
    # ONE SCALE FOR EVERY FIGURE (`FigureSpec.map_scale`): the capsule at the map scale's mm per µm whatever the names
    # and the bars take, the grid centred in the width; columns the width cannot hold are drawn as large as it holds
    scale_mm = float(spec.map_scale) if spec.map_scale is not None else map_scale_standard(spec)
    room = (total - label_w - right - row_bar_w - gap * (len(kinds) - 1)) / len(kinds)
    # the room the TEXT can never take from the cells: the names' column at its widest and the bars' numbers at a
    # standard width, so whether and how far the cells shrink depends on the width, the columns and the settings,
    # never on the names (two figures of one width and one count of columns shrink alike)
    names_most = 0.28 * total if spec.headings else 0.5
    bar_most = max(row_bar_w, 1.2 + 1.8 + 0.8 + measure_small("0.0000") * _PT_MM * 1.05) if placement == "rows" else 0.0
    room_std = min(room, (total - names_most - right - bar_most - gap * (len(kinds) - 1)) / len(kinds))
    at_scale = scale_mm * length
    cw = min(at_scale, room_std)
    shrunk = at_scale > room_std + 1e-9
    left = max(0.0, (room - cw) * len(kinds) / 2.0)          # the grid centred in the width chosen
    title = _title_lines(spec, family, total - 1.0)
    title_top = top
    top += len(title) * spec.pt() * 1.3 * _PT_MM + (0.8 if title else 0.0)
    titles = dict(zip(kinds_for_counts(payload["method"], _spatial_counts(payload), _spatial_ranges(payload)),
                      panel_titles_for_counts(payload["method"], _spatial_counts(payload), _spatial_ranges(payload))))
    title_lines = [chart_text.wrap_text(titles[k], measure, cw / _PT_MM) for k in kinds]
    head = (max(len(t) for t in title_lines) * pt * 1.25 * _PT_MM + 1.2) if spec.headings else 0.5
    cells = _spatial_cells(payload, spec)
    errors = payload.get("errors", {})
    # THE SCALE BAR'S ROOM grows with its text: its length is written beside the bar, centred on it 1.2 mm down, so
    # half a line of text and a millimetre must fit before the colour bars (found by the overnight tester, 4 Nov
    # 2026: on a slide the "µ" of "1 µm" touched the colour bar's frame under it)
    row_gap = 2.0
    scalebar = max(4.5, 1.2 + 0.6 * small * _PT_MM + 1.0) if spec.scale_bar else 1.2
    # No colour bar where the tracks are drawn (the screen hides its bars there too). Under the columns, a bar
    # and its title; beside the rows or under the maps, the bottom keeps one line naming the scale and its unit.
    bar = {"none": 0.0, "columns": 1.8 + small * 2.7 * _PT_MM}.get(placement, small * 1.6 * _PT_MM)
    # A BAR UNDER EACH MAP (per map): the strip and its two numbers, between the capsule and its line.
    map_bar = 1.8 + small * 1.5 * _PT_MM + 0.6 if placement == "maps" else 0.0
    fixed = top + head + scalebar + bar + 1.5

    def wrapped(column_mm):
        lines = {key: (chart_text.wrap_text(text, measure_small, column_mm / _PT_MM)
                       if text else []) for key, text in cells.items()}
        n = max([len(v) for v in lines.values()] + [0])
        return lines, n * small * 1.2 * _PT_MM + (0.6 if n else 0.0)

    cell_lines, fig_line = wrapped(cw)
    ch = cw / (length / width)
    # EACH ROW AS TALL AS AT THE SCALE: capsules drawn smaller to fit the width leave the rest of their row's height
    # under it, so the figure is as tall as a figure whose cells fit ("figure height preserve as well")
    row_pad = 0.0
    if shrunk:
        _lines, line_at_scale = wrapped(at_scale)
        row_pad = max(0.0, (at_scale * width / length + line_at_scale) - (ch + fig_line))
    row_h = ch + map_bar + fig_line + row_pad + row_gap
    findings = ()
    if spec.height_mm is not None:
        # A set height never shrinks the capsules (they are on one scale so figures match; until 5 Nov 2026 it shrank
        # them to half): it is checked, and what does not fit is said.
        needed = fixed + len(rows) * row_h
        if needed > spec.height_mm + 0.5:
            findings = (f"the maps need about {math.ceil(needed)} mm at this width; at "
                        f"{spec.height_mm:g} mm they do not fit (auto height fits them)",)
        height = float(spec.height_mm)
    else:
        natural = fixed + len(rows) * row_h
        height = float(min(600.0, max(40.0, round(natural, 1))))
    # the scale drawn, on the check line
    notes = ((f"Cells drawn at {cw / length:.2f} mm per µm. At {scale_mm:g} the columns do not fit this width, so "
              f"the cells are drawn as large as it allows and each row keeps its height") if shrunk else
             f"Cells drawn at {scale_mm:.2f} mm per µm"
             + (" (the default; type a Map scale to change it)" if spec.map_scale is None else ""),)
    notes = (notes[0] + ".",)
    return {"rows": rows, "kinds": kinds, "length": length, "width": width,
            "label_w": label_w + left, "names": name_lines, "titles": title_lines, "cw": cw,
            "ch": ch, "gap": gap, "head": head, "fig_line": fig_line, "row_gap": row_gap,
            "scalebar": scalebar, "bar": bar, "height": height, "cells": cell_lines,
            "placement": placement, "row_bar_w": row_bar_w, "map_bar": map_bar,
            "errors": errors, "findings": findings, "notes": notes, "top": top, "title": title,
            "title_top": title_top, "row_pad": row_pad}


def _draw_spatial(fig, payload: dict, spec: FigureSpec, family: str, lay: dict) -> tuple:
    """The Spatial Map grid in print, from the grid's own payloads
    (`SpatialMapMatrix.export_payload`): rows of conditions (or one condition's days),
    columns of populations, each map clipped to the capsule, masked bins grey, a panel too
    thin to map greyed and named, the colour scale decided over the whole grid BEFORE
    drawing (`spatial_map_view.matrix_limits`, the screen's), a scale bar and the bars.
    Confinement (SLA) in the Tracks view draws each set's TRACKS instead, as lines in v2's
    colours clipped to the capsule (`_draw_sla_tracks`), with no colour bar."""
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    from .spatial_map_view import (
        NEFF_SCALE_TOP,
        has_panel_units,
        matrix_limits,
        neff_display_panel,
        empty_cell_lines,
        neff_tops,
        paint_field,
        smooth_display,
    )

    from matplotlib.figure import SubFigure

    if not isinstance(fig, SubFigure):
        fig.set_layout_engine("none")       # a composed panel's root keeps the composer's
    width_mm, height_mm = spec.size_mm()
    pt, small, line = spec.pt(), spec.small_pt(), spec.line_pt()

    def rect(x, y_top, w, h):
        return [x / width_mm, 1.0 - (y_top + h) / height_mm, w / width_mm, h / height_mm]

    def at(x, y):
        return x / width_mm, 1.0 - y / height_mm

    for k, title_line in enumerate(lay.get("title", ())):
        fig.text(*at(width_mm / 2.0, lay["title_top"] + (k + 0.85) * pt * 1.3 * _PT_MM),
                 title_line, ha="center", va="baseline", fontsize=pt, fontweight="bold",
                 color=_page(spec)["ink"])
    present = _spatial_present(payload)
    scale, rendering = payload.get("scale", "population"), payload.get("rendering", "bins")
    limits, field = matrix_limits(present, scale)
    tops = neff_tops(present, scale) if rendering == "neff" else {}
    page = _page(spec)
    cmap = _ramp_cmap(payload["ramp"], page["masked"])
    length, cell_w = lay["length"], lay["width"]
    cw, ch, gap = lay["cw"], lay["ch"], lay["gap"]
    x_cols = [lay["label_w"] + c * (cw + gap) for c in range(len(lay["kinds"]))]
    edge = max(MIN_LINE_PT * 2, line * 0.8)
    tracks_view = _sla_tracks(payload)

    y = lay["top"] + lay["head"]
    map_bars, row_bars = [], []           # drawn after the maps, where the scale puts them
    for c, lines in enumerate(lay["titles"] if spec.headings else ()):
        fig.text(*at(x_cols[c] + cw / 2.0, y - 0.8), "\n".join(lines), ha="center",
                 va="bottom", fontsize=pt, linespacing=1.1)
    for r, row in enumerate(lay["rows"]):
        if spec.headings:
            fig.text(*at(lay["label_w"] - 1.5, y + ch / 2.0), "\n".join(lay["names"][r]),
                     ha="right", va="center", fontsize=pt, linespacing=1.1)
        p = present.get(row)
        if p is None:
            reason = str(lay["errors"].get(row, "no map"))
            measure = lambda s: _text_width_pt(s, small, family)            # noqa: E731
            fig.text(*at(x_cols[0], y + ch / 2.0), "\n".join(chart_text.wrap_text(
                reason, measure, (x_cols[-1] + cw - x_cols[0]) / _PT_MM)[:3]),
                ha="left", va="center", fontsize=small, color=chart_text.FLAG_COLOR)
        for c, kind in enumerate(lay["kinds"]):
            # NOT DRAWN AT THIS ROW'S COUNT (each condition its own since 1 Nov 2026): no outline -- an empty
            # capsule reads as a map with nothing in it -- but WHY, where the map would stand (3 Nov 2026: "or instead
            # of empty, what else could we show?"): "One population: no split", and with Numbers on, a Best row's BIC
            # margin (`spatial_map_view.empty_cell_lines`)
            if p is None:
                continue
            if kind not in (p.get("panels") or {}):
                note = empty_cell_lines(p, kind, margin=spec.numbers)
                measure = lambda s: _text_width_pt(s, small, family)            # noqa: E731
                lines = [w for one in note for w in chart_text.wrap_text(one, measure, cw / _PT_MM)]
                fig.text(*at(x_cols[c] + cw / 2.0, y + ch / 2.0), "\n".join(lines), ha="center", va="center",
                         fontsize=small, color=page["muted"], linespacing=1.15)
                continue
            ax = fig.add_axes(rect(x_cols[c], y, cw, ch))
            ax.set_xlim(0, length)
            ax.set_ylim(0, cell_w)
            ax.axis("off")
            clip = _capsule(length, cell_w, ax.transData, facecolor="none", edgecolor="none")
            ax.add_patch(clip)
            panel = p["panels"].get(kind)
            if panel is not None and tracks_view:
                # THE TRACKS VIEW (Confinement (SLA)): every track of the set as a line in
                # v2's colour, unfolded, clipped to the analytic capsule -- no grid, no
                # masking, no sparse grey (all three belong to the density), as on screen.
                _draw_sla_tracks(ax, panel, length, cell_w, clip, spec)
            elif panel is not None and panel.get("sparse") and rendering != "neff":
                ax.add_patch(_capsule(length, cell_w, ax.transData, facecolor=page["sparse"],
                                      edgecolor="none"))
                ax.text(length / 2.0, cell_w / 2.0,
                        f"n = {panel['n_jumps']:,.0f}\ntoo sparse to map", ha="center",
                        va="center", fontsize=small, color=page["muted"], linespacing=1.1)
            elif panel is not None:
                outline = None
                if rendering == "neff":
                    shown = neff_display_panel(panel, tops.get(row, {}).get(kind, NEFF_SCALE_TOP))
                    if shown is None:
                        raise FigureExportError("this run predates n_eff: run the maps again")
                    values, masked, lo, hi = shown["neff_pos"], shown["masked"], 0.0, 1.0
                    outline = shown.get("outline")
                else:
                    # A map in its own unit (SDA) paints its own grid: a mean is never a density.
                    values, masked = panel[paint_field(panel, field)], panel["masked"]
                    lo, hi = limits.get(row, {}).get(kind) or (0.0, 1.0)
                grid = (smooth_display(values, masked) if rendering == "smooth" else
                        [[None if m else v for v, m in zip(vr, mr)]
                         for vr, mr in zip(values, masked)])
                arr = np.array([[np.nan if v is None else float(v) for v in gr] for gr in grid])
                image = ax.imshow(np.ma.masked_invalid(arr), origin="lower",
                                  extent=(0, length, 0, cell_w), cmap=cmap, vmin=lo,
                                  vmax=hi if hi > lo else lo + 1e-12,
                                  interpolation="nearest", aspect="auto")
                image.set_clip_path(clip)
                if outline:
                    n_short, n_long = len(outline), len(outline[0])
                    bw, bh = length / n_long, cell_w / n_short
                    for iy, orow in enumerate(outline):
                        for ix, on in enumerate(orow):
                            if on:
                                ring = Rectangle((ix * bw, iy * bh), bw, bh, fill=False,
                                                 edgecolor=_page(spec)["ink"], lw=edge)
                                ring.set_clip_path(clip)
                                ax.add_patch(ring)
            # The Tracks view outlines the model cell in ink, as v2's confined map did
            # (`confTracks.m:121`) and as the screen does; a map keeps the grey edge.
            ax.add_patch(_capsule(length, cell_w, ax.transData, facecolor="none",
                                  edgecolor=page["ink"] if tracks_view else _OUTLINE_GREY,
                                  lw=edge * 1.6 if tracks_view else edge))
            if lay["placement"] == "maps" and panel is not None:
                ends = _spatial_map_ends(payload, row, kind)
                if ends:
                    map_bars.append((x_cols[c], y + ch + 0.6, cw, ends, kind))
            lines = lay["cells"].get((row, kind)) or []
            if lines:
                fig.text(*at(x_cols[c], y + ch + lay["map_bar"] + 0.6), "\n".join(lines), ha="left",
                         va="top", fontsize=small, color=page["muted"], linespacing=1.15)
        if lay["placement"] == "rows" and p is not None:
            ends = _spatial_row_ends(payload, row)
            if ends:
                row_bars.append((x_cols[-1] + cw + 1.2, y, ch, ends[0]))
        y += ch + lay["map_bar"] + lay["fig_line"] + lay.get("row_pad", 0.0) + lay["row_gap"]

    # THE SCALE BAR: the drawn capsule is the model cell, so 1 um on it is 1 um of that frame. Its length typed
    # in the dialog, else the maps' own; off, it is not drawn and its room is gone (`_spatial_layout`).
    if spec.scale_bar:
        bar_um = _spatial_bar_um(payload, spec, length)
        bar_mm = cw * bar_um / length
        yb = y + 1.2
        fig.add_artist(Line2D([at(x_cols[0], yb)[0], at(x_cols[0] + bar_mm, yb)[0]],
                              [at(0, yb)[1]] * 2, transform=fig.transSubfigure,
                              color=_page(spec)["ink"], lw=max(0.75, line * 1.5),
                              solid_capstyle="butt"))
        fig.text(*at(x_cols[0] + bar_mm + 1.0, yb), f"{bar_um:g} µm", ha="left", va="center",
                 fontsize=small, color=_page(spec)["ink"])

    # THE COLOUR BARS: as many as there are scales in play, as on screen.
    y_bar = y + lay["scalebar"]
    title = _title(spec, "spatial_maps", "x", payload)
    unit = _neff_unit(next(iter(present.values())))
    own = scale in ("row", "panel")
    # SDA: every map carries its own unit, so each bar names its own (on screen too) and
    # no single bar spans the grid; a title typed in the Export dialog replaces them all.
    per_unit = rendering != "neff" and has_panel_units(present.values())
    kind_units, kind_quantity = {}, {}
    for kind in lay["kinds"]:
        first = next((p["panels"][kind] for p in present.values()
                      if p.get("panels", {}).get(kind)), None)
        kind_units[kind] = (first or {}).get("unit")
        kind_quantity[kind] = (first or {}).get("quantity")

    def bar_title(kind):
        if spec.x_title is not None or not per_unit:
            return title
        return kind_units.get(kind) or ""

    def bar_ends(ends, w, kind=None):
        """A bar's two numbers (`chart_text.scale_ends`, THE formatter of every colour bar's ends, the screen's too),
        as they are when they stand side by side under a bar ``w`` mm wide (four maps on a single-column slide leave
        about 31 mm a bar), otherwise shorter -- a probability as a percentage, two figures, one. Words ("low", "1")
        stay as they are."""
        if isinstance(ends[0], str):
            return tuple(ends)

        def fits(texts):
            # 3 mm between the two, so they read as two numbers, not one.
            return (_text_width_pt(texts[0], small, family)
                    + _text_width_pt(texts[1], small, family)) * _PT_MM + 3.0 <= w

        return chart_text.scale_ends(ends[0], ends[1], fits, percent=kind_quantity.get(kind) == "probability")

    def colour_bar(x, w, ends, kind=None, titled=True, from_left=False):
        ends, title = bar_ends(ends, w, kind), (bar_title(kind) if titled else "")
        bax = fig.add_axes(rect(x, y_bar, w, 1.8))
        bax._smtracker_colour_bar = True               # `check_figure`: no other text may touch it
        bax.imshow(np.linspace(0.0, 1.0, 256)[None, :], aspect="auto", cmap=cmap,
                   extent=(0, 1, 0, 1))
        bax.set_yticks([])
        bax.set_xticks([0.0, 1.0])
        labels = bax.set_xticklabels(list(ends))
        labels[0].set_ha("left")
        labels[-1].set_ha("right")
        bax.tick_params(axis="x", length=pt * 0.25, pad=1.0)
        for spine in bax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(edge)
            spine.set_edgecolor(_OUTLINE_GREY)
        if title:
            bax.set_xlabel(title, fontsize=small, labelpad=1.0, loc="left" if from_left else "center")

    def ends_for(kind):
        if rendering == "neff":
            column = {tops_row[kind] for tops_row in tops.values() if kind in tops_row}
            if own or len(column) != 1:
                return "1", "own top"
            return "1", f"{column.pop():g}+ {unit}"
        if own:
            return "low", "high"
        column = [lim[kind] for lim in limits.values() if lim.get(kind)]
        if not column:
            return None
        lo, hi = column[0]
        return lo, hi                            # numbers: `bar_ends` formats them to the bar (the ".3g" before
        #                                          printed a density's 2nd percentile as "9.39e-06")

    def strip_bar(x, y_top, w, ends, kind):
        """A horizontal bar of its own (per map): the ramp and its two numbers, shortened until they fit side by
        side under it (a map is narrow: four on a slide leave about 31 mm), as `fitted` does for a column."""
        ends = bar_ends(ends, w, kind)
        bax = fig.add_axes(rect(x, y_top, w, 1.8))
        bax.imshow(np.linspace(0.0, 1.0, 256)[None, :], aspect="auto", cmap=cmap, extent=(0, 1, 0, 1))
        bax.set_yticks([])
        bax.set_xticks([0.0, 1.0])
        labels = bax.set_xticklabels(list(ends))
        labels[0].set_ha("left")
        labels[-1].set_ha("right")
        bax.tick_params(axis="x", length=pt * 0.25, pad=1.0, labelsize=small)
        for spine in bax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(edge)
            spine.set_edgecolor(_OUTLINE_GREY)

    def upright_bar(x, y_top, h, ends):
        """A vertical bar beside a row (per condition): the ramp up the strip, its numbers to the right."""
        bax = fig.add_axes(rect(x, y_top, 1.8, h))
        bax.imshow(np.linspace(0.0, 1.0, 256)[:, None], aspect="auto", cmap=cmap, origin="lower",
                   extent=(0, 1, 0, 1))
        bax.set_xticks([])
        bax.yaxis.tick_right()
        bax.set_yticks([0.0, 1.0])
        labels = bax.set_yticklabels(list(ends))
        labels[0].set_va("bottom")
        labels[-1].set_va("top")
        bax.tick_params(axis="y", length=pt * 0.25, pad=1.0, labelsize=small)
        for spine in bax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(edge)
            spine.set_edgecolor(_OUTLINE_GREY)

    if tracks_view or lay["placement"] == "none":
        pass                                 # no density drawn, or the Colour bar box is off
    elif lay["placement"] in ("rows", "maps"):
        for x, y_top, w, ends, kind in map_bars:
            strip_bar(x, y_top, w, ends, kind)
        for x, y_top, h, ends in row_bars:
            upright_bar(x, y_top, h, tuple(ends) if isinstance(ends[0], str) else chart_text.scale_ends(*ends))
        from .spatial_map_view import matrix_bar_note

        what = matrix_bar_note(scale, payload.get("row_unit", "condition"), lay["placement"])
        line = (f"colour scale {what}" if per_unit or not title else f"{title} · colour scale {what}")
        fig.text(*at(x_cols[0], y_bar + small * 0.9 * _PT_MM), line, ha="left", va="center",
                 fontsize=small, color=page["muted"])
    elif scale == "all" and rendering != "neff" and not per_unit:
        ends = ends_for(lay["kinds"][0])
        if ends:
            colour_bar(x_cols[0], x_cols[-1] + cw - x_cols[0], ends)
    else:
        # One title shared by every bar (n_eff's "n_eff (tracks), log scale" under four SDA maps) that is wider than a
        # column is written ONCE, from the first bar's left edge: centred under each it ran past the figure's edge.
        drawn = [(c, kind, ends_for(kind)) for c, kind in enumerate(lay["kinds"])]
        drawn = [(c, kind, ends) for c, kind, ends in drawn if ends]
        shared = {bar_title(kind) for _c, kind, _e in drawn}
        once = (len(drawn) > 1 and len(shared) == 1 and bool(next(iter(shared)))
                and _text_width_pt(next(iter(shared)), small, family) * _PT_MM > cw)
        for i, (c, kind, ends) in enumerate(drawn):
            colour_bar(x_cols[c], cw, ends, kind, titled=not once or i == 0, from_left=once)
    table, more = _spatial_source(payload, spec, field)
    return table, more, spatial_legend(payload), _spatial_source_notes(payload)


def _draw_sla_tracks(ax, panel: dict, length: float, cell_w: float, clip, spec: FigureSpec):
    """One SLA set's tracks on its capsule: ``panel["tracks"]`` are polylines in BOX
    FRACTIONS of the drawn capsule (u along it from the left pole, v up from the bottom wall,
    `spatialsla.track_polylines`), drawn in ``panel["ink"]`` (v2's colour) as one collection
    clipped to the analytic capsule, thin enough that a dense set still shows its lines. A
    panel with no track says so; a run from before the Tracks view is refused by name."""
    from matplotlib.collections import LineCollection

    if "tracks" not in panel or not panel.get("ink"):
        raise FigureExportError("this run has no Tracks view; run the maps again")
    lines = [[(u * length, v * cell_w) for u, v in polyline]
             for polyline in panel["tracks"] if len(polyline) >= 2]
    if not lines:
        ax.text(length / 2.0, cell_w / 2.0, "no tracks", ha="center", va="center",
                fontsize=spec.small_pt(), color=_page(spec)["muted"])
        return
    collection = LineCollection(lines, colors="#" + str(panel["ink"]).lstrip("#"),
                                linewidths=max(MIN_LINE_PT, spec.line_pt() * 0.4),
                                joinstyle="round", capstyle="round", zorder=2)
    collection.set_clip_path(clip)
    ax.add_collection(collection)


def _spatial_source(payload: dict, spec: FigureSpec, field: str) -> tuple:
    """Every bin inside the capsule, per map, as the number it was coloured by; a second
    sheet with each map's totals, its masking and the colour limits it was drawn at."""
    from .spatial_map_view import (
        has_panel_units,
        kinds_for_counts,
        matrix_limits,
        paint_field,
        panel_titles_for_counts,
    )

    present = _spatial_present(payload)
    rendering = payload.get("rendering", "bins")
    titles = dict(zip(kinds_for_counts(payload["method"], _spatial_counts(payload), _spatial_ranges(payload)),
                      panel_titles_for_counts(payload["method"], _spatial_counts(payload), _spatial_ranges(payload))))
    unit_word = "Replicate" if payload.get("row_unit") == "day" else "Condition"
    base = next(iter(present.values()))
    # SDA: the maps are in different units, so the value column holds each map's own number
    # (a probability per bin, or the mean per bin) and a Unit column says which.
    per_unit = rendering != "neff" and has_panel_units(present.values())
    if rendering == "neff":
        value_name = f"n_eff ({_neff_unit(base)})"
    elif per_unit:
        value_name = ("Value (Localisations: share of the condition's localisations in the "
                      "bin; other maps: the smoothed mean per bin)")
    elif field == "density":
        value_name = "Density (share of the map's weight in the bin; sums to 1 over the cell)"
    else:
        value_name = f"{base.get('weight_unit', 'weighted jumps')} in the bin (smoothed)"
    columns = [unit_word, "Shown as", "Population", "Bin along", "Bin across",
               "Along the cell (µm)", "Across the cell (µm)", value_name, "Masked"]
    if per_unit:
        columns = columns[:2] + ["Map", "Unit"] + columns[3:]
    rows = []
    for row in _spatial_rows(payload):
        p = present.get(row)
        if p is None:
            continue
        label = _row_label(payload, row)
        length, width = float(p["cell_length_um"]), float(p["cell_width_um"])
        for kind in _spatial_kinds(payload):
            panel = p["panels"].get(kind)
            if panel is None:
                continue
            values = (panel.get("neff") if rendering == "neff"
                      else panel[paint_field(panel, field)])
            if values is None:
                continue
            what = ([titles.get(kind, kind), panel.get("unit", "")] if per_unit
                    else [titles.get(kind, kind)])
            n_short, n_long = len(values), len(values[0])
            for iy in range(n_short):
                for ix in range(n_long):
                    x = (ix + 0.5) * length / n_long
                    yv = (iy + 0.5) * width / n_short
                    if not _in_capsule(x, yv, length, width):
                        continue
                    v = values[iy][ix]
                    rows.append([label, spec.shown_name(label), *what, ix + 1,
                                 iy + 1, round(x, 4), round(yv - width / 2.0, 4),
                                 None if v is None else float(v), bool(panel["masked"][iy][ix])])
    limits, _field = matrix_limits(present, payload.get("scale", "population"))
    summary = []
    for row in _spatial_rows(payload):
        p = present.get(row)
        for kind in _spatial_kinds(payload):
            panel = None if p is None else p["panels"].get(kind)
            if panel is None:
                continue
            lim = limits.get(row, {}).get(kind)
            summary.append([_row_label(payload, row), titles.get(kind, kind),
                            float(panel["n_jumps"]), int(panel.get("n_masked", 0)),
                            bool(panel.get("sparse")),
                            None if not lim else float(lim[0]),
                            None if not lim else float(lim[1])]
                           + ([panel.get("count_unit", ""), panel.get("unit", "")]
                              if per_unit else []))
    sheet = ("Maps", "Each map's totals and colour limits",
             ("Colour limits: the 2-98 % window the colours span (none for a panel "
              "too thin to map).",),
             [unit_word, "Map" if per_unit else "Population",
              "Count behind the map" if per_unit
              else base.get("weight_unit", "weighted jumps").capitalize(),
              "Bins masked", "Too sparse to map", "Colour low", "Colour high"]
             + (["Counted", "Unit"] if per_unit else []), summary)
    if not _sla_tracks(payload):
        return (columns, rows), (sheet,)
    # THE DRAWN LINES, point by point: in the capsule's box fractions (what the figure and
    # the screen paint) and in µm of the drawn model cell, as the bins are.
    drawn = []
    for row in _spatial_rows(payload):
        p = present.get(row)
        if p is None:
            continue
        label = _row_label(payload, row)
        length, width = float(p["cell_length_um"]), float(p["cell_width_um"])
        for kind in _spatial_kinds(payload):
            panel = p["panels"].get(kind)
            if panel is None:
                continue
            polylines = [line for line in panel.get("tracks") or () if len(line) >= 2]
            for t, polyline in enumerate(polylines, 1):
                for k, (u, v) in enumerate(polyline, 1):
                    drawn.append([label, spec.shown_name(label), titles.get(kind, kind), t, k,
                                  float(u), float(v), round(float(u) * length, 4),
                                  round(float(v) * width - width / 2.0, 4)])
    tracks = ("Tracks", "Every drawn track, point by point",
              ("Each line of the figure is a track of the set (a dwelling segment in the "
               "dwelling column), drawn unfolded. A line breaks where a localisation falls "
               "outside the capsule.",
               "u, v: position in the drawn capsule's box, u along it from the left pole and v from the "
               "bottom wall (0-1). Along the cell: µm from that pole. Across: µm from the axis "
               "(negative and positive are the two sides)."),
              [unit_word, "Shown as", "Set", "Track", "Point", "u (box, 0-1)", "v (box, 0-1)",
               "Along the cell (µm)", "Across the cell (µm)"], drawn)
    return (columns, rows), (sheet, tracks)


def _spatial_settings(payload: dict) -> list:
    from .spatial_map_view import matrix_caption_rows

    present = list(_spatial_present(payload).values())
    return matrix_caption_rows(present, payload.get("cmap", "jet"),
                               payload.get("rendering", "bins"),
                               unit=payload.get("row_unit", "condition"),
                               scale=payload.get("scale", "population"))


def _spatial_source_notes(payload: dict) -> tuple:
    from .spatial_map_view import map_warning_for

    notes = ["Along the cell: from one pole of the drawn capsule. Across: from its axis "
             "(negative and positive are the two sides). Only bins inside the capsule are "
             "listed.", *(f"{name}: {value}" for name, value in _spatial_settings(payload))]
    if payload.get("rendering") == "smooth":
        notes.append("The figure is smoothed for display; these are the bins before smoothing.")
    if _sla_tracks(payload):
        notes.append("The figure draws each set's tracks as lines in SMTracker 2's colours (the Tracks "
                     "view). This table holds the bins the same localisations were counted "
                     "into, and the Tracks sheet holds every drawn point.")
    elif payload.get("rendering") == "tracks":
        notes.append("The screen showed the Tracks view, which draws Confinement (SLA) maps "
                     "only; the figure draws these density bins.")
    warning = map_warning_for(payload.get("payloads", {}), _spatial_rows(payload),
                              unit=payload.get("row_unit", "condition"))
    if warning:
        notes.append(warning)
    return tuple(notes)


def spatial_legend(payload: dict) -> str:
    """A drafted legend for the Spatial Map grid."""
    from .spatial_map_view import map_warning_for

    present = _spatial_present(payload)
    base = next(iter(present.values()))
    method = payload.get("method", "sqd")
    day = payload.get("row_unit") == "day"
    what = "localisations" if method in ("appd", "sla") else "jumps"
    rows = (f"the replicates of {payload.get('condition') or 'one condition'} (rows; one fit "
            "per condition, shared by every replicate)" if day else "each condition (rows)")
    if _sla_tracks(payload):
        parts = [f"Where each confinement set's tracks run inside the cell, for {rows} and each "
                 "set (columns). Every cell is projected onto its own centreline and drawn onto "
                 f"one {base['cell_length_um']:.2f} × {base['cell_width_um']:.2f} µm capsule "
                 "(scale bar). Each line is one track or dwelling segment, drawn unfolded in SMTracker 2's "
                 "colours (free blue, mixed-behaviour green, dwelling red) and broken where a localisation falls "
                 "outside the capsule. Lines overlap, so the picture shows where tracks run, not how many."]
    elif method == "hmm" and base.get("show") in ("angles", "turning"):
        # THE HMM MAP'S TURNING BACK (22 Nov 2026, `spatialangles`): the data alone, or beyond chance
        parts = [f"Where each diffusive state of the hidden Markov model turns back, for {rows} and each state "
                 "(columns). Each pair of consecutive steps is placed at the localisation the two share and "
                 "weighted by the probability that both steps were in the state; pairs across a switch are left "
                 "out. A bin shows, of its pairs within 30° of forward or straight back, the share that turned back"
                 + (", minus the same share for free molecules started at the same places in the same cells (each "
                    "cell's capsule in 3D, as wide as the space the fastest state's molecules fill, the state's "
                    "observed step size, each localisation's error, blur over the camera's exposure, linked through "
                    "the tracker's radius): 0 is a free molecule there, above 0 more turning back. "
                    if base["show"] == "turning" else
                    " (0.5, no preference; no simulation is subtracted, and a free molecule also turns back near the "
                    "wall and where its steps are as small as the localisation error). ")
                 + 
                 "Every cell is projected onto its own centreline and drawn onto one "
                 f"{base['cell_length_um']:.2f} × {base['cell_width_um']:.2f} µm capsule (scale bar). Grey "
                 "bins are undefined (too few pairs), not empty."]
    elif method == "sda":
        parts = [f"Where the molecules are, and how fast they move there, for {rows} and each "
                 "map (columns). Every cell is projected onto its own centreline and drawn onto "
                 f"one {base['cell_length_um']:.2f} × {base['cell_width_um']:.2f} µm capsule "
                 "(scale bar). Grey bins are undefined (too few localisations), not empty. "
                 "Each map's colour bar has its own unit."]
    else:
        parts = [f"Where each population's {what} fall inside the cell, for {rows} and each "
                 "population (columns). Every cell is projected onto its own centreline and "
                 f"drawn onto one {base['cell_length_um']:.2f} × {base['cell_width_um']:.2f} µm "
                 "capsule (scale bar). Grey bins are undefined (too few to trust), not "
                 "empty."]
    # THE ALL COLUMN, said once before the split it stands beside: All jumps for SQD (3 Nov 2026) and HMM, All TRACKS
    # for apparent D (ruled 4 Nov 2026, D3: its maps place every localisation of whole tracks, not jumps)
    every = {"sqd": "All jumps: every jump once, as one population would draw it. ",
             "appd": "All tracks: every localisation of every track, before the split (an apparent-D map places "
                     "whole tracks, not jumps). ",
             "hmm": "All jumps: every step once, placed by its state. "}.get(method, "")
    counts = {int(p.get("n_pop") or 2) for p in present.values()}
    if method in ("sqd", "appd") and base.get("split") == "manual":
        from .spatial_map_view import manual_split_text

        parts.append(every + ("Jumps are split by their length" if method == "sqd" else
                              "Tracks are split by their mean apparent D")
                     + f" into ranges {manual_split_text(base)}; nothing is fitted, and each map is one range.")
    elif method == "sqd" and counts == {1}:
        parts.append("One SQD population: every jump in one map, nothing split.")
    elif method == "sqd":
        parts.append(every + "Jumps are split into short and long by the SQD fit. At two populations a hard "
                     "split is a cut on jump length, so the groups are jumps of different lengths, "
                     "not bound and free molecules."
                     + (" A condition with one population has no split, and its split columns say so (with the "
                        "BIC margin where Best chose the count)." if 1 in counts else ""))
    elif method == "appd":
        parts.append(every + "Tracks are split by their apparent D where the fitted populations cross.")
    elif method == "sla":
        parts.append("Localisations of the three confinement sets of SMTracker 2 (Stationary Localisation "
                     "Analysis): free tracks, tracks holding a dwell (mixed behaviour) and the dwelling "
                     "localisations themselves, a subset of the second. A dwell is a run of steps inside one "
                     "moving circle (radius and minimum dwell are settings). Each share is given beside the rate "
                     "found on free Brownian tracks.")
    elif method == "sda":
        parts.append("SDA maps as in SMTracker 2: the share of the localisations in each bin; "
                     "the mean per bin of each step's speed |Δr|/τ and apparent D |Δr|²/(4τ); "
                     "and the acceleration |Δr₂ − Δr₁|/τ² of each pair of consecutive steps, "
                     "drawn where the two steps meet. These are apparent values, bounded by the tracker's "
                     "linking radius and by localisation error; the acceleration of a diffusing molecule "
                     "is mostly localisation noise, not a force.")
    else:
        states = " or ".join(str(n) for n in sorted(counts))
        parts.append(every + f"Steps are labelled by each condition's fitted HMM at {states} "
                     f"state{'' if counts == {1} else 's'}; nothing is refitted.")
    settings = _spatial_settings(payload)
    parts.append("Settings: " + "; ".join(f"{name}: {value}" for name, value in settings) + ".")
    warning = map_warning_for(payload.get("payloads", {}), _spatial_rows(payload),
                              unit=payload.get("row_unit", "condition"))
    if warning:
        parts.append(warning)
    return " ".join(parts)


# -- a movie's image with its outlines and tracks ------------------------------------ #
def _cell_view(payload: dict) -> tuple:
    """``(x0, y0, x1, y1)`` the picture shows, in 0-based pixels: the crop, or the field."""
    image = payload.get("image")
    if image is None:
        raise FigureExportError(payload.get("image_error")
                                or "this movie has no image to draw the tracks on")
    crop = payload.get("crop")
    if crop:
        return tuple(float(v) for v in crop)
    return -0.5, -0.5, image["width"] - 0.5, image["height"] - 0.5


def _cell_image_height(payload: dict, spec: FigureSpec) -> float:
    """The height that keeps the image's own proportions at this width, plus the bar."""
    x0, y0, x1, y1 = _cell_view(payload)
    bar = spec.pt() * 4.2 * _PT_MM if payload.get("colour_mode") == "appd" else 0.0
    return float(min(600.0, max(30.0, round(spec.width_value() * (y1 - y0) / (x1 - x0)
                                             + bar + 2.0, 1))))


def _shown_tracks(payload: dict) -> list:
    cell = payload.get("cell")
    return [t for t in payload["tracks"] if cell is None or t["cell"] == cell]


def _draw_cell_image(fig, ax, payload: dict, spec: FigureSpec, family: str) -> None:
    """The Images & tracks view in print, from its own payload (`images_view.movie_payload`):
    the grey image as recorded (the page's 0.5-99.5 % stretch), the cell outlines, the tracks
    in the page's colours (`images_view.track_ink`), a scale bar in µm, and the apparent-D
    colour bar when the tracks are coloured by it. One cell zoomed shows that cell's tracks."""
    import numpy as np
    from matplotlib.cm import ScalarMappable
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap, LogNorm

    x0, y0, x1, y1 = _cell_view(payload)
    image = payload["image"]
    grey = np.frombuffer(image["pixels"], dtype=np.uint8).reshape(image["height"],
                                                                  image["width"])
    display = payload.get("display") or {}
    if display.get("brightness") or display.get("contrast"):
        from .images_view import display_lut

        # The page's brightness and contrast, the SAME table, so print matches the screen.
        lut = np.asarray(display_lut(display.get("brightness", 0),
                                     display.get("contrast", 0)), dtype=np.uint8)
        grey = lut[grey]
    line = spec.line_pt()
    colour_table = display.get("lut") or "grey"
    if colour_table != "grey":
        from .images_view import lut_colours

        # The page's colour table, the same 256 colours (`images_view.lut_colours`).
        rgb = np.asarray(lut_colours(colour_table), dtype=np.uint8)[grey]
        ax.imshow(rgb, interpolation="nearest", zorder=0)
    else:
        ax.imshow(grey, cmap="gray", vmin=0, vmax=255, interpolation="nearest", zorder=0)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if payload.get("show_outlines", True):
        for outline in payload["outlines"]:
            pts = outline["points"] + outline["points"][:1]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=payload["outline_ink"],
                    lw=max(0.5, line), zorder=2)
    if payload.get("show_tracks", True):
        tracks = _shown_tracks(payload)
        ax.add_collection(LineCollection([t["points"] for t in tracks],
                                         colors=[t["colour"] for t in tracks],
                                         linewidths=max(0.5, line), zorder=3))
    per_px = float(payload["um_per_pixel"])
    from .images_view import scale_bar_text

    bar_um = _bar_length(payload, spec) if spec.scale_bar else 0.0
    bar_px = bar_um / per_px
    left, bottom = x0 + 0.05 * (x1 - x0), y1 - 0.06 * (y1 - y0)
    from matplotlib import patheffects

    if spec.scale_bar:
        # The bar and its length together (asked for: "instead of number -> scale bar
        # (whole)"). White with a dark edge: white alone vanishes on a light phase image.
        edge = [patheffects.withStroke(linewidth=max(1.0, line * 2.5) + 1.2,
                                       foreground="#000000", alpha=0.55)]
        ax.plot([left, left + bar_px], [bottom, bottom], color="white",
                lw=max(1.0, line * 2.5), solid_capstyle="butt", zorder=4, path_effects=edge)
        ax.text(left + bar_px / 2.0, bottom - 0.025 * (y1 - y0), scale_bar_text(bar_um),
                ha="center", va="bottom", color="white", fontsize=spec.small_pt(), zorder=4,
                path_effects=[patheffects.withStroke(linewidth=1.5, foreground="#000000",
                                                     alpha=0.55)])
    if (spec.legend and payload.get("colour_mode") == "appd"
            and payload.get("show_tracks", True)):
        lo, hi = payload["appd_range"]
        cmap = LinearSegmentedColormap.from_list("appd", list(payload["ramp"]))
        bar = fig.colorbar(ScalarMappable(norm=LogNorm(lo, hi), cmap=cmap), ax=ax,
                           orientation="horizontal", fraction=0.06, pad=0.02, aspect=30)
        bar.outline.set_linewidth(max(MIN_LINE_PT * 2, line * 0.6))
        # Plain numbers, as the screen writes them -- not 10^-1.
        from matplotlib.ticker import FixedLocator, NullLocator

        ticks = chart_text.log_ticks(lo, hi)
        bar.ax.xaxis.set_major_locator(FixedLocator(ticks))
        # the one colour-bar formatter (`chart_text.scale_number`), as the screen's ramp writes its ends
        bar.ax.set_xticklabels([chart_text.scale_number(v) for v in ticks])
        bar.ax.xaxis.set_minor_locator(NullLocator())
        title = _title(spec, "cell_image", "x", payload)
        if title:
            bar.set_label(title)
        bar.ax.tick_params(labelsize=spec.small_pt())


def scale_bar_auto(payload: dict) -> float:
    """The length the cell image's scale bar takes when none is typed: a round value near
    an eighth of the width shown, so it follows the zoom (a whole field, one cell). The
    Spatial Map's: 1 µm on a capsule 1.5 µm or longer, else 0.5 µm."""
    from .images_view import scale_bar_um

    if "payloads" in payload:
        base = next(iter(_spatial_present(payload).values()), {})
        return 1.0 if float(base.get("cell_length_um", 3.0)) >= 1.5 else 0.5
    x0, _y0, x1, _y1 = _cell_view(payload)
    return scale_bar_um((x1 - x0) * float(payload["um_per_pixel"]))


def parse_map_scale(text: str):
    """A typed map scale in mm per µm ("6", "6.5", "6 mm/µm", "6,5"), or None for empty or unreadable -- None is Auto
    (`FigureSpec.map_scale`)."""
    text = str(text or "").strip().lower()
    for unit in ("mm per µm", "mm/µm", "mm per um", "mm/um", "mm"):
        text = text.replace(unit, "")
    text = text.replace(",", ".").strip()
    try:
        value = float(text)
    except ValueError:
        return None
    return value if 0 < value <= 200 else None


def parse_length(text: str):
    """A typed length in µm ("2", "0.5", "2 µm", "2um"), or None for empty or unreadable --
    None means automatic, which the preview then shows."""
    text = str(text or "").strip().lower()
    for unit in ("µm", "um", "μm"):
        text = text.replace(unit, "")
    text = text.replace(",", ".").strip()
    try:
        value = float(text)
    except ValueError:
        return None
    return value if value > 0 else None


def _bar_length(payload: dict, spec: FigureSpec) -> float:
    """The scale bar's length: typed, or automatic. A typed one that does not fit the
    image is REFUSED with what would, never shortened behind the user's back."""
    if spec.scale_bar_um is None:
        return scale_bar_auto(payload)
    x0, _y0, x1, _y1 = _cell_view(payload)
    width_um = (x1 - x0) * float(payload["um_per_pixel"])
    if spec.scale_bar_um > 0.9 * width_um:
        raise FigureExportError(
            f"a {spec.scale_bar_um:g} µm scale bar does not fit an image {width_um:.1f} µm "
            f"wide; type {0.9 * width_um:.1f} µm or less, or leave it empty for automatic")
    return float(spec.scale_bar_um)


def _spatial_bar_um(payload: dict, spec: FigureSpec, length_um: float) -> float:
    """The Spatial Map's scale bar: typed, or the maps' own (`scale_bar_auto`). A typed one longer than most of
    the capsule is REFUSED with what would fit, as the cell image's is."""
    if spec.scale_bar_um is None:
        return scale_bar_auto(payload)
    if spec.scale_bar_um > 0.9 * length_um:
        raise FigureExportError(
            f"a {spec.scale_bar_um:g} µm scale bar does not fit a capsule {length_um:.2f} µm long; type "
            f"{0.9 * length_um:.2f} µm or less, or leave it empty for automatic")
    return float(spec.scale_bar_um)


def _cell_image_source(payload: dict) -> tuple:
    """Every localisation drawn: its track, frame and position in pixels and µm."""
    per_px = float(payload["um_per_pixel"])
    columns = ["Movie", "Cell", "Track", "Frame", "x (px)", "y (px)", "x (µm)", "y (µm)",
               "Apparent D of the track (µm²/s)"]
    rows = []
    for track in _shown_tracks(payload):
        for (x, y), frame in zip(track["points"], track["frames"]):
            # µm as the analysis holds them: 1-based pixels times the pixel size.
            rows.append([payload["movie"], track["cell"], track["uid"], int(frame),
                         round(x, 3), round(y, 3), round((x + 1.0) * per_px, 5),
                         round((y + 1.0) * per_px, 5), track["appd"]])
    return columns, rows


def _cell_image_notes(payload: dict) -> tuple:
    return (f"Pixels are 0-based from the image's top-left corner; µm = (pixel + 1) x "
            f"{payload['pixel_size_nm']:g} nm, as the analysis holds positions.",
            "Apparent D: the track's mean squared step / 4 tau, which averages over any switching "
            "within the track.",
            payload["caption"])


def cell_image_legend(payload: dict, spec: FigureSpec | None = None) -> str:
    """A drafted legend for a movie or cell with its tracks."""
    what = (f"cell {payload['cell'].rsplit(':', 1)[-1]} of {payload['movie']}"
            if payload.get("cell") else payload["movie"])
    if payload["background"] == "movie":
        image = (f"Frame {int(payload.get('frame') or payload['time']['first']):,} of the "
                 f"recorded movie of {what}")
    else:
        kind = ("Phase-contrast" if payload["background"] == "phase"
                else "Fluorescence (sum)" if payload["background"] == "sum"
                else f"{payload.get('background_label') or payload['background']} fluorescence")
        image = f"{kind} image of {what}"
    parts = [f"{image} ({payload['condition']}) with the segmented cell "
             "outlines (yellow) and single-molecule tracks"]
    if payload.get("colour_mode") == "appd":
        lo, hi = payload["appd_range"]
        parts[0] += (f", each coloured by its apparent diffusion coefficient on a log scale "
                     f"from {lo:.3g} (blue) to {hi:.3g} µm²/s (red), the condition's 2-98 % "
                     "range.")
    else:
        parts[0] += "."
    display = payload.get("display") or {}
    from .images_view import display_words

    parts.append("Image display: " + (display.get("words") or display_words(
        display.get("brightness", 0), display.get("contrast", 0), display.get("lut") or "grey")) + ".")
    if spec is None or spec.scale_bar:
        from .images_view import scale_bar_text

        parts.append("Scale bar, " + scale_bar_text(
            _bar_length(payload, spec or FigureSpec())) + ".")
    parts.append("Tracks shown: " + str(len(_shown_tracks(payload))) + ".")
    if payload.get("trail_frame") is not None:
        from .images_view import TRAIL_FRAMES

        parts.append(f"Each track as it stood at frame {int(payload['trail_frame']):,}: its "
                     f"path over the last {TRAIL_FRAMES} frames.")
    parts.append("An apparent D averages over switching within a track. It describes the data; the "
                 "population readout comes from the SQD fit.")
    return " ".join(parts)


def _sqd_points(data, colours: dict) -> list:
    from .bubble_view import fraction_text

    def colour_of(column):
        return colours.get(data.conditions[column]) or figstyle.OKABE_ITO[
            1 + column % (len(figstyle.OKABE_ITO) - 1)]

    return [(p.column, p.x_offset, p.d, p.fraction, fraction_text(p, compact=True),
             colour_of(p.column), p.style_index) for p in data.points]


def _replicate_columns(payload: dict) -> list:
    return [f"Replicate {rep['replicate']}" for rep in payload["replicates"]]


def _replicate_points(payload: dict, colour) -> list:
    """The replicate-days page's bubbles: each replicate's populations at its D (its own
    under "replicate" scope, the shared one otherwise), each replicate in its SHADE of the
    condition's colour (`figstyle.day_shade`, as the screen chart draws it), or in the
    state colours when there is no condition colour."""
    points = []
    n_reps = len(payload["replicates"])
    for column, rep in enumerate(payload["replicates"]):
        ds = rep.get("D") or payload.get("shared_D") or []
        for k, (d, f) in enumerate(zip(ds, rep["fractions"])):
            if d < 0 or (d == 0 and not (payload.get("zero_jumps") or {}).get("immobile")):
                continue                     # a D of 0 is drawn only as the immobile population (`_draw_columns`)
            ink = (figstyle.day_shade(colour, column, n_reps) if colour
                   else figstyle.STATE_COLORS.get(k + 1, figstyle.OKABE_ITO[1 + k % 7]))
            points.append((column, 0.0, float(d), float(f), f"{100 * f:.0f}%", ink, 0))
    return points


_CROSSED_MARKERS = {"circle": "o", "square": "s", "diamond": "D", "triangle": "^"}


def _draw_sqd_crossed(fig, ax, payload: dict, spec: FigureSpec, family: str) -> None:
    """v2's crossed plot in print (`bubble_view.crossed_payload`): fraction across, D up on a log axis, a point per
    population per condition, bars only where the Run has a spread; colour the condition, marker the population."""
    from matplotlib.lines import Line2D

    cr = payload["crossed"]
    if not cr.get("available"):
        raise FigureExportError(cr.get("why", "nothing to draw"))
    colours = payload.get("colours") or {}
    page = _page(spec)
    line, small = spec.line_pt(), spec.small_pt()
    for pt in cr["points"]:
        colour = _strong(spec, colours.get(pt["condition"]) or QC_DATA)
        xerr = None if not pt["x_bar"] else [[pt["x"] - pt["x_bar"][0]], [pt["x_bar"][1] - pt["x"]]]
        yerr = None if not pt["y_bar"] else [[pt["y"] - pt["y_bar"][0]], [pt["y_bar"][1] - pt["y"]]]
        # a D of 0 (exact-zero jumps) at its "0" tick, `chart_text.log_drawn`
        ax.errorbar([pt["x"]], [chart_text.log_drawn(pt["y"], cr.get("zero_d"))], xerr=xerr, yerr=yerr, fmt=_CROSSED_MARKERS[pt["marker"]], color=colour,
                    ms=small * 0.8, mec=page["bg"], mew=max(MIN_LINE_PT, line * 0.6), elinewidth=max(0.6, line),
                    capsize=small * 0.3, capthick=max(0.6, line), zorder=3)
    ax.set_yscale("log")
    ax.set_xlim(*cr["x_range"])
    ax.set_ylim(*cr["y_range"])
    _log_y_ticks(ax, cr["y_ticks"], cr.get("y_tick_labels"))
    ax.set_xlabel(_title(spec, "sqd_crossed", "x"))
    ax.set_ylabel(_title(spec, "sqd_crossed", "y"))
    _qc_grid(ax, spec)
    if spec.legend:
        handles = [Line2D([], [], color=_strong(spec, colours.get(c) or QC_DATA), marker="o", ls="none",
                          ms=small * 0.7) for c, _n in cr["legend_conditions"]]
        names = [spec.shown_name(c) for c, _n in cr["legend_conditions"]]
        handles += [Line2D([], [], color=page["muted"], marker=_CROSSED_MARKERS[m], ls="none", ms=small * 0.7,
                           markerfacecolor="none") for m, _n in cr["legend_populations"]]
        names += [n for _m, n in cr["legend_populations"]]
        _legend_above(ax, handles, names, spec, family)


def _log_y_ticks(ax, ticks, labels=None) -> None:
    from matplotlib.ticker import FixedLocator, NullLocator

    ax.yaxis.set_major_locator(FixedLocator(list(ticks)))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels(list(labels) if labels else [f"{t:g}" for t in ticks])


def _sqd_crossed_source(payload: dict, spec: FigureSpec) -> tuple:
    rows = []
    for pt in payload["crossed"]["points"]:
        xb, yb = pt["x_bar"] or (None, None), pt["y_bar"] or (None, None)
        rows.append([spec.shown_name(pt["condition"]), pt["population"], round(pt["x"], 3),
                     None if xb[0] is None else round(xb[0], 3), None if xb[1] is None else round(xb[1], 3),
                     pt["y"], yb[0], yb[1]])
    return list(payload["crossed"]["columns"]), rows


def sqd_crossed_legend(payload: dict) -> str:
    data = payload.get("data")
    text = ("The SQD populations as fraction against D (SMTracker 2's crossed plot). Each point is one population of "
            "one condition (colour: the condition; marker: the population, slowest first), with its fraction across "
            "and its apparent D up (log scale). " + payload["crossed"]["caption"])
    if data is not None:
        rest = bubble_legend(data)
        cut = rest.find("Fractions are")
        text += " " + (rest[cut:] if cut >= 0 else rest)
    return text


def bubble_legend(data, shown=str) -> str:
    """A drafted figure legend for the bubble chart, in words, from the plot's own data.

    Written to be CHECKED, not pasted: it says what the encoding means, what the spread
    is (a spread between days is not a confidence interval) and that D is apparent --
    the three things a reader most often gets wrong from this figure. ``shown`` names a
    column as the figure does (`FigureSpec.shown_name`)."""
    weighting = getattr(data, "fraction_weighting", "track")
    what = "displacements" if weighting == "step" else "tracks (each track counted once)"
    parts = ["Diffusive-state populations from squared-displacement (SQD) fits. Each "
             "bubble is one population: its height is the apparent diffusion coefficient "
             "D (log scale), its area the fraction of " + what + " in that population, "
             "and the number under it is that fraction."]
    spreads = {p.spread for p in data.points}
    days = sorted({p.n_replicates for p in data.points})
    days_txt = "-".join(str(d) for d in (days[0], days[-1])) if len(days) > 1 else (
        str(days[0]) if days else "?")
    pseudo = getattr(data, "pseudo", None) or {}
    if "sd" in spreads and pseudo:
        from .bubble_view import chunk_words

        parts.append(f"Fractions are the mean ± SD over {chunk_words(pseudo)}. Each chunk's fraction is fitted "
                     "with its condition's D held, and the SD is the spread between chunks, not a confidence "
                     "interval.")
    elif "sd" in spreads:
        parts.append(f"Fractions are the mean ± SD over replicates (n = {days_txt} "
                     "replicates per condition). The SD is the spread between replicates, not a "
                     "confidence interval.")
    if "ci" in spreads:
        from .comparison import SINGLE_SESSION_RESAMPLES

        unit = {"movie": "movies"}.get(getattr(data, "uncertainty", "cell"), "cells")
        parts.append("Fractions are from one fit to all data, with a 95 % confidence interval from "
                     f"{SINGLE_SESSION_RESAMPLES:,} resamples of the {unit}, the model refitted on each "
                     "(percentile interval).")
    if "none" in spreads:
        parts.append("Fractions are from one fit to all replicates pooled; no "
                     "interval is shown.")
    parts += _chunks_withheld_sentences(data, shown)
    for warning in getattr(data, "warnings", None) or ():
        parts.append(f"Check the fit: {str(warning).rstrip('.')}.")
    lag = getattr(data, "lag_frames", 1) or 1
    if lag == "1-4":
        parts.append("Squared displacements 1 to 4 frames apart were fitted together (SMTracker 2's \u201c1-4 "
                     "together\u201d), giving one D and one set of fractions. The number of populations was "
                     "chosen at a lag of one frame. Localisation error is not modelled, so a population at the "
                     "floor reads lower than at a lag of one frame.")
        lag = 1
    lag = int(lag)
    if lag != 1:
        ms = getattr(data, "lag_ms", None)
        parts.append(f"Squared displacements {lag} frames apart" + (f" ({ms:g} ms)" if ms else "")
                     + f" were fitted, so D is the apparent D at that lag. A population of localisation error alone "
                     f"reads {lag} times lower than at a lag of one frame; a diffusing state reads the same.")
    per_spot = bool(getattr(data, "precision", ""))
    if per_spot:
        from .bubble_view import PER_SPOT_D_IS

        parts.append("Each step's own localisation error entered the fit (per-spot precision), taken from its "
                     "two spots through a calibration measured on immobile spots (" + data.precision + "), "
                     f"by maximum likelihood, so D is {PER_SPOT_D_IS}. A moving population's D is still apparent.")
    scope = getattr(data, "scope", "condition")
    if scope == "group":
        from .bubble_view import sharing_caption

        parts.append(sharing_caption(getattr(data, "sharing_factor", ""), getattr(data, "sharing_groups", {}))
                     + ", and by their replicates (one simultaneous fit per group).")
    from .bubble_view import mixed_scopes

    scopes = mixed_scopes(data)
    if scopes:
        # A SHARED D PER CONDITION (1 Nov 2026): the one sentence below would be wrong for some columns, so each
        # condition is named with its own, as its sample line says
        def listed(names):
            return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]

        from .workspace_view import SCOPE_SAMPLE_WORDS

        within = [str(shown(c)) for c, s in scopes.items() if s == "condition"]
        alone = [str(shown(c)) for c, s in scopes.items() if s == "replicate"]
        said = []
        # the words written under each column (`workspace_view.SCOPE_SAMPLE_WORDS`), each with what it fits
        if within:
            said.append(f"{SCOPE_SAMPLE_WORDS['condition']} (D fitted once per condition and shared by its "
                        f"replicates) for " + listed(within))
        if alone:
            said.append(f"{SCOPE_SAMPLE_WORDS['replicate']} (each replicate its own D) for " + listed(alone))
        parts.append("D was forced differently between conditions, as written under each column: " + "; ".join(said)
                     + ". A significance test between two conditions fitted differently is refused because the "
                       "difference would include the method.")
    else:
        parts.append({"condition": "D is fitted once per condition and shared by its "
                                   "replicates.",
                      "replicate": "Each replicate has its own D.",
                      "global": "One set of D values is shared by every condition "
                                "(simultaneous fit).",
                      "all": "One set of D values is shared by every condition "
                             "(simultaneous fit)."}.get(scope, ""))
    if data.series_labels:
        parts.append("Fits under different arrangements are drawn side by side and distinguished "
                     "by their outline: " + ", ".join(data.series_labels) + ".")
    # each column's model, and the counts a Best column also stands for (`bubble_view.fold_best`)
    def _models_of(f):
        return [str(f.get("model") or f["n_pop"])] + [str(m) for m in f.get("also") or ()]

    fits = getattr(data, "fits", None) or []
    models = list(dict.fromkeys(m for f in fits for m in _models_of(f)))
    if len(models) >= 2:
        from .bubble_view import folded_words, models_words

        # a condition that shows fewer of the models (`ComparisonSpec.side_by_side_shown`) is named with its own
        per = {}
        for f in data.fits:
            per.setdefault(str(f["condition"]), []).extend(_models_of(f))
        fewer = [f"{shown(c)} with {models_words([x for x in models if x in m])} only" for c, m in per.items()
                 if len(set(m)) < len(models)]
        parts.append(f"Each condition is fitted with {models_words(models)}" + (f" ({'; '.join(fewer)})" if fewer
                                                                                 else "")
                     + ", drawn side by side under its name, the first in the darkest shade of its colour. This "
                       "compares models on the same replicates, not samples.")
        folded = [dict(f, condition=shown(f["condition"])) for f in fits if f.get("also")]
        if folded:
            # BEST ONCE: the count Best chose and a count also picked are one fit, drawn once
            parts.append(folded_words(folded) + " (the same fit).")
    if data.errors:
        parts.append("No fit for " + "; ".join(
            f"{shown(data.conditions[i])} ({reason})" for i, reason in data.errors.items()) + ".")
    # THE CLOSING SENTENCE AGREES WITH THE PER-SPOT ONE (7 Nov 2026, the per-spot audit's CORRECTION 2: it said "D values
    # are apparent: the slowest population lies near the localization floor" four sentences after "corrected for
    # localisation error"). Per spot the slowest is no longer at the floor (where its D sits on the fit's bound, the
    # fit's checks say "not resolved"); the fastest is still near the tracking ceiling.
    parts.append("The fastest population's D lies near the tracking ceiling, so D values are bounds, not "
                 "physical coefficients; the fractions are the main readout." if per_spot else
                 "D values are apparent: the slowest population lies near the "
                 "localization floor and the fastest near the tracking ceiling, so they "
                 "are bounds, not physical coefficients. The fractions are the "
                 "main readout.")
    return " ".join(p for p in parts if p)


def replicate_legend(payload: dict, condition: str = "", *, shaded: bool = False) -> str:
    """A drafted legend for the replicate-days figure, from the fit's own payload.
    ``shaded``: the replicates are drawn as shades of the condition's colour."""
    weighting = payload.get("weighting", "track")
    what = "displacements" if weighting == "step" else "tracks (each track counted once)"
    who = f" of {condition}" if condition else ""
    reps = payload.get("replicates") or []
    parts = [f"Squared-displacement (SQD) fit{who}, shown per replicate ({payload.get('unit', 'day')}). "
             "Each column is one replicate and each bubble one population. Its height is the "
             "apparent diffusion coefficient D (log scale); its area and the number under it are "
             "the fraction of " + what + " in that population."]
    if shaded and len(payload.get("replicates") or []) > 1:
        parts.append("Replicates are shades of the condition's colour (darkest: the first).")
    shared = payload.get("shared_D")
    if shared:
        parts.append("D is shared by the replicates (" + ", ".join(f"{d:.3g}" for d in shared)
                     + " µm²/s); only the fractions are fitted per replicate.")
    else:
        parts.append("Each replicate is fitted with its own D.")
    if reps:
        cells = [int(r.get("n_cells", 0)) for r in reps]
        tracks = [int(r.get("n_tracks", 0)) for r in reps]
        parts.append(f"{len(reps)} replicates; {min(cells)}-{max(cells)} cells and "
                     f"{min(tracks)}-{max(tracks)} tracks each.")
    parts.append("D values are apparent: the slowest population lies near the "
                 "localization floor and the fastest near the tracking ceiling, so they "
                 "are bounds, not physical coefficients. The fractions are the "
                 "main readout.")
    return " ".join(parts)


def _sqd_source(data, spec: FigureSpec) -> tuple:
    """One row per drawn population, as numbers; a condition that fit nothing is a row
    naming why (a missing row would read as "no data")."""
    from .bubble_view import SAMPLE_UNITS, column_condition, mixed_scopes, scope_words

    columns = ["Condition", "Shown as", "Series", "Population", "D (µm²/s)", "Fraction",
               "Spread", "SD", "CI low", "CI high", "Replicates",
               "Sample: spread over", "Sample: n", "Tracks fitted"]
    series = list(getattr(data, "series_labels", []) or [])
    samples = getattr(data, "samples", None) or {}
    # A SHARED D PER CONDITION (1 Nov 2026): a column of its own only where the conditions differ in it, as on the
    # figure's sample line; one Force same D across for every condition leaves the sheet as it was
    from .workspace_view import SHARED_D_NAME

    scopes = mixed_scopes(data)
    if scopes:
        columns.append(SHARED_D_NAME)

    def sample(column):
        # the column's sample, as written under it (`bubble_view.sample_text`): its unit, how many, the tracks
        got = samples.get(column_condition(data, column)) or {}
        unit = got.get("unit")
        words = "replicates pooled (no spread)" if unit == "pooled" else SAMPLE_UNITS.get(unit, ("", ""))[1]
        out = [words or None, got.get("n"), got.get("tracks")]
        if scopes:
            scope = scopes.get(column_condition(data, column))
            out.append(scope_words(scope) if scope else None)
        return out

    rows = []
    for p in sorted(data.points, key=lambda p: (p.column, p.style_index, p.population)):
        condition = str(data.conditions[p.column])
        low, high = (p.interval if p.interval else (None, None))
        rows.append([condition, spec.shown_name(condition),
                     series[p.style_index] if p.style_index < len(series) else "",
                     int(p.population), float(p.d), float(p.fraction), p.spread,
                     float(p.fraction_sd) if p.spread == "sd" else None,
                     None if low is None else float(low),
                     None if high is None else float(high), int(p.n_replicates)] + sample(p.column))
    for column, reason in data.errors.items():
        condition = str(data.conditions[column])
        rows.append([condition, spec.shown_name(condition), "", None, None, None,
                     f"no fit: {reason}", None, None, None, None, None, None, None] + ([None] if scopes else []))
    return columns, rows


def _sqd_source_notes(data) -> tuple:
    weighting = getattr(data, "fraction_weighting", "track")
    pseudo = getattr(data, "pseudo", None) or {}
    if pseudo:
        from .bubble_view import chunk_words

        spread = f"Spread: sd = SD over {chunk_words(pseudo)}, not a confidence interval."
        held = list(pseudo.get("withheld") or {})
        if held:
            from .comparison import CHUNKS_WITHHELD

            spread += f" none = {', '.join(held)}: {CHUNKS_WITHHELD}; one pooled fit, no interval."
    else:
        spread = ("Spread: sd = SD over replicates (a spread, not a confidence interval); "
                  "ci = 95 % interval from resampling cells; none = one pooled fit, no interval.")
    from .bubble_view import mixed_scopes
    from .workspace_view import SHARED_D_NAME, scope_words

    shared = ((f"{SHARED_D_NAME}: how the condition's D was fitted, {scope_words('condition')} (one D shared by "
               f"its replicates, each replicate's fractions at it) or {scope_words('replicate')}. A test between "
               f"two conditions fitted differently is refused.",) if mixed_scopes(data) else ())
    return ("D: apparent diffusion coefficient of the population, µm²/s, from the "
            "squared-displacement (SQD) fit.",
            "Fraction: the population's share of "
            + ("displacements" if weighting == "step" else "tracks") + ", 0 to 1.",
            spread) + shared


def _replicate_source(fit: dict, spec: FigureSpec) -> tuple:
    columns = ["Replicate", "Shown as", "Cells", "Tracks", "Population", "D (µm²/s)",
               "Fraction", "D shared"]
    rows = []
    shared = fit.get("shared_D")
    for rep in fit.get("replicates") or []:
        name = f"Replicate {rep['replicate']}"
        ds = rep.get("D") or shared or []
        for k, (d, f) in enumerate(zip(ds, rep["fractions"])):
            rows.append([str(rep["replicate"]), spec.shown_name(name),
                         int(rep.get("n_cells", 0)), int(rep.get("n_tracks", 0)), k + 1,
                         float(d), float(f), bool(shared)])
    return columns, rows


def _replicate_source_notes(fit: dict) -> tuple:
    return ("D: apparent diffusion coefficient, µm²/s. It is shared by the replicates when "
            "'D shared' is true and fitted per replicate otherwise.",
            "Fraction: the population's share of "
            + ("displacements" if fit.get("weighting") == "step" else "tracks")
            + " in that replicate, 0 to 1.",
            f"Replicate unit: {fit.get('unit', 'day')}.")


# -- significance brackets (ruled 13 Oct 2026) ------------------------------------------ #
#: The spec's significance fields: per chart, kept by a composed panel, never remembered
#: for the next chart.
SIGNIFICANCE_WORDS = ("significance", "significance_pairs", "significance_population",
                      "significance_test", "significance_label", "significance_stars",
                      "significance_margin", "significance_correction")
#: How far a bracket's ends stop short of its columns' centres, in em of the body text: two
#: brackets meeting at one column keep their own legs (one shared leg reads as one long
#: bracket), and the legs still stand over the bubbles (a fixed share of a column put them
#: visibly beside them at two columns' width).
_BRACKET_INSET_EM = 0.3


def _significance_problem(spec: FigureSpec) -> str:
    """Why the spec's significance fields cannot be used, or ""."""
    from .significance import CORRECTIONS, LABELS, TESTS, parse_stars

    if spec.significance_test not in TESTS:
        return f"the significance test must be one of {', '.join(TESTS)}"
    if spec.significance_correction not in CORRECTIONS:
        return f"the correction over the brackets must be one of {', '.join(CORRECTIONS)}"
    if spec.significance_label not in LABELS:
        return f"a bracket says one of {', '.join(LABELS)}"
    margin = spec.significance_margin
    if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not 0 < margin < 100:
        return "the equivalence margin is a number of percentage points above 0 and below 100"
    try:
        parse_stars(", ".join(str(t) for t in spec.significance_stars))
    except ValueError as exc:
        return str(exc)
    from .significance import HMM_PANEL_WORDS

    population = spec.significance_population
    # an HMM comparison's bracket compares a PANEL (5 Nov 2026), named by its key
    if population not in ("all", "below") and population not in HMM_PANEL_WORDS and (
            isinstance(population, bool) or not isinstance(population, int) or population < 1):
        return ("the population compared must be a whole number from 1, all, or below (on the HMM comparison a "
                "panel: occupancy, rate_in, rate_out or D)")
    seen = set()
    for pair in spec.significance_pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            return "a significance pair needs two conditions"
        if str(pair[0]) == str(pair[1]):
            return "a significance pair needs two different conditions"
        key = frozenset((str(pair[0]), str(pair[1])))
        if key in seen:
            return f"{pair[0]} and {pair[1]} are compared twice"
        seen.add(key)
    return ""


def _significance_plan(kind: str, payload: dict, spec: FigureSpec, family: str):
    """The brackets this figure asks for, or None: ``{"comparisons", "brackets" (each
    ``{"a", "b", "population", "side", "x0", "x1", "text", "level", "spots"}``: its columns,
    whose bubbles it stands at and on which side, where its legs stand, what it says, its
    tier within its population, and that population's bubbles it spans as ``(column, D,
    fraction)``), "above_pt", "below_pt", "band_pt", "findings"}``.

    The statistics are `significance.compare`'s (Welch on per-day values, or the nested
    mixed model on per-cell values, the spec's correction over the distinct tests); what a bracket says and
    where it stands are `significance.bracket_marks`', the screen chart's own. A figure that
    asks for brackets it cannot carry -- pooled points, an overlay, a chart with no per-day
    fits -- is REFUSED by name, never drawn without them; a single pair with no test is drawn
    as nothing and named in the checks."""
    from . import significance as sig

    if (kind not in sig.OFFERED or kind == sig.HMM_KIND or not spec.significance
            or not spec.significance_pairs):
        return None                         # the HMM comparison's brackets are `_hmm_bracket_plan`'s
    try:
        found = sig.compare(kind, payload, spec.significance_pairs,
                            spec.significance_population, spec.significance_test,
                            margin_pp=float(spec.significance_margin),
                            correction=spec.significance_correction)
    except ValueError as exc:
        raise FigureExportError(f"significance brackets were asked for, but {exc}") from None
    data = payload["data"]
    layout = getattr(data, "layout", None)
    x_of = data.column_x if _grouped(layout) else float
    span = (x_of(len(data.conditions) - 1) - x_of(0) + 1.2) if _grouped(layout) else None
    spacing = _column_spacing(spec, len(data.conditions), span)
    inset = min(0.1, _BRACKET_INSET_EM * spec.pt() / spacing)
    marks = sig.bracket_marks(found, payload, spec.significance_label,
                              spec.significance_stars)
    groups: dict = {}
    for mark in marks:
        # the bubbles by column NUMBER, the legs at column POSITIONS (grouped columns stand apart)
        mark["spots"] = _bracket_spots(data, mark)
        xa, xb = x_of(mark["a"]), x_of(mark["b"])
        mark.update(ia=mark["a"], ib=mark["b"], a=xa, b=xb, x0=xa + inset, x1=xb - inset)
        groups.setdefault((mark["population"], mark["side"]), []).append(mark)
    for group in groups.values():
        for mark, level in zip(group, _bracket_levels(group, spacing, spec.small_pt(), family)):
            mark["level"] = level
    top = max((pt.population for pt in data.points), default=1)
    above = [m["level"] for m in marks if m["side"] == "above" and m["population"] == top]
    below = [m["level"] for m in marks if m["side"] == "below"]
    above_pt = _bracket_band_pt(max(above) + 1 if above else 0, spec)
    below_pt = _bracket_band_pt(max(below) + 1 if below else 0, spec)
    findings = [f"no bracket for {spec.shown_name(c.a)} against {spec.shown_name(c.b)}: "
                f"{c.reason}" for c in found if not c.tested]
    return {"comparisons": found, "brackets": marks, "above_pt": above_pt,
            "below_pt": below_pt, "band_pt": above_pt + below_pt, "findings": findings,
            # a bracket standing on a population "Check the fit" flags: no mark, a sentence in the legend (4 Nov 2026)
            "warnings": sig.bracket_warnings(found, data, spec.shown_name)}


def _bracket_spots(data, mark) -> list:
    """``[(column, D, fraction)]`` of the bubbles ``mark`` stands at in the columns it spans: its
    population's, or, for the share below the D threshold, every bubble below it (the slowest
    population's when none is)."""
    span = [pt for pt in data.points if mark["a"] <= pt.column <= mark["b"]]
    if mark["population"] == "below":
        under = [pt for pt in span if pt.d < mark["below_d"]]
        chosen = under or [pt for pt in span if pt.population == 1]
    else:
        chosen = [pt for pt in span if pt.population == mark["population"]]
    # where each bubble is DRAWN: a D of 0 at its "0" tick (`_draw_columns`, `chart_text.log_drawn`)
    zero_d = chart_text.zero_d_stand_in(pt.d for pt in data.points)
    return [(pt.column, chart_text.log_drawn(pt.d, zero_d), pt.fraction) for pt in chosen]


def _bracket_levels(marks, spacing: float, small: float, family: str) -> list:
    """Each bracket's tier, 0 nearest its bubbles: `significance.bracket_tiers` (the screen's
    rule too), with the text's width measured in this figure's font and points."""
    from .significance import bracket_tiers

    return bracket_tiers(
        marks, lambda m: (_text_width_pt(m["text"], small, family) / 2.0 + 0.3 * small) / spacing)


def _bracket_geometry(spec: FigureSpec) -> tuple:
    """``(gap from the bubbles, leg, lift of the text off its line, tier)`` in points."""
    pt, small = spec.pt(), spec.small_pt()
    leg, lift = 0.35 * pt, 0.15 * pt
    return 0.4 * pt, leg, lift, leg + lift + 1.25 * small + 0.25 * pt


def _bracket_band_pt(tiers: int, spec: FigureSpec) -> float:
    """The height of ``tiers`` stacked brackets, in points."""
    if tiers <= 0:
        return 0.0
    base, _leg, _lift, tier = _bracket_geometry(spec)
    return base + tiers * tier


def _points_over_plot(fig, ax):
    """A y coordinate in POINTS ABOVE THE TOP OF ``ax``'s plot, as a transform for
    `blended_transform_factory`: brackets placed in it follow the plot wherever the layout
    puts it, and stay a fixed size in print. Their texts and lines are not clipped, so the
    constrained layout counts them and shrinks the plot to make room -- no bracket can sit
    on a bubble, and a typed D range is drawn exactly as typed."""
    from matplotlib.transforms import Affine2D, ScaledTranslation

    return (Affine2D().scale(1.0 / 72.0) + fig.dpi_scale_trans
            + ScaledTranslation(0.0, 1.0, ax.transAxes))


def _points_from(fig, ax, d: float):
    """A y coordinate in POINTS FROM THE HEIGHT OF D = ``d`` on ``ax``, as a transform for
    `blended_transform_factory`: a bracket drawn in it stays anchored to its bubble wherever
    the layout puts the plot, and a fixed size in print."""
    from matplotlib.transforms import Affine2D, ScaledTranslation

    return (Affine2D().scale(1.0 / 72.0) + fig.dpi_scale_trans
            + ScaledTranslation(0.0, d, ax.transData))


def _draw_brackets(fig, ax, marks, spec: FigureSpec, *, per_decade: float, reach) -> None:
    """A bracket per tested pair AT ITS POPULATION's bubbles, the screen chart's placement:
    over the highest of them in the columns it spans (legs down, the text over the line), or,
    for the slowest population, under the lowest number (legs up, the text beneath), stacked
    by tier within the population. ``per_decade`` is the points one decade of D takes,
    ``reach(spot, side)`` how far a bubble and its number stand out over ("above") or under
    ("below") its centre, in points (`chart_text.number_reach`)."""
    import math as _math

    from matplotlib.lines import Line2D
    from matplotlib.transforms import blended_transform_factory

    if not marks:
        return
    base, leg, lift, tier = _bracket_geometry(spec)
    ink = _page(spec)["ink"]
    width = max(MIN_LINE_PT * 2, spec.line_pt() * 0.8)
    for mark in marks:
        spots = mark.get("spots") or []
        if not spots:
            continue
        lo, hi, x0, x1 = mark["a"], mark["b"], mark["x0"], mark["x1"]
        if mark["side"] == "below":
            spot = min(spots, key=lambda s: _math.log10(s[1]) * per_decade - reach(s, "below"))
            d, edge = spot[1], -reach(spot, "below")
            y = edge - base - leg - mark["level"] * tier
            legs, text_y, va = y + leg, y - lift, "top"
        else:
            spot = max(spots, key=lambda s: _math.log10(s[1]) * per_decade + reach(s, "above"))
            d, edge = spot[1], reach(spot, "above")
            y = edge + base + leg + mark["level"] * tier
            legs, text_y, va = y - leg, y + lift, "bottom"
        at = blended_transform_factory(ax.transData, _points_from(fig, ax, d))
        ax.add_artist(Line2D([x0, x0, x1, x1], [legs, y, y, legs], transform=at,
                             color=ink, linewidth=width, solid_capstyle="butt",
                             solid_joinstyle="miter", clip_on=False, zorder=6))
        ax.text((lo + hi) / 2.0, text_y, mark["text"], transform=at, ha="center", va=va,
                fontsize=spec.small_pt(), color=ink, clip_on=False, zorder=6)


def _with_bracket_room(kind: str, payload: dict, spec: FigureSpec, family: str):
    """An AUTOMATIC height grows by the bracket band, so the plot keeps its size and the
    brackets get their own room (capped at the journal's tallest; a slide keeps 16:9 and a
    height that was set is kept, the band then taken from the plot)."""
    if spec.height_mm is not None or spec.preset in ("slides", "slides_dark"):
        return spec
    plan = _significance_plan(kind, payload, spec, family)
    if not plan or not plan["band_pt"]:
        return spec
    height = spec.default_height_mm() + plan["band_pt"] * _PT_MM
    tallest = spec.preset_values.get("max_height_mm")
    return replace(spec, height_mm=round(min(height, tallest) if tallest else height, 1))


def _hmm_bracket_plan(payload: dict, spec: FigureSpec):
    """The HMM comparison's brackets (5 Nov 2026), or None: ``{"comparisons", "marks", "kind"}`` -- the statistics
    `significance.compare`'s (each panel's per-replicate values, Welch or the paired test, the spec's correction
    over every bracket),
    the marks the screen strip draws. Asked for and impossible is REFUSED by name, as on the bubbles."""
    from . import significance as sig

    if not spec.significance or not spec.significance_pairs:
        return None
    try:
        found = sig.compare(sig.HMM_KIND, payload, spec.significance_pairs, spec.significance_population,
                            spec.significance_test, correction=spec.significance_correction)
    except ValueError as exc:
        raise FigureExportError(f"significance brackets were asked for, but {exc}") from None
    return {"comparisons": found, "kind": sig.HMM_KIND,
            "marks": sig.bracket_marks(found, payload, spec.significance_label, spec.significance_stars)}


def _significance_sheets(plan: dict, spec: FigureSpec) -> tuple:
    """The source data's "Comparisons" sheet (one row per pair picked, every statistic as a
    number, the test, the correction and RULE 0 in its header) and "Per-day values" (or
    "Per-cell values" for the mixed model: each value the test used)."""
    from . import significance as sig

    found = plan["comparisons"]
    test = spec.significance_test
    columns, rows = sig.comparison_table(found, spec.shown_name, spec.significance_label,
                                         spec.significance_stars)
    v_columns, v_rows = sig.values_table(found, spec.shown_name)
    kind = plan.get("kind", "")
    v_name, v_title, v_notes = sig.values_sheet(test, kind)
    return (("Comparisons", "Comparisons: the significance brackets",
             sig.sheet_notes(test, kind, spec.significance_correction), columns, rows),
            (v_name, v_title, v_notes, v_columns, v_rows))


def _significance_legend(legend: str, plan: dict, spec: FigureSpec) -> str:
    """The drafted legend with the brackets' sentence after it."""
    from . import significance as sig

    words = sig.legend_sentence(plan["comparisons"], spec.shown_name, spec.significance_label,
                                spec.significance_stars)
    # A BRACKET ON A FLAGGED POPULATION (ruled 4 Nov 2026): drawn as any other, the reason here
    words = " ".join([words] + [f"Check the fit: {w}." for w in plan.get("warnings") or ()]).strip()
    return f"{legend} {words}" if words else legend


def _sheet_name(text: str, used: set) -> str:
    """An Excel sheet name: at most 31 characters, no bracket, colon, star, question mark
    or slash of either kind, and unique in the workbook."""
    clean = "".join("_" if c in '[]:*?/\\' else c for c in str(text)).strip() or "Sheet"
    name, k = clean[:31], 2
    while name.lower() in used:
        suffix = f" ({k})"
        name, k = clean[:31 - len(suffix)] + suffix, k + 1
    used.add(name.lower())
    return name


def write_source_workbook(path, sheets) -> Path:
    """Write ``sheets`` -- ``[(sheet name, title, header lines, columns, rows)]`` -- as one
    workbook: each sheet the figure's title and header lines, a blank row, then the table
    with its header in bold, numbers as numbers, the panes frozen under the header."""
    from openpyxl import Workbook
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter

    book = Workbook()
    book.remove(book.active)
    used: set = set()
    for name, title, lines, columns, rows in sheets:
        sheet = book.create_sheet(_sheet_name(name, used))
        sheet.append([title])
        sheet["A1"].font = Font(bold=True, size=12)
        for line in lines:
            sheet.append([line])
        sheet.append([])
        sheet.append(list(columns))
        head = sheet.max_row
        for cell in sheet[head]:
            cell.font = Font(bold=True)
        for row in rows:
            sheet.append(list(row))
        sheet.freeze_panes = f"A{head + 1}"
        for index, column in enumerate(columns, start=1):
            values = [str(column)] + [str(r[index - 1]) for r in rows
                                      if index - 1 < len(r) and r[index - 1] is not None]
            sheet.column_dimensions[get_column_letter(index)].width = min(
                48, max(10, max(len(v) for v in values) + 2))
    path = Path(path)
    book.save(path)
    return path


#: kind -> its title: the charts this exporter draws. Payloads:
#: * "sqd_bubbles": ``{"data": BubblePlotData, "colours": {condition: hex}}``;
#: * "sqd_replicates": ``{"payload": replicate fit payload, "colour": hex or None,
#:   "condition": name}`` -- the SQD page's replicate-days drill-in.
#: * "msd_overlay" / "msd_condition": ``{"entries": [{"key", "name", "colour",
#:   "result"}]}`` -- `msd_view.compute_msd_view`'s results, every condition on shared axes
#:   or one condition with its per-track curves.
#: * "appd_bubbles": ``{"data": BubblePlotData, "colours"}`` -- the gamma-mixture fit.
#: * "appd_histogram" / "appd_checks": ``{"entries": [{"key", "name", "colour", "result", "selected"}]}`` -- ONE
#:   condition's `appd_view.appd_condition` result (its Distributions card, with the fits its Show draws -- a tuple
#:   since 3 Nov 2026, an int before -- and its ``"components"`` and ``"residuals"`` boxes; its Checks card).
#: * "appd_overview": the same entries, one per condition (the Distributions page's All conditions view, 3 Nov 2026).
#: * "appd_cells": ``{"cells": appd_view.cells_payload(...), "labels", "colours"}`` -- the Per cell chart.
# -- a table --------------------------------------------------------------------- #
#: The gap between a table's columns, in em of its text: the least, and the usual one
#: headings are wrapped to keep. Extra width all goes to the gaps, so the rules run the
#: figure's whole width (a table stopping short left a white band beside it).
_TABLE_GAP_EM = (0.6, 1.2)
#: Cells that mean "nothing here" and decide nothing about a column's alignment.
_EMPTY_CELLS = {"", "—", "–", "-", "n/a", "NA", "nan", "None"}


def _table_content(payload: dict) -> tuple:
    """``(columns, rows)`` as the table SHOWS them: text, one list per row, padded."""
    columns = [str(c) for c in payload.get("columns") or ()]
    rows = [[("" if v is None else str(v)) for v in row] for row in payload.get("rows") or ()]
    rows = [(row + [""] * len(columns))[:len(columns)] for row in rows]
    return columns, rows


#: A number at the start of a cell: "1,984", "0.0673", "37.6 %", "-2", "<0.01", "3e-4".
_LEADING_NUMBER = re.compile(
    r"[<>≤≥~≈]?\s*[-+\u2212]?(?:\d{1,3}(?:,\d{3})+|\d+)?(?:\.\d+)?(?:[eE][-+]?\d+)?\s*%?")


def _numeric_text(text: str) -> bool:
    """Whether a cell reads as a number (aligned right): it STARTS with one, and whatever
    follows is a spread, a range or a bracket ("37.6 % ± 2.1", "1,001–1,501", "0 (0 %)").
    A name that merely begins with digits ("20260923_B.subtilis...") is a word: the first
    rule read it as a number and right-aligned a column of names."""
    s = text.strip()
    match = _LEADING_NUMBER.match(s)
    if not match or not re.search(r"\d", match.group(0)):
        return False
    rest = s[match.end():].strip()
    return not rest or rest[0] in "±(–-/×"


_ADVANCE_RENDERER = {}
_UNHINTED = []


def _width_pt(text: str, prop, dpi: float = 100.0) -> float:
    """The widest line of ``text`` in points as matplotlib DRAWS it in font ``prop``, at whichever of the two
    resolutions that matter draws it WIDER: ``dpi`` (100, where `check_figure` judges and the font is HINTED) and the
    font's own unhinted metrics (a PDF, an SVG, and a 300 or 600 dpi raster, within 1.3 %). Hinting at 100 dpi is not
    one-sided: MEASURED 30 Oct 2026 over eight titles and labels, it drew Arial 7 pt 3-11 % WIDER than the PDF and
    Arial 6 pt 3-14 % NARROWER -- "mNeonGreen, measured · 2 populations" was 34.3 mm at 100 dpi (inside its 34.5 mm
    plot, `check_figure` clean) and 37.8 mm in the PDF and the 600 dpi PNG, cut at the figure's edge."""
    return max(_hinted_pt(text, prop, dpi), _own_pt(text, prop))


def _hinted_pt(text: str, prop, dpi: float = 100.0) -> float:
    """The widest line of ``text`` as the Agg renderer draws it at ``dpi``, hinted, in points."""
    from matplotlib.backends.backend_agg import RendererAgg

    renderer = _ADVANCE_RENDERER.get(dpi)
    if renderer is None:
        renderer = _ADVANCE_RENDERER[dpi] = RendererAgg(10, 10, dpi)
    return max((float(renderer.get_text_width_height_descent(line, prop, ismath=False)[0]) * 72.0 / dpi
                for line in str(text).split("\n") if line), default=0.0)


def _own_pt(text: str, prop) -> float:
    """The widest line of ``text`` by the font's own metrics, unhinted, in points: what a PDF and an SVG draw."""
    from matplotlib.textpath import TextToPath

    if not _UNHINTED:
        _UNHINTED.append(TextToPath())
    return max((float(_UNHINTED[0].get_text_width_height_descent(line, prop, ismath=False)[0])
                for line in str(text).split("\n") if line), default=0.0)


def _advance_pt(text: str, size_pt: float, family: str, weight: str = "normal") -> float:
    """The width matplotlib DRAWS ``text`` at, in points (`_width_pt`: the wider of 100 dpi and the PDF's metrics).
    `_text_width_pt` measures the INK and read a table's cells 7-18 % narrow, which cut its last column at the
    figure's edge; a table is laid out with this instead. Until 30 Oct 2026 this measured at 100 dpi alone, thought
    to be where hinting draws widest -- true at 7 pt, 14 % narrow at 6 pt."""
    from matplotlib.font_manager import FontProperties

    if not text:
        return 0.0
    return _width_pt(text, FontProperties(family=family, size=size_pt, weight=weight))


def _table_align(columns, rows) -> list:
    out = []
    for j in range(len(columns)):
        cells = [r[j].strip() for r in rows if r[j].strip() not in _EMPTY_CELLS]
        out.append("right" if cells and all(_numeric_text(c) for c in cells) else "left")
    return out


def _table_layout(payload: dict, spec: FigureSpec, family: str, top: float = 0.0) -> dict:
    """Every size in a table, in mm, from the figure's width and the text: each column as
    wide as its widest cell, its heading on as FEW lines as the width allows (never cut
    inside a word, never elided), the gaps spread so the rules run the width of the figure,
    up to four em; a table wider than the figure is NAMED. The height follows from the
    rows."""
    columns, rows = _table_content(payload)
    if not columns:
        raise FigureExportError("this table has no columns to draw")
    pt = spec.pt()
    em = pt * _PT_MM
    measure = lambda s: _advance_pt(s, pt, family)                              # noqa: E731
    measure_bold = lambda s: _advance_pt(s, pt, family, "bold")                 # noqa: E731
    n = len(columns)
    cell_w = [max([measure(r[j]) * _PT_MM for r in rows] + [0.0]) for j in range(n)]
    side = 0.5
    room = spec.width_value() - 2 * side
    least, usual = (k * em for k in _TABLE_GAP_EM)

    def wrap(j, width_pt):
        # never elided and never cut inside a word: a heading takes the lines it needs
        return chart_text.wrap_text(columns[j], measure_bold, width_pt + 0.5, max_lines=99)

    def narrowest(j, lines):
        """The narrowest width (pt) at which heading ``j`` takes ``lines`` lines or fewer."""
        hi = measure_bold(columns[j])
        lo = max([measure_bold(word) for word in columns[j].split()] + [0.0])
        if lines <= 1 or len(wrap(j, lo)) <= lines:
            return hi if lines <= 1 else lo
        for _ in range(20):
            mid = (lo + hi) / 2.0
            lo, hi = (lo, mid) if len(wrap(j, mid)) <= lines else (mid, hi)
        return hi

    # THE FEWEST HEADING LINES THAT FIT: every heading on one line if the usual gaps allow
    # it, else two, else three; four at the least gaps is the last resort.
    for lines_wanted, gap_floor in ((1, usual), (2, usual), (3, usual), (4, least)):
        col_w = [max(cell_w[j], narrowest(j, lines_wanted) * _PT_MM, em) for j in range(n)]
        if sum(col_w) + gap_floor * (n - 1) <= room:
            break
    # Width left over after the usual gaps goes to the heading it saves a line for most
    # cheaply, one line at a time.
    while n > 1:
        spare = room - sum(col_w) - usual * (n - 1)
        offers = []
        for j in range(n):
            have = len(wrap(j, col_w[j] / _PT_MM))
            if have > 1:
                cost = narrowest(j, have - 1) * _PT_MM - col_w[j]
                if 0 < cost <= spare:
                    offers.append((cost, j, have - 1))
        if not offers:
            break
        cost, j, _lines = min(offers)
        col_w[j] += cost
    findings = []
    gap = 0.0 if n == 1 else max(least, (room - sum(col_w)) / (n - 1))
    if n == 1:
        col_w[0] = max(col_w[0], room)
    width = sum(col_w) + gap * (n - 1)
    if width > room + 0.5:
        findings.append(f"the table needs {math.ceil(width + 2 * side)} mm at {pt:g} pt but "
                        f"the figure is {spec.width_value():g} mm wide; widen the figure, "
                        "use a smaller font or leave out columns")
    heads = [wrap(j, col_w[j] / _PT_MM) for j in range(n)]
    lines = max(len(h) for h in heads)
    leading = pt * 1.2 * _PT_MM
    row_h = pt * 1.6 * _PT_MM
    title = _title_lines(spec, family, room)
    title_top = top
    top += len(title) * leading + (0.8 if title else 0.0)
    y_top = top + 0.6
    y_mid = y_top + 0.45 * em + lines * leading + 0.35 * em
    y_body = y_mid + 0.2 * em
    y_bottom = y_body + len(rows) * row_h + 0.2 * em
    natural = round(y_bottom + 0.8, 1)
    if spec.height_mm is not None:
        height = float(spec.height_mm)
        if natural > height + 0.5:
            findings.append(f"the table needs {math.ceil(natural)} mm at this width and font, so at "
                            f"{height:g} mm its last rows are cut; use automatic height")
    else:
        height = float(min(600.0, max(10.0, natural)))
    xs, x = [], side
    for w in col_w:
        xs.append(x)
        x += w + gap
    return {"columns": columns, "rows": rows, "heads": heads, "col_x": xs, "col_w": col_w,
            "align": list(payload.get("align") or _table_align(columns, rows)),
            "x0": side, "x1": side + width, "y_top": y_top, "y_mid": y_mid,
            "y_body": y_body, "y_bottom": y_bottom, "row_h": row_h, "leading": leading,
            "height": height, "findings": tuple(findings), "top": top, "title": title,
            "title_top": title_top}


def _draw_table(fig, payload: dict, spec: FigureSpec, family: str, lay: dict) -> tuple:
    """A table in print, journal style: a heavy rule above and below, a light one under
    the headings, no lines between rows, numbers aligned right. Text stays text."""
    from matplotlib.lines import Line2D

    page = _page(spec)
    width_mm, height_mm = spec.size_mm()
    pt = spec.pt()

    def at(x, y):
        return x / width_mm, 1.0 - y / height_mm

    for k, title_line in enumerate(lay.get("title", ())):
        fig.text(*at((lay["x0"] + lay["x1"]) / 2.0,
                     lay["title_top"] + (k + 0.85) * lay["leading"]), title_line,
                 ha="center", va="baseline", fontsize=pt, fontweight="bold", color=page["ink"])
    light = max(MIN_LINE_PT, spec.line_pt())
    heavy = max(light * 1.6, MIN_LINE_PT)
    for y, lw in ((lay["y_top"], heavy), (lay["y_mid"], light), (lay["y_bottom"], heavy)):
        fig.add_artist(Line2D([at(lay["x0"], y)[0], at(lay["x1"], y)[0]], [at(0, y)[1]] * 2,
                              transform=fig.transSubfigure, color=page["ink"], lw=lw,
                              solid_capstyle="butt"))
    # Each heading line on a SHARED BASELINE, the last line of every heading on the same
    # one: aligned by its box, a heading with a descender ("Days") sat higher than the rest.
    base = lay["y_mid"] - 0.6 * pt * _PT_MM
    for j, lines in enumerate(lay["heads"]):
        right = lay["align"][j] == "right"
        x = lay["col_x"][j] + (lay["col_w"][j] if right else 0.0)
        for k, line in enumerate(reversed(lines)):
            fig.text(*at(x, base - k * lay["leading"]), line,
                     ha="right" if right else "left", va="baseline", fontsize=pt,
                     fontweight="bold", color=page["ink"])
    for i, row in enumerate(lay["rows"]):
        y = lay["y_body"] + (i + 0.5) * lay["row_h"]
        if y > height_mm:
            break
        for j, text in enumerate(row):
            if not text:
                continue
            right = lay["align"][j] == "right"
            x = lay["col_x"][j] + (lay["col_w"][j] if right else 0.0)
            fig.text(*at(x, y), text, ha="right" if right else "left", va="center_baseline",
                     fontsize=pt, color=page["ink"])
    return _table_source(payload), (), table_legend(payload), _table_notes(payload)


def _cell_number(text: str):
    """A cell as a number when it is one ("1,984" -> 1984, "37.6 %" -> 37.6), else None."""
    s = text.strip().replace("\u2212", "-").replace(",", "").replace("\u2009", "")
    s = s.replace("\u202f", "").replace(" ", "")
    if s.endswith("%"):
        s = s[:-1]
    try:
        value = float(s)
    except ValueError:
        return None
    return int(value) if value.is_integer() and "." not in s and "e" not in s.lower() else value


def _table_source(payload: dict) -> tuple:
    """The table's numbers AS NUMBERS: a cell that is a number is written as one, a column
    of percentages loses its sign and says "(%)" in its heading, anything else is text."""
    columns, rows = _table_content(payload)
    out_cols, out_rows = [], [list(r) for r in rows]
    for j, name in enumerate(columns):
        cells = [r[j].strip() for r in rows if r[j].strip() not in _EMPTY_CELLS]
        numbers = [_cell_number(c) for c in cells]
        if cells and all(v is not None for v in numbers):
            percent = all(c.endswith("%") for c in cells)
            out_cols.append(f"{name} (%)" if percent and "%" not in name else name)
            for r in out_rows:
                v = r[j].strip()
                r[j] = _cell_number(v) if v not in _EMPTY_CELLS else ""
        else:
            out_cols.append(name)
    return out_cols, out_rows


def _table_notes(payload: dict) -> tuple:
    notes = ["The table as shown. Numbers are written as numbers, and percentage columns have no % sign."]
    if payload.get("caption"):
        notes.append(str(payload["caption"]))
    return tuple(notes)


def table_fits(payload: dict, spec: FigureSpec) -> bool:
    """Whether the table fits ``spec``'s width at its font."""
    family, _note = resolve_font(spec.font)
    return not _table_layout(payload, replace(spec, height_mm=None), family)["findings"]


def roomy_table_spec(payload: dict, spec: FigureSpec) -> FigureSpec:
    """The settings a table NEVER EXPORTED BEFORE starts from: ``spec``, or the journal's
    full width when the table does not fit the column (the Sample size needs 103 mm at
    7 pt). A width set by hand, or remembered for this table, is never changed here."""
    if spec.width in ("custom", "double") or table_fits(payload, spec):
        return spec
    return with_changes(spec, width="double", height_mm=None)


def table_legend(payload: dict) -> str:
    """A drafted legend for a table: its title, what it is of, and its caption."""
    title = str(payload.get("title") or "Table")
    of = f" ({payload['condition']})" if payload.get("condition") else ""
    caption = str(payload.get("caption") or "").strip()
    text = f"{title}{of}."
    if caption:
        text += " " + caption.rstrip(".") + "."
    return text


KINDS = {"sqd_bubbles": "SQD populations", "sqd_replicates": "SQD replicate fits",
         "msd_overlay": "Ensemble MSD", "msd_condition": "MSD",
         "appd_bubbles": "Apparent-D populations",
         "appd_histogram": "Apparent-D distribution", "appd_cells": "Apparent-D per cell",
         "appd_checks": "Apparent-D checks", "appd_overview": "Apparent-D, every condition",
         "msd_checks": "MSD checks", "msd_v2": "MSD beside v2",
         "clusters_condition": "Clustering",
         "clusters_shares": "Clustering, shares across conditions",
         "spatial_profiles": "Where the jumps sit", "appd_cell_length": "Slow share against cell length",
         "sqd_crossed": "SQD fraction against D",
         "kinetics_condition": "Triggered response",
         "kinetics_populations": "Populations over time",
         "kinetics_overview": "Populations over time, every condition",
         "kinetics_mock": "Against the mock addition",
         "hmm_condition": "HMM state diagram",
         "hmm_comparison": "HMM across conditions",
         "hmm_overview": "HMM state diagrams, every condition",
         "hmm_angles": "HMM jump angles",
         "jumps_condition": "Jump distances",
         "jumps_overview": "Jump distances, every condition",
         "jd_condition": "Signed jumps",
         "qc_length": "Track-length distribution", "qc_neighbour": "Neighbor share",
         "qc_outliers": "Per-unit outliers", "qc_incell": "Tracks inside a cell outline",
         "spatial_maps": "Spatial map", "cell_image": "Cell image", "table": "Table"}
#: * "hmm_condition": ``{"entries": [{"key", "name", "colour", "result"}]}`` -- ONE
#:   condition's `hmm_view.compute_hmm_view` result, the HMM card's.
#: * "hmm_overview": ``{"entries": [... as "hmm_condition"'s, one per condition]}`` -- every card's state diagram in
#:   one figure (5 Nov 2026), in the section's order, the conditions the comparison card's eyes show.
#: * "hmm_comparison": `hmm_view.comparison_payload` as the section's "Across conditions"
#:   card holds it (``{"panels", "conditions", "caption"}``) -- every condition's slow state.
#: * "hmm_angles": `hmm_angles_view.export_payload` as the HMM section's Jump angles page holds it (22 Nov 2026): the
#:   strip across conditions, each condition's histograms, the Welch tables.
#: * "jumps_condition": ``{"entries": [{"key", "name", "colour", "result", "selected"}]}``
#:   -- ONE condition's `jumps_view.compute_jump_view` result and the fits its card shows
#:   (``selected``, a tuple of counts; an old single count or "all" still reads; `FigureSpec.chosen_fit`
#:   overrides it), drawn with the residual plots `FigureSpec.pearson_residuals` / `.ecdf_residuals` keep; the
#:   entry's ``"legend_at"`` (where the card's key was dragged; None: above the plots) puts the key inside.
#: * "jumps_overview": ``{"entries": [... as "jumps_condition"'s, one per condition], "layout", "order"}`` -- the
#:   Jump distances page's All conditions view: every histogram on one x axis, each as its card shows it, one plot
#:   per condition or per fit (``"layout"``: "condition" | "fit", the conditions ``"order"`` "down" | "across";
#:   one plot per condition when absent). Each entry's ``"components"`` is its card's Components box. The residual
#:   plots under every plot only when asked (`FigureSpec.pearson_residuals` / `.ecdf_residuals`, off by default here).
#: Kinds with no y axis: the dialog's Y title is off (the X title names the colour bar).
NO_Y_TITLE = frozenset({"spatial_maps", "cell_image"})
#: Kinds with no axis at all: the dialog shows neither title.
NO_TITLES = frozenset({"table", "hmm_condition", "hmm_comparison", "hmm_overview", "hmm_angles"})

_NUMBERS, _GRID, _LEGEND = ("numbers", "Numbers"), ("grid", "Grid"), ("legend", "Legend")
#: The bubble charts' sample under each column (`FigureSpec.sample_n`), named as asked: "a Show box ... ("n")".
_SAMPLE = ("sample_n", "n")
#: The drafted legend's sentence for it.
SAMPLE_SENTENCE = ("Under each column: the sample, meaning the replicates the spread is over (movies or cells where "
                   "the interval resamples them, or the replicates pooled where no spread is drawn) and the "
                   "tracks fitted")
#: Its end where the conditions differ in Force same D across (`bubble_view.mixed_scopes`, 1 Nov 2026), in the
#: words written under the columns (`workspace_view.SCOPE_SAMPLE_WORDS`).
def _sample_scope_clause() -> str:
    from .workspace_view import SCOPE_SAMPLE_WORDS

    return (f", then how its D was fitted ({SCOPE_SAMPLE_WORDS['condition']}: one D shared by the condition's "
            f"replicates; {SCOPE_SAMPLE_WORDS['replicate']}: each replicate its own)")


SAMPLE_SCOPE_CLAUSE = _sample_scope_clause()


def _has_samples(payload) -> bool:
    """Whether a bubble chart's payload carries the sample under its columns (a Run made before does not)."""
    data = payload.get("data") if isinstance(payload, dict) else None
    return bool(getattr(data, "samples", None))
#: The jump-distance figure's two residual plots, named as its card names them (`FigureSpec.pearson_residuals`).
_PEARSON, _ECDF = ("pearson_residuals", "Pearson residuals"), ("ecdf_residuals", "ECDF residuals")
#: The jump-distance figures' components (`FigureSpec.components`; the card's own box until ticked here).
_COMPONENTS = ("components", "Components")


def _has_key(kind: str, payload) -> bool:
    """Whether this chart draws a key at all, so a Legend box would change something."""
    def get(name):
        return (payload.get(name) if isinstance(payload, dict)
                else getattr(payload, name, None))

    if kind in ("sqd_bubbles", "sqd_replicates", "appd_bubbles"):
        return bool(get("series_labels"))
    if kind == "msd_overlay":
        return len(get("entries") or ()) > 1
    if kind == "msd_condition":
        return False
    return True


#: The axes whose start and end the Export dialog offers, per kind: a categorical axis
#: (conditions, movies, cells in rank) has none.
_RANGE_AXES = {"sqd_bubbles": ("y",), "sqd_replicates": ("y",), "appd_bubbles": ("y",),
               "appd_histogram": ("y", "x"), "appd_checks": ("y", "x"), "appd_cells": ("y",), "appd_overview": ("x",),
               "msd_checks": ("y",), "msd_v2": ("y", "x"), "clusters_condition": ("y", "x"),
               "clusters_shares": ("y",), "spatial_profiles": ("y",), "appd_cell_length": ("y", "x"), "sqd_crossed": ("y", "x"),
               "msd_overlay": ("y", "x"),
               "msd_condition": ("y", "x"), "kinetics_condition": ("y", "x"), "kinetics_populations": ("x",),
               "kinetics_overview": ("x",), "kinetics_mock": ("y",),
               "jumps_condition": ("y", "x"), "jumps_overview": ("x",), "jd_condition": ("y", "x"),
               "qc_length": ("y", "x"), "qc_neighbour": ("y",), "qc_outliers": ("y",),
               "qc_incell": ("y",)}
#: Kinds laid out by hand in mm (no matplotlib axes): their title is part of the layout.
_MANUAL_KINDS = ("spatial_maps", "table")


def axis_ranges(kind: str) -> tuple:
    """``("y",)``, ``("y", "x")`` or ``()``: the axes of ``kind`` whose range can be set."""
    return _RANGE_AXES.get(kind, ())


def parse_number(text):
    """A typed axis end: ``""`` is None (the chart's own), "0,01" reads as 0.01; anything
    else raises ValueError naming the text."""
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        return float(raw.replace(",", ".").replace("\u2212", "-"))
    except ValueError:
        raise ValueError(f"“{raw}” is not a number") from None


def _apply_ranges(ax, spec: FigureSpec, kind: str) -> None:
    """Set the axis ends typed in the dialog (None keeps the chart's own), and re-tick a
    log axis whose ticks were fixed for the old ends. A log axis starting at or below 0 is
    refused by name."""
    from matplotlib.ticker import FixedLocator

    for axis in axis_ranges(kind):
        want = (spec.y_min, spec.y_max) if axis == "y" else (spec.x_min, spec.x_max)
        if want == (None, None):
            continue
        get, put = (ax.get_ylim, ax.set_ylim) if axis == "y" else (ax.get_xlim, ax.set_xlim)
        scale = ax.get_yscale() if axis == "y" else ax.get_xscale()
        lo, hi = get()
        lo = want[0] if want[0] is not None else lo
        hi = want[1] if want[1] is not None else hi
        if hi <= lo:
            raise FigureExportError(f"the {axis} axis would start at {lo:g} and end at "
                                    f"{hi:g}; it must start below where it ends")
        if scale == "log" and lo <= 0:
            raise FigureExportError(f"the {axis} axis is logarithmic, so it must start "
                                    "above 0")
        put(lo, hi)
        target = ax.yaxis if axis == "y" else ax.xaxis
        if scale == "log" and isinstance(target.get_major_locator(), FixedLocator):
            ticks = chart_text.log_ticks(lo, hi)
            target.set_major_locator(FixedLocator(ticks))
            target.set_ticklabels([f"{v:g}" for v in ticks])


def _title_lines(spec: FigureSpec, family: str, width_mm: float) -> list:
    """The title wrapped to ``width_mm``: for the kinds laid out by hand."""
    if not spec.title:
        return []
    measure = lambda s: _advance_pt(s, spec.pt(), family, "bold")          # noqa: E731
    return chart_text.wrap_text(spec.title, measure, width_mm / _PT_MM, max_lines=4)


def show_options(kind: str, payload=None) -> tuple:
    """The Export dialog's Show boxes for this chart: ``((spec field, label), ...)``, ONLY
    the ones that change it. Asked: "what is grid and legend? it doesnt remove anything" --
    a box that removes nothing reads as broken. The cell image has no numbers, grid or
    key; its boxes are its scale bar and its colour bar (`legend`)."""
    if kind == "cell_image":
        return (("scale_bar", "Scale bar"), ("legend", "Colour bar"))
    if kind == "table":
        return ()
    if kind == "spatial_maps":
        # Asked 16 Oct 2026: "a disable option for the scale bar, color map etc". Each removes its room too.
        return (_NUMBERS, ("scale_bar", "Scale bar"), ("legend", "Colour bar"), ("headings", "Labels"))
    if kind == "kinetics_condition":
        return (_NUMBERS, _GRID)
    if kind in ("kinetics_populations", "kinetics_overview"):
        # Numbers: the verdict's first line on the D panel; Legend: the populations' key; with two populations or
        # more, the slowest's D panel and the other populations' shares (8 Nov 2026, `FigureSpec.slowest_d`)
        rows = ((("slowest_d", "Slowest D"), ("other_shares", "Other shares"))
                if _populations_of(payload) >= 2 else ())
        half = ((("half_time", "Half-time"),) if any(((e or {}).get("result") or {}).get("half_time")
                                                      for e in (payload or {}).get("entries") or ()) else ())
        # the conditions as rows (8 Nov 2026), offered on the every-condition figure
        across = ((("conditions_as_rows", "Conditions as rows"),) if kind == "kinetics_overview" else ())
        return (_NUMBERS, _GRID, _LEGEND) + rows + half + across
    if kind == "kinetics_mock":
        # Numbers: the change beyond the mock over each panel
        return (_NUMBERS, _GRID)
    if kind in ("hmm_condition", "hmm_overview"):
        # No grid on a diagram; Numbers are the labels (D, occupancy, rates), Legend the
        # line under it saying what the area and the arrows are (under the overview's grid, once); "± on rates" where
        # a spread was computed (`FigureSpec.rate_spread`, off), "± on occupancy" likewise (`occupancy_spread`)
        return ((_NUMBERS, _LEGEND) + ((("rate_spread", "± on rates"),) if _hmm_has_spread(payload) else ())
                + ((("occupancy_spread", "± on occupancy"),) if _hmm_has_occupancy_spread(payload) else ()))
    if kind == "hmm_angles":
        # Legend: the keys over the strip and over the histograms
        return (_GRID, _LEGEND)
    if kind == "hmm_comparison":
        # No number is printed on it (the source data hold them); Legend is the key over
        # the panels saying what a circle, a bar and a band are; "Pair lines" where two conditions share a pair tag
        # and "Dwell time" (12 Nov 2026) wherever the payload carries the dwell panel
        return ((_GRID, _LEGEND) + ((("pair_lines", "Pair lines"),) if (payload or {}).get("paired") else ())
                + ((("hmm_dwell", "Dwell time"),) if (payload or {}).get("dwell_panel") else ()))
    if kind == "jumps_condition":
        # Asked 30 Oct 2026: "also make it optional to export ecdf and residuals in export window and in composer",
        # and the components ("maybe we can make this optional").
        return (_NUMBERS, _GRID, _LEGEND, _COMPONENTS, _PEARSON, _ECDF)
    if kind in ("sqd_bubbles", "appd_bubbles"):
        return ((_NUMBERS, _GRID) + ((_LEGEND,) if _has_key(kind, payload) else ())
                + ((_SAMPLE,) if _has_samples(payload) else ()))
    if kind == "jumps_overview":
        # the residual plots too, under every tile (30 Oct 2026: "maybe we include it in exporter for every jump
        # distance as an option? even for each/all condition etc windows"), off until ticked (`RESIDUALS_BY_DEFAULT`)
        return (_NUMBERS, _GRID, _LEGEND, _COMPONENTS, _PEARSON, _ECDF)
    if kind in ("appd_histogram", "appd_overview"):
        # the apparent-D distributions as the jump figures (3 Nov 2026): each fit's populations and both residual
        # plots optional (`_appd_strips`)
        return (_NUMBERS, _GRID, _LEGEND, _COMPONENTS, _PEARSON, _ECDF)
    return (_NUMBERS, _GRID) + ((_LEGEND,) if _has_key(kind, payload) else ())


def jumps_y_title(kind: str, payload, spec: FigureSpec) -> str:
    """A jump-distance figure's own y axis title, in the unit it draws (`shown_y_mode`), with the bin width."""
    from .jumps_view import bin_width, y_label

    entries = (payload or {}).get("entries") or []
    widths = [bin_width(e["result"]["edges_nm"]) for e in entries if "edges_nm" in (e.get("result") or {})]
    return y_label(shown_y_mode(kind, payload, spec) or "probability", widths)


def y_mode_choices(kind: str) -> tuple:
    """The Export dialog's Y unit row for the jump-distance figures: ``((mode, words), ...)``; ``()`` elsewhere."""
    from .jumps_view import Y_MODE_LABELS, Y_MODES

    return tuple(zip(Y_MODES, Y_MODE_LABELS)) if kind in Y_MODE_KINDS else ()


#: The most columns the Grid row offers, and a spec may hold (`FigureSpec.grid_columns`).
GRID_MAX_COLUMNS = 6
#: Kinds whose panels stand in a grid the Export dialog can set (`grid_choices`).
GRID_KINDS = ("hmm_overview", "jumps_overview")


def grid_choices(kind: str, payload, spec: FigureSpec | None = None) -> tuple:
    """The Export dialog's Grid row (5 Nov 2026): ``((columns, words), ...)`` -- Auto (None), then every number of
    columns up to the conditions drawn (at most `GRID_MAX_COLUMNS`; those ``spec`` leaves out not counted), named
    "rows × columns" ("2 × 2" for four conditions in two columns); ``()`` for a kind without a grid or with one."""
    if kind not in GRID_KINDS:
        return ()
    if kind == "jumps_overview" and ((payload or {}).get("layout") or "condition") != "condition":
        return ()                         # one plot per fit: its grid is the conditions by the fits
    left_out = set(spec.left_out) if spec is not None else set()
    n = len([e for e in (payload or {}).get("entries") or [] if e.get("name") not in left_out])
    if n < 2:
        return ()
    return ((None, "Auto"),) + tuple((c, f"{math.ceil(n / c)} × {c}") for c in range(1, min(n, GRID_MAX_COLUMNS) + 1))


def shown_y_mode(kind: str, payload, spec: FigureSpec):
    """The unit the figure draws (the dialog's row starts on it): the spec's, else the card's (Export all: the
    cards' when they agree, else the standard); None for any other kind."""
    if kind not in Y_MODE_KINDS:
        return None
    entries = (payload or {}).get("entries") or []
    if kind == "jumps_condition" and entries:
        return _y_mode_of(entries[0], spec)
    return _overview_y_mode(entries, spec)


def show_checked(kind: str, payload, spec: FigureSpec, key: str) -> bool:
    """Whether the Export dialog's Show box ``key`` starts ticked: the spec's value -- except Components while the
    spec follows the cards (None): ticked when the card's box is (every card's, on the overview)."""
    value = getattr(spec, key)
    if key == "components" and value is None:
        entries = (payload or {}).get("entries") or [] if isinstance(payload, dict) else []
        return all(bool(e.get("components", True)) for e in entries)
    if key in ("pearson_residuals", "ecdf_residuals"):
        if kind in ("appd_histogram", "appd_overview"):
            # an apparent-D card's figure starts as its card shows them (`_appd_strips`)
            entries = (payload or {}).get("entries") or [] if isinstance(payload, dict) else []
            return key.split("_")[0] in _appd_strips(spec, kind, entries)
        # the figure's own default until the box is ticked (`RESIDUALS_BY_DEFAULT`: the card's on, Export all's off)
        return residuals_on(spec, kind, key.split("_")[0])
    return bool(value)


def default_titles(kind: str, payload: dict | None = None) -> tuple:
    """``(y, x)``: the chart's own axis titles, as the screen shows them. The per-unit QC
    chart's depend on what it draws (`payload`): the quantity flagged, and the unit."""
    if kind == "spatial_maps":
        return "", _spatial_unit(payload or {})
    if kind in NO_TITLES:
        return "", ""
    if kind == "cell_image":
        return "", ("apparent D (µm²/s)" if (payload or {}).get("colour_mode") == "appd"
                    else "")
    if kind == "qc_length":
        return chart_text.QC_TRACKS_LABEL, chart_text.QC_STEPS_LABEL
    if kind == "qc_neighbour":
        return chart_text.QC_NEIGHBOUR_LABEL, ""
    if kind == "qc_incell":
        return chart_text.QC_TRACKS_LABEL, ""
    if kind == "qc_outliers":
        from .qc_view import sorted_axis_title

        block = (payload or {}).get("block") or {}
        return (block.get("axis_label") or "value",
                sorted_axis_title(block.get("unit", "unit")))
    if kind == "msd_checks":
        return chart_text.D_LINEAR_LABEL, "lags fitted"
    if kind == "spatial_profiles":
        return "density", "from the nearer pole (nm)"
    if kind == "sqd_crossed":
        return chart_text.D_AXIS_LABEL, "fraction (%)"
    if kind == "appd_cell_length":
        return chart_text.SLOW_SHARE_CELL_LABEL, "cell length (µm)"
    if kind == "msd_v2":
        return chart_text.QC_TRACKS_LABEL, "per-track σ = √|b/4|, sign of b kept (nm)"
    if kind == "clusters_shares":
        return "Share of tracks (%)", ""
    if kind.startswith("msd") or kind == "clusters_condition":
        return chart_text.MSD_AXIS_LABEL, chart_text.LAG_AXIS_LABEL
    if kind in ("jumps_condition", "jumps_overview"):
        # the y axis in the unit the cards show (30 Oct 2026; Probability, the standard, when no card says)
        return jumps_y_title(kind, payload, FigureSpec()), chart_text.JUMP_DISTANCE_LABEL
    if kind == "jd_condition":
        return chart_text.JUMPS_COUNT_LABEL, "signed jump (nm)"
    if kind in ("appd_histogram", "appd_checks", "appd_overview"):
        return chart_text.QC_TRACKS_LABEL, chart_text.APPD_AXIS_LABEL
    if kind == "appd_cells":
        return chart_text.SLOW_SHARE_CELL_LABEL, ""
    if kind in ("kinetics_populations", "kinetics_overview"):
        entries = (payload or {}).get("entries") or [{}]
        return chart_text.D_AXIS_LABEL, chart_text.induction_time_label(entries[0].get("time_unit") or "s")
    if kind == "kinetics_mock":
        from .kinetics_mock_view import axis_title

        entries = [e for e in (payload or {}).get("entries") or () if ((e.get("result") or {}).get("mock") or {})]
        return axis_title(entries[0]["result"]["mock"] if entries else {"readout": "mean_D"}), ""
    if kind.startswith("kinetics"):
        return chart_text.SLOW_PCT_LABEL, chart_text.TRIGGER_TIME_LABEL
    return chart_text.D_AXIS_LABEL, ""


def column_names(kind: str, payload: dict) -> list:
    """The names under a figure's columns, before any renaming: one dialog field each."""
    if kind == "sqd_bubbles":
        return [str(c) for c in payload["data"].conditions]
    if kind == "sqd_replicates":
        return _replicate_columns(payload["payload"])
    if kind == "hmm_comparison":
        return [str(r["name"]) for r in _comparison_rows(payload)]
    if kind.startswith(("msd", "kinetics", "hmm", "jumps", "jd_", "clusters")) or kind in ("appd_histogram", "appd_checks",
                                                                                 "appd_overview"):
        return [str(e["name"]) for e in payload["entries"]]
    if kind in ("qc_neighbour", "qc_incell"):
        return _qc_movie_names(payload["block"])
    if kind == "spatial_maps":
        return [_row_label(payload, r) for r in _spatial_rows(payload)]
    if kind == "spatial_profiles":
        return [str(ln["name"]) for ln in payload["panels"][0]["payload"]["lines"]]
    if kind == "sqd_crossed":
        return [str(c) for c, _n in payload["crossed"]["legend_conditions"]]
    if kind == "appd_cell_length":
        labels = payload.get("labels") or {}
        return [str(labels.get(c, c)) for c in payload.get("points") or {}]
    if kind == "appd_cells":
        labels = payload.get("labels") or {}
        return [str(labels.get(col["condition"], col["condition"])) for col in payload["cells"]["columns"]]
    if kind == "appd_bubbles":
        return [str(c) for c in payload["data"].conditions]
    return []


#: `FigureSpec.record`'s mark: written by a version where "" means NO title (`from_dict`).
TITLE_RULE = 2
#: `FigureSpec.record`'s mark: written by a version where a residual plot's None is the figure's own default.
RESIDUAL_RULE = 2


def _condition_of(kind: str, payload) -> str:
    """The one condition a chart draws, by the name it is shown under, else ""."""
    if not isinstance(payload, dict):
        return ""
    label = payload.get("condition_label") or payload.get("condition")
    if label:
        return str(label)
    entries = payload.get("entries") or ()
    if kind != "msd_overlay" and len(entries) == 1 and isinstance(entries[0], dict):
        return str(entries[0].get("name") or entries[0].get("key") or "")
    return ""


def auto_title(kind: str, payload) -> str:
    """The title a figure starts with: the PROJECT's name -- the folder that holds its
    condition folders, e.g. "protein A" -- and for a chart of ONE condition (an HMM card, a
    jump-distance card, one MSD or kinetics curve, one condition's replicate days, its QC
    panels, its image) that condition's name after it: "protein A · Glucose". Asked for on
    13 Oct 2026 ("for SQD bubble plots ... protein A (folder name). for hmm ... protein A +
    Condition"). The window puts ``"project"`` (and ``"condition_label"``) in the payload;
    without them this is "" and the figure has no title."""
    project = str(payload.get("project") or "").strip() if isinstance(payload, dict) else ""
    if not project:
        return ""
    return " · ".join(part for part in (project, _condition_of(kind, payload)) if part)


def with_title(spec: FigureSpec, kind: str, payload) -> FigureSpec:
    """``spec`` with an automatic title (None) made concrete; a typed or empty one kept."""
    if spec.title is None:
        return replace(spec, title=auto_title(kind, payload))
    return spec


def _centre_title(fig, spec: FigureSpec | None = None) -> None:
    """The figure's title over the MIDDLE OF THE PLOT (asked for: "the title ... should be in
    the middle on top of the graph, not in the side"): the layout is run, every text wider
    than its room is set smaller or wrapped (`_fit_texts`, given ``spec``), then the title
    is moved to the centre of every visible axes' span (colour bars left out), which moves
    nothing else -- a title's sideways position takes no room in the layout."""
    title = getattr(fig, "_suptitle", None)
    engine = fig.get_layout_engine()
    if engine is not None:
        engine.execute(fig)
    elif _shown(title):
        # a figure that places its own axes (the HMM diagram): no engine pads the title
        title.set_y(1.0 - 0.6 / (fig.get_size_inches()[1] * 25.4))
    if spec is not None:
        _fit_texts(fig, spec)
    _title_over_plots(fig)


def _title_over_plots(holder) -> None:
    """``holder``'s title -- a whole figure's, or a composed panel's sub-figure's, its layout already run -- over
    the centre of every visible axes' span (colour bars left out), in the holder's OWN fractions: an axes'
    position inside a sub-figure is a fraction of that sub-figure, as the title's x is. It is KEPT INSIDE its
    holder (30 Oct 2026): a title too wide to stand over the plots' middle moves in from the edge rather than run
    past it, and on a composed panel it starts right of the letter (``holder._smtracker_keep_left``, a fraction of
    the holder) -- measured as it is drawn, after `_fit_texts` set it smaller or wrapped it."""
    title = getattr(holder, "_suptitle", None)
    if not _shown(title):
        return
    boxes = [ax.get_position() for ax in holder.axes
             if ax.get_visible() and not hasattr(ax, "_colorbar")]
    if not boxes:
        return
    x = (min(b.x0 for b in boxes) + max(b.x1 for b in boxes)) / 2.0
    dpi = float(getattr(holder, "dpi", 100.0))
    width = holder.bbox.width * 72.0 / dpi
    if width > 0:
        half = _width_pt(title.get_text(), title.get_fontproperties(), dpi) / 2.0 / width
        pad = _FIT_PAD_PT / width
        keep = float(getattr(holder, "_smtracker_keep_left", 0.0))
        lo, hi = keep + pad + half, 1.0 - pad - half
        x = min(max(x, lo), hi) if lo <= hi else (keep + 1.0) / 2.0
    title.set_x(x)
    title.set_horizontalalignment("center")


# -- text that fits its room (30 Oct 2026) ----------------------------------------- #
# Reported: "in 1 column view, the jump distance plots (2x2 for example) the title gets cut. make when something
# gets cut -> scale down text size". ONE RULE for every kind, standalone and composed, run after layout: a title wider
# than its panel, a plot title or axis title that is cut, collides with other text or squeezes its plot (reaches past
# everything else the plot carries, so the layout shrinks the plot: `_squeezing`), or a key that is cut or collides,
# is set smaller in `FIT_STEP_PT` steps down to the journal's minimum (`FigureSpec.min_pt`: Nature 5, Cell 6, eLife
# 7, poster 18 pt) and, only if it still does not fit there, wrapped at spaces (`chart_text.wrap_text`,
# `SPACE_BREAKS`) -- never cut, never run past its panel. Text that does none of these keeps its size. Its room: a
# figure's or a composed panel's title, the holder's width less its padding (and a panel's letter); a panel title or
# x title, its plot's width; a y title, its plot's height; a key, its panel, and its plot's width when it sits inside
# the plot. Sibling panel titles share one size (`_even_titles`). Text is measured AS THE PDF DRAWS IT where that is
# wider than at 100 dpi (`_width_pt`), and `check_figure` judges the same way and names every text set smaller (a
# note, `FigureNote`, not a warning).
#: How much smaller a text is set per step, in points.
FIT_STEP_PT = 0.5
#: A text still too wide at the floor is wrapped at spaces onto at most this many lines -- a ceiling never reached, so
#: nothing is elided; a WORD wider than the room stays whole on its own line ("Pea" / "rson" over a 6 mm residual
#: strip read worse than "Pearson" a millimetre past it), the one case a text may stand past its room.
FIT_MAX_LINES = 24
#: The room a title keeps from its holder's edges, in points: the layout's own padding (`_new_figure`'s w_pad).
_FIT_PAD_PT = 2.0
#: How many times texts are fitted and the layout run again (a wrapped y title takes width from the plots, which can
#: leave a title over them too wide in turn); it settles in one or two.
_FIT_PASSES = 3
#: What a fitted text is called in `check_figure`'s note, one and several.
_FIT_WORDS = {"title": ("title", "titles"), "panel title": ("panel title", "panel titles"),
              "axis title": ("axis title", "axis titles"), "key": ("key entry", "key entries"),
              "number": ("bubble number", "bubble numbers")}


class FigureNote(str):
    """A `check_figure` line that INFORMS rather than warns: a text set smaller or wrapped to fit its room. The
    Export dialog and the Composer show it as a note (•), not a problem (⚠)."""


def _shown(text) -> bool:
    return text is not None and text.get_visible() and bool(text.get_text().strip())


def _math(text) -> bool:
    """Mathtext (``$...$``): measured as plain text it would read far wider than it draws, so it is left alone."""
    try:
        return bool(text._preprocess_math(text.get_text())[1])
    except Exception:                                    # noqa: BLE001 - private API; plain text then
        return False


def _holders(fig) -> list:
    """``fig`` and every sub-figure in it, nested ones included."""
    out = [fig]
    for sub in getattr(fig, "subfigs", ()):
        out += _holders(sub)
    return out


def _text_rooms(fig) -> list:
    """``[(text or key, room in points, what)]``: every title, axis title and key of ``fig`` and its sub-figures,
    with the room it has after layout (a key's is decided where its box is known, `_fit_legend`). A holder laid out
    by hand in millimetres (a spatial map, a table: `_MANUAL_KINDS`) is left out: its own layout wrapped its text."""
    per_pt = fig.dpi / 72.0
    rooms = []
    for holder in _holders(fig):
        if getattr(holder, "_smtracker_kind", None) in _MANUAL_KINDS:
            continue
        width = holder.bbox.width / per_pt
        title = getattr(holder, "_suptitle", None)
        if _shown(title):
            keep = float(getattr(holder, "_smtracker_keep_left", 0.0)) * width
            rooms.append((title, width - keep - 2.0 * _FIT_PAD_PT, "title"))
        for legend in getattr(holder, "legends", ()):
            if legend.get_visible():
                rooms.append((legend, (None, holder), "key"))
    for ax in fig.get_axes():
        holder = ax.get_figure(root=False)
        if not ax.get_visible() or getattr(holder, "_smtracker_kind", None) in _MANUAL_KINDS:
            continue
        w, h = ax.bbox.width / per_pt, ax.bbox.height / per_pt
        for title in (ax.title, ax._left_title, ax._right_title):
            if _shown(title):
                rooms.append((title, w, "panel title"))
        if ax.axison:
            if ax.xaxis.get_visible() and _shown(ax.xaxis.label):
                # a title centred across several panels says how many panel widths it has (8 Nov 2026)
                span = float(getattr(ax.xaxis.label, "_smtracker_span", 1.0))
                rooms.append((ax.xaxis.label, w * span, "axis title"))
            if ax.yaxis.get_visible() and _shown(ax.yaxis.label):
                rooms.append((ax.yaxis.label, h, "axis title"))
        legend = ax.get_legend()
        if legend is not None and legend.get_visible():
            rooms.append((legend, (ax, holder), "key"))
    return rooms


def _fitted(text, size: float, what: str) -> None:
    """Remember what ``text`` was before it was first fitted, for `check_figure`'s note and a later pass."""
    if not hasattr(text, "_smtracker_fit_from"):
        text._smtracker_fit_from = float(size)
        text._smtracker_fit_text = text.get_text()
        text._smtracker_fit_what = what


def _fit_one(text, room: float, floor: float, what: str) -> bool:
    """``text`` made no wider than ``room`` points; True when it changed. IN THIS ORDER (30 Oct 2026, the
    experimenter's: "first try the SHORT form of a fit label ... then shrink, then wrap"): its SHORT form where it has
    one (`chart_text.short_form`: "1 population" -> "1-pop"; marked ``_smtracker_short``), then set smaller to
    ``floor``, then (at ``floor``) wrapped. Never larger than it is: a later pass only takes it further down."""
    if room <= 0 or _math(text):
        return False
    prop = text.get_fontproperties()
    size = prop.get_size_in_points()
    shown = text.get_text()

    def measure(s, at):
        p = prop.copy()
        p.set_size(at)
        return _width_pt(s, p)

    if measure(shown, size) <= room + 1e-6:
        return False
    original = getattr(text, "_smtracker_fit_text", shown)
    base = chart_text.short_form(original)
    floor = min(floor, size)
    if shown in (original, base):
        fitted, fits = chart_text.shrink_to_fit(base, measure, room, size, floor, FIT_STEP_PT)
        if fits:
            _fitted(text, size, what)
            if base != shown:
                text.set_text(base)
            text._smtracker_short = base != original
            text.set_fontsize(fitted)
            return True
    lines = chart_text.wrap_text(" ".join(base.split()), lambda s: measure(s, floor), room,
                                 max_lines=FIT_MAX_LINES, breaks=chart_text.SPACE_BREAKS, cut_words=False)
    wrapped = "\n".join(lines)
    if wrapped == shown and abs(size - floor) < 1e-6:
        return False
    _fitted(text, size, what)
    text._smtracker_short = base != original
    text.set_text(wrapped)
    text.set_fontsize(floor)
    return True


def _fit_legend(legend, where: tuple, floor: float, renderer, dpi: float) -> bool:
    """A key past its room set smaller in steps to ``floor``, then (one column) its entries wrapped. Its room: INSIDE
    its panel always (a figure, a composed panel's sub-figure), and no wider than the plot when it sits inside a plot
    (a key in a column of its own beside the plot, an axes with no axis, may be as wide as the panel lets it).
    Measured as drawn, plus what the PDF draws wider (`_width_pt`)."""
    ax, holder = where
    texts = [t for t in legend.get_texts() if _shown(t) and not _math(t)]
    if _shown(legend.get_title()) and not _math(legend.get_title()):
        texts.append(legend.get_title())
    if not texts:
        return False
    per_pt = dpi / 72.0
    columns = max(1, int(getattr(legend, "_ncols", 1) or 1))
    box = legend.get_window_extent(renderer)
    in_plot = (ax is not None and ax.axison
               and ax.bbox.contains((box.x0 + box.x1) / 2.0, (box.y0 + box.y1) / 2.0))

    def past() -> float:
        """How far the key runs past its room, in points (0 or less: it fits). Its TEXT against the panel's edges
        (the box's own padding, `borderpad`, is blank), its box against a plot it sits in."""
        props = [t.get_fontproperties() for t in texts]
        if columns == 1:
            # one column: its widest line as the PDF draws it, against its widest as drawn here
            excess = (max(_width_pt(t.get_text(), p, dpi) for t, p in zip(texts, props))
                      - max(_hinted_pt(t.get_text(), p, dpi) for t, p in zip(texts, props)))
        else:
            excess = columns * max(_width_pt(t.get_text(), p, dpi) - _hinted_pt(t.get_text(), p, dpi)
                                   for t, p in zip(texts, props))
        excess = max(0.0, excess) * per_pt
        pad = float(getattr(legend, "borderpad", 0.0) or 0.0) * float(getattr(legend, "_fontsize", 0.0)) * per_pt
        now, edge = legend.get_window_extent(renderer), holder.bbox
        over = max(edge.x0 - (now.x0 + pad), (now.x1 - pad + excess) - edge.x1)
        if in_plot:
            over = max(over, now.width + excess - ax.bbox.width)
        return over / per_pt

    if past() <= 1e-6:
        return False
    changed = False
    while past() > 1e-6:
        moved = False
        for t in texts:
            now = t.get_fontproperties().get_size_in_points()
            start = getattr(t, "_smtracker_fit_from", now)
            s = max(min(floor, start), now - FIT_STEP_PT)
            if s < now - 1e-6:
                _fitted(t, now, "key")
                t.set_fontsize(s)
                moved = changed = True
        if not moved:
            break
    over = past()
    if over > 1e-6 and columns == 1:
        # Still too wide at the floor: each entry wider than what the key leaves for text is wrapped.
        widest = max(_width_pt(t.get_text(), t.get_fontproperties(), dpi) for t in texts)
        for_text = widest - over
        for t in texts:
            prop = t.get_fontproperties()
            if for_text > 0 and _width_pt(t.get_text(), prop, dpi) > for_text + 1e-6:
                _fitted(t, prop.get_size_in_points(), "key")
                wrapped = "\n".join(chart_text.wrap_text(
                    " ".join(t._smtracker_fit_text.split()), lambda s, p=prop: _width_pt(s, p, dpi), for_text,
                    max_lines=FIT_MAX_LINES, breaks=chart_text.SPACE_BREAKS, cut_words=False))
                if wrapped != t.get_text():
                    t.set_text(wrapped)
                    changed = True
    return changed


def _even_titles(fig) -> bool:
    """SIBLING PANEL TITLES AT ONE SIZE AND IN ONE FORM: the plot titles of one holder (a figure, a composed panel)
    that were drawn at one size all take the smallest any of them needed -- fitted one by one, a 2 x 2 read 5.5, 5,
    5.5 and 5.5 pt -- and when one had to take its SHORT form (`_fit_one`) every sibling that has one takes it too,
    so a row never reads "1 population" beside "2-pop". True when one changed."""
    groups = {}
    for ax in fig.get_axes():
        holder = ax.get_figure(root=False)
        if not ax.get_visible() or getattr(holder, "_smtracker_kind", None) in _MANUAL_KINDS:
            continue
        for title in (ax.title, ax._left_title, ax._right_title):
            if _shown(title) and not _math(title):
                start = getattr(title, "_smtracker_fit_from", title.get_fontsize())
                groups.setdefault((id(holder), round(start, 2)), []).append(title)
    changed = False
    for titles in groups.values():
        if any(getattr(t, "_smtracker_short", False) for t in titles):
            for t in titles:
                original = getattr(t, "_smtracker_fit_text", t.get_text())
                short = chart_text.short_form(original)
                if getattr(t, "_smtracker_short", False) or short == original or "\n" in t.get_text():
                    continue
                _fitted(t, t.get_fontsize(), "panel title")
                t.set_text(short)
                t._smtracker_short = True
                changed = True
        low = min(t.get_fontsize() for t in titles)
        for t in titles:
            if t.get_fontsize() > low + 1e-6:
                _fitted(t, t.get_fontsize(), "panel title")
                t.set_fontsize(low)
                changed = True
    return changed


def _clashing(fig, drawn) -> set:
    """``id``s of the texts in ``drawn`` (`_drawn_texts`) that are CUT OR COLLIDE: past their panel (a figure, a composed
    panel's sub-figure) or the page, or over another text (`_clash`, `check_figure`'s test; a key's own entries
    never count against each other)."""
    from matplotlib.legend import Legend

    group = {}
    for legend in fig.findobj(Legend):
        for t in legend.get_texts() + [legend.get_title()]:
            group[id(t)] = id(legend)
    frame = fig.bbox
    out = set()
    for i, (t, box) in enumerate(drawn):
        holder = t.get_figure(root=False)
        if _past(box, frame) or (holder is not None and _past(box, holder.bbox)):
            out.add(id(t))
        for u, other in drawn[i + 1:]:
            if group.get(id(t), id(t)) != group.get(id(u), id(u)) and _clash(box, other):
                out.update((id(t), id(u)))
    return out


def _squeezing(fig, drawn) -> set:
    """``id``s of the plot titles and axis titles in ``drawn`` that SQUEEZE their plot: along the text (across for an
    upright one, up for a quarter-turned one) they reach past everything else the plot carries -- its frame, tick
    labels, other titles, texts and key -- so constrained layout shrinks the plot to make them room. Measured: a
    slide's spatial profiles, each x title at 16 pt wider than its plot, left the plots 7 mm wide, and a redraw
    collapsed them further. Overhang into room the plot's other decorations already take (a y title beside a key
    above and names below) squeezes nothing and is left alone."""
    from matplotlib.transforms import Bbox

    boxes = {id(t): box for t, box in drawn}
    out = set()
    for ax in fig.get_axes():
        if not ax.get_visible():
            continue
        own = [ax.title, ax._left_title, ax._right_title, ax.xaxis.label, ax.yaxis.label]
        carried = (list(ax.xaxis.get_ticklabels(which="both")) + list(ax.yaxis.get_ticklabels(which="both"))
                   + list(ax.texts) + own)
        legend = ax.get_legend()
        frame = [ax.bbox] + ([legend.get_window_extent()] if legend is not None and legend.get_visible() else [])
        for t in own:
            if id(t) not in boxes or _math(t):
                continue
            rest = Bbox.union(frame + [boxes[id(u)] for u in carried if u is not t and id(u) in boxes])
            box = boxes[id(t)]
            if round(t.get_rotation()) % 180 == 90:
                if box.y0 < rest.y0 - 1 or box.y1 > rest.y1 + 1:
                    out.add(id(t))
            elif box.x0 < rest.x0 - 1 or box.x1 > rest.x1 + 1:
                out.add(id(t))
    return out


def _fit_texts(fig, spec: FigureSpec) -> None:
    """Every title, axis title and key of ``fig`` (a whole figure; a composed one's panels included) that does not
    fit set smaller, then wrapped (see the section's head). WHEN: a figure's or panel's title when it is wider than
    its panel; a plot title or an axis title when it is CUT, COLLIDES (`_clashing`) or SQUEEZES its plot
    (`_squeezing`); a key when it is cut or collides -- each then to its strict room. Measured as the PDF draws it.
    A y title longer than a short plot, beside a key above it and names below, is left as it is (the first version
    shrank every text longer than its plot, eight standard figures that were fine: "when something gets cut"). The
    layout runs (a draw that renders nothing) before each pass, since a smaller or wrapped text takes different room."""
    from matplotlib.legend import Legend

    floor = spec.min_pt()
    for _pass in range(_FIT_PASSES):
        rooms = _text_rooms(fig)
        clashing = crowding = set()
        renderer = None
        if any(what != "title" for _artist, _room, what in rooms):
            # positions current and every text measured: a draw that renders nothing. Not for a figure whose only
            # candidates are titles (the HMM diagram, axes off): a draw would only make tick labels it never shows
            fig.draw_without_rendering()
            renderer = fig._get_renderer()
            drawn = _drawn_texts(fig, renderer)
            clashing = _clashing(fig, drawn)
            crowding = clashing | _squeezing(fig, drawn)
            rooms = _text_rooms(fig)
        changed = False
        for artist, room, what in rooms:
            if isinstance(artist, Legend):
                entries = artist.get_texts() + [artist.get_title()]
                if any(id(t) in clashing or hasattr(t, "_smtracker_fit_from") for t in entries):
                    changed = _fit_legend(artist, room, floor, renderer, fig.dpi) or changed
            elif what == "title" or id(artist) in crowding or hasattr(artist, "_smtracker_fit_from"):
                changed = _fit_one(artist, room, floor, what) or changed
        changed = _even_titles(fig) or changed
        if not changed:
            break
    engine = fig.get_layout_engine()
    if engine is not None:
        engine.execute(fig)
    _unclash_samples(fig, spec)


#: How many measured passes `_unclash_samples` makes before it stops (each a draw that renders nothing).
_SAMPLE_PASSES = 14


def _unclash_samples(fig, spec: FigureSpec) -> None:
    """THE SAMPLE UNDER EACH COLUMN, MEASURED AS DRAWN (7 Nov 2026): where two neighbouring columns' samples
    ("3 replicates" / "5,438 tracks") overlap once the layout has run, every sample of that chart is set smaller
    together (one size), down to the journal's minimum; then each part is broken into its words, one a line; then,
    only where nothing else keeps them apart, smaller still -- the checks then name text under the minimum
    (`check_figure`). Never moved aside. A no-op where nothing overlaps."""
    charts = [[t for t in ax.texts if getattr(t, "_smtracker_sample", False) and t.get_visible()]
              for ax in fig.get_axes()]
    charts = [texts for texts in charts if len(texts) > 1]
    if not charts:
        return
    floor = spec.min_pt()
    gap = 0.5 * fig.dpi / 72.0                              # half a point between neighbours

    def clash(texts, renderer) -> bool:
        boxes = sorted((t.get_window_extent(renderer) for t in texts), key=lambda b: b.x0)
        return any(a.x1 + gap > b.x0 and a.y0 < b.y1 and b.y0 < a.y1 for a, b in zip(boxes, boxes[1:]))

    for _pass in range(_SAMPLE_PASSES):
        fig.draw_without_rendering()
        renderer = fig._get_renderer()
        changed = False
        for texts in charts:
            if not clash(texts, renderer):
                continue
            changed = True
            size = texts[0].get_fontsize()
            if size > floor + 1e-6:
                for t in texts:
                    t.set_fontsize(max(floor, size * 0.9))
            elif any(" " in t.get_text() for t in texts):
                for t in texts:
                    t.set_text("\n".join(w for line in t.get_text().split("\n") for w in line.split(" ") if w))
            else:
                for t in texts:
                    t.set_fontsize(size * 0.9)
        engine = fig.get_layout_engine()
        if engine is not None and changed:
            engine.execute(fig)
        if not changed:
            break


def _fit_notes(texts, spec: FigureSpec) -> list:
    """`check_figure`'s notes: every text `_fit_texts` set smaller or wrapped, by what it is and how far."""
    groups = {}
    for t in texts:
        start = getattr(t, "_smtracker_fit_from", None)
        if start is None:
            continue
        size = round(t.get_fontsize(), 2)
        lines = t.get_text().count("\n") + 1
        short = bool(getattr(t, "_smtracker_short", False))
        wrapped = lines > 1 and t.get_text() != t._smtracker_fit_text
        if not wrapped and not short and size >= round(start, 2) - 1e-6:
            continue
        key = (t._smtracker_fit_what, round(start, 2), size, lines if wrapped else 0, short)
        groups.setdefault(key, []).append(_short(t._smtracker_fit_text))
    journal = PRESET_LABELS[spec.preset]
    notes = []
    for (what, start, size, lines, short), names in groups.items():
        one, many = _FIT_WORDS.get(what, (what, what + "s"))
        count = len(names)
        names = list(dict.fromkeys(names))                     # four panels of one title: counted 4, quoted once
        shown = ", ".join(f"“{n}”" for n in names[:2]) + (", …" if len(names) > 2 else "")
        noun = one if count == 1 else f"{count} {many}"
        at = f"{size:g} pt" + (f" ({journal}'s minimum)" if abs(size - spec.min_pt()) < 1e-6 else "")
        before = f", from {start:g} pt," if size < start - 1e-6 else ""
        if short:
            # the short form first (`_fit_one`): "N-pop" for "N population(s)", then smaller, then wrapped
            how = ["in short form (“-pop” for “population”)"]
            if size < start - 1e-6:
                how.append(f"set at {at}, from {start:g} pt")
            if lines:
                how.append(f"wrapped onto {lines} lines")
            notes.append(FigureNote(f"{noun} {', '.join(how[:-1]) + ' and ' + how[-1] if len(how) > 1 else how[0]}"
                                    f" to fit: {shown}"))
        elif lines:
            notes.append(FigureNote(f"{noun} set at {at}{before} and wrapped onto {lines} lines to fit: {shown}"))
        else:
            notes.append(FigureNote(f"{noun} set at {at}{before} to fit: {shown}"))
    return notes


def build_figure(kind: str, payload: dict, spec: FigureSpec) -> Built:
    """Draw ``kind`` from ``payload`` at ``spec``'s size, font and options."""
    if kind not in KINDS:
        raise FigureExportError(f"no exporter for {kind!r} yet")
    spec = with_title(spec, kind, payload)
    problem = spec.problem()
    if problem:
        raise FigureExportError(problem)
    import matplotlib as mpl

    family, note = resolve_font(spec.font)
    if kind == "hmm_comparison":
        payload = hmm_comparison_panels(payload, spec)
    spec = _with_bracket_room(kind, payload, spec, family)
    layout = None
    if kind == "spatial_maps":
        # The grid's height follows from its rows: capsules on one scale (`FigureSpec.map_scale`) at the cell's
        # proportions.
        layout = _spatial_layout(payload, spec, family)
        if spec.height_mm is None:
            spec = replace(spec, height_mm=layout["height"])
    elif kind == "cell_image" and spec.height_mm is None:
        spec = replace(spec, height_mm=_cell_image_height(payload, spec))
    elif kind in ("hmm_condition", "hmm_overview") and spec.height_mm is None:
        spec = replace(spec, height_mm=_hmm_figure_height(payload, spec, kind))
    elif kind == "jumps_condition" and spec.height_mm is None:
        spec = replace(spec, height_mm=_jumps_height(spec))
    elif kind == "jumps_overview" and spec.height_mm is None:
        spec = replace(spec, height_mm=_jumps_overview_height(payload, spec))
    elif kind == "kinetics_populations" and spec.height_mm is None:
        spec = replace(spec, height_mm=_populations_height(spec, _panel_rows(_single_entry(payload, "populations")
                                                                              ["result"]["chart"], spec)))
    elif kind == "kinetics_overview" and spec.height_mm is None:
        spec = replace(spec, height_mm=_populations_overview_height(payload, spec))
    elif kind == "spatial_profiles" and spec.height_mm is None:
        spec = replace(spec, height_mm=_profiles_height(payload, spec))
    elif kind in ("appd_checks", "msd_checks") and spec.height_mm is None:
        spec = replace(spec, height_mm=_appd_checks_height(spec))
    elif kind == "appd_histogram" and spec.height_mm is None:
        # taller by the residual plots it draws under the histogram (3 Nov 2026)
        spec = replace(spec, height_mm=_appd_histogram_height(
            spec, _appd_strips(spec, kind, (payload or {}).get("entries") or [])))
    elif kind == "appd_overview" and spec.height_mm is None:
        spec = replace(spec, height_mm=_appd_overview_height(payload, spec))
    elif kind == "hmm_angles" and spec.height_mm is None:
        spec = replace(spec, height_mm=_hmm_angles_height(payload, spec))
    elif kind == "hmm_comparison" and spec.height_mm is None:
        # As tall as its panels need at this width (`_comparison_layout`).
        spec = replace(spec, height_mm=_comparison_layout(payload, spec, family)["height"])
    elif kind == "table":
        # A table is as tall as its rows.
        layout = _table_layout(payload, spec, family)
        if spec.height_mm is None:
            spec = replace(spec, height_mm=layout["height"])
    rc = _rc(spec, family)
    fig = _new_figure(spec, rc)
    with mpl.rc_context(rc):
        table, more, legend, source_notes = _draw_kind(fig, kind, payload, spec, family,
                                                       layout)
        if kind not in _MANUAL_KINDS:
            # every text wider than its room set smaller, then wrapped (`_fit_texts`), and the title centred
            _centre_title(fig, spec)
    return Built(figure=fig, spec=spec, kind=kind, family=family,
                 notes=([note] if note else []) + list((layout or {}).get("notes", ())), rc=rc, table=table,
                 legend=legend,
                 title=KINDS[kind], notes_for_source=source_notes, more_sheets=more,
                 layout_findings=tuple((layout or {}).get("findings", ())))


#: The payload key holding the condition colours' page variants (`condition_colours.ConditionColours.page_variants`):
#: ``{colour as the screen drew it: [on a white page, on a dark page]}``, put there by the workspace for every chart it
#: exports (`WorkspaceWindow._titled`): `condition_colours.PAGE_COLOURS`, imported at the top.


def page_coloured(payload, dark: bool):
    """``payload`` with every condition colour in the variant for the FIGURE's page -- the dark one on a dark preset
    (Dark slides), the light one on every other -- whatever theme the screen was in (found by the overnight tester, 4
    Nov 2026: in the dark theme an olive condition was exported yellow on a white Nature page). Only values under a
    key naming a colour ("colour", "colours", "color" ...) are read; a payload without `PAGE_COLOURS` is returned as
    it is. Returns a new payload; the caller's is never changed."""
    variants = payload.get(PAGE_COLOURS) if isinstance(payload, dict) else None
    if not variants:
        return payload
    pick = {str(k).lower(): v[1 if dark else 0] for k, v in variants.items()}

    def swap(value):
        if isinstance(value, str):
            return pick.get(value.lower(), value)
        if isinstance(value, dict):
            return {k: swap(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(swap(v) for v in value)
        return value

    def walk(value, coloured=False):
        if coloured:
            return swap(value)
        if isinstance(value, dict):
            return {k: walk(v, isinstance(k, str) and "colo" in k.lower()) for k, v in value.items()}
        if isinstance(value, list) and value and isinstance(value[0], (dict, list, tuple)):
            return [walk(v) for v in value]
        if isinstance(value, tuple) and value and isinstance(value[0], (dict, list, tuple)):
            return tuple(walk(v) for v in value)
        return value

    out = walk({k: v for k, v in payload.items() if k != PAGE_COLOURS})
    out[PAGE_COLOURS] = variants
    return out


def _draw_kind(fig, kind: str, payload: dict, spec: FigureSpec, family: str,
               layout=None) -> tuple:
    """Draw ``kind`` into ``fig`` -- a whole figure, or one composed panel's sub-figure --
    and return ``(table, more sheets, legend, source notes)``."""
    if kind not in KINDS:
        raise FigureExportError(f"no exporter for {kind!r} yet")
    # the condition colours for THIS figure's page, never the screen's theme
    payload = page_coloured(payload, spec.dark)
    if kind in GROUPED_NAME_KINDS and isinstance(payload, dict):
        # a grouped chart renamed by its group headers and column labels: every column's own name follows them
        spec = with_part_names(spec, payload.get("data"))
    # what this holder draws: `_fit_texts` leaves a kind laid out by hand (`_MANUAL_KINDS`) alone
    fig._smtracker_kind = kind
    ax = fig.add_subplot(111)
    more = ()
    if kind == "spatial_maps":
        fig.delaxes(ax)
        table, more, legend, source_notes = _draw_spatial(fig, payload, spec, family,
                                                          layout)
    elif kind == "table":
        fig.delaxes(ax)
        table, more, legend, source_notes = _draw_table(
            fig, payload, spec, family, layout or _table_layout(payload, spec, family))
    elif kind == "cell_image":
        _draw_cell_image(fig, ax, payload, spec, family)
        table = _cell_image_source(payload)
        legend, source_notes = cell_image_legend(payload, spec), _cell_image_notes(payload)
    elif kind in ("sqd_bubbles", "appd_bubbles"):
        data = payload["data"]
        plan = _significance_plan(kind, payload, spec, family)
        from .bubble_view import sample_marks

        _draw_columns(fig, ax, [str(c) for c in data.conditions],
                      _sqd_points(data, payload.get("colours") or {}), spec, family,
                      errors=data.errors, series_labels=data.series_labels, plan=plan,
                      layout=getattr(data, "layout", None),
                      samples=sample_marks(data) if spec.sample_n else None)
        table = _sqd_source(data, spec)
        if kind == "sqd_bubbles":
            legend, source_notes = bubble_legend(data, spec.shown_name), _sqd_source_notes(data)
        else:
            legend, source_notes = appd_bubble_legend(data, spec.shown_name), _appd_bubble_notes(data)
        if spec.sample_n and sample_marks(data):
            from .bubble_view import mixed_scopes

            # a Shared D per condition ends each line with how its D was fitted (`bubble_view.sample_marks`)
            legend += (" " + SAMPLE_SENTENCE + (SAMPLE_SCOPE_CLAUSE if mixed_scopes(data) else "")
                       + (" (shown once under each group, whose columns are fits of one sample)."
                          if getattr(data, "fits", None) else "."))
        if plan:
            more, legend = _significance_sheets(plan, spec), _significance_legend(
                legend, plan, spec)
    elif kind == "kinetics_populations":
        entry = _single_entry(payload, "populations")
        fig.delaxes(ax)
        # Two panels over one x axis; the dialog's x title and range are the lower panel's.
        ax = _draw_populations(fig, entry, spec)
        table, more = _populations_source(entry, spec)
        legend, source_notes = populations_legend(entry, spec), _populations_notes(entry)
    elif kind == "kinetics_overview":
        fig.delaxes(ax)
        # One column per condition, D over share, one time axis and one y range per row
        ax = _draw_populations_overview(fig, payload, spec)
        table, more = _populations_overview_source(payload, spec)
        legend, source_notes = populations_overview_legend(payload, spec), _populations_overview_notes(payload)
    elif kind == "kinetics_mock":
        fig.delaxes(ax)
        # One panel per treated condition, its two columns (treated, its mock), one y axis
        ax = _draw_mock(fig, payload, spec)
        table, more = _mock_source(payload, spec)
        legend, source_notes = mock_legend(payload, spec), _mock_notes(payload)
    elif kind.startswith("kinetics"):
        entries = payload["entries"]
        if not entries:
            raise FigureExportError("no kinetics result to draw yet")
        _draw_kinetics(fig, ax, entries[0], spec, kind)
        table, more = _kinetics_source(entries[0], spec)
        legend = kinetics_legend(entries[0])
        source_notes = _kinetics_source_notes(entries[0])
    elif kind == "hmm_condition":
        entry = _single_entry(payload, "HMM")
        fig.delaxes(ax)
        # the figure of one: the state diagram placed by hand in mm, its name in the key (`_draw_hmm_figure`)
        ax = _draw_hmm_figure(fig, payload, spec, family, kind)
        table, more = _hmm_source(entry, spec)
        legend, source_notes = hmm_legend(entry), _hmm_source_notes(entry)
    elif kind == "hmm_overview":
        fig.delaxes(ax)
        # one diagram per condition kept, each placed by hand in mm (`_draw_hmm_figure`)
        ax = _draw_hmm_figure(fig, payload, spec, family)
        table, more = _hmm_overview_source(payload, spec)
        legend, source_notes = hmm_overview_legend(payload, spec), _hmm_overview_notes(payload, spec)
    elif kind == "jumps_condition":
        entry = _single_entry(payload, "jump-distance")
        fig.delaxes(ax)
        # Three plots over one x axis; the dialog's titles and ranges are the histogram's.
        ax = _draw_jumps(fig, entry, spec, family)
        table, more = _jumps_source(entry, spec)
        legend, source_notes = jumps_legend(entry, spec), _jumps_source_notes(entry, spec)
    elif kind == "jumps_overview":
        fig.delaxes(ax)
        # One histogram per condition over one x axis; the dialog's x range is shared by all.
        ax = _draw_jumps_overview(fig, payload, spec, family)
        table, more = _jumps_overview_source(payload, spec)
        legend, source_notes = jumps_overview_legend(payload, spec), _jumps_overview_notes(payload, spec)
    elif kind == "jd_condition":
        entry = _single_entry(payload, "signed-jump")
        _draw_jd(fig, ax, entry, spec, family)
        table, more = _jd_source(entry, spec)
        legend, source_notes = jd_legend(entry, spec), (jd_legend(entry, spec),)
    elif kind == "hmm_angles":
        fig.delaxes(ax)
        ax = _draw_hmm_angles(fig, payload, spec, family)
        table, more = _hmm_angles_source(payload, spec)
        legend, source_notes = hmm_angles_legend(payload, spec), _hmm_angles_notes(payload)
    elif kind == "hmm_comparison":
        fig.delaxes(ax)
        # Four panels and a key over them; no axis title or range is the dialog's.
        plan = _hmm_bracket_plan(payload, spec)
        ax = _draw_hmm_comparison(fig, payload, spec, family, marks=plan["marks"] if plan else ())
        table, more = _hmm_comparison_source(payload, spec)
        legend = hmm_comparison_legend(payload, spec, tested=bool(plan))
        source_notes = _hmm_comparison_notes(payload, tested=bool(plan))
        if plan:
            # the brackets' statistics beside the panels' numbers (5 Nov 2026)
            more = tuple(more) + _significance_sheets(plan, spec)
            legend = _significance_legend(legend, plan, spec)
    elif kind.startswith("qc_"):
        table, more, legend, source_notes = _draw_qc(fig, ax, kind, payload, spec, family)
    elif kind == "appd_histogram":
        entry = _single_entry(payload, "apparent-D")
        fig.delaxes(ax)
        ax = _draw_appd_histogram(fig, entry, spec, family)
        table, more = _appd_histogram_source(entry, spec)
        legend, source_notes = appd_histogram_legend(entry, spec), _appd_histogram_notes(entry, spec)
    elif kind == "appd_overview":
        fig.delaxes(ax)
        ax = _draw_appd_overview(fig, payload, spec, family)
        table, more = _appd_overview_source(payload, spec)
        legend, source_notes = appd_overview_legend(payload, spec), _appd_overview_notes(payload, spec)
    elif kind == "appd_cells":
        _draw_appd_cells(fig, ax, payload, spec, family)
        table, more = _appd_cells_source(payload, spec)
        legend, source_notes = appd_cells_legend(payload), _appd_cells_notes(payload)
    elif kind == "appd_checks":
        entry = _single_entry(payload, "apparent-D")
        fig.delaxes(ax)
        # Two plots; the dialog's titles and ranges are the switching check's.
        ax = _draw_appd_checks(fig, entry, spec, family)
        table, more = _appd_checks_source(entry, spec)
        legend, source_notes = appd_checks_legend(entry), _appd_checks_notes(entry)
    elif kind == "sqd_crossed":
        _draw_sqd_crossed(fig, ax, payload, spec, family)
        table = _sqd_crossed_source(payload, spec)
        legend = sqd_crossed_legend(payload)
        source_notes = (payload["crossed"]["caption"],)
    elif kind == "spatial_profiles":
        fig.delaxes(ax)
        ax = _draw_spatial_profiles(fig, payload, spec, family)
        table, more = _spatial_profiles_source(payload, spec)
        legend, source_notes = spatial_profiles_legend(payload), (spatial_profiles_legend(payload),)
    elif kind == "appd_cell_length":
        _draw_appd_cell_length(fig, ax, payload, spec, family)
        table, more = _appd_cell_length_source(payload, spec)
        legend, source_notes = appd_cell_length_legend(payload), (payload.get("caption", ""),)
    elif kind == "msd_checks":
        entry = _single_entry(payload, "MSD")
        fig.delaxes(ax)
        ax = _draw_msd_checks(fig, entry, spec, family)
        table, more = _msd_checks_source(entry, spec)
        legend, source_notes = msd_checks_legend(entry), _msd_checks_notes(entry)
    elif kind == "clusters_shares":
        _draw_cluster_shares(fig, ax, payload, spec, family)
        table, more = _clusters_shares_source(payload, spec)
        legend, source_notes = clusters_shares_legend(payload), tuple(payload["card"]["lines"])
    elif kind == "clusters_condition":
        entry = _single_entry(payload, "clustering")
        _draw_clusters(fig, ax, entry, spec, family)
        table, more = _clusters_source(entry, spec)
        legend, source_notes = clusters_legend(entry), tuple(entry["result"]["lines"])
    elif kind == "msd_v2":
        entry = _single_entry(payload, "MSD")
        _draw_msd_v2(fig, ax, entry, spec, family)
        table, more = _msd_v2_source(entry, spec)
        legend, source_notes = msd_v2_legend(entry), _msd_v2_notes(entry)
    elif kind.startswith("msd"):
        entries = payload["entries"]
        if not entries:
            raise FigureExportError("no MSD result to draw yet")
        _draw_msd(fig, ax, entries, spec, family, kind)
        table = _msd_source(entries, spec)
        more = _msd_day_sheet(entries, spec)
        legend = msd_legend(entries, spec)
        source_notes = _msd_source_notes(entries, spec)
    else:
        fit = payload["payload"]
        _draw_columns(fig, ax, _replicate_columns(fit),
                      _replicate_points(fit, payload.get("colour")), spec, family)
        table = _replicate_source(fit, spec)
        legend = replicate_legend(fit, payload.get("condition", ""),
                                  shaded=bool(payload.get("colour")))
        source_notes = _replicate_source_notes(fit)
    if kind not in _MANUAL_KINDS:
        _apply_ranges(ax, spec, kind)
        if spec.title:
            # Over everything (a legend above the plot included): constrained layout makes
            # room for a sub/figure's own title.
            # NO POSITION GIVEN, ON PURPOSE: constrained layout reserves the title's room only
            # for a title it may place itself (`_autopos`). Placed at an explicit x and y it
            # took no room, and centred it was drawn over a key above the plot (found
            # 13 Oct 2026). Centred by default; a whole figure then moves it over the plot's
            # middle (`_centre_title`).
            fig.suptitle(spec.title, fontsize=spec.pt(), fontweight="bold",
                         color=_page(spec)["ink"])
    return table, more, legend, source_notes


# -- the composer ---------------------------------------------------------------- #
#: A composed panel's WIDTH, a fraction of the figure's full width (3 Nov 2026, asked: "why not 1/3, 1/2, 2/3, 1", "all
#: scaling to the width of one full figure"), in place of a span over a grid's columns: panels of 1/2, 1/3 and 2/3 now
#: share one figure, and a chart exported at 2/3 arrives at 2/3 (`composer.width_for`). 1/4 and 3/4 keep what a
#: four-column grid drew. A row takes panels in reading order until the next would pass the full width.
PANEL_WIDTHS = ("1/4", "1/3", "1/2", "2/3", "3/4", "1")
PANEL_WIDTH_LABELS = {"1/4": "1/4", "1/3": "1/3", "1/2": "1/2", "2/3": "2/3", "3/4": "3/4", "1": "Full"}
#: The columns of the grids drawn before 3 Nov 2026: a record that carries a ``span`` over ``columns`` (a composition
#: file of version 1, a script) is drawn at the fraction ``span / columns``.
COMPOSE_COLUMNS = (1, 2, 3, 4)
#: A grid's rows: None grows with the panels, a number fixes the grid.
COMPOSE_ROWS = (None, 1, 2, 3, 4, 5, 6)
#: A panel's size inside its cell, in per cent: drawn again at that size (text keeps its
#: point size, a PDF or SVG stays vector) and centred, so one image can be matched to another.
PANEL_SCALE = (30, 100)
LETTER_CASES = ("journal", "lower", "upper", "none")
LETTER_CASE_LABELS = {"journal": "Journal", "lower": "a, b, c", "upper": "A, B, C",
                      "none": "None"}
#: What a panel keeps from the Export dialog it was added from: its words and what it shows.
#: Size, font, journal and files are the COMPOSITION's.
PANEL_WORDS = ("numbers", "sample_n", "grid", "legend", "scale_bar", "scale_bar_um", "headings", "y_title",
               "x_title", "names", "title", "y_min", "y_max", "x_min", "x_max", "chosen_fit",
               "pearson_residuals", "ecdf_residuals", "components", "y_mode", "grid_columns", "left_out", "rate_spread",
               "occupancy_spread", "map_scale", "pair_lines", "slowest_d", "other_shares", "half_time",
               "conditions_as_rows", "hmm_dwell")


def panel_words(spec: FigureSpec) -> dict:
    """The words and shows a panel takes from the dialog it was added from."""
    return {key: getattr(spec, key) for key in PANEL_WORDS}


#: A composed panel keeps its significance brackets too: they were picked for that chart,
#: which the panel is a snapshot of (never carried to the NEXT chart: `remembered_record`).
PANEL_WORDS = PANEL_WORDS + SIGNIFICANCE_WORDS


def panel_title(kind: str, payload: dict) -> str:
    """What a panel is called in the composer's list: the chart, then what it shows."""
    if kind == "table":
        detail = " · ".join(str(v) for v in (payload.get("title"), payload.get("condition"))
                            if v)
        return f"Table · {detail}" if detail else "Table"
    detail = str(payload.get("condition") or "")
    if not detail:
        try:
            names = column_names(kind, payload)
        except (KeyError, TypeError, AttributeError):
            names = []
        if kind.startswith(("qc_",)):
            names = []
        detail = ", ".join(names[:4]) + (f" +{len(names) - 4}" if len(names) > 4 else "")
    return f"{KINDS.get(kind, kind)} · {detail}" if detail else KINDS.get(kind, kind)


def panel_letter(index: int, case: str, preset: str) -> str:
    """Panel ``index``'s letter: a, b, c ... in the preset's case, or "" for none."""
    if case not in LETTER_CASES:
        raise FigureExportError(f"letters must be one of {LETTER_CASES}, got {case!r}")
    if case == "none":
        return ""
    if case == "journal":
        case = figstyle.PANEL_LETTERS.get(preset, ("upper", 8.0))[0]
    letter = chr(ord("a") + index % 26) * (index // 26 + 1)
    return letter.upper() if case == "upper" else letter


def width_fraction(width) -> Fraction:
    """The fraction a width key names ("2/3" -> 2/3; `PANEL_WIDTHS`); anything else is refused by name."""
    if str(width) not in PANEL_WIDTHS:
        raise FigureExportError(f"a panel's width is one of {', '.join(PANEL_WIDTHS)} of the figure, "
                                f"not {width!r}")
    return Fraction(str(width))


def fraction_mm(width, full: float, gutter: float) -> float:
    """How wide ``width`` (a key of `PANEL_WIDTHS`) of a ``full`` mm figure is, in mm, with ``gutter`` mm between
    panels: ``f (full + gutter) - gutter``, so two halves, three thirds, or 2/3 beside 1/3 fill it exactly with one
    gutter between each pair (Cell's 174 mm: 1/2 = 85 mm, its one column; 2/3 = 114.7 mm, its 1.5 columns 114)."""
    return float(width_fraction(width)) * (float(full) + float(gutter)) - float(gutter)


def legacy_width(span, columns) -> str:
    """The width key a ``span`` over a grid of ``columns`` drew (records before 3 Nov 2026): ``span / columns``."""
    if columns not in COMPOSE_COLUMNS:
        raise FigureExportError(f"columns must be one of {COMPOSE_COLUMNS}, got {columns!r}")
    fraction = Fraction(max(1, min(int(span), int(columns))), int(columns))
    return str(fraction) if fraction != 1 else "1"


def compose_places(widths, row_spans, rows=None) -> tuple:
    """Panels of ``widths`` (keys of `PANEL_WIDTHS`) in reading order: ``(places, n_rows, units)``. The full width is
    cut into ``units`` equal parts, the fewest every width fills whole (halves and thirds: 6), each place
    ``(index, row, first unit, units spanned, row span)`` as `compose_grid` places them on that many columns."""
    fractions = [width_fraction(w) for w in widths]
    units = 1
    for f in fractions:
        units = units * f.denominator // math.gcd(units, f.denominator)
    places, n_rows = _grid_places([int(f * units) for f in fractions], row_spans, units, rows)
    return places, n_rows, units


def compose_rows(spans, columns: int) -> list:
    """Panels in reading order, as rows of ``(panel index, first column, span)``: a panel
    that does not fit what is left of a row starts the next one. One-row spans only; the
    composer's grid is `compose_grid`."""
    if columns not in COMPOSE_COLUMNS:
        raise FigureExportError(f"columns must be one of {COMPOSE_COLUMNS}, got {columns!r}")
    rows, row, used = [], [], 0
    for index, span in enumerate(spans):
        span = max(1, min(int(span), columns))
        if row and used + span > columns:
            rows.append(row)
            row, used = [], 0
        row.append((index, used, span))
        used += span
    if row:
        rows.append(row)
    return rows


def compose_grid(spans, row_spans, columns: int, rows=None) -> tuple:
    """Panels in reading order on a ``rows`` x ``columns`` grid (asked for: "we need to
    define the grid (4x2, etc)"): ``(places, n_rows)``, each place ``(index, row, column,
    column span, row span)``. A panel takes the first free block at or after the one before
    it, so the reading order holds and a hole a wide panel leaves is not back-filled by a
    later one. ``rows`` None grows the grid as the panels need; a fixed count too small for
    them is REFUSED with what it takes, never silently grown here (the composer grows it,
    and says so, when a panel is ADDED to a full grid). Empty cells of a fixed grid stay
    empty."""
    if columns not in COMPOSE_COLUMNS:
        raise FigureExportError(f"columns must be one of {COMPOSE_COLUMNS}, got {columns!r}")
    return _grid_places(spans, row_spans, columns, rows)


def _grid_places(spans, row_spans, columns: int, rows=None) -> tuple:
    """`compose_grid` on any number of columns (`compose_places`' units)."""
    if rows not in COMPOSE_ROWS:
        raise FigureExportError(f"rows must be one of {COMPOSE_ROWS}, got {rows!r}")
    taken, places, cursor = set(), [], 0
    for index, (cs, rs) in enumerate(zip(spans, row_spans)):
        cs = max(1, min(int(cs), columns))
        rs = max(1, int(rs)) if rows is None else max(1, min(int(rs), rows))
        pos = cursor
        while True:
            r, c = divmod(pos, columns)
            if c + cs <= columns and all((r + i, c + j) not in taken
                                         for i in range(rs) for j in range(cs)):
                break
            pos += 1
        taken.update((r + i, c + j) for i in range(rs) for j in range(cs))
        places.append((index, r, c, cs, rs))
        cursor = pos + cs
    needed = max((r + rs for _i, r, _c, _cs, rs in places), default=0)
    if rows is not None and needed > rows:
        raise FigureExportError(
            f"{len(places)} panel{'s' if len(places) != 1 else ''} need {needed} rows, and the grid has "
            f"{rows}: add a row, or set Rows to Auto")
    return places, (rows if rows is not None else needed)


def compose_gutters(spec: FigureSpec) -> tuple:
    """``(between columns, between rows)`` in mm: wider on slides and posters."""
    return ((12.0, 10.0) if spec.preset == "poster" else
            (8.0, 6.0) if spec.preset in LARGE_PRESETS else (4.0, 3.0))


def _panel_scale(panel: dict) -> float:
    """A panel's size in its cell as a fraction (`PANEL_SCALE`, per cent, clamped)."""
    lo, hi = PANEL_SCALE
    return max(lo, min(hi, float(panel.get("scale", hi)))) / 100.0


def _panel_spec(spec: FigureSpec, panel: dict, width: float, height) -> FigureSpec:
    words = {k: v for k, v in (panel.get("words") or {}).items() if k in PANEL_WORDS}
    if "names" in words:
        words["names"] = tuple(tuple(pair) for pair in words["names"] or ())
    if words.get("title", None) is None:
        # an automatic title, as the panel's Export dialog showed it
        words["title"] = auto_title(panel.get("kind", ""), panel.get("payload"))
    return replace(spec, width="custom", width_mm=float(width),
                   height_mm=None if height is None else float(height), **words)


def compose(panels, spec: FigureSpec, *, columns: int = 2, rows=None,
            letters: str = "journal") -> Built:
    """Several panels -- each ``{"kind", "payload", "words", "title", "width", "row_span",
    "scale", "match"}``, as the composer holds them -- as ONE figure at ``spec``'s width, each
    panel its ``width`` of it (`PANEL_WIDTHS`, placed by `compose_places` in ``rows``): one matplotlib
    sub-figure per panel (so text stays text in every format), lettered per the journal,
    each row as tall as its tallest panel, and a workbook with a sheet per panel. A record with a
    ``span`` and no ``width`` (before 3 Nov 2026) is drawn at ``span / columns`` (`legacy_width`).

    A panel's ``scale`` (per cent) draws it that much smaller inside its cell. ``match``
    (``("height" | "width", index of another panel)``) SNAPS it to that panel instead
    (asked for: "A and B have different sizes -> snap A to height of B, or width of C"):
    its plot or image area is made exactly as tall (or wide) as the other's and lined up
    with it -- top edges for a height, left edges for a width. Text keeps its point size,
    so the size that does it is found by drawing, measuring and drawing again
    (`MATCH_PASSES`); a match the cell cannot hold is named, never forced."""
    import matplotlib as mpl

    panels = list(panels)
    if not panels:
        raise FigureExportError("there is no panel yet; press Add to composer in a chart's Export dialog")
    problem = spec.problem()
    if problem:
        raise FigureExportError(problem)
    for panel in panels:
        if panel.get("kind") not in KINDS:
            raise FigureExportError(f"no exporter for {panel.get('kind')!r}")
    family, note = resolve_font(spec.font)
    total = spec.width_value()
    gut_x, gut_y = compose_gutters(spec)
    widths = [p["width"] if p.get("width") is not None else legacy_width(p.get("span", 1), columns)
              for p in panels]
    # the grid's columns are the UNITS every width fills whole; ``cell`` is one unit
    places, n_rows, columns = compose_places(widths, [p.get("row_span", 1) for p in panels], rows)
    cell = (total - gut_x * (columns - 1)) / columns
    letter_pt = figstyle.PANEL_LETTERS.get(spec.preset, ("upper", 8.0))[1]
    letter_mm = letter_pt * 1.3 * _PT_MM
    lettered = [panel_letter(i, letters, spec.preset) for i in range(len(panels))]

    def width_of(span):
        return cell * span + gut_x * (span - 1)

    def natural(index, width, inside=True):
        """The height a panel needs at ``width``; ``inside`` counts its letter, which a
        resized panel leaves on its cell rather than in its drawing."""
        panel = panels[index]
        lettered_here = bool(lettered[index]) and inside
        if panel["kind"] == "spatial_maps":
            top = letter_mm + 0.8 if lettered_here else 1.0
            return _spatial_layout(panel["payload"], _panel_spec(spec, panel, width, None),
                                   family, top=top)["height"]
        if panel["kind"] == "cell_image":
            return (_cell_image_height(panel["payload"], _panel_spec(spec, panel, width, None))
                    + (letter_mm if lettered_here else 0.0))
        if panel["kind"] == "table":
            return _table_layout(panel["payload"], _panel_spec(spec, panel, width, None),
                                 family, top=letter_mm + 0.8 if lettered_here else 0.0)["height"]
        if panel["kind"] in ("hmm_condition", "hmm_overview"):
            # the diagrams as tall as their text needs at this width (`_hmm_layout`), plus the letter
            return (_hmm_figure_height(panel["payload"], _panel_spec(spec, panel, width, None), panel["kind"])
                    + (letter_mm if lettered_here else 0.0))
        if panel["kind"] == "hmm_comparison":
            # Four panels in a row, two by two or stacked, as its width allows.
            return (_comparison_layout(panel["payload"], _panel_spec(spec, panel, width, None),
                                       family)["height"]
                    + (letter_mm if lettered_here else 0.0))
        if panel["kind"] == "jumps_overview":
            return (_jumps_overview_height(panel["payload"], _panel_spec(spec, panel, width, None))
                    + (letter_mm if lettered_here else 0.0))
        if panel["kind"] == "appd_overview":
            return (_appd_overview_height(panel["payload"], _panel_spec(spec, panel, width, None))
                    + (letter_mm if lettered_here else 0.0))
        if panel["kind"] == "appd_histogram" and spec.preset not in LARGE_PRESETS:
            # taller by the residual plots under the histogram (3 Nov 2026, `_appd_histogram_height`'s share)
            strips = _appd_strips(_panel_spec(spec, panel, width, None), "appd_histogram",
                                  (panel["payload"] or {}).get("entries") or [])
            share = sum(_JUMPS_SHARES[n] for n in ("histogram", *strips)) / _JUMPS_SHARES["histogram"]
            return (max(40.0, min(75.0, width * 0.72)) * (1.0 + (share - 1.0) * 0.8)
                    + (letter_mm if lettered_here else 0.0))
        if panel["kind"] == "jumps_condition" and spec.preset not in LARGE_PRESETS:
            # Three stacked plots: about as tall as wide (`_jumps_height`), shorter by a residual plot turned off.
            return (_jumps_stacked(max(50.0, min(130.0, width * 1.05)),
                                   _panel_spec(spec, panel, width, None), 40.0)
                    + (letter_mm if lettered_here else 0.0))
        if panel["kind"] in ("appd_checks", "msd_checks", "spatial_profiles") and spec.preset not in LARGE_PRESETS:
            # Two stacked plots (`_appd_checks_height`).
            return (max(50.0, min(120.0, width * 0.95))
                    + (letter_mm if lettered_here else 0.0))
        if spec.preset in LARGE_PRESETS:
            # Sixteen lines of the body text at least: a poster's 24 pt in a panel sized by
            # its width alone left the plot no room between its legend and its axis title.
            return max(60.0, spec.pt() * _PT_MM * 16.0, min(300.0, width * 0.6))
        return max(40.0, min(75.0, width * 0.72))

    # Each row as tall as its tallest one-row panel AT ITS SIZE (a panel drawn at 60 % asks
    # for 60 % of its height, so its row can shrink with it); a panel over several rows then
    # widens the rows it spans by whatever it still lacks; a row no panel sizes (an empty
    # row of a fixed grid, or one only spanned) takes the others' mean. A set height scales
    # every row, and every panel's own height, by one factor.
    def need(index, cs, scale):
        if scale >= 1.0 - 1e-6:
            return natural(index, width_of(cs))
        return natural(index, width_of(cs) * scale, inside=False)

    def heights_for(scales):
        needs = {index: need(index, cs, scales[index]) for index, _r, _c, cs, _rs in places}
        heights = [0.0] * n_rows
        for index, r, _c, _cs, rs in places:
            if rs == 1:
                heights[r] = max(heights[r], needs[index])
        for index, r, _c, _cs, rs in places:
            if rs > 1:
                have = sum(heights[r:r + rs]) + gut_y * (rs - 1)
                need_here = needs[index]
                if need_here > have:
                    for k in range(r, r + rs):
                        heights[k] += (need_here - have) / rs
        sized = [h for h in heights if h > 0]
        fill = sum(sized) / len(sized) if sized else max(40.0, min(75.0, width_of(columns / 2) * 0.72))
        heights = [h if h > 0 else fill for h in heights]
        factor = 1.0
        if spec.height_mm is not None:
            room = spec.height_mm - gut_y * (n_rows - 1)
            factor = room / sum(heights)
            heights = [h * factor for h in heights]
        sizes = {}
        for index, r, _c, cs, rs in places:
            cw = width_of(cs)
            ch = sum(heights[r:r + rs]) + gut_y * (rs - 1)
            if scales[index] >= 1.0 - 1e-6:
                sizes[index] = (cw, ch, cw, ch)
            else:
                # exactly the height its drawing needs at its width: no empty band above
                # or below it, so a snap lines up its plot rather than its box
                sizes[index] = (cw, ch, cw * scales[index], min(ch, needs[index] * factor))
        return heights, sizes

    findings = []
    widest = spec.preset_values.get("double_mm")
    if widest and total > widest + 1e-6:
        findings.append(f"{total:g} mm is wider than the {widest:g} mm "
                        f"{PRESET_LABELS[spec.preset]} prints; make a panel narrower")

    state = {}

    def draw(scales, offsets):
        heights, sizes = heights_for(scales)
        height = round(sum(heights) + gut_y * (n_rows - 1), 1)
        whole = replace(spec, height_mm=min(600.0, max(20.0, height)))
        rc = _rc(whole, family)
        state.update(whole=whole, rc=rc)
        return _compose_draw(panels, places, heights, columns, cell, gut_x, gut_y, whole,
                             rc, family, letter_pt, letter_mm, lettered, scales, offsets,
                             sizes, spec)

    scales = [_panel_scale(p) for p in panels]
    offsets = [None] * len(panels)
    dropped = []
    matches = _matches(panels, dropped)
    for i, dimension, target in dropped:
        name, other = lettered[i] or str(i + 1), lettered[target] or str(target + 1)
        findings.append(f"panel {name} cannot snap to the {dimension} of {other} because {other} is "
                        f"already snapped to {name}; remove one of the two snaps")
    fig, parts, more_findings, boxes = draw(scales, offsets)
    if matches:
        # 1. THE SIZE: drawn, measured and drawn again until each snapped panel's plot is as
        #    tall (or wide) as its target's -- text keeps its point size, so the size that
        #    does it is found, not computed.
        for _pass in range(MATCH_PASSES):
            moved = False
            for i, dimension, target in matches:
                a, b = boxes.get(i), boxes.get(target)
                if not a or not b:
                    continue
                k = 1 if dimension == "height" else 0
                own = a["axes"][k + 2] - a["axes"][k]
                want = b["axes"][k + 2] - b["axes"][k]
                if own <= 0 or want <= 0:
                    continue
                scale = min(1.0, max(MATCH_FLOOR, scales[i] * want / own))
                if abs(scale - scales[i]) > 5e-4:
                    scales[i], moved = scale, True
            if not moved:
                break
            fig, parts, more_findings, boxes = draw(scales, offsets)
        # 2. THE LINE: once the layout is settled it is frozen, and each snapped panel is
        #    moved by the measured distance -- top edges level for a height, left edges for a
        #    width -- when the two share a row (a height) or a column (a width). Moving the
        #    drawing after layout, because the layout re-spaces the inside of a resized
        #    panel and undid an offset set before it.
        fig.set_layout_engine("none")
        for i, dimension, target in matches:
            a, b = boxes.get(i), boxes.get(target)
            if not a or not b:
                continue
            if dimension == "height":
                shared = min(a["cell"][3], b["cell"][3]) > max(a["cell"][1], b["cell"][1])
                shift = (0.0, b["axes"][3] - a["axes"][3])
            else:
                shared = min(a["cell"][2], b["cell"][2]) > max(a["cell"][0], b["cell"][0])
                shift = (b["axes"][0] - a["axes"][0], 0.0)
            if shared and max(abs(shift[0]), abs(shift[1])) > 0.01:
                _shift_panel(fig, a["holder"], shift)
                a["axes"] = (a["axes"][0] + shift[0], a["axes"][1] + shift[1],
                             a["axes"][2] + shift[0], a["axes"][3] + shift[1])
        for i, dimension, target in matches:
            a, b = boxes.get(i), boxes.get(target)
            if not a or not b:
                continue
            k = 1 if dimension == "height" else 0
            own = a["axes"][k + 2] - a["axes"][k]
            want = b["axes"][k + 2] - b["axes"][k]
            if abs(own - want) > 0.5:
                name = lettered[i] or str(i + 1)
                other = lettered[target] or str(target + 1)
                findings.append(
                    f"panel {name} cannot reach the {dimension} of {other} inside its cell "
                    f"({own:.1f} mm against {want:.1f}); give {name} more "
                    f"{'rows' if dimension == 'height' else 'width'}, or match {other} "
                    f"to {name} instead")
    with mpl.rc_context(state["rc"]):
        # every text wider than its room set smaller, then wrapped (`_fit_texts`), and each title centred
        _centre_panel_titles(fig, state["whole"])
    for part in parts:
        part["scale"] = round(scales[part["index"]] * 100.0)
    parts.sort(key=lambda part: part["index"])
    legend = " ".join(f"({part['letter']}) {part['legend']}" if part["letter"]
                      else part["legend"] for part in parts)
    return Built(figure=fig, spec=state["whole"], kind="composed", family=family,
                 notes=[note] if note else [], rc=state["rc"], table=parts[0]["table"],
                 legend=legend, title="Composed figure", notes_for_source=(), more_sheets=(),
                 layout_findings=tuple(findings + more_findings), parts=tuple(parts))


def _centre_panel_titles(fig, spec: FigureSpec | None = None) -> None:
    """Every composed panel's title over the middle of its own plot (`_title_over_plots`): the layout is run
    first (after a snap it is frozen, and its placeholder engine runs nothing), then every text wider than its
    room in any panel is set smaller or wrapped (`_fit_texts`, given ``spec``)."""
    engine = fig.get_layout_engine()
    if engine is not None:
        engine.execute(fig)
    if spec is not None:
        _fit_texts(fig, spec)
    for holder in getattr(fig, "_smtracker_titles", ()):
        _title_over_plots(holder)


#: How many times a snapped panel is drawn, measured and drawn again (it converges in two or
#: three: text keeps its point size, so a plot's size is not proportional to its panel's).
MATCH_PASSES = 6
#: The smallest a snapped panel is drawn, as a fraction of its cell.
MATCH_FLOOR = 0.15
#: A shrunk panel's spacer narrower than this is left out (see `_compose_draw`).
MIN_SPACER_MM = 0.5


def _matches(panels, dropped=None) -> list:
    """``[(index, dimension, target index)]``: every panel snapped to ANOTHER panel that
    exists, ordered so a panel's target is settled before it (a chain a -> b -> c resolves).
    A snap that would close a LOOP (c -> a while a -> c, directly or along a chain) is left
    out -- the later panel's, in panel order -- and put in ``dropped``. The first version
    dropped a loop only when it was met from one side: visited from the other it kept both
    snaps, the two panels resized each other back and forth on every pass, and the layout
    solver once ABORTED the whole process on what that left (found by a random sweep)."""
    wanted = {}
    for i, panel in enumerate(panels):
        match = panel.get("match")
        if not match:
            continue
        dimension, target = match
        if dimension in ("height", "width") and isinstance(target, int)                 and 0 <= target < len(panels) and target != i:
            wanted[i] = (dimension, target)
    accepted = {}
    for i in sorted(wanted):
        dimension, target = wanted[i]
        step, loop = target, False
        while step in accepted:
            step = accepted[step][1]
            if step == i:
                loop = True
                break
        if loop:
            if dropped is not None:
                dropped.append((i, dimension, target))
            continue
        accepted[i] = (dimension, target)
    order, placed = [], set()

    def place(i):
        if i in placed or i not in accepted:
            return
        placed.add(i)
        place(accepted[i][1])
        order.append((i,) + accepted[i])

    for i in sorted(accepted):
        place(i)
    return order


def _compose_draw(panels, places, heights, columns, cell, gut_x, gut_y, whole, rc, family,
                  letter_pt, letter_mm, lettered, scales, offsets, sizes, spec):
    """One drawing of the composition: ``(figure, parts, findings, boxes)``, where
    ``boxes[index]`` holds the panel's ``cell``, ``content`` and plot/image ``axes``
    rectangles in mm (figure coordinates, origin bottom left), measured after layout."""
    import matplotlib as mpl

    fig = _new_figure(whole, rc)
    titles = fig._smtracker_titles = []
    width_ratios = [r for c in range(columns) for r in ((cell, gut_x) if c < columns - 1
                                                        else (cell,))]
    height_ratios = [r for i, h in enumerate(heights)
                     for r in ((h, gut_y) if i < len(heights) - 1 else (h,))]
    grid = fig.add_gridspec(len(height_ratios), len(width_ratios), width_ratios=width_ratios,
                            height_ratios=height_ratios)
    parts, findings, holders = [], [], {}
    with mpl.rc_context(rc):
        for index, r, first, span, rspan in places:
            panel = panels[index]
            scale = scales[index]
            cw, ch, w, h = sizes[index]
            pspec = _panel_spec(spec, panel, w, h)
            sub = fig.add_subfigure(grid[2 * r:2 * (r + rspan) - 1,
                                         2 * first:2 * (first + span) - 1])
            target = sub
            if scale < 1.0 - 1e-6:
                # SMALLER IN ITS CELL, drawn again at that size: text keeps its point size
                # and a PDF or SVG stays vector, so nothing is lost. Centred, or where a
                # snap put it (`offsets`, mm from the cell's top left).
                dx, dy = offsets[index] if offsets[index] is not None else \
                    ((cw - w) / 2.0, (ch - h) / 2.0)
                # NO SPACER UNDER HALF A MILLIMETRE. A panel filling its cell one way left
                # a 1e-6 mm strip either side, and matplotlib's layout solver ABORTED the
                # whole process on it -- no exception, SMTracker gone (a bubble chart at
                # 80 % in a 3-column grid was enough; found by a random sweep of layouts).
                cols = [c for c in (("pad", dx), ("panel", w), ("pad", cw - w - dx))
                        if c[0] == "panel" or c[1] >= MIN_SPACER_MM]
                rows = [r for r in (("pad", dy), ("panel", h), ("pad", ch - h - dy))
                        if r[0] == "panel" or r[1] >= MIN_SPACER_MM]
                inner = sub.add_gridspec(
                    len(rows), len(cols), width_ratios=[c[1] for c in cols],
                    height_ratios=[r[1] for r in rows],
                    left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
                target = sub.add_subfigure(inner[[r[0] for r in rows].index("panel"),
                                                 [c[0] for c in cols].index("panel")])
            letter = lettered[index]
            layout = None
            if panel["kind"] == "spatial_maps":
                layout = _spatial_layout(panel["payload"], pspec, family,
                                         top=letter_mm + 0.8 if letter and scale >= 1.0
                                         else 1.0)
                findings += [f"panel {letter or index + 1}: {f}"
                             for f in layout.get("findings", ())]
            elif panel["kind"] == "table":
                layout = _table_layout(panel["payload"], pspec, family,
                                       top=letter_mm + 0.8 if letter and scale >= 1.0 else 0.0)
                findings += [f"panel {letter or index + 1}: {f}"
                             for f in layout.get("findings", ())]
            table, more, legend, notes = _draw_kind(target, panel["kind"], panel["payload"],
                                                    pspec, family, layout)
            titled = bool(pspec.title) and panel["kind"] not in _MANUAL_KINDS
            if titled:
                # The panel's title over the MIDDLE OF ITS PLOT, as the same chart exported alone
                # (`_centre_title`), moved there once the layout has run (`_centre_panel_titles`).
                # `_draw_kind` placed it with NO POSITION, so the layout keeps its room. It was
                # re-placed here at x = right of the letter, y = 1.0: flush left, off the plot's
                # middle by half the plot, and (an explicit y) given no room -- drawn into the plot
                # (reported 30 Oct 2026: "the title is not in the middle of bubble plot anymore but
                # left"). The letter shares the top line: a title too wide to centre starts right of it.
                # The letter's room, a fraction of the panel: the title stays right of it, and `_fit_texts`
                # counts it out of the title's room.
                room = (_advance_pt(letter, letter_pt, family, "bold") + letter_pt * 0.6) * _PT_MM \
                    if letter else 0.0
                target._smtracker_keep_left = room / max(w, 1e-9)
                titles.append(target)
            if letter and not (titled and target is sub):
                # The letter's ROOM: a blank title of its size, so the layout keeps the corner
                # free. The letter itself is drawn below, over everything.
                sub.suptitle(" ", x=0.0, y=1.0, ha="left", va="top", fontsize=letter_pt,
                             fontweight="bold")
            if letter:
                # THE LETTER IS ALWAYS ON TOP (reported: "sometimes A, B etc is behind the
                # image ... should always be on top"). It is the WHOLE figure's text, drawn
                # after every panel (a figure draws its children by z-order, every panel's
                # sub-figure at 0), placed by the panel's own transform, so it lands in the
                # panel's corner wherever the layout put that panel -- and nothing a panel
                # draws, shrunk, snapped or moved, can cover it.
                fig.text(0.0, 1.0, letter, transform=sub.transSubfigure, ha="left",
                         va="top", fontsize=letter_pt, fontweight="bold", zorder=10,
                         color=_page(pspec)["ink"])
            holders[index] = (sub, target)
            parts.append({"index": index, "letter": letter, "kind": panel["kind"],
                          "title": panel.get("title") or KINDS[panel["kind"]],
                          "table": table, "notes": tuple(notes), "more": tuple(more),
                          "legend": legend})
    boxes = {}
    if any(p.get("match") for p in panels):
        with mpl.rc_context(rc):
            fig.draw_without_rendering()
        per_mm = fig.dpi / 25.4

        def mm(bbox):
            return (bbox.x0 / per_mm, bbox.y0 / per_mm, bbox.x1 / per_mm, bbox.y1 / per_mm)

        for index, (sub, target) in holders.items():
            drawn = [ax for ax in target.axes if ax.get_visible()
                     and ax.get_label() != "<colorbar>"]
            if not drawn:
                continue
            # DISPLAY coordinates: inside a sub-figure an axes' `get_position()` is a
            # fraction of THAT sub-figure, and reading it as the page's put every plot
            # 180 mm wide.
            extents = [ax.bbox for ax in drawn]
            union = (min(e.x0 for e in extents), min(e.y0 for e in extents),
                     max(e.x1 for e in extents), max(e.y1 for e in extents))
            boxes[index] = {"cell": mm(sub.bbox), "content": mm(target.bbox),
                            "axes": tuple(v / per_mm for v in union), "holder": target}
    return fig, parts, findings, boxes


def _shift_panel(fig, holder, shift_mm) -> None:
    """Move every axes a panel drew (its plot, its colour bar, the text they carry) by
    ``shift_mm`` = (x, y) on the page. The layout must already be frozen, or the next draw
    would put them back."""
    from matplotlib.transforms import Bbox

    per_mm = fig.dpi / 25.4
    inverse = holder.transSubfigure.inverted()
    for ax in holder.axes:
        box = ax.get_window_extent()
        moved = Bbox.from_extents(box.x0 + shift_mm[0] * per_mm, box.y0 + shift_mm[1] * per_mm,
                                  box.x1 + shift_mm[0] * per_mm, box.y1 + shift_mm[1] * per_mm)
        ax.set_position(inverse.transform_bbox(moved))


# -- checks ---------------------------------------------------------------------- #
def check_figure(built: Built) -> list:
    """What a journal or a reader would trip on, at the final size. Never raises."""
    import matplotlib as mpl
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.lines import Line2D

    fig = built.figure
    found = list(built.layout_findings)
    # What a drawing knew it could not do, kept on its axes: a picked pair with no p.
    found += [line for holder in fig.findobj(lambda a: bool(getattr(a, "_smtracker_findings",
                                                                     None)))
              for line in holder._smtracker_findings]
    with mpl.rc_context(built.rc):
        canvas = FigureCanvasAgg(fig)
        renderer = canvas.get_renderer()
        fig.draw(renderer)
        drawn = _drawn_texts(fig, renderer)
        texts = [t for t, _box in drawn]
        floor = built.spec.min_pt()
        journal = PRESET_LABELS[built.spec.preset]
        small = sorted({round(t.get_fontsize(), 1) for t in texts
                        if t.get_fontsize() < floor - 1e-6})
        if small:
            found.append(f"text at {', '.join(f'{s:g}' for s in small)} pt is under the "
                         f"{floor:g} pt {journal} accepts; raise the font size or use a wider figure")
        tallest = built.spec.preset_values.get("max_height_mm")
        if tallest and built.size_mm()[1] > tallest + 1e-6:
            found.append(f"{built.size_mm()[1]:g} mm is taller than the {tallest:g} mm "
                         f"{journal} allows; reduce the height")
        frame = fig.bbox
        boxes = [(t, box) for t, box in drawn if box.width > 0 and box.height > 0]
        for t, box in boxes:
            if _past(box, frame):
                found.append(f"“{_short(t.get_text())}” runs off the figure; shorten it or widen the figure")
        for i, (a, ba) in enumerate(boxes):
            for b, bb in boxes[i + 1:]:
                if _clash(ba, bb):
                    found.append(f"“{_short(a.get_text())}” overlaps "
                                 f"“{_short(b.get_text())}”; shorten one or use a smaller font")
        # A TEXT ON A COLOUR BAR: any other text touching a bar's strip (found by the overnight tester, 4 Nov 2026: a
        # slide's scale-bar length touched the bar under it and nothing was reported). Touching, not the 15 % two texts
        # may share: a glyph on a strip of colour is read as part of the scale.
        for ax in fig.axes:
            if not (getattr(ax, "_smtracker_colour_bar", False) and ax.get_visible()):
                continue
            strip = ax.get_window_extent(renderer)
            for t, box in boxes:
                if t.axes is ax:
                    continue
                w = min(box.x1, strip.x1) - max(box.x0, strip.x0)
                h = min(box.y1, strip.y1) - max(box.y0, strip.y0)
                if w > 0.5 and h > 0.5:
                    found.append(f"“{_short(t.get_text())}” touches a colour bar; move it clear of the bar")
        thin = [ln for ln in fig.findobj(Line2D)
                if ln.get_visible() and 0 < ln.get_linewidth() < MIN_LINE_PT
                and ln.get_linestyle() not in ("None", "none", "")]
        if thin:
            found.append(f"{len(thin)} line(s) under {MIN_LINE_PT:g} pt may vanish in print; make them thicker")
        # NOTES, NOT PROBLEMS: what `_fit_texts` set smaller or wrapped to fit (`FigureNote`)
        found += _fit_notes(texts, built.spec)
    return list(dict.fromkeys(found))


def _drawn_texts(fig, renderer) -> list:
    """``[(text, box)]``: every text ``fig`` DRAWS (visible, not blank), its box as wide as the PDF draws it
    (`_honest_box`: judged at 100 dpi alone, a panel title cut by 3 mm in the PDF and the 600 dpi PNG read clean, 30
    Oct 2026). ``fig`` must just have been drawn with ``renderer``. ONLY WHAT IS DRAWN: an axis keeps tick labels for
    ticks just past its end that it never draws (a 0.18 s tick on a 0.17 s axis), and counting them reported text
    "running off the figure" that is not there."""
    from matplotlib.text import Text

    spare = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            ticks = axis.get_major_ticks() + axis.get_minor_ticks()
            if not axis.get_visible() or not ax.axison:
                # A twin axis's shared x is hidden whole, and a map's axes are off:
                # their labels are never drawn.
                spare.update(id(label) for tick in ticks
                             for label in (tick.label1, tick.label2))
                spare.add(id(axis.label))
                continue
            try:
                drawn = {id(tick) for tick in axis._update_ticks()}
            except Exception:                           # noqa: BLE001 - private API
                continue
            spare.update(id(label) for tick in ticks if id(tick) not in drawn
                         for label in (tick.label1, tick.label2))
    return [(t, _honest_box(t, t.get_window_extent(renderer), fig.dpi)) for t in fig.findobj(Text)
            if t.get_visible() and t.get_text().strip() and id(t) not in spare]


def _past(box, frame) -> bool:
    """``box`` runs past ``frame`` (more than a pixel at 100 dpi, a quarter millimetre)."""
    return (box.x0 < frame.x0 - 1 or box.y0 < frame.y0 - 1
            or box.x1 > frame.x1 + 1 or box.y1 > frame.y1 + 1)


def _clash(ba, bb) -> bool:
    """Two text boxes overlap by more than a pixel each way and 15 % of the smaller."""
    w = min(ba.x1, bb.x1) - max(ba.x0, bb.x0)
    h = min(ba.y1, bb.y1) - max(ba.y0, bb.y0)
    return w > 1 and h > 1 and w * h > 0.15 * min(ba.width * ba.height, bb.width * bb.height)


def _honest_box(text, box, dpi: float):
    """``box`` (``text`` drawn at ``dpi``) widened, along the text and by its alignment, to the width the font's own
    metrics draw it at -- a PDF's, an SVG's, a 600 dpi PNG's -- where that is wider (`_width_pt`). Upright and
    quarter-turned text only; mathtext is left as drawn."""
    from matplotlib.transforms import Bbox

    turn = round(text.get_rotation()) % 180
    if turn not in (0, 90) or _math(text):
        return box
    per_pt = dpi / 72.0
    extra = _own_pt(text.get_text(), text.get_fontproperties()) * per_pt - (box.width if turn == 0 else box.height)
    if extra <= 0.5:
        return box
    if turn == 0:
        ha = text.get_horizontalalignment()
        left = extra if ha == "right" else extra / 2.0 if ha == "center" else 0.0
        return Bbox.from_extents(box.x0 - left, box.y0, box.x1 + extra - left, box.y1)
    return Bbox.from_extents(box.x0, box.y0 - extra / 2.0, box.x1, box.y1 + extra / 2.0)


def _short(text: str, n: int = 28) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n - 1] + "…"


# -- preview and files ----------------------------------------------------------- #
def preview_png(built: Built, width_px: int) -> bytes:
    """The figure as PNG bytes, ``width_px`` wide, for the dialog's preview."""
    import matplotlib as mpl

    fig = built.figure
    dpi = max(20.0, float(width_px) / fig.get_figwidth())
    buffer = io.BytesIO()
    with mpl.rc_context(built.rc):
        fig.savefig(buffer, format="png", dpi=dpi,
                    facecolor=built.rc.get("savefig.facecolor", "white"))
    return buffer.getvalue()


def _base(path) -> Path:
    """The path without a format extension the user may have typed."""
    path = Path(path)
    if path.suffix.lower().lstrip(".") in FORMATS:
        path = path.with_suffix("")
    return path


def _write_csv(path: Path, columns, rows) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(["" if v is None else v for v in row] for row in rows)


def write_figure(built: Built, path, *, provenance=None, stamp: str = "", composition=None) -> list:
    """Write the figure in every chosen format, plus the numbers, the drafted legend,
    the provenance and ``.figure.json``. Returns the paths written.

    ``composition`` (a Composer figure: ``{"composed": {"panels", "rows", "letters"},
    "spec": the Composer's settings}``) also writes ``<name>.composer``, the composition file that
    opens the figure again in the Composer (`composer.write_composition`), listed in
    ``.figure.json``. A composition that cannot be written raises (`composer.ComposerError`) after
    everything else is on disk, ``.figure.json`` saying why (``"composition_not_written"``)."""
    import matplotlib as mpl

    base = _base(path)
    base.parent.mkdir(parents=True, exist_ok=True)
    spec, fig = built.spec, built.figure
    creator = f"SMTracker {stamp}".strip()
    written = []
    with mpl.rc_context(built.rc):
        for fmt in spec.formats:
            out = Path(f"{base}.{fmt}")
            kwargs = {"format": fmt, "facecolor": built.rc.get("savefig.facecolor", "white")}
            if fmt in ("png", "tiff"):
                kwargs["dpi"] = spec.dpi
            if fmt == "tiff":
                kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
            if fmt in ("pdf", "svg"):
                kwargs["metadata"] = {"Creator": creator}
            if fmt == "png":
                kwargs["metadata"] = {"Software": creator}
            fig.savefig(out, **kwargs)
            written.append(out)
    when = datetime.now().isoformat(timespec="seconds")
    if built.parts:
        # A COMPOSED figure: one sheet per panel (its further sheets after it, named for
        # it) and one CSV per panel, so every number keeps the letter it is printed under.
        sheets = []
        for number, part in enumerate(built.parts, 1):
            tag = part["letter"] or str(number)
            columns, rows = part["table"]
            table_path = Path(f"{base}.{tag}.csv")
            _write_csv(table_path, columns, rows)
            written.append(table_path)
            sheets.append((f"{tag} {part['title']}", f"Source data, panel {tag}: "
                           f"{part['title']}", (*part["notes"], f"Exported {when} by {creator}."),
                           columns, rows))
            sheets += [(f"{tag} {name}", title, notes, cols, rws)
                       for name, title, notes, cols, rws in part["more"]]
        written.append(write_source_workbook(f"{base}.source_data.xlsx", sheets))
    else:
        columns, rows = built.table
        table_path = Path(f"{base}.csv")
        _write_csv(table_path, columns, rows)
        written.append(table_path)
        header = (*built.notes_for_source, f"Exported {when} by {creator}.")
        written.append(write_source_workbook(
            f"{base}.source_data.xlsx",
            [(built.title, f"Source data: {built.title}", header, columns, rows),
             *built.more_sheets]))
    legend_path = Path(f"{base}.legend.txt")
    legend_path.write_text(
        "Draft figure legend. Check every sentence against your analysis before use.\n\n"
        + built.legend + "\n", encoding="utf-8")
    written.append(legend_path)
    if provenance is not None and hasattr(provenance, "write_beside"):
        written.append(Path(provenance.write_beside(written[0])))
    composition_file = composition_problem = None
    if composition is not None:
        from .composer import ComposerError, write_composition

        try:
            composition_file = write_composition(Path(f"{base}.composer"), composition["composed"],
                                                 composition["spec"], build=creator)
            written.append(composition_file)
        except ComposerError as exc:
            composition_problem = exc                          # raised once the figure is whole on disk
    width, height = built.size_mm()
    record = {
        "kind": built.kind,
        "title": built.title,
        "spec": spec.record(),
        "size_mm": [width, height],
        "font_used": built.family,
        "notes": built.notes,
        "checks": check_figure(built),
        "panels": [{"letter": p["letter"], "kind": p["kind"], "title": p["title"]}
                   for p in built.parts],
        "files": [p.name for p in written],
        "written": datetime.now().isoformat(timespec="seconds"),
        "software": creator,
    }
    if composition_file is not None:
        record["composition"] = composition_file.name       # opens the figure again in the Composer
    if composition_problem is not None:
        record["composition_not_written"] = str(composition_problem)
    record_path = Path(f"{base}.figure.json")
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    written.append(record_path)
    if composition_problem is not None:
        raise composition_problem
    return written


def export_many(jobs, folder, *, stamp: str = "", progress=None) -> list:
    """Every job written into ``folder``, one after another -- "export every figure".

    A job is ``{"kind", "payload", "spec", "name", "provenance"}``; ``kind`` "composed"
    takes ``payload = {"panels", "columns", "letters"}`` (the composer's figure). A job the
    exporter REFUSES (`FigureExportError`: nothing drawn yet, an image missing) is kept as a
    row with its reason and the rest carry on; anything else raises, as everywhere here.
    Two jobs with one name get ``_2``, ``_3``. ``figures.txt`` beside them lists every figure,
    its files and its checks, so a folder written after a re-run says what is in it.
    Returns one row per job: ``{"name", "kind", "files", "checks", "error"}``."""
    from .composer import ComposerError

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    jobs = list(jobs)
    used: set = set()
    rows = []
    for index, job in enumerate(jobs):
        base = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(job["name"]))
        name, n = base or "figure", 2
        while name in used:
            name, n = f"{base}_{n}", n + 1
        used.add(name)
        if progress is not None:
            progress(index, len(jobs), name)
        try:
            composition = None
            if job["kind"] == "composed":
                panels = job["payload"]
                built = compose(panels["panels"], job["spec"], columns=panels.get("columns", 2),
                                rows=panels.get("rows"), letters=panels["letters"])
                # the composed figure's composition file beside it (`composer.write_composition`)
                composition = {"composed": panels, "spec": job["spec"]}
            else:
                built = build_figure(job["kind"], job["payload"], job["spec"])
            note = None
            try:
                files = [p.name for p in write_figure(built, folder / name, provenance=job.get("provenance"),
                                                      stamp=stamp, composition=composition)]
            except ComposerError as exc:
                # the figure is written; only its composition file is not: said, the rest carry on
                files = sorted(p.name for p in folder.glob(f"{name}.*"))
                note = str(exc)
            rows.append({"name": name, "kind": job["kind"], "files": files,
                         "checks": check_figure(built), "error": None, "note": note})
        except FigureExportError as exc:
            rows.append({"name": name, "kind": job["kind"], "files": [], "checks": [],
                         "error": str(exc)})
    lines = [f"Figures written {datetime.now().isoformat(timespec='seconds')} by "
             f"{('SMTracker ' + stamp).strip()}", ""]
    for row in rows:
        if row["error"]:
            lines.append(f"{row['name']} ({row['kind']}): NOT WRITTEN -- {row['error']}")
            continue
        lines.append(f"{row['name']} ({row['kind']}): {', '.join(row['files'])}")
        lines += [f"    check: {c}" for c in row["checks"]]
        if row.get("note"):
            lines.append(f"    {row['note']}")
    (folder / "figures.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def with_changes(spec: FigureSpec, **changes) -> FigureSpec:
    """``spec`` with ``changes``; the dialog's one way to edit a spec."""
    return replace(spec, **changes)
