"""Figure style + export foundation, defaults and pipeline, not final aesthetics.

This is the *foundation*: a colourblind-safe palette, a fixed state-colour
grammar shared across every view, journal-preset dimensions and publication font
sizes, and a vector + high-DPI raster export path off the single render layer.
The final "does it look Nature-ready" tuning is deliberately left for the experimenter,
this makes that tuning a one-line change, not a rebuild.

Why these choices
-----------------
- **Okabe–Ito** is the default categorical palette: eight hues distinguishable
  under the common colour-vision deficiencies, so a figure is readable by
  everyone by default rather than as an afterthought.
- **The state-colour grammar is fixed and global.** Slow/bound, fast, and the
  third state get the *same* colours in the fraction bar, the SuperPlot, the HMM
  view and the maps, so a reader learns the code once. ``smtracker.plots`` imports
  it from here, so there is one source of truth.
- **Journal presets** carry real column widths (Nature 89/183 mm, Cell 85/174 mm,
  eLife 85/174 mm) and publication point sizes, so a figure is laid out at its
  final size from the start: no rescaling surprises.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

__all__ = [
    "FigStyleError",
    "OKABE_ITO",
    "STATE_COLORS",
    "STATE_NAMES",
    "CONDITION_COLORS",
    "CONDITION_FILLS",
    "JOURNAL_PRESETS",
    "rc_for",
    "style",
    "figure_size_inches",
    "export_figure",
]


class FigStyleError(RuntimeError):
    """Raised when a style or export is misconfigured."""


#: Okabe–Ito colourblind-safe categorical palette (8 hues + black).
OKABE_ITO = [
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
]

#: The GLOBAL state-colour grammar, the same everywhere. Drawn from Okabe–Ito.
STATE_COLORS = {
    1: "#0072B2",   # slow / bound  , blue
    2: "#D55E00",   # fast          , vermillion
    3: "#009E73",   # third state   , bluish green
    -1: "#999999",  # no state / gap, grey
}
STATE_NAMES = {1: "slow / bound", 2: "fast", 3: "third state", -1: "no state (gap)"}

#: The GLOBAL condition-colour grammar. Teal for glucose, terracotta for malate, with
#: paler fills for interval bands - lifted from the six manuscript figure scripts that
#: already agreed on it (`figure_manuscript_files`, `figure_manuscript_spatial`,
#: `figure_spatial_tca`, `figure_peripheral_counts`, `figure_hmm_four_panel`,
#: `figure_hmm_singles`).
#:
#: IT LIVES HERE BECAUSE IT WAS REDEFINED IN EVERY SCRIPT AND HAD ALREADY DRIFTED into
#: three different pairs: this one in six scripts, "#0072B2"/"#D55E00" in
#: `figure_manuscript_main` (plus `confinement_simulation` and `cell_geometry_report`),
#: and "#2b6cb0"/"#c05621" in `compare_spatial_glucose_malate`. Those three are NOT
#: reconciled here - changing a shipped figure's colours is the experimenter's call, not
#: a tidy-up - but a new figure now has one place to read them from.
#:
#: DELIBERATELY NOT the Okabe-Ito blue/vermillion pair: `STATE_COLORS` already owns
#: "#0072B2" for slow and "#D55E00" for fast, so using them for conditions would make one
#: colour mean two things in any figure that shows both, which is exactly the figure this
#: grammar exists for.
CONDITION_COLORS = {"glucose": "#49A88D", "malate": "#D47B59"}

#: The Apparent Diffusion distribution's ink, the legacy figure's choice: per-cell points
#: by replicate day (single session: one slate), the violin a neutral slate. Here so the
#: screen chart and its exported figure read them from one place.
DAY_COLORS = ("#3182ce", "#e53e3e", "#38a169", "#805ad5", "#dd6b20", "#319795")
SINGLE_SESSION_COLOR = "#4a5568"
VIOLIN_FILL = "#a0aec0"


def day_shade(base: str, index: int, n_days: int) -> str:
    """Replicate day ``index`` (0-based) of ``n_days`` as a SHADE of its condition's colour
    ``base``: the first day darkest, the last lightest, ``base`` in between.

    Asked for: "the day colour could be the same color as the main bubble color just in
    different shades". A condition then keeps ONE colour everywhere -- the days are told
    apart by lightness inside it -- where a separate day palette put a sixth set of
    colours on a chart whose colours already meant conditions. Spans 30 % toward black to
    40 % toward white, so three days stay apart and none fades into the background."""
    value = base.lstrip("#")
    rgb = [int(value[i:i + 2], 16) for i in (0, 2, 4)]
    if n_days <= 1:
        return "#" + value.lower()
    t = -0.30 + 0.70 * index / (n_days - 1)
    if t < 0:
        mixed = [c * (1 + t) for c in rgb]
    else:
        mixed = [c + (255 - c) * t for c in rgb]
    return "#" + "".join(f"{round(min(255, max(0, c))):02x}" for c in mixed)
CONDITION_FILLS = {"glucose": "#94D4C0", "malate": "#FDAF91"}

#: Panel letters per preset, ``(case, size pt)``, drawn bold and upright. NATURE is its own
#: rule: "8 pt bold, upright (not italic) a, b, c" (Guide to Preparing Final Artwork,
#: nature.com/documents/nature-final-artwork.pdf), checked 10 Oct 2026. CELL Press: uppercase
#: bold from SECONDARY summaries only (cell.com refused the request), 8 pt from their
#: 8-12 pt. eLIFE: no stated rule found; its published figures letter panels A, B, C, so
#: uppercase 8 pt is THIS TOOL'S ASSUMPTION. Screen and slides are this tool's own. Every
#: one is only the composer's default -- the letter case is a choice there.
PANEL_LETTERS = {"nature": ("lower", 8.0), "cell": ("upper", 8.0), "elife": ("upper", 8.0),
                 "screen": ("upper", 13.0), "slides": ("upper", 20.0),
                 "slides_dark": ("upper", 20.0), "poster": ("upper", 32.0)}

#: Journal presets: column widths (mm), body and minimum text sizes (pt), and the maximum
#: height where the journal states one. CHECKED 10 Oct 2026 against the journals' guides:
#: Nature 89 / 183 mm, max height 170 mm, text 5-7 pt, Helvetica or Arial, panel letters
#: 8 pt bold lowercase (research-figure-guide.nature.com) -- Nature states NO 1.5-column
#: width, so there is none here (an earlier 120 mm was invented and is gone); Cell Press
#: 85 / 114 / 174 mm, 6-8 pt (Cell Reports' guide; cell.com refused the request); eLife
#: 85 / 170 mm, at least 7 pt (secondary summaries only: eLife's own page was not
#: reachable -- recheck before relying on it). ``screen`` and ``slides`` are this tool's own:
#: a 16:9 slide inserted at 100 % is 338 mm wide, so a full-width figure is 300 mm. They (and the poster) state
#: NO 1.5-column width since 3 Nov 2026: theirs was 1.5 x one column, three QUARTERS of the full width, which
#: the Export dialog's 2/3 would have misnamed; their 2/3 is now the full width divided like every other
#: fraction (`figure_export.fraction_mm`).
JOURNAL_PRESETS = {
    "nature": {"single_mm": 89.0, "onehalf_mm": None, "double_mm": 183.0,
               "font_pt": 7.0, "min_font_pt": 5.0, "max_height_mm": 170.0,
               "title_pt": 8.0, "line_pt": 0.75},
    "cell": {"single_mm": 85.0, "onehalf_mm": 114.0, "double_mm": 174.0,
             "font_pt": 7.0, "min_font_pt": 6.0, "max_height_mm": None,
             "title_pt": 8.0, "line_pt": 0.75},
    "elife": {"single_mm": 85.0, "onehalf_mm": None, "double_mm": 170.0,
              "font_pt": 8.0, "min_font_pt": 7.0, "max_height_mm": None,
              "title_pt": 9.0, "line_pt": 1.0},
    "screen": {"single_mm": 160.0, "onehalf_mm": None, "double_mm": 320.0,
               "font_pt": 11.0, "min_font_pt": 5.0, "max_height_mm": None,
               "title_pt": 13.0, "line_pt": 1.5},
    "slides": {"single_mm": 150.0, "onehalf_mm": None, "double_mm": 300.0,
               "font_pt": 16.0, "min_font_pt": 5.0, "max_height_mm": None,
               "title_pt": 18.0, "line_pt": 1.5},
    #: Slides on a DARK background: the slide sizes, every colour read from a dark page.
    "slides_dark": {"single_mm": 150.0, "onehalf_mm": None, "double_mm": 300.0,
                    "font_pt": 16.0, "min_font_pt": 5.0, "max_height_mm": None,
                    "title_pt": 18.0, "line_pt": 1.5, "dark": True},
    #: A poster panel read from about two metres: this tool's own convention (no journal
    #: states one) -- 24 pt text, nothing under 18, on an A0 sheet's 250 / 375 / 500 mm.
    "poster": {"single_mm": 250.0, "onehalf_mm": None, "double_mm": 500.0,
               "font_pt": 24.0, "min_font_pt": 18.0, "max_height_mm": None,
               "title_pt": 28.0, "line_pt": 2.5},
}

_MM_PER_INCH = 25.4
#: Vector formats worth exporting; the rest are raster (need a dpi).
_VECTOR = {"svg", "pdf", "eps"}


def rc_for(preset: str = "screen") -> dict:
    """matplotlib rcParams for a journal preset: fonts, line widths, palette."""
    if preset not in JOURNAL_PRESETS:
        raise FigStyleError(
            f"unknown preset {preset!r}; choose from {sorted(JOURNAL_PRESETS)}")
    from cycler import cycler

    p = JOURNAL_PRESETS[preset]
    return {
        "font.size": p["font_pt"],
        "axes.titlesize": p["title_pt"],
        "axes.labelsize": p["font_pt"],
        "xtick.labelsize": p["font_pt"] - 1,
        "ytick.labelsize": p["font_pt"] - 1,
        "legend.fontsize": p["font_pt"] - 1,
        "lines.linewidth": p["line_pt"],
        "axes.linewidth": max(0.5, p["line_pt"] * 0.8),
        "axes.prop_cycle": cycler(color=OKABE_ITO[1:]),   # skip pure black for lines
        "svg.fonttype": "none",                            # editable text in SVG
        "pdf.fonttype": 42,                                # embed TrueType in PDF
        "figure.dpi": 100,
        # -- Aesthetic pass (item E): TYPOGRAPHY, SPACING, THEME ------------- #
        # A clean sans-serif stack (falls back gracefully when a face is absent);
        # softer near-black ink than pure #000 reads calmer at small sizes; open
        # frames (no top/right spine, no legend box) and outward ticks are the
        # modern journal look. All cosmetic: no text content or explicit position
        # moves, so the neutral/position gates are untouched.
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "mathtext.default": "regular",
        "text.color": "#1a202c",
        "axes.edgecolor": "#4a5568",
        "axes.labelcolor": "#1a202c",
        "axes.titlecolor": "#1a202c",
        "axes.titlelocation": "left",
        "axes.titlepad": 6.0,
        "axes.labelpad": 3.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": "#4a5568",
        "ytick.color": "#4a5568",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": max(0.5, p["line_pt"] * 0.8),
        "ytick.major.width": max(0.5, p["line_pt"] * 0.8),
        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "legend.borderaxespad": 0.4,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }


@contextmanager
def style(preset: str = "screen"):
    """Temporarily apply a journal preset's rcParams (restored on exit)."""
    import matplotlib as mpl

    with mpl.rc_context(rc_for(preset)):
        yield


def figure_size_inches(preset: str = "nature", *, width: str = "single",
                       aspect: float = 0.66) -> tuple[float, float]:
    """(width, height) in inches for a preset column width and aspect ratio."""
    if preset not in JOURNAL_PRESETS:
        raise FigStyleError(f"unknown preset {preset!r}")
    if width not in ("single", "onehalf", "double"):
        raise FigStyleError(
            f"width must be 'single', 'onehalf' or 'double', got {width!r}")
    if aspect <= 0:
        raise FigStyleError(f"aspect must be > 0, got {aspect}")
    mm = JOURNAL_PRESETS[preset][f"{width}_mm"]
    if mm is None:
        raise FigStyleError(f"{preset} states no {width} column width")
    w = mm / _MM_PER_INCH
    return (w, w * aspect)


def export_figure(figure, path, *, formats=("png", "svg", "pdf"), dpi=600,
                  provenance=None):
    """Export a Figure to vector + high-DPI raster, from the one render layer.

    Writes ``<path>.<ext>`` for each format (vectors for svg/pdf/eps, raster at
    ``dpi`` otherwise). If a ``provenance`` stamp is given, a companion
    ``<path>.provenance.json`` is written beside the first output so a shared
    figure remains traceable. Returns the list of written paths.
    """
    path = Path(path)
    if not formats:
        raise FigStyleError("no export formats given")
    written = []
    for fmt in formats:
        fmt = fmt.lower().lstrip(".")
        out = path.with_suffix(f".{fmt}")
        kwargs = {"bbox_inches": "tight"}
        if fmt not in _VECTOR:
            kwargs["dpi"] = dpi
        figure.savefig(out, format=fmt, **kwargs)
        written.append(out)
    if provenance is not None and written:
        provenance.write_beside(written[0])
    return written
