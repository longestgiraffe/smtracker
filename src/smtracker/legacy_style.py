"""SMTracker 2.0's look, for the legacy window: the colours and matplotlib settings (`docs/legacy_window_plan.md`).

Headless (no Qt): the drawing modules `legacy_draw_*` read these, and `smtracker.ui.legacy.style` re-exports them beside
its style sheet. 2.0 was one white MATLAB figure (`SMTracker.m:66-77`): axes at font size 14 with the box on.
"""
from __future__ import annotations

#: MATLAB's figure and panel colours (`Color [1 1 1]`, the Console's 0.95 grey, the table stripe 0.94).
WHITE = "#ffffff"
CONSOLE_GREY = "#f2f2f2"
STRIPE_GREY = "#f0f0f0"
PANEL_EDGE = "#a8a8a8"
#: The menus' colours (27 Nov 2026). The app's palette follows the saved light/dark scheme, and a menu the style sheet
#: left unpainted took its ground and its hover from that palette while the sheet forced black text: dark on dark
#: when hovered. Every menu colour is set here, so the window reads the same in both schemes.
MENU_TEXT = "#000000"
MENU_HOVER = "#cce8ff"
MENU_DISABLED = "#767676"

#: MATLAB's default axes colour order since R2014b, for anything 2.0 drew in its default colours.
MATLAB_LINES = ("#0072bd", "#d95319", "#edb120", "#7e2f8e", "#77ac30", "#4dbeee", "#a2142f")
#: 2.0's confined-map classes (manual p. 11; 2021 paper Fig. 3): free blue, mixed green, dwelling red.
SLA_COLOURS = {"free": "#0000ff", "mixed": "#00b050", "dwelling": "#ff0000"}
#: 2.0's apparent-D populations (manual p. 20): static red, mobile blue, fast mobile green.
APPD_POP_COLOURS = ("#ff0000", "#0000ff", "#00a000")
#: 2.0's MSD and SQD line styles by population count (`plotCDF`, `SMTracker.m:5330`): 1 dotted, 2 solid, 3 dashed.
POP_LINESTYLES = {1: ":", 2: "-", 3: "--"}
#: The font 2.0's figures and uicontrols read in (MATLAB's default sans on each system).
FONT_FAMILY = ("Helvetica", "Arial", "Inter", "DejaVu Sans")
#: `DefaultAxesFontSize 14` (`SMTracker.m:70`), scaled to the window's smaller panels.
AXES_FONT_PT = 11

def matlab_rc() -> dict:
    """matplotlib settings for an axes that reads as MATLAB's: box on, ticks in and mirrored, no top-right open
    frame, MATLAB's sans at `AXES_FONT_PT`, white ground."""
    return {
        "font.family": "sans-serif",
        "font.sans-serif": list(FONT_FAMILY),
        "font.size": AXES_FONT_PT,
        "axes.titlesize": AXES_FONT_PT,
        "axes.titleweight": "bold",
        "axes.labelsize": AXES_FONT_PT,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#262626",
        "axes.facecolor": WHITE,
        "figure.facecolor": WHITE,
        "axes.prop_cycle": __import__("cycler").cycler(color=list(MATLAB_LINES)),
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.labelsize": AXES_FONT_PT - 1,
        "ytick.labelsize": AXES_FONT_PT - 1,
        "grid.color": "#262626",
        "grid.alpha": 0.15,
        "grid.linewidth": 0.5,
        "legend.frameon": True,
        "legend.edgecolor": "#262626",
        "legend.fancybox": False,
        "legend.fontsize": AXES_FONT_PT - 2,
        "lines.linewidth": 1.0,
    }


def boxplot_style() -> dict:
    """MATLAB's `boxplot` look: blue box, red median, black dashed whiskers, red '+' outliers."""
    return {
        "boxprops": {"color": "#0000ff", "linewidth": 0.8},
        "medianprops": {"color": "#ff0000", "linewidth": 0.8},
        "whiskerprops": {"color": "#000000", "linestyle": "--", "linewidth": 0.8},
        "capprops": {"color": "#000000", "linewidth": 0.8},
        "flierprops": {"marker": "+", "markeredgecolor": "#ff0000", "markersize": 5},
    }
