"""The workspace's left rail: shadcn's `Sidebar`, carrying the analysis nav.

Ported from `prototype/shadcn/workspace.py` (git history, commit 09da35d).
Two things were deliberately dropped in the port, not merely deferred:

**No multi-workspace switcher.** The prototype's header let you flip between
several mock datasets (`Ctrl 1` / `Ctrl 2` / a cross-workspace "Dashboard") --
a concept this application does not have: one onboarding flow loads one
project, and switching to another means going back through it. So the header
here is a plain "back to onboarding" trigger (`WorkspaceHeader.opened`, wired
by whoever builds this window), not a menu of workspaces that do not exist.

**Two sections so far, "SQD" and "Apparent-D Distribution".** The prototype's
rail held four sections and nine subsections; the old comparison browser's
tabs are being ported one at a time. `SECTIONS` is still a tuple of `(key,
title, glyph, subs)`, so adding the second one really was the one-line
change the first version of this note promised -- `Sidebar.__init__`'s own
`subs[0][0]` lookups mean every entry needs AT LEAST one sub even where
the page itself has only one view (see `window.py`'s own `_on_selected`).

Everything else -- the collapse tween, the tooltip-when-collapsed behaviour,
the truncating labels -- is the prototype's, unchanged in shape.

**THE THREE GROUPS (4 Nov 2026).** Asked: "make the left page overview a bit more
modern and better. MSD should be up, composer should stand out a bit more, images
and tracks, and qc as well" (and "maybe we can reorder it as well"). One flat list
of ten rows became three groups (`GROUPS`), each in the form that says what it is:

- **Data** -- QC and Images & tracks, two TILES side by side (`DataTile`: icon over
  label, shadcn's `outline` button), the look at the data before any fit;
- **Analyses** -- SQD, MSD (moved up to second), Apparent Diffusion, HMM, Spatial
  Map, Kinetics, Clustering: the `NavButton` rows as before, each in a
  `_SectionItem` that holds what belongs under it (`set_pages`' list today);
- **Figures** -- the Composer, pinned at the rail's FOOT under the Running list as
  the one filled control in the rail (`FeatureButton`: shadcn's `default` button,
  `bg-primary`), where the sections' figures are put together.

Collapsed to the icon rail the tiles morph into two icon cells, each group label
fades into a hairline (`_GroupLabel`, shadcn's `SidebarSeparator`) and the Composer
keeps its fill. The groups between the header and the foot scroll (`_BodyScroll`,
shadcn's `SidebarContent overflow-auto`), so the rail never pins the window's
minimum height. Another window's rail (preparation, outlines) passes its own
`sections` and stays ONE plain list of rows. The keys are unchanged.

**SUBPAGES, DESIGNED FOR, NOT BUILT.** Every analysis section will get a "+" that
adds a copy of it with its own settings and results ("SQD 2"), listed under the
section, renamable and reorderable. It slots in here without another redesign:
the "+" is shadcn's `SidebarMenuAction` (shown on hover) at the row's right end,
where `set_pages` already puts its chevron; the copies are a list in the row's
`_SectionItem`, under the row and above its pages, drawn as `_PageList` draws pages
(indented, a hairline down the left); a copy's views are keyed
``<section>~<n>.<page>`` so `select`, `trail` and the window's `<section>` split keep
working on the part before the dot. Data and Figures take no "+".
"""
from __future__ import annotations

from typing import ClassVar

from qtpy.QtCore import (QEasingCurve, QEvent, QPoint, QRect, QRectF, QSize, Qt,
                         QVariantAnimation, Signal)
from qtpy.QtGui import QColor, QFontMetrics, QPainter, QPen
from qtpy.QtWidgets import (QHBoxLayout, QLabel, QLayout, QScrollArea, QSizePolicy,
                            QVBoxLayout, QWidget)

from ... import subpages as sp
from ..theme import typography as ty
from ..widgets import tooltip
from ..widgets.menus import Panel, menu_row, retriggered
from ..widgets.primitives import HoverFill, ShadowHost, Surface, icon, label, mix, slide_ms
from ..widgets.primitives import discard, shadow_room

__all__ = ["GROUPS", "RAIL_W", "SECTIONS", "SIDEBAR_W", "SUBS", "Breadcrumb", "DataTile",
          "FeatureButton", "NavButton", "Sidebar", "SidebarTrigger", "trail_for"]

#: This panel is the registry's sidebar rendered at the registry's OWN
#: 16 px root -- every size below is `rem(...)` of the registry's own
#: class, read off `ui/sidebar.tsx` directly (`SIDEBAR_WIDTH` = 16rem,
#: `SidebarGroupLabel` = text-xs, `SidebarMenuButton` = text-sm, ...).
#:
#: **CORRECTED, and the correction is worth recording rather than just
#: applied.** The prototype's own `workspace.py` module docstring set
#: this to 18, reasoning that the reference page "renders [the sidebar]
#: at ~288" against the registry's own 256 (16rem at a 16 px root) --
#: and 18 is exactly 16 x (288/256). That 288 was a PIXEL COUNT measured
#: off a rendered screenshot, not the registry's own `16rem` class, taken
#: while developing on a Windows laptop at 125% OS scale -- asked
#: directly, months later, whether that scale could have distorted the
#: measurement. The exact ratio (1.125) does not match 125% scaling
#: cleanly (256 x 1.25 = 320, not 288), so the precise mechanism is not
#: pinned down -- but the SHAPE of the mistake is the same either way: a
#: physical/rendered pixel count stood in for the CSS logical value, and
#: because `rem(x) = round(x * ROOT_PX)` drives every size in this file
#: from the one constant, that single mismeasurement quietly inflated
#: the whole sidebar -- width, fonts, icons, row heights -- by the same
#: ~9-14% (rounding differences between `round()` and `int()` at
#: different call sites keep it from being a clean 1.125 everywhere).
#: Fixed by trusting the registry's literal numbers instead of a
#: re-measured screenshot; `toolbar.ROOT_PX` MUST be changed alongside
#: this one, per its own comment.
ROOT_PX = 16


def rem(x: float) -> int:
    return round(x * ROOT_PX)


#: `SIDEBAR_WIDTH` = 16rem, `SIDEBAR_WIDTH_ICON` = 3rem.
SIDEBAR_W = rem(16)
RAIL_W = rem(3)
#: Added to the widths above, not taken out of them -- a border-box border
#: eats a pixel and squeezes the workspace tile against it otherwise.
EDGE = 1

INSET_PAD = rem(0.5)
GAP = rem(0.5)
GLYPH = rem(1)
TILE = rem(2)

#: `h-8` / `h-7` / `h-12` -- a row, a nested row, the switcher-sized header.
ROW_H = rem(2)
SUB_H = rem(1.75)
SWITCHER_H = rem(3)

TRIGGER = rem(1.75)

TEXT_REM = 0.875
TEXT_SM_REM = 0.75
TEXT, TEXT_SM = "base", "sm"


def text_px(size: float) -> int:
    return int(size * ROOT_PX)


#: A COMPENSATING weight, not the spec's own number. The registry's own
#: `font-medium` is CSS weight 500, matched exactly by `WEIGHTS["medium"]`
#: (`typography.py`) -- but shadcn's own reference site renders in Geist,
#: not the Inter this application bundles (swapped in for a documented,
#: unrelated reason -- a missing glyph sent Qt walking 2,942 families for
#: a fallback; see `typography.py`'s own font-swap note), and the two
#: typefaces do not carry the same visual weight at the same numeric one.
#: Reported directly ("HiWi Test and SQD look bolder than their
#: counterparts") and MEASURED, not eyeballed: pixel-scanning the two
#: header titles in the SAME reference screenshot (identical capture
#: pipeline, so the comparison is not confounded by scale or compression)
#: gives a stroke-width-to-cap-height ratio of 0.150 for Geist's `font-
#: medium` against 0.190 for Inter's -- Inter's medium reads about 27 %
#: thicker at the identical CSS weight. Calibrating that SAME ratio
#: against Inter's own weight/thickness curve (rendered synthetically at
#: several `wght` values and pixel-measured the same way) lands the
#: match near Inter's own REGULAR weight, not a subtle trim off 500 --
#: 410 here, deliberately short of the noisy precision the measurement
#: does not actually support. Applied via the variable font's own `wght`
#: axis (`ty.ui`'s `wght=`), not by changing `weight="medium"` to
#: `"normal"` outright: the CODE still says which spec step a label is
#: (medium, matching the registry), this constant is what makes IT, on
#: THIS bundled face, look like that step is supposed to.
MEDIUM_WGHT = 410


TITLE_PX = round(TEXT_REM * ROOT_PX)
LEADING_TIGHT = 1.25
LABEL_ALPHA = 0.70
NAV_STROKE = 2.0

FULL_BTN = SIDEBAR_W - 2 * INSET_PAD
ICON_BTN = ROW_H

#: The groups' measures (4 Nov 2026). `SidebarMenu`'s `gap-1` between rows, a `p-2`'s worth between groups.
ROW_GAP = rem(0.25)
GROUP_GAP = rem(0.5)
#: A Data tile, `h-14`: an 18 px glyph over a 13 px label, `px-1.5` on each side ("Images & tracks" semibold is
#: 103 px of the 104 a tile has).
TILE_H = rem(3.5)
TILE_GLYPH = rem(1.125)
TILE_PAD = rem(0.375)
TILE_TOP = rem(0.6875)
TILE_TEXT_PX = ty.SIZES["control"]
#: The Composer, shadcn's `h-9` default button.
FEATURE_H = rem(2.25)
FEATURE_PAD = rem(0.75)
#: A group label, `h-7`: a step under shadcn's `h-8`, so two jobs still show whole under the groups on a
#: 1366 x 768 laptop (measured 4 Nov 2026: at `h-8` they were 19 px short with a two-line project name).
LABEL_H = rem(1.75)
#: A group label's height on the icon rail, where it is a hairline between the groups.
SEP_H = rem(0.5625)

#: The navigation tree: `(key, title, glyph, subs)`. `subs` is kept for a
#: future page's tab strip, which is where the prototype put subsections
#: rather than in this rail -- see the module docstring.
SECTIONS: tuple = (
    # "QC", first -- the original tab's own ruling was "run BEFORE any
    # condition comparison", and a rail ordering is a one-line, easily
    # reversed choice if this doesn't feel right once on screen. "check"
    # (a plain tick, distinct from the other three's chart/pulse/grid
    # shapes) rather than borrowing "aperture"/"scan", which read as
    # camera/detector glyphs and would misname a section about trusting
    # the DATA, not the acquisition. One sub, "Overview" -- the section is
    # one page (a General page and a Detail drill-in, not a tab strip).
    ("qc", "QC", "check", (("qc.overview", "Overview"),)),
    # "Images & tracks": v2's start screen -- a condition's movies with their outlines and
    # tracks (`images_page`). Beside QC because both are a look at the data before any fit.
    ("images", "Images & tracks", "image", (("images.overview", "Overview"),)),
    # Two pages, switched by the strip at the top of the section: the bubbles, and the
    # jump histogram against the 1-3 population fits with their goodness of fit. (Signed jumps, a third page from
    # 17 Oct 2026, was taken off on 22 Nov 2026: "i dont think we need signed jumps btw"; `smtracker.jd` stays.)
    ("sqd", "SQD", "chart", (("sqd.populations", "Populations"),
                             ("sqd.jumps", "Jump distances"))),
    # "MSD": v2's MSD tab, kept and improved (`msd_chart`; rebuilt 16 Oct 2026 into four pages the sub-views
    # follow). `"chart-spline"` -- a curve on axes, which is what the section draws; no other section uses it.
    # Second of the analyses since 4 Nov 2026 ("MSD should be up"); it was ninth, behind Clustering.
    ("msd", "MSD", "chart-spline",
     (("msd.curves", "Curves"), ("msd.checks", "Checks"))),
    # `"activity"` -- a pulse line, deliberately not `"chart"` again: the
    # two sections should never look like the same glyph at a glance in
    # the rail's own icon-only collapsed state. One sub, "Overview", not
    # "Distribution" or "Bubble Plot" -- the card itself now toggles
    # between those two DISPLAYS (`ApparentDCard`'s own `Segmented`), so
    # naming the sub after either one would misname it the moment the
    # OTHER mode is selected; "Overview" is the neutral term that covers
    # both.
    # Four pages since the rebuild of 16 Oct 2026 (`appd_section`), switched by the strip at the section's top.
    ("appd", "Apparent Diffusion", "activity",
     (("appd.populations", "Populations"), ("appd.distributions", "Distributions"),
      ("appd.cells", "Per cell"), ("appd.checks", "Checks"))),
    # "HMM": the diffusive-state HMM as a section of its own (`hmm_chart`), asked for as
    # "the hmm tab". `"arrow-left-right"` -- switching between two states, which is what it
    # measures; no other section uses it.
    # pages behind the section's strip (`hmm_chart.PAGES`), named by what they draw since 23 Nov 2026: the state
    # diagrams, the slow state in every condition, the Jump angles of this page's fits (a section of its own 22-23
    # Nov; its map is the Spatial Map's HMM Show), the table of one state
    ("hmm", "HMM", "arrow-left-right", (("hmm.cards", "State diagrams"), ("hmm.overview", "Slow state"),
                                        ("hmm.angles", "Jump angles"), ("hmm.table", "Table"))),
    # `"grid"` -- four panes, which is literally what this section draws
    # (conditions down, populations across) and, by the rule the comment
    # above records, the THIRD distinct glyph: not `"chart"` (SQD's) and
    # not `"activity"` (Apparent Diffusion's), so the collapsed icon-only
    # rail still tells the three apart at a glance.
    #
    # "Spatial Map", NOT "SQD map": the view is SQD-only today by
    # instruction, but `comparison.spatial_maps` already offers the same
    # grid from appD and HMM, and a section titled after one estimator
    # would be misnamed the moment a second is routed into it. One sub,
    # "Overview", matching `appd`'s -- the section is one page.
    ("spatial", "Spatial Map", "grid",
     (("spatial.overview", "Overview"),)),
    # "Kinetics": the triggered-kinetics tool (`smtracker.kinetics`), a section of its
    # own. `"clock"` -- time since a trigger is the whole design, and no other section
    # uses it. NOT the HMM switching rates, which live under Apparent Diffusion as
    # "Kinetics (HMM)"; that card is a different analysis with a different question.
    ("kinetics", "Kinetics", "clock",
     (("kinetics.overview", "Overview"),)),
    # "Clustering": v2's Clustering tab (`clustering_section`, 17 Oct 2026), k-means on the tracks' MSD curves
    # against a free-Brownian null. `"git-fork"` -- one set splitting into branches; no other section uses it.
    ("clusters", "Clustering", "git-fork",
     (("clusters.overview", "Overview"),)),
    # "Composer": the figure composer (`composer_page`), last because it assembles what the
    # sections above produce. `"panels-top-left"` -- one frame cut into panels.
    ("composer", "Composer", "panels-top-left",
     (("composer.overview", "Overview"),)),
)

#: The rail's groups, top to bottom: ``(key, label, form, section keys)`` (4 Nov 2026, the module docstring).
#: ``form`` is how the group's sections are drawn: ``"tiles"`` side by side (`DataTile`), ``"rows"`` one under
#: another (`NavButton`, each in a `_SectionItem`), ``"feature"`` one filled button pinned at the foot
#: (`FeatureButton`). Every section is in exactly one group, in `SECTIONS`' order (`_checked_groups`).
GROUPS: tuple = (
    ("data", "Data", "tiles", ("qc", "images")),
    ("analyses", "Analyses", "rows", ("sqd", "msd", "appd", "hmm", "spatial", "kinetics", "clusters")),
    ("figures", "Figures", "feature", ("composer",)),
)
FORMS = ("tiles", "rows", "feature")


def _checked_groups(groups, sections) -> tuple:
    """``groups`` as given, after checking them against ``sections``: every section in exactly one group, the
    groups in `sections`' order, a known form, at most one feature group (the foot holds one button). A rail that
    drops or doubles a section would hide a page with no error, so it is refused here, when it is built."""
    groups = tuple((str(k), str(lab), str(form), tuple(keys)) for k, lab, form, keys in groups)
    order = [k for k, *_ in sections]
    listed = [s for *_g, keys in groups for s in keys]
    if sorted(listed) != sorted(order) or len(set(listed)) != len(listed):
        raise ValueError(f"the sidebar's groups must hold every section once: groups {listed}, sections {order}")
    if listed != order:
        raise ValueError(f"the sidebar's groups must list the sections in their order: {listed} != {order}")
    bad = [form for *_k, form, _s in groups if form not in FORMS]
    if bad:
        raise ValueError(f"unknown sidebar group form(s) {bad}; known: {FORMS}")
    features = [g for g in groups if g[2] == "feature"]
    if len(features) > 1 or any(len(g[3]) != 1 for g in features):
        raise ValueError("the sidebar's foot holds one feature button")
    return groups


GROUPS = _checked_groups(GROUPS, SECTIONS)

#: Section key -> its subsections, indexed the way a page's tab strip wants.
SUBS: dict = {k: subs for k, _t, _g, subs in SECTIONS}
#: Section key -> its title, for `trail_for` and a future tab-strip breadcrumb.
SECTION_TITLE: dict = {k: t for k, t, _g, _s in SECTIONS}


def trail_for(view: str) -> list:
    """The breadcrumb trail for a `section.subsection` id: the section's own
    TITLE, then the subsection's own LABEL -- e.g. `["SQD", "Populations"]``
    for `"sqd.populations"`. Reads `SECTIONS` rather than title-casing the key
    (`"sqd.populations".replace(".", " / ").title()` gave `"Sqd / Populations"`,
    wrong for an acronym) so a section named something that does not title-case
    cleanly is never a rename away from a wrong breadcrumb.
    """
    section_key, _, _sub_key = view.partition(".")
    title = SECTION_TITLE.get(section_key)
    if title is None:
        return [view]
    sub = next((lab for key, lab in SUBS.get(section_key, ()) if key == view),
              None)
    return [title] if sub is None else [title, sub]


def _fade(a: str, b: str, k: float) -> str:
    ca, cb = QColor(a), QColor(b)
    return QColor.fromRgbF(
        ca.redF() + (cb.redF() - ca.redF()) * k,
        ca.greenF() + (cb.greenF() - ca.greenF()) * k,
        ca.blueF() + (cb.blueF() - ca.blueF()) * k).name()


class NavButton(Surface):
    """`SidebarMenuButton`: `h-8 rounded-md p-2 gap-2 text-sm`.

    One class covers a plain section row and the header (`size="lg"`, with a
    subtitle) -- see `WorkspaceHeader` below, which is this class with a tile
    glyph inserted ahead of it.
    """

    clicked = Signal()

    HEIGHTS: ClassVar[dict] = {}
    #: THE CURRENT SECTION, UNMISTAKABLE (asked 16 Oct 2026: "can we highlight it more on the menu to the left?"):
    #: shadcn's active row is the faint `sidebar_accent` fill and a medium weight, too close to the rest. Now a
    #: clearly darker tint of the sidebar's primary ink, a bar of it at the row's left edge, and a semibold title.
    ACTIVE_TINT = 0.11
    BAR_W = 3.0

    @staticmethod
    def active_fill(t) -> str:
        return mix(t.sidebar_primary, t.sidebar, NavButton.ACTIVE_TINT)

    def __init__(self, t, text: str = "", *, glyph: str | None = None,
                 size: str = "default", active: bool = False,
                 sub: str | None = None, trailing: str | None = None,
                 pad: int | None = None, bar: bool = True, parent=None):
        super().__init__(t, radius=t.radius_sm, shadowed=False, parent=parent)
        pad = INSET_PAD if pad is None else pad
        self.t = t
        self._active = active
        #: The current row's bar at the left edge (a section's page row has none: its section's
        #: own row already carries the bar, and two would read as two places at once).
        self._bar = bar
        self.setObjectName(f"nav{id(self)}")
        self.setFixedHeight(self.HEIGHTS[size])
        self.setCursor(Qt.PointingHandCursor)
        self._rest = self.active_fill(t) if active else t.sidebar
        self._fill = HoverFill(
            self, lambda c: (f"QWidget#{self.objectName()} {{ background:{c};"
                             f" border:none;"
                             f" border-radius:{t.radius_sm:g}px; }}"),
            self._rest, self.active_fill(t) if active else t.sidebar_accent)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(pad, 0, pad, 0)
        lay.setSpacing(GAP)
        ink = t.sidebar_accent_foreground if active else t.sidebar_foreground

        if glyph:
            self.glyph = icon(glyph, GLYPH, ink, stroke=NAV_STROKE)
            lay.addWidget(self.glyph)
        else:
            self.glyph = None

        if sub is not None:
            stack = QVBoxLayout()
            stack.setContentsMargins(0, 0, 0, 0)
            stack.setSpacing(0)
            self.title = label(
                text, ty.ui(TEXT, "medium", px=TITLE_PX, wght=MEDIUM_WGHT), ink)
            self.subtitle = label(sub, ty.ui(TEXT_SM, px=text_px(TEXT_SM_REM)),
                                  t.muted_foreground)
            self.title.setFixedHeight(round(TITLE_PX * LEADING_TIGHT))
            self.subtitle.setFixedHeight(
                round(text_px(TEXT_SM_REM) * LEADING_TIGHT))
            stack.addStretch(1)
            stack.addWidget(self.title)
            stack.addWidget(self.subtitle)
            stack.addStretch(1)
            lay.addLayout(stack)
        elif text:
            weight = "semibold" if active else "normal"
            self.title = label(text, ty.ui(TEXT, weight, px=text_px(TEXT_REM)), ink)
            lay.addWidget(self.title)
        else:
            self.title = None

        for lab in (self.title, getattr(self, "subtitle", None)):
            if lab is not None:
                lab.setMinimumWidth(0)
                lab.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self._text_x = pad + (GLYPH + GAP if glyph else 0)
        lay.addStretch(1)
        self.trailing = (icon(trailing, GLYPH, t.sidebar_foreground,
                              stroke=NAV_STROKE) if trailing else None)
        if self.trailing is not None:
            lay.addWidget(self.trailing)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self.rect().contains(e.pos()):
            self.clicked.emit()

    def squeeze(self, width: int) -> None:
        """Clip the words to what the button is currently wide enough for."""
        room = max(0, width - self._text_x - INSET_PAD)
        for lab in (self.title, getattr(self, "subtitle", None)):
            if lab is not None:
                lab.setMaximumWidth(room)
                lab.setVisible(room > 0)
        if self.trailing is not None:
            self.trailing.setVisible(room > 0)

    def set_active(self, on: bool) -> None:
        """Re-point the resting fill, weight and ink at once.

        `set_colours`, not `rest = ...; resync()` -- `resync` re-derives the
        hover from where the POINTER is, so deactivating a row the pointer has
        already left would change nothing and leave it lit.
        """
        if on == self._active:
            return
        self._active = on
        t = self.t
        self._fill.set_colours(self.active_fill(t) if on else t.sidebar,
                               self.active_fill(t) if on else t.sidebar_accent)
        ink = t.sidebar_accent_foreground if on else t.sidebar_foreground
        if self.title is not None:
            self.title.setFont(ty.ui(TEXT, "semibold" if on else "normal", px=text_px(TEXT_REM)))
            self.title.setStyleSheet(f"color:{ink}; background:transparent;")
        if self.glyph is not None:
            self.glyph.colour = ink
            self.glyph.update()
        self.update()

    def is_active(self) -> bool:
        return self._active

    def paintEvent(self, e):
        """The fill (the stylesheet's), then, on the current row, its bar at the left edge."""
        super().paintEvent(e)
        if not (self._active and self._bar):
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(self.t.sidebar_primary))
        h = float(self.height())
        p.drawRoundedRect(QRectF(0.0, h * 0.2, self.BAR_W, h * 0.6), 1.5, 1.5)
        p.end()


NavButton.HEIGHTS = {"default": ROW_H, "sm": SUB_H, "lg": SWITCHER_H}


#: A section's page list opening or shutting, ms (mockup 35's 200 ms).
PAGES_MS = 200
#: shadcn's `SidebarMenuSub`: `mx-3.5 border-l px-2.5 py-0.5 gap-1`.
SUB_INDENT, SUB_PAD, SUB_GAP = rem(0.875), rem(0.625), rem(0.25)


class _Chevron(QWidget):
    """shadcn's collapsible chevron at a section row's right end: `chevron-right`, turned a
    quarter down while the section's pages are listed. A click toggles the list only; it never
    selects the section (the row's own click does)."""

    clicked = Signal()

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.angle = 0.0
        self.setFixedSize(GLYPH, GLYPH)
        self.setCursor(Qt.PointingHandCursor)

    def set_angle(self, a: float) -> None:
        self.angle = float(a)
        self.update()

    def mousePressEvent(self, e):
        e.accept()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self.rect().contains(e.pos()):
            self.clicked.emit()
        e.accept()

    def paintEvent(self, _e):
        from ..theme import icons

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        c = self.rect().center()
        p.translate(c.x() + 0.5, c.y() + 0.5)
        p.rotate(self.angle)
        p.translate(-(c.x() + 0.5), -(c.y() + 0.5))
        icons.paint(p, "chevron-right", QRectF(self.rect()), self.t.sidebar_foreground,
                    stroke=NAV_STROKE)
        p.end()


class _PageList(QWidget):
    """A section's pages under its row, shadcn's `SidebarMenuSub`: indented, a hairline down the
    left, one `NavButton(size="sm")` per page. Opens and shuts by its height over `PAGES_MS`,
    the words clipped rather than squeezed."""

    picked = Signal(str)

    def __init__(self, t, subs, parent=None):
        super().__init__(parent)
        self.t = t
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("background:transparent;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(SUB_INDENT + SUB_PAD, SUB_GAP // 2, 0, SUB_GAP // 2)
        lay.setSpacing(SUB_GAP)
        self.rows: dict[str, NavButton] = {}
        for view, text in subs:
            row = NavButton(t, text, size="sm", bar=False)
            row.clicked.connect(lambda v=view: self.picked.emit(v))
            self.rows[view] = row
            lay.addWidget(row)
        self.k = 1.0
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(PAGES_MS)
        self._anim.setEasingCurve(QEasingCurve.InOutCubic)
        self._anim.valueChanged.connect(self._step)
        self.chevron: _Chevron | None = None

    def natural(self) -> int:
        lay = self.layout()
        m = lay.contentsMargins()
        n = len(self.rows)
        return m.top() + m.bottom() + n * SUB_H + max(0, n - 1) * lay.spacing()

    def set_open(self, on: bool, animate: bool = True) -> None:
        target = 1.0 if on else 0.0
        self._anim.stop()
        if not animate or not self.isVisible():
            self._step(target, force=True)
            return
        blocked = self._anim.blockSignals(True)
        self._anim.setStartValue(self.k)
        self._anim.setEndValue(target)
        self._anim.blockSignals(blocked)
        self._anim.start()

    def _step(self, v, force: bool = False) -> None:
        if not force and self._anim.state() != QVariantAnimation.Running:
            return
        self.k = float(v)
        self.setFixedHeight(round(self.natural() * self.k))
        if self.chevron is not None:
            self.chevron.set_angle(90.0 * self.k)

    def set_current(self, view: str) -> None:
        for v, row in self.rows.items():
            row.set_active(v == view)

    def squeeze(self, width: int) -> None:
        inner = max(0, width - SUB_INDENT - SUB_PAD)
        for row in self.rows.values():
            row.setMaximumWidth(inner)
            row.squeeze(inner)

    def paintEvent(self, e):
        super().paintEvent(e)
        p = QPainter(self)
        w = 1.0 / self.devicePixelRatioF()
        m = self.layout().contentsMargins()
        p.fillRect(QRectF(SUB_INDENT, m.top(), w, max(0.0, self.height() - m.top() - m.bottom())),
                   QColor(self.t.sidebar_border))
        p.end()


def _lerp(a: float, b: float, k: float) -> float:
    return a + (b - a) * k


class _GroupLabel(QWidget):
    """shadcn's `SidebarGroupLabel` (`text-xs font-medium`, 70 % ink, `h-7`: `LABEL_H`) that becomes a `SidebarSeparator` on
    the icon rail: through the collapse the words fade and the height falls to ``rail_h``, where a hairline
    between the groups fades in. ``rail_h`` 0 (the first group, and a one-group rail) folds it away as the one
    label did before the groups. Painted, never a QLabel: a one-line label is a minimum WIDTH, and the rail
    shrinks to 48 px."""

    def __init__(self, t, text: str, *, rail_h: int = 0, parent=None):
        super().__init__(parent)
        self.t = t
        self._text = text
        self.rail_h = int(rail_h)
        self.k = 0.0
        self._font = ty.ui(TEXT_SM, "medium", px=text_px(TEXT_SM_REM), wght=MEDIUM_WGHT)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.setFixedHeight(LABEL_H)

    def text(self) -> str:
        return self._text

    def sizeHint(self) -> QSize:
        return QSize(0, self.height())

    def minimumSizeHint(self) -> QSize:
        return QSize(0, self.height())

    def set_k(self, k: float) -> None:
        """Where the collapse is, 0 (wide) to 1 (the icon rail)."""
        self.k = float(k)
        h = round(_lerp(LABEL_H, self.rail_h, self.k))
        if h != self.height():
            self.setFixedHeight(h)
        self.update()

    def paintEvent(self, _e):
        t = self.t
        p = QPainter(self)
        # the words go over the first half of the slide, the hairline comes over the second: never one through
        # the other
        words, line = min(1.0, 2.0 * self.k), max(0.0, 2.0 * self.k - 1.0)
        if words < 1.0 and self.height() > 0:
            p.setFont(self._font)
            p.setPen(QColor(_fade(mix(t.sidebar_foreground, t.sidebar, LABEL_ALPHA), t.sidebar, words)))
            room = max(0, self.width() - INSET_PAD)
            text = QFontMetrics(self._font).elidedText(self._text, Qt.ElideRight, room)
            p.drawText(QRectF(INSET_PAD, 0.0, float(room), float(self.height())),
                       int(Qt.AlignLeft | Qt.AlignVCenter), text)
        if self.rail_h and line > 0.0:
            # level between the groups: the gap above a label is GROUP_GAP, below it ROW_GAP
            w = 1.0 / self.devicePixelRatioF()
            y = max(0.0, (self.height() - (GROUP_GAP - ROW_GAP)) / 2.0)
            p.fillRect(QRectF(0.0, y, float(self.width()), w),
                       QColor(_fade(t.sidebar, t.sidebar_border, line)))
        p.end()


class _Face(Surface):
    """A control that paints its own face: `HoverFill` fades the colour (its stylesheet stays transparent) and
    `paintEvent` draws the shape, the glyph and the words at whatever size the collapse has reached."""

    clicked = Signal()

    def __init__(self, t, text: str, glyph: str, *, rest: str, hover: str, radius: float,
                 shadowed: bool = False, shadow=None, parent=None):
        super().__init__(t, radius=radius, shadowed=shadowed, shadow=shadow, parent=parent)
        self.t = t
        self.text = text
        self.glyph = glyph
        self.k = 0.0
        self._active = False
        self._face = rest
        self.setObjectName(f"face{id(self)}")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setCursor(Qt.PointingHandCursor)
        self._fill = HoverFill(self, self._style, rest, hover)

    def _style(self, colour) -> str:
        self._face = colour if isinstance(colour, str) else QColor(colour).name()
        self.update()
        return f"QWidget#{self.objectName()} {{ background:transparent; border:none; }}"

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self.rect().contains(e.pos()):
            self.clicked.emit()

    def is_active(self) -> bool:
        return self._active

    def set_k(self, k: float) -> None:
        self.k = float(k)
        self.update()


class DataTile(_Face):
    """QC or Images & tracks: a tile, its glyph over its name -- shadcn's `outline` button (`border
    bg-background shadow-xs hover:bg-accent`, dark `bg-input/30 border-input hover:bg-input/50`). The current one
    takes the rows' active tint, a border of the primary ink and a semibold name. On the icon rail it is a 32 px
    cell with the glyph alone (`_TileLayout` moves it there; the words fade over the first half)."""

    def __init__(self, t, text: str, glyph: str, parent=None):
        dark = t.color_scheme == "dark"
        rest = mix(t.input, t.sidebar, 0.30) if dark else t.background
        hover = mix(t.input, t.sidebar, 0.50) if dark else t.sidebar_accent
        super().__init__(t, text, glyph, rest=rest, hover=hover, radius=t.radius_lg,
                         shadowed=True, shadow=t.shadow_xs, parent=parent)
        self._rest, self._hover = rest, hover
        self._font = ty.ui("control", "medium", px=TILE_TEXT_PX)
        self._font_on = ty.ui("control", "semibold", px=TILE_TEXT_PX)

    def sizeHint(self) -> QSize:
        return QSize((FULL_BTN - GAP) // 2, TILE_H)

    def minimumSizeHint(self) -> QSize:
        return QSize(0, 0)

    def elided(self) -> bool:
        """Whether the name is cut (the groups' scroll bar can narrow a tile); its tooltip then gives it whole."""
        font = self._font_on if self._active else self._font
        return QFontMetrics(font).horizontalAdvance(self.text) > self.width() - 2 * TILE_PAD

    def set_active(self, on: bool) -> None:
        if on == self._active:
            return
        self._active = on
        fill = NavButton.active_fill(self.t)
        self._fill.set_colours(fill if on else self._rest, fill if on else self._hover)
        self.update()

    def paintEvent(self, e):
        super().paintEvent(e)
        from ..theme import icons

        t, k = self.t, self.k
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        self.radius = _lerp(t.radius_lg, t.radius_md, k)
        edge = t.sidebar_primary if self._active else (t.input if t.color_scheme == "dark" else t.sidebar_border)
        p.setPen(QPen(QColor(edge), 1.0))
        p.setBrush(QColor(self._face))
        p.drawRoundedRect(r, self.radius, self.radius)
        ink = t.sidebar_accent_foreground if self._active else t.sidebar_foreground
        size = _lerp(TILE_GLYPH, GLYPH, k)
        cy = _lerp(TILE_TOP + TILE_GLYPH / 2.0, self.height() / 2.0, k)
        cx = self.width() / 2.0
        icons.paint(p, self.glyph, QRectF(cx - size / 2.0, cy - size / 2.0, size, size), ink, stroke=NAV_STROKE)
        alpha = max(0.0, 1.0 - 2.0 * k)
        if alpha > 0.0:
            font = self._font_on if self._active else self._font
            fm = QFontMetrics(font)
            room = max(0, self.width() - 2 * TILE_PAD)
            text = fm.elidedText(self.text, Qt.ElideRight, room)
            colour = QColor(ink)
            colour.setAlphaF(alpha)
            p.setFont(font)
            p.setPen(colour)
            top = cy + size / 2.0 + rem(0.25)
            p.drawText(QRectF(TILE_PAD, top, float(room), float(fm.height())),
                       int(Qt.AlignHCenter | Qt.AlignTop), text)
        p.end()


class _TileLayout(QLayout):
    """The Data tiles' layout: side by side on the wide rail, one under the other on the icon rail, and every
    frame of the collapse between (``k``). A layout's `setGeometry`, never a `resizeEvent` (CLAUDE.md's trap)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items: list = []
        self.k = 0.0
        self.room = 0

    def addItem(self, item) -> None:  # noqa: N802 - Qt's name
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, i):  # noqa: N802 - Qt's name
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):  # noqa: N802 - Qt's name
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def height_at(self, k: float) -> int:
        n = max(1, len(self._items))
        return round(_lerp(TILE_H, n * ROW_H + (n - 1) * ROW_GAP, k)) + self.room

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt's name
        return QSize(0, self.height_at(self.k))

    def minimumSize(self) -> QSize:  # noqa: N802 - Qt's name
        return QSize(0, self.height_at(self.k))

    def setGeometry(self, rect) -> None:  # noqa: N802 - Qt's name
        super().setGeometry(rect)
        n = len(self._items)
        if not n:
            return
        k, width = self.k, rect.width()
        wide_w = (width - (n - 1) * GAP) / n
        # every edge on a straight line, so no cell ever stands past the row's height (`height_at`)
        for i, item in enumerate(self._items):
            x = _lerp(i * (wide_w + GAP), 0.0, k)
            y = _lerp(0.0, i * (ROW_H + ROW_GAP), k)
            w = _lerp(wide_w, float(width), k)
            h = _lerp(TILE_H, ROW_H, k)
            item.setGeometry(QRect(rect.x() + round(x), rect.y() + round(y), round(w), round(h)))

    def set_k(self, k: float) -> None:
        self.k = float(k)
        self.invalidate()


class _TileRow(ShadowHost):
    """The Data group's tiles, holding the `shadow-xs` they cast (a shadow is painted by an ancestor)."""

    def __init__(self, t, parent=None):
        super().__init__(t, parent)
        self.setStyleSheet("background:transparent;")
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self._lay = _TileLayout(self)
        self._lay.room = shadow_room(t, t.shadow_xs)
        self.tiles: list = []

    def add(self, tile: DataTile) -> None:
        self._lay.addWidget(tile)
        self.tiles.append(tile)

    def set_k(self, k: float) -> None:
        self._lay.set_k(k)
        for tile in self.tiles:
            tile.set_k(k)
        self.updateGeometry()


class FeatureButton(_Face):
    """The Composer, pinned at the rail's foot: shadcn's `default` button (`h-9 bg-primary text-primary-foreground
    hover:bg-primary/90`, no shadow), the one filled control in the rail. On the Composer page a ring INSIDE its
    edge says so (an outer ring would stand past the rows' edge). On the icon rail it is a 32 px filled cell with
    the glyph; the words clip as the rows' do."""

    #: shadcn's `hover:bg-primary/90`.
    HOVER = 0.90
    RING_INSET = 2.5

    def __init__(self, t, text: str, glyph: str, parent=None):
        super().__init__(t, text, glyph, rest=t.sidebar_primary,
                         hover=mix(t.sidebar_primary, t.sidebar, FeatureButton.HOVER), radius=t.radius_md,
                         parent=parent)
        self._font = ty.ui(TEXT, "medium", px=text_px(TEXT_REM))
        self._font_on = ty.ui(TEXT, "semibold", px=text_px(TEXT_REM))
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.setFixedHeight(FEATURE_H)

    def sizeHint(self) -> QSize:
        return QSize(FULL_BTN, self.height())

    def minimumSizeHint(self) -> QSize:
        return QSize(0, self.height())

    def set_active(self, on: bool) -> None:
        if on == self._active:
            return
        self._active = on
        self.update()

    def rail(self, k: float, width: int) -> None:
        """One frame of the collapse: the cell's width and height, and where the glyph sits."""
        self.setMaximumWidth(width)
        h = round(_lerp(FEATURE_H, ROW_H, k))
        if h != self.height():
            self.setFixedHeight(h)
        self.set_k(k)

    def paintEvent(self, e):
        super().paintEvent(e)
        from ..theme import icons

        t, k = self.t, self.k
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        r = QRectF(self.rect())
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(self._face))
        p.drawRoundedRect(r, self.radius, self.radius)
        ink = t.sidebar_primary_foreground
        if self._active:
            ring = QColor(ink)
            ring.setAlphaF(0.55)
            p.setPen(QPen(ring, 1.5))
            p.setBrush(Qt.NoBrush)
            inset = self.RING_INSET
            p.drawRoundedRect(r.adjusted(inset, inset, -inset, -inset), self.radius - inset + 0.5,
                              self.radius - inset + 0.5)
        x = _lerp(FEATURE_PAD, (ICON_BTN - GLYPH) / 2.0, k)
        y = (self.height() - GLYPH) / 2.0
        icons.paint(p, self.glyph, QRectF(x, y, float(GLYPH), float(GLYPH)), ink, stroke=NAV_STROKE)
        left = x + GLYPH + GAP
        room = self.width() - left - FEATURE_PAD
        if room > 0:
            font = self._font_on if self._active else self._font
            p.setFont(font)
            p.setPen(QColor(ink))
            p.setClipRect(QRectF(left, 0.0, float(room), float(self.height())))
            p.drawText(QRectF(left, 0.0, float(FULL_BTN), float(self.height())),
                       int(Qt.AlignLeft | Qt.AlignVCenter), self.text)
        p.end()


class _SectionItem(QWidget):
    """One section of a rows group: its row and, under it, what belongs to it -- its added pages ("SQD 2", the
    SUBPAGES: `subpage_rows.SubpageList`) and its pages' list (`set_pages`), in that order, in the same column."""

    def __init__(self, row: NavButton, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background:transparent;")
        self.row = row
        #: The added pages' list (`put_subpages`), None for a section that takes none.
        self.subs = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(row)

    def put_under(self, widget: QWidget) -> None:
        """``widget`` (the pages' list) under the row, and under the added pages if there are any (into the layout
        BEFORE any visibility is set)."""
        self.layout().insertWidget(2 if self.subs is not None else 1, widget)

    def put_subpages(self, widget: QWidget) -> None:
        """The added pages' list directly under the row."""
        self.subs = widget
        self.layout().insertWidget(1, widget)


class _BodyScroll(QScrollArea):
    """The groups between the header and the foot, shadcn's `SidebarContent` (`min-h-0 flex-1 overflow-auto`):
    as tall as every group where the window has the room (its hint), three rows where it has not (its minimum),
    so a long folder name, the HMM pages, the Running list and a 720 px screen never pin the window's minimum
    height. Takes no keyboard focus: a click in the rail leaves the keys with the page (the HMM Test page's
    W / S, the arrows)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFocusPolicy(Qt.NoFocus)
        self.setStyleSheet("QScrollArea { background:transparent; border:none; }")
        self.viewport().setAutoFillBackground(False)

    def setWidget(self, widget) -> None:  # noqa: N802 - Qt's name
        super().setWidget(widget)
        widget.installEventFilter(self)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt's name
        if watched is self.widget() and event.type() == QEvent.LayoutRequest:
            self.updateGeometry()
        return super().eventFilter(watched, event)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt's name
        inner = self.widget()
        if inner is None:
            return super().sizeHint()
        return QSize(super().sizeHint().width(), inner.sizeHint().height())

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt's name
        inner = self.widget()
        full = inner.sizeHint().height() if inner is not None else 0
        return QSize(0, min(full, 3 * ROW_H + 2 * ROW_GAP))


class _RailColumn(QLayout):
    """The rail's column, top to bottom: the header, the groups (`_BodyScroll`), the Running list, the Figures
    group. The header and the Figures group keep their height. Where the window is short the room goes to the
    GROUPS first (every section in sight), then to the Running list (every row), and only then do the groups
    scroll; the Running list never drops below one row (its minimum). Room to spare stands between the groups and
    the Running list, which sits on the Figures group at the foot. A QVBoxLayout cannot say this: short of every
    size hint it takes the shortfall from each item equally, so the navigation scrolled while a job ran.
    Laid out in `setGeometry` (CLAUDE.md's trap: never in a `resizeEvent`)."""

    #: The roles `add` takes: kept at its hint, the scrolling groups, the Running list.
    ROLES = ("fixed", "body", "jobs")

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items: list = []
        self._roles: list = []

    def add(self, widget: QWidget, role: str) -> None:
        if role not in self.ROLES:
            raise ValueError(f"unknown rail role {role!r}")
        self._roles.append(role)
        self.addWidget(widget)

    def addItem(self, item) -> None:  # noqa: N802 - Qt's name
        self._items.append(item)
        if len(self._roles) < len(self._items):
            self._roles.append("fixed")

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, i):  # noqa: N802 - Qt's name
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):  # noqa: N802 - Qt's name
        if not 0 <= i < len(self._items):
            return None
        self._roles.pop(i)
        return self._items.pop(i)

    @staticmethod
    def _heights(item) -> tuple:
        """``(minimum, hint)`` of an item, nothing for a hidden one."""
        if item.isEmpty():
            return 0, 0
        lo = item.minimumSize().height()
        return lo, max(lo, item.sizeHint().height())

    def _extra(self) -> int:
        m = self.contentsMargins()
        return m.top() + m.bottom()

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt's name
        return QSize(SIDEBAR_W, self._extra() + sum(self._heights(item)[1] for item in self._items))

    def minimumSize(self) -> QSize:  # noqa: N802 - Qt's name
        total = 0
        for item, role in zip(self._items, self._roles):
            lo, hint = self._heights(item)
            total += hint if role == "fixed" else lo
        return QSize(0, self._extra() + total)

    def plan(self, height: int) -> list:
        """Every item's height in a column ``height`` px tall (margins taken off), in order."""
        sizes = [self._heights(item) for item in self._items]
        room = height - sum(hint for (_lo, hint), role in zip(sizes, self._roles) if role == "fixed")
        body = [i for i, role in enumerate(self._roles) if role == "body"]
        jobs = [i for i, role in enumerate(self._roles) if role == "jobs"]
        body_lo = sum(sizes[i][0] for i in body)
        body_full = sum(sizes[i][1] for i in body)
        jobs_lo = sum(sizes[i][0] for i in jobs)
        jobs_full = sum(sizes[i][1] for i in jobs)
        jobs_h = max(jobs_lo, min(jobs_full, room - body_full))
        body_h = max(body_lo, room - jobs_h)
        out = []
        for i, role in enumerate(self._roles):
            if role == "fixed":
                out.append(sizes[i][1])
            elif role == "body":
                out.append(body_h if i == body[0] else 0)
            else:
                out.append(jobs_h if i == jobs[0] else 0)
        return out

    def setGeometry(self, rect) -> None:  # noqa: N802 - Qt's name
        super().setGeometry(rect)
        r = self.contentsRect()
        heights = self.plan(r.height())
        body_at = next((i for i, role in enumerate(self._roles) if role == "body"), len(self._items))
        y = r.top()
        for i in range(0, min(body_at + 1, len(self._items))):       # the header and the groups, from the top
            self._items[i].setGeometry(QRect(r.left(), y, r.width(), heights[i]))
            y += heights[i]
        y = r.bottom() + 1
        for i in range(len(self._items) - 1, body_at, -1):          # the Running list and the foot, from the bottom
            y -= heights[i]
            self._items[i].setGeometry(QRect(r.left(), y, r.width(), heights[i]))


class _Tile(Surface):
    """`size-8 rounded-lg bg-sidebar-primary` -- the one saturated mark in the
    rail, and the only thing that survives the collapse to a 48 px strip."""

    def __init__(self, t, glyph: str, parent=None):
        super().__init__(t, radius=t.radius_lg, shadowed=False, parent=parent)
        self.t, self._glyph = t, glyph
        self.setFixedSize(TILE, TILE)

    def paintEvent(self, e):
        super().paintEvent(e)
        from ..theme import icons

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        # THE LOGO IN THE CORNER (13 Nov 2026, asked: "add the logo/icon. launcher, top corner"): the app icon,
        # which is itself a rounded tile; the glyph on the rail's colour where the icon did not ship
        from ...appicon import qicon

        logo = qicon()
        if logo is not None:
            logo.paint(p, self.rect())
            p.end()
            return
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(self.t.sidebar_primary))
        p.drawRoundedRect(self.rect(), self.t.radius_lg, self.t.radius_lg)
        inset = (TILE - GLYPH) / 2.0
        icons.paint(p, self._glyph, QRectF(inset, inset, float(GLYPH), float(GLYPH)),
                    self.t.sidebar_primary_foreground, stroke=NAV_STROKE)
        p.end()


def wrap_lines(text: str, fm, room: int) -> list:
    """``text`` broken into lines no wider than ``room`` pixels in ``fm``: at spaces, and inside a word that is
    wider than a line on its own. Never drops a character; one line for an empty text."""
    if room <= 0:
        return [text]
    lines, current = [], ""
    for word in text.split(" "):
        trial = f"{current} {word}" if current else word
        if fm.horizontalAdvance(trial) <= room:
            current = trial
            continue
        if current:
            lines.append(current)
        while len(word) > 1 and fm.horizontalAdvance(word) > room:
            cut = len(word) - 1
            while cut > 1 and fm.horizontalAdvance(word[:cut]) > room:
                cut -= 1
            lines.append(word[:cut])
            word = word[cut:]
        current = word
    lines.append(current)
    return lines


class WorkspaceHeader(NavButton):
    """The header row: which project is open, and the one way back.

    Unlike the prototype's `WorkspaceSwitcher`, this opens no menu of OTHER
    workspaces -- there is only ever one loaded project, and reaching another
    means going back through onboarding. A click emits `opened`; the window
    that owns this sidebar opens its menu: Open another folder, Back to menu
    (asked 15 Oct 2026: "we need a go back button or something to open a new
    folder"). The chevron says it opens something.
    """

    opened = Signal()

    def __init__(self, t, title: str, subtitle: str, *, glyph: str = "folder",
                 parent=None):
        super().__init__(t, title, size="lg", sub=subtitle, trailing="chevrons-up-down", parent=parent)
        self._tile = _Tile(t, glyph)
        self.layout().insertWidget(0, self._tile)
        self.clicked.connect(self.opened.emit)
        self._full_title = title
        self.setToolTip(title)
        self._fit_title(FULL_BTN, INSET_PAD)

    # THE WHOLE FOLDER NAME, NEVER CUT (asked 20 Oct 2026: "name of folder, show all, dont cut"). The title was one
    # line of fixed height that could shrink to nothing, and `NavButton.squeeze` measured its room without the tile
    # or the chevron, so a long name was clipped at the rail's edge with no ellipsis ("B subtilis 3610 protein C mNG"
    # of a longer folder name). It is now broken onto as many lines as it needs -- at spaces, and inside a word
    # longer than a line, since folder names often have none -- and the expanded header grows to hold them
    # (`expanded_height`, read by `Sidebar._step`). Broken by hand rather than word-wrapped: a wrapped QLabel makes
    # every layout above it height-for-width (`FitLabel`'s measurement), and Qt's wrap cannot break a word.

    def _room(self, width: int, pad: int) -> int:
        """The width left for the words: the row less its padding, the tile and the chevron."""
        trailing = GAP + GLYPH if self.trailing is not None else 0
        return max(0, width - 2 * pad - TILE - GAP - trailing)

    def _lines(self, room: int) -> list:
        return wrap_lines(self._full_title, QFontMetrics(self.title.font()), room)

    def _fit_title(self, width: int, pad: int) -> None:
        room = self._room(width, pad)
        lines = self._lines(room)
        self.title.setText("\n".join(lines))
        self.title.setFixedHeight(round(TITLE_PX * LEADING_TIGHT)
                                  + (len(lines) - 1) * QFontMetrics(self.title.font()).lineSpacing())
        self.setFixedHeight(self.expanded_height())

    def expanded_height(self) -> int:
        """The header's height on the expanded rail: one line's `SWITCHER_H`, plus a line for each further line
        the name needs."""
        extra = len(self._lines(self._room(FULL_BTN, INSET_PAD))) - 1
        return SWITCHER_H + extra * QFontMetrics(self.title.font()).lineSpacing()

    def squeeze(self, width: int) -> None:
        """While the rail slides the words keep their full-width lines and are only clipped; collapsed, hidden."""
        room = self._room(width, INSET_PAD)
        for lab in (self.title, self.subtitle):
            lab.setMaximumWidth(max(0, room))
            lab.setVisible(room > 0)

    def set_project(self, title: str, subtitle: str) -> None:
        self._full_title = title
        self.setToolTip(title)
        self.subtitle.setText(subtitle)
        self._fit_title(FULL_BTN, INSET_PAD)


class SidebarTrigger(Surface):
    """`SidebarTrigger`: a `ghost` icon Button with `panel-left`.

    Lives in the MAIN AREA, not the rail -- half its job is bringing back a
    rail that is not on screen to be clicked.
    """

    clicked = Signal()

    def __init__(self, t, parent=None):
        super().__init__(t, radius=t.radius_sm, shadowed=False, parent=parent)
        self.t = t
        self.setObjectName(f"trig{id(self)}")
        self.setFixedSize(TRIGGER, TRIGGER)
        self.setCursor(Qt.PointingHandCursor)
        HoverFill(self, lambda c: (f"QWidget#{self.objectName()} "
                                   f"{{ background:{c}; border:none;"
                                   f" border-radius:{t.radius_sm:g}px; }}"),
                  t.background, t.accent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(icon("panel-left", GLYPH, t.foreground, stroke=NAV_STROKE),
                      alignment=Qt.AlignCenter)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self.rect().contains(e.pos()):
            self.clicked.emit()


class Breadcrumb(QWidget):
    """shadcn's `Breadcrumb`, read off `ui/breadcrumb.tsx`::

        BreadcrumbList       flex items-center gap-1.5 text-sm
                             text-muted-foreground sm:gap-2.5
        BreadcrumbItem       inline-flex items-center gap-1.5
        BreadcrumbSeparator  [&>svg]:size-3.5   (ChevronRight by default)
        BreadcrumbPage       font-normal text-foreground

    So the trail is MUTED and only the last item takes the page's ink -- that
    contrast is the whole component, and it is what says which of the two
    names you are actually looking at. Ported from `prototype/shadcn/
    workspace.py` (git history, commit 09da35d) in place of the plain,
    uniformly-coloured "SQD / Populations" label this window opened with,
    which had neither the muted/foreground split nor a real separator glyph
    (a literal "/" rather than shadcn's `ChevronRight`).

    **`BreadcrumbLink`'s `hover:text-foreground` used to not be ported, and
    that was a decision worth revisiting rather than a permanent rule.** The
    original reasoning stood as long as it was true: "with one section there
    is nowhere for an earlier crumb to lead yet", and an inert crumb that
    lights up under the pointer is the lie that distinction exists to
    prevent. That changed the moment a THIRD, dynamic crumb existed (a
    condition's own detail page, under `SQD > Populations > Glucose`) --
    now the two crumbs BEHIND the page one really do lead somewhere, so they
    get the registry's real `BreadcrumbLink` treatment (`crumb_clicked`,
    below), while the LAST crumb stays exactly as inert as before
    (`BreadcrumbPage`, never a link -- there is nothing to click on the page
    you are already looking at).

    **THE LAST CRUMB CAN ALSO CARRY A DROPDOWN, per shadcn's own `Breadcrumb`
    docs page (`BreadcrumbItem` + `DropdownMenu`).** Asked for directly: a
    condition's detail page names itself as the last crumb ("Glucose"), and
    switching to another condition used to mean a crumb click back to
    Populations, THEN a second click on a different bubble. `set_trail`'s
    `dropdown_options` turns that last crumb into a `_CrumbDropdown` -- the
    same inert foreground text, plus a small chevron that opens every OTHER
    option in a `Panel`, exactly the popover `toolbar.SelectButton` already
    opens for a value picker. Still the "page you are already looking at"
    (`BreadcrumbPage`'s text stays put), so it is not a `_CrumbLink`: this is
    a MENU for leaving, not a link.
    """

    #: `sm:gap-2.5` -- this page is never narrow enough for the `gap-1.5` base.
    GAP = rem(0.625)

    #: A non-last crumb was clicked -- its INDEX into the trail most recently
    #: passed to `set_trail`. The caller decides what that index means (for
    #: this workspace, today: any index before the last one means "leave the
    #: detail page"); `Breadcrumb` itself has no notion of pages or sections.
    crumb_clicked = Signal(int)

    #: The last crumb's dropdown offered a DIFFERENT option, and it was
    #: picked. `Breadcrumb` has no notion of what a "condition" is -- the
    #: caller decides what emitting this means, same as `crumb_clicked`.
    dropdown_picked = Signal(str)

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.setStyleSheet("background:transparent;")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(self.GAP)
        self._lay = lay
        self._parts: list = []

    def set_trail(self, names, *, dropdown_options=None) -> None:
        """Replace the trail. Rebuilt rather than relabelled: the LAST crumb
        differs from the others in colour AND interactivity, so keeping
        widgets around would mean re-deciding which one is the page on every
        change.

        `dropdown_options`, when given, is every choice the LAST crumb's
        dropdown should offer -- typically every loaded condition, itself
        included; that one is filtered out inside `_CrumbDropdown` (nothing
        to switch TO by picking where you already are). `None` (the default,
        every OTHER call site) keeps the last crumb a plain, inert
        `BreadcrumbPage` exactly as before.
        """
        while self._lay.count():
            item = self._lay.takeAt(0)
            w = item.widget()
            if w is not None:
                discard(w)
        self._parts = []
        t = self.t
        names = list(names)
        for i, name in enumerate(names):
            last = i == len(names) - 1
            if i:
                # `[&>svg]:size-3.5`, at the list's muted ink.
                self._lay.addWidget(icon("chevron-right", rem(0.875),
                                         t.muted_foreground))
            # Checked here, not just left to `_CrumbDropdown` itself:
            # filtering OUT the current name (a single-condition project's
            # only option) can leave nothing to switch to, and a chevron
            # that opens onto an empty menu is worse than no chevron -- it
            # invites a click with nowhere for it to lead.
            others = [o for o in (dropdown_options or ()) if o != name]
            if last and others:
                crumb = _CrumbDropdown(t, name, ty.ui("sm", px=text_px(TEXT_REM)),
                                       others)
                crumb.picked.connect(self.dropdown_picked.emit)
            elif last:
                # `BreadcrumbPage`: inert, full foreground ink -- see the
                # class docstring for why this one never becomes a link.
                crumb = label(name, ty.ui("sm", px=text_px(TEXT_REM)),
                              t.foreground)
            else:
                # `BreadcrumbLink`: clickable, muted until hovered.
                crumb = _CrumbLink(t, name, ty.ui("sm", px=text_px(TEXT_REM)))
                crumb.clicked.connect(
                    lambda _checked=False, idx=i: self.crumb_clicked.emit(idx))
            self._lay.addWidget(crumb)
            self._parts.append(crumb)

    def trail(self) -> list:
        """What is on screen, for anything that needs to check it."""
        return [w.text() for w in self._parts]


class _CrumbLink(QLabel):
    """shadcn's `BreadcrumbLink`: clickable, `hover:text-foreground` --
    a TEXT-COLOUR-only hover, no fill. `HoverFill` exists for a filled
    background hover this component's own registry does not ask for, so
    this repaints its own stylesheet directly instead of reaching for it.

    Built directly rather than through `widgets.primitives.label()`: that
    helper sets `WA_TransparentForMouseEvents` on every label it makes
    (correct for the non-interactive text runs it exists for), which would
    make a click here fall straight through to whatever is behind it.
    """

    clicked = Signal()

    def __init__(self, t, text: str, font, parent=None):
        super().__init__(text, parent)
        self.t = t
        self.setFont(font)
        self._muted()
        self.setCursor(Qt.PointingHandCursor)

    def _muted(self) -> None:
        self.setStyleSheet(f"color:{self.t.muted_foreground}; background:transparent;")

    def _lit(self) -> None:
        self.setStyleSheet(f"color:{self.t.foreground}; background:transparent;")

    def enterEvent(self, e):
        super().enterEvent(e)
        self._lit()

    def leaveEvent(self, e):
        super().leaveEvent(e)
        self._muted()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self.rect().contains(e.position().toPoint()):
            self.clicked.emit()
        super().mouseReleaseEvent(e)


class _CrumbDropdown(QWidget):
    """The last crumb, WITH a dropdown -- shadcn's own `Breadcrumb` docs page
    composes exactly this out of `BreadcrumbItem` + `DropdownMenu`. Text
    stays `BreadcrumbPage`'s full foreground ink (this IS the page you are
    on); the chevron beside it opens a `Panel` listing every OTHER option,
    the same popover machinery `toolbar.SelectButton` already opens for a
    value picker (`menu_row`, `Panel.popup`).

    A container around a `label()` + an `icon()`, not a `QLabel` subclass
    like `_CrumbLink`: both of those helpers mark their widget `WA_
    TransparentForMouseEvents` (`label()` always; `Icon` unconditionally),
    so a click on either passes straight through to THIS widget's own
    `mouseReleaseEvent` -- the whole row is one click target, text and
    chevron alike, matching the reference screenshot where the chevron sits
    flush against the word with no gap a click could miss.
    """

    #: Another option was picked. `""` never reaches here -- see `options`.
    picked = Signal(str)

    def __init__(self, t, name: str, font, options, parent=None):
        super().__init__(parent)
        self.t = t
        self._name = name
        #: Every OTHER loaded condition -- picking the one already showing
        #: would be a no-op dressed as a menu item, so it is dropped here
        #: rather than trusted to callers to exclude.
        self._options = tuple(o for o in options if o != name)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(rem(0.25))
        lay.addWidget(label(name, font, t.foreground))
        lay.addWidget(icon("chevron-down", rem(0.875), t.muted_foreground, stroke=2.0))
        if self._options:
            self.setCursor(Qt.PointingHandCursor)

    def text(self) -> str:
        """Matches `QLabel.text()` -- `Breadcrumb.trail()` reads either
        crumb type through this one name."""
        return self._name

    def mouseReleaseEvent(self, e):
        if e.button() != Qt.LeftButton or not self._options:
            super().mouseReleaseEvent(e)
            return
        if retriggered(self):
            return
        t = self.t
        menu = Panel(t, self.window(), root=ROOT_PX)
        menu.set_trigger(self)
        for name in self._options:
            row = menu_row(t, menu, name, root=ROOT_PX,
                          on_click=lambda n=name: self.picked.emit(n))
            # `_Row` carries a flat `12.5rem` (200 px) floor -- right for a
            # full DropdownMenu with icons/badges/submenus, wide beside a
            # condition name like "Malate". Reported directly ("the width
            # of the picker is too long"). Overriding it here, per row,
            # rather than in `_Row` itself: that floor is shared by every
            # OTHER menu in the app and none of them asked to shrink.
            row.setMinimumWidth(0)
        menu.popup(self.mapToGlobal(QPoint(0, self.height() + 4)))
        super().mouseReleaseEvent(e)


class Sidebar(QWidget):
    """The rail itself, and the one animation it owns: the collapse tween.

    Collapsing is a WIDTH tween between `SIDEBAR_W` and `RAIL_W` rather than a
    hide -- at `RAIL_W` the section glyphs are still there and still
    clickable, so navigation survives the collapse instead of being replaced
    by a gap.
    """

    #: See the prototype's own measurement of this number (`workspace.py`,
    #: `Sidebar.COLLAPSE_MS`): 150 ms is the snappy end, chosen knowing the
    #: page repaints at ~32 ms against a 16.7 ms budget while sliding.
    COLLAPSE_MS = slide_ms(SIDEBAR_W - RAIL_W)

    opened = Signal()
    collapsed_changed = Signal(bool)
    slide_started = Signal()
    slide_finished = Signal()
    #: The active subsection changed -- a future page/toolbar follows it.
    selected = Signal(str)
    #: Where the header row's middle currently sits, so a page's own
    #: `SidebarTrigger` can stay level with it through the collapse.
    header_centre = Signal(float)
    #: The SUBPAGES (`smtracker.subpages`): the "+" of a section, and what the user did to an added page's row. The
    #: sidebar changes nothing itself; the window owns the model and calls `set_subpages` with the result.
    subpage_add_requested = Signal(str)
    subpage_renamed = Signal(str, int, str)
    subpage_deleted = Signal(str, int)
    subpage_moved = Signal(str, int, int)
    subpage_dropped = Signal(str, int, int)

    def __init__(self, t, *, project_title: str = "SMTracker",
                 project_subtitle: str = "", sections: tuple = SECTIONS,
                 group_label: str = "Analysis", groups: tuple | None = None, parent=None):
        """``sections`` and ``groups`` default to the ANALYSIS workspace's (`SECTIONS` in `GROUPS`), so every
        existing caller is unchanged; the preparation and outlines windows pass their own steps, which are ONE
        plain list of rows under ``group_label`` unless ``groups`` says otherwise.
        """
        super().__init__(parent)
        self.t = t
        self.collapsed = False
        #: This rail's own navigation tree, `SECTIONS`-shaped.
        self._spec = tuple(sections)
        self._subs = {k: subs for k, _t, _g, subs in self._spec}
        if groups is None:
            groups = (GROUPS if self._spec == SECTIONS
                      else (("", group_label, "rows", tuple(k for k, *_ in self._spec)),))
        #: The rail's groups, `GROUPS`-shaped and checked against the sections.
        self._group_spec = _checked_groups(groups, self._spec)
        self.current = self._spec[0][3][0][0]
        self.setObjectName("sidebar")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(
            f"QWidget#sidebar {{ background:{t.sidebar}; border:none; }}"
            f"QWidget#sbhead, QWidget#sbbody, QWidget#sbfoot {{ background:{t.sidebar};"
            f" border:none; }}")
        self.setFixedWidth(SIDEBAR_W + EDGE)

        lay = _RailColumn(self)
        lay.setContentsMargins(0, 0, EDGE, 0)
        lay.setSpacing(0)

        head = QWidget()
        head.setObjectName("sbhead")
        hl = QVBoxLayout(head)
        hl.setContentsMargins(INSET_PAD, INSET_PAD, INSET_PAD, INSET_PAD)
        hl.setSpacing(GAP)
        self.header = WorkspaceHeader(t, project_title, project_subtitle)
        self.header.opened.connect(self.opened.emit)
        hl.addWidget(self.header)
        lay.add(head, "fixed")

        body = QWidget()
        body.setObjectName("sbbody")
        bl = QVBoxLayout(body)
        bl.setContentsMargins(INSET_PAD, INSET_PAD, INSET_PAD, 0)
        bl.setSpacing(GROUP_GAP)
        #: `add_group`'s headings (the preparation window's conditions) are plain labels.
        self._label_css = (f"padding-left:{INSET_PAD}px;"
                           f" background:transparent; color:%s;")

        self._sections: dict[str, QWidget] = {}
        self._last: dict[str, str] = {
            k: subs[0][0] for k, _t, _g, subs in self._spec}
        self._last[self.current.split(".")[0]] = self.current
        #: Section key -> its control: a `NavButton` row, a `DataTile` or the `FeatureButton`.
        self._items: dict[str, QWidget] = {}
        #: Section key -> the `_SectionItem` holding a row and what is listed under it.
        self._section_items: dict[str, _SectionItem] = {}
        #: Section key -> the added pages, ``[(number, name)]`` in order (`set_subpages`); and the lists and "+"s.
        self._subpages: dict[str, list] = {}
        self._subpage_lists: dict = {}
        self._plus: dict = {}
        #: Section key -> its group's form ("tiles", "rows", "feature").
        self._form: dict[str, str] = {}
        self._tips: list = []
        #: Every group's label, top to bottom (each fades to a hairline on the icon rail).
        self._labels: list = []
        self._tile_rows: list = []
        self.feature: FeatureButton | None = None
        titles = {k: (title, glyph) for k, title, glyph, _s in self._spec}
        active = self.current.split(".")[0]
        foot_groups = []
        for index, (gkey, glabel, form, keys) in enumerate(self._group_spec):
            if form == "feature":
                foot_groups.append((gkey, glabel, keys))
                continue
            group = QWidget()
            group.setObjectName(f"sbgroup-{gkey}" if gkey else "sbgroup")
            group.setStyleSheet("background:transparent;")
            gl = QVBoxLayout(group)
            gl.setContentsMargins(0, 0, 0, 0)
            gl.setSpacing(ROW_GAP)
            heading = _GroupLabel(t, glabel, rail_h=SEP_H if index else 0)
            self._labels.append(heading)
            gl.addWidget(heading)
            if form == "tiles":
                tiles = _TileRow(t)
                for key in keys:
                    title, glyph = titles[key]
                    tile = DataTile(t, title, glyph)
                    tile.set_active(key == active)
                    self._register(key, title, tile, form)
                    tiles.add(tile)
                self._tile_rows.append(tiles)
                gl.addWidget(tiles)
            else:
                for key in keys:
                    title, glyph = titles[key]
                    # No trailing chevron -- a section's pages are its own tab strip, unless it
                    # lists them here (`set_pages`), which adds the chevron then.
                    btn = NavButton(t, title, glyph=glyph, active=(key == active))
                    self._register(key, title, btn, form)
                    item = _SectionItem(btn)
                    self._section_items[key] = item
                    if self._spec == SECTIONS and key in sp.ENABLED:
                        self._add_subpage_controls(key, btn, item)
                    gl.addWidget(item)
            bl.addWidget(group)
        #: The first group's label (the one label of a one-group rail).
        self.group_label = self._labels[0] if self._labels else None
        #: Where `add_group` puts a further group (the preparation window's conditions):
        #: below the sections, above the stretch that keeps both at the top.
        self._body_lay = bl
        bl.addStretch(1)
        # always the view's width, whatever the rows' minimum: the rail sets the width, the rows squeeze into it.
        # A scroll area otherwise sizes its page to at least its minimum, and through the collapse the rail narrows
        # a pass before the rows do, so the tiles stood 180 px wide in a 48 px view (found by the GUI sweep's
        # CLIPPED check, 4 Nov 2026)
        body.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self._scroll = _BodyScroll()
        self._scroll.setWidget(body)
        lay.add(self._scroll, "body")
        self.body = body
        # THE JOBS RUNNING IN THE BACKGROUND, at the foot (`jobs_panel`, asked 2 Nov 2026): hidden while none runs;
        # a row opens its section. Under them the Figures group: the Composer, pinned where it never moves.
        from .jobs_panel import JobsPanel

        running = QWidget()
        running.setObjectName("sbfoot")
        rl = QVBoxLayout(running)
        rl.setContentsMargins(INSET_PAD, 0, INSET_PAD, 0)
        rl.setSpacing(0)
        self.jobs = JobsPanel(t)
        self.jobs.picked.connect(self.select)
        rl.addWidget(self.jobs)
        lay.add(running, "jobs")
        for gkey, glabel, (key,) in foot_groups:
            foot = QWidget()
            foot.setObjectName("sbfoot")
            fl = QVBoxLayout(foot)
            fl.setContentsMargins(INSET_PAD, 0, INSET_PAD, INSET_PAD)
            fl.setSpacing(0)
            group = QWidget()
            group.setObjectName(f"sbgroup-{gkey}" if gkey else "sbgroup")
            group.setStyleSheet("background:transparent;")
            gl = QVBoxLayout(group)
            gl.setContentsMargins(0, 0, 0, 0)
            gl.setSpacing(ROW_GAP)
            heading = _GroupLabel(t, glabel, rail_h=SEP_H)
            self._labels.append(heading)
            gl.addWidget(heading)
            title, glyph = titles[key]
            self.feature = FeatureButton(t, title, glyph)
            self.feature.set_active(key == active)
            self._register(key, title, self.feature, "feature")
            gl.addWidget(self.feature)
            fl.addWidget(group)
            lay.add(foot, "fixed")
        #: Extra groups, hidden while the rail is collapsed to icons.
        self._groups: list = []
        #: Sections that list their pages under their row (`set_pages`): section -> _PageList.
        self._lists: dict[str, _PageList] = {}
        self._chevrons: dict[str, _Chevron] = {}

        self._anim = QVariantAnimation(self)
        self._anim.setDuration(self.COLLAPSE_MS)
        self._anim.setEasingCurve(QEasingCurve.Linear)
        self._anim.valueChanged.connect(self._step)
        self._anim.finished.connect(self.slide_finished)
        # the page lists come back once the rail is wide again (they have no icon form)
        self.slide_finished.connect(self._show_lists)

    # -- subpages: the "+" and the added pages' rows (`smtracker.subpages`) -------------------------------------- #
    def _add_subpage_controls(self, key: str, btn: NavButton, item: _SectionItem) -> None:
        """The "+" at ``key``'s row (shown on hover, left of the chevron) and the empty list of its added pages."""
        from .subpage_rows import PlusAction, SubpageList, attach_row_ink, make_row

        title = SECTION_TITLE.get(key, key)
        plus = PlusAction(self.t, GLYPH + 2, f"Add a copy of {title} with its own settings and results")
        plus.clicked.connect(lambda k=key: self.subpage_add_requested.emit(k))
        btn.layout().addWidget(plus)
        # always shown on the wide rail, faint until the row is pointed at (hidden on the icon rail)
        attach_row_ink(btn, plus)
        plus.setVisible(not self.collapsed)
        self._plus[key] = plus
        # the "+" in a right-click menu too (ruled 4 Nov 2026), so a page can be added without hovering for it
        btn.setContextMenuPolicy(Qt.CustomContextMenu)
        btn.customContextMenuRequested.connect(lambda _pos, k=key, b=btn: self._section_menu(k, b))
        lst = SubpageList(self.t, make_row(NavButton), SUB_INDENT, SUB_PAD, SUB_GAP, SUB_H, GLYPH)
        lst.picked.connect(self.select)
        lst.renamed.connect(lambda n, text, k=key: self.subpage_renamed.emit(k, n, text))
        lst.deleted.connect(lambda n, k=key: self.subpage_deleted.emit(k, n))
        lst.moved.connect(lambda n, step, k=key: self.subpage_moved.emit(k, n, step))
        lst.dropped.connect(lambda n, i, k=key: self.subpage_dropped.emit(k, n, i))
        item.put_subpages(lst)
        lst.hide()
        self._subpage_lists[key] = lst

    def _section_menu(self, key: str, btn: NavButton):
        """The right-click menu of ``key``'s row: "New page", which asks for a page as the "+" does (a copy of the
        page in hand). Returned (and held) so a test can press its row."""
        menu = Panel(self.t, self.window())
        menu.set_trigger(btn)
        row = menu_row(self.t, menu, "New page", glyph="plus",
                       on_click=lambda k=key: self.subpage_add_requested.emit(k))
        menu.popup_under(btn)
        self._section_menu_panel = menu
        return row

    @property
    def current_base(self) -> str:
        """The current view as the section's OWN page would name it (``"sqd~2.jumps"`` -> ``"sqd.jumps"``): what
        the rail, the toolbar and every per-section rule read."""
        return sp.base_view(self.current)

    @property
    def current_page(self) -> str:
        """The current view's page key (``"sqd~2"``; ``"sqd"`` for a section's own page)."""
        return sp.page_of_view(self.current)

    def subpages(self, section: str) -> list:
        """``[(number, name)]`` of ``section``'s added pages, in order."""
        return list(self._subpages.get(section, ()))

    def set_subpages(self, section: str, entries) -> None:
        """Redraw ``section``'s added pages from ``entries`` = ``[(number, name)]``. When the page on show was one
        that is gone, the section's own page is selected (and ``selected`` emitted)."""
        if section not in self._subpage_lists:
            raise KeyError(section)
        entries = [(int(n), str(name)) for n, name in entries]
        self._subpages[section] = entries
        page = self._subs[section][0][0].partition(".")[2]
        lst = self._subpage_lists[section]
        lst.set_pages([(n, sp.join_view(section, n, page), name) for n, name in entries], self.current)
        lst.squeeze(min(FULL_BTN, self._items[section].maximumWidth()))
        lst.setVisible(bool(entries) and not self.collapsed)
        sec, number, _p = sp.split_view(self.current)
        if sec == section and number is not None and number not in {n for n, _name in entries}:
            self.select(section)
        elif sec == section and number is not None:
            self.select(self.current)               # the highlight and the breadcrumb of a renamed page
        # a remembered view of a page that is gone points at the section's own
        last = sp.split_view(self._last.get(section, section))
        if last[1] is not None and last[1] not in {n for n, _name in entries}:
            self._last[section] = self._subs[section][0][0]

    def _register(self, key: str, title: str, control, form: str) -> None:
        """One section's control: clicking it opens the section; on the icon rail its title is its tooltip."""
        control.clicked.connect(lambda k=key: self.select_section(k))
        self._sections[key] = control
        self._items[key] = control
        self._form[key] = form
        cut = control.elided if isinstance(control, DataTile) else (lambda: False)
        self._tips.append(tooltip.attach(
            self.t, control, title, side="right", enabled=lambda: self.collapsed or cut()))

    def form(self, section: str) -> str:
        """How ``section`` is drawn: "tiles", "rows" or "feature" (`GROUPS`)."""
        return self._form[section]

    def groups(self) -> tuple:
        """The rail's groups, top to bottom, `GROUPS`-shaped."""
        return self._group_spec

    def paintEvent(self, e):
        """The styled fill, then the rail's edge at ONE device pixel."""
        super().paintEvent(e)
        p = QPainter(self)
        w = 1.0 / self.devicePixelRatioF()
        p.fillRect(QRectF(self.width() - w, 0.0, w, float(self.height())),
                   QColor(self.t.sidebar_border))
        p.end()

    # -- behaviour --------------------------------------------------------- #
    def select(self, view: str) -> None:
        """Make `view` current. `view` is a SUBSECTION key; `current` keeps
        the full `section.subsection` id so a future toolbar/page can key its
        per-section controls off one string rather than a pair that can
        disagree. An ADDED page's view carries its number after the section
        (``"sqd~2.jumps"``, `smtracker.subpages`)."""
        section, number, _page = sp.split_view(view)
        if section not in self._items:
            raise KeyError(view)
        # a page the section no longer has (a session or a theme switch naming the HMM Test
        # page's "Conditions" while that page is off): its first page, never a dead id
        base = sp.base_view(view)
        if base not in {v for v, _l in self._subs[section]}:
            base = self._subs[section][0][0]
        # an added page that is gone (deleted, or a session of another build): the section's own
        if number is not None and number not in {n for n, _name in self._subpages.get(section, ())}:
            number = None
        view = base if number is None else sp.join_view(section, number, base.partition(".")[2])
        self._items[sp.split_view(self.current)[0]].set_active(False)
        self.current = view
        self._last[section] = view
        # an added page is current: its own row carries the bar and the section's stays unlit (two lit rows would
        # read as two places at once); on the icon rail, which has no added rows, the section's icon is lit
        self._items[section].set_active(number is None or self.collapsed)
        for lst in self._lists.values():
            lst.set_current(base)
        for lst in self._subpage_lists.values():
            lst.set_current(view)
        if self._form.get(section) != "feature":
            # the groups scroll where the window is short: the current section is never left out of sight
            self._scroll.ensureWidgetVisible(self._items[section], 0, INSET_PAD)
        self.selected.emit(view)

    def select_section(self, key: str) -> None:
        if key in self._lists and not self.pages_open(key):
            self.set_pages_open(key, True)
        # the row opens the section's OWN page (asked 4 Nov 2026: "clicking on SQD ... keeps being on sqd 2"), on the
        # tab last used; an added page is opened from its own row under it
        self.select(sp.base_view(self._last.get(key, self._subs[key][0][0])))

    # -- a section's pages, listed under its row -------------------------------------- #
    #: `ui.json`'s key for which sections have their pages listed.
    OPEN_KEY = "sidebar_pages_open"

    def pages(self, section: str) -> tuple:
        return tuple(self._subs[section])

    def trail(self, view: str) -> list:
        """The breadcrumb for `view` from the LIVE pages (`set_pages` changes them), so the HMM
        Test page reads "HMM > Conditions"; `trail_for` reads the fixed `SECTIONS`. An added page reads its
        own name in the section's place ("SQD 2 > Populations")."""
        section, number, _page = sp.split_view(view)
        title = next((t for k, t, _g, _s in self._spec if k == section), None)
        if title is None:
            return [view]
        if number is not None:
            title = next((name for n, name in self._subpages.get(section, ()) if n == number), title)
        base = sp.base_view(view)
        sub = next((lab for key, lab in self._subs.get(section, ()) if key == base), None)
        return [title] if sub is None else [title, sub]

    def set_pages(self, section: str, subs) -> None:
        """Replace `section`'s pages. Two or more are LISTED under its row, shadcn's collapsible
        sub-menu, with a chevron; one is not (the row is the page). Only a section that calls
        this lists its pages -- SQD, AppD and MSD keep their tab strips (23 Oct 2026). A tile or
        the feature button lists nothing: its pages change, nothing is drawn."""
        subs = tuple((str(v), str(lab)) for v, lab in subs)
        if not subs:
            raise ValueError("a section needs at least one page")
        if section not in self._items:
            raise KeyError(section)
        self._subs[section] = subs
        if self._last.get(section) not in {v for v, _l in subs}:
            self._last[section] = subs[0][0]
        old = self._lists.pop(section, None)
        if old is not None:
            discard(old)
        if self._form.get(section) != "rows":
            return
        btn = self._items[section]
        chev = self._chevrons.get(section)
        if len(subs) < 2:
            if chev is not None:
                chev.hide()
            return
        if chev is None:
            chev = _Chevron(self.t)
            chev.clicked.connect(lambda k=section: self.set_pages_open(k, not self.pages_open(k)))
            btn.layout().addWidget(chev)
            self._chevrons[section] = chev
        lst = _PageList(self.t, subs)
        lst.chevron = chev
        lst.picked.connect(self.select)
        lst.set_current(self.current)
        # into the layout BEFORE any visibility is set (`test_no_stray_windows`)
        self._section_items[section].put_under(lst)
        self._lists[section] = lst
        lst.set_open(self._remembered_open().get(section, True), animate=False)
        lst.squeeze(min(FULL_BTN, btn.maximumWidth()))
        lst.setVisible(not self.collapsed)
        chev.setVisible(not self.collapsed)

    def pages_open(self, section: str) -> bool:
        """Listed, or on its way to being (a list mid-slide counts by where it is heading)."""
        lst = self._lists.get(section)
        if lst is None:
            return False
        if lst._anim.state() == QVariantAnimation.Running:
            return lst._anim.endValue() == 1.0
        return lst.k > 0.5

    def set_pages_open(self, section: str, on: bool, animate: bool = True) -> None:
        lst = self._lists.get(section)
        if lst is None:
            return
        lst.set_open(on, animate=animate)
        from .. import theme

        state = self._remembered_open()
        state[section] = bool(on)
        theme.remember(self.OPEN_KEY, state)

    def _remembered_open(self) -> dict:
        from .. import theme

        v = theme.remembered(self.OPEN_KEY)
        return dict(v) if isinstance(v, dict) else {}

    def _show_lists(self) -> None:
        if self.collapsed:
            return
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.jobs.set_compact(False)
        for section, lst in self._lists.items():
            lst.squeeze(FULL_BTN)
            lst.show()
            self._chevrons[section].show()
        for section, lst in self._subpage_lists.items():
            lst.squeeze(FULL_BTN)
            lst.setVisible(bool(self._subpages.get(section)))
            self._plus[section].show()

    def set_jobs(self, jobs) -> None:
        """The jobs running in the background (`jobs_panel.JobsPanel.set_jobs`), at the rail's foot."""
        self.jobs.set_jobs(jobs)

    def set_project(self, title: str, subtitle: str) -> None:
        self.header.set_project(title, subtitle)

    def add_group(self, title: str, content: QWidget) -> QWidget:
        """A second labelled group under the sections, e.g. the preparation window's
        condition rows. It has no icon form, so it is HIDDEN while the rail is collapsed
        rather than squeezed. Returns the group's own widget."""
        group = QWidget()
        group.setStyleSheet("background:transparent;")
        gl = QVBoxLayout(group)
        gl.setContentsMargins(0, rem(0.75), 0, 0)
        gl.setSpacing(rem(0.25))
        heading = label(title, ty.ui(TEXT_SM, "medium", px=text_px(TEXT_SM_REM),
                                     wght=MEDIUM_WGHT),
                        mix(self.t.sidebar_foreground, self.t.sidebar, LABEL_ALPHA))
        heading.setFixedHeight(LABEL_H)
        heading.setStyleSheet(self._label_css % mix(self.t.sidebar_foreground,
                                                    self.t.sidebar, LABEL_ALPHA))
        gl.addWidget(heading)
        gl.addWidget(content)
        # Into the layout BEFORE any visibility is set (a widget shown with no parent is
        # a window of its own -- `test_no_stray_windows`).
        self._body_lay.insertWidget(self._body_lay.count() - 1, group)
        group.setVisible(not self.collapsed)
        self._groups.append(group)
        return group

    def set_collapsed(self, on: bool, animate: bool = True) -> None:
        """Tween to the icon rail. The words CLIP; they do not blink out."""
        if on == self.collapsed:
            return
        self.collapsed = on
        self.collapsed_changed.emit(on)
        for group in getattr(self, "_groups", ()):
            group.setVisible(not on)
        # no scroll bar while the rail slides or is an icon rail: 10 px of a 32 px cell; the wheel still scrolls
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        if on:
            # the running jobs as one spinner on the icon rail; the list comes back once the rail is wide
            self.jobs.set_compact(True)
            # no icon form: gone for the whole collapse, back once the rail is wide (`_show_lists`)
            for section, lst in getattr(self, "_lists", {}).items():
                lst.hide()
                self._chevrons[section].hide()
            for section, lst in self._subpage_lists.items():
                lst.hide()
                self._plus[section].hide()
        # the section row of an added page on show is lit on the icon rail, unlit when the rows are back
        cur_section, cur_number, _p = sp.split_view(self.current)
        if cur_number is not None and cur_section in self._items:
            self._items[cur_section].set_active(on)
        self._anim.stop()
        target = float((RAIL_W if on else SIDEBAR_W) + EDGE)
        if not animate:
            self._step(target)
            self.slide_finished.emit()
            return
        # Signals blocked while the range is set: `stop()` leaves `currentTime`
        # at the end of the previous run, so `setEndValue` would recompute the
        # current value at t=end and emit it -- a full-span bounce on every
        # collapse. See the prototype's own measurement of this bug.
        blocked = self._anim.blockSignals(True)
        self._anim.setStartValue(float(self.width()))
        self._anim.setEndValue(target)
        self._anim.blockSignals(blocked)
        self.slide_started.emit()
        self._anim.start()

    def _step(self, w) -> None:
        """One frame of the collapse: the rail width, the row width/height,
        and the group heading's fade, all driven off the same k in [0, 1]."""
        span = SIDEBAR_W - RAIL_W
        k = 0.0 if span <= 0 else max(0.0, min(1.0, (SIDEBAR_W + EDGE - w) / span))
        self.setFixedWidth(round(w))

        btn_w = round(FULL_BTN + (ICON_BTN - FULL_BTN) * k)
        pad = round(INSET_PAD * (1.0 - k))
        self.header.layout().setContentsMargins(pad, 0, pad, 0)
        rows = [c for c in self._sections.values() if isinstance(c, NavButton)]
        for btn in (self.header, *rows):
            btn.setMaximumWidth(btn_w)
            btn.squeeze(btn_w)
        for tiles in self._tile_rows:
            tiles.set_k(k)
        if self.feature is not None:
            self.feature.rail(k, btn_w)

        top = self.header.expanded_height()          # one line, or as many as the folder name needs
        head_h = round(top + (ROW_H - top) * k)
        self.header.setFixedHeight(head_h)
        self.header_centre.emit(INSET_PAD + head_h / 2.0)

        # each group's label fades; the first folds away, the others become a hairline between groups
        for heading in self._labels:
            heading.set_k(k)
