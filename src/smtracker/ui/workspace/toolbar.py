"""The workspace's right-hand tool rail: Compute, then three View tools.

Ported from `prototype/shadcn/toolbar.py` (git history, commit 09da35d).

The split is the one this project settled on after tracing the old comparison
browser's SQD populations tab control by control (see `compute.py`'s module
docstring for the full mapping): `ComputeMark` NAVIGATES to a dedicated page
holding the parameters that change the FIT (`window.ComputeConfigPage` --
outgrew a popover, see `compute.py`'s own docstring for why). Everything
else on THIS rail stays a popover and changes only what is already-computed
DATA is shown, never needing a refit:

* **Visualization properties -- REMOVED.** It held a mobility-threshold slider,
  a `live` checkbox, a D-correction toggle and a Reference picker, and nothing in
  the window listened to any of them: the slider and the picker emitted signals no
  handler was connected to, and the D-correction value was read by nothing (the
  bubble chart's own docstring called that toggle "deliberately NOT ported").
  Controls that move nothing on screen were worse than none, so they went, and
  with them the tool, which had nothing left in it.
* **Visibility** -- which of this view's figures are drawn, and the Spatial Map's ramp, rendering and scale.
* **Significance** (its own tool since 4 Nov 2026; the block sat under Visibility, a name that says "show or hide")
  -- the pairs of conditions compared, the test, what a bracket says; on the SQD and Apparent Diffusion Populations
  pages, dimmed elsewhere.
* **Conditions** -- the conditions' colours (palette and each one's swatch), their
  order, and with a folder design the grouping and colouring by factor. On the old
  browser this ALSO decided which conditions were shown and fit; the eye each row
  carried here hid nothing (no handler in the window) and was removed on 4 Nov 2026.
  A condition is hidden by its chart key's own eye.

`Data tables` (the prototype's fourth, removed tool) is not ported: its
reason for leaving stands here unchanged -- a table is the data OF a figure,
so it lives under that figure once figures exist, not as a free-floating dock.

POPOVERS ARE `menus.Panel`, NOT `menus.Menu` -- the same move commit 0683772
made for the onboarding screens' Filter/Regroup/swatch/Palette popovers, and
for the same reason: a `QMenu` is an OS window, and Windows draws its own drop
shadow for one outside the window itself, landing ON the 1 px `--border` and
darkening it to unrecoverable effect (measured there: 167 against a Card's
234, same token). A `Panel` is a child widget with no OS surface, so there is
nothing for DWM to shadow. `styled_menu(t, parent)` would have handed back a
`Menu` here regardless, since it only returns a `Panel` when `parent` already
IS one -- so the popovers below construct `Panel` directly rather than going
through it, exactly as the onboarding fix did.
"""
from __future__ import annotations

from qtpy.QtCore import QEvent, QObject, QPoint, QRect, QRectF, Qt, QTimer, QVariantAnimation, Signal
from qtpy.QtGui import QColor, QCursor, QFont, QPainter
from qtpy.QtWidgets import QApplication, QHBoxLayout, QLineEdit, QVBoxLayout, QWidget

from ..theme import hcl, icons
from ..theme import typography as ty
from ..widgets import tooltip as tip
from ..widgets.controls import Button, Segmented
from ... import rail_view
from ... import theme_view
from ...condition_colours import ConditionColours
from ... import condition_colours
from ..widgets.menus import Panel, close_panels_under, end_outros, menu_row, menu_section, retriggered
from ..widgets.swatches import PaletteRow, SwatchRow
from ..widgets.primitives import HoverFill, Surface, icon, label
from ..widgets.primitives import discard
from ... import condition_order

__all__ = [
    "CONDITIONS",
    "FIGURES",
    "PALETTES",
    "TOOLS",
    "ComputeMark",
    "EyeRow",
    "SelectButton",
    "SignificanceBody",
    "ToolRail",
    "Toolbar",
]

#: MUST equal `sidebar.ROOT_PX` -- restated rather than imported, since
#: `sidebar` will eventually place this rail and importing back would cycle.
#: `sidebar.ROOT_PX` corrected 18 -> 16 (a screenshot-measured width stood
#: in for the registry's own `16rem` -- see that constant's own comment
#: for the full story); restated here too.
ROOT_PX = 16


def rem(x: float) -> int:
    return round(x * ROOT_PX)


def text_px(rem_: float) -> int:
    return int(rem_ * ROOT_PX)


RAIL_W = rem(3)
EDGE = 1
PAD = rem(0.5)
GLYPH = rem(1)
BTN = rem(2)

#: The Compute mark: the same 36 px box as a tool button, a larger glyph.
MARK = rem(2)
MARK_GLYPH = rem(1.125)

SPIN_MS = 1000

POP_MIN = rem(12)
POP_PAD = rem(1)

#: The three tools, in rail order: key, glyph, name.
TOOLS: tuple[tuple[str, str, str], ...] = (
    ("visibility", "layers", "Visibility"),
    ("significance", "asterisk", "Significance"),
    ("conditions", "flask-conical", "Conditions"),
    # The section's settings as a receipt (`metadata.MetadataBody`), moved off the page.
    ("metadata", "receipt-text", "Metadata"),
)
#: Each tool's HOVER: its name and what it holds (4 Nov 2026, rail review: the bare names behind three icons -- layers,
#: a flask, a receipt -- said nothing of where the significance brackets or the colours are). Visibility's own is
#: `Toolbar.VISIBILITY_TIP`, which changes with the page.
#: The legacy window's button at the rail's foot (ruled 27 Nov 2026: "just put it in the bar bottom right, not
#: prominent in analysis tools") and its tooltip.
#: Asked 27 Nov 2026: "too much text for smtracker 2 button... just say smtracker 2 legacy window and analysis tools".
LEGACY_TIP = "SMTracker 2 legacy window and analysis tools"

TOOL_TIPS: dict = {
    "visibility": "Visibility: which figures of this page are drawn",
    "significance": rail_view.SIGNIFICANCE_TIP,
    "conditions": "Conditions: their colours and order, in every section",
    "metadata": "Metadata: the settings and replicates of the result on screen",
}

#: The figures each section draws, for the Visibility list. `(name,
#: is_nested)` -- the per-replicate side panel is drawn INSIDE the bubble
#: chart, so it is indented rather than listed as a third peer.
#:
#: The two `"appd-detail-*"` entries are the Apparent-D condition drill-in
#: page's OWN charts, one section per tab (`ApparentDDetail.tabs`) -- asked
#: for directly, scoped to whichever tab is active rather than always
#: listing all five. Names match the real `_ChartCard` titles verbatim
#: (`fractions_chart.FractionsChecksGrid` / `dynamics_chart.
#: DynamicsEvidenceGrid`), since `EyeRow`'s own `key` IS this name
#: (`VisibilityBody.set_section` below) -- `window.py` maps it back to each
#: grid's own short internal key.
FIGURES: dict[str, tuple[tuple[str, bool], ...]] = {
    # Empty: none of the five QC panels is a hide/show pair the way SQD's Slow/Fast populations
    # are, so there is nothing real to offer here yet. (A section or page absent from this table
    # lists nothing too, `FIGURES.get`, since 4 Nov 2026; a page of SQD other than Populations is
    # one. Where a section lists nothing and has no other block, the rail DIMS the tool,
    # `Toolbar._sync_visibility_tool`, rather than open an empty card.)
    "qc": (),
    # Nothing to hide or show: one card per condition. Indexed directly, so it must exist.
    "kinetics": (),
    "clusters": (),
    "msd": (),
    # The composer has no chart of its own to hide parts of.
    "composer": (),
    # Outlines and tracks are toggled on the page itself, beside the image.
    "images": (),
    # One card per condition (the state diagram and its tables): nothing to hide or show.
    "hmm": (),
    # EMPTY, like "qc": its two rows ("SQD populations", "Per-replicate fits") were
    # routed to a documented no-op in `window._on_visibility_toggled` -- a switch
    # that hides nothing. Per-day points on the bubble are the version worth
    # building; until then there is nothing to hide or show here.
    "sqd": (),
    # EMPTY too, for SQD's reason (found by the overnight tester, 4 Nov 2026): "Apparent Diffusion" and
    # "Per-condition detail" were rows `window._on_visibility_toggled` routed nowhere -- eyes that hid nothing. The
    # Significance block below them is the section's real Visibility content.
    "appd": (),
    "appd-detail-fractions": (("Fraction (per-cell)", False),
                              ("Distribution (per-track appD)", False)),
    "appd-detail-dynamics": (("State count (saSPT)", False),
                             ("Kinetics (HMM)", False),
                             ("Clustering", False)),
    # The Spatial Map SECTION: each name hides a whole population COLUMN of the grid. Its
    # two panels are peers, not nested -- hiding one never rescales the other
    # (`SpatialMapGrid.set_panel_visible`).
    "spatial": (("Short jumps", False), ("Long jumps", False)),
    # The section's Apparent Diffusion COMPUTATIONAL METHOD variant -- named
    # "-appd" rather than reusing the bare "appd" key, which already names
    # the unrelated Apparent-D SIDEBAR section (the distribution/SuperPlot
    # page). The map splits where its two fitted populations cross, per condition
    # (`spatial_map_view`), so the rows name no fixed D: they once said 0.15.
    "spatial-appd": (("Slow", False),
                     ("Fast", False)),
    # HMM (Kinetics)'s own COMPUTATIONAL METHOD variant, same shape as the
    # "-appd" pair above and the same reason for the suffix: "hmm" is not a
    # bare key anywhere else in this table, but named consistently with its
    # siblings rather than left the odd one out.
    "spatial-hmm": (("Slow (HMM state)", False),
                    ("Fast (HMM state)", False)),
    # Confinement (SLA): v2's three sets (`spatial_map_view.PANEL_KEYS_BY_METHOD`).
    "spatial-sla": (("Free tracks", False), ("Mixed-behaviour tracks", False),
                    ("Dwelling segments", False)),
    # SDA: v2's four maps (`spatial_map_view.PANEL_KEYS_BY_METHOD["sda"]`).
    "spatial-sda": (("Localisations", False), ("Speed", False),
                    ("Diffusion (apparent D)", False), ("Acceleration", False)),
}

#: The COLORMAP each section offers, keyed the same way as `FIGURES`. A
#: section absent from here shows no colormap block at all, which is most of
#: them -- a line chart has no ramp to choose.
#:
#: **This is DISPLAY, and it is the one control in this tool that is not a
#: visibility toggle.** It lives beside them because it answers the same
#: question -- what does this figure look like -- and changes no number: the
#: map's payload is identical under either ramp.
#:
#: **Two options, and the choice is NOT settled here.** `spatialsqd.
#: DEFAULT_CMAP` is `jet` by the PI's ruling, with its cost recorded beside
#: it (108 of 255 L* steps run backwards, so two densities print at the same
#: brightness and it does not survive greyscale); reversed `magma` is one of
#: the alternatives already rendered for the manuscript and is monotonic. No
#: verdict between them was ever written down, so both are offered and the
#: map's caption states which was used.
#: The Colormap row's last pill: it opens the custom-ramp editor (`ramp_dialog`) rather than
#: naming a ramp. A user's ramp travels as "custom:<name>" (`colour_ramps.CUSTOM_PREFIX`).
CUSTOM_PILL = "custom"
CUSTOM_KEY_PREFIX = "custom:"
COLORMAPS: dict[str, tuple[tuple[str, str], ...]] = {
    "spatial": (("jet", "jet"), ("magma", "magma"), ("viridis", "viridis"),
                (CUSTOM_PILL, "Custom…")),
    # Same ramp choice -- the colormap is DISPLAY and changes no number, so
    # it does not vary by computational method the way the panel names do.
    # A missing key here would silently drop the colormap picker for every
    # appD-computed map (`VisibilityBody`'s own `COLORMAPS.get(section, ())`).
    "spatial-appd": (("jet", "jet"), ("magma", "magma"), ("viridis", "viridis"),
                     (CUSTOM_PILL, "Custom…")),
    "spatial-hmm": (("jet", "jet"), ("magma", "magma"), ("viridis", "viridis"),
                    (CUSTOM_PILL, "Custom…")),
    "spatial-sla": (("jet", "jet"), ("magma", "magma"), ("viridis", "viridis"),
                    (CUSTOM_PILL, "Custom…")),
    # SDA opens in v2's own SDA ramp, reversed hot (asked for: "with same color map like
    # v2"); the others stay one click away. jet stays every OTHER method's default.
    "spatial-sda": (("hot_r", "v2 hot"), ("jet", "jet"), ("magma", "magma"),
                    ("viridis", "viridis"), (CUSTOM_PILL, "Custom…")),
}

#: The sections that can draw their grid as plain BINS or as a SMOOTH picture
#: of the same bins. DISPLAY only, exactly like the colormap above -- not one
#: number in the payload moves -- so it sits here and not in the Run dialog.
#: Bins is the default and smooth is opt-in (`spatial_map_view.RENDERINGS`,
#: whose first entry is the default and whose order this list must follow).
#: "n_eff" paints how many independent contributors built each bin instead of the
#: density -- also display only (the mask it shows was decided by the run).
RENDERINGS: tuple[tuple[str, str], ...] = (("bins", "Bins"), ("smooth", "Smooth"),
                                           ("neff", "n_eff"), ("tracks", "Tracks"))
RENDERING_SECTIONS: frozenset[str] = frozenset(COLORMAPS)
#: The Rendering pills' tooltips, in `RENDERINGS` order.
RENDERING_TIPS: tuple[str, ...] = (
    "Each bin as computed",
    "Smoothed for display, bilinear between bin centres; the data resolution is still the bin",
    "How many independent tracks (or cells) built each bin, instead of where the jumps are; "
    "the map says more under \u201cWhat is n_eff\u201d",
    "Confinement (SLA) maps: every track of each set as a line, in SMTracker 2's colours (free "
    "blue, mixed-behaviour green, dwelling red)",
)
#: The sections whose maps have TRACKS to draw: Confinement (SLA) only. Every other
#: method's pills leave "Tracks" out -- offered everywhere, it drew nothing on any other
#: map (reported 13 Oct 2026: "rendering -> tracks ... is an option for all maps, but only
#: works for SLA").
TRACK_SECTIONS: frozenset[str] = frozenset({"spatial-sla"})


def renderings_for(section: str | None) -> tuple:
    """The Rendering pills ``section`` offers, in `RENDERINGS` order: ``((key, text,
    tip), ...)``."""
    return tuple((key, text, tip) for (key, text), tip in zip(RENDERINGS, RENDERING_TIPS)
                 if key != "tracks" or section in TRACK_SECTIONS)


#: The Spatial Map grid's colour scale (`spatial_map_view.MATRIX_SCALES`, same order,
#: the default first). Display only, like the ramp: no number in a payload moves.
SCALES: tuple[tuple[str, str], ...] = (("population", "Population"), ("row", "Condition"),
                                       ("panel", "Map"), ("all", "All"))
#: What each pill means, as its tooltip: short pills keep the popover as narrow as its other rows
#: (the four full phrases made it 382 px against Rendering's 176).
SCALE_TIPS: tuple[str, ...] = (
    "One scale per population, shared by every condition (the default)",
    "One scale per condition (per replicate on the replicate page), shared by its populations",
    "Each map on its own scale, in counts: strongest contrast, no comparison by colour",
    "One scale for every map: both populations and every condition",
)

#: The workspace's conditions, in palette order. Placeholder pending real
#: wiring -- `Toolbar.set_conditions` overwrites this per loaded project.
CONDITIONS: tuple[tuple[str, str], ...] = (
    ("glucose", "Glucose"), ("malate", "Malate"),
)

PALETTES: tuple[tuple[str, str], ...] = tuple(
    (p.key, p.label) for p in hcl.PALETTES.values())


class ToolButton(Surface):
    """A rail button: shadcn's `Button variant="ghost" size="icon"`."""

    clicked = Signal(str)

    def __init__(self, t, key: str, glyph: str, *, size: int = BTN,
                 rest: str | None = None, parent=None):
        super().__init__(t, radius=t.radius_md, shadowed=False, parent=parent)
        self.t, self.key = t, key
        self._active = False
        #: False: dimmed, no hover, a click does nothing (`set_available`).
        self._available = True
        self._rest = t.background if rest is None else rest
        self.setObjectName(f"tool{id(self)}")
        self.setFixedSize(size, size)
        self.setCursor(Qt.PointingHandCursor)
        # `over_popups`: this rail's popovers open BESIDE it, so its buttons
        # stay visible and clickable while one is up -- a pointer crossing
        # them is a real hover. See `ToolRail.track_pointer`.
        self._fill = HoverFill(
            self, lambda c: (f"QWidget#{self.objectName()} {{ background:{c};"
                             f" border:none;"
                             f" border-radius:{t.radius_md:g}px; }}"),
            self._rest, t.accent, over_popups=True)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._lay = lay
        self.glyph = icon(glyph, GLYPH, t.foreground, stroke=2.0)
        lay.addWidget(self.glyph, alignment=Qt.AlignCenter)
        #: STATE ON THE RAIL (4 Nov 2026, rail review idea 5): a dot (something is hidden), a count badge (the
        #: pairs set; ``_badge_lit`` False when the brackets are off) and up to three condition colours, so the
        #: tool says what it holds without opening. All drawn in `paintEvent`.
        self._dot = False
        self._badge: str | None = None
        self._badge_lit = True
        self._swatches: tuple = ()

    def set_dot(self, on: bool) -> None:
        """A small disc at the button's top right: the tool holds a choice that changes the page (a figure hidden)."""
        if bool(on) != self._dot:
            self._dot = bool(on)
            self.update()

    def set_badge(self, text: str | None, lit: bool = True) -> None:
        """A count at the top right (the Significance tool's pairs); ``lit`` False draws it muted (pairs set, brackets
        off). None or empty: no badge."""
        text = text or None
        if (text, bool(lit)) != (self._badge, self._badge_lit):
            self._badge, self._badge_lit = text, bool(lit)
            self.update()

    def set_swatches(self, colours) -> None:
        """Up to `rail_view.SWATCHES` condition colours as small dots under the glyph (the Conditions tool)."""
        colours = rail_view.swatch_colours(colours)
        if colours != self._swatches:
            self._swatches = colours
            # the glyph rises a little to leave the dots their row
            self._lay.setContentsMargins(0, 0, 0, 5 if colours else 0)
            self.update()

    def paintEvent(self, e):
        super().paintEvent(e)
        if not (self._dot or self._badge or self._swatches):
            return
        t = self.t
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        ring = QColor(t.accent if self._active else self._rest)
        w, h = self.width(), self.height()
        if self._swatches:
            d, gap = 5.0, 2.5
            x0 = (w - (len(self._swatches) * d + (len(self._swatches) - 1) * gap)) / 2.0
            y = h - d - 4.0
            for i, colour in enumerate(self._swatches):
                rect = QRectF(x0 + i * (d + gap), y, d, d)
                p.setPen(Qt.NoPen)
                p.setBrush(ring)
                p.drawEllipse(rect.adjusted(-1.0, -1.0, 1.0, 1.0))        # a ring in the button's own colour
                p.setBrush(QColor(colour))
                p.drawEllipse(rect)
        if self._badge:
            font = QFont(p.font())
            font.setPixelSize(9)
            font.setBold(True)
            p.setFont(font)
            width = max(12.0, p.fontMetrics().horizontalAdvance(self._badge) + 6.0)
            rect = QRectF(w - width - 0.5, 0.5, width, 12.0)
            p.setPen(Qt.NoPen)
            p.setBrush(ring)
            p.drawRoundedRect(rect.adjusted(-1.0, -1.0, 1.0, 1.0), 7.5, 7.5)
            p.setBrush(QColor(t.primary if self._badge_lit else t.muted_foreground))
            p.drawRoundedRect(rect, 6.5, 6.5)
            p.setPen(QColor(t.primary_foreground if self._badge_lit else t.background))
            p.drawText(rect, Qt.AlignCenter, self._badge)
        elif self._dot:
            rect = QRectF(w - 11.0, 4.0, 7.0, 7.0)
            p.setPen(Qt.NoPen)
            p.setBrush(ring)
            p.drawEllipse(rect.adjusted(-1.5, -1.5, 1.5, 1.5))
            p.setBrush(QColor(t.primary))
            p.drawEllipse(rect)
        p.end()

    def set_active(self, on: bool) -> None:
        if on == self._active:
            return
        self._active = on
        t = self.t
        self._fill.set_colours(t.accent if on else self._rest, t.accent)
        self.update()

    @property
    def available(self) -> bool:
        return self._available

    def set_available(self, on: bool) -> None:
        """Dimmed and inert where the tool has nothing to offer on this page (the Visibility tool on QC, MSD, Jump
        distances ...), rather than opening an empty popover. Kept in its place, so the rail never shifts."""
        on = bool(on)
        if on == self._available:
            return
        self._available = on
        t = self.t
        self.glyph.colour = t.foreground if on else t.muted_foreground
        self.glyph.update()
        self.setCursor(Qt.PointingHandCursor if on else Qt.ArrowCursor)
        rest = t.accent if self._active else self._rest
        self._fill.set_colours(rest, t.accent if on else rest)

    def mouseReleaseEvent(self, e):
        if not self._available:
            return
        if e.button() == Qt.LeftButton and self.rect().contains(e.position().toPoint()):
            self.clicked.emit(self.key)


class ComputeMark(QWidget):
    """The page's ACTION, as a filled disc at the top of the rail.

    A disc rather than the rail's own ghost-button language, because this is
    the only control here that changes what the figures say -- see the
    module docstring's whole split. `glyph` is a pure function of two
    booleans: idle -> play; running + not hovered -> a turning loader;
    running + hovered -> a stop square (hover is the affordance that a
    running job can be cancelled by clicking it).
    """

    pressed = Signal()

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.setFixedSize(MARK, MARK)
        self.setCursor(Qt.PointingHandCursor)
        self._running = False
        self._hovered = False
        #: False: dimmed and inert, on a page with nothing to compute (`set_available`).
        self._available = True
        self._angle = 0.0
        self._spin = QVariantAnimation(self)
        self._spin.setStartValue(0.0)
        self._spin.setEndValue(360.0)
        self._spin.setDuration(SPIN_MS)
        self._spin.setLoopCount(-1)
        self._spin.valueChanged.connect(self._turn)

    @property
    def running(self) -> bool:
        return self._running

    @property
    def glyph(self) -> str:
        if not self._running:
            return "play"
        return "square" if self._hovered else "loader-circle"

    @property
    def available(self) -> bool:
        return self._available

    def set_available(self, on: bool) -> None:
        """Dimmed and inert on a page with nothing to compute (Composer, Images & tracks), where a press did
        nothing (4 Nov 2026, rail review). Kept in its place, so the rail never shifts."""
        on = bool(on)
        if on == self._available:
            return
        self._available = on
        self.setCursor(Qt.PointingHandCursor if on else Qt.ArrowCursor)
        self.update()

    def set_running(self, on: bool) -> None:
        if on == self._running:
            return
        self._running = on
        if on:
            self._angle = 0.0
            self._spin.start()
            self._pace()
        else:
            self._spin.stop()
        self.update()

    def set_hovered(self, on: bool) -> None:
        if on == self._hovered:
            return
        self._hovered = on
        self._pace()
        if self._running:
            self.update()

    def _pace(self) -> None:
        """Pause the turn while the stop square is up, rather than tick 60
        times a second behind a static glyph nobody can see move."""
        if not self._running:
            return
        from qtpy.QtCore import QAbstractAnimation

        state = self._spin.state()
        if self._hovered and state == QAbstractAnimation.Running:
            self._spin.pause()
        elif not self._hovered and state == QAbstractAnimation.Paused:
            self._spin.resume()

    def _turn(self, value) -> None:
        self._angle = float(value)
        self.update()

    def enterEvent(self, e):
        super().enterEvent(e)
        self.set_hovered(True)

    def leaveEvent(self, e):
        super().leaveEvent(e)
        self.set_hovered(False)

    def mouseReleaseEvent(self, e):
        if not self._available:
            return
        if e.button() == Qt.LeftButton and self.rect().contains(e.position().toPoint()):
            self.pressed.emit()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(self.t.primary if self._available else self.t.muted))
        box = QRectF(0, 0, float(self.width()), float(self.height()))
        p.drawEllipse(box)
        inset = (MARK - MARK_GLYPH) / 2.0
        glyph_rect = QRectF(inset, inset, float(MARK_GLYPH), float(MARK_GLYPH))
        name = self.glyph
        if name == "loader-circle":
            centre = box.center()
            p.translate(centre)
            p.rotate(self._angle)
            p.translate(-centre)
        icons.paint(p, name, glyph_rect,
                    self.t.primary_foreground if self._available else self.t.muted_foreground, stroke=2.0)
        p.end()


class _RailTracker(QObject):
    """Pointer moves, seen where they are actually delivered: the popup.

    A popup grabs the mouse, so the rail hears no Enter/Leave/Move for as
    long as one of its own popovers is open -- and these popovers open
    BESIDE the rail rather than over it, so the other buttons must keep
    hovering. An app-wide filter is the only place those moves are visible.

    **LEAVE is watched too, and moves alone left a tooltip stranded.**
    `_pointer_moved` lights a hover from GEOMETRY, not from a real widget
    Enter, so it is Qt's own `HoverFill._CursorWatch` gap one level up: a
    tooltip shown this way has nothing but the NEXT move to clear it, and
    the pointer leaving the rail for the desktop entirely -- the ordinary
    way to leave a RIGHT-hand rail, off the window's own edge -- delivers
    no further move at all. Reported as a tool's tooltip staying on screen
    after hovering away. `HoverFill` already documents the general shape:
    "the widget that reports the departure is not the widget that is
    stuck" -- Leave is exactly the event that says the pointer left
    somewhere this application will not hear about again.
    """

    def __init__(self, rail):
        super().__init__(rail)
        self._rail = rail

    def eventFilter(self, obj, event):
        if event.type() in (QEvent.MouseMove, QEvent.HoverMove,
                            QEvent.Leave, QEvent.HoverLeave):
            self._rail._pointer_moved()
        return False


class ToolRail(QWidget):
    """The strip itself: the mark, then the tools, top-aligned, and the light/dark
    toggle at the foot."""

    picked = Signal(str)
    compute_pressed = Signal()
    #: The light/dark toggle at the foot of the rail -- a plain button, not a tool:
    #: it opens no popover, so it is kept out of `buttons` and `targets`, whose
    #: every entry the pointer tracker treats as a popover trigger.
    theme_pressed = Signal()
    #: The options screen (`ui.widgets.options_dialog`), above the theme toggle: a plain
    #: button too, for the same reason.
    options_pressed = Signal()
    #: The rail's Help: the manual at its analysis chapter.
    help_pressed = Signal()
    #: The legacy window (SMTracker 2.0's tabs on these conditions), the rail's small button above Options.
    legacy_pressed = Signal()

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.setObjectName("toolrail")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(
            f"QWidget#toolrail {{ background:{t.background}; border:none; }}")
        self.setFixedWidth(RAIL_W + EDGE)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(PAD + EDGE, PAD, PAD, PAD)
        lay.setSpacing(rem(0.25))
        self.buttons: dict[str, ToolButton] = {}
        self._tips: list = []
        self.compute = ComputeMark(t)
        self.compute.pressed.connect(self.compute_pressed.emit)
        lay.addWidget(self.compute)
        lay.addSpacing(rem(0.75))
        self._compute_tip = tip.attach(t, self.compute, "Compute",
                                       side="left", top_level=True)
        self._tips.append(self._compute_tip)
        self._tracker = None
        self._hovered: str | None = None
        for key, glyph, title in TOOLS:
            b = ToolButton(t, key, glyph)
            b.clicked.connect(self.picked.emit)
            self._tips.append(tip.attach(t, b, TOOL_TIPS.get(key, title), side="left", top_level=True))
            lay.addWidget(b)
            self.buttons[key] = b
        lay.addStretch(1)
        self.help_button = ToolButton(t, "help", "info")
        self.help_button.clicked.connect(lambda _key: self.help_pressed.emit())
        self._tips.append(tip.attach(t, self.help_button, "Help (the manual, on the website or here)", side="left",
                                     top_level=True))
        lay.addWidget(self.help_button)
        self.legacy_button = ToolButton(t, "legacy", "history")
        self.legacy_button.clicked.connect(lambda _key: self.legacy_pressed.emit())
        self._tips.append(tip.attach(t, self.legacy_button, LEGACY_TIP, side="left", top_level=True))
        lay.addWidget(self.legacy_button)
        self.options_button = ToolButton(t, "options", "sliders-horizontal")
        self.options_button.clicked.connect(lambda _key: self.options_pressed.emit())
        self._tips.append(tip.attach(t, self.options_button, "Options", side="left",
                                     top_level=True))
        lay.addWidget(self.options_button)
        # The glyph names what a click SWITCHES TO, the usual convention: a moon
        # while light, a sun while dark.
        scheme = getattr(t, "color_scheme", "light")
        self.theme_button = ToolButton(t, "theme", theme_view.switch_glyph(scheme))
        self.theme_button.clicked.connect(lambda _key: self.theme_pressed.emit())
        self._tips.append(tip.attach(
            t, self.theme_button, theme_view.switch_tip(scheme),
            side="left", top_level=True))
        lay.addWidget(self.theme_button)
        self.targets: dict = {"compute": self.compute, **self.buttons}

    def set_active(self, key: str | None) -> None:
        for k, b in self.buttons.items():
            b.set_active(k == key)

    def set_tool_tip(self, key: str, text: str) -> None:
        """The hover text of the tool ``key`` (`TOOLS`), e.g. why it is dimmed on this page."""
        button = self.buttons.get(key)
        watch = next((w for w in self._tips if w.target is button), None) if button is not None else None
        if watch is None or watch.text == text:
            return
        watch.text = text                       # read at the next hover: the one tip window is shared by every tool

    def set_compute_tip(self, text: str) -> None:
        watch = self._compute_tip
        if watch.text == text:
            return
        watch.text = text
        if watch.tip.isVisible():
            # `settled=True`: the pointer never left Compute for the text to
            # have changed under it -- see `Tooltip.show_for`'s own doc.
            watch.tip.show_for(self.compute, text, watch.side, settled=True)

    def track_pointer(self, on: bool) -> None:
        app = QApplication.instance()
        if app is None:
            return
        if on and self._tracker is None:
            self._tracker = _RailTracker(self)
            app.installEventFilter(self._tracker)
            self.destroyed.connect(lambda *_: self._drop_tracker(app))
        elif not on and self._tracker is not None:
            app.removeEventFilter(self._tracker)
            self._tracker = None
            self._hovered = None
            # Resetting the BOOKKEEPING is not the same as the WIDGET going
            # away -- the same distinction `Tooltip._vanish` exists for.
            # `_pointer_moved` can leave a tip standing (see `_RailTracker`'s
            # own docstring); closing the popover is the one moment this rail
            # can still act on its own, since the per-widget Enter/Leave path
            # takes back over immediately after and cannot be trusted to have
            # a genuine Leave queued up for whichever button this leaves
            # behind.
            #
            # **UNCONDITIONALLY -- and that was itself a bug.** A blind
            # `dismiss_tips()` here does not know the cursor might genuinely
            # still be ON the trigger that is closing: `_open_popover`'s
            # retrigger branch calls `show_tip(key)` for exactly that case
            # (click the same icon again -- the cursor never left it), and
            # this runs on the SAME popover-closing signal, in no guaranteed
            # order against it. Reported as "click the same icon again, the
            # tooltip appears and then disappears" -- `show_tip` showing it
            # correctly, immediately undone by this. So dismiss only when
            # `Tooltip.is_over_target` says the pointer has genuinely left.
            #
            # **REPORTED AGAIN after every rail icon started SHARING one
            # `top_level` window** (see `tooltip._SHARED_TOP_LEVEL`), and for
            # exactly the reason its own docstring warns about: this used to
            # ask the geometry question PER ICON (`watch.target` against the
            # cursor) and dismiss THAT icon's own tip -- harmless when every
            # icon owned a separate window, since dismissing icon B's never
            # touched icon A's. Now every `watch.tip` in this list IS the
            # same object, so the loop still asked the right question for
            # icon A (skipped it, correctly) but then fired `dismiss()` on
            # the very same shared window anyway via B, C or D's own turn
            # through the loop -- undoing the skip it had just made. There is
            # only ONE tip to ask about now, not one per icon, so ask it
            # once via its own `is_over_target` rather than re-deriving the
            # same geometry question per icon against a target that may not
            # even be the one it is currently showing for.
            seen: set = set()
            for watch in self._tips:
                tooltip = watch.tip
                if id(tooltip) in seen:
                    continue
                seen.add(id(tooltip))
                if tooltip.is_over_target():
                    continue
                tooltip.dismiss(instant=True)

    @staticmethod
    def _drop_tracker(app) -> None:
        for obj in app.findChildren(_RailTracker):
            app.removeEventFilter(obj)

    def _pointer_moved(self) -> None:
        for b in self.buttons.values():
            b._fill.resync()
        at = QCursor.pos()
        under = next((k for k, w in self.targets.items()
                      if w.rect().contains(w.mapFromGlobal(at))), None)
        self.compute.set_hovered(under == "compute")
        if under == self._hovered:
            return
        self._hovered = under
        self.dismiss_tips(instant=under is not None)
        if under is not None and not tip.pressed():
            self.show_tip(under)

    def show_tip(self, key: str) -> None:
        button = self.targets.get(key)
        if button is None:
            return
        watch = next((w for w in self._tips if w.target is button), None)
        if watch is not None:
            # `settled=True`: both callers use this ONLY while a popover has
            # the rail's mouse grab -- `_open_popover`'s retrigger-close
            # branch (the icon was never un-hovered) and `_pointer_moved`
            # (geometry, not a real Enter). Neither is the ordinary
            # arriving-from-outside hover the slide-in exists for, and the
            # retrigger case in particular has a REAL click landing during
            # the ~150 ms the reserved geometry would otherwise overlap the
            # icon -- see `Tooltip.show_for`'s own doc for what that costs.
            watch.tip.show_for(button, watch.text, watch.side, settled=True)

    def dismiss_tips(self, instant: bool = True) -> None:
        for watch in self._tips:
            watch.tip.dismiss(instant=instant)

    def paintEvent(self, e):
        super().paintEvent(e)
        p = QPainter(self)
        dpr = self.devicePixelRatioF()
        x = round(0.5 * dpr) / dpr
        p.fillRect(QRectF(x, 0, 1.0 / dpr, self.height()), QColor(self.t.border))
        p.end()


class EyeRow(Surface):
    """One thing's on/off, as an eye (Visibility); or, with ``toggles=False``, a swatch row
    (Conditions): the condition's colour leading its name, the whole row opening the colour
    picker.

    `dot` is a condition's series colour: on the right of a toggling row, in the eye's lane
    of a swatch row.
    """

    toggled = Signal(str, bool)
    #: The colour dot was clicked (Conditions only): the key, and the dot's
    #: rect in GLOBAL coordinates, to anchor a colour picker under it.
    dot_clicked = Signal(str, object)

    HEIGHT = rem(2)
    #: The glyphs of on and off, and whether off dims the name: a `CheckRow` overrides all three.
    GLYPH_ON, GLYPH_OFF, DIM_OFF = "eye", "eye-off", True
    DOT = rem(0.5)
    #: How far round the dot a click still counts as on it -- a 8 px dot is
    #: too small a target on its own.
    DOT_HIT = rem(1.25)

    def __init__(self, t, key: str, text: str, *, on: bool = True,
                 nested: bool = False, dot: str | None = None, toggles: bool = True, parent=None):
        super().__init__(t, radius=t.radius_sm, shadowed=False, parent=parent)
        self.t, self.key, self._on, self._dot = t, key, on, dot
        #: False: a SWATCH ROW (the Conditions tool's): no eye, the dot leads the name, and a click anywhere on the
        #: row opens the colour picker. Its eye hid nothing -- `ConditionsBody.toggled` reached no handler in the
        #: window (found 4 Nov 2026, rail review): an eye that hides nothing is worse than none, the rule the
        #: Visualization tool and the SQD rows were removed by.
        self._toggles = bool(toggles)
        self.setObjectName(f"eye{id(self)}")
        self.setFixedHeight(self.HEIGHT)
        self.setCursor(Qt.PointingHandCursor)
        self._fill = HoverFill(
            self, lambda c: (f"QWidget#{self.objectName()} {{ background:{c};"
                             f" border:none;"
                             f" border-radius:{t.radius_sm:g}px; }}"),
            t.popover, t.accent)
        lay = QHBoxLayout(self)
        # the dot is painted at the right edge, so its room is the layout's right margin: the name never runs under
        # it (found by the overnight tester, 4 Nov 2026: "protein A glucos" with the dot over its last letter)
        if not self._toggles:
            # a swatch row's dot is painted at the LEFT, in the lane the eye takes on a toggling row
            lay.setContentsMargins(rem(0.375) + GLYPH + rem(0.5), 0, rem(0.375), 0)
        else:
            lay.setContentsMargins(rem(0.375), 0, rem(0.375) + (self.DOT + rem(0.5) if dot is not None else 0), 0)
        lay.setSpacing(rem(0.5))
        if not self._toggles:
            self.glyph = None
        elif nested:
            # A nested figure is drawn INSIDE its parent, and the indent says
            # so; it carries no eye of its own -- you cannot hide the
            # per-replicate strip while showing the chart it is drawn in.
            lay.addSpacing(GLYPH + rem(0.5))
            self.glyph = None
        else:
            self.glyph = icon(self.GLYPH_ON if on else self.GLYPH_OFF, GLYPH, t.foreground,
                              stroke=2.0)
            lay.addWidget(self.glyph)
        ink = t.foreground if (on or not self.DIM_OFF) and not nested else t.muted_foreground
        self.title = label(text, ty.ui("sm", px=text_px(0.875)), ink)
        lay.addWidget(self.title)
        lay.addStretch(1)
        self._nested = nested

    def paintEvent(self, e):
        super().paintEvent(e)
        if self._dot is None:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(self._dot))
        r = self.DOT / 2.0
        p.drawEllipse(self._dot_rect())
        p.end()

    def _dot_rect(self) -> QRectF:
        r = self.DOT / 2.0
        if not self._toggles:
            # centred in the lane a toggling row's eye takes, so the names line up with Visibility's
            return QRectF(rem(0.375) + (GLYPH - self.DOT) / 2.0,
                          self.height() / 2.0 - r, self.DOT, self.DOT)
        return QRectF(self.width() - rem(0.375) - self.DOT,
                      self.height() / 2.0 - r, self.DOT, self.DOT)

    def _on_dot(self, pos) -> bool:
        if self._dot is None:
            return False
        c = self._dot_rect().center()
        h = self.DOT_HIT / 2.0
        return abs(pos.x() - c.x()) <= h and abs(pos.y() - c.y()) <= h

    def set_dot(self, colour: str) -> None:
        self._dot = colour
        self.update()

    def set_on(self, on: bool) -> None:
        if on == self._on or self._nested or not self._toggles:
            return
        self._on = on
        self.glyph.name = self.GLYPH_ON if on else self.GLYPH_OFF
        self.glyph.colour = self.t.foreground if on else self.t.muted_foreground
        self.glyph.update()
        self.title.setStyleSheet(
            f"color:{self.t.foreground if on or not self.DIM_OFF else self.t.muted_foreground};"
            f" background:transparent;")

    def mouseReleaseEvent(self, e):
        if e.button() != Qt.LeftButton or self._nested:
            return
        pos = e.position().toPoint()
        if not self.rect().contains(pos):
            return
        if self._on_dot(pos) or (not self._toggles and self._dot is not None):
            r = self._dot_rect().toAlignedRect()
            self.dot_clicked.emit(
                self.key, QRect(self.mapToGlobal(r.topLeft()), r.size()))
            return
        self.set_on(not self._on)
        self.toggled.emit(self.key, self._on)


class CheckRow(EyeRow):
    """One line of a checklist (the Significance tool's pairs): a check box, then the name. Off is a plain empty box
    and the name keeps its ink, so a list of choices does not read as disabled."""

    GLYPH_ON, GLYPH_OFF, DIM_OFF = "square-check", "square", False


class SelectButton(Surface):
    """shadcn's `SelectTrigger`: the current value and a chevron, full width."""

    changed = Signal(str)

    HEIGHT = rem(2)

    def __init__(self, t, options, *, current: str, animated: bool = True,
                 tags=None, parent=None):
        super().__init__(t, radius=t.radius_md, shadowed=False, parent=parent)
        self.t, self.options, self.current = t, tuple(options), current
        self._animated = animated
        #: Optional callable returning ``{option key: short tag}``, shown
        #: muted at the right of that row ("already used elsewhere"). Asked at
        #: every open, so it always reflects the current state. Purely
        #: informational: every option stays selectable.
        self._tags = tags
        self.setObjectName(f"sel{id(self)}")
        self.setFixedHeight(self.HEIGHT)
        self.setCursor(Qt.PointingHandCursor)
        self._fill = HoverFill(
            self, lambda c: (f"QWidget#{self.objectName()} {{ background:{c};"
                             f" border:1px solid {t.input};"
                             f" border-radius:{t.radius_md:g}px; }}"),
            t.background, t.background)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(rem(0.5), 0, rem(0.5), 0)
        lay.setSpacing(rem(0.5))
        self.title = label(self._label_for(current), ty.ui("sm", px=text_px(0.875)),
                           t.foreground)
        lay.addWidget(self.title)
        lay.addStretch(1)
        lay.addWidget(icon("chevron-down", GLYPH, t.muted_foreground, stroke=2.0))

    def _label_for(self, key: str) -> str:
        return next((lab for k, lab in self.options if k == key), key)

    def set_current(self, key: str) -> None:
        self.current = key
        self.title.setText(self._label_for(key))

    def mouseReleaseEvent(self, e):
        if e.button() != Qt.LeftButton:
            return
        if retriggered(self):
            return
        t = self.t
        # `Panel`, in the page -- see the module docstring.
        menu = Panel(t, self.window(), root=ROOT_PX, animated=self._animated)
        menu.set_trigger(self)
        current_row = None
        tags = dict(self._tags()) if self._tags else {}
        for key, lab in self.options:
            row = menu_row(t, menu, lab, current=(key == self.current),
                           count=tags.get(key), root=ROOT_PX,
                           on_click=lambda k=key: self._pick(k))
            if key == self.current:
                current_row = row
        menu.set_minimum_visible_width(self.width())
        # `popup_aligned`, not a flat offset below the trigger -- shadcn's
        # `Select` opens with the CURRENTLY SELECTED row where the trigger
        # itself sits (`Panel.popup_aligned`'s own docstring). Falls back to
        # the old flush-below placement only if `self.current` matches
        # nothing in `self.options` (a stale value from a rebuilt list) --
        # `current_row` would stay `None` and there is nothing to align to.
        if current_row is not None:
            menu.popup_aligned(self, current_row)
        else:
            menu.popup(self.mapToGlobal(QPoint(0, self.height() + 4)))

    def _pick(self, key: str) -> None:
        self.set_current(key)
        self.changed.emit(key)


def _caption(t, text: str):
    return label(text, ty.ui("xs", "medium", px=text_px(0.75)), t.muted_foreground)


def _title(t, text: str):
    return label(text, ty.ui("sm", "semibold", px=text_px(0.875)), t.foreground)


def _population_tip(key, words: str, kind: str = "") -> str:
    """The hover of a population pill in the Significance block: whose fraction a bracket compares (on the HMM
    comparison, which panel)."""
    from ... import significance as sig

    if kind == sig.HMM_KIND:
        return ("Every panel: a bracket on each, with the correction applied over all of them" if key == "all" else
                f"{words}: a bracket on that panel, from each condition's per-replicate HMM fits")
    if key == "all":
        return "Every population's fraction: a bracket for each"
    if key == "below":
        return (f"The share of molecules in states at {words} (the run's threshold). It means the same whatever "
                "the number of populations, so conditions fitted with different counts can be compared.")
    return f"Population {key}'s fraction (1 is the slowest)"


#: The Pairs shortcut's entries (`SignificanceBlock.rules`): the prompt it rests on, the rules, and the prefix of
#: "Each against [condition]" (the condition's key follows).
RULE_PROMPT, RULE_EVERY, RULE_NONE, RULE_AGAINST = "", "every", "none", "against:"


class SignificanceBlock(QWidget):
    """The Visibility popover's SIGNIFICANCE block, on the SQD and Apparent D sections (asked
    for 14 Oct 2026: "add in the viewing options next to the plots a way to show significance
    in the plot ... then during export, it exports how u see it").

    Brackets on or off (an eye, like the figures above it), the test (a list since 24 Oct 2026:
    `significance.tests_for` -- on the SQD bubbles every test, on the apparent-D chart the two on
    per-replicate values -- with the equivalence test's margin beside it when that is chosen), whose fraction (a population, or all), what a bracket says (stars,
    the exact p, or both) and the star thresholds (asked for 14 Oct 2026: "option to change
    what each star means; standard should be the normal values"), and the pairs: a checklist of
    every pair of the chart's conditions under two shortcuts ("Every pair", "Each against [condition]"). Under them, each pair's p or why it has none (`significance.status_lines`).
    Every change is emitted whole (`changed`, a `significance.ViewChoice`); the window keeps
    one per section, paints the chart from it and exports from it. Nothing here computes."""

    changed = Signal(object)

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.setStyleSheet("background:transparent;")
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(rem(0.5))
        self._lay.addWidget(_caption(t, "Significance"))
        self._body = None
        self._shape = None
        self.choice = None
        self.eye = self.test = self.population = self.rules = None
        self.label = self.stars = self.margin = self.margin_row = None
        self.correction = None
        self._tests: tuple = ()
        self.checks: dict = {}
        self.lines = label("", ty.ui("xs", tabular=True), t.muted_foreground)
        self.lines.setWordWrap(True)
        # Wrapped, never widening the popover: a line per test, and "all" makes many.
        self.lines.setMaximumWidth(rem(18))
        self._lay.addWidget(self.lines)

    def configure(self, conditions, populations, mixed: bool, choice, paired: bool = False, kind: str = "") -> None:
        """``conditions``: ``((name, shown), ...)`` the chart draws; ``populations``:
        `significance.population_choices`; ``mixed``: whether the mixed model is offered; ``paired``: whether
        replicates were declared as pairs on the Groups board (the paired test is offered only then); ``kind``: the
        chart's figure kind (the HMM comparison offers its own tests and panels, `significance.tests_for`)."""
        # A CHOICE OF THE PAIRED TEST WITHOUT PAIRS (a session saved with pairs, opened after the box was unticked)
        # stays in the list, so the list shows the test the brackets compute and the line under it says why it refuses
        paired = bool(paired) or getattr(choice, "test", "") == "paired"
        shape = (tuple(conditions), tuple(populations), bool(mixed), paired, str(kind))
        if shape != self._shape:
            self._build(*shape)
            self._shape = shape
        self._show(choice)
        # NO CHART, NO CONTROLS (4 Nov 2026, rail review): before a Run the pair pickers were empty and `+` refused
        # with nothing to pick; the line under them ("Run the fit first ...") is all there is to say.
        self._body.setVisible(bool(shape[0]))

    def _build(self, conditions, populations, mixed, paired=False, kind="") -> None:
        from ... import significance as sig

        t = self.t
        if self._body is not None:
            discard(self._body)
        body = QWidget()
        body.setStyleSheet("background:transparent;")
        lay = QVBoxLayout(body)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(rem(0.5))
        self.eye = EyeRow(t, "brackets", "Brackets on the chart", on=False)
        self.eye.setToolTip("Draw the pairs' brackets on the chart; the export draws them as shown")
        self.eye.toggled.connect(lambda _k, on: self._emit(on=bool(on)))
        lay.addWidget(self.eye)
        # THE TEST, A LIST: six do not fit side by side in the popover (`significance.TESTS`).
        self._tests = sig.tests_for(mixed, paired, kind)
        self.test = SelectButton(t, [(k, sig.TEST_LABELS[k]) for k in self._tests], current=self._tests[0])
        self.test.setToolTip(sig.TEST_TIPS[self._tests[0]])
        # a test picked by hand is kept: a chunk chart's standard (`significance.resolved`) never swaps it back
        self.test.changed.connect(lambda key: self._emit(test=key, test_chosen=True))
        lay.addWidget(self.test)
        # The equivalence margin, shown only for the equivalence test: chosen before looking.
        self.margin_row = QWidget()
        self.margin_row.setStyleSheet("background:transparent;")
        margin_lay = QHBoxLayout(self.margin_row)
        margin_lay.setContentsMargins(0, 0, 0, 0)
        margin_lay.setSpacing(rem(0.375))
        self.margin_words = label("Same within ±", ty.ui("xs"), t.muted_foreground)
        margin_lay.addWidget(self.margin_words)
        self.margin = QLineEdit(f"{sig.DEFAULT_MARGIN_PP:g}")
        self.margin.setFont(ty.ui("sm", px=text_px(0.875)))
        self.margin.setFixedHeight(SelectButton.HEIGHT)
        self.margin.setToolTip("The largest difference, in percentage points of the fraction, that still counts "
                               "as the same. Set it before looking at the results, not chosen from the data "
                               "you are testing.")
        self.margin.setStyleSheet(
            f"QLineEdit {{ background:{t.background}; color:{t.foreground};"
            f" border:1px solid {t.input}; border-radius:{t.radius_md:g}px;"
            f" padding:0px 8px; }}")
        self.margin.editingFinished.connect(self._margin_typed)
        margin_lay.addWidget(self.margin, 1)
        margin_lay.addWidget(label("pp", ty.ui("xs"), t.muted_foreground))
        lay.addWidget(self.margin_row)
        self.margin_row.setVisible(False)
        self._pops = [k for k, _w in populations]
        self.population = None
        if populations:
            self.population = Segmented(t, [w for _k, w in populations], current=0,
                                        size="small", tips=[_population_tip(k, w, kind) for k, w in populations])
            self.population.changed.connect(lambda i: self._emit(population=self._pops[i]))
            lay.addWidget(self.population)
        self.label = Segmented(t, [sig.LABEL_WORDS[k] for k in sig.LABELS], current=0,
                               size="small",
                               tips=["Stars on the adjusted p (the thresholds and the correction below); the "
                                     "exact p goes into the legend and the source data.",
                                     "The exact adjusted p on the bracket.",
                                     "Stars and the exact p on the bracket."])
        self.label.changed.connect(lambda i: self._emit(label=sig.LABELS[i]))
        lay.addWidget(self.label)
        stars_row = QHBoxLayout()
        stars_row.setSpacing(rem(0.375))
        stars_row.addWidget(label("Stars at p <", ty.ui("xs"), t.muted_foreground))
        self.stars = QLineEdit(", ".join(f"{v:g}" for v in sig.DEFAULT_STARS))
        self.stars.setFont(ty.ui("sm", px=text_px(0.875)))
        self.stars.setFixedHeight(SelectButton.HEIGHT)
        self.stars.setToolTip("One threshold per star, largest first: 0.05, 0.01, 0.001 gives "
                              "*, ** and ***; add 0.0001 for ****. Above the first: ns.")
        self.stars.setStyleSheet(
            f"QLineEdit {{ background:{t.background}; color:{t.foreground};"
            f" border:1px solid {t.input}; border-radius:{t.radius_md:g}px;"
            f" padding:0px 8px; }}")
        self.stars.editingFinished.connect(self._stars_typed)
        stars_row.addWidget(self.stars, 1)
        lay.addLayout(stars_row)
        # THE CORRECTION over the brackets shown (10 Nov 2026): Holm, the standard, or Benjamini-Hochberg, which the
        # lab's own legends name. A list, never pills: "Benjamini-Hochberg" beside "Holm" is wider than the popover.
        self.correction = SelectButton(
            t, [(k, f"Correction: {sig.CORRECTION_NAMES[k]}") for k in sig.CORRECTIONS],
            current=sig.DEFAULT_CORRECTION)
        self.correction.setToolTip(sig.CORRECTION_TIPS[sig.DEFAULT_CORRECTION])
        self.correction.changed.connect(lambda key: self._emit(correction=key))
        lay.addWidget(self.correction)
        lay.addWidget(_caption(t, "Pairs"))
        options = tuple((name, shown) for name, shown in conditions)
        names = [name for name, _shown in options]
        # THE SHORTCUTS (4 Nov 2026, rail review): "Every pair" and "Each against [condition]" SET the pairs, "No pairs"
        # clears them. They only pick; the correction (Holm unless chosen otherwise) runs over whatever is picked.
        rules = [(RULE_PROMPT, "Pick pairs by a rule")]
        if len(names) > 1:
            rules.append((RULE_EVERY, "Every pair"))
            rules += [(RULE_AGAINST + name, f"Each against {shown}") for name, shown in options]
            rules.append((RULE_NONE, "No pairs"))
        self.rules = SelectButton(t, rules, current=RULE_PROMPT)
        self.rules.setToolTip("Set the pairs in one click. A pair is two conditions, never two replicates")
        self.rules.changed.connect(self._apply_rule)
        lay.addWidget(self.rules)
        # THE CHECKLIST: every pair, in the chart's order; a tick compares it. No pair is offered twice or with itself.
        self._pair_lay = QVBoxLayout()
        self._pair_lay.setContentsMargins(0, 0, 0, 0)
        self._pair_lay.setSpacing(0)
        self.checks = {}
        for i, (a, shown_a) in enumerate(options):
            for b, shown_b in options[i + 1:]:
                row = CheckRow(t, f"{a}{b}", f"{shown_a} vs {shown_b}", on=False)
                row.setToolTip(f"Compare {shown_a} with {shown_b}")
                row.toggled.connect(lambda _k, on, a=a, b=b: self._toggle_pair(a, b, on))
                self._pair_lay.addWidget(row)
                self.checks[rail_view.pair_key(a, b)] = row
        lay.addLayout(self._pair_lay)
        self._shown = dict(options)
        self._names = names
        self._lay.insertWidget(1, body)
        self._body = body

    def _show(self, choice) -> None:
        from ... import significance as sig

        self.choice = choice
        if self.eye is not None and self.eye._on != choice.on:
            self.eye.set_on(choice.on)
        if self.test is not None and choice.test in self._tests:
            if self.test.current != choice.test:
                self.test.set_current(choice.test)
            self.test.setToolTip(sig.TEST_TIPS[choice.test])
        if self.margin_row is not None:
            self.margin_row.setVisible(choice.test in sig.MARGIN_TESTS)
            self.margin_words.setText("Differ by more than ±" if choice.test == "met" else "Same within ±")
            if not self.margin.hasFocus():
                self.margin.setText(f"{choice.margin:g}")
        if self.population is not None and choice.population in self._pops:
            index = self._pops.index(choice.population)
            if self.population.current() != index:
                self.population.blockSignals(True)
                self.population.select(index)
                self.population.blockSignals(False)
        if self.label is not None and choice.label in sig.LABELS:
            index = sig.LABELS.index(choice.label)
            if self.label.current() != index:
                self.label.blockSignals(True)
                self.label.select(index)
                self.label.blockSignals(False)
        if self.stars is not None and not self.stars.hasFocus():
            self.stars.setText(", ".join(f"{v:g}" for v in choice.stars))
        chosen = getattr(choice, "correction", sig.DEFAULT_CORRECTION)
        if self.correction is not None and chosen in sig.CORRECTIONS:
            if self.correction.current != chosen:
                self.correction.set_current(chosen)
            self.correction.setToolTip(sig.CORRECTION_TIPS[chosen])
        picked = {rail_view.pair_key(a, b) for a, b in choice.pairs}
        for key, row in self.checks.items():
            row.set_on(key in picked)

    def show_lines(self, lines, why: str = "") -> None:
        text = why or "\n".join(lines or [])
        if not text and self.choice is not None and not self.choice.pairs:
            text = "Tick the pairs to compare, or pick a rule."
        self.lines.setText(text)

    def _emit(self, **changes) -> None:
        from dataclasses import replace

        if self.choice is None:
            return
        self.choice = replace(self.choice, **changes)
        self.changed.emit(self.choice)

    def _stars_typed(self) -> None:
        from ... import significance as sig

        if self.choice is None or self.stars is None:
            return
        try:
            stars = sig.parse_stars(self.stars.text())
        except ValueError as exc:
            self.lines.setText(str(exc))
            return
        if stars != tuple(self.choice.stars):
            self._emit(stars=stars)

    def _margin_typed(self) -> None:
        if self.choice is None or self.margin is None:
            return
        try:
            margin = float(self.margin.text().replace(",", ".").strip())
        except ValueError:
            self.lines.setText("the margin is a number of percentage points, e.g. 5")
            return
        if not 0 < margin < 100:
            self.lines.setText("the margin is above 0 and below 100 percentage points")
            return
        if margin != self.choice.margin:
            self._emit(margin=margin)

    def _set_pairs(self, pairs) -> None:
        # A pair picked is a pair asked to be seen: brackets on, unless switched off since.
        if self.choice is None:
            return
        self._emit(pairs=tuple(pairs), on=True if not self.choice.pairs and pairs else self.choice.on)

    def _toggle_pair(self, a: str, b: str, on: bool) -> None:
        if self.choice is not None:
            self._set_pairs(rail_view.pairs_toggled(self.choice.pairs, a, b, on))

    def _apply_rule(self, key: str) -> None:
        """A shortcut: the pairs become what the rule says. The select springs back to its prompt."""
        self.rules.set_current(RULE_PROMPT)
        if self.choice is None or key == RULE_PROMPT:
            return
        if key == RULE_EVERY:
            pairs = rail_view.pairs_every(self._names)
        elif key == RULE_NONE:
            pairs = ()
        else:
            pairs = rail_view.pairs_against(self._names, key[len(RULE_AGAINST):])
        self._set_pairs(pairs)


class SignificanceBody(QWidget):
    """The Significance tool's popover (4 Nov 2026, rail review): the `SignificanceBlock`, on its own button, no
    longer under Visibility. Shown on the sections whose bubble chart can carry brackets
    (`significance.SECTION_KINDS`), configured by the window (`set_significance`), which keeps each section's choice;
    the tool is dimmed where the page draws no such chart (`has_content`). Nothing here computes."""

    #: The block's choice (`significance.ViewChoice`), routed by the window to the section on screen.
    significance_changed = Signal(object)
    #: Shown or hidden: the rail dims the tool and writes its count.
    content_changed = Signal()

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.setStyleSheet("background:transparent;")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.significance = SignificanceBlock(t)
        self.significance.changed.connect(self.significance_changed.emit)
        self.significance.hide()
        outer.addWidget(self.significance)
        outer.addStretch(1)

    def has_content(self) -> bool:
        return not self.significance.isHidden()

    def set_significance(self, conditions, populations, mixed, choice, lines=(), why: str = "",
                         paired: bool = False, kind: str = "") -> None:
        """Show the block for the current section, configured (see `SignificanceBlock.configure`), with its lines."""
        self.significance.configure(conditions, populations, mixed, choice, paired, kind)
        self.significance.show_lines(lines, why)
        self.significance.show()
        self.content_changed.emit()

    def hide_significance(self) -> None:
        was_hidden = self.significance.isHidden()
        self.significance.hide()
        if not was_hidden:
            self.content_changed.emit()

    def badge(self) -> str | None:
        """The count of pairs set, for the rail (`rail_view.pair_badge`); None while there is no chart."""
        choice = self.significance.choice
        return rail_view.pair_badge(choice.pairs) if self.has_content() and choice is not None else None


class VisibilityBody(QWidget):
    """Visibility: which of this section's figures are drawn, and -- where
    the section has one -- which colormap they are drawn in. (The significance brackets are the Significance tool's
    since 4 Nov 2026: `SignificanceBody`.)

    **The colormap is here rather than in a Visualization tool on purpose.**
    This tool answers "what does this figure look like"; the Visualization tool
    that sat beside it (since removed, see the module docstring) was meant to
    answer "what is drawn from the data" (the D correction, the
    mobility threshold, the reference condition). A ramp changes no number,
    so it belongs on this side of that line -- and the controls that DO
    change the map's numbers (`n_min`, fold, assignment, step position) sit
    in the map's own Run dialog for the same reason, not here.
    """

    toggled = Signal(str, bool)
    #: The section's colormap key (`COLORMAPS`' own first element), emitted
    #: on every pick. `window.py` routes it by CURRENT section, exactly as
    #: it routes `toggled`.
    colormap_changed = Signal(str)
    #: The grid's rendering key ("bins" / "smooth"), emitted on every pick and
    #: routed by CURRENT section, like `colormap_changed`.
    rendering_changed = Signal(str)
    #: "Custom…" was pressed: the window opens the ramp editor (`ramp_dialog`).
    custom_ramp_requested = Signal()
    #: The grid's colour scale key (`SCALES`), routed by CURRENT section like the others.
    scale_changed = Signal(str)
    #: What the popover holds may have changed (a section, a figure switched off): the rail dims the tool where it
    #: holds nothing (`has_content`) and marks it while a figure is hidden.
    content_changed = Signal()

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.setStyleSheet("background:transparent;")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(rem(0.5))
        #: Hidden on a section that lists no figures (SQD, QC, MSD, ...): a heading with nothing under it read as
        #: something missing (3 Nov 2026, reported).
        self._figures_caption = _caption(t, "Figures")
        outer.addWidget(self._figures_caption)
        self._rows = QVBoxLayout()
        self._rows.setContentsMargins(0, 0, 0, 0)
        self._rows.setSpacing(0)
        outer.addLayout(self._rows)
        #: Said under the figures' caption where there are no rows YET (the Spatial Map before a map is drawn).
        self._figures_note = label("", ty.ui("xs"), t.muted_foreground)
        self._figures_note.setWordWrap(True)
        self._figures_note.setMaximumWidth(rem(18))
        self._figures_note.hide()
        outer.addWidget(self._figures_note)
        #: The Colormap block, built ONCE and rebuilt only when the section
        #: actually offers a different ramp list -- a `Segmented` rebuilds
        #: its pills on every `select()`, and an ancestor `Surface` keeps a
        #: shadow cache pointing at them (see `Segmented`'s own
        #: `_shadow_boundary` note), so churning this widget per section
        #: switch buys nothing and risks that cache.
        self._cmap_block = QWidget()
        self._cmap_block.setStyleSheet("background:transparent;")
        self._cmap_lay = QVBoxLayout(self._cmap_block)
        self._cmap_lay.setContentsMargins(0, 0, 0, 0)
        self._cmap_lay.setSpacing(rem(0.5))
        self._cmap_lay.addWidget(_caption(t, "Colormap"))
        self._cmap_block.hide()
        outer.addWidget(self._cmap_block)
        #: The Rendering block: built once, shown only for the sections in
        #: `RENDERING_SECTIONS`.
        self._render_block = QWidget()
        self._render_block.setStyleSheet("background:transparent;")
        render_lay = QVBoxLayout(self._render_block)
        render_lay.setContentsMargins(0, 0, 0, 0)
        render_lay.setSpacing(rem(0.5))
        render_lay.addWidget(_caption(t, "Rendering"))
        #: The pills now shown, ``((key, text, tip), ...)``: a method without tracks has no
        #: Tracks pill (`renderings_for`), so the set is rebuilt when the section changes.
        self._renderings = renderings_for(None)
        self._render_lay = render_lay
        self.rendering = self._rendering_pills(self._renderings, 0)
        render_lay.addWidget(self.rendering)
        render_lay.addWidget(_caption(t, "Colour scale"))
        self.scale = Segmented(t, [text for _, text in SCALES], current=0, size="small",
                               tips=list(SCALE_TIPS))
        self.scale.changed.connect(lambda i: self.scale_changed.emit(SCALES[i][0]))
        render_lay.addWidget(self.scale)
        self._render_block.hide()
        outer.addWidget(self._render_block)
        # spare height (a popover held taller than its body) goes under the blocks, never between their rows
        outer.addStretch(1)
        #: The keys the pills stand for, in pill order -- `Segmented` is
        #: index-based, and an index is not a name.
        self._cmap_keys: tuple[str, ...] = ()
        #: The ramp the maps are painted with now (a custom one's full key included).
        self._cmap_current = "jet"
        self.colormap: Segmented | None = None
        self._section: str | None = None
        self.rows: dict[str, EyeRow] = {}
        self.set_section("sqd")

    def _set_colormaps(self, section: str) -> None:
        """Show, rebuild or hide the Colormap block for `section`."""
        options = COLORMAPS.get(section, ())
        keys = tuple(key for key, _ in options)
        if keys == self._cmap_keys:
            self._cmap_block.setVisible(bool(keys))
            return
        if self.colormap is not None:
            self._cmap_lay.removeWidget(self.colormap)
            discard(self.colormap)
            self.colormap = None
        self._cmap_keys = keys
        if not keys:
            self._cmap_block.hide()
            return
        self.colormap = Segmented(self.t, [text for _, text in options],
                                  current=0, size="small")
        self.colormap.changed.connect(self._on_colormap_pill)
        # "Custom…" already raised (a custom ramp in use) emits no `changed` on a click, so
        # the editor would open only after picking another ramp first -- reported.
        self.colormap.pressed.connect(self._on_colormap_press)
        self._cmap_lay.addWidget(self.colormap)
        self._cmap_block.show()

    def _on_colormap_pill(self, index: int) -> None:
        key = self._cmap_keys[index]
        if key == CUSTOM_PILL:
            # THE PILL OPENS THE EDITOR; it does not change the map by itself. It springs
            # back to the ramp on screen until the editor's "Save and use" names one.
            self.set_colormap(self._cmap_current)
            self.custom_ramp_requested.emit()
            return
        self._cmap_current = key
        self.colormap_changed.emit(key)

    def _on_colormap_press(self, index: int) -> None:
        if (self._cmap_keys[index] == CUSTOM_PILL and self.colormap is not None
                and self.colormap.current() == index):
            self.custom_ramp_requested.emit()

    def set_colormap(self, key: str) -> None:
        """Point the pills at `key` WITHOUT re-emitting -- used to restore a
        section's own remembered ramp on re-entry, the same reason
        `set_section` takes `states`. A custom ramp's key lights "Custom…"."""
        pill = CUSTOM_PILL if str(key).startswith(CUSTOM_KEY_PREFIX) else key
        if self.colormap is None or pill not in self._cmap_keys:
            return
        self._cmap_current = key
        index = self._cmap_keys.index(pill)
        if index != self.colormap.current():
            self.colormap.blockSignals(True)
            self.colormap.select(index)
            self.colormap.blockSignals(False)

    def _rendering_pills(self, offered: tuple, current: int):
        pills = Segmented(self.t, [text for _k, text, _tip in offered], current=current,
                          size="small", tips=[tip for _k, _text, tip in offered])
        pills.changed.connect(lambda i: self.rendering_changed.emit(self._renderings[i][0]))
        return pills

    def rendering_keys(self) -> tuple:
        """The renderings the current section offers, in pill order."""
        return tuple(key for key, _text, _tip in self._renderings)

    def _set_renderings(self, section: str | None) -> None:
        """Offer ``section``'s renderings: the pills are rebuilt only when the set changes,
        keeping the current choice where it is still offered (else the first, Bins). Does
        not emit: the window moves a section off a rendering it no longer offers."""
        offered = renderings_for(section)
        if offered == self._renderings:
            return
        keep = self._renderings[self.rendering.current()][0]
        keys = [key for key, _text, _tip in offered]
        self._renderings = offered
        pills = self._rendering_pills(offered, keys.index(keep) if keep in keys else 0)
        # into the layout BEFORE the old one goes (a widget shown with no parent is a
        # window of its own -- `primitives.discard`)
        self._render_lay.insertWidget(self._render_lay.indexOf(self.rendering), pills)
        discard(self.rendering)
        self.rendering = pills

    def set_rendering(self, key: str) -> None:
        """Point the Rendering pills at `key` WITHOUT re-emitting, so a
        section's own remembered choice is restored on re-entry."""
        keys = list(self.rendering_keys())
        if key not in keys:
            return
        index = keys.index(key)
        if index != self.rendering.current():
            self.rendering.blockSignals(True)
            self.rendering.select(index)
            self.rendering.blockSignals(False)

    def set_scale(self, key: str) -> None:
        """Point the Colour scale pills at `key` WITHOUT re-emitting (restore on re-entry)."""
        keys = [k for k, _ in SCALES]
        if key not in keys:
            return
        index = keys.index(key)
        if index != self.scale.current():
            self.scale.blockSignals(True)
            self.scale.select(index)
            self.scale.blockSignals(False)

    @property
    def section(self) -> str | None:
        return self._section

    def set_section(self, section: str, states: dict[str, bool] | None = None, names=None) -> None:
        """``names``: the rows, in order, where the section's charts are not a fixed list (the Spatial Map's columns,
        `SpatialMapMatrix.panel_keys`: All jumps, Middle, a split by hand's ranges -- 3 Nov 2026, reported: "visibility
        in maps has short and long jumps to toggle visibility off but not middle and all"); else `FIGURES[section]`.
        A change of names rebuilds the rows even within one section.

        `states` gives each row's initial on/off (default `True` for a
        name it doesn't mention) -- callers whose section can be revisited
        with a persisted toggle state (the Apparent-D detail page's own
        two sections) pass their own remembered state back in, since a
        bare rebuild would otherwise reset every row to visible every
        time. Applied even on the same-section no-op below, since a caller
        may re-enter the identical section with different remembered
        state (switching away and back through a DIFFERENT section in
        between already changes `self._section`, but a caller re-asserting
        state without an intervening section change should still work)."""
        # A section or page absent from `FIGURES` lists nothing (a page of SQD other than Populations, 4 Nov 2026).
        spec = FIGURES.get(section, ()) if names is None else tuple((str(n), False) for n in names)
        if section == self._section and tuple(self.rows) == tuple(name for name, _nested in spec):
            if states is not None:
                for name, on in states.items():
                    row = self.rows.get(name)
                    if row is not None:
                        row.set_on(on)
                self.content_changed.emit()
            return
        self._section = section
        self._set_colormaps(section)
        self._set_renderings(section)
        self._render_block.setVisible(section in RENDERING_SECTIONS)
        while self._rows.count():
            item = self._rows.takeAt(0)
            w = item.widget()
            if w is not None:
                discard(w)
        self.rows = {}
        self._figures_note.hide()               # a note belongs to the section that set it (`set_figures_note`)
        self._figures_caption.setVisible(bool(spec))
        for name, nested in spec:
            on = True if states is None else states.get(name, True)
            r = EyeRow(self.t, name, name, on=on, nested=nested)
            r.toggled.connect(self.toggled.emit)
            r.toggled.connect(lambda _k, _on: self.content_changed.emit())
            self._rows.addWidget(r)
            self.rows[name] = r
        self.content_changed.emit()

    def set_figures_note(self, text: str) -> None:
        """A line in place of the figure rows while the section has none yet ("Run the map to choose which maps it
        shows."); empty hides it. The Figures caption stands over it."""
        self._figures_note.setText(text)
        self._figures_note.setVisible(bool(text))
        self._figures_caption.setVisible(bool(self.rows) or bool(text))

    def has_content(self) -> bool:
        """Does the popover hold anything for the section it describes: a figure row, a colormap, a rendering or
        or a scale. Read by `Toolbar`, which dims the tool where it does not."""
        return bool(self.rows) or not (self._cmap_block.isHidden() and self._render_block.isHidden())

    def hidden_count(self) -> int:
        """How many of the rows are switched off now: the rail marks the tool with a dot while any is."""
        return rail_view.hidden_count({name: row._on for name, row in self.rows.items()})


class _PaletteSelect(SelectButton):
    """The palette picker: the same list Confirm Project shows -- SUPPORTED palettes
    (a colour for every condition) to choose from, UNSUPPORTED ones shown disabled
    with why -- opening and closing without animation (asked for)."""

    def __init__(self, t, body: "ConditionsBody", parent=None):
        super().__init__(t, PALETTES, current=body.model.palette, animated=False,
                         parent=parent)
        self._body = body

    def _label_for(self, key: str) -> str:
        """A palette's name: the user's own are not in `PALETTES` (made after this module was read)."""
        palette = hcl.PALETTES.get(key)
        return palette.label if palette is not None else super()._label_for(key)

    def mouseReleaseEvent(self, e):
        if e.button() != Qt.LeftButton or retriggered(self):
            return
        t, model = self.t, self._body.model
        menu = Panel(t, self.window(), root=ROOT_PX, animated=False)
        menu.set_trigger(self)
        n = len(model.conditions)
        for heading, keys in (("Supported", model.supported()),
                              ("Unsupported", model.unsupported())):
            if not keys:
                continue
            menu_section(t, menu, heading, root=ROOT_PX)
            for key in keys:
                pal = hcl.PALETTES[key]
                ok = heading == "Supported"
                row = PaletteRow(
                    t, pal.label, model.pool(dark=self._body.dark, palette=key)[:5],
                    current=key == model.palette, disabled=not ok,
                    note="" if ok else f"{len(pal.order)} colours, {n} conditions")
                if ok:
                    row.triggered.connect(
                        lambda k=key: (self._pick(k), menu.close_chain()))
                menu.add_widget(row)
        # YOUR OWN (6 Nov 2026): make one, or edit one made before (`PaletteMaker`)
        menu_row(t, menu, "Make a palette\u2026", glyph="plus", root=ROOT_PX,
                 on_click=lambda: QTimer.singleShot(0, self._body.make_palette))
        menu.set_minimum_visible_width(self.width())
        menu.popup(self.mapToGlobal(QPoint(0, self.height() + 4)))


#: A press on a row's grip becomes a drag after this many pixels (a click is not a drag).
ORDER_DRAG_SLOP = 4
#: The bar under the rows while the order shown is not the one in use (6 Nov 2026).
ORDER_PENDING_TEXT = "New order, not applied yet"
ORDER_APPLY_TEXT = "Apply order"


class _OrderGrip(QWidget):
    """A row's grip (six dots): drag it up or down to put the condition elsewhere in the order SHOWN; nothing is
    reordered in the workspace until Apply order (`ConditionsBody`)."""

    dragging = Signal(str, int)         # key, the pointer's y in the rows box's coordinates
    dropped = Signal(str)               # key

    def __init__(self, t, key: str, box: QWidget, parent=None):
        super().__init__(parent)
        self.key, self._box = key, box
        self._press: QPoint | None = None
        self._drag = False
        self.setFixedSize(rem(1), rem(2))
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        glyph = icon("grip", rem(0.875), t.muted_foreground)
        glyph.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        lay.addWidget(glyph, 0, Qt.AlignCenter)
        self.setCursor(Qt.OpenHandCursor)
        self.setToolTip("Drag up or down to move this condition; Apply order puts it to use")

    def mousePressEvent(self, e):                  # noqa: N802 - Qt's name
        if e.button() == Qt.LeftButton:
            self._press, self._drag = e.pos(), False
            self.setCursor(Qt.ClosedHandCursor)
            e.accept()
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):                   # noqa: N802 - Qt's name
        if self._press is not None and (e.buttons() & Qt.LeftButton):
            if not self._drag and abs(e.pos().y() - self._press.y()) >= ORDER_DRAG_SLOP:
                self._drag = True
            if self._drag:
                self.dragging.emit(self.key, self.mapTo(self._box, e.pos()).y())
            e.accept()
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):                # noqa: N802 - Qt's name
        was_drag = self._drag
        self._press, self._drag = None, False
        self.setCursor(Qt.OpenHandCursor)
        if was_drag:
            self.dropped.emit(self.key)
            e.accept()
            return
        super().mouseReleaseEvent(e)


class _OrderRows(QWidget):
    """The Conditions tool's rows, and while a row is dragged the line where it would land."""

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.setStyleSheet("background:transparent;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.target: int | None = None

    def _holders(self) -> list:
        lay = self.layout()
        return [lay.itemAt(i).widget() for i in range(lay.count()) if lay.itemAt(i).widget() is not None]

    def index_at(self, y: int) -> int:
        """Where a row dropped at ``y`` lands: the number of rows whose middle is above it (0 to the row count)."""
        return sum(1 for w in self._holders() if w.geometry().center().y() < y)

    def set_target(self, index: int | None) -> None:
        if index != self.target:
            self.target = index
            self.update()

    def paintEvent(self, e):                       # noqa: N802 - Qt's name
        super().paintEvent(e)
        if self.target is None:
            return
        holders = self._holders()
        if not holders:
            return
        y = (holders[self.target].geometry().top() if self.target < len(holders)
             else holders[-1].geometry().bottom() + 1)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(self.t.primary))
        p.drawRoundedRect(QRectF(0, max(0, y - 1), self.width(), 2), 1, 1)
        p.end()


class ConditionsBody(QWidget):
    """Conditions: the workspace's samples, and their colours.

    Workspace-WIDE while living in a per-section tool rail, which is why the
    scope line under the title says so: a colour or an order set here changes
    every section's figures, not just this one's.

    **NO EYES (4 Nov 2026).** Each row carried an eye that toggled its own glyph and nothing else:
    `toggled` was connected to no handler in the window, so "hiding" a condition hid nothing anywhere. The rows are
    swatch rows now (`EyeRow(toggles=False)`): the colour leads the name and a click on the row picks it. Hiding a
    condition is each chart key's own eye (the bubbles, MSD, HMM), which does hide it.

    **THE COLOURS ARE `condition_colours.ConditionColours`**, the SAME model
    Confirm Project edits and hands on: a palette and one slot per condition,
    read through `hcl.slots` (fitted to the page, in declared order). It once
    read `hcl.series`, which is sorted by lightness and RAISES past the
    palette's size -- so it disagreed with Confirm Project on the same palette
    and could not open at all with eight conditions on Okabe-Ito. Clicking a
    condition's dot picks its colour (taking a held one swaps); the palette
    picker resets every condition to its own slot. Every change emits
    `colours_changed` and is remembered for the project.
    """

    #: The palette or any condition's colour changed. The window re-colours
    #: every section from `colour_for`.
    colours_changed = Signal()
    #: The list of conditions (or their order) was set: the rail's icon shows the first colours (`Toolbar`).
    conditions_set = Signal()
    #: Kept for older callers: the palette key, after a palette change.
    palette_changed = Signal(str)
    #: The nesting chosen (`factor_layout`): a tuple of factor names, outer first.
    nesting_changed = Signal(object)
    #: THE ORDER THE CONDITIONS ARE DRAWN IN (`condition_order`, 3 Nov 2026). STAGED since 6 Nov 2026 (asked: "clicking
    #: can take lot of time when software is not really responsive. maybe we can do it drag and drop and make an
    #: accept or cancel button, only if accept (or ok) button is hit , it gets reordered"): a row's arrows and its grip
    #: (drag) rearrange only the list shown here (`condition_order.move` / `arranged`, a folder design's groups kept
    #: together); a bar under the rows offers Cancel and Apply order, and only Apply asks the window for the order
    #: (the keys, in order). The window follows it everywhere and hands the list back (`set_conditions`).
    order_requested = Signal(object)

    def __init__(self, t, project: str = "", conditions=CONDITIONS, parent=None, *,
                 colours: ConditionColours | None = None,
                 colour_key: str | None = None):
        super().__init__(parent)
        self.t = t
        self.dark = bool(getattr(t, "_dark", False))
        self._conditions = tuple(conditions)
        self.colour_key = colour_key
        self.model = self._fit_model(colours)
        self.setStyleSheet("background:transparent;")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(rem(0.5))
        outer.addWidget(_title(t, "Conditions"))
        self.scope = label(f"{project} · all sections" if project else "all sections",
                           ty.ui("xs", px=text_px(0.75)), t.muted_foreground)
        outer.addWidget(self.scope)
        # GROUP BY (folder designs, 24 Oct 2026): the nesting of the project's factors, shown only when it has two
        # or more (`set_nesting`). Above the palette: it decides the order every row below is in.
        self._nesting_host = QVBoxLayout()
        self._nesting_host.setContentsMargins(0, 0, 0, 0)
        self._nesting_host.setSpacing(rem(0.375))
        outer.addLayout(self._nesting_host)
        self.nesting = None
        self._orders: list = []
        #: COLOUR BY (25 Oct 2026, asked: "same colour different shade, user can define"): the factors and the
        #: nesting the colouring follows (`set_factor_colouring`), and its select
        self._factors: dict = {}
        self._factor_names: tuple = ()
        self._nest: tuple = ()
        self.colour_by = None
        # The PALETTE sits above the rows that draw from it: every dot under
        # it can only be one of this palette's colours.
        self.palette = _PaletteSelect(t, self)
        self.palette.changed.connect(self.set_palette)
        outer.addWidget(self.palette)
        self.rows_box = _OrderRows(t)
        self._rows = self.rows_box.layout()
        outer.addWidget(self.rows_box)
        #: The order shown while it differs from the one in use (None: the one in use), and the bar that applies it
        self._staged: list | None = None
        self.pending = QWidget()
        self.pending.setStyleSheet("background:transparent;")
        bar = QHBoxLayout(self.pending)
        bar.setContentsMargins(0, rem(0.375), 0, 0)
        bar.setSpacing(rem(0.375))
        self.pending_note = label(ORDER_PENDING_TEXT, ty.ui("xs", px=text_px(0.75)), t.muted_foreground)
        bar.addWidget(self.pending_note, 1)
        self.cancel_order = Button(t, "Cancel", variant="ghost", size="sm")
        self.cancel_order.clicked.connect(self.cancel_staged)
        bar.addWidget(self.cancel_order, 0, Qt.AlignVCenter)
        self.apply_order = Button(t, ORDER_APPLY_TEXT, variant="default", size="sm")
        self.apply_order.clicked.connect(self.apply_staged)
        bar.addWidget(self.apply_order, 0, Qt.AlignVCenter)
        outer.addWidget(self.pending)
        self.pending.hide()
        # THE SAFEGUARD (6 Nov 2026): more conditions than the palette's colours, or two hard to tell apart, said
        # where the colours are chosen (`condition_colours.crowding_text`)
        from ..widgets.primitives import fit_label

        self.crowding = fit_label("", ty.ui("xs", px=text_px(0.75)), t.muted_foreground)
        outer.addWidget(self.crowding)
        self.rows: dict[str, EyeRow] = {}
        self._build_rows()
        self._sync_crowding()

    def _keys(self) -> list[str]:
        return [key for key, _name in self._conditions]

    def _sync_crowding(self) -> None:
        from ...condition_colours import crowding_text

        crowding = getattr(self, "crowding", None)
        if crowding is None:
            return
        text = crowding_text(self.model, dict(self._conditions), dark=self.dark)
        crowding.setText(text or "")
        crowding.setVisible(bool(text))

    def shown_order(self) -> list[str]:
        """The order the rows show: the staged one while there is one, else the one in use."""
        return list(self._staged) if self._staged is not None else self._keys()

    def _fit_model(self, colours) -> ConditionColours:
        """``colours`` where it covers exactly these conditions; else the one
        remembered for the project; else the default."""
        keys = self._keys()
        if colours is not None and sorted(colours.conditions) == sorted(keys):
            return ConditionColours.from_json(keys, colours.to_json())
        return condition_colours.recall(self.colour_key, keys)

    def _build_rows(self) -> None:
        while self._rows.count():
            item = self._rows.takeAt(0)
            w = item.widget()
            if w is not None:
                discard(w)
        self.rows = {}
        #: Each row's arrows (asked 3 Nov 2026: "set order ... change order of how samples are arranged. not only sqd
        #: bubbles, but everywhere possible"): earlier or later in every chart, list and export of the workspace --
        #: once Apply order is pressed (6 Nov 2026); and its grip, dragged
        self.arrows: dict[str, tuple] = {}
        self.grips: dict[str, _OrderGrip] = {}
        names = dict(self._conditions)
        keys = self.shown_order()
        for i, key in enumerate(keys):
            name = names[key]
            holder = QWidget()
            holder.setStyleSheet("background:transparent;")
            lay = QHBoxLayout(holder)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(2)
            grip = _OrderGrip(self.t, key, self.rows_box)
            grip.dragging.connect(self._on_drag)
            grip.dropped.connect(self._on_drop)
            lay.addWidget(grip, 0, Qt.AlignVCenter)
            self.grips[key] = grip
            # A SWATCH ROW, no eye: the eye here hid nothing (no handler in the window; 4 Nov 2026). A click on
            # the row picks the condition's colour.
            r = EyeRow(self.t, key, name, dot=self.model.colour(key, dark=self.dark), toggles=False)
            r.setToolTip(f"{name}: click to change its colour")
            r.dot_clicked.connect(self._pick_colour)
            lay.addWidget(r, 1)
            # On the POPOVER's colour: on the page's they were black chips on the dark theme's grey card. The ends'
            # arrows (the first row's up, the last's down) are DIMMED, not only disabled: they looked as live as the
            # others and did nothing (4 Nov 2026, rail review).
            up = ToolButton(self.t, "up", "chevron-up", size=rem(1.5), rest=self.t.popover)
            down = ToolButton(self.t, "down", "chevron-down", size=rem(1.5), rest=self.t.popover)
            up.setToolTip("Earlier: further left in every chart, higher in every list and export (Apply order)")
            down.setToolTip("Later: further right in every chart, lower in every list and export (Apply order)")
            for button in (up, down):
                lay.addWidget(button, 0, Qt.AlignVCenter)
            for button, live in ((up, i > 0), (down, i < len(keys) - 1)):
                button.setEnabled(live)
                button.set_available(live)
            up.clicked.connect(lambda _k, k=key: self.move(k, -1))
            down.clicked.connect(lambda _k, k=key: self.move(k, 1))
            self._rows.addWidget(holder)
            self.rows[key] = r
            self.arrows[key] = (up, down)
        self.rows_box.set_target(None)
        self.pending.setVisible(self._staged is not None)

    # -- the order: staged here, applied by the window -------------------------------------------------------- #
    def _stage(self, order) -> None:
        """Show ``order``; staged while it differs from the order in use, which nothing here changes."""
        order = list(order)
        self._staged = None if order == self._keys() else order
        self._build_rows()

    def move(self, key: str, step: int) -> None:
        """``key`` one place earlier (``step`` -1) or later (+1) in the order SHOWN (`condition_order.move`: a
        folder design's groups kept together); Apply order puts it to use."""
        if key in self._keys():
            self._stage(condition_order.move(self.shown_order(), key, int(step), self._factors, self._nest))

    def drop(self, key: str, index: int) -> None:
        """``key`` dragged to place ``index`` (0 to the row count: before that row) of the order shown, then through
        the folder design's nesting (`condition_order.arranged`, which keeps its groups together)."""
        order = self.shown_order()
        if key not in order:
            return
        i = order.index(key)
        j = max(0, min(len(order), int(index)))
        order.insert(j - 1 if j > i else j, order.pop(i))
        self._stage(condition_order.arranged(order, None, self._factors, self._nest))

    def _on_drag(self, key: str, y: int) -> None:
        self.rows_box.set_target(self.rows_box.index_at(y))

    def _on_drop(self, key: str) -> None:
        target = self.rows_box.target
        self.rows_box.set_target(None)
        if target is not None:
            # after the grip's own release handler: the rebuild discards the grip being released
            QTimer.singleShot(0, lambda k=key, i=target: self.drop(k, i))

    def cancel_staged(self) -> None:
        """Back to the order in use."""
        self._stage(self._keys())

    def apply_staged(self) -> None:
        """Ask the window for the order shown (`order_requested`); it hands the list back (`set_conditions`)."""
        if self._staged is None:
            return
        order, self._staged = list(self._staged), None
        self.order_requested.emit(order)
        self._build_rows()

    def set_conditions(self, conditions) -> None:
        """Re-point at a freshly loaded project's own condition names (or the order the window now uses): an order
        staged here is dropped."""
        self._conditions = tuple(conditions)
        self._staged = None
        self.model = self._fit_model(self.model)
        self._apply_by(self.model.by)
        self.palette.set_current(self.model.palette)
        self._build_rows()
        self.conditions_set.emit()

    # -- colour by a factor (folder designs) ------------------------------ #
    def set_factor_colouring(self, factors: dict, names, nesting) -> None:
        """Offer "Colour by" for a project with two or more factors: one hue per value of a factor, each condition
        a shade of it (`ConditionColours.colour_by`). Until the user picks, it follows the nesting's INNER factor
        (the one compared within each group); a pick is remembered for the project."""
        self._factors, self._factor_names, self._nest = dict(factors or {}), tuple(names or ()), tuple(nesting or ())
        if self.colour_by is not None:
            discard(self.colour_by)
            self.colour_by = None
        if len(self._factor_names) < 2:
            return
        by = self.model.by if self.model.by_chosen else (self._nest or self._factor_names)[-1]
        self._apply_by(by)
        options = [("", "Condition")] + [(n, n) for n in self._factor_names]
        self.colour_by = SelectButton(self.t, options, current=by)
        self.colour_by.setToolTip("One colour per value of this factor, each condition a shade of it; or one colour "
                                  "per condition. Click a dot to change its colour.")
        self.colour_by.changed.connect(self._choose_by)
        self._nesting_host.addWidget(_caption(self.t, "Colour by"))
        self._nesting_host.addWidget(self.colour_by)
        self._build_rows()

    def follow_nesting(self, nesting) -> None:
        """The nesting changed: a colouring nobody chose follows its new inner factor."""
        self._nest = tuple(nesting or ())
        if not self._factor_names:
            return
        by = self.model.by if self.model.by_chosen else self._nest[-1]
        self._apply_by(by)
        if self.colour_by is not None:
            self.colour_by.set_current(by)
        self._changed()

    def _apply_by(self, by: str) -> None:
        if not self._factors:
            return
        shade = tuple(n for n in (self._nest or self._factor_names) if n != by) if by else ()
        self.model.colour_by(self._factors, by, shade)

    def _choose_by(self, by: str) -> None:
        self.model.by_chosen = True
        self._apply_by(by)
        self._changed()

    def set_nesting(self, orders, current) -> None:
        """Offer the nestings ``orders`` (`factor_layout.orders`, tuples of factor names), ``current`` raised; none
        (fewer than two factors) shows nothing. A choice emits `nesting_changed`."""
        from ...factor_layout import order_words

        while self._nesting_host.count():
            item = self._nesting_host.takeAt(0)
            if item.widget() is not None:
                discard(item.widget())
        self._orders = [tuple(o) for o in orders]
        self.nesting = None
        if len(self._orders) < 2:
            return
        current = tuple(current) if tuple(current) in self._orders else self._orders[0]
        self._nesting_host.addWidget(_caption(self.t, "Group by"))
        words = [(order_words(o), order_words(o)) for o in self._orders]
        self.nesting = SelectButton(self.t, words, current=order_words(current))
        self.nesting.setToolTip("Which factor groups the columns of every comparison chart (outer first), and the "
                                "order the conditions are listed in. Changes no number.")
        by_words = {order_words(o): o for o in self._orders}
        self.nesting.changed.connect(lambda key: self.nesting_changed.emit(by_words[key]))
        self._nesting_host.addWidget(self.nesting)

    def colour_for(self, key: str) -> str | None:
        """This condition's own colour -- the SAME one its row's dot paints.
        ``None`` if ``key`` is not one of this workspace's conditions.

        The one place any OTHER view (a chart, an export) should read a
        condition's colour from, rather than computing a palette itself.
        """
        return self.model.colour(key, dark=self.dark)

    def colours(self) -> ConditionColours:
        """A copy of the current choice -- what a rebuilt window is handed."""
        return self.model.copy()

    # -- changing -------------------------------------------------------- #
    def set_palette(self, key: str) -> None:
        if self.model.set_palette(key):
            self.palette.set_current(key)
            self._changed()
            self.palette_changed.emit(key)

    def assign(self, key: str, slot: int) -> None:
        if self.model.assign(key, slot):
            self._changed()

    def set_custom(self, key: str, colour: str | None) -> None:
        """``key`` in ``colour`` by hand (None: back to its palette colour); remembered with the project's colours."""
        if self.model.set_custom(key, colour):
            self._changed()

    def custom_colour(self, key: str) -> None:
        """The system colour picker for ``key``; its colour by hand on OK."""
        from qtpy.QtGui import QColor
        from qtpy.QtWidgets import QColorDialog

        chosen = QColorDialog.getColor(QColor(self.model.colour(key, dark=self.dark) or "#3f90da"), self._host_window(),
                                       f"Colour for {dict(self._conditions).get(key, key)}")
        if chosen.isValid():
            self.set_custom(key, chosen.name())

    def _host_window(self):
        """The window to open a dialog over: this tool's while its popover is open, else the workspace window (the tool
        sits in a hidden holder of its own while the popover is closed)."""
        window = self.window()
        if window is self or not window.isVisible():
            from qtpy.QtWidgets import QApplication

            window = next((w for w in QApplication.topLevelWidgets()
                           if w.isVisible() and getattr(w, "toolbar", None) is not None
                           and getattr(w.toolbar, "conditions", None) is self), window)
        return window

    def make_palette(self) -> None:
        """The palette maker (`PaletteMaker`) over the window, starting from the palette in use; a palette saved there
        is chosen when it has a colour for every condition."""
        from ..widgets.palette_maker import PaletteMaker

        window = self._host_window()
        maker = PaletteMaker(self.t, start=self.model.palette, content=window.centralWidget()
                             if hasattr(window, "centralWidget") else None, parent=window)
        maker.saved.connect(self._palette_made)
        maker.closed.connect(maker.deleteLater)
        self.palette_maker = maker
        maker.open()

    def _palette_made(self, key: str) -> None:
        if key in self.model.supported():
            self.set_palette(key)
        else:
            self._changed()

    def _changed(self) -> None:
        self._sync_crowding()
        for key, row in self.rows.items():
            row.set_dot(self.model.colour(key, dark=self.dark))
        condition_colours.remember(self.colour_key, self.model)
        self.colours_changed.emit()

    def _pick_colour(self, key: str, at) -> None:
        """Which colour ``key`` is drawn in -- Confirm Project's own swatch row.
        Taking a colour another condition holds SWAPS the two."""
        if retriggered(at):
            return
        t, model = self.t, self.model
        names = dict(self._conditions)
        pool = model.pool(dark=self.dark)
        order = list(hcl.PALETTES[model.palette].order)
        colour_names = order + [f"colour {len(order) + i + 1}"
                                for i in range(max(0, len(pool) - len(order)))]
        menu = Panel(t, self.window(), root=ROOT_PX, animated=False)
        menu.set_trigger(at)
        value = model.value_of(key)
        if value is not None:
            # COLOURED BY A FACTOR: the pick is the VALUE's hue (every condition of it, each in its shade)
            by_slot = {s: v for v, s in model.value_slots.items()}
            current, owner = model.value_slots[value], value
            holders = [by_slot.get(s) for s in range(len(pool))]
        else:
            current, owner = model.slots[key], names.get(key, key)
            holders = [names.get(model.holder_of(s), model.holder_of(s)) for s in range(len(pool))]
        row = SwatchRow(t, pool, colour_names, current=current, holders=holders, owner=owner)
        row.picked.connect(lambda slot: (self.assign(key, slot), menu.close_chain()))
        menu.add_widget(row)
        # A COLOUR BY HAND (6 Nov 2026, "or to manually adjust some colors"): any colour, over the palette's
        menu_row(t, menu, "Custom colour\u2026", glyph="pencil", root=ROOT_PX,
                 on_click=lambda: QTimer.singleShot(0, lambda: self.custom_colour(key)))
        if key in model.custom:
            menu_row(t, menu, "Use the palette's colour", glyph="undo", root=ROOT_PX,
                     on_click=lambda: self.set_custom(key, None))
        width = row.sizeHint().width()
        menu.popup(QPoint(at.right() - width, at.bottom() + 6))

    # Older name, kept for callers that still connect to it.
    def _repaint_dots(self, palette: str) -> None:
        self.set_palette(palette)


class Toolbar(QWidget):
    """The rail, and the popovers it opens. ONE open surface at a time."""

    #: The mark was pressed. NOT "Run" -- Compute is a dedicated PAGE now
    #: (`window.ComputeConfigPage`, reached via the sidebar breadcrumb),
    #: not a popover this toolbar hosts, so pressing the mark is pure
    #: navigation and this bubbles it up for the window to act on. See
    #: `compute.py`'s module docstring for why it outgrew a popover.
    compute_pressed = Signal()
    #: The section the popovers now describe (after `set_section`): the window configures the
    #: Visibility popover's Significance block for it.
    section_set = Signal(str)

    def __init__(self, t, *, project: str = "", conditions=CONDITIONS, parent=None,
                 colours=None, colour_key: str | None = None):
        super().__init__(parent)
        self.t = t
        self.setStyleSheet("background:transparent;")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.rail = ToolRail(t)
        self.rail.picked.connect(self._pick)
        self.rail.compute_pressed.connect(self.compute_pressed.emit)
        # THE METADATA PANEL is a column of this row, left of the rail -- a panel, not a
        # popover (the UI designer: "I meant a panel, not a tooltip"). Its rail button toggles it.
        from .metadata import MetadataDock
        self.metadata_dock = MetadataDock(t)
        self.metadata_dock.opened.connect(lambda _on: self._sync_rail())
        lay.addWidget(self.metadata_dock)
        lay.addWidget(self.rail)

        self.visibility = VisibilityBody(t)
        self.significance = SignificanceBody(t)
        self.conditions = ConditionsBody(t, project, conditions,
                                         colours=colours, colour_key=colour_key)
        #: The receipt itself, inside the panel (`window._sync_metadata` fills it).
        self.metadata = self.metadata_dock.body
        self._bodies = {"visibility": self.visibility,
                        "significance": self.significance,
                        "conditions": self.conditions}
        self._pop = None
        self._open_key: str | None = None
        #: The section the Significance tool describes (what `set_section` was last given): brackets belong to a PAGE
        #: (`rail_view.rail_section`), and the window reads this, not Visibility's section, to route a choice.
        self.significance_section = ""
        self.visibility.content_changed.connect(self._sync_visibility_tool)
        self.significance.content_changed.connect(self._sync_significance_tool)
        self.conditions.colours_changed.connect(self._sync_conditions_mark)
        self.conditions.conditions_set.connect(self._sync_conditions_mark)
        self._sync_visibility_tool()
        self._sync_significance_tool()
        self._sync_conditions_mark()

    def set_section(self, section: str) -> None:
        """Follow the sidebar. ``section`` may name a page ("sqd.jumps"): Visibility then offers what THAT page
        draws. (The Test button, the Spatial Map's and the HMM's redesigns to try, is gone since 23 Nov 2026: the HMM's
        was retired, the Spatial Map's became its replicate page.)"""
        self.visibility.set_section(section)
        self.significance_section = section
        self.metadata_dock.recheck()            # another page: beside it or over it (`MetadataDock.page_slack`)
        self.section_set.emit(section)
        self._sync_visibility_tool()
        self._sync_significance_tool()

    #: The Visibility tool's hover text, with something to offer and without (4 Nov 2026).
    VISIBILITY_TIP = TOOL_TIPS["visibility"]
    VISIBILITY_EMPTY_TIP = "Visibility: nothing on this page to show or hide"
    SIGNIFICANCE_TIP = rail_view.SIGNIFICANCE_TIP
    SIGNIFICANCE_EMPTY_TIP = rail_view.SIGNIFICANCE_EMPTY_TIP

    def _sync_visibility_tool(self) -> None:
        """The Visibility tool is SHOWN only where its popover holds something (hidden on QC, MSD, HMM, Jump
        distances ...): it opened an empty card there (4 Nov 2026, rail review) and was dimmed; hidden since the same
        night, as Significance and Test are ("a button that does nothing should not be there"). An open popover that
        has just lost its content closes."""
        has = self.visibility.has_content()
        hidden = self.visibility.hidden_count()
        button = self.rail.buttons["visibility"]
        button.set_available(has)
        button.setVisible(has)
        button.set_dot(hidden > 0)
        words = self.VISIBILITY_TIP + (f" ({hidden} hidden)" if hidden else "")
        self.rail.set_tool_tip("visibility", words if has else self.VISIBILITY_EMPTY_TIP)
        if not has and self._open_key == "visibility" and self._pop is not None:
            # its own close: `closed` fires, so the rail and `_pop` follow (a bare `close()` only hides it)
            self._pop.close_chain()

    def _sync_significance_tool(self) -> None:
        """The Significance tool is SHOWN only where the page on screen draws a bubble chart to bracket (the
        Populations pages of SQD and Apparent Diffusion) and HIDDEN elsewhere, as the Test button is (asked 4 Nov
        2026 on the Spatial Map: "why is there a significance button when its not clickable"; it was dimmed so the
        rail never shifted). An open popover that has just lost its content closes."""
        has = self.significance.has_content()
        badge = self.significance.badge()
        button = self.rail.buttons["significance"]
        button.set_available(has)
        button.setVisible(has)
        choice = self.significance.significance.choice
        on = bool(choice is not None and choice.on)
        button.set_badge(badge, lit=on)
        words = self.SIGNIFICANCE_TIP
        if badge:
            plural = "" if badge == "1" else "s"
            words += f" ({badge} pair{plural}, brackets {'on' if on else 'off'})"
        self.rail.set_tool_tip("significance", words if has else self.SIGNIFICANCE_EMPTY_TIP)
        if not has and self._open_key == "significance" and self._pop is not None:
            self._pop.close_chain()

    def _sync_conditions_mark(self) -> None:
        """The first condition colours on the Conditions icon, in the colours the charts draw them in."""
        keys = [key for key, _name in self.conditions._conditions]
        self.rail.buttons["conditions"].set_swatches([self.conditions.colour_for(k) for k in keys])

    def drop_significance(self) -> None:
        """The page now on screen is not one the brackets belong to (Apparent Diffusion's per-condition detail): the
        tool describes nothing until `set_section` names a page again."""
        self.significance_section = ""
        self.significance.hide_significance()

    def set_conditions(self, conditions) -> None:
        self.conditions.set_conditions(conditions)
        self._sync_conditions_mark()

    def _pick(self, key: str) -> None:
        if key == "metadata":
            self.metadata_dock.toggle()
            self._sync_rail()
            return
        self._open_popover(key)

    def _sync_rail(self) -> None:
        for k, b in self.rail.buttons.items():
            b.set_active(k == self._open_key
                         or (k == "metadata" and self.metadata_dock.is_open()))

    def _open_popover(self, key: str) -> None:
        button = self.rail.buttons[key]
        if retriggered(button):
            self._open_key = None
            self._sync_rail()
            self.rail.show_tip(key)
            return
        self.rail.dismiss_tips()
        end_outros()
        t = self.t
        body = self._bodies[key]
        # `Panel`, in the page -- see the module docstring. Constructed
        # directly rather than through `styled_menu`, which only hands back a
        # Panel when its PARENT already is one; a rail button is not.
        menu = Panel(t, button.window(), root=ROOT_PX)
        menu.set_trigger(button)
        menu.set_side("left")
        pad = POP_PAD - rem(0.25)
        holder = QWidget()
        holder.setStyleSheet("background:transparent;")
        hl = QVBoxLayout(holder)
        hl.setContentsMargins(pad, pad, pad, pad)
        hl.setSpacing(0)
        # BORROWED, not given away -- see the prototype's own note: a top-level
        # panel frees itself once closed, and a body parented into it would go
        # with it, so the second open would reach a deleted C++ object.
        hl.addWidget(body)
        body.show()
        if key in ("visibility", "significance", "conditions"):
            # THE VISIBILITY BODY GROWS WHILE OPEN (pairs added with `+`, a p line per pair): the panel follows it,
            # and where the window is shorter than the body it scrolls rather than drawing its rows over each other
            # (3 Nov 2026, reported). So does Conditions since its Apply order bar comes and goes (6 Nov 2026).
            from qtpy.QtWidgets import QFrame, QScrollArea
            from ..widgets.primitives import FitScroll, scrollbar_css
            scroll = FitScroll()
            scroll.setWidget(holder)
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.NoFrame)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            scroll.setStyleSheet("QScrollArea { background:transparent; border:none; }" + scrollbar_css(t))
            scroll.viewport().setStyleSheet("background:transparent;")
            menu.add_widget(scroll)
            menu.follow_content(True)
        else:
            menu.add_widget(holder)
        menu.set_minimum_visible_width(POP_MIN)
        self._pop = menu
        self._open_key = key
        self._sync_rail()
        self.rail.track_pointer(True)
        # `closed`, not `aboutToHide` -- a Panel is a widget, not a QMenu.
        menu.closed.connect(lambda k=key: self._popover_closed(k))
        # `visible_width()`, NOT `sizeHint().width()` -- `popup(at)` already
        # treats `at` as where the CARD (not the wider, shadow-padded widget)
        # lands, so positioning by "4 px left of the rail" from the full
        # `sizeHint()` double-counts the shadow room and opened the popover
        # noticeably further from the rail than the 4 px this was meant to be.
        width = max(menu.visible_width(), POP_MIN)
        at = self.rail.mapToGlobal(QPoint(0, 0))
        menu.popup(QPoint(at.x() - 4 - width, button.mapToGlobal(QPoint(0, 0)).y()))

    def _popover_closed(self, key: str) -> None:
        body = self._bodies.get(key)
        if body is not None:
            close_panels_under(body)        # a drop-down opened inside it never outlives it
            body.setParent(self)
            body.hide()
        if self._open_key == key:
            self._open_key = None
        self._pop = None
        self.rail.track_pointer(False)
        self._sync_rail()
