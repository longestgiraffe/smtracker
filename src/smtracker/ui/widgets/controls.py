"""Buttons and the segmented toggle -- the things you press.

Assembled from a stylesheet plus a layout rather than a custom ``paintEvent``,
and that is not a stylistic preference -- it is what the shape asks for.
shadcn's button is ``rounded-md``: a 1 px border at 6 px radius, which Qt's
stylesheet engine draws correctly on its own.

A FULLY ROUNDED button would not have this luxury. A pill's 2 px stroke on a
semicircular cap gets clipped at the widget edge and renders with a flat chord
-- the thing that reads as "jagged" -- unless you inset the stroke by half its
width and draw it yourself. Worth stating plainly, because it is one of the few
places where a design choice directly removes Qt code rather than merely
changing values in it.
"""

from __future__ import annotations

from qtpy.QtCore import (
    QLocale,
    QEvent,
    QObject,
    QPointF,
    QRect,
    QRectF,
    Qt,
    Signal,
)
from qtpy.QtGui import QColor, QCursor, QPainter, QPainterPath, QPen
from qtpy.QtWidgets import (
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSpinBox,
    QStyle,
    QStyleOption,
    QWidget,
)

from . import tooltip
from ..theme import icons
from ..theme import typography as ty
from .primitives import (
    ColorFade,
    Icon,
    Surface,
    draw_shadow,
    icon,
    label,
    mix,
)
from .primitives import discard

__all__ = ["Button", "FloatField", "NumberField", "Segmented", "Slider", "Toggles"]


def _clear(colour: str) -> str:
    """``colour`` at zero opacity, as ``#00rrggbb`` (which `QColor` reads back)."""
    c = QColor(colour)
    c.setAlpha(0)
    return c.name(QColor.NameFormat.HexArgb)


def _css(colour: str) -> str:
    """A colour for a stylesheet: ``rgba(...)`` when it is see-through, since a
    ``#rrggbb`` would drop the opacity, else as given."""
    c = QColor(colour)
    if c.alpha() < 255:
        return f"rgba({c.red()}, {c.green()}, {c.blue()}, {c.alpha()})"
    return colour


class Button(Surface):
    """shadcn's Button, in the three variants this screen uses.

    ``default`` is a filled zinc-900 -- shadcn's primary is a NEUTRAL, not a
    brand hue, which is why the only saturated colour left on this screen
    belongs to the data.
    """

    clicked = Signal()

    #: variant -> (fill, ink, border or None)
    def _skin(self, variant: str):
        t = self.t
        return {
            "default": (t.primary, t.primary_foreground, None),
            "outline": (t.card, t.foreground, t.input),
            # NOT "transparent": that is BLACK at zero opacity, and the hover fade
            # interpolated from it and wrote each frame as `#rrggbb`, dropping the
            # opacity -- so a ghost button turned black after the pointer left it
            # (reported on the u-track card's "Show all settings", 12 Oct 2026). Its
            # rest is its own hover colour at zero opacity, so only the opacity fades.
            "ghost": (_clear(t.accent), t.foreground, None),
        }[variant]

    def _hover_fill(self, variant: str) -> str:
        """The registry's hover for each variant, as a flat colour.

        shadcn writes these as alpha over whatever is behind
        (``hover:bg-primary/90``); Qt has no such compositing in a stylesheet
        fill, so the blend is done here against the page. Same result, because
        these controls always sit on the page or on a card, both of which are
        ``--background``/``--card``.

            default   hover:bg-primary/90
            outline   hover:bg-accent hover:text-accent-foreground
            ghost     hover:bg-accent
        """
        t = self.t
        if variant == "default":
            return mix(t.primary, t.background, 0.90)
        return t.accent

    def __init__(self, t, text: str, *, variant: str = "outline",
                 size: str = "default", glyph: str | None = None,
                 parent=None):
        # ONLY ``outline`` is shadowed, and with ``shadow-xs`` -- read off the
        # registry rather than assumed. A filled ``default`` button carries no
        # shadow at all there, and this one used to draw the Card's
        # ``shadow-sm`` under both, which is two steps too heavy on the filled
        # one and one step on the outline.
        super().__init__(t, radius=t.radius_md,
                         shadowed=variant == "outline", shadow=t.shadow_xs,
                         parent=parent)
        self._variant = variant
        self._hover = False
        #: `active:translate-y-px` -- see `mousePressEvent`.
        self._pressed = False
        #: `aria-expanded` -- its popup is open. See `set_expanded`.
        self._expanded = False
        fill, ink, edge = self._skin(variant)
        self._fill, self._edge = fill, edge
        #: The colour ON SCREEN this frame, which is not ``_fill`` while the
        #: transition is running. See `_fade`.
        self._now = fill
        self._fade = ColorFade(self, self._on_fade)
        height = t.control_height if size == "default" else t.control_height_sm
        pad = 16 if size == "default" else 12

        self.setObjectName("btn")
        self._restyle()
        self.setFixedHeight(height)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setCursor(Qt.PointingHandCursor)

        # 13 AND 14 px glyphs -- ONE NOTCH UNDER the registry, deliberately.
        #
        # The registry is unambiguous: `text-sm` at both sizes (`sm` is
        # `h-8 gap-1.5 rounded-md px-3` and does not override the base
        # `text-sm`; only `xs` pairs `text-xs` with `h-6`), with `size-4`
        # glyphs. That was tried, at 14/16, and read a size too large on this
        # screen. It is a judgement and it is recorded as a DEVIATION, not
        # dressed up as a correction.
        #
        # What is not a judgement: the type and the glyph move together. The
        # earlier 13 px text carried a 12 px MUTED glyph, and that combination
        # is the one the registry really does rule out -- an icon in a third
        # colour the component does not have, at a size unrelated to its label.
        step = "control" if size == "small" else "sm"
        lay = QHBoxLayout(self)
        lay.setContentsMargins(pad, 0, pad, 0)
        # ``gap-2`` at the default size, ``gap-1.5`` at small -- the registry
        # gives the two sizes different gaps, where this used 6 for both.
        lay.setSpacing(8 if size == "default" else 6)
        if glyph:
            # `size-4` AND the button's own ink. Both off the registry's base
            # class: `[&_svg:not([class*='size-'])]:size-4` sizes every button
            # icon at 16 (only `xs` drops it to `size-3`), and no variant sets a
            # colour on the svg -- it inherits `currentColor`, so the glyph is
            # the same ink as the label. This drew 12 px in `muted-foreground`,
            # a third weight the component does not have: the icon read as a
            # decoration attached to the label rather than half of it.
            lay.addWidget(icon(glyph, icons.CONTROL, ink))
        self.title = label(text, ty.ui(step, "medium"), ink)
        lay.addWidget(self.title)

    def set_text(self, text: str) -> None:
        """Relabel in place.

        A button whose text is its STATE needs this -- the page's Compute reads
        `Compute` / `Computing` / `Recompute` off the job. Kept to the label
        rather than rebuilding the button, because rebuilding would drop the
        hover fade mid-transition and re-seed the press state.
        """
        if self.title.text() != text:
            self.title.setText(text)
            # `w-fit`: the button is Fixed/Fixed, so it does not re-measure on
            # its own and the new word would be clipped by the old width.
            self.title.adjustSize()
            self.adjustSize()
            self.updateGeometry()

    def text(self) -> str:
        return self.title.text()

    def changeEvent(self, e):
        """`disabled:opacity-50`, which Qt does not give a styled QFrame for free.

        A `QPushButton` greys itself; this is a `Surface` with its own fill, so
        `setEnabled(False)` changed the click behaviour (Qt stops delivering
        mouse events) and NOTHING on screen -- a button that looks ready and
        does nothing, which is worse than one that is visibly not ready. Caught
        on the Confirm screen, where `Open workspace` waits for a load.
        """
        from qtpy.QtCore import QEvent

        if e.type() == QEvent.EnabledChange:
            self._restyle()
            for child in self.findChildren(QLabel):
                child.setStyleSheet(
                    f"color:{self._ink()}; background:transparent;")
            for child in self.findChildren(Icon):
                child.colour = self._ink()
                child.update()
        super().changeEvent(e)

    def _ink(self) -> str:
        """The label colour for this variant and state."""
        _fill, ink, _edge = self._skin(self._variant)
        return ink if self.isEnabled() else mix(ink, self.t.background, 0.5)

    def _restyle(self) -> None:
        t = self.t
        edge = self._edge
        # A stylesheet MARGIN, which Qt applies to the widget's own box: the
        # painted frame AND its contents shift together and the widget's
        # geometry does not change, so nothing in the layout around it moves.
        # MEASURED: frame top 0 -> 1, label y 1 -> 2, height still 32.
        dip = " margin-top:1px; margin-bottom:-1px;" if self._pressed else ""
        # The FILL fades with the ink -- a filled `default` button whose label
        # went to 50 % over a full-strength zinc-900 would read as low-contrast
        # text on a live button rather than as a button that is not ready.
        if self.isEnabled() or QColor(self._now).alpha() < 255:
            fill = _css(self._now)
        else:
            fill = mix(self._now, t.background, 0.5)
        if edge and not self.isEnabled():
            edge = mix(edge, t.background, 0.5)
        self.setStyleSheet(
            f"QFrame#btn {{ background:{fill};"
            f" border:{f'1px solid {edge}' if edge else 'none'};"
            f" border-radius:{t.radius_md:g}px;{dip} }}")

    def _on_fade(self, colour) -> None:
        # The OPACITY IS KEPT: `name()` alone is `#rrggbb` and turns a see-through
        # frame of a ghost button's fade into an opaque one.
        self._now = (colour.name(QColor.NameFormat.HexArgb) if colour.alpha() < 255
                     else colour.name())
        self._restyle()

    # -- the transition ------------------------------------------------------ #
    #
    # `transition-all` is in the registry's BASE class, and it is the whole of
    # what a shadcn button animates. CHECKED, because the obvious guess is
    # wrong: there is no `active:` anything in `buttonVariants` -- no press
    # fill, no `active:scale`, nothing. A click produces no state change at all
    # for a mouse user. What reads as the button being alive is the FILL
    # FADING when the pointer arrives and leaves, over Tailwind's default
    # 150 ms on `cubic-bezier(0.4, 0, 0.2, 1)`; this snapped instantly, which
    # is the difference actually being felt.
    #
    # So the fade is implemented and a press effect is NOT invented. Adding one
    # would be designing past the registry while claiming to follow it.
    def enterEvent(self, e):
        self._hover = True
        if not self._expanded:
            self._fade.to(self._now, self._hover_fill(self._variant))
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._hover = False
        # Dragging off a held button releases the dip, as `:active` does.
        #
        # `_restyle` EXPLICITLY, not via the fade: `ColorFade.to` returns early
        # when the colour is already at rest, so a button left without ever
        # having been hovered would clear the flag and never repaint -- it would
        # stay visibly dipped. Caught by a check that pressed without hovering
        # first, which a person cannot do and a probe does by default.
        was_pressed, self._pressed = self._pressed, False
        if was_pressed:
            self._restyle()
        if not self._expanded:
            self._fade.to(self._now, self._fill)
        super().leaveEvent(e)

    def set_expanded(self, on: bool) -> None:
        """`aria-expanded:bg-muted` -- hold the fill while the popup is open.

        This is the OTHER half of `active:not-aria-[haspopup]`, and without it
        that rule leaves a trigger with no feedback at all. `.cn-button-variant-
        outline` carries `aria-expanded:bg-muted aria-expanded:text-foreground`,
        so a button whose menu is open stays filled -- that is what stands in
        for the press dip it deliberately does not get.

        MEASURED before this existed: clicking `Filter` sent the button a
        `Leave` the moment the menu took the mouse grab, so it faded back to
        #FFFFFF and sat there, unpressed-looking, with its own menu open under
        it. Rest #FFFFFF, hover #F4F4F5, menu open #FFFFFF -- where the registry
        wants #F4F4F5.

        Not faded on the way in: the popup appears at once, and easing the
        trigger's fill behind it reads as lagging the thing it just opened.
        """
        if on == self._expanded:
            return
        self._expanded = on
        if on:
            self._fade.stop()
            self._now = self.t.muted
            self._restyle()
        else:
            # The pointer may have moved away while the popup was up, so the
            # resting state is re-derived from WHERE IT IS rather than from the
            # stale `_hover` flag -- the Leave that arrived with the grab left
            # that flag saying "outside" even when it is not.
            over = self.rect().contains(self.mapFromGlobal(QCursor.pos()))
            self._hover = over
            self._fade.to(self._now,
                          self._hover_fill(self._variant) if over else self._fill)

    # -- the press ----------------------------------------------------------- #
    #
    # `active:translate-y-px` -- the button moves DOWN one pixel while held. It
    # is in the base class of every one of shadcn's eight styles (luma, lyra,
    # maia, mira, nova, rhea, sera, vega), identically.
    #
    # This was previously recorded here as "shadcn has no press state", which
    # was wrong, and wrong in an interesting way: it is true of the Radix
    # `new-york-v4` registry that `shadcn view @shadcn/button` returns, where
    # `buttonVariants` really does contain no `active:` anything. The newer Base
    # UI registry moved the visuals out of the .tsx into CSS classes
    # (`.cn-button`), so the .tsx looks even emptier while the press state lives
    # a file away. Checking one registry and concluding about the component was
    # the mistake.
    #
    # EVERY button dips, including menu and dialog triggers -- a DELIBERATE
    # departure, and the reasoning is worth keeping because the registry's own
    # rule turns out to be inert in this palette.
    #
    # The registry writes `active:not-aria-[haspopup]:translate-y-px`, exempting
    # triggers on the grounds that the popup appearing is already the feedback,
    # with `aria-expanded:bg-muted` holding the trigger filled meanwhile. That
    # substitute is invisible here, and MEASURED to be:
    #
    #     light   accent #F4F4F5   muted #F4F4F5    identical
    #     dark    accent #27272A   muted #27272A    identical
    #
    # `hover:bg-muted` and `aria-expanded:bg-muted` are the SAME token, and a
    # button is always hovered at the moment it is clicked -- so the fill does
    # not change and the exemption leaves a trigger with no click feedback at
    # all. That is true of shadcn on the web too; it is a property of the zinc
    # palette, not of this port. `set_expanded` is kept because it still earns
    # its place when the pointer moves AWAY while the popup is open (the button
    # holds its fill instead of going white), but it cannot carry the press.
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._pressed = True
            self._restyle()
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e):
        if self._pressed:
            self._pressed = False
            self._restyle()
        if e.button() == Qt.LeftButton and self.rect().contains(e.pos()):
            self.clicked.emit()
        super().mouseReleaseEvent(e)




class _TabInk(QObject):
    """`hover:text-foreground` on a tab you are not on, faded by `transition-all`.

    An inactive `TabsTrigger` is deliberately dimmed and comes to full ink under
    the pointer -- the registry, verbatim::

        text-foreground/60            inactive, light
        hover:text-foreground         hover
        dark:text-muted-foreground    inactive, dark
        dark:hover:text-foreground    hover
        transition-all                so it fades rather than snaps

    Note the ASYMMETRY, which is easy to flatten by mistake: light dims the
    FOREGROUND to 60 %, dark switches to `muted-foreground` outright. Measured
    against this palette they are NOT the same colour -- on dark both land on
    #A1A1AA, but on light `foreground/60` over the tray is **#676769** where
    `muted-foreground` is #71717A. This port had the dark case right and the
    light case a shade too pale.

    Drives the label AND the glyph: an icon-only tab has no label, and the
    registry's rule is about the trigger's ink rather than about text.
    """

    def __init__(self, pill, parts, rest: str, hover: str):
        super().__init__(pill)
        self._parts = parts
        self._rest, self._hover = rest, hover
        self._now = rest
        self._fade = ColorFade(self, self._apply)
        pill.setAttribute(Qt.WA_Hover, True)
        pill.installEventFilter(self)

    def _apply(self, colour) -> None:
        self._now = colour.name()
        for w in self._parts:
            if isinstance(w, Icon):
                w.colour = self._now
                w.update()
            else:
                w.setStyleSheet(f"color:{self._now}; background:transparent;")

    def set_rest(self, colour: str) -> None:
        """Re-point what a `Leave` fades back TO, and jump the CURRENT
        colour there immediately, cancelling any fade already in flight.

        For `Segmented._restyle_enabled` -- a pill's dimmed-vs-full-
        strength baseline can move with no `Leave` event to announce it
        (the pointer is still sitting on a pill the instant `Lock Model`
        disables its row), and disabled pills stop receiving hover events
        at all (see `eventFilter`'s own guard) -- so nothing would ever
        fade a stranded-mid-hover pill back down on its own. `ColorFade.
        to(x, x)` alone is not enough here: it stops the running
        animation but, being a no-op start==end fade, never calls
        `_apply`, so the STOPPED colour would stay wherever the animation
        had reached rather than snapping to `colour`.
        """
        self._rest = colour
        self._fade.stop()
        self._apply(QColor(colour))

    def eventFilter(self, obj, event):
        # `disabled:` in shadcn also means `pointer-events-none` -- no
        # hover reaction at all, not merely no click. Reported directly:
        # a pill hovered the instant its row got disabled (`Lock Model`)
        # stayed lit at full ink, reading as a hover effect surviving on
        # a control that should have none. Checked on `obj` (the pill),
        # not `self`, which has no enabled state of its own.
        if not obj.isEnabled():
            return False
        kind = event.type()
        if kind in (QEvent.Enter, QEvent.HoverEnter):
            self._fade.to(self._now, self._hover)
        elif kind in (QEvent.Leave, QEvent.HoverLeave):
            self._fade.to(self._now, self._rest)
        return False


class _PillHover(QObject):
    """Turns a pill's Enter/Leave into `Segmented.hovered`. One filter per
    tray, installed on every pill (re)built, so the pill's own index rides on
    a property rather than on this object -- `select` rebuilds the pills."""

    def __init__(self, seg):
        super().__init__(seg)
        self._seg = seg

    def eventFilter(self, obj, e):
        kind = e.type()
        if kind == QEvent.Enter:
            index = obj.property("segIndex")
            self._seg.hovered.emit(-1 if index is None else int(index))
        elif kind == QEvent.Leave:
            self._seg.hovered.emit(-1)
        return False


class Segmented(QWidget):
    """shadcn's ``TabsList`` shape: a muted tray with one raised pill.

    The same widget serves the condition tabs and the Cards/Table view switch,
    because in shadcn they ARE the same component -- which is itself the
    finding. Those two jobs are conventionally given different shapes, an
    underline tab strip and a segmented control; here they get one, so the
    screen carries one fewer idea. Whether that is a simplification or a loss
    of signal is a judgement, not a fact, and it is exactly the kind of thing
    this probe exists to make visible.

    **It paints its own pill's shadow, and every one of the three ways that
    can fail was live.** `TabsTrigger` carries
    ``group-data-[variant=default]/tabs-list:data-[state=active]:shadow-sm``,
    the active pill has always passed ``shadowed=True``, and the shadow reached
    the screen in NONE of this prototype's four segmented controls. A widget
    cannot paint its own shadow -- the nearest :class:`Surface` ancestor does
    -- and the tray was a plain ``QWidget``, so the job fell upwards to
    whatever happened to be above it:

    * the condition tabs and the icon-only view switch had no `Surface`
      ancestor at all, so the answer was *nobody*;
    * the Cards/Table switch had one, `GroupsPanel` -- which paints BEFORE its
      children, so the tray's own opaque ``--muted`` fill covered the shadow
      the panel had just drawn. This is the exact case `own_surfaces` warns
      about;
    * the popover's controls resolved to the rail's `ToolButton`, which is in
      a DIFFERENT WINDOW: `paint_shadows` maps with ``mapTo``, so the shadow
      was placed inside a 28 px button in the main window and clipped away.

    Painting it here fixes all three at once and puts the shadow where it
    belongs in the paint order -- over the tray's fill, under the pill.

    **Directly, not via :class:`ShadowHost`.** That base was tried first and
    works, but it is built for a container holding an unknown, changing set of
    cards: it caches ``findChildren(Surface)``, and overrides ``event()`` to
    drop that cache, which puts a Python call on every event this widget sees.
    A tray has exactly ONE raised pill and already tracks which -- so there is
    nothing to discover and nothing to invalidate, and `draw_shadow` is called
    once with the geometry we already hold. (Round-robined, the `ShadowHost`
    version showed the same median outro pacing, 16.0 ms against 15.9, but
    spiked past vsync on 2 of 7 runs where the plain widget never did in 8.
    Not conclusive on its own; the direct call makes the question moot.)

    ONE deviation, noted because it is real: CSS does not clip a ``box-shadow``
    to the parent, and Qt clips a painter to its widget. The tray's padding is
    3-4 px against a 4 px shadow, so at most the outermost pixel of a 5 %-alpha
    shadow is lost at the tray's edge. Painting it in the tray's PARENT instead
    would restore that pixel and bring back the covering bug above.
    """

    changed = Signal(int)
    #: A pill was CLICKED, the current one included -- `changed` stays silent on a click of
    #: the pill already raised. For a pill that is an action as well as a state (the
    #: Colormap row's "Custom…" reopens its editor when it is already the ramp in use).
    pressed = Signal(int)
    #: The pointer is over pill ``index``, or ``-1`` when it left. Emitted
    #: from the pills' own Enter/Leave, so it survives `select`'s rebuild.
    hovered = Signal(int)
    #: A tab's colour dot was clicked: (index, the dot's global rect).
    #:
    #: The RECT, not just the index, because a popover that opens at the
    #: panel's corner instead of under the thing you clicked makes you
    #: re-find your own pointer. The caller cannot recover this position
    #: on its own -- only the dot knows where it ended up.
    swatch_clicked = Signal(int, QRect)

    def __init__(self, t, items, *, current: int = 0, glyphs=None,
                 swatches=None, tips=None, size: str = "default", parent=None):
        super().__init__(parent)
        self.t = t
        self._current = current
        #: Tells `primitives.own_surfaces` to stop here, the same as an
        #: actual `Surface` would -- this tray's own `paintEvent` already
        #: draws its raised pill's shadow directly (see the class docstring:
        #: every ancestor-drawn case for that pill was historically broken),
        #: so an outer `Surface` finding a pill via `findChildren` and
        #: drawing a SECOND shadow for it is never correct. Load-bearing once
        #: a `Surface` ancestor exists above a `Segmented` that rebuilds its
        #: pills (`select()`) -- without it that ancestor's shadow cache
        #: keeps a reference to the pill `select()` just deleted, and paints
        #: it at a wrong position for a frame. See `own_surfaces`.
        self._shadow_boundary = True
        height = t.control_height + 4 if size == "default" else 32
        pad = 4 if size == "default" else 3
        self.setObjectName("tray")
        # A QWidget SUBCLASS does not paint a stylesheet background on its own
        # -- Qt only does that for plain QWidget instances, or for subclasses
        # that reimplement paintEvent. Without this the tray is invisible and
        # the design reads as a white pill floating on white, which is exactly
        # how it first rendered. `paintEvent` below draws it explicitly now, so
        # this is belt AND braces; kept because it is what was proven here.
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(
            f"QWidget#tray {{ background:{t.muted}; border:none;"
            # ``rounded-md`` at EVERY size: the registry gives the base class
            # and each size variant the same one, so the radius does not track
            # the height the way it used to here.
            f" border-radius:{t.radius_md:g}px; }}")
        self.setFixedHeight(height)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self._items = list(items)
        self._hover_filter = _PillHover(self)
        self._glyphs, self._swatches = glyphs, swatches
        #: Tooltips belong to the GROUP, not to the pill they are on: `select`
        #: destroys and recreates every pill, so a tooltip attached from
        #: outside would be watching a widget that no longer exists after the
        #: first click. Measured -- the tab tooltips worked until a tab was
        #: used, then silently stopped.
        self._tips = list(tips) if tips else None
        self._tip_watch: list = []
        self._size = size
        self._pill_h = height - 2 * pad
        lay = QHBoxLayout(self)
        lay.setContentsMargins(pad, pad, pad, pad)
        lay.setSpacing(2 if size == "small" else 4)
        self._pills: list[QWidget] = []
        for i, text in enumerate(items):
            pill = self._pill(text, i == current, self._pill_h, size,
                              glyphs[i] if glyphs else None,
                              swatches[i] if swatches else None)
            # The pill is what you click, so the pill is what carries the
            # handler. Putting it on the tray instead would mean hit-testing
            # against child geometry to work out which segment was meant.
            pill.setCursor(Qt.PointingHandCursor)
            pill.mouseReleaseEvent = lambda _e, k=i: self._press(k)
            pill.setProperty("segIndex", i)
            pill.installEventFilter(self._hover_filter)
            lay.addWidget(pill)
            self._pills.append(pill)
        self._retip()

    def _retip(self) -> None:
        """(Re)hang the tooltips on the CURRENT pills."""
        if not self._tips:
            return
        self._tip_watch = [
            tooltip.attach(self.t, self._pills[i], text)
            for i, text in enumerate(self._tips) if i < len(self._pills)
        ]

    def changeEvent(self, e):
        """`disabled:opacity-50`, the same defect `Button.changeEvent`
        already documents and fixes for itself -- a `Surface`-based pill
        paints its OWN fill and ink, so `setEnabled(False)` changes the
        click behaviour (Qt stops delivering mouse events) and NOTHING on
        screen. Reported directly: locking `ConditionRow.model` via
        `Lock Model` left every row's `Best`/`1`../`saSPT` pills looking
        exactly as pressable as before, read as "I can't select anything
        in Conditions any more" rather than what it actually was (Lock
        Model set to anything but `Per-Condition`, disabling the
        per-condition choice by design -- see `ConditionRow.set_locked`).
        """
        if e.type() == QEvent.EnabledChange:
            self._restyle_enabled()
        super().changeEvent(e)

    def _restyle_enabled(self) -> None:
        """Re-tint every pill for the current enabled state, without
        rebuilding them -- `select()`'s own rebuild is for a CHANGE of
        selection; this is the same pills, dimmed or not. Uses
        `_pill_colors`, the SAME computation `_pill`'s fresh build does,
        so enabling back returns bit-for-bit what `_pill` would have
        built fresh."""
        for i, pill in enumerate(self._pills):
            active = self._is_on(i)
            fill, ink = self._pill_colors(active)
            pill.setStyleSheet(
                f"QFrame#seg {{ background:{fill}; border:none;"
                f" border-radius:{pill.radius:g}px; }}")
            tabink = getattr(pill, "_ink", None)
            if tabink is not None:
                # An INACTIVE pill with a hover fade (`_TabInk`, only
                # built `if not active and parts` -- see `_pill`) has its
                # own idea of the current colour, independent of this
                # stylesheet write. `set_rest` re-points AND snaps it,
                # so a pill hovered the instant it gets disabled cannot
                # stay stranded at full ink -- see its own docstring.
                tabink.set_rest(ink)
            else:
                for child in pill.findChildren(QLabel):
                    child.setStyleSheet(f"color:{ink}; background:transparent;")
                for child in pill.findChildren(Icon):
                    child.colour = ink
                    child.update()

    def current(self) -> int:
        return self._current

    def _is_on(self, index: int) -> bool:
        """Whether pill ``index`` is RAISED: the one current pill here; any of several in `Toggles`. The one
        test the rebuild, the enabled re-tint and the shadow all read."""
        return index == self._current

    def pill(self, index: int):
        """One segment, so a caller can hang a tooltip on it.

        Per PILL, not on the tray: two icons in one tray need two different
        labels, and a tooltip on the group could only ever say one thing.
        """
        return self._pills[index]

    def set_labels(self, items) -> None:
        """New words on the SAME pills, the selection kept and nothing emitted: for a label that
        names something learnt after the control was built (the Spatial Map's "1 px (106 nm)",
        whose pixel is the data set's). The count cannot change -- that is a different control."""
        items = [str(i) for i in items]
        if len(items) != len(self._items):
            raise ValueError(f"set_labels keeps {len(self._items)} pills, got {len(items)}")
        if items == self._items:
            return
        self._items = items
        self._rebuild_pills()

    def set_swatches(self, colours) -> None:
        """Re-point the dots without touching the selection."""
        self._swatches = list(colours)
        keep, self._current = self._current, -1
        self.select(keep)

    def _press(self, index: int) -> None:
        """A click on pill ``index``: `pressed` always, then the ordinary `select`."""
        self.pressed.emit(index)
        self.select(index)

    def select(self, index: int) -> None:
        """Move the raised pill, and tell anyone who asked.

        Rebuilt rather than restyled: the active pill differs from an inactive
        one in fill, shadow AND ink colour, and three coordinated restyles is
        the kind of thing that goes half-done. At two to three segments the
        rebuild is free.

        The shadow needs no part of that: `paintEvent` reads `_current` when
        it draws, so it follows the selection on its own.

        ``-1`` DESELECTS -- no pill raised, nothing to draw a shadow under
        (`paintEvent`'s own `0 <= self._current` guard already covers that
        half; this is the other half, the rebuild). For a group whose real
        choice lives in a SIBLING `Segmented` -- two tabbed rows standing in
        for one value neither alone can list -- rather than faking a shared
        selection by leaving a stale pill lit here. A caller reading
        `current()` on a possibly-deselected instance must treat ``-1`` as
        "nothing here", never index into its own `items` with it: Python
        wraps a bare ``-1`` index to the LAST item, which is the one silent
        way this could read as a real, wrong answer instead of an absent one.

        **THE REBUILD RUNS WITH THE WHOLE WINDOW'S REPAINTING SUSPENDED**
        (`self.window()`, not just this tray). Each `removeWidget` /
        `insertWidget` in the loop below invalidates layout on its own, and
        each invalidation can trigger its own intermediate repaint before the
        loop finishes -- invisible against an opaque ancestor, but against a
        TRANSLUCENT one (the Compute dialog's `bg-black/50` overlay) each of
        those partial repaints re-blends that alpha fill on top of whatever
        the previous partial repaint already blended, and the accumulation
        flashes visibly in exactly this tray's own rectangle for a frame --
        reported as "the picker flickers" and reproduced by switching a
        `Segmented` inside that dialog specifically; the SAME control in an
        opaque toolbar popover does not, which is what pointed at the
        ancestor rather than the tray. `self.window()` because the flashing
        ancestor sits several levels above this widget -- `setUpdatesEnabled
        (False)` on the tray alone would not reach it.
        """
        if index == self._current or (index != -1 and not 0 <= index < len(self._items)):
            return
        self._current = index
        self._rebuild_pills()
        self.changed.emit(index)

    def _rebuild_pills(self) -> None:
        """Every pill built again, raised where `_is_on` says (`select`'s docstring: why rebuilt, and why with the
        whole window's repainting suspended)."""
        window = self.window()
        window.setUpdatesEnabled(False)
        try:
            lay = self.layout()
            for i, old in enumerate(self._pills):
                lay.removeWidget(old)
                discard(old)
                pill = self._pill(self._items[i], self._is_on(i), self._pill_h,
                                  self._size,
                                  self._glyphs[i] if self._glyphs else None,
                                  self._swatches[i] if self._swatches else None)
                pill.setCursor(Qt.PointingHandCursor)
                pill.mouseReleaseEvent = lambda _e, k=i: self._press(k)
                pill.setProperty("segIndex", i)
                pill.installEventFilter(self._hover_filter)
                lay.insertWidget(i, pill)
                self._pills[i] = pill
            self._retip()
        finally:
            window.setUpdatesEnabled(True)

    def paintEvent(self, e):
        """The tray, then the raised pill's shadow. In that order.

        A shadow belongs UNDER the thing it describes and OVER what that thing
        sits on, and Qt paints a parent before its children -- so drawing it
        here lands it between the tray's fill and the pill, which is exactly
        where ``shadow-sm`` goes.
        """
        opt = QStyleOption()
        opt.initFrom(self)
        p = QPainter(self)
        self.style().drawPrimitive(QStyle.PE_Widget, opt, p, self)
        for i, pill in enumerate(self._pills):
            if self._is_on(i) and pill.shadowed:
                p.setRenderHint(QPainter.Antialiasing, True)
                draw_shadow(p, QRectF(pill.geometry()), pill.radius, self.t,
                            pill.shadow)
        p.end()

    def _pill_colors(self, active: bool) -> tuple[str, str]:
        """The fill and ink for one pill, given whether it is the
        SELECTED one and whether the whole tray is currently enabled --
        the one place `_pill` (a fresh build, used by `__init__` AND by
        `select`'s own rebuild) and `_restyle_enabled` (an in-place
        retint on an `EnabledChange`) both compute this, so the two
        cannot drift the way they did when `select`'s rebuild ignored
        `isEnabled()` entirely: pushing a value onto a LOCKED (disabled)
        `ConditionRow.model` via `.select()` silently rebuilt it at full
        strength, un-dimming a row that was still meant to read as
        disabled. `text-foreground/60` on light, `dark:text-muted-
        foreground` on dark for the base dim -- see `_TabInk` for why the
        two schemes differ.
        """
        t = self.t
        dim = (t.muted_foreground if t.color_scheme == "dark"
              else mix(t.foreground, t.muted, 0.6))
        ink = t.foreground if active else dim
        fill = t.card if active else "transparent"
        if not self.isEnabled():
            ink = mix(ink, t.background, 0.5)
            if active:
                fill = mix(fill, t.background, 0.5)
        return fill, ink

    def _pill(self, text: str, active: bool, height: int, size: str,
              glyph: str | None, swatch: str | None) -> QWidget:
        """One segment. Active gets ``bg-background`` + ``shadow-sm``.

        The inactive segment is transparent rather than filled with the tray
        colour: on the tray it looks identical, and it means an inactive pill
        never has to be kept in sync with the tray's own fill.
        """
        t = self.t
        fill, ink = self._pill_colors(active)
        w = Surface(t, radius=t.radius_md if size == "default" else t.radius_sm,
                    shadowed=active)
        w.setObjectName("seg")
        w.setStyleSheet(
            f"QFrame#seg {{ background:{fill};"
            f" border:none; border-radius:{w.radius:g}px; }}")
        w.setFixedHeight(height)
        lay = QHBoxLayout(w)
        lay.setContentsMargins(10 if size == "small" else 12, 0,
                               10 if size == "small" else 12, 0)
        lay.setSpacing(7)
        parts = []
        if swatch:
            dot = QWidget()
            dot.setFixedSize(10, 10)
            dot.setStyleSheet(f"background:{swatch}; border-radius:2px;")
            # The dot is a control of its own: clicking the LABEL selects the
            # tab, clicking the COLOUR asks to change it. Two jobs in one pill,
            # separated by which part you hit -- which is what the swatch being
            # there implies, and what a separate button would clutter.
            dot.setCursor(Qt.PointingHandCursor)
            index = self._items.index(text) if text in self._items else 0
            dot.mouseReleaseEvent = (
                lambda _e, k=index, d=dot: self.swatch_clicked.emit(
                    k, QRect(d.mapToGlobal(d.rect().topLeft()), d.size())))
            lay.addWidget(dot)
        if glyph:
            # WITH text, the Button's 14; ALONE, the registry's 16.
            #
            # `TabsTrigger` carries the same `size-4` rule the Button does, so
            # these two follow the Button exactly -- the Cards/Table switch
            # shares a toolbar row with three of them, and two sizes of the same
            # thing in one row is worse than either size. An icon-only pill is
            # the other case: nothing is beside the glyph for it to be a size
            # too large next to, and it is the whole label, so it keeps 16.
            glyph_w = icon(glyph, icons.LG if not text else icons.CONTROL, ink)
            parts.append(glyph_w)
            lay.addWidget(glyph_w)
        words = text.split("\t")
        # Whatever the Button does, for the same reason as the glyph above.
        step = "control" if size == "small" else "sm"
        if words[0]:
            lead = label(words[0], ty.ui(step, "medium"), ink)
            parts.append(lead)
            lay.addWidget(lead)
        elif glyph:
            # ICON-ONLY. An empty QLabel is not free -- it still claims its
            # font's width and the layout's spacing, so a two-icon tab group
            # would sit lopsided with invisible text padding inside it. The
            # margins tighten to a square-ish `icon` shape instead.
            lay.setContentsMargins(9, 0, 9, 0)
            lay.setSpacing(0)
        if len(words) > 1:
            lay.addWidget(label(words[1], ty.ui(step, "normal"),
                                t.muted_foreground))
        if not active and parts:
            # Only an INACTIVE trigger gets a hover: the one you are on is
            # already at full ink and has nowhere to go.
            w._ink = _TabInk(w, parts, ink, t.foreground)
        return w


class Toggles(Segmented):
    """`Segmented`'s tray with SEVERAL pills raised at once, each on or off on its own (asked 30 Oct 2026 of the Jump
    distances page's Show: "just make it clickable and the one u click appears, when u click again it disappears,
    so u can click 1 2 3 and all appear").

    CONTROLLED, so the rule stays headless: a click only says which pill (`pressed`, the index); the owner works
    out what is on now -- a toggle, and on that page the last fit left on staying on (`jumps_view.toggle_show`) --
    and lights it with `set_on`, which is silent. `on()` lists the raised pills; `current()` is -1 (there is no one
    choice); `select(i)` raises ``i`` alone. Every raised pill wears the active look and its shadow
    (`Segmented._is_on`), so several read as several choices made, not as one."""

    def __init__(self, t, items, *, on=(), tips=None, size: str = "default", parent=None):
        super().__init__(t, items, current=-1, tips=tips, size=size, parent=parent)
        self._on: set = set()
        self.set_on(on)

    def _is_on(self, index: int) -> bool:
        return index in getattr(self, "_on", ())

    def on(self) -> list:
        """The raised pills' indices, ascending."""
        return sorted(self._on)

    def set_on(self, indices) -> None:
        """Raise exactly ``indices`` (others lowered); out-of-range ones are ignored. Silent: the caller decided."""
        wanted = {int(i) for i in indices if 0 <= int(i) < len(self._items)}
        if wanted == self._on:
            return
        self._on = wanted
        self._rebuild_pills()
        self.update()

    def _press(self, index: int) -> None:
        """A click: `pressed` only -- the owner decides what is on (`set_on`)."""
        self.pressed.emit(index)

    def select(self, index: int) -> None:
        """Pill ``index`` alone (-1: none)."""
        self.set_on([] if index == -1 else [index])


class Slider(QWidget):
    """shadcn's ``Slider``, read off `ui/slider.tsx`::

        SliderTrack   relative grow overflow-hidden rounded-full bg-muted
                      data-[orientation=horizontal]:h-1.5
        SliderRange   absolute bg-primary
        SliderThumb   block size-4 shrink-0 rounded-full border border-primary
                      bg-white shadow-sm ring-ring/50 hover:ring-4

    Painted rather than a styled `QSlider`, for the reason `Progress` gives
    one class over: Qt applies a stylesheet ``border-radius`` in DEVICE pixels,
    so on a 1.25 screen every radius here would render 20 % small -- and a
    fully-rounded 7 px groove that is 20 % under is visibly a rectangle with
    soft corners. It also avoids overriding a `QSlider`'s own groove, handle,
    tick and focus rules back to nothing, which is more code than drawing two
    rounded rects and a circle.

    **The thumb is `bg-white`, not `--background`.** The registry says `white`
    literally, and that is not an oversight to normalise: a thumb has to read
    as raised off the track in both themes, and in dark mode `--background` is
    near-black -- the same fill as the groove it slides in.
    """

    changed = Signal(int)

    #: `h-1.5`, `size-4` and `ring-4` -- as PIXELS, because this module works
    #: in the registry's own 16 px root (a Button is `h-9` = `t.control_height`
    #: = 36). The panel-root `rem()` helpers belong to the surfaces that set a
    #: different root, not here.
    TRACK_H = 6
    THUMB = 16
    #: Drawn as a halo rather than an outline, so it grows OUTWARD from the
    #: thumb the way `ring-4` does rather than eating into it.
    RING = 4

    def __init__(self, t, lo: int, hi: int, value: int, parent=None):
        super().__init__(parent)
        self.t = t
        self.lo, self.hi = lo, hi
        self._value = max(lo, min(hi, value))
        self._hover = False
        self._down = False
        self.setMouseTracking(True)
        self.setCursor(Qt.PointingHandCursor)
        # Tall enough for the thumb AND its ring: Qt clips a child's painting
        # to its own rect, so a widget sized to the thumb alone would slice the
        # halo into four arcs.
        self.setFixedHeight(self.THUMB + 2 * self.RING)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def value(self) -> int:
        return self._value

    def set_value(self, v: int) -> None:
        v = max(self.lo, min(self.hi, int(v)))
        if v != self._value:
            self._value = v
            self.update()
            self.changed.emit(v)

    # -- geometry ----------------------------------------------------------- #
    def _span(self) -> float:
        """The travel of the thumb's CENTRE, inset by its own radius so the
        thumb never overhangs the widget at either end."""
        return max(1.0, self.width() - self.THUMB)

    def _frac(self) -> float:
        return (self._value - self.lo) / max(1, self.hi - self.lo)

    def _centre_x(self) -> float:
        return self.THUMB / 2.0 + self._frac() * self._span()

    def _value_at(self, x: float) -> int:
        f = (x - self.THUMB / 2.0) / self._span()
        return round(self.lo + max(0.0, min(1.0, f)) * (self.hi - self.lo))

    # -- input -------------------------------------------------------------- #
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._down = True
            self.set_value(self._value_at(e.position().x()))

    def mouseMoveEvent(self, e):
        if self._down:
            self.set_value(self._value_at(e.position().x()))
        else:
            near = abs(e.position().x() - self._centre_x()) <= self.THUMB
            if near != self._hover:
                self._hover = near
                self.update()

    def mouseReleaseEvent(self, e):
        self._down = False

    def leaveEvent(self, _e):
        if self._hover:
            self._hover = False
            self.update()

    def paintEvent(self, _e):
        t = self.t
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        mid = self.height() / 2.0
        cx = self._centre_x()

        # The track, then the filled range over it. `overflow-hidden` on the
        # track is why the range does not need its own rounding.
        p.setPen(Qt.NoPen)
        track = QRectF(0.0, mid - self.TRACK_H / 2.0,
                       float(self.width()), float(self.TRACK_H))
        r = self.TRACK_H / 2.0
        p.setBrush(QColor(t.muted))
        p.drawRoundedRect(track, r, r)
        if cx > 0:
            p.save()
            path = QPainterPath()
            path.addRoundedRect(track, r, r)
            p.setClipPath(path)
            p.setBrush(QColor(t.primary))
            p.drawRect(QRectF(track.left(), track.top(), cx, track.height()))
            p.restore()

        # `hover:ring-4` at `ring-ring/50`, under the thumb so the thumb's own
        # border stays crisp on top of it.
        if self._hover or self._down:
            p.setBrush(QColor(mix(t.ring, t.background, 0.5)))
            p.drawEllipse(QPointF(cx, mid), self.THUMB / 2.0 + self.RING,
                          self.THUMB / 2.0 + self.RING)

        # `size-4 rounded-full border border-primary bg-white shadow-sm`.
        pen = QPen(QColor(t.primary))
        pen.setWidthF(1.0)
        p.setPen(pen)
        p.setBrush(QColor("#ffffff"))
        p.drawEllipse(QPointF(cx, mid), self.THUMB / 2.0 - 0.5,
                      self.THUMB / 2.0 - 0.5)
        p.end()


class NumberField(Surface):
    """A number you can type OR step -- shadcn's Input, flanked by two buttons.

    shadcn's own registry has no number field (``search "number"`` finds
    nothing in ``@shadcn``); the Basic UI one lives in a third-party registry
    that is not configured here. So this is COMPOSED from the two specs that
    were fetched and verified:

        Input:  h-9 rounded-md border border-input bg-transparent px-3 text-sm
        Button: icon-xs is ``size-6 rounded-md`` with an ``size-3`` svg

    Qt's native ``QSpinBox`` arrows are platform chrome with no shadcn
    equivalent -- two tiny stacked triangles in the system's own style -- so
    they stay off and the steppers are real buttons at a real hit size.
    """

    changed = Signal(int)

    def __init__(self, t, low: int, high: int, value: int, *, width: int = 132,
                 parent=None):
        super().__init__(t, radius=t.radius_md, shadowed=True, parent=parent)
        self.setObjectName("numfield")
        self.setStyleSheet(
            f"QFrame#numfield {{ background:{t.card};"
            f" border:1px solid {t.input};"
            f" border-radius:{t.radius_md:g}px; }}")
        self.setFixedSize(width, t.control_height)

        self._box = QSpinBox()
        self._box.setRange(low, high)
        self._box.setValue(value)
        self._box.setFont(ty.ui("sm", "medium"))
        self._box.setButtonSymbols(QSpinBox.NoButtons)
        self._box.setAlignment(Qt.AlignCenter)
        self._box.setFrame(False)
        self._box.setStyleSheet(
            f"QSpinBox {{ background:transparent; color:{t.foreground};"
            f" border:none; padding:0px; }}")
        self._box.valueChanged.connect(self.changed.emit)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(3, 3, 3, 3)
        lay.setSpacing(0)
        lay.addWidget(self._step("minus", -1))
        lay.addWidget(self._box, 1)
        lay.addWidget(self._step("plus", +1))
        self._fit_width(width)

    #: The two steppers' width, the layout's margins around them and the frame's border (`_step`, the layout above,
    #: the style sheet's 1px border, which the layout's area lies inside).
    _STEPPER_W, _MARGIN, _BORDER = 26, 3, 1

    def _fit_width(self, width: int) -> None:
        """The width asked for, or wider when the box's widest value would not fit: the box's own size hint (its
        range's widest number, decimals and suffix) between the two steppers. Found by the GUI sweep (22 Oct 2026):
        the Composer's font size at 100 px read "7.0 p", its box 40 px wide for the 70 "40.0 pt" needs."""
        self.ensurePolished()
        self._box.ensurePolished()           # the style sheet's font and padding are part of the box's hint
        need = 2 * (self._BORDER + self._MARGIN + self._STEPPER_W) + self._box.sizeHint().width()
        self.setFixedSize(max(int(width), need), self.t.control_height)

    def _step(self, glyph: str, delta: int) -> QWidget:
        t = self.t
        btn = QFrame()
        btn.setObjectName("step")
        # inside the border as well as the margins (the GUI sweep, 22 Oct 2026: 30 px in 28 drew over the border)
        btn.setFixedSize(self._STEPPER_W, t.control_height - 2 * (self._MARGIN + self._BORDER))
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet(
            f"QFrame#step {{ background:transparent; border:none;"
            f" border-radius:{t.radius_sm:g}px; }}"
            f"QFrame#step:hover {{ background:{t.accent}; }}")
        bl = QHBoxLayout(btn)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.addWidget(icon(glyph, icons.SM, t.muted_foreground),
                     alignment=Qt.AlignCenter)
        btn.mouseReleaseEvent = lambda _e, d=delta: self._box.setValue(
            self._box.value() + d)
        return btn

    def value(self) -> int:
        return self._box.value()

    def setValue(self, value: int) -> None:
        self._box.setValue(value)


class FloatField(NumberField):
    """`NumberField` for a DECIMAL: the same Input-between-two-steppers, holding a
    `QDoubleSpinBox`. Added for the preparation window's "flat within %" (0.5 % is where
    One user's MetaMorph floors land), which an integer field cannot hold. ``suffix`` is drawn
    inside the box ("%", " nm")."""

    changed = Signal(float)
    #: Enter pressed or the box left (the spin box's `editingFinished`), for a page that
    #: must hand the keyboard back to something else once a number is typed.
    finished = Signal()

    def __init__(self, t, low: float, high: float, value: float, *, step: float = 0.1,
                 decimals: int = 1, suffix: str = "", width: int = 132, parent=None):
        Surface.__init__(self, t, radius=t.radius_md, shadowed=True, parent=parent)
        self.t = t
        self.setObjectName("numfield")
        self.setStyleSheet(
            f"QFrame#numfield {{ background:{t.card};"
            f" border:1px solid {t.input};"
            f" border-radius:{t.radius_md:g}px; }}")
        self.setFixedSize(width, t.control_height)
        self._box = QDoubleSpinBox()
        # A POINT, whatever the desktop's locale: a German Windows showed "2,0 %" beside
        # every other number in this application written with a point.
        self._box.setLocale(QLocale.c())
        self._box.setDecimals(decimals)
        self._box.setRange(low, high)
        self._box.setSingleStep(step)
        self._box.setValue(value)
        if suffix:
            self._box.setSuffix(suffix)
        self._box.setFont(ty.ui("sm", "medium"))
        self._box.setButtonSymbols(QDoubleSpinBox.NoButtons)
        self._box.setAlignment(Qt.AlignCenter)
        self._box.setFrame(False)
        self._box.setStyleSheet(
            f"QDoubleSpinBox {{ background:transparent; color:{t.foreground};"
            f" border:none; padding:0px; }}")
        self._box.valueChanged.connect(self.changed.emit)
        self._box.editingFinished.connect(self.finished.emit)
        self._step_size = step
        lay = QHBoxLayout(self)
        lay.setContentsMargins(3, 3, 3, 3)
        lay.setSpacing(0)
        lay.addWidget(self._step("minus", -1))
        lay.addWidget(self._box, 1)
        lay.addWidget(self._step("plus", +1))
        self._fit_width(width)

    def _step(self, glyph: str, delta: int) -> QWidget:
        btn = super()._step(glyph, delta)
        btn.mouseReleaseEvent = lambda _e, d=delta: self._box.setValue(
            self._box.value() + d * self._step_size)
        return btn

    def value(self) -> float:
        return float(self._box.value())

    def setValue(self, value: float) -> None:
        self._box.setValue(float(value))
