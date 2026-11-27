"""shadcn's DropdownMenu, as a QMenu — plus the chip that reports a live filter.

Fetched from the registry rather than remembered::

    Content: rounded-md border bg-popover p-1 text-popover-foreground shadow-md
    Item:    px-2 py-1.5 text-sm rounded-sm  focus:bg-accent
    Label:   px-2 py-1.5 text-sm font-medium

**Rows are widgets, not QActions.** A ``QAction`` renders through the platform
style, so a stylesheet can recolour it but cannot put a day badge or a live
count inside it — and both are the reason to open this menu at all. Each row is
therefore a ``QWidgetAction`` wrapping a small painted widget, which also means
the hover fill is ``--accent`` rather than the platform highlight.

**``popup``, never ``exec``.** ``exec`` spins its own event loop and blocks,
which makes the menu untestable and stops the window repainting underneath it.
"""

from __future__ import annotations

import time

from qtpy.QtCore import (
    QEasingCurve,
    QEvent,
    QPoint,
    QPointF,
    QRect,
    QRectF,
    QSize,
    Qt,
    QTimer,
    QVariantAnimation,
    Signal,
)
from qtpy.QtGui import QColor, QCursor, QPainter, QPen
from qtpy.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QMenu,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from ..theme import icons
from ..theme import typography as ty
from .primitives import (Badge, discard, draw_shadow, icon, label, row,
                         shadow_room)

__all__ = [
    "FilterChip",
    "Menu",
    "menu_note",
    "menu_row",
    "menu_section",
    "close_panels_under",
    "retriggered",
    "styled_menu",
]


#: The px root a menu is measured at.
#:
#: This module was written for the 16 px root the onboarding screens use.
#: **`workspace.ROOT_PX` was 18 for a while, and is 16 now -- corrected,
#: not just changed** (a screenshot-measured sidebar width had stood in
#: for the registry's own `16rem`; see `sidebar.ROOT_PX`'s own comment
#: for the full story). Left here as the reason `root=` exists as a
#: parameter at all rather than a fixed constant: a menu popping out of a
#: panel at a DIFFERENT root has to be measured the same way that panel
#: is, or it reads as a smaller menu belonging to a different application
#: -- which is exactly how it read while the mismatch was live, `text-sm`
#: at 14 beside a sidebar whose own `text-sm` was 15. Now that `workspace
#: .ROOT_PX` matches `MENU_ROOT` again, every menu opened from the
#: workspace panel takes the DEFAULT branch below regardless of whether
#: it passes `root=` explicitly -- the parameter stays, for whichever
#: panel is next built at a genuinely different root.
#:
#: One number rather than a pile of size arguments, for the reason the panel
#: itself is built that way: a layout is a set of RATIOS, and moving one
#: without the rest is what makes a thing look stretched rather than scaled.
#: Every default below is the 16 px value, so nothing that does not pass a root
#: changes by a pixel.
MENU_ROOT = 16


def _rem(x: float, root: int = MENU_ROOT) -> int:
    """A Tailwind rem value at a given root."""
    return round(x * root)


def _type_px(rem_: float, root: int) -> int | None:
    """Pixel size for a type class, or ``None`` to keep the ramp's own step.

    TRUNCATED, matching ``workspace.text_px``: at an 18 px root `text-sm` is
    15.75, and rounding it up made the type visibly heavier than the reference.
    """
    return None if root == MENU_ROOT else int(rem_ * root)


class Menu(QMenu):
    """A dropdown that can close its whole chain.

    A submenu row has to dismiss the parent as well as itself, or picking "a
    day" leaves the menu it came from standing open behind the change it just
    made.
    """

    #: `duration-100` -- the registry spells it out on `.cn-popover-content`
    #: and on every menu/dropdown/select surface beside it. Qt's own menu fade
    #: runs ~150 ms and LINEAR; MEASURED here, opacity 0 -> 1 in even steps
    #: between 43 ms and 149 ms after the popup appeared.
    INTRO_MS = 100

    #: `slide-in-from-top-2` -- Tailwind's `2` is 0.5rem.
    SLIDE_PX = 8

    #: The most the zoom may move a surface's far CORNER, in pixels.
    #:
    #: 15 px is what `zoom-in-95` already gives both pickers -- 14.8 and 14.7,
    #: which is why they read alike despite one being twice the height of the
    #: other. See :meth:`_zoom_from`.
    MAX_CORNER_PX = 15.0

    #: `zoom-in-95` -- the surface starts at 95 % and grows into place, and
    #: `zoom-out-95` -- it shrinks back to the same 95 % on the way out.
    ZOOM_FROM = 0.95

    #: `data-closed:animate-out fade-out-0 zoom-out-95`. NO slide: the registry
    #: has `slide-in-from-*` and no `slide-out-to-*`, so the surface fades and
    #: shrinks in place rather than retreating toward its trigger.
    #:
    #: **OFF by default, after four attempts that each still flickered.**
    #:
    #: The intro is easy because it animates a window that stays alive. The
    #: outro has to animate one that is being DESTROYED: the real menu must go
    #: the instant it closes or every dismissal decision above has to stay
    #: correct while it is neither open nor closed. So a stand-in plays it, and
    #: a stand-in is a handoff. What was tried, and what each still showed:
    #:
    #:   1. second window, shown late      ~28 ms with neither on screen
    #:   2. second window, resize+opacity  six DWM ops on a layered window
    #:   3. second window, fixed + painted a window still appears over the page
    #:   4. child overlay of the main window, painted in the page
    #:
    #: (4) is the right shape -- no second window, no z-order, no translucency
    #: -- and it still flickered. What is MEASURED is a page repaint over the
    #: menu's rect at ~5.5 ms and the overlay's own paint ~25 ms later, across
    #: the popup teardown. What is NOT established is that this is the artifact:
    #: Qt does not repaint clean regions, so a missing paint EVENT does not mean
    #: missing pixels, and the difference was never observed, only inferred.
    #:
    #: Qt's own `UI_FadeMenu` was checked as a way out of the handoff entirely,
    #: since it runs on the real window: it only fades menus IN. Measured, a
    #: close with the effect on and off is identical -- opacity 1.0, hidden.
    #:
    #: ON, because an instant disappearance is worse than an imperfect fade.
    #: `OUTRO_MS` is the knob that makes the artifact describable: run
    #: `slowclose.py`, which sets it to two seconds, and whatever is wrong
    #: happens slowly enough to name.
    OUTRO = True
    OUTRO_MS = 100

    def __init__(self, t, parent=None, *, root: int = MENU_ROOT):
        super().__init__(parent)
        self.root = root
        #: Free this menu once it has closed.
        #:
        #: **Every trigger in this prototype builds a FRESH menu on each open**
        #: -- all seven call sites of `styled_menu` -- and none of them was ever
        #: freed. They stayed parented to their trigger for the life of the
        #: app: measured at +24 widgets per open, 3 255 -> 3 735 after twenty,
        #: and the cost that shows is the BUILD, which grew from ~35 ms to
        #: 81 ms. That is the reported "everything goes laggy after opening the
        #: workspace menu": not the animation frames, which stay at 15 ms, but
        #: the click-to-open latency, and it gets worse every time.
        #:
        #: A SUBMENU must not do this -- it hides and re-shows while its parent
        #: stays open, and freeing it would leave `_Row.submenu` dangling. A
        #: submenu is exactly a menu whose parent is a menu (`styled_menu(t,
        #: menu)`), so the distinction needs no flag at the call site, and
        #: deleting the root takes its submenus with it as children.
        #:
        #: Settable, for a caller that means to keep one across opens.
        self.free_on_hide = not isinstance(parent, Menu)
        self._freed = False
        self.t = t
        # Qt's platform fade is switched OFF, or it stacks with ours and the
        # menu takes 250 ms to arrive. This is the ONE animation on the surface.
        app = QApplication.instance()
        if app is not None:
            app.setEffectEnabled(Qt.UI_FadeMenu, False)
            app.setEffectEnabled(Qt.UI_AnimateMenu, False)
        #: Which side of its trigger this opens on, in shadcn's sense --
        #: `data-[side=bottom]` slides in FROM THE TOP, and so on. Set by
        #: whoever pops it; submenus set "right" for themselves.
        self._side = "bottom"
        #: What opened this -- a widget or a global rect. See `retriggered`.
        self._trigger = None
        #: True once the intro has landed -- see `_refresh_ghost`.
        self._intro_done = False
        self._home = None
        #: The size the menu settles at; the intro grows into it.
        self._final = None
        #: The snapshot shown while the window is smaller than its contents.
        self._veil = None
        #: Set in `showEvent`, consumed by the first `paintEvent`.
        self._want_backdrop = False
        #: Kept past the intro, so the outro has a picture to shrink.
        self._shot = None
        #: The overlay that plays the outro, mapped during the intro.
        self._ghost = None
        self._intro = QVariantAnimation(self)
        self._intro.setDuration(self.INTRO_MS)
        # Tailwind's `animate-in` default timing function is plain `ease`,
        # i.e. cubic-bezier(0.25, 0.1, 0.25, 1) -- taken exactly rather than
        # approximated with a named Qt curve, the same as `primitives.tw_ease`.
        curve = QEasingCurve(QEasingCurve.BezierSpline)
        curve.addCubicBezierSegment(QPointF(0.25, 0.1), QPointF(0.25, 1.0),
                                    QPointF(1.0, 1.0))
        self._intro.setEasingCurve(curve)
        self._intro.setStartValue(0.0)
        self._intro.setEndValue(1.0)
        self._intro.valueChanged.connect(self._on_intro)
        #: The row whose submenu is currently showing, or ``None``.
        self._sub_open = None
        pad = _rem(0.25, root)              # `p-1` on the content, `my-1`
        self.setStyleSheet(                 # on the separator
            f"QMenu {{ background:{t.popover}; border:1px solid {t.border};"
            f" border-radius:{t.radius_md:g}px; padding:{pad}px; }}"
            f"QMenu::item {{ background:transparent; padding:0px; }}"
            f"QMenu::separator {{ height:1px; background:{t.border};"
            f" margin:{pad}px 0px; }}")

    def paintEvent(self, e):
        """Fill the whole window BEFORE the rounded panel is drawn, and take
        the outro's backdrop once there is something on screen.

        The dark blobs this removes were not a shadow and not the page showing
        through -- they were the corner cut-outs of ``border-radius`` rendering
        as near-black, because **this popup composites alpha against black**
        rather than against what is behind it. Measured: a stylesheet fill of
        ``rgba(255,0,0,0.5)`` over a WHITE host renders ``[38, 25, 25]``, where
        real translucency would give roughly ``[255, 128, 128]``.

        So every pixel the rounded box does not cover comes out dark, and the
        artifact scales with the radius -- present at any radius above 0, and
        already there before this window's radius was corrected::

            radius  0   BR 232   clean
            radius  4   BR 130
            radius  6   BR 120   <- the value this prototype used to ship
            radius  8   BR 120, TR 205, BL 199
            radius 12   BR 120, TR 158, BL 154

        None of the usual suspects were involved: the window class carries no
        ``CS_DROPSHADOW`` bit, ``NoDropShadowWindowHint`` changes nothing, the
        DWM corner preference changes nothing, and a rounded window mask makes
        it worse (all four corners to ~70). Clearing to transparent first makes
        it worse still (down to 2), which is what pinned the cause.

        Filling opaquely first means the corners show the PAGE colour instead
        of black. That is not what is literally behind the popover -- over a
        dark element you would see a pale notch rather than a dark one -- but
        these popovers open over the page background, where it is invisible.
        """
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(self.t.background))
        p.end()
        super().paintEvent(e)

    # -- hover and submenus are driven from HERE, not from the rows ---------- #
    #
    # A row used to open its own submenu in ``enterEvent``. That works exactly
    # once: a QMenu popup GRABS THE MOUSE, so from the moment the submenu
    # appears the parent's rows stop receiving enter and leave events at all.
    # The symptoms were all one cause -- the parent menu felt locked (no other
    # row would light up), and the row that opened the submenu kept its
    # highlight forever, so clicking a second row left two rows lit.
    #
    # Qt's own submenu machinery is not available here: these are
    # ``QWidgetAction`` rows, so ``QMenu`` never tracks an active action for
    # them. The menu therefore watches the POINTER instead. An application-wide
    # filter sees mouse moves wherever the grab sends them, and the menu
    # resolves them against its own rows by global geometry.
        # AFTER the paint, and only once: the first pixel of the menu is on
        # screen by the time this runs, so the grab's cost is no longer in
        # front of it. Still well before the intro lands, which is the only
        # thing that needs it.
        if self._want_backdrop:
            self._want_backdrop = False
            QTimer.singleShot(0, self._grab_backdrop)

    def _zoom_from(self) -> float:
        """`zoom-in-95`, CAPPED IN PIXELS rather than left proportional.

        The one deliberate departure from the registry's motion, and it answers
        two reports at once. A flat duration with a PROPORTIONAL zoom means a
        tall surface's edges sweep further in the same time -- measured on the
        three menus here, where the intro is frame-for-frame identical (8
        frames, ~16 ms each) and only the geometry differs::

            menu      height   zoom grows   +slide   edge speed
            Filter       257       12.9px   20.9px    209 px/s
            palette      151        7.6px   15.6px    156 px/s
            swatch        82        4.1px   12.1px    121 px/s

        1.7x between tallest and shortest, which is why the biggest menu reads
        as faster than the pickers while every number in the code says they are
        the same. shadcn has this too -- `duration-100` on a `zoom-in-95`
        behaves identically on the web -- so this is a deviation, not a fix.

        **Cap the DISTANCE, do not stretch the TIME.** Scaling the duration was
        tried first and traded one complaint for two: the menu then read as
        slow, and its blur got worse. A raster of text scaled off the pixel grid
        is softer than one at rest -- MEASURED, mean local contrast 8.3 at scale
        1.0 against 6.8-7.2 at every intermediate scale, and NOT monotonic in
        the scale distance, so a gentler zoom does not sharpen the frames. All a
        longer duration does is leave those softer frames on screen 35 % longer.
        Distance moves the speed without touching the exposure.

        **On the CORNER, not on the height** -- and the first cut got that
        wrong in a way worth keeping. Capping the height alone shrinks the
        horizontal growth by the same factor, so the tall menu ended up moving
        LESS than the pickers rather than the same::

            menu        size      zoom   grows W   grows H    corner
            Filter   256x257     0.971     7.5px     7.5px    10.6px   <- 28 % under
            palette  256x151     0.950    12.7px     7.5px    14.8px
            swatch    282x82     0.950    14.1px     4.1px    14.7px

        The two surfaces that read correctly agree on ONE number and it is the
        corner: 14.8 and 14.7, despite one being nearly twice the height of the
        other and the other being wider. That is the invariant to hold, and
        neither axis on its own is.

        Never MORE than the registry's 95 %: only a surface whose corner would
        overshoot the cap gets a gentler zoom, and both pickers sit just under
        it, so they are untouched.
        """
        w, h = self._width(), self._height()
        corner = (w * w + h * h) ** 0.5
        if corner <= 0:
            return self.ZOOM_FROM
        return max(self.ZOOM_FROM, 1.0 - self.MAX_CORNER_PX / corner)

    def _width(self) -> float:
        settled = self._final if self._final is not None else self.sizeHint()
        return float(settled.width())

    def _height(self) -> float:
        settled = self._final if self._final is not None else self.sizeHint()
        return float(settled.height())

    def set_trigger(self, target) -> None:
        """The control that opens this, so a second click on it can close it."""
        self._trigger = target

    def popup_under(self, trigger, align: str = "start", offset: int = 4) -> None:
        """Open under ``trigger``, aligned to its START or its END edge.

        ``align="end"`` is Radix's own prop, and it is what a trigger sitting
        against the right margin needs: the surface's right edge meets the
        trigger's right edge. WITHOUT it the popup grows rightward, runs past
        the screen, and Qt CLAMPS it -- so it still appears, at an arbitrary
        offset from the control it belongs to, which reads as a misplaced
        popover rather than as a surface that ran out of room.

        ``offset`` is Radix's ``sideOffset``, default 4 (checked against the
        registry, not remembered).
        """
        rect = trigger.rect()
        below = trigger.mapToGlobal(rect.bottomLeft()).y() + offset
        # POLISH AND SIZE FIRST. An unshown `QMenu` full of `QWidgetAction`s
        # reports a `sizeHint` from before its rows are laid out -- MEASURED at
        # 599 px against the 256 it opens at, which put the surface 343 px left
        # of where it belonged.
        self.ensurePolished()
        self.adjustSize()
        if align == "end":
            # `topRight` is INCLUSIVE in Qt -- x is left + width - 1 -- so the
            # exclusive right edge is one past it.
            right = trigger.mapToGlobal(rect.topRight()).x() + 1
            x = right - self.width()
        else:
            x = trigger.mapToGlobal(rect.topLeft()).x()
        self.popup(QPoint(x, below))

    def set_side(self, side: str) -> None:
        """Where this sits relative to its trigger: bottom / top / left / right."""
        self._side = side

    def _offset(self, amount: float) -> QPoint:
        """`slide-in-from-*`: the popover starts pushed AWAY from its trigger."""
        dx = dy = 0
        if self._side == "bottom":      # below the trigger -> in from the top
            dy = -amount
        elif self._side == "top":
            dy = amount
        elif self._side == "right":     # to the right -> in from the left
            dx = -amount
        elif self._side == "left":
            dx = amount
        return QPoint(round(dx), round(dy))

    def _on_intro(self, value) -> None:
        """One frame of `fade-in-0 zoom-in-95 slide-in-from-*-2`."""
        t = float(value)
        self.setWindowOpacity(t)        # `fade-in-0`
        if self._home is None or self._final is None:
            return
        # `zoom-in-95`, done by resizing the WINDOW rather than the widgets.
        #
        # A Qt popup cannot scale its children -- resize it and they relayout,
        # so at 95 % the rows clip instead of shrinking. So the window really
        # does become 95 % and a SNAPSHOT of the finished menu is painted into
        # it, scaled. The children are covered for the 100 ms it takes; they are
        # laid out at whatever size the window happens to be, and nobody sees
        # it. This is the `GlideSurface` trick from `smoothscroll`, which exists
        # for the same reason: Qt has no transform for a live widget tree.
        low = self._zoom_from()
        scale = low + (1.0 - low) * t
        w = max(1, round(self._final.width() * scale))
        h = max(1, round(self._final.height() * scale))
        # The ORIGIN is the corner nearest the trigger, which is what
        # `origin-(--radix-popover-content-transform-origin)` resolves to: a
        # menu below its trigger grows from its top-left, a submenu to the
        # right grows from its left edge. Both are the top-left here, so the
        # position is just the slide -- the box grows down and right from it.
        at = self._home + self._offset(self.SLIDE_PX * (1.0 - t))
        self.setGeometry(at.x(), at.y(), w, h)
        if self._veil is not None:
            self._veil.setGeometry(0, 0, w, h)
            self._veil.raise_()
            self._veil.update()
        if t >= 1.0:
            self._freeze_content(False)
            self._land()
            # The intro is over, so the overlay's rect now matches what is on
            # screen; before this the menu is smaller and offset, and a painted
            # overlay would show around it.
            self._intro_done = True
            self._refresh_ghost()

    def _freeze_content(self, on: bool) -> None:
        """Hold the action widgets' painting for the length of the intro.

        NOT `self.setUpdatesEnabled` -- that would freeze the veil too, and the
        veil is the only thing on screen. Targeted at the widgets the veil is
        covering.
        """
        for act in self.actions():
            w = act.defaultWidget() if hasattr(act, "defaultWidget") else None
            if w is not None:
                w.setUpdatesEnabled(not on)

    def _spawn_ghost(self) -> None:
        """Start the outro, if the overlay has a picture to fade.

        Nothing is grabbed or painted here, and that is the point: painting
        during a close does not work (`repaint()` inside `hideEvent` draws
        nothing, measured) and an overlay that is empty at that moment IS the
        flicker -- the menu vanishing, a gap, then the picture appearing to
        fade. The overlay has been holding the current picture since the intro
        landed; see `_refresh_ghost`.
        """
        ghost, self._ghost = self._ghost, None
        # `_disposed` is a plain PYTHON attribute, so it is still readable when
        # the C++ object behind the wrapper is gone -- which is the case this
        # guard is for. `end_outros()` disposes an overlay a menu still points
        # at, and by the time that menu hides, `self._ghost` is a wrapper over
        # nothing: passing it to `singleShot` as a context object raised
        # `Internal C++ object (_Ghost) already deleted` from inside
        # `hideEvent`, on stderr, with the process still exiting 0.
        if ghost is None or ghost._disposed:
            return
        # THE PAGE, grabbed HERE and not in `_refresh_ghost`.
        #
        # The backdrop makes the overlay opaque, which is what keeps its frame
        # rate even -- but it is the expensive half. `_refresh_ghost` runs on
        # every pointer move over an open menu, and a host grab there cost
        # 8.4 ms each time; it pushed the INTRO out from ~120 ms to 248 ms,
        # caught by the check on how long the intro takes to land. Here it runs
        # once per close, and measured 1.0 ms -- the host is warm by then.
        #
        # Deferring it costs nothing in freshness: the menu is its own window so
        # the host's grab never contains it, and an action that changes the page
        # has already run (`menu_row` calls `on_click()` before `_close`).
        # EVERY path lets go of the overlay, or it outlives the menu. It is
        # built on OPEN now, so one that never gets to play -- the outro turned
        # off, or a menu closed before its intro landed -- would sit in
        # `_GHOSTS` for the life of the app. Measured as eight live overlays
        # after eight open/close cycles, fast or slow.
        if (not self.OUTRO or self._home is None or self._final is None
                or not ghost.armed()):
            ghost.dispose()
            return
        # SHOWN ALL ALONG, FADED LATER. Qt destroying the native popup blocks
        # the loop for ~28 ms (measured: `close()` 5.1 ms, the first
        # `processEvents` after it 28.1 ms). Nothing can be drawn during it, so
        # the fade waits for the loop to come back -- started inline it runs on
        # wall clock through the block and its first visible frame lands at
        # 0.64, with the head of the curve never painted.
        # `singleShot(ms, context, slot)` -- the THREE-argument form, so the
        # timer is dropped if the ghost dies before it fires. It can: a rail
        # popover replacing another calls `end_outros()`, which disposes this
        # overlay in the same turn it was spawned, and the two-argument form
        # then fired into a deleted C++ object (`RuntimeError` out of
        # `QVariantAnimation.setDuration`, on stderr, exit code still 0 --
        # which is exactly how a real fault hides).
        QTimer.singleShot(0, ghost, lambda: ghost.run(
            self.OUTRO_MS, self._intro.easingCurve()))

    def _refresh_ghost(self) -> None:
        """Keep the outro's overlay holding the menu's CURRENT picture.

        **This is the flicker fix, and it is about WHEN, not what.** The picture
        cannot be painted during the close: `repaint()` inside `hideEvent` draws
        nothing (measured, zero paint events), so an overlay that only receives
        its picture at close time is EMPTY for the frame the menu vanishes on.
        Described from the screen at a two-second outro: "the menu suddenly
        disappears and reappears, and after that the close animation works
        smoothly" -- a gap, then the fade, exactly.

        So it is painted while the menu is still up and covering it, where the
        paint does land and nobody sees it. Refreshed on every appearance change
        rather than once, because a menu is rarely closed in the state it opened
        in -- a row is hovered, a submenu trigger is lit.
        """
        ghost = self._ghost
        if ghost is None or not self.isVisible() or not self._intro_done:
            return
        shot = self.grab()
        if shot.isNull() or shot.width() < 2:
            return
        ghost.hold(shot, self._zoom_from())
        # The page, ONCE per menu-open and never per pointer move.
        #
        # This is the expensive half -- a host grab measured 8.4 ms, and
        # `_refresh_ghost` runs on every move over an open menu, so doing it
        # here every time pushed the INTRO from ~120 ms to 248 ms (caught by
        # the check on how long the intro takes to land). Doing it only at
        # CLOSE was worse still: `WA_OpaquePaintEvent` set on an
        # already-mapped widget does not take, and the pacing of all three
        # menus fell to ~22 ms a frame.
        #
        # So it is grabbed the first time the overlay takes a picture, which
        # sets the attribute while there is still time for Qt to honour it,
        # and refreshed cheaply at close by `_spawn_ghost`.


    def _grab_backdrop(self) -> None:
        """Give the overlay the page beneath it, if both still exist.

        Only ever runs while the overlay is empty -- see the call site.
        """
        ghost = self._ghost
        if ghost is None or not ghost.needs_backdrop() or ghost.armed():
            return
        try:
            ghost.set_backdrop(self._page_under())
        except RuntimeError:
            pass                # the menu went away before this ran

    def _page_under(self):
        """The page beneath this menu, as a pixmap, or ``None``.

        The menu is its own window, so the host's `grab` never contains it --
        what comes back is exactly what the shrinking picture will uncover.
        """
        host, ghost = self._host(), self._ghost
        if host is None or ghost is None:
            return None
        # The OVERLAY's rect, which is the SNAPPED one -- grabbing the menu's
        # own rect instead puts the copy back at the wrong sub-pixel phase.
        return host.grab(ghost.geometry())

    def _host(self):
        """The main window this menu belongs to -- where the outro is drawn."""
        owner = self.parentWidget()
        return owner.window() if owner is not None else None

    def _land(self) -> None:
        """Put the real widgets back and drop the snapshot."""
        if self._veil is not None:
            self._veil.hide()
            discard(self._veil)
            self._veil = None
        self.setWindowOpacity(1.0)
        if self._home is not None and self._final is not None:
            # `resize` + `move`, NOT `setGeometry`. Landing usually happens on
            # the last intro frame, where either works -- but it also happens
            # from `hideEvent`, when a menu is dismissed part-way through, and
            # MEASURED there: `setGeometry` on a window already being hidden
            # does not take, leaving the widget parked at 243x244 with the next
            # `showEvent` about to read that as its full size. `resize` does
            # take while hidden.
            self.resize(self._final)
            self.move(self._home)

    def remeasure(self) -> None:
        """Tell this menu its widget actions may have changed size.

        **A `QMenu` caches the rectangle of every action and has no public way
        to say that cache is stale.** It is marked dirty by `actionEvent` --
        an action added, removed or CHANGED -- and by nothing else. A
        `QWidgetAction` whose widget grows while the menu is hidden therefore
        goes unnoticed: `sizeHint()` keeps answering with the old rects.

        That only bites a menu that is REUSED, which is exactly what
        `free_on_hide = False` makes the settings popover. Measured, alternating
        a one-control subsection with a three-control one::

            holder (the widget) wants   320 px      <- correct
            menu.sizeHint()             198 px      <- the previous open's
            what the user saw           clipped, the Run button cut off

        `showEvent` then reads that stale hint into `_final` and `_land()`
        resizes to it, so the popover opened **one open behind** its content --
        and `sizeHint()` read correctly a moment later, once the show had
        dirtied the cache, which is what made it look like a paint bug.

        The poke is `QAction::changed`, the signal `QWidget::actionEvent`
        listens to and the only public route to that dirty flag. `iconText` is
        the field with nothing behind it: a `QWidgetAction` draws its widget and
        nothing else, so no icon, no text and no metric depends on it. Round
        trip measured at **0.10 ms**, against **11.9 ms** for the honest-looking
        alternative of removing and re-adding the action -- which would hand
        back most of the 36 ms -> 0.3 ms that reusing the menu bought.
        """
        for action in self.actions():
            if action.defaultWidget() is not None:
                action.setIconText("\u200b")
                action.setIconText("")

    def showEvent(self, e):
        super().showEvent(e)
        # `data-open:animate-in fade-in-0 slide-in-from-*-2`, at `duration-100`.
        #
        # NOT `zoom-in-95`, and that is the one part of the registry's opening
        # animation this does not reproduce: a Qt popup cannot scale itself
        # without relaying out its child widgets, which at 95 % would clip rows
        # rather than shrink them. Fade and slide are exact; the missing 5 %
        # grow is recorded here rather than faked.
        self._intro_done = False
        self._home = self.pos()
        # The settled size, from `sizeHint()` rather than from `size()`.
        #
        # `size()` is wrong at exactly one moment, and it is a moment users hit:
        # a menu dismissed PART-WAY through its intro is left at 95 %, and
        # neither `setGeometry` nor `resize` inside `hideEvent` takes on a
        # window already being hidden -- MEASURED, both, which is why this is
        # fixed here on the read rather than there on the write. The next
        # `popup()` would otherwise read 243x244 as full size, and the menu
        # would open small and stay small.
        #
        # `sizeHint()` is recomputed from the actions each time, and measured
        # equal to the settled size (256x257 both ways). The one case it would
        # not be is a menu taller than the screen, which Qt clamps; none here
        # comes close, and a clamped menu would grow past the screen edge for
        # 100 ms rather than break.
        self._final = self.sizeHint()
        # Grabbed AFTER `super().showEvent`, so the rows are laid out and the
        # rounded panel from `paintEvent` is in the picture. The veil is a
        # child, so it inherits the popup's window and needs no compositing --
        # which matters here, where a translucent popup renders against black.
        self._shot = self.grab()
        self._veil = _Veil(self, self._shot)
        # AND STOP THEM PAINTING while the veil is over them.
        #
        # The comment above is right that nobody sees the children during the
        # intro -- but they were still being repainted through all of it. The
        # window is resized on every frame, which relayouts and redraws the
        # whole content behind a snapshot that is covering it. MEASURED on the
        # heaviest popover here (three segmented controls, a slider, a note and
        # a button): 375 widget paints in the 420 ms after the press, of which
        # 143 were QLabels redrawing themselves invisibly.
        #
        # `setUpdatesEnabled(False)` stops the paints and leaves the layout
        # alone, which is the half that matters: the layout is what keeps the
        # children at the size `_land` hands them.
        self._freeze_content(True)
        # The outro's overlay is BUILT AND MAPPED NOW, empty and invisible, in
        # the page behind this menu. See `_Ghost.__init__`.
        host = self._host()
        if self.OUTRO and self._ghost is None and host is not None:
            # A child cannot be drawn outside its parent, so a menu hanging off
            # the window edge gets no outro rather than a clipped one.
            spot = QRect(host.mapFromGlobal(self._home), self._final)
            step = _Ghost.snap_step(host)
            if host.rect().contains(spot.adjusted(-step, -step, step, step)):
                self._ghost = _Ghost(host, spot)
                # THE PAGE, GRABBED NOW -- while the overlay is still EMPTY.
                #
                # `host.grab()` renders the host AND ITS CHILDREN, and the
                # overlay is one of them. Taken any later it comes back holding
                # the menu's own picture, and the outro then draws a full-size
                # menu (the backdrop) with a shrinking one over it: "slightly
                # squeezed, fully expanded, slightly squeezed" until it
                # vanishes. An empty overlay paints nothing
                # (`WA_NoSystemBackground`), so before the first `hold()` the
                # grab is clean.
                #
                # Deferred until AFTER THE MENU HAS PAINTED, not merely a
                # turn -- see `paintEvent`. A turn was not enough: `singleShot(0)`
                # still runs before the first paint, so the grab sat squarely
                # between the click and anything appearing.
                #
                # MEASURED, opening the workspace switcher on a freshly shown
                # screen: this grab costs **126.9 ms the first time**, 11.8 ms
                # the second and ~1.2 ms after -- it is Qt rendering that widget
                # tree into an offscreen surface for the first time, and the
                # region is only 345x320, so it is not the size. Click to first
                # pixel was 168 ms on that first open against 23 ms once warm,
                # which is exactly the "the first menu is not snappy" report.
                #
                # It is the OUTRO's backdrop: nothing needs it until the intro
                # lands, ~100 ms later. Moving it behind the first paint takes
                # it off the path between the click and the menu appearing.
                self._want_backdrop = True
        self._veil.show()
        self._veil.raise_()
        self._on_intro(0.0)
        self._intro.setDuration(self.INTRO_MS)
        self._intro.stop()
        self._intro.start()
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

    def hideEvent(self, e):
        # The outro is handed to a stand-in BEFORE anything else, so the menu
        # itself goes away on exactly the schedule it always did. See `_Ghost`.
        self._spawn_ghost()
        # Land it: a menu hidden mid-intro would come back at 0.4 opacity,
        # two-thirds size and offset, the next time Qt reuses the widget.
        self._intro.stop()
        self._land()
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        self._close_sub()
        for r in self._rows():
            r.set_hover(False)
        super().hideEvent(e)
        # AFTER `super()`, and only once. The outro itself is safe either way:
        # `_Ghost` is a child of the HOST window, not of this menu.
        #
        # DEFERRED past the outro rather than deleted here, because the click
        # that closes a menu is still being DELIVERED at this point -- press
        # closes the popup, release follows -- and freeing inside that sequence
        # hands the rest of it a dead object. Qt's own dispatch holds a
        # QPointer and survives it; anything driving the widget directly does
        # not. The leak is bounded either way: what mattered was that the menus
        # lived FOREVER.
        #
        # `singleShot(ms, context, slot)` and not the two-argument form: with a
        # context object the timer is dropped if the menu is destroyed first
        # (the window closing takes it), instead of firing into a deleted C++
        # object.
        if self.free_on_hide and not self._freed:
            self._freed = True
            QTimer.singleShot(self.OUTRO_MS + 100, self, self.deleteLater)

    def _chain_rects(self):
        """This menu and every submenu standing open under it, in GLOBAL
        coordinates -- which a top-level popup's ``geometry()`` already is."""
        yield self.geometry()
        row = self._sub_open
        while row is not None and row.submenu is not None and row.submenu.isVisible():
            sub = row.submenu
            yield sub.geometry()
            row = getattr(sub, "_sub_open", None)

    def eventFilter(self, obj, event):
        if event.type() is QEvent.MouseButtonPress and self.isVisible():
            at = getattr(event, "globalPosition", None)
            gpos = at().toPoint() if at is not None else QCursor.pos()
            # Remember a press on our OWN trigger, whatever else happens next.
            # Qt is about to dismiss this menu and then hand the same click to
            # the trigger; `retriggered` is how the opener knows to sit it out.
            if self._trigger is not None:
                rect = _global_rect(self._trigger)
                if rect.contains(gpos):
                    _DISMISSED["rect"] = rect
                    _DISMISSED["at"] = time.monotonic()
        if (event.type() is QEvent.MouseButtonPress and self.isVisible()
                and self._sub_open is not None):
            # ONE CLICK DISMISSES THE WHOLE CHAIN.
            #
            # These submenus are independent popups shown with `popup()`, not
            # Qt's own `addMenu` children (they cannot be -- the rows are
            # QWidgetActions with painted hover, see the note above). Qt's
            # dismissal only ever closes the TOPMOST popup, so a click away with
            # a submenu up closed the submenu and left its parent standing --
            # MEASURED at two clicks to clear, which is the whole bug.
            #
            # ``self._sub_open is not None`` is doing real work, and the first
            # version without it BROKE something: every menu ran this, so a
            # submenu dismissed ITSELF on a press over its parent. Qt was
            # already closing it, and the two dismissals together took the
            # parent down with them -- so clicking one submenu trigger while
            # another was open closed the entire menu instead of switching
            # submenus. MEASURED both ways (`probe_trigger.py`): without the
            # guard 2 menus -> 0, with it 2 -> 1, which is what the unfixed
            # code did and what should happen.
            #
            # So this intervenes ONLY in the case Qt gets wrong -- a menu with a
            # submenu standing open -- and leaves the ordinary single popup to
            # Qt, which has always handled it correctly.
            #
            # A press inside ANY menu in the chain is not a dismissal either: it
            # is a row being picked or a trigger being clicked, and both close
            # what they need to on their own.
            at = getattr(event, "globalPosition", None)
            gpos = at().toPoint() if at is not None else QCursor.pos()
            if not any(r.contains(gpos) for r in self._chain_rects()):
                self.close()        # `hideEvent` takes the submenu with it
            return False
        if (event.type() in (QEvent.MouseMove, QEvent.HoverMove)
                and self.isVisible()):
            # The EVENT's position, not ``QCursor.pos()``. The cursor global is
            # a second, independent read that can disagree with the move being
            # handled, and it makes the whole path untestable -- a synthesised
            # move would be resolved against wherever the real mouse happens to
            # be sitting.
            at = getattr(event, "globalPosition", None)
            self._sync(at().toPoint() if at is not None else QCursor.pos())
        return False

    def _rows(self) -> list:
        out = []
        for action in self.actions():
            if isinstance(action, QWidgetAction):
                w = action.defaultWidget()
                if isinstance(w, _Row):
                    out.append(w)
        return out

    def _close_sub(self) -> None:
        row, self._sub_open = self._sub_open, None
        if row is not None and row.submenu is not None:
            row.submenu.close()

    def _sync(self, gpos) -> None:
        """Point the highlight and the open submenu at wherever the mouse is."""
        open_row = self._sub_open
        if open_row is not None and open_row.submenu is not None:
            sub = open_row.submenu
            # Inside the submenu: the trigger stays lit, which is shadcn's
            # ``data-[state=open]:bg-accent`` on DropdownMenuSubTrigger, and
            # the submenu is its own Menu looking after its own rows.
            if sub.isVisible() and sub.geometry().contains(gpos):
                return

        hit = None
        for r in self._rows():
            if QRect(r.mapToGlobal(QPoint(0, 0)), r.size()).contains(gpos):
                hit = r
                break

        # CLOSE FIRST, then light. The other order leaves the outgoing trigger
        # lit: it is still ``_sub_open`` with a visible submenu at the moment
        # the highlights are computed, so it passes the "keep me lit" test and
        # nothing clears it afterwards -- two rows highlighted at once, which
        # is exactly the bug this method exists to fix.
        if hit is not None and self._sub_open is not None and self._sub_open is not hit:
            self._close_sub()

        # A row that has run out of pointer keeps its highlight only while its
        # own submenu is still showing -- shadcn's
        # ``data-[state=open]:bg-accent`` on DropdownMenuSubTrigger.
        for r in self._rows():
            lit = r is hit or (r is self._sub_open and r.submenu is not None
                               and r.submenu.isVisible())
            r.set_hover(lit)

        # The appearance just changed, so the outro's picture is out of date.
        # Cheap: `_refresh_ghost` grabs only when a menu is open and settled.
        self._refresh_ghost()

        if hit is None:
            return                      # off the rows entirely: leave the
            #                             submenu alone rather than snatching
            #                             it away mid-travel toward it
        if hit.has_submenu and self._sub_open is not hit:
            hit.set_hover(True)
            # HERE is where a lazy submenu is built -- the first hover of the
            # row that owns it, which is the first moment it can be seen.
            sub = hit.submenu
            sub.set_side("right")           # it opens to the RIGHT of the row
            sub.popup(hit.mapToGlobal(QPoint(hit.width() - 4, -4)))
            self._sub_open = hit

    def close_chain(self) -> None:
        node = self
        while isinstance(node, QMenu):
            node.close()
            node = node.parent()


class Panel(QWidget):
    """A popover drawn IN THE PAGE, as a child of the main window.

    **Why it is not a `QMenu`.** A popup is an OS window and DWM shadows one on
    its own terms -- outside the window and darkest against its edge, so it
    lands ON the border. MEASURED over a white page: the 1 px `--border`
    (`#E4E4E7`, 228) rendered at **167** against the browser's 234. Six levers
    failed to remove it -- `NoDropShadowWindowHint`, the `CS_DROPSHADOW` class
    bit (cleared and verified, on first and later popups alike), the DWM
    non-client policy, a window region, `DWMWA_WINDOW_CORNER_PREFERENCE`, and a
    layered window -- and the window carries no shadow-inducing style at all
    (`WS_POPUP | WS_VISIBLE | WS_CLIPCHILDREN`, no border, no frame). A child
    widget has no OS surface, so there is nothing to shadow: measured border
    **234**, which is the browser's number.

    **THE ANIMATION IS `Menu`'s, PORTED RATHER THAN APPROXIMATED.** A first
    attempt drove a `QGraphicsEffect` that scaled the RENDERED result. Every
    number matched `Menu` frame for frame -- same curve, same 0.95, same 8 px,
    same duration -- and it still read as a different animation: reported as
    "it just plainly opens top to bottom" against `Menu`'s "gradually zoom from
    left to right", with no close animation at all. Identical numbers, different
    motion, because `Menu` RESIZES ITS SURFACE so the box visibly grows wider,
    while a 5 % transform of a bitmap disappears next to an 8 px slide.

    So this does what `Menu._on_intro` does: hold a snapshot over the frozen
    rows, resize the widget for real, and let the snapshot fill it. `_Veil` is
    reused unchanged.

    THE COST, stated rather than found later: it cannot leave the main window,
    so a popover near an edge is clamped inside rather than overflowing.
    """

    closed = Signal()

    #: Taken from `Menu` rather than restated, so the two cannot drift. They
    #: are hand-tuned values; see `Menu.INTRO_MS`.
    INTRO_MS = Menu.INTRO_MS
    OUTRO_MS = Menu.OUTRO_MS
    SLIDE_PX = Menu.SLIDE_PX
    ZOOM_FROM = Menu.ZOOM_FROM
    MAX_CORNER_PX = Menu.MAX_CORNER_PX

    def __init__(self, t, host, *, root: int = MENU_ROOT, parent_panel=None,
                 animated: bool = True):
        super().__init__(host)
        self.t = t
        #: False opens and closes at once, with no veil, zoom or fade. Opt-in
        #: per instance, so every other popover keeps its animation.
        self._animated = animated
        self._root = root
        self._parent_panel = parent_panel
        self._trigger = None
        self._children = []
        self._rows = []
        self._room = shadow_room(t, t.shadow_md) + 2
        self._veil = None
        self._shot = None
        self._final = None
        self._home = None
        self._closing = False
        self._opacity = 1.0
        #: Re-measured while open when the content changes (`follow_content`).
        self._follows = False
        #: Which corner the surface grows OUT of, i.e. which edge stays put
        #: while it scales. Radix's `origin-(--radix-popover-content-transform
        #: -origin)` resolves to the corner nearest the trigger, so a popover
        #: aligned to its trigger's right edge grows right-to-left.
        #: `popup_under` sets it from `align`.
        self._anchor = "start"
        #: Which side of its trigger this sits on, which decides the direction
        #: `slide-in-from-*` comes from. A submenu opens to the "right" and so
        #: slides in from the left; a menu under its trigger slides from the
        #: top. Same vocabulary as `Menu.set_side`.
        self._side = "bottom"
        #: The row whose submenu is showing, or None.
        self._sub_open = None
        #: A SUBMENU IS REUSED, NOT DESTROYED -- `Menu.free_on_hide` makes the
        #: same distinction with the same test. `_Row` caches the submenu it
        #: built, so deleting one on close leaves that cache pointing at a dead
        #: C++ object and the next hover raises
        #: `Internal C++ object (Panel) already deleted` from `adjustSize`.
        #: Only a top-level panel frees itself.
        self._free_on_hide = parent_panel is None
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        pad = _rem(0.25, root)
        lay = QVBoxLayout(self)
        inset = self._room + pad
        lay.setContentsMargins(inset, inset, inset, inset)
        lay.setSpacing(0)
        self._lay = lay
        self._anim = QVariantAnimation(self)
        curve = QEasingCurve(QEasingCurve.BezierSpline)
        curve.addCubicBezierSegment(QPointF(0.25, 0.1), QPointF(0.25, 1.0),
                                    QPointF(1.0, 1.0))
        self._anim.setEasingCurve(curve)
        self._anim.valueChanged.connect(self._on_frame)
        self._anim.finished.connect(self._on_finished)
        self.hide()

    # -- building ------------------------------------------------------------ #

    def add_widget(self, w):
        self._lay.addWidget(w)
        self._rows.append(w)

    def addSeparator(self):
        """`QMenu::separator` as a widget: 1 px of `--border`, `my-1`."""
        pad = _rem(0.25, self._root)
        line = QWidget()
        line.setFixedHeight(1)
        line.setStyleSheet("background:%s;" % self.t.border)
        self._lay.addSpacing(pad)
        self._lay.addWidget(line)
        self._lay.addSpacing(pad)
        self._rows.append(line)

    def set_trigger(self, target):
        self._trigger = target

    def panel_rect(self):
        r = self._room
        return self.rect().adjusted(r, r, -r, -r)

    # -- painting ------------------------------------------------------------ #

    def paintEvent(self, _e):
        # NOTHING while the veil is up: it holds the finished picture and paints
        # it at whatever size this widget currently is, opacity included.
        if self._veil is not None:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        box = QRectF(self.panel_rect())
        radius = self.t.radius_md
        draw_shadow(p, box, radius, self.t, self.t.shadow_md)
        p.setPen(QPen(QColor(self.t.border), 1.0))
        p.setBrush(QColor(self.t.popover))
        p.drawRoundedRect(box.adjusted(0.5, 0.5, -0.5, -0.5), radius, radius)
        p.end()

    # -- the animation, ported from `Menu._on_intro` ------------------------- #

    def _zoom_from(self):
        """`zoom-in-95`, capped so the far corner never travels more than
        `MAX_CORNER_PX`. `Menu._zoom_from` records the measurements."""
        if self._final is None:
            return self.ZOOM_FROM
        w = float(self._final.width() - 2 * self._room)
        h = float(self._final.height() - 2 * self._room)
        corner = (w * w + h * h) ** 0.5
        if corner <= 0:
            return self.ZOOM_FROM
        return max(self.ZOOM_FROM, 1.0 - self.MAX_CORNER_PX / corner)

    def _freeze(self, on):
        for w in self._rows:
            try:
                w.setUpdatesEnabled(not on)
            except RuntimeError:
                pass

    def _start(self, closing):
        if not self._animated:
            # `popup` has already placed and shown the panel at its final
            # size, so opening needs nothing; closing goes straight to
            # disposal.
            if closing:
                self._dispose()
            return
        self._closing = closing
        if self._final is None or not closing:
            self._final = self.size()
        if self._veil is None:
            self._shot = self.grab()
            self._veil = _Veil(self, self._shot)
            self._veil.show()
            self._veil.raise_()
            self._freeze(True)
        self._anim.stop()
        self._anim.setDuration(self.OUTRO_MS if closing else self.INTRO_MS)
        self._anim.setStartValue(self._opacity if closing else 0.0)
        self._anim.setEndValue(0.0 if closing else 1.0)
        self._anim.start()

    def _on_frame(self, value):
        t = float(value)
        self._opacity = t
        if self._final is None or self._home is None:
            return
        low = self._zoom_from()
        scale = low + (1.0 - low) * t
        w = max(1, round(self._final.width() * scale))
        h = max(1, round(self._final.height() * scale))
        # `slide-in-from-top-2` on the way IN. The registry gives no
        # `slide-out-to-*`, so the outro shrinks and fades in place.
        # `slide-in-from-*`, in the direction this surface's SIDE implies. The
        # registry gives no `slide-out-to-*`, so the outro shrinks in place.
        off = (QPoint(0, 0) if self._closing
               else self._slide_offset(self.SLIDE_PX * (1.0 - t)))
        # END-ANCHORED surfaces keep their RIGHT edge still and grow leftward;
        # the default keeps the left edge still and grows rightward, which is
        # what `Menu` does for every menu that opens under its trigger.
        x = self._home.x()
        if self._anchor == "end":
            x += self._final.width() - w
        self.setGeometry(x + off.x(), self._home.y() + off.y(), w, h)
        if self._veil is not None:
            self._veil.set_opacity(t)
            self._veil.setGeometry(0, 0, w, h)
            self._veil.raise_()
            self._veil.update()

    def _on_finished(self):
        if self._closing:
            self._dispose()
            return
        self._drop_veil()
        if self._home is not None and self._final is not None:
            self.resize(self._final)
            self.move(self._home)
        # content that changed while the veil hid it is measured now
        self.refit()

    # -- following its content ----------------------------------------------- #

    def follow_content(self, on: bool = True) -> None:
        """Re-measure while OPEN whenever the content's size changes (`refit`).

        Reported 3 Nov 2026: pairs added with `+` in the Significance tool's popover were laid into
        the height measured when it opened, so its rows were drawn over each other and each pair's remove button
        was cut to its lower half. A panel is a CHILD of the window, in no layout, so nothing else resizes it.
        Opt-in: a menu's rows do not change while it is open. A following panel is also held to the window's
        height at opening; its content scrolls there (`primitives.FitScroll`)."""
        self._follows = bool(on)

    def event(self, e):
        handled = super().event(e)
        # The layout has already been activated when the event reaches here (Qt hands a LayoutRequest to the
        # widget's layout first), so the size hint is the new one.
        if e.type() == QEvent.LayoutRequest and getattr(self, "_follows", False):
            self.refit()
        return handled

    def _fitted_size(self) -> QSize:
        """The size hint, no smaller than the panel's minimum and no taller than the window it is drawn in.

        THE MINIMUM COUNTS (4 Nov 2026, rail review): a panel whose content is narrower than its minimum
        (`set_minimum_visible_width`; the rail's `POP_MIN`) was placed by the narrow hint while Qt drew it at the
        minimum. `refit` keeps the right edge of a panel left of its trigger as ``x + width - hint``, so every
        layout request moved it right by the difference: an all but empty Visibility popover walked 150 px a step
        over the rail and out of the window."""
        size = self.sizeHint().expandedTo(self.minimumSize())
        host = self.parentWidget()
        if host is not None and getattr(self, "_follows", False):
            size.setHeight(min(size.height(), host.height()))
        return size

    def refit(self) -> None:
        """Resize an open, following panel to its content, keeping the edge it opened against (the right edge for
        a panel left of its trigger, else the left) and the top, and staying inside the window."""
        if not getattr(self, "_follows", False) or not self.isVisible() or self._closing:
            return
        if self._veil is not None or self._anim.state() == QVariantAnimation.Running:
            return                                  # `_on_finished` measures again
        host = self.parentWidget()
        if host is None:
            return
        size = self._fitted_size()
        if size == self.size():
            return
        x = self.x() + self.width() - size.width() if self._side == "left" else self.x()
        x = max(0, min(x, host.width() - size.width()))
        y = max(0, min(self.y(), host.height() - size.height()))
        self.setGeometry(x, y, size.width(), size.height())
        self._home = QPoint(x, y)
        self._final = QSize(size)

    def _drop_veil(self):
        self._freeze(False)
        if self._veil is not None:
            self._veil.hide()
            discard(self._veil)
            self._veil = None
        self._shot = None

    # -- showing and dismissing ---------------------------------------------- #

    def popup(self, at):
        """Open with the PANEL's top-left at the global point ``at``."""
        host = self.parentWidget()
        self.adjustSize()
        size = self._fitted_size()
        local = host.mapFromGlobal(at) - QPoint(self._room, self._room)
        x = max(0, min(local.x(), host.width() - size.width()))
        y = max(0, min(local.y(), host.height() - size.height()))
        self.resize(size)
        self.move(x, y)
        self._home = QPoint(x, y)
        self.show()
        self.raise_()
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
        self._start(False)

    def visible_width(self) -> int:
        """The CARD's own width -- `sizeHint()` minus the shadow room on both
        sides. `popup(at)` already treats `at` as where the visible card
        (not the wider, room-padded widget) should land, so a caller placing
        this panel by one of ITS OWN edges (the right edge against a rail, the
        end of a trigger it is aligned to) has to subtract this, not the raw
        `sizeHint()` -- subtracting the room-padded width doubles the gap the
        room already buys once inside `popup()`."""
        return self.sizeHint().width() - 2 * self._room

    def set_minimum_visible_width(self, width: int) -> None:
        """`setMinimumWidth`, but of the CARD -- see `visible_width`.

        `setMinimumWidth` alone constrains the room-padded WIDGET, so a floor
        meant to keep the card at least `width` wide silently became a floor
        on `width + 2 * room` instead -- and every caller sizing a popover by
        that floor (this project's own `POP_MIN`) then measured its visible
        width wrong by `2 * room` wherever the floor was the binding one.
        """
        self.setMinimumWidth(width + 2 * self._room)

    def popup_under(self, trigger, align="start", offset=4):
        self.set_trigger(trigger)
        self._anchor = align
        self.adjustSize()
        rect = trigger.rect()
        below = trigger.mapToGlobal(rect.bottomLeft()).y() + offset
        panel_w = self.visible_width()
        if align == "end":
            x = trigger.mapToGlobal(rect.topRight()).x() + 1 - panel_w
        else:
            x = trigger.mapToGlobal(rect.topLeft()).x()
        self.popup(QPoint(x, below))

    def popup_aligned(self, trigger, aligned_row) -> None:
        """Open positioned so ``aligned_row`` -- one of this panel's OWN
        rows, already added via `add_widget` -- lines up with ``trigger``,
        rather than the panel always opening flush below it.

        shadcn's `Select`: when a value is already chosen, the dropdown
        opens with THAT row sitting where the trigger itself is, so
        re-opening a long list to change a selection near its end does not
        mean scrolling past every option above it to find where you
        currently are. `SelectButton` is the caller -- pass it the `_Row`
        for the CURRENTLY SELECTED option.

        **``aligned_row.y()`` is NOT used, and reaching for it first is what
        cost the most time here.** This panel is still `hide()`-den at this
        point (`popup()` below is what first shows it) -- and a QVBoxLayout
        does not finish placing its CHILDREN at a hidden widget's geometry
        the way `adjustSize()`/`sizeHint()` finish computing the widget's
        own overall SIZE. MEASURED directly: `aligned_row.y()` read right
        after `adjustSize()` came back essentially the FIRST row's offset
        regardless of which row was actually asked for, off by hundreds of
        pixels on a long list. Forcing a `show()` first fixed most of it but
        not reliably all of it (the intro's own zoom/slide animation
        transiently resizes this same widget, which can disturb a layout
        pass caught mid-flight) -- a real but needless dependency on
        exactly when Qt gets around to a lazy layout, for a number this
        class can already compute without asking Qt anything: every row
        added here has a FIXED height set at construction
        (`_Row.__init__`'s own `setFixedHeight`), spacing between them is
        `self._lay.spacing()`, and the top inset is the layout's own
        content margin -- summing those for the rows ABOVE ``aligned_row``
        is exact, regardless of whether this widget has ever been shown.

        `popup`'s own edge clamping still applies unmodified: a selection
        near either end of a long list opens flush against that edge
        rather than running off-window, exactly as a plain `popup` already
        does -- this only changes where the IDEAL, unclamped position is.
        """
        self.set_trigger(trigger)
        idx = self._rows.index(aligned_row)
        top_inset = self._lay.contentsMargins().top()
        spacing = self._lay.spacing()
        row_top = (top_inset + idx * spacing
                  + sum(w.height() for w in self._rows[:idx]))
        row_mid = row_top + aligned_row.height() // 2
        trigger_mid = trigger.mapToGlobal(QPoint(0, trigger.height() // 2)).y()
        at = QPoint(trigger.mapToGlobal(QPoint(0, 0)).x(),
                    trigger_mid - row_mid + self._room)
        self.popup(at)

    def set_side(self, side):
        """Where this sits relative to its trigger: bottom / top / left / right."""
        self._side = side

    def _slide_offset(self, amount):
        """`slide-in-from-*`: the surface starts pushed AWAY from its trigger.

        Ported from `Menu._offset` so a submenu slides in from the left rather
        than from the top, which is what `slide-in-from-left-2` means.
        """
        dx = dy = 0
        if self._side == "bottom":
            dy = -amount
        elif self._side == "top":
            dy = amount
        elif self._side == "right":
            dx = -amount
        elif self._side == "left":
            dx = amount
        return QPoint(round(dx), round(dy))

    def _panel_rows(self):
        """The hoverable rows -- those that behave like `_Row`."""
        return [w for w in self._rows if hasattr(w, "set_hover")]

    def _close_sub(self):
        """Close the open submenu ONLY -- never the chain above it.

        `close_chain()` walks UP to the top and closes that, which is right for
        a pick or a click outside and catastrophic here: moving the pointer from
        one submenu row to the next calls this first, so with `close_chain` the
        parent tore itself down and the incoming submenu was left on screen
        alone. Reported as exactly that -- "A day" worked, "A group" made the
        parent vanish -- because the second hover is the first one that has an
        outgoing submenu to close. `Menu._close_sub` calls `close()`, and this
        is its equivalent.
        """
        row, self._sub_open = self._sub_open, None
        if row is not None and row.submenu is not None:
            row.submenu._close()

    def _sync(self, gpos):
        """Point the highlight and the open submenu at wherever the mouse is.

        The shape is `Menu._sync`'s, including the ordering it records: CLOSE
        the outgoing submenu BEFORE computing highlights, or the row that just
        lost the pointer still passes the "keep me lit" test and two rows stay
        highlighted at once.
        """
        open_row = self._sub_open
        if open_row is not None and open_row.submenu is not None:
            sub = open_row.submenu
            tl = sub.mapToGlobal(QPoint(0, 0))
            if sub.isVisible() and QRect(tl, sub.size()).contains(gpos):
                return

        hit = None
        for r in self._panel_rows():
            if QRect(r.mapToGlobal(QPoint(0, 0)), r.size()).contains(gpos):
                hit = r
                break

        if hit is not None and self._sub_open is not None and self._sub_open is not hit:
            self._close_sub()

        for r in self._panel_rows():
            lit = r is hit or (r is self._sub_open and r.submenu is not None
                               and r.submenu.isVisible())
            r.set_hover(lit)

        if hit is None:
            return
        if getattr(hit, "has_submenu", False) and self._sub_open is not hit:
            hit.set_hover(True)
            try:
                sub = hit.submenu
                if sub is None:
                    return
                sub.isVisible()          # touches C++; raises if it is gone
            except RuntimeError:
                return
            # RIGHT IF IT FITS, ELSE FLIP LEFT -- never clamp.
            #
            # `popup` keeps a surface inside the window, which for a submenu
            # means being pushed back OVER its parent. That is not just ugly:
            # the overlapping submenu then covers the parent's own rows, so
            # `_sync` reads the pointer as being inside the submenu, returns
            # early, and the NEXT submenu never opens. Flipping is what Radix
            # does when a side does not fit, and it keeps the two surfaces
            # side by side.
            sub.adjustSize()
            panel_w = sub.visible_width()
            host = self.parentWidget()
            right_at = hit.mapToGlobal(QPoint(hit.width() - 4, -4))
            if host.mapFromGlobal(right_at).x() + panel_w > host.width():
                sub.set_side("left")
                sub.popup(hit.mapToGlobal(QPoint(4 - panel_w, -4)))
            else:
                sub.set_side("right")
                sub.popup(right_at)
            self._sub_open = hit

    def close_chain(self):
        top = self
        while top._parent_panel is not None:
            top = top._parent_panel
        top._close()

    def _close(self):
        if self._closing:
            return
        # `menu_row`'s `on_click` runs BEFORE this (`widget.triggered.connect(lambda:
        # (on_click(), _close(menu)))`), and an `on_click` that rebuilds its host --
        # every group-board row does, on a plain state change -- can tear this very
        # Panel down synchronously, `deleteLater` and all, before `_close` ever runs.
        # Touch the C++ side FIRST and bail if it is already gone, same idiom `_sync`
        # uses two screens up for a submenu that may have vanished underneath it.
        try:
            visible = self.isVisible()      # touches C++; raises if it is gone
        except RuntimeError:
            return
        self._close_sub()
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        if not visible:
            self._dispose()
            return
        self._start(True)

    def _dispose(self):
        """End of the outro: hide, and free only if this is a top-level panel."""
        self._drop_veil()
        self.hide()
        self._closing = False          # reopenable
        self.closed.emit()
        if not self._free_on_hide:
            return
        for kid in list(self._children):
            kid._teardown()
        self._children = []
        discard(self)

    def _teardown(self):
        """Free for good. Only a top-level panel's disposal reaches here, and
        it takes its submenus with it -- they are children of the HOST window,
        not of this widget, so Qt would not collect them on its own."""
        for kid in list(self._children):
            kid._teardown()
        self._children = []
        self._drop_veil()
        self.hide()
        discard(self)

    def hideEvent(self, e):
        """Stop listening the moment this leaves the screen.

        The belt to `_close`'s braces: whatever hides a panel -- a row's close,
        the window closing, a caller reaching for `QWidget.close()` -- it must
        not be left filtering the application's mouse moves and opening
        submenus for a surface nobody can see.
        """
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        self._sub_open = None
        super().hideEvent(e)

    def eventFilter(self, obj, event):
        # A hidden panel filters nothing: `hideEvent` removes this, but the
        # event already in flight can still arrive.
        if not self.isVisible():
            return False
        kind = event.type()
        if kind == QEvent.MouseMove:
            gp = (event.globalPosition().toPoint()
                  if hasattr(event, "globalPosition") else event.globalPos())
            self._sync(gp)
        elif kind == QEvent.MouseButtonPress:
            gp = (event.globalPosition().toPoint()
                  if hasattr(event, "globalPosition") else event.globalPos())
            if not self._inside_chain(gp):
                rect = self._trigger_rect()
                if rect is not None:
                    _DISMISSED["rect"] = rect
                    _DISMISSED["at"] = time.monotonic()
                self.close_chain()
        elif kind == QEvent.KeyPress and event.key() == Qt.Key_Escape:
            self.close_chain()
        return False

    def _trigger_rect(self):
        """The trigger's global rect, whether it is a widget or a bare rect.

        A swatch dot has no lasting widget -- the pill is rebuilt on every tab
        change -- so `_pick_colour` passes the RECT it was clicked in.
        `retriggered` already accepts either; this keeps the pair symmetrical,
        or a click on the dot that closed the popover would reopen it.
        """
        trig = self._trigger
        if trig is None:
            return None
        if isinstance(trig, QRect):
            return trig
        try:
            return QRect(trig.mapToGlobal(QPoint(0, 0)), trig.size())
        except (AttributeError, RuntimeError):
            return None

    def _inside_chain(self, gp):
        node = self
        while node is not None:
            if node.isVisible():
                tl = node.mapToGlobal(QPoint(0, 0))
                if QRect(tl, node.size()).contains(gp):
                    return True
            node = node._parent_panel
        return False


def styled_menu(t, parent=None, *, root: int = MENU_ROOT):
    """A dropdown surface, of whichever kind its parent implies.

    A submenu is built by a factory that says `styled_menu(t, menu)` -- so when
    ``menu`` is a :class:`Panel` this has to hand back a `Panel` too, or a
    popover drawn in the page would sprout a native popup child with its own OS
    shadow. Callers do not change; the parent decides.
    """
    if isinstance(parent, Panel):
        sub = Panel(t, parent.window(), root=root, parent_panel=parent)
        parent._children.append(sub)
        return sub
    return Menu(t, parent, root=root)


def close_panels_under(widget) -> None:
    """Close every open top-level `Panel` opened from a trigger inside ``widget``.

    A drop-down opened from inside a popover (the Significance block's test list, a pair picker) is a panel of its
    own, not a sub-panel of the popover, so the popover's own close left it standing: when the rail's Visibility
    popover closed under it (the page switched), the list stayed over the next page (4 Nov 2026, rail review)."""
    try:
        host = widget.window()
    except RuntimeError:
        return
    for panel in host.findChildren(Panel):
        try:
            trigger = panel._trigger
            if (panel._parent_panel is None and panel.isVisible() and isinstance(trigger, QWidget)
                    and widget.isAncestorOf(trigger)):
                panel.close_chain()
        except RuntimeError:                 # a panel or its trigger already deleted
            continue


def end_outros() -> None:
    """Finish every outro still playing, immediately.

    For the case where a new surface REPLACES an old one at the same anchor --
    the tool rail, whose four popovers all open against the same edge. The
    outro is a snapshot of the closing popover drawn in the page; if the next
    one opens over it, whatever the new surface does not cover stays on screen
    as a pale outline of the old, which reads as a rendering fault rather than
    as a transition. MEASURED: close the Visibility popover (271 px wide) and
    open Visualization (226) within the outro's 100 ms, and the wider one's
    edge shows down the left of the narrower.

    Not called from `Menu` itself: a menu closing while an unrelated one opens
    elsewhere on screen SHOULD play its exit. This is for surfaces that share a
    place, and the caller is the one that knows.
    """
    for ghost in list(_GHOSTS):
        ghost.dispose()


def warm_backdrop(window) -> None:
    """Pay a widget tree's first offscreen render, off the click path.

    **The whole cost is initialising the path, not drawing the pixels.** A menu
    grabs the page behind it for its outro (`Menu._page_under`), and MEASURED on
    a freshly shown screen that grab costs **126.9 ms the first time**, 11.8 ms
    the second and ~1.2 ms after -- for a 345x320 region. A **1x1** grab of the
    same window costs 51 ms and leaves the real one at 3.5 ms, so the size is
    irrelevant: it is Qt setting up the offscreen render for that tree.

    It has to be paid somewhere, and there are only three places:

      * before the menu appears -- the original, and it put 126 ms between the
        click and the first pixel;
      * during the intro -- where moving it merely relocated the stall, and the
        fade visibly lost frames (a check counts them, and it went red);
      * HERE, just after a screen is shown, when nothing is waiting on it.

    Per WINDOW TREE, so it must run for each screen rather than once at startup:
    warming the home screen measurably did NOT help the confirm screen's first
    menu.
    """
    if window is None:
        return
    try:
        window.grab(QRect(0, 0, 1, 1))
    except RuntimeError:            # the window went away first
        pass


#: The trigger whose own click just dismissed its popup, and when.
#:
#: A popup grabs the mouse, so a click on the trigger of an OPEN menu is
#: dismissal first -- and then the trigger receives the click too and opens it
#: straight back up. From the outside the menu never closes. This is the one
#: piece of state that lets an opener tell "you clicked me to open" from "you
#: clicked me to close": see :func:`retriggered`.
#:
#: A global rect rather than the widget, so a trigger that is not a widget works
#: too -- the colour swatch is a dot inside a pill that gets rebuilt, and is
#: identified by the rect it was clicked in.
_DISMISSED: dict = {"rect": None, "at": 0.0}

#: How long a dismissal suppresses a reopen. Long enough to cover the click that
#: caused it, short enough that a deliberate second click always opens.
RETRIGGER_MS = 400


def _global_rect(target) -> QRect:
    if isinstance(target, QRect):
        return target
    return QRect(target.mapToGlobal(QPoint(0, 0)), target.size())


def retriggered(target) -> bool:
    """Did this trigger's own click just close its popup? CONSUMES the answer.

    Openers call it first and return early when it is True, which turns
    click-to-open into click-to-toggle.
    """
    rect, at = _DISMISSED["rect"], _DISMISSED["at"]
    _DISMISSED["rect"] = None
    if rect is None or (time.monotonic() - at) * 1000 > RETRIGGER_MS:
        return False
    return rect == _global_rect(target)


#: Every outro currently playing.
#:
#: A parentless QWidget with no Python reference is garbage collected, taking
#: the C++ widget with it -- MEASURED as an outro that never appeared at all:
#: zero ghosts on screen at every sample, and zero alive afterwards, from code
#: that had just constructed one. Top-level Qt widgets are usually kept alive by
#: their parent; these have none by design, so the module holds them until they
#: are done.
_GHOSTS: set = set()


class _Ghost(QWidget):
    """`data-closed:animate-out fade-out-0 zoom-out-95`, drawn IN THE PAGE.

    **A child of the main window, not a window of its own** -- which is the
    whole point, and took three tries to arrive at.

    The real menu has to be gone the instant it closes, or every dismissal
    decision in :class:`Menu` (the chain close, the press-outside test, the
    submenu handover) has to stay correct while it is neither open nor closed.
    So something else has to play the outro. The obvious something is a second
    top-level window standing where the menu was -- and that is a HANDOFF: a new
    window has to appear exactly where the old one vanished, in the same frame,
    at the same z-order, showing the same pixels. Each attempt fixed one part of
    it and flickered on another:

    * shown late (deferred with the fade) -> ~28 ms with neither on screen;
    * `resize()` + `setWindowOpacity()` per frame -> six DWM operations on a
      layered window in 100 ms;
    * fixed window, painted internally -> still a second window appearing and
      disappearing over the page.

    A child widget has no handoff at all. It is in the main window's backing
    store, so it appears in the same composite as everything else, needs no
    z-order agreement, and -- because Qt paints a parent before its children --
    it can simply blend over whatever the page already drew. No backdrop
    snapshot, no translucency (a popup WINDOW here composites against black; a
    child widget does not), no window geometry, no window opacity.

    The one thing it cannot do is extend past the main window. Menus here open
    inside it; `Menu._spawn_ghost` skips the outro rather than clip one that
    does not.
    """

    @staticmethod
    def snap_step(widget) -> int:
        """The smallest logical step that lands on a WHOLE device pixel.

        1 at ratio 1.0 or 2.0; **4 at ratio 1.25**, which is this screen.
        """
        dpr = widget.devicePixelRatioF()
        step = 1
        while (step * dpr) % 1 != 0 and step < 32:
            step += 1
        return step

    def __init__(self, host, menu_rect):
        super().__init__(host)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        # No erase, and no repaint of the page beneath either. See
        # `paintEvent`: this holds its own copy of the page, so Qt has nothing
        # to compose it against and the parent stays out of the animation.
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self._shot = None
        self._rect = None
        self._t = 1.0
        self._playing = False
        #: The scale this outro shrinks to -- the menu's, not the class default.
        self._zoom = Menu.ZOOM_FROM
        #: The easing the ZOOM uses. The opacity is linear; see `run`.
        self._curve = None
        #: The page under the menu, so this can paint every pixel it covers.
        self._backdrop = None
        #: Where the MENU sits inside this widget, and how big it is -- the two
        #: differ because the widget is snapped outward to whole device pixels.
        self._origin = QPoint(0, 0)
        self._menu = menu_rect.size()
        self._anim = QVariantAnimation(self)
        self._anim.setStartValue(1.0)
        self._anim.setEndValue(0.0)
        self._anim.valueChanged.connect(self._frame)
        self._anim.finished.connect(self._done)
        #: Set by `dispose`. A deferred `run` scheduled before it must not
        #: touch this object afterwards -- see `Menu._spawn_ghost`.
        self._disposed = False
        _GHOSTS.add(self)
        self.destroyed.connect(lambda *_: _GHOSTS.discard(self))
        # SNAPPED TO WHOLE DEVICE PIXELS, and this is the alignment fix.
        #
        # `host.grab(rect)` re-renders the tree with the origin translated to
        # `rect.topLeft()` and rasterises at the device ratio. If that corner is
        # not a whole device pixel, the copy comes out at a different SUB-PIXEL
        # PHASE from the screen -- text and hairlines land a fraction of a pixel
        # off, so the page behind a closing menu is subtly displaced from the
        # page beside it. MEASURED at ratio 1.25, where a menu at logical x=910
        # is device x=1137.5: **8.9 %** of sampled pixels differ from what the
        # window renders, and at EVERY whole-pixel offset -- so it is not a
        # shift that could be nudged away. Snapped to x=908 (device 1135.0) the
        # mismatch is **0.0 %**.
        #
        # Reported as a progress bar "heavily misaligned" across the edge of a
        # closing menu, which is what a hairline at the wrong phase looks like
        # beside the same hairline at the right one.
        step = self.snap_step(host)
        sx = (menu_rect.x() // step) * step
        sy = (menu_rect.y() // step) * step
        sw = -(-(menu_rect.right() + 1 - sx) // step) * step
        sh = -(-(menu_rect.bottom() + 1 - sy) // step) * step
        self._origin = QPoint(menu_rect.x() - sx, menu_rect.y() - sy)
        self.setGeometry(sx, sy, sw, sh)
        # SHOWN AT OPEN TIME, empty -- and this is the flicker fix.
        #
        # `repaint()` is a no-op on an unmapped widget, and a widget shown
        # moments ago is not mapped yet: measured right after `show()`,
        # `isVisible()` True but `WA_Mapped` False. So an overlay created at
        # CLOSE time cannot paint itself in the closing turn, the page repaints
        # the area the popup vacated without it, and one frame shows the bare
        # page -- menu, page, menu picture, fade. The paint log named it:
        # `OnboardingWindow` over the menu's rect at 5.8 ms, first `_Ghost`
        # paint at 33.1 ms.
        #
        # Showing it here lets Qt map it while the menu is still up and covering
        # it. With no picture yet it paints NOTHING, and `WA_NoSystemBackground`
        # means nothing is erased either, so it is invisible while it waits.
        self.show()

    def needs_backdrop(self) -> bool:
        """Has it never had one? The first grab is the expensive one."""
        return self._backdrop is None

    def set_backdrop(self, backdrop) -> None:
        """The page this covers, so it can paint every pixel and stay opaque."""
        if backdrop is not None and not backdrop.isNull():
            self._backdrop = backdrop
            self.setAttribute(Qt.WA_OpaquePaintEvent, True)

    def hold(self, shot, zoom_from: float) -> None:
        """Take the menu's current picture and paint it, NOW.

        Called while the menu is still up and covering this exactly, so the
        paint lands and nobody sees it. By the time the menu hides, the pixels
        are already in the page.
        """
        self._shot = shot
        self._rect = QRect(self._origin, self._menu)
        self._t = 1.0
        self._zoom = zoom_from
        self.raise_()
        self.repaint()

    def armed(self) -> bool:
        """Is there a picture ready to fade?"""
        return self._shot is not None and not self._shot.isNull()

    def playing(self) -> bool:
        """Is the outro RUNNING? An overlay merely holding a picture behind an
        open menu is not -- it is invisible and idle, and counting it as an
        outro is how a leak looks like an animation."""
        return self._playing

    def run(self, ms, curve) -> None:
        """Start fading, once the loop is free to draw it.

        **The OPACITY runs linear; the ZOOM keeps the curve.** They were one
        eased value, reusing the intro's `ease` for the exit, and that put the
        whole perceptible fade in the first third. MEASURED as visible change
        per 10 % of the outro's time -- the mean absolute pixel difference
        between consecutive frames, which is what the eye is actually judging::

            ease (the intro's)  5.7 6.7 6.6 4.2 3.0 1.6 1.3 0.7 0.4 0.0   3 still
            linear              5.8 4.1 3.4 3.7 3.0 2.6 2.6 2.5 2.3 1.8   0 still
            ease-in             0.4 4.9 1.4 3.8 3.4 5.0 3.2 3.2 3.3 3.0   1 still

        With `ease` the last three steps change almost nothing: the fade is over
        by roughly 40 % of the duration and the rest is an imperceptible tail,
        which reads as "it fades quickly from one frame to another rather than
        gradually" and as the animation being faster than its own duration.
        Linear is the only one of the three with no near-still step at either
        end.

        This is also the conventional split rather than an invention: easing
        curves exist to make MOTION feel physical, and a plain opacity crossfade
        has no momentum to model -- Material says fades should be linear. So the
        zoom, which is motion, keeps the registry's curve.
        """
        if self._disposed:
            return          # disposed between being spawned and getting here
        if self._rect is None:
            self.dispose()
            return
        self._playing = True
        self._curve = curve
        self._anim.setDuration(ms)
        self._anim.setEasingCurve(QEasingCurve.Linear)
        self._anim.start()

    def _frame(self, value) -> None:
        self._t = float(value)
        self.update()

    def _done(self) -> None:
        self.dispose()

    def dispose(self) -> None:
        """Out of the page and out of the registry, for good.

        **And REPAINT what it was covering, in this turn.** Hiding a child
        marks the vacated rect dirty on its parent and Qt flushes that with the
        next paint -- which is fine when nothing else is happening, and a race
        when a popup opens in the same gesture. That is exactly the case this
        method exists for: clicking one rail button while another's popover is
        open closes the first and opens the second inside ONE click, and
        MEASURED, the ghost is disposed at ~8 ms while the new popover appears
        at ~20 ms. The queued flush does not land first, so the old surface's
        snapshot is left on screen around whatever the new one does not cover.
        Clicking AWAY first and then opening the next is clean, because the
        repaint has a turn to itself -- which is precisely the difference the
        report described.

        `repaint`, not `update`: `update` only queues, and queuing is the thing
        that loses. One rect, once per disposal. The same fix, for the same
        reason, as `tooltip.Tooltip._vanish`.
        """
        if self._disposed:
            return
        self._disposed = True
        self._playing = False
        area = self.geometry()
        host = self.parentWidget()
        self.hide()
        self.setParent(None)
        if host is not None:
            host.repaint(area)
        self.deleteLater()

    def paintEvent(self, _e):
        """One frame: the page, then the shrinking menu blended over it.

        **Painting the page HERE is what keeps the frame rate even**, and it is
        not an optimisation for its own sake. A child widget that leaves gaps
        makes Qt repaint the parent underneath first, so every outro frame cost
        a repaint of the page across the menu's whole rect -- and the menus are
        not the same size. MEASURED at `OUTRO_MS = 100`, where there are only
        six frames to lose::

            Filter   256x257   6 paints, 20.5 ms apart    <- misses vsync
            palette  256x151   6 paints, 14.8 ms apart
            swatch    282x82   6 paints, 14.4 ms apart

        Which is exactly why `slowclose.py` shows no fault: at 2000 ms all three
        pace at 16.0 ms and the per-frame cost vanishes into the interval. The
        report that named it was "the periods look the same in slowclose, but
        the Filter still seems faster in the app" -- uneven pacing, not
        duration, and only measurable at the real duration.

        With the page held here the widget is opaque, Qt composes nothing
        beneath it, and the cost stops scaling with the menu's area.

        THE TRADE: the page is a still, taken once per menu-open, so if an
        action changed the page the uncovered margin shows the older one for
        100 ms. Refreshing it at close was tried and cost the pacing again --
        ~22 ms a frame for every menu, the fault back for all three rather than
        one. The margin is 8 % of the overlay at the START of the outro (a
        ~10 px L-strip), shrinking to nothing, under a picture that is itself
        fading out. An invisible staleness beats a visible judder.
        """
        if self._shot is None or self._rect is None:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)
        if self._backdrop is not None:
            # BY POINT, NOT BY RECT -- and the difference is visible.
            #
            # `drawPixmap(rect, pixmap)` runs the scaler even when the rect is
            # exactly the pixmap's size, so the page came out RESAMPLED at 1:1:
            # measured, 14 % of sampled pixels differ from the source, which is
            # a soft, faintly doubled page for as long as the outro lasts and a
            # snap back to normal when it ends. That snap is what reads as
            # "blurry and slightly distorted behind the menu". The point
            # overload honours the pixmap's device ratio and blits -- measured
            # byte-identical to the source.
            p.drawPixmap(0, 0, self._backdrop)
        p.setOpacity(self._t)                          # `fade-out-0`, LINEAR
        low = self._zoom
        # The zoom is MOTION, so it keeps the registry's easing -- run over the
        # same linear clock. See `run` for why the two are split.
        eased = self._t
        if self._curve is not None:
            eased = 1.0 - self._curve.valueForProgress(1.0 - self._t)
        scale = low + (1.0 - low) * eased
        # From the MENU's corner inside this widget, not the widget's own --
        # they differ by the snap.
        w = max(1, round(self._menu.width() * scale))
        h = max(1, round(self._menu.height() * scale))
        if w == self._menu.width() and h == self._menu.height():
            p.drawPixmap(self._origin, self._shot)   # first frame, unscaled
        else:
            p.drawPixmap(QRect(self._origin, QSize(w, h)), self._shot)
        p.end()


class _Veil(QWidget):
    """A snapshot of the finished menu, painted to fill whatever size the
    window currently is. See :meth:`Menu._on_intro` for why it exists."""

    def __init__(self, parent, shot):
        super().__init__(parent)
        self._shot = shot
        #: 1.0 for `Menu`, which fades the whole WINDOW instead. `Panel` is a
        #: child widget and has no window opacity to set, so its fade happens
        #: here. Default leaves `Menu` untouched.
        self._opacity = 1.0
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)

    def set_opacity(self, value) -> None:
        self._opacity = max(0.0, min(1.0, float(value)))

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)
        if self._opacity < 1.0:
            p.setOpacity(self._opacity)
        p.drawPixmap(self.rect(), self._shot)
        p.end()


class _MenuTile(QWidget):
    """``flex size-6 items-center justify-center rounded-md border``.

    The mark the ``sidebar-07`` team switcher puts on every row of its menu.
    Painted rather than styled for the reason the hover pill is: a stylesheet
    ``border-radius`` is applied in DEVICE pixels here, so on a 1.25 screen it
    renders 20 % small.

    **Every row gets the same one, the open workspace included.** A `filled`
    variant was tried for it -- the trigger's own ``bg-sidebar-primary`` tile,
    so the row you are on wore the mark the header wears. It was wrong on
    sight: the block draws exactly one kind of tile in its menu, and a solid
    black square among outlines reads as a different KIND of row rather than as
    "you are here". The menu lists destinations; which one you are in is the
    header's job, and the header is directly above it.
    """

    def __init__(self, t, glyph: str, *, size: int, glyph_px: int,
                 parent=None):
        super().__init__(parent)
        self.t, self._glyph = t, glyph
        self._glyph_px = glyph_px
        self.setFixedSize(size, size)
        # The row owns the click; a child that ate it would make the tile a
        # dead spot in the middle of its own menu item.
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        r = self.t.radius_md
        # Inset by half the pen, or the stroke straddles the widget edge and
        # the outer half is clipped away.
        p.setPen(QPen(QColor(self.t.border), 1.0))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5),
                          r - 0.5, r - 0.5)
        ink = self.t.foreground
        g = float(self._glyph_px)
        off = (self.width() - g) / 2.0
        icons.paint(p, self._glyph, QRectF(off, off, g, g), ink)
        p.end()


class _Row(QWidget):
    """One menu item: ``px-2 py-1.5 text-sm rounded-sm``, hover ``bg-accent``."""

    triggered = Signal()

    def __init__(self, t, text: str, *, badge=None, count=None,
                 submenu: QMenu | None = None, current: bool = False,
                 glyph: str | None = None, muted: bool = False,
                 root: int = MENU_ROOT, submenu_factory=None, parent=None):
        super().__init__(parent)
        self.t = t
        #: Built on first need -- see the :attr:`submenu` property.
        self._submenu = submenu
        self._make_submenu = submenu_factory
        self._hover = False
        self.setObjectName("mrow")
        # No WA_StyledBackground and no stylesheet on the row itself: the
        # hover fill is painted (see paintEvent). It went through a stylesheet
        # first, which needed that attribute to draw at all on a QWidget
        # subclass -- and then rendered its radius in device pixels anyway.
        self._paint(False)
        self.setCursor(Qt.PointingHandCursor)
        pad = _rem(0.5, root)                   # `px-2`
        if glyph is None:
            # ``py-1.5`` (6 px each side) + ``text-sm``'s 20 px line box = 32.
            self.setFixedHeight(2 * _rem(0.375, root) + _rem(1.25, root))
        else:
            # The block overrides the item to ``gap-2 p-2`` around a `size-6`
            # tile, and THAT is where the reference's airiness comes from: 40 px
            # per row against the base item's 32. It is not spacing BETWEEN the
            # rows -- `DropdownMenuContent` has no `gap` and the rows sit flush
            # -- it is the rows themselves being taller. Chasing it with a gap
            # would have looked close and been wrong the moment a row hovers,
            # since the hover pill is the row.
            self.setFixedHeight(2 * pad + _rem(1.5, root))
        self.setAttribute(Qt.WA_Hover, True)

        text_px = _type_px(0.875, root)         # `text-sm`
        small_px = _type_px(0.75, root)         # `text-xs`
        cells = []
        if glyph is not None:
            cells.append(_MenuTile(t, glyph, size=_rem(1.5, root),
                                   glyph_px=_rem(0.875, root)))
        # `text-muted-foreground font-medium` -- the block's "Add team" row,
        # which is an ACTION rather than a destination and says so in ink.
        cells.append(label(text, ty.ui("sm", "medium" if muted else "normal",
                                       px=text_px),
                           t.muted_foreground if muted else t.foreground))
        if badge is not None:
            fill, ink, tag = badge
            cells.append(Badge(t, tag, fill, ink))
        cells.append(None)
        if count is not None:
            cells.append(label(str(count), ty.ui("xs", px=small_px),
                               t.muted_foreground))
        # **`SelectItem`'s indicator sits ON THE RIGHT** (`absolute right-2`
        # in the actual registry component), not the left -- this used to sit
        # in the FIRST cell, ahead of the label, which is Radix's OWN
        # `DropdownMenuCheckboxItem` convention (a different component) and
        # not what a value picker like `toolbar.SelectButton` is. Moved
        # after the stretch so it lands at the far right, matching count and
        # the submenu chevron.
        if current:
            # **STROKE, NOT COLOUR, is what read as grey.** `t.foreground` is
            # already `#09090B` here -- effectively black, the same token the
            # row's own TEXT uses. The default icon stroke (`icons.STROKE`,
            # 1.25) is tuned for a leaner look at the sizes most glyphs in
            # this app are drawn at; a `size-4` CHECK at that thin a stroke
            # anti-aliases into something visibly lighter than solid text
            # beside it, at the IDENTICAL colour value. `stroke=2.0` matches
            # lucide's own default weight (what the reference actually
            # renders) and what this codebase already reaches for whenever a
            # glyph needs to read as solid rather than lean -- `ToolButton`'s
            # own glyph and `SelectButton`'s chevron both already do this.
            cells.append(icon("check", icons.SM, t.foreground, stroke=2.0))
        if self.has_submenu:
            cells.append(icon("chevron-down", icons.XS, t.muted_foreground))
        # ONE layout, not a layout holding a `row()` holding another.
        #
        # The wrapper cost a whole extra QWidget and QHBoxLayout per row for
        # nothing: this row paints its own hover pill in `paintEvent` and sets
        # no background, so the transparent holder was never doing any work.
        # MEASURED building the same three cells 300 times: 706 us nested
        # against 354 us flat -- half, and about 2.8 ms off an eight-row menu's
        # build. `row()` stays as it is for everyone else; only this one caller
        # stopped needing it.
        lay = QHBoxLayout(self)
        lay.setContentsMargins(pad, 0, pad, 0)
        lay.setSpacing(_rem(0.5, root))
        for cell in cells:
            if cell is None:
                lay.addStretch(1)       # `ml-auto` on the shortcut
            else:
                lay.addWidget(cell)
        self.setMinimumWidth(_rem(12.5, root))

    @property
    def has_submenu(self) -> bool:
        """Does this row open a submenu? Answered WITHOUT building it."""
        return self._submenu is not None or self._make_submenu is not None

    @property
    def submenu(self):
        """The submenu, built on first access if it arrived as a factory.

        **Lazy because a submenu is a menu you may never look at.** The lanes
        board's Filter menu builds three menus and seventeen rows, and twelve of
        those rows are inside "A day" and "A group" -- shown only if you hover
        the row that owns them, and paid for on every open regardless.

        A property rather than an explicit `ensure()` call, so every existing
        reader keeps working: `Menu` asks for `.submenu` at exactly the moment
        it is about to pop one, and the places that merely TEST for one use
        `has_submenu`, which does not build.
        """
        if self._submenu is None and self._make_submenu is not None:
            self._submenu, self._make_submenu = self._make_submenu(), None
        return self._submenu

    def _paint(self, hover: bool) -> None:
        """Only a repaint request now -- see :meth:`paintEvent`."""
        self.update()

    def paintEvent(self, e):
        """The hover pill is PAINTED, not styled.

        A stylesheet ``border-radius`` is applied here in DEVICE pixels, so on
        a 1.25 screen every radius renders 20 %% small. Calibrated on this row,
        stylesheet value against the arc actually drawn::

            requested  4  ->  1.6 logical px
            requested  6  ->  3.2
            requested  9  ->  5.6
            requested 12  ->  8.0

        A slope of 0.8 = 1/1.25 is the ratio itself, so ``rounded-sm``'s 6 px
        was landing at about 4.8 -- visibly squarer than shadcn's, which is
        exactly what it looked like. Compensating by multiplying by the ratio
        would bake this screen's scaling into a token, so the fill is drawn
        directly instead: ``QPainter`` takes logical coordinates and antialiases
        the arc, which is how every other rounded rect here is already drawn.
        """
        if not self._hover:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(self.t.accent))
        r = self.t.radius_sm
        p.drawRoundedRect(QRectF(self.rect()), r, r)
        p.end()

    def set_hover(self, on: bool) -> None:
        """Idempotent: ``setStyleSheet`` is far too costly to run per mouse
        move, and :meth:`Menu._sync` runs on every one of them."""
        if on != self._hover:
            self._hover = on
            self._paint(on)

    def mouseReleaseEvent(self, e):
        # `has_submenu`, NOT `.submenu`: testing the property here would build
        # a submenu just to discover the row has one, on every click of it.
        if e.button() == Qt.LeftButton and not self.has_submenu:
            self.triggered.emit()


def _attach(menu, widget):
    """Put ``widget`` into ``menu``, whichever kind of surface it is.

    :class:`Panel` lays widgets out directly; a `QMenu` needs each one wrapped
    in a `QWidgetAction`. One function, so the `menu_*` builders serve both
    while the popovers are being moved off the native window.
    """
    add = getattr(menu, "add_widget", None)
    if add is not None:
        add(widget)
        return
    action = QWidgetAction(menu)
    action.setDefaultWidget(widget)
    menu.addAction(action)


#: ``menu`` in the builders below is a `Menu` OR a :class:`Panel` -- the
#: annotation said `QMenu` and that is how a Panel came to be closed with
#: `QWidget.close()`, leaving it hidden and still filtering events. They are
#: untyped rather than mistyped; `_attach` and `_close` ask for the capability.
def menu_row(t, menu, text: str, *, badge=None, count=None,
             submenu: QMenu | None = None, current: bool = False,
             glyph: str | None = None, muted: bool = False,
             root: int = MENU_ROOT, submenu_factory=None,
             on_click=None) -> _Row:
    """One menu item.

    ``submenu_factory`` is ``submenu``'s lazy form: a callable returning the
    menu, run on the first hover of this row rather than while the parent menu
    is being built. Prefer it whenever the submenu has more than a row or two
    in it -- see :attr:`_Row.submenu`.
    """
    widget = _Row(t, text, badge=badge, count=count, submenu=submenu,
                  current=current, glyph=glyph, muted=muted, root=root,
                  submenu_factory=submenu_factory)
    if on_click is not None:
        widget.triggered.connect(lambda: (on_click(), _close(menu)))
    _attach(menu, widget)
    return widget


def _close(menu) -> None:
    """Dismiss the whole chain a row belongs to.

    ASKS FOR THE CAPABILITY, does not test the class. This read
    ``isinstance(menu, Menu)`` and fell through to `QWidget.close()` for
    anything else -- so a :class:`Panel` was merely HIDDEN: its app-wide event
    filter stayed installed, `_sync` kept running on every mouse move, and the
    next hover opened a submenu belonging to a menu that was no longer on
    screen. Reported as exactly that: pick "No group yet", move the mouse, and
    a stray submenu appears.
    """
    closer = getattr(menu, "close_chain", None)
    if closer is not None:
        closer()
    else:
        menu.close()


def menu_section(t, menu, text: str, *, root: int = MENU_ROOT) -> None:
    """``DropdownMenuLabel``: ``px-2 py-1.5 text-sm font-medium``, muted here so
    it reads as a heading rather than as another item."""
    holder = row(label(text, ty.ui("xs", "medium", px=_type_px(0.75, root)),
                       t.muted_foreground),
                 None, spacing=0,
                 margins=(_rem(0.5, root), _rem(0.375, root),
                          _rem(0.5, root), _rem(0.25, root)))
    _attach(menu, holder)


def menu_note(t, menu, text: str) -> None:
    """A closing sentence that says what the menu does NOT do.

    "Hides rows here. Nothing leaves the fit." is the whole reason the filter is
    safe to use, and it belongs where the choice is made rather than in a manual.
    """
    menu.addSeparator()
    note = label(text, ty.ui("xs"), t.muted_foreground)
    note.setWordWrap(True)
    note.setFixedWidth(230)
    holder = row(note, spacing=0, margins=(8, 4, 8, 6))
    _attach(menu, holder)


class FilterChip(QWidget):
    """The active filter, and the way out of it.

    Present only while a filter is on. A permanently visible dropdown reading
    "Filter: everything" spends toolbar width to say that nothing is happening;
    a button plus a chip that appears only when it must says the same in
    nothing at all.
    """

    dismissed = Signal()

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.setObjectName("chip")
        self.setStyleSheet(
            f"QWidget#chip {{ background:{t.muted}; border:none;"
            f" border-radius:{t.radius_md:g}px; }}")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setFixedHeight(24)
        self._text = label("", ty.ui("xs", "medium"), t.foreground)
        close = icon("x", icons.XS, t.muted_foreground)
        close.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        close.setCursor(Qt.PointingHandCursor)
        close.mouseReleaseEvent = lambda _e: self.dismissed.emit()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(row(self._text, close, spacing=6, margins=(8, 0, 7, 0)))

    def set_text(self, text: str) -> None:
        self._text.setText(text)
