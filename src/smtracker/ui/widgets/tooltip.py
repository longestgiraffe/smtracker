"""shadcn's Tooltip, painted as an in-window overlay.

Read off the registry (``shadcn view @shadcn/tooltip``) rather than remembered::

    TooltipContent   rounded-md bg-foreground px-3 py-1.5 text-xs text-background
    TooltipArrow     size-2.5 rotate-45 rounded-[2px] bg-foreground
    sideOffset       0
    delayDuration    0

Two of those are the whole character of the component and are easy to lose.

**The TYPE is the one deliberate departure: 13/500, not `text-xs`/400.**

Worth being precise about what that fixes, because the obvious explanation is
wrong. It is tempting to say inverted text rasterises thinner -- MEASURED, it
does not. "Record" drawn white-on-foreground against the same string
dark-on-background::

    12 px / 400, on the dark pill    339 inked px   61 % at full opacity
    12 px / 400, on the light page   298 inked px   37 % at full opacity

The pill is the BETTER of the two. So the fix is not compensation for the
renderer.

What is actually wrong is the ROLE. At ``text-xs``/400 the tooltip is set
identically to every muted caption on the screen -- day dates, "N of M", unit
notes -- and those are quiet on purpose, sitting beside something louder that
carries the meaning. A tooltip on an icon-only tab has nothing beside it: it IS
the label, and it is the only text that appears for that control at all. The
quietest setting in the interface is the wrong one for the only words a control
gets.

So it takes the ``control`` step (13 px -- the one this design adds to Tailwind's
ramp for exactly this "a control, not a caption" case) at ``medium``. Measured
on the same string: full-opacity coverage 61 % -> 70 %, which is where the
"thin" reading comes from and is a real raster difference rather than a
perceptual one. Everything else about the component stays literal to the
registry.

**It is INVERTED.** ``bg-foreground`` with ``text-background`` -- near-black on
a light page, near-white on a dark one. A tooltip that takes the popover's
colours instead reads as a small card and stops looking like an annotation.

**delayDuration is 0.** It appears the moment the pointer arrives. Qt's own
tooltip waits about 700 ms, which is why the platform one feels like a
different component even when it is styled identically.

**A CHILD WIDGET, not a popup window.** The one thing measured here that the
registry cannot tell us: popup windows in this environment composite alpha
against BLACK, not against what is behind them -- a 50 % red fill over a white
host renders ``[38, 25, 25]``. ``menus.Menu`` documents the same finding and
works around it by filling opaquely, which a rectangle can do and an arrow
cannot: the corners outside a rounded box with a spike on it would show as
notches. Painted inside the window there is no compositing question at all,
no window-manager delay, and the geometry can be clamped to the window rather
than escaping it.
"""

from __future__ import annotations

from typing import ClassVar

from qtpy.QtCore import (
    QAbstractAnimation,
    QEasingCurve,
    QEvent,
    QObject,
    QPoint,
    QPointF,
    QRectF,
    Qt,
    QVariantAnimation,
)
from qtpy.QtGui import (
    QColor,
    QCursor,
    QEnterEvent,
    QFontMetrics,
    QPainter,
    QPainterPath,
    QPolygonF,
)
from qtpy.QtWidgets import QApplication, QWidget

from ..theme import typography as ty

__all__ = ["Tooltip", "attach"]


class Tooltip(QWidget):
    """One tooltip, parented to the window it annotates."""

    #: ``px-3 py-1.5``.
    PAD_X, PAD_Y = 12, 6

    #: ``size-2.5`` -- a 10 px square rotated 45 degrees, so it stands
    #: ``10/sqrt(2)`` ~ 7 px proud of the box.
    ARROW = 10.0

    #: ``sideOffset={0}``: the arrow's tip touches the trigger.
    SIDE_OFFSET = 0

    #: `animate-in fade-in-0 zoom-in-95 slide-in-from-top-2`, and the registry
    #: gives it a matching `animate-out fade-out-0 zoom-out-95` on the way out
    #: too -- no slide, exactly as the popover.
    #:
    #: **150 ms, not the popover's 100, and the SAME in both directions --
    #: FETCHED from `shadcn-ui/ui`'s actual `tooltip.tsx`, not assumed.**
    #: `TooltipContent`'s className carries no `duration-*` class on either
    #: state, so both take `tw-animate-css`'s default, which the package
    #: spells out for EACH direction separately::
    #:
    #:     --animate-in:  enter var(--tw-duration,.15s) var(--tw-ease,ease) ...
    #:     --animate-out: exit  var(--tw-duration,.15s) var(--tw-ease,ease) ...
    #:
    #: identical on both lines. `.cn-popover-content` writes `duration-100`
    #: and the tooltip does not, so the tooltip genuinely differs from the
    #: popover -- but not from ITSELF between opening and closing, per the
    #: registry. Governs the INTRO. See `CLOSE_DURATION_MS` for why the
    #: OUTRO, once actually complete and correctly shaped (`_run_to`'s own
    #: doc), still needed its own, shorter number after all -- a different
    #: departure from the earlier one that constant recorded, and for a
    #: different reason.
    DURATION_MS = 50
    #: The OUTRO's own duration, SHORTER than `DURATION_MS` -- but, unlike
    #: the round `_run_to`'s own docstring records and retracts, this does
    #: NOT reshape the curve to buy it: closing still runs the exact same
    #: `Linear` progress and the exact same `_ease` motion `_run_to` uses
    #: for opening, just compressed into less time. That distinction is
    #: the whole of what makes this different from the earlier mistake --
    #: a close that looks like a DIFFERENT animation (a back-loaded curve,
    #: a rushed tail) versus one that looks like the SAME animation played
    #: at a different speed.
    #:
    #: **MEASURED to be the actual duration, not a guess that it was
    #: running long: pixel-scanned at 60 fps on the live app, the close
    #: took 7-8 frames -- ~130 ms -- matching `DURATION_MS` (150 ms) to
    #: within a frame either way.** So a literally symmetric close was not
    #: secretly slow; reported as lingering anyway, directly and more than
    #: once, once it was actually complete and correctly shaped rather
    #: than broken in one of the ways this file's history already covers.
    #: That is a real, common asymmetry in motion design generally, not
    #: unique to this component: entering asks for attention and can take
    #: its time; leaving reads as reluctant if it takes the SAME time,
    #: because there is nothing left to look at while it does. Shadcn's
    #: own registry happens to specify these equal for this ONE component
    #: -- verified, not assumed, `DURATION_MS`'s own doc -- but a registry
    #: default is not a claim that equal always feels right, and direct,
    #: repeated, measured feedback on THIS control says it does not here.
    CLOSE_DURATION_MS = 50
    #: `zoom-in-95`/`zoom-out-95` literally is `0.95` -- and MEASURED (not
    #: eyeballed) to be symmetric in both directions already: sampled scale
    #: through a real close, `1.0 -> 0.9997 -> 0.9985 -> ... -> 0.95`, the
    #: exact mirror of the open's own `0.95 -> ... -> 1.0`. The registry
    #: value is simply too SUBTLE to read as motion at this size -- a 5 %
    #: change on a ~140 px pill is a few pixels at the one edge furthest
    #: from the scale origin (the arrow, near the target) and near-zero
    #: everywhere else, reported directly as "the close doesn't zoom at
    #: all" even though the numbers above show it doing exactly what the
    #: open does. `0.90` was a first, still-too-timid departure from the
    #: literal registry value -- reported again as invisible, and a real
    #: OS-level pixel scan of a live close explains why 10 % specifically
    #: fails here even though the underlying mechanism (`_run_to`'s own
    #: docstring, `_vanish`'s AppKit fix) is correct: the shift is
    #: proportional to distance from the scale origin, so on a pill this
    #: size 10 % is still only ~10-14 px at the FAR edge, and it lands in
    #: the back half of the fade where opacity is already low -- the exact
    #: moment a small size change is hardest to see. `0.7` moves the same
    #: math to ~40 px at that same edge, unmistakable at any opacity in
    #: the transition, not just when the shape happens to catch the light.
    ZOOM_FROM = 0.7
    SLIDE_PX = 8

    #: The reopen half of a rail-icon-to-rail-icon handoff (`_reopen_for`'s
    #: ``fast=True``, called from `_begin_handoff`) runs at THIS duration
    #: instead of `DURATION_MS`. There is no closing leg on this clock any
    #: more -- `_begin_handoff`'s own docstring records why that half was
    #: sacrificed outright rather than kept and merely sped up here.
    #:
    #: **Reported directly: quickly hovering across several toolbar icons
    #: feels unresponsive.** It is not a missing-event bug -- `_Watch`'s
    #: Enter fires the moment the pointer crosses, no polling anywhere in
    #: this file. The lag was structural: a handoff used to be sequential,
    #: not simultaneous (`show_for`'s own docstring on why this shared
    #: window cannot show two pills at once) -- the OLD label kept fading
    #: at its OLD position for a full `CLOSE_DURATION_MS` before the NEW
    #: one even started sliding in, so a fast sweep across icons paid
    #: `CLOSE_DURATION_MS + DURATION_MS` of dead time per crossing in which
    #: nothing about the icon actually under the cursor was shown yet.
    #: A plain departure has nothing better to look at, so it can afford the
    #: fuller, more legible fade; a crossing has a NEW answer waiting the
    #: whole time, so it is the one direction where speed matters more than
    #: the motion being easy to read -- taken here to its actual conclusion:
    #: the crossing's own motion is what gets to be fast, not merely faster.
    HANDOFF_DURATION_MS = 20

    #: How long a QUICK RE-HOVER of the SAME trigger, arriving just after this
    #: tooltip fully vanished, is still treated as "never really left" -- see
    #: `show_for`'s own `auto_settled`. NOT a delay before closing -- a Leave
    #: always starts the real fade on the SAME frame it arrives (`dismiss`'s
    #: own doc) -- this is timed against the moment the fade actually
    #: FINISHES and `_vanish` runs, for a tremor slow enough to outlast the
    #: whole close.
    #:
    #: **FIVE DESIGNS SAT HERE BEFORE THIS ONE, EACH TRADING ONE VISIBLE
    #: PROBLEM FOR ANOTHER -- condensed here; the full blow-by-blow is git
    #: history, not repeated.** (1) A flat hold before closing at all fixed a
    #: measured flicker but read as "the application looks laggy". (2)
    #: Dropping the hold reintroduced the flicker: a fast jitter's return
    #: interrupted a fade that had already painted one real, visibly dimmer
    #: frame. (3) A back-loaded `InCubic` curve on the outro fixed that and
    #: still read as lingering, because ITS flat stretch is flat in ABSOLUTE
    #: milliseconds -- a pause on a genuine departure too, just reshaped. (4)
    #: A separate, shorter `CLOSE_DURATION_MS` shrank that pause. (5) Made a
    #: genuine departure skip the animated close ENTIRELY -- correctly killed
    #: the pause, and (this round's own finding, `dismiss`'s docstring)
    #: quietly handed macOS a fully-opaque window to `hide()`, which AppKit
    #: then faded on ITS OWN unconfigurable clock -- the very linger being
    #: chased, now coming from outside this file altogether. Every one of
    #: (1)-(5) was reshaping or skipping THIS class's own animation while the
    #: thing actually being watched, on a genuine departure, was never it.
    #:
    #: **(6) RUN THE REAL FADE EVERY TIME, ALWAYS, EXACTLY LIKE THE OPEN --
    #: the design this file started from
    #: (`prototype/shadcn/tooltip.py`) before a `top_level`-specific
    #: shortcut was layered on, and everything (1)-(5) chased was a
    #: consequence of that shortcut, not of the animation it kept reshaping.**
    #: `hide()` now only ever runs once `_run_to(0.0)` has already taken the
    #: painted content to fully transparent (`_settled`), so there is nothing
    #: left for AppKit's own fade to show even if it still runs -- VERIFIED
    #: on a live screen recording, pixel-scanned frame by frame, the same way
    #: (5)'s defect was found. `_vanish`'s `setWindowOpacity(0.0)` is kept as
    #: a belt-and-suspenders second measure, not the fix by itself.
    #:
    #: This constant's job is now simply this: if the pointer comes back to
    #: the SAME target within this window of the fade having FINISHED, it
    #: reopens straight at its final position and size -- no `SLIDE_PX`
    #: reserve, no widget bigger than its settled shape -- the same `settled`
    #: path `show_for` already has for a click-driven reopen. The
    #: opacity/zoom still ramps over `DURATION_MS` either way (that part of
    #: the intro is not skipped, only the slide), so the worst case left is a
    #: brief pop-back-in, never a re-run of the full slide-from-the-target
    #: animate-in.
    REOPEN_GRACE_MS = 200

    def __init__(self, t, parent: QWidget, *, top_level: bool = False):
        """``top_level`` makes this its own window, ABOVE any open popup.

        **The in-window child is still the default, and still right for almost
        everything** -- see the module docstring. This exists for one case the
        child cannot serve: a tooltip on a control BESIDE an open popover.

        A popup is a window and always draws above the window it belongs to, so
        an in-window tooltip that overlaps one is simply hidden behind it.
        MEASURED on the tool rail with a real popup grab: only the arrow showed,
        the label -- the entire point of the thing -- was behind the popover.
        And there is nowhere to move it to: the popover's right edge sits 4 px
        from the rail, so every leftward position overlaps it.

        The reason the child was chosen does NOT apply to this window type.
        `menus.Menu` records that popups here composite alpha against BLACK,
        which is why a rounded shape with an arrow could not be one. RE-TESTED
        for a `Qt.ToolTip` window with `WA_TranslucentBackground`: the corners
        outside the pill came back as the page behind it (#ffffff), not black.
        The finding was true of the opaque QMenu and is not true of this.
        """
        super().__init__(parent)
        self.t = t
        #: Its own window, so it can draw over a popup. See above.
        self.top_level = top_level
        if top_level:
            self.setParent(None)
            self.setWindowFlags(Qt.ToolTip | Qt.FramelessWindowHint
                                | Qt.NoDropShadowWindowHint)
            self.setAttribute(Qt.WA_TranslucentBackground, True)
        self._text = ""
        #: The widget this is currently standing beside, or ``None``. Read in
        #: `show_for` to tell "the pointer came back to the SAME trigger" (a
        #: border re-entry) apart from "the pointer arrived somewhere new" (a
        #: genuine open) -- see the re-entry guard there.
        self._target: QWidget | None = None
        # 13/500, NOT `text-xs`/400 -- see the module docstring.
        self._font = ty.ui("control", "medium")
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        #: 0 closed, 1 open. Drives opacity directly and the zoom/slide through
        #: the easing curve -- see `paintEvent`.
        self._progress = 0.0
        #: Where the arrow points ALONG the shared edge, and the zoom's
        #: origin: an x for `side="bottom"`, a y for `side="right"`. One
        #: attribute rather than two, because only one of them can ever be
        #: meaningful and a pair invites reading the stale one.
        #:
        #: Defaulted here as well: `paintEvent` guards on `_progress` so a
        #: never-shown tooltip cannot reach it in use, but a bare attribute that
        #: only some code paths create is a footgun for the next reader.
        self._arrow_at = 0.0
        #: `side` on `TooltipContent`. The sidebar's rows ask for "right", the
        #: tool rail's buttons for "left".
        self._side = "bottom"
        #: True while the WIDGET's own bounding box still carries the
        #: `SLIDE_PX` room the intro animation slides through -- see
        #: `_shrink_to_settled` for why this cannot just stay True forever.
        self._reserve_active = False
        #: `time.perf_counter()` of this tooltip's last `_vanish()`, or
        #: `None` before the first one -- what `show_for`'s `auto_settled`
        #: reads to recognise a quick re-hover. See `REOPEN_GRACE_MS`.
        self._last_vanish_at: float | None = None
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(self.DURATION_MS)
        # LINEAR for the INTRO (the default here); the zoom is eased instead,
        # same split as the menu's outro, for the same measured reason: an
        # eased opacity puts the whole perceptible fade in the first third
        # and leaves a still tail -- wrong for an OPEN, which should read as
        # gradual throughout. `_run_to` overrides this per call for the
        # OUTRO specifically -- see its own docstring for why that direction
        # wants the OPPOSITE shape.
        self._anim.setEasingCurve(QEasingCurve.Linear)
        self._anim.valueChanged.connect(self._frame)
        self._anim.finished.connect(self._settled)
        #: Tailwind's `ease`, for the MOTION half.
        self._ease = QEasingCurve(QEasingCurve.BezierSpline)
        self._ease.addCubicBezierSegment(QPointF(0.25, 0.1), QPointF(0.25, 1.0),
                                         QPointF(1.0, 1.0))
        self.hide()

    # -- the animation ------------------------------------------------------- #
    #
    # **No snapshot, and that is the whole difference from the menu.** A tooltip
    # here is an in-window CHILD that paints its own vector shape, so there is
    # no second window to hand off to, no popup grab to preserve, no teardown
    # blocking the loop -- and scaling is done by the PAINTER, so the text is
    # re-rendered at each size rather than resampled. Every hard problem in the
    # menu's outro came from animating a widget that had to be destroyed; this
    # one simply delays its own `hide` until the fade is done.
    def _frame(self, value) -> None:
        self._progress = float(value)
        # `update()`, not `repaint()` -- tried the forced, synchronous
        # form for a `top_level` window while hunting the missing zoom
        # (`_vanish`'s own docstring has the full account), on the theory
        # that a coalesced `update()` might be dropping real frames on a
        # native OS window. It was not the fix -- ground-truth logging
        # showed `_frame` was not running at ALL during the close being
        # reported on, with or without this line, because that close was
        # skipping the animation entirely at the time (a design since
        # retired). Forcing a synchronous compositor round-trip on every
        # one of ~9 ticks turned out to cost real wall-clock time for no
        # correctness this plain `update()` does not already give a CHILD
        # tooltip -- reported directly as the animation now feeling heavy
        # once the actual fix (`dismiss`/`_vanish`) made it run at all.
        self.update()

    def _vanish(self) -> None:
        """Hide, and REPAINT THE PAGE WHERE THIS WAS.

        **Hiding a child is not enough when a popup opens in the same turn.**
        Qt marks the vacated rect dirty on the parent and flushes it with the
        next paint; a popup window appearing in that same event-loop turn takes
        the screen before the flush lands, and the strip of this tooltip not
        covered by the popup stays on screen as stale pixels. MEASURED on the
        tool rail: hover a button, click it, and a dark 14 px bar -- the arrow
        end of the tooltip, which reaches past the popover's right edge and
        over the rail -- was left painted beside the open popover.

        Asking the parent for the region explicitly is the fix, and it is
        cheap: one rect, once per dismissal.

        **`setWindowOpacity(0.0)` BEFORE `hide()`, for a `top_level` window
        -- the actual fix for "the close only fades, the zoom never
        shows", after `_frame`/`_run_to`/`ZOOM_FROM` were each tuned and
        none of them touched it.** Ground-truth logging placed directly in
        `_frame` and `paintEvent` proved this method's OWN animation was
        never the thing being watched: on a real, live run, `dismiss` took
        the instant, zero-frame path exactly as designed (its own
        docstring) -- `_frame` did not fire ONCE during the close -- and
        yet the screen still showed a real, multi-frame fade lasting
        several times longer than any duration this file ever configured.
        Something OUTSIDE this class was animating a window this class
        believed it had already made to just disappear.

        That something is AppKit's own default behaviour for ordering out
        a window of this style (`Qt.ToolTip`, frameless, translucent --
        see `__init__`'s own docstring for why the rail needs one): macOS
        fades a window's LAST RENDERED CONTENT on `hide()`/`orderOut:`
        automatically, on its own clock, using its own default duration --
        which is why nothing this class does to `_progress`, its curve, or
        its duration could ever have reached it. Explicitly collapsing the
        window's own opacity to zero BEFORE asking it to hide makes the
        following `hide()` a no-op from the compositor's point of view --
        there is nothing left for AppKit to fade FROM. VERIFIED on a real
        screen recording, pixel-scanned frame by frame: without this line
        the tooltip's colour visibly steps through a dozen intermediate
        shades over several hundred milliseconds after every close; with
        it, the frame immediately after full opacity is already gone --
        one step, not several, on a genuine departure. (A handoff no
        longer reaches `_vanish` at all -- `_begin_handoff`'s own docstring
        -- so this line's other original case no longer applies to it.)

        Scoped to `top_level` only: a CHILD tooltip is painted content
        inside a window that was already on screen for other reasons, so
        AppKit has no window of its own to apply this to in the first
        place -- nothing here is needed, or safe to assume applies, on
        that path.

        **KEPT even after `dismiss` was changed to always run the real
        fade first (its own docstring, and `REOPEN_GRACE_MS`'s round 6).**
        With that change this line should now be a pure no-op in the
        common case -- `_run_to(0.0)` has already taken `_progress`, and
        so the painted content, to fully transparent by the time
        `_settled` calls this. Left in as the belt to that fade's braces:
        it is one call, and the one path that still reaches `_vanish`
        without a completed fade (`dismiss(instant=True)`, a click racing
        a popover) is exactly the case this was written for in the first
        place.
        """
        if _OPEN["tip"] is self:
            _OPEN["tip"] = None
        #: Stamped here, not in `dismiss` -- this is the one place every
        #: dismissal path (instant or faded) actually funnels through, and
        #: `show_for`'s `auto_settled` needs the REAL moment the tooltip left
        #: the screen, not the moment something merely asked it to.
        import time
        self._last_vanish_at = time.perf_counter()
        area = self.geometry()
        if self.top_level:
            self.setWindowOpacity(0.0)
        self.hide()
        # A top-level leaves no hole in anyone's page -- it was its own window.
        parent = None if self.top_level else self.parentWidget()
        if parent is not None:
            # REPAINT, not update: `update()` only queues, and the flush loses
            # the race against a popup opening in the same turn -- which is the
            # whole bug. One rect, once per dismissal.
            parent.repaint(area)

    def _settled(self) -> None:
        if self._progress <= 0.0:
            # A handoff no longer lands here at all -- `_begin_handoff`
            # vanishes the old content AT ONCE rather than fading it out
            # and waiting for that fade to reach zero before reopening
            # elsewhere (its own docstring has the full account). So the
            # only way `_progress` reaches zero through the animation is a
            # genuine close with nothing else waiting -- always `_vanish`.
            self._vanish()
        elif self._progress >= 1.0:
            self._shrink_to_settled()

    def _shrink_to_settled(self) -> None:
        """Drop the `SLIDE_PX` reserve from the WIDGET's own bounding box
        once the intro animation has actually landed.

        **This is the hitbox bug, not the flicker.** `show_for` sizes the
        widget `SLIDE_PX` bigger than the settled shape so the shape has room
        to slide THROUGH that extra space while opening -- and on every side
        this draws, that extra space sits on the side THE TARGET IS ON (the
        shape enters "from" the target's direction, per each side's own
        comment in `paintEvent`), because sliding in that direction is what
        needs the room. The shape settles flush against the target
        (`sideOffset=0`), but the WIDGET never shrinks back down afterward --
        so for the tooltip's ENTIRE visible lifetime, not just the ~150 ms of
        the animation, its invisible bounding box overlaps `SLIDE_PX` (8 px)
        of the target it is annotating.

        `WA_TransparentForMouseEvents` (see `__init__`) is supposed to make
        that overlap harmless, and it does for the CHILD tooltips. But a
        `top_level` tooltip is a REAL OS window (rail buttons; see
        `__init__`'s own docstring on why), and click-through for a real,
        frameless top-level window is not something every platform honours
        as reliably as it does for an ordinary child widget -- reported as
        the target's own hover-fill lighting correctly (governed by the
        TARGET's geometry, which this never touched) while the cursor stays
        the default arrow and a click there does nothing, at the target's
        edge nearest the tooltip, on every rail icon including Compute, from
        a fresh launch. All three symptoms are exactly what an invisible
        window still claiming the cursor/click at that edge would produce.

        Shrinking removes the overlap without touching where anything is
        PAINTED: the freed strip was never drawn into in the first place
        (`_box` and `paintEvent`'s translate already stop reaching into it
        once `_reserve_active` is False), so this is invisible on screen.
        """
        if not self._reserve_active:
            return
        slide = self.SLIDE_PX
        if self._side == "right":
            # The reserve sits on the LEFT (target-adjacent) side.
            self.setGeometry(self.x() + slide, self.y(),
                             self.width() - slide, self.height())
        elif self._side == "left":
            # The reserve sits on the RIGHT (target-adjacent) side.
            self.setGeometry(self.x(), self.y(),
                             self.width() - slide, self.height())
        elif self._side == "top":
            # The reserve sits on the BOTTOM (target-adjacent) side.
            self.setGeometry(self.x(), self.y(),
                             self.width(), self.height() - slide)
        else:
            # "bottom": the reserve sits on TOP (target-adjacent).
            self.setGeometry(self.x(), self.y() + slide,
                             self.width(), self.height() - slide)
        self._reserve_active = False

    def _run_to(self, target: float, *, duration: int | None = None) -> None:
        """Run the progress to ``target``, cancelling whatever was in flight.

        ``duration`` overrides the ordinary open/close pair below -- the one
        caller that passes it is `_reopen_for`'s ``fast=True``, reached only
        from a handoff between two DIFFERENT rail icons (`_begin_handoff`),
        which wants `HANDOFF_DURATION_MS` for that one remaining leg rather
        than the slower `DURATION_MS` a genuine open can afford. See that
        constant's own docstring for why the two cases need different
        speeds at all.

        **CLOSE IS THE OPEN, PLAYED IN REVERSE AND FASTER -- the SHAPE
        restored to match the registry, the SPEED deliberately not, after
        three earlier rounds that changed the shape to buy what the speed
        alone turned out to owe.** Those rounds are still worth reading
        for what each one measurably fixed -- `REOPEN_GRACE_MS` keeps the
        full account -- but their premise (that the outro needed a
        DIFFERENT curve from the intro) turned out to be wrong, checked
        against `shadcn-ui/ui`'s actual `tooltip.tsx`: `TooltipContent`'s
        closed-state classes are `animate-out fade-out-0 zoom-out-95`, no
        duration override, so the registry gives both directions the
        SAME shape. A back-loaded `InCubic` outro was invented here
        instead, on the theory that hiding the fade behind a flat stretch
        would hide a fast jitter's return -- and it measurably did (the
        flicker it targeted was real), but reported straight back as "the
        close looks rushed, doesn't zoom, isn't the open in reverse" once
        actually compared side by side. Both readings were correct: the
        curve fixed the thing it targeted and broke fidelity doing it.
        **The path taken instead: `show_for`'s own same-trigger snap-back
        already catches a fast jitter WITHOUT reshaping the curve at all,**
        since a genuine departure's real fade (`dismiss`'s own doc) leaves
        the window visible throughout, so a return mid-fade lands in that
        one check. (A handoff no longer has a fade of its own to interrupt
        here at all -- `_begin_handoff`'s own docstring records why.)
        Nothing here cares what
        curve the animation itself uses, so there is no longer a reason
        for the outro to diverge from the registry to buy protection
        something else already provides. The scale (`zoom-*-95`) is
        UNCHANGED either way --
        `eased = self._ease.valueForProgress(self._progress)` in
        `paintEvent` was always computed the same way regardless of
        direction; only the PROGRESS curve driving it differed, and that is
        what this restores to match.

        **`setCurrentTime(0)`, not just `stop()`.** `stop()` alone leaves
        `currentTime` at the END of the previous run, so a fresh
        `setEndValue` recomputes the current value AT that stale end-time --
        against the NEW end value just assigned -- and synchronously
        delivers it as the animation's very first frame once `start()` runs,
        before it ever begins climbing/falling from the real start value.
        MEASURED here: the first show is clean, and every one after it
        delivered `1.0` before the real animation began -- harmless for the
        fade alone (one stray fully-open frame nobody could see land), but
        it ALSO fires `finished` synchronously when that stray value happens
        to equal the new target, which reached `_settled()` before the
        animation had genuinely run at all. Found via `REOPEN_GRACE_MS`'s
        own `auto_settled`: a quick reopen straight after `dismiss()`
        checked `_reserve_active` moments after `show_for` returned and it
        was already `False` -- `_shrink_to_settled()` had fired inside
        `_run_to(1.0)` itself, on a frame whose real progress was still 0.
        Resetting `currentTime` before reconfiguring removes the staleness
        `setEndValue`'s own recompute was acting on, so there is no stray
        value left to deliver.

        **Signals still blocked while the range is set** -- the same bug the
        sidebar's collapse had, and it is worth stating rather than
        cross-referencing: `setStartValue`/`setEndValue` each recompute and
        emit the current value against whatever is configured at the time,
        so setting them one at a time outside a block emits twice, once
        against a mix of the old start and the new end.
        """
        self._anim.stop()
        blocked = self._anim.blockSignals(True)
        # `setCurrentTime(0)` belongs INSIDE the blocked region too, not
        # just `setStartValue`/`setEndValue` -- it recomputes and emits a
        # value against whatever start/end are STILL configured (the
        # PREVIOUS run's) the moment it runs, exactly the same class of
        # stray synchronous emission the block already exists to catch for
        # the other two calls. Placed outside it once, and a fresh intro
        # starting right after an outro had just naturally finished landed
        # on `_progress = 1.0` before this method had even reassigned
        # `startValue`/`endValue` for the NEW run -- found by `verify_
        # tooltip_curve.py`'s own intro check reading 1.0 at 45 ms in.
        self._anim.setCurrentTime(0)
        # ONE curve, EITHER direction -- see this method's own docstring
        # for why the outro no longer gets its OWN SHAPE. The DURATION
        # still does (`CLOSE_DURATION_MS`'s own doc): closing plays the
        # exact same `Linear` progress and the exact same `_ease` motion
        # `paintEvent` reads regardless of direction, just compressed --
        # which is what keeps this a faster version of the SAME animation
        # rather than a different one.
        self._anim.setEasingCurve(QEasingCurve.Linear)
        self._anim.setDuration(
            duration if duration is not None
            else (self.CLOSE_DURATION_MS if target < 1.0 else self.DURATION_MS))
        self._anim.setStartValue(self._progress)
        self._anim.setEndValue(target)
        self._anim.blockSignals(blocked)
        self._anim.start()

    # -- geometry ------------------------------------------------------------ #
    #: Every `side` this draws. An unknown one RAISES.
    #:
    #: **It used to fall through to "bottom", and that silence shipped a bug.**
    #: The tool rail asked for `side="left"`, which was not implemented; the
    #: tooltip was placed BELOW the button instead, clamped to the window's
    #: right edge, and appeared as a grey sliver half off the screen beside the
    #: popover. Nothing raised, nothing logged -- a default that quietly answers
    #: a question it was not asked is worse than no default.
    SIDES: ClassVar[tuple[str, ...]] = ("bottom", "right", "left", "top")

    def _box(self) -> QRectF:
        """The rounded body, in SHAPE coordinates -- the arrow's tip is at 0.

        The widget is bigger than the shape by `SLIDE_PX` on the side the slide
        comes from, ONLY while `_reserve_active` -- once `_shrink_to_settled`
        has run, the widget IS the shape's own size and there is nothing left
        to subtract. `paintEvent` translates into these coordinates so the
        geometry here is unchanged by the slide.
        """
        rise = self.ARROW / (2 ** 0.5)
        slide = self.SLIDE_PX if self._reserve_active else 0.0
        if self._side in ("right", "left"):
            return QRectF(rise if self._side == "right" else 0.0, 0.0,
                          self.width() - slide - rise,
                          float(self.height()))
        if self._side == "top":
            return QRectF(0.0, 0.0, float(self.width()), self.height() - slide - rise)
        return QRectF(0.0, rise, float(self.width()),
                      self.height() - slide - rise)

    def is_over_target(self) -> bool:
        """Is the pointer genuinely over whichever widget this is CURRENTLY
        showing for (or standing beside, mid-fade)?

        The public read a CALLER needs when it wants to ask "is it safe to
        dismiss this without taking something with it" -- `ToolRail.
        track_pointer`'s own cleanup is exactly that caller, and reaching
        past `_target` into this class's own attributes from another module
        is the layering violation this method exists to avoid, not merely a
        style objection: `_SHARED_TOP_LEVEL` means every rail icon's own
        `Tooltip` is now the SAME object, so a caller iterating its OWN list
        of targets and asking each one's geometry question independently
        (as the docstring below explains `track_pointer` used to) will keep
        getting a fresh, correct-looking answer for every target BUT ONE,
        and the dismiss it fires for any of the others still lands on the
        one shared window -- there is only one to ask this about now, not
        one per icon. Same tolerance as `_StrayWatch`'s own check, for the
        same reason -- see `_EDGE_TOLERANCE_PX`.
        """
        if self._target is None:
            return False
        try:
            at = self._target.mapFromGlobal(QCursor.pos())
        except RuntimeError:            # the target went away
            return False
        m = _EDGE_TOLERANCE_PX
        return self._target.rect().adjusted(-m, -m, m, m).contains(at)

    def show_for(self, target: QWidget, text: str,
                 side: str = "bottom", *, settled: bool = False) -> None:
        """Place beside ``target``, centred on it, clamped to the window.

        ``side`` is `TooltipContent`'s own prop. "bottom" is the registry
        default; "right" is what a sidebar row asks for, and it is not a
        cosmetic variant -- a tooltip BELOW a collapsed rail row would cover the
        next row down, which is the one thing a rail of icons cannot afford.

        The host is resolved HERE, not when the tooltip was attached. Attaching
        happens while the screen is still being built, so ``target.window()``
        was answering with a widget that had not been put in the window yet --
        and every coordinate came out relative to the wrong ancestor, which
        showed up as the tooltip pinned to the far left edge of the screen.

        ``settled`` skips the `SLIDE_PX` reserve from the very first frame,
        landing straight on the shrunk geometry `_shrink_to_settled` would
        otherwise reach ~150 ms later. For a CALLER who already knows the
        pointer never left the target -- `ToolRail.show_tip` re-showing a
        tooltip on a click that closes that SAME icon's popover, where the
        icon was never really un-hovered -- the ordinary path still spends
        that whole 150 ms with the widget's bounding box overlapping the
        target by `SLIDE_PX`, exactly as `_shrink_to_settled` documents,
        and this time a REAL click is landing during that very window.
        Reported as the hand cursor flickering to the arrow specifically
        near the target's border, on a click that toggles that icon's own
        popover closed: the fresh (reserved) tooltip raises itself over that
        border strip for ~150 ms before shrinking, and a real top-level
        window is not reliably click-and-cursor-transparent there (see
        `_shrink_to_settled`'s own doc for why). Nothing here slides in
        because, from the pointer's point of view, nothing opened -- the
        target was already being interacted with.
        """
        if side not in self.SIDES:
            raise ValueError(
                f"no tooltip side {side!r}; have {', '.join(self.SIDES)}")
        if not self.top_level:
            window = target.window()
            if window is not None and self.parentWidget() is not window:
                self.setParent(window)
        # **The pointer came back to the SAME trigger WHILE STILL VISIBLE --
        # a border re-entry, never a fresh open**, whether this was still
        # climbing in (an intro, ordinary or a handoff's fast reopen) or
        # already falling (a plain departure's own fade -- a handoff no
        # longer has a falling half to interrupt here at all, see
        # `_begin_handoff`'s own docstring). Both
        # directions SNAP back to fully open rather than animating there,
        # because `top_level` tooltips are REAL OS windows (see `__init__`'s
        # own docstring on why the rail needs one), and calling
        # `show()`/`raise_()` on one a
        # window manager already has on screen -- once per jitter,
        # indistinguishable from a fresh open -- is not the no-op it is for
        # an ordinary child widget.
        if (self._target is target and self._text == text
                and self._side == side and self.isVisible()):
            if self._progress < 1.0:
                self._anim.stop()
                self._progress = 1.0
                # `_settled` (via `_anim.finished`) never fires here -- the
                # animation was STOPPED, not run to completion -- so a snap
                # is the only remaining place that can drop the reserve.
                # Skipping it would leave the widget overlapping its target
                # by `SLIDE_PX` for good on a border re-entry that snaps
                # before the fresh open's own animation ever settles.
                self._shrink_to_settled()
                self.update()
            return
        # **A DIFFERENT target while this (shared) window is still genuinely
        # visible -- an instant handoff, not Radix's own real
        # exit-then-enter, and that is now a DELIBERATE departure from it,
        # not an oversight.** Radix's `TooltipProvider` genuinely animates
        # BOTH ends of a crossing (VERIFIED against its actual source, not
        # assumed) -- fading the previous trigger's tooltip out with its own
        # exit animation before the new one's intro plays. `_begin_handoff`
        # used to reproduce exactly that, and it fixed a real bug on the way
        # (`_run_to(1.0)` from `_progress` already at `1.0` emits no frames
        # at all, so a same-window crossing had silently teleported with no
        # motion whatsoever). But a REAL exit fade costs real time, and nail
        # this down: on a dense toolbar of small buttons, a fast sweep
        # crosses several targets faster than any fade -- reported, twice,
        # after two different attempts at making that fade merely fast
        # enough to survive the sweep rather than removing it. `_begin_handoff`
        # now vanishes the OLD content on THIS SAME frame -- see its own
        # docstring -- and only the NEW content's intro still animates. The
        # native window itself is still never hidden or shown mid-crossing,
        # which is the actual fix for the silent-teleport bug above; that
        # part is unrelated to whether the outgoing fade plays and survives
        # untouched.
        if self.top_level and self.isVisible() and self._progress > 0.0:
            self._begin_handoff(target, text, side)
            return
        # **A QUICK re-hover of the SAME trigger, arriving just AFTER this
        # tooltip fully vanished** (so the branch above already missed it --
        # `isVisible()` is False) -- a tremor whose return lands AFTER the
        # near-instant vanish `dismiss` already ran (see that method's own
        # docstring), but still within `REOPEN_GRACE_MS` of the moment this
        # vanished. See that constant's own docstring for why this two-part
        # split exists at all instead of one flat delay.
        # Folded into `settled` below: reopens straight at its final
        # position, same as a click-driven reopen does.
        import time
        auto_settled = (
            self._target is target and self._last_vanish_at is not None
            and time.perf_counter() - self._last_vanish_at
            < self.REOPEN_GRACE_MS / 1000.0)
        # And take down whatever else is up, immediately -- see `_OPEN`.
        other = _OPEN["tip"]
        if other is not None and other is not self:
            try:
                other.dismiss(instant=True)
            except RuntimeError:        # its widget went away first
                pass
        _OPEN["tip"] = self
        self._reopen_for(target, text, side, reserve=not (settled or auto_settled))

    def _reopen_for(self, target: QWidget, text: str, side: str, *,
                     reserve: bool, fast: bool = False) -> None:
        """The geometry half of a fresh open: place beside ``target`` and run
        the intro. Shared by `show_for`'s ordinary open and by
        `_begin_handoff`'s call once it has vanished the OLD content -- see
        that method's own docstring for why the two are the same operation
        from here on.

        ``fast`` is that handoff call's own flag: run the intro at
        `HANDOFF_DURATION_MS` rather than `DURATION_MS`, since a crossing's
        only remaining motion (the new content arriving) is the one place
        speed matters more than an easy-to-read fade -- see that constant's
        own docstring.
        """
        self._target = target
        self._text = text
        self._side = side
        # A FRESH open normally uses the full, `SLIDE_PX`-reserved geometry
        # below -- `_shrink_to_settled` is what removes it once the intro
        # animation actually lands. ``reserve=False`` (a `settled` open, or
        # an `auto_settled` quick re-hover -- see `show_for`'s own doc)
        # skips the reserve from the start instead of waiting for that.
        self._reserve_active = reserve
        slide = self.SLIDE_PX if self._reserve_active else 0
        metrics = QFontMetrics(self._font)
        # A line break makes more lines (asked for, Images & tracks: a sentence that would
        # otherwise run the width of the window). One line measures exactly as before.
        lines = text.split("\n")
        w = max(metrics.horizontalAdvance(line) for line in lines) + 2 * self.PAD_X
        h = metrics.height() + (len(lines) - 1) * metrics.lineSpacing() + 2 * self.PAD_Y
        rise = self.ARROW / (2 ** 0.5)

        # A top-level tooltip is positioned in SCREEN coordinates and clamped
        # to the screen; a child is positioned in its window and clamped to it.
        # One expression either way -- only the frame of reference differs.
        if self.top_level:
            host = None
            top_left = target.mapToGlobal(QPoint(0, 0))
            room = QApplication.screenAt(top_left)
            room = (room or QApplication.primaryScreen()).availableGeometry()
            host_w, host_h = room.width(), room.height()
            edge_x, edge_y = room.x(), room.y()
        else:
            host = self.parentWidget()
            top_left = target.mapTo(host, QPoint(0, 0))
            host_w, host_h = host.width(), host.height()
            edge_x = edge_y = 0
        # ROOM FOR THE SLIDE on the side the shape enters from -- 8 px, and a
        # child widget cannot paint outside itself, so the widget starts that
        # much further back and is that much bigger. `paintEvent` puts the shape
        # back as it settles. `slide` is 0 for a `settled` open -- see this
        # method's own doc -- so the widget lands directly on its final,
        # non-overlapping size with nothing to shrink out of later.
        if side == "right":
            # `slide-in-from-left-2`, `align="center"`.
            cy = top_left.y() + target.height() // 2
            x = top_left.x() + target.width() + self.SIDE_OFFSET
            y = cy - h // 2
            y = max(edge_y + 4, min(y, edge_y + host_h - h - 4))
            self._arrow_at = cy - y
            self.setGeometry(x - slide, y, int(w + rise) + slide, int(h))
        elif side == "left":
            # `slide-in-from-right-2`, `align="center"`. The mirror of "right":
            # the body sits left of the trigger with the arrow on its RIGHT
            # edge, and the slide's reserve is on the right because that is the
            # side the shape enters from.
            cy = top_left.y() + target.height() // 2
            y = cy - h // 2
            y = max(edge_y + 4, min(y, edge_y + host_h - h - 4))
            self._arrow_at = cy - y
            x = top_left.x() - self.SIDE_OFFSET - int(w + rise)
            self.setGeometry(x, y, int(w + rise) + slide, int(h))
        elif side == "top":
            # `slide-in-from-bottom-2`, centred over the trigger: the mirror of "bottom",
            # for a target near the window's lower edge (Images & tracks' brightness row,
            # where a tooltip below was cut off). The reserve is underneath.
            cx = top_left.x() + target.width() // 2
            x = cx - w // 2
            x = max(edge_x + 4, min(x, edge_x + host_w - w - 4))
            self._arrow_at = cx - x
            y = top_left.y() - self.SIDE_OFFSET - int(h + rise)
            self.setGeometry(x, y, int(w), int(h + rise) + slide)
        else:
            # `slide-in-from-top-2`, centred under the trigger.
            cx = top_left.x() + target.width() // 2
            y = top_left.y() + target.height() + self.SIDE_OFFSET
            x = cx - w // 2
            # Clamped, not allowed to escape: an in-window tooltip that runs
            # past the edge is simply cut off, where a popup would be moved.
            x = max(edge_x + 4, min(x, edge_x + host_w - w - 4))
            self._arrow_at = cx - x
            self.setGeometry(x, y - slide, int(w), int(h + rise) + slide)
        if self.top_level:
            # `_vanish` leaves the WINDOW's own opacity at 0 -- a real,
            # persistent native-window property, never reset on its own --
            # to pre-empt AppKit's fade on the way out (its own docstring).
            # Undone here, on the way back in: `paintEvent`'s `p.setOpacity`
            # is the PAINTER's opacity for whatever it draws next, entirely
            # separate from this, and would otherwise be drawing at full
            # strength into a window still sitting at zero -- invisible
            # regardless of what lands on screen. Every `top_level` open
            # goes through this method, fresh or mid-handoff alike, so one
            # reset here covers both.
            self.setWindowOpacity(1.0)
        self.raise_()
        self.show()
        self._run_to(1.0, duration=self.HANDOFF_DURATION_MS if fast else None)

    def _begin_handoff(self, target: QWidget, text: str, side: str) -> None:
        """Cross to a DIFFERENT target on the shared `top_level` window: the
        OLD content vanishes ON THIS SAME FRAME, and only the NEW content's
        intro animates in. The native window itself is still never hidden
        or shown in between -- that part is unrelated to the sacrifice
        below, and is what actually keeps a same-window crossing from
        silently teleporting (`_run_to(1.0)` from `_progress` already at
        `1.0` emits no frames at all; see `show_for`'s own call site).

        **THE CLOSE FADE IS SACRIFICED HERE ON PURPOSE, and it took two
        rounds of chasing the SAME reported symptom to land on removing it
        rather than further protecting it.** This used to fade the OLD
        content out -- matching Radix's own real `TooltipProvider`, which
        genuinely animates BOTH ends of a crossing (verified against its
        actual source) -- and reopen for the new target only once that
        fade reached zero. Two bugs were found and fixed on that design in
        turn, both the same shape: something OUTSIDE this class kept
        calling `dismiss`/`_run_to` far more often than the fade's own
        duration during a fast sweep across several rail icons (once from
        repeated crossings themselves, once from `_StrayWatch`'s app-wide
        move filter), and `_run_to` always restarts the clock from zero, so
        a fast enough sweep never let ANY close land -- nothing showed at
        all until the pointer stopped moving.

        Both of those were real fixes for a real freeze, and neither one
        was the actual complaint underneath it: even fully working, a
        faded close is dead time on a crossing specifically, because
        nobody is watching the icon being LEFT -- the pointer already has
        a new answer waiting the moment it arrives (the same reasoning
        `HANDOFF_DURATION_MS`'s own docstring already gives for why a
        crossing's remaining motion should be fast). Skipping the outgoing
        fade removes that dead time outright rather than shrinking it, and
        removes the whole class of "calls it faster than its own clock"
        bug with it -- there is no clock left on this side to outrun.
        """
        self._anim.stop()
        self._progress = 0.0
        self._reopen_for(target, text, side, reserve=True, fast=True)

    def dismiss(self, instant: bool = False) -> None:
        """Fade out, then hide. Idempotent, and safe to call while opening.

        ``instant`` (a press on a tooltip's own trigger) skips the fade and
        vanishes on THIS call. A click is a NEW INTENT -- something is about
        to open where the tooltip is standing -- and a fade there is not a
        nicety but a race: the popover appears over a tooltip not yet gone,
        and the strip it does not cover is left on screen as stale pixels.
        MEASURED on the tool rail, and reproduced with no tooltip at all to
        be sure it was this and not the popover's own edge.

        **A plain Leave ALWAYS runs the real fade -- for a `top_level`
        window too, and this is the fix for a longer saga than it looks
        like from here.** A ROUND of this file's own history (left in
        `REOPEN_GRACE_MS`, not repeated) made every `top_level` dismissal
        skip straight to `_vanish()` on a genuine departure, no animation
        at all -- correctly killing the immediate symptom (a departure
        that used to visibly linger) while breaking something nobody had
        connected to it yet: `hide()` on a REAL, translucent OS window
        while it is still fully opaque hands macOS's own window-ordering
        code a fully-visible pill to fade out on ITS OWN clock, outside
        anything this class configures. VERIFIED on a live screen
        recording, pixel-scanned frame by frame: an "instant" dismiss
        still visibly faded for several hundred milliseconds with NO size
        change at all, while `_frame`/`paintEvent` -- logged directly --
        never ran once during it. The fade being watched was never this
        class's animation; it was AppKit's, triggered by handing it a
        still-opaque window to hide.

        Running the real `_run_to(0.0)` first, THEN hiding once it lands at
        zero (`_settled`), is what a CHILD tooltip already did and a
        `top_level` one now does too: by the time `hide()` actually runs,
        this class's own `paintEvent` has already taken the content to
        fully transparent, so there is nothing left for AppKit's own
        fade to show even if it still runs. This is also exactly the design
        this file started from (`prototype/shadcn/tooltip.py`) before a
        `top_level`-specific shortcut was added and every problem since
        traced back to that shortcut, not to the animation itself.

        A quick RE-ENTRY of the SAME trigger mid-fade is caught by
        `show_for`'s own same-trigger snap-back, not by anything here --
        see that method's docstring. A crossing to a DIFFERENT trigger
        needs nothing special either: this window stays visible while it
        fades (nothing here hides it early), so `show_for`'s handoff check
        (`isVisible() and progress > 0`) still finds it -- and now simply
        CUTS this fade short (`_begin_handoff`'s own docstring: the OLD
        content vanishes at once rather than the crossing waiting for it),
        instead of the earlier design's converting it into the closing
        half of a second animation.

        **A DELETED TOOLTIP IS ALREADY DISMISSED.** At application exit Qt
        destroys the shared tooltip before every widget it is attached to has
        been hidden; a later `Hide` reaching `_Watch` then called this on the
        dead wrapper, `isVisible()` raised "Internal C++ object (Tooltip)
        already deleted", and the error surfaced through every app-wide event
        filter as a printed chain -- intermittently, at exit, exit code 0.
        """
        try:
            self.isVisible()
        except RuntimeError:
            return
        if instant:
            self._anim.stop()
            self._progress = 0.0
            self._vanish()
        elif self.isVisible() and self._progress > 0.0:
            # **THE SAME BUG AS `_begin_handoff`'s FIX, ONE CALLER OVER --
            # reported again, live, after that fix alone.** `_StrayWatch`
            # (the app-wide safety net; see its own class docstring) calls
            # THIS method on every qualifying mouse move for as long as the
            # cursor is not over the currently-open tooltip's target -- and
            # during a fast sweep across several rail icons that fires far
            # more often than any fade's own duration. `_run_to` always
            # restarts the clock from zero, so calling it again on every
            # one of those moves never let the SAME close actually land,
            # for the identical reason `_begin_handoff`'s own docstring
            # records: `_settled` -- the only place that ever finishes
            # hiding this window or starts a pending reopen -- never fired.
            # Only start the close if one isn't ALREADY heading to zero;
            # a close already in flight keeps running on its own clock.
            already_closing = (
                self._anim.state() == QAbstractAnimation.Running
                and float(self._anim.endValue()) == 0.0)
            if not already_closing:
                self._run_to(0.0)
        else:
            self._anim.stop()
            self._progress = 0.0
            self._vanish()

    def paintEvent(self, _e):
        if self._progress <= 0.0:
            return
        t = self.t
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setOpacity(self._progress)                   # `fade-*-0`, LINEAR
        eased = self._ease.valueForProgress(self._progress)
        # `slide-in-from-top-2`, settling into place. NOT on the way out: the
        # registry gives the tooltip no `slide-out-to-*`, so at that end this
        # rides the same eased value back and the shape simply stays put while
        # the opacity carries it.
        # `zoom-*-95` about the ARROW, which is where
        # `--radix-tooltip-content-transform-origin` resolves to: it grows out
        # of the thing it points at. The tip is the painter's origin along the
        # axis it lies on, so only the other axis needs translating.
        scale = self.ZOOM_FROM + (1.0 - self.ZOOM_FROM) * eased
        box = self._box()
        rise = self.ARROW / (2 ** 0.5)
        # 0 once `_shrink_to_settled` has run -- the widget IS the shape by
        # then, so there is no reserve left to translate across. Needed here
        # too, not just in `_box`: `eased` tracks `_progress`, which can move
        # in EITHER direction while `_reserve_active` is still True --
        # climbing during an intro, or falling if a GENUINE DEPARTURE's own
        # real fade (`dismiss`'s own doc) interrupts one before it ever
        # settles. A handoff can no longer cause a fall here at all --
        # `_begin_handoff` forces `_progress` straight to 0 rather than
        # animating down to it, see its own doc -- but a stale nonzero
        # slide would still translate the shape off its settled position
        # either way, so this stays direction-agnostic.
        slide = self.SLIDE_PX if self._reserve_active else 0.0
        # The slide, then the zoom about the ARROW -- which is where
        # `--radix-tooltip-content-transform-origin` resolves to, so the shape
        # grows out of the thing it points at. The origin is the tip, and only
        # "left" has it away from the painter's own origin.
        if self._side == "right":
            p.translate(slide * eased, 0.0)
            origin = (0.0, self._arrow_at)
        elif self._side == "left":
            # Enters from the RIGHT, so the offset runs the other way: 8 px out
            # at the start, nothing at rest.
            p.translate(slide * (1.0 - eased), 0.0)
            origin = (box.width() + rise, self._arrow_at)
        elif self._side == "top":
            # Enters from BELOW: 8 px down at the start, nothing at rest.
            p.translate(0.0, slide * (1.0 - eased))
            origin = (self._arrow_at, box.height() + rise)
        else:
            p.translate(0.0, slide * eased)
            origin = (self._arrow_at, 0.0)
        p.translate(*origin)
        p.scale(scale, scale)
        p.translate(-origin[0], -origin[1])

        path = QPainterPath()
        path.setFillRule(Qt.WindingFill)
        path.addRoundedRect(box, t.radius_md, t.radius_md)
        # `rotate-45` on a square, half of it buried in the body -- the same
        # shape the registry draws, not a triangle that only resembles it.
        if self._side == "right":
            tip = QPointF(0.0, self._arrow_at)
            corners = [tip,
                       QPointF(rise, tip.y() - rise),
                       QPointF(2 * rise, tip.y()),
                       QPointF(rise, tip.y() + rise)]
        elif self._side == "left":
            # WOUND THE SAME WAY AS THE BODY, and that is the whole of a bug.
            #
            # `WindingFill` adds overlapping subpaths and SUBTRACTS ones that
            # wind the other way. Listing this diamond's corners as the mirror
            # of the "right" one reverses its direction, so the half buried in
            # the body cancelled the body instead of merging with it -- a white
            # seam the exact width of the overlap, between the pill and its own
            # arrow. Visible only when magnified, which is why it read as "the
            # tooltip looks weird" rather than as a hole.
            edge = box.width()
            tip = QPointF(edge + rise, self._arrow_at)
            corners = [tip,
                       QPointF(edge, tip.y() + rise),
                       QPointF(edge - rise, tip.y()),
                       QPointF(edge, tip.y() - rise)]
        elif self._side == "top":
            # The mirror of "bottom", so its corners run the other way round to wind as
            # the body does (see "left": the opposite winding cuts a seam).
            edge = box.height()
            tip = QPointF(self._arrow_at, edge + rise)
            corners = [tip,
                       QPointF(tip.x() - rise, edge),
                       QPointF(tip.x(), edge - rise),
                       QPointF(tip.x() + rise, edge)]
        else:
            tip = QPointF(self._arrow_at, 0.0)
            corners = [tip,
                       QPointF(tip.x() + rise, rise),
                       QPointF(tip.x(), 2 * rise),
                       QPointF(tip.x() - rise, rise)]
        path.addPolygon(QPolygonF(corners))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(t.foreground))       # INVERTED, per `bg-foreground`
        p.drawPath(path)

        p.setFont(self._font)
        p.setPen(QColor(t.background))         # `text-background`
        if "\n" in self._text:                 # several lines read from the left
            p.drawText(box.adjusted(self.PAD_X, 0, -self.PAD_X, 0),
                       Qt.AlignLeft | Qt.AlignVCenter, self._text)
        else:
            p.drawText(box, Qt.AlignCenter, self._text)
        p.end()


#: Is a mouse button down anywhere in this application?
#:
#: **A tooltip is a HOVER affordance, and mid-click the user is acting rather
#: than inspecting.** Without this, a popup releasing its mouse grab produced
#: exactly one unwanted tooltip: closing a popover sends an `Enter` to whatever
#: is under the pointer, and if that is another rail button the tooltip opened
#: on the press and was taken down again by the release two milliseconds later.
#: MEASURED: `show_for` at 8.3 ms, gone at 10.5 ms -- a flicker, and one the
#: user never asked for, because the pointer never actually arrived anywhere.
#:
#: `QApplication.mouseButtons()` would answer this for real input and NOT for
#: synthetic events, so the state is tracked from the events themselves --
#: which is what makes the behaviour testable rather than merely asserted.
_PRESSED = {"down": False}

#: A small tolerance, in device-independent pixels, on every "is the pointer
#: really over this target" check below (`_Watch`'s Enter guard and
#: `_StrayWatch`'s ongoing one).
#:
#: **Both checks round-trip through GLOBAL coordinates** (`mapFromGlobal` of
#: either `QCursor.pos()` or an event's own `globalPosition()`), which Qt's
#: OWN Enter/Leave delivery never has to do -- that runs on one coordinate
#: space throughout. Under fractional/HiDPI scaling the round trip (local ->
#: global -> local) is not guaranteed to land back on the exact same pixel it
#: started from, so a cursor Qt itself considers "inside" (that is WHY it
#: sent this Enter) can read as one pixel outside once re-derived here -- and
#: for `_StrayWatch`, which re-checks on every subsequent move, a resting
#: cursor with ordinary hardware jitter (a trackpad in particular reports
#: micro-moves while "stationary") can flip across that rounding edge
#: repeatedly. The first flip to "outside" dismisses the tooltip, and
#: nothing brings it back on its own -- Qt sends exactly one Enter per
#: boundary crossing, and the cursor never actually leaves-and-reenters the
#: widget in this scenario, so there is no second Enter to retry with.
#: Reported as a sustained hover showing no tooltip at all, with the
#: target's OWN hover-fill (a separate mechanism, `HoverFill` in
#: `primitives.py`, with no coordinate round trip in its path) lit
#: correctly throughout -- the same tell this file already uses to
#: distinguish a genuine departure from a bookkeeping error.
#:
#: 2 px is generous against rounding drift and negligible against a real
#: departure, which moves the cursor by many pixels on any actual pointing
#: device -- so this does not trade back any of the linger fixes above.
_EDGE_TOLERANCE_PX = 2

#: The tooltip currently on screen. There is only ever one.
#:
#: **A departure's own close motion made this necessary.** A CHILD tooltip's
#: leave fades out over `DURATION_MS` rather than vanishing on the spot --
#: right when the pointer leaves for nothing, and wrong the moment it
#: arrives somewhere else UNRELATED (a sidebar row's own tip, say, while a
#: rail button's was still fading -- two separate `Tooltip` objects that
#: know nothing about each other): the new one opened while the old one was
#: still fading. MEASURED before either had its own dismissal path fixed,
#: crossing between two rail buttons back when each still owned a separate
#: window: the old one stayed on screen **259 ms** after the crossing, with
#: 51 frames showing both at once.
#:
#: So opening one closes any other AT ONCE (`dismiss(instant=True)`, which
#: skips the fade entirely). The fade is for leaving, never for handing
#: over -- and putting that rule here rather than in each caller is what
#: makes it hold between a rail button and a sidebar row, which know nothing
#: about each other.
_OPEN: dict = {"tip": None}

_PRESS_WATCH = None


class _PressWatch(QObject):
    """Tracks whether a mouse button is held, application-wide."""

    def eventFilter(self, obj, event):
        kind = event.type()
        if kind is QEvent.MouseButtonPress:
            _PRESSED["down"] = True
        elif kind is QEvent.MouseButtonRelease:
            _PRESSED["down"] = False
        return False


def pressed() -> bool:
    """Is a mouse button held? The public read of :data:`_PRESSED`.

    Exposed because the tool rail has to make the same judgement from its own
    pointer tracking -- see `toolbar.ToolRail.track_pointer` -- and reaching
    into a module private from another module is how two copies of one rule
    start disagreeing.
    """
    return _PRESSED["down"]


def _watch_presses() -> None:
    """Install the press tracker once, when the first tooltip is attached."""
    global _PRESS_WATCH
    if _PRESS_WATCH is not None:
        return
    app = QApplication.instance()
    if app is not None:
        _PRESS_WATCH = _PressWatch(app)
        app.installEventFilter(_PRESS_WATCH)


_STRAY_WATCH = None


class _StrayWatch(QObject):
    """A shown tooltip whose target the cursor has genuinely left: the
    safety net, not the mechanism.

    Two mechanisms have already been found and fixed here for one symptom
    (a tooltip stuck up with its OWN target's hover-fill correctly unlit):
    a `HoverEnter` re-issued by the target's own `setStyleSheet` churn, and a
    `top_level` tooltip's REAL OS window appearing/disappearing beside the
    cursor causing the platform to redeliver a plain `Enter` with no real
    motion behind it. Reported AGAIN after both fixes. Rather than keep
    chasing individual event-redelivery mechanisms one platform quirk at a
    time -- `_Watch.eventFilter`'s Enter guard already does that for the two
    found so far -- this re-derives the ground truth independently, the same
    way `HoverFill._CursorWatch` already does for the identical class of
    problem one level down (a highlight lit by an untrustworthy event with
    nothing guaranteed to ever correct it): on every move or leave ANYWHERE
    in the app, check whether the CURRENTLY OPEN tooltip's target still
    truly contains the cursor, and dismiss it if not. This does not care
    WHICH event resurrected it or WHY -- only whether the pointer is
    actually there right now.

    `_OPEN["tip"]` names the target it is FOR (`Tooltip._target`), so this
    costs one `rect().contains()` when something is open and nothing at all
    otherwise -- there is normally one open tooltip, often none.
    """

    KINDS = (QEvent.MouseMove, QEvent.HoverMove, QEvent.Leave, QEvent.HoverLeave)

    def eventFilter(self, obj, event):
        if event.type() in self.KINDS:
            open_tip = _OPEN["tip"]
            if open_tip is not None:
                # A DESTROYED TOOLTIP REGISTERED AS OPEN is a closed one. Checked
                # first, not left to whichever Qt call happens next: a Python attribute
                # such as `_target` still reads on a dead wrapper, so without this it
                # stayed registered and this watchdog kept acting on it. (NOT the fix
                # for the intermittent "Tooltip already deleted" printed at exit --
                # that one comes from `Tooltip.event` during teardown; measured 1 in
                # 3 runs at ba1761f too, so it predates this code.)
                try:
                    open_tip.isVisible()
                except RuntimeError:
                    _OPEN["tip"] = None
                    return False
            try:
                target = open_tip._target if open_tip is not None else None
                if target is not None:
                    try:
                        at = target.mapFromGlobal(QCursor.pos())
                        m = _EDGE_TOLERANCE_PX
                        inside = target.rect().adjusted(-m, -m, m, m).contains(at)
                    except RuntimeError:        # the target went away
                        inside = False
                    if not inside:
                        open_tip.dismiss()
            except RuntimeError:
                # The tooltip itself went away between the check above and here.
                _OPEN["tip"] = None
        return False


def _watch_strays() -> None:
    """Install the stray-tooltip watchdog once, when the first tooltip is
    attached -- see :class:`_StrayWatch`."""
    global _STRAY_WATCH
    if _STRAY_WATCH is not None:
        return
    app = QApplication.instance()
    if app is not None:
        _STRAY_WATCH = _StrayWatch(app)
        app.installEventFilter(_STRAY_WATCH)


class _Watch(QObject):
    """Shows a tooltip while the pointer is over one widget."""

    def __init__(self, target: QWidget, tip: Tooltip, text: str,
                 side: str = "bottom", enabled=None):
        super().__init__(target)
        self.target, self.tip, self.text = target, tip, text
        self.side = side
        #: `hidden` on `TooltipContent`, as a predicate.
        #:
        #: The registry does not attach the tooltip conditionally -- it always
        #: renders it and passes `hidden={state !== "collapsed"}`. So the
        #: WIRING is constant and only the showing is conditional, which is why
        #: this is a callable read at hover time rather than a flag read when
        #: the sidebar is built: the sidebar collapses long after that.
        self.enabled = enabled
        # **NOT `WA_Hover`, and this was the bug behind a tooltip staying up
        # after a slow border hover with the target's OWN hover-fill already
        # correctly off.** `HoverFill` (`primitives.py`) already documents why:
        # "Enter/Leave are ordinary events and are always delivered; no
        # `WA_Hover` is needed" -- and it needs none, tracking the identical
        # buttons with plain Enter/Leave alone. `WA_Hover` additionally turns
        # on Qt's HOVER machinery, which is NOT simply "Enter/Leave under a
        # different name": `HoverEnter`/`HoverLeave` are Qt's own recomputed
        # signal, re-issued on internal triggers (a style or geometry change,
        # a repaint) rather than strictly on real cursor motion -- and this
        # widget's OWN `HoverFill` repaints it via `setStyleSheet` on every
        # frame of the very fade that a real Leave starts. A `HoverEnter`
        # landing from that churn, with the cursor genuinely still outside,
        # was cancelling the pending dismiss (`show_for` always stops the
        # grace timer on any Enter-class event) forever -- while `HoverFill`,
        # never watching Hover events, correctly stayed unlit throughout.
        # Enter/Leave alone is what `HoverFill` already proves reliable for
        # these exact widgets; `Hide` stays, for a target torn down mid-hover.
        target.installEventFilter(self)
        _watch_presses()
        _watch_strays()

    def eventFilter(self, obj, event):
        kind = event.type()
        if kind not in (QEvent.Enter, QEvent.MouseButtonPress, QEvent.Leave, QEvent.Hide):
            return False
        try:
            self.tip.isVisible()
        except RuntimeError:
            # The shared tooltip is already destroyed (application exit): nothing to show
            # or dismiss, and touching it would raise through every app-wide filter.
            return False
        if kind is QEvent.Enter:
            if self.enabled is not None and not self.enabled():
                return False
            if _PRESSED["down"]:
                # Mid-click: this Enter is a popup releasing its grab, not the
                # pointer arriving. See `_PRESSED`. Whoever the click turns out
                # to be for decides what to show -- `ToolRail.show_tip` is the
                # one caller that does.
                return False
            # **VERIFIED against where the pointer WAS when this Enter was
            # raised, not against `QCursor.pos()` read NOW.** Dropping
            # `WA_Hover` (see `__init__`) closed one source of a spurious
            # Enter -- a `HoverEnter` re-issued by this widget's OWN
            # `setStyleSheet` churn -- but a `top_level` tooltip is a REAL OS
            # window (see the class docstring), and a window appearing or
            # disappearing right beside the cursor is exactly the kind of
            # window-stack change some platforms answer by re-evaluating --
            # and re-delivering -- Enter/Leave for whatever is now
            # underneath, with no real pointer motion behind it. Reported
            # AGAIN after the `WA_Hover` fix: the same border-hover symptom,
            # same tell (the target's OWN `HoverFill` correctly unlit
            # throughout). `show_for` unconditionally cancels any pending
            # dismiss on an Enter, so ONE such phantom event is enough to
            # strand a tooltip for good -- there is no source left to ever
            # ask again.
            #
            # **A `QCursor.pos()` READ HERE ANSWERS A DIFFERENT QUESTION THAN
            # THE ONE INTENDED, and it cost real Enters.** Reported as a
            # tooltip sometimes not appearing at all despite a sustained
            # hover with the target's OWN hover-fill lit correctly -- the
            # mirror image of the phantom-Enter bug this check exists to
            # catch. `QCursor.pos()` is the LIVE position, read at whatever
            # moment THIS handler happens to run, which is not necessarily
            # the moment the Enter was raised: a busy main thread, or simply
            # a fast pass across adjacent rail icons, can let the pointer
            # travel on to the NEXT icon before this one's own queued Enter
            # is dispatched -- so a perfectly genuine Enter for THIS icon got
            # rejected because the cursor was, by the time anyone asked,
            # already sitting over the one beside it. A `QEnterEvent` (what
            # Qt's own dispatch actually delivers here; a bare `QEvent` only
            # in a synthetic test) carries `globalPosition()` -- WHERE THE
            # POINTER WAS at the instant Qt detected the boundary crossing,
            # which is the ground truth this check was always meant to read.
            # A genuinely phantom Enter is caught exactly as before: it is
            # synthesised at whatever position the cursor happens to be
            # sitting NOW (elsewhere), never inside this target's rect, so
            # this still rejects it -- only a REAL, merely-late-processed
            # Enter is now judged by where it was actually raised rather
            # than by how long it sat in the queue. `_StrayWatch` remains the
            # backstop for anything that slips past this on its own terms
            # (a real Leave arriving before this Enter, say), so relaxing
            # this one check does not reopen the phantom-Enter bug it was
            # built for.
            gpos = (event.globalPosition().toPoint()
                    if isinstance(event, QEnterEvent) else QCursor.pos())
            at = self.target.mapFromGlobal(gpos)
            # `_EDGE_TOLERANCE_PX`: the same global-coordinate round trip
            # `_StrayWatch` needs it for, and for the identical reason -- see
            # that constant's own docstring.
            m = _EDGE_TOLERANCE_PX
            if not self.target.rect().adjusted(-m, -m, m, m).contains(at):
                return False
            # No timer: `delayDuration={0}`.
            self.tip.show_for(self.target, self.text, self.side)
        elif kind is QEvent.MouseButtonPress:
            # Instantly: whatever this click opens is about to be drawn over
            # the tooltip. See `Tooltip.dismiss`.
            self.tip.dismiss(instant=True)
        elif kind is QEvent.Leave:
            # **REPORTED: the tooltip appears and then disappears at once,
            # with the cursor still genuinely resting on the icon (the hand
            # cursor stays up throughout, which is driven by Qt's own,
            # separate hit-testing and is not fooled by this).** This
            # class's own Enter guard, above, already names the mechanism:
            # a `top_level` tooltip is a REAL OS window, and one appearing
            # or repositioning RIGHT BESIDE THE CURSOR is exactly the kind
            # of window-stack change some platforms answer by
            # re-evaluating -- and RE-DELIVERING -- Enter/Leave for
            # whatever is now underneath, with no real pointer motion
            # behind it. That guard was written for the ENTER side only; a
            # `Leave` here was trusted unconditionally, and a spurious one
            # dismisses a tooltip nothing actually left. It went
            # unnoticed for longer than the Enter-side version because it
            # used to be rare (this window changed native state only on a
            # genuine departure or a handoff) -- `_begin_handoff`'s own
            # instant reopen now pops/repositions this SAME window
            # SYNCHRONOUSLY on every ordinary crossing between rail icons,
            # which is exactly the trigger this mechanism needs, far more
            # often than before.
            #
            # Verified the SAME way `_StrayWatch` already verifies its own
            # dismissals -- re-derive from the cursor's ACTUAL current
            # position rather than trusting the event's mere existence.
            # `Hide` (below) gets no such check: the target itself is gone
            # there, so there is nothing left to verify against.
            #
            # **REPORTED AGAIN: stuck open, this time after a genuine
            # departure.** `is_over_target` and `_StrayWatch` -- the two
            # other places in this file asking this exact question --
            # both wrap it in `except RuntimeError` for a target destroyed
            # mid-flight; this one did not. An exception here raises
            # BEFORE `self.tip.dismiss()` on the line below ever runs, so a
            # genuine Leave landing on a target that happened to be torn
            # down in the same moment left the tooltip with nothing left
            # to close it -- the opposite failure from the spurious-Leave
            # bug this check was written for, and worse, since that one at
            # least still closed on the NEXT real move (`_StrayWatch`'s own
            # backstop); a raised exception here never reaches even that.
            try:
                at = self.target.mapFromGlobal(QCursor.pos())
            except RuntimeError:        # the target went away
                self.tip.dismiss()
                return False
            m = _EDGE_TOLERANCE_PX
            if self.target.rect().adjusted(-m, -m, m, m).contains(at):
                return False        # spurious -- the cursor never left
            self.tip.dismiss()
        elif kind is QEvent.Hide:
            self.tip.dismiss()
        return False


#: The one `top_level` window every `top_level=True` attachment shares,
#: created the first time anything asks for one.
#:
#: **ONE NATIVE WINDOW PER RAIL ICON WAS THE FRAGILITY ITSELF, not the
#: animation sitting on top of it.** `_OPEN` already says, in its own words,
#: "there is only ever one" tooltip on screen -- but that was enforced by
#: DISMISSING whichever OTHER native window was open, not by there being only
#: one window to begin with. Scanning three rail icons therefore meant three
#: separate OS windows individually mapped and unmapped in quick succession,
#: each a fresh round trip to the window manager.
#:
#: MEASURED, not theorised: reproduced offscreen by hovering a `top_level`
#: target and then moving straight to a plain sibling widget nearby -- the
#: sibling received NO Enter, Leave or Move at all, on either widget's own
#: filter or the application-wide ones, until the cursor was moved again.
#: One recently-shown-and-hidden top-level window left the platform's own
#: idea of "what is under the cursor" wrong for whatever came right after
#: it, with nothing -- not a Leave, not a stray move -- to correct it. That
#: is a strictly worse version of the exact class of bug `_StrayWatch`
#: exists for, except this time there is no event left to re-derive from at
#: all: the mechanism generating the correction is the same one the bug
#: disrupted. Offscreen is not real hardware, but the file already carries
#: several platform quirks specific to real top-level windows appearing and
#: disappearing beside the cursor (`__init__`, `_shrink_to_settled`,
#: `_Watch`'s own Enter guard) -- this is the same family, one layer up.
#:
#: A SHARED window removes the mechanism rather than working around another
#: one of its symptoms: crossing from one rail icon to the next now
#: repositions and re-texts the SAME native window instead of tearing one
#: down and standing up a different one. Safe precisely because `_OPEN`
#: already treats there as being only one open at a time -- sharing the
#: object makes that literally true instead of merely enforced.
_SHARED_TOP_LEVEL: Tooltip | None = None


def _shared_top_level_tip(t) -> Tooltip:
    global _SHARED_TOP_LEVEL
    if _SHARED_TOP_LEVEL is None:
        # No stable `parent` to give it -- `Tooltip.__init__` detaches a
        # `top_level` window immediately anyway (see its own docstring), and
        # this one outlives any single target.
        _SHARED_TOP_LEVEL = Tooltip(t, None, top_level=True)
    else:
        # Kept current in case a theme switch ever hands `attach` a new `t`
        # -- cheap, and avoids a shared window painting a previous theme's
        # colours after one.
        _SHARED_TOP_LEVEL.t = t
    return _SHARED_TOP_LEVEL


def attach(t, target: QWidget, text: str, host: QWidget | None = None, *,
           side: str = "bottom", enabled=None, top_level: bool = False):
    """Give ``target`` a shadcn tooltip. Returns the watcher.

    ``host`` is what the tooltip is painted on. Left out, it is resolved to the
    target's WINDOW at show time rather than now -- see :meth:`Tooltip.show_for`
    for why "now" is too early. Ignored when ``top_level`` -- see
    :data:`_SHARED_TOP_LEVEL` for why every such attachment shares ONE window
    rather than each getting its own.

    ``side`` is `TooltipContent`'s prop; ``enabled`` is its `hidden`, inverted
    and as a predicate -- see :class:`_Watch`.
    """
    tip = (_shared_top_level_tip(t) if top_level
           else Tooltip(t, host or target, top_level=False))
    return _Watch(target, tip, text, side=side, enabled=enabled)
