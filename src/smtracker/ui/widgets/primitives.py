"""The handful of shadcn components this screen needs, as Qt widgets.

Deliberately NOT a component library. Each class here exists because the
Confirm screen uses it, and stops at what that screen needs -- the point of the
probe is to see the design language in Qt, not to reimplement shadcn.

**The shadow is painted by the PARENT, and that is forced, not chosen.**
Qt clips a widget's painting to its own rectangle, so a card cannot put a soft
edge in the gap beside it. The PARENT has to draw it: :class:`ShadowHost` is
that parent, and because children paint after it, z-order comes free.

**How little there is to paint.** ``shadow-sm`` is
``0 1px 2px rgb(0 0 0 / 0.05)`` -- 3 px of room in total, which is nearly
invisible by design: shadcn separates surfaces with a 1 px border and uses the
shadow only to stop a white card dissolving into a white page. A heavier shadow
on top of that border, of the kind a shadow-first design would ask for, reads as
overdressed here -- both cues are saying the same thing.
"""

from __future__ import annotations

import functools
import math

from qtpy.QtCore import (
    QEasingCurve,
    QEvent,
    QObject,
    QPoint,
    QPointF,
    QRectF,
    QSize,
    Qt,
    QVariantAnimation,
    Signal,
)
from qtpy.QtGui import QColor, QCursor, QFont, QFontMetricsF, QPainter, QPainterPath, QPen
from qtpy.QtWidgets import (
    QAbstractScrollArea,
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QStyle,
    QStyleOption,
    QVBoxLayout,
    QWidget,
)

from ..theme import icons
from ..theme import typography as ty

__all__ = [
    "Badge",
    "Card",
    "Checkbox",
    "ColorFade",
    "FitLabel",
    "FitScroll",
    "MiddleElided",
    "activity_label",
    "Icon",
    "Progress",
    "ShadowHost",
    "Surface",
    "col",
    "day_badge",
    "discard",
    "draw_shadow",
    "fit_label",
    "icon",
    "label",
    "mix",
    "row",
    "scrollbar_css",
    "shadow_room",
    "slide_ms",
    "tw_ease",
]


#: How fast a sliding panel's edge moves here, in pixels per millisecond.
#:
#: **This is the source, and the sidebar's duration derives FROM it** --
#: `Sidebar.COLLAPSE_MS` is `slide_ms(SIDEBAR_W - RAIL_W)`. It reads backwards
#: from how it was written (the sidebar used to own the number and this was
#: `234 / 150`), and the reason for the flip is that the dependency could not
#: go the other way: `primitives` cannot import `workspace`, so a constant
#: derived here from the sidebar's two numbers was really just a COPY of them,
#: and it went stale the moment the duration changed -- 1.56 px/ms against the
#: sidebar's actual 1.17, which is the drift this constant exists to prevent.
#:
#: 234 px in 150 ms. Anything else that slides matches the SPEED, not the
#: duration. See `Sidebar.COLLAPSE_MS` for why it is 150 and not the 250 a
#: smoothness sweep would pick on its own.
#:
#: Matching durations is the mistake this constant exists to stop, and it was
#: made once already: the tables dock was also 150 ms, which sounds like
#: agreement and is not -- it travelled 378 px, so its edge moved 1.6x faster
#: than the sidebar's out of the same window, and read as a snap next to a
#: slide. What the eye matches across two panels is how fast the edge moves.
#:
#: The figures checks pin the sidebar's duration against this, so retuning
#: either one carries everything with it or fails loudly.
PANEL_SPEED = 234 / 150


def slide_ms(distance: float, *, floor: int = 60) -> int:
    """A duration for `distance` px at :data:`PANEL_SPEED`, in whole tens.

    `floor` keeps a very short slide from finishing inside a frame or two,
    where it stops reading as movement and starts reading as a jump -- at
    1.56 px/ms a 40 px drawer would be 26 ms, which on a 60 Hz screen is
    between one and two frames.
    """
    return max(floor, round(abs(distance) / PANEL_SPEED / 10) * 10)


# -- little helpers -------------------------------------------------------- #

def mix(top: str, under: str, alpha: float) -> str:
    """``top`` at ``alpha`` over ``under``, as a hex string.

    shadcn writes half-strength fills as alpha over whatever is behind
    (``hover:bg-muted/50``); a Qt stylesheet fill does no compositing, so the
    blend happens here against the surface the thing actually sits on. Same
    result, because every caller sits on ``--background`` or ``--card``.
    """
    a, b = QColor(top), QColor(under)
    return QColor(
        round(a.red() * alpha + b.red() * (1 - alpha)),
        round(a.green() * alpha + b.green() * (1 - alpha)),
        round(a.blue() * alpha + b.blue() * (1 - alpha)),
    ).name()


#: shadcn's `ScrollBar`: `w-2.5` (10 px) with a `rounded-full bg-border` thumb.
#: The one source both `onboarding.window.OnboardingWindow` and any other
#: scroll area in this UI stack read, so the two can never drift the way they
#: would as two typed-out copies of the same CSS -- which is what put Qt's own
#: wide, platform-chrome scrollbar back once already (see that window's own
#: comment on the incident).
SCROLLBAR_W = 10


def scrollbar_css(t) -> str:
    """The `QScrollBar` rule block, set on a ROOT widget so every scroll area
    beneath it inherits it (Qt stylesheets cascade by type). Qt's native bar
    reads as different-application chrome inside this design; this is what
    replaces it with shadcn's own."""
    return (
        f"QScrollBar:vertical {{ background:transparent; width:{SCROLLBAR_W}px;"
        f" margin:0px; }}"
        f"QScrollBar:horizontal {{ background:transparent; height:{SCROLLBAR_W}px;"
        f" margin:0px; }}"
        f"QScrollBar::handle {{ background:{t.border}; border-radius:4px;"
        f" margin:1px; }}"
        f"QScrollBar::handle:vertical {{ min-height:28px; }}"
        f"QScrollBar::handle:horizontal {{ min-width:28px; }}"
        f"QScrollBar::handle:hover {{ background:{t.muted_foreground}; }}"
        f"QScrollBar::add-line, QScrollBar::sub-line {{ width:0px; height:0px; }}"
        f"QScrollBar::add-page, QScrollBar::sub-page {{ background:transparent; }}"
    )


#: Tailwind's default transition, which is what a bare ``transition-all`` or
#: ``transition-colors`` resolves to: 150 ms on ``cubic-bezier(0.4, 0, 0.2, 1)``.
#: Spelled exactly rather than approximated with a named Qt curve -- Qt can take
#: the bezier directly, so there is no reason to guess at it.
TRANSITION_MS = 150


def tw_ease() -> QEasingCurve:
    """Tailwind's default timing function, as a QEasingCurve."""
    curve = QEasingCurve(QEasingCurve.BezierSpline)
    curve.addCubicBezierSegment(QPointF(0.4, 0.0), QPointF(0.2, 1.0),
                                QPointF(1.0, 1.0))
    return curve


class ColorFade(QVariantAnimation):
    """``transition-colors`` for a Qt stylesheet, which cannot animate one.

    CSS transitions a property; a Qt stylesheet has no such thing, so the colour
    is animated here and the rule is re-applied each frame. That is the whole
    trick, and the cost is worth knowing rather than assuming -- see the README:
    a stylesheet re-parse per frame is cheap for ONE control and is why this is
    a shared object rather than something each widget invents.

    ``apply`` takes the QColor for this frame.
    """

    def __init__(self, owner, apply, ms: int = TRANSITION_MS):
        super().__init__(owner)
        self.setDuration(ms)
        self.setEasingCurve(tw_ease())
        self.valueChanged.connect(apply)

    def to(self, start: str, end: str) -> None:
        """Run from ``start`` to ``end``, cancelling whatever was in flight.

        **``stop()`` comes FIRST, before the already-there check, and that
        ordering is the whole of a bug that shipped.** The check used to return
        early -- "same colour, nothing to do" -- and left a running animation
        alone. It is about the DESTINATION and it forgot the one in flight::

            Enter   now=#ffffff        to(#ffffff -> #fafafa)   fade starts
            Leave   now=#ffffff        to(#ffffff -> #ffffff)   SKIP, no stop
                                        ^ the pointer left before the fade's
                                          first frame, so `now` had not moved
            ...the enter-fade carries on and lands on #fafafa, fully lit,
            with `_in` already False.

        That is how a table ended up with three rows highlighted at once from
        nothing but moving the mouse: any row crossed faster than one animation
        frame (~16 ms) got stranded lit. CSS has the cancelling semantics --
        losing `:hover` re-targets the property and the running transition
        reverses -- so this is a faithfulness fix, not an invention.
        """
        self.stop()
        if QColor(start) == QColor(end):
            return          # already there; the in-flight fade is now cancelled
        self.setStartValue(QColor(start))
        self.setEndValue(QColor(end))
        self.start()


class HoverFill(QObject):
    """``transition-colors`` on a widget's own background.

    ``style`` is a callable taking the fill for this frame and returning the
    whole stylesheet rule, so the caller keeps ownership of everything else in
    it -- a row's bottom border, a card's edge and radius.

    An event filter rather than an ``enterEvent`` override, so this works on any
    widget without subclassing it -- a ``band()``-styled QWidget and a Surface
    alike. Enter/Leave are ordinary events and are always delivered; no
    ``WA_Hover`` is needed, which the stylesheet ``:hover`` route did require.

    **It survives the pointer crossing onto a child**, which is the property the
    whole thing rests on: a row is nested wrappers, and a cell that stole the
    hover would make the band flicker as you moved across it. Qt computes
    enter/leave from the difference of the two ancestor chains, so a common
    ancestor is never left. That was measured for the stylesheet version and
    holds here for the same reason -- and it is pinned by a check, because "same
    reason" is not the same as "checked".
    """

    def __init__(self, widget, style, rest: str, hover: str,
                 ms: int = TRANSITION_MS, over_popups: bool = False):
        super().__init__(widget)
        #: Keep re-deriving while a popup is open. OFF by default, and the
        #: default is the right one for almost everything: a menu standing over
        #: a table must not light the rows beneath it.
        #:
        #: A tool rail is the exception, because its popovers open BESIDE it
        #: rather than over it -- the buttons stay visible and stay clickable,
        #: so a pointer crossing them is a real hover and should look like one.
        #: See `toolbar.ToolRail.track_pointer`.
        self.over_popups = over_popups
        self.widget, self.style = widget, style
        self.rest, self.hover = rest, hover
        self._now = rest
        self._in = False
        self._fade = ColorFade(self, self._on_fade, ms)
        self._area = None
        self._attach()
        widget.installEventFilter(self)
        # A row that is rebuilt while lit would otherwise sit in `_LIT` for the
        # life of the app, and every pointer move would touch a dead C++ object.
        widget.destroyed.connect(lambda *_: _LIT.discard(self))
        widget.setStyleSheet(style(rest))

    def _attach(self) -> bool:
        """Find the scroll area above this widget and hook it. Lazily, and
        that is not an optimisation -- it is the only order that works.

        A row is BUILT before it is added to its layout, so at construction
        ``parentWidget()`` is None and there is no scroll area to find yet.
        The first version resolved it in ``__init__``, found nothing, installed
        nothing, and left the bug exactly as it was -- silently, because a
        missing ancestor is not an error. Re-tried on Show and ParentChange,
        which is when a row actually acquires one.
        """
        if self._area is not None:
            return True
        w = self.widget.parentWidget()
        while w is not None:
            if isinstance(w, QAbstractScrollArea):
                self._area = w
                _hook_scroll(w)
                return True
            w = w.parentWidget()
        return False

    def resync(self) -> None:
        """Re-derive the hover from WHERE THE POINTER IS, not from events.

        Enter/Leave are mouse-movement events, and scrolling moves the widgets
        instead of the mouse. Qt sends nothing: the row that scrolled away keeps
        its highlight (and even keeps reporting ``underMouse() == True``), and
        the row that arrived lights up on the user's next 1 px nudge. MEASURED:
        hover a row, wheel three notches, twitch the mouse -> TWO rows lit.

        Geometry rather than ``underMouse()`` or ``widgetAt``, deliberately.
        ``underMouse()`` is the stale flag that caused this. ``widgetAt``
        respects clipping but answers with whatever is topmost, and during a
        smooth-scroll glide that is the snapshot surface covering the viewport
        -- so every row would read as unhovered mid-glide and the highlight
        would blink off and back on. A rect test against the cursor, clipped to
        the scroll viewport, is stable through the glide.
        """
        w = self.widget
        if not w.isVisible():
            return
        if QApplication.activePopupWidget() is not None and not self.over_popups:
            return                      # a menu is over the table; leave it be
        pos = QCursor.pos()
        inside = w.rect().contains(w.mapFromGlobal(pos))
        if inside and self._area is not None:
            # Clipped: a row scrolled under the header still has global geometry
            # up there, and would otherwise claim a pointer it cannot see.
            vp = self._viewport()
            if vp is not None:
                inside = vp.rect().contains(vp.mapFromGlobal(pos))
        if inside != self._in:
            self._set_in(inside)
            self._fade.to(self._now, self.hover if inside else self.rest)

    def _viewport(self):
        """The viewport of the scroll area this widget is in, or None.

        THE AREA MAY BE GONE (reported 5 Nov 2026 as "RuntimeError: Internal C++
        object (FitScroll) already deleted" from `resync`): the area is found once
        and kept, and a widget moved out of it before it was deleted kept asking a
        dead object -- on every pointer move, from the lit registry. A dead area is
        forgotten and the widget's own ancestors asked again (`_attach`)."""
        try:
            return self._area.viewport()
        except RuntimeError:
            self._area = None
            if self._attach():
                try:
                    return self._area.viewport()
                except RuntimeError:
                    self._area = None
            return None

    def override(self, colour: str | None) -> None:
        """Paint `colour` immediately in place of whatever the real,
        pointer-derived hover state would show -- or, with `colour=None`,
        release the override and repaint whatever `_in` (the REAL flag,
        set only by an actual Enter/Leave and never touched here) says it
        should be.

        For a caller that needs this widget to read "highlighted" for a
        reason that has nothing to do with where the pointer actually is
        (`widgets.table._TableRow.set_highlighted`, lighting a row from a
        chart hover elsewhere on screen). Painting over the fill directly
        with `widget.setStyleSheet(...)` looks like it works and then fails
        silently the first time anything tries to clear it: `_now` never
        moved, and EVERY guard this class has for "is a repaint actually
        needed" compares against `_now` -- `resync`'s own
        `inside != self._in` and `ColorFade.to`'s `start == end` alike.
        Skip updating it and both guards agree nothing changed, so a row
        can be un-highlighted and stay lit on screen regardless of how many
        times `resync()` is called afterward. `override` keeps `_now`
        truthful through the whole cycle, which is what lets a later
        release actually reach the paint call.
        """
        self._fade.stop()
        self._now = colour if colour is not None else (
            self.hover if self._in else self.rest)
        self.widget.setStyleSheet(self.style(self._now))

    def _on_fade(self, colour) -> None:
        self._now = colour.name()
        self.widget.setStyleSheet(self.style(self._now))

    def set_colours(self, rest: str, hover: str) -> None:
        """Re-point at a new pair, e.g. when a row becomes selected.

        Jumps rather than fades: the state changed under the pointer for a
        reason the user just caused, and easing that would lag the click.
        """
        self.rest, self.hover = rest, hover
        self._fade.stop()
        self._now = hover if self._in else rest
        self.widget.setStyleSheet(self.style(self._now))

    def _set_in(self, on: bool) -> None:
        """Set the hover flag and keep the lit REGISTRY in step."""
        self._in = on
        if on:
            _LIT.add(self)
            _watch_cursor()
        else:
            _LIT.discard(self)

    def _clear_others(self) -> None:
        """Re-derive every OTHER lit fill, because Qt will not.

        ``resync`` can turn a fill on for a widget Qt never sent ``Enter`` to --
        that is its whole job, since scrolling moves widgets rather than the
        mouse. The consequence is easy to miss and was shipped: **Qt only sends
        ``Leave`` to a widget it sent ``Enter`` to**, so a row lit by ``resync``
        has nothing that will ever turn it off. Move the pointer to another row
        without scrolling and both stay lit -- reproduced by mixing random moves
        and wheels until two rows were lit, with `_in` True on both.

        So a real ``Enter`` -- which is Qt telling us authoritatively where the
        pointer is -- is used to invalidate everyone else. Scoped to the WINDOW
        rather than to this scroll area: a card lit by a scroll in one lane is
        not a sibling of a row in the table, and it has the same problem.

        Only lit fills are re-derived, so the usual cost is nothing at all.
        """
        window = self.widget.window()
        if window is None:
            return
        for other in window.findChildren(HoverFill):
            if other is not self and other._in:
                other.resync()

    def eventFilter(self, obj, event):
        if obj is self.widget:
            if event.type() is QEvent.ParentChange:
                # MOVED: the scroll area found before may not hold it any more (and may be deleted while it lives
                # on), so it is looked up again from the new ancestors
                self._area = None
                self._attach()
            elif event.type() is QEvent.Show:
                self._attach()
            elif event.type() is QEvent.Enter:
                self._set_in(True)
                self._fade.to(self._now, self.hover)
                # NOT a resync of self: Qt just stated where the pointer is, and
                # that is better evidence than a geometry test (it accounts for
                # clipping, stacking and grabs). Only the others are re-derived.
                self._clear_others()
            elif event.type() is QEvent.Leave:
                self._set_in(False)
                self._fade.to(self._now, self.rest)
        return False


#: Every :class:`HoverFill` that currently believes it is hovered.
#:
#: There is normally one, often none. It exists so the cursor watcher below has
#: something O(1) to check instead of walking the widget tree on every move.
_LIT: set = set()

_WATCH = None


class _CursorWatch(QObject):
    """Re-derives any LIT fill on every pointer move, application-wide.

    The last hole in the hover, and it needed a third mechanism to close:

    * Qt's Enter/Leave handle the ordinary case.
    * ``_hook_scroll`` handles content moving under a still pointer.
    * This handles a fill that ``resync`` lit, which Qt will NEVER send Leave to
      -- because Qt only sends Leave to a widget it sent Enter to. Clearing
      those on another row's Enter is not enough: move the pointer to the panel
      header, or off the table entirely, and no row gets an Enter at all, so the
      highlight just stays. MEASURED as a stranded row after a scroll-then-move.

    **LEAVE is watched as well as the moves, and the moves alone were not
    enough.** This is re-derivation driven by pointer MOTION, so it only works
    while the pointer is still over something this application owns. Take it off
    the window entirely and the moves stop arriving -- there is no further event
    to re-derive on, and the row stays lit for as long as the pointer is
    somewhere else. REPRODUCED deterministically: place the pointer on a row,
    wheel the table under it so the row beneath is lit by ``resync`` with no
    Enter behind it, then move to the desktop -- the row stayed lit.

    A Leave is exactly the event that says "the pointer is no longer where it
    was", and Qt does deliver one to whatever the pointer was over, even though
    the STRANDED row never gets its own. Watched application-wide for the same
    reason the moves are: the widget that reports the departure is not the
    widget that is stuck.

    The cost is a rect test per event per LIT fill, and there is at most one.
    """

    #: Motion says where the pointer went; ``Leave`` says it went somewhere this
    #: application will not hear about.
    KINDS = (QEvent.MouseMove, QEvent.HoverMove, QEvent.Leave,
             QEvent.HoverLeave)

    #: **A periodic SWEEP was tried here and reverted, which is worth keeping.**
    #:
    #: The idea: while anything is lit, ask where the pointer is ten times a
    #: second, so the hover self-corrects even when no event arrives at all. It
    #: works, and it is wrong. Re-deriving on a timer makes the REAL cursor the
    #: only truth, so a fill lit any other way is extinguished within 100 ms --
    #: which is every synthetic ``QEnterEvent``, and four previously stable
    #: checks went red on the first run. A hover that cannot be driven except by
    #: the physical mouse is not more correct, only less testable.
    #:
    #: The events above cover every case the application actually has. What they
    #: do NOT cover is a programmatic ``QCursor.setPos`` that the platform
    #: delivers no motion for -- a bench problem, fixed in the bench.

    def eventFilter(self, obj, event):
        if _LIT and event.type() in self.KINDS:
            for fill in list(_LIT):
                fill.resync()
        return False


def _watch_cursor() -> None:
    """Install the watcher once, the first time anything lights up."""
    global _WATCH
    if _WATCH is not None:
        return
    app = QApplication.instance()
    if app is not None:
        _WATCH = _CursorWatch(app)
        app.installEventFilter(_WATCH)


def _hook_scroll(area) -> None:
    """Make ``area`` re-sync every :class:`HoverFill` under it when it scrolls.

    Installed ONCE per scroll area, by whichever HoverFill is built first.

    **Called STRAIGHT from ``valueChanged``, not coalesced.** The first version
    routed it through a zero-interval single-shot timer, on the reasoning that a
    glide fires ``valueChanged`` every frame and the walk should happen once per
    event-loop turn. That reasoning was never checked against a clock, and both
    halves of it were wrong:

    * The walk is not expensive. MEASURED on the 28-row table --
      ``findChildren`` 0.056 ms, the resync walk 0.174 ms, so **0.23 ms** per
      scroll event and about 2 ms across a whole glide.
    * The timer was expensive. A zero-interval timer fires when the event loop
      next goes idle, and during a glide it does not: the 16 ms frame timer
      keeps it busy. MEASURED, the walk that should have run at 45 ms ran at
      **182 ms** -- 140 ms of pure latency to save 0.2 ms of work.

    That latency was the whole of the "the new row lights up late" report. With
    the walk direct, the state changes on the glide's first frame and the fade
    finishes BEHIND the frozen surface, so the highlight is already correct at
    the moment the surface is handed back.
    """
    if area is None or getattr(area, "_hover_sync", None) is not None:
        return

    def sync():
        for f in area.findChildren(HoverFill):
            f.resync()

    area._hover_sync = sync
    for bar in (area.verticalScrollBar(), area.horizontalScrollBar()):
        bar.valueChanged.connect(sync)


def discard(widget) -> None:
    """Take a widget out for good: HIDE it, detach it, delete it later.

    HIDING FIRST IS THE FIX, and it is one line. Adding a widget to a layout whose
    parent is visible QUEUES that widget's show. If it is detached (`setParent(None)`)
    before the queued show runs -- a rebuild that runs twice in one turn of the event
    loop, which the breadcrumb does on every section switch -- the show lands on a
    widget that is now parentless, and Qt opens it as its OWN WINDOW, which
    `deleteLater` then closes. Reported: "opening stuff makes invisible windows open
    and close". An explicitly hidden widget is never shown by that queued call.
    Reproduced in a test (`test_discard_never_flashes_a_window`).
    """
    if widget is None:
        return
    widget.hide()
    widget.setParent(None)
    widget.deleteLater()


def label(text: str, font: QFont, colour: str) -> QLabel:
    """A non-interactive text run. Transparent so a parent fill shows through."""
    w = QLabel(text)
    w.setFont(font)
    w.setStyleSheet(f"color:{colour}; background:transparent;")
    w.setAttribute(Qt.WA_TransparentForMouseEvents, True)
    return w


class MiddleElided(QLabel):
    """One line shortened in the MIDDLE to the width it is given (a folder path keeps its
    drive and its own name); the whole text is the hover. Its minimum is small, so it never
    widens a page: a path has no spaces to wrap at."""

    def __init__(self, text: str, font: QFont, colour: str):
        super().__init__()
        self._full = str(text)
        self.setFont(font)
        self.setStyleSheet(f"color:{colour}; background:transparent;")
        self.setToolTip(self._full)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.setMinimumWidth(40)
        self.setText(self._full)

    def full_text(self) -> str:
        return self._full

    def set_full_text(self, text: str) -> None:
        self._full = str(text)
        self.setToolTip(self._full)
        self._elide()

    def resizeEvent(self, e) -> None:  # noqa: N802 - Qt's name
        super().resizeEvent(e)
        self._elide()

    def _elide(self) -> None:
        self.setText(self.fontMetrics().elidedText(self._full, Qt.ElideMiddle,
                                                   max(1, self.width())))


class EndElided(MiddleElided):
    """One line of prose shortened at its END ("…") to the width it is given, the whole text the hover. It asks for
    its whole text's width (so it is shown whole wherever there is room) but can shrink to its small minimum: a tip
    beside a card's title was cut under the card's buttons at 1366 px (found by the overnight tester, 4 Nov 2026)."""

    def __init__(self, text: str, font: QFont, colour: str):
        super().__init__(text, font, colour)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt's name
        width = int(QFontMetricsF(self.font()).horizontalAdvance(self._full)) + 2
        return QSize(width, super().sizeHint().height())

    def set_full_text(self, text: str) -> None:
        super().set_full_text(text)
        self.updateGeometry()                     # its hint is the whole text's width

    def _elide(self) -> None:
        self.setText(self.fontMetrics().elidedText(self._full, Qt.ElideRight, max(1, self.width())))


class FitLabel(QLabel):
    """A word-wrapped label that pins its OWN height to its text, so no layout has to ask.

    **A plain wrapped `QLabel` makes every layout above it height-for-width.**
    `setWordWrap(True)` sets `heightForWidth` on the label's size policy, every enclosing
    layout then reports `hasHeightForWidth()`, and on each width change Qt re-solves
    height-for-width down the whole chain -- a `QStackedWidget` asks EVERY page, hidden
    ones included. MEASURED while the sidebar slides (one width change per frame): three
    wrapped captions on the QC detail page cost **30.2 ms** of layout per frame against
    4.3 ms without them, and a section placeholder's one description 8.7 against 1.2.

    So the policy flag is switched OFF -- after every text change, since `QLabel` turns it
    back on each time its text is set -- and the label sets a FIXED height from `heightForWidth(width)` whenever its width, text or
    font changes. Fixed, not a minimum: with the flag off a layout falls back to
    `sizeHint`, which over-sizes a wrapped label (measured 30 -> 45 px), so a minimum
    alone left a gap under the text. The height moves only when the line count does.
    """

    def __init__(self, text: str = "", parent: QWidget | None = None, *, min_height: int = 0):
        super().__init__(text, parent)
        self._floor = min_height
        self.setWordWrap(True)
        self._fit()

    def _fit(self) -> None:
        # `QLabel` turns the flag back ON whenever its text changes (`updateLabel`), so it
        # is cleared here, after every change, not once at construction.
        policy = self.sizePolicy()
        if policy.hasHeightForWidth():
            policy.setHeightForWidth(False)
            self.setSizePolicy(policy)
        # Clear the old bounds before measuring: `heightForWidth` never reports less than
        # the minimum, so without this a label could only ever grow and a shorter text
        # would keep the height of the longest one shown before.
        self.setMinimumHeight(0)
        self.setMaximumHeight(16777215)
        h = max(self._floor, self.heightForWidth(max(self.width(), 1)))
        if h > 0:
            self.setFixedHeight(h)

    def setText(self, text: str) -> None:
        super().setText(text)
        self._fit()

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._fit()

    def changeEvent(self, e) -> None:
        super().changeEvent(e)
        if e.type() == QEvent.FontChange:
            self._fit()


def fit_label(text: str, font: QFont, colour: str, *, min_height: int = 0) -> FitLabel:
    """`label`, wrapped, as a :class:`FitLabel` (never shorter than `min_height`)."""
    w = FitLabel(text, min_height=min_height)
    w.setFont(font)
    w.setStyleSheet(f"color:{colour}; background:transparent;")
    w.setAttribute(Qt.WA_TransparentForMouseEvents, True)
    return w


class _ActivityLabel(FitLabel):
    """`activity_label`'s label: its hover is always its whole text."""

    def setText(self, text: str) -> None:
        super().setText(text)
        self.setToolTip(text)


def activity_label(font: QFont, colour: str) -> FitLabel:
    """A window header's ACTIVITY line ("Options saved: ...", "u-track is starting ..."):
    wrapped onto further lines in whatever room the header leaves between its breadcrumb and
    its buttons, right-aligned, the whole text on hover. Add it with stretch 1 IN PLACE of
    the header's stretch. Found 3 Nov 2026 (tester B6): as a plain one-line label a long
    message asked for its whole width as the header's minimum and pushed the preparation and
    outlines windows to 1633 px on a 1366 px screen, Options and Back to menu off it."""
    w = _ActivityLabel("")
    w.setFont(font)
    w.setStyleSheet(f"color:{colour}; background:transparent;")
    w.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    w.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
    w.setMinimumWidth(0)
    return w


class ShrinkLabel(QLabel):
    """A one-line TITLE that fits the width it is given (asked 30 Oct 2026: "when something gets cut -> scale down
    text size"): first its SHORT form where it has one ("2 populations" -> "2-pop", asked the same evening), then set
    smaller in 1 px steps down to `chart_text.TITLE_FLOOR_PX`, then elided (a " · " title's last part kept whole:
    "…Green, measured · 2-pop"), the whole text on hover -- the rule is `chart_text.fit_title`, this only measures
    with Qt's metrics. ``elided`` is True whenever the title is not shown whole.

    A plain `QLabel` asks for its whole text as its MINIMUM width, so a long title ("mNeonGreen, measured · 2
    populations" on a Jump distances tile at Size S) widened its card past its column or was cut at the card's edge
    under the card's floating buttons. This one prefers its whole text but can shrink to a few characters, keeps the
    full size's HEIGHT whatever it is set at (cards in a row keep their baselines), and leaves ``reserve`` pixels free
    at its right (`set_reserve`: what floats over it, a card's Axes and Export buttons)."""

    def __init__(self, text: str, font: QFont, colour: str, *, t=None,
                 floor: float | None = None, parent: QWidget | None = None):
        from ... import chart_text

        super().__init__(parent)
        self._full = str(text)
        self._base = QFont(font)
        self._floor = float(chart_text.TITLE_FLOOR_PX if floor is None else floor)
        self._reserve = 0
        self._fitting = False
        #: True while the title is not shown whole -- short or elided (its hover then shows it whole).
        self.elided = False
        self.setFont(self._base)
        self.setStyleSheet(f"color:{colour}; background:transparent;")
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self._tip = None
        if t is not None:
            from . import tooltip

            self._tip = tooltip.attach(t, self, self._full, top_level=True, enabled=lambda: self.elided)
        self._fit()

    def full_text(self) -> str:
        return self._full

    def text(self) -> str:
        """The whole title (what is drawn may be elided: `shown_text`)."""
        return self._full

    def shown_text(self) -> str:
        """The title as drawn: whole, or elided in the middle."""
        return QLabel.text(self)

    def text_px(self) -> int:
        """The pixel size the title is drawn at now."""
        return self.font().pixelSize()

    def setText(self, text: str) -> None:
        self._full = str(text)
        if self._tip is not None:
            self._tip.text = self._full
        self.updateGeometry()
        self._fit()

    def set_reserve(self, px: int) -> None:
        px = max(0, int(px))
        if px != self._reserve:
            self._reserve = px
            self._fit()

    def _at(self, px: float) -> QFont:
        font = QFont(self._base)
        font.setPixelSize(max(1, int(round(px))))
        return font

    def _base_px(self) -> int:
        px = self._base.pixelSize()
        return px if px > 0 else max(1, round(self._base.pointSizeF() * 96.0 / 72.0))

    def sizeHint(self) -> QSize:
        m = self.contentsMargins()
        metrics = QFontMetricsF(self._at(self._base_px()))
        return QSize(math.ceil(metrics.horizontalAdvance(self._full)) + m.left() + m.right() + self._reserve,
                     math.ceil(metrics.height()) + m.top() + m.bottom())

    def minimumSizeHint(self) -> QSize:
        m = self.contentsMargins()
        least = QFontMetricsF(self._at(self._floor)).horizontalAdvance("Mm…")
        return QSize(math.ceil(least) + m.left() + m.right(), self.sizeHint().height())

    def _fit(self) -> None:
        from ... import chart_text

        if self._fitting:
            return
        self._fitting = True
        try:
            width = max(0.0, float(self.contentsRect().width() - self._reserve))

            def measure(s, px):
                return QFontMetricsF(self._at(px)).horizontalAdvance(s)

            px, shown, elided = chart_text.fit_title(self._full, measure, width, self._base_px(), self._floor, 1.0)
            if self.font().pixelSize() != int(round(px)):
                self.setFont(self._at(px))
            if QLabel.text(self) != shown:
                QLabel.setText(self, shown)
            self.elided = elided
        finally:
            self._fitting = False

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._fit()


class Icon(QWidget):
    """One lucide glyph, painted at draw time.

    A widget with a ``paintEvent`` rather than a ``QLabel`` holding a pixmap.
    Nothing is rasterised until Qt asks for it, at the size and device ratio it
    is actually being drawn at -- so there is no ratio to assume, nothing to
    stamp onto a pixmap, and nothing to resample. Every icon fault in this
    prototype came from the pixmap route; :mod:`shadcn.icons` has the history.
    """

    def __init__(self, name: str, px: int, colour: str, parent=None, *,
                 stroke: float | None = None):
        super().__init__(parent)
        self.name = name
        self.colour = colour
        #: Per-icon override of :data:`icons.STROKE`. The module default was
        #: calibrated by eye for glyphs sitting beside body text; a nav rail's
        #: icons are the row's only mark and read as grey hairlines at it.
        self.stroke = stroke
        self.setFixedSize(px, px)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)

    def paintEvent(self, _e):
        p = QPainter(self)
        icons.paint(p, self.name, QRectF(self.rect()), self.colour,
                    stroke=self.stroke)
        p.end()


def icon(name: str, px: int, colour: str, parent=None, *,
         stroke: float | None = None) -> Icon:
    """A lucide glyph as a fixed-size widget, so it holds a lane in a row."""
    return Icon(name, px, colour, parent, stroke=stroke)


def breathe() -> None:
    """Let the event loop paint MID-BUILD, without letting a click in.

    Qt widgets can only be constructed on the GUI thread, so a screen that takes
    a second to build blocks it for a second however the work is arranged -- the
    spinner stops, and a stopped spinner is what a user reads as a crash. This
    is the one lever left: run the loop's timers and paints at a safe point, so
    the animation keeps moving while the build continues.

    ``ExcludeUserInputEvents`` IS THE WHOLE SAFETY. Without it a click could
    re-enter the very code doing the building -- cancel the load whose result is
    half-built, or switch a tab that does not exist yet. Input is queued, not
    dropped: it is delivered when the build finishes and the loop runs properly
    again.
    """
    from qtpy.QtCore import QEventLoop
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance()
    if app is not None:
        app.processEvents(QEventLoop.ExcludeUserInputEvents)


#: One turn per second, linear. shadcn's `Spinner` is `Loader2Icon` with
#: `animate-spin`, and Tailwind's `animate-spin` is exactly
#: ``spin 1s linear infinite``. NOT eased: an eased rotation reads as a thing
#: struggling rather than a thing working.
SPIN_MS = 1000


class Spinner(Icon):
    """shadcn's ``Spinner``: ``Loader2Icon`` at ``size-4``, ``animate-spin``.

    THE PAINTER IS ROTATED, not a pixmap. `Icon` renders SVG into the live
    painter at draw time, so turning the painter turns real geometry -- a
    rotated pixmap shimmers, because it is resampling a raster every frame at
    an angle it was never drawn for.

    The animation runs only while the widget is visible. A spinner left turning
    behind a screen nobody is looking at is a repaint per frame, forever, and
    it is exactly the kind of thing that never shows up as a bug.
    """

    def __init__(self, colour: str, px: int = 16, parent=None, *,
                 stroke: float | None = None):
        super().__init__("loader-circle", px, colour, parent, stroke=stroke)
        self._angle = 0.0
        self._spin = QVariantAnimation(self)
        self._spin.setStartValue(0.0)
        self._spin.setEndValue(360.0)
        self._spin.setDuration(SPIN_MS)
        self._spin.setLoopCount(-1)
        self._spin.setEasingCurve(QEasingCurve.Linear)
        self._spin.valueChanged.connect(self._turn)

    def _turn(self, value) -> None:
        self._angle = float(value)
        self.update()

    def showEvent(self, e):
        super().showEvent(e)
        self._spin.start()

    def hideEvent(self, e):
        self._spin.stop()
        super().hideEvent(e)

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        box = QRectF(self.rect())
        centre = box.center()
        p.translate(centre)
        p.rotate(self._angle)
        p.translate(-centre)
        icons.paint(p, self.name, box, self.colour, stroke=self.stroke)
        p.end()


def band(widget: QWidget, name: str, css: str) -> QWidget:
    """Style one widget WITHOUT the rule reaching its children.

    An unscoped ``setStyleSheet("border-bottom: 1px solid ...")`` cascades: Qt
    applies it to the widget AND to every descendant, so a table row's rule
    draws a little underline beneath each of its own cells. Scoping by object
    name confines it, and it is easy to get wrong because the unscoped version
    looks right until the widget has children.
    """
    widget.setObjectName(name)
    widget.setStyleSheet(f"QWidget#{name} {{ {css} }}")
    return widget


def row(*items, spacing: int = 8, margins=(0, 0, 0, 0)) -> QWidget:
    """A horizontal strip. ``None`` becomes a stretch, an int becomes a spacer."""
    w = QWidget()
    w.setStyleSheet("background:transparent;")
    lay = QHBoxLayout(w)
    lay.setContentsMargins(*margins)
    lay.setSpacing(spacing)
    for item in items:
        if item is None:
            lay.addStretch(1)
        elif isinstance(item, int):
            lay.addSpacing(item)
        else:
            lay.addWidget(item)
    return w


def col(*items, spacing: int = 6, margins=(0, 0, 0, 0)) -> QWidget:
    """A vertical stack, same conventions as :func:`row`."""
    w = QWidget()
    w.setStyleSheet("background:transparent;")
    lay = QVBoxLayout(w)
    lay.setContentsMargins(*margins)
    lay.setSpacing(spacing)
    for item in items:
        if item is None:
            lay.addStretch(1)
        elif isinstance(item, int):
            lay.addSpacing(item)
        else:
            lay.addWidget(item)
    return w


# -- shadow ---------------------------------------------------------------- #

def shadow_room(t, spec=None) -> int:
    """Pixels a shadow needs OUTSIDE its card. One source of truth for spacing.

    ``spec`` picks the rung of the ladder. It defaults to ``shadow_sm`` because
    that is what every caller here wanted until a toast asked for
    ``shadow_lg`` -- which needs 22 px of room against shadow-sm's 4, so a
    widget sized from the wrong rung crops its own shadow at the corners.
    """
    return max(1, max(math.ceil(blur + spread + max(abs(dx), abs(dy)))
                      for dx, dy, blur, spread, _a in (spec or t.shadow_sm)))


@functools.lru_cache(maxsize=64)
def _ring_alphas(blur: float, alpha: float) -> tuple[float, ...]:
    """Per-ring alpha for a Gaussian edge, outermost ring first.

    A CSS blur radius means "a Gaussian with sigma = blur/2", so a blurred
    straight edge has coverage ``alpha * 0.5 * erfc(d / (sigma * sqrt 2))`` at
    distance ``d`` outside it -- half the alpha AT the edge, trailing off over
    about ``blur``. The previous painter spread the alpha evenly over the rings
    instead, which is a LINEAR ramp: too solid at the edge, too abrupt where it
    ends, and the reason the result read as blunt.

    Rings are filled shapes, so a point at distance ``d`` is painted by every
    ring from ``d`` outward. Under source-over that composites to
    ``1 - prod(1 - a_i)``, and inverting it gives each ring exactly the alpha
    that lands the cumulative coverage on the target curve::

        a_d = 1 - (1 - A(d)) / (1 - A(d + 1))

    Not the additive approximation: at these alphas the difference is small,
    but the exact form costs one division and cannot drift as layers stack.
    """
    if blur <= 0:
        return (alpha,)
    sigma = blur / 2.0
    extent = max(1, math.ceil(blur))

    def coverage(d: float) -> float:
        return alpha * 0.5 * math.erfc(d / (sigma * math.sqrt(2.0)))

    out = []
    for d in range(extent, -1, -1):
        outer = coverage(d + 1) if d < extent else 0.0
        inner = coverage(d)
        out.append(0.0 if outer >= 1.0 else 1.0 - (1.0 - inner) / (1.0 - outer))
    return tuple(out)


def draw_shadow(p: QPainter, rect: QRectF, radius: float, t,
                spec=None) -> None:
    """A Tailwind shadow: every layer, each as concentric rounded rects.

    **The interior is clipped away, and that reverses an earlier measurement.**
    The naive version drew every ring whole and was kept because a
    clipped-to-a-ring version had MEASURED SLOWER on an earlier screen
    (8.3 ms vs 7.5 ms) -- true for a small card, and with a ``QRegion`` doing
    the clipping. The workspace panel's inset page is 1247x828, where the
    interior fills dominate completely, and a ``QPainterPath`` clip is cheaper
    to set up than the region was::

        inset page  1247x828   plain 9.30 ms   ring-clipped 2.14 ms   4.3x
        series card  348x79    plain 0.85 ms   ring-clipped 0.60 ms   1.4x
        small card   222x63    plain 0.69 ms   ring-clipped 0.56 ms   1.2x

    Faster at every size, so this is not a trade. It was worth 8 ms a frame on
    the sidebar's collapse, which is half that animation's budget.

    Safe because the interior is never seen: the shadow is painted by an
    ANCESTOR, before the surface itself draws over that exact footprint. It
    would matter for a shadowed surface with a transparent middle, which
    nothing here is.
    """
    hole = QPainterPath()
    hole.addRoundedRect(rect, radius, radius)
    outer = QPainterPath()
    outer.addRect(QRectF(0.0, 0.0, float(p.device().width()),
                         float(p.device().height())))
    p.save()
    # IntersectClip, so whatever the paint event already limited us to survives.
    p.setClipPath(outer.subtracted(hole), Qt.IntersectClip)
    p.setPen(Qt.NoPen)
    for dx, dy, blur, spread, alpha in (spec or t.shadow_sm):
        alphas = _ring_alphas(blur, alpha)
        extent = len(alphas) - 1
        for step, a in enumerate(alphas):
            if a <= 0.0:
                continue
            # Outermost first. `grow` is how far this ring sits outside the
            # SPREAD shape, which for a second layer is smaller than the card.
            grow = spread + (extent - step)
            ink = QColor(0, 0, 0)
            ink.setAlphaF(min(1.0, a))
            p.setBrush(ink)
            box = rect.adjusted(-grow + dx, -grow + dy, grow + dx, grow + dy)
            if box.width() <= 0 or box.height() <= 0:
                continue
            r = max(0.0, radius + grow)
            p.drawRoundedRect(box, r, r)
    p.restore()


def own_surfaces(root) -> list:
    """Surfaces whose nearest :class:`Surface` ancestor is ``root`` itself.

    A shadow is painted by an ANCESTOR, before that ancestor's children draw.
    So a Button nested inside a Card can never be shadowed by the page: the
    Card paints over it. Measured -- the outline Button's ``shadow-xs`` read
    255 under the button in every state, including with the shadow forced back
    to ``shadow-sm``, because it was never reaching the screen at all.

    Each surface therefore draws the shadows of the surfaces DIRECTLY inside
    it, and this split keeps that from being drawn twice.

    **A widget carrying `_shadow_boundary = True` also stops the walk**, the
    same as an actual :class:`Surface` would -- a duck-typed flag rather than
    a second `isinstance` target, since the one caller of this (`Segmented`,
    in `widgets.controls`) cannot be imported here without a cycle (`controls`
    already imports FROM this module). `Segmented` sets it because it is a
    documented exception to "an ancestor draws the shadow": its own
    `paintEvent` already draws its raised pill's shadow directly (see its own
    docstring -- every ancestor-drawn case for that pill was historically
    live and broken), so an outer `Surface` finding the pill via
    `findChildren` and trying to draw ANOTHER copy is not a second correct
    shadow, it is a stale one. MEASURED: `select()` destroys and recreates
    every pill on a change, `ChildAdded`/`ChildRemoved` is delivered only to
    `Segmented` itself (Qt does not bubble child-events to grandparents), so
    an outer `Surface`'s cached list keeps referencing the ORPHANED pill --
    `mapTo` on it does not even raise, it logs "parent must be in parent
    hierarchy" and returns a wrong position silently, which is a shadow
    flashing in the wrong place for a frame every time a picker's selection
    changes. Reproduced and confirmed before this fix, not assumed.
    """
    out = []
    for s in root.findChildren(Surface):
        w = s.parentWidget()
        nearest = None
        while w is not None and w is not root:
            if isinstance(w, Surface) or getattr(w, "_shadow_boundary", False):
                nearest = w
                break
            w = w.parentWidget()
        if nearest is None:
            out.append(s)
    return out


def _scroll_clip(card, root) -> QRectF | None:
    """The part of every scroll VIEWPORT between ``card`` and ``root`` that is on
    screen, in ``root``'s coordinates, or ``None`` when the card is in no scroll area.

    An ancestor draws the shadow, so the viewport's clipping -- which hides the card
    itself -- never reached its shadow: a row scrolled out of the Compute dialog's
    condition list left its checkbox's shadow as a faint square below the list.
    Only viewports clip, deliberately: a card's shadow is meant to spill past its own
    parent, just not out of a scrolled region.
    """
    clip = None
    w = card.parentWidget()
    while w is not None and w is not root:
        parent = w.parentWidget()
        if isinstance(parent, QAbstractScrollArea) and parent.viewport() is w:
            at = w.mapTo(root, QPoint(0, 0))
            area = QRectF(at.x(), at.y(), w.width(), w.height())
            clip = area if clip is None else clip.intersected(area)
        w = parent
    return clip


def paint_shadows(root, p: QPainter, t, exposed: QRectF, cards) -> bool:
    """Draw each card's shadow onto ``root``. False if the cache went stale."""
    room = shadow_room(t)
    for card in list(cards):
        try:
            if not card.isVisible() or not card.shadowed:
                continue
            at = card.mapTo(root, QPoint(0, 0))
            box = QRectF(at.x(), at.y(), card.width(), card.height())
            clip = _scroll_clip(card, root)
        except RuntimeError:
            return False
        if not box.adjusted(-room, -room, room, room).intersects(exposed):
            continue
        if clip is None:
            draw_shadow(p, box, card.radius, t, card.shadow)
            continue
        if not clip.intersects(box):
            continue
        p.save()
        p.setClipRect(clip, Qt.IntersectClip)
        draw_shadow(p, box, card.radius, t, card.shadow)
        p.restore()
    return True


class ShadowHost(QWidget):
    """A container that paints ``shadow-sm`` under every :class:`Card` inside it.

    The child list is cached and dropped on any structural change, because
    ``findChildren`` on every repaint is the kind of cost that only shows up
    once the panel is full. A destroyed child raises ``RuntimeError`` through
    its Python wrapper rather than returning something falsy, so that is caught
    explicitly and the cache rebuilt.
    """

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self._cards: list[Card] | None = None

    def event(self, e):
        # `LayoutRequest` is NOT in this list, and that is the point. It says
        # geometry moved, not that the SET of surfaces changed -- and
        # `paint_shadows` re-reads every position with `mapTo` on each paint,
        # so a cached list survives a relayout intact. Invalidating on it meant
        # a full `findChildren(Surface)` walk over 418 widgets on EVERY FRAME of
        # the sidebar's collapse, which is a cache that rebuilds itself exactly
        # when it is most needed.
        if e.type() in (QEvent.ChildAdded, QEvent.ChildRemoved):
            self._cards = None
        return super().event(e)

    def paintEvent(self, e):
        opt = QStyleOption()
        opt.initFrom(self)
        p = QPainter(self)
        self.style().drawPrimitive(QStyle.PE_Widget, opt, p, self)
        p.setRenderHint(QPainter.Antialiasing, True)
        if self._cards is None:
            self._cards = own_surfaces(self)
        if not paint_shadows(self, p, self.t, QRectF(e.rect()), self._cards):
            self._cards = None
        p.end()


# -- surfaces -------------------------------------------------------------- #

class Surface(QFrame):
    """Anything :class:`ShadowHost` should draw ``shadow-sm`` beneath.

    A base rather than a duck-typed attribute check, so ``findChildren`` can
    ask Qt for them by type instead of walking every widget in the tree and
    testing for a marker. Cards and buttons both qualify -- in shadcn both
    carry ``shadow-sm``, and having one rule serve both is what stops the two
    drifting apart the first time the shadow is re-tuned.
    """

    def __init__(self, t, *, radius: float, shadowed: bool, shadow=None,
                 parent=None):
        super().__init__(parent)
        self.t = t
        self.radius = radius
        self.shadowed = shadowed
        #: Which step of the ladder to draw. ``None`` means ``shadow-sm``, the
        #: Card's. A Button's ``outline`` variant overrides it with
        #: ``shadow-xs``, which is what the registry gives it -- one step
        #: lighter and a single layer.
        self.shadow = shadow
        self._kids: list | None = None

    def event(self, e):
        # See `ShadowHost.event`: the child LIST does not change on a relayout.
        if e.type() in (QEvent.ChildAdded, QEvent.ChildRemoved):
            self._kids = None
        return super().event(e)

    def paintEvent(self, e):
        """Paint this surface, then the shadows of the surfaces inside it.

        In that order and no other: a shadow belongs UNDER the thing it
        describes and OVER whatever this surface is filled with, and Qt draws
        children after their parent, so both come out right.
        """
        super().paintEvent(e)
        if self._kids is None:
            self._kids = own_surfaces(self)
        if not self._kids:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        if not paint_shadows(self, p, self.t, QRectF(e.rect()), self._kids):
            self._kids = None
        p.end()


class Card(Surface):
    """``rounded-lg border bg-card shadow-sm``.

    ``shadowed`` exists because shadcn nests cards without nesting elevation --
    a group inside the groups panel gets the border and not the shadow. Two
    stacked shadows read as a lifted thing on a lifted thing, which is a claim
    about hierarchy this screen does not want to make.
    """

    def __init__(self, t, *, radius: float | None = None,
                 shadowed: bool = True, fill: str | None = None, parent=None):
        # ``rounded-xl`` -- the registry's own class for Card, 14 px, not the
        # 10 px ``rounded-lg`` a card is easy to assume.
        super().__init__(t, radius=t.radius_xl if radius is None else radius,
                         shadowed=shadowed, parent=parent)
        self.setObjectName("card")
        self.setStyleSheet(
            f"QFrame#card {{ background:{fill or t.card};"
            f" border:1px solid {t.border};"
            f" border-radius:{self.radius:g}px; }}")


class Badge(QFrame):
    """``rounded-md px-2 text-xs font-semibold`` on an explicit fill.

    shadcn's Badge takes a variant; this one takes colours, because the day
    tags sit on a computed ramp that no variant covers.
    """

    def __init__(self, t, text: str, fill: str, ink: str, *,
                 step: str = "xs", weight: str = "semibold", pad: int = 7,
                 parent=None):
        super().__init__(parent)
        self.t = t
        self.setObjectName("badge")
        self.setStyleSheet(
            f"QFrame#badge {{ background:{fill}; border:none;"
            f" border-radius:{t.radius_md:g}px; }}")
        self._label = label(text, ty.ui(step, weight), ink)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(pad, 0, pad, 0)
        lay.setSpacing(0)
        lay.addWidget(self._label)
        self.setFixedHeight(20)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        #: Set by :func:`day_badge` on badges that sit on the DAY ramp, so a
        #: recolour can find exactly those and leave every other badge alone.
        self.day_index: int | None = None

    def set_colours(self, fill: str, ink: str) -> None:
        """Repaint in place. Two stylesheet writes, no rebuild.

        This is the whole reason :class:`Badge` keeps a handle on its own label:
        recolouring by REBUILDING the panels that contain these badges cost 948
        ms for one colour and 2 728 ms for a palette switch, measured, because
        it destroyed and recreated 83 series cards to change a fill.
        """
        self.setStyleSheet(
            f"QFrame#badge {{ background:{fill}; border:none;"
            f" border-radius:{self.t.radius_md:g}px; }}")
        self._label.setStyleSheet(f"color:{ink}; background:transparent;")

    def set_text(self, text: str) -> None:
        """Repaint the label in place — the text counterpart to
        `set_colours`, added for the same reason: a live count badge (a
        section header's own condition/day/series count) changes far more
        often than it needs rebuilding for."""
        self._label.setText(text)


class Progress(QWidget):
    """shadcn's Progress: ``h-2 w-full rounded-full bg-primary/20``.

    ``h-2`` is 8 px and it is not negotiable per-site -- a 6 px variant was
    tried inline and is what made these stop reading as shadcn.

    Painted rather than assembled from two styled frames: a stylesheet
    border-radius on a child of a rounded parent leaves a visible corner notch
    at these sizes, and the fill has to clip to the track's own rounding.
    """

    def __init__(self, t, value: float, *, height: int = 8,
                 width: int | None = None, parent=None):
        super().__init__(parent)
        self.t = t
        self.value = max(0.0, min(1.0, value))
        self.setFixedHeight(height)
        if width is not None:
            self.setFixedWidth(width)
        else:
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_value(self, value: float) -> None:
        """Move the indicator. Repaints only when the pixel would change.

        A loader ticking once per movie asks for a repaint thirty times; on a
        72 px bar most of those land on the same pixel, and a repaint that
        cannot be seen is a repaint that should not happen.
        """
        value = max(0.0, min(1.0, float(value)))
        if round(value * self.width()) == round(self.value * self.width()):
            self.value = value
            return
        self.value = value
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        h = self.height()
        p.setPen(Qt.NoPen)
        # THE TRACK IS ``bg-primary/20``, not a grey.
        #
        # Fetched from the registry rather than remembered:
        #   h-2 w-full overflow-hidden rounded-full bg-primary/20
        #   indicator: bg-primary
        #
        # It was hard-coded to zinc (#E4E4E7 / #3F3F46), which is close in the
        # light scheme and wrong in the dark one -- shadcn's dark track is the
        # near-white primary at 20 % over a dark card, a warm pale grey, where
        # a zinc-700 reads as a blue-tinted hole. Deriving it from ``primary``
        # also means a re-themed accent carries into the track for free, which
        # is the entire point of expressing it as an alpha of a token.
        track = QColor(self.t.primary)
        track.setAlphaF(0.2)
        p.setBrush(track)
        p.drawRoundedRect(QRectF(0, 0, self.width(), h), h / 2, h / 2)
        if self.value > 0:
            p.setBrush(QColor(self.t.primary))
            p.drawRoundedRect(QRectF(0, 0, self.width() * self.value, h),
                              h / 2, h / 2)
        p.end()


def _device_box(w) -> tuple[QRectF, float]:
    """The 16 px box snapped to whole DEVICE pixels, and the pen to match.

    **This is what made a table's checkboxes look smudged on the right.** Qt
    lays widgets out in LOGICAL pixels, but a screen at e.g. 1.25x puts a
    widget at logical x = 54 at device x = 67.5 -- half a device pixel off
    the grid. A 1 px border straddling that phase is rendered across TWO
    columns at half coverage each, which is exactly the blur one sees.

    It shows in a table column and not in a scattered layout because every
    row's tick sits at the SAME x, so all of them share the worst possible
    phase (0.50) and the softness repeats down the column.

    So the box is snapped INWARD to whole device pixels (outward would push
    it past the widget, which clips), and the pen is rounded to a whole
    number of device pixels -- at 1.25x a nominal 1 px border is 1.25 device
    px, which cannot be drawn crisply at any position.

    Returns the PATH rect (inset by half the pen, as Qt strokes centred) and
    the pen width, both in logical units.
    """
    dpr = w.devicePixelRatioF()
    # Relative to the WINDOW, not the screen. Both put the box on the device
    # grid, but a global basis makes the widget repaint differently whenever
    # the window moves -- and a frozen copy of the page grabbed at the old
    # position would then no longer match what is drawn under it.
    at = w.mapTo(w.window(), QPoint(0, 0))
    ox, oy = at.x() * dpr, at.y() * dpr
    pen = max(1.0, float(round(dpr))) / dpr
    out = []
    for origin, extent in ((ox, w.width()), (oy, w.height())):
        lo = math.ceil(origin - 1e-6)
        hi = math.floor(origin + extent * dpr + 1e-6)
        out.append(((lo - origin) / dpr, (hi - lo) / dpr))
    (x, bw), (y, bh) = out
    half = pen / 2.0
    return QRectF(x + half, y + half, bw - pen, bh - pen), pen


def _lerp(a: QColor, b: QColor, k: float) -> QColor:
    """One frame of a colour transition, alpha included."""
    return QColor.fromRgbF(a.redF() + (b.redF() - a.redF()) * k,
                           a.greenF() + (b.greenF() - a.greenF()) * k,
                           a.blueF() + (b.blueF() - a.blueF()) * k,
                           a.alphaF() + (b.alphaF() - a.alphaF()) * k)


class Checkbox(Surface):
    """shadcn's Checkbox: ``size-4 shrink-0 rounded-[4px] border shadow-xs``.

    Painted rather than a styled ``QCheckBox``: Qt draws the indicator through
    the platform style, so a stylesheet can move it but not restyle the tick
    itself, and the result is a Windows checkmark inside a shadcn box.

    **The BOX animates; the tick's SHAPE does not.** Both are in the registry
    and they are different statements::

        root       ... transition-colors ...      <- .cn-checkbox, 150 ms
        indicator  ... text-current transition-none

    ``transition-none`` on the indicator stops the tick getting its OWN
    transition -- no fading or scaling glyph. It does not freeze the colour,
    because the indicator is ``text-current``: it inherits, so the root's
    ``transition-colors`` carries the tick's ink along with the fill and the
    border. All three ramp together over one 150 ms curve, which is why the
    tick stays legible against a fill that is still moving.

    The tick is painted only while ``checked``, matching Radix: the Indicator
    MOUNTS on check and UNMOUNTS on uncheck, so it has no exit. Checking ramps
    the tick in with the fill; unchecking drops it at once and lets the box
    fade out under it.

    Neither registry gives the checkbox a ``hover:`` state.

    Moved here from `onboarding.groups` (where it was built and pixel-verified
    against the registry) so `widgets`-layer callers -- `widgets.table`'s own
    `FacetedFilter` popover among them -- can use the SAME checkbox rather
    than a second, rougher approximation. `onboarding.groups` and every other
    caller now import it from here; nothing about its behaviour changed in
    the move.
    """

    toggled = Signal(bool)

    def __init__(self, t, checked: bool = True, parent=None):
        # A :class:`Surface`, purely so the ``shadow-xs`` in the class list gets
        # drawn. A widget CANNOT paint its own shadow -- Qt clips a child to its
        # own rect and a shadow lives entirely outside that rect, so it would be
        # clipped away exactly where it shows. The rule here is that the nearest
        # Surface ANCESTOR paints it, which is why this is a Surface rather than
        # a QWidget trying to draw one itself.
        super().__init__(t, radius=4.0, shadowed=True, shadow=t.shadow_xs,
                         parent=parent)
        self.checked = checked
        self.setFixedSize(16, 16)
        self.setCursor(Qt.PointingHandCursor)
        #: 0 = unchecked, 1 = checked. The colours are all read off this, so
        #: one curve drives the fill, the border and the tick together.
        self._k = 1.0 if checked else 0.0
        self._fade = QVariantAnimation(self)
        self._fade.setDuration(TRANSITION_MS)
        self._fade.setEasingCurve(tw_ease())
        self._fade.valueChanged.connect(self._step)

    def _step(self, k) -> None:
        self._k = float(k)
        self.update()

    def set_checked(self, on: bool, animate: bool = True) -> None:
        """Toggle, running ``transition-colors`` unless told not to."""
        if on == self.checked:
            return
        self.checked = on
        self._fade.stop()
        if not animate:
            self._k = 1.0 if on else 0.0
            self.update()
            return
        self._fade.setStartValue(self._k)
        self._fade.setEndValue(1.0 if on else 0.0)
        self._fade.start()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self.rect().contains(e.pos()):
            self.set_checked(not self.checked)
            self.toggled.emit(self.checked)

    def paintEvent(self, e):
        super().paintEvent(e)
        t = self.t
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        box, pen = _device_box(self)
        # `rounded-[4px]` is the radius of the border-box's OUTER edge, with
        # the 1 px border drawn inside it. Qt's pen straddles the path, so a
        # path radius of 4 on a half-inset rect puts the outer edge at 4.5 --
        # measurably rounder than the registry, which is exactly the sort of
        # half-pixel that reads as "not quite shadcn" on a 16 px box. Half the
        # pen width comes off so the OUTER edge lands on `self.radius`.
        corner = self.radius - pen / 2.0
        k = self._k
        # `bg-transparent -> bg-primary` on light, `dark:bg-input/30 ->
        # bg-primary` on dark. A CSS transition out of `transparent` runs
        # PREMULTIPLIED -- it is the same hue at rising alpha, not a slide
        # through grey -- so the light start is `primary` at alpha 0.
        if t.color_scheme == "dark":
            rest = QColor(t.input)
            rest.setAlphaF(0.30)
        else:
            rest = QColor(t.primary)
            rest.setAlphaF(0.0)
        p.setPen(QPen(_lerp(QColor(t.input), QColor(t.primary), k), pen))
        p.setBrush(_lerp(rest, QColor(t.primary), k))
        p.drawRoundedRect(box, corner, corner)
        if self.checked:
            # ``CheckIcon size-3.5`` = 14 px inside the 16 px box. Drawn at 12
            # here, which read as a small tick in a roomy box. The INK rides
            # the same curve, via the indicator's `text-current`.
            ink = _lerp(QColor(t.foreground), QColor(t.primary_foreground), k)
            icons.paint(p, "check", QRectF(1, 1, 14, 14), ink.name(), stroke=3)
        p.end()


def day_badge(t, fills, inks, text: str, index: int, **kw) -> Badge:
    """A badge on the DAY ramp, tagged with which day it is.

    The tag is what makes an in-place recolour possible: a walker can pick out
    the badges that follow the condition's hue and leave the neutral ones --
    the day COUNT beside "Replicate days", a group's series count -- untouched.
    Without it, recolouring means rebuilding whatever contains them.
    """
    i = min(max(index - 1, 0), len(fills) - 1)
    badge = Badge(t, text, fills[i], inks[i], **kw)
    badge.day_index = index
    return badge


class FitScroll(QScrollArea):
    """A column of a dialog (the Export dialog's settings and preview, the Spatial Map dialog's settings), SCROLLING
    when the window is too short for it. Moved here from `export_dialog` (22 Oct 2026) so both share ONE definition.

    Asked for 14 Oct 2026: "if the screen is too small (resolution) in the exporter, the items
    don't get scrollable but they cluster over each other". The card sits centred in a window
    that may be shorter than it, and a layout squeezed below its own minimum draws its rows
    over each other. Here the size HINT is the whole column (a tall window shows all of it,
    no scroll bar) and the MINIMUM is a few rows (a short window shrinks this, not the rows).
    A column that grows later (the p lines under the pairs arrive with the preview) says so to
    the card: a scroll area does not pass its widget's new size on by itself."""

    MIN_HEIGHT = 160

    def setWidget(self, widget) -> None:
        super().setWidget(widget)
        widget.installEventFilter(self)

    def eventFilter(self, watched, event) -> bool:
        if watched is self.widget() and event.type() == QEvent.LayoutRequest:
            self.updateGeometry()
        return super().eventFilter(watched, event)

    def _bar(self) -> int:
        return self.verticalScrollBar().sizeHint().width()

    def sizeHint(self) -> QSize:
        inner = self.widget()
        if inner is None:
            return super().sizeHint()
        hint = inner.sizeHint()
        return QSize(hint.width() + self._bar(), hint.height())

    def minimumSizeHint(self) -> QSize:
        """A few rows (`MIN_HEIGHT`), or the whole column where it is shorter than that: a short column asked for
        160 px all the same, and the rail's Visibility popover, before a Run two lines of text, opened 190 px tall
        with the lines spread down it (4 Nov 2026, rail review)."""
        inner = self.widget()
        width = inner.minimumSizeHint().width() if inner is not None else 0
        height = min(self.MIN_HEIGHT, inner.sizeHint().height()) if inner is not None else self.MIN_HEIGHT
        return QSize(width + self._bar(), max(0, height))
