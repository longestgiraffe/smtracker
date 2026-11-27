"""Wheel scrolling that eases instead of jumping — and can afford to.

**The first version eased the right value at the wrong frame rate.** It moved a
target toward the scrollbar on a 16 ms timer, which is correct arithmetic and
looked like a stutter, because the page could not paint that fast. Measured on
the Confirm page, one notch::

    frame gaps   p50  0.2   p90 53.9   max 81.3 ms     ~18 fps, not 60
    paint rect   656 px every frame                    the WHOLE viewport
    tail         55 56 57 58 59 60                     one px per ~45 ms

**Why it is that expensive, and why no easing curve can fix it.** A
``QScrollArea`` scrolls by MOVING its content widget, and Qt repaints every
visible child when it moves — it does not blit. Counted here: **320 widget
repaints for a single pixel of scroll**, out of 1035 descendants::

    QLabel 116   QWidget 84   Badge 25   Icon 16   SeriesCard 15   ...

That is not a fault in this page. A bare ``QScrollArea`` holding 300 plain
``QLabel``s repaints ~28 of them per pixel too, and marking the content opaque
changes nothing (4.57 → 4.35 ms/step, same paint count). Scrolled children
repaint; that is simply what the widget path costs.

**So the glide does not scroll widgets at all.** At the start of a gesture the
content is rendered ONCE into a pixmap, updates are suspended on the content
subtree, and each frame blits that pixmap at a new offset::

    snapshot the page     ~30 ms   (~100 ms cold, once per session)
    blit one frame          1.2 ms  -> ~830 fps of headroom
    the old repaint path   40-54 ms -> 18 fps

The cost moves from every frame to once per gesture, which is the whole point:
a single 30 ms hitch as motion begins is two frames, where the old path spent
that much on each of twelve. Notches arriving mid-glide reuse the snapshot, so
a fast scroll pays nothing extra.

**Correctness comes from never keeping the snapshot.** It is captured when a
glide starts and dropped when it ends, so it cannot go stale behind a content
change; anything that could invalidate it mid-glide (the scroll range moving
because the day table opened, a viewport resize) ends the glide instead of
being tracked. The one thing that MUST be ordered carefully is the handover:
updates are re-enabled and the content repainted SYNCHRONOUSLY before the
surface is hidden, or the stale pixels underneath flash for a frame.

**Easing is frame-time-corrected.** A fixed fraction per tick silently turns a
late frame into a longer glide — which is exactly what the old measurements
show. The decay is a time constant instead, so the curve describes the same
motion whether frames arrive every 16 ms or every 30.

**Precise-pixel devices are left alone.** A trackpad already delivers smooth,
high-frequency deltas; easing those again adds latency and makes the surface
feel like it is dragging behind your fingers. Those events fall through to Qt
untouched.

Scoped deliberately to ONE scroll area — the page. Every other scroller in this
window is nested inside another one, and a swallowed wheel event has to be
handed up when the inner area reaches its limit; that hand-off is the part
worth doing carefully rather than quickly. The page has nothing outside it, so
it has no hand-off to get wrong.
"""

from __future__ import annotations

import math

from qtpy.QtCore import QEvent, QObject, QPoint, QPointF, Qt, QTimer
from qtpy.QtGui import QColor, QPainter, QPixmap
from qtpy.QtWidgets import QAbstractSlider, QApplication, QWidget

__all__ = ["GlideSurface", "SmoothScroller", "attach", "stop_all"]


class GlideSurface(QWidget):
    """A viewport-sized overlay that paints a snapshot of the page.

    It exists only while a glide is running. Between glides it is hidden and
    holds no pixmap, so it can never show content that has since changed.
    """

    def __init__(self, viewport: QWidget, background: QColor | None = None):
        super().__init__(viewport)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._pix: QPixmap | None = None
        self._offset = (0.0, 0.0)
        self._bg = background
        self.hide()

    def capture(self, content: QWidget) -> bool:
        """Render ``content`` once. False if there is nothing worth painting."""
        if content.width() <= 0 or content.height() <= 0:
            return False
        dpr = self.devicePixelRatioF()
        pix = QPixmap(round(content.width() * dpr), round(content.height() * dpr))
        pix.setDevicePixelRatio(dpr)
        # The page paints a transparent background over the viewport's, so the
        # snapshot needs that colour under it or the glide shows through.
        pix.fill(self._bg if self._bg is not None else Qt.transparent)
        content.render(pix, QPoint(0, 0))
        self._pix = pix
        return True

    def release(self) -> None:
        self._pix = None

    def set_offset(self, x: float, y: float) -> None:
        """Move the snapshot. Snapped to whole DEVICE pixels, not logical ones.

        At ratio 1.25 that is a 0.8 px grid — finer than the scrollbar's own
        integer steps, so the glide is smoother than the value driving it, and
        still lands on the pixel grid so text never resamples into a blur.

        Both axes, because the lanes board scrolls SIDEWAYS: it is the one
        scroller here whose content is wider than its viewport rather than
        taller.
        """
        dpr = self.devicePixelRatioF() or 1.0
        snapped = (round(x * dpr) / dpr, round(y * dpr) / dpr)
        if snapped != self._offset:
            self._offset = snapped
            self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        if self._bg is not None:
            p.fillRect(self.rect(), self._bg)
        if self._pix is not None:
            p.drawPixmap(QPointF(-self._offset[0], -self._offset[1]), self._pix)
        p.end()


class SmoothScroller(QObject):
    """Eases one scroll area toward a target the wheel keeps moving."""

    #: ~60 Hz. The timer itself is not the problem — measured here at p50
    #: 15.9 ms, and ``PreciseTimer`` makes no difference — so the interval is
    #: an upper bound on the frame rate, not a promise about it. The easing
    #: below reads the clock rather than trusting this number.
    INTERVAL_MS = 16

    #: Exponential time constant. The fraction of the remaining gap covered in
    #: one 16 ms frame is ``1 - exp(-16/TAU)``::
    #:
    #:     TAU 35   0.37 per frame   ~161 ms to settle
    #:     TAU 45   0.30 per frame   ~207 ms
    #:     TAU 60   0.23 per frame   ~276 ms
    #:
    #: 45 suits this page: the whole scroll range is 159 px, so a glide that
    #: outlasts the distance it covers reads as lag rather than as motion.
    TAU_MS = 45.0

    #: Below this the glide is over -- the remainder is snapped, so it always
    #: LANDS exactly on the target whatever this is set to.
    #:
    #: An exponential ease never arrives, so the tail is pure waiting: at
    #: TAU 45 ms it costs ~31 ms to cross from half a pixel to a quarter, and
    #: nobody can see either. MEASURED, one notch on the table, glide only::
    #:
    #:     SNAP_PX   device px (DPR 1.25)   glide      lands at
    #:      0.25            0.31            272 ms        30
    #:      0.5             0.62            246 ms        30
    #:      1.0             1.25            212 ms        30
    #:      2.0             2.50            185 ms        30
    #:
    #: 0.5 is the largest step that is still SUB-PIXEL on every ratio we ship
    #: on (0.5 device px at DPR 1.0, 0.62 here). 1.0 saves another 34 ms but
    #: snaps a whole device pixel at DPR 1.0, which is a visible tick at the end
    #: of a scroll -- the wrong thing to trade for 34 ms.
    SNAP_PX = 0.5

    #: A frame this late means the app stalled (a dialog, a relayout). Easing
    #: across it would teleport the page; clamp instead and lose a little time.
    MAX_FRAME_MS = 50.0

    #: How much of Qt's own wheel step one notch travels.
    #:
    #: Qt's answer here is ``singleStep * wheelScrollLines`` = 20 x 3 = **60 px**
    #: at the Windows default, which is tuned for a long document and is far too
    #: coarse for this page. Measured, notches needed to cross the whole range::
    #:
    #:     state                     range   60 px   40 px   30 px   24 px
    #:     day table closed          159      2.6     4.0     5.3     6.6
    #:     day table open            484      8.1    12.1    16.1    20.2
    #:     closed, short window      283      4.7     7.1     9.4    11.8
    #:     open, short window        608     10.1    15.2    20.3    25.3
    #:
    #: Half is the choice: 5.3 notches across the page you normally see, and 20
    #: in the tallest state -- about a second of wheel, and notches accumulate,
    #: so a fast spin still covers it in one glide. Going finer than this makes
    #: the tall state tedious without making the common one better.
    #:
    #: A SCALE, not a pixel count, so someone who has set their wheel to 1 line
    #: or 6 still gets what they asked for -- just less of it.
    NOTCH_SCALE = 0.5

    #: How long the page stays frozen after the glide settles.
    #:
    #: Nobody scrolls one notch. A burst is what a wheel produces, and without
    #: this each notch in it would pay BOTH once-per-gesture costs again --
    #: measured at ~30 ms to snapshot and ~35 ms to hand back::
    #:
    #:     blit      2.2 ms avg   (max 5.6)
    #:     setValue  0.1 ms       (updates suspended: the move is free)
    #:     tick      0.25 ms      except the last, 35.0 ms -- the handover
    #:
    #: So the surface is kept up for a moment after motion stops. A notch
    #: arriving inside that window resumes on the SAME snapshot: no capture, no
    #: handover, nothing between the wheel and the next blit. Any mouse press
    #: ends it immediately regardless.
    #:
    #: **160 -> 80, because the original number was never measured against what
    #: it protects.** The rationale was right ("nobody scrolls one notch") and
    #: the value was a guess, and a guess that costs something: while the
    #: surface is up the page is a STATIC IMAGE, so a row that should light
    #: under the pointer cannot -- which is exactly the "the hover comes back
    #: late" report.
    #:
    #: What the grace actually protects is narrower than it looks. A burst's
    #: notches arrive 20-80 ms apart, which is far INSIDE the glide (~250 ms for
    #: one notch), and a notch arriving mid-glide never needed a capture anyway.
    #: The grace only matters for the second of two notches separated by more
    #: than a whole glide. MEASURED, captures for two notches ``d`` apart --
    #: 1 means the snapshot was reused, 2 means a fresh capture::
    #:
    #:     GRACE   d=150  200  250  300  400
    #:       160       1    1    1    1    1
    #:       120       1    1    1    1    2
    #:       100       1    1    1    1    2
    #:        80       1    1    1    1    2
    #:        60       1    1    1    2    2
    #:
    #: 80 behaves identically to 160 out to a 300 ms gap -- well past any burst
    #: -- and hands the page back 80 ms sooner. Below 80 it starts re-capturing
    #: on gestures a person actually makes.
    GRACE_MS = 80

    def __init__(self, area, background=None, notch_scale=None,
                 notch_px=None, page_px=None, parent=None):
        super().__init__(parent or area)
        self.area = area
        #: The bar the CURRENT glide drives. Chosen per gesture, not fixed:
        #: the lanes board has no vertical range at all.
        self.bar = area.verticalScrollBar()
        self.enabled = True
        self.notch_scale = (self.NOTCH_SCALE if notch_scale is None
                            else float(notch_scale))
        self.notch_px = None if notch_px is None else float(notch_px)
        #: How far a PAGE action travels, overriding Qt's ``pageStep``. See
        #: :meth:`_on_action`; the lanes board is the reason it exists.
        self.page_px = None if page_px is None else float(page_px)
        self._position = float(self.bar.value())
        self._target = self._position
        self._last_ms = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(self.INTERVAL_MS)
        self._timer.timeout.connect(self._tick)
        self._grace = QTimer(self)
        self._grace.setSingleShot(True)
        self._grace.setInterval(self.GRACE_MS)
        self._grace.timeout.connect(self._end)
        self._elapsed = None
        colour = QColor(background) if background else None
        self._surface = GlideSurface(area.viewport(), colour)
        self._gliding = False
        area.viewport().installEventFilter(self)
        # Anything that can invalidate the snapshot ends the glide rather than
        # being tracked: a stale snapshot is a wrong picture, not a slow one.
        for b in (area.verticalScrollBar(), area.horizontalScrollBar()):
            b.rangeChanged.connect(self._interrupt)
            # The BAR ITSELF is an input, and it was the one still jumping.
            b.actionTriggered.connect(
                lambda action, bar=b: self._on_action(bar, action))
        # How an exhausted inner wheel finds its way out: chaining walks the
        # parent chain looking for this marker.
        area._smooth_scroller = self
        _LIVE.append(self)
        area.destroyed.connect(lambda *_: _forget(self))

    # -- the filter ---------------------------------------------------------- #
    def eventFilter(self, obj, event):
        kind = event.type()
        if kind in (QEvent.MouseButtonPress, QEvent.KeyPress) and self._gliding:
            # Anything that can CHANGE the page ends the freeze at once, or the
            # change would sit invisible behind the snapshot until the grace
            # window expired. The filter for this is on the application, not on
            # the viewport: a click lands on whichever deep child is under the
            # pointer, and the viewport never sees it.
            self._interrupt()
            return False
        if kind is QEvent.Resize and obj is self.area.viewport():
            self._interrupt()   # a resize invalidates the snapshot
            return False
        if kind is not QEvent.Wheel or not self.enabled:
            return False
        # A trackpad or a precision mouse: already smooth, hands off.
        if not event.pixelDelta().isNull():
            self._interrupt()
            return False
        degrees = event.angleDelta().y() or event.angleDelta().x()
        if not degrees:
            return False
        if self.push(degrees, direct=True):
            return True
        # Out of room. Qt will NOT do this for us -- measured: an exhausted
        # inner wheel is SWALLOWED, the page does not move -- so the notch is
        # handed to the nearest ancestor that can use it, explicitly.
        return self._chain(degrees)

    # -- axis, room and chaining --------------------------------------------- #
    def _pick_bar(self, direct: bool):
        """Which scrollbar this notch should drive, or None for neither.

        Vertical wins wherever it exists. A DIRECT wheel over an area that only
        scrolls sideways drives its horizontal bar — that is the lanes board,
        which otherwise could not be wheeled at all (measured: Qt does not map
        the axis here, so the board was drag-only).

        A CHAINED notch never takes that mapping. Scrolling down inside a lane
        that has run out must not turn into a sideways pan of the board behind
        it; the notch keeps travelling up until it finds something vertical.
        """
        v = self.area.verticalScrollBar()
        if self._usable(v, self.area.verticalScrollBarPolicy()):
            return v
        if direct:
            h = self.area.horizontalScrollBar()
            if self._usable(h, self.area.horizontalScrollBarPolicy()):
                return h
        return None

    @staticmethod
    def _usable(bar, policy) -> bool:
        """Can this bar actually take a notch?

        POLICY, not range, and not ``isVisible()`` -- both of the obvious tests
        are wrong here, measured on this window:

        * The lanes board's vertical bar is ``ScrollBarAlwaysOff`` and still
          reports a range of **0..2**, a two-pixel residue of content 360 px
          tall in a 358 px viewport. Range alone hands every notch to a bar
          nobody can see, which is exactly what swallowed the sideways pan.
        * The table's vertical bar has a real 630 px range and reports
          ``isVisible() == False`` whenever the stack is showing the lanes
          instead. Visibility answers "is this page on screen", not "does this
          area scroll".
        """
        return (policy is not Qt.ScrollBarAlwaysOff
                and bar.maximum() > bar.minimum())

    def push(self, degrees: float, direct: bool = True) -> bool:
        """Aim this scroller at a new target. False if it has no room to move."""
        if not self.enabled:
            return False
        bar = self._pick_bar(direct)
        if bar is None:
            return False
        base = self._base_for(bar)
        return self._aim(bar, base, base - degrees / 120.0 * self._notch(bar))

    def _base_for(self, bar) -> float:
        """Where a fresh aim starts from, after settling any axis change."""
        if self._gliding and bar is not self.bar:
            self._interrupt()          # the axis changed under the glide
        return self._target if self._gliding else float(bar.value())

    def _aim(self, bar, base: float, want: float) -> bool:
        """Glide ``bar`` from ``base`` to ``want``. False if it cannot move.

        The shared tail of every input -- a wheel notch and a click on the
        scrollbar differ only in how they arrive at ``want``.
        """
        target = max(float(bar.minimum()), min(want, float(bar.maximum())))
        if target == base:
            return False               # at the limit: the caller chains onward
        was_gliding = self._gliding
        self.bar = bar
        if not was_gliding:
            self._position = base
        self._target = target
        if not was_gliding and not self._begin():
            return False               # no surface: let Qt scroll it plainly
        self._resume()
        return True

    #: The scrollbar actions that JUMP, and so are worth easing.
    #:
    #: ``SliderMove`` is deliberately absent, and that is the whole design of
    #: this: a thumb drag is DIRECT MANIPULATION and must track the pointer
    #: 1:1. Easing it would leave the content lagging behind the thumb the user
    #: is holding, which reads as the app being slow rather than as motion.
    #: Stored as PLAIN INTS, matched through :func:`_action_value`.
    #:
    #: PySide6 delivers this signal's argument as a bare ``int`` while the enum
    #: members are ``SliderAction`` objects, and the two do not hash equal --
    #: so a set of enum members silently matches NOTHING. Worth spelling out
    #: because of HOW it fails: no error, no warning. The first version of this
    #: passed lint, connected cleanly and ran on every single click, and eased
    #: nothing; from the outside it was indistinguishable from the signal never
    #: having been wired up. It was found by printing ``in_set`` -- the guard,
    #: not the machinery -- rather than by reading the code again.
    #:
    #: (``int(member)`` does not work either: this enum is not an ``IntEnum``,
    #: so the conversion raises. ``.value`` is the way across.)
    _STEP_ACTIONS = frozenset(a.value for a in (
        QAbstractSlider.SliderSingleStepAdd, QAbstractSlider.SliderSingleStepSub,
        QAbstractSlider.SliderPageStepAdd, QAbstractSlider.SliderPageStepSub,
        QAbstractSlider.SliderToMinimum, QAbstractSlider.SliderToMaximum,
    ))

    #: The two of :attr:`_STEP_ACTIONS` that ``page_px`` overrides.
    _PAGE_ACTIONS = frozenset(a.value for a in (
        QAbstractSlider.SliderPageStepAdd, QAbstractSlider.SliderPageStepSub,
    ))

    def _on_action(self, bar, action: int) -> None:
        """Take over a scrollbar's own jump and glide it instead.

        MEASURED on the lanes board, which is the case this was asked for --
        the value of the horizontal bar, sampled every 8 ms::

            wheel                12 intermediate values   (already eased)
            click in the trough   1                       0 -> 514, one frame
            thumb drag            1                       correct: 1:1

        So the wheel was smooth and the BAR was not. A trough click there is
        not a small jump either: the board is 1912 px of content in a 1398 px
        viewport, so ``pageStep`` (1398) exceeds the whole range (514) and one
        click crosses everything there is.

        The handover is the documented one, not a trick. When ``actionTriggered``
        fires, Qt has already adjusted ``sliderPosition`` but has NOT propagated
        it to ``value``, and the docs name this signal as the place to adjust an
        action. So the intended destination is read off ``sliderPosition``, the
        position is put back, and the glide takes it from there. Qt's own
        ``blocktracking`` is set for the duration of the emit, so putting it
        back cannot leak a ``valueChanged`` of its own.

        Auto-repeat (a held press) accumulates correctly: each repeat computes
        from wherever the glide has reached and extends the target.

        ``page_px``, where a caller sets it, replaces how FAR a page action
        travels while leaving Qt to decide the direction.
        """
        # Whichever form the binding hands over -- see `_STEP_ACTIONS`.
        if not self.enabled or getattr(action, "value",
                                       action) not in self._STEP_ACTIONS:
            return
        want = float(bar.sliderPosition())
        base = self._base_for(bar)
        if self.page_px is not None and int(action) in self._PAGE_ACTIONS:
            # Qt's own page is a VIEWPORT, which on this board is more than the
            # whole range -- so every trough click is "go to the far end" and
            # there is nowhere to land. `page_px` replaces the distance, not the
            # direction: the sign still comes from what Qt decided to do.
            step = self.page_px if want > base else -self.page_px
            want = base + step
        if want == base:
            return
        bar.setSliderPosition(int(base))
        self._aim(bar, base, want)

    def _chain(self, degrees: float) -> bool:
        """Give the notch to the nearest ANCESTOR scroller that can use it."""
        w = self.area.parentWidget()
        while w is not None:
            other = getattr(w, "_smooth_scroller", None)
            if other is not None and other is not self and other.push(
                    degrees, direct=False):
                return True
            w = w.parentWidget()
        return False

    #: The wheel setting Windows ships with, and what ``notch_px`` is quoted at.
    DEFAULT_WHEEL_LINES = 3

    def _notch(self, bar=None) -> float:
        """How far one detent travels.

        Normally Qt's own step, scaled down. ``notch_px`` replaces that answer
        outright for an area where Qt's is content-derived and useless: the
        lanes board's horizontal ``singleStep`` yields **202 px** a detent,
        which crosses the whole 524 px board in 2.6 of them — the same fault
        the page had at 60. It is still multiplied by the user's line setting,
        so someone who has asked for a coarser wheel still gets one.
        """
        bar = self.bar if bar is None else bar
        lines = QApplication.wheelScrollLines()
        if self.notch_px is not None:
            return max(self.notch_px * lines / self.DEFAULT_WHEEL_LINES, 1.0)
        return max(bar.singleStep() * lines * self.notch_scale, 1.0)

    def _clamp(self) -> None:
        self._target = max(float(self.bar.minimum()),
                           min(self._target, float(self.bar.maximum())))

    # -- the glide ----------------------------------------------------------- #
    def _begin(self) -> bool:
        """Snapshot the page, freeze it, and start the timer."""
        content = self.area.widget()
        if content is None:
            return False
        # ONE frozen area at a time. These scrollers nest -- a lane list lives
        # inside the board, which lives inside the page -- and freezing an
        # ancestor suspends updates on the descendant's surface too, so its
        # blits would go nowhere. Whoever is starting wins.
        for other in _LIVE:
            if other is not self:
                other._interrupt()
        surface = self._surface
        surface.setGeometry(self.area.viewport().rect())
        if not surface.capture(content):
            return False
        surface.set_offset(*self._offsets())
        surface.show()
        surface.raise_()
        # Anything else floating over the viewport -- the selection bar -- is a
        # live control and belongs above the frozen picture of the page.
        for child in self.area.viewport().children():
            if isinstance(child, QWidget) and child not in (surface, content):
                child.raise_()
        content.setUpdatesEnabled(False)
        self._gliding = True
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
        return True

    def _resume(self) -> None:
        """(Re)start the frame timer, with the clock reset to NOW.

        Resetting matters: a notch arriving after the grace window has been
        counting would otherwise be handed a large ``dt`` on its first frame
        and jump most of the way there before drawing anything.
        """
        self._grace.stop()
        if not self._timer.isActive():
            self._last_ms = 0.0
            self._elapsed = _Stopwatch()
            self._timer.start()

    def _tick(self) -> None:
        now = self._elapsed.ms() if self._elapsed else self._last_ms + self.INTERVAL_MS
        dt = min(max(now - self._last_ms, 1.0), self.MAX_FRAME_MS)
        self._last_ms = now
        gap = self._target - self._position
        if abs(gap) < self.SNAP_PX:
            self._position = self._target
            self.bar.setValue(round(self._position))
            self._surface.set_offset(*self._offsets())
            # Settled, but not finished: hold the frozen page for a moment in
            # case this was one notch of a burst.
            self._timer.stop()
            self._grace.start()
            return
        # Frame-time-corrected, so a late frame covers the ground it should
        # have rather than stretching the glide by exactly how late it was.
        self._position += gap * (1.0 - math.exp(-dt / self.TAU_MS))
        self.bar.setValue(round(self._position))
        self._surface.set_offset(*self._offsets())

    def _offsets(self) -> tuple[float, float]:
        """Where the snapshot sits. The idle axis reads its bar, not zero.

        The table can be scrolled sideways by dragging while the wheel moves it
        vertically; taking the other axis from its own bar keeps the frozen
        picture where the live widgets were.
        """
        h = float(self.area.horizontalScrollBar().value())
        v = float(self.area.verticalScrollBar().value())
        if self.bar.orientation() is Qt.Horizontal:
            h = self._position
        else:
            v = self._position
        return h, v

    def _end(self) -> None:
        """Hand the page back to the widgets. ORDER MATTERS."""
        self._timer.stop()
        self._grace.stop()
        if not self._gliding:
            return
        self._gliding = False
        self._elapsed = None
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        self.bar.setValue(round(self._position))
        content = self.area.widget()
        if content is not None:
            content.setUpdatesEnabled(True)
            # Synchronous, BEFORE the surface goes: the widgets underneath hold
            # whatever pixels they had when the glide began, and hiding first
            # would show them for a frame.
            content.repaint()
        self._surface.hide()
        self._surface.release()

    def _interrupt(self, *_args) -> None:
        """End a glide whose snapshot can no longer be trusted."""
        if self._gliding:
            self._clamp()
            self._position = self._target
            self._end()
        else:
            self._position = float(self.bar.value())
            self._target = self._position

    def stop(self) -> None:
        """End any glide immediately. Used before a screenshot."""
        self._interrupt()


class _Stopwatch:
    """Monotonic milliseconds. Separate so the tick maths stays testable."""

    def __init__(self):
        import time
        self._t0 = time.perf_counter()

    def ms(self) -> float:
        import time
        return (time.perf_counter() - self._t0) * 1000.0


#: Every live scroller, so a glide can freeze exactly one area and a screenshot
#: can settle all of them. Entries are dropped when their scroll area dies.
_LIVE: list[SmoothScroller] = []


def _forget(scroller) -> None:
    if scroller in _LIVE:
        _LIVE.remove(scroller)


def stop_all() -> None:
    """Settle every glide. What ``shoot.py`` calls before a grab."""
    for scroller in _LIVE:
        scroller.stop()


def attach(area, background=None, notch_scale=None, notch_px=None,
           page_px=None, enabled: bool = True) -> SmoothScroller:
    """Give ``area`` an eased wheel and an eased scrollbar. Returns the scroller.

    ``background`` is the colour the page paints on; the snapshot needs it
    because the content itself is transparent.

    Three knobs, for the three ways an area is scrolled:

    * ``notch_scale`` -- one wheel detent as a fraction of Qt's own step.
    * ``notch_px`` -- one wheel detent as a flat distance, where Qt's step is
      content-derived and useless.
    * ``page_px`` -- how far the SCROLLBAR's page action travels, where Qt's
      viewport-sized page overshoots the whole range.

    A thumb drag has no knob and never will: it is 1:1 by definition.
    """
    scroller = SmoothScroller(area, background=background,
                              notch_scale=notch_scale, notch_px=notch_px,
                              page_px=page_px)
    scroller.enabled = enabled
    return scroller
