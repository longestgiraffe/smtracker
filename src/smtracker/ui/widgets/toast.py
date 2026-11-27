"""Sonner, as far as a computation that outlives the page you started it on.

PORTED from `prototype/shadcn/toasts.py` (the UI designer's), kept as close to it as the
application allows: the imports point at `ui.theme` / `ui.widgets`, Qt comes
through `qtpy`, and the root is the workspace's 16 px. How the workspace uses it
is in `ui.workspace.window` ("TOASTS"): sections there run SIDE BY SIDE, so the
replace-it prompt is asked within a section, not across the whole window.

shadcn's toast IS Sonner (the older ``toast``/``Toaster`` pair is deprecated in
the registry in its favour), so the geometry and the motion here are Sonner's:
bottom-right, ``bg-popover`` on a ``rounded-lg`` with ``shadow-lg``, one title
line and an optional description, slid up into place and faded, auto-dismissed
after four seconds.

**Why this module exists, which is not "because notifications are nice".**
A computation can now be started and then walked away from -- you press the mark
on `Diffusion analysis`, cross to `Gaussian mixture`, and the job keeps running.
The moment that is true, a finished job has nowhere left to announce itself: the
mark on the page you are now looking at belongs to *that* page and correctly
shows `play`. The toast is the only channel a result can arrive on once the
reader has moved. Without it, "you may navigate freely during a computation"
silently also means "results you never hear about".

**Two kinds of surface, and they behave differently on purpose.**

*A notification* -- "this finished" -- auto-dismisses. Nothing is lost by
missing it; the figures are drawn either way.

*A prompt* -- "something is already running, replace it?" -- does NOT. A
decision that expires is a decision lost, and worse, one the reader cannot tell
they lost. Sonner spells this ``duration: Infinity``; here it is
``duration=None``. It is really a non-modal dialogue wearing a toast's clothes,
and it needs the one thing a notification does not: **two exits.** `Proceed`,
and a close that means *keep what is running*. With only an action button,
walking away is indistinguishable from choosing, and afterwards neither the
reader nor the code can tell which happened.

**The hover pause is not decoration either.** Four seconds is enough to notice a
toast and not always enough to read one, and a message that vanishes mid-read is
worse than one that never appeared -- the reader knows they missed something.
Sonner pauses on hover for the same reason.
"""

from __future__ import annotations

from qtpy.QtCore import (
    QEasingCurve,
    QEvent,
    QMargins,
    QObject,
    QRectF,
    Qt,
    QTimer,
    QVariantAnimation,
    Signal,
)
from qtpy.QtGui import QPainter
from qtpy.QtWidgets import (
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ...chart_text import PATH_BREAKS, wrap_text
from ..theme import typography as ty
from .controls import Button
from .primitives import Surface, draw_shadow, icon, label, shadow_room

__all__ = ["DURATION_MS", "MOTION_MS", "TOAST_W", "Toast", "Toaster"]

#: Must equal the workspace's ``ROOT_PX`` (``toolbar.ROOT_PX``, 16 here; the
#: prototype's root was 18), for the reason ``toolbar.ROOT_PX`` gives.
ROOT_PX = 16


def rem(x: float) -> int:
    return round(x * ROOT_PX)


def text_px(rem_: float) -> int:
    return int(rem_ * ROOT_PX)


#: Sonner's own: ``width: 356px``, ``--offset: 32px``, ``gap: 14px``. Expressed
#: in this panel's rem so they scale with everything else, rather than being
#: three numbers that happen to be right at one root.
TOAST_W = rem(22)               # 352, against Sonner's 356
OFFSET = rem(1.5)               # in from the page's bottom-right corner
GAP = 14                        # Sonner's ``--gap``: between toasts, and the peek

#: ``p-4`` inside, and the registry's ``rounded-lg`` for a floating surface.
PAD = rem(1.0)

#: Four seconds -- Sonner's default, long enough to read a subsection name and
#: short enough not to pile up. A prompt passes ``None`` instead.
DURATION_MS = 4000

#: ONE MOTION FOR EVERY MOVEMENT: arriving, leaving, the others closing the gap,
#: slipping back in the stack, and the hover expand -- ruled, so none falls out of
#: step with the rest. 280 ms, under Sonner's 400, on an EASE-OUT, on purpose. Each
#: of these answers something the reader just did, and at 400 ms on ease-in-out
#: almost nothing moved for the first ~100 ms, which read as lag (reported twice);
#: ease-out moves at full speed on the first frame and settles gently. 200 ms was
#: then reported as too fast, so 280 -- the one number to tune.
MOTION_MS = 280


def motion_curve() -> QEasingCurve:
    return QEasingCurve(QEasingCurve.OutCubic)


#: THE STACK, Sonner's rules read from its ``styles.css``: collapsed, each toast
#: ``n`` places behind the front lifts by ``n * GAP`` and scales by
#: ``1 - SCALE_STEP * n``, takes the FRONT toast's height, and hides its content
#: (only its edge peeks out); at most ``VISIBLE`` are drawn, the rest are kept,
#: invisible, until those in front leave. Hovering the stack expands it.
SCALE_STEP = 0.05
VISIBLE = 3


def _lerp(a: float, b: float, f: float) -> float:
    return a + (b - a) * f


class Toast(QWidget):
    """One notification, or one question.

    The widget is BIGGER than the card it draws: ``shadow-lg`` needs 22 px
    outside the rounded rect, and a child widget cannot paint past its own
    edges, so the card is inset by that much and the surrounding pixels are
    where the shadow lands. The same reserve a tooltip keeps for its slide.

    **A smaller card is GEOMETRY, not a transform.** Qt cannot scale a child
    widget, so a toast further back in the stack is drawn by giving its card a
    smaller rect (the `Toaster` sets the widget's geometry every frame). The text
    is not scaled at all: it lives in ``content``, laid out once at full size,
    centred in the card and clipped by it, and FADED out behind the front toast
    -- which is what Sonner does too (``opacity: 0`` on a back toast's content).
    """

    #: Fully out, and ready to be dropped from the stack.
    gone = Signal()

    def __init__(self, t, title: str, *, description: str | None = None,
                 action: str | None = None, duration: int | None = DURATION_MS,
                 parent=None):
        super().__init__(parent)
        self.t = t
        self.room = shadow_room(t, t.shadow_lg)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self._on_action = None
        self._on_close = None
        self._dismissing = False
        self._held = False
        self._duration = duration
        #: 0 = a full height below its seat and invisible, 1 = seated.
        self._at = 0.0
        #: Places behind the front toast, animated toward the stack's answer.
        self._depth = 0.0
        self._depth_target = 0.0
        #: The card's height at full size -- what `fit` measured.
        self.natural_h = 0
        self._toaster = None

        card = Surface(t, radius=t.radius_lg, shadowed=False, parent=self)
        card.setObjectName(f"toast{id(self)}")
        card.setStyleSheet(
            f"QFrame#{card.objectName()} {{ background:{t.popover};"
            f" border:1px solid {t.border};"
            f" border-radius:{t.radius_lg:g}px; }}")
        self.card = card
        # The text and buttons: a SIBLING of the card, laid over it, never its child.
        # Two nested opacity effects do not repaint here -- the outer one caches its
        # picture, so text faded out behind the front toast never came back when the
        # stack expanded. Card, text and shadow each carry their own opacity instead.
        self.content = QWidget(self)
        self.content.setAttribute(Qt.WA_TranslucentBackground, True)
        row = QHBoxLayout(self.content)
        row.setContentsMargins(PAD, PAD, PAD, PAD)
        row.setSpacing(rem(0.75))

        words = QVBoxLayout()
        words.setContentsMargins(0, 0, 0, 0)
        words.setSpacing(rem(0.25))
        self.title = label(title, ty.ui("sm", "medium", px=text_px(0.875)),
                           t.foreground)
        self.title.setWordWrap(True)
        # PREFERRED, not Minimum: Minimum makes a wrapped label's size HINT its
        # minimum, and Qt takes that hint at a narrow default width (three lines
        # where two fit), so the card could not shrink to its text -- measured 123
        # px against the 89 the layout needs. The height comes from `fit` instead.
        self.title.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        words.addWidget(self.title)
        self.description = None
        self._description_text = description or ""
        if description:
            self.description = label(description,
                                     ty.ui("sm", px=text_px(0.875)),
                                     t.muted_foreground)
            self.description.setWordWrap(True)
            self.description.setSizePolicy(QSizePolicy.Preferred,
                                           QSizePolicy.Preferred)
            words.addWidget(self.description)
        # Any height the card is given beyond what its text needs goes BELOW the
        # text, never between title and description: a wrapped title measured at
        # one width and laid out at another otherwise opened a gap between them.
        words.addStretch(1)
        row.addLayout(words, 1)

        # THE ACTION, then the close, left to right: the reader scans to the
        # affirmative first, and the way out sits at the far edge, where a way
        # out belongs.
        #
        # BOTH TOP-ALIGNED, and that is a correction. Centring the action --
        # which is what Sonner does, and what looks right on its one-line demo
        # toasts -- put the button level with the MIDDLE of a four-line column
        # here, i.e. wedged between the title and the description, as though it
        # belonged to neither. Aligned to the top it sits beside the title,
        # which is the line it answers.
        self.action = None
        if action is not None:
            self.action = Button(t, action, variant="default", size="small")
            self.action.clicked.connect(self._acted)
            row.addWidget(self.action, 0, Qt.AlignTop)
        self.close_btn = _Close(t)
        self.close_btn.clicked.connect(self._closed)
        row.addWidget(self.close_btn, 0, Qt.AlignTop)

        # The auto-dismissal, and the hover that holds it. A prompt has no
        # timer at all, so there is nothing for a hover to pause -- which is
        # the whole of the difference between the two kinds.
        self._life = QTimer(self)
        self._life.setSingleShot(True)
        self._life.timeout.connect(self.dismiss)

        #: The whole toast's opacity (arriving, leaving, out of view) and the text's
        #: own (0 behind the front toast); see `set_opacity`.
        self._opacity = 0.0
        self._content_opacity = 1.0
        self._card_fade = QGraphicsOpacityEffect(self.card)
        self._card_fade.setOpacity(0.0)
        self.card.setGraphicsEffect(self._card_fade)
        self._content_fade = QGraphicsOpacityEffect(self.content)
        self._content_fade.setOpacity(0.0)
        self.content.setGraphicsEffect(self._content_fade)

        self._depth_anim = QVariantAnimation(self)
        self._depth_anim.setDuration(MOTION_MS)
        self._depth_anim.setEasingCurve(motion_curve())
        self._depth_anim.valueChanged.connect(self._on_depth)

        self._slide = QVariantAnimation(self)
        self._slide.setDuration(MOTION_MS)
        self._slide.setEasingCurve(motion_curve())
        self._slide.valueChanged.connect(self._on_slide)
        self._slide.finished.connect(self._slide_done)

    # -- geometry ---------------------------------------------------------- #

    def fit(self) -> None:
        """Measure the card's full-size height for ``TOAST_W``.

        NO HEIGHT-FOR-WIDTH LEFT ON SCREEN: a word-wrapped label with the flag set is
        what the workspace's own guard forbids (it makes every layout above it
        re-solve on each width change -- `primitives.FitLabel`). So the text column
        is laid out at the real width FIRST, each label is pinned to the height its
        text needs at the width it actually got, and the content is then measured
        with nothing left to negotiate. (The card's own hint once took the wrapped
        title at a guessed width and came out a line too tall.)
        """
        lay = self.content.layout()
        labels = [w for w in (self.title, self.description) if w is not None]
        for lab in labels:
            lab.setMinimumHeight(0)
            lab.setMaximumHeight(16777215)
        self.content.resize(TOAST_W, self.content.height() or 1)
        lay.activate()
        lay.setGeometry(self.content.rect())
        if self.description is not None and self.description.width() > 1:
            # A FOLDER IN THE TEXT breaks after its separators (`chart_text.PATH_BREAKS`): a path has no spaces, and a
            # word-wrapped label cut it at the card's edge, so where the files went could not be read (found by the
            # overnight tester, 4 Nov 2026). Lines at the width the label actually got, every character kept.
            fm = self.description.fontMetrics()
            self.description.setText("\n".join(wrap_text(
                self._description_text, fm.horizontalAdvance, self.description.width() - 2, max_lines=12,
                breaks=PATH_BREAKS)))
        for lab in labels:
            pol = lab.sizePolicy()
            pol.setHeightForWidth(False)
            lab.setSizePolicy(pol)
            lab.setFixedHeight(lab.heightForWidth(max(1, lab.width())))
        lay.activate()
        self.natural_h = lay.totalSizeHint().height()
        self.content.resize(TOAST_W, self.natural_h)
        self.resize(TOAST_W + 2 * self.room, self.natural_h + 2 * self.room)

    def card_height(self) -> int:
        return self.height() - 2 * self.room

    def sizeHint(self):
        return self.content.sizeHint().grownBy(
            QMargins(self.room, self.room, self.room, self.room))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        w, h = self.width() - 2 * self.room, self.height() - 2 * self.room
        self.card.setGeometry(self.room, self.room, w, h)
        # Over the card, at the card's size: behind the front toast the text is
        # faded out, so a card a few per cent narrower never shows it reflowed.
        self.content.setGeometry(self.room, self.room, w, max(h, 1))

    def paintEvent(self, e):
        """``shadow-lg`` under the card, in the room reserved for it."""
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setOpacity(self._opacity)
        draw_shadow(p, QRectF(self.card.geometry()), self.t.radius_lg,
                    self.t, self.t.shadow_lg)
        p.end()

    def set_opacity(self, whole: float, content: float) -> None:
        """The toast's opacity, and its text's on top of that."""
        whole = max(0.0, min(1.0, whole))
        content = max(0.0, min(1.0, content))
        if whole != self._opacity:
            self._opacity = whole
            self._card_fade.setOpacity(whole)
            self.update()
        self._content_opacity = content
        self._content_fade.setOpacity(whole * content)
        # EFFECTS OFF AT FULL OPACITY. An opacity effect left on a widget that is moved
        # and resized can keep a stale, EMPTY picture of it and paint that instead:
        # reported as a toast left blank (card and x only) after another was closed --
        # every toast moving up a place drew no text, at opacity 1.0 throughout. Off,
        # the widget paints itself; on only while it is actually fading.
        self._card_fade.setEnabled(whole < 1.0)
        self._content_fade.setEnabled(whole * content < 1.0)

    # -- the stack ---------------------------------------------------------- #

    def move_to_depth(self, depth: float) -> None:
        """Slip to ``depth`` places behind the front -- retargeted, never queued, so
        a second arrival mid-slide carries on from wherever this one is."""
        if depth == self._depth_target and self._depth_anim.state():
            return
        self._depth_target = depth
        self._depth_anim.stop()
        self._depth_anim.setStartValue(float(self._depth))
        self._depth_anim.setEndValue(float(depth))
        self._depth_anim.start()

    def _on_depth(self, value) -> None:
        self._depth = float(value)
        if self._toaster is not None:
            self._toaster.relayout()

    def hold(self) -> None:
        """Stop the clock (the stack is hovered, or this toast is out of view)."""
        if self._life.isActive():
            self._life.stop()
            self._held = True

    def release(self) -> None:
        """Start the clock again -- in FULL, not resumed: the reader looked away and
        back, so the message gets its whole life again (Sonner's behaviour)."""
        if self._dismissing or self._duration is None or self._life.isActive():
            return
        if self._held or not getattr(self, "_started", False):
            self._held = False
            self._started = True
            self._life.start(self._duration)

    # -- life -------------------------------------------------------------- #

    def on_action(self, fn):
        self._on_action = fn
        return self

    def on_close(self, fn):
        """Called when the reader dismisses it, or when the clock does.

        A prompt reads this as *keep what is running* -- see the module
        docstring on why one action button is not enough.
        """
        self._on_close = fn
        return self

    def enter(self) -> None:
        """Slide up into place. The clock is the `Toaster`'s to start: it runs only
        while this toast is in view and the stack is not hovered."""
        self._slide.stop()
        self._slide.setStartValue(self._at)
        self._slide.setEndValue(1.0)
        self._slide.start()

    def dismiss(self, *, instant: bool = False) -> None:
        """Slide back down and fade. Idempotent -- the clock and a click race.

        The race is real: the reader can press `Proceed` on the frame the clock
        runs out, and dismissing twice would emit `gone` twice and take two
        toasts out of a stack of one.
        """
        if self._dismissing:
            return
        self._dismissing = True
        self._life.stop()
        if self._toaster is not None:
            # The others close up behind the one leaving.
            self._toaster.restack()
        if instant:
            self._at = 0.0
            self.gone.emit()
            return
        self._slide.stop()
        self._slide.setStartValue(self._at)
        self._slide.setEndValue(0.0)
        self._slide.start()

    @property
    def dismissing(self) -> bool:
        return self._dismissing

    def _acted(self) -> None:
        fn = self._on_action
        # Cleared BEFORE the callback runs: `Proceed` starts a job, which posts
        # another toast, which re-lays the stack -- and a second press arriving
        # inside that window would replace a job that was already replaced.
        self._on_action = self._on_close = None
        self.dismiss()
        if fn is not None:
            fn()

    def _closed(self) -> None:
        fn = self._on_close
        self._on_action = self._on_close = None
        self.dismiss()
        if fn is not None:
            fn()

    def _on_slide(self, value) -> None:
        self._at = float(value)
        if self._toaster is not None:
            self._toaster.relayout()

    def _slide_done(self) -> None:
        if self._at <= 0.0:
            self.gone.emit()

    # -- hovering the stack expands it ------------------------------------ #

    def enterEvent(self, e):
        super().enterEvent(e)
        if self._toaster is not None:
            self._toaster.hover(True)

    def leaveEvent(self, e):
        super().leaveEvent(e)
        if self._toaster is not None:
            self._toaster.hover(False)


class _Close(QWidget):
    """Sonner's dismiss button: a small ``x`` in ``text-muted-foreground``.

    ALWAYS drawn, where Sonner reveals it on hover. The prompt's close is not a
    tidying-up affordance -- it is one of the two answers to the question, and
    an answer you have to discover by hovering has not been offered.
    """

    clicked = Signal()

    SIZE = rem(1.25)

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.setFixedSize(self.SIZE, self.SIZE)
        self.setCursor(Qt.PointingHandCursor)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.glyph = icon("x", rem(0.875), t.muted_foreground, stroke=2.0)
        lay.addWidget(self.glyph, alignment=Qt.AlignCenter)

    def enterEvent(self, e):
        super().enterEvent(e)
        self.glyph.colour = self.t.foreground
        self.glyph.update()

    def leaveEvent(self, e):
        super().leaveEvent(e)
        self.glyph.colour = self.t.muted_foreground
        self.glyph.update()

    def mouseReleaseEvent(self, e):
        if (e.button() == Qt.LeftButton
                and self.rect().contains(e.position().toPoint())):
            self.clicked.emit()


class Toaster(QObject):
    """The stack, bottom-right of ``host`` -- Sonner's, collapsed and expanded.

    Newest at the FRONT (bottom), older ones behind it. **Collapsed**, the front
    toast is drawn whole and each older one ``n`` places back sits ``n * GAP``
    higher, ``1 - SCALE_STEP * n`` the size, at the front toast's height, with its
    text faded out -- an edge peeking above. **Hovering** expands the stack: every
    toast at full size, one above the other with ``GAP`` between, text visible and
    every clock held, so a buried prompt's `Proceed` can be reached.

    **Placement is recomputed, never animated per toast.** Each toast animates only
    its own floats -- ``_at`` (arriving or leaving) and ``_depth`` (how far back) --
    and the stack's ``_expand``; every frame `relayout` turns them into geometry. So
    an arrival, an exit, the shuffle behind it and the hover are one expression.

    **Nothing is retired.** Past ``VISIBLE`` a toast is kept, invisible and inert,
    and comes back into view as those in front leave (Sonner's ``data-visible``).
    Dismissing the oldest instead could silently drop an open prompt whose answer
    the window was still waiting for.
    """

    def __init__(self, t, host: QWidget, *, right_inset=None):
        super().__init__(host)
        self.t = t
        self.host = host
        #: A callable giving the width at the host's right edge the stack keeps clear (the workspace's tool rail,
        #: whose Help, Options and theme buttons a toast covered: 4 Nov 2026, rail review); None keeps none.
        self.right_inset = right_inset
        self.toasts: list = []
        self._expand = 0.0
        self._hovered = False
        self._expand_anim = QVariantAnimation(self)
        self._expand_anim.setDuration(MOTION_MS)
        self._expand_anim.setEasingCurve(motion_curve())
        self._expand_anim.valueChanged.connect(self._on_expand)
        host.installEventFilter(self)

    def post(self, title: str, *, description: str | None = None,
             action: str | None = None,
             duration: int | None = DURATION_MS) -> Toast:
        """Put one up, at the front. Returns it, so a prompt can be withdrawn later."""
        toast = Toast(self.t, title, description=description, action=action,
                      duration=duration, parent=self.host)
        toast._toaster = self
        toast.gone.connect(lambda w=toast: self._drop(w))
        toast.fit()
        self.toasts.append(toast)
        toast.show()
        toast.raise_()
        # MEASURED AGAIN once shown: before the first show the button and labels
        # are not yet polished, so the text column is measured narrower than it
        # ends up and the card came out a line too tall.
        toast.fit()
        self.restack()
        toast.enter()
        return toast

    # -- the stack ----------------------------------------------------------- #

    def _standing(self) -> list:
        """Toasts still in the stack, NEWEST FIRST (depth 0 first)."""
        return [w for w in reversed(self.toasts) if not w.dismissing]

    def restack(self) -> None:
        """Send every standing toast to its place: the newest to the front, each
        older one a place further back. A leaving toast keeps the place it had."""
        for depth, toast in enumerate(self._standing()):
            toast.move_to_depth(float(depth))
        self._clocks()
        self.relayout()

    def _clocks(self) -> None:
        """A clock runs only while its toast is in view and the stack is not
        hovered; anything else holds it."""
        for depth, toast in enumerate(self._standing()):
            if self._hovered or depth >= VISIBLE:
                toast.hold()
            else:
                toast.release()

    def hover(self, entering: bool) -> None:
        if entering:
            self._set_hovered(True)
        else:
            # Checked a beat later: moving from one card to the next leaves one
            # and enters the other, and the stack must not collapse in between.
            QTimer.singleShot(0, self, self._check_left)       # dropped with the toaster

    def _check_left(self) -> None:
        # THE BEAT LATER CAN OUTLIVE A TOAST: a window closing deletes its toasts between
        # the leave and this check ("Internal C++ object (Toast) already deleted", raised in
        # four tests' teardown, 15 Oct 2026). A deleted toast is not under the pointer.
        def under(w) -> bool:
            try:
                return w.isVisible() and w.underMouse()
            except RuntimeError:
                return False

        if not any(under(w) for w in list(self.toasts)):
            self._set_hovered(False)

    def _set_hovered(self, on: bool) -> None:
        if on == self._hovered:
            return
        self._hovered = on
        self._expand_anim.stop()
        self._expand_anim.setStartValue(float(self._expand))
        self._expand_anim.setEndValue(1.0 if on else 0.0)
        self._expand_anim.start()
        self._clocks()

    def _on_expand(self, value) -> None:
        self._expand = float(value)
        self.relayout()

    def relayout(self) -> None:
        """Seat every toast from its floats -- see the class docstring.

        Collapsed: bottom-centre anchored, ``GAP * depth`` higher, scaled
        ``1 - SCALE_STEP * depth``, at the front toast's height. Expanded: full size,
        stacked upward with ``GAP``. The two are blended by ``_expand``; ``_at``
        slides a toast in from (or out to) below and fades it.
        """
        standing = self._standing()
        front = standing[0] if standing else (self.toasts[-1] if self.toasts else None)
        if front is None:
            return
        front_h = front.natural_h
        e = self._expand
        base = self.host.height() - OFFSET
        inset = max(0, int(self.right_inset())) if self.right_inset is not None else 0
        centre = self.host.width() - inset - OFFSET - TOAST_W / 2.0
        # Expanded offsets, newest first, each toast weighted by how far it is in.
        stacked = {}
        acc = 0.0
        for toast in sorted(self.toasts, key=lambda w: w._depth):
            stacked[toast] = acc
            acc += (toast.natural_h + GAP) * toast._at
        for toast in self.toasts:
            d = max(0.0, toast._depth)
            back = min(d, 1.0)
            s = 1.0 - SCALE_STEP * d
            # collapsed
            h_c = _lerp(toast.natural_h, front_h, back) * s
            bottom_c = base - GAP * d
            w_c = TOAST_W * s
            # expanded
            h_e = toast.natural_h
            bottom_e = base - stacked[toast]
            w = _lerp(w_c, TOAST_W, e)
            h = _lerp(h_c, h_e, e)
            bottom = _lerp(bottom_c, bottom_e, e)
            bottom += (1.0 - toast._at) * (h + OFFSET)
            room = toast.room
            toast.setGeometry(round(centre - w / 2.0) - room,
                              round(bottom - h) - room,
                              round(w) + 2 * room, round(h) + 2 * room)
            in_view = max(0.0, min(1.0, VISIBLE - d))
            toast.set_opacity(max(0.0, min(1.0, toast._at)) * in_view,
                              1.0 - back * (1.0 - e))
            inert = in_view < 0.5 or (d >= 0.5 and e < 0.5)
            toast.setAttribute(Qt.WA_TransparentForMouseEvents, inert)

    def dismiss_all(self) -> None:
        for toast in list(self.toasts):
            toast.dismiss()

    def _drop(self, toast: Toast) -> None:
        if toast in self.toasts:
            self.toasts.remove(toast)
        toast.hide()
        toast.deleteLater()
        if not self.toasts:
            self._set_hovered(False)
        self.restack()

    def eventFilter(self, obj, event):
        if obj is self.host and event.type() == QEvent.Resize:
            self.relayout()
        return False
