"""shadcn's `Dialog`, as an IN-WINDOW overlay -- never a `QDialog` (an OS
window: its own title bar, its own Cmd+Tab entry, an OS-drawn drop shadow
standing in for the registry's `shadow-lg`), which is a real, visible
difference from what the shadcn component actually IS in a browser: a
translucent backdrop drawn OVER the existing page (`DialogOverlay:
bg-black/50`), with the card floating on top of it -- never a second window.

``OverlayDialog`` is the shared base: a `ShadowHost` sized to cover a host
window's own content, holding one centred child (``self._card``, a bare
`Surface` -- `Card` is not used, it is `bg-card`/`shadow-sm` where this needs
`bg-background`/`shadow-lg`) built from the registry line
``DialogContent: rounded-lg border bg-background p-6 shadow-lg gap-4``.
Subclasses build their own header/body/footer into ``self.card_lay``.

**THE BACKDROP IS BLURRED, HAND-PAINTED, NOT A STYLESHEET FILL.** The base
registry ships flat `bg-black/50`, no blur -- what `paintEvent` draws is the
popular `backdrop-blur` addition on top of it. **The tint itself is NOT that
literal `/50`** -- `_OVERLAY_ALPHA` is 27, not 128, MEASURED against a real
render of the registry's own Dialog rather than trusted from the class name:
pixel-sampled backdrop luminance there ~225/255 on a page whose own
background is already near-white; this app's own background samples the
same way (~252/255 pre-tint), so matching that target back-solves to alpha
~0.107, not 0.50. There is no Qt per-widget equivalent of the browser's
continuously-recomposited `backdrop-filter`, so `_snapshot_backdrop` grabs
``self._content`` (never the parent window itself, which would include this
overlay's own last-painted frame once shown once) and runs it through
`QGraphicsBlurEffect` ONCE, at `open` time, before the content is disabled.
That one-time-ness is not a compromise: `open` disables the very thing being
snapshotted in the same breath, so nothing behind this overlay changes while
it is up regardless of which technique drew it. A parent resize while the
overlay is visible re-snapshots (the old one would otherwise be the wrong
size) -- see `eventFilter`.

**THE BLOCKING IS A DISABLE, NOT JUST A COVERING.** `open` disables
``content`` for as long as the overlay is up, which is what actually stops
input to the rest of the app -- a disabled widget cannot hold focus, so
whatever had it loses it the instant `content` is disabled, and Qt will not
deliver clicks to it either. The backdrop covering the window is the VISUAL
half of that; the disable is what would still hold even if some future
widget somehow sat outside the backdrop's paint area. This replaces
`Qt.ApplicationModal` entirely, and does something that window-modality flag
could not: there is no separate top-level window left to give it to.

**`content` IS EXPLICIT, NEVER INFERRED.** An earlier revision of this class
(`ComputeDialog`, before the two dialogs here shared a base) read
`self.parentWidget().centralWidget()` -- a `QMainWindow`-only API, and this
overlay now serves `OnboardingWindow` too, a plain `QWidget` with no such
method. Guessing via `hasattr` would have been the wrong fix: a silently
guessed target is exactly the class of defect this project's own conventions
refuse elsewhere (an unmeasured, unstated default standing in for a real
answer). `content` is a REQUIRED keyword -- the widget to blur-snapshot and
disable -- decided by the caller, who knows what "the rest of the window"
actually is for its own host.

**ESCAPE, THE ONE THING WORTH BEING EXPLICIT ABOUT ANYWAY.** A plain
`QWidget` has no built-in "Escape closes me" the way `QDialog` did, so
`keyPressEvent` swallowing it is no longer undoing a default -- it is
stating, in code, that Escape does nothing here, so a future reader does not
have to go prove the negative themselves. Outside-click needs no equivalent
guard: with `content` disabled, a click landing anywhere in the window
either hits the backdrop (absorbed, no handler) or the card (routed to
whichever control is under it) -- there is no path left for it to reach
anything this overlay is supposed to be blocking. The only proactive exits
are each subclass's own buttons and the corner `X` (`CloseButton`,
`DialogClose`).

**THE CARD FADES AND ZOOMS IN AND OUT, `_ANIM_MS` = 200 LONG -- READ OFF THE
REAL REGISTRY, NOT ASSUMED.** `dialog.tsx`'s `DialogOverlay` carries
`fade-in-0`/`fade-out-0`, and `DialogContent` carries those TWO PLUS
`zoom-in-95`/`zoom-out-95` at `duration-200` -- so the backdrop only ever
fades, the card fades AND scales between 95% and 100%. The easing is
Tailwind's `animate-in`/`animate-out` default, plain CSS `ease`
(cubic-bezier(0.25, 0.1, 0.25, 1)) -- the SAME curve `menus.Menu`'s own
intro already uses for the identical utility, reused rather than
re-derived; `primitives.tw_ease()` is a DIFFERENT Tailwind default
(`transition`, not `animate-in`) and would be the wrong curve here.

**``RegroupDialog``: it previews, then applies.** ``Project.regroup``
deliberately returns the proposal WITHOUT applying it, so the dialog can
show you 5·5·5·5·5·3 before you agree to it. Regrouping is the one action on
this screen that discards an arrangement you may have spent time on, and an
undo you have to think about is worse than a preview you cannot miss.

**The extreme cases are the ones worth designing for.** 6 x 52 over 28
series puts everything in the first group and leaves five empty; 20 x 1
makes twenty groups of one. Both are reachable in two clicks, so both have
to look deliberate rather than broken: empty groups are SHOWN as empty
rather than skipped, and the preview scrolls at a fixed height rather than
growing the dialog past the bottom of the screen.
"""

from __future__ import annotations

from qtpy.QtCore import (
    QEasingCurve,
    QEvent,
    QPointF,
    QRectF,
    QSize,
    Qt,
    QVariantAnimation,
    Signal,
)
from qtpy.QtGui import QColor, QPainter, QPainterPath, QPixmap
from qtpy.QtWidgets import (
    QGraphicsBlurEffect,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QHBoxLayout,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..theme import typography as ty
from .controls import Button, NumberField, Segmented
from .primitives import (
    Badge,
    HoverFill,
    ShadowHost,
    Surface,
    col,
    day_badge,
    draw_shadow,
    icon,
    label,
    own_surfaces,
    paint_shadows,
    row,
    shadow_room,
)
from .primitives import discard

__all__ = ["CloseButton", "OverlayDialog", "RegroupDialog"]


class CloseButton(Surface):
    """The corner `X` -- shadcn's `DialogClose`. Same weight of action as
    a dialog's own Cancel/Apply, deliberately: pressing it is a proactive
    exit, unlike the Escape an `OverlayDialog` refuses and the outside-click
    it structurally cannot receive -- see the module docstring.
    """

    clicked = Signal()

    def __init__(self, t, parent=None):
        super().__init__(t, radius=t.radius_sm, shadowed=False, parent=parent)
        self.setObjectName(f"dlgclose{id(self)}")
        self.setFixedSize(28, 28)
        self.setCursor(Qt.PointingHandCursor)
        HoverFill(self, lambda c: (f"QWidget#{self.objectName()} "
                                   f"{{ background:{c}; border:none;"
                                   f" border-radius:{t.radius_sm:g}px; }}"),
                  t.background, t.accent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(icon("x", 16, t.muted_foreground, stroke=2.0),
                      alignment=Qt.AlignCenter)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self.rect().contains(e.pos()):
            self.clicked.emit()


class OverlayDialog(ShadowHost):
    """shadcn's `Dialog`, as an in-window overlay -- see the module
    docstring. A backdrop (`self`, hand-painted -- see `paintEvent`)
    covering ``parent``'s entire client area, holding one centred child
    (`self._card`) that subclasses build their own header/body/footer into
    via `self.card_lay`.
    """

    #: NOT the registry's literal `bg-black/50` -- see the module docstring.
    _OVERLAY_ALPHA = 27

    #: The base registry ships no blur at all -- this approximates the
    #: popular `backdrop-blur` addition on top of `bg-black/50`, tuned by
    #: eye against a real run of this dialog, not derived from anything.
    #: In LOGICAL pixels; `_snapshot_backdrop` scales it by the grabbed
    #: pixmap's own device pixel ratio, so it reads the same strength on a
    #: HiDPI screen as on a plain one.
    _BLUR_RADIUS = 18.0

    #: shadcn's own `DialogContent`: `duration-200`, `fade-in-0`/
    #: `fade-out-0`, `zoom-in-95`/`zoom-out-95` -- see the module docstring.
    _ANIM_MS = 200
    _ZOOM_FROM = 0.95

    #: Fired once `_finish_close` has fully re-enabled `content` and
    #: hidden the overlay -- a caller constructing one instance per open
    #: (`RegroupDialog`'s own call site) connects this to `deleteLater` so
    #: repeated opens don't accumulate hidden instances still holding a
    #: live `eventFilter` on the window.
    closed = Signal()
    #: Fired when the intro has LANDED (the card at full size, the real widget shown) -- once
    #: per open, never on a close. Work that would compete with the intro for the interpreter
    #: (`ExportDialog`'s preview, a matplotlib build on a worker that held the GIL and dropped
    #: half the zoom's frames, measured) starts here, not when the dialog is built.
    opened = Signal()

    def __init__(self, t, content=None, *, parent=None):
        super().__init__(t, parent)
        self.t = t
        #: The widget to blur-snapshot and disable while this overlay is
        #: open -- see the module docstring's "`content` IS EXPLICIT".
        self._content_widget = content
        #: Set by a subclass's own "this was a real action" exit (Run,
        #: Apply), read only by `_dismiss` -- the one flag that tells it
        #: whether this close already reported an action, or needs
        #: `_on_dismiss_without_action` first.
        self._acted = False
        #: Disabled for as long as this overlay is open (`open`) and
        #: re-enabled on `_finish_close`. `None` when nothing is currently
        #: disabled -- no `content` given (standalone / test construction)
        #: or `open` was never called.
        self._content: QWidget | None = None
        #: A blurred snapshot of `self._content`, taken by
        #: `_snapshot_backdrop` -- `None` until `open` has run at least
        #: once (the standalone / test construction never sets it, and
        #: `paintEvent` falls back to a flat tint when it is absent).
        self._backdrop: QPixmap | None = None
        # So Escape reaches `keyPressEvent` the instant the overlay opens,
        # before any child has been clicked into focus -- see `open`.
        self.setFocusPolicy(Qt.StrongFocus)

        #: The card's own open/close animation -- `_start_dialog_anim`.
        #: `_anim_snapshot` is `None` whenever the card is settled (open or
        #: fully closed); non-`None` for the duration of an intro or
        #: outro, which is also what `paintEvent` reads to decide whether
        #: to draw the scaled snapshot instead of `self._card`'s own
        #: shadow. `_anim_value` is the CURRENT 0..1 progress, read by
        #: `paintEvent` outside the `valueChanged` slot itself so a stray
        #: repaint between ticks still has a real number to use.
        #: `_anim_closing` says which end `_on_dialog_anim_finished`
        #: should land on.
        self._anim_snapshot: QPixmap | None = None
        #: The card's shadow, pre-rendered once per intro/outro by
        #: `_render_shadow_pixmap` -- see that method's own docstring.
        #: `None` on the same schedule as `_anim_snapshot`.
        self._anim_shadow: QPixmap | None = None
        self._anim_value: float = 1.0
        self._anim_closing = False
        self._anim = QVariantAnimation(self)
        curve = QEasingCurve(QEasingCurve.BezierSpline)
        curve.addCubicBezierSegment(QPointF(0.25, 0.1), QPointF(0.25, 1.0),
                                    QPointF(1.0, 1.0))
        self._anim.setEasingCurve(curve)
        self._anim.valueChanged.connect(self._on_dialog_anim)
        self._anim.finished.connect(self._on_dialog_anim_finished)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        # `radius_xl` (14px), not the `radius_lg` (10px) the registry's own
        # `rounded-lg` class name implies -- MEASURED against a real render
        # of `ui.shadcn.com`'s own Dialog (pixel-scanned corner: ~13px
        # there against ~11px here at the same screenshot scale).
        self._card = Surface(t, radius=t.radius_xl, shadowed=True,
                             shadow=t.shadow_lg)
        self._card.setObjectName(f"overlaycard{id(self)}")
        self._card.setStyleSheet(
            f"QFrame#overlaycard{id(self)} {{ background:{t.background};"
            f" border:1px solid {t.border};"
            f" border-radius:{t.radius_xl:g}px; }}")
        self.card_lay = QVBoxLayout(self._card)
        self.card_lay.setContentsMargins(24, 20, 24, 20)
        self.card_lay.setSpacing(16)

        outer.addWidget(self._card, alignment=Qt.AlignCenter)

        # Tracks the parent's own size so a resize while the overlay is open
        # keeps it covering the whole window -- see `eventFilter`. Harmless
        # to install even while hidden; `open` is what actually shows it.
        if parent is not None:
            parent.installEventFilter(self)

        self.hide()

    # -- opening / closing ---------------------------------------------- #

    def open(self) -> None:
        """Cover `self.parentWidget()`'s entire client area and disable
        `self._content_widget` -- see the module docstring's "THE BLOCKING
        IS A DISABLE, NOT JUST A COVERING". With no parent, this just
        shows the overlay as its own top-level widget (Qt's ordinary
        rule), which is enough for headless tests -- `self._backdrop`
        stays `None` there, so `paintEvent` falls back to the flat tint
        alone."""
        window = self.parentWidget()
        if window is not None:
            self.setGeometry(window.rect())
        content = self._content_widget
        if content is not None:
            # BEFORE disabling: a disabled widget can still be grabbed,
            # but the snapshot belongs to the moment the background was
            # last interactive, not to whatever `setEnabled(False)` might
            # repaint it as.
            self._content = content
            self._snapshot_backdrop()
            self._content.setEnabled(False)
        self.show()
        self.raise_()
        self.setFocus()
        self._start_dialog_anim(opening=True)

    def _activate_content_layout(self) -> None:
        """Hook: a subclass whose BODY resizes asynchronously (a height
        slide, say) overrides this to force that content's own layout to
        settle before `_start_dialog_anim`/`_resync_after_resize` grab or
        recentre `self._card` -- see `_start_dialog_anim`'s own doc for why
        that matters. A no-op here: most dialogs need nothing extra beyond
        the card's own layout, which `_start_dialog_anim` activates
        regardless."""

    def _on_dismiss_without_action(self) -> None:
        """Hook: called once, from `_dismiss`, only when this close is NOT
        already reporting a real action (`self._acted` is still `False`) --
        a subclass overrides it to emit its own "nothing happened" signal
        (`cancelled`, typically)."""

    def _start_dialog_anim(self, *, opening: bool) -> None:
        """Play the card's entrance or exit -- fade + zoom between
        `_ZOOM_FROM` and 1.0, `_ANIM_MS` long. `self._card` is hidden for
        the whole run and a fresh snapshot of it (`show()` first, in case
        the LAST run through here left it hidden -- opening after a prior
        close, or an outro landing after an interrupted intro) is drawn
        scaled and faded in `paintEvent` instead: nothing here has to
        fight `outer`'s own `Qt.AlignCenter` placement of `self._card`,
        or reproduce it, since the real widget's geometry is never
        touched.

        Layout is force-activated bottom-up before the grab, the same
        three calls `_resync_after_resize` already needs and for the same
        reason: a child's new size reaches its parent's layout through a
        POSTED event, not synchronously, so `self._card` can still be
        reporting a stale geometry the instant after `show()`.

        Blocks signals around `stop()` / `setDuration()` / `setStartValue
        ()` / `setEndValue()`: `stop()` on any prior run -- finished OR
        interrupted -- leaves `currentTime` sitting at that run's own end,
        and an unguarded setter recomputes and EMITS at that stale time
        against the values just assigned.
        """
        self._activate_content_layout()
        self._card.layout().activate()
        self.layout().activate()
        self._card.show()
        self._anim_snapshot = self._card.grab()
        # The shadow, rendered ONCE here rather than repainted fresh every
        # animation frame -- see `_render_shadow_pixmap`'s own docstring
        # for why that mattered.
        self._anim_shadow = self._render_shadow_pixmap()
        self._card.hide()
        self._anim_closing = not opening
        start, end = (0.0, 1.0) if opening else (1.0, 0.0)
        self._anim_value = start
        blocked = self._anim.blockSignals(True)
        self._anim.stop()
        self._anim.setDuration(self._ANIM_MS)
        self._anim.setStartValue(start)
        self._anim.setEndValue(end)
        self._anim.blockSignals(blocked)
        self._anim.start()
        self.update()

    def _render_shadow_pixmap(self) -> QPixmap:
        """The card's shadow, rendered ONCE per intro/outro rather than
        repainted fresh on every animation frame.

        `draw_shadow` draws several concentric rounded rects PER layer of
        the spec (`shadow_lg` here, the heaviest rung on the ladder) --
        real per-call cost, the same one that function's own docstring
        already quotes elsewhere ("worth 8 ms a frame on the sidebar's
        collapse"). The card's own scaling already gets this for free
        (`self._card.grab()` runs once, `paintEvent` just scales the
        result), and the first version of the shadow fix called
        `draw_shadow` straight from `paintEvent` instead -- MEASURED as
        the cause, reported directly ("the animation is laggy"), because
        that ran the full multi-ring paint on every one of the ~12 frames
        an intro plays. This is the same fix applied to the same problem:
        pay the real cost once, then just `drawPixmap` a cached result.

        Sized with `shadow_room` on every edge so the blur has room to
        extend past the card's own rect without being clipped by the
        pixmap's bounds -- the same margin `Panel`'s own popover already
        reserves around itself for an identical reason.
        """
        room = shadow_room(self.t, self._card.shadow)
        card_rect = self._card.rect()
        pm = QPixmap(QSize(card_rect.width() + 2 * room,
                           card_rect.height() + 2 * room))
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing, True)
        local = QRectF(room, room, card_rect.width(), card_rect.height())
        draw_shadow(p, local, self._card.radius, self.t, self._card.shadow)
        p.end()
        return pm

    def _on_dialog_anim(self, v) -> None:
        self._anim_value = float(v)
        self.update()

    def _on_dialog_anim_finished(self) -> None:
        """The intro landed at full size/opacity (show the real card,
        drop the snapshot) or the outro landed at nothing (finish the
        close this animation was standing in for -- see `_dismiss`)."""
        self._anim_snapshot = None
        self._anim_shadow = None
        if self._anim_closing:
            self._finish_close()
        else:
            self._card.show()
        self.update()
        if not self._anim_closing:
            self.opened.emit()

    def _snapshot_backdrop(self) -> None:
        """Grab `self._content` -- never `self.parentWidget()` itself,
        which would include this overlay's own last-painted frame once it
        has been shown once -- and blur it, ONCE, for the backdrop.

        A snapshot, not a live `backdrop-filter`: Qt has no per-widget
        equivalent of the browser's continuously-recomposited blur. That is
        not a compromise here specifically -- `open` disables the very
        widget being snapshotted in the same breath, so nothing behind this
        overlay changes while it is up regardless of which technique drew
        it. `eventFilter` re-snapshots on a parent resize while this overlay
        is visible, since the OLD snapshot would otherwise be the wrong
        size.
        """
        content = self._content
        if content is None:
            return
        self._backdrop = self._blur(content.grab())

    def _blur(self, pixmap: QPixmap) -> QPixmap:
        """`pixmap`, Gaussian-blurred, via the one Qt has: render it through
        a `QGraphicsBlurEffect` in a throwaway scene. `_BLUR_RADIUS` is
        scaled by the pixmap's own device pixel ratio -- `grab()` on a
        HiDPI screen returns a pixmap with MORE raw pixels per logical one,
        and the effect's radius is in raw pixels, so an unscaled radius
        would read as half as strong on a Retina screen as on a plain one.
        """
        dpr = pixmap.devicePixelRatio()
        full = pixmap.size()
        # AT A QUARTER OF THE RESOLUTION, then scaled back: an 18 px blur keeps no detail a
        # quarter-size image lacks, and the full Retina snapshot (2940 x 1662) cost 31 ms on
        # every open, before anything showed (measured); a quarter costs about 4 ms
        f = self._BLUR_DOWNSCALE
        small = pixmap.toImage().scaled(max(1, full.width() // f), max(1, full.height() // f),
                                        Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        small_pm = QPixmap.fromImage(small)
        small_pm.setDevicePixelRatio(1.0)
        blurred = self._blur_raw(small_pm, self._BLUR_RADIUS * dpr / f)
        result = QPixmap.fromImage(blurred.toImage().scaled(
            full, Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
        result.setDevicePixelRatio(dpr)
        return result

    #: How much smaller `_blur` blurs than the snapshot it is given.
    _BLUR_DOWNSCALE = 4

    @staticmethod
    def _blur_raw(pixmap: QPixmap, radius: float) -> QPixmap:
        """``pixmap`` (device pixel ratio 1) blurred by ``radius`` raw pixels, through the one
        Gaussian blur Qt has: a `QGraphicsBlurEffect` in a throwaway scene."""
        scene = QGraphicsScene()
        item = QGraphicsPixmapItem(pixmap)
        effect = QGraphicsBlurEffect()
        effect.setBlurRadius(radius)
        effect.setBlurHints(QGraphicsBlurEffect.QualityHint)
        item.setGraphicsEffect(effect)
        scene.addItem(item)
        result = QPixmap(pixmap.size())
        result.fill(Qt.transparent)
        painter = QPainter(result)
        painter.setRenderHint(QPainter.Antialiasing, True)
        rect = QRectF(0, 0, pixmap.width(), pixmap.height())
        scene.render(painter, rect, rect)
        painter.end()
        return result

    def _resync_after_resize(self) -> None:
        """A subclass whose BODY resizes asynchronously (a height slide)
        connects that resize signal here -- one tick of catch-up.

        `self._card` shrinking/recentring does not, on its own, invalidate
        the backdrop region it just vacated, so this repaints it -- but
        MEASURED, not assumed, `update()` alone was not enough: `self.
        _card`'s own geometry is NOT resynchronised the moment the body's
        height changes underneath it. Each level's layout reacts to its
        child's new size hint through a POSTED `LayoutRequest` event, not
        synchronously, and that is true at EVERY level of this nesting, not
        just the outermost. So all three levels are force-activated here,
        bottom-up, in the order the geometry actually has to settle in,
        before `repaint()` (synchronous, not `update()`'s deferred and
        possibly-coalesced one) draws the backdrop against `self._card`'s
        true CURRENT geometry rather than last frame's.
        """
        self._activate_content_layout()
        self._card.layout().activate()
        self.layout().activate()
        self.repaint()

    def eventFilter(self, obj, event):
        if obj is self.parentWidget() and event.type() == QEvent.Resize:
            self.setGeometry(obj.rect())
            if self.isVisible() and self._content is not None:
                # The old snapshot is now the wrong size -- re-take it
                # rather than scale it, so a resize while this overlay is
                # open never shows a stretched backdrop.
                self._snapshot_backdrop()
                self.update()
        return super().eventFilter(obj, event)

    def paintEvent(self, e) -> None:
        """The backdrop (blurred snapshot, if `open` took one, under the
        `bg-black/50` tint either way -- see `_snapshot_backdrop`), then
        EITHER `self._card`'s own shadow (settled state, the same
        two-step `ShadowHost.paintEvent` does for any of its children --
        reimplemented rather than inherited, since the backdrop is
        hand-painted now and `super().paintEvent` would additionally try
        to fill from a stylesheet this class no longer sets) OR, for the
        duration of an intro/outro, a scaled and faded snapshot of it in
        place of the real (hidden) widget -- see `_start_dialog_anim`.
        The backdrop itself fades in step with the card, matching
        `DialogOverlay`'s own `fade-in-0`/`fade-out-0`.
        """
        animating = self._anim_snapshot is not None
        opacity = self._anim_value if animating else 1.0

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        if self._backdrop is not None:
            p.setOpacity(opacity)
            p.drawPixmap(0, 0, self._backdrop)
            p.setOpacity(1.0)
        p.fillRect(self.rect(), QColor(0, 0, 0, round(self._OVERLAY_ALPHA * opacity)))
        p.end()

        if animating:
            # `drawPixmap(QRectF, QPixmap, QRectF)` scales natively -- no
            # manual QTransform needed. The snapshot was taken at the
            # card's own FINAL geometry, so shrinking the TARGET rect
            # toward `_ZOOM_FROM` around that same centre is the whole
            # zoom.
            #
            # FLOATING POINT THE WHOLE WAY THROUGH -- no `round()`/`//`
            # anywhere in this block, which is a CORRECTION: an earlier
            # version rounded `w`/`h` to `int` and built an integer `QRect`
            # from `//`-truncated half-widths, so every frame independently
            # snapped four numbers (w, h, x, y) to whole pixels from FRESH
            # floats rather than tracking one continuously-growing box --
            # and, once the shadow pixmap was added, a FIFTH independent
            # rounding (`QRectF.adjusted(...).toRect()`) on top of that.
            # Four-or-five separate roundings of numbers that are supposed
            # to move in lockstep do not always round the same way frame
            # to frame (this file's own comment on the intro's shape says
            # "most movement happens early" -- so near the END, the TRUE
            # per-frame change is often under 1 px, which is exactly where
            # independent whole-pixel snapping stops tracking the smooth
            # curve and starts reading as noise instead). Reported
            # directly, against a real dialog, as "squiggles ... in the
            # last few frames" -- precisely where this predicts it. Qt's
            # own `SmoothPixmapTransform` rasteriser handles the actual
            # device-pixel snap, ONCE, consistently, if it is simply GIVEN
            # the floating-point target instead of a pre-quantised one.
            scale = self._ZOOM_FROM + (1.0 - self._ZOOM_FROM) * self._anim_value
            final = QRectF(self._card.geometry())
            w, h = final.width() * scale, final.height() * scale
            c = final.center()
            target = QRectF(c.x() - w / 2, c.y() - h / 2, w, h)
            p2 = QPainter(self)
            p2.setRenderHint(QPainter.Antialiasing, True)
            p2.setRenderHint(QPainter.SmoothPixmapTransform, True)
            # CLIPPED to a rounded rect -- `QWidget.grab()` does NOT keep
            # `self._card`'s own rounded corners transparent (MEASURED:
            # the grabbed pixmap's corner pixels come back opaque white,
            # `(255, 255, 255, 255)`, not alpha 0). Undimmed that reads as
            # a normal square card; faded by `opacity` it reads as a
            # square "window" showing through around the rounded one.
            # The clip path's own radius scales WITH the card, `t.radius_xl
            # * scale`, so the corner looks like a genuinely smaller
            # rounded rect at 95% rather than the full-size radius on a
            # shrunk box.
            path = QPainterPath()
            path.addRoundedRect(target, self.t.radius_xl * scale,
                                self.t.radius_xl * scale)
            # THE SHADOW, MISSING FOR THE WHOLE ANIMATION UNTIL THIS BLOCK.
            # `self._anim_snapshot` is `self._card.grab()` -- the card's
            # OWN pixels only. The shadow is painted by the ANCESTOR
            # (`paint_shadows`, below, once the real widget is showing
            # again), never by the card itself, so it was never part of
            # what got grabbed or scaled — the whole ~200ms intro/outro
            # ran with no shadow at all and it popped in abruptly on the
            # frame the animation handed back to the real widget. Reported
            # directly, against a real dialog, as "the frame before the
            # final frame doesn't have shadow".
            #
            # DRAWN FROM `self._anim_shadow`, A CACHE, NOT `draw_shadow`
            # CALLED HERE. The first version of this fix called
            # `draw_shadow` directly in this branch — correct, and
            # reported directly as laggy: that function draws several
            # concentric rounded rects PER layer of `shadow_lg`, real work
            # repeated on every one of the ~12 frames an intro plays.
            # `_render_shadow_pixmap` pays that cost ONCE, in
            # `_start_dialog_anim`; scaling and compositing a pixmap here
            # is the same cheap operation the card snapshot already does.
            # `room` undoes that pixmap's own padding (`shadow_room`) so
            # the shadow lands centred on `target` exactly as it would
            # around the real, settled card.
            room = shadow_room(self.t, self._card.shadow) * scale
            shadow_target = target.adjusted(-room, -room, room, room)
            p2.setOpacity(opacity)
            p2.drawPixmap(shadow_target, self._anim_shadow,
                          QRectF(self._anim_shadow.rect()))
            p2.setClipPath(path)
            p2.drawPixmap(target, self._anim_snapshot,
                          QRectF(self._anim_snapshot.rect()))
            p2.end()
            return

        if self._cards is None:
            self._cards = own_surfaces(self)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        if not paint_shadows(self, p, self.t, QRectF(e.rect()), self._cards):
            self._cards = None
        p.end()

    def _dismiss(self) -> None:
        """A subclass's own exit buttons (Cancel/Apply/Run, the corner
        `X`) all land here. An action signal (if any) has already fired
        BEFORE this is called -- a caller has no reason to wait on a
        cosmetic exit animation -- but the actual re-enable-and-hide
        (`_finish_close`) is deferred until the outro finishes, matching
        Radix's own `Dialog`: `content` stays blocked and the backdrop
        stays visible, fading, for the WHOLE outro, not just cosmetically
        covering it."""
        if not self._acted:
            self._on_dismiss_without_action()
        self._acted = False
        self._start_dialog_anim(opening=False)

    def _finish_close(self) -> None:
        """The outro (`_start_dialog_anim(opening=False)`) has fully
        faded and shrunk away -- re-enable whatever `open` disabled,
        actually hide the overlay, and tell whoever is listening that this
        instance is done and safe to discard (`closed`; a caller opening a
        fresh instance per click connects this to `deleteLater`)."""
        if self._content is not None:
            self._content.setEnabled(True)
            self._content = None
        self.hide()
        self.closed.emit()

    def keyPressEvent(self, e) -> None:
        """Escape does nothing -- see the module docstring's "ESCAPE, THE
        ONE THING WORTH BEING EXPLICIT ABOUT ANYWAY"."""
        if e.key() == Qt.Key_Escape:
            e.accept()
            return
        super().keyPressEvent(e)


class RegroupDialog(OverlayDialog):
    """Linear, day-blind, and it says so.

    TWO WAYS TO CUT, BOTH CONSECUTIVE AND IN FILE ORDER. "By series number"
    takes only the count and cuts where the numbering breaks
    (`Project.split`), so "3 groups" is three groups whatever their sizes --
    asked for after a project numbered 001-009 / 031-040 / 041-050, could not be
    reached by any ``3 x per``. "Fixed size" is the original ``groups x per``
    (`Project.regroup`). The first is the default because it is the one a
    count alone describes.
    """

    #: The two ways to cut, in the order the switch shows them.
    MODES = ("By series number", "Fixed size")

    #: The preview never grows past this; it scrolls instead. Twenty groups
    #: of one is a legal request, and a dialog that becomes taller than the
    #: screen to show it has no way back to its own buttons.
    PREVIEW_H = 190

    applied = Signal()
    cancelled = Signal()

    def __init__(self, t, project, fills, inks, *, content=None, parent=None):
        super().__init__(t, content, parent=parent)
        self.p = project
        self.fills, self.inks = fills, inks
        self._card.setMinimumWidth(720)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(8)
        # `lg`, not `base` -- `theme.typography`'s own ladder-audit comment
        # reserves `lg` for a MODAL's own title, measured specifically for
        # THIS dialog ("`dialogs.RegroupDialog` is the case it exists
        # for"). A different, independently-measured decision from
        # `ComputeDialog`'s title size, and not one sharing a base class
        # should silently overwrite.
        head.addWidget(label("Regroup series", ty.ui("lg", "semibold"),
                             t.foreground))
        head.addStretch(1)
        self._close_btn = CloseButton(t)
        self._close_btn.clicked.connect(self._on_cancel)
        head.addWidget(self._close_btn)
        self.card_lay.addLayout(head)

        subtitle = label(
            "Consecutive blocks, in file order. "
            "Imaging dates are not consulted.",
            ty.ui("sm"), t.muted_foreground)
        subtitle.setWordWrap(True)
        self.card_lay.addWidget(subtitle)

        self._mode = Segmented(t, list(self.MODES), current=0, size="small")
        self._mode.changed.connect(lambda _i: self._refresh())
        mode_row = QHBoxLayout()
        mode_row.setContentsMargins(0, 0, 0, 0)
        mode_row.addWidget(self._mode)
        mode_row.addStretch(1)
        self.card_lay.addLayout(mode_row)

        self._groups = NumberField(t, 1, 20, len(project.groups))
        self._per = NumberField(t, 1, 60, 5)
        self._sum = label("", ty.ui("sm", "medium"), t.foreground)
        self._sum.setWordWrap(True)
        self._sizes = label("", ty.ui("xs"), t.muted_foreground)
        self._sizes.setWordWrap(True)

        fields = QHBoxLayout()
        fields.setContentsMargins(0, 0, 0, 0)
        fields.setSpacing(20)
        cells = []
        for title, box in (("Groups", self._groups),
                           ("Series in each", self._per)):
            cell = col(label(title, ty.ui("xs", "medium"), t.muted_foreground),
                       box, spacing=6)
            cell.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            fields.addWidget(cell)
            cells.append(cell)
        # Only "Fixed size" has a size to set.
        self._per_cell = cells[1]
        summary = col(self._sum, self._sizes, spacing=4)
        summary.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        fields.addWidget(summary, 1)
        holder = QWidget()
        holder.setStyleSheet("background:transparent;")
        holder.setLayout(fields)
        holder.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.card_lay.addWidget(holder)

        self._preview = QWidget()
        self._preview.setStyleSheet("background:transparent;")
        self._preview_lay = QVBoxLayout(self._preview)
        self._preview_lay.setContentsMargins(0, 0, 0, 0)
        self._preview_lay.setSpacing(6)
        scroll = QScrollArea()
        scroll.setWidget(self._preview)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setFixedHeight(self.PREVIEW_H)
        scroll.setStyleSheet("QScrollArea { background:transparent; }")
        self.card_lay.addWidget(scroll)

        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.setSpacing(8)
        footer.addStretch(1)
        cancel_btn = Button(t, "Cancel", variant="outline")
        cancel_btn.clicked.connect(self._on_cancel)
        apply_btn = Button(t, "Apply", variant="default")
        apply_btn.clicked.connect(self._on_apply)
        footer.addWidget(cancel_btn)
        footer.addWidget(apply_btn)
        self.card_lay.addLayout(footer)

        for box in (self._groups, self._per):
            box.changed.connect(self._refresh)
        self._refresh()

    def by_series(self) -> bool:
        return self._mode.current() == 0

    def blocks(self) -> list[list[str]]:
        if self.by_series():
            return self.p.split(self._groups.value())
        return self.p.regroup(self._groups.value(), self._per.value())

    def _refresh(self) -> None:
        p = self.p
        blocks = self.blocks()
        dated = len([s for s in p.series if s.day is not None])
        self._per_cell.setVisible(not self.by_series())
        want = self._groups.value() * self._per.value()
        if self.by_series():
            self._sum.setText(
                f"{dated} series into {self._groups.value()}, cut where the "
                f"series numbering jumps or starts a new ten.")
        else:
            self._sum.setText(
                f"{self._groups.value()} × {self._per.value()} = {want}, but there "
                f"are {dated}; the last group takes what is left."
                if want != dated else
                f"{self._groups.value()} × {self._per.value()} = {dated}, "
                f"none left over.")
        filled = [b for b in blocks if b]
        empties = len(blocks) - len(filled)
        self._sizes.setText(
            " · ".join(str(len(b)) for b in blocks)
            + (f"   ({empties} would be empty)" if empties else ""))

        while self._preview_lay.count():
            item = self._preview_lay.takeAt(0)
            widget = item.widget()
            if widget is not None:
                discard(widget)
        for i, block in enumerate(blocks):
            self._preview_lay.addWidget(self._block_row(i, block))
        self._preview_lay.addStretch(1)

    def _block_row(self, i: int, block: list[str]) -> QWidget:
        """One proposed group. EMPTY ONES ARE DRAWN, not skipped.

        Skipping them made ``6 × 52`` render as a single row and look like the
        dialog had failed, when it was in fact telling the truth: five of the
        six groups would be empty. A request that produces nothing has to say
        so where the request was made.
        """
        t, p = self.t, self.p
        if not block:
            line = row(label(f"Group {i + 1}", ty.ui("sm", "medium"),
                             t.muted_foreground),
                       None,
                       label("empty", ty.ui("xs"), t.muted_foreground),
                       spacing=6, margins=(10, 6, 10, 6))
        else:
            days = sorted({p.series_by_name(n).day for n in block
                           if p.series_by_name(n).day is not None})
            tags = [day_badge(t, self.fills, self.inks, f"D{d}", d)
                    for d in days[:6]]
            if len(days) > 6:
                # Seven day tags in a preview row is a list, not a summary.
                tags.append(Badge(t, f"+{len(days) - 6}", t.muted,
                                  t.muted_foreground))
            line = row(label(f"Group {i + 1}", ty.ui("sm", "medium"),
                             t.foreground),
                       *tags, None,
                       label(f"{len(block)} series · {block[0]}–{block[-1]}",
                             ty.ui("xs"), t.muted_foreground),
                       spacing=6, margins=(10, 6, 10, 6))
        line.setObjectName(f"pv{i}")
        line.setStyleSheet(
            f"QWidget#pv{i} {{ background:{t.subtle}; border:1px solid "
            f"{t.border}; border-radius:{t.radius_sm:g}px; }}")
        line.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        return line

    def _on_apply(self) -> None:
        self._acted = True
        self.applied.emit()
        self._dismiss()

    def _on_cancel(self) -> None:
        self._dismiss()

    def _on_dismiss_without_action(self) -> None:
        self.cancelled.emit()


class NoticeDialog(OverlayDialog):
    """A title, a sentence or two and OK: what a load decided on the user's behalf (`onboarding_view.
    unsegmented_notice`). Escape or the cross closes it like OK -- there is nothing to choose."""

    acknowledged = Signal()

    def __init__(self, t, title: str, body: str, *, content=None, parent=None, extra=None):
        """``extra``: an optional ``(words, callback)`` for a second button left of OK (the saved report's
        "Report on GitHub"); pressing it calls ``callback`` and leaves the notice open."""
        super().__init__(t, content, parent=parent)
        self._card.setMinimumWidth(460)
        self._card.setMaximumWidth(560)
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(8)
        self.title_label = label(title, ty.ui("lg", "semibold"), t.foreground)
        self.title_label.setWordWrap(True)
        head.addWidget(self.title_label, 1)
        self._close_btn = CloseButton(t)
        self._close_btn.clicked.connect(self._on_ok)
        head.addWidget(self._close_btn)
        self.card_lay.addLayout(head)
        self.body_label = label(body, ty.ui("sm"), t.muted_foreground)
        self.body_label.setWordWrap(True)
        self.card_lay.addWidget(self.body_label)
        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.addStretch(1)
        self.extra_button = None
        if extra is not None:
            words, callback = extra
            self.extra_button = Button(t, words, variant="outline")
            self.extra_button.clicked.connect(lambda *_: callback())
            footer.addWidget(self.extra_button)
        self.ok_button = Button(t, "OK", variant="default")
        self.ok_button.clicked.connect(self._on_ok)
        footer.addWidget(self.ok_button)
        self.card_lay.addLayout(footer)

    def _on_ok(self) -> None:
        self._acted = True
        self.acknowledged.emit()
        self._dismiss()

    def _on_dismiss_without_action(self) -> None:
        self.acknowledged.emit()
