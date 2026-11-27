"""The first window: which STAGE of the work, drawn as three shadcn cards.

The doors, their order and their words are `smtracker.launcher`'s and stay there
— this module only draws them. That split is the same one the whole project
runs on: `launcher.STAGES` is data, `describe_stages()` renders it for a headless
run or a `--help` line, and this renders it for a screen. A stage cannot be
enabled in one and absent from the other because neither owns the list.

**A card, not a `QCommandLinkButton`.** The control this replaced carried a
heading and a description inside one clickable target, which was the right SHAPE
and the wrong widget: it does not grow to fit a wrapped description, it clips it
silently, and the first render lost the last line of two cards. Every card here
is sized by its own text, so a blurb that grows makes the window taller instead
of disappearing.

**The glyphs are the only thing here that is not in the prototype.** Its
onboarding flow has no stage menu — that screen belongs to the application, not
to the mock-up — so the three icons were chosen from the lucide set
`ui.theme.icons.PATHS` already carries rather than by adding path data by hand:
`scan`'s corner brackets for a field of view being prepared, `grid-2x2` for the
mesh, `chart`'s axes for the analysis. Distinct at 20 px in both schemes, which
is the whole requirement — a set of icons that read as the same mark at a glance
is worse than no icons, because it costs the width and returns nothing.

**Neutral, deliberately.** The control this replaced drew its chevrons in the
palette's highlight colour. shadcn's primary is a zinc, not a brand hue, and the
rule the rest of this shell follows is that the only saturated colour on a screen
belongs to the DATA. A blue arrow on a menu spends the one signal the analysis
windows need.

Keyboard behaviour is preserved exactly, because it is load-bearing: the default
stage is focused and answers Return, so the analysis path stays one keypress from
launch. This project records what the alternative costs — the analysis catalogue
and the track views were once reachable only behind flags the no-argument flow
never set, and nobody found them.
"""
from __future__ import annotations

from qtpy.QtCore import QRectF, Qt, Signal
from qtpy.QtGui import QColor, QFontMetricsF, QPainter, QPen
from qtpy.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ... import theme_view
from .. import theme
from ..theme import typography as ty
from ..widgets import tooltip as tip
from ..widgets.controls import Button
from ..workspace.toolbar import ToolButton
from ..widgets.primitives import (
    Card,
    ColorFade,
    ShadowHost,
    icon,
    label,
    shadow_room,
)

__all__ = ["STAGE_GLYPHS", "RING_PX", "StageCard", "StagesDialog", "choose_stage"]

#: `ring-[3px]` -- the width of the focus halo, and it must fit in the shadow
#: room the host already reserves (4 px for `shadow-sm`). A wider ring would be
#: cropped by the host's own edge, which is the failure the room exists to
#: prevent for shadows.
RING_PX = 3

#: `ring-ring/50`. The halo is the SOFT half of the focus state and the border
#: recolour is the hard half; either alone reads as something other than focus --
#: the border alone reads as one card being subtly wrong.
RING_ALPHA = 0.5

#: One lucide glyph per stage key, and no two alike. Keys are
#: `launcher.STAGES`'; a stage without an entry draws no glyph rather than
#: raising, because a missing icon must not be able to stop the application
#: opening its own first window.
STAGE_GLYPHS = {
    "track": "scan",
    "segment": "grid-2x2",
    "analyse": "chart",
    "simulate": "flask-conical",
}

#: `text-base` at `font-semibold` -- shadcn's own `CardTitle`, which carries
#: `leading-none font-semibold` and NO `text-*` class, so it inherits the
#: 16 px root. Both numbers are the registry's rather than ours.
#:
#: 16/600, DOWN FROM 18/475, IN TWO STEPS -- AND THE FIRST STEP ALONE WAS
#: WRONG. The cards were specified at 18/475 before the section titles inside
#: the app were tuned; those settled at 16/600, leaving the launcher a step
#: larger and a step lighter than every heading after it. Dropping to 16 on
#: its own made the card WORSE, not better: 16 over the 14 px body is a ratio
#: of 1.14, which `typography.SIZES` already records as "too small to read as
#: deliberate", and MEASURED it left the title only 1.47 px taller in cap
#: height than the text beneath it (13.09 -> 11.64, against a body cap of
#: 10.17). Both differentiators were weak at once -- 1.14 in size and 75 units
#: in weight. 600 makes the WEIGHT carry the distinction, which is what the
#: registry does and what the section titles already do.
TITLE_PX = 16
TITLE_WGHT = 600


class StageCard(Card):
    """One door: a glyph, a title, and the sentence that says what is behind it.

    **The press effect is the registry's `active:translate-y-px`** -- a
    stylesheet MARGIN, not a move. Qt applies it to the widget's own box, so the
    painted frame and its contents shift down together while the widget's
    geometry stays put and nothing in the layout around it moves. A card that
    changed height on press would shove the two below it down a pixel each time.

    **The hover is a FADE, and that is what makes it feel alive.** shadcn's
    `transition-colors` is 150 ms on `cubic-bezier(0.4, 0, 0.2, 1)`; a Qt
    stylesheet cannot transition anything, so `ColorFade` animates the colour and
    re-applies the rule each frame. Snapping was tried first and reads as a
    different, cheaper control.

    A DISABLED card does neither, and says why through its tooltip. `launcher`
    has the rule: a stage that is not built is shown disabled rather than hidden,
    because a user should see the shape of the whole tool -- and an enabled
    button that opens nothing is worse than an absent one.
    """

    clicked = Signal()

    def __init__(self, t, *, key: str, glyph: str | None, title: str,
                 summary: str, enabled: bool = True, parent=None):
        super().__init__(t, parent=parent)
        #: The `launcher.STAGES` key this door returns, and its heading. Both are
        #: read by the tests, which assert what is on the WIDGET rather than what
        #: is in the data behind it -- "enabled" is a claim about the card.
        self.stage_key = key
        self.stage_title = title
        self._enabled = enabled
        self._pressed = False
        #: `:focus-visible`, NOT `:focus` -- see `focusInEvent`. The card can
        #: hold the focus with no ring on it at all, which is the resting state
        #: of this window.
        self._ring = False
        self._fill = t.card
        #: The colour ON SCREEN this frame, which is not `_fill` while the fade
        #: is running.
        self._now = t.card
        self._fade = ColorFade(self, self._on_fade)

        ink = t.foreground if enabled else t.muted_foreground
        sub = t.muted_foreground

        lay = QHBoxLayout(self)
        # `p-4` down each side, and a little less at top and bottom: the title's
        # line box already carries leading, so equal padding measures equal and
        # looks bottom-heavy.
        lay.setContentsMargins(16, 14, 16, 15)
        lay.setSpacing(14)

        text = QVBoxLayout()
        text.setContentsMargins(0, 0, 0, 0)
        text.setSpacing(3)
        self._title = label(title, ty.ui("base", "semibold", px=TITLE_PX,
                                         wght=TITLE_WGHT), ink)
        # FIXED, so the title is exactly one line and never absorbs the card's
        # spare height. It did: a QLabel's default policy lets it grow, so a card
        # with one line of summary handed its title 7 px it had nowhere else to
        # put, the text re-centred inside a taller box, and the glyph beside it --
        # sized to a natural line -- was left pointing at the gap above the
        # letters. The two short cards were 4.5 px out and the tall one was not,
        # which is what made it look like a bug in one card rather than a rule.
        self._title.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        text.addWidget(self._title)

        if glyph is not None:
            # CENTRED ON THE TITLE'S OWN LINE, and measured rather than nudged.
            #
            # `AlignTop` put the glyph's box at the top of the card's content
            # rect, level with the top of the title's LINE BOX -- which is not
            # where the letters start. A line box carries the face's ascent, and
            # Inter at 18 px has 17.44 of ascent above the baseline for 13.09 of
            # cap height, so the capitals begin 4.3 px below the box they sit in.
            # MEASURED on the built card, under the Geist this screen was first
            # drawn in: the glyph's centre landed at y=25.0 against the title's
            # cap centre at 30.5. Five and a half pixels is small enough to look
            # like a mistake rather than a decision, which is the worst place for
            # it to be. The face changed and the gap narrowed; the defect is a
            # property of AlignTop, not of the font, so the fix below stands.
            #
            # The holder is one title-line tall with the glyph centred inside,
            # so the two centres coincide by CONSTRUCTION -- no offset to retune
            # when the step or the face changes. `AlignTop` then pins that line
            # to the top of the card, which is what keeps the glyph on the
            # heading rather than floating between the heading and the summary.
            holder = QWidget()
            holder.setFixedHeight(self._title.sizeHint().height())
            inside = QVBoxLayout(holder)
            inside.setContentsMargins(0, 0, 0, 0)
            inside.addWidget(icon(glyph, 20, ink), 0, Qt.AlignVCenter)
            lay.addWidget(holder, 0, Qt.AlignTop)
            lay.setStretch(0, 0)
        self._summary = label(summary, ty.ui("sm"), sub)
        # THE CARD IS SIZED BY ITS TEXT, and getting that to happen takes all
        # three of these. `setWordWrap` lets the sentence break; `Ignored`
        # horizontally stops the label claiming the width of its longest line, so
        # the window is sized by the design rather than by the longest blurb; and
        # `setHeightForWidth` is what makes the layout ASK how tall the label is
        # once it knows how wide it will be. Without the third the height is a
        # guess made at the label's natural width, and the guess is generous --
        # it put 60 px of white inside every card.
        self._summary.setWordWrap(True)
        policy = self._summary.sizePolicy()
        policy.setHorizontalPolicy(QSizePolicy.Ignored)
        policy.setVerticalPolicy(QSizePolicy.Minimum)
        policy.setHeightForWidth(True)
        self._summary.setSizePolicy(policy)
        # AND AN EXPLICIT MINIMUM, which is the other half of the same problem.
        # A wrapped QLabel reports its `minimumSizeHint` for the NARROWEST it
        # could be asked to be, where this sentence wraps to five or six lines --
        # so the card's minimum was 19 px taller than the card, the window's
        # minimum came out 38 px taller than its own size hint, and a window that
        # cannot reach its natural height puts the difference somewhere. It went
        # under the cards, as a band of white above Cancel.
        #
        # An explicit minimum takes precedence over the hint. One line is honest:
        # the window never gets near it, since `setMinimumWidth(600)` below keeps
        # every card wide enough for two.
        self._summary.setMinimumHeight(QFontMetricsF(self._summary.font()).height())
        text.addWidget(self._summary)
        # Spare height pools HERE, under the text, rather than being shared out
        # between the title and the summary and opening the gap between them.
        text.addStretch(1)
        lay.addLayout(text, 1)

        # PREFERRED, not Minimum: `Minimum` means "the hint is a floor, grow if
        # there is room", so three cards in a column split the window's slack
        # between them and each one's title drifted away from its own summary.
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        if enabled:
            self.setCursor(Qt.PointingHandCursor)
            self.setFocusPolicy(Qt.StrongFocus)
        else:
            self.setFocusPolicy(Qt.NoFocus)
        self._restyle()

    # -- painting ------------------------------------------------------------ #

    def _restyle(self) -> None:
        t = self.t
        dip = " margin-top:1px; margin-bottom:-1px;" if self._pressed else ""
        # `focus-visible:border-ring` -- the HARD half of the focus state. The
        # soft half is the 3 px halo, which a card cannot draw because Qt clips
        # a widget to its own rectangle; `_CardHost` paints it, in the room it
        # already reserves for shadows.
        edge = t.ring if self._ring else t.border
        self.setStyleSheet(
            f"QFrame#card {{ background:{self._now};"
            f" border:1px solid {edge};"
            f" border-radius:{self.radius:g}px;{dip} }}")

    def _on_fade(self, colour) -> None:
        self._now = colour.name()
        self._restyle()

    # -- pointer -------------------------------------------------------------- #

    def enterEvent(self, e):
        if self._enabled:
            self._fade.to(self._now, self.t.accent)
        super().enterEvent(e)

    def leaveEvent(self, e):
        # `_restyle` EXPLICITLY, and not through the fade: `ColorFade.to` returns
        # early when the colour is already at rest, so a card pressed without
        # ever having been hovered would clear the flag and never repaint -- it
        # would stay visibly dipped. A person cannot do that; a keyboard can.
        was_pressed, self._pressed = self._pressed, False
        if was_pressed:
            self._restyle()
        if self._enabled:
            self._fade.to(self._now, self._fill)
        super().leaveEvent(e)

    def mousePressEvent(self, e):
        if self._enabled and e.button() == Qt.LeftButton:
            self._pressed = True
            # A CLICK PUTS THE RING AWAY, even on the card that already had it.
            # Qt sends no `focusInEvent` when the focused widget is the one
            # clicked, so the reason test above never runs and a card tabbed to
            # and then clicked would keep its halo for the rest of the window's
            # life. Browsers drop `:focus-visible` on exactly this interaction.
            if self._ring:
                self._ring = False
                self._tell_host()
            self._restyle()
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e):
        if self._pressed:
            self._pressed = False
            self._restyle()
            # Only a release INSIDE the card is a click, which is the escape
            # hatch every button has: press, think better of it, drag off.
            if e.button() == Qt.LeftButton and self.rect().contains(e.pos()):
                self.clicked.emit()
        super().mouseReleaseEvent(e)

    # -- keyboard ------------------------------------------------------------- #

    def _tell_host(self) -> None:
        """Hand the focus to whoever draws the halo, since this card cannot.

        Qt clips a widget's painting to its own rectangle, so a ring OUTSIDE the
        border is not the card's to draw -- the same constraint that puts every
        shadow in the parent. Pushed rather than polled: a host that swept its
        children for the focused one on every paint would be doing
        `findChildren` per frame, which is the cost the shadow cache exists to
        avoid.
        """
        host = self.parent()
        setter = getattr(host, "set_focused_card", None)
        if setter is not None:
            setter(self if self._ring else None)

    #: The focus reasons a browser calls `:focus-visible`. Qt hands the reason to
    #: `focusInEvent`, which is the whole mechanism -- Tab, Shift-Tab and a
    #: shortcut are a person navigating by keyboard and need to be shown where
    #: they are; a click already tells them, and `setFocus()` at open is the
    #: application talking to itself.
    KEYBOARD_FOCUS = (Qt.TabFocusReason, Qt.BacktabFocusReason,
                      Qt.ShortcutFocusReason)

    def focusInEvent(self, e):
        """Ring on KEYBOARD focus only, which is the registry's rule verbatim.

        `buttonVariants` carries `focus-visible:border-ring
        focus-visible:ring-[3px] focus-visible:ring-ring/50` -- every one of the
        three is behind `focus-visible`, and that pseudo-class does not match a
        mouse click or a programmatic `.focus()`. Reading it as `:focus` puts a
        3 px halo on the default door the moment the window opens and leaves it
        on whichever card was last clicked, which is not a focus indicator any
        more, just one card wearing a heavier outline than its neighbours.

        The default door still holds the focus at open. It simply does not
        ADVERTISE it, and Return reaches it either way.
        """
        self._ring = e.reason() in self.KEYBOARD_FOCUS
        self._restyle()
        self._tell_host()
        super().focusInEvent(e)

    def focusOutEvent(self, e):
        self._ring = False
        self._pressed = False
        self._restyle()
        self._tell_host()
        super().focusOutEvent(e)

    def keyPressEvent(self, e):
        """Return and Space activate, and both SHOW the press.

        Space is the platform convention for a focused control and Return is what
        this window's default stage answers, so a keyboard user reaches the
        analysis path without touching the pointer. The dip is applied on the way
        in and cleared on release, so the card looks pressed rather than blinking
        once, which is what a repaint-free activation looks like.
        """
        if self._enabled and e.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            if not e.isAutoRepeat():
                self._pressed = True
                self._restyle()
            e.accept()
            return
        super().keyPressEvent(e)

    def keyReleaseEvent(self, e):
        if (self._enabled and self._pressed and not e.isAutoRepeat()
                and e.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space)):
            self._pressed = False
            self._restyle()
            self.clicked.emit()
            e.accept()
            return
        super().keyReleaseEvent(e)


class _CardHost(ShadowHost):
    """The cards' parent: it paints their shadows, and the focus halo.

    Both for the same reason. A card cannot draw beyond its own rectangle, so
    anything that belongs OUTSIDE its border -- the soft edge under it, the ring
    around it -- has to come from the widget it sits in. Children paint after
    their parent, so the card's own fill and border land on top of both and the
    z-order costs nothing.

    The ring fits because the host already reserves `shadow_room` on every side
    for the shadows; it is drawn in the margin rather than in the gap between
    two cards, so no ring can touch its neighbour.
    """

    def __init__(self, t, parent=None):
        super().__init__(t, parent)
        self._focused: StageCard | None = None

    def set_focused_card(self, card) -> None:
        if card is not self._focused:
            self._focused = card
            self.update()

    def paintEvent(self, e):
        super().paintEvent(e)
        card = self._focused
        if card is None:
            return
        try:
            box = QRectF(card.geometry())
        except RuntimeError:                    # the card went away under us
            self._focused = None
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        colour = QColor(self.t.ring)
        colour.setAlphaF(RING_ALPHA)
        # CENTRED ON THE PATH, like every Qt pen: half the stroke would sit
        # under the card's own border and be painted over, so the rect is grown
        # by half the width to put the whole ring outside it. The radius follows,
        # or the ring's corners cut across the card's.
        pen = QPen(colour, RING_PX)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        grown = box.adjusted(-RING_PX / 2, -RING_PX / 2, RING_PX / 2, RING_PX / 2)
        p.drawRoundedRect(grown, card.radius + RING_PX / 2, card.radius + RING_PX / 2)
        p.end()


class StagesDialog(QDialog):
    """The window itself: a wordmark, the doors, and a way out.

    Reads `launcher.STAGES` at construction rather than taking a list, so the
    window and `describe_stages()` cannot disagree about what exists.
    """

    def __init__(self, t, stages, *, default: str | None = None, parent=None):
        super().__init__(parent)
        self.t = t
        self._chosen: str | None = None
        #: True once the light / dark switch was pressed: the dialog closed so `choose_stage` can show a new one
        #: in the saved scheme (`_switch_theme`).
        self.theme_switched = False
        self.update_release = None
        self.update_notice = None
        from ...buildstamp import build_stamp

        self.setWindowTitle(f"SMTracker  ·  {build_stamp()}")
        self.setStyleSheet(f"QDialog {{ background:{t.background}; }}")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(28, 24, 28, 20)
        outer.setSpacing(18)
        outer.addWidget(self._header())

        # THE CARDS SIT IN A ShadowHost, which is not decoration. Qt clips a
        # widget's painting to its own rectangle, so a card cannot put a soft
        # edge in the gap beside it -- the PARENT draws every shadow, and
        # children paint after their parent, so the z-order comes free. The
        # margin is what the shadow needs outside each card; without it the
        # bottom row's shadow is cropped by the host's own edge.
        room = shadow_room(t)
        host = _CardHost(t)
        cards = QVBoxLayout(host)
        cards.setContentsMargins(room, room, room, room)
        cards.setSpacing(10)

        self._cards: dict[str, StageCard] = {}
        for key, title, blurb, enabled in stages:
            card = StageCard(
                t,
                key=key,
                glyph=STAGE_GLYPHS.get(key),
                title=title,
                summary=_summary(blurb),
                enabled=enabled,
            )
            # THE CARD TAKES ONE SENTENCE, THE TOOLTIP TAKES ALL OF IT. Every
            # blurb is written as a headline and then a detail; a card that
            # showed both would be a paragraph with a border, and one that
            # dropped the detail would lose what `describe_stages()` prints.
            card.setToolTip(_detail(blurb))
            card.clicked.connect(lambda k=key: self._pick(k))
            cards.addWidget(card)
            self._cards[key] = card
        # The cards keep their own heights; anything left over sits below the
        # last one. Without this the column shares it out, and three cards that
        # should be sized by their text are instead sized by the window.
        cards.addStretch(1)
        outer.addWidget(host)

        cancel = Button(t, "Cancel", variant="outline")
        cancel.clicked.connect(self.reject)
        # REPORT A PROBLEM, at the first window too (10 Nov 2026): the app may have closed on its own the last
        # time, and this is where a person is when they start it again. One zip of the log, the build and the
        # versions (`crashlog.report_bundle`); the workspace's header menu has the same.
        self.report_button = Button(t, "Report a problem\u2026", variant="ghost")
        self.report_button.setToolTip("Save a zip to send when something went wrong. It holds the error log, the "
                                      "build and package versions and the settings file, but no tracks, images "
                                      "or results.")
        self.report_button.clicked.connect(self.report_problem)
        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.addWidget(self.report_button)
        footer.addStretch(1)
        footer.addWidget(cancel)
        outer.addLayout(footer)

        self.setMinimumWidth(600)
        #: The door Return reaches. `QPushButton.setDefault` is what a Qt dialog
        #: normally uses for this, and a card is not a button, so the dialog
        #: carries the fact itself -- see `keyPressEvent`.
        self.default_key = default if default in self._cards else None
        chosen = self._cards.get(default)
        if chosen is not None and chosen.isEnabled():
            chosen.setFocus()

    def show_update(self, release) -> None:
        """A newer release is out (`widgets.update_notice.start_check`): a notice under the header, in the window.
        It never blocks the start; Not now removes it."""
        from ..widgets.update_notice import UpdateNotice

        if getattr(self, "update_notice", None) is not None:
            return
        #: kept, so the window rebuilt by the light / dark switch shows it again until "Not now"
        self.update_release = release
        self.update_notice = UpdateNotice(self.t, release, parent=self, on_later=self._dismiss_update)
        self.layout().insertWidget(1, self.update_notice)
        self.update_notice.setVisible(True)      # after it is in the layout: a widget added to a shown window starts hidden
        # the window grows by the notice, never past its screen, so the doors keep their room
        room = self.screen().availableGeometry().height() if self.screen() is not None else self.sizeHint().height()
        self.resize(self.width(), min(max(self.height(), self.sizeHint().height()), room))

    def _dismiss_update(self) -> None:
        self.update_release = None
        self.update_notice.dismiss()
        self.update_notice = None

    def report_problem(self, dest: str | None = None):
        """Save the report (asked where when ``dest`` is None) and say where it is and what it holds."""
        from ..widgets.report import report_from_dialog

        return report_from_dialog(self, dest)

    def cards(self) -> dict[str, StageCard]:
        """The live cards by stage key, in the order `launcher.STAGES` gives them."""
        return dict(self._cards)

    def keyPressEvent(self, e):
        """Return runs the DEFAULT door, wherever the focus happens to be.

        A `QDialog` gives this to whichever button is `setDefault(True)`, and
        nothing here is a `QPushButton` -- so with the focus on Cancel, or
        nowhere, Return did nothing at all. That silently costs the property this
        window exists to keep: the analysis path is one keypress from launch, and
        it is what every existing user expects `smtracker` to do.

        A card that HAS the focus handles its own Return, so this fires only when
        the focus is elsewhere.
        """
        if e.key() in (Qt.Key_Return, Qt.Key_Enter) and self.default_key:
            card = self._cards.get(self.default_key)
            if card is not None and card.isEnabled() and not card.hasFocus():
                self._pick(self.default_key)
                e.accept()
                return
        super().keyPressEvent(e)

    def _header(self) -> QWidget:
        """The wordmark, and the one question this window asks."""
        t = self.t
        bar = QWidget()
        # FIXED VERTICALLY, or the header eats the window's slack. A QVBoxLayout
        # hands spare height to whichever child will take it, and a bare QWidget
        # takes all of it -- which put 90 px of white between the wordmark and
        # the question under it.
        bar.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(14)

        from ...appicon import logo_pixmap

        titles = QVBoxLayout()
        titles.setContentsMargins(0, 0, 0, 0)
        titles.setSpacing(6)
        # THE WORDMARK (13 Nov 2026), in the theme's own version; the typed name where it did not ship
        logo = logo_pixmap(t.color_scheme == "dark", 34, self.devicePixelRatioF())
        if logo is not None:
            mark = QLabel()
            mark.setPixmap(logo)
            mark.setAccessibleName("SMTracker")
            mark.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            titles.addWidget(mark, 0, Qt.AlignLeft)
        else:
            titles.addWidget(label("SMTracker", ty.ui("2xl", "semibold"), t.foreground))
        titles.addWidget(label("Where would you like to start?", ty.ui("sm"),
                               t.muted_foreground))
        lay.addLayout(titles, 1)
        # THE LIGHT / DARK SWITCH (27 Nov 2026), the analysis rail's own button: its glyph names what a press
        # switches to (`theme_view`). Top right, level with the wordmark.
        scheme = getattr(t, "color_scheme", "light")
        self.theme_button = ToolButton(t, "theme", theme_view.switch_glyph(scheme))
        self.theme_button.clicked.connect(lambda _key: self._switch_theme())
        self._theme_tip = tip.attach(t, self.theme_button, theme_view.switch_tip(scheme), side="left")
        lay.addWidget(self.theme_button, 0, Qt.AlignTop)
        return bar

    def _switch_theme(self) -> None:
        """Light <-> dark: SAVED (the one setting every window reads), then this window closed so `choose_stage`
        shows it again in the other scheme. REBUILT, not recoloured, as the workspace is: stylesheets bake colour in
        at construction (`ui.workspace.window._toggle_theme`)."""
        theme.save_scheme(theme_view.other_scheme(getattr(self.t, "color_scheme", "light")))
        self.theme_switched = True
        self.reject()

    def _pick(self, key: str) -> None:
        self._chosen = key
        self.accept()

    @property
    def chosen(self) -> str | None:
        return self._chosen


def _summary(blurb: str) -> str:
    """The one sentence that answers "what is this", from a two-part blurb.

    A headline is often two sentences, and two wrap to three lines at this width,
    where a card stops being scannable. The rest is one hover away.
    """
    headline, _, _detail_part = blurb.strip().partition("\n")
    first, stop, _rest = headline.strip().partition(". ")
    return (first + ".") if stop else headline.strip()


def _detail(blurb: str) -> str:
    """Everything the card does not show, for the tooltip. Never empty."""
    headline, _, detail = blurb.strip().partition("\n")
    _first, stop, rest = headline.strip().partition(". ")
    parts = [x.strip() for x in (rest if stop else "", detail) if x.strip()]
    return "\n".join(parts) or blurb.strip()


def choose_stage(*, parent=None, default: str | None = None,
                 color_scheme: str | None = None) -> str | None:
    """Show the launcher; return the chosen stage key, or ``None`` if cancelled.

    ``color_scheme`` is for a screenshot run that wants to force one; left alone
    it is the SAVED scheme (`theme.current`), which every stage window reads too.
    The light / dark switch saves the other scheme and shows the launcher again,
    in the same place, so the window the caller waits on is always the one on
    screen; once a switch has been pressed the saved scheme rules over ``color_scheme``.
    """
    from qtpy.QtWidgets import QApplication

    # CONSTRUCTING A WIDGET BEFORE A QApplication EXISTS ABORTS THE PROCESS.
    # Not an exception -- Qt calls abort(), so there is no traceback, no message
    # and nothing to catch; `smtracker` with no arguments once died at exit 127
    # with zero bytes of output. This is the FIRST window the application opens,
    # so nothing upstream has made one.
    QApplication.instance() or QApplication([])
    from ...appicon import apply as _apply_icon
    _apply_icon()

    from ... import launcher
    from ..widgets.update_notice import start_check
    # its own size, centred on the screen it opens on and never past it (`ui.widgets.screenfit`)
    from ..widgets.screenfit import fit_to_screen

    where = None
    carried = None
    while True:
        dialog = StagesDialog(theme.current(color_scheme), launcher.STAGES,
                              default=default or launcher.STAGE_ANALYSE,
                              parent=parent)
        fit_to_screen(dialog, maximise=False)
        if where is None:
            start_check(dialog)             # once a day, in the background; a notice appears only for a newer release
        elif carried is not None:
            dialog.show_update(carried)
        if where is not None:
            dialog.move(where)              # the rebuilt window opens where the switched one stood
        accepted = dialog.exec_() if hasattr(dialog, "exec_") else dialog.exec()
        if not dialog.theme_switched:
            return dialog.chosen if accepted else None
        where, color_scheme, carried = dialog.pos(), None, getattr(dialog, "update_release", None)
