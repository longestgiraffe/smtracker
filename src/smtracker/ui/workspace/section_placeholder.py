"""The FIRST-RUN gate for a workspace section: one big placeholder that owns
the whole "nothing computed yet" moment, replacing N empty chart cards with
a single focused one -- requested directly, after weighing and rejecting two
narrower ideas: a skeleton+spinner+progress bar PER chart (does not scale --
a section with several charts would show several independent "loading"
widgets with nothing tying them together) and a spinner living in a rail
popover (workable, but this reads better with nowhere else for the eye to
go while nothing exists on screen yet).

**Empty is a property of the SECTION, not of any one chart.** Whether a
result exists is true or false for the whole section at once, by
construction, in the exact case this was built for (several charts computed
together, or meaningless read alone) -- so it is answered once here, not
once per chart. `WorkspaceWindow` shows this in place of every chart card
until the FIRST successful Run, then never again: switching BACK to this
placeholder on a later Run (even a failed one) was considered and rejected
-- once there is something real on screen, dimming it in place
(`bubble_chart.BubbleCard.set_running`) keeps that "what did I have a
second ago" anchor a full collapse-and-rebuild would throw away, and is
cheap to repeat on every iteration of a tune-and-rerun workflow. This
widget's own gate is therefore a ONE-WAY door: `WorkspaceWindow` never
constructs a second one and never rewinds `_ever_succeeded`.

**No fake phases, no fake progress number.** `compute_bubble_view` reports
once, at the end, on a worker thread -- there is no intermediate checkpoint
to show a percentage or a checklist against, and a progress bar that cannot
move honestly is worse than a plain spinner. `set_running` shows a spinner
and a "Computing..." label only; a phase checklist is a real future
addition, but it needs the fit pipeline to report phase boundaries first,
which is backend work this widget does not attempt to fake.

**Cancel is a SOFT cancel.** The fit may be deep inside a scipy multistart
loop by the time anyone clicks it, which cannot be interrupted cleanly.
`cancel_requested` asks the caller to stop CARING about the in-flight
result (bump the generation token the way a superseded Run already does),
not to stop the CPU work -- the worker finishes on its own and its result
is dropped, exactly as an ordinary superseded Run's already is.
"""
from __future__ import annotations

from qtpy.QtCore import Qt, QVariantAnimation, Signal
from qtpy.QtGui import QPainter
from qtpy.QtWidgets import QVBoxLayout, QWidget

from ..theme import icons
from ..theme import typography as ty
from ..widgets.controls import Button
from ..widgets.primitives import fit_label, icon, label
from .bubble_chart import CARD_TITLE_PX, CARD_TITLE_WEIGHT
from .toolbar import rem

__all__ = ["SectionPlaceholder"]

#: The rail's own spin speed (`toolbar.SPIN_MS`) -- restated rather than
#: imported, since importing a private-looking module constant across files
#: for a single number is worse than the small duplication; both are meant
#: to read as ONE visual language (a fit is running, however it is shown),
#: so a change to one without the other would be the real defect.
_SPIN_MS = 1000
_SPINNER_PX = rem(2)


class _Spinner(QWidget):
    """A turning `loader-circle`, forever, while shown.

    `toolbar.ComputeMark`'s own spin technique (`QVariantAnimation`, 0 to
    360, looped), without that widget's hover-to-stop-square affordance --
    this one has an explicit Cancel button beside it instead, so it has no
    second meaning to carry.
    """

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.setFixedSize(_SPINNER_PX, _SPINNER_PX)
        self._angle = 0.0
        self._spin = QVariantAnimation(self)
        self._spin.setStartValue(0.0)
        self._spin.setEndValue(360.0)
        self._spin.setDuration(_SPIN_MS)
        self._spin.setLoopCount(-1)
        self._spin.valueChanged.connect(self._turn)

    def _turn(self, value) -> None:
        self._angle = float(value)
        self.update()

    def showEvent(self, e):
        super().showEvent(e)
        self._spin.start()

    def hideEvent(self, e):
        super().hideEvent(e)
        self._spin.stop()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        box_rect = self.rect()
        centre = box_rect.center()
        p.translate(centre)
        p.rotate(self._angle)
        p.translate(-centre)
        icons.paint(p, "loader-circle", box_rect, self.t.muted_foreground, stroke=2.0)
        p.end()


class SectionPlaceholder(QWidget):
    """The whole "nothing computed yet" surface for one section -- idle
    (icon, title, description, a Run button) or running (spinner, a
    "Computing..." label, a Cancel button), toggled by `set_running`.

    A flat `t.muted` fill, not a bordered `widgets.primitives.Card` --
    deliberately: every OTHER surface in this workspace that carries real
    content is a Card, so this one reading as a quiet, un-elevated tray is
    itself part of the message ("there is nothing real here yet").
    """

    #: The idle Run button was clicked -- the caller opens the SAME Compute
    #: dialog the rail's own mark does (`WorkspaceWindow._open_compute_dialog`)
    #: rather than firing a Run blind: a first-time user still has to choose
    #: conditions, so this is a shortcut to the form, never a second control
    #: with its own idea of what to run.
    run_requested = Signal()

    #: The running Cancel button was clicked. See the module docstring's
    #: "Cancel is a SOFT cancel" for what the caller should actually do.
    cancel_requested = Signal()

    def __init__(self, t, *, title: str, description: str,
                icon_name: str = "chart", parent=None):
        super().__init__(parent)
        self.t = t
        self._title_text = title
        self.setObjectName(f"sectionplaceholder{id(self)}")
        self.setStyleSheet(
            f"QWidget#sectionplaceholder{id(self)} {{ background:{t.muted};"
            f" border-radius:{t.radius_xl:g}px; }}")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(rem(2), rem(2), rem(2), rem(2))
        outer.addStretch(1)

        self._icon = icon(icon_name, rem(2), t.muted_foreground)
        outer.addWidget(self._icon, alignment=Qt.AlignHCenter)
        self._spinner = _Spinner(t)
        self._spinner.hide()
        outer.addWidget(self._spinner, alignment=Qt.AlignHCenter)
        # GRADUATED, not even, spacing down this whole stack -- asked for
        # directly, after the title read as level with the description
        # rather than as its heading. The two gaps here USED to be 0.75
        # (icon->title) then 0.4 (title->description): tighter to the
        # description than to its own icon, so proximity itself grouped
        # the title with the text below it instead of the header above.
        # Reversed: title pulled toward the icon (one header cluster),
        # then MORE air before the description starts, and description
        # -> button (below) stays the largest gap of the three -- weight
        # and colour still carry the title/description split (see
        # `typography.py`'s own ladder-rule note), spacing is what makes
        # the GROUPING agree with it rather than fight it.
        outer.addSpacing(rem(0.5))

        # 575, not the shared `CARD_TITLE_WGHT` (600) -- a deliberate, LOCAL
        # trial for this ONE title, asked for directly; `BubbleCard`'s and
        # `ReplicateDetail`'s own titles stay at 600. `weight=
        # CARD_TITLE_WEIGHT` ("semibold") is kept as the non-variable
        # fallback -- `typography.ui`'s own note: the `wght` axis wins over
        # the enum whenever both are set, so a font with no axis still
        # renders sensibly at 600 rather than falling back to 400.
        self._title = label(title, ty.ui("base", CARD_TITLE_WEIGHT, px=CARD_TITLE_PX,
                                         wght=575), t.foreground)
        self._title.setAlignment(Qt.AlignHCenter)
        outer.addWidget(self._title)
        outer.addSpacing(rem(0.75))

        # A `FitLabel`: it pins its own height to its text (never less than 44,
        # so the placeholder does not jump between one line and two). A plain
        # wrapped `QLabel` made the whole window height-for-width, which cost
        # ~7 ms of layout on every frame of the sidebar's slide -- see
        # `primitives.FitLabel`.
        self._description = fit_label(description, ty.ui("sm"), t.muted_foreground,
                                      min_height=44)
        self._description.setAlignment(Qt.AlignHCenter)
        self._description.setFixedWidth(rem(24))
        outer.addWidget(self._description, alignment=Qt.AlignHCenter)
        outer.addSpacing(rem(1))

        self._run_btn = Button(t, "Run analysis", variant="default", glyph="play")
        self._run_btn.clicked.connect(self.run_requested.emit)
        outer.addWidget(self._run_btn, alignment=Qt.AlignHCenter)

        self._cancel_btn = Button(t, "Cancel", variant="outline")
        self._cancel_btn.clicked.connect(self.cancel_requested.emit)
        self._cancel_btn.hide()
        outer.addWidget(self._cancel_btn, alignment=Qt.AlignHCenter)

        outer.addStretch(1)
        self._running = False

    def set_running(self, on: bool) -> None:
        if on == self._running:
            return
        self._running = on
        self._icon.setVisible(not on)
        self._spinner.setVisible(on)
        self._description.setVisible(not on)
        self._run_btn.setVisible(not on)
        self._cancel_btn.setVisible(on)
        self._title.setText("Computing…" if on else self._title_text)
        if on:
            self._description_text = self._description.text()
        else:
            self._description.setText(getattr(self, "_description_text",
                                              self._description.text()))

    def set_progress(self, text: str) -> None:
        """What a running computation is doing, under "Computing…" -- in the
        description's own slot, which is hidden while running and restored after."""
        if not self._running:
            return
        self._description.setText(text)
        self._description.setVisible(bool(text))
