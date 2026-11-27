"""The cap on track length, as an in-window dialog (QC's "Max length").

Asked for after QC showed a 91-step track: "set a filter here so tracks with x (user setting)
steps filter out". The cap changes the table EVERY analysis reads, so what it would leave out
is shown live -- in all and per condition -- before Apply, with its cost said: long tracks are
preferentially slow, so a cap lowers a slow fraction.

THIN QT: the preview is `qc_view.track_cap_lines`, the words `qc_view.TRACK_CAP_ABOUT`; the
window applies the cap (`project.with_track_cap`) on the uncapped project it keeps.
"""
from __future__ import annotations

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QHBoxLayout

from ...qc_view import TRACK_CAP_ABOUT, track_cap_lines
from ..theme import typography as ty
from ..widgets.controls import Button, NumberField
from ..widgets.dialogs import CloseButton, OverlayDialog
from ..widgets.primitives import label

__all__ = ["TrackCapDialog"]

#: The field's top: no real track is this long, and a cap at the longest track is no cap.
_MAX_FIELD = 100_000


class TrackCapDialog(OverlayDialog):
    """A number of steps, its cost, and Apply / No cap."""

    #: the cap in steps, or None to remove it
    applied = Signal(object)

    def __init__(self, t, *, canonical, current: int | None = None, content=None,
                 parent=None, order=()):
        super().__init__(t, content, parent=parent)
        self._canonical = canonical
        #: the conditions' order, the window's (`condition_order`): the per-condition line follows it
        self._order = list(order or ())
        self._current = current
        self._card.setMinimumWidth(560)
        self._card.setMaximumWidth(640)
        head = QHBoxLayout()
        head.addWidget(label("Leave out long tracks", ty.ui("lg", "semibold"), t.foreground))
        head.addStretch(1)
        close = CloseButton(t)
        close.clicked.connect(self._dismiss)
        head.addWidget(close)
        self.card_lay.addLayout(head)
        about = label(TRACK_CAP_ABOUT, ty.ui("sm"), t.muted_foreground)
        about.setWordWrap(True)
        self.card_lay.addWidget(about)

        longest = track_cap_lines(canonical, _MAX_FIELD)["longest"]
        row = QHBoxLayout()
        row.setSpacing(10)
        row.addWidget(label("Longest track kept", ty.ui("sm", "medium"), t.muted_foreground))
        start = current if current is not None else max(4, longest)
        self.steps = NumberField(t, 4, _MAX_FIELD, int(start), width=150)
        self.steps.changed.connect(lambda _v: self._refresh())
        row.addWidget(self.steps)
        row.addWidget(label("steps", ty.ui("sm"), t.muted_foreground))
        row.addStretch(1)
        self.card_lay.addLayout(row)

        self.cost = label("", ty.ui("sm", "semibold", tabular=True), t.foreground)
        self.cost.setWordWrap(True)
        self.cost.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.card_lay.addWidget(self.cost)
        self.per_condition = label("", ty.ui("xs", tabular=True), t.muted_foreground)
        self.per_condition.setWordWrap(True)
        self.card_lay.addWidget(self.per_condition)

        foot = QHBoxLayout()
        self.clear_button = Button(t, "No cap", variant="outline")
        self.clear_button.setToolTip("Put every track back")
        self.clear_button.clicked.connect(self._clear)
        self.clear_button.setVisible(current is not None)
        foot.addWidget(self.clear_button)
        foot.addStretch(1)
        cancel = Button(t, "Cancel", variant="outline")
        cancel.clicked.connect(self._dismiss)
        self.apply_button = Button(t, "Apply", variant="default")
        self.apply_button.clicked.connect(self._apply)
        foot.addWidget(cancel)
        foot.addWidget(self.apply_button)
        self.card_lay.addLayout(foot)
        self._refresh()

    def _refresh(self) -> None:
        lines = track_cap_lines(self._canonical, int(self.steps.value()), self._order)
        self.cost.setText(lines["cost"])
        self.per_condition.setText("   ·   ".join(lines["per_condition"]))
        self.per_condition.setVisible(bool(lines["per_condition"]))
        self.apply_button.setEnabled(lines["ok"])

    def _apply(self) -> None:
        if not self.apply_button.isEnabled():
            return
        self._acted = True
        self.applied.emit(int(self.steps.value()))
        self._dismiss()

    def _clear(self) -> None:
        self._acted = True
        self.applied.emit(None)
        self._dismiss()
