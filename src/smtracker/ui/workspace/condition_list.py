"""The conditions a Run includes, one fixed-height row each (`ConditionList`): shared by the HMM and Jump distances
Run dialogs. It lived in `msd_chart` until the MSD section's rebuild (16 Oct 2026), whose dialog fits every condition
and no longer needs it."""
from __future__ import annotations

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QHBoxLayout, QScrollArea, QVBoxLayout, QWidget

from ..theme import typography as ty
from ..widgets.primitives import Checkbox, label
from .toolbar import rem

__all__ = ["ConditionList"]


class ConditionList(QScrollArea):
    """The conditions to include, one fixed-height row each, scrolling past `VISIBLE` rows.

    Fixed heights, not a stretchable grid: a grid handed less height than it wants squeezes
    its rows onto each other, which is how the first version of the MSD dialog drew four
    condition names on top of one another in a window that was not full screen."""

    ROW_H = 28
    VISIBLE = 5

    toggled = Signal()

    def __init__(self, t, conditions: list, chosen, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setStyleSheet("QScrollArea { background:transparent; }")
        host = QWidget()
        host.setStyleSheet("background:transparent;")
        lay = QVBoxLayout(host)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.boxes: dict = {}
        for key, name in conditions:
            row = QWidget()
            row.setFixedHeight(self.ROW_H)
            row_lay = QHBoxLayout(row)
            row_lay.setContentsMargins(0, 0, 0, 0)
            row_lay.setSpacing(rem(0.6))
            box = Checkbox(t, checked=chosen is None or key in chosen)
            box.toggled.connect(lambda _on: self.toggled.emit())
            row_lay.addWidget(box)
            row_lay.addWidget(label(name, ty.ui("sm"), t.foreground), 1)
            lay.addWidget(row)
            self.boxes[key] = box
        lay.addStretch(1)
        self.setWidget(host)
        self.setFixedHeight(self.ROW_H * max(1, min(len(conditions), self.VISIBLE)))

    def included(self) -> dict:
        return {key: box.checked for key, box in self.boxes.items()}
