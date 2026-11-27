"""The u-track settings, on Detect & link when u-track is the tracker.

THIN QT. The settings, their labels, units and one-line helpers are
`prep_view.UTRACK_FIELDS`; the values are `prep_view.utrack_cells`; what differs from the
lab standard is `prep_view.utrack_differences`; the rows per condition are
`prep_view.utrack_condition_rows`. An edit is reported as typed and filed by the window
through `prep_model.set_utrack`, which refuses what u-track would not take.

ONE SET FOR EVERY CONDITION. u-track's detection is a statistical test against each spot's
own local background, so one significance serves movies of different brightness -- the
lab ran all 2063 movies on the lab's data drive with one set. What IS per condition (the PSF u-track derives
from the optics, the outlines the background test needs) is in the table under the card.

The everyday settings are shown; the rest sit behind "Show all settings", as u-track's own
GUI keeps them behind its "Advanced" buttons.
"""
from __future__ import annotations

from qtpy.QtCore import Signal
from qtpy.QtWidgets import QGridLayout, QHBoxLayout, QVBoxLayout, QWidget

from ...prep_view import UTRACK_CHOICES, UTRACK_FIELDS, UTRACK_GROUPS
from ..theme import typography as ty
from ..widgets.controls import Button, Segmented
from ..widgets.primitives import Card, label
from ..widgets.settings import InfoLink
from .calibration_page import Field

__all__ = ["UTrackCard"]


class UTrackCard(QWidget):
    """The settings card and the per-condition table. Signals carry requests."""

    #: setting key, the text as typed ("1"/"0" for a switch, the index for a choice)
    edited = Signal(str, str)
    lab_requested = Signal()

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.setStyleSheet("background:transparent;")
        self._controls: dict = {}
        self._advanced: list = []
        self._show_all = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        card = Card(t, shadowed=False)
        cl = QVBoxLayout(card)
        cl.setContentsMargins(16, 14, 16, 14)
        cl.setSpacing(10)
        head = QHBoxLayout()
        head.setSpacing(8)
        head.addWidget(label("u-track settings", ty.ui("sm", "semibold"), t.foreground))
        head.addWidget(label("one set for every condition", ty.ui("xs"),
                             t.muted_foreground))
        head.addStretch(1)
        self.more = Button(t, "Show all settings", size="sm", variant="ghost")
        self.more.clicked.connect(self._toggle_all)
        head.addWidget(self.more)
        lab = Button(t, "Standard settings", size="sm", glyph="undo")
        lab.setToolTip("Reset every setting to the standard.")
        lab.clicked.connect(self.lab_requested.emit)
        head.addWidget(lab)
        cl.addLayout(head)

        # ONE GRID FOR EVERY GROUP, so the boxes and the helpers line up down the card;
        # a grid per group sized each group's label column to its own longest label.
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)
        grid.setColumnStretch(3, 1)
        row = 0
        for g_index, group in enumerate(UTRACK_GROUPS):
            if g_index:
                grid.setRowMinimumHeight(row, 6)
                row += 1
            grid.addWidget(label(group, ty.ui("xs", "semibold"), t.muted_foreground),
                           row, 0, 1, 3)
            row += 1
            for key, text, unit, g, kind, advanced, help_text in UTRACK_FIELDS:
                if g != group:
                    continue
                name = label(text, ty.ui("sm"), t.foreground)
                control = self._control(key, kind, unit)
                info = InfoLink(t, "what it does", help_text)
                grid.addWidget(name, row, 0)
                grid.addWidget(control, row, 1)
                grid.addWidget(info, row, 2)
                if advanced:
                    self._advanced += [name, control, info]
                row += 1
        cl.addLayout(grid)

        self.differs = label("", ty.ui("xs", "medium"), t.foreground)
        self.differs.setWordWrap(True)
        cl.addWidget(self.differs)
        lay.addWidget(card)

        table = Card(t, shadowed=False)
        self._grid = QGridLayout(table)
        self._grid.setContentsMargins(14, 12, 14, 12)
        self._grid.setHorizontalSpacing(14)
        self._grid.setVerticalSpacing(6)
        self._grid.setColumnStretch(3, 1)
        for col, text in enumerate(("Condition", "PSF sigma", "Outlines", "Status")):
            self._grid.addWidget(label(text, ty.ui("xs", "medium"), t.muted_foreground),
                                 0, col)
        self._rows: dict = {}
        lay.addWidget(table)
        self._apply_show_all()

    def _control(self, key: str, kind: str, unit: str):
        t = self.t
        if kind == "switch":
            control = Segmented(t, ["Off", "On"], current=0, size="sm")
            control.changed.connect(lambda i, k=key: self.edited.emit(k, str(i)))
        elif kind == "choice":
            control = Segmented(t, list(UTRACK_CHOICES[key]), current=0, size="sm")
            control.changed.connect(lambda i, k=key: self.edited.emit(k, str(i)))
        else:
            # A pair is typed "0.5, 0.01" and the birth/death cost may be negative: the
            # numeric validator would refuse both, so those are plain text, checked by
            # `prep_model.set_utrack` like every other value.
            plain = key in ("brown_scaling", "amp_ratio_limit", "strategy_bd")
            control = Field(t, numeric=not plain, unit=unit, width=170)
            control.committed.connect(lambda text, k=key: self.edited.emit(k, text))
        self._controls[key] = (kind, control)
        return control

    def _toggle_all(self) -> None:
        self._show_all = not self._show_all
        self._apply_show_all()

    def _apply_show_all(self) -> None:
        for widget in self._advanced:
            widget.setVisible(self._show_all)
        self.more.set_text("Show fewer settings" if self._show_all else "Show all settings")

    # -- what the window tells the card ---------------------------------------- #
    def set_values(self, cells: dict, differences: list) -> None:
        """``cells``: `prep_view.utrack_cells`; ``differences``:
        `prep_view.utrack_differences`."""
        for key, (kind, control) in self._controls.items():
            text = cells.get(key, "")
            if kind in ("switch", "choice"):
                index = int(text or 0)
                if control.current() != index:
                    control.select(index)
            else:
                control.set_text(text)
        if differences:
            self.differs.setText("Differs from the standard settings: " + "; ".join(differences))
        else:
            self.differs.setText("Same as the standard settings.")

    def set_rows(self, rows: list) -> None:
        """``rows``: `prep_view.utrack_condition_rows`."""
        t = self.t
        for row in rows:
            widgets = self._rows.get(row["name"])
            if widgets is None:
                r = len(self._rows) + 1
                self._grid.addWidget(label(row["name"], ty.ui("sm", "medium"),
                                           t.foreground), r, 0)
                widgets = {}
                for col, key in enumerate(("psf", "background", "status"), start=1):
                    cell = label("", ty.ui("xs"), t.muted_foreground)
                    cell.setWordWrap(key == "status")
                    self._grid.addWidget(cell, r, col)
                    widgets[key] = cell
                self._rows[row["name"]] = widgets
            for key in ("psf", "background", "status"):
                widgets[key].setText(row[key])
