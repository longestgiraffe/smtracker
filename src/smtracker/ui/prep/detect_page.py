"""Detect & link: per condition, the LoG threshold (measured from its own noise), the
spot radius (derived from its own optics), the linking radius and the frame gap -- or,
with u-track as the tracker, u-track's own settings (`utrack_card.UTrackCard`).

ONE TRACKER SWITCH, SHOWN ON THIS PAGE AND ON TRACK. Both pages hold a `Segmented`
reporting `tracker_changed`; the window keeps the one value and sets both, so the two
cannot disagree about which settings a run will use.

THIN QT. Cells show `prep_view.tracking_cells`, the status `prep_view.tracking_status`,
and the lines below the grid -- what the radius and gap mean for this acquisition, how
crowded the field is, the noise movie by movie -- are `prep_view`'s. An edit is reported
as typed and filed by the window through `prep_model.set_tracking`.
"""
from __future__ import annotations

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QGridLayout, QHBoxLayout, QVBoxLayout, QWidget

from ...chart_text import fold_words
from ...prep_guide import NAMES, TRACKING_KEYS
from ...prep_model import RADIUS_CHECK_MULTIPLE
from ...prep_view import TRACKERS, TRACKING_COLUMNS
from ..theme import typography as ty
from ..widgets.controls import Button, Segmented
from ..widgets.primitives import Card, label
from .calibration_page import Field
from .guide import GuidePanel
from .utrack_card import UTrackCard

__all__ = ["DetectPage"]


class DetectPage(QWidget):
    """The grid and the lines. Signals carry requests; the window answers them."""

    #: condition, field key, the text as typed
    edited = Signal(str, str, str)
    measure_requested = Signal(str)
    measure_all_requested = Signal()
    copy_requested = Signal(str)
    classic_requested = Signal()
    #: condition, setting -- a cell was clicked into, so the guide shows that setting
    #: for that condition.
    focus = Signal(str, str)
    #: the linking radius's two helpers, for the condition the guide shows
    pilot_requested = Signal()
    radius_check_requested = Signal()
    #: "trackmate" | "utrack": the switch was moved
    tracker_changed = Signal(str)

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.setStyleSheet("background:transparent;")
        self._rows: dict = {}
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(12)
        top = QHBoxLayout()
        top.setSpacing(12)
        top.addWidget(label("Detect & link", ty.ui("lg", "semibold"), t.foreground))
        top.addStretch(1)
        top.addWidget(label("Tracker", ty.ui("xs", "medium"), t.muted_foreground))
        self.tracker = Segmented(t, [words for _key, words in TRACKERS], current=0, size="sm")
        self.tracker.changed.connect(
            lambda i: self.tracker_changed.emit(TRACKERS[i][0]))
        top.addWidget(self.tracker)
        outer.addLayout(top)
        #: u-track's settings, shown instead of TrackMate's when u-track is the tracker.
        self.utrack = UTrackCard(t)
        outer.addWidget(self.utrack)
        #: Everything below is TrackMate's.
        self.trackmate = QWidget()
        self.trackmate.setStyleSheet("background:transparent;")
        lay = QVBoxLayout(self.trackmate)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)
        outer.addWidget(self.trackmate)
        intro = label(
            "Each movie is detected at a threshold measured from its condition's noise "
            "inside the crops (the cell shows the median). The spot radius follows the "
            "optics. Linking defaults to 5 px with a gap of 3 frames. A value you type "
            "applies to every movie.", ty.ui("sm"), t.muted_foreground)
        intro.setWordWrap(True)
        lay.addWidget(intro)

        card = Card(t, shadowed=False)
        self._grid = QGridLayout(card)
        self._grid.setContentsMargins(14, 12, 14, 12)
        self._grid.setHorizontalSpacing(10)
        self._grid.setVerticalSpacing(8)
        head = ty.ui("xs", "medium")
        self._grid.addWidget(label("Condition", head, t.muted_foreground), 0, 0)
        for col, (_key, heading, _unit) in enumerate(TRACKING_COLUMNS, start=1):
            self._grid.addWidget(label(heading, head, t.muted_foreground), 0, col)
        self._grid.addWidget(label("Status", head, t.muted_foreground), 0,
                             len(TRACKING_COLUMNS) + 1)
        lay.addWidget(card)

        foot = QHBoxLayout()
        foot.setSpacing(10)
        copy = Button(t, "Copy linking radius and gap to all", size="sm")
        copy.setToolTip("Copies the linking radius and gap only. The threshold and the spot "
                        "radius stay specific to each condition.")
        copy.clicked.connect(
            lambda: self._rows and self.copy_requested.emit(next(iter(self._rows))))
        foot.addWidget(copy)
        foot.addStretch(1)
        classic = Button(t, "Classic window", size="sm", glyph="arrow-right")
        classic.setToolTip("Open the classic tracking window on this folder, with the "
                           "ranges you saved.")
        classic.clicked.connect(self.classic_requested.emit)
        foot.addWidget(classic)
        self._measure_all = Button(t, "Measure all thresholds", variant="default")
        self._measure_all.clicked.connect(self.measure_all_requested.emit)
        foot.addWidget(self._measure_all)
        lay.addLayout(foot)

        self.note = label("", ty.ui("xs", "medium"), t.foreground)
        self.note.setWordWrap(True)
        # The note speaks for either tracker, so it sits above both settings blocks.
        outer.insertWidget(1, self.note)
        self.guide = GuidePanel(t, TRACKING_KEYS, NAMES, remember_as="prep_guide_detect",
                                default_open=True)
        lay.addWidget(self.guide)
        measure = Button(t, "Measure from tracks here", size="sm")
        measure.setToolTip("Reads the step lengths of tracks already in this condition and "
                           "how often a second molecule lies within the radius. No D is "
                           "needed.")
        measure.clicked.connect(self.pilot_requested.emit)
        self.guide.add_action("linking_max_distance_px", measure)
        wider = Button(t, "Check a wider radius (about 2 min)", size="sm")
        wider.setToolTip(f"Tracks two movies at this radius and at {fold_words(RADIUS_CHECK_MULTIPLE, of=True)} it, "
                         "everything else identical, and counts the steps the narrow "
                         "radius could not see.")
        wider.clicked.connect(self.radius_check_requested.emit)
        self.guide.add_action("linking_max_distance_px", wider)
        self.lines = label("", ty.ui("xs", tabular=True), t.muted_foreground)
        self.lines.setWordWrap(True)
        self.lines.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.lines)
        outer.addStretch(1)
        self.set_tracker("trackmate")

    def set_tracker(self, key: str) -> None:
        """Show ``key``'s settings ("trackmate" | "utrack") and move the switch to it."""
        keys = [k for k, _w in TRACKERS]
        index = keys.index(key)
        if self.tracker.current() != index:
            self.tracker.select(index)
        self.utrack.setVisible(key == "utrack")
        self.trackmate.setVisible(key != "utrack")

    def set_rows(self, rows: list) -> None:
        """``rows``: ``[{"name", "cells": {key: text}, "status", "measuring"}]``."""
        for row in rows:
            widgets = self._rows.get(row["name"]) or self._add_row(row["name"])
            for key, field in widgets["fields"].items():
                field.set_text(row["cells"].get(key, ""))
            widgets["status"].setText(row["status"])
            widgets["measure"].setEnabled(not row.get("measuring"))

    def _add_row(self, name: str) -> dict:
        t = self.t
        r = len(self._rows) + 1
        self._grid.addWidget(label(name, ty.ui("sm", "medium"), t.foreground), r, 0)
        fields = {}
        for col, (key, _heading, unit) in enumerate(TRACKING_COLUMNS, start=1):
            field = Field(t, numeric=True, unit=unit, width=120)
            field.committed.connect(lambda text, k=key, n=name: self.edited.emit(n, k, text))
            field.focused.connect(lambda k=key, n=name: self.focus.emit(n, k))
            self._grid.addWidget(field, r, col)
            fields[key] = field
        status = label("", ty.ui("xs"), t.muted_foreground)
        self._grid.addWidget(status, r, len(TRACKING_COLUMNS) + 1)
        measure = Button(t, "Measure", size="sm")
        measure.clicked.connect(lambda n=name: self.measure_requested.emit(n))
        self._grid.addWidget(measure, r, len(TRACKING_COLUMNS) + 2)
        self._rows[name] = {"fields": fields, "status": status, "measure": measure}
        return self._rows[name]

    def set_note(self, text: str) -> None:
        self.note.setText(text)

    def set_lines(self, lines: list) -> None:
        self.lines.setText("\n".join(lines))
