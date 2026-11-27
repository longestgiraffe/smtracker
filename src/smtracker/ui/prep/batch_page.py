"""Batch: track every READY condition under several parent folders, overnight.

Asked for (13 Oct 2026): "a batch mode where u can run multiple parent folders in batches
overnight". RULED: a condition runs only where every movie has a SAVED crop range and the
calibration its run needs; anything missing is named in its row and skipped.

THIN QT. The shared page (`ui.widgets.batch.BatchPage`) plus this window's settings: the
tracker (the window's one switch, also on Detect & link and Track), Replace earlier runs,
for TrackMate whether a missing threshold may be measured inside the saved crops, for
the movies tracked at once (both trackers) and where u-track works. Every rule and word is
`smtracker.batch_model`; the window starts the run (`PrepJobs.batch_run`).
"""
from __future__ import annotations

from qtpy.QtCore import Qt, Signal

from ...batch_model import DEFAULT_SESSIONS, MAX_SESSIONS
from ...prep_view import TRACKERS
from ..theme import typography as ty
from ..widgets.batch import BatchPage
from ..widgets.controls import Segmented
from ..widgets.primitives import Checkbox, label

__all__ = ["TrackBatchPage", "TRACK_COLUMNS"]

TRACK_COLUMNS = ("Folder", "Condition", "Layout", "Movies", "Status")


class TrackBatchPage(BatchPage):
    """The tracking batch: the shared page and its settings row."""

    #: "trackmate" | "utrack"
    tracker_changed = Signal(str)
    replace_changed = Signal(bool)
    measure_changed = Signal(bool)
    #: Movies at once, 1..`MAX_SESSIONS` (the Track page's "Movies at once", one number for both)
    sessions_changed = Signal(int)

    def __init__(self, t, parent=None):
        super().__init__(
            t, title="Batch", columns=TRACK_COLUMNS, run_text="Run batch",
            intro=("Tracks every condition under several parent folders, one after the other. "
                   "A condition runs only if every movie has a saved crop range and the "
                   "calibration its run needs; otherwise it is skipped and its row says what "
                   "is missing. A report (smtracker_batch_<time>.csv) is written into the first "
                   "parent folder."),
            parent=parent)
        s = self.settings
        self.tracker = Segmented(t, [words for _key, words in TRACKERS], current=0, size="sm")
        self.tracker.changed.connect(lambda i: self.tracker_changed.emit(TRACKERS[i][0]))
        s.addLayout(self.group(label("Tracker", ty.ui("xs", "medium"), t.muted_foreground),
                               self.tracker))
        self.replace = Checkbox(t, checked=False)
        self.replace.toggled.connect(self.replace_changed.emit)
        s.addLayout(self.group(self.replace,
                               label("Replace earlier runs", ty.ui("sm"), t.foreground)))
        from ...batch_model import MEASURE_MISSING_THRESHOLD

        self.measure = Checkbox(t, checked=MEASURE_MISSING_THRESHOLD)
        self.measure.toggled.connect(self.measure_changed.emit)
        self.measure_label = label("Measure a missing threshold inside the saved crops",
                                   ty.ui("sm"), t.foreground)
        self.measure_label.setToolTip(
            "A freshly opened folder has no TrackMate threshold. Ticked: the batch measures "
            "one inside each condition's saved crops, using the multiplier in Options. "
            "Unticked: such a condition is skipped and named.")
        # `label` lets the mouse through, which would leave the tooltip unreachable.
        self.measure_label.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        s.addLayout(self.group(self.measure, self.measure_label))
        self.sessions_label = label("Movies at once", ty.ui("xs", "medium"),
                                    t.muted_foreground)
        self.sessions = Segmented(t, [str(n) for n in range(1, MAX_SESSIONS + 1)],
                                  current=DEFAULT_SESSIONS - 1, size="sm")
        self.sessions.changed.connect(lambda i: self.sessions_changed.emit(i + 1))
        s.addLayout(self.group(self.sessions_label, self.sessions))
        s.addStretch(1)
        self.set_tracker("trackmate")

    def set_tracker(self, key: str) -> None:
        """Show ``key``'s settings: the threshold box for TrackMate; movies at once for both."""
        keys = [k for k, _w in TRACKERS]
        index = keys.index(key)
        if self.tracker.current() != index:
            self.tracker.select(index)
        utrack = key == "utrack"
        for widget in (self.measure, self.measure_label):
            widget.setVisible(not utrack)

    def set_at_once(self, n: int, tip: str = "") -> None:
        """Show ``n`` movies at once (the window keeps one number for Track and Batch)."""
        if self.sessions.current() != n - 1:
            self.sessions.select(n - 1)
        for widget in (self.sessions_label, self.sessions):
            widget.setToolTip(tip)

    def set_busy(self, busy: bool, *, can_run: bool = True) -> None:
        super().set_busy(busy, can_run=can_run)
        for widget in (self.tracker, self.replace, self.measure, self.sessions):
            widget.setEnabled(not busy)

    def set_measure_k(self, k: float) -> None:
        self.measure_label.setText(
            f"Measure a missing threshold inside the saved crops (k = {float(k):g})")
