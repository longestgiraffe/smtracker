"""Track: find the tracker -- Fiji for TrackMate, or MATLAB and u-track -- then run it
condition by condition, one at a time.

THE SAME PAGE FOR BOTH TRACKERS. The switch at the top (also on Detect & link) decides
which engine card shows and which words the intro and the Replace box use; the condition
rows, the queue, Stop, the log and the finished banner are one set, filled by the window
from `prep_view.track_rows` or `prep_view.utrack_track_rows`.

THIN QT. Every word is `prep_view` (`fiji_lines`, `track_rows`, `progress_line`,
`run_header`, `finished_banner`); the run itself is `prep_model.track_condition`, which the window starts
through `PrepJobs.track`. This page only shows rows and carries requests.

ONE RUN AT A TIME, A QUEUE FOR THE REST. Each condition is a Fiji subprocess per movie,
about 30 s a movie, against one Fiji install -- so "Track all" queues, it does not loop
(the classic window's rule, for the same reason). STOP takes effect after the movie in
hand: Fiji is a subprocess, and killing it would leave a half-written XML where a reader
expects a whole one.

THE END OF A RUN IS SAID, NOT LEFT TO A LOG LINE. When the last queued condition has run,
a banner names what was tracked (and, first, anything that failed or was stopped), with a
Done button at its foot that closes the window and returns to the menu -- asked for after a
finished run read only "<condition> done" in small print.
"""
from __future__ import annotations

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from ...prep_model import MAX_AT_ONCE
from ...prep_view import TRACKERS
from ..theme import typography as ty
from ..widgets.controls import Button, Segmented
from ..widgets.primitives import Card, Checkbox, Progress, discard, icon, label
from ..widgets.settings import WARN_INK

__all__ = ["TrackPage"]


class TrackPage(QWidget):
    """The Fiji card, the condition rows and the run log. Signals carry requests."""

    track_requested = Signal(str)
    track_all_requested = Signal()
    stop_requested = Signal()
    overwrite_changed = Signal(bool)
    choose_fiji_requested = Signal()
    choose_matlab_requested = Signal()
    choose_utrack_requested = Signal()
    #: "trackmate" | "utrack": the switch was moved
    tracker_changed = Signal(str)
    #: "Movies at once", 1..`prep_model.MAX_AT_ONCE`
    at_once_changed = Signal(int)
    #: The finished banner's Done: close the window, back to the menu.
    done_requested = Signal()

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.setStyleSheet("background:transparent;")
        self._rows: dict = {}
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)
        top = QHBoxLayout()
        top.setSpacing(12)
        top.addWidget(label("Track", ty.ui("lg", "semibold"), t.foreground))
        top.addStretch(1)
        top.addWidget(label("Tracker", ty.ui("xs", "medium"), t.muted_foreground))
        self.tracker = Segmented(t, [words for _key, words in TRACKERS], current=0, size="sm")
        self.tracker.changed.connect(
            lambda i: self.tracker_changed.emit(TRACKERS[i][0]))
        top.addWidget(self.tracker)
        lay.addLayout(top)
        self.intro = label("", ty.ui("sm"), t.muted_foreground)
        self.intro.setWordWrap(True)
        lay.addWidget(self.intro)

        fiji = Card(t, shadowed=False)
        fl = QVBoxLayout(fiji)
        fl.setContentsMargins(14, 12, 14, 12)
        fl.setSpacing(8)
        self.engine_title = label("Fiji", ty.ui("sm", "semibold"), t.foreground)
        fl.addWidget(self.engine_title)
        self.fiji_text = label("looking for Fiji…", ty.ui("xs"), t.muted_foreground)
        self.fiji_text.setWordWrap(True)
        self.fiji_text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        fl.addWidget(self.fiji_text)
        self.utrack_text = label("looking for MATLAB and u-track…", ty.ui("xs"),
                                 t.muted_foreground)
        self.utrack_text.setWordWrap(True)
        self.utrack_text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        fl.addWidget(self.utrack_text)
        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        self._choose_fiji = Button(t, "Choose Fiji folder…", size="sm", glyph="folder-open")
        self._choose_fiji.clicked.connect(self.choose_fiji_requested.emit)
        buttons.addWidget(self._choose_fiji)
        self._choose_matlab = Button(t, "Choose matlab.exe…", size="sm", glyph="folder-open")
        self._choose_matlab.clicked.connect(self.choose_matlab_requested.emit)
        buttons.addWidget(self._choose_matlab)
        self._choose_utrack = Button(t, "Choose u-track folder…", size="sm",
                                     glyph="folder-open")
        self._choose_utrack.clicked.connect(self.choose_utrack_requested.emit)
        buttons.addWidget(self._choose_utrack)
        buttons.addStretch(1)
        fl.addLayout(buttons)
        lay.addWidget(fiji)

        card = Card(t, shadowed=False)
        self._grid = QGridLayout(card)
        self._grid.setContentsMargins(14, 12, 14, 12)
        self._grid.setHorizontalSpacing(12)
        self._grid.setVerticalSpacing(8)
        self._grid.setColumnStretch(2, 1)
        head = ty.ui("xs", "medium")
        for col, text in enumerate(("Condition", "Movies", "Status", "Progress")):
            self._grid.addWidget(label(text, head, t.muted_foreground), 0, col)
        lay.addWidget(card)

        foot = QHBoxLayout()
        foot.setSpacing(8)
        self.overwrite = Checkbox(t, checked=False)
        self.overwrite.toggled.connect(self.overwrite_changed.emit)
        foot.addWidget(self.overwrite)
        self.replace_label = label("", ty.ui("sm"), t.foreground)
        foot.addWidget(self.replace_label)
        foot.addStretch(1)
        self.at_once_label = label("Movies at once", ty.ui("xs", "medium"), t.muted_foreground)
        foot.addWidget(self.at_once_label)
        self.at_once = Segmented(t, [str(n) for n in range(1, MAX_AT_ONCE + 1)], current=0, size="sm")
        self.at_once.changed.connect(lambda i: self.at_once_changed.emit(i + 1))
        foot.addWidget(self.at_once)
        self._stop = Button(t, "Stop after this movie", size="sm")
        self._stop.clicked.connect(self.stop_requested.emit)
        self._stop.setVisible(False)
        foot.addWidget(self._stop)
        self._track_all = Button(t, "Track all ready conditions", variant="default")
        self._track_all.clicked.connect(self.track_all_requested.emit)
        foot.addWidget(self._track_all)
        lay.addLayout(foot)

        self.note = label("", ty.ui("xs", "medium"), t.foreground)
        self.note.setWordWrap(True)
        lay.addWidget(self.note)

        # The end of a run, said plainly; hidden until a run has ended.
        self.finished = Card(t, shadowed=False)
        fin = QVBoxLayout(self.finished)
        fin.setContentsMargins(16, 14, 16, 14)
        fin.setSpacing(6)
        title_row = QHBoxLayout()
        title_row.setSpacing(8)
        self._finished_icon_host = QHBoxLayout()
        self._finished_icon_host.setContentsMargins(0, 0, 0, 0)
        title_row.addLayout(self._finished_icon_host)
        self.finished_title = label("", ty.ui("base", "semibold"), t.foreground)
        title_row.addWidget(self.finished_title)
        title_row.addStretch(1)
        fin.addLayout(title_row)
        self._finished_lines = QVBoxLayout()
        self._finished_lines.setSpacing(2)
        fin.addLayout(self._finished_lines)
        done_row = QHBoxLayout()
        done_row.addStretch(1)
        self.done = Button(t, "Done", variant="default")
        self.done.setToolTip("Close this window and go back to the menu, where the "
                             "analysis is the next door.")
        self.done.clicked.connect(self.done_requested.emit)
        done_row.addWidget(self.done)
        fin.addLayout(done_row)
        lay.addWidget(self.finished)
        self.finished.setVisible(False)
        self._finished_widgets: list = []
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(ty.mono("xs"))
        self.log.setMinimumHeight(220)
        self.log.setStyleSheet(
            f"QPlainTextEdit {{ background:{t.card}; color:{t.foreground};"
            f" border:1px solid {t.border}; border-radius:{t.radius_md:g}px;"
            f" padding:6px; }}")
        lay.addWidget(self.log, 1)

        self.set_tracker("trackmate")

    # -- what the window tells the page ---------------------------------------- #
    def set_tracker(self, key: str) -> None:
        """Show ``key``'s engine and words ("trackmate" | "utrack")."""
        keys = [k for k, _w in TRACKERS]
        index = keys.index(key)
        if self.tracker.current() != index:
            self.tracker.select(index)
        utrack = key == "utrack"
        self.engine_title.setText("MATLAB and u-track" if utrack else "Fiji")
        self.fiji_text.setVisible(not utrack)
        self._choose_fiji.setVisible(not utrack)
        self.utrack_text.setVisible(utrack)
        self._choose_matlab.setVisible(utrack)
        self._choose_utrack.setVisible(utrack)
        if utrack:
            self.intro.setText(
                "u-track runs inside your MATLAB, one condition at a time, in as many sessions "
                "as Movies at once, on the ranges and calibration saved in crops.json. Each "
                "crop is written to tif/<movie>/<movie>/ and the tracks land in "
                "tif/<movie>/TrackingPackage/, where the analysis reads them. The settings "
                "are recorded in tif/utrack_run.json.")
            self.replace_label.setText("Replace an earlier u-track run in tif/")
            self.replace_label.setToolTip(
                "Off: a movie that already holds a u-track run (tif/<movie>/movieData.mat) "
                "is not tracked again. On: its frames, movieData.mat and tracks are "
                "replaced; u-track keeps the old movieData.mat in backups/.")
        else:
            self.intro.setText(
                "TrackMate runs inside Fiji, one condition at a time, on the ranges and "
                "calibration saved in each condition's crops.json. The tracks go to "
                "<condition>/trackmate/ beside a run.json that records the settings; the "
                "analysis reads them from there.")
            self.replace_label.setText("Replace an earlier run in trackmate/")
            self.replace_label.setToolTip(
                "Off: a condition whose trackmate/ already holds tracks is not tracked "
                "again, because two runs in one folder cannot be told apart afterwards.")

    def set_at_once(self, n: int, tip: str = "") -> None:
        """Show ``n`` movies at once, and what that means for the tracker in use."""
        if self.at_once.current() != n - 1:
            self.at_once.select(n - 1)
        for widget in (self.at_once_label, self.at_once):
            widget.setToolTip(tip)

    def set_fiji(self, lines: list) -> None:
        self.fiji_text.setText("\n".join(lines))

    def set_utrack(self, lines: list) -> None:
        self.utrack_text.setText("\n".join(lines))

    def set_rows(self, rows: list, *, busy: bool) -> None:
        """``rows``: `prep_view.track_rows`. ``busy`` while a run is in flight: a row's
        Track then QUEUES, and Stop is offered."""
        for row in rows:
            widgets = self._rows.get(row["name"]) or self._add_row(row["name"])
            widgets["movies"].setText(row["movies"])
            widgets["status"].setText(row["status"])
            progress = row.get("progress")
            widgets["bar"].setVisible(progress is not None)
            if progress is not None:
                done, total = progress
                widgets["bar"].set_value(done / total if total else 0.0)
            widgets["track"].setEnabled(bool(row["ready"]))
        self._stop.setVisible(busy)
        self.at_once.setEnabled(not busy)      # a run keeps the number it started with
        self._track_all.setEnabled(any(r["ready"] for r in rows))

    def _add_row(self, name: str) -> dict:
        t = self.t
        r = len(self._rows) + 1
        self._grid.addWidget(label(name, ty.ui("sm", "medium"), t.foreground), r, 0)
        movies = label("", ty.ui("xs", tabular=True), t.muted_foreground)
        self._grid.addWidget(movies, r, 1)
        status = label("", ty.ui("xs"), t.muted_foreground)
        status.setWordWrap(True)
        self._grid.addWidget(status, r, 2)
        bar = Progress(t, 0.0, width=110)
        self._grid.addWidget(bar, r, 3)
        bar.setVisible(False)
        track = Button(t, "Track", size="sm")
        track.clicked.connect(lambda n=name: self.track_requested.emit(n))
        self._grid.addWidget(track, r, 4)
        self._rows[name] = {"movies": movies, "status": status, "bar": bar,
                            "track": track}
        return self._rows[name]

    def set_note(self, text: str) -> None:
        self.note.setText(text)

    def show_finished(self, banner: dict) -> None:
        """``banner``: `prep_view.finished_banner` -- title, lines, tone."""
        self.hide_finished()
        warn = banner["tone"] != "ok"
        ink = WARN_INK if warn else self.t.foreground
        glyph = icon("alert" if warn else "check", 18, ink, stroke=2.0)
        self._finished_icon_host.addWidget(glyph)
        self.finished_title.setText(banner["title"])
        self.finished_title.setStyleSheet(f"color:{ink}; background:transparent;")
        widgets = [glyph]
        for i, line in enumerate(banner["lines"]):
            text = label(line, ty.ui("sm" if i == 0 else "xs"),
                         self.t.foreground if i == 0 else self.t.muted_foreground)
            text.setWordWrap(True)
            self._finished_lines.addWidget(text)
            widgets.append(text)
        self._finished_widgets = widgets
        self.finished.setVisible(True)

    def hide_finished(self) -> None:
        for widget in self._finished_widgets:
            discard(widget)
        self._finished_widgets = []
        self.finished.setVisible(False)

    def append(self, *lines: str) -> None:
        for line in lines:
            self.log.appendPlainText(line)
