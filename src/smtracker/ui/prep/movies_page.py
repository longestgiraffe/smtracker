"""Movies & crop: the movie list, the crop chart, and the controls that set ranges.

THIN QT. The page shows what the window hands it -- `prep_view.movie_rows`,
`prep_view.crop_chart`, `prep_view.floor_note` -- and reports what was asked for through
signals. It holds no session and decides nothing; the window files every request into
`prep_model` and hands the page the result.
"""
from __future__ import annotations

from qtpy.QtCore import QRectF, Qt, QTimer, Signal
from qtpy.QtGui import QColor, QPainter
from qtpy.QtWidgets import (
    QHBoxLayout,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ...prep_guide import CROP_KEYS, NAMES
from ...prep_view import START_RULES
from ..theme import typography as ty
from ..widgets.controls import Button, FloatField, NumberField, Segmented
from ..widgets.primitives import Card, Surface, discard, label, mix, scrollbar_css
from .crop_chart import CropChart, chart_colours
from .guide import GuidePanel

__all__ = ["MoviesPage"]

#: How long the start/end fields wait after the last keystroke before committing, so
#: typing "1500" is one edit rather than four.
EDIT_SETTLE_MS = 450


class _MovieRow(Surface):
    """One movie in the list: a status dot, the name, its range and its state."""

    def __init__(self, t, row: dict, on_click, parent=None):
        super().__init__(t, radius=t.radius_md, shadowed=False, parent=parent)
        self.t = t
        self.movie = row["movie"]
        self._on_click = on_click
        self._selected = False
        self._status = row["status"]
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(46)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(26, 5, 8, 5)
        lay.setSpacing(1)
        self._name = label(row["movie"], ty.ui("sm", "medium"), t.foreground)
        self._line = label("", ty.ui("xs", tabular=True), t.muted_foreground)
        lay.addWidget(self._name)
        lay.addWidget(self._line)
        self.update_row(row)

    def update_row(self, row: dict) -> None:
        self._status = row["status"]
        unsaved = " · unsaved" if row["unsaved"] else ""
        self._line.setText(f"{row['range']}{unsaved}   {row['detail']}")
        self._line.setToolTip(row["detail"] if row["status"] == "error" else "")
        self.update()

    def set_selected(self, on: bool) -> None:
        self._selected = on
        self._restyle()

    def _restyle(self) -> None:
        fill = self.t.accent if self._selected else "transparent"
        self.setStyleSheet(f"background:{fill}; border:none;"
                           f" border-radius:{self.t.radius_md:g}px;")

    def paintEvent(self, e) -> None:
        super().paintEvent(e)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        colours = chart_colours(self.t)
        fill = {"read": colours["proposal"], "error": colours["floor"]}.get(
            self._status, mix(self.t.muted_foreground, self.t.background, 0.5))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(fill))
        p.drawEllipse(QRectF(9, self.height() / 2 - 4, 8, 8))
        p.end()

    def mouseReleaseEvent(self, e) -> None:
        if e.button() == Qt.LeftButton:
            self._on_click(self.movie)
        super().mouseReleaseEvent(e)


class MoviesPage(QWidget):
    """The page. Signals carry requests; nothing here changes a range on its own."""

    movie_selected = Signal(str)
    rule_changed = Signal(str)
    tolerance_changed = Signal(float)
    propose_all = Signal()
    #: "Propose for all conditions" (23 Oct 2026); ``propose_all`` is this condition's.
    propose_every = Signal()
    propose_one = Signal(str)
    common_start = Signal(int)
    common_end = Signal(int)
    #: "Track N frames": every movie ends N frames after its own start (23 Oct 2026).
    common_length = Signal(int)
    #: movie, start, end -- from the chart's drag or the two fields.
    crop_edited = Signal(str, int, int)
    #: the crop guide's "Measure from the tracks here", for the tail
    purity_requested = Signal()

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.setStyleSheet("background:transparent;")
        self._rows: dict = {}
        self._order: list = []
        self._current: str | None = None
        self._n_frames: int | None = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)

        # -- head: the condition and how far it has got -------------------------- #
        head = QHBoxLayout()
        head.setSpacing(12)
        titles = QVBoxLayout()
        titles.setSpacing(2)
        self.title = label("", ty.ui("lg", "semibold"), t.foreground)
        self.status = label("", ty.ui("sm"), t.muted_foreground)
        titles.addWidget(self.title)
        titles.addWidget(self.status)
        head.addLayout(titles, 1)
        self._propose_all = Button(t, "Propose for this condition", variant="default")
        self._propose_all.clicked.connect(self.propose_all.emit)
        head.addWidget(self._propose_all, alignment=Qt.AlignTop)
        self._propose_every = Button(t, "Propose for all conditions", variant="outline")
        self._propose_every.setToolTip("Every movie of every condition, each condition at its own "
                                       "'Flat within' value.")
        self._propose_every.clicked.connect(self.propose_every.emit)
        head.addWidget(self._propose_every, alignment=Qt.AlignTop)
        lay.addLayout(head)

        # -- the rule, the tolerance, and the two common cuts --------------------- #
        controls = Card(t, shadowed=False)
        cl = QVBoxLayout(controls)
        cl.setContentsMargins(14, 10, 14, 10)
        cl.setSpacing(8)
        first = QHBoxLayout()
        first.setSpacing(10)
        first.addWidget(label("Start rule", ty.ui("sm", "medium"), t.muted_foreground))
        self.rule = Segmented(t, [lab for _k, lab in START_RULES], current=0, size="sm")
        self.rule.changed.connect(lambda i: self.rule_changed.emit(START_RULES[i][0]))
        first.addWidget(self.rule)
        first.addSpacing(12)
        first.addWidget(label("Flat within", ty.ui("sm", "medium"), t.muted_foreground))
        self.flatness = FloatField(t, 0.1, 25.0, 2.0, step=0.05, decimals=2,
                                   suffix=" %", width=128)
        self.flatness.setToolTip(
            "How far the frame mean may vary inside a stretch still called flat, as a "
            "percentage of its level; the floor is where the longest such stretch begins. "
            "Each condition has its own value, capped at 15 % of the signal above the "
            "camera's dark level. A typed value applies once you pause.")
        # ONCE TYPING PAUSES, not per key (23 Oct 2026: "its slow to type a number in this
        # field, maybe first finish typing, then compute"): 600 ms after the last change, or
        # at once on Enter / leaving the field.
        self._flat_timer = QTimer(self)
        self._flat_timer.setSingleShot(True)
        self._flat_timer.setInterval(600)
        self._flat_timer.timeout.connect(self._flat_settled)
        self.flatness.changed.connect(lambda _v: self._flat_timer.start())
        self.flatness._box.editingFinished.connect(self._flat_settled)
        first.addWidget(self.flatness)
        #: why this condition reads at this value: chosen for its camera offset, or typed
        self.flat_why = label("", ty.ui("xs"), t.muted_foreground)
        first.addWidget(self.flat_why)
        first.addStretch(1)
        cl.addLayout(first)
        second = QHBoxLayout()
        second.setSpacing(10)
        second.addWidget(label("Every movie starts at", ty.ui("sm", "medium"),
                               t.muted_foreground))
        self.start_all = NumberField(t, 1, 10_000_000, 500, width=140)
        second.addWidget(self.start_all)
        cut_start = Button(t, "Cut all starts", size="sm")
        cut_start.clicked.connect(lambda: self.common_start.emit(self.start_all.value()))
        second.addWidget(cut_start)
        second.addSpacing(18)
        second.addWidget(label("ends at", ty.ui("sm", "medium"), t.muted_foreground))
        self.end_all = NumberField(t, 2, 10_000_000, 4000, width=140)
        second.addWidget(self.end_all)
        cut_end = Button(t, "Cut all tails", size="sm")
        cut_end.clicked.connect(lambda: self.common_end.emit(self.end_all.value()))
        second.addWidget(cut_end)
        second.addStretch(1)
        cl.addLayout(second)
        # ONE LENGTH FROM EACH MOVIE'S OWN START (asked 23 Oct 2026): "track the first x
        # frames after the cut range begins" -- a movie starting at 400 ends at 1399 for 1000.
        third = QHBoxLayout()
        third.setSpacing(10)
        third.addWidget(label("or every movie tracks", ty.ui("sm", "medium"), t.muted_foreground))
        self.length_all = NumberField(t, 2, 10_000_000, 1000, width=140)
        third.addWidget(self.length_all)
        third.addWidget(label("frames from its own start", ty.ui("sm", "medium"),
                              t.muted_foreground))
        track_n = Button(t, "Track N frames", size="sm")
        track_n.setToolTip("Every movie keeps its start and ends this many frames later; a "
                           "movie too short for it ends at its last frame and is listed.")
        track_n.clicked.connect(lambda: self.common_length.emit(self.length_all.value()))
        third.addWidget(track_n)
        third.addStretch(1)
        cl.addLayout(third)
        lay.addWidget(controls)
        # FOLDED BY DEFAULT: the chart is what this page is for, and an open guide takes
        # its height. The fold is remembered.
        self.guide = GuidePanel(t, CROP_KEYS, NAMES, remember_as="prep_guide_crop",
                                default_open=False, art_height=170)
        lay.addWidget(self.guide)
        measure = Button(t, "Measure from the tracks here", size="sm")
        measure.setToolTip("Reads this condition's tracks and outlines: per block of frames, "
                           "the background's share of the in-cell rate and how much data "
                           "the block holds. A few seconds.")
        measure.clicked.connect(self.purity_requested.emit)
        self.guide.add_action("end", measure)

        # -- body: the list, and the chart with its edge fields ------------------- #
        body = QHBoxLayout()
        body.setSpacing(14)
        self._list_host = QWidget()
        self._list_host.setStyleSheet("background:transparent;")
        self._list_lay = QVBoxLayout(self._list_host)
        self._list_lay.setContentsMargins(0, 0, 6, 0)
        self._list_lay.setSpacing(2)
        self._list_lay.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidget(self._list_host)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { background:transparent; border:none; }"
                             + scrollbar_css(t))
        scroll.setFixedWidth(250)
        body.addWidget(scroll)

        right = QVBoxLayout()
        right.setSpacing(8)
        self.chart = CropChart(t)
        self.chart.setMinimumHeight(420)
        self.chart.crop_moving.connect(self._chart_moving)
        self.chart.crop_committed.connect(self._chart_committed)
        right.addWidget(self.chart, 1)
        edges = QHBoxLayout()
        edges.setSpacing(10)
        edges.addWidget(label("This movie keeps", ty.ui("sm", "medium"), t.muted_foreground))
        self.start = NumberField(t, 1, 10_000_000, 1, width=140)
        self.end = NumberField(t, 2, 10_000_000, 2, width=140)
        edges.addWidget(self.start)
        edges.addWidget(label("to", ty.ui("sm"), t.muted_foreground))
        edges.addWidget(self.end)
        self._propose_one = Button(t, "Propose", size="sm")
        self._propose_one.clicked.connect(
            lambda: self._current and self.propose_one.emit(self._current))
        edges.addWidget(self._propose_one)
        edges.addStretch(1)
        right.addLayout(edges)
        #: The floor line for the movie on screen (redrawn with every movie) ...
        self.floor = label("", ty.ui("xs"), t.muted_foreground)
        self.floor.setWordWrap(True)
        right.addWidget(self.floor)
        #: ... and, apart from it, the result of the last action, which a redraw must
        #: not overwrite: "refused", "cut in 2 other conditions".
        self.note = label("", ty.ui("xs", "medium"), t.foreground)
        self.note.setWordWrap(True)
        right.addWidget(self.note)
        body.addLayout(right, 1)
        lay.addLayout(body, 1)

        #: The fields commit once they have settled (see `EDIT_SETTLE_MS`).
        self._settle = QTimer(self)
        self._settle.setSingleShot(True)
        self._settle.setInterval(EDIT_SETTLE_MS)
        self._settle.timeout.connect(self._fields_committed)
        self._quiet = False
        self.start.changed.connect(self._field_changed)
        self.end.changed.connect(self._field_changed)

    # -- what the window hands in ------------------------------------------------ #
    def set_condition(self, title: str, status: str) -> None:
        self.title.setText(title)
        self.status.setText(status)

    def set_rows(self, rows: list) -> None:
        """Update the list in place: rows are added and updated, never rebuilt, so the
        list does not flicker as each movie arrives."""
        wanted = [r["movie"] for r in rows]
        for stem in [s for s in self._rows if s not in wanted]:
            discard(self._rows.pop(stem))
        for i, row in enumerate(rows):
            widget = self._rows.get(row["movie"])
            if widget is None:
                widget = _MovieRow(self.t, row, self._select)
                self._list_lay.insertWidget(i, widget)
                self._rows[row["movie"]] = widget
            else:
                widget.update_row(row)
        self._order = wanted
        for stem, widget in self._rows.items():
            widget.set_selected(stem == self._current)

    def set_chart(self, payload: dict | None, *, message: str = "") -> None:
        if payload is None:
            self.chart.set_message(message or "Choose a movie.")
            self._n_frames = None
            return
        self.chart.set_payload(payload)
        self._n_frames = int(payload["n_frames"])
        start, end = payload["crop"] or (1, self._n_frames)
        self._set_fields(start, end)

    def set_note(self, text: str) -> None:
        """The result of the last action. Kept until the next one."""
        self.note.setText(text)

    def set_floor(self, text: str) -> None:
        """The floor line of the movie on screen."""
        self.floor.setText(text)

    def select(self, stem: str | None) -> None:
        self._current = stem
        for s, widget in self._rows.items():
            widget.set_selected(s == stem)

    def current(self) -> str | None:
        return self._current

    def set_tolerance(self, percent: float, note: str = "") -> None:
        """The box at the condition on screen's value, and why, without sending a change."""
        self._flat_timer.stop()
        self.flatness.blockSignals(True)
        self.flatness.setValue(percent)
        self.flatness.blockSignals(False)
        self._flat_sent = round(float(percent), 6)
        self.flat_why.setText("automatic" if note.startswith("set to") else
                              "typed" if note.startswith("typed") else "")
        self.flat_why.setToolTip(note)

    def _flat_settled(self) -> None:
        """Typing paused or ended: send the value once, if it moved."""
        self._flat_timer.stop()
        value = round(float(self.flatness.value()), 6)
        if value != getattr(self, "_flat_sent", None):
            self._flat_sent = value
            self.tolerance_changed.emit(value)

    # -- internals --------------------------------------------------------------- #
    def _select(self, stem: str) -> None:
        if stem != self._current:
            self.select(stem)
            self.movie_selected.emit(stem)

    def _set_fields(self, start: int, end: int) -> None:
        self._quiet = True
        self.start.setValue(int(start))
        self.end.setValue(int(end))
        self._quiet = False

    def _chart_moving(self, start: int, end: int) -> None:
        self._set_fields(start, end)

    def _chart_committed(self, start: int, end: int) -> None:
        if self._current is not None:
            self.crop_edited.emit(self._current, int(start), int(end))

    def _field_changed(self, _value) -> None:
        if self._quiet or self._current is None:
            return
        self.chart.set_edges(self.start.value(), self.end.value())
        self._settle.start()

    def _fields_committed(self) -> None:
        if self._current is not None:
            self.crop_edited.emit(self._current, self.start.value(), self.end.value())
