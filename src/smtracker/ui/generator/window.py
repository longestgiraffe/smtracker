"""The fourth stage: simulated data (25 Nov 2026, steps 3 and 6 of `docs/generator_plan.md`).

Asked: "a generator would be cool. where u can set how many diffusive populations, which D, etc, which cell size
etc.", and "a fourth separate openable window after launching smtracker"; then "can we make it a bit more
beautfil? maybe multiple pages clickable next iwth animations explaning stuff and some helpers?".

A rail of PAGES (`generator_preview.PAGES`), each its settings beside a live picture:
  Start              examples to begin from, tracks or movies, recipes; the set as it stands
  Populations        each condition's populations, D and share; typical values; the bubbles
  Where in the cell  each population's place and the places' sizes; an ANIMATED cell, moved by the generator's own
                     engine on a background thread, its timer running only while the page is shown
  Switching          the switching and its rates; the state diagram with each population's dwell
  Kinetics           a change after an addition; the response, the movies on its clock, before and after
  Cells and field    the cells and the field; the field drawn at its pixel's scale, or why the cells do not fit
  Camera and movie   the frame interval, the movie and the camera; one rendered frame, the expected precision,
                     and a D past the linking limit
  Build              the replicates, the name, the folder, Build with its progress and log, Open in ...; every
                     problem with the page it is put right on
The condition pages share the condition picker above them.

THIN QT: every field, word, range and editing rule is `smtracker.generator_view`; which page shows what, the
problems' pages and every picture's numbers are `smtracker.generator_preview`; the pictures are painted by
`ui.generator.pictures`; the build is `smtracker.generator.build`. The build and the two slow pictures (the cell and
the frame) run on background threads (never the process pool), their results sent back by signal.
"""
from __future__ import annotations

import copy
import json
import threading
import time
from pathlib import Path

from qtpy.QtCore import QObject, Qt, QTimer, Signal
from qtpy.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ... import generator as gen
from ... import generator_preview as gp
from ... import generator_view as gv
from ...run_progress import progress_payload
from .. import theme
from ..theme import typography as ty
from ..widgets.controls import Button, FloatField, NumberField, Segmented
from ..widgets.options_dialog import _ChoiceBox
from ..widgets.primitives import Card, Checkbox, FitScroll, activity_label, discard, label, scrollbar_css
from ..widgets.progress import RunProgress
from ..widgets.settings import CUT_INK, WARN_INK
from ..workspace.sidebar import Sidebar, SidebarTrigger
from .pictures import BubblePicture, CellPicture, CurvePicture, DiagramPicture, FieldPicture, FramePicture

__all__ = ["GeneratorWindow", "open_generator_window", "STATE_KEY", "FOLDER_KEY", "PRESETS_KEY"]

#: The form as last built, remembered per user (`theme.remember`), and the folder sets were last written into.
STATE_KEY = "generator_state"
FOLDER_KEY = "generator_parent"
#: The user's own movie presets (`generator_view.add_preset`): {name: the movie fields}, remembered per user.
PRESETS_KEY = "generator_presets"
#: The picture column's width, and the least it shrinks to on a small screen.
PIC_W = 420
PIC_MIN_W = 320
#: A slow picture (the cell, the frame) is drawn this long after the last change.
DEBOUNCE_MS = 400
_OPEN_WINDOWS: set = set()
#: The pictures drawn on a background thread: the page each is on, and what draws it. The field is among them
#: because a field too small for its cells is found only after the placer's many tries (seconds).
SLOW_PICTURES = {"cell": ("where", lambda state, condition: gp.cell_animation(state, condition)),
                 "frame": ("camera", lambda state, condition: gp.frame_preview(state, condition)),
                 "field": ("cells", lambda state, _condition: gp.field_layout(state))}


def _line_edit(t, text: str, *, width: int | None = None, placeholder: str = "") -> QLineEdit:
    edit = QLineEdit(str(text))
    edit.setFont(ty.ui("sm"))
    edit.setPlaceholderText(placeholder)
    edit.setStyleSheet(f"QLineEdit {{ background:{t.card}; color:{t.foreground}; border:1px solid {t.input};"
                       f" border-radius:{t.radius_md:g}px; padding:3px 8px; min-height:{t.control_height - 8}px; }}")
    if width:
        edit.setFixedWidth(width)
    return edit


def _float(t, value: float, low: float, high: float, decimals: int, width: int = 112) -> FloatField:
    step = 1.0 if decimals <= 0 else 0.5 if decimals == 1 else 10.0 ** (1 - decimals)
    return FloatField(t, float(low), float(high), float(value), step=step, decimals=decimals, width=width)


def _swatch(colour: str) -> QWidget:
    dot = QWidget()
    dot.setFixedSize(12, 12)
    dot.setStyleSheet(f"background:{colour}; border-radius:6px;")
    return dot


def _words(text: str, t, *, size: str = "xs", colour: str | None = None, weight: str = "normal"):
    words = label(text, ty.ui(size, weight), colour or t.muted_foreground)
    words.setWordWrap(True)
    return words


class _BuildSignals(QObject):
    line = Signal(str)
    progress = Signal(int, int)
    done = Signal(object)
    failed = Signal(str)


class _PreviewSignals(QObject):
    ready = Signal(str, int, object)


class _ConditionParts(QObject):
    """One condition's form, in the parts the condition pages show: the populations, their places, the switching,
    the kinetics. Values are written into ``cond`` as they change (``edited``); a change of shape asks for the
    pages to be drawn again (``reshape``)."""

    edited = Signal()
    reshape = Signal()

    def __init__(self, t, cond: dict, movies_per_replicate, parent=None):
        super().__init__(parent)
        self.t, self.cond = t, cond
        self._movies = movies_per_replicate
        self.share_sum = None

    def _names(self) -> list:
        return [p["name"] or f"population {i + 1}" for i, p in enumerate(self.cond["populations"])]

    # -- populations ------------------------------------------------------------- #
    def populations_widget(self) -> QWidget:
        t, cond = self.t, self.cond
        box = QWidget()
        grid = QGridLayout(box)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        matrix = cond["switching"] == "matrix"
        heads = ("", "Population", "D (µm²/s)", "Share" if not matrix else "Share (from the rates)")
        for col, words in enumerate(heads):
            grid.addWidget(label(words, ty.ui("xs", "medium"), t.muted_foreground), 0, col)
        for i, pop in enumerate(cond["populations"]):
            name = _line_edit(t, pop["name"], width=140, placeholder=f"population {i + 1}")
            name.textEdited.connect(lambda text, p=pop: self._set(p, "name", text))
            d = _float(t, pop["D"], 0.0, 100.0, 3)
            d.changed.connect(lambda v, p=pop: self._set(p, "D", v))
            share = _float(t, pop["share"], 0.0, 1.0, 3)
            share.changed.connect(lambda v, p=pop: self._set(p, "share", v))
            share.setEnabled(not matrix)
            remove = Button(t, "Remove", size="sm", variant="ghost")
            remove.setEnabled(len(cond["populations"]) > 1)
            remove.clicked.connect(lambda _=False, j=i: self._remove(j))
            grid.addWidget(_swatch(gp.POP_COLOURS[i % len(gp.POP_COLOURS)]), i + 1, 0, alignment=Qt.AlignCenter)
            for col, w in enumerate((name, d, share, remove), start=1):
                grid.addWidget(w, i + 1, col)
        row = len(cond["populations"]) + 1
        add = Button(t, "Add population", size="sm", glyph="plus")
        add.setEnabled(len(cond["populations"]) < gen.MAX_POPULATIONS)
        add.clicked.connect(self._add)
        grid.addWidget(add, row, 1, 1, 2, alignment=Qt.AlignLeft)
        self.share_sum = label("", ty.ui("xs", tabular=True), t.muted_foreground)
        grid.addWidget(self.share_sum, row, 3, 1, 2, alignment=Qt.AlignLeft)
        grid.setColumnStretch(5, 1)
        return box

    def share_words(self) -> tuple:
        """The shares' total and whether it is 1 (a matrix condition's shares come from its rates)."""
        if self.cond["switching"] == "matrix":
            return "Shares follow from the rates.", True
        total = sum(float(p["share"]) for p in self.cond["populations"])
        return f"The shares add up to {total:.3f}", abs(total - 1.0) <= 1e-6

    def _set(self, target: dict, key: str, value) -> None:
        target[key] = value
        self.edited.emit()

    def _add(self) -> None:
        gv.add_population(self.cond)
        self.reshape.emit()

    def _remove(self, index: int) -> None:
        gv.remove_population(self.cond, index)
        self.reshape.emit()

    # -- places --------------------------------------------------------------------- #
    def where_widget(self) -> QWidget:
        t, cond = self.t, self.cond
        box = QWidget()
        grid = QGridLayout(box)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)
        where_keys = list(gv.WHERE_WORDS)
        for i, (pop, name) in enumerate(zip(cond["populations"], self._names())):
            grid.addWidget(_swatch(gp.POP_COLOURS[i % len(gp.POP_COLOURS)]), i, 0, alignment=Qt.AlignCenter)
            grid.addWidget(label(f"{name}, D {float(pop['D']):g} µm²/s", ty.ui("sm"), t.foreground), i, 1)
            where = _ChoiceBox(t, [gv.WHERE_WORDS[k] for k in where_keys], where_keys.index(pop["where"]))
            where.setMinimumContentsLength(16)
            where.currentIndexChanged.connect(lambda idx, p=pop: self._set(p, "where", where_keys[idx]))
            grid.addWidget(where, i, 2)
        grid.setColumnStretch(3, 1)
        return box

    # -- switching --------------------------------------------------------------- #
    def switching_widget(self) -> QWidget:
        t, cond = self.t, self.cond
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        if len(cond["populations"]) == 1:
            lay.addWidget(_words("One population cannot switch. Add a population on the Populations page.", t))
        else:
            seg = Segmented(t, [gv.SWITCHING_WORDS[k] for k in gv.SWITCHING], current=gv.SWITCHING.index(
                cond["switching"]), size="sm")
            seg.setMinimumWidth(seg.sizeHint().width())
            seg.changed.connect(self._switch_mode)
            lay.addWidget(seg, alignment=Qt.AlignLeft)
            lay.addWidget(_words(gv.SWITCHING_NOTES[cond["switching"]], t))
            if cond["switching"] == "exchange":
                row = QHBoxLayout()
                row.addWidget(label("Exchange rate (per s)", ty.ui("sm"), t.foreground))
                ex = _float(t, cond["exchange_per_s"], 0.001, 1000.0, 3)
                ex.changed.connect(lambda v: self._set(cond, "exchange_per_s", v))
                row.addWidget(ex)
                row.addStretch(1)
                lay.addLayout(row)
            if cond["switching"] == "matrix":
                lay.addWidget(self._matrix(cond["matrix"]))
        row = QHBoxLayout()
        row.addWidget(label("Spread between replicates (log SD)", ty.ui("sm"), t.foreground))
        sd = _float(t, cond["replicate_sd"], 0.0, 3.0, 2)
        sd.changed.connect(lambda v: self._set(cond, "replicate_sd", v))
        row.addWidget(sd)
        row.addStretch(1)
        lay.addLayout(row)
        lay.addWidget(_words("Each replicate's binding is drawn this far from the condition's own, so the "
                             "replicates differ as real cultures do (0: they are alike).", t))
        return box

    def _matrix(self, matrix: list) -> QWidget:
        t = self.t
        box = QWidget()
        grid = QGridLayout(box)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(4)
        names = self._names()
        grid.addWidget(label("from \\ to", ty.ui("xs"), t.muted_foreground), 0, 0)
        for j, n in enumerate(names):
            grid.addWidget(label(n, ty.ui("xs", "medium"), t.muted_foreground), 0, j + 1)
        for i, n in enumerate(names):
            grid.addWidget(label(n, ty.ui("xs", "medium"), t.muted_foreground), i + 1, 0)
            for j in range(len(names)):
                if i == j:
                    grid.addWidget(label("", ty.ui("xs"), t.muted_foreground), i + 1, j + 1)
                    continue
                cell = _float(t, matrix[i][j], 0.0, 10000.0, 3, width=96)
                cell.changed.connect(lambda v, a=i, b=j, m=matrix: self._set_cell(m, a, b, v))
                grid.addWidget(cell, i + 1, j + 1)
        grid.setColumnStretch(len(names) + 1, 1)
        return box

    def _set_cell(self, matrix: list, i: int, j: int, value: float) -> None:
        matrix[i][j] = value
        self.edited.emit()

    def _switch_mode(self, index: int) -> None:
        self.cond["switching"] = gv.SWITCHING[index]
        self.reshape.emit()

    # -- kinetics ---------------------------------------------------------------- #
    def kinetics_widget(self) -> QWidget:
        t, cond = self.t, self.cond
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        head = QHBoxLayout()
        on = Checkbox(t, checked=cond.get("kinetics") is not None)
        on.toggled.connect(self._kinetics_on)
        head.addWidget(on)
        head.addWidget(label("A change after an addition", ty.ui("sm", "semibold"), t.foreground))
        head.addStretch(1)
        lay.addLayout(head)
        kin = cond.get("kinetics")
        if not kin:
            lay.addWidget(_words("Off: the condition stays as its populations say for the whole set.", t))
            return box
        lay.addWidget(_words("Each movie is recorded at its own time after the addition; a negative time is "
                             "before it (the baseline). After the delay, every value moves from before to after, "
                             "half of it one half-time later.", t))
        minutes = QHBoxLayout()
        minutes.addWidget(label("Movie times (min)", ty.ui("sm"), t.foreground))
        edit = _line_edit(t, kin["minutes"], placeholder="-6, -4, -2, 0.5, 1.5, 3")
        edit.textEdited.connect(lambda text: self._set(kin, "minutes", text))
        minutes.addWidget(edit, 1)
        lay.addLayout(minutes)
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(6)
        grid.setColumnStretch(0, 1)
        for k, (key, words, high, dec) in enumerate((("delay_s", "Delay (s)", 1e5, 1),
                                                     ("half_time_s", "Half-time (s)", 1e5, 1),
                                                     ("onset_sd_replicate_s", "Onset spread between replicates (s)",
                                                      1e4, 1),
                                                     ("onset_sd_cell_s", "Onset spread between cells (s)", 1e4, 1))):
            grid.addWidget(label(words, ty.ui("sm"), t.foreground), k, 0)
            field = _float(t, kin[key], 0.001 if key == "half_time_s" else 0.0, high, dec)
            field.changed.connect(lambda v, kk=key: self._set(kin, kk, v))
            grid.addWidget(field, k, 1, alignment=Qt.AlignRight)
        lay.addLayout(grid)
        mock_row = QHBoxLayout()
        mock = Checkbox(t, checked=bool(kin["mock"]))
        mock.toggled.connect(self._mock)
        mock_row.addWidget(mock)
        mock_row.addWidget(label("A mock: the addition changes nothing", ty.ui("sm"), t.foreground))
        mock_row.addStretch(1)
        lay.addLayout(mock_row)
        if kin["mock"]:
            return box
        after = QWidget()
        g = QGridLayout(after)
        g.setContentsMargins(0, 0, 0, 0)
        g.setHorizontalSpacing(8)
        matrix = cond["switching"] == "matrix"
        g.addWidget(label("After the addition", ty.ui("xs", "medium"), t.muted_foreground), 0, 1)
        g.addWidget(label("D (µm²/s)", ty.ui("xs", "medium"), t.muted_foreground), 0, 2)
        if not matrix:
            g.addWidget(label("Share", ty.ui("xs", "medium"), t.muted_foreground), 0, 3)
        for i, (name, a) in enumerate(zip(self._names(), kin["after"])):
            g.addWidget(_swatch(gp.POP_COLOURS[i % len(gp.POP_COLOURS)]), i + 1, 0, alignment=Qt.AlignCenter)
            g.addWidget(label(name, ty.ui("sm"), t.foreground), i + 1, 1)
            d = _float(t, a["D"], 0.0, 100.0, 3)
            d.changed.connect(lambda v, aa=a: self._set(aa, "D", v))
            g.addWidget(d, i + 1, 2)
            if not matrix:
                s = _float(t, a["share"], 0.0, 1.0, 3)
                s.changed.connect(lambda v, aa=a: self._set(aa, "share", v))
                g.addWidget(s, i + 1, 3)
        g.setColumnStretch(4, 1)
        lay.addWidget(after)
        if matrix:
            lay.addWidget(label("Rates after the addition (per s)", ty.ui("xs", "medium"), t.muted_foreground))
            lay.addWidget(self._matrix(kin["after_matrix"]))
        return box

    def _kinetics_on(self, on: bool) -> None:
        gv.set_kinetics(self.cond, on, self._movies())
        self.reshape.emit()

    def _mock(self, on: bool) -> None:
        self.cond["kinetics"]["mock"] = bool(on)
        self.reshape.emit()


class _Page(QWidget):
    """One page: its cards in a scrolling column on the left (drawn again when the form changes shape, then the
    widgets built once, the Build page's), a picture card on the right."""

    def __init__(self, t, *, title: str, picture: QWidget | None = None, caption: str = "", parent=None):
        super().__init__(parent)
        self.t = t
        self.setStyleSheet("background:transparent;")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(16)
        self.body = QWidget()
        self.body.setStyleSheet("background:transparent;")
        self.body_lay = QVBoxLayout(self.body)
        self.body_lay.setContentsMargins(0, 0, 8, 0)
        self.body_lay.setSpacing(12)
        self.body_lay.addStretch(1)
        scroll = FitScroll()
        scroll.setWidget(self.body)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setStyleSheet("QScrollArea { background:transparent; border:none; }" + scrollbar_css(t))
        scroll.viewport().setStyleSheet("background:transparent;")
        lay.addWidget(scroll, 1)
        self.side = Card(t)
        # its full width where the window has it, narrower on a smaller screen (never a fixed width: that is a
        # minimum the window obeys by growing past the screen)
        self.side.setMinimumWidth(PIC_MIN_W)
        self.side.setMaximumWidth(PIC_W)
        self.side.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        side = QVBoxLayout(self.side)
        side.setContentsMargins(18, 14, 18, 16)
        side.setSpacing(8)
        side.addWidget(label(title, ty.ui("base", "semibold"), t.foreground))
        self.picture = picture
        if picture is not None:
            side.addWidget(picture, 1)
        self.caption = _words(caption, t)
        side.addWidget(self.caption)
        self.side_lay = side
        if picture is None:
            side.addStretch(1)
        lay.addWidget(self.side)
        self.cards: list = []
        self.fixed: list = []

    def set_cards(self, cards: list) -> None:
        for card in self.cards:
            self.body_lay.removeWidget(card)
            discard(card)
        self.cards = list(cards)
        for index, card in enumerate(self.cards):
            self.body_lay.insertWidget(index, card)

    def add_fixed(self, widget: QWidget) -> None:
        self.body_lay.insertWidget(self.body_lay.count() - 1, widget)
        self.fixed.append(widget)

    def add_helper(self, widget: QWidget) -> None:
        self.side_lay.addWidget(widget)


class GeneratorWindow(QMainWindow):
    """The pages, the condition picker, Build, and where the set opens next."""

    def __init__(self, t, *, state: dict | None = None, parent=None):
        super().__init__(parent)
        self.t = t
        from ...buildstamp import build_stamp

        self.setWindowTitle(f"SMTracker · simulated data  ·  {build_stamp()}")
        self.setStyleSheet(f"QMainWindow {{ background:{t.background}; }}" + scrollbar_css(t))
        self.state = state if state is not None else self._remembered_state()
        self.current = 0
        self.page = "start"
        self.return_to_menu = False
        #: ``(stage, folder)`` after "Open in ...": the door (`app.main`) opens that stage on the folder.
        self.next_stage: tuple | None = None
        self.built: Path | None = None
        self._thread: threading.Thread | None = None
        self._build_started: float | None = None
        self._build_done = (0, 0)
        self.signals = _BuildSignals()
        self.signals.line.connect(self._log_line)
        self.signals.progress.connect(self._build_progress)
        self.signals.done.connect(self._built)
        self.signals.failed.connect(self._failed)
        self.preview_signals = _PreviewSignals()
        self.preview_signals.ready.connect(self._preview_ready)
        self._previews: dict = {}
        self._field_error = ""
        for key in SLOW_PICTURES:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(lambda k=key: self._start_preview(k))
            self._previews[key] = {"timer": timer, "busy": False, "again": False, "token": 0, "shown": None,
                                   "want": None}
        self.panel: _ConditionParts | None = None
        self._cards: list = []
        self._dirty = set(gp.PAGE_TITLES)
        self._page_sig: dict = {}
        self._link_pages: tuple | None = None

        central = QWidget()
        outer = QHBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        sections = tuple((key, title, glyph, ((f"{key}.page", title),)) for key, title, glyph in gp.PAGES)
        self.sidebar = Sidebar(t, project_title="Simulated data", project_subtitle=gv.folder_name(self.state),
                               sections=sections, group_label="Pages")
        self.sidebar.selected.connect(self._on_view)
        self.sidebar.opened.connect(lambda: self.show_page("start"))
        outer.addWidget(self.sidebar)
        column = QWidget()
        column.setStyleSheet(f"background:{t.background};")
        col = QVBoxLayout(column)
        col.setContentsMargins(26, 16, 26, 16)
        col.setSpacing(10)
        head = QHBoxLayout()
        head.setSpacing(8)
        trigger = SidebarTrigger(t)
        trigger.clicked.connect(lambda: self.sidebar.set_collapsed(not self.sidebar.collapsed))
        head.addWidget(trigger)
        self.page_title = label("", ty.ui("lg", "semibold"), t.foreground)
        head.addWidget(self.page_title)
        self.activity = activity_label(ty.ui("xs"), t.muted_foreground)
        head.addWidget(self.activity, 1)
        load = Button(t, "Load recipe…", size="sm", glyph="folder-open")
        load.setToolTip("Open a recipe saved before (a .json file); the form takes its values.")
        load.clicked.connect(self.load_recipe)
        head.addWidget(load)
        save = Button(t, "Save recipe…", size="sm")
        save.setToolTip("Save the form as a recipe (.json), to build the same set again or share it.")
        save.clicked.connect(self.save_recipe)
        head.addWidget(save)
        back = Button(t, "Back to menu", size="sm")
        back.setToolTip("Return to the stage menu. Closing the window quits the tool.")
        back.clicked.connect(self.back_to_menu)
        head.addWidget(back)
        col.addLayout(head)
        self.page_note = _words("", t)
        col.addWidget(self.page_note)
        self.condition_bar = QWidget()
        self.condition_bar.setStyleSheet("background:transparent;")
        self.condition_lay = QHBoxLayout(self.condition_bar)
        self.condition_lay.setContentsMargins(0, 0, 0, 0)
        self.condition_lay.setSpacing(8)
        col.addWidget(self.condition_bar)
        self.stack = QStackedWidget()
        self.stack.setStyleSheet("background:transparent;")
        col.addWidget(self.stack, 1)
        foot = QHBoxLayout()
        foot.setSpacing(8)
        self.page_problems = _words("", t, size="sm", colour=CUT_INK)
        foot.addWidget(self.page_problems, 1)
        self.back_button = Button(t, "Back", size="sm", variant="ghost", glyph="chevron-left")
        self.back_button.clicked.connect(lambda: self._step_page(-1))
        foot.addWidget(self.back_button)
        self.next_button = Button(t, "Next", size="sm", glyph="chevron-right")
        self.next_button.clicked.connect(lambda: self._step_page(1))
        foot.addWidget(self.next_button)
        col.addLayout(foot)
        outer.addWidget(column, 1)
        self.setCentralWidget(central)

        self.pictures = {"start": BubblePicture(t), "populations": BubblePicture(t), "where": CellPicture(
            t, fps=gp.ANIMATION_FPS), "switching": DiagramPicture(t), "kinetics": CurvePicture(t),
            "cells": FieldPicture(t), "camera": FramePicture(t)}
        captions = {
            "start": "Every condition's populations as the SQD chart will show them: height is D on a log axis, "
                     "area is the share.",
            "populations": "Height is D on a log axis, area is the share. D 0 (bound) sits on its own line at "
                           "the bottom.",
            "where": "",
            "switching": "Each circle is a population, its area its share; each arrow a rate, per second. A "
                         "molecule stays in a population for one over its rates out, on average.",
            "kinetics": "The response every value follows: nothing until the delay, then half of the change one "
                        "half-time later. Grey: the movies, each at its own time. Under it, each population's share "
                        "before and after.",
            "cells": "One field as a build places its cells, at the pixel's scale (a line per pixel, or every "
                     "tenth). Cells never overlap.",
            "camera": "A corner of one frame, rendered as a build renders it: the molecules blurred over the "
                      "exposure, the background, the beam, the cells' glow and the camera's noise.",
            "build": ""}
        titles = {"start": "The set as it stands", "populations": "The populations", "where": "One cell, live",
                  "switching": "Switching", "kinetics": "After the addition", "cells": "One field",
                  "camera": "One frame", "build": "What will be built"}
        self.pages: dict = {}
        for key, _title, _glyph in gp.PAGES:
            page = _Page(t, title=titles[key], picture=self.pictures.get(key), caption=captions[key])
            self.pages[key] = page
            self.stack.addWidget(page)
        self._helpers()
        self._build_fixed()
        self.rebuild()
        self.show_page("start")

    # -- the state ---------------------------------------------------------------- #
    @staticmethod
    def _remembered_state() -> dict:
        state = theme.remembered(STATE_KEY)
        if isinstance(state, dict):
            try:
                gv.state_to_recipe(state)
                return state
            except Exception:                              # noqa: BLE001 - a remembered form that no longer reads
                pass
        return gv.default_state()

    def _movies(self) -> int:
        return int(self.state["fields"]["movies_per_replicate"])

    def _movies_out(self) -> bool:
        return self.state["fields"].get("output") == "movies"

    # -- pages -------------------------------------------------------------------- #
    def show_page(self, key: str) -> None:
        """Show page ``key`` (`generator_preview.PAGES`), drawing it first if the form changed since."""
        if key not in self.pages:
            raise KeyError(key)
        self.page = key
        # drawn again where the form changed since it was drawn (a name typed on another page, a D changed)
        if key in self._dirty or self._page_sig.get(key) != self._form_sig():
            self._build_page(key)
        self._cards = self.pages[key].cards
        self.stack.setCurrentWidget(self.pages[key])
        self.page_title.setText(gp.PAGE_TITLES[key])
        self.page_note.setText(gp.PAGE_NOTES[key])
        self.condition_bar.setVisible(key in gp.CONDITION_PAGES)
        if self.sidebar.current != f"{key}.page":
            self.sidebar.select(f"{key}.page")
        keys = [k for k, _t, _g in gp.PAGES]
        i = keys.index(key)
        self.back_button.setVisible(i > 0)
        self.next_button.setVisible(i < len(keys) - 1)
        if i < len(keys) - 1:
            self.next_button.set_text(f"Next: {gp.PAGE_TITLES[keys[i + 1]]}")
        self.refresh()

    def _on_view(self, view: str) -> None:
        key = view.split(".")[0]
        if key != self.page and key in self.pages:
            self.show_page(key)

    def _step_page(self, step: int) -> None:
        keys = [k for k, _t, _g in gp.PAGES]
        i = min(max(keys.index(self.page) + step, 0), len(keys) - 1)
        self.show_page(keys[i])

    def rebuild(self) -> None:
        """Draw the pages again from the state (a shape changed, a recipe or an example loaded, a preset applied):
        the page shown now, the others when they are next shown."""
        self.current = min(self.current, len(self.state["conditions"]) - 1)
        old = self.panel
        self.panel = _ConditionParts(self.t, self.state["conditions"][self.current], self._movies, parent=self)
        self.panel.edited.connect(self.refresh)
        self.panel.reshape.connect(lambda: QTimer.singleShot(0, self.rebuild))
        if old is not None:
            old.deleteLater()
        self._condition_bar()
        self.sidebar.set_project("Simulated data", gv.folder_name(self.state))
        self._dirty = set(gp.PAGE_TITLES)
        self._build_page(self.page)
        self._cards = self.pages[self.page].cards
        self.refresh()

    def _build_page(self, key: str) -> None:
        builders = {"start": self._start_cards, "populations": self._population_cards, "where": self._where_cards,
                    "switching": self._switching_cards, "kinetics": self._kinetics_cards, "cells": self._cell_cards,
                    "camera": self._camera_cards, "build": self._build_cards}
        self.pages[key].set_cards(builders[key]())
        self._dirty.discard(key)
        self._page_sig[key] = self._form_sig()

    def _form_sig(self) -> str:
        return json.dumps([self.state, self.current, self._movies_out()], sort_keys=True, default=str)

    def _card(self, title: str, note: str = "") -> tuple:
        card = Card(self.t)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(18, 14, 18, 16)
        lay.setSpacing(8)
        lay.addWidget(label(title, ty.ui("base", "semibold"), self.t.foreground))
        if note:
            lay.addWidget(_words(note, self.t))
        return card, lay

    def _fields_card(self, title: str, fields: list, note: str = "") -> tuple:
        card, lay = self._card(title, note)
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(6)
        grid.setColumnStretch(0, 1)
        for r, field in enumerate(fields):
            name = field.label + (f"  ({field.unit})" if field.unit else "")
            words = label(name, ty.ui("sm"), self.t.foreground)
            words.setWordWrap(True)
            if field.help:
                words.setToolTip(field.help)
            grid.addWidget(words, r, 0)
            grid.addWidget(self._control(field), r, 1, alignment=Qt.AlignRight)
        lay.addLayout(grid)
        return card, lay

    # -- the condition picker -------------------------------------------------------- #
    def _condition_bar(self) -> None:
        t = self.t
        while self.condition_lay.count():
            item = self.condition_lay.takeAt(0)
            if item.widget() is not None:
                discard(item.widget())
        names = [c["name"] or f"condition {i + 1}" for i, c in enumerate(self.state["conditions"])]
        self.condition_lay.addWidget(label("Condition", ty.ui("sm", "medium"), t.foreground))
        if len(names) <= 4:
            tabs = Segmented(t, names, current=self.current, size="sm")
            tabs.setMinimumWidth(tabs.sizeHint().width())
            tabs.changed.connect(self._pick)
        else:
            tabs = _ChoiceBox(t, names, self.current)
            tabs.currentIndexChanged.connect(self._pick)
        self.condition_lay.addWidget(tabs)
        # BESIDE THE TABS (asked 25 Nov 2026: "the button to add condition is too far away")
        add_button = Button(t, "Add condition", size="sm", glyph="plus")
        add_button.setToolTip("A new condition, a copy of this one: change what should differ.")
        add_button.clicked.connect(self._add_condition)
        self.condition_lay.addWidget(add_button)
        self.condition_lay.addSpacing(12)
        name = _line_edit(t, self.state["conditions"][self.current]["name"], width=180, placeholder="name")
        name.setToolTip("The condition's name: its folder in the set.")
        name.textEdited.connect(self._rename)
        name.editingFinished.connect(lambda: QTimer.singleShot(0, self.rebuild))
        self.condition_lay.addWidget(name)
        remove = Button(t, "Remove", size="sm", variant="ghost")
        remove.setToolTip("Remove this condition.")
        remove.setEnabled(len(names) > 1)
        remove.clicked.connect(self._remove_condition)
        self.condition_lay.addWidget(remove)
        self.condition_lay.addStretch(1)

    def _rename(self, text: str) -> None:
        self.state["conditions"][self.current]["name"] = text
        self.refresh()

    def _pick(self, index: int) -> None:
        self.current = int(index)
        self.rebuild()

    def _add_condition(self) -> None:
        self.current = gv.add_condition(self.state, self.current)
        self.rebuild()

    def _remove_condition(self) -> None:
        gv.remove_condition(self.state, self.current)
        self.current = max(0, self.current - 1)
        self.rebuild()

    # -- each page's cards ------------------------------------------------------------- #
    def _start_cards(self) -> list:
        t = self.t
        card, lay = self._card("Start from an example", "Each fills the whole form; change anything after.")
        for key, title, words in gp.EXAMPLES:
            row = QHBoxLayout()
            row.setSpacing(12)
            text = QVBoxLayout()
            text.setSpacing(2)
            text.addWidget(label(title, ty.ui("sm", "semibold"), t.foreground))
            text.addWidget(_words(words, t))
            row.addLayout(text, 1)
            use = Button(t, "Use this", size="sm")
            use.clicked.connect(lambda _=False, k=key: self.use_example(k))
            row.addWidget(use, alignment=Qt.AlignVCenter)
            lay.addLayout(row)
        out, olay = self._card("What to build")
        keys = ["tracks", "movies"]
        current = keys.index(self.state["fields"]["output"]) if self.state["fields"]["output"] in keys else 0
        seg = Segmented(t, ["Tracks", "Movies"], current=current, size="sm")
        seg.setMinimumWidth(seg.sizeHint().width())
        seg.changed.connect(lambda i: self._field("output", keys[i], reshape=True))
        olay.addWidget(seg, alignment=Qt.AlignLeft)
        olay.addWidget(_words(
            "Tracks: written perfectly linked, with the true outlines, to test the analysis. Quick and small."
            if current == 0 else
            "Movies: camera files to track in Image preparation + tracking, which also tests detection, linking "
            "and the outlines. Slower, and large.", t, size="sm", colour=t.foreground))
        how, hlay = self._card("How the pages go")
        hlay.addWidget(_words("Populations, then where in the cell they live, whether they switch, a change after an "
                              "addition, the cells and the field, the camera, and Build. The rail on the left goes "
                              "to any page; Build names every page with something to put right.", t,
                              size="sm", colour=t.foreground))
        return [card, out, how]

    def _population_cards(self) -> list:
        t = self.t
        name = self.state["conditions"][self.current]["name"] or "this condition"
        card, lay = self._card(f"Populations in {name}", "D 0 is a bound molecule: it reads at the localisation "
                                                         "floor, not at 0.")
        lay.addWidget(self.panel.populations_widget())
        typical, tlay = self._card("Typical D, for a start")
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(4)
        grid.setColumnStretch(0, 1)
        for r, (what, value) in enumerate(gp.TYPICAL_D):
            grid.addWidget(label(what, ty.ui("sm"), t.foreground), r, 0)
            grid.addWidget(label(f"{value} µm²/s", ty.ui("sm", tabular=True), t.muted_foreground), r, 1)
        tlay.addLayout(grid)
        tlay.addWidget(_words("A fit reads the slowest D near the localisation floor and the fastest near the "
                              "linking limit (Camera and movie), so the shares are what a set tests best.", t))
        return [card, typical]

    def _where_cards(self) -> list:
        card, lay = self._card("Where each population lives", "A population at foci is bound at binding sites; put "
                                                               "a slow one there.")
        lay.addWidget(self.panel.where_widget())
        places, play = self._fields_card("The places in the cell", gp.fields_on(self.state, "where"),
                                         "Their sizes, and where the foci sit. Only a place a population lives in "
                                         "changes the set.")
        self.where_warnings = _words("", self.t, size="sm", colour=WARN_INK)
        play.addWidget(self.where_warnings)
        self.where_warnings.setVisible(False)
        return [card, places]

    def _switching_cards(self) -> list:
        name = self.state["conditions"][self.current]["name"] or "this condition"
        card, lay = self._card(f"Switching in {name}")
        lay.addWidget(self.panel.switching_widget())
        return [card]

    def _kinetics_cards(self) -> list:
        name = self.state["conditions"][self.current]["name"] or "this condition"
        card, lay = self._card(f"Kinetics in {name}")
        lay.addWidget(self.panel.kinetics_widget())
        return [card]

    def _cell_cards(self) -> list:
        fields = gp.fields_on(self.state, "cells")
        field_keys = ("field_px_x", "field_px_y")
        cells, _ = self._fields_card("Cells", [f for f in fields if f.key not in field_keys])
        field, _ = self._fields_card("The field", [f for f in fields if f.key in field_keys],
                                     "A movie's size in camera pixels.")
        return [cells, field]

    def _camera_cards(self) -> list:
        t = self.t
        movie, _ = self._fields_card("Camera and movie", gp.fields_on(self.state, "camera"))
        if not self._movies_out():
            note, nlay = self._card("Tracks have no camera")
            nlay.addWidget(_words("Tracks are written with the localisation error above. Choose Movies on Start to "
                                  "simulate a camera, its noise and every imperfection of a real movie.", t,
                                  size="sm", colour=t.foreground))
            return [movie, note]
        imperfect, ilay = self._card("Movie imperfections", "Start from a preset, then change any value. In a movie "
                                                            "the localisation error is not set: it follows from the "
                                                            "photons, the background and the PSF.")
        for row in self._preset_rows():
            ilay.addLayout(row)
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(6)
        grid.setColumnStretch(0, 1)
        for r, field in enumerate(gp.fields_on(self.state, "imperfections")):
            name = field.label + (f"  ({field.unit})" if field.unit else "")
            words = label(name, ty.ui("sm"), t.foreground)
            words.setWordWrap(True)
            if field.help:
                words.setToolTip(field.help)
            grid.addWidget(words, r, 0)
            grid.addWidget(self._control(field), r, 1, alignment=Qt.AlignRight)
        ilay.addLayout(grid)
        return [movie, imperfect]

    def _build_cards(self) -> list:
        card, _ = self._fields_card("Replicates and name", gp.fields_on(self.state, "build"),
                                    "Every replicate is built independently of every other, within a condition and "
                                    "across conditions.")
        return [card]

    def _helpers(self) -> None:
        """The words under the pictures that change with the form: the camera's precision and linking limit."""
        t = self.t
        self.precision_words = _words("", t, size="sm", colour=t.foreground)
        self.pages["camera"].add_helper(self.precision_words)
        self.ceiling_words = _words("", t, size="sm", colour=WARN_INK)
        self.pages["camera"].add_helper(self.ceiling_words)
        self.ceiling_words.setVisible(False)
        self.start_summary = _words("", t, size="sm", colour=t.foreground)
        self.pages["start"].add_helper(self.start_summary)
        self.summary = _words("", t, size="sm", colour=t.foreground)
        self.pages["build"].side_lay.insertWidget(1, self.summary)
        self.warnings = _words("", t, size="sm", colour=WARN_INK)
        self.pages["build"].side_lay.insertWidget(2, self.warnings)
        self.warnings.setVisible(False)

    def _build_fixed(self) -> None:
        """The Build page's folder, Build button, progress, log and Open, built once."""
        t = self.t
        page = self.pages["build"]
        where, wlay = self._card("Where it is written", "A new folder: a set is never written over one.")
        folder_row = QHBoxLayout()
        self.folder = _line_edit(t, "")
        self.folder.textEdited.connect(self._folder_edited)
        folder_row.addWidget(self.folder, 1)
        choose = Button(t, "Choose…", size="sm", variant="ghost")
        choose.clicked.connect(self.choose_parent)
        folder_row.addWidget(choose)
        wlay.addLayout(folder_row)
        page.add_fixed(where)
        self._suggest_folder()
        build, blay = self._card("Build")
        self.problems = _words("", t, size="sm", colour=CUT_INK)
        blay.addWidget(self.problems)
        self.problem_links = QWidget()
        self.problem_links.setStyleSheet("background:transparent;")
        self.problem_links_lay = QHBoxLayout(self.problem_links)
        self.problem_links_lay.setContentsMargins(0, 0, 0, 0)
        self.problem_links_lay.setSpacing(6)
        blay.addWidget(self.problem_links)
        row = QHBoxLayout()
        self.build_button = Button(t, "Build", size="sm", variant="default", glyph="play")
        self.build_button.clicked.connect(self.build)
        row.addWidget(self.build_button)
        row.addStretch(1)
        self.log_toggle = Button(t, "Show the log", size="sm", variant="ghost", glyph="list")
        self.log_toggle.clicked.connect(lambda: self.show_log(not self.log.isVisibleTo(self)))
        row.addWidget(self.log_toggle)
        blay.addLayout(row)
        self.run_progress = RunProgress(t)
        blay.addWidget(self.run_progress)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(ty.ui("xs"))
        self.log.setStyleSheet(f"QPlainTextEdit {{ background:{t.background}; color:{t.foreground};"
                               f" border:1px solid {t.border}; border-radius:{t.radius_md:g}px; }}")
        self.log.setMinimumHeight(140)
        # INTO THE LAYOUT FIRST, THEN HIDDEN (`docs/claude/qt_and_packaging.md`)
        blay.addWidget(self.log)
        self.log.setVisible(False)
        self.open_button = Button(t, "Open in analysis", size="sm", variant="default")
        self.open_button.clicked.connect(self.open_next)
        blay.addWidget(self.open_button)
        self.open_button.setVisible(False)
        page.add_fixed(build)
        self._clock = QTimer(self)
        self._clock.setInterval(1000)
        self._clock.timeout.connect(self._show_build_progress)

    def show_log(self, on: bool) -> None:
        self.log.setVisible(bool(on))
        self.log_toggle.set_text("Hide the log" if on else "Show the log")

    # -- controls ----------------------------------------------------------------------- #
    def _control(self, field) -> QWidget:
        t = self.t
        value = self.state["fields"][field.key]
        reshape = field.key in gp.RESHAPE_KEYS
        if field.kind == "switch":
            box = Checkbox(t, checked=bool(value))
            box.toggled.connect(lambda v, k=field.key: self._field(k, bool(v)))
            return box
        if field.kind == "text":
            edit = _line_edit(t, value, width=220)
            edit.textEdited.connect(lambda v, k=field.key: self._field(k, v))
            return edit
        if field.kind == "choice":
            keys = [k for k, _w in field.choices]
            words = [w for _k, w in field.choices]
            index = keys.index(value) if value in keys else 0
            box = _ChoiceBox(t, words, index)
            box.currentIndexChanged.connect(lambda i, k=field.key, ks=keys: self._field(k, ks[i], reshape=reshape))
            return box
        if field.kind == "int":
            box = NumberField(t, int(field.low), int(field.high), int(value), width=132)
            box.changed.connect(lambda v, k=field.key: self._field(k, int(v)))
            return box
        box = _float(t, value, field.low, field.high, field.decimals, width=132)
        box.changed.connect(lambda v, k=field.key: self._field(k, float(v)))
        return box

    def _field(self, key: str, value, *, reshape: bool = False) -> None:
        self.state["fields"][key] = value
        if key == "name":
            self._suggest_folder()
            self.sidebar.set_project("Simulated data", gv.folder_name(self.state))
        if reshape:
            QTimer.singleShot(0, self.rebuild)
        else:
            self.refresh()

    def use_example(self, key: str) -> None:
        """Fill the form from an example (`generator_preview.EXAMPLES`), keeping the choice of tracks or movies and
        the movie's camera values."""
        new = gp.example_state(key)
        for k in gv.PRESET_FIELDS + ("output",):
            if k in self.state["fields"]:
                new["fields"][k] = self.state["fields"][k]
        new["preset"] = self.state.get("preset", new["preset"])
        self.state = new
        self.current = 0
        self._suggest_folder()
        self.rebuild()
        self.activity.setText(f"Started from the example: {dict((k, w) for k, w, _d in gp.EXAMPLES)[key]}")

    # -- movie presets: the built-in ones and the user's own --------------------------- #
    @staticmethod
    def presets_library() -> dict:
        library = theme.remembered(PRESETS_KEY)
        return dict(library) if isinstance(library, dict) else {}

    def _preset_rows(self) -> list:
        t = self.t
        row = QHBoxLayout()
        row.addWidget(label("Start from", ty.ui("sm"), t.foreground))
        choices = gv.preset_choices(self.presets_library())
        keys = [k for k, _w in choices]
        current = self.state.get("preset")
        self.preset_box = _ChoiceBox(t, [w for _k, w in choices], keys.index(current) if current in keys else 0)
        self.preset_box.activated.connect(lambda i, ks=keys: self._preset(ks[i]))
        row.addWidget(self.preset_box)
        self.remove_preset_button = Button(t, "Remove", size="sm", variant="ghost")
        self.remove_preset_button.setEnabled(bool(current) and str(current).startswith(gv.OWN_PRESET))
        self.remove_preset_button.clicked.connect(self.remove_own_preset)
        row.addWidget(self.remove_preset_button)
        row.addStretch(1)
        own = QHBoxLayout()
        self.preset_name = _line_edit(t, "", width=180, placeholder="name")
        own.addWidget(self.preset_name)
        save = Button(t, "Save as my preset", size="sm", glyph="save")
        save.setToolTip("Keep these movie values (camera, photons, background and every imperfection) under a "
                        "name, to start from them next time.")
        save.clicked.connect(lambda: self.save_own_preset(self.preset_name.text()))
        own.addWidget(save)
        own.addStretch(1)
        return [row, own]

    def _preset(self, key: str) -> None:
        try:
            gv.apply_preset(self.state, key, self.presets_library())
        except gen.GeneratorError as exc:
            self.activity.setText(str(exc))
            return
        QTimer.singleShot(0, self.rebuild)

    def save_own_preset(self, name: str) -> bool:
        try:
            library = gv.add_preset(self.presets_library(), name, self.state)
        except gen.GeneratorError as exc:
            self.activity.setText(str(exc))
            return False
        theme.remember(PRESETS_KEY, library)
        self.state["preset"] = gv.OWN_PRESET + str(name).strip()
        self.activity.setText(f"Preset saved: {str(name).strip()}")
        QTimer.singleShot(0, self.rebuild)
        return True

    def remove_own_preset(self) -> None:
        key = str(self.state.get("preset") or "")
        if not key.startswith(gv.OWN_PRESET):
            return
        theme.remember(PRESETS_KEY, gv.remove_preset(self.presets_library(), key))
        self.state["preset"] = "clean"
        self.activity.setText(f"Preset removed: {key[len(gv.OWN_PRESET):]}")
        QTimer.singleShot(0, self.rebuild)

    # -- the folder ------------------------------------------------------------------------ #
    def _suggest_folder(self) -> None:
        """The set's name under the folder last chosen, until a folder is typed by hand."""
        if getattr(self, "_folder_typed", False) or not hasattr(self, "folder"):
            return
        parent = theme.remembered(FOLDER_KEY) or str(Path.home() / "Documents")
        self.folder.setText(str(Path(parent) / gv.folder_name(self.state)))

    def _folder_edited(self, _text: str) -> None:
        self._folder_typed = True
        self.refresh()

    def choose_parent(self) -> None:
        start = str(Path(self.folder.text()).parent) if self.folder.text() else str(Path.home())
        got = QFileDialog.getExistingDirectory(self, "Choose where the set is written", start)
        if got:
            theme.remember(FOLDER_KEY, got)
            self._folder_typed = False
            self._suggest_folder()
            self.refresh()

    # -- what the window says ---------------------------------------------------------------- #
    def refresh(self) -> None:
        """The summary, every problem with its page, the warnings, and the shown page's picture."""
        lines = gv.summary_lines(self.state)
        self.summary.setText("\n".join(lines))
        self.start_summary.setText("\n".join(lines))
        problems = gv.with_folder_check(self.state, self.folder.text().strip())
        listed = gp.page_problems(problems)
        self.problems.setText("\n".join(f"{gp.PAGE_TITLES[pg]}: {text}" for pg, text in listed))
        self.problems.setVisible(bool(listed))
        self._problem_links([pg for pg, _t in listed])
        here = [text for pg, text in listed if pg == self.page]
        elsewhere = sorted({pg for pg, _t in listed if pg != self.page}, key=list(gp.PAGE_TITLES).index)
        words = list(here)
        if elsewhere and self.page != "build":
            words.append("To put right on " + ", ".join(gp.PAGE_TITLES[pg] for pg in elsewhere)
                         + " (all listed on Build).")
        self.page_problems.setText("\n".join(words))
        self._show_warnings()
        if self.page == "where" and hasattr(self, "where_warnings"):
            foci = gv.warnings(self.state)
            self.where_warnings.setText("\n".join(foci))
            self.where_warnings.setVisible(bool(foci))
        if self.panel is not None and self.panel.share_sum is not None and self.page == "populations":
            text, ok = self.panel.share_words()
            self.panel.share_sum.setText(text)
            self.panel.share_sum.setStyleSheet(f"color:{self.t.muted_foreground if ok else CUT_INK};")
        running = self._thread is not None and self._thread.is_alive()
        self.build_button.setEnabled(not problems and not running)
        self.refresh_picture()

    def _show_warnings(self) -> None:
        """What builds but may not be what was meant: a fast population at foci, a D past the linking limit, and a
        field too small for its cells as last drawn (the build would stop there)."""
        notes = gv.warnings(self.state) + gp.ceiling_warnings(self.state)
        if self._field_error:
            notes.append(f"Cells and field: {self._field_error}")
        self.warnings.setText("\n".join(notes))
        self.warnings.setVisible(bool(notes))

    def _problem_links(self, pages: list) -> None:
        """A "Go to" button for each page with something to put right."""
        wanted = tuple(sorted(set(pages) - {"build"}, key=list(gp.PAGE_TITLES).index))
        if wanted == self._link_pages:
            return
        self._link_pages = wanted
        while self.problem_links_lay.count():
            item = self.problem_links_lay.takeAt(0)
            if item.widget() is not None:
                discard(item.widget())
        for pg in wanted:
            go = Button(self.t, f"Go to {gp.PAGE_TITLES[pg]}", size="sm", variant="ghost", glyph="arrow-right")
            go.clicked.connect(lambda _=False, k=pg: self.show_page(k))
            self.problem_links_lay.addWidget(go)
        self.problem_links_lay.addStretch(1)
        self.problem_links.setVisible(bool(wanted))

    def refresh_picture(self) -> None:
        """The shown page's picture: drawn at once where it is quick, after a short pause on a background thread
        where it is not (the cell, the frame)."""
        page = self.page
        picture = self.pictures.get(page)
        try:
            if page in ("start", "populations"):
                picture.set_payload(gp.bubbles(self.state))
            elif page == "switching":
                picture.set_payload(gp.state_diagram(self.state, self.current))
            elif page == "kinetics":
                picture.set_payload(gp.kinetic_curve(self.state, self.current))
            elif page == "cells":
                self._schedule("field")
            elif page == "where":
                self._schedule("cell")
            elif page == "camera":
                self._camera_helpers()
                if self._movies_out():
                    self._schedule("frame")
                else:
                    picture.set_message("Tracks are written without a camera: choose Movies on Start to see a "
                                        "frame.")
                    self._previews["frame"]["shown"] = None
        except (gen.GeneratorError, ValueError, TypeError, KeyError, IndexError) as exc:
            picture.set_message(f"Cannot draw this yet: {exc}")

    def _camera_helpers(self) -> None:
        if self._movies_out():
            nm = gp.expected_precision_nm(self.state)
            self.precision_words.setText(
                "" if nm is None else f"Expected precision of a still molecule: about {nm:.0f} nm, from the photons, "
                                      f"the background and the PSF (Mortensen 2010). A moving one is blurred over "
                                      f"the exposure, so reads worse.")
        else:
            self.precision_words.setText(f"Localisation error, as set: "
                                         f"{float(self.state['fields']['localisation_error_nm']):g} nm.")
        ceiling = gp.ceiling_warnings(self.state)
        self.ceiling_words.setText("\n".join(ceiling))
        self.ceiling_words.setVisible(bool(ceiling))

    # -- the slow pictures, off the GUI thread ---------------------------------------------- #
    #: Fields no picture reads: changing them draws nothing again.
    _UNDRAWN = ("name", "replicates", "movies_per_replicate", "seed", "write_outlines")

    def _signature(self) -> str:
        fields = {k: v for k, v in self.state["fields"].items() if k not in self._UNDRAWN}
        return json.dumps([fields, self.state["conditions"][self.current], self.current], sort_keys=True,
                          default=str)

    def _schedule(self, key: str) -> None:
        pv = self._previews[key]
        sig = self._signature()
        if sig == pv["shown"] and not pv["busy"]:
            return
        pv["want"] = sig
        if pv["shown"] is None and not pv["busy"]:
            self.pictures[SLOW_PICTURES[key][0]].set_message("Drawing…")
        pv["timer"].start(DEBOUNCE_MS)

    def _start_preview(self, key: str) -> None:
        pv = self._previews[key]
        if pv["busy"]:
            pv["again"] = True
            return
        pv["busy"] = True
        pv["token"] += 1
        token, sig = pv["token"], pv["want"]
        state, condition = copy.deepcopy(self.state), self.current
        fn = SLOW_PICTURES[key][1]

        def run():
            try:
                payload = fn(state, condition)
            except Exception as exc:                       # noqa: BLE001 - said on the picture, never swallowed
                payload = {"error": f"The preview failed: {type(exc).__name__}: {exc}"}
            self.preview_signals.ready.emit(key, token, (sig, payload))

        threading.Thread(target=run, name=f"generator-preview-{key}", daemon=True).start()

    def _preview_ready(self, key: str, token: int, packed) -> None:
        pv = self._previews[key]
        pv["busy"] = False
        sig, payload = packed
        if token == pv["token"]:
            pv["shown"] = sig
            picture = self.pictures[SLOW_PICTURES[key][0]]
            if payload.get("error") and not payload.get("outline") and key == "cell":
                picture.set_message(payload["error"])
            else:
                picture.set_payload(payload)
            if key == "cell":
                self.pages["where"].caption.setText(payload.get("caption", ""))
            if key == "field":
                self._field_error = payload.get("error", "")
                self._show_warnings()
        if pv["again"] or (pv["want"] != pv["shown"]):
            pv["again"] = False
            if pv["want"] is not None and pv["want"] != pv["shown"]:
                self._start_preview(key)

    def previews_idle(self) -> bool:
        """No slow picture being drawn or waiting to be (for the tests and the screenshots)."""
        return all(not pv["busy"] and not pv["timer"].isActive() for pv in self._previews.values())

    # -- build ---------------------------------------------------------------------- #
    def build(self) -> None:
        folder = self.folder.text().strip()
        problems = gv.with_folder_check(self.state, folder)
        if problems:
            self.refresh()
            return
        recipe = gv.state_to_recipe(self.state)
        theme.remember(STATE_KEY, self.state)
        self.log.clear()
        self.open_button.setVisible(False)
        self.activity.setText("Building…")
        self._log_line(gen.describe(recipe))
        target = Path(folder)
        self._build_started = time.monotonic()
        self._build_done = (0, len(recipe.conditions) * recipe.replicates)
        self._show_build_progress()
        self._clock.start()

        def run():
            try:
                truth = gen.build(recipe, target, on_line=self.signals.line.emit,
                                  on_progress=self.signals.progress.emit)
                self.signals.done.emit((str(target), truth))
            except Exception as exc:                      # noqa: BLE001 - said in the window, never swallowed
                self.signals.failed.emit(f"{type(exc).__name__}: {exc}")

        self._thread = threading.Thread(target=run, name="generator-build", daemon=True)
        self._thread.start()
        self.refresh()

    def _build_progress(self, done: int, total: int) -> None:
        self._build_done = (int(done), int(total))
        self._show_build_progress()

    def _show_build_progress(self) -> None:
        done, total = self._build_done
        self.run_progress.set_payload(progress_payload(title="Building the set", unit="replicate", done=done,
                                                       total=total, started=self._build_started))

    def _end_build(self) -> None:
        self._thread = None
        self._clock.stop()
        self.run_progress.set_payload(None)

    def _log_line(self, line: str) -> None:
        self.log.appendPlainText(str(line))

    def _built(self, payload) -> None:
        folder, _truth = payload
        self.built = Path(folder)
        self._end_build()
        self.activity.setText(f"Built: {folder}")
        self._log_line(f"Built: {folder}")
        stage = gv.next_stage(self.state)
        self.open_button.set_text("Open in analysis" if stage == "analyse" else "Open in image preparation + tracking")
        self.open_button.setVisible(True)
        self.refresh()

    def _failed(self, message: str) -> None:
        self._end_build()
        self.activity.setText("Not built")
        self._log_line(f"Not built: {message}")
        self.show_log(True)
        self.refresh()

    def open_next(self) -> None:
        if self.built is None:
            return
        stage = gv.next_stage(self.state)
        if stage == "analyse":
            from ...session import remember_project

            remember_project(self.built, conditions=len(self.state["conditions"]))
        self.next_stage = (stage, str(self.built))
        self.close()

    def back_to_menu(self) -> None:
        self.return_to_menu = True
        self.close()

    def closeEvent(self, event):                            # noqa: N802 - Qt's name
        if self._thread is not None and self._thread.is_alive():
            self.activity.setText("A build is running: wait for it to finish before closing.")
            self.return_to_menu = False
            self.next_stage = None
            event.ignore()
            return
        self.pictures["where"].timer.stop()
        super().closeEvent(event)

    # -- recipes -------------------------------------------------------------------- #
    def load_recipe(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load a recipe", str(Path.home()), "Recipe (*.json)")
        if path:
            self.load_recipe_file(path)

    def load_recipe_file(self, path) -> None:
        try:
            recipe = gen.load_recipe(path)
        except (gen.GeneratorError, OSError, ValueError, KeyError, TypeError) as exc:
            self.activity.setText(f"That recipe does not read: {exc}")
            return
        self.state = gv.recipe_to_state(recipe)
        self.current = 0
        self._suggest_folder()
        self.rebuild()
        self.activity.setText(f"Recipe loaded: {Path(path).name}")

    def save_recipe(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save the recipe", str(Path.home() / "recipe.json"),
                                              "Recipe (*.json)")
        if path:
            self.save_recipe_file(path)

    def save_recipe_file(self, path) -> None:
        try:
            gen.save_recipe(gv.state_to_recipe(self.state), path)
        except gen.GeneratorError as exc:
            self.activity.setText(f"Not saved: {exc}")
            return
        self.activity.setText(f"Recipe saved: {Path(path).name}")


def open_generator_window(*, color_scheme=None, show: bool = True, state: dict | None = None):
    """Build the window and show it; returns ``(application, window)`` for the door (`app.main`)."""
    app = QApplication.instance() or QApplication([])
    from ...appicon import apply as _apply_icon

    _apply_icon()
    t = theme.current(color_scheme)
    window = GeneratorWindow(t, state=state)
    _OPEN_WINDOWS.add(window)
    from ..widgets.screenfit import show_fitted

    if show:
        show_fitted(window)
    return app, window
