"""Which tracker to read, asked where a condition was tracked by both.

Reported: a condition holding u-track AND TrackMate output stopped the analysis door with
an error. `grouping.detect_project_source` is right to refuse a silent choice -- the two
disagree by about 1 pp in the slow fraction and 10-18 % in D on the same movies -- so the
answer is a question: one row per condition tracked twice, u-track or TrackMate, NOTHING
preselected, and Open only once every row is answered. Conditions with one tracker ask
nothing (asked for: "only if two or more options"). When the answers mix trackers across
conditions the page says so under the rows and still opens; the loader records each
condition's tracker (`canonical.attrs["trackers"]`).

The rules and the words are `onboarding_view`'s (`tracker_questions`,
`tracker_choice_ready`, `tracker_note`); this module only lays them out.
"""
from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QHBoxLayout, QSizePolicy, QVBoxLayout, QWidget

from ...onboarding_view import (TRACKER_LABELS, tracker_choice_ready, tracker_note,
                                tracker_questions, tracker_subtitle)
from ..theme import typography as ty
from ..widgets.controls import Button, Segmented
from ..widgets.primitives import Card, ShadowHost, band, icon, label, row

__all__ = ["TrackerChoicePage"]


class TrackerChoicePage(ShadowHost):
    """Emits ``chosen({condition: "utrack" | "trackmate"})`` for EVERY condition (the
    choice where one was asked, the only tracker elsewhere), or ``cancelled``."""

    chosen = Signal(object)
    cancelled = Signal()

    def __init__(self, t, root, options: dict, parent=None):
        super().__init__(t, parent)
        self.t = t
        self.options = dict(options)
        self.choice: dict = {}
        self._pickers: dict = {}
        self.setStyleSheet(f"background:{t.background};")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(48, 40, 48, 40)
        lay.setSpacing(0)
        lay.addStretch(1)

        root = Path(root)
        head = label("Choose the tracker", ty.ui("2xl", "semibold"), t.foreground)
        head.setAlignment(Qt.AlignHCenter)
        sub = self._words(f"{root.name or root}: {tracker_subtitle(self.options)}", ty.ui("sm"), t.muted_foreground)
        lay.addWidget(head)
        lay.addSpacing(6)
        lay.addLayout(self._column(sub))
        lay.addSpacing(24)

        card = Card(t)
        card.setMaximumWidth(640)
        card.setMinimumWidth(460)
        cl = QVBoxLayout(card)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)
        cl.addWidget(band(
            row(icon("arrow-left-right", 14, t.muted_foreground),
                label("Condition", ty.ui("sm", "semibold"), t.foreground), None,
                label("Read tracks from", ty.ui("xs"), t.muted_foreground),
                spacing=8, margins=(16, 12, 16, 12)),
            "trackersHead", f"background:transparent; border-bottom:1px solid {t.border};"))
        for question in tracker_questions(self.options):
            name = question["name"]
            keys = [key for key, _words in question["options"]]
            picker = Segmented(t, [words for _key, words in question["options"]],
                               current=-1, size="sm")
            picker.changed.connect(lambda i, n=name, k=keys: self._pick(n, k[i]))
            self._pickers[name] = picker
            cl.addWidget(band(
                row(label(name, ty.ui("sm", "medium"), t.foreground), None, picker,
                    spacing=12, margins=(16, 10, 16, 10)),
                "trackersRow", "background:transparent;"))
        holder = QWidget()
        holder.setStyleSheet("background:transparent;")
        hl = QVBoxLayout(holder)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.addWidget(card, alignment=Qt.AlignHCenter)
        lay.addWidget(holder)
        # NOTHING TO CHOOSE, NO TABLE (tester B7, 3 Nov 2026): conditions from different tools, none tracked twice,
        # drew the card as an empty "Condition / Read tracks from" header
        self.table = holder
        holder.setVisible(bool(self._pickers))
        lay.addSpacing(14 if self._pickers else 0)

        # THE NOTE IS READ WHOLE (tester B7): a wrapped label given only a maximum width, centred in the page's
        # layout, was cut after three lines ("... 8 px linking mNG,") and its warning never shown
        self.note = self._words("", ty.ui("xs"), t.muted_foreground)
        lay.addLayout(self._column(self.note))
        lay.addSpacing(16)

        self.back = Button(t, "Back", variant="outline")
        self.back.clicked.connect(self.cancelled.emit)
        self.open_button = Button(t, "Open", variant="default", glyph="folder-open")
        self.open_button.clicked.connect(self._open)
        buttons = row(None, self.back, self.open_button, None, spacing=8)
        lay.addWidget(buttons)
        lay.addStretch(1)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self._refresh()

    #: The widest the subtitle and the note run before they wrap.
    TEXT_W = 640

    @staticmethod
    def _words(text: str, font, colour: str):
        """A centred line that wraps and never asks for width: its column gives it one."""
        w = label(text, font, colour)
        w.setAlignment(Qt.AlignHCenter)
        w.setWordWrap(True)
        w.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        w.setMinimumWidth(0)
        return w

    def _column(self, widget) -> QHBoxLayout:
        """``widget`` centred in a column at most `TEXT_W` wide, as wide as the page allows below that."""
        box = QWidget()
        box.setStyleSheet("background:transparent;")
        box.setMaximumWidth(self.TEXT_W)
        box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        bl = QVBoxLayout(box)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.addWidget(widget)
        centred = QHBoxLayout()
        centred.setContentsMargins(0, 0, 0, 0)
        centred.addStretch(1)
        centred.addWidget(box, 100)
        centred.addStretch(1)
        return centred

    def _pick(self, name: str, key: str) -> None:
        self.choice[name] = key
        self._refresh()

    def _refresh(self) -> None:
        self.open_button.setEnabled(tracker_choice_ready(self.options, self.choice))
        self.note.setText(tracker_note(self.options, self.choice))
        self.note.setVisible(bool(self.note.text()))

    def answer(self) -> dict:
        """Every condition's tracker: the choice where asked, the only one elsewhere."""
        out = {}
        for name, found in self.options.items():
            pick = self.choice.get(name)
            if pick is None and len(found) == 1:
                pick = next(iter(found))
            if pick in TRACKER_LABELS:
                out[name] = pick
        return out

    def _open(self) -> None:
        if tracker_choice_ready(self.options, self.choice):
            self.chosen.emit(self.answer())
