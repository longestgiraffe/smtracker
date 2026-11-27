"""The header menu's "Compare with the simulation's truth": a generated set's declared and counted values beside the
last SQD Run's fit (25 Nov 2026). Thin Qt: the table and its notes are `smtracker.truth_view.truth_table`; this lays
them out."""
from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QGridLayout, QHBoxLayout, QScrollArea, QWidget

from ..theme import typography as ty
from ..widgets.controls import Button
from ..widgets.dialogs import CloseButton, OverlayDialog
from ..widgets.primitives import FitScroll, label, scrollbar_css

__all__ = ["TruthDialog"]


class TruthDialog(OverlayDialog):
    """One table: per condition and population, declared, counted and fitted."""

    def __init__(self, t, table: dict, source: str, *, content=None, parent=None):
        super().__init__(t, content, parent=parent)
        self._card.setMinimumWidth(760)
        head = QHBoxLayout()
        head.addWidget(label("The simulation's truth", ty.ui("lg", "semibold"), t.foreground))
        head.addStretch(1)
        close = CloseButton(t)
        close.clicked.connect(self._dismiss)
        head.addWidget(close)
        self.card_lay.addLayout(head)
        where = label(f"From {source}", ty.ui("xs"), t.muted_foreground)
        where.setWordWrap(True)
        self.card_lay.addWidget(where)
        body = QWidget()
        body.setStyleSheet("background:transparent;")
        grid = QGridLayout(body)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(4)
        for col, words in enumerate(table["header"]):
            grid.addWidget(label(words, ty.ui("xs", "semibold"), t.muted_foreground), 0, col,
                           alignment=Qt.AlignLeft if col < 3 else Qt.AlignRight)
        for r, row in enumerate(table["rows"], 1):
            for col, words in enumerate(row):
                strong = col in (5, 7)
                grid.addWidget(label(str(words), (ty.ui("sm", "medium") if strong else ty.ui("sm")), t.foreground),
                               r, col, alignment=Qt.AlignLeft if col < 3 else Qt.AlignRight)
        scroll = FitScroll()
        scroll.setWidget(body)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setStyleSheet("QScrollArea { background:transparent; border:none; }" + scrollbar_css(t))
        scroll.viewport().setStyleSheet("background:transparent;")
        scroll.setMaximumHeight(min(420, body.sizeHint().height() + 8))
        self.card_lay.addWidget(scroll)
        for note in table["notes"]:
            words = label(note, ty.ui("xs"), t.muted_foreground)
            words.setWordWrap(True)
            self.card_lay.addWidget(words)
        foot = QHBoxLayout()
        foot.addStretch(1)
        done = Button(t, "Close", size="sm")
        done.clicked.connect(self._dismiss)
        foot.addWidget(done)
        self.card_lay.addLayout(foot)
