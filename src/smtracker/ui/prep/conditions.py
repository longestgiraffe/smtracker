"""The condition rows in the preparation sidebar: the only condition picker.

Ruled for the classic window ("why not remove the picker and just click on the
condition below"): the rows that show progress ARE the picker, so there is one control
for one choice and it carries the state. Each row is `prep_view.sidebar_row`'s plain
dict -- a bar, and the line beside it.

A LONG NAME IS ELIDED IN THE MIDDLE. Reported from the outlines window: three conditions
named ``20260923_B.subtilis_HNC_mNG.d2.1`` / ``.d2.2`` / ``.d2.3`` were cut at the
right, so all three rows read the same. The ending is what tells conditions of one
experiment apart, so it stays (``20260923_B.sub…mNG.d2.1``); the full name is the
tooltip.

THE CONDITION IN HAND IS MARKED PLAINLY (asked 15 Oct 2026: "its a bit grey in the list, but
its hard to see"). The nav's active fill (`sidebar_accent`, #F4F4F5 on #FAFAFA) was the only
mark; the row now has a darker fill, a dark outline and a semibold name. Every row carries a
1 px border (transparent when not in hand), so selecting one does not shift its text.
"""
from __future__ import annotations

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from ..theme import typography as ty
from ..widgets.primitives import Progress, Surface, label, mix

__all__ = ["ConditionList"]


class _MiddleElided(QLabel):
    """A one-line label that elides in the MIDDLE to whatever width it is given."""

    def __init__(self, text: str, font, colour: str):
        super().__init__()
        self._full = text
        self.setFont(font)
        self.setStyleSheet(f"color:{colour}; background:transparent;")
        self.setToolTip(text)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.setMinimumWidth(40)
        self._elide()

    def full_text(self) -> str:
        return self._full

    def resizeEvent(self, e) -> None:  # noqa: N802 - Qt's name
        super().resizeEvent(e)
        self._elide()

    def _elide(self) -> None:
        self.setText(self.fontMetrics().elidedText(self._full, Qt.ElideMiddle,
                                                   max(1, self.width())))


class _Row(Surface):
    def __init__(self, t, name: str, on_click, parent=None):
        super().__init__(t, radius=t.radius_md, shadowed=False, parent=parent)
        self.t = t
        self.name = name
        self._on_click = on_click
        self._active = False
        self.setCursor(Qt.PointingHandCursor)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 7)
        lay.setSpacing(4)
        self._title = _MiddleElided(name, ty.ui("sm", "medium"), t.sidebar_foreground)
        self._bar = Progress(t, 0.0, height=6)
        self._line = label("", ty.ui("xs", tabular=True),
                           mix(t.sidebar_foreground, t.sidebar, 0.62))
        self._line.setWordWrap(True)
        lay.addWidget(self._title)
        lay.addWidget(self._bar)
        lay.addWidget(self._line)
        self._restyle()

    def update_row(self, row: dict) -> None:
        self._bar.set_value(row["done"] / max(row["total"], 1))
        self._line.setText(row["line"])

    def set_active(self, on: bool) -> None:
        self._active = on
        self._restyle()

    def _restyle(self) -> None:
        t = self.t
        fill = mix(t.sidebar_foreground, t.sidebar, 0.10) if self._active else "transparent"
        edge = mix(t.sidebar_foreground, t.sidebar, 0.70) if self._active else "transparent"
        # BY NAME, so the border is the row's and not drawn again round every label in it
        self.setObjectName("conditionRow")
        self.setStyleSheet(f"#conditionRow {{ background:{fill}; border:1px solid {edge};"
                           f" border-radius:{t.radius_md:g}px; }}")
        self._title.setFont(ty.ui("sm", "semibold" if self._active else "medium"))
        self._title._elide()

    def is_active(self) -> bool:
        return self._active

    def mouseReleaseEvent(self, e) -> None:
        if e.button() == Qt.LeftButton:
            self._on_click(self.name)
        super().mouseReleaseEvent(e)


class ConditionList(QWidget):
    """Rows added and updated in place; ``picked(name)`` when one is clicked."""

    picked = Signal(str)

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.setStyleSheet("background:transparent;")
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(2)
        self._rows: dict = {}
        self._active: str | None = None

    def set_rows(self, rows: list) -> None:
        for row in rows:
            widget = self._rows.get(row["name"])
            if widget is None:
                widget = _Row(self.t, row["name"], self.picked.emit)
                self._lay.addWidget(widget)
                self._rows[row["name"]] = widget
            widget.update_row(row)
        self.set_active(self._active)

    def set_active(self, name: str | None) -> None:
        self._active = name
        for n, widget in self._rows.items():
            widget.set_active(n == name)

    def names(self) -> list:
        return list(self._rows)
