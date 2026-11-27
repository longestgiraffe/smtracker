"""The Axes button's popover, one widget for every chart that lets you type its ranges (30 Oct 2026, asked of the
Jump distances page's: "the axes button is cool, we can use it in other plots like sqd for y axis etc?").

THIN: a row of from / to fields per axis the chart offers (``rows``, `smtracker.axes_view.axis_rows`), Auto and
Apply. The text is read and checked by `axes_view.parse` -- a log axis at or below 0, an end under its floor, from not
below to are refused BY NAME, shown here and nothing applied. ``applied`` carries ``{key: (from, to) or None}``; Auto
empties every field and applies all None (automatic: fitted to what is drawn). The owner keeps the ranges (they stay
until Auto) and redraws.
"""
from __future__ import annotations

from qtpy.QtCore import QPointF, Qt, Signal
from qtpy.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLineEdit, QVBoxLayout, QWidget

from ...axes_view import parse
from ..theme import typography as ty
from .controls import Button, Segmented
from .primitives import discard, label

__all__ = ["AxesPopover"]


def _style_field(t, field: QLineEdit) -> None:
    """A from / to field in the theme's own colours, as every field under `ui/` sets them (the Significance block's
    are the model): in the dark theme the unstyled fields were dark on the dark card with no border, and their
    "auto" all but invisible (4 Nov 2026, rail review)."""
    # the placeholder ("auto") follows: Qt draws it in the style sheet's text colour at half strength
    field.setStyleSheet(f"QLineEdit {{ background:{t.background}; color:{t.foreground};"
                        f" border:1px solid {t.input}; border-radius:{t.radius_md:g}px; padding:2px 6px; }}")


class AxesPopover(QFrame):
    """From / to fields per axis, Auto and Apply; `show_for` fills and opens it under its button."""

    applied = Signal(object)

    def __init__(self, t, parent=None):
        super().__init__(parent, Qt.Popup)
        self.t = t
        self.setObjectName("axesPop")
        self.setStyleSheet(f"QFrame#axesPop {{ background:{t.card}; border:1px solid {t.border};"
                           f" border-radius:{t.radius_md:g}px; }}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(6)
        self.title = label("Axes", ty.ui("sm", "semibold"), t.foreground)
        lay.addWidget(self.title)
        self._grid_host = QWidget()
        self._grid_host.setStyleSheet("background:transparent;")
        self._grid = QGridLayout(self._grid_host)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(6)
        self._grid.setVerticalSpacing(6)
        lay.addWidget(self._grid_host)
        self._choice_layout = QVBoxLayout()
        self._choice_layout.setContentsMargins(0, 0, 0, 0)
        self._choice_layout.setSpacing(4)
        lay.addLayout(self._choice_layout)
        self.message = label("", ty.ui("xs"), t.muted_foreground)
        self.message.setWordWrap(True)
        self.message.setMaximumWidth(380)
        lay.addWidget(self.message)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        auto = Button(t, "Auto", size="sm", variant="outline")
        auto.clicked.connect(self.auto)
        buttons.addWidget(auto)
        apply_button = Button(t, "Apply", size="sm", variant="default")
        apply_button.clicked.connect(self.apply)
        buttons.addWidget(apply_button)
        lay.addLayout(buttons)
        #: The rows shown (`axes_view.axis_rows`), and their fields ``{(key, "from" | "to"): QLineEdit}``.
        self.rows: list = []
        self.fields: dict = {}
        self._cells: list = []
        #: Choices beside the ranges (``show_for(choices=...)``): ``{key: (values, Segmented)}``, e.g. the time axis'
        #: unit. Their values go out in ``applied`` beside the ranges, under their keys.
        self.choices: dict = {}
        self._choice_hosts: list = []

    def _build_choices(self, choices) -> None:
        for w in self._choice_hosts:
            self._choice_layout.removeWidget(w)
            discard(w)
        self._choice_hosts, self.choices = [], {}
        for choice in choices or ():
            name = label(choice["label"], ty.ui("xs"), self.t.muted_foreground)
            pills = Segmented(self.t, [text for _value, text in choice["options"]],
                              current=[v for v, _t in choice["options"]].index(choice["value"]))
            row = QHBoxLayout()
            row.addWidget(name)
            row.addWidget(pills)
            row.addStretch(1)
            host = QWidget()
            host.setStyleSheet("background:transparent;")
            host.setLayout(row)
            self._choice_layout.addWidget(host)
            self._choice_hosts.append(host)
            self.choices[choice["key"]] = ([v for v, _t in choice["options"]], pills)

    def _build(self, rows) -> None:
        for w in self._cells:
            self._grid.removeWidget(w)
            discard(w)
        self._cells, self.fields = [], {}
        self.rows = [dict(r) for r in rows]
        for r, row in enumerate(self.rows):
            name = label(row["label"] + (" (log)" if row.get("log") else ""), ty.ui("xs"), self.t.muted_foreground)
            self._grid.addWidget(name, r, 0)
            self._cells.append(name)
            for c, end in enumerate(("from", "to")):
                word = label(end, ty.ui("xs"), self.t.muted_foreground)
                self._grid.addWidget(word, r, 1 + 2 * c)
                field = QLineEdit()
                field.setPlaceholderText("auto")
                field.setFixedWidth(72)
                field.setFont(ty.ui("sm"))
                _style_field(self.t, field)
                field.returnPressed.connect(self.apply)
                self._grid.addWidget(field, r, 2 + 2 * c)
                self.fields[(row["key"], end)] = field
                self._cells += [word, field]

    def show_for(self, rows, axes: dict, anchor=None, *, note: str = "", choices=None) -> None:
        """Rows for ``rows`` filled from the chart's typed ``axes`` (empty where automatic), opened under ``anchor``
        (a widget), else where it is. ``note`` is said under the fields (what resets a range). ``choices``
        ``[{"key", "label", "options": [(value, text)], "value"}]``: pills under the fields (the Kinetics time axis'
        unit); the pick goes out under its key beside the ranges."""
        if [dict(r) for r in rows] != self.rows:
            self._build(rows)
        self._build_choices(choices)
        for (key, end), field in self.fields.items():
            pair = (axes or {}).get(key) or (None, None)
            value = pair[0 if end == "from" else 1]
            field.setText("" if value is None else f"{value:g}")
        self.message.setText(note or "Empty: automatic, fitted to what is drawn. A typed range stays until Auto.")
        self.adjustSize()
        if anchor is not None:
            self.move(anchor.mapToGlobal(anchor.rect().bottomRight()) - QPointF(self.width(), -4).toPoint())
        self.show()

    def apply(self) -> None:
        try:
            axes = parse(self.rows, {key: field.text() for key, field in self.fields.items()})
        except ValueError as exc:
            self.message.setText(f"Not applied: {exc}.")
            return
        axes.update({key: values[pills.current()] for key, (values, pills) in self.choices.items()})
        self.applied.emit(axes)
        self.hide()

    def auto(self) -> None:
        """Every range automatic; a choice (the time unit) is a unit, not a range, and stays as picked."""
        for field in self.fields.values():
            field.setText("")
        out = {row["key"]: None for row in self.rows}
        out.update({key: values[pills.current()] for key, (values, pills) in self.choices.items()})
        self.applied.emit(out)
        self.hide()
