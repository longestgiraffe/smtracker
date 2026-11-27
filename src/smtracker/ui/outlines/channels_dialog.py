"""The editor's image channels: which folder the outlines are drawn on, and up to three
fluorescence folders shown beside it.

Asked for: "options to change BF, and up to 3 other fluorescence images, just define the
folder name there, so instead of sum, I could write for example gfp or rfp, if other labs
do it like this. these options could be saveable". The names are folder NAMES looked up
beside every condition's images, so a path is refused; the set is remembered per user and
read at the next start. THIN QT: the rules are `segmentation.editor_project`'s
(`channels_problem`, `normalise_channels`).
"""
from __future__ import annotations

from qtpy.QtCore import Signal
from qtpy.QtWidgets import QGridLayout, QHBoxLayout, QLineEdit

from ...segmentation.editor_project import MAX_FLUOR, channels_problem
from ..theme import typography as ty
from ..widgets.controls import Button
from ..widgets.dialogs import CloseButton, OverlayDialog
from ..widgets.primitives import label

__all__ = ["ChannelsDialog"]


class _NameField(QLineEdit):
    """A folder-name input in the shadcn Input styling."""

    def __init__(self, t, text: str, placeholder: str):
        super().__init__(text)
        self.setFont(ty.ui("sm"))
        self.setPlaceholderText(placeholder)
        self.setFixedSize(200, t.control_height)
        self.setStyleSheet(
            f"QLineEdit {{ background:{t.card}; color:{t.foreground};"
            f" border:1px solid {t.input}; border-radius:{t.radius_md:g}px;"
            f" padding:0px 10px; }}")


class ChannelsDialog(OverlayDialog):
    """Four folder names, checked as typed; Save hands them back and closes."""

    saved = Signal(dict)

    def __init__(self, t, channels: dict, *, content=None, parent=None):
        super().__init__(t, content, parent=parent)
        self.t = t
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.addWidget(label("Image channels", ty.ui("base", "semibold"), t.foreground))
        head.addStretch(1)
        close = CloseButton(t)
        close.clicked.connect(self._dismiss)
        head.addWidget(close)
        self.card_lay.addLayout(head)
        note = label("Names of the folders next to each condition's images. They are "
                     "remembered for the next session.", ty.ui("sm"), t.muted_foreground)
        note.setWordWrap(True)
        note.setMaximumWidth(460)
        self.card_lay.addWidget(note)

        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(8)
        fluors = list(channels.get("fluor", []))
        self.bright = _NameField(t, channels.get("bright", ""), "phase")
        self.fluors = [_NameField(t, fluors[k] if k < len(fluors) else "",
                                  "sum" if k == 0 else "optional")
                       for k in range(MAX_FLUOR)]
        rows = [("Brightfield", self.bright, "outlines are drawn on this")]
        rows += [(f"Fluorescence {k + 1}", field,
                  "shown as a view and as an overlay on the brightfield" if k == 0 else "optional")
                 for k, field in enumerate(self.fluors)]
        for r, (name, field, what) in enumerate(rows):
            grid.addWidget(label(name, ty.ui("sm", "medium"), t.foreground), r, 0)
            grid.addWidget(field, r, 1)
            grid.addWidget(label(what, ty.ui("xs"), t.muted_foreground), r, 2)
            field.textChanged.connect(lambda _text: self._refresh())
        self.card_lay.addLayout(grid)

        footer = QHBoxLayout()
        footer.setSpacing(8)
        self.hint = label("", ty.ui("xs"), t.muted_foreground)
        footer.addWidget(self.hint)
        footer.addStretch(1)
        cancel = Button(t, "Cancel", variant="outline")
        cancel.clicked.connect(self._dismiss)
        self.save_button = Button(t, "Save", variant="default")
        self.save_button.clicked.connect(self._save)
        footer.addWidget(cancel)
        footer.addWidget(self.save_button)
        self.card_lay.addLayout(footer)
        self._refresh()

    def values(self) -> dict:
        return {"bright": self.bright.text().strip(),
                "fluor": [f.text().strip() for f in self.fluors if f.text().strip()]}

    def _refresh(self) -> None:
        value = self.values()
        problem = channels_problem(value["bright"], value["fluor"])
        self.hint.setText(problem)
        self.save_button.setEnabled(not problem)

    def _save(self) -> None:
        value = self.values()
        if channels_problem(value["bright"], value["fluor"]):
            return
        self._acted = True
        self.saved.emit(value)
        self._dismiss()
