"""Images & tracks: the experimenter's own fluorescence channels (agreed 17 Oct 2026: "just the different fluor
channels can be added and named and folder direction can be given for them").

One row per channel: its NAME (what the Background pill says) and its FOLDER, typed or picked with Browse. A folder
inside a condition's folder is kept by its path from there, so every condition looks in its own folder of that name
(as ``phase/`` and ``sum/``); a folder elsewhere is kept as it is. THIN QT: the rules are `images_view`'s
(`channels_problem`, `channel_folder_for`); the window remembers the set per project.
"""
from __future__ import annotations

from qtpy.QtCore import Signal
from qtpy.QtWidgets import QFileDialog, QGridLayout, QHBoxLayout, QLineEdit

from ...images_view import channel_folder_for, channels_problem
from ..theme import typography as ty
from ..widgets.controls import Button
from ..widgets.dialogs import CloseButton, OverlayDialog
from ..widgets.primitives import discard, label

__all__ = ["ImageChannelsDialog", "MAX_CHANNELS"]

#: As many channels as a bar of pills holds beside Phase, Sum and Movie.
MAX_CHANNELS = 4


class _Field(QLineEdit):
    """An input in the shadcn Input styling."""

    def __init__(self, t, text: str, placeholder: str, width: int):
        super().__init__(text)
        self.setFont(ty.ui("sm"))
        self.setPlaceholderText(placeholder)
        self.setFixedSize(width, t.control_height)
        self.setStyleSheet(
            f"QLineEdit {{ background:{t.card}; color:{t.foreground};"
            f" border:1px solid {t.input}; border-radius:{t.radius_md:g}px;"
            f" padding:0px 10px; }}")


class ImageChannelsDialog(OverlayDialog):
    """The channel rows, checked as typed; Save hands back ``[{"name", "folder"}]`` and closes."""

    saved = Signal(list)

    def __init__(self, t, channels: list, roots: list, *, content=None, parent=None):
        super().__init__(t, content, parent=parent)
        self.t = t
        self._roots = [str(r) for r in roots]
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.addWidget(label("Fluorescence channels", ty.ui("base", "semibold"), t.foreground))
        head.addStretch(1)
        close = CloseButton(t)
        close.clicked.connect(self._dismiss)
        head.addWidget(close)
        self.card_lay.addLayout(head)
        # EXPLICIT LINES, NOT WORD WRAP: a wrapped note reported less height than it took, and the card squeezed the
        # channel rows over each other to make up for it.
        note = label("Each channel becomes a Background beside Phase, Sum and Movie. A movie's image is\n"
                     "<movie>.tif in its folder. A folder inside a condition's folder is looked up in every\n"
                     "condition, as phase and sum are; a folder elsewhere is used as given. Saved with the project.",
                     ty.ui("sm"), t.muted_foreground)
        self.card_lay.addWidget(note)
        # THE GRID IN THE CARD'S OWN LAYOUT, as the outlines' channel dialog has it: inside a host widget the card
        # gave the rows 66 px for their 80 and drew the second over the first.
        self._grid = QGridLayout()
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(10)
        self._grid.setVerticalSpacing(8)
        self.card_lay.addLayout(self._grid)
        self.rows: list[tuple] = []
        for ch in channels or []:
            self.add_row(ch.get("name", ""), ch.get("folder", ""))
        if not self.rows:
            self.add_row("", "")
        self.add_button = Button(t, "Add channel", size="small", variant="outline")
        self.add_button.clicked.connect(lambda: self.add_row("", ""))
        add = QHBoxLayout()
        add.addWidget(self.add_button)
        add.addStretch(1)
        self.card_lay.addLayout(add)

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

    # -- rows ------------------------------------------------------------------------------------- #
    def add_row(self, name: str, folder: str) -> None:
        if len(self.rows) >= MAX_CHANNELS:
            return
        t = self.t
        name_field = _Field(t, name, "name, e.g. mCherry", 160)
        folder_field = _Field(t, folder, "folder, e.g. mcherry", 260)
        browse = Button(t, "Browse…", size="small", variant="outline")
        remove = Button(t, "Remove", size="small", variant="ghost")
        row = (name_field, folder_field, browse, remove)
        browse.clicked.connect(lambda r=row: self._browse(r))
        remove.clicked.connect(lambda r=row: self.remove_row(r))
        for field in (name_field, folder_field):
            field.textChanged.connect(lambda _text: self._refresh())
        index = len(self.rows)
        for col, widget in enumerate(row):
            self._grid.addWidget(widget, index, col)
        self.rows.append(row)
        self._refresh()

    def remove_row(self, row) -> None:
        if row not in self.rows:
            return
        self.rows.remove(row)
        for widget in row:
            discard(widget)
        for index, kept in enumerate(self.rows):
            for col, widget in enumerate(kept):
                self._grid.addWidget(widget, index, col)
        self._refresh()

    def _browse(self, row) -> None:
        start = self._roots[0] if self._roots else ""
        picked = QFileDialog.getExistingDirectory(self, "The folder of this channel's images", start)
        if picked:
            self.set_folder(row, picked)

    def set_folder(self, row, picked: str) -> None:
        """A picked folder, kept relative to the condition folder it lies in (`channel_folder_for`)."""
        row[1].setText(channel_folder_for(picked, self._roots))

    # -- values ----------------------------------------------------------------------------------- #
    def values(self) -> list:
        """Every row with something typed, as ``[{"name", "folder"}]``."""
        out = []
        for name, folder, _b, _r in self.rows:
            if name.text().strip() or folder.text().strip():
                out.append({"name": name.text().strip(), "folder": folder.text().strip()})
        return out

    def _refresh(self) -> None:
        if not hasattr(self, "hint"):                 # the first rows are added before the footer exists
            return
        problem = channels_problem(self.values())
        self.hint.setText(problem or "")
        self.save_button.setEnabled(not problem)
        if hasattr(self, "add_button"):
            self.add_button.setEnabled(len(self.rows) < MAX_CHANNELS)

    def _save(self) -> None:
        value = self.values()
        if channels_problem(value):
            return
        self._acted = True
        self.saved.emit(value)
        self._dismiss()
