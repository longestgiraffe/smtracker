"""The custom colour-ramp editor: pick 2-5 colours, see the ramp and its brightness, save it.

Asked for: "a custom setting where users can use custom colors and save and load". Opened
from the Visibility tool's Colormap row ("Custom…"). THIN QT: building the ramp, the
brightness check, the store's JSON shape and the import/export files are all
`smtracker.colour_ramps`; this module draws them and asks for colours and file names.

The ramps are the user's and live per user (`ui.json`, through the window), like the
condition colours; Export/Import move them between people as a small file.
"""
from __future__ import annotations

from qtpy.QtCore import QRectF, Signal
from qtpy.QtGui import QColor, QPainter
from qtpy.QtWidgets import (
    QColorDialog,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QWidget,
)

from ...colour_ramps import (
    MAX_STOPS,
    MIN_STOPS,
    CustomRamp,
    RampError,
    brightness_note,
    export_ramps,
    import_ramps,
    sample,
)
from ..theme import typography as ty
from ..widgets.controls import Button
from ..widgets.dialogs import CloseButton, OverlayDialog
from ..widgets.primitives import label
from ..widgets.settings import WARN_INK
from ..widgets.swatches import Swatch

__all__ = ["RampDialog", "RampPreview"]

#: The ramp a new one starts from: light to dark blue, which passes the brightness check.
NEW_STOPS = ("#f7fbff", "#6baed6", "#08306b")
NEW_ITEM = "New ramp"


class RampPreview(QWidget):
    """The ramp as the maps will paint it (`colour_ramps.sample`)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(26)
        self._ramp: tuple = ()

    def set_ramp(self, ramp) -> None:
        self._ramp = tuple(ramp)
        self.update()

    def paintEvent(self, _e) -> None:
        if not self._ramp:
            return
        p = QPainter(self)
        n, w = len(self._ramp), self.width()
        for i, colour in enumerate(self._ramp):
            p.fillRect(QRectF(i * w / n, 0, w / n + 1.0, float(self.height())),
                       QColor("#" + colour))
        p.end()


def _combo_css(t) -> str:
    return (f"QComboBox {{ background:{t.card}; color:{t.foreground};"
            f" border:1px solid {t.input}; border-radius:{t.radius_md:g}px;"
            f" padding:3px 8px; min-height:{t.control_height_sm - 8}px; }}"
            f"QComboBox QAbstractItemView {{ background:{t.card}; color:{t.foreground};"
            f" selection-background-color:{t.accent}; selection-color:{t.foreground}; }}"
            f"QComboBox::drop-down {{ border:none; width:22px; }}")


class RampDialog(OverlayDialog):
    """Edit the user's custom ramps. `saved(list)` carries the whole list after every
    save, delete or import (the window stores it); `used(str)` a ramp's key to paint with."""

    saved = Signal(list)
    used = Signal(str)

    def __init__(self, t, *, ramps, current: str | None = None, content=None, parent=None):
        super().__init__(t, content, parent=parent)
        self.t = t
        self._card.setMinimumWidth(520)
        self._ramps: list = list(ramps)
        self._stops: list = list(NEW_STOPS)

        head = QHBoxLayout()
        head.addWidget(label("Custom colour ramps", ty.ui("lg", "semibold"), t.foreground))
        head.addStretch(1)
        close = CloseButton(t)
        close.clicked.connect(self._dismiss)
        head.addWidget(close)
        self.card_lay.addLayout(head)
        sub = label("Pick two to five colours, low values first. The maps paint the ramp "
                    "below; saved ramps stay on this computer and can be exported to share.",
                    ty.ui("sm"), t.muted_foreground)
        sub.setWordWrap(True)
        self.card_lay.addWidget(sub)

        pick = QHBoxLayout()
        pick.addWidget(label("Ramp", ty.ui("sm", "medium"), t.muted_foreground))
        self.choice = QComboBox()
        self.choice.setStyleSheet(_combo_css(t))
        self.choice.setFont(ty.ui("sm"))
        self.choice.currentTextChanged.connect(self._on_choice)
        pick.addWidget(self.choice, 1)
        self.card_lay.addLayout(pick)

        name_row = QHBoxLayout()
        name_row.addWidget(label("Name", ty.ui("sm", "medium"), t.muted_foreground))
        self.name = QLineEdit("")
        self.name.setPlaceholderText("e.g. Blues")
        self.name.setFont(ty.ui("sm"))
        self.name.setStyleSheet(
            f"QLineEdit {{ background:{t.card}; color:{t.foreground};"
            f" border:1px solid {t.input}; border-radius:{t.radius_md:g}px; padding:4px 8px; }}")
        self.name.textChanged.connect(lambda _t: self._refresh())
        name_row.addWidget(self.name, 1)
        self.card_lay.addLayout(name_row)

        stops_row = QHBoxLayout()
        stops_row.addWidget(label("Colours", ty.ui("sm", "medium"), t.muted_foreground))
        self._swatch_box = QHBoxLayout()
        self._swatch_box.setSpacing(2)
        stops_row.addLayout(self._swatch_box)
        stops_row.addStretch(1)
        self.add_stop = Button(t, "Add colour", variant="outline", size="small")
        self.add_stop.clicked.connect(self._add_stop)
        self.remove_stop = Button(t, "Remove last", variant="outline", size="small")
        self.remove_stop.clicked.connect(self._remove_stop)
        stops_row.addWidget(self.add_stop)
        stops_row.addWidget(self.remove_stop)
        self.card_lay.addLayout(stops_row)

        self.preview = RampPreview()
        self.card_lay.addWidget(self.preview)
        self.note = label("", ty.ui("xs"), t.muted_foreground)
        self.note.setWordWrap(True)
        self.card_lay.addWidget(self.note)
        self.status = label("", ty.ui("xs"), t.muted_foreground)
        self.status.setWordWrap(True)
        self.card_lay.addWidget(self.status)

        foot = QHBoxLayout()
        self.import_button = Button(t, "Import…", variant="outline")
        self.import_button.clicked.connect(self._import)
        self.export_button = Button(t, "Export…", variant="outline")
        self.export_button.clicked.connect(self._export)
        self.delete_button = Button(t, "Delete", variant="outline")
        self.delete_button.clicked.connect(self._delete)
        foot.addWidget(self.import_button)
        foot.addWidget(self.export_button)
        foot.addWidget(self.delete_button)
        foot.addStretch(1)
        self.save_button = Button(t, "Save", variant="outline")
        self.save_button.clicked.connect(self._save)
        self.use_button = Button(t, "Save and use", variant="default")
        self.use_button.clicked.connect(self._use)
        foot.addWidget(self.save_button)
        foot.addWidget(self.use_button)
        self.card_lay.addLayout(foot)

        self._fill_choice(current_name=self._name_of(current))

    # -- state -------------------------------------------------------------------
    @staticmethod
    def _name_of(key: str | None) -> str | None:
        from ...colour_ramps import is_custom_key, name_of_key

        return name_of_key(key) if key and is_custom_key(key) else None

    def _fill_choice(self, current_name: str | None = None) -> None:
        self.choice.blockSignals(True)
        self.choice.clear()
        self.choice.addItems([NEW_ITEM] + [r.name for r in self._ramps])
        target = current_name if current_name in {r.name for r in self._ramps} else (
            self._ramps[0].name if self._ramps else NEW_ITEM)
        self.choice.setCurrentText(target)
        self.choice.blockSignals(False)
        self._on_choice(target)

    def _on_choice(self, text: str) -> None:
        ramp = next((r for r in self._ramps if r.name == text), None)
        self.name.setText("" if ramp is None else ramp.name)
        self._stops = list(NEW_STOPS if ramp is None else ramp.stops)
        self._rebuild_swatches()
        self._refresh()

    def _rebuild_swatches(self) -> None:
        while self._swatch_box.count():
            item = self._swatch_box.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        for i, colour in enumerate(self._stops):
            swatch = Swatch(self.t, colour, i)
            swatch.setToolTip("Click to change this colour")
            swatch.picked.connect(self._pick_colour)
            self._swatch_box.addWidget(swatch)
        self.add_stop.setEnabled(len(self._stops) < MAX_STOPS)
        self.remove_stop.setEnabled(len(self._stops) > MIN_STOPS)

    def candidate(self) -> CustomRamp | None:
        """The ramp on screen, or None while it is not a valid one (the status says why)."""
        try:
            return CustomRamp(self.name.text(), tuple(self._stops))
        except RampError as exc:
            self.status.setText(str(exc))
            return None

    def _refresh(self) -> None:
        ramp = sample(self._stops)
        self.preview.set_ramp(ramp)
        warning = brightness_note(ramp)
        self.note.setText(("Warning: " + warning) if warning else
                          "Brightness rises or falls steadily along this ramp: it reads in "
                          "greyscale and every value has its own brightness.")
        self.note.setStyleSheet(f"color:{WARN_INK if warning else self.t.muted_foreground};")
        self.status.setText("")
        ok = self.candidate() is not None
        self.save_button.setEnabled(ok)
        self.use_button.setEnabled(ok)
        stored = self.choice.currentText() != NEW_ITEM
        self.delete_button.setEnabled(stored)
        self.export_button.setEnabled(bool(self._ramps))

    # -- actions -------------------------------------------------------------------
    def _pick_colour(self, index: int) -> None:
        chosen = QColorDialog.getColor(QColor(self._stops[index]), self, "Ramp colour")
        if chosen.isValid():
            self._set_stop(index, chosen.name())

    def _set_stop(self, index: int, colour: str) -> None:
        self._stops[index] = colour
        self._rebuild_swatches()
        self._refresh()

    def _add_stop(self) -> None:
        if len(self._stops) < MAX_STOPS:
            self._stops.append(self._stops[-1])
            self._rebuild_swatches()
            self._refresh()

    def _remove_stop(self) -> None:
        if len(self._stops) > MIN_STOPS:
            self._stops.pop()
            self._rebuild_swatches()
            self._refresh()

    def _save(self) -> CustomRamp | None:
        ramp = self.candidate()
        if ramp is None:
            return None
        editing = self.choice.currentText()
        kept = [r for r in self._ramps
                if r.name.lower() != ramp.name.lower() and r.name != editing]
        self._ramps = sorted(kept + [ramp], key=lambda r: r.name.lower())
        self.saved.emit(list(self._ramps))
        self._fill_choice(current_name=ramp.name)
        self.status.setText(f"saved {ramp.name}")
        return ramp

    def _use(self) -> None:
        ramp = self._save()
        if ramp is None:
            return
        self._acted = True
        self.used.emit(ramp.key)
        self._dismiss()

    def _delete(self) -> None:
        name = self.choice.currentText()
        if name == NEW_ITEM:
            return
        self._ramps = [r for r in self._ramps if r.name != name]
        self.saved.emit(list(self._ramps))
        self._fill_choice()
        self.status.setText(f"deleted {name}")

    def _export(self) -> None:
        path, _filter = QFileDialog.getSaveFileName(self, "Export colour ramps",
                                                    "colour_ramps.json", "JSON (*.json)")
        if path:
            export_ramps(path, self._ramps)
            self.status.setText(f"exported {len(self._ramps)} ramp(s) to {path}")

    def _import(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(self, "Import colour ramps", "",
                                                    "JSON (*.json)")
        if path:
            self.import_file(path)

    def import_file(self, path) -> None:
        """Add the ramps in ``path``; a ramp of the same name is replaced and named."""
        try:
            incoming = import_ramps(path)
        except RampError as exc:
            self.status.setText(str(exc))
            return
        names = {r.name.lower() for r in incoming}
        replaced = [r.name for r in self._ramps if r.name.lower() in names]
        self._ramps = sorted([r for r in self._ramps if r.name.lower() not in names] + incoming,
                             key=lambda r: r.name.lower())
        self.saved.emit(list(self._ramps))
        self._fill_choice(current_name=incoming[0].name if incoming else None)
        self.status.setText(f"imported {len(incoming)} ramp(s)"
                            + (f"; replaced {', '.join(replaced)}" if replaced else ""))
