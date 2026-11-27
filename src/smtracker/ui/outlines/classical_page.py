"""The outlines window's CLASSICAL SETTINGS page: the classical segmenter's sliders, beside a live preview of what a
run would draw on a field of an open condition (asked 15 Oct 2026: "can we try the manual version? ... i mean with
the settings"). Thin Qt: the settings, the preview and the words are `segmentation.classical_view`'s; this paints
them and says when the settings are saved."""
from __future__ import annotations

from qtpy.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from qtpy.QtGui import QColor, QImage, QPainter, QPen, QPolygonF
from qtpy.QtWidgets import QComboBox, QGridLayout, QHBoxLayout, QSizePolicy, QVBoxLayout, QWidget

from ...segmentation import classical_view as cv
from ..theme import typography as ty
from ..widgets import tooltip
from ..widgets.controls import Button, Slider
from ..widgets.primitives import Card, fit_label, label
from .segment_page import _combo_css

__all__ = ["ClassicalPage"]


class _Preview(QWidget):
    """The field's image at the widget's size, the outlines a run would write (cyan) and the dropped candidates (a
    small red cross each)."""

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.setMinimumSize(320, 320)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._payload = None
        self._image = None
        self._message = "Open a folder and pick a condition."

    def set_message(self, text: str) -> None:
        self._payload, self._image, self._message = None, None, text
        self.update()

    def set_payload(self, payload: dict) -> None:
        self._payload = payload
        self._image = QImage(payload["pixels"], payload["width"], payload["height"], payload["width"],
                             QImage.Format_Grayscale8).copy()
        self.update()

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(self.t.muted))
        if self._payload is None or self._image is None:
            p.setPen(QColor(self.t.muted_foreground))
            p.setFont(ty.ui("sm"))
            p.drawText(self.rect().adjusted(20, 20, -20, -20), Qt.AlignCenter | Qt.TextWordWrap, self._message)
            p.end()
            return
        w, h = self._payload["width"], self._payload["height"]
        k = min(self.width() / w, self.height() / h)
        ox, oy = (self.width() - w * k) / 2.0, (self.height() - h * k) / 2.0
        p.drawImage(QRectF(ox, oy, w * k, h * k), self._image)
        p.setRenderHint(QPainter.Antialiasing, True)

        def at(x, y):
            return QPointF(ox + (x + 0.5) * k, oy + (y + 0.5) * k)

        p.setPen(QPen(QColor("#22d3ee"), 1.6))
        p.setBrush(Qt.NoBrush)
        for outline in self._payload["outlines"]:
            p.drawPolygon(QPolygonF([at(x, y) for x, y in outline]))
        p.setPen(QPen(QColor("#ef4444"), 1.6))
        for d in self._payload["dropped"]:
            c = at(d["x"], d["y"])
            p.drawLine(c + QPointF(-4, -4), c + QPointF(4, 4))
            p.drawLine(c + QPointF(-4, 4), c + QPointF(4, -4))
        p.end()


class ClassicalPage(QWidget):
    """``settings_saved(dict)``: the settings to remember and use for every classical run."""

    settings_saved = Signal(object)

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.setStyleSheet("background:transparent;")
        self._tips: list = []
        self._conditions: dict = {}
        self._params = cv.params_from(None)
        self._saved = cv.to_saved(self._params)
        self._syncing = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self._update_preview)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)
        lay.addWidget(label("Classical settings", ty.ui("lg", "semibold"), t.foreground))
        intro = label("Finds glowing cells on the sum image by threshold and shape. Tune it on one "
                      "field; the preview shows what a run would write. Cyan lines are outlines, red "
                      "crosses are dropped candidates (reasons below).", ty.ui("sm"), t.muted_foreground)
        intro.setWordWrap(True)
        lay.addWidget(intro)

        pick = QHBoxLayout()
        pick.setSpacing(8)
        pick.addWidget(label("Condition", ty.ui("sm"), t.muted_foreground))
        self.condition_box = QComboBox()
        self.condition_box.setFont(ty.ui("sm"))
        self.condition_box.setStyleSheet(_combo_css(t))
        self.condition_box.setMinimumWidth(240)
        self.condition_box.currentTextChanged.connect(self._condition_picked)
        pick.addWidget(self.condition_box)
        pick.addWidget(label("Field", ty.ui("sm"), t.muted_foreground))
        self.field_box = QComboBox()
        self.field_box.setFont(ty.ui("sm"))
        self.field_box.setStyleSheet(_combo_css(t))
        self.field_box.setMinimumWidth(140)
        self.field_box.currentTextChanged.connect(lambda _t: self._schedule())
        pick.addWidget(self.field_box)
        self.prev_field = Button(t, "", size="sm", variant="ghost", glyph="chevron-left")
        self.next_field = Button(t, "", size="sm", variant="ghost", glyph="chevron-right")
        self.prev_field.clicked.connect(lambda: self._step(-1))
        self.next_field.clicked.connect(lambda: self._step(1))
        pick.addWidget(self.prev_field)
        pick.addWidget(self.next_field)
        pick.addStretch(1)
        lay.addLayout(pick)

        body = QHBoxLayout()
        body.setSpacing(16)
        left = QVBoxLayout()
        left.setSpacing(6)
        self.preview = _Preview(t)
        left.addWidget(self.preview, 1)
        self.caption = label("", ty.ui("sm", "medium"), t.foreground)
        left.addWidget(self.caption)
        self.dropped = label("", ty.ui("xs"), t.muted_foreground)
        self.dropped.setWordWrap(True)
        left.addWidget(self.dropped)
        body.addLayout(left, 1)

        card = Card(t, shadowed=False)
        card.setFixedWidth(330)
        grid = QGridLayout(card)
        grid.setContentsMargins(14, 12, 14, 12)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        self.sliders, self.values = {}, {}
        for row, s in enumerate(cv.SETTINGS):
            name = label(s.label, ty.ui("sm"), t.foreground)
            self._tips.append(tooltip.attach(t, name, s.tip))
            value = label("", ty.ui("sm", tabular=True), t.muted_foreground)
            slider = Slider(t, 0, cv.slider_value(s, s.high), cv.slider_value(s, getattr(self._params, s.key)))
            self._tips.append(tooltip.attach(t, slider, s.tip))
            slider.changed.connect(lambda _v, key=s.key: self._slid(key))
            grid.addWidget(name, 2 * row, 0)
            grid.addWidget(value, 2 * row, 1, Qt.AlignRight)
            grid.addWidget(slider, 2 * row + 1, 0, 1, 2)
            self.sliders[s.key], self.values[s.key] = slider, value
        buttons = QHBoxLayout()
        self.reset_button = Button(t, "Reset to defaults", size="sm")
        self.reset_button.clicked.connect(self.reset)
        self.save_button = Button(t, "Use these settings", size="sm", variant="default")
        self.save_button.clicked.connect(self.save)
        buttons.addWidget(self.reset_button)
        buttons.addStretch(1)
        buttons.addWidget(self.save_button)
        grid.addLayout(buttons, 2 * len(cv.SETTINGS), 0, 1, 2)
        # `fit_label`: its height pinned to its two lines at the card's width, so a short window (1280 x 720) never
        # lays the card out from a one-line guess and draws the note over the buttons (30 Oct 2026)
        self.state = fit_label("", ty.ui("xs"), t.muted_foreground)
        grid.addWidget(self.state, 2 * len(cv.SETTINGS) + 1, 0, 1, 2)
        body.addWidget(card, 0, Qt.AlignTop)
        lay.addLayout(body, 1)
        self._show_values()

    # -- what the window tells the page ------------------------------------------------ #
    def set_conditions(self, conditions: dict, current: str | None = None) -> None:
        """``{condition: (fields, image_path(field) -> Path | None)}`` of the open folder."""
        self._conditions = dict(conditions)
        names = list(self._conditions)
        self._syncing = True
        try:
            if [self.condition_box.itemText(i) for i in range(self.condition_box.count())] != names:
                self.condition_box.clear()
                self.condition_box.addItems(names)
            if current in self._conditions:
                self.condition_box.setCurrentText(current)
        finally:
            self._syncing = False
        self._condition_picked(self.condition_box.currentText())

    def set_settings(self, saved: dict | None) -> None:
        """The remembered settings (None: the defaults)."""
        self._params = cv.params_from(saved)
        self._saved = cv.to_saved(self._params)
        self._syncing = True
        try:
            for s in cv.SETTINGS:
                self.sliders[s.key].set_value(cv.slider_value(s, getattr(self._params, s.key)))
        finally:
            self._syncing = False
        self._show_values()
        self._schedule()

    def settings(self) -> dict:
        return cv.to_saved(self._params)

    # -- the user ---------------------------------------------------------------------- #
    def _condition_picked(self, name: str) -> None:
        if self._syncing:
            return
        fields = list(self._conditions.get(name, ((), None))[0])
        self.field_box.blockSignals(True)
        self.field_box.clear()
        self.field_box.addItems(fields)
        self.field_box.blockSignals(False)
        self._schedule()

    def _step(self, step: int) -> None:
        n = self.field_box.count()
        if n:
            self.field_box.setCurrentIndex((self.field_box.currentIndex() + step) % n)

    def _slid(self, key: str) -> None:
        if self._syncing:
            return
        from dataclasses import replace

        setting = next(s for s in cv.SETTINGS if s.key == key)
        self._params = replace(self._params, **{key: cv.value_of(setting, self.sliders[key].value())})
        self._show_values()
        self._schedule()

    def reset(self) -> None:
        """The sliders back to the defaults, NOT remembered until "Use these settings".

        `set_settings(None)` takes its argument as what is already saved, so using it here
        marked the defaults as saved ("Saved.", the button off) while the window still
        used, and the next run read, the old settings (the UI designer, 26 Nov 2026). The saved
        settings stay what they were; the page says the defaults are not saved yet.
        """
        remembered = self._saved
        self.set_settings(None)
        self._saved = remembered
        self._show_values()

    def save(self) -> None:
        self._saved = cv.to_saved(self._params)
        self.settings_saved.emit(dict(self._saved))
        self._show_values()

    # -- showing ----------------------------------------------------------------------- #
    def _show_values(self) -> None:
        for s in cv.SETTINGS:
            self.values[s.key].setText(cv.shown(s, getattr(self._params, s.key)))
        unsaved = cv.to_saved(self._params) != self._saved
        self.save_button.setEnabled(unsaved)
        self.state.setText("Not saved yet. Segment and Batch still use the saved settings." if unsaved else
                           "Saved. Segment and Batch use these settings.")

    def _schedule(self) -> None:
        self._timer.start()

    def _update_preview(self) -> None:
        name, field = self.condition_box.currentText(), self.field_box.currentText()
        entry = self._conditions.get(name)
        if not entry or not field:
            self.preview.set_message("Open a folder and pick a condition.")
            self.caption.setText("")
            self.dropped.setText("")
            return
        path = entry[1](field) if entry[1] else None
        if path is None:
            self.preview.set_message(f"{name} has no sum image for {field}. The classical segmenter reads the sum/ folder.")
            self.caption.setText("")
            self.dropped.setText("")
            return
        try:
            import tifffile

            payload = cv.preview(tifffile.imread(str(path)), self._params)
        except Exception as exc:                                  # noqa: BLE001 - shown, never fatal
            self.preview.set_message(f"{field}: {type(exc).__name__}: {exc}")
            return
        self.preview.set_payload(payload)
        self.caption.setText(payload["caption"])
        reasons = {}
        for d in payload["dropped"]:
            key = d["why"].split(";")[0]
            reasons[key] = reasons.get(key, 0) + 1
        self.dropped.setText("Dropped: " + "; ".join(f"{why} ({n})" if n > 1 else why
                                                     for why, n in list(reasons.items())[:6]) if reasons else "")
