"""The pixel size from the camera and the optics, as an in-window dialog.

The one calibration number that cannot be read off the optics: NA and wavelength give
the spot's WIDTH, not the sampling, and the .nd2's own value drops the tube lens
(16 um / 100x = 160 nm against 16 um / 150x = 106.7 with the ~1.5x tube lens -- every D
off by 2.28x). So the arithmetic is shown LIVE, before anything is applied, with the
trap named when the tube lens is left at 1, and the camera pixel and objective are
pre-filled from the acquisition where it records them: the user supplies the one term
the file drops.

THIN QT: the arithmetic is `prep_model.camera_pixel_size`, the words
`prep_view.pixel_size_line`.
"""
from __future__ import annotations

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QGridLayout, QHBoxLayout

from ...prep_model import PrepError, camera_pixel_size
from ...prep_view import pixel_size_line
from ..theme import typography as ty
from ..widgets.controls import Button, FloatField, Segmented
from ..widgets.dialogs import CloseButton, OverlayDialog
from ..widgets.primitives import label

__all__ = ["PixelSizeDialog"]


class PixelSizeDialog(OverlayDialog):
    """Camera pixel / (objective x tube lens), applied to one condition or all."""

    #: the pixel size in nm, and True to apply it to every condition
    applied = Signal(float, bool)

    def __init__(self, t, *, condition: str, camera: dict, content=None, parent=None):
        super().__init__(t, content, parent=parent)
        self._card.setMinimumWidth(560)
        head = QHBoxLayout()
        head.addWidget(label("Pixel size from the camera", ty.ui("lg", "semibold"),
                             t.foreground))
        head.addStretch(1)
        close = CloseButton(t)
        close.clicked.connect(self._dismiss)
        head.addWidget(close)
        self.card_lay.addLayout(head)
        sub = label("The sample-plane pixel is the camera's pixel divided by the whole "
                    "magnification, tube lens included. It is not read from a .nd2, which can "
                    "report a different value.", ty.ui("sm"),
                    t.muted_foreground)
        sub.setWordWrap(True)
        self.card_lay.addWidget(sub)

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)
        self.camera = FloatField(t, 0.0, 100.0, float(camera.get("camera_pixel_um") or 0.0),
                                 step=0.1, decimals=2, suffix=" µm", width=150)
        self.objective = FloatField(t, 0.0, 250.0,
                                    float(camera.get("objective_magnification") or 0.0),
                                    step=1.0, decimals=1, suffix=" x", width=150)
        self.tube = FloatField(t, 0.0, 10.0, 1.0, step=0.05, decimals=2, suffix=" x",
                               width=150)
        self.tube.setToolTip("The magnifier between objective and camera: 1.0 if there is "
                             "none. Leaving it at 1.0 when there is one makes the pixel size too "
                             "large, as in some .nd2 files.")
        rows = (("Camera pixel", self.camera,
                 "from the acquisition" if camera.get("camera_pixel_um") else "Andor DU-897: 16 µm"),
                ("Objective", self.objective,
                 "from the acquisition" if camera.get("objective_magnification") else ""),
                ("Tube lens", self.tube, "the factor a .nd2 omits"))
        for r, (name, field, note) in enumerate(rows):
            grid.addWidget(label(name, ty.ui("sm", "medium"), t.muted_foreground), r, 0)
            grid.addWidget(field, r, 1)
            grid.addWidget(label(note, ty.ui("xs"), t.muted_foreground), r, 2)
            field.changed.connect(lambda _v: self._refresh())
        grid.setColumnStretch(2, 1)
        self.card_lay.addLayout(grid)
        self.result = label("", ty.ui("sm", "semibold", tabular=True), t.foreground)
        self.result.setWordWrap(True)
        self.result.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.card_lay.addWidget(self.result)

        foot = QHBoxLayout()
        foot.addWidget(label("Apply to", ty.ui("sm", "medium"), t.muted_foreground))
        self.scope = Segmented(t, [condition, "every condition"], current=0, size="sm")
        foot.addWidget(self.scope)
        foot.addStretch(1)
        cancel = Button(t, "Cancel", variant="outline")
        cancel.clicked.connect(self._dismiss)
        self.apply_button = Button(t, "Apply", variant="default")
        self.apply_button.clicked.connect(self._apply)
        foot.addWidget(cancel)
        foot.addWidget(self.apply_button)
        self.card_lay.addLayout(foot)
        self._refresh()

    def value(self) -> float | None:
        try:
            return camera_pixel_size(self.camera.value(), self.objective.value(),
                                     self.tube.value())
        except PrepError:
            return None

    def _refresh(self) -> None:
        value = self.value()
        self.result.setText(pixel_size_line(self.camera.value(), self.objective.value(),
                                            self.tube.value(), value))
        self.apply_button.setEnabled(value is not None)

    def _apply(self) -> None:
        value = self.value()
        if value is None:
            return
        self._acted = True
        self.applied.emit(float(value), self.scope.current() == 1)
        self._dismiss()
