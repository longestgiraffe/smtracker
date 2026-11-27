"""Batch: segment the fields WITHOUT an outline in every condition under several parent
folders, one condition after the other -- overnight, say.

Asked for (13 Oct 2026): "a batch mode where u can run multiple parent folders in batches
overnight. Both for image prep / tracking and segmentation". A run never replaces an outline
unless Replace is ticked (the Segment page's rule): it segments only the fields with no
`cell_meshes/<field>.mat`, keeping every outline there, hand-corrected ones included.

THIN QT. The shared page (`ui.widgets.batch.BatchPage`) plus Replace and the model line.
Every condition uses the Options' standard model (its folder follows the model); the rows,
the readiness and the words are `smtracker.batch_model`'s; the window runs it.
"""
from __future__ import annotations

from qtpy.QtCore import Signal

from ..theme import typography as ty
from ..widgets.batch import BatchPage
from ..widgets.primitives import Checkbox, label

__all__ = ["OutlineBatchPage", "OUTLINE_COLUMNS"]

OUTLINE_COLUMNS = ("Folder", "Condition", "Layout", "Fields", "Without outline", "Model",
                   "Status")


class OutlineBatchPage(BatchPage):
    """The segmentation batch: the shared page, Replace, and which model runs."""

    replace_changed = Signal(bool)

    def __init__(self, t, parent=None):
        super().__init__(
            t, title="Batch", columns=OUTLINE_COLUMNS, run_text="Run batch",
            stop_text="Stop after the field in hand", ignorable=True,
            intro=("Segment every condition under several parent folders, one after the "
                   "other. Fields that already have an outline, including hand-corrected "
                   "ones, are kept unless Replace is ticked. Each condition uses the model "
                   "set in Options. Ignore leaves a condition out until you press Include."),
            parent=parent)
        s = self.settings
        self.replace = Checkbox(t, checked=False)
        self.replace.toggled.connect(self.replace_changed.emit)
        s.addLayout(self.group(self.replace,
                               label("Replace existing outlines", ty.ui("sm"), t.foreground)))
        s.addStretch(1)

    def set_busy(self, busy: bool, *, can_run: bool = True) -> None:
        super().set_busy(busy, can_run=can_run)
        self.replace.setEnabled(not busy)
