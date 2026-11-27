"""The model trainer's own window: pick what to train on, train or export, keep models.

THIN QT, the house rule: no numpy, pandas or scipy at module level and no geometry
anywhere. Every number on screen came from `segmentation.training` or
`segmentation.models`, both pure and covered headless, and the run happens on a worker
thread so the console fills as it goes.

    conditions        tick which to train on; each says how many corrections it has
    labels            meshes (what you corrected here) or a folder of label images
    base model        what to fine-tune FROM; fine-tuning never edits it
    held out          fields kept back, so the result can be judged
    [Build] [Train here] [Export…]
    saved models      name, what it saw, and whether it beat its base

A SEPARATE WINDOW RATHER THAN A PANEL, because the segmentation window is already the
place where a run is configured and watched, and training is a different job with a
different unit: it consumes CORRECTIONS across conditions, not movies within one. Bolting
it on would have made the condition table mean two things at once.

**THE CORRECTIONS ARE THE LABELS.** There is no labelling step: `masks.labels_from_outlines`
turns an outline a human fixed into a training label, using the same `inpolygon` rule the
CELL FILTER uses, so a label cannot disagree with the analysis about which pixels are cell.

**NOTHING HERE DECIDES WHETHER A MODEL IS GOOD.** `training.score_against_base` measures
the new model AND the one it came from on held-out fields, and the store keeps both. A
model that was never scored is listed as exactly that.
"""
from __future__ import annotations

__all__ = ["SCRATCH_LABEL", "TrainerWindow", "launch"]

#: The "start from" entry that means train a NEW model rather than fine-tune one.
#: Spelled out rather than an empty entry: "(none)" in a model picker reads as
#: "not chosen yet", and this is a deliberate and much more expensive choice.
SCRATCH_LABEL = ": train a NEW model from scratch: "

#: Held open so a parentless window is not collected the moment `launch` returns —
#: CPython drops the wrapper, Qt takes the C++ widget with it, and a running worker
#: thread is torn down mid-flight (exit 139). Released in `closeEvent`, and only when
#: the close is actually accepted.
_OPEN_WINDOWS = set()


def _build_window_class():
    """Qt imported inside, so this module imports on a machine with no GUI stack."""
    from pathlib import Path

    from qtpy.QtCore import QObject, Qt, QThread, Signal
    from qtpy.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                                QGroupBox, QHBoxLayout, QLabel, QListWidget,
                                QListWidgetItem, QMainWindow, QMessageBox,
                                QPlainTextEdit, QPushButton, QSpinBox, QVBoxLayout,
                                QWidget)

    class _Worker(QObject):
        """Trains off the GUI thread. It CALLS; it does not compute."""

        line = Signal(str)
        done = Signal(object)
        failed = Signal(str)

        def __init__(self, bundle):
            super().__init__()
            self._bundle = bundle

        def run(self):
            try:
                from .training import train_bundle

                path = train_bundle(self._bundle, on_line=self.line.emit)
                self.done.emit(path)
            except Exception as exc:                      # noqa: BLE001 - reported
                import traceback

                self.failed.emit(f"{type(exc).__name__}: {exc}\n"
                                 f"{traceback.format_exc()}")

    class TrainerWindow(QMainWindow):
        """Build a training set from corrections, train or export it, keep the models."""

        def __init__(self, root=None, parent=None):
            super().__init__(parent)
            self.setWindowTitle("SMTracker · model trainer")
            self.resize(880, 720)
            self._root = Path(root) if root else None
            self._conditions: list = []
            self._bundle = None
            self._thread = None
            self._worker = None

            central = QWidget()
            layout = QVBoxLayout(central)

            top = QHBoxLayout()
            self._open = QPushButton("Open folder…")
            self._open.clicked.connect(self._pick_folder)
            top.addWidget(self._open)
            self._header = QLabel("no folder chosen")
            self._header.setWordWrap(True)
            top.addWidget(self._header, 1)
            layout.addLayout(top)

            # ── what to train on ─────────────────────────────────────────────
            what = QGroupBox("1 · what to train on")
            what_layout = QVBoxLayout(what)
            self._conditions_list = QListWidget()
            self._conditions_list.setMinimumHeight(120)
            self._conditions_list.itemChanged.connect(lambda _i: self._refresh_counts())
            what_layout.addWidget(self._conditions_list)

            row = QHBoxLayout()
            row.addWidget(QLabel("labels from"))
            self._labels_from = QComboBox()
            # MESHES FIRST, because those are the outlines a human fixed IN THIS TOOL.
            # A folder of label images is the other way labels arrive, and the two are
            # not the same claim about what a model was trained on.
            self._labels_from.addItems(["cell_meshes (what you corrected)",
                                        "masks/ (label images)"])
            self._labels_from.currentTextChanged.connect(lambda _t: self._rescan())
            row.addWidget(self._labels_from, 1)
            row.addWidget(QLabel("images in"))
            self._image_folder = QComboBox()
            self._image_folder.setEditable(True)
            self._image_folder.addItems(["phase", "sum"])
            self._image_folder.currentTextChanged.connect(lambda _t: self._rescan())
            row.addWidget(self._image_folder)
            what_layout.addLayout(row)
            layout.addWidget(what)

            # ── how ──────────────────────────────────────────────────────────
            how = QGroupBox("2 · how")
            how_layout = QHBoxLayout(how)
            how_layout.addWidget(QLabel("start from"))
            self._base = QComboBox()
            from .segment import builtin_models
            PRETRAINED = builtin_models()

            # TWO DIFFERENT JOBS IN ONE CONTROL, and the wording has to say
            # which. Fine-tuning adjusts a model that already knows what a cell
            # looks like; a NEW model learns that from these images alone, and
            # the data cost is the difference. Asked for after the fine-tune
            # shipped: "what i mean is a window to train a new model".
            from . import backends

            # A NEW MODEL FROM SCRATCH is Omnipose's only (Cellpose 4 has no such route), so that entry is
            # offered only where Omnipose is installed; Cellpose-SAM is offered where Cellpose 4 is.
            self._base.addItems(list(PRETRAINED) + ([SCRATCH_LABEL] if backends.omnipose_importable() else []))
            self._base.setToolTip(
                "Fine-tuning starts from the chosen model and saves a new file. Training a "
                "new model starts from nothing and needs far more corrected data (hundreds "
                "of labelled images against tens). It is still scored against a "
                "pretrained model.")
            self._base.currentTextChanged.connect(self._base_changed)
            how_layout.addWidget(self._base, 1)

            how_layout.addWidget(QLabel("hold out"))
            self._holdout = QDoubleSpinBox()
            self._holdout.setRange(0.0, 0.5)
            self._holdout.setSingleStep(0.05)
            self._holdout.setValue(0.2)
            self._holdout.setToolTip(
                "The share of corrected fields kept out of training, so the result can be "
                "scored against the base model on fields neither has seen. At 0 a trained "
                "model cannot be scored.")
            how_layout.addWidget(self._holdout)

            how_layout.addWidget(QLabel("epochs"))
            self._epochs = QSpinBox()
            self._epochs.setRange(1, 2000)
            self._epochs.setValue(100)
            how_layout.addWidget(self._epochs)
            layout.addWidget(how)

            # ── do it ────────────────────────────────────────────────────────
            do = QHBoxLayout()
            self._build = QPushButton("Build training set")
            self._build.clicked.connect(self._build_clicked)
            do.addWidget(self._build)
            self._train = QPushButton("Train here")
            self._train.clicked.connect(self._train_clicked)
            self._train.setEnabled(False)
            do.addWidget(self._train)
            self._export = QPushButton("Export to…")
            self._export.clicked.connect(self._export_clicked)
            self._export.setEnabled(False)
            self._export.setToolTip(
                "Copy the training set to another machine, such as a cluster, a GPU "
                "computer or Google Drive for Colab. It holds its own images, labels, "
                "held-out fields, README and notebook, and does not need SMTracker.")
            do.addWidget(self._export)
            do.addStretch(1)
            layout.addLayout(do)

            self._runtime_note = QLabel("")
            self._runtime_note.setWordWrap(True)
            self._runtime_note.setStyleSheet("color: #888;")
            layout.addWidget(self._runtime_note)

            # ── models ───────────────────────────────────────────────────────
            saved = QGroupBox("3 · saved models")
            saved_layout = QVBoxLayout(saved)
            self._models = QListWidget()
            self._models.setMinimumHeight(90)
            saved_layout.addWidget(self._models)
            model_row = QHBoxLayout()
            self._save_model = QPushButton("Save a model file…")
            self._save_model.clicked.connect(self._save_model_clicked)
            self._save_model.setToolTip(
                "Copy a trained model into the model store with a record of the corrections "
                "it was trained on.")
            model_row.addWidget(self._save_model)
            self._delete_model = QPushButton("Delete")
            self._delete_model.clicked.connect(self._delete_model_clicked)
            model_row.addWidget(self._delete_model)
            model_row.addStretch(1)
            saved_layout.addLayout(model_row)
            layout.addWidget(saved)

            self._console = QPlainTextEdit()
            self._console.setReadOnly(True)
            self._console.setStyleSheet("font-family: Consolas, monospace;")
            self._console.setMinimumHeight(120)
            layout.addWidget(self._console, 1)

            self.setCentralWidget(central)
            self._refresh_models()
            self._check_runtime()
            if self._root is not None:
                self._rescan()
            else:
                self._say("Open a folder of conditions whose outlines you have corrected.")

        # -- plumbing ---------------------------------------------------------
        def _say(self, *lines) -> None:
            for line in lines:
                self._console.appendPlainText(str(line))

        def _check_runtime(self, engine=None) -> None:
            """Say up front whether training can happen here, and why not.

            NOT ON A THREAD, unlike the segmentation window's GPU box: this window is
            opened deliberately to train, so the answer is the first thing it owes and a
            `find_spec` plus one torch import is what that costs.
            """
            from .training import training_runtime_available

            ok, why = training_runtime_available(engine)
            self._train.setEnabled(False)
            self._runtime_note.setText(
                "A GPU is available here, so 'Train here' will fine-tune in this window."
                if ok else f"Cannot train in this window: {why}")
            self._can_train = ok

        def _base_changed(self, text: str) -> None:
            """Say what a NEW model costs, at the moment it is chosen rather than after
            a GPU has been spent on it."""
            from . import backends
            from .segment import SegmentationError
            from .training import FROM_SCRATCH_NOTE

            if text == SCRATCH_LABEL:
                self._say("", FROM_SCRATCH_NOTE)
            # THE ENGINE OF THE CHOSEN BASE decides whether this machine can train it (Cellpose-SAM needs
            # a GPU with about 6 GB; Omnipose runs only from source)
            try:
                engine = backends.OMNIPOSE_BACKEND if text == SCRATCH_LABEL else backends.backend_of(text)
            except SegmentationError:
                engine = None
            if hasattr(self, "_train"):
                self._check_runtime(engine)
                if self._bundle is not None:
                    self._train.setEnabled(bool(getattr(self, "_can_train", False)))

        def _pick_folder(self) -> None:
            chosen = QFileDialog.getExistingDirectory(self, "Folder holding conditions")
            if chosen:
                self._root = Path(chosen)
                self._rescan()

        # -- what to train on -------------------------------------------------
        def _label_kwargs(self) -> dict:
            return ({} if self._labels_from.currentIndex() == 0
                    else {"mask_folder": "masks"})

        def _rescan(self) -> None:
            """List the conditions and how many corrections each one has."""
            from .training import TrainingError, discover_training_fields

            if self._root is None:
                return
            self._header.setText(str(self._root))
            self._conditions_list.blockSignals(True)
            self._conditions_list.clear()
            self._conditions = []
            folders = [p for p in sorted(self._root.iterdir()) if p.is_dir()]
            if not folders:
                folders = [self._root]
            for folder in folders:
                try:
                    found, missing = discover_training_fields(
                        folder, image_folder=self._image_folder.currentText(),
                        **self._label_kwargs())
                except TrainingError:
                    continue
                if not found:
                    continue
                self._conditions.append((folder, found))
                item = QListWidgetItem(
                    f"{folder.name}: {len(found)} corrected field(s)"
                    + (f", {len(missing)} label(s) with no image" if missing else ""))
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked)
                self._conditions_list.addItem(item)
            self._conditions_list.blockSignals(False)
            if not self._conditions:
                self._say(f"No corrected outlines under {self._root}. Correct some in "
                          f"Edit outlines first, or switch 'labels from' to masks/.")
            self._refresh_counts()

        def _chosen_fields(self) -> list:
            """Every field from the ticked conditions."""
            out = []
            for index, (_folder, found) in enumerate(self._conditions):
                item = self._conditions_list.item(index)
                if item is not None and item.checkState() == Qt.Checked:
                    out.extend(found)
            return out

        def _refresh_counts(self) -> None:
            fields = self._chosen_fields()
            self._build.setEnabled(bool(fields))
            self._build.setText(
                f"Build training set ({len(fields)} field(s))" if fields
                else "Build training set")

        # -- build, train, export ---------------------------------------------
        def _build_clicked(self) -> None:
            from .training import TrainingError, build_training_set

            fields = self._chosen_fields()
            if not fields:
                return
            out = self._root / "training_set"
            try:
                scratch = self._base.currentText() == SCRATCH_LABEL
                built = build_training_set(
                    fields, out,
                    base_model=None if scratch else self._base.currentText(),
                    model_name=(f"{self._root.name}_"
                                + ("scratch" if scratch
                                   else self._base.currentText())),
                    channels=(self._image_folder.currentText(),),
                    holdout=float(self._holdout.value()),
                    n_epochs=int(self._epochs.value()))
            except TrainingError as exc:
                self._say(f"Could not build a training set: {exc}")
                return
            self._bundle = built.root
            self._say("", *built.summary().splitlines())
            self._export.setEnabled(True)
            self._train.setEnabled(bool(getattr(self, "_can_train", False)))
            if not built.holdout:
                # SAID AGAIN HERE, because it is the one thing that cannot be fixed
                # after training: without held-out fields there is nothing to judge on.
                self._say("Nothing was held back, so a trained model cannot be compared "
                          "with its base model. Correct more fields to fix this.")

        def _train_clicked(self) -> None:
            if self._bundle is None:
                self._say("Build a training set first.")
                return
            self._build.setEnabled(False)
            self._train.setEnabled(False)
            self._say("", "Fine-tuning has started. You can keep using this window.")
            self._thread = QThread(self)
            self._worker = _Worker(self._bundle)
            self._worker.moveToThread(self._thread)
            self._thread.started.connect(self._worker.run)
            self._worker.line.connect(self._say)
            self._worker.done.connect(self._trained)
            self._worker.failed.connect(self._crashed)
            self._thread.start()

        def _trained(self, path) -> None:
            self._say(f"Trained: {path}",
                      "Score it against the base model on the held-out fields before "
                      "using it. 'Save a model file…' records that comparison.")
            self._stop_thread()

        def _crashed(self, text: str) -> None:
            self._say("", "Training stopped:", *str(text).splitlines()[:12])
            self._stop_thread()

        def _stop_thread(self) -> None:
            if self._thread is not None:
                self._thread.quit()
                self._thread.wait()
                self._thread = None
            self._worker = None
            self._build.setEnabled(True)
            self._train.setEnabled(bool(getattr(self, "_can_train", False))
                                   and self._bundle is not None)

        def _export_clicked(self) -> None:
            import shutil

            if self._bundle is None:
                return
            chosen = QFileDialog.getExistingDirectory(self, "Copy the training set to")
            if not chosen:
                return
            target = Path(chosen) / self._bundle.name
            try:
                shutil.copytree(self._bundle, target, dirs_exist_ok=True)
            except OSError as exc:
                self._say(f"Could not copy: {exc}")
                return
            self._say(f"Copied to {target}",
                      "It holds its own images, labels, held-out fields, README and "
                      "Colab notebook, and does not need SMTracker.")

        # -- the model store --------------------------------------------------
        def _refresh_models(self) -> None:
            from . import models as ms

            self._models.clear()
            for saved in ms.list_models():
                row = QListWidgetItem(saved.summary())
                # THE NAME RIDES WITH THE ROW: the line reads "name, from base, ...: verdict",
                # so cutting it at a ':' never gave the name back (the UI designer, 26 Nov 2026).
                row.setData(Qt.UserRole, saved.name)
                self._models.addItem(row)
            for name, why in ms.broken_models():
                # NAMED RATHER THAN HIDDEN: a model that vanishes from the list looks
                # like one that was never saved.
                row = QListWidgetItem(f"{name}: unreadable: {why}")
                row.setData(Qt.UserRole, name)
                self._models.addItem(row)

        def _save_model_clicked(self) -> None:
            from qtpy.QtWidgets import QInputDialog

            from . import models as ms

            chosen, _ = QFileDialog.getOpenFileName(self, "Trained model file")
            if not chosen:
                return
            name, ok = QInputDialog.getText(self, "Save model", "Name it:")
            if not ok or not name.strip():
                return
            comment, _ok = QInputDialog.getText(
                self, "Save model", "What is it for? (kept with the model)")
            info = self._bundle_info()
            # RECOGNISED FROM THE FILE (Cellpose 4 or Omnipose), refused by name if neither
            from . import backends
            from .segment import SegmentationError

            try:
                info["engine"] = backends.checkpoint_kind(chosen)
            except SegmentationError as exc:
                self._say(f"Not saved: {exc}")
                return
            try:
                ms.save_model(name.strip(), chosen, comment=comment.strip(), **info)
            except ms.ModelStoreError as exc:
                if "already saved" not in str(exc):
                    self._say(str(exc))
                    return
                if QMessageBox.question(
                        self, "Replace it?",
                        f"A model called {name.strip()!r} already exists. Replace it?") != \
                        QMessageBox.Yes:
                    return
                ms.save_model(name.strip(), chosen, comment=comment.strip(),
                              overwrite=True, **info)
            self._refresh_models()
            self._say(f"Saved {name.strip()!r}.")

        def _bundle_info(self) -> dict:
            """What the built set says, so a saved model records what it saw."""
            import json

            from .training import PROVENANCE_FILE

            if self._bundle is None:
                return {"base_model": self._base.currentText()}
            try:
                info = json.loads(
                    (self._bundle / PROVENANCE_FILE).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return {"base_model": self._base.currentText()}
            return {
                "base_model": info.get("base_model", ""),
                "channels": tuple(info.get("channels", ())),
                "conditions": tuple(sorted(
                    {f.split("__")[0] for f in info.get("train_fields", ())})),
                "n_cells": int(info.get("n_cells", 0)),
                "n_train_fields": int(info.get("n_train_fields", 0)),
                "label_kinds": tuple(info.get("label_kinds", ())),
            }

        def _delete_model_clicked(self) -> None:
            from . import models as ms

            item = self._models.currentItem()
            if item is None:
                return
            name = item.data(Qt.UserRole)
            if QMessageBox.question(
                    self, "Delete model?",
                    f"Delete {name!r} from the store?") != QMessageBox.Yes:
                return
            try:
                ms.delete_model(name)
            except ms.ModelStoreError as exc:
                self._say(str(exc))
                return
            self._refresh_models()
            self._say(f"Deleted {name!r}.")

        def closeEvent(self, event):                       # noqa: N802 - Qt's name
            """Join the worker before Qt collects it.

            A QThread destroyed while running calls `abort()` — no exception, no
            traceback, exit 127. The same silent death a QDialog built with no
            QApplication gives, and this project has hit both.
            """
            if self._thread is not None:
                self._thread.quit()
                self._thread.wait()
                self._thread = None
            super().closeEvent(event)
            if event.isAccepted():
                _OPEN_WINDOWS.discard(self)

    return TrainerWindow


def launch(root=None, *, show=True, block=False):
    """Open the trainer. Returns the window, which is also held in `_OPEN_WINDOWS`."""
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    from ..appicon import apply as _apply_icon

    _apply_icon()
    window = _build_window_class()(root)
    # HELD. A parentless window nobody keeps is collected the moment this returns, and
    # it takes any running worker thread with it.
    _OPEN_WINDOWS.add(window)
    if show:
        from ..ui.widgets.screenfit import show_fitted   # never past the screen (30 Oct 2026)

        show_fitted(window)
    if block:
        app.exec()
    return window


def __getattr__(name):
    """`TrainerWindow` built on demand, so importing this module needs no Qt."""
    if name == "TrainerWindow":
        return _build_window_class()
    raise AttributeError(name)
