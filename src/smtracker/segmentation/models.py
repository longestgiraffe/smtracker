"""Saved segmentation models, and what each one saw.

THE SAME SHAPE AS `smtracker.presets`, and for the same reasons: per-user rather than in
the installation folder (a frozen install is often unwritable and is replaced wholesale on
upgrade), overridable by an environment variable so a test never writes to the real user
directory and a lab can point a shared folder at it.

WHAT A MODEL FILE CANNOT SAY FOR ITSELF is everything that decides whether to trust it:
which corrections it was fine-tuned on, how many cells that was, which channels it reads,
and whether it actually beat the model it came from. A `.pt` beside a name is a model
nobody can judge a year later, which is the same defect a preset without its provenance
has. So the store keeps the weights and a record, and the record is not optional.

**THE SCORE IS PART OF THE RECORD, NOT A DECORATION.** `training.score_labels` measures a
model against its base on held-out fields; `SavedModel.beat_base` reports what that
comparison said, and `None` means it was never made. A model that was never scored is
usable and is marked as such -- refusing it would be worse, since a model trained on a
cluster arrives before its score does -- but it can never be mistaken for one that was.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

__all__ = [
    "MODEL_ENV",
    "ModelStoreError",
    "SavedModel",
    "broken_models",
    "delete_model",
    "list_models",
    "load_model",
    "model_dir",
    "save_model",
]

#: Environment override, so a test never writes to the real user directory.
MODEL_ENV = "SMTRACKER_MODELS"


class ModelStoreError(RuntimeError):
    """Refusing to save, read or delete a model, with the reason."""


def model_dir() -> Path:
    """Where saved models live. Created on demand, never in the install folder."""
    override = os.environ.get(MODEL_ENV)
    if override:
        return Path(override).expanduser()
    if os.name == "nt" and os.environ.get("APPDATA"):
        return Path(os.environ["APPDATA"]) / "SMTracker" / "models"
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base).expanduser() if base else Path.home() / ".config"
    return root / "smtracker" / "models"


@dataclass(frozen=True)
class SavedModel:
    """One fine-tuned model and the record of what produced it."""

    name: str
    #: The weights, inside the store. Copied in rather than referenced: a model pointing
    #: at a path on a cluster is a model that stops existing when somebody tidies up.
    weights: Path
    #: What it was fine-tuned FROM. Kept because the comparison that judges this model
    #: is against that one, and because fine-tuning never edits its base.
    base_model: str = ""
    #: Channel ROLES in priority order: the first is what the model reads, a second is
    #: passed to cellpose rather than discarded.
    channels: tuple = ()
    comment: str = ""
    #: Which conditions and how many corrected cells it saw. A model that cannot say
    #: what it was trained on cannot be judged against a new dataset.
    conditions: tuple = ()
    n_cells: int = 0
    n_train_fields: int = 0
    #: "mesh", "mask", or both — corrections made here against labels from elsewhere.
    label_kinds: tuple = ()
    #: `training.score_labels` for this model and for its base, on the SAME held-out
    #: fields. Empty means it was never scored, which is a state and not a failure.
    score: dict = field(default_factory=dict)
    base_score: dict = field(default_factory=dict)
    #: The mask threshold the model was SCORED at, which is the one it runs at (`segment.
    #: config_for`); None runs the default. A fine-tuned model's output shifts, so one
    #: fixed threshold can report no cells for a model that works (14 Oct 2026: a
    #: lab model found 61 % of held-out cells at -2 and 15 % at the default 0).
    mask_threshold: float | None = None
    #: The engine the weights run on, ``"cellpose"`` or ``"omnipose"``, READ FROM THE FILE when it was saved
    #: (`backends.checkpoint_kind`); ``""`` in a record written before 26 Nov 2026 (then the base model names it).
    engine: str = ""

    @property
    def scored(self) -> bool:
        return bool(self.score and self.base_score)

    @property
    def beat_base(self):
        """True/False if it was scored, ``None`` if it never was.

        NONE IS NOT FALSE. A model trained on a cluster arrives before its score does,
        and treating "not yet measured" as "did not help" would bury a good model.
        """
        if not self.scored:
            return None
        mine = float(self.score.get("mean_iou", 0.0))
        theirs = float(self.base_score.get("mean_iou", 0.0))
        if mine != theirs:
            return mine > theirs
        # A TIE ON IoU IS BROKEN BY THE CELL COUNT, because finding the right NUMBER of
        # cells is what the cell filter actually depends on.
        return abs(int(self.score.get("count_error", 0))) < abs(
            int(self.base_score.get("count_error", 0)))

    def summary(self) -> str:
        bits = [f"{self.name}"]
        if self.base_model:
            bits.append(f"from {self.base_model}")
        if self.n_cells:
            bits.append(f"{self.n_cells} corrected cell(s) "
                        f"over {self.n_train_fields} field(s)")
        if self.channels:
            bits.append("reads " + "+".join(self.channels))
        line = ", ".join(bits)
        if self.beat_base is None:
            line += ": not scored against its base model"
        else:
            verdict = "beats" if self.beat_base else "does not beat"
            line += (f": {verdict} {self.base_model} on held-out fields "
                     f"(IoU {self.score.get('mean_iou', 0):.3f} against "
                     f"{self.base_score.get('mean_iou', 0):.3f})")
        return line + (f": {self.comment}" if self.comment else "")


def _path_for(name: str) -> Path:
    safe = "".join(c for c in name if c.isalnum() or c in " ._-").strip()
    if not safe:
        raise ModelStoreError(
            f"{name!r} has no usable characters for a file name. Give the model a "
            f"name you can recognise in a list.")
    return model_dir() / safe


def save_model(name: str, weights, *, base_model: str = "", channels=(),
               comment: str = "", conditions=(), n_cells: int = 0,
               n_train_fields: int = 0, label_kinds=(), score=None, base_score=None,
               mask_threshold: float | None = None, engine: str = "",
               overwrite: bool = False) -> SavedModel:
    """Copy a trained model into the store with its record.

    THE WEIGHTS ARE COPIED, not referenced. A store entry pointing at a path inside
    somebody's scratch folder or a cluster share is an entry that stops working without
    anything changing here.

    OVERWRITING IS EXPLICIT, as it is for presets: two different models under one name
    is an ambiguity nothing downstream could untangle, and here it is worse than for a
    preset because the older model may be the one a published figure used.
    """
    weights = Path(weights)
    if not weights.is_file():
        raise ModelStoreError(f"No model file at {weights}.")
    folder = _path_for(name)
    if folder.exists() and not overwrite:
        raise ModelStoreError(
            f"A model called {name!r} is already saved. Replace it, or use another name.")
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / weights.name
    if weights.resolve() != target.resolve():
        shutil.copy2(weights, target)

    record = SavedModel(
        name=name, weights=target, base_model=base_model,
        channels=tuple(channels), comment=comment, conditions=tuple(conditions),
        n_cells=int(n_cells), n_train_fields=int(n_train_fields),
        label_kinds=tuple(label_kinds), score=dict(score or {}),
        base_score=dict(base_score or {}),
        mask_threshold=None if mask_threshold is None else float(mask_threshold),
        engine=str(engine),
    )
    payload = asdict(record)
    payload["weights"] = target.name          # relative, so the store can be moved
    (folder / "model.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return record


def load_model(name: str) -> SavedModel:
    """One saved model, or raise saying which part is missing."""
    folder = _path_for(name)
    record = folder / "model.json"
    if not record.is_file():
        raise ModelStoreError(f"No model called {name!r} in {model_dir()}.")
    try:
        payload = json.loads(record.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ModelStoreError(f"{record}: not readable JSON ({exc})") from exc
    weights = folder / str(payload.get("weights", ""))
    if not weights.is_file():
        # THE RECORD WITHOUT THE WEIGHTS IS NOT A MODEL, and saying so beats handing
        # back an entry whose `weights` path does not exist.
        raise ModelStoreError(
            f"{name!r} has a record but no weights file at {weights.name}.")
    return SavedModel(
        name=str(payload.get("name", name)), weights=weights,
        base_model=str(payload.get("base_model", "")),
        channels=tuple(payload.get("channels", ())),
        comment=str(payload.get("comment", "")),
        conditions=tuple(payload.get("conditions", ())),
        n_cells=int(payload.get("n_cells", 0)),
        n_train_fields=int(payload.get("n_train_fields", 0)),
        label_kinds=tuple(payload.get("label_kinds", ())),
        score=dict(payload.get("score", {})),
        base_score=dict(payload.get("base_score", {})),
        # absent in a record written before 14 Oct 2026: the default threshold, as then
        mask_threshold=(None if payload.get("mask_threshold") is None
                        else float(payload["mask_threshold"])),
        engine=str(payload.get("engine", "")),
    )


def list_models() -> tuple:
    """Every readable saved model, by name. Unreadable ones are `broken_models`."""
    folder = model_dir()
    if not folder.is_dir():
        return ()
    out = []
    for child in sorted(folder.iterdir()):
        if not child.is_dir():
            continue
        try:
            out.append(load_model(child.name))
        except ModelStoreError:
            continue
    return tuple(out)


def broken_models() -> tuple:
    """Saved models that cannot be read, by name and reason.

    NAMED RATHER THAN HIDDEN. A model that vanishes from the list looks like one that
    was never saved, and the commonest cause here is a weights file somebody moved.
    """
    folder = model_dir()
    if not folder.is_dir():
        return ()
    out = []
    for child in sorted(folder.iterdir()):
        if not child.is_dir():
            continue
        try:
            load_model(child.name)
        except ModelStoreError as exc:
            out.append((child.name, str(exc)))
    return tuple(out)


def delete_model(name: str) -> None:
    """Remove a saved model and its record."""
    folder = _path_for(name)
    if not folder.is_dir():
        raise ModelStoreError(f"No model called {name!r} in {model_dir()}.")
    shutil.rmtree(folder)
