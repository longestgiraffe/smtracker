"""Turn corrections into a training set, and a training set into a model.

THE LABELS ALREADY EXIST. Every time somebody opens the mesh editor and fixes an outline
they produce ground truth, and `masks.labels_from_outlines` already turns outlines into a
labelled image -- its own docstring says why: "a corrected mesh is also the best training
label anyone will ever have for this data: a human looked at it." So the loop closes
without a new labelling step:

    segment -> correct by hand -> train on the corrections -> segment again

ONE ARTEFACT, TWO WAYS TO RUN IT. `build_training_set` writes a self-contained folder:
the images, the labels, a `train.py` and the provenance. Training locally runs that same
bundle in-process; exporting it to a cluster runs the same script there. The bundle IS
the interface, so there is one training path and two ways to invoke it -- the same shape
as `playback` and napari, where both routes exist over one set of numbers.

FINE-TUNING PRODUCES A NEW MODEL, never edits the old one. That is deliberate: a model
you cannot go back from is a model whose earlier results cannot be reproduced, and the
comparison below needs the base model to still exist.

**A TRAINED MODEL WITHOUT A HELD-OUT COMPARISON IS UNFALSIFIABLE.** `split_fields` keeps
some corrected fields out of training and `score_labels` measures the new model against
the OLD one on exactly those, in IoU and in cell count. Without it "Train" is a button
that always produces something and never says whether it helped -- the same defect this
project already records for k-means, which always returns clusters.

WHAT IS NOT ESTABLISHED HERE: how many corrected cells make an adequate training set.
Nothing in this project measures that, so nothing here asserts a minimum beyond refusing
the degenerate case. The count travels on the provenance so a model can always be read
against how much it saw.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "TrainingError",
    "score_against_base",
    "train_bundle",
    "training_runtime_available",
    "TrainingField",
    "TrainingSet",
    "build_training_set",
    "discover_training_fields",
    "labels_for_field",
    "score_labels",
    "split_fields",
]


def _apply_omnipose_runtime_patch() -> None:
    """`omnipose.core.random_crop_warp` has two defects that stop training on sparse
    fields the moment the augmenter's random crop misses every cell:

    (1) at recursion depth > 100 it calls
        ``skimage.io.imsave('/home/kcutler/DataDrive/debug/imgNN.png', img[0])`` —
        an absolute path that only exists on the developer's machine, and imageio
        raises ``OSError: cannot write mode F as PNG`` even when the directory does;
    (2) then it raises ``ValueError`` which propagates out of ``train()`` and kills
        the whole run.

    Both are debug/guard code left in a released package. Bacterial fields are sparse
    by nature: 3 cells over ~5 % of a 128×128 frame is typical for this project, and
    a random rotation crop lands on background often enough that ~1 field in 700 per
    epoch trips the guard. Left unhandled, an overnight train dies within an hour.

    This wrapper catches those two ValueErrors, plus the OSError from the debug save,
    and returns a zero-filled (image, label, scale) triple of the requested shape.
    A zero-filled label is background, so the loss ignores it — the cost is at most
    a fraction of one batch element per problem field per epoch, invisible to the
    trained model.

    Idempotent: safe to call more than once.
    """
    try:
        import omnipose.core as core
    except ImportError:
        return
    if getattr(core.random_crop_warp, "_smtracker_patched", False):
        return
    import numpy as np

    orig = core.random_crop_warp

    def safe(img, Y, links, nt, tyx, nchan, *args, **kwargs):
        try:
            return orig(img, Y, links, nt, tyx, nchan, *args, **kwargs)
        except (ValueError, OSError, FileNotFoundError) as exc:
            msg = str(exc)
            if ("Sparse or over-dense" in msg
                    or "Recusion depth" in msg
                    or "kcutler" in msg
                    or "cannot write mode F" in msg):
                imgi = np.zeros((nchan,) + tuple(tyx), np.float32)
                lbl = np.zeros((nt,) + tuple(tyx), np.float32)
                scale = np.ones(len(tyx), np.float32)
                return imgi, lbl, scale
            raise

    safe._smtracker_patched = True
    core.random_crop_warp = safe
    try:
        import cellpose_omni.transforms as tr
        if hasattr(tr, "random_crop_warp"):
            tr.random_crop_warp = safe
    except ImportError:
        pass

#: Written beside a bundle so a model can say what it saw.
PROVENANCE_FILE = "provenance.json"

#: `base_model=None` means TRAIN A NEW MODEL rather than fine-tune an existing one:
#: cellpose gets `pretrained_model=False` and starts from nothing.
#:
#: THE TWO ARE NOT INTERCHANGEABLE AND THE DATA COST IS THE DIFFERENCE. Fine-tuning
#: adjusts a model that already knows what a bacterium looks like, which is why tens of
#: corrected fields move it; a new model has to learn that from these images alone.
#: Cellpose's own guidance is hundreds of labelled images for training from scratch --
#: that is THEIR figure, not one measured here, and nothing in this project establishes
#: a threshold for this data. What the tool can do is say which of the two is happening
#: and report the cell count beside it, which is why `base_model` is recorded as null
#: rather than being quietly replaced by a default.
FROM_SCRATCH_NOTE = (
    "Training a new model from scratch, not fine-tuning. It needs many more labelled "
    "images than a fine-tune (cellpose suggests hundreds against tens; not measured here). Score it against "
    "a pretrained model before you trust it.")

#: Where a bundle keeps its two halves. Flat folders of matching stems, because the
#: training script should be readable by somebody who has never seen this codebase.
IMAGE_DIR = "images"
LABEL_DIR = "labels"

#: The held-out half, written the same way. IT MUST TRAVEL WITH THE BUNDLE: the score is
#: computed where the training happened, which may be a cluster that cannot see the
#: original project at all. Naming the source paths alone was not enough -- the first
#: version did that and the scorer skipped every field.
HOLDOUT_IMAGE_DIR = "holdout/images"
HOLDOUT_LABEL_DIR = "holdout/labels"


class TrainingError(RuntimeError):
    """Refusing to build or score a training set, with the reason."""


@dataclass(frozen=True)
class TrainingField:
    """One field that can be trained on: an image, and labels a human corrected."""

    condition: str
    field: str
    image: Path
    #: The `.mat` holding corrected outlines, or a label image. Never both.
    labels: Path
    #: "mesh" or "mask" — which reader produced the labels.
    kind: str

    @property
    def stem(self) -> str:
        """Unique across conditions, because two conditions both have `series001`."""
        return f"{self.condition}__{self.field}"


@dataclass
class TrainingSet:
    """A built bundle, and what went into it."""

    root: Path
    train: list = field(default_factory=list)
    holdout: list = field(default_factory=list)
    base_model: str = ""
    channels: tuple = ()
    n_cells: int = 0
    notes: list = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"training set at {self.root}",
            f"  base model   {self.base_model}",
            f"  channels     {'+'.join(self.channels) if self.channels else '(none)'}",
            f"  train        {len(self.train)} field(s)",
            f"  held out     {len(self.holdout)} field(s)",
            f"  cells        {self.n_cells} corrected",
        ]
        # THE COUNT, NOT A VERDICT ON IT. Nothing here measures how many corrected cells
        # are enough, so nothing here claims a number is sufficient.
        lines.append("  Note: no minimum training size is established for this data; "
                     "read the cell count above beside any model trained from it.")
        lines.extend(f"  {note}" for note in self.notes)
        return "\n".join(lines)


def labels_for_field(field_path, *, shape=None):
    """A labelled integer image for one field, from a mesh `.mat` or a mask image.

    BOTH, because labels arrive two ways. Corrections made in this tool are Oufti meshes;
    labels from anywhere else are usually an integer-labelled image. They land as the
    same array and the bundle records which produced it, since a model trained on
    somebody else's masks and one trained on your own corrections are not the same claim.

    Meshes go through `masks.labels_from_outlines`, which uses `cellmesh.points_in_polygon`
    -- MATLAB `inpolygon`'s `in | on`, the same rule the CELL FILTER uses. A mask drawn by
    any other rasteriser would disagree with the analysis about which pixels are cell.
    """
    import numpy as np

    field_path = Path(field_path)
    if field_path.suffix.lower() == ".mat":
        if shape is None:
            raise TrainingError(
                f"{field_path.name}: an image shape is needed to draw the outlines as labels.")
        from ..cellmesh import read_cell_meshes
        from .masks import labels_from_outlines

        # AN UNSEGMENTED FIELD IS NOT TRAINING DATA, and it is an ordinary state: the
        # reader rightly REFUSES an empty `cellList` by default, because the analysis
        # only uses tracks inside a cell and a silent skip there would drop a whole
        # movie. Here the opposite is true -- a field with no outlines simply has no
        # labels to learn from -- so it is asked for knowingly and reported by the
        # caller. Measured: one lab movie is exactly this, and it aborted the
        # whole build.
        cells = read_cell_meshes(field_path, on_empty="empty")
        # TWO CELL SHAPES, ONE CONVERTER. `cellmesh.CellMesh` calls its contour
        # `polygon`; the editor's own document calls it `outline`, which is what
        # `labels_from_outlines` looks for by name before falling through to the object
        # itself. Passing a `CellMesh` straight in therefore rasterised nothing --
        # silently, since the converter skips anything that is not an (N, 2) array.
        outlines = []
        for cell in cells or ():
            poly = getattr(cell, "polygon", None)
            if poly is None:
                poly = getattr(cell, "outline", None)
            if poly is not None:
                outlines.append(poly)
        if not outlines:
            raise TrainingError(f"{field_path.name}: it has no outlines.")
        return labels_from_outlines(outlines, shape)

    import tifffile

    labels = np.asarray(tifffile.imread(field_path))
    if labels.ndim != 2:
        raise TrainingError(
            f"{field_path.name}: a label image must be 2-D, but its shape is {labels.shape}.")
    if not np.issubdtype(labels.dtype, np.integer):
        # NOT COERCED. A float image here is a probability map or a raw frame, not
        # labels, and rounding it would invent cells.
        raise TrainingError(
            f"{field_path.name}: labels must be integers, but this image holds {labels.dtype}.")
    return labels.astype("int32")


def discover_training_fields(condition_root, *, condition=None, image_folder="phase",
                             mesh_folder="cell_meshes", mask_folder=None):
    """Fields under one condition that have BOTH an image and corrected labels.

    A field with an image and no labels is not training data, and one with labels and no
    image cannot be trained on: both are skipped, and the caller is told how many, because
    "47 fields" and "47 fields, 30 of them unusable" are different training sets.

    The brightfield images (``image_folder="phase"``) lie in ``phase/`` OR beside the
    movies (ruled 18 Oct 2026; `channels.loose_snapshots`), and every file is keyed by the
    MOVIE's name (`channels.snapshot_field`: ``series001.nd2.tif`` is ``series001``), which
    is the name its outlines carry. This raised on a loose condition and lost a
    ``.nd2.tif`` (handoff 18 Oct 2026, place 3).
    """
    from .channels import loose_snapshots, snapshot_field

    condition_root = Path(condition_root)
    name = condition or condition_root.name
    images = condition_root / image_folder
    if images.is_dir():
        files = sorted(p for p in images.iterdir()
                       if p.is_file() and p.suffix.lower() in (".tif", ".tiff"))
    elif image_folder.lower() == "phase" and (files := loose_snapshots(condition_root)):
        pass
    elif image_folder.lower() == "phase":
        raise TrainingError(f"No brightfield images under {condition_root}: it has neither a "
                            f"phase/ folder nor snapshots beside the movies.")
    else:
        raise TrainingError(f"No {image_folder}/ folder under {condition_root}.")

    by_stem = {}
    for path in files:
        field = snapshot_field(path)
        if field not in by_stem or path.stem == field:   # the exact name wins a tie
            by_stem[field] = path

    found, skipped = [], []
    from ..cellmesh import mesh_folder as _mesh_folder

    label_dir = (condition_root / mask_folder if mask_folder
                 else _mesh_folder(condition_root, mesh_folder))
    kind = "mask" if mask_folder else "mesh"
    if not label_dir.is_dir():
        raise TrainingError(f"No {label_dir.name}/ folder under {condition_root}.")
    for path in sorted(label_dir.iterdir()):
        if not path.is_file():
            continue
        if kind == "mesh" and path.suffix.lower() != ".mat":
            continue
        if kind == "mask" and path.suffix.lower() not in (".tif", ".tiff"):
            continue
        field = snapshot_field(path)
        image = by_stem.get(field)
        if image is None:
            skipped.append(field)
            continue
        found.append(TrainingField(condition=name, field=field, image=image,
                                   labels=path, kind=kind))
    return found, skipped


def split_fields(fields, *, holdout=0.2, seed=0):
    """``(train, holdout)`` — DETERMINISTIC, so a score can be reproduced.

    Sorted by stem and then sampled with a fixed seed, never by whatever order the
    filesystem returned: a held-out score computed on a different split is a different
    measurement wearing the same name.

    A set too small to hold anything out returns everything as training and NO holdout,
    which the caller must report rather than quietly scoring on training data.
    """
    import random

    ordered = sorted(fields, key=lambda f: f.stem)
    if len(ordered) < 2:
        return list(ordered), []
    n_out = max(1, int(round(len(ordered) * float(holdout))))
    n_out = min(n_out, len(ordered) - 1)          # never hold out everything
    picked = random.Random(seed).sample(range(len(ordered)), n_out)
    keep = set(picked)
    return ([f for i, f in enumerate(ordered) if i not in keep],
            [f for i, f in enumerate(ordered) if i in keep])


def score_labels(truth, predicted, *, iou_threshold: float = 0.5) -> dict:
    """Compare two labelled images: matched cells, IoU, and the cell-count error.

    ONE-TO-ONE MATCHING at ``iou_threshold``, greedily by descending IoU, so a model that
    splits every cell in two cannot score well by covering the truth twice -- the same
    rule `moviesim.match_to_truth` uses for detections, and for the same reason.

    Returns counts and means rather than a verdict. Whether a model is better is a
    comparison between two of these on the SAME held-out fields, which is the caller's
    job and the only thing that makes a trained model falsifiable.
    """
    import numpy as np

    truth = np.asarray(truth)
    predicted = np.asarray(predicted)
    if truth.shape != predicted.shape:
        raise TrainingError(
            f"The shapes differ: truth {truth.shape}, predicted {predicted.shape}.")

    t_ids = [i for i in np.unique(truth) if i != 0]
    p_ids = [i for i in np.unique(predicted) if i != 0]
    pairs = []
    for t in t_ids:
        t_mask = truth == t
        t_area = int(t_mask.sum())
        for p in np.unique(predicted[t_mask]):
            if p == 0:
                continue
            p_mask = predicted == p
            inter = int((t_mask & p_mask).sum())
            union = t_area + int(p_mask.sum()) - inter
            if union:
                pairs.append((inter / union, int(t), int(p)))
    pairs.sort(reverse=True)

    used_t, used_p, matched = set(), set(), []
    for iou, t, p in pairs:
        if iou < iou_threshold or t in used_t or p in used_p:
            continue
        used_t.add(t)
        used_p.add(p)
        matched.append(iou)
    return {
        "n_truth": len(t_ids),
        "n_predicted": len(p_ids),
        "n_matched": len(matched),
        "mean_iou": float(np.mean(matched)) if matched else 0.0,
        "count_error": len(p_ids) - len(t_ids),
        "iou_threshold": float(iou_threshold),
    }


#: The script written into every bundle. It is deliberately plain: somebody running this
#: on a cluster has never seen this codebase, and a training script that imported
#: `smtracker` would need the whole package installed beside a GPU.
_TRAIN_SCRIPT = '''"""Fine-tune an omnipose model on corrected cell outlines.

Written by smtracker. Self-contained: it needs omnipose 1.1.4 (with its OWN cellpose_omni,
never a separate cellpose-omni package over it) and ncolor 1.5.x, and nothing from smtracker.

    python train.py                # uses provenance.json for the base model and name

The result is a NEW model file under models/, never an edit of the base model.

THE RECIPE OF THE ENVIRONMENT THAT TRAINED THIS LAB'S MODELS (omnipose_clean):
checkpoints are loaded with nclasses=3, dim=2, nchan=2; torch's JIT fuser is off; the
boundary loss's BCE gets its inputs clamped to [0, 1] (omnipose 1.1.4 otherwise dies
with "input_val >= zero && input_val <= one"), applied here at run time rather than by
editing omnipose's own file; and train() is given empty links, min_train_masks=1,
rescale=False and the learning rate in provenance.json.
"""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
INFO = json.loads((HERE / "provenance.json").read_text(encoding="utf-8"))
#: The base model's own weights, copied in when the bundle was built, so a fine-tune from
#: a lab model works on a machine that never had it.
BASE_FILE = HERE / "base_model.pt"


def prepare():
    """The JIT fuser off, the aicsimageio/tifffile alias, the clamped BCE. Idempotent."""
    import torch
    try:
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
    except Exception:
        pass
    try:
        import tifffile
        if not hasattr(tifffile.TIFF, "RESUNIT") and hasattr(tifffile, "RESUNIT"):
            tifffile.TIFF.RESUNIT = tifffile.RESUNIT
    except ImportError:
        pass
    try:
        import omnipose.loss as loss
    except ImportError:
        return
    bce = getattr(loss, "BatchMeanBSE", None)
    if bce is not None and not getattr(bce.forward, "_clamped", False):
        original = bce.forward

        def forward(self, pred, target):
            return original(self, pred.clamp(0, 1), target.clamp(0, 1))

        forward._clamped = True
        bce.forward = forward


def load_model(which, gpu=True):
    """A checkpoint (a path, or a built-in name such as bact_phase_omni) as 1.1.4 wants it."""
    prepare()
    from cellpose_omni import models
    path = pathlib.Path(str(which))
    if not path.is_file():
        cached = pathlib.Path.home() / ".cellpose" / "models" / f"{which}torch_0"
        if cached.is_file():
            path = cached
        else:
            return models.CellposeModel(gpu=gpu, model_type=str(which))
    return models.CellposeModel(gpu=gpu, pretrained_model=str(path), nclasses=3, dim=2,
                                nchan=2)


def _patch_omnipose():
    """omnipose.core.random_crop_warp crashes on sparse crops; see smtracker docs."""
    try:
        import omnipose.core as core
    except ImportError:
        return
    if getattr(core.random_crop_warp, "_smtracker_patched", False):
        return
    import numpy as np
    orig = core.random_crop_warp

    def safe(img, Y, links, nt, tyx, nchan, *args, **kwargs):
        try:
            return orig(img, Y, links, nt, tyx, nchan, *args, **kwargs)
        except (ValueError, OSError, FileNotFoundError) as exc:
            m = str(exc)
            if ("Sparse or over-dense" in m or "Recusion depth" in m
                    or "kcutler" in m or "cannot write mode F" in m):
                return (np.zeros((nchan,) + tuple(tyx), np.float32),
                        np.zeros((nt,) + tuple(tyx), np.float32),
                        np.ones(len(tyx), np.float32))
            raise

    safe._smtracker_patched = True
    core.random_crop_warp = safe
    try:
        import cellpose_omni.transforms as tr
        if hasattr(tr, "random_crop_warp"):
            tr.random_crop_warp = safe
    except ImportError:
        pass


def main():
    import numpy as np
    import tifffile

    prepare()
    from cellpose_omni import models

    _patch_omnipose()

    images, labels = [], []
    for image_path in sorted((HERE / "images").glob("*.tif")):
        label_path = HERE / "labels" / image_path.name
        if not label_path.is_file():
            continue
        images.append(np.asarray(tifffile.imread(image_path)))
        labels.append(np.asarray(tifffile.imread(label_path)))
    if not images:
        sys.exit("no image/label pairs found beside this script")
    print(f"{len(images)} field(s), base model {INFO.get('base_model')!r}")

    out = HERE / "models"
    out.mkdir(exist_ok=True)
    scratch = INFO.get("from_scratch") or not INFO.get("base_model")
    if scratch:
        print("training a NEW model from scratch (no pretrained weights)")
        model = models.CellposeModel(gpu=True, pretrained_model=False, omni=True,
                                     nclasses=3, dim=2, nchan=2)
    else:
        model = load_model(BASE_FILE if BASE_FILE.is_file() else INFO["base_model"])
    path = model.train(images, labels, train_links=[None] * len(images),
                       channels=INFO.get("cellpose_channels", [0, 0]),
                       save_path=str(out), netstr=INFO["model_name"],
                       n_epochs=int(INFO.get("n_epochs", 100)),
                       learning_rate=float(INFO.get("learning_rate", 0.05)),
                       min_train_masks=1, rescale=False,
                       # save_each: without it omnipose OVERWRITES one file at every save_every,
                       # and only the last ten epochs survive as numbered checkpoints
                       **({"save_each": True} if INFO.get("save_every") else {}),
                       **{k: int(INFO[k]) for k in ("batch_size", "save_every") if INFO.get(k)})
    print("wrote", path)
    print("Now score it against", INFO.get("compare_with") or INFO.get("base_model"),
          "on the held-out fields in holdout/ -- a model without that comparison cannot "
          "be judged.")


if __name__ == "__main__":
    main()
'''


_TRAIN_SCRIPT_CELLPOSE = '''"""Fine-tune Cellpose-SAM (Cellpose 4) on corrected cell outlines.

Written by smtracker. Self-contained: it needs cellpose >= 4 and a CUDA GPU, and nothing from smtracker.

    python train.py                # uses provenance.json for the base model, the name and the settings

The result is a NEW model file under models/, never an edit of the base model.

Cellpose-SAM is a 300 million parameter network: training it needs about 6 GB of GPU memory at a batch of 1
(the network is trained in 32-bit floats). A card with 4 GB runs out and says so; use a larger one, or Colab.
"""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
INFO = json.loads((HERE / "provenance.json").read_text(encoding="utf-8"))
#: The base model's own weights, copied in when the bundle was built if they were on the machine that built it;
#: otherwise cellpose downloads "cpsam" itself, here.
BASE_FILE = HERE / "base_model.pt"


def main():
    import numpy as np
    import tifffile
    import torch
    from cellpose import models, train

    if not torch.cuda.is_available():
        sys.exit("no CUDA GPU is visible: Cellpose-SAM cannot be trained on a CPU in any useful time")
    images, labels = [], []
    for image_path in sorted((HERE / "images").glob("*.tif")):
        label_path = HERE / "labels" / image_path.name
        if not label_path.is_file():
            continue
        images.append(np.asarray(tifffile.imread(image_path)))
        labels.append(np.asarray(tifffile.imread(label_path)))
    if not images:
        sys.exit("no image/label pairs found beside this script")
    base = str(BASE_FILE) if BASE_FILE.is_file() else str(INFO.get("base_model") or "cpsam")
    print(f"{len(images)} field(s), base model {base!r}")
    model = models.CellposeModel(gpu=True, pretrained_model=base)
    kwargs = dict(train_data=images, train_labels=labels, save_path=str(HERE),
                  model_name=INFO["model_name"], n_epochs=int(INFO.get("n_epochs", 100)),
                  learning_rate=float(INFO.get("learning_rate", 1e-5)),
                  batch_size=int(INFO.get("batch_size", 1)), min_train_masks=1)
    if INFO.get("save_every"):
        kwargs.update(save_every=int(INFO["save_every"]), save_each=True)
    path, _train_losses, _test_losses = train.train_seg(model.net, **kwargs)
    print("wrote", path)
    print("Now score it against", INFO.get("compare_with") or INFO.get("base_model"),
          "on the held-out fields in holdout/ -- a model without that comparison cannot be judged.")


if __name__ == "__main__":
    main()
'''


_README_CELLPOSE = """# Training set for Cellpose-SAM (Cellpose 4)

Built by SMTracker from cell outlines a human corrected. Everything needed is in this
folder; nothing here imports SMTracker.

    images/          the fields to train on
    labels/          their corrected outlines, as integer label images
    holdout/         fields kept OUT of training, with their labels
    provenance.json  what this was built from, and the base model
    train.py         fine-tune, on a machine with a CUDA GPU and cellpose 4

## On a machine with a GPU

    pip install cellpose           # version 4, with a CUDA build of torch
    python train.py

Cellpose-SAM needs about 6 GB of GPU memory to train at a batch of 1. A 4 GB card runs out.

## Then judge it

Training always produces a model. Whether it is BETTER is a separate question, and
`holdout/` exists to answer it: those fields were never trained on. Score the new model
and the base model on them and compare. SMTracker records the comparison beside the model
when you save it.
"""

#: Written beside the script. A bundle that arrives on a cluster or in a browser has no
#: window to explain it, and "copy it somewhere with a GPU" is not instructions.
_README = """# Training set for a cellpose/omnipose model

Built by SMTracker from cell outlines a human corrected. Everything needed is in this
folder; nothing here imports SMTracker.

    images/          the fields to train on
    labels/          their corrected outlines, as integer label images
    holdout/         fields kept OUT of training, with their labels
    provenance.json  what this was built from, and the base model
    train.py         fine-tune, on any machine with a GPU and cellpose
    train_colab.ipynb  the same thing in a browser, if you have no GPU here

## On a machine with a GPU

    pip install cellpose-omni        # or cellpose
    python train.py

## In Google Colab

Open `train_colab.ipynb`, set the runtime to a GPU (Runtime -> Change runtime type ->
T4), upload this whole folder, and run the cells.

## Then judge it

Training always produces a model. Whether it is BETTER is a separate question, and
`holdout/` exists to answer it: those fields were never trained on. Score the new model
and the base model on them and compare. A model without that comparison cannot be judged,
and SMTracker records the comparison beside the model when you save it.
"""


_COLAB_NOTEBOOK = '{\n "cells": [\n  {\n   "cell_type": "markdown",\n   "metadata": {},\n   "source": [\n    "# Fine-tune a cell-segmentation model\\\\n",\n    "\\\\n",\n    "Built by SMTracker from outlines a human corrected. Set the runtime to a GPU ",\n    "first: **Runtime -> Change runtime type -> T4 GPU**.\\\\n"\n   ]\n  },\n  {\n   "cell_type": "code",\n   "metadata": {},\n   "source": [\n    "# 1. The runtime. Training on CPU here is impractical rather than slow.\\\\n",\n    "import torch\\\\n",\n    "print(\'GPU:\', torch.cuda.is_available())\\\\n",\n    "assert torch.cuda.is_available(), \'set Runtime -> Change runtime type -> T4 GPU\'\\\\n"\n   ],\n   "execution_count": null,\n   "outputs": []\n  },\n  {\n   "cell_type": "code",\n   "metadata": {},\n   "source": [\n    "!pip -q install omnipose==1.1.4 ncolor==1.5.3\\\\n"\n   ],\n   "execution_count": null,\n   "outputs": []\n  },\n  {\n   "cell_type": "markdown",\n   "metadata": {},\n   "source": [\n    "## 2. Upload the bundle\\\\n",\n    "\\\\n",\n    "Zip this whole folder on your machine and upload it, or mount Drive and point ",\n    "`BUNDLE` at it.\\\\n"\n   ]\n  },\n  {\n   "cell_type": "code",\n   "metadata": {},\n   "source": [\n    "import pathlib, zipfile\\\\n",\n    "from google.colab import files\\\\n",\n    "up = files.upload()                      # choose your zipped bundle\\\\n",\n    "name = next(iter(up))\\\\n",\n    "with zipfile.ZipFile(name) as z:\\\\n",\n    "    z.extractall(\'bundle\')\\\\n",\n    "BUNDLE = next(p.parent for p in pathlib.Path(\'bundle\').rglob(\'provenance.json\'))\\\\n",\n    "print(\'bundle at\', BUNDLE)\\\\n"\n   ],\n   "execution_count": null,\n   "outputs": []\n  },\n  {\n   "cell_type": "code",\n   "metadata": {},\n   "source": [\n    "# 3. Train. This is the SAME train.py that runs on a cluster.\\\\n",\n    "%cd {BUNDLE}\\\\n",\n    "!python train.py\\\\n"\n   ],\n   "execution_count": null,\n   "outputs": []\n  },\n  {\n   "cell_type": "markdown",\n   "metadata": {},\n   "source": [\n    "## 4. Judge it before you use it\\\\n",\n    "\\\\n",\n    "`holdout/` holds fields the training never saw. Score the NEW model and the ",\n    "BASE model on them: training always produces a model, and whether it helped is ",\n    "a different question.\\\\n"\n   ]\n  },\n  {\n   "cell_type": "code",\n   "metadata": {},\n   "source": [\n    "import json, sys, numpy as np, tifffile, pathlib\\\\n",\n    "from cellpose_omni import models\\\\n",\n    "info = json.loads((BUNDLE / \'provenance.json\').read_text())\\\\n",\n    "new = sorted((BUNDLE / \'models\').rglob(\'*\'))[-1]\\\\n",\n    "\\\\n",\n    "def iou(truth, pred):\\\\n",\n    "    best = []\\\\n",\n    "    for t in [i for i in np.unique(truth) if i]:\\\\n",\n    "        tm = truth == t\\\\n",\n    "        scores = [ (tm & (pred == p)).sum() / ((tm | (pred == p)).sum() or 1)\\\\n",\n    "                   for p in np.unique(pred[tm]) if p ]\\\\n",\n    "        best.append(max(scores) if scores else 0.0)\\\\n",\n    "    return float(np.mean(best)) if best else 0.0\\\\n",\n    "\\\\n",\n    "for label, which in ((\'new\', str(new)), (\'base\', info[\'base_model\'])):\\\\n",\n    "    sys.path.insert(0, str(BUNDLE)); import train; m = train.load_model(which)\\\\n",\n    "    got = []\\\\n",\n    "    for img_path in sorted((BUNDLE / \'holdout\' / \'images\').glob(\'*.tif\')):\\\\n",\n    "        truth = tifffile.imread(BUNDLE / \'holdout\' / \'labels\' / img_path.name)\\\\n",\n    "        masks, _, _ = m.eval(tifffile.imread(img_path), channels=[0, 0])\\\\n",\n    "        got.append(iou(np.asarray(truth), np.asarray(masks)))\\\\n",\n    "    print(label, \'mean IoU\', round(float(np.mean(got)), 3), \'over\', len(got), \'field(s)\')\\\\n"\n   ],\n   "execution_count": null,\n   "outputs": []\n  },\n  {\n   "cell_type": "code",\n   "metadata": {},\n   "source": [\n    "# 5. Download the model, then Load model... in SMTracker\'s segmentation window.\\\\n",\n    "from google.colab import files\\\\n",\n    "files.download(str(new))\\\\n"\n   ],\n   "execution_count": null,\n   "outputs": []\n  }\n ],\n "metadata": {\n  "accelerator": "GPU",\n  "colab": {\n   "provenance": []\n  },\n  "kernelspec": {\n   "display_name": "Python 3",\n   "name": "python3"\n  },\n  "language_info": {\n   "name": "python"\n  }\n },\n "nbformat": 4,\n "nbformat_minor": 0\n}'


def build_training_set(fields, out_dir, *, base_model, model_name: str,
                       channels=(), holdout=0.2, seed=0, n_epochs: int = 100,
                       learning_rate: float | None = None, batch_size: int | None = None,
                       save_every: int | None = None,
                       compare_with: str | None = None):
    """Write a self-contained bundle: images, labels, a script and the provenance.

    THE SAME BUNDLE runs locally and on a cluster. Nothing here trains -- building the
    set and running it are separate so the expensive half can happen anywhere, and so a
    set can be inspected before a GPU is spent on it.

    ``base_model`` is what to fine-tune FROM, or ``None`` to train a NEW model from
    scratch. The two are different jobs with different data costs; see
    `FROM_SCRATCH_NOTE`.

    ``compare_with`` is the model the result is judged against on the held-out fields.
    It defaults to the base model when fine-tuning, and must be named explicitly when
    training from scratch -- a new model with no yardstick is exactly the unfalsifiable
    thing this module exists to prevent, and "it segmented something" is not a result.
    """
    import tifffile

    fields = list(fields)
    if not fields:
        raise TrainingError(
            "No field has both an image and corrected labels. Correct some outlines "
            "first, or point at a folder of label images.")
    # THE ENGINE THE BASE MODEL RUNS ON decides the script, the learning rate and the batch: Cellpose 4
    # (Cellpose-SAM and what is trained from it) or Omnipose. A new model from scratch is Omnipose's only
    # (Cellpose 4 cannot start from nothing). Read from the model's name or file (`backends.backend_of`).
    from . import backends
    from .segment import SegmentationError as _SegErr

    scratch_base = base_model in (None, "", "none")
    engine_note = ""
    try:
        engine = backends.OMNIPOSE_BACKEND if scratch_base else backends.backend_of(base_model)
    except _SegErr as exc:
        # A BASE THAT CANNOT BE FOUND HERE (a lab model on another machine) was always only NOTED, the bundle
        # built for the engine this lab trains with (Omnipose); it is noted again, with the engine it assumed.
        engine = backends.OMNIPOSE_BACKEND
        engine_note = (f"could not tell which engine {base_model!r} runs on ({exc}); the bundle is built for "
                       f"Omnipose, as bundles always were")
    if engine == backends.CLASSICAL_BACKEND:
        raise TrainingError("The classical segmenter has nothing to train.")
    if learning_rate is None:
        learning_rate = 1e-5 if engine == backends.CELLPOSE_BACKEND else 0.05
    if batch_size is None and engine == backends.CELLPOSE_BACKEND:
        batch_size = 1
    out_dir = Path(out_dir)
    train, holdout_fields = split_fields(fields, holdout=holdout, seed=seed)

    import numpy as np

    for folder in (IMAGE_DIR, LABEL_DIR, HOLDOUT_IMAGE_DIR, HOLDOUT_LABEL_DIR):
        (out_dir / folder).mkdir(parents=True, exist_ok=True)
    notes, n_cells = ([engine_note] if engine_note else []), 0
    written, kept_back = [], []

    def convert(one, image_dir, label_dir):
        """Write one field's pair, or note why it could not be written."""
        try:
            image = tifffile.imread(one.image)
            labels = labels_for_field(one.labels, shape=image.shape[-2:])
        except (TrainingError, OSError, ValueError, RuntimeError) as exc:
            # NAMED, NOT DROPPED. A field silently missing from a training set is a
            # training set nobody can reconstruct.
            notes.append(f"skipped {one.stem}: {exc}")
            return None
        tifffile.imwrite(out_dir / image_dir / f"{one.stem}.tif", image)
        tifffile.imwrite(out_dir / label_dir / f"{one.stem}.tif",
                         labels.astype("uint16"))
        return labels

    for one in train:
        labels = convert(one, IMAGE_DIR, LABEL_DIR)
        if labels is None:
            continue
        n_cells += int(len({int(i) for i in np.unique(labels)} - {0}))
        written.append(one)
    # THE HELD-OUT HALF TRAVELS TOO. Writing only the training pairs and naming the
    # held-out SOURCES left the scorer with paths that a cluster cannot see, so it
    # skipped every field and reported nothing -- a comparison that silently does not
    # happen is worse than one that refuses.
    for one in holdout_fields:
        if convert(one, HOLDOUT_IMAGE_DIR, HOLDOUT_LABEL_DIR) is not None:
            kept_back.append(one)
    if not written:
        raise TrainingError(
            "Every field failed to convert: " + "; ".join(notes[:3]))

    scratch = base_model in (None, "", "none")
    info = {
        # NULL MEANS FROM SCRATCH, and it is recorded rather than replaced by a default
        # so a model can always say which of the two jobs produced it.
        "base_model": None if scratch else base_model,
        "from_scratch": bool(scratch),
        # THE YARDSTICK, always named. A model from scratch has no base to be measured
        # against, and measuring it against nothing is how "it trained" gets mistaken
        # for "it works".
        "compare_with": str(compare_with or (None if scratch else base_model)
                            or "bact_phase_omni"),
        # THE ENGINE the bundle trains with (26 Nov 2026); a bundle without it is an Omnipose one.
        "backend": engine,
        "model_name": model_name,
        "channels": list(channels),
        # CELLPOSE'S OWN CONVENTION, kept separate from the ROLE names above: [primary,
        # secondary], 0 = greyscale. One role means greyscale; a second means the model
        # is given it as cellpose's second channel rather than having it discarded.
        "cellpose_channels": [0, 0] if len(channels) < 2 else [1, 2],
        "n_epochs": int(n_epochs),
        # The working recipe's rate (the lab's models record "lr 0.05"), on record so the
        # script and the in-app path train alike.
        "learning_rate": float(learning_rate),
        # THE BATCH AND THE CHECKPOINTS, recorded only when set (14 Oct 2026). cellpose's
        # batch of 8 overflowed this machine's 4 GB GTX 1650, and Windows then spills a GPU's
        # memory into system RAM: 9 GB in 25 s on one user's fields, caught by the memory guard.
        # Mid-run checkpoints (``save_every``) let the held-out score pick the epoch; the
        # training loss cannot (a collapsed model has a healthy loss).
        **({"batch_size": int(batch_size)} if batch_size else {}),
        **({"save_every": int(save_every)} if save_every else {}),
        "n_train_fields": len(written),
        "n_cells": int(n_cells),
        "label_kinds": sorted({one.kind for one in written}),
        "train_fields": [one.stem for one in written],
        # THE HELD-OUT SET IS PART OF THE RECORD, so the score can be recomputed. A
        # score on a different split is a different measurement with the same name.
        "holdout_fields": [one.stem for one in kept_back],
        # WHERE THEY CAME FROM, for provenance only. The images and labels themselves
        # are in `holdout/`, because the score is computed where the training ran.
        "holdout_sources": {one.stem: str(one.image) for one in kept_back},
        "seed": int(seed),
        "written_by": "smtracker.segmentation.training.build_training_set",
    }
    (out_dir / PROVENANCE_FILE).write_text(json.dumps(info, indent=2), encoding="utf-8")
    cellpose_bundle = engine == backends.CELLPOSE_BACKEND
    (out_dir / "train.py").write_text(_TRAIN_SCRIPT_CELLPOSE if cellpose_bundle else _TRAIN_SCRIPT,
                                      encoding="utf-8")
    # THE BASE MODEL TRAVELS WITH THE BUNDLE. A lab model (bact_bsub_phase_smt) lives in
    # One user's model store; a cluster or another machine has only the built-ins, and
    # the script would otherwise fail there or, worse, fall back to another base.
    if not scratch:
        from .segment import CELLPOSE, SegmentationError, resolve_model_file

        try:
            if base_model in CELLPOSE:
                # Cellpose-SAM: the file if this machine has it, else the script downloads it where it runs
                have = backends.cellpose_weights_path(backends.CPSAM)
                if not backends.cpsam_downloaded():
                    raise SegmentationError(f"Cellpose-SAM is not downloaded on this machine ({have})")
                shutil.copy2(have, out_dir / "base_model.pt")
            else:
                shutil.copy2(resolve_model_file(base_model), out_dir / "base_model.pt")
        except SegmentationError as exc:
            notes.append(f"base model weights not copied in ({exc}); the script will look "
                         f"for {base_model!r} where it runs")
    # A BUNDLE THAT ARRIVES SOMEWHERE ELSE HAS NO WINDOW TO EXPLAIN IT, and "copy it
    # somewhere with a GPU" is not instructions. The notebook is the same `train.py`
    # with the Colab-shaped steps around it -- runtime, upload, train, SCORE, download.
    (out_dir / "README.md").write_text(_README_CELLPOSE if cellpose_bundle else _README, encoding="utf-8")
    if not cellpose_bundle:                    # the notebook installs and scores Omnipose
        (out_dir / "train_colab.ipynb").write_text(_COLAB_NOTEBOOK, encoding="utf-8")

    if not kept_back:
        notes.append("No held-out fields: too few to keep any back, so a trained model "
                     "cannot be scored against the base model here.")
    if scratch:
        notes.append(FROM_SCRATCH_NOTE)
    return TrainingSet(root=out_dir, train=written, holdout=kept_back,
                       base_model="(from scratch)" if scratch else base_model,
                       channels=tuple(channels), n_cells=n_cells, notes=notes)

def _clamp_bce_loss() -> None:
    """omnipose 1.1.4's boundary loss applies a probability BCE to raw outputs and dies with
    "input_val >= zero && input_val <= one". The environment that trained this lab's models
    clamps the inputs to [0, 1] by EDITING omnipose's own loss.py (the segmentation repo's
    `ensure_loss_patch`); the same clamp is applied here at run time, so no installed file
    is touched. The bundle's `train.py` does the same. Idempotent."""
    try:
        import omnipose.loss as loss
    except ImportError:                                   # pragma: no cover - env
        return
    bce = getattr(loss, "BatchMeanBSE", None)
    if bce is None or getattr(bce.forward, "_clamped", False):
        return
    original = bce.forward

    def forward(self, pred, target):
        return original(self, pred.clamp(0, 1), target.clamp(0, 1))

    forward._clamped = True
    bce.forward = forward


def training_runtime_available(backend: str | None = None) -> tuple:
    """``(available, why not)`` for training LOCALLY, for ``backend`` (``"cellpose"`` or ``"omnipose"``), or for
    either when it is None.

    Two separate requirements and they fail differently, so they are reported apart: the
    engine must be importable at all, and a GPU must be there. Training on CPU is
    not offered rather than merely slow -- a fine-tune that would take a day is a window
    that looks hung, which is the failure this stage has already had once.

    By `find_spec` for the package, so asking costs nothing; the GPU question does import
    torch and must never be asked while a widget is being built.
    """
    from . import backends

    wanted = (backend,) if backend else (backends.CELLPOSE_BACKEND, backends.OMNIPOSE_BACKEND)
    present = {backends.CELLPOSE_BACKEND: backends.cellpose_importable(),
               backends.OMNIPOSE_BACKEND: backends.omnipose_importable()}
    if not any(present.get(name, False) for name in wanted):
        if backends._frozen():
            return False, backends.backend_missing_words(wanted[0]) + " Export the training set instead."
        return False, ("no cellpose installed here. Export the training set and run it "
                       "where there is one: the bundle is self-contained")
    try:
        import torch
    except Exception as exc:                              # noqa: BLE001 - reported
        return False, f"torch will not import ({type(exc).__name__}); export instead"
    # TWO DIFFERENT FAILURES THAT LOOK ALIKE, and they want opposite answers.
    # `cuda.is_available()` is False both when there is no GPU and when torch is a
    # CPU-ONLY BUILD on a machine that has one -- and the second is the ordinary case
    # here, because this project's own environment installs CPU torch. Reported as
    # "no CUDA device visible" it sent somebody with a working GTX 1650 to the export
    # route for no reason. Measured on this machine: torch 2.13.0+cpu,
    # `torch.version.cuda` None, `nvidia-smi` showing the card quite happily.
    if getattr(torch.version, "cuda", None) is None:
        return False, (
            "This Python has a CPU-only torch, so it cannot use a GPU even if the "
            "machine has one. Install a CUDA build of torch, or export the training "
            "set and run it on another machine.")
    if not torch.cuda.is_available():
        return False, ("torch is a CUDA build but no device is visible (check the driver, or "
                       "whether another program is using the GPU). Fine-tuning on CPU is "
                       "impractically slow; export the training set instead.")
    return True, ""


#: What fine-tuning Cellpose-SAM needs of the GPU, in GiB, at a batch of one: the 300 million weights in 32-bit
#: floats (cellpose converts the network for training), their gradients and AdamW's two moments, about 5 GB, and
#: the activations on top. Said BEFORE training starts on a smaller card (`_train_cellpose`), never discovered as
#: a crash. An estimate from the parameter count; the real run on the lab's 4 GB card is recorded in
#: docs/claude/segmentation.md.
CELLPOSE_TRAIN_GIB = 6.0


def train_bundle(bundle_dir, *, on_line=None, use_gpu=True):
    """Fine-tune from a built bundle, in this process. Returns the new model's path.

    THE SAME BUNDLE THE CLUSTER RUNS. Local training is not a second implementation: it
    reads the same `provenance.json`, the same images and labels, and produces the same
    thing in the same place. If the two ever disagree, one of them is wrong and there
    would be no way to tell which. The bundle's ``backend`` names the engine, Cellpose 4 or
    Omnipose (a bundle without one is Omnipose's).

    Nothing here decides whether the result is good. That is `score_against_base`, and
    keeping them apart is what stops "it trained" being mistaken for "it helped".
    """
    import numpy as np
    import tifffile

    bundle_dir = Path(bundle_dir)
    record = bundle_dir / PROVENANCE_FILE
    if not record.is_file():
        raise TrainingError(f"No {PROVENANCE_FILE} in {bundle_dir}. Build the training set first.")
    info = json.loads(record.read_text(encoding="utf-8"))

    from . import backends

    engine = info.get("backend") or backends.OMNIPOSE_BACKEND
    ok, why = training_runtime_available(engine)
    if not ok:
        raise TrainingError(f"Cannot train here: {why}")

    images, labels = [], []
    for image_path in sorted((bundle_dir / IMAGE_DIR).glob("*.tif")):
        label_path = bundle_dir / LABEL_DIR / image_path.name
        if not label_path.is_file():
            continue
        images.append(np.asarray(tifffile.imread(image_path)))
        labels.append(np.asarray(tifffile.imread(label_path)))
    if not images:
        raise TrainingError(f"No image/label pairs in {bundle_dir}.")

    say = on_line or (lambda _line: None)
    if engine == backends.CELLPOSE_BACKEND:
        return _train_cellpose(bundle_dir, info, images, labels, say, use_gpu)
    return _train_omnipose(bundle_dir, info, images, labels, say, use_gpu)


def _train_cellpose(bundle_dir, info, images, labels, say, use_gpu) -> Path:
    """Cellpose 4's route: ``train.train_seg`` on the network of the base model, the bundle's settings, the
    result a NEW file under ``<bundle>/models/``. The same call the bundle's own ``train.py`` makes.

    The network is loaded FRESH (not the segmenting cache: training turns it into 32-bit floats), and the
    GPU's memory is checked against `CELLPOSE_TRAIN_GIB` first, so a card that cannot hold it says so in
    words instead of dying in a CUDA error; a CUDA out-of-memory that comes anyway is named the same way."""
    from . import backends
    from .segment import SegmentationError

    if not use_gpu:
        raise TrainingError("Cellpose-SAM is trained on a GPU only; a CPU run would take days.")
    import torch

    gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    name = torch.cuda.get_device_name(0)
    if gib < CELLPOSE_TRAIN_GIB:
        raise TrainingError(
            f"Not trained: fine-tuning Cellpose-SAM needs about {CELLPOSE_TRAIN_GIB:.0f} GB of GPU memory and "
            f"{name} has {gib:.1f} GB. The training set is built; export it and train it on a larger card, or "
            f"on Colab (the bundle's train.py does it).")
    base = info.get("base_model") or backends.CPSAM
    bundle_base = bundle_dir / "base_model.pt"
    say(f"{len(images)} field(s), fine-tuning Cellpose-SAM from {base!r} on {name} ({gib:.1f} GB)")
    try:
        net = backends.load_cellpose(str(bundle_base) if bundle_base.is_file() else str(base),
                                     use_gpu=True, cache=False)
        from cellpose import train as cp_train

        kwargs = dict(train_data=images, train_labels=labels, save_path=str(bundle_dir),
                      model_name=info["model_name"], n_epochs=int(info.get("n_epochs", 100)),
                      learning_rate=float(info.get("learning_rate", 1e-5)),
                      batch_size=int(info.get("batch_size", 1)), min_train_masks=1)
        if info.get("save_every"):
            kwargs.update(save_every=int(info["save_every"]), save_each=True)
        (bundle_dir / "models").mkdir(exist_ok=True)
        path, _train_losses, _test_losses = cp_train.train_seg(net.net, **kwargs)
    except torch.cuda.OutOfMemoryError as exc:
        raise TrainingError(
            f"Not trained: {name} ({gib:.1f} GB) ran out of memory fine-tuning Cellpose-SAM. The training set "
            f"is built; export it and train it on a larger card, or on Colab. ({exc})") from exc
    except SegmentationError as exc:
        raise TrainingError(str(exc)) from exc
    finally:
        backends.clear_model_cache()
    say(f"wrote {path}")
    return Path(path)


def _train_omnipose(bundle_dir, info, images, labels, say, use_gpu) -> Path:
    """The Omnipose route (source runs only: Omnipose is never bundled)."""
    try:
        from cellpose_omni import models
    except ImportError:                                   # pragma: no cover - env
        from cellpose import models

    from .segment import (
        OMNI_NCLASSES,
        _load_model,
        _require_omnipose_reconstruction,
        omnipose_compat,
    )

    # Training an omnipose model where cellpose_omni cannot use omnipose takes cellpose's
    # branch and crashes in `labels_to_flows`; say why instead (see
    # `segment.omnipose_reconstruction`).
    _require_omnipose_reconstruction()

    scratch = bool(info.get("from_scratch")) or info.get("base_model") in (None, "")
    say(f"{len(images)} field(s), "
        + ("training a NEW model from scratch"
           if scratch else f"fine-tuning from {info['base_model']!r}"))
    if scratch:
        say(FROM_SCRATCH_NOTE)
    out = bundle_dir / "models"
    out.mkdir(exist_ok=True)
    # `_apply_omnipose_runtime_patch` guards against the sparse-crop crash inside
    # cellpose_omni; see the helper's docstring for the two bugs and why the fallback
    # is safe. Measured on the heli bundles: the guard fires a few times a minute on
    # fields of ~3 cells over ~4 % of the frame, and without it an overnight run dies
    # within the hour.
    _apply_omnipose_runtime_patch()
    # THE FINE-TUNE GOES THROUGH `_load_model`, NOT `model_type=`. Omnipose's bacterial
    # models are four-class and the constructor defaults to three, so `model_type=` built
    # the wrong architecture, cellpose PRINTED "failed to load model" and carried on with
    # generic `cyto` -- a fine-tune from a base nobody asked for, reported as a success.
    # Measured: the loaded weights were byte-identical to `cytotorch_0`. Same defect as
    # `segment_image` carried, and `_load_model` verifies what it actually loaded.
    omnipose_compat()
    _clamp_bce_loss()
    # `pretrained_model=False` IS what makes it a new model. Passing `model_type=None`
    # alone still loads cellpose's default weights in some builds, which would be a
    # fine-tune wearing a scratch model's label.
    #
    # A scratch net in omnipose 1.1.4's numbering: four outputs are `nclasses=3`, and the
    # bacterial models are two-channel (`segment.checkpoint_nchan`).
    model = (models.CellposeModel(gpu=use_gpu, pretrained_model=False, omni=True,
                                  nclasses=OMNI_NCLASSES - 1, dim=2, nchan=2)
             if scratch else _load_model(info["base_model"], use_gpu=use_gpu))

    # THE THREE `train` ARGUMENTS THAT ARE NOT DEFAULTS, each because the default is
    # wrong for THIS data rather than wrong in general:
    #
    # `netstr` is what this build calls the model name (`model_name=` raises TypeError).
    #
    # `min_train_masks=1`: cellpose DISCARDS any field with fewer masks, defaulting to 5.
    # These fields hold a median of 4 cells, so the default would silently drop over half
    # the bundle and train on the crowded fields alone -- an unrepresentative subset,
    # reported as a full run.
    #
    # `rescale=False`: the recipe this project validated. Leaving it on lets cellpose
    # resize every field to its own diameter prior, which is the auto-rescale collapse
    # already recorded for the first overnight run.
    # `train_links`: omnipose 1.1.4 zips the labels with their links and fails on None.
    kwargs = dict(train_links=[None] * len(images),
                  channels=info.get("cellpose_channels", [0, 0]),
                  save_path=str(out), netstr=info["model_name"],
                  n_epochs=int(info.get("n_epochs", 100)),
                  min_train_masks=1, rescale=False)
    if info.get("learning_rate") is not None:
        kwargs["learning_rate"] = float(info["learning_rate"])
    for key in ("batch_size", "save_every"):
        if info.get(key) is not None:
            kwargs[key] = int(info[key])
    if info.get("save_every") is not None:
        # KEEP every mid-run checkpoint (found 14 Oct 2026): without ``save_each`` omnipose
        # overwrites ONE file at each ``save_every`` and numbers only the last ten epochs, so
        # the held-out score could pick among epochs 90-99 and nothing earlier.
        kwargs["save_each"] = True
    path = model.train(images, labels, **kwargs)
    say(f"wrote {path}")
    return Path(path)




def score_against_base(bundle_dir, model_path, *, base_model=None, use_gpu=True,
                       on_line=None) -> dict:
    """Score a new model AND its base on the bundle's own held-out fields.

    ``{"model": ..., "base": ..., "n_fields": n}``, both measured on exactly the fields
    the training never saw -- which travel INSIDE the bundle, so this runs wherever the
    training did.

    BOTH, ALWAYS. A number for the new model alone says nothing: the question is never
    "is this good" but "is this better than what it came from", and the base model on the
    same fields is the only thing that answers it.
    """
    import numpy as np
    import tifffile

    from .segment import SegmentConfig, segment_image

    bundle_dir = Path(bundle_dir)
    info = json.loads((bundle_dir / PROVENANCE_FILE).read_text(encoding="utf-8"))
    # THE YARDSTICK, which is the base model for a fine-tune and an explicitly named
    # pretrained model for one trained from scratch. Never nothing.
    base_model = base_model or info.get("base_model") or info.get(
        "compare_with") or "bact_phase_omni"
    pairs = sorted((bundle_dir / HOLDOUT_IMAGE_DIR).glob("*.tif"))
    if not pairs:
        raise TrainingError(
            "This bundle held nothing back, so a score would be on training data. "
            "Build it from more corrected fields.")

    say = on_line or (lambda _line: None)
    totals = {"model": [], "base": []}
    for image_path in pairs:
        label_path = bundle_dir / HOLDOUT_LABEL_DIR / image_path.name
        if not label_path.is_file():
            say(f"  {image_path.stem}: no held-out label; skipped")
            continue
        image = np.asarray(tifffile.imread(image_path))
        truth = np.asarray(tifffile.imread(label_path))
        for which, name in (("model", str(model_path)), ("base", str(base_model))):
            cfg = SegmentConfig(model=name, channels=("primary",), use_gpu=use_gpu)
            totals[which].append(score_labels(truth, segment_image(image, cfg)))
    if not totals["model"]:
        raise TrainingError("No held-out field could be scored.")

    def mean(rows, key):
        return float(sum(r[key] for r in rows) / len(rows))

    result = {
        "n_fields": len(totals["model"]),
        "model": {"mean_iou": mean(totals["model"], "mean_iou"),
                  "count_error": mean(totals["model"], "count_error")},
        "base": {"mean_iou": mean(totals["base"], "mean_iou"),
                 "count_error": mean(totals["base"], "count_error")},
    }
    say(f"held-out {result['n_fields']} field(s): new IoU "
        f"{result['model']['mean_iou']:.3f} against base "
        f"{result['base']['mean_iou']:.3f}")
    return result
