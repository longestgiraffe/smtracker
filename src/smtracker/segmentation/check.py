"""What the command line says about segmentation: ``smtracker --segment-check`` and ``--train-bundle``.

Headless and Qt-free, so a FROZEN build can be asked, with no window and no Python beside it,
whether its segmentation works (`packaging/build.py`'s verify step asks it, and so can a lab member
who is told "it does not segment"):

- ``--segment-check`` alone: which engines this build carries (the classical segmenter, Cellpose 4,
  Omnipose), the torch and cellpose versions, whether CUDA is available. Exit 0 whenever it could
  answer; the LITE build answers "no torch" and exits 0, because that is its design.
- ``--segment-check IMAGE [--model NAME] [--cpu]``: also segments the one image through the same code
  path the windows use (`segment.segment_field`: model, masks, capsules, outlines), reports the
  cells found, and exits 1 on a refusal or no cells. ``--model`` is a built-in name (``cpsam``,
  ``classical_sum``, ``bact_phase_omni``) or a model file.
- ``--train-bundle DIR``: fine-tunes from a bundle built by the trainer (`training.train_bundle`), in
  this process: the frozen app has no separate python to run the bundle's ``train.py``, so the same
  call the trainer window makes is made here.
"""
from __future__ import annotations

from pathlib import Path

__all__ = ["engines_report", "segment_check", "train_check"]


def engines_report() -> list:
    """``[line]``: the engines, torch, cellpose and the GPU, one fact a line. Never imports torch unless it
    is installed; never raises."""
    from . import backends

    lines = [f"segmentation engines: {', '.join(backends.available_backends())}"]
    try:
        import torch
    except ImportError:
        lines.append("torch: not part of this build (the classical segmenter only)")
    except Exception as exc:                              # noqa: BLE001 - reported
        lines.append(f"torch: will not import ({type(exc).__name__}: {exc})")
    else:
        cuda = getattr(torch.version, "cuda", None)
        lines.append(f"torch: {torch.__version__}")
        try:
            usable = bool(torch.cuda.is_available())
            device = torch.cuda.get_device_name(0) if usable else "none"
            gib = (torch.cuda.get_device_properties(0).total_memory / 2**30) if usable else 0.0
        except Exception as exc:                          # noqa: BLE001 - reported
            usable, device, gib = False, f"{type(exc).__name__}: {exc}", 0.0
        lines.append(f"cuda: built for {cuda or 'no CUDA'}; available {usable}; device {device}"
                     + (f"; {gib:.1f} GB" if usable else ""))
    version = backends.cellpose_version()
    lines.append(f"cellpose: {version or 'not part of this build'}"
                 + ("" if not version or backends.cellpose_importable()
                    else " (older than 4: Cellpose-SAM cannot run)"))
    lines.append("omnipose: " + ("present (source run)" if backends.omnipose_importable()
                                 else "not part of this build"))
    lines.append(f"Cellpose-SAM weights: {'on this computer' if backends.cpsam_downloaded() else 'not downloaded'}"
                 f" ({backends.cellpose_model_dir()})")
    return lines


def segment_check(image, model: str = "cpsam", *, use_gpu: bool = True, out=None, say=print) -> int:
    """Segment ONE image through the windows' own code path and say what came out. Returns 0 when at least one
    cell was found, else 1. ``out`` is a folder for the outline file (a scratch folder when None, removed)."""
    import tempfile

    import numpy as np
    import tifffile

    from . import backends
    from .segment import SegmentationError, config_for, prepare_model, segment_field

    for line in engines_report():
        say(line)
    image = Path(image)
    if not image.is_file():
        say(f"no such image: {image}")
        return 1
    pixels = np.asarray(tifffile.imread(image))
    if pixels.ndim != 2:
        say(f"{image.name}: expected one 2-D image, got shape {pixels.shape}")
        return 1
    role = "phase" if model in ("cpsam",) or "phase" in model else "sum"
    try:
        cfg = config_for(model, (role,), use_gpu=bool(use_gpu))
        prepare_model(cfg, on_status=say)
        scratch = None if out else tempfile.TemporaryDirectory(prefix="smtracker-segcheck-")
        folder = Path(out) if out else Path(scratch.name)
        folder.mkdir(parents=True, exist_ok=True)
        result = segment_field({role: pixels}, image.stem, folder / f"{image.stem}.mat", cfg)
        if scratch is not None:
            scratch.cleanup()
    except SegmentationError as exc:
        say(f"segmentation refused: {exc}")
        return 1
    if result.error:
        say(f"segmentation failed: {result.error}")
        return 1
    say(f"segmented {image.name} with {model} on {'GPU' if cfg.use_gpu else 'CPU'} "
        f"({backends.backend_of(model)}): {result.masks_found} mask(s), {result.cells} cell(s) "
        f"after the capsule fit, {result.seconds:.1f} s"
        + (f"; rejected {dict(result.rejected)}" if result.rejected else ""))
    return 0 if result.cells > 0 else 1


def train_check(bundle, *, say=print) -> int:
    """Fine-tune from a trainer bundle in this process (`training.train_bundle`); 0 when a model file came out."""
    from .training import TrainingError, train_bundle

    for line in engines_report():
        say(line)
    try:
        path = train_bundle(bundle, on_line=say)
    except TrainingError as exc:
        say(f"not trained: {exc}")
        return 1
    say(f"trained: {path}")
    return 0 if Path(path).is_file() else 1
