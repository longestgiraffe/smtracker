"""The segmentation BACKENDS: which engines this install has, which one a model file belongs to,
and Cellpose 4 (Cellpose-SAM) itself. Headless; torch and cellpose are imported inside the calls
that need them, so this module imports (and its checks answer) where neither is installed.

Three engines, told apart by what they need and what they may ship with:

- ``classical``: the classical segmenter. No network, no torch. Always available.
- ``cellpose``: Cellpose 4 (MouseLand, BSD) with the generalist Cellpose-SAM model, ``cpsam``.
  Not in the packaged program (one build, no torch); it runs once the segmentation pack is downloaded (`smtracker.segpack`).
- ``omnipose``: the bacterial models (``bact_phase_omni``, ``bact_fluor_omni``) and the lab's
  fine-tunes of them. Omnipose's licence is non-commercial, which cannot sit inside a GPL
  application, so it is NEVER bundled; it works when SMTracker runs from source with omnipose
  installed, and is offered only then.

WHAT IS OFFERED IS WHAT CAN RUN (`available_backends`): by `find_spec`, so asking costs nothing
and imports nothing. The window never lists a model that would refuse at the first field.

A MODEL FILE IS RECOGNISED FROM ITS CHECKPOINT (`checkpoint_kind`), never from its name or its
folder: an omnipose network has an ``output.2.weight`` and a ``downsample`` stack, a Cellpose 4
one (Cellpose-SAM or its DINO sibling) has an ``encoder.patch_embed`` and an ``out.weight``.
Neither is refused by name, and a file that is both or neither is.

CELLPOSE-SAM'S WEIGHTS (about 1.2 GB) are NOT shipped. They download once, at first use, from the
URL cellpose itself uses (`CELLPOSE_URL`) into cellpose's own model folder
(`cellpose_model_dir`: ``CELLPOSE_LOCAL_MODELS_PATH``, else ``~/.cellpose/models``). Cellpose's
own downloader is not used, for two reasons that were read in its source (4.2.1.1): it switches
certificate checking off for the WHOLE PROCESS (`ssl._create_default_https_context`), and it has
no progress hook, so a window could not show that anything was happening. This one verifies TLS,
reports progress, writes beside the target and renames when the size matches, and a failure is
said by name (URL, folder, reason), never by falling back to another model.

**THE MODEL THAT LOADED IS THE MODEL ASKED FOR.** `cellpose.models.CellposeModel` falls back to
``cpsam_v2`` with a logged warning when the name it is given is not a file or a name it knows
(the silent-fallback defect this package has met before with cellpose_omni). So a file is always
passed, never a name, and `load_cellpose` asks the object afterwards which file it loaded.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

__all__ = [
    "CLASSICAL_BACKEND", "CELLPOSE_BACKEND", "OMNIPOSE_BACKEND", "BACKEND_NAMES",
    "CELLPOSE_URL", "CPSAM", "CPSAM_APPROX_BYTES", "PACK_WORDS",
    "available_backends", "omnipose_importable", "cellpose_importable", "cellpose_version",
    "cellpose_model_dir", "cellpose_weights_path", "cpsam_downloaded", "download_cpsam",
    "checkpoint_kind", "checkpoint_shapes", "backend_of", "load_cellpose", "run_cellpose", "backend_missing_words",
    "clear_model_cache",
]

CLASSICAL_BACKEND = "classical"
CELLPOSE_BACKEND = "cellpose"
OMNIPOSE_BACKEND = "omnipose"
BACKEND_NAMES = {CLASSICAL_BACKEND: "the classical segmenter",
                 CELLPOSE_BACKEND: "Cellpose",
                 OMNIPOSE_BACKEND: "Omnipose"}

#: The built-in Cellpose-SAM model, the name it is listed and stored under.
CPSAM = "cpsam"
#: Where cellpose 4.2.1.1 fetches its models from (``cellpose.models._MODEL_URL``).
CELLPOSE_URL = "https://huggingface.co/mouseland/cellpose-sam/resolve/main/"
#: What the download is about, for the line said before it starts (the file is 1,233,587,898 bytes).
CPSAM_APPROX_BYTES = 1_233_587_898
#: What the packaged program says in place of the models: there is ONE build, without torch; the models come with the
#: segmentation pack, downloaded on request (`smtracker.segpack`, ruled 26 Nov 2026).
PACK_WORDS = ("This program segments with the classical segmenter only until the segmentation pack (PyTorch and "
              "Cellpose) is downloaded. Open Segmentation pack in the window's header to get it.")


def _spec(name: str) -> bool:
    if name in sys.modules:                               # already imported counts as importable
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def omnipose_importable() -> bool:
    """Whether the Omnipose stack is here (omnipose and its cellpose_omni)."""
    return _spec("omnipose") and _spec("cellpose_omni")


def cellpose_importable() -> bool:
    """Whether Cellpose 4 is here. Version 4 is the one with ``cellpose.vit`` (the SAM network);
    an older cellpose has no such module and cannot run Cellpose-SAM, so it does not count."""
    return _spec("cellpose") and _spec("cellpose.vit") and _spec("torch")


def cellpose_version() -> str:
    """The installed cellpose's version, or ``""``."""
    if not _spec("cellpose"):
        return ""
    try:
        from cellpose.version import version
    except Exception:                                     # noqa: BLE001 - reported as absent
        return ""
    return str(version)


def available_backends() -> tuple:
    """The engines this install can run, in the order the window lists them: the classical
    segmenter always, then Cellpose, then Omnipose."""
    out = [CLASSICAL_BACKEND]
    if cellpose_importable():
        out.append(CELLPOSE_BACKEND)
    if omnipose_importable():
        out.append(OMNIPOSE_BACKEND)
    return tuple(out)


def backend_missing_words(backend: str) -> str:
    """Why ``backend`` cannot run here, in plain words; ``""`` where it can. Never offers an
    install (the optional-package rule for a frozen build: a build that lacks it is the wrong build)."""
    if backend == CLASSICAL_BACKEND:
        return ""
    if backend in available_backends():
        return ""
    if backend == CELLPOSE_BACKEND:
        return PACK_WORDS if _frozen() else (
            "Cellpose 4 is not installed in this Python, so Cellpose-SAM cannot run here.")
    if backend == OMNIPOSE_BACKEND:
        return ("The Omnipose models are not part of the installed program: Omnipose's licence does "
                "not allow it. They run only when SMTracker is started from source with Omnipose installed."
                if _frozen() else
                "Omnipose is not installed in this Python, so the Omnipose models cannot run here.")
    return f"unknown segmentation engine {backend!r}"


def _frozen() -> bool:
    import sys

    return bool(getattr(sys, "frozen", False))


# ------------------------------------------------------------------- the weights
def cellpose_model_dir() -> Path:
    """Where Cellpose keeps its models: ``CELLPOSE_LOCAL_MODELS_PATH`` when set, else
    ``~/.cellpose/models`` (the rule cellpose's own ``models.MODEL_DIR`` follows, read here at
    call time so it can be pointed at another drive before the first download)."""
    env = os.environ.get("CELLPOSE_LOCAL_MODELS_PATH")
    return Path(env) if env else Path.home() / ".cellpose" / "models"


def cellpose_weights_path(name: str = CPSAM) -> Path:
    return cellpose_model_dir() / name


def cpsam_downloaded() -> bool:
    """Whether Cellpose-SAM's weights are already on this computer."""
    path = cellpose_weights_path(CPSAM)
    return path.is_file() and path.stat().st_size > 0.9 * CPSAM_APPROX_BYTES


def download_cpsam(on_progress=None, *, timeout: float = 60.0, chunk: int = 1 << 20) -> Path:
    """Fetch Cellpose-SAM's weights into `cellpose_model_dir`, once. Returns the file.

    ``on_progress(done_bytes, total_bytes_or_None)`` is called as the bytes arrive. Written to
    ``cpsam.part`` and renamed only when the bytes received match the server's length, so a cut
    connection never leaves a file that reads as the model. Raises `SegmentationError` naming the
    address, the folder and the reason on any failure.
    """
    import urllib.error
    import urllib.request

    from .segment import SegmentationError

    target = cellpose_weights_path(CPSAM)
    if cpsam_downloaded():
        return target
    url = CELLPOSE_URL + CPSAM
    folder = target.parent
    part = target.with_name(target.name + ".part")
    try:
        folder.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=timeout) as reply:         # verified TLS
            length = reply.headers.get("Content-Length")
            total = int(length) if length else None
            done = 0
            with open(part, "wb") as out:
                while True:
                    block = reply.read(chunk)
                    if not block:
                        break
                    out.write(block)
                    done += len(block)
                    if on_progress is not None:
                        on_progress(done, total)
        if total is not None and done != total:
            raise SegmentationError(f"the download stopped after {done} of {total} bytes")
        if done < 0.9 * CPSAM_APPROX_BYTES:
            raise SegmentationError(f"the file is {done} bytes, far short of the expected "
                                    f"{CPSAM_APPROX_BYTES}")
        os.replace(part, target)
    except SegmentationError as exc:
        _drop(part)
        raise SegmentationError(f"Cellpose-SAM's weights could not be downloaded from {url} into "
                                f"{folder}: {exc}. Nothing else was used in their place.") from exc
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        _drop(part)
        raise SegmentationError(f"Cellpose-SAM's weights could not be downloaded from {url} into "
                                f"{folder}: {type(exc).__name__}: {exc}. Check the internet "
                                f"connection and the free space on that drive. Nothing else was "
                                f"used in their place.") from exc
    return target


def _drop(path: Path) -> None:
    try:
        Path(path).unlink()
    except OSError:
        pass


# ------------------------------------------------------------------- recognising a file
def checkpoint_shapes(path) -> dict:
    """``{tensor name: shape}`` of a checkpoint, read without loading its weights (memory-mapped)."""
    try:
        import torch
    except ImportError as exc:
        from .segment import SegmentationError

        raise SegmentationError(
            f"{path} cannot be read here: torch is not part of this build. "
            + PACK_WORDS) from exc
    try:
        state = torch.load(Path(path), map_location="cpu", weights_only=True, mmap=True)
    except Exception:                                     # noqa: BLE001 - a legacy file has no mmap
        state = torch.load(Path(path), map_location="cpu", weights_only=True)
    return {k: tuple(v.shape) for k, v in state.items()} if hasattr(state, "items") else {}


def checkpoint_kind(path) -> str:
    """``"cellpose"`` or ``"omnipose"`` for a checkpoint FILE, read from its tensors. Raises
    `SegmentationError` naming the file when it is neither (or looks like both)."""
    from .segment import SegmentationError

    path = Path(path)
    # asked once per field by a run, so a file is read once while it is unchanged
    try:
        stat = path.stat()
        stamp = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    except OSError:
        stamp = None
    if stamp is not None and stamp in _KIND_CACHE:
        return _KIND_CACHE[stamp]
    kind = _checkpoint_kind(path)
    if stamp is not None:
        _KIND_CACHE[stamp] = kind
    return kind


_KIND_CACHE: dict = {}


def _checkpoint_kind(path: Path) -> str:
    from .segment import SegmentationError

    try:
        shapes = checkpoint_shapes(path)
    except SegmentationError:
        raise
    except Exception as exc:                              # noqa: BLE001 - named
        raise SegmentationError(f"{path} is not a model file torch can read "
                                f"({type(exc).__name__}: {exc}).") from exc
    keys = set(shapes)
    cellpose4 = "encoder.patch_embed.proj.weight" in keys and "out.weight" in keys
    # THE CELLPOSE_OMNI FAMILY: a `downsample.` stack and an `output.2.weight`. Omnipose's bacterial networks (four
    # output channels) and Cellpose 3's own (cyto, cyto2, nuclei: three) share every layer name, and one engine runs
    # both (cellpose_omni, which `_load_model` has always told apart by the output count), so the family is one
    # engine here and a source run keeps running the older models it always ran.
    omnipose = "output.2.weight" in keys and any(k.startswith("downsample.") for k in keys)
    if cellpose4 and not omnipose:
        return CELLPOSE_BACKEND
    if omnipose and not cellpose4:
        return OMNIPOSE_BACKEND
    raise SegmentationError(
        f"{path.name} is neither a Cellpose 4 model (Cellpose-SAM and the models trained from it) nor an "
        f"Omnipose model. It holds {len(keys)} tensors, "
        + ("those of both at once." if cellpose4 and omnipose else "none of the layers either engine has."))


def backend_of(model: str) -> str:
    """The engine a model NAME or FILE runs on: the classical segmenter's names, ``cpsam``, the
    built-in Omnipose names, else the checkpoint's own kind (the file is read)."""
    from .segment import CELLPOSE, CLASSICAL, PRETRAINED, RECIPES, SegmentationError, resolve_model_file

    if model in CLASSICAL:
        return CLASSICAL_BACKEND
    if model in CELLPOSE:
        return CELLPOSE_BACKEND
    if model in PRETRAINED or model in RECIPES:
        return OMNIPOSE_BACKEND
    try:
        return checkpoint_kind(resolve_model_file(model))
    except SegmentationError:
        raise


# ------------------------------------------------------------------- running Cellpose 4
_MODEL_CACHE: dict = {}


def clear_model_cache() -> None:
    """Drop the loaded Cellpose network (frees its GPU memory)."""
    _MODEL_CACHE.clear()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:                                     # noqa: BLE001 - best effort
        pass


def load_cellpose(model: str, *, use_gpu: bool, on_progress=None, cache: bool = True):
    """The Cellpose 4 network for ``model`` (``cpsam``, or a checkpoint file), CHECKED to be the
    file asked for. One network is kept between fields (reloading 1.2 GB per image is the slow
    part), keyed by file and device."""
    from .segment import CELLPOSE, SegmentationError, resolve_model_file

    if not cellpose_importable():
        raise SegmentationError(backend_missing_words(CELLPOSE_BACKEND))
    if model in CELLPOSE:
        path = download_cpsam(on_progress)
    else:
        path = resolve_model_file(model)
        kind = checkpoint_kind(path)
        if kind != CELLPOSE_BACKEND:
            raise SegmentationError(f"{path.name} is an {BACKEND_NAMES[kind]} model, not a Cellpose 4 one.")
    key = (str(Path(path).resolve()), bool(use_gpu))
    cached = _MODEL_CACHE.get(key) if cache else None
    if cached is not None:
        return cached
    if cache:
        _MODEL_CACHE.clear()                              # one network at a time (4 GB cards)
    from cellpose import models as cp_models

    net = cp_models.CellposeModel(gpu=bool(use_gpu), pretrained_model=str(path))
    loaded = getattr(net, "pretrained_model", None)
    if loaded is None or Path(str(loaded)).resolve() != Path(path).resolve():
        raise SegmentationError(
            f"cellpose loaded {loaded!r} when asked for {path}. It falls back to another model "
            f"when it cannot use the one it is given, and the run would have used the wrong model.")
    if use_gpu and not bool(getattr(net, "gpu", False)):
        raise SegmentationError("the GPU was asked for but cellpose is running on the CPU. Untick "
                                "the GPU box, or fix the CUDA driver; the run would otherwise be "
                                "much slower than the person at the window expects.")
    if cache:
        _MODEL_CACHE[key] = net
    return net


def run_cellpose(image, cfg):
    """Segment one image with Cellpose 4: an int32 label image (0 = background).

    Reads the run's settings as Cellpose does: ``mask_threshold`` is its ``cellprob_threshold``,
    ``flow_threshold`` its own, ``diameter`` 0 lets the model decide. ``niter`` is left to the model
    (Cellpose 4 sizes it from the cell), as the recipes leave it.
    """
    import numpy as np

    from .segment import normalise

    net = load_cellpose(cfg.model, use_gpu=cfg.use_gpu)
    pixels = normalise(image).astype(np.float32)          # Cellpose normalises again, by percentile
    masks, _flows, _styles = net.eval(
        pixels, diameter=(float(cfg.diameter) if cfg.diameter else None),
        cellprob_threshold=float(cfg.mask_threshold), flow_threshold=float(cfg.flow_threshold),
        batch_size=_batch_for(net), resample=True, progress=None)
    return np.asarray(masks, dtype=np.int32)


def _batch_for(net) -> int:
    """Tiles per forward pass. 8 is cellpose's own; a card with under 6 GB (the lab's GTX 1650 has 4)
    runs 2, and the CPU runs 4."""
    try:
        import torch

        if getattr(net, "gpu", False) and torch.cuda.is_available():
            gb = torch.cuda.get_device_properties(0).total_memory / 2**30
            return 8 if gb >= 6 else 2
    except Exception:                                     # noqa: BLE001 - the default
        pass
    return 4
