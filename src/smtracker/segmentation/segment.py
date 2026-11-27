"""Images become cell outlines: model -> masks -> capsules -> Oufti `.mat`.

The runtime the segmentation stage stands on. Pure: it takes paths and settings and
returns a report; nothing here opens a window, and the window calls this.

The channel is a required argument and has no default. Measured on real cells with
the photons redistributed and the geometry held identical, a DNA-bound protein segmented
on its own fluorescence returns outlines at **48 % of the true cell area** in 3 of 3
fields, and a membrane-bound one at 57 % with one field yielding no cells at all. Those
outlines look plausible, and a cell filter built on them drops exactly the peripheral
tracks the spatial claims rest on. `phase` images the CELL and is correct for any
localisation; `sum` is better where the label is cytosolic, which the software cannot
know about the next user's protein.

And it is checked rather than trusted. `fill_profile` measures where the signal sits
inside each mask (uniform, edge-peaked or centre-peaked) so a run on the wrong channel
says so instead of quietly handing back small cells. It is a REPORT, not a refusal: a
genuinely membrane-localised protein imaged deliberately is not an error.

Every rejected mask is named. A field that comes back with no cells is
indistinguishable from an empty field unless the run says which masks were dropped and
why, so the counts and reasons travel on the report.

What this does not do: choose a channel, choose a model, or write anywhere the caller did
not name. Training lives elsewhere and needs a GPU environment this never imports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .masks import MaskConfig, build_capsule

__all__ = [
    "SegmentConfig",
    "FieldResult",
    "RunReport",
    "PRETRAINED",
    "RECIPES",
    "CLASSICAL",
    "CELLPOSE",
    "builtin_models",
    "config_for",
    "prepare_model",
    "normalise",
    "fill_profile",
    "segment_image",
    "segment_field",
    "segment_folder",
]

#: The two models that ship with omnipose, and the channel each expects. A model trained
#: here declares its own roles; these are the bootstrap, for a project with no model yet.
PRETRAINED = {
    "bact_phase_omni": ("phase",),
    "bact_fluor_omni": ("sum",),
}

#: RECIPES: a pretrained model read with its own settings and its own mask -> outline steps,
#: offered beside the models under a name of its own (14 Oct 2026: the experimenter's March
#: 2026 pipeline, `recipe_march`). ``{name: (roles, label)}``; `config_for` turns one into the
#: run's `SegmentConfig`.
#: THE CLASSICAL SEGMENTER (`classical`), offered beside the models (asked 14 Oct 2026: "another way to segment
#: cells, user friendly ... smart settings"; "we can leave our current cell pose as well"): no network, no GPU, no
#: training, about 0.05 s a field, every decision readable. On the SUM image only -- measured on nine outlined
#: datasets, phase F1 0.0-0.4 (a textured pad looks locally like cells); sum F1 0.32-0.68 against Omnipose's
#: 0.33-0.84, so Omnipose stays the default (`docs/claude/segmentation.md`).
CLASSICAL = {"classical_sum": (("sum",), "Classical, sum image (no GPU, no training)")}

#: CELLPOSE 4's built-in model, offered beside the others when cellpose >= 4 can be imported
#: (`backends.cellpose_importable`): the generalist Cellpose-SAM, whose weights (about 1.2 GB) download once at
#: first use (`backends.download_cpsam`) and are never shipped (ruled 26 Nov 2026). ``{name: (roles, label)}``.
#: It reads PHASE by default (it is a generalist and the cell is what an outline is of); the folder it reads
#: is one click away on the Segment page.
CELLPOSE = {"cpsam": (("phase",), "Cellpose-SAM (generalist; downloads 1.2 GB at first use)")}

RECIPES = {
    "pipeline_2026_03": (("phase",),
                         "Phase pipeline: bact_phase_omni, chains split"),
    "pipeline_v3_2026_04": (("sum",), "Sum-image pipeline: bact_fluor_omni, chains split"),
}


def builtin_models() -> dict:
    """``{name: roles}`` of the built-in models THIS INSTALL CAN RUN: Cellpose-SAM where cellpose 4 is
    installed, the two Omnipose models where Omnipose is (`backends.available_backends`)."""
    from . import backends

    have = set(backends.available_backends())
    out = {}
    if backends.CELLPOSE_BACKEND in have:
        out.update({name: roles for name, (roles, _words) in CELLPOSE.items()})
    if backends.OMNIPOSE_BACKEND in have:
        out.update(PRETRAINED)
    return out


def _recipe_module(name: str):
    """The module holding recipe ``name``'s settings and steps."""
    if name == "pipeline_2026_03":
        from . import recipe_march as module
    elif name == "pipeline_v3_2026_04":
        from . import recipe_v3 as module
    else:
        raise SegmentationError(f"No segmentation recipe called {name!r}.")
    return module


@dataclass(frozen=True)
class SegmentConfig:
    """Everything the run needs that is not a path.

    `mask_threshold` 0.0 and `flow_threshold` 0.4 are the settings the experimenter's own
    v3 tool uses on fluorescence. Lowering the mask threshold grows the masks toward the
    true cell area and SHATTERS them: measured against curated meshes, -3.0 reaches 0.91
    of the true area but returns 7 labels where there is 1 cell, against 3 at 0.0. The
    capsule fit needs one mask per cell, so the coherent setting wins.
    """

    model: str = "bact_fluor_omni"
    channels: tuple[str, ...] = ("sum",)
    mask_threshold: float = 0.0
    flow_threshold: float = 0.4
    #: `None` lets the model choose; some builds of cellpose_omni require an integer and
    #: raise `TypeError: int() argument ... not 'NoneType'` without one, so the caller can
    #: supply it. Above about 50 the masks stop changing (measured: identical at 50, 100
    #: and 200), so this is a compatibility knob and not a tuning one.
    niter: int | None = 200
    #: Drop a mask smaller than this before fitting. Below it there is no cell.
    min_mask_area_px: int = 60
    #: A mask touching the border is a cell cut off by the field of view. Kept only when
    #: it is large enough to be most of a cell; the fit is on what is visible either way.
    border_margin_px: int = 3
    border_keep_area_px: int = 300
    #: Cellpose's size prior, in pixels of equivalent-circle diameter. 0 lets the model
    #: decide. It MATTERS on a generalist model and barely does on a bacterial one:
    #: measured on 12 WalKR fields, `cyto2` finds 1.25 cells/field at 30 and 1.42 at 45
    #: (true cells are 55 x 14 px, so an equivalent diameter near 30-45), while the
    #: omnipose models ignore it. No default beyond the model's own.
    diameter: int = 0
    #: Omnipose decoding. True is right for every `bact_*` model and WRONG for cellpose's
    #: own `cyto`/`cyto2`/`nuclei`, which are three-class and must be decoded as cellpose.
    #: Set from the CHECKPOINT rather than guessed -- see `_load_model`.
    omni: bool = True
    use_gpu: bool = False
    mask_config: MaskConfig = field(default_factory=MaskConfig)
    #: A RECIPE's mask -> outline steps instead of SMTracker's own (`RECIPES`); "" for ours.
    recipe: str = ""
    #: The CLASSICAL segmenter's settings (`classical.ClassicalParams`) when ``model`` is one of `CLASSICAL`.
    classical: object = None


def config_for(model: str, channels=None, *, use_gpu: bool = False,
               classical: dict | None = None) -> "SegmentConfig":
    """The run's settings for ``model``: a model's own, or a recipe's (its model, its
    thresholds, its steps), or the classical segmenter's from ``classical`` (the settings page's
    remembered dict, `classical_view`). The one place a name becomes settings."""
    if model in CLASSICAL:
        from .classical_view import params_from

        roles = tuple(channels or CLASSICAL[model][0])
        return SegmentConfig(model=model, channels=roles, use_gpu=False,
                             classical=params_from(classical, channel=roles[0]))
    if model in CELLPOSE:
        return SegmentConfig(model=model, channels=tuple(channels or CELLPOSE[model][0]),
                             omni=False, use_gpu=bool(use_gpu), niter=None)
    if model in RECIPES:
        recipe = _recipe_module(model)
        # niter left to the model, as the recipes' own calls left it: 200 (ours) gave 9 masks
        # where it gave 3 on one of one user's fields (measured 14 Oct 2026).
        return SegmentConfig(model=recipe.MODEL,
                             channels=tuple(channels or RECIPES[model][0]),
                             mask_threshold=recipe.MASK_THRESHOLD,
                             flow_threshold=recipe.FLOW_THRESHOLD, use_gpu=bool(use_gpu),
                             niter=None, recipe=recipe.NAME)
    cfg = SegmentConfig(model=model, channels=tuple(channels or PRETRAINED.get(model, ("phase",))),
                        use_gpu=bool(use_gpu))
    # A SAVED MODEL RUNS AT THE THRESHOLD IT WAS SCORED AT, when its record names one
    # (`models.SavedModel.mask_threshold`). The window lists saved models by their weights.
    from .models import ModelStoreError, list_models

    try:
        saved = [m for m in list_models() if str(m.weights) == str(model)]
    except (ModelStoreError, OSError):
        saved = []
    if saved and saved[0].mask_threshold is not None:
        from dataclasses import replace

        cfg = replace(cfg, mask_threshold=float(saved[0].mask_threshold))
    return cfg


@dataclass
class FieldResult:
    """One field: what was written, what was rejected, and what the signal looked like."""

    field: str
    cells: int = 0
    masks_found: int = 0
    rejected: dict = field(default_factory=dict)     # reason -> count
    path: Path | None = None
    fill: str = ""                                    # uniform / edge / centre
    #: Whether outlines drawn on PHASE sit where the fluorescence is. Empty when the
    #: question does not arise: no fluorescence channel, or the segmentation used it.
    alignment: str = ""
    alignment_shift_px: float = 0.0
    seconds: float = 0.0
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


@dataclass
class RunReport:
    """Every field, and the totals a console line needs."""

    results: list = field(default_factory=list)
    model: str = ""
    channels: tuple = ()

    @property
    def cells(self) -> int:
        return sum(r.cells for r in self.results)

    @property
    def failed(self) -> list:
        return [r for r in self.results if not r.ok]

    def rejected_total(self) -> dict:
        out: dict[str, int] = {}
        for r in self.results:
            for reason, n in r.rejected.items():
                out[reason] = out.get(reason, 0) + n
        return out

    def summary(self) -> str:
        lines = [f"{len(self.results)} field(s), {self.cells} cells, "
                 f"model {self.model}, channels {'+'.join(self.channels)}"]
        rejected = self.rejected_total()
        if rejected:
            lines.append("  masks rejected: " + ", ".join(
                f"{n} {reason}" for reason, n in sorted(rejected.items(),
                                                        key=lambda kv: -kv[1])))
        odd = [r for r in self.results if r.fill and r.fill != "uniform"]
        if odd:
            kinds = sorted({r.fill for r in odd})
            lines.append(
                f"  Note: the signal is {', '.join(kinds)}-peaked inside the masks in "
                f"{len(odd)} of {len(self.results)} fields. That is what a membrane- or "
                f"DNA-bound label looks like, and segmenting it gives outlines about "
                f"half the true cell area. If this protein is not cytosolic, segment "
                f"the phase channel instead.")
        for r in self.failed:
            lines.append(f"  failed {r.field}: {r.error}")
        return "\n".join(lines)


def normalise(image) -> np.ndarray:
    """Scale to 0-1. The models were trained on 8-bit images with the full range used."""
    a = np.asarray(image, dtype=np.float32)
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo + 1e-9)


#: Where the (inner - outer) / total shell imbalance falls for each localisation.
#:
#: **CALIBRATED AGAINST A KNOWN-CYTOSOLIC PROTEIN, because the flat assumption is wrong.**
#: The obvious thresholds, symmetric about zero, on the reasoning that a cytosolic label
#: fills the shells evenly, call real cytosolic cells centre-peaked in 4 of 4 fields.
#: Two effects push a genuinely uniform label toward the middle: the PSF dims the signal
#: at the boundary, and a cylinder seen in PROJECTION has more path length through its
#: axis than near its wall. So "uniform" is a measured band, not zero.
#:
#: MEASURED on a cytosolic protein, over six fields against their curated
#: meshes: **+0.201 to +0.293**, median +0.242. Against the two extremes rendered into
#: the same cells at the same photon budget: **membrane -0.681**, **DNA +0.705**. The
#: cuts sit in the empty space either side of the cytosolic band, roughly midway to each
#: extreme, so a real cytosolic run reads `uniform` with margin on both sides.
EDGE_CUT = -0.15
CENTRE_CUT = 0.50


def fill_profile(image, mask, *, edges=(0.33, 0.66)) -> str:
    """Where the signal sits inside a mask: `uniform`, `edge` or `centre`.

    The check that catches the wrong channel. Depth is the normalised distance transform
    (0 at the boundary, 1 at the deepest point) and the mean signal is taken in three
    shells. A membrane label piles into the outer shell; a nucleoid into the inner one.

    Returns `uniform` for anything between the two measured cuts, because this line is
    read by someone deciding whether their run is wrong and a weak preference is not
    evidence. See `EDGE_CUT` / `CENTRE_CUT` for what the bands were measured against.
    """
    from scipy.ndimage import distance_transform_edt

    mask = np.asarray(mask, dtype=bool)
    if mask.sum() < 20:
        return ""
    depth = distance_transform_edt(mask)
    depth = depth / (depth.max() or 1.0)
    signal = np.asarray(image, dtype=float)
    if (~mask).any():
        signal = signal - np.median(signal[~mask])
    shells = []
    bounds = (0.0, edges[0], edges[1], 1.01)
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        sel = mask & (depth >= lo) & (depth < hi)
        shells.append(float(signal[sel].mean()) if sel.any() else 0.0)
    outer, middle, inner = shells
    total = abs(outer) + abs(middle) + abs(inner)
    if total < 1e-9:
        return ""
    imbalance = (inner - outer) / total
    if imbalance <= EDGE_CUT:
        return "edge"
    if imbalance >= CENTRE_CUT:
        return "centre"
    return "uniform"



class SegmentationError(RuntimeError):
    """A segmentation run could not be set up as asked.

    Its own type, like every other module here, so a caller can tell a model that would
    not load from a mask that came out empty. The window catches the type and shows the
    message; an untyped RuntimeError would reach the console as a traceback.
    """

#: Omnipose's bacterial models are FOUR-class (it adds a distance field); cellpose's own
#: are three. The constructor defaults to `omni=False, nclasses=3`, so a four-class
#: checkpoint loaded without both flags fails its `load_state_dict` -- and cellpose
#: PRINTS "failed to load model" and carries on with whatever net it already built.
#: Nothing raises. Measured on this machine: `bact_fluor_omni`, `bact_phase_omni` and
#: every custom checkpoint all came back with byte-identical weights, those of generic
#: `cytotorch_0`. So the segmentation path silently ran the wrong model for everything.
OMNI_NCLASSES = 4

#: Downloaded omnipose weights are named `<model>torch_0` under `~/.cellpose/models`.
#: THE NAME IS RESOLVED HERE BECAUSE THIS BUILD CANNOT: `cellpose_omni.models.MODEL_NAMES`
#: holds no `bact_*` entry, so `model_type="bact_phase_omni"` matches nothing and falls
#: through to the default. The files are on disk; only the registry is missing.
CELLPOSE_MODEL_DIR = Path.home() / ".cellpose" / "models"


def resolve_model_file(model: str) -> Path:
    """The checkpoint `model` names, whether that is a path or a built-in name.

    Raises rather than falling back. A model that cannot be found is the one thing this
    must not paper over: cellpose's own fallback is what hid the defect above.
    """
    direct = Path(model)
    if direct.is_file():
        return direct
    # THE PROJECT'S OWN STORE IS SEARCHED TOO. Without it `_load_model` raised for every
    # model this tool trained -- they live per-user, not in cellpose's cache -- so a
    # model the store lists could not be run by name.
    try:
        from .models import list_models
        for saved in list_models():
            if saved.name == model and Path(saved.weights).is_file():
                return Path(saved.weights)
    except Exception:                                     # pragma: no cover - store opt.
        pass
    for candidate in (CELLPOSE_MODEL_DIR / f"{model}torch_0",
                      CELLPOSE_MODEL_DIR / model):
        if candidate.is_file():
            return candidate
    have = sorted(p.name for p in CELLPOSE_MODEL_DIR.glob("*")) if (
        CELLPOSE_MODEL_DIR.is_dir()) else []
    raise SegmentationError(
        f"No weights for {model!r}: it is not a file and not in {CELLPOSE_MODEL_DIR}. "
        f"Found there: {have or 'nothing'}.")


def checkpoint_nclasses(path) -> int:
    """How many output classes a checkpoint has: 3 for cellpose, 4 for omnipose.

    Read from the weights rather than assumed. Asserting 4 silently mis-decoded every
    three-class model, and asserting 3 would mis-decode every bacterial one.
    """
    import torch

    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    weight = state.get("output.2.weight")
    if weight is None:
        raise SegmentationError(f"{path} has no output layer, so it is not a cellpose checkpoint.")
    return int(weight.shape[0])


def checkpoint_nchan(path) -> int:
    """How many INPUT channels a checkpoint was built for (its first convolution's input),
    read from the weights. The bacterial omnipose models and every fine-tune of them are
    TWO-channel even though the images are greyscale, and omnipose 1.1.4's constructor
    defaults to one: built with one, `load_state_dict` fails ("running_mean should contain
    2 elements not 1") and cellpose PRINTS that and carries on with another net."""
    import torch

    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    first = state.get("downsample.down.res_down_0.conv.conv_0.2.weight")
    if first is None:
        raise SegmentationError(f"{path} has no first convolution, so it is not a cellpose checkpoint.")
    return int(first.shape[1])


def _contain_omnipose_logging() -> None:
    """Stop omnipose 1.1.4 from taking over Python's ROOT logger. Idempotent.

    **IMPORTING `omnipose.core` RECONFIGURES LOGGING FOR THE WHOLE PROCESS.** Its
    `logger.setup_logger` (run at import by `core`, `utils` and `gpu`) sets the root level
    to INFO, CLEARS every root handler and installs one whose `ColoredFormatter` WRITES the
    formatted line back into `record.msg` while leaving `record.args` in place. So after a
    segmentation every library's INFO chatter went to the console, and any second handler
    formatting the same record raises "not all arguments converted during string
    formatting": found 12 Oct 2026 as a figure-export test failing only in the full suite,
    where fontTools' "%s pruned" (written while subsetting a PDF font) met pytest's own
    handler after a segmentation test had imported omnipose.

    Two halves, because omnipose is not always imported through here: `setup_logger` is
    replaced by one that only names a logger, so a later import configures nothing, and a
    handler it already installed is removed, with the root level back at Python's default.
    """
    import logging

    try:
        from omnipose import logger as omni_logger
    except ImportError:                                   # pragma: no cover - env
        return
    if not getattr(omni_logger.setup_logger, "_smtracker_contained", False):
        def setup_logger(name=omni_logger.__name__):
            return logging.getLogger(name)

        setup_logger._smtracker_contained = True
        omni_logger.setup_logger = setup_logger
    root = logging.getLogger()
    theirs = [h for h in root.handlers
              if isinstance(h.formatter, omni_logger.ColoredFormatter)]
    for handler in theirs:
        root.removeHandler(handler)
    if theirs and root.level == logging.INFO:
        root.setLevel(logging.WARNING)


def omnipose_compat() -> None:
    """What omnipose 1.1.4 needs before it is imported here. Idempotent; call it first.

    THE RECIPE OF THE WORKING ENVIRONMENT (`miniconda3/envs/omnipose_clean`, and the
    segmentation repo's `_prepare_torch`), carried into this one:

    - `aicsimageio`, imported at the top of omnipose's own `cellpose_omni.io`, reads
      `tifffile.TIFF.RESUNIT`, which newer tifffile moved to `tifffile.RESUNIT`. Aliasing it
      lets that import succeed WITHOUT downgrading tifffile, which the rest of SMTracker
      reads raw movies with. Nothing here uses aicsimageio's readers -- images arrive as
      arrays -- so only the import has to work.
    - torch's JIT fuser off, as the working recipe does before inference and training.
    - omnipose's logging kept to omnipose (`_contain_omnipose_logging`).
    """
    _contain_omnipose_logging()
    # omnipose 1.1.4 calls scikit-image's `remove_small_holes(area_threshold=...)`, renamed in
    # skimage 0.26: a FutureWarning per mask, harmless, that filled the console (reported
    # 14 Oct 2026). Only that warning, only from omnipose.
    import warnings

    warnings.filterwarnings("ignore", message=r".*area_threshold.*", category=FutureWarning,
                            module=r"omnipose.*")
    try:
        import tifffile

        if not hasattr(tifffile.TIFF, "RESUNIT") and hasattr(tifffile, "RESUNIT"):
            tifffile.TIFF.RESUNIT = tifffile.RESUNIT
    except ImportError:                                   # pragma: no cover - env
        pass
    try:
        import torch

        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
    except Exception:                                     # noqa: BLE001 - optional knobs
        pass

def omnipose_reconstruction() -> tuple[bool, str]:
    """``(available, why not)``: will cellpose_omni rebuild an omnipose model's masks with
    OMNIPOSE's own method?

    **IT FALLS BACK IN SILENCE OTHERWISE.** `cellpose_omni` decides whether omnipose is
    installed once, at import, by trying `from omnipose.utils import format_labels` and
    `import ncolor`, and swallows the error. When that import fails, an omnipose model's
    flows are turned into masks by CELLPOSE's reconstruction, which on these rod-shaped
    cells returns small fragments: MEASURED on this project's held-out fields, the
    `bact_phase_omni` base model found 0 of 40 of one user's corrected cells and 2 of 36 of
    a second set's, two blobs per field. Training takes the same wrong branch and crashes in
    `labels_to_flows`. The packages that broke it here were a stand-alone cellpose-omni
    0.9.1 beside omnipose 1.1.4 (whose `utils` no longer re-exports `format_labels`) and
    ncolor 2.0.2 (which dropped the `unique_nonzero` omnipose 1.1.4 imports).
    """
    omnipose_compat()
    try:
        from cellpose_omni import io as omni_io
    except ImportError as exc:
        return False, f"cellpose_omni cannot be imported ({exc})"
    if getattr(omni_io, "OMNI_INSTALLED", False):
        return True, "omnipose's mask reconstruction is available"
    causes = []
    for statement in ("from omnipose.utils import format_labels", "import ncolor",
                      "import omnipose.core"):
        try:
            exec(statement, {})
        except Exception as exc:                          # noqa: BLE001 - named below
            causes.append(f"`{statement}` fails: {type(exc).__name__}: {exc}")
    from importlib.metadata import PackageNotFoundError, version

    def _version(name):
        try:
            return version(name)
        except PackageNotFoundError:
            return "absent"

    installed = ", ".join(f"{name} {_version(name)}"
                          for name in ("omnipose", "cellpose-omni", "ncolor"))
    return False, (
        "cellpose_omni cannot use omnipose in this Python, so an omnipose model's output "
        "would be turned into masks by cellpose's method, which returns fragments instead "
        "of whole cells. " + ("; ".join(causes) or "cellpose_omni reports omnipose as missing")
        + f". Installed: {installed}. These packages must be versions that work together.")


def _require_omnipose_reconstruction() -> None:
    ok, why = omnipose_reconstruction()
    if not ok:
        raise SegmentationError(why)


def _load_model(model: str, *, use_gpu: bool, omni=None, nclasses=None):
    """Build the net and CHECK it loaded what was asked for.

    The check is not ceremony. cellpose reports a failed `load_state_dict` on stdout and
    returns a working object built from other weights, so the only way to know which
    model ran is to ask the object afterwards.
    """
    from . import backends

    if not backends.omnipose_importable():
        raise SegmentationError(backends.backend_missing_words(backends.OMNIPOSE_BACKEND))
    omnipose_compat()
    from cellpose_omni import models

    path = resolve_model_file(model)
    if omni is None or nclasses is None:
        found = checkpoint_nclasses(path)
        nclasses = found if nclasses is None else nclasses
        # A THREE-CLASS CHECKPOINT IS A CELLPOSE MODEL AND MUST NOT BE DECODED AS
        # OMNIPOSE, and the reverse. Reading it off the file removes the guess: the
        # bacterial models here are four-class and `cyto`/`cyto2`/`nuclei` are three.
        omni = (found == OMNI_NCLASSES) if omni is None else omni
    if omni:
        _require_omnipose_reconstruction()
    # OMNIPOSE 1.1.4 COUNTS CLASSES WITHOUT ONE OF THE OUTPUTS: a bacterial checkpoint has
    # four output channels (two flows, distance, boundary) and its constructor calls that
    # `nclasses=3`; a cellpose one has three and is `nclasses=2`. `nclasses` here stays the
    # OUTPUT count, read off the file. The working recipe's `nclasses=3, dim=2, nchan=2`.
    net = models.CellposeModel(gpu=use_gpu, omni=omni, nclasses=nclasses - 1, dim=2,
                               nchan=checkpoint_nchan(path), pretrained_model=str(path))
    loaded = getattr(net, "pretrained_model", None)
    if isinstance(loaded, (list, tuple)):
        loaded = loaded[0] if loaded else None
    if loaded is None or Path(str(loaded)).resolve() != path.resolve():
        raise SegmentationError(
            f"cellpose loaded {loaded!r} when asked for {path}. This happens when the "
            f"checkpoint does not fit the architecture, and the run would have used "
            f"the wrong model.")
    return net


def prepare_model(cfg: SegmentConfig, on_status=None) -> None:
    """Everything a run needs BEFORE its first field, said as it happens: the engine must be here
    (a refusal by name otherwise), Cellpose-SAM's weights are downloaded if they are not (with the
    bytes so far through ``on_status(text)``), and the network is loaded. The classical segmenter
    has nothing to prepare. Raises `SegmentationError`; never falls back to another model."""
    from . import backends

    say = on_status or (lambda _text: None)
    engine = backends.backend_of(cfg.model)
    if engine == backends.CLASSICAL_BACKEND:
        return
    why = backends.backend_missing_words(engine)
    if why:
        raise SegmentationError(why)
    if engine == backends.CELLPOSE_BACKEND:
        if cfg.model in CELLPOSE and not backends.cpsam_downloaded():
            say(f"Downloading Cellpose-SAM's weights (about {backends.CPSAM_APPROX_BYTES / 2**30:.1f} GB, "
                f"once) from {backends.CELLPOSE_URL} to {backends.cellpose_model_dir()}")
            last = [-1.0]

            def progress(done, total):
                share = done / total if total else 0.0
                if share - last[0] >= 0.05 or (total and done >= total):
                    last[0] = share
                    say(f"Downloading Cellpose-SAM: {done / 2**20:.0f} MB"
                        + (f" of {total / 2**20:.0f} MB ({share * 100:.0f} %)" if total else ""))

            backends.download_cpsam(progress)
        say("Loading Cellpose-SAM" if cfg.model in CELLPOSE else f"Loading {Path(cfg.model).name}")
        backends.load_cellpose(cfg.model, use_gpu=cfg.use_gpu)
    else:
        _require_omnipose_reconstruction()
        _load_model(cfg.model, use_gpu=cfg.use_gpu)


def segment_image(image, cfg: SegmentConfig):
    """Run the model. Returns an integer label image.

    The engine is chosen from the MODEL (`backends.backend_of`: the classical segmenter's names,
    Cellpose-SAM, the built-in Omnipose names, else the checkpoint file's own tensors) and must be
    present in this install, or this raises by name: nothing falls back to another model.

    The import is deferred: `cellpose_omni` pulls in torch, and this module must be
    importable (and the rest of it testable) on a machine that has neither. `_load_model`
    imports it, after `omnipose_compat` has run.
    """
    from . import backends

    engine = backends.backend_of(cfg.model)
    if engine == backends.CLASSICAL_BACKEND:
        # NO NETWORK: `classical` thresholds, splits and shape-filters (no torch, no GPU)
        from .classical import ClassicalParams, segment as classical_segment

        params = cfg.classical or ClassicalParams(channel=cfg.channels[0])
        return np.asarray(classical_segment(image, params), dtype=np.int32)
    why = backends.backend_missing_words(engine)
    if why:
        raise SegmentationError(why)
    if engine == backends.CELLPOSE_BACKEND:
        return backends.run_cellpose(image, cfg)
    model = _load_model(cfg.model, use_gpu=cfg.use_gpu)
    omni = bool(getattr(model, "omni", cfg.omni))
    eight_bit = (normalise(image) * 255).astype(np.uint8)
    kwargs = dict(channels=[0, 0], diameter=cfg.diameter,
                  mask_threshold=cfg.mask_threshold,
                  flow_threshold=cfg.flow_threshold, omni=omni, cluster=omni,
                  resample=True, verbose=False)
    if cfg.niter is not None:
        kwargs["niter"] = cfg.niter
    masks, _flows, _styles = model.eval(eight_bit, **kwargs)
    return np.asarray(masks, dtype=np.int32)


def _usable_labels(masks, cfg: SegmentConfig):
    """Labels big enough to be a cell, with border-touching ones handled explicitly."""
    from skimage.measure import regionprops

    height, width = masks.shape
    keep = []
    for region in regionprops(masks):
        if region.area < cfg.min_mask_area_px:
            continue
        min_r, min_c, max_r, max_c = region.bbox
        at_border = (min_r < cfg.border_margin_px or min_c < cfg.border_margin_px
                     or max_r > height - cfg.border_margin_px
                     or max_c > width - cfg.border_margin_px)
        if at_border and region.area < cfg.border_keep_area_px:
            continue
        keep.append(region.label)
    return keep


def segment_field(images: dict, name: str, out_path=None, cfg: SegmentConfig | None = None,
                  image_shape=None) -> FieldResult:
    """One field: masks, capsules, and the `.mat` if `out_path` is given.

    `images` is `{role: array}`. The FIRST of `cfg.channels` is what the model reads;
    the others are carried so the fill check can look at the fluorescence even when the
    segmentation ran on phase.
    """
    import time

    from ..cellmesh import CellMesh, write_cell_meshes

    cfg = cfg or SegmentConfig()
    started = time.perf_counter()
    result = FieldResult(field=name)
    primary = cfg.channels[0]
    if primary not in images:
        result.error = f"no {primary!r} image for this field"
        return result

    try:
        masks = segment_image(images[primary], cfg)
    except Exception as exc:                              # noqa: BLE001 - reported
        result.error = f"{type(exc).__name__}: {exc}"
        result.seconds = time.perf_counter() - started
        return result

    if cfg.recipe:
        return _recipe_field(cfg, masks, images, name, out_path, image_shape, result, started)

    labels = _usable_labels(masks, cfg)
    result.masks_found = len(labels)

    meshes, fills = [], []
    for index, label in enumerate(labels, start=1):
        one = masks == label
        fit = build_capsule(one, cfg.mask_config)
        if not fit.ok:
            key = fit.reason.split(";")[0].split(",")[0]
            result.rejected[key] = result.rejected.get(key, 0) + 1
            continue
        meshes.append(CellMesh(movie=name, cell_id=str(len(meshes) + 1),
                               polygon=fit.outline, mesh=fit.mesh))
        # THE FILL CHECK LOOKS AT THE FLUORESCENCE, whichever channel was segmented. On
        # a phase run that is the point: it is how a phase-segmented project still says
        # "this protein is not cytosolic".
        probe = images.get("sum", images[primary])
        kind = fill_profile(probe, one)
        if kind:
            fills.append(kind)

    result.cells = len(meshes)
    if fills:
        result.fill = max(set(fills), key=fills.count)
    if out_path is not None and meshes:
        shape = image_shape or np.asarray(images[primary]).shape[-2:]
        result.path = write_cell_meshes(out_path, meshes, image_shape=tuple(shape), pixel_origin=0,
                                        overwrite=True)
    result.seconds = time.perf_counter() - started
    return result


def _recipe_field(cfg, masks, images, name, out_path, image_shape, result, started):
    """`segment_field` for a recipe: its own mask -> outline steps (`recipe_march`), then the
    same fill check and the same writer as SMTracker's path."""
    import time

    from ..cellmesh import CellMesh, write_cell_meshes

    cells, stats = _recipe_module(cfg.recipe).meshes_from_masks(masks)
    result.masks_found = int(stats["inside"] + stats["border"])
    if stats["failed"]:
        result.rejected["the recipe's shape gates"] = int(stats["failed"])
    primary = cfg.channels[0]
    probe = images.get("sum", images[primary])
    meshes, fills = [], []
    for mesh, outline, parent in cells:
        meshes.append(CellMesh(movie=name, cell_id=str(len(meshes) + 1),
                               polygon=np.asarray(outline, dtype=float),
                               mesh=np.asarray(mesh, dtype=float)))
        kind = fill_profile(probe, masks == parent)
        if kind:
            fills.append(kind)
    result.cells = len(meshes)
    if fills:
        result.fill = max(set(fills), key=fills.count)
    if out_path is not None and meshes:
        shape = image_shape or np.asarray(images[primary]).shape[-2:]
        result.path = write_cell_meshes(out_path, meshes, image_shape=tuple(shape), pixel_origin=0,
                                        overwrite=True)
    result.seconds = time.perf_counter() - started
    return result


def segment_folder(resolution, out_dir=None, cfg: SegmentConfig | None = None,
                   fields=None, progress=None) -> RunReport:
    """Every field a channel resolution names. Writes one `.mat` per field.

    `resolution` is a `channels.ChannelResolution`; `progress(done, total, result)` is
    called after each field so a window can show a line without this module knowing what
    a window is.

    One FIELD'S failure is not the RUN'S. A model that raises on one image should not
    cost the other twenty-nine: the failure is recorded on that field and named in the
    summary, which is the same rule the batch tracking and the HMM deposit follow.
    """
    import tifffile

    from .channels import common_fields, image_stems

    cfg = cfg or SegmentConfig()
    report = RunReport(model=cfg.model, channels=tuple(cfg.channels))
    names = list(fields) if fields is not None else common_fields(resolution)
    # ONE REFUSAL FOR THE RUN, not the same error on every field: an omnipose model in a
    # Python that cannot rebuild its masks fails them all (and before this guard, quietly
    # fragmented them all). A model that cannot be found is left to the per-field load,
    # which names it.
    from . import backends

    engine = backends.backend_of(cfg.model) if names else backends.CLASSICAL_BACKEND
    if engine != backends.CLASSICAL_BACKEND:
        why = backends.backend_missing_words(engine)
        if why:
            raise SegmentationError(why)
    if engine == backends.OMNIPOSE_BACKEND and names:
        # only a FOUR-class (Omnipose) checkpoint is rebuilt with Omnipose's method; a three-class cellpose one is
        # right with cellpose's own, and the guard must not refuse it
        try:
            omni_model = checkpoint_nclasses(resolve_model_file(cfg.model)) == OMNI_NCLASSES
        except SegmentationError:
            omni_model = True            # the built-in Omnipose names, whose weights are fetched when they load
        if omni_model:
            _require_omnipose_reconstruction()
    out_dir = Path(out_dir) if out_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    for done, name in enumerate(names, start=1):
        images = {}
        for role in resolution.resolved:
            # THE RESOLUTION KNOWS THE FILE: a loose snapshot's field is its movie's name
            # (`channels.snapshot_field`), which need not be the file's stem.
            candidate = resolution.image_path(role, name)
            if candidate is not None:
                try:
                    images[role] = tifffile.imread(candidate)
                except Exception:                         # noqa: BLE001 - reported below
                    pass
        if not images:
            result = FieldResult(field=name, error="no readable image in any channel")
        else:
            # INTO the movie's outline file if it is filed under another name with its number
            # (`cellmesh.outline_file`), never a second file beside it
            from ..cellmesh import outline_file

            result = segment_field(images, name,
                                   out_path=outline_file(out_dir, name) if out_dir else None,
                                   cfg=cfg)
        _check_alignment(result, images, cfg)
        report.results.append(result)
        if progress is not None:
            progress(done, len(names), result)
    return report


#: Channel roles that are FLUORESCENCE, the frames the tracks actually come from, and
#: therefore the thing an outline has to agree with.
FLUORESCENCE_ROLES = ("sum", "fluor", "fluorescence", "signal")


def _check_alignment(result, images, cfg) -> None:
    """Did the outlines land where the signal is? Fills `result.alignment`.

    ASKED FOR: "whenever mesh is drawn in brightfield, there needs to be a check if there
    is a huge shift in alignment (a few pixel is fine)". Phase and fluorescence come
    through different light paths and the registration drifts, so an outline traced on
    phase can sit several pixels off the cell as the TRACKING sees it, and nothing
    downstream complains, the cell filter simply keeps the wrong localisations.

    The check only asks the question where it exists. If the segmentation itself used
    the fluorescence channel then the outlines came FROM that image and would align with
    it by construction: a check there would report perfect agreement and mean nothing.
    So it runs only when the model's roles are phase-only and a fluorescence channel is
    present to compare against.

    It never raises and never corrects: a failure here must not cost a field its
    outlines, and moving a mesh is the experimenter's decision, not this function's.
    """
    if not result.ok or result.path is None:
        return
    roles = tuple(getattr(cfg, "channels", ()) or ())
    if any(role in FLUORESCENCE_ROLES for role in roles):
        result.alignment = ("not checked: the segmentation used the fluorescence "
                            "channel, so the outlines align with it by construction")
        return
    reference = next((images[r] for r in FLUORESCENCE_ROLES if r in images), None)
    if reference is None:
        result.alignment = ("not checked: no fluorescence channel in this folder to "
                            "compare the outlines against")
        return
    try:
        from ..cellmesh import read_cell_meshes
        from .alignment import mesh_image_offset
        from .masks import labels_from_outlines

        meshes = read_cell_meshes(result.path, movie=result.field, on_empty="empty")
        if not meshes:
            return
        frame = np.asarray(reference)
        if frame.ndim > 2:
            frame = frame.reshape(-1, *frame.shape[-2:]).mean(axis=0)
        labels = labels_from_outlines(
            [type("C", (), {"outline": np.asarray(m.polygon)})() for m in meshes],
            frame.shape, cell_ids=[int(m.cell_id) for m in meshes])
        got = mesh_image_offset(labels, frame)
    except Exception as exc:                          # noqa: BLE001 - reported, not raised
        result.alignment = f"not checked: {type(exc).__name__}: {exc}"
        return
    result.alignment = got["verdict"]
    result.alignment_shift_px = float(got["shift_px"])


@dataclass
class BatchReport:
    """Several conditions segmented in one run, each named by its folder.

    Asked for so a parent folder of conditions is one press rather than one session per
    condition. The folder NAME is the label, which is how this project distinguishes
    conditions everywhere else.

    ONE CONDITION'S FAILURE IS NOT THE RUN'S, the same rule `segment_folder` applies to
    a field and `trackmate_run.run_batch` applies to a movie. A condition whose channels
    do not resolve is a NAMED row carrying the reason, never an absence: three conditions
    where four were asked for reads as "no data", not as "one failed".
    """

    reports: dict = field(default_factory=dict)
    failures: dict = field(default_factory=dict)

    @property
    def n_ok(self) -> int:
        return len(self.reports)

    def summary(self) -> str:
        lines = [f"{self.n_ok} condition(s) segmented, {len(self.failures)} failed"]
        for name, report in self.reports.items():
            lines.append(f"  {name}: {report.summary()}"
                         if hasattr(report, "summary") else f"  {name}: done")
        for name, why in self.failures.items():
            lines.append(f"  {name}: failed, {why}")
        return "\n".join(lines)


def segment_conditions(conditions, roles, *, cfg: SegmentConfig | None = None,
                       configs: dict | None = None, overrides: dict | None = None,
                       out_name: str = "cell_meshes", progress=None,
                       field_progress=None) -> BatchReport:
    """Segment every condition folder in ``conditions``; the outlines go to
    ``<condition>/<out_name>/``.

    ``roles`` is the channel-role mapping `channels.resolve_channels` takes, used for any
    condition that has no config of its own.

    EACH CONDITION MAY HAVE ITS OWN MODEL. ``configs`` is ``{condition name:
    SegmentConfig}``; a condition absent from it falls back to ``cfg``. This entry point
    once took ONE model for all of them and said so in capitals -- that a condition
    needing its own model "must be run on its own, and this function cannot express
    that". It can now, because the restriction was never about the batch: a model is
    chosen per condition (phase against fluorescence, and a model trained on one strain),
    and forcing one across a parent folder made the ordinary case the awkward one.

    THE CHANNELS FOLLOW THE MODEL, and that is not a convenience. `bact_phase_omni` reads
    `phase` and `bact_fluor_omni` reads `sum` -- they are different IMAGES, not different
    weights over one image -- so a per-condition model with a shared channel mapping
    would read the wrong picture. A config's own ``channels`` is therefore used as that
    condition's roles; ``roles`` applies only where no config was given.

    ``overrides`` is ``{condition name: {role: path}}``, the folder a role is READ FROM
    when the user has said so. `resolve_channels` never second-guesses an override,
    including one naming a folder whose name says something else -- the person looking at
    the images knows what is in them -- so this is how a condition whose folders are named
    unusually is segmented without renaming anything on disk.

    ``progress(done, total, name)`` is called as each condition finishes;
    ``field_progress(name, done, total, result)`` is forwarded to `segment_folder` so a
    console keeps its per-field lines instead of going quiet for the length of a batch.
    """
    from .channels import resolve_channels

    batch = BatchReport()
    conditions = [Path(c) for c in conditions]
    configs = dict(configs or {})
    unknown = sorted(set(configs) - {Path(c).name for c in conditions})
    if unknown:
        # NAMED, NOT IGNORED. A config keyed on a condition that is not in the batch is
        # a typo or a stale name, and silently dropping it would run that condition
        # under someone else's model with nothing on screen to say so.
        raise ValueError(
            f"configs name condition(s) that are not in this batch: {', '.join(unknown)}")
    from ..cellmesh import mesh_folder as _mesh_folder

    for done, folder in enumerate(conditions, start=1):
        name = folder.name
        try:
            here = configs.get(name, cfg)
            # THE CHANNELS FOLLOW THE MODEL where a config was given for this condition,
            # since the two pretrained models read different IMAGES.
            want = tuple(here.channels) if (name in configs and here is not None) \
                else roles
            resolution = resolve_channels(
                folder, want, overrides=(overrides or {}).get(name))
            if not resolution.resolved:
                raise RuntimeError(
                    f"The image folders for {list(want)} could not be found under {folder}.")
            inner = None
            if field_progress is not None:
                def inner(fdone, ftotal, result, _name=name):
                    field_progress(_name, fdone, ftotal, result)
            batch.reports[name] = segment_folder(
                resolution, _mesh_folder(folder, out_name), here, progress=inner)
        except Exception as exc:                              # noqa: BLE001 - named
            batch.failures[name] = f"{type(exc).__name__}: {exc}"
        if progress is not None:
            progress(done, len(conditions), name)
    return batch
