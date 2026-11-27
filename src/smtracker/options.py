"""Each stage's OPTIONS SCREEN: the few standards a user sets once.

Asked for (13 Oct 2026): "an option screen for each tool (imageprep/track, segmentation,
analysis). should be not too much, just maybe some important things. maybe like a standard
setting editor for tracking ... (utrack and trackmate)".

AN OPTION IS A STANDARD, NEVER A SECOND CONTROL. Every value here already has its control
on a page: the Movies page's start rule and tolerance, the Detect & link grid, the u-track
card, the segmentation GPU switch and model, the Compute dialog. The option is what that
control STARTS FROM in a new session. Saving a new standard moves only the values that
still sit at the old one (`follows`); a value changed on a page always wins. Two controls
deciding one number is the defect this project has unpicked three times.

EVERY OPTION HAS A READER, named on it (``read_by``) and pinned by
`tests/test_options.py`: a declared, documented setting that nothing reads is worse than
none (`require_all_inside`).

HEADLESS: plain values in and out. The windows read the stored record with
``theme.remembered(OPTIONS_KEY)`` and hand it here; this module never touches a file.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["OPTIONS_KEY", "Option", "OptionError", "options_for", "TOOL_TITLES", "option",
           "parse", "read", "values", "update", "reset", "differences", "follows",
           "utrack_standard", "UTRACK_STANDARD_FIELDS", "compute_values",
           "outline_tolerance_nm", "max_tau_spread", "notify_after", "updates_at_start"]

#: The one ui.json key every stage's options live under, ``{option key: value}``.
OPTIONS_KEY = "options"


class OptionError(ValueError):
    """A value an option cannot take, named with the option and its range."""


@dataclass(frozen=True)
class Option:
    key: str
    label: str
    group: str
    kind: str                     # "number", "choice", "switch" or "folder"
    default: object
    help: str
    #: Where the value is read, so it cannot become a setting nothing reads.
    read_by: str
    unit: str = ""
    low: float | None = None
    high: float | None = None
    integer: bool = False
    decimals: int = 2
    #: ``((value, words), ...)`` for a choice; empty when the window supplies them
    #: (the segmentation models, which include the user's own).
    choices: tuple = ()


def _utrack(key: str, label: str, unit: str, *, low, high, integer=False, decimals=2):
    from .prep_view import UTRACK_FIELDS
    from .utrack_run import LAB_STANDARD

    words = {k: help_ for k, _l, _u, _g, _kind, _adv, help_ in UTRACK_FIELDS}[key]
    return Option(f"utrack.{key}", label, "u-track standard", "number",
                  getattr(LAB_STANDARD, key), words,
                  read_by="ui.prep.window: the u-track card starts from it and Lab standard "
                          "resets to it",
                  unit=unit, low=low, high=high, integer=integer, decimals=decimals)


def _notify_default() -> int:
    from .run_progress import NOTIFY_AFTER_S

    return int(NOTIFY_AFTER_S)


_NOTIFY = Option(
    "notify_after_s", "Pop-up when a job ran at least", "When a job finishes", "number",
    _notify_default(), "A job that ran at least this long ends with a pop-up saying what it did, and the "
          "taskbar flashes. 0 shows a pop-up after every job.",
    read_by="ui.widgets.progress.notify_after_s", unit="s", low=0, high=24 * 3600,
    integer=True)


#: SHARED like `_NOTIFY`: every screen lists it, one key in the one record (27 Nov 2026, `smtracker.updates`).
_UPDATES = Option(
    "check_updates_at_start", "Check for updates at start", "Updates", "switch", True,
    "Once a day, when SMTracker starts, it asks GitHub whether a newer version is published and tells you "
    "if there is one. It sends nothing about you or your data, and installs nothing until you press the "
    "button. Help > Check for updates still works when this is off.",
    read_by="ui.widgets.update_notice.start_check (updates.run_check, enabled=)")


def _prep_options() -> tuple:
    from .movieprep import FLAT_TOLERANCE
    from .prep_view import START_RULES
    from .trackmate_run import ADAPTIVE_BASE, LINKING_RADIUS_PX, UTRACK_MATCHED_K

    return (
        _NOTIFY,
        _UPDATES,
        Option("crop_start_rule", "Start rule", "Crop proposals", "choice", "strictest",
               "Which floor a proposed crop starts at. Strictest takes the latest of the "
               "four. You can still change it on the Movies page.",
               read_by="ui.prep.window: the Movies page's Start rule",
               choices=tuple(START_RULES)),
        Option("crop_flat_tolerance_pct", "Flat within", "Crop proposals", "number",
               round(FLAT_TOLERANCE * 100, 2),
               "How far the frame mean may wander inside a stretch still called flat, as a "
               "percentage of its level. Lower it for a camera with a high dark level and a "
               "weak signal, where 2 % can be a large part of the signal.",
               read_by="ui.prep.window: the Movies page's Flat within",
               unit="%", low=0.1, high=25.0, decimals=2),
        Option("trackmate_linking_px", "Linking radius", "TrackMate standard", "number",
               float(LINKING_RADIUS_PX),
               "What a condition starts with in the Detect & link grid. It caps the fastest D "
               "that can be linked, and the cap falls with a longer frame interval: at 50 ms, "
               "5 px covers D only up to about 0.34 µm²/s.",
               read_by="prep_model.seed_tracking (base=)", unit="px", low=0.5, high=50.0,
               decimals=1),
        Option("trackmate_frame_gap", "Frame gap", "TrackMate standard", "number",
               int(ADAPTIVE_BASE["max_frame_gap"]),
               "Frames a molecule may vanish for and still be linked. The gap is a time, so "
               "the same number of frames allows a longer disappearance at a longer frame "
               "interval.",
               read_by="prep_model.seed_tracking (base=)", unit="frames", low=0, high=20,
               integer=True),
        Option("trackmate_spot_precision", "Fit each spot's precision", "TrackMate standard", "switch", True,
               "After TrackMate, each tracked spot is refitted with a 2D Gaussian and its precision saved "
               "beside the XML; positions and tracks do not change. Per-spot precision needs this, and the "
               "condition's optics.",
               read_by="prep_model.seed_tracking (base=) -> run_inputs (spot_precision)"),
        # A NEW KEY with the scale: a k stored before 13 Oct 2026 was on the old LoG scale
        # (sigma^2/2 times TrackMate's), and read on the new one it would be far too lenient.
        Option("trackmate_threshold_k_tm", "Threshold", "TrackMate standard", "number",
               float(UTRACK_MATCHED_K),
               "Detection threshold as a multiple of each movie's LoG noise, on TrackMate's "
               "spot-quality scale. The default was calibrated on one microscope; on another, "
               "match the spot count of a u-track run.",
               read_by="ui.prep.jobs.measure_noise and prep_model.rethreshold (k=)",
               unit="x TrackMate's LoG noise", low=1.0, high=20.0, decimals=2),
        Option("utrack_work_folder", "u-track work folder", "u-track standard", "folder", "",
               "Where u-track keeps its working frames. Empty means beside the data. Choose an SSD if "
               "the data is on a hard disk; 40 movies of 2500 frames need about 3 GB.",
               read_by="utrack_run.run_batch (work_folder=), via prep_model."
                       "track_condition_utrack"),
        _utrack("alpha_loc_max", "Spot significance (alphaLocMax)", "", low=1e-6, high=0.5,
                decimals=4),
        _utrack("max_search_radius_px", "Search radius, largest", "px", low=0.5, high=50.0,
                decimals=1),
        _utrack("min_search_radius_px", "Search radius, smallest", "px", low=0.0, high=50.0,
                decimals=1),
        _utrack("brown_std_mult", "Radius multiplier", "x step SD", low=0.5, high=10.0,
                decimals=1),
        _utrack("frames_bridged", "Frames bridged", "frames", low=0, high=20, integer=True),
        _utrack("min_track_len", "Shortest segment kept", "localisations", low=2, high=100,
                integer=True),
    )


def _outline_options() -> tuple:
    from .segmentation.outline_model import OutlineCondition

    return (
        _NOTIFY,
        _UPDATES,
        Option("segment_model", "Model for a new condition", "Segmentation", "choice",
               OutlineCondition.model,
               "The model every condition starts with when a folder opens; you can change it "
               "on the condition's row. The pretrained phase model reads phase/, the "
               "fluorescence model sum/, and a saved model the images it was trained on.",
               read_by="segmentation.outline_model.open_outlines (model=)"),
        Option("segment_use_gpu", "Use the GPU when one is found", "Segmentation", "switch",
               True,
               "Off runs on the CPU even when a GPU is present. This is much slower; use it "
               "when the GPU is busy with something else.",
               read_by="ui.outlines.segment_page.set_gpu (use=)"),
    )


#: The analysis's Compute choices, ``(value, words)``. The VALUES are the Compute dialog's
#: own (`ui.workspace.compute.COMPUTE_CONTROLS`); `tests/test_options.py` holds them equal,
#: so a renamed value fails the suite instead of making a standard nothing can apply.
_WEIGHTING = (("step", "Step"), ("track", "Track"))
_UNCERTAINTY = (("none", "None"), ("day", "Replicates"), ("movie", "Movies"), ("cell", "Cells"), ("chunk", "Chunks"))
_ECDF = (("fast", "Fast"), ("full", "Full"))
_OPTIMISER = (("analytic", "Fast"), ("numeric", "Standard"))


def _analysis_options() -> tuple:
    return (
        _UPDATES,
        Option("sqd_weighting", "Weighting", "SQD fit (the Compute dialog)", "choice", "step",
               "Step counts every displacement once (the share of observed time, as in "
               "SMTracker 2); Track counts every track once (the share of molecules). Track "
               "gives a lower slow fraction, because long tracks are mostly slow.",
               read_by="ui.workspace.window: the Compute dialog's Weighting",
               choices=_WEIGHTING),
        Option("sqd_uncertainty", "Uncertainty", "SQD fit (the Compute dialog)", "choice",
               "day",
               "Replicates: fractions per replicate, each bubble their mean ± SD. "
               "None: one pooled fit. Movies, Cells: a movie or cell bootstrap, "
               "for single-session data only, and slow. Chunks: random chunks of each condition's "
               "movies as pseudo-replicates, for data without replicates.",
               read_by="ui.workspace.window: the Compute dialog's Uncertainty",
               choices=_UNCERTAINTY),
        Option("sqd_chunks", "Chunks per condition", "SQD fit (the Compute dialog)", "number",
               float(_chunks_default()),
               "Under Uncertainty Chunks, how many random chunks each condition's movies are dealt into (its "
               "cells when it has fewer than two movies per chunk). More chunks give more points for a test, "
               "each from less data.",
               read_by="ui.workspace.window: the Compute dialog's Chunks per condition",
               low=float(_chunks_range()[0]), high=float(_chunks_range()[1]), decimals=0),
        Option("sqd_ecdf", "ECDF", "SQD fit (the Compute dialog)", "choice", "fast",
               "Fast evaluates the fit at 2000 points along the ECDF (fractions change by "
               "under 0.001 points); Full evaluates it at every point, which is exact but slower.",
               read_by="ui.workspace.window: the Compute dialog's ECDF", choices=_ECDF),
        Option("sqd_optimiser", "Optimiser", "SQD fit (the Compute dialog)", "choice",
               "analytic",
               "Fast fits with the model's exact derivatives, about twofold faster at 3 "
               "populations; Standard estimates them numerically. Both give the same fits. "
               "The Jump distances page uses this setting too.",
               read_by="ui.workspace.window: the Compute dialog's Optimiser (and the Jump "
                       "distances page)",
               choices=_OPTIMISER),
        Option("sqd_best_tries_four", "Best (BIC) also tries 4 populations",
               "SQD fit (the Compute dialog)", "switch", False,
               "Best (BIC) picks the population count by BIC from 1 to 3. When on, it also tries 4 "
               "(about 70 s more per condition, 200 starts). SMTracker 2 fits 1-3 only.",
               read_by="ui.workspace.window: the Compute dialog's four-population box"),
        Option("cell_outline_tolerance_nm", "Outline tolerance", "Cell filter", "number", 0.0,
               "A track is kept when every point lies inside the cell outline widened by this "
               "distance; 0 is SMTracker 2's filter. Tight outlines raise the slow fraction. "
               "Applies when a project is next opened.",
               read_by="project.load_project (outline_tolerance_nm=), via the analysis doors "
                       "(ui.onboarding.scan, open_gui, app)",
               unit="nm", low=0.0, high=500.0, decimals=0),
        Option("tau_rescale_max_pct", "One frame interval for movies within", "Frame interval", "number",
               _tau_rescale_default(),
               "Movies whose frame intervals differ by less than this (largest over smallest, minus 1) are "
               "rescaled to the most common interval. Movies further apart stay separate and a pooled fit "
               "refuses them. 0 never rescales. Applies when a project is next opened.",
               read_by="project.load_project (max_tau_spread=), via the analysis doors "
                       "(ui.onboarding.scan, open_gui, app)",
               unit="%", low=0.0, high=25.0, decimals=1),
    )


#: The stages with an options screen, in the launcher's order.
TOOL_TITLES = {"prep": "Preparation & tracking options",
               "outlines": "Cell outline options",
               "analysis": "Analysis options"}

_BUILDERS = {"prep": _prep_options, "outlines": _outline_options,
             "analysis": _analysis_options}
_CACHE: dict = {}


def _tool(tool: str) -> tuple:
    if tool not in _BUILDERS:
        raise OptionError(f"no options screen called {tool!r}")
    if tool not in _CACHE:
        _CACHE[tool] = _BUILDERS[tool]()
    return _CACHE[tool]


def options_for(tool: str) -> tuple:
    """``(Option, ...)`` of one screen, in its order. Built on first use: the preparation
    list imports the tracking modules, which the analysis door never needs."""
    return _tool(tool)


def option(key: str, tool: str | None = None) -> Option:
    for name in ([tool] if tool else list(_BUILDERS)):
        for opt in _tool(name):
            if opt.key == key:
                return opt
    raise OptionError(f"no option called {key!r}")


def parse(opt: Option, raw, *, choices=None):
    """``raw`` as ``opt``'s value, or `OptionError` naming the option and what it takes.
    ``choices`` overrides the option's own (the window's list of models)."""
    if opt.kind == "switch":
        if isinstance(raw, str):
            return raw.strip().lower() in ("1", "true", "on", "yes")
        return bool(raw)
    if opt.kind == "folder":
        from pathlib import Path

        text = str(raw or "").strip()
        if text and not Path(text).is_dir():
            raise OptionError(f"{opt.label}: {text} is not a folder")
        return text
    if opt.kind == "choice":
        allowed = [v for v, _w in (choices if choices is not None else opt.choices)]
        if raw not in allowed:
            raise OptionError(f"{opt.label}: {raw!r} is not one of "
                              f"{', '.join(map(str, allowed))}")
        return raw
    try:
        number = float(str(raw).replace(",", ".")) if isinstance(raw, str) else float(raw)
    except (TypeError, ValueError) as exc:
        raise OptionError(f"{opt.label}: {raw!r} is not a number") from exc
    if number != number:
        raise OptionError(f"{opt.label}: not a number")
    if opt.integer:
        if number != int(number):
            raise OptionError(f"{opt.label}: {raw!r} is not a whole number")
        number = int(number)
    unit = f" {opt.unit}" if opt.unit else ""
    if opt.low is not None and number < opt.low:
        raise OptionError(f"{opt.label}: {number:g}{unit} is below {opt.low:g}{unit}")
    if opt.high is not None and number > opt.high:
        raise OptionError(f"{opt.label}: {number:g}{unit} is above {opt.high:g}{unit}")
    return number


def read(key: str, stored, *, tool: str | None = None, choices=None):
    """The stored value, or the default when none is stored or the stored one no longer
    parses (a record from another version, a deleted model): a bad record never half
    applies, as with the u-track settings."""
    opt = option(key, tool)
    record = stored if isinstance(stored, dict) else {}
    if key not in record:
        return opt.default
    try:
        return parse(opt, record[key], choices=choices)
    except OptionError:
        return opt.default


def values(tool: str, stored, *, choices: dict | None = None) -> dict:
    """``{key: value}`` for every option of ``tool``."""
    choices = choices or {}
    return {o.key: read(o.key, stored, tool=tool, choices=choices.get(o.key))
            for o in _tool(tool)}


def update(stored, tool: str, changes: dict, *, choices: dict | None = None) -> dict:
    """A NEW record with ``changes`` (``{key: raw}``) applied, every one validated first:
    one bad value refuses the whole save, named, so a screen never saves half. A value
    equal to its default is dropped, so the record holds only what was chosen."""
    choices = choices or {}
    record = dict(stored) if isinstance(stored, dict) else {}
    keys = {o.key: o for o in _tool(tool)}
    parsed = {}
    for key, raw in changes.items():
        if key not in keys:
            raise OptionError(f"no option called {key!r} in {TOOL_TITLES[tool]}")
        parsed[key] = parse(keys[key], raw, choices=choices.get(key))
    if any(k.startswith("utrack.") for k in parsed):
        utrack_standard({**record, **parsed})            # u-track's own refusals, named
    for key, value in parsed.items():
        if value == keys[key].default:
            record.pop(key, None)
        else:
            record[key] = value
    return record


def reset(stored, tool: str) -> dict:
    """A new record without ``tool``'s options (they return to their defaults). An option
    two screens share (the pop-up) is reset with either."""
    keys = {o.key for o in _tool(tool)}
    return {k: v for k, v in (stored or {}).items() if k not in keys}


def differences(tool: str, stored, *, choices: dict | None = None) -> list:
    """One line per option set away from its default: ``"Linking radius 8 px (default
    5 px)"``. Empty when everything is at its default."""
    choices = choices or {}
    out = []
    for opt in _tool(tool):
        value = read(opt.key, stored, tool=tool, choices=choices.get(opt.key))
        if value != opt.default:
            out.append(f"{opt.label} {_words(opt, value, choices.get(opt.key))} "
                       f"(default {_words(opt, opt.default, choices.get(opt.key))})")
    return out


def _words(opt: Option, value, choices=None) -> str:
    if opt.kind == "switch":
        return "on" if value else "off"
    if opt.kind == "folder":
        return value or "in place"
    if opt.kind == "choice":
        return dict(choices if choices is not None else opt.choices).get(value, str(value))
    unit = f" {opt.unit}" if opt.unit else ""
    return f"{value:g}{unit}"


def follows(current, old, new):
    """What a value on a page becomes when its standard moves from ``old`` to ``new``:
    the new standard if it still sat at the old one, else unchanged (it was chosen)."""
    return new if current == old else current


def compute_values(stored) -> dict:
    """The analysis standards as the Compute dialog's own values
    (`ui.workspace.compute.ComputePanel.apply_values`).

    SHARED D IS NOT A STANDARD, the General page's or a row's (checked 1 Nov 2026, when each condition row got its
    own): a row's belongs to one condition of one project, which a standard for every session cannot name, and none
    is given here -- so `apply_values` leaves both as they are (`shared_d_rows` absent keeps each row's own)."""
    got = values("analysis", stored)
    return {"weighting": got["sqd_weighting"], "uncertainty": got["sqd_uncertainty"],
            "ecdf": got["sqd_ecdf"], "gradient": got["sqd_optimiser"], "chunks": int(got["sqd_chunks"]),
            "best_max_pop": 4 if got["sqd_best_tries_four"] else 3}


def _chunks_default() -> int:
    from .workspace_view import CHUNKS_DEFAULT

    return CHUNKS_DEFAULT


def _chunks_range() -> tuple:
    from .workspace_view import CHUNKS_RANGE

    return CHUNKS_RANGE


def _tau_rescale_default() -> float:
    from .project import MAX_TAU_RESCALE

    return round(100.0 * MAX_TAU_RESCALE, 6)


def max_tau_spread(stored) -> float:
    """The largest frame-interval spread the analysis doors put on one tau
    (`project.load_project`), as a fraction: `project.MAX_TAU_RESCALE` unless the user set one."""
    return float(read("tau_rescale_max_pct", stored, tool="analysis")) / 100.0


def notify_after(stored) -> float | None:
    """The pop-up threshold in seconds the user SAVED on an Options screen (prep and
    outlines share it: one key in the one record), or None when none is saved, so the
    caller can fall back. Read from `_NOTIFY` itself: building a screen's list imports the
    tracking modules, which the outlines window never needs."""
    record = stored if isinstance(stored, dict) else {}
    if _NOTIFY.key not in record:
        return None
    try:
        return float(parse(_NOTIFY, record[_NOTIFY.key]))
    except OptionError:
        return None


def updates_at_start(stored) -> bool:
    """Whether the start asks GitHub for a newer version: on unless the user turned it off on an Options screen."""
    record = stored if isinstance(stored, dict) else {}
    try:
        return bool(parse(_UPDATES, record.get(_UPDATES.key, _UPDATES.default)))
    except OptionError:
        return True


def outline_tolerance_nm(stored) -> float:
    """The cell filter's outline tolerance the analysis doors load with
    (`project.load_project`): 0, the legacy filter, unless the user set one."""
    return float(read("cell_outline_tolerance_nm", stored, tool="analysis"))


#: The u-track settings the standard holds, in the card's order. The rest (PSF, the
#: background test, the advanced ones) stay the lab's, and per session on the card.
UTRACK_STANDARD_FIELDS = ("alpha_loc_max", "max_search_radius_px", "min_search_radius_px",
                          "brown_std_mult", "frames_bridged", "min_track_len")


def utrack_standard(stored):
    """The user's u-track standard: `utrack_run.LAB_STANDARD` with the options' values.
    Raises `OptionError` with u-track's own reason when the combination is refused (a
    smallest search radius above the largest, say)."""
    from dataclasses import replace

    from .utrack_run import LAB_STANDARD, UTrackError

    record = stored if isinstance(stored, dict) else {}
    changes = {name: read(f"utrack.{name}", record, tool="prep")
               for name in UTRACK_STANDARD_FIELDS}
    try:
        return replace(LAB_STANDARD, **changes)
    except UTrackError as exc:
        raise OptionError(f"u-track standard: {exc}") from exc
