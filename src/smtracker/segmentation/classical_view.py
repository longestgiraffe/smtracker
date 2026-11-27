"""The CLASSICAL SEGMENTER's settings page, headless: the sliders, a preview drawn exactly as a run draws (the label
image through the capsule fit, `segment.segment_field`'s own steps), and the settings as the plain dict that is
remembered (ui.json key `SETTINGS_KEY`) and travels to a run (`OutlineCondition.classical` -> `segment.config_for`).

Asked for (15 Oct 2026, of the classical segmenter): "can we try the manual version? ... i mean with the settings".
Plain payloads, so `ui.outlines.classical_page` paints with no numpy.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace

import numpy as np

from .classical import ClassicalParams, explain, segment

__all__ = ["SETTINGS_KEY", "Setting", "SETTINGS", "params_from", "to_saved", "preview", "words", "slider_value",
           "value_of", "shown"]

#: ui.json key: the classical segmenter's settings, used by every classical run (Segment and Batch).
SETTINGS_KEY = "classical_segmenter"


@dataclass(frozen=True)
class Setting:
    key: str
    label: str
    low: float
    high: float
    step: float
    unit: str
    tip: str
    #: The value that means "not applied", shown as ``off_word`` (the maximum length's 0: no maximum).
    off_at: float | None = None
    off_word: str = ""


#: The sliders, in the order they are shown; each a `ClassicalParams` field.
SETTINGS = (
    Setting("cell_width_px", "Cell width", 4.0, 20.0, 0.5, "px",
            "The width of a cell in this image. Candidates much thinner or wider are dropped, and touching cells "
            "are split relative to it. B. subtilis is about 9 px wide at 106 nm per pixel."),
    Setting("sensitivity", "Sensitivity", 0.3, 1.5, 0.05, "x",
            "The threshold, as a multiple of Otsu's automatic level. Lower finds dimmer cells and more "
            "background; higher finds only the brightest cells."),
    Setting("split_depth", "Split touching cells", 0.05, 0.8, 0.05, "",
            "How deep the waist between two touching cells must be before they are split. Lower splits more "
            "readily; higher keeps chains whole."),
    Setting("min_length_px", "Minimum length", 4.0, 40.0, 1.0, "px", "Shorter candidates are dropped."),
    Setting("max_length_px", "Maximum length", 0.0, 150.0, 5.0, "px",
            "Longer candidates, such as a chain of cells left whole, are dropped. Lower 'Split touching cells' "
            "to split them instead. All the way left means no maximum. B. subtilis is about 20-50 px long at "
            "106 nm per pixel.", off_at=0.0, off_word="none"),
    Setting("grow", "Grow to the edge", 0.0, 0.5, 0.05, "x width",
            "Grows each cell outward by this fraction of its width, so the outline reaches the cell's edge. "
            "The threshold alone finds only the cell body."),
    Setting("smooth_px", "Smoothing", 0.0, 3.0, 0.25, "px",
            "Blur applied before the threshold. More smoothing joins patchy fluorescence into one body; too much "
            "joins neighbouring cells."),
)
_BY_KEY = {s.key: s for s in SETTINGS}


def slider_value(setting: Setting, value: float) -> int:
    """A setting's value as the integer a slider holds (steps above the low end)."""
    return int(round((float(value) - setting.low) / setting.step))


def value_of(setting: Setting, position: int) -> float:
    """The value at a slider position, clamped to the setting's range."""
    return float(min(setting.high, max(setting.low, setting.low + int(position) * setting.step)))


def params_from(saved: dict | None, channel: str = "sum") -> ClassicalParams:
    """`ClassicalParams` from the remembered dict; a missing or out-of-range entry keeps the default."""
    params = ClassicalParams(channel=channel)
    names = {f.name for f in fields(ClassicalParams)}
    changes = {}
    for key, value in (saved or {}).items():
        if key not in names or key == "channel":
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        setting = _BY_KEY.get(key)
        if setting is not None and not setting.low <= value <= setting.high:
            continue
        changes[key] = value
    return replace(params, **changes)


def to_saved(params: ClassicalParams) -> dict:
    """The settings as the plain dict that is remembered: the slider fields only."""
    values = asdict(params)
    return {s.key: float(values[s.key]) for s in SETTINGS}


def shown(setting: Setting, value: float) -> str:
    """A setting's value as the page shows it: with its unit, or its off word."""
    if setting.off_at is not None and float(value) == setting.off_at:
        return setting.off_word
    return f"{float(value):g}{(' ' + setting.unit) if setting.unit else ''}"


def words(params: ClassicalParams) -> str:
    """The settings in one line, for the run log and the page."""
    return ", ".join(f"{s.label.lower()} {shown(s, getattr(params, s.key))}" for s in SETTINGS)


def _grey(image) -> tuple:
    img = np.asarray(image, dtype=float)
    while img.ndim > 2:
        img = img.mean(axis=0)
    lo, hi = np.percentile(img, [0.5, 99.5])
    grey = np.clip((img - lo) / ((hi - lo) or 1.0) * 255.0, 0, 255).astype(np.uint8)
    return int(grey.shape[1]), int(grey.shape[0]), np.ascontiguousarray(grey).tobytes()


def preview(image, params: ClassicalParams) -> dict:
    """What a run would write for ``image`` with ``params``: the grey image (bytes), every cell's outline as the run
    draws it (``[[x, y], ...]``, 0-based pixels), and every dropped candidate with its reason."""
    from .masks import MaskConfig, build_capsule

    width, height, pixels = _grey(image)
    labels = segment(image, params)
    outlines = []
    for lab in [v for v in np.unique(labels) if v]:
        fit = build_capsule(labels == lab, MaskConfig(cell_width_px=float(params.cell_width_px)))
        if fit.ok:
            outlines.append([[float(x), float(y)] for x, y in np.asarray(fit.outline)])
    dropped = [{"x": float(col), "y": float(row), "why": why}
               for (row, col), kept, why in explain(image, params) if not kept]
    n = len(outlines)
    return {"width": width, "height": height, "pixels": pixels, "outlines": outlines, "dropped": dropped,
            "count": n, "caption": (f"{n} cell{'s' if n != 1 else ''} with these settings"
                                    + (f"; {len(dropped)} candidate(s) dropped" if dropped else ""))}
