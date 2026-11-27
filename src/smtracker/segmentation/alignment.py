"""Does a mesh drawn on brightfield sit where the fluorescence actually is?

Asked for by the experimenter: "whenever mesh is drawn in brightfield, there needs to be
a check if there is a huge shift in alignment (a few pixel is fine)".

Phase and fluorescence are recorded through different light paths and the registration
between them drifts, so an outline traced on phase can sit a few pixels off the cell as
the tracking sees it. A few pixels is tolerable. A large shift is not, and it is silent:
every downstream number still computes, the cell filter just keeps the wrong
localisations.

The measurement is how far the outlines must move to best cover the signal. Shift the
mask over a small range, score each position by the mean image intensity it encloses,
take the peak. That needs no feature detection and no assumption that phase and
fluorescence look alike, which they do not.

An empty field, a dark movie, or a mask covering most of the frame all give a flat score
surface whose peak is noise. `contrast` reports how far the best position stands above
the typical one, and the verdict says "cannot tell" below a stated floor rather than
returning a shift nobody should act on.
"""
from __future__ import annotations

import numpy as np

__all__ = ["AlignmentError", "mesh_image_offset", "TOLERANCE_PX", "MIN_CONTRAST"]


class AlignmentError(ValueError):
    """Raised when the inputs cannot be compared at all."""


#: A shift at or below this is not worth reporting: the experimenter's "a few pixel is
#: fine". A tolerance, not a correction. Nothing is moved; the check only reports.
TOLERANCE_PX = 3.0

#: How far the best position must stand above the median position, as a fraction, before
#: a shift is believed. Below it the score surface is flat and its peak means nothing.
MIN_CONTRAST = 0.02


def mesh_image_offset(labels, image, *, max_shift_px: int = 8,
                      tolerance_px: float = TOLERANCE_PX,
                      min_contrast: float = MIN_CONTRAST) -> dict:
    """How far the outlines must move to best cover the signal in ``image``.

    ``labels`` is a label mask as `masks.labels_from_outlines` writes, where any
    non-zero pixel counts as cell. ``image`` is the fluorescence frame the tracks came
    from (the ``sum`` projection), same shape.

    Returns ``dx``, ``dy`` (pixels the MASK must move), ``shift_px`` (their magnitude),
    ``contrast``, ``ok`` and a ``verdict`` sentence. Nothing is moved and nothing is
    written: this reports.
    """
    labels = np.asarray(labels)
    image = np.asarray(image, dtype=float)
    if labels.shape != image.shape:
        raise AlignmentError(
            f"mask {labels.shape} and image {image.shape} are different sizes; they must "
            f"be the same field")
    if max_shift_px < 1:
        raise AlignmentError(f"max_shift_px must be >= 1, got {max_shift_px}")
    mask = labels > 0
    if not mask.any():
        raise AlignmentError("the mask has no cells, so there is nothing to align")
    if mask.mean() > 0.5:
        raise AlignmentError(
            f"the mask covers {mask.mean():.0%} of the field; with almost everything "
            f"inside a cell the score cannot tell one position from another.")

    height, width = labels.shape
    span = range(-int(max_shift_px), int(max_shift_px) + 1)
    scores = np.full((len(span), len(span)), np.nan)
    for i, dy in enumerate(span):
        for j, dx in enumerate(span):
            # Slicing rather than np.roll: a roll wraps cells round the edge and
            # scores them against the opposite side of the field, inventing a peak.
            ys = slice(max(dy, 0), height + min(dy, 0))
            xs = slice(max(dx, 0), width + min(dx, 0))
            yt = slice(max(-dy, 0), height + min(-dy, 0))
            xt = slice(max(-dx, 0), width + min(-dx, 0))
            shifted = mask[yt, xt]
            if not shifted.any():
                continue
            scores[i, j] = image[ys, xs][shifted].mean()

    if np.all(np.isnan(scores)):
        raise AlignmentError("no shift left any cell inside the field")
    best = np.unravel_index(np.nanargmax(scores), scores.shape)
    dy, dx = span[best[0]], span[best[1]]
    peak = float(scores[best])
    typical = float(np.nanmedian(scores))
    contrast = (peak - typical) / abs(typical) if typical else 0.0
    shift = float(np.hypot(dx, dy))

    if contrast < min_contrast:
        return {"dx": 0, "dy": 0, "shift_px": 0.0, "contrast": contrast, "ok": None,
                "verdict": (f"cannot tell: the best position is only "
                            f"{contrast * 100:.1f} % better than a typical one, so this "
                            f"field has too little signal to align to.")}
    if shift <= tolerance_px:
        return {"dx": dx, "dy": dy, "shift_px": shift, "contrast": contrast, "ok": True,
                "verdict": (f"aligned: best offset ({dx:+d}, {dy:+d}) px, within the "
                            f"{tolerance_px:g} px tolerance")}
    return {"dx": dx, "dy": dy, "shift_px": shift, "contrast": contrast, "ok": False,
            "verdict": (f"shifted by ({dx:+d}, {dy:+d}) px = {shift:.1f} px, beyond the "
                        f"{tolerance_px:g} px tolerance. Outlines drawn on phase do not "
                        f"sit where the fluorescence is, so the cell filter silently keeps the "
                        f"wrong localisations.")}
