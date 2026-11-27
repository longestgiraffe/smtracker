"""CLASSICAL cell segmentation with smart settings: no network, no GPU, every step explainable. Headless.

Asked for (14 Oct 2026): "I'm not happy with the state of omnipose ... is there another way to segment cells, user
friendly, not like microbej, where we can have smart settings to detect cells in phase or sum" -- and "output needs
to be that meshes mat readable in smtracker". So this returns a LABEL IMAGE, and `segment.segment_field` turns it
into capsule meshes and `cell_meshes/<movie>.mat` exactly as it does a model's masks.

THE STEPS, for one image (`segment`):
1. Normalise (0.5-99.5 %), and turn cells BRIGHT: the sum image as it is (cells glow), phase inverted (cells are
   dark on a grey field with a bright halo).
2. Flatten the background: subtract a wide Gaussian (``background_px``), so a vignetted field or an uneven pad
   thresholds evenly.
3. Smooth lightly (``smooth_px``) and threshold at Otsu's level times ``sensitivity`` (below 1 takes dimmer cells).
4. Clean: fill holes, open with a disk of a quarter cell width (strips the phase halo's thin ring), drop specks.
5. SPLIT touching cells: a distance transform's ridge seeds each cell (markers at least ``split_depth`` x the half
   width above the saddle between two cells), a watershed on the image's own intensity grows them back, so the
   dark septum between two cells (sum) or the bright one (phase) is where they part.
6. Keep what is shaped like a cell: width within ``width_tolerance`` of ``cell_width_px``, length at least
   ``min_length_px`` and, when ``max_length_px`` is set, at most that (a chain left whole is dropped, not drawn as one
   cell), solidity at least ``min_solidity`` (a clump of two crossing cells is not solid), and not lying in a bright
   band along the image's edge (``edge_band``).

A BRIGHT BAND ALONG THE EDGE (3 Nov 2026, tester B8). One user's protein F series001 sum image has its top rows at about
63,000 counts against about 12,000 mid-image: Otsu's level fell between the band and the rest, the band was drawn as
two "cells" and the real cell was never a candidate. With ``edge_band`` (the default) a region touching the edge that
runs along at least half of it and is shallow (`_band`) is a band: every candidate overlapping it is dropped and said,
and where it was ALL the first threshold found, the level is measured again without it (and without the dark trough
its flattening leaves). A CELL CUT BY THE EDGE IS KEPT, as before. Scored against the outlines of copies of one user's
two strains' fields (49 sum images, 323 outlined cells, a candidate matched at IoU 0.2): dropping every edge-touching
candidate removed 21 outlined cells with 22 false ones (58 matched / 28 unmatched -> 37 / 6), re-thresholding every
banded field merged neighbours into chains (-> 43 / 10); the band rule as built -> see `docs/claude/segmentation.md`.

SMART SETTINGS (`calibrate`): from a few points the user clicks INSIDE cells, the cell width is measured (the region
grown from each click at the local threshold, its minor axis) and the sensitivity is set so the clicked cells clear
the threshold; the rest stay at their defaults. `explain` says why each candidate was kept or dropped, in words.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

__all__ = ["ClassicalParams", "segment", "calibrate", "explain", "CHANNELS"]

CHANNELS = ("sum", "phase")


@dataclass(frozen=True)
class ClassicalParams:
    """The settings, in pixels. `calibrate` sets ``cell_width_px`` and ``sensitivity`` from clicked cells."""

    channel: str = "sum"
    #: The cells' width in this image: ~0.9 um B. subtilis at 106 nm plus the blur (9 px measured best on nine
    #: outlined datasets, 14 Oct 2026).
    cell_width_px: float = 9.0
    #: Multiplies Otsu's threshold: below 1 takes dimmer cells, above 1 only the brightest.
    sensitivity: float = 1.0
    #: Background subtracted: a Gaussian this wide (much wider than a cell).
    background_px: float = 25.0
    #: Smoothing before the threshold (1 px measured best; 2 px merged neighbours on the nine datasets).
    smooth_px: float = 1.0
    #: Each kept cell is grown by this fraction of the width: an outline is drawn on the cell's EDGE (its halo),
    #: the threshold finds its body.
    grow: float = 0.2
    #: Phase only: a candidate must be darker than the field by this many noise SDs, AND ringed by a bright halo.
    phase_contrast_sd: float = 3.0
    #: Where touching cells part: "intensity" (the watershed on the image: the dim septum between two bright cells
    #: -- what the first version did, and what looked right on the sum images) or "shape" (where the outline
    #: narrows; immune to foci inside one cell).
    split_on: str = "intensity"
    #: How far above the saddle between two cells a seed must stand, as a fraction of the half width: higher
    #: splits less (chains stay whole), lower splits more.
    split_depth: float = 0.3
    width_tolerance: float = 0.45
    min_length_px: float = 12.0
    #: Longer candidates are dropped (asked 15 Oct 2026: "there should be a maximum cell length"); 0 = no maximum,
    #: the default, so the measured defaults are unchanged.
    max_length_px: float = 0.0
    min_solidity: float = 0.75
    #: Look for a bright band along the image's edge and keep it out of the threshold and the cells (see the module
    #: docstring). Off is the segmenter before 3 Nov 2026.
    edge_band: bool = True

    def __post_init__(self):
        if self.channel not in CHANNELS:
            raise ValueError(f"channel must be one of {CHANNELS}, got {self.channel!r}")
        if not self.cell_width_px > 2:
            raise ValueError("cell_width_px must be > 2")


def _bright_cells(image, params: ClassicalParams) -> np.ndarray:
    """Steps 1-2: normalised, cells bright, background flattened."""
    from scipy.ndimage import gaussian_filter

    img = np.asarray(image, dtype=float)
    while img.ndim > 2:
        img = img.mean(axis=0)
    lo, hi = np.percentile(img, [0.5, 99.5])
    img = np.clip((img - lo) / ((hi - lo) or 1.0), 0.0, 1.0)
    if params.channel == "phase":
        img = 1.0 - img
    flat = img - gaussian_filter(img, params.background_px)
    return gaussian_filter(flat, params.smooth_px) if params.smooth_px else flat


def _level(bright: np.ndarray, params: ClassicalParams) -> float:
    """The threshold: Otsu's on the sum image (cells glow over a dark field); on PHASE the strongly dark tail only
    -- the field's own noise (a textured pad) spans the middle, and Otsu's level cuts into it, so a phase cell must
    stand ``phase_contrast_sd`` robust SDs above the field's median."""
    from skimage.filters import threshold_otsu

    if params.channel == "phase":
        med = float(np.median(bright))
        mad = float(np.median(np.abs(bright - med))) * 1.4826 or 1e-6
        return (med + params.phase_contrast_sd * mad) * float(params.sensitivity)
    return threshold_otsu(bright) * float(params.sensitivity)


#: A BAND, not a cell: a foreground region touching the image's edge that runs along at least this share of the
#: image's width or height and is at most `BAND_DEPTH` of it deep (the bright rows at the top of one user's two strains'
#: sum images: 160 x 11 and 176 x 18 px). A cell cut by the edge is short along it and is never one.
BAND_SPAN = 0.5
BAND_DEPTH = 0.25


def _edge_labels(labels: np.ndarray) -> set:
    """The labels of a label image that touch its outermost rows or columns."""
    rim = np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
    return {int(v) for v in np.unique(rim) if v}


def _band(regions: np.ndarray) -> np.ndarray | None:
    """The mask of the regions of a label image that are a bright BAND along its edge (`BAND_SPAN`, `BAND_DEPTH`),
    or None when there is none."""
    h, w = regions.shape
    band = np.zeros(regions.shape, bool)
    for lab in _edge_labels(regions):
        rows, cols = np.nonzero(regions == lab)
        tall, wide = rows.max() - rows.min() + 1, cols.max() - cols.min() + 1
        if (wide >= BAND_SPAN * w and tall <= BAND_DEPTH * h) or (tall >= BAND_SPAN * h and wide <= BAND_DEPTH * w):
            band |= regions == lab
    return band if band.any() else None


def _threshold(bright: np.ndarray, level: float, params: ClassicalParams) -> np.ndarray:
    from scipy.ndimage import binary_fill_holes, binary_opening, label
    from skimage.morphology import disk

    fg = binary_fill_holes(bright > level)
    radius = max(1, int(round(params.cell_width_px / 4.0)))
    fg = binary_opening(fg, structure=disk(radius))
    regions, _n = label(fg)
    sizes = np.bincount(regions.ravel())
    min_area = max(10, int(0.5 * params.cell_width_px * params.min_length_px))
    keep = sizes >= min_area
    keep[0] = False
    return keep[regions]


def _band_and_foreground(bright: np.ndarray, params: ClassicalParams) -> tuple:
    """Steps 3-4: ``(foreground, band)``, thresholded and cleaned; ``band`` is a bright band along the image's edge
    (`_band`) or None. Where the band is ALL the first threshold found (it set the level above every cell), the
    level is measured again on the rest of the image: the band, a cell width around it and the dark trough its
    flattening leaves left out. Where cells were found beside it, the level stands, as it always did."""
    from scipy.ndimage import binary_dilation, label

    fg = _threshold(bright, _level(bright, params), params)
    if not params.edge_band or not fg.any():
        return fg, None
    regions, _n = label(fg)
    band = _band(regions)
    if band is None or (fg & ~band).any():
        return fg, band
    margin = binary_dilation(band, iterations=max(1, int(round(params.cell_width_px))))
    rest = bright[~margin]
    # THE TROUGH LEFT OUT TOO: flattening subtracts a wide blur of the band, which leaves the field beside it dark
    # (to -0.16 against a field at 0 on protein F series001, for some 40 rows), and Otsu's level on the rest would part
    # that trough from everything else instead of the cells from the field
    med = float(np.median(rest)) if rest.size else 0.0
    mad = float(np.median(np.abs(rest - med))) * 1.4826 if rest.size else 0.0
    rest = rest[rest >= med - 3.0 * mad]
    if rest.size < 100:
        return fg, band
    return _threshold(bright, _level(rest, params), params), band


def _foreground(bright: np.ndarray, params: ClassicalParams) -> np.ndarray:
    """Steps 3-4: the cleaned foreground (`_band_and_foreground`)."""
    return _band_and_foreground(bright, params)[0]


def _split(fg: np.ndarray, bright: np.ndarray, params: ClassicalParams) -> np.ndarray:
    """Step 5: one label per cell, parted where the SHAPE narrows (the septum). On the distance transform, not the
    intensity: a cell's patchy fluorescence (foci) made ridges inside one cell and split it in pieces."""
    from scipy.ndimage import distance_transform_edt, gaussian_filter, label
    from skimage.morphology import h_maxima
    from skimage.segmentation import watershed

    dist = gaussian_filter(distance_transform_edt(fg), 1.0)
    depth = max(0.5, float(params.split_depth) * params.cell_width_px / 2.0)
    markers, _n = label(h_maxima(dist, depth))
    return watershed(-bright if params.split_on == "intensity" else -dist, markers, mask=fg)


def _measure(labels: np.ndarray):
    from skimage.measure import regionprops

    return regionprops(labels)


def _verdicts(labels: np.ndarray, params: ClassicalParams, band: np.ndarray | None = None) -> list:
    """Step 6, per candidate: ``(label, kept, reason)``. ``band``: a bright band along the edge (`_band_and_foreground`);
    a candidate overlapping it is dropped."""
    out = []
    w, tol = params.cell_width_px, params.width_tolerance
    in_band = set() if band is None else {int(v) for v in np.unique(labels[band]) if v}
    for r in _measure(labels):
        width, length = r.axis_minor_length, r.axis_major_length
        if r.label in in_band:
            out.append((r.label, False, "a bright band along the image's edge, not a cell"))
        elif not (1 - tol) * w <= width <= (1 + tol) * w:
            out.append((r.label, False, f"{width:.1f} px wide; cells here are {w:.1f}"))
        elif length < params.min_length_px:
            out.append((r.label, False, f"{length:.1f} px long, under {params.min_length_px:.0f}"))
        elif params.max_length_px and length > params.max_length_px:
            out.append((r.label, False, f"{length:.1f} px long, over {params.max_length_px:.0f}"))
        elif r.solidity < params.min_solidity:
            out.append((r.label, False, f"solidity {r.solidity:.2f}: likely two cells crossing"))
        else:
            out.append((r.label, True, f"{width:.1f} x {length:.1f} px"))
    return out


def segment(image, params: ClassicalParams | None = None) -> np.ndarray:
    """An integer label image of the cells in ``image`` (0 = background)."""
    from skimage.segmentation import expand_labels

    params = params or ClassicalParams()
    bright = _bright_cells(image, params)
    fg, band = _band_and_foreground(bright, params)
    labels = _split(fg, bright, params)
    keep = {lab for lab, kept, _why in _verdicts(labels, params, band) if kept}
    out = np.zeros_like(labels, dtype=np.int32)
    for new, lab in enumerate(sorted(keep), start=1):
        out[labels == lab] = new
    grow = int(round(float(params.grow) * params.cell_width_px))
    return expand_labels(out, grow) if grow > 0 else out


def explain(image, params: ClassicalParams | None = None) -> list:
    """Why each candidate was kept or dropped: ``[(centroid (row, col), kept, reason)]``."""
    params = params or ClassicalParams()
    bright = _bright_cells(image, params)
    fg, band = _band_and_foreground(bright, params)
    labels = _split(fg, bright, params)
    where = {r.label: tuple(round(c, 1) for c in r.centroid) for r in _measure(labels)}
    return [(where[lab], kept, why) for lab, kept, why in _verdicts(labels, params, band)]


def calibrate(image, clicks, params: ClassicalParams | None = None) -> ClassicalParams:
    """Settings from a few points clicked INSIDE cells (``[(row, col), ...]``): the cells' width, measured on the
    region each click grows into at the local threshold, and a sensitivity that lets every clicked cell clear the
    threshold. Raises if no click lands on something cell-like."""
    from scipy.ndimage import distance_transform_edt, label
    from skimage.filters import threshold_otsu

    params = params or ClassicalParams()
    bright = _bright_cells(image, params)
    otsu = threshold_otsu(bright)
    widths, levels = [], []
    for row, col in clicks:
        r, c = int(round(row)), int(round(col))
        if not (0 <= r < bright.shape[0] and 0 <= c < bright.shape[1]):
            continue
        peak = float(bright[max(0, r - 1):r + 2, max(0, c - 1):c + 2].max())
        if peak <= 0:
            continue
        # the cell is what lies above half way between the background (0 after flattening) and this cell's peak;
        # its WIDTH is read AT the click (twice the distance to that region's edge, the most within 2 px of it),
        # never from the whole region, which on a hazy image runs on into its neighbours (MM sum images read
        # 22-27 px "wide" that way, 14 Oct 2026)
        regions, _n = label(bright > 0.5 * peak)
        here = regions[r, c]
        if not here:
            continue
        dist = distance_transform_edt(regions == here)
        local = dist[max(0, r - 2):r + 3, max(0, c - 2):c + 3].max()
        if local < 1.5:
            continue
        widths.append(2.0 * float(local))
        levels.append(0.5 * peak)
    if not widths:
        raise ValueError("no click landed inside a cell-like region: click the middle of two or three cells")
    width = float(np.median(widths))
    # every clicked cell above the threshold, with a margin; never above Otsu's own level
    sensitivity = float(min(1.0, 0.9 * min(levels) / otsu)) if otsu > 0 else params.sensitivity
    return replace(params, cell_width_px=width, sensitivity=max(0.2, sensitivity),
                   min_length_px=max(params.min_length_px, 1.2 * width))
